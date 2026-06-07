/**
 * openclaw_bridge.mjs — OpenClaw Integration (sarvam-only, production)
 *
 * Reality check from live testing:
 *   ✅ sarvam/sarvam-105b  — the ONLY model configured with API key in this OpenClaw instance
 *   ❌ cerebras            — no API key configured
 *   ❌ chutes/deepseek     — unknown model error
 *   ❌ byteplus/kimi       — no API key configured
 *
 * So: ALL inference goes through sarvam. We handle sarvam's prompt-length
 * sensitivity by chunking prompts appropriately per task type.
 *
 * What we DO use from OpenClaw (all confirmed working):
 *   openclaw infer model run     — sarvam-105b inference (120k context)
 *   openclaw infer web search    — DuckDuckGo paper discovery
 *   openclaw infer web fetch     — URL content fetch (native fallback)
 *   openclaw infer embedding create — local embeddinggemma-300m
 *   openclaw memory search       — semantic workspace memory search
 *   openclaw memory index        — reindex after writing memory files
 *   openclaw sessions export-trajectory — run trace export
 */

import { spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { acquireSlot } from "./rate_limiter.mjs";

// ─── Core helper ─────────────────────────────────────────────────────────────

/** Strip null bytes and control characters from a string for safe use in spawnSync args. */
function safeArg(s) {
  if (typeof s !== "string") return s;
  return s
    .replace(/\x00/g, " ")
    .replace(/[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]/g, " ")
    .replace(/\uFFFD/g, "?");
}

function runClaw(args, { timeoutMs = 60000, cwd = process.cwd() } = {}) {
  // Sanitize all string args to prevent "null bytes" crash in spawnSync
  const safeArgs = args.map(safeArg);
  const result = spawnSync("openclaw", safeArgs, {
    encoding: "utf8",
    timeout: timeoutMs,
    cwd,
    env: { ...process.env },
    maxBuffer: 32 * 1024 * 1024,
  });
  if (result.error) throw new Error(`openclaw spawn error: ${result.error.message}`);
  return { stdout: result.stdout || "", stderr: result.stderr || "", code: result.status ?? -1 };
}

function parseJson(raw, fallback = null) {
  if (!raw) return fallback;
  const cleaned = raw.trim()
    .replace(/^```json\s*/i, "").replace(/^```\s*/i, "").replace(/```\s*$/, "").trim();
  try { return JSON.parse(cleaned); } catch {}
  // Extract first JSON structure
  const arrMatch = cleaned.match(/\[[\s\S]*\]/);
  const objMatch = cleaned.match(/\{[\s\S]*\}/);
  try { if (arrMatch) return JSON.parse(arrMatch[0]); } catch {}
  try { if (objMatch) return JSON.parse(objMatch[0]); } catch {}
  return fallback;
}

function extractText(parsed, rawStdout) {
  if (!parsed) return rawStdout?.trim() || null;
  return parsed.outputs?.[0]?.text
    || parsed.text
    || parsed.content
    || rawStdout?.trim()
    || null;
}

// ─── Model Inference (sarvam-105b, the only configured model) ─────────────────

// Sarvam-105b: rate limit = ~20 requests/min (3s between calls minimum)
// When sarvam returns empty: it's rate-limited. Wait and retry.
// Sarvam-105b: 117k/125k context. Large INPUT is reliable (tested 29k chars OK);
// only very long OUTPUT is flaky, so prompts ask for compact JSON. We pin the
// best available model explicitly and allow large prompts to use the context.
const SARVAM_MODEL = "sarvam/sarvam-105b";
const SARVAM_SAFE_LIMIT = 48000;
const SARVAM_MIN_INTERVAL_MS = 3000;  // Minimum 3s between calls
const SARVAM_RETRY_DELAYS = [3000, 5000, 8000, 12000];  // 4 retries with increasing backoff — sarvam must not fail

let lastCallTime = 0;

async function waitForRateLimit() {
  const now = Date.now();
  const elapsed = now - lastCallTime;
  if (elapsed < SARVAM_MIN_INTERVAL_MS) {
    await new Promise(r => setTimeout(r, SARVAM_MIN_INTERVAL_MS - elapsed));
  }
  lastCallTime = Date.now();
}

/**
 * Run a prompt through sarvam-105b.
 * Automatically retries with truncated prompt if response is empty.
 *
 * @param {string} prompt - The full prompt
 * @param {object} opts
 *   timeoutMs: number (default 90000)
 *   maxPromptLen: number (default 6000)
 */
export async function inferModel(prompt, {
  timeoutMs = 90000,
  maxPromptLen = SARVAM_SAFE_LIMIT,
  tier,
  cascade,
  thinking,
} = {}) {
  // Strip null bytes and other control characters that cause spawnSync to fail.
  // PDF text often contains binary garbage including \x00.
  let p = String(prompt || "")
    .replace(/\x00/g, " ")          // null bytes — the primary offender
    .replace(/[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]/g, " ")  // other control chars
    .replace(/\uFFFD/g, "?");       // UTF-8 replacement chars from bad PDF decode

  if (p.length > maxPromptLen) {
    console.error(`  [OC] Prompt ${p.length} chars → truncating to ${maxPromptLen}`);
    p = p.slice(0, maxPromptLen) + "\n\n[Truncated. Give best-effort JSON response.]"
  }

  const args = ["infer", "model", "run", "--model", SARVAM_MODEL, "--prompt", p, "--json"];

  // Retry loop with rate-limit backoff
  const attempts = [0, ...SARVAM_RETRY_DELAYS];  // 0ms first attempt, then backoffs
  for (let i = 0; i < attempts.length; i++) {
    if (attempts[i] > 0) {
      console.error(`  [OC] Waiting ${attempts[i]/1000}s before retry ${i}/${attempts.length - 1}...`);
      await new Promise(r => setTimeout(r, attempts[i]));
    }

    // Respect the GLOBAL cross-process rate limit (40/min per account; we target
    // ~36/min). Retries count as calls because they hit the API too.
    await acquireSlot();

    try {
      const result = runClaw(args, { timeoutMs });

      if (result.code !== 0) {
        const errMsg = result.stderr?.slice(0, 150) || "unknown error";
        console.error(`  [OC/sarvam] exit ${result.code}: ${errMsg}`);
        // Always retry — sarvam failures are often transient (rate limit, timeout, empty)
        continue;
      }

      const parsed = parseJson(result.stdout, null);
      const text = extractText(parsed, result.stdout);

      if (!text || text.length < 2) {
        console.error(`  [OC/sarvam] Empty response (attempt ${i + 1}/${attempts.length})`);
        continue;  // Retry with backoff
      }

      return text;
    } catch (err) {
      console.error(`  [OC/sarvam] Exception: ${err.message}`);
      if (i === attempts.length - 1) return null;
    }
  }

  console.error(`  [OC/sarvam] All ${attempts.length} attempts failed.`);
  return null;
}


// Compatibility shims (MODEL_TIERS referenced by llm_client)
export const MODEL_TIERS = {
  FAST:   null,   // all use sarvam default routing
  MEDIUM: null,
  DEEP:   null,
  BACKUP: null,
};

// ─── Local Embeddings ─────────────────────────────────────────────────────────

/**
 * Create a local embedding using embeddinggemma-300m (no API key needed).
 * Returns float[] or null.
 */
export async function createEmbedding(text) {
  const clean = text.trim().slice(0, 8000);
  try {
    const result = runClaw([
      "infer", "embedding", "create",
      "--text", clean,
      "--json",
    ], { timeoutMs: 15000 });

    if (result.code !== 0) return null;
    const parsed = parseJson(result.stdout, null);
    return parsed?.outputs?.[0]?.embedding || parsed?.embedding || null;
  } catch {
    return null;
  }
}

/**
 * Cosine similarity between two vectors.
 */
export function cosineSimilarity(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom > 0 ? dot / denom : 0;
}

/**
 * Semantic search using local embeddings.
 * Returns top-K matches sorted by cosine similarity.
 */
export async function semanticSearch(query, corpus, topK = 5) {
  const queryEmb = await createEmbedding(query);
  if (!queryEmb) return [];

  const results = [];
  for (const item of corpus) {
    const emb = await createEmbedding(item.text);
    if (emb) results.push({ id: item.id, similarity: cosineSimilarity(queryEmb, emb) });
  }

  return results.sort((a, b) => b.similarity - a.similarity).slice(0, topK);
}

// ─── Web Search ──────────────────────────────────────────────────────────────

export async function webSearch(query, { limit = 10, provider } = {}) {
  const args = ["infer", "web", "search", "--query", query, "--limit", String(limit), "--json"];
  if (provider) args.push("--provider", provider);

  try {
    const result = runClaw(args, { timeoutMs: 20000 });
    if (result.code !== 0) return [];
    const parsed = parseJson(result.stdout, null);
    if (!parsed) return [];

    function strip(t) {
      return (t || "")
        .replace(/<<<EXTERNAL_UNTRUSTED_CONTENT[^>]*>>>/g, "")
        .replace(/<<<END_EXTERNAL_UNTRUSTED_CONTENT[^>]*>>>/g, "")
        .trim();
    }

    let items = parsed.outputs?.[0]?.result?.results || (Array.isArray(parsed) ? parsed : parsed.results) || [];
    return items.map(r => ({
      title: strip(r.title || r.name || ""),
      url: r.url || r.link || "",
      snippet: strip(r.snippet || r.description || r.body || ""),
    })).filter(r => r.url);
  } catch {
    return [];
  }
}

// ─── Web Fetch ───────────────────────────────────────────────────────────────

export async function webFetch(url, { format = "text" } = {}) {
  // Try openclaw fetch first, fall back to native
  try {
    const result = runClaw(
      ["infer", "web", "fetch", "--url", url, "--format", format, "--json"],
      { timeoutMs: 30000 }
    );
    if (result.code === 0) {
      const parsed = parseJson(result.stdout, null);
      const content = parsed?.outputs?.[0]?.result?.text || parsed?.text || parsed?.content;
      if (content && content.length > 50) return content;
    }
  } catch {}
  return await nativeFetch(url);
}

async function nativeFetch(url) {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 20000);
    const res = await fetch(url, {
      signal: controller.signal,
      headers: { "User-Agent": "OpenResearchOS/2.0 (research agent)" },
    });
    clearTimeout(timeout);
    if (!res.ok) return null;
    const text = await res.text();
    return text
      .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, "")
      .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, "")
      .replace(/<[^>]+>/g, " ")
      .replace(/\s{3,}/g, "\n\n")
      .slice(0, 8000);
  } catch {
    return null;
  }
}

// ─── Memory ──────────────────────────────────────────────────────────────────

export async function memorySearch(query, { maxResults = 20 } = {}) {
  try {
    const result = runClaw(
      ["memory", "search", "--query", query, "--max-results", String(maxResults), "--json"],
      { timeoutMs: 15000 }
    );
    if (result.code !== 0) return [];
    const parsed = parseJson(result.stdout, null);
    if (!parsed) return [];
    const items = Array.isArray(parsed) ? parsed : (parsed.results || []);
    return items.map(r => ({
      file: r.file || r.path || "",
      score: r.score || 0,
      excerpt: r.excerpt || r.content || r.text || "",
    }));
  } catch {
    return [];
  }
}

export async function memoryIndex() {
  try {
    runClaw(["memory", "index", "--force"], { timeoutMs: 20000 });
  } catch {}
}

// ─── Trajectory Export ────────────────────────────────────────────────────────

export async function exportTrajectory(sessionKey, outputDir) {
  try {
    const result = runClaw(
      ["sessions", "export-trajectory", "--session-key", sessionKey, "--output", outputDir, "--json"],
      { timeoutMs: 60000 }
    );
    if (result.code !== 0) return null;
    return parseJson(result.stdout, { output_dir: outputDir });
  } catch {
    return null;
  }
}

// ─── Agent Turns ─────────────────────────────────────────────────────────────

export async function agentTurn(message, { sessionKey, thinking, model, timeoutMs = 300000 } = {}) {
  const args = ["agent", "--message", message, "--json"];
  if (sessionKey) args.push("--session-key", sessionKey);
  if (thinking) args.push("--thinking", thinking);
  if (model) args.push("--model", model);

  try {
    const result = runClaw(args, { timeoutMs });
    if (result.code !== 0) return null;
    return parseJson(result.stdout, { reply: result.stdout });
  } catch {
    return null;
  }
}

// ─── Health Check ─────────────────────────────────────────────────────────────

export async function healthCheck() {
  try {
    const result = runClaw(["health"], { timeoutMs: 10000 });
    return result.code === 0;
  } catch {
    return false;
  }
}

export async function webSearchHealthCheck() {
  try {
    const results = await webSearch("long context transformer attention", { limit: 1 });
    return Array.isArray(results) && results.length > 0;
  } catch {
    return false;
  }
}

// inferWithThinking — sarvam doesn't support thinking mode, just use inferModel
export async function inferWithThinking(prompt, opts = {}) {
  return inferModel(prompt, opts);
}

export async function probeModels() {
  const text = await inferModel('Reply: {"ok":true}', { timeoutMs: 15000 });
  return { fast: !!text, medium: !!text, deep: !!text };
}
