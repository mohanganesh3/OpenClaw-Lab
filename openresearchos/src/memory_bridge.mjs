/**
 * Memory Bridge for OpenResearchOS.
 *
 * Persists and loads durable lessons, failed ideas, and experiment results
 * to/from the local OpenClaw memory workspace to allow continuous improvement
 * across different research runs.
 */

import { mkdir, writeFile, readFile, readdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";

// Store memory in the home ~/.openclaw/workspace/memory or fallback to project .openclaw/memory
const HOME_CLAW_DIR = join(homedir(), ".openclaw", "workspace", "memory");
const PROJECT_CLAW_DIR = join(process.cwd(), ".openclaw", "memory");

function getMemoryDir(type = "") {
  // If ~/.openclaw/workspace exists, use it, otherwise use local project dir
  const base = existsSync(join(homedir(), ".openclaw")) ? HOME_CLAW_DIR : PROJECT_CLAW_DIR;
  return type ? join(base, type) : base;
}

async function ensureDir(path) {
  if (!existsSync(path)) {
    await mkdir(path, { recursive: true });
  }
}

/**
 * Saves a memory item to disk.
 */
export async function saveToMemory(type, key, content) {
  try {
    const dir = getMemoryDir(type);
    await ensureDir(dir);
    const filePath = join(dir, `${key.replace(/[^a-z0-9_-]/gi, "_").toLowerCase()}.md`);
    // Ensure content is always a string
    const text = typeof content === "string" ? content : JSON.stringify(content, null, 2);
    await writeFile(filePath, text, "utf8");
    console.error(`  [Memory] Saved memory item to ${type}/${key}.md`);
    // Reindex so OpenClaw memory search picks this up immediately
    await reindexMemory();
  } catch (error) {
    console.error(`  [Memory] Error saving to memory: ${error.message}`);
  }
}

/**
 * Loads all memory items for a specific type.
 * Returns array of { key, content }
 */
export async function loadFromMemory(type) {
  try {
    const dir = getMemoryDir(type);
    if (!existsSync(dir)) return [];
    const files = await readdir(dir);
    const results = [];
    for (const file of files) {
      if (file.endsWith(".md")) {
        const content = await readFile(join(dir, file), "utf8");
        results.push({
          key: file.replace(/\.md$/, ""),
          content,
        });
      }
    }
    return results;
  } catch (error) {
    console.error(`  [Memory] Error loading memory: ${error.message}`);
    return [];
  }
}

/**
 * Compares a candidate idea's title and pitch with past failed ideas in memory.
 * Returns a list of overlaps.
 */
export async function checkIdeaAgainstMemory(ideaTitle, ideaPitch) {
  const failedMemories = await loadFromMemory("failed_ideas");
  const overlaps = [];

  for (const mem of failedMemories) {
    // Basic Jaccard keyword overlap
    const overlap = calculateJaccard(
      `${ideaTitle} ${ideaPitch}`,
      mem.content
    );
    if (overlap > 0.45) {
      overlaps.push({
        key: mem.key,
        overlap: Number(overlap.toFixed(3)),
        content: mem.content,
      });
    }
  }

  return overlaps.sort((a, b) => b.overlap - a.overlap);
}

function calculateJaccard(a, b) {
  const stopWords = new Set("the a an and or but for with without of in on to from by as is are was were be been being this that these those into using use used via at across based towards toward through than then it its their our we show propose proposed".split(" "));
  
  function getKeywords(text) {
    const counts = new Set();
    for (const raw of String(text).toLowerCase().match(/[a-z][a-z0-9-]{3,}/g) || []) {
      const word = raw.replace(/^-|-$/g, "");
      if (word && !stopWords.has(word)) {
        counts.add(word);
      }
    }
    return counts;
  }

  const aw = getKeywords(a);
  const bw = getKeywords(b);
  if (!aw.size || !bw.size) return 0;
  
  let inter = 0;
  for (const word of aw) {
    if (bw.has(word)) inter += 1;
  }
  return inter / (aw.size + bw.size - inter);
}
/**
 * Semantic memory search using OpenClaw's memory search command.
 * Searches across all workspace memory files for relevant past runs,
 * failed ideas, and lessons learned.
 *
 * Falls back to keyword-based search if OpenClaw memory search is unavailable.
 */
export async function searchMemory(query) {
  // Try OpenClaw semantic memory search first
  try {
    const { spawnSync } = await import("node:child_process");
    const result = spawnSync(
      "openclaw",
      ["memory", "search", "--query", query, "--max-results", "20", "--json"],
      { encoding: "utf8", timeout: 15000 }
    );
    if (result.status === 0 && result.stdout?.trim()) {
      let parsed;
      try {
        parsed = JSON.parse(result.stdout);
      } catch {
        // Not JSON — openclaw returned plain text; parse score + path lines
        // Format: "0.352 memory/file.md:1-1\ntext\n"
        const items = [];
        const lines = result.stdout.split("\n");
        for (let i = 0; i < lines.length; i++) {
          const m = lines[i].match(/^([0-9.]+)\s+(.+?):(\d+)/);
          if (m) {
            items.push({
              score: parseFloat(m[1]),
              file: m[2],
              content: lines[i + 1]?.trim() || "",
            });
          }
        }
        if (items.length > 0) {
          console.error(`  [Memory] OpenClaw semantic search found ${items.length} results for: ${query}`);
          return items.map((r) => ({
            key: r.file,
            score: r.score,
            excerpt: r.content,
            source: "openclaw_memory_search",
          }));
        }
      }
      // Parsed JSON successfully
      if (parsed) {
        const items = Array.isArray(parsed) ? parsed : (parsed.results || []);
        console.error(`  [Memory] OpenClaw semantic search found ${items.length} results for: ${query}`);
        return items.map((r) => ({
          key: r.file || r.path || "",
          score: r.score || 0,
          excerpt: r.excerpt || r.content || r.text || "",
          source: "openclaw_memory_search",
        }));
      }
    }
  } catch {
    // OpenClaw memory search not available — fall through to keyword search
  }

  // Fallback: keyword search across memory files
  console.error(`  [Memory] Falling back to keyword search for: ${query}`);
  const allMemory = [
    ...(await loadFromMemory("failed_ideas")),
    ...(await loadFromMemory("lessons")),
    ...(await loadFromMemory("run_summaries")),
  ];
  const results = [];
  for (const mem of allMemory) {
    const score = calculateJaccard(query, mem.content);
    if (score > 0.15) {
      results.push({
        key: mem.key,
        score: Number(score.toFixed(3)),
        excerpt: mem.content.slice(0, 300),
        source: "keyword_search",
      });
    }
  }
  return results.sort((a, b) => b.score - a.score).slice(0, 20);
}

/**
 * Reindex OpenClaw memory after writing new files.
 * Ensures new memory entries are searchable immediately.
 */
export async function reindexMemory() {
  try {
    const { spawnSync } = await import("node:child_process");
    spawnSync("openclaw", ["memory", "index", "--force"], {
      encoding: "utf8",
      timeout: 20000,
    });
    console.error("  [Memory] OpenClaw memory index refreshed.");
  } catch {
    // Non-fatal — memory will be indexed on next heartbeat
  }
}
