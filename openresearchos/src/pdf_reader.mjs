/**
 * PDF Reader for OpenResearchOS v2 — OpenClaw Native.
 *
 * ALL web fetching goes through OpenClaw (openclaw infer web fetch).
 * Terminal/Python is ONLY used for local PDF conversion (binary processing).
 *
 * Priority chain for full-text reading:
 *   1. OpenClaw web fetch → ar5iv HTML (LaTeX → HTML, full paper)
 *   2. OpenClaw web fetch → arXiv HTML (arxiv.org/html/)
 *   3. OpenClaw web fetch → arXiv abstract page
 *   4. Terminal: download PDF + convert (local, last resort)
 *   5. Abstract from metadata (fallback)
 */

import { readFile, writeFile, mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as bridge from "./openclaw_bridge.mjs";
import * as terminal from "./terminal.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const UV = "/Users/mohanganesh/.local/bin/uv";

/**
 * Genuine full-paper extraction: download the PDF and read EVERY page with
 * PyMuPDF, preserving body text, tables (find_tables → Markdown), and figure
 * captions. This is the high-quality primary path (vs truncated HTML).
 */
export async function extractPdfFull(pdfUrl, evidenceDir, evidenceId) {
  const pdfPath = join(evidenceDir, "paper.pdf");
  const outDir = join(evidenceDir, "fulltext");
  const outPath = join(outDir, "fulltext.md");
  await mkdir(outDir, { recursive: true }).catch(() => {});

  if (existsSync(outPath)) {
    const cached = await readFile(outPath, "utf8");
    if (cached.length > 1000) {
      console.error(`  [PDFReader] 📦 Cached full text: ${evidenceId} (${cached.length} chars)`);
      // Sanitize cached text in case it was written before null-byte fix
      const sanitized = cached.replace(/\x00/g, " ").replace(/[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]/g, " ");
      return { success: true, text: sanitized, tool: "pymupdf_cached" };
    }
  }
  if (!pdfUrl) return { success: false };
  if (!existsSync(pdfPath)) {
    const dl = await terminal.downloadPdf(pdfUrl, evidenceDir, "paper.pdf");
    if (!dl) return { success: false };
  }
  const script = join(__dirname, "pdf_full_extract.py");
  const res = terminal.run(
    `${UV} run --with pymupdf python "${script}" "${pdfPath}" "${outPath}"`,
    { timeoutMs: 150000, silent: true }
  );
  if (existsSync(outPath)) {
    const text = await readFile(outPath, "utf8");
    if (text.length > 1000) {
      let meta = {};
      const m = (res.stdout || "").match(/\{[^{}]*"ok"[^{}]*\}/);
      try { if (m) meta = JSON.parse(m[0]); } catch {}
      console.error(`  [PDFReader] ✅ PyMuPDF full: ${evidenceId} — ${text.length} chars, ${meta.pages || "?"} pages, ${meta.tables || 0} tables, ${meta.captions || 0} captions`);
      return { success: true, text, tool: "pymupdf_full", meta };
    }
  }
  return { success: false };
}

const SECTION_PATTERNS = {
  abstract:     /(?:^|\n)#+?\s*(?:abstract)/i,
  introduction: /(?:^|\n)#+?\s*(?:\d+\.?\s*)?introduction/i,
  related:      /(?:^|\n)#+?\s*(?:\d+\.?\s*)?related\s+work/i,
  method:       /(?:^|\n)#+?\s*(?:\d+\.?\s*)?(?:method(?:ology)?|approach|model|framework|our\s+(?:method|approach|model))/i,
  experiments:  /(?:^|\n)#+?\s*(?:\d+\.?\s*)?(?:experiment|evaluation|setup|experiment(?:al)?\s+(?:setup|design))/i,
  results:      /(?:^|\n)#+?\s*(?:\d+\.?\s*)?(?:result|performance|comparison|main\s+result)/i,
  ablation:     /(?:^|\n)#+?\s*(?:\d+\.?\s*)?ablation/i,
  limitations:  /(?:^|\n)#+?\s*(?:\d+\.?\s*)?(?:limitation|discuss|future\s+work|conclusion)/i,
};

// ─── OpenClaw-Native Paper Fetching ─────────────────────────────────────────

/**
 * Fetch full paper text using OpenClaw web fetch.
 * This is the primary method — uses `openclaw infer web fetch` under the hood.
 *
 * @param {string} arxivId     - e.g. "2303.11366"
 * @param {string} evidenceDir - local dir to cache results
 * @returns {{ success, text, tool }}
 */
export async function fetchArxivHtml(arxivId, evidenceDir) {
  const cleanId = arxivId.replace(/v\d+$/, "");
  const outPath = join(evidenceDir, "fulltext", "fulltext_html.md");

  await mkdir(join(evidenceDir, "fulltext"), { recursive: true }).catch(() => {});

  // Use cached result if available
  if (existsSync(outPath)) {
    const cached = await readFile(outPath, "utf8");
    if (cached.length > 200) {
      console.error(`  [PDFReader] 📦 Cached: ${arxivId} (${cached.length} chars)`);
      return { success: true, text: cached, tool: "cached", markdownPath: outPath };
    }
  }

  // ── Priority 1: OpenClaw → ar5iv (full LaTeX-rendered HTML) ──
  const ar5ivUrl = `https://ar5iv.labs.arxiv.org/html/${cleanId}`;
  console.error(`  [PDFReader] 🌐 OpenClaw fetch: ar5iv ${cleanId}`);
  const ar5ivText = await bridge.webFetch(ar5ivUrl, { format: "text" });
  if (ar5ivText && ar5ivText.length > 1000) {
    const cleaned = cleanFetchedText(ar5ivText);
    if (cleaned.length > 500) {
      await writeFile(outPath, cleaned.slice(0, 100000), "utf8").catch(() => {});
      console.error(`  [PDFReader] ✅ ar5iv via OpenClaw: ${arxivId} — ${cleaned.length} chars`);
      return { success: true, text: cleaned, tool: "arxiv_html_openclaw", markdownPath: outPath };
    }
  }

  // ── Priority 2: OpenClaw → arXiv HTML endpoint ──
  const htmlUrl = `https://arxiv.org/html/${cleanId}`;
  console.error(`  [PDFReader] 🌐 OpenClaw fetch: arxiv html ${cleanId}`);
  const htmlText = await bridge.webFetch(htmlUrl, { format: "text" });
  if (htmlText && htmlText.length > 500) {
    const cleaned = cleanFetchedText(htmlText);
    if (cleaned.length > 300) {
      await writeFile(outPath, cleaned.slice(0, 100000), "utf8").catch(() => {});
      console.error(`  [PDFReader] ✅ arxiv/html via OpenClaw: ${arxivId} — ${cleaned.length} chars`);
      return { success: true, text: cleaned, tool: "arxiv_html2_openclaw", markdownPath: outPath };
    }
  }

  // ── Priority 3: OpenClaw → abstract page ──
  const absUrl = `https://arxiv.org/abs/${cleanId}`;
  console.error(`  [PDFReader] 🌐 OpenClaw fetch: arxiv abstract ${cleanId}`);
  const absText = await bridge.webFetch(absUrl, { format: "text" });
  if (absText && absText.length > 100) {
    const cleaned = cleanFetchedText(absText).slice(0, 8000);
    await writeFile(outPath, cleaned, "utf8").catch(() => {});
    console.error(`  [PDFReader] ℹ️  abstract via OpenClaw: ${arxivId} — ${cleaned.length} chars`);
    return { success: true, text: cleaned, tool: "arxiv_abstract_openclaw", markdownPath: outPath };
  }

  return { success: false };
}

/**
 * Download PDF locally and convert to markdown (terminal-based).
 * Used only when OpenClaw web fetch cannot return readable text.
 */
export async function downloadAndConvert(pdfUrl, evidenceDir, evidenceId) {
  const pdfPath = join(evidenceDir, "paper.pdf");
  const outputDir = join(evidenceDir, "fulltext");
  const mdPath = join(outputDir, "fulltext.md");

  // Use cached conversion
  if (existsSync(mdPath)) {
    const text = await readFile(mdPath, "utf8");
    return { success: true, markdownPath: mdPath, text, tool: "cached" };
  }

  // First: try OpenClaw to fetch the PDF URL as text (Firecrawl can read PDFs)
  console.error(`  [PDFReader] 🌐 OpenClaw fetch PDF: ${pdfUrl.slice(0, 70)}`);
  const pdfText = await bridge.webFetch(pdfUrl, { format: "text" });
  if (pdfText && pdfText.length > 500) {
    const cleaned = cleanFetchedText(pdfText);
    await mkdir(outputDir, { recursive: true }).catch(() => {});
    await writeFile(mdPath, cleaned.slice(0, 100000), "utf8").catch(() => {});
    console.error(`  [PDFReader] ✅ PDF via OpenClaw: ${evidenceId} — ${cleaned.length} chars`);
    return { success: true, markdownPath: mdPath, text: cleaned, tool: "pdf_openclaw" };
  }

  // Fallback: local download + terminal conversion
  if (!existsSync(pdfPath)) {
    console.error(`  [PDFReader] 📥 Local download: ${evidenceId}`);
    const dl = await terminal.downloadPdf(pdfUrl, evidenceDir, "paper.pdf");
    if (!dl) {
      console.error(`  [PDFReader] ❌ Download failed for ${evidenceId}`);
      return { success: false };
    }
  }

  return terminal.pdfToMarkdown(pdfPath, outputDir);
}

// ─── Text Cleaning ───────────────────────────────────────────────────────────

/**
 * Clean text returned by OpenClaw web fetch.
 * OpenClaw may return HTML with EXTERNAL_UNTRUSTED_CONTENT wrappers.
 */
function cleanFetchedText(text) {
  return text
    // Strip OpenClaw's trust wrappers
    .replace(/<<<EXTERNAL_UNTRUSTED_CONTENT[^>]*>>>/g, "")
    .replace(/<<<END_EXTERNAL_UNTRUSTED_CONTENT[^>]*>>>/g, "")
    // Strip HTML if present
    .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, "")
    .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, "")
    .replace(/<[^>]+>/g, " ")
    // Clean whitespace
    .replace(/\n{3,}/g, "\n\n")
    .replace(/ {3,}/g, " ")
    .trim();
}

// ─── Section Extraction ─────────────────────────────────────────────────────

/**
 * Split full-text into research paper sections.
 */
export function extractSections(fullText) {
  const sections = {};
  const sortedKeys = Object.keys(SECTION_PATTERNS);

  const positions = [];
  for (const key of sortedKeys) {
    const match = SECTION_PATTERNS[key].exec(fullText);
    if (match) positions.push({ key, index: match.index });
  }
  positions.sort((a, b) => a.index - b.index);

  for (let i = 0; i < positions.length; i++) {
    const start = positions[i].index;
    const end = i + 1 < positions.length ? positions[i + 1].index : fullText.length;
    sections[positions[i].key] = fullText.slice(start, end).slice(0, 16000).trim();
  }

  // Fallback: chunk if no sections found
  if (Object.keys(sections).length === 0) {
    const chunkSize = Math.floor(fullText.length / 4);
    sections.introduction = fullText.slice(0, chunkSize).trim();
    sections.method = fullText.slice(chunkSize, 2 * chunkSize).trim();
    sections.results = fullText.slice(2 * chunkSize, 3 * chunkSize).trim();
    sections.conclusion = fullText.slice(3 * chunkSize).trim();
  }

  return sections;
}

// ─── Deep LLM Extraction ────────────────────────────────────────────────────

/**
 * Use OpenClaw's LLM (via llm.ask → openclaw infer model run) to extract
 * structured information from a paper's full text sections.
 */
export async function deepExtract(sections, llmClient, paperTitle, topic = "") {
  const intro = sections.introduction || "";
  const methodSection = sections.method || sections.experiments || intro || "";
  const resultsSection = sections.results || sections.ablation || sections.experiments || "";
  const ablationSection = sections.ablation || "";
  const limitSection = sections.limitations || sections.related || "";
  const abstract = sections.abstract || intro.slice(0, 1200);

  const jsonOf = (s) => { const m = s && s.match(/\{[\s\S]*\}/); if (!m) return null; try { return JSON.parse(m[0]); } catch { return null; } };

  // ── Pass A: identity, problem, contributions, method ──
  const pA = `You are a researcher taking structured notes on a paper. Be precise, use the paper's own facts.
PAPER: "${paperTitle}"

=== ABSTRACT / INTRO ===
${(abstract + "\n" + intro).slice(0, 4000)}

=== METHOD / APPROACH ===
${methodSection.slice(0, 9000)}

Return ONLY JSON (no markdown):
{
  "problem_addressed": "the problem/gap this paper targets (1 sentence)",
  "core_claim": "one-sentence thesis",
  "key_contributions": ["contribution 1", "contribution 2"],
  "novelty": "what is new vs prior work",
  "method_summary": "exact technical approach, 1-3 sentences",
  "method_components": ["building block 1", "building block 2"],
  "key_innovation": "the single most important new idea / mechanism (WHY it works)",
  "assumptions": ["assumptions the method relies on"],
  "theoretical_grounding": "any proof/bound/derivation, or null"
}`;

  // ── Pass B: setup, results, tables (real numbers) ──
  const pB = `From the EXPERIMENTS/RESULTS of the paper "${paperTitle}", extract the setup and concrete numeric findings. Tables are included as Markdown — read them.

=== EXPERIMENTAL SETUP + RESULTS + TABLES ===
${(resultsSection + "\n" + ablationSection).slice(0, 13000)}

Return ONLY JSON (no markdown):
{
  "datasets_used": ["dataset names (+ size if stated)"],
  "baselines_compared": ["baseline methods"],
  "metrics_used": ["accuracy","ECE","AUROC", "..."],
  "hyperparameters": "key training settings if stated, else null",
  "compute_required": "e.g. '8xA100 3d' / 'single GPU' / 'CPU'",
  "seeds": "number of seeds/runs if stated, else null",
  "headline_result": "best result with its number AND dataset",
  "results_table": [{"method":"name","dataset":"name","metric":"acc","value":"94.2%"}],
  "ablations": ["what each ablated component contributes"],
  "improvement_over_baseline": "delta vs baseline if stated, else null"
}`;

  // ── Pass C: critical reading + relevance to our topic ──
  const pC = `Critically assess the paper "${paperTitle}" for a research project on: "${topic || "the topic"}".

=== ABSTRACT ===
${abstract.slice(0, 1500)}
=== LIMITATIONS / DISCUSSION / RELATED ===
${limitSection.slice(0, 6000)}

Return ONLY JSON (no markdown):
{
  "stated_limitations": ["limitations the authors admit"],
  "unstated_weaknesses": ["reviewer-style weaknesses not admitted"],
  "threats_to_validity": ["confounds / generalization risks"],
  "reproducibility": "high|medium|low (code? data? seeds?)",
  "open_questions": ["unanswered questions"],
  "future_work": ["authors' suggested next steps"],
  "gaps_this_enables": ["concrete new research directions this paper opens"],
  "relevance_to_topic": "how this connects to the project topic (1 sentence)",
  "relation_type": "baseline|competitor|method_source|dataset_source|foundational|tangential",
  "can_run_on_macbook": true_or_false,
  "reusable_assets": ["datasets/code/methods we could reuse locally"],
  "code_available": true_or_false,
  "code_url": "github url or null"
}`;

  const runPass = async (label, prompt) => {
    try {
      const r = await llmClient.complete(prompt);
      const parsed = jsonOf(r);
      if (!parsed) console.error(`  [PDFReader] ${label} returned no JSON`);
      return parsed;
    } catch (err) {
      console.error(`  [PDFReader] ${label} failed: ${err.message}`);
      return null;
    }
  };

  const [a, b, c] = [await runPass("passA", pA), await runPass("passB", pB), await runPass("passC", pC)];

  if (!a && !b && !c) return fallbackExtract(sections, paperTitle);

  const card = { ...(a || {}), ...(b || {}), ...(c || {}) };
  // Back-compat aliases used elsewhere in the pipeline
  card.best_result = card.headline_result || card.best_result || null;
  card.key_results = card.results_table && card.results_table.length
    ? Object.fromEntries(card.results_table.slice(0, 6).map((r, i) => [`${r.dataset || "set"}_${r.metric || i}`, r.value]))
    : (card.key_results || {});
  card.limitations_stated = card.stated_limitations || [];
  card.reproducibility = card.reproducibility || (card.code_available ? "medium" : "low");
  return card;
}

function fallbackExtract(sections, paperTitle) {
  const allText = Object.values(sections).join(" ");
  const datasetPatterns = [
    /\b(MNIST|CIFAR-\d+|ImageNet|GLUE|SuperGLUE|MMLU|HellaSwag|ARC|TruthfulQA|GSM8K|HumanEval|MATH|WikiText|PTB|SQuAD|CoQA|RACE|BoolQ)\b/gi,
  ];
  const datasets = new Set();
  for (const pat of datasetPatterns) {
    let m;
    while ((m = pat.exec(allText)) !== null) datasets.add(m[1] || m[0]);
  }
  const accMatch = allText.match(/(?:accuracy|acc)[:\s]+(\d+\.?\d*)\s*%?/i);

  return {
    method_summary: `Research paper: ${paperTitle.slice(0, 100)}`,
    key_innovation: "See full text",
    datasets_used: [...datasets].slice(0, 5),
    baselines_compared: [],
    key_results: accMatch ? { accuracy: parseFloat(accMatch[1]) } : {},
    best_result: null,
    limitations_stated: [],
    code_available: allText.toLowerCase().includes("github.com"),
    code_url: (allText.match(/https?:\/\/github\.com\/[^\s"')]+/) || [])[0] || null,
    reproducibility: "unknown",
    compute_required: "unknown",
    can_run_on_macbook: true,
  };
}

// ─── Main: Full Paper Processing ────────────────────────────────────────────

/**
 * Process a single paper end-to-end:
 * 1. Fetch full text (OpenClaw web fetch → PDF fallback)
 * 2. Extract sections
 * 3. LLM deep extraction (OpenClaw infer model run)
 * 4. Save to evidence directory
 */
export async function processFullPaper(paper, evidenceDir, llmClient, topic = "") {
  await mkdir(evidenceDir, { recursive: true }).catch(() => {});

  const deepReadPath = join(evidenceDir, "deep_read.json");
  if (existsSync(deepReadPath)) {
    console.error(`  [PDFReader] 📦 Cached deep_read: ${paper.title?.slice(0, 50)}`);
    return JSON.parse(await readFile(deepReadPath, "utf8"));
  }

  let fullText = null;
  let tool = "none";
  let extractMeta = null;

  // ── 1. PRIMARY: download PDF + PyMuPDF full read (every page + tables + captions) ──
  const pdfUrl = paper.openAccessPdf || paper.pdf_url ||
    (paper.arxiv_id ? `https://arxiv.org/pdf/${paper.arxiv_id.replace(/v\d+$/, "")}.pdf` : null);
  if (pdfUrl) {
    const full = await extractPdfFull(pdfUrl, evidenceDir, paper.evidence_id || "EV?");
    if (full.success && full.text) {
      fullText = full.text;
      tool = full.tool;
      extractMeta = full.meta || null;
    }
  }

  // ── 2. Fallback: OpenClaw fetch arXiv HTML (full LaTeX render) ──
  if (!fullText && paper.arxiv_id) {
    const htmlResult = await fetchArxivHtml(paper.arxiv_id, evidenceDir);
    if (htmlResult.success && htmlResult.text) {
      fullText = htmlResult.text;
      tool = htmlResult.tool;
    }
  }

  // ── 3. Fallback: OpenClaw fetch PDF/paper URL directly ──
  if (!fullText) {
    const anyUrl = pdfUrl || paper.url;
    if (anyUrl) {
      const converted = await downloadAndConvert(anyUrl, evidenceDir, paper.evidence_id || "EV?");
      if (converted.success) {
        fullText = converted.text;
        tool = converted.tool;
      }
    }
  }

  // ── 4. Abstract only (last resort) ──
  if (!fullText && paper.abstract) {
    console.error(`  [PDFReader] ⚠️ Abstract only: ${paper.title?.slice(0, 50)}`);
    fullText = `# ${paper.title}\n\n## Abstract\n\n${paper.abstract}`;
    tool = "abstract_only";
  }

  if (!fullText) {
    return { error: "No text available", title: paper.title };
  }

  // Sanitize the full text to prevent null byte crashes in spawnSync downstream
  fullText = fullText
    .replace(/\x00/g, " ")
    .replace(/[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]/g, " ");

  // Save raw full text
  const ftPath = join(evidenceDir, "fulltext.md");
  if (!existsSync(ftPath)) {
    await writeFile(ftPath, fullText, "utf8").catch(() => {});
  }

  // Extract sections
  const sections = extractSections(fullText);
  await writeFile(
    join(evidenceDir, "sections.json"),
    JSON.stringify(Object.fromEntries(Object.entries(sections).map(([k, v]) => [k, v.slice(0, 200)])), null, 2),
    "utf8"
  ).catch(() => {});

  // Deep LLM extraction via OpenClaw infer model run (3-pass Paper Card)
  const deepRead = await deepExtract(sections, llmClient, paper.title || "", topic);
  deepRead.evidence_id = paper.evidence_id;
  deepRead.title = paper.title;
  deepRead.year = paper.year;
  deepRead.arxiv_id = paper.arxiv_id;
  deepRead.full_text_tool = tool;
  deepRead.full_text_chars = fullText.length;
  deepRead.sections_found = Object.keys(sections);
  if (extractMeta) {
    deepRead.pdf_pages = extractMeta.pages;
    deepRead.pdf_tables = extractMeta.tables;
    deepRead.pdf_captions = extractMeta.captions;
  }

  await writeFile(deepReadPath, JSON.stringify(deepRead, null, 2), "utf8");
  console.error(`  [PDFReader] ✅ Deep read: ${paper.title?.slice(0, 50)} | ${tool} | ${fullText.length} chars`);

  return deepRead;
}

/**
 * Process all papers with rate limiting.
 * Each paper: OpenClaw web fetch → sections → LLM extraction.
 */
export async function processAllPapers(papers, runsDir, llmClient, { maxPapers = 60, delayMs = 1200 } = {}) {
  const results = [];
  const subset = papers.slice(0, maxPapers);

  console.error(`\n${"═".repeat(60)}`);
  console.error(`[PDFReader] Deep reading ${subset.length} papers via OpenClaw...`);
  console.error(`${"═".repeat(60)}`);

  for (let i = 0; i < subset.length; i++) {
    const paper = subset[i];
    const evidenceId = paper.evidence_id || `EV${String(i + 1).padStart(3, "0")}`;
    paper.evidence_id = evidenceId;

    const evidenceDir = join(runsDir, "evidence", evidenceId);
    console.error(`\n[PDFReader] [${i + 1}/${subset.length}] ${evidenceId}: ${paper.title?.slice(0, 60)}`);

    try {
      const deepRead = await processFullPaper(paper, evidenceDir, llmClient);
      results.push({ ...paper, deep_read: deepRead });
    } catch (err) {
      console.error(`  [PDFReader] ❌ Error: ${err.message}`);
      results.push({ ...paper, deep_read: { error: err.message } });
    }

    if (i < subset.length - 1) {
      await new Promise((r) => setTimeout(r, delayMs));
    }
  }

  console.error(`\n[PDFReader] ✅ Done: ${results.filter(r => !r.deep_read?.error).length}/${results.length} papers deep-read`);
  return results;
}
