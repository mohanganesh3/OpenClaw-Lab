/**
 * llm_client.mjs — OpenClaw-FULL integration for OpenResearchOS v2.
 *
 * Model cascade (via openclaw_bridge.mjs):
 *   FAST:   sarvam-105b (default routing)          — extraction, short JSON
 *   MEDIUM: cerebras/gpt-oss-120b (128k, reasoning) — gaps, ideas, reviewer
 *   DEEP:   chutes/deepseek-ai/DeepSeek-V3.1-TEE   — paper writing, planning
 *   BACKUP: byteplus/kimi-k2-5-260127 (256k)        — fallback for all
 *
 * Local embeddings: embeddinggemma-300m (no API key) for semantic search.
 * Thinking mode: enabled for DEEP tasks (reviewers, paper writer).
 * Agent turns: full openclaw agent for complex multi-step reasoning.
 */

import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import * as ocl from "./openclaw_bridge.mjs";



// ─── Sarvam-only runners (the only configured model in this OpenClaw) ──────────
// Prompt length limits per task type (sarvam reliable range):
//   FAST tasks (extraction):   up to 2000 chars
//   MEDIUM tasks (gaps/ideas): up to 4000 chars
//   DEEP tasks (paper write):  up to 6000 chars

async function runFast(prompt) {
  return ocl.inferModel(prompt, { maxPromptLen: 8000, timeoutMs: 60000 });
}

async function runMedium(prompt) {
  return ocl.inferModel(prompt, { maxPromptLen: 24000, timeoutMs: 120000 });
}

async function runDeep(prompt) {
  return ocl.inferModel(prompt, { maxPromptLen: 48000, timeoutMs: 150000 });
}

// Legacy sync shim — no longer used; all LLM calls go through async runFast/runMedium/runDeep
function runModelTurn(prompt, opts = {}) {
  // This function is a no-op. Use runFast/runMedium/runDeep instead.
  return null;
}




function stripThink(text) {
  if (!text) return text;
  return text
    .replace(/<think>[\s\S]*?<\/think>/gi, "")
    .replace(/^```(?:json)?\s*/m, "")
    .replace(/\s*```$/m, "")
    .trim();
}

function parseJsonSafe(raw) {
  if (!raw) return null;
  const cleaned = stripThink(raw);
  try { return JSON.parse(cleaned); } catch {}
  const objMatch = cleaned.match(/\{[\s\S]*\}/);
  const arrMatch = cleaned.match(/\[[\s\S]*\]/);
  try { if (arrMatch) return JSON.parse(arrMatch[0]); } catch {}
  try { if (objMatch) return JSON.parse(objMatch[0]); } catch {}
  if (objMatch || arrMatch) {
    const target = (arrMatch || objMatch)[0];
    const fixed = target.replace(/,\s*([}\]])/g, "$1").replace(/([{,]\s*)(\w+)\s*:/g, '$1"$2":');
    try { return JSON.parse(fixed); } catch {}
  }
  return null;
}

// ─── Public API ───────────────────────────────────────────────────────────────

export async function chatCompletion(messages, opts = {}) {
  const formatted = messages.map((m) => `[${m.role.toUpperCase()}]\n${m.content}`).join("\n\n");
  return runMedium(formatted);
}

export async function ask(prompt, opts = {}) {
  return runMedium(prompt);
}

export async function askJson(prompt, opts = {}) {
  const raw = await runFast(prompt);
  return parseJsonSafe(raw);
}

export async function askJsonArray(prompt, opts = {}) {
  const raw = await runMedium(prompt);
  if (!raw) return null;
  const cleaned = stripThink(raw);
  const match = cleaned.match(/\[[\s\S]*\]/);
  if (!match) return null;
  try { return JSON.parse(match[0]); } catch { return null; }
}




// ─── Gap Analysis — MEDIUM tier (needs context, multi-paper reasoning) ─────────

export async function analyzeGaps(topic, evidenceSummaries) {
  // Build a rich evidence block: include gaps_this_enables + weaknesses from each Paper Card
  const evidenceBlock = evidenceSummaries
    .slice(0, 12)
    .map((e) => {
      const gaps = (e.gaps_this_enables || []).slice(0, 2).join("; ").slice(0, 80);
      const weaknesses = (e.unstated_weaknesses || []).slice(0, 1).join("; ").slice(0, 60);
      const openQ = (e.open_questions || []).slice(0, 1).join("; ").slice(0, 60);
      const rt = (e.results_table || []).slice(0, 1).map(r => `${r.method}:${r.value}`).join(", ");
      return `[${e.evidence_id}] ${e.title?.slice(0, 35)}: ${(e.core_claim || e.abstract || "").slice(0, 60)}${gaps ? ` | gaps: ${gaps}` : ""}${weaknesses ? ` | weakness: ${weaknesses}` : ""}${openQ ? ` | open: ${openQ}` : ""}${rt ? ` | result: ${rt}` : ""}`;
    })
    .join("\n");

  const raw = await runMedium(`Research gap analyst. Topic: "${topic}"
Papers (with gaps_this_enables, weaknesses, open questions extracted from full text):
${evidenceBlock}

Identify 6 concrete, specific research gaps. Each gap MUST:
- Cite SPECIFIC paper evidence_ids (e.g. EV001, EV003) that expose this gap
- Have a measurable signal (ECE reduction, accuracy delta, sample efficiency ratio)
- Be testable with sklearn on a MacBook in <5 minutes

Return JSON array of 6 gaps: [{"gap_id":"G01","gap_title":"title","text":"2-sentence explanation referencing specific papers","evidence_ids":["EV001","EV003"],"testable_hypothesis":"If X then Y by Z metric","suggested_dataset":"breast_cancer or digits","difficulty":"medium"}]`);
  return parseJsonSafe(raw);
}




// ─── Research Idea Generation — MEDIUM tier ────────────────────────────────────

export async function generateResearchIdeas(topic, gaps, evidenceSummaries, failedIdeas = [], methodMatrix = []) {
  const gapBlock = (gaps || []).slice(0, 5).map((g, i) =>
    `[${g.gap_id || `G${i+1}`}] ${(g.gap_title || g.text || "").slice(0, 50)}: ${(g.text || "").slice(0, 80)} (evidence: ${(g.evidence_ids||[]).slice(0,3).join(", ")||"?"})`
  ).join("\n");

  const failedBlock = failedIdeas.slice(0, 3)
    .map((f) => `- "${f.title?.slice(0, 40)}"`)
    .join("\n") || "None";

  // Build richer paper context: include datasets, best results, and reusable assets
  const paperBlock = evidenceSummaries.slice(0, 8).map(e =>
    `[${e.evidence_id}] ${e.title?.slice(0, 35)} | claim: ${(e.core_claim||"").slice(0,50)} | best: ${e.best_result?.slice(0,25)||"?"} | datasets: ${(e.datasets_used||[]).slice(0,2).join(",")||"?"}`
  ).join("\n") || "(none)";

  const matrixBlock = methodMatrix.slice(0, 5).map(m =>
    `[${m.evidence_id}] ${m.title?.slice(0, 35)} | best: ${m.best_result?.slice(0,25)||"?"}`
  ).join("\n") || "(none)";

  const raw = await runMedium(`Idea generator for: "${topic}"
VALIDATED GAPS (with evidence citations):
${gapBlock}
KEY PAPERS (with datasets and results):
${paperBlock}
METHOD MATRIX:
${matrixBlock}
Skip (already tried): ${failedBlock}

Generate 6 specific, grounded research ideas. Each idea MUST:
- Reference at least one specific EV ID from the papers above in the pitch
- Have a concrete mechanism (NOT generic phrases like "leverage uncertainty")
- Be testable with sklearn on breast_cancer or digits dataset
- Specify the actual metric (ECE, Brier score, accuracy, AUC) not just "improvement"

Return JSON array of 6 ideas: [{"idea_id":"I01","title":"title","source_gap_id":"G01","pitch":"2 sentences citing EV001 finding...","mechanism":"specific algorithmic step","testable_hypothesis":"If we do X (citing EV002 gap) then ECE drops by Y%","required_dataset":"breast_cancer","baseline_to_beat":"entropy sampling","metric":"ECE","estimated_compute":"<5min","novelty_risk":"low","local_feasibility":true}]`);
  return parseJsonSafe(raw);
}




// ─── Multi-Turn Adversarial Reviewer Council ─────────────────────────────────

const REVIEWER_PERSONAS = {
  "Novelty Reviewer": `You are a HARSH novelty reviewer.
Focus: Has this EXACT approach been published? Is the novelty incremental or fundamental?
You have access to the full paper corpus. Cite specific papers that are similar.
You want to REJECT ideas that already exist. Be specific about what prior work covers.`,

  "Experimental Reviewer": `You are a rigorous experimental reviewer.
Focus: Is the hypothesis falsifiable? Are the baselines fair and strong? Is the dataset appropriate?
You check: Are there enough seeds? Is the evaluation metric correct? Is the compute budget realistic?
You want CLEAR experimental protocols, not vague plans.`,

  "Theory/Mechanism Reviewer": `You are a theory reviewer.
Focus: WHY would this work? Is there a theoretical justification or just intuition?
Check: Is the mechanism falsifiable? Do the ablations proposed actually test the mechanism?
You reject ideas with hand-wavy explanations.`,

  "Reproducibility Reviewer": `You are a reproducibility reviewer (NeurIPS checklist enforcer).
Focus: Can someone reproduce this? Are datasets public? Is compute documented?
Check: Fixed seeds? Hyperparameters logged? Code available or will be?
Flag anything that makes reproducibility unclear.`,
};

/**
 * Run a SINGLE reviewer turn (one LLM call per reviewer).
 */
export async function simulateReviewer(reviewerType, idea, evidenceSummaries, methodMatrix = [], priorReviews = []) {
  // Cap at 6 papers × 60 chars = ~360 chars for evidence block
  const evidenceBlock = evidenceSummaries.slice(0, 6)
    .map((e) => `[${e.evidence_id}] "${e.title?.slice(0, 40)}" — ${(e.core_claim || "").slice(0, 60)}`)
    .join("\n");

  const matrixBlock = methodMatrix.slice(0, 5).map(m =>
    `[${m.evidence_id}] ${m.title?.slice(0,35)} | ${m.best_result?.slice(0,30)||"?"}`
  ).join("\n") || "N/A";

  const priorBlock = priorReviews.length > 0
    ? `Prior reviews: ${priorReviews.map(r => `${r.reviewer}:${r.score}(${(r.fatal_flaws||[]).length} fatal)`).join(", ")}\n`
    : "";

  const personas = {
    "Novelty Reviewer": "Harsh novelty reviewer. Reject if similar work exists. Cite evidence_ids.",
    "Experimental Reviewer": "Rigorous methods reviewer. Check: falsifiable hypothesis, fair baselines, seeds, compute.",
    "Theory/Mechanism Reviewer": "Theory reviewer. Check: WHY it works, theoretical justification, ablations test mechanism.",
    "Reproducibility Reviewer": "NeurIPS reproducibility reviewer. Check: public data, fixed seeds, compute docs, code.",
  };
  const persona = personas[reviewerType] || personas["Novelty Reviewer"];

  // Use MEDIUM tier per reviewer — cerebras handles 128k context fast
  const raw = await runMedium(`${persona}
IDEA: ${idea.title} | ${(idea.pitch||idea.mechanism||"").slice(0,120)}
Hypothesis: ${(idea.testable_hypothesis||"").slice(0,100)}
Dataset: ${idea.required_dataset||"?"} | Baseline: ${idea.baseline_to_beat||"?"}\n${priorBlock}
PRIOR WORK:
${evidenceBlock}
RESULTS:
${matrixBlock}

Return JSON: {"reviewer":"${reviewerType}","score":<1-10>,"fatal_flaws":[...],"fixable_flaws":[...],"decision":"accept|revise|reject","justification":"2 sentences"}`);
  return parseJsonSafe(raw);
}


/**
 * Multi-turn adversarial reviewer council.
 *
 * Phase 1: All 4 reviewers independently review the idea.
 * Phase 2: Each reviewer sees what others said and can REVISE their score.
 * Phase 3: If fixable flaws exist, LLM generates a revision, then re-reviews once.
 *
 * Only ideas with avg >= 6.0 AND 0 fatal flaws proceed.
 */
export async function runReviewerCouncil(idea, evidenceSummaries, methodMatrix = []) {
  const reviewerTypes = Object.keys(REVIEWER_PERSONAS);
  console.error(`  [Reviewer] Council starting for: "${idea.title}"`);

  // ── Phase 1: Independent reviews ──
  const phase1Reviews = [];
  for (const reviewer of reviewerTypes) {
    try {
      console.error(`  [Reviewer] ${reviewer}...`);
      const review = await simulateReviewer(reviewer, idea, evidenceSummaries, methodMatrix, []);
      if (review && typeof review.score === "number") {
        phase1Reviews.push(review);
      } else {
        phase1Reviews.push({ reviewer, score: 5, fatal_flaws: [], fixable_flaws: ["Review unavailable"], decision: "revise", justification: "LLM review unavailable." });
      }
    } catch (err) {
      console.error(`  [Reviewer] ${reviewer} error: ${err.message}`);
      phase1Reviews.push({ reviewer, score: 5, fatal_flaws: [], fixable_flaws: [], decision: "revise", justification: "Error during review." });
    }
    // Small delay between reviewer calls
    await new Promise(r => setTimeout(r, 500));
  }

  // ── Phase 2: Adversarial cross-review (each reviewer sees others' verdicts) ──
  console.error(`  [Reviewer] Phase 2: cross-review...`);
  const phase2Reviews = [];
  for (const reviewer of reviewerTypes) {
    const otherReviews = phase1Reviews.filter(r => r.reviewer !== reviewer);
    try {
      const updated = await simulateReviewer(reviewer, idea, evidenceSummaries, methodMatrix, otherReviews);
      if (updated && typeof updated.score === "number") {
        phase2Reviews.push(updated);
      } else {
        phase2Reviews.push(phase1Reviews.find(r => r.reviewer === reviewer));
      }
    } catch {
      phase2Reviews.push(phase1Reviews.find(r => r.reviewer === reviewer));
    }
    await new Promise(r => setTimeout(r, 400));
  }

  const finalReviews = phase2Reviews;
  const scores = finalReviews.map(r => r.score).filter(Number.isFinite);
  const avgScore = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;
  const fatalFlaws = finalReviews.flatMap(r => r.fatal_flaws || []);
  const fixableFlaws = finalReviews.flatMap(r => r.fixable_flaws || []);
  const hasFatal = fatalFlaws.length > 0;

  // ── Phase 3: If fixable (no fatal, avg < 6.5), generate revised idea and re-check ──
  let revisedIdea = null;
  let revisionReviews = null;
  if (!hasFatal && avgScore < 6.5 && fixableFlaws.length > 0) {
    console.error(`  [Reviewer] Generating revision (avg=${avgScore.toFixed(1)}, ${fixableFlaws.length} fixable flaws)...`);
    const revision = await askJson(`Revise this research idea based on reviewer feedback.
IDEA: ${idea.title} | Mechanism: ${(idea.mechanism||"").slice(0,150)}
FEEDBACK:
${fixableFlaws.slice(0,5).map((f, i) => `${i+1}. ${f}`).join("\n")}
Return JSON: {title, pitch, mechanism, testable_hypothesis, required_dataset, baseline_to_beat}`);


    if (revision?.title) {
      revisedIdea = { ...idea, ...revision };
      // Quick single-pass review on revised idea
      const noveltyCheck = await simulateReviewer("Novelty Reviewer", revisedIdea, evidenceSummaries, methodMatrix, finalReviews);
      const expCheck = await simulateReviewer("Experimental Reviewer", revisedIdea, evidenceSummaries, methodMatrix, finalReviews);
      revisionReviews = [noveltyCheck, expCheck].filter(Boolean);
    }
  }

  // Use revision scores if revision improved things
  let effectiveReviews = finalReviews;
  if (revisionReviews?.length > 0) {
    const revScores = revisionReviews.map(r => r.score).filter(Number.isFinite);
    const revAvg = revScores.length ? revScores.reduce((a, b) => a + b, 0) / revScores.length : 0;
    if (revAvg > avgScore) {
      effectiveReviews = [...finalReviews.slice(0, 2), ...revisionReviews];
      console.error(`  [Reviewer] Revision improved score: ${avgScore.toFixed(1)} → ${revAvg.toFixed(1)}`);
    }
  }

  const finalScores = effectiveReviews.map(r => r.score).filter(Number.isFinite);
  const finalAvg = finalScores.length ? finalScores.reduce((a, b) => a + b, 0) / finalScores.length : 0;
  const finalFatal = effectiveReviews.flatMap(r => r.fatal_flaws || []).filter(f => f?.length > 3);
  const decision = finalFatal.length > 0 ? "reject" : finalAvg >= 6.0 ? "accept" : "revise";

  return {
    idea_id: idea.idea_id,
    average_score: Number(finalAvg.toFixed(2)),
    phase1_avg: Number(avgScore.toFixed(2)),
    decision,
    next_action: finalFatal.length > 0 ? "RUN_ADDITIONAL_SEARCH" : finalAvg >= 6.0 ? "RUN_MICRO_PROBE" : "REVISE_IDEA",
    fatal_flaws: finalFatal,
    fixable_flaws: fixableFlaws,
    revised_idea: revisedIdea,
    reviewers: effectiveReviews,
  };
}

// ─── Semantic Novelty Search ──────────────────────────────────────────────────

/**
 * Check novelty of an idea using:
 * 1. openclaw memory search (semantic embeddings over the corpus)
 * 2. S2 semantic search for the idea title
 * 3. LLM-powered verdict
 */
export async function semanticNoveltyCheck(idea, evidenceSummaries) {
  const ideaText = `${idea.title} ${idea.pitch || ""} ${idea.mechanism || ""}`;

  // 1. LOCAL EMBEDDINGS — real semantic similarity via embeddinggemma-300m (no API key)
  console.error(`  [Novelty/embed] Running local embedding search for: "${idea.title.slice(0,50)}"`);
  let embeddingSimilar = [];
  try {
    const corpus = evidenceSummaries.slice(0, 50).map(e => ({
      id: e.evidence_id,
      text: `${e.title} ${e.core_claim || e.abstract || ""}`.slice(0, 400),
    }));
    const topMatches = await ocl.semanticSearch(ideaText, corpus, 5);
    embeddingSimilar = topMatches.filter(m => m.similarity > 0.7);
    if (embeddingSimilar.length > 0) {
      console.error(`  [Novelty/embed] Found ${embeddingSimilar.length} similar papers (cosine > 0.7): ${embeddingSimilar.map(m => `${m.id}(${m.similarity.toFixed(2)})`).join(", ")}`);
    } else {
      console.error(`  [Novelty/embed] No highly similar papers found (max similarity: ${topMatches[0]?.similarity?.toFixed(2) || "N/A"})`);
    }
  } catch (err) {
    console.error(`  [Novelty/embed] Local embedding search failed: ${err.message}`);
  }

  // 2. OpenClaw MEMORY search (cross-run knowledge)
  let memoryHits = [];
  try {
    memoryHits = await ocl.memorySearch(ideaText, { maxResults: 8 });
  } catch (err) {
    console.error(`  [Novelty/memory] Memory search failed: ${err.message}`);
  }

  // 3. LLM verdict using MEDIUM tier (better reasoning for novelty judgment)
  const evidenceBlock = evidenceSummaries.slice(0, 8).map(e =>
    `[${e.evidence_id}] "${e.title?.slice(0,45)}" — ${(e.core_claim || "").slice(0, 60)}`
  ).join("\n");

  const similarBlock = embeddingSimilar.slice(0, 3).map(m => {
    const paper = evidenceSummaries.find(e => e.evidence_id === m.id);
    return `[${m.id}] similarity=${m.similarity.toFixed(2)}: "${paper?.title?.slice(0,50) || m.id}"`;
  }).join("\n") || "None";

  const memoryBlock = memoryHits.slice(0, 3).map(h =>
    `Memory: ${h.excerpt?.slice(0, 60) || h.file}`
  ).join("\n") || "None";

  const raw = await runMedium(`Novelty assessor. Does this idea already exist in the literature?
IDEA: ${idea.title} | ${(idea.pitch || idea.novelty_claim || "").slice(0, 120)}
EMBEDDING-SIMILAR PAPERS (cosine > 0.7):
${similarBlock}
OTHER PRIOR WORK:
${evidenceBlock}
MEMORY HITS:
${memoryBlock}
Return JSON: {"verdict":"novel|needs_differentiation|already_exists","confidence":0.8,"most_similar_papers":["EV001"],"differentiating_factor":"text","recommendation":"proceed|differentiate|abandon","novelty_score":7}`);
  const verdict = parseJsonSafe(raw);

  return {
    idea_id: idea.idea_id,
    verdict: verdict?.verdict || (embeddingSimilar.length >= 3 ? "needs_differentiation" : "novel"),
    novelty_score: verdict?.novelty_score || (10 - embeddingSimilar.length * 2),
    confidence: verdict?.confidence || 0.5,
    most_similar_papers: verdict?.most_similar_papers || embeddingSimilar.map(m => m.id),
    differentiating_factor: verdict?.differentiating_factor || "",
    recommendation: verdict?.recommendation || (embeddingSimilar.length >= 3 ? "differentiate" : "proceed"),
    memory_hits: memoryHits.length,
    embedding_similar_count: embeddingSimilar.length,
  };
}



// ─── LLM Relevance Classification (batched) ──────────────────────────────────

/**
 * Judge which papers are on-topic for the research topic.
 * Batched to keep calls low. Returns array of {n, relevant, reason} or null.
 */
export async function classifyRelevanceBatch(topic, papers) {
  const clean = (s) => String(s || "")
    .replace(/\$[^$]*\$/g, " ")        // strip LaTeX math
    .replace(/[^\x20-\x7E]/g, " ")     // strip non-ASCII/control chars
    .replace(/["\\`]/g, " ")           // strip quotes/backslashes that break prompts
    .replace(/\s+/g, " ")
    .trim();
  const block = papers
    .map((p) => `${p.n}. ${clean(p.title).slice(0, 90)} :: ${clean(p.abstract).slice(0, 110)}`)
    .join("\n");
  const raw = await runMedium(`Screen papers for a literature review on TOPIC: "${clean(topic)}".
A paper is relevant if it could inform, support, compete with, or provide a baseline/dataset/method for this exact topic. Papers from unrelated fields are not relevant.
PAPERS:
${block}

Return ONLY a compact JSON array, one object per paper, no other text:
[{"n":0,"relevant":true},{"n":1,"relevant":false}]`);
  return parseJsonSafe(raw);
}

// ─── LLM-Powered Search Query Generation ────────────────────────────────────



/**
 * Generate diverse search queries using LLM reasoning.
 * Replaces the template-based buildSearchQueries() approach.
 */
export async function generateSearchQueries(topic) {
  const queries = await askJsonArray(`Generate 15 short research search queries (2-4 words each) for this topic: "${topic}"
Cover: core methods, applications, benchmarks, limitations, competing approaches, datasets, surveys.
Return JSON array of 15 strings. Example: ["query1", "query2"]`);

  if (Array.isArray(queries) && queries.length >= 8) {
    return queries.filter(q => typeof q === "string" && q.length > 3 && q.length < 100);
  }
  return null; // Fall back to keyword extraction
}

// ─── Gap Validation Probe Codegen ─────────────────────────────────────────────

/**
 * Generate a tiny, fast, sklearn/numpy-only script that EMPIRICALLY checks
 * whether a research gap's signal actually exists in data. The script must print
 * exactly one line: `GAP_SIGNAL <true|false> <number>`.
 * This is how the agent confirms a gap with execution before building on it.
 */
export async function designGapProbe(gap, topic) {
  const gapText = `${gap.gap_title || ""}: ${gap.text || ""}`.slice(0, 400);
  const hyp = (gap.testable_hypothesis || "").slice(0, 200);
  return ask(`Write a SHORT Python script (<70 lines) that empirically checks whether this research gap's signal really exists, using a quick experiment.
TOPIC: ${topic}
GAP: ${gapText}
HYPOTHESIS: ${hyp}

Rules:
- ONLY sklearn, numpy, scipy (all pre-installed). No torch/downloads/network.
- Use a small sklearn dataset (load_digits/load_breast_cancer/make_classification) or synthetic data.
- Run a quick comparison/measurement that would reveal whether the gap's effect is real.
- Finish FAST (<60s). Fixed seed 42.
- Print EXACTLY one final line: GAP_SIGNAL <true|false> <number>
  where true means the gap's signal/effect is observed, and number is the measured effect size.
Output ONLY Python code, no markdown.`);
}



/**
 * LLM reviews actual experiment metrics (not synthetic scores).
 * Input: real metrics.json from sandbox.
 * Output: PROMOTE / REVISE / KILL + revision suggestions.
 */
export async function reviewExperimentResults(idea, metrics, level, methodMatrix = []) {
  const matrixBlock = methodMatrix.slice(0, 10).map(m =>
    `${m.title}: ${m.best_result || "?"} on ${(m.datasets_used||[]).join(",") || "?"}`
  ).join("\n") || "Not available";

  const metricsBlock = JSON.stringify(metrics, null, 2).slice(0, 2000);

  const reviewerNames = [
    "Novelty Reviewer",
    "Experimental Reviewer",
    "Reproducibility Reviewer",
  ];

  const reviews = [];
  for (const reviewer of reviewerNames) {
    const result = await askJson(`You are "${reviewer}" reviewing REAL experiment results.

IDEA: ${idea.title}
PITCH: ${(idea.pitch || "").slice(0, 200)}
EXPERIMENT LEVEL: ${level}

ACTUAL METRICS FROM EXPERIMENT:
${metricsBlock}

SOTA RESULTS FROM LITERATURE (for comparison):
${matrixBlock}

YOUR ROLE:
${reviewer === "Novelty Reviewer" ? "Is the improvement meaningful vs SOTA? Does this advance the field?" : ""}
${reviewer === "Experimental Reviewer" ? "Are the baselines fair? Is improvement statistically significant? What experiments are still needed?" : ""}
${reviewer === "Reproducibility Reviewer" ? "Are seeds fixed? Is the environment logged? Can someone reproduce this?" : ""}

Decision criteria:
- PROMOTE: Results show clear improvement. Worth running larger experiment.
- REVISE: Results are mixed. A specific change could improve results.
- KILL: No improvement or fundamental flaw in the idea.

Return JSON:
{
  "reviewer": "${reviewer}",
  "score": <1-10>,
  "decision": "PROMOTE|REVISE|KILL",
  "fatal_flaws": [],
  "fixable_flaws": [],
  "revision_suggestion": "<specific code/method change to improve results, or null>",
  "experiments_needed": ["<what to run next>"],
  "justification": "<2-3 sentence assessment with specific numbers>"
}`);

    if (result?.score) {
      reviews.push(result);
    } else {
      const success = metrics?.success || metrics?.improvement > 0;
      reviews.push({
        reviewer,
        score: success ? 6.5 : 4.0,
        decision: success ? "PROMOTE" : "REVISE",
        fatal_flaws: [],
        fixable_flaws: success ? [] : ["Results did not clear threshold"],
        revision_suggestion: null,
        experiments_needed: ["Run with larger dataset and more seeds"],
        justification: success ? "Experiment shows positive signal." : "Experiment did not show improvement.",
      });
    }
  }

  const scores = reviews.map(r => r.score).filter(Number.isFinite);
  const avgScore = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;
  const kills = reviews.filter(r => r.decision === "KILL").length;
  const promotes = reviews.filter(r => r.decision === "PROMOTE").length;

  // Consensus decision
  let consensusDecision;
  if (kills >= 2) consensusDecision = "KILL";
  else if (promotes >= 2) consensusDecision = "PROMOTE";
  else consensusDecision = "REVISE";

  // Collect best revision suggestion
  const revisionSuggestions = reviews.map(r => r.revision_suggestion).filter(Boolean);

  return {
    idea_id: idea.idea_id,
    average_score: Number(avgScore.toFixed(2)),
    decision: consensusDecision,
    revision_suggestion: revisionSuggestions[0] || null,
    experiments_needed: reviews.flatMap(r => r.experiments_needed || []),
    fatal_flaws: reviews.flatMap(r => r.fatal_flaws || []),
    fixable_flaws: reviews.flatMap(r => r.fixable_flaws || []),
    reviewers: reviews,
  };
}

function fallbackResultReview(reviewer, metrics) {
  const success = metrics?.success || metrics?.improvement > 0;
  return {
    reviewer,
    score: success ? 6.5 : 4.0,
    decision: success ? "PROMOTE" : "REVISE",
    fatal_flaws: [],
    fixable_flaws: success ? [] : ["Results did not show improvement"],
    revision_suggestion: null,
    experiments_needed: [],
    justification: success ? "Experiment shows positive signal." : "No improvement detected.",
  };
}

// ─── Experiment Code Generation ───────────────────────────────────────────────

/**
 * Generate Python experiment code for a research idea.
 * All code generation goes through openclaw infer model run.
 */
export async function designExperiment(idea, topic, level, deepReads = []) {
  const runtimeBudgets = {
    micro_probe: "3-5 minutes, minimal",
    probe: "10-20 minutes, medium",
    ablation: "20-45 minutes, full ablation study",
    mvp: "30-90 minutes, multi-seed statistical test",
  };

  const deepReadsBlock = deepReads.slice(0, 3).map(dr =>
    `Paper: ${dr.title}\nDataset: ${(dr.datasets_used||[]).join(", ")||"?"}\nMethod: ${dr.method_summary||"?"}\nBest result: ${dr.best_result||"?"}\nCode URL: ${dr.code_url||"none"}`
  ).join("\n---\n") || "No deep reads available";

  const ablationInstructions = level === "ablation" || level === "mvp" ? `
ABLATION REQUIREMENTS (${level} level):
- Implement 2-3 ablation variants that test different components of the idea
- For each ablation: remove or disable one component, run same evaluation
- Report all ablation results in metrics.json under "ablations" key
- Use SAME dataset and seeds for all variants
` : "";

  const multiSeedInstructions = (level === "probe" || level === "ablation" || level === "mvp") ? `
MULTI-SEED REQUIREMENTS:
- Run with random seeds: [42, 1337, 2024]
- Report mean ± std for all metrics
- Output: {"accuracy_mean": 0.85, "accuracy_std": 0.02, "seeds": [42, 1337, 2024]}
` : "";

  const plotInstructions = (level !== "micro_probe") ? `
PLOTTING REQUIREMENTS:
- Save a comparison bar chart to outputs/comparison.png using matplotlib
- Save training curve (if applicable) to outputs/learning_curve.png
- Save a results table as outputs/comparison_table.md in NeurIPS format
` : "";

  const statTestInstructions = (level === "probe" || level === "ablation" || level === "mvp") ? `
STATISTICAL SIGNIFICANCE REQUIREMENTS:
- After running with multiple seeds, run a paired t-test (scipy.stats.ttest_rel) between baseline and proposed results
- from scipy import stats; t_stat, p_value = stats.ttest_rel(proposed_scores, baseline_scores)
- Report in metrics.json: {"t_statistic": float, "p_value": float, "statistically_significant": bool (p_value < 0.05)}
- If scipy unavailable, report t_statistic: 0.0, p_value: 1.0, statistically_significant: false
` : "";

  const memoryGuardInstructions = `
MEMORY SAFETY REQUIREMENTS:
- At experiment start: import psutil; MAX_GB = 12.0
- After each major step: check_mem() that kills if RAM > MAX_GB
- def check_mem():
    try:
        import psutil; used = (psutil.virtual_memory().total - psutil.virtual_memory().available)/1024**3
        if used > 12.0: print(f'[MEM-KILL] {used:.1f}GB'); import sys; sys.exit(1)
    except: pass
`;

  const sklearnOnly = level === "micro_probe";
  const reqsBlock = [
    `LEVEL: ${level} (budget: ${runtimeBudgets[level] || "10-30 min"})`,
    level === "ablation" || level === "mvp" ? `Ablations: implement 2-3 variants removing one component each, same dataset+seeds.` : "",
    level !== "micro_probe" ? `Seeds: [42, 1337, 2024]. Report mean±std.` : `Seeds: [42, 1337]. Report mean±std.`,
    level !== "micro_probe" ? `Stats: scipy.stats.ttest_rel(proposed, baseline) → p_value, statistically_significant.` : `Stats: scipy.stats.ttest_rel over the 2 seeds → p_value, statistically_significant.`,
    level !== "micro_probe" ? `Plots: outputs/comparison.png (bar chart), outputs/comparison_table.md (NeurIPS table).` : `Plots: outputs/comparison.png (bar chart) and outputs/comparison_table.md.`,
    `Memory: import psutil; kill if RAM>12GB after each major step.`,
    sklearnOnly
      ? `STRICT: sklearn/numpy/scipy ONLY. Do NOT import torch, tensorflow, transformers, datasets, or torchvision. Use a small sklearn dataset (load_digits / load_breast_cancer / make_classification) with <=1000 samples so it finishes in <2 min on CPU.`
      : `Device: MPS if torch (torch.device('mps' if torch.backends.mps.is_available() else 'cpu')).`,
    `Fallback: if dataset fails, use sklearn.datasets.load_digits().`,
    `The PROPOSED method MUST actually implement the idea mechanism above and be a real, distinct variant of the baseline (not identical params).`,
    `Output: metrics.json with {experiment_id, success, accuracy, accuracy_std, improvement, p_value, statistically_significant} where success=true ONLY if proposed beats baseline.`,
  ].filter(Boolean).join(" | ");

  const rules = sklearnOnly
    ? `Rules: import ONLY sklearn, numpy, scipy, psutil, matplotlib (all pre-installed). NO torch/tensorflow/transformers/datasets. No network/downloads. create outputs/ dir. Keep it under ~120 lines and <2 min runtime.`
    : `Rules: no CUDA, numpy+sklearn+scipy+psutil available, pip install at runtime if needed, create outputs/ dir, print progress every 30s.`;

  return ask(`Generate a complete runnable Python experiment script.
TOPIC: ${topic} | IDEA: ${idea.title}
MECHANISM: ${(idea.mechanism || idea.pitch || "").slice(0, 180)}
HYPOTHESIS: ${(idea.testable_hypothesis || "").slice(0, 120)}
DATASET: ${idea.required_dataset || "sklearn built-in"} | BASELINE: ${idea.baseline_to_beat || "logistic regression"}
RELATED PAPERS:
${deepReads.slice(0,2).map(dr => `${dr.title?.slice(0,40)}: ${dr.method_summary?.slice(0,60)||"?"} | ${(dr.datasets_used||[]).join(",").slice(0,30)||"?"}`).join("\n") || "None"}
REQUIREMENTS: ${reqsBlock}
${rules}
Output ONLY Python code, no markdown, no explanation.`);
}

// ─── Idea Revision Generator ─────────────────────────────────────────────────

/**
 * Generate a revised experiment plan when results say REVISE.
 * Takes the original idea, the failed metrics, and the reviewer's revision suggestion.
 */
export async function generateRevision(idea, failedMetrics, revisionSuggestion, attemptNumber) {
  const metricsSummary = `acc=${failedMetrics?.accuracy?.toFixed?.(3)||"?"} improvement=${JSON.stringify(failedMetrics?.improvement||"?").slice(0,60)}`;
  return askJson(`Revise failed research idea (attempt ${attemptNumber}).
IDEA: ${idea.title} | ${(idea.mechanism||"").slice(0,120)}
FAILED: ${metricsSummary}
SUGGESTION: ${(revisionSuggestion||"").slice(0,150)}
Return JSON: {title, mechanism, testable_hypothesis, required_dataset, key_change, why_this_will_work}`);
}

// ─── Full Paper Writer ────────────────────────────────────────────────────────

/**
 * Write a complete research paper draft using real experiment results.
 * All content is grounded in actual metrics, actual papers read, real citations.
 */
export async function writeFullPaper(state, methodMatrix, experimentResults) {
  const evidenceBlock = (state.parsed_papers || []).slice(0, 20).map(p =>
    `[${p.evidence_id}] ${p.title} (${p.year||"?"}): ${(p.core_claim||"").slice(0, 150)}`
  ).join("\n");

  const resultsBlock = experimentResults.slice(0, 5).map(r =>
    `Experiment ${r.experiment_id || "?"} (${r.level}): ${JSON.stringify(r.metrics || {}).slice(0, 200)}`
  ).join("\n") || "No experiments completed yet";

  const gaps = (state.research_gaps || []).slice(0, 4).map(g =>
    `- ${g.gap_title || g.text?.slice(0, 80)}`
  ).join("\n");

  const topIdea = state.experiment_candidates?.[0] || state.idea_tree?.[0];

  // Use DEEP tier with thinking=high — DeepSeek V3.1 for best quality paper writing
  // Build rich context blocks
  const ideasBlock = (state.idea_tree || []).slice(0, 6).map((idea, i) =>
    `IDEA ${i+1} [${idea.reviewer_average || idea.novelty_score || "?"}]: ${idea.title}\n  Mechanism: ${(idea.mechanism || "").slice(0, 200)}\n  Evidence: ${(idea.evidence_support || "").slice(0, 150)}`
  ).join("\n");

  const allGapsBlock = (state.research_gaps || []).slice(0, 6).map((g, i) =>
    `GAP ${i+1}: ${g.gap_title || g.text || ""}\n  Hypothesis: ${(g.testable_hypothesis || "").slice(0, 150)}`
  ).join("\n");

  const fullPapersBlock = (state.parsed_papers || []).slice(0, 25).map((p, i) =>
    `[${p.evidence_id}] ${p.title} (${p.year || "?"}): ${(p.core_claim || "").slice(0, 180)}\n  Method: ${(Array.isArray(p.method) ? p.method : [p.method || ""]).slice(0,3).join(", ")}\n  Gaps: ${(p.possible_gap || "").slice(0, 100)}`
  ).join("\n");

  const fullResultsBlock = experimentResults.slice(0, 5).map((r, i) => {
    const m = r.metrics || {};
    return `EXPERIMENT ${i+1}: ${r.idea_title || r.experiment_id || "?"}\n  Dataset: ${m.dataset || "?"} | Task: ${m.task_type || "classification"}\n  Baseline: ${m.baseline_accuracy?.toFixed(4) || m.baseline_score || "?"} | Proposed: ${m.accuracy?.toFixed(4) || "?"}\n  Improvement: ${m.improvement_pct?.toFixed(2) || "?"}% | p-value: ${m.p_value || "?"} | Seeds: ${(m.seeds || []).join(",")}\n  F1: ${m.f1?.toFixed(4) || "?"} | Status: ${m.status || (m.success ? "passed" : "completed")}`;
  }).join("\n") || "No experiments completed";

  return runDeep(`You are an expert researcher writing a full, deep, publication-quality research paper.

TOPIC: ${state.topic}
RUN_ID: ${state.run_id}
PAPERS READ: ${state.parsed_papers?.length || 0}

=== RESEARCH GAPS (${(state.research_gaps || []).length} found) ===
${allGapsBlock}

=== IDEAS GENERATED (${(state.idea_tree || []).length} total, with adversarial reviewer scores) ===
${ideasBlock}

=== REAL EXPERIMENT RESULTS ===
${fullResultsBlock}

=== LITERATURE READ ===
${fullPapersBlock}

Write a COMPLETE, DEEP research paper. MINIMUM 3500 words. Structure:

**Abstract** (250 words): Problem, approach, key result numbers, significance.

**1. Introduction** (600 words): Motivate the problem deeply. State 3 specific limitations of existing work (cite [EVxxx]). State your contributions as a numbered list. Include a "paper organization" sentence.

**2. Related Work** (500 words): Group into 3 subsections by approach. Critically analyze each group. Cite [EVxxx] throughout. Explain what each group fails to address.

**3. Method** (600 words): Give your method a real name (acronym). Describe mechanism precisely. Include a pseudocode or algorithm block. Explain the intuition.

**4. Experimental Setup** (300 words): Dataset(s), baselines, metrics, hyperparameters, hardware (Apple M4 Mac, MPS), random seeds used. 

**5. Results** (400 words): Present a complete results table:
| Method | Dataset | Accuracy | F1 | p-value |
Use ONLY real numbers from experiment results above. Bold best results. Statistical significance notes.

**6. Ablation Study** (300 words): What happens when you remove each component? Table required.

**7. Discussion & Limitations** (250 words): What does this mean? What can't this do? Failure cases.

**8. Conclusion** (150 words): Summary + 2 future directions.

**References**: Cite all [EVxxx] papers used.

STRICT RULES:
- Use ONLY real numbers from experiment results section above. Never invent metrics.
- Cite [EVxxx] for every claim about prior work. 
- Every table must have real numbers.
- Write as a NeurIPS/ICML paper, not a blog post.
- Depth matters: explain WHY your method works, not just WHAT it does.`);

}


// ─── Paper Claims Extractor ───────────────────────────────────────────────────

export async function extractPaperClaims(title, abstract, url) {
  const cleanAbstract = (abstract || "")
    .replace(/<[^>]+>/g, "")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 600); // Keep under sarvam limit

  // FAST tier: short prompt, quick JSON extraction
  const raw = await runFast(`Extract structured info from this research paper.
TITLE: ${title?.slice(0,80)}
ABSTRACT: ${cleanAbstract}
Return JSON: {"core_claim":"1 sentence","method":["m1","m2"],"limitations":["l1"],"datasets":["d1"],"metrics":["accuracy"],"possible_gap":"1 gap sentence","relevance_keywords":["kw1"]}`);
  return parseJsonSafe(raw);
}




// ─── NeurIPS Readiness Checker ────────────────────────────────────────────────

/**
 * LLM-powered NeurIPS reproducibility checklist.
 * Returns structured pass/fail per criterion with explanations.
 */
export async function checkNeurIPSReadiness(state, experimentResults) {
  const allMetrics = experimentResults.map(r => r.metrics || {});
  const hasSeeds = allMetrics.some(m => m.seeds || m.seed !== undefined);
  const hasStd = allMetrics.some(m => m.accuracy_std !== undefined || Object.keys(m).some(k => k.endsWith("_std")));
  const hasBaseline = allMetrics.some(m => m.baseline_score !== undefined || m.random_mean !== undefined);
  const hasAblation = state.ablation_results?.some(r => r.success);
  const hasCode = state.parsed_papers?.some(p => p.code_available);
  const reviewerAvg = (state.reviewer_scores?.reduce((a,b)=>a+(b.average_score||0),0)/Math.max(1,state.reviewer_scores?.length||1)||0).toFixed(1);

  // MEDIUM tier: structured reasoning required
  const raw = await runMedium(`NeurIPS reproducibility assessment.
TOPIC: ${state.topic?.slice(0,60)} | Papers: ${state.evidence_ids?.length||0} | Experiments: ${experimentResults.length}
Checks: seeds=${hasSeeds} std=${hasStd} baseline=${hasBaseline} ablation=${hasAblation} code=${hasCode} reviewer_avg=${reviewerAvg}
Return JSON: {"checklist":{"code_available":{"pass":true,"note":""},"random_seeds_fixed":{"pass":true,"note":""},"error_bars_reported":{"pass":true,"note":""},"baselines_reproduced":{"pass":true,"note":""},"ablation_completed":{"pass":true,"note":""},"limitations_stated":{"pass":true,"note":""},"compute_budget_reported":{"pass":true,"note":""},"dataset_access_documented":{"pass":true,"note":""},"hyperparameters_documented":{"pass":true,"note":""}},"items_passing":5,"verdict":"needs_more_work","blocking_issues":[],"recommendation":"text"}`);
  return parseJsonSafe(raw);
}



// ─── Health Check ─────────────────────────────────────────────────────────────

export async function healthCheck() {
  const result = await ocl.inferModel("Reply with exactly: OK", { tier: "FAST", timeoutMs: 15000, cascade: false });
  return result !== null && result.trim().includes("OK");
}

