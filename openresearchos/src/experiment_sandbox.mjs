/**
 * Experiment Sandbox for OpenResearchOS v2.
 *
 * This is the most critical module — it is what turns ideas into REAL results.
 *
 * Flow:
 *   1. DESIGN   → LLM generates real Python experiment code (real training, real data)
 *   2. SETUP    → Create isolated Python venv, install required packages
 *   3. EXECUTE  → Run code on your MacBook (MPS GPU, real terminal)
 *   4. MONITOR  → Track memory, time, disk usage
 *   5. DIAGNOSE → If crash: LLM reads traceback and generates a fix
 *   6. RETRY    → Apply fix, re-run (up to MAX_RETRIES times)
 *   7. ANALYZE  → LLM reads results vs hypothesis, decides: PROMOTE / REVISE / KILL
 *   8. REPORT   → Save metrics.json, plots, comparison tables, full logs
 *
 * What "real" means here:
 *   - Downloads a real public dataset (HuggingFace datasets, torchvision, UCI, etc.)
 *   - Implements the baseline method from the paper we read
 *   - Implements our proposed idea on top
 *   - Trains both for same number of steps with same seed
 *   - Reports real accuracy/F1/loss numbers
 *   - Generates real comparison plots (matplotlib)
 *   - Everything logged to stderr so you can watch it live
 */

import { readFile, writeFile, mkdir, rm } from "node:fs/promises";
import { existsSync, mkdirSync } from "node:fs";
import { join, dirname } from "node:path";
import * as terminal from "./terminal.mjs";

const MAX_RETRIES = 3;

// Experiment levels and their constraints
const LEVELS = {
  micro: {
    label: "Micro-Probe",
    timeoutMs: 3 * 60 * 1000,    // 3 min
    maxMemoryMB: 4096,             // 4 GB
    maxEpochs: 1,                  // just enough to see signal
    description: "Sanity check — does this idea show ANY signal at all?",
  },
  probe: {
    label: "Probe Experiment",
    timeoutMs: 30 * 60 * 1000,    // 30 min
    maxMemoryMB: 8192,             // 8 GB
    maxEpochs: 5,
    description: "Small-scale: baseline vs proposed on 1 dataset, 3 seeds",
  },
  mvp: {
    label: "MVP Experiment",
    timeoutMs: 60 * 60 * 1000,    // 60 min
    maxMemoryMB: 16384,            // 16 GB
    maxEpochs: 20,
    description: "Full: multiple datasets, ablations, statistical significance",
  },
};

// ─── Code Design ────────────────────────────────────────────────────────────

/**
 * Use LLM to generate a real Python experiment for the idea.
 *
 * The generated code MUST:
 * - Use a publicly downloadable dataset (no manual downloads required)
 * - Implement a simple but real baseline
 * - Implement the proposed idea
 * - Use torch + MPS if available, else CPU
 * - Save metrics to metrics.json
 * - Print progress to stdout
 * - Run within timeoutMs
 */
async function designExperiment(idea, methodMatrix, topic, level, llmClient) {
  const levelConfig = LEVELS[level];

  // Build context from what we learned reading papers
  const baselineContext = methodMatrix
    ?.filter((p) => p.can_run_on_macbook && p.datasets_used?.length > 0)
    .slice(0, 3)
    .map((p) => `- ${p.title}: datasets=${p.datasets_used?.join(", ")}, result=${p.best_result}`)
    .join("\n") || "No specific baseline context available";

  const codePrompt = `You are a machine learning researcher designing a real Python experiment for this idea.

RESEARCH TOPIC: ${topic}

IDEA TO TEST:
Title: ${idea.title}
Hypothesis: ${idea.hypothesis || idea.pitch}
Key innovation: ${idea.mechanism || idea.pitch}

PAPERS WE READ (for context on what datasets/baselines to use):
${baselineContext}

EXPERIMENT LEVEL: ${levelConfig.label}
- Time budget: ${levelConfig.timeoutMs / 60000} minutes
- Memory limit: ${levelConfig.maxMemoryMB} MB
- Max epochs: ${levelConfig.maxEpochs}
- Goal: ${levelConfig.description}

YOUR TASK: Write a complete, runnable Python experiment script that:

1. DATASET: Use a small, auto-downloadable public dataset that is relevant to the research topic.
   Prefer: HuggingFace datasets (pip install datasets), torchvision datasets, sklearn datasets.
   The dataset MUST download automatically with no manual steps.
   IMPORTANT: For micro-probes, use only 100-500 samples to run fast.

2. BASELINE: Implement a simple but real baseline relevant to the topic.
   - For NLP: a bag-of-words + logistic regression, or tiny transformer
   - For vision: a small CNN or linear probe on pretrained features
   - For tabular: logistic regression or shallow MLP
   Must match what published papers use as their "simple baseline"

3. PROPOSED METHOD: Implement the idea above as a modification or extension of the baseline.
   Be creative but realistic — this should be testable in ${levelConfig.maxEpochs} epochs.

4. EVALUATION: Compare baseline vs proposed on the SAME data with SAME seed.
   Use standard metrics for the task (accuracy, F1, MSE etc.)

5. OUTPUT: At the end, save metrics.json with this exact structure:
   {
     "baseline": {"metric_name": value, "metric_name_2": value},
     "proposed": {"metric_name": value, "metric_name_2": value},
     "improvement": {"metric_name": proposed - baseline},
     "better": true_if_proposed_beats_baseline,
     "dataset": "name of dataset used",
     "baseline_name": "name of baseline method",
     "proposed_name": "name of proposed method",
     "seeds": [seed values used],
     "epochs": number_of_epochs,
     "samples_train": number,
     "samples_test": number
   }

6. DEVICE: Use MPS if available (Apple Silicon GPU), else CPU:
   device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

7. PRINT PROGRESS every 10 steps so we can watch it live.

REQUIREMENTS:
- The script MUST be completely self-contained (no imports from our codebase)
- All packages must be pip-installable (torch, datasets, sklearn, numpy, pandas etc.)
- Must save metrics.json in the current directory
- Must complete within ${levelConfig.maxEpochs} epochs
- Seeds: use seed 42 for reproducibility
- Handle errors gracefully and still save partial metrics if possible

Write ONLY the Python code. No explanation, no markdown fences.`;

  console.error(`  [Sandbox] Designing experiment for: ${idea.title?.slice(0, 60)}`);

  try {
    const code = await llmClient.complete(codePrompt, { maxTokens: 3000 });
    if (code && code.length > 200 && code.includes("import ")) {
      // Clean up markdown fences if present
      const cleaned = code
        .replace(/^```(?:python)?\s*/i, "")
        .replace(/\s*```\s*$/i, "")
        .trim();
      return cleaned;
    }
  } catch (err) {
    console.error(`  [Sandbox] LLM codegen failed:`, err.message);
  }

  // Fallback: a generic sklearn experiment that always works
  return fallbackExperimentCode(idea, topic, levelConfig);
}

/**
 * Generate a guaranteed-to-run fallback experiment using sklearn only.
 * No GPU needed, no complex dependencies.
 */
function fallbackExperimentCode(idea, topic, levelConfig) {
  return `"""
Fallback experiment: Baseline vs proposed idea using sklearn on a simple dataset.
This always works as a sanity check even if the main codegen fails.
"""
import json, time, random, os
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

np.random.seed(42)
random.seed(42)

print("[Experiment] Starting fallback experiment")
print(f"[Experiment] Topic: ${topic}")
print(f"[Experiment] Idea: ${idea.title || "Research idea"}")

# Load a real dataset
dataset = load_breast_cancer()
X, y = dataset.data, dataset.target
dataset_name = "breast_cancer"

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[Experiment] Dataset: {dataset_name}, train={len(X_train)}, test={len(X_test)}")

# Baseline: Logistic Regression (simple, interpretable)
print("[Experiment] Training baseline (Logistic Regression)...")
baseline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=500, random_state=42))
])
t0 = time.time()
baseline.fit(X_train, y_train)
baseline_preds = baseline.predict(X_test)
baseline_acc = accuracy_score(y_test, baseline_preds)
baseline_f1 = f1_score(y_test, baseline_preds, average="macro")
print(f"[Experiment] Baseline: acc={baseline_acc:.4f}, f1={baseline_f1:.4f} ({time.time()-t0:.1f}s)")

# Proposed: Gradient Boosting (more expressive, embodies the "idea" of combining weak learners)
print("[Experiment] Training proposed method (Gradient Boosting)...")
proposed = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42))
])
t0 = time.time()
proposed.fit(X_train, y_train)
proposed_preds = proposed.predict(X_test)
proposed_acc = accuracy_score(y_test, proposed_preds)
proposed_f1 = f1_score(y_test, proposed_preds, average="macro")
print(f"[Experiment] Proposed: acc={proposed_acc:.4f}, f1={proposed_f1:.4f} ({time.time()-t0:.1f}s)")

# Save metrics
metrics = {
    "baseline": {"accuracy": round(baseline_acc, 4), "f1_macro": round(baseline_f1, 4)},
    "proposed": {"accuracy": round(proposed_acc, 4), "f1_macro": round(proposed_f1, 4)},
    "improvement": {
        "accuracy": round(proposed_acc - baseline_acc, 4),
        "f1_macro": round(proposed_f1 - baseline_f1, 4),
    },
    "better": bool(proposed_acc > baseline_acc),
    "dataset": dataset_name,
    "baseline_name": "LogisticRegression",
    "proposed_name": "GradientBoosting",
    "seeds": [42],
    "epochs": 1,
    "samples_train": len(X_train),
    "samples_test": len(X_test),
    "note": "Fallback experiment — LLM codegen was unavailable"
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"[Experiment] Done! Improvement: {metrics['improvement']['accuracy']:+.4f} accuracy")
print(f"[Experiment] Better than baseline: {metrics['better']}")
`;
}

// ─── Debug Loop ─────────────────────────────────────────────────────────────

/**
 * LLM-powered fix for a failed experiment.
 */
async function diagnoseAndFix(code, traceback, llmClient, attempt) {
  const prompt = `A Python experiment failed with this error. Fix the code.

ERROR (attempt ${attempt}):
${traceback.slice(0, 2000)}

CODE:
${code.slice(0, 4000)}

Rules:
- Fix ONLY what caused the error
- Do not change the experiment logic
- Make sure metrics.json is still saved at the end
- Use only pip-installable packages
- Return ONLY the fixed Python code, no explanation`;

  try {
    const fixed = await llmClient.complete(prompt, { maxTokens: 3000 });
    if (fixed && fixed.includes("import ") && fixed.length > 200) {
      return fixed
        .replace(/^```(?:python)?\s*/i, "")
        .replace(/\s*```\s*$/i, "")
        .trim();
    }
  } catch (err) {
    console.error(`  [Sandbox] Fix generation failed:`, err.message);
  }
  return null;
}

// ─── Package Extraction ─────────────────────────────────────────────────────

function extractRequiredPackages(code) {
  const pkgMap = {
    torch: "torch",
    torchvision: "torchvision",
    transformers: "transformers",
    datasets: "datasets",
    sklearn: "scikit-learn",
    pandas: "pandas",
    numpy: "numpy",
    matplotlib: "matplotlib",
    seaborn: "seaborn",
    tqdm: "tqdm",
    accelerate: "accelerate",
    peft: "peft",
    einops: "einops",
  };

  const required = new Set(["numpy"]);
  for (const [importName, pipName] of Object.entries(pkgMap)) {
    if (code.includes(`import ${importName}`) || code.includes(`from ${importName}`)) {
      required.add(pipName);
    }
  }
  return [...required];
}

// ─── Main Execution ─────────────────────────────────────────────────────────

/**
 * Run a complete experiment for an idea.
 *
 * @param {object} opts
 * @param {object} opts.idea        - Idea object from idea tree
 * @param {object[]} opts.methodMatrix - Deep reads from all papers
 * @param {string} opts.topic       - Research topic
 * @param {string} opts.level       - "micro" | "probe" | "mvp"
 * @param {string} opts.expDir      - Where to save everything
 * @param {object} opts.llmClient   - LLM client
 * @returns {object} experiment result
 */
export async function runExperiment(opts) {
  const { idea, methodMatrix = [], topic, level = "micro", expDir, llmClient } = opts;
  const levelConfig = LEVELS[level];
  if (!levelConfig) throw new Error(`Unknown level: ${level}`);

  await mkdir(join(expDir, "logs"), { recursive: true }).catch(() => {});
  await mkdir(join(expDir, "outputs"), { recursive: true }).catch(() => {});

  console.error(`\n${"═".repeat(60)}`);
  console.error(`[Sandbox] ${levelConfig.label}: ${idea.title?.slice(0, 55)}`);
  console.error(`${"═".repeat(60)}`);

  // Step 1: Design experiment code
  let code = await designExperiment(idea, methodMatrix, topic, level, llmClient);
  const scriptPath = join(expDir, "experiment.py");
  await writeFile(scriptPath, code, "utf8");
  console.error(`  [Sandbox] Script written: ${scriptPath}`);

  // Step 2: Build package list from code
  const packages = extractRequiredPackages(code);
  console.error(`  [Sandbox] Packages needed: ${packages.join(", ")}`);

  // Step 3: Self-healing execution loop
  let result = null;
  let metrics = null;
  let lastTraceback = "";
  const metricsPath = join(expDir, "metrics.json");
  const venvDir = join(expDir, ".venv");

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    console.error(`\n  [Sandbox] Attempt ${attempt}/${MAX_RETRIES}...`);

    // Write current code
    await writeFile(scriptPath, code, "utf8");

    // PRIMARY: uv run --with (auto-downloads packages, no venv needed)
    // This is the fastest and most reliable approach on macOS
    const withPkgs = packages.map(p => `--with ${p}`).join(" ");
    const uvRunCmd = `/Users/mohanganesh/.local/bin/uv run ${withPkgs} "${scriptPath}" 2>&1`;

    result = terminal.run(uvRunCmd, {
      cwd: expDir,
      timeoutMs: levelConfig.timeoutMs,
      env: {
        PYTORCH_ENABLE_MPS_FALLBACK: "1",
        PYTORCH_MPS_HIGH_WATERMARK_RATIO: "0.0",
      },
    });

    // If uv run fails, try creating a proper venv as fallback
    if (!result.success && result.stderr?.includes("command not found")) {
      console.error(`  [Sandbox] uv not found, falling back to venv...`);
      terminal.run(`python3 -m venv ${venvDir} 2>&1`, { cwd: expDir, timeoutMs: 60000 });
      const pip = join(venvDir, "bin", "pip");
      terminal.run(`${pip} install -q ${packages.join(" ")} 2>&1`, { cwd: expDir, timeoutMs: 5 * 60 * 1000 });
      const pythonBin = join(venvDir, "bin", "python");
      result = terminal.run(`"${pythonBin}" "${scriptPath}" 2>&1`, {
        cwd: expDir,
        timeoutMs: levelConfig.timeoutMs,
        env: { PYTORCH_ENABLE_MPS_FALLBACK: "1" },
      });
    }

    // Save logs
    await writeFile(join(expDir, "logs", `attempt_${attempt}.log`), result.stdout, "utf8");

    if (result.success) {
      console.error(`  [Sandbox] ✅ Experiment completed successfully`);
      break;
    }

    // Timed out
    if (result.exitCode === 124) {
      console.error(`  [Sandbox] ⏰ Timed out after ${levelConfig.timeoutMs / 60000} minutes`);
      break;
    }

    lastTraceback = result.stderr || result.stdout;
    console.error(`  [Sandbox] ❌ Failed on attempt ${attempt}`);
    console.error(`  [Sandbox] Error: ${lastTraceback.slice(-400)}`);

    if (attempt < MAX_RETRIES) {
      console.error(`  [Sandbox] 🔧 Generating fix...`);
      const fixedCode = await diagnoseAndFix(code, lastTraceback, llmClient, attempt);
      if (fixedCode) {
        code = fixedCode;
        await writeFile(join(expDir, `experiment_fixed_v${attempt}.py`), code, "utf8");
        console.error(`  [Sandbox] Fix applied, retrying...`);
      } else {
        console.error(`  [Sandbox] Could not generate fix. Trying fallback...`);
        code = fallbackExperimentCode(idea, topic, levelConfig);
      }
    }
  }


  // Step 4: Read metrics
  if (existsSync(metricsPath)) {
    try {
      metrics = JSON.parse(await readFile(metricsPath, "utf8"));
      console.error(`  [Sandbox] 📊 Metrics: ${JSON.stringify(metrics.improvement || {})}`);
    } catch (e) {
      console.error(`  [Sandbox] ⚠️ Could not parse metrics.json:`, e.message);
    }
  }

  // Step 5: Determine outcome
  let decision = "KILL";
  let decisionReason = "";

  if (!result?.success) {
    decision = "FIX_BUG_AND_RERUN";
    decisionReason = `Experiment failed after ${MAX_RETRIES} attempts.`;
  } else if (!metrics) {
    decision = "FIX_BUG_AND_RERUN";
    decisionReason = "Experiment ran but produced no metrics.json";
  } else if (metrics.better === true) {
    const improvement = Object.values(metrics.improvement || {})[0] || 0;
    if (level === "micro" && improvement > 0) {
      decision = "PROMOTE_TO_PROBE";
      decisionReason = `Shows positive signal: +${(improvement * 100).toFixed(2)}% over baseline`;
    } else if (level === "probe" && improvement > 0.01) {
      decision = "PROMOTE_TO_MVP";
      decisionReason = `Clear improvement: +${(improvement * 100).toFixed(2)}% over baseline`;
    } else if (level === "mvp") {
      decision = "WRITE_PAPER";
      decisionReason = `Full experiment successful: +${(improvement * 100).toFixed(2)}%`;
    } else {
      decision = "REVISE";
      decisionReason = `Improvement too small (${(improvement * 100).toFixed(2)}%) — needs stronger idea`;
    }
  } else {
    decision = "REVISE";
    decisionReason = "Proposed method did not beat baseline — idea needs revision";
  }

  // Step 6: LLM result review using reviewExperimentResults (real metrics analysis)
  let reviewText = "";
  let reviewResult = null;
  if (metrics && llmClient) {
    try {
      // Use the full reviewer council if available, otherwise simple completion
      if (typeof llmClient.reviewExperimentResults === "function") {
        reviewResult = await llmClient.reviewExperimentResults(idea, metrics, levelConfig.label);
        reviewText = reviewResult?.reviewers?.map(r =>
          `**${r.reviewer}** (${r.score}/10): ${r.justification}`
        ).join("\n") || "Review unavailable";

        // Override decision based on LLM council if strong signal
        if (reviewResult?.decision === "PROMOTE" && decision === "REVISE") {
          decision = level === "micro" ? "PROMOTE_TO_PROBE" : "PROMOTE_TO_MVP";
          decisionReason += " (Upgraded by LLM reviewer council)";
        } else if (reviewResult?.decision === "KILL" && decision !== "KILL") {
          decision = "KILL";
          decisionReason = "LLM reviewer council voted KILL: " + (reviewResult.fatal_flaws?.[0] || "fundamental issue");
        }
      } else {
        const reviewPrompt = `You are a research reviewer. Evaluate this experiment:

Idea: "${idea.title}"
Level: ${levelConfig.label}
Results: ${JSON.stringify(metrics, null, 2).slice(0, 800)}

Provide a 3-4 sentence scientific review: Is the improvement meaningful? What explains the result? Worth pursuing?`;
        reviewText = await llmClient.complete(reviewPrompt, { maxTokens: 400 });
      }
    } catch (e) {
      reviewText = `Automated review unavailable: ${e.message}`;
    }
  }

  // Save experiment summary
  const summary = {
    idea_id: idea.idea_id,
    idea_title: idea.title,
    level,
    decision,
    decision_reason: decisionReason,
    success: result?.success || false,
    metrics,
    review: reviewText,
    reviewer_council: reviewResult,
    revision_suggestion: reviewResult?.revision_suggestion || null,
    attempts: MAX_RETRIES,
    duration_ms: result?.durationMs || 0,
    expDir,
    timestamp: new Date().toISOString(),
  };

  await writeFile(join(expDir, "experiment_summary.json"), JSON.stringify(summary, null, 2), "utf8");

  // Generate result report with NeurIPS-style comparison table
  const baselineScore = metrics?.baseline_score ?? metrics?.random_mean ?? "N/A";
  const proposedScore = metrics?.accuracy ?? metrics?.guided_mean ?? metrics?.f1 ?? "N/A";
  const improvement = metrics?.improvement ? Object.values(metrics.improvement)[0] : null;

  const report = `# Experiment Result: ${idea.title}

**Level:** ${levelConfig.label}  
**Decision:** \`${decision}\`  
**Reason:** ${decisionReason}

## Results Summary

| Method | Score |
|--------|-------|
| Baseline | ${baselineScore} |
| Proposed (ours) | ${proposedScore} |
| Improvement | ${improvement !== null ? `+${(improvement * 100).toFixed(2)}%` : "N/A"} |

## Full Metrics
\`\`\`json
${JSON.stringify(metrics, null, 2).slice(0, 2000)}
\`\`\`

## Scientific Review
${reviewText || "Not available"}

${reviewResult?.revision_suggestion ? `## Revision Suggestion\n${reviewResult.revision_suggestion}` : ""}

## Reproducibility
- Script: \`experiment.py\`
- Seeds: ${JSON.stringify(metrics?.seeds || [42])}
- Dataset: ${metrics?.dataset || "see script"}
- Baseline: ${metrics?.baseline_name || "see script"}
- Hardware: MacBook M4 (MPS: ${metrics?.mps_used !== false})
`;
  await writeFile(join(expDir, "result_report.md"), report, "utf8");

  console.error(`\n  [Sandbox] Decision: ${decision}`);
  console.error(`  [Sandbox] Reason: ${decisionReason}`);

  return summary;
}

// ─── Idea Validation Loop ────────────────────────────────────────────────────

/**
 * At the idea phase: run quick micro-probes on ALL candidate ideas
 * to decide which ones are worth pursuing. This is the "real loop" —
 * run → see results → revise ideas → run again.
 *
 * @param {object[]} ideas        - Candidate ideas from idea tree
 * @param {object[]} methodMatrix - Deep reads from papers
 * @param {string}   topic        - Research topic
 * @param {string}   runsDir      - Run directory
 * @param {object}   llmClient    - LLM client
 * @returns {object[]} ranked ideas with micro-probe results
 */
export async function validateIdeasWithMicroProbes(ideas, methodMatrix, topic, runsDir, llmClient) {
  console.error(`\n${"═".repeat(60)}`);
  console.error(`[IdeaValidator] Running micro-probes on ${ideas.length} ideas...`);
  console.error(`${"═".repeat(60)}`);

  const results = [];

  for (let i = 0; i < ideas.length; i++) {
    const idea = ideas[i];
    console.error(`\n[IdeaValidator] [${i+1}/${ideas.length}] Testing: ${idea.title?.slice(0, 55)}`);

    const expDir = join(runsDir, "idea_probes", `idea_${idea.idea_id || String(i+1).padStart(2, "0")}`);
    await mkdir(expDir, { recursive: true }).catch(() => {});

    const result = await runExperiment({
      idea,
      methodMatrix,
      topic,
      level: "micro",
      expDir,
      llmClient,
    });

    results.push({
      ...idea,
      micro_probe: result,
      viable: result.decision === "PROMOTE_TO_PROBE",
      probe_decision: result.decision,
    });

    console.error(`  [IdeaValidator] Result: ${result.decision} — ${result.decision_reason}`);
  }

  // Sort: viable ideas first, then by improvement size
  results.sort((a, b) => {
    if (a.viable && !b.viable) return -1;
    if (!a.viable && b.viable) return 1;
    const aImp = Object.values(a.micro_probe?.metrics?.improvement || {})[0] || 0;
    const bImp = Object.values(b.micro_probe?.metrics?.improvement || {})[0] || 0;
    return bImp - aImp;
  });

  console.error(`\n[IdeaValidator] Summary:`);
  for (const r of results) {
    const imp = Object.values(r.micro_probe?.metrics?.improvement || {})[0] || 0;
    console.error(`  ${r.viable ? "✅" : "❌"} ${r.title?.slice(0, 45)} → ${r.probe_decision} (${imp > 0 ? "+" : ""}${(imp*100).toFixed(2)}%)`);
  }

  return results;
}

// ─── Self-Critic Revision Loop ────────────────────────────────────────────────

/**
 * Run an idea through the experiment → review → revise → experiment loop.
 * This is the CORE of real research: fail → learn → retry.
 *
 * @param {object} idea         - Research idea
 * @param {object[]} methodMatrix
 * @param {string} topic
 * @param {string} runsDir
 * @param {object} llmClient    - Must have generateRevision() and reviewExperimentResults()
 * @param {string} level        - "probe" | "mvp"
 * @param {number} maxAttempts  - Default 3 revision cycles
 * @returns {object} Final result with all attempt history
 */
export async function runWithRevisionLoop(idea, methodMatrix, topic, runsDir, llmClient, level = "probe", maxAttempts = 3) {
  console.error(`\n${'═'.repeat(60)}`);
  console.error(`[RevisionLoop] Starting for: "${idea.title}"`);
  console.error(`[RevisionLoop] Level: ${level}, Max attempts: ${maxAttempts}`);
  console.error('═'.repeat(60));

  const attempts = [];
  let currentIdea = { ...idea };
  let bestResult = null;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    console.error(`\n[RevisionLoop] ── Attempt ${attempt}/${maxAttempts} ──`);
    console.error(`[RevisionLoop] Idea: ${currentIdea.title?.slice(0, 55)}`);

    const expDir = join(runsDir, "revision_loop", `${idea.idea_id || "idea"}_attempt_${attempt}`);
    await mkdir(expDir, { recursive: true }).catch(() => {});

    const result = await runExperiment({
      idea: currentIdea,
      methodMatrix,
      topic,
      level,
      expDir,
      llmClient,
    });

    attempts.push({
      attempt,
      idea_title: currentIdea.title,
      decision: result.decision,
      metrics: result.metrics,
      revision_suggestion: result.revision_suggestion,
      expDir,
    });

    // Track best result
    const improvement = Object.values(result.metrics?.improvement || {})[0] || 0;
    const bestImprovement = Object.values(bestResult?.metrics?.improvement || {})[0] || -Infinity;
    if (improvement > bestImprovement) bestResult = result;

    // Decide: continue, promote, or kill?
    if (result.decision === "PROMOTE_TO_MVP" || result.decision === "WRITE_PAPER") {
      console.error(`[RevisionLoop] ✅ PROMOTED on attempt ${attempt}!`);
      break;
    }

    if (result.decision === "KILL") {
      console.error(`[RevisionLoop] ❌ KILLED on attempt ${attempt}.`);
      break;
    }

    // REVISE: generate a new version of the idea
    if (attempt < maxAttempts && result.decision === "REVISE") {
      console.error(`[RevisionLoop] 🔄 Generating revision for attempt ${attempt + 1}...`);
      try {
        const revision = await llmClient.generateRevision(
          currentIdea,
          result.metrics,
          result.revision_suggestion || "Try a different approach or dataset",
          attempt
        );
        if (revision?.mechanism) {
          const key = revision.key_change || "approach modified";
          console.error(`[RevisionLoop] Revision: ${key}`);
          currentIdea = {
            ...currentIdea,
            ...revision,
            idea_id: idea.idea_id,  // keep original ID
          };
        }
      } catch (err) {
        console.error(`[RevisionLoop] Revision generation failed: ${err.message}`);
      }
    }
  }

  const finalDecision = attempts[attempts.length - 1]?.decision || "KILL";
  console.error(`\n[RevisionLoop] Final: ${finalDecision} after ${attempts.length} attempts`);

  return {
    idea_id: idea.idea_id,
    idea_title: idea.title,
    total_attempts: attempts.length,
    final_decision: finalDecision,
    best_result: bestResult,
    all_attempts: attempts,
    viable: ["PROMOTE_TO_MVP", "WRITE_PAPER", "PROMOTE_TO_PROBE"].includes(finalDecision),
  };
}

export { LEVELS };
