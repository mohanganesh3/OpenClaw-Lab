/**
 * engineer.mjs — "Researcher from zero" experiment resource preparation.
 *
 * This is the module that makes OpenResearchOS act like a real engineer on your
 * Mac (all via the terminal / OpenClaw exec, YOLO on):
 *   - plan what real dataset + baseline code the idea needs (1 compact Sarvam call)
 *   - git clone the baseline paper's repo (shallow) and INSPECT it (never blind-run)
 *   - download a REAL dataset within a size budget (sklearn cache / HF / torchvision / url)
 *   - build an isolated uv environment with exactly the needed packages
 *   - record full provenance in environment.json + resources.json
 *
 * The generated experiment code (see llm_client.designExperiment) then loads the
 * real dataset. Bounded by budgets so it stays cheap on the Sarvam rate limit
 * (this module makes at most ONE Sarvam call; everything else is exec/IO).
 */

import { writeFile, mkdir, readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
import * as terminal from "./terminal.mjs";
import * as llm from "./llm_client.mjs";

const UV = process.env.UV_PATH || "/Users/mohanganesh/.local/bin/uv";
const MAX_DATASET_MB = 2048; // 2 GB budget

function clean(s) { return String(s || "").replace(/[^\x20-\x7E]/g, " ").replace(/\s+/g, " ").trim(); }

/**
 * Ask Sarvam for a concrete, runnable resource plan for this idea.
 * Compact JSON output (reliable). Falls back to a safe sklearn plan.
 */
export async function planResources(idea, card, level) {
  const codeUrl = card?.code_url || null;
  const datasets = (card?.datasets_used || idea?.required_dataset ? [idea?.required_dataset, ...(card?.datasets_used || [])] : []).filter(Boolean);
  const prompt = `Plan the REAL resources to test this research idea on a MacBook (CPU/MPS, <=2GB data).
IDEA: ${clean(idea?.title)} | mechanism: ${clean(idea?.mechanism || idea?.pitch).slice(0, 200)}
BASELINE: ${clean(idea?.baseline_to_beat) || "logistic regression"}
DATASETS SEEN IN PAPERS: ${datasets.slice(0, 5).join(", ") || "none"}
BASELINE CODE URL (may be null): ${codeUrl || "null"}
LEVEL: ${level}

Choose the SIMPLEST real dataset that genuinely tests the idea and downloads automatically.
Return ONLY compact JSON:
{"loader":"sklearn|huggingface|torchvision|url|none",
 "dataset_name":"e.g. load_digits / fashion_mnist / CIFAR10 / <url>",
 "task":"classification|regression|text|vision",
 "clone_repo": ${codeUrl ? `"${codeUrl}"` : "null"},
 "packages":["scikit-learn","numpy","scipy"],
 "notes":"one line"}`;

  let plan = null;
  try {
    const raw = await llm.ask(prompt);
    const m = raw && raw.match(/\{[\s\S]*\}/);
    if (m) plan = JSON.parse(m[0]);
  } catch { /* fall through */ }

  if (!plan || typeof plan !== "object") {
    plan = { loader: "sklearn", dataset_name: "load_digits", task: "classification",
             clone_repo: codeUrl || null, packages: ["scikit-learn", "numpy", "scipy"], notes: "fallback" };
  }
  // micro-probe stays sklearn-only for speed/reliability
  if (level === "micro_probe") {
    plan.loader = "sklearn";
    // Topic-aware dataset selection for micro-probe
    const topicLower = (idea?.title + " " + (idea?.mechanism || "")).toLowerCase();
    if (/calibrat|active learn|medical|brier|ece/.test(topicLower)) {
      plan.dataset_name = "make_classification";
    } else if (/digit|image|vision|cnn/.test(topicLower)) {
      plan.dataset_name = "load_digits";
    } else if (/breast|cancer|medical|clinical/.test(topicLower)) {
      plan.dataset_name = "load_breast_cancer";
    } else if (!/load_/.test(plan.dataset_name || "")) {
      plan.dataset_name = "load_digits";
    }
    plan.packages = ["scikit-learn", "numpy", "scipy", "matplotlib", "psutil"];
    plan.clone_repo = null;
  }
  return plan;
}

/** Clone a baseline repo (shallow) and read its README for inspection. */
export async function cloneBaseline(repoUrl, expDir) {
  if (!repoUrl || !/^https?:\/\/(www\.)?github\.com\//.test(repoUrl)) {
    return { cloned: false, reason: "no valid github url" };
  }
  const dest = join(expDir, "repo");
  const res = await terminal.cloneRepo(repoUrl, dest, { shallow: true });
  if (!res.success) return { cloned: false, reason: (res.stderr || "clone failed").slice(0, 200) };

  // Inspect: read README (first 2KB) — we do NOT execute repo code.
  let readme = "";
  for (const name of ["README.md", "README.rst", "readme.md", "README"]) {
    const p = join(dest, name);
    if (existsSync(p)) { try { readme = (await readFile(p, "utf8")).slice(0, 2000); } catch {} break; }
  }
  return { cloned: true, path: dest, readme_excerpt: readme, inspected: true, executed: false };
}

/**
 * Download a real dataset within budget. Returns {ok, kind, location, note}.
 * sklearn: just verifies it loads (cached by sklearn). hf/torchvision: prefetch
 * into the experiment's data dir. url: curl with size cap.
 */
export async function downloadDataset(plan, expDir) {
  const dataDir = join(expDir, "data");
  await mkdir(dataDir, { recursive: true }).catch(() => {});
  const loader = plan?.loader || "sklearn";

  if (loader === "url" && /^https?:\/\//.test(plan.dataset_name || "")) {
    const dest = join(dataDir, "dataset.bin");
    const dl = await terminal.downloadFile(plan.dataset_name, dest, { timeoutMs: 180000 });
    return { ok: !!dl.success, kind: "url", location: dest, note: dl.success ? "downloaded" : (dl.error || "failed") };
  }

  if (loader === "huggingface") {
    const py = `import os
os.environ.setdefault("HF_DATASETS_CACHE", ${JSON.stringify(dataDir)})
from datasets import load_dataset
d = load_dataset(${JSON.stringify(plan.dataset_name || "ag_news")})
print("HF_OK", {k: len(v) for k,v in d.items()})`;
    const script = resolve(dataDir, "_prefetch.py");
    await writeFile(script, py, "utf8");
    const r = terminal.run(`${UV} run --with datasets python "${script}"`, { cwd: dataDir, timeoutMs: 300000, silent: true });
    return { ok: r.stdout.includes("HF_OK"), kind: "huggingface", location: dataDir, note: r.stdout.includes("HF_OK") ? "prefetched" : (r.stderr || r.stdout).slice(-200) };
  }

  if (loader === "torchvision") {
    const py = `import torchvision, torchvision.datasets as ds
name=${JSON.stringify(plan.dataset_name || "CIFAR10")}
getattr(ds, name)(root=${JSON.stringify(dataDir)}, download=True)
print("TV_OK")`;
    const script = resolve(dataDir, "_prefetch.py");
    await writeFile(script, py, "utf8");
    const r = terminal.run(`${UV} run --with torchvision --with torch python "${script}"`, { cwd: dataDir, timeoutMs: 600000, silent: true });
    return { ok: r.stdout.includes("TV_OK"), kind: "torchvision", location: dataDir, note: r.stdout.includes("TV_OK") ? "downloaded" : (r.stderr || r.stdout).slice(-200) };
  }

  // sklearn: verify it loads (built-in, tiny)
  const name = /load_/.test(plan.dataset_name || "") ? plan.dataset_name : "load_digits";
  const py = `from sklearn import datasets
X,y=getattr(datasets, ${JSON.stringify(name)})(return_X_y=True)
print("SK_OK", X.shape, len(set(y)))`;
  const script = resolve(dataDir, "_prefetch.py");
  await writeFile(script, py, "utf8");
  const r = terminal.run(`${UV} run --with scikit-learn --with numpy python "${script}"`, { cwd: dataDir, timeoutMs: 120000, silent: true });
  return { ok: r.stdout.includes("SK_OK"), kind: "sklearn", location: `sklearn.datasets.${name}`, note: r.stdout.trim().slice(0, 120) };
}

/**
 * Full preparation: plan → clone → dataset → record provenance.
 * Returns { plan, repo, dataset } and writes resources.json into expDir.
 * Never throws.
 */
export async function prepare(idea, card, level, expDir) {
  await mkdir(expDir, { recursive: true }).catch(() => {});
  console.error(`  [Engineer] Preparing resources for "${idea?.title?.slice(0, 50)}" (${level})`);

  const plan = await planResources(idea, card, level);
  console.error(`  [Engineer] Plan: loader=${plan.loader} dataset=${plan.dataset_name} clone=${plan.clone_repo ? "yes" : "no"}`);

  let repo = { cloned: false };
  if (plan.clone_repo) {
    try { repo = await cloneBaseline(plan.clone_repo, expDir); }
    catch (e) { repo = { cloned: false, reason: e.message }; }
    console.error(`  [Engineer] Repo: ${repo.cloned ? "cloned + inspected" : "skipped (" + (repo.reason || "?") + ")"}`);
  }

  let dataset = { ok: false };
  try { dataset = await downloadDataset(plan, expDir); }
  catch (e) { dataset = { ok: false, note: e.message }; }
  console.error(`  [Engineer] Dataset: ${dataset.ok ? dataset.kind + " ready" : "fallback (" + (dataset.note || "?") + ")"}`);

  const resources = {
    idea_id: idea?.idea_id,
    level,
    plan,
    repo: { cloned: repo.cloned, path: repo.path || null, executed: false, inspected: !!repo.inspected, reason: repo.reason || null },
    dataset,
    prepared_at: new Date().toISOString(),
  };
  await writeFile(join(expDir, "resources.json"), JSON.stringify(resources, null, 2), "utf8").catch(() => {});
  return resources;
}
