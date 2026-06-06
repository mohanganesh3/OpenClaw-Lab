# OpenResearchOS v2 — Execution-Grounded Research Operating System

> This supersedes `OPENRESEARCHOS_HIGH_BAR_PLAN.md`. The old plan is preserved.
> v2 keeps the philosophy (gates, traceability, refusal to overclaim) and fixes
> the reason every autonomous run currently fails: the research half never runs.

---

## 0. The One Idea That Changes Everything

**The agent must not trust the LLM's brain. It must prove every belief by writing and running code on the laptop.**

Sarvam-105b is the only model we have. It is good at language and extraction,
weak at deep reasoning and unreliable at long structured output. So we stop
asking it to *reason its way* to research conclusions. Instead:

- The LLM proposes (gaps, ideas, hypotheses, code).
- The **terminal proves or kills** them (run the code, read the numbers).
- A belief is only "true" once an experiment on this Mac produces a number that supports it.

This is also our novelty versus AI Scientist v2 / Agent Laboratory:

> **OpenResearchOS is execution-grounded and fully traceable: no claim survives
> unless a reproducible local experiment, with logged stdout/metrics, backs it.**

That story is honest, demoable, and exactly Professor Xie's lane (reliable,
traceable agents). It also happens to be the only way to get good output from a
weak model: ground it in execution, not imagination.

---

## 1. Why v1 Fails Today (measured, not guessed)

Inspected ~55 runs in `runs/`. Findings:

1. **Only 1 of 55 runs ever completed the loop.** Every autonomous daemon run
   collects 69–70 papers, then produces **0 ideas, 0 experiments**, and writes a
   failure report. The research half is dead in autonomous mode.

2. **Root cause A — fragile single mega-stage.** `ideasRun` generates up to 20
   ideas, then runs `20 ideas × 4 reviewers × 2 turns` = 160+ sequential Sarvam
   calls (min 3s each, ~20 req/min rate limit) and only writes `idea_tree.md` at
   the very end. The daemon times out mid-loop → nothing is ever saved → 0 ideas.

3. **Root cause B — Sarvam asked to do too much per call.** Long prompts with 4
   reviewer personas and strict JSON schemas return empty/unparseable output,
   the stage throws, run dies.

4. **Evidence relevance is broken.** "Efficient attention" runs locked papers on
   quantum medical computing, reward hacking, point clouds. No relevance gate.

5. **Papers are not read fully.** Summaries are short abstract slices. Method
   matrix is mostly `?`. Gaps are therefore shallow and ungrounded.

6. **Experiments are toys** (`sklearn.load_digits`, logistic regression) that
   don't match the bar of a lab whose AIBuildAI is #1 on MLE-Bench.

---

## 2. Design Principles for a Weak-Model + Strong-Execution System

| Principle | What it means concretely |
|---|---|
| **Tiny calls, never mega-calls** | One Sarvam call = one paper field, OR one idea, OR one reviewer. Never "4 reviewers in one prompt." |
| **Persist after every unit** | Save state after each paper, each idea, each review. A killed process loses one unit, not the whole stage. |
| **Never throw — always degrade** | Any failed LLM call returns a typed fallback and logs it. A stage cannot crash the run. |
| **Execution is truth** | Gaps and ideas are scored by running code, not by LLM confidence. |
| **Resumable by construction** | Every stage checks "already done?" and skips. Re-running a run continues where it stopped. |
| **Schema-validate + repair JSON** | Every JSON call goes through parse → repair → validate → fallback. |
| **Budget-aware** | Each stage has a max-call budget so it finishes within the daemon window. |

---

## 3. The v2 Pipeline (what you described, made concrete)

```
TOPIC
 → SCOPE & FEASIBILITY (local vs remote)
 → SEARCH PLAN (LLM queries + relevance gate)
 → DISCOVER (arXiv / OpenAlex / S2 / web)
 → RELEVANCE GATE  ← NEW: drop off-topic papers before locking
 → EVIDENCE LOCK (stable EV ids, snapshots)
 → DEEP READ ALL PAPERS  ← NEW: full-text, field-by-field, one call per field
 → CLAIM GRAPH + METHOD MATRIX (real datasets, results, baselines)
 → GROUNDED GAP SYNTHESIS  ← gaps must cite ≥2 papers + a measurable signal
 → GAP VALIDATION PROBE  ← NEW: write+run tiny code to confirm the gap is real
 → IDEA GENERATION (grounded in validated gaps)
 → IDEA REVIEW (one reviewer per call, incremental save)
 → NOVELTY CHECK (embeddings + live search)
 → EXPERIMENT LADDER: micro-probe → probe → ablation → MVP
     each: codegen → run on Mac → read metrics → reviewer critique → iterate
 → COMPETITOR COMPARISON  ← NEW: numbers vs method matrix, not vibes
 → READINESS GATE → paper OR honest failure report
 → VENUE FIT → TRAJECTORY EXPORT
```

New/upgraded stages are marked. The spine is your old plan; the difference is
**every reasoning stage is backed by execution and incremental persistence.**

---

## 4. Stage Specs (the parts that were broken)

### 4.1 Relevance Gate (NEW)

After discovery, before locking, score each paper for topic relevance:

- Cheap deterministic first pass: keyword/embedding cosine vs the topic.
- One short Sarvam call only for borderline papers: `{relevant: true|false, reason}`.
- Drop papers below threshold. Log dropped papers to `dropped_evidence.json` (traceable).
- Target: ≥ 80% of locked papers genuinely on-topic.

### 4.2 Deep Read All Papers (UPGRADE)

For every locked paper, extract full text (PDF reader already exists), then run
**one Sarvam call per field** (not one call for everything):

```
fields = [problem, method, datasets, metrics, main_result,
          baselines_compared, limitations, stated_future_work, code_url]
```

- Each call: ≤ 800 chars context, single field, strict tiny JSON.
- Persist the paper summary immediately after each paper (resumable).
- This fills the method matrix with real datasets/results instead of `?`.

### 4.3 Grounded Gap Synthesis + Gap Validation Probe (NEW — your "real gaps")

A gap is only accepted if it has:

1. ≥ 2 supporting evidence IDs,
2. a **measurable signal** (a metric/quantity that would change if the gap is real),
3. a **local validation probe**: a tiny script the agent writes and runs to
   confirm the gap actually exists in data before any idea is built on it.

Example: gap = "method X degrades past 32k tokens." Validation probe = generate
synthetic long sequences, run a cheap proxy, confirm the degradation trend
exists. If the probe shows no signal, the gap is killed. **This is the agent
checking reality instead of hallucinating a gap.**

### 4.4 Idea Generation + Incremental Review (FIX the crash)

- Generate 6 ideas (not 20), each grounded in a *validated* gap.
- **Review one idea with one reviewer per call**, and `saveState` after each.
- Write `idea_tree.md` and `reviewer_pass_1.md` incrementally as data arrives.
- Reviewer budget cap (e.g. 6 ideas × 2 key reviewers = 12 calls, not 160).
- Any failed call → fallback review, never throws.

Result: even if the daemon restarts, ideas and partial reviews are on disk.

### 4.5 Experiment Ladder — the heart of "use my laptop" (UPGRADE)

For each surviving idea, the agent **actually drives the terminal via OpenClaw**:

1. Create workspace `runs/<run>/experiments/<exp_id>/`.
2. Write `experiment_spec.json` (hypothesis, metric, baseline, ablation, budget).
3. Set up env with `uv` (record `environment.json`).
4. **Codegen with Sarvam, then run on the Mac (MPS).** Capture stdout/stderr.
5. If it crashes: feed the traceback back to Sarvam, regenerate, rerun (max N tries).
   → This is the "not blind hallucination — run, see error, fix, rerun" loop.
6. Read `metrics.json` (real numbers).
7. Reviewer critiques the *real metrics* → exactly one next action
   (PROMOTE / ADD_BASELINE / ADD_ABLATION / FIX_AND_RERUN / CHANGE_METRIC /
   NARROW_CLAIM / KILL / MARK_REMOTE).
8. Iterate up the ladder: micro-probe → probe → ablation → MVP.

Self-debugging codegen (step 5) is the single highest-value upgrade. It turns a
weak model into a working engineer because the *compiler and the data* correct it.

### 4.6 Competitor Comparison (NEW)

Before any paper claim, compare the idea's real numbers against the method
matrix (real results from read papers) on the same/comparable dataset. Output a
difference table. No "improvement" claim without a baseline number to beat.

---

## 5. Sarvam Hardening Layer (so the only model we have actually works)

Build one module `llm_safe.mjs` that wraps every call:

- `askField(context, field)` → single-field extraction, ≤ 800 chars.
- `askJsonStrict(prompt, schema)` → parse → regex-repair → schema-validate →
  fallback object. Never returns malformed data, never throws.
- Global queue honoring the 3s / 20-rpm limit with backoff (already partly there).
- Per-stage call budget + time budget; stage stops gracefully when hit.
- Every call logged to `runs/<run>/llm_calls.jsonl` for the trajectory/audit.

Rule: if a prompt is longer than ~1,500 chars or asks for more than one object,
it is split. No exceptions.

---

## 6. What Stays (your good instincts)

- Deterministic state machine + artifact-per-stage (great auditability).
- Gates + refusal to write unsupported papers (matches Xie's interest exactly).
- Sarvam **voice** layer (Indic STT/TTS) as the interface — distinctive, kept.
- Telegram control + 24/7 daemon — kept, but only switched on after the loop is green.
- Trajectory export — kept; now also includes `llm_calls.jsonl` and experiment logs.

---

## 7. Build Order (do not skip; each must be green before the next)

> **STATUS (verified by a full unattended run on "calibration-aware active
> learning for image classification"):**
> Phase 1 ✅ done — loop survives end-to-end, 6 ideas generated+persisted,
> budgeted incremental reviewer council (12 calls), real micro-probe with real
> metrics on the Mac, honest failure report, `verify` passes clean.
> Phase 2 ◑ partial — LLM relevance gate live (kept 59 / dropped 9 off-topic);
> full per-field deep read still pending.
> Remaining biggest quality lever: experiments currently test a generic
> baseline (sklearn) rather than the *specific idea mechanism*, so good ideas
> cannot yet "win" → see Phase 4.

**Phase 1 — Make the loop survive (no new features). ✅ DONE**
- `llm_safe.mjs` (budget, never-throw, `llm_calls.jsonl` audit) — added.
- Idea generation + review rewritten: capped to 6, persisted immediately,
  budgeted incremental council, state saved after every idea.
- Fixed the `decision`-undefined crash + broken Python template + self-healing
  experiment runner (package detection + LLM fix + deterministic fallback).
- Verified: `demo` runs unattended → non-zero ideas + ≥1 real experiment.

**Phase 2 — Make evidence real. ✅ DONE (deep read upgraded)**
- ✅ LLM relevance gate (batched, sanitized, budgeted, keeps-when-unsure).
- ✅ **Genuine full-paper reading**: downloads the PDF and reads EVERY page with
  PyMuPDF — body text + tables (`find_tables` → Markdown) + figure/table
  captions. Verified: a 14-page paper → ~70k chars, 11 tables, 7 captions
  (vs the old 8k truncated, table-stripped HTML).
- ✅ Two-pass LLM extraction: method/datasets/baselines + results/tables/numbers.
- Tool usage is OpenClaw-native where it adds value (model, web search, web
  fetch, embeddings, memory, trajectory) + local `uv`/PyMuPDF for binary PDF
  work (higher fidelity than OpenClaw web-fetch for tables).

**Phase 3 — Make gaps and ideas grounded.** (gap validation probes — pending)

**Phase 4 — Make experiments self-correcting AND idea-specific.**
- The probe/mvp path already has the LLM-codegen + revision loop.
- TODO: make the micro-probe test the *actual idea mechanism* (not generic
  sklearn), so a genuinely good idea can beat baseline and reach a paper draft.
  This is the top remaining quality item.

**Phase 5 — Demo polish for Professor Xie.** (voice, Telegram, sample run, note)

---

## 8. Honest Positioning for the Professor

> OpenResearchOS is an **execution-grounded, fully-traceable research operating
> system**. It runs on a single laptop model (Sarvam) but stays reliable because
> it validates every gap, idea, and claim by writing and running code, logging
> all stdout/metrics, and refusing to write a paper unless reproducible local
> experiments support it.

Not claimed: automatic A* papers, large-model training on a Mac.
Claimed and demoable: disciplined process, self-correcting experiments,
full audit trail, honest refusal — exactly the reliability/traceability agenda.

---

## 9. Success Criteria (measurable)

- A fresh topic runs unattended to ≥ 1 completed experiment with real metrics.
- 0 stages can crash the run (fault injection test passes).
- ≥ 80% of locked evidence is on-topic.
- Every gap cites ≥ 2 papers and has a validation probe result.
- Every paper claim links to an evidence ID or an experiment log line.
- The whole run resumes correctly after a kill at any point.

---

## 10. Clarification: What "SCOPE & FEASIBILITY (local vs remote)" Means

Before deep work, the agent classifies whether the topic can actually be tested
on *this* MacBook (M4, 36 GB, MPS):

- `local_experiment` — runnable here: small models, synthetic/small data,
  sklearn / PyTorch-MPS, simulations. The agent may produce **validated** claims.
- `remote_compute_needed` — needs GPU cluster / large training / big benchmark.
  The agent produces a research map + experiment *plan*, but must NOT claim a
  validated result it could not run.
- `data_needed` — blocked until a dataset is obtained.
- `theory_only` — needs proof/derivation/simulation, not a normal experiment.
- `survey_only` — can produce a literature map but not a validated contribution.

Purpose: stop the agent from pretending it ran something it physically can't.
Honesty about feasibility is part of the traceability story.

---

## 11. How OpenClaw Drives the MacBook (the engine for "use my laptop fully")

OpenClaw gives an agent full control of this Mac through built-in tools:

| Capability | OpenClaw tool | Used for |
|---|---|---|
| Run any shell command | **`exec`** | `uv venv`, `pip install`, `python run.py`, `mkdir`, `git clone`, capture stdout/stderr |
| Create / edit files | **`apply_patch`** | write experiment code, configs, data scripts |
| Read papers | **`pdf`** | full-text extraction |
| Inspect dynamic pages | **`browser`** | GitHub, OpenReview, Papers with Code, leaderboards |
| Discover + snapshot sources | **`web search` / `web fetch`** | literature + competitor discovery |
| Parallel specialists | **sub-agents** | literature scout, reviewer, experiment runner |
| Long-horizon scheduling | **cron / background tasks / standing orders** | resume work across hours/days |
| Audit trail | **trajectory bundles** | prove every tool call to the professor |

**Exec approvals / YOLO mode.** Host exec is gated by `~/.openclaw/exec-approvals.json`
plus `tools.exec.*` config. This Mac's effective policy is already
`security=full, ask=off, askFallback=full` — i.e. **YOLO is on**, so the agent
runs commands and writes files with no prompts. To make it explicit/persistent:

```bash
openclaw exec-policy preset yolo
```

To tighten later (e.g. for a public demo): `tools.exec.security=allowlist` + `ask=on-miss`.

### Architecture decision: deterministic orchestration, OpenClaw execution

Because Sarvam is a weak reasoner, we do **NOT** let the agent loop freelance
every step. Instead:

- **Node runner (`openresearch.mjs`) owns the loop, state machine, and gates.**
  It is deterministic, resumable, and cannot get "lost."
- **OpenClaw executes the machine work**: `exec` for env setup + running code,
  `apply_patch`/file writes for code, `pdf`/`browser`/`search` for evidence,
  Sarvam for *narrow* generate-this / fix-this-code / extract-this-field calls.
- The **self-correcting experiment loop** is driven by Node:
  `codegen (Sarvam) → run (exec) → read traceback → fix (Sarvam) → rerun`,
  with a max-attempts budget. The compiler and the data correct the weak model.

This is the "researcher from zero" behavior the vision wants — set up envs,
write code, run, debug, iterate — but with reliable orchestration so it survives
long-horizon runs.

### Optional: OpenClaw-native live demo path

For the professor's "wow" moment, expose a thin path where a single
`openclaw agent` turn (with YOLO exec + the research skill) drives the Mac live
on a small task. This shows OpenClaw-native autonomy directly, while the
deterministic runner remains the workhorse for full runs.

### New success criterion

- The agent sets up a fresh experiment environment (`uv` venv + deps), writes
  code, runs it, hits an error, fixes it, and reruns to success — all logged,
  unattended, on this MacBook.

---

## 12. What "Reading a Paper" Must Produce — the Paper Card

Reading a paper is not "grab the abstract." A real researcher reads to fill a
structured mental model. We persist that as a **Paper Card** (`deep_read.json`
per evidence id + a human-readable `paper_summaries/EVxxx.md`). Every downstream
stage consumes it, so it must be rich and grounded in the full text + tables.

### Sources fed into extraction
- Full PDF read by PyMuPDF: every page (cap 60), tables → Markdown, figure/table
  captions. ~30–120k chars instead of an 8k truncation.
- Section split (abstract/intro/related/method/experiments/results/ablation/limits).

### Paper Card schema (what we store)

**Identity**: `evidence_id, title, year, venue, arxiv_id, url, citation_count`.

**Problem & contribution**
- `problem_addressed` — the problem/gap the paper targets.
- `core_claim` — one-sentence thesis.
- `key_contributions` — list.
- `novelty` — what is new vs prior work.

**Method**
- `method_summary` — exact technical approach.
- `method_components` — building blocks.
- `key_innovation` / `mechanism` — WHY it works.
- `assumptions`, `theoretical_grounding` (bounds/proofs if any).

**Experimental setup**
- `datasets_used` (+ sizes), `baselines_compared`, `metrics_used`,
  `hyperparameters`, `compute_required`, `seeds`.

**Results (real numbers from tables)**
- `headline_result` — best number + dataset.
- `results_table` — method × dataset × metric → value (from extracted tables).
- `ablations` — what each component contributes.
- `improvement_over_baseline`.

**Critical reading (researcher judgement)**
- `stated_limitations` (authors' own).
- `unstated_weaknesses` (reviewer-style critique).
- `threats_to_validity`, `reproducibility` (code/data/seeds).
- `open_questions` / `future_work`.
- `gaps_this_enables` — new research this opens (**drives idea generation**).

**Relevance to our topic**
- `relevance_to_topic`, `relation_type`
  (baseline | competitor | method_source | dataset_source | foundational | tangential),
- `can_run_on_macbook`, `reusable_assets` (datasets/code we could reuse locally).

**Provenance**: `full_text_tool, pdf_pages, pdf_tables, pdf_captions, full_text_chars`.

### Extraction = 3 compact passes (large input, short JSON output)
1. **Identity + Problem + Method + Contributions** (intro+method, ~10k chars).
2. **Setup + Results + Tables** (experiments/results incl. Markdown tables, ~12k).
3. **Critical read + Relevance** (limits/conclusion + abstract + our topic).

Each pass sends large context (sarvam-105b handles ~30k chars reliably) but asks
for small JSON, which is the reliable mode. Passes merge into one Paper Card;
any failed pass degrades gracefully (keeps the other passes).

### Who consumes the Paper Card
- Method matrix ← `results_table, datasets_used, baselines_compared`.
- Gap synthesis ← `gaps_this_enables, unstated_weaknesses, open_questions`.
- Idea generation ← `novelty, gaps_this_enables, reusable_assets`.
- Novelty tribunal ← `core_claim, relation_type`.
- Experiment design ← `datasets_used, baselines_compared, hyperparameters, can_run_on_macbook`.
- Paper writing ← citations + `results_table` + related-work framing.

---

# 13. MASTER PLAN (full-codebase audit + the path to the goal)

## 13.1 Accurate current-state map (every component)

**Control surfaces**
- `src/research_daemon.mjs` — the real front door. Idles, listens to Telegram via
  `bot_flags.json`/`daemon_state.json` file-IPC, runs the 7-stage pipeline by
  spawning `node src/openresearch.mjs <stage>` per stage, with per-stage
  timeouts, pause/resume/skip/stop between stages, and live progress pings.
- `src/telegram_bot.mjs` — polished bot: inline buttons + full command set,
  results retrieval (`/paper /gaps /ideas /runs`). Raw Telegram Bot API.
- `channels/command_router.mjs` — simpler one-shot command→runner with allowlist
  (the OpenClaw-channel-facing entry).

**Pipeline** (`src/openresearch.mjs`, ~4.1k lines): deterministic state machine,
stages start/discover/read/map/ideas/plan/approve/run-*/write/verify, resumable,
incremental saves, artifact-per-stage.

**Discovery**: `semantic_scholar.mjs` (search, **bulkSearch, batchFetchPapers,
snowballCitations, fetchReferences, fetchCitations, getOpenAccessPdf**),
`arxiv_client.mjs` (search/latest/fetchPaper, pdf/source URLs), OpenAlex + OpenClaw
web search in the runner.

**Reading**: `pdf_reader.mjs` — PyMuPDF full read (all pages + tables + captions)
→ 3-pass Paper Card. `pdf_full_extract.py` does the extraction.

**Experiments**: `terminal.mjs` (real exec: **cloneRepo, downloadFile,
downloadPdf, createExperimentEnv, installPackages, runPython, pdfToMarkdown,
checkMPS**) + `experiment_sandbox.mjs` (designExperiment, self-healing run,
revision loop) + the runner's own micro-probe path with deterministic template.

**Model layer**: `openclaw_bridge.mjs` (sarvam-105b pinned, 48k char limit,
process-local 3s throttle, retry/backoff, embeddings, web search/fetch, memory,
trajectory) + `llm_client.mjs` (relevance, gaps, ideas, reviewers, codegen,
paper writer) + `llm_safe.mjs` (RunBudget, never-throw, llm_calls.jsonl).

**Key finding:** the "researcher-from-scratch" abilities (clone repo, download
dataset, build env) and citation-graph tools ALREADY EXIST but are NOT wired
into the experiment/reading flow. Much of the upgrade is wiring, not building.

## 13.2 What must improve a lot
1. Experiments must truly test the idea (clone baseline repo, download real
   dataset, build env, implement the mechanism) — not fall back to generic sklearn.
2. Global cross-process Sarvam throttle at ~36/min (current 3s/process is unsafe
   across stages/sub-agents and over-conservative).
3. OpenClaw-native orchestration: Coordinator skill + sub-agents + Telegram via
   OpenClaw channel, so the trajectory shows a real OpenClaw agent team.
4. Gap-validation probes (confirm a gap with a tiny run).
5. Citation-graph use (snowball) for a real research map.
6. Cross-run memory/dedup so repeated runs get smarter, not wasteful.

## 13.3 The target architecture (OpenClaw-native, researcher-from-scratch)

```
Telegram (OpenClaw channel)
   │  /start <topic>  (text or Sarvam voice)
   ▼
COORDINATOR  (OpenClaw agent + skill; owns state machine + gates)
   │  spawns sub-agents (sessions_spawn), all sharing ONE global 36/min Sarvam bucket
   ├── Literature Scout sub-agent ── web_search + browser (GitHub, PwC, OpenReview)
   │                                 + S2 bulk + snowball citation graph
   ├── Paper Reader sub-agents (parallel) ── exec: download PDF → PyMuPDF full read
   │                                          → 3-pass Paper Card
   ├── Gap/Idea reasoning ── grounded in Paper Cards (+ gap validation probes via exec)
   ├── Reviewer Council sub-agents ── adversarial review (budgeted)
   └── Engineer sub-agent ── exec: mkdir workspace · git clone baseline repo
                              · download real dataset (HF/torchvision/curl, budgeted)
                              · uv venv + install · apply_patch idea code
                              · run on Mac (MPS) · read traceback · fix · rerun
                              · micro→probe→ablation→MVP
   ▼
Readiness gate → paper_draft.md OR research_failure_report.md
   ▼
Venue fit · Sarvam TTS summary · OpenClaw trajectory export (the proof)
```

Parallelism is for I/O/compute (downloads, clones, env builds, runs, browser).
LLM calls are always serialized through the global 36/min bucket (the 40/min cap
is per-account and shared, so parallel LLM calls do not help).

## 13.4 The researcher-from-scratch experiment protocol (Engineer)

For a promoted idea, the Engineer (via OpenClaw `exec`, YOLO already on):
1. `mkdir` experiment workspace + write `experiment_spec.json`, `approval.json`.
2. If a relevant paper has code: `git clone --depth 1 <repo>` into `repo/`,
   inspect (read README/entrypoints) — never blind-run untrusted code.
3. Download a real dataset within budget: HuggingFace `datasets`, `torchvision`,
   or `curl` a URL; cap size (default 2 GB); record source in `environment.json`.
4. `uv venv` + install exactly the detected packages (sklearn/torch/etc.).
5. `apply_patch` idea-specific code that implements the mechanism vs a baseline,
   fixed seeds, error bars.
6. Run on the Mac (MPS), capture stdout/stderr/metrics; on failure read traceback,
   fix, rerun (self-correcting, bounded).
7. Reviewer critiques real metrics → exactly one next action.
All of this is `exec`/file work — cheap on the Sarvam rate budget.

## 13.5 Rate + cost budget (authoritative)
- sarvam-105b Starter: **40 req/min**, token-bucket, per-account (shared by all
  keys + sub-agents). Pricing input ₹4 / output ₹16 per 1M tokens. Credits: ₹900.
- ~113 LLM calls/run ≈ ₹4–12/run ⇒ ~75–200 runs on ₹900. Rate, not cost, is the
  ceiling. Throttle target ~36/min with retries counted.

## 13.6 Phases (ordered; each green before next)
- **A. Global throttle** (cross-process token bucket ~36/min). Small, unblocks all.
- **B. Engineer-from-scratch** (wire terminal.mjs clone/download/env into the
  experiment stage; idea code truly implements the mechanism). Biggest quality lever.
- **C. Gap-validation probes** + **citation-graph research map** (wire S2 snowball).
- **D. OpenClaw-native orchestration** (Coordinator skill + sub-agents + Telegram
  via OpenClaw channel + richer trajectory). The showcase.
- **E. Demo package** (one green run with a real win, trajectory, note, memory/dedup).
