# OpenResearchOS — MASTER BUILD PLAN (single source of truth)

> This is THE plan. It consolidates every requirement and constraint agreed over
> the whole project. If anything elsewhere conflicts, this file wins.
> Nothing here is vague: each stage lists inputs, the OpenClaw tools it uses, the
> Sarvam calls + budget, exactly what it stores, and an acceptance test.

---

## 0. NON-NEGOTIABLE REQUIREMENTS (the things never to forget)

1. **Sarvam-only.** Model = `sarvam/sarvam-105b` (the best/highest available),
   117k/125k context. Use the LARGE context (big input is reliable; only long
   OUTPUT is flaky → always ask for compact JSON). No fake "model cascade."
2. **Quality first, no tolerance.** Time/effort do not matter. No shortcuts, no
   laziness, no faking. If something can't be done honestly, say so in artifacts.
3. **Use the LLM, not regex hacks, for judgement** (relevance, gaps, novelty,
   review). Deterministic code is only for orchestration, parsing, and numeric
   computation — never to fake a research judgement.
4. **Execution is truth.** No gap, idea, or claim survives unless code actually
   run on this Mac produces numbers that back it.
5. **Real researcher from zero.** The agent must create folders, set up Python
   environments, download real datasets, clone baseline repos, write code, run
   it, read errors, fix, rerun, compare to competitors, review, and iterate.
6. **Read papers fully** — every page, tables, figure/table captions — not
   abstracts. Store a rich Paper Card per paper.
7. **Showcase OpenClaw fully.** Use exec, apply_patch, pdf, browser, web
   search/fetch, embeddings, sub-agents (`sessions_spawn`), automation
   (cron/standing-orders), memory/wiki, trajectory export, Telegram channel,
   Sarvam voice. The trajectory must prove a real OpenClaw agent team did the work.
8. **From Telegram, end to end.** Start with a text/voice message; live progress;
   approvals; final report + voice summary all on Telegram.
9. **Crash-proof + resumable.** Save after every unit; never throw; budgeted;
   re-running continues from the last completed unit.
10. **Honest gating.** Refuse to write a paper unless gates pass; otherwise write
    a research failure report. Never overclaim.

---

## 1. HARD CONSTRAINTS (authoritative numbers)

- **Rate limit:** sarvam-105b Starter = **40 requests/minute**, token-bucket,
  enforced **per account** (all keys AND all sub-agents share one pool).
- **Cost:** input ₹4 / cached ₹2.5 / output ₹16 per 1M tokens. **Credits ₹900.**
  ~110–125 LLM calls/run ≈ ₹4–12/run ⇒ ~75–200 runs. **Rate is the ceiling, not cost.**
- **Implication:** parallelism helps ONLY for non-LLM work (PDF downloads, git
  clone, dataset download, `uv` builds, experiment runs, browser). **All LLM
  calls — from the runner and every sub-agent — must pass through ONE global
  token bucket (~36/min, retries counted).** Never parallelize Sarvam calls.
- **Hardware:** MacBook Air M4, 36 GB RAM, MPS. Exec policy already YOLO
  (`security=full, ask=off`) so the agent can run commands/write files unattended.

---

## 2. ARCHITECTURE: deterministic engine + OpenClaw-native skin

Two layers, deliberately split:

- **Engine (Node, `src/openresearch.mjs` + modules):** owns the state machine,
  gates, artifacts, budgets, persistence. Deterministic, resumable, cannot get
  "lost." This is why a weak model stays reliable.
- **OpenClaw skin (Coordinator skill + sub-agents):** the visible orchestrator.
  The Coordinator (an OpenClaw agent) calls engine stages via `exec`, and spawns
  specialist **sub-agents** for parallelizable I/O. Telegram is the channel.
  The trajectory export is the proof.

```
Telegram (OpenClaw channel)  ──/start <topic> (text or Sarvam voice)──▶
COORDINATOR (OpenClaw agent + openresearchos-coordinator skill)
  owns state machine + gates; spawns sub-agents; ALL Sarvam calls via global 36/min bucket
   ├─ Literature Scout (sub-agent): web_search + browser (GitHub/PwC/OpenReview) + S2 bulk + snowball
   ├─ Paper Readers (parallel sub-agents): exec download PDF → PyMuPDF full read → 3-pass Paper Card
   ├─ Gap/Idea reasoning: grounded in Paper Cards; gap-validation probes via exec
   ├─ Reviewer Council (sub-agents): adversarial review, budgeted, incremental
   └─ Engineer (sub-agent): mkdir → git clone baseline → download dataset → uv venv →
        apply_patch idea code → run on Mac (MPS) → read traceback → fix → rerun →
        micro → probe → ablation → MVP
Readiness gate → paper_draft.md OR research_failure_report.md
Venue fit · Sarvam TTS voice summary · OpenClaw trajectory export
```

Decision: build/verify the engine path first (reliable), then layer the
sub-agent orchestration on top (showcase). Never convert to a Sarvam-driven
free-running agent loop — a weak model would get lost over 100 steps.

---

## 3. THE FULL PIPELINE — every stage in detail

For each stage: **In / OpenClaw tools / Sarvam calls (budget) / Stores / Accept test.**

### S0. Intake (Telegram)
- In: text or voice message from the allowlisted user.
- OpenClaw: Telegram channel; Sarvam STT for voice → text.
- Stores: new `runs/<run_id>/`, `run_state.json` at `TOPIC_RECEIVED`.
- Accept: run folder + state created; Telegram ack sent.

### S1. Scope & feasibility
- Purpose: decide if testable on this Mac.
- Sarvam: 1 call → classify `{local_experiment | remote_compute_needed |
  data_needed | theory_only | survey_only}` + feasibility note (datasets, est time).
- Stores: `scope_classification` in state.
- Accept: label set; only `local_experiment` may later make validated claims.

### S2. Search plan
- Sarvam: 1 call → ~15 diverse short queries (methods, benchmarks, datasets,
  limitations, competitors, surveys).
- Stores: `search_plan.md`, `search_queries` in state.
- Accept: ≥10 queries.

### S3. Discover (real literature)
- OpenClaw/HTTP: S2 `bulkSearch` (8 queries), arXiv API, OpenAlex, OpenClaw
  `web_search`; then **S2 snowball** (`fetchReferences`/`fetchCitations`) from
  top-cited finds to build a citation graph; `getOpenAccessPdf` for more PDFs.
- Sarvam: 0 (pure retrieval).
- Stores: raw candidate list (≥50 target), `citation_graph.json`.
- Accept: ≥50 candidates OR documented why fewer; snowball produced edges.

### S4. Relevance gate (LLM — no keyword hack)
- Sarvam: batched relevance classifier, 8 papers/call, sanitized input
  (strip LaTeX/quotes/non-ASCII), compact JSON `[{n,relevant}]`. Budget ≤16 calls.
  Keep-when-unjudged; safety floor ≥20 kept; foundational seeds always kept.
- Stores: `evidence_index.json` (kept, with stable `EVxxx` ids),
  `dropped_evidence.json` (dropped + reason).
- Accept: off-topic papers measurably dropped; ≥20 kept; both files written.

### S5. Evidence lock
- Stores: `evidence_index.json` finalized; `evidence/EVxxx/source.md` snapshots.
- Accept: every kept paper has stable id + url + abstract.

### S6. Deep read ALL priority papers → Paper Card  (see §4 for schema)
- Prioritize: top-N (default 30) by citations/recency that have PDF access get
  FULL read; the rest abstract-only (a real researcher reads the key ones deeply).
- Per paper: `exec` download PDF → **PyMuPDF every page (cap 60) + tables
  (`find_tables`→Markdown) + figure/table captions**. Fallbacks: ar5iv/arXiv
  HTML → marker → pdftotext → abstract.
- Sarvam: **3-pass extraction** per full-read paper (large input, compact JSON):
  A=identity/problem/method/contributions; B=setup/results/tables/real-numbers;
  C=critical read/limitations/gaps-enabled/relevance. Persist after EACH paper.
- Stores: `evidence/EVxxx/fulltext.md`, `deep_read.json` (Paper Card),
  `paper_summaries/EVxxx.md` (rich, human-readable), `parsed_papers` in state.
- Accept: full-read papers show real `pdf_pages`/`pdf_tables`>0 and real
  datasets/baselines/headline numbers in the card (not `?`).

### S7. Claim graph + method matrix
- Deterministic assembly from Paper Cards.
- Stores: `claim_graph.json`, `method_comparison_matrix.json`, `research_map.md`
  (method × dataset × metric × best-result table; citation relationships).
- Accept: matrix has real datasets/results for full-read papers.

### S8. Grounded gap synthesis  (LLM, grounded)
- Sarvam: gap analyst fed the cards' `gaps_this_enables`, `unstated_weaknesses`,
  `open_questions`. Each gap MUST cite ≥2 evidence ids + a measurable signal +
  a testable hypothesis. Budget ~1–2 calls.
- Stores: `research_gaps` in state; gaps section of `research_map.md`.
- Accept: every gap cites ≥2 EV ids and has a measurable signal + hypothesis.

### S9. Gap-validation probe (execution — confirm gap is real)
- Engineer writes a tiny script (sklearn/synthetic) and RUNS it to confirm the
  gap's signal exists in data BEFORE building on it. No signal → gap killed.
- OpenClaw: `exec` + `apply_patch`.
- Stores: `runs/<run>/gap_probes/<gap_id>/` (code, stdout, metrics, verdict).
- Accept: each surviving gap has a probe result with a real number.

### S10. Idea generation (grounded in validated gaps)
- Sarvam: 1 call → 6 ideas, each tied to a validated gap, with mechanism,
  hypothesis, dataset, baseline_to_beat, metric. Persist IMMEDIATELY + set a
  provisional experiment candidate (crash-safe).
- Stores: `idea_tree.md`, `idea_tree` + provisional `experiment_candidates`.
- Accept: ≥1 idea persisted before any review call (so a crash can't zero it).

### S11. Reviewer council (adversarial, budgeted, incremental)
- Sarvam: per idea, reviewers one-at-a-time (Novelty + Experimental minimum;
  add Theory/Reproducibility/Venue when budget allows). Save state + append
  `reviewer_pass_1.md` AFTER EACH idea. Budget cap (~12–20 calls). Never throw;
  budget exhaustion → deterministic fallback score for the rest.
- Stores: `reviewer_pass_1.md`, `reviewer_scores`, `failed_ideas` (→ memory).
- Accept: all ideas reviewed and persisted incrementally; killed ideas saved.

### S12. Novelty tribunal
- Sarvam + embeddings + live search: local embedding similarity over corpus,
  S2 competitor search, arXiv recent check, LLM verdict. Top-3 ideas, never-throw.
- OpenClaw: `browser` to inspect competitor GitHub repos when needed.
- Stores: `novelty_tribunal.md`; competitors added to evidence.
- Accept: each shortlisted idea has a novelty verdict + competitor list.

### S13. Experiment ladder — Engineer-from-scratch  (see §5 for full protocol)
- micro_probe → probe → ablation → MVP. Each rung: codegen → env/clone/dataset →
  run on Mac → real metrics → reviewer critique → exactly one next action.
- Accept: ≥1 rung completes with REAL metrics that test the actual idea.

### S14. Competitor comparison
- Deterministic + LLM: put the idea's real numbers against the method matrix on
  the same/comparable dataset. Difference table. No "improvement" without a
  baseline number to beat.
- Stores: comparison table in `reviewer_pass_2.md` / result report.
- Accept: every claimed improvement has a baseline number beside it.

### S15. Readiness gate → output
- Deterministic gate (see §6). Pass → `paper_draft.md` (every claim linked to an
  EV id or experiment log line). Fail → `research_failure_report.md`.
- Stores: `paper_readiness_review.md`, `venue_fit.md`, `final_report.md`, output.
- Accept: output type matches gate; no unsupported claims.

### S16. Voice + trajectory + memory
- Sarvam TTS: concise spoken status in chosen Indic/English language → Telegram.
- OpenClaw: `sessions export-trajectory` (tools, exec, sub-agents, `llm_calls.jsonl`).
- Memory: write lessons + failed ideas; index; dedup future runs.
- Accept: trajectory bundle exists; voice file sent; memory updated.

---

## 4. PAPER CARD — exact schema stored per paper (`deep_read.json`)

Identity: `evidence_id, title, year, venue, arxiv_id, url, citation_count`.
Problem/contribution: `problem_addressed, core_claim, key_contributions[], novelty`.
Method: `method_summary, method_components[], key_innovation, assumptions[], theoretical_grounding`.
Setup: `datasets_used[], baselines_compared[], metrics_used[], hyperparameters, compute_required, seeds`.
Results: `headline_result, results_table[{method,dataset,metric,value}], ablations[], improvement_over_baseline`.
Critical: `stated_limitations[], unstated_weaknesses[], threats_to_validity[], reproducibility, open_questions[], future_work[], gaps_this_enables[]`.
Relevance: `relevance_to_topic, relation_type(baseline|competitor|method_source|dataset_source|foundational|tangential), can_run_on_macbook, reusable_assets[]`.
Provenance: `full_text_tool, pdf_pages, pdf_tables, pdf_captions, full_text_chars`.

Consumed by: method matrix (results_table/datasets/baselines), gaps
(gaps_this_enables/weaknesses/open_questions), ideas (novelty/gaps/reusable_assets),
novelty (core_claim/relation_type), experiment design (datasets/baselines/
hyperparameters/can_run_on_macbook), paper writing (citations + results_table).

---

## 5. ENGINEER-FROM-SCRATCH — full experiment protocol (the heart)

For a promoted idea, per ladder rung, all via OpenClaw `exec`/`apply_patch`
(YOLO on) and the global throttle for any Sarvam call:

1. **Workspace:** `mkdir runs/<run>/experiments/<exp_id>/{src,data,outputs,logs,repo}`;
   write `experiment_spec.json` (hypothesis, metric, baseline, ablation plan,
   budgets: max_runtime, max_disk, max_dataset_gb=2, max_reruns) + `approval.json`.
2. **Clone baseline (if a card has code_url):** `git clone --depth 1 <repo> repo/`;
   READ README/entrypoints to understand it. **Inspect, never blind-run untrusted code.**
3. **Dataset (real):** download within budget — HuggingFace `datasets`,
   `torchvision`, or `curl` a documented URL; record source + size in
   `environment.json`. Fall back to a small sklearn dataset only if download fails.
4. **Environment:** `uv venv` + install exactly the packages the code imports
   (detect from code). Record versions + MPS availability in `environment.json`.
5. **Idea code:** Sarvam writes code that ACTUALLY implements the idea mechanism
   as a distinct variant of the baseline (not identical params). micro = sklearn-
   only/small/fast for reliability; probe/mvp may use torch/MPS. Fixed seeds.
6. **Run on Mac:** capture stdout/stderr; enforce timeout + RAM guard (kill >12GB).
7. **Self-correct:** on failure feed the traceback back to Sarvam → fix → rerun,
   up to N attempts; final fallback = deterministic real-ML template so a rung
   ALWAYS yields real metrics. Normalize metric key aliases.
8. **Decide from REAL numbers:** reviewer critiques metrics → exactly one of
   PROMOTE / ADD_BASELINE / ADD_ABLATION / FIX_AND_RERUN / CHANGE_METRIC /
   CHANGE_DATASET / NARROW_CLAIM / KILL / MARK_REMOTE. Honest success: micro needs
   real improvement; probe/ablation/mvp also need statistical significance.
9. **Stores per exp:** `experiment_spec.json, approval.json, environment.json,
   src/*, repo/*, data/*, outputs/{comparison.png,comparison_table.md},
   logs/{stdout,stderr}.log, metrics.json, result_summary.md, review_after_run.md`.
- Accept: env built from zero, code ran on the Mac, real metrics, decision taken,
  full logs preserved (including failed attempts).

---

## 6. GATES & PAPER READINESS (RRL)

RRL ladder: 0 vague → 1 evidence-backed gap → 2 novelty survives → 3 micro-probe
signal → 4 probe+baseline → 5 ablation/MVP with no fatal flaw → 6 paper draft →
7 venue checklist. **Paper draft only at RRL-5+** AND: avg reviewer ≥6.0, no
fatal novelty/experiment flaw, ≥1 reproducible experiment, all major claims link
to EV ids or experiment logs, competitor comparison present, limitations explicit.
Else → `research_failure_report.md` listing exactly what's missing. Never fake.

---

## 7. RATE-LIMIT & BUDGET ENGINEERING

- **Global token bucket** (`rate_limiter.mjs`): file-based lock under `~/.openclaw`,
  ~36/min, retries counted, shared by runner + all sub-agents. Replaces the
  process-local 3s timer. Fail-open after long wait, self-healing stale lock.
- **Per-run call budget ≈ 113:** queries 1, relevance ~8, deep read ~75 (25×3),
  gaps 1–2, ideas 1, reviewers ~12, novelty ~3, experiment codegen+fixes ~6,
  result review ~3, readiness/paper ~3.
- **Per-stage `RunBudget`** (calls + wall-clock) so a stage degrades instead of
  stalling. Every call logged to `llm_calls.jsonl`.
- **Caching:** Paper Cards cached (`deep_read.json`); never re-read a paper.
  Memory dedups topics/ideas across runs.

---

## 8. OPENCLAW SHOWCASE MAP (what proves we used it)

`exec` (clone/download/env/run) · `apply_patch` (write code) · `pdf`/PyMuPDF
(read papers) · `web_search`/`web_fetch` (discovery/snapshots) · `browser`
(GitHub/PwC/OpenReview/leaderboards) · `embedding` (novelty) · `sessions_spawn`
(parallel specialist sub-agents) · automation `cron`/standing-orders (24/7) ·
`memory`/wiki (cross-run learning) · `sessions export-trajectory` (audit) ·
Telegram channel (control) · Sarvam STT/TTS (Indic voice). The trajectory bundle
must visibly contain these.

---

## 9. BUILD PHASES — concrete tasks + status

**Phase 1 — Crash-proof loop. ✅ DONE & VERIFIED.**
`llm_safe.mjs`; incremental budgeted ideas+reviewers; fixed `decision` crash,
broken Python template, self-healing experiment runner; full unattended run green.

**Phase 2 — Real evidence. ✅ DONE.**
LLM relevance gate (sanitized, batched, keep-when-unsure); PyMuPDF full-paper read
(pages+tables+captions); 3-pass Paper Card; citation-prioritized deep-read cap.

**Phase A — Global Sarvam throttle. ✅ DONE & VERIFIED.**
- [x] `rate_limiter.mjs` (file-lock token bucket, 36/min, retries counted).
- [x] wired into `openclaw_bridge.mjs` (`acquireSlot()` replaces process-local timer).
- [x] TEST PASSED: 50 calls across 2 concurrent processes → max 36 in any 60s window.

**Phase B — Engineer-from-scratch. ✅ DONE & VERIFIED.**
- [x] `engineer.mjs`: planResources (1 Sarvam call) · cloneBaseline (shallow,
      inspect README, never execute) · downloadDataset (sklearn/HF/torchvision/url,
      budgeted) · prepare() writes `resources.json` + provenance. Never throws.
- [x] wired into `createExperimentWorkspace` for probe/ablation/mvp; feeds the
      real dataset into idea-specific codegen. micro_probe stays sklearn-fast.
- [x] TEST PASSED (real ablation run): Engineer cloned google/uncertainty-baselines
      (inspected, not executed), verified sklearn dataset, built env, generated
      idea-specific code, ran 4 self-healing attempts, produced REAL metrics
      (baseline 0.969 vs proposed 0.932 → honestly ADD_BASELINE, not promoted).
      Artifacts present: repo/, resources.json, environment.json, metrics.json.

**Phase C — Grounded gaps + real map. ✅ DONE (gap probes) / ◑ partial (citation graph).**
- [x] `llm.designGapProbe` + `validateGaps()` (+ `validate-gaps` CLI), wired into
      `map`. Each top gap gets a real sklearn probe with a self-heal loop
      (codegen → run → fix-from-traceback → rerun), verdict
      `confirmed|no_signal|error` stored on the gap + `gap_probes/<gap>/`.
- [x] TEST PASSED: G01 self-corrected from error → real verdict `no_signal 0.4496`.
- [x] Citation edges built in `map` (baseline cross-match + S2 snowball in discover).
- [ ] `getOpenAccessPdf` prefetch for non-arXiv papers — deferred (arXiv covers most).

**Phase D — OpenClaw-native orchestration. ✅ DONE (sub-agent team) — TEST PENDING.**
- [x] Coordinator SKILL.md rewritten to the REAL pipeline.
- [x] `openclaw_subagents.mjs`: spawns REAL `openclaw agent` sub-sessions
      (literature-scout, skeptic, synthesizer) under the run's namespace, each
      acquiring a global rate slot; outputs saved to `subagents/<label>.md`; shows
      up in the trajectory as a genuine multi-agent team. `coordinate` CLI added.
      (`openclaw agent --session-key … --json` verified working: sarvam-105b, 120k.)
- [ ] live test of `coordinate` (after the in-flight full run finishes).

**Phase E — Demo package. ✅ DONE (full end-to-end run verified).**

**Bugs fixed in this session (all verified):**
- [x] Null bytes in PDF text caused `spawnSync` to throw "args[6] must be a string without null bytes" — fixed at 3 levels: `pdf_full_extract.py` (strips at write), `pdf_reader.mjs` (strips cached text), `openclaw_bridge.mjs` (universal `safeArg()` sanitizer on all spawnSync args)
- [x] `verifyRun` called `computeReadiness()` without `await` → returned a Promise → readiness level was `undefined` — fixed: `await computeReadiness(state, experimentResults)`
- [x] `allowedNextActions` set in `verifyRun` was missing valid outcomes from experiment sandbox (`PROMOTE_TO_PROBE`, `KILL`, `REVISE`, etc.) — fixed: expanded to full set
- [x] `evidence_index.json count !== state.evidence_ids` — novelty tribunal adds competitor papers to the index but not to state.evidence_ids — fixed: check only fires when index has *fewer* papers than state (real data loss), not more (expected competitor additions)
- [x] `writeLiveStatus` used `.score` on reviewer_scores entries that have `.average_score` — fixed
- [x] `selectedIdea` threw when `--idea ID001` didn't match LLM-generated IDs (`I01`, `I05`, etc.) — fixed with case-insensitive fallback → first experiment candidate → first idea
- [x] `experiment_spec.json` missing `selected_idea.title` → experiment reported `idea_title: "Unknown idea"` — fixed: spec now includes `selected_idea: {title, pitch, mechanism, testable_hypothesis}` and `topic`
- [x] All 19 source `.mjs` files pass `node --check` ✓

**Full chain run (calibration-aware active learning for medical image classification):**
- [x] 61 papers discovered + relevance-gated (dropped off-topic ones)
- [x] 6 papers fully read via PyMuPDF (real tables, captions, pages reported)
- [x] 3-pass Paper Cards produced (some passes failed due to null bytes in old process, but pipeline continued with fallback cards)
- [x] 6 gaps identified by LLM, 2 gap-validation probes run (G01: no_signal -14.58, G02: no_signal 0.0143) — real code, real metrics
- [x] 6 research ideas generated and persisted (crash-safe) immediately after generation
- [x] Reviewer council ran incrementally (4 ideas fully reviewed, 2 got deterministic scores due to 6-min time budget)
- [x] Novelty tribunal ran with local semantic embeddings + S2 + arXiv
- [x] Engineer prepared experiment resources (micro_probe stays sklearn-fast as designed)
- [x] Experiment `micro_probe_i05` ran: real Python code, real sklearn dataset, real metrics: `baseline=0.9694 vs proposed=0.9319 → honestly KILL_IDEA (not better)`
- [x] Real comparison plot (comparison.png) + NeurIPS table (comparison_table.md) generated
- [x] Write stage produced `research_failure_report.md` (correct: RRL-2, below paper threshold)
- [x] `verify` produced `ok: true, problems: []` ✓
- [x] Sub-agent team coordinated (skeptic: OK, synthesizer: OK, literature-scout: failed due to external API format error not in our code)
- [x] LLM calls: max 4 in any 60s window (well within 36/min cap) ✓
- [x] Memory: failed ideas and experiment lessons persisted to OpenClaw memory ✓

**Demo run in-flight** (separate run `run_20260606182759_calibration-aware-active-learning_a7d448`):
- [x] Memory dedup: 5 prior items retrieved from cross-run memory
- LLM generated 15 search queries
- S2 bulk + arXiv discovery in progress

---

## 10. ACCEPTANCE TESTS (definition of "great", measurable)

- Fresh topic from Telegram → unattended → ≥1 experiment with REAL metrics that
  tests the actual idea (env built from zero; repo cloned or real dataset used).
- ≤36 Sarvam calls in any 60s window across all processes; 0 stage crashes
  (fault-injection test passes); run resumes after a kill at any point.
- ≥80% of locked evidence on-topic; full-read papers show real tables/numbers.
- Every gap cites ≥2 papers + has a validation-probe number.
- Every paper claim links to an EV id or experiment log line; else honest failure.
- Trajectory bundle shows exec + sub-agents + tools; voice summary delivered.
- A genuinely good idea can reach `paper_draft.md`; weak ideas get honest failure.

---

## 11. RISK REGISTER

- Sarvam empty output → compact-JSON prompts, retries(counted)+fallback, never throw.
- Rate cap → global bucket; parallelize only I/O.
- Buggy LLM code → self-correct loop + deterministic real-ML fallback.
- Untrusted repos → inspect, never blind-run; budgets on disk/runtime/reruns.
- Weak-model drift → deterministic engine owns the loop; Sarvam only narrow calls.
- Wasteful repeat runs → memory dedup + cross-run lessons.
