# OpenResearchOS: High-Bar OpenClaw Research Operating System

## Purpose

OpenResearchOS is a topic-agnostic research operating system built on OpenClaw.
The user gives a research area, and the system runs a disciplined research loop:

`Topic -> Scope -> Search Plan -> Live Search -> Evidence Lock -> Paper Reading -> Research Map -> Idea Tree -> Reviewer Attack -> Novelty Search -> Micro-Probes -> Probe Experiments -> Result Review -> Revision Loop -> MVP Experiment -> Readiness Gate -> Paper or Failure Report -> Venue Fit -> Trace Export`

The goal is not to promise acceptance at a top conference. The goal is to make the agent behave like a serious junior researcher:

- it reads current and foundational work,
- maps claims, gaps, and competitors,
- generates many ideas instead of one shallow idea,
- attacks ideas with simulated reviewers,
- runs cheap local experiments during thinking,
- promotes only ideas that survive evidence and experiment gates,
- refuses to write unsupported papers,
- exports a trace showing tools, sources, decisions, and experiment logs.

## OpenClaw-Native Design

This project must be built using OpenClaw, not as a disconnected script.

OpenClaw is responsible for:

- skill-based research workflows,
- web and paper discovery,
- browser inspection of dynamic pages,
- PDF reading,
- subagent parallelism,
- local execution through `exec`,
- memory/wiki persistence,
- Telegram command entry,
- Sarvam voice input and output,
- trajectory export for auditability.

The local OpenResearchOS runner is responsible for deterministic state and artifacts:

- `run_state.json`,
- experiment workspaces,
- reviewer score files,
- logs,
- metrics,
- paper-readiness decisions,
- resumability.

This split is deliberate. OpenClaw gives the agent intelligence and tools; OpenResearchOS gives the process hard rails.

## Core Principle

The agent must not produce a one-shot research idea.

Every idea must pass through:

1. evidence collection,
2. prior-art search,
3. reviewer attack,
4. novelty tribunal,
5. local micro-probe where feasible,
6. result critique,
7. revision or rejection,
8. paper-readiness review.

If the evidence is weak, the system outputs a research failure report instead of pretending it has a paper.

## Research Run Protocol

Every research run has a persistent `run_id`.

Run directory:

```text
openresearchos/runs/<run_id>/
```

Required state in `run_state.json`:

- `run_id`
- `topic`
- `created_at`
- `current_state`
- `scope_classification`
- `search_queries`
- `evidence_ids`
- `parsed_papers`
- `claim_graph`
- `research_gaps`
- `idea_tree`
- `reviewer_scores`
- `experiment_candidates`
- `approval_status`
- `micro_probe_results`
- `probe_results`
- `mvp_results`
- `failed_ideas`
- `validation_results`
- `paper_readiness_level`
- `final_output_path`

The run must resume from the last completed state. No completed evidence, failed idea, or failed experiment should be discarded.

## State Machine

The full state machine is:

1. `TOPIC_RECEIVED`
2. `TOPIC_SCOPED`
3. `SEARCH_PLAN_CREATED`
4. `LITERATURE_DISCOVERY_RUNNING`
5. `EVIDENCE_LOCKED`
6. `PAPERS_PARSED`
7. `CLAIM_GRAPH_BUILT`
8. `RESEARCH_MAP_READY`
9. `GAPS_IDENTIFIED`
10. `IDEA_TREE_GENERATED`
11. `REVIEWER_PASS_1_RUNNING`
12. `IDEAS_REVISED_OR_REJECTED`
13. `NOVELTY_TRIBUNAL_RUNNING`
14. `IDEAS_SHORTLISTED`
15. `MICRO_PROBE_PLANNED`
16. `AWAITING_HUMAN_APPROVAL`
17. `MICRO_PROBE_RUNNING`
18. `MICRO_PROBE_REVIEWED`
19. `PROBE_EXPERIMENT_PLANNED`
20. `PROBE_RUNNING`
21. `RESULT_REVIEW_RUNNING`
22. `REVISION_DECISION`
23. `ABLATION_OR_MVP_PLANNED`
24. `MVP_EXPERIMENT_RUNNING`
25. `PAPER_READINESS_REVIEW`
26. `PAPER_DRAFTED_OR_STOPPED`
27. `VENUE_FIT_DONE`
28. `TRACE_EXPORTED`

## Topic Triage

Before deep work, the system classifies the topic:

- `local_experiment`: feasible on Mac M4 with small code/data.
- `remote_compute_needed`: needs GPU, large model training, or large benchmark reproduction.
- `data_needed`: cannot test until a dataset is obtained.
- `theory_only`: needs proof, derivation, or simulation instead of normal experiments.
- `survey_only`: can produce a literature map/report but not a validated contribution.

If the topic is not `local_experiment`, the system may still produce:

- research map,
- evidence index,
- gap map,
- idea tree,
- remote experiment plan,
- failure or limitation report.

It must not claim a validated contribution without experiments.

## OpenClaw Capability Map

### `web_search`

Used for:

- recent paper discovery,
- foundational paper discovery,
- competitor search,
- benchmark and dataset discovery,
- venue and workshop discovery,
- near-duplicate idea search.

Search is repeated at different stages. The first search maps the field; later searches attack specific ideas.

### `web_fetch`

Used for:

- freezing readable web evidence,
- saving source snapshots,
- capturing abstracts, blog posts, documentation, and benchmark pages.

Evidence must receive stable evidence IDs such as `EV001`, `EV002`, and so on.

### `pdf`

Used for:

- reading papers,
- extracting methods,
- claims,
- assumptions,
- evaluation metrics,
- limitations,
- future work.

PDF outputs feed `paper_summaries/`, `claim_graph.json`, and `research_map.md`.

### `browser`

Used when `web_fetch` is insufficient:

- GitHub repositories,
- OpenReview pages,
- Papers with Code,
- leaderboards,
- dynamic benchmark pages,
- pages requiring JavaScript.

The browser is also useful during professor demos because the visible page proves the agent inspected real sources.

### `sessions_spawn`

Used to run specialist OpenClaw subagents in parallel:

- Literature Scout,
- Evidence Locker,
- Paper Reader,
- Gap Mapper,
- Novelty Tribunal,
- Reviewer Council,
- Experiment Planner,
- Venue Analyst.

The coordinator must merge results and keep one authoritative `run_state.json`.

### `skills`

OpenResearchOS is exposed as OpenClaw skills:

- `openresearchos-coordinator`
- `openresearchos-literature-scout`
- `openresearchos-evidence-locker`
- `openresearchos-reviewer-council`
- `openresearchos-experiment-runner`
- `openresearchos-paper-writer`

Each skill contains repeatable instructions and calls the local runner through OpenClaw `exec` when deterministic state updates are required.

### `exec`

Used for:

- creating run directories,
- updating state,
- generating experiment code,
- running local micro-probes,
- running probe experiments,
- collecting metrics,
- writing logs.

Execution is always tied to an experiment workspace and approval file.

### `memory/wiki`

Used for:

- durable lessons,
- failed ideas,
- reusable paper summaries,
- recurring reviewer criticisms,
- known benchmark constraints,
- source-grounded research notes.

Failed ideas are not deleted. They are stored because they improve future search.

### Telegram

Telegram is the first professional remote-control channel.

Expected commands:

```text
/start_research <topic>
/status <run_id>
/approve_micro_probe <run_id> <idea_id>
/reject <run_id> <idea_id>
/summarize <run_id>
/export_trace <run_id>
```

Security rules:

- private allowlist only,
- no public bot access,
- no experiment execution without explicit approval,
- no outbound messages to professors without human approval.

### WhatsApp

WhatsApp is second priority after Telegram works. It can mirror status updates and voice summaries, but it should not be the first integration because Telegram is easier to control safely.

### Sarvam STT/TTS

Sarvam gives the project a strong India-first research-agent angle.

Voice flow:

1. User sends speech in English, Hindi, or code-mixed language.
2. Sarvam STT converts speech to text.
3. OpenClaw interprets the research command.
4. OpenResearchOS updates the run.
5. Sarvam TTS speaks a short progress summary back.

Voice should be used for commands and summaries, not for hiding evidence. Research artifacts must still be written to files.

### Trajectory Export

Every serious demo must end with an OpenClaw trajectory export.

The export proves:

- which tools were used,
- which sources were read,
- which prompts and decisions happened,
- which experiments ran,
- where final artifacts were written.

## Agent Team

### Research Coordinator

Owns the full state machine.

Responsibilities:

- receives topic,
- creates run ID,
- routes subagents,
- updates state,
- checks gates,
- prevents unsupported paper drafting,
- prepares final package.

### Search Strategist

Builds search queries.

Responsibilities:

- recent-paper queries,
- foundational-paper queries,
- competitor queries,
- benchmark queries,
- venue queries,
- near-duplicate idea queries.

### Literature Scout

Finds papers, repos, datasets, and benchmarks.

Responsibilities:

- live search,
- OpenAlex/arXiv/Semantic Scholar style discovery,
- GitHub repo identification,
- OpenReview and Papers with Code lookup,
- source diversity.

### Evidence Locker

Freezes evidence.

Responsibilities:

- assigns evidence IDs,
- saves URLs,
- saves paper metadata,
- saves fetched text,
- stores access dates,
- flags unavailable PDFs.

### Paper Reader

Reads and summarizes papers.

Responsibilities:

- method extraction,
- claims,
- assumptions,
- datasets,
- metrics,
- limitations,
- future work,
- reusable paper summaries.

### Gap Mapper

Builds the research map.

Responsibilities:

- clusters papers,
- identifies limitations,
- maps disagreements,
- finds under-tested assumptions,
- proposes research gaps.

### Idea Tree Searcher

Generates many candidate ideas.

Responsibilities:

- generate 20 rough ideas,
- cluster duplicates,
- remove shallow ideas,
- keep 5 candidates,
- produce mechanisms and experiment sketches.

### Novelty Tribunal

Aggressively searches for prior art.

Responsibilities:

- near-identical idea search,
- competitor comparison,
- difference table,
- novelty risk score,
- rejection if idea is already solved.

### Reviewer Council

Simulates peer review.

Reviewer roles:

- Novelty Reviewer,
- Experimental Reviewer,
- Theory/Mechanism Reviewer,
- Reproducibility Reviewer,
- Venue Reviewer.

Each reviewer returns:

- score from 1 to 10,
- fatal flaws,
- fixable flaws,
- required experiments,
- missing related work,
- accept/revise/reject decision.

### Experiment Planner

Converts surviving ideas into local experiments.

Responsibilities:

- defines hypothesis,
- metric,
- baseline,
- ablation,
- dataset,
- runtime budget,
- expected result,
- failure interpretation.

### Experiment Runner

Runs approved experiments on the Mac.

Responsibilities:

- creates workspace,
- records environment,
- runs code,
- captures stdout/stderr,
- writes metrics,
- preserves failed runs.

### Paper Writer

Writes only after gates pass.

Responsibilities:

- paper-style draft,
- supported claims only,
- limitations,
- related work,
- experiment section,
- citation and evidence references.

If gates fail, this agent writes `research_failure_report.md`.

### Venue Analyst

Compares output against venues.

Responsibilities:

- workshop fit,
- conference fit,
- missing evidence,
- competitor comparison,
- submission checklist.

## Research Quality Gates

An idea advances only if it passes these gates:

### Evidence Gate

At least 10 relevant sources are reviewed, or a smaller corpus is explicitly justified.

### Freshness Gate

The source set must include recent work and foundational work.

### Prior-Art Gate

The system must explicitly search for near-identical ideas and competitors.

### Novelty Gate

The idea must have a clear difference from existing methods.

### Mechanism Gate

The idea must explain why it should work.

### Metric Gate

Success metrics must be defined before experiments.

### Baseline Gate

At least one baseline or ablation must be planned.

### Micro-Probe Gate

An early local test must produce useful signal when the topic is locally testable.

### Feasibility Gate

The experiment must be runnable locally, or the run must be marked `remote_compute_needed`.

### Reviewer Gate

No fatal reviewer flaw can remain.

### Result Gate

Experiment results must produce useful evidence, even if negative.

### Paper Gate

Final claims must link to evidence IDs or experiment logs.

## Experiment Ladder

The system runs experiments throughout thinking, not only at the end.

### Micro-Probe

Purpose:

- test idea signal during thinking,
- avoid wasting time on weak ideas.

Runtime:

- 1 to 5 minutes.

Data:

- synthetic data,
- toy data,
- tiny subset,
- simulation.

Output:

- early metric,
- failure mode,
- decision to promote, revise, or kill.

### Probe

Purpose:

- check if the idea deserves more work.

Runtime:

- 5 to 20 minutes.

Requirements:

- at least one baseline,
- fixed metric,
- fixed seed,
- useful metric.

### Ablation

Purpose:

- test which part of the method matters.

Runtime:

- 10 to 45 minutes.

Requirements:

- method variants,
- controlled comparison,
- clear interpretation.

### MVP Experiment

Purpose:

- produce paper-supporting evidence.

Runtime:

- 30 to 120 minutes.

Requirements:

- baseline,
- ablation,
- fixed seeds,
- metrics,
- plots,
- logs,
- reviewer critique.

Anything larger is marked `remote_compute_needed`.

## Local Experiment Workspace

Each experiment gets its own directory:

```text
openresearchos/runs/<run_id>/experiments/<experiment_id>/
```

Required files:

```text
experiment_spec.json
approval.json
environment.json
pyproject.toml or requirements.txt
src/
data/
outputs/
logs/stdout.log
logs/stderr.log
metrics.json
result_summary.md
review_after_run.md
```

`experiment_spec.json` contains:

- hypothesis,
- method,
- metric,
- dataset,
- baseline,
- ablation plan,
- expected result,
- failure interpretation,
- max runtime,
- max disk,
- max reruns,
- allowed network actions.

`approval.json` contains:

- approval status,
- approving human,
- timestamp,
- runtime budget,
- disk budget,
- dataset budget,
- allowed commands,
- disallowed commands.

`environment.json` contains:

- OS,
- hardware,
- Python version,
- package versions,
- uv version,
- MPS availability,
- command used.

## Local Experiment Execution Method

1. Planner writes `experiment_spec.json`.
2. Reviewer Council reviews the experiment plan.
3. Human approves the experiment.
4. OpenClaw `exec` creates or uses a `uv` environment.
5. Code runs with fixed seed and budget.
6. Logs are captured to `logs/stdout.log` and `logs/stderr.log`.
7. Metrics are written to `metrics.json`.
8. Result summary is written to `result_summary.md`.
9. Reviewer Council critiques the result.
10. Coordinator chooses exactly one next action.

## Experiment Safety Rules

- Human approval is required before running generated code.
- Untrusted GitHub repositories are inspected, not executed directly.
- Prefer minimal reimplementation over running unknown scripts.
- Every experiment has max runtime, max disk, max dataset size, and max reruns.
- Default max dataset download is 2 GB.
- Default max reruns per idea is 5.
- No huge SOTA claims from local toy runs.
- Large experiments become remote-compute plans.
- Failed experiments are preserved.

## Reviewer Simulation

Reviewer Council runs at least three times.

### Reviewer Pass 1: Idea Review

Attacks:

- novelty,
- mechanism,
- feasibility,
- experiment design,
- source support.

Possible decisions:

- `ACCEPT_TO_NEXT_STAGE`
- `REVISE_IDEA`
- `RUN_ADDITIONAL_SEARCH`
- `RUN_MICRO_PROBE`
- `KILL_IDEA`
- `MARK_REMOTE_COMPUTE_NEEDED`

### Reviewer Pass 2: Result Review

Attacks:

- metric choice,
- baseline weakness,
- ablation weakness,
- failure cases,
- overclaims,
- reproducibility.

Possible decisions:

- `PROMOTE_TO_NEXT_LEVEL`
- `ADD_BASELINE`
- `ADD_ABLATION`
- `FIX_BUG_AND_RERUN`
- `CHANGE_METRIC`
- `CHANGE_DATASET`
- `NARROW_CLAIM`
- `KILL_IDEA`
- `SEARCH_MORE_PRIOR_ART`
- `MARK_REMOTE_COMPUTE_NEEDED`

### Reviewer Pass 3: Paper Review

Attacks:

- unsupported claims,
- missing related work,
- weak experiments,
- bad venue fit,
- unclear contribution,
- missing limitations.

The system writes a paper draft only if no fatal issue remains.

## Iteration Policy

After every experiment, the system must choose exactly one next action:

- `PROMOTE_TO_NEXT_LEVEL`
- `ADD_BASELINE`
- `ADD_ABLATION`
- `FIX_BUG_AND_RERUN`
- `CHANGE_METRIC`
- `CHANGE_DATASET`
- `NARROW_CLAIM`
- `KILL_IDEA`
- `SEARCH_MORE_PRIOR_ART`
- `MARK_REMOTE_COMPUTE_NEEDED`

It cannot say only “promising” or “interesting.” Every result must change the next step.

## Paper Readiness Levels

- `RRL-0`: vague idea.
- `RRL-1`: evidence-backed gap.
- `RRL-2`: novelty survives prior-art search.
- `RRL-3`: micro-probe completed.
- `RRL-4`: probe with baseline completed.
- `RRL-5`: ablation or MVP evidence exists and reviewers find no fatal flaw.
- `RRL-6`: paper draft with supported claims.
- `RRL-7`: venue-specific checklist passed.

Only `RRL-5+` may produce a serious paper-style draft.

## Paper Draft Rule

A paper draft is allowed only if:

- average reviewer score is at least 6.0,
- no fatal novelty flaw remains,
- no fatal experiment flaw remains,
- at least one reproducible experiment exists,
- all major claims have evidence IDs or experiment logs,
- competitor comparison is complete,
- limitations are explicit.

If these conditions fail, the system writes `research_failure_report.md`.

## Deliverables For Each Run

Required artifacts:

```text
run_state.json
search_plan.md
evidence_index.json
paper_summaries/
claim_graph.json
research_map.md
idea_tree.md
reviewer_pass_1.md
novelty_tribunal.md
micro_probe_plan.md
micro_probe_results.md
experiment_plan.md
experiment_logs/
metrics.json
reviewer_pass_2.md
revision_plan.md
paper_readiness_review.md
venue_fit.md
final_report.md
paper_draft.md
research_failure_report.md
trajectory_export/
```

`paper_draft.md` is created only when gates pass.

`research_failure_report.md` is created when gates fail.

## Professor Demo Flow

The professor demo should be short, concrete, and auditable.

1. User sends a topic on Telegram.
2. OpenClaw starts the OpenResearchOS coordinator skill.
3. Agent scopes topic and classifies feasibility.
4. Subagents search recent and foundational papers.
5. Evidence Locker saves URLs, PDFs, and fetched snapshots.
6. Paper Reader builds claims and limitations.
7. Gap Mapper creates `research_map.md`.
8. Idea Tree Searcher generates many ideas.
9. Reviewer Council kills weak ideas.
10. Novelty Tribunal searches competitors.
11. Agent proposes a micro-probe.
12. User approves.
13. Micro-probe runs locally on the Mac.
14. Reviewer Council critiques the result.
15. Agent revises, adds baseline, or kills the idea.
16. Strong idea gets promoted to probe or MVP experiment.
17. Final reviewer checks paper readiness.
18. If ready, agent drafts paper-style report.
19. If weak, agent writes failure report.
20. Venue Analyst explains workshop/conference fit.
21. OpenClaw trajectory export proves the process.

## Suggested First Demo Topic

Use a topic that is feasible locally and aligned with Professor Xie’s project:

```text
traceable autonomous research agents that use reviewer simulation and micro-experiments to reduce weak research ideas
```

This is stronger than starting with a huge healthcare foundation-model topic because it can be tested on the Mac quickly while still matching AI-Build-AI and research-agent work.

After the system works, it can be applied to healthcare foundation models, active learning, calibration, and reliable biomedical agents.

## Implementation Phases

### Phase 1: OpenClaw Skill Scaffold

Create skills:

- coordinator,
- literature scout,
- evidence locker,
- reviewer council,
- experiment runner,
- paper writer.

Register them through OpenClaw `skills.load.extraDirs`.

Success condition:

- OpenClaw lists the OpenResearchOS skills.

### Phase 2: Deterministic Local Runner

Build a local CLI that creates runs and artifacts:

```bash
node openresearchos/src/openresearch.mjs start --topic "<topic>"
node openresearchos/src/openresearch.mjs discover --run <run_id>
node openresearchos/src/openresearch.mjs read --run <run_id>
node openresearchos/src/openresearch.mjs map --run <run_id>
node openresearchos/src/openresearch.mjs ideas --run <run_id>
node openresearchos/src/openresearch.mjs approve --run <run_id> --idea <idea_id>
node openresearchos/src/openresearch.mjs run-micro-probe --run <run_id>
node openresearchos/src/openresearch.mjs write --run <run_id>
node openresearchos/src/openresearch.mjs status --run <run_id>
```

Success condition:

- one command can run a full local demo and produce all required artifacts.

### Phase 3: Live OpenClaw Research Workflow

Make the coordinator skill use OpenClaw tools:

- `web_search`,
- `web_fetch`,
- `pdf`,
- `browser`,
- `sessions_spawn`,
- `exec`,
- `memory/wiki`.

Success condition:

- run artifacts cite real evidence gathered by OpenClaw.

### Phase 4: Local Experiment Loop

Implement micro-probe and probe workspaces.

Success condition:

- every experiment has spec, approval, environment, source, logs, metrics, result summary, and review.

### Phase 5: Telegram And Sarvam Voice

Add command entry and spoken summaries.

Success condition:

- a Telegram command starts a run,
- Sarvam can transcribe a voice command,
- Sarvam can speak a concise run status.

### Phase 6: Professor Package

Create:

- demo script,
- technical note,
- sample completed run,
- trajectory export,
- final report.

Success condition:

- Professor Xie can understand the system without live explanation.

## Acceptance Criteria

- The system never writes a paper from a one-pass idea.
- The system runs experiments during idea development, not only at the end.
- Every final claim links to evidence or experiment logs.
- Every experiment has config, command, seed, metrics, stdout, stderr.
- Failed experiments are preserved.
- Reviewer criticism changes the next action.
- Unsupported claims are blocked or labeled speculative.
- Telegram access is not public.
- The demo can resume after interruption.
- Final package is understandable to Professor Xie without live explanation.

## Non-Goals For First MVP

- No claim of guaranteed A* conference acceptance.
- No large foundation-model training on the Mac.
- No execution of untrusted GitHub repos.
- No public Telegram bot.
- No automatic email or message to Professor Xie without human approval.
- No unsupported healthcare or clinical claims.

## Honest Positioning

OpenResearchOS should be presented as:

> A traceable OpenClaw-based research operating system that forces autonomous research agents through evidence, novelty, review, and experiment gates before writing.

It should not be presented as:

> An agent that automatically produces A* papers.

The strongest claim is process quality and traceability:

- serious literature workflow,
- repeatable reviewer simulation,
- local experiment loop,
- refusal to overclaim,
- complete audit trail.

That is credible, impressive, and aligned with AI-Build-AI.
