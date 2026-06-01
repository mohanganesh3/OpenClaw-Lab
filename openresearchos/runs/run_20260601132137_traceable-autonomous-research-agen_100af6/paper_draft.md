# Draft: Traceable Reviewer-Gated Research Loops For Autonomous Research Agents

## Abstract

We present an OpenClaw-based research operating system that connects literature evidence, idea search, reviewer simulation, local experiment probes, and paper-readiness gates. The system is evaluated only within the logged local experiment setting for run `run_20260601132137_traceable-autonomous-research-agen_100af6`; claims are intentionally limited to this evidence.

## Contribution

1. A resumable state machine for autonomous research runs. Supported by `run_state.json`, `search_plan.md`, and evidence IDs EV001, EV002, EV003.
2. Reviewer-gated idea promotion before experiment spending. Supported by `reviewer_pass_1.md`, `novelty_tribunal.md`, and 20 reviewer score records.
3. A local experiment ladder that tests ideas during ideation. Supported by `micro_probe:micro_probe_id010`, `probe:probe_id010`, `ablation:ablation_id010`, `mvp:mvp_id010`.
4. Evidence-linked final reporting that refuses unsupported paper drafts. Supported by `paper_readiness_review.md`, `venue_fit.md`, and the exclusive paper/failure output gate.

## Evidence

Evidence IDs: EV001, EV002, EV003, EV004, EV005, EV006, EV007, EV008, EV009, EV010

## Selected Idea

ID010: Human Approval Gates for Agent Experiments for traceable autonomous research

## Results

Micro-probes: 1

Probes: 1

Ablations: 1

MVP experiments: 1

## Experiment Evidence

- micro_probe `micro_probe_id010`: metrics `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260601132137_traceable-autonomous-research-agen_100af6/experiments/micro_probe_id010/metrics.json`; review `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260601132137_traceable-autonomous-research-agen_100af6/experiments/micro_probe_id010/review_after_run.md`; logs `experiments/micro_probe_id010/logs/stdout.log` and `experiments/micro_probe_id010/logs/stderr.log`.
- probe `probe_id010`: metrics `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260601132137_traceable-autonomous-research-agen_100af6/experiments/probe_id010/metrics.json`; review `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260601132137_traceable-autonomous-research-agen_100af6/experiments/probe_id010/review_after_run.md`; logs `experiments/probe_id010/logs/stdout.log` and `experiments/probe_id010/logs/stderr.log`.
- ablation `ablation_id010`: metrics `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260601132137_traceable-autonomous-research-agen_100af6/experiments/ablation_id010/metrics.json`; review `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260601132137_traceable-autonomous-research-agen_100af6/experiments/ablation_id010/review_after_run.md`; logs `experiments/ablation_id010/logs/stdout.log` and `experiments/ablation_id010/logs/stderr.log`.
- mvp `mvp_id010`: metrics `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260601132137_traceable-autonomous-research-agen_100af6/experiments/mvp_id010/metrics.json`; review `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260601132137_traceable-autonomous-research-agen_100af6/experiments/mvp_id010/review_after_run.md`; logs `experiments/mvp_id010/logs/stdout.log` and `experiments/mvp_id010/logs/stderr.log`.

## Competitor And Novelty Evidence

- Competitor comparison is recorded in `novelty_tribunal.md`.
- Source-grounded paper summaries are recorded in `paper_summaries/`.
- Claim links are recorded in `claim_graph.json`.

Readiness: `RRL-5`

## Limitations

- Local synthetic experiment evidence does not establish domain SOTA.
- Full novelty still requires OpenClaw browser/pdf review of near competitors.
- Human approval remains required for stronger experiments and final claims.
