# Professor Demo Script

## Goal

Show that OpenResearchOS is a serious OpenClaw-based research loop, not a one-shot idea generator.

## Suggested Topic

```text
traceable autonomous research agents that use reviewer simulation and micro-experiments to reduce weak research ideas
```

## Demo Steps

1. Open OpenClaw and select the OpenResearchOS coordinator skill.
2. Start a run:

```bash
node src/openresearch.mjs start --topic "traceable autonomous research agents that use reviewer simulation and micro-experiments to reduce weak research ideas"
```

3. Show `run_state.json` and `search_plan.md`.
4. Run discovery and evidence locking:

```bash
node src/openresearch.mjs discover --run <run_id>
node src/openresearch.mjs read --run <run_id>
node src/openresearch.mjs map --run <run_id>
```

5. Show:

- `evidence_index.json`,
- `paper_summaries/`,
- `claim_graph.json`,
- `research_map.md`.

6. Generate ideas and reviewer pass:

```bash
node src/openresearch.mjs ideas --run <run_id>
```

7. Show:

- `idea_tree.md`,
- `reviewer_pass_1.md`,
- `novelty_tribunal.md`.

8. Approve and run a micro-probe:

```bash
node src/openresearch.mjs approve --run <run_id> --level micro_probe --idea <idea_id>
node src/openresearch.mjs run-micro-probe --run <run_id>
```

9. Show experiment workspace:

```text
runs/<run_id>/experiments/micro_probe_<idea_id>/
```

Highlight:

- `experiment_spec.json`,
- `approval.json`,
- `environment.json`,
- `src/research_probe.py`,
- `logs/stdout.log`,
- `logs/stderr.log`,
- `metrics.json`,
- `result_summary.md`,
- `review_after_run.md`.

10. Write final readiness report:

```bash
node src/openresearch.mjs write --run <run_id>
```

11. Show:

- `paper_readiness_review.md`,
- `venue_fit.md`,
- `final_report.md`,
- `research_failure_report.md` or `paper_draft.md`.

12. Export OpenClaw trajectory:

```text
/export-trajectory <run_id>
```

## Demo Message

The strongest sentence to say:

> This is not claiming to automatically produce A* papers. It is a traceable OpenClaw research operating system that forces an autonomous agent through evidence, novelty, reviewer, experiment, and paper-readiness gates before it is allowed to write.

