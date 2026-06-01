# OpenResearchOS

OpenResearchOS is an OpenClaw-native research operating system for traceable autonomous research loops.

It implements the high-bar plan in:

```text
docs/OPENRESEARCHOS_HIGH_BAR_PLAN.md
```

## Architecture

OpenClaw provides the agent layer:

- skills,
- web search,
- web fetch,
- PDF reading,
- browser inspection,
- subagents,
- local `exec`,
- memory/wiki,
- Telegram,
- Sarvam voice,
- trajectory export.

The local runner provides deterministic state:

- run directories,
- `run_state.json`,
- evidence index,
- claim graph,
- idea tree,
- reviewer scores,
- experiment specs,
- approvals,
- stdout/stderr logs,
- metrics,
- readiness reports.

## Commands

Run from this directory:

```bash
node src/openresearch.mjs demo --topic "traceable autonomous research agents that use reviewer simulation and micro-experiments to reduce weak research ideas"
```

Individual stages:

```bash
node src/openresearch.mjs start --topic "<topic>"
node src/openresearch.mjs discover --run <run_id>
node src/openresearch.mjs read --run <run_id>
node src/openresearch.mjs map --run <run_id>
node src/openresearch.mjs ideas --run <run_id>
node src/openresearch.mjs approve --run <run_id> --level micro_probe --idea <idea_id>
node src/openresearch.mjs run-micro-probe --run <run_id>
node src/openresearch.mjs run-probe --run <run_id>
node src/openresearch.mjs run-ablation --run <run_id>
node src/openresearch.mjs run-mvp --run <run_id>
node src/openresearch.mjs write --run <run_id>
node src/openresearch.mjs status --run <run_id>
node src/openresearch.mjs verify --run <run_id>
node src/openresearch.mjs voice-summary --run <run_id> --target-language en-IN
node src/openresearch.mjs voice-transcribe --file <audio.wav>
```

Voice note/status support uses Sarvam through the local CLI provider. The Chrome Control UI `Start Talk` button is a separate browser realtime Talk path; in OpenClaw 2026.5.28 it requires a configured realtime voice provider and does not use the Sarvam local STT/TTS loop.

Smoke test:

```bash
npm run smoke
```

The smoke test uses fallback seed evidence so it can run even when literature APIs are unavailable.

## OpenClaw Skill Registration

Register this directory as an OpenClaw skill root:

```bash
openclaw config patch --stdin <<'JSON'
{
  "skills": {
    "load": {
      "extraDirs": [
        "/Users/mohanganesh/OpenClaw-Lab/openresearchos/skills"
      ]
    }
  }
}
JSON
```

Then restart the gateway:

```bash
openclaw gateway restart
```

Verify:

```bash
openclaw skills list --json
```

Expected skill names:

- `openresearchos-coordinator`
- `openresearchos-literature-scout`
- `openresearchos-evidence-locker`
- `openresearchos-reviewer-council`
- `openresearchos-experiment-runner`
- `openresearchos-paper-writer`

## Run Artifacts

Each run is written to:

```text
runs/<run_id>/
```

Key artifacts:

- `run_state.json`
- `search_plan.md`
- `evidence_index.json`
- `paper_summaries/`
- `claim_graph.json`
- `research_map.md`
- `idea_tree.md`
- `reviewer_pass_1.md`
- `novelty_tribunal.md`
- `micro_probe_plan.md`
- `micro_probe_results.md`
- `experiment_plan.md`
- `experiment_logs/`
- `metrics.json`
- `reviewer_pass_2.md`
- `revision_plan.md`
- `paper_readiness_review.md`
- `venue_fit.md`
- `final_report.md`
- `paper_draft.md` only if gates pass
- `research_failure_report.md` if gates fail
- `trace_export_command.md`

## Safety

- Human approval is required before experiment execution.
- Untrusted repositories are inspected, not executed.
- Local experiments use generated, minimal code and synthetic or approved small data.
- Large experiments are marked `remote_compute_needed`.
- Unsupported claims produce a failure report, not a fake paper.
