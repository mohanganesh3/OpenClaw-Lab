# OpenClaw Trajectory Export

Run ID: `run_20260601153719_traceable-autonomous-research-agen_2b78cd`

Session Key: `openresearchos-run_20260601153719_traceable-autonomous-research-agen_2b78cd`

Export Status: session not found — run through OpenClaw agent for full trajectory capture.

## How To Export

```bash
openclaw sessions export-trajectory \
  --session-key "openresearchos-run_20260601153719_traceable-autonomous-research-agen_2b78cd" \
  --output "/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260601153719_traceable-autonomous-research-agen_2b78cd/trajectory_export"
```

## Or run the full pipeline through OpenClaw:

```bash
openclaw agent --message "Resume OpenResearchOS run run_20260601153719_traceable-autonomous-research-agen_2b78cd and write the final report" \
  --session-key "openresearchos-run_20260601153719_traceable-autonomous-research-agen_2b78cd"
```
