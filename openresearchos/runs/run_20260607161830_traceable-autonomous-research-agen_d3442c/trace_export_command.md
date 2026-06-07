# OpenClaw Trajectory Export

Run ID: `run_20260607161830_traceable-autonomous-research-agen_d3442c`

Session Key: `openresearchos-run_20260607161830_traceable-autonomous-research-agen_d3442c`

Export Status: session not found — run through OpenClaw agent for full trajectory capture.

## How To Export

```bash
openclaw sessions export-trajectory \
  --session-key "openresearchos-run_20260607161830_traceable-autonomous-research-agen_d3442c" \
  --output "/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260607161830_traceable-autonomous-research-agen_d3442c/trajectory_export"
```

## Or run the full pipeline through OpenClaw:

```bash
openclaw agent --message "Resume OpenResearchOS run run_20260607161830_traceable-autonomous-research-agen_d3442c and write the final report" \
  --session-key "openresearchos-run_20260607161830_traceable-autonomous-research-agen_d3442c"
```
