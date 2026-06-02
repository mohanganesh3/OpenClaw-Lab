# OpenClaw Trajectory Export

Run ID: `run_20260602163945_continual-learning-catastrophic-fo_c9523c`

Session Key: `openresearchos-run_20260602163945_continual-learning-catastrophic-fo_c9523c`

Export Status: session not found — run through OpenClaw agent for full trajectory capture.

## How To Export

```bash
openclaw sessions export-trajectory \
  --session-key "openresearchos-run_20260602163945_continual-learning-catastrophic-fo_c9523c" \
  --output "/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260602163945_continual-learning-catastrophic-fo_c9523c/trajectory_export"
```

## Or run the full pipeline through OpenClaw:

```bash
openclaw agent --message "Resume OpenResearchOS run run_20260602163945_continual-learning-catastrophic-fo_c9523c and write the final report" \
  --session-key "openresearchos-run_20260602163945_continual-learning-catastrophic-fo_c9523c"
```
