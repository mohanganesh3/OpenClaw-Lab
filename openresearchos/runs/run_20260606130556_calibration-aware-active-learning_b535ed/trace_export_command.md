# OpenClaw Trajectory Export

Run ID: `run_20260606130556_calibration-aware-active-learning_b535ed`

Session Key: `openresearchos-run_20260606130556_calibration-aware-active-learning_b535ed`

Export Status: session not found — run through OpenClaw agent for full trajectory capture.

## How To Export

```bash
openclaw sessions export-trajectory \
  --session-key "openresearchos-run_20260606130556_calibration-aware-active-learning_b535ed" \
  --output "/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260606130556_calibration-aware-active-learning_b535ed/trajectory_export"
```

## Or run the full pipeline through OpenClaw:

```bash
openclaw agent --message "Resume OpenResearchOS run run_20260606130556_calibration-aware-active-learning_b535ed and write the final report" \
  --session-key "openresearchos-run_20260606130556_calibration-aware-active-learning_b535ed"
```
