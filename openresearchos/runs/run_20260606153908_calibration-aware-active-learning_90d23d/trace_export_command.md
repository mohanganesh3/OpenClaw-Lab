# OpenClaw Trajectory Export

Run ID: `run_20260606153908_calibration-aware-active-learning_90d23d`

Session Key: `openresearchos-run_20260606153908_calibration-aware-active-learning_90d23d`

Export Status: session not found — run through OpenClaw agent for full trajectory capture.

## How To Export

```bash
openclaw sessions export-trajectory \
  --session-key "openresearchos-run_20260606153908_calibration-aware-active-learning_90d23d" \
  --output "/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260606153908_calibration-aware-active-learning_90d23d/trajectory_export"
```

## Or run the full pipeline through OpenClaw:

```bash
openclaw agent --message "Resume OpenResearchOS run run_20260606153908_calibration-aware-active-learning_90d23d and write the final report" \
  --session-key "openresearchos-run_20260606153908_calibration-aware-active-learning_90d23d"
```
