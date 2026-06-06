# OpenClaw Trajectory Export

Run ID: `run_20260606131917_calibration-aware-active-learning_30f1c3`

Session Key: `openresearchos-run_20260606131917_calibration-aware-active-learning_30f1c3`

Export Status: session not found — run through OpenClaw agent for full trajectory capture.

## How To Export

```bash
openclaw sessions export-trajectory \
  --session-key "openresearchos-run_20260606131917_calibration-aware-active-learning_30f1c3" \
  --output "/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260606131917_calibration-aware-active-learning_30f1c3/trajectory_export"
```

## Or run the full pipeline through OpenClaw:

```bash
openclaw agent --message "Resume OpenResearchOS run run_20260606131917_calibration-aware-active-learning_30f1c3 and write the final report" \
  --session-key "openresearchos-run_20260606131917_calibration-aware-active-learning_30f1c3"
```
