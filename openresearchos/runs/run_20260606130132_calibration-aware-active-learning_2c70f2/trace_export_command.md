# OpenClaw Trajectory Export

Run ID: `run_20260606130132_calibration-aware-active-learning_2c70f2`

Session Key: `openresearchos-run_20260606130132_calibration-aware-active-learning_2c70f2`

Export Status: session not found — run through OpenClaw agent for full trajectory capture.

## How To Export

```bash
openclaw sessions export-trajectory \
  --session-key "openresearchos-run_20260606130132_calibration-aware-active-learning_2c70f2" \
  --output "/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260606130132_calibration-aware-active-learning_2c70f2/trajectory_export"
```

## Or run the full pipeline through OpenClaw:

```bash
openclaw agent --message "Resume OpenResearchOS run run_20260606130132_calibration-aware-active-learning_2c70f2 and write the final report" \
  --session-key "openresearchos-run_20260606130132_calibration-aware-active-learning_2c70f2"
```
