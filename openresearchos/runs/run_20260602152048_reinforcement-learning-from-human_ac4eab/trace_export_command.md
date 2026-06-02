# OpenClaw Trajectory Export

Run ID: `run_20260602152048_reinforcement-learning-from-human_ac4eab`

Session Key: `openresearchos-run_20260602152048_reinforcement-learning-from-human_ac4eab`

Export Status: session not found — run through OpenClaw agent for full trajectory capture.

## How To Export

```bash
openclaw sessions export-trajectory \
  --session-key "openresearchos-run_20260602152048_reinforcement-learning-from-human_ac4eab" \
  --output "/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260602152048_reinforcement-learning-from-human_ac4eab/trajectory_export"
```

## Or run the full pipeline through OpenClaw:

```bash
openclaw agent --message "Resume OpenResearchOS run run_20260602152048_reinforcement-learning-from-human_ac4eab and write the final report" \
  --session-key "openresearchos-run_20260602152048_reinforcement-learning-from-human_ac4eab"
```
