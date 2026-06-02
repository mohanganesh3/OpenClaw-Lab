# OpenClaw Trajectory Export

Run ID: `run_20260602054449_agent-reliability-in-healthcare-ai_9d2c5f`

Session Key: `openresearchos-run_20260602054449_agent-reliability-in-healthcare-ai_9d2c5f`

Export Status: session not found — run through OpenClaw agent for full trajectory capture.

## How To Export

```bash
openclaw sessions export-trajectory \
  --session-key "openresearchos-run_20260602054449_agent-reliability-in-healthcare-ai_9d2c5f" \
  --output "/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260602054449_agent-reliability-in-healthcare-ai_9d2c5f/trajectory_export"
```

## Or run the full pipeline through OpenClaw:

```bash
openclaw agent --message "Resume OpenResearchOS run run_20260602054449_agent-reliability-in-healthcare-ai_9d2c5f and write the final report" \
  --session-key "openresearchos-run_20260602054449_agent-reliability-in-healthcare-ai_9d2c5f"
```
