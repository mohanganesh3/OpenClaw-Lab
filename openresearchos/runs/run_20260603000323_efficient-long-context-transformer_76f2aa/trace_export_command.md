# OpenClaw Trajectory Export

Run ID: `run_20260603000323_efficient-long-context-transformer_76f2aa`

Session Key: `openresearchos-run_20260603000323_efficient-long-context-transformer_76f2aa`

Export Status: session not found — run through OpenClaw agent for full trajectory capture.

## How To Export

```bash
openclaw sessions export-trajectory \
  --session-key "openresearchos-run_20260603000323_efficient-long-context-transformer_76f2aa" \
  --output "/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260603000323_efficient-long-context-transformer_76f2aa/trajectory_export"
```

## Or run the full pipeline through OpenClaw:

```bash
openclaw agent --message "Resume OpenResearchOS run run_20260603000323_efficient-long-context-transformer_76f2aa and write the final report" \
  --session-key "openresearchos-run_20260603000323_efficient-long-context-transformer_76f2aa"
```
