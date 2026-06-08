# OpenClaw Trajectory Export

Run ID: `run_20260608104616_uncertainty-aware-pseudo-label-sel_3e0902`

Session Key: `openresearchos-run_20260608104616_uncertainty-aware-pseudo-label-sel_3e0902`

Export Status: session not found — run through OpenClaw agent for full trajectory capture.

## How To Export

```bash
openclaw sessions export-trajectory \
  --session-key "openresearchos-run_20260608104616_uncertainty-aware-pseudo-label-sel_3e0902" \
  --output "/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260608104616_uncertainty-aware-pseudo-label-sel_3e0902/trajectory_export"
```

## Or run the full pipeline through OpenClaw:

```bash
openclaw agent --message "Resume OpenResearchOS run run_20260608104616_uncertainty-aware-pseudo-label-sel_3e0902 and write the final report" \
  --session-key "openresearchos-run_20260608104616_uncertainty-aware-pseudo-label-sel_3e0902"
```
