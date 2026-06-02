# OpenClaw Trajectory Export

Run ID: `run_20260602132228_mixture-of-experts-routing-strateg_f176cc`

Session Key: `openresearchos-run_20260602132228_mixture-of-experts-routing-strateg_f176cc`

Export Status: session not found — run through OpenClaw agent for full trajectory capture.

## How To Export

```bash
openclaw sessions export-trajectory \
  --session-key "openresearchos-run_20260602132228_mixture-of-experts-routing-strateg_f176cc" \
  --output "/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260602132228_mixture-of-experts-routing-strateg_f176cc/trajectory_export"
```

## Or run the full pipeline through OpenClaw:

```bash
openclaw agent --message "Resume OpenResearchOS run run_20260602132228_mixture-of-experts-routing-strateg_f176cc and write the final report" \
  --session-key "openresearchos-run_20260602132228_mixture-of-experts-routing-strateg_f176cc"
```
