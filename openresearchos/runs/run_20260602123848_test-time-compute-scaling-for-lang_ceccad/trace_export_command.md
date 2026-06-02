# OpenClaw Trajectory Export

Run ID: `run_20260602123848_test-time-compute-scaling-for-lang_ceccad`

Session Key: `openresearchos-run_20260602123848_test-time-compute-scaling-for-lang_ceccad`

Export Status: session not found — run through OpenClaw agent for full trajectory capture.

## How To Export

```bash
openclaw sessions export-trajectory \
  --session-key "openresearchos-run_20260602123848_test-time-compute-scaling-for-lang_ceccad" \
  --output "/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260602123848_test-time-compute-scaling-for-lang_ceccad/trajectory_export"
```

## Or run the full pipeline through OpenClaw:

```bash
openclaw agent --message "Resume OpenResearchOS run run_20260602123848_test-time-compute-scaling-for-lang_ceccad and write the final report" \
  --session-key "openresearchos-run_20260602123848_test-time-compute-scaling-for-lang_ceccad"
```
