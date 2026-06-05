# OpenClaw Trajectory Export

Run ID: `run_20260604170141_efficient-attention-mechanisms_08c60d`

Session Key: `openresearchos-run_20260604170141_efficient-attention-mechanisms_08c60d`

Export Status: session not found — run through OpenClaw agent for full trajectory capture.

## How To Export

```bash
openclaw sessions export-trajectory \
  --session-key "openresearchos-run_20260604170141_efficient-attention-mechanisms_08c60d" \
  --output "/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260604170141_efficient-attention-mechanisms_08c60d/trajectory_export"
```

## Or run the full pipeline through OpenClaw:

```bash
openclaw agent --message "Resume OpenResearchOS run run_20260604170141_efficient-attention-mechanisms_08c60d and write the final report" \
  --session-key "openresearchos-run_20260604170141_efficient-attention-mechanisms_08c60d"
```
