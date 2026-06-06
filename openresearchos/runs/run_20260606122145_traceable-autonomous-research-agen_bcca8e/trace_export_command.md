# OpenClaw Trajectory Export

Run ID: `run_20260606122145_traceable-autonomous-research-agen_bcca8e`

Session Key: `openresearchos-run_20260606122145_traceable-autonomous-research-agen_bcca8e`

Export Status: session not found — run through OpenClaw agent for full trajectory capture.

## How To Export

```bash
openclaw sessions export-trajectory \
  --session-key "openresearchos-run_20260606122145_traceable-autonomous-research-agen_bcca8e" \
  --output "/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260606122145_traceable-autonomous-research-agen_bcca8e/trajectory_export"
```

## Or run the full pipeline through OpenClaw:

```bash
openclaw agent --message "Resume OpenResearchOS run run_20260606122145_traceable-autonomous-research-agen_bcca8e and write the final report" \
  --session-key "openresearchos-run_20260606122145_traceable-autonomous-research-agen_bcca8e"
```
