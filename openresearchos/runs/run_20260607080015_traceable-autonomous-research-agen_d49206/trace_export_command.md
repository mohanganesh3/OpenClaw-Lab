# OpenClaw Trajectory Export

Run ID: `run_20260607080015_traceable-autonomous-research-agen_d49206`

Session Key: `openresearchos-run_20260607080015_traceable-autonomous-research-agen_d49206`

Export Status: session not found — run through OpenClaw agent for full trajectory capture.

## How To Export

```bash
openclaw sessions export-trajectory \
  --session-key "openresearchos-run_20260607080015_traceable-autonomous-research-agen_d49206" \
  --output "/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260607080015_traceable-autonomous-research-agen_d49206/trajectory_export"
```

## Or run the full pipeline through OpenClaw:

```bash
openclaw agent --message "Resume OpenResearchOS run run_20260607080015_traceable-autonomous-research-agen_d49206 and write the final report" \
  --session-key "openresearchos-run_20260607080015_traceable-autonomous-research-agen_d49206"
```
