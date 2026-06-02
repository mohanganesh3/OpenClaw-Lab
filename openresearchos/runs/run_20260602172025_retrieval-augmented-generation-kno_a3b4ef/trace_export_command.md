# OpenClaw Trajectory Export

Run ID: `run_20260602172025_retrieval-augmented-generation-kno_a3b4ef`

Session Key: `openresearchos-run_20260602172025_retrieval-augmented-generation-kno_a3b4ef`

Export Status: session not found — run through OpenClaw agent for full trajectory capture.

## How To Export

```bash
openclaw sessions export-trajectory \
  --session-key "openresearchos-run_20260602172025_retrieval-augmented-generation-kno_a3b4ef" \
  --output "/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260602172025_retrieval-augmented-generation-kno_a3b4ef/trajectory_export"
```

## Or run the full pipeline through OpenClaw:

```bash
openclaw agent --message "Resume OpenResearchOS run run_20260602172025_retrieval-augmented-generation-kno_a3b4ef and write the final report" \
  --session-key "openresearchos-run_20260602172025_retrieval-augmented-generation-kno_a3b4ef"
```
