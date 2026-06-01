# Micro-Probe Plan

Experiment: `micro_probe_id004`

Idea: `ID004` Failure Memory for Research Agents for traceable autonomous research

Hypothesis: Failure Memory for Research Agents for traceable autonomous research will improve guided idea selection utility over a random baseline under a controlled synthetic research-loop simulation.

Metric: `guided_selection_utility_minus_random_baseline`

Baseline: random idea selection

Runtime Budget: 1-5 minutes

Workspace:

```text
/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260601143610_traceable-autonomous-research-agen_198fff/experiments/micro_probe_id004
```

Approval:

```json
{
  "approved": true,
  "approved_by": "demo_user",
  "approval_note": "Approved before execution.",
  "approved_at": "2026-06-01T14:43:52.663Z",
  "budget": {
    "runtime_minutes": "1-5",
    "max_disk_mb": 250,
    "max_dataset_download_gb": 0,
    "max_reruns": 5
  },
  "allowed_commands": [
    "uv run python src/research_probe.py",
    "python3 src/research_probe.py"
  ],
  "disallowed_commands": [
    "rm -rf",
    "curl | sh",
    "executing untrusted repositories"
  ]
}
```
