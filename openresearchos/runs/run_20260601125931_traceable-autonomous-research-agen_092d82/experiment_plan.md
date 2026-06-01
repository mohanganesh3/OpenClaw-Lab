# MVP Experiment Plan

Experiment: `mvp_id010`

Idea: `ID010` Human Approval Gates for Agent Experiments for traceable autonomous research

Hypothesis: Human Approval Gates for Agent Experiments for traceable autonomous research will improve guided idea selection utility over a random baseline under a controlled synthetic research-loop simulation.

Metric: `guided_selection_utility_minus_random_baseline`

Baseline: random idea selection

Runtime Budget: 30-120 minutes

Workspace:

```text
/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260601125931_traceable-autonomous-research-agen_092d82/experiments/mvp_id010
```

Approval:

```json
{
  "approved": true,
  "approved_by": "local_user",
  "approval_note": "Approved before execution.",
  "approved_at": "2026-06-01T13:04:47.935Z",
  "budget": {
    "runtime_minutes": "30-120",
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
