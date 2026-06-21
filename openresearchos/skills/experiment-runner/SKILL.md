---
name: openresearchos-experiment-runner
description: Plan, approve, run, and review OpenResearchOS experiments on Mac M4 using OpenClaw exec tool with human approval gates and memory persistence.
---

# OpenResearchOS Experiment Runner

Use this skill for all local experiment execution. Every experiment runs through OpenClaw's `exec` tool with explicit human approval gates.

## Non-Negotiable Safety Rules

> [!CAUTION]
> **Human Approval is MANDATORY**: Always show the experiment code to the user and get explicit "yes" before running anything.
> **Memory check first**: Always run `openclaw memory search` before generating new experiments to avoid repeating failed approaches.

## Pre-Experiment Memory Check (OpenClaw Native)

Before every experiment, check if this approach has already failed:

```bash
openclaw memory search \
  --query "<idea_title> <mechanism_keywords> failed experiment" \
  --max-results 10 --json
```

If results have score > 0.6, report the past failure to the user and ask whether to proceed.

## Experiment Levels

| Level | Runtime | Purpose |
|---|---|---|
| `micro_probe` | 1-5 min | Signal test — does the mechanism show any improvement over random? |
| `probe` | 5-20 min | Baseline-backed test with larger sample size |
| `ablation` | 10-45 min | Remove individual components to quantify each contribution |
| `mvp` | 30-120 min | Full experiment with real dataset (conference-level evidence) |

## Full Workflow

### Step 1: Plan

```bash
node src/openresearch.mjs approve \
  --run <run_id> \
  --level <level> \
  --idea <idea_id> \
  --by "<username>"
```

This generates the Python experiment code and writes it to the run workspace.

### Step 2: Show Code to User (REQUIRED)

Read the generated code and present it to the user:
```bash
cat runs/<run_id>/experiments/<idea_id>_<level>/src/research_probe.py
```

Do NOT proceed until the user explicitly approves.

### Step 3: Run via exec (After Approval)

```bash
node src/openresearch.mjs run-experiment \
  --run <run_id> \
  --idea <idea_id> \
  --level <level>
```

The `exec` tool handles:
- Python virtual environment setup via `uv`
- Package installation (numpy, scikit-learn, matplotlib)
- Fixed random seed (1729) for reproducibility
- Metric capture to `metrics.json`
- Plot generation to `outputs/`

### Step 4: Review Results

```bash
cat runs/<run_id>/experiments/<idea_id>_<level>/metrics.json
cat runs/<run_id>/reviewer_pass_2.md
```

### Step 5: Persist Results to OpenClaw Memory

After every experiment (success OR failure):

```bash
# OpenClaw memory is updated automatically by the bridge, but verify:
openclaw memory search --query "<idea_id> experiment result" --json
```

If failed, the idea is automatically written to `~/.openclaw/workspace/memory/failed_ideas/`.

### Step 6: Promote or Kill

- **Improvement > threshold and no fatal flaw** → run next level:
  ```bash
  node src/openresearch.mjs approve --run <run_id> --level probe --idea <idea_id> --by "auto_promote"
  ```
- **Improvement below threshold** → stop, write failure report:
  ```bash
  node src/openresearch.mjs write --run <run_id>
  ```

## Reproducibility Requirements

Every experiment MUST have:
- `random.seed(1729)` and `numpy.random.seed(1729)`
- `python_version`, `platform`, and `numpy_version` logged in `metrics.json`
- `stdout.log` and `stderr.log` captured
- At least one plot in `outputs/`

## Tool Selection Matrix

| Task | OpenClaw Tool |
|---|---|
| Check past failures | `openclaw memory search --query "..."` |
| Plan experiment | `exec: node src/openresearch.mjs approve ...` |
| Read generated code | `exec: cat runs/.../research_probe.py` |
| Run experiment | `exec: node src/openresearch.mjs run-experiment ...` |
| Read metrics | `exec: cat runs/.../metrics.json` |
| Read reviewer pass 2 | `exec: cat runs/.../reviewer_pass_2.md` |
