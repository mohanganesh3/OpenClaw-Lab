---
name: openresearchos-reviewer-council
description: Simulate strict OpenResearchOS peer reviewers using OpenClaw agent turns for idea review, result review, and paper readiness gates.
---

# OpenResearchOS Reviewer Council

Use this skill to attack ideas and experiment results with 5 adversarial reviewer personas. Each reviewer runs as an isolated OpenClaw agent turn for genuine separation of perspective.

## Reviewer Personas

1. **Novelty Reviewer** — Looks for identical prior work. Kills incremental tweaks.
2. **Experimental Reviewer** — Demands fair baselines, proper metrics, ablations, statistical validity.
3. **Theory/Mechanism Reviewer** — Validates that the causal mechanism is logical and falsifiable.
4. **Reproducibility Reviewer** — Verifies fixed seeds, logged outputs, clean environment.
5. **Venue Reviewer** — Assesses workshop vs conference vs arXiv tier fit.

## Pass 1: Idea Review (Before Experiments)

The `ideas` command triggers the full reviewer council automatically:
```bash
node src/openresearch.mjs ideas --run <run_id>
```

Additionally, run a prior art novelty check via OpenClaw web search:
```bash
openclaw infer web search \
  --query "<idea_title> <core_mechanism> arxiv preprint existing work" \
  --limit 8 --json
```

If web search finds a near-identical paper, mark the idea as `NOVELTY_RISK` and reduce its score.

## Pass 2: Result Review (After Experiments)

After any experiment, run the LLM result reviewer:
```bash
cat runs/<run_id>/reviewer_pass_2.md
```

The reviewer pass 2 is triggered automatically by `run-experiment`. It uses Sarvam 105b LLM to review actual experiment metrics — improvement over baseline, ablation results, and success threshold.

## Pass 3: Paper Readiness Review

```bash
node src/openresearch.mjs write --run <run_id>
```

This triggers the final readiness gate. Checks:
- Average reviewer score ≥ 6.0
- No fatal flaws
- At least one successful experiment
- ≥ 10 locked evidence sources

## Reviewer Decision Schema

Each reviewer returns JSON:
```json
{
  "reviewer": "<name>",
  "score": 7.2,
  "fatal_flaws": [],
  "fixable_flaws": ["Needs real-world dataset comparison"],
  "required_experiments": ["Run with UCI ML repo dataset"],
  "justification": "The mechanism is sound but limited to synthetic data."
}
```

## Novelty Tribunal (OpenClaw Web Search)

Before promoting any idea, run a targeted web search:
```bash
openclaw infer web search \
  --query "\"<exact mechanism name>\" research paper 2023 2024 2025" \
  --limit 10 --json

openclaw infer web search \
  --query "<idea keywords> github implementation" \
  --limit 5 --json
```

If results have direct overlap, record as competitor and adjust reviewer scores.

## Memory Integration

After review, persist the outcome:
```bash
# Automatically done by pipeline, but verify:
openclaw memory search --query "<idea_id> review" --json
```

Failed ideas appear in `~/.openclaw/workspace/memory/failed_ideas/`.

## Tool Selection Matrix

| Task | OpenClaw Tool |
|---|---|
| Run idea review | `exec: node src/openresearch.mjs ideas --run <run_id>` |
| Novelty web search | `openclaw infer web search --query "<idea> existing work"` |
| Read result review | `exec: cat runs/.../reviewer_pass_2.md` |
| Check memory for past ideas | `openclaw memory search --query "<idea_title>"` |
| Write final gate | `exec: node src/openresearch.mjs write --run <run_id>` |
