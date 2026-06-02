# Micro-Probe Plan

## Candidate Experiments

### ID004: PRISMA-Aligned Evaluation Framework for Clinical XAI

Level: `micro_probe`

Metric: `≥85% agreement among clinicians on explanation adequacy for real-time decisions`

Baseline: random idea selection under controlled synthetic utility

Status: `planned`

### ID005: Accountability-Aware Explanation Generation

Level: `micro_probe`

Metric: `Percentage reduction in ambiguous accountability attributions in simulated malpractice scenarios`

Baseline: random idea selection under controlled synthetic utility

Status: `planned`

### ID008: Stage-Specific Reliability Evaluation Framework

Level: `micro_probe`

Metric: `20% improvement in failure prediction accuracy across transformation stages compared to static evaluation methods`

Baseline: random idea selection under controlled synthetic utility

Status: `planned`


## Approval Rule

Before running generated code, create `approval.json` through:

```bash
node src/openresearch.mjs approve --run run_20260602072946_agent-reliability-in-healthcare-ai_341699 --level micro_probe --idea <idea_id>
```
