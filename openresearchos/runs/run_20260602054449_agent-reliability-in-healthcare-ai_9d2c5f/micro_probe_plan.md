# Micro-Probe Plan

## Candidate Experiments

### ID001: Evidence Chain Visualization for AI Agents

Level: `micro_probe`

Metric: `Percentage of agent decisions with complete evidence chain visualization`

Baseline: random idea selection under controlled synthetic utility

Status: `planned`

### ID004: Predictive Research Idea Vetting

Level: `micro_probe`

Metric: `Accuracy of predicting reviewer scores within ±10% on a test set of 100 research ideas`

Baseline: random idea selection under controlled synthetic utility

Status: `planned`

### ID009: Venue-Fit Output Validator

Level: `micro_probe`

Metric: `Percentage of outputs passing venue-fit validation without human revision`

Baseline: random idea selection under controlled synthetic utility

Status: `planned`


## Approval Rule

Before running generated code, create `approval.json` through:

```bash
node src/openresearch.mjs approve --run run_20260602054449_agent-reliability-in-healthcare-ai_9d2c5f --level micro_probe --idea <idea_id>
```
