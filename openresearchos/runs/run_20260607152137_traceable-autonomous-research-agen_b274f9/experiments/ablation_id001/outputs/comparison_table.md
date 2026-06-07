# Experimental Results — rl_simulation

## Topic
traceable autonomous research agents smoke 95e44e

## Idea
Traceability gap: agents may produce ideas without an auditable chain from sourc

## Hypothesis
Traceability gap: agents may produce ideas without an auditable chain from source evidence to experiment decision.

## Results on multi_armed_bandit_k10

| Method | Score (mean ± std) | F1 (weighted) | Seeds |
|--------|-------------------|---------------|-------|
| Epsilon-Greedy (eps=0.1) | 0.9350 ± 0.0118 | 0.9350 | [42, 1337] |
| **UCB (c=2.0)** | **0.9451 ± 0.0226** | **0.9451** | [42, 1337] |

**Improvement**: +0.0101 (+1.01%)  
**t-statistic**: 0.2932 | **p-value**: 0.8184 (❌ not significant)  
**Duration**: 0.0s | **Platform**: macOS-26.1-arm64-arm-64bit
