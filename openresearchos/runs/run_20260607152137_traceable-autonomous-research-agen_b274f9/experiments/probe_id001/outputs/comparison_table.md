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
| Epsilon-Greedy (eps=0.1) | 0.9196 ± 0.0239 | 0.9196 | [42, 1337, 2024] |
| **UCB (c=2.0)** | **0.9352 ± 0.0231** | **0.9352** | [42, 1337, 2024] |

**Improvement**: +0.0156 (+1.56%)  
**t-statistic**: 0.7592 | **p-value**: 0.5270 (❌ not significant)  
**Duration**: 0.0s | **Platform**: macOS-26.1-arm64-arm-64bit
