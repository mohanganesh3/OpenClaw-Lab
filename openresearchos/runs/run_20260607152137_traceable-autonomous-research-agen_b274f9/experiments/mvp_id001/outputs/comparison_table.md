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
| Epsilon-Greedy (eps=0.1) | 0.8911 ± 0.0444 | 0.8911 | [42, 1337, 2024, 99, 7] |
| **UCB (c=2.0)** | **0.9334 ± 0.0200** | **0.9334** | [42, 1337, 2024, 99, 7] |

**Improvement**: +0.0423 (+4.23%)  
**t-statistic**: 1.7233 | **p-value**: 0.1599 (❌ not significant)  
**Duration**: 0.0s | **Platform**: macOS-26.1-arm64-arm-64bit
