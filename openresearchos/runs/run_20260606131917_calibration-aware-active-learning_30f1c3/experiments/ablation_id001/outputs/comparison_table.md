# Experimental Results — classification

## Topic


## Idea
Unknown idea

## Hypothesis
Evidence-to-Experiment Trace Ledger for calibration-aware active learning will improve guided idea selection utility over a random baseline under a controlled synthetic research-loop simulation.

## Results on sklearn_digits_gradient_boosting

| Method | Accuracy (mean ± std) | F1 (weighted) | Seeds |
|--------|----------------------|---------------|-------|
| Baseline | 0.9694 ± 0.0028 | 0.9695 | [42, 1337] |
| **Proposed** | **0.9319 ± 0.0014** | **0.9319** | [42, 1337] |

**Improvement**: -0.0375 (-3.75%)  
**t-statistic**: -9.0000 | **p-value**: 0.0704 (❌ not significant)  
**Duration**: 10.9s | **Platform**: macOS-26.1-arm64-arm-64bit
