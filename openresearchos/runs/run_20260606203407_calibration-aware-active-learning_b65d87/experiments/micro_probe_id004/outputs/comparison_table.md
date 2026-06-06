# Experimental Results — calibration_aware_active_learning

## Topic
calibration-aware active learning for medical image classification

## Idea
Failure Memory for Research Agents for calibration-aware active learning

## Hypothesis
Failure Memory for Research Agents for calibration-aware active learning will improve guided idea selection utility over a random baseline under a controlled synthetic research-loop simulation.

## Results on synthetic_classification_imbalanced_calibration_AL

| Method | Score (mean ± std) | F1 (weighted) | Seeds |
|--------|-------------------|---------------|-------|
| Entropy Sampling + Sigmoid Calibration (LogReg) | 0.9381 ± 0.0050 | 0.8360 | [42, 1337] |
| **Calib-Weighted Acq + Isotonic Cal (SVC-RBF)** | **0.9491 ± 0.0024** | **0.9110** | [42, 1337] |

**Improvement**: +0.0110 (+1.10%)  
**t-statistic**: 4.3193 | **p-value**: 0.1448 (❌ not significant)  
**Duration**: 0.2s | **Platform**: macOS-26.1-arm64-arm-64bit
