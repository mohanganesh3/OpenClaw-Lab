# Experimental Results — calibration_aware_active_learning

## Topic
calibration-aware active learning under noisy labels for small medical tabular datasets

## Idea
Reflexion-Enhanced Calibration for Medical Tabular Active Learning

## Hypothesis
If we apply Reflexion-style self-reflection to calibration-aware active learning, ECE will decrease by 15% compared to standard entropy sampling on breast_cancer dataset.

## Results on synthetic_classification_imbalanced_calibration_AL

| Method | Score (mean ± std) | F1 (weighted) | Seeds |
|--------|-------------------|---------------|-------|
| Entropy Sampling + Sigmoid Calibration (LogReg) | 0.9381 ± 0.0050 | 0.8360 | [42, 1337] |
| **Calib-Weighted Acq + Isotonic Cal (SVC-RBF)** | **0.9491 ± 0.0024** | **0.9110** | [42, 1337] |

**Improvement**: +0.0110 (+1.10%)  
**t-statistic**: 4.3193 | **p-value**: 0.1448 (❌ not significant)  
**Duration**: 0.3s | **Platform**: macOS-26.1-arm64-arm-64bit
