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
| Entropy Sampling + Sigmoid Calibration (LogReg) | 0.9189 ± 0.0175 | 0.8436 | [42, 1337, 2024, 99, 7] |
| **Calib-Weighted Acq + Isotonic Cal (SVC-RBF)** | **0.9591 ± 0.0086** | **0.9164** | [42, 1337, 2024, 99, 7] |

**Improvement**: +0.0402 (+4.02%)  
**t-statistic**: 3.1337 | **p-value**: 0.0351 (✅ significant)  
**Duration**: 0.4s | **Platform**: macOS-26.1-arm64-arm-64bit
