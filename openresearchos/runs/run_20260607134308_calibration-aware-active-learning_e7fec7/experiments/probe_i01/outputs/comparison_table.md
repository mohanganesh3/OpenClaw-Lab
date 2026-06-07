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
| Entropy Sampling + Sigmoid Calibration (LogReg) | 0.9245 ± 0.0196 | 0.8473 | [42, 1337, 2024] |
| **Calib-Weighted Acq + Isotonic Cal (SVC-RBF)** | **0.9555 ± 0.0093** | **0.9100** | [42, 1337, 2024] |

**Improvement**: +0.0309 (+3.09%)  
**t-statistic**: 1.5470 | **p-value**: 0.2619 (❌ not significant)  
**Duration**: 0.5s | **Platform**: macOS-26.1-arm64-arm-64bit
