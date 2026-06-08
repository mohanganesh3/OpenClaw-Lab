# Experimental Results — uncertainty_aware_pseudo_label_ssl

## Topic
uncertainty-aware pseudo-label selection

## Idea
Ensemble Uncertainty Estimation Combining SAM and ATTA

## Hypothesis
If we combine SAM and ATTA, ECE will reduce by 25% compared to using either method alone on breast_cancer dataset.

## Results on sklearn_digits_medical_image_ssl_proxy_label_noise

| Method | Score (mean ± std) | F1 (weighted) | Seeds |
|--------|-------------------|---------------|-------|
| Naive confidence pseudo-labeling (LogReg) | 0.5004 ± 0.0423 | 0.4975 | [42, 1337, 2024, 99, 7] |
| **Entropy-filtered class-balanced pseudo-labeling (RF)** | **0.6351 ± 0.0474** | **0.6151** | [42, 1337, 2024, 99, 7] |

**Improvement**: +0.1347 (+13.47%)  
**t-statistic**: 14.1045 | **p-value**: 0.0001 (✅ significant)  
**Duration**: 1.3s | **Platform**: macOS-26.1-arm64-arm-64bit
