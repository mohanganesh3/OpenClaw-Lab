# Experimental Results — uncertainty_aware_pseudo_label_ssl

## Topic
uncertainty-aware pseudo-label selection for semi-supervised medical image classification under label noise

## Idea
Uncertainty-calibrated pseudo-label acceptance under label noise

## Hypothesis
Entropy-filtered, class-balanced pseudo-label selection will improve held-out classification accuracy over naive confidence pseudo-labeling when the initial labeled set contains injected label noise.

## Results on sklearn_digits_medical_image_ssl_proxy_label_noise

| Method | Score (mean ± std) | F1 (weighted) | Seeds |
|--------|-------------------|---------------|-------|
| Naive confidence pseudo-labeling (LogReg) | 0.5074 ± 0.0139 | 0.4978 | [42, 1337, 2024] |
| **Entropy-filtered class-balanced pseudo-labeling (RF)** | **0.6326 ± 0.0258** | **0.6104** | [42, 1337, 2024] |

**Improvement**: +0.1252 (+12.52%)  
**t-statistic**: 9.2060 | **p-value**: 0.0116 (✅ significant)  
**Duration**: 0.8s | **Platform**: macOS-26.1-arm64-arm-64bit
