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
| Naive confidence pseudo-labeling (LogReg) | 0.5056 ± 0.0167 | 0.4915 | [42, 1337] |
| **Entropy-filtered class-balanced pseudo-labeling (RF)** | **0.6433 ± 0.0256** | **0.6156** | [42, 1337] |

**Improvement**: +0.1378 (+13.78%)  
**t-statistic**: 15.5000 | **p-value**: 0.0410 (✅ significant)  
**Duration**: 0.5s | **Platform**: macOS-26.1-arm64-arm-64bit
