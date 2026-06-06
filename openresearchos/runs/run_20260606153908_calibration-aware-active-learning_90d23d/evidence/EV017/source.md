# EV017: Active Learning for Pneumonia Detection with Vision Transformers and Bayesian Uncertainty Estimation

URL: https://www.semanticscholar.org/paper/6b1e82477914e04aaf6699127e3a4d5878805d4d
Year: 2025
Source: semantic_scholar
Arxiv: n/a

## Abstract

Pneumonia remains a major global health challenge, and timely diagnosis via chest X-ray interpretation is critical for effective clinical intervention. While deep learning methods have shown promising performance, their reliance on large annotated datasets and inability to quantify predictive uncertainty limit their reliability in real-world deployment. This study proposes a novel framework that integrates Vision Transformers (ViT) with Monte Carlo (MC) Dropout for uncertainty-aware pneumonia detection. By retaining dropout layers during inference, the model estimates predictive confidence across multiple stochastic forward passes, enabling it to identify ambiguous cases for expert review. To further enhance data efficiency, the system incorporates an active learning strategy that prioritizes uncertain samples for annotation. Experiments on a publicly available chest X-ray dataset demonstrate that the proposed ViT + MC Dropout model achieves a classification accuracy of 94.1%, sensitivity 96.0% and achieves a higher AUC of 0.98. Moreover, its uncertainty calibration is significantly better, with 85% of predictions above 90% confidence score. The active learning loop further enhances model efficiency, improving ViT accuracy of 89.26% with fewer labeled samples.
