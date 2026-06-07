# EV015: Noise-Robust Generative Hashing for Cross-Modal Retrieval

URL: https://www.semanticscholar.org/paper/041eaab2c7fb83cc9185206bf502b751c0ad6c99
Year: 2025
Source: semantic_scholar
Arxiv: n/a

## Abstract

Deep hashing has proven remarkable effectiveness for large-scale cross-modal retrieval, yet its performance is highly vulnerable to supervisory noise, such as mismatched cross-modal correspondences and incorrect category labels. Such noise is prevalent in real-world scenarios, where correspondence mismatches and label inaccuracies often coexist, posing significant challenges for learning accurate multimodal representations. Existing methods typically address only a single type of noise in isolation and neglect the potential value of noisy data, resulting in limited performance gains. To address these challenges, we propose Noise-Robust Generative Hashing (NRGH), a unified framework designed to accommodate various forms of noise inherent in cross-modal retrieval. Specifically, NRGH introduces a hash-driven noise estimation module that computes the confidence score for each multimodal sample by combining frozen auxiliary hash functions with a Gaussian mixture model. Guided by these confidence scores, NRGH performs data correction through two stages: generative text refinement and multi-label probability calibration. The former leverages a pre-trained vision-language model to generate descriptive captions that refine noisy textual information, while the latter corrects noisy labels using confidence-aware soft labels. Furthermore, a dynamic margin contrastive loss adaptively modulates the data contribution of each sample based on its confidence, enabling sample-level adaptive learning. Extensive experiments on benchmark datasets demonstrate that NRGH significantly exceeds state-of-the-art baselines in various noisy scenarios, delivering superior robustness and accuracy. Our source codes and datasets are available at https://github.com/xiaolaohuuu/NRGH.
