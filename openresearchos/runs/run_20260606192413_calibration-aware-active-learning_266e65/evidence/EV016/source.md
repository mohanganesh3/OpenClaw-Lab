# EV016: Uncertainty-Guided Dual-Domain Learning for Reliable Skin Lesion Segmentation

URL: https://www.semanticscholar.org/paper/5ff94fb99b4d098ceeb795e1d99ed0473b3215ef
Year: 2026
Source: semantic_scholar
Arxiv: 2605.09600

## Abstract

Accurate skin lesion segmentation is vital for dermoscopic Computer-Aided Diagnosis. However, visual ambiguity and morphological irregularity often defeat spatial modeling, necessitating multi-domain architectures. Existing paradigms frequently overlook the active use of prediction uncertainty, leading to deterministic frameworks that suffer from blind cross-domain fusion and overfit to label noise. To address these issues, we propose the Uncertainty-Guided Dual-Domain Network (UGDD-Net). UGDD-Net introduces a novel"Glance-and-Gaze"mechanism to transform uncertainty into an active guiding signal. Specifically, the Uncertainty-Guided Bi-directional Feature Fusion (UGBFF) module uses pixel-level uncertainty to modulate spatial-spectral interactions. The Uncertainty-Guided Graph Refinement (UGGR) module constructs a topology-aware graph to propagate reliable semantic consensus and refine uncertain nodes. Finally, the Uncertainty-Guided Margin-Adaptive Loss (UGML) enforces strict constraints on confident pixels while relaxing penalties on uncertain ones to improve statistical calibration. Extensive experiments on ISIC2017, ISIC2018, PH2, and HAM10000 datasets demonstrate that UGDD-Net achieves state-of-the-art performance, especially on"Hard Samples". Our uncertainty maps align with expert inter-observer variability, providing robust interpretability for human-machine collaborative diagnosis.
