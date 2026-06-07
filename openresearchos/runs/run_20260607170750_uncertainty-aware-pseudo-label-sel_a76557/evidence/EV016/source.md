# EV016: Learning Disentangled Stain and Structural Representations for Semi-Supervised Histopathology Segmentation

URL: https://www.semanticscholar.org/paper/15abe98f988b96c980b47d07c6f8436c8f60a04a
Year: 2025
Source: semantic_scholar
Arxiv: 2507.03923

## Abstract

Accurate gland segmentation in histopathology images is essential for cancer diagnosis and prognosis. However, significant variability in Hematoxylin and Eosin (H&E) staining and tissue morphology, combined with limited annotated data, poses major challenges for automated segmentation. To address this, we propose Color-Structure Dual-Student (CSDS), a novel semi-supervised segmentation framework designed to learn disentangled representations of stain appearance and tissue structure. CSDS comprises two specialized student networks: one trained on stain-augmented inputs to model chromatic variation, and the other on structure-augmented inputs to capture morphological cues. A shared teacher network, updated via Exponential Moving Average (EMA), supervises both students through pseudo-labels. To further improve label reliability, we introduce stain-aware and structure-aware uncertainty estimation modules that adaptively modulate the contribution of each student during training. Experiments on the GlaS and CRAG datasets show that CSDS achieves state-of-the-art performance in low-label settings, with Dice score improvements of up to 1.2% on GlaS and 0.7% on CRAG at 5% labeled data, and 0.7% and 1.4% at 10%. Our code and pre-trained models are available at https://github.com/hieuphamha19/CSDS.
