# EV016: Uncertainty Reactivation: Dynamic Contrastive Correction for Semi-Supervised Medical Image Segmentation

URL: https://www.semanticscholar.org/paper/09bf5d1b333c4c6bf1fd85391a766f41b07362e1
Year: 2025
Source: semantic_scholar
Arxiv: n/a

## Abstract

Semi-supervised medical image segmentation has advanced significantly by utilizing pseudo-labeled annotations. However, ensuring pseudo-label accuracy remains challenging, often causing misclassification and confirmation bias. Existing methods mainly use prediction uncertainty to exclude or downweight uncertain regions, but these areas frequently coincide with diagnostically important zones, such as lesion cores or tissue boundaries. Neglecting them can thus degrade segmentation performance. To address this, we propose a Dynamic Contrastive Correction Network (DCCN) that corrects uncertain regions instead of ignoring them. DCCN aligns features from high-uncertainty areas with dynamically assigned classes via contrastive learning, reconstructing the uncertain feature space. Additionally, a Multi-layer Sampling (MLS) module leverages boundary-aware sampling to focus contrastive learning on uncertain tissue boundaries. Experiments on two public datasets show that DCCN surpasses previous SOTA methods and effectively mitigates the challenges of high-uncertainty regions.
