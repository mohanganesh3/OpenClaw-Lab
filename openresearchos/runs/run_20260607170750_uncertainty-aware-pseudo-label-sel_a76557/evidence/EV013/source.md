# EV013: Exploring Feature Representation Learning for Semi-Supervised Medical Image Segmentation

URL: https://www.semanticscholar.org/paper/0fd5b1c880d2dd51408d3c3b32d554c26c1549a0
Year: 2021
Source: semantic_scholar
Arxiv: 2111.10989

## Abstract

This article presents a simple yet effective two-stage framework for semi-supervised medical image segmentation. Unlike prior state-of-the-art semi-supervised segmentation methods that predominantly rely on pseudo supervision directly on predictions, such as consistency regularization and pseudo labeling, our key insight is to explore the feature representation learning with labeled and unlabeled (i.e., pseudo labeled) images to regularize a more compact and better-separated feature space, which paves the way for low-density decision boundary learning and therefore enhances the segmentation performance. A stage-adaptive contrastive learning method is proposed, containing a boundary-aware contrastive loss that takes advantage of the labeled images in the first stage, as well as a prototype-aware contrastive loss to optimize both labeled and pseudo labeled images in the second stage. To obtain more accurate prototype estimation, which plays a critical role in prototype-aware contrastive learning, we present an aleatoric uncertainty-aware method to generate higher quality pseudo labels. Aleatoric-uncertainty adaptive (AUA) adaptively regularizes prediction consistency by taking advantage of image ambiguity, which, given its significance, is underexplored by existing works. Our method achieves the best results on three public medical image segmentation benchmarks.
