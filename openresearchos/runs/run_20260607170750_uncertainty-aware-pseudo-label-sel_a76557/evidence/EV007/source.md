# EV007: SemiSeg-CAW: Semi-Supervised Segmentation of Ultrasound Images by Leveraging Class-Level Information and an Adaptive Multi-Loss Function

URL: https://www.semanticscholar.org/paper/03f54375f9d3bd9b2a3ad458e9c4af2830b94436
Year: 2025
Source: semantic_scholar
Arxiv: n/a

## Abstract

The limited availability of pixel-level annotated medical images complicates training supervised segmentation models, as these models require large datasets. To deal with this issue, SemiSeg-CAW, a semi-supervised segmentation framework that leverages class-level information and an adaptive multi-loss function, is proposed to reduce dependency on extensive annotations. The model combines segmentation and classification tasks in a multitask architecture that includes segmentation, classification, weight generation, and ClassElevateSeg modules. In this framework, the ClassElevateSeg module is initially pre-trained and then fine-tuned jointly with the main model to produce auxiliary feature maps that support the main model, while the adaptive weighting strategy computes a dynamic combination of classification and segmentation losses using trainable weights. The proposed approach enables effective use of both labeled and unlabeled images with class-level information by compensating for the shortage of pixel-level labels. Experimental evaluation on two public ultrasound datasets demonstrates that SemiSeg-CAW consistently outperforms fully supervised segmentation models when trained with equal or fewer labeled samples. The results suggest that incorporating class-level information with adaptive loss weighting provides an effective strategy for semi-supervised medical image segmentation and can improve the segmentation performance in situations with limited annotations.
