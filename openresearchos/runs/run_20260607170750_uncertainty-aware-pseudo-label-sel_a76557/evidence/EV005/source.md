# EV005: Addressing Data Scarcity in 3D Trauma Detection through Self-Supervised and Semi-Supervised Learning with Vertex Relative Position Encoding

URL: https://www.semanticscholar.org/paper/0189c7848a95a176005ccfecee7e2ec646ea5867
Year: 2026
Source: semantic_scholar
Arxiv: 2603.12514

## Abstract

Accurate detection and localization of traumatic injuries in abdominal CT scans remains a critical challenge in emergency radiology, primarily due to severe scarcity of annotated medical data. This paper presents a label-efficient approach combining self-supervised pre-training with semi-supervised detection for 3D medical image analysis. We employ patch-based Masked Image Modeling (MIM) to pre-train a 3D U-Net encoder on 1,206 CT volumes without annotations, learning robust anatomical representations. The pretrained encoder enables two downstream clinical tasks: 3D injury detection using VDETR with Vertex Relative Position Encoding, and multi-label injury classification. For detection, semi-supervised learning with 2,000 unlabeled volumes and consistency regularization achieves 56.57% validation mAP@0.50 and 45.30% test mAP@0.50 with only 144 labeled training samples, representing a 115% improvement over supervised-only training. For classification, expanding to 2,244 labeled samples yields 94.07% test accuracy across seven injury categories using only a frozen encoder, demonstrating immediately transferable self-supervised features. Our results validate that self-supervised pre-training combined with semi-supervised learning effectively addresses label scarcity in medical imaging, enabling robust 3D object detection with limited annotations.
