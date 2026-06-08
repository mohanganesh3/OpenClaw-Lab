# EV015: Uncertainty-Aware Extreme Point Tracing for Weakly Supervised Ultrasound Image Segmentation

URL: https://www.semanticscholar.org/paper/0656a99bb7fcbeb179ee3729b8f1c638bafca8ed
Year: 2025
Source: semantic_scholar
Arxiv: 2510.15666

## Abstract

Automatic medical image segmentation is a fundamental step in computer-aided diagnosis, yet fully supervised approaches demand extensive pixel-level annotations that are costly and time-consuming. To alleviate this burden, we propose a weakly supervised segmentation framework that leverages only four extreme points as annotation. Specifically, bounding boxes derived from the extreme points are used as prompts for the Segment Anything Model 2 (SAM2) to generate reliable initial pseudo labels. These pseudo labels are progressively refined by an enhanced Feature-Guided Extreme Point Masking (FGEPM) algorithm, which incorporates Monte Carlo dropout-based uncertainty estimation to construct a unified gradient uncertainty cost map for boundary tracing. Furthermore, a dual-branch Uncertainty-aware Scale Consistency (USC) loss and a box alignment loss are introduced to ensure spatial consistency and precise boundary alignment during training. Extensive experiments on two public ultrasound datasets, BUSI and UNS, demonstrate that our method achieves performance comparable to, and even surpassing fully supervised counterparts while significantly reducing annotation cost. These results validate the effectiveness and practicality of the proposed weakly supervised framework for ultrasound image segmentation.
