# EV001: VGRSS: Datasets and Models for Visual Grounding in Remote Sensing Ship Images

URL: https://www.semanticscholar.org/paper/0061c4d950647903ae232285132aa385619698fa
Year: 2025
Source: semantic_scholar
Arxiv: n/a

## Abstract

This article introduces a task named visual grounding of remote sensing ship (VGRSS) images. The goal of VGRSS is to locate ship objects in remote sensing images guided by natural language. Extensive research has been conducted on multimodal processing of remote sensing images and text to retrieve rich information from remote sensing images using natural language. However, due to the unique characteristics of remote sensing ship images, ship localization using natural language remains a challenge. Therefore, in this work, we construct datasets for the VGRSS task and explore deep learning models. Specifically, our contributions can be summarized as follows: first, we construct two remote sensing ship datasets for visual grounding. One is based on the optical remote sensing ship target detection benchmark dataset, named RSSVG, while the other is based on the synthetic aperture radar (SAR) dataset, named SARVG. Second, we propose a language-guided visual feature enhancement (LVFE) module. This module enhances visual features through language guidance before visual-linguistic fusion (VLF). Third, we propose a VLF module based on multimodal feature stacking. This module inputs the stacked language and visual features, and then performs feature fusion using a Transformer, enabling effective cross-modal interaction and integration. Fourth, we introduce a novel loss calculation method by incorporating enhanced intersection over union (EIoU) into the loss function. Finally, we benchmark extensive state-of-the-art (SOTA) natural image visual grounding (VG) methods on the constructed RSSVG and SARVG datasets, then provide insightful analysis based on the results. This work offers valuable insights for developing better VGRSS models.
