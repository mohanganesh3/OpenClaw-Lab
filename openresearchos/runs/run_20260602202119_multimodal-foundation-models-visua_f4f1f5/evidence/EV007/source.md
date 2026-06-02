# EV007: Visual Knowledge-Enhanced LLaVA for Fine-Grained Multimodal Named Entity Recognition and Grounding

URL: https://www.semanticscholar.org/paper/121e8b6defee8ce60eeb145e45c4eac05270d5b8
Year: 2026
Source: semantic_scholar
Arxiv: n/a

## Abstract

The rapid growth of multimodal data has highlighted the significance of Fine-Grained Multimodal Named Entity Recognition and Grounding (FMNERG), which focuses on extracting entities and their corresponding groundings from image-text pairs. Existing approaches typically extract entities and then associate them with entity groundings using object detection methods. However, these approaches face challenges, including the use of diverse multimodal feature representations and insufficient visual knowledge, which hinder their ability to effectively link entities to images and limit overall performance. To address these limitations, we propose the visual knowledge-enhanced LLaVA (VKEL) framework, a two-stage model designed to integrate visual knowledge with multimodal learning. In the first stage, VKEL improves entity recognition by augmenting datasets with synthetic image-text pairs and optimizing alignment through lightweight fine-tuning. In the second stage, VKEL overcomes grounding limitations by incorporating consistent and accurate visual knowledge from large language models and utilizing object annotations to guide entity identification within images. This stage enhances the model’s ability to disambiguate similar entities and improve the precision of entity grounding. Extensive experiments on the FMNERG benchmark demonstrate that VKEL surpasses the SOTA by 10.24% in F1 score, with significant improvements in both fine-grained entity recognition and entity grounding performance.
