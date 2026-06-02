# EV016: Contrastive Vision-Language Alignment Makes Efficient Instruction Learner

URL: https://www.semanticscholar.org/paper/2b41b3e23d6b8b84445abb77870aa3b9e71c75d8
Year: 2023
Source: semantic_scholar
Arxiv: 2311.17945

## Abstract

We study the task of extending the large language model (LLM) into a vision-language instruction-following model. This task is crucial but challenging since the LLM is trained on text modality only, making it hard to effectively digest the visual modality. To address this, existing methods typically train a visual adapter to align the representation between a pre-trained vision transformer (ViT) and the LLM by a generative image captioning loss. However, we find that the generative objective can only produce weak alignment for vision and language, making the aligned vision-language model very hungry for the instruction fine-tuning data. In this paper, we propose CG-VLM that applies both Contrastive and Generative alignment objectives to effectively align the representation of ViT and LLM. Different from image level and sentence level alignment in common contrastive learning settings, CG-VLM aligns the image-patch level features and text-token level embeddings, which, however, is very hard to achieve as no explicit grounding patch-token relation provided in standard image captioning datasets. To address this issue, we propose to maximize the averaged similarity between pooled image-patch features and text-token embeddings. Extensive experiments demonstrate that the proposed CG-VLM produces strong vision-language alignment and is an efficient instruction learner. For example, using only 10% instruction tuning data, we reach 95% performance of state-of-the-art method LLaVA [29] on the zero-shot ScienceQA-Image benchmark.
