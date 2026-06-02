# EV011: ExGra-Med: Extended Context Graph Alignment for Medical Vision-Language Models

URL: https://www.semanticscholar.org/paper/20085d97121b8b958d42030b1bdb9da143d42f55
Year: 2024
Source: semantic_scholar
Arxiv: 2410.02615

## Abstract

State-of-the-art medical multi-modal LLMs (med-MLLMs), such as LLaVA-Med and BioMedGPT, primarily depend on scaling model size and data volume, with training driven largely by autoregressive objectives. However, we reveal that this approach can lead to weak vision-language alignment, making these models overly dependent on costly instruction-following data. To address this, we introduce ExGra-Med, a novel multi-graph alignment framework that jointly aligns images, instruction responses, and extended captions in the latent space, advancing semantic grounding and cross-modal coherence. To scale to large LLMs (e.g., LLaMA-7B), we develop an efficient end-to-end training scheme using black-box gradient estimation, enabling fast and scalable optimization. Empirically, ExGra-Med matches LLaVA-Med's performance using just 10% of the pre-training data, achieving a 20.13% gain on VQA-RAD and approaching full-data performance. It also outperforms strong baselines like BioMedGPT and RadFM on visual chatbot and zero-shot classification tasks, demonstrating its promise for efficient, high-quality vision-language integration in medical AI.
