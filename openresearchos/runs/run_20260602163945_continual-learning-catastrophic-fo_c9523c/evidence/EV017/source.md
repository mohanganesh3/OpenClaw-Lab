# EV017: Continual Unsupervised Domain Adaptation for Semantic Segmentation

URL: https://www.semanticscholar.org/paper/0095bdc4f2e1250368d4a40e894f4b81f9b8134b
Year: 2020
Source: semantic_scholar
Arxiv: 2010.09236

## Abstract

Unsupervised Domain Adaptation (UDA) for semantic segmentation has been favorably applied to real-world scenarios in which pixel-level labels are hard to be obtained. In most of the existing UDA methods, all target data are assumed to be introduced simultaneously. Yet, the data are usually presented sequentially in the real world. Moreover, Continual UDA, which deals with more practical scenarios with multiple target domains in the continual learning setting, has not been actively explored. In this light, we propose Continual UDA for semantic segmentation based on a newly designed Expanding Target-specific Memory (ETM) framework. Our novel ETM framework contains Target-specific Memory (TM) for each target domain to alleviate catastrophic forgetting. Furthermore, a proposed Double Hinge Adversarial (DHA) loss leads the network to produce better UDA performance overall. Our design of the TM and training objectives let the semantic segmentation network adapt to the current target domain while preserving the knowledge learned on previous target domains. The model with the proposed framework outperforms other state-of-the-art models in continual learning settings on standard benchmarks such as GTA5, SYNTHIA, CityScapes, IDD, and Cross-City datasets. The source code is available at https://github.com/joonh-kim/ETM.
