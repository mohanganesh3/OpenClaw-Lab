# EV009: Rewarding Structural Conformance of Reasoning using Process Mining

URL: https://www.semanticscholar.org/paper/019e5c85556553d3679b815db12721361f805f7a
Year: 2025
Source: semantic_scholar
Arxiv: 2510.25065

## Abstract

Recent advances in sparse reward policy gradient methods have enabled effective reinforcement learning (RL)-based language model post-training. However, for reasoning tasks such as mathematical problem solving, binarized outcome rewards provide limited feedback on intermediate reasoning steps. While some studies have attempted to address this issue by estimating overall reasoning quality, it remains unclear whether these rewards are reliable proxies for the quality of stepwise reasoning. In this study, we consider reasoning as a structured process and propose TACReward, the reward model that can be seamlessly integrated into sparse reward policy gradient methods without additional human annotation costs or architectural modifications. TACReward aggregates stepwise structural deviations between teacher and policy reasoning using process mining techniques, producing a scalar output reward range of [0, 1] to indicate reasoning quality. Experiments on multiple mathematical reasoning benchmarks demonstrate that integrating the TACReward into sparse reward frameworks encourages the policy model to improve the structural quality of reasoning. Consequently, this leads to consistent performance improvements over existing sparse reward frameworks. Our code and checkpoints are publicly available at https://github.com/Thrillcrazyer/TACReward and https://huggingface.co/Thrillcrazyer/TACReward7B.
