[2509.16127] BaseReward: A Strong Baseline for Multimodal Reward Model

BaseReward: A Strong Baseline for Multimodal Reward Model

Yi-Fan Zhang  ∗,2  , Haihua Yang  ♠,∗,1  , Huanyu Zhang  2  , Yang Shi  4

Zezhou Chen  2  ,
Haochen Tian  2  , Chaoyou Fu  3,†  , Kai Wu  1  , Bo Cui  1

Xu Wang  1  , Jianfei Pan  1  , Haotian Wang  5  , Zhang Zhang  2,†  , Liang Wang  2

1

ByteDance

2

CASIA

3

NJU

4

PKU

5

THU

♠

Project Leader

∗

Equal Contribution

†

Corresponding Author

Abstract

The rapid advancement of Multimodal Large Language Models (MLLMs) has made aligning them with human preferences a critical challenge. Reward Models (RMs) are a core technology for achieving this goal, but a systematic guide for building state-of-the-art Multimodal Reward Models (MRMs) is currently lacking in both academia and industry. Through exhaustive experimental analysis, this paper aims to provide a clear “recipe” for constructing high-performance MRMs. We systematically investigate every crucial component in the MRM development pipeline, including  reward modeling paradigms  (e.g., Naive-RM, Critic-based RM, and Generative RM),  reward head architecture ,  training strategies ,  data curation  (covering over ten multimodal and text-only preference datasets),  backbone model  and  model scale , and  ensemble methods .

Based on these experimental insights, we introduce  BaseReward , a powerful and efficient baseline for multimodal reward modeling. BaseReward adopts a simple yet effective architecture, built upon a Qwen2.5-VL backbone, featuring an optimized two-layer reward head, and is trained on a carefully curated mixture of high-quality multimodal and text-only preference data. Our results show that BaseReward establishes a new  state-of-the-art (SOTA)  on major benchmarks such as MM-RLHF-Reward Bench, VL-Reward Bench, and Multimodal Reward Bench, outperforming previous open-source and proprietary models. Furthermore, to validate its practical utility beyond static benchmarks, we integrate BaseReward into a real-world reinforcement learning pipeline, successfully enhancing an MLLM’s performance across various perception, reasoning, and conversational tasks. This work not only delivers a top-tier MRM but, more importantly, provides the community with a clear, empirically-backed guide for developing robust reward models for the next generation of MLLMs.

Contents

1  Introduction

2  Related Work

3  Recipe for Building MRM

3.1  Preliminary

3.2  Evaluation Benchmarks and Metrics

4  Experimental Analysis

4.1  Reward Modeling Approaches

4.2  Reward Model Design

4.3  Training Regularization Strategies

4.4  Common Training Datasets

4.5  Optimizing Multimodal RMs for Pure-Text Tasks

4.6  Impact of Base Model Selection and Scale

4.7  Ensemble Strategies for Reward Models

5  BaseReward

5.1  Structure and Training Strategy

5.2  Baseline Algorithms

5.3  Evaluation Results on MRM Benchmark

5.4  Reinforcement Learning with BaseReward

5.4.1  Experimental Setup

5.4.2  Results and Analysis

6  Conclusion and Limitation

Introduction

The rapid advancement of Multimodal Large Language Models (MLLMs)  (Yang et al.,  2024 ; Team et al.,  2025a ; Zhang et al.,  2024a ; Xiaomi,  2025 ; Chen et al.,  2023a ; Fu et al.,  2025 )  has ushered in a new era of AI capabilities, enabling sophisticated understanding and generation across diverse data modalities, including text, images, video, and audio. Despite these impressive achievements, a central challenge remains: ensuring that these powerful models consistently produce outputs that are helpful, harmless, and aligned with human values and preferences. A pivotal technology to address this challenge is the reward model (RM), which is trained to evaluate and score model outputs based on human feedback. These reward models serve as crucial learning signals for fine-tuning MLLMs via methods such as Reinforcement Learning from Human Feedback (RLHF)  (Sun et al.,  2023 ; Ouyang et al.,  2022 ; Zhang et al.,  2025a ) , effectively steering the models toward safer, more reliable, and user-aligned behaviors.

While the concept of reward modeling is well-established for text-only Large Language Models (LLMs), the blueprint for constructing state-of-the-art Multimodal Reward Models (MRMs)  (Pu et al.,  2025 ; Chen et al.,  2024a ; Xiong et al.,  2024 ; Wang et al.,  2025a ; Zang et al.,  2025 ; Zhang et al.,  2025b )  remains less clear. Currently, state-of-the-art MLLMs, each employ distinct reward modeling strategies, incorporating various domain-specific techniques. For instance, Seed 1.5 VL  (Team,  2025 )  and Keye-VL  (Team et al.,  2025a )  utilize generative reward models, with the former enhancing reliability by comparing rollout content against golden references. Mimo-VL  (Xiaomi,  2025 )  employs dual reward models—one specialized for text-only questions and another for multimodal tasks. GLM 4.1 V Thinking  (Team et al.,  2025b )  adopts domain-specific reward strategies tailored to different data categories. Despite this diversity in approaches, the research landscape lacks a systematic, comprehensive study to guide researchers effectively. Critical questions remain unanswered: Which reward model architecture delivers optimal performance? What constitutes the most effective architectural design for reward models? How do different data sources—including text-only preference data—influence multimodal performance? What roles do the MLLM backbone architecture and model scale play in determining effectiveness?

This paper provides a “recipe” for building a high-performance MRM by conducting an exhaustive experimental analysis to answer these fundamental questions. We systematically investigate every crucial component of the MRM development pipeline:

•

Reward Modeling Paradigms:  We compare the performance of Naive, Critic-based, and Generative reward models to identify the most efficient and effective approach.

•

Architectural Design:  We perform detailed ablations on the reward head’s structure, including the number of layers and the choice of activation functions.

•

Training Strategies:  We analyze the impact of common regularization techniques, such as zero-coefficient regularization and length normalization, on model performance.

•

Data Curation:  We evaluate the influence of over ten different multimodal and text-only preference datasets, revealing the surprising efficacy of text data in enhancing multimodal judgment and the necessity of careful data selection.

•

Backbone and Scale:  We assess how the choice of the underlying MLLM backbone and its parameter scale affect final reward modeling capabilities.

•

Ensemble Methods:  We explore various ensemble strategies to combine the strengths of diverse models, pushing performance beyond what any single model can achieve.

Based on insights gained from our extensive experiments, we present  BaseReward , a powerful and efficient baseline for multimodal reward modeling. BaseReward leverages a simple yet effective architecture built upon the Qwen2.5-VL  (Bai et al.,  2025 )  backbone, enhanced with an optimized two-layer reward head, and trained on a carefully curated mixture of high-quality multimodal and text-only preference data. Our model sets a new state-of-the-art (SOTA), surpassing previous open-source and proprietary systems, including Claude 3.7 Sonnet and R1-Reward  (Zhang et al.,  2025b ) , across major benchmarks such as MM-RLHF-Reward Bench  (Zhang et al.,  2025a )  (improving by approximately 11%), VL-Reward Bench  (Li et al.,  2024a )  (improving by approximately 18%), and Multimodal Reward Bench  (Yasunaga et al.,  2025 ) . Additionally, to demonstrate its practical utility beyond static benchmarks, we integrate BaseReward into a real-world reinforcement learning process. As detailed in Section

5.4  , using BaseReward to provide the reward signal leads to consistent performance gains when fine-tuning an MLLM across a diverse range of