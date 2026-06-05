[2507.20592] PhaseNAS: Language-Model Driven Architecture Search with Dynamic Phase Adaptation

1

1  institutetext:

Qiyuan Lab, Beijing, China

1

1  email:  {kongfei,shanxiaohan,huyanwei,lijianmin}@qiyuanlab.com

PhaseNAS: Language-Model Driven Architecture Search with Dynamic Phase Adaptation

Fei Kong

Xiaohan Shan

Yanwei Hu

Jianmin Li

Abstract

Neural Architecture Search (NAS) is challenged by the trade-off between search space exploration and efficiency, especially for complex tasks. While recent LLM-based NAS methods have shown promise, they often suffer from static search strategies and ambiguous architecture representations.
We propose  PhaseNAS , an LLM-based NAS framework with dynamic phase transitions guided by real-time score thresholds and a structured architecture template language for consistent code generation.
On the NAS-Bench-Macro benchmark, PhaseNAS consistently discovers architectures with higher accuracy and better rank. For image classification (CIFAR-10/100), PhaseNAS reduces search time by up to 86% while maintaining or improving accuracy. In object detection, it automatically produces YOLOv8 variants with higher mAP and lower resource cost.
These results demonstrate that PhaseNAS enables efficient, adaptive, and generalizable NAS across diverse vision tasks.

keywords:  Neural Architecture Search, Large Language Models, Deep Learning, Autonomous Intelligence

1  Introduction

Neural Architecture Search (NAS) has become a foundational technique for automating neural network design in modern AI systems  [ 1 ]

[ 2 ]

[ 3 ] . Its relevance is especially prominent in autonomous and intelligent systems, where task-specific adaptation and resource efficiency are critical. However, traditional methods such as evolutionary algorithms  [ 4 ]  and reinforcement learning  [ 5 ]  face substantial computational costs, particularly in large-scale model scenarios  [ 6 ] . Gradient-based NAS approaches like DARTS  [ 7 ]  and meta-learning strategies have been proposed to improve efficiency, yet scalability and generalization remain open challenges  [ 8 ] .

Recent advances explore integrating large language models (LLMs) into NAS, enabling LLMs to reason over architecture design using natural language  [ 9 ]

[ 10 ]

[ 11 ] . While these approaches show promise, they face a fundamental resource allocation problem: different search phases require different computational capabilities, yet current methods use static LLM configurations throughout the entire search process. This leads to substantial inefficiency: using large LLMs for broad exploration wastes computational resources, while small LLMs are insufficient for fine-grained architectural refinement. Current LLM-based NAS methods suffer from several key limitations stemming from this resource mismatch:

(1)  Resource Mismatch in Search Phases : Current frameworks use fixed LLM configurations regardless of search phase requirements. Smaller models can efficiently handle broad exploration, while architectural refinement demands sophisticated reasoning capabilities that only larger models can provide.
(2)  Static Resource Allocation : Most existing methods lack adaptive mechanisms to match computational resources to search phase complexity  [ 12 ]

[ 13 ]  , leading to either over-provisioning (wasted computation) or under-provisioning (poor search quality).
(3)  Ambiguous Architecture Representation : Many previous methods employ rigid or specially-tokenized architecture descriptions, which introduce ambiguity and comprehension challenges for LLMs, leading to frequent generation failures during the search process  [ 9 ]  [ 14 ] .

These issues are particularly pronounced in complex downstream tasks such as object detection, where NAS must optimize not only accuracy but also constraints on parameters, FLOPs, and latency. Prior work like EfficientDet  [ 15 ]  and NAS-FPN  [ 16 ]  shows the promise of architecture search in detection, but they often rely on hand-crafted search spaces and fixed search strategies.

To address these challenges, we propose PhaseNAS—a dynamic, LLM-based neural architecture search framework tailored for both lightweight classification tasks and complex perception tasks common in unmanned systems, such as object detection. Using YOLOv8  [ 17 ]  as a baseline, we demonstrate how LLMs can dynamically generate and refine architectures that outperform hand-designed baselines in detection performance while remaining resource-efficient. Our method builds on recent LLM-based program synthesis efforts  [ 11 ,  18 ] , incorporating structured templates to bridge natural language prompts with executable network components.

Unlike prior works that focus mainly on image classification, our framework extends LLM-based NAS to object detection tasks, highlighting its versatility in more complex vision scenarios. Specifically,  PhaseNAS  adopts a dynamic, two-phase LLM-based NAS: an initial broad exploration phase to cover diverse architectural candidates, followed by a refinement phase that iteratively improves promising designs. Phase transitions are guided by real-time score thresholds, ensuring efficient allocation of search resources and adaptive balance between exploration and exploitation.

Our work advances neural architecture search through the following key contributions:

1.

Phase-Aware Dynamic Search : We introduce a phase-aware controller that adaptively alternates between broad exploration and focused refinement, guided by the distribution of top-performing architectures. This eliminates manual scheduling and improves convergence efficiency for both image classification and object detection tasks.

2.

Unified Architecture Template Language : We propose a structured, parameterized template system that bridges natural language prompts and executable network code. This reduces semantic ambiguity and decoding failures, enabling robust, large-scale LLM-driven architecture generation for complex models such as YOLO-based detectors.

3.

Detection-Specific Architecture Scoring : We design a novel principled scoring approach for object detection architectures without full training, leveraging a Zen-Score-inspired metric tailored for detection. This enables rapid, resource-efficient architecture search in detection scenarios, which is previously unexplored.

4.

Comprehensive Validation : Extensive experiments on CIFAR-10/100 and COCO demonstrate the effectiveness of PhaseNAS. Our method discovers YOLOv8 variants with higher mAP and lower computation, and achieves up to 97.34% accuracy on classification tasks with substantially reduced search costs.

2  Proposed Method

The fundamental insight of PhaseNAS is that neural architecture search exhibits phase-dependent computational requirements. Early exploration benefits from broad sampling across the search space, which smaller language models can efficiently handle. Later refinement requires sophisticated reasoning about architectural trade-offs and subtle optimizations, demanding the advanced capabilities of larger models.
This observation leads to our core principle:  dynamically match computational resources to search phase complexity . Rather than using a fixed LLM throughout the search, PhaseNAS adaptively transitions between appropriately-sized models based on real-time assessment of search progress.

Figure 1 :

Overview of PhaseNAS: Phase I uses small LLM for efficient exploration, Phase II uses large LLM for focused refinement. The framework adaptively transitions between phases based on search progress.

As illustrated in Figure

1  , the algorithm’s core innovation lies in its dynamic model scaling strategy, where the exploration phase utilizes a smaller, cost-efficient language model for broad and rapid architectural discovery, significantly reducing search cost. Once promising candidate architectures emerge, the refinement phase transitions to a larger, more capable language model, which further optimiz