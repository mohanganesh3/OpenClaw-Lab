[2505.14231] UniVG-R1: Reasoning Guided Universal Visual Grounding with Reinforcement Learning

†

†  footnotetext:

* Work done during the internship at AMAP, Alibaba Group.

†  \dagger

Corresponding author

‡  \ddagger

Project lead

UniVG-R1: Reasoning Guided Universal Visual Grounding with Reinforcement Learning

Sule Bai  1,2,∗  , Mingxing Li 2 , Yong Liu 1 , Jing Tang 2 , Haoji Zhang 1 ,

Lei Sun  2

‡  \ddagger

, Xiangxiang Chu  2  , Yansong Tang  1

†  \dagger

1 Tsinghua Shenzhen International Graduate School, Tsinghua University

2 AMAP, Alibaba Group

{bsl23@mails.,tang.yansong@sz.}tsinghua.edu.cn

Abstract

Traditional visual grounding methods primarily focus on single-image scenarios with simple textual references. However, extending these methods to real-world scenarios that involve implicit and complex instructions, particularly in conjunction with multiple images, poses significant challenges, which is mainly due to the lack of advanced reasoning ability across diverse multi-modal contexts.
In this work, we aim to address the more practical universal grounding task, and propose UniVG-R1, a reasoning guided multimodal large language model (MLLM) for universal visual grounding, which enhances reasoning capabilities through reinforcement learning (RL) combined with cold-start data.
Specifically, we first construct a high-quality Chain-of-Thought (CoT) grounding dataset, annotated with detailed reasoning chains, to guide the model towards correct reasoning paths via supervised fine-tuning. Subsequently, we perform rule-based reinforcement learning to encourage the model to identify correct reasoning chains, thereby incentivizing its reasoning capabilities.
In addition, we identify a difficulty bias arising from the prevalence of easy samples as RL training progresses, and we propose a difficulty-aware weight adjustment strategy to further strengthen the performance.
Experimental results demonstrate the effectiveness of UniVG-R1, which achieves state-of-the-art performance on MIG-Bench with a 9.1% improvement over the previous method. Furthermore, our model exhibits strong generalizability, achieving an average improvement of 23.4% in zero-shot performance across four image and video reasoning grounding benchmarks. The project page can be accessed  here .

1  Introduction

Visual grounding is a significant task that aims to recognize and localize target regions in images with the guidance of instructions.
Conventional setting

yu2016refcoco  ;  mao2016refcoco

typically localizes objects based on predefined categories or explicit simple instructions (e.g., “the blue shirt”).
It struggles to perform comprehension of implicit user instructions jointly with complex visual contexts. For example, handling nuanced queries like “Which furniture in Image-2 can deal with the objects in Image-1?” (as shown in

Figure ˜ 1  ) requires advanced reasoning of user instructions across multiple images.
Therefore, we focus on achieving universal visual grounding by unlocking a broader spectrum of
challenging scenarios in this work.

Figure 1:

UniVG-R1  tackles a wide range of visual grounding tasks with complex and implicit instructions. By combining GRPO training with a cold-start initialization, it effectively reasons over instructions and visual inputs, significantly improving grounding performance. Our model achieves state-of-the-art results on MIG-Bench and exhibits superior zero-shot performance on four reasoning-guided grounding benchmarks with an average 23.4% improvement.

To effectively tackle this universal and sophisticated visual grounding task, the ability to reason complex and implicit correspondence across diverse visual contexts is crucial.
However, most previous works

yu2016refcoco  ;  mao2016refcoco  ;  xiao2024hivg  ;  deng2021transvg

have focused on localizing targets within single-image scenarios with intuitive instructions, which demonstrates a remarkable divergence from the requirements commonly observed in real-world applications.
With the development of Large Language Models (LLMs), some works

lai2024lisa  ;  wang2024llmseg  ;  chen2023shikra  ;  zhan2024griffon

propose to leverage the powerful comprehension ability of LLMs to facilitate grounding task.
Despite the great progress in understanding text instructions, these works are still limited to single image scenarios and fail to incorporate modeling of correlations across multiple images.
Recently,
Migician

li2025migician

introduces a multi-image grounding benchmark encompassing diverse grounding tasks, thereby advancing foundational initiatives to bridge this research gap.
However, Migician does not incorporate an explicit reasoning process during training, thereby falling short in terms of  advanced reasoning capabilities , particularly in handling  complex and implicit instructions across diverse images  that are essential for universal visual grounding.

Recognizing these limitations, we draw inspirations from the recent success of large reasoning models

jaech2024openaio1  ;  guo2025deepseek  ;  team2025kimi

, such as DeepSeek-R1

guo2025deepseek

, which employs rule-based reinforcement learning (RL) to significantly enhance large language model performance in solving challenging problems requiring in-depth reasoning.
To this end, we explore the potential of the RL paradigm in this work and present UniVG-R1, a powerful reasoning guided MLLM designed for universal grounding. Specifically, we initially conduct experiments using pure RL on recent advanced MLLMs (e.g., Qwen2-VL

wang2024qwen2vl

), but find that it struggles to generate correct reasoning, leading to suboptimal performance.
We ascribe this limitation to inherent constraints in the model’s intrinsic knowledge base when handling multi-image contexts, which critically hinders effective exploration of the reasoning space solely through RL.
To address this limitation, we construct a high-quality Chain-of-Thought (CoT)

wei2022chain

grounding dataset comprising 90k samples across diverse tasks, each annotated with reasoning chains and further validated by MLLMs to ensure correctness. Based on this dataset, we employ a two-stage training protocol. The first stage utilizes a cold-start supervised fine-tuning training, which directs the model towards correct reasoning pathways, then it is followed by a Group Relative Policy Optimization (GRPO) training with an IoU-based verifiable reward functions, further incentivizing the model’s reasoning capabilities.

Furthermore, we identify an inherent difficulty bias in the GRPO algorithm. Since GRPO computes the relative advantage within each group by normalization, it overlooks the varying difficulty among different samples. Consequently, easier samples receive policy gradient updates of a magnitude similar to that of more challenging, lower-performing samples. This bias diminishes the training efficiency, especially as the proportion of easy samples increases during the RL training. To address this issue, we propose a simple online difficulty-aware weight adjustment strategy that dynamically scales the gradients of samples based on their difficulty, thereby encouraging more policy gradient updates from harder samples. We experiment with multiple difficulty metrics and empirically find that all variants consistently yield additional performance improvements.

With the above designs modeling and consolidating reasoning abilities for diverse correspondence, our UniVG-R1 is capable of effectively addressing complex multimodal contexts, facilitating versatile and generalizable visual grounding applications in real-word scenarios. To demonstrate the effectiveness of our method, we conduct extensive evaluations on MIG-Bench

li2025migician

, achieving state-of-the-art results with an average improvement of more than 9% on ten tasks.
Furthermore, our model demonstrates superior generalizability, evidenced by significant zero-shot performance gains on a group of benchm