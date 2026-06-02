[2507.04453] ESSA: Evolutionary Strategies for Scalable Alignment

\newunicodechar

♢ \textdiamondsuit

ESSA: Evolutionary Strategies for Scalable Alignment

Daria Korotyshova

&amp;Boris Shaposhnikov

Alexey Malakhov

&amp;Nikita Surnachev

&amp;George Bredis

Alexey Gorbatovski

&amp;Viacheslav Sinii

T-Tech
&amp;Daniil Gavrilov

Corresponding author: b.shaposhnikov@tbank.ru

Abstract

Large Language Models (LLMs) are increasingly relying on alignment techniques to ensure that their outputs match human preferences. Although reinforcement learning from human feedback (RLHF) is the dominant approach, it has high computational costs, memory requirements, and training instability, particularly when scaling to larger models.
This paper introduces ESSA (Evolutionary Strategies for Scalable Alignment), a new framework that uses Evolutionary Strategies (ES) to efficiently align LLMs without the need for gradient computation. ES is well-suited for LLM alignment due to its favorable properties, such as high parallelizability, memory efficiency, robustness to sparse rewards, and fewer data samples required for convergence, especially when starting from a strong pre-trained policy. Moreover, ES eliminates the need for extensive hyperparameter tuning, making the alignment process simpler and more stable.
Although ES excels in low-dimensional optimization, it poses a challenge when applied to high-dimensional LLMs. To address this challenge, we propose a parameter-efficient architectural modification that reduces the dimensionality of optimization through low-rank adaptation. We evaluated our approach on mathematical reasoning tasks with verifiable accuracy-based metrics, demonstrating that ESSA converges faster and is more data efficient than gradient-based methods like Group Relative Policy Optimization (GRPO). Our findings establish ES as a promising and scalable alternative to gradient-based alignment, paving the way for efficient post-training of large language models.

1  Introduction

Large Language Models (LLMs) have achieved impressive capabilities across a range of tasks, from conversation and summarization to mathematical reasoning and programming. However, these capabilities hinge not just on scale but on alignment: ensuring that models produce outputs that are aligned with user expectations. Reinforcement learning from human feedback has become the de facto standard for this alignment process, allowing models to learn human-favored behaviors  (Ouyang et al.,,  2022 ) .

However, RLHF and its variants, for example those built on Proximal Policy Optimization (PPO)  (Schulman et al.,,  2017 )  and REINFORCE  (Sutton et al.,,  1999 ) , are fundamentally limited. In the online setting, these methods require large amounts of computational resources, suffer from training instability, exhibit high variance, and struggle with sparse rewards, making them difficult to scale to models exceeding tens of billions of parameters  (Zheng et al.,,  2023 ; Sheng et al.,,  2024 ) . Offline methods  (Rafailov et al.,,  2024 ; Hong et al.,,  2024 )  simplify the process but introduce their own challenges: they are fundamentally constrained by the quality and coverage of fixed datasets, which limits generalization and alignment quality  (Tang et al.,,  2024 ; Xu et al.,,  2024 ; Chu et al.,,  2025 ) .

In response to these challenges, this paper explores Evolutionary Strategies as a novel and scalable solution for LLM alignment, an approach previously underexplored in this context due to perceived limitations with high-dimensional optimization. ES offers several advantages directly suited to modern alignment challenges: it is highly parallelizable, scales efficiently across many computing nodes, remains robust even with sparse rewards, and requires much less memory by operating in inference-only mode without gradient computations  (Salimans et al.,,  2017 ) . In particular, ES methods are most effective when initialized with a strong policy  (Chrabaszcz et al.,,  2018 ) . This property is particularly relevant for LLM alignment, where training typically starts from a well-performing supervised fine-tuned (SFT) model. Given these promising features, we propose  ESSA (Evolutionary Strategies for Scalable Alignment) , a framework for aligning LLMs using ES, and present the first comprehensive study of ES methods for aligning large-scale language models.

To demonstrate ESSA’s effectiveness and circumvent complexities related to learned reward proxies, we evaluate our approach on mathematical reasoning benchmarks where correctness (accuracy) serves as a clear, unambiguous reward signal. For comparison, we use the GRPO method  (Shao et al.,,  2024 ) , which has become a widely adopted baseline due to its strong performance in mathematical and reasoning tasks. Building on this foundation, our contributions are as follows:

•

We adapt the low rank adaptation (LoRA) technique to the ES paradigm, reducing the effective dimensionality while retaining the expressivity of the full model.

•

We show that ESSA achieves faster convergence and better data efficiency compared to the GRPO method.

•

We demonstrate that ESSA scales efficiently across distributed compute environments and avoids the memory bottlenecks associated with backpropagation-based alignment.

2  Related Works

Online Alignment

RLHF, typically instantiated with PPO or other policy gradient methods  (Ouyang et al.,,  2022 ; Schulman et al.,,  2017 ) , remains the standard for aligning LLMs with human preferences. Although effective in conversational tasks, RLHF suffers from high variance, sparse rewards, and significant memory overhead for large models  (Zheng et al.,,  2023 ; Sheng et al.,,  2024 ) . Stability-focused variants like RLOO and GRPO still require extensive hyperparameter tuning and gradient estimation  (Shao et al.,,  2024 ) . Actor–critic or value-free approaches (e.g., RLOO) struggle with scalability and sparse rewards in complex reasoning tasks  (Ahmadian et al.,,  2024 ) .

Parameter-efficient training of LLMs

To reduce the cost of fine-tuning large models, parameter-efficient methods have been introduced such as adapters  (Houlsby et al.,,  2019 ) , prefix-tuning  (Li and Liang,,  2021 ) , and LoRA  (Hu et al.,,  2021 ) . These approaches freeze most weights and update only a small subset, with LoRA reducing parameter count via low-rank adaptation. Transformer 2

(Sun et al.,,  2025 )  further reduces the LoRA parameter space using single value fine-tuning, similar to our method.

Evolution strategies

Evolutionary Strategies, such as Covariance Matrix Adaptation (CMA-ES)  (Hansen and Ostermeier,,  2001 )  and NES  (Wierstra et al.,,  2014 ) , have shown promise for gradient-free optimization, robustness to sparse rewards, and strong parallelism. NSGA-III  (Liu et al.,,  2025 )  prunes model weights based on importance, while CMA-ES has been used for black-box prompt optimization in LMaaS scenarios  ( Sun et al., 2022b,  ;  Sun et al., 2022a,  ) , though still relying on gradient-based pretraining. Zero-order optimizers  (Zhang et al.,,  2024 )  reduce memory by approximating gradients through loss differences, matching gradients on simple tasks but trailing in complex reasoning. Applications of ES to LLM alignment are rare due to high-dimensionality concerns. GENOME/GENOME+  (Zhang et al.,,  2025 )  evolve populations by combining weights, and LoRAHub  (Huang et al.,,  2024 )  combines LoRA adapters to reduce parameter space, yet both depend on pre-trained weights. DFO  (Jin et al.,,  2024 )  exclusively uses ES to optimize LoRA adapters, reducing dimensionality but with interpretability challenges. Our work advances this line by applying ES to singular LoRA matrices, making black-box optimization interpretable and feasible even at large scale.

3  ESSA

3.1  Motivation

With today’s tools, RL training involves three stages: sampling sequential trajectories (often accelerated using specializ