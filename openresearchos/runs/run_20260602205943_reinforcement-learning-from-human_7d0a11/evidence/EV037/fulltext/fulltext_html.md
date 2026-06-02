[2508.21016] Inference-Time Alignment Control for Diffusion Models with Reinforcement Learning Guidance

Inference-Time Alignment Control for Diffusion Models with

Reinforcement Learning Guidance

Luozhijie Jin 1  \equalcontrib ,
Zijie Qiu 1  \equalcontrib ,
Jie Liu 3 ,
Zijie Diao 1 ,
Lifeng Qiao 4 ,

Ning Ding 2  \correspondingauthor ,
Alex Lamb 2  \correspondingauthor ,
Xipeng Qiu 1  \correspondingauthor

Abstract

Denoising-based generative models, particularly diffusion and flow matching algorithms, have achieved remarkable success. However, aligning their output distributions with complex downstream objectives—such as human preferences, compositional accuracy, or data compressibility—remains challenging. While reinforcement learning (RL) fine-tuning methods, inspired by advances in RL from human feedback (RLHF) for large language models, have been adapted to these generative frameworks, current RL approaches are suboptimal for diffusion models and offer limited flexibility in controlling alignment strength after fine-tuning.
In this work, we reinterpret RL fine-tuning for diffusion models through the lens of stochastic differential equations and implicit reward conditioning.
We introduce  Reinforcement Learning Guidance  (RLG), an inference-time method that adapts Classifier-Free Guidance (CFG) by combining the outputs of the base and RL fine-tuned models via a geometric average.
Our theoretical analysis shows that RLG’s guidance scale is mathematically equivalent to adjusting the KL-regularization coefficient in standard RL objectives, enabling dynamic control over the alignment-quality trade-off without further training.
Extensive experiments demonstrate that RLG consistently improves the performance of RL fine-tuned models across various architectures, RL algorithms, and downstream tasks, including human preferences, compositional control, compressibility, and text rendering. Furthermore, RLG supports both interpolation and extrapolation, thereby offering unprecedented flexibility in controlling generative alignment. Our approach provides a practical and theoretically sound solution for enhancing and controlling diffusion model alignment at inference. The source code for RLG is publicly available at the Github:
 https://github.com/jinluo12345/Reinforcement-learning-guidance .

Figure 1:  Selected qualitative results for the human preference alignment task using SD3.5-M with GRPO and our RLG. The PickScore is displayed on each image. As the RLG scale increases, the images generally become more detailed, aesthetically pleasing, which is corroborated by the rising PickScores.

Introduction

While denoising-based generative models—primarily diffusion  (Ho, Jain, and Abbeel  2020 ; Rombach et al.  2022 )  and flow matching  (Lipman et al.  2022 ; Esser et al.  2024 )  algorithms—have gained widespread acceptance and usage, a new challenge has emerged: aligning the learned distribution with complex downstream objectives. Such alignment may involve making the generative distribution consistent with human preferences  (Kirstain et al.  2023 )  or ensuring compliance with stringent generation requirements, such as compositional correctness  (Ghosh, Hajishirzi, and Schmidt  2023 ) , fidelity in text rendering  (Liu et al.  2025b ) , or data compressibility  (Black et al.  2023 ) . Existing methods  (Liu et al.  2024 )  for aligning denoising-based generative models include reward-weighted regression  ( Peng et al.  ; Lee et al.  2023 ; Fan et al.  2025a ) , direct reward fine-tuning  (Xu et al.  2023 ; Prabhudesai et al.  2023 ; Clark et al.  2023 ) , and reinforcement learning (RL) fine-tuning.

Owing to significant advancements in Reinforcement Learning from Human Feedback (RLHF)  (Black et al.  2023 ; Lee et al.  2023 )  for Large Language Models (LLMs), an increasing number of researchers are adapting RL fine-tuning techniques from LLMs to denoising-based generative models. This adaptation is primarily achieved by interpreting the iterative denoising process as a multi-step decision-making problem, which facilitates the application of RL algorithms—such as REINFORCE  (Williams  1992 ; Mohamed et al.  2020 ; Black et al.  2023 ) , Direct Preference Optimization (DPO)  (Rafailov et al.  2023 ; Wallace et al.  2024 ) , and Group Relative Policy Optimization (GRPO)  (Shao et al.  2024 ; Liu et al.  2025b ) —to diffusion models. However, current RL methods for diffusion models still exhibit several limitations, primarily in two respects. First, the exact probability of a sampled image is intractable due to the nature of diffusion algorithms, which undermines the effectiveness of existing RL algorithms  (Black et al.  2023 ; Gong et al.  2025 ) . Second, the degree to which the base model aligns with downstream objectives remains fixed after RL fine-tuning and is sensitive to hyperparameter choices, such as the Kullback–Leibler (KL) coefficient. This inflexibility prevents users from dynamically balancing alignment and generation quality, which may be crucial in applications such as compressibility.

In this work, we draw inspiration from the stochastic differential equation (SDE) nature of denoising-based generative models  (Song et al.  2020 ) , which motivates us to interpret RL fine-tuning of diffusion models as a special case of generation conditioned on implicit rewards learned through reinforcement learning objectives  (Rafailov et al.  2024 ; Zhu, Xiao, and Honavar  2025 ; Cui et al.  2025 ) . Building upon this perspective, we introduce an inference-time enhancement technique,  Reinforcement Learning Guidance  (RLG), which adapts the established controlling approach, Classifier-Free Guidance (CFG)  (Ho and Salimans  2022 ; Zheng et al.  2023 ) , by computing a weighted sum of the outputs from the base model and the RL fine-tuned model using a geometric average. We theoretically demonstrate that this weighted averaging has the same effect as modifying the KL coefficient in RL fine-tuning, but crucially, it requires no additional training.

Empirical results on downstream tasks demonstrate that RLG enhances the performance of RL fine-tuned models across various model types (diffusion and flow matching), a range of RL methods (policy gradient, DPO, GRPO, etc.), and multiple downstream objectives (image aesthetics, compositional control, compressibility, text rendering, inpainting, and personalized generation). Furthermore, RLG supports both interpolation and extrapolation, thereby offering substantial flexibility in controlling the degree of alignment with downstream objectives.

Our contributions are summarized as follows:

•

We propose  Reinforcement Learning Guidance  (RLG), a novel, training-free approach for enhancing and controlling the alignment of diffusion models at inference time.

•

We provide a theoretical foundation for RLG, demonstrating that its guidance scale is mathematically equivalent to adjusting the KL-regularization coefficient in the underlying RL objective. This analysis formally accounts for the effectiveness of extrapolation (

w  &gt;  1

w&gt;1

).

•

We perform extensive experiments on a diverse set of alignment tasks, showing that RLG consistently enhances performance by enabling models to surpass their original fine-tuned capabilities, while also allowing for flexible trade-offs between competing objectives.

Background

Diffusion and Flow-based Generative Models

Generative modeling has advanced rapidly, particularly through diffusion  (Ho, Jain, and Abbeel  2020 ; Song, Meng, and Ermon  2020 ; Rombach et al.  2022 )  and flow-based  (Lipman et al.  2022 ; Liu, Gong, and Liu  2022 ; Esser et al.  2024 )  methods. Modern diffusion models, such as DDPM  (Ho, Jain, and Abbeel  2020 ) , DDIM  (Song, Meng, and Ermon  2020 ) , and Stable Diffusion  (Rombach et al.  2022 ) , generate samples by gradually corrupting data with noise via a forward stochastic process (SDE) and training a neural network to reverse this p