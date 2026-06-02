[2405.17956] Hybrid Preference Optimization: Augmenting Direct Preference Optimization with Auxiliary Objectives

Hybrid Preference Optimization: Augmenting Direct Preference Optimization with Auxiliary Objectives

Anirudhan Badrinath

Prabhat Agarwal

Jiajing Xu

Pinterest Labs

{abadrinath, pagarwal, jiajing}@pinterest.com

Abstract

For aligning large language models (LLMs), prior work has leveraged reinforcement learning via human feedback (RLHF) or variations of direct preference optimization (DPO). While DPO offers a simpler framework based on maximum likelihood estimation, it compromises on the ability to tune language models to easily maximize non-differentiable and non-binary objectives according to the LLM designer’s preferences (e.g., using simpler language or minimizing specific kinds of harmful content). These may neither align with user preferences nor even be able to be captured tractably by binary preference data. To leverage the simplicity and performance of DPO with the generalizability of RL, we propose a hybrid approach between DPO and RLHF. With a simple augmentation to the implicit reward decomposition of DPO, we allow for tuning LLMs to maximize a set of arbitrary auxiliary rewards using offline RL. The proposed method, Hybrid Preference Optimization (HPO), shows the ability to effectively generalize to both user preferences and auxiliary designer objectives, while preserving alignment performance across a range of challenging benchmarks and model sizes.

1  Introduction

Language models (LMs) have shown capability to mimic language effectively across a variety of datasets and tasks  (Brown et al.,,  2020 ; Radford et al.,,  2019 ; Touvron et al.,,  2023 ) . Given a large corpus of text collected across a diverse set of sources, most successful generative LMs are trained on next-token prediction objectives. Consequently, they exhibit a variety of different skillsets, but mimicking text may not always exhibit desirable generation capabilities, e.g., producing intelligent responses to questions or high-quality code. In order to further refine the LM’s capabilities to tailor responses to human preferences, we leverage human-labeled preference datasets and perform task-specific fine-tuning and feedback alignment  (Ouyang et al.,,  2022 ) .

Traditionally, alignment to human preferences has leveraged reinforcement learning via human feedback (RLHF)  (Akrour et al.,,  2011 ; Christiano et al.,,  2017 ) . Equipped with a relatively small dataset of feedback collected from a fine-tuned LM, we train one or more reward models using maximum likelihood estimation (MLE)  (Ouyang et al.,,  2022 ) . Using the trained reward models, we apply a reinforcement learning (RL) algorithm to the LM to maximize those generated rewards. Typically, the RL algorithm of choice is Proximal Policy Optimization (PPO), developed in order to promote training stability for policy gradient algorithms  (Schulman et al.,,  2017 ) . However, despite this, RLHF often remains unstable during training  (Rafailov et al.,,  2024 ) , and especially for on-policy techniques like PPO, the training time remains a concern due to generation through sampling from the LM  (Baheti et al.,,  2023 ) . While offline RL techniques have been attempted to mitigate the training efficiency, they either incur additional training instability, requiring loss clipping or additional penalty terms  (Baheti et al.,,  2023 ; Snell et al.,,  2022 ) .

Recent work has shown that an alternative approach to alignment, Direct Preference Optimization (DPO), can yield a simple MLE objective that is more stable and often outperforms RLHF. Through reframing and reparameterizing the standard RLHF objective with the Bradley-Terry or Plackett-Luce preference models  (Rafailov et al.,,  2024 ) , it entirely bypasses training a reward model and trains significantly faster than on-policy RL techniques that sample from the LM. Extensions that leverage different preference models, such as the Kahneman-Tversky Prospect Theory  (Kahneman and Tversky,,  1979 ; Ethayarajh et al.,,  2024 ) , or generalizations to arbitrary

Ψ

Ψ

\Psi

-preference optimization objectives  (Azar et al.,,  2023 ) . However, while DPO and its extensions presents significant advantages, the aforementioned techniques lack the ability to incorporate arbitrary non-differentiable objectives like RLHF. For instance, it lacks the direct capability to minimize unsafe content or control the text reading level through an additional objective in a sample-efficient manner. Consequently, its applicability as a standalone, sample-efficient alignment framework that is usable in the real world is diminished.

To leverage the strengths of each approach, we propose a hybrid technique that leverages the simplicity of DPO for maximizing preference alignment in feedback datasets, while allowing for arbitrary auxiliary objectives with offline RL. The highlights of our proposed approach, Hybrid Preference Optimization (HPO), are as follows:

•

With roughly ten additional lines of model code on top of KTO, HPO shows the ability to minimize important auxiliary objectives, including various forms of toxicity and reading level, while equalling or surpassing the overall performance achieved by KTO, offline PPO, and DPO.

•

Despite using RL, HPO uses a simple weighted maximum likelihood objective, removing the need for sampling, loss clipping, importance sampling, or bootstrapping, making it

–

more stable and efficient during training compared to prior work using PPO

–

more theoretically principled than techniques using PPO for offline RL

2  Related Work

Traditionally, alignment methods have been based on reinforcement learning via human feedback (RLHF), which typically involves training reward models using maximum likelihood estimation (MLE) and applying an RL algorithm to tune the LM to maximize rewards  (Akrour et al.,,  2011 ; Cheng et al.,,  2011 ; Christiano et al.,,  2017 ; Askell et al.,,  2021 ; Rame et al.,,  2024 ) . RLHF is often performed with on-policy methods such as Proximal Policy Optimization (PPO)  (Schulman et al.,,  2017 ) , but these have been shown to be computationally expensive and often unstable  (Ouyang et al.,,  2022 ) .

To mitigate these issues with RLHF,  Baheti et al., ( 2023 )  propose an offline importance sampling-based approach, reducing training cost yet introducing instability into training that requires clipping.  Snell et al., ( 2022 )  propose an offline approach that adapts Implicit Q-Learning  (Kostrikov et al.,,  2021 ) , but it requires sampling generations from an LM and many additional tricks for stability.  Ethayarajh et al., ( 2024 )  propose an offline-only variant of PPO to reduce training cost, but both PPO and its predecessor, TRPO, require on-policy samples  (Schulman et al.,,  2015 ) .

Direct preference optimization (DPO) style objectives reframe RLHF as a maximum likelihood task by reparameterizing the reward function using a chosen preference model (e.g.,  Bradley and Terry, ( 1952 ) ,  Kahneman and Tversky, ( 1979 ) ). They have shown improvements in performance, stability, and efficiency  (Rafailov et al.,,  2024 ; Ethayarajh et al.,,  2024 ; Azar et al.,,  2023 ) . Extensions have further improved these objectives through the addition of an offset  (Amini et al.,,  2024 ) , rejection sampling  (Liu et al.,,  2023 ; Khaki et al.,,  2024 ) , and in-context learning  (Song et al.,,  2024 ) .

However, these aforementioned methods that optimize for preferences using MLE lack the capability of maximizing arbitrary non-differentiable objectives (e.g., empathy, readability, or safety) without additional data or features, which limits their practical usage. To resolve this,  Liu et al., ( 2024 )  propose a technique for safe DPO, but it is quite limited in its scope.
 Wang et al., ( 2023 )  explore multi-objective learning with DPO using a margin-based approach that is able to optimize arbitrary auxil