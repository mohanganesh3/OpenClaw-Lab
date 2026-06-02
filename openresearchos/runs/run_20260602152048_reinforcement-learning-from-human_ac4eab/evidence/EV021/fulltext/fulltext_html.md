[2510.01555] Rethinking KL Regularization in RLHF: From Value Estimation to Gradient Optimization

Rethinking KL Regularization in RLHF: From Value Estimation to Gradient Optimization

Kezhao Liu

liukzh9@mail2.sysu.edu.cn 
&amp;Jason Klein Liu  1

1  footnotemark:

1

jasonkleinlove@gmail.com 
&amp;Mingtao Chen

cmtmeton@gmail.com 
&amp;YiMing Liu

liuym225@mail2.sysu.edu.cn

Co-first authors.Corresponding author who led the project.

Abstract

Reinforcement Learning from Human Feedback (RLHF) leverages a Kullback-Leibler (KL) divergence loss to stabilize training and prevent overfitting. However, in methods such as GRPO, its implementation may be guided by principles from numerical value estimation—a practice that overlooks the term’s functional role as an optimization loss. To analyze this issue, we establish a unified framework that connects two seemingly distinct implementation styles: using the mathematical term

𝒌  𝒏

\bm{k_{n}}

as a detached coefficient for the policy’s score function (‘

𝒌  𝒏

\bm{k_{n}}

in reward ’) or as a direct loss function through which gradients are propagated (‘

𝒌  𝒏

\bm{k_{n}}

as loss ’). We show that the latter can always be analyzed via an equivalent gradient coefficient in the former, unifying the two perspectives. Through this framework, we prove that the conventional ‘

𝒌  𝟏

\bm{k_{1}}

in reward ’ (like PPO) is the principled loss for Reverse KL (RKL) regularization. We further establish a key finding: under on-policy conditions, the ‘

𝒌  𝟐

\bm{k_{2}}

as loss ’ formulation is, in fact, gradient-equivalent to ‘

𝒌  𝟏

\bm{k_{1}}

in reward ’. This equivalence, first proven in our work, identifies both as the theoretically sound implementations of the RKL objective. In contrast, we show that the recently adopted ‘

𝒌  𝟑

\bm{k_{3}}

as loss ’ (like GRPO) is merely a first-order, biased approximation of the principled loss. Furthermore, we argue that common off-policy implementations of ‘

𝒌  𝒏

\bm{k_{n}}

as loss ’ methods are biased due to neglected importance sampling, and we propose a principled correction. Our findings provide a comprehensive, gradient-based rationale for choosing and correctly implementing KL regularization, paving the way for more robust and effective RLHF systems.

1  Introduction

The training of state-of-the-art Large Language Models (LLMs) is a multistage process. Following large-scale pretraining and the Supervised Fine-Tuning (SFT) to learn instruction-following behaviors, a final post-training stage, Reinforcement Learning from Human Feedback (RLHF), is often employed. The objective of RLHF is twofold; it serves to align the model more closely with complex human values  (Ouyang et al.,  2022 )  and, increasingly, to push the performance limits in specialized reasoning tasks such as mathematics and code generation, as seen in models such as DeepSeek-Math  (Shao et al.,  2024 ) . A core component of this RLHF process is  KL regularization , implemented through a loss term derived from the Kullback-Leibler (KL) divergence  (Kullback &amp; Leibler,  1951 ) . The  KL loss  serves not only to stabilize the training process but also to improve generalization by preventing the policy from overfitting the reward signal and deviating excessively from the initial SFT model  (Ouyang et al.,  2022 ; Stiennon et al.,  2020 ) .

Despite the critical role of the KL loss, its theoretical foundations in the optimization context remain underexplored. The choice of its specific mathematical form is often guided by principles from numerical  value estimation , not from the perspective of gradient-based  optimization . This category error has led to a proliferation of ad-hoc implementations and suboptimal algorithm designs, exemplified by recent methods like GRPO that adopt sure estimators under the mistaken assumption that good value estimation properties translate to effective gradients. This paper argues that a gradient-centric perspective is essential for designing robust and effective RLHF algorithms.

We perform a systematic, gradient-based analysis of the KL loss to address these issues. We first establish a unified framework that connects two seemingly distinct implementation styles: using the mathematical term

𝒌  𝒏

\bm{k_{n}}

as a detached coefficient (‘

𝒌  𝒏

\bm{k_{n}}

in reward ’) or as a direct loss function (‘

𝒌  𝒏

\bm{k_{n}}

as loss ’). This framework allows us to analyze any implementation by examining its equivalent gradient coefficient. Using this lens, we first use the ‘

𝒌  𝟏

\bm{k_{1}}

as loss ’ case as a counterexample to demonstrate the mistakes of the value estimation perspective. We then prove that the conventional ‘

𝒌  𝟏

\bm{k_{1}}

in reward ’ and the ‘

𝒌  𝟐

\bm{k_{2}}

as loss ’ formulations are, in fact, gradient equivalent and represent the principled approach to reverse KL regularization. Finally, we analyze popular alternatives like ‘

𝒌  𝟑

\bm{k_{3}}

as loss ’, revealing their nature as biased approximations, and address a common but critical bug in their off-policy implementation.

Our main contributions are threefold:

1.

A Gradient-Centric Framework for KL Regularization.  We revisit KL regularization in RLHF, shifting the focus from value estimation to gradient optimization. We demonstrate the necessity of this perspective using ‘

𝒌  𝟏

\bm{k_{1}}

as loss ’ as a powerful counterexample, showing how an unbiased value estimator yields a completely ineffective optimization signal.

2.

Identification of Principled KL Implementations.  We prove that the conventional ‘

𝒌  𝟏

\bm{k_{1}}

in reward ’ formulation correctly implements the KL gradient. We further establish a key, previously unrecognized equivalence: ‘

𝒌  𝟏

\bm{k_{1}}

in reward ’ is gradient-equivalent to ‘

𝒌  𝟐

\bm{k_{2}}

as loss ’. This discovery solidifies both as theoretically sound choices for KL regularization.

3.

Analysis of Alternative Implementations and a Practical Correction.  We analyze popular alternative approaches, showing that ‘

𝒌  𝟑

\bm{k_{3}}

as loss ’ (used in GRPO) is a biased first-order approximation of the principled gradient, leading to weaker regularization or potential instability. Furthermore, we identify a common pitfall in off-policy algorithms where ‘

k  n

k_{n}

as loss’ methods are often implemented without correct importance sampling, and we provide a principled correction for this bias.

2  Related Work

KL Value Estimation.

Since the expectation of the KL divergence is often intractable, it is typically estimated by Monte Carlo sampling. Prior analyses have primarily assessed these estimators as value estimators  (John,  2020 ) , characterizing

k  1

k_{1}

as unbiased but high variance,

k  2

k_{2}

as biased but lower variance, and

k  3

k_{3}

as an ”optimal”, low variance and unbiased choice. We first challenge the claimed superiority of

k  3

k_{3}

as a value estimator, showing that its advertised properties often fail to hold in practical settings. More importantly, this emphasis is misplaced when these estimators are used for regularization in RLHF. We show that a gradient-centric perspective is essential: conclusions drawn from value estimation do not necessarily translate into effective optimization.

RLHF Methods.

OpenRLHF  (Hu et al.,  2024 )  is the first framework that uses vLLM  Kwon et al. ( 2023 )  to accelerate the rollout phase in RLHF training, and incorporates a variety of techniques that make RLHF training more stable. Since then, several training frameworks have emerged, including Verl  Sheng et al. ( 2024 ) , slime Zhu et al. ( 2025 ) , and ROLL  Wang et al. ( 2025 ) . These frameworks primarily support PPO  (Ouyang et al.,  2022 )  and its variants, focusing on improving training stability, particularly addressing challenges in training the critic model. VAPO  Yue et al. ( 2025 )  proposed pretraining the critic model to mitigate these issues, while GRPO and Rei