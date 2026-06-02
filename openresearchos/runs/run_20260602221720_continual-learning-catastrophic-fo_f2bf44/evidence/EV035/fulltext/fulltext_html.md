[2509.07430] The Choice of Divergence: A Neglected Key to Mitigating Diversity Collapse in Reinforcement Learning with Verifiable Reward

The Choice of Divergence: A Neglected Key to Mitigating Diversity Collapse in Reinforcement Learning with Verifiable Reward

Long Li  1

,  Jiaran Hao  1

,  Jason Klein Liu  1

,  Zhijian Zhou  2

,  Xiaoyu Tan  1

Wei Chu  1

,  Zhe Wang  3

,  Shirui Pan  3

,  Chao Qu  1,2

,  Yuan Qi  1,2

1 INFLY TECH,  2 Fudan University,  3 Griffith University

{seamoke111@gmail.com}

Corresponding author.

Abstract

A central paradox in fine-tuning Large Language Models (LLMs) with Reinforcement Learning with Verifiable Reward (RLVR) is the frequent degradation of multi-attempt performance (Pass@k) despite improvements in single-attempt accuracy (Pass@1). This is often accompanied by catastrophic forgetting, where models lose previously acquired skills. While various methods have been proposed, the choice and function of the divergence term have been surprisingly unexamined as a proactive solution. We argue that standard RLVR objectives—both those using the mode-seeking reverse KL-divergence and those forgoing a divergence term entirely—lack a crucial mechanism for knowledge retention. The reverse-KL actively accelerates this decay by narrowing the policy, while its absence provides no safeguard against the model drifting from its diverse knowledge base.

We propose a fundamental shift in perspective: using the divergence term itself as the solution. Our framework, Diversity-Preserving Hybrid RL (DPH-RL), leverages mass-covering f-divergences (like forward-KL and JS-divergence) to function as a “rehearsal mechanism”. By continuously referencing the initial policy, this approach forces the model to maintain broad solution coverage. Extensive experiments on math and SQL generation demonstrate that DPH-RL not only resolves the Pass@k degradation but improves both Pass@1 and Pass@k in- and out-of-domain. Additionally, DPH-RL is more training-efficient because it computes f-divergence using generator functions, requiring only sampling from the initial policy and no online reference model. Our work highlights a crucial, overlooked axis for improving RLVR, demonstrating that the proper selection of a divergence measure is a powerful tool for building more general and diverse reasoning models.

https://github.com/seamoke/DPH-RL

Figure 1:  The left panel shows the performance of a model trained exclusively on the Bird dataset, evaluated across different test sets. The X-axis represents the degree of domain and task divergence from the training set. Specifically, Bird is the original SQL dataset, Spider is another dataset for the same SQL task but from a different domain, and Math represents the average result for a class of mathematical tasks. The datasets are ordered from right to left, showing an increasing divergence from the distribution of our RL training set. The Y-axis shows the relative Pass@k score, calculated as (Current Model’s Pass@k) / (Base Model’s Pass@k). The right panel visualizes the distributions of reverse-KL and forward-KL.

1  Introduction

Reinforcement Learning with Verifiable Rewards (RLVR) has recently demonstrated significant success in enhancing the mathematical and coding capabilities of Large Language Models (LLMs)

(Yue et al.,  2025a ; OpenAI,  2023 ; Li et al.,  2024d ; Guo et al.,  2025 ) . Despite significant successes, a critical paradox shadows the field: While RLVR-tuned models consistently improve the probability of generating a correct solution in a single attempt (Pass@1), they often fail to enhance—and in some cases, actively degrade—performance when multiple solution attempts are permitted (Pass@k), especially when compared to their base models  (Yue et al.,  2025a ) . This puzzling observation has given rise to the troubling hypothesis that RLVR may not be endowing models with genuinely new reasoning capabilities, but rather re-weighting and narrowing their focus onto a few known reasoning paths at the expense of solution diversity.

This discrepancy is often linked to entropy collapse, where the model’s output distribution narrows, reducing solution diversity. Some approaches tackle this by forgoing a KL-divergence term entirely  (Yu et al.,  2025 ) , yet still observe a decline in diversity. As shown in the left panel of Figure

1  , both GRPO  (Shao et al.,  2024 )  and DAPO  (Yu et al.,  2025 )  without KL exhibit a significant decline in Pass@k after training. The issue of maintaining exploration is not new; traditional reinforcement learning has a rich history of using entropy regularizers, such as in Soft Actor-Critic (SAC), which is based on the soft Bellman equation  (Haarnoja et al.,  2018a ) . However, in practice, the temperature coefficient

α  \alpha

that balances the reward and entropy is notoriously difficult to tune. Subsequent work on SAC reformulated the problem as a dual optimization with a lower bound on entropy  (Haarnoja et al.,  2018b ) , but this often amounts to transforming one difficult problem into another . Within the LLM domain, entropy regularization has also been explored, but it can create an undesirable trade-off by negatively impacting Pass@1 performance. As an alternative, a separate line of research has focused on directly optimizing the Pass@k metric itself  ( Mahdavi et al.,  ; Walder &amp; Karkhanis,  2025 ) . Another research line resorts to the external help from the stronger reasoning model like deepseek-R1 to break through the boundary of the base model  ( Yan et al.,  ; Dong et al.,  2025a )

In this paper, we approach this from a different perspective, focusing on the KL-divergence term used in most RL fine-tuning for LLMs. The standard choice is the reverse-KL divergence,

D  KL

(

π  θ

|  |

π

r  ​  e  ​  f

)

=

𝔼

π  θ

log

π  θ

π

r  ​  e  ​  f

D_{\text{KL}}(\pi_{\theta}||\pi_{ref})=\mathbb{E}_{\pi_{\theta}}\log\frac{\pi_{\theta}}{\pi_{ref}}

(Kazemnejad et al.,  2024 ; Schulman et al.,  2017 ; Shao et al.,  2024 ) . It is well-established that reverse-KL is mode-seeking  (Bishop &amp; Nasrabadi,  2006 )  (as illustrated in Figure 1), a property that theoretically encourages the policy to converge on a single high-probability solution, thus suppressing diversity. We validated this with a simple experiment in Figure

5  : When an RL model is trained with a reverse-KL objective, it almost exclusively produces a single solution style. However, training with a forward-KL objective allows it to generate three or more styles in 60% of cases. Furthermore, we observe that models trained with  reverse-KL  or  no KL  term suffer from catastrophic forgetting—a gradual decline in accuracy on problems the model could previously solve correctly. As illustrated in Figure

4  , after RL training, GRPO and DAPO policy only solve around

85  %

85\%

queries it previously solved correctly. This phenomenon is a known challenge in sequential learning paradigms with neural networks, not a fundamental flaw of RL, which in principle does not suffer from this in tabular cases  (Hamadanian et al.,  2025 ) .

Based on these observations, we propose several alternatives to the reverse-KL. A straightforward example is the forward-KL divergence,

D  KL

(

π

r  ​  e  ​  f

|  |

π  θ

)

D_{\text{KL}}(\pi_{ref}||\pi_{\theta})

. Theoretically, forward-KL is mass-covering, which penalizes the policy for failing to assign probability mass to any solution present in the reference distribution, thereby preserving diversity. From a practitioner’s standpoint, the forward-KL objective

𝔼

π

r  ​  e  ​  f

​

log  ⁡

π

r  ​  e  ​  f

π  θ

\mathbb{E}_{\pi_{ref}}\log\frac{\pi_{ref}}{\pi_{\theta}}

, which computes its expectation over samples from the reference policy, effectively creates an “anchor dataset.” This forces the model to continuously revisit and rehearse its original knowledge base. In contrast, the reverse-KL objective samples from the current