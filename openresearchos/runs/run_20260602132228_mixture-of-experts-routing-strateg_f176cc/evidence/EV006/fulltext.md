[2512.01374] Stabilizing Reinforcement Learning with LLMs: Formulation and Practices

Stabilizing Reinforcement Learning with LLMs:

Formulation and Practices

Chujie Zheng

Kai Dang

Bowen Yu  †

†  footnotemark:

Mingze Li

Huiqiang Jiang

Junrong Lin

Yuqiong Liu

Hao Lin

Chencan Wu

Feng Hu

An Yang

Jingren Zhou

Junyang Lin

Qwen Team, Alibaba Inc

Corresponding authors.

Abstract

This paper proposes a novel formulation for reinforcement learning (RL) with large language models, explaining why and under what conditions the true sequence-level reward can be optimized via a surrogate token-level objective in policy gradient methods such as REINFORCE.
Specifically, through a first-order approximation, we show that this surrogate becomes increasingly valid only when both the training–inference discrepancy and policy staleness are minimized.
This insight provides a principled explanation for the crucial role of several widely adopted techniques in stabilizing RL training, including importance sampling correction, clipping, and particularly Routing Replay for Mixture-of-Experts (MoE) models.
Through extensive experiments with a 30B MoE model totaling hundreds of thousands of GPU hours, we show that for on-policy training, the basic policy gradient algorithm with importance sampling correction achieves the highest training stability.
When off-policy updates are introduced to accelerate convergence, combining clipping and Routing Replay becomes essential to mitigate the instability caused by policy staleness.
Notably, once training is stabilized, prolonged optimization consistently yields comparable final performance regardless of cold-start initialization.
We hope that the shared insights and the developed recipes for stable RL training will facilitate future research.

1  Introduction

Reinforcement learning (RL) has become a key technical paradigm for enhancing large language models’ (LLMs) ability to tackle complex problem-solving tasks  (OpenAI,  2024 ; Guo et al.,  2025 ; Yang et al.,  2025 ) , while a stable training process  1

1  1 By  stable training , we refer to a training process in which model performance steadily improves over training steps—reflected in both the training reward and benchmark scores—and, crucially, the model’s internal state evolves smoothly and without abrupt shifts.
The model state can be monitored via a set of diagnostic metrics, including the training-inference KL divergence and entropy reported in our later experiments (§

4  ).
Stable training implies that the model can consistently improve in a healthy manner throughout extended or multi-stage training, with a lower risk of unexpected behaviors.

is crucial for successfully scaling RL.
Due to the contextual nature of language, RL with LLMs usually employs sequence-level rewards, i.e., a scalar score assigned based on the complete model response.
However, mainstream RL algorithms, such as REINFORCE and GRPO, typically employ token-level optimization objectives.
This mismatch between the reward (assigned at the sequence level) and the optimization unit (typically at the token level) raises concerns about the soundness and training stability of such approaches, while some studies have proposed directly adopting sequence-level optimization objectives  (Zheng et al.,  2025 ; Liu et al.,  2025a ) .
In particular, token-level optimization objectives also pose unique challenges for RL training with Mixture-of-Experts (MoE) models.
For instance, the dynamic expert routing mechanism can invalidate the token-level importance sampling ratios in MoE models  (Zheng et al.,  2025 ) .
However, it remains unclear whether optimizing sequence-level rewards using token-level objectives is justified, and if so, to what extent (or under what conditions) such an approach is valid.

In this paper, we propose a novel formulation for RL with LLMs.
The key insight is that, to optimize the expected sequence-level reward, we can employ a surrogate token-level objective as its first-order approximation.
Specifically, this approximation is likely to hold only when both (1) the numerical discrepancy between the training and inference engines (i.e., the training–inference discrepancy) and (2) the discrepancy between the rollout policy that samples responses and the target policy to be optimized (i.e., policy staleness) are minimized.
This insight provides a principled explanation of how several techniques for stabilizing RL training work.
For example,
(1) the importance sampling weight is an inherent component of the surrogate token-level objective under the first-order approximation;
(2) the clipping mechanism can restrain policy staleness by preventing aggressive policy updates;
(3) for MoE models, the Routing Replay approach  (Zheng et al.,  2025 ; Ma et al.,  2025 ) , which fixes the routed experts during policy optimization, can reduce both the training–inference discrepancy and policy staleness.

To empirically validate our insight and investigate practical recipes for stable RL training, we conduct extensive experiments with a 30B MoE model, amounting to hundreds of thousands of GPU hours.
Our main conclusions include:
(1) For on-policy training  2

2  2 In this paper, we use the term  on-policy  to indicate that the rollout policy that samples responses is identical to the target policy to be optimized using these responses (omitting the training–inference discrepancy), while  off-policy  indicates that the two policies are different.

, the basic policy gradient algorithm with importance sampling correction yields the highest training stability;
(2) When off-policy updates are introduced to accelerate convergence, i.e., a large batch of responses is split into mini-batches for multiple gradient updates, combining clipping and Routing Replay becomes necessary to mitigate instability caused by policy staleness;
(3) Once training is stabilized, models with different cold-start initializations consistently achieve comparable final performance.
This motivates future work to focus more on RL itself rather than overly on the specifics of cold-start initialization, as differences arising from the latter are expected to vanish given prolonged RL training.

In summary, this paper makes contributions along two axes:

•

Theoretically, we propose a novel formulation for reinforcement learning with LLMs, revealing the conditions under which optimizing sequence-level rewards via token-level objectives is justified.
Specifically, the validity of the underlying first-order approximation hinges on jointly minimizing the training–inference discrepancy and policy staleness.

•

Empirically, through extensive experiments with MoE models spanning hundreds of thousands of GPU hours, we demonstrate that several techniques that preserve the validity of the first-order approximation consistently exhibit practical efficacy in stabilizing RL training, particularly the Routing Replay approach tailored for MoE models.
We hope that the developed recipes for stable RL training will facilitate future research.

2  Formulation for Reinforcement Learning with LLMs

2.1  Notation

We define an autoregressive LLM parameterized by

θ  \theta

as a policy

π  θ

\pi_{\theta}

.
We use

x  x

to denote an input prompt and

𝒟  \mathcal{D}

as the prompt set.
Under the policy

π  θ

\pi_{\theta}

, the likelihood of a response

y  y

to a prompt

x  x

is denoted as

π  θ

​

(

y  |  x

)

=

∏

t  =  1

|  y  |

π  θ

​

(

y  t

|

x  ,

y

&lt;  t

)

\pi_{\theta}(y|x)=\prod_{t=1}^{|y|}\pi_{\theta}(y_{t}|x,y_{&lt;t})

where

|  y  |

|y|

is the number of tokens in

y  y

.
Given the contextual nature of language, we focus on the sequence-level reward setting, where a whole response

y  y

is assigned a single scalar reward

R  ​

(  x  ,  y  )

R(x,y)

.
We do not consider the value-based setting (e.g., PPO,  Schulman et al.  2017  ), where policy optimization is steered by a value model that assigns scalar scores t