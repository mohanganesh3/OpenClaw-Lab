[2510.25065] Reasoning-Aware GRPO using Process Mining

Reasoning-Aware GRPO using Process Mining

Taekhyun Park

Dept. of Data Science

Pusan National University

Busan, Republic of Korea

pthpark1@pusan.ac.kr

&amp;Yongjae Lee  ∗

Dept. of Industrial Engineering

Pusan National University
 Busan, Republic of Korea

yongzzai1102@pusan.ac.kr

&amp;Hyerim Bae

Dept. of Data Science

Pusan National University

Busan, Republic of Korea

hrbae@pusan.ac.kr

These authors contributed equally. Corresponding author

Abstract

Reinforcement learning (RL)-based post-training has been crucial for enabling multi-step reasoning in large reasoning models (LRMs),
yet current reward schemes are typically outcome-centric. We propose PM4GRPO, a reasoning-aware Group Relative Policy Optimization (GRPO)
that augments standard answer/format rewards with signals over the reasoning procedure.
To this end, process mining techniques are utilized to compute a scalar conformance reward that measures how closely a policy model’s
reasoning aligns with the pretrained teacher model.
The empirical results on five benchmarks demonstrate that PM4GRPO significantly outperforms
existing methodologies for GRPO-based post-training. These results highlight
that leveraging process mining for reasoning-aware GRPO effectively enhances the reasoning capabilities of policy models.

K  eywords  Process Mining

⋅  \cdot

Large Reasoning Model

⋅  \cdot

Group Relative Policy Optimization

Code:

Thrillcrazyer/THIP

Models:

THIP-7B

1  Introduction

Large Language Models, capable of multi-step reasoning processes (i.e., chain-of-thought), often referred to as
Large Reasoning Models (LRMs), have demonstrated impressive performance across a wide variety of complex tasks  [ 1 ] .
A key factor enabling such reasoning behaviors is reinforcement learning (RL)-based post-training,
which aligns model policies with high-level reasoning objectives  [ 2 ] .
Among these methods, DeepSeek-R1’s GRPO  [ 3 ]  has emerged as a notable
advancement due to its simplified objective design and more stable optimization dynamics compared to traditional
approaches such as DPO  [ 4 ]  and PPO  [ 5 ] .
Motivated by these advantages, GRPO-based RL post-training frameworks have rapidly gained traction and continue to evolve  [ 6 ] .

Despite these advances, GRPO-inspired approaches focus solely on optimizing final answers, neglecting the underlying reasoning processes  [ 1 ] .
Current reward schemes are typically outcome-centric or depend on superficial textual attributes (e.g., length, formatting consistency, keyword matches)  [ 7 ] .
Such reward formulations fail to capture the process by which the model derives its answer, often leading to suboptimal behaviors such
as unnecessary verbosity, speculative leaps in reasoning, or accidental correctness without genuine understanding  [ 8 ] .

To address this limitation, we adopt the perspective that the reasoning or thinking of an LRM can itself be viewed as a process,
i.e., Thinking is a Process ( THIP ).
The key idea is to extract the reasoning traces of a pretrained, large teacher model in the form of event logs,
and to use a reward signal that measures how well the student model’s self-generated reasoning process aligns with
the teacher’s process. To achieve this, we utilize Process Mining (PM)  [ 9 ] , a set of techniques designed to analyze process execution logs,
enabling us to evaluate reasoning quality at the process-level rather than solely from final outcomes.

In this short paper, we propose a novel reasoning-aware GRPO framework using PM, named  PM4GRPO ,
which integrates PM techniques into the reward design of GRPO. Our contributions are summarized as follows:

•

We introduce  PM4GRPO , a novel GRPO framework featuring a reward model that incorporates the reasoning process itself into post-training.

•

For each query, the policy model derive its own reasoning trace, which is then transformed into a process model.
The framework quantitatively measures how well this process model conforms to the Teacher Model’s reasoning (i.e., conformance checking) and integrates this measure into the reward signal.

•

PM4GRPO does not enforce the policy model to replicate the Teacher Model’s reasoning strictly.
Instead, it encourages reasoning alignment while preserving the model’s freedom of thought, thereby enhancing the effectiveness of reinforcement learning.

The rest of this paper is organized as follows. In Section  2  , we describe the proposed PM4GRPO framework in detail.
Section  3

presents experimental results demonstrating the effectiveness of our approach.
Finally, we conclude the paper in Section  4

with a summary and future research directions.

2  PM4GRPO: Reasoning-Aware GRPO using Process Mining

Figure 1:  Illustration of the Reasoning-Aware GRPO using Process Mining.

To design a reasoning-aware GRPO framework, we propose to integrate PM techniques into the GRPO algorithms.
Our method, illustrated in Figure

1  , enhances the original GRPO by incorporating a process reward that evaluates
how well the reasoning process of the policy model aligns with that of a pretrained teacher model.
Two existing PM techniques are utilized to compute this process reward:
(1) Inductive miner  [ 10 ]  to construct process models from the policy model’s reasoning process, and
(2) Alignment-based conformance checking  [ 11 ]

CC  \operatorname{CC}

to compare the induced process models with the teacher model’s reasoning process.
In this paper, inductive miner and alignment-based conformance checking are denoted by

IM  \operatorname{IM}

and

CC  \operatorname{CC}

, respectively.
Note that these techniques are applied independently to each problem that needs
to be addressed, and the reward is assigned at the problem (i.e., sequence) level
rather than the token level. For each problem, once the reasoning process is completed,
a process model is constructed and compared with the teacher model’s reasoning.

2.1  Group Sequence Policy Optimization

Recently, GRPO is designed to fine-tune language models with group-level normalized rewards [ 3 ] .
Typically, it optimizes the policy model by maximizing the expected reward, which is typically based on the
correctness and format of the final answer. GRPO modifies the standard policy gradient objective by
introducing relative advantages within sets of responses corresponding to the same query [ 12 ] .
However, GRPO performs off-policy correction at the token level via token-wise
importance ratios, while the reward in our framework is only computed at the
problem (i.e., sequence) level after the entire reasoning process is completed.
Zheng et al. [ 13 ]  addressed this issue by proposing a GRPO-inspired framework named Group Sequence Policy Optimization (GSPO).
GSPO defines the importance ratio based on the sequence-level likelihood, ensuring that the unit of optimization matches the sequence-level
reward. Specifically, let

π  θ

\pi_{\theta}

denote the policy model with parameters

θ  \theta

,

x  ∼  𝒟

x\sim\mathcal{D}

be a query sampled from the dataset

𝒟  \mathcal{D}

,

G  G

be the group size and

y  y

be outputs. The objective function of GSPO can be defined as follows:

𝒥

G  ​  S  ​  P  ​  O

​

(  θ  )

\displaystyle\mathcal{J}_{GSPO}(\theta)

=

𝔼

x  ∼  𝒟  ,

{

y  i

}

i  =  1

G

∼

π

θ

o  ​  l  ​  d

(  ⋅  |  x  )

\displaystyle=\mathbb{E}_{x\sim\mathcal{D},{\left\{y_{i}\right\}_{i=1}^{G}}\sim\pi_{\theta_{old}}(\cdot|x)}

(1)

[

1  G

​

∑

i  =  1

G

min  ⁡

(

r  i

​

(  θ  )

​

A  ^

i

,

clip  ⁡

(

r  i

​

(  θ  )

,

1  −  ϵ

,

1  +  ϵ

)

​

A  ^

i

)

]

\displaystyle\quad\bigg[\frac{1}{G}\sum_{i=1}^{G}\operatorname{min}\left(r_{i}(\theta)\hat{A}_{i},\operatorname{clip}\left(r_{i}(\theta),1-\epsilon,1+\epsilon\right)\hat{A}_{i}\right)\bigg]

where

ϵ  \epsilon

is a hyperparameter controlling the clipping range and the group-based
advantage estimation

A  ^

i

\hat{A}_{i}

is defined