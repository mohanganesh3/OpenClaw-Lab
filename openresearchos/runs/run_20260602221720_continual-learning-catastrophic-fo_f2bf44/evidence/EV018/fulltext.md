[2509.25240] HAMMER: Hamiltonian Curiosity Augmented Large Language Model Reinforcement

HAMMER : Hamiltonian Curiosity Augmented Large Language Model Reinforcement

Ming Yang  1,2

Xiaofan Li  3,2∗

Zhiyuan Ma  2

Dengliang Shi  2

Jintao Du  2

Yu Cheng  2

Weiguo Zheng  1

1 Fudan University  2 Tiansuan Lab, Ant Group Co., Ltd.  3 East China Normal University

yangm24@m.edu.cn, funzi@stu.ecnu.edu.cn

{mazhiyuan.mzy,dengliang.sdl,lingke.djt,cy122623}@antgroup.com

zhengweiguo@fudan.edu.cn

Equal contribution.
Ming Yang: idea, code, experiments, and writing.
Xiaofan Li: experiment setup and writing.
Work done during internship at Ant Group.Corresponding author.

Abstract

Recent curriculum reinforcement learning for large language models (LLMs) typically rely on difficulty-based annotations for data filtering and ordering.
However, such methods suffer from local optimization, where continual training on simple samples in the early steps can cause the policy to lose its exploration.
We propose a novel schema, namely
 Hamiltonian curiosity augmented large language model reinforcement (HAMMER) ,
that transfers diversity metrics, commonly used in dataset evaluation, into the dynamic reinforcement learning procedure, where training samples are ordered via a minimum-semantic Hamiltonian path
making the initial training retrain more exploration.
From a theoretical perspective of generalization bounds, diversity-driven ordering facilitates stable convergence.
Empirical evaluations indicate that  HAMMER  stimulates model “curiosity” and consistently achieves a 3% to 4% average accuracy gain across diverse inference benchmark.

1  Introduction

Recently, Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful tool for enhancing complex reasoning in large language models (LLMs), significantly boosting their reasoning capabilities  (Luong et al.,  2024 ; Zhang et al.,  2024b ; Lambert et al.,  2025 ) . During training, LLMs generate diverse responses to prompts and receive corresponding rewards  (Guo et al.,  2025 ; Shao et al.,  2024 ; Team et al.,  2025 ) . By learning from outcome reward, these models develop the ability to produce more comprehensive reasoning traces  (Chen et al.,  2025a ; DeepSeek-AI et al.,  2025 ) , leading to improved performance on downstream tasks. The success of large reasoning models (e.g., OpenAI-o1  (Jaech et al.,  2024 )  and DeepSeek-R1  (DeepSeek-AI et al.,  2025 ) ) demonstrates that RLVR effectively expands the capabilities of LLMs.

Group Relative Policy Optimization (GRPO) proposed by  Shao et al. ( 2024 )  is a key RLVR algorithm that extends Proximal Policy Optimization (PPO) proposed by  Schulman et al. ( 2017 ) , by sampling groups of responses to estimate group-relative advantages. Given reward

r  r

, group size

G  G

, policy ratio

ρ  t

=

π  θ

​

(

o  t

|

q  ,

o

&lt;  t

)

π

θ  old

​

(

o  t

|

q  ,

o

&lt;  t

)

\rho_{t}=\frac{\pi_{\theta}(o_{t}|q,o_{&lt;t})}{\pi_{\theta_{\text{old}}}(o_{t}|q,o_{&lt;t})}

with bound

ε  \varepsilon

, GRPO’s objective function is

𝒥  ​

(  θ  )

=

𝔼  ​

{

1  G

​

∑

i  =  1

G

1

|

o  i

|

​

∑

t  =  1

|

o  i

|

min  ⁡

(

ρ

i  ,  t

​

A  ^

i  ,  t

,

c  ​  l  ​  i  ​  p  ​

(

ρ

i  ,  t

,

1  −  ε

,

1  +  ε

)

​

A  ^

i  ,  t

)

−

β  ​

𝔻  KL

}

,

\mathcal{J}(\theta)=\mathbb{E}\left\{\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_{i}|}\sum_{t=1}^{|o_{i}|}\min\left(\rho_{i,t}\hat{A}_{i,t},clip\left(\rho_{i,t},1-\varepsilon,1+\varepsilon\right)\hat{A}_{i,t}\right)-\beta\mathbb{D}_{\text{KL}}\right\},

where KL divergence to reference policy is

𝔻  KL

\mathbb{D}_{\text{KL}}

with penalty factor

β  \beta

.
The normalized advantage is

A  ^

i  ,  t

=

r

i  ,  t

−

mean  ​

(

r

i  ,  t

)

std  ​

(

r

i  ,  t

)

\hat{A}_{i,t}=\frac{r_{i,t}-\text{mean}(r_{i,t})}{\text{std}(r_{i,t})}

. The expectation

𝔼  \mathbb{E}

follows

(  q  ,  a  )

∼  𝒳

(q,a)\sim\mathcal{X}

and

{

o  i

}

i  =  1

G

∼

π  θ

(  ⋅  |  q  )

\{o_{i}\}_{i=1}^{G}\sim\pi_{\theta}(\cdot|q)

.
Subsequently, variants of GRPO, like Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO)  (Yu et al.,  2025 ) , were proposed to optimize the GRPO.

Beyond optimization algorithms, some works explore  data-centric  strategies to improve efficiency. Inspired by human education,
Curriculum Learning (CL) has been applied to LLM reinforcement  (Bengio et al.,  2009 ; Narvekar et al.,  2020 ) , most studies rely on difficulty-based sequencing of Chain-of-Thought (CoT) annotations  (Parashar et al.,  2025 ; Qiu et al.,  2025 ) . Such approaches typically mimic “easy-to-hard” progressions but require costly difficulty assessments, often via  pass@k  testing or advanced-model labeling (e.g., OpenAI-o1  (Jaech et al.,  2024 ) , Deepseek-R1  (DeepSeek-AI et al.,  2025 ) ) and suffer from local optimization. We consider adopting the diversity order, but diversity-based classical methods such as Coreset Selection (CS)  (Koh &amp; Liang,  2017 ; Sener &amp; Savarese,  2017 ; Lewis &amp; Catlett,  1994 ; Zhang et al.,  2024a )  are all sampling methods whose reduction-oriented design leads to performance bottlenecks  (Mehra et al.,  2025 ) .

1.1  Motivation

Reinforcement learning with LLMs often exhibits high variance and unstable convergence, particularly in the early stages of training  (Chen et al.,  2025b ) . Traditional curriculum learning  (Narvekar et al.,  2020 )  typically follows an “easy-to-hard” strategy  (Parashar et al.,  2025 ; Qiu et al.,  2025 ) .
However, in RLVR, such naive difficulty-based training often fails: the model quickly exploits easy samples for consistent rewards, while harder ones incur repeated penalties. This early imbalance discourages exploration, leading the policy to overfit to easy problems early and become trapped in local optima, ultimately slowing convergence.
Our ablation study confirms this inefficiency (Figure

4(b)  ).
To improve training, we propose a different perspective:  diversity can effectively guide RLVR . Presenting semantically diverse samples early allows the model to explore the input space more thoroughly, reduce the generalization gap, and accelerate convergence, as theoretically justified in Section

4  .
In short, we transform diversity from a static dataset property to an active principle for curriculum design in LLM reinforcement learning.

1.2  Our Approach and Contributions

In this paper, we present a novel and effective schema,  H  amiltonian Curiosity  A ug M ented Large Language  M od E l  R einforcement (HAMMER) ,
which transfers diversity metrics from large model data evaluation into the dynamic process of reinforcement learning.
The schema consists of two main components.
First, it leverages the backbone LLM to obtain semantic similarity embeddings.
Compared to external embedding models, this approach generates sentence representations that are more consistent with the model’s internal training dynamics.
Second, the embeddings are used to compute pairwise semantic similarity, and a Hamiltonian Curiosity Order is constructed to define a curriculum learning sequence.
This process can be viewed as solving a Hamiltonian cycle that minimizes semantic similarity, enabling the model to greedily encounter the most diverse samples early in training.
As a result, the model can achieve partial convergence on some samples early, improving the stability of reinforcement learning.

From a learning-theoretic perspective, we derive the generalization bound of  HAMMER .
Theorem

1

shows that early diverse training does not compromise the optimal policy,
while Theorem

2

demonstrates that diverse subsets effectively tighten the generalization bound.
Moreover, Theorem

3

establishes that optimizing  HAMMER ’s semantic diversity path is equivalent to maximizing the dataset diversity score, which captures the overall likelihood that a sample substantially differs from the rest of the dataset (see