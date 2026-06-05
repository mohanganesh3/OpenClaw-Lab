[2510.23027] Towards Stable and Effective Reinforcement Learning for Mixture-of-Experts

Towards Stable and Effective Reinforcement
 Learning for Mixture-of-Experts

Di Zhang †‡  Xun Wu †  Shaohan Huang †  Yaru Hao †  Li Dong †

Zewen Chi †

Zhifang Sui ‡

Furu Wei †

†  Microsoft Research

‡  Peking University

https://aka.ms/GeneralAI

Abstract

Recent advances in reinforcement learning (RL) have substantially improved the training of large-scale language models, leading to significant gains in generation quality and reasoning ability. However, most existing research focuses on dense models, while RL training for Mixture-of-Experts (MoE) architectures remains underexplored. To address the instability commonly observed in MoE training, we propose a novel router-aware approach to optimize importance sampling (IS) weights in off-policy RL. Specifically, we design a rescaling strategy guided by router logits, which effectively reduces gradient variance and mitigates training divergence. Experimental results demonstrate that our method significantly improves both the convergence stability and the final performance of MoE models, highlighting the potential of RL algorithmic innovations tailored to MoE architectures and providing a promising direction for efficient training of large-scale expert models.

1  Introduction

Reinforcement Learning (RL) has demonstrated strong potential in enhancing LLM reasoning through inference-time scaling, as exemplified by the OpenAI-o1 model  (Ope,  24 ) . More recently, DeepSeek-R1  (GYZ  +  ,  25 )  has shown that reinforcement learning with verifiable rewards (RLVR), which relies on simple, rule-based reward functions, can elicit emergent reasoning abilities and deliver substantial performance gains on challenging tasks such as mathematical problem solving and program synthesis  (YLY  +  ,  25 ; TBB  +  ,  25 ; CLG  +  ,  25 ) . In parallel, Mixture-of-Experts (MoE) architectures have emerged as an efficient approach to scaling model capacity  (FZS,  22 ) . By activating only a small subset of experts per token, MoE models achieve higher parameter efficiency while keeping computation cost nearly constant, making them particularly attractive for large-scale RL training where compute efficiency is critical.

Despite these advances, applying RLVR to MoE models remains highly challenging due to stability issues  ( ZLL  +  25b,  ; CLG  +  ,  25 ; YLY  +  ,  25 ) . A central difficulty is  router fluctuation : the set of experts selected for the same input token may vary significantly across policy updates  (DDM  +  ,  22 ;  ZLL  +  25b,  ) . Such routing drift not only increases the variance of importance sampling (IS) weights, but can also destabilize optimization and even cause reward collapse. Furthermore, most implementations adopt token-level IS ratios  (SWD  +  ,  17 ) , which are poorly aligned with the sequence-level rewards typically used in RLVR, introducing additional variance and further compounding instability. Prior work such as GSPO proposes computing IS ratios at the sequence level to mitigate this issue, yet it does not fundamentally address the instability introduced by router fluctuations.

We experimentally explored two natural stabilization strategies: freezing the router parameters and routing replay, where routing decisions are cached and reused across updates. However, both approaches proved unsatisfactory—freezing the router hampers model adaptability, while routing replay restricts exploration and degrades performance. These results suggest that rigid control over the router is suboptimal, and a more flexible mechanism is needed.

To address these challenges, we propose  Router-Shift Policy Optimization  ( RSPO ), an RL algorithm specifically designed for MoE architectures to achieve stable and efficient training. Instead of fully constraining the router,  RSPO  introduces a  router shift ratio , computed from router scores between the current and old policies. This ratio quantifies the degree of routing deviation for each token and is used to softly rescale IS weights. In doing so,  RSPO  reduces gradient variance and mitigates divergence caused by router instability, while preserving the router’s capacity to adapt.

We validate the effectiveness of  RSPO  on the Qwen2.5 model for the countdown task and on Qwen3-30B-A3B across multiple mathematical reasoning benchmarks. Extensive experiments demonstrate that  RSPO  achieves more stable training and superior performance. Our main contributions are summarized as follows:

•

We propose  RSPO , which combines a router-aware rescaling strategy with sequence-level aggregation of importance sampling ratios, effectively stabilizing off-policy RL training for MoE models.

•

Unlike router freezing or routing replay,  RSPO  adopts a soft adjustment mechanism: it leverages a router shift ratio to quantify routing deviation for each token and adaptively reweight updates, limiting overly large updates while retaining router flexibility.

•

We evaluate  RSPO  on both the countdown task and multiple mathematical reasoning benchmarks, demonstrating its stability and effectiveness, and highlighting the importance of incorporating router-aware strategies and sequence-level importance weighting in RL for MoE models.

2  Preliminaries

Group Relative Policy Optimization (GRPO)

Traditional reinforcement learning (RL) algorithms, such as Proximal Policy Optimization (PPO)  (SWD  +  ,  17 ) , have been widely applied to RL training of large language models (LLMs). However, PPO still suffers from high computational cost and challenges in tuning value model. To address these limitations, GRPO  (SWZ  +  ,  24 )  builds upon PPO by removing the value model and introducing a group-relative advantage estimation.

Specifically, for a given query

x  x

, GRPO samples

G  G

candidate responses

{

y  i

}

i  =  1

G

\{y_{i}\}_{i=1}^{G}

, computes their relative advantages within the group, and optimizes the following objective:

𝒥  GRPO

​

(  θ  )

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

θ  old

(  ⋅  |  x  )

​

[

1  G

​

∑

i  =  1

G

1

|

y  i

|

​

∑

t  =  1

|

y  i

|

min

⁡

(

w

i  ,  t

​

(  θ  )

​

A  ^

i  ,  t

,

clip

​

(

w

i  ,  t

​

(  θ  )

,

1  −  ϵ

,

1  +  ϵ

)

​

A  ^

i  ,  t

)

]

,

\displaystyle\mathcal{J}_{\text{GRPO}}(\theta)=\mathbb{E}_{x\sim\mathcal{D},\,\{y_{i}\}_{i=1}^{G}\sim\pi_{\theta_{\text{old}}}(\cdot|x)}\!\!\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|y_{i}|}\sum_{t=1}^{|y_{i}|}\min\!\Big(w_{i,t}(\theta)\,\hat{A}_{i,t},\,\text{clip}\!\big(w_{i,t}(\theta),1-\epsilon,1+\epsilon\big)\hat{A}_{i,t}\Big)\right],

(1)

where

G  G

denotes the number of responses generated for each query

x  x

. For each token

y

i  ,  t

y_{i,t}

, the importance sampling ratio

w

i  ,  t

​

(  θ  )

w_{i,t}(\theta)

and the group-normalized advantage

A  ^

i  ,  t

\hat{A}_{i,t}

are given by:

w

i  ,  t

​

(  θ  )

=

π  θ

​

(

y

i  ,  t

∣

x  ,

y

i  ,

&lt;  t

)

π

θ  old

​

(

y

i  ,  t

∣

x  ,

y

i  ,

&lt;  t

)

,

A  ^

i

=

A  ^

i  ,  t

=

r  ​

(  x  ,

y  i

)

−

mean

​

(

{

r  ​

(  x  ,

y  i

)

}

i  =  1

G

)

std

​

(

{

r  ​

(  x  ,

y  i

)

}

i  =  1

G

)

.

\displaystyle w_{i,t}(\theta)=\frac{\pi_{\theta}(y_{i,t}\mid x,y_{i,&lt;t})}{\pi_{\theta_{\text{old}}}(y_{i,t}\mid x,y_{i,&lt;t})},\qquad\hat{A}_{i}=\hat{A}_{i,t}=\frac{r(x,y_{i})-\text{mean}\!\big(\{r(x,y_{i})\}_{i=1}^{G}\big)}{\text{std}\!\big(\{r(x,y_{i})\}_{i=1}^{G}\big)}.

(2)

Group Sequence Policy Optimization (GSPO).

Unlike GRPO, which performs importance sampling and clipping at the token level, Group Sequence Policy Optimization (GSPO)  ( ZLL  +  25b,  )  defines the importance sampling ratio at the  sequence level  and applies clipping accordingly. This modification corrects the misalignment between sequence-level rewards and token-level importance sampling ratios present in GRPO under the RLVR objective. Th