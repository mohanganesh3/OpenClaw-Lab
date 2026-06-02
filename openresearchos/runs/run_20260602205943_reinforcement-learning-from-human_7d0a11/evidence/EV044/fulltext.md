[2410.15096] GDPO: Learning to Directly Align Language Models with Diversity Using GFlowNets

GDPO: Learning to Directly Align Language Models
 with Diversity Using GFlowNets

Oh Joon Kwon  Daiki E. Matsunaga  Kee-Eung Kim

KAIST AI, Seoul, Korea

{ojkwon, dematsunaga, kekim}@ai.kaist.ac.kr

Abstract

A critical component of the current generation of language models is preference alignment,
which aims to precisely control the model’s behavior to meet human needs and values.
The most notable among such methods is Reinforcement Learning with Human Feedback (RLHF) and its offline variant Direct Preference Optimization (DPO),
both of which seek to maximize a reward model based on human preferences.
In particular, DPO derives reward signals directly from the offline preference data,
but in doing so overfits the reward signals and generates suboptimal responses that may contain human biases in the dataset.
In this work, we propose a practical application of a diversity-seeking RL algorithm called GFlowNet-DPO (GDPO) in an offline preference alignment setting to curtail such challenges.
Empirical results show GDPO can generate far more diverse responses than the baseline methods that are still relatively aligned with human values in dialog generation and summarization tasks.

1  Introduction

The goal of language model (LM) alignment is to steer the model’s generation to produce outputs deemed desirable to human needs and values.
Reinforcement learning with human feedback (RLHF) is one such critical technique, as evidenced by notable applications such as ChatGPT  Achiam et al. ( 2023 )  and Claude  Ouyang et al. ( 2022 ) .
The classical RLHF pipeline involves training the reward model from human feedback and optimizing the policy with the learned reward model by RL, e.g. proximal policy optimization (PPO)  Schulman et al. ( 2017 ) .
Despite its effectiveness, this pipeline is known to be sample-inefficient and unstable.
Moreover, its optimal performance hinges on the code-level details and meticulously tuned hyperparameters, making it difficult to reproduce its success with limited computational resources.

To simplify this complex RLHF pipeline, recent works have explored offline learning algorithms such as Direct Preference Optimization (DPO)  Rafailov et al. ( 2024b ) ,
which aims to improve the efficiency and stability of RLHF by leveraging human feedback data to derive reward signals directly.
While convenient and compute-efficient due to the offline nature of its training,
theoretical results suggest that DPO tends to overfit on the reward signal  Azar et al. ( 2023 )  and learns to reject undesired responses at a faster rate than it learns to accept desired responses, limiting the model’s learning capacity  Feng et al. ( 2024 ) .
To overcome these challenges, other works  Azar et al. ( 2023 ); Xu et al. ( 2024 ); Zhao et al. ( 2023b )  have proposed regularized objectives, but none directly aims to model the diversity of the distribution.
Instead, they tend to settle around local modes in reward distributions, which may be suboptimal. This lack of diversity may hinder its applicability to creative use-cases  Castricato et al. ( 2022 )  or under-represent certain demographics in the LM’s responses  Lahoti et al. ( 2023 ) .

In this work, we directly tackle the goal of preference alignment from the perspective of Bayesian inference.
In particular, we utilize GFlowNets  Bengio et al. ( 2023 ) , which has recently been introduced as a principled method for amortized sampling of multimodal distributions in proportion to a given reward distribution.
Sampling proportionally to the reward distribution results in diverse yet high-reward samples.
While there has been an application of GFlowNets for tuning LLMs to induce a latent chain-of-thought  Hu et al. ( 2024 ) ,
there is no established method for using GFlowNets in the context of  offline  alignment of LMs without relying on an explicit reward model.

To this end, we propose  G FlowNet- D irect  P reference  O ptimization (GDPO), providing an efficient offline method for language model alignment.
Similar to DPO, GDPO learns the policy by extracting reward signals directly from the offline preference dataset, but this task is modeled as an inference task
using the GFlowNet.
Empirically, we show that GDPO can generate more diverse responses than the baselines in both dialogue generation and summarization tasks while remaining aligned with the preference dataset.

2  Preliminaries

We define the token-wise Markov Decision Process (MDP) as a tuple

⟨  𝒮  ,  𝒜  ,  f  ,  r  ,

ρ  0

⟩

𝒮  𝒜  𝑓  𝑟

subscript  𝜌  0

\langle\mathcal{S},\mathcal{A},f,r,\rho_{0}\rangle

, where the state space

𝒮

𝒮

\mathcal{S}

consists of tokens generated so far, action space

𝒜

𝒜

\mathcal{A}

is the vocabulary of tokens, transition

f

𝑓

f

is the string concatenation, and the initial distribution

ρ  0

subscript  𝜌  0

\rho_{0}

is the distribution over the prompt

𝒙

𝒙

\bm{x}

.
The episode ends when the model generates the end-of-sequence (EOS) token (denoted

⊤

top

\top

), from which no future reward is given.
The resulting trajectory after iterative sampling from the policy is the response

𝒚  =

𝒚  n

:=

y

1  :  n

⊤

𝒚

subscript  𝒚  𝑛

assign

limit-from

subscript  𝑦

:  1  𝑛

top

\bm{y}=\bm{y}_{n}:=y_{1:n}\top

.
For notational simplicity, we shall denote the initial state

s  0

:=  𝒙

assign

subscript  𝑠  0

𝒙

s_{0}:=\bm{x}

and the terminal state

s  f

:=

𝒙  ;  𝒚

assign

subscript  𝑠  𝑓

𝒙  𝒚

s_{f}:=\bm{x};\bm{y}

.

2.1  Generative Flow Networks (GFlowNets)

GFlowNets offer a way to sample a compositional object from a high-dimensional distribution by taking a sequence of actions according to a learned policy, where the unnormalized probability distribution of the resulting objects converges to the reward distribution  Bengio et al. ( 2023 ) .
This positions GFlowNet at the intersection of Markov Chain Monte-Carlo (MCMC) methods and neural network-based generative models.

The policy interacts with an MDP, represented as a directed acyclic graph (DAG) augmented with some nonnegative function

F

𝐹

F

called  flow .
The state with no parent is the initial state

s  0

subscript  𝑠  0

s_{0}

, and there is exactly one such state in the network.
The states with no children are terminal states referred to as

s  f

subscript  𝑠  𝑓

s_{f}

, which result in the objects of interest.
The reward is defined on terminal states, i.e.

r  :

𝒴  →

ℝ

≥  0

:  𝑟

→  𝒴

subscript  ℝ

absent  0

r:\mathcal{Y}\to\mathbb{R}_{\geq 0}

.

The flow is defined on a complete trajectory,

τ  :=

(

s  0

→

s  1

→  …  →

s  n

)

∈  𝒯

assign  𝜏

→

subscript  𝑠  0

subscript  𝑠  1

→

…

→

subscript  𝑠  𝑛

𝒯

\tau:=(s_{0}\to s_{1}\to\dots\to s_{n})\in\mathcal{T}

, as

F  :

𝒯  →

ℝ

≥  0

:  𝐹

→  𝒯

subscript  ℝ

absent  0

F:\mathcal{T}\to\mathbb{R}_{\geq 0}

.
The state flow for any state

F  ​

(  s  )

=

∑

τ  :

s  ∈  τ

F  ​

(  τ  )

𝐹  𝑠

subscript

:  𝜏

𝑠  𝜏

𝐹  𝜏

F(s)=\sum_{\tau:s\in\tau}F(\tau)

is the total flow through a state and the edge flow

F  ​

(

s  →

s  ′

)

=

∑

τ  :

(

s  →

s  ′

)

∈  τ

F  ​

(  τ  )

𝐹

→  𝑠

superscript  𝑠  ′

subscript

:  𝜏

→  𝑠

superscript  𝑠  ′

𝜏

𝐹  𝜏

F(s\to s^{\prime})=\sum_{\tau:(s\to s^{\prime})\in\tau}F(\tau)

is the total flow through an edge.
Note that every complete trajectory contains the initial state, hence one can define a total flow

Z  :=

F  ​

(  𝒯  )

=

F  ​

(

s  0

)

assign  𝑍

𝐹  𝒯

𝐹

subscript  𝑠  0

Z:=F(\mathcal{T})=F(s_{0})

which normalizes the flow to induce a probability measure on

𝒢

𝒢

\mathcal{G}

.

From here, a flow is defined to be Markovian if there is a distribution

π

(  ⋅  ∣  s  )

\pi(\cdot\mid s)

over the children of a non-terminal state,

Ch  ​

(  s  )

Ch

𝑠

\text{Ch}(s)

where

s  ≠

s  f

𝑠

subscript  𝑠  𝑓

s\neq s_{f}

suc