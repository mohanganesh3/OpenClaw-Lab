[2604.22229] Preserve Support, Not Correspondence: Dynamic Routing for Offline Reinforcement Learning

Preserve Support, Not Correspondence: Dynamic Routing for Offline Reinforcement Learning

Zhancun Mu

Affiliation:  School of Intelligence ScienceTechnology, Peking University {muzhancun,zhaogy24}@stu.pku.edu.cnyiwu-zhong@outlook.com  chizhang.cz@pku.edu.cn

Guangyu Zhao

Affiliation:  School of Intelligence ScienceTechnology, Peking University {muzhancun,zhaogy24}@stu.pku.edu.cnyiwu-zhong@outlook.com  chizhang.cz@pku.edu.cn

Yiwu Zhong

Affiliation:  School of Intelligence ScienceTechnology, Peking University {muzhancun,zhaogy24}@stu.pku.edu.cnyiwu-zhong@outlook.com  chizhang.cz@pku.edu.cn

Chi Zhang

Affiliation:  School of Intelligence ScienceTechnology, Peking University {muzhancun,zhaogy24}@stu.pku.edu.cnyiwu-zhong@outlook.com  chizhang.cz@pku.edu.cn

Affiliation:

Correspondence to: chizhang.cz@pku.edu.cn

Abstract

One-step offline RL actors are attractive because they avoid backpropagating through long iterative samplers and keep inference cheap, but they still have to improve under a critic without drifting away from actions that the dataset can support.
In recent one-step extraction pipelines, a strong iterative teacher provides one target action for each latent draw, and the same student output is asked to do both jobs: move toward higher Q and stay near that paired endpoint.
If those two directions disagree, the loss resolves them as a compromise on that same sample, even when a nearby better action remains locally supported by the data.
We propose DROL, a latent-conditioned one-step actor trained with top-1 dynamic routing.
For each state, the actor samples

K  K

candidate actions from a bounded latent prior, assigns each dataset action to its nearest candidate, and updates only that winner with

behavior cloning

(  BC  ) and critic guidance.
Because the routing is recomputed from the current candidate geometry, ownership of a supported region can shift across candidates over the course of learning.
This gives a one-step actor room to make local improvements that pointwise extraction struggles to capture, while retaining single-pass inference at test time.
On OGBench and D4RL, DROL is competitive with the one-step FQL baseline, improving many OGBench task groups while remaining strong on both AntMaze and Adroit.
Project page:  https://muzhancun.github.io/preprints/DROL .

1  Introduction

Offline

Reinforcement Learning

(  RL  ) requires improving a policy while staying within the action regions supported by a fixed dataset  (Levine et al.,  2020 ) .
This requirement becomes harder to satisfy on multimodal benchmarks such as OGBench  (Park et al.,  2025a ) , where nearby states can admit several distinct yet reasonable actions.
Generative actors are attractive in this regime because they represent such multimodal behavior better than standard unimodal policies  (Sohl-Dickstein et al.,  2015 ; Lipman et al.,  2023 ; Wang et al.,  2023 ; Hansen-Estruch et al.,  2023 ; Park et al.,  2025b ) .

Many strong generative offline

RL

methods rely on iterative sampling during training, inference, or both  (Wang et al.,  2023 ; Hansen-Estruch et al.,  2023 ; Espinosa-Dice et al.,  2025 ; Mu et al.,  2026 ) .
A one-step actor is attractive because it avoids backpropagating through a long iterative sampler and keeps execution cheap.
One common way to obtain such an actor, exemplified by FQL  (Park et al.,  2025b ) , is to distill a stronger iterative teacher: for a latent draw

z  z

, the teacher produces

a  ~

=

μ  ψ

​

(  s  ,  z  )

\tilde{a}=\mu_{\psi}(s,z)

, and the student output

f  θ

​

(  s  ,  z  )

f_{\theta}(s,z)

is trained to improve Q while staying close to

a  ~

\tilde{a}

.
The difficulty is structural: within each update, the same sampled output must serve both roles.
It is pushed toward higher Q while simultaneously being pulled back toward its paired teacher endpoint

a  ~

\tilde{a}

.
When these two directions disagree, the loss resolves the conflict by compromising on that same sample.

For a multimodal actor, different samples can naturally cover different supported modes.
 The key question is whether that ownership can shift during training. 
If it cannot, a sampled output remains responsible for its original endpoint even when a nearby supported improvement is available.
If it can, that sample may move toward the better action while another takes over the old region.
In FQL-style extraction, ownership stays attached to the same sampled output.
In DROL, it can transfer across candidates.
This tension is illustrated in

Figure ˜ 1  .

Figure 1:

Preserve support, not correspondence.
Left: pointwise extraction assigns both improvement and reconstruction to the same sampled output.
Right: DROL routes each dataset action to the nearest candidate, so ownership of a supported region can transfer across candidates.

DROL retains the cheap one-step actor but changes the regularizer.
For each state, it samples

K  K

candidate actions, routes each dataset action to its nearest candidate,

k  ∗

​

(  s  ,  a  )

=

arg  ⁡

min

k  ∈

[  K  ]

⁡

‖

a  ^

k

−  a

‖

2  2

,

k^{*}(s,a)=\arg\min_{k\in[K]}\|\hat{a}^{k}-a\|_{2}^{2},

and updates only that winner with behavior cloning and Q-improvement.
As shown in

Figure ˜ 1  , one candidate can move toward a better local action while another retains ownership of the old supported region.
The key difference from FQL-style extraction is that the behavior-cloning pull can change owners across candidates rather than remaining attached to a single sampled output.
This design allows different latent regions to specialize to different parts of local support while preserving one-step inference at test time.

Our main contributions are:

•

We identify pointwise latent-to-teacher correspondence as an unnecessary constraint in one-step extraction for multimodal offline

RL  , and argue that local action support is the object worth preserving.

•

We propose DROL, a routed candidate-set actor that samples multiple actions during training, assigns each data action to its nearest candidate, and updates only the routed winner.

•

We analyze the mechanism of routing, including non-collapse, responsibility transfer, and the role of the routing budget

K  K

, and evaluate the resulting one-step actor on OGBench and D4RL  (Fu et al.,  2020 )  with routing visualizations and

K  K

ablations.

2  Preliminaries and Problem Setup

We consider an

Markov Decision Process

(  MDP  )

(  𝒮  ,  𝒜  ,  r  ,  p  ,  γ  )

(\mathcal{S},\mathcal{A},r,p,\gamma)

together with a fixed offline dataset

𝒟  =

{

(  s  ,  a  ,  r  ,

s  ′

)

}

\mathcal{D}=\{(s,a,r,s^{\prime})\}

.
Given a critic

Q  ϕ

Q_{\phi}

, the actor should seek high-value actions while staying close to the state-conditional action regions supported by the data.
At an abstract level, offline

RL

can be written as

max

π  θ

\displaystyle\max_{\pi_{\theta}}

𝔼

s  ∼  𝒟  ,  a  ∼

π  θ

(  ⋅  |  s  )

​

[

Q  ϕ

​

(  s  ,  a  )

]

\displaystyle\mathbb{E}_{s\sim\mathcal{D},\,a\sim\pi_{\theta}(\cdot|s)}[Q_{\phi}(s,a)]

(1)

s.t.

𝔻

(

π  θ

(  ⋅  |  s  )

,

π  β

(  ⋅  |  s  )

)

≤  ϵ  ,

\displaystyle\mathbb{D}\bigl(\pi_{\theta}(\cdot|s),\pi_{\beta}(\cdot|s)\bigr)\leq\epsilon,

where

π  β

\pi_{\beta}

denotes the implicit behavior distribution and

𝔻  \mathbb{D}

is a discrepancy that keeps policy improvement within trustworthy regions.
In the multimodal setting studied here, exact duplicates at a single observation are rare, so we instead search within a small neighborhood of states around

s  s

rather than relying on literal repeated supervision at the exact same state.
For this paper, the central question is which object the regularizer is actually trying to preserve.

We train the critic with the usual Bellman regression objective

ℒ  Q

​

(  ϕ  )

=

𝔼

(  s  ,  a  ,  r  ,

s  ′

)

∼  𝒟

​

[

(

Q  ϕ

​