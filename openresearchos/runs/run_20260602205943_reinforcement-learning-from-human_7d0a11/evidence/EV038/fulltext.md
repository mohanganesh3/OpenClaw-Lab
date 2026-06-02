[2406.02900] Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms

Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms

Rafael Rafailov

Stanford University

rafailov@cs.stanford.edu

&amp;Yaswanth Chittepu  1

1  footnotemark:

1

UMass Amherst

ychittepu@umass.edu

&amp;Ryan Park  1

1  footnotemark:

1

Stanford University

rypark@stanford.edu

&amp;Harshit Sikchi  1

1  footnotemark:

1

UT Austin

hsikchi@utexas.edu

&amp;Joey Hejna  1

1  footnotemark:

1

Stanford University

jhejna@cs.stanford.edu

&amp;W. Bradley Knox

UT Austin

bradknox@cs.utexas.edu

Chelsea Finn

Stanford University

cbfinn@cs.stanford.edu

&amp;Scott Niekum

UMass Amherst

sniekum@cs.umass.edu

Equal Contribution, Dice Rolling

Abstract

Reinforcement Learning from Human Feedback (RLHF) has been crucial to the recent success of Large Language Models (LLMs), however, it is often a complex and brittle process. In the classical RLHF framework, a reward model is first trained to represent human preferences, which is in turn used by an online reinforcement learning (RL) algorithm to optimize the LLM. A prominent issue with such methods is  reward over-optimization  or  reward hacking , where performance as measured by the learned proxy reward model increases, but true quality plateaus or even deteriorates. Direct Alignment Algorithms (DDAs) like Direct Preference Optimization have emerged as alternatives to the classical RLHF pipeline by circumventing the reward modeling phase. However, although DAAs do not use a separate proxy reward model, they still commonly deteriorate from over-optimization. While the so-called reward hacking phenomenon is not well-defined for DAAs, we still uncover similar trends: at higher KL budgets, DAA algorithms exhibit similar degradation patterns to their classic RLHF counterparts. In particular, we find that DAA methods deteriorate not only across a wide range of KL budgets but also often before even a single epoch of the dataset is completed. Through extensive empirical experimentation, this work formulates and formalizes the reward over-optimization or hacking problem for DAAs and explores its consequences across objectives, training regimes, and model scales.

1  Introduction

Recent advancements in Large Language Models (LLMs) have broadened their capabilities significantly, enabling applications in code generation, mathematical reasoning, tool use, and interactive communication. These improvements have popularized LLMs across various domains. Reinforcement Learning from Human Feedback (RLHF) has been instrumental in these advances and is now integral to sophisticated LLM training regimes  [ 10 ,  55 ] . Before alignment, LLMs, trained on vast text corpses to predict subsequent tokens  [ 45 ,  8 ]  are often unwieldy and hard to use. Today, leading LLMs incorporate variants of the RLHF framework  [ 14 ,  68 ,  36 ]  to align them with human intent, which generally involves a multi-stage process. Specifically, users evaluate model responses to assorted prompts in order to train a reward model that encapsulates human preferences  [ 10 ,  55 ,  71 ,  5 ,  61 ] . Then, the refined LLM maximizes the expected learned reward function using a reinforcement learning (RL) algorithm  [ 50 ,  1 ,  64 ] . Despite its efficacy, this procedure is complex and computationally intensive, particularly in its latter stages.

Goodhart’s Law  [ 25 ,  11 ] , that “when a measure becomes a target, it ceases to be a good measure”, has often been cited as a core shortcoming of RLHF. Standard RLHF methods optimize a learned, but imperfect reward function which ends up amplifying the reward model’s shortcomings. Empirically, this phenomenon was first extensively characterized by  Gao et al. [ 21 ] , who coined the term “reward over-optimization”, and has been seen consistently in recent findings  [ 61 ,  16 ,  14 ] . While reward over-optimization has been studied in the context of the aforementioned RLHF procedure, recent contemporary methods for aligning LLMs circumvent the reward learning procedure, necessitating a new characterization of the over-optimization phenomena.

This new broad class of algorithms, which we refer to as Direct Alignment Algorithms (DAAs), bypass the traditional RLHF pipeline by re-parameterizing the reward model directly through the optimal policy derived during the reinforcement learning phase. DAA methods, like Direct Preference Optimization  [ 46 ] , have gained popularity  [ 14 ,  28 ]  as they often reduce computational demands. Yet, despite not fitting a reward function, DAAs still exhibit over-optimization trends similar to those of traditional RLHF methods using a learned reward function. In some sense, this is puzzling: DAAs can be viewed as simply learning a reward function with supervised learning from which the optimal policy is deterministically mapped, however more seems to be at play than simple supervised learning.

In this work, we investigate the over-fitting phenomena present in DAA algorithms through extensive experimentation. First, we unify a number of different recent methods  [ 46 ,  67 ,  4 ]  under the DAA framework. Then, across different model scales and hyper-parameters, we show that DAAs exhibit a type of reward over-optimization consistent with that previously observed in RLHF  [ 21 ] . Specifically, we find that at different KL-divergence budgets DAAs exhibit degradation patterns similar to those found in RLHF. Interestingly, we also find that performance within a single epoch is not always as consistent as expected for DAAs. Finally, we explain why this happens by appealing to the under-constrained nature of the optimization problem used in DAAs.

2  Preliminaries

In this section, we first outline the core components of the standard RLHF pipeline  [ 71 ,  55 ,  5 ,  41 ] ). Then, we examine prior literature to characterize the reward over-optimization exhibited by standard RLHF methods. Finally, we provide a unifying view of direct alignment algorithms (DAAs) which will guide our analysis of their training dynamics in the next section.

2.1  Reinforcement Learning From Human Feedback

The standard RLHF pipeline consists of three distinct stages with the goal of aligning the LLM with human preferences.

Supervised Fine Tuning (SFT) : First, a dataset of prompts

x

𝑥

x

and high-quality answers

y

𝑦

y

are used to train an LLM for instruction following via maximum likelihood estimation over next-tokens. We refer to the resultant model as

π  SFT

​

(

y  |  x

)

subscript  𝜋

SFT

conditional  𝑦  𝑥

\pi_{\text{SFT}}{(y|x)}

and consider the entire prompt and answer strings to be single variables.

Reward Modeling : Second, the SFT model

π  SFT

​

(

y  |  x

)

subscript  𝜋

SFT

conditional  𝑦  𝑥

\pi_{\text{SFT}}{(y|x)}

is used to learn a reward function over human preferences. Specifically, the SFT model is queried to produce pairs of answers

(

y  1

,

y  2

)

∼

π  SFT

​

(

y  |  x

)

similar-to

subscript  𝑦  1

subscript  𝑦  2

subscript  𝜋

SFT

conditional  𝑦  𝑥

(y_{1},y_{2})\sim\pi_{\text{SFT}}{(y|x)}

, for every prompt

x

𝑥

x

in a dataset. Then, users select their preferred answers, resulting in ranking

y  w

≻

y  l

∣  x

succeeds

subscript  𝑦  𝑤

conditional

subscript  𝑦  𝑙

𝑥

y_{w}\succ y_{l}\mid x

where

y  w

subscript  𝑦  𝑤

y_{w}

and

y  l

subscript  𝑦  𝑙

y_{l}

are the preferred and dispreferred answers respectively. Typically, user rankings are assumed to be distributed according to the Bradley-Terry (BT) model  [ 7 ]

p  ​

(

y  1

≻

y  2

∣  x

)

=

exp  ⁡

(

r  ​

(  x  ,

y  1

)

)

exp  ⁡

(

r  ​

(  x  ,

y  1

)

)

+

exp  ⁡

(

r  ​

(  x  ,

y  2

)

)

=

σ  ​

(

r  ​

(  x  ,

y  1

)

−

r  ​

(  x  ,

y  2

)

)

𝑝

succeeds

subscript  𝑦  1

conditional

subscript  𝑦  2

𝑥

𝑟

𝑥

subscript  𝑦  1

𝑟

𝑥

subscript  𝑦  1

𝑟

𝑥

subscript  𝑦  2

𝜎

𝑟