[2511.09105] Cost-Minimized Label-Flipping Poisoning Attack to LLM Alignment

Cost-Minimized Label-Flipping Poisoning Attack to LLM Alignment

Shigeki Kusaka \equalcontrib  1 ,
Keita Saito \equalcontrib  1 ,
Mikoto Kudo \equalcontrib  1,2 ,
Takumi Tanabe 3 ,
Akifumi Wachi 3 ,
 Youhei Akimoto 1,2,4

Abstract

Large language models (LLMs) are increasingly deployed in real-world systems, making it critical to understand their vulnerabilities. While data poisoning attacks during RLHF/DPO alignment have been studied empirically, their theoretical foundations remain unclear. We investigate the minimum-cost poisoning attack required to steer an LLM’s policy toward an attacker’s target by flipping preference labels during RLHF/DPO, without altering the compared outputs. We formulate this as a convex optimization problem with linear constraints, deriving lower and upper bounds on the minimum attack cost. As a byproduct of this theoretical analysis, we show that any existing label-flipping attack can be post-processed via our proposed method to reduce the number of label flips required while preserving the intended poisoning effect. Empirical results demonstrate that this cost-minimization post-processing can significantly reduce poisoning costs over baselines, particularly when the reward model’s feature dimension is small relative to the dataset size. These findings highlight fundamental vulnerabilities in RLHF/DPO pipelines and provide tools to evaluate their robustness against low-cost poisoning attacks.

Code  —  https://github.com/akimotolab/PoisoningCostMinimization

Introduction

Vulnerability of LLM

As large language models (LLMs) are increasingly deployed in real-world applications,
understanding their vulnerabilities is essential for ensuring their effective and safe use.
Adversarial attacks expose these vulnerabilities, supporting red-teaming efforts  ( shayegani2023surveyvulnerabilitieslargelanguage ) .
Representative adversarial attacks at inference time include jail-breaking  ( wei2023neurips ;  chao2023jailbreaking ;  Mehrotra2024neurips ;  zou2023universal )  and prompt injection  ( greshake2023aisec ;  Liu2024usenix ) , where attackers craft malicious inputs to elicit unintended outputs from LLMs.
At training time,  data poisoning attacks  modify the training dataset to induce undesired behaviors or embed backdoor triggers in the resulting LLM. In this paper, we focus on the data poisoning attack on LLMs.

Poisoning attacks on preference alignment

LLMs are susceptible to data poisoning attacks due to their multi-stage training pipeline, which typically includes pre-training, supervised fine-tuning (SFT), and alignment via reinforcement learning from human feedback (RLHF)  ( rlhf )  or direct preference optimization (DPO)  ( dpo ) .
Recent empirical studies have demonstrated that LLMs can be compromised through poisoning during both the SFT phase  ( wan2023poisoning ;  Shu2023neurips )  and the RLHF/DPO phase  ( wu2024preference ;  rlhfpoison ;  pathmanathan2024poisoning ;  baumgartner2024best ;  pathmanathan2024advbdgen ;  tramer2024universal ) , raising concerns about their robustness to adversarial manipulation.
However, the theoretical foundations of poisoning attacks in the RLHF/DPO phase remain largely unexplored, leaving open questions about the fundamental vulnerabilities of these methods and potential defenses.
A theoretical understanding is crucial to ascertain the worst-case scenarios for victims, which empirical studies cannot fully reveal.

Objective

We theoretically investigate the minimum cost of the attack to successfully steer the optimal LLM policy toward the attacker’s target policy during the RLHF/DPO phase.
We consider that an attacker who operates as an annotator, tasked with evaluating two outputs

y  y

and

z  z

in a given context

x  x

, and providing a binary preference label (

w  =  1

w=1

if

y  y

is preferred; otherwise

w  =

−  1

w=-1

).
While the attacker can not modify

(  x  ,  y  ,  z  )

(x,y,z)

, they can arbitrarily set the preference label

w  w

.
Our goal is to determine the minimum number of label flips from the benign labels required to induce the attacker’s desired behavior and to design a method for constructing a malicious dataset that achieves this objective with minimal cost.
By quantifying these costs, our analysis is expected to guide the design of robust RLHF/DPO pipelines that can detect or mitigate such low-cost poisoning attacks, ensuring the safe deployment of LLMs in real-world settings.

Contributions

In this work, we provide the first theoretical analysis of the minimal cost required to steer LLM policies via label-flipping attacks during RLHF/DPO alignment. By formulating the problem as a convex (or linear) optimization, we derive tight lower and upper bounds on the minimum number of label flips needed to induce a target policy. As a byproduct of this analysis, we develop a post-processing method that can be applied to any existing label-flipping attack to reduce its cost while preserving its intended poisoning effect. This approach is particularly effective in practical LLM alignment pipelines, where the dataset size is significantly greater than the feature dimension of the reward model, enabling attackers to exploit redundancy in the data in the feature space to minimize the cost of targeted poisoning.

Preliminaries

LLM alignment is often conducted in two stages: supervised fine-tuning (SFT) and learning from human feedback.
Given an LLM pre-trained with a large corpus in an unsupervised manner, it is fine-tuned using human-annotated input-output pairs

(  x  ,  y  )

(x,y)

to produce more relevant output for specific downstream tasks of interest.
Learning from human feedback then aims to further align LLMs with human preferences.
In this step, the LLM is trained using a dataset of preferences,

𝒟  L

=

{

(  x  ,  y  ,  z  ,  w  )

}

\mathcal{D}_{L}=\{(x,y,z,w)\}

, where

x  x

is the input,

y  y

and

z  z

are two candidate outputs, and

w  ∈

{

−  1

,  1  }

w\in\{-1,1\}

is the preference label indicating whether

y  y

is preferred to

z  z

(

w  =  1

w=1

) or not (

w  =

−  1

w=-1

). The LLM is trained to assign higher probabilities to preferred outputs.
Reinforcement learning from human feedback (RLHF) is often employed for this purpose.

RLHF first trains a reward model

r  ​

(  x  ,  y  )

r(x,y)

from the preference dataset.
The human preference is typically modeled as the Bradley-Terry model:

Pr  ⁡

[

w  =

1  ∣

x  ,  y  ,  z

]

=

σ  ​

(

r  ​

(  x  ,  y  )

−

r  ​

(  x  ,  z  )

)

,

\Pr[w=1\mid x,y,z]=\sigma(r(x,y)-r(x,z)),

(1)

where

σ  ​

(  t  )

=

1

1  +

exp  ⁡

(

−  t

)

\sigma(t)=\frac{1}{1+\exp(-t)}

is the sigmoid function.
The reward model is trained via maximum likelihood estimation by minimizing

ℒ  ​

(  r  )

=

−

∑

(  x  ,  y  ,  z  ,  w  )

∈

𝒟  L

log  ⁡  σ

​

(

w  ​

(

r  ​

(  x  ,  y  )

−

r  ​

(  x  ,  z  )

)

)

.

\displaystyle\mathcal{L}(r)=-\sum_{(x,y,z,w)\in\mathcal{D}_{L}}\log\sigma(w(r(x,y)-r(x,z))).

(2)

Let

r  ^

\widehat{r}

denote the obtained reward model.
The LM policy

π  \pi

is then trained to maximize the obtained reward under the KL-regularization:

𝔼

x  ∼  ρ

​

[

𝔼

y  ∼

π  ​

(

y  ∣  x

)

​

[

r  ^

​

(  x  ,  y  )

]

−

τ  ​

D  KL

​

(

π  ∥

π  ref

)

]

,

\displaystyle\mathbb{E}_{x\sim\rho}[\mathbb{E}_{y\sim\pi(y\mid x)}[\widehat{r}(x,y)]-\tau D_{\mathrm{KL}}(\pi\parallel\pi_{\mathrm{ref}})],

(3)

where

ρ  \rho

is a distribution over the context;

π  ref

\pi_{\mathrm{ref}}

is the reference policy, typically the SFT policy used to initialize RLHF; and

τ  \tau

is a parameter controlling the deviation from

π  ref

\pi_{\mathrm{ref}}

.

Direct Preference Optimization (DPO) is an alternative to RLHF that directly optimizes the LM policy from the preference dataset.
The optimal policy that maximizes (  3  ) is:

π  r

​

(

y  ∣  x

)

=

1

Z

r