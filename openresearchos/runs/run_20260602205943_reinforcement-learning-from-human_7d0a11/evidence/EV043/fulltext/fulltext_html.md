[2502.05773] PIPA: Preference Alignment as Prior-Informed Statistical Estimation

PIPA: Preference Alignment as Prior-Informed Statistical Estimation

Junbo Li

Zhangyang Wang

Qiang Liu

Abstract

Offline preference alignment for language models such as Direct Preference Optimization (DPO) is favored for its effectiveness and simplicity, eliminating the need for costly reinforcement learning. Various offline algorithms have been developed for different data settings, yet they lack a unified understanding.
In this study, we introduce Pior-Informed Preference Alignment (PIPA), a unified, RL-free probabilistic framework that formulates language model preference alignment as a Maximum Likelihood Estimation (MLE) problem with prior constraints. This method effectively accommodates both paired and unpaired data, as well as answer and step-level annotations. We illustrate that DPO and KTO are special cases with different prior constraints within our framework. By integrating different types of prior information, we developed two variations of PIPA: PIPA-M and PIPA-N. Both algorithms demonstrate a

3  ∼

10  %

3\sim 10\%

performance enhancement on the GSM8K and MATH benchmarks across all configurations, achieving these gains without additional training or computational costs compared to existing algorithms.

1  Introduction

Pre-training large language models (LLMs) from scratch on trillions of text tokens allows for accurate prediction next tokens in natural language

(  Achiam et al.  ,

2023  ;  Dubey et al.  ,

2024  ;  Liu et al.  ,

2024a

)

. Following this, alignment, achieved through fine-tuning on smaller, high-quality datasets designed for specific tasks, becomes critical for enabling the model to develop specialized skills, such as engaging in conversation

(  Ouyang et al.  ,

2022

)

, math reasoning

(  Shao et al.  ,

2024  ;  Yang et al.  ,

2024

)

, coding

(  Zhu et al.  ,

2024

)

, web agent

(  Qin et al.  ,

2025

)

, and more.
The fundamental approach to alignment involves supervised fine-tuning (SFT) on the target domain, which essentially maximizes the likelihood of predicting the next token.
However, numerous empirical studies have shown that simple SFT on preferred samples is inadequate for attaining optimal performance

(  Shao et al.  ,

2024  ;  Ouyang et al.  ,

2022

)

.

Moving beyond basic imitation learning in SFT, it is suggested to learn from both positive and negative samples. Sample quality can be measured by training reward models to capture general preferences

(  Dong et al.  ,

2024

)

or leveraging accurate rule-based rewards

(  Guo et al.  ,

2025

)

for specific tasks like math and coding. By treating the autoregressive generation of LLMs as a Markov decision process (MDP), traditional reinforcement learning (RL) algorithms can be effectively applied, such as PPO

(  Ouyang et al.  ,

2022

)

, SAC

(  Liu et al.  ,

2024b

)

, REINFORCE

(  Ahmadian et al.  ,

2024

)

, etc.

While online RL-based methods deliver strong performance, they face challenges such as high training costs, instability, and the need for a strong base model as the initial policy. As a result, offline algorithms like direct preference optimization (DPO)

(  Rafailov et al.  ,

2024

)

are often preferred, thanks to their effectiveness and simplicity, particularly when high-quality datasets are accessible.
The original DPO algorithm has several limitations. It relies on paired preference data, which is not essential for tasks with ground truth such as math and coding. Additionally, it is unable to accommodate step-level annotations. Furthermore, it treats all tokens equally, lacking token-level credit assignment.
To address these issues, a series of approaches have been developed, such as Kahneman-Tversky Optimization (KTO)

(  Ethayarajh et al.  ,

2024

)

for unpaired data, Step-DPO

(  Lai et al.  ,

2024  ;  Lu et al.  ,

2024

)

and Step-KTO

(  Lin et al.  ,

2025

)

for step-level annotations, and RTO

(  Zhong et al.  ,

2024

)

, TDPO

(  Zeng et al.  ,

2024

)

, and OREO

(  Wang et al.  ,

2024

)

for fine-grained token-level DPO. However, these methods are designed from specific perspectives, each addressing only particular challenges, and they lack a unified understanding to integrate their solutions.

In this work, we introduce a unified framework designed to address all the aforementioned challenges in offline approaches.
Rather than framing the alignment problem within offline RL, we reformulate it as a maximum likelihood estimation (MLE) problem with prior constraints, operating within a purely probabilistic framework called Prior-Informed Preference Alignment (PIPA).
From a statistical estimation perspective, we analyze the suboptimality of supervised fine-tuning (SFT). We demonstrate that both the original DPO and KTO algorithms can be interpreted as special cases within our framework, differing in the prior information they incorporate and the loss used. Building on the PIPA framework, we propose two variants, PIPA-M and PIPA-N, that incorporate prior information in different fashions.
The probabilistic formulation naturally accommodates unpaired data and extends to step-level annotations.
Our PIPA functions as a versatile plug-in loss design that seamlessly integrates with any (iterative) data generation pipeline in existing alignment framework.
Furthermore, we show that PIPA training effectively learns token-level credit assignment, yielding precise per-token value estimations that may enable search during test-time inference.

Our contributions can be summarized as follows:

∙  \bullet

We formulate preference alignment as a prior-informed conditional probability estimation problem that is RL-free and provides clear theoretical insight.

∙  \bullet

Our approach does not need paired preference data, and seamlessly unifies both answer-wise and step-wise settings under a single, theoretically grounded framework.

∙  \bullet

Compared to existing approaches such as DPO

(  Rafailov et al.  ,

2024

)

and KTO

(  Ethayarajh et al.  ,

2024

)

, our algorithm achieves improved performance without additional computational overhead.

1.1  Related Work

Learning from preference data

RL has become a key framework for leveraging preference data for LLM alignment, with early methods like PPO

(  Schulman et al.  ,

2017

)

, which first trains a reward model on pairwise human feedback

(  Ouyang et al.  ,

2022

)

. Due to PPO’s high training cost, direct policy optimization methods without online RL have been explored, integrating policy and reward learning into a single stage. Notable works include DPO

(  Rafailov et al.  ,

2024

)

, SLiC

(  Zhao et al.  ,

2023

)

, IPO

(  Azar et al.  ,

2024

)

, GPO

(  Tang et al.  ,

2024

)

, and SimPO

(  Meng et al.  ,

2024

)

.
For fine-grained token-level optimization, DPO variants like TDPO

(  Zeng et al.  ,

2024

)

, TIS-DPO

(  Liu et al.  ,

2024c

)

, RTO

(  Zhong et al.  ,

2024

)

, and OREO

(  Wang et al.  ,

2024

)

have been introduced. To address step-level annotation inspired by PRM

(  Lightman et al.  ,

2023

)

, methods such as Step-DPO

(  Lai et al.  ,

2024

)

, SCDPO

(  Lu et al.  ,

2024

)

, and SVPO

(  Chen et al.  ,

2024a

)

have emerged. To relax pairwise data constraints, particularly for tasks with ground truth like math and coding, KTO

(  Ethayarajh et al.  ,

2024

)

, Step-KTO

(  Lin et al.  ,

2025

)

, and OREO

(  Wang et al.  ,

2024

)

have been proposed.
Our PIPA framework addresses all these challenges within a unified paradigm, demonstrating that existing algorithms like DPO and KTO can be interpreted as special cases within our approach.

Probabilistic alignment

In addition to reward maximization, some research approaches alignment from a probabilistic perspective.

(  Abdolmaleki et al.  ,

2024

)

decompose label likelihood into a target distribution and a hidden distribution,