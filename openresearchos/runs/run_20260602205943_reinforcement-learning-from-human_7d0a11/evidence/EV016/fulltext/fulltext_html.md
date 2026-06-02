[2409.17472] Autoregressive Multi-trait Essay Scoring via Reinforcement Learning with Scoring-aware Multiple Rewards

Autoregressive Multi-trait Essay Scoring via Reinforcement Learning with Scoring-aware Multiple Rewards

Heejin Do  1  , Sangwon Ryu  1  , Gary Geunbae Lee  1,2

1  Graduate School of Artificial Intelligence, POSTECH, South Korea

2  Department of Computer Science and Engineering, POSTECH, South Korea

{heejindo, ryusangwon, gblee}@postech.ac.kr

Abstract

Recent advances in automated essay scoring (AES) have shifted towards evaluating multiple traits to provide enriched feedback. Like typical AES systems, multi-trait AES employs the quadratic weighted kappa (QWK) to measure agreement with human raters, aligning closely with the rating schema; however, its non-differentiable nature prevents its direct use in neural network training. In this paper, we propose Scoring-aware Multi-reward Reinforcement Learning (SaMRL), which integrates actual evaluation schemes into the training process by designing QWK-based rewards with a mean-squared error penalty for multi-trait AES. Existing reinforcement learning (RL) applications in AES are limited to classification models despite associated performance degradation, as RL requires probability distributions; instead, we adopt an autoregressive score generation framework to leverage token generation probabilities for robust multi-trait score predictions. Empirical analyses demonstrate that SaMRL facilitates model training, notably enhancing scoring of previously inferior prompts.

Autoregressive Multi-trait Essay Scoring via Reinforcement Learning with Scoring-aware Multiple Rewards

Heejin Do  1  , Sangwon Ryu  1  , Gary Geunbae Lee  1,2

1  Graduate School of Artificial Intelligence, POSTECH, South Korea

2  Department of Computer Science and Engineering, POSTECH, South Korea

{heejindo, ryusangwon, gblee}@postech.ac.kr

1  Introduction

An essay can be evaluated from diverse perspectives, such as  Content ,  Sentence Fluency , and  Organization . As providing multi-view assessment is essential for enhancing the learner’s writing skill, recent attention to automated essay scoring (AES) systems has shifted from solely relying on holistic scoring  Taghipour and Ng ( 2016 ); Dong and Zhang ( 2016 ); Dong et al. ( 2017 ); Wang et al. ( 2022 )  to evaluating multiple trait scores  Kumar et al. ( 2022 ); Ridley et al. ( 2021 ); Do et al. ( 2024 ) . Although simultaneous assessment for multiple traits is more challenging than a holistic paradigm, it has been much less explored.

Typically, AES systems are evaluated using the Quadratic Weighted Kappa (QWK)  Cohen ( 1968 )  score, which measures the agreement between human ratings and model predictions. Despite its effectiveness and close alignment with real-world rating schemes, its non-differentiable nature prevents direct use in neural-network training  Wang et al. ( 2018 ) . Instead, previous AES models predominantly utilized cross-entropy or mean squared error (MSE) loss to train classification- or regression-based AES models, respectively (Figure

1  ).

Figure 1:  Overview of distinct AES frameworks. The autoregressive framework eliminates the need for multiple trait-wise layers. Classification and autoregressive AES models probabilistically predict final scores; hence, a policy gradient reinforcement algorithm is applicable.

In this paper, we propose a Scoring-aware Multi-reward Reinforcement Learning (SaMRL) method to unlock the potential of the QWK for training multi-trait AES systems. By constructing multiple rewards of bi-directional QWK and MSE penalty, SaMRL effectively incorporates nuanced measurement schemes in training phrases. Generally, the QWK score derives from a set of essays rather than a single score; thus, when applied conventionally in a batch set, it assigns the same metric to every sample. To ensure stable training, we introduce trait-wise comparison for QWK, integrating them with the batch-wise calculation to construct a bi-directional QWK reward.

Applying RL in AES is underexplored, and prior work  Wang et al. ( 2018 )  is limited to holistic scoring. Further, their method is restricted to the classification approach despite its inferior performance than the regression, as policy gradients for RL require probability distributions. Unlike prior works, we treat AES as a generation paradigm  Do et al. ( 2024 ) , leveraging token generation probability distributions instead of categorical ones for policy gradient. Note that standard autoregressive AES is trained with cross-entropy and does not reflect any score-related metrics (i.e., MSE or QWK) during training; however, SaMRL enables direct parameter updates based on those scoring-aware metrics.

Extensive experiments on the representative ASAP and ASAP++ datasets demonstrate scoring enhancement over robust baselines across the traits and prompts. Comprehensive Analyses, which compare SaMRL with both single-reward applications and unidirectional use of QWK rewards, further reveal the robustness of our method. Notably, significant improvements observed on prompts with a broader score range highlight the overcoming of challenges posed by prior use of RL in AES.

2  Related works

Multi-trait essay scoring

Although automated essay scoring has achieved notable success  Dong et al. ( 2017 ); Yang et al. ( 2020 ); Wang et al. ( 2022 ) , research on multi-trait essay scoring is still underdeveloped and requires further exploration. Early attempts for multi-trait AES lie in constructing multiple trait-specific layers or models for different predictions  Mathias and Bhattacharyya ( 2020 ); Ridley et al. ( 2021 ); Kumar et al. ( 2022 ); Do et al. ( 2023 ) . Pointing out the inefficiency of duplicating individual encoder-only models generating a single score,  Do et al. ( 2024 )  sequentially produces multi-trait scores by defining scoring as a decoder-introduced text generation. Autoregressively generating multi-trait scores significantly improved all trait-scoring performance, achieving stat-of-the-art results. Further, it reduces the burden of designing separate trait-specific layers as it consecutively predicts full trait scores for entire prompts with a single model. To take advantage of the efficiency and high performance, we introduce our RL method based on their autoregressive AES paradigm.

RL for text generation

Recently, RL has been actively employed across diverse natural language generation tasks. Notably, the advent of reinforcement learning from human feedback (RLHF) has improved the capabilities of general-purpose large language models (LLMs) such as GPT, showing the strength of RL  Ouyang et al. ( 2022 ) . Numerous researchers have applied RL to specific downstream tasks such as text summarization  Paulus et al. ( 2018 ); Dong et al. ( 2018 ); Chen and Bansal ( 2018 ); Narayan et al. ( 2018 ); Pasunuru and Bansal ( 2018 ); Gunasekara et al. ( 2021 ); Parnell et al. ( 2022 ); Roit et al. ( 2023 ); Ribeiro et al. ( 2023 ); Su et al. ( 2023 ); Ryu et al. ( 2024 ); Singh et al. ( 2024 ) , machine translation  Wu et al. ( 2018 ); He et al. ( 2024 ) , and reasoning  Havrilla et al. ( 2024 ); Dutta et al. ( 2024 ); Xi et al. ( 2024 ); Lu et al. ( 2024 ) . They aim to enhance performance by using rewards tailored to their specific objectives. For instance, in text summarization,  Stiennon et al. ( 2020 )  employ human feedback as a reward model to generate summaries that align with human preferences, while  Roit et al. ( 2023 )  use the entailment relationship between summary and source document as a reward to generate a factually consistent summary. In arithmetic reasoning problems,  Dutta et al. ( 2024 )  utilizes a non-differentiable symbolic solver as a reward to address multi-step reasoning. We integrate the RL framework to unexplored autoregressive multi-trait AES by introducing novel multiple scoring-aware rewards.

3  Preliminary

We adopt the policy gradient reinfo