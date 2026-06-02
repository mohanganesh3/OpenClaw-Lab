[2408.02861] A Framework for Fine-Tuning LLMs using Heterogeneous Feedback

A Framework for Fine-Tuning LLMs using Heterogeneous Feedback

Ryan Aponte  1  ,
 Ryan A. Rossi  2  ,
 Shunan Guo 2  ,
 Franck Dernoncourt 2  ,

Tong Yu 2  ,
 Xiang Chen 2  ,
 Subrata Mitra 2  ,
 Nedim Lipka  2

1 Carnegie Mellon University,
 2 Adobe Research

Abstract

Large language models (LLMs) have been applied to a wide range of tasks, including text summarization, web navigation, and chatbots. They have benefitted from supervised fine-tuning (SFT) and reinforcement learning from human feedback (RLHF) following an unsupervised pretraining. These datasets can be difficult to collect, limited in scope, and vary in sample quality. Additionally, datasets can vary extensively in supervision format, from numerical to binary as well as multi-dimensional with many different values. We present a framework for fine-tuning LLMs using heterogeneous feedback, which has two main components. First, we combine the heterogeneous feedback data into a single supervision format, compatible with methods like SFT and RLHF. Next, given this unified feedback dataset, we extract a high-quality and diverse subset to obtain performance increases potentially exceeding the full dataset. We conduct extensive experiments to understand the effectiveness of these techniques for incorporating heterogeneous feedback, and demonstrate improvements from using a high-quality and diverse subset of the data. We find that our framework is able to improve models in multiple areas simultaneously, such as in instruction following and bias reduction.

A Framework for Fine-Tuning LLMs using Heterogeneous Feedback

Ryan Aponte  1 ,
Ryan A. Rossi  2 ,
Shunan Guo 2 ,
Franck Dernoncourt 2 ,

Tong Yu 2  ,
 Xiang Chen 2  ,
 Subrata Mitra 2  ,
 Nedim Lipka  2

1 Carnegie Mellon University,
 2 Adobe Research

1  Introduction

LLMs are fine-tuned for a variety of purposes, such as for instruction following in InstructGPT  Ouyang et al. ( 2022 ) . The fine-tuning process generally begins with collecting examples of desired model behavior and performing supervised learning. Some models stop at SFT  Chiang et al. ( 2023 ) , while InstructGPT follows this by training a reward model based on binary human preference data. The fine-tuned model is then further refined using RLHF, using a signal from the reward model. In the example of InstructGPT, the algorithm used is Proximal Policy Optimization (PPO)  Schulman et al. ( 2017 ) . For each of these steps, the fine-tuning dataset uses a single form of supervision.
Fine-tuning datasets exist for a variety of purposes, from training chat-based assistants in OASST  Köpf et al. ( 2023 ) , coreference resolution in WinoGrande  Sakaguchi et al. ( 2019 ) , helpfulness, honesty, and harmlessness in Anthropic HHH  Nakano et al. ( 2021 ) , and logical reasoning in OpenPlatypus  Lee et al. ( 2024 ) . Supervision format varies, from binary preference in Anthropic HHH, to several numerical labels OASST, to a string response in OpenPlatypus. Although fine-tuning has been successful in mitigating the limitations of pretrained LLMs, these methods require data of a single supervision type, restricting the scope of preference data.
Recent work has filtered fine-tuning datasets to reduce cost and increase quality  Wang et al. ( 2024 ) .

Wu et al. ( 2023 )  use LLMs to generate embeddings for fine-tuning data which is clustered with k-center-greedy  Sener and Savarese ( 2018 )  using an iterative process.

Kung et al. ( 2023 )  randomly delete words in prompts and measure how the response probability changes as a measure of the model’s uncertainty.

Li et al. ( 2024 )  outperform Alpaca as evaluated by LLM preference using only 5% of its fine-tuning data.

We present a framework to use multiple fine-tuning data types, permitting the use of more fine-tuning datasets and fine-tuning for multiple tasks simultaneously. Using multiple datasets enables fine-tuning for different goals simultaneously, such as for logical reasoning and to reduce bias, and provides a more accurate view of human preference by broadening the scope of fine-tuning data. Our framework selects a high-quality and diverse subset of the data to make fine-tuning more effective.

Figure 1:  Framework. First, we concatenate the datasets into a dataset of heterogeneous feedback. We then score samples based on quality and prompt diversity, remove a fraction of the samples (a hyperparameter), forming the homogeneous dataset

D

t  ​  r  ​  a  ​  i  ​  n

subscript  𝐷

𝑡  𝑟  𝑎  𝑖  𝑛

D_{train}

. Standard fine-tuning methods are then applied to a pre-trained LLM.

2  Framework

The primary contribution of our framework is to be able to use fine-tuning data of heterogeneous supervision. Figure

1

includes a high-level overview. Our framework utilizes the simplest supervision, such as binary preference, and projects all remaining datasets into that format. Because some data may be redundant in the unified dataset, we filter for quality and diversity to generate

D

t  ​  r  ​  a  ​  i  ​  n

subscript  𝐷

𝑡  𝑟  𝑎  𝑖  𝑛

D_{train}

. For simplicity, we use this dataset for both the SFT and RLHF steps of fine-tuning, however this is not a requirement. This generates an LLM fine-tuned with high-quality and diverse data, LLaMA-HD.

2.1  Primary fine-tuning dataset

Given a dataset

𝒟

𝒟

\mathcal{D}

of prompts with two responses using binary preference,

𝒟  =

{

(

P  i

,

A

i  ,  0

,

A

i  ,  1

}

i  =  1

M

\mathcal{D}=\{(P_{i},A_{i,0},A_{i,1}\}_{i=1}^{M}

(1)

where

P

𝑃

P

is the prompt,

A

i  ,  0

subscript  𝐴

𝑖  0

A_{i,0}

and

A

i  ,  1

subscript  𝐴

𝑖  1

A_{i,1}

are answers to the prompt, with

A

i  ,  0

subscript  𝐴

𝑖  0

A_{i,0}

defined as the preferred response to the prompt.
This type of dataset takes the form of binary preference due to two example responses to a single prompt. Examples here do not convey a sense of quality, thus prohibiting ranking.

2.2  Secondary fine-tuning dataset

Given a dataset

𝒟  ∗

superscript  𝒟

\mathcal{D^{*}}

of user-specific prompts and responses (question-answer tuples):

𝒟  ∗

=

{

(

P  i

,

A  i

,

𝐲  i

)

}

i  =  1

N

superscript  𝒟

superscript

subscript

subscript  𝑃  𝑖

subscript  𝐴  𝑖

subscript  𝐲  𝑖

𝑖  1

𝑁

\mathcal{D^{*}}=\{(P_{i},A_{i},{\bf y}_{i})\}_{i=1}^{N}

(2)

where

P  i

subscript  𝑃  𝑖

P_{i}

and

A  i

subscript  𝐴  𝑖

A_{i}

are the

i

𝑖

i

th prompt and response pair,
respectively, and

𝐲  i

∈

ℝ  k

subscript  𝐲  𝑖

superscript  ℝ  𝑘

{\bf y}_{i}\in\mathbb{R}^{k}

is the real-valued vector denoting the score of various labels for that pair. For a dataset of this type to be compatible with our method, it is necessary that there are multiple responses to the same prompt. For example,

(

P  i

,

A

i  ′

,

𝐲

i  ′

)

∈

D  ∗

subscript  𝑃  𝑖

subscript  𝐴

superscript  𝑖  ′

subscript  𝐲

superscript  𝑖  ′

superscript  𝐷

(P_{i},A_{i^{\prime}},{\bf y}_{i^{\prime}})\in D^{*}

(3)

can be the second response to the prompt. This process can be repeated for arbitrarily many datasets. A general method for one axis of supervision is included in Appendix

A.2  .

2.3  Simple Unionization for Feedback

We take

𝒟  ∗

superscript  𝒟

\mathcal{D^{*}}

and create a dictionary with prompt as key and responses as a list of all responses to that prompt. This requires at least two responses for each prompt to be considered. We can conduct quality and diversity filtering on these prompts, and then select the preferred response pairs. Once we have tuples containing a prompt, preferred, and non-preferred response, our data from

D  ∗

superscript  𝐷

D^{*}

are now in the same format as

D

𝐷

D

, so we take the union.

2.4  Quality Selection

We infer example quality based on the numerical labels of responses. For datasets with multiple numerical labels, selection of the label is a hyperparameter l