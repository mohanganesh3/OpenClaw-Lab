[2504.06020] Information-Theoretic Reward Decomposition for Generalizable RLHF

Information-Theoretic Reward Decomposition for Generalizable RLHF

Liyuan Mao

Haoran Xu

Amy Zhang

Weinan Zhang

Chenjia Bai

Abstract

A generalizable reward model is crucial in Reinforcement Learning from Human Feedback (RLHF) as it enables correctly evaluating unseen prompt-response pairs.
However, existing reward models lack this ability, as they are typically trained by increasing the reward gap between chosen and rejected responses, while overlooking the prompts that the responses are conditioned on. Consequently, when the trained reward model is evaluated on prompt-response pairs that lie outside the data distribution, neglecting the effect of prompts may result in poor generalization of the reward model. To address this issue, we decompose the reward value into two independent components: prompt-free reward and prompt-related reward. Prompt-free reward represents the evaluation that is determined only by responses, while the prompt-related reward reflects the reward that derives from both the prompt and the response.
We extract these two components from an information-theoretic
perspective, which requires no extra models.
Subsequently, we propose a new reward learning algorithm by prioritizing data samples based on their prompt-free reward values.
Through toy examples, we demonstrate that the extracted prompt-free and prompt-related rewards effectively characterize two parts of the reward model. Further, standard evaluations show that our method improves both the alignment performance and the generalization capability of the reward model.

Machine Learning, ICML

Figure 1:

Left : reward gaps calculated with corresponding prompt and randomly sampled prompts using QRM-Llama3-8B

1

1  footnotemark:

1

0

0  footnotetext:  xx

on two different datasets that were used for training. When calculating with other prompts, the curves show the mean and the std.  Right : illustrative failure case where the reward gap overly depends on the responses. Solid lines represent corresponding prompt-response pairs (used in training), while dashed lines represent non-corresponding pairs (unseen during training). Since the reward gap overly depends on the responses, it generalizes poorly to novel prompt-response pairs constructed even with seen but non-corresponding prompts.

1  Introduction

Reinforcement Learning from Human Feedback (RLHF) is an effective approach for Large Language Models (LLMs) alignment  (Christiano et al.,  2017 ; Bai et al.,  2022 ) . Within a wide range of RLHF methods, reward learning plays a pivotal role. These methods typically first train a reward model on a static dataset and then leverage it to do Reinforcement Learning (RL)  (Ouyang et al.,  2022 ; Dong et al.,  2024 ) . Compared with methods that are free of using reward models  (Rafailov et al.,  2024b ; Zhao et al.,  2023 ) , the advantage of such methods is their capacity to leverage the generalization capability of the reward model to evaluate outside-of-distribution prompt-response pairs. These prompt-response pairs with generated rewards can be used to further improve the LLM’s performance  (Stiennon et al.,  2020 ; Ziegler et al.,  2019 ) .

Clearly, learning a generalizable reward model is central to this scenario. However, we found that standard reward training does not guarantee sufficient generalization capability. In reward model training, the primary goal is typically better distinguishing between chosen and rejected responses. To achieve this, the reward model does not necessarily require consideration of the corresponding prompt. Taking reward learning based on Bradley-Terry (BT) model as an example.
Since the potential response space is vastly larger than the dataset size, different data samples typically contain distinct response pairs. As long as the reward gap within each pair increases, the training loss will decrease effectively.
This can occur even if the reward model only takes the responses as the input. For the reward model that separately calculates rewards for chosen and rejected prompt-response pairs, the situation becomes similar as the inputs to the reward model only differ starting from the response.

Perhaps surprisingly, such a phenomenon indeed appears in current reward models, even those that achieve SOTA performance on benchmarks.
As shown in Fig.

1

(left), after replacing the corresponding prompt with other prompts, the reward gaps still center around their original values.
This issue, where responses dominate the reward gap, does not affect training but leads to catastrophic results when evaluating novel prompt-response pairs. The illustrative example in Fig.

1

(right) shows this. When considering each prompt-response pair separately within the training dataset, the reward model appears to perform perfectly.
However, when querying preferences after replacing the original prompt with other even in-distribution prompts (which are also meaningful queries), the reward model can yield significantly inaccurate or even reversed preferences. This generalization issue will become more pronounced when dealing with unseen prompt-response pairs encountered during evaluation.
All of these highlight the need to distinguish two components of the reward value: one part is the value determined solely by the response, and the other is the value that can only be derived by simultaneously considering both the prompt and the response. We refer to the former as  prompt-free  reward and the latter as  prompt-related  reward.

To address this, we propose a novel method of decomposition to extract these two components from an information-theoretic perspective, without requiring extra models.
After that, we use the extracted prompt-free reward to guide the reward learning process, prioritizing training samples based on their prompt-free reward gap values.
We verify our method through several toy examples and standard evaluations based on commonly used datasets and base models.
In toy examples, the extracted prompt-free and prompt-related reward gaps characterize the reward model’s preference bias about response-only features and the generalizable preference information, respectively.
Moreover, in standard experiments with common datasets, the reward model trained with our method outperforms strong baselines. These experiments show that considering both prompt-free and prompt-related rewards during training enhances the alignment performance and generalization capabilities of the reward model.

†

†  footnotetext:

1 https://huggingface.co/nicolinho/QRM-Llama3-8B.

2  Preliminaries

Standard preference learning assumes the existence of the preference oracle which determines

ℙ  ​

(

y  w

≻

y  l

|  x

)

\mathbb{P}(y_{w}\succ y_{l}|x)

, where the probability of response

y  w

y_{w}

is more preferred than

y  l

y_{l}

conditioned on the prompt

x  x

. Given a preference dataset

D  =

{

(  x  ,

y  w

,

y  l

)

i

}

i  =  1

N

D=\{(x,y_{w},y_{l})_{i}\}_{i=1}^{N}

where the prompt-response tuples

(  x  ,

y  w

,

y  l

)

(x,y_{w},y_{l})

are generated following

ℙ  ​

(

y  w

≻

y  l

|  x

)

\mathbb{P}(y_{w}\succ y_{l}|x)

, our objective is to estimate the preference oracle from it.

There exist several methods to model the preference oracle  (Luce,  1959 ) , among which the Bradley-Terry (BT) model  (Bradley &amp; Terry,  1952 )  is the most widely used. The BT model further assumes that the preference oracle can be represented as a reward model

r  :

(  x  ,  y  )

→  ℝ

r:(x,y)\to\mathbb{R}

that satisfies:

ℙ  r

​

(

y  w

≻

y  l

|  x

)

=

exp  ⁡

(

r  ​

(  x  ,

y  w

)

)

exp  ⁡

(

r  ​

(  x  ,

y  w

)

)

+

exp  ⁡

(

r  ​

(  x  ,

y  l

)

)

.

\mathbb{P}_{r}(y_{w}\succ y_{l}|x)=\frac{\exp(r(x,y_{w}))}{\exp(r(x,y_{w}))+\exp(r(x,y_{l}))}.

(1)

Under this assumption, standard methods leverage a parameterized reward model

r  θ

r_{\theta}