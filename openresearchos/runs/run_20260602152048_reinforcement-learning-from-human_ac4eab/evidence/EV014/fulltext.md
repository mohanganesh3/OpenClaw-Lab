[2505.19653] Token-Importance Guided Direct Preference Optimization

Token-Importance Guided Direct Preference Optimization

Ning Yang 1

Hai Lin 1

Yibo Liu 1

Baoliang Tian 2

Guoqing Liu 3

Haijun Zhang 4

1 Institute of Automation, Chinese Academy of Sciences

2 ByteDance

3 Microsoft Research AI4Science

4 University of Science and Technology Beijing

Correspondence to: Ning Yang &lt;ning.yang@ia.ac.cn&gt;

Abstract

Ensuring that large language models (LLMs) generate outputs aligned with human preferences is important for safe and effective AI interactions. While Direct Preference Optimization (DPO) employs an implicit reward function to optimize the policy model, however, it and its related variants overlook the differential importance of individual tokens and are sensitive to judgment noise in preference datasets during generation. Although recent methods attempt to assess the important weight of tokens via probability prediction or simplistic weighting schemes, these evaluation methods are prone to biases and still cannot fully address these issues. To solve this problem, we propose the Token-Importance Guided Direct Preference Optimization (TI-DPO), which introduces two key innovations: the gradient-based token-importance weights that dynamically prioritize critical tokens, and a triple loss that explicitly guides model outputs to approach human-preferred responses and stay away from non-preferred responses.
Experimental results show that TI-DPO achieves higher accuracy and stronger generative diversity, providing more stable and computationally efficient solutions compared with DPO and other RLHF methods.

1  Introduction

LLM has shown proficiency in Natural Language Processing (NLP)  [  1  ,

2  ] , logical reasoning  [  3  ] , and code generation  [  4  ] , emerging as a focal point of recent research. However, as models may generate outputs inconsistent with intended purposes or ethical standards, human preference alignment aims to ensure that LLMs adhere to human values  [  5  ,

6  ] , producing beneficial and harmless content. Against this backdrop, Reinforcement Learning from Human Feedback (RLHF) has become a prevailing approach for achieving alignment  [  7  ,

8  ,

9  ] . It leverages human-annotated preference data to train reward models and fine-tunes LLMs using Reinforcement Learning (RL) methods  [  10  ,

11  ]  like Proximal Policy Optimization (PPO)  [  12  ] . While effective, RLHF suffers from inherent limitations: its reliance on iterative policy optimization and complex reward modeling leads to training instability and exorbitant computational costs  [  13  ] .

Addressing these constraints, Direct Preference Optimization (DPO)  [  14  ]  eliminates the necessity for a reward model. Instead, it employs direct pairwise preference comparisons to optimise model policies. Inspired by DPO’s implicit reward mechanism, a series of preference optimization models have been proposed in recent years, such as ORPO  [  7  ] , f-DPO  [  15  ]  and CPO  [  16  ] .
However, DPO’s loss function does not account for the varying importance of individual tokens during the generation of preferred responses  [  17  ] , further intensifying the problem of sampling distribution change. Consider a model generating medical advice: a single misweighted token, for example, "safe" versus "risk", could drastically alter the response’s safety. Traditional DPO treats all tokens equally, potentially overlooking such critical distinctions.

Motivated by these challenges, researchers have proposed token-level variants of DPO such as TDPO  [  17  ] , Logic-RL  [  3  ] , RTO  [  18  ]  and SimPO  [  19  ] , aiming to decompose preference alignment into fine-grained token contributions. Based on these methods, some works have additionally considered the impact of the token importance weights, but they often rely on probabilistic proxies  [  20  ]  or simplistic weighting schemes  [  21  ] , which are prone to bias and fail to fully leverage the structural relationships between semantically similar responses. By contrast, we propose Token-Importance Guided Direct Preference Optimization (TI-DPO), using gradient attribution to dynamically compute token-importance weights. Token importance weights act as a ‘spotlight’, focusing optimization on tokens that most influence human judgments.

Furthermore, recent studies  [  20  ,

19  ]  have noticed some shortcomings in DPO’s design: it compares merely "good" versus "bad" samples. Concurrently, LLMs generate preference data through various methods  [  22  ] , such as rejection sampling  [  23  ]  and employing different prompts for distinct preference levels  [  22  ] , which spurred us to define different levels of samples. Under such a background, our TI-DPO method constructs a "good-bad" triplet relationship by incorporating the intermediate generated outputs  [  24  ] . This triplet structure explicitly guides the intermediate output to approach human preferences and distance from non-preferred responses, achieving fine-grained preference alignment and promoting a continuous gradient of preference learning, rather than relying solely on binary comparisons.

The following contributions are made in the course of this work:

•

Based on DPO, we propose TI-DPO, a token-level approach with important weights. Integrating triplet loss with gradient-based token importance weighting, TI-DPO focuses optimization on critical tokens, mitigating distribution shift and improving alignment precision.

•

Theoretically, we prove TI-DPO achieves a tighter loss bound than DPO (Theorem

4.2  ), ensuring more stable optimization. This theorem formally provides a new perspective on comprehending the superiority of TI-DPO in terms of alignment accuracy.

•

Experiment results indicate that TI-DPO surpasses existing methods in aligning LLMs with human preferences. In preference comparisons, TI-DPO achieves the highest accuracy on TruthfulQA and IFEval tasks, and we compare TI-DPO with three base instruction models across multiple task dimensions with radar graphs. Additionally, ablation experiments confirm the necessity of its token-importance guidance and triplet loss components.

2  Related Work

Human Preference Alignment

Human preference alignment has emerged as a critical research paradigm in recent years, focusing on enabling model responses to align with human values and preferences. Early advancements mainly focused on RLHF  [  26  ,

27  ]  based on PPO  [  12  ] . Due to PPO consuming substantial training resources, some recently proposed RL methods aim to reduce overhead by decreasing parameters of PPO  [  13  ]  or abandoning the critic model  [  28  ] . However, these RL methods may suffer from overfitting in optimal responses.
To mitigate this issue, Hu  et al.

[  8  ]  introduced the Reinforce++ model, which employs batch-wise standardized rewards to prevent overfitting and enhance the prompt diversity during training. Concurrently, beyond RL approaches, Rafailov  et al.  introduced DPO  [  14  ] , which obviates the need for explicit reward modeling through implicit preference learning. This implicit reward mechanism has inspired a wave of subsequent works  [  29  ] , such as the SimPO algorithm proposed by Meng  et al.

[  19  ] , which utilizes the sequence-averaged log probability as an implicit reward signal to streamline optimization.
Notwithstanding these advancements, DPO’s reliance on large-scale human-annotated preference datasets  [  30  ]  has motivated derivative studies  [  31  ,

9  ]  aimed at reducing data requirements. A notable example is RS-DPO  [  32  ] , which integrates rejection sampling (RS) with DPO to alleviate data scarcity. However, a more fundamental limitation pertains to the binary nature of traditional preference labels. Although the KTO method proposed by Ethayarajh  et al.

[  33  ]  effectively reduces the reliance on paired preference labels in DPO, most current RLHF and r