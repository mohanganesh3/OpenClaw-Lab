[2402.01694] ARGS: Alignment as Reward-Guided Search

\useunder

\ul

ARGS: Alignment as Reward-Guided Search

Maxim Khanov 1* , Jirayu Burapacheep 2* , Yixuan Li 1

University of Wisconsin-Madison 1

Stanford University 2

mkhanov@wisc.edu ,  jirayu@stanford.edu ,  sharonli@cs.wisc.edu

Abstract

Aligning large language models with human objectives is paramount, yet common approaches including RLHF suffer from unstable and resource-intensive training. In response to this challenge, we introduce  Args , Alignment as Reward-Guided Search, a novel framework that integrates alignment into the decoding process, eliminating the need for expensive RL training. By adjusting the model’s probabilistic predictions using a reward signal,  Args  generates texts with semantic diversity while being aligned with human preferences, offering a promising and flexible solution for aligning language models. Notably,  Args  demonstrates consistent enhancements in average reward compared to baselines across diverse alignment tasks and various model dimensions. For example, under the same greedy-based decoding strategy, our method improves the average reward by 19.56% relative to the baseline and secures a preference or tie score of 64.33% in GPT-4 evaluation. We believe that our framework, emphasizing decoding-time alignment, paves the way for more responsive language models in the future. Code is publicly available at:  https://github.com/deeplearning-wisc/args .

*

*  footnotetext:  Equal contributions. Work done while J.B. was an undergraduate researcher at UW-Madison.

1  Introduction

Large language models (LLMs) trained on massive datasets exhibit a remarkable ability to handle a wide array of tasks  (Wei et al.,  2022 ; Kaddour et al.,  2023 ) . However, due to the varied nature of their training data, these models can inadvertently generate misinformation and harmful outputs  (Gehman et al.,  2020 ; Weidinger et al.,  2021 ; Deshpande et al.,  2023 ) . This concern underscores the urgent challenge of language model alignment: ensuring these models’ behaviors agree with human objectives and safety considerations  (Ngo et al.,  2023 ; Casper et al.,  2023 ) .

In recent years, a spectrum of alignment strategies have emerged, with prominent methods showcasing the effectiveness of reinforcement learning with human feedback (RLHF)  (Christiano et al.,  2017 ; Ziegler et al.,  2019 ; Ouyang et al.,  2022 ; Bai et al.,  2022 ) . RLHF has gained widespread adoption among state-of-the-art models, including OpenAI’s GPT-4  (OpenAI,  2023 ) , Anthropic’s Claude  (Anthropic,  2023 ) , Google’s Bard  (Google,  2023 ) , and Meta’s Llama 2-Chat  (Touvron et al.,  2023b ) . A pivotal component within RLHF is proximal policy optimization (PPO), which employs an external reward model that mirrors human preferences for its optimization process. However, as noted in previous studies  (Henderson et al.,  2017 ; Wang et al.,  2023 ; Rafailov et al.,  2023 ; Zheng et al.,  2023b ) , implementing PPO introduces challenges of unstable and costly training. Furthermore, the need to repeat PPO training when altering the reward model hinders rapid customization to evolving datasets and emerging needs.

To address the aforementioned challenge, we introduce Alignment as Reward-Guided Search, or  Args , a novel framework designed to enhance the alignment of generated text with human-desired preferences.  Args  achieves this by employing a reward mechanism that directly guides the text generation process of a language model. Unlike traditional alignment approaches, our method integrates alignment into the decoding process, enabling quick realignments without having to go through the exhaustive process of retraining the foundational model using PPO. This is especially valuable in today’s rapidly changing field of machine learning, and ensures that models remain relevant and responsive to contemporary requirements without the need for extensive overhauls. Specifically, at each decoding step, our key idea is to adjust the model’s probabilistic prediction using a reward signal. This adjustment is crucial as it enables the generated text to both  (1)

maintain the semantic relevance with respect to the previous context, and

(2)

align with the reward criteria and human preference . These two sub-goals can be flexibly traded off with proper weighting on the reward signal, which degenerates to the standard maximum-likelihood decoding when the weight is zero. Notably, our reward-guided score can be seamlessly integrated with various token selection strategies, including both greedy and stochastic sampling.

Figure 1:  Illustration of  Args  (Alignment as Reward-Guided Search) framework.

We validate  Args  on the large-scale HH-RLHF (Helpful and Harmless) dataset  (Bai et al.,  2022 )  and demonstrate that our technique effectively guides the generation towards outputs that are preferable. For example, our method improves the average reward by

↑

↑

\uparrow

19.56% relative to the standard decoding and secures a preference or tie score of 64.33% in GPT-4 evaluation. Moreover, our method excels at generating lexically diverse continuations without compromising their contextual consistency. Qualitatively,  Args  offers less redundant and more informative outputs than the standard maximum-likelihood decoding, as illustrated in Table

1  . Additionally, we further emphasize the versatility of ARGS and demonstrate consistent improvement across different model architectures (LLaMa and OPT), sizes, and alignment tasks including Stanford Human Preferences (SHP) dataset  (Ethayarajh et al.,  2022 ) .
To summarize our contributions:

1.

We propose a novel framework  Args  , which postulates the alignment process as a reward-guided search problem that runs during decoding time. This framework not only omits the need for expensive RL training but also facilitates flexible customization to emerging needs.

2.

We conduct both qualitative and quantitative evaluations of  Args ’s performance, showcasing its superiority over existing approaches.  Args  effectively guides the outputs of the neural language model in alignment with human preferences.

3.

Importantly,  Args  brings a new perspective of decoding-time alignment to the field of AI safety. While traditional alignment strategies focus on optimization during the training phase, decoding-time alignment emphasizes the pivotal role of post-training adjustments. Such a shift in focus allows models to adjust to new reward signals and user requirements without the need for extensive retraining. We hope this inspires further research into post hoc alignment, leading to more efficient and safer AI systems in real-world applications.

2

Args : Alignment as Reward-Guided Search

In this section, we introduce  Args , a novel decoding framework that facilitates the alignment of generated text with human preferences, by employing a reward mechanism that directly guides the text generation process of a language model. Our method has two main components: (1)  reward-guided scoring , which assigns scores to possible continuations of the text, and (2)  token selection , which selects a continuation. We detail the reward-guided scoring method in Section

2.1

and the token selection methods in Section

2.2  .

2.1  Reward-Guided Scoring

Our goal is to steer the decoded outputs of language models in alignment with human preference.
At each decoding step, our key idea is to adjust the model’s probabilistic prediction by a reward signal (Figure

1  ). This adjustment is crucial as it enables the model to generate text that is not only coherent and contextually relevant but also tailored to satisfy specific alignment criteria or objectives.

Specifically, a reward model (RM) assigns a scalar reward value to each response. Following  Stiennon et al. ( 2020 ) , reward models are often trained on a dataset comprised of paired comparisons between two responses generated for th