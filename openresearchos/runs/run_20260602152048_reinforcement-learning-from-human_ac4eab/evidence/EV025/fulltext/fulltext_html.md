[2407.00087] ARES: Alternating Reinforcement Learning and Supervised Fine-Tuning for Enhanced Multi-Modal Chain-of-Thought Reasoning Through Diverse AI Feedback

ARES: Alternating Reinforcement Learning and Supervised Fine-Tuning for Enhanced Multi-Modal Chain-of-Thought Reasoning
 Through Diverse AI Feedback

Ju-Seung Byun

Equal contribution

Jiyun Chun  †

†  footnotemark:

Jihyung Kil

Andrew Perrault

Department of Computer Science and Engineering
 The Ohio State University

{byun.83,chun.203,kil.5,perrault.17}@osu.edu

Abstract

Large Multimodal Models (LMMs) excel at comprehending human instructions and demonstrate remarkable results across a broad spectrum of tasks. Reinforcement Learning from Human Feedback (RLHF) and AI Feedback (RLAIF) further refine LLMs by aligning them with specific preferences. These methods primarily use ranking-based feedback for entire generations. With advanced AI models (Teacher), such as  GPT-4  and  Claude 3 Opus , we can request various types of detailed feedback that are expensive for humans to provide. We propose a two-stage algorithm  ARES  that  A lternates  RE inforcement Learning (RL) and  S upervised Fine-Tuning (SFT). First, we request the Teacher to score how much each sentence contributes to solving the problem in a Chain-of-Thought (CoT). This sentence-level feedback allows us to consider individual valuable segments, providing more granular rewards for the RL procedure. Second, we ask the Teacher to correct the wrong reasoning after the RL stage. The RL procedure requires massive efforts for hyperparameter tuning and often generates errors like repetitive words and incomplete sentences. With the correction feedback, we stabilize the RL fine-tuned model through SFT. We conduct experiments on multi-model dataset ScienceQA and A-OKVQA to demonstrate the effectiveness of our proposal. ARES rationale reasoning achieves around 70% win rate against baseline models judged by  GPT-4o . Additionally, we observe that the improved rationale reasoning leads to a 2.5% increase in inference answer accuracy on average for the multi-modal datasets.

1

1  1 Code:  https://github.com/Amyyyyeah/ARES

ARES: Alternating Reinforcement Learning and Supervised Fine-Tuning for Enhanced Multi-Modal Chain-of-Thought Reasoning
 Through Diverse AI Feedback

1  Introduction

Large Language Models (LLMs) and Large Multimodal Models demonstrate remarkable performance across diverse language and multi-modal tasks  (Brown et al.,  2020 ; Chowdhery et al.,  2022 ; Touvron et al.,  2023 ; Zhang et al.,  2022a ; Liu et al.,  2023a ; Goyal et al.,  2023 ) . However, these Large Models (LMs) often generate toxic and biased content  (Gehman et al.,  2020 ; Tamkin et al.,  2021 )  because LMs are primarily trained to predict the next token based on extensive corpus datasets. To align LM behavior more closely with user preferences, previous works  (Glaese et al.,  2022 ; Ouyang et al.,  2022 )  fine-tune their models using Reinforcement Learning from Human Feedback (RLHF). Furthermore, with advancements in LMs, an advanced LM feedback start replacing costly human feedback, yielding Reinforcement Learning from AI Feedback (RLAIF)  (Lee et al.,  2023 ; Bai et al.,  2022 ; Yuan et al.,  2024 ) .

Figure 1:

Overview of ARES:  ARES alternates RL and SFT. The Teacher provides scores (rewards) for each sentence for RL.  Blue  indicates the advantages of RL, and  red  indicates potential degeneration. ARES corrects the issues through the Teacher’s correction feedback.

However, RLHF and RLAIF encounter two significant challenges. First, both methods often utilize ranking-based feedback  (Ouyang et al.,  2022 ) , which orders the preferences of generated samples. For instance, if sample

A

𝐴

A

is preferred over sample

B

𝐵

B

, the model is fine-tuned to generate more outputs like

A

𝐴

A

and fewer like

B

𝐵

B

. However, if

B

𝐵

B

contains certain valuable parts, these parts are often disregarded. To alleviate this issue,  Lightman et al. ( 2023 ); Luo et al. ( 2024 )  propose sentence-level feedback, applying it solely to the reward model without Reinforcement Learning (RL), called the Process-supervised Reward Model (PRM). It demonstrates its potential through search algorithms on the PRM, such as best-of-

N

𝑁

N

or Monte Carlo Tree Search (MCTS). Furthermore,  Wang et al. ( 2024 )  demonstrate the effectiveness of RL with sentence-level feedback by heuristically scoring each sentence in math problems, where evaluating the predicted answer is straightforward. Thus, sentence-level feedback exhibits significant promise compared to existing ranking-based feedback. Nonetheless, acquiring sentence-level feedback is more costly than ranking-based feedback.

Second, the RL process is inherently unstable and requires extensive hyperparameter tuning  (Eimer et al.,  2023 ) . This instability often results in the generation of repetitive words and truncated sentences. Hyperparameter tuning becomes an enormous burden as the model size increases, making exhaustive tuning seemingly impossible, especially for individuals. The existing RLHF method  (Ouyang et al.,  2022 )  recycles the dataset used in pretraining within the loss function with Proximal Policy Optimization (PPO)  (Schulman et al.,  2017 )  to mitigate this degeneration problem. However, this approach prevents the model from fully maximizing the sum of rewards through RL and may limit the opportunity for diverse improvements, which differ from the pretraining distribution.

In this work, we aim to address the two challenges mentioned above through various types of feedback using an advanced AI model as a Teacher. Many advanced AI models, including  GPT-4  and  Claude 3 Opus , are already used as evaluators for many tasks and generate reliable human-level answers  (Liu et al.,  2023b ; Sottana et al.,  2023 ) .

1  )

1)

We request a score from Teacher for each sentence, ranging from

0.0

0.0

0.0

to

1.0

1.0

1.0

. Each score indicates how much a sentence contributes to solving the problem. This provides detailed reward feedback to the training model and can be applied to both mathematical and more general multi-modal Chain-of-Thought (CoT) (or rationale reasoning) problems.

2  )

2)

We ask the Teacher to identify and correct minor errors in the RL results, such as incorrect or cut-off parts. With this corrected dataset, we fine-tune the model using Supervised Fine-Tuning (SFT). This stage allows the model to maximize the rewards while properly deviating from the pretraining distribution. In summary, We propose a hybrid algorithm  ARES  that  A lternates  RE inforcement Learning and  S upervised Fine-Tuning.

To evaluate how much rationale reasoning can be improved through the ARES framework, we use the ScienceQA and A-OKVQA datasets, which are large-scale, multi-modal datasets that include rationale reasoning data. We use Multimodal-CoT (MM-CoT)  (Zhang et al.,  2023b )  as the baseline. MM-CoT employs two separate models: one model is responsible for generating rationale, and the other model, an inference model, processes the concatenated input (problem and generated rationale). This distinct framework enhances performance, even with relatively smaller models like

Flan-Alpaca  Base

subscript

Flan-Alpaca

Base

\text{Flan-Alpaca}_{\text{Base}}

(Chia et al.,  2023 )  (251M) and

Flan-Alpaca  Large

subscript

Flan-Alpaca

Large

\text{Flan-Alpaca}_{\text{Large}}

(790M) with ViT feature  (Dosovitskiy et al.,  2021 ) . We perform ARES on the rationale reasoning model of MM-CoT. We compare ARES rationale reasoning with that of MM-CoT through  GPT-4o , determining which rationale is better and computing the win rate. Additionally, we check whether the improved rationale reasoning leads to better answer accuracy. Our results show that our rationale reasoning outperforms the baselines with around 70% win rate and demonstrates 2.5% increase in inference answer accura