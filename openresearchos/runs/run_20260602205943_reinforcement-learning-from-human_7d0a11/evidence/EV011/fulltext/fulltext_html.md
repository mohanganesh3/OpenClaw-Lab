[2204.05862] Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback

Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback

Yuntao Bai, Andy Jones, Kamal Ndousse,
&amp; Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort,

Correspondence to: {yuntao, jared}@anthropic.com

Author contributions are listed at the end of the paper.

Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion,
 &amp; Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds,

Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt,

Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark,

Sam McCandlish, Chris Olah, Ben Mann, Jared Kaplan  †

†  footnotemark:

\AND

Anthropic

Abstract

We apply preference modeling and reinforcement learning from human feedback (RLHF) to finetune language models to act as helpful and harmless assistants. We find this alignment training improves performance on almost all NLP evaluations, and is fully compatible with training for specialized skills such as python coding and summarization. We explore an iterated online mode of training, where preference models and RL policies are updated on a weekly cadence with fresh human feedback data, efficiently improving our datasets and models. Finally, we investigate the robustness of RLHF training, and identify a roughly linear relation between the RL reward and the square root of the KL divergence between the policy and its initialization. Alongside our main results, we perform peripheral analyses on calibration, competing objectives, and the use of OOD detection, compare our models with human writers, and provide samples from our models using prompts appearing in recent related work.

Contents

1  Introduction

1.1  Contributions

1.2  Summary of Evaluations and Metrics

1.3  Related Work

2  Data Collection

2.1  Task Specification and Crowdworkers

2.2  Helpfulness and Harmlessness (Red Teaming) Datasets

2.3  Models Deployed to the Feedback Interface and Associated Data Distributions

2.4  Comparing Models with Elo Scores

3  Preference Modeling for Helpfulness and Harmlessness

3.1  Models and Training Setup

3.2  Basic Scaling Results

3.3  Calibration of Preference Models and Implications for RL

3.4  Evaluating Helpful and Harmless Preference Models

4  Reinforcement Learning from Human Feedback

4.1  Training Setup

4.2  Robustness Experiments

4.3  An Approximately Linear Relation Between

D  KL

subscript  𝐷  KL

\sqrt{D_{\rm KL}}

and Reward

4.4  Tension Between Helpfulness and Harmlessness in RLHF Training

4.5  Iterated Online RLHF

4.6  Evaluations: Alignment Bonus, Honesty, and Biases

5  Competing Objectives, Specialized Skills, and OOD Detection

5.1  Mixing Helpful and Harmless Objectives

5.2  Summarization as a Specialized Skill

5.3  Natural Language RLHF on Code-Finetuned Models

5.4  Applying Out-of-Distribution Detection to Reject Strange or Harmful Requests

6  Qualitative Examples and Comparisons

6.1  Comparison with Human Writers

6.2  Sensitive Questions and Avoidance versus Engagement

6.3  Example Dialogues

7  Discussion

7.1  Limitations

7.2  Alignment Data as a Public Good

7.3  Broader Impacts

A  Details, Analysis, and Evaluations of Supervised Training

A.1  Context Distillation

A.2  Preference Modeling

A.3  Scaling of PM with Model and Dataset Size

B  Details, Analysis, and Evaluations of RLHF

B.1  Training Setup

B.2  More on Robustness Studies

B.3  Details of ‘Online’ RLHF

B.4  Robustness of ‘Online’ RLHF

B.5  Crowdworker Comparisons and Elo Scores

B.6  Elo Scores for Rejection Sampling Models

B.7  Stack Overflow Results

B.8  Further Analysis of RLHF on Code-Model Snapshots

B.9  Details of Applying Out-of-Distribution Detection to Reject Strange or Harmful Requests

B.10  Gender Bias Evaluation Details

C  Samples from PALMS, LaMDA, and InstructGPT Prompts

C.1  PALMS Sensitive Questions

C.2  InstructGPT Prompts

C.3  LaMDA Prompts

D  Details on Data Collection and Crowdworkers

D.1  Overview

D.2  Instructions and Interface

D.3  Data Quality Measurement Challenges

E  Details on NLP Evaluations Formatting and Prompts

1  Introduction

We would like to develop techniques to train AI agents that are helpful, honest, and harmless  [ Askell et al., 2021 ] . In this paper we show that we can train a relatively helpful and harmless  1

1  1 We do not focus explicitly on honesty/truthfulness in this paper, as we believe that techniques other than pure human feedback may be more efficient and effective at training models to be honest. But we certainly believe that honesty is a crucial goal for AI alignment, and our models do improve on evaluations of honesty (see Figure

5  ).

(HH) natural language assistant by collecting human preference data and applying the techniques of preference modeling (PMing) and reinforcement learning from human feedback (RLHF). Our full training process is summarized in Figure

2  .

Figure 1:  This plot summarizes crowdworker preferences for a variety of models, including context-distilled models, RLHF models trained on our ‘static’ dataset, and RLHF models trained by an iterated ‘online’ method for either helpfulness and harmlessness (HH) or for helpfulness only.
We present both Elo scores and a match to the frequency with which crowdworkers prefer samples as compared to the 52B context-distilled model.
For both helpfulness and harmlessness, a higher score is more desirable.

Our goal is not to define or prescribe what ‘helpful’ and ‘harmless’ mean but to evaluate the effectiveness of our training techniques, so for the most part we simply let our crowdworkers interpret these concepts as they see fit. We treat helpfulness and harmlessness separately, collecting distinct human-preference datasets for each. For helpfulness, we ask crowdworkers to solicit our models to assist with any purely text-based tasks such as answering questions, writing or editing documents, or discussing plans and decisions. For harmlessness, we invite crowdworkers to adversarially probe or ‘red-team’ our language models in order to provoke harmful responses: either to help them with harmful goals, such as planning a bank robbery, or to cause the AI to use toxic language.  2

2  2 We warn crowdworkers that they may encounter upsetting content, and we frequently invite them to cease this task and pursue ‘helpful’ mode instead; we will discuss our approach to red-teaming in a forthcoming publication.

At each stage of their conversations with the AI assistant, crowdworkers are presented with two possible responses. Those engaged in the helpfulness task are instructed to choose the more helpful and honest (i.e. better) response. Those engaged in the red teaming task are instructed to choose the more harmful (i.e. worse) response. These conversations and the expressed human preferences form our datasets.  3

3  3 Our helpfulness data is available at  https://github.com/anthropics/hh-rlhf , and our harmlessness data will be made available in the future. Our work has benefited from other publicly available alignment-related data, such as for summarization  [ Stiennon et al., 2020 ] , and we hope that the release of such datasets can be a standard practice for researchers working towards safe and beneficial AI.

Helpfulness and harmlessness often stand in opposition to each other. An excessive focus on avoiding harm can lead to ‘safe’ responses that don’t actually address the needs of the human. An excessive focus on being helpful can lead to responses that help humans cause harm or generate toxic content. We demonstrate this tension quantitatively by showing that preference models trained to primarily evaluate one of these qualities perform very poorly (much worse than chance) on the other. Fortunately, we find that PMs trained on a mixture of both datasets can nevertheless learn the right lessons and behave h