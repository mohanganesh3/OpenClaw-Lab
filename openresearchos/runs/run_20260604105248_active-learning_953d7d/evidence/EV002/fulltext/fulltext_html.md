[2406.11657] Can LLM be a Personalized Judge?

Can LLM be a Personalized Judge?

Yijiang River Dong ∗

Tiancheng Hu ∗

Nigel Collier

{yd358, th656, nhc30}@cam.ac.uk

University of Cambridge

Abstract

Ensuring that large language models (LLMs) reflect diverse user values and preferences is crucial as their user bases expand globally. It is therefore encouraging to see the growing interest in LLM personalization within the research community. However, current works often rely on the LLM-as-a-Judge approach for evaluation without thoroughly examining its validity. In this paper, we investigate the reliability of LLM-as-a- Personalized -Judge—asking LLMs to judge user preferences based on personas. Our findings suggest that directly applying LLM-as-a-Personalized-Judge is less reliable than previously assumed, showing low and inconsistent agreement with human ground truth. The personas typically used are often overly simplistic, resulting in low predictive power. To address these issues, we introduce verbal uncertainty estimation into the LLM-as-a-Personalized-Judge pipeline, allowing the model to express low confidence on uncertain judgments. This adjustment leads to much higher agreement (above 80%) on high-certainty samples for binary tasks. Through human evaluation, we find that the LLM-as-a-Personalized-Judge achieves comparable performance to third-party humans evaluation and even surpasses human performance on high-certainty samples. Our work indicates that certainty-enhanced LLM-as-a-Personalized-Judge offers a promising direction for developing more reliable and scalable methods for evaluating LLM personalization. Our code is available at  https://github.com/dong-river/Personalized-Judge .

Can LLM be a Personalized Judge?

Yijiang River Dong  ∗

and  Tiancheng Hu  ∗

and  Nigel Collier

{yd358, th656, nhc30}@cam.ac.uk

University of Cambridge

*

*  footnotetext:  Equal contribution

1  Introduction

Figure 1:

Overall workflow of Personalized Judge.  Given a subjective question and two distinct responses, we ask an LLM to infer the preference of a real user based on a user persona. We also ask the LLM to estimate its certainty level in this prediction. The inferred preference is then compared against the user’s self-reported ground truth to evaluate the performance of the Judge.

As large language models (LLMs) gain widespread adoption among global users with diverse backgrounds, it is imperative to ensure these models designed to reflect their values and preferences  Sorensen et al. ( 2024 ); Kirk et al. ( 2024 ) . However, the current alignment process often assumes a homogeneous set of human preferences and ignores individual perspectives, even in context-dependent, subjective tasks  Santurkar et al. ( 2023 ) . Therefore, efforts have been made to fine-tune LLMs to encode individual preferences or enhance role-playing capabilities  Jang et al. ( 2023 ); Shao et al. ( 2023 ); Occhipinti et al. ( 2023 ); Li et al. ( 2024a ); Andukuri et al. ( 2024 )  with “LLM-as-a-Judge” as the main evaluation metric  Zheng et al. ( 2023 ) , often without adequate validation.

Despite “LLM-as-a-Judge” showing high agreement with human annotators in many tasks, its effectiveness for personalization tasks remains largely unscrutinized. MT-Bench  Zheng et al. ( 2023 )  includes a role-playing component but only considered simplistic personas, such as "imagine you are a doctor," without addressing more complex personas that encompass demographics, user descriptions, and prior interactions—settings increasingly employed in recent research. Furthermore, a persona description may not always be contextually relevant. Knowing that someone is a doctor, for instance, provides little insight into their favorite types of beverages. We refer to this issue as the  persona sparsity issue .  1

1  1 Our use of the term “persona sparsity” diverges from works like  Zheng et al. ( 2020 ); Song et al. ( 2021 ) . While they typically refer to the scarcity of naturalistic dialog data directly reflecting persona variables, we highlight a related but distinct problem: the available persona variables may not offer an informed prior about the person involved for a specific task.

In this paper, we examine the validity of LLM-as-a-Judge for personalization, where the objective is to generate personalized outputs based on a given user persona (see Figure

1  ). We assess performance on tasks where ground truth data is available, including PRISM  Kirk et al. ( 2024 ) , OpinionQA  Santurkar et al. ( 2023 ) , Public Reddit  Staab et al. ( 2024 ) , and Empathetic Conversation  Omitaomu et al. ( 2022 ) . To address the issue of persona sparsity, we then propose a verbal uncertainty estimation component into the Judge pipeline. By articulating its own certainty levels, an LLM can assign lower certainty to samples for which it perceives insufficient predictive power. Additionally, we conduct a crowdsourcing experiment and compare the performance of LLM-as-a-Personalized-Judge to third-person human evaluation.

Our findings are as follows: (1) Contrary to previous assumptions, standard LLM-as-a-Judge is not sufficiently reliable for personalization tasks, showing only around 70% agreement with human judgments in binary choice scenarios, and dropping below 60% for certain tasks. (2) We identify persona sparsity as a major factor contributing to this unreliability. To address this, we introduce verbal uncertainty estimation into the LLM-as-Personalized-Judge process and achieve above 80% performance in high-certainty samples. (3) In a crowdsourcing experiment, we find that LLM-as-a-Personalized-Judge achieves performance comparable to third-person  2

2  2 Here, first-person evaluation refers to judgments made by the individuals for whom the personalization is intended, reflecting their own preferences and values. Third-person evaluation involves external annotators who assess the personalization based on persona descriptions rather than personal preferences.

human judgment and even surpasses human performance on high-certainty samples. While first-person human evaluation from diverse backgrounds remains the gold standard for personalization, in the absence of such annotations, LLM-as-a-Personalized-Judge with certainty thresholding could serve as an effective and scalable alternative.

2  Background and Related Work

Personalization  in machine learning refers to the process of tailoring a model’s output to suit the unique preferences, needs, and behaviors of individual users (see  Fan and Poole ( 2006 )  for an in-depth discussion). This concept is at the core of recommender systems  Sarwar et al. ( 2001 ) , and been explored in various contexts in NLP, such as dialogue system  Li et al. ( 2016 ); Zhang et al. ( 2018 ) , summarization  Díaz and Gervás ( 2007 ); Yan et al. ( 2011 ) , user profiling and computational sociolinguistics  Nguyen et al. ( 2016 ) . These studies typically aim to understand the diverse linguistic patterns of users from varying backgrounds and contexts and to integrate persona information to enhance task performance. For surveys on these topics, see  Flek ( 2020 ); Hovy and Yang ( 2021 ); Yang et al. ( 2024 ) .

In the context of LLMs, personalization has become even more critical due to the vast, diverse, and ever-growing user base. The necessity to align LLMs to a pluralistic set of user needs is discussed in  Sorensen et al. ( 2024 ) . However, the current alignment processes typically assume a single set of human preferences and researchers are just beginning to explore methods to address the varied preferences and values of different users, either at the collective level  Conitzer et al. ( 2024 ); Klingefjord et al. ( 2024 )  or at individual level  Salemi et al. ( 2023 ); Gao et al. ( 2024 ); Li et al. ( 2024b ); Jang et al. ( 2023 ); Wang et al. ( 2023 ) . Our study focuses on the evaluation aspect of personalized alignment approaches.

A challenging issue