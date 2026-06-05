[2308.03688] AgentBench: Evaluating LLMs as Agents

\useunder

\ul

\doparttoc  \faketableofcontents

AgentBench : Evaluating LLMs as Agents

Xiao Liu

Tsinghua University

Hao Yu

Tsinghua University

Hanchen Zhang

Tsinghua University

Yifan Xu

Tsinghua University

Xuanyu Lei

Tsinghua University

Hanyu Lai

Tsinghua University

Yu Gu

The Ohio State University

Hangliang Ding

Tsinghua University

Kaiwen Men

Tsinghua University

Kejuan Yang

Tsinghua University

Shudan Zhang

Tsinghua University

Xiang Deng

The Ohio State University

Aohan Zeng

Tsinghua University

Zhengxiao Du

Tsinghua University

Chenhui Zhang

Tsinghua University

Sheng Shen

UC Berkeley

Tianjun Zhang

UC Berkeley

Yu Su

The Ohio State University

Huan Sun

The Ohio State University

Minlie Huang

Tsinghua University

Yuxiao Dong

Tsinghua University

Jie Tang

Tsinghua University

Abstract

Large Language Models (LLMs) are becoming increasingly smart and autonomous, targeting real-world pragmatic missions beyond traditional NLP tasks.
As a result, there has been an urgent need to  evaluate LLMs as agents  on challenging tasks in interactive environments.
We present  AgentBench , a multi-dimensional evolving benchmark that currently consists of 8 distinct environments to assess LLM-as-Agent’s reasoning and decision-making abilities in a multi-turn open-ended generation setting.
Our extensive test over 27 API-based and open-sourced (OSS) LLMs shows that, while top commercial LLMs present a strong ability of acting as agents in complex environments, there is a significant disparity in performance between them and OSS competitors.
We identify the typical reasons of failures in environments and LLMs, showing that poor long-term reasoning, decision-making, and instruction following abilities are the main obstacles for developing usable LLM agents.
Training on code and high quality multi-turn alignment data could improve agent performance.
Datasets, environments, and an integrated evaluation package for  AgentBench  are released at  https://github.com/THUDM/AgentBench .

1

1  footnotetext:  XL and HY are lead authors that contributed equally. Email:  {shawliu9,longinyh}@gmail.com

2

2  footnotetext:  Work partially done when HY, YG visited Tsinghua University.

3

3  footnotetext:  Website for  AgentBench  leaderboard &amp; demos:  https://llmbench.ai/agent

Figure 1:  An overview of LLMs on  AgentBench . While LLMs begin to manifest their proficiency in LLM-as-Agent, gaps between models and the distance toward practical usability are significant.

1  Introduction

Intelligent agents and autonomous entities  (Searle,  1970 ; Maes,  1994 ; Wooldridge &amp; Jennings,  1995 )  that are capable of decision-making and action execution in particular environments have been key concepts of artificial intelligence (AI) historically.
Notwithstanding substantial advancements in deep learning algorithms applied in both computer vision and natural language processing (NLP), their potential for developing efficient and practically usable assisting agents remains largely unexplored.

The advent of Large Language Models (LLMs)  (Brown et al.,  2020 ; Chowdhery et al.,  2022 ; Touvron et al.,  2023 ) , such as GPT-4  (OpenAI,  2023 ) , has brought plenty of new opportunities to this realm.
Through extensive alignment training  (Ouyang et al.,  2022 ; Wei et al.,  2022a ; Sanh et al.,  2022 ) , LLMs have not only mastered traditional NLP tasks but also showcased an impressive ability to comprehend human intent and execute instructions.
This has spurred the development of various LLM-based applications for autonomous goal completion (like AutoGPT  (Richards,  2023 ) , BabyAGI  (Nakajima,  2023 ) , AgentGPT  (age,  2023 ) ) as well as LLM agents situated in social and game contexts  (Park et al.,  2023 ; Wang et al.,  2023b ; Zhu et al.,  2023 ) , sparking substantial public interest and discussions.

Despite these advancements, the lack of a systematic and standard benchmark to evaluate LLM-as-Agent presents a critical challenge.
Historically, text-based game environments  (Osborne et al.,  2022 ; Côté et al.,  2019 ; Hausknecht et al.,  2020 ; Urbanek et al.,  2019 )  have been employed for language agent evaluation.
But they often suffer from the limitation of closed, discrete action spaces, as well as their primarily narrow focus on models’ commonsense grounding.
More recently, attempts on embodied agents  (Reed et al.,  2022 ; Huang et al.,  2022 ; Ahn et al.,  2022 )  have employed complicated multi-modal simulators based on games  (Küttler et al.,  2020 ; Fan et al.,  2022 ) , GUI  (Shi et al.,  2017 ; Toyama et al.,  2021 ) , and indoor scenes  (Shen et al.,  2021 ; Srivastava et al.,  2022 ) .
However, these simulators, despite their complexity, do not accurately reflect the practical use cases of LLMs, and their multi-modal nature creates a hurdle for the urgent evaluation of existing text-only LLMs.
Finally, most benchmarks now for agents focus on single environments and thus fail to provide a comprehensive overview of LLMs across diverse application scenarios.

Figure 2:

AgentBench  is the first systematic benchmark to evaluate LLM-as-Agent on a wide array of real-world challenges and 8 distinct environments. In total, 27 LLMs are examined in this edition.

To address these challenges, we introduce  AgentBench , a multi-dimensional benchmark designed to evaluate LLM-as-Agent across a spectrum of different environments.
 AgentBench  encompasses eight distinct environments (Cf. Figure

4  ), which could be categorized into three types of groundings:

•

Code:  Operating System, Database, Knowledge Graph  (Anonymous,  2023 )

•

Game:  Digital Card Game, Lateral Thinking Puzzles, House-Holding  (Shridhar et al.,  2020b )

•

Web:  Web Shopping  (Yao et al.,  2022 ) , Web Browsing  (Deng et al.,  2023 )

All datasets, whether newly created or adapted from existent ones, are meticulously designed and reformulated to simulate interactive environments where text-only LLMs can operate as autonomous agents.
 AgentBench  thus systematically evaluate an LLM’s core abilities, including following instructions  (Ouyang et al.,  2022 ) , coding  (Chen et al.,  2021 ) , knowledge acquisition  (Joshi et al.,  2017 ; Talmor et al.,  2019 ) , logical reasoning  (Srivastava et al.,  2023 ) , and commonsense grounding  (Shridhar et al.,  2020a ) .
It serves as an ideal testbed for both LLM and agent evaluation.

In addition, we develop a unified evaluation toolkit for LLMs to operate on diverse customized agent tasks, thus enabling a comprehensive benchmarking of the LLM-as-Agent ability of 27 different LLMs on  AgentBench , including both API-based and OSS models.
Our results reveal that top-tier models like GPT-4 are capable of handling a wide array of real-world tasks, indicating the potential for developing a potent, continuously learning agent.
However, we also note a significant performance gap between these top-tier models and their OSS competitors.
Despite the recent success of OSS LLMs and their competitive scores on several benchmarks  (Li et al.,  2023 ; Chen et al.,  2021 ; Cobbe et al.,  2021 ) , their performance on the challenging  AgentBench  tasks lags considerably.
This underscores the necessity for additional efforts to enhance the learning abilities of OSS LLMs.

We identify portions of agent task failures in different environments and LLMs, unveiling the insufficient abilities of long-term reasoning, decision-making, and instruction following in existing LLMs.
Comparisons between different LLMs manifest that a proper strategy of introducing code training can help improve LLM-as-Agent. Alignment training over high-quality data (e.g., data generated by  gpt-4 ) could also help improve LLM agents.
In summary, our contributions are:

•

We introduce the concept of evaluating LLMs as agents and present  AgentBench , a comprehensive benchmark to standardize the evaluation. It defines