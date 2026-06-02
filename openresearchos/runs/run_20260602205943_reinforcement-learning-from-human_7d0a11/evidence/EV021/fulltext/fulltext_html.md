[2509.09751] Meta-Learning Reinforcement Learning for Crypto-Return Prediction

Meta-Learning Reinforcement Learning for Crypto-Return Prediction

Junqiao Wang 1,2  ,
 Zhaoyang Guan 3  ,
 Guanyu Liu 10  ,
 Tianze Xia 4  ,
 Xianzhi Li 5  ,
 Shuo Yin 8  ,

Xinyuan Song 9

Chuhan Cheng 1  ,

Tianyu Shi 6  ,
 Alex Lee 7  ,

1 Sichuan University,
 2 Shanghai Artificial Intelligence Laboratory,
 3 Northwestern University,

4 Huazhong University of Science and Technology,
 5 Queen’s University,

6 University of Toronto,
 7 TrueNorth,
 8 Tsinghua University,

9 Emory University,
 10 University of Macau

Correspondence:

ty.shi@mail.utoronto.ca

Abstract

Predicting cryptocurrency returns is notoriously difficult: price movements are driven by a fast-shifting blend of on-chain activity, news flow, and social sentiment, while labeled training data are scarce and expensive. In this paper, we present  Meta-RL-Crypto , a unified transformer-based architecture that unifies meta-learning and reinforcement learning (RL) to create a fully self-improving trading agent. Starting from a vanilla instruction-tuned LLM, the agent iteratively alternates between three roles—actor, judge, and meta-judge—in a closed-loop architecture. This learning process requires no additional human supervision. It can leverage multimodal market inputs and internal preference feedback. The agent in the system continuously refines both the trading policy and evaluation criteria. Experiments across diverse market regimes demonstrate that Meta-RL-Crypto shows good performance on the technical indicators of the real market and outperforming other LLM-based baselines.

Meta-Learning Reinforcement Learning for Crypto-Return Prediction

Junqiao Wang 1,2 ,
Zhaoyang Guan 3 ,
Guanyu Liu 10 ,
Tianze Xia 4 ,
Xianzhi Li 5 ,
Shuo Yin 8 ,

Xinyuan Song 9

Chuhan Cheng 1  ,

Tianyu Shi 6  ,
 Alex Lee 7  ,

1 Sichuan University,
 2 Shanghai Artificial Intelligence Laboratory,
 3 Northwestern University,

4 Huazhong University of Science and Technology,
 5 Queen’s University,

6 University of Toronto,
 7 TrueNorth,
 8 Tsinghua University,

9 Emory University,
 10 University of Macau

Correspondence:

ty.shi@mail.utoronto.ca

1  Introduction

Large Language Models (LLMs) have shown promise in financial tasks like sentiment analysis and time-series reasoning Makri et al. ( 2025 ) , but face two key challenges: (i) reducing dependence on human-curated data, and (ii) reliably forecasting volatile cryptocurrency markets influenced by on-chain data, news, and social sentiment.

To address the challenges in cryptocurrency prediction, we propose Meta-RL-Crypto, a framework that combines meta-reward-driven self-improvement with multi-modal trading intelligence in Figure  LABEL:fig:architecture . The central innovation is a triple-loop learning process in which a single large language model (LLM) takes on three distinct roles. First, the Actor processes on-chain metrics (such as gas fees and transaction graphs), news, and sentiment to generate next-day forecasts for crypto-assets  Meister and Price ( 2024 ); Liu and Jia ( 2025 ) . Next, the Judge evaluates these forecasts using a multi-objective reward vector, which incorporates absolute returns, the Sharpe ratio, drawdown control, and sentiment alignment. This approach reduces potential bias from relying on a single metric. Finally, the Meta-Judge refines the Judge’s reward policy through preference comparisons, helping to prevent reward drift and length bias  Lee et al. ( 2024 ); Wu et al. ( 2024 ) . This closed-loop system enables continuous self-improvement without human intervention, allowing it to adapt dynamically to shifts in the market.

Figure 1:

Overall Architecture of Meta-RL-Crypto. 
The system consists of a shared LLM that cyclically adopts the roles of  Actor ,  Judge , and  Meta-Judge .
Market signals (on-chain metrics, off-chain news, sentiment scores) are encoded into structured prompts used by the Actor to generate forecasts.
Each prediction is then scored by the Judge using a multi-dimensional reward vector, which the Meta-Judge uses to enforce preference consistency and evaluate the Judge itself.

Our contributions are as follows:

•

Unified Meta-Learning RL Framework:  We combine meta-reward self-improvement with crypto-specific trading goals, creating a single actor–judge–meta-judge loop that works on raw multimodal data.

•

Multi-Objective Reward Design:  A supervisory system using financial and emotional incentives helps prevent reward hacking and promotes better trading behavior.

•

Empirical Validation:  Initial experiments with BTC, ETH, and SOL daily trading show that Meta-RL-Crypto outperforms models like Informer and PatchTST, as well as traditional indicators like MACD, while matching or surpassing GPT-4-based baselines—without using extra human-labeled data.

2  Related Works

2.1  LLMs for Financial Analysis and Cryptocurrency Markets

Recent advancements in large language models (LLMs) have transformed financial-market research, though most studies have concentrated on equities, leaving the rich on-chain data of cryptocurrencies less explored  Roumeliotis et al. ( 2024 ); Li et al. ( 2024 ) . Early time-series forecasting in finance evolved from traditional econometrics to machine learning and reinforcement learning methods, with Long Short-Term Memory networks proving effective for sequential price data  Fjellström ( 2022 ); Siami‐Namini and Siami Namin ( 2018 ); Sezer et al. ( 2019 ) . Transformer-based models, such as Informer  Zhou et al. ( 2021 ) , AutoFormer  Wu et al. ( 2021 ) , PatchTST  Nie et al. ( 2022 ) , and TimesNet  Liu et al. ( 2023b ) , now lead the field in long-horizon forecasting. In addition, domain-specific LLMs—such as FinGPT  Liu et al. ( 2023a ) , BloombergGPT  Wu et al. ( 2023 ) , and FinMA  Xie et al. ( 2023 ) —expand LLM capabilities to sentiment analysis, entity recognition, question answering, and market prediction. While these techniques show promise for more robust decision-making, their application to cryptocurrency markets remains underexplored. Our work addresses this gap by integrating both on-chain and off-chain signals with self-reflective LLM agents to navigate the volatile, information-rich landscape of cryptocurrency assets.

Reinforcement learning has shown promise in addressing the challenges of delayed feedback, making it well-suited for tasks like comment ranking, where the effects of actions may not be apparent until later. For instance, Yahoo’s ranking system used contextual bandits to dynamically optimize comment visibility based on user interactions  Kulkarni and Rodd ( 2020 ) . In finance, RL agents have used social media sentiment signals, such as those from Twitter (X), to inform trading decisions and adjust portfolios  Xiao and Chen ( 2018 ) . Building on these insights, our work treats comment ranking as an RL problem by modeling tweet-comment pairs as sequential episodes and using subsequent market returns as delayed reward signals. This setup facilitates long-term credit assignment, aligning model decisions with financial outcomes and providing a more meaningful evaluation of comment informativeness. Reward modeling in financial text analysis has traditionally focused on price prediction tasks  Jiang et al. ( 2017 ) , where models predict asset price movements based on textual data. Recent approaches have aimed to align textual signals with economic indicators or other quantitative metrics  Yang et al. ( 2020 ) , but still depend on extensive manual supervision and predefined labels.

2.2  Meta-Learning in Reinforcement Learning

Recent approaches have explored how a single instruction-tuned seed model can generate its own training signals through self-play  Chen et al. ( 2024 ); Fang et al. ( 2025 ); Zhang et al. ( 2024 ); Shinn et al. ( 2025 ) . In these frameworks, the model assumes three roles: the Actor generates multiple candidate responses for each user prompt; the