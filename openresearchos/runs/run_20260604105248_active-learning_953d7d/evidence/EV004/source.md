# EV004: A Subset Selection Strategy for Gaussian Process Q-Learning of Process Optimization and Control

URL: https://www.semanticscholar.org/paper/001798724209a3449cb0a346fd11c149c91bde24
Year: 2025
Source: semantic_scholar
Arxiv: n/a

## Abstract

This work addresses a practical challenge in batch process optimization: the need for sample efficient learning methods due to the high cost and time-intensive nature of running physical batch processes. While reinforcement learning (RL) offers a promising framework for optimizing batch processes, traditional approaches require numerous experimental runs to converge to optimal policies. A novel sample efficient RL method that leverages Gaussian Processes (GPs) to accelerate learning from limited batch data is proposed. However, the direct application of GPs becomes computationally intractable as data accumulates batch-to-batch, and their performance degrades when training distributions shift during policy improvement. To address these challenges, an integrated framework that combines Q-learning with GPs was developed and a strategic subset selection mechanism using determinantal point processes is introduced to maintain computational efficiency while preserving diverse, high-performing samples. The method exploits problem structure and backward induction to further maximize sample efficiency and incorporates both aleatoric and epistemic uncertainty for robust policy improvements. The approach is demonstrated on a non-isothermal semi-batch reactor case study, showing significantly improved learning efficiency compared to exact GP strategies while maintaining computational tractability. The results highlight the method's practical applicability to industrial batch process optimization where experimental data is limited and costly to obtain.
