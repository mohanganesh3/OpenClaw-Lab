# EV004: Probabilistic Uncertain Reward Model: A Natural Generalization of Bradley-Terry Reward Model

URL: https://www.semanticscholar.org/paper/00b883250ce2bfe6e983426882e355539feca9b8
Year: 2025
Source: semantic_scholar
Arxiv: 2503.22480

## Abstract

Reinforcement learning from human feedback (RLHF) is a critical technique for training large language models. However, conventional reward models based on the Bradley-Terry model (BTRM) often suffer from overconfidence when faced with inconsistent labels or out-of-distribution samples, leading to reward hacking, where the policy model blindly optimizes for proxy rewards while degrading true performance. This paper proposes the Probabilistic Uncertain Reward Model (PURM), which generalizes the Bradley-Terry model to learn the reward distributions that emerged from the preference data. We theoretically derive the loss function of PURM and introduce a novel method that uses the overlap between distributions to quantify uncertainty. Empirical results show that PURM outperforms existing methods with more accurate reward and sound uncertainty estimations, and sustains effective learning for more optimization steps and obtain higher maximum win rate in RLHF. The data and code of this paper are released at https://anonymous.4open.science/r/Probabilistic-Uncertain-Reward-Model/
