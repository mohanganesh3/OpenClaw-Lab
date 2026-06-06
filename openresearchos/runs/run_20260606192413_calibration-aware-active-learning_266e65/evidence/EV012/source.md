# EV012: RewardUQ: A Unified Framework for Uncertainty-Aware Reward Models

URL: https://www.semanticscholar.org/paper/3ed59a4a81d0b71e45fb13505ae53e933d802d62
Year: 2026
Source: semantic_scholar
Arxiv: 2602.24040

## Abstract

Reward models are central to aligning large language models (LLMs) with human preferences. Yet most approaches rely on pointwise reward estimates that overlook the epistemic uncertainty in reward models arising from limited human feedback. Recent work suggests that quantifying this uncertainty can reduce the costs of human annotation via uncertainty-guided active learning and mitigate reward overoptimization in LLM post-training. However, uncertainty-aware reward models have so far been adopted without thorough comparison, leaving them poorly understood. This work introduces a unified framework, RewardUQ, to systematically evaluate uncertainty quantification for reward models. We compare common methods along standard metrics measuring accuracy and calibration, and we propose a new ranking strategy incorporating both dimensions for a simplified comparison. Our experimental results suggest that model size and initialization have the most meaningful impact on performance, and most prior work could have benefited from alternative design choices. To foster the development and evaluation of new methods and aid the deployment in downstream applications, we release our open-source framework as a Python package. Our code is available at https://github.com/lasgroup/rewarduq.
