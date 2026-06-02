# EV002: Towards Stable and Effective Reinforcement Learning for Mixture-of-Experts

URL: https://www.semanticscholar.org/paper/0f37ecc85a5532092c8628e7c17232a3af3a5154
Year: 2025
Source: semantic_scholar
Arxiv: 2510.23027

## Abstract

Recent advances in reinforcement learning (RL) have substantially improved the training of large-scale language models, leading to significant gains in generation quality and reasoning ability. However, most existing research focuses on dense models, while RL training for Mixture-of-Experts (MoE) architectures remains underexplored. To address the instability commonly observed in MoE training, we propose a novel router-aware approach to optimize importance sampling (IS) weights in off-policy RL. Specifically, we design a rescaling strategy guided by router logits, which effectively reduces gradient variance and mitigates training divergence. Experimental results demonstrate that our method significantly improves both the convergence stability and the final performance of MoE models, highlighting the potential of RL algorithmic innovations tailored to MoE architectures and providing a promising direction for efficient training of large-scale expert models.
