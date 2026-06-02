# EV015: SHE: Stepwise Hybrid Examination Reinforcement Learning Framework for E-commerce Search Relevance

URL: https://www.semanticscholar.org/paper/0395f85751310e37fa9f89a73fb6ce8a95d81445
Year: 2025
Source: semantic_scholar
Arxiv: 2510.07972

## Abstract

Query-product relevance prediction is vital for AI-driven e-commerce, yet current LLM-based approaches face a dilemma: SFT and DPO struggle with long-tail generalization due to coarse supervision, while traditional RLVR suffers from sparse feedback that fails to correct intermediate reasoning errors. We propose Stepwise Hybrid Examination (SHE), an RL framework that ensures logical consistency through Stepwise Reward Policy Optimization (SRPO). SRPO utilizes a hybrid reward mechanism-combining generative reward models with human-annotated verifiers-to provide fine-grained, step-level signals. To further enhance stability, SHE incorporates diversified data filtering to maintain policy entropy and a multi-stage curriculum learning protocol for progressive skill acquisition. Extensive experiments on real-world search benchmarks show that SHE improves both reasoning quality and relevance-prediction accuracy in large-scale e-commerce settings, outperforming SFT, DPO, GRPO, and other baselines, while also enhancing interpretability and robustness.
