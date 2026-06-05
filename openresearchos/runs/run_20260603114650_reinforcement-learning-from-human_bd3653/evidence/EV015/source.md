# EV015: ESSA: Evolutionary Strategies for Scalable Alignment

URL: https://www.semanticscholar.org/paper/0311a0660b02ec1face3936a81e8ab5186789475
Year: 2025
Source: semantic_scholar
Arxiv: 2507.04453

## Abstract

Alignment of Large Language Models (LLMs) typically relies on Reinforcement Learning from Human Feedback (RLHF) with gradient-based optimizers such as Proximal Policy Optimization (PPO) or Group Relative Policy Optimization (GRPO). While effective, these methods require complex distributed training, large memory budgets, and careful hyperparameter tuning, all of which become increasingly difficult at billion-parameter scale. We present ESSA, Evolutionary Strategies for Scalable Alignment, a gradient-free framework that aligns LLMs using only forward inference and black-box optimization. ESSA focuses optimization on Low-Rank Adapters (LoRA) and further compresses their parameter space by optimizing only the singular values from an singular value decomposition (SVD) of each adapter matrix. This dimensionality reduction makes evolutionary search practical even for very large models and allows efficient operation in quantized INT4 and INT8 inference mode. Across these benchmarks ESSA improves the test accuracy of Qwen2.5-Math-7B by 12.6% on GSM8K and 14.8% on PRM800K, and raises the accuracy of LLaMA3.1-8B on IFEval by 22.5%, all compared with GRPO. In large-scale settings ESSA shows stronger scaling than gradient-based methods: on Qwen2.5-32B for PRM800K it reaches near-optimal accuracy twice as fast on 16 GPUs and six times as fast on 128 GPUs compared with GRPO. These results position evolutionary strategies as a compelling, hardware-friendly alternative to gradient-based LLM alignment, combining competitive quality with substantially reduced wall-clock time and engineering overhead.
