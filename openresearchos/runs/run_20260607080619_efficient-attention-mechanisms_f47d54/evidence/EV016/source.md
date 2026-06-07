# EV016: Robust Filter Attention: Self-Attention as Precision-Weighted State Estimation

URL: https://www.semanticscholar.org/paper/008e38b78335874466e907b8bb6c9a6e136fd58f
Year: 2025
Source: semantic_scholar
Arxiv: 2509.04154

## Abstract

We introduce Robust Filter Attention (RFA), a formulation of self-attention as a robust state estimator. Each token is treated as a noisy observation of a latent trajectory governed by a linear stochastic differential equation (SDE), and attention weights are determined by consistency under this model rather than static feature similarity. Under isotropic noise and decay assumptions, RFA matches the computational complexity of standard attention. On language modeling benchmarks, RFA achieves lower perplexity than RoPE within the training window while remaining stable under zero-shot extrapolation to longer contexts. The framework also provides a dynamical interpretation of standard positional mechanisms, connecting rotational embeddings and recency biases to transport and uncertainty propagation induced by stochastic dynamics.
