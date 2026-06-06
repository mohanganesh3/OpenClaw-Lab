# EV007: Beyond the Loss Curve: Scaling Laws, Active Learning, and the Limits of Learning from Exact Posteriors

URL: https://www.semanticscholar.org/paper/05d1d0125e3fc2ff45d8726bf0f6d51bc0bc3148
Year: 2026
Source: semantic_scholar
Arxiv: 2602.00315

## Abstract

How close are neural networks to the best they could possibly do? Standard benchmarks cannot answer this because they lack access to the true posterior p(y|x). We use class-conditional normalizing flows as oracles that make exact posteriors tractable on realistic images (AFHQ, ImageNet). This enables five lines of investigation. Scaling laws: Prediction error decomposes into irreducible aleatoric uncertainty and reducible epistemic error; the epistemic component follows a power law in dataset size, continuing to shrink even when total loss plateaus. Limits of learning: The aleatoric floor is exactly measurable, and architectures differ markedly in how they approach it: ResNets exhibit clean power-law scaling while Vision Transformers stall in low-data regimes. Soft labels: Oracle posteriors contain learnable structure beyond class labels: training with exact posteriors outperforms hard labels and yields near-perfect calibration. Distribution shift: The oracle computes exact KL divergence of controlled perturbations, revealing that shift type matters more than shift magnitude: class imbalance barely affects accuracy at divergence values where input noise causes catastrophic degradation. Active learning: Exact epistemic uncertainty distinguishes genuinely informative samples from inherently ambiguous ones, improving sample efficiency. Our framework reveals that standard metrics hide ongoing learning, mask architectural differences, and cannot diagnose the nature of distribution shift.
