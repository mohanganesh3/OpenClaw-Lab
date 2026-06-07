# EV019: Lightweight hybrid neural network with physics consistency regularization for lithium-ion battery state of health estimation

URL: https://www.semanticscholar.org/paper/00a812e48c1333bcd50811aebc7a32af82b29894
Year: 2026
Source: semantic_scholar
Arxiv: n/a

## Abstract

Accurate State of Health (SOH) prediction for lithium-ion batteries is critical, yet conventional data-driven methods often lack interpretability and can generalize poorly under distribution shifts, while model-driven methods are computationally costly. PINN-inspired regularization provides a practical compromise by incorporating physics-consistency priors into data-driven models. This study proposes an Improved Lightweight Hybrid PINN-inspired model (IHPINN) with three core innovations. First, an efficient feature extractor replaces large fully connected layers with grouped linearization, low-rank approximation, and parameter sharing, significantly reducing weights and floating-point operations (FLOPs). Second, an adaptive activation module learnably gates sinusoidal and Swish activations, adjusting nonlinearity to mitigate gradient issues and stabilize convergence. Third, a shared-attention mechanism uses a single shared projection for Q, K, and V to reduce redundancy. The resulting model combines data fitting with PINN-inspired regularization (learned dynamics consistency and monotonicity) to achieve stable temporal representations and a better data-regularization balance with fewer parameters. Compared to a baseline PINN, IHPINN achieves a 52.8% MSE improvement on the HUST dataset with 19.3% fewer parameters and 14.4% less training time, and a 39.1% MSE improvement on the MIT dataset.
