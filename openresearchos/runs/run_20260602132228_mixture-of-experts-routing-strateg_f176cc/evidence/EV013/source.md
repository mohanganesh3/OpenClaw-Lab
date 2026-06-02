# EV013: PP-MoE: A Physics-Prioritized Mixture of Experts Scheme for Adaptive Channel Estimation

URL: https://www.semanticscholar.org/paper/33ce894ce1d1568e0a2482d734c9c613a553e795
Year: 2026
Source: semantic_scholar
Arxiv: n/a

## Abstract

Accurate Channel State Information is prerequisite for intelligent sensing and ubiquitous connectivity. However, the diversity of channel conditions—from sparse to dense and static to fast-varying—fundamentally challenges traditional single and fixed estimation algorithms. To address this issue, this paper proposes a Physics-Prioritized Mixture of Experts (PP-MoE) scheme, leveraging the MoE paradigm’s ability to allocate resources to specialized experts tailored for distinct physical environments. The proposed scheme features an innovative heterogeneous expert library, where the architecture of each expert is customized with embedded physical priors to match its specific propagation environment. To enable intelligent scheduling, we design a hybrid decision gating network that collaboratively leverages physical formula computation and data-driven deep learning to achieve accurate channel environment identification and expert routing. Furthermore, to overcome the expert collapse problem, we propose a three-stage training strategy—pretraining, freezing, and fine-tuning to ensure training stability specialization. Extensive simulations demonstrate that PP-MoE significantly outperforms traditional and deep learning baselines. Notably, in the low-SNR region (0–15 dB), it achieves an NMSE nearly an order of magnitude lower than LMMSE. Additionally, PP-MoE maintains high efficiency with only 0.0256 GFLOPs. This work provides an effective paradigm for designing adaptive and physically reliable wireless physical layers.
