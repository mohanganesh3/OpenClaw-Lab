# EV005: Adaptive-expert-weight-based load balance scheme for dynamic routing of MoE

URL: https://www.semanticscholar.org/paper/1a9e1eec225416bacacc74349c3f5f62f88a0668
Year: 2025
Source: semantic_scholar
Arxiv: n/a

## Abstract

Load imbalance is a major performance bottleneck in training mixture-of-experts (MoE) models, as unbalanced expert loads can lead to routing collapse. Most existing approaches address this issue by introducing auxiliary loss functions to balance the load; however, the hyperparameters within these loss functions often need to be tuned for different tasks. Furthermore, increasing the number of activated experts tends to exacerbate load imbalance, while fixing the activation count can reduce the model’s confidence in handling difficult tasks. To address these challenges, this paper proposes a dynamically balanced routing strategy that employs a threshold-based dynamic routing algorithm. After each routing step, the method adjusts expert weights to influence the load distribution in the subsequent routing. Unlike loss-function-based balancing methods, our approach operates directly at the routing level, avoiding gradient perturbations that could degrade model quality, while dynamically routing to make more efficient use of computational resources. Experiments on Natural Language Understanding (NLU) benchmarks demonstrate that the proposed method achieves accuracy comparable to top-2 routing, while significantly reducing the load standard deviation (e.g., from 12.25 to 1.18 on MNLI). In addition, threshold-based dynamic expert activation reduces model parameters and provides a new perspective for mitigating load imbalance among experts.
