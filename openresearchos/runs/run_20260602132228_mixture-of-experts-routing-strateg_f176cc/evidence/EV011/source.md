# EV011: Deep heterogeneity learning for cross-city transit forecasting: a differentially private federated framework with mixture-of-experts and seasonal decomposition

URL: https://www.semanticscholar.org/paper/30c165f88a7f30594cbec85588dc7a76cd1b53ab
Year: 2026
Source: semantic_scholar
Arxiv: n/a

## Abstract


 
 Accurate prediction of transit flows is fundamental to optimizing intelligent transportation systems; however, centralized forecasting is frequently obstructed by heterogeneous, Non-Independent and Identically Distributed (Non-IID) cross-city data and stringent data privacy regulations.
 
 
 
 We propose X-FedFormer, a novel framework integrating Federated Learning (FL) with Differential Privacy (DP) and a deep learning architecture combining a Mixture-of-Experts (MoE) mechanism with a Seasonal-Trend Decomposition module. The framework is evaluated on a statistically validated synthetic dataset faithfully simulating realistic inflow and outflow patterns across ten diverse urban environments (90 days of hourly records, 30 routes per city, 64,800 observations per city).
 
 
 
 X-FedFormer significantly outperforms state-of-the-art federated baselines including FedProx, achieving an aggregate coefficient of determination of 0.922 and a mean absolute error (MAE) of 7.93 passengers across all participating cities. A Wilcoxon signed-rank test confirms statistical significance over the strongest baseline (p = 0.018). Ablation studies confirm that the MoE and seasonal decomposition modules reduce forecasting error by approximately 11% and 16%, respectively, compared to standard architectures.
 
 
 
 The model maintains high predictive utility even under strict differential privacy guarantees (ε ≈ 2), establishing a viable privacy-utility operating point for practical deployment. These findings present a scalable, robust solution for urban computing that effectively balances algorithmic performance with data sovereignty in smart city applications.

