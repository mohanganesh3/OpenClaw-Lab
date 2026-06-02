# EV012: MoGERNN: An Inductive Traffic Predictor for Unobserved Locations in Dynamic Sensing Networks

URL: https://www.semanticscholar.org/paper/02b944584b13178dc15c9b489c0afea44df19f11
Year: 2025
Source: semantic_scholar
Arxiv: 2501.12281

## Abstract

Given a partially observed road network, how can we predict the traffic state of interested unobserved locations? Traffic prediction is crucial for advanced traffic management systems, with deep learning approaches showing exceptional performance. However, most existing approaches assume sensors are deployed at all locations of interest, which is impractical due to financial constraints. Furthermore, these methods are typically fragile to structural changes in sensing networks, which require costly retraining even for minor changes in sensor configuration. To address these challenges, we propose MoGERNN, an inductive spatio-temporal graph model with two key components: (i) a Mixture of Graph Experts (MoGE) with sparse gating mechanisms that dynamically route nodes to specialized graph aggregators, capturing heterogeneous spatial dependencies efficiently; (ii) a graph encoder-decoder architecture that leverages these embeddings to capture both spatial and temporal dependencies for comprehensive traffic state prediction. Experiments on two real-world datasets show MoGERNN consistently outperforms baseline methods for both observed and unobserved locations. MoGERNN can accurately predict congestion evolution even in areas without sensors, offering valuable information for traffic management. Moreover, MoGERNN is adaptable to the changes of sensor network, maintaining competitive performance even compared to its retrained counterpart. Tests performed with different numbers of available sensors confirm its consistent superiority, and ablation studies validate the effectiveness of its key modules. The code of this work is publicly available at: https://github.com/ZJU-TSELab/MoGERNN.
