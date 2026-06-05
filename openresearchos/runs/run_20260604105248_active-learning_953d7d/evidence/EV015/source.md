# EV015: Graph-Based LSTM for Anti-money Laundering: Experimenting Temporal Graph Convolutional Network with Bitcoin Data

URL: https://www.semanticscholar.org/paper/005068f0ec70e22950830230e4bd1868e430a8cd
Year: 2022
Source: semantic_scholar
Arxiv: n/a

## Abstract

Elliptic data—one of the largest Bitcoin transaction graphs—has admitted promising results in many studies using classical supervised learning and graph convolutional network models for anti-money laundering. Despite the promising results provided by these studies, only few have considered the temporal information of this dataset, wherein the results were not very satisfactory. Moreover, there is very sparse existing literature that applies active learning to this type of blockchain dataset. In this paper, we develop a classification model that combines long-short-term memory with GCN—referred to as temporal-GCN—that classifies the illicit transactions of Elliptic data using its transaction’s features only. Subsequently, we present an active learning framework applied to the large-scale Bitcoin transaction graph dataset, unlike previous studies on this dataset. Uncertainties for active learning are obtained using Monte-Carlo dropout (MC-dropout) and Monte-Carlo based adversarial attack (MC-AA) which are Bayesian approximations. Active learning frameworks with these methods are compared using various acquisition functions that appeared in the literature. To the best of our knowledge, MC-AA method is the first time to be examined in the context of active learning. Our main finding is that temporal-GCN model has attained significant success in comparison to the previous studies with the same experimental settings on the same dataset. Moreover, we evaluate the performance of the provided acquisition functions using MC-AA and MC-dropout and compare the result against the baseline random sampling model.
