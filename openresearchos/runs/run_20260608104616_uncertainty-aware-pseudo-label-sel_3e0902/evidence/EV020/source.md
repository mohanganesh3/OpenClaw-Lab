# EV020: FedDSSL: Decentralized Federated Semi-Supervised Learning for Limitedly Annotated Data

URL: https://www.semanticscholar.org/paper/01894356b747fe0968f797e48edb928b4576f451
Year: 2025
Source: semantic_scholar
Arxiv: n/a

## Abstract

Federated Semi-Supervised Learning (FSSL) integrates Semi-Supervised Learning (SSL) with the federated framework, enabling clients to collaboratively train a global model using their local labeled and unlabeled data while preserving data privacy. Existing methods (e.g., FedMatch, FedSSL) rely on a central server to aggregate model parameters and utilize a centralized proxy dataset to guide the training process. However, these approaches face privacy risks, cause negative transfer due to data heterogeneity, and violate the lightweight design principle of federated learning. This paper proposes a decentralized framework, FedDSSL, which employs a dynamic topology structure to adaptively adjust the connection weights among clients, thereby enhancing collaborative efficiency. FedDSSL adopts a topology graph, where connection weights between clients are adaptively adjusted based on data similarity, enabling more efficient collaborative training. To replace the centralized proxy dataset, FedDSSL utilizes local self-supervised pre-training and crossclient knowledge distillation for regularization alignment. Additionally, FedDSSL introduces a distributed optimization strategy, employing multi-client collaborative validation and dynamic consistency regularization to improve the quality of pseudo-labels and model robustness. Experimental results demonstrate that FedDSSL outperforms mainstream methods in both IID and non-IID scenarios. It provides an efficient and lightweight solution for privacy-sensitive fields such as healthcare, significantly enhancing model robustness and generalization ability.
