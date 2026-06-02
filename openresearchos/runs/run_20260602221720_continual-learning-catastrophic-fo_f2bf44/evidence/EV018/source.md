# EV018: HAMMER: Hamiltonian Curiosity Augmented Large Language Model Reinforcement

URL: https://www.semanticscholar.org/paper/00965f7fc184245a81e1308a25e9233fe7f94a82
Year: 2025
Source: semantic_scholar
Arxiv: 2509.25240

## Abstract

Recent curriculum reinforcement learning for large language models (LLMs) typically rely on difficulty-based annotations for data filtering and ordering. However, such methods suffer from local optimization, where continual training on simple samples in the early steps can cause the policy to lose its exploration. We propose a novel schema, namely Hamiltonian curiosity augmented large language model reinforcement (HAMMER), that transfers diversity metrics, commonly used in dataset evaluation, into the dynamic reinforcement learning procedure, where training samples are ordered via a minimum-semantic Hamiltonian path making the initial training retrain more exploration. From a theoretical perspective of generalization bounds, diversity-driven ordering facilitates stable convergence. Empirical evaluations indicate that HAMMER stimulates model"curiosity"and consistently achieves a 3% to 4% average accuracy gain across diverse inference benchmark.
