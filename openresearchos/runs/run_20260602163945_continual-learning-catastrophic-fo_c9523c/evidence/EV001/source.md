# EV001: A Distributed Gaussian Process Model for Multi-Robot Mapping

URL: https://www.semanticscholar.org/paper/0001a985fc5b597ec8781ff78ca82f81d26669fe
Year: 2026
Source: semantic_scholar
Arxiv: 2603.07351

## Abstract

We propose DistGP: a multi-robot learning method for collaborative learning of a global function using only local experience and computation. We utilise a sparse Gaussian process (GP) model with a factorisation that mirrors the multi-robot structure of the task, and admits distributed training via Gaussian belief propagation (GBP). Our loopy model outperforms Tree-Structured GPs \cite{bui2014tree} and can be trained online and in settings with dynamic connectivity. We show that such distributed, asynchronous training can reach the same performance as a centralised, batch-trained model, albeit with slower convergence. Last, we compare to DiNNO \cite{yu2022dinno}, a distributed neural network (NN) optimiser, and find DistGP achieves superior accuracy, is more robust to sparse communication and is better able to learn continually.
