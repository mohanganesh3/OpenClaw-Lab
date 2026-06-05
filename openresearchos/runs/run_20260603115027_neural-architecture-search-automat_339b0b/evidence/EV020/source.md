# EV020: Automated Design Space Exploration for Optimized Deployment of DNN on Arm Cortex-A CPUs

URL: https://www.semanticscholar.org/paper/00b0452cc6cefc7a288042c5979fcc60b71c1e58
Year: 2020
Source: semantic_scholar
Arxiv: 2006.05181

## Abstract

The spread of deep learning on embedded devices has prompted the development of numerous methods to optimize the deployment of deep neural networks (DNNs). Works have mainly focused on: 1) efficient DNN architectures; 2) network optimization techniques, such as pruning and quantization; 3) optimized algorithms to speed up the execution of the most computational intensive layers; and 4) dedicated hardware to accelerate the data flow and computation. However, there is a lack of research on cross-level optimization as the space of approaches becomes too large to test and obtain a globally optimized solution. Thus, leading to suboptimal deployment in terms of latency, accuracy, and memory. In this work, we first detail and analyze the methods to improve the deployment of DNNs across the different levels of software optimization. Building on this knowledge, we present an automated exploration framework to ease the deployment of DNNs. The framework relies on a reinforcement learning search that, combined with a deep learning inference framework, automatically explores the design space and learns an optimized solution that speeds up the performance and reduces the memory on embedded CPU platforms. Thus, we present a set of results for state-of-the-art DNNs on a range of Arm Cortex-A CPU platforms achieving up to $4\times $ improvement in performance and over $2\times $ reduction in memory with negligible loss in accuracy with respect to the BLAS floating-point implementation.
