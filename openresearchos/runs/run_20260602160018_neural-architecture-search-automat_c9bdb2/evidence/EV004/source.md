# EV004: GPNAS: A Neural Network Architecture Search Framework Based on Graphical Predictor

URL: https://www.semanticscholar.org/paper/00c999e82838358c48ec4ad2a6c2553c6e113acf
Year: 2021
Source: semantic_scholar
Arxiv: 2103.11820

## Abstract

In practice, the problems encountered in Neural Architecture Search (NAS) training are not simple problems, but often a series of difficult combinations (wrong compensation estimation, curse of dimension, overfitting, high complexity, etc.). In this paper, we propose a framework to decouple network structure from operator search space, and use two BOHBs to search alternatively. Considering that activation function and initialization are also important parts of neural network, the generalization ability of the model will be affected. We introduce an activation function and an initialization method domain, and add them into the operator search space to form a generalized search space, so as to improve the generalization ability of the child model. We then trained a GCN-based predictor using feedback from the child model. This can not only improve the search efficiency, but also solve the problem of dimension curse. Next, unlike other NAS studies, we used predictors to analyze the stability of different network structures. Finally, we applied our framework to neural structure search and achieved significant improvements on multiple datasets.
