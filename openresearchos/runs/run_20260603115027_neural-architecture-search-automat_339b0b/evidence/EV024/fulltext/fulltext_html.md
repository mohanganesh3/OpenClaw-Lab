[2103.11820] GPNAS: A Neural Network Architecture Search Framework Based on Graphical Predictor

GPNAS: A Neural Network Architecture Search Framework Based on Graphical Predictor

1 st  Dige Ai

dept. Renmin University of China

Beijing, China

aidg@stu.cpu.edu.com

2 nd  Hong Zhang

dept. Massachusetts Institute of Technology

Massachusetts, USA

lishanzai@gmail.com

Abstract

In practice, the problems encountered in Neural Architecture Search (NAS) training are not simple problems, but often a series of difficult combinations (wrong compensation estimation, curse of dimension, overfitting, high complexity, etc.). In this paper, we propose a framework to decouple network structure from operator search space, and use two BOHBs to search alternatively. Considering that activation function and initialization are also important parts of neural network, the generalization ability of the model will be affected. We introduce an activation function and an initialization method domain, and add them into the operator search space to form a generalized search space, so as to improve the generalization ability of the child model. We then trained a GCN-based predictor using feedback from the child model. This can not only improve the search efficiency, but also solve the problem of dimension curse. Next, unlike other NAS studies, we used predictors to analyze the stability of different network structures. Finally, we applied our framework to neural structure search and achieved significant improvements on multiple datasets.

Index Terms:

Deep Learning, Neural Architecture Search, Graph Convolutional Networks

I

Introduction

Due to the amazing characterization ability of deep neural networks (DNNs), it has achieved very good performance in various tasks, such as target detection  [ 34 ,  28 ,  33 ] , natural language processing  [ 17 ,  37 ,  12 ] , speech recognition  [ 18 ,  29 ] , face recognition  [ 11 ,  30 ,  8 ] , etc. Early neural network architecture searches were manual, using a multi-experiment approach to configure a good network by optimizing and adjusting different operators and their internal parameters.

In recent studies, NAS is considered to be a statistics-based model that can be optimized. Its goal is to find an optimal architecture that satisfies certain constraints, the most important of which is the search space. In general, the search space is coupled by a set of candidates for operator and skip-connection. Operators are used to implement linear and nonlinear transformations, or even splices, during data flow, such as max-pooling and 5*5 convolution operations. Skip-connections describe how operators are connected within a network. They transform an otherwise linear network structure into a complex topology.
Based on the above search space, many valuable works have been proposed by researchers to find better network architectures. Some studies have adopted evolutionary algorithms  [ 25 ,  41 ,  32 ,  26 ] , which require a large amount of computation. Later, reinforcement learning methods (RL-based methods) [ 31 ,  6 ,  7 ]  and gradient-based methods  [ 27 ,  7 ,  42 ,  40 ]  were designed, which reduced the calculation cost, but failed to solve the problem of credit allocation. Zela [ 45 ]  used BOHB to implement the search of neural network architecture, and they combined operators in a reasonable way to form a compound function. In fact, this is a kind of manual prior, which requires the engineer to have strong model design ability, which we believe defeates the original purpose of NAS design. Liu [ 27 ]  proposed a differentiable neural network architecture search model, which can directly optimize the searcher through gradient descent method. However, the searched cell structure will not be directly applied to the network required by the engineering. Instead, the cells in the tiny network will be transplanted to the larger network structure through an agent mechanism. In this way, although the search speed is improved, the training goal and evaluation goal of the searcher are not unified, which may lead to overfitting.

Jiang [ 20 ]  believed that the search paths of skip-connection and operator are different, and the search space of the two should not be coupled.Therefore, they decomposed the search space of skip-connection and operator, and designed an alternate search model based on RL, in which two controllers were used to search in the search space of both.However, their analysis of the search path of skip-connection has a large noise, and we believe that it is inefficient to search for skip-connection using RL. In order to catalyze search, Liu [ 25 ]  designed a predictor that can predict the performance of network structure, and adopted a progressive growth method to gradually increase the number of blocks. However, due to the limitations of the LSTM model, the predictor they designed can only predict the performance of child models without skip-connections, which greatly limits its versatility.

Based on the above analysis, we believe that the above methods are only upgraded or improved in a certain aspect. When applied to practical work, certain methods will always make unnecessary compromises in search due to one or more defects. This is because the previous work only implemented the fingers or eyes of the Transformers, and we wanted to go one step further and turn the fingers or eyes into arms and heads. In summary, our contributions are as follows:

•

We built an alternate neural network search framework to decouple the search space of skip-connection and operator. We abandoned the search framework of RL and adopted BOHB as the searcher to improve efficiency.

•

We designed a GCN-based(Graph Convolutional Network) predictor to predict performance with a Skip-Connection child model.

•

We design a generalized search space, which not only contains the traditional 13 candidate operators, but also adds the parameter initialization method and activation function of operator-level, and combines them in the form of Cartesian product.This expanded the search space, but still kept the search performance at a high level thanks to our ”dual accelerators” and decoupling Settings.

II

Related Work

Recent years, more and more studies have focused on neural network architecture search. NASnet  [ 48 ,  49 ]  applied reinforcement learning to NAS for the first time, and each searched network would be retrained and evaluated. ENAS [ 31 ]  solved the problem of retraining by using parameter sharing  [ 39 ]  for a class of pruning method, try to reduce the search space of redundancy  [ 27 ,  9 ]  attempts to weaken the discrete space for continuous space, and adopts the method based on gradient optimization search  [ 15 ,  45 ]  built a searcher based on bayesian method, and uses a hyperband method to optimize the allocation of resources. Most of the above studies are based on some method to improve, and they are often limited in practical application.

Our work draws on and improves upon previous work. Zoph [ 49 ]  proposed a basic operator search space on which much of the current work has been implemented. Jiang [ 20 ]  built an alternate search framework based on RL, decoupled the search space of skip-connection and operator, and added regularization on this basis to reduce network overfitting. Deng [ 10 ]  used an LSTM-based predictor to predict the performance of the sampled subnetworks, and built a coding system of operators for operator embedding. Falkner [ 15 ]  combined Bayesian optimization with Hyperband method to realize a highly robust neural network architecture searcher, BOHB. Our method combines the advantages of the above work and improves the disadvantages: alternating search and search space decoupled root can avoid the low search efficiency caused by different search paths. In order to further improve the efficiency, we replace the dual RL architecture proposed by Jiang [ 20 ]  with two BOHBs. Predictor can