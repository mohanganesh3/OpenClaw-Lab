[2603.07351] A Distributed Gaussian Process Model for Multi-Robot Mapping

A Distributed Gaussian Process Model for Multi-Robot Mapping

Seth Nabarro  1∗  , Mark van der Wilk  2

and Andrew J. Davison  1

1  Dyson Robotics Lab, Imperial College London  2  Department of Computer Science, University of Oxford ∗  sdn09@ic.ac.uk

Abstract

We propose  DistGP : a multi-robot learning method for collaborative learning of a global function using only local experience and computation. We utilise a sparse Gaussian process (GP) model with a factorisation that mirrors the multi-robot structure of the task, and admits distributed training via Gaussian belief propagation (GBP). Our loopy model outperforms Tree-Structured GPs  [ 4 ]  and can be trained online and in settings with dynamic connectivity. We show that such distributed, asynchronous training can reach the same performance as a centralised, batch-trained model, albeit with slower convergence. Last, we compare to DiNNO  [ 22 ] , a distributed neural network (NN) optimiser, and find DistGP achieves superior accuracy, is more robust to sparse communication and is better able to learn continually.

I

Introduction

(a)

Before inter-robot exchange

(b)

After inter-robot exchange

Figure 1 :

In  DistGP , each robot maintains a block of inducing points which parameterise their local GP (far-left) and into which observations can be fused. Communicating robots harmonise models by connecting inducing points via GP consistency factors and exchanging GBP messages (far-right). Here, the map at each point is predicted by the closest robot. Before communication (centre-left) the map is discontinuous and inaccurate, but this is resolved using only local inter-robot exchanges (centre-right).

Many mapping tasks involve large areas and dynamic maps, suggesting a divide-and-conquer approach using multiple robots. Centralised communication and computation is often not feasible in domains like environmental monitoring  [ 9 ] , robotic agriculture  [ 1 ]  or space exploration  [ 21 ] , which involve many robots accumulating large datasets in poorly connected regions. Further, centralised approaches can be slow where robots may need up-to-date local models to guide exploration. These considerations motivate decentralised approaches, with local computation and mesh connectivity.

In this work, we present DistGP: a multi-robot algorithm to fit an arbitrary global function using only local observation and computation; with peer-to-peer communication. We use sparse GPs whose inducing points approximate a full GP at lower cost, and act as local summaries of the function. They provide a useful abstraction for data fusion and inter-robot exchange. Further, they are assumed to be Gaussian distributed, enabling decentralised inference with GBP.

We first consider tree-structured GPs (TSGPs)  [ 4 ] , whose inducing points are partitioned into sparsely connected blocks. Each block covers a distinct region of the input space and is maintained by a single robot. Robots exchange GBP messages with their neighbours to ensure their inducing points are consistent. TSGP restricts connectivity to a tree to guarantee rapid convergence, however we show empirically that this constraint limits accuracy: inducing points close in Euclidean space may not be directly connected, leading to discontinuities at their boundary. Further, in dynamic multi-robot scenarios, two communicating robots are prevented from sharing information if doing so would introduce a cycle.

To address this limitation, we propose a simple extension: relax the tree constraint and allow cycles. While GBP in our loopy model requires more iterations to converge, we demonstrate a substantial improvement in accuracy. Moreover, our approach is better suited to dynamic and asynchronous settings, where inter-robot communication patterns are continually changing – unsupported by TSGP, whose message schedule requires fixed connectivity. Our loopy model naturally exploits all robot encounters and converges with asynchronous message passing.
Our proposed loopy factor graph model is illustrated in Fig.

1  .

Our motivations are similar to Yu et al.  [ 22 ] , who use a distributed NN optimiser (DiNNO) for multi-robot learning.
In contrast to DiNNO, DistGP i) enforces inter-robot consistency locally, only where both robots have inducing points, rather than in global weight-space, ii) is a probabilistic model supporting on-the-fly data fusion for online learning, and iii) is scalable as each robot stores only a local map estimate.

In short our contributions are as follows:

1.

A new distributed GP model, trainable with GBP, which outperforms TSGPs  [ 4 ] .

2.

Empirical analysis showing the model can be built on-the-fly and admits distributed, asynchronous inference via GBP. It is thus well-suited to multi-robot mapping.

3.

Experimental evaluation of our method on environmental monitoring and occupancy mapping simulations. We show superior performance to DiNNO  [ 22 ] .

II

Related Work

Distributed NN Maps

Many works build distributed implicit neural maps through inter-robot collaboration  [ 22 ,  8 ,  2 ,  23 ,  24 ] , either using consensus alternating direction method of multipliers (CADMM)  [ 3 ]  or direct regularisation to constrain NN weights to be similar. DiNNO  [ 22 ]  is most similar to ours as it is a general method for multi-robot function learning rather than a specific method for distributed

3  3

D reconstruction. While distributed neural maps enforce agreement in global weight space, we represent maps with factorised GPs which enable local consistency constraints, specific to regions in which both robots have inducing points.

In DiNNO, each robot maintains a NN estimate of the global map, which are brought into agreement by CADMM. In this work however, we seek to build a distributed map: each robot maintains their local segment of the map and inter-robot interactions are used to ensure each segment is consistent with its neighbours. We argue DistGP is:

1.

more scalable.  For problems with large areas and large numbers of robots, each robot having a global map is inefficient and reaching consensus is more challenging.

2.

more modular.  We divide the map into segments which can be individually copied and communicated as required. In contrast, it is unclear how a map stored in the weights of a NN can be subdivided.

3.

also able to generate a shared global map.  In the extreme case that each robot shares its own inducing point posterior, and all others it has received thus far, we can achieve global map sharing as in DiNNO.

Multi-Robot GPs

As in this work,  [ 13 ]  use a distributed GP for multi-robot learning. Each robot first trains a local GP on its observations and they are combined as a mixture of GPs, with location-varying weights. The Expectation Maximisation algorithm optimises the local models in tandem with the mixture weights of the different observations. Unlike our method, full connectivity is needed for normalised mixture weights, and  all  robots must generate a prediction at any test point. In addition, the use of dense GPs precludes scaling to large datasets.  [ 11 ]  present a multi-agent inducing point GP method for human trajectory prediction, however they do not consider the problem of distributed learning of a function over the space being navigated.

Multi-Robot GBP

Recent work has shown GBP to be an effective distributed solver. Our work draws inspiration from  [ 14 ] , in using GBP exchange between robots to enable collaborative estimation. However,  [ 14 ]  estimate the robot locations, where we estimate a distributed map. Like us,  [ 16 ]  utilise GBP for multi-robot mapping. Each robot maintains a grid of the input space with one variable per cell to represent its function value. These are updated via i) observation by the robot or ii) inter-robot communication where two robots enforce consistency between their pairs of grid vari