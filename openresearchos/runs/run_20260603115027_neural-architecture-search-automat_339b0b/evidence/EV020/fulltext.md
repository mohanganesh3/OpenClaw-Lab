[2006.05181] Automated Design Space Exploration for optimised Deployment of DNN on Arm Cortex-A CPUs

Automated Design Space Exploration for optimised Deployment of DNN on Arm Cortex-A CPUs

Miguel de Prado

miguel.deprado@he-arch.ch

Andrew Mundy
3

andrew.mundy@arm.com

Rabia Saeed
1

Rabia.Saeed@he-arc.ch

Maurizo Denna
4

maurizio.denna@nviso.ch

Nuria Pazos
1

Nuria.Pazos@he-arc.ch

Luca Benini
2

lbenini@iis.ee.ethz.ch

1 He-Arc Ingenierie, HES-SO
2Integrated System Lab, ETH Zurich
3Arm Ltd.
4Nviso

Abstract

The spread of deep learning on embedded devices has prompted the development of numerous methods to optimise the deployment of deep neural networks (DNN). Works have mainly focused on:  i)  efficient DNN architectures,  ii)  network optimisation techniques such as pruning and quantisation,  iii)  optimised algorithms to speed up the execution of the most computational intensive layers and,  iv)  dedicated hardware to accelerate the data flow and computation. However, there is a lack of research on cross-level optimisation as the space of approaches becomes too large to test and obtain a globally optimised solution. Thus, leading to suboptimal deployment in terms of latency, accuracy, and memory.

In this work, we first detail and analyse the methods to improve the deployment of DNNs across the different levels of software optimisation. Building on this knowledge, we present an automated exploration framework to ease the deployment of DNNs. The framework relies on a Reinforcement Learning search that, combined with a deep learning inference framework, automatically explores the design space and learns an optimised solution that speeds up the performance and reduces the memory on embedded CPU platforms. Thus, we present a set of results for state-of-the-art DNNs on a range of Arm Cortex-A CPU platforms achieving up to

4  ×

4\times

improvement in performance and over

2  ×

2\times

reduction in memory with negligible loss in accuracy with respect to the BLAS floating-point implementation.

I

Introduction

Deep learning has grown quickly in the last years, achieving remarkable results in computer vision  [ 1 ,  2 ,  3 ]  and speech recognition  [ 4 ,  5 ] . Recently, the focus has shifted towards the deployment of such DNNs on resource-constrained devices to make the myriad of devices on the edge intelligent. In contrast to cloud environments, edge devices are often severely constrained in terms of computing power, memory, and energy consumption, which is available to a given application. These constraints hamper deployment of deep learning solutions to edge devices and require innovation in the design of deep learning systems (or neural network architectures), and in the software which executes them.

Figure 1:

Optimisation categories  for the deployment of DNNs at different levels of the software stack.

Numerous research works have focused on optimizing the deployment of DNN through the development of  i)  efficient DNN architectures such as MobileNets  [ 2 ] , SqueezeNet  [ 6 ] , including hardware-aware neural architecture search (NAS)  [ 7 ,  8 ] ,  ii)  optimisation techniques such as pruning and quantisation  [ 9 ,  10 ,  11 ,  12 ] ,  iii)  optimised algorithms to speedup the execution of the most computational layers, e.g., general matrix multiplication (GEMM)  [ 13 ]  or Winograd  [ 14 ]  and,  iv)  dedicated hardware to accelerate the data flow and parallelise the computation  [ 15 ,  16 ,  17 ] .

There is, however, a lack of research on cross-level optimisation and design space exploration (DSE) for a complete end-to-end solution  [ 18 ] . The space of approaches for DNN deployment becomes too large to fully explore and obtain an optimal implementation as each layer of the neural network may be executed following a different optimisation technique, algorithm, or even in a different processor, resulting in a different performance, memory footprint or power consumption. The complexity of exploring and combining the wide variety of design options usually results in a sub-optimal deployment solution  [ 19 ] .

The objective of this work is to ease the deployment of pre-trained DNNs for industrial applications by automatically exploring the design space and finding an optimised solution to speed up the performance and reduce the memory on embedded CPU platforms. To that end, we employ LPDNN  [ 20 ] , a deep learning framework that enables the deployment and inference of DNN implementations. We focus on software optimisation for the deployment on Arm CPU cores as these represent the majority of processors on mobile and IoT devices and have extensive support for DNN inference  [ 21 ] . Our work is complementary to DNN architecture design for further deployment optimisation and could also be applied to dedicated hardware. Our contributions are the following:

•

We analyse methods to improve the deployment of DNNs across different levels of software optimisation (Section

II  ) and introduce the range of techniques provided by LPDNN to optimise DNN inference (Section

III  ).

•

We present QS-DNN (Section

V  ), an automatic exploration framework based on Reinforcement Learning (RL), that, combined with LPDNN, finds an optimised combination of design options that speeds up DNN inference and reduces memory for a target platform.

•

We present a set of results for state-of-the-art DNNs on a wide range of Arm Cortex-A CPU platforms that cover the current spectrum of the deployment on mobile devices (Section

VI  ). Finally, we show a comparison between RL and several other search algorithms when evaluated on our deployment design space.

Next, we present the deployment optimisation of DNNs across different levels of the software stack.

II

Background: Deployment Optimisation of Deep Neural Networks

Given the constraints imposed on edge devices, namely, relatively limited compute performance, small memory capacities, and thermal and power consumption restrictions, there are several goals for which one may choose to optimise. For example, one might decide to sacrifice neural network inference latency to reduce overall power consumption, or to stay within a more limited memory capacity. Depending on the goal of the optimisation, neural networks present a range of software optimisation opportunities. We divide these opportunities into several broad categories as shown in Fig.

1  :

II-A

Network Design

We define network design optimisation to be the set of techniques that tailor the structure of a network before, or during training, to improve the latency or cost of network inference. Examples of this are MobileNet-V1/V2  [ 2 ,  22 ] , SqueezeNet  [ 6 ]  and, ShuffleNet  [ 23 ]  which were manually shaped thanks to the expertise of the authors. A set of newer works introduced the neural architecture search (NAS) as a technique to reduce the high-level human knowledge needed for the conception of such architectures. Examples of this are MNASnet  [ 24 ] , FbNet  [ 8 ] , and Lemonade  [ 25 ]  which used hardware-aware NAS via reinforcement learning, evolutionary algorithms or gradient-based methods to discover neural network structures with both good accuracy and high performance.

Distillation is another technique where a neural network teacher can transfer its learned knowledge to student networks. Students are constructed to present lower computational complexity and memory cost than their teachers. The student imitates the teacher over a training dataset and obtains high accuracy while reducing the complexity of the neural network. Works implementing this technique are  [ 26 ,  27 ,  28 ] .

II-B

Network Optimisation

In contrast to neural network design, network optimisation techniques take an existing network and modify the way it is represented, for example by exploiting lower-precision data types (quantisation), the inherent sparsity of weights and activations (pruning) or fusion of linear operations (layer fusion). Techni