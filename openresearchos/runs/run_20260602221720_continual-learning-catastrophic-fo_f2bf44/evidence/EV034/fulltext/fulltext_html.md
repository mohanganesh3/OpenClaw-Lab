[2505.19680] Cut out and Replay: A Simple yet Versatile Strategy for Multi-Label Online Continual Learning

Cut out and Replay: A Simple yet Versatile Strategy for Multi-Label Online Continual Learning

Xinrui Wang

Shao-yuan Li

Jiaqiang Zhang

Songcan Chen

Abstract

Multi-Label Online Continual Learning (MOCL) requires models to learn continuously from endless multi-label data streams, facing complex challenges including persistent catastrophic forgetting, potential missing labels, and uncontrollable imbalanced class distributions. While existing MOCL methods attempt to address these challenges through various techniques,  they all overlook label-specific region identifying and feature learning  - a fundamental solution rooted in multi-label learning but challenging to achieve in the online setting with incremental and partial supervision. To this end, we first leverage the inherent structural information of input data to evaluate and verify the innate localization capability of different pre-trained models. Then, we propose CUTER (CUT-out-and-Experience-Replay), a simple yet versatile strategy that provides fine-grained supervision signals by further identifying, strengthening and cutting out label-specific regions for efficient experience replay. It not only enables models to simultaneously address catastrophic forgetting, missing labels, and class imbalance challenges, but also serves as an orthogonal solution that seamlessly integrates with existing approaches. Extensive experiments on multiple multi-label image benchmarks demonstrate the superiority of our proposed method. The code is available at  https://github.com/wxr99/Cut-Replay

1  Introduction

Online continual learning (OCL) enables models to learn from continuous, endless data streams. Significant progress has been made in this area through various techniques to mitigate catastrophic forgetting, such as knowledge distillation, gradient regularization, and experience replay.

Figure 1:  Two unique challenges in MOCL compared with traditional OCL: (1) Massive missing past and future labels in both coming data stream and memory buffer. (2) Severe class imbalance that persists in the memory buffer even with re-balancing strategies like CEBS (Wei &amp; Li,  2019a ; Yan et al.,  2021 ) .

However, these OCL approaches focus primarily on single-label classification, while real-world data often exhibit multiple semantic concepts and objects, motivating the study of Multi-Label Online Continual Learning (MOCL)  (Kim et al.,  2020 ) . Figure

1

shows an illustrative example. At each time step

t  t

, the training data

D  t

D_{t}

concerning label space

Y  t

{Y}_{t}

for task

t  t

arrive in a streaming manner. For each example

(  x  ,  y  )

∈

D  t

(x,y)\in D_{t}

,

y  ⊂

Y  t

y\subset{Y}_{t}

denotes the tagged relevant label set. Given a sequence of

T  T

tasks, the objective of MOCL is to efficiently adapt the learning model to the current task while preventing catastrophic forgetting of previously learned tasks.

Compared to single-label scenarios, MOCL faces two specific data challenges: (1) Pervasive missing labels: samples in task

t  t

are only annotated with labels from

Y  t

Y_{t}

, even when containing objects from old classes

𝒴

1  :

t  −  1

\mathcal{Y}_{1:t-1}

or future classes

𝒴

t  +  1

:  T

\mathcal{Y}_{t+1:T}

. These unlabeled classes become  false negatives , aggravating catastrophic forgetting. (2) Uncontrollable imbalanced classes: the categories
often follow a long-tailed distribution. Training on such data would lead the model to be biased towards overfitting the head classes and underfitting the tail classes. The  co-occurrence  of head and tail classes within one sample further complicates this issue.

Several works have attempted to address these challenges.  (Kim et al.,  2020 )  first identified the severe forgetting of minority classes in long-tailed scenarios and proposed a Partitioning Reservoir Sampling (PRS) strategy to balance head and tail classes in replay-based approaches. To address the computational inefficiency of PRS,  (Liang &amp; Li,  2022 )  developed Optimizing Class Distribution in Memory (OCDM), which reformulates memory updates as a sample selection optimization problem solvable through a linear-time greedy algorithm. The challenge of missing labels in MOCL was first highlighted by  (Du et al.,  2022 ) , which introduced an Augmented Graph Convolutional Network (AGCN). This model generates predictions for previously seen classes while modeling dynamic label relationships across sequential tasks and mitigating forgetting through distillation and relationship-preserving losses. Building on this,  (Dong et al.,  2023 )  proposed the Knowledge Restore and Transfer (KRT) framework, which combines dynamic pseudo-labeling for old classes with session-specific knowledge transfer. More recently,  (Du et al.,  2024 )  tackled both challenges simultaneously through two key components: Asymmetric Knowledge Distillation (AKD) and Online Relabeling (OR). AKD rebalances the learning process by emphasizing negative label learning in classification loss while reducing the impact of overconfident predictions in distillation loss. OR complements this by recovering missing labels in the memory buffer through online relabeling.

Although these works demonstrate promising results, their feature learning mechanisms have inherent limitations with multi-label data. The conventional approach of extracting a single feature vector per example, along with techniques like pseudo-labeling and resampling, suffers from co-occurrence bias between head and tail classes. This limitation is particularly critical with missing labels and class imbalance, where discriminative feature learning requires minimizing interference from label co-occurrence patterns.

With the above understanding, we resort to label-specific feature learning, which was shown to be superior to unified sample-wise features in offline multi-label learning  (Zhang &amp; Wu,  2014 ) . Suppose we can successfully identify label-specific regions in images through a straightforward cut-out-and-replay mechanism, i.e., cutting out these regions and storing them in the memory buffer for replay. This mechanism would naturally avoid label co-occurrence interference and missing label issues. Furthermore, with object-level regions and supervision signals stored in the buffer, it enables more effective experience replay under the same memory overhead, where class imbalance is easily addressed by controlling class distributions in the buffer.

To this end, we propose CUTER (CUT-out-and-Experience-Replay), a simple yet versatile approach that efficiently localizes, strengthens, and cuts out label-specific regions for MOCL with richer fine-grained supervision signals. Motivated by the recent inspiring trials on vision pre-trained models’ ability for zero-shot coarse segmentation  (Caron et al.,  2021 ; Siméoni et al.,  2021 ; Wang et al.,  2023b ) , we make a thorough study of the pre-trained modes’ localization capability. Through extensive empirical validations over several widely used pre-trained models (e.g., DINO, MoCo, MAE), and established theoretical supports from graph theory, we show that the averaged Fiedler value (second smallest eigenvalue of graph Laplacian) of the feature patch similarity graph can serve as a valid evaluation measure for selecting pre-trained models good at localizing precise label-specific regions. The identified salient regions are then selectively stored in a memory buffer based on their prediction confidence and label alignment, effectively transforming multi-label image classification replay into multiple single-label sub-image classification tasks. To further combat the forgetting impact of the model’s localization ability during the continual learning process, we draw inspiration from image segmentation principles and incorporate a low-rank constraint on the fea