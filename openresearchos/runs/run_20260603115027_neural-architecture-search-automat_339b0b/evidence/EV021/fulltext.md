[2104.00031] NetAdaptV2: Efficient Neural Architecture Search with Fast Super-Network Training and Architecture Optimization

NetAdaptV2: Efficient Neural Architecture Search with

Fast Super-Network Training and Architecture Optimization

Tien-Ju Yang

Yi-Lun Liao

Vivienne Sze
 Massachusetts Institute of Technology

{tjy,ylliao,sze}@mit.edu

Abstract

Neural architecture search (NAS) typically consists of three main steps: training a super-network, training and evaluating sampled deep neural networks (DNNs), and training the discovered DNN. Most of the existing efforts speed up some steps at the cost of a significant slowdown of other steps or sacrificing the support of non-differentiable search metrics. The unbalanced reduction in the time spent per step limits the total search time reduction, and the inability to support non-differentiable search metrics limits the performance of discovered DNNs.

In this paper, we present NetAdaptV2 with three innovations to better balance the time spent for each step while supporting non-differentiable search metrics. First, we propose channel-level bypass connections that merge network depth and layer width into a single search dimension to reduce the time for training and evaluating sampled DNNs. Second, ordered dropout is proposed to train multiple DNNs in a single forward-backward pass to decrease the time for training a super-network. Third, we propose the multi-layer coordinate descent optimizer that considers the interplay of multiple layers in each iteration of optimization to improve the performance of discovered DNNs while supporting non-differentiable search metrics. With these innovations, NetAdaptV2 reduces the total search time by up to

5.8  ×

5.8\times

on ImageNet and

2.4  ×

2.4\times

on NYU Depth V2, respectively, and discovers DNNs with better accuracy-latency/accuracy-MAC trade-offs than state-of-the-art NAS works. Moreover, the discovered DNN outperforms NAS-discovered MobileNetV3 by 1.8% higher top-1 accuracy with the same latency.  1

1  1 The project website:  http://netadapt.mit.edu .

Figure 1 :

The comparison between NetAdaptV2 and related works. The number above a marker is the corresponding total search time measured on NVIDIA V100 GPUs.

1  Introduction

Neural architecture search (NAS) applies machine learning to automatically discover deep neural networks (DNNs) with better performance (e.g., better accuracy-latency trade-offs) by sampling the search space, which is the union of all discoverable DNNs. The search time is one key metric for NAS algorithms, which accounts for three steps: 1) training a  super-network , whose weights are shared by all the DNNs in the search space and trained by minimizing the loss across them, 2) training and evaluating sampled DNNs (referred to as  samples ), and 3) training the discovered DNN. Another important metric for NAS is whether it supports non-differentiable search metrics such as hardware metrics (e.g., latency and energy). Incorporating hardware metrics into NAS is the key to improving the performance of the discovered DNNs  [ 1 ,  2 ,  3 ,  4 ,  5 ] .

There is usually a trade-off between the time spent for the three steps and the support of non-differentiable search metrics. For example, early reinforcement-learning-based NAS methods  [ 6 ,  7 ,  2 ]  suffer from the long time for training and evaluating samples. Using a super-network  [ 8 ,  9 ,  10 ,  11 ,  12 ,  13 ,  14 ,  15 ,  16 ]  solves this problem, but super-network training is typically time-consuming and becomes the new time bottleneck. The gradient-based methods  [ 17 ,  18 ,  19 ,  20 ,  3 ,  21 ,  22 ,  23 ,  24 ]  reduce the time for training a super-network and training and evaluating samples at the cost of sacrificing the support of non-differentiable search metrics. In summary, many existing works either have an unbalanced reduction in the time spent per step (i.e., optimizing some steps at the cost of a significant increase in the time for other steps), which still leads to a long  total  search time, or are unable to support non-differentiable search metrics, which limits the performance of the discovered DNNs.

In this paper, we propose an efficient NAS algorithm, NetAdaptV2, to significantly reduce the  total  search time by introducing three innovations to  better balance  the reduction in the time spent per step while supporting non-differentiable search metrics:

Channel-level bypass connections (mainly reduce the time for training and evaluating samples, Sec.

2.2  ) : Early NAS works only search for DNNs with different numbers of filters (referred to as  layer widths ). To improve the performance of the discovered DNN, more recent works search for DNNs with different numbers of layers (referred to as  network depths ) in addition to different layer widths at the cost of training and evaluating more samples because network depths and layer widths are usually considered independently. In NetAdaptV2, we propose  channel-level bypass connections  to merge network depth and layer width into a single search dimension, which requires only searching for layer width and hence reduces the number of samples.

Ordered dropout (mainly reduces the time for training a super-network, Sec.

2.3  ) : We adopt the idea of super-network to reduce the time for training and evaluating samples. In previous works,  each  DNN in the search space requires one forward-backward pass to train. As a result, training multiple DNNs in the search space requires multiple forward-backward passes, which results in a long training time. To address the problem, we propose  ordered dropout  to jointly train multiple DNNs in a  single  forward-backward pass, which decreases the required number of forward-backward passes for a given number of DNNs and hence the time for training a super-network.

Multi-layer coordinate descent optimizer (mainly reduces the time for training and evaluating samples and supports non-differentiable search metrics, Sec.

2.4  ):  NetAdaptV1  [ 1 ]  and MobileNetV3  [ 25 ] , which utilizes NetAdaptV1, have demonstrated the effectiveness of the single-layer coordinate descent (SCD) optimizer  [ 26 ]  in discovering high-performance DNN architectures. The SCD optimizer supports both differentiable and non-differentiable search metrics and has only a few interpretable hyper-parameters that need to be tuned, such as the per-iteration resource reduction. However, there are two shortcomings of the SCD optimizer. First, it only considers one layer per optimization iteration. Failing to consider the joint effect of multiple layers may lead to a worse decision and hence sub-optimal performance. Second, the per-iteration resource reduction (e.g., latency reduction) is limited by the layer with the smallest resource consumption (e.g., latency). It may take a large number of iterations to search for a very deep network because the per-iteration resource reduction is relatively small compared with the network resource consumption. To address these shortcomings, we propose the  multi-layer coordinate descent (MCD) optimizer  that considers multiple layers per optimization iteration to improve performance while reducing search time and preserving the support of non-differentiable search metrics.

Fig.

1

(and Table

1  ) compares NetAdaptV2 with related works. NetAdaptV2 can reduce the search time by up to

5.8  ×

5.8\times

and

2.4  ×

2.4\times

on ImageNet  [ 27 ]  and NYU Depth V2  [ 28 ]  respectively and discover DNNs with better performance than state-of-the-art NAS works. Moreover, compared to NAS-discovered MobileNetV3  [ 25 ] , the discovered DNN has

1.8  %

percent  1.8

1.8\%

higher accuracy with the same latency.

2  Methodology: NetAdaptV2

2.1  Algorithm Overview

NetAdaptV2 searches for DNNs with different network depths, layer widths, and kernel sizes. The proposed  channel-level bypass connections (CBCs, Sec.

2.2  )  enables NetAdaptV2 to discover DNNs