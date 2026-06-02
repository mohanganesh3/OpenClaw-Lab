[2106.08846] Algorithm to Compilation Co-design: An Integrated View of Neural Network Sparsity

Algorithm to Compilation Co-design: An Integrated View of Neural Network Sparsity

Fu-Ming Guo

Fidelity Investments

fuming.guo@fmr.com

&amp;&amp;Austin Huang

Fidelity Investments

austinh@alum.mit.edu

Abstract

Reducing computation cost, inference latency, and memory footprint of neural networks are frequently cited as research motivations for pruning and sparsity. However, operationalizing those benefits and understanding the end-to-end effect of algorithm design and regularization on the runtime execution is not often examined in depth.

Here we apply structured and unstructured pruning to attention weights of transformer blocks of the BERT language model, while also expanding block sparse representation (BSR) operations in the TVM compiler. Integration of BSR operations enables the TVM runtime execution to leverage structured pattern sparsity induced by model regularization.

This integrated view of pruning algorithms enables us to study relationships between modeling decisions and their direct impact on sparsity-enhanced execution. Our main findings are: 1) we validate that performance benefits of structured sparsity block regularization must be enabled by the BSR augmentations to TVM, with 4x speedup relative to vanilla PyTorch and 2.2x speedup relative to standard TVM compilation (without expanded BSR support). 2) for BERT attention weights, the end-to-end optimal block sparsity shape in this CPU inference context is not a square block (as in  Gray et al. [ 2017 ] ) but rather a linear 32x1 block 3) the relationship between performance and block size / shape is is suggestive of how model regularization parameters interact with task scheduler optimizations resulting in the observed end-to-end performance.

1  Introduction

Capabilities of neural networks have accelerated in the last decade and that progress has been accompanied by a productive tension between two competing goals. One goal is to expand the boundaries of functionality and performance, which has been accompanied by increasing scale in data and compute. A second goal is for new capabilities to have broad impact and operationalization. This goal tends towards the opposite direction - shrinking down compute and data required to achieve a capability.

For example, expansion of data and compute has led to recent NLP advances showing how large language models have unprecedented generalization capabilities  [Brown et al.,  2020 , Raffel et al.,  2019 ] . These models should enable new realtime human-model interactions and entirely novel model development process where capabilities are instantiated at inference time, or can be rapidly adapted using lightweight methods such as prefix tuning  [Li and Liang,  2021 ] . However computational cost is an impediment to the impact and adoption of such models. How do we make these models accessible for both small and large scale research and deployment? Could such models be used in conjunction with privacy-preserving AI which requires model computation on edge devices? How can these language models be embedded at low cost into human-in-the-loop interactions requiring realtime latency?

One proposed answer to these questions has been the literature around sparsification and pruning of neural networks. Since the 1980s, we have known that it is usually possible to prune most parameters from trained neural networks without affecting accuracy  [LeCun et al.,  1990 ] .  Han et al. [ 2015 ]  reduced number of parameters of AlexNet  [Krizhevsky et al.,  2012 ]  by

9  ×

9\times

and VGG  [Simonyan and Zisserman,  2014 ]  by

13  ×

13\times

using connection pruning. The lottery ticket hypothesis was proposed by  Frankle and Carbin [ 2018 ] , which observes that a subnetwork of randomly-initialized network can replace the original network with the same performance.  Chen et al. [ 2020 ,  2021 ]  demonstrate the core LTH observations remain generally relevent in transformer models for both computer vision and natural language processing. Although current-generation CPUs and GPUs do not immediately benefit from sparsity, there is an active research area dedicated to writing libraries to
accelerate sparse neural networks on these platforms  [Elsen et al.,  2020 ]  and next generation hardware has native sparsity support (e.g., the NVIDIA A100, GraphCore IPU, and Cerebras Wafer-Scale Engine).

Although pruning is often motivated by performance, algorithms are often studied in isolation separate from their consequences with respect to compilation and execution.
However, interactions between model regularization choices, model compilation, and inference execution can have subtle-yet-critical effects on performance.

In this research, we implement both unstructured and structured sparsification of the attention weights of BERT alongside BSR sparsity optimizations in the TVM compiler  [Chen et al.,  2018 ] . We show how algorithms and compiler optimizations interact at different levels of the abstraction stack to determine end-to-end performance.

2  Methods

2.1  Structured Sparsification

Following the conventional pruning formulation, we consider the following optimization problem  [Han et al.,  2015 ] :

minimize  𝐰

∑

f  ​

(  𝐰  )

+

λ  ​

‖  𝐰  ‖

p

,

subscript

minimize

𝐰

𝑓  𝐰

𝜆

subscript

norm  𝐰

𝑝

\displaystyle\begin{array}[]{ll}\displaystyle\operatorname*{\text{minimize}}_{\mathbf{w}}&amp;\displaystyle\sum f(\mathbf{w})+\lambda\|\mathbf{w}\|_{p},\end{array}

(2)

where

‖  𝐰  ‖

norm  𝐰

\|\mathbf{w}\|

denotes the parameters of a neural network model,

‖  𝐰  ‖

p

subscript

norm  𝐰

𝑝

\|\mathbf{w}\|_{p}

denotes the

ℓ  p

subscript  ℓ  𝑝

\ell_{p}

norm of

𝐰

𝐰

\mathbf{w}

for

p  ∈

{  0  ,  1  }

𝑝

0  1

p\in\{0,1\}

. Note that equation

2

can be interpreted as the Lagrangian form of the problem:

minimize

𝐰  ∈

ℝ  d

f  0

​

(  𝐰  )

subject to

‖  𝐰  ‖

p

≤  τ

,

subscript

minimize

𝐰

superscript  ℝ  𝑑

subscript  𝑓  0

𝐰

subject to

subscript

norm  𝐰

𝑝

𝜏

\displaystyle\begin{array}[]{ll}\displaystyle\operatorname*{\text{minimize}}_{\mathbf{w}\in\mathbb{R}^{d}}&amp;\displaystyle f_{0}(\mathbf{w})\\
\operatorname*{\text{subject to}}&amp;\|\mathbf{w}\|_{p}\leq\tau,\end{array}

(5)

where

f  0

subscript  𝑓  0

f_{0}

is the pruning loss,

p  ∈

{  0  ,  1  }

𝑝

0  1

p\in\{0,1\}

, and

τ

𝜏

\tau

is the tolerance of nonzero weights. To obtain models with structured sparsity, we calculate our norm

‖  𝐰  ‖

p

subscript

norm  𝐰

𝑝

\|\mathbf{w}\|_{p}

in a structured group manner

‖  𝐰  ‖

p

=

∑

n  =  1

N

∑

b  =  1

B

‖

𝐰

b  ,  n

‖

p

subscript

norm  𝐰

𝑝

superscript

subscript

𝑛  1

𝑁

superscript

subscript

𝑏  1

𝐵

subscript

norm

subscript  𝐰

𝑏  𝑛

𝑝

\displaystyle\|\mathbf{w}\|_{p}=\sum_{n=1}^{N}\sum_{b=1}^{B}\|\mathbf{w}_{b,n}\|_{p}

(6)

A weight matrix or convolution kernel can be divided into blocks with sparsity determined by the outcome of the model optimization. Here

B

𝐵

B

is the block size and

N

𝑁

N

is the number of blocks that comprise the weight matrix or convolution kernel.

In contrast to the standard (unstructured)

ℓ  ​  1

ℓ  1

\ell 1

/ lasso procedure, group sparsity regularizes towards sparsity within each block, leading to a smaller set of more common used intra-block patterns, at least in the regime where

B

𝐵

B

is sufficiently small.

2.2  TVM Compiler Integration

Gray et al. [ 2017 ], Gale et al. [ 2020 ]  has shown the advantage of block sparsity in executing Transformer  [Vaswani et al.,  2017 ]  models on GPU.  Zhang et al. [ 2021 ], Gale et al. [ 2020 ]  demonstrates that the compiler scheduling introduces tremendous inference speed up on neural network.
We augmented the TVM compiler with the following additions to achieve inference speed up on sparse neural network:

•

We expand support for Block Sparse Ro