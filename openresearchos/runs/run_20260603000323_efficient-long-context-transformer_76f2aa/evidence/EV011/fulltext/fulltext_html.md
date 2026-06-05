[2310.01889] Ring Attention with Blockwise Transformers for Near-Infinite Context

Ring Attention with Blockwise

Transformers for Near-Infinite Context

Hao Liu, Matei Zaharia, Pieter Abbeel

UC Berkeley

hao.liu@cs.berkeley.edu

Abstract

Transformers have emerged as the architecture of choice for many state-of-the-art AI models, showcasing exceptional performance across a wide range of AI applications. However, the memory demands imposed by Transformers limit their ability to handle long sequences, thereby posing challenges in utilizing videos, actions, and other long-form sequences and modalities in complex environments. We present a novel approach, Ring Attention with Blockwise Transformers (Ring Attention), which leverages blockwise computation of self-attention and feedforward to distribute long sequences across multiple devices while fully overlapping the communication of key-value blocks with the computation of blockwise attention. Our approach enables training and inference of sequences that are up to device count times longer than those achievable by prior memory-efficient Transformers, without resorting to approximations or incurring additional communication and computation overheads.
Extensive experiments on language modeling and reinforcement learning tasks demonstrate the effectiveness of our approach in allowing millions of tokens context size and improving performance.

1

1  1  Code:  https://github.com/lhao499/llm_large_context

.

1  Introduction

Transformers  [ 37 ]  have become the backbone of many state-of-the-art AI systems that have demonstrated impressive performance across a wide range of AI problems.
Transformers achieve this success through their architecture design that uses self-attention and position-wise feedforward mechanisms.
However, scaling up the context length of Transformers is a challenge  [ 29 ] , since the inherited architecture design of Transformers,  i . e . the self-attention has memory cost quadratic in the input sequence length,
which makes it challenging to scale to longer input sequences.
Large context Transformers are essential for
tackling a diverse array of AI challenges, ranging from processing books and high-resolution images to analyzing long videos and complex codebases.
They excel at extracting information from the interconnected web and hyperlinked content, and are crucial for handling complex scientific experiment data.
There have been emerging use cases of language models with significantly expanded context than before: GPT-3.5  [ 32 ]  with context length 16K, GPT-4  [ 29 ]  with context length 32k, MosaicML’s MPT  [ 25 ]  with context length 65k, and Anthropic’s Claude  [ 1 ]  with context length 100k.

Driven by the significance, there has been surging research interests in reducing memory cost. One line of research leverages the observation that the softmax matrix in self-attention can be computed without materializing the full matrix  [ 24 ]  which has led to the development of blockwise computation of self-attention and feedforward  [ 30 ,  9 ,  23 ]  without making approximations.
Despite the reduced memory, a significant challenge still arises from storing the output of each layer.
This necessity arises from self-attention’s inherent nature, involving interactions among all elements (n to n interactions). The subsequent layer’s self-attention relies on accessing all of the prior layer’s outputs. Failing to do so would increase computational costs cubically, as every output must be recomputed for each sequence element, rendering it impractical for longer sequences.

Figure 1:

Maximum context length under end-to-end large-scale training on TPUv4-1024.
Baselines are vanilla transformers  [ 37 ] , memory efficient transformers  [ 30 ] , and memory efficient attention and feedforward (blockwise parallel transformers)  [ 23 ] .
Our proposed approach Ring Attention allows training up to device count times longer sequence than baselines and enables the training of sequences that exceed millions in length without making approximations nor adding any overheads to communication and computation.

These components facilitate the efficient capture of long-range dependencies between input tokens, and enable scalability through highly parallel computations.
To put the memory demand in perspective, even when dealing with a batch size of 1, processing 100 million tokens requires over 1000GB of memory for a modest model with a hidden size of 1024. This is much greater than the capacity of contemporary GPUs and TPUs, which typically have less than 100GB of high-bandwidth memory (HBM).

To tackle this challenge, we make a key observation: by performing self-attention and feedforward network computations in a blockwise fashion  [ 23 ] , we can distribute sequence dimensions across multiple devices, allowing concurrent computation and communication.
This insight stems from the fact that when we compute the attention on a block-by-block basis, the results are invariant to the ordering of these blockwise computations.
Our method distributes the outer loop of computing blockwise attention among hosts, with each device managing its respective input block. For the inner loop, every device computes blockwise attention and feedforward operations specific to its designated input block.
Host devices form a conceptual ring, where during the inner loop, each device sends a copy of its key-value blocks being used for blockwise computation to the next device in the ring, while simultaneously receiving key-value blocks from the previous one.
As long as block computations take longer than block transfers, overlapping these processes results in no added overhead compared to standard transformers.
The use of a ring topology for computing self-attention has also been studied in prior work  [ 21 ]  but it incurs non-overlapped communication overheads similar to sequence parallelism, making it infeasible for large context sizes. Our work utilizes blockwise parallel transformers  [ 23 ]  to substantially reduce memory costs, enabling zero-overhead scaling of context size across tens of millions of tokens during both training and inference, and allowing for the use of an arbitrarily large context size.
Since our approach overlaps the communication of key-value blocks between hosts in a ring through blockwise computation of transformers, we name it Ring Attention with Blockwise Parallel Transformers (Ring Attention).

We evaluate the effectiveness of our approach on language modeling benchmarks. Our experiments show that Ring Attention can reduce the memory requirements of Transformers, enabling us to train more than 500 times longer sequence than prior memory efficient state-of-the-arts and enables the training of sequences that exceed 100 million in length without making approximations to attention. Importantly, Ring Attention eliminates the memory constraints imposed by individual devices, empowering the training and inference of sequences with lengths that scale in proportion to the number of devices, essentially achieving near-infinite context size.

Our contributions are twofold: (a) proposing a memory efficient transformers architecture that allows the context length to scale linearly with the number of devices while maintaining performance, eliminating the memory bottleneck imposed by individual devices, and (b) demonstrating the effectiveness of our approach through extensive experiments.

2  Large Context Memory Constraint

Given input sequences

Q  ,  K  ,  V

∈

ℝ

s  ×  d

𝑄  𝐾  𝑉

superscript  ℝ

𝑠  𝑑

Q,K,V\in\mathbb{R}^{s\times d}

where

s

𝑠

s

is the sequence length and

d

𝑑

d

is the head dimension.
We compute the matrix of outputs as:

Attention  ​

(  Q  ,  K  ,  V  )

=

softmax  ​

(

Q  ​

K  T

d

)

​  V

,

Attention

𝑄  𝐾  𝑉

softmax

𝑄

superscript  𝐾  𝑇

𝑑

𝑉

\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(\frac{QK^{T}}{\sqrt{d}})V,

where

softmax

softmax

\mathrm{softmax}