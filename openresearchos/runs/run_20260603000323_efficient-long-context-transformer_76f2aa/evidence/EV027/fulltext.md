[2512.16615] Trainable Log-linear Sparse Attention for Efficient Diffusion Transformers

Trainable Log-linear Sparse Attention for Efficient Diffusion Transformers

Yifan Zhou 1

Zeqi Xiao 1

Tianyi Wei 1

Shuai Yang

2

Xingang Pan  1

1 S-Lab, Nanyang Technological University

2  Wangxuan Institute of Computer Technology, Peking University

{yifan006, zeqi001, tianyi.wei, xingang.pan}@ntu.edu.sg

williamyang@pku.edu.cn

Abstract

Diffusion Transformers (DiTs) set the state of the art in visual generation, yet their quadratic self-attention cost fundamentally limits scaling to long token sequences. Recent Top-

K  K

sparse attention approaches reduce the computation of DiTs by compressing tokens into block-wise representation and selecting a small set of relevant key blocks, but still suffer from (i) quadratic selection cost on compressed tokens and (ii) increasing

K  K

required to maintain model quality as sequences grow. We identify that their inefficiency is due to the single-level design, as a single coarse level is insufficient to represent the global structure.
In this paper, we introduce  Log-linear Sparse Attention (LLSA) , a trainable sparse attention mechanism for extremely long token sequences that reduces both selection and attention costs from quadratic to  log-linear complexity  by utilizing a hierarchical structure. LLSA performs hierarchical Top-

K  K

selection,
progressively adopting sparse Top-

K  K

selection with the indices found at the previous level,
and introduces a Hierarchical KV Enrichment mechanism that preserves global context while using fewer tokens of different granularity during attention computation.
To support efficient training, we develop a high-performance GPU implementation that uses only sparse indices for both the forward and backward passes, eliminating the need for dense attention masks.
We evaluate LLSA on high-resolution pixel-space image generation without using patchification and VAE encoding. LLSA accelerates attention inference by

28.27  ×

28.27\times

and DiT training by

6.09  ×

6.09\times

on

256  ×  256

256\times 256

pixel token sequences, while maintaining generation quality. The results demonstrate that LLSA offers a promising direction for training long-sequence DiTs efficiently.

Figure 1 :

Comparison between a general Top-

K  K

sparse attention and our Log-linear Sparse Attention (LLSA). In the example, we use a token sequence of length

N  =  8

N=8

, block size

B  =  2

B=2

, Top-

K  K

parameter

K  =  1

K=1

. To reduce the complexity of the selection stage from

O  ​

(

N  2

)

O(N^{2})

to

O  ​

(  N  )

O(N)

, we extend single-level selection to

O  ​

(

log  ⁡  N

)

O(\log N)

levels. To achieve this, we compute the Top-

K  K

of the full sequence on the coarsest level and recursively compute the sparse Top-

K  K

on the remaining levels. To preserve the global context for attention, we enrich the key, value sets for each query with coarse tokens of length

O  ​

(

K  ​

log  ⁡  N

)

O(K\log N)

found in the selection stage.

1  Introduction

Diffusion Transformers (DiTs)  [ dit ]  have become the state-of-the-art backbone for visual generation tasks. As resolution and sequence length scale up, the dominant bottleneck lies in the quadratic complexity of full self-attention  [ transformer ] , where the computation cost grows as

O  ​

(

N  2

)

O(N^{2})

with token length

N  N

. In practice, this prevents DiTs from scaling to high-resolution images or long video sequences. For example, FLUX  [ flux ]  operates on a

64  ×  64

64\times 64

latent image (4096 tokens), while Wan 2.1  [ wan ]  uses

21  ×  45  ×  80

21\times 45\times 80

latent videos (75,600 tokens). Scaling to longer sequences requires a fundamentally more efficient attention mechanism.

Sparse attention has recently emerged as a promising alternative to full attention  [ zhangspargeattention ,  xi2025sparse ,  xia2025training ] . A widely adopted variant is Top-

K  K

block sparse attention  [ yuan2025native ,  lu2025moba ,  zhangspargeattention ] , which operates in three stages: (1) compress query and key tokens into coarse representations that summarize the block-wise information; (2) compute the coarse similarity scores between compressed tokens, and select the Top-

K  K

key blocks for each query block; (3) perform block sparse attention on the selected blocks. Although effective for moderate sequence lengths, this paradigm faces two major limitations when scaling further: (1) The selection stage still incurs quadratic cost on compressed tokens; (2) To maintain global context, prior methods set the sparsity as a constant and use a larger

K  K

for longer sequences

[ vsa ,  sla ,  vmoba ] . These limitations arise from the  single-level  design of existing Top-

K  K

sparse attention: a single coarse-grained view is insufficient to represent global structure for long sequences.

A natural solution is to extend the single level into  hierarchical  structure, where global information can be represented using only

O  ​

(

log  ⁡  N

)

O(\log N)

coarse tokens of progressively coarser granularity.

Inspired by prior work showing that the dense attention matrix can be approximated by hierarchical coarse attention matrices

[ zeng2022multi ,  zhu2021h ] , we introduce  Log-linear Sparse Attention (LLSA) , a trainable sparse attention mechanism that reduces attention complexity from quadratic to  log-linear . LLSA builds upon Top-

K  K

sparse attention with two key innovations: (1) We compress query and key features across multiple logarithmic hierarchy levels and progressively perform Top-

K  K

selection from coarse to fine. This hierarchical design reduces the complexity of the selection stage from

O  ​

(

N  2

)

O(N^{2})

to

O  ​

(  N  )

O(N)

. (2) Instead of using a large

K  K

for longer sequences, we propose a  Hierarchical KV Enrichment  mechanism that incorporates coarse key/value representations selected at higher hierarchy levels into attention computing. This preserves global context and mitigates information loss from sparsification, allowing LLSA to operate with significantly smaller

K  K

and lower cost.

Efficient GPU implementation is essential for Top-

K  K

sparse attention. In standard FlashAttention  [ flashattention1 ] , the sparse indices are represented as a binary mask. Constructing and processing this mask leads to quadratic memory and computation overhead. To avoid this, we implement a GPU-efficient Top-

K  K

indexing algorithm that operates directly on sparse indices. In the forward pass, we gather only the selected key blocks. In the backward pass, we dynamically compute the reverse lookup of sparse indices via a lightweight sparse index transpose kernel. This ensures end-to-end log-linear complexity during training.

We evaluate LLSA on high-resolution pixel space image generation. Specifically, we train pixel DiTs without patchification and VAE encoding up to

256  ×  256

256\times 256

(65,536) pixel tokens with one H200 GPU. LLSA significantly improves the training efficiency of full attention DiTs by

6.09  ×

6.09\times

while maintaining generation quality. Compared to existing Top-

K  K

sparse attention algorithms  [ sla ,  vsa ] , LLSA achieves higher generation quality and training throughput. Thanks to Hierarchical KV Enrichment, LLSA maintains global context even with a significantly smaller

K  K

.
In our experiments, LLSA with

K  =  8

K=8

outperforms prior Top-

K  K

methods even when their

K  K

is substantially larger (

K  =  20

K=20

or

K  =  32

K=32

), highlighting the practical efficiency of our design. We further integrate LLSA into PixelFlow  [ chen2025pixelflow ]  to show its capacity on ImageNet  [ imagenet ] . Moreover, our backward kernel achieves nearly constant throughput across different sequence lengths, confirming the linear complexity of our implementation.

In summary, the contributions