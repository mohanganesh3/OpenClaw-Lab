[2106.04426] Hash Layers For Large Sparse Models

Hash Layers For Large Sparse Models

Stephen Roller  Sainbayar Sukhbaatar  Arthur Szlam  Jason Weston

Facebook AI Research

Abstract

We investigate the training of sparse layers that use different parameters for different inputs based on hashing in large Transformer models. Specifically, we modify the feedforward layer to hash to different sets of weights depending on the current token, over all tokens in the sequence. We show that this procedure either outperforms or is competitive with learning-to-route mixture-of-expert methods such as Switch Transformers and BASE Layers, while requiring no routing parameters or extra terms in the objective function such as a load balancing loss, and no sophisticated assignment algorithm. We study the performance of different hashing techniques, hash sizes and input features, and show that balanced and random hashes focused on the most local features work best, compared to either learning clusters or using longer-range context. We show our approach works well both on large language modeling and dialogue tasks, and on downstream fine-tuning tasks.

1  Introduction

Recent studies of Transformer models have shown a clear trend towards improvements with scale in data and model size  [ 1 ] , mirroring the same trend in Machine Learning more generally.
However, when architected naively, larger (in terms of parameter count) models are slower to train and to evaluate; and at extreme scale, with current computer systems,
necessitate complex engineering to facilitate communication between workers. To address these challenges, researchers have studied Mixtures-of-Experts (MoE) models  [ 2 ,  3 ,  4 ,  5 ,  6 ,  7 ,  8 ] , where a “gater” routes computation through a sparse subset of the weights of the model (the “expert modules”). Specifically in the setting of Transformers for Natural Language Processing (NLP), recent approaches have led to state of the art performance in language modeling  [ 8 ] . MoE models allow increasing the number of parameters in the model while holding steady the number of computations that affect a given sample.

A key component to a MoE model is the routing (gating) strategy. While MoE models can be computationally advantageous per parameter compared to a dense model, they might be functionally less powerful per parameter. A poor routing strategy might lead to expert modules that are not properly specialized (essentially making a stochastic ensemble model); or overly specialized, using the data assignment function to overfit. Meanwhile, the routing strategy itself must be efficient.

A standard approach is to train a layer of weights that makes the routing decision based upon the input to the layer to be routed. Classically, this may have been implemented with a softmax over the choice of expert modules, and fitted via backpropagation. However, a dense softmax requires all expert modules to run on all data points at train time, which negates the computational savings. Several works have shown that sparsity can be maintained during training, e.g.  [ 9 ,  7 ,  8 ,  10 ] . In particular, Switch Transformers  [ 8 ]  select the top expert per token using a softmax over the token’s hidden state, but require a load balancing term in the objective function or they
can become imbalanced or degenerate, giving poor results. BASE Layers  [ 10 ]  employ a linear assignment algorithm to try to resolve the same problem.

In this work, we describe a simple, sparse, efficient routing strategy based on hashing input tokens that is effective in the Transformers-for-NLP setting.
We show this approach is effective on a number of datasets, comparing favorably to both Switch Transformers and BASE Layers. As the routing strategy requires no extra parameters, no change to the objective function or assignment algorithm, its simplicity means it is robust, fast and easy to implement.
We provide detailed analysis to explain why our method works, and in which conditions. Given that when training very large models one may typically have only one shot given the required compute budget, and experimenters will be unable to try many parameter choices, we hence advocate our approach as a strong candidate for such a setting.

…

…

…

…

“We”

“eat”

“every”

“taco”

\pgfmathresult pt

1

\pgfmathresult pt

MoE

FFN

1

…

\pgfmathresult pt

2

\pgfmathresult pt

MoE

FFN

2

…

\pgfmathresult pt

3

\pgfmathresult pt

MoE

FFN

3

…

\pgfmathresult pt

4

\pgfmathresult pt

MoE

FFN

4

…

\pgfmathresult pt

self-attention

Layer

l

𝑙

l

Layer

l  +  1

𝑙  1

l+1

FFN  1

subscript

FFN

1

\mbox{FFN}_{1}

FFN  2

subscript

FFN

2

\mbox{FFN}_{2}

FFN  3

subscript

FFN

3

\mbox{FFN}_{3}

Hash

3

𝐡  ¯

l

superscript

¯  𝐡

𝑙

\bar{\mathbf{h}}^{l}

𝐡  l

superscript  𝐡  𝑙

\mathbf{h}^{l}

MoE FFN

Figure 1:

Overview of the Hash Layer.  Tokens are routed to fixed expert modules based on their hash.

2  Background

Let us first introduce the Mixture-of-Experts setting where we apply our hash-based routing strategy.
We use the same setting as  [ 11 ,  8 ,  10 ]  where a feedforward network (FFN) in a Transformer is replaced by its MoE version.
Given a tokenized input sequence

{

x  1

,

x  2

,  …  ,

x  T

}

subscript  𝑥  1

subscript  𝑥  2

…

subscript  𝑥  𝑇

\{x_{1},x_{2},\dots,x_{T}\}

of

T

𝑇

T

tokens,
a representation for each token is computed in parallel by a standard
Transformer  [ 12 ]

𝐡  1  L

,

𝐡  2  L

,  …  ,

𝐡  T  L

=

Transformer  ​

(

x  1

,

x  2

,  …  ,

x  T

)

.

subscript

superscript  𝐡  𝐿

1

subscript

superscript  𝐡  𝐿

2

…

subscript

superscript  𝐡  𝐿

𝑇

Transformer

subscript  𝑥  1

subscript  𝑥  2

…

subscript  𝑥  𝑇

\mathbf{h}^{L}_{1},\mathbf{h}^{L}_{2},\ldots,\mathbf{h}^{L}_{T}=\textsc{Transformer}(x_{1},x_{2},\dots,x_{T}).

(1)

The Transformer consists of

L

𝐿

L

layers that computes final hidden states for each token,
and each layer is composed of self-attention and FFN sublayers, where FFNs are two-layer fully connected networks

𝐡  ¯

t  l

=

SelfAttn  ​

(

𝐡  t

l  −  1

)

𝐡  t  l

=

FFN  ​

(

𝐡  ¯

t  l

)

.

formulae-sequence

superscript

subscript

¯  𝐡

𝑡

𝑙

SelfAttn

superscript

subscript  𝐡  𝑡

𝑙  1

superscript

subscript  𝐡  𝑡

𝑙

FFN

superscript

subscript

¯  𝐡

𝑡

𝑙

\bar{\mathbf{h}}_{t}^{l}=\mbox{SelfAttn}(\mathbf{h}_{t}^{l-1})\quad\quad\mathbf{h}_{t}^{l}=\mbox{FFN}(\bar{\mathbf{h}}_{t}^{l}).

(2)

Here we omit skip-connections and normalization for brevity.
We can then replace one or more of the FFN sublayers with expert modules. Replacing the FNN at layer

l

𝑙

l

with

K

𝐾

K

expert FFNs, their output is then mixed with some gating function

g  ​

(  ⋅  )

𝑔  ⋅

g(\cdot)

:

𝐡  t  l

=

FFN  ​

(

𝐡  ¯

t  l

)

→

𝐡  t  l

=

∑

i  =  1

K

g  i

​

(

𝐡  ¯

t  l

)

​

FFN  i

​

(

𝐡  ¯

t  l

)

,

t  =

1  ,  …  ,  T

,

formulae-sequence

subscript

superscript  𝐡  𝑙

𝑡

FFN

superscript

subscript

¯  𝐡

𝑡

𝑙

→

formulae-sequence

subscript

superscript  𝐡  𝑙

𝑡

superscript

subscript

𝑖  1

𝐾

subscript  𝑔  𝑖

superscript

subscript

¯  𝐡

𝑡

𝑙

subscript

FFN

𝑖

superscript

subscript

¯  𝐡

𝑡

𝑙

𝑡

1  …  𝑇

\mathbf{h}^{l}_{t}=\mbox{FFN}(\bar{\mathbf{h}}_{t}^{l})\quad\rightarrow\quad\mathbf{h}^{l}_{t}=\sum_{i=1}^{K}g_{i}(\bar{\mathbf{h}}_{t}^{l})~{}\mbox{FFN}_{i}(\bar{\mathbf{h}}_{t}^{l}),~{}~{}~{}~{}t=1,\dots,T,

(3)

where importantly each token is routed to a different mixture of experts, as the gating function depends on the token’s specific hidden state

𝐡  ¯

t  l

superscript

subscript

¯  𝐡

𝑡

𝑙

\bar{\mathbf{h}}_{t}^{l}

.

Sparse MoE methods assume gating values

g  i

subscript  𝑔  𝑖

g_{i}

are often zero, so only a few experts need to be computed for better efficiency. As expert FFNs do not share parameters, the number of parameters increases with

K

𝐾

K

while the amount of compu