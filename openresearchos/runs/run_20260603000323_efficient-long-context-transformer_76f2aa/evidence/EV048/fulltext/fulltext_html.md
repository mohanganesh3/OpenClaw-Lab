[2512.11411] Sliced ReLU attention: Quasi-linear contextual expressivity via sorting

Sliced ReLU attention: Quasi-linear contextual expressivity via sorting

Siwan Boufadene

and

François-Xavier Vialard

Abstract.

We introduce  sliced ReLU attention , a new attention mechanism that departs structurally from both softmax and ReLU-based alternatives. Instead of applying a nonlinearity to pairwise dot products, we operate on one-dimensional projections of key–query  differences  and leverage sorting to obtain quasi-linear complexity. This construction yields a differentiable, non-symmetric kernel that can be computed in

O  ​

(

n  ​

log  ⁡

(  n  )

)

O(n\log(n))

through a sorting procedure, making it suitable for very long contexts.
Beyond computational benefits, the model retains strong theoretical expressive power: we establish two in-context expressivity results, previously known for softmax attention, showing that sliced ReLU attention preserves the ability to perform nontrivial sequence-to-sequence disentangling tasks and satisfies a contextual universal approximation property.
Finally, we illustrate the potential practical interest of this kernel in small-scale experiments.

LIGM, Université Gustave Eiffel

siwan.boufadene@univ-eiffel.fr

francois-xavier.vialard@univ-eiffel.fr

1.  Introduction

Ever since their introduction, Transformers  [ Vas+17 ]  and their variants have significantly transformed the field of machine learning. Originally introduced for natural language processing (NLP), they now achieve outstanding performance across a wide range of tasks, including text understanding and generation  [ Bro+20 ,  Dev+19 ] , computer vision  [ Dos+21 ,  Liu+21 ] , and audio processing  [ Bae+20 ,  Che+22 ] . A key strength of Transformers lies in their ability to model dependencies over arbitrary-length contexts. Inputs are decomposed into sequences of tokens, which interact in the Transformer computations through their defining attention mechanism. Self-attention allows tokens to attend to each other across the entire input, enabling the modeling of long-range dependencies in a flexible, expressive, and context-driven way.

Main challenges:  A key limitation of standard attention is its quadratic computational cost with respect to the input length. This restricts the application of Transformer architectures to tasks involving long sequences, such as document-level understanding, long-text summarization, or high-resolution image analysis. Softmax attention provides precise and expressive interactions between tokens, and benefits from a solid theoretical foundation. Consequently, any efficient alternative should aim to preserve comparable theoretical guarantees.

Related works:  There have been several approaches to tackle this computational bottleneck. One direction focuses on more efficient implementations of standard softmax attention, such as FlashAttention  [ Dao23 ] . Although these methods optimize memory usage and constant factors, their overall complexity remains quadratic in the input length. A second line of methods seeks to approximate the attention matrix, either through random features  [ Cho+22 ] , low-rank factorization  [ Win+20 ,  Hu+22 ] , or sparse or local mechanisms such as sliding windows, clustered attention or global-local mixtures  [ BPC20 ,  VKF20 ,  Zah+21 ] . The linearization of the softmax around the origin leads to linear attention architectures  [ Kat+20 ,  Qin+22 ] , where the attention output is computed using kernelized feature maps via the dot product of two learnable maps, scaling linearly with sequence length. While fast, these models often suffer from degraded performance on tasks requiring rich global interactions. Other attention-free architectures have been proposed, such as state space sequence models  [ GGR22 ,  Goe+22 ]  or convolutional models  [ Liu+22 ] , which sometimes match or outperform Transformers on long-sequence tasks. We refer to  [ Tay+22 ,  Aro+25 ]  for a more in-depth review of efficient models.

From a theoretical perspective, recent analyses have provided deeper insight into the success of softmax-based attention architectures across diverse tasks. Transformers have been shown to possess strong universal approximation properties, enabling them to represent arbitrary sequence-to-sequence mappings  [ Yun+20 ] . More recent theoretical works study the  mean-field  limit of Transformers, extending attention to infinitely long contexts represented as probability measures in a Euclidean space  [ VBC20 ,  San+22 ] . The optimal transport framework has been used to study the smoothness of Transformers in measure spaces  [ CAP24 ] , and universality results have been generalized to such spaces  [ FHP24 ] . Similar properties hold for other attention computations, such as sigmoid-based variants in  [ Ram+25 ] . Transformers have also been studied in a continuous-depth limit, where they act as universal measure-to-measure interpolators  [ GRR24 ] . Clustering properties of the resulting attention dynamics are studied in  [ Ges+24 ,  VBC20 ] .

Main contributions:

Our work fits within the broader effort to investigate alternative attention mechanisms through a kernel-based perspective.
Somewhat surprisingly, given the importance of attention in modern architectures, relatively few fundamentally distinct alternatives have been proposed.
Moreover, the reasons of the empirical dominance of softmax attention against its competitors remain poorly understood, although a few hypotheses have been recently proposed  [ Aro+23 ,  Dur+25 ,  She+24 ] .
Most of these alternatives are driven by the goal of mitigating the quadratic computational cost inherent to attention, a cost that typically arises in any mechanism relying on pairwise interactions.
 However, in certain one-dimensional settings, this footprint can be reduced to quasi-linear time through sorting.  Motivated by the possibility of generating an attention-like mechanism through sorting,
we introduce a new quasi-linear Transformer architecture, based on a sliced kernel computation, extending ideas from  [ BFV25 ] . In contrast to other efficient architectures, our method enables exact, global, and expressive kernel-based interactions while maintaining a quasi-linear computational cost and linear memory usage.

From a theoretical standpoint, we show that our model satisfies the same universality properties as those established for softmax attention in  [ FHP24 ] , with depth as the only asymptotically growing parameter. We also prove that our proposed ReLU attention is a universal approximator of sequence-to-sequence functions, in the same way that softmax attention is  [ Yun+20 ] .

Finally, we provide an empirical validation of the method on a range of standard Transformer benchmarks. We first evaluate sliced ReLU attention on the Long Range Arena benchmark  [ Tay+20 ] , which is specifically designed to test the ability to model long-context reasoning. We then examine its behavior on geometric classification problems such as ModelNet40  [ Wu+15 ] , using a Point Cloud Transformer architecture  [ Guo+21 ] .

Structure of the paper:  In Section

2  , we begin by introducing the standard softmax self-attention mechanism, followed by a presentation of our model. We link the

ReLU  \operatorname{ReLU}

kernel to the Energy Distance kernel, adapting the sliced computation of  [ BFV25 ]  to our case. We establish the

O  ​

(

n  ​

log  ⁡  n

)

O(n\log n)

complexity of the resulting attention layer, and provide justification for the normalization and centering choices, coming from the conditionally positive definite nature of the

ReLU  \operatorname{ReLU}

kernel. In Section

3  , we study the expressivity properties of our

ReLU  \operatorname{ReLU}

attention mechanism. We first show that it can approximate arbitrary sequence-to-sequence mappings, demonstrating that it preserves the key expressive capabilities of softmax attention.