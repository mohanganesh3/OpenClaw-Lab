[2510.27004] Mixture-of-Transformers Learn Faster: A Theoretical Study on Classification Problems

Mixture-of-Transformers Learn Faster: A Theoretical Study on Classification Problems

Hongbo Li

Department of ECE

The Ohio State University

li.15242@osu.edu

Qinhang Wu

Department of CSE

The Ohio State University

wu.5677@osu.edu

Sen Lin

Department of Computer Science

University of Houston

slin50@central.uh.edu

Yingbin Liang

Department of ECE

The Ohio State University

liang.889@osu.edu

Ness B. Shroff

Department of ECE
 Department of CSE
 The Ohio State University

shroff.11@osu.edu

Abstract

Mixture-of-Experts (MoE) models improve transformer efficiency but lack a unified theoretical explanation—especially when both feed-forward and attention layers are allowed to specialize.
To this end, we study the Mixture-of-Transformers (MoT), a tractable theoretical framework in which each transformer block acts as an expert governed by a continuously trained gating network. This design allows us to isolate and study the core learning dynamics of expert specialization and attention alignment.
In particular, we develop a three-stage training algorithm with continuous training of the gating network, and show that each transformer expert specializes in a distinct class of tasks and that the gating network accurately routes data samples to the correct expert.
Our analysis shows how expert specialization reduces gradient conflicts and makes each subtask strongly convex.
We prove that the training drives the expected prediction loss to near zero in

𝒪  ​

(

log  ⁡

(

ϵ

−  1

)

)

\mathcal{O}(\log(\epsilon^{-1}))

iteration steps, significantly improving over the

𝒪  ​

(

ϵ

−  1

)

\mathcal{O}(\epsilon^{-1})

rate for a single transformer.
We further validate our theoretical findings through extensive real-data experiments, demonstrating the practical effectiveness of MoT.
Together, these results offer the first unified theoretical account of transformer-level specialization and learning dynamics, providing practical guidance for designing efficient large-scale models.

1  Introduction

Recently, the transformer architecture  [ 1 ]  has emerged as the foundational model across a wide range of machine learning domains, including computer vision ( [ 14 ,  15 ,  16 ] ), natural language processing ( [ 18 ,  17 ] ), and speech processing ( [ 19 ,  20 ] ).
Despite its success, transformers often face scalability challenges and significant computational costs when applied to diverse or large-scale tasks—particularly when these tasks involve conflicting or heterogeneous feature patterns.
As a result, improving the scalability and training efficiency of transformers remains a pressing and open research problem.

A prominent strategy to address these issues is to introduce Mixture-of-Experts (MoE) layers that route tokens or samples to specialized feed-forward networks ( [ 21 ,  22 ,  23 ,  7 ,  13 ] ).
While these approaches significantly expand model capacity, they almost always leave self-attention shared across all experts, restricting specialization to FFNs ( [ 21 ,  7 ] ).
Given the central role of self-attention in capturing complex dependencies and enabling the expressive power of transformers, recent works have proposed incorporating Mixture-of-attention (MoA) mechanisms into the expert architecture, demonstrating improved empirical performance ( [ 25 ,  3 ,  24 ] ).
However, these efforts are almost entirely empirical. No existing theory explains how attention- and FFN-level specialization interact, or how such architectures converge during training.

We close this gap by studying the  Mixture-of-Transformers (MoT)  model, a tractable theoretical framework in which each transformer block acts as a distinct expert with its own attention and FFN layers.
A gating network dynamically assigns data to specialized experts.
This setting lets us isolate and analyze the joint learning dynamics of expert specialization and attention alignment—something prior work has not addressed.
We summarize our main contributions as follows.

•

To the best of our knowledge, this paper presents the  first theoretical analysis with benefit characterization for full-transformer specialization.  We model MoT under a general mixture-of-classification setup. To better understand the role of the gating network in data assignment and the specialization of self-attention and feed-forward networks (FFNs), we introduce a three-stage training algorithm with continuous gating updates. In the first stage, we freeze the self-attention layers and train only the FFNs to encourage diversity across transformers. We prove that each transformer expert specializes in a distinct class of tasks, and the gating network accurately routes data samples to the correct expert. In the second stage, we freeze the FFNs and train the attention layers. We show that self-attention further reduces the training loss by extracting relevant classification signals, highlighting a key advantage of MoT over attention-absent MoE models. Finally, in the third stage, we fine-tune the FFNs to reinforce specialization.

•

We provide  theoretical guarantees  on the convergence of the MoT architecture to the optimum under our three-stage training process.
Specifically, we show that the expected prediction loss converges to within

ϵ  \epsilon

-accuracy in

𝒪  ​

(

log  ⁡

(

ϵ

−  1

)

)

\mathcal{O}(\log(\epsilon^{-1}))

iterations via gradient descent.
Compared to existing theoretical results for multi-head transformers trained on mixture-of-classification problems ( [ 9 ] ), our analysis shows that MoT significantly shortens the convergence time from

𝒪  ​

(

ϵ

−  1

)

\mathcal{O}(\epsilon^{-1})

to

𝒪  ​

(

log  ⁡

(

ϵ

−  1

)

)

\mathcal{O}(\log(\epsilon^{-1}))

.
This improvement stems from expert specialization, which reduces the impact of conflicting data gradients and simplifies the classification problem faced by each transformer.
As a result, MoT benefits from the strong convexity of the per-expert loss functions, enabling faster and more stable convergence.

•

Finally, our  extensive real-data experiments  validate our theoretical findings and provide practical guidance for designing effective MoT systems. Interestingly, we observe that on simple datasets, a single expert suffices to solve all tasks, leading the router to direct most data to a small subset of transformers. In contrast, for more complex datasets, each transformer specializes effectively, resulting in a well-partitioned and efficient MoT system where specialization offers a clear advantage over a single multi-head transformer.

2  Related Works

Transformers. 
Transformers have been extensively studied since the introduction of the self-attention mechanism  [ 1 ] , focusing on improving model efficiency, generalization, and adaptability.
Empirical studies have explored transformer architectures across multiple domains, including encoder-only models (e.g.,  [ 26 ,  28 ,  27 ] ) and encoder-decoder architectures (e.g.,  [ 29 ,  30 ,  31 ] ).
In computer vision, ViT demonstrated that treating images as sequences of patches enables transformer-based models to match CNN performance  [ 32 ,  34 ,  33 ] .
Meanwhile, transformers have also found applications in other machine learning areas such as speech processing  [ 19 ,  20 ,  36 ] .

On the theoretical side, recent efforts have aimed to formally understand the capabilities and limitations of Transformer models  [ 4 ,  12 ,  11 ,  37 ,  9 ] .
A popular line of work investigates in-context learning, where transformers equipped with linear or softmax attention can solve tasks such as linear regression  [ 12 ,  11 ] , classification  [ 5 ] , and causal inference  [ 38 ]  directly from token sequences, without parameter updates.
Other studies analyze how Transformers learn tasks like binary classification  [ 4 ] , topic modeling  [ 39 ] , and position-feature correlations  [ 37 ]