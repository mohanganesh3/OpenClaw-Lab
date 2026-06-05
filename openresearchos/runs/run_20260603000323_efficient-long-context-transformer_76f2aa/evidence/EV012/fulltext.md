[2410.06718] MatMamba: A Matryoshka State Space Model

MatMamba: A Matryoshka State Space Model

Abhinav Shukla  †

&amp;Sai Vemprala  †

&amp;Aditya Kusupati  ⋄

&amp;Ashish Kapoor  †

\AND

†  Scaled Foundations

⋄  University of Washington

{abhinav,sai,ashish}@scaledfoundations.ai, kusupati@cs.uw.edu

AK is currently at Google DeepMind

Abstract

State Space Models (SSMs) like Mamba2 are a promising alternative to Transformers, with faster theoretical training and inference times – especially for long context lengths. Recent work on Matryoshka Representation Learning – and its application to Transformer backbones in works like MatFormer – showed how to introduce nested granularities of smaller submodels in one universal elastic model. In this work, we present MatMamba: a state space model which combines Matryoshka-style learning with Mamba2, by modifying the block to contain nested dimensions to enable joint training and adaptive inference. MatMamba allows for efficient and adaptive deployment across various model sizes. We train a single large MatMamba model and are able to get a number of smaller nested models for free – while maintaining or improving upon the performance of a baseline smaller model trained from scratch. We train language and image models at a variety of parameter sizes from 35M to 1.4B. Our results on ImageNet and FineWeb show that MatMamba models scale comparably to Transformers, while having more efficient inference characteristics. This makes MatMamba a practically viable option for deploying large-scale models in an elastic way based on the available inference compute. Code and models are open sourced at  https://github.com/ScaledFoundations/MatMamba

1  Introduction

Deep learning practitioners often train different sizes of the same kind of model to facilitate deployment in a variety of ranges of available inference compute. For example, the Llama 3.2  (Dubey et al.,  2024 )  series has 1B, 3B, 11B, and 90B variations. These models are extremely powerful individually – but due to independent training do not necessarily share the same metric space – a property which can be extremely useful for inference applications like speculative decoding  (Leviathan et al.,  2023 ) , hybrid cloud-edge inference, or just general input or compute adaptive processing.
Moreover, because training these models is expensive, we typically see only a few chosen sizes trained. This is not desirable in situations where the deployment setup can optimally support an intermediate model (e.g. a 2B model), but has to settle for the less accurate 1B model instead.

Techniques like model compression and distillation aim to address these issues, but require additional training (for which data may not be available), and can sometimes drop accuracy  (Jaiswal et al.,  2023 ) . Thus, methods that offer adaptive inference out of the box at intermediate granularities are extremely useful. This has been explored for Transformers  (Devvrit et al.,  2023 ; Cai et al.,  2024b )  and ConvNets  (Yu &amp; Huang,  2019 ; Cai et al.,  2019 ) . The core focus of this work is to try to enable out of the box adaptive inference in a newer architecture: Mamba2  (Dao &amp; Gu,  2024 ) .

State Space Models like Mamba2  (Dao &amp; Gu,  2024 )  and a number of other related newer architectures (see Section

2  ) have shown tremendous potential as they try to improve on the efficiency of Transformers, while maintaining their potency as accurate and general sequence processing architectures. Mamba2 has comparable scaling properties to Transformers, while being significantly faster at longer context lengths.

In this work, we introduce MatMamba, a nested Matryoshka structure  (Kusupati et al.,  2022 )  within a Mamba2 block  (Dao &amp; Gu,  2024 ) . MatMamba enables the extraction of hundreds of nested submodels from the same set of weights, without requiring any additional training during deployment. MatMamba is a general-purpose sequence processing architecture that can be applied to any type of model (encoder/decoder), modality (language/vision/sound/actions), loss function, or learning algorithm compatible with a Transformer or Mamba2 layer.

The philosophically closest work to MatMamba is MatFormer  (Devvrit et al.,  2023 )  – which imposes a nested structure on the FFN block in a Transformer layer. We use the same concept to impose a nested structure on any learnable parameter in a Mamba2 block that depends upon the hidden dimensionality of the block. Formally, a MatMamba block consists of a nested combination of

g

𝑔

g

Mamba2 blocks

M  i

subscript  𝑀  𝑖

M_{i}

, such that

M  1

⊂

M  2

⊂  …  ⊂

M  g

subscript  𝑀  1

subscript  𝑀  2

…

subscript  𝑀  𝑔

M_{1}\subset M_{2}\subset...\subset M_{g}

, where

M  i

⊂

M  j

subscript  𝑀  𝑖

subscript  𝑀  𝑗

M_{i}\subset M_{j}

means that all the parameters of a sub-block

M  i

subscript  𝑀  𝑖

M_{i}

are present in

M  j

subscript  𝑀  𝑗

M_{j}

. We train the model using

g

𝑔

g

forward passes with gradient accumulation followed by a single backward pass for parameter updates (see Figure

1  ).

By jointly training all

g

𝑔

g

granularities, the smallest sub-blocks are incentivized to represent the most important information, like in Matryoshka Representation Learning  (Kusupati et al.,  2022 ) . We can now use any of the

g

𝑔

g

nested sub-blocks

M  i

subscript  𝑀  𝑖

M_{i}

flexibly. Additionally, we can flexibly slice the block along  any  dimensionality (even beyond the

g

𝑔

g

explicitly optimized granularities). Using Mix’n’Match (Section

3.4  ), we can perform this operation over multiple layers at varying granularities to flexibly extract a combinatorially large number of models from the single larger model. We observe that these extracted models preserve the metric space of the larger model, and are accurate across a variety of tested tasks – effectively allowing us to choose a tradeoff between model performance and compute.

We train MatMamba-based vision models (MatMamba-Vision), and find that: (a) MatMamba-Vision models scale as well as baseline Mamba2 based models at all

g  =  4

𝑔  4

g=4

granularities; (b) Using Mix’n’Match, we can flexibly extract submodels between the explicitly optimized granularities. The submodels span (and sometimes exceed) the pareto optimal accuracy-vs-compute curve; (c) MatMamba-Vision models are significantly faster at higher resolutions than ViTs, making them promising candidates for long-form and high resolution visual tasks, while enabling adaptive visual processing with the nested submodels (see Section

4.1.1  ).

Furthermore, MatMamba-Vision models can act as elastic image encoders for adaptive image retrieval. We can encode visual datasets with the largest model, and because the smaller submodels share its metric space, we can use them as query encoders, needing drastically lower compute with minimal loss in accuracy (see Section

4.1.2  ).

We also train MatMamba-based decoder language models (MatMamba-LM) at various sizes from 130M-1.4B parameters, and at

g  =  4

𝑔  4

g=4

granularities. We make similar observations here too, that MatMamba-LM models scale as well as Mamba2 baselines with the same architecture for all nested granularities. We also observe interesting homogenous scaling behaviour between the nested granularities for different models (see Section

4.2  ).

Through MatMamba, for the first time, we bring together the adaptivity of Matryoshka-style learning and the efficiency of state space models (SSMs) like Mamba2  (Dao &amp; Gu,  2024 ) .

We make the following research contributions:

1.

We introduce MatMamba, which imposes a nested Matryoshka structure on a Mamba2 state space model. We jointly optimize all nested granularities to train a single elastic model.

2.

We show that MatMamba models scale as well as the baseline Mamba2 models for a variety of model sizes from 35M-1.4B parameters on language and vis