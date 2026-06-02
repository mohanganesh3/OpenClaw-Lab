[2506.05340] Exploring Diffusion Transformer Designs via Grafting

Exploring Diffusion Transformer Designs via Grafting

Keshigeyan Chandrasegaran

∗‡1,2

Michael Poli

∗1,2

Daniel Y. Fu

3,4

Dongjun Kim

1

Lea M. Hadzic

1

Manling Li

1,5

Agrim Gupta

6

Stefano Massaroli

2

Azalia Mirhoseini

1

Juan Carlos Niebles

1,7†

Stefano Ermon

1†

Li Fei-Fei

1†

1  Stanford University

2  Liquid AI  3  Together AI  4  UC San Diego

5  Northwestern University  6  Google DeepMind  7  Salesforce Research

grafting.stanford.edu

Abstract

Designing model architectures requires decisions such as
selecting operators (e.g., attention, convolution) and configurations (e.g., depth, width).
However, evaluating the impact of these decisions on model quality requires costly pretraining, limiting architectural investigation.
Inspired by how new software is built on existing code, we ask: can new architecture designs be studied using pretrained models?
To this end, we present  grafting , a simple approach for editing pretrained diffusion transformers (DiTs) to materialize new architectures under small compute budgets.
Informed by our analysis of activation behavior and attention locality,
we construct a testbed based on the DiT-XL/2 design to study the impact of grafting on model quality.
Using this testbed, we
develop a family of hybrid designs via grafting:
replacing softmax attention with gated convolution, local attention, and linear attention, and replacing MLPs with variable expansion ratio and convolutional variants.
Notably, many hybrid designs achieve good quality (FID: 2.38–2.64 vs. 2.27 for DiT-XL/2)
using

&lt;  2

&lt;2

% pretraining compute.
We then graft a text-to-image model (PixArt-

Σ  \Sigma

), achieving a 1.43

×  \times

speedup with less than a 2% drop in GenEval score.
Finally, we present a case study that restructures DiT-XL/2 by converting every pair of sequential transformer blocks into parallel blocks via grafting.
This reduces model depth by 2

×  \times

and yields better quality (FID: 2.77) than other models of comparable depth.
Together, we show that new diffusion model designs can be explored by grafting pretrained DiTs, with edits ranging from operator replacement to architecture restructuring.
Code and grafted models:  grafting.stanford.edu .

0

0  footnotetext:

∗  Equal contribution.

†

Equal senior authorship.

0

0  footnotetext:

‡

Part of this work was done at Liquid AI.

0

0  footnotetext:

Correspondence to  {keshik,poli}@stanford.edu

1  Introduction

Model architecture design plays a central role in machine learning, alongside data, algorithms, compute, and benchmarks.
It defines a learnable function and entails key decisions, including the choice of operators (e.g., attention, convolution) and configurations (e.g., model depth, width).
Despite this, insight into architectures—what works and what doesn’t—is difficult to obtain due to the prohibitive costs of training models from scratch, especially in today’s foundation model era.
As a result, studying
new
architectures remains a challenge, particularly for generative models.
Much like
how new software is built on existing code rather than written from scratch, can pretrained models serve as scaffolds for studying new architectures?
In this work, we investigate  architectural editing of pretrained models to study new architecture designs. 
We focus on diffusion transformers (DiTs), a class of generative transformers widely used for image and video generation

peebles2023scalable

;

videoworldsimulators2024

;

gupta2023photorealistic

.

A pretrained model implements a computational graph to perform tasks such as image or video generation.
Given a new architectural idea and a pretrained model, we investigate whether the idea can be materialized by modifying its computational graph under small compute budgets.
For example, one might hypothesize that a convolutional design could replace
Multi-Head Attention (MHA) or Multi-Layer Perceptron (MLP)
in a DiT.
A simple way to materialize this idea is to replace MHA or MLP operators with a convolutional operator, while preserving model quality.
This raises two key questions:
(Q1)  operator initialization : How to initialize a new operator before integrating it into the computational graph?
(Q2)  error accumulation : How to mitigate error propagation as multiple operators are integrated into the computational graph?

To address these questions, we present  grafting  1

1

1

Grafting draws inspiration from horticultural grafting, where efficient components (scions) are integrated into established systems (rootstock) to enhance functionality, such as yield and disease resistance

eliezer_grafting

.

,
a simple two-stage approach to architecture editing (Fig.

1  ). Grafting proceeds as follows:
(i)  activation distillation :
This stage transfers the functionality of the original operator to the new one by distilling its activations using a regression objective.
(ii)  lightweight finetuning :
This stage mitigates error propagation caused by integrating multiple new operators by finetuning using limited data.
Architectural editing spans multiple strategies—adding, removing, and replacing

zhanglolcats

;

wang2024the

;

bick2024transformers

operators.
We focus on  operator replacement  as the core strategy:
swapping one operator for another.
Other strategies can be viewed as special cases of replacement.

Figure 1:

Grafting overview.

(a,b) Model architecture design via grafting.  Studying new model architecture designs requires costly pretraining. Grafting materializes new architectures by editing pretrained models under small compute budgets (Sec.

3  ).
 (c) Class-conditional image generation.  Samples generated by hybrid architectures obtained via grafting (Sec.

4  ).
 (d) High-resolution text-to-image generation.  2048

×  \times

2048 samples generated using a grafted model (Sec.

5  ).
 (e) Depth

→  \rightarrow

width case study.  Samples generated using a model restructured via grafting (depth: 28

→  \rightarrow

14) (Sec.

6  ).

The space of architectural editing is vast, raising a practical question: what types of replacements should we study?
We first establish a  self-grafting baseline , where we replace all MHA and MLP operators in DiT-XL/2 with randomly initialized counterparts.
Despite the scale of this intervention, our grafting procedure achieves near-baseline model quality using under 1% of pretraining compute.
Building on this,
we focus on replacing existing operators with efficient alternatives, aiming to reduce model FLOPs while preserving quality. We also explore replacements that increase model FLOPs to examine broader design choices.
To study this systematically, we construct a  testbed  based on DiT-XL/2 and define a set of architectural edits to evaluate how different grafting schemes affect model quality.
We organize our design space along four axes:
(1) which operator to replace (e.g., MHA, MLP);
(2) what to replace it with (e.g., convolutions);
(3) how to select layers for replacement (e.g., all layers); and
(4) replacement ratio (full vs. partial).
We focus on replacing MHA and MLP operators, as they account for a large fraction of model FLOPs.
Replacements for MHA and MLP operators are motivated by empirical findings and prior architectural designs: our locality analysis supports local operators for MHA, while for MLP, we adopt ideas from prior work

fu2023monarch

;

komatsuzakisparse

;

kaplan2020scaling

.

We validate our grafting approach in increasingly challenging generative modeling setups:

Result I: Grafting yields hybrid architecture designs with good quality for class-conditional image generation (Sec.

4.2  ). 
We validate grafting using our testbed.
For MHA (softmax attention), we explore several alternatives: local gated convolution (Hyena-SE, and our proposed Hyena-X/ Hyena-Y), local attention (sliding window), and linear attention (Mamba-2).
For MLPs, alternatives