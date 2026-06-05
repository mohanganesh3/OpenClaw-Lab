[2402.18443] LeMo-NADe: Multi-Parameter Neural Architecture Discovery with LLMs

LeMo-NADe: Multi-Parameter Neural Architecture Discovery with LLMs

Md Hafizur Rahman

Department of Electrical and Computer Engineering
 University of Maine
 Orono, ME 04469

md.hafizur.rahman@maine.edu

&amp;Prabuddha Chakraborty

Department of Electrical and Computer Engineering
 University of Maine
 Orono, ME 04469

prabuddha@maine.edu

Abstract

Building efficient neural network architectures can be a time-consuming task requiring extensive expert knowledge. This task becomes particularly challenging for edge devices because one has to consider parameters such as power consumption during inferencing, model size, inferencing speed, and  \ch CO2 emissions. In this article, we introduce a novel framework designed to automatically discover new neural network architectures based on user-defined parameters, an expert system, and an LLM trained on a large amount of open-domain knowledge. The introduced framework (LeMo-NADe) is tailored to be used by non-AI experts, does not require a predetermined neural architecture search space, and considers a large set of edge device-specific parameters. We implement and validate this proposed neural architecture discovery framework using CIFAR-10, CIFAR-100, and ImageNet16-120 datasets while using GPT-4 Turbo and Gemini as the LLM component. We observe that the proposed framework can rapidly (within hours) discover intricate neural network models that perform extremely well across a diverse set of application settings defined by the user.

1  Introduction

Neural networks have found extensive application across various fields such as healthcare  [ 1 ,  2 ,  3 ] , surveillance  [ 4 ,  5 ] , Industry 4.0  [ 6 ,  7 ,  8 ] , and Internet of Things (IoT)  [ 9 ,  10 ,  11 ] . A neural network can be composed of a large number of layers of different types while sporting diverse hyperparameters. Hence, for a given task: (1) finding the right set of neural layers; (2) connecting them in the right topology; and (3) selecting the most optimal hyperparameters for each layer can be a daunting task requiring a large amount of computation resources, human expert involvement, and time. Requiring a given neural network to perform under specific resource-constrained conditions (a case for many IoT/Edge devices) can add to the complexity of the neural architecture search process. For example designing a neural network to have more than

x  %

percent  𝑥

x\%

accuracy for a given task is a hard problem to solve but it becomes harder if we further constrain the problem with additional parameters such as frames-per-second requirements during inferences and power consumption limits.

Traditional neural architecture search (NAS) frameworks are typically designed to identify the optimal architecture within a specified search space. This approach, although powerful, is limited by the said pre-defined search space failing to innovate and create wildly new architectures. Traditional NAS techniques are designed to primarily focus on improving task accuracy with very little emphasis on additional parameters such as frames-per-second (FPS), power consumption, and green house gas emission that have become more relevant in recent years.

To mitigate these concerns we propose a  L arge Languag e

Mo del guided  N eural  A rchitecture  D iscov e ry (LeMo-NADe) framework that can allow the discovery of novel neural network architecture without relying on a pre-defined search space. This will be achieved through an iterative approach utilizing a large language model (LLM) and an expert system for driving the LLM towards the target discovery. The expert system will use a set of configurable rules and several user-defined metrics to generate a set of instructions for the LLM leading to progressive refinement of the generated neural architecture.

To validate the LeMo-NADe framework, we perform extensive experimentation using the CIFAR-10, CIFAR-100, and ImageNet16-120 datasets. We use the framework to generate many different neural networks for diverse application requirements and priorities. Neural networks generated using LeMo-NADe for CIFAR-10 (

89.41  %

percent  89.41

89.41\%

test accuracy) and CIFAR-100 (

67.90  %

percent  67.90

67.90\%

test accuracy) showed near state-of-the-art level performance. For ImageNet16-120 LeMo-NADe was also able to generate fairly competitive architectures (

31.02  %

percent  31.02

31.02\%

test accuracy). LeMo-NADe is also very efficient (time, energy consumption, and  \ch CO2 Emissions) in terms of model generation/training. While using GPT-4 Turbo  1

1  1  gpt-4-1106-preview

[ 12 ]  (as the backend LLM), LeMo-NADe was able to generate and train CIFAR-100 models in about 4.81 hours consuming only about 0.50 kWh-PUE energy. LeMo-NADe is also capable of prioritizing other metrics (besides accuracy) towards creating neural architecture that are optimal for different IoT/Edge requirements (e.g. high speed low accuracy inferencing) and more importantly, has created novel neural architectures from scratch creating a new opportunity for search-space agnostic neural architecture search research.

To summarize, we offer the following contributions:

1.

Formalize and design a search-space agnostic neural architecture discovery framework (LeMo-NADe) leveraging Large Language Models (LLMs).

2.

Propose an expert system with associated rules and relevant metrics that is capable of driving a given LLM towards discovering different neural architectures.

3.

Implement LeMo-NADe as a highly configurable/efficient tool for immediate application and easy future extensions.

4.

Qualitatively and quantitatively evaluate LeMo-NADe using CIFAR-10, CIFAR-100, and ImageNet16-120 datasets for diverse settings and application requirements.

2  Background and Motivation

In this section, we briefly describe the recent advances in the domain of neural architecture search. We also highlight the motivation behind our work stemming from different shortcomings of traditional NAS frameworks.

2.1  Neural Architecture Search

Methods of Neural Architecture Search (NAS) extensively applied across various research areas, such as image processing  [ 13 ,  14 ,  15 ] , signal processing  [ 16 ,  17 ,  18 ] , object detection  [ 19 ,  20 ] , and natural language processing [ 21 ,  22 ] . It involves identifying the best neural network (judged traditionally based on only accuracy) for a given task through repeated trials.
The early NAS techniques worked mainly based on the evolutionary algorithms (EA)  [ 23 ]  and reinforcement learning (RL)  [ 24 ] . Although these methods showed promising result by building quality networks they require high computing power and time. To solve this issue, weight-reusing  [ 25 ]  approaches were proposed that avoids the necessity of training each design from the beginning resulting in low computation cost. One-shot approaches for NAS  [ 26 ]  were also proposed which involves training a large network called SuperNet that incorporates every conceivable architecture within the search domain.
DNAS  [ 27 ]  is another weight re-using approach where all the SubNet parameters are optimized by gradient decent.

NAS is hard to reproduce due to its high computational power and time. To limit this issue researchers proposed NAS benchmark dataset that contains all the possible architecture with corresponding evaluation results. One NAS dataset is NAS-Bench-101  [ 28 ]  that contains 5 million distinct neural architectures and is trained on CIFAR-10 dataset. The NAS-Bench-201  [ 29 ]  dataset has 15625 cell layouts and is derived from a cell-based search technique and is trained on CIFAR-10  [ 30 ] , CIFAR-100  [ 30 ]  and ImageNet16-120  [ 31 ]  datasets. In  [ 32 ] , the authors proposed a NAS method named as

β

𝛽

\beta

-DARTS to solve weak generalization ability found in DARTS method. They used the NAS-Bench-201 to evaluate th