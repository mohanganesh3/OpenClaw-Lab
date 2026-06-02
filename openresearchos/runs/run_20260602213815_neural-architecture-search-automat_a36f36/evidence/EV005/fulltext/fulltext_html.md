[2109.04925] Rapid Model Architecture Adaption for Meta-Learning

Rapid Model Architecture Adaption for Meta-Learning

Yiren Zhao

University of Cambridge

yaz21@cam.ac.uk

&amp;Xitong Gao

Shenzhen Institute of Advanced Technology, CAS

xt.gao@siat.ac.cn

&amp;Ilia Shumailov

University of Cambridge

is410@cam.ac.uk

&amp;Nicolo Fusi

Microsoft Research

fusi@microsoft.com

&amp;Robert Mullins

University of Cambridge

Robert.Mullins@cl.cam.ac.uk

Abstract

Network Architecture Search (NAS) methods have recently gathered much attention.
They design networks with better performance
and use a much shorter search time compared to traditional manual tuning.
Despite their efficiency in model deployments,
most NAS algorithms target a single task on a fixed hardware system.
However, real-life few-shot learning environments often cover a great number of tasks (

T

𝑇

T

)
and deployments on a wide variety of hardware platforms (

H

𝐻

H

).

The combinatorial search complexity

T  ×  H

𝑇  𝐻

T\times H

creates
a fundamental search efficiency challenge if one naively applies existing NAS methods to these scenarios.
To overcome this issue, we show, for the first time, how to rapidly adapt model architectures to new tasks
in a  many-task many-hardware  few-shot learning setup by integrating Model Agnostic Meta Learning (MAML)
into the NAS flow.
The proposed NAS method (H-Meta-NAS)
is hardware-aware and performs optimisation in the MAML framework.
H-Meta-NAS shows a Pareto dominance compared to a variety of NAS and manual baselines in popular few-shot learning benchmarks with various hardware platforms and constraints.
In particular, on the 5-way 1-shot Mini-ImageNet classification task,
the proposed method
outperforms the best manual baseline
by a large margin (

5.21  %

percent  5.21

5.21\%

in accuracy) using

60  %

percent  60

60\%

less computation.

1  Introduction

Existing Network Architecture Search (NAS) methods show promising performance on image  (Zoph and Le,  2016 ; Liu et al.,  2018 ) , language  (Guo et al.,  2019 ; So et al.,  2019 )  and graph data  (Zhao et al.,  2020 ) .
The automation not only reduces the human effort required for architecture tuning but also produces architectures with state-of-the-art performance in domains like image classification  (Zoph and Le,  2016 )  and language modeling  (So et al.,  2019 ) .
Most NAS methods today focus on a single task with a fixed hardware system, yet real-life model deployments covering multiple tasks and various hardware platforms will significantly prolong this process.
As illustrated in

Figure

1  , a common design flow is to re-engineer the architecture and train for different task(

T

𝑇

T

)-hardware(

H

𝐻

H

) pairs with different constraints (

C

𝐶

C

).
The architectural engineering phase can be accomplished whether manually or by using an established NAS procedure.
The major challenge is designing an efficient algorithmic method to overcome the quickly scaling

𝒪  ​

(

T  ​  H  ​  C

)

𝒪

𝑇  𝐻  𝐶

\mathcal{O}(THC)

search complexity described in

Figure

1  .

Figure 1:

Deploying networks in a  many-task many-device  few-shot learning setup. This implies a large search complexity

𝒪  ​

(

T  ​  H  ​  C

)

𝒪

𝑇  𝐻  𝐶

\mathcal{O}(THC)

.

Few-shot learning systems follow exactly this  many-task many-device  setup,
when considering deployments on different user devices on key applications
such as facial  (Guo et al.,  2020 )  and speech recognition  (Hsu et al.,  2020 ) .
A task in few-shot learning normally takes an

N

𝑁

N

-way

K

𝐾

K

-shot
formulation, where it contains

N

𝑁

N

classes with

K

𝐾

K

support samples and

Q

𝑄

Q

query samples in each class.
Model-Agnostic Meta-Learning (MAML), incorporating the idea of learning to learn, builds a meta-model using a great number of training tasks, and then adapts the meta-model to unseen test tasks using only a very small number of gradient updates  (Finn et al.,  2017 ) .
MAML then becomes a powerful and elegant approach for few-shot learning – its ability to quickly adapt to new tasks can potentially shrink the

𝒪  ​

(

T  ​  H  ​  C

)

𝒪

𝑇  𝐻  𝐶

\mathcal{O}(THC)

complexity illustrated in

Figure

1

to

𝒪  ​

(

H  ​  C

)

𝒪

𝐻  𝐶

\mathcal{O}(HC)

.
In the meantime, hardware-aware NAS methods  (Cai et al.,  2019 ,  2018 ; Xu et al.,  2020 ) ,  e.g.  the train-once-for-all technique  (Cai et al.,  2019 ) , support deployments of searched models to fit to different hardware platforms with various latency constraints.
These hardware-aware NAS techniques further reduce the search complexity from

𝒪  ​

(

T  ​  H  ​  C

)

𝒪

𝑇  𝐻  𝐶

\mathcal{O}(THC)

to

𝒪  ​

(  T  )

𝒪  𝑇

\mathcal{O}(T)

(Cai et al.,  2018 ) .

In this paper, we propose a novel Hardware-aware Meta Network Architecture Search (H-Meta-NAS).
Integration of the MAML framework into hardware-aware NAS theoretically reduces the search complexity from

𝒪  ​

(

T  ​  H  ​  C

)

𝒪

𝑇  𝐻  𝐶

\mathcal{O}(THC)

to

𝒪  ​

(  1  )

𝒪  1

\mathcal{O}{(1)}

, allowing for a rapid adaption of model architectures to unseen tasks on new hardware systems.
However, we identified the following challenges in this integration:

•

Classic NAS search space contains many over-parameterised sub-models, this makes it hard to tackle the over-fitting phenomenon in few-shot learning.

•

Hardware-aware NAS profiles latency for sub-networks on each task-hardware pair, this profiling can be prolonged significantly with a great number of tasks and, more importantly, if the targeting device has scarce computation resources.

To tackle these challenges, we then propose to use Global Expansion (GE) and Adaptive Number of Layers (ANL) to allow a drastic change in model capabilities for tasks with varying difficulties. Our experiments later demonstrate that such changes alleviate over-fitting in few-shot learning and improve the accuracy significantly.
We also present a novel layer-wise profiling strategy to allow reuse of profiling information across different tasks.

In this paper, we make the following contributions:

•

We propose a novel Hardware-aware Network Architecture Search for Meta learning (H-Meta-NAS). H-Meta-NAS quickly adapts meta-architectures to new tasks with hardware-awareness and can be conditioned with various device-specific latency constraints.
The proposed NAS reduces search complexity from

𝒪  ​

(

T  ​  H  ​  C

)

𝒪

𝑇  𝐻  𝐶

\mathcal{O}(THC)

to

𝒪  ​

(  1  )

𝒪  1

\mathcal{O}(1)

in a realistic  many-task many-device  few-shot learning setup. We extensively evaluate H-Meta-NAS on various hardware platforms (GPU, CPU, mCPU, IoT, ASIC accelerator) and efficiency constraints (latency and model size), our latency-accuracy performance curve demonstrates a pareto dominance.

•

We propose a task-agnostic layer-wise profiling strategy for the NAS. This profiling reduces the profiling run-time from around

10  5

superscript  10  5

10^{5}

hours to 1.2 hours when targeting hardware with limited capabilities ( e.g.  IoT devices).

•

We show several tricks for the NAS algorithm, named Global Expansion and Adaptively Number of Layers respectively. These methods help the NAS to overcome the over-fitting problem in few-shot learning from the architectural perspective.

2  Related work

Few-shot learning in the MAML framework

Inspired by human’s ability to learn from only a few tasks and generalise the knowledge to unseen problems, a meta learner is trained over a distribution of tasks with the hope of generalising its learned knowledge to new tasks  Finn et al. ( 2017 ) .

arg  ​  min

θ

⁡

(

𝔼

𝒯  ∈  𝕋

​

[

ℒ  θ

​

(  𝒯  )

]

)

subscript

arg  min

𝜃

subscript  𝔼

𝒯  𝕋

delimited-[]

subscript  ℒ  𝜃

𝒯

\operatorname*{arg\,min}_{\theta}(\mathbb{E}_{\mathcal{T}\in\mathbb{T}}[\mathcal{L}_{\theta}(\mathcal{T})])

(1)

Equation

1

captures the optimisation objective of meta