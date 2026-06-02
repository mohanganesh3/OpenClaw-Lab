[2007.09380] CATCH: Context-based Meta Reinforcement Learning for Transferrable Architecture Search

1

1  institutetext:  The University of Hong Kong

2

2  institutetext:  Huawei Noah’s Ark Lab

3

3  institutetext:  Sun Yat-sen University

4

4  institutetext:  The Hong Kong University of Science and Technology

CATCH: Context-based Meta Reinforcement Learning for Transferrable
Architecture Search

Xin Chen

Equal contribution.11

Yawen Duan  ⋆

11

Zewei Chen

22

Hang Xu

22

Zihao Chen

22

Xiaodan Liang

33

Tong Zhang

Correspondence to: tongzhang@tongzhang-ml.org44

Zhenguo Li

22

Abstract

Neural Architecture Search (NAS) achieved many breakthroughs in recent years. In spite of its remarkable progress, many algorithms are restricted
to particular search spaces. They also lack efficient mechanisms to
reuse knowledge when confronting multiple tasks. These challenges
preclude their applicability, and motivate our proposal of CATCH,
a novel Context-bAsed meTa reinforcement learning (RL) algorithm for transferrable
arChitecture searcH. The combination of meta-learning and RL allows CATCH to efficiently adapt to new tasks while being agnostic to search spaces. CATCH utilizes a probabilistic encoder to encode task properties into latent context variables, which then guide CATCH’s controller to quickly  \say catch top-performing networks. The contexts also assist a network
evaluator in filtering inferior candidates and speed up learning. Extensive
experiments demonstrate CATCH’s universality and search
efficiency over many other widely-recognized algorithms. It is also
capable of handling cross-domain architecture search as competitive
networks on ImageNet, COCO, and Cityscapes are identified. This is
the first work to our knowledge that proposes an efficient transferrable
NAS solution while maintaining robustness across various settings.

Keywords:  Neural Architecture Search, Meta Reinforcement Learning

1  Introduction

The emergence of many high-performance neural networks has been one
of the pivotal forces pushing forward the progress of deep learning
research and production. Recently, many neural networks discovered
by Neural Architecture Search (NAS) methods have surpassed manually
designed ones on a variety of domains including image classification  [ 47 ,  61 ] , object detection  [ 61 ] , semantic segmentation  [ 5 ] , and recommendation systems  [ 31 ] .
Many potential applications of practical interests are calling for
solutions that can (1) efficiently handle a myriad of tasks, (2) be
widely applicable to different search spaces, and (3) maintain their
levels of competency across various settings. We believe these are
important yet somewhat neglected aspects in the past research, and
a transformative NAS algorithm should be able to respond to these
needs to make a real influence.

Figure 1:  Upper: drawbacks of current NAS schemes. Lower: the overall framework
of CATCH. Our search agent, CATCHer, consists of three core
components: context encoder, RL controller and network evaluator.
CATCHer first goes through the meta-training phase to learn an initial
search policy, then it adapts to target tasks efficiently.

Many algorithms  [ 33 ,  37 ]  have been proposed
to improve the efficiency of NAS. However, they lack mechanisms to
seek and preserve information that can be meaningfully reused. Hence,
these algorithms can only repeatedly and inefficiently search from
scratch when encountering new tasks. To tackle this problem, a rising
direction of NAS attempts to create efficient transferrable algorithms.
Several works  [ 23 ,  36 ]  try to search
for architectures that perform well across tasks, but the solutions
may not be optimal on the target tasks, especially when the target task distributions
are distant from the training task distributions. Some recent works  [ 28 ,  15 ] 
use meta-learning  [ 16 ,  27 ]  for one-shot NAS instead. With recent critiques  [ 56 ,  26 ] 
pointing out some one-shot solutions’ dependence on
particular search spaces and sensitivity to hyperparameters, many
concerns arise on the practicality of these meta NAS works based on
one-shot methods. To avoid ambiguity, throughout this paper,  tasks  are defined as problems that share the same action space, but differ in reward functions. In NAS, the change of either the dataset or domain (e.g. from classification to detection) alters the underlying reward function, and thus can be treated as different tasks.

Striking a balance between universality and efficiency is hard.
Solving the universality problem needs a policy to disentangle from
specifics of search spaces, which uproots an important foundation
of many efficient algorithms. The aim to improve efficiency on multiple
tasks naturally links us to a transfer/meta learning paradigm. Meta
Reinforcement Learning (RL)  [ 38 ,  25 ] 
offers a solution to achieving both efficiency and universality,
which largely inspired our proposal of CATCH, a novel context-guided
meta reinforcement learning framework that is both search space-agnostic
and swiftly adaptive to new tasks.

The search agent in our framework, namely CATCHer, acts as the decision-maker
to quickly  \say catch top-performing
networks on a task. As is shown in Figure

1  , it is first
trained on a set of meta-training tasks then deployed to target tasks
for fast adaptation. CATCHer leverages three core components: context
encoder, RL controller, and network evaluator. The context encoder
adopts an amortized variational inference approach  [ 1 ,  38 ,  24 ] 
to encode task properties into latent context variables that guide
the controller and evaluator. The RL controller makes sequential decisions
to generate candidate networks in a stochastic manner. The network
evaluator predicts the performance of the candidate networks and decides
which nets are valuable for training. All three components are optimized
in an end-to-end manner.

We test the method’s universality and adaptation efficiency on two
fundamentally different search spaces: cell-based search space  [ 13 ] 
and Residual block-based  [ 19 ,  57 ]  search space.
The former focuses on cell structure design, while the latter targets
macro skeleton search. With NAS-Bench-201  [ 13 ] , we
can compare CATCH fairly with other algorithms by eliminating performance
fluctuations rising from different search spaces and training settings.
Our experiments demonstrate CATCH’s superiority over various other
works, including R-EA  [ 40 ]  and DARTS  [ 33 ] .
On Residual block-based search space, we use image classification tasks on sub-datasets of ImageNet  [ 10 ] 
as meta-training tasks, and then adapt the CATCHer to target tasks, such as image classification on full ImageNet, object detection on COCO  [ 30 ] , and semantic segmentation on Cityscapes  [ 9 ] .
CATCH discovered networks on these tasks with competitive performance
and inference latency. Our results demonstrated CATCH’s
robustness across various settings, easing previously raised concerns
of NAS algorithms’ sensitivity to search space, random seeds, and
tendencies to overfit to only one or two reported tasks.

Our key contribution is the first attempt to design an efficient and
universal transferrable NAS framework. It swiftly handles various
tasks through fast adaptation, and robustly maintains competitive
performance across different settings. Our work brings along new perspectives
on solving NAS problems, including using amortized variational inference
to generate task characteristics that inform network designs. It also
demonstrates the possibility of creating efficient sample-based NAS
solutions that are comparable with widely-recognized one-shot methods.
With competitive networks identified across classification, detection,
and segmentation domains, it further opens the investigation on the
feasibility of cross-domain architecture search.

2  Related Work

NAS is an algorithmic approach to design neural networks through searching
over cand