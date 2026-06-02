[2105.11694] FNAS: Uncertainty-Aware Fast Neural Architecture Search

\floatsetup

[table]capposition=top
 \newfloatcommand capbtabboxtable[][ \FBwidth ]

FNAS: Uncertainty-Aware Fast Neural Architecture Search

Jihao Liu

SenseTime X-Lab

CUHK - SenseTime Joint Lab, The Chinese University of Hong Kong

Ming Zhang

SenseTime X-Lab

Yangting Sun

SenseTime X-Lab

Boxiao Liu

SenseTime X-Lab

Yu Liu

SenseTime X-Lab

CUHK - SenseTime Joint Lab, The Chinese University of Hong Kong

Hongsheng Li

CUHK - SenseTime Joint Lab, The Chinese University of Hong Kong

School of CST, Xidian University

Abstract

Reinforcement learning (RL)-based neural architecture search (NAS)

generally

guarantees better convergence yet suffers from

the requirement of

huge computational resources compared with gradient-based approaches, due to the

rollout bottleneck

– exhaustive training of each sampled architecture on the proxy tasks.

In this paper, we propose

a general pipeline to accelerate the convergence of the rollout process as well as the RL process in NAS. It is motivated by the interesting observation that both the architecture and the parameter knowledge can be transferred between different search processes and even different tasks.
We first introduce an uncertainty-aware critic (value function) in Proximal Policy Optimization (PPO)

[

27

]

to take advantage of the architecture knowledge in previous search processes, which stabilizes the training process and reduce the searching time by 4 times.
In addition, an architecture knowledge pool together with a block similarity function is proposed to utilize parameter knowledge and reduces the searching time by 2 times. To the best of our knowledge, this is the first method that introduces a block-level weight sharing scheme in RL-based NAS. The block similarity function guarantees a 100% hit ratio with strict fairness

[

5

]

.
Besides, we show

that

an off-policy correction factor used in “replay buffer” of RL optimization can further reduce half of the searching time.
Experiments on the Mobile Neural Architecture Search (MNAS)

[

30

]

search space show that the proposed Fast Neural Architecture Search (FNAS) accelerates the standard RL-based NAS process by

∼

similar-to

\sim

10x (e.g., 20,000 GPU hours to 2,000 GPU hours for MNAS), and guarantees better performance on various vision tasks.

1  Introduction

The architecture of a convolutional neural network (CNN) is crucial for many deep learning tasks such as image classification

[

31

]

and object detection

[

32

]

. The widespread use of neural architecture search (NAS) methods such as differentiable, one-shot,
evolutional, and RL-based approaches have effectively dealt with architecture design problems.
Despite having high performance due to its sampling-based mechanism

[

30  ,

41  ,

31

]

, RL-based NAS tends to require unbearable computing resources which discourages the research community from exploring it further.

The main obstacles to the propagation of RL-based NAS algorithm come from the following two aspects: a) it’s necessary to sample a large number of architectures from the search space to ensure the convergence of the RL agent, b) the inevitable training and evaluation cost of these

architecture

samples on proxy tasks. For example, the seminal RL-based NAS

[

40

]

approach requires 12,800 generations of architectures. The state-of-the-art MNAS

[

30

]

and MobileNet-V3

[

12

]

require 8000 or more generations to

find

the optimal architecture.
Coupled with

∼

similar-to

\sim

5 epochs training for each generation, the whole search process costs nearly 64 TPUv2 devices for 96 hours or 20,000 GPU hours on V100 for just one single searching process. With no access to reduce the unbearable computational cost, RL-based NAS is hard to make more widespread influence than differential

[

21  ,

4

]

, and one-shot based

[

1  ,

9

]

methods.

On the contrary, the high efficiency of one-shot NAS family brings it continuous research attention. Instead of sampling a huge number of sub-networks, one-shot NAS assembles them into a single super-network. The parameters are shared between different sub-networks during the training of the super-network. In this way, the training process is condensed from training thousands of sub-networks into training a super-network. However, this

weight sharing

strategy may

cause

problems

of inaccurate

performance estimation of sub-networks. For example, two sub-networks may propagate conflicting gradients to their shared components,

which

may converge to favor one of the sub-networks and repel the other

one

randomly. This conflicting phenomenon may result in instability of the search process and inferior final architectures, compared with RL-based methods.

In this work, we

aim at combining advantages

of both RL-based methods and one-shot methods.
The proposed method is based on two

important  key observations

:
First, the optimal architectures for different tasks have

certain

common architecture knowledge (similar sub-architectures in different search processes’ optimal architectures).
Second, the parameter knowledge (weights at samples’ training checkpoints) can also be transferred across different searching settings and even tasks.

Based on the two observations, to transfer architecture knowledge, we develop Uncertainty-Aware Critic (UAC) to learn the architecture-performance joint distribution from previous search processes in an unbiased manner, utilizing the transferability of the architecture knowledge, which reduces the needed samples in RL optimization process by 50%.
For the transferable

parameter knowledge  , we propose an Architecture Knowledge Pool (AKP) to restore the block-level

[

30

]

parameters and fairly share them as new sample architectures’ initialization, which speed up each sample’s convergence for

∼

similar-to

\sim

2 times.
Finally, we also develop an Architecture Experience Buffer (AEB) with an off-policy correctness factor to store the previously trained models for reusing in RL optimization, with half of the search time saved. Under the same environment as MNAS

[

30

]

with MobileNet-v3

[

12

]

, FNAS speeds up the search process by 10

×

\times

and the searched architecture performs even better.

To summarize, our main contributions are as follows:

1.

We propose FNAS, which introduces three acceleration modules, uncertainty-aware critic, architecture knowledge pool, and architecture experience buffer, to speed up reinforcement-learning-based neural architecture search by

∼

similar-to

\sim

10

×

\times

.

2.

We show that the knowledge of neural architecture search processes can be transferred, which is utilized to improve sample efficiency of reinforcement learning agent process and training efficiency of each sampled architecture.

3.

We demonstrate new state-of-the-art accuracy on ImageNet classification, face recognition, and COCO object detection with comparable computational constraints.

Figure 1 :

The pipeline of FNAS. The proposed modules are highlighted in orange. Architectures are sampled by the RL agent and then passed to Uncertainty-Aware Critic (UAC) for predicting performance and the corresponding uncertainty. Then a decision module will determine whether the sample needs to be trained by Trainer. The Architecture Knowledge Pool (AKP) helps to initialize new samples for training. Half of the samples in one batch come from Architecture Experience Buffer (AEB), the other half come from Trainer or UAC’s Value Network.

2  Related Works

From the perspective of how to

estimation the performance

of architectures, NAS methods can be

classified

into two categories, sampling-based and

weight-sharing  -based

methods  .

Sampling-based methods

generally sample a large number of

architectures from the architecture search space and train them independently. Based on the

evaluated

performance of the