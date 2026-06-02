[2604.06938] POS‑ISP: Pipeline Optimization at the Sequence Level for Task‑aware ISP

POS‑ISP: Pipeline Optimization at the Sequence Level for Task‑aware ISP

Jiyun Won

Heemin Yang

Woohyeok Kim

Jungseul Ok

Sunghyun Cho

POSTECH CSE &amp; GSAI {w1jyun, heeminid, woohyeok, jungseul, s.cho}@postech.ac.kr

Abstract

Recent work has explored optimizing image signal processing (ISP) pipelines for various tasks by composing predefined modules and adapting them to task-specific objectives. However, jointly optimizing module sequences and parameters remains challenging. Existing approaches rely on neural architecture search (NAS) or step-wise reinforcement learning (RL), but NAS suffers from a training-inference mismatch, while step-wise RL leads to unstable training and high computational overhead due to stage-wise decision-making.
We propose POS-ISP, a sequence-level RL framework that formulates modular ISP optimization as a global sequence prediction problem. Our method predicts the entire module sequence and its parameters in a single forward pass and optimizes the pipeline using a terminal task reward, eliminating the need for intermediate supervision and redundant executions. Experiments across multiple downstream tasks show that POS-ISP improves task performance while reducing computational cost, highlighting sequence-level optimization as a stable and efficient paradigm for task-aware ISP. The project page is available at  https://w1jyun.github.io/POS-ISP

1  Introduction

Image signal processors (ISPs) transform RAW sensor data captured by digital cameras into sRGB images suitable for human perception or machine vision.
Conventional ISPs apply a fixed chain of operations such as white balance and tone mapping that are primarily designed to enhance image quality.
While such fixed pipelines are suitable for general photography, they often fail to align with the preferences or objectives of specific tasks, ranging from visual appearance optimization to high-level vision tasks such as object detection and semantic segmentation.
Although ISPs can be manually tuned by golden-eye experts, the process is time-consuming and difficult because it requires precise adjustment of many tightly coupled parameters for each task.
As a result, it is difficult to achieve consistent and optimal performance across different objectives.

To obtain enhanced ISP pipelines for downstream tasks, data-driven approaches have recently been proposed that learn ISPs directly from data.
Among these, modular approaches have attracted particular attention due to their practical advantages.
They decompose the ISP pipeline into well-established operations such as white balance and denoising, and optimize the pipeline in a task-driven manner.
This modular design is especially appealing because the operations are already integrated into imaging systems and have low computational complexity, making them suitable for practical deployment  [  5  ,

6  ] .
However, despite this efficiency, optimizing modular ISPs remains difficult, since selecting the best sequence of modules and tuning parameters often requires non-differentiable search procedures.

To address this challenge, several approaches have recently explored neural architecture search (NAS) or reinforcement learning (RL) for modular ISP optimization  [  27  ,

21  ,

24  ] .
While they resolve the issue of non-differentiable optimization, they also introduce new limitations.
First, the NAS-based method  [  27  ]  enables gradient-based optimization by mixing the outputs of candidate modules.
However, the reliance on mixture training causes inconsistency at inference, where the modules are discretely selected.
Second, RL-based methods  [  21  ,

24  ]  model ISP optimization as a stepwise RL formulation that performs sequential decision-making at each intermediate stage of the ISP pipeline.
Unfortunately, such a formulation requires repeated evaluations and relies on future reward estimation, resulting in unstable training and high computational overhead.
This instability is a well-known issue in deep reinforcement learning, arising from the difficulty of stabilizing bootstrapped value estimation under function approximation  [  10  ,

23  ] .
Moreover, since the decision process must be repeatedly evaluated at each stage to determine the next action, this stepwise formulation is structurally inefficient.

In this paper, we present  POS-ISP , a novel RL framework for searching optimal modular ISP pipelines tailored to downstream tasks. Unlike existing RL methods that make stepwise decisions, POS-ISP performs sequence-level optimization by evaluating the entire pipeline with a single final reward. This formulation enables direct evaluation of the final result, avoiding unstable future reward estimation and leading to more stable optimization. It also captures dependencies between ISP modules, allowing the policy to consider the global pipeline context when predicting module sequences. Furthermore, POS-ISP predicts the entire pipeline in a single forward pass, significantly reducing memory and computation. Such efficiency is essential for ISPs deployed on mobile or edge devices, where they must function as lightweight pre-processing components.

To enable sequence-level, context-aware optimization, POS-ISP adopts a carefully designed network to predict the module sequence, named  sequence predictor , along with  parameter predictor  for predicting module parameters.
The sequence predictor is a recurrent policy network that predicts the entire module sequence by leveraging contextual information from preceding modules.
At each recurrent step, the sequence predictor takes the previously selected module along with the hidden state, which contains contextual information of preceding modules, and predicts a probability distribution over the module candidates.
Thanks to this context-aware and lightweight recurrent design, POS-ISP can predict the module sequence with reduced computational cost and memory overhead while considering the dependencies between the modules.
In parallel, parameter predictor predicts module parameters with a small encoder–decoder network conditioned on the input image, enabling image-adaptive parameter prediction.
The predicted module sequence and its corresponding parameters together form a complete ISP pipeline, whose output image is evaluated based on task-driven performance.

We validate POS-ISP by measuring its task-specific performance after optimization for multiple tasks, including object detection, instance segmentation, and image enhancement.
Extensive experiments demonstrate that POS-ISP outperforms other task-aware ISP optimization methods both quantitatively and qualitatively, with a lower computational cost and memory footprint.

Our main contributions can be summarized as follows:

•

We introduce POS-ISP, a framework that performs sequence-level optimization of the ISP pipeline by predicting the entire pipeline in a single forward pass, directly optimizing the final task reward without relying on unstable stepwise supervision.

•

We design a recurrent sequence predictor that enables sequence-level prediction while capturing inter-module dependencies for context-aware optimization.

•

We evaluate POS-ISP on object detection, instance segmentation, and image enhancement. Extensive experiments demonstrate that POS-ISP achieves state-of-the-art performance with substantially reduced computational cost and memory usage.

2  Related Work

With the advancement of deep learning, several works have been proposed to replace conventional ISPs with end-to-end deep neural networks  [  14  ,

1  ,

2  ,

9  ,

26  ,

28  ] .
They aim to design neural networks that learn RAW-to-RGB mappings.
Thanks to strong image priors, they have shown promising performance in mimicking the ISPs.

Beyond directly learning RAW-to-RGB mappings, several works have explored optimizing ISP configurations for downstream tasks