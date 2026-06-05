[2506.01392] Sparse Imagination for Efficient Visual World Model Planning

Sparse Imagination
 for Efficient Visual World Model Planning

Junha Chun  1  ,  Youngjoon Jeong  2

1

1  footnotemark:

1

,  Taesup Kim  2

1 Department of Electrical and Computer Engineering, Seoul National University

2 Graduate School of Data Science, Seoul National University

These authors contributed equally.Corresponding author.

Abstract

World model based planning has significantly improved decision-making in complex environments by enabling agents to simulate future states and make informed choices.
However, ensuring the prediction accuracy of world models often demands substantial computational resources, posing a major challenge for real-time applications.
This computational burden is particularly restrictive in robotics, where resources are severely constrained.
To address this limitation, we propose a  Sparse Imagination for Efficient Visual World Model Planning , which enhances computational efficiency by reducing the number of tokens processed during forward prediction.
Our method leverages a sparsely trained vision-based world model based on transformers with randomized grouped attention strategy, allowing the model to adaptively adjust the number of tokens processed based on the computational resource.
By enabling sparse imagination (rollout), our approach significantly accelerates planning while maintaining high control fidelity.
Experimental results demonstrate that sparse imagination preserves task performance while dramatically improving inference efficiency, paving the way for the deployment of world models in real-time decision-making scenarios.

1  Introduction

By “imagining” future trajectories in a learned world model of the environment, agents can perform sophisticated decision making without trial-and-error in the real world

hafner2019planet  ;  hafner2019dreamer

. Recent world model advancements for visual control tasks, which necessitate reasoning over high-dimensional image observations, increasingly leverage high-level visual features derived from self-supervised vision models

caron2021dino  ;  oquab2023dinov2  ;  nair2022r3m  ;  xiao2022mvp

.
For example, Zhou et al.

zhou2024dinowm

introduces a world model that predicts future Vision Transformer (ViT) patch tokens (DINO features) instead of pixels or single vector representation, enabling zero-shot planning for diverse tasks. By retaining spatially rich visual tokens, such approaches achieve strong generalization on control tasks with complex observations. A downside, however, is that using dozens or hundreds of visual tokens per image incurs high, quadratic computational cost during inference, especially if many rollout simulations are needed for planning each action. This raises a crucial question:

Can we retain the advantages of detailed visual world models while enhancing computational efficiency for planning?

In this paper, we address the above question by introducing the concept of  sparse imagination , accelerating world model inference by deliberately using only a sparse subset of visual tokens during forward prediction. Our key insight is that ViT-based image representations contain redundancies

raghu2021vision  ;  pan2021ia  ;  chen2022principle  ;  kim2024learning

–not all patch tokens are necessary for forward prediction or decision making.

To achieve this, we propose a random dropout-based token selection mechanism for the world model during inference that dynamically encodes and predicts only a subset of patch tokens.
Furthermore, we train the world model using randomized grouped attention to accommodate dynamic token selection.
The resulting system performs planning on the subset of latent patch tokens, reducing computational cost while maintaining competitive performance.
The primary contributions of this work are as follows:

•

We introduce a simple yet effective training and inference method for world models that leverages patch feature dropout to enable sparse visual imagination.

•

Our sparse imagination approach, which utilizes randomly sampling token subsets, achieves substantial inference speedup while preserving competitive task performance across diverse visual control tasks compared to full-patch and single-vector baselines.

•

Our work offers a thorough experimental analysis explaining the operational principles behind the success of the sparse imagination with planning framework.

•

We show through a comparative study that simple random dropout performs comparably to or better than sophisticated token selection and merging methods.

2  Related Work

World Models for Control.

World model based decision making, enabling agents to simulate futures for informed choices, has been a long-standing goal in reinforcement learning and robotics

kaelbling1996planning  ;  wm

. The field transitioned from computationally heavy pixel-level models

finn2016video  ;  ebert2018visualforesight

to latent dynamics models like PlaNet

hafner2019planet

and Dreamer

hafner2019dreamer  ;  hafner2020dreamerv2  ;  hafner2023dreamerv3

, which achieved impressive results by leveraging planning and latent imagination. However, most existing latent world models compress entire images into relatively low-dimensional vectors. While compact, these representations can struggle to capture the fine-grained spatial information necessary for high-precision tasks, as evidenced by their performance on manipulation benchmarks

zhou2024dinowm  ;  tsagkas2025PVR

.

Visual Representations for Decision Making.

A recent trend in decision making leverages powerful pre-trained visual encoders for rich state representations

nair2022r3m  ;  xiao2022mvp

. While effective for representation learning, methods using a single image embedding (e.g., CLS token) from encoders like ResNet or ViT often lose crucial fine-grained spatial details, hindering performance on tasks requiring precise spatial understanding, such as manipulation

tsagkas2025PVR  ;  zhou2024dinowm

. To address this, approaches using spatial features like ViT patch embeddings for decision making have been developed, exemplified by DINO-WM

zhou2024dinowm

which uses a set of DINO patch tokens as the state. Building on this paradigm, our work utilizes pre-trained ViT patch features for their spatial granularity. Our key innovation is an inference-time mechanism to drop these tokens during imagination, significantly enhancing computational efficiency. This motivates our central question: how can we retain the effectiveness of rich patch-based representations while substantially reducing computational cost on decision masking?

Figure 1 :

Sparse Imagination.  We propose an accelerated strategy for visual world model based planning with sparse imagination, leveraging a patch token dropout mechanism to achieve efficient model predictive control (MPC) rollouts during inference. At each iteration of MPC, a random subset of visual patches is selected, and predictions are generated exclusively based on these sampled patches. Actions are optimized using the Cross-Entropy Method (CEM) by evaluating their predicted outcomes against the goal in latent space, calculated solely from the selected patches. This random dropout pattern is dynamically updated at every MPC iteration, thereby enhancing the model’s robustness.

Vision Transformers and Token Efficiency.

The quadratic scaling of ViT self-attention poses a significant computational challenge, particularly for sequential models like world models

dosovitskiy2020vit

. Improving inference efficiency via token reduction is thus a key vision research area. Existing methods employ diverse strategies such as learned/attention-based selection

rao2021dynamicvit  ;  luo2024ltrp  ;  yin2022avit  ;  Liang2021evit  ;  wang2022vtclfc  ;  Zhang2023star

, merging

bolya2022tome  ;  feng2023efficient  ;  haurum2024agglomerative

, and training-time dropout

liu2023patchdropout

. These demon