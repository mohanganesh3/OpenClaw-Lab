[2510.23479] MergeMix: A Unified Augmentation Paradigm for Visual and Multi-Modal Understanding

MergeMix: A Unified Augmentation Paradigm for Visual and Multi-Modal Understanding

Xin Jin  1,⋆

Siyuan Li  1,2,⋆

Siyong Jian

1

Kai Yu

1

Huan Wang  1,†

1 Westlake University, Hangzhou, China

2 Zhejiang University, College of Computer Science and Technology, Hangzhou, China
 {jinxin86; lisiyuan; wanghuan}@westlake.edu.cn

⋆  \star

Equal contribution

†  \dagger

Corresponding author

https://github.com/JinXins/MergeMix

Abstract

Vision–language alignment in multi-modal large language models (MLLMs) typically relies on supervised fine-tuning (SFT) or reinforcement learning (RL). SFT is stable and efficient but requires large-scale human annotations and cannot capture subtle preferences, while RL brings in a reward signal for training, but suffers from overhead and instability. These limitations highlight a trade-off between scalability, robustness, and alignment quality. To address this, we propose MergeMix, a training-time augmentation paradigm that bridges SFT and RL. It first applies an attention-aware image mixing via token merge with more cluster representation and spatial context, and then presents a preference-driven training paradigm for MLLMs by building preference pairs with mixed images and raw images, and optimizing via SimPO loss. As a mixup augmentation, MergeMix enhances attention consistency and efficiency, surpassing other heuristic-based methods in classification. Extensive experiments demonstrate that MergeMix achieves competitive accuracy with improved efficiency, providing a scalable approach to preference alignment in classification and MLLMs.

1  Introduction

Multi-modal Large Language Models (MLLMs)  (Liu et al.,  2024b ; Bai et al.,  2025 ; Tong et al.,  2024 )  have recently demonstrated remarkable capabilities in integrating visual and textual information, enabling a wide range of applications from visual question answering to multi-modal reasoning. Since MLLMs are typically pre-trained on massive web-scale datasets, forcing them to possess a wide range of knowledge and general reasoning capabilities, Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL)-based preference optimization  (Yang et al.,  2025b )  have emerged as two primary paradigms for aligning MLLMs with human preferences and specific task requirements. However, SFT depends on high-quality instruction–response annotations and optimizes the likelihood of reference responses, which does not explicitly model relative preferences between outputs. RL-based methods such as RLHF are more preference-aware, but they require an additional reward model that may encode bias or be exploited by reward.

Due to the shortcomings of data quality and training efficiency, some works  (Zhu et al.,  2024 ;  2025 ; Luo et al.,  2024 ; Tan et al.,  2025 ; Wang et al.,  2024 )  try to build performance pairs for optimization. How to build the preference pair with control and high-quality data for model training is the remaining open question. For example, SeVa  (Zhu et al.,  2024 )  proposed a preference optimization method by building a loser through some classic augmentation (i.e., RandomCrop). Then, select the different responses for optimizing the model by a DPO loss  (Rafailov et al.,  2023 ) . However, these methods have two drawbacks: the augmentations are highly random, and the DPO loss cannot be related to the data, which means SeVa can only select useful training data. Those technical causes SeVa can not control the quality of the loser, which is harmful for some visual question answering tasks, and reduces the training data by selecting “hard negatives”. Hence, we investigate an interesting question:  Is it necessary to propose novel techniques rather than some classical machine learning methods in the MLLM scenario?

In this paper, we revisit the mixup augmentation,
which synthesizes mixed samples and corresponding labels with given mixing ratios.
However, two main challenges arise as illustrated in Figure

1  : (1) achieving an optimal trade-off between efficiency and performance of mixup augmentations that rely on saliency-based metrics,
(2) extending the augmentation to MLLMs properly, from classical image corruptions to data-dependent samples.
Motivated by these perspectives, we propose a novel training framework called  MergeMix , which builds preference pairs for MLLM training through data augmentation methods and ranking loss, thereby bridging the gap between SFT and RL. Figure

2

shows the two scenarios of MergeMix.  (a)  We introduce MergeMix, a novel data augmentation that generates mixed samples through token merge techniques. A bipartite soft matching gathers the similarity information that brings the context, making the mask retain useful features. Meanwhile, MergeMix links the merge ratio and mixing ratio, aligning the information density of samples, enabling precise mixing data generation.  (b)  We propose a preference-driven SFT paradigm for MLLMs, where augmented samples are regarded as non-preferred responses ( Loser ) and clean samples as preferred responses ( Winner ). This enables preference optimization via SimPO loss  (Meng et al.,  2024 )  without relying on reward models. Altogether, Figure

1

shows these contributions yield an efficient and effective training strategy that achieves stronger alignment with human preferences while preserving the stability and scalability of SFT. Since the optimization object has a direct relationship with augmentation, it obtains a more robust ability in calibration.
Extensive experiments show that MergeMix, as a training-time augmentation paradigm, achieves competitive performance in both image classification and MLLM benchmark with favorable efficiency.

Our contributions can be summarized as:

(a)

We use the naive token merge to obtain a local clustered attention map and recover the initial shape by source map, which can generate the mixed image with more continuous features.

(b)

We design a training paradigm for supervised fine-tuning of MLLMs by generating augmented images as losers, building preference pairs with raw images, and optimization via ranking loss.

(c)

We validate that our method achieves state-of-the-art on several image classification datasets and benchmarks, along with the advantages of our training paradigm on several MLLM benchmarks.

Figure 1:

Efficiency and performance for MergeMix:  (a) The training time vs. accuracy of mixup methods with the DeiT-Small model. (b) The image classification Top-1 accuracy vs. training epochs of different mixup methods on the CIFAR100 dataset with the DeiT-Tiny model. (c) The radar plot of the results on part VQA tasks by LLaVA-7B, LLaVA with SFT, and MergeMix.

2  Related Work

In this section, we introduce the existing mixup approaches for image classification and token compression approaches in multi-modal large language models for efficient training or inference.

Mixup Augmentations

The Mixup method mitigates model overfitting by generating augmented samples through mixing two different images within a mini-batch. Broadly, Mixup methods can be categorized into two types:  Static , which relies on human priors or randomness, and  Adaptive , a data-dependent type that leverages certain metrics to guide the mixing process.  (i) Static:  MixUp  (Zhang et al.,  2017 )  generates mixed samples via linear interpolation with

λ  \lambda

. CutMix  (Yun et al.,  2019 )  extends this idea from the global pixel level to a local patch level by constructing a mask of size proportional to

λ  \lambda

to mix images. ResizeMix  (Qin et al.,  2020 )  ensures that features from at least one class are always preserved in the mixed sample by resizing the source image before mixing. Other methods, e.g., FMix  (Harris et al.,  2020 ) , SmoothMix  (Lee et al.,  2020 ) , GridMix  (Baek et al.,  2021 ) , and StarMix  (Jin et al.,  2024b ) , focus on impr