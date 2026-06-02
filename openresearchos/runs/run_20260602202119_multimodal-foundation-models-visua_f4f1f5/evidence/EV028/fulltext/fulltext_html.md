[2503.19404] LangBridge: Interpreting Image as a Combination of Language Embeddings

LangBridge: Interpreting Image as a Combination of Language Embeddings

Jiaqi Liao  1∗  , Yuwei Niu  6,8∗  , Fanqing Meng  5,1∗  , Hao Li  2,1  , Changyao Tian  2,1  , Yinuo Du

Yuwen Xiong  1  , Dianqi Li,
Xizhou Zhu  3,4  , Li Yuan  6,7  , Jifeng Dai

3  ,

1  ​

🖂

{}^{3,1\textsuperscript{\Letter}}

, Yu Cheng

2  ​

🖂

{}^{2\textsuperscript{\Letter}}

1

OpenGVLab, Shanghai AI Laboratory

2

The Chinese University of Hong Kong

3

Tsinghua University

4

SenseTime Research

5

Shanghai Jiao Tong University

6

Peking University

7

PengCheng Laboratory

8

Chongqing University

Abstract

Recent years have witnessed remarkable advances in Large Vision-Language Models (LVLMs), which have achieved human-level performance across various complex vision-language tasks. Following LLaVA’s paradigm, mainstream LVLMs typically employ a shallow MLP for visual-language alignment through a two-stage training process: pretraining for cross-modal alignment followed by instruction tuning. While this approach has proven effective, the underlying mechanisms of how MLPs bridge the modality gap remain poorly understood. Although some research has explored how LLMs process transformed visual tokens, few studies have investigated the fundamental alignment mechanism. Furthermore, the MLP adapter requires retraining whenever switching LLM backbones. To address these limitations, we first investigate the working principles of MLP adapters and discover that they learn to project visual embeddings into subspaces spanned by corresponding text embeddings progressively. Based on this insight, we propose LangBridge, a novel adapter that explicitly maps visual tokens to linear combinations of LLM vocabulary embeddings. This innovative design enables pretraining-free adapter transfer across different LLMs while maintaining performance. Our experimental results demonstrate that a LangBridge adapter pre-trained on Qwen2-0.5B can be directly applied to larger models such as LLaMA3-8B or Qwen2.5-14B while maintaining competitive performance. Overall, LangBridge enables interpretable vision-language alignment by grounding visual representations in LLM vocab embedding, while its plug-and-play design ensures efficient reuse across multiple LLMs with nearly no performance degradation. See our project page at  https://jiaqiliao77.github.io/LangBridge.github.io/ .

†

†  ∗ Equal contribution.

1  Introduction

Figure 1 :

Comparison of different connector types in LVLM:  (a) The MLP directly maps visual features into the LLM’s text embedding space. (b) The Ovis method uses a visual embedding table to produce structural visual embeddings and align the modalities. (c) LangBridge decomposes visual features into weighted combinations of LLM’s vocabulary vectors to form the visual embeddings

Recent years have witnessed remarkable progress in Large Vision-Language Models (LVLMs)  [  1  ,

18  ,

53  ,

21  ,

15  ,

5  ,

39  ,

17  ] , which have demonstrated impressive capabilities in complex vision-language tasks, showcasing sophisticated understanding and reasoning across both visual and textual modalities. As a result, these models are increasingly being deployed in real-world applications, ranging from autonomous driving to robotics  [  31  ,

3  ,

24  ] .

As shown in Figure

1

(a), mainstream Large Vision-Language Models follow the architecture of LLaVA  [  18  ] , leveraging Vision Transformers (ViTs) for visual feature extraction and Multi-Layer Perceptrons (MLPs) to transform these features into visual embeddings aligned with the LLM’s text embedding dimension. These models also adopt LLaVA’s two-stage training process: (1) MLP warm-up for cross-modal alignment and (2) full-model instruction tuning. While this approach has proven effective, two critical challenges remain: 1) the underlying mechanisms of how MLPs bridge the modality gap are poorly understood. Although some research has explored how LLMs process transformed visual tokens  [  12  ,

49  ] , few studies have investigated the fundamental alignment mechanisms of MLPs, and 2) the MLP adapter requires retraining whenever switching LLM backbones due to input dimension mismatch and distributional shifts in feature representations.

To address these limitations, we systematically investigate the working principles of MLP adapters through two critical aspects: 1) how visual semantics are encoded in transformed embeddings, and 2) how MLPs learn cross-modal alignment during training. Our methodology employs visual analysis across four distinct training stages: 1) Pretrain-100 steps, 2) Pretrain-1000 steps, 3) Pretrain-2000 steps, and 4) SFT-tuned Model. For each stage, we extract visual features using Vision Transformers, transform them through stage-specific MLPs into visual embeddings, and compute cosine similarities between these visual embeddings and text embeddings. Through this process, we identify the top semantically related text tokens for each visual embedding at each training phase.

As visualized in Figure

2

through circular graphs where nodes represent different training phases spanning from early to late stages (e.g. 1- 4), we can draw two key conclusions: 1. Visual embeddings demonstrate strong correlations with semantically-related text tokens. As shown in Figure

2  , green apple image patch exhibits high similarity with text tokens such as “Green Apple”. Similarly, the silhouetted sunset scene correlates with concepts like “Eclipse Sun”. This suggests that MLPs effectively map visual features into the text embedding space of LLMs, particularly in regions close to the semantic spaces formed by the corresponding text embeddings. 2. The projection capability of MLPs develops progressively during training. Figure

2

illustrates this evolution through multiple semantic pathways: from “numerous” to ”five”, from ”Arts” to “Green Apple”, and in the right scene, from “We Plat” to “Shakespeare”, from “Kiss” to “Wed Kiss”. These relationships become increasingly well-defined through the training stages, confirming our hypothesis that MLPs gradually learn to map visual features into regions proximate to their corresponding text embedding subspaces.

Building upon this insight, we introduce an explicit transformation approach called  Language Basis Vector Projection . This approach explicitly projects visual features into the text embedding subspace of the LLM by representing them as linear combinations of the LLM’s vocabulary embeddings. Based on this theoretical foundation, we propose  LangBridge , a novel adapter architecture. The adapter first maps visual features to probability distributions that capture their semantic similarity with vocabulary text embeddings in the LLM. Subsequently, it generates the final visual embeddings by combining these text embeddings, weighted according to the computed probabilities. This innovative design elegantly addresses the challenge of model retraining when switching between different LLM backbones. The key insight is that our adapter only learns the linear combination relationship between visual patches and vocabulary embeddings, rather than directly mapping between different embedding dimensions. Since the adapter’s output is always a probability distribution over a fixed vocabulary, it can be used to weigh and combine the corresponding vocabulary embeddings of any target LLM, regardless of their embedding dimensions.

Extensive experiments across multiple benchmarks and LLMs demonstrate the effectiveness of our approach. A key advantage of LangBridge is its ability to enable pre-training-free transfer between different LLMs. Specifically, LangBridge trained on smaller models (e.g., Qwen2-0.5B or LLaMA-3-8B) can be directly applied to larger models like Qwen2-7B and Qwen2.5-7B during supervised fine-tuning while maintaining comparable performance. This capability significantly reduces comp