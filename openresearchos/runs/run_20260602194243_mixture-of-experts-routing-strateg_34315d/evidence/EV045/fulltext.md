[2511.12207] Mixture of States: Routing Token-Level Dynamics for Multimodal Generation

1]KAUST
2]Meta AI
 \contribution [†]Joint First Authors
 \contribution [∗]Core Contributors

Mixture of States: Routing Token-Level Dynamics for Multimodal Generation

Haozhe Liu

Ding Liu

Mingchen Zhuge

Zijian Zhou

Tian Xie

Sen He

Yukang Yang

Shuming Liu

Yuren Cong

Jiadong Guo

Hongyu Xu

Ke Xu

Kam-Woh Ng

Juan C. Pérez

Juan-Manuel Pérez-Rúa

Tao Xiang

Wei Liu

Shikun Liu

Jürgen Schmidhuber

[

[

haozhe.liu@kaust.edu.sa

dingliu@meta.com

( December 5, 2025 )

Abstract

We introduce MoS (Mixture of States), a novel fusion paradigm for multimodal diffusion models that merges modalities using flexible, state-based interactions. The core of MoS is a learnable, token-wise router that creates denoising timestep- and input-dependent interactions between modalities’ hidden states, precisely aligning token-level features with the diffusion trajectory. This router sparsely selects the top-

k  k

hidden states and is trained with an

ϵ  \epsilon

-greedy strategy, efficiently selecting contextual features with minimal learnable parameters and negligible computational overhead. We validate our design with text-to-image generation (MoS-Image) and editing (MoS-Editing), which achieve state-of-the-art results. With only 3B to 5B parameters, our models match or surpass counterparts up to

4  ×

4\times

larger. These findings establish MoS as a flexible and compute-efficient paradigm for scaling multimodal diffusion models.

\correspondence

;  (Project Lead)

MoS-Image

MoS-Edit

Figure 1 :

Generation examples by MoS-Image (left) and MoS-Edit (right).  MoS introduces a learnable, token-wise router that efficiently aggregates feature states across modalities. This allows for high-quality visual synthesis, producing photorealistic and stylized outputs from text and image inputs with precise control and quality.

1  Introduction

Multimodal generation is a fundamental application of modern AI, enabling models to synthesize high-quality visual content such as images and videos from conditional inputs  (Rombach et al.,  2022 ; Gao et al.,  2025 ; Baldridge et al.,  2024 ; Polyak et al.,  2024a ; Wu et al.,  2025a ; Liu et al.,  2025a ) . In this paper, we focus on  text-to-image generation  and  instruction-based image editing  tasks, a domain where a central challenge lies in  effectively aligning textual and visual signals . This problem is non-trivial, as the modalities rely on different modeling objectives: text models are typically trained with contrastive learning  (Radford et al.,  2021 ) , masked-token prediction  (Devlin et al.,  2019 ; Raffel et al.,  2020 ) , or next-token prediction  (Team et al.,  2024 ; Touvron et al.,  2023 ; Brown et al.,  2020 ) , whereas visual models often adopt diffusion-based generation  (Ho et al.,  2020 ; Neal,  2001 ; Jarzynski,  1997 )  or flow matching  (Lipman et al.,  2023 ; Esser et al.,  2024b ) . Consequently, alignment requires bridging not only heterogeneous representations but also distinct designs across modalities.

Prior studies address this challenge through various hand-crafted designs. These dominant fusion techniques, as well as the method we propose, are built upon the transformer architecture  (Vaswani et al.,  2017 ; Dosovitskiy et al.,  2020 ; Schmidhuber,  1992 ) , which serves as a powerful backbone for modeling both textual and visual representations. The primary strategies include: i)  Cross-attention  methods  (Rombach et al.,  2022 ; Vaswani et al.,  2017 ; Chen et al.,  2023 ; Xie et al.,  2024 )  insert new attention blocks into the visual model, projecting text embeddings onto key-value vectors to enable cross-modal token interactions. ii)  Self-attention  methods  (Esser et al.,  2024a ; Chen et al.,  2025a ; Qin et al.,  2025 )  instead concatenate text and visual tokens into a unified sequence, processed by shared attention layers. While this allows for deeper, bidirectional fusion than cross-attention and often yield stronger performance, its computational cost is often prohibitive, scaling quadratically with the combined sequence length; iii)  MoT (Mixture-of-Transformers) , a more recent method  (Deng et al.,  2025 ; Liao et al.,  2025 ; Liang et al.,  2025 ; Shi et al.,  2024 ) , establishing layer-wise cross-modal connections by sharing key–value vectors between corresponding text and visual blocks. This method facilitates a finer-grained interaction, but its rigid, layer-by-layer design imposes a strong architectural constraint: the text and visual backbones must be  symmetric, with a one-to-one block correspondence.

Through our ablations, we have identified three critical design principles for improving text-visual representation alignment that challenge the hand-crafted/fixed-interaction paradigms of prior work:

•

Layer selection should be adaptive, not fixed. 
We find that using a single fixed layer, typically the final-layer feature from the text branch, as commonly adopted in cross- and self-attention methods, is suboptimal. Furthermore, the rigid one-to-one layer correspondence of MoT assumes that text and visual features align symmetrically, an assumption for which we find no experimental support. This suggests that diffusion models do not consume language features in a strictly sequential or layer-aligned manner, making a  flexible selection  mechanism essential.

•

Conditional signals should be dynamic and timestep-dependent. 
We validate that the common design in modern text-to-image models, which encodes the text embedding once and keep it static, creates an "information mismatch" with the dynamic nature of the denoising process. We argue that the conditional guidance needs to  adapt as the input noise level and denoising step change .

•

Conditional signals should be token-specific. 
Our findings indicate that it is more effective to allow each token to source its representation adaptively from different layers, rather than using a single, shared layer embedding to represent all tokens uniformly. This supports a  more granular, token-level view  of context conditioning.

(a)

Cross-Attention

(b)

Self-Attention

(c)

MoT

(d)

MoS (Ours)

Figure 2 :

MoS enables sparse and dynamic interactions across modalities and transformers.  We illustrate MoS with text-to-image generation. Previous approaches, such as  (a) cross-attention  and  (b) self-attention , typically provide only the final text encoder block’s embedding as input to the visual branch, limiting the richness of cross-modal information.  (c) MoT (Mixture-of-Transformers)  attempts finer-grained interaction by passing outputs from all text blocks in a rigid, layer-by-layer fashion. In contrast, our proposed  (d) MoS (Mixture of States)  employs a learnable sparse interaction that dynamically links  any  text block to  any  visual block. The routing adapts to the current input, comprising the text prompt, visual latents, and denoising step embeddings, enabling flexible and efficient multimodal fusion.

To this end, we introduce Mixture of States (MoS), a new framework that enables multimodal interaction to adapt dynamically to the input and denoising step.
Unlike prior fixed-interaction designs, MoS grants the vision branch access to  all textual hidden states  across  all layers  and employs a learnable, token-wise router to selectively aggregate features at each denoising step. As shown in Fig.

2  , this sparse yet dynamic routing allows  any visual token, at any denoising step and within any transformer block, to attend to tokens from any layer of the text encoder.  This effectively bridges the gap between textual representations and visual diffusion dynamics.

We systematically explore the design space of MoS, including its input formulation, architectural configuration, and training strategy. Building on this foundation, we present a family of multimodal generation models that support two multimodal gene