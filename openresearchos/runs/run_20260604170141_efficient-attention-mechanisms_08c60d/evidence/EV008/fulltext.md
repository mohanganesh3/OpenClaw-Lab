[2502.12677] Spiking Vision Transformer with Saccadic Attention

Spiking Vision Transformer with Saccadic

Attention

Shuai Wang  1  , Malu Zhang  1  , Dehao Zhang  1  , Ammar Belatreche  2  ,  Yichen Xiao  1

,

Yu Liang  1

,  Yimeng Shan  3

,  Qian Sun  1

,  Enqi Zhang  1

,  Yang Yang  1

1  University of Electronic Science and Technology of China

2  Northumbria University,

3  Liaoning Technical University

Corresponding author: maluzhang@uestc.edu.cn

Abstract

The combination of Spiking Neural Networks (SNNs) and Vision Transformers (ViTs) holds potential for achieving both energy efficiency and high performance, particularly suitable for edge vision applications. However, a significant performance gap still exists between SNN-based ViTs and their ANN counterparts. Here, we first analyze why SNN-based ViTs suffer from limited performance and identify a mismatch between the vanilla self-attention mechanism and spatio-temporal spike trains. This mismatch results in degraded spatial relevance and limited temporal interactions. To address these issues, we draw inspiration from biological saccadic attention mechanisms and introduce an innovative Saccadic Spike Self-Attention (SSSA) method. Specifically, in the spatial domain, SSSA employs a novel spike distribution-based method to effectively assess the relevance between Query and Key pairs in SNN-based ViTs. Temporally, SSSA employs a saccadic interaction module that dynamically focuses on selected visual areas at each timestep and significantly enhances whole scene understanding through temporal interactions.
Building on the SSSA mechanism, we develop a SNN-based Vision Transformer (SNN-ViT). Extensive experiments across various visual tasks demonstrate that SNN-ViT achieves state-of-the-art performance with linear computational complexity. The effectiveness and efficiency of the SNN-ViT highlight its potential for power-critical edge vision applications.

1  Introduction

Vision Transformers (ViTs)  (Dosovitskiy,  2020 )  revolutionize the traditional computer vision field, achieving higher performance in many vision tasks such as image classification  (Chen et al.,  2021 ; Han et al.,  2023 )  and object detection  (Fang et al.,  2021c ; Touvron et al.,  2021 ) . However,
ViTs always demand significant computational and memory resources, which greatly restricts their deployment in resource-constrained edge vision environments  (Wu et al.,  2022 ; Graham et al.,  2021 ) .
Consequently, the development of energy-efficient and high-performance solutions remains a significant area of research that necessitates further investigation  (Cai et al.,  2019 ; Han et al.,  2020b ) .

Spiking Neural Networks (SNNs), as the third generation of neural networks  (Maass,  1997 ; Gerstner &amp; Kistler,  2002 ; Izhikevich,  2003 ; Masquelier et al.,  2008 ) , mimics biological information transmission mechanisms using discrete spikes as the medium for information exchange. Spiking neurons fire spikes only upon activation and remain silent at other times.
This event-driven mechanism  (Caviglia et al.,  2014 )  promotes sparse synapse operations and avoids multiply-accumulate (MAC) operations, which significantly boost the energy efficiency of these models  (Zhang et al.,  2023 ) .
However, the architectures of most SNN-based models still revolve around traditional structures such as CNNs  (Fang et al.,  2021b ; Xing et al.,  2019 )  and ResNets  (Fang et al.,  2021a ; Hu et al.,  2024 ) , which exhibit a significant performance gap compared to ViTs.

In recent years, numerous researchers have dedicated efforts to develop SNN-based ViT models. However, most studies  (Zhou et al.,  2023b ; Wang et al.,  2023b )  retain energy-intensive MAC operations in self-attention computational paradigm and not fully take advantage of SNNs’ energy efficiency. Furthermore, these approaches still rely on the Dot-Product operation to measure the spatial relevance between Query (

Q  Q

) and Key (

K  K

) pairs. However, they fail to account for whether the Dot-Product is well-suited to the binary spike characteristics of SNNs.
Subsequently, inspired by Metaformer  (Yu et al.,  2023 ) , Spike-driven V2  (Yao et al.,  2024b )  introduces a MAC-free method, and SpikingResformer  (Shi et al.,  2024 )  combines ResNet-based architecture and self-attention computation paradigm to further reduce parameters. These methods ensure the high performance of SNN-based ViTs while achieving a full spike-driven manner, offering significant energy savings. Nevertheless, these studies treat self-attention computational paradigm merely as an efficient token mixer  (Yu et al.,  2022 ) , without exploring an effective paradigm suited to spike trains. Furthermore, these methods primarily focus on spatial feature extraction, overlooking the temporal dynamics of SNNs. Consequently, exploring spiking self-attention paradigms tailored to the spatio-temporal characteristic of SNNs represents a potential area for improvement.

Biological vision dynamically captures and understands visual scenes through saccadic mechanisms  (Melcher &amp; Morrone,  2003 ; Binda &amp; Morrone,  2018 ; Guadron et al.,  2022 ) . It focuses on specific visual areas at each moment and utilizes dynamic saccadic movements across the temporal domain to achieve a contextual understanding of the entire visual scene  (Hanning et al.,  2023 ) . Compared to vanilla self-attention mechanisms  (Liu et al.,  2021b ) , it offers higher energy and computational efficiency.
Additionally, the saccadic process involves intense temporal interactions  (Idrees et al.,  2020 ) , which closely align with the unique temporal characteristics of SNNs.
Therefore, we draw inspiration from the saccadic mechanisms to design a Saccadic Spike Self-Attention (SSSA) method. The SSSA method adapts to the spatio-temporal characteristics of SNNs, enabling an efficient and effective comprehensive understanding of visual scenes.
Based on this, we further develop a SNN-based Saccadic Vision Transformer. The summary contributions are as follows:

•

We thoroughly analyze the reasons for the mismatch between the vanilla self-attention mechanism and SNNs. In the spatial domain, the binary and sparse nature of spikes creates significant magnitude differences between

Q  Q

and

K  K

in SNN-based ViTs, making it difficult for vanilla self-attention to assess spatial relevance. Additionally, vanilla self-attention is designed for ANNs and neglects the temporal interactions among timesteps in SNNs, limiting its ability to explore information in the temporal domain.

•

We propose a Saccadic Spike Self-Attention (SSSA) mechanism specifically designed for SNNs’ spatio-temporal characteristics. In the spatial domain, SSSA introduces a novel spike distribution-based method to measure relevance between

Q  Q

and

K  K

pairs effectively. Temporally, SSSA introduces a saccadic interaction module that dynamically focuses on selected visual areas and achieves a comprehensive understanding of the whole scene.

•

To further enhance the computational efficiency of SSSA, we introduce a linear complexity version called SSSA-V2. It is mathematically linear scaling mapping to SSSA, preserving all performance benefits. Additionally, SSSA-V2 successfully reduces computational complexity to a linear level and works in a fully event-driven manner.

•

Building on the proposed SSSA mechanisms, we develop a SNN-based Vision Transformer (SNN-ViT) architecture. Extensive experiments are conducted on various visual tasks demonstrating that SNN-ViT achieves SOTA performance with linear computational complexity. It presents a promising approach for achieving both high-performance and energy-efficient visual solutions.

2  Related Work

Vision Transformers:  ViTs segment images into patches and apply self-attention  (Vaswani,  2017 ; Kenton &amp; Toutanova,  2019 )  to learn inter-patch relationships, outperforming CNNs across multiple vis