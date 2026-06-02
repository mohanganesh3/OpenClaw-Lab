[2206.03382] 1 Introduction

marginparsep has been altered.
 topmargin has been altered.
 marginparwidth has been altered.
 marginparpush has been altered.

The page layout violates the ICML style. 
Please do not change the page layout, or include packages like geometry,
savetrees, or fullpage, which change it for you.
We’re not able to reliably undo arbitrary changes to the style. Please remove
the offending package(s), or layout-changing commands and try again.

Tutel: Adaptive Mixture-of-Experts at Scale

Anonymous Authors  1

Abstract

Sparsely-gated mixture-of-experts (MoE) has been widely adopted to scale deep learning models to trillion-plus parameters with fixed computational cost. The algorithmic performance of MoE relies on its token routing mechanism that forwards each input token to the right sub-models or  experts . While token routing dynamically determines the amount of expert workload at runtime, existing systems suffer inefficient computation due to their  static execution , namely static parallelism and pipelining, which does not adapt to the dynamic workload.

We present  Tutel , a highly scalable stack design and implementation for MoE with dynamically adaptive parallelism and pipelining.
 Tutel  designs an identical layout for distributing MoE model parameters and input data, which can be leveraged by
switchable parallelism and dynamic pipelining methods without
mathematical inequivalence or tensor migration overhead.
This enables adaptive parallelism/pipelining optimization at  zero cost  during runtime.
Based on this key design,  Tutel  also implements various MoE acceleration techniques including Flexible All-to-All, two-dimensional hierarchical (2DH) All-to-All, fast encode/decode, etc. Aggregating all techniques,  Tutel  finally delivers
 4.96

×

\times

and  5.75

×

\times

speedup of a single MoE layer over 16 and 2,048 A100 GPUs, respectively, over the previous state-of-the-art.

Our evaluation shows that  Tutel  efficiently and effectively runs a real-world MoE-based model named SwinV2-MoE, built upon Swin Transformer V2, a state-of-the-art computer vision architecture. On efficiency,  Tutel  accelerates SwinV2-MoE, achieving up to

1.55  ×

1.55\times

and

2.11  ×

2.11\times

speedup in training and inference over Fairseq, respectively. On effectiveness, the SwinV2-MoE model achieves superior accuracy in both pre-training and down-stream computer vision tasks such as COCO object detection than the counterpart dense model, indicating the readiness of  Tutel  for end-to-end real-world model training and inference.

†

†  footnotetext:

1 Anonymous Institution, Anonymous City, Anonymous Region, Anonymous Country.
Correspondence to: Anonymous Author &lt;anon.email@domain.com&gt;.
Preliminary work. Under review by the
Machine Learning and Systems (MLSys) Conference. Do not distribute.

1  Introduction

In recent years,
the community has found that enrolling more model parameters is one of the most straight-forward but less sophisticated way to improve the performance of deep learning (DL) algorithms  (Kaplan et al.,  2020 ) .
However, model capacity is often limited by computing resource and energy cost  (Sharir et al.,  2020 ) .
To tackle this, sparsely-gated Mixture-of-Experts (MoE)  (Shazeer et al.,  2017 )  introduces a  sparse  architecture by employing multiple parallel sub-models called  experts , where each input is only forwarded to a few experts based on an intelligent gating function.
Unlike dense layers, this method scales the model capacity up at only sublinearly increasing computational cost.
Nowadays, MoE is one of the most popular approaches demonstrated to scale DNNs to trillion-plus parameters  (Fedus et al.,  2022 ) , paving the way for models capable of learning even more information.

While MoE-based algorithms open up a huge scale-up/out opportunity, the  dynamic nature of MoE  introduces fundamental system-side challenges that have not been seen before in most of previous DL algorithms and systems.
To be specific, each MoE layer consists of a certain number of parallel experts that are distributed over accelerators (GPUs in this work), where each GPU dispatches each input data to several best-fit experts according to an intelligent gating function and get the corresponding outputs back to combine them.
This implies that the workload of experts is fundamentally uncertain – it depends on input data and the gating function. Both of them change at every iteration in practice.
In our experiments (see

Figure

1  ), the workload changes up to

4.38  ×

4.38\times

in a single training and different layers have different workload.

Previous DL systems, including the latest MoE frameworks  (Lepikhin et al.,  2021 ; Ott et al.,  2019 ; Rajbhandari et al.,  2022 ; He et al.,  2022 ) , are mostly based on static runtime execution that does not fit dynamic MoE characteristics. The major pitfall comes from that experts often fail to leverage the best-performing parallelism because the optimal one differs depending on the dynamic workload.
It is non-trivial to dynamically adjust parallelism at runtime as it typically incurs a large redistribution overhead or GPU memory consumption in existing systems.
Other approaches such as  load balancing loss

Fedus et al. ( 2022 )  try to tackle this issue by manipulating the MoE algorithm, but it often harms model accuracy in our experiments (see

Section

2.1  ).

Figure 1:  Dynamically changing workload of MoE layers during an end-to-end training of the MoE version of Swin Transformer V2  Liu et al. ( 2021 ;  2022 )  thin-tiny (left) and base (right) models. The y-axis is the needed expert capacity at runtime, which indicates the amount of workload (see details in

Section

2.1  ). For a neat view, only the 1st, 4th, and 10th layers are shown out of 10 total MoE layers in the model.

This paper presents  Tutel , a system that thoroughly optimizes MoE at any scale by adaptive methods specialized for dynamic MoE workload.
The key mechanism is  adaptive parallelism switching  that dynamically switches the parallelism strategy at every iteration without any extra overhead of switching. Specifically, unlike existing systems that use different tensor layouts for different parallelism strategies, we leverage only a single distribution layout that covers all possibly optimal strategies. This frees the system from reformatting the input data or weights when we switch the parallelism strategy, hence zero-cost switching. Based on our communication cost analysis of all kinds of parallelism, we ensure that adaptive parallelism does not compromise the optimal parallelism strategy.

Tutel  is a fully implemented framework for diverse MoE algorithms at scale. Over the adaptive parallelism switching, it delivers several optimization techniques for efficient and adaptive MoE, including adaptive pipelining, the 2-dimensional hierarchical (2DH) All-to-All algorithm, fast encode/decode with sparse computation on GPU, etc.
 Tutel  has been open sourced on GitHub  1

1  1  https://github.com/microsoft/tutel

and already been integrated into Fairseq  Ott et al. ( 2019 )  and DeepSpeed  Microsoft ( 2023 ) . Our extensive experiments over Azure A100 clusters  Azure ( 2023 )  show that with 128 GPUs,  Tutel  delivers up to

3.11  ×

3.11\times

of MoE-layer speedup, and

1.55  ×

1.55\times

/

2.11  ×

2.11\times

speedup for end-to-end training / inference of a real-world model (SwinV2-MoE), compared to that of using the original Fairseq. For 2,048 GPUs, the MoE-layer speedup is further improved to

5.75  ×

5.75\times

.

Our key contributions are as follows:

•

Provide detailed analysis on the dynamic nature of MoE and following challenges in existing frameworks.

•

Propose adaptive parallelism switching that efficiently handles dynamic workload of MoE, which achieves

1.35  ×  ∼  14.57  ×

1.35\times\sim 14.57\times

speedup of a single MoE layer.

•

Aggregating all acceleration techni