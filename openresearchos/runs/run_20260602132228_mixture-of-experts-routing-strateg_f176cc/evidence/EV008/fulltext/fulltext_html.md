[2602.05711] OmniMoE: An Efficient MoE by Orchestrating Atomic Experts at Scale

OmniMoE: An Efficient MoE by Orchestrating Atomic Experts at Scale

Jingze Shi

Zhangyang Peng

Yizhang Zhu

Yifan Wu

Guang Liu

Yuyu Luo

Abstract

Mixture-of-Experts (MoE) architectures are evolving towards finer granularity to improve parameter efficiency. However, existing MoE designs face an inherent trade-off between the granularity of expert specialization and hardware execution efficiency.
We propose  OmniMoE , a system–algorithm co-designed framework that pushes expert granularity to its logical extreme. OmniMoE introduces vector-level  Atomic Experts , enabling scalable routing and execution within a single MoE layer, while retaining a shared dense MLP branch for general-purpose processing.
Although this  atomic  design maximizes capacity, it poses severe challenges for routing complexity and memory access. To address these, OmniMoE adopts a system-algorithm co-design: (i) a  Cartesian Product Router  that decomposes the massive index space to reduce routing complexity from

O  ​

(  N  )

O(N)

to

O  ​

(

N

)

O(\sqrt{N})

; and (ii)  Expert-Centric Scheduling  that inverts the execution order to turn scattered, memory-bound lookups into efficient dense matrix operations. Validated on seven benchmarks, OmniMoE (with 1.7B active parameters) achieves 50.9% zero-shot accuracy across seven benchmarks, outperforming coarse-grained (e.g., DeepSeekMoE) and fine-grained (e.g., PEER) baselines. Crucially, OmniMoE reduces inference latency from 73ms to 6.7ms (a 10.9

×  \times

speedup) compared to PEER, demonstrating that massive-scale fine-grained MoE can be fast and accurate. Our code is open-sourced at  https://github.com/flash-algo/omni-moe .

Machine Learning, ICML

1  Introduction

Figure 1 :

Activation Patterns and System Optimization. 
(a) Coarse-grained MoE activates large experts, inevitably involving redundant parameters and wasting computation. (b) Fine-grained MoE improves parameter efficiency, but suffers from bandwidth bottlenecks due to scattered, fragmented memory accesses. (c) Our OmniMoE employs a universally activated shared dense MLP, and uses expert-centric scheduling to reorganize fine-grained expert fetches into contiguous, coalesced memory accesses, achieving both high parameter efficiency and hardware-efficient execution.

Mixture-of-Experts (MoEs) has emerged as a key approach to mitigating scaling bottlenecks by partially decoupling model capacity from per-token computation  (Fedus et al.,  2022 ) .
By activating only a subset of experts for each token, MoEs allow for massive parameter scaling while maintaining manageable inference budgets.
A central design choice in MoE is the  granularity  of experts, which largely determines both routing precision and system efficiency. Broadly, existing designs fall into two categories: coarse-grained MoEs and fine-grained MoEs.

Coarse-Grained MoEs. 
Coarse-grained architectures represent the dominant paradigm in contemporary large-scale language models. Representative systems  (Du et al.,  2022 ; Jiang et al.,  2024 ; Zoph et al.,  2022 )  such as DeepSeek-V3  (DeepSeek-AI et al.,  2025 )  (256 experts) and KIMI-K2  (Team et al.,  2025 )  (384 experts) instantiate each expert as a complete dense FFN, benefiting from hardware-efficient dense matmuls (via Tensor Cores), contiguous VRAM access, and
shared-expert general knowledge and training stability

(Dai et al.,  2024 ; Nguyen et al.,  2025 ; DeepSeek-AI et al.,  2025 ; Team et al.,  2025 ) .
Despite the success, coarse-grained MoEs inherently suffer from  imprecise activation

(Szatkowski et al.,  2024 )  and  low flexibility . Specifically, activating large expert blocks incurs computation on parameters irrelevant to specific tokens (orange nodes in Figure

1

(a)), leading to computational waste  (Cheng et al.,  2025 ; Li et al.,  2023 ; Szatkowski et al.,  2024 ) .
Moreover, their rigid size hinders adaptation to limited hardware:
coarse granularity restricts scaling flexibility, forcing steep, discrete memory increments, when adjusting expert counts.

Fine-grained MoEs.  Fine-grained architectures seek to maximize expressivity by utilizing millions of lightweight experts (e.g., embeddings).
MoE scaling-law analyses  (Ludziejewski et al.,  2024 ; Clark et al.,  2022 )  suggest that, under a fixed training-token budget, performance improves with the total number of activated experts.
This motivates  fine-grained  MoEs  (He,  2024 ; Nogueira dos Santos et al.,  2024 )  that use extra lightweight experts. For example, PEER  (He,  2024 )  scales to millions of experts by adopting a Product Key Memory (PKM  (Lample et al.,  2019 ) ) style design, enabling precise routing and fine-grained control over both model capacity and activated parameters through smooth scaling.

However, scaling fine-grained experts to massive magnitudes introduces three system challenges.
(i)  Limited expressivity:  existing designs (e.g., PEER  (He,  2024 ) ) reduce experts to static parameter vectors. This restricts the expert computation to linear vector aggregation, stripping away the token-dependent nonlinear transformations (e.g., MLP projections) essential for modeling complex linguistic dependencies.
(ii)  Routing overhead:  scaling to a large expert pool increases routing cost and load imbalance, resulting in skewed expert utilization at scale.
(iii)  Hardware inefficiency:  scattered activations trigger random memory I/O, shifting execution from compute-bound to memory-bound and degrading GPU utilization.
As illustrated in Figure

1  (b), while fine-grained experts ensure precise activation, the active parameters are inherently scattered across memory, which triggers frequent, non-contiguous memory accesses, inevitably shifting the execution bottleneck from computation to memory bandwidth.

While coarse-grained MoEs benefit from hardware-friendly architecture, the fine-grained ones leverage high activation efficiency and flexibility.
This raises a key question:  Is it possible to reconcile the parameter efficiency of fine-grained models with the hardware efficiency of coarse-grained architectures? 
Realizing this synergy is non-trivial. It requires a holistic orchestration that simultaneously enhances the expressivity of fine-grained experts, minimizing routing overhead in large expert spaces, and reshaping irregular sparse accesses into hardware-efficient execution.

Figure 2 :

Overview of the OmniMoE Architecture. 
The framework operates via two parallel pathways to balance efficiency and expressivity.
 (a) Dynamic Expert Assembly (Top):  For  Longtail Knowledge Retrieval  objective, we employ a  Cartesian Product Router  (decomposed into Row/Column routers) to efficiently compute routing scores

𝐠  x

\mathbf{g}_{x}

and identify the top-

K  K

expert indices

ℐ  x

\mathcal{I}_{x}

.
Then the system dynamically retrieves specific parameter slices from the global matrices

W  ,  V

W,V

to assemble compact, token-dependent parameter blocks

w  x

,

v  x

w_{x},v_{x}

for the final gated projection.
 (b) Shared Expert (Bottom):  A dense MLP which is always active to handling  General Semantics .
The final output is obtained by aggregating the outputs from the sparse, routed branch and the shared dense branch.

Our Methodology and Contributions.  To address the aforementioned challenges, we propose  OmniMoE , a system-algorithm co-designed MoE framework that synergizes the precise parameter activation of fine-grained experts with the hardware efficiency of coarse-grained designs. The core architectural innovation lies in a hybrid parallel design that combines a shared dense MLP for capturing general semantic knowledge, with a massive pool of routed fine-grained experts that specialize in long-tail knowledge retrieval. To orchestrate the activation of these fine-grained experts at scale, we introduce three tightly integrated contributions that jointly res