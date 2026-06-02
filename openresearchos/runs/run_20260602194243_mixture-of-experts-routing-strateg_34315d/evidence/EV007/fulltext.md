[2509.17238] MoEs Are Stronger than You Think: Hyper-Parallel Inference Scaling with RoE

MoEs Are Stronger than You Think:

Hyper-Parallel Inference Scaling with RoE

Soheil Zibakhsh  1

1  1 Work done during an internship at Apple.

2

2  2 Corresponding authors:  szibakhshshabgahi@ucsd.edu, m_samraghrazlighi@apple.com

1,2 ,
Mohammad Samragh  2

2  footnotemark:

2

1  ,
Kumari Nishu 1  ,
Lauren Hannah 1  ,

Arnav Kundu 1  ,  and
Minsik Cho 1

1 Apple,  2 University of California San Diego

Abstract

The generation quality of large language models (LLMs) is often improved by utilizing inference-time sequence-level scaling methods (e.g., Chain-of-Thought). We introduce  hyper-parallel scaling , a complementary framework that improves prediction quality at the token level.
Hyper-parallel scaling computes and aggregates multiple output proposals for a single token from the model.
We implement this concept in Mixture-of-Experts (MoE) models, which we refer to as Roster of Experts (RoE).
RoE is a training-free inference algorithm that turns a single MoE into a dynamic ensemble of MoEs.
RoE injects controlled stochasticity into the expert routing mechanism, enabling it to sample multiple diverse experts for each token and aggregate their outputs for a more accurate final prediction.
To overcome the computational cost, we introduce an efficient batching strategy and a specialized KV-caching mechanism that minimizes compute and memory overhead.
For example, RoE enables a 7B MoE model to match the performance of a 10.5B MoE model while using 30% less compute for inference. These gains are achieved without any fine-tuning of model parameters.

1  Introduction

Figure 1:  A categorization of inference-time scaling strategies.  (I) Sequential Scaling:  Enhancing performance by generating longer, structured outputs like a chain of thought  (Wei et al.,  2022 ) .  (II) Parallel Scaling:  Generating multiple token sequences and aggregating them, as in Self-Consistency  (Wang et al.,  2022 ) .  (III) Hyper-Parallel Scaling:  A novel paradigm, instantiated by RoE, that aggregates results from diverse internal computation paths on a per-token basis.

Extensive data and substantial computational resources have fueled recent advancements in language models.
While the simplest method for generating responses is greedy decoding, the quality of model outputs often requires enhancement at inference time. A growing line of work in this area focuses on test-time scaling, which aims to improve the performance of the sequence generation process.
Existing test-time scaling approaches typically fall into two orthogonal categories: sequential scaling, where the model produces longer, more structured outputs (e.g., Chain-of-Thought  (Wei et al.,  2022 ) ); and parallel scaling, where multiple independent sequences are generated and then aggregated (e.g., self-consistency  (Wang et al.,  2022 ) ). The general notion of these categories is marked as “Sequential Scaling” and “Parallel Scaling” in Figure

1  .

In this paper, we pose an orthogonal question: Can we improve a model’s intrinsic next-token prediction capability by allocating more computation at inference time? In other words, can we increase the model’s internal compute during inference to enhance the quality of every generated token? We refer to this new paradigm as  hyper-parallel scaling , as shown in Figure

1  . This approach improves generation quality even under the simplest decoding strategy, greedy decoding. To isolate the gains attributable to hyper-parallel scaling, we focus our experiments on evaluating greedy decoding quality throughout the paper.

Hyper-parallel scaling aims to unlock a model’s full potential by increasing the computation allocated to each token at inference time. One way to realize this idea is by introducing controlled variation within each transformer block  (Shelmanov et al.,  2021 )  and recomputing the layer output multiple times. Another approach is to reuse each layer repeatedly in a recurrent manner, thereby increasing computation without adding parameters  (Lin et al.,  2022 ) . While many variants are possible, we focus on sparsely activated Mixture-of-Experts (MoE) models, which provide an ideal architecture for implementing this concept.

Mixture of experts (MoE) models have become a leading solution for frontier large language models  (Shazeer et al.,  2017 ; Comanici et al.,  2025 ; Dai et al.,  2024 ) . Since they activate only a fraction of their parameters per forward pass, they naturally raise the central question of hyper-parallel scaling: can the inactive experts be leveraged at inference time to boost performance? Simply increasing the number of active experts does not work, as models are not trained to aggregate information from larger expert sets. To address this, we propose Roster of Experts (RoE), a training-free inference technique that treats a single MoE as a dynamic ensemble. RoE adds controlled stochasticity into the router’s expert selection, runs multiple stochastic forward passes per token, and aggregates the resulting logits into a single, higher-quality prediction, all without model fine-tuning.

As is evident, a naive implementation of RoE would incur substantial redundant computation. We address this by exploiting the overlap across forward passes and merging them into a single batched call to the LLM. Furthermore, we introduce a specialized caching mechanism to reduce the KV-cache size required for RoE generation.
In short, the contributions in this work are as follows:

•

We introduce hyper-parallel scaling, a novel inference paradigm that allocates additional compute at test time to diversify a model’s internal computations, thereby improving the quality of each token prediction.

•

We propose Roster of Experts (RoE), a training-free approach to hyper-parallel scaling in MoE models that ensembles diverse computational paths. RoE leverages Gumbel-Top-K routing to inject controlled stochasticity into expert selection and introduces execution and KV-cache optimizations for efficient inference.

•

We demonstrate the superior efficiency of RoE compared to conventional model scaling. For instance, we demonstrate that RoE can enhance the OlMoE-7B model  (Muennighoff et al.,  2024 )  to achieve the performance of a 10.5B model, with a 30% latency decrease compared to its larger counterpart. The enhancement requires no model finetuning.

2  RoE: Hyper Parallel Scaling of Mixture of Experts

Figure 2:  An illustration of the Roster of Experts (RoE) method.  Left:  For a single input,

n  n

distinct experts are sampled by adding stochasticity to the expert routing at each MoE layer, and the resulting output logits are aggregated to form the final prediction.  Right:  A closer view of a single MoE layer shows

k  =  2

k=2

active experts (dark orange), where Gumbel noise (dark blue) is added to the router logits, and the top-

k  k

experts are selected based on these modified logits.

Roster of Experts (RoE) enhances a pre-trained MoE model’s performance by treating it as a dynamic ensemble. In designing RoE, we hypothesize that making controlled variations in routing still yields high-quality predictions. The rationale behind our claim is straightforward: during training, the model already encounters a wide range of expert combinations, so there is no reason not to exploit the same diversity at test time.

Given the aforementioned insight, we provide a high-level illustration of RoE in Figure

2  . At each generation step, the MoE model generates multiple candidate output logits for a single input by sampling diverse expert selections from the model. These outputs are then aggregated to produce a single, more accurate prediction. This process relies on two key components: a stochastic routing mechanism to create diverse paths, and an efficient inference strategy to ensure practicality.

2.1  Gumbel-Top-K Routing for Path Diversity

Standard MoE model