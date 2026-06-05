[2602.03560] 1 Introduction

\titlefont

HySparse: A Hybrid Sparse Attention Architecture

with Oracle Token Selection and KV Cache Sharing

Yizhao Gao

Jianyu Wei

Qihao Zhang

Yu Cheng

Shimao Chen

Zhengju Tang

Zihan Jiang

Yifan Song

Hailin Zhang

Liang Zhao

Bo Yang

Gang Wang

Shijie Cao

Fuli Luo  ⋄

LLM-Core Xiaomi

\abscontent

0

0  footnotetext:

⋄  Corresponding author.

1  Introduction

The demand for long-context capabilities has become a cornerstone of modern Large Language Models (LLMs), driven by emerging paradigms such as test-time scaling  [ o1 ,  r1 ]  and agentic workflows  [ k2 ,  anthropic_building_effective_agents_2024 ] .
Yet, the self-attention mechanism in standard Transformers scales quadratically with sequence length, causing computational latency and cost to grow prohibitively as context length increases.

Sparse attention offers a straightforward and effective solution to mitigate this quadratic bottleneck  [ sparsetransformer ] .
Sparse attention computes attention over a selected subset of important tokens rather than all tokens in the sequence.
Existing methods can be broadly categorized into training-free and trainable approaches. Training-free methods rely on fixed patterns or heuristic to select important tokens  [ h2o ,  streamingllm ,  minference ,  quest ,  lserve ,  duo ] . Trainable sparse attention learns which tokens to attend during training, either via low-cost self-distillation  [ seerattn_v1 ,  dsa ]  or by being directly integrated into pre-training  [ nsa ,  moba ] . Nevertheless, sparse attention architectures still suffer from two fundamental limitations:

(1) Proxy-based Sparse Token Selection. 
Sparse attention fundamentally depends on a selection mechanism to identify important tokens prior to attention computation.
Existing methods typically rely on lightweight proxies, such as predefined patterns, heuristics, approximate estimates, or additional selection modules  [ minference ,  seerattn_v1 ,  moba ,  seer-r ,  nsa ,  dsa ] .
However, these proxies are inherently approximate and may fail to capture true token importance, particularly in long and evolving contexts.
As a result, sparse token selection is often bounded by the fidelity of the proxy, potentially limiting the expressive power of sparse attention.
While learnable sparse attention alleviates selection errors by learning token selection during training, it does not fundamentally eliminate the proxy-based bottleneck and introduces additional selection modules that increase training complexity.

(2) Computation Reduction without Memory Relief. 
Modern sparse attention methods increasingly adopt dynamic sparsity to preserve model fidelity. Unlike static patterns (e.g., fixed strides or block structures), which can reduce KV cache storage but often incur noticeable performance degradation, dynamic approaches typically retain the full KV cache. This is because complete KV cache eviction is irreversible and destructive, as token importance may shift as generation progresses and context evolves. While dynamic sparse attention can effectively reduce computation, it provides no relief for memory consumption. Maintaining a full-sized KV cache therefore remains a dominant bottleneck for serving throughput and maximum batch size, limiting the practical benefits of sparse attention in long-context settings.

To address these challenges, we introduce  Hybrid Sparse Attention (HySparse) .
The key idea is to interleave every full attention layer with multiple sparse attention layers, where the sparse layers strategically derive important token selection and KV caches from the preceding full layer.
This design is motivated by two empirical observations in recent literature: token saliency is stable across consecutive layers (§  2.3  ), and cross-layer KV cache sharing reduces memory without hurting performance (§  2.4  ).
In HySparse, full attention can precisely identify important token selection and already produces KV caches, which sparse layers can directly reuse.
By reusing the important token indices from full attention, sparse selection becomes oracle-guided.
This eliminates the need for auxiliary proxy modules and ensures stable end-to-end training.
By reusing the KV caches from full attention, sparse attention adds no per-layer KV overhead, effectively alleviating the memory pressure associated with dynamic sparse attention.
Meanwhile, inspired by hybrid sliding window attention (SWA) architectures  [ gemma3 ,  gpt-oss ,  xiao2026mimo ] , HySparse augments each sparse attention layer with an additional SWA branch that maintains a small, local KV cache to enhance short-range modeling capacity.

We evaluate HySparse on both 7B dense and 80B Mixture-of-Experts (MoE) model settings.
For the 7B dense model, we adopt a full-to-sparse layer ratio of 1:3, while for the 80B MoE model, a more aggressive 1:11 ratio is used. In both cases, the final layer employs full attention to preserve global aggregation.
Across tasks and context lengths, HySparse consistently outperforms both full attention and hybrid SWA baselines, without incurring any additional KV cache cost relative to the hybrid SWA baseline.
Remarkably, in the HySparse 80B MoE model with 49 total layers,  only 5 layers use full attention , meaning nearly

10  ×

10\times

KV cache reduction, while the models still delivers substantial performance gains.
Compared with Hybrid SWA, HySparse can significantly reduce the number of full attention layers, effectively pushing the hybrid ratio to its limit.
In summary, these results indicate that HySparse provides a simple and effective architectural solution to the core limitations of sparse attention, achieving strong long-context modeling capability with clear efficiency and memory advantages.

2  Background &amp; Motivation

2.1  Training-free vs. Trainable Sparse Attention

Sparse attention methods can be divided into training-free and trainable approaches.
Training-free methods rely on fixed patterns or heuristics to identify important tokens.
They are applied as a drop-in modification at inference, enabling fast sparsity decisions with minimal computational cost  [ streamingllm ,  quest ,  duo ,  h2o ] .
However, applying sparsity only at inference creates a training–inference mismatch, which may lead to error accumulation in long decoding or multi-step reasoning  [ hu2026lil ,  liu2025quantizationreasoning ,  he2025nondeterminism ] .
Trainable sparse attention methods, in contrast, learn token importance during training through lightweight selection modules.
By integrating sparsity into the training process, they improve alignment between training and inference, selecting more informative tokens with higher recall and overall accuracy  [ seerattn_v1 ,  seer-r ,  dsa ,  nsa ,  moba ,  minicpm4 ,  zhao2025infllm ] .
However, training these selection modules are non-trivial.
One approach uses auxiliary losses, such as self-distillation, to align the gating or indexer module with the original dense attention  [ seerattn_v1 ,  seer-r ,  dsa ] .
These methods are simple but suboptimal.
Alternatively, NSA  [ nsa ]  performs end-to-end sparse pretraining by injecting the compressed attention (selection module) output into the main attention.
This design allows the selection module to receive learning signals only indirectly through the final attention output, without direct supervision on its token selection decisions.

2.2  Hybrid Attention Architecture

To reduce quadratic compute and KV cache costs, hybrid attention has emerged as a promising solution for scaling context length.
For instance, MiniMax-01  [ li2025minimax ]  integrates both linear attention and softmax attention mechanisms in a structured pattern. Similarly, Qwen3-Next  [ qwen3next2025 ]  and Kimi Linear  [ kda ]  incorporate Gated DeltaNet  [ gdn ]  or its variants. The Nemotron family  [ nemotron2 ,  blakeman2025nemotronh ]  and Jamba  [ lieber2024jamba ]  integrate Mamba modu