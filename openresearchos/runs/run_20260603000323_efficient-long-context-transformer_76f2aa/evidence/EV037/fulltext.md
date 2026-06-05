[2503.03588] PowerAttention: Exponentially Scaling of Receptive Fields for Effective Sparse Attention

PowerAttention: Exponentially Scaling of Receptive Fields for Effective Sparse Attention

Lida Chen

Dong Xu

Chenxin An

Xintao Wang

Yikai Zhang

Jiangjie Chen

Zujie Liang

Feng Wei

Jiaqing Liang

Yanghua Xiao

Wei Wang

Abstract

Large Language Models (LLMs) face efficiency bottlenecks due to the quadratic complexity of the attention mechanism when processing long contexts.
Sparse attention methods offer a promising solution, but existing approaches often suffer from incomplete effective context and/or require complex implementation of pipeline.
We present a comprehensive analysis of sparse attention for autoregressive LLMs from the respective of receptive field, recognize the suboptimal nature of existing methods for expanding the receptive field, and introduce  PowerAttention , a novel sparse attention design that facilitates effective and complete context extension through the theoretical analysis.
 PowerAttention  achieves exponential receptive field growth in

d  d

-layer LLMs, allowing each output token to attend to

2  d

2^{d}

tokens, ensuring completeness and continuity of the receptive field.
Experiments demonstrate that  PowerAttention  outperforms existing static sparse attention methods by

5  ∼

40  %

5\sim 40\%

, especially on tasks demanding long-range dependencies like Passkey Retrieval and RULER, while maintaining a comparable time complexity to sliding window attention.
Efficiency evaluations further highlight  PowerAttention ’s superior speedup in both prefilling and decoding phases compared with dynamic sparse attentions and full attention (

3.0  ×

3.0\times

faster on 128K context), making it a highly effective and user-friendly solution for processing long sequences in LLMs.

Sparse Attention, Long-context Scaling, Large Language Models

1  Introduction

Figure 1:

Layer-wise receptive field analysis of sparse attention patterns.
(a) illustrates the information flow across six layers with a simplified 128-block example, while (b) presents the quantitative evaluation on Qwen2-7B with 32K context length.
The actual token retrieval capability closely matches the theoretical receptive field growth for both patterns.
Within the maximum information propagation depth,  PowerAttention ’s exponential growth in receptive field leads to significantly higher accuracy compared to sliding window’s linear expansion.
Detailed implementation is provided in Appendix

A  .

Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse NLP tasks.
Increasing context length allows LLMs to support more complex applications like long chain-of-thought reasoning  (OpenAI,  2024 ; Qwen,  2024 ; DeepSeek-AI et al.,  2025 ) , agents in complex environments  (Park et al.,  2023 ; Zhou et al.,  2023 ; Chen et al.,  2023 ) , and long document question answering  (Chevalier et al.,  2024 ; Wang et al.,  2024 ; An et al.,  2023 ) .
However, the quadratic complexity of the attention mechanism poses a significant efficiency bottleneck for Transformer-based LLMs when processing long contexts.

To address the inefficiency of Transformer, recent studies have explored sparse attention  (Correia et al.,  2019 ; Beltagy et al.,  2020 ; Roy et al.,  2021 ; Liu et al.,  2022 ; Anagnostidis et al.,  2023 ; Jiang et al.,  2024 ) , which reduces computational complexity by restricting each token to attend to only a fixed number of tokens instead of the full sequence.
The  static  pattern uses a pre-defined attention mask such as the classic sliding window attention, while the  dynamic  pattern requires the model to be trained with full attention and to update the defined attention mask at inference stage, such as MInference  (Jiang et al.,  2024 ) .
Dynamic patterns usually achieves better performance in downstream tasks, but the static counterpart features efficiency optimization in training stage and can better handle new tokens during decoding.

However, both the two mainstream sparse attention methods have predominantly relied on intuitive heuristics and experimental results, lacking theoretical analysis to explain their effectiveness.
In this paper, we address this critical gap by presenting a novel comprehensive analysis of sparse attention methods for autoregressive LLMs, providing new insights for designing efficient attention for the future.
Our analysis starts from the information flow across LLM layers.
Consider how information flows within an LLM: at each layer, a token receives information from other tokens it can attend to via self-attention and propagates this aggregated information to subsequent layers.
To analyze this process systematically, we introduce the concept of  model receptive field , defined as the maximum set of context tokens that the model can utilize during output generation, and model it as a Directed Acyclic Graph (DAG) where different static sparse attention patterns correspond to different edge sets.
Although different sparse patterns with the same sparsity result in identical single-layer receptive field sizes, well-designed patterns can achieve much larger effective receptive fields across multiple layers through efficient information propagation (Figure

1  (a)).

Based on this analysis framework, we identify two critical limitations of existing static sparse attention designs that prevent them from achieving optimal receptive fields:
(1) information from tokens at certain positions cannot be retrieved by the final output,
and (2) they exhibit low efficiency in expanding the receptive field layer by layer, as demonstrated by sliding window’s linear growth in Figure

1  (b).
Based on these insights, we propose  PowerAttention , a novel sparse attention pattern that can achieve an effective balance between efficiency and performance, both theoretically and experimentally (Figure

1  (b)).
Specifically, by calculating attention between tokens at power-of-2 distances,  PowerAttention  achieves exponential expansion of the receptive field across layers while requiring only

O  ​

(

log  ⁡  n

)

O(\log n)

tokens to ensure the receptive field covers the entire sequence, demonstrating significant potential for ultra-long sequences and high-sparsity scenarios.

We conduct comprehensive experiments to evaluate both the model performance and efficiency of existing static sparse attention methods and  PowerAttention .
On long-range dependency tasks like Passkey Retrieval and RULER,  PowerAttention  significantly outperforms other static sparse attention methods.
In terms of efficiency, static sparse attention methods with the same sparsity show similar performance, and outperform both full attention and dynamic sparse attention methods like MInference in both prefilling and decoding phases by

1.2  ∼  30  ×

1.2\sim 30\times

.

In summary, our contributions are:

•

We establish an analysis framework for studying static sparse attention patterns in autoregressive LLMs, which explains why certain patterns are effective.

•

We design a novel static sparse attention pattern,  PowerAttention , that achieves the best balance between efficiency and performance, both theoretically and experimentally.

•

We conduct extensive experiments demonstrating that  PowerAttention  achieves superior performance compared to existing static sparse attention methods while maintaining state-of-the-art efficiency.

2  Related Work

Dynamic Sparse Attention

It has been widely observed that attention patterns are often highly sparse  (Liu et al.,  2022 ) , allowing certain correlation computations between tokens to be omitted without significantly degrading the model performance.
Dynamic sparse attention mechanism predicts the necessary sparse pattern based on the input context and relevant information, which focuses on either prefilling  (Roy et al.,  2021 ; Qu et al.,  2022 ; Liu et al.,  2022 ; Ribar et al.,  2024 ; Jiang e