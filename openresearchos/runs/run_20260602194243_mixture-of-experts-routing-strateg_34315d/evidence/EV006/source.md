# EV006: DrDiff: Dynamic Routing Diffusion with Hierarchical Attention for Breaking the Efficiency-Quality Trade-off

URL: https://www.semanticscholar.org/paper/0135919988d65d434557ec60ca9f6d5bb6a1c325
Year: 2025
Source: semantic_scholar
Arxiv: 2509.02785

## Abstract

This paper introduces DrDiff, a novel framework for long-text generation that overcomes the efficiency-quality trade-off through three core technologies. First, we design a dynamic expert scheduling mechanism that intelligently allocates computational resources during the diffusion process based on text complexity, enabling more efficient handling of text generation tasks of varying difficulty. Second, we introduce a Hierarchical Sparse Attention (HSA) mechanism that adaptively adjusts attention patterns according to a variety of input lengths, reducing computational complexity from O($n^2$) to O($n$) while maintaining model performance. Finally, we propose a soft absorption guidance optimization strategy that combines with DPM-solver++ to reduce diffusion steps, significantly improving generation speed. Comprehensive experiments on various long-text generation benchmarks demonstrate the superiority of our DrDiff over the existing SOTA methods.
