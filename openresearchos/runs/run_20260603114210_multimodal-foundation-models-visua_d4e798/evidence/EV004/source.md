# EV004: Scaling-Aware Adapter for Structure-Grounded LLM Reasoning

URL: https://www.semanticscholar.org/paper/0203b2695912593fd0c7a6754f1e8caaa76e39b7
Year: 2026
Source: semantic_scholar
Arxiv: 2602.02780

## Abstract

Large language models (LLMs) are enabling reasoning over 2D and 3D structures, yet existing methods remain modality-specific and typically compress structural inputs through sequence-based tokenization or fixed-length query connectors. Such architectures either omit the geometric grounding requisite for mitigating structural hallucinations, or impose inflexible modality fusion bottlenecks that concurrently over-compress and suboptimally allocate structural tokens, thereby impeding the realization of generalized all-atom reasoning. We introduce Cuttlefish, a unified multimodal LLM that grounds language reasoning in geometric cues while scaling modality tokens with structural complexity. First, Scaling-Aware Patching leverages an instruction-conditioned gating mechanism to generate variable-size patches over structural graphs, adaptively scaling the query token budget with structural complexity to mitigate fixed-length connector bottlenecks. Second, Geometry Grounding Adapter refines these adaptive tokens via cross-attention to modality embeddings and injects the resulting modality tokens into the LLM, exposing explicit geometric cues to reduce structural hallucination. Experiments across interdisciplinary all-atom benchmarks demonstrate that Cuttlefish achieves superior performance in heterogeneous structure-grounded reasoning. Code: github.com/zihao-jing/Cuttlefish.
