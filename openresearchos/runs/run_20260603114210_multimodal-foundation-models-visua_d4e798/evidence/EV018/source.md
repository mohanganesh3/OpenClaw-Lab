# EV018: Real-Time Visual Attribution Streaming in Thinking Model

URL: https://www.semanticscholar.org/paper/092b3b7b9a41f4dee26e697456f22213de8bcfe7
Year: 2026
Source: semantic_scholar
Arxiv: 2604.16587

## Abstract

We present an amortized framework for real-time visual attribution streaming in multimodal thinking models. When these models generate code from a screenshot or solve math problems from images, their long reasoning traces should be grounded in visual evidence. However, verifying this reliance is challenging: faithful causal methods require costly repeated backward passes or perturbations, while raw attention maps offer instant access, they lack causal validity. To resolve this, we introduce an amortized approach that learns to estimate the causal effects of semantic regions directly from the rich signals encoded in attention features. Across five diverse benchmarks and four thinking models, our approach achieves faithfulness comparable to exhaustive causal methods while enabling visual attribution streaming, where users observe grounding evidence as the model reasons, not after. Our results demonstrate that real-time, faithful attribution in multimodal thinking models is achievable through lightweight learning, not brute-force computation.
