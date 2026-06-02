# EV009: Understanding the Visual Projection Space of Multimodal LLMs

URL: https://www.semanticscholar.org/paper/17dff405482138b44f64e41178d2e0088852f516
Year: 2026
Source: semantic_scholar
Arxiv: n/a

## Abstract

What role does a single vision token play inside a multimodal large language model (MLLM)? Despite recent successes, most MLLMs adopt a simple design: a projected visual feature z = P(fx) prepended to the text sequence. Yet it remains unclear whether this vector merely provides context or actively steers generation. We propose a geometric probing framework that analyzes latent–token alignment, intrinsic dimensionality, and perturbation sensitivity. Across four datasets and three representative MLLMs (LLaVA, BLIP-2, Kosmos-2), we find clear operating regimes: BLIP-2 enforces rigid low-rank compression with strong alignment but near-zero sensitivity, LLaVA exhibits flexible high-dimensional mappings with high responsiveness, and Kosmos-2 balances between them. These signatures correlate with downstream behavior—SQA correctness and VQAv2 hallucination severity—showing that reduced or excessive sensitivity predicts unreliable grounding. Our results highlight geometry as a diagnostic lens for vision–language coupling and offer actionable guidance for projection design, alignment objectives, and user-steerable multimodal generation.
