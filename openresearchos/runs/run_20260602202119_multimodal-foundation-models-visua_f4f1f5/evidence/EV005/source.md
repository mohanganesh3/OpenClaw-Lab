# EV005: MM-R$^3$: On (In-)Consistency of Vision-Language Models (VLMs)

URL: https://www.semanticscholar.org/paper/095cad58d58a7c06d85c3c1ff4f9f42fe934b114
Year: 2024
Source: semantic_scholar
Arxiv: 2410.04778

## Abstract

With the advent of LLMs and variants, a flurry of research has emerged, analyzing the performance of such models across an array of tasks. While most studies focus on evaluating the capabilities of state-of-the-art (SoTA) Vision Language Models (VLMs) through task accuracy (e.g., visual question answering, grounding), our work explores the related but complementary aspect of consistency - the ability of a VLM to produce semantically similar or identical responses to semantically similar queries. We note that consistency is a fundamental prerequisite (necessary but not sufficient condition) for robustness and trust in VLMs. Armed with this perspective, we propose the MM-R3 benchmark, which allows us to analyze performance, in terms of consistency and accuracy, of SoTA VLMs on three tasks: Question Rephrasing, Image Restyling, and Context Reasoning. Our analysis reveals that consistency does not always align with accuracy, indicating that models with higher accuracy are not necessarily more consistent, and vice versa. Furthermore, we propose a simple yet effective mitigation strategy in the form of an adapter module trained to minimize inconsistency across prompts. With our proposed strategy, we are able to achieve absolute improvements of 5.7% and 12.5%, on average on widely used VLMs such as BLIP-2 and LLaVa 1.5M in terms of consistency over their existing counterparts.
