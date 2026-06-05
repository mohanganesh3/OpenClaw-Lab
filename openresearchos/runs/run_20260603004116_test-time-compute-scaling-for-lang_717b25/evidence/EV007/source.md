# EV007: GRIP: In-Parameter Graph Reasoning through Fine-Tuning Large Language Models

URL: https://www.semanticscholar.org/paper/000b4b45fec757fc339f9112a6feb3c3f8bf5de2
Year: 2025
Source: semantic_scholar
Arxiv: 2511.07457

## Abstract

Large Language Models (LLMs) have demonstrated remarkable capabilities in modeling sequential textual data and generalizing across diverse tasks. However, adapting LLMs to effectively handle structural data, such as knowledge graphs or web data, remains a challenging problem. Some approaches adopt complex strategies to convert graphs into text sequences, resulting in significant token overhead and rendering them impractical for large-scale graphs. Others introduce additional modules to encode graphs into fixed-size token representations for LLMs. However, these methods typically require large-scale post-training on graph-text corpus and complex alignment procedures, yet often yield sub-optimal results due to poor modality alignment. Inspired by in-parameter knowledge injection for test-time adaptation of LLMs, we propose GRIP, a novel framework that equips LLMs with the ability to internalize complex relational information from graphs through carefully designed fine-tuning tasks. This knowledge is efficiently stored within lightweight LoRA parameters, enabling the fine-tuned LLM to perform a wide range of graph-related tasks without requiring access to the original graph at inference time. Extensive experiments across multiple benchmarks validate the effectiveness and efficiency of our approach.
