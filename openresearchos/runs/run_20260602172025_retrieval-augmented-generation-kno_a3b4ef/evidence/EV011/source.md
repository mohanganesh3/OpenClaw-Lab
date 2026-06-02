# EV011: Heterogeneous LLM Methods for Ontology Learning (Few-Shot Prompting, Ensemble Typing, and Attention-Based Taxonomies)

URL: https://www.semanticscholar.org/paper/01475410a471fbeafca75c806acfb6be6489f0a3
Year: 2025
Source: semantic_scholar
Arxiv: 2508.19428

## Abstract

We present a comprehensive system for addressing Tasks A, B, and C of the LLMs4OL 2025 challenge, which together span the full ontology construction pipeline: term extraction, typing, and taxonomy discovery. Our approach combines retrieval-augmented prompting, zero-shot classification, and attention-based graph modeling — each tailored to the demands of the respective task.
For Task A, we jointly extract domain-specific terms and their ontological types using a retrieval-augmented generation (RAG) pipeline.Training data was reformulated into a document to terms and types correspondence, while test-time inference leverages semantically similar training examples. This single-pass method requires no model finetuning and improves overall performance through lexical augmentation
Task B, which involves assigning types to given terms, is handled via a dual strategy. In the few-shot setting (for domains with labeled training data), we reuse the RAG scheme with few-shot prompting. In the zero-shot setting (for previously unseen domains), we use a zero-shot classifier that combines cosine similarity scores from multiple embedding models using confidence-based weighting.
In Task C, we model taxonomy discovery as graph inference. Using embeddings of type labels, we train a lightweight cross-attention layer to predict is-a relations by approximating a soft adjacency matrix.
These modular, task-specific solutions enabled us to achieve top-ranking results in the official leaderboard across all three tasks. Taken together these strategies showcase the scalability, adaptability, and robustness of LLM-based architectures for ontology learning across heterogeneous domains.
Code is available at: https://github.com/BelyaevaAlex/LLMs4OL-Challenge-Alexbek
