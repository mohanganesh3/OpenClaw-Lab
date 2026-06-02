# EV006: Leveraging large language models for rare disease named entity recognition

URL: https://www.semanticscholar.org/paper/00490bf4b6773493772569ce0c8a549534fa6fb5
Year: 2025
Source: semantic_scholar
Arxiv: 2508.09323

## Abstract

Named Entity Recognition (NER) in the rare disease domain poses unique challenges due to limited labeled data, semantic ambiguity between entity types, and long-tail distributions. In this study, we evaluate the capabilities of GPT-4o for rare disease NER under low-resource settings, using a range of prompt-based strategies including zero-shot prompting, few-shot in-context learning, retrieval-augmented generation (RAG), and task-level fine-tuning. We design a structured prompting framework that encodes domain-specific knowledge and disambiguation rules for four entity types. We further introduce two semantically guided few-shot example selection methods to improve in-context performance while reducing labeling effort. Experiments on the RareDis Corpus show that GPT-4o achieves competitive or superior performance compared to BioClinicalBERT, with task-level fine-tuning yielding the strongest performance among the evaluated approaches and improving upon the previously reported BioClinicalBERT baseline. Cost-performance analysis reveals that few-shot prompting delivers high returns at low token budgets. RAG provides limited overall gains but can improve recall for challenging entity types, especially signs and symptoms. An error taxonomy highlights common failure modes such as boundary drift and type confusion, suggesting opportunities for post-processing and hybrid refinement. Our results demonstrate that prompt-optimized LLMs can serve as effective, scalable alternatives to traditional supervised models in biomedical NER, particularly in rare disease applications where annotated data is scarce.
