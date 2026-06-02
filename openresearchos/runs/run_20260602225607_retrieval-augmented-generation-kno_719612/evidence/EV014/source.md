# EV014: FinGEAR: Financial Mapping-Guided Enhanced Answer Retrieval

URL: https://www.semanticscholar.org/paper/005f84fa94ace6c19a0bd5aa8df1ff0b6d92326a
Year: 2025
Source: semantic_scholar
Arxiv: 2509.12042

## Abstract

Financial disclosures such as 10-K filings present challenging retrieval problems due to their length, regulatory section hierarchy, and domain-specific language, which standard retrieval-augmented generation (RAG) models underuse. We introduce FinGEAR (Financial Mapping-Guided Enhanced Answer Retrieval), a retrieval framework tailored to financial documents. FinGEAR combines a finance lexicon for Item-level guidance (FLAM), dual hierarchical indices for within-Item search (Summary Tree and Question Tree), and a two-stage cross-encoder reranker. This design aligns retrieval with disclosure structure and terminology, enabling fine-grained, query-aware context selection. Evaluated on full 10-Ks with queries aligned to the FinQA dataset, FinGEAR delivers consistent gains in precision, recall, F1, and relevancy, improving F1 by up to 56.7% over flat RAG, 12.5% over graph-based RAGs, and 217.6% over prior tree-based systems, while also increasing downstream answer accuracy with a fixed reader. By jointly modeling section hierarchy and domain lexicon signals, FinGEAR improves retrieval fidelity and provides a practical foundation for high-stakes financial analysis.
