# EV016: An Auditable LLM-RAG Architecture for Financial Document Intelligence and Decision Support

URL: https://www.semanticscholar.org/paper/01e265324b4200f69b4d49c05aa2ed24c8515d52
Year: 2026
Source: semantic_scholar
Arxiv: n/a

## Abstract

Financial analysis increasingly depends on the ability to transform heterogeneous textual evidence into reliable, verifiable, and actionable knowledge. However, adoption in finance requires generated outputs to be not only accurate, but also traceable and auditable. This work presents an audit-oriented LLM-RAG architecture for financial document intelligence. Rather than proposing a new foundation model, the contribution is a reproducible pipeline that integrates financial document processing, hybrid retrieval, evidence-grounded generation, structured validation, and persistent audit artifacts within a state-machine-based workflow. Designed for analyst-facing use, the system produces structured answers linked to explicit evidence while preserving the intermediate artifacts needed to inspect, reproduce, and validate each result. Experiments on AI-FinanceQA, a benchmark of heterogeneous financial documents and analyst-style questions, show that hybrid retrieval with reranking improves evidence selection over single-signal baselines and that the selected LLM backend achieves a compliance-oriented score of Scomp=0.9527. Additional experiments on FinQA confirm that targeted evidence selection improves numerical robustness and semantic alignment compared with uncontrolled context expansion. Overall, the proposed architecture provides an evidence-grounded and audit-oriented framework that supports human review rather than replacing expert financial judgment.
