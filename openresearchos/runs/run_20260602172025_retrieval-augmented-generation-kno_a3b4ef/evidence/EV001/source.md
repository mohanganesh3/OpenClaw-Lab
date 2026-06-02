# EV001: MeVe: A Modular System for Memory Verification and Effective Context Control in Language Models

URL: https://www.semanticscholar.org/paper/0053b77fb23ede4c58ebd18e9d5daaf779b2213e
Year: 2025
Source: semantic_scholar
Arxiv: 2509.01514

## Abstract

Retrieval-Augmented Generation (RAG) systems typically face constraints because of their inherent mechanism: a simple top-k semantic search [1]. The approach often leads to the incorporation of irrelevant or redundant information in the context, degrading performance and efficiency [10][11]. This paper presents MeVe, a novel modular architecture intended for Memory Verification and smart context composition. MeVe rethinks the RAG paradigm by proposing a five-phase modular design that distinctly breaks down the retrieval and context composition process into distinct, auditable, and independently tunable phases: initial retrieval, relevance verification, fallback retrieval, context prioritization, and token budgeting. This architecture enables fine-grained control of what knowledge is made available to an LLM, enabling task-dependent filtering and adaptation. We release a reference implementation of MeVe as a proof of concept and evaluate its performance on knowledge-heavy QA tasks over a subset of English Wikipedia [22]. Our results demonstrate that by actively verifying information before composition, MeVe significantly improves context efficiency, achieving a 57% reduction on the Wikipedia dataset and a 75% reduction on the more complex HotpotQA dataset compared to standard RAG implementations [25]. This work provides a framework for more scalable and reliable LLM applications. By refining and distilling contextual information, MeVe offers a path toward better grounding and more accurate factual support [16].
