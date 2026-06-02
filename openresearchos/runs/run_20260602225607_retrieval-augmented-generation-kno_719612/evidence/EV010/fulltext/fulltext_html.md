[2509.01514] MeVe: A Modular System for Memory Verification and Effective Context Control in Language Models

1

1  institutetext:

Independent Researcher

Alta, Norway

MeVe: A Modular System for Memory Verification and Effective Context Control in Language Models

Andreas Ottem

Abstract

Retrieval-Augmented Generation (RAG) systems typically face constraints because of their inherent mechanism: a simple top-k semantic search  [ 1 ] . The approach often leads to the incorporation of irrelevant or redundant information in the context, degrading performance and efficiency  [ 10 ]  [ 11 ] . This paper presents MeVe, a novel modular architecture intended for Memory Verification and smart context composition.

MeVe rethinks the RAG paradigm by proposing a five-phase modular design that distinctly breaks down the retrieval and context composition process into distinct, auditable, and independently tunable phases: initial retrieval, relevance verification, fallback retrieval, context prioritization, and token budgeting. This architecture enables fine-grained control of what knowledge is made available to an LLM, enabling task-dependent filtering and adaptation. We release a reference implementation of MeVe as a proof of concept and evaluate its performance on knowledge-heavy QA tasks over a subset of English Wikipedia  [ 22 ] . Our results demonstrate that by actively verifying information before composition, MeVe significantly improves context efficiency, achieving a 57% reduction on the Wikipedia dataset and a 75% reduction on the more complex HotpotQA dataset compared to standard RAG implementations  [ 25 ] . This work provides a framework for more scalable and reliable LLM applications. By refining and distilling contextual information, MeVe offers a path toward better grounding and more accurate factual support  [ 16 ] .

1  Introduction

The impressive abilities of Large Language Models (LLMs) are fundamentally reliant on the ability to reason in a particular context  [ 6 ]

[ 7 ] . As models continue to utilize larger context windows and retrieve information from external memory, the key challenge shifts from simple information retrieval to smart contextual management  [ 5 ] . It includes the dynamic fetching of contextually relevant knowledge, rigorous verification of its relevance, and efficient composition with the objective of improving LLM performance  [ 6 ]

[ 14 ] .

Retrieval-Augmented Generation (RAG) has also become a strong baseline for augmenting LLMs with out-of-model knowledge  [ 1 ] .

Traditional RAG systems commonly operate by performing a direct, often ”top-k” semantic search against a pre-indexed knowledge base, retrieving the most similar documents and directly appending them to the LLMs input prompt. While effective in providing external knowledge, this straightforward mechanism can lead to issues such as including irrelevant, redundant or even contradictory information  [ 14 ]

[ 1 ]

[ 2 ] . Yet, typical implementations tightly intertwine retrieval and injection into a single monolithic step, usually by picking a fixed number of embedding-nearest neighbors and simply adding them to the prompt  [ 2 ]

[ 3 ] . This common practice makes verification implicit, heavily reduces modularity, and tends to have a difficult time gracefully scaling to dynamic memory expansion or tight token budget requirements  [ 17 ] . The direct injection of possibly irrelevant or redundant information can lead to ”context pollution”  [ 11 ]

[ 14 ] , degrading the quality and effectiveness of LLM responses and increasing the risk of factual error or ”hallucinations”  [ 10 ]

[ 14 ]

[ 13 ] . This challenge is compounded by the fact that retrieval quality is highly dependent on the preciseness and structure of the knowledge corpus itself  [ 2 ] .

As a solution to such fundamental shortcomings, we present MeVe, a new method that fundamentally redesigns retrieval, verification, prioritization, and budgeting as separate, configurable, and auditable phases  [ 4 ]

[ 19 ] . MeVe’s key contribution is its modular decomposition of the RAG pipeline, transforming it from a singular operation into a structured, multi-stage process with explicit control over content quality and composition. MeVe does not aim to substitute RAG but instead redefines traditional RAG as one possible configuration within a more sophisticated, memory-aware architecture. The modular design founded upon the principles of explicit verification of information and strict retrieval mechanisms aim to alleviate typical failure modes that have been witnessed in state-of-the-art RAG systems, including context confusion and propagation of irrelevant or misleading information  [ 10 ]

[ 11 ]

[ 16 ] .
The MeVe framework provides a modular and systematic approach to LLM context management, leading to enhanced efficiency and control in tasks involving large amounts of knowledge  [ 17 ] . This paper is organized as follows: Section 2 reviews relevant literature, Section 3 details the MeVe architecture, Section 4 presents our empirical evaluation, and Section 5 discusses the results, leading to our conclusion in Section 6.

2  Related Work and Conceptual Origins

2.1  Retrieval-Augmented Generation (RAG)

RAG enhances LLMs by anchoring them to external knowledge  [ 1 ] . While effective, the standard top-k paradigm is notoriously vulnerable to ”distractor documents”, which is semantically similar information that is contextually irrelevant  [ 10 ]

[ 11 ]

[ 13 ] . Despite the presence of sophisticated methods such as re-ranking, these are typically applied as secondary corrections to an already formed, imperfect candidate pool  [ 9 ]

[ 2 ] . MeVe addresses this issue by bringing explicit verification to a basic, integral step of the memory processing paradigm, thereby setting itself apart from solutions founded solely upon early retrieval or later re-ranking for determining relevance  [ 2 ]

[ 3 ] .

2.2  Long-Context Architectures and the Need for Filtering

The introduction of Long-Context Architectures, such as in Ring Attention systems or those controlled by MemGPT, enables LLMs to process much greater levels of information  [ 5 ]

[ 7 ] . These architectures alone do not inherently ensure the quality or applicability of the augmented context  [ 6 ]

[ 5 ] . The ”garbage in, garbage out” principle is particularly evident in such systems: unchecked data not only increases computational overhead but also causes context pollution, heightening the risk of hallucination  [ 2 ]

[ 14 ]

[ 16 ] . MeVe acts as a discerning gatekeeper for these broad contexts, ensuring that only high-utility information is passed on to the LLM  [ 17 ] .

2.3  Hybrid Search and Multi-Stage Filtering

MeVe integrates concepts of classical ”information retrieval,” especially via hybrid search and multi-stage filtering  [ 8 ]

[ 19 ] . By pairing a keyword-based fallback mechanism with dense retrieval methods, our system employs a hybrid search approach to provide robustness against vector-only search pitfalls like semantic drift problems  [ 15 ] . MeVe’s design allows fine-grained control of each stage of retrieval and filtering, building on prior modular IR pipelines  [ 4 ]

[ 19 ] .

3  The MeVe Architectural Framework

MeVe is designed as a serial pipeline of compatible modules that convert a user query (

q  q

) into a dense, high-utility terminal context (

C

f  ​  i  ​  n  ​  a  ​  l

C_{final}

) for an LLM  [ 4 ] . It offers a structured procedure that enables controlled and optimized functioning at each critical phase of context generation  [ 17 ] . MeVe distinguishes itself from conventional RAG architectures through its five-phase modular design, offering enhanced control and transparency over the knowledge supplied to an LM. Unlike existing end-to-end RAG systems that often treat retrieval and text composition as a single, opaque step, MeVe isolates and optimizes each stage of memory processing. This mod