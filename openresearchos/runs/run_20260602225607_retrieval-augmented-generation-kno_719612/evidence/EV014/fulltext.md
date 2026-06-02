[2509.12042] FinGEAR: Financial Mapping-Guided Enhanced Answer Retrieval

FinGEAR: Financial Mapping-Guided Enhanced Answer Retrieval

Ying Li 1

Mengyu Wang 1

Miguel de Carvalho 1,2

Sotirios Sabanis 1,3,4

Tiejun Ma 1,5

1

The University of Edinburgh, United Kingdom

2 University of Aveiro, Portugal

3 National Technical University of Athens, Greece

4 Archimedes/Athena Research Centre, Greece

5 The Artificial Intelligence Applications Institute, The University of Edinburgh, United Kingdom

{sunnie.y.li, mengyu.wang, miguel.decarvalho, s.sabanis, tiejun.ma}@ed.ac.uk

Abstract

Financial disclosures such as 10-K filings pose challenging retrieval problems because of their length, regulatory section hierarchy, and domain-specific language, which standard retrieval-augmented generation (RAG) models underuse. We present  FinGEAR  (Financial Mapping-Guided Enhanced Answer Retrieval), a retrieval framework tailored to financial documents. FinGEAR combines a finance lexicon for Item-level guidance (FLAM), dual hierarchical indices for within-Item search (Summary Tree and Question Tree), and a two-stage cross-encoder reranker. This design aligns retrieval with disclosure structure and terminology, enabling fine-grained, query-aware context selection. Evaluated on full 10-Ks with the FinQA dataset, FinGEAR delivers consistent gains in precision, recall, F1, and relevancy, improving F1 by up to 56.7% over flat RAG, 12.5% over graph-based RAGs, and 217.6% over prior tree-based systems, while also increasing downstream answer accuracy with a fixed reader. By jointly modeling section hierarchy and domain lexicon signals, FinGEAR improves retrieval fidelity and provides a practical foundation for high-stakes financial analysis.

FinGEAR: Financial Mapping-Guided Enhanced Answer Retrieval

Ying Li 1

Mengyu Wang 1

Miguel de Carvalho 1,2

Sotirios Sabanis 1,3,4

Tiejun Ma 1,5

1 The University of Edinburgh, United Kingdom

2 University of Aveiro, Portugal

3 National Technical University of Athens, Greece

4 Archimedes/Athena Research Centre, Greece

5 The Artificial Intelligence Applications Institute, The University of Edinburgh, United Kingdom

{sunnie.y.li, mengyu.wang, miguel.decarvalho, s.sabanis, tiejun.ma}@ed.ac.uk

1  Introduction

Financial disclosures such as 10-K filings are key for investment analysis, regulatory monitoring, and risk assessment. They are long (often 100+ pages) and organized by SEC-mandated Items, for example, Item 1 (Business), Item 1A (Risk Factors), Item 7 (Management’s Discussion and Analysis), and Item 8 (Financial Statements). These sections mix narrative text, tables, and footnotes. Many financial NLP tasks, including sentiment analysis, trend detection, entity extraction, risk detection, and question answering, depend on first retrieving the right passages from these filings. Retrieval is difficult because relevant evidence may be spread across multiple Items or years, and domain synonyms (e.g., “sales” vs. “revenue”) and cross-references are common. Therefore, retrieval remains a major bottleneck for current work  Reddy et al. ( 2024 ); Edge et al. ( 2024 ); Asai et al. ( 2023 ); Guo et al. ( 2024 ) .

Recent efforts such as DocFinQA  Reddy et al. ( 2024 )  underscore the difficulty of applying question answering to full-length financial filings. Yet these systems typically treat retrieval as a separate external step and rely on fixed-size chunks or off-the-shelf retrievers, without aligning it to the SEC Item hierarchy and the terminology used in financial reports. This reveals a broader limitation: current retrieval methods struggle with hierarchical organization, domain terms, and the need for precise evidence in financial analysis. FinGEAR directly addresses this gap by redefining retrieval as a first-class objective, tailored to the realities of regulatory filings and their analytical use cases.

Recent advances in Large Language Models (LLMs)  Achiam et al. ( 2023 ); Dubey et al. ( 2024 ); Wu et al. ( 2023 )  and Retrieval-Augmented Generation (RAG)  Lewis et al. ( 2021 )  have enabled progress in financial document analysis by grounding outputs in retrieved evidence. Our analysis identifies three core limitations in current retrieval pipelines that constrain downstream performance across financial NLP tasks:
(1)  Lack of structure awareness : fixed-size segmentation discards the logical hierarchy of disclosures, leading to misaligned context retrieval;
(2)  Lack of financial specificity : generic retrievers fail to distinguish nuanced but crucial concepts (e.g., “net income” vs. “operating income”);
(3)  Dense-only retrieval is hard to control and explain : pure vector similarity offers limited interpretability in evidence-heavy settings.

To address these issues, we present  FinGEAR ,  Fin ancial Mapping- G uided  E nhanced  A nswer  R etrieval, a retrieval-centric framework designed for long, professionally authored, semi-structured disclosures. FinGEAR treats retrieval as the core problem, aiming to surface content that is structurally coherent, financially grounded, and useful across tasks.

FinGEAR introduces three key contributions:
(1)  Document–Query hierarchical alignment , which captures the structural layout of financial documents via a Summary Tree and enables query-sensitive retrieval through a structurally mirrored Question Tree;
(2)  Financial Lexicon-Aware Mapping (FLAM) , which steers retrieval using domain-specific term clusters and lexicon-weighted scoring;
(3)  Hybrid dense–sparse retrieval , which integrates sparse keyword anchoring with dense embedding similarity to balance interpretability and relevance.

Evaluated on full 10-K filings, FinGEAR achieves up to 138% higher retrieval F1 than flat RAG, up to 28% over graph-based baselines (e.g., LightRAG), and up to 263% over prior tree-based systems. Ablation studies confirm that these gains derive from the combined design of its structural and domain-aware modules. While FinGEAR does not directly optimize for reasoning tasks, downstream experiments confirm that enhanced retrieval leads to better answer accuracy, reinforcing retrieval quality as the foundation of financial document understanding.

To our knowledge, FinGEAR is the first retrieval-first system tailored to financial disclosures. It offers a principled and modular foundation for structured, explainable, and task-flexible financial NLP.

2  Related Work

2.1  Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG)  Lewis et al. ( 2021 )  augments language models by fetching relevant context from external corpora, reducing the need for full-model fine-tuning  Guu et al. ( 2020 ); Ram et al. ( 2023 ) . Advanced variants such as Self-RAG  Asai et al. ( 2023 )  and Adaptive RAG  Jeong et al. ( 2024 )  improve coordination between retrievers and generators but still use fixed-size chunks. This makes it hard to preserve document structure and can introduce drift in long documents, as seen in long-form QA benchmarks like ELI5  Fan et al. ( 2019 ) . Recent work addresses context-length limits with longer-context models (e.g., Transformer-XL  Dai et al. ( 2019 ) ), retrieval-aware chunking  Zhong et al. ( 2025 ) , and studies of position bias  Liu et al. ( 2023 ) . These efforts mainly target input length and do not solve structured retrieval in financial domains.

2.2  Hierarchical and Graph-Based Retrieval

Hierarchical methods such as RAPTOR  Sarthi et al. ( 2024 )  and HiQA  Chen et al. ( 2024 )  represent documents as trees and retrieve recursively from higher-level summaries. Graph-based systems, including GraphRAG  Edge et al. ( 2024 )  and LightRAG  Guo et al. ( 2024 ) , model relations between entities and sections to support multi-hop reasoning. In particular, GraphRAG builds local–global graphs over LLM-extracted entities and community summaries and retrieves via community-level traversal, while LightRAG performs dual-level query decomposition with lightweigh