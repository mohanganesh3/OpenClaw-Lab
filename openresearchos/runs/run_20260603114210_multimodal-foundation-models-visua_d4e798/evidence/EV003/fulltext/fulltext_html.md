[2601.03262] Roles of MLLMs in Visually Rich Document Retrieval for RAG: A Survey

Roles of MLLMs in Visually Rich Document Retrieval for RAG: A Survey

Xiantao Zhang

Beihang University

zhangxiantao@buaa.edu.cn

Abstract

Visually rich documents (VRDs) challenge retrieval-augmented generation (RAG) with layout-dependent semantics, brittle OCR, and evidence spread across complex figures and structured tables.
This survey examines how Multimodal Large Language Models (MLLMs) are being used to make VRD retrieval practical for RAG.
We organize the literature into three roles:  Modality-Unifying Captioners ,  Multimodal Embedders , and  End-to-End Representers .
We compare these roles along retrieval granularity, information fidelity, latency and index size, and compatibility with reranking and grounding.
We also outline key trade-offs and offer some practical guidance on when to favor each role.
Finally, we identify promising directions for future research, including adaptive retrieval units, model size reduction, and the development of evaluation methods.

Roles of MLLMs in Visually Rich Document Retrieval for RAG: A Survey

Xiantao Zhang

Beihang University

zhangxiantao@buaa.edu.cn

1  Introduction

Visually rich documents (VRDs), such as PDFs, scanned pages, slide decks, reports, forms, and infographics, encode meaning through the interplay of text, layout, figures, and graphics.
As retrieval-augmented generation (RAG)  Lewis  et al.  ( 2020 )  becomes a default pattern for grounding large language models (LLMs)  Guu  et al.  ( 2020 ); Borgeaud  et al.  ( 2022 ); Izacard  et al.  ( 2023 ); Nakano  et al.  ( 2022 ) , many real-world deployments are moving beyond plain text to these document types  Ma  et al.  ( 2024 ); Faysse  et al.  ( 2025 ); Yu  et al.  ( 2025 ); Suri  et al.  ( 2025 ); Tanaka  et al.  ( 2025 ) .
This shift strains classical text-only RAG pipelines, motivating the broader development of multimodal RAG (MM-RAG) systems designed to retrieve and reason over varied data types, including images and tables  Chen  et al.  ( 2022 ); Yasunaga  et al.  ( 2023 ) .

However, VRDs represent a uniquely difficult case for MM-RAG.
Unlike retrieving standalone images or text, VRD retrieval must contend with meaning derived from the  fusion  of layout, embedded text, and graphics.
Consequently, traditional preprocessing steps like optical character recognition (OCR) and layout parsing remain brittle and lossy, fine-grained visual cues vanish in textual proxies, and evidence may span multiple pages or views.
Recent surveys in document understanding  Ding  et al.  ( 2025 )  echo this shift, underscoring both the opportunity and difficulty of learning over text–layout–vision jointly.

At the same time, a new generation of methods argues for  seeing  pages directly.
Document Screenshot Embedding (DSE)  Ma  et al.  ( 2024 )  treats a page screenshot as the unit of indexing, avoiding preprocessing choices that introduce error and latency. Its premise is pragmatic: keep all information available at retrieval time.
Likewise, ColPali  Faysse  et al.  ( 2025 )  fuses Vision–Language Models (VLMs) with late-interaction matching, and results show that learning directly over page images can simplify pipelines and improve effectiveness.
Beyond retrieval alone, VisRAG  Yu  et al.  ( 2025 )  integrates vision-based retrieval with generation, adopting the page-as-image abstraction end-to-end to mitigate conversion loss during both retrieval and answer synthesis.

VRD-centric evaluation has also matured. Beyond classic DocVQA  Mathew  et al.  ( 2021 ) , InfographicVQA  Mathew  et al.  ( 2022 ) , and SlideVQA  Tanaka  et al.  ( 2023 ) , newer resources now increasingly stress chart reasoning and multi-slide evidence aggregation reflecting practical needs like finding a single number inside a plot or tracing an argument across a deck  Tamber  et al.  ( 2025 ); Yang  et al.  ( 2025 ); Liu  et al.  ( 2025 ); Chen  et al.  ( 2025b ); Peng  et al.  ( 2025 ) .
These datasets collectively highlight why retrieval must respect both layout and visual semantics, not only text.

Scope and goal

This survey focuses specifically on visually rich document retrieval for RAG.
We analyze how Multimodal Large Language Models (MLLMs) are used to index and retrieve pages, page regions, tables, figures, and slide content for RAG over documents.
Our goal is to distill design patterns, compare empirical trends, and surface trade-offs that matter for building reliable, cost-aware systems.

Contributions

This survey makes the following contributions:

1.

Role-based taxonomy of VRD–RAG.  We organize how MLLMs enter the pipeline into three roles tailored to documents.

2.

Comparative analysis of key trade-offs.  We contrast these roles in terms of retrieval unit, robustness to OCR and layout errors, latency and indexing cost, and compatibility with reranking and grounding, summarizing evidence from recent VRD-focused work.

3.

Practical takeaways and open challenges.  We discuss when to favor caption-first vs. image-first retrieval, how to balance page-level recall with element-level precision, how to budget compute and storage for multimodal indices, and where evaluation lags behind given the current benchmarks.

Organization

§  2

reviews background on RAG, multimodal retrieval, and MLLMs. §  3

develops the three-role framework and contrasts representative approaches. §  4

examines trade-offs and open challenges. §  5

concludes with takeaways and future directions.

Figure 1:  Overview of how MLLMs enter VRD retrieval for RAG across three roles.
 Left:

Modality-Unifying Captioners  (§  3.1  );
 Middle:

Multimodal Embedders  (§  3.2  );
 Right:

End-to-End Representers  (§  3.3  ).
Each panel sketches the pipeline from document intake to retrieval and answer synthesis, highlighting typical retrieval units and index types.

2  Related Work

2.1  Retrieval-Augmented Generation

RAG combines a retriever and a generator to bridge retrieval-based and generative models.
This hybrid approach dynamically retrieves documents to condition generation, enhancing factual accuracy and access to knowledge beyond training data  Shuster  et al.  ( 2021 ); Gao  et al.  ( 2024 ); Lewis  et al.  ( 2020 ); Asai  et al.  ( 2024 ); Shi  et al.  ( 2024 ); Izacard  et al.  ( 2023 ); Borgeaud  et al.  ( 2022 ); Li  et al.  ( 2024 ) .
Recent research expanded RAG to open-domain QA  Guu  et al.  ( 2020 ); Mao  et al.  ( 2021 ); Siriwardhana  et al.  ( 2023 ) , dialogue systems  Thulke  et al.  ( 2021 ); Komeili  et al.  ( 2022 ); Li  et al.  ( 2022b ) , and multimodal tasks  Chen  et al.  ( 2022 ); Yasunaga  et al.  ( 2023 ); Hu  et al.  ( 2023 ); Luo  et al.  ( 2024 ); Ren  et al.  ( 2025 ); Jeong  et al.  ( 2025 ) , highlighting its potential for integrating diverse knowledge into NLP pipelines.

2.2  Multimodal Retrieval

Recent studies have demonstrated that multimodal retrieval and RAG significantly enhance LLMs by integrating diverse data modalities, such as text, images, and audio.
The seminal work of MuRAG  Chen  et al.  ( 2022 )  inaugurated the era of end-to-end multimodal retrieval-augmented transformers, a pioneering innovation that has since been shown to enhance performance in a range of tasks, including question answering, by leveraging external multimodal memory.
In a similar manner, RA-CM3  Yasunaga  et al.  ( 2023 )  was the first to demonstrate the capabilities of joint retrieval and text and image generation, achieving superior performance compared to models such as DALL-E  Ramesh  et al.  ( 2021 ) , while being more efficient.
 Wei  et al.  ( 2024 )  proposed UniIR, a universal multimodal retrieval model designed to handle a wide range of tasks.
Subsequent advancements include GENIUS  Kim  et al.  ( 2025 ) , a universal generative framework for multimodal search, and UMaT  Bi and Xu ( 2025 ) , which unifies video and audio data via textual representations for long-form question-answeri