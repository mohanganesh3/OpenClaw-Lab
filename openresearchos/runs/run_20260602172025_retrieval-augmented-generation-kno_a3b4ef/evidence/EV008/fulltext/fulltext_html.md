[2604.13721] FRAGATA: Semantic Retrieval of HPC Support Tickets via Hybrid RAG over 20 Years of Request Tracker History

FRAGATA: Semantic Retrieval of HPC Support Tickets via Hybrid RAG over 20 Years of Request Tracker History

Santiago Paramés-Estévez

A Spanish version of this paper has been accepted at Jornadas SARTECO 2026. Code is available at:  https://github.com/s-parames/fragata

Galicia Supercomputing Center (CESGA), Santiago de Compostela, e-mail:  sparames@cesga.es .

Nicolás Filloy-Montesino

Universidade de Vigo.

Jorge Fernández-Fabeiro 2 
and José Carlos-Mouriño Gallego 2

Abstract

The technical support team of a supercomputing centre accumulates, over the course of decades, a large volume of resolved incidents that constitute critical operational knowledge. At the Galicia Supercomputing Center (CESGA) this history has been managed for over twenty years with Request Tracker (RT), whose built-in search engine has significant limitations that hinder knowledge reuse by the support staff. This paper presents  Fragata , a semantic ticket search system that combines modern information retrieval techniques with the full RT history. The system can find relevant past incidents regardless of language, the presence of typos, or the specific wording of the query. The architecture is deployed on CESGA’s infrastructure, supports incremental updates without service interruption, and offloads the most expensive stages to the FinisTerrae III supercomputer. Preliminary results show a substantial qualitative improvement over RT’s native search.

keywords 
RAG, automation, information retrieval, semantic search, HPC, Request Tracker, CESGA.

1  Introduction

High-Performance Computing (HPC) centres operate heterogeneous platforms hosting hundreds of scientific applications, compilers, parallel libraries, resource schedulers, and distributed storage systems. In this environment, the technical support team acts as a critical interface between research users and the infrastructure, resolving incidents ranging from MPI compilation errors to storage quotas, Conda environment configuration, or network failures on GPU nodes. The Galicia Supercomputing Center (CESGA) has managed this workflow for over two decades using Request Tracker (RT)  [ rt ] , a widely used open-source ticketing tool.

RT offers two main interaction channels: a web interface (available only to support staff) and e-mail. CESGA users interact with the system exclusively via e-mail, sending and replying to messages that RT automatically converts into tickets and associated responses. This dynamic has direct implications for the quality of the stored text, as detailed in Section

3.2  .

The history accumulated by CESGA in RT spans over twenty years of conversations between support staff and users, constituting an invaluable operational memory: a significant proportion of the incidents received today are variants of problems solved years ago. However, RT version 4.4.1 has severe limitations in its built-in search: it does not index the full ticket body, is case-sensitive, does not tolerate typos, does not normalize morphological variants, and lacks any notion of semantic similarity. As a result, both veteran and newly onboarded staff struggle to locate relevant past cases, leading to duplicated effort, loss of institutional knowledge, and increased mean resolution time.

This paper presents  Fragata , a semantic ticket retrieval system designed and deployed at CESGA to mitigate these limitations.  Fragata  applies the  Retrieval-Augmented Generation  (RAG) paradigm  [ lewis2020rag ] , an approach that improves search quality by first retrieving the most relevant documents from a knowledge base and presenting them as context. Specifically, the system combines dense retrieval based on  embeddings , numerical vector representations of text meaning, with classical BM25 lexical retrieval  [ robertson2009bm25 ]  and  reranking  via  cross-encoders

[ nogueira2019passage ] . The system also integrates complementary sources (the centre’s technical documentation, scientific application manuals, and repositories) and is deployed on a hybrid architecture comprising a virtual machine and the FinisTerrae III supercomputer, with incremental ingestion and hot-reload of the search engine.

The main contributions are: (i) the design of a reproducible protocol for extraction, normalization, and chunking of RT’s SQL history; (ii) a hybrid retrieval architecture with weighted fusion and query-aware reranking; (iii) a hot-swap mechanism that guarantees continuous availability during re-indexing; and (iv) operational integration with an HPC scheduler to offload expensive ingestion stages without penalizing service latency.

2  Related work

2.1  Neural information retrieval

Information retrieval has undergone a profound transformation in recent years thanks to  transformer -based models  [ vaswani2017attention ] . These neural network architectures, which underpin models such as BERT  [ devlin2019bert ] , learn to represent texts as dense numerical vectors ( embeddings ) that capture the semantic meaning of words and sentences. Two texts with similar meaning yield vectors that are close in the embedding space, enabling the retrieval of relevant documents by measuring vector distances rather than relying on exact word matches. In particular, Sentence-BERT  [ reimers2019sentencebert ]  adapted BERT to efficiently produce sentence-level embeddings, and dense retrieval systems such as DPR  [ karpukhin2020dpr ]  demonstrated that this approach can outperform traditional lexical search on question-answering tasks.

However, purely dense retrieval exhibits weaknesses on queries containing highly specific terminology, proper names, or technical identifiers, where exact word matching remains decisive. Consequently, hybrid approaches that combine BM25, a classical retrieval algorithm based on term frequencies, with dense retrieval  [ lin2021pyserini ]  have become the practical state of the art. These systems benefit especially from a subsequent  reranking  stage, in which a  cross-encoder

[ nogueira2019passage ]  jointly evaluates the query and each candidate to reorder the results with greater precision.

The RAG paradigm  [ lewis2020rag ,  gao2024ragsurvey ]  integrates this retrieval chain into a complete pipeline: given a query, the system retrieves the most relevant documents from a knowledge base. This approach leverages large volumes of information without needing to retrain models, making it especially well-suited to domains with accumulated knowledge such as technical support.

2.2  Technical support automation

The application of natural language processing techniques to ticketing systems has been explored in various contexts. Potharaju et al.  [ potharaju2013tickets ]  analysed corpora of network trouble tickets to infer problems automatically, while Zhou et al.  [ zhou2016ticketrouting ]  proposed machine learning methods for intelligent ticket routing. These works focus on classification and triage, but do not address semantic retrieval over accumulated ticket histories.

In the HPC domain specifically, published efforts have focused predominantly on job monitoring and automatic diagnosis systems, whereas the systematic exploitation of knowledge contained in support histories has received less attention.  Fragata  occupies precisely this niche: high-quality semantic retrieval over a heterogeneous corpus dominated by real technical e-mail, with the operational requirements of a production service.

3  Data sources and preparation

Figure 1:  Ticket processing pipeline: from SQL extraction of the RT history to the generation of  embeddings  for semantic search.

Figure

1

shows the overall data processing pipeline, which starts from the direct extraction of the RT history and culminates in the generation of vector representations ( embeddings ) that enable semantic search.

3.1  RT history dump

The limit