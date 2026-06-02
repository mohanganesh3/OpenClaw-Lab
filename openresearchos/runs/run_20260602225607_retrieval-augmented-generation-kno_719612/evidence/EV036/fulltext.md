[2504.05324] Hybrid Retrieval for Hallucination Mitigation in Large Language Models: A Comparative Analysis

Hybrid Retrieval for Hallucination Mitigation in Large Language Models: A Comparative Analysis

Chandana sree Mala

Corresponding Author: Chandana sree Mala, email: [c.mala@studenti.unipi.it, chandana.mala@sns.it], ORCiD: https://orcid.org/0009-0004-7500-6121

Gizem Gezici

Fosca Giannotti

Department of Computer Science, University of Pisa

Department of Computer Science, Scuola Normale Superiore

Abstract

Large Language Models (LLMs) excel in language comprehension and generation but are prone to hallucinations, producing factually incorrect or unsupported outputs. Retrieval-Augmented Generation (RAG) systems mitigate this by grounding LLM responses with external knowledge. This study evaluates the relationship between retriever effectiveness and hallucination reduction in LLMs using three retrieval approaches: sparse retrieval (BM25-based keyword search), dense retrieval (semantic search with Sentence Transformers), and the proposed hybrid retrieval module which incorporates information from query expansion and further fuses the results of sparse and dense retrievers through a dynamically-weighted Reciprocal Rank Fusion (RRF) score.
Using the HaluBench dataset, a benchmark for hallucinations in Question Answering tasks, we assess retrieval performance with MAP and NDCG metrics, focusing on the relevance of the top-3 retrieved documents. Results show that the hybrid retriever has a better relevance score outperforming both sparse and dense retrievers. Further evaluation of LLM-generated answers against ground truth using metrics like accuracy, hallucination rate, and rejection rate reveals that the hybrid retriever achieves the highest accuracy on fails, the lowest hallucination rate, and the lowest rejection rate. These findings highlight the hybrid retriever’s ability to enhance retrieval relevance, reduce hallucination rates, and improve LLM reliability, emphasizing the importance of advanced retrieval techniques in mitigating hallucinations and improving response accuracy.

keywords:

Retrieval Augmented Generation, Large Language Models, Hallucination Mitigation, Retrieval Performance, Query Expansion, HaluBench

1  Introduction

Advancements in natural language processing (NLP) have brought large language models to the forefront, revolutionizing both academic research and practical applications in diverse domains.
RAG is an approach that enhances LLMs by integrating retrieval mechanisms to improve response accuracy and reduce hallucinations  [ 1 ] .
Instead of relying solely on the model’s internal knowledge, RAG retrieves relevant external documents from a knowledge source (e.g., databases, search engines, or vector stores) and incorporates them into the generation process.
By integrating retrieval mechanisms from external sources, RAG effectively addresses major limitations of standalone LLMs  [ 2 ,  3 ] , including the high costs associated with training and fine-tuning  [ 4 ] , the issue of hallucination  [ 5 ,  6 ,  7 ,  8 ] , and constraints imposed by the input window  [ 9 ]  and knowledge cut-off [ 1 ] .
Moreover, RAG has already become a foundational technology in various real-world products like Contextual AI [14] and Cohere [15].

RAG system blends the encyclopedic memory of a search engine with the generative models and consists of two main modules as the retrieval phase (R) and the generation phase (G). In the retrieval phase, a retriever fetches relevant documents based on the input query using three retrieval approaches: a sparse retriever leveraging (

B  ​  M  ​  25

BM25

[ 10 ] -based lexical matching), a dense retriever(using embeddings from Sentence Transformers), or a hybrid approach (combining both methods). These retrieval algorithms have been inspired from Information Retrieval (IR), where search systems seek for alternative retrieval approaches to satisfy the information need of users, i.e. retrieving the most relevant documents at the top positions of a ranked list with respect to a given user query [ 11 ] .
Many popular web search engines employ

B  ​  M  ​  25

BM25

or similar ranking algorithms to determine the relevance of search results for a given query.

This paper explores the effectiveness of different retrieval methods in reducing hallucinations.
Note that  hallucinations  occur when the generated answers are not faithful to the context (intrinsic hallucinations) or don’t align with factual reality (extrinsic hallucinations)  [ 12 ,  13 ] . In this paper, we focus solely on intrinsic hallucinations since in real-world settings, user-provided documents may contain information that conflicts with external knowledge sources.

To the best of our knowledge, this is the first study that evaluates the hybrid retrieval performance in mitigating hallucinations.
Our main contributions are as follows:

•

We use a query expansion module to increase the coverage of the hybrid retrieval phase.

•

We evaluated how different types of retrieval performance affect hallucinations in LLM generated outputs.

This paper is organized by introducing the motivation behind reducing hallucinations in LLMs through Retrieval-Augmented Generation. The second section surveys recent RAG studies, highlighting key retrieval strategies and their relevance to mitigating hallucinations. In the third section, we detail our hybrid retrieval methodology, underscoring query expansion and dynamic weighting. The fourth section outlines the experimental setup and results on the dataset, and the paper concludes with final observations on the effectiveness of the proposed hybrid retriever followed by future work.

2  Related Work

RAG systems have emerged as a promising solution to the inherent limitations of LLMs, particularly their tendency to hallucinate or generate inaccurate information  [ 14 ,  15 ] . By integrating retrieval mechanisms, RAG systems retrieve relevant external knowledge during the retrieval phase, which is then incorporated into the query. This ensures that the LLM’s generated output is informed by up-to-date and contextually relevant information [ 16 ] .

Early work in  [ 17 ]  and  [ 8 ]  demonstrated that complementing LLMs with specialized retrievers can substantially ground the generated text in factual evidence.This has spurred research into a variety of domain-specific and application-specific RAG approaches, such as  [ 18 ,  19 ] , where sophisticated modules decrease hallucinations by parsing industry abbreviations and consolidating context from heterogeneous sources.

Additionally,  [ 20 ,  21 ]  and  [ 22 ,  23 ]  illustrate both benchmark comparisons and methodological guides for improving retrieval accuracy, with an emphasis on ensuring that even black-box LLMs can trace back to reliable evidence like discussed in this paper [ 24 ] .

Recent research has focused on enhancing the efficiency and performance of RAG systems by improving their retrieval components like discussed in this papers [ 25 ]  and  [ 26 ,  16 ]  highlight how fusing dense and sparse retrieval signals yields higher relevance in challenging Q&amp;A contexts [ 27 ] .

This fusion approach is further explored in  [ 28 ]  and  [ 29 ,  30 ] , where rank fusion, weighted scoring, and dynamic weighting strategies emerge as key factors for precise, context-rich retrieval. Contributions such as  [ 2 ,  3 ,  31 ,  32 ]  and offer an analytical lens through which prompt optimization, domain adaptation, and query expansion recommender modules, most recent paper [ 22 ]  demonstrate that by expanding the query to relevant fields may enhance response quality by improving the relevance of the retrieved information which can further reduce irrelevance or hallucinations.

Despite considerable progress in hybrid retrieval and RAG systems, gaps remain in understanding how retrieval approaches dynamically adapt to specific query scenarios and how these a