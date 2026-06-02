[2502.19596] Trustworthy Answers, Messier Data: Bridging the Gap in Low-Resource Retrieval-Augmented Generation for Domain Expert Systems

Trustworthy Answers, Messier Data: Bridging the Gap in Low-Resource Retrieval-Augmented Generation for Domain Expert Systems

Nayoung Choi  1

,  Grace Byun  1

,  Andrew Chung  1

,  Ellie S. Paek  1

,

Shinsun Lee 1,2  ,  Jinho D. Choi 1

1 Emory University,  2 Hyundai Motor Company

{nayoung.choi, grace.byun, andrew.chung, ellie.paek, shinsun.lee, jinho.choi}@emory.edu

Work done while at Emory University as a visiting scholar.

Abstract

RAG has become a key technique for enhancing LLMs by reducing hallucinations, especially in domain expert systems where LLMs may lack sufficient inherent knowledge. However, developing these systems in low-resource settings introduces several challenges: (1) handling heterogeneous data sources, (2) optimizing retrieval phase for trustworthy answers, and (3) evaluating generated answers across diverse aspects. To address these, we introduce a data generation pipeline that transforms raw multi-modal data into structured corpus and Q&amp;A pairs, an advanced re-ranking phase improving retrieval precision, and a reference matching algorithm enhancing answer traceability. Applied to the automotive engineering domain, our system improves factual correctness (+1.94), informativeness (+1.16), and helpfulness (+1.67) over a non-RAG baseline, based on a 1-5 scale by an LLM judge. These results highlight the effectiveness of our approach across distinct aspects, with strong answer grounding and transparency.

Trustworthy Answers, Messier Data: Bridging the Gap in Low-Resource Retrieval-Augmented Generation for Domain Expert Systems

Nayoung Choi 1 , Grace Byun 1 , Andrew Chung 1 , Ellie S. Paek 1 ,

Shinsun Lee  †

†  thanks:

Work done while at Emory University as a visiting scholar.

1,2  ,  Jinho D. Choi 1

1 Emory University,  2 Hyundai Motor Company

{nayoung.choi, grace.byun, andrew.chung, ellie.paek, shinsun.lee, jinho.choi}@emory.edu

1  Introduction

Retrieval-Augmented Generation (RAG) has shown potential in reducing hallucinations and providing up-to-date knowledge in Large Language Models (LLMs). This success has grown interest in domain expert RAG-Question Answering (QA) systems to meet specialized knowledge needs. While previous studies ( Han et al.  2024  ;  Siriwardhana et al.  2023  ;  Mao et al.  2024  ) have proposed general methods for adapting RAG models to domain knowledge bases—such as syntactic QA pair generation or model fine-tuning—they face several challenges in low-resource domains.

In practical settings, available data sources in low-resource domains are often presented in heterogeneous formats and exhibit an unstructured nature, making their direct integration into RAG system development challenging  Hong et al. ( 2024 ) . General models may not have enough inherent knowledge in low-resource domains  Zhao et al. ( 2024 ) , making fine-tuning essential to adapt the model to specific knowledge requirements. However, the lack of structured data for training further complicates this process. In addition, data privacy and security concerns restrict the full utilization of API-based LLMs ( Achiam et al.  2023  ;  Anthropic  2024  ) within RAG systems, necessitating the use of open-source LLMs ( Yang et al.  2024  ;  Touvron et al.  2023  ).

The retrieval phase is another key aspect of domain expert RAG systems, as referencing accurate documents is essential for generating reliable answers. However, research on improving ranking in the retrieval phase or tracing the documents referenced to generate the answer remains limited. Most domain expert RAG frameworks rely on a single-stage retrieval process, with few exploring multi-stage approaches ( Nogueira et al.  2019  ;  Nogueira et al.  2020  ;  Karpukhin et al.  2020  ), such as retrieval followed by re-ranking—a method widely used in Information Retrieval (IR)—which can help ensure the use of the most relevant references.

The evaluation of RAG systems is also an area that has not been fully addressed. Many studies continue to rely on gold answer similarity metrics, such as overlapping words between the generated answer and the ground truth, which inadequately capture critical dimensions like faithfulness, coherence, and contextual relevance. Recently, the LLM-as-a-judge framework  Zheng et al. ( 2023 )  has gained attention as a qualitative alternative. However, these methods often overlook diverse evaluation aspects, and a standardized framework for assessing RAG systems has yet to emerge.

Figure 1:  When a user question is given, our RAG system retrieves and re-ranks the text chunks, generates answers using the top

k  k

relevant chunks, and ensures each part of the answer is backed by clear references The example shown was originally in Korean and translated into English.

In this work, we address three key challenges and present the RAG development pipeline, demonstrating its application in the automotive engineering domain, with a specific focus on QA for vehicle crash collision tests. First, we present a data generation pipeline (Section

3.1  ), leveraging diverse formats of internal documents from an automobile company. Second, we incorporate an advanced re-ranking phase (Section

3.2  ) and a reference-matching algorithm (Section

3.4  ), not only enhancing the accuracy of the final answers but also improving traceability by tagging sources for each segment of the answer. Third, we evaluate the answers obtained from our generation model (Section

3.3  ) from multiple perspectives, emphasizing their qualitative aspects. Our ultimate goal is a fully local system, ensuring data privacy and independence from external servers. The overall workflow of our RAG system is shown in Figure

1  , and our key contributions are summarized as follows:

•

We propose a data generation pipeline that transforms multi-modal raw data into a structured corpus and high-quality Q&amp;A pairs.

•

We integrate re-ranking and reference matching to enhance retrieval precision and answer traceability, ensuring a reliable RAG system.  1

1  1 The full RAG pipeline code will be publicly available:  https://github.com/anonymous

•

We assess the final answer from diverse qualitative perspectives, evaluating each along distinct, non-overlapping dimensions.

2  Related Work

Data Processing

RAG-Studio ( Mao et al.  2024  ) employs synthetic data generation for in-domain adaptation, reducing reliance on costly human-labeled datasets. However, it assumes access to well-structured data, limiting its applicability in scenarios with unstructured raw data. To bridge this gap,  Hong et al. ( 2024 )  tackle challenges with real-world formats (e.g., DOC, HWP), proposing a chunking method that converts documents to HTML, retaining structural information in low-resource settings. Meanwhile,  Guan et al. ( 2024 )  address the issue of short and noisy e-commerce datasets in RAG system development by building a new dataset from a raw corpus crawled from an e-commerce website, providing a richer resource.

Retrieval in RAG

Wang et al. ( 2024 )  highlight the importance of re-ranking modules in RAG systems to enhance the relevance of the retrieved documents. Similarly,  Zhao et al. ( 2024 )  demonstrate that the ranking position of the gold document in the retrieval phase plays a significant role in determining the quality of the final answer. Given these insights, optimizing the retrieval phase is crucial for obtaining accurate context, particularly in specialized, low-resource domains where LLMs lack sufficient inherent knowledge  Beauchemin et al. ( 2024 ) . Despite these findings, the analysis of the effectiveness and applicability of re-rankers in domain expert RAG systems remains underexplored.

RAG Evaluation

The evaluation of RAG systems  Yu et al. ( 2024 )  has relied on text similarity-based metrics such as BLEU  Papineni et al.