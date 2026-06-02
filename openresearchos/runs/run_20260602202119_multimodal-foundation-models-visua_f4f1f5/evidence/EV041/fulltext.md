[2505.16849] Walk&amp;Retrieve: Simple Yet Effective Zero-shot Retrieval-Augmented Generation via Knowledge Graph Walks

Walk&amp;Retrieve: Simple Yet Effective Zero-shot Retrieval-Augmented Generation via Knowledge Graph Walks

Martin Böckling

martin.boeckling@uni-mannheim.de

0000-0002-1143-4686

University of Mannheim  Mannheim  Germany

,

Heiko Paulheim

heiko.paulheim@uni-mannheim.de

0000-0003-4386-8195

University of Mannheim  Mannheim  Germany

and

Andreea Iana

andreea.iana@uni-mannheim.de

0000-0002-7248-7503

University of Mannheim  Mannheim  Germany

(2025)

Abstract.

Large Language Models (LLMs) have showcased impressive reasoning abilities, but often suffer from hallucinations or outdated knowledge. Knowledge Graph (KG)-based Retrieval-Augmented Generation (RAG) remedies these shortcomings by grounding LLM responses in structured external information from a knowledge base. However, many KG-based RAG approaches struggle with (i) aligning KG and textual representations, (ii) balancing retrieval accuracy and efficiency, and (iii) adapting to dynamically updated KGs.
In this work, we introduce  Walk&amp;Retrieve , a simple yet effective KG-based framework that leverages walk-based graph traversal and knowledge verbalization for corpus generation for zero-shot RAG. Built around efficient KG walks, our method does not require fine-tuning on domain-specific data, enabling seamless adaptation to KG updates, reducing computational overhead, and allowing integration with any off-the-shelf backbone LLM. Despite its simplicity,  Walk&amp;Retrieve  performs competitively, often outperforming existing RAG systems in response accuracy and hallucination reduction. Moreover, it demonstrates lower query latency and robust scalability to large KGs, highlighting the potential of lightweight retrieval strategies as strong baselines for future RAG research.

Knowledge Graph Retrieval-Augmented Generation, Graph Walks, Zero-Shot Retrieval, Question Answering

†

†  copyright:  acmlicensed

†

†  journalyear:  2025

†

†  doi:  XXXXXXX.XXXXXXX

†

†  conference:  Information Retrieval’s Role in RAG Systems; July 17,
2025; Padua, Italy

†

†  ccs:  Information systems Information retrieval

†

†  ccs:  Information systems Language models

†

†  ccs:  Information systems Question answering

1.  Introduction

Large Language Models (LLMs) are pivotal to question answering (QA) due to their strong language understanding and text generation capabilities  (Brown et al.,  2020 ; Ouyang et al.,  2022 ; Zhao et al.,  2023 ; Liu et al.,  2023 ; Touvron et al.,  2023 ; Liévin et al.,  2024 ) .
However, LLMs often (i) struggle with outdated knowledge, (ii) lack interpretability due to their black-box nature  (Danilevsky et al.,  2020 ) , and (iii) can hallucinate convincingly yet factually inaccurate answers  (Rawte et al.,  2023 ; Ji et al.,  2023 ; Huang et al.,  2024 ) . These issues are particularly pronounced in knowledge-intensive tasks  (Mallen et al.,  2023 ) , when dealing with domain-specific  (Tonmoy et al.,  2024 ; Sun et al.,  2024b )  or rapidly changing knowledge  (Vu et al.,  2024 ) .
Retrieval-augmented generation (RAG) mitigates these limitations by grounding responses in relevant external information  (Lewis et al.,  2020 ; Gao et al.,  2023 ; Fan et al.,  2024 ) .
Yet, text-based RAG primarily relies on semantic similarity search of textual content  (Fan et al.,  2024 ) , which fails to capture the relational knowledge necessary to integrate passages with large semantic distance from the query in multi-step reasoning  (Larson and Truitt,  2024 ; Chen et al.,  2024 ; Jin et al.,  2024 ; Peng et al.,  2024 ; Ma et al.,  2024 ) .

Consequently, several works leverage knowledge graphs (KGs) – structured knowledge bases representing real-world information as networks of entities and relations  (Hogan et al.,  2021 )  – as external information sources to overcome standard RAG limitations  (Peng et al.,  2024 ) .
Given a query, KG-based RAG systems retrieve relevant facts as nodes, triplets, paths, or subgraphs using graph search algorithms, or parametric retrievers based on graph neural networks or language models  (Peng et al.,  2024 ) . The retrieved graph data is then reformatted for the language model – via linearized triples  (Kim et al.,  2023 ) , natural language descriptions  (Wu et al.,  2023 ; Li et al.,  2023 ; Wen et al.,  2023 ; Edge et al.,  2024 ; Fatemi et al.,  2024 ) , code-like forms  (Guo et al.,  2023 ) , or node sequences  (Luo et al.,  2024 ; Sun et al.,  2024a ; Mavromatis and Karypis,  2024 )  – and finally used by an LLM to generates the final response  (Peng et al.,  2024 ) .

Figure 1 .

Overview of the  Walk&amp;Retrieve  framework: (1) We combine walk-based graph traversal with knowledge verbalization for corpus generation; (2) The answer is generated with a prompt augmenting the query with the most similar verbalized walks.

The existing body of work exhibits several drawbacks.
First, augmenting a query with relevant KG triples  (Sen et al.,  2023 ; Saleh et al.,  2024 ; Li et al.,  2025 )  can lead to suboptimal retrieval performance due to the misalignment of structured graphs and the sequential token-based nature of the language model.
Although converting KG data to a LLM-suitable tokenized format can help, naive triple linearization  (He et al.,  2024 ; Baek et al.,  2023 ) , which directly converts KG triples into plain text without considering context, coherence, or structural nuances, often produces semantically incoherent descriptions  (Wu et al.,  2024 ) .  1

1  1 Given the triples:  A Fistful of Dollars

→  \rightarrow

writtenBy

→  \rightarrow

Sergio Leone , and  The Godfather Part II

→  \rightarrow

sequelOf

→  \rightarrow

The Godfather , a prompt based on naive linearization would be:  These facts might be relevant to answer the question: (A Fistful of Dollars, writtenBy, Sergio Leone), (The Godfather Part II, sequelOf, The Godfather) […].

Second, RAG systems that directly reason over KGs with LLMs perform a step-by-step graph traversal for fact retrieval  (Jin et al.,  2024 ; Sun et al.,  2024a ; Ma et al.,  2024 ) . This requires multiple LLM calls per query, significantly increasing complexity and latency.
Third, KG-based RAG models often fine-tune retrievers  (Wu et al.,  2023 ; Guo et al.,  2024 ; Luo et al.,  2024 )  or generators  (Yasunaga et al.,  2021 ; He et al.,  2024 ; Hu et al.,  2024 ; Luo et al.,  2024 ; Mavromatis and Karypis,  2024 )  on task-specific data to better adapt to diverse KG structures and vocabularies. However, collecting high-quality instruction data is costly  (Cao et al.,  2023 ) , and fine-tuning large models – even with parameter-efficient methods  (Hu et al.,  2022 ; Chai et al.,  2023 ; Perozzi et al.,  2024 )  – is expensive and limits generalization to dynamic KGs or unseen domains  (Li et al.,  2025 ; Wu et al.,  2023 ) .

Contributions. 
We propose  Walk&amp;Retrieve , a lightweight zero-shot KG-based RAG framework, designed as a simple yet competitive baseline to address these challenges.
It combines efficient graph traversal, via random or breadth-first search walks, with verbalization of KG-derived information to build a contextual corpus of relevant facts for each KG entity. At inference, we retrieve the most similar nodes to the query, and their corresponding walks, respectively. We generate the final answer by prompting an LLM with the query, augmented with this relevant context.
Unlike many existing KG-based RAG systems,  Walk&amp;Retrieve :
 (1)  is  adaptable  to dynamic KGs – updates (e.g., node insertion or deletion) require no retraining, as new knowledge can be added by incrementally generating additional walks;
 (2)  is more  efficient , requiring no fine-tuning of the backbone LLM, and only a single LLM call per query;
 (3)  enables  zero-shot RAG  with any  off-the-shelf LLM .
We show that  Walk&amp;Retrieve  consistently generates accurate resp