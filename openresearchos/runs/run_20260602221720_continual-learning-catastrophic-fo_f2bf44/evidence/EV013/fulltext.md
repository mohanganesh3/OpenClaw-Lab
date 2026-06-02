[2601.02708] CREAM: Continual Retrieval on Dynamic Streaming Corpora with Adaptive Soft Memory

\setcctype

by

CREAM: Continual Retrieval on Dynamic Streaming Corpora with Adaptive Soft Memory

HuiJeong Son

huijeong.son@korea.ac.kr

Korea University  Seoul, Korea

,

Hyeongu Kang

hyeongu˙kang@korea.ac.kr

Korea University  Seoul, Korea

,

Sunho Kim

sunho˙kim@korea.ac.kr

Korea University  Seoul, Korea

,

Subeen Ho

hosubin02@korea.ac.kr

Korea University  Seoul, Korea

,

SeongKu Kang

seongkukang@korea.ac.kr

Korea University  Seoul, Korea

,

Dongha Lee

donalee@yonsei.ac.kr

Yonsei University  Seoul, Korea

and

Susik Yoon

susik@korea.ac.kr

Korea University  Seoul, Korea

(2026)

Abstract.

Information retrieval (IR) in dynamic data streams is a crucial task, as shifts in data distribution degrade the performance of AI-powered IR systems. To mitigate this issue, memory-based continual learning has been widely adopted for IR. However, existing methods rely on a fixed set of queries with ground-truth documents, which limits generalization to unseen data, making them impractical for real-world applications. To enable more effective learning with unseen topics of a new corpus without ground-truth labels, we propose CREAM, a self-supervised framework for memory-based continual retrieval. CREAM captures the evolving semantics of streaming queries and documents into dynamically structured soft memory and leverages it to adapt to both seen and unseen topics in an unsupervised setting. We realize this through three key techniques: fine-grained similarity estimation, regularized cluster prototyping, and stratified coreset sampling.
Experiments on two benchmark datasets demonstrate that CREAM exhibits superior adaptability and retrieval accuracy, outperforming the strongest method in a label-free setting by 27.79% in Success@5 and 44.5% in Recall@10 on average, and achieving performance comparable to or even exceeding that of supervised methods.

Information Retrieval, Continual Retrieval, Self-supervision.

†

†  copyright:  acmlicensed

†

†  journalyear:  2026

†

†  copyright:  cc

†

†  conference:  Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1; August 09–13, 2026; Jeju Island, Republic of Korea

†

†  booktitle:  Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1 (KDD ’26), August 09–13, 2026, Jeju Island, Republic of Korea

†

†  doi:  10.1145/3770854.3780281

†

†  isbn:  979-8-4007-2258-5/2026/08

†

†  ccs:  Information systems Novelty in information retrieval

Resource Availability:

The source code of this paper has been made publicly available at

10.6084/m9.figshare.30957539  .

Figure 1 .

Comparison of existing (top) and our (bottom) approaches for memory-based continual retrieval.

1.  Introduction

1.1.  Background

Information retrieval (IR) in online environments, powering real-time retrieval-augmented generation  (Lewis  et al. ,  2020 )  and agentic context engineering  (Zhang  et al. ,  2025 ) , is emerging as a key technology for various downstream applications. For example, in a real-time news summarization system  (Yoon  et al. ,  2023a ) , a query “ What are the current issues in the global supply chain? ” would require retrieving relevant articles covering current events, such as geopolitical conflicts or new tariff policies, at the time of the query. Identifying relevant documents with dynamically evolving topics is challenging, as pretrained retrieval models become outdated under domain shifts. This challenge is especially critical in real-world IR systems requiring timely and accurate responses, which is more pronounced in emerging agentic AI frameworks that facilitate real-time decision-making  (Sapkota  et al. ,  2025 ) .

Specifically, consider a scenario where topics gradually shift from the medical domain to the business domain. In a query “ Has the agent been approved? ”, the term  agent  typically refers to a drug in the medical domain, whereas it denotes a person or agency in the business context. If an IR system has not adapted to the new domain, it may return irrelevant medical documents, potentially leading to an incorrect answer such as “ The FDA approved the therapeutic agent. ” In contrast, if a system rapidly adapts to the emerging domain while retaining relevant knowledge from the previous domain, it is capable of retrieving appropriate business-related documents and producing a more appropriate answer, such as “ The licensing board approved the real estate agent’s application. ”

1.2.  Existing Efforts

In a typical IR system, an encoder is optimized to enhance the semantic similarity between query-document pairs labeled as relevant, with these labels obtained through human annotation. As illustrated in Figure

1  , when the distribution of queries and documents evolves over time with diverse topics, matching relevant pairs becomes increasingly difficult unless the encoder is continually updated to reflect the evolving corpora. In practice, web-scale corpora in typical IR systems involve a continuous influx of documents and queries, making retraining on all past data for training computationally inefficient and often infeasible. The naive incremental update of the encoder, however, suffers from catastrophic forgetting, a well-known issue in deep learning where previously acquired knowledge is overwritten by new information  (French,  1999 ) . To address this challenge, existing continual retrieval methods adopt memory-based continual learning strategies  (Rolnick  et al. ,  2019 ; Aljundi  et al. ,  2019a ,  b ; Yoon  et al. ,  2021 ; Cai  et al. ,  2023 ; Chen  et al. ,  2023 )  to acquire new knowledge without forgetting the old one.

As shown in the upper part of Figure

1  , existing methods with memory-based continual learning strategies employ dedicated storage for a fixed set of given queries and their corresponding ground-truth documents, which can be referred to as  hard memory . Query-document pairs are sampled from hard memory to update the encoder, while new documents relevant to the predefined queries are added from the streaming corpora. This approach has proven effective in scenarios where the topical distribution of the corpus remains relatively stable and consistent with the predefined query set  (Cai  et al. ,  2023 ) . However, simply reusing shift-unaware, predefined query-document pairs stored in hard memory for continual training can cause the encoder to learn information that is less relevant to the current topic distribution, leading to poor adaptation to distributional changes. Moreover, in real-time applications  (Yoon  et al. ,  2023b ) , human-curated supervision from a set of fixed queries with ground-truth documents is not always available in a timely manner  (Monarch,  2021 ) , making the hard memory strategy impractical. As a result, such methods fail to support effective retrieval on newly emerging topics, and may degrade performance on the initial or ongoing topics.

1.3.  Main Idea and Contributions

To address these practical limitations, we propose a novel concept called  soft memory  that can adapt to the ever-changing topical distributions of queries and documents. Soft memory dynamically tracks relevant queries and documents across varying topics, making it better than hard memory for adapting to evolving topic distributions, especially without supervision.
As illustrated in the bottom part of Figure

1  , semantically similar queries and documents in the streaming corpora are continuously grouped and expanded within the memory. For example, the soft memory may begin with creating a group of documents and queries representing initial topics, then add groups related to ongoing topics, and eventually incorporate new groups for emerging topics while phasing out older ones. By leveraging its dynamic structural representation of evolving topic distributions, soft memory enables self-su