# EV013: CREAM: Continual Retrieval on Dynamic Streaming Corpora with Adaptive Soft Memory

URL: https://www.semanticscholar.org/paper/006a9f5b8180244df4a69b4577b744bfc68624e4
Year: 2026
Source: semantic_scholar
Arxiv: 2601.02708

## Abstract

Information retrieval (IR) in dynamic data streams is a crucial task, as shifts in data distribution degrade the performance of AI-powered IR systems. To mitigate this issue, memory-based continual learning has been widely adopted for IR. However, existing methods rely on a fixed set of queries with ground-truth documents, which limits generalization to unseen data, making them impractical for real-world applications. To enable more effective learning with unseen topics of a new corpus without ground-truth labels, we propose CREAM, a self-supervised framework for memory-based continual retrieval. CREAM captures the evolving semantics of streaming queries and documents into dynamically structured soft memory and leverages it to adapt to both seen and unseen topics in an unsupervised setting. We realize this through three key techniques: fine-grained similarity estimation, regularized cluster prototyping, and stratified coreset sampling. Experiments on two benchmark datasets demonstrate that CREAM exhibits superior adaptability and retrieval accuracy, outperforming the strongest method in a label-free setting by 27.79% in Success@5 and 44.5% in Recall@10 on average, and achieving performance comparable to or even exceeding that of supervised methods.
