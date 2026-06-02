# EV020: Addressing LLM Challenges: A Hybrid Framework for Duplicate Question Detection

URL: https://www.semanticscholar.org/paper/027ffa80b1de993967c1ff85502fa7219f7400ff
Year: 2024
Source: semantic_scholar
Arxiv: n/a

## Abstract

Detecting similar or duplicate texts or questions is a critical subtask in information retrieval. Despite the remarkable advancements achieved by large language models (LLMs) in the question answering (QA) domain in recent years, challenges such as unreliable responses caused by LLM hallucinations and outdated information have highlighted the continued importance of information retrieval techniques. The integration of retrieval-augmented generation (RAG) with LLMs has mitigated some of these limitations, enhancing the reliability of applying LLMs to domain-specific knowledge. This paper uses Chinese educational questions as a case study and introduces an innovative framework for duplicate question detection, termed the Balanced Duplicate Question Detector (BDQD). The BDQD incorporates multiple extensible detectors and a machine learning-based balancer. Experimental results reveal that detecting duplicate questions involves an inherent trade-off between precision and recall. Logistic regression proves effective in balancing thresholds across different detectors, establishing a more stable and reliable standard for duplicate question determination. Finally, we propose a hybrid QA framework that considers both cost and efficiency, integrating the lightweight question retrieval architecture developed in this study with RAG and LLMs. This framework offers practical recommendations for building robust and efficient QA systems.
