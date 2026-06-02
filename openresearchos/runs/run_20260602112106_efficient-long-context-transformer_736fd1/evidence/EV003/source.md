# EV003: Augmenting Security Logs with Artificial Intelligence: Are Deep Models the Missing Piece?

URL: https://www.semanticscholar.org/paper/000a0d1374df550192fadd04f71b3c27c2250de1
Year: 2025
Source: semantic_scholar
Arxiv: n/a

## Abstract

The analysis of security logs remains a major challenge for modern Security Information and Event Management (SIEM) systems due to insufficient standardization and diversity of log formats. While Artificial Intelligence (AI) offers great potential for automating monitoring, its use is limited by data sensitivity and a lack of annotated datasets. Augmentation can help generate realistic synthetic logs, providing broader opportunities for AI deployment. This article presents a framework for training language models to generate structured log variants, focusing on key metadata fields while maintaining syntactic consistency and semantic relevance. This framework increases data diversity, reduces the need for manual labeling, and facilitates the integration of AI into Security Operations Centers (SOCs), thereby enhancing operational efficiency. A heterogeneous corpus from 49 sources was cleaned, deduplicated, and transformed into semantically distinct entities. Two augmentation strategies were evaluated: Masked Language Modeling (MLM) and Next Word Prediction (NWP). Eight transformer-based models were finetuned and tested on simulated attack scenarios generated using the Atomic Red Team framework and compared with largescale models to assess accuracy and computational efficiency. The results demonstrate the potential of domain-specific language models for context-aware protocol augmentation, contributing to more efficient and automated security systems.
