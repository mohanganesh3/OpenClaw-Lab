# EV004: Patch-Based Contrastive Learning and Memory Consolidation for Online Unsupervised Continual Learning

URL: https://www.semanticscholar.org/paper/0015104cda9c12375dcecb6e1592ddee3d154d4e
Year: 2024
Source: semantic_scholar
Arxiv: 2409.16391

## Abstract

We focus on a relatively unexplored learning paradigm known as {\em Online Unsupervised Continual Learning} (O-UCL), where an agent receives a non-stationary, unlabeled data stream and progressively learns to identify an increasing number of classes. This paradigm is designed to model real-world applications where encountering novelty is the norm, such as exploring a terrain with several unknown and time-varying entities. Unlike prior work in unsupervised, continual, or online learning, O-UCL combines all three areas into a single challenging and realistic learning paradigm. In this setting, agents are frequently evaluated and must aim to maintain the best possible representation at any point of the data stream, rather than at the end of pre-specified offline tasks. The proposed approach, called \textbf{P}atch-based \textbf{C}ontrastive learning and \textbf{M}emory \textbf{C}onsolidation (PCMC), builds a compositional understanding of data by identifying and clustering patch-level features. Embeddings for these patch-level features are extracted with an encoder trained via patch-based contrastive learning. PCMC incorporates new data into its distribution while avoiding catastrophic forgetting, and it consolidates memory examples during ``sleep"periods. We evaluate PCMC's performance on streams created from the ImageNet and Places365 datasets. Additionally, we explore various versions of the PCMC algorithm and compare its performance against several existing methods and simple baselines.
