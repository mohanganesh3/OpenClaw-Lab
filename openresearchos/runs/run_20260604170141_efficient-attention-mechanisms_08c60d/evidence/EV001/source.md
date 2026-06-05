# EV001: Improving Commonsense Question Answering by Graph-based Iterative Retrieval over Multiple Knowledge Sources

URL: https://www.semanticscholar.org/paper/001bcf2306afe90b5cf54cd0f04f8a808d7d134f
Year: 2020
Source: semantic_scholar
Arxiv: 2011.02705

## Abstract

In order to facilitate natural language understanding, the key is to engage commonsense or background knowledge. However, how to engage commonsense effectively in question answering systems is still under exploration in both research academia and industry. In this paper, we propose a novel question-answering method by integrating multiple knowledge sources, i.e. ConceptNet, Wikipedia, and the Cambridge Dictionary, to boost the performance. More concretely, we first introduce a novel graph-based iterative knowledge retrieval module, which iteratively retrieves concepts and entities related to the given question and its choices from multiple knowledge sources. Afterward, we use a pre-trained language model to encode the question, retrieved knowledge and choices, and propose an answer choice-aware attention mechanism to fuse all hidden representations of the previous modules. Finally, the linear classifier for specific tasks is used to predict the answer. Experimental results on the CommonsenseQA dataset show that our method significantly outperforms other competitive methods and achieves the new state-of-the-art. In addition, further ablation studies demonstrate the effectiveness of our graph-based iterative knowledge retrieval module and the answer choice-aware attention module in retrieving and synthesizing background knowledge from multiple knowledge sources.
