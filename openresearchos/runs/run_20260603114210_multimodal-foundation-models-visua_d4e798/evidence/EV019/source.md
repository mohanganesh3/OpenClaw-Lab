# EV019: Distildoc: Deep Reinforcement Learning for Token-Efficient Multimodal Document Question Answering

URL: https://www.semanticscholar.org/paper/09aa7a7661baec4c1e8235bb2eac5f11bc9722df
Year: 2026
Source: semantic_scholar
Arxiv: n/a

## Abstract

Large Language Models (LLM) are now being used for document-based question answering  across numerous areas such as healthcare, education, and science. The primary problem associated with the use of LLM's is the increase in inference costs based on the number of tokens used for input. This issue tends to become more problematic when dealing with documents that include text, structured tables, and visual content. Although retrieval-augmented generation (RAG) can improve factual grounding, it will result in higher token usage due to large, heterogeneous context. The existing solutions to this problem like LeanContext are geared towards text-only retrieval and utilize tabular Q-learning methods that will not scale well within a high dimensional multimodal environment. To address these issues, we propose a new multimodal question-answering framework called DistilDoc, which is based on deep reinforcement learning techniques designed to be token-efficient. DistilDoc uses an integrated embedding space to consolidate text chunks, each serialized row of a table, and captions of images. So, the context compression process can be modelled as a sequential decision-making problem. DistilDoc uses an off-policy Deep Q-network (DQN) with a state representation of 4608 dimensions to learn optimum compression strategies over 128 possible actions. Experimental evaluation on Arxiv papers, OpenStax textbooks, and WHO reports demonstrates that DistilDoc reduces token usage by 55.86% with a 0.6 drop in sematic similarity. These results highlight the effectiveness of Deep Q-Network on multimodal context compression in improving RAG efficiency.
