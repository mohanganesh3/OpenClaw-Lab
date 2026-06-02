# EV017: Application of Knowledge Distillation Based on Transfer Learning of ERNIE Model in Intelligent Dialogue Intention Recognition

URL: https://www.semanticscholar.org/paper/00313b801d269c67f070105ed8c7e781de63fa6c
Year: 2022
Source: semantic_scholar
Arxiv: n/a

## Abstract

The ‘intention’ classification of a user question is an important element of a task-engine driven chatbot. The essence of a user question’s intention understanding is the text classification. The transfer learning, such as BERT (Bidirectional Encoder Representations from Transformers) and ERNIE (Enhanced Representation through Knowledge Integration), has put the text classification task into a new level, but the BERT and ERNIE model are difficult to support high QPS (queries per second) intelligent dialogue systems due to computational performance issues. In reality, the simple classification model usually shows a high computational performance, but they are limited by low accuracy. In this paper, we use knowledge of the ERNIE model to distill the FastText model; the ERNIE model works as a teacher model to predict the massive online unlabeled data for data enhancement, and then guides the training of the student model of FastText with better computational efficiency. The FastText model is distilled by the ERNIE model in chatbot intention classification. This not only guarantees the superiority of its original computational performance, but also the intention classification accuracy has been significantly improved.
