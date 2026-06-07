# EV006: Robust Surgical Tools Detection in Endoscopic Videos with Noisy Data

URL: https://www.semanticscholar.org/paper/0f185f21a8ba68b4b1078ad9d8c46f509687fe61
Year: 2023
Source: semantic_scholar
Arxiv: 2307.01232

## Abstract

Over the past few years, surgical data science has attracted substantial interest from the machine learning (ML) community. Various studies have demonstrated the efficacy of emerging ML techniques in analysing surgical data, particularly recordings of procedures, for digitizing clinical and non-clinical functions like preoperative planning, context-aware decision-making, and operating skill assessment. However, this field is still in its infancy and lacks representative, well-annotated datasets for training robust models in intermediate ML tasks. Also, existing datasets suffer from inaccurate labels, hindering the development of reliable models. In this paper, we propose a systematic methodology for developing robust models for surgical tool detection using noisy data. Our methodology introduces two key innovations: (1) an intelligent active learning strategy for minimal dataset identification and label correction by human experts; and (2) an assembling strategy for a student-teacher model-based self-training framework to achieve the robust classification of 14 surgical tools in a semi-supervised fashion. Furthermore, we employ weighted data loaders to handle difficult class labels and address class imbalance issues. The proposed methodology achieves an average F1-score of 85.88\% for the ensemble model-based self-training with class weights, and 80.88\% without class weights for noisy labels. Also, our proposed method significantly outperforms existing approaches, which effectively demonstrates its effectiveness.
