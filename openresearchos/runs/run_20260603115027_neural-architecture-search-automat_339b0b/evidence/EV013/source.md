# EV013: ENASFL: A Federated Neural Architecture Search Scheme for Heterogeneous Deep Models in Distributed Edge Computing Systems

URL: https://www.semanticscholar.org/paper/008687f49bd7b6b4aeda65f7e9ab76923119f7f5
Year: 2024
Source: semantic_scholar
Arxiv: n/a

## Abstract

With the development of edge computing, deep neural networks (DNNs) are deployed in heterogeneous edge devices, which have considerable differences in equipment attributes. This makes it necessary to design and train model structures to ensure each deployment's accuracy, even when dealing with the same image recognition task. Such a huge project will consume a lot of human and computing resources. To automate the design, Neural Architecture Search (NAS) is proposed to explore the possibly efficient structure. To maintain parameters in situ, Federated Learning (FL) can train models cooperatively without disclosing the original data. However, combining NAS and FL is still in its infancy. This paper proposes an Edge-empowered Neural Architecture Search Scheme with Federated Learning (ENASFL), which can serve edge devices with different lightweight models. For the reusability of blocks in different models, our work searches for a multi-exit model, then selects the appropriate branch for each edge device. Further, we design a novel aggregation strategy for the cooperative training of these branches, which enables devices to work cooperatively and improve training efficiency. Experimental evaluations based on three real datasets demonstrate that the accuracy is improved by 3% on average when the model has compressed to $4\times$ the baseline network.
