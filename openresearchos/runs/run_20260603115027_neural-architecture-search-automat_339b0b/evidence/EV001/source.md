# EV001: Probeable DARTS with Application to Computational Pathology

URL: https://www.semanticscholar.org/paper/000cad8dfc9c3aa602688f1baafbd3a6f2e504ec
Year: 2021
Source: semantic_scholar
Arxiv: 2108.06859

## Abstract

AI technology has made remarkable achievements in computational pathology (CPath), especially with the help of deep neural networks. However, the network performance is highly related to architecture design, which commonly requires human experts with domain knowledge. In this paper, we combat this challenge with the recent advance in neural architecture search (NAS) to find an optimal network for CPath applications. In particular, we use differentiable architecture search (DARTS) for its efficiency. We first adopt a probing metric to show that the original DARTS lacks proper hyperparameter tuning on the CIFAR dataset, and how the generalization issue can be addressed using an adaptive optimization strategy. We then apply our searching framework on CPath applications by searching for the optimum network architecture on a histological tissue type dataset (ADP). Results show that the searched network outperforms state-of-the-art networks in terms of pre-diction accuracy and computation complexity. We further conduct extensive experiments to demonstrate the transfer-ability of the searched network to new CPath applications, the robustness against downscaled inputs, as well as the reliability of predictions.
