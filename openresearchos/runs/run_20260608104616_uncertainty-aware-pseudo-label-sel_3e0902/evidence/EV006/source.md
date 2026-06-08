# EV006: FSSL-UC: Federated Semi-Supervised Learning Model With Uncertainty and Consistency

URL: https://www.semanticscholar.org/paper/087c69856393a4d9968a95346e0902548e679048
Year: 2026
Source: semantic_scholar
Arxiv: n/a

## Abstract

Federated learning (FL) in practical applications frequently encounters the challenge of limited labeled data availability on client devices, where data annotation is often expensive or requires domain expertise. This scarcity, especially under nonindependent and nonidentically distributed (non-IID) settings, leads to error accumulation from pseudo-labels and under-utilization of unlabeled data. To overcome this limitation, we propose a federated semi-supervised learning (FSSL) approach that jointly uses labeled and unlabeled data distributed across both server and clients. Building upon existing FSSL frameworks, our method introduces two key innovations: 1) an optimized client consistency regularization mechanism using Kullback–Leibler (KL) divergence for low-confidence samples; and 2) an advanced pseudo-labeling strategy incorporating entropy meaning loss (EML) for high-confidence samples, which are designed to maximize the utility of unlabeled data and enhance model generalization capabilities. Our classification experiments on common (CIFAR-10, CIFAR-100, MNIST, and SVHN) and medical datasets (brain tumor and nail disease) show that the proposed method outperforms the existing baseline models. On CIFAR-10, with only 10% labeled data [independent and identically distributed (IID)], the proposed FSSL-UC achieves a 1.33% higher accuracy than FedMatch, while under the more challenging setting of 1% labeled data (non-IID), the improvement reaches 6.33%. On the brain tumor dataset, with 1% labeled data, FSSL-UC outperforms FedMatch by 14.47% (IID) and 7.61% (non-IID). In addition, FSSL-UC also showed stable advantages on MNIST, SVHN, and nail disease datasets, demonstrating its robustness and generalization ability.
