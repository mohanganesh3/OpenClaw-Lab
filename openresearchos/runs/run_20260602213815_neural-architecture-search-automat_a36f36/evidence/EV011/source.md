# EV011: Efficient Self-Supervised Neural Architecture Search

URL: https://www.semanticscholar.org/paper/017f215a0e772dae8bab423c13c11ab20419b4e3
Year: 2025
Source: semantic_scholar
Arxiv: n/a

## Abstract

Deep Neural Networks (DNNs) have successfully demonstrated superior performance on many tasks across multiple domains. Their success is made possible by expert practitioners' careful design of neural architectures. This manual handcrafted design requires a colossal number of computational resources, time, and memory to arrive at an optimal architecture. Automated Neural Architecture Search (NAS) is a promising area to explore to overcome these issues. However, optimizing a network for a job is a tedious task that requires lengthy search time, high processor needs, and a thorough examination of enormous possibilities. The need of the hour is to develop a strategy that saves time while maintaining an excellent level of accuracy. In this paper, we design, explore, and experiment with various differentiable NAS methods which are memory, time, and compute efficient. We also explore the role and efficacy of self-supervision to guide the search for optimal architectures. Self-Supervision offers numerous advantages such as facilitating the use of unlabelled data and making the “learning” non-task specific, thereby improving transfer to other tasks. To study the inclusion of self-supervision into the search process, we propose a simple loss function consisting of a convex combination of supervised cross-entropy loss and self-supervision loss. In addition, we carried out various analyses to characterize the performance of different approaches considered in this paper. The inspection of results obtained from various experiments on CIFAR-10 reveals that the proposed methodology balances time and accuracy while staying as near as possible to the state-of-the-art results.
