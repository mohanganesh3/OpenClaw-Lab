# EV018: Searching for Network Width With Bilaterally Coupled Network

URL: https://www.semanticscholar.org/paper/0097a0507074cfd13cf84204cd3cc7144f21c71e
Year: 2022
Source: semantic_scholar
Arxiv: 2203.13714

## Abstract

Searching for a more compact network width recently serves as an effective way of channel pruning for the deployment of convolutional neural networks (CNNs) under hardware constraints. To fulfil the searching, a one-shot supernet is usually leveraged to efficiently evaluate the performance w.r.t. different network widths. However, current methods mainly follow a unilaterally augmented (UA) principle for the evaluation of each width, which induces the training unfairness of channels in supernet. In this article, we introduce a new supernet called Bilaterally Coupled Network (BCNet) to address this issue. In BCNet, each channel is fairly trained and responsible for the same amount of network widths, thus each network width can be evaluated more accurately. Besides, we propose to reduce the redundant search space and present the BCNetV2 as the enhanced supernet to ensure rigorous training fairness over channels. Furthermore, we leverage a stochastic complementary strategy for training the BCNet, and propose a prior initial population sampling method to boost the performance of the evolutionary search. We also propose a new open-source width search benchmark on macro structures named Channel-Bench-Macro for the better comparisons of the width search algorithms with MobileNet- and ResNet-like architectures. Extensive experiments on the benchmark datasets demonstrate that our method can achieve state-of-the-art performance.
