# EV010: Residual Knowledge Retention For Edge Devices

URL: https://www.semanticscholar.org/paper/004e2b1d10d61088c581552709f702e82fb2901d
Year: 2021
Source: semantic_scholar
Arxiv: n/a

## Abstract

This paper proposes an approach for continual learning, Knowledge Retention (KR), that learns new information without accessing data from previous tasks. A KR unit is based on the embedding layer that identifies the important kernel in the convolution layer, which preserves key parameters and allows the weights to be reused across tasks. To construct higher-order generalization, we design a Residual Knowledge Retention (RKR) architecture that facilitates the network to stack deeper layers. Additionally, we rethink the benefits of different residual blocks respectively after employing depthwise convolutions. A surprising observation is that the basic block taking advantage of depthwise convolutions achieves higher representational power and builds a more lightweight model than the bottleneck block counterpart. On the Alternating CIFAR10/100 benchmark, we empirically show that the KR unit can be integrated into diverse networks and effectively prevents catastrophic forgetting. Finally, we demonstrate that RKR significantly outperforms the existing state-of-the-art continual learning methods with at least 6 times lower model complexity in two different scenarios for continual learning, which supports that the proposed approach is more competitive for resource-limited edge devices.
