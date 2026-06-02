# EV015: Multi-Population Evolutionary Neural Architecture Search via Multiple Zero-Cost Proxies

URL: https://www.semanticscholar.org/paper/01f84f1ba2aceac3baf245ee8edc06a9d485e1ec
Year: 2025
Source: semantic_scholar
Arxiv: n/a

## Abstract

Zero-cost proxies have attracted growing attention in neural architecture search (NAS) for their efficiency in evaluating neural architectures at low computational cost. However, empirical studies have revealed that zero-cost proxies exhibit inherent biases, leading to performance degradation in complex scenarios. To alleviate this limitation, this paper proposes the Multi-Proxy Multi-Population Evolutionary Algorithm (MMEA) that integrates the strengths of multiple complementary proxies to improve search effectiveness. Specifically, each sub-population is associated with a distinct zero-cost proxy that serves as its optimization objective for approximating the performance of candidate architectures. Subsequently, multiple high-performance architectures are evolved within each sub-population under the guidance of their respective proxies. Furthermore, the Low-Fidelity Adaptive Proxy Selection Mechanism (LAPSM) is proposed to identify the most promising architecture by leveraging limited training resources to effectively select the most suitable proxy. Experimental results on the CIFAR-10, CIFAR-100, and ImageNet16-120 datasets. The result demonstrate that the proposed method outperforms state-of-the-art NAS algorithms.
