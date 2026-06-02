# EV007: EC2MoE: Adaptive End-Cloud Pipeline Collaboration Enabling Scalable Mixture-of-Experts Inference

URL: https://www.semanticscholar.org/paper/1d13b1f6a6c9314c6855a06b30664f8593aa4073
Year: 2025
Source: semantic_scholar
Arxiv: 2508.06024

## Abstract

The Mixture-of-Experts (MoE) paradigm has emerged as a promising solution to scale up model capacity while maintaining inference efficiency. However, deploying MoE models across heterogeneous end-cloud environments poses new challenges in expert scheduling, communication overhead, and resource heterogeneity. In this paper, we propose EC2MoE, an adaptive framework for scalable MoE inference via end-cloud pipeline collaboration. First, we design a hardware-aware lightweight group gate network that enhances expert selection and computational efficiency. By incorporating a hardware-aware local expert selection mechanism, the system adaptively filters candidate experts based on real-time device profiles. A lightweight group gate module then integrates local and global gating outputs to achieve high-quality expert routing with minimal overhead. Second, we develop a pipeline optimization mechanism based on endcloud collaboration to accelerate MoE inference. This includes an encoder-decoder structure based on low-rank compression, which reduces transmission and computation costs. And a route-aware heuristic pipeline scheduling algorithm that dynamically allocates inference stages across devices according to workload and network topology. Extensive experiments show that EC2MoE can increase throughput by 2.2x to 5.1x and reduce end-to-end latency by 53% to 67% while maintaining high accuracy compared to state-of-the-art methods. It also maintains good scalability under dynamic load and network environments.
