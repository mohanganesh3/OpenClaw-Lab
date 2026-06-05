# EV015: MCSim: A Discrete-Event Simulation Framework for Studying Multi-Cloud Networks

URL: https://www.semanticscholar.org/paper/001dc6e7404b355ceadc02c6b3342f83207b9694
Year: 2025
Source: semantic_scholar
Arxiv: n/a

## Abstract

As enterprises adopt multi-cloud strategies understanding network behavior across heterogeneous cloud infrastructures becomes critical for performance and cost optimization. However, existing network simulators lack support for cloudnative hierarchies (providers, regions, zones) and cannot incorporate empirical inter-cloud measurements or egress cost modeling. We present MCSim, a discrete-event simulator designed specifically for empirical trace-driven multi-cloud network analysis. MCSim models hierarchical cloud topologies with time-varying metrics derived from real measurements, supports cost-aware routing policies, and also supports hybrid emulation by applying simulated network conditions to real hosts using Linux traffic control (tc), bridging simulation with real-world validation. We validate MCSim using 50+ million measurements across $\mathbf{1 0 5}$ VMs in AWS, GCP, and Azure regions. Statistical analysis shows MCSim achieves ${\lt }1 \%$ error in average latency reproduction, with $\mathbf{8 7 \%}$ of inter-cloud pairs showing $\lt \mathbf{5 m s}$ RMSE. Case studies demonstrate MCSim’s ability to analyze cost-performance tradeoffs (27% latency reduction with 57% cost increase), simulate failure scenarios (i.e., impact from regional outages), and enable hybrid testing with ${\lt }2 \%$ deviation from real deployments. MCSim scales to 1000+ nodes with spatial parallelization, enabling reproducible multi-cloud experimentation at unprecedented scale and fidelity.
