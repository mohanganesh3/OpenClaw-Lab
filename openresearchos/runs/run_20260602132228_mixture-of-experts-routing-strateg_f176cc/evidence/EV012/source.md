# EV012: Multi-Layer Scheduling for MoE-Based LLM Reasoning

URL: https://www.semanticscholar.org/paper/30c830ce134ff4220932a0b4e6b59500c44ad615
Year: 2026
Source: semantic_scholar
Arxiv: 2602.21626

## Abstract

Large Language Models (LLMs) have achieved remarkable success across a wide range of tasks, but serving them efficiently at scale remains a critical challenge due to their substantial computational and latency demands. While most existing inference frameworks rely on simple scheduling strategies such as First-Come-First-Serve (FCFS) at the engine level and Round-Robin (RR) at the scheduler or coordinator level, they often fail to fully utilize system resources and may suffer from issues such as head-of-line blocking and load imbalance. Recent advances in Mixture-of-Experts (MoE) models have also introduced new challenges in scheduling arising from expert parallelism and routing complexity. This research proposes a multi-layer scheduling framework tailored for MoE-based LLM serving. It targets scheduling at three levels: request-level, enginelevel, and expert-level. At the request level, we explore algorithms such as Shortest-Job-First (SJF) and priority-aware aging to improve throughput and reduce latency. At the engine level, we design load-aware dispatching strategies that account for the current prefix token load, KV cache utilization, and user stickiness to achieve better resource matching. At the expert level, we focus on alleviating expert hotspots and strategically placing inter-layer expert dependencies to balance load and improve routing efficiency. Extensive experimental results from more than 100 experiments conducted under diverse workload distributions show that our approach consistently outperforms the state-of-theart inference framework vLLM, achieving up to 17.8% reduction in Time To First Token (TTFT) latency and 13.3% reduction in Time-Per-Output-Token (TPOT) latency.
