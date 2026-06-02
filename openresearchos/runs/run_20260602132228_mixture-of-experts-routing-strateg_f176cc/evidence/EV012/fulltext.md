[2602.21626] Multi-Layer Scheduling for MoE-Based LLM Reasoning

Multi-Layer Scheduling for MoE-Based LLM Reasoning

Yifan Sun1,
Gholamreza Haffari2,
Minxian Xu3,
Rajkumar Buyya4,
Adel N. Toosi1

Abstract

Large Language Models (LLMs) have achieved remarkable success across a wide range of tasks, but serving them efficiently at scale remains a critical challenge due to their substantial computational and latency demands. While most existing inference frameworks rely on simple scheduling strategies such as First-Come-First-Serve (FCFS) at the engine level and Round-Robin (RR) at the scheduler or coordinator level, they often fail to fully utilize system resources and may suffer from issues such as head-of-line blocking and load imbalance. Recent advances in Mixture-of-Experts (MoE) models have also introduced new challenges in scheduling arising from expert parallelism and routing complexity. This research proposes a multi-layer scheduling framework tailored for MoE-based LLM serving. It targets scheduling at three levels: request-level, engine-level, and expert-level. At the request level, we explore algorithms such as Shortest-Job-First (SJF) and priority-aware aging to improve throughput and reduce latency. At the engine level, we design load-aware dispatching strategies that account for the current prefix token load, KV cache utilization, and user stickiness to achieve better resource matching. At the expert level, we focus on alleviating expert hotspots and strategically placing inter-layer expert dependencies to balance load and improve routing efficiency. Extensive experimental results from more than 100 experiments conducted under diverse workload distributions show that our approach consistently outperforms the state-of-the-art inference framework vLLM, achieving up to 17.8% reduction in Time To First Token (TTFT) latency and 13.3% reduction in Time-Per-Output-Token (TPOT) latency.

I

Introduction

In recent years, Large Language Models (LLMs) have developed rapidly  [ 16 ] , helping millions of users worldwide with tasks such as natural language understanding, content generation, programming assistance, and multimodal reasoning. From personal productivity tools to enterprise AI services, LLMs are reshaping the way people interact with information and technology.

Early advances in LLMs were largely driven by scaling dense Transformer-based  [ 24 ]  models and continuously pushing the limits of model parameters. For example, GPT-1 had 117M parameters  [ 18 ] , followed by GPT-2 with 124M to 1.5B parameters  [ 19 ] , and GPT-3  [ 3 ]  with 175B parameters. These large-scale dense models achieved outstanding performance across a wide range of tasks by continuously increasing parameter counts. However, as the size of dense models grows, they place increasing demands on computational resources, inference latency, and deployment cost. Therefore, efficiently serving such models in real-world applications has become a critical problem that warrants immediate attention.

Given these growing concerns, increasing attention has been drawn to sparse model architectures,
particularly Mixture of Experts (MoE) models  [ 4 ] . The release of DeepSeek R1  [ 7 ]  demonstrated that sparse MoE models can perform as intelligently as traditional dense models while using significantly fewer computational resources. In MoE models, only a small subset of expert networks (i.e., specialized subnetworks) are activated per token during inference, enabling the model to scale to trillions of parameters while keeping the per-token computation manageable. Following DeepSeek R1, models such as Mixtral  [ 10 ]  and MoE variants of LLaMA-4  [ 15 ]  have also adopted MoE architectures, demonstrating the effectiveness of this approach in achieving strong performance and improved inference efficiency.

Although MoE architectures hold great promise, they also introduce new challenges in system design and serving. In particular, MoE models typically rely on expert parallelism, distributing experts across multiple GPUs or nodes. To further increase inference throughput and support large-batch processing, expert parallelism is often combined with data parallelism. While this layered parallelism enables efficient hardware utilization, current LLM serving frameworks still mainly adopt scheduling strategies originally designed for dense models. At the engine level, simple Round-Robin dispatching ignores load conditions such as prefix token volume and KV Cache usage, often leading to load imbalance and underutilization. At the request level, First-Come-First-Serve remains the default, which can cause head-of-line blocking and long-tail latency under heavy workloads. More critically, at the expert level, existing systems fail to account for inter-layer expert affinity, dependency patterns, and GPU-local hotspots, resulting in uneven workload distribution across experts and devices.

To address these challenges, we propose  Gimbal , a multi-layer scheduling framework designed to stabilize and optimize MoE-based LLM serving, with improvements in the following three areas:

•

Engine-Level Scheduling : adopting strategies such as Shortest-Job-First and priority-aware aging to reduce long-tail latency and improve throughput.

•

Request-Level Scheduling : employing load-aware and affinity-preserving dispatching that considers prefix tokens, KV Cache usage, and user stickiness, to reduce Time To First Token and Time Per Output Token.

•

Expert-Level Scheduling : mitigating expert hotspots and strategically placing dependent experts to improve efficiency and balance workload across GPUs.

Although our evaluation focuses on a representative MoE model, Gimbal is not tied to any model-specific internals. It relies only on general routing and scheduling signals exposed by MoE inference frameworks, and is therefore applicable to a broad range of MoE architectures, including Mixtral, Switch Transformer, and DeepSeek-style models.

Overall, the proposed multi-layer scheduling framework delivers substantial performance improvements for MoE-based LLM serving. By jointly optimizing request-level prioritization, engine-level load-aware dispatching, and expert-level dependency-aware placement, our approach effectively reduces both queuing delay and execution inefficiency under diverse workloads. Experimental results demonstrate that Gimbal achieves up to a 17.8% reduction in Time To First Token and a 13.3% reduction in Time-Per-Output-Token compared to the state-of-the-art baseline, while maintaining comparable throughput.

The remainder of the paper is organized as follows: Section

II

introduces the fundamental concepts of LLM and MoE architectures and discusses existing serving frameworks and their limitations. Section

III

presents the design of Gimbal and elaborates on its multi-layer scheduling mechanisms. Section

IV

describes the implementation of the Gimbal framework. Section

V

evaluates Gimbal’s performance under various workloads and compares it against baselines. In the next section, we review related work, and Section

VII

concludes the paper and discusses directions for future research.

II

Background and Motivation

This section provides an overview of the foundations of large language models and sparse Mixture-of-Experts architectures, followed by a discussion of modern LLM serving frameworks and their scheduling mechanisms. We then outline the limitations of existing systems and highlight the key motivations that drive the design of Gimbal.

II-A

LLM and MoE model Basics

Modern large language models are built upon the Transformer architecture  [ 24 ] , where each layer combines attention, feedforward networks, residual connections, and normalization. In self-attention, the hidden states are projected into queries (

Q  Q

), keys (

K  K

), and values (

V  V

), and computes attention via scaled dot-product. During inference, LLMs follow an autoregressive decoding process w