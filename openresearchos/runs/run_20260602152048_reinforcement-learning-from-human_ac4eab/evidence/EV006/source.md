# EV006: Accelerating Long-Tail Generation in Synchronous RLHF Training via Adaptive Tensor Parallelism

URL: https://www.semanticscholar.org/paper/0120336949fb9165d0389598c343df4a498dfdbe
Year: 2026
Source: semantic_scholar
Arxiv: 2605.23945

## Abstract

Reinforcement Learning from Human Feedback (RLHF) has become a key post-training paradigm for improving model quality. However, the synchronous three-stage RLHF pipeline is often bottlenecked by the generation stage, where response-length skew causes the effective batch size to shrink rapidly during decoding, leaving GPUs underutilized while a few long responses remain unfinished. Mainstream frameworks employ a static tensor parallelism (TP) configuration that cannot adapt to changing batch characteristics, leaving substantial performance headroom unexplored. We propose PAT, an adaptive TP method that dynamically reconfigures TP during the generation stage of each RLHF iteration. PAT introduces two key techniques. First, a predictor-guided online reconfiguration method decides both the reconfiguration point and the target TP configuration based on offline profiling, triggering reconfiguration only when the predicted latency benefit outweighs the reconfiguration overhead. Second, a lightweight online reconfiguration mechanism updates only the states and layouts affected by TP changes: it adapts unfinished decoding states through a cost-model-based choice between KV-cache migration and recomputation, performs in-place weight resharding, and reuses cached communication groups. We implement PAT on top of SGLang and integrate it with the VeRL framework. Evaluations on LLaMA3.1-8B and Qwen3-14B using DeepScaleR show that PAT reduces generation latency by up to 34.6% and end-to-end RLHF training iteration latency by up to 27.2% compared to the original VeRL setup.
