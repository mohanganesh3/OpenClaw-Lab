# EV012: Weapon Detection Using Deep Learning: A Systematic Review

URL: https://www.semanticscholar.org/paper/b4cbf5cf489519c1d71275e268907a8eb3a96a2d
Year: 2026
Source: semantic_scholar
Arxiv: n/a

## Abstract

Weapon detection is a time-critical vision task spanning public-space CCTV, body-worn cameras, and UAVs. This work presents a systematic review of deep learning approaches across visible-spectrum modalities, covering model families, datasets, evaluation protocols, and deployment constraints. We find that YOLO-family one-stage detectors dominate visible-spectrum applications by offering the best accuracy–latency trade-off for real-time inference, while X-ray research leverages large public benchmarks (e.g., SIXRay, OPIXray, PIDray) and increasingly adopts attention and anchor-free designs to handle clutter, overlap, and deformation. Performance is highly sensitive to scale-aware preprocessing, dataset realism/diversity, and, for video, temporal smoothing/aggregation. Persistent gaps include cross-site generalization, small/occluded weapon detection, rigorous calibration/uncertainty, and responsible deployment (privacy, bias, adversarial robustness). We distill best practices—compact YOLO variants with multi-scale heads and optional tiling; brightness/contrast normalization; synthetic+real data mixing; pose/orientation cues when architecture-compatible—and outline opportunities in foundation-model features, multimodal fusion (RGB–X-ray/IR), self-/active learning for long-tail classes, and standardized benchmarking with consistent mAP/F1/FPS reporting and small-object stratification..
