# EV001: PhaseNAS: Language-Model Driven Architecture Search with Dynamic Phase Adaptation

URL: https://www.semanticscholar.org/paper/00415faf1b11526845cde1e293f53b0ac8e81910
Year: 2025
Source: semantic_scholar
Arxiv: 2507.20592

## Abstract

Neural Architecture Search (NAS) is challenged by the trade-off between search space exploration and efficiency, especially for complex tasks. While recent LLM-based NAS methods have shown promise, they often suffer from static search strategies and ambiguous architecture representations. We propose PhaseNAS, an LLM-based NAS framework with dynamic phase transitions guided by real-time score thresholds and a structured architecture template language for consistent code generation. On the NAS-Bench-Macro benchmark, PhaseNAS consistently discovers architectures with higher accuracy and better rank. For image classification (CIFAR-10/100), PhaseNAS reduces search time by up to 86% while maintaining or improving accuracy. In object detection, it automatically produces YOLOv8 variants with higher mAP and lower resource cost. These results demonstrate that PhaseNAS enables efficient, adaptive, and generalizable NAS across diverse vision tasks.
