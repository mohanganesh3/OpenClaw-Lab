# EV012: AL-X0: Cost-Aware Active Learning for Cloud-Scale NLP via Zero-Shot Proxy Valuation

URL: https://www.semanticscholar.org/paper/3c027b46f1fab9ee8eabceade136b660e84bd405
Year: 2026
Source: semantic_scholar
Arxiv: n/a

## Abstract

Active Learning (AL) is widely adopted to reduce annotation costs in large-scale machine learning, yet standard methods remain vulnerable to two fundamental failure modes: (i) cold-start instability, where untrained models produce uncalibrated uncertainty estimates that behave indistinguishably from random sampling in early rounds, and (ii) cost collapse, wherein naive cost-aware heuristics that normalize utility by annotation time degrade model quality by systematically selecting trivial, low-information samples. This work proposes AL-X0, a principled framework that decouples information valuation from model confidence through three interacting mechanisms: (1) Zero-Shot Proxy Valuation (ZSPV) using embedding geometry, (2) a Consensus Engine for adaptive signal fusion, and (3) a Dual-Head Cost Brain for cost estimation. However, empirical validation reveals that model calibration in early rounds remains a significant bottleneck, limiting the effectiveness of uncertaintybased selection. We therefore introduce CAL-Log, an improved variant that explicitly models annotation uncertainty calibration dynamics. CAL-Log achieves 30–40% better cost-efficiency than AL-X0 while maintaining core design principles, offering a practical solution for large-scale annotation in cloud-scale NLP environments.
