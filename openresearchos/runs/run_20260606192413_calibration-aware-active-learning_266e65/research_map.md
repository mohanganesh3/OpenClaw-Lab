# Research Map

Run: `run_20260606192413_calibration-aware-active-learning_266e65`
Topic: calibration-aware active learning for medical image classification
Scope: `local_experiment`
Feasibility: Yes, this research topic is feasible on a MacBook M4 with 36GB RAM and Apple MPS GPU. Suitable public datasets include CheXpert (chest X-rays), ISIC Archive (skin lesions), and DeepLesion (CT scans), which are large enough for active learning but manageable with your hardware. Experiment iterations would likely take 4-8 hours per run for model training and evaluation, depending on the active learning strategy and calibration methods implemented. The offline nature of your setup actually benefits

## Summary

- **60 papers** discovered and deep-read
- **30 papers** with full text extraction
- **20 papers** with MacBook-runnable experiments
- **7 papers** with available code
- **0 citation relationships** mapped

## Method Comparison Matrix

| Paper | Dataset | Method | Best Result | Limitations |
|-------|---------|--------|-------------|-------------|
| [EV001] The AI Scientist: Towards Fully Automate | ? | The AI Scientist uses frontier LLMs to generate no | ? |  |
| [EV002] The AI Scientist-v2: Workshop-Level Auto | SCAN, COGS, IWSLT, GeoQuery, s | The system employs a progressive agentic tree-sear | No significant improvement in generalization performance with compositional regularization on synthetic arithmetic expression datasets |  |
| [EV003] Agent Laboratory: Using LLM Agents as Re | ? | The method involves an autonomous LLM-based framew | ? |  |
| [EV004] MLE-bench: Evaluating Machine Learning A | ? | MLE-bench evaluates ML agents on predicting scalar | ? | Limited evaluation on real-world ML engi |
| [EV005] AIBuildAI-2: Scalable Agentic AI Researc | ? | AIBuildAI-2 uses a hierarchical sub-agent design w | ? | Performance of agents is limited by para |
| [EV006] Reflexion: Language Agents with Verbal R | HumanEval Python, HotPotQA (10 | Reflexion formalizes an iterative optimization pro | 0.80 on HotPotQA with Reflexion and gpt-4 | The paper does not explicitly address ca |
| [EV007] Self-Refine: Iterative Refinement with S | CommonGen-Hard, Acronym Genera | SELF-REFINE generates an initial output using an L | GPT-4: 93.1% in Math Reasoning (vs 92.9% base) | The paper doesn't explicitly state limit |
| [EV008] Voyager: An Open-Ended Embodied Agent wi | ? | VOYAGER interacts with GPT-4 via blackbox queries  | ? |  |
| [EV010] SWE-agent: Agent-Computer Interfaces Ena | SWE-bench Lite dataset, SWE-be | SWE-agent enables LMs to interact with computers t | Resolution performance across different turns with varying success rates |  |
| [EV012] RewardUQ: A Unified Framework for Uncert | ? | Trains four UQ architectures using embeddings from | ? |  |
| [EV013] Bayesian E(3)-Equivariant Interatomic Po | ? | The approach develops Bayesian E(3)-equivariant ML | ? | The method is specifically designed for  |
| [EV016] Uncertainty-Guided Dual-Domain Learning  | ISIC2017, ISIC2018, PH2, HAM10 | UGDD-Net employs a dual-domain architecture with s | IoU of 79.92% on Hard Samples (UGML) | The abstract doesn't explicitly state li |
| [EV019] Probabilistic Embeddings for Frozen Visi | MS-COCO, CUB-200-2011 | GroVE builds a Gaussian Process Latent Variable Mo | 0.79±0.02 KL-Divergence score for Image-to-Text alignment on COCO |  |
| [EV024] Uncertainty-aware phase fraction predict | 70,000 compositions from Ti, F | Refractory multi-principal element alloys (RMPEAs) | 0.99 R² for Laves phase prediction |  |
| [EV026] Revisiting Unknowns: Towards Effective a | CIFAR-10, CIFAR-100, Tiny-Imag | Trains a dual-head classifier (LCE for known class | OSAL achieves 94.2% accuracy on CIFAR-10 |  |
| [EV027] OmniDP: Beyond-FOV Large-Workspace Human | Teleoperated trajectories usin | The method processes panoramic LiDAR point clouds  | OmniDP achieved 82/120 total success rate across all tasks (Table I) | Conventional RGB-D solutions suffer from |
| [EV031] Bayesian Uncertainty Quantification with | Chassis dynamometer tests, Hig | Trains multiple LSTM models with gate-wise Gaussia | t-NLL Anchor achieved best calibration with standardized variance near unity (≈1.13) for Sedan and IONIQ 5 datasets | The model is specifically designed for E |
| [EV036] Quantum Probabilistic Label Refining: En | MNIST, Fashion-MNIST | Encode inputs into multi-qubit quantum states usin | M3 achieves 97.87% accuracy on MNIST at Std=0.1 | The paper doesn't explicitly state limit |
| [EV039] Network Inversion for Uncertainty-Aware  | MNIST, FashionMNIST, SVHN, CIF | The method extends an n-class classifier to an (n+ | 99.4% accuracy for MNIST trained model tested on CIFAR-10 as OOD |  |
| [EV040] Generating time-consistent dynamics with | ? | The method trains a time-consistency discriminator | ? |  |

## Research Gaps

### G01: Calibration-Aware Active Learning Framework for Medical Images

No existing framework specifically addresses calibration-aware active learning for medical image classification. The AI Scientist (EV001) and Agent Laboratory (EV003) frameworks lack this capability.

**Hypothesis**: If we implement calibration-aware active learning, ECE will reduce by at least 15% compared to standard active learning
**Suggested dataset**: breast_cancer
**Baseline paper**: TBD
**Evidence**: EV001, EV003
**Difficulty**: medium

### G02: Compositional Regularization in Calibration-Aware Active Learning

Compositional regularization showed no significant improvement vs baseline (EV002). We need to explore if this changes when integrated with calibration-aware active learning.

**Hypothesis**: If we apply compositional regularization in a calibration-aware active learning framework, accuracy will improve by at least 5% compared to the baseline
**Suggested dataset**: digits
**Baseline paper**: TBD
**Evidence**: EV002
**Difficulty**: easy

### G03: Self-Refinement for Medical Image Classification

Self-refinement has not been adapted for medical image classification (EV007). We need to explore how self-refinement can improve calibration in medical imaging.

**Hypothesis**: If we adapt self-refinement for medical image classification, sample efficiency will improve by at least 20% (fewer samples needed for same accuracy)
**Suggested dataset**: breast_cancer
**Baseline paper**: TBD
**Evidence**: EV007
**Difficulty**: medium

### G04: Reflection Mechanism for Visual Classification

The reflection mechanism may not transfer well to image classification (EV006). We need to explore how reflection can be adapted for visual inputs.

**Hypothesis**: If we adapt the reflection mechanism for image classification, ECE will reduce by at least 10% compared to standard classification
**Suggested dataset**: digits
**Baseline paper**: TBD
**Evidence**: EV006
**Difficulty**: medium

### G05: Automated Model Building with Calibration Integration

Automated model building for medical image classification with calibration integration is not addressed (EV005). We need to explore how to integrate calibration into automated model building.

**Hypothesis**: If we integrate calibration into automated model building for medical image classification, accuracy will improve by at least 7% compared to non-calibrated models
**Suggested dataset**: breast_cancer
**Baseline paper**: TBD
**Evidence**: EV005
**Difficulty**: hard

### G06: Specialized Benchmarks for Medical ML Engineering

No specialized benchmarks exist for medical ML engineering with calibration focus (EV004). We need to create benchmarks that specifically test calibration-aware active learning in medical imaging.

**Hypothesis**: If we create a specialized benchmark for calibration-aware active learning in medical imaging, agent performance on this benchmark will be at least 25% better than on general ML benchmarks
**Suggested dataset**: breast_cancer
**Baseline paper**: TBD
**Evidence**: EV004
**Difficulty**: hard

## Citation Relationships

- No citation relationships detected

## MacBook-Runnable Experiments Available

- [EV006] Reflexion: Language Agents with Verbal Reinforcement Learnin — 0.80 on HotPotQA with Reflexion and gpt-4 — Code: null
- [EV043] Low Rank for Rank: Uncertainty-Aware Task-Specific LLM Ranki — ? — Code: yes
- [EV058] Generative Flow Networks for Model Adaptation in Digital Twi — GFlowNet recovers dominant basins in multimodal landscape (sf=0.3) with >95% alignment to exact distribution in ranked profiles — Code: null
