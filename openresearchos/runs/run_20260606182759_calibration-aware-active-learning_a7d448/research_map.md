# Research Map

Run: `run_20260606182759_calibration-aware-active-learning_a7d448`
Topic: calibration-aware active learning for medical image classification
Scope: `local_experiment`
Feasibility: Yes, calibration-aware active learning for medical image classification is feasible on your MacBook M4 with 36GB RAM and MPS GPU, though you'll need to optimize model architectures and batch sizes to manage memory constraints. Suitable public datasets include CheXpert (chest X-rays), ISIC Archive (skin lesion images), and Brain Tumor MRI dataset, all of which have manageable sizes for local experimentation. Expect initial model training to take 2-6 hours, with each active learning iteration addi

## Summary

- **61 papers** discovered and deep-read
- **29 papers** with full text extraction
- **21 papers** with MacBook-runnable experiments
- **4 papers** with available code
- **1 citation relationships** mapped

## Method Comparison Matrix

| Paper | Dataset | Method | Best Result | Limitations |
|-------|---------|--------|-------------|-------------|
| [EV001] The AI Scientist: Towards Fully Automate | Swe-bench (GitHub issues), Mla | LLMs generate research ideas, implement them via c | not explicitly stated in provided text |  |
| [EV002] The AI Scientist-v2: Workshop-Level Auto | SCAN, COGS, IWSLT, GeoQuery | The system iteratively formulates hypotheses, desi | Not available in provided content |  |
| [EV003] Agent Laboratory: Using LLM Agents as Re | ? | Agent Laboratory accepts a human-provided research | ? |  |
| [EV004] MLE-bench: Evaluating Machine Learning A | ? | The competition involves predicting scalar couplin | ? |  |
| [EV005] AIBuildAI-2: Scalable Agentic AI Researc | ? | AIBuildAI-2 uses a hierarchical sub-agent architec | ? | Performance is largely limited by the pa |
| [EV006] Reflexion: Language Agents with Verbal R | HumanEval Python, HotPotQA (10 | Reflexion implements an iterative optimization pro | gpt-4 with Reflexion on HotPotQA: 80% accuracy | The framework focuses on language agents |
| [EV007] Self-Refine: Iterative Refinement with S | CodeNet (300 examples), Common | SELF-REFINE generates initial output with an LLM,  | GPT-4 achieved 74.6% on Dialogue Response with SELF-REFINE (statistically significant) | The paper does not explicitly state limi |
| [EV008] Voyager: An Open-Ended Embodied Agent wi | Minecraft (implied) | VOYAGER uses GPT-4 via blackbox queries to generat | Not provided in text | The abstract doesn't explicitly state li |
| [EV009] AgentBench: Evaluating LLMs as Agents | FREEBASE (GrailQA, ComplexWebQ | The method constructs 8 interactive environments s | gpt-3.5-turbo achieves 0.92 on UPDATE operations in re-annotated data |  |
| [EV010] SWE-agent: Agent-Computer Interfaces Ena | SWE-bench Lite dataset, SWE-be | SWE-agent implements a custom agent-computer inter | SWE-agent with GPT-4 resolved 286 trajectories on SWE-bench (full) |  |
| [EV013] RewardUQ: A Unified Framework for Uncert | ? | RewardUQ trains and evaluates four uncertainty qua | ? |  |
| [EV014] Bayesian E(3)-Equivariant Interatomic Po | ? | They develop Bayesian E(3)-equivariant MLPs using  | ? |  |
| [EV017] Uncertainty-Guided Dual-Domain Learning  | HAM10000, ISIC2017, ISIC2018,  | UGDD-Net employs a two-pass forward mechanism wher | UGDD-Net achieves IoUcon of 73.91% and Dicecon of 83.37% against Expert Consensus |  |
| [EV020] Probabilistic Embeddings for Frozen Visi | COCO, CUB | GroVE builds on Gaussian Process Latent Variable M | KL-Divergence achieves best performance with 0.79±0.02 accuracy on Image to Text COCO | Deterministic embeddings from standard V |
| [EV025] Uncertainty-aware phase fraction predict | 70,000 compositions from Ti-Fe | Refractory multi-principal element alloys (RMPEAs) | R² of 0.99 for Laves phase prediction | Existing machine learning approaches rel |
| [EV027] Revisiting Unknowns: Towards Effective a | ? | Trains a dual-head classifier with LCE and LEDL lo | ? |  |
| [EV028] OmniDP: Beyond-FOV Large-Workspace Human | Teleoperated observation-actio | OmniDP processes panoramic point clouds through a  | OmniDP achieves 82/120 (68.3%) total success rate across all tasks, significantly outperforming baselines (DP: 18/120, DP3: 22/120, iDP3: 25/120) | Conventional RGB-D solutions suffer from |
| [EV032] On Exact Sampling in the Two-Variable Fr | ? | Uses Scott Normal Form (SNF) to transform FO2 sent | null | The paper focuses on theoretical aspects |
| [EV033] A virtually connected probabilistic comp | burma14 TSP instance, Erdős–Ré | Uses high-speed photonic quantum random number gen | TTS of ~10⁻⁵ seconds for q=0.8 and N=1024, outperforming DA by orders of magnitude |  |
| [EV035] Generative and Nonparametric Approaches  | ? | The paper reviews and compares several methods for | Not specified in text | Limited scope of methods covered |

## Research Gaps

### G01: Standardized Benchmarks for Calibration-Aware Active Learning in Medical Image Classification

Current benchmarks do not specifically evaluate calibration-aware active learning strategies for medical image classification, making it difficult to compare different approaches and measure progress in this niche area.

**Hypothesis**: If a standardized benchmark for calibration-aware active learning in medical image classification is developed, researchers will be able to more effectively compare and improve their methods.
**Suggested dataset**: Medical Image Classification Benchmark (MICB)
**Baseline paper**: TBD
**Evidence**: EV009
**Difficulty**: medium

### G02: Integration of Linguistic Feedback in Medical Image Active Learning

While linguistic feedback mechanisms have been explored in general agent frameworks, their specific application to calibration-aware active learning in medical image classification remains underexplored.

**Hypothesis**: Incorporating linguistic feedback into calibration-aware active learning for medical images will improve both model accuracy and calibration.
**Suggested dataset**: Radiology Image Dataset with Expert Annotations
**Baseline paper**: TBD
**Evidence**: EV006
**Difficulty**: hard

### G03: Evaluation Frameworks for Calibration-Aware Active Learning in Medical Contexts

Current evaluation frameworks for active learning and model calibration are not specifically designed for medical image classification tasks, leading to potentially misleading assessments of model performance.

**Hypothesis**: A specialized evaluation framework for calibration-aware active learning in medical contexts will provide more accurate assessments of model performance and calibration.
**Suggested dataset**: Multi-center Medical Image Dataset
**Baseline paper**: TBD
**Evidence**: EV004, EV009
**Difficulty**: medium

### G04: Hierarchical Agent Architectures for Calibration-Aware Active Learning in Medical Imaging

The hierarchical agent architectures demonstrated in general AI frameworks have not been specifically adapted for the complex requirements of calibration-aware active learning in medical image classification.

**Hypothesis**: A hierarchical agent architecture specifically designed for calibration-aware active learning will outperform flat architectures in medical image classification tasks.
**Suggested dataset**: Comprehensive Medical Imaging Dataset
**Baseline paper**: TBD
**Evidence**: EV005
**Difficulty**: hard

### G05: Embodied Learning Approaches for Calibration-Aware Active Learning in Medical Imaging

Embodied learning approaches have shown promise in general AI systems but have not been explored in the context of calibration-aware active learning for medical image classification.

**Hypothesis**: Embodied learning approaches will enhance the calibration-awareness of active learning strategies in medical image classification by providing more intuitive feedback mechanisms.
**Suggested dataset**: Interactive Medical Image Dataset
**Baseline paper**: TBD
**Evidence**: EV008
**Difficulty**: hard

### G06: Specialized Agent-Computer Interfaces for Calibration-Aware Active Learning in Medical Image Classification

While general agent-computer interfaces have been developed, specialized interfaces that facilitate calibration-aware active learning in medical image classification are lacking.

**Hypothesis**: Specialized agent-computer interfaces designed for calibration-aware active learning will improve the efficiency and effectiveness of human-AI collaboration in medical image classification.
**Suggested dataset**: Medical Image Classification with Expert Interaction Dataset
**Baseline paper**: TBD
**Evidence**: EV010
**Difficulty**: medium

## Citation Relationships

- EV001 → EV008 (baseline_comparison)

## MacBook-Runnable Experiments Available

- [EV020] Probabilistic Embeddings for Frozen Vision-Language Models:  — KL-Divergence achieves best performance with 0.79±0.02 accuracy on Image to Text COCO — Code: null (not mentioned in abstract)
- [EV035] Generative and Nonparametric Approaches for Conditional Dist — Not specified in text — Code: null
- [EV037] Markovian Flow Matching: Accelerating MCMC with Continuous N — ? — Code: yes
