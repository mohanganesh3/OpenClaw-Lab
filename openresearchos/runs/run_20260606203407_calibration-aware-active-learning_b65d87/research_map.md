# Research Map

Run: `run_20260606203407_calibration-aware-active-learning_b65d87`
Topic: calibration-aware active learning for medical image classification
Scope: `local_experiment`
Feasibility: The research topic is feasible on your MacBook M4 with 36GB RAM and Apple MPS GPU, especially for moderate-sized medical image datasets. Suitable public datasets include ChestX-ray14 (112,120 images), CheXpert (224,316 images), and ISIC Archive (25,331 images), which can be downloaded beforehand. Experiment time would likely range from 6-24 hours for initial experiments, depending on dataset size, model complexity, and number of active learning iterations. The MPS GPU should provide reasonable p

## Summary

- **59 papers** discovered and deep-read
- **30 papers** with full text extraction
- **21 papers** with MacBook-runnable experiments
- **9 papers** with available code
- **0 citation relationships** mapped

## Method Comparison Matrix

| Paper | Dataset | Method | Best Result | Limitations |
|-------|---------|--------|-------------|-------------|
| [EV001] The AI Scientist: Towards Fully Automate | ? | The AI Scientist framework generates novel researc | ? |  |
| [EV002] The AI Scientist-v2: Workshop-Level Auto | SCAN, COGS, IWSLT, GeoQuery, s | The system uses a progressive agentic tree-search  | No significant improvement in generalization performance with compositional regularization | The abstract doesn't explicitly state li |
| [EV003] Agent Laboratory: Using LLM Agents as Re | ? | Agent Laboratory uses a pipeline of specialized LL | ? | The framework's quality depends on the q |
| [EV004] MLE-bench: Evaluating Machine Learning A | train.csv, test.csv, structure | The method involves training ML agents to predict  | o1-preview achieved 34.3 ± 2.4% in Low complexity category |  |
| [EV005] AIBuildAI-2: Scalable Agentic AI Researc | ? | AIBuildAI-2 initializes by collecting and organizi | No specific numeric results provided in the text |  |
| [EV006] Reflexion: Language Agents with Verbal R | HumanEval Python, HotPotQA (10 | Reflexion is formalized as an iterative optimizati | Reflexion with CoT (GT) + gpt-4 achieved 0.80 Pass@1 accuracy on HotPotQA |  |
| [EV007] Self-Refine: Iterative Refinement with S | CommonGen-Hard (extension of C | SELF-REFINE generates an initial output using an L | GPT-4 + SELF-REFINE achieves 93.1% on Math Reasoning (vs 92.9% base), with 36.2% on Sentiment Reversal (vs 3.8% base) | The paper doesn't explicitly mention lim |
| [EV008] Voyager: An Open-Ended Embodied Agent wi | ? | VOYAGER employs GPT-4 via blackbox queries to gene | ? |  |
| [EV009] AgentBench: Evaluating LLMs as Agents | GrailQA, ComplexWebQuestions,  | The paper introduces AGENTBENCH, a benchmark consi | gpt-3.5-turbo new achieves 0.92 on UPDATE operations | The paper identifies typical reasons for |
| [EV010] SWE-agent: Agent-Computer Interfaces Ena | SWE-bench Lite dataset, SWE-be | SWE-agent uses a language model agent interacting  | Table 7 shows % Resolved performance for task instances from different years |  |
| [EV013] RewardUQ: A Unified Framework for Uncert | ? | RewardUQ implements four uncertainty-aware reward  | ? | Uncertainty-aware reward models have bee |
| [EV014] Bayesian E(3)-Equivariant Interatomic Po | ? | Implements Bayesian neural networks with E(3)-equi | ? |  |
| [EV017] Uncertainty-Guided Dual-Domain Learning  | HAM10000 | UGDD-Net employs a dual-domain architecture with s | UGML achieved IoU of 92.61% and Dice of 96.01% on the main dataset, and IoU of 79.92% on Hard Samples |  |
| [EV020] Probabilistic Embeddings for Frozen Visi | COCO, CUB | GroVE builds on Gaussian Process Latent Variable M | KL-Divergence achieves 0.79±0.02 for Image to Text on COCO | Existing approaches require large datase |
| [EV025] Uncertainty-aware phase fraction predict | 70,000 compositions from Ti, F | Refractory multi-principal element alloys (RMPEAs) | MDN models achieved R² values of 0.97 (FCC), 0.98 (BCC), 0.99 (Laves), 0.89 (Sigma), 0.96 (Heusler), and 0.98 (Liquid) | The abstract doesn't explicitly state li |
| [EV027] Revisiting Unknowns: Towards Effective a | ? | Trains classifier with LCE for primary head and LE | ? |  |
| [EV028] OmniDP: Beyond-FOV Large-Workspace Human | Teleoperated LiDAR point cloud | OmniDP processes panoramic LiDAR point clouds thro | OmniDP achieved 82/120 total success rate (16/20 Pick & Place, 12/20 Pour OV, 15/20 Pick & Place real, 12/20 Hand Over OV, 11/20 Wipe OV) | Existing approaches to expanding percept |
| [EV033] Advance warning of $\gamma$-ray blazar f | Fermi-LAT light curves for gam | The paper uses Bayesian Blocks to identify flare i | Polynomial Logistic Regression (PLR) achieved WATCH ROC AUC = 0.891 and WATCH AP = 0.396 on held-out data | The use of strictly causal backtesting l |
| [EV037] Beyond the Loss Curve: Scaling Laws, Act | AFHQ, ImageNet-64, ImageNet-12 | The authors use class-conditional normalizing flow | Epistemic error follows power-law decay (KL ∝ N^{-α}) across architectures and scales to ImageNet-1000 | Uses AFHQ and ImageNet datasets rather t |
| [EV038] Adaptive Outer-Loop Control of Quadrotor | quadrotor dynamics with payloa | The approach trains an optimal outer-loop policy,  | Adaptive Controller with RDP achieves RMSE of 0.024 m (0% payload) |  |

## Research Gaps

### G01: Verbal Reflection for Uncertainty Estimation in Active Learning

Combining verbal reflection mechanisms with uncertainty estimation to improve calibration in active learning for medical images (EV003, EV006).

**Hypothesis**: If verbal reflection is integrated with uncertainty estimation, then ECE will decrease by at least 10% compared to standard active learning.
**Suggested dataset**: breast_cancer
**Baseline paper**: TBD
**Evidence**: EV003, EV006
**Difficulty**: medium

### G02: Self-Refinement for Image Classification Calibration

Adapting self-refinement techniques from text to image classification for better calibration (EV007, EV008).

**Hypothesis**: If self-refinement is applied to image classification, then accuracy will increase by at least 5% compared to baseline.
**Suggested dataset**: digits
**Baseline paper**: TBD
**Evidence**: EV007, EV008
**Difficulty**: easy

### G03: LLM Agents for Domain-Specific Medical Image Classification

Evaluating LLM agents for domain-specific tasks in medical image classification with calibration awareness (EV003, EV009).

**Hypothesis**: If LLM agents are used for active learning, then sample efficiency will improve by at least 20% (fewer samples needed for same accuracy).
**Suggested dataset**: breast_cancer
**Baseline paper**: TBD
**Evidence**: EV003, EV009
**Difficulty**: medium

### G04: Bayesian Active Learning with Reflection Mechanisms

Integrating Bayesian active learning with reflection mechanisms for better calibration (EV006, EV011).

**Hypothesis**: If Bayesian active learning is combined with reflection, then ECE will decrease by at least 15% compared to Bayesian active learning alone.
**Suggested dataset**: digits
**Baseline paper**: TBD
**Evidence**: EV006, EV011
**Difficulty**: medium

### G05: Cost-Aware Active Learning for Medical Image Calibration

Developing cost-aware active learning frameworks specifically for medical image classification with calibration (EV008, EV012).

**Hypothesis**: If cost-aware active learning is applied, then cost per accuracy point will decrease by at least 10% compared to standard active learning.
**Suggested dataset**: breast_cancer
**Baseline paper**: TBD
**Evidence**: EV008, EV012
**Difficulty**: hard

### G06: Network-Aware Active Learning with Calibration

Network-aware active learning with calibration for medical image classification (EV003, EV011).

**Hypothesis**: If network-aware active learning is used, then accuracy will increase by at least 5% compared to standard active learning.
**Suggested dataset**: digits
**Baseline paper**: TBD
**Evidence**: EV003, EV011
**Difficulty**: medium

## Citation Relationships

- No citation relationships detected

## MacBook-Runnable Experiments Available

- [EV002] The AI Scientist-v2: Workshop-Level Automated Scientific Dis — No significant improvement in generalization performance with compositional regularization — Code: null
- [EV003] Agent Laboratory: Using LLM Agents as Research Assistants — ? — Code: https://github.com/agent-lab/agent-laboratory
- [EV025] Uncertainty-aware phase fraction prediction and active-learn — MDN models achieved R² values of 0.97 (FCC), 0.98 (BCC), 0.99 (Laves), 0.89 (Sigma), 0.96 (Heusler), and 0.98 (Liquid) — Code: yes
- [EV049] Robust Conformal Outlier Detection under Contaminated Refere — Label-Trim method with Power=1.627 (±0.0501) and Type-I Error=0.009 (±0.0003) on six visual datasets at α=0.01 and r=5% — Code: null
- [EV052] Active Sequential Posterior Estimation for Sample-Efficient  — ASNPE outperforms other methods across most settings with better RMSNE scores using fewer simulations — Code: yes
- [EV053] VNDUQE: Information-Theoretic Novelty Detection using Deep V — Combined strategy achieves 95.3% average AUROC, outperforming baseline MSP by 10.3 percentage points — Code: null
- [EV057] Gaussian Process Molecule Property Prediction with FlowMO — GP-SSK achieved RMSE of 0.24 on Photoswitch dataset — Code: yes
