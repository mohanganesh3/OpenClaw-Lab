# Research Map

Run: `run_20260606131917_calibration-aware-active-learning_30f1c3`
Topic: calibration-aware active learning for image classification
Scope: `local_experiment`
Feasibility: The calibration-aware active learning for image classification topic is feasible on your MacBook M4 with 36GB RAM and MPS GPU, especially with smaller to medium-sized datasets. I recommend starting with CIFAR-10/100 for initial experiments, followed by Places365 or Oxford-IIIT Pet Dataset for more diverse scenarios. Expect experiments with CIFAR-10 to complete in approximately 2-4 hours, while larger datasets like ImageNet would require 1-2 days depending on the number of active learning iterati

## Summary

- **59 papers** discovered and deep-read
- **33 papers** with full text extraction
- **28 papers** with MacBook-runnable experiments
- **5 papers** with available code
- **1 citation relationships** mapped

## Method Comparison Matrix

| Paper | Dataset | Method | Best Result | Limitations |
|-------|---------|--------|-------------|-------------|
| [EV001] The AI Scientist: Towards Fully Automate | ? | The AI Scientist generates novel research ideas, w | Papers exceeding acceptance threshold at a top machine learning conference as judged by automated reviewer. |  |
| [EV002] The AI Scientist-v2: Workshop-Level Auto | ? | The AI Scientist-v2 eliminates reliance on human-a | First fully AI-generated paper successfully navigating peer review with acceptance above human threshold |  |
| [EV003] Agent Laboratory: Using LLM Agents as Re | ? | Agent Laboratory is a framework that uses LLM agen | 84% decrease in research expenses compared to previous autonomous research methods |  |
| [EV004] MLE-bench: Evaluating Machine Learning A | Kaggle competitions | MLE-bench evaluates ML engineering capabilities by | 16.9% of competitions |  |
| [EV005] AIBuildAI-2: Scalable Agentic AI Researc | MLE-Bench | AIBuildAI-2 is a knowledge-enhanced agent with an  | First place on MLE-Bench with 70.7% medal rate |  |
| [EV006] Reflexion: Language Agents with Verbal R | ? | Uses verbal feedback and memory to help agents imp | ? |  |
| [EV007] Self-Refine: Iterative Refinement with S | ? | Research paper: Self-Refine: Iterative Refinement  | ? |  |
| [EV008] Voyager: An Open-Ended Embodied Agent wi | ? | Demonstrates open-ended skill acquisition with mem | ? |  |
| [EV009] AgentBench: Evaluating LLMs as Agents | ? | Introduces agent evaluation tasks and motivates st | ? |  |
| [EV010] SWE-agent: Agent-Computer Interfaces Ena | SWE-bench, HumanEvalFix | SWE-agent introduces a custom agent-computer inter | 87.7% on HumanEvalFix |  |
| [EV011] Ask-n-Learn: Active Learning via Reliabl | ? | Deep predictive models rely on human supervision i | ? |  |
| [EV012] Active Output Selection Strategies for M | ? | Active learning shows promise to decrease test ben | ? |  |
| [EV017] Multi-Tier Labeling and Physics-Informed | Two-Line Element (TLE) records | The paper presents a multi-tier labeling cascade t | 62.8% decay recall | The model is framed as a high-recall tri |
| [EV021] Robust Conformal Outlier Detection under | ? | Conformal prediction is a flexible framework for c | ? |  |
| [EV024] Active Sequential Posterior Estimation f | ? | Computer simulations have long presented the excit | ? |  |
| [EV025] VNDUQE: Information-Theoretic Novelty De | MNIST | The paper uses Deep Variational Information Bottle | 100% AUROC on noise for far-OOD samples using KL divergence |  |
| [EV026] A Machine Learning Approach Capturing Hi | ? | Mechanism to account for hidden parameter variatio | ? |  |
| [EV028] Deep-Learning Control of Lower-Limb Exos | ? | Partial-assistance exoskeletons hold significant p | ? |  |
| [EV029] Gaussian Process Molecule Property Predi | ? | We present FlowMO: an open-source Python library f | ? |  |
| [EV031] Murphys Laws of AI Alignment: Why the Ga | ? | Research paper: Murphys Laws of AI Alignment: Why  | ? |  |

## Research Gaps

### G01: Automated Discovery of Calibration-Aware AL Strategies

While AI agents can generate research ideas, they have not been tasked with specifically discovering novel active learning strategies that integrate model calibration. The current generation of AI Scientists focuses on broad idea generation, not on the nuanced intersection of uncertainty sampling and calibration metrics.

**Hypothesis**: If an AI Scientist agent is given the topic 'calibration-aware active learning for image classification', it will not generate a novel, testable hypothesis about the interaction between uncertainty sampling and calibration metrics.
**Suggested dataset**: CIFAR-10
**Baseline paper**: TBD
**Evidence**: EV001, EV002
**Difficulty**: medium

### G02: Self-Reflective Active Learning Loops

Active learning systems lack a mechanism for self-reflection on their querying strategy, as demonstrated by agents like Reflexion and Self-Refine. An AL loop could be improved by having the system critique its own sample selection based on resulting calibration changes.

**Hypothesis**: If an active learning system incorporates a Reflexion-like feedback loop to critique its own querying strategy, it will achieve better model calibration (e.g., lower ECE) for the same number of queried labels compared to a standard AL system.
**Suggested dataset**: CheXpert
**Baseline paper**: TBD
**Evidence**: EV006, EV007
**Difficulty**: hard

### G03: Benchmarking Calibration-Aware AL

There is no established benchmark for evaluating different calibration-aware active learning methods, similar to how AgentBench and MLE-bench evaluate general agent capabilities. This gap prevents systematic comparison and progress in the field.

**Hypothesis**: If a new benchmark for calibration-aware active learning is created, it will reveal that existing state-of-the-art AL methods perform poorly on tasks requiring high-stakes calibration, highlighting the need for new research.
**Suggested dataset**: A custom benchmark dataset with 'hard-to-calibrate' samples.
**Baseline paper**: TBD
**Evidence**: EV009, EV004
**Difficulty**: medium

### G04: Augmenting AL with External Knowledge

Active learning agents do not leverage external knowledge bases to inform their querying decisions. An agent could query an external API or knowledge base to resolve ambiguity before deciding to query a human oracle, improving efficiency.

**Hypothesis**: If an active learning system is augmented with an external knowledge base (like AIBuildAI-2), it will select more informative samples to query, leading to faster convergence in both accuracy and calibration.
**Suggested dataset**: CUB-200-2011 (fine-grained bird classification)
**Baseline paper**: TBD
**Evidence**: EV005, EV010
**Difficulty**: hard

### G05: Calibration in Open-Ended AL Scenarios

Current active learning is designed for a fixed set of classes, whereas open-ended learning agents like Voyager continuously acquire new skills. The role of calibration-awareness is undefined when the model's output space is constantly expanding.

**Hypothesis**: If an open-ended active learning agent is made calibration-aware, it will be better at distinguishing between a poorly calibrated sample of a known class and a truly novel class requiring new labels.
**Suggested dataset**: A dataset simulating a stream of new classes, such as incremental splits of ImageNet.
**Baseline paper**: TBD
**Evidence**: EV008
**Difficulty**: hard

### G06: Human-Centric Query Design via Agent Interfaces

Active learning focuses on optimizing for the model, not the human annotator. Agents like SWE-agent use custom interfaces; an AL agent could use a similar interface to design queries that are not only informative for the model but also easy for a human to label.

**Hypothesis**: If an active learning agent uses a custom interface to present queries, it can select samples that reduce human annotation time by prioritizing clarity and ease of labeling over pure model uncertainty.
**Suggested dataset**: A dataset with subjective or complex labels, such as art genre classification.
**Baseline paper**: TBD
**Evidence**: EV010
**Difficulty**: medium

## Citation Relationships

- EV004 → EV008 (baseline_comparison)

## MacBook-Runnable Experiments Available

- [EV003] Agent Laboratory: Using LLM Agents as Research Assistants — 84% decrease in research expenses compared to previous autonomous research methods — Code: yes
- [EV004] MLE-bench: Evaluating Machine Learning Agents on Machine Lea — 16.9% of competitions — Code: yes
- [EV010] SWE-agent: Agent-Computer Interfaces Enable Automated Softwa — 87.7% on HumanEvalFix — Code: yes
