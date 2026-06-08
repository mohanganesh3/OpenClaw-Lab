# Research Map

Run: `run_20260608104616_uncertainty-aware-pseudo-label-sel_3e0902`
Topic: uncertainty-aware pseudo-label selection
Scope: `local_experiment`
Feasibility: The research topic on uncertainty-aware pseudo-label selection is highly feasible on your MacBook M4 with 36GB RAM and MPS GPU, as this hardware can handle most semi-supervised learning workloads effectively. Suitable public datasets include CIFAR-10/100 for initial experiments, SVHN for more challenging scenarios, and potentially ImageNet for scaling tests (all downloadable beforehand). You can expect initial experiments to complete within 1-2 days, with comprehensive results achievable in 1-2 

## Summary

- **59 papers** discovered and deep-read
- **20 papers** with full text extraction
- **11 papers** with MacBook-runnable experiments
- **3 papers** with available code
- **0 citation relationships** mapped

## Method Comparison Matrix

| Paper | Dataset | Method | Best Result | Limitations |
|-------|---------|--------|-------------|-------------|
| [EV001] Sam-Driven Weakly Supervised Nodule Segm | ? | The framework generates three types of bounding bo | ? | The paper assumes aspect ratio annotatio |
| [EV003] Pseudo-D: Informing Multi-View Uncertain | aortic stenosis classification | The method leverages neural network training dynam | Pseudo-D achieves best performance: Study-level MAE ↓ 0.180, ACC ↑ 0.820, ECE ↓ 0.071, AURC ↑ 0.885 | Focus is limited to medical images, whic |
| [EV004] You can't handle the (dirty) truth: Data | MNIST (100 labeled, 60000 unla | DIPS characterizes and selects useful labeled and  | DIPS achieved 94.2% accuracy on CIFAR-10 with 20% label noise | The paper acknowledges that existing pse |
| [EV010] Annotation-Efficient Active Test-Time Ad | PACS (4 domains, 7 categories) | CPATTA employs two complementary conformal predict | CPATTA with α=0.2 achieves 87.13% post-adaptation accuracy on PACS, improving over SimATTA by nearly 9% |  |
| [EV013] PTCL: Pseudo-Label Temporal Curriculum L | Wikipedia, Reddit, Dsub, CoOAG | Dynamic node classification is critical for modeli | 87.97% AUC (PTCL with TGN backbone on Wikipedia) |  |
| [EV015] Uncertainty-Aware Extreme Point Tracing  | Breast UltraSound Images (BUSI | The framework first uses four extreme points to ge | BUSI dataset: 67.70% IoU / 76.89% Dice (exceeding fully supervised 66.71% IoU / 76.01% Dice); UNS dataset: 67.39% IoU / 79.12% Dice (comparable to fully supervised 67.54% IoU / 79.08% Dice) | The paper relies on only four extreme po |
| [EV017] A Semi-Supervised Inf-Net Framework for  | LUNA16 dataset (annotated CT v | The framework utilizes Inf-Net architecture with P | Semi-Inf-Net with 0.784 accuracy on 20,000 images (labeled + pseudo-labeled) | The abstract is cut off before discussin |
| [EV025] CRMSP: A Semi-supervised Approach for Ke | FUNSD (149 train/50 test), COR | CRMSP utilizes a base model with labeled/unlabeled | CRMSP achieved 91.30% F1-score on CORD with 10% labeled data using LayoutLMv3, close to fully-supervised result of 93.30% | The paper does not explicitly state limi |
| [EV026] DRIVE: Dual-Robustness via Information V | Office-31, Office-Home, Domain | DRIVE employs a dual-model architecture with one m | DRIVE achieves 92.7% mean accuracy on Office-31 | Existing SFUDA methods often rely on sin |
| [EV029] Learning Label Refinement and Threshold  | CIFAR-10 (n1=1500, n1=500), CI | SEVAL optimizes logit offsets π to minimize cross- | ReMixMatch with SEVAL: 86.7% on CIFAR-100 n1=50 | The abstract doesn't explicitly state li |
| [EV030] Toward Fast Personalized Semi-Supervised | MNIST, CIFAR-10 | FedCPSL simultaneously optimizes global model θ, p | FedCPSL achieves 98.2% accuracy on MNIST (m=0.1N) and 80.7% on CIFAR-10 (m=0.1N) |  |
| [EV035] PromptEM: Prompt-tuning for Low-resource | REL-HETER, SEMI-HOMO, SEMI-HET | Entity Matching (EM), which aims to identify wheth | PromptEM achieves 100% F1-score on REL-HETER and REL-TEXT datasets |  |
| [EV038] Complementing Semi-Supervised Learning w | CIFAR-10, SVHN, CIFAR-100, Min | Proposes an unsupervised uncertainty-aware loss fu | 79.23% on CIFAR-100 with 4000 labels | Existing SSL methods do not incorporate  |
| [EV040] Domain-Specific, Semi-Supervised Transfe | TCIA dataset, DeepLesion datas | The authors propose MAKNet, a lightweight architec | LesaNet (TCIA) on hand-labeled test set: AUC 0.9403, F1 0.4972, Precision 0.4754, Recall 0.7531 | Knowledge transfer across disparate doma |
| [EV041] Reliable Source Approximation: Source-Fr | Public vestibular schwannoma d | RSA deploys a conditional diffusion model to gener | Ours with centralized fine-tuning achieved Dice of 77.83 ± 0.98 and ASSD of 1.24 ± 0.11 | Current SFUDA approaches cannot tackle c |
| [EV042] Wasserstein Gradient Boosting: A Framewo | ? | WGBoost extends gradient boosting by using Wassers | ? | Not explicitly stated in the abstract |
| [EV044] Adaptive Betweenness Clustering for Semi | Office-31, DomainNet, Office-H | Constructs a heterogeneous graph between labeled s | G-ABC achieved 72.02% mean accuracy on Office-31 3-shot |  |
| [EV045] Scribble-based 3D Multiple Abdominal Org | WORD dataset (cervical cancer  | TDNet uses a shared encoder feeding three decoders | Average DSC of 85.74±5.22% on WORD testing set | Performance gap between weakly-supervise |
| [EV049] Transfer learning with pre-trained condi | ? | The approach synthesizes artificial target dataset | ? | Assumes access to conditional source gen |
| [EV050] Semi-Supervised Learning Based on Refere | target dataset with sizes: 30, | The method involves pre-training a reference model | Best result with ω = 0.1 and 2000 recordings achieving MOS of 4.14 ± 0.02 and WER of 0.2% | The paper does not explicitly state any  |

## Research Gaps

### G01: Domain Shift in SAM-based Pseudo-labeling for Medical Imaging

SAM-based pseudo-labeling suffers from domain shift when applied to medical imaging, as evidenced by EV001's weakness and EV003's focus on uncertainty-aware pseudo-labeling in medical imaging.

**Hypothesis**: If SAM is applied to medical images, the accuracy will be lower than on natural images by at least 10% due to domain shift.
**Suggested dataset**: breast_cancer or digits
**Baseline paper**: TBD
**Evidence**: EV001, EV003
**Difficulty**: medium

### G02: Calibration Methods in Uncertainty-Aware Pseudo-labeling

EV003 highlights a lack of details on calibration methods in uncertainty-aware pseudo-labeling, indicating a need for more principled approaches.

**Hypothesis**: Implementing temperature scaling will reduce ECE by at least 0.05 compared to vanilla pseudo-labeling.
**Suggested dataset**: digits
**Baseline paper**: TBD
**Evidence**: EV003
**Difficulty**: medium

### G03: Labeled Data Quality Impact on Pseudo-label Selection

EV004 suggests that labeled data quality is crucial in pseudo-labeling, but its impact is not fully explored.

**Hypothesis**: Introducing 20% label noise will decrease accuracy by at least 5% compared to clean labels.
**Suggested dataset**: breast_cancer
**Baseline paper**: TBD
**Evidence**: EV004
**Difficulty**: easy

### G04: Generalization of ATTA to Non-Classification Tasks

EV010 notes unclear generalization of ATTA beyond classification, indicating a gap in applying uncertainty-aware methods to other tasks.

**Hypothesis**: CPATTA applied to regression will achieve at least 80% of the performance of a supervised baseline.
**Suggested dataset**: Boston housing
**Baseline paper**: TBD
**Evidence**: EV010
**Difficulty**: hard

### G05: Domain Shift in Uncertainty-Aware Methods

Both EV001 and EV010 highlight domain shift as a critical issue, with open questions on how uncertainty-aware methods perform across domains.

**Hypothesis**: Uncertainty-aware methods will show a performance drop of at least 15% when tested on a different domain.
**Suggested dataset**: digits or Fashion-MNIST
**Baseline paper**: TBD
**Evidence**: EV001, EV010
**Difficulty**: medium

### G06: Combining Confidence and Uncertainty in Pseudo-labels

EV010 suggests potential for combining confidence and uncertainty in pseudo-label selection, which is not yet explored.

**Hypothesis**: A combined confidence-uncertainty metric will improve accuracy by at least 3% over using either alone.
**Suggested dataset**: breast_cancer
**Baseline paper**: TBD
**Evidence**: EV010
**Difficulty**: medium

## Citation Relationships

- No citation relationships detected

## MacBook-Runnable Experiments Available

- [EV029] Learning Label Refinement and Threshold Adjustment for Imbal — ReMixMatch with SEVAL: 86.7% on CIFAR-100 n1=50 — Code: yes
- [EV038] Complementing Semi-Supervised Learning with Uncertainty Quan — 79.23% on CIFAR-100 with 4000 labels — Code: null
