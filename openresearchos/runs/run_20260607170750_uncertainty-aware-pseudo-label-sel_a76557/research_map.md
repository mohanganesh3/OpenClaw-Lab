# Research Map

Run: `run_20260607170750_uncertainty-aware-pseudo-label-sel_a76557`
Topic: uncertainty-aware pseudo-label selection for semi-supervised medical image classification under label noise
Scope: `local_experiment`
Feasibility: This research topic is highly feasible on a MacBook M4 with 36GB RAM and Apple MPS GPU, as the hardware specifications are well-suited for semi-supervised learning with medical images. Suitable public datasets include the Medical Segmentation Decathlon (with datasets like brain tumor or liver imaging), CheXpert (chest X-rays with labeled/unlabeled splits), and the COVID-19 Radiography Database. For experiment time, I estimate approximately 4-6 hours per dataset for initial model training, 2-3 ho

## Summary

- **60 papers** discovered and deep-read
- **20 papers** with full text extraction
- **11 papers** with MacBook-runnable experiments
- **3 papers** with available code
- **3 citation relationships** mapped

## Method Comparison Matrix

| Paper | Dataset | Method | Best Result | Limitations |
|-------|---------|--------|-------------|-------------|
| [EV001] Pseudo-D: Informing Multi-View Uncertain | aortic stenosis classification | The method records model predictions across traini | Pseudo-D achieves best study-level AURC of 0.885 and MAE of 0.180 |  |
| [EV003] Sam-Driven Weakly Supervised Nodule Segm | Thyroid Ultrasound Dataset (42 | The framework first generates three bounding box p | Ours achieved DSC of 73.5±22.2 and HD95 of 32.8±71.5 on Thyroid Ultrasound, and DSC of 75.5±19.8 and HD95 of 52.3±36.5 on Breast Ultrasound |  |
| [EV005] Addressing Data Scarcity in 3D Trauma De | RSNA 2023 dataset (3,147 patie | The approach uses patch-based Masked Image Modelin | Semi-supervised learning achieved 56.57% mAP@0.50 (114% improvement) for 3D injury detection |  |
| [EV006] Mutual Evidential Deep Learning for Medi | Left Atrium Dataset (LA) - 100 | MEDL employs two segmentation networks with differ | Ours: ACDC RV DSC 89.01%, JAC 80.80%, ASD 7.28mm, 95HD 1.70mm; LV DSC 93.34%, JAC 87.81%, ASD 4.25mm, 95HD 1.23mm; Avg DSC 90.40%, JAC 82.94%, ASD 3.38mm, 95HD 0.83mm | The paper acknowledges that model perfor |
| [EV008] SemSim: Revisiting Weak-to-Strong Consis | ACDC (200 cardiac MRI images f | Semi-supervised learning (SSL) for medical image s | SemSim achieved 0.896 DSC and 2.3mm 95HD on ACDC with 10% labeled data |  |
| [EV009] Rethinking Semi-Supervised Medical Image | ACDC (200 3D cardiac MRI scans | ARCO uses variance-reduced estimation with stratif | null | Focuses on medical image segmentation ra |
| [EV010] EVIL: Evidential Inference Learning for  | Automated Cardiac Diagnosis Ch | Recently, uncertainty-aware methods have attracted | EVIL achieves DSC of 89.43% with 30% labeled data | The paper mentions that current methods  |
| [EV012] Semi-supervised learning for medical ima | HAM10000 (skin cancer dataset) | ABCL creates a blended target distribution that co | UDA-WeightedCE-ABCL achieved UAR of 0.68, G-mean of 0.66, and average AUC of 0.96 on HAM10000 dataset |  |
| [EV013] Exploring Feature Representation Learnin | Pancreas dataset (12 labeled,  | The method employs a two-stage framework: first us | Ours achieves 79.81% Dice on Pancreas dataset (12 labeled, 48 unlabeled), surpassing state-of-the-art by 0.54% |  |
| [EV014] Domain-Specific, Semi-Supervised Transfe | TCIA dataset, DeepLesion datas | MAKNet uses dense blocks with MAKConv layers, foll | 0.9403 AUC on hand-labeled test set for LesaNet (TCIA) on DeepLesion dataset | Knowledge transfer across disparate doma |
| [EV016] Learning Disentangled Stain and Structur | GlaS-2017, CRAG-2019 | CSDS employs two specialized student networks (col | CSDS achieves Dice of 85.06 ± 0.57 and Jaccard of 74.87 ± 0.82 on GlaS with 10% labeled data | Limited information available from the i |
| [EV018] Uncertainty-Aware Extreme Point Tracing  | BUSI (Breast UltraSound Images | The framework uses SAM2 with bounding box prompts  | Our method achieved 67.70% IoU / 76.89% Dice on BUSI and 67.39% IoU / 79.12% Dice on UNS, surpassing fully supervised performance on BUSI | Limited to ultrasound images, may not ge |
| [EV023] DRIVE: Dual-Robustness via Information V | Office-31, Office-Home, Domain | DRIVE employs a dual-model, two-stage approach wit | Office-31: 92.7% mean accuracy | Performance degradation on extremely low |
| [EV027] Anomaly Detection in Medical Imaging - A | Brain MRI, Chest X-ray, Mammog | Qualitative literature review using Google Scholar | No specific headline result provided |  |
| [EV028] Uncertainty-Guided Mutual Consistency Le | LA dataset, BraTS 2019 dataset | A dual-task network with segmentation and signed d | UG-MCL achieved 90.16% Dice on LA dataset with 16 labeled scans and 64 unlabeled scans | The method is primarily designed for seg |
| [EV032] Reliable fairness auditing with semi-sup | MIMIC-III | Combines small labeled dataset with large unlabele | Infairness achieves lowest bias and variance for ∆TPR estimation on MIMIC-III | Limited to fairness auditing application |
| [EV035] Less Is More: Surgical Phase Recognition | Cholec80, M2CAI16 | Uses Monte Carlo Dropout for uncertainty estimatio | [object Object] | The method is specifically designed for  |
| [EV036] Complementing Semi-Supervised Learning w | CIFAR-10 (1000/4000 labels), S | The approach proposes an unsupervised uncertainty- | 94.92% on CIFAR-10 with 1000 labels | Existing SSL methods do not incorporate  |
| [EV037] Uncertainty-Aware Semi-Supervised Few Sh | PASCAL-5i (20 classes, 4 folds | Meta-train a neural network to jointly predict seg | Ours + Iq + R achieves 56.27% mean-IoU (1-shot) and 64.73% (5-shot) on PASCAL-5i with M=12 unlabeled images |  |
| [EV038] Uncertainty-Aware Distillation for Semi- | CIFAR100, miniImageNet, CUB200 | The framework operates in incremental sessions: (1 | UaD-CE achieves 54.50% on CIFAR100, 50.52% on miniImageNet, and 60.72% on CUB200 in the final session |  |

## Research Gaps

### G01: Uncertainty-calibrated pseudo-label acceptance under label noise

The evidence repeatedly studies uncertainty-aware semi-supervised medical imaging, but the selection rule for accepting pseudo-labels under noisy labeled seeds is still under-tested. A local proxy can test whether entropy, margin, and agreement filters reduce noisy pseudo-label promotion compared with naive confidence thresholding.

**Hypothesis**: Entropy-filtered, class-balanced pseudo-label selection will improve held-out classification accuracy over naive confidence pseudo-labeling when the initial labeled set contains injected label noise.
**Suggested dataset**: sklearn_digits medical-image SSL proxy with controlled label noise; remote follow-up: MedMNIST/BreastMNIST or organ-specific medical image datasets.
**Baseline paper**: [EV001] Pseudo-D: Informing Multi-View Uncertainty Estimation with Calibrated Neural Training Dynamics
**Evidence**: EV001, EV005, EV006, EV009, EV010
**Difficulty**: medium

### G02: Class-balanced pseudo-labeling for scarce medical labels

Several papers report gains from SSL, but small medical datasets can amplify class imbalance and confirmation bias. The gap is whether pseudo-label selection should explicitly balance classes instead of only using max probability.

**Hypothesis**: Class-balanced low-entropy selection will reduce minority-class collapse and improve macro/weighted F1 against max-confidence selection.
**Suggested dataset**: sklearn_digits SSL proxy with small labeled-per-class budget; remote follow-up: imbalanced medical image classification dataset.
**Baseline paper**: [EV001] Pseudo-D: Informing Multi-View Uncertainty Estimation with Calibrated Neural Training Dynamics
**Evidence**: EV005, EV009, EV010, EV012, EV013
**Difficulty**: medium

### G03: Calibration versus accuracy tradeoff in medical pseudo-label selection

Uncertainty-aware methods often optimize accuracy-like metrics, while medical deployment needs calibrated confidence. The gap is to measure whether pseudo-label filters improve accuracy while preserving calibration under noisy labels.

**Hypothesis**: Adding entropy and agreement constraints will improve accuracy without increasing expected calibration error compared with raw confidence pseudo-labeling.
**Suggested dataset**: controlled SSL proxy with noisy labels and calibration metrics; remote follow-up: MedMNIST with ECE/Brier evaluation.
**Baseline paper**: [EV001] Pseudo-D: Informing Multi-View Uncertainty Estimation with Calibrated Neural Training Dynamics
**Evidence**: EV003, EV005, EV006, EV009, EV010
**Difficulty**: medium

### G04: Failure-aware rejection of ambiguous pseudo-labels

The papers expose diagnostic uncertainty and unlabeled medical data as recurring problems, but many methods do not make pseudo-label rejection decisions auditable. The gap is a traceable reject-or-accept ledger for ambiguous unlabeled examples.

**Hypothesis**: A reject ledger based on entropy, margin, and disagreement will identify ambiguous samples that hurt baseline pseudo-label training.
**Suggested dataset**: sklearn_digits SSL proxy with saved selected/rejected pseudo-label tables.
**Baseline paper**: [EV001] Pseudo-D: Informing Multi-View Uncertainty Estimation with Calibrated Neural Training Dynamics
**Evidence**: EV005, EV006, EV009, EV010, EV013
**Difficulty**: medium

### G05: Small-run reproducibility gap for SSL medical-image claims

Many relevant papers use medical datasets or segmentation pipelines that are expensive to reproduce. The gap is a MacBook-runnable proxy experiment that validates the mechanism before claiming medical-image contribution.

**Hypothesis**: A reproducible small proxy can falsify or support the pseudo-label selection mechanism before a remote medical benchmark is attempted.
**Suggested dataset**: sklearn_digits medical-image SSL proxy; remote_compute_needed for full medical benchmark reproduction.
**Baseline paper**: [EV001] Pseudo-D: Informing Multi-View Uncertainty Estimation with Calibrated Neural Training Dynamics
**Evidence**: EV027, EV032, EV036, EV040, EV002
**Difficulty**: medium

### G06: Prior-art differentiation for uncertainty-aware SSL medical classification

The field contains many uncertainty-aware SSL methods, especially for segmentation. A contribution must differentiate classification-specific pseudo-label selection under label noise from existing consistency, co-training, and evidential-learning methods.

**Hypothesis**: A novelty tribunal comparing selection rule, uncertainty estimator, dataset setting, and failure mode will separate a narrow classification contribution from near-duplicate prior art.
**Suggested dataset**: literature-backed competitor table plus local proxy experiment.
**Baseline paper**: [EV009] Rethinking Semi-Supervised Medical Image Segmentation: A Variance-Reduction Perspective
**Evidence**: EV001, EV002, EV003, EV005, EV006
**Difficulty**: medium

## Citation Relationships

- EV005 → EV004 (baseline_comparison)
- EV013 → EV048 (baseline_comparison)
- EV013 → EV003 (baseline_comparison)

## MacBook-Runnable Experiments Available

- [EV001] Pseudo-D: Informing Multi-View Uncertainty Estimation with C — Pseudo-D achieves best study-level AURC of 0.885 and MAE of 0.180 — Code: yes
- [EV003] Sam-Driven Weakly Supervised Nodule Segmentation with Uncert — Ours achieved DSC of 73.5±22.2 and HD95 of 32.8±71.5 on Thyroid Ultrasound, and DSC of 75.5±19.8 and HD95 of 52.3±36.5 on Breast Ultrasound — Code: null
- [EV023] DRIVE: Dual-Robustness via Information Variability and Entro — Office-31: 92.7% mean accuracy — Code: yes
