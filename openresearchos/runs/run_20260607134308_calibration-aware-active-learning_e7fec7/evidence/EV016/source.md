# EV016: A Leakage-Aware Drug Discovery Workflow for PKM2 and MAPK1 Integrating Scaffold Validation, Molecular Docking and Structural Triage

URL: https://www.semanticscholar.org/paper/857fe56b39ba5250d249e1dc8c6ef548007c0fb2
Year: 2026
Source: semantic_scholar
Arxiv: n/a

## Abstract

Computer-aided drug discovery increasingly depends on virtual-screening workflows that remain reliable under severe class imbalance, chemical redundancy and early-recognition constraints. In this study, we developed a leakage-aware prioritization workflow for two cancer-relevant targets, pyruvate kinase M2 (PKM2) and mitogen-activated protein kinase 1 (MAPK1/ERK2), using the LIT-PCBA benchmark. The workflow combines canonical-SMILES curation, duplicate and label-conflict auditing, scaffold-aware validation, a non-learning nearest-active Tanimoto baseline, imbalance-aware machine-learning models, repeated-seed robustness analysis, isotonic probability calibration, ensemble-disagreement estimation, absorption, distribution, metabolism, excretion and toxicity (ADMET)-aware triage, molecular docking, and residue-level contact analysis. Benchmark enrichment is interpreted alongside calibration, ADMET filtering, docking and residue-contact evidence, rather than as a standalone discovery claim. PKM2 emerged as the clearer machine-learning case, with scaffold-aware tree models improving early recognition beyond the nearest-active similarity baseline and yielding top-ranked candidates supported by calibrated activity scores, ADMET profiles, docking scores, and residue-contact fingerprints. MAPK1 provided a biologically relevant contrast target, where ligand-neighborhood similarity remained competitive and downstream structural triage became more decisive than ligand-based ranking alone. These results support a conservative drug-discovery workflow in which leakage-aware benchmarking, calibration, uncertainty, and molecular-level triage remain visible throughout candidate prioritization.
