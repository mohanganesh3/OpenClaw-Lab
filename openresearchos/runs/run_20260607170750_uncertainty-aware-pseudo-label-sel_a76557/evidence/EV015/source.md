# EV015: Semi-supervised Strong-Teacher Consistency Learning for few-shot cardiac MRI image segmentation

URL: https://www.semanticscholar.org/paper/0076218c94092dc69a339da92f764b24e1023a23
Year: 2025
Source: semantic_scholar
Arxiv: n/a

## Abstract

BACKGROUND AND OBJECTIVE
Cardiovascular disease is a leading cause of mortality worldwide. Automated analysis of heart structures in MRI is crucial for effective diagnostics. While supervised learning has advanced the field of medical image segmentation, it however requires extensive labelled data, which is often limited for cardiac MRI.


METHODS
Drawing on the principle of consistency learning, we introduce a novel semi-supervised Strong-Teacher Consistency Network for few-shot multi-class cardiac MRI image segmentation, leveraging largely available unlabelled data. This model incorporates a student-teacher architecture. A multi-teacher structure is introduced to learn diverse perspectives and avoid local optimals when dealing with largely varying cardiac structures and anatomical features. It employs a hybrid loss that emphasizes consistency between student and teacher representations, alongside supervised losses (e.g., Dice and Cross-entropy), tailored to the challenge of unlabelled data. Additionally, we introduced feature-space virtual adversarial training to enhance robust feature learning and model stability.


RESULTS
Evaluation and ablation studies on the MM-WHS and ACDC benchmark datasets show that the proposed model outperforms nine state-of-the-art semi-supervised methods, particularly with limited annotated data. It achieves 90.14% accuracy on MM-WHS and 78.45% accuracy on ACDC at labelling rates of 25% and 1%, respectively. It also highlights its unique advantages over fully-supervised and single-teacher approaches.
