# EV004: Revisiting Unknowns: Towards Effective and Efficient Open-Set Active Learning

URL: https://www.semanticscholar.org/paper/a1ff254e516dcdfc97ce26dbe53bcb753ffa3877
Year: 2026
Source: semantic_scholar
Arxiv: 2603.07898

## Abstract

Open-set active learning (OSAL) aims to identify informative samples for annotation when unlabeled data may contain previously unseen classes-a common challenge in safety-critical and open-world scenarios. Existing approaches typically rely on separately trained open-set detectors, introducing substantial training overhead and overlooking the supervisory value of labeled unknowns for improving known-class learning. In this paper, we propose E$^2$OAL (Effective and Efficient Open-set Active Learning), a unified and detector-free framework that fully exploits labeled unknowns for both stronger supervision and more reliable querying. E$^2$OAL first uncovers the latent class structure of unknowns through label-guided clustering in a frozen contrastively pre-trained feature space, optimized by a structure-aware F1-product objective. To leverage labeled unknowns, it employs a Dirichlet-calibrated auxiliary head that jointly models known and unknown categories, improving both confidence calibration and known-class discrimination. Building on this, a logit-margin purity score estimates the likelihood of known classes to construct a high-purity candidate pool, while an OSAL-specific informativeness metric prioritizes partially ambiguous yet reliable samples. These components together form a flexible two-stage query strategy with adaptive precision control and minimal hyperparameter sensitivity. Extensive experiments across multiple OSAL benchmarks demonstrate that E$^2$OAL consistently surpasses state-of-the-art methods in accuracy, efficiency, and query precision, highlighting its effectiveness and practicality for real-world applications. The code is available at github.com/chenchenzong/E2OAL.
