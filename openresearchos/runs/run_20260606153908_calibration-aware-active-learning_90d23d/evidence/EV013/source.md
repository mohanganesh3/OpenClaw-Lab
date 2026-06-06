# EV013: Bayesian E(3)-Equivariant Interatomic Potential with Iterative Restratification of Many-body Message Passing

URL: https://www.semanticscholar.org/paper/42f70db330a43258d32eb3361d6d5b588df402bc
Year: 2025
Source: semantic_scholar
Arxiv: 2510.03046

## Abstract

Machine learning potentials (MLPs) have become essential for large-scale atomistic simulations, enabling ab initio-level accuracy with computational efficiency. However, current MLPs struggle with uncertainty quantification, limiting their reliability for active learning, calibration, and out-of-distribution (OOD) detection. We address these challenges by developing Bayesian E(3) equivariant MLPs with iterative restratification of many-body message passing. Our approach introduces the joint energy-force negative log-likelihood (NLL$_\text{JEF}$) loss function, which explicitly models uncertainty in both energies and interatomic forces, yielding substantially improved accuracy compared to conventional NLL losses. We systematically benchmark multiple Bayesian approaches, including deep ensembles with mean-variance estimation, stochastic weight averaging Gaussian, improved variational online Newton, and Laplace approximation by evaluating their performance on uncertainty prediction, OOD detection, calibration, and active learning tasks. We further demonstrate that NLL$_\text{JEF}$ facilitates efficient active learning by quantifying energy and force uncertainties. Using Bayesian active learning by disagreement (BALD), our framework outperforms random sampling and energy-uncertainty-based sampling. Our results demonstrate that Bayesian MLPs achieve competitive accuracy with state-of-the-art models while enabling uncertainty-guided active learning, OOD detection, and energy/forces calibration. This work establishes Bayesian equivariant neural networks as a powerful framework for developing uncertainty-aware MLPs for atomistic simulations at scale.
