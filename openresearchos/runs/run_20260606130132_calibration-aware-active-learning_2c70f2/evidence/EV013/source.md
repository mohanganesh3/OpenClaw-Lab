# EV013: Uncertainty quantification enables reliable deep learning for protein–ligand binding affinity prediction

URL: https://www.semanticscholar.org/paper/0844b5be2c6bcee16cf81cc7bd85a856d6a17a72
Year: 2025
Source: semantic_scholar
Arxiv: n/a

## Abstract

Deep learning (DL) algorithms have increasingly been applied to predict protein-ligand binding affinity, a critical step in drug design. Yet, many models still struggle to generalize to unseen data, and when coupled with the absence of confidence estimates for predictions, they hinder effective decision-making. To address these challenges, we thoroughly compare five uncertainty quantification methods: Deep Ensemble, Monte Carlo Dropout, Laplace approximation, Bayes by Backprop, and Evidential Neural Networks. Notably, Bayes by Backprop—applied for the first time in this study area—offers a novel and promising approach to uncertainty quantification. To ensure unbiased training and validation, we leverage the Leak-Proof PDBBind dataset and rigorously evaluate performance across multiple external test sets. Our results reveal that a feed-forward neural network (FFNN) using extended connectivity interaction features (ECIF) as a protein-ligand representation, paired with the Bayes by Backprop method, achieves superior predictive performance and highly reliable uncertainty quantification. Notably, Bayes by Backprop demonstrated balanced performance across multiple evaluation metrics, particularly excelling in calibration without needing additional recalibration. Our findings not only advance the state of uncertainty quantification in deep learning models for binding affinity prediction but also open avenues for more reliable, calibrated, and reproducible applications in drug discovery and active learning-driven model development.
