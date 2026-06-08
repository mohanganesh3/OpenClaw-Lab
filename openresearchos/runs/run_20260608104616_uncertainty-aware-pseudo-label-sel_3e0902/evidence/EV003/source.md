# EV003: Pseudo-D: Informing Multi-View Uncertainty Estimation with Calibrated Neural Training Dynamics

URL: https://www.semanticscholar.org/paper/035bdadfaa183eca70e1869774ef01a32ca0c4b5
Year: 2025
Source: semantic_scholar
Arxiv: 2509.11800

## Abstract

Computer-aided diagnosis systems must make critical decisions from medical images that are often noisy, ambiguous, or conflicting, yet today's models are trained on overly simplistic labels that ignore diagnostic uncertainty. One-hot labels erase inter-rater variability and force models to make overconfident predictions, especially when faced with incomplete or artifact-laden inputs. We address this gap by introducing a novel framework that brings uncertainty back into the label space. Our method leverages neural network training dynamics (NNTD) to assess the inherent difficulty of each training sample. By aggregating and calibrating model predictions during training, we generate uncertainty-aware pseudo-labels that reflect the ambiguity encountered during learning. This label augmentation approach is architecture-agnostic and can be applied to any supervised learning pipeline to enhance uncertainty estimation and robustness. We validate our approach on a challenging echocardiography classification benchmark, demonstrating superior performance over specialized baselines in calibration, selective classification, and multi-view fusion.
