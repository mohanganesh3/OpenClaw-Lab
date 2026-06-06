# EV020: Robust Interacting Multiple Model Unscented Particle Filter for Navigation

URL: https://www.semanticscholar.org/paper/0045aa8437d38056f7c0d476ef1ac0c8d7ab21b2
Year: 2020
Source: semantic_scholar
Arxiv: n/a

## Abstract

In order to solve the problems of particle degradation and difficulty in selecting importance density function in particle filter algorithm, a robust interacting multiple model unscented particle filter algorithm is presented, which is based on the advantages of interacting multiple model and particle filter algorithms. This algorithm can use the unscented transformation to get the particles that contain the latest measurement information of each model and calculate the robust equivalent weight function. This robust factor is designed to adjust the estimation and variance, and the important distribution function adaptively obtained is closer to the true distribution. Then, the particles weights can be flexibly adjusted in real time by using Euclidean distance to improve the computational efficiency during the resampling process. In addition, this filter process can comprehensively describe the uncertainty of the statistics characteristic of observation noise between different models. The diversity of available particles is increased, and the filter precision is improved. The proposed algorithm is applied to the SINS/GPS integrated navigation system, and the simulation analysis results demonstrate that the algorithm can effectively improve the filter performance and the calculation precision in positioning of integrated navigation system; thus, it provides a new method for nonlinear model filter.
