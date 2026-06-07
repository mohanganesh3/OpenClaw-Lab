# EV019: Calibration of uncertainty in the active learning of machine learning force fields

URL: https://www.semanticscholar.org/paper/00bfc9643b0c7fa719c79bfa8dd330846792c426
Year: 2023
Source: semantic_scholar
Arxiv: n/a

## Abstract

FFLUX is a machine learning force field that uses the maximum expected prediction error (MEPE) active learning algorithm to improve the efficiency of model training. MEPE uses the predictive uncertainty of a Gaussian process (GP) to balance exploration and exploitation when selecting the next training sample. However, the predictive uncertainty of a GP is unlikely to be accurate or precise immediately after training. We hypothesize that calibrating the uncertainty quantification within MEPE will improve active learning performance. We develop and test two methods to improve uncertainty estimates: post-hoc calibration of predictive uncertainty using the CRUDE algorithm, and replacing the GP with a student-t process. We investigate the impact of these methods on MEPE for single sample and batch sample active learning. Our findings suggest that post-hoc calibration does not improve the performance of active learning using the MEPE method. However, we do find that the student-t process can outperform active learning strategies and random sampling using a GP if the training set is sufficiently large.
