# EV016: MOSE: A Monotonic Selectivity Estimator Using Learned CDF

URL: https://www.semanticscholar.org/paper/b67c9d9c32ee5dc8c4394fe78e8715e7e5fdb1c8
Year: 2023
Source: semantic_scholar
Arxiv: n/a

## Abstract

The accuracy of selectivity estimation is of vital importance to create good query plans. Traditional estimators such as histograms make several assumptions during estimation that can lead to huge errors. Recently the database community started exploring the usage of machine learning in selectivity estimation and won great achievements. However, due to the black box models they used, existing learning-based methods still face a variety of new challenges, including high estimation latency, large training data demanding, and occurrence of illogical results. In this work, we propose a learning-based MOnotonic Selectivity Estimator (MOSE) to address these challenges. We first learn a multi-dimensional Cumulative Distribution Function (CDF) of the data in a supervised method and then compute selectivity for ad hoc query predicates at rum-time. We propose a novel regularizer and an effective attribute-aware calibration method to improve the estimation accuracy. To further improve the model efficiency, we design a mutual information based attributes partition algorithm for model ensemble. With regard to the heavy cost of training data collection, we design a model-based active learning strategy to generate high-quality training data cost-effectively. We conduct extensive experiments on both real-world and synthetic datasets and the results show that MOSE outperforms the state-of-the-art methods in terms of accuracy and efficiency.
