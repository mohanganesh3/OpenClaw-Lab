# EV020: Efficient Uncertainty Estimation in Semantic Segmentation via Distillation

URL: https://www.semanticscholar.org/paper/006d395b773636e539d0b23aecb1a8e23c99c20c
Year: 2021
Source: semantic_scholar
Arxiv: n/a

## Abstract

Deep neural networks typically make predictions with little regard for the probability that a prediction might be incorrect. Attempts to address this often involve input data undergoing multiple forward passes, either of multiple models or of multiple configurations of a single model, and consensus among outputs is used as a measure of confidence. This can be computationally expensive, as the time taken to process a single input sample increases linearly with the number of output samples being generated, an important consideration in real-time scenarios such as autonomous driving, and so we propose Uncertainty Distillation as a more efficient method for quantifying prediction uncertainty. Inspired by the concept of Knowledge Distillation, whereby the performance of a compact model is improved by training it to mimic the outputs of a larger model, we train a compact model to mimic the output distribution of a large ensemble of models, such that for each output there is a prediction and a predicted level of uncertainty for that prediction. We apply Uncertainty Distillation in the context of a semantic segmentation task for autonomous vehicle scene understanding and demonstrate a capability to reliably predict pixelwise uncertainty over the resultant class probability map. We also show that the aggregate pixel uncertainty across an image can be used as a metric for reliable detection of out-of-distribution data.
