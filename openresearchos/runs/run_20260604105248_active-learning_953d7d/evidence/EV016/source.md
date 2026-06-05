# EV016: Towards black-box parameter estimation

URL: https://www.semanticscholar.org/paper/0052a8b498c1d03f61a03035bb5941796ca37503
Year: 2023
Source: semantic_scholar
Arxiv: 2303.15041

## Abstract

Deep learning algorithms have recently been shown to be a successful tool in estimating parameters of statistical models for which simulation is easy, but likelihood computation is challenging. This is achieved by sampling a large number of parameter values from a distribution, which is typically chosen to be non-informative and cover as much of the parameter space as possible. However, for high-dimensional and large parameter spaces, covering all possible reasonable parameter values is infeasible. We propose a new sequential training procedure that reduces simulation cost and guides simulations toward the region of high parameter density based on estimates of the neural network and the observed data. Our following proposal aims to fit time series models to newly collected data at no cost using a pre-trained neural network with simulated time series of a fixed length. These approaches can successfully estimate and quantify the uncertainty of parameters from non-Gaussian models with complex spatial and temporal dependencies. The success of our methods is a first step towards a fully flexible automatic black-box estimation framework.
