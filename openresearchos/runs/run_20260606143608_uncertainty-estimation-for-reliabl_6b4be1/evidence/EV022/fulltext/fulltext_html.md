[2303.15041] 1 Introduction

Towards Black-Box Parameter Estimation

Amanda Lenzi  1

1

1

School of Mathematics, University of Edinburgh,
Edinburgh EH9 3FD, Scotland, United Kingdom.

and
Haavard Rue  2

2

2

Statistics Program,
King Abdullah University of Science and Technology,
Thuwal 23955-6900, Saudi Arabia.

Abstract

Deep learning algorithms have recently shown to be a successful tool in estimating parameters of statistical models for which simulation is easy, but likelihood computation is challenging.
But the success of these approaches depends on simulating parameters that sufficiently reproduce the observed data, and, at present, there is a lack of efficient methods to produce these simulations.
We develop new black-box procedures to estimate parameters of statistical models based only on weak parameter structure assumptions.
For well-structured likelihoods with frequent occurrences, such as in time series, this is achieved by pre-training a deep neural network on an extensive simulated database that covers a wide range of data sizes.
For other types of complex dependencies, an iterative algorithm guides simulations to the correct parameter region in multiple rounds.
These approaches can successfully estimate and quantify the uncertainty of parameters from non-Gaussian models with complex spatial and temporal dependencies.
The success of our methods is a first step towards a fully flexible automatic black-box estimation framework.

Keywords:  Deep neural networks, intractable likelihoods, sequential, time-series

Short title : Black-box Estimation

1  Introduction

Statistical modeling consists of first devising stochastic models for phenomena we want to learn about and then to relate those models to data.
These stochastic models have unknown parameters and the second step boils down to estimating these parameters from data through the likelihood function.
However, there might be a discrepancy between these two steps, as models for describing mechanisms aim for scientific adequacy rather than computational tractability.
Indeed, as soon as we move away from Gaussian processes as the canonical model for dependent data, likelihood computation becomes effectively impossible, and inference is too complicated for traditional estimation methods.
Consider, for instance, datasets from finance or climate science, where skewness and jumps are commonly present and
calculating the likelihood in closed form is often impossible, ruling out any numerical likelihood maximization and Bayesian methods.
Yet it is possible to simulate from those models, and the question becomes whether the simulations look like the data.

Much effort has been directed toward the development of methods of approximate parameter estimation, often referred to as indirect inference  (Gourieroux et al.,,  1993 ) , likelihood-free inference  (Grelaud et al.,,  2009 ; Gutmann and Corander,,  2016 ) , simulation-based inference  (Nickl and Pötscher,,  2010 )  or synthetic likelihood  (Wood,,  2010 ) ; for an overview, see, for example, the review by  Hartig et al., ( 2011 ); Cranmer et al., ( 2020 ) .
The typical assumption by the different methods is that exact likelihood evaluation is hard to obtain but it is easy to simulate from the model given the parameter values, and the basic idea is to identify the model parameters which yield simulated data that resemble the observed data.
The most common in this umbrella is arguably approximate Bayesian computation (ABC)  (Fearnhead and Prangle,,  2012 ; Frazier et al.,,  2018 ; Sisson et al.,,  2018 ) , which avoids evaluating intractable likelihoods by matching summary statistics from observations with those computed from simulated data based on parameters drawn from a predefined prior distribution.
The likelihood is approximated by the probability that the condition

γ  ​

(

x  sim

,

x  obs

)

&lt;  ϵ

𝛾

subscript  𝑥

sim

subscript  𝑥

obs

italic-ϵ

\gamma(x_{\mbox{sim}},x_{\mbox{obs}})&lt;\epsilon

is satisfied, where

γ

𝛾

\gamma

is some distance measure and the value of

ϵ

italic-ϵ

\epsilon

is a trade-off between sample efficiency and inference quality.
In simpler cases, sufficient statistics are used as they provide all the information in the data, however, for complex models they are unlike to exist and it is not obvious which statistics will be most informative.
Several works have proposed procedures for designing summary statistics  (Fearnhead and Prangle,,  2012 ; Jiang et al.,,  2017 ) .

Despite its popularity, ABC is not scalable to large numbers of observations, since
inference for new data requires repeating most steps of the procedure. To address this issue, recent progress in deep learning capabilities, such as the integration of advanced automatic differentiation and probabilistic programming within the modeling workflow, were used, e.g., to approximate density ratios proportional to the likelihood estimation using a classifier in  Hermans et al., ( 2020 ) , parameter estimation in Gaussian processes  (Gerber and Nychka,,  2020 )  and in multivariate extremes  (Lenzi et al.,,  2021 ; Sainsbury-Dale et al.,,  2022 ) .
Specifically for parameter estimation of spatial covariance functions,  Gerber and Nychka, ( 2020 )  used maximum likelihood estimators (MLEs) for training convolutional neural networks (CNNs).
While they showed that their framework improves the computational aspects of inference for classical spatial models, it is case-specific and not scalable, as the CNN must be retrained with new MLEs for each new testing dataset.
Additionally, exact methods, such as MLEs, are not available for intractable models that most benefit from an alternative estimation approach.
Alternatively,  Lenzi et al., ( 2021 )  avoided the shortcoming of using MLEs and proposed to computed parameter estimates from approximate likelihood methods and used those to design training data for the CNN.

The ABC and deep learning approaches mentioned above work best when the simulated data are from the same parameter space as the observations’ parameters, which are unknown.
In high dimensions and large parameter spaces, constructing a training parameter space large enough to cover all possible reasonable parameter values is infeasible. Instead, one needs a rule for picking the region to simulate data.
However, ideally, the inference mechanism should be able to automatically perform inference without restrictive assumptions about the generating process, knowledge from experts, or computationally expensive preliminary steps.
This work is about bridging this gap by employing general black-box methodologies that only require weak assumptions on the data-generating process and generalize to new datasets without needing to repeat computationally expensive steps.
The proposed methods are a step further toward an automatic and generic workflow for performing statistical inference in intractable models.

In more detail, our black-box approaches can be divided into two categories:
(A) a fully automatic iterative approach that modifies the training data using arbitrary, dynamically updated distribution parameters until it reaches the actual parameters in the data, and
(B) A database that is extensive enough to handle estimation for different data sizes from the same model but that needs to be trained only once.
We first demonstrate the success of the sequential framework in (A) in an independent, identically distributed (i.i.d.) Gaussian example and further extend it to intractable likelihoods for modeling spatial extremes. In both cases, the estimators dramatically reduce the bias of an initial guess and eventually approximate the actual parameter quite accurately, even when the initial training does not contain the truth.
The strategy in (B) is designed for stationary time series data, and we show its usefulness for estimating parameters of a model widely used in finance, namely non-Gaussian stochastic volatility models.
In such applica