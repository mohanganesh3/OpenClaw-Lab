[2407.15401] Data Space Inversion for Efficient Predictions and Uncertainty Quantification for Geothermal Models

Data Space Inversion for Efficient Predictions and Uncertainty Quantification for Geothermal Models

Abstract

The ability to make accurate predictions with quantified uncertainty provides a crucial foundation for the successful management of a geothermal reservoir. Conventional approaches for making predictions using geothermal reservoir models involve estimating unknown model parameters using field data, then propagating the uncertainty in these estimates through to the predictive quantities of interest. However, the unknown parameters are not always of direct interest; instead, the predictions are of primary importance. Data space inversion (DSI) is an alternative methodology that allows for the efficient estimation of predictive quantities of interest, with quantified uncertainty, that avoids the need to estimate model parameters entirely. In this paper, we evaluate the applicability of DSI to geothermal reservoir modelling. We first review the processes of model calibration, prediction and uncertainty quantification from a Bayesian perspective, and introduce data space inversion as a simple, efficient technique for approximating the posterior predictive distribution. We then apply the DSI framework to two model problems in geothermal reservoir modelling. We evaluate the accuracy and efficiency of DSI relative to other common methods for uncertainty quantification, study how the number of reservoir model simulations affects the resulting approximation to the posterior predictive distribution, and demonstrate how the framework can be enhanced through the use of suitable reparametrisations. Our results support the idea that data space inversion is a simple, robust and efficient technique for making predictions with quantified uncertainty using geothermal reservoir models, providing a useful alternative to more conventional approaches.

\draftfalse  \journalname

Water Resources Research

Department of Engineering Science and Biomedical Engineering, University of Auckland, Auckland 1010, New Zealand

\correspondingauthor

Alex de Beeradeb970@aucklanduni.ac.nz

{keypoints}

We review data space inversion (DSI), an efficient method for making model predictions with quantified uncertainty, in a geothermal context

We provide a thorough evaluation of the DSI framework using two model problems arising in geothermal reservoir engineering

Our results show that DSI can produce accurate predictions at a significantly lower computational cost than other conventional methods

1  Introduction

Computational models are widely used in geothermal reservoir engineering to facilitate effective decision making  [ O’Sullivan  \BBA  O’Sullivan ( \APACyear 2016) ] . One of the key features of these models is the ability to make predictions with quantified uncertainty. Computing accurate predictions and uncertainty estimates generally requires calibration of the model; that is, the estimation of model parameters, such as the subsurface permeability structure and the strength and location of the deep mass upflows at the base of the system, using observations such as downhole temperature and pressure measurements. In many situations, the parameters themselves are not of direct interest; instead, the predictions are of primary importance. However, the calibration process is typically the most computationally demanding step in the process of making predictions.

Here we discuss the application of the data space inversion (DSI) methodology  [ Sun  \BBA  Durlofsky ( \APACyear 2017) ,  Sun  \BOthers . ( \APACyear 2017) ]  for making predictions as well as providing associated estimates of uncertainty. The DSI framework provides several computational advantages, the most significant being the ability to effectively circumvent the calibration process; rather, it simply estimates the values of predictive quantities of interest conditioned on measured data. This idea is sometimes referred to as  direct forecasting . Furthermore, the DSI approach does not require access to model derivatives (in fact, the method can be applied to non-differentiable models) while much of the required computation can be carried out in parallel. The DSI approach has been used successfully in a variety of applications, including subsurface hydrology  [ Delottier  \BOthers . ( \APACyear 2023) ,  Jiang  \BOthers . ( \APACyear 2020) ,  Jiang  \BBA  Durlofsky ( \APACyear 2021) ,  Wu  \BOthers . ( \APACyear 2021) ]  and petroleum engineering  [ Lima  \BOthers . ( \APACyear 2020) ,  Liu  \BOthers . ( \APACyear 2021) ] . The application of the framework to geothermal reservoir modelling, however, is largely unexplored and potentially more challenging as the governing equations are generally highly nonlinear in the geothermal context, with simulation non-convergence being a common issue (see, e.g.,  \citeA croucher2020waiwera, o2013improved).

We note that the idea of direct forecasting is not exclusive to DSI; in particular, the Bayesian evidential learning (BEL) framework  [ Scheidt  \BOthers . ( \APACyear 2018) ]  also involves direct forecasting. Like DSI, the BEL framework has been applied in a variety of subsurface modelling applications  [ Hermans  \BOthers . ( \APACyear 2018) ,  Michel  \BOthers . ( \APACyear 2020) ,  Pradhan  \BBA  Mukerji ( \APACyear 2020) ] ; most notably,  \citeA athens2019monte demonstrate the application of BEL to predict the temperature in a geothermal target area of a synthetic model based on Dixie Valley, Nevada. We note, however, that only single-phase, natural state simulations are considered in this study; by contrast, we apply the DSI framework to a two-phase problem, and consider both natural state and production history simulations.

In this paper, we build on our previous work  [ Power  \BOthers . ( \APACyear 2022) ]  to investigate the applicability of the DSI methodology to geothermal reservoir modelling. We consider two synthetic model problems (outlined in Section

3  ); one based on a simplified two-dimensional reservoir and one based on a large-scale, transient three-dimensional reservoir. Through these model problems, we provide a numerical comparison between the DSI framework and other methods for uncertainty quantification in subsurface modelling, investigate how the number of reservoir model simulations affects the resulting approximation to the posterior predictive distribution, and illustrate the value of applying suitable transformations to the observations or predictive quantities of interest prior to the use of the DSI framework.

Before introducing the DSI framework, we briefly recall the key concepts and steps involved in a typical (Bayesian) statistical approach to geothermal model calibration and prediction, and the associated uncertainty quantification.

2  Methodology

We first introduce some important notation used throughout the remainder of this paper. We reserve bold lowercase letters for vectors (i.e.,

𝒗  ∈

ℝ  n

𝒗

superscript  ℝ  𝑛

\boldsymbol{v}\in\mathbb{R}^{n}

) and bold uppercase letters for matrices (i.e.,

𝑨  ∈

ℝ

n  ×  m

𝑨

superscript  ℝ

𝑛  𝑚

\boldsymbol{A}\in\mathbb{R}^{n\times m}

). For a symmetric positive definite matrix

𝑮  ∈

ℝ

n  ×  n

𝑮

superscript  ℝ

𝑛  𝑛

\boldsymbol{G}\in\mathbb{R}^{n\times n}

and vector

𝒗  ∈

ℝ  n

𝒗

superscript  ℝ  𝑛

\boldsymbol{v}\in\mathbb{R}^{n}

, we let

‖  𝒗  ‖

𝑮

subscript

norm  𝒗

𝑮

\left\|\boldsymbol{v}\right\|_{\boldsymbol{G}}

denote the weighted Euclidean norm; i.e.,

‖  𝒗  ‖

𝑮

:=

𝒗  ⊤

​  𝑮  ​  𝒗

assign

subscript

norm  𝒗

𝑮

superscript  𝒗  top

𝑮  𝒗

\left\|\boldsymbol{v}\right\|_{\boldsymbol{G}}:=\sqrt{\boldsymbol{v}^{\top}\boldsymbol{G}\boldsymbol{v}}

.
Finally, we denote by

𝒩  ​

(

𝒗  0

,  𝑮  )

𝒩

subscript  𝒗  0

𝑮

\mathcal{N}(\boldsymbol{v}_{0},\boldsymbol{G})

the Gaussian (normal) distribution with mean

�