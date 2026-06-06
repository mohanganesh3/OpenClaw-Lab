[2505.18284] Tube Loss based Deep Networks For Improving the Probabilistic Forecasting of Wind Speed

Tube Loss based Deep Networks For Improving the Probabilistic Forecasting of Wind Speed

Pritam Anand, , Aadesh Minz, and Asish Joel

This work was funded by the Smart Energy Learning Center at DA-IICT, Gandhinagar through Grant no.- CSR-25/BSES/A6-PA/SELC.  Pritam Anand, Aadesh Minz and Asish Joel are with DA-IICT, Gandhinagar. (e-mail: pritam

_  \_

anand@daiict.ac.in)

Abstract

Uncertainty Quantification (UQ) in wind speed forecasting is a critical challenge in wind power production due to the inherently volatile nature of wind. By quantifying the associated risks and returns, UQ supports more effective decision-making for grid operations and participation in the electricity market. In this paper, we design a sequence of deep learning based probabilistic forecasting methods by using the Tube loss function for wind speed forecasting. The Tube loss function is a simple and model agnostic Prediction Interval (PI) estimation approach and can obtain the narrow PI with asymptotical coverage guarantees without any distribution assumption. Our deep probabilistic forecasting models effectively incorporate popular architectures such as LSTM, GRU, and TCN within the Tube loss framework. We further design a simple yet effective heuristic for tuning the

δ  \delta

parameter of the Tube loss function so that our deep forecasting models obtain the narrower PI without compromising its calibration ability. We have considered three wind datasets, containing the hourly recording of the wind speed, collected from three distinct location namely Jaisalmer, Los Angeles and San Fransico. Our numerical results demonstrate that the proposed deep forecasting models produce more reliable and narrower PIs compared to recently developed probabilistic wind forecasting methods.

Index Terms:

wind speed forecasting, uncertainty quantification, probabilistic forecasting, deep learning, quantile regression.

I

Introduction

I-A

Background

It is well known that fossil fuels cannot meet our future energy needs because they are rapidly depleting and pose significant environmental risks. Wind energy is one of the fastest-growing clean sources of renewable energy, gaining sufficient global attention due to its unlimited availability and minimal environmental impact.
In 2023, new global wind power capacity is going to surpass 100 GW for the first time  [ 1 ] . The Global Wind Energy Council (GWEC) estimates the wind power capacity will reach to 143 GW by the end of this decade [ 1 ] . However, the high volatility and uncertainty present significant challenges for maintaining the safe and stable operation of power systems, scheduling turbine maintenance, and integrating with the power grid  [ 2 ]  [ 3 ] . Therefore, the effective wind speed modelling and prediction plays a crucial role in overall wind power generation process.

There are plethora of methods and modeling techniques in the literature for wind speed forecasting using the historical wind data. But, due to highly random and chaotic nature of wind speed, their forecasting always involves a significant degree of uncertainty, which may lead to poor decisions in energy system management  [ 2 ] . Therefore, quantifying these uncertainties is crucial to mitigate the risks associated with various decisions related with wind energy management  [ 4 ]  such as power system reserve setting  [ 5 ]  , unit commitment  [ 6 ]  and market trading  [ 7 ] .

I-B

Overview of the previous work

Given the time-series data, the probabilistic forecasting models obtain a pair of functions such that the future observation would lie in the tube constructed by the estimated pair of functions with a given confidence

1  −  α

1-\alpha

. Ultimately, the estimated pair of functions represent two different conditional quantiles  [ 8 ]  of the predictive distribution, such that their difference corresponds to the given confidence. Probabilistic forecasting is essentially a problem of estimating Prediction Interval (PI) with time-series data with a given target calibration

1  −  α

1-\alpha

.

There are several methods for probabilistic forecasting in literature but, they can be easily divided into two category on the basis of their initial assumption. There are a number of models which apriory assumes the predictive distribution of the data and estimate the parameters of the assumed distribution efficiently for obtaining the probabilistic forecast. These models are known as  Parametric or Distribution based  models.

In  [ 9 ]  and  [ 10 ] , Khosravi et al. have used the Mean-Variance Estimation (MVE) method for probabilistic forecasting of wind power. They assume that the marginal distribution of target values follows Gaussian distribution and use a Neural Network (NN) to estimate the mean and variance of this distribution.
Afrasiabi et al. have used the mixture of Gaussian density for modeling the predictive distribution by stacking different deep neural architectures  [ 3 ] . In  [ 11 ]  and  [ 12 ] , authors realizes the need of modeling the predictive distribution with non-Gaussian distribution for wind speed forecasting.
However, the prior assumption of predictive distribution in these models are not sufficient to obtain the consistent performance across different variety of wind data.

In distribution-free setting, the wind probabilistic forecasting methods can be mainly divided into three major groups.

(1)

One group of wind models aims to estimate the overall predictive distribution for probabilistic forecasting. Many of these models, such as those by Bessa et al. (2012) and Khorramdel et al. (2018), use variants of kernel density estimation to estimate the wind density function. Probabilistic forecasts are then obtained by calculating the quantiles of the wind distribution function.

(2)

The second group of wind models directly estimates the pair of conditional quantile functions for obtaining probabilistic forecast by minimizing
the pinball loss function  [ 8 ] . These models have gained popularity among researchers for modeling wind power due to its simplicity and consistent performance. Some of the noteworthy variants are  [ 2 ]

[ 13 ]

[ 14 ]

[ 15 ]

[ 16 ]

[ 17 ]

[ 18 ]

[ 19 ] .
One of the notable limitation of this approach in the deep learning application is that it requires the training of pair of deep learning models one by one for targeting the estimation of pair of quantiles.

(3)

The third group of wind models minimize a loss function to simultaneously and directly estimate the pair of functions for probabilistic forecasting. As compared to quantile based deep forecasting models, they have two clear advantages. The first advantage is that these models require to be trained once and can directly estimate the both bound functions simultaneously. The second advantage is that they also allow the minimization of the width of the PI tube in their optimization problem as they are estimating the bound functions of the PI simultaneously. The Lower Upper Bound Estimation (LUBE)  [ 20 ]  is one of the first popular model which designs loss function for PI estimation task. Later, the LUBE method was used with wind time-series data for probabilistic forecasting tasks in several literature . Some important of them are  [ 21 ]

[ 22 ]

[ 23 ]

[ 24 ]

[ 25 ]

[ 26 ] . But, one of the major problem with the LUBE loss function is that it is a step like function and not differentiable, which poses the challenge in solving its optimization problem. This also limits its effective extension in deep learning framework, which generally require the gradient descent method based back-propagation for efficient network training. Taking motivation from this, Pearce et al., have proposed the Quality Driven (QD) loss function  [ 27 ]  for high quality PI estimation, which is similar to the loss function used in LUBE method. Though, QD loss fun