[2511.06538] Bayesian Uncertainty Quantification with Anchored Ensembles for Robust EV Power Consumption Prediction

Bayesian Uncertainty Quantification with Anchored Ensembles for Robust EV Power Consumption Prediction

Ghazal Farhani

Connected &amp; Automated Vehicles, National Research Council Canada

London, Canada

ghazal.farhani@nrc-cnrc.gc.ca

&amp;Taufiq Rahman

Connected &amp; Automated Vehicles, National Research Council Canada

London, Canada

Taufiq.Rahman@nrc-cnrc.gc.ca

&amp;Kieran Humphries

Transportation Emissions &amp; Electrification Laboratory
 Environment and Climate Change Canada

Kieran.Humphries@ec.gc.ca

Abstract

Accurate EV power estimation underpins range prediction and energy management, yet practitioners need  both  point accuracy and trustworthy uncertainty. We propose an anchored-ensemble Long Short-Term Memory (LSTM) with a Student-

t  t

likelihood that jointly captures epistemic (model) and aleatoric (data) uncertainty. Anchoring imposes a Gaussian weight prior (MAP training), yielding posterior-like diversity without test-time sampling, while the

t  t

-head provides heavy-tailed robustness and closed-form prediction intervals. Using vehicle-kinematic time series (e.g., speed, motor RPM), our model attains strong accuracy—RMSE

3.36  ±  1.10

3.36\!\pm\!1.10

, MAE

2.21  ±  0.89

2.21\!\pm\!0.89

,

R  2

=

0.93  ±  0.02

R^{2}=0.93\!\pm\!0.02

, explained variance

0.93  ±  0.02

0.93\!\pm\!0.02

—and delivers well-calibrated uncertainty bands with near-nominal coverage. Against competitive baselines (Student-

t  t

MC dropout; quantile regression with/without anchoring), our method matches or improves log-scores while producing  sharper  intervals at the same coverage. Crucially for real-time deployment, inference is a single deterministic pass per ensemble member (or a weight-averaged collapse), eliminating Monte Carlo latency. The result is a compact, theoretically grounded estimator that couples accuracy, calibration, and systems efficiency—enabling reliable range estimation and decision-making for production EV energy management.

1  Introduction

Road transportation is a major contributor to energy use and emissions globally. In the EU, it accounts for over 70% of transport-sector greenhouse gases  [ european2021transport ] , while in Canada, transportation represents approximately 25% of total greenhouse gas emissions, with road transport comprising the majority of this share  [ eccc2023emissions ] . Battery electric vehicles (BEVs) can substantially reduce these emissions through zero tailpipe output, regenerative braking, and improved urban efficiency. However, practical deployment is constrained by driving-range uncertainty, which depends on driver behavior, road grade,
surface conditions, weather, and battery characteristics (type/age). As transportation electrification accelerates globally, reliable energy management systems with quantified uncertainty become essential for EV adoption and user confidence. Accurate, real-time (or near real-time)
energy consumption and range estimation are therefore essential for mission planning, eco-routing, and coordinated driving (e.g., platooning)
 [ shen2023personalized ,  madhusudhanan2021computationally ,  wager2016driving ,  zhang2022optimal ,  zhang2021eco ,  barhoumi2025fuel ] .

Energy modeling approaches fall broadly into physics-based and data-driven categories. Physics-based models encode vehicle dynamics from first principles  [ qu2022urban ,  fiori2016power ,  yuan2017method ] , while data-driven models learn nonlinear mappings from kinematics to power/energy directly from data  [ pan2023development ,  maia2015electrical ] . Recent surveys report strong performance of machine learning for EV energy prediction  [ zhang2024review ,  chen2021data ] , but they also highlight the growing importance of uncertainty quantification (UQ) for deployment: operators require not only point estimates but also calibrated confidence measures. For applications such as range prediction in electric powertrains, especially in mission-critical contexts (e.g., ambulances or police vehicles), operators must not only obtain accurate point predictions but also understand how confident the model is in those predictions. UQ provides this essential capability by characterizing two key types of uncertainty:  aleatoric , arising from inherent variability in driving conditions, sensor noise, and exogenous factors, and  epistemic , stemming from limited, biased, or non-representative training data  [ valdenegro2022deeper ] .

In prior work  [ yahyaabadi2025deep ] , we benchmarked Long Short-Term Memory (LSTM) networks against Temporal Convolutional Networks (TCNs), Transformers, and Random Forest on EV telematics for power prediction and found that LSTMs achieved comparable accuracy (RMSE/MAE) to the more complex architectures. To avoid re-opening architecture comparisons in this paper, we focus exclusively on LSTMs and refer readers to  [ yahyaabadi2025deep ]  for the full benchmarking protocol and results. This focus also facilitates our Bayesian treatment of weights and makes weight evolution and calibration diagnostics more interpretable within a recurrent setting.

In this paper by using real-world telematics, we estimate instantaneous power consumption with an LSTM and perform a comprehensive UQ analysis. We adopt a Bayesian ensembling method  [ pearce2020uncertainty ]  to capture  epistemic  uncertainty and extend its formulation to LSTM architectures; we additionally incorporate a probabilistic output layer to model  aleatoric  uncertainty, yielding predictive intervals suitable for online range management. The resulting uncertainty bands track error scales and provide actionable confidence bounds for battery management and eco-driving.

Contributions:  (i) We develop an LSTM-based EV power estimator with principled uncertainty quantification (UQ) that yields calibrated prediction intervals. (ii) We provide a mathematical extension of a Bayesian ensembling scheme— anchored networks , originally proposed for feedforward models—to recurrent LSTM architectures, and derive the corresponding training objective. (iii) We validate on real telematics data and report both accuracy and calibration metrics. (iv) To isolate model-uncertainty effects, we benchmark epistemic UQ via anchored networks against MC dropout  [ gal2016dropout ] . (v) For aleatoric UQ, we demonstrate that a Student’s-

t  t

negative log-likelihood loss provides superior calibration and accuracy compared to quantile regression.

Paper Organization: 
Section

2

reviews learning-based BEV power estimation and UQ, and identifies gaps in the literature. Section

3

presents the Bayesian anchored-ensemble methodology, sketches its extension to LSTM networks, introduces the aleatoric loss, and derives the complete training objective. Section

4.1

briefly describes the dataset; since part of it is detailed elsewhere  [ yahyaabadi2025deep ] , we focus here on the chassis-dynamometer experiments, feature selection, and network architecture. Section

5

reports results and compares (i) anchored ensembles with a Student’s-

t  t

negative log-likelihood, (ii) anchored ensembles with quantile loss, (iii) MC dropout with a Student’s-

t  t

negative log-likelihood, and (iv) MC dropout with quantile loss. We show that the Student’s-

t  t

loss yields better-calibrated intervals than quantile loss, and that anchored ensembles achieve performance comparable to MC dropout while offering a more practical choice for our setting. Section

7

concludes.

2  Related Work

This section reviews deep learning methods for battery electric vehicle (BEV) power/energy estimation and summarizes prior efforts on uncertainty quantification (UQ) in this context.

Early data-driven studies employed feedforward neural networks (FNNs) for segment-level energy prediction. For example,  [ de2017data ]  combined multiple linear regression wit