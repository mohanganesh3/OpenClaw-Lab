<!-- page 1 -->
BAYESIAN UNCERTAINTY QUANTIFICATION WITH ANCHORED
ENSEMBLES FOR ROBUST EV POWER CONSUMPTION
PREDICTION
Ghazal Farhani
Connected & Automated Vehicles, National Research Council Canada
London, Canada
ghazal.farhani@nrc-cnrc.gc.ca
Taufiq Rahman
Connected & Automated Vehicles, National Research Council Canada
London, Canada
Taufiq.Rahman@nrc-cnrc.gc.ca
Kieran Humphries
Transportation Emissions & Electrification Laboratory
Environment and Climate Change Canada
Kieran.Humphries@ec.gc.ca
November 11, 2025
ABSTRACT
Accurate EV power estimation underpins range prediction and energy management, yet practitioners
need both point accuracy and trustworthy uncertainty. We propose an anchored-ensemble Long
Short-Term Memory (LSTM) with a Student-t likelihood that jointly captures epistemic (model) and
aleatoric (data) uncertainty. Anchoring imposes a Gaussian weight prior (MAP training), yielding
posterior-like diversity without test-time sampling, while the t-head provides heavy-tailed robustness
and closed-form prediction intervals. Using vehicle-kinematic time series (e.g., speed, motor RPM),
our model attains strong accuracy—RMSE 3.36±1.10, MAE 2.21±0.89, R2 = 0.93±0.02, explained
variance 0.93±0.02—and delivers well-calibrated uncertainty bands with near-nominal coverage.
Against competitive baselines (Student-t MC dropout; quantile regression with/without anchoring),
our method matches or improves log-scores while producing sharper intervals at the same coverage.
Crucially for real-time deployment, inference is a single deterministic pass per ensemble member (or
a weight-averaged collapse), eliminating Monte Carlo latency. The result is a compact, theoretically
grounded estimator that couples accuracy, calibration, and systems efficiency—enabling reliable
range estimation and decision-making for production EV energy management.
1
Introduction
Road transportation is a major contributor to energy use and emissions globally. In the EU, it accounts for over 70% of
transport-sector greenhouse gases [1], while in Canada, transportation represents approximately 25% of total greenhouse
gas emissions, with road transport comprising the majority of this share [2]. Battery electric vehicles (BEVs) can
substantially reduce these emissions through zero tailpipe output, regenerative braking, and improved urban efficiency.
However, practical deployment is constrained by driving-range uncertainty, which depends on driver behavior, road
grade, surface conditions, weather, and battery characteristics (type/age). As transportation electrification accelerates
globally, reliable energy management systems with quantified uncertainty become essential for EV adoption and user
arXiv:2511.06538v1  [cs.LG]  9 Nov 2025


<!-- page 2 -->
A PREPRINT - NOVEMBER 11, 2025
confidence. Accurate, real-time (or near real-time) energy consumption and range estimation are therefore essential for
mission planning, eco-routing, and coordinated driving (e.g., platooning) [3–8].
Energy modeling approaches fall broadly into physics-based and data-driven categories. Physics-based models encode
vehicle dynamics from first principles [9–11], while data-driven models learn nonlinear mappings from kinematics to
power/energy directly from data [12,13]. Recent surveys report strong performance of machine learning for EV energy
prediction [14,15], but they also highlight the growing importance of uncertainty quantification (UQ) for deployment:
operators require not only point estimates but also calibrated confidence measures. For applications such as range
prediction in electric powertrains, especially in mission-critical contexts (e.g., ambulances or police vehicles), operators
must not only obtain accurate point predictions but also understand how confident the model is in those predictions.
UQ provides this essential capability by characterizing two key types of uncertainty: aleatoric, arising from inherent
variability in driving conditions, sensor noise, and exogenous factors, and epistemic, stemming from limited, biased, or
non-representative training data [16].
In prior work [17], we benchmarked Long Short-Term Memory (LSTM) networks against Temporal Convolutional
Networks (TCNs), Transformers, and Random Forest on EV telematics for power prediction and found that LSTMs
achieved comparable accuracy (RMSE/MAE) to the more complex architectures. To avoid re-opening architecture
comparisons in this paper, we focus exclusively on LSTMs and refer readers to [17] for the full benchmarking protocol
and results. This focus also facilitates our Bayesian treatment of weights and makes weight evolution and calibration
diagnostics more interpretable within a recurrent setting.
In this paper by using real-world telematics, we estimate instantaneous power consumption with an LSTM and perform
a comprehensive UQ analysis. We adopt a Bayesian ensembling method [18] to capture epistemic uncertainty and
extend its formulation to LSTM architectures; we additionally incorporate a probabilistic output layer to model aleatoric
uncertainty, yielding predictive intervals suitable for online range management. The resulting uncertainty bands track
error scales and provide actionable confidence bounds for battery management and eco-driving.
Contributions: (i) We develop an LSTM-based EV power estimator with principled uncertainty quantification
(UQ) that yields calibrated prediction intervals. (ii) We provide a mathematical extension of a Bayesian ensembling
scheme—anchored networks, originally proposed for feedforward models—to recurrent LSTM architectures, and derive
the corresponding training objective. (iii) We validate on real telematics data and report both accuracy and calibration
metrics. (iv) To isolate model-uncertainty effects, we benchmark epistemic UQ via anchored networks against MC
dropout [19]. (v) For aleatoric UQ, we demonstrate that a Student’s-t negative log-likelihood loss provides superior
calibration and accuracy compared to quantile regression.
Paper Organization: Section 2 reviews learning-based BEV power estimation and UQ, and identifies gaps in the
literature. Section 3 presents the Bayesian anchored-ensemble methodology, sketches its extension to LSTM networks,
introduces the aleatoric loss, and derives the complete training objective. Section 4.1 briefly describes the dataset;
since part of it is detailed elsewhere [17], we focus here on the chassis-dynamometer experiments, feature selection,
and network architecture. Section 5 reports results and compares (i) anchored ensembles with a Student’s-t negative
log-likelihood, (ii) anchored ensembles with quantile loss, (iii) MC dropout with a Student’s-t negative log-likelihood,
and (iv) MC dropout with quantile loss. We show that the Student’s-t loss yields better-calibrated intervals than quantile
loss, and that anchored ensembles achieve performance comparable to MC dropout while offering a more practical
choice for our setting. Section 7 concludes.
2
Related Work
This section reviews deep learning methods for battery electric vehicle (BEV) power/energy estimation and summarizes
prior efforts on uncertainty quantification (UQ) in this context.
Early data-driven studies employed feedforward neural networks (FNNs) for segment-level energy prediction. For
example, [20] combined multiple linear regression with a neural network using kinematic and road-context features (e.g.,
speed, acceleration, road condition), reporting mean absolute error (MAE) in the 12–14% range. Recurrent architectures,
notably Long Short-Term Memory (LSTM) networks, have since shown stronger performance by exploiting temporal
dependencies. Chen et al. [15] trained an LSTM on instantaneous speed, acceleration, and road grade, achieving
approximately 3% mean absolute percentage error (MAPE) for cumulative energy use.
Uncertainty quantification (UQ) is essential across many scientific and engineering disciplines, as it provides confidence
levels for model predictions. From optimal inverse estimation methods to neural networks, the importance of UQ
in numerical modeling has been well established [21–24]. Nevertheless, UQ applications in battery electric vehicle
(BEV) energy modeling remain sparse. One prevalent approach substitutes mean-squared error with quantile loss
to generate prediction intervals. For instance, Chen et al. [25] employed Quantile LSTM (QLSTM) to achieve
2


<!-- page 3 -->
A PREPRINT - NOVEMBER 11, 2025
80% and 90% coverage for BEV charging-demand forecasts, while Zhu et al. [26] applied quantile recurrent neural
networks (QRNN) to predict cumulative energy at the 90% confidence level. Quantile regression has similarly proven
popular in related energy forecasting tasks, such as wind power prediction [27]. Despite being model-agnostic
and computationally tractable, quantile-based methods predominantly capture aleatoric uncertainty (irreducible data
noise) while neglecting epistemic uncertainty (reducible model uncertainty arising from insufficient or unrepresentative
training data). Furthermore, these approaches do not produce complete predictive distributions, lack natural interpolation
between specified quantiles, and require computational effort that scales linearly with the number of quantiles estimated.
To capture epistemic uncertainty, Monte Carlo (MC) dropout has been widely adopted (e.g., [28–30]), following the
insight of [19] that applying dropout at inference approximates Bayesian model uncertainty.
Beyond BEV-specific literature, a substantial body of work addresses UQ more broadly in machine learning. Bayesian
theory (e.g., MacKay, 1992 [31]) provides a principled framework by defining posterior distributions over model
parameters, thereby embedding epistemic uncertainty directly in the model and enabling the use of informative priors.
In practice, however, full Bayesian inference for modern deep networks is computationally prohibitive; Markov chain
Monte Carlo (MCMC) [32], while accurate, is rarely tractable at scale.
Ensemble-based methods offer a practical alternative but are often heuristic. The Bayesian ensembling approach
of [18] provides a principled alternative grounded in Bayesian point estimation and has shown practical utility across
domains [22]. Originally developed for feedforward networks, we present a natural extension of its mathematical
formulation to LSTM architectures in this work.
3
Methodology: Bayesian Anchored Ensembles for LSTMs
In this section, we briefly describe the Bayesian anchored ensemble method, then state a proposition showing that the
anchoring idea extends naturally to LSTM models. Finally, we present a unified loss that jointly estimates aleatoric and
epistemic uncertainties within the power-estimation pipeline.
3.1
Bayesian Anchored Ensemble for Epistemic Uncertainty
We model epistemic uncertainty via a weight prior and MAP training. With parameters W and data D = {(xn, yn)}N
n=1,
the average negative log-posterior is
J (W) = 1
N
N
X
n=1
ℓ
 ˆy(xn; W), yn
 
|
{z
}
Ldata(W )
+ 1
2(W −µp)⊤Σ−1
p (W −µp)
|
{z
}
Lprior(W )
,
(1)
where ℓis the per-sample NLL (e.g., Student-t), and p(W) = N(µp, Σp). Anchored ensembles [18] draw anchors
W (m)
anc ∼N(µp, Σp) and train members by replacing µp with W (m)
anc
in Lprior; this yields posterior-like diversity without
test-time sampling.
3.2
Extension to LSTMs
Let the LSTM parameters be partitioned by gates/layers as W = L
g∈G Wg, G = {i, f, o, c, head}. Using a factorized
Gaussian prior p(W) = Q
g∈G N
 Wg; W (0)
g
, σ2
gI
 
, the MAP objective (1) becomes
J (W) = Ldata(W) +
X
g∈G
1
2σ2g
Wg −W (0)
g
2
2,
(2)
i.e., the data term depends on all gates via the recurrence, while the prior contributes independent quadratic penalties
per gate. In anchored ensembles, draw gate-wise anchors W (m)
g,anc and replace W (0)
g
by W (m)
g,anc in (2) for each member m.
Proposition (MAP with gate-wise Gaussian priors)
Under the sample-factorized likelihood and block-factorized
prior above, any MAP estimator minimizes (2). Proof: See Appendix 7.
3


<!-- page 4 -->
A PREPRINT - NOVEMBER 11, 2025
3.3
Aleatoric Uncertainty: Heteroscedastic Gaussian Negative Log-Likelihood
To learn aleatoric (data-dependent) noise, it is common to minimize the heteroscedastic Gaussian NLL [19], however,
when residuals are heavy-tailed, a Student’s t likelihood is more robust:
Lt = 1
N
N
X
i=1
h
log s(xi) + 1
2 log(νπ) + log Γ( ν
2) −log Γ( ν+1
2 )
+ ν+1
2
log
 
1 + (yi −ˆy(xi))2
ν s(xi)2
 i
,
(3)
where s(x) > 0 is the predicted scale and ν > 0 the degrees of freedom. As ν →∞, (3) approaches the Gaussian
distribution; if ν > 2, Var[Y | x] =
ν
ν−2 s(x)2.
3.4
Aleatoric Uncertainty and Combined Objective
For heavy-tailed residuals we use a Student-t head with location µ(x), scale s(x)>0, and dof ν. The per-sample t-NLL
(omitting constants) is ℓt
 y | µ, s, ν
 
= log s + ν+1
2
log
 1 + (y−µ)2
ν s2
 
. We train an anchored ensemble with the unified
t+prior objective (member m):
J (m) = 1
N
X
n
ℓt
 yn | µ(m)
n
, s(m)
n
, ν
 
+
X
g∈G
∥W (m)
g
−W (m)
g,anc∥2
2
2σ2g
.
(4)
Predictive intervals at level 1 −α use t1−α/2,ν s(m)(x) and ensemble averaging for epistemic spread.
3.5
Evaluation metrics
The performance of the time-series regression model is evaluated using two standard metrics: Mean Absolute Error
(MAE) and Root Mean Square Error (RMSE). The MAE is defined as
MAE = 1
N
N
X
i=1
|yi −ˆyi| ,
(5)
while the RMSE is given by
RMSE =
v
u
u
t 1
N
N
X
i=1
(yi −ˆyi)2.
(6)
In both expressions, yi denotes the ground truth and ˆyi the predicted value from the neural network model for the i-th
sample, and N is the total number of samples.
4
Data Collection and Model Architecture
This section outlines the experimental data collection, details the model architecture selection, and concludes with the
feature selection approach.
4.1
Data Collection
To probe model performance under both idealized and realistic operating conditions, we assembled two complementary
datasets. The first comprises four controlled chassis–dynamometer tests following the Highway Fuel Economy Test
(HWFET) cycle—two sedans, one hatchback, and one truck. Each HWFET run lasts 800 s and covers 16.45 km at an
average speed of 77.7 km/h. The laboratory setting minimizes exogenous variability (e.g., road friction changes, wind,
temperature gradients), providing a high–fidelity environment to learn the mapping between vehicle kinematics and
power consumption. From each vehicle we extracted signals such as velocity, acceleration, resistive force, driving
(tractive) force, and related measures. Although all four vehicles share the same velocity trajectory by design, their
instantaneous power profiles differ; consequently, for each vehicle we trained on 70% of the samples and evaluated on
the remaining 30%.
To assess external validity and examine how predictive uncertainty evolves in realistic highway driving, we also collected
a real–world dataset from a 2022 Hyundai IONIQ 5 on Ontario Highway 403 (London →Sarnia). Vehicle telematics
4


<!-- page 5 -->
A PREPRINT - NOVEMBER 11, 2025
Feature Set
RMSE
MAE
1
Set 1
2.4
1.7
2
Set 2
2.5
1.7
3
Set 3
8.1
6.6
4
Set 4
11.2
9.2
5
Set 5
5.5
3.3
Table 1: Ablation on feature subsets. S1: {DY_DriveRef, DY_Parasitic, DY_flt_force, Speed, Acceleration}; S2:
{DY_DriveRef, DY_Parasitic, DY_Roadld, DY_flt_force, Speed, Acceleration}; S3: {DY_DriveRef, DY_Parasitic,
DY_Roadld, DY_flt_force, Speed}; S4: {DY_DriveRef, DY_flt_force}; S5: {Acceleration}.
were captured using the “Car Scanner” smartphone application through the OBD-II port, recording parameters reported
by the vehicle. Unlike the dyno data, these measurements—and the acquisition method—are inherently noisier and
subject to uncontrolled factors, offering a more representative testbed of on–road conditions and commodity sensing.
This pairing of datasets—highly controlled vs. real–world—allows us to study both the learnability of the model in
ideal conditions and the growth (or drift) of estimation uncertainty under practical deployment.
4.2
Model Architecture Selection
The model architecture and training hyperparameters were selected through randomized trials and empirical evaluation.
Specifically, the number of LSTM layers was randomly chosen from {2, 4, 6, 8}, while the hidden dimension was
selected from {32, 64, 128}. The anchored variance for each LSTM gate was sampled from a uniform range of
[0.001, 0.1]. The number of training epochs was selected from {50, 100, 300, 500}, and the learning rate was randomly
sampled from the range [10−5, 10−2]. Optimization algorithms tested included Adam, stochastic gradient descent,
AdaGrad, and RMSprop.
To balance model performance with computational efficiency, the final configuration employed a four-layer LSTM
architecture with a hidden dimension of 32 and an anchored variance of 0.01. Since all input features were normalized to
the [0, 1] range, this choice of anchored variance was considered appropriate, ensuring that the learned weights remained
on a comparable scale. For ensemble training, 30 models with identical architecture were trained independently, each
for 300 epochs using a learning rate of 0.001 and the Adam optimization algorithm.
4.3
Feature Selection
Following the feature–selection process described in [17], we adopted the same set of features since the highway dataset
used in our work is identical to that study. While some prior works (e.g., [33,34]) included battery current as an input,
we followed the approach of [17] and relied solely on vehicle kinematics. This design choice ensures that the deep
learning model learns the intrinsic relationship between vehicle telemetry and battery power consumption without
depending on electrical measurements. Accordingly, the core features are vehicle velocity, acceleration, and motor
torque.
In the chassis–dynamometer experiments, multiple signals were recorded; however, consistent with our decision to
exclude electrical variables (current/voltage), the candidate input set was limited to: longitudinal speed (Speed),
smoothed longitudinal acceleration (Acceleration), tractive (drive) force at the wheels/axle (DY_flt_force),
road–load resistance (aerodynamic, rolling, and grade) (DY_Roadld), parasitic/drivetrain and accessory losses
(DY_Parasitic), and driver/ECU drive request (DY_DriveRef). Randomized feature–subset evaluations were con-
ducted on more than 15 different configurations; representative results are summarized in Table 1.
As shown in the table, the feature combination in the first row achieved the best performance, with RMSE = 2.4 and
MAE = 1.7. Acceleration emerged as a critical predictor: excluding it caused RMSE and MAE to rise sharply (8.1
and 6.6, respectively). To further test this, we trained a model using only acceleration as input (Row 5). Although
acceleration alone contributed substantially, the error metrics increased markedly, confirming that the additional features
in Row 1 collectively yield the lowest RMSE and MAE.
5


**[Table p5.1]**
| Feature Set |  | RMSE | MAE |
| --- | --- | --- | --- |
| 1 | Set 1 | 2.4 | 1.7 |
| 2 | Set 2 | 2.5 | 1.7 |
| 3 | Set 3 | 8.1 | 6.6 |
| 4 | Set 4 | 11.2 | 9.2 |
| 5 | Set 5 | 5.5 | 3.3 |

[CAPTION] Table 1: Ablation on feature subsets. S1: {DY_DriveRef, DY_Parasitic, DY_flt_force, Speed, Acceleration}; S2:


<!-- page 6 -->
A PREPRINT - NOVEMBER 11, 2025
Figure 1: Estimated power consumption for the sedan (solid dark red) and the truck (solid violet); ground truth shown as
dashed orange and dashed blue, respectively. Confidence intervals are depicted by the shaded yellow and green bands.
5
Result
5.1
The Anchored LSTM Model
The anchored LSTM was implemented in Python with Keras and trained on an NVIDIA T4 GPU. After training,
we evaluated the model on a held-out test set. Table II reports RMSE, MAE, R2, and explained variance for four
vehicles using the chassis dynamometer experiments and the IONIQ 5 measurements. For visualization, Fig. 1 shows
two representative cases—a sedan and a truck. Dashed lines denote ground truth, solid lines the anchored model’s
predictions, and the shaded bands the combined (aleatoric + epistemic) uncertainty intervals. Although the chassis
dynamometer velocity profiles are identical, differences across fleets lead to distinct power demands; accordingly, each
model is trained for a fleet-specific pattern rather than a single shared mapping.
5.2
Comparative Study
Given the widespread use of quantile regression for aleatoric uncertainty [15, 20] and MC dropout for epistemic
uncertainty [19], we include three comparison baselines: (i) Quantile–Anchor: quantile loss with our anchored
ensemble; (ii) Quantile–Dropout: quantile loss with MC dropout; and (iii) t–NLL Dropout: Student’s-t negative
log-likelihood with MC dropout. These variants allow a systematic assessment of aleatoric modeling (quantiles vs.
t-likelihood) and epistemic modeling (anchoring vs. dropout), alongside our primary t–NLL Anchor model.
We report RMSE, MAE, R2, and explained variance for four vehicles using chassis-dynamometer experiments and
highway tests by ION5 in Table II. For visualization, Fig. 2 shows Sedan results. The left panel compares t–NLL
Anchor (dark red) and t–NLL Dropout (violet), with their confidence intervals (yellow and light blue bands). Consistent
with Table II, both achieve low MAE/RMSE and high R2 and explained variance, with similar aleatoric and epistemic
uncertainty. The right panel presents Quantile–Anchor (dark blue) and Quantile–Dropout (light blue) with corresponding
bands (light pink and light green); although errors remain low, intervals are noticeably wider.
To evaluate calibration, we apply the binomial proportion confidence-interval test [35] at α = 0.1 (90% target). Table 3
reports empirical coverage with its CI bounds, average band width, and a standardized variance metric (ideal = 1;
> 1 indicates under-dispersion/narrow bands; < 1 over-dispersion/wide bands). Moreover, t–NLL Anchor attains
coverage within the bounds with standardized variance near 1 (well calibrated); t–NLL Dropout is slightly conservative
(standardized variance < 1 and coverage near the upper bound); both quantile variants are overly conservative
(standardized variance ≪1 with substantially wider bands).
Figure 3 summarizes the IONIQ 5 highway experiments. The left panel compares ground truth with the t–NLL Anchor
and t–NLL Dropout predictions (with their confidence bands), while the right panel shows the Quantile–Anchor and
Quantile–Dropout counterparts. Because these data were collected on open highways via a mobile app (as opposed to
6

[CAPTION] Figure 1: Estimated power consumption for the sedan (solid dark red) and the truck (solid violet); ground truth shown as

[CAPTION] Figure 3 summarizes the IONIQ 5 highway experiments. The left panel compares ground truth with the t–NLL Anchor


<!-- page 7 -->
A PREPRINT - NOVEMBER 11, 2025
Figure 2: Left Panel: t–NLL Anchor (dark red) and t–NLL Dropout (violet), with their confidence intervals (yellow and
light blue bands) Right Panel: Quantile–Anchor (dark blue) and Quantile–Dropout (light blue) with corresponding
bands (light pink and light green)
Figure 3: Left Panel: t–NLL Anchor (dark red) and t–NLL Dropout (violet), with their confidence intervals (yellow and
light blue bands) Right Panel: Quantile–Anchor (dark blue) and Quantile–Dropout (light blue) with corresponding
bands (light pink and light green)
controlled chassis–dynamometer tests; see [17]), RMSE/MAE are slightly higher overall. Even so, the t–NLL models
maintain low errors and good calibration (Table 3). In contrast, Quantile–Dropout exhibits noticeably larger errors and
misses fine-scale structure, whereas Quantile–Anchor fits more details but produces overly wide intervals (e.g., ∼95%
bands with standardized variance ≈0.10), indicating under-confidence. Calibration metrics further favor anchoring:
t–NLL Anchor attains standardized variance near unity (≈1.13), while t–NLL Dropout is more under-dispersed
(≈1.45; narrower-than-ideal bands), making the anchored approach the more suitable choice in this setting.
6
Discussion
6.1
Theoretical Discussion
We presented an anchored LSTM framework for EV power estimation that delivers both point predictions and calibrated
uncertainty, jointly capturing epistemic (model) and aleatoric (data) uncertainty components. Beyond empirical
performance gains, our contribution provides a clean, end-to-end probabilistic formulation with explicit priors and
distribution-aware uncertainty quantification. Specifically:
• Anchoring for LSTM networks: We extended the Bayesian anchoring method to LSTM architectures. This
approach yields a posterior-approximating ensemble by sampling anchors w(m)
0
and retraining, establishing
an explicit connection between ensemble diversity and epistemic uncertainty.
• Heavy-tailed likelihood with analytical intervals: By employing the Student’s-t distribution as a loss
function, the predictive interval at confidence level 1 −α has an analytical form: I1−α(x) = µ(x) ± t1−α/2,ν.
This formulation provides provable robustness to outliers compared to Gaussian negative log-likelihood while
maintaining likelihood-based calibration.
• Explicit uncertainty separation: The anchored ensemble quantifies epistemic uncertainty through between-
member variance, while the learned scale parameter within the loss function captures aleatoric noise.
7

[CAPTION] Figure 2: Left Panel: t–NLL Anchor (dark red) and t–NLL Dropout (violet), with their confidence intervals (yellow and

[CAPTION] Figure 3: Left Panel: t–NLL Anchor (dark red) and t–NLL Dropout (violet), with their confidence intervals (yellow and


<!-- page 8 -->
A PREPRINT - NOVEMBER 11, 2025
NLL Anchor
Quantile Drop Out
Vehicle
RMSE
MAE
AU
EU
R2
EV
Vehicle
RMSE
MAE
AU
EU
R2
EV
Sedan_1
2.5
1.3
1.6
1.2
0.93
0.94
Sedan_1
2.5
1.6
5.5
6.5
0.93
0.94
Sedan_2
2.3
1.7
1.3
1.35
0.90
0.94
Sedan_2
2.7
1.7
6.5
7.8
0.89
0.93
Truck
3.5
2.0
1.5
1.4
0.95
0.95
Truck
5.7
4.3
6.9
7.9
0.87
0.87
Hatchback
3.1
2.1
0.95
1.45
0.89
0.88
Hatchback
4.5
3.4
3.7
4.7
0.76
0.78
IONIQ 5
5.4
3.9
7.0
3.5
0.98
0.91
IONIQ 5
16.2
9.6
5.2
1.0
0.08
0.13
NLL Drop Out
Quantile Anchor
Vehicle
RMSE
MAE
AU
EU
R2
EV
Vehicle
RMSE
MAE
AU
EU
R2
EV
Sedan_1
2.5
1.7
1.7
1.5
0.93
0.94
Sedan_1
3.9
2.5
1.5
10.2
0.83
0.85
Sedan_2
2.7
2.1
1.9
1.8
0.90
0.90
Sedan_2
2.8
1.8
5.5
11.1
0.85
0.85
Truck
5.0
3.2
2.2
1.8
0.89
0.90
Truck
5.7
3.5
1.8
16.0
0.85
0.85
Hatchback
2.7
1.8
1.3
1.4
0.91
0.91
Hatchback
3.2
2.3
0.90
8.7
0.87
0.87
IONIQ 5
7.7
3.3
4.8
1.8
0.78
0.79
IONIQ 5
8.6
3.7
3.3
28
0.73
0.75
Table 2: Comprehensive results across five vehicle classes—two sedans, one pickup truck, one hatchback, and a
Hyundai IONIQ 5—comparing four uncertainty heads: t–NLL Anchor, t–NLL Dropout, Quantile Anchor, and Quantile
MC Dropout.
Sedan
Model
Convergence
Low
High
Width
StVar
NLL Anchor
0.898
0.889
0.907
5.03
1.13
NLL Dropout
0.911
0.902
0.919
7.69
0.948
Quantile Anchor
0.966
0.960
0.971
32.32
0.36
Quantile Dropout
1.000
0.999
1.000
27.59
0.06
Model
IONIQ 5
NLL Anchor
0.919
0.885
0.944
12.59
1.13
NLL Dropout
0.949
0.920
0.968
18.48
1.45
Quantile Anchor
0.991
0.974
0.996
95.53
0.11
Quantile Dropout
0.327
0.279
0.379
17.89
9.15
Table 3: Confidence-interval analysis. ‘Low’ and ‘High’ are the lower and upper bounds of the binomial coverage band;
‘Width’ is the mean interval width; ‘StVar’ is the variance of standardized residuals.
This framework yields a model that is (i) principled—implementing MAP estimation with a proper heavy-tailed
likelihood and exact t t-distribution quantile intervals; (ii) diagnosable—enabling calibration verification against known
quantiles and binomial coverage scores; and (iii) efficient—eliminating test-time sampling requirements inherent in MC
dropout approaches.
6.2
Practical Implications and Model Performance
In this study, we demonstrated that the NLL-anchored model effectively captures variability in battery power estimation.
To our knowledge, this represents the first work to separately quantify both aleatoric and epistemic uncertainties while
developing a mathematically rigorous probabilistic approach for epistemic uncertainty quantification. Our key findings
indicate that the proposed model provides well-calibrated confidence intervals. As expected, real-world highway data
exhibited slightly increased RMSE and MAE values compared to chassis dynamometer experiments; however, the
results remain strong. Through binomial proportion tests, we have verified that the uncertainty bands are well-calibrated,
indicating high confidence in the model’s predictions.
The existing literature on EV battery energy estimation predominantly employs quantile loss for capturing aleatoric
noise [15,20]. We used quantile regression as a benchmark against our proposed Student’s-t distribution loss function.
Notably, for the noisier IONIQ 5 highway data, quantile regression performed substantially worse. The model became
overly conservative, producing excessively wide confidence bands that were nearly meaningless, with standardized
variance values around 0.1. In contrast, the Student’s-t loss function demonstrated significantly better performance
across all datasets. Even in controlled chassis dynamometer experiments, quantile regression yielded inferior results.
We attribute this performance difference to several factors: empirically, quantile-based variants produced systematically
wider intervals despite comparable RMSE/MAE. This behavior is consistent with (1) a more challenging optimization
landscape compared to MSE-like objectives, (2) multi-quantile training (τ ∈0.1, 0.5, 0.9) creating a complex multi-task
8


**[Table p8.1]**
| NLL Anchor |  |  |  |  |  |  | Quantile Drop Out |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vehicle | RMSE | MAE | AU | EU | R2 | EV | Vehicle | RMSE | MAE | AU | EU | R2 | EV |
| Sedan 1 _ | 2.5 | 1.3 | 1.6 | 1.2 | 0.93 | 0.94 | Sedan 1 _ | 2.5 | 1.6 | 5.5 | 6.5 | 0.93 | 0.94 |
| Sedan 2 _ | 2.3 | 1.7 | 1.3 | 1.35 | 0.90 | 0.94 | Sedan 2 _ | 2.7 | 1.7 | 6.5 | 7.8 | 0.89 | 0.93 |
| Truck | 3.5 | 2.0 | 1.5 | 1.4 | 0.95 | 0.95 | Truck | 5.7 | 4.3 | 6.9 | 7.9 | 0.87 | 0.87 |
| Hatchback | 3.1 | 2.1 | 0.95 | 1.45 | 0.89 | 0.88 | Hatchback | 4.5 | 3.4 | 3.7 | 4.7 | 0.76 | 0.78 |
| IONIQ 5 | 5.4 | 3.9 | 7.0 | 3.5 | 0.98 | 0.91 | IONIQ 5 | 16.2 | 9.6 | 5.2 | 1.0 | 0.08 | 0.13 |
| NLL Drop Out |  |  |  |  |  |  | Quantile Anchor |  |  |  |  |  |  |
| Vehicle | RMSE | MAE | AU | EU | R2 | EV | Vehicle | RMSE | MAE | AU | EU | R2 | EV |
| Sedan 1 _ | 2.5 | 1.7 | 1.7 | 1.5 | 0.93 | 0.94 | Sedan 1 _ | 3.9 | 2.5 | 1.5 | 10.2 | 0.83 | 0.85 |
| Sedan 2 _ | 2.7 | 2.1 | 1.9 | 1.8 | 0.90 | 0.90 | Sedan 2 _ | 2.8 | 1.8 | 5.5 | 11.1 | 0.85 | 0.85 |
| Truck | 5.0 | 3.2 | 2.2 | 1.8 | 0.89 | 0.90 | Truck | 5.7 | 3.5 | 1.8 | 16.0 | 0.85 | 0.85 |
| Hatchback | 2.7 | 1.8 | 1.3 | 1.4 | 0.91 | 0.91 | Hatchback | 3.2 | 2.3 | 0.90 | 8.7 | 0.87 | 0.87 |
| IONIQ 5 | 7.7 | 3.3 | 4.8 | 1.8 | 0.78 | 0.79 | IONIQ 5 | 8.6 | 3.7 | 3.3 | 28 | 0.73 | 0.75 |


**[Table p8.2]**
| Model | Sedan |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  | Convergence | Low | High | Width | StVar |
| NLL Anchor | 0.898 | 0.889 | 0.907 | 5.03 | 1.13 |
| NLL Dropout | 0.911 | 0.902 | 0.919 | 7.69 | 0.948 |
| Quantile Anchor | 0.966 | 0.960 | 0.971 | 32.32 | 0.36 |
| Quantile Dropout | 1.000 | 0.999 | 1.000 | 27.59 | 0.06 |
| Model | IONIQ 5 |  |  |  |  |
| NLL Anchor | 0.919 | 0.885 | 0.944 | 12.59 | 1.13 |
| NLL Dropout | 0.949 | 0.920 | 0.968 | 18.48 | 1.45 |
| Quantile Anchor | 0.991 | 0.974 | 0.996 | 95.53 | 0.11 |
| Quantile Dropout | 0.327 | 0.279 | 0.379 | 17.89 | 9.15 |

[CAPTION] Table 2: Comprehensive results across five vehicle classes—two sedans, one pickup truck, one hatchback, and a

[CAPTION] Table 3: Confidence-interval analysis. ‘Low’ and ‘High’ are the lower and upper bounds of the binomial coverage band;


<!-- page 9 -->
A PREPRINT - NOVEMBER 11, 2025
learning problem, and (3) the nonparametric nature of quantile regression, which lacks distributional priors and tends
toward conservative coverage at the expense of precision.
Compared to MC dropout, the anchored LSTM achieves similar accuracy and calibration while offering significant
practical advantages: inference does not require stochastic sampling during test time. This makes the anchored approach
particularly suitable for real-time energy management applications with strict latency constraints. Furthermore, the
ensemble is trained once and evaluated with a single forward pass per member (or aggregated via weight averaging),
enabling straightforward scaling to larger datasets and multi-vehicle scenarios without incurring test-time Monte Carlo
overhead.
Overall, our results demonstrate that the anchored LSTM with a Student’s-t output head represents a practical and
reliable alternative to both MC dropout and quantile methods for uncertainty-aware EV power prediction, combining
strong accuracy, competitive calibration, and deployment-ready efficiency.
While this study serves as a proof-of-concept for learning calibrated epistemic and aleatoric uncertainties using both
chassis dynamometer and highway driving data, a limitation remains in the dataset size for developing a fully generalized
model. Expanding to larger, more diverse datasets represents an important direction for future work to enhance model
generalization across varying driving conditions and vehicle types.
7
Conclusion
In this study, we addressed the critical challenge of range anxiety in electric vehicles by developing an accurate power
consumption estimation framework based on vehicle kinematics. Reliable power prediction is essential for optimizing
battery charging schedules, developing energy-efficient driving strategies, and enabling effective real-time energy
management and route planning. For these applications to be trustworthy, power consumption predictions must be
accompanied by well-calibrated confidence intervals.
To this end, we implemented an anchored Bayesian ensemble approach. While originally developed for standard
feedforward neural networks, we demonstrated that this methodology extends naturally to LSTM architectures, providing
a principled framework for uncertainty quantification in sequential data modeling.
Our model achieved excellent performance across all five vehicle fleets (combining chassis dynamometer and highway
data), with an average RMSE of 3.36 ± 1.1, MAE of 2.21 ± 0.89, R2 of 0.93 ± 0.02, and explained variance of
0.93 ± 0.02. More importantly, we demonstrated that the model provides meaningful, well-calibrated uncertainty bands
that reliably capture both epistemic and aleatoric uncertainties.
In comprehensive benchmarking against alternative approaches, our method performed comparably to or better than
existing techniques. While achieving similar accuracy to MC dropout, our anchored ensemble offers significant practical
advantages—particularly for real-time applications—by eliminating the need for stochastic sampling during inference.
This makes our approach more suitable for deployment in latency-sensitive energy management systems.
The framework presented in this work provides a solid foundation for uncertainty-aware EV power prediction, balancing
theoretical rigor with practical deployability to help alleviate range anxiety through reliable energy forecasting.
ACKNOWLEDGMENT
This work was co-funded by Transport Canada’s ecoTECHNOLOGY for Vehicles program and the National Research
Council Canada’s Clean and Energy Efficient Transportation program. The views and opinions of the authors expressed
herein do not necessarily state or reflect those of Transport Canada.
References
[1] European Environment Agency, “Transport and environment report 2021: Decarbonising road transport—
the role of vehicles, fuels and transport demand,” Copenhagen, Denmark, 2022. [Online]. Available:
https://www.eea.europa.eu/publications/transport-and-environment-report-2021
[2] Environment and Climate Change Canada, “National inventory report 1990–2021: Greenhouse gas sources and
sinks in canada,” Environment and Climate Change Canada, Gatineau, QC, Canada, Tech. Rep., 2023, part 1.
[Online]. Available: https://publications.gc.ca/site/eng/9.506002/publication.html
9


<!-- page 10 -->
A PREPRINT - NOVEMBER 11, 2025
[3] H. Shen, X. Zhou, H. Ahn, M. Lamantia, P. Chen, and J. Wang, “Personalized velocity and energy prediction for
electric vehicles with road features in consideration,” IEEE Transactions on Transportation Electrification, vol. 9,
no. 3, pp. 3958–3969, 2023.
[4] A. K. Madhusudhanan, X. Na, and D. Cebon, “A computationally efficient framework for modelling energy
consumption of ice and electric vehicles,” Energies, vol. 14, no. 7, p. 2031, 2021.
[5] G. Wager, J. Whale, and T. Braunl, “Driving electric vehicles at highway speeds: The effect of higher driving
speeds on energy consumption and driving range for electric vehicles in australia,” Renewable and sustainable
energy reviews, vol. 63, pp. 158–165, 2016.
[6] Y. Zhang, X. Qu, and L. Tong, “Optimal eco-driving control of autonomous and electric trucks in adaptation to
highway topography: Energy minimization and battery life extension,” IEEE Transactions on Transportation
Electrification, vol. 8, no. 2, pp. 2149–2163, 2022.
[7] J. Zhang, T.-Q. Tang, Y. Yan, and X. Qu, “Eco-driving control for connected and automated electric vehicles at
signalized intersections with wireless charging,” Applied Energy, vol. 282, p. 116215, 2021.
[8] O. Barhoumi, G. Farhani, T. Rahman, M. H. Zaki, S. Tahar, and F. Araji, “Fuel consumption in platoons: A
literature review,” arXiv preprint arXiv:2508.10891, 2025.
[9] X. Qu, Y. Liu, Y. Chen, and Y. Bie, “Urban electric bus operation management: Review and outlook,” J. Automob.
Saf. Energy Sav, vol. 13, pp. 407–420, 2022.
[10] C. Fiori, K. Ahn, and H. A. Rakha, “Power-based electric vehicle energy consumption model: Model development
and validation,” Applied Energy, vol. 168, pp. 257–268, 2016.
[11] X. Yuan, C. Zhang, G. Hong, X. Huang, and L. Li, “Method for evaluating the real-world driving energy
consumptions of electric vehicles,” Energy, vol. 141, pp. 1955–1968, 2017.
[12] Y. Pan, W. Fang, and W. Zhang, “Development of an energy consumption prediction model for battery electric
vehicles in real-world driving: A combined approach of short-trip segment division and deep learning,” Journal of
Cleaner Production, vol. 400, p. 136742, 2023.
[13] R. Maia, M. Silva, R. Araújo, and U. Nunes, “Electrical vehicle modeling: A fuzzy logic model for regenerative
braking,” Expert systems with applications, vol. 42, no. 22, pp. 8504–8519, 2015.
[14] X. Zhang, Z. Zhang, Y. Liu, Z. Xu, and X. Qu, “A review of machine learning approaches for electric vehicle
energy consumption modelling in urban transportation,” Renewable Energy, vol. 234, p. 121243, 2024.
[15] Y. Chen, Y. Zhang, and R. Sun, “Data-driven estimation of energy consumption for electric bus under real-world
driving conditions,” Transportation Research Part D: Transport and Environment, vol. 98, p. 102969, 2021.
[16] M. Valdenegro-Toro and D. S. Mori, “A deeper look into aleatoric and epistemic uncertainty disentanglement,” in
2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW).
IEEE, 2022,
pp. 1508–1516.
[17] R. Yahyaabadi, G. Farhani, T. Rahman, S. Nikan, A. Jirjees, and F. Araji, “Deep learning-based analysis of power
consumption in gasoline, electric, and hybrid vehicles,” arXiv preprint arXiv:2508.08034, 2025.
[18] T. Pearce, F. Leibfried, and A. Brintrup, “Uncertainty in neural networks: Approximately bayesian ensembling,”
in International conference on artificial intelligence and statistics.
PMLR, 2020, pp. 234–244.
[19] Y. Gal and Z. Ghahramani, “Dropout as a bayesian approximation: Representing model uncertainty in deep
learning,” in international conference on machine learning.
PMLR, 2016, pp. 1050–1059.
[20] C. De Cauwer, W. Verbeke, T. Coosemans, S. Faid, and J. Van Mierlo, “A data-driven method for energy
consumption prediction and energy-efficient routing of electric vehicles in real-world conditions,” Energies,
vol. 10, no. 5, p. 608, 2017.
[21] G. Farhani, R. J. Sica, S. Godin-Beekmann, and A. Haefele, “Optimal estimation method retrievals of stratospheric
ozone profiles from a dial,” Atmospheric Measurement Techniques, vol. 12, no. 4, pp. 2097–2111, 2019.
[22] G. Farhani, G. Martucci, T. Roberts, A. Haefele, and R. J. Sica, “A bayesian neural network approach for
tropospheric temperature retrievals from a lidar instrument,” International Journal of Remote Sensing, vol. 44,
no. 5, pp. 1611–1627, 2023.
[23] T. J. Sullivan, Introduction to uncertainty quantification.
Springer, 2015, vol. 63.
[24] M. Abdar, F. Pourpanah, S. Hussain, D. Rezazadegan, L. Liu, M. Ghavamzadeh, P. Fieguth, X. Cao, A. Khosravi,
U. R. Acharya et al., “A review of uncertainty quantification in deep learning: Techniques, applications and
challenges,” Information fusion, vol. 76, pp. 243–297, 2021.
10


<!-- page 11 -->
A PREPRINT - NOVEMBER 11, 2025
[25] Y. Chen, B. Pang, X. Xiang, T. Lu, T. Xia, and G. Geng, “Probabilistic forecasting of electric vehicle charging
load using composite quantile regression lstm,” in 2023 IEEE/IAS Industrial and Commercial Power System Asia
(I&CPS Asia).
IEEE, 2023, pp. 984–989.
[26] Q. Zhu, Y. Huang, C. F. Lee, P. Liu, J. Zhang, and T. Wik, “Predicting electric vehicle energy consumption from
field data using machine learning,” IEEE Transactions on Transportation Electrification, 2024.
[27] A. Faustine and L. Pereira, “Fpseq2q: Fully parameterized sequence to quantile regression for net-load forecasting
with uncertainty estimates,” IEEE Transactions on Smart Grid, vol. 13, no. 3, pp. 2440–2451, 2022.
[28] K. K. Yalamanchi, S. Kommalapati, P. Pal, N. Kuzhagaliyeva, A. S. AlRamadan, B. Mohan, Y. Pei, S. M.
Sarathy, E. Cenker, and J. Badra, “Uncertainty quantification of a deep learning fuel property prediction model,”
Applications in Energy and Combustion Science, vol. 16, p. 100211, 2023.
[29] N. Hashemipour, J. Aghaei, P. C. Del Granado, A. Kavousi-Fard, T. Niknam, M. Shafie-khah, and J. P. Catalão,
“Uncertainty modeling for participation of electric vehicles in collaborative energy consumption,” IEEE Transac-
tions on Vehicular Technology, vol. 71, no. 10, pp. 10 293–10 302, 2022.
[30] H. Wang and D.-Y. Yeung, “A survey on bayesian deep learning,” ACM computing surveys (csur), vol. 53, no. 5,
pp. 1–37, 2020.
[31] D. J. MacKay, “A practical bayesian framework for backpropagation networks,” Neural computation, vol. 4, no. 3,
pp. 448–472, 1992.
[32] D. Van Ravenzwaaij, P. Cassey, and S. D. Brown, “A simple introduction to markov chain monte–carlo sampling,”
Psychonomic bulletin & review, vol. 25, no. 1, pp. 143–154, 2018.
[33] M. N. Nabi, B. Ray, F. Rashid, W. Al Hussam, and S. Muyeen, “Parametric analysis and prediction of energy
consumption of electric vehicles using machine learning,” Journal of Energy Storage, vol. 72, p. 108226, 2023.
[34] W. Achariyaviriya, W. Wongsapai, K. Janpoom, T. Katongtung, Y. Mona, N. Tippayawong, and P. Suttakul,
“Estimating energy consumption of battery electric vehicles using vehicle sensor data and machine learning
approaches,” Energies, vol. 16, no. 17, p. 6351, 2023.
[35] S. E. Vollset, “Confidence intervals for a binomial proportion,” Statistics in medicine, vol. 12, no. 9, pp. 809–824,
1993.
11


<!-- page 12 -->
A PREPRINT - NOVEMBER 11, 2025
Appendix
We begin by specifying the LSTM architecture in Section 7. Subsequently, Section 7 generalizes the anchored MAP
framework to this model and outlines a sketch of the proof.
LSTM
For sequential data, LSTM networks are effective due to their ability to retain temporal dependencies, which is useful
for energy estimation from time-series inputs (e.g., velocity, torque, acceleration). An LSTM cell comprises input,
forget, and output gates that regulate information flow. At time t, given input xt and previous hidden state ht−1,
it = σ
 Wxixt + bi + Whiht−1 + bhi
 
,
ft = σ
 Wxfxt + bf + Whfht−1 + bhf
 
,
ot = σ
 Wxoxt + bo + Whoht−1 + bho
 
,
˜ct = tanh
 Wxcxt + bc + Whcht−1 + bhc
 
,
ct = ft ⊙ct−1 + it ⊙˜ct,
ht = ot ⊙tanh(ct),
where σ(·) is the logistic function; W{·} and b{·} are gate-specific weights and biases; and ⊙denotes elementwise
multiplication. In our EV power estimation context, xt represents vehicle operational parameters and the target y is
instantaneous power.
Building on the partition W = L
g∈G Wg defined above, we place independent Gaussian priors (or anchors) on each
block Wg and optimize MAP.
Extension of Anchor model into LSTM architecture
Proposition (MAP with gate-wise Gaussian priors)
Let D = {(xn, yn)}N
n=1 and let ˆy(·; W) denote the LSTM
predictor with
W =
M
g∈G
Wg,
G = {input i, forget f, output o, candidate c, head}.
Assume the likelihood factorizes over samples, p(D | W) = QN
n=1 p(yn | xn, W), and the prior factorizes over blocks,
p(W) =
Y
g∈G
N
 Wg; W (0)
g
, σ2
pI
 
.
Define the empirical data term
Ldata(W) = 1
N
N
X
n=1
ℓ
 ˆy(xn; W), yn
 
,
with ℓ(ˆy, y) = −log p(y | ˆy).
Then any MAP estimator minimizes
J (W) = Ldata(W) +
1
2σ2p
X
g∈G
Wg −W (0)
g
2
2
i.e., the loss is a function of all gates plus a Gaussian quadratic penalty applied separately to each block.
Proof.
The average negative log-posterior is
−1
N log p(W | D) = 1
N
N
X
n=1
 
−log p(yn | xn, W)
 
−1
N log p(W) + C,
and ℓ(ˆy(xn; W), yn) = −log p(yn | xn, W) gives the data term. Using −log N(w; µ, σ2I) =
1
2σ2 ∥w −µ∥2
2 + C and
prior independence,
−log p(W) =
X
g∈G
1
2σ2p ∥Wg −W (0)
g
∥2
2 + C′,
so minimizing −log p(W | D) is equivalent (up to constants) to minimizing J (W).
12