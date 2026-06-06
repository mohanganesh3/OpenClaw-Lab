<!-- page 1 -->
An Analysis of Temporal Dropout in Earth
Observation Time Series for Regression Tasks
Miro Miranda1,2[0009−0002−8195−9776], Francisco Mena1,2[0000−0002−5004−6571],
and Andreas Dengel1,2[0000−0002−6100−8255]
1 University of Kaiserslautern-Landau, Kaiserslautern, Germany
2 German Research Center for Artificial Intelligence, Kaiserslautern, Germany
{miro.miranda_lorenz,francisco.mena,andreas.dengel}@dfki.de
Abstract. Missing instances in time series data impose a significant
challenge to deep learning models, particularly in regression tasks. In
the Earth Observation field, satellite failure or cloud occlusion frequently
results in missing time-steps, introducing uncertainties in the predicted
output and causing a decline in predictive performance. While many
studies address missing time-steps through data augmentation to im-
prove model robustness, the uncertainty arising at the input level is com-
monly overlooked. To address this gap, we introduce Monte Carlo Tem-
poral Dropout (MC-TD), a method that explicitly accounts for input-
level uncertainty by randomly dropping time-steps during inference using
a predefined dropout ratio, thereby simulating the effect of missing data.
To bypass the need for costly searches for the optimal dropout ratio,
we extend this approach with Monte Carlo Concrete Temporal Dropout
(MC-ConcTD), a method that learns the optimal dropout distribution
directly. Both MC-TD and MC-ConcTD are applied during inference,
leveraging Monte Carlo sampling for uncertainty quantification. Exper-
iments on three EO time-series datasets demonstrate that MC-ConcTD
improves predictive performance and uncertainty calibration compared
to existing approaches. Additionally, we highlight the advantages of adap-
tive dropout tuning over manual selection, making uncertainty quantifi-
cation more robust and accessible for EO applications.
Keywords: Dropout · Earth Observation · Time Series · Regression ·
Uncertainty Quantification
1
Introduction
Recently, Deep Learning (DL) models have been widely used in the Earth Obser-
vation (EO) field to find optimal data-driven solutions in different applications
[3]. DL effectively processes complex and heterogeneous sensor data, allowing for
accurate analysis of environmental patterns [27]. For instance, predicting con-
tinuous values such as crop yield [38], water content in plants [39], and surface
forecast [40] involves processing complex data. In the EO field, processing time
series sensor data is essential for understanding the changes and dynamics of our
planet [29]. However, sensors may experience anomalies and occlusions, leading
arXiv:2504.06915v1  [cs.LG]  9 Apr 2025


<!-- page 2 -->
2
M. Miranda et al.
to missing data over certain time-steps [43,15,11]. For instance, clouds obstruct
the sunlight in optical images, as on average, 67% of Earth’s surface is covered
by clouds [22], often resulting in uncertain predicted outputs. Addressing miss-
ing data in time series is a prevalent challenge in the DL field, with various
modeling solutions and pre-processing techniques developed to mitigate its im-
pact and ensure accurate predictions. While various techniques exist to mitigate
missing data [15,26], few explicitly quantify the uncertainty it introduces, par-
ticularly the impact of input-level uncertainty. Addressing this gap is crucial for
improving model reliability in EO regression applications.
In this work, we analyze two models that simulate missing data across time
series during training and inference for a dual purpose. During training, it acts as
an augmentation technique to increase model generalization to missing data in
temporal EO data. During inference, by generating multiple Monte Carlo (MC)
samples with different missing patterns, it acts as an Uncertainty Quantifica-
tion (UQ) mechanism, thereby increasing the trustworthiness of a prediction.
The first model is built on the dropout variational distribution applied dur-
ing inference [13] and over time, referred to as Monte Carlo Temporal Dropout
(MC-TD). However, the optimal dropout value is a difficult hyperparameter to
tune, requiring a resource-intensive process that depends on each dataset and
missing data type [26]. Instead of manually searching, we propose using Concrete
Dropout (ConcD) [14] to learn the optimal dropout ratio using standard gradient
descent. We refer to this method as Monte Carlo Concrete Temporal Dropout
(MC-ConcTD), where ConcD is applied across time and during inference. The
ConcD [14] approximates a Bernoulli distribution using a continuous relaxation,
enabling the dropout probability to be optimized through gradient-based learn-
ing [24]. Unlike traditional dropout-based approaches, this model automatically
learns an optimal dropout distribution, eliminating costly hyperparameter tun-
ing.
We validate MC-TD and MC-ConcTD on regression tasks spanning three EO
datasets with temporal sensor data. Experimental results demonstrate that MC-
ConcTD improves both predictive accuracy and uncertainty calibration, while
also eliminating the need for manual dropout tuning. The Concrete Temporal
Dropout (ConcTD) not only enhances predictive performance but also improves
the accessibility and practicality of UQ in EO regression applications. The code is
publicly available at https://github.com/mmiranda-l/Temporal-Dropout/.
2
Related Work
Missing data in time series. Irregular sampled time series are common in signal
processing [36]. Temporal observations can easily suffer from problems in their
collection, causing irregular observations with missing data. Numerous research
in DL has focused on learning to impute time series, such as BRITS [4], mTAN
[44], and SAITS [10], while others focused on adapting models to ignore the miss-
ing data, such as D-GRU [5] and MissFormer [1]. In the EO field, missing spatial
and temporal data are a common phenomenon due to real-world data collection


<!-- page 3 -->
Temporal Dropout for EO Time Series Data
3
constraints [43]. This includes, sensor noise, sensor failure, and cloud occlusion,
affecting observations and introducing uncertainty. This problem negatively af-
fects the performance of predictive models, where more missing data translates
to worse predictions [20,12]. However, some strategies in the EO field mitigate
the negative effect of missing data, such as including features from different
sensors or using dropout techniques [35,15]. Recently, the Temporal Dropout
(TD) technique, which involves randomly dropping time steps, has been used
to enhance the model performance [15,47,18]. Furthermore, some studies lever-
age missing data as an augmentation technique to enhance generalization. For
instance, Weitland et al. [50] randomly mask the end of a time-series for gener-
alization to an early crop classification. On the other hand, some studies have
focused on reconstruction tasks that recover the missing time-steps. For instance,
Chen et al. [6] use a polynomial fit based on the Savitzky–Golay filter, while oth-
ers use DL models based on Multi-layer Perceptron (MLP) [8], or convolutions
[42].
Regression with time series data. Regression tasks have been lesser explored
than classification tasks with time-series data [32]. One reason might be that
DL models are not as effective for regression as for classification tasks, as shown
by Tan et al. [46]. The intrinsic challenge of predicting a continuous value in-
stead of a categorical value is often overlooked. Nevertheless, in the EO field,
there are numerous applications involving pixel-level regression with time series
data. For instance, Nguyen et al. [33] use multi-spectral data and weather time
series for pixel-wise crop yield prediction. They use a MLP model for processing
the time series data as individual features. However, it has been shown recently
that Recurrent Neural Network (RNN) models obtain better results. This is un-
derlined by [37,30]. Furthermore, Maimaitijiang et al. [25] consider the use of
drone-based optical time series data for the same task, obtaining images unaf-
fected by cloud occlusion. Another pixel-wise task studied in the literature is
cloud removal. This task has been explored with multi-spectral optical time se-
ries data [41], as well as including radar time series [11]. Similarly, in the Earth
surface forecast, multi-spectral optical and weather time series have been used
with spatio-temporal models to obtain accurate predictions [40], such as using
ConvLSTM models [9].
Uncertainty estimation. Uncertainty estimation holds particular promise in safety-
critical applications, including the EO field [16]. Reliable uncertainty estimates
improve decision-making by identifying predictions with high confidence and
those that should be treated with caution [21,19]. Uncertainty in predictive
models is typically classified into two categories: epistemic and aleatoric [21,19].
While the former captures the uncertainty coming from the model itself, the lat-
ter represents the uncertainty coming from the data exclusively, such as sensor
noise. In regression, the aleatoric uncertainty is often directly learned from the
data using proper scoring rules such as the Gaussian Negative Log-Likelihood
(NLL) loss [34]. In contrast, epistemic uncertainty is typically estimated by mod-
eling a distribution over the weight space, often based on Bayes’ Theorem. This


<!-- page 4 -->
4
M. Miranda et al.
includes approximate Bayesian inference [2], or dropout techniques [13,14,31]. In
the EO field, numerous studies have assessed the effectiveness of UQ methods,
particularly in applications such as crop yield prediction with time series data,
as evidenced in [23]. However, most of these studies overlook the uncertainty
arising at the input level, which can be crucial in improving model reliability
and robustness. In our work, we explore the simulation of missing data in EO
time series for regression tasks. To the best of our knowledge, there is no method
in the EO literature that accounts for the uncertainty arising from the input time
series using missing time steps.
3
Temporal Dropout for Uncertainty Estimation
3.1
Notation & Preliminaries
Let x ∈X and y ∈Y denote samples from the input and target spaces, respec-
tively. In the case of multivariate time series data with T time-steps, x is given
as x = (x0, ..., xT ). We further denote data pairs as (x, y), and define the goal
to predict the target y from x. Thus, we aim to find a model f θ(·) : X 7−→Y
such that f θ(x) = ˆy, by optimizing over the model’s parameters θ.
Uncertainty estimation: We define the problem of learning the uncertainty in the
predicted output (ˆy = {µ, σ2}) propagated through the uncertainty in the input
and the model itself. To estimate the epistemic uncertainty (EU), we model
the expected output and variance over multiple stochastic forward passes, by
placing a distribution over the network. Given independent input samples x(l)
drawn from the input distribution, EU is approximated as follows:
E[f θ(x)] = µEU ≈L−1 X
l
f θ(x(l))
V ar[f θ(x)] = σ2
EU ≈(L −1)−1 X
l
(f θ(x(l)) −E[f θ(x)])2.
To estimate the aleatoric uncertainty (AU), which arises from the data, we as-
sume a Normal distribution parameterized by two output heads: µ(x), σ2(x) =
f θ(x). Both terms can be optimized by minimizing the NLL [34]:
LNLL(y, x) = log σ2(x) + (µ(x) −y)2
σ2(x)
.
(1)
This approach captures AU in σ2(x), allowing the model to capture data re-
lated uncertainty. In Valdenegro-Toro et al. [49] a relationship between input
and output uncertainty is given, showing that input uncertainty is propagated
through the model, finally resulting in model, or epistemic uncertainty (EU) by
leveraging stochastic MC sampling. We follow this argumentation. Finally, the
predictive uncertainty (PU) is the combined effect of all uncertainties.


<!-- page 5 -->
Temporal Dropout for EO Time Series Data
5
3.2
Temporal Dropout
We leverage TD to estimate uncertainty. The dropout technique [45] is commonly
used in hidden layers of DL models during training. Nevertheless, applying this
technique at the input-level across time has shown good regularization perfor-
mance in the EO field [28,15,47,26,18]. This prevents the model from focusing
on individual time-steps and simultaneously handling missing time-steps, which
randomly occur in EO data with sensor failure and cloud coverage. Thus, the
multivariate time series data x is masked out as
ˆx = x ⊙(1 −p),
(2)
where p ∈[0, 1]T is the binary dropout mask, which is drawn from a Bernoulli
distribution, i.e. pi ∼Bern(α), ∀i = 1, ..., T, with α as the dropout ratio. Thus,
the input x becomes a random variable ˆx, providing additional network regu-
larization. In contrast, we estimate the uncertainty arising from missing time
steps, by enabling TD at the inference level. Gal et al. [13] demonstrated that
casting dropout approximates Bayesian inference in any neural network architec-
ture. Similarly, we extend this formulation along the temporal dimension, named
Monte Carlo Temporal Dropout (MC-TD). This corresponds to sampling
L different dropout masks for prediction, p(l)
i
∼Bern(α), ∀l = 1, . . . , L, and
applying them to the input data as ˆx(l) = x ⊙(1 −p(l)).
3.3
Concrete Temporal Dropout
When using the TD technique, an optimal dropout ratio value (α) has to be
identified to get a well-calibrated model. This is a computationally expensive
and time-consuming process, especially when working with over-parametrized
models, commonly used in the EO field and time series modeling. In Gal et al.
[14] a continuous relaxation of the dropout technique is proposed, allowing the
dropout ratio to be a learnable parameter in combination with standard gradient
descent. The soft dropout mask ˜p is defined as:
˜p = CS(α, u, τ) = σ
  
(log
α
(1 −α) + log
u
(1 −u)
  1
τ
 
,
(3)
with σ the sigmoid function, τ ∈R1 a temperature value controlling the smooth-
ness of the approximation, α ∈R1 a learnable parameter in the model, and u
a uniform random variable, ui ∼Uni(0, 1), ∀i = 1, ..., T. The latter acts as
the auxiliary variable in the sampling reparametrisation [24]. We use this sam-
pling strategy in the dropout across time, named Concrete Temporal Dropout
(ConcTD). Furthermore, to get the uncertainty, we use this technique at in-
ference time, referred to as Monte Carlo Concrete Temporal Dropout
(MC-ConcTD). This corresponds to obtaining L samples of the Uniform dis-
tribution, u(l)
i
∼Uni(0, 1), ∀l = 1, . . . , L, and forward over the model with data
as ˆx(l) = x ⊙(1 −CS(α, u(l), τ)). This technique is illustrated in Figure 1.
Additionally, in Figure 2 we schematically illustrate TD and ConcTD.


<!-- page 6 -->
6
M. Miranda et al.
Mean prediction
Aleatoric uncertainty
Epistemic uncertainty
ConcTD
ML 
Model
Input 
data
Ground truth
Loss 
function
Two-value output
Fig. 1: Illustration of a DL model prediction with the ConcTD technique and
three MC samples indexed by i ∈{1, 2, 3}.
Temporal Dropout (TD)
Concrete TD (ConcTD)
time series data 
binary 
dropping vector 
masked time series data 
user-defined (fixed)
 learnable parameter
(continuous) binary 
dropping vector 
masked time series data 
[
]
Fig. 2: Illustration of TD and ConcTD techniques over time series data.
4
Experiments
4.1
Datasets
We use three publicly available EO datasets with temporal sensor data. The
characteristics of these datasets are presented in Table 1.
Swiss Crop Yield (SwissYield) We use a dataset for crop yield estimation [38],
focused on cereals. This is a regression task in which the crop yield is estimated in
tons per hectare (t/ha) from sensor data. The data was collected in Switzerland
between 2017 and 2021. The temporal features are multi-spectral optical sensor
(from Sentinel-2, with 10 bands) and weather data (with 5 bands). The time
series are from seeding to harvesting, with an approximate 5 days of sampling
rate. All features were spatially interpolated to a pixel resolution of 10 m.
Live Fuel Moisture Content (LFMC) We use a dataset for moisture content
estimation [39]. This involves a regression task in which the vegetation water
(moisture) per dry biomass (in percentage) in a given location is predicted. The
data was collected between 2015 and 2019 in the western US. The temporal
features are multi-spectral optical sensor (from Landsat-8, with 8 bands) and
radar sensor (from Sentinel-1, with 3 bands). These features were re-sampled
monthly during a four-month window before the moisture measurement, i.e.
a signal of 4 time-steps is used. Additional static sensors are the topographic


**[Table p6.1]**
|  | ConcTD MM odL e l fuL no cts is o n Mean prediction Input data Aleatoric uncertainty Two-value output Ground truth Epistemic uncertainty |
| --- | --- |
|  |  |


**[Table p6.2]**
| user | -defined (fixed) |
| --- | --- |


**[Table p6.3]**
|  | learnable parameter |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  |  | [ |  | ] |  |  |

[CAPTION] Fig. 1: Illustration of a DL model prediction with the ConcTD technique and

[CAPTION] Fig. 2: Illustration of TD and ConcTD techniques over time series data.


<!-- page 7 -->
Temporal Dropout for EO Time Series Data
7
Dataset
Samples
Series
length
Features
Avg.
target
Std.
target
SwissYield
54,098
16-55
15
7.356
2.001
LFMC
2,578
4 (fixed)
61
103.987
39.562
PM2.5
167,309
120 (fixed)
9
73.673
68.546
Table 1: Datasets description and statistics.
information, soil properties, canopy height, and land-cover class. All features
were interpolated to a pixel resolution of 250 m.
Particles Matter 2.5 (PM2.5) We use a dataset for PM2.5 estimation [7]. This
involves a regression task in which the concentration of PM2.5 (particles measure
2.5 microns or fewer in diameter) in the air (in µg/m3) in a particular city is
predicted. The data was collected between 2010 and 2015 in five Chinese cities.
The temporal features are atmospheric conditions (with 3 bands), atmospheric
dynamics (with 4 bands), and precipitation (with 2 bands). These are captured
at hourly resolution, where we consider a three-day window for prediction.
4.2
Setup
We use an early fusion strategy that concatenates all features along the time
steps. For the architecture, we adopt a RNN model with 2 layers based on
long-short term memory units to extract the temporal information. Then, one
additional layer is used to map the last hidden state of the RNN into a hidden
dimension. Finally, two prediction heads are employed, reflecting the mean and
variance, respectively. All layers consist of 128 units, including 20% of dropout.
In addition, we use batch normalization layers after the RNN model. We train the
models for 100 epochs with an early stopping criteria based on a patience of five.
The optimization is carried out with the ADAM optimizer and a batch size of
128 over the Mean Squared Error (MSE) function. For MC sampling, 20 samples
are used as in [13]. We first evaluate MC-TD and MC-ConcTD against each
other and then against standard UQ methods for time series regression tasks,
including MC-Dropout [13] and a Variatioinal Inference (VI) [2]. For comparison,
all models employ the same architecture. If not differently specified, a TD ratio
of 0.3 is used for the MC-TD model, thereby aiming for a balance between
regularization and model capacity.
4.3
Results
We compare the performance of MC-ConcTD and MC-TD models across all
datasets in Table 2. Both approaches demonstrate strong performance across all
three datasets. However, MC-ConcTD consistently outperforms MC-TD on all
regression metrics for every dataset. For instance, a maximum improvement of
6 percentage points (p.p) is shown in R2 on the PM2.5 dataset.

[CAPTION] Table 1: Datasets description and statistics.


<!-- page 8 -->
8
M. Miranda et al.
In Figure 3 (a), we evaluate the performance of various dropout ratios in the
MC-TD model. The results indicate that a lower dropout ratio consistently im-
proves performance across all datasets, with a clear decline in model performance
as the ratio increases. Additionally, we observe an increase in PU with increas-
ing TD ratio and decreasing performance. In Figure 3 (b) the calibration error
(ECE) for every TD ratio is illustrated, together with the constant ECE for
the MC-ConcTD model. Marker sizes indicate normalized PU. Similarly, we ob-
serve an increase in calibration error with increased uncertainty, except for the
SwissYield dataset, and decreased performance. Ultimately, in Figure 3 (c), the
learned dropout ratios of the MC-ConcTD model across all folds are illustrated.
Notably, the dropout ratios remain below 0.25 with significant instability across
folds and datasets. For instance, approaching zero for the PM2.5 dataset. The
results underline the difficulty of manually tuning the dropout ratio.
In Figure 4, we illustrate the model calibration. We define calibration as the
models’ capabilities to accurately reflect the true probabilities of observed out-
comes. For instance, for an 80% confidence level, an event should occur approx-
imately 80% of the time. Ideally, a perfectly calibrated model aligns with the
diagonal line. In the SwissYield and LFMC datasets, the models are overconfi-
dent, though the MC-ConcTD model demonstrates relatively better calibration
overall. Conversely, for the PM2.5 dataset, the models exhibit better calibration,
especially for the MC-ConcTD model. In Figure 5, prediction-versus-target plots
Dataset
Model
R2(↑) RMSE(↓)
MAE(↓)
ECE(↓)
PU (↓)
SwissYield MC-TD
0.76
0.99
0.72
2.14
1.00
MC-ConcTD
0.79
0.93
0.66
2.23
1.23
LFMC
MC-TD
0.69
21.92
15.67
3.77
1.01
MC-ConcTD
0.72
20.80
14.76
3.41
8.01
PM2.5
MC-TD
0.71
36.36
24.56
2.92
16.00
MC-ConcTD
0.77
31.77
21.61
2.15
5.80
Table 2: Results of the two variants of dropout across time. The best results are
highlighted in bold. The MC-TD model uses a dropout ratio of 0.3.
0.0
0.1
0.2
0.3
0.4
0.5
Temporal Dropout Ratio
0.625
0.650
0.675
0.700
0.725
0.750
0.775
R2
SwissYield
LFMC
PM2.5
MC-TD
MC-ConcTD
(a) R2 values
0.0
0.1
0.2
0.3
0.4
0.5
Temporal Dropout Ratio
1.5
2.0
2.5
3.0
3.5
4.0
ECE
SwissYield
LFMC
PM2.5
MC-TD
MC-ConcTD
(b) ECE
00
01
02
03
04
05
06
07
08
09
K-fold
0.00
0.05
0.10
0.15
0.20
0.25
Dropout Ratio
SwissYield
LFMC
PM25_P3
(c) TD ratio with ConcTD
Fig. 3: Model performance in R2 and ECE under different TD settings, where
marker size indicates the normalized PU.


**[Table p8.1]**
| Dataset Model | R2(↑) RMSE(↓) MAE(↓) ECE(↓) PU (↓) |
| --- | --- |
| MC-TD SwissYield MC-ConcTD | 0.76 0.99 0.72 2.14 1.00 0.79 0.93 0.66 2.23 1.23 |
| MC-TD LFMC MC-ConcTD | 0.69 21.92 15.67 3.77 1.01 0.72 20.80 14.76 3.41 8.01 |
| MC-TD PM2.5 MC-ConcTD | 0.71 36.36 24.56 2.92 16.00 0.77 31.77 21.61 2.15 5.80 |


**[Table p8.2]**
|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
| SwissYi LFMC PM2.5 | eld |  |  |  |  |  |
| MC-TD MC-Con | cTD |  |  |  |  |  |


**[Table p8.3]**
|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  | SwissYield LFMC PM2.5 |
|  |  |  |  |  |  | MC-TD MC-ConcTD |


**[Table p8.4]**
|  |  |  |  |  |  |  |  | Sw LF PM | issYield MC 25_P3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |

[CAPTION] Table 2: Results of the two variants of dropout across time. The best results are

[CAPTION] Fig. 3: Model performance in R2 and ECE under different TD settings, where


<!-- page 9 -->
Temporal Dropout for EO Time Series Data
9
0.0
0.2
0.4
0.6
0.8
1.0
Mean Predicted Probability
0.0
0.2
0.4
0.6
0.8
1.0
Mean True Probability
MC-TD
MC-ConcTD
MC-Dropout
Ideal Calibration
(a) SwissYield
0.0
0.2
0.4
0.6
0.8
1.0
Mean Predicted Probability
0.0
0.2
0.4
0.6
0.8
1.0
Mean True Probability
MC-TD
MC-ConcTD
MC-Dropout
Ideal Calibration
(b) LFMC
0.0
0.2
0.4
0.6
0.8
1.0
Mean Predicted Probability
0.0
0.2
0.4
0.6
0.8
1.0
Mean True Probability
MC-TD
MC-ConcTD
MC-Dropout
Ideal Calibration
(c) PM2.5
Fig. 4: Model calibration across confidence levels for all datasets.
0
2
4
6
8
10
12
14
Ground truth
0
2
4
6
8
10
12
Prediction
Prediction vs ground truth 
50
100
150
200
250
300
Ground truth
50
100
150
200
250
300
Prediction
Prediction vs ground truth 
0
200
400
600
800
1000
Ground truth
0
500
1000
1500
2000
2500
3000
Prediction
Prediction vs ground truth 
(a) MC-Dropout
0
2
4
6
8
10
12
14
Ground truth
0
2
4
6
8
10
12
Prediction
Prediction vs ground truth 
50
100
150
200
250
300
Ground truth
50
100
150
200
250
300
Prediction
Prediction vs ground truth 
0
200
400
600
800
1000
Ground truth
0
500
1000
1500
2000
2500
Prediction
Prediction vs ground truth 
(b) MC-TD
0
2
4
6
8
10
12
14
Ground truth
0
2
4
6
8
10
12
Prediction
Prediction vs ground truth 
50
100
150
200
250
300
Ground truth
50
100
150
200
250
300
Prediction
Prediction vs ground truth 
0
200
400
600
800
1000
Ground truth
0
250
500
750
1000
1250
1500
Prediction
Prediction vs ground truth 
(c) MC-ConcTD
PM2.5
LFMC
SwissYield
Fig. 5: Prediction over target plots for three regression datasets. The marker size
indicates the uncertainty in each prediction.
are shown for MC-Dropout, the MC-TD, and MC-ConcTD models on the three
datasets. To analyze uncertainty estimates, the predictions are scaled based on
the PU. Overall, the plots show a good alignment between predicted and tar-
get values. However, the distinct characteristics of the datasets become evident,
with the PM2.5 dataset exhibiting very high uncertainties and the SwissYield
dataset showing consistently low uncertainties. Furthermore, we observe that
higher uncertainties are associated with lower alignment between predicted and
target values, particularly for the MC-ConcTD model.
Finally, we compare the introduced models against two established UQ meth-
ods, namely MC-Dropout and VI. The results are summarized in Table 3. No-
tably, the VI performs poorly on regression metrics. The MC-ConcTD model


**[Table p9.1]**
| 1.0 MC-TD MC-ConcTD MC-Dropout 0.8 Ideal Calibration Probability 0.6 True 0.4 Mean 0.2 0.0 0.0 0.2 0.4 0.6 0.8 1.0 Mean Predicted Probability | 1.0 MC-TD MC-ConcTD MC-Dropout 0.8 Ideal Calibration Probability 0.6 True 0.4 Mean 0.2 0.0 0.0 0.2 0.4 0.6 0.8 1.0 Mean Predicted Probability |
| --- | --- |


**[Table p9.2]**
| Prediction vs ground truth 12 10 8 Prediction 6 4 2 0 0 2 4 Gr6ound tr8uth 10 12 14 | Prediction vs ground truth 12 10 8 Prediction 6 4 2 0 0 2 4 Gr6ound tr8uth 10 12 14 | Prediction vs ground truth 12 10 8 Prediction 6 4 2 0 0 2 4 Gr6ound tr8uth 10 12 14 |
| --- | --- | --- |
| Prediction vs ground truth 300 250 200 Prediction 150 100 50 50 100 Gr1o5u0nd tru2t0h0 250 300 | Prediction vs ground truth 300 250 200 Prediction 150 100 50 50 100 Gr1o5u0nd tru2t0h0 250 300 | Prediction vs ground truth 300 250 200 Prediction 150 100 50 50 100 Gr1o5u0nd tru2t0h0 250 300 |
| 3000 Prediction vs ground truth 2500 2000 Prediction 1500 1000 500 0 0 200 4G0r0ound tru6t0h0 800 1000 | Prediction vs ground truth 2500 2000 1500 Prediction 1000 500 0 0 200 4G0r0ound tru6t0h0 800 1000 | Prediction vs ground truth 1500 1250 1000 Prediction 750 500 250 0 0 200 4G0r0ound tru6t0h0 800 1000 |

[CAPTION] Fig. 4: Model calibration across confidence levels for all datasets.

[CAPTION] Fig. 5: Prediction over target plots for three regression datasets. The marker size


<!-- page 10 -->
10
M. Miranda et al.
Dataset
Model
R2 (↑) RMSE (↓)
MAE (↓)
ECE (↓)
PU (↓)
SwissYield
MC-Dropout
0.78
0.94
0.67
2.21
1.02
MC-TD
0.76
0.99
0.72
2.14
1.00
MC-ConcTD
0.79
0.93
0.66
2.23
1.23
VI
0.51
2.02
1.57
0.73
23.57
LFMC
MC-Dropout
0.72
20.69
14.64
2.43
1.11
MC-TD
0.69
21.92
15.67
3.77
1.01
MC-ConcTD
0.72
20.80
14.76
3.41
8.01
VI
0.53
27.00
20.00
4.48
4.01
PM2.5
MC-Dropout
0.74
33.89
22.45
1.17
3.99
MC-TD
0.71
36.36
24.56
2.92
16.00
MC-ConcTD
0.77
31.77
21.61
2.15
5.80
VI
0.30
57.00
39.00
3.57
2.49
Table 3: Performance comparison for various UQ models.
achieves the best R2 scores on the SwissYield and PM2.5 datasets, while be-
ing equal to the MC-Dropout on the LFMC dataset. Considering all evaluation
metrics, we find that the MC-ConcTD has the best overall results.
5
Discussion
We empirically prove that TD at the inference level improves model performance
over common UQ methods on various EO regression tasks (Table 2 and Table 3),
thereby confirming related studies [18]. However, calibrating the dropout ratio
in any DL model is challenging as it requires exploring a continuous hyperpa-
rameter space. The same applies to the MC-TD method. Consequently, MC-TD
can only find the optimal dropout ratio at high computational search costs.
Interestingly, higher dropout ratios or missing time-steps consistently result in
reduced performance with increased uncertainty and calibration error, with an
optimum value of around 0.1. The optimal value must balance predictive perfor-
mance, uncertainty, and calibration error. This underlines the need for effective
management of missing instances. Particularly, when time windows are small,
determining the optimal ratio remains challenging. Nevertheless, MC-ConcTD
demonstrates adaptability by learning different values based on the validation
scenario (fold). This adaptability arises from the model’s ability to dynamically
adjust the dropout value using the ConcD [14], leading to improved performance
and calibration. Nevertheless, we observe high variability across datasets. More-
over, we notice that there may be multiple optimal dropout ratios, as illustrated
in Fig. 3 (c). For instance, for the SwissYield dataset, the learned ratio of the
second fold is approximately 0.25, whereas in the sixth fold, it is approximately
0.01. We attribute this variability to the distinct characteristics of each dataset,
including varying time series length, number of features, and dataset size (Ta-
ble 1). For instance, the LFMC dataset is characterized by only four time steps,
which could potentially explain the poor performance of TD-based models on


**[Table p10.1]**
| Dataset Model | R2 (↑) RMSE (↓) MAE (↓) ECE (↓) PU (↓) |
| --- | --- |
| MC-Dropout MC-TD SwissYield MC-ConcTD VI | 0.78 0.94 0.67 2.21 1.02 0.76 0.99 0.72 2.14 1.00 0.79 0.93 0.66 2.23 1.23 0.51 2.02 1.57 0.73 23.57 |
| MC-Dropout MC-TD LFMC MC-ConcTD VI | 0.72 20.69 14.64 2.43 1.11 0.69 21.92 15.67 3.77 1.01 0.72 20.80 14.76 3.41 8.01 0.53 27.00 20.00 4.48 4.01 |
| MC-Dropout MC-TD PM2.5 MC-ConcTD VI | 0.74 33.89 22.45 1.17 3.99 0.71 36.36 24.56 2.92 16.00 0.77 31.77 21.61 2.15 5.80 0.30 57.00 39.00 3.57 2.49 |

[CAPTION] Table 3: Performance comparison for various UQ models.


<!-- page 11 -->
Temporal Dropout for EO Time Series Data
11
this dataset. In contrast, the PM2.5 dataset has 120 time steps, but has only
short temporal dependencies compared to the SwissYield dataset. Therefore,
dropping the previous time-steps can significantly reduce performance. As a re-
sult, TD is less effective in this context, potentially explaining the low TD ratios
in Fig. 3 (c). Overall, ConcTD enhances the robustness of UQ across different EO
datasets, by continuously adapting to unique data characteristics while reducing
the computational burden of optimal ratio search, making UQ more accessible
for EO applications with time series data.
Limitations The proposed methods and experimental setup have limitations
that must be considered. We validate the models using EO data without real
missing values in the time series, which could potentially differ from scenarios in-
volving actual missing data. This limitation may affect how the models perform
in real-world situations where missing values are present. Additionally, we use
only an RNN encoder to learn the temporal patterns and only few UQ methods
for comparison. However, the primary goal of this study is to demonstrate the
effectiveness of TD and ConcTD in enhancing predictive performance and UQ,
rather than identifying the optimal architecture for each dataset and use case.
Nevertheless, additional models and architectures must be evaluated in the fu-
ture, including Transformers, and other UQ methods. Moreover, while TD has
shown significant improvements in related literature [18], we observe only small
improvements in this study on individual datasets. This may be attributed to
the task and the unique data characteristics, including short time series length
(LFMC), short temporal dependencies (PM2.5), or noise in remote sensing data.
More datasets must be considered to better understand the robustness of the
proposed methods. Finally, while we demonstrate improvements in our models
regarding the literature, we emphasize that poor calibration remains an ongo-
ing challenge that requires careful evaluation, an issue commonly encountered
in DL [17,48]. Specifically, EO data is often impacted by noisy und uncertain
measurements and spatio-temporal distribution shifts, which can lead to poor
calibration when the model is applied to unknown environments. Further ex-
ploring model calibration, including post-hoc calibration, will be required before
deploying models into practice.
6
Conclusion
This work underscores the importance of input uncertainty in time-series data
for EO regression applications. To address this, we introduced two novel uncer-
tainty estimation methods, namely MC-TD and MC-ConcTD. Both methods
account for input uncertainty in time series data by applying Temporal Dropout
(TD) at the inference level using Monte Carlo sampling. While MC-TD requires
manual and expensive tuning of the dropout ratio, MC-ConcTD learns this au-
tomatically as a free parameter. We empirically demonstrate their effectiveness
in enhancing predictive performance in various EO datasets. While these models
improve the accessibility of uncertainty estimation in EO applications, they also


<!-- page 12 -->
12
M. Miranda et al.
present challenges that require further investigation. Future research will focus
on validating these approaches across diverse EO applications and by considering
model calibration.
Acknowledgments. We acknowledge the support of the University of Kaiserslautern-
Landau (RPTU) and the German Research Center for Artificial Intelligence (DFKI).
Disclosure of Interests. The authors have no competing interests to declare that
are relevant to the content of this article.
References
1. Becker, S., Hug, R., Huebner, W., Arens, M., Morris, B.T.: Missformer:(in-)
attention-based handling of missing observations for trajectory filtering and pre-
diction. In: International Symposium on Visual Computing. pp. 521–533 (2021)
2. Blundell, C., Cornebise, J., Kavukcuoglu, K., et al.: Weight uncertainty in neural
network. In: International Conference on Machine Learning. pp. 1613–1622 (2015)
3. Camps-Valls, G., Tuia, D., Zhu, X.X., Reichstein, M.: Deep learning for the Earth
sciences: A comprehensive approach to remote sensing, climate science and geo-
sciences. John Wiley & Sons (2021)
4. Cao, W., Wang, D., Li, J., Zhou, H., et al.: Brits: Bidirectional recurrent imputation
for time series. Advances in Neural Information Processing Systems 31 (2018)
5. Che, Z., Purushotham, S., Cho, K., Sontag, D., Liu, Y.: Recurrent neural networks
for multivariate time series with missing values. Scientific reports 8(1), 6085 (2018)
6. Chen, J., Jönsson, P., Tamura, M., Gu, Z., Matsushita, B., Eklundh, L.: A simple
method for reconstructing a high-quality NDVI time-series data set based on the
Savitzky–Golay filter. Remote Sensing of Environment 91(3-4), 332–344 (2004)
7. Chen, S.: PM2.5 Data of Five Chinese Cities. UCI Machine Learning Repo. (2017)
8. Das, M., Ghosh, S.K.: A deep-learning-based forecasting ensemble to predict miss-
ing data for remote sensing analysis. IEEE Journal of Selected Topics in Applied
Earth Observations and Remote Sensing 10(12), 5228–5236 (2017)
9. Diaconu, C.A., Saha, S., Günnemann, S., et al.: Understanding the role of weather
data for earth surface forecasting using a ConvLSTM-based model. In: IEEE/CVF
Conference on Computer Vision and Pattern Recognition. pp. 1362–1371 (2022)
10. Du, W., Côté, D., Liu, Y.: Saits: Self-attention-based imputation for time series.
Expert Systems with Applications 219, 119619 (2023)
11. Ebel, P., Garnot, V.S.F., Schmitt, M., et al.: UnCRtainTS: Uncertainty quantifi-
cation for cloud removal in optical satellite time series. In: IEEE/CVF Conference
on Computer Vision and Pattern Recognition. pp. 2086–2096 (2023)
12. Ferrari, F., Ferreira, M.P., Almeida, C.A., Feitosa, R.Q.: Fusing Sentinel-1 and
Sentinel-2 images for deforestation detection in the Brazilian Amazon under diverse
cloud conditions. IEEE Geoscience and Remote Sensing Letters 20, 1–5 (2023)
13. Gal, Y., Ghahramani, Z.: Dropout as a Bayesian approximation: Representing
model uncertainty in deep learning. In: International Conference on Machine Learn-
ing. pp. 1050–1059 (2016)
14. Gal, Y., Hron, J., Kendall, A.: Concrete dropout. Advances in Neural Information
Processing Systems 30 (2017)


<!-- page 13 -->
Temporal Dropout for EO Time Series Data
13
15. Garnot, V.S.F., Landrieu, L., Chehata, N.: Multi-modal temporal attention models
for crop mapping from satellite time series. ISPRS Journal of Photogrammetry and
Remote Sensing 187, 294–305 (2022)
16. Gawlikowski, J., Tassi, C.R.N., Ali, M., Lee, J., Humt, M., Feng, J., Kruspe, A.,
Triebel, R., Jung, P., Roscher, R., et al.: A survey of uncertainty in deep neural
networks. Artificial Intelligence Review 56(Suppl 1), 1513–1589 (2023)
17. Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q.: On calibration of modern neural
networks. In: International Conference on Machine Learning. pp. 1321–1330 (2017)
18. Helber, P., Bischke, B., Packbier, C., Habelitz, P., Seefeldt, F.: An operational
approach to large-scale crop yield prediction with spatio-temporal machine learning
models. In: IGARSS 2024-2024 IEEE International Geoscience and Remote Sensing
Symposium. pp. 4299–4302. IEEE (2024)
19. Hüllermeier, E., Waegeman, W.: Aleatoric and epistemic uncertainty in machine
learning: An introduction to concepts and methods. Machine Learning 110, 457–
506 (2021)
20. Inglada, J., Vincent, A., Arias, M., Marais-Sicre, C.: Improved early crop type
identification by joint use of high temporal resolution SAR and optical image time
series. Remote Sensing 8(5), 362 (2016)
21. Kendall, A., Gal, Y.: What uncertainties do we need in Bayesian deep learning for
computer vision? Advances in Neural Information Processing Systems 30 (2017)
22. King, M.D., Platnick, S., Menzel, W.P., Ackerman, S.A., Hubanks, P.A.: Spa-
tial and temporal distribution of clouds observed by MODIS onboard the Terra
and Aqua satellites. IEEE Transactions on Geoscience and Remote Sensing 51(7),
3826–3852 (2013)
23. Ma, Y., Zhang, Z., Kang, Y., Özdoğan, M.: Corn yield prediction and uncertainty
analysis based on remotely sensed variables using a bayesian neural network ap-
proach. Remote Sensing of Environment 259, 112408 (2021)
24. Maddison, C.J., Mnih, A., Teh, Y.W.: The Concrete distribution: A continuous
relaxation of discrete random variables. In: International Conference on Learning
Representations (2017)
25. Maimaitijiang, M., Sagan, V., Sidike, P., Hartling, S., Esposito, F., Fritschi, F.B.:
Soybean yield prediction from UAV using multimodal data fusion and deep learn-
ing. Remote Sensing of Environment 237, 111599 (2020)
26. Mena, F., Arenas, D., Dengel, A.: Increasing the robustness of model predictions
to missing sensors in earth observation. arXiv preprint arXiv:2407.15512 (2024)
27. Mena, F., Arenas, D., Nuske, M., Dengel, A.: Common practices and taxonomy in
deep multi-view fusion for remote sensing applications. IEEE Journal of Selected
Topics in Applied Earth Observations and Remote Sensing pp. 4797 – 4818 (2024)
28. Metzger, N., Turkoglu, M.O., D’Aronco, S., Wegner, J.D., Schindler, K.: Crop
classification under varying cloud cover with neural ordinary differential equations.
IEEE Transactions on Geoscience and Remote Sensing 60, 1–12 (2021)
29. Miller, L., Pelletier, C., Webb, G.I.: Deep learning for satellite image time-series
analysis: A review. IEEE Geoscience and Remote Sensing Magazine (2024)
30. Miranda, M., Pathak, D., Nuske, M., Dengel, A.: Multi-modal fusion methods
with local neighborhood information for crop yield prediction at field and subfield
levels. In: IGARSS 2024-2024 IEEE International Geoscience and Remote Sensing
Symposium. pp. 4307–4311. IEEE (2024)
31. Mobiny, A., Yuan, P., Moulik, S.K., Garg, N., Wu, C.C., Van Nguyen, H.: Drop-
connect is effective in modeling uncertainty of bayesian deep networks. Scientific
reports 11(1), 5458 (2021)


<!-- page 14 -->
14
M. Miranda et al.
32. Mohammadi Foumani, N., Miller, L., Tan, C.W., Webb, G.I., Forestier, G., Salehi,
M.: Deep learning for time series classification and extrinsic regression: A current
survey. ACM Computing Surveys 56(9), 1–45 (2024)
33. Nguyen, L.H., Zhu, J., Lin, Z., Du, H., Yang, Z., Guo, W., Jin, F.: Spatial-temporal
multi-task learning for within-field cotton yield prediction. In: Advances in Knowl-
edge Discovery and Data Mining. pp. 343–354 (2019)
34. Nix, D.A., Weigend, A.S.: Estimating the mean and variance of the target proba-
bility distribution. In: IEEE International Conference on Neural Networks. vol. 1,
pp. 55–60 (1994)
35. Ofori-Ampofo, S., Pelletier, C., Lang, S.: Crop type mapping from optical and
radar time series using attention-based deep learning. Remote Sensing 13(22),
4668 (2021)
36. Orfanidis, S.J.: Introduction to signal processing. Prentice-Hall, Inc. (1995)
37. Pathak, D., Miranda, M., Mena, F., Sanchez, C., et al.: Predicting crop yield
with machine learning: An extensive analysis of input modalities and models on
a field and sub-field level. In: IEEE International Geoscience and Remote Sensing
Symposium. pp. 2767–2770 (2023)
38. Perich, G., Turkoglu, M.O., Graf, L.V., Wegner, J.D., Aasen, H., Walter, A.,
Liebisch, F.: Pixel-based yield mapping and prediction from Sentinel-2 using spec-
tral indices and neural networks. Field Crops Research 292, 108824 (2023)
39. Rao, K., Williams, A.P., Flefil, J.F., Konings, A.G.: SAR-enhanced mapping of
live fuel moisture content. Remote Sensing of Environment 245, 111797 (2020)
40. Requena-Mesa, C., Benson, V., Reichstein, M., Runge, J., Denzler, J.: Earth-
net2021: A large-scale dataset and challenge for Earth surface forecasting as a
guided video prediction task. In: IEEE/CVF Conference on Computer Vision and
Pattern Recognition. pp. 1132–1142 (2021)
41. Sarukkai, V., Jain, A., Uzkent, B., Ermon, S.: Cloud removal from satellite images
using spatiotemporal generator networks. In: IEEE/CVF Winter Conference on
Applications of Computer Vision. pp. 1796–1805 (2020)
42. Scarpa, G., Gargiulo, M., Mazza, A., Gaetano, R.: A CNN-based fusion method
for feature extraction from Sentinel data. Remote Sensing 10(2), 236 (2018)
43. Shen, H., Li, X., Cheng, Q., Zeng, C., Yang, G., Li, H., Zhang, L.: Missing informa-
tion reconstruction of remote sensing data: A technical review. IEEE Geoscience
and Remote Sensing Magazine 3(3), 61–85 (2015)
44. Shukla, S.N., Marlin, B.: Multi-time attention networks for irregularly sampled
time series. In: International Conference on Learning Representations (2021)
45. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., Salakhutdinov, R.:
Dropout: A simple way to prevent neural networks from overfitting. Journal of
Machine Learning Research 15(1), 1929–1958 (2014)
46. Tan, C.W., Bergmeir, C., Petitjean, F., Webb, G.I.: Time series extrinsic regression:
Predicting numeric values from time series data. Data Mining and Knowledge
Discovery 35(3), 1032–1060 (2021)
47. Tseng, G., Cartuyvels, R., Zvonkov, I., Purohit, M., Rolnick, D., Kerner, H.R.:
Lightweight, pre-trained transformers for remote sensing timeseries. In: NeurIPS
2023 Workshop on Tackling Climate Change with Machine Learning (2023)
48. Valdenegro-Toro, M.: I find your lack of uncertainty in computer vision disturbing.
In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition. pp. 1263–1272 (2021)
49. Valdenegro-Toro, M., de Jong, I.P., Zullich, M.: Unified uncertainties: Combining
input, data and model uncertainty into a single formulation. arXiv e-prints pp.
arXiv–2406 (2024)


<!-- page 15 -->
Temporal Dropout for EO Time Series Data
15
50. Weilandt, F., Behling, R., Goncalves, R., Madadi, A., Richter, L., Sanona, T.,
Spengler, D., Welsch, J.: Early crop classification via multi-modal satellite data
fusion and temporal attention. Remote Sensing 15(3), 799 (2023)