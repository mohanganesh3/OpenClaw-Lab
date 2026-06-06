<!-- page 1 -->
A Machine Learning Approach Capturing Hidden Parameters in Autonomous
Thin-Film Deposition
Yuanlong Zheng1,2, Connor Blake2, Layla Mravac2, Fengxue Zhang3, Yuxin Chen3, and Shuolong Yang2
1Department of Physics, University of Chicago, Chicago, IL 60637, USA
2Pritzker School of Molecular Engineering, University of Chicago, Chicago, IL 60637, USA and
3Department of Computer Science, University of Chicago, Chicago, IL 60637, USA
(Dated: December 2, 2024)
Abstract
The integration of machine learning and robotics into thin film deposition is transforming material
discovery and optimization. However, challenges remain in achieving a fully autonomous cycle of
deposition, characterization, and decision-making. Additionally, the inherent sensitivity of thin film
growth to hidden parameters such as substrate conditions and chamber conditions can compromise
the performance of machine learning models. In this work, we demonstrate a fully autonomous phys-
ical vapor deposition system that combines in-situ optical spectroscopy, a high-throughput robotic
sample handling system, and Gaussian Process Regression models.
By employing a calibration
layer to account for hidden parameter variations and an active learning algorithm to optimize the
exploration of the parameter space, the system fabricates silver thin films with optical reflected
power ratios within 2.5% of the target in an average of 2.3 attempts. This approach significantly
reduces the time and labor required for thin film deposition, showcasing the potential of machine
learning-driven automation in accelerating material development.
I.
INTRODUCTION
In physical vapor deposition (PVD) of thin film ma-
terials, the traditional human-led process encompasses
numerous cycles of selecting deposition parameters, per-
forming deposition, characterizing film properties, and
re-adjusting the parameters accordingly. The recent de-
velopment of machine learning (ML) [1], coupled with ad-
vancements in robotics [2], now potentially enables fully
automating this process, liberating researchers from the
repetitive cycles and accelerating the optimization of ma-
terial properties [3].
Several studies have sought to integrate ML with the
PVD process. Typically, these approaches involved train-
ing ML models that map deposition parameters, such as
substrate temperature, deposition rate, and flux ratio, to
material properties, such as stoichiometry [4–6], electri-
cal conductivity [7–9], surface morphology [10, 11], crys-
tallinity [12–17], and superconducting critical tempera-
tures [18]. The trained models are then used to predict
material properties, with Bayesian optimization (BO)
frequently employed to autonomously determine the de-
position parameters for subsequent samples [9, 19, 20].
Such optimization has the potential to replace human
decision-making, efficiently exploring the deposition pa-
rameter space to optimize the material properties [21].
However, ML models are known for their sensitivity to
noise in the training data. In the meantime, the thin film
deposition process can be prone to irreproducibility due
to factors such as different initial substrate conditions
and chamber environments [5, 8, 18]. The inconsistency
in the growth outcome may be treated by the model as
noise which undermines the model’s performance. These
“hidden parameters” that lead to the irrepeatability of
thin film properties are often difficult to capture and in-
corporate into the model. The field of ML-assisted thin
film deposition currently lacks a systematic approach to
effectively account for these hidden parameters.
Beyond algorithmic challenges, the full potential of
ML-assisted PVD hinges on the complete automation
of hardware systems.
PVD systems require high vac-
uum (HV) or ultra-high vacuum (UHV) environments,
which present significant challenges for fully automating
sample transfer and characterization processes. As such,
most studies in the field of ML-assisted thin film deposi-
tion still rely on traditional manual handling of samples,
which limits sample throughput and hinders the realiza-
tion of the fully autonomous ML-assisted PVD [9, 18, 22].
Shimizu et al. demonstrated a system fully automating
the deposition of Nb-doped TiO2 films and the minimiza-
tion of its resistance [23]. However, the system requires
complex multi-chamber systems with sophisticated trans-
fer mechanisms, thereby increasing the complexity of the
setup and limiting its scalability. Achieving a scalable
method to fully automate PVD systems is a critical hur-
dle to unlocking the full promise of ML integration in
thin film deposition.
II.
METHODS AND RESULTS
In this work, we address inefficiencies in current ML-
driven thin film deposition by developing a fully au-
tomated system with high sample throughput and a
mechanism to account for hidden parameter variations.
Our system integrates a UHV chamber with a 72-slot
robotic sample handling system optimized for maximiz-
ing throughput, in-situ optical characterization, and ML-
powered prediction of characterization outputs and opti-
mal growth conditions. To account for the varying hid-
den parameters, we introduce a calibration layer that
provides critical information about the initial condition
arXiv:2411.18721v1  [cond-mat.mtrl-sci]  27 Nov 2024


<!-- page 2 -->
2
FIG. 1: A self-learning autonomous physical vapor deposition system for silver thin film growth. (a) The autonomous
learning cycle incorporates (i) identification of the growth condition with the highest model uncertainty, (ii) sample
growth, and (iii) updating the model with new data. (b) The autonomous testing cycle incorporating (i) prediction
of the optimal growth condition that minimizes the loss function, (ii) sample growth, and (iii) assessment of success.
(c) Schematic illustration of the autonomous deposition setup featuring robotic sample handling and in-situ optical
characterization.
of each sample and the chamber environment. Further-
more, we employ an autonomous learning approach using
Gaussian Process Regression (GPR) models to efficiently
explore the multidimensional input space and accurately
predict the target properties of the films. By integrat-
ing these techniques, we have engineered an ML-driven,
closed-loop deposition system, thereby eliminating the
need for human intervention at any stage of the growth
cycle. We demonstrate the autonomous synthesis of sil-
ver thin films with optical properties within 2.5% of the
user-specified target in an average of 2.3 attempts, un-
derscoring the potential of our ML-driven system for sig-
nificantly accelerating the optimization of material prop-
erties.
1. Introduction of the System
To demonstrate the principles of ML-driven au-
tonomous PVD, we seek to fabricate silver thin films
with user-specified optical properties. The growth con-
ditions of silver films such as the growth rate, substrate
temperature, and film thickness [24] impact the films’
porosity [25], grain size [26], and electron-phonon inter-
actions [27]. All of these aspects are correlated with the
real and imaginary parts of the optical constants for sil-
ver films. It is difficult to model all of these mechanisms
using simple physics laws, which warrants an ML-driven
material optimization.
Our autonomous PVD system incorporates a shadow
mask beneath a 72-slot sample handling system, ensuring
that only one sample is exposed to the molecular beam at
a time (Figure 1(c)). Silver (99.999%, Thermo Fisher) is
deposited onto double-side polished BK7 substrates us-
ing an effusion cell (MBE-Komponenten) at a base pres-
sure of < 5 × 10−9 mbar and a deposition pressure of
1 × 10−8 mbar. The reflectivity and absorptivity of the
silver thin films are characterized using five p-polarized
lasers with wavelengths of 443, 514, 689, 781, and 817
nm (Coherent StingRay). The lasers are mounted on a
computer numerically controlled (CNC) linear rail (Fig-
ure 1(c)) pointing at the substrate with an incident angle
of 45 degrees. The transmitted (Pt) and reflected power
(Pr) are measured, and we define the reflected power ra-
tio R and absorptivity A :
R =
Pr
Pr + Pt
,
A = Pi −Pr −Pt
Pi
where Pi denotes the incident power. For convenience,
the reflected power ratio and the absorptivity at 443 nm

[CAPTION] FIG. 1: A self-learning autonomous physical vapor deposition system for silver thin film growth. (a) The autonomous


<!-- page 3 -->
3
are denoted as R443 and A443, respectively, and similarly
for other wavelengths.
All hardware control and data acquisition are managed
by MATLAB scripts which allow the system to grow and
characterize up to 72 samples consecutively without hu-
man intervention. The optical characterization data are
fed into the ML algorithm, which predicts the growth
condition required for the sample to attain user-specified
R values at targeted wavelengths. The process is divided
into the autonomous learning and autonomous testing
stages.
In the learning stage, optical characterization
data are used to train ML models, which then determine
the next point to explore for further learning (Figure
1(a)). In the testing stage, the trained model guides the
system to fabricate silver thin films with user-specified
optical properties (Figure 1(b)).
2. Calibration Layer
For thin film deposition, slight differences in substrate
and chamber conditions can significantly affect growth
dynamics, especially for the first few atomic layers [28].
Parameters such as substrate surface roughness, and
residual pressure of various elements in the chamber are
unknown to us. Although some of these parameters can
be measured with various difficulties, it is infeasible for
one to exhaust all the factors in the deposition process.
To quantify the effects of these hidden parameters on the
sample properties, we first grow 24 samples at an effu-
sion cell temperature (T) of 875°C and a deposition time
(t) of 5000 seconds. After 5000 seconds, the R817 of the
samples has a distribution with a standard deviation of
0.061 (Figure 2(c)).
To account for this variability, we find that the R’s
of the 24 samples measured at t = 1000 seconds can be
used to quantify the fluctuations in the growth dynamics
and its correlation with R’s at later times. Here we plot
R443, R689, R817 at 5000 seconds against their R689 at
1000 seconds and fit them with a simple linear function
(Figure 2 (a-c)). The standard deviation of the R817 at
5000 seconds is 0.061. Meanwhile, its root-mean-square
deviation (RMSD) from the linear fit is 0.032. This sug-
gests that R689 at 1000 seconds is moderately correlated
with the variation of R’s at later times. Therefore, by
initially growing a thin layer at T = 875◦C for 1000 sec-
onds and measuring its R689, we partially captures the
effect of the hidden parameters on the growth outcome.
We define this layer as the calibration layer, and its R689
is denoted as Rc. We choose 689 nm as the wavelength
for Rc due to its median position among the five wave-
lengths, thereby providing the most representative infor-
mation.
For all samples in the subsequent experiments, we first
grow the calibration layer and record Rc. The sample
is then grown for time t at temperature T. R and A
are collected for all wavelengths throughout the growth
(Figure 3). To maximize data acquisition efficiency, we
divide the total deposition time into 98-second blocks.
Within each block, the reflected and transmitted power
at each wavelength are sequentially measured starting at
(a)
(b)
(c)
(d)
FIG. 2: Calibration layer accounts for variations in op-
tical properties of silver films. (a-c) A linear fit between
R443, R689, R817 at 5000 seconds and R689 at 1000 sec-
onds. The moderate R-squared values indicate that the
variances of R’s at later times can be partially captured
by R at 1000 seconds. (d) Translucent planes of time
required to reach specified R443 values for given effusion
cell temperatures and Rc’s. Rc is negatively correlated
with the time required to reach a given R443.
the time points 0, 16, 32, 48, and 64 seconds (Table 1).
The intervals between measurements are limited by the
speed of the linear rail and the collection period of 5 sec-
onds per wavelength. Note that Rc, though a measured
value, is treated as an input parameter of the model in
the dataset.
FIG. 3: Illustration of the data acquisition process for
each iteration in the autonomous learning stage: the cal-
ibration layer is first grown to determine the Rc, followed
by measuring optical properties at all 5 wavelengths in
cycles.
3. Autonomous Learning
GPR models are a form of probabilistic supervised
learning models derived from non-parametric statistics.

[CAPTION] FIG. 2: Calibration layer accounts for variations in op-

[CAPTION] FIG. 3: Illustration of the data acquisition process for


<!-- page 4 -->
4
Model Inputs
Model Outputs
Temperature (°C) Time (s)
Rc
Wavelength (nm)
R
A
872.5
0
0.038
443
0.356
0.422
872.5
16
0.038
514
0.158
0.194
872.5
32
0.038
689
0.044
0.114
872.5
48
0.038
781
0.039
0.048
872.5
64
0.038
817
0.034
0.014
872.5
98
0.038
443
0.394
0.442
872.5
. . .
0.038
. . .
. . .
. . .
872.5
tmax
0.038
817
0.824
0.363
TABLE I: An exemplary data set taken at the effusion
cell temperature 872.5 ℃, with model inputs and out-
puts.
GPR models are defined as a continuous set of distribu-
tions f(x) where f maps values x ∈Rn to the space of
Gaussian probability density functions on y. For any two
data points (x1, y1) and (x2, y2), the covariance between
y1 and y2 is assumed to be purely a function of x1 and
x2, k(x1, x2).
A GPR model is fully characterized by its prior func-
tion g(x), mean function m(x) ≡E[f(x)], and covariance
function k(x1, x2). The prior function g(x) describes a
”guess” for how the output space should appear in the
absence of data.
The mean function describes the ex-
pected y value for a given input x based on the set of
data points {(xi, yi)}. The covariance function, referred
to as a kernel, describes how well one should be able to
predict ˜y given ˜x and (x, y). Here, we used a radial basis
function (RBF) kernel.
kRBF(x1, x2) = exp
 
−1
2(x1 −x2)T Θ−2(x1 −x2)
 
(1)
chosen for its ability to act as a universal approximator
[29]. The length scale matrix Θ = diag(θ1, θ2, . . . , θn)
contains the length-scale parameters which are optimized
during learning and describe the width of the Gaussian
kernel used for prediction.
Mapping the data and prior distribution to the pre-
dicted output (the posterior distribution) is known as
conditioning and is computed via Bayes’ rule. As such,
the predicted posterior distribution at a point ˜x based on
observed data {(xi, yi)} is given by a normal distribution:
f(˜x) ∼N(ki(K−1)ijyj, k(˜x, ˜x) −ki(K−1)ijkj)
(2)
where ki = k(˜x, xi) and Kij = k(xi, xj) for symmetric
kernels like RBF. Here we have used the prior g(x) = 0
for simplicity. From here, this distribution will be writ-
ten as GPRy(x) = (µy, σy) with µy and σy referred to
as the “predicted mean” and “predicted uncertainty” re-
spectively. While this distribution appears to be com-
putationally intractable for large datasets, the function
m(x) has a closed-form representation for the RBF kernel
and is readily computed by libraries such as GPyTorch
[30].
We employ two GPR models for R and A with RBF
kernels, both having input parameters x = (T, t, λ, Rc),
and outputs (µR, σR) and (µA, σA), respectively. Here,
we define two modes of generating training data for mod-
els: predefined learning and autonomous learning.
In
predefined learning, the growth parameters of the train-
ing samples are set to divide the parameter space into
equal increments.
In autonomous learning, the model
autonomously determines these growth parameters in a
way that optimally learns the parameter space.
A series of predefined learning is used to initialize the
model during which 9 samples are grown between 820
and 880°C with 7.5°C increments and for t of
tmax(T) = 3.22 × 1014 × e−0.0285T (◦C)(seconds)
(3)
where tmax is determined such that each sample reaches
a thickness yielding an R > 0.8 for all wavelengths. This
ensures the generation of data over a wide range of R
values for training the model, while also avoiding the
unnecessary time spent growing films until R asymptoti-
cally approaches 1. The exponential relationship between
tmax and T assumes that silver’s vapor pressure, and con-
sequently the growth rate, increases exponentially with
T, ensuring that all samples at tmax have approximately
the same thickness. During the growth of each sample,
R and A are repeatedly measured at each wavelength to
obtain data between t = 0 and tmax (Table 1).
The system proceeds to the autonomous learning stage.
After each sample’s calibration layer is grown and Rc is
measured, the effusion cell temperature T for the remain-
der of the growth is selected within the interval of [820,
880] °C according to:
Tselected = arg max
T



qPtmax(T )
t=0
P
λ σR(T, t, λ, Rc)2
tmax(T)



(4)
Here the T with the highest prediction uncertainty
for all relevant times and wavelengths is selected for the
growth. The sample is then grown and measured at a se-
ries of times, as shown in Figure (3), until t = tmax(Tgrow)
specified by Eqn. (3). Such a complete deposition and
characterization process, until the growth time reaches
tmax(Tgrow), is termed one iteration (Figure 3).
Note
that the model only needs to choose T, among all input
parameters (T, t, λ, Rc), because data are collected for
each sample repeatedly over time and for all wavelengths,
while Rc is a result of the current chamber and substrate
initial condition and cannot be controlled by the algo-
rithm. After each iteration of deposition and character-
ization is finished, the GPR model is updated with the
new data. This process is repeated for 8 iterations until
the maximum uncertainty in the parameter space con-
verges (Figure 4 (b)). The total learning stage consists


<!-- page 5 -->
5
(a)
(b)
(c)
FIG. 4: Performance of autonomous learning in the fab-
rication of silver thin films. (a) Evolution of model pre-
dicted uncertainty, averaged over all Rλ’s, during au-
tonomous learning.
(b) Convergence of the maximum
uncertainty during autonomous learning. (c) Comparing
the model prediction errors, defined as the differences
between all measured and predicted Rλ’s.
The error
distribution when adopting the calibration layer and au-
tonomous learning is benchmarked against the case with-
out these techniques.
of 17 samples, including 9 from predefined learning and
8 from autonomous learning.
Figure 4(a) shows the evolution of the average pre-
dicted uncertainty for all R’s during autonomous learn-
ing. After 8 iterations, the uncertainty over the entire pa-
rameter space becomes more uniformly distributed and
has an average value of 0.032. The uncertainty does not
further decrease below this level. The remaining uncer-
tainty is likely due to the hidden parameters unaccounted
by the calibration layer and measurement noises (e.g. un-
certainty in laser power measurements).
Unlike typical Bayesian optimizations aimed at opti-
mizing a single output [31], our model is trained to ac-
quire comprehensive information over the entire param-
eter space, enabling the system to respond to arbitrary
R requests at given wavelengths.
4. Autonomous Testing
4.1 Single-wavelength R Targets
We select 5 random single-wavelength targets Rtarget
λ
,
one for each wavelength of the lasers. Given a certain
Rc, there exist infinitely many (T, t) that can achieve the
requested R at the specified wavelength. The degeneracy
is removed by also aiming for the minimum Aλ. The loss
function for each single-wavelength target is defined as
Lλ = (µR,λ −Rtarget
λ
)2 + 4σ2
R,λ + µ2
A ,λ + 4σ2
A ,λ
if |µR,λ −Rtarget
λ
| > 0.01 :
Lλ 7→Lλ + 100(µR,λ −Rtarget
λ
)2
(5)
The uncertainties of predicted R and A are added
to the loss function to penalize growth conditions with
high uncertainties. The loss increases rapidly when the
difference between the target and predicted R exceeds
0.01 to prioritize proximity to the target value of R.
The loss function is minimized using the Adam opti-
mizer [32], an adaptive learning rate optimization algo-
rithm designed to handle sparse gradients on noisy prob-
lems. The set of growth parameters (T, t) at the mini-
mum of the loss function is used for deposition. After
deposition, the sample’s measured Rλ is compared to
Rtarget
λ
. If |Rλ −Rtarget
λ
| < 0.025, the growth is con-
sidered successful, and the system moves to the next
target. If unsuccessful, the model is updated with the
obtained optical characterization data from the current
sample and re-attempts the target until success. Table
2 displays the results for 5 single-wavelength targets. It
took 2 attempts on average for a target to be successfully
achieved. For each of the 10 samples grown during this
stage, the model makes 5 predictions on its R for each
wavelength. For these 50 total predictions, the mean ab-
solute error (MAE) between the model predictions and
measured results is 0.0246, which demonstrates the ac-
curacy of the prediction. Moreover, the average model
predicted uncertainty is 0.0267, and its proximity to the
MAE demonstrates the accuracy of the model’s estimate
of its uncertainty.
To further demonstrate the effectiveness of the imple-
mentation of calibration layers and autonomous learning,
we perform a control experiment without these methods
to benchmark the model prediction performance. In the
control experiment, the model only goes through the pre-
defined learning process, in which 17 samples are grown
at T between 820 and 880°C with 3.75°C increments,
and for t as specified in Eqn.(3). The system is then re-
quested to produce silver thin films that satisfy the same
5 single-wavelength target. Despite the control experi-
ment having the same amount of training sample data
as the previous experiment, it requires on average 3.6
attempts to successfully achieve each target. The MAE
between the model predictions and measured results is
0.0618, also significantly increased from the previous ex-
periment (Figure 4(c)). The control experiment hence

[CAPTION] FIG. 4: Performance of autonomous learning in the fab-


<!-- page 6 -->
6
demonstrates the superior performance of the model us-
ing calibration layers and autonomous learning.
Target
Attempt 1
2
3
4
5
6
7
Rtarget
443
= 0.61
0.609
Rtarget
514
= 0.47
0.413
0.452
Rtarget
689
= 0.88
0.891
Rtarget
781
= 0.30
0.298
Rtarget
817
= 0.58
0.545
0.636 0.541 0.510 0.585
Rtarget
443
= 0.85
N/A
N/A 0.914 0.845
Rtarget
781
= 0.47
0.654 0.507
Rtarget
443
= 0.85
N/A
N/A 0.797 N/A 0.879 0.779 0.817
Rtarget
781
= 0.35
0.331
0.489 0.325 0.342
TABLE II: Results of achieving specified optical proper-
ties with single and multi-wavelength targets.
4.2 Multi-wavelength Target
We also specify R targets spanning multiple wave-
lengths. The loss function is defined as
L =
X
{λ}specified
(µR,λ −Rtarget
λ
)2
(6)
As the number of wavelengths specified in the target
increases, it is not guaranteed that there exists a set of
growth parameters (T, t, Rc) that could produce a film
with the desired optical properties. Therefore, when the
loss function is minimized and any of the µR,λ’s is still >
0.01 away from the target, the algorithm decides that no
growth parameters can be found to reach the target at the
current sample’s Rc. The system aborts this sample and
grows the calibration layer on the subsequent substrates
until it finds one with a set of (T, t, Rc) that yields µR,λ
within ±0.01 of the target for all specified wavelengths.
The system proceeds to deposit the sample at the
found (T, t) and subsequently measures its optical prop-
erties {Rλ, Aλ}.
The system then updates the model
with all collected data. If
X
{λ}specified
  Rλ −Rtarget
λ
   < 0.025n
(7)
where n is the number of specified wavelengths in the
target, the growth is considered success, and the system
moves on to the next target. If unsuccessful, the system
makes a new prediction using the updated model and
re-attempts the target until success.
Table 2 displays the results using multi-wavelength tar-
gets. For certain samples, the model is unable to find a
set of growth parameters that can yield a film to reach
the specified multi-wavelength targets, denoted as N/A.
These samples do not count towards the number of failed
attempts. The system aborts the sample and proceeds to
the next substrate. Upon switching to a sample with an
Rc that leads to an achievable growth condition to obtain
the desired optical properties, it begins the growth and
can achieve the target within the allowed error. In this
stage, 6 attempts are made to successfully achieve 2 tar-
gets. The success in achieving the two multi-wavelength
targets demonstrates the versatility of our system to con-
trol the silver thin film’s power-splitting spectrum over
multiple wavelengths.
III.
DISCUSSION AND CONCLUSION
In this work, we have established the ability of a fully
autonomous ML-enabled system to address persisting
challenges in the field of PVD thin film growth, namely
the lengthy and labor-intensive human involvement and
the difficulty of adapting to hidden parameters in the
growth conditions. We see three components critical to
a fully automated thin film deposition system: full me-
chanical and software automation, active learning for ex-
ploring the parameter space during data collection, and
active targeting of desired outputs. Previous studies have
implemented two of the three such as automation and ac-
tive data collection [33, 34] or manually-executed active
training and target-seeking [9], but to our knowledge,
none have performed all three. Our work demonstrates
the first system to implement all three in the field of thin
film deposition. Using Bayesian ML, we have engineered
a fully autonomous system that learns the dynamics of
silver growth while accounting for variations in substrates
and chamber conditions, and reliably grows samples with
desired optical properties.
We choose to work with silver thin films because they
are well-understood but retain the difficult aspects of thin
film growth. As such, the methods that we develop are
broadly applicable. The procedure we demonstrate of us-
ing predefined learning to map the parameter space, au-
tonomous learning to minimize model uncertainty, and
optimization to seek specific targets is readily generaliz-
able to other thin film growth tasks.
With the fully autonomous system, we grow a total of
38 samples in the training and testing stage and collect
more than 20,000 data points without human interven-
tion. This represents a significant advantage over human-
led experimentation. This high-throughput operation is
primarily enabled by our use of a 72-slot sample handing
system and in-situ quasi-continuous optical characteri-
zation.
By designing chambers for high-capacity sam-
ple transfer and measurement, similar throughput can
be achieved for other thin film fabrication and in-situ
characterization tasks.
In contrast to other ML techniques such as artificial
neural networks, Bayesian optimization allows the quan-
tification of prediction uncertainties.
This aspect has
been used to decide optimal points of parameter space
exploration [9, 18, 19]. Moreover, because of this ability

[CAPTION] Table 2 displays the results using multi-wavelength tar-


<!-- page 7 -->
7
to quantify uncertainty, our model sets numerical thresh-
olds to determine both when to stop the learning process
and when to conclude that a target is unattainable. By
exploring the variety of decisions that the model is able
to make, we further reduce the amount of human involve-
ment in the process.
The high throughput of the system also enables the
quantitative examination of the fluctuations in R’s given
the same growth parameters (T, t). This fluctuation is
attributed to the hidden parameters in the growth pro-
cess. We introduce the concept of the calibration layer
that allows the model to partially capture the effect of
these hidden parameters. With the information on the
otherwise unknown hidden parameters, significant im-
provement in model prediction performance is achieved.
The fluctuations in the growth process have been found
in previous studies utilizing ML for thin film deposi-
tion [5, 18], though its effect on the material property
and model performance is often neglected. Shrivastava
et al.[8] have made an effort to deliberately avoid grow-
ing at growth parameters that are particularly sensitive
to such fluctuations. Our work represents an important
step toward quantifying these hidden parameters by in-
corporating the calibration layer measurement Rc into
the model’s parameter space.
One can extend the calibration layer approach by in-
corporating additional checkpoints along the deposition
trajectories and optimize in a higher-dimensional pa-
rameter space.
This could eventually lead to a quasi-
continuous adaptive control algorithm where the system
adjusts in real time based on feedback. However, imple-
menting such adaptive control using reinforcement learn-
ing in complex experimental settings poses challenges, as
it typically requires substantial amount of training trials
[35]. Our method balances elements of adaptive control
with efficient model training, accounting for both the lim-
ited data availability and the inherent variability in the
material growth process. With more advanced reinforce-
ment learning algorithms [36] and higher sample through-
put, one can envision that the quasi-continuous adaptive
control may be implemented in thin film deposition in
the future.
Acknowledgments
This work is supported by the National Science Foun-
dation (NSF CNS-2019131) and University of Chicago
Big Idea Generator Seed Grant. This work made use of
the shared facilities at the University of Chicago Materi-
als Research Science and Engineering Center, supported
by National Science Foundation under award number
DMR-2011854.
This work made use of the Pritzker
Nanofabrication Facility at the Pritzker School of Molec-
ular Engineering at the University of Chicago, which re-
ceives support from Soft and Hybrid Nanotechnology Ex-
perimental (SHyNE) Resource (NSF ECCS-2025633), a
node of the National Science Foundation’s National Nan-
otechnology Coordinated Infrastructure.
REFERENCES
[1] M. I. Jordan and T. M. Mitchell, Machine learning:
Trends, perspectives, and prospects, Science 349, 255
(2015).
[2] E. Garcia, M. A. Jimenez, P. G. De Santos, and M. Ar-
mada, The evolution of robotics research, IEEE Robotics
I& Automation Magazine 14, 90 (2007).
[3] D. Morgan and R. Jacobs, Opportunities and challenges
for machine learning in materials science, Annual Review
of Materials Research 50, 71 (2020).
[4] Y. K. Wakabayashi,
T. Otsuka,
Y. Krockenberger,
H. Sawada, Y. Taniyasu, and H. Yamamoto, Stoichiomet-
ric growth of srtio3 films via bayesian optimization with
adaptive prior mean, APL Machine Learning 1 (2023).
[5] D. M. F´ebba, K. R. Talley, K. Johnson, S. Schaefer, S. R.
Bauers, J. S. Mangum, and A. Zakutayev, Autonomous
sputter synthesis of thin film nitrides with composition
controlled by bayesian optimization of optical plasma
emission, APL Materials 11, 7 (2023).
[6] N. S. Johnson, A. A. Mishra, D. J. Kirsch, and A. Mehta,
Active learning for rapid targeted synthesis of composi-
tionally complex alloys, Materials 17, 4038 (2024).
[7] T. Ishiyama, K. Nozawa, T. Nishida, T. Suemasu, and
K. Toko, Bayesian optimization-driven enhancement of
the thermoelectric properties of polycrystalline iii-v semi-
conductor thin films, NPG Asia Materials 16, 17 (2024).
[8] A. Shrivastava, M. Kalaswad, J. O. Custer, D. P. Adams,
and H. N. Najm, Bayesian optimization for stable prop-
erties amid processing fluctuations in sputter deposition,
Journal of Vacuum Science & Technology A 42 (2024).
[9] Y. K. Wakabayashi,
T. Otsuka,
Y. Krockenberger,
H. Sawada, Y. Taniyasu, and H. Yamamoto, Machine-
learning-assisted thin-film growth:
Bayesian optimiza-
tion in molecular beam epitaxy of srruo3 thin films, APL
Materials 7, 10 (2019).
[10] A. S. Messecar, S. M. Durbin, and R. A. Makin, Quan-
tum and classical machine learning investigation of syn-
thesis–structure relationships in epitaxially grown wide
band gap semiconductors, MRS Communications 14, 660
(2024).
[11] C. Shen, W. Zhan, K. Xin, M. Li, Z. Sun, H. Cong, C. Xu,
J. Tang, Z. Wu, B. Xu, and Z. Wei, Machine-learning-
assisted
and
real-time-feedback-controlled
growth
of
inas/gaas quantum dots, Nature Communications 15,
2724 (2024).
[12] H. J. Kim, M. Chong, T. G. Rhee, Y. G. Khim, M. H.
Jung, Y. M. Kim, H. Y. Jeong, B. K. Choi, and Y. J.
Chang, Machine-learning-assisted analysis of transition
metal dichalcogenide thin-film growth, Nano Conver-
gence 10, 10 (2023).
[13] S. R. Provence, S. Thapa, R. Paudel, T. K. Truttmann,
A. Prakash, B. Jalan, and R. B. Comes, Machine learning
analysis of perovskite oxides grown by molecular beam
epitaxy, Physical Review Materials 4, 083807 (2020).
[14] D. Guevarra, L. Zhou, M. H. Richter, A. Shinde, D. Chen,
C. P. Gomes, and J. M. Gregoire, Materials struc-
ture–property factorization for identification of synergis-
tic phase interactions in complex solar fuels photoanodes,
npj Computational Materials 8, 57 (2022).
[15] Z. Ni and H. Matsui, Phase control of heterogeneous
HfXZr(1-X)O2 thin films by machine learning, Japanese
Journal of Applied Physics 61, SH1009 (2022).


<!-- page 8 -->
8
[16] H. Liang, V. Stanev, A. G. Kusne, Y. Tsukahara, K. Ito,
R. Takahashi, M. Lippmaa, and I. Takeuchi, Applica-
tion of machine learning to reflection high-energy elec-
tron diffraction images for automated structural phase
mapping, Physical Review Materials 6, 063805 (2022).
[17] S. Ament, M. Amsler, D. R. Sutherland, M. C. Chang,
D. Guevarra, A. B. Connolly, J. M. Gregoire, M. O.
Thompson, C. P. Gomes, and R. B. Van Dover, Au-
tonomous materials synthesis via hierarchical active
learning of nonequilibrium phase diagrams, Science Ad-
vances 7, eabg4930 (2021).
[18] I. Ohkubo, Z. Hou, J. N. Lee, T. Aizawa, M. Lipp-
maa, T. Chikyow, and T. Mori, Realization of closed-
loop optimization of epitaxial titanium nitride thin-film
growth via machine learning, Materials Today Physics
16, 100296 (2021).
[19] Y. K. Wakabayashi,
T. Otsuka,
Y. Krockenberger,
H. Sawada, Y. Taniyasu, and H. Yamamoto, Bayesian op-
timization with experimental failure for high-throughput
materials growth, npj Computational Materials 8, 180
(2022).
[20] D. Packwood, Bayesian Optimization for Materials Sci-
ence (Springer Singapore, Singapore, 2017).
[21] B. Shahriari, K. Swersky, Z. Wang, R. P. Adams, and
N. De Freitas, Taking the human out of the loop: A
review of bayesian optimization, Proceedings of the IEEE
104, 148 (2015).
[22] A. G. Kusne, H. Yu, C. Wu, H. Zhang, J. Hattrick-
Simpers, B. DeCost, S. Sarker, C. Oses, C. Toher, S. Cur-
tarolo, and A. V. Davydov, On-the-fly closed-loop mate-
rials discovery via bayesian active learning, Nature Com-
munications 11, 5966 (2020).
[23] R. Shimizu, S. Kobayashi, Y. Watanabe, Y. Ando, and
T. Hitosugi, Autonomous materials synthesis by machine
learning and robotics, APL Materials 8, 11 (2020).
[24] P. Zhao, W. Su, R. Wang, X. Xu, and F. Zhang, Proper-
ties of thin silver films with different thickness, Physica
E: Low-dimensional Systems and Nanostructures 41, 387
(2009).
[25] H. Savaloni and M. Firouzi-Arani, Dependence of the op-
tical properties of uhv deposited silver thin films on the
deposition parameters and their relation to the nanos-
tructure of the films, Philosophical Magazine 88, 711
(2008).
[26] H. Reddy, U. Guler, K. Chaudhuri, A. Dutta, A. V. Kild-
ishev, V. M. Shalaev, and A. Boltasseva, Temperature-
dependent optical properties of single crystalline and
polycrystalline silver thin films, ACS Photonics 4, 1083
(2017).
[27] J. Choi, F. Cheng, J. W. Cleary, L. Sun, C. K. Dass,
J. R. Hendrickson, and X. Li, Optical dielectric constants
of single crystalline silver films in the long wavelength
range, Optical Materials Express 10, 693 (2020).
[28] B. D. Faeth, S.-L. Yang, J. K. Kawasaki, J. N. Nelson,
P. Mishra, C. T. Parzyck, C. Li, D. G. Schlom, and K. M.
Shen, Incoherent cooper pairing and pseudogap behav-
ior in single-layer FeSe/SrTio3, Phys. Rev. X 11, 021054
(2021).
[29] C. E. Rasmussen and C. K. I. Williams, Gaussian Pro-
cesses for Machine Learning (The MIT Press, 2006).
[30] J. R. Gardner, G. Pleiss, D. Bindel, K. Q. Weinberger,
and A. G. Wilson, Gpytorch: Blackbox matrix-matrix
gaussian process inference with gpu acceleration (2021),
arXiv:1809.11165 [cs.LG].
[31] S. Greenhill,
S. Rana,
S. Gupta,
P. Vellanki, and
S. Venkatesh, Bayesian optimization for adaptive exper-
imental design: A review, IEEE Access 8, 13937 (2020).
[32] D. P. Kingma and J. Ba, Adam: A method for stochastic
optimization (2017), arXiv:1412.6980 [cs.LG].
[33] M. M. Noack, K. G. Yager, M. Fukuto, G. S. Doerk,
R. Li, and J. A. Sethian, A kriging-based approach to
autonomous experimentation with applications to x-ray
scattering, Scientific Reports 9, 11809 (2019).
[34] M. M. Noack, G. S. Doerk, R. Li, J. K. Streit, R. A.
Vaia, K. G. Yager, and M. Fukuto, Autonomous materi-
als discovery driven by gaussian process regression with
inhomogeneous measurement noise and anisotropic ker-
nels, Scientific Reports 10, 17663 (2020).
[35] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Ve-
ness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K.
Fidjeland, G. Ostrovski, and S. Petersen, Human-level
control through deep reinforcement learning, Nature 518,
529 (2015).
[36] S. Kamthe and M. Deisenroth, Data-efficient reinforce-
ment learning with probabilistic model predictive control,
in International Conference on Artificial Intelligence and
Statistics (PMLR, 2018) pp. 1701–1710.