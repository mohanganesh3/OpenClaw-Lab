<!-- page 1 -->
Draft version May 11, 2026
Typeset using LATEX twocolumn style in AASTeX631
Advance warning of γ-ray blazar flares from Fermi-LAT light curves:
a strictly causal machine-learning backtest
Zahir Shah1 and Sikandar Akbar2
1Department of Physics, Central University of Kashmir, Ganderbal 191201, India
2Department of Physics, University of Kashmir, Srinagar 190006, India
ABSTRACT
Long-term Fermi-LAT monitoring makes it possible to ask whether a blazar light curve shows signs
of an upcoming flare before the flare becomes obvious in the γ-ray emission. We present a strictly
causal machine-learning framework for forecasting γ-ray blazar flares from 3-d binned LAT light curves.
Flare intervals are identified with Bayesian Blocks, and each light curve is sampled with 365-d trailing
windows from which 42 variability features are measured. We train separate WATCH and TRIGGER
models: WATCH predicts whether flare activity will appear within the next 90 d, while TRIGGER
predicts whether a new flare onset will occur within the next 45 d. To avoid temporal leakage, all
scaling, calibration, threshold selection, and validation use only the pre-cutoff data before MJD 60000.
We apply the method to the FSRQ 4FGL J1048.4+7143, using 13 bright blazars as auxiliary training
sources. Among logistic regression, polynomial logistic regression, and random forest classifiers, poly-
nomial logistic regression gives the strongest held-out WATCH performance, with ROC AUC = 0.891,
average precision = 0.396, and a block-permutation probability pperm = 0.006. At the selected WATCH
threshold, it recovers 18 of the 21 positive windows in the held-out WATCH set, corresponding to a
recall of 0.86. The same model also gives the best held-out TRIGGER ranking, with TRIGGER AUC
= 0.770 and TRIGGER AP = 0.123, although no reliable pre-onset TRIGGER alert is obtained. The
WATCH state appears before both held-out flare episodes, with final alerts 4.5 and 2.5 d before onset.
The corresponding broader WATCH-active periods begin 88.5 and 72.5 d before flare onset. These
results suggest that long-term Fermi-LAT light curves contain useful predictive information about the
build-up to blazar flares.
Keywords:
galaxies: active – galaxies: jets – gamma-rays: galaxies – methods: data analysis –
methods: statistical
1. INTRODUCTION
Blazars are a subclass of radio-loud active galactic
nuclei (AGN) whose relativistic jets are oriented at
small angles to our line of sight (Urry & Padovani
1995).
This geometric alignment causes the jet emis-
sion to be strongly Doppler boosted, making blazars
the most luminous sources of non-thermal radiation
in the Universe.
They emit across the electromag-
netic spectrum—from radio waves to GeV/TeV ener-
shahzahir4@gmail.com
Corresponding author: Sikandar Akbar
darprince46@gmail.com
gies (Blandford & Rees 1978; Ghisellini et al. 1993).
Blazars are broadly classified into BL Lacertae objects
(BL Lacs) and flat-spectrum radio quasars (FSRQs)
based on the rest-frame equivalent width of their op-
tical emission lines (Stickel et al. 1991; Padovani &
Giommi 1995). BL Lacs are further subdivided accord-
ing to the synchrotron peak frequency νsyn
peak into high-
synchrotron-peaked (HBL), intermediate-synchrotron-
peaked (IBL), and low-synchrotron-peaked (LBL) sub-
classes (Padovani & Giommi 1995; Ackermann et al.
2011).
The broadband spectral energy distribution
(SED) of blazars is characterized by two broad humps.
The low-energy hump, peaking from infrared to X-rays,
is attributed to synchrotron emission from relativistic
electrons in the jet.
The high-energy hump, peaking
arXiv:2605.07680v1  [astro-ph.HE]  8 May 2026


<!-- page 2 -->
2
from X-rays to γ-rays, is generally explained within
the one-zone leptonic framework as inverse Compton
(IC) scattering of either the synchrotron photon field
(synchrotron self-Compton, SSC; Maraschi et al. 1992;
Bloom & Marscher 1996) or external photon fields such
as the broad-line region (BLR) or dusty torus (external
Compton, EC; Dermer & Schlickeiser 1993; Sikora et al.
1994; Shah et al. 2017; Akbar et al. 2024). Hadronic
models, invoking proton synchrotron or photo-meson
cascades, provide alternative explanations for the high-
energy emission (Mannheim 1993; M¨ucke et al. 2003;
B¨ottcher et al. 2013).
The Fermi Large Area Telescope (LAT; Atwood et al.
2009) has revolutionized the study of blazar variabil-
ity since its launch in 2008.
Operating as an all-sky
monitor in the 0.1–300 GeV band, the LAT has now
provided nearly continuous multi-year light curves for
hundreds of blazars. These sources exhibit extreme flux
variability, with timescales as short as minutes in VHE-
emitting HBLs such as Mrk 501 (Albert et al. 2007)
and PKS 2155−304 (Aharonian et al. 2007). Such rapid
variability places tight constraints on the size and phys-
ical conditions of the emitting region. The Fermi-LAT
mission has revealed a wide range of variability phe-
nomena in blazars, including quasi-periodic oscillations
(QPOs) (Ackermann et al. 2015; Akbar et al. 2025; Ak-
bar 2026), log-normal flux behaviour (Shah et al. 2018),
orphan flares (MacDonald et al. 2015), spectral hystere-
sis (Katarzy´nski et al. 2005). Major γ−ray outbursts
in FSRQs such as 3C 454.3, 3C 279, PKS 1510−089, and
CTA 102 further highlight the broad dynamical range
accessible to Fermi-LAT observations.
Statistical studies of Fermi-LAT blazar light curves
show that the γ-ray flux distributions of blazars are bet-
ter described by a log-normal probability density func-
tion (Giebels & Degrange 2009; Shah et al. 2018; Malik
et al. 2025). Such behavior favors multiplicative vari-
ability processes and point to fluctuations that couple
the accretion flow to the jet, although the physical ori-
gin is not yet uniquely established (Biteau & Giebels
2012).
The variability amplitude is commonly quan-
tified through the fractional variability, Fvar (Vaughan
et al. 2003). Applied to large LAT blazar samples, these
diagnostics show that FSRQs are, on average, more vari-
able than BL Lacs at γ-ray energies (Abdo et al. 2010;
Nalewajko 2013; Shah et al. 2025); this trend also re-
mains evident in multi-timescale analyses of 3-, 7-, and
30-d Fermi-LAT light curves.
These statistical properties are not only useful for de-
scribing blazar variability, but also for asking whether
the light curve contains measurable signatures of an
approaching flare.
Understanding the temporal evo-
lution of blazar flares has direct implications for jet
physics, particle acceleration, and the underlying emis-
sion processes. From an observational perspective, γ-
ray flares are often accompanied by activity at other
wavelengths and therefore provide a strong motivation
for coordinated radio, optical, X-ray, and VHE cam-
paigns. This has led to continuous public monitoring
through the Fermi-LAT monitored-source light curves
and optical surveys such as ASAS-SN, as well as rapid
follow-up with pointed facilities such as Swift, NuSTAR,
and AstroSat. It has therefore become increasingly im-
portant to explore whether blazar flares can be antici-
pated, rather than identified only after they are already
in progress.
Despite these extensive monitoring and follow-up
efforts, most current flare-response strategies remain
largely reactive. They typically identify a source only
after it has already entered a bright state, or they rely on
a small number of multi-wavelength indicators to trig-
ger further observations. Such information is valuable,
but it does not by itself provide a statistically controlled
forecast of whether a flare is likely within a specified fu-
ture time window. This motivates a machine-learning
approach that searches the recent LAT light curve for
combinations of variability features that may appear be-
fore a flare. At the same time, flare forecasting is espe-
cially vulnerable to temporal data leakage, because la-
bels, feature scaling, probability calibration, or model
validation can inadvertently use information from after
the prediction time. The central question is whether the
recent variability history of a blazar contains enough
information to provide advance warning of an upcom-
ing flare. In this paper we therefore develop a strictly
causal machine-learning framework for forecasting γ-ray
blazar flares from 3-d binned Fermi-LAT light curves.
We apply this framework to 4FGL J1048.4+7143 as the
primary target source, while using 13 additional bright
blazars to provide auxiliary training data and improve
the statistical basis of the model.
The paper is organized as follows. In Section 2, we de-
scribe the Fermi-LAT data and the preparation of the
light curves.
Section 3 presents the methodology, in-
cluding Bayesian-Blocks flare identification, feature con-
struction, model training, calibration, and evaluation.
Section 4 gives the held-out backtest results and the in-
ferred warning times.
In Section 5, we discuss these
results in the broader context of blazar variability and
flare physics. Section 6 summarizes our main conclu-
sions.
2. DATA


<!-- page 3 -->
3
To test whether recent γ-ray variability can provide
advance warning of a flare, we require long, uniformly
processed light curves that sample multiple activity cy-
cles for each source.
The Fermi-LAT (Atwood et al.
2009) is well suited to this purpose.
It is a pair-
conversion γ-ray telescope covering the energy range
from ∼20 MeV to >300 GeV, with a peak effective area
of >8000 cm2 at normal incidence, an angular resolu-
tion (68% containment radius) of ∼0.◦6 at 1 GeV, and
an energy resolution of ∼10%. Operating in continuous
all-sky survey mode, the LAT scans the full sky every
∼3 hr and provides nearly uninterrupted long-term mon-
itoring of individual blazars. Over the mission lifetime,
this has yielded more than 17 years of continuous γ-ray
data, encompassing multiple distinct active episodes for
each source in our sample (Table 1).
We use light curves from the Fermi-LAT Light Curve
Repository (LCR; Abdollahi et al. 2023), a public
archive that provides uniformly processed 3-d, 7-d, and
30-d binned light curves for 1525 variable sources from
the Fourth Fermi-LAT Source Catalogue.
Each light
curve is derived from an unbinned maximum-likelihood
analysis of Pass 8 SOURCE-class events in the energy
range 100 MeV–100 GeV within a 12◦region of inter-
est. In this work, we use the 3-d binned light curves
of the 14 bright γ-ray blazars listed in Table 1.
For
each time bin, we extract the integrated photon flux F
and its uncertainty σF , the Test Statistic TS = 2∆ln L,
and the photon spectral index Γ with its uncertainty σΓ
when available. The 3-d binning provides a balance be-
tween temporal resolution and photon statistics: shorter
bins often do not contain enough counts for stable likeli-
hood fits except during bright flares, whereas longer bins
tend to smooth out sub-week variability that may carry
predictive information about an approaching flare. All
subsequent quality filtering and feature extraction are
applied to these 3-d binned light curves, as described in
Section 2.1.
2.1. Sample selection and quality cuts
We analyse a sample of 14 bright γ-ray blazars selected
from the fourth Fermi-LAT source catalogue and in-
cluded in the Fermi-LAT Light Curve Repository (LCR;
Abdollahi et al. 2023).
The light curves span 2008
August to 2026 March, and in this work we use 3-d
binned light curves.
Before Bayesian-Blocks segmen-
tation and feature extraction, each light curve is sub-
jected to a uniform set of quality cuts. We retain only
bins with TS ≥4, positive reported flux, and a finite
flux uncertainty.
In addition, a hard upper limit of
Fmax = 10−4 ph cm−2 s−1 is applied to remove isolated
outlier bins, which are likely to arise from processing ar-
tifacts or solar contamination. Very small reported flux
uncertainties can make the Bayesian-Blocks segmenta-
tion too sensitive to a few individual bins.
To avoid
this, we apply a minimum uncertainty of σF ≥0.10 f
|F|
in each segmentation run, where f
|F| is the median abso-
lute flux of the corresponding light-curve segment. This
floor does not replace the reported uncertainties alto-
gether; it only raises those bins whose reported errors
are smaller than the adopted minimum level. This pre-
vents a small number of formally high-precision bins
from driving spurious change points. The original mea-
surement uncertainties are retained without modifica-
tion for the subsequent rolling-window feature extrac-
tion. All 14 sources remain persistently detected over
multi-year baselines and exhibit peak-to-quiescent flux
ratios Rpk/q ≡Fpeak/Fq large enough to support flare-
classification training (Table 1).
2.1.1. Target source
The primary prediction target is 4FGL J1048.4+7143,
the flat-spectrum radio quasar (FSRQ) S5 1044+71, at
redshift z = 1.15 (Polatidis et al. 1995).
The source
is included in the Fermi-LAT monitored-source list and
has shown repeated LAT-reported episodes of enhanced
γ-ray activity (Ojha et al. 2013; Ojha & Carpen 2017).
It is a bright and strongly variable FSRQ with pro-
nounced multi-epoch γ-ray activity over the full Fermi
mission baseline.
During a major flare reported in
early 2017, the daily flux above 100 MeV reached (1.1 ±
0.2) × 10−6 ph cm−2 s−1 (Ojha & Carpen 2017). Its 3-
d binned light curve yields 1445 bins after quality cuts
(Section 2.1). The quiescent flux level, defined as the
median of the lowest 30 per cent of the training-period
flux distribution, is Fq ≃7.67 × 10−8 ph cm−2 s−1, and
the peak-to-quiescent ratio is Rpk/q = 14.5 (Table 1).
The long light curve of this source, spanning more than
17 years and containing multiple activity cycles, makes
it well suited for training and held-out evaluation of pre-
dictive models.
2.1.2. Auxiliary training sources
The remaining 13 sources listed in Table 1 are used
exclusively as auxiliary training data; no forecasts are
issued for these sources.
The sample includes mostly
FSRQs spanning a redshift range 0.1 ≲z ≲1.6. The
auxiliary sample spans a range of blazar classes, red-
shifts, and brightness levels. This allows the classifier
to learn from a broader variety of variability patterns
and spectral behavior.
At the same time, all of the
selected sources show strong and sustained γ-ray flar-
ing activity over the full Fermi mission baseline.
All
auxiliary sources are drawn from the Fermi-LAT LCR


<!-- page 4 -->
4
and satisfy the same quality criteria as the target source
(Section 2.1). Each source exhibits substantial peak-to-
quiescent flux variability, with Rpk/q ≡Fpeak/Fq values
ranging from 7.0 to 161.0 (Table 1), ensuring that el-
evated flux states are well represented in the training
data. The selection deliberately excludes sources with
very low variability amplitudes, to avoid biasing the clas-
sifier towards quiescent-dominated behavior that would
be uninformative for flare prediction.
The sources in our sample span more than an order
of magnitude in brightness. We therefore normalize all
flux-dependent features by the quiescent flux level of
each source, Fq, defined as the median of the lowest
30 per cent of its training-epoch flux distribution (Sec-
tion 3.3). This places the features on a common relative
scale and allows the classifier to learn shared variability
patterns rather than source-specific flux levels. The aux-
iliary sources also enlarge the training set and increase
its statistical diversity, providing many more examples
of pre-flare behavior than the target light curve alone.
The strictly causal train–test split used for the rolling
windows is described in Section 3.5.
3. METHODOLOGY
3.1. Overview
With the source sample, light curves, and quality
cuts now defined, we next describe how the forecasting
dataset is constructed and evaluated.
The full work-
flow is summarized in Figure 1. We first identify γ-ray
flare intervals using the Bayesian-Blocks analysis on the
training and full light curves. We then extract variabil-
ity features from rolling windows to build the super-
vised learning dataset.
This is followed by a strictly
causal train–test split, model training, and probabil-
ity calibration.
The WATCH and TRIGGER thresh-
olds are then determined using only the causal TRAIN-
score sequences. Finally, the models are evaluated on
the held-out target stream, including the warning-time
analysis. This causal separation is maintained through-
out the pipeline so that no information from the future
enters the training or evaluation steps.
3.2. Bayesian-Blocks flare identification
We identify flare intervals using the Bayesian-Blocks
(BB) method (Scargle et al. 2013), as implemented in
astropy (Astropy Collaboration et al. 2022).
This
method divides the light curve into a set of piecewise-
constant blocks while taking the reported flux uncertain-
ties into account. For the BB segmentation we adopt a
false-alarm probability of p0 = 0.05, which provides a
conservative balance between detecting real flux changes
and avoiding spurious change points. A block is classi-
Fermi-LAT light curves
3-d binned flux, flux error, photon index, and TS
target source plus 13 auxiliary bright blazars
Quality cuts and source reference levels
𝑇𝑆
≥
4, 𝐹
>
0, finite flux errors; define the
TRAIN quiescent level 𝐹q for feature normalisation
Bayesian-Blocks flare intervals
TRAIN-BB: 𝑡≤𝑡cut = MJD 60000 for training labels
FULL-BB: complete target light curve for held-out scoring only
Rolling-window samples and forecast labels
𝑊
= 365 d, step = 7 d, 42 features per window
WATCH: flare overlaps the next 90 d by at least 3 d;
TRIGGER: flare onset within the next
45 d, excluding active-flare windows
Strictly causal split
TRAIN: 𝑇end ≤MJD 59910;
TEST: 𝑇end > MJD 59910
MJD 59910 = 𝑡cut −90 d prevents horizon leakage
TRAIN-only fitting
median imputation;
linear-model scaling;
separate WATCH/TRIGGER
LR, PLR, and RF fits;
Platt calibration; causal
TRAIN-score thresholds
Held-out evaluation
FULL-BB labels; WATCH
AUC, AP, BSS, 𝐹2, MCC;
TRIGGER AUC, AP;
block tests, bootstrap CI
Operational outputs
probability timeline with causal 7-window trailing display mean;
WATCH and TRIGGER score streams combined into
Normal/WATCH/TRIGGER alert states
Lead-time analysis
raw, unsmoothed probabilities; two-window persistence;
WATCH lookback 90 d, TRIGGER lead capped at 45 d
Figure 1.
Schematic of the strictly causal blazar flare-
forecasting pipeline. The light curves first pass through qual-
ity cuts, and the TRAIN-only quiescent level Fq is defined
for later normalization of flux-dependent features. Bayesian
Blocks are then computed in two ways: TRAIN-BB uses
only data up to tcut = MJD 60000 for training labels, while
FULL-BB is used only to score the held-out target stream.
Rolling 365-d windows with a 7-d step are assigned two fu-
ture targets: a WATCH label based on flare overlap within
the next 90 d and a TRIGGER label based on flare onset
within the next 45 d.
The train/test boundary is placed
at Tboundary = tcut −90 d = MJD 59910, so that no train-
ing WATCH horizon reaches into the held-out period. Each
classifier family is then fitted separately for WATCH and
TRIGGER, calibrated on a trailing TRAIN tail, and thresh-
olded using causal TRAIN-score sequences before the fixed
pipeline is evaluated on the held-out windows.
fied as a flare only if it satisfies both a flux and a duration
condition. Its mean flux must satisfy
ˆµb ≥max(ˆµ70th%, µq + 3σq) ,
(1)
where ˆµb is the mean flux of the bth block. The quantity
ˆµ70th% is the 70th percentile of the mean fluxes of all
blocks returned by that BB run. To define the quiescent
level, we first identify the blocks whose mean fluxes lie in
the lowest 40 per cent of that same set of block means.
The quantities µq and σq are then taken as the mean and
standard deviation of those low-flux blocks. In addition,
the block duration must satisfy ∆tb ≥9 d. Adjacent
flare blocks separated by ≤9 d are merged into a single
flare interval, so that multi-peaked outbursts are not
split into several smaller events. Blocks spanning large
gaps in the light curve are not classified as flare blocks.
To preserve temporal causality, we perform two separate

[CAPTION] Figure 1.
Schematic of the strictly causal blazar flare-


<!-- page 5 -->
5
Table 1. Properties of the 14 Fermi-LAT blazar light curves used in this study. Nbins is the number of 3-d bins surviving
quality cuts; Nflare is the number of training-epoch bins classified as flaring by Bayesian Blocks; Fq is the median flux of the
lowest 30 per cent of training-epoch bins; and Rpk/q ≡Fpeak/Fq. The target source is highlighted in bold.
4FGL name
Common name
Class
z
Role
Nbins
Nflare
Fq (10−8 ph cm−2 s−1)
Rpk/q
J1048.4+7143
S5 1044+71
FSRQ
1.15
Target+Train
1445
186
7.67
14.5
J1512.8−0906
PKS 1510−089
FSRQ
0.361
Train
1842
79
28.82
31.9
J1224.9+2122
W Com (ON 231)
IBL
0.102
Train
1168
33
13.10
57.3
J0904.9−5734
PKS 0903−573
FSRQ
—
Train
1241
30
9.89
64.6
J1256.1−0547
3C 279
FSRQ
0.536
Train
1945
37
29.10
53.6
J2253.9+1609
3C 454.3
FSRQ
0.859
Train
1996
50
37.38
153.3
J0538.8−4405
PKS 0537−441
FSRQ
0.892
Train
1957
263
11.20
8.9
J0739.2+0137
PKS 0736+017
FSRQ
0.189
Train
1313
116
11.63
20.1
J1159.5+2914
B2 1156+295
FSRQ
0.729
Train
1668
121
11.18
161.0
J0730.3−1141
PKS 0727−115
FSRQ
1.591
Train
1428
247
13.10
7.0
J1310.5+3221
B2 1308+326
ISP
0.997
Train
1076
102
7.08
44.9
J1443.9+2501
PKS 1441+25
FSRQ
0.939
Train
797
79
4.56
25.0
J1522.1+3144
B2 1520+31
FSRQ
1.487
Train
1406
217
14.30
11.2
J0403.9−3605
PKS 0402−362
FSRQ
1.417
Train
1410
73
10.90
45.1
BB runs for each source. The TRAIN-BB run uses only
data up to the cutoff time, tcut = MJD 60000, and its
flare intervals are used to define the training labels. The
FULL-BB run uses the complete light curve and is used
only for the held-out evaluation. This separation ensures
that information from after the cutoff does not enter the
training step.
3.3. Feature engineering
Using the cleaned light curves, we construct the
machine-learning dataset with rolling windows. For each
window ending at Tend, we use the preceding 365 d of
data, i.e. the interval [Tend −W, Tend] with W = 365
d, and shift the window forward in steps of 7 d. Only
windows containing at least 30 valid data points are re-
tained. From each window we compute a 42-dimensional
feature vector that summarizes the variability behavior
of the source. Because the blazars in our sample span
a wide range of flux levels, all flux-dependent features
are normalized by the quiescent flux of that source, Fq.
Here Fq is defined from the training epoch as the median
of the lowest 30 per cent of the flux distribution. This
places the flux-dependent features on a common relative
scale and allows direct comparison across sources. The
42 features are grouped into the following categories.
Flux distribution and tail statistics. —The first group con-
tains 16 features and describes the overall shape of the
flux distribution within each window. We first measure
the average spacing between valid data points, together
with the mean flux and the standard deviation of the
flux, both normalized by Fq. We also include the two
parameters of a log-normal fit, (µln, σln), along with the
skewness and kurtosis, so that the width and asymme-
try of the distribution are captured.
To describe the
bright end of the distribution, we add several tail mea-
sures. One feature gives the fraction of flux values in
the current window that lie above the 95th-percentile
flux level of the same source during its pre-cutoff train-
ing epoch. We also measure the fraction of points in the
current window that lie more than 2σ above the window
mean.
In addition, we include the ratios p95/median
and p99/⟨F⟩, where p95 and p99 are the 95th and 99th
percentiles of the flux distribution within the current
window. We further include the interquartile range nor-
malized by the median, the peak-to-mean ratio, the Gini
coefficient, and the Kolmogorov–Smirnov statistic and
p-value for the log-normal fit. Together, these features
describe the typical flux level, the spread of the distri-
bution, and the strength of the high-flux tail.
Variability amplitude and temporal evolution. —The second
group contains 15 features and is designed to describe
how strongly the source varies, and whether that vari-
ability becomes stronger towards the end of the win-
dow. We first use two standard measures of the over-
all variability amplitude: the fractional variability Fvar
(Vaughan et al. 2003) and the normalised excess vari-
ance σ2
NXS (Edelson et al. 2002). We then measure the
long-term behaviour of the light curve through the lin-
ear slope of the normalised flux F/Fq and the slope of
the log-flux ln F. The latter is useful for capturing mul-
tiplicative changes in the flux.
To check whether the
source is brightening towards the end of the window,
we compare the mean flux in the first and last 90, 45,

[CAPTION] Table 1. Properties of the 14 Fermi-LAT blazar light curves used in this study. Nbins is the number of 3-d bins surviving


<!-- page 6 -->
6
and 30 d of the same window. We also measure the flux
slope over the final 30 and 45 d, and the fraction of bins
above 1.5Fq over these same recent intervals. To follow
slower changes across the full 365-d window, we divide
the window into four equal parts and measure how the
mean flux, Fvar, and the log-normal width σln change
across these sub-intervals. Finally, we include the time
since the most recent TRAIN-BB flare onset as a sim-
ple measure of possible recurrence behavior. Together,
these features describe both the strength of the variabil-
ity and its temporal build-up within the window.
Temporal timing and periodicity. —The third group con-
tains 4 features and describes the time structure of the
variability. We first use the Lomb–Scargle periodogram
to search for any preferred variability timescale within
the window. From this, we take three quantities: the
highest periodogram power, the period at which this
maximum occurs, and the corresponding false-alarm
probability. We then use the first-order structure func-
tion to measure how the flux variations grow with time
lag, and include its slope as an additional feature. To-
gether, these quantities describe whether the variability
shows a characteristic timescale and how strongly the
flux changes are correlated across time.
Spectral evolution. —The final group contains 7 features
and describes how the photon spectral index, Γ, evolves
within each window. We first measure the mean value of
Γ, its standard deviation, and its linear slope with time.
We also include the difference between the mean spectral
index in the current window and the quiescent spectral
index of the same source, where the quiescent value is
defined from the training epoch. To examine whether
the spectrum changes towards the end of the window, we
further compare the mean Γ values in the first and last
90, 45, and 30 d of the same window. Together, these
features are intended to capture spectral hardening or
softening before major γ-ray activity.
3.4. Label definition
Once the 42 features have been computed for each
rolling window, the next step is to assign the future la-
bel that the model is asked to predict.
Each sample
is based only on the past 365-d interval ending at Tend,
but its label is defined from what happens after Tend. In
this work, we assign two future labels to each window:
a WATCH label and a TRIGGER label. The WATCH
label, YW, is designed to identify windows that are fol-
lowed by flare activity within the next 90 d. For this pur-
pose, we define the future horizon as (Tend, Tend +90 d].
A window is assigned YW = 1 if any Bayesian-Blocks
flare interval overlaps this future horizon by at least
∆min = 3 d; otherwise it is assigned YW = 0.
The
minimum-overlap condition avoids labelling a window
as positive when only a very small decaying part of a
flare enters the horizon. The TRIGGER label, YT, is
more restrictive and is intended to capture the start of
a flare rather than flare activity in general.
Here we
use a shorter, onset-focused horizon of 45 d.
A win-
dow is assigned YT = 1 if a flare onset occurs within
(Tend, Tend + 45 d], and YT = 0 otherwise.
To keep
this task focused on advance warning, windows that al-
ready lie inside an ongoing flare interval are not counted
as trigger-positive, even if the same flare began within
the next 45 d. For model training, both WATCH and
TRIGGER labels are derived from the TRAIN-BB flare
intervals and flare onsets, constructed using only data
up to the cutoff time tcut = MJD 60000. For the held-
out evaluation of the target source, the same two labels
are recomputed retrospectively from the FULL-BB seg-
mentation of the complete light curve. These held-out
labels are used only for performance evaluation. They
are not used for model training, probability calibration,
or threshold selection.
3.5. Strictly causal train–test split
Once the WATCH and TRIGGER labels have been
defined, the next step is to separate the rolling-window
samples into training and held-out sets.
This step is
especially important in a forecasting problem, because
leakage can arise even when the input window itself uses
only past data: the label is still defined by what happens
after the window end time. To keep the analysis causal,
we must consider not only the past data used as input,
but also the next 90 d used to assign the WATCH label.
The temporal cutoff is set at tcut = MJD 60000. The
WATCH label is based on whether a flare occurs within
the next H = 90 d. To ensure that no training label de-
pends on information from after the cutoff, we exclude
any window whose 90-d WATCH horizon extends be-
yond tcut. The corresponding boundary in window end
time is therefore
Tboundary = tcut −H = MJD 60000 −90 = MJD 59910.
(2)
Training windows satisfy Tend ≤Tboundary, so the full
90-d WATCH horizon of every training sample remains
before the cutoff. All windows with Tend > Tboundary are
placed in the held-out set. This implies that some held-
out windows can still end before MJD 60000, but they
are excluded from training because their future WATCH
horizon extends beyond the cutoff.
The shorter 45-d
TRIGGER horizon is automatically contained within
the same split, so no TRIGGER training label depends
on post-cutoff information either. In the code, this split


<!-- page 7 -->
7
is applied directly through the window end time Tend.
The training set is built from all sources using windows
with Tend ≤MJD 59910, whereas the post-boundary
windows of the target source are kept separate for the
final causal backtest.
3.6. Machine learning classification
We now train the machine-learning models that use
the 42-dimensional feature vectors to estimate the prob-
ability of a future flare. Because WATCH and TRIG-
GER represent two different forecasting tasks, each
model family is trained separately for the two labels.
The WATCH model returns the probability ˆpW
=
P(YW = 1 | x), while the TRIGGER model returns
ˆpT = P(YT = 1 | x). We use a small set of supervised
classifiers that span a range of model complexity, from
simple linear models to a non-linear tree-based model.
For the linear classifiers, the input features are standard-
ized to zero mean and unit variance before fitting. Both
WATCH and TRIGGER are class-imbalanced problems,
with many more negative windows than positive ones.
To account for this, we use balanced class weights in the
logistic regression, polynomial logistic regression, and
random-forest models.
Logistic Regression (LR). —We use logistic regression as
the baseline linear classifier.
In this model, the flare
probability is obtained from a weighted sum of the 42
input features:
P(Y = 1|x) = σ(w⊤x + b),
σ(z) =
1
1 + e−z .
(3)
Here x is the 42-dimensional feature vector of a given
window, w is the set of learned weights, and b is the
intercept term. The quantity w⊤x + b gives the linear
model score, and the sigmoid function σ(z) converts this
score into a value between 0 and 1. This value is inter-
preted as the predicted probability that the window be-
longs to the positive class, i.e. either WATCH-positive
or TRIGGER-positive, depending on the task being fit-
ted. The model is fitted with L2 regularization (C = 1)
using the lbfgs solver.
Polynomial Logistic Regression (PLR). —A purely linear
model can miss cases where two or more features become
important only when they vary together. To account
for this, we expand the original 42 input features by
adding all second-order terms, including squared terms
(x2
i ) and pairwise products (xixj). This polynomial ex-
pansion retains the original linear terms as well. The
total number of features therefore increases from 42 to
42 + 42(43)/2 = 945. Because this expanded space is
much larger, we retain only the 50 most informative
features using ANOVA F-score ranking (SelectKBest;
Pedregosa et al. 2011). A logistic- regression model is
then fitted in this reduced feature space with stronger
L2 regularization (C = 0.05) to limit overfitting. This
model can capture simple feature interactions that are
missed by standard logistic regression, for example when
increased variability and spectral change occur together
before a flare.
Random Forest (RF). —Random forest is a non-linear en-
semble model made up of 200 decision trees (Breiman
2001). Each tree is trained on a bootstrap resample of
the training windows. At each split, the tree examines
only a random subset of the input features, rather than
all 42 features at once. This reduces the similarity be-
tween the trees and makes the final model more robust.
The predicted flare probability is then obtained by aver-
aging the probabilities returned by all trees. To account
for the smaller number of positive windows, we use bal-
anced class weights during training. To reduce overfit-
ting, we require at least 5 samples in each terminal leaf.
Unlike the linear models, RF does not need an explicit
expansion of the feature space. It can learn non-linear
relations and interactions between features directly from
the data. We also compute the out-of-bag score during
training. This uses the samples left out of the bootstrap
resampling for each tree and provides an internal check
of model consistency (Breiman 1996).
3.7. Probability calibration and threshold selection
Before the models are evaluated on the held-out tar-
get stream, their raw scores must be converted into cal-
ibrated probabilities and then into practical alert de-
cisions. Both steps are performed using only the pre-
cutoff TRAIN data, so that the full procedure remains
causal. The raw output of a classifier is not always a
reliable probability.
For this reason, we calibrate the
WATCH and TRIGGER models before choosing the fi-
nal alert thresholds. We use the sigmoid calibration of
Platt (1999), in which a raw model score s is converted
into a calibrated probability
ˆp = σ(As + B),
(4)
where σ is the logistic sigmoid function, and A and B
are determined from part of the TRAIN sequence. This
step does not change the time ordering of the samples;
it only adjusts the probability scale so that the model
output is more consistent with the observed fraction of
positive windows.
To keep this step causal, the cali-
bration uses only the most recent part of TRAIN. We
first keep the last 20 per cent of the TRAIN sequence as
a calibration tail. If this tail contains too few positive


<!-- page 8 -->
8
windows, it is expanded in steps of 5 per cent, up to a
maximum of 60 per cent, until it contains both positive
and negative windows. The earlier part of TRAIN is
then used to fit the base model, while the trailing tail
is used only to adjust its probability scale. In this way,
the calibration is based entirely on pre-cutoff data and
is anchored to the most recent behavior before the cut-
off. If the calibration split still does not contain both
classes, the model is left uncalibrated.
For threshold selection, we then need one probabil-
ity for each TRAIN window, kept in time order. For
the earlier TRAIN windows, we mimic real forecast-
ing:
the model is first trained on the oldest part of
the TRAIN sequence and used to predict the next set
of windows. The training range is then extended for-
ward in time, and the same procedure is repeated. For
the final calibration tail, we use the calibrated prob-
abilities directly. Joining these two parts gives a sin-
gle causal TRAIN probability sequence for each task.
These sequences are used only for threshold selection.
The WATCH and TRIGGER models return a probabil-
ity for every window, but in practice a probability alone
is not enough: we also need a rule for deciding when
it is high enough to issue an alert. We therefore deter-
mine separate thresholds for the WATCH and TRIG-
GER tasks from their own causal TRAIN probability
sequences. For WATCH, these causal TRAIN probabil-
ities are first constructed separately within each source
and are then pooled, so that each blazar contributes
its own time-ordered pre-cutoff WATCH behavior be-
fore a common threshold is chosen. For TRIGGER, the
threshold is still determined from the combined causal
TRAIN trigger sequence. The two tasks are treated sep-
arately because they are designed for different purposes:
WATCH is intended to identify an elevated-risk state
before flare activity, whereas TRIGGER is intended to
identify a more selective pre-onset alert.
WATCH threshold. —To turn the WATCH probability
into a practical alert, we must choose a single prob-
ability threshold, τW.
We test candidate thresholds
from 0.05 to 0.95 in steps of 0.01.
At a trial thresh-
old τ, any window with ˆpW(Tend) ≥τ is classified
as WATCH-positive. These predictions are then com-
pared with the true WATCH labels of the TRAIN win-
dows. For each trial threshold, we compute the preci-
sion P(τ) and recall R(τ). Here precision is the fraction
of predicted WATCH-positive windows that are truly
WATCH-positive, while recall is the fraction of true
WATCH-positive windows that are successfully recov-
ered. We then evaluate the recall-weighted F2 score,
F2(τ) =
5 P(τ) R(τ)
4 P(τ) + R(τ),
(5)
and choose the threshold that maximises it:
τW = arg max
τ
F2(τ).
(6)
We use the F2 score because it gives more weight to re-
call than to precision. This is appropriate for WATCH,
where missing a genuine pre-flare window is usually
more costly than issuing an extra early warning.
TRIGGER threshold and alert states. —The TRIGGER
threshold is selected in a similar way, but with a more
conservative goal. Whereas WATCH is intended to iden-
tify an elevated-risk state, TRIGGER is meant to pro-
vide a smaller number of more selective pre-onset alerts.
For this reason, we again test thresholds from 0.05 to
0.95 in steps of 0.01, but now use the onset-based TRIG-
GER probabilities and labels. At each trial threshold,
windows with ˆpT(Tend) ≥τ are classified as TRIGGER-
positive and compared with the true TRIGGER labels of
the TRAIN windows. For every trial threshold, we com-
pute the precision, recall, the precision-weighted F0.5
score, and the alert fraction.
Here the alert fraction
means the fraction of TRAIN windows that would be
placed in the TRIGGER state at that threshold. We
use F0.5 rather than F2 because TRIGGER is intended
to be more selective, so precision is more important here
than recall. Not every threshold is accepted. We require
the precision to be at least 0.25 and the alert fraction
to remain below 0.20. Among the thresholds that sat-
isfy these conditions, we choose the one that gives the
best overall TRIGGER behavior. In the code, this is
done with a score that is dominated by F0.5, with a
small bonus for higher precision and a small penalty for
placing too many windows in TRIGGER. If no thresh-
old satisfies all of the constraints, the code falls back to
the best available threshold, giving highest priority to
precision and then to the remaining ranking measures.
Once the WATCH and TRIGGER thresholds, τW and
τT, have been fixed, each window is assigned to one
of three operational states. A window is first checked
against the TRIGGER model. If ˆpT(Tend) ≥τT, it is
assigned to the TRIGGER state. If this condition is not
met, the WATCH model is then checked. A window is
assigned to WATCH when ˆpW(Tend) ≥τW, and to Nor-
mal otherwise. In this way, TRIGGER takes priority
over WATCH. For the timeline figures, we show a causal
seven-window trailing mean of the WATCH and TRIG-
GER probabilities in order to make the visual trends
easier to see. However, the threshold selection, perfor-
mance metrics, and lead-time calculations are all based
on the raw calibrated probabilities.
3.8. Model evaluation


<!-- page 9 -->
9
After calibrating the WATCH and TRIGGER proba-
bilities and fixing the final thresholds from the TRAIN
data, we evaluate the models on the held-out target win-
dows. This tells us how well the method performs on
data that were not used in model fitting, calibration, or
threshold selection. We examine the results from four re-
lated viewpoints: how well the model ranks flare-related
windows above non-flare windows, how well the selected
WATCH threshold works as a binary forecast, how reli-
able the WATCH probability scale is, and how much ad-
vance warning the WATCH and TRIGGER alerts pro-
vide before flare onset. Because the onset-based TRIG-
GER task contains fewer positive examples, we summa-
rize it mainly through held-out area under the receiver-
operating characteristic curve (ROC AUC) and the Av-
erage Precision (AP) and the observed warning behav-
ior.
To evaluate ranking performance, we use the ROC
AUC and the AP, ROC AUC measures how well the
model assigns higher scores to true positive windows
than to true negative windows, without assuming any
particular probability threshold.
AP summarizes the
corresponding precision–recall curve, and is especially
useful here because positive windows are much rarer
than negative ones (Saito & Rehmsmeier 2015).
We
report these two ranking metrics for both the WATCH
and TRIGGER probabilities.
To check whether the WATCH probabilities are mean-
ingful as probabilities, and not only as ranking scores,
we use the Brier Skill Score (BSS). This compares the
average probability error of the model with that of a sim-
ple baseline forecast that assigns the same probability to
every window, equal to the fraction of WATCH-positive
windows in the evaluation set, ¯y. The Brier score of the
model is BSmodel = (ˆp −y)2, while the reference forecast
has BSclim = ¯y(1 −¯y). The Brier Skill Score is then
BSS = 1 −BSmodel
BSclim
= 1 −(ˆp −y)2
¯y(1 −¯y).
(7)
Here y is the true binary WATCH label, ˆp is the pre-
dicted WATCH probability, and ¯y is the fraction of
WATCH-positive windows in the evaluation set. A pos-
itive BSS means that the model probabilities are better
than a simple baseline that assigns the same WATCH
probability to every window, whereas a negative BSS
means that the predicted probabilities should be inter-
preted with caution, because their numerical values are
less reliable than a simple constant baseline forecast.
We also evaluate the WATCH model (classifier trained
using the WATCH labels) after converting the probabil-
ities into binary WATCH alerts at the selected thresh-
old. This conversion is done by comparing the WATCH
probability of each window with the selected WATCH
threshold, and assigning the window to the WATCH
state when the probability exceeds that threshold. For
this purpose, we report the F2 score and the Matthews
Correlation Coefficient (MCC). The F2 score is useful
here because it gives more weight to recall than to pre-
cision, which is appropriate when missing a flare-related
window is more costly than issuing an extra alert. We
also report MCC, which uses all four entries of the con-
fusion matrix and therefore gives a balanced summary
of binary classification performance:
MCC =
TP · TN −FP · FN
p
(TP + FP)(TP + FN)(TN + FP)(TN + FN)
,
(8)
where TP, TN, FP, and FN are the numbers of true
positives, true negatives, false positives, and false nega-
tives, respectively. MCC is especially useful because it
remains informative even when the numbers of positive
and negative windows are very different.
The TEST windows are not independent, because
each 365-d rolling window is separated from the next
by only 7 d and therefore overlaps heavily with it. As a
result, neighboring TEST windows share about 98 per
cent of their input data, so standard significance tests
that assume independent samples are not suitable. To
test whether the held-out WATCH ROC AUC could
arise by chance, we use a block-shuffle permutation test
(Politis & Romano 1994; Good 2013). In this test, the
WATCH labels in the TEST sequence are divided into
contiguous blocks of length LB = 9 windows, corre-
sponding to about 63 d. These blocks are then randomly
reordered, while the model scores are kept fixed. This
preserves the short-range grouping of positive labels, but
removes their true association with the predicted proba-
bilities. For each shuffled realization, we recompute the
WATCH AUC and compare it with the observed held-
out value. The empirical p-value is
p = #{AUCperm ≥AUCobs} + 1
Nperm,eff + 1
,
(9)
where AUCobs is the observed held-out WATCH AUC,
AUCperm is the AUC obtained after block shuffling, and
Nperm,eff is the number of valid shuffled realizations. In
this work we use 1000 permutation trials. The Phipson–
Smyth correction is included so that the p-value remains
well defined even if none of the shuffled realizations ex-
ceeds the observed AUC (Phipson & Smyth 2010).
We estimate the uncertainty of the held-out WATCH
AUC with a circular block bootstrap (K”unsch 1989;
Politis & Romano 1992). This is again needed because
the TEST windows are not independent. In each boot-
strap realization, we rebuild a TEST-like sequence by


<!-- page 10 -->
10
drawing blocks of length LB with replacement, while
allowing the blocks to wrap around at the ends of the
time series. We then recompute the WATCH AUC for
each resampled sequence. The 95 per cent confidence in-
terval is taken from the 2.5th and 97.5th percentiles of
the bootstrap AUC distribution, using the fiducial block
length LB = 9 windows and 500 bootstrap resamples.
As a sensitivity check, we repeat the same calculation
for block lengths of 5, 9, and 15 windows, using 200
resamples for each case.
In addition to the held-out test, we also assess model
behavior within the pre-cutoff data by performing a
purged walk-forward validation on the combined TRAIN
sample (de Prado 2018).
In each fold, the model is
trained on an earlier segment of the TRAIN sequence
and tested on a later segment, so that the time ordering
is preserved. A purge gap of Lpurge = 6 windows (about
42 d) is left between the training and validation seg-
ments to reduce leakage from neighboring windows. In
the present implementation, the initial training segment
contains at least 120 windows, each validation chunk
contains 90 windows, and the walk-forward procedure
advances in 90-window steps. The validation predictions
from all folds are then combined to obtain a pooled out-
of-fold AUC, which provides a leakage-resistant sum-
mary of training-period performance.
Finally, we examine whether the WATCH and TRIG-
GER alerts provide useful advance warning before a
flare. For each flare onset associated with the held-out
target period, we search the raw probability sequence
for valid WATCH or TRIGGER threshold crossings be-
fore the onset. To avoid counting isolated one-window
fluctuations, a crossing is accepted only if the threshold
remains exceeded for at least two consecutive windows.
For WATCH, the main search is carried out within 90
d before the flare onset. For TRIGGER, the same pro-
cedure is used, but the maximum allowed lead time is
limited to 45 d, making TRIGGER a more conservative
near-term alert. For each detected case, we record three
lead-time measures: the earliest valid crossing, the lat-
est valid crossing, and the start of the alert episode con-
taining the final valid crossing. In the last case, nearby
alert episodes are linked if they are separated by fewer
than five consecutive non-alert windows.
As a sensi-
tivity check, we repeat the search with 30-d and 180-d
lookback windows. Because the number of held-out flare
onsets is small, these lead times are interpreted as an op-
erational summary rather than as a precise population
measurement.
4. RESULTS
After defining the WATCH and TRIGGER labels,
training the models, calibrating the probability outputs,
and fixing the alert thresholds using only the TRAIN
data, we now turn to the held-out target-source results.
We first examine the Bayesian-Blocks flare segmentation
used to define the labels, then compare the classifiers on
the held-out WATCH and TRIGGER tasks, and finally
study how the best-performing model behaves as a prac-
tical alert system.
Figure 2 compares the Bayesian-Blocks flare segmen-
tation of the target source obtained in the TRAIN-
BB and FULL-BB runs.
The TRAIN-BB segmenta-
tion is derived using only the data before the cutoff at
tcut = MJD 60000, and is used to assign the training
labels. The FULL-BB segmentation is derived from the
complete light curve and is used only to evaluate the
held-out forecasts.
In this way, the training step re-
mains causal, while the held-out predictions can still be
compared with the flare activity that actually occurred.
With the training and scoring segmentations defined,
we next evaluate the forecasting performance on the
held-out target-source windows. The held-out set con-
sists of rolling input windows from the target light curve.
Each window spans 365 d, is shifted forward in steps
of 7 d, and is identified by its end time Tend.
The
held-out set contains all target-source windows with
Tend > Tboundary, giving 172 rolling-window samples
with end times spanning MJD 59911.5–61108.5. These
windows are not used in model fitting.
They are la-
belled only for evaluation, using the FULL-BB flare
intervals and flare onsets defined in Section 3.4.
Un-
der the WATCH definition, 21 held-out windows are
positive and 151 are negative. Under the onset-based
TRIGGER definition, 9 windows are positive and 163
are negative.
Table 2 summarizes the held-out per-
formance of the three classifiers. For each model, the
WATCH AUC and WATCH AP are computed from the
90-d WATCH probabilities and measure how well the
21 WATCH-positive windows are ranked above the 151
WATCH-negative windows. The TRIGGER AUC and
TRIGGER AP are computed separately from the 45-
d onset-based TRIGGER probabilities.
The WATCH
AUC confidence interval, the block-permutation p-value,
and the Brier Skill Score are then obtained with the pro-
cedures described in Section 3.8. These procedures are
chosen to account for the strong overlap between neigh-
boring rolling windows.
Polynomial
logistic
regression
(PLR)
gives
the
strongest held-out WATCH performance, with WATCH
ROC AUC = 0.891 and WATCH AP = 0.396.
This
means that, in the held-out target period, PLR is

[CAPTION] Figure 2 compares the Bayesian-Blocks flare segmen-

[CAPTION] Table 2 summarizes the held-out per-


<!-- page 11 -->
11
(a) TRAIN-BB segmentation using only data with
t ≤MJD 60000.
(b) FULL-BB segmentation of the complete light curve.
Figure 2. Bayesian-Blocks diagnostics for the target source. In each panel, the upper plot shows the 3-d Fermi-LAT light
curve together with the Bayesian-Blocks representation. Each semi-transparent shaded strip spans one Bayesian block in time
and extends from zero flux to the block-mean flux: red/pink strips mark blocks classified as flaring, whereas light-blue strips
mark non-flaring blocks. The red dashed line is the flare-selection threshold from equation (1); its numerical value is listed in
the panel legend and differs slightly between TRAIN-BB and FULL-BB because the threshold is computed separately for each
segmentation. The lower plot shows the block-mean flux against block mid-time, and the red points identify the flare blocks
used to define flare intervals.
the most effective at assigning higher WATCH prob-
abilities to true WATCH-positive windows than to
WATCH-negative ones.
Its block-permutation proba-
bility, pperm = 0.006, further shows that this rank-
ing performance is unlikely to arise by chance.
The
same model also gives the strongest held-out TRIG-
GER ranking, with TRIGGER AUC = 0.770 and TRIG-
GER AP = 0.123.
Random forest gives the highest
pooled out-of-fold WATCH AUC on the TRAIN data.
This means that, within the pre-cutoff walk-forward
validation, RF performs best at separating WATCH-
positive and WATCH-negative windows. However, PLR
also gives a positive held-out WATCH Brier Skill Score
(BSS = 0.115), whereas the held-out RF Brier Skill
Score is negative (BSS = −0.204). This means that PLR
not only gives the strongest held-out WATCH ranking,
but also produces WATCH probabilities that are mod-
estly better than a simple constant baseline forecast. In
other words, RF still performs well within the TRAIN
period, but PLR gives the strongest overall discrimina-
tion on the unseen held-out data. For this reason, PLR
remains the best overall model for the main forecasting
task in the present backtest.
For completeness, we also construct a soft-voting
WATCH ensemble by averaging the raw calibrated
WATCH probabilities from logistic regression, polyno-
mial logistic regression, and random forest on each held-
out window. The WATCH AUC and WATCH AP of the
ensemble are then computed from this averaged proba-
bility sequence, rather than by averaging the individual
model AUC values. On the held-out WATCH task, the
ensemble reaches AUC = 0.771 and AP = 0.306, so
it does not outperform PLR. Nevertheless, the ensem-
ble timeline remains useful as a comparison, because it
highlights the same broad elevated-risk intervals before
the two held-out flare episodes (Fig. 3). In our analysis,
the ensemble is constructed only for the WATCH prob-
abilities; no corresponding ensemble is defined for the
TRIGGER task.
Since PLR gives the strongest held-out WATCH per-
formance, we examine its diagnostic results in more de-
tail.
For WATCH, the threshold is fixed in advance
from the pooled source-local causal TRAIN WATCH
scores by maximising the F2 score, as described in Sec-
tion 3.7. For PLR, this gives τW = 0.12. We then ap-
ply this fixed threshold to the raw calibrated WATCH
probabilities of the held-out target windows, so that

[CAPTION] Figure 2. Bayesian-Blocks diagnostics for the target source. In each panel, the upper plot shows the 3-d Fermi-LAT light


<!-- page 12 -->
12
Table 2. Summary of model performance on the combined TRAIN sample and on the held-out target-source TEST set.
Model
WATCH pooled OOF AUC
WATCH AUC
WATCH AUC 95% CI
pperm
WATCH AP
WATCH BSS
TRIGGER AUC
TRIGGER AP
Logistic Regression
0.823
0.687
0.421–0.939
0.160
0.247
−0.161
0.558
0.079
Polynomial Logistic Regression
0.841
0.891
0.802–0.980
0.006
0.396
0.115
0.770
0.123
Random Forest
0.859
0.750
0.500–0.954
0.094
0.284
−0.204
0.763
0.110
Notes. Each row reports the held-out performance of one classifier. The TEST set contains 172 rolling windows from the target source
with Tend > Tboundary. Under the WATCH definition, 21 windows are positive and 151 are negative; under the TRIGGER definition,
9 windows are positive and 163 are negative. The WATCH pooled OOF AUC is the pooled out-of-fold ROC AUC from purged
walk-forward validation on the combined TRAIN sample. WATCH AUC and WATCH AP are computed from the held-out WATCH
probabilities, while TRIGGER AUC and TRIGGER AP are computed from the separate onset-based TRIGGER probabilities. The
WATCH AUC 95% confidence interval is obtained from a circular block bootstrap with 500 trials and a 9-window block length. The
permutation probability pperm is the block-shuffle probability, based on 1000 trials, of obtaining a WATCH AUC at least as large
as the observed value by chance. WATCH BSS is the Brier Skill Score relative to a simple baseline forecast that assigns the same
WATCH probability to every held-out window, equal to the held-out WATCH-positive fraction.
Figure 3. WATCH-only soft-voting ensemble for the target
source. For each rolling window, the raw calibrated WATCH
probabilities from logistic regression, polynomial logistic re-
gression, and random forest are averaged to form the ensem-
ble WATCH probability. The top panel shows the 3-d binned
Fermi-LAT flux; the salmon-shaded bands mark the FULL-
BB flare intervals, and the navy dashed vertical line marks
the train/test boundary at Tboundary = MJD 59910.
The
lower panel shows the ensemble WATCH probability after
applying a causal seven-window trailing mean for visual clar-
ity. Filled green circles mark the WATCH-positive FULL-BB
scoring windows, dark-red dashed vertical lines mark FULL-
BB flare onsets, and the pale shaded region marks the held-
out test interval. The orange dotted horizontal line shows
the mean WATCH threshold of the three component models,
⟨τW⟩= (0.06 + 0.12 + 0.10)/3 = 0.093. The red dashed hor-
izontal line shows the corresponding mean of the three indi-
vidual TRIGGER thresholds, ⟨τT⟩= (0.82+0.73+0.35)/3 =
0.633, and is included only as a visual guide; no separate en-
semble TRIGGER probability is constructed in this work.
each window with ˆpW ≥τW is counted as a WATCH-
positive forecast. With this threshold, the held-out con-
fusion matrix is TP = 18, FP = 31, TN = 120, and
FN = 3.
These counts correspond to a precision of
18/49 = 0.367 and a recall of 18/21 = 0.857, giving
F2 = 0.677 and MCC = 0.473.
In other words, the
model recovers 18 of the 21 WATCH-positive held-out
windows, while 31 of the 151 WATCH-negative windows
are incorrectly assigned to WATCH. These threshold-
specific values are obtained directly from the confusion
matrix at τW = 0.12.
Figure 4 shows the held-out
ROC and precision–recall curves of the PLR WATCH
probabilities, together with the reliability curve.
The
ROC AUC and AP summarize the ranking performance
across all thresholds, whereas the precision, recall, F2,
and MCC values above describe the binary WATCH be-
havior only at the selected threshold τW = 0.12. The
improvement of PLR over the purely linear model sug-
gests that useful pre-flare information is carried by com-
binations of variability features, rather than by single
features alone.
Good ranking performance does not necessarily mean
that the WATCH probabilities are perfectly calibrated.
Figure 4 shows the reliability curve of the PLR WATCH
probabilities. In this panel, the horizontal axis gives the
mean predicted WATCH probability in each bin, while
the vertical axis gives the observed fraction of WATCH-
positive windows in the same bin.
The overall trend
is upward, which means that bins with higher predicted
WATCH probabilities generally contain a larger fraction
of true WATCH-positive windows. In the present back-
test, the held-out WATCH Brier Skill Score is positive
(BSS = 0.115). This means that the numerical WATCH
probabilities are modestly better than a simple baseline
forecast that assigns the same WATCH probability to
every held-out window. The WATCH output can there-
fore be interpreted as a meaningful probabilistic indica-
tor of flare risk, although its clearest practical use is still
to identify elevated-risk intervals before flare onset.
With this probabilistic WATCH interpretation in
mind, we next examine how the PLR WATCH and
TRIGGER probabilities evolve in time. Figure 5 shows
the target-source probability timeline.
In the lower
panel, the solid blue curve is the WATCH probability

[CAPTION] Table 2. Summary of model performance on the combined TRAIN sample and on the held-out target-source TEST set.

[CAPTION] Figure 3. WATCH-only soft-voting ensemble for the target

[CAPTION] Figure 4 shows the held-out

[CAPTION] Figure 4 shows the reliability curve of the PLR WATCH


<!-- page 13 -->
13
(a) ROC curve.
(b) Precision–recall curve.
(c) Reliability curve.
Figure 4. Held-out WATCH diagnostic curves for the PLR model. In the ROC panel, the blue step-like curve is obtained by
varying the WATCH probability threshold across the held-out windows and plotting the true-positive rate against the false-
positive rate at each step. The black dashed diagonal marks the random-ranking case, and the legend gives the area under the
curve (AUC = 0.891). In the precision–recall panel, the blue curve shows the same threshold sweep in precision–recall space. The
dashed horizontal line marks the held-out WATCH positive fraction, 21/172 ≃0.12, which is the precision expected from random
alerting; the legend gives the average precision (AP = 0.396). In the reliability panel, the held-out windows are grouped into
bins with similar numbers of samples according to their predicted WATCH probability. For each bin, the horizontal coordinate
shows the mean predicted probability and the vertical coordinate shows the fraction of truly WATCH-positive windows; the
error bars show the binomial uncertainty. The black dashed diagonal marks perfect agreement between predicted probability
and observed positive fraction. The panel annotation gives the Brier score, the Brier Skill Score, and the held-out WATCH
positive fraction (Brier = 0.0948, BSS = 0.115, base rate = 0.122).
and the red dashed curve is the TRIGGER probability.
For visual clarity, both curves are shown after applying a
causal seven-window trailing mean. The orange dotted
horizontal line marks the WATCH threshold, τW = 0.12,
and the red dashed horizontal line marks the TRIGGER
threshold, τT = 0.73. In this smoothed display, both
held-out flare onsets near MJD 60014 and MJD 60068
are preceded by WATCH-level probabilities.
By con-
trast, the smoothed TRIGGER curve does not cross the
TRIGGER threshold before either flare. Before the sec-
ond flare it rises close to the threshold, but still remains
below it.
The open red circles mark the true onset-
based TRIGGER-positive windows; they are evaluation
labels, not threshold crossings of the model. The raw
held-out TRIGGER probabilities also remain below the
TRIGGER threshold, so no valid pre-onset TRIGGER
crossing is obtained in either the smoothed display or
the raw scoring sequence.
To make the warning behaviour more explicit, we next
measure when the PLR model enters the WATCH state
before the two held-out flare onsets. Here a window is
counted as WATCH-active when the raw WATCH prob-
ability exceeds the WATCH threshold, τW = 0.12, for at
least two consecutive windows. The lead time is then the
time difference between that accepted WATCH window
and the flare onset. We use the raw WATCH probabili-
ties for this calculation, rather than the smoothed curves
shown in Fig. 5, so that the timing is not shifted by the
visual smoothing. We report two related timing mea-
sures. The first is the last accepted WATCH window
before flare onset. This means the final window in the
last valid WATCH sequence before the flare begins, and
it gives a conservative estimate of the final warning time.
For the two held-out flares, these last accepted WATCH
windows occur at MJD 60009.5 and MJD 60065.5. The
corresponding lead times, defined as the time between
the accepted WATCH window and the flare onset, are
4.5 and 2.5 d, with a median of 3.5 d. The second mea-
sure is the start of the WATCH episode that contains
this final pre-flare warning.
Here a WATCH episode
means the broader period during which the source stays
in the WATCH state; nearby WATCH runs are linked
into the same episode if they are separated by fewer than
five consecutive non-WATCH windows. Using this defi-
nition, the two WATCH episodes begin at MJD 59925.5
and MJD 59995.5, corresponding to much longer lead
times of 88.5 and 72.5 d. In simple terms, the model
first identifies a broad period of elevated flare risk well
before the flare, and this is followed by a shorter final
WATCH warning close to flare onset. No comparable
pre-onset TRIGGER detection is found.
5. DISCUSSION
The possibility of predicting blazar flares is closely
connected to the unique time-domain role of the Fermi-

[CAPTION] Figure 4. Held-out WATCH diagnostic curves for the PLR model. In the ROC panel, the blue step-like curve is obtained by


<!-- page 14 -->
14
Figure 5. PLR alert-state timeline. The top panel shows
the 3-d binned Fermi-LAT flux as black points; salmon shad-
ing marks FULL-BB flare intervals, and the navy dashed
vertical line marks the train/test boundary at Tboundary =
MJD 59910. The lower panel shows the causal seven-window
trailing means of the WATCH score (solid blue) and the
TRIGGER score (dark-red dashed).
Filled green circles
mark windows that are WATCH-positive under the FULL-
BB scoring labels, while open dark-red circles mark windows
that are TRIGGER-positive under the onset-based scoring
labels. Dark-red dashed vertical lines mark FULL-BB flare
onsets, the pale navy background marks the held-out test
stream, the orange dotted horizontal line is the WATCH
threshold (τW = 0.12), the red dashed horizontal line is the
TRIGGER threshold (τT = 0.73), and the grey dash-dotted
horizontal line is a visual probability guide at ˆp = 0.50. Lead-
time values quoted in the text are computed separately from
the raw, unsmoothed probabilities using the primary 90-d
look-back window and the two-window persistence require-
ment.
LAT . Since the start of the mission, the LAT has mon-
itored the γ-ray sky with a high duty cycle, producing
long and nearly uniform light curves for a large popula-
tion of variable sources (Atwood et al. 2009; Abdollahi
et al. 2023). The Fermi-LAT Light Curve Repository is
especially important in this context: it provides multi-
cadence light curves for more than 1500 variable LAT
sources, most of which are blazars, and therefore turns
blazar variability from a collection of isolated events into
a statistical forecasting problem. Instead of asking only
whether a source is flaring now, one can ask whether
its recent light curve resembles source states that have
preceded flares in the past.
This is the motivation behind the present work. We
use a sample of bright Fermi-LAT blazars to learn
source-normalized variability patterns, and then test
whether those patterns carry predictive information for
the target source 4FGL J1048.4+7143. The source nor-
malization is important because different blazars have
different average fluxes, noise levels, and flare ampli-
tudes. The aim is therefore not to learn that a particu-
lar absolute flux is high, but to learn whether the recent
variability state of a source has become flare-like relative
to its own long-term behavior.
The physical basis for this approach is that a blazar
flare, although it may appear as a rapid γ-ray outburst,
need not be a completely sudden event.
In shock-in-
jet, turbulent, and magnetic-reconnection pictures, the
final radiative peak is produced by changes in the jet
environment, particle acceleration, magnetic field struc-
ture, Doppler factor, or external photon field (Marscher
& Gear 1985; Spada et al. 2001; Marscher 2014; Sironi
et al. 2015). These changes can leave statistical traces in
the light curve before the peak itself. Long-term studies
have shown that blazar variability is structured, often
non-Gaussian or log-normal, and strongly dependent on
the source state (Giebels & Degrange 2009; Shah et al.
2018). Thus the 365-d rolling window used here is not
meant to predict a flare from a single data point. It asks
whether the source has entered a broader elevated-risk
state before the flare becomes obvious.
This forecasting view is also valuable observationally.
Existing survey-based alerts are extremely useful, but
they are mostly reactive: the wide-field instrument first
detects that a source has already become active, and
pointed optical, X-ray, radio, or TeV facilities are then
asked to follow it. A predictive WATCH state would
move part of this process earlier. Even a modest warn-
ing can help observers increase cadence, request Target-
of-Opportunity observations, prepare multi-wavelength
coverage, and avoid missing the rise of a flare.
This
is particularly important because major blazar out-
bursts often require simultaneous information from sev-
eral bands to locate the emitting region and distinguish
between competing jet scenarios (Marscher et al. 2010;
Agudo et al. 2011; Hayashida et al. 2015; Raiteri et al.
2017).
This places the present work between two familiar ap-
proaches to blazar variability. On the observational side,
flare monitoring often relies on simple flux-threshold
alerts in the γ-ray light curve, or on tracking individ-
ual indicators such as optical polarization degree, and
EVPA rotation (Blinov et al. 2015; Kiehlmann et al.
2016). These indicators are physically informative, but
they are usually followed one at a time, or in a small
number of combinations. For that reason, they do not
easily capture the broader, non-linear variability pat-
terns that may develop across several time-scales before

[CAPTION] Figure 5. PLR alert-state timeline. The top panel shows


<!-- page 15 -->
15
a flare.
On the machine-learning side, most work on
γ-ray blazars has focused on retrospective classification
rather than future-time forecasting. This includes the
classification of blazar candidates of uncertain type in
Fermi-LAT catalogues (Chiaro et al. 2016; Kovaˇcevi´c
et al. 2020; Sahakyan et al. 2023; Tolamatti et al. 2023),
as well as γ-ray blazar classification using self-supervised
learning (Bhatta et al. 2024). The problem studied here
is different.
For a source that is already being mon-
itored, we ask whether the recent light curve indicates
elevated flare risk within a specified future horizon. This
forecasting setting also requires special care, because fu-
ture information can leak into the labels, preprocessing,
calibration, or validation. Such temporal leakage is a
well-known problem in time-series forecasting (Hynd-
man & Athanasopoulos 2018), and it is the reason for
the strict causal design adopted in this work.
Within this framework, the main result of the present
backtest is that a strictly causal pipeline does recover
non-trivial predictive information from the Fermi-LAT
light curve. Among the tested models, polynomial logis-
tic regression gives the strongest held-out WATCH per-
formance, with AUC = 0.891, AP = 0.396, and a block-
permutation probability pperm = 0.006. This shows that
the WATCH ranking skill is unlikely to arise by chance,
even after accounting for the strong overlap between
neighboring rolling windows. The same model also gives
the strongest held-out TRIGGER ranking, with TRIG-
GER AUC = 0.770 and TRIGGER AP = 0.123. How-
ever, because the TRIGGER task has only a small num-
ber of positive held-out windows and no separate per-
mutation significance test is applied, this result should
be interpreted more cautiously.
Overall, the held-out
results suggest that the recent Fermi-LAT light curve
does contain useful information about future flare risk.
The success of PLR is also physically suggestive. A
purely linear model uses each variability feature sep-
arately, whereas PLR can respond to pairwise combi-
nations of features.
Its better held-out performance
therefore suggests that the pre-flare signal is not car-
ried by one extreme quantity alone. Instead, it is likely
encoded in combinations of moderate changes, such as
enhanced variability together with a change in tempo-
ral structure or spectral behavior.
This is consistent
with multi-wavelength studies in which γ-ray activity
is connected to optical polarisation rotations, radio-
core activity, spectral changes, and complex, sometimes
multi-zone, jet behaviour (Marscher et al. 2010; Agudo
et al. 2011; Hayashida et al. 2015; Raiteri et al. 2017).
The machine-learning result therefore matches a famil-
iar physical picture: blazar flares are not only large flux
excursions, but episodes in which several aspects of the
jet state change together.
The WATCH/TRIGGER formulation is important for
interpreting the alert behavour. WATCH asks whether
any flare interval will overlap the next 90 d, whereas
TRIGGER asks whether a flare onset will occur within
the next 45 d while excluding windows already inside a
flare. For PLR, the causal TRAIN optimization selects
τW = 0.12 and τT = 0.73. The WATCH threshold is
learned from source-local causal TRAIN WATCH scores
pooled across the training blazars, so that each source
contributes its own pre-cutoff WATCH history before
a common threshold is chosen.
This lower WATCH
threshold is consistent with the purpose of WATCH as a
broad elevated-risk state, whereas the higher TRIGGER
threshold still reflects the more conservative precision-
weighted trigger objective and the smaller number of
positive trigger windows available for training. The cor-
rect interpretation is therefore that the present data sup-
port a broad WATCH state more robustly than a sparse
high-confidence trigger.
The alert behavior should be interpreted in the same
way. In the held-out interval, the PLR WATCH state
appears before both flare onsets, whereas no valid pre-
onset TRIGGER crossing is obtained.
This does not
make the model useless; rather, it shows that the present
system is better suited to identifying elevated-risk inter-
vals than to issuing rare high-confidence triggers. The
last accepted WATCH windows occur 4.5 and 2.5 d be-
fore the two held-out flare onsets, while the starts of the
broader WATCH-active episodes occur 88.5 and 72.5 d
before onset.
These two timescales describe different
kinds of warning: the shorter one is the final warning
close to flare onset, whereas the longer one marks an
earlier period during which the source already appears
more flare-prone than usual.
The WATCH probability scale also deserves a nuanced
interpretation. For the adopted PLR model, the held-
out Brier Skill Score is positive (BSS = 0.115), so the
WATCH output is not only a relative flare-risk score,
but also modestly better than a simple constant base-
line forecast in absolute probability terms. This does
not mean that the calibration is perfect, but it does show
that the WATCH probabilities carry useful probabilistic
information. By contrast, the random forest has a neg-
ative held-out BSS (BSS = −0.204), showing again that
discrimination skill and probability calibration do not
necessarily select the same model. In the present work,
the main scientific aim is advance warning before flare
onset, so held-out ranking skill, threshold behavior, and
lead-time performance provide the most relevant basis
for adopting PLR as the reference model.


<!-- page 16 -->
16
These results also depend on the strict temporal de-
sign of the analysis, not only on the choice of classifier.
The separation between TRAIN-BB and FULL-BB, the
use of TRAIN-only preprocessing, the causal probability
calibration, and the split at tcut −H are all intended to
ensure that the reported forecasting skill is genuinely out
of sample. The main limitation of the present backtest
is that the held-out interval contains only two flare on-
sets. The lead-time values should therefore be viewed as
provisional, rather than as precise population-level num-
bers. The next step is to apply the same causal protocol
to a larger sample of bright LAT blazars, test whether
the WATCH thresholds remain useful across sources,
and improve probability calibration with a larger num-
ber of positive calibration windows. A particularly use-
ful extension would be to include contemporaneous op-
tical, radio, X-ray, and polarization data, because these
observables trace different parts of the jet and may help
turn a broad WATCH state into a more selective Target-
of-Opportunity trigger. In this sense, the present result
is best viewed as a first demonstration that long-term
Fermi-LAT light curves contain predictive information,
rather than as the final form of a real-time blazar flare
alert system.
6. SUMMARY AND CONCLUSIONS
We have presented a strictly causal machine-learning
framework for forecasting γ-ray blazar flares from long-
term Fermi-LAT light curves. The pipeline combines
Bayesian-Blocks flare identification, rolling-window vari-
ability features, separate WATCH and TRIGGER tar-
gets, TRAIN-only calibration and threshold selection,
and a fully held-out backtest on 4FGL J1048.4+7143.
The main conclusions are as follows.
1. A strictly causal backtest still retains measurable
predictive skill. Among the tested models, polyno-
mial logistic regression (PLR) gives the strongest
held-out WATCH performance, with AUC
=
0.891, AP =
0.396, and a block-permutation
p-value of 0.006.
The same model also gives
the strongest held-out TRIGGER ranking, with
TRIGGER AUC = 0.770 and TRIGGER AP
= 0.123.
These results suggest that long-term
LAT light curves contain useful pre-flare informa-
tion even when temporal leakage is explicitly con-
trolled.
2. The best held-out model is not simply the most
flexible one. Although the random forest reaches
the largest pooled out-of-fold WATCH AUC on
the TRAIN data, PLR gives the strongest held-out
WATCH discrimination and also a positive held-
out WATCH Brier Skill Score. This suggests that
pairwise interactions between variability features
are useful, but that greater model complexity does
not automatically lead to better post-cutoff gen-
eralization.
3. Operationally, the present system behaves mainly
as a WATCH predictor rather than as a hard trig-
ger.
For the adopted PLR model, the held-out
WATCH threshold τW = 0.12 recovers 18 of the 21
WATCH-positive windows, and the WATCH state
appears before both held-out flare onsets. The last
accepted WATCH windows occur 4.5 and 2.5 d
before flare onset, while the starts of the broader
WATCH-active episodes occur 88.5 and 72.5 d be-
fore onset.
4. The onset-based TRIGGER task remains more
challenging. The causal TRAIN optimisation se-
lects a PLR trigger threshold of τT = 0.73, and no
valid pre-onset TRIGGER crossing is obtained in
the held-out interval. The present data therefore
support a broad elevated-risk WATCH state more
robustly than a sparse high-confidence TRIGGER
state.
5. The WATCH probability scale is modestly reliable
in absolute terms. For the adopted PLR model,
the held-out WATCH BSS is positive (BSS =
0.115), so the outputs are better than a simple
constant baseline forecast and can be interpreted
as meaningful flare-risk probabilities, even though
their most useful practical role remains the iden-
tification of elevated-risk states.
The present results are encouraging but not yet defini-
tive, because the held-out interval contains only two
flare onsets. The next steps are therefore clear: apply
the same causal protocol to a larger set of bright LAT
blazars, test whether WATCH thresholds and probabil-
ity behaviour remain stable across sources, and improve
the probability calibration with more positive calibra-
tion examples. A particularly valuable extension will be
the inclusion of contemporaneous optical, radio, X-ray,
and polarization information, which may help convert
a broad WATCH state into a more selective Target-of-
Opportunity trigger.
Even in its present form, the framework provides a re-
producible and leakage-resistant way to translate blazar
γ-ray variability into advance-warning states. If this be-
havior is confirmed on a larger sample, causal WATCH-
style alerts from Fermi-LAT light curves could become
useful for scheduling intensified monitoring and coordi-
nated multi-wavelength follow-up before the brightest
phase of a flare.


<!-- page 17 -->
17
ACKNOWLEDGEMENTS
ZS is supported by the Department of Science and
Technology (DST), Govt. of India, under the INSPIRE
Faculty grant (DST/INSPIRE/04/2020/002319).
The
author thanks the Fermi-LAT Collaboration for mak-
ing the Light Curve Repository publicly available. This
research made use of astropy, a community-developed
core Python package for Astronomy (Astropy Collabora-
tion et al. 2022); scikit-learn (Pedregosa et al. 2011);
NumPy (Harris et al. 2020); SciPy (Virtanen et al.
2020); Matplotlib (Hunter 2007).
REFERENCES
Abdo, A. A., Ackermann, M., Ajello, M., et al. 2010, ApJ,
722, 520, doi: 10.1088/0004-637X/722/1/520
Abdollahi, S., Ajello, M., Baldini, L., et al. 2023, ApJS,
265, 31, doi: 10.3847/1538-4365/acbb6a
Ackermann, M., Ajello, M., Allafort, A., et al. 2011, ApJ,
743, 171, doi: 10.1088/0004-637X/743/2/171
Ackermann, M., Ajello, M., Albert, A., et al. 2015, ApJL,
813, L41, doi: 10.1088/2041-8205/813/2/L41
Agudo, I., Jorstad, S. G., Marscher, A. P., et al. 2011,
ApJL, 726, L13, doi: 10.1088/2041-8205/726/1/L13
Aharonian, F., Akhperjanian, A. G., Bazer-Bachi, A. R.,
et al. 2007, ApJL, 664, L71, doi: 10.1086/520635
Akbar, S. 2026, Journal of High Energy Astrophysics, 53,
100608, doi: https://doi.org/10.1016/j.jheap.2026.100608
Akbar, S., Shah, Z., Misra, R., Boked, S., & Iqbal, N. 2025,
PhRvD, 112, 063061, doi: 10.1103/zxgv-fzv5
Akbar, S., Shah, Z., Misra, R., & Iqbal, N. 2024, ApJ, 977,
111, doi: 10.3847/1538-4357/ad8ddb
Albert, J., Aliu, E., Anderhub, H., et al. 2007, ApJ, 669,
862, doi: 10.1086/521382
Astropy Collaboration, Price-Whelan, A. M., Lim, P. L.,
et al. 2022, ApJ, 935, 167, doi: 10.3847/1538-4357/ac7c74
Atwood, W. B., Abdo, A. A., Ackermann, M., et al. 2009,
ApJ, 697, 1071, doi: 10.1088/0004-637X/697/2/1071
Bhatta, G., Gharat, S., Borthakur, A., & Kumar, A. 2024,
MNRAS, 528, 976, doi: 10.1093/mnras/stae028
Biteau, J., & Giebels, B. 2012, A&A, 548, A123,
doi: 10.1051/0004-6361/201220056
Blandford, R. D., & Rees, M. J. 1978, PhyS, 17, 265,
doi: 10.1088/0031-8949/17/3/020
Blinov, D., Pavlidou, V., Papadakis, I., et al. 2015,
MNRAS, 453, 1669, doi: 10.1093/mnras/stv1723
Bloom, S. D., & Marscher, A. P. 1996, ApJ, 461, 657,
doi: 10.1086/177092
B¨ottcher, M., Reimer, A., Sweeney, K., & Prakash, A.
2013, ApJ, 768, 54, doi: 10.1088/0004-637X/768/1/54
Breiman, L. 1996, Machine Learning, 24, 123,
doi: 10.1023/A:1018054314350
—. 2001, Machine Learning, 45, 5,
doi: 10.1023/A:1010933404324
Chiaro, G., Salvetti, D., La Mura, G., et al. 2016, MNRAS,
462, 3180, doi: 10.1093/mnras/stw1830
de Prado, M. L. 2018, Advances in Financial Machine
Learning (Wiley)
Dermer, C. D., & Schlickeiser, R. 1993, ApJ, 416, 458,
doi: 10.1086/173251
Edelson, R., Turner, T. J., Pounds, K., et al. 2002, ApJ,
568, 610, doi: 10.1086/323779
Ghisellini, G., Padovani, P., Celotti, A., & Maraschi, L.
1993, ApJ, 407, 65, doi: 10.1086/172493
Giebels, B., & Degrange, B. 2009, A&A, 503, 797,
doi: 10.1051/0004-6361/200912303
Good, P. 2013, Permutation Tests: A Practical Guide to
Resampling Methods for Testing Hypotheses (New York:
Springer)
Harris, C. R., Millman, K. J., van der Walt, S. J., et al.
2020, Nature, 585, 357, doi: 10.1038/s41586-020-2649-2
Hayashida, M., Nalewajko, K., Madejski, G. M., et al. 2015,
ApJ, 807, 79, doi: 10.1088/0004-637X/807/1/79
Hunter, J. D. 2007, Computing in Science and Engineering,
9, 90, doi: 10.1109/MCSE.2007.55
Hyndman, R. J., & Athanasopoulos, G. 2018, Forecasting:
Principles and Practice, 2nd edn. (Melbourne: OTexts)
Katarzy´nski, K., Ghisellini, G., Tavecchio, F., et al. 2005,
A&A, 433, 479, doi: 10.1051/0004-6361:20041556
Kiehlmann, S., Savolainen, T., Jorstad, S. G., et al. 2016,
A&A, 590, A10, doi: 10.1051/0004-6361/201527725
Kovaˇcevi´c, M., Chiaro, G., Cutini, S., & Tosti, G. 2020,
MNRAS, 493, 1926, doi: 10.1093/mnras/staa394
K”unsch, H. R. 1989, The Annals of Statistics, 17, 1217
MacDonald, N. R., Marscher, A. P., Jorstad, S. G., &
Joshi, M. 2015, ApJ, 804, 111,
doi: 10.1088/0004-637X/804/2/111
Malik, Z., Akbar, S., Shah, Z., et al. 2025, MNRAS, 539,
2185, doi: 10.1093/mnras/staf620
Mannheim, K. 1993, A&A, 269, 67,
doi: 10.48550/arXiv.astro-ph/9302006
Maraschi, L., Ghisellini, G., & Celotti, A. 1992, ApJL, 397,
L5, doi: 10.1086/186531
Marscher, A. P. 2014, ApJ, 780, 87,
doi: 10.1088/0004-637X/780/1/87


<!-- page 18 -->
18
Marscher, A. P., & Gear, W. K. 1985, ApJ, 298, 114,
doi: 10.1086/163592
Marscher, A. P., Jorstad, S. G., Larionov, V. M., et al. 2010,
ApJL, 710, L126, doi: 10.1088/2041-8205/710/2/L126
M¨ucke, A., Protheroe, R. J., Engel, R., Rachen, J. P., &
Stanev, T. 2003, Astroparticle Physics, 18, 593,
doi: 10.1016/S0927-6505(02)00185-8
Nalewajko, K. 2013, MNRAS, 430, 1324,
doi: 10.1093/mnras/sts711
Ojha, R., & Carpen, B. 2017, The Astronomer’s Telegram,
9928, 1
Ojha, R., Carpenter, B., & Dutka, M. 2013, The
Astronomer’s Telegram, 4941, 1
Padovani, P., & Giommi, P. 1995, ApJ, 444, 567,
doi: 10.1086/175631
Pedregosa, F., Varoquaux, G., Gramfort, A., et al. 2011,
Journal of Machine Learning Research, 12, 2825,
doi: 10.48550/arXiv.1201.0490
Phipson, B., & Smyth, G. K. 2010, Statistical Applications
in Genetics and Molecular Biology, 9, Article 39
Platt, J. C. 1999, in Advances in Large Margin Classifiers,
ed. A. J. Smola, P. Bartlett, B. Sch”olkopf, &
D. Schuurmans (Cambridge, MA: MIT Press), 61–74
Polatidis, A. G., Wilkinson, P. N., Xu, W., et al. 1995,
ApJS, 98, 1, doi: 10.1086/192152
Politis, D. N., & Romano, J. P. 1992, in Exploring the
Limits of Bootstrap, ed. R. LePage & L. Billard (New
York: Wiley), 263–270
Politis, D. N., & Romano, J. P. 1994, Journal of the
American Statistical Association, 89, 1303,
doi: 10.1080/01621459.1994.10476870
Raiteri, C. M., Villata, M., Acosta-Pulido, J. A., et al.
2017, Nature, 552, 374, doi: 10.1038/nature24623
Sahakyan, N., Vardanyan, V., & Khachatryan, M. 2023,
MNRAS, 519, 3000, doi: 10.1093/mnras/stac3701
Saito, T., & Rehmsmeier, M. 2015, PLoS ONE, 10,
e0118432, doi: 10.1371/journal.pone.0118432
Scargle, J. D., Norris, J. P., Jackson, B., & Chiang, J. 2013,
ApJ, 764, 167, doi: 10.1088/0004-637X/764/2/167
Shah, Z., Mankuzhiyil, N., Sinha, A., et al. 2018, Research
in Astronomy and Astrophysics, 18, 141,
doi: 10.1088/1674-4527/18/11/141
Shah, Z., Sahayanathan, S., Mankuzhiyil, N., et al. 2017,
MNRAS, 470, 3283, doi: 10.1093/mnras/stx1194
Shah, Z., Dar, A. A., Akbar, S., et al. 2025, Phys. Rev. D,
111, 123052, doi: 10.1103/61tz-jk8c
Sikora, M., Begelman, M. C., & Rees, M. J. 1994, ApJ, 421,
153, doi: 10.1086/173633
Sironi, L., Petropoulou, M., & Giannios, D. 2015, MNRAS,
450, 183, doi: 10.1093/mnras/stv641
Spada, M., Ghisellini, G., Lazzati, D., & Celotti, A. 2001,
MNRAS, 325, 1559,
doi: 10.1046/j.1365-8711.2001.04557.x
Stickel, M., Padovani, P., Urry, C. M., Fried, J. W., &
Kuehr, H. 1991, ApJ, 374, 431, doi: 10.1086/170133
Tolamatti, A., Singh, K. K., & Yadav, K. K. 2023,
MNRAS, 523, 5341, doi: 10.1093/mnras/stad1826
Urry, C. M., & Padovani, P. 1995, PASP, 107, 803,
doi: 10.1086/133630
Vaughan, S., Edelson, R., Warwick, R. S., & Uttley, P.
2003, MNRAS, 345, 1271,
doi: 10.1046/j.1365-2966.2003.07042.x
Virtanen, P., Gommers, R., Oliphant, T. E., et al. 2020,
Nature Medicine, 17, 261,
doi: 10.1038/s41592-019-0686-2