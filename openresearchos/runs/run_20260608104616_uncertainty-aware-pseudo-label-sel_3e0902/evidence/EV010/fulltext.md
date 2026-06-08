<!-- page 1 -->
ANNOTATION-EFFICIENT ACTIVE TEST-TIME ADAPTATION WITH CONFORMAL
PREDICTION
Tingyu Shi1
Fan Lyu2
Shaoliang Peng3∗
1 Computer Science and Engineering, University of California San Diego
2 New Laboratory of Pattern Recognition, Institute of Automation, Chinese Academy of Sciences
3 College of Computer Science and Electronic Engineering, Hunan University
ABSTRACT
Active Test-Time Adaptation (ATTA) improves model robustness
under domain shift by selectively querying human annotations at
deployment, but existing methods use heuristic uncertainty mea-
sures and suffer from low data selection efficiency, wasting human
annotation budget. We propose Conformal Prediction Active TTA
(CPATTA), which first brings principled, coverage-guaranteed un-
certainty into ATTA. CPATTA employs smoothed conformal scores
with a top-K certainty measure, an online weight-update algorithm
driven by pseudo coverage, a domain-shift detector that adapts
human supervision, and a staged update scheme balances human-
labeled and model-labeled data. Extensive experiments demonstrate
that CPATTA consistently outperforms the state-of-the-art ATTA
methods by around 5% in accuracy.
Our code and datasets are
available at https://github.com/tingyushi/CPATTA.
Index Terms— Domain Shift, Test-Time Adaptation, Active
Test-Time Adaptation, Conformal Prediction
1. INTRODUCTION
Test-Time Adaptation (TTA) aims to update pretrained models on-
the-fly using unlabeled test-time data, enabling them to handle do-
main shifts where the distribution of test samples differs from the
training set. Such domain shifts are common in real-world appli-
cations, including autonomous driving under varying weather condi-
tions [1, 2], cross-hospital MRI imaging with different devices [3, 4],
and speech recognition with diverse accents and noise [5, 6]. Despite
its growing popularity [7–10], TTA often suffers from suboptimal
real-time and post-adaptation performance due to the lack of ground-
truth supervision. Inspired by Active Learning, the Active Test-Time
Adaptation (ATTA) paradigm [11] introduces selective human anno-
tations during adaptation, enabling models to leverage supervised
signals for improved robustness.
Existing methods on ATTA propose diverse criteria for selecting
human-annotated samples. SimATTA [11] employs fixed entropy
upper and lower thresholds combined with incremental clustering to
determine samples for human annotation. CEMA [12] uses a fixed
entropy lower bound and a dynamically adjusted upper bound to se-
lect samples for human annotation. EATTA [13] looks for samples
located at the border of the source and the shifted domain by per-
turbing the softmax scores. However, existing ATTA methods suffer
from low data selection efficiency, as illustrated in Fig. 1. A sub-
stantial portion of the annotated samples turns out to be redundant,
since they could already be correctly predicted by the model, lead-
ing to a significant waste of limited human supervision. Such inef-
∗Corresponding author. Email: slpeng@hnu.edu.cn
Fig. 1. Comparison between existing ATTA methods and CPATTA.
Selection efficiency is defined as the fraction of useful samples: cor-
rect predictions for model-annotated data and incorrect predictions
for human-annotated data. CPATTA achieves higher selection effi-
ciency in both cases, enabling better real-time and post-adaptation
performance under the same annotation budget.
ficiency not only diminishes the value of annotations but also con-
strains the overall adaptation performance. These limitations high-
light the need for a more effective selection mechanism that can max-
imize the utility of human annotations during test-time adaptation.
To address the inefficiency of existing ATTA methods, a natural
idea is to introduce a principled measure of uncertainty. Conformal
prediction (CP) provides such a tool by transforming a model’s
single-point output into a prediction set with statistical coverage
guarantees.
The size of this set reflects the model’s uncertainty,
making CP an appealing candidate for guiding sample selection.
However, applying CP to ATTA is non-trivial. Classical CP relies on
the assumption that calibration and test data are exchangeable [14–
16], which does not hold in ATTA since calibration data come from
the source domain while test samples belong to a shifted domain.
Although recent extensions of CP have relaxed this assumption, they
still cannot provide valid guarantees under the dynamic and evolving
conditions of test-time adaptation [17–19]. This gap motivates us to
explore how CP can be adapted and extended to improve annotation
efficiency in the ATTA setting.
In this paper, we propose Conformal Prediction Active TTA
(CPATTA), a framework that integrates CP into ATTA. The core
idea is to replace heuristic uncertainty measures with principled,
coverage-guaranteed ones, and to adapt them to dynamic test-time
environments. Specifically, CPATTA introduces three key compo-
nents. First, it employs smoothed conformal scores and a top-K
certainty measure to provide fine-grained uncertainty signals, en-
abling more efficient allocation of scarce human annotations and re-
liable pseudo-labeling. Second, it develops an online weight-update
arXiv:2509.25692v1  [cs.LG]  30 Sep 2025

[CAPTION] Fig. 1. Comparison between existing ATTA methods and CPATTA.


<!-- page 2 -->
Fig. 2. Overview of CPATTA method. The real-time model f(·; θ) makes the predictions right away when an online batch of data arrives.
If CPATTA detects that current batch’s domain is different from the previous batch’s domain(Dt ̸= Dt−1), the algorithm selects more data
for human annotation to ensure the real-time performance. Two CPs provide uncertainty measures for each sample within the batch; human
annotates uncertain samples while the model annotates certain samples. Then, CPATTA updates the weights of two CPs based on the pseudo
coverages calculated from model predictions and prediction sets.
algorithm that leverages pseudo coverage as feedback to dynami-
cally correct coverage under domain shifts, ensuring that uncertainty
estimates remain calibrated to the user-chosen risk level.
Third,
CPATTA incorporates a domain-shift detector that increases human
supervision when a new domain is encountered, preventing error ac-
cumulation at the onset of sudden distributional changes. Together,
these designs yield an annotation strategy that is both efficient and
reliable, leading to improved real-time and long-term adaptation per-
formance. Experimental results show that our method outperforms
existing ATTA methods.
2. METHOD
2.1. Problem Statement and Conformal Prediction
Let f(·; ϕ) denote a model pretrained on a source domain dataset
DS. At deployment, a real-time model f
 ·; θt 
is initialized with the
pretrained parameters, i.e., θ0 = ϕ. During test-time, unlabeled data
arrive sequentially in mini-batches Bt = {xS′
1 , · · · , xS′
Nt}, drawn
from a shifted target domain S′. The key challenge of ATTA is to
adapt f
 ·; θt 
to this evolving target domain under a limited an-
notation budget. Specifically, at each step t, the model receives an
unlabeled batch Bt, selects a subset of samples for human annota-
tion while another subset are pseudo-labeled by the model itself, and
then updates its parameters from θt to θt+1 using both human- and
model-annotated data. The objective of ATTA is to maximize predic-
tive performance by efficiently utilizing limited human supervision
and continuously improving the deployed model in real time.
Existing ATTA methods suffer from low data selection effi-
ciency, which makes the choice of samples for annotation a critical
issue. Conformal Prediction (CP) offers a classical framework to
quantify model uncertainty and guide sample selection. CP aug-
ments a predictor with prediction sets by leveraging a small-size
labeled calibration set Dcal = {(xi, yi)}n
i=1 and a nonconformity
score function S. For a test input xn+1, the prediction set is
C(xn+1) = {y ∈Y|S(xn+1, y) ≤τ} ,
(1)
where τ is the quantile conformal threshold.
τ = Quantile1−α
 {S(xi, yi)}n
i=1
 
.
(2)
For a classification problem, a typical choice of S is one minus the
softmax score of the correct label. Under the exchangeability (same
distribution) assumption between calibration and test data, CP guar-
antees coverage with high probability [14]: 1−α ≤P (y ∈C(x)) ≤
1 −α +
1
1+n, where n is the calibration set size. C(x) serves as a
measure of uncertainty, where larger sets indicate lower uncertainty.
Although CP leverages a small subset of source-domain data, these
samples are not reused for training but only for calibrating confor-
mal predictors to estimate prediction uncertainty. In test-time appli-
cations where annotation resources are critical, such as autonomous
driving with safety risks, or medical imaging with costly expert la-
beling, this calibration reduces redundant human annotations and
improves real-time as well as post-adaptation performance.
However, this guarantee relies on exchangeability assumption,
which is violated in ATTA because calibration data are drawn from
the source domain while test batches come from a shifted domain.
This mismatch creates a coverage gap between the target and ac-
tual coverage. Although weighted extensions of CP [17, 18] attempt
to recover coverage guarantees by reweighting samples, they depend
on accurate weight estimation, which is infeasible under the dynamic
nature of ATTA. Therefore, existing CP methods cannot be directly
applied to ATTA, and this limitation motivates us to design a new
weight-estimation algorithm that ensures valid coverage and reli-
able sample selection in ATTA.
2.2. Uncertainty-Guided Annotation with CP
Standard CP relies on prediction set size as an uncertainty proxy,
which is too coarse for ATTA. We instead adopt smoothed predic-
tion sets [20], converting hard set membership into soft inclusion
scores and yielding more fine-grained uncertainty signals that adapt
to dynamic test-time shifts. Given a normalized nonconformity score
function S, the soft score of including label y for input x is
E(x, y; τ) = σ ((τ −S(x, y)) /T) ,
(3)
where σ(x) =
1
1+e−x . T represents temperature. For a test input x
with label space Y = {1, . . . , L}, we sort labels in descending order
E(x, y(1); τ) ≥E(x, y(2); τ) ≥· · · ≥E(x, y(L); τ),
(4)
and define the top-K certainty score as
CertK(x; τ) = Ek∈[1,K]E(x, y(k); τ),
(5)

[CAPTION] Fig. 2. Overview of CPATTA method. The real-time model f(·; θ) makes the predictions right away when an online batch of data arrives.


<!-- page 3 -->
which reflects how strongly the model supports its most plausible
labels, which is more informative compared to the hard CP.
Second, we leverage two complementary conformal predictors:
one based on the pretrained model f(·; ϕ) and one on the real-time
adapted model f
 ·; θt 
. For each test batch Bt, these predictors
yield thresholds τpre and τrt, from which we compute
Certpre
K (x) = CertK(x; τpre),
Certrt
K(x) = CertK(x; τrt).
(6)
Annotation is then allocated by sending the Nhuman least-certain
samples under the real-time predictor to human annotators, while
the Nmodel most-certain ones under the pretrained predictor receive
pseudo-labels.
The selected samples are stored in human buffer
BufH and model buffer BufM for subsequent parameter updates.
Finally, to adapt annotation effort to domain dynamics, we in-
tegrate a domain-change detector. Specifically, we employ the Do-
main Shift Signal (DSS) [21] to test whether the current batch Bt
originates from a new domain. If a shift is detected, we temporar-
ily increase the human annotation budget from Nhuman to N ′
human ≥
Nhuman, reflecting the reduced reliability of prior knowledge and ac-
celerating adaptation at the onset of new domains.
2.3. Adaptive Weighting for CP in ATTA
Although uncertainty-guided annotation improves efficiency, valid
coverage is still required for reliable adaptation. Since calibration
data come from the source domain, the exchangeability assumption
of classical CP is violated under domain shift (Sec. 2.1). To address
this, we introduce a dynamic weighting mechanism that updates CP
online during adaptation.
Pseudo Coverage as Feedback. Since ground-truth labels are un-
available at test time, true coverage cannot be computed. We thus de-
fine pseudo coverage, which treats the real-time model’s prediction
as a surrogate label. For batch Bt, the pseudo coverage of real-time
and pretrained CPs is
PCt
rt = Ex∈Bt1
 
h
 f
 x; θt  
∈Ct
rt(x)
 
,
(7)
PCt
pre = Ex∈Bt1
 
h
 f
 x; θt  
∈Ct
pre(x)
 
,
(8)
where h(f(x; θt)) is the predicted label. These values act as online
feedback, revealing whether each CP under- or over-covers relative
to the target level.
Online Weight Updates. Guided by pseudo coverage, we update
the weights wt
rt and wt
pre assigned to real-time and pretrained CPs.
Instead of fixed weights, we employ an exponential update rule that
decreases weights when under-coverage is detected and increases
them when over-coverage occurs, thereby steering coverage towards
1 −α. For the real-time CP, the update is
wt
rt = wt−1
rt
T t
rt
,
T t
rt = exp
 
(1 −α) −PSt−1
rt
 
· T t−1
rt
,
(9)
with a symmetric update for wt
pre. This enables CP to self-calibrate
under evolving domains using only unlabeled feedback.
Coverage Guarantee.
Following the theoretical framework of
weighted CP [17, 18], our adaptive weighting scheme satisfies the
coverage bound:
1 −α −
w
nw + 1
Xn
i=1 dT V (Z, Zi) ≤P (x ∈C(x))
(10)
≤1 −α +
1
nw + 1 +
w
nw + 1
Xn
i=1 dT V (Z, Zi),
where dT V is total variation distance, Z = ((xi, yi))n+1
i=1 , Zi swaps
(xi, yi) with (xn+1, yn+1), and we can substitute w with wt
rt or
wt
pre. In essence, pseudo coverage serves as an online correction sig-
nal, keeping CP coverage near the target level despite domain shift.
2.4. Model Update
After each test batch is processed, the real-time model parameters
are updated before the next batch arrives. The training objective
combines cross-entropy losses from both buffers:
LH(θ) = E(x,y)∈BufH −log(softmax (f (x; θ))y),
(11)
LM(θ) = E(x,ˆy)∈BufM −log(softmax (f (x; θ))ˆy).
(12)
softmax(·)y denotes the softmax probability assigned to label y.
To balance reliability and efficiency, we adopt a two-stage up-
date with buffer-specific learning rates. First, parameters are updated
using human-labeled data with rate ηH
θt+ 1
2 = θt −ηH∇θLH(θt),
(13)
and then further updated with model-labeled data using rate ηM:
θt+1 = θt+ 1
2 −ηM∇θLM(θ
1
2 ).
(14)
This staged update scheme reflects the design principle of trust but
expand.
Reliable human supervision anchors the model update,
while additional pseudo-labeled data broaden adaptation at low cost.
The method ensures that human annotations dominate parameter
correction, whereas model labels serve as auxiliary signals that
accelerate adaptation without overwhelming reliability.
2.5. Discussion: CPATTA vs. Existing Nonexchangeable CP
Prior extensions of CP under distribution shift typically rely on
fixed or heuristic weighting. For instance, weighted CP [17, 18]
rebalances calibration scores via predefined constants or geomet-
ric decay. Such static schemes cannot adapt to evolving domains
in ATTA, where calibration–target distribution similarity changes
across batches. Similarly, QTC [19] adjusts thresholds using unla-
beled test data, but depends on stable model predictions. In ATTA,
the model is continuously updated and errors accumulate, making
these scores unreliable and the resulting coverage unstable.
In contrast, CPATTA leverages pseudo coverage as an online
feedback signal and adaptively reweights CP to track distribution
drift. This design enables principled and dynamic coverage con-
trol, forming a key distinction from existing nonexchangeable CP
approaches and completing our method.
3. EXPERIMENTS
Experiment Details. We evaluate on three domain shift datasets,
including PACS, VLCS and Tiny-ImageNet-C. PACS [22] consists
of 4 domains (Photo, Art painting, Cartoon, Sketch) spanning 7 cat-
egories. VLCS [23] combines 4 other datasets with 5 classes, pro-
viding natural distribution shifts across domains. Tiny-ImageNet-C
[24] extends Tiny-ImageNet with 200 classes by adding 15 corrup-
tion types at 5 severity levels, and we use severity 5 for the exper-
iment. Evaluation is based on real-time accuracy, post-adaptation
accuracy, and coverage gap. For PACS and VLCS, we sample 50
images per class from the source domain, forming calibration sets of
350 and 250 images, respectively. For Tiny-ImageNet-C, we select
3 images per class, yielding 600 images. For PACS and VLCS, we
select the 3 most uncertain samples per batch for human annotation,
while for Tiny-ImageNet-C we select 2. Accordingly, the human
annotation budget is capped at 300 for PACS/VLCS and 3000 for
Tiny-ImageNet-C across all methods to ensure fair comparison. We
set K = 1 for top-K certainty score for all 3 datasets. The backbone
model is ResNet-18 [25], which is pretrained on domain P, C, and
brightness for the three datasets, respectively.


<!-- page 4 -->
Table 1. Real-Time and Post-Adaptation Accuracy on PACS, VLCS and Tiny-ImageNet-C. Acc refers to the overall accuracy.
PACS
VLCS
Tiny-ImageNet-C
Real-Time
Post-Adapt
Real-Time
Post-Adapt
Real-Time
Post-Adapt
Method
A
C
S
Acc
Acc
L
S
V
Acc
Acc
Acc
Acc
Tent [7]
67.19
68.98
67.09
67.64
75.06
46.26
41.22
55.72
47.91
56.90
12.69
13.86
CoTTA [9]
65.68
65.87
64.55
65.17
70.84
45.84
39.49
54.92
46.89
47.33
3.44
9.10
SimATTA [11]
72.56
49.36
72.49
65.99
77.85
65.38
59.78
63.71
62.80
71.97
33.30
41.81
CEMA [12]
51.56
66.98
67.01
63.20
76.45
55.18
43.39
49.35
48.91
63.93
32.09
41.58
EATTA [13]
70.07
69.16
65.60
67.89
75.07
44.56
39.48
53.14
45.88
54.22
13.65
15.57
CPATTA(Ours, α = 0.1)
71.39
68.05
76.18
72.71
85.25
57.99
64.87
70.50
64.95
76.79
34.61
43.74
CPATTA(Ours, α = 0.2)
73.10
70.01
79.51
75.26
87.13
58.11
61.88
73.34
64.96
77.72
34.61
43.73
CPATTA(Ours, α = 0.3)
72.51
73.76
74.60
73.85
85.69
57.77
62.52
72.66
64.84
76.35
34.77
44.59
Table 2. Data selection efficiencies comparisons.
PACS
VLCS
Tiny-ImageNet-C
Method
EffH
EffM
EffH
EffM
EffH
EffM
SimATTA
47.60
67.60
57.04
64.78
79.25
50.00
CEMA
24.91
N/A
44.38
N/A
75.82
N/A
EATTA
53.71
90.71
58.38
63.60
62.90
32.31
CPATTA(Ours, α = 0.1)
66.30
93.41
62.89
81.91
90.62
47.55
CPATTA(Ours, α = 0.2)
65.94
93.65
61.86
85.11
90.62
47.55
CPATTA(Ours, α = 0.3)
67.39
94.59
64.29
84.04
90.45
48.14
Compared Methods.
We compare against two TTA methods,
Tent [7] and CoTTA [9], and three ATTA methods, CEMA [12],
SimATTA [11], and EATTA [13]. For Tent and CoTTA, we adopt
the enhanced TTA protocol from [11], which grants access to ad-
ditional randomly sampled labels. For CEMA, we retain only its
selective mechanism, replacing foundation-model annotations with
human annotations, so there are no model annotations for CEMA.
Real-Time and Post-Adaptation Accuracy. Table 1 reports real-
time and post-adaptation accuracies on PACS, VLCS, and Tiny-
ImageNet-C with different miscoverage levels (α = 0.1, 0.2, 0.3).
CPATTA achieves clear and consistent improvements across all
benchmarks. On PACS, CPATTA with α = 0.2 reaches 75.26%
real-time accuracy and 87.13% post-adaptation accuracy, improving
over SimATTA by nearly 9% in real-time and post-adaptation. On
VLCS, CPATTA also delivers strong gains, with 64.96% real-time
and 77.72% post-adaptation accuracy, outperforming CEMA and
EATTA by over 15% and 20% respectively. Even on the challenging
Tiny-ImageNet-C benchmark, CPATTA surpasses all ATTA base-
lines, achieving the best post-adaptation accuracy of 44.59%. These
results demonstrate that CPATTA is not only more effective than
SOTA methods but also more adaptable to diverse domain shifts.
Data Selection Efficiency. We formally define efficiency of human
annotation(EffH) and model annotation(EffM) as follows:
EffH = E(x,y)∈BffH 1 [y ̸= ˆy] ,
(Model Predicts Wrong)
(15)
EffM = E(x,ˆy)∈BffM 1 [y = ˆy] .
(Model Predicts Right)
(16)
As shown in Table 2, CPATTA achieves the highest EffH across all
datasets, while maintaining competitive or superior EffM. The re-
sults are stable across different α values, demonstrating CPATTA’s
effectiveness in improving annotation efficiency in ATTA.
Coverage Gap. To evaluate the effectiveness of our adaptive weight-
ing algorithm, we compare our CP with ExCP [15] and QTC [14]
on VLCS. Fig. 3 reports the average coverage gap for both real-
time model CP and pretrained model CP, along with real-time and
post-adaptation accuracy. Across all α values, our CP consistently
achieves the lowest coverage gap for both CPs. Furthermore, by en-
suring better calibration, our method also delivers higher real-time
and post-adaptation accuracy.
Ablation Study. We conduct ablation studies on PACS and VLCS,
as shown in Table 3. RS denotes random selection. GD corresponds
to geometric decay weights(wi = ρn+1−i, ρ = 0.9), following
[17, 18]. AW denotes adaptive weighting (Sec. 2.3). The results
Fig. 3. Compare our CP with other CPs on VLCS
Table 3. Ablation Study on PACS and VLCS
PACS
VLCS
Real-Time
Post-Adapt
Real-Time
Post-Adapt
RS
68.16
78.37
55.58
70.77
CP + GD
74.31
83.37
63.29
75.14
CP + AW
75.26
87.13
64.96
77.72
show that selection with CP improves performance over random se-
lection, and incorporating adaptive weighting yields further gains,
demonstrating the robustness of our approach.
Replay Experiment. To further show the effectiveness of CP in
ATTA, we allow the compared methods access to the labeled cali-
bration set as replay data during adaptation. Table 4 reports results
on PACS. While SimATTA with replay improves accuracy on do-
main A, CPATTA achieves superior performance on the other two
domains and obtains the best overall real-time and post-adaptation
accuracy. This confirms that leveraging the calibration set through
CP is more effective than using it as replay data.
Table 4. Replay Experiment on PACS
Real-Time
Post-Adapt
Method
A
C
S
Acc
Acc
SimATTA with Replay
74.32
50.64
72.77
66.92
79.14
CEMA with Replay
65.58
68.64
58.28
63.00
75.38
EATTA with Replay
69.97
69.07
65.59
67.65
74.85
CPATTA(Ours, α = 0.1)
71.39
68.05
76.18
72.71
85.25
CPATTA(Ours, α = 0.2)
73.10
70.01
79.51
75.26
87.13
CPATTA(Ours, α = 0.3)
72.51
73.76
74.60
73.85
85.69
4. CONCLUSION
In this paper, we employ CP as a principled uncertainty measure to
guide data selection for human and model annotation in ATTA. To
address the coverage gap of CP under domain shifts, we propose
an adaptive weighting algorithm that enhances robustness and yields
more reliable calibration across diverse environments. Our method
improves efficiency of both human- and model-annotated data, re-
ducing waste of scarce supervision while enabling more trustworthy
pseudo-labels and stronger adaptation. Experiments show CPATTA
achieves roughly 5% higher accuracy than SOTA ATTA methods.
Future work includes developing better model update strategies and
leveraging statistical distributions to characterize annotation errors
in realistic deployment.


**[Table p4.1]**
| Method | PACS |  | VLCS |  | Tiny-ImageNet-C |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | Real-Time A C S Acc | Post-Adapt Acc | Real-Time L S V Acc | Post-Adapt Acc | Real-Time Acc | Post-Adapt Acc |
| Tent [7] CoTTA [9] SimATTA [11] CEMA [12] EATTA [13] CPATTA(Ours, α = 0.1) CPATTA(Ours, α = 0.2) CPATTA(Ours, α = 0.3) | 67.19 68.98 67.09 67.64 65.68 65.87 64.55 65.17 72.56 49.36 72.49 65.99 51.56 66.98 67.01 63.20 70.07 69.16 65.60 67.89 71.39 68.05 76.18 72.71 73.10 70.01 79.51 75.26 72.51 73.76 74.60 73.85 | 75.06 70.84 77.85 76.45 75.07 85.25 87.13 85.69 | 46.26 41.22 55.72 47.91 45.84 39.49 54.92 46.89 65.38 59.78 63.71 62.80 55.18 43.39 49.35 48.91 44.56 39.48 53.14 45.88 57.99 64.87 70.50 64.95 58.11 61.88 73.34 64.96 57.77 62.52 72.66 64.84 | 56.90 47.33 71.97 63.93 54.22 76.79 77.72 76.35 | 12.69 3.44 33.30 32.09 13.65 34.61 34.61 34.77 | 13.86 9.10 41.81 41.58 15.57 43.74 43.73 44.59 |


**[Table p4.2]**
| Method | PACS EffH EffM | VLCS EffH EffM | Tiny-ImageNet-C EffH EffM |
| --- | --- | --- | --- |
| SimATTA CEMA EATTA CPATTA(Ours, α = 0.1) CPATTA(Ours, α = 0.2) CPATTA(Ours, α = 0.3) | 47.60 67.60 24.91 N/A 53.71 90.71 66.30 93.41 65.94 93.65 67.39 94.59 | 57.04 64.78 44.38 N/A 58.38 63.60 62.89 81.91 61.86 85.11 64.29 84.04 | 79.25 50.00 75.82 N/A 62.90 32.31 90.62 47.55 90.62 47.55 90.45 48.14 |


**[Table p4.3]**
|  | PACS Real-Time Post-Adapt | VLCS Real-Time Post-Adapt |
| --- | --- | --- |
| RS CP + GD CP + AW | 68.16 78.37 74.31 83.37 75.26 87.13 | 55.58 70.77 63.29 75.14 64.96 77.72 |


**[Table p4.4]**
| Method | Real-Time A C S Acc | Post-Adapt Acc |
| --- | --- | --- |
| SimATTA with Replay CEMA with Replay EATTA with Replay CPATTA(Ours, α = 0.1) CPATTA(Ours, α = 0.2) CPATTA(Ours, α = 0.3) | 74.32 50.64 72.77 66.92 65.58 68.64 58.28 63.00 69.97 69.07 65.59 67.65 71.39 68.05 76.18 72.71 73.10 70.01 79.51 75.26 72.51 73.76 74.60 73.85 | 79.14 75.38 74.85 85.25 87.13 85.69 |

[CAPTION] Table 1. Real-Time and Post-Adaptation Accuracy on PACS, VLCS and Tiny-ImageNet-C. Acc refers to the overall accuracy.

[CAPTION] Table 2. Data selection efficiencies comparisons.

[CAPTION] Fig. 3. Compare our CP with other CPs on VLCS

[CAPTION] Table 3. Ablation Study on PACS and VLCS

[CAPTION] Table 4. Replay Experiment on PACS


<!-- page 5 -->
References
[1] Tao Sun, Mattia Segu, Janis Postels, Yuxuan Wang, Luc
Van Gool, Bernt Schiele, Federico Tombari, and Fisher Yu,
“Shift: a synthetic driving dataset for continuous multi-task
domain adaptation,” in Proceedings of the IEEE/CVF confer-
ence on computer vision and pattern recognition, 2022, pp.
21371–21382.
[2] Guofa Li, Zefeng Ji, Xingda Qu, Rui Zhou, and Dongpu Cao,
“Cross-domain object detection for autonomous driving: A
stepwise domain adaptative yolo approach,” IEEE Transac-
tions on Intelligent Vehicles, vol. 7, no. 3, pp. 603–615, 2022.
[3] Ekaterina Kondrateva, Marina Pominova, Elena Popova,
Maxim Sharaev, Alexander Bernstein, and Evgeny Burnaev,
“Domain shift in computer vision models for mri data anal-
ysis: an overview,” in International Conference on Machine
Vision. SPIE, 2021, vol. 11605, pp. 126–133.
[4] Jonas Richiardi, Veronica Ravano, Nataliia Molchanova, Pe-
dro M Gordaliza, Tobias Kober, and Meritxell Bach Cuadra,
“Domain shift, domain adaptation, and generalization: A focus
on mri,” in Trustworthy AI in Medical Imaging, pp. 127–151.
Elsevier, 2025.
[5] Sining Sun, Binbin Zhang, Lei Xie, and Yanning Zhang,
“An unsupervised deep domain adaptation approach for robust
speech recognition,”
Neurocomputing, vol. 257, pp. 79–87,
2017.
[6] Shahram Ghorbani and John HL Hansen,
“Domain ex-
pansion for end-to-end speech recognition: Applications for
accent/dialect speech,”
IEEE/ACM Transactions on Audio,
Speech, and Language Processing, vol. 31, pp. 762–774, 2022.
[7] Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Ol-
shausen, and Trevor Darrell,
“Tent: Fully test-time adapta-
tion by entropy minimization,” in International Conference on
Learning Representations, 2021.
[8] Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Yaofo Chen, Shi-
jian Zheng, Peilin Zhao, and Mingkui Tan, “Efficient test-time
model adaptation without forgetting,” in International confer-
ence on machine learning. PMLR, 2022, pp. 16888–16905.
[9] Qin Wang, Olga Fink, Luc Van Gool, and Dengxin Dai, “Con-
tinual test-time domain adaptation,”
in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recog-
nition, 2022, pp. 7201–7211.
[10] Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Zhiquan Wen,
Yaofo Chen, Peilin Zhao, and Mingkui Tan, “Towards stable
test-time adaptation in dynamic wild world,” in Internetional
Conference on Learning Representations, 2023.
[11] Shurui Gui, Xiner Li, and Shuiwang Ji, “Active test-time adap-
tation: Theoretical analyses and an algorithm,” in International
Conference on Learning Representations, 2024.
[12] Yaofo Chen, Shuaicheng Niu, Shoukai Xu, Hengjie Song,
Yaowei Wang, and Mingkui Tan, “Towards robust and effi-
cient cloud-edge elastic model adaptation via selective entropy
distillation,” in International Conference on Learning Repre-
sentations, 2024.
[13] Guowei Wang and Changxing Ding, “Effortless active label-
ing for long-term test-time adaptation,” in Proceedings of the
Computer Vision and Pattern Recognition Conference, 2025,
pp. 25633–25642.
[14] Anastasios N Angelopoulos and Stephen Bates, “A gentle in-
troduction to conformal prediction and distribution-free uncer-
tainty quantification,” arXiv preprint arXiv:2107.07511, 2021.
[15] Glenn Shafer and Vladimir Vovk,
“A tutorial on conformal
prediction.,” Journal of Machine Learning Research, vol. 9,
no. 3, 2008.
[16] Anastasios Nikolas Angelopoulos, Stephen Bates, Adam
Fisch, Lihua Lei, and Tal Schuster,
“Conformal risk con-
trol,”
in International Conference on Learning Representa-
tions, 2024.
[17] Ant´onio Farinhas, Chrysoula Zerva, Dennis Thomas Ulmer,
and Andre Martins, “Non-exchangeable conformal risk con-
trol,”
in International Conference on Learning Representa-
tions, 2024.
[18] Rina Foygel Barber, Emmanuel J Candes, Aaditya Ramdas,
and Ryan J Tibshirani,
“Conformal prediction beyond ex-
changeability,”
The Annals of Statistics, vol. 51, no. 2, pp.
816–845, 2023.
[19] Fatih Furkan Yilmaz and Reinhard Heckel, “Test-time recal-
ibration of conformal predictors under distribution shift based
on unlabeled examples,”
arXiv preprint arXiv:2210.04166,
2022.
[20] David Stutz, Krishnamurthy Dj Dvijotham, Ali Taylan Cemgil,
and Arnaud Doucet,
“Learning optimal conformal classi-
fiers,” in International Conference on Learning Representa-
tions, 2022.
[21] Goirik Chakrabarty, Manogna Sreenivas, and Soma Biswas,
“A simple signal for domain shift,”
in Proceedings of
the IEEE/CVF International Conference on Computer Vision,
2023, pp. 3577–3584.
[22] Da Li,
Yongxin Yang,
Yi-Zhe Song,
and Timothy M
Hospedales, “Deeper, broader and artier domain generaliza-
tion,” in Proceedings of the IEEE international conference on
computer vision, 2017, pp. 5542–5550.
[23] Chen Fang, Ye Xu, and Daniel N Rockmore, “Unbiased met-
ric learning: On the utilization of multiple datasets and web
images for softening bias,” in Proceedings of the IEEE inter-
national conference on computer vision, 2013, pp. 1657–1664.
[24] Dan Hendrycks and Thomas Dietterich, “Benchmarking neu-
ral network robustness to common corruptions and perturba-
tions,” in International Conference on Learning Representa-
tions, 2019.
[25] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun,
“Deep residual learning for image recognition,” in Proceed-
ings of the IEEE conference on computer vision and pattern
recognition, 2016, pp. 770–778.