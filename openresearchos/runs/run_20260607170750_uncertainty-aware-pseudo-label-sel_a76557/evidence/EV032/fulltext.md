<!-- page 1 -->
Reliable fairness auditing with semi-supervised inference
Jianhui Gao and Jessica Gronsbell∗
Department of Statistics, University of Toronto, Toronto, ON, Canada
May 19, 2026
Abstract
Machine learning (ML) models often exhibit bias that can exacerbate inequities in
biomedical applications. Fairness auditing, the process of evaluating a model’s per-
formance across subpopulations, is critical for identifying and mitigating these biases.
However, audits typically rely on large volumes of labeled data, which are costly and
labor-intensive to obtain. To address this challenge, we introduce Infairness, a unified
framework for auditing a wide range of fairness criteria using semi-supervised infer-
ence. Our approach combines a small labeled dataset with a large unlabeled dataset
by imputing missing outcomes via regression with carefully selected nonlinear basis
functions. Through extensive theoretical and empirical analyses, we show that our
proposed estimator is (i) robust to specification of the ML or imputation model and
(ii) substantially more efficient than supervised estimation based solely on the labeled
data. In two real-world fairness audits using electronic health record and medical imag-
ing data, Infairness reduces variance by approximately 50% compared to supervised
estimation, underscoring its value for reliable fairness auditing with limited labeled
data.
Keywords:
Group fairness; Machine learning; Missing data; Semi-supervised inference
1
Introduction
Machine learning (ML) is used across healthcare and biomedical research to improve patient
outcomes, enhance diagnostic accuracy, and optimize treatment effectiveness (Rajpurkar
∗Email: j.gronsbell@utoronto.ca
1
arXiv:2505.12181v2  [stat.ME]  17 May 2026


<!-- page 2 -->
et al., 2022). However, mounting evidence demonstrates that ML models can exhibit un-
fairness by making less favorable decisions for certain groups or individuals (Mehrabi et al.,
2021). In a landmark study, Obermeyer et al. illustrated the damaging impact of algorithmic
bias in a commercial system used to guide treatment decisions for millions of patients in the
U.S. (Obermeyer et al., 2019). The algorithm predicted healthcare cost as a proxy for health-
care need. As a result, Black patients were disproportionately under-identified for high-risk
care management programs as they incur lower costs than White patients with comparable
medical burden due to structural disparities in healthcare access and treatment. This study
fortified a more socially conscious approach to ML deployment and led to further scrutiny
of ostensibly accurate models in biomedical applications. A subsequent systematic audit
of computer vision algorithms across diverse medical imaging modalities found that nearly
all models exhibited performance disparities across protected attributes, including age, sex,
and race (Xu et al., 2024). A benchmarking study of widely used electronic health record
(EHR) phenotyping algorithms for pneumonia and sepsis identified significant variability in
diagnostic performance across groups defined by gender, race, and ethnicity (Ding et al.,
2024). These findings collectively highlight that algorithmic unfairness is not only prevalent,
but that it can perpetuate and deepen existing inequity if left unaddressed.
In response, researchers now utilize a variety of statistical methods to discover and quan-
tify such disparities, a process referred to as “group fairness auditing” (Rajkomar et al.,
2018). While a central component of responsibly deploying ML systems, reliable audits re-
quire sufficiently large labeled datasets to detect differences in subgroup performance. This
is a major bottleneck in biomedical applications as labeled data is often scarce due to the
specialized expertise, time, and cost required for annotation. For instance, it takes an an-
notator 2-15 minutes to label a medical image whereas commonplace images, such as those
in ImageNet, can be labeled at a rate of two images per second through crowdsourcing (Li,
2010). In the analysis of EHR data, labeling a single discharge summary can take up to 10
minutes and often involves multiple experienced clinicians to ensure consensus (Gehrmann
et al., 2018).
Consequently, fairness audits within the biomedical domain often fall within a semi-
supervised (SS) setting, in which a small labeled dataset is accompanied by a much larger
unlabeled one. Traditional supervised auditing approaches rely solely on the labeled data,
which limits estimation precision and increases the risk of missing performance disparities.
While SS inference has been widely used to improve efficiency by leveraging both labeled
and unlabeled data, it has yet to be rigorously developed in the context of group fairness
auditing. To fill this gap, we introduce a unified SS framework, which we term Infairness,
that improves the precision of group fairness audits and, in turn, enables more powerful
2


<!-- page 3 -->
detection of subgroup differences without increasing labeling demands.
The remainder of this article is organized as follows. Section 2 reviews relevant literature.
Section 3 introduces the problem set-up.
Section 4 details the methodology.
Section 5
presents theoretical results. Sections 6 and 7 presents simulated and real data examples,
respectively. We conclude in Section 8 with a discussion of the implications of our work.
2
Related literature
We first review related areas of statistics and ML to situate our proposal within the context
of classical and modern literature.
2.1
Missing data and semi-supervised (SS) inference
The goal of SS inference is to determine whether and how unlabeled data can be leveraged to
improve the efficiency of supervised estimators based on the labeled data. SS inference is a
special case of a missing data problem in which the positivity assumption is violated due to
the disparate sizes of the labeled and unlabeled data. Existing work has focused on familiar
problems such as mean estimation, regression, and M-estimation (e.g., Zhang et al., 2019;
Song et al., 2023; Xu et al., 2025). An underexplored area is model performance evaluation.
Gronsbell and Cai (2018) proposed a nonparametric method for estimation of the ROC
curve and Gronsbell et al. (2022) extended these ideas for estimation of the misclassification
error and calibration. Related approaches have also appeared within transfer and weakly
supervised learning (e.g., Gao et al., 2024; Kiyasseh et al., 2024). These methods, however,
are designed for overall model performance evaluation rather than fairness auditing.
2.2
Prediction-powered inference
SS inference is closely related to prediction-powered inference (PPI). PPI is a modern ex-
tension of a missing problem, where predictions from pre-trained ML models are used to
impute missing outcomes (Angelopoulos et al., 2023). PPI has been primarily studied for
Z-estimation and only recently extended for evaluating binary classification performance
(Boyeau et al., 2025). PPI approaches reduce the variance of supervised estimation with a
control-variate derived from the predictions in the unlabeled data. Our proposal instead uses
an imputation-based approach and differs in two important ways. First, PPI methods focus
on asymptotic regimes where the labeled size grows proportionally with the total sample size.
Many biomedical settings fall within the SS framework in which positivity is violated, which
our method accommodates. Second, we propose to recalibrate the predictions using auxiliary
covariates that are often available during fairness auditing to further improve efficiency.
3


<!-- page 4 -->
2.3
Group fairness methods
Traditionally, group fairness audits rely on supervised estimation based on labeled data,
which can lead to high variance when subgroup sample sizes are small. Several methods
address this limitation by borrowing information from the full labeled dataset. Miller et al.
(2021) proposed model-based estimators for subgroup accuracy measures while van Breugel
et al. (2023) used generative models to augment labeled data with synthetic data. These
approaches, however, do not leverage unlabeled data. The only existing SS method for fair-
ness auditing is the beta calibration (BC) approach of Ji et al. (2020), which targets a small
number of fairness metrics. While flexible, BC relies on parametric assumptions that, when
violated, can lead to biased, albeit very precise, inferences about subgroup performance. This
behavior can foster overconfidence in a distorted understanding of a model’s performance and
undermine the scientific rationale for utilizing SS inference. Our proposal is guaranteed to
provide consistent estimation and valid inference without relying on parametric assumptions.
3
Preliminaries
3.1
Problem set-up
We consider an ML model for predicting a binary outcome Y . As our focus is on group
fairness auditing, we aim to evaluate whether the performance of the model differs across
groups defined by a categorical protected attribute A. The ML model is trained on features
X, which may or may not include A. X need not be a vector and may be an image, text,
tensor, or other type of data.
During validation, a vector of auxiliary covariates W =
(W1, . . . , Wp)T if often available for analysis.
For example, in medical imaging or EHR
applications, Y may be a disease of interest, X a chest X-ray or discharge summary, and
W a set of additional protected attributes or clinical variables not utilized during model
training.
With a slight abuse of notation, the validation data therefore consists of a labeled dataset
with n independent and identically distributed (i.i.d.)
samples, {(Yi, XT
i , WT
i , Ai)T | i =
1, ..., n}, together with an independent unlabeled dataset with N i.i.d. realizations of
{(XT
i , WT
i , Ai)T | i = n + 1, ..., n + N}. We focus on the classical SS setting in which it is
assumed that (i) the labeled and unlabeled data arise from the same underlying distribution
(or equivalently that Y is missing completely at random) and (ii) n/N →0 as n →∞(e.g.,
Chakrabortty and Cai, 2018; Zhang et al., 2019). This setting occurs when an investigator
has access to a large set of unlabeled data and a limited budget is available to select samples
uniformly at random for labeling such as in the imaging and EHR examples from Section 1.
To ensure that group-specific performance metrics can be estimated, we further assume
4


<!-- page 5 -->
that there is sufficient labeled and unlabeled data within each of the groups defined by A.
To this end, let na = Pn
i=1 I(Ai = a) and Na = Pn+N
i=n+1 I(Ai = a) denote the number of
labeled and unlabeled observations in the group with A = a, respectively. We assume that
na/n →ρa ∈(0, 1) as n →∞and Na/N →ρa ∈(0, 1) as N →∞for each a. These
assumptions, together with assumption (ii), imply that na/Na →0 as n →∞.
We make no assumptions on the ML model, denoted as ˆf, other than that it is trained
on a dataset, Dtrain, that is independent of the validation dataset. In practice, models are
commonly trained with weakly-supervised methods that accommodate scenarios with scarce
labeled data or simply use pre-trained models, such as LLMs or computer vision algorithms
(e.g., Yu et al., 2018; Seyyed-Kalantari et al., 2021). The predictions of Y are denoted as
S = ˆf(X) and are available for all (n + N) observations in the validation data. The final
classification for Y is D = I(S ≥c) for some pre-specified threshold c ∈(0, 1).
3.2
Parameter of interest
To quantify the performance of the ML model in predicting Y within groups defined by
A, we consider a wide range of group fairness metrics based on accuracy, discrimination,
and calibration. Table 1 summarizes common group-specific metrics that measure model
performance within each group with A = a, denoted generally as Ma. For example, the
group-specific TPR is defined as TPRa := P (D = 1 | Y = 1, A = a) . Note that the proba-
bility is taken over the distribution of (Y, D, A) conditioning on Dtrain. To simplify notation,
we omit explicit conditioning on Dtrain, with the understanding that all target parameters
and corresponding inference procedures are defined conditionally on the training data. To
unify notation for the various group-specific metrics, we define µZ
a = E(Z | A = a) for a
random variable Z. For example, the group-specific TPR is TPRa = µDY
a
µYa .
Existing work within the fairness literature primarily focuses on comparing group-specific
metrics across two groups defined by a binary protected attribute (Mehrabi et al., 2021; Gao
et al., 2025). We similarly focus on the two-group setting and discuss extensions to the multi-
group setting in Section 8. In the two-group setting, fairness criteria are most often defined in
terms of the difference between the group-specific metrics, denoted as ∆M = M0 −M1 and
the goal is to conduct inference on ∆M. Table 2 summarizes common group fairness criteria
(Hardt et al., 2016). Again using the TPR as an example, ∆TPR = TPR0 −TPR1, and when
∆TPR = 0 the model is said to satisfy the equal opportunity criterion. For exposition, we use
equal opportunity as a running example in Section 4.
5


<!-- page 6 -->
Table 1: Definitions and unified notation for group-specific performance metrics.
Notations: Y : outcome, A: protected attribute, S: model prediction, D: model classification
based on thresholding S, Ma: model performance metric within the group with A = a,
µZ
a = E(Z | A = a). Acronyms: TPR: True positive rate, FPR: False positive rate, PPV:
Positive predictive value, NPV: Negative predictive value, , F1: F1-score, ACC: Accuracy,
BS: Brier score.
Ma
Definition
Unified Notation
TPRa
P (D = 1 | Y = 1, A = a)
µDY
a
/µY
a
FPRa
P (D = 1 | Y = 0, A = a)
(µD
a −µDY
a
)/(1 −µY
a )
PPVa
P (Y = 1 | D = 1, A = a)
µDY
a
/µD
a
NPVa
P (Y = 0 | D = 0, A = a)
(1 −µD
a −µY
a + µDY
a
)/(1 −µD
a )
F1a
2/
 TPR−1
a + PPV−1
a
 
2µDY
a
/(µD
a + µY
a )
ACCa
1 −E
 
(Y −D)2 | A = a
	
1 −µY
a −µD
a + 2µDY
a
BSa
E
 
(Y −S)2 | A = a
	
µS2
a −2µSY
a
+ µY
a
Table 2: Common group fairness criteria and corresponding definitions in the
setting of a binary protected attribute. A model satisfies a given fairness criterion
if the corresponding fairness measure(s), ∆M = M0 −M1, is zero. Common choices of
group-specific performance, Ma, include the true positive rate (TPR), false positive rate
(FPR), positive predictive value (PPV), negative predictive value (NPV), accuracy (ACC),
Brier score (BS), and F1 score (F1).
Fairness Criterion
Definition
Equal Opportunity
∆TPR = 0
Predictive Equality
∆FPR = 0
(Positive) Predictive Parity
∆PPV = 0
(Negative) Predictive Parity
∆NPV = 0
F1 Score Parity
∆F1 = 0
Overall Accuracy Equality
∆ACC = 0
Brier Score Parity
∆BS = 0
6

[CAPTION] Table 1: Definitions and unified notation for group-specific performance metrics.

[CAPTION] Table 2: Common group fairness criteria and corresponding definitions in the


<!-- page 7 -->
4
Methods
4.1
Review of supervised inference for group fairness auditing
We first revisit the traditional supervised approach to evaluating group fairness using only
the labeled data. Constructing supervised estimators for the metrics in Table 1 requires
estimating µZ
a for Z ∈{Y, D, S2, SY, DY } within the labeled data as n−1
a
Pn
i=1 ZiI(Ai = a).
Supervised estimators for Ma and ∆M are then obtained by substituting the estimator of
µZ
a into the formulas provided in Table 1. For example, the estimator of ∆TPR is
b∆
SUP
TPR := [
TPR
SUP
0
−[
TPR
SUP
1
= n−1
0
Pn
i=1 DiYiI(Ai = 0)
n−1
0
Pn
i=1 YiI(Ai = 0)
−n−1
1
Pn
i=1 DiYiI(Ai = 1)
n−1
1
Pn
i=1 YiI(Ai = 1) .
To conduct inference on ∆TPR, large sample confidence intervals can be constructed based on
the asymptotic normality of b∆SUP
TPR detailed in Theorem S1 in the Supplementary Material.
4.2
Infairness: Proposed SS inference method for group fairness
auditing
The key distinction between the SS and supervised approaches to inference is that the SS
approach uses the large unlabeled dataset alongside the small labeled dataset to improve
estimation efficiency.
Intuitively, the efficiency gain arises from exploiting the abundant
unlabeled data within each group defined by A, which provides near-complete information
on the joint distribution of the prediction, S, and the auxiliary covariates, W. This intuition
provides the underlying motivation for our proposal, Infairness.
Analogously to the supervised setting, constructing SS estimators for the group fairness
metrics in Table 1 requires estimation of µZ
a for Z ∈{Y, D, S2, SY, DY }. The Infairness
estimators for Ma and ∆M are obtained by substituting the SS estimators of µZ
a into the
formulas provided in Table 1. To this end, we consider two cases. First, consider µZ
a for
Z ∈{Y, SY, DY } as these functionals depend on Y , which is only available in the labeled
data. To illustrate the potential utility of the unlabeled data in estimation, note that µZ
a =
E(Z | A = a) can be rewritten as
µZ
a = E{ϕ(Z)E(Y | S, W, A = a) | A = a}
where
ϕ(Z) =









1
if Z = Y
S
if Z = SY
D
if Z = DY.
(1)
The expression in (1) highlights that µZ
a inherently depends on the distribution of (S, WT)T |
A = a and hence estimation may be improved by making careful use of the unlabeled data.
7


<!-- page 8 -->
Moreover, a natural SS estimator can be obtained by constructing the empirical analogue
of (1).
Specifically, we estimate µZ
a by (i) imputing the missing Y with an estimate of
E(Y | S, W, A = a) learned from the labeled data and (ii) averaging the resulting imputa-
tions over the unlabeled data in the group with A = a. The potential efficiency gain relative
to supervised estimation hinges on the quality of the imputation in step (i) and is driven by
step (ii), which leverages the abundant unlabeled data. In contrast, for the second case with
Z ∈{D, S2}, µZ
a does not depend on Y and therefore estimation does not require imputa-
tion. We can simply obtain estimators for µZ
a with the corresponding empirical averages in
the unlabeled data. Similar to the imputation-based SS estimators, these estimators offer
improved precision relative to their supervised counterparts due to averaging over a much
larger sample size.
4.2.1
Imputation strategy
The key challenge in an imputation-based approach is balancing robustness with efficiency.
The imputations must be constructed in such a way that the Infairness estimator is con-
sistent for its target while also yielding precision gains from the unlabeled data. A fully
nonparametric method achieves both goals, but generally does not perform well when the
dimension of (S, WT)T is moderate due to the curse of dimensionality (Chakrabortty and
Cai, 2018; Tan et al., 2025). We therefore propose to impute Y within each group by fitting
the working model
E(Y | S, W, A = a) = g (θ
T
aBa)
(2)
where Ba = Ba(S, W) is a set of basis functions of fixed dimension and g(·) : R →[0, 1]
is a specified smooth monotone function such as the expit function. The basis expansion
can include nonlinear and interaction effects to achieve a flexible representation of E(Y |
S, W, A = a) and can also be carefully designed to ensure consistency of the Infairness
estimator. With respect to the former, we suggest using flexible spline models and study the
practical impact of the choice of basis in our empirical studies in Section 6. With respect to
the latter, we propose to estimate θa from the labeled data with bθa, which is the solution to
n−1
a
n
X
i=1
Ba,i {Yi −g (θ
T
aBa,i)} I(Ai = a) −λnaθa = 0
(3)
and λna = o
 
n
−1
2
a
 
is a penalty term used to stabilize model fitting. We show in Supple-
mentary Section S2.3.1 that bθa is consistent for θa, defined as the solution to E[Ba{Y −
g(θT
aBa)}I(A = a)] = 0. The imputations are computed as bma = g(bθT
aBa) within the un-
labeled data and used to evaluate the fairness criteria. For example, returning to equal
8


<!-- page 9 -->
opportunity, the Infairness estimator of ∆TPR is:
b∆
SS
TPR = [
TPR
SS
0 −[
TPR
SS
1 = N −1
0
Pn+N
i=n+1 Di bm0,iI(Ai = 0)
N −1
0
Pn+N
i=n+1 bm0,iI(Ai = 0)
−N −1
1
Pn+N
i=n+1 Di bm1,iI(Ai = 1)
N −1
1
Pn+N
i=n+1 bm1,iI(Ai = 1)
.
Importantly, the Infairness estimators are consistent for their targets provided that SS
estimators of µZ
a are consistent for their targets. For Z ∈{Y, SY, DY }, µZ
a depends on Y
and the SS estimators are based on the imputations for Y . We therefore require that
E{(1, D, S)
T(Y −bma)I(A = a) | bma}
p→0 as na →∞.
(4)
While this condition is satisfied if the working model in (2) is correctly specified, the func-
tional form of E(Y | S, W, A = a) may be misspecified in practice. We can, however, ensure
that the condition (4) is satisfied by simply including an intercept, S, and D in the basis
functions as bθa is derived from the estimating equation in (3).
5
Asymptotic results
We next summarize the key asymptotic properties of the Infairness estimator in Theorem 1.
Detailed derivations are provided in Sections S2.2 – S2.3.
Theorem 1 (Influence function of ˆ∆SS
M). Under the assumptions in Section 3.1 and
regularity conditions in Section S2.1, ˆ∆SS
M
p→∆M and
n
1
2
 
ˆ∆
SS
M −∆M
 
= n−1
2
X
a∈{0,1}
(−1)aρ−1
a
n
X
i=1
IF
SS
Ma
 Yi, Ba,i; c, θa
 
I(Ai = a) + op(1)
where IF
SS
Ma(Yi, Ba,i; c, θa) is specified in Table 3.
Therefore, n
1
2
 
ˆ∆SS
M −∆M
 
converges
weakly to a zero-mean Gaussian distribution with variance
X
a∈{0,1}
ρ−1
a E
h 
IF
SS
Ma
 Yi, Ba,i; c, θa
 	2 | A = a
i
.
Theorem S1 in the Supplementary Materials provides similar results for the supervised
estimator. We utilize Theorems S1 and 1 to compare the asymptotic variances of the two
estimators in the following corollary, with the proof detailed in Section S2.4.
Corollary 1 (Variance comparison of ˆ∆SUP
M
and ˆ∆SS
M). When the imputation model in
9


<!-- page 10 -->
Table 3: Influence functions of ˆ∆SUP
M
and ˆ∆SS
M.
Notations: Y : outcome, S: model
prediction, c: cutoff, D: model classification based on thresholding S at c, Ma: model
performance metric within the group with A = a, µZ
a = E(Z | A = a), Ba = Ba(S, W): a
finite set of basis functions of fixed dimension that includes an intercept, g(·) : R →[0, 1]:
a specified smooth monot one function, θa: solution to E[Ba{Y −g(θTBa)}I(A = a)] = 0.
Acronyms: TPR: True positive rate, FPR: False positive rate, PPV: Positive predictive
value, NPV: Negative predictive value, F1: F1-score, ACC: Accuracy, BS: Brier score.
Ma
IF
SUP
Ma(Yi, Si; c)
IF
SS
Ma(Yi, Ba(Si, Wi); θa, c)
TPRa
(µY
a )−1{Yi(Di −TPRa)}
(µY
a )−1{Yi −g(θ
T
aBa,i)} (Di −TPRa)
FPRa
(1 −µY
a )−1 {(1 −Yi)(Di −FPRa)}
(1 −µY
a )−1{Yi −g(θ
T
aBa,i)} (FPRa −Di)
PPVa
(µD
a )−1 {Di(Yi −PPVa)}
(µD
a )−1Di{Yi −g(θ
T
aBa,i)}
NPVa
(1 −µD
a )−1 {(1 −Di)(1 −Yi −NPVa)}
(1 −µD
a )−1(Di −1){Yi −g(θ
T
aBa,i)}
F1a
 µD
a + µY
a
 −1 {Di (Yi −F1a) + Yi (Di −F1a)}
 µD
a + µY
a
 −1 {Yi −g(θ
T
aBa,i)} (2Di −F1a)
ACCa
1 −(Yi −Di)2 −ACCa
{Yi −g(θ
T
aBa,i)}(2Di −1)
BSa
(Si −Yi)2 −BSa
{Yi −g(θ
T
aBa,i)}(1 −2Si)
equation (2) is correctly specified and E(Y | S, W, A = a) ̸= E(Y | A = a),
X
a∈{0,1}
ρ−1
a E
h 
IF
SS
Ma(Yi, Ba,i; c, θa)
	2 | A = a
i
<
X
a∈{0,1}
ρ−1
a E
h 
IF
SUP
Ma(Yi, Si; c)
	2 | A = a
i
.
Corollary 1 provides a sufficient condition under which the Infairness estimator is the-
oretically guaranteed to be more efficient than the supervised estimator. Specifically, this
occurs when the group-specific imputation models are correctly specified and when the un-
labeled data contains meaningful information about Y . This result underlies our motivation
in constructing the Infairness estimator, that is, the imputation model should be specified
flexibly enough so that it can extract as much information as possible from the unlabeled
data. While we cannot provide strict theoretical guarantees when the imputation model is
misspecified, our proposal will generally yield efficiency gains provided the imputation model
is a close approximation to E(Y | S, W, A = a). We corroborate this heuristic justification
in our empirical studies in Sections 6 and 7.
6
Simulation studies
We consider two sets of simulation studies. First, we consider a stylized setting in which
we directly generate S to provide intuition for the Infairness procedure and to illustrate its
potential benefit over its comparators. Second, we consider a more traditional set-up where
we fit an ML model on an independent dataset to generate S. For both studies, we present
results for n = 400 and N = 20, 000 and include results for n = 1000 in Section S3.4.
10

[CAPTION] Table 3: Influence functions of ˆ∆SUP


<!-- page 11 -->
6.1
Evaluation metrics and methods
We compare the supervised estimator with 3 SS approaches: the beta calibration (BC)
approach of Ji et al. (2020), the nonparametric (NP) approach of Gronsbell and Cai (2018),
and the proposed Infairness method. The BC and NP approaches base imputation entirely
on S and do not utilize W. The BC approach imputes the missing Y with beta calibrated
predictions while the NP approach estimates E(Y | S) with kernel smoothing. For Infairness,
we impute Y using natural cubic spline regression models with three knots based on (i) S
alone (Infairness(S)), (ii) S with additive effects of W (Infairness(S + W)), and (iii) S with
interaction effects between S and W (Infairness(S × W)). Further details regarding the
implementation of each method are provided in Section S1.3.
We assess bias, coverage probability (CP), and relative efficiency (RE). Bias is evaluated
by comparing average point estimates to their true values. The true parameter values are
approximated using supervised estimates derived from a fully labeled dataset of size 106. CP
is assessed by the empirical coverage of 95% confidence intervals constructed using asymptotic
standard errors.
We utilize 10-fold cross-validation for standard error estimation for all
methods to improve finite sample performance. Further details are provided in Section S1.3.
RE is defined as the ratio of the mean squared error (MSE) of the supervised estimator to
that of each SS method. All results are summarized over 10,000 Monte Carlo replicates.
6.2
Stylized setting
We consider two scenarios in the stylized setting.
In both scenarios, we generate Y ∼
Bernoulli(0.3), A ∼Bernoulli(0.4), and an auxiliary covariate W ∼Bernoulli(0.5). The
prediction S was generated so that S | Y = y, A = a, W = w ∼Beta(αy,w,a, βy,w,a). Scenario
1 considers an ideal setting in which S ⊥(A, W) | Y so that the assumption of the BC
method is satisfied and S performs equally well across groups defined by A and W. Scenario
2 considers a setting in which the distribution of S | Y also depends on (A, W) and S
has variable performance across groups. In both scenarios, D = I(S > 0.5). A detailed
description of the data generation is provided in Supplemental Section S3.1.
For simplicity, we continue with our running example of equal opportunity. Figure 1
summarizes the results. In Scenario 1, BC, NP, and the Infairness estimators exhibit mini-
mal bias, achieve nominal coverage, and improve efficiency relative to supervised estimation.
The BC method is the most efficient approach as its parametric assumptions are satisfied.
However, in Scenario 2, the assumptions of the BC method are violated, which results in
coverage near 70% and substantial underestimation of the difference in the group-specific
TPRs. The BC method also yields estimates with improved efficiency over supervised es-
timation. Practically, this translates into an overconfidence that S is less biased than it
11


<!-- page 12 -->
Scenario 1
Scenario 2
(a) Point estimate
−0.1
0.0
0.1
Difference in true positive rate
−0.3
−0.2
−0.1
0.0
Difference in true positive rate
(b) Coverage probability
0.80
0.85
0.90
0.95
1.00
Coverage probability
0.7
0.8
0.9
1.0
Coverage probability
(c) Relative efficiency
0.0
0.5
1.0
1.5
2.0
Supervised
BC
NP
Infairness
(S)
Infairness
(S + W)
Infairness
(S x W)
Relative efficiency
0
1
2
Supervised
BC
NP
Infairness
(S)
Infairness
(S + W)
Infairness
(S x W)
Relative efficiency
Figure 1: Stylized setting for estimating the difference in true positive rates.
Results are based on 104 Monte Carlo replicates with 20,000 validation observations and 400
labeled observations per replicate. Panel (a) shows the average point estimate of ∆TPR =
TPR0 −TPR1, with vertical lines corresponding to the average estimate plus or minus 1.96
times the empirical standard error; the dashed horizontal line marks the truth. Panel (b)
shows empirical coverage probability of nominal 95% confidence intervals, with the shaded
region indicating coverage between 0.90 and 1.00. Panel (c) shows relative efficiency, defined
as the ratio of the supervised mean squared error to the mean squared error of each method.
truly is. With n = 1000, CP deviates further from the nominal level and falls below 40%
(Figure S1). The NP method remains approximately unbiased with near-nominal cover-
age, but is less efficient than Infairness(S × W), which effectively leverages the additional
information in W. As the additive effect of W is minimal by design in this scenario, the
alternative Infairness approaches do not properly utilize the information in W. However,
both approaches do not incur efficiency loss relative to NP. Although an oversimplification of
a real-world audit, these findings underscore the gap that Infairness fills: it always provides
consistent estimation of the target parameter while also improving over NP when additional
information in W is adequately incorporated into the imputation model. This take-away is
reinforced in the more complex simulation scenarios considered in Section 6.3.
6.3
Traditional setting
We next generate (X⊤, W⊤, A)⊤from a 16-dimensional mean-zero multivariate normal. The
covariance between the kth and ℓth random variables is 3 · (0.4)|k−ℓ|. The first 10 variables
12


**[Table p12.1]**
| Scenario 1 Scenario 2 (a) Point estimate |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rate 0.1 positive 0.0 true in Difference −0.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.0 rate −0.1 positive true −0.2 in Difference −0.3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| (b) Coverage probability |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 1.00 probability 0.95 0.90 Coverage 0.85 0.80 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1.0 probability 0.9 0.8 Coverage 0.7 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| (c) Relative efficiency |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 2.0 efficiency 1.5 1.0 Relative 0.5 0.0 Supervised BC NP Infairness Infairness Infairness (S) (S + W) (S x W) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 2 efficiency 1 Relative 0 Supervised BC NP Infairness Infairness Infairness (S) (S + W) (S x W) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

[CAPTION] Figure 1: Stylized setting for estimating the difference in true positive rates.


<!-- page 13 -->
are used as features X for model training, the next 5 as auxiliary covariates W available
at auditing, and the final variable is thresholded to form the binary protected attribute A
with prevalence 0.4. The outcome Y is generated under four scenarios of varied complexity
that impact the fit of the ML model used to generate S. Scenario A uses a logistic model
in which the linear predictor contains nonlinear and interaction effects. Scenario B applies
a nonlinear transformation to the linear predictor. Scenario C uses a complementary log-log
model with a nonlinear transformation of the linear predictor. Scenario D uses a tree-based
model in which the probability of the outcome is constant within regions defined by X, W,
and A. Detailed data generating mechanisms are provided in Section S3.2. In Section S3.6,
we also consider an extreme scenario in which S is completely uninformative of Y . In all
scenarios, S is obtained by fitting a logistic regression model of Y on X using an independent
labeled dataset of size 104 generated from the same distribution. The final classification is
taken as D = I(S > 0.5).
While selecting appropriate metrics for fairness auditing is
context-dependent, we consider all metrics in Table 2 for the purposes of illustration.
The results are presented in Figure 2 and are consistent with the take-aways of the stylized
simulation study. Specifically, across all scenarios, the supervised estimator, NP, and the
Infairness estimators exhibit minimal bias for all metrics and maintain coverage close to the
nominal level. In Scenarios A and B where model misspecification is less severe, BC is also
approximately unbiased, though with slightly lower coverage. In Scenarios C and D, BC
exhibits substantial bias and under-coverage. Analogously to the stylized setting, increasing
the labeled sample size does not remove this bias. With n = 1,000, BC remains biased and
the CP drops to approximately 80-85% in Scenario C and 69-81% in Scenario D (Figure S2).
We therefore focus our comparisons of efficiency on the NP and Infairness methods.
Both NP and Infairness generally improve upon the supervised estimator in all settings.
However, the Infairness approaches that adjust for W often provides additional gains over
NP, particularly for ∆F1, ∆FPR, and ∆TPR. In Scenarios A and B, Infairness(S × W) and
Infairness(S + W) provide gains of 16-37% over NP and Infairness(S), while in Scenarios C
and D, the gains are approximately 10%. This pattern reflects the benefit of incorporating
auxiliary covariates W when they contain information about Y beyond S. It is also im-
portant to note that Infairness(S) is generally no worse than NP and Infairness(S × W) is
no worse than using Infairness(S + W). This suggests that richer imputation models are
generally beneficial when the labeled sample size is sufficient to support the added model
complexity.
13


<!-- page 14 -->
−0.2
−0.1
0.0
0.1
0.2
Estimate
Scenario A
0.80
0.85
0.90
0.95
1.00
CP
0
1
2
∆ACC
∆BS
∆F1
∆FPR
∆NPV
∆PPV
∆TPR
Relative efficiency
−0.2
−0.1
0.0
0.1
0.2
Estimate
Scenario B
0.80
0.85
0.90
0.95
1.00
CP
0.0
0.5
1.0
1.5
2.0
2.5
∆ACC
∆BS
∆F1
∆FPR
∆NPV
∆PPV
∆TPR
Relative efficiency
−0.2
0.0
0.2
0.4
Estimate
Scenario C
0.80
0.85
0.90
0.95
1.00
CP
0.0
0.5
1.0
1.5
2.0
2.5
∆ACC
∆BS
∆F1
∆FPR
∆NPV
∆PPV
∆TPR
Relative efficiency
−0.4
−0.2
0.0
0.2
Estimate
Scenario D
0.80
0.85
0.90
0.95
1.00
CP
0
1
2
∆ACC
∆BS
∆F1
∆FPR
∆NPV
∆PPV
∆TPR
Relative efficiency
Supervised
BC
NP
Infairness(S)
Infairness(S + W)
Infairness(S x W)
Figure 2: Group fairness auditing across simulation scenarios. Results are based on
104 Monte Carlo replicates with 400 labeled and 20,000 unlabeled observations per replicate.
Each panel corresponds to one simulation scenario. For each scenario, the three rows report
point estimates, empirical coverage probability (CP), and relative efficiency (RE) for group
differences in the performance metrics. In the point estimate row, vertical bars indicate
approximate 95% intervals based on empirical standard errors, and black horizontal bar
mark the oracle values. In the CP row, the dashed line marks nominal 95% coverage and
the shaded region indicates coverage between 0.90 and 1.00. In the RE row, RE is defined
as the ratio of the supervised mean squared error (MSE) to the MSE of each method.
14


**[Table p14.1]**
| Scenario A 0.2 0.1 Estimate 0.0 −0.1 −0.2 | Scenario B 0.2 0.1 Estimate 0.0 −0.1 −0.2 |
| --- | --- |
|  |  |
| 1.00 0.95 0.90 CP 0.85 0.80 | 1.00 0.95 0.90 CP 0.85 0.80 |
| y | y |
| efficienc 2 1 Relative 0 ∆ ∆ ∆F1 ∆FPR ∆NPV ∆PPV ∆TPR ACC BS | 2.5 efficienc 2.0 1.5 1.0 Relative 0.5 0.0 ∆ ∆ ∆F1 ∆FPR ∆NPV ∆PPV ∆TPR ACC BS |
|  |  |
| Scenario C 0.4 Estimate 0.2 0.0 −0.2 | Scenario D 0.2 Estimate 0.0 −0.2 −0.4 |
|  |  |
| 1.00 0.95 0.90 CP 0.85 0.80 | 1.00 0.95 0.90 CP 0.85 0.80 |
| y | y |
| 2.5 efficienc 2.0 1.5 1.0 Relative 0.5 0.0 ∆ ∆ ∆F1 ∆FPR ∆NPV ∆PPV ∆TPR ACC BS Supervised BC NP Infairn | efficienc 2 1 Relative 0 ∆ ∆ ∆F1 ∆FPR ∆NPV ∆PPV ∆TPR ACC BS ess(S) Infairness(S + W) Infairness(S x W) |

[CAPTION] Figure 2: Group fairness auditing across simulation scenarios. Results are based on


<!-- page 15 -->
7
Real data analysis
We next apply our method to audit fairness in an EHR-based phenotyping model for de-
pression (Gehrmann et al., 2018). We focus on equal opportunity as lower rates of disease
detection and diagnosis are often found among high-risk individuals within certain subgroups
and, in many use cases, the benefit of diagnosis often outweighs potential harm. However,
as the choice of fairness metric is ultimately use-case dependent, we also consider predictive
equality for purposes of illustration. It is worth noting that metrics such as PPV, NPV, and
F1-score depend on the disease prevalence, which differs across groups in these data, making
them less directly comparable (Gao et al., 2025). Throughout these analyses, we compare
the three estimators from Section 6 that provide consistent estimation of the fairness metrics:
supervised, NP, and Infairness. For Infairness, the group-specific imputation models were
selected using BIC among the three candidate models in Section S1.3. An additional audit
of a chest X-ray classifier is provided in Section S4.2.
We consider depression phenotyping from discharge summaries from the Medical Infor-
mation Mart for Intensive Care III (MIMIC-III), a publicly available database of de-identified
EHR data for over 40,000 patients admitted to critical care units of the Beth Israel Deaconess
Medical Center between 2001 and 2012 (Johnson et al., 2016). We restrict our analysis to
each patient’s first admission and exclude patients classified as “frequent flyers” who have
more than three admissions within a calendar year (Gehrmann et al., 2018).
The gold-
standard depression label is available for n = 872 patients while N = 28,766 patients are
unlabeled. The ML model was trained with a weakly supervised approach and the details
are provided in Section S4.1. Fairness was audited across race (White: 73% vs non-White:
27%). The auxiliary covariates, W, included age, sex, marital status, and insurance type.
Figure 3 shows results for equal opportunity and predictive equality. The point estimates
from all three methods are generally within a reasonable range of one another and the 95%
confidence intervals suggest that there is no statistically significant difference in performance
across groups. The main difference lies in precision: NP and Infairness substantially reduce
variance relative to the supervised estimator, with Infairness providing the largest reduction.
The RE of the Infairness estimators of ∆TPR and ∆FPR exceed 1.77, implying that our
proposal reduces variance by at least 43.5% relative to the supervised estimator. Practically,
this translates into Infairness requiring 43.5% fewer labeled observations than the supervised
estimator to achieve the same precision. This is a 25% improvement over the efficiency gains
afforded by NP, which relies solely on S for imputation.
15

[CAPTION] Figure 3 shows results for equal opportunity and predictive equality. The point estimates


<!-- page 16 -->
−0.2
−0.1
0.0
0.1
Estimate
0.0
0.5
1.0
1.5
2.0
∆FPR
∆TPR
Relative Efficiency
Supervised
NP
Infairness
Figure 3: FPR and TPR disparity estimates in the EHR phenotyping application
across race. Disparities are defined as the difference in group-specific performance between
the non-White and White patients.
16

[CAPTION] Figure 3: FPR and TPR disparity estimates in the EHR phenotyping application


<!-- page 17 -->
8
Discussion
Motivated by ML in biomedical applications, we proposed a SS method, called Infairness, to
audit group fairness when labeled data is limited, but unlabeled data is abundant. Infairness
uses a carefully constructed imputation model to enable evaluation of fairness criteria using
both the labeled and unlabeled data. Our theoretical and empirical results highlight a simple
message. With a sufficiently flexible imputation model, Infairness can yield more efficient
inference than its supervised counterpart and often outperforms existing comparators when
W contains information about Y beyond S. Importantly, we did not observe efficiency loss
relative to existing approaches, suggesting the utility of Infairness for real-world auditing.
We close by commenting on several related issues and potential extensions.
First, although our primary focus is on inference for difference-based disparity metrics as
they are most commonly used in practice, Infairness is also applicable to to overall perfor-
mance metrics as well as ratio-based metrics (e.g., M0
M1) using inference procedures derived
from Theorem 1. In addition, Infairness supports hypothesis testing for ϵ-fairness, a relaxed
fairness criterion allowing group-specific disparities to fall within a small threshold, ϵ (Denis
et al., 2023). Second, many protected attributes define more than two groups, and methods
for evaluating multi-group fairness remain an active area of research. Early approaches relied
on visualizations of model performance across groups, using tools such as Aequitas (Saleiro
et al., 2019). Recent work has proposed meta-metrics, such as the max-min difference, max
absolute difference, and variance of group-specific performance metrics. However, Lum et al.
(2022) showed that supervised estimators for these meta-metrics are biased. Cherian and
Cand`es (2024) later framed multi-group fairness as a multiple hypothesis testing problem,
introducing a bootstrap-based method to jointly bound disparities across groups. Developing
SS approaches for multi-group settings is a focus of our ongoing work.
Third, while our emphasis has been on group fairness, individual fairness and causal fair-
ness are emerging within biomedical applications. Individual fairness requires that similar
individuals receive similar predictions, while causal fairness seeks to uncover causal rela-
tionships between protected attributes and model outputs (e.g., Castiglione et al., 2022).
Extending our framework to support these fairness notions is an avenue of future research.
Fourth, while we consider incorporating auxiliary covariates into the imputation step, includ-
ing features used for training the ML model may further improve efficiency, particularly when
the ML model is misspecified and the number of features is not large. However, care must
be taken to avoid overfitting and a resulting loss in the efficiency gain of the SS estimator
when the group-specific labeled sample sizes are not large. Additionally, small group-specific
sample sizes can impact the variance estimates of Infairness. In Section S3.6, we conducted a
17


<!-- page 18 -->
simulation study with n = 100 and the smallest group-specific sample size around 40. While
Infairness does have reduced coverage, its performance is no more impacted than supervised
estimation or alternative SS approaches (Figure S3).
Finally, Infairness relies on a design-based choice that labeled data are obtained as a
simple random sample from the underlying data pool.
In practice, alternative sampling
strategies can provide more efficient use of labeling resources.
For example, labels can
instead be sampled uniformly at random within levels of A for fixed values of n0 and n1,
in which case the proposed method remains valid. In other applications, the probability of
observing labels may depend on the fully observed variables (e.g., (XT, WT, A)T), implying
that Y is missing at random. Approaches that adjust for selection bias, such as inverse
probability weighting, would be required to obtain valid inference.
Ongoing work in SS
inference addresses more general missingness mechanisms and incorporating these advances
into our current framework is a practically useful direction (e.g., Zhang et al., 2023).
Acknowledgements
J. Gronsbell is grateful for support of an NSERC Discovery Grant (RGPIN-2021-03734) and
a Methodologist Seed Funding Grant from the University of Toronto Data Science Institute.
References
Anastasios N. Angelopoulos, Stephen Bates, Clara Fannjiang, Michael I. Jordan, and Tijana
Zrnic. Prediction-powered inference. Science, 382(6671):669–674, 2023. doi: 10.1126/
science.adi6000. URL https://www.science.org/doi/abs/10.1126/science.adi6000.
Pierre Boyeau, Anastasios Nikolas Angelopoulos, Tianle Li, Nir Yosef, Jitendra Malik, and
Michael I. Jordan. AutoEval done right: Using synthetic data for model evaluation. In
Proceedings of the 42nd International Conference on Machine Learning, volume 267, pages
5276–5290, 13–19 Jul 2025. URL https://proceedings.mlr.press/v267/boyeau25a.
html.
Giuseppe Castiglione, Ga Wu, Christopher Srinivasa, and Simon Prince.
fAux: Testing
Individual Fairness via Gradient Alignment, October 2022.
URL http://arxiv.org/
abs/2210.06288. arXiv:2210.06288 [stat].
Abhishek Chakrabortty and Tianxi Cai.
Efficient and adaptive linear regression
in semi-supervised settings.
The Annals of Statistics,
46(4):1541–1572,
August
2018.
ISSN 0090-5364,
2168-8966.
doi:
10.1214/17-AOS1594.
URL https:
//projecteuclid.org/journals/annals-of-statistics/volume-46/issue-4/
18


<!-- page 19 -->
Efficient-and-adaptive-linear-regression-in-semi-supervised-settings/
10.1214/17-AOS1594.full.
John J. Cherian and Emmanuel J. Cand`es.
Statistical Inference for Fairness Auditing.
Journal of Machine Learning Research, 25(149):1–49, 2024. ISSN 1533-7928. URL http:
//jmlr.org/papers/v25/23-0739.html.
Christophe Denis, Romuald Elie, Mohamed Hebiri, and Fran¸cois Hu.
Fairness guaran-
tee in multi-class classification, March 2023. URL http://arxiv.org/abs/2109.13642.
arXiv:2109.13642 [math].
Sirui Ding, Shenghan Zhang, Xia Hu, and Na Zou.
Identify and mitigate bias in elec-
tronic phenotyping:
A comprehensive study from computational perspective.
Jour-
nal of Biomedical Informatics, 156:104671, August 2024.
ISSN 1532-0464.
doi:
10.
1016/j.jbi.2024.104671. URL https://www.sciencedirect.com/science/article/pii/
S1532046424000893.
Jianhui Gao, Clara-Lea Bonzel, Chuan Hong, Paul Varghese, Karim Zakir, and Jessica
Gronsbell. Semi-supervised ROC analysis for reliable and streamlined evaluation of phe-
notyping algorithms. Journal of the American Medical Informatics Association, 31(3):
640–650, February 2024. ISSN 1067-5027, 1527-974X. doi: 10.1093/jamia/ocad226. URL
https://academic.oup.com/jamia/article/31/3/640/7486839.
Jianhui Gao, Benson Chou, Zachary R. McCaw, Hilary Thurston, Paul Varghese, Chuan
Hong, and Jessica Gronsbell.
What is fair?
defining fairness in machine learning for
health. Statistics in Medicine, 44(20–22):e70234, 2025.
Sebastian Gehrmann, Franck Dernoncourt, Yeran Li, Eric T. Carlson, Joy T. Wu, Jonathan
Welt, et al. Comparing deep learning and concept extraction based methods for patient
phenotyping from clinical narratives. PLoS ONE, 13(2):e0192360, February 2018. ISSN
1932-6203. doi: 10.1371/journal.pone.0192360. URL https://www.ncbi.nlm.nih.gov/
pmc/articles/PMC5813927/.
Jessica Gronsbell, Molei Liu, Lu Tian, and Tianxi Cai.
Efficient Evaluation of Pre-
diction Rules in Semi-Supervised Settings under Stratified Sampling.
Journal of the
Royal Statistical Society. Series B, Statistical methodology, 84(4):1353, April 2022. doi:
10.1111/rssb.12502. URL https://pmc.ncbi.nlm.nih.gov/articles/PMC9586151/.
Jessica L. Gronsbell and Tianxi Cai. Semi-supervised approaches to efficient evaluation of
model prediction performance. Journal of the Royal Statistical Society. Series B (Statistical
19


<!-- page 20 -->
Methodology), 80(3):579–594, 2018.
ISSN 1369-7412.
URL https://www.jstor.org/
stable/26773169.
Moritz Hardt,
Eric Price,
Eric Price,
and Nati Srebro.
Equality of Opportu-
nity in Supervised Learning.
In Advances in Neural Information Processing Sys-
tems, volume 29, 2016.
URL https://proceedings.neurips.cc/paper/2016/file/
9d2682367c3935defcb1f9e247a97c0d-Paper.pdf.
Disi Ji, Padhraic Smyth, and Mark Steyvers. Can I trust my fairness metric? assessing fair-
ness with unlabeled data and Bayesian inference. In Proceedings of the 34th International
Conference on Neural Information Processing Systems, NIPS’20, pages 18600–18612, De-
cember 2020. ISBN 978-1-71382-954-6.
Alistair E. W. Johnson, Tom J. Pollard, Lu Shen, Li-wei H. Lehman, Mengling Feng, et al.
MIMIC-III, a freely accessible critical care database. Scientific Data, 3(1):160035, May
2016.
ISSN 2052-4463.
doi: 10.1038/sdata.2016.35.
URL https://www.nature.com/
articles/sdata201635.
Dani Kiyasseh, Aaron Cohen, Chengsheng Jiang, and Nicholas Altieri.
A framework
for evaluating clinical artificial intelligence systems without ground-truth annotations.
Nature Communications, 15(1):1808, February 2024.
ISSN 2041-1723.
doi: 10.1038/
s41467-024-46000-9. URL https://www.nature.com/articles/s41467-024-46000-9.
Fei-Fei Li. Crowdsourcing, Benchmarking & Other Cool Things, 2010. URL https://www.
image-net.org/static_files/papers/ImageNet_2010.pdf. ImageNet.
Kristian Lum, Yunfeng Zhang, and Amanda Bower. De-biasing “bias” measurement. In
2022 ACM Conference on Fairness, Accountability, and Transparency, pages 379–389,
June 2022. ISBN 978-1-4503-9352-2. doi: 10.1145/3531146.3533105. URL https://dl.
acm.org/doi/10.1145/3531146.3533105.
Ninareh Mehrabi, Fred Morstatter, Nripsuta Saxena, Kristina Lerman, and Aram Galstyan.
A Survey on Bias and Fairness in Machine Learning. ACM Computing Surveys, 54(6):
115:1–115:35, July 2021. ISSN 0360-0300. doi: 10.1145/3457607. URL https://dl.acm.
org/doi/10.1145/3457607.
Andrew C. Miller, Leon A. Gatys, Joseph Futoma, and Emily Fox. Model-based metrics:
Sample-efficient estimates of predictive model subpopulation performance. In Proceedings
of the 6th Machine Learning for Healthcare Conference, pages 308–336, October 2021.
URL https://proceedings.mlr.press/v149/miller21a.html.
20


<!-- page 21 -->
Ziad Obermeyer, Brian Powers, Christine Vogeli, and Sendhil Mullainathan.
Dissecting
racial bias in an algorithm used to manage the health of populations. Science, 366(6464):
447–453, October 2019. ISSN 0036-8075, 1095-9203. doi: 10.1126/science.aax2342. URL
https://www.science.org/doi/10.1126/science.aax2342.
Alvin Rajkomar, Michaela Hardt, Michael D. Howell, Greg Corrado, and Marshall H. Chin.
Ensuring Fairness in Machine Learning to Advance Health Equity. Annals of Internal
Medicine, 169(12):866–872, December 2018. ISSN 1539-3704. doi: 10.7326/M18-1990.
Pranav Rajpurkar, Emma Chen, Oishi Banerjee, and Eric J. Topol.
AI in health and
medicine. Nature Medicine, 28(1):31–38, January 2022. ISSN 1546-170X. doi: 10.1038/
s41591-021-01614-0. URL https://www.nature.com/articles/s41591-021-01614-0.
Pedro Saleiro, Benedict Kuester, Loren Hinkson, Jesse London, Abby Stevens, Ari Anisfeld,
et al. Aequitas: A Bias and Fairness Audit Toolkit, April 2019. URL http://arxiv.org/
abs/1811.05577. arXiv:1811.05577.
Laleh Seyyed-Kalantari, Haoran Zhang, Matthew B. A. McDermott, Irene Y. Chen, and
Marzyeh Ghassemi. Underdiagnosis bias of artificial intelligence algorithms applied to
chest radiographs in under-served patient populations.
Nature Medicine, 27(12):2176–
2182, December 2021. ISSN 1546-170X. doi: 10.1038/s41591-021-01595-0. URL https:
//www.nature.com/articles/s41591-021-01595-0.
Shanshan Song, Yuanyuan Lin, and Yong Zhou. A General M-estimation Theory in Semi-
Supervised Framework. Journal of the American Statistical Association, 0(0):1–11, 2023.
ISSN 0162-1459. doi: 10.1080/01621459.2023.2169699. URL https://doi.org/10.1080/
01621459.2023.2169699.
Tao Tan, Shuyi Zhang, and Yong Zhou. Efficient semiparametric estimation in two-sample
comparison via semisupervised learning. Canadian Journal of Statistics, 53(2):e11813,
2025.
Boris van Breugel, Nabeel Seedat, Fergus Imrie, and Mihaela van der Schaar.
Can
You Rely on Your Model Evaluation?
Improving Model Evaluation with Synthetic
Test Data. Advances in Neural Information Processing Systems, 36:1889–1904, Decem-
ber 2023.
URL https://proceedings.neurips.cc/paper_files/paper/2023/hash/
05fb0f4e645cad23e0ab59d6b9901428-Abstract-Conference.html.
Zichun Xu, Daniela Witten, and Ali Shojaie. A unified framework for semiparametrically
efficient semi-supervised learning. arXiv preprint arXiv:2502.17741, 2025.
21


<!-- page 22 -->
Zikang Xu, Jun Li, Qingsong Yao, Han Li, Mingyue Zhao, and S. Kevin Zhou. Addressing
fairness issues in deep learning-based medical image analysis: a systematic review. npj Dig-
ital Medicine, 7(1):1–16, October 2024. ISSN 2398-6352. doi: 10.1038/s41746-024-01276-5.
URL https://www.nature.com/articles/s41746-024-01276-5.
Sheng Yu, Yumeng Ma, Jessica Gronsbell, Tianrun Cai, Ashwin N. Ananthakrishnan, et al.
Enabling phenotypic big data with PheNorm. Journal of the American Medical Informatics
Association, 25(1):54–60, January 2018. ISSN 1527-974X. doi: 10.1093/jamia/ocx111.
Anru Zhang,
Lawrence D. Brown,
and T. Tony Cai.
Semi-supervised inference:
General theory and estimation of means.
The Annals of Statistics, 47(5):2538–
2566,
October 2019.
ISSN 0090-5364,
2168-8966.
doi:
10.1214/18-AOS1756.
URL
https://projecteuclid.org/journals/annals-of-statistics/volume-47/
issue-5/Semi-supervised-inference-General-theory-and-estimation-of-means/
10.1214/18-AOS1756.full.
Yuqian Zhang, Abhishek Chakrabortty, and Jelena Bradic. Double robust semi-supervised
inference for the mean: selection bias under mar labeling with decaying overlap. Informa-
tion and Inference: A Journal of the IMA, 12(3):2066–2159, 2023.
Supporting Information
The supplementary material contains proofs, additional empirical results and code for re-
producibility.
22


<!-- page 23 -->
S1
Implementation Details of SS Estimators
S1.1
Beta Calibration Method of (Ji et al., 2020)
We implement a frequentist version of the method proposed by Ji et al. (2020). Specifically,
for each group A = a, we apply Beta calibration (BC) (Kull et al., 2017) to recalibrate the
predictions S by fitting the following logistic regression model on the labeled data:
logit{P(Y = 1 | S, A = a)} = ζ0,a + ζ1,a log S + ζ2,a log(1 −S).
Let ˆζ0,a, ˆζ1,a, ˆζ2,a denote the estimated coefficients. The calibrated prediction for group A = a
is SBeta
a
= expit{ˆζ0,a + ˆζ1,a log S + ˆζ2,a log(1 −S)}. We then use SBeta
a
to impute the missing
outcome in group A = a. Again using TPR as an example, the corresponding estimator of
∆TPR is
b∆BC
TPR := [
TPR
BC
0
−[
TPR
BC
1
= N −1
0
PN
i=n+1 DiSBeta
0,i I(Ai = 0)
N −1
0
PN
i=n+1 SBeta
0,i I(Ai = 0)
−N −1
1
PN
i=n+1 DiSBeta
1,i I(Ai = 1)
N −1
1
PN
i=n+1 SBeta
1,i I(Ai = 1)
.
S1.2
Nonparametric Method of Gronsbell and Cai (2018)
We extend a group-specific version of the nonparametric (NP) approach of Gronsbell and
Cai (2018). Specifically, for each group A = a, we estimate the conditional risk function
ma(s) = E(Y | S, A = a) with kernel smoothing as
bma(s) =
Pn
i=1 Kha(Si −s)YiI(Ai = a)
Pn
i=1 Kha(Si −s)I(Ai = a) ,
where Kha(u) = h−1
a K(u/ha) is a given smooth, symmetric kernel function, ha = h(na) is a
bandwidth controlling the level of smoothing, and nh2
a →∞and nh4
a →0 as na →∞. This
under-smoothed bandwidth is needed to ensure consistency and the authors recommend to
use ha = σa/n−0.45
a
, with σa is the standard deviation of S within group A = a.
We use bma(·) to impute the missing outcomes in group A = a. Again using TPR as an
example, the corresponding estimator of ∆TPR is
b∆NP
TPR := [
TPR
NP
0
−[
TPR
NP
1
= N −1
0
Pn+N
i=n+1 Di bm0(Si)I(Ai = 0)
N −1
0
Pn+N
i=n+1 bm0(Si)I(Ai = 0)
−N −1
1
Pn+N
i=n+1 Di bm1(Si)I(Ai = 1)
N −1
1
Pn+N
i=n+1 bm1(Si)I(Ai = 1)
.
1


<!-- page 24 -->
S1.3
Basis Construction for Infairness
We construct estimates of the adjusted and unadjusted conditional risk function using a natu-
ral cubic spline basis expansion of S (Hastie et al., 2009). Let NK(S) = (N1(S), . . . , NK+2(S))⊤
denote a natural cubic spline basis of dimension K + 2 constructed using a set of knots
κ1 < · · · < κK. Such basis functions are piecewise cubic with continuous first and second
derivatives and are constrained to be linear beyond the boundary knots. We consider three
candidate basis constructions for the group-specific imputation model:
B(1)
a (S, W) =
 1, D, NK(S)⊤ ⊤,
B(2)
a (S, W) =
 1, D, NK(S)⊤, W⊤ ⊤,
B(3)
a (S, W) =
 1, D, NK(S)⊤, W⊤, (NK(S) ⊗W)⊤ ⊤,
where NK(S)⊗W denotes all pairwise products between the spline basis functions of S and
the components of W. For r ∈{1, 2, 3}, the corresponding imputation model is
m(r)
a (S, W) = g
n
(bθ(r)
a )⊤B(r)
a (S, W)
o
,
where g(u) = {1 + exp(−u)}−1 is the logistic inverse link. These three candidate models
correspond to increasing levels of flexibility:
• Infairness(S): nonlinear dependence on S only
• Infairness(S + W): nonlinear effect of S with additive effects of W
• Infairness(S × W): additive effects together with interactions between S and W.
Throughout our simulation and real data, we chose K = 3 equally spaced knots. In our
simulation study, we present results for the three versions of Infairness. In the real data, we
select the model with the lowest BIC defined as:
−2ℓ(bθa) + da min{n0.1
a , log(na)},
where ℓ(·) is the logistic log-likelihood, da is the number of regression parameters, and na is
the group-specific labeled sample size.
S1.4
Variance Estimation for Infairness
To improve the finite-sample performance of the influence-function based variance estima-
tion, we use k-fold cross validation. We detail the procedure for TPR below.
2


<!-- page 25 -->
For each group a, we randomly divide the labeled observations with A = a into k approx-
imately equal folds. Let Ik
a denote the indices of the observations in the kth fold and La \Ik
a
denote the indices of the observations not in the kth fold. For fold k, we fit the imputation
model using observations with indices in La \ Ik
a to obtain bθ(−k)
a
. We then calculate the
corresponding Infairness estimator of the group-specific TPR as
 
[
TPR
SS (−k)
a
= N −1
a
Pn+N
i=n+1 Di ˆm(−k)
a,i I(Ai = a)
N −1
a
Pn+N
i=n+1 ˆm(−k)
a,i I(Ai = a)
where ˆm(−k)
a,i
is the imputed value using bθ(−k)
a
. The influence function contribution for the
ith observation is then computed as
(ˆµY
a )−1{Yi −ˆm(−k)
a,i }
 
Di −
 
[
TPR
SS (−k)
a
 
I(i ∈Ik
a).
We apply the same procedure for all metrics as well as all the supervised, BC, and NP
methods.
S2
Asymptotic Analysis
As described in Section 3 of the main text, we make no assumptions on the ML model,
ˆf, other than that it is trained on an independent dataset, denoted as Dtrain. For all the
group-specific performance metrics, the target parameters are defined conditional on Dtrain.
For example, the group-specific TPR is
TPRa = P (D = 1 | Y = 1, A = a)
where the probability is taken over the distribution of (Y, XT, A)T conditioning on Dtrain. To
simplify notation, we omit explicit conditioning on Dtrain, with the understanding that all
target parameters and corresponding inference procedures are defined conditionally on the
training data.
S2.1
Regularity Conditions
To facilitate our discussion of asymptotic properties, we assume the following regularity
conditions hold.
Condition 1. Both µY
a and µS
a are bounded away from 0 and 1.
Condition 2. (A) The basis, Ba, contains S, D, and an intercept, has compact support,
and is of fixed dimension. (B) The density function of Ba is continuously differentiable in
3


<!-- page 26 -->
its continuous components. (C) There is at least one non-zero component of ¯θa and the link
function is smooth.
Condition 3. There is no vector θa such that P(θT
aBa,i > θT
aBa,j | Yi > Yj) = 1 and
E
 
BaBT
a ˙g(¯θT
aBa)
 
is positive definite.
Condition 1 ensures that the target parameters can be appropriately defined. Condition
2 is a standard assumption in Z-estimation theory while condition 3 ensures that there is
no θa that can perfectly separate the data based on Y within each group defined by A = a
(van der Vaart, 1998; Tian et al., 2007).
S2.2
Asymptotic Properties of Supervised Estimators
Let ˆµZ
a = n−1
a
Pn
i=1 ZiI(Ai = a) denote the supervised estimator of µZ
a = E[Z | A = a] for
Z ∈{Y, D, S2, SY, DY }. We first provide a simple Lemma that will be useful in establishing
the asymptotic linear expansion of b∆SUP
M .
Lemma 1. Under the assumption that n−1 Pn
i=1 I(Ai = a) = na/n
p→ρa ∈(0, 1) as n →∞
from Section 3.1,
n
1
2  ˆµZ
a −µZ
a
 
= n−1
2ρ−1
a
n
X
i=1
 
(Zi −µZ
a )I(Ai = a)
	
+ op(1).
Proof. The result is a direct consequence of the assumption that na/n
p→ρa as n →∞.
That is,
n
1
2  ˆµZ
a −µZ
a
 
= n
1
2
 n−1 Pn
i=1(Zi −µZ
a )I(Ai = a)
n−1 Pn
i=1 I(Ai = a)
 
= n−1
2ρ−1
a
n
X
i=1
(Zi −µZ
a )I(Ai = a) + op(1)
where the second equality follows from Slutsky’s Theorem.
S2.2.1
Asymptotic Properties of c
MSUP
a
Lemma 2. Under the assumptions in Sections 3.1 and S2.1, for any metrics Ma defined in
Table 1, c
MSUP
a
p→Ma and
n
1
2
 
c
M
SUP
a
−Ma
 
= n−1
2ρ−1
a
n
X
i=1
IF
SUP
Ma(Yi, Si; c)I(Ai = a) + op(1)
4


<!-- page 27 -->
where IF
SUP
Ma(Yi, Si; c) is given in Table 3. As n →∞, n
1
2
 
c
MSUP
a
−Ma
 
converges weakly to
a zero-mean Gaussian distribution with variance
ρ−1
a E
h 
IF
SUP
Ma(Yi, Si; c)
	2 | A = a
i
.
Proof. The consistency of c
MSUP
a
for Ma follows from the consistency of ˆµZ for µZ for Z ∈
{Y, D, S2, SY, DY }. We detail the derivation of the asymptotic expansion for TPRa. For all
metrics Ma defined in Table 1, the same techniques can be applied. By Lemma 1,
n
1
2(ˆµY
a −µY
a ) = n−1
2ρ−1
a
n
X
i=1
(Yi −µY
a )I(Ai = a) + op(1)
and
n
1
2(ˆµDY
a
−µDY
a
) = n−1
2ρ−1
a
n
X
i=1
(DiYi −µDY
a
)I(Ai = a) + op(1).
Since µY
a > 0, a first-order Taylor expansion yields
[
TPR
SUP
a
= ˆµDY
a
ˆµY
a
= µDY
a
µY
a
+ ρ−1
a n−1
n
X
i=1
h
(µY
a )−1
−(µY
a )−1TPRa
i "
(DiYi −µDY
a
)I(Ai = a)
(Yi −µY
a )I(Ai = a)
#
+ op(n−1
2).
Therefore,
n
1
2
 
[
TPR
SUP
a
−TPRa
 
= n−1
2ρ−1
a
n
X
i=1
(µY
a )−1{Yi(Di −TPRa)}I(Ai = a) + op(1)
= n−1
2ρ−1
a
n
X
i=1
IF
SUP
TPRa(Yi, Si; c)I(Ai = a) + op(1).
By the standard Central Limit Theorem (CLT), n
1
2
 
[
TPR
SUP
a
−TPRa
 
converges weakly to
a zero-mean Gaussian distribution with variance ρ−1
a E
h 
IF
SUP
TPRa(Yi, Si; c)
	2 | Ai = a
i
.
S2.2.2
Asymptotic Properties of ˆ∆SUP
M
We next apply Lemma 2 to find the asymptotic linear expansion of ∆M.
Theorem S1. Under the assumptions in Sections 3.1 and S2.1, ˆ∆SUP
M
p→∆M and
n
1
2
 
ˆ∆
SUP
M −∆M
 
=
X
a∈{0,1}
(−1)aρ−1
a n−1
2
n
X
i=1
IF
SUP
Ma(Yi, Si; c)I(Ai = a) + op(1)
where IF
SUP
Ma(Yi, Si; c) is given in Table 3. Therefore, n
1
2
 
ˆ∆SUP
M −∆M
 
converges weakly to
5


<!-- page 28 -->
a zero-mean Gaussian distribution with variance
X
a∈{0,1}
ρ−1
a E
h 
IF
SUP
Ma(Yi, Si; c)
	2 | A = a
i
.
Proof. By Lemma 2, we have c
MSUP
a
p→Ma and
n
1
2
 
c
M
SUP
a
−Ma
 
= ρ−1
a n−1
2
n
X
i=1
IF
SUP
Ma(Yi, Si; c)I(Ai = a) + op(1).
Thus, ˆ∆SUP
M
p→∆M and c
MSUP
a
is also an asymptotically linear estimator with the following
expansion
n
1
2
 
ˆ∆
SUP
M −∆M
 
=
X
a∈{0,1}
(−1)aρ−1
a n−1
2
n
X
i=1
IF
SUP
Ma(Yi, Si; c)I(Ai = a) + op(1).
Therefore, the standard CLT implies n
1
2
 
c
MSUP
a
−Ma
 
converges weakly to a zero-mean
Gaussian distribution with variance P
a∈{0,1} ρ−1
a E
h 
IF
SUP
Ma(Yi, Si; c)
	2 | A = a
i
.
S2.3
Asymptotic Properties of Semi-supervised Estimators
S2.3.1
Asymptotic Properties of bθa
In the semi-supervised setting, we begin by considering the properties of bθa, which solves the
estimating equation Qna
a (θa; λna) := n−1
a
Pn
i=1 Ba,i {Yi −g (θT
aBa,i)} I(Ai = a) −λnaθa = 0,
where Ba,i = Ba(Si, Wi).
Lemma 3. Let ¯θa be the solution to Qa(θa) = E [Ba {Y −g (θT
aBa)} I(A = a)] = 0. Under
conditions 2 – 3, bθa
p→¯θa. In addition,
n
1
2(bθa −¯θa) = ρ−1
a V−1
a
"
n−1
2
n
X
i=1
Ba,i
 
Yi −g
 ¯θ
T
aBa,i
 	
I(Ai = a)
#
+ op(1),
where Va = E
 
∇g
 ¯θT
aBa
 
BaBT
a | A = a
	
.
Therefore n
1
2(bθa −¯θa) converges weakly to a
zero-mean Gaussian distribution with variance
ρ−1
a V−1
a E
h 
Y −g
 ¯θ
T
aBa
 	2 BaB
T
a | A = a
i
(V−1
a )
T.
Proof. To show that bθa is consistent for ¯θa it suffices to verify that (i) supθa ||Qna
a (θa; λna) −
Qa(θa)||2
p→0 and (ii) ¯θa is the unique solution to Qa(θa) = 0 (Theorem 5.9, van der Vaart,
6


<!-- page 29 -->
1998). For (i), we note that λna = o(n
−1
2
a ) and the class of functions,
{Ba{Y −g (θ
T
aBa)}I(A = a) : θa ∈Θa}
is uniformly bounded and hence a Glivenko-Cantelli class (Theorem 19.4 van der Vaart,
1998). Under conditions 2 – 3 and again using the fact λna = o(n−1/2
a
), the arguments in
Appendix A of Tian et al. (2007) imply that (ii) holds. It then follows that bθa
p→¯θa.
To establish the weak convergence of n
1
2(bθa−¯θa), note that a first-order Taylor expansion
of Qna
a (bθa; λna) about ¯θa together with the fact that λna = o(n
−1
2
a ) implies that
n
1
2(bθa −¯θa) = ρ−1
a V−1
a
"
n−1
2
n
X
i=1
Ba,i
 
Yi −g
 ¯θ
T
aBa,i
 	
I(Ai = a)
#
+ op(1).
The standard CLT implies n
1
2(bθa−¯θa) converges weakly to a zero-mean Gaussian distribution
with variance ρ−1
a V−1
a E
h 
Y −g
 ¯θT
aBa
 	2 BaBT
a | A = a
i
(V−1
a )T.
S2.3.2
Asymptotic Properties of ˜µZ
a
We next derive the asymptotic expansions for the SS estimators of µZ
a , denoted as ˜µZ
a , for
Z ∈{Y, D, SY, DY, S2}.
Lemma 4. Under the assumptions in Sections 3.1 and S2.1, the following results hold:
Case 1. For Z ∈{D, S2}, n
1
2  eµZ
a −µZ
a
 
p→0.
Case 2. For Z ∈{Y, SY, DY },
n
1
2  ˜µZ
a −µZ
a
 
= ρ−1
a
(
n−1
2
n
X
i=1
 Zi −Zi
 
I(Ai = a)
)
+ op(1)
where Zi = g
 ¯θT
aBa,i
  Zi
Yi . Therefore, n
1
2  eµZ
a −µZ
a
 
converges in distribution to a zero-mean
Gaussian distribution with variance: ρ−1
a E
n Z −Z
 2 | A = a
o
.
Proof. First, consider Case 1 in which the SS estimators of µZ do not require imputation.
Using the same arguments as the proof as in Lemma 1, it follows that
N
1
2  ˜µZ
a −µZ
a
 
= ρ−1
a
(
N −1
2
N
X
i=1
(Zi −µZ
a )I(Ai = a)
)
+ op(1)
and hence n
1
2  ˜µZ
a −µZ
a
 
= op(1) given that n/N →0 as n →∞.
Next, consider Case 2 in which the SS estimators of µZ require imputation. We show the
7


<!-- page 30 -->
derivation for Z = Y , which can be applied for Z ∈{SY, DY }. We begin by rewriting
n
1
2  ˜µY
a −µY
a
 
(1)
= n
1
2
"(
N −1
a
n+N
X
i=n+1
g
 
bθ
T
aBa,i
 
I(Ai = a)
)
−µY
a
#
= n
1
2
"
N −1
a
n+N
X
i=n+1
n
g
 
bθ
T
aBa,i
 
−g
 ¯θ
T
aBa,i
 o
I(Ai = a)
#
+
  n
N
  1
2
"
ρ−1
a N −1
2
n+N
X
i=n+1
 
g
 ¯θ
T
aBa,i
 
−µY
a
	
I(Ai = a)
#
+ op(1)
= n
1
2
"
ρ−1
a N −1
n+N
X
i=n+1
n
g
 
bθ
T
aBa,i
 
−g
 ¯θ
T
aBa,i
 o
I(Ai = a)
#
+ op(1)
(2)
The third equality follows from the assumption that limn→∞n/N = 0 and the fact that
ρ−1
a N −1
2
n+N
X
i=n+1
 
g
 ¯θ
T
aBa,i
 
−µY
a
	
I(Ai = a) = Op(1)
due to application of the standard CLT and using the fact that
E
  
Y −g
 ¯θ
T
aBa
 	
I(A = a)
 
= 0
by definition of ¯θa. For the remaining term in (2), the consistency of bθa for ¯θa, together with
a first-order Taylor expansion, implies
n
1
2
"
ρ−1
a N −1
n+N
X
i=n+1
n
g
 
bθ
T
aBa,i
 
−g
 ¯θ
T
aBa,i
 o
I(Ai = a)
#
=
(
ρ−1
a N −1
n+N
X
i=n+1
∇g
 ¯θ
T
aBa,i
 
B
T
a,iI(Ai = a)
) n
n
1
2(bθa −¯θa)
o
+ op(1).
(3)
By Lemma 3,
n
1
2(bθa −¯θa) = ρ−1
a V−1
a
"
n−1
2
n
X
i=1
Ba,i
 
Yi −g
 ¯θ
T
aBa,i
 	
I(Ai = a)
#
+ op(1).
8


<!-- page 31 -->
By the Weak Law of Large Numbers,
ρ−1
a N −1
n+N
X
i=n+1
 
∇g
 ¯θ
T
aBa,i
 
B
T
a,iI(Ai = a)
	
= E
 
∇g
 ¯θ
T
aBa
 
B
T
a | A = a
	
+ op(1).
To further simplify (3), we also note that
E
 
∇g
 ¯θ
T
aBa
 
B
T
a | A = a
	
V−1
a
= arg min
β
E
 
∇g
 ¯θ
T
aBa
 
(1 −β
TBa)2 | A = a
	
Since Ba contains an intercept, the β that minimizes the above expression is a vector with
the first entry equal to 1 and the remaining entries equal to 0. Therefore,
E
 
∇g
 ¯θ
T
aBa
 
B
T
a | A = a
	
V−1
a Ba = 1
and
n
1
2  ˜µY
a −µY
a
 
= ρ−1
a
"
n−1
2
n
X
i=1
 
Yi −g
 ¯θ
T
aBa,i
 	
I(Ai = a)
#
+ op(1).
It then follows by the standard CLT that n
1
2  ˜µY
a −µY
a
 
converges weakly to a zero-mean
Gaussian distribution with variance ρ−1
a E
h 
Y −g
 ¯θT
aBa
 	2 | A = a
i
.
S2.3.3
Asymptotic Properties of c
MSS
a
We use Lemma 4 to derive the asymptotic properties of c
MSS
a in a similar manner to c
MSUP
a
.
Lemma 5. Under the assumptions in Sections 3.1 and S2.1, c
MSS
a
p→Ma and
n
1
2
 
c
M
SS
a −Ma
 
= ρ−1
a n−1
2
n
X
i=1
IF
SS
Ma(Yi, Ba,i; c, ¯θa)I(Ai = a) + op(1)
where Ba,i = Ba(Si, Wi) and IF
SS
Ma(Yi, Ba,i; c, ¯θa) is given in Table 3. Therefore, n
1
2
 
c
MSS
a −Ma
 
converges weakly to a zero-mean Gaussian distribution with variance
ρ−1
a E
h 
IF
SS
Ma
 Yi, Ba,i; c, ¯θa
 	2 | A = a
i
.
Proof. The consistency of c
MSS
a for Ma follows from the consistency of ˜µZ for µZ for Z ∈
{Y, D, S2, SY, DY }. Similar to the supervised estimator, we detail the derivation for TPRa
9


<!-- page 32 -->
and note that all other metrics can be derived with similar arguments. By Lemma 4,
n
1
2(˜µY
a −µY
a ) = ρ−1
a
"
n−1
2
n
X
i=1
 
Yi −g
 ¯θ
T
aBa,i
 	
I(Ai = a)
#
+ op(1)
and
n
1
2(˜µDY
a
−µDY
a
) = ρ−1
a
"
n−1
2
n
X
i=1
Di
 
Yi −g
 ¯θ
T
aBa,i
 	
I(Ai = a)
#
+ op(1).
Similar to the proof of Lemma 2, a first-order Taylor expansion yields
˜µDY
a
˜µY
a
= µDY
a
µY
a
+ ρ−1
a n−1
n
X
i=1
h
(µY
a )−1
−(µY
a )−1TPRa
i "
Di
 
Yi −g
 ¯θT
aBa,i
 	
I(Ai = a)
 
Yi −g
 ¯θT
aBa,i
 	
I(Ai = a)
#
+ op(n−1
2).
Therefore,
n
1
2
 
[
TPR
SS
a −TPRa
 
= ρ−1
a
"
n−1
2
n
X
i=1
(µY
a )−1  
Yi −g
 ¯θ
T
aBa,i
 	
(Di −TPRa)I(Ai = a)
#
+ op(1)
= ρ−1
a
(
n−1
2
n
X
i=1
IF
SS
TPRa
 Yi, Ba,i; c, ¯θa
 
I(Ai = a)
)
+ op(1).
By the standard CLT, n
1
2
 
[
TPR
SS
a −TPRa
 
converges weakly to a zero-mean Gaussian dis-
tribution with variance ρ−1
a E
h 
IF
SS
TPRa
 Yi, Ba,i; c, ¯θa
 	2 | Ai = a
i
.
S2.3.4
Asymptotic Properties of ˆ∆SS
M
We conclude by providing derivations for the results in the main text.
Theorem 1 (Influence function of ˆ∆SS
M). Under the assumptions in Section 3.1 and
regularity conditions in Section S2.1, ˆ∆SS
M
p→∆M and
n
1
2
 
ˆ∆
SS
M −∆M
 
= n−1
2
X
a∈{0,1}
(−1)aρ−1
a
n
X
i=1
IF
SS
Ma
 Yi, Ba,i; c, ¯θa
 
I(Ai = a) + op(1)
where IF
SS
Ma(Yi, Ba,i; c, ¯θa) is specified in Table 3.
Therefore, n
1
2
 
ˆ∆SS
M −∆M
 
converges
weakly to a zero-mean Gaussian distribution with variance
X
a∈{0,1}
ρ−1
a E
h 
IF
SS
Ma
 Yi, Ba,i; c, ¯θa
 	2 | A = a
i
.
Proof. The consistency of ˆ∆SS
M for ∆M follows from Lemma 5. The asymptotic expansion
10


<!-- page 33 -->
for ˆ∆SS
M similarly follows from Lemma 5 and is given by,
n
1
2
 
ˆ∆
SS
M −∆M
 
=
X
a∈{0,1}
(−1)aρ−1
a
(
n−1
2
n
X
i=1
IF
SS
Ma
 Yi, Ba,i; c, ¯θa
 
I(Ai = a)
)
+ op(1).
Using same argument as in the proof of Theorem S1, n
1
2
 
ˆ∆SS
M −∆M
 
converges weakly to
a zero-mean Gaussian distribution with variance
X
a∈{0,1}
ρ−1
a E
h 
IF
SS
Ma
 Yi, Ba,i; c, ¯θa
 	2 | A = a
i
.
S2.4
Asymptotic Variance Analysis of ˆ∆
SUP
M and ˆ∆
SS
M
Theorems S1 and 1 establish that both estimators ˆ∆SUP
M and ˆ∆SS
M are asymptotically linear.
However, in the influence function IF
SS
Ma
 Yi, Ba,i; c, ¯θa
 
provided in Table 3, Yi is centered
at its imputed value g(¯θT
aBa,i).
This difference in centering is responsible for the lower
asymptotic variance of ˆ∆SS
M relative to ˆ∆SUP
M . We establish this result more formally in the
following corollary.
Corollary 1 (Variance comparison of ˆ∆SUP
M
and ˆ∆SS
M). When the imputation model in
equation (2) in the main text is correctly specified and E(Y | S, W, A = a) ̸= E(Y | A = a),
X
a∈{0,1}
ρ−1
a E
h 
IF
SS
Ma(Yi, Ba,i; c, ¯θa)
	2 | A = a
i
<
X
a∈{0,1}
ρ−1
a E
h 
IF
SUP
Ma(Yi, Si; c)
	2 | A = a
i
.
Proof. We again verify the result for ∆TPR and note that other metrics can be derived
similarly. When the imputation model (2) is correct,
IF
SS
TPRa
 Yi, Ba,i; c, ¯θa
 
= (µY
a )−1{Yi −E(Yi | Si, Wi, Ai = a)} (Di −TPRa)
and
IF
SUP
TPRa (Yi, Si; c) = (µY
a )−1{Yi(Di −TPRa)}.
11


<!-- page 34 -->
Thus,
E
h 
IF
SS
TPRa
 Yi, Ba,i; c, ¯θa
 	2 | A = a
i
=E
 
(µY
a )−2{Yi −E(Yi | Si, Wi, Ai = a)}2 (Di −TPRa)2 | A = a
 
=E
h 
IF
SUP
TPRa (Yi, Si; c)
	2 | A = a
i
−E
 
(µY
a )−2{E(Yi | Si, Wi, Ai = a)}2 (Di −TPRa)2 | A = a
 
<E
h 
IF
SUP
TPRa (Yi, Si; c)
	2 | A = a
i
.
The last inequality is strict due to regularity condition 1. It follows that
X
a∈{0,1}
ρ−1
a E
h 
IF
SS
TPRa
 Yi, Ba,i; c, ¯θa
 	2 | A = a
i
<
X
a∈{0,1}
ρ−1
a E
h 
IF
SUP
TPRa (Yi, Si; c)
	2 | A = a
i
.
We note the assumption that E(Y | S, W, A = a) ̸= E(Y | A = a) is only needed to establish
this result for metrics based on the NPV and PPV.
S3
Simulation Studies
S3.1
Stylized Setting Data Generation
For the stylized setting, we generate a binary protected attribute
A ∼Bernoulli(0.6),
and a binary auxiliary covariate
W ∼Bernoulli(0.5).
The outcome is generated independently as
Y ∼Bernoulli(0.3).
Conditional on (Y, A, W), we generate the prediction directly from a beta distribution as
S | Y = y, A = a, W = w ∼Beta{αawy, βawy}.
The final binary prediction of Y is D = I(S ≥0.5). We set N = 10, 0000 and n = 400. In
the first scenario, S ⊥W | Y = y, A = a. In the second scenario, S ̸⊥W | Y = y, A = a.
12


<!-- page 35 -->
Scenario 1
The parameters of the group-specific beta distributions are summarized in 4.
A
W
Y
αawy
βawy
0, 1
0, 1
1
7
5
0, 1
0, 1
0
5
7
(4)
Under this data-generating mechanism, the assumptions of the beta calibration (BC) method
are satisfied as
logit{E(Y | S, A, W)} = log(3/7) + 2 log S −2 log(1 −S).
Scenario 2
The distribution of S within group A = 1 remains the same as before, while
the the distribution of S within group A = 0 depends on W. The parameters of the group-
specific beta distributions are summarized in 5.
A
W
Y
αawy
βawy
0
0
1
14.0
4.0
0
1
1
4.5
10.0
0
0
0
2.5
10.0
0
1
0
7.0
5.0
1
0, 1
1
7
5
1
0, 1
0
5
7
(5)
For group A = 0, marginalizing over W induces a relationship between Y and S that is not
captured by standard BC:
logit{E(Y | S, W, A = 0)} = log(3/7) + 11.5 log S −6 log(1 −S)
−2.6W −14W log S + 11W log(1 −S).
S3.2
Traditional Setting Data Generation
The outcome Y was generated under four different scenarios to vary the quality of the fit of
the ML model. In scenarios A-C, the intercepts α0 and α1 are chosen such P(Y = 1) ≈0.3.
13


<!-- page 36 -->
S3.2.1
Scenario A: Logistic model with nonlinear and interaction terms
The outcome Y is generated according to a logistic model with group-specific coefficients
such that
P(Y = 1 | X, W, A = 0) = expit
 α0 + 1.9X1 + 1.9X2 + 0.9X3 + 0.9X4 + 1.6(W1 + W2 + W3)
+ 0.4X2
2 −0.5X3
3 + 0.6X5X6 + 0.1X2X5X6
 
,
P(Y = 1 | X, W, A = 1) = expit
 α1 + 1.7X1 + 1.7X2 + 0.7X3 + 0.7X4 + 1.2(W1 + W2 + W3)
+ 0.4X2
2 −0.5X3
3 + 0.8X5X6 −0.1X3X5X6
 
,
and expit(x) = (1 + e−x)−1.
S3.2.2
Scenario B: Non-logistic activation function with non-linear hidden layer
The outcome Y is generated using a non-logistic activation function applied to the group-
specific non-linear predictors such that
P(Y = 1 | X, W, A = 0) = exp
n
−
 α0 + 0.40X1 −0.30X2 + 0.15X3 −0.15X4
+ 0.25W1 −0.20W2 + 0.20W3
 2o
, and
P(Y = 1 | X, W, A = 1) = exp
n
−
 α1 + 0.35X1 −0.25X2 + 0.20X3 −0.20X4
+ 0.15W1 −0.15W2 + 0.20W3
 2o
.
S3.2.3
Scenario C: Complementary log-log
The outcome Y is generated using a complementary log-log model with group-specific non-
linear transformations applied to the linear predictor that introduce local non-monotonicity
into the resulting risk score. We first generate the overall linear predictor as
η = −0.5 + X1 + 0.8X2 + 0.6X3 + 0.5X4 + 0.3X5
−0.1X5X2 + 0.1X2
3 −0.03X3
3
+ 0.9W1 + 0.6W2 −0.6W3
+ 0.3X1W1 −0.3X2W2 + 0.2X3W3 + 0.2X4W1 −0.2X5W2.
14


<!-- page 37 -->
We then standardize η as z = η−E(η)
sd(η) and obtain the group-specific event probabilities as
P(Y = 1 | X, W, A = 0) = 1 −exp
h
−exp
n
α0 + 3.2z + 7.5 exp
 
−(z −0.55)2
2(0.18)2
 oi
, and
P(Y = 1 | X, W, A = 1) = 1 −exp
h
−exp
n
α1 + 3.2z + 7.5 exp
 
−(z + 0.30)2
2(0.18)2
 oi
.
S3.2.4
Scenario D: Tree Structure
In the tree-based setting, the event probability is a piecewise-constant function of two linear
predictors based on X and W. Define
ηX = X1 + X2 −X3 + 0.35X4 −0.25X5 + 0.2X6 + 0.15X1X2 −0.1X3X4,
and
ηW = 0.85W3 + 0.75W4 −0.75W5 + 0.15W3W4.
The risk score P(Y = 1 | X, W, A) is specified as a piecewise-constant function of ηX and
ηW, with values given in Table S1.
The likelihood of the outcome increases across the
ηW intervals within each ηX interval, but varies non-monotonically across ηX intervals and
differently within each protected group.
Table S1: P(Y = 1 | X, W, A) in the tree-based setting.
A = 0
A = 1
ηX interval
ηW ≤−0.8
−0.8 < ηW ≤1.1
ηW > 1.1
ηW ≤−1.2
−1.2 < ηW ≤0.8
ηW > 0.8
ηX ≤−3.0
0.50
0.74
0.94
0.03
0.23
0.47
−3.0 < ηX ≤−1.7
0.03
0.20
0.44
0.40
0.64
0.88
−1.7 < ηX ≤−0.7
0.46
0.70
0.94
0.03
0.20
0.44
−0.7 < ηX ≤0.4
0.03
0.24
0.48
0.42
0.66
0.90
0.4 < ηX ≤1.5
0.10
0.34
0.58
0.50
0.74
0.94
1.5 < ηX ≤3.2
0.46
0.70
0.94
0.03
0.23
0.47
ηX > 3.2
0.18
0.42
0.66
0.54
0.78
0.94
15


**[Table p37.1]**
| A = 0 η interval η ≤−0.8 −0.8 < η ≤1.1 η > 1.1 X W W W | A = 1 |
| --- | --- |
|  | η ≤−1.2 −1.2 < η ≤0.8 η > 0.8 W W W |
| η ≤−3.0 0.50 0.74 0.94 X −3.0 < η ≤−1.7 0.03 0.20 0.44 X −1.7 < η ≤−0.7 0.46 0.70 0.94 X −0.7 < η ≤0.4 0.03 0.24 0.48 X 0.4 < η ≤1.5 0.10 0.34 0.58 X 1.5 < η ≤3.2 0.46 0.70 0.94 X η > 3.2 0.18 0.42 0.66 X | 0.03 0.23 0.47 0.40 0.64 0.88 0.03 0.20 0.44 0.42 0.66 0.90 0.50 0.74 0.94 0.03 0.23 0.47 0.54 0.78 0.94 |


<!-- page 38 -->
S3.3
Stylized Setting with n = 1000
Scenario 1
Scenario 2
(a) Point estimate
−0.10
−0.05
0.00
0.05
0.10
Difference in true positive rate
−0.3
−0.2
−0.1
Difference in true positive rate
(b) Coverage probability
0.80
0.85
0.90
0.95
1.00
Coverage probability
0.4
0.6
0.8
1.0
Coverage probability
(c) Relative efficiency
0.0
0.5
1.0
1.5
2.0
Supervised
BC
NP
Infairness
(S)
Infairness
(S + W)
Infairness
(S x W)
Relative efficiency
0
1
2
Supervised
BC
NP
Infairness
(S)
Infairness
(S + W)
Infairness
(S x W)
Relative efficiency
Figure S1: Stylized setting for estimating the difference in true positive rates.
Results are based on 104 Monte Carlo replicates with 20,000 validation observations and 1000
labeled observations per replicate. Panel (a) shows the average point estimate of ∆TPR =
TPR0 −TPR1, with vertical lines corresponding to the average estimate plus or minus 1.96
times the empirical standard error; the dashed horizontal line marks the truth. Panel (b)
shows empirical coverage probability of nominal 95% confidence intervals, with the shaded
region indicating coverage between 0.90 and 1.00. Panel (c) shows relative efficiency, defined
as the ratio of the supervised mean squared error to the mean squared error of each method.
16


**[Table p38.1]**
| Scenario 1 Scenario 2 (a) Point estimate |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.10 rate 0.05 positive 0.00 true in Difference −0.05 −0.10 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | rate −0.1 positive true −0.2 in Difference −0.3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| (b) Coverage probability |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 1.00 probability 0.95 0.90 Coverage 0.85 0.80 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1.0 probability 0.8 0.6 Coverage 0.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| (c) Relative efficiency |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 2.0 efficiency 1.5 1.0 Relative 0.5 0.0 Supervised BC NP Infairness Infairness Infairness (S) (S + W) (S x W) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 2 efficiency 1 Relative 0 Supervised BC NP Infairness Infairness Infairness (S) (S + W) (S x W) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |


<!-- page 39 -->
S3.4
Traditional Setting with n = 1000
−0.1
0.0
0.1
Estimate
Scenario A
0.7
0.8
0.9
1.0
CP
0
1
2
∆ACC
∆BS
∆F1
∆FPR
∆NPV
∆PPV
∆TPR
Relative efficiency
−0.1
0.0
0.1
Estimate
Scenario B
0.7
0.8
0.9
1.0
CP
0.0
0.5
1.0
1.5
2.0
2.5
∆ACC
∆BS
∆F1
∆FPR
∆NPV
∆PPV
∆TPR
Relative efficiency
−0.2
0.0
0.2
0.4
Estimate
Scenario C
0.7
0.8
0.9
1.0
CP
0.0
0.5
1.0
1.5
2.0
2.5
∆ACC
∆BS
∆F1
∆FPR
∆NPV
∆PPV
∆TPR
Relative efficiency
−0.3
−0.2
−0.1
0.0
0.1
0.2
Estimate
Scenario D
0.7
0.8
0.9
1.0
CP
0.0
0.5
1.0
1.5
2.0
2.5
∆ACC
∆BS
∆F1
∆FPR
∆NPV
∆PPV
∆TPR
Relative efficiency
Supervised
BC
NP
Infairness(S)
Infairness(S + W)
Infairness(S x W)
Figure S2: Group fairness auditing across simulation scenarios. Results are based
on 104 Monte Carlo replicates with 1,000 labeled and 20,000 unlabeled observations per
replicate. Each panel corresponds to one simulation scenario. For each scenario, the three
rows report point estimates, empirical coverage probability (CP), and relative efficiency
(RE). In the point estimate row, vertical bars indicate approximate 95% intervals based on
empirical standard errors and the black horizontal bar is the true values. In the CP row, the
dashed line marks nominal 95% coverage and the shaded region indicates coverage between
0.90 and 1.00.
17


**[Table p39.1]**
| Scenario A 0.1 Estimate 0.0 −0.1 | Scenario B 0.1 Estimate 0.0 −0.1 |
| --- | --- |
|  |  |
| 1.0 0.9 CP 0.8 0.7 | 1.0 0.9 CP 0.8 0.7 |
| y | y |
| efficienc 2 1 Relative 0 ∆ ∆ ∆F1 ∆FPR ∆NPV ∆PPV ∆TPR ACC BS | 2.5 efficienc 2.0 1.5 1.0 Relative 0.5 0.0 ∆ ∆ ∆F1 ∆FPR ∆NPV ∆PPV ∆TPR ACC BS |
|  |  |
| Scenario C 0.4 Estimate 0.2 0.0 −0.2 | Scenario D 0.2 0.1 Estimate 0.0 −0.1 −0.2 −0.3 |
|  |  |
| 1.0 0.9 CP 0.8 0.7 | 1.0 0.9 CP 0.8 0.7 |
| y | y |
| 2.5 efficienc 2.0 1.5 1.0 Relative 0.5 0.0 ∆ ∆ ∆F1 ∆FPR ∆NPV ∆PPV ∆TPR ACC BS Supervised BC NP Infairn | 2.5 efficienc 2.0 1.5 1.0 Relative 0.5 0.0 ∆ ∆ ∆F1 ∆FPR ∆NPV ∆PPV ∆TPR ACC BS ess(S) Infairness(S + W) Infairness(S x W) |


<!-- page 40 -->
S3.5
Small Sample Size n = 100
Group 0
0.00
0.25
0.50
0.75
Estimate
0.80
0.85
0.90
0.95
1.00
CP
0
1
2
ACC
BS
F1
FPR
NPV
PPV
TPR
RE
Group 1
0.00
0.25
0.50
0.75
1.00
Estimate
0.80
0.85
0.90
0.95
1.00
CP
0.0
0.5
1.0
1.5
2.0
2.5
ACC
BS
F1
FPR
NPV
PPV
TPR
RE
Delta
−0.25
0.00
0.25
Delta
0.80
0.85
0.90
0.95
1.00
CP
0
1
2
∆ACC
∆BS
∆F1
∆FPR
∆NPV
∆PPV
∆TPR
RE
Supervised
BC
NP
Infairness(S)
Infairness(S + W)
Infairness(S x W)
Figure S3: Group fairness auditing in Scenario A with n = 100. Results are based on
104 Monte Carlo replicates with 100 labeled and 20,000 unlabeled observations per replicate.
Each panel corresponds to group-specific results or their differences. For each panel, the
three rows report point estimates, empirical coverage probability (CP), and relative efficiency
(RE) for group differences in the performance metrics. In the point estimate row, vertical
bars indicate approximate 95% intervals based on empirical standard errors and the black
horizontal bar marks the truth. In the CP row, the dashed line marks nominal 95% coverage
and the shaded region indicates coverage between 0.90 and 1.00.
18


**[Table p40.1]**
| Group 0 Group 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.75 Estimate 0.50 0.25 0.00 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1.00 0.75 Estimate 0.50 0.25 0.00 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 1.00 0.95 0.90 CP 0.85 0.80 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1.00 0.95 0.90 CP 0.85 0.80 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 2.5 2.0 2 1.5 RE RE 1 1.0 0.5 0 0.0 ACC BS F1 FPR NPV PPV TPR |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | ACC BS F1 FPR NPV PPV TPR |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  | Delta |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  | 0.25 0.00 Delta −0.25 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  | 1.00 0.95 0.90 CP 0.85 0.80 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  | 2 RE 1 0 ∆ACC ∆BS ∆F1 ∆FPR ∆NPV ∆PPV ∆TPR Supervised BC NP Infairness(S) Infairness(S + W) Infairness(S x W) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |


<!-- page 41 -->
S3.6
Uninformative prediction
As an extreme setting, we randomly generate the prediction from a Uniform[0,1] distribution.
Results are presented in Figure S4. Our proposed method remains unbiased and has nominal
coverage.
With respective to efficiency, Infairness does not provide meaningful gains in
∆PPV, ∆NPV and ∆F1, but does not lose efficiency either. Infairness can still gain in the
remaining metrics, which is briefly detailed below for ∆TPR and ∆PPV.
When S ⊥⊥Y , our imputation model ˆma
p→µY
a = E(Y |A = a). The IF for the supervised
TPR estimator is (µY
a )−1{Yi(Di −TPRa)}, while the IF for Infairness estimator is centered
at (µY
a )−1{Yi −E(Y |A = a)}(Di −TPRa). This leads to variance reduction in estimating
TPRa as
aVar([
TPR
sup
a ) = (µY
a )−2E
 
Y 2
i (Di −TPRa)2 | Ai = a
 
= (µY
a )−2E(Yi | Ai = a)E
 
(Di −TPRa)2 | Ai = a
 
= 1
µY
a
TPRa(1 −TPRa)
while
aVar([
TPR
SS
a ) = (µY
a )−2E
 
(Yi −µY
a )2 | Ai = a
 
E
 
(Di −TPRa)2 | Ai = a
 
= 1 −µY
a
µY
a
TPRa(1 −TPRa).
For PPV, however, when S ⊥⊥Y , we have
PPVa = E(Yi | Di = 1, Ai = a) = E(Yi | Ai = a) = µY
a .
Thus the IF for the supervised, (µD
a )−1Di(Yi−PPVa) and the IF for the Infairness, (µD
a )−1Di(Yi−
µY
a ) are identical.
19


<!-- page 42 -->
−0.2
−0.1
0.0
0.1
0.2
Estimate
a) Point estimate
0.80
0.85
0.90
0.95
1.00
Coverage probability
b) Coverage probability
0
1
2
3
∆ACC
∆BS
∆F1
∆FPR
∆NPV
∆PPV
∆TPR
Relative efficiency
c) Relative efficiency
Supervised
Infairness(S)
Figure S4: Group fairness auditing with predictions generated from Uniform[0,1].
Results are based on 104 Monte Carlo replicates with 400 labeled and 20,000 unlabeled
observations per replicate. The three rows report point estimates, empirical coverage proba-
bility (CP), and relative efficiency (RE) for group differences in the performance metrics. In
the point estimate row, vertical bars indicate approximate 95% intervals based on empirical
standard errors and the black horizontal bar marks the true values. In the CP row, the
dashed line marks nominal 95% coverage and the shaded region indicates coverage between
0.90 and 1.00. In the RE row, RE is defined as the ratio of the supervised mean squared
error (MSE) to the MSE of each method.
20


<!-- page 43 -->
S4
Real Data Analysis
S4.1
EHR-based phenotyping
We construct the prediction using SuperLearner, an ensemble method that combines multiple
ML algorithms, with a silver-standard depression label using a random sample of 3,000 un-
labeled observations. Specifically, the SuperLearner library includes regularized generalized
linear models (SL.glmnet), Bayesian generalized linear models (SL.bayesglm), random forests
(SL.ranger), gradient boosting machines (SL.xgboost), and a mean model as a baseline.
The
silver-standard label deemed patients as having depression if (i) two depression-related Con-
cept Unique Identifiers (CUIs) were present in the patient’s discharge summary (C4049644
and C0011581) or (ii) there were depression-related International Classification of Diseases,
Ninth Revision (ICD-9) codes (296.20 – 296.26, 296.30 – 296.36) in the patient’s record.
The features, X, included 22 other relevant CUIs extracted from discharge summaries using
cTAKES software and 5 structured EHR features, including counts of evaluations, medicine
notes, surgery notes, ICU stays, and prescriptions. The final classification for depression was
obtained by thresholding the predictions at a cut-off that achieves overall FPR around 10%.
S4.2
Chest X-ray audit
We study classification of cardiomegaly from the CheXpert dataset, which contains 224,316
chest radiographs from 65,240 patients (Irvin et al., 2019). We randomly selected one image
per patient for fairness auditing. The labeled dataset consists of n = 500 radiologist-labeled
images while the remaining N = 64,740 images are unlabeled. The prediction S is obtained
from a publicly available pre-trained DenseNet-121 model with weights from TorchXRayVi-
sion (Cohen et al., 2022). We evaluate fairness across sex (male vs female; 55.5% vs 44.5%)
and also have available an auxiliary covariate, age.
Figure S5 shows results for the audit across sex.
Similar to the prior example, the
supervised, NP, and Infairness estimators yield relatively similar point estimates. The 95%
confidence intervals for ∆FPR suggest that predictive equality is satisfied while Infairness
suggests a potential violation of equal opportunity. The RE of Infairness is 2.56 for ∆TPR
and 1.97 for ∆FPR, which corresponds to variance reductions of 60.9% and 56.3%.
The
efficiency gains of Infairness relative to NP are less substantial than the EHR phenotyping
example, likely due to the inclusion of a single auxiliary covariate. Prior work demonstrates
that chest X-ray classifiers exhibit variable performance across socioeconomic status and race
(Seyyed-Kalantari et al., 2021). These variables could yield higher precision of Infairness,
but were unavailable in the version of our dataset.
21


<!-- page 44 -->
−0.2
−0.1
0.0
0.1
Estimated Value
0.0
0.5
1.0
1.5
2.0
∆FPR
∆TPR
Relative Efficiency
Supervised
NP
Infairness
Figure S5: FPR and TPR disparity estimates Chest X-ray audit across sex. Dis-
parities are defined as the difference in group-specific performance between the female and
male.
22


<!-- page 45 -->
References
Joseph Paul Cohen, Joseph D. Viviano, Paul Bertin, Paul Morrison, Parsa Torabian,
Matteo Guarrera, Matthew P Lungren, Akshay Chaudhari, Rupert Brooks, Moham-
mad Hashir, and Hadrien Bertrand. Torchxrayvision: A library of chest x-ray datasets
and models.
In Ender Konukoglu, Bjoern Menze, Archana Venkataraman, Christian
Baumgartner, Qi Dou, and Shadi Albarqouni, editors, Proceedings of The 5th Inter-
national Conference on Medical Imaging with Deep Learning, volume 172 of Proceed-
ings of Machine Learning Research, pages 231–249. PMLR, 06–08 Jul 2022.
URL
https://proceedings.mlr.press/v172/cohen22a.html.
Jessica L. Gronsbell and Tianxi Cai. Semi-supervised approaches to efficient evaluation of
model prediction performance. Journal of the Royal Statistical Society. Series B (Statistical
Methodology), 80(3):579–594, 2018.
ISSN 1369-7412.
URL https://www.jstor.org/
stable/26773169.
Trevor Hastie, Robert Tibshirani, and Jerome Friedman. Basis Expansions and Regular-
ization. In Trevor Hastie, Robert Tibshirani, and Jerome Friedman, editors, The Ele-
ments of Statistical Learning: Data Mining, Inference, and Prediction, Springer Series in
Statistics, pages 139–189. Springer, New York, NY, 2009. ISBN 978-0-387-84858-7. doi:
10.1007/978-0-387-84858-7 5. URL https://doi.org/10.1007/978-0-387-84858-7_5.
Jeremy Irvin, Pranav Rajpurkar, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute,
Henrik Marklund, Behzad Haghgoo, Robyn Ball, Katie Shpanskaya, Jayne Seekins,
David A. Mong, Safwan S. Halabi, Jesse K. Sandberg, Ricky Jones, David B. Larson,
Curtis P. Langlotz, Bhavik N. Patel, Matthew P. Lungren, and Andrew Y. Ng. Chex-
pert: a large chest radiograph dataset with uncertainty labels and expert comparison. In
Proceedings of the Thirty-Third AAAI Conference on Artificial Intelligence and Thirty-
First Innovative Applications of Artificial Intelligence Conference and Ninth AAAI Sym-
posium on Educational Advances in Artificial Intelligence, AAAI’19/IAAI’19/EAAI’19.
AAAI Press, 2019.
ISBN 978-1-57735-809-1.
doi: 10.1609/aaai.v33i01.3301590.
URL
https://doi.org/10.1609/aaai.v33i01.3301590.
Disi Ji, Padhraic Smyth, and Mark Steyvers. Can I trust my fairness metric? assessing fair-
ness with unlabeled data and Bayesian inference. In Proceedings of the 34th International
Conference on Neural Information Processing Systems, NIPS’20, pages 18600–18612, De-
cember 2020. ISBN 978-1-71382-954-6.
Meelis Kull, Telmo Silva Filho, and Peter Flach. Beta calibration: a well-founded and easily
23


<!-- page 46 -->
implemented improvement on logistic calibration for binary classifiers. In Proceedings of
the 20th International Conference on Artificial Intelligence and Statistics, pages 623–631.
PMLR, April 2017. URL https://proceedings.mlr.press/v54/kull17a.html. ISSN:
2640-3498.
Laleh Seyyed-Kalantari, Haoran Zhang, Matthew BA McDermott, Irene Y Chen, and
Marzyeh Ghassemi. Underdiagnosis bias of artificial intelligence algorithms applied to
chest radiographs in under-served patient populations.
Nature medicine, 27(12):2176–
2182, 2021.
L. Tian, T. Cai, E. Goetghebeur, and L. J. Wei. Model evaluation based on the sampling
distribution of estimated absolute prediction error.
Biometrika, 94(2):297–311, Febru-
ary 2007.
ISSN 0006-3444, 1464-3510.
doi:
10.1093/biomet/asm036.
URL https:
//academic.oup.com/biomet/article-lookup/doi/10.1093/biomet/asm036.
A. W. van der Vaart. Asymptotic Statistics. Cambridge Series in Statistical and Probabilis-
tic Mathematics. Cambridge University Press, Cambridge, 1998. ISBN 978-0-521-78450-
4. doi: 10.1017/CBO9780511802256. URL https://www.cambridge.org/core/books/
asymptotic-statistics/A3C7DAD3F7E66A1FA60E9C8FE132EE1D.
24