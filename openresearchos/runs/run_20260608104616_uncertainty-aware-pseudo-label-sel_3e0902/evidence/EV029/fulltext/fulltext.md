<!-- page 1 -->
Learning Label Refinement and Threshold Adjustment for
Imbalanced Semi-Supervised Learning
Zeju Li ∗
zeju.li@ndcn.ox.ac.uk
FMRIB Centre, Wellcome Centre for Integrative Neuroimaging, University of Oxford, Oxford, UK
Ying-Qiu Zheng
ying-qiu.zheng@ndcn.ox.ac.uk
FMRIB Centre, Wellcome Centre for Integrative Neuroimaging, University of Oxford, Oxford, UK
Chen Chen
chen.chen2@sheffield.ac.uk
Department of Computer Science, University of Sheffield, Sheffield, UK
Department of Engineering Science, University of Oxford, Oxford, UK
Department of Computing, Imperial College London, London, UK
Saad Jbabdi
saad.jbabdi@ndcn.ox.ac.uk
FMRIB Centre, Wellcome Centre for Integrative Neuroimaging, University of Oxford, Oxford, UK
Abstract
Semi-supervised learning (SSL) algorithms struggle to perform well when exposed to im-
balanced training data. In this scenario, the generated pseudo-labels can exhibit a bias
towards the majority class, and models that employ these pseudo-labels can further am-
plify this bias. Here we investigate pseudo-labeling strategies for imbalanced SSL including
pseudo-label refinement and threshold adjustment, through the lens of statistical analy-
sis.
We find that existing SSL algorithms which generate pseudo-labels using heuristic
strategies or uncalibrated model confidence are unreliable when imbalanced class distri-
butions bias pseudo-labels. To address this, we introduce SEmi-supervised learning with
pseudo-label optimization based on VALidation data (SEVAL) to enhance the quality of
pseudo-labelling for imbalanced SSL. We propose to learn refinement and thresholding pa-
rameters from a partition of the training dataset in a class-balanced way. SEVAL adapts to
specific tasks with improved pseudo-labels accuracy and ensures pseudo-labels correctness
on a per-class basis. Our experiments show that SEVAL surpasses state-of-the-art SSL
methods, delivering more accurate and effective pseudo-labels in various imbalanced SSL
situations. SEVAL, with its simplicity and flexibility, can enhance various SSL techniques
effectively. The code is publicly available 1.
Keywords:
Imbalanced Semi-Supervised Learning, Pseudo-Labeling, Long-Tailed Learn-
ing
1 Introduction
Semi-supervised learning (SSL) algorithms are trained on datasets that contain both labelled
and unlabelled samples Chapelle et al. (2009). SSL improves representation learning and
refines decision boundaries without relying on large volumes of labelled data, which are
labor-intensive to collect.
Many SSL algorithms have been introduced, with one of the
∗. Corresponding author
1. https://github.com/ZerojumpLine/SEVAL
©2024 Zeju Li and Ying-Qiu Zheng and Chen Chen and Saad Jbabdi.
License: CC-BY 4.0, see https://creativecommons.org/licenses/by/4.0/.
arXiv:2407.05370v2  [cs.LG]  17 Sep 2024


<!-- page 2 -->
SEVAL
most prevalent assumptions being consistency Zhou et al. (2003), which requires the decision
boundaries to lie in low density areas. As a means of accomplishing this, pseudo-labels are
introduced in the context of SSL Scudder (1965), and this concept has been extended to
several variants employing diverse pseudo-label generation strategies Laine and Aila (2016);
Berthelot et al. (2019b,a); Sohn et al. (2020); Wang et al. (2022b). In the pseudo-labelling
framework, models trained with labelled data periodically classify the unlabelled samples,
and samples that are confidently classified are incorporated into the training set.
The performance of pseudo-label based SSL algorithms depends on the quality of the
pseudo-labels Chen et al. (2023). In real-world applications, the performance of these SSL
algorithms often degrades due to the prevalence of class imbalance in real-world datasets Liu
et al. (2019). When trained with imbalanced training data X, the model f will be biased
at inference and tends to predict the majority class Cao et al. (2019); Li et al. (2020).
Consequently, this heightened sensitivity negatively impacts the pseudo-labels produced by
SSL algorithms, leading to an ever increasing bias in models trained with these labels.
Therefore, finding a better way to obtain pseudo-labels for imbalanced SSL algorithms
is an important and popular research topic. But it remains an unsolved problem despite
many efforts in this area. In this paper, we examine the process of pseudo-label generation
from a statistical viewpoint and derive strategies for pseudo-label refinement and threshold
adjustment in a class-imbalanced setup. Surprisingly, we find that although existing heuris-
tic solutions can reduce bias to some extent, they can still be suboptimal. They are either
not theoretically sound, or are impacted by the choice of metrics that are not properly
designed to identify thresholds for high-quality label selection.
Our key insight (detailed in Section 3) is that both pseudo-label refinement and selection
rely on test data distribution.
Hence, we propose to utilize a small fraction of distinct
labelled datasets, as a proxy for unseen test data, to improve the quality of pseudo-labels 2.
Our method is named SEVAL, which is short for SEmi-supervised learning with pseudo-
label optimization based on VALidation data.
At its core, SEVAL refines the decision
boundaries of pseudo-labels using a partition of the training dataset before proceeding with
the standard training process. Similar to AutoML Zoph and Le (2016); Ho et al. (2019),
SEVAL can adapt to specific tasks by learning from the imbalanced data itself, resulting in
a better fit. Moreover, SEVAL learns thresholds that can effectively prioritize the selection
of samples from the high-precision class, which we find is critical but typically overlooked
by current model confidence-based dynamic threshold solutions Zhang et al. (2021); Guo
and Li (2022).
The contributions of this paper are as follow:
• In the context of pseudo-labelling in imbalanced SSL,we derive the theoretically opti-
mal offsets for pseudo-label refinement and propose efficient strategies for thresholding.
• We propose to learn a curriculum of pseudo-label adjustment offsets using a partition
of training data. The derived offsets do not rely on a calibrated model and improve
accuracy in both pseudo-labeling and inference.
2. Note that in practice, we learn the parameters with a partition of the labeled training dataset before the
standard SSL process, thus not requiring any additional data.
2


<!-- page 3 -->
SEVAL
• We propose to learn a curriculum of thresholds to select correctly classified pseudo-
labels using a partition of training data with a novel optimization function.
Our
strategies outperform existing methods by relaxing the trade-off assumption of Pre-
cision and Recall.
• We combine the above mentioned two techniques into a unified learning framework,
SEVAL, and find that it outperforms state-of-the-art pseudo-label based SSL methods
such as DARP, Adsh, FlexMatch and DASO under various imbalanced scenarios.
2 Related Work
2.1
Semi-Supervised learning
Semi-supervised learning has been a longstanding research focus.
The majority of ap-
proaches have been developed under the assumption of consistency, wherein samples with
similar features are expected to exhibit proximity in the label space Chapelle et al. (2009);
Zhou et al. (2003). Compared with graph-based methods Iscen et al. (2019); Kamnitsas
et al. (2018), perturbation-based methods Xie et al. (2020); Miyato et al. (2018) and genera-
tive model-based methods Li et al. (2017); Gong et al. (2023), using pseudo-labels is a more
straightforward and empirically stronger solution for deep neural networks Van Engelen and
Hoos (2020). It periodically learns from the model itself to encourage entropy minimiza-
tion Grandvalet and Bengio (2004). This process helps to position decision boundaries in
low-density areas, resulting in more consistent labeling.
Deep neural networks are particularly suited for pseudo-label-based approaches due to
their strong classification accuracy, enabling them to generate high-quality pseudo-labels Lee
et al. (2013); Van Engelen and Hoos (2020). Several methods have been explored to generate
pseudo-labels with a high level of accuracy Wang et al. (2022b); Xu et al. (2021). For exam-
ple, Mean-Teacher Tarvainen and Valpola (2017) calculates pseudo-labels using the output
of an exponential moving average model along the training iterations; MixMatch Berthelot
et al. (2019b) derives pseudo-labels by averaging the model predictions across various trans-
formed versions of the same sample; FixMatch Sohn et al. (2020) estimates pseudo-labels of
a strongly augmented sample with the model confidence on its weakly augmented version.
Many of these approaches falter when faced with class imbalance in the training data, a
frequent occurrence in real-world datasets.
2.2
Imbalanced Semi-Supervised Learning
Current pseudo-labelling strategies for SSL algorithms face challenges in real-world data
generalization due to class imbalance. Using pseudo-labels in the context of class imbalance
can lead to an ever increasing bias towards the majority class during training as more and
more samples are pseudo-labelled. Three main categories of methods address this challenge
in the literature.
2.2.1
Long-tailed learning-based methods
The first group of methods alters the cost function computed using the labeled samples
to train a balanced classifier, consequently leading to improved pseudo-labels. Long-tailed
3


<!-- page 4 -->
SEVAL
learning presents a complex problem in machine learning, wherein models are trained on
data with a distribution characterized by a long tail.
In such distributions, classes are
imbalanced, with the tail classes consistently being underrepresented.
The research on
long-tailed recognition Chawla et al. (2002); Kang et al. (2019); Menon et al. (2020); Tian
et al. (2020), which focuses on building balanced classifiers through adjusted cost functions
or model structures in a fully supervised learning setting, often serves as inspiration for
works in imbalanced SSL. For example, BiS He et al. (2021) and SimiS Chen et al. (2022)
resample the labelled and pseudo-labelled training datasets to build balanced classifier.
ABC Lee et al. (2021) and Cossl Fan et al. (2022) decouple the feature learning and classifier
learning with a two head model architecture. Similarly, L2AC Wang et al. (2022a) further
decouples the feature and classifier learning by building an explicit bias attractor via bi-
level optimization. SAW reweights unlabelled samples from different classes based on the
learning difficulties Lai et al. (2022b).
2.2.2
Pseudo-label refinement-based methods
The second category of methods modify targets of the produced pseudo-labels to allevi-
ate the class bias caused by an imbalanced classifier.
DARP Kim et al. (2020) refines
pseudo-labels by aligning their distribution with the target distribution. SaR Lai et al.
(2022a) aligns pseudo-labels to true distributions using distribution alignment-based miti-
gation vector. On the other hand, logit adjustment (LA) has been explored in the context
of SSL to address class bias by shifting the logit using the logarithm of the class priors’
ratio, based on the class frequency from the training dataset Wei and Gan (2023).
2.2.3
Threshold adjustment-based methods
The last group of methods aims to resample unlabelled data to achieve a balanced distri-
bution across classes. Adsh Guo and Li (2022) utilizes an adaptive threshold to ensure
that similar portions of pseudo-labels are selected for each class. Built upon FixMatch,
FlexMatch and FreeMatch Zhang et al. (2021); Wang et al. (2022c) choose confidently clas-
sified samples based on the model’s learning progress, which results in the selection of more
samples if the model is not learning well. InPL Yu et al. (2023) proposes to rely on an
energy score instead of maximum class probability to select confident samples. InstanT Li
et al. (2024) derives instance-wise thresholds based on the sample features. UPS Rizve
et al. (2021) utilizes both the confidence and uncertainty of a network prediction to select
accurate subsets of pseudo-labels.
Some hybrid methods simultaneously adjust the cost functions and refine the pseudo-
labels.
For instance, in addition to a bootstrap sampling strategy, CReST+ Wei et al.
(2021) utilizes distribution alignment to adjust the class bias of pseudo-labels. DASO Oh
et al. (2022) improves pseudo-labels with semantic pseudo-labels and regularizes the feature
encoder by aligning balanced semantic prototypes. ACR Wei and Gan (2023) is a holistic
approach that builds upon the successes of ABC, FixMatch and MixMatch, and utilizes LA
to refine pseudo-labels Menon et al. (2020), yielding impressive results.
In contrast to the above, our proposed method, SEVAL, seamlessly integrates into SSL
pipelines without necessitating alterations to the learning strategies, data sampling pro-
cess, or additional pseudo-label calculations.
In addition, unlike many imbalanced SSL
4


<!-- page 5 -->
SEVAL
algorithms such as DARP and CreST+, SEVAL does not make strong assumptions on
the label distribution of unlabelled data. Thus it can be applied to scenarios where the
distributions of labelled and unlabelled data are distinct without any modifications.
3 Limitations of Current Methods
In this section, we begin by summarizing the framework of current pseudo-label based
semi-supervised learning (SSL). Next, we break down the design of current imbalanced SSL
into two key components: pseudo-label refinement and threshold adjustment.
We then
offer insights into these components through theoretical analysis, highlighting previously
overlooked aspects.
3.1
Preliminaries
We consider the problem of C-class imbalanced semi-supervised classification. Let X be
the input space and Y = {1, 2, . . . , C} be the label space. We are given a set of N labelled
samples X = {(xi, yi)}N
i=1 and a set of M unlabelled samples U = {(ui)}M
i=1 in order to
learn an optimal function or model f that maps the input feature space to the label space
f : X →RC
+. f can be viewed as an estimate of the conditional probability distribution
P(Y |X).
In deep learning, f can be implemented as a network followed by a softmax
function σ in order to produce the probability score pc for each class c: pc = σ(z)c =
ezc
PC
j=1 ezj
where z is the raw output (in C-dim) produced by the network. The predicted class is
assigned to the one with the highest probability.
Typically, the parameters in f can be optimized by minimizing an empirical risk com-
puted from the labelled dataset RH,X (f) =
1
N
PN
i=1 H(yi, pX
i ), where pX
i
is the model
predicted probabilities on the labelled data point and H can be implemented as the most
commonly used cross-entropy loss. To further utilize unlabelled data, a common approach
is to apply pseudo-labeling to unlabelled data where the estimated label is generated from
the network’s predicted probability vector q ∈RC Lee et al. (2013). As there is no ground
truth, we use the current model to produce a pseudo-label probability vector qi ∈RC for
an unlabelled sample ui and the pseudo-label ˆyi is determined as arg maxj(qij). In this
way, we obtain ˆU = {(ui, ˆyi)}M
i=1, where each (ui, ˆyi) ∈(X × Y ). As the predicted label
quality can be very poor especially at the early stage and for some challenging data, it is
common to apply a threshold to identify reliable labels for model optimization. We describe
the case of hard pseudo-labels for simplicity, but the method generalizes to the case of soft
pseudo-labels. The risk function on the unlabelled data can be defined as:
ˆRH, ˆU(f) = 1
M
M
X
i=1
1(max
j (qij) ≥τ)H(ˆyi, pU
i ),
(1)
where 1 is the indicator function and τ is a predefined threshold that filters out pseudo-
labels with low confidence, and pU
i is the predicted probability for the unlabelled data.
In a class imbalanced setting, we have a challenge: the distribution of samples across
the C classes is highly uneven with varying numbers of samples per class nC : n1, ..., nc.
Some classes have many samples (majority classes), while others have very few (minority
5


<!-- page 6 -->
SEVAL
classes). The class imbalance ratio is defined as γ = max(nC)
min(nC) , and in an unbalanced setting
can e.g., exceed 10. In such case, the pseudo-labels on the unlabelled data can be biased
to the majority class, which further amplifies the class imbalance problem. During testing,
the model often predicts the majority class instead of the true class.
In this paper, to alleviate the issue of class imbalance-induced bias, we propose a method
to refine the probability vector used for pseudo-labelling qi 3. We also propose to adjust
the threshold to operate on a class-specific basis, i.e. we use a vector τ ∈RC of threshold
values to achieve accuracy fairness. The model can then dynamically select the appropriate
thresholds based on its prediction. In the following section, we will bypass the computation
of pseudo-label probability qi and concentrate on our contributions.
3.2
Pseudo-Label Refinement
DA Berthelot et al.
(2019a)
LA Menon et al.
(2020)
DASO Oh et al.
(2022)
SEVAL (c.f.
Section 4.1)
Estimation of
optimal clas-
sifier fU(X)
f(X)PU(Y )
ˆPU(Y )
f(X)PT (Y )
PX (Y )
Blending similarity
based pseudo-label
f(X)PT (Y )
π∗
Note
ˆPU(Y ) is the model
prediction of PU(Y ). The
refined pseudo-label is not
optimal on PT (X, Y ).
Inaccurate as f is
suboptimal and
uncalibrated.
Relying on the
effectiveness of
blending strategies.
Optimizing the decision
boundary on U using V
as a proxy without
assuming a specific f.
Table 1: Theoretical comparisons of SEVAL and other pseudo-label refinement methods
including distribution alignment (DA) Berthelot et al. (2019a); Wei et al. (2021);
Lai et al. (2022a); Kim et al. (2020), logit adjustment (LA) Wei and Gan (2023);
Menon et al. (2020) and DASO Oh et al. (2022). X: Labelled training data; U:
Unlabelled training data; T : Test data; V: An independent labelled data.
For an unlabelled sample ui, we determine its pseudo-label probability qi based on
its corresponding pseudo-label logit ˆzU
i .
In the process of pseudo-label refinement, we
aim to adjust the decision boundaries for ˆzU
i
with offset π ∈RC to reduce class biases.
Many methods in the literature have been discussed to utilize different π from different
perspectives, such as distribution alignment.
However, we find that none of them are
suitable for imbalanced SSL. Here, we aim to shed new light on this problem by analyzing
optimal threshold adjustment strategies from a statistical perspective Saerens et al. (2002).
We assume that the test distribution T shares identical class conditionals with the training
dataset X (i.e., PX (X|Y ) = PT (X|Y )) and deviates solely in terms of class priors (PX (Y ) ̸=
PT (Y )), we can assert:
3. We use different notations q and pU to denote the probability obtained for pseudo-labelling and model
optimization. This is common in existing SSL algorithms, where the two can be obtained in different ways
for better model generalization Laine and Aila (2016); Sohn et al. (2020); Berthelot et al. (2019b,a). For
example, in FixMatch Sohn et al. (2020), qi = f(Aw(ui)) is estimated on a weakly augmented sample
of an input image for reliable supervision whereas pU
i
is estimated using a strongly-augmented (i.e.
RandAugment Cubuk et al. (2020)) version As(ui) as model input for the same instance i.
6

[CAPTION] Table 1: Theoretical comparisons of SEVAL and other pseudo-label refinement methods


<!-- page 7 -->
SEVAL
Proposition 1 Given that a classifier f∗(X) is optimized on PX (X, Y ),
fT (X) ∝f∗(X)PT (Y )
PX (Y )
,
(2)
is the optimal Bayes classifier on PT (X, Y ), where PX (X|Y ) = PT (X|Y ) and PX (Y ) ̸=
PT (Y ).
Corollary 2 The classifier fU(X) = fT (X) should be also optimal on the resampled unla-
belled data PU(X, Y )PT (Y )
PU(Y )
, where PT (X|Y ) = PU(X|Y ) and PT (Y ) ̸= PU(Y ).
This proposition provides insight into the formulation of pseudo-label offsets: it de-
pends on the distribution of test data, PT , instead of the distribution of unlabelled data, PU.
From this analytical viewpoint, we present a summarized Table 1 of current pseudo-label
refinement solutions. DA Berthelot et al. (2019a); Wei et al. (2021); Kim et al. (2020) is
a commonly employed technique to make balanced prediction for different classes which
align the predicted class priors to true class priors of U, making the model being fair Bridle
et al. (1991). It only reduces the calibration errors but cannot be optimally fair because
it does not take PT into account. LA modifies the network prediction from arg maxc(ˆzU
ic)
to arg maxc(ˆzU
ic −λ log πc), where λ is a hyper-parameter and π is determined as the em-
pirical class frequency Menon et al. (2020); Zhou and Liu (2005); Lazarow et al. (2023).
It shares similar design with Eq. 1, however, recall that theorem 1 provides a justification
for employing logit thresholding when optimal probabilities f∗(X) are accessible. Although
neural networks strive to mimic these probabilities, it is not realistic for LA as the classifier
is not optimal during training and neural networks are often uncalibrated and over confi-
dent Guo et al. (2017). The importance of model calibration for SSL is also highlighted
in Loh et al. (2022). The accurate estimation of classifier bias requires the calculation of
a conditional confusion matrix, which always requires holdout data Lipton et al. (2018).
However, tackling label distribution shifts is rarely discussed in the context of SSL.
3.3
Threshold Adjustment
Here, we look into the impact of pseudo-label quality on the SSL. For simplicity, we consider
a model f is trained with ˆU for binary classification using supervised learning loss L. (u, ˆy)
is an arbitrary sample drawn from ˆU. In this section we refer U = {(ui, yi)}M
i=1 to the oracle
distribution which contains the inaccessible real label y for u. ρ is the noise rate such that
1 −ρ = P ˆU(y = ˆy) and ρ < 0.5. The expected risk is RL,U(f) := E(u,y)∼U[L(f(u), y)].
The theorem presented below indicates that, with the number of samples in set U is fixed
as M, better model performance is achieved through training with a dataset exhibiting a
lower noise rate ρ.
Theorem 3 Given ˆf is the model after optimizing with ˆU. Let L be L-Lipschitz in all the
predictions, for any δ > 0, with probability at least 1 −δ,
RL,U( ˆf) ≤min
f∈F RL,U(f) + 4LpR(F) + 2
r
log(1/δ)
2M
,
(3)
7


<!-- page 8 -->
SEVAL
where the Rademacher complexity R(F) is defined by R(F) := Exi,ϵi
h
supf∈F
1
M
PM
i=1 ϵif(ui)
i
for function class F and ϵ1, . . . , ϵM are i.i.d. Rademacher variables. Lp ≤
2L
1−2ρ is the Lip-
schitz constant.
We provide proofs in Appendix Section A.
The insight of Theorem 3 is simple: given that the label size is fixed at M, we should
reduce the noise rate ρ for a better model. To achieve this, we want to increase the accuracy
of the pseudo-label, of which the marginal distribution over ˆy can be calculated as:
P ˆU(y = ˆy) =
C
X
j=1
P ˆU(y = j|ˆy = j)
|
{z
}
Precision
P ˆU(ˆy = j)
|
{z
}
Accessible
.
(4)
This indicates that, if we can have an approximation of Precision, we can feasibly tune
the pseudo-label accuracy by controlling the sampling strategies based on class-specific
thresholds τ. However, it is not possible with
P ˆU(y = ˆy) =
C
X
j=1
P ˆU(ˆy = j|y = j)
|
{z
}
Recall
P ˆU(y = j)
|
{z
}
Inaccessible
,
(5)
because in practice we do not have the information of ground truth label for ui. Formally,
we claim that:
Lemma 4 A better thresholds τ for choosing effective pseudo-labels should be derived from
class-wise Precision, instead of class-wise Recall.
However, existing dynamic threshold approaches Zhang et al. (2021); Wang et al. (2022c);
Guo and Li (2022) derive the threshold for class c based on estimated recall. Specifically,
they all rely on the maximum class probability of class c, i.e. P ′
c =
1
Kc
PK
i=1 1ic maxj pU
ij,
where 1ic = 1(arg maxj(pV
ij) = c) is 1 if the predicted most probable class is c and 0 other-
wise and Kc = PK
i=1 1ic is the number of samples predicted as c. Maximum class probability
can be used to estimate accuracy Guo et al. (2017) (which is equivalent to Recall when
assessed on a per-class basis since negative samples are not considered) of test samples Garg
et al. (2022); Li et al. (2022). Existing dynamic techniques such as FlexMatch tend to prior-
itize selections from classes linked to lower maximum class probability. Consequently, these
approaches lead to a higher sampling frequency from classes exhibiting lower Recall.
We argue that their strategies are built upon the assumption that Recall and Preci-
sion is always a trade-off as a result of moving decision boundaries, e.g. low Recall leads
to high Precision. However, they would fall short if this does not hold. For example, we
should choose as much as possible if the class is well-classified, e.g. high Recall and high
Precision. However, following their strategy, they will choose few from them. We illustrate
this in Fig. 1 with the two-moons example. While Case 1 and Case 2 are the most common
scenarios, current maximum class probability-based approaches struggle to estimate thresh-
olds effectively in other cases. We substantiate this assertion in the experimental section,
where we find that Case 3 frequently arises for the minority class in imbalanced SSL and
is currently not adequately addressed, as shown in Section 5.6 and Appendix Section D.
8


<!-- page 9 -->
SEVAL
Existing 𝑃′! 
based methods 
selecting less (𝜏! ↑)
SEVAL 
selecting more (𝜏! ↓)
/
/
Labelled data and 
Initial predictions 
After 
Optimization
Existing 𝑃′! 
based methods 
selecting more (𝜏! ↓)
SEVAL 
selecting less (𝜏! ↑)
Labelled data and 
Initial predictions 
After 
Optimization
SEVAL 
selecting less (𝜏! ↑)
Labelled data and 
Initial predictions 
SEVAL 
selecting more (𝜏! ↓)
Labelled data and 
Initial predictions 
After 
Optimization
After 
Optimization
After 
Optimization
After 
Optimization
After 
Optimization
After 
Optimization
After 
Optimization
0.825
0.825
0.825
0.825
0.833
0.833
0.750
0.750
0.750
0.750
0.808
0.808
0.892
0.892
0.892
0.892
0.892
0.942
0.850
0.850
0.850
0.850
0.817
0.850
Existing 𝑃′! 
based methods 
selecting less (𝜏! ↑)
Existing 𝑃′! 
based methods 
selecting more (𝜏! ↓)
Case 2: 
Low Recall
High Precision
Case 4: 
Low Recall
Low Precision
Case 1: 
High Recall
Low Precision
Case 3: 
High Recall
High Precision
: Unlabelled samples 𝒖" from two classes
: Original/Regularized decision boundary
: Selected unlabelled samples 𝒖" with wrong pseudo-labels
/
/
: Selected unlabelled samples 𝒖" with correct pseudo-labels
𝑃′!: Maximum class probability 
which estimates class-wise Recall
Figure 1: Two-moons toy experiments illustrating the relationship between threshold choice
and model performance for class
. Accuracy appears in the bottom right. Cur-
rent maximum class probability-based dynamic thresholding methods such as
FlexMatch Zhang et al. (2021), emphasizing Recall, may not be reliable for
Case 3 and Case 4. In comparison, SEVAL derived thresholds, reflecting Preci-
sion, fit all cases well.
4 SEVAL
Fig. 2 shows an overview of SEVAL. The comparative advantage of SEVAL over extant
methodologies is summarized in Table 1 and illustrated in Fig. 1. Importantly, we propose
to optimize these parameters using a separate labelled holdout dataset. Independent of the
training dataset X and U, we assume we have access to a holdout dataset V = {(xi, yi)}K
i=1,
which contains kc samples for class c. We make no assumptions regarding kc; that is, V can
either be balanced or imbalanced 4.
4.1
Learning Pseudo-Label Refinement
We want to further harness the potential of pseudo-label refinement by optimizing π from
the data itself.
Assuming the holdout data distribution has the same class conditional
likelihood as others and PT (Y ) is uniform, SEVAL can directly estimate the optimal decision
boundary as required in Proposition 1. Specifically, the optimal offsets π, are optimized
4. In practice, V is normally separated from X, c.f. Section 4.3.
9

[CAPTION] Figure 1: Two-moons toy experiments illustrating the relationship between threshold choice

[CAPTION] Fig. 2 shows an overview of SEVAL. The comparative advantage of SEVAL over extant


<!-- page 10 -->
SEVAL
Pseudo-Label Refinement
Threshold Adjustment
Estimate offset 
𝝅∈ℝ! based on 
class prior of 
labelled samples 𝒏
Estimate thresholds 𝝉∈ℝ! 
based on maximum class 
probability 𝑃′"
Existing 
Methods
SEVAL
Learn a 
curriculum of logit 
offsets 𝝅∈ℝ!
Learn a 
curriculum of 
threshold 𝝉∈ℝ!
: Unlabelled samples 𝒖# from two classes
/
: Original/Regularized decision boundary
: Selected unlabelled samples 𝒖# with wrong pseudo labels
/
/
/
: Selected unlabelled samples 𝒖# with correct pseudo labels
𝝅
𝝉
𝝅∗
𝝉∗
A few labelled 
validation data 𝒱
Rely on uncalibrated 
model confidence or 
heuristic strategies 
and sub-optimal
Theoretically sound 
and employs proper 
metric to select high 
quality pseudo-labels
Figure 2: Overview of SEVAL optimization process which consists of two learning strate-
gies aiming at mitigating bias in pseudo-labels within imbalanced SSL scenarios:
1) Pseudo-label refinement and 2) Threshold Adjustment. The curriculum for
parameter learning is determined through the evaluation of holdout data perfor-
mance, ensuring greater accuracy while preventing overfitting.
using the labelled holdout data V with:
π∗= arg min
π
1
K
K
X
i=1
H(yi, pV
i ) = arg min
π
1
K
K
X
i=1
H(yi, σ(zV
i −log π)).
(6)
Subsequently, we can compute the refined pseudo-label logit as ˆzU
i −log π∗, which are
expected to become more accurate on a class-wise basis. Of note, we utilize the final learned
π∗to refine the test results and expect it to perform better than LA.
4.2
Learning Threshold Adjustment
We find Precision cannot be solely determined by the network as it also relies on non-
maximum probability, as we show in Section 5.6. Thus, here we propose a novel strategy
to learn the optimal thresholds based on an external holdout dataset V. We optimize the
thresholds in a manner that ensures the selected samples from different classes achieve the
same accuracy level of t. This is achieved by:
τ ∗
c =
(
arg minτc
  A(τc) −t
  
if
t < αc
0
otherwise ,
(7)
where A(τc) calculates the accuracy of samples where the maximum probability exceeds τc :
A(τc) = 1
sc
K
X
i=1
1ic1(yi = c)1(max
j (pV
ij) > τc),
(8)
10

[CAPTION] Figure 2: Overview of SEVAL optimization process which consists of two learning strate-


<!-- page 11 -->
SEVAL
where sc = PK
i=1 1ic1(maxj(pV
ij) > τc) is the number of samples predicted as class c with
confidence larger than τc, where αc =
1
Kc
PK
i=1 1ic1(yi = c) is the average accuracy of all the
samples predicted as class c.
Notably, the optimized thresholds are inversely related to Precision and possess practical utility
in handling classes with varying accuracy. Therefore, we believe this cost function is better suited
for fair threshold optimization across diverse class difficulties. In practical scenarios, we often face
difficulties in directly determining the threshold through Eq. 7 due to the imbalances in holdout data
and constraints arising from a limited sample size. To address these issues, we employ normalized
cost functions and group-based learning, detailed further in Appendix Section C.
After obtaining the optimal refinement parameters, for pseudo-label ˆyi = arg maxj(qij) and pre-
dicted class y′
i = arg maxj(pU
ij), we can calculate the unlabelled loss ˆRH, ˆU(f) =
1
M
PM
i=1 1(maxj(qij) ≥
τy′
i)H(ˆyi, pU
i ) to update our classification model parameters. The estimation process of π and τ is
summarized in Algorithm 1.
Algorithm 1 SEVAL parameter estimation process, π∗, τ ∗←ESTIM
 V, {zV
i }K
i=1
 
Require:
1: V = {(xi, yi)}K
i=1: validation data, {zV
i }K
i=1: network prediction of V.
2: C: Number of classes, t: Requested per class accuracy of the pseudo-label, k: Number of sample
per class for V.
3: π∗= arg minπ
1
K
PK
i=1 H(yi, σ(zV
i −log π))
▷In practice, the parameter estimation process is achieved by bound-constrained solvers.
4: ωV = 1/k
▷The minority class is assigned higher weights to prioritize class-specific accuracy.
5: for c in C do
6:
Calculate class-wise accuracy αc =
1
Kc
PK
i=1 ωV
yi1ic1(yi = c)
▷For each class c, 1ic = 1(arg maxj(pV
ij) = c) and Kc = PK
i=1 ωV
yi1ic
7:
if αc > t then
8:
τ ∗
c = arg minτc
   1
sc
PK
i=1 ωV
yi1ic1(yi = c)1(maxj(pV
ij) > τc) −t
  
▷sc = PK
i=1 ωV
yi1ic1(maxj(pV
ij) > τc) is the relative number of samples predicted as class
c with confidence larger than τc
9:
else
10:
τ ∗
c = 0
▷The quality of the pseudo-labels is satisfactory, and we make use of all of them.
11:
end if
12: end for
4.3
Curriculum Learning
To bypass more data collection efforts, we learn the curriculum of π and τ based on a partition of
labelled training dataset X thus we do not require additional samples. Specifically, before standard
SSL process, we partition X into two subset X ′ and V′ which contain the same number of samples
to learn the curriculum.
In order to ensure curriculum stability, we update the parameters with exponential moving
average. Specifically, when we learn a curriculum of length L, after several iterations, we optimize
π and τ sequentially based on current model status. We then calculate the curriculum for step l as
π(l) = ηππ(l−1) + (1 −ηπ)π(l)∗and τ (l) = ηττ (l−1) + (1 −ητ)τ (l)∗. We use this to refine pseudo-
label before the next SEVAL parameter update. We summarize the training process of SEVAL in
Algorithm 2.
11


<!-- page 12 -->
SEVAL
Algorithm 2 Imbalanced semi-supervised learning with SEVAL.
Require:
1: X = {(xi, yi)}N
i=1: labelled training data, U = {ui}M
i=1: unlabelled training data, f(·): network for
classification.
2: T: Total training iterations, C: Number of classes, L: length of the curriculum, ηπ, ητ: Momentum
decay ratio of offsets and thresholds.
3: Initialize the SEVAL parameters as l = 1, π(l) =
 
1, 1, . . . , 1
 
|
{z
}
C
and τ (l) =
 
0.95, 0.95, . . . , 0.95
 
|
{z
}
C
.
▷Estimate a curriculum of the SEVAL parameters based on a partition of the training dataset.
4: Randomly partition X into two subsets, X ′ = {(xi, yi)}K
i=1 and V′ = {(xi, yi)}K
i=1, each containing an
equal number of data points.
5: for iter in [1, . . . , T] do
6:
Calculate the pseudo-label logit for unlabelled data U and obtain {ˆzU
i }M
i=1.
▷Note: FixMatch
achieves this by utilizing two augmented versions of the unlabelled data.
7:
Calculate the pseudo-label probability qi = σ(ˆzU
i −log π(l)).
8:
For pseudo-label ˆyi = arg maxj qij and predicted class y′
i = arg maxj pU
ij, calculate the unlabelled loss
ˆRH, ˆ
U(f) =
1
M
PM
i=1 1(maxj(qij) ≥τ (l)
y′
i )H(ˆyi, pU
i ).
9:
Update the network f with labelled loss ˆRH,X (f) calculated using X ′ and ˆRH, ˆ
U(f) via SGD optimizer.
10:
if iter%(T/L) = 0 then
11:
l = iterL/T
12:
Calculate the prediction on V′ using exponential moving average model and obtain {zV
i }K
i=1.
13:
π(l)∗, τ (l)∗= ESTIM(V′, {zV
i }K
i=1)
▷SEVAL parameter estimation process.
14:
π(l) = ηππ(l−1) + (1 −ηπ)π(l)∗, τ (l) = ηττ (l−1) + (1 −ητ)τ (l)∗
15:
end if
16: end for
▷Standard SSL process.
17: for iter in [1, . . . , T] do
18:
l = ⌈iterL/T⌉
19:
Calculate the pseudo-label logit for unlabelled data U and obtain {ˆzU
i }M
i=1.
20:
Calculate the pseudo-label probability qi = σ(ˆzU
i −log π(l)).
21:
Calculate the unlabelled loss ˆRH, ˆ
U(f) =
1
M
PM
i=1 1(maxj(qij) ≥τ (l)
y′
i )H(ˆyi, pU
i ).
22:
Update the network f with labelled loss ˆRH,X (f) calculated using X and ˆRH, ˆ
U(f) via SGD optimizer.
23: end for
▷Post-hoc processing with final learned parameters.
24: Given a test sample xi, the logit is adjusted from zi to zi −log π(L)∗.
5 Experiments
We conducted experiments on several imbalanced SSL benchmarks including CIFAR-10-LT, CIFAR-
100-LT Krizhevsky et al. (2009) and STL-10-LT Coates et al. (2011) under the same codebase,
following Oh et al. (2022). Specifically, we choose wide ResNet-28-2 Zagoruyko and Komodakis
(2016) as the feature extractor and train the network at a resolution of 32 × 32. We train the neural
networks for 250,000 iterations with fixed learning rate of 0.03. We control the imbalance ratios for
both labelled and unlabelled data (γl and γu) and exponentially decrease the number of samples
per class. More experiment details are provided in Appendix Section C. For most experiments,
we employ FixMatch to calculate the pseudo-label and make the prediction using the exponential
moving average version of the classifier following Sohn et al. (2020). We report the average test
accuracy along with its variance, derived from three distinct random seeds.
12


<!-- page 13 -->
SEVAL
Method type
CIFAR10-LT
CIFAR100-LT
STL10-LT
γl = γu = 100
γl = γu = 10
γl = 20, γu: unknown
Algorithm
LTL
PLR
THA
n1 = 500
n1 = 1500
n1 = 50
n1 = 150
n1 = 150
n1 = 450
m1 = 4000
m1 = 3000
m1 = 400
m1 = 300
M = 100, 000
Supervised
47.3 ±0.95
61.9 ±0.41
29.6 ±0.57
46.9 ±0.22
39.4 ±1.40
51.7 ±2.21
w/ LA Menon et al. (2020)
✓
53.3 ±0.44
70.6 ±0.21
30.2 ±0.44
48.7 ±0.89
42.0 ±1.24
55.8 ±2.22
FixMatch Sohn et al. (2020)
67.8 ±1.13
77.5 ±1.32
45.2 ±0.55
56.5 ±0.06
47.6 ±4.87
64.0 ±2.27
w/ DARP Kim et al. (2020)
✓
74.5 ±0.78
77.8 ±0.63
49.4 ±0.20
58.1 ±0.44
59.9 ±2.17
72.3 ±0.60
w/ FlexMatch Zhang et al. (2021)
✓
74.0 ±0.64
78.2 ±0.45
49.9 ±0.61
58.7 ±0.24
48.3 ±2.75
66.9 ±2.34
w/ Adsh Guo and Li (2022)
✓
73.0 ±3.46
77.2 ±1.01
49.6 ±0.64
58.9 ±0.71
60.0 ±1.75
71.4 ±1.37
w/ FreeMatch Wang et al. (2022c)
✓
✓
73.8 ±0.87
77.7 ±0.23
49.8 ±1.02
59.1 ±0.59
63.5 ±2.61
73.9 ±0.48
w/ SEVAL-PL
✓
✓
77.7 ±1.38
79.7 ±0.53
50.8 ±0.84
59.4 ±0.08
67.4 ±0.79
75.2 ±0.48
w/ ABC Wei et al. (2021)
✓
78.9 ±0.82
83.8 ±0.36
47.5 ±0.18
59.1 ±0.21
58.1 ±2.50
74.5 ±0.99
w/ SAW Lai et al. (2022b)
✓
74.6 ±2.50
80.1 ±1.12
45.9 ±1.85
58.2 ±0.18
62.4 ±0.86
74.0 ±0.28
w/ CReST+ Wei et al. (2021)
✓
✓
76.3 ±0.86
78.1 ±0.42
44.5 ±0.94
57.1 ±0.65
56.0 ±3.19
68.5 ±1.88
w/ DASO Oh et al. (2022)
✓
✓
76.0 ±0.37
79.1 ±0.75
49.8 ±0.24
59.2 ±0.35
65.7 ±1.78
75.3 ±0.44
w/ ACR Wei and Gan (2023)
✓
✓
✓
80.2 ±0.78
83.8 ±0.13
50.6 ±0.13
60.7 ±0.23
65.6 ±0.11
76.3 ±0.57
w/ SEVAL
✓
✓
✓
82.8 ±0.56
85.3 ±0.25
51.4 ±0.95
60.8 ±0.28
67.4 ±0.69
75.7 ±0.36
Table 2: Accuracy on CIFAR10-LT, CIFAR100-LT and STL10-LT. We divide SSL algo-
rithms into different groups including long-tailed learning (LTL), pseudo-label re-
finement (PLR) and threshold adjustment (THA). PLR and THA based methods
only modify pseudo-label probability qi and threshold τ, respectively. Best results
within the same category are in bold for each configuration.
Method type
Semi-Aves
Algorithm
LTL
PLR
THA
U = Uin
U = Uin + Uout
FixMatch Sohn et al. (2020)
59.9 ±0.08
52.6 ±0.14
w/ DARP Kim et al. (2020)
✓
60.3 ±0.24
54.7 ±0.06
w/ SEVAL-PL
✓
✓
60.6 ±0.18
56.4 ±0.10
w/ CReST+ Wei et al. (2021)
✓
✓
60.0 ±0.03
54.3 ±0.59
w/ DASO Oh et al. (2022)
✓
✓
59.3 ±0.28
56.6 ±0.32
w/ SEVAL
✓
✓
✓
60.7 ±0.17
56.7 ±0.15
Table 3: Accuracy on Semi-Aves. Best results within the same category are in bold for
each configuration.
5.1
Main Results
We compared SEVAL with different SSL algorithms and summarize the test accuracy results in
Table 2. To ensure fair comparison of algorithm performance, in this table, we mark SSL algorithms
based on the way they tackle the imbalance challenge. In particular, techniques such as DARP, which
exclusively manipulate the probability of pseudo-labels π, are denoted as pseudo-label refinement
(PLR). In contrast, approaches like FlexMatch, which solely alter the threshold τ, are termed as
threshold adjustment (THA). We denote other methods that apply regularization techniques to the
model’s cost function using labelled data as long-tailed learning (LTL). In addition to SEVAL results,
we also report the results of SEVAL-PL, which forgoes any post-hoc adjustments on test samples.
This ensures that its results are directly comparable with its counterparts.
As shown in Table 2, SEVAL-PL outperform other PLR and THA based methods such as
DARP, FlexMatch and FreeMatch with a considerable margin.
This indicates that SEVAL can
13

[CAPTION] Table 2: Accuracy on CIFAR10-LT, CIFAR100-LT and STL10-LT. We divide SSL algo-

[CAPTION] Table 3: Accuracy on Semi-Aves. Best results within the same category are in bold for

[CAPTION] Table 2. To ensure fair comparison of algorithm performance, in this table, we mark SSL algorithms


<!-- page 14 -->
SEVAL
provide better pseudo-label for the models by learning a better curriculum for π and τ. When
compared with other hybrid methods including ABC, CReST+, DASO, ACR, SEVAL demonstrates
significant advantages in most scenarios. Relying solely on the strength of pseudo-labeling, SEVAL
delivers highly competitive performance in the realm of imbalanced SSL. Importantly, given its
straightforward framework, SEVAL can be integrated with other SSL concepts to enhance accuracy,
a point we delve into later in the ablation study. We provide a summary of additional experimental
results conducted under diverse realistic or extreme settings in Appendix Section B.
We further apply SEVAL to the realistic imbalanced SSL dataset, Semi-Aves Su and Maji (2021),
which captures a situation where a portion of the unlabelled data originates from previously unseen
classes. This dataset, contained 200 classes with different long-tailed distribution. In addition to
labelled data, Semi-Aves also contains imbalanced unlabelled data Uin and unlabelled open-set data
Uout from another 800 classes.
Following previous works Su et al. (2021); Oh et al. (2022), we
conducted experiments using Uin or a combination of Uin and Uout. We summarize the results in
Table 3. This dataset poses a challenge due to the limited number of samples in the tail class, with
only around 15 samples per class. It has been observed that SEVAL performs effectively in such a
demanding scenario.
5.2
Varied Imbalanced Ratios
Similar to results in Table 2, we evaluate SEVAL on CIFAR10-LT with different imbalanced ratios.
We find that SEVAL consistently outperforms its counterparts across different γl values.
Since
SEVAL does not make any assumptions about the distribution of unlabeled data, it can be robustly
implemented in scenarios where γl ̸= γu. In these settings, SEVAL’s performance advantage over
its counterparts is even more pronounced.
Method type
CIFAR10-LT
γl = 100, γu = 1
γl = 100, γu = 1/100
γl = γu = 150
Algorithm
LTL
PLR
THA
n1 = 500
n1 = 1500
n1 = 500
n1 = 1500
n1 = 500
n1 = 1500
m1 = 4000
m1 = 3000
m1 = 4000
m1 = 3000
m1 = 4000
m1 = 3000
FixMatch Sohn et al. (2020)
73.0 ±3.81
81.5 ±1.15
62.5 ±0.94
71.8 ±1.70
62.9 ±0.36
72.4 ±1.03
w/ DARP Kim et al. (2020)
✓
82.5 ±0.75
84.6 ±0.34
70.1 ±0.22
80.0 ±0.93
67.2 ±0.32
73.6 ±0.73
w/ SEVAL-PL
✓
✓
89.4 ±0.53
89.2 ±0.02
77.7 ±0.91
80.9 ±0.66
71.9 ±1.10
74.7 ±0.63
w/ CReST+ Wei et al. (2021)
✓
✓
82.2 ±1.53
86.4 ±0.42
62.9 ±1.39
72.9 ±2.00
67.5 ±0.45
73.7 ±0.34
w/ DASO Oh et al. (2022)
✓
✓
86.6 ±0.84
88.8 ±0.59
71.0 ±0.95
80.3 ±0.65
70.1 ±1.81
75.1 ±0.77
w/ SEVAL
✓
✓
✓
90.3 ±0.61
90.6 ±0.47
79.2 ±0.83
82.9 ±1.78
79.8 ±0.42
83.3 ±0.40
Table 4: Accuracy on CIFAR10-LT with different imbalanced ratios. Best results within
the same category are in bold for each configuration.
5.3
Low Labelled Data Scheme
SEVAL acquires a curriculum of parameters by partitioning the training dataset.
This raises a
crucial question: can SEVAL remain effective with a very limited number of labeled samples? To
explore this, we conduct a stress test by training SEVAL with a minimal amount of labeled data.
In the first experimental configuration, we keep the imbalance ratio constant while reducing the
number of labeled samples (n1 = 200). In this extreme case, only two samples are labeled for the
tail class. In the second configuration, we use a balanced labeled training dataset, but with a total
of 100 and 40 samples for training. The results are summarized in Table 5. We find that SEVAL
performs well in both scenarios, indicating that SEVAL can be a reliable option even when the
labeled dataset is very small.
14

[CAPTION] Table 3. This dataset poses a challenge due to the limited number of samples in the tail class, with

[CAPTION] Table 4: Accuracy on CIFAR10-LT with different imbalanced ratios. Best results within


<!-- page 15 -->
SEVAL
Method type
CIFAR10-LT
γl = γu = 100
γl = 1, γu = 100
Algorithm
LTL
PLR
THA
n1 = 200
n1 = 10
n1 = 4
m1 = 4000
m1 = 4000
m1 = 4000
FixMatch Sohn et al. (2020)
64.3 ±0.83
65.3 ±0.80
44.7 ±3.33
w/ FreeMatch Wang et al. (2022c)
✓
✓
67.4 ±1.09
58.4 ±0.76
50.7 ±1.95
w/ SEVAL-PL
✓
✓
69.3 ±0.66
68.3 ±0.56
51.5 ±1.51
w/ DASO Oh et al. (2022)
✓
✓
67.2 ±1.25
61.2 ±0.96
48.6 ±2.81
w/ SEVAL
✓
✓
✓
71.2 ±0.80
68.9 ±0.25
52.7 ±1.83
Table 5: Accuracy on CIFAR10-LT under the setting of extremely few labelled samples.
Best results within the same category are in bold for each configuration.
5.4
Performance Analysis
To closely examine the distinct contributions of π and τ, we carry out an ablation study where
SEVAL optimizes just one of them, respectively termed SEVAL-PLR and SEVAL-THA. As summa-
rized in Table 6, SEVAL-PLR and SEVAL-THA can still outperform their counterparts, DARP and
FlexMatch, respectively. When tuning both parameters, SEVAL-PL can achieve the best results.
Method type
CIFAR10-LT
CIFAR100-LT
Algorithm
LTL
PLR
THA
n1 = 500, m1 = 4000
n1 = 150, m1 = 300
γl = γu = 100
γl = γu = 10
FixMatch Sohn et al. (2020)
67.8 ±1.13
56.5 ±0.06
w/ DARP Kim et al. (2020)
✓
74.5 ±0.78
58.1 ±0.44
w/ SEVAL-PLR
✓
76.7 ±0.82
59.3 ±0.30
w/ FlexMatch Zhang et al. (2021)
✓
74.0 ±0.64
58.7 ±0.24
w/ SEVAL-THA
✓
77.0 ±0.93
59.1 ±0.18
w/ SEVAL-PL
✓
✓
77.7 ±1.38
59.4 ±0.08
Table 6: Comparison of SEVAL when only optimizing π (SEVAL-PLR) or only optimiz-
ing τ (SEVAL-THA). SEVAL outperforms counterparts with identical parameter
settings under different imbalanced SSL scenarios. SEVAL-PL, with its sequential
optimization of both π and τ, yields further improvements in accuracy.
5.4.1
Pseudo-Label Refinement
In order to comprehensively and quantitatively investigate the accuracy of pseudo-label refined by
different approaches, here we define G as the sum of accuracy gain and balanced accuracy gain
of pseudo-label over training iterations.
Both sample-wise accuracy and class-wise accuracy are
crucial measures for evaluating the quality of pseudo-labels. A low sample-specific accuracy can
lead to noisier pseudo-labels, adversely affecting model performance. Meanwhile, a low class-specific
accuracy often indicates a bias towards the dominant classes. Therefore, we propose a metric G
15

[CAPTION] Table 5: Accuracy on CIFAR10-LT under the setting of extremely few labelled samples.

[CAPTION] Table 6: Comparison of SEVAL when only optimizing π (SEVAL-PLR) or only optimiz-


<!-- page 16 -->
SEVAL
(a)
(b)
(c)
Figure 3: (a) The evolution of Gain across training iterations.
SEVAL accumulates a
higher accuracy of pseudo-label than its counterparts. (b) The evolution of Cor-
rectness across training iterations. SEVAL can build better trade-off between
quality and quantity. (c) The evolution of test accuracy across training iterations.
SEVAL-PL outperforms other pseudo-label refinement methods.
which is the combination of these two metrics. Specifically, given the pseudo-label ˆyi and predicted
class y′
i of unlabelled dataset U, we calculate G as:
G =
PM
i=1[1(ˆy′
i = yi) −1(ˆyi = yi)]
M
|
{z
}
Sample-Wise Accuracy Gain
+
C
X
c=1
M
X
i=1
1(ˆy′
i = c)1(ˆy′
i = yi) −1(ˆyi = c)1(ˆyi = yi)
mcC
|
{z
}
Class-Wise Accuracy Gain
.
(9)
To evaluate the cumulative impact of pseudo-labels, we calculate Gain(iter) as the accuracy
gain at training iteration iter and monitor Gain(iter) = Piter
j=1 G(j)/iter throughout the training
iterations. The results of SEVAL along with DARP and adjusting pseudo-label logit ˆzU
c with LA are
summarized in Fig. 3(a). We note that SEVAL consistently delivers a positive Gain throughout the
training iterations. In contrast, DARP and LA tend to reduce the accuracy of pseudo-labels during
the later stages of the training process. After a warm-up period, DARP adjusts the distribution of
pseudo-labels to match the inherent distribution of unlabelled data. However, it doesn’t guarantee
the accuracy of the pseudo-labels, thus not optimal. While LA can enhance class-wise accuracy, it
isn’t always the best fit for every stage of the model’s learning. Consequently, noisy pseudo-labels
from the majority class can impede the model’s training. SEVAL learns a smooth curriculum of
parameters for pseudo-label refinement from the data itself, therefore bringing more stable improve-
ments. We can further validate the effectiveness of SEVAL from the test accuracy curves shown in
Fig. 3(c) where SEVAL-PL outperforms LA and DARP.
5.4.2
Threshold Adjustment
Quantity and quality are two essential factors for pseudo-labels, as highlighted in Chen et al. (2023).
Quantity refers to the number of correctly labeled samples produced by pseudo-label algorithms,
while quality indicates the proportion of correctly labeled samples after applying confidence-based
thresholding.
In order to access the effectiveness of pseudo-label, we propose a metric called
Correctness, which is a combination of quantity and quality. Having just high quantity or just
high quality isn’t enough for effective pseudo-labels. For instance, setting exceedingly high thresh-
olds might lead to the selection of a limited number of accurately labelled samples (high quality).
16


**[Table p16.1]**
|  | (a) | (b |  |  |  |  |  | (c) |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  | (b | ) |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
| Figure which i class y′ i | 3: (a) The evolution of Gai higher accuracy of pseudo- rectness across training i quality and quantity. (c) T SEVAL-PL outperforms ot s the combination of these two m of unlabelled dataset U, we calc |  |  | n across training iterations. label than its counterparts. (b) terations. SEVAL can build b he evolution of test accuracy ac her pseudo-label refinement m etrics. Specifically, given the pseu ulate G as: C M |  |  |  |  | SEVAL accumulates a The evolution of Cor- etter trade-off between ross training iterations. ethods. do-label yˆ and predicted i |


**[Table p16.2]**
|  | (a) |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |


**[Table p16.3]**
| (c) |  |  |  |
| --- | --- | --- | --- |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |

[CAPTION] Figure 3: (a) The evolution of Gain across training iterations.


<!-- page 17 -->
SEVAL
However, this is not always the ideal approach, and the opposite holds true for quantity. Therefore,
we propose a metric Correctness which combine quality and quantity. In particular, factoring in
the potential imbalance of unlabelled data, we utilize a class frequency based weight term ωU = 1/m
to normalize this metric, yielding:
Correctness =
C
PM
i=1 ωUyi
|
{z
}
Quantity
C
PM
i=1 ωUyi1(maxj(qij) ≥τy′
i)
|
{z
}
Quality
,
(10)
where, C = PM
i=1 ωU
yi1(ˆyi = yi)1(maxj(qij) ≥τy′
i) is the relative number of correctly labelled
samples. We show Correctness of SEVAL with FixMatch, FlexMatch and FreeMatch in Fig. 3(a).
We observe that FlexMatch and FreeMatch can both improve Correctness, while SEVAL can boost
even more. We observe that the test accuracy follows a trend similar to Correctness, as shown in
Fig. 3(b). This demonstrates that the thresholds set by SEVAL not only ensure a high quantity but
also attain high accuracy for pseudo-labels, making them efficient in the model’s learning process.
5.5
Ablation Study
(a)
(b)
(c)
Figure 4: (a) Test accuracy when SEVAL is adapted to pseudo-label based SSL algorithms
other than FixMatch under the setting of CIFAR-10 n1 = 1500. SEVAL can
readily improve the performance of other SSL algorithsm. (b) Test accuracy when
SEVAL employs varied types of post-hoc adjustment parameters. The learned
post-hoc parameters consistently enhance performance, particularly in CIFAR-
10 experiments.
(c) Test accuracy when SEVAL is optimized using different
validation samples under the setting of CIFAR-10 n1 = 500. SEVAL requires few
validation samples to learn the optimal curriculum of parameters.
5.5.1
Flexibility and Compatibility
We apply SEVAL to other pseudo-label based SSL algorithms including Mean-Teacher, MixMatch
and ReMixMatch and report the results with the setting of CIFAR-100 n1 = 50 in Fig. 4(a). We
find SEVAl can bring substantial improvements to these methods and is more effective than DASO.
Of note the results of ReMixMatch w/SEVAL is higher than the results of FixMatch w/ SEVAL in
Table 2 (86.7 vs 85.3). This may indicates that ReMixMatch is fit imbalanced SSL better. Due to
its simplicity, SEVAL can be readily combined with other SSL algorithms that focus on LTL instead
17


**[Table p17.1]**
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Figur |  | e | 4: | ( o r S p | a) th ea E o | T er di VA st- | est th ly i L ho | acc an mpr em c pa | ura Fix ove ploy ram | cy when S Match un the perfo s varied t eters con |  |  | EV de rm yp sis | A r an es te | L th ce o nt | is e o f ly | ad sett f ot pos en | apte ing her t-ho hanc | d t of SS c a e p | o ps CIF L al dju erf | eudo-label based SSL algorithms AR-10 n = 1500. SEVAL can 1 gorithsm. (b) Test accuracy when stment parameters. The learned ormance, particularly in CIFAR- |  |  |

[CAPTION] Figure 4: (a) Test accuracy when SEVAL is adapted to pseudo-label based SSL algorithms

[CAPTION] Table 2 (86.7 vs 85.3). This may indicates that ReMixMatch is fit imbalanced SSL better. Due to


<!-- page 18 -->
SEVAL
of PLR and THA. For example, SEVAL pairs effectively with the semantic alignment regularization
introduced by DASO. By incorporating this loss into our FixMatch experiments with SEVAL, we
were able to boost the test accuracy from 51.4 to 52.4 using the CIFAR-100 n1 = 50 configuration.
We compare with the post-hoc adjustment process with LA in Fig. 4(b). We find that those
post-hoc parameters can improve the model performance in the setting of CIFAR-10. In other cases,
our post-hoc adjustment doesn’t lead to a decrease in prediction accuracy. However, LA sometimes
does, as seen in the case of STL-10. This could be due to the complexity of the confusion matrix in
those instances, where the class bias is not adequately addressed by simple offsets.
5.5.2
Data-Efficiency
Here we explore if SEVAL requires a substantial number of validation samples for curriculum learn-
ing. To do so, we keep the training dataset the same and optimize SEVAL parameters using balanced
validation dataset with varied numbers of labelled samples using the CIFAR-10 n1 = 500 configura-
tion, as shown in Fig. 4(c). We find that SEVAL consistently identifies similar π and τ. When we
train the model using these curricula, there aren’t significant differences even when the validation
samples per class ranges from 10 to 500. This suggests that SEVAL is both data-efficient and re-
silient. We conduct stress tests on SEVAL and observe its effectiveness, even with only 40 labelled
samples in total, as detailed in the Section 5.3.
5.6
Analysis of Learned Thresholds
(a)
(b)
Case 3:
High Recall
High Precision
Case 4:
Low Recall
Low Precision
Estimated Precision 𝑄′!
Figure 5: The correlation of different metrics between test Precision of FixMatch on
CIFAR10-LT n1 = 500.
(a) The correlation of SEVAL learned τc and maxi-
mum class probability P ′
c between test Precision. Each point represents a class
c and the size of the points indicate the number of samples in the labelled train-
ing dataset nc. Note that maximum class probability P ′
c is the basis of current
dynamic threshold method to derive thresholds. For example, FlexMatch selects
more samples for classes associated with lower P ′
c. However, as highlighted by
red arrows, P ′
c does not correlated with Precision thus Pc based on methods will
fail Case 3: High Recall & High Precision and Case 4: Low Recall & Low
Precision in Fig. 1. (b) Due to the lack of calibration in the network output
probability, the estimated precision derived from the probability does not align
with the actual Precision, thus cannot be a reliable metric to derive thresholds.
18


**[Table p18.1]**
| (a) Case High R High Pre Case 4: Low Recall Low Precision | 3: eca cis | (b) ll ion Estimated Precision 𝑄′! |
| --- | --- | --- |


**[Table p18.2]**
| (a) |  |  |  |  |  | Cas |
| --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  | H |  | High igh Pr |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  | Case 4: |  |  |  |  |
|  |  |  |  |  |  |  |
|  | Lo | Low Reca w Precis | ll ion |  |  |  |


**[Table p18.3]**
| (b) |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |

[CAPTION] Figure 5: The correlation of different metrics between test Precision of FixMatch on


<!-- page 19 -->
SEVAL
We try to determine the effectiveness of thresholds by looking into Precision of different classes,
which should serve as approximate indicators of suitable thresholds. We illustrate an example of
optimized thresholds and the learning status of FixMatch on CIFAR10-LT n1 = 500 in Fig. 5(a),
where SEVAL learns τc to be low for classes that have high Precision. In contrast, maximum class
probability P ′
c, does not show clear correction with Precision. Specifically, as highlighted with the
red arrows, Pc remains high for classes that exhibit high Precision. Consequently, maximum class
probability-based threshold methods such as FlexMatch will tune the threshold to be high for classes
with large Pc, inadequately addressing Case 3 and Case 4 as elaborated in Section 3.
Instead of depending on an independent labelleddataset, we also attempt to estimate Precision
using the model probability, so as to leverage the estimated precision to determine the appropriate
thresholds. Specifically, we estimate the Precision of class c as:
Q′
c =
PK
i=1 1ic maxj pU
ij
PK
i=1 pU
ic
.
(11)
We visualize the estimated precision in Fig. 5(b). We find the the estimated precision does not
align with the actual Precision. This is because the model is uncalibrated and the Q′
c is heavily
decided by the true positives parts (e.g. numerator in Eq. 11), thus cannot reflect the real model
precision. Thus it is a essential to utilize a holdout labelled dataset to derive the optimal thresholds.
Finally, we look into the class-wise performance of SEVAL and its counterparts in Fig 6. When
compared with alternative methods, SEVAL achieves overall better performance with higher Recall
on minority classes and higher Precision on majority classes. In this case, class 6 falls into Case
3: High Recall & High Precision while class 5 falls into Case 4: Low Recall & Low Precision.
SEVAL shows advantages on these classes.
(a)
(b)
(c)
CIFAR10-LT
𝛾! = 𝛾" = 100, 𝑛# = 500, 𝑚# = 4000
Figure 6: Class-wise performance for different SSL methods. Class indexes are arranged in
descending order according to their class frequencies. We find that SEVAL achieve
better overall performance than its counterparts by making neural networks more
sensitive to minority classes.
6 Conclusion and Future Work
In this study, we present SEVAL and highlight its benefits in imbalanced SSL across a wide range of
application scenarios. SEVAL sheds new light on pseudo-label generalization, which is a foundation
for many leading SSL algorithms. SEVAL is both straightforward and potent, requiring no extra
19


**[Table p19.1]**
| CIFAR10-LT 𝛾 = 𝛾 = 100, 𝑛 = 500, 𝑚 = 4000 ! " # # (a) (b) (c) |  |
| --- | --- |
| (a) | (c) |


**[Table p19.2]**
|  | (a) |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |


**[Table p19.3]**
| (b) |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |


**[Table p19.4]**
| (c) |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |

[CAPTION] Figure 6: Class-wise performance for different SSL methods. Class indexes are arranged in


<!-- page 20 -->
SEVAL
computation once the curriculum is acquired. As such, it can be effortlessly integrated into other
SSL algorithms and paired with LTL methods to address class imbalance.
We believe that the
concept of optimizing parameters or accessing unbiased learning status using a partition of the
labeled training dataset could spark further innovations in long-tailed recognition and SSL. We feel
that the specific interplay between label refinement and threshold adjustment remains an intriguing
question for subsequent research.
In the future, by leveraging Bayesian or bootstrap techniques, we may eliminate the need for in-
ternal validation in SEVAL by improving model calibration Loh et al. (2022); Vucetic and Obradovic
(2001). We also plan to analyze SEVAL within the theoretical framework of SSL Mey and Loog
(2022) to acquire deeper insights.
Acknowledgments and Disclosure of Funding
SJ is supported by a Wellcome Senior Research Fellowship (221933/Z/20/Z) and a Wellcome Collab-
orative Award (215573/Z/19/Z). The Wellcome Centre for Integrative Neuroimaging is supported by
core funding from the Wellcome Trust (203139/Z/16/Z). The computational aspects of this research
were partly carried out at Oxford Biomedical Research Computing (BMRC), which is funded by the
NIHR Oxford BRC with additional support from the Wellcome Trust Core Award Grant Number
203141/Z/16/Z.
20


<!-- page 21 -->
SEVAL
Appendix
Table of Contents
A
Proofs
22
A.1 Proof of Proposition 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
22
A.2 Proof of Theorem 3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
22
A.3 Proof of Lemma 5
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
23
B
Additional Experiments
24
B.1
Sensitivity Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
24
B.2
Results on ImageNet-127 . . . . . . . . . . . . . . . . . . . . . . . . . . . .
24
B.3
Integration with Other SSL Frameworks . . . . . . . . . . . . . . . . . . .
25
C
Implementation Details
26
C.1 Learning with Imbalanced Validation Data
. . . . . . . . . . . . . . . . .
26
C.2 Learning Thresholds within Groups . . . . . . . . . . . . . . . . . . . . . .
26
C.3 Benchmarks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
27
C.4 Hyper-Parameters
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
27
C.5 SEVAL with Other SSL Algorithms
. . . . . . . . . . . . . . . . . . . . .
27
D
Classes of Different Performance
28
21


<!-- page 22 -->
SEVAL
Appendix A. Proofs
A.1
Proof of Proposition 1
fT (X) = PT (Y |X)
= PT (X|Y )PT (Y )
PT (X)
,
(12)
when we assume there does not exist conditional shifts between training and test dataset follow-
ing Saerens et al. (2002), e.g. PX (X|Y ) = PT (X|Y ), we can rewrite Eq. 12 as:
PT (X|Y )PT (Y )
PT (X)
= PT (X|Y )PT (Y )
PT (X)
· PX (X)
PX (X)
∝PX (X|Y )PT (Y )
PX (X)
= PX (X, Y )PT (Y )
PX (X)PX (Y )
= PX (Y |X)PT (Y )
PX (Y )
= f ∗(X)PT (Y )
PX (Y )
.
(13)
■
A.2
Proof of Theorem 3
We derive the generalization bound based on the Rademacher complexity method Bartlett and
Mendelson (2002) following the analysis for noisy label training Natarajan et al. (2013); Liu and
Tao (2015).
Lemma 5 We define:
ˆL(·, y) := (1 −ρ)L(·, y) −ρL(·, 3 −y)
1 −2ρ
.
(14)
With this loss function, we should have:
Eˆy[ ˆL(·, ˆy)] = L(·, y).
(15)
Given ˆR ˆ
L, ˆU(f) :=
1
M
PM
i=1 ˆL(f(ui), ˆyi) the empirical risk on ˆU and R ˆ
L, ˆU(f) is the corresponding
expected risk, we have the basic generalization bound as:
max
f∈F | ˆR ˆ
L, ˆU(f) −R ˆ
L, ˆU(f)| ≤2R( ˆL ◦F) +
r
log(1/δ)
2M
,
(16)
where:
R( ˆL ◦F) := Eui,ˆyi,ϵi
"
sup
f∈F
1
M
M
X
i=1
ϵi ˆL(f(ui), ˆyi)
#
.
(17)
If L is L-Lipschitz then ˆL is Lρ Lipschitz with:
Lρ =
L
1 −2ρ.
(18)
22


<!-- page 23 -->
SEVAL
Based on Talagrand’s Lemma Mohri et al. (2018), we have:
R( ˆL ◦F) ≤LρR(F),
(19)
where R(F) := Exi,ϵi
h
supf∈F
1
M
PM
i=1 ϵif(ui)
i
is the Rademacher complexity for function class
F and ϵ1, . . . , ϵM are i.i.d. Rademacher variables.
Let ˆf be the model after optimizing with ˆU, and let f ∗be the minimization of the expected risk
RL,U over F. Then, we have:
RL,U( ˆf) −RL,U(f ∗)
= R ˆ
L, ˆU( ˆf) −R ˆ
L, ˆU(f ∗)
=
  ˆR ˆ
L, ˆU( ˆf) −ˆR ˆ
L, ˆU(f ∗)
 
+
  ˆR ˆ
L, ˆU(f ∗) −R ˆ
L, ˆU(f ∗)
 
+
 R ˆ
L, ˆU( ˆf) −ˆR ˆ
L, ˆU( ˆf)
 
≤0 + 2 max
f∈F | ˆR ˆ
L, ˆU(f) −R ˆ
L, ˆU(f)|
(20)
After we combine Eq. 16 and Eq. 20, we obtain the bound described in Theorem 3.
■
A.3
Proof of Lemma 5
Based on Eq. 15, we calculate the cases for y = 1 and y = 2 and obtain:
(1 −ρ) ˆL(·, 1) + ρ ˆL(·, 2) = L(·, 1),
(21)
(1 −ρ) ˆL(·, 2) + ρ ˆL(·, 1) = L(·, 2).
(22)
By solving the two equations, we yield:
ˆL(·, 1) = (1 −ρ)L(·, 1) −ρL(·, 2)
1 −2ρ
,
(23)
ˆL(·, 2) = (1 −ρ)L(·, 2) −ρL(·, 1)
1 −2ρ
.
(24)
■
23


<!-- page 24 -->
SEVAL
Appendix B. Additional Experiments
In this section, we present additional experimental results conducted under various settings to assess
the generalizability of SEVAL.
B.1
Sensitivity Analysis
We perform experiments with SEVAL, varying the core hyperparameters, and present the results
in Table 7. Our findings indicate that SEVAL exhibits robustness, showing insensitivity to hyper-
parameter variations within a reasonable range.
CIFAR10-LT, γl = 100
Hyper-parameter
n1 = 500, m1 = 4000
t = 0.6
82.5 ±0.45
t = 0.7
82.2 ±0.11
t = 0.75
(reported)
82.8 ±0.56
ηπ = 0.995
81.4 ±0.36
ηπ = 0.999
(reported)
82.8 ±0.56
ηπ = 0.9995
82.5 ±0.35
ητ = 0.995
81.5 ±0.38
ητ = 0.999
(reported)
82.8 ±0.56
ητ = 0.9995
82.9 ±0.09
Table 7: Sensitivity analysis of hyper-parameters t, ηπ and ητ. Best results are in bold for
each configuration.
B.2
Results on ImageNet-127
Method type
Small-
ImageNet-127
Algorithm
LTL
PLR
THA
FixMatch
29.4
w/ FreeMatch
✓
✓
30.0
w/ SEVAL-PL
✓
✓
31.0
w/ SAW
✓
29.4
w/ DASO
✓
✓
29.4
w/ SEVAL
✓
✓
✓
34.8
Table 8: Averaged class recall on Small-ImageNet-127. Best results within the same cate-
gory are in bold for each configuration.
We conduct experiments on Small ImageNet-127 Su and Maji (2021) with a reduced image size
of 32 × 32 following Fan et al. (2022). ImageNet-127 consolidates the 1000 classes of ImageNet
24

[CAPTION] Table 7: Sensitivity analysis of hyper-parameters t, ηπ and ητ. Best results are in bold for

[CAPTION] Table 8: Averaged class recall on Small-ImageNet-127. Best results within the same cate-


<!-- page 25 -->
SEVAL
into 127 categories according to the WordNet hierarchy. This results in a naturally long-tailed class
distribution with an imbalance ratio of γl = γu = 286. We randomly select 10% of the training
samples as the labeled set and utilize the remainder as unlabeled data. We conduct experiments
using ResNet-50 and Adam optimizer. Results are summarized in Table 8.
B.3
Integration with Other SSL Frameworks
As an extension to results in Fig. 4, we summarize the results when introducing SEVAL into other
SSL frameworks in Table 9. We summarize the implementation details of those methods in Section C.
CIFAR10-LT
CIFAR100-LT
Algorithm
n1 = 1500, m1 = 3000
n1 = 150, m1 = 300
γl = γu = 100
γl = γu = 10
Mean Teacher Tarvainen and Valpola (2017)
68.6 ±0.88
52.1 ±0.09
w/ DASO Oh et al. (2022)
70.7 ±0.59
52.5 ±0.37
w/ SEVAL
77.6 ±0.63
53.8 ±0.24
MixMatch Berthelot et al. (2019b)
65.7 ±0.23
54.2 ±0.47
w/ DASO Oh et al. (2022)
70.9 ±1.91
55.6 ±0.49
w/ SEVAL
81.8 ±0.82
57.8 ±0.26
ReMixMatch Berthelot et al. (2019a)
77.0 ±0.55
61.5 ±0.57
w/ DASO Oh et al. (2022)
80.2 ±0.68
62.1 ±0.69
w/ SEVAL
86.7 ±0.71
63.1 ±0.38
Table 9: Accuracy on CIFAR10-LT based on SSL methods other than FixMatch.
Best
results within the same category are in bold for each configuration.
25

[CAPTION] Table 9: Accuracy on CIFAR10-LT based on SSL methods other than FixMatch.


<!-- page 26 -->
SEVAL
Appendix C. Implementation Details
C.1
Learning with Imbalanced Validation Data
As the labelled training dataset X is imbalanced, in practice, it is hard to obtain a balanced split V
to learn a curriculum of threshold τ. However, when we optimize τ using an imbalanced validation V
following Eq. 7, the optimized results would be biased. More precisely, the majority class consistently
exhibits high Precision, leading to a lower threshold, while the opposite holds true for the minority
class. Therefore, we utilize the class frequency of the labelled validation data k to normalize the
cost function. Specifically, we calculate the class weight as ωV = 1/k. This parameter would assign
large weight to the minority class and small weight to the majority classes. Then we replace all the
1ic with ωV
yi1ic in Eq. 7, obtaining:
τ ∗
c =
(
arg minτc
   1
sc
PK
i=1 ωV
yi1ic1(yi = c)1(maxj(pV
ij) > τc) −t
  
if
t < αc
0
otherwise ,
(25)
where sc = PK
i=1 ωV
yi1ic1(maxj(pV
ij) > τc) is the relative number of samples predicted as class
c with confidence larger than τc, where αc =
1
Kc
PK
i=1 ωV
yi1ic1(yi = c) is the average balanced
accuracy of all the samples predicted as class c and Kc = PK
i=1 ωV
yi1ic is the relative number of
samples predicted as c. This modification can normalize the number of samples within the cost
function. Consequently, we can directly learn the thresholds τ using imbalanced validation data.
C.2
Learning Thresholds within Groups
When we learn τ based on the validation data V, the optimization process could be unstable as
sometimes we have very few samples per class (e.g. less than 10 samples). In this case, even if we
can re-weight the validation samples based on their class prior k, it is hard to have enough samples to
obtain stable τ curriculum for the minority classes, especially when minc(kc) < 10. Assuming equal
class priors should result in similar thresholds, we propose to optimize thresholds within groups,
pinpointing the ideal ones that fulfill the accuracy requirement for every classes within the group.
We assume the samples of different classes kc are arranged in descending order. In other words,
k1 is the maximum, and kC is the minimum. Instead of optimizing τc for an individual class c, we
optimize for groups such that the learned τb can satisfy the accuracy requirements for B classes.
Specifically, the optimal ˜τ ∈RC/B is determined as:
˜τ ∗
b =
(
arg min˜τb
   1
˜sb
PbB+B
c=bB+1
PK
i=1 1ic1(yi = c)1(maxj(pV
ij) > ˜τb) −t
  
if
t < ˜αb
0
otherwise ,
(26)
where ˜sb = PbB+B
c=bB+1
PK
i=1 1ic1(maxj(pV
ij) > ˜τb) is the number of samples that are chosen in this
group based on the threshold ˜τb and ˜αb =
1
PbB+B
c=bB+1 Kc
PbB+B
c=bB+1
PK
i=1 1ic1(yi = c) is the average
accuracy of all the samples predicted as class in this group.
If we set B = 1, Eq. 26 becomes
equivalent to Eq. 7.
Furthermore, in practice, we find that in imbalanced SSL settings, the minority classes sometimes
have very few samples, making it difficult to optimize the thresholds correctly based on Eq. 26. In
this case, we also set the learned ˜τ ∗
b to be 0, in order to leverage more data from the minority classes.
Formally, we denote ˜Kb = PbB+B
c=bB+1
PK
i=1 ωV
yi1ic as the relative number of predicted samples within
group b. When ˜Kb < PK
i=1
BωV
yi
e1C or PK
i=1 1(yi = c) < e2, where e1 and e2 are hyper-parameters that
we both set to 10 for all experiments, we also have ˜τ ∗
b = 0 and keep their corresponding πc within
group b as low as πc = minj(πj). This implies:
26


<!-- page 27 -->
SEVAL
• In instances where the models exhibit a pronounced bias, limiting their capability to detect
over 10% of the samples within a particular group, we adjust the associated thresholds and
consequently increase our sample selection.
• When a group comprises fewer than 10 samples, the feasibility of optimizing thresholds based
on proportion diminishes, necessitating an enhanced sample selection.
C.3
Benchmarks
We conduct experiments upon the code base of Oh et al. (2022) for experiments of CIFAR10-LT,
CIFAR100-LT and STL10-LT. We take some baseline results from the DASO paper Oh et al. (2022)
to Table 2, Table 4 and Table 9. including the results of supervised baselines, DARP, CReST+,
ABC and DASO.
As DASO Oh et al. (2022) does not supply the code for the Semi-Aves experiments, we conduct all
the experiments for this setting ourselves. We train ResNet-50 He et al. (2016) which is pretrained on
ImageNet Deng et al. (2009) for the task of Semi-Aves following Su and Maji (2021). In accordance
with Oh et al. (2022), we merge the training and validation datasets provided by the challenge,
yielding a total of 5,959 samples for training which come from 200 classes. We conduct experiments
utilizing 26,640 unlabelled samples which share the same label space with X in the U = Uin setting,
and 148,848 unlabelled samples of which 122,208 are from open-set classes in the U = Uin + Uout
setting. For experiments on Semi-Aves, we set the base learning rate as 0.005. We train the network
for 45,000 iterations. The learning rate is linear warmed up during the first 25,00 iterations, and
degrade after 15,000 and 30,000, with a factor of 10. We choose training batch size as 32. The
images are firstly cropped to 256 × 256. During training, the images are then randomly cropped to
224 × 224. At inference time, the images are cropped in the center with size 224 × 224.
C.4
Hyper-Parameters
Here we summarize all the hyper-parameters we choose in this experiments to ease reproducibility.
Hyper-parameter
CIFAR10-LT, γl = 100
CIFAR100-LT, γl = 10
STL10-LT, γl = 20
Semi-Aves
n1 = 500
n1 = 1500
n1 = 50
n1 = 150
n1 = 150
n1 = 450
U = Uin
U = Uin + Uout
m1 = 4000
m1 = 3000
m1 = 400
m1 = 300
M = 100, 000
C
10
100
10
200
T
250,000
250,000
250,000
45,000
t
0.75
0.5
0.65
0.7
0.6
0.9
0.99
tV
0.9
0.65
0.7
0.95
0.85
—–
L
500
100
500
90
B
2
25
10
2
1
10
ηπ
0.999
0.95
0.9
0.995
0.99
0.9
ητ
0.999
0.95
0.9
0.9995
0.999
0.99
0.9
Table 10: Experiment-specific hyper-parameters. tV is the required accuracy if we directly
optimize τ along the training process using a separate validation dataset.
C.5
SEVAL with Other SSL Algorithms
Here, we provide implementation details of how SEVAL can be integrated into other pseudo-labeling
based SSL algorithms. Specifically, we apply SEVAL to Mean Teacher Tarvainen and Valpola (2017),
MixMatch Berthelot et al. (2019b) and ReMixMatch Berthelot et al. (2019a). These algorithms
27

[CAPTION] Table 10: Experiment-specific hyper-parameters. tV is the required accuracy if we directly


<!-- page 28 -->
SEVAL
produce pseudo-label ˆyi based on its corresponding pseudo-label probability qi and logit ˆzU
i
in
different ways. SEVAL can be easily adapted by refining qi using the learned offset π∗.
It should be noted that these SSL algorithms do not include the process of filtering out pseudo-
labels with low confidence. Therefore, for simplicity and fair comparison, we do not include the
threshold adjustment into these methods. We expect that SEVAL can further enhance performance
through threshold adjustment and plan to explore this further in the future.
C.5.1
Mean Teacher
Mean Teacher generates pseudo-label logit ˆzU
i based on a EMA version of the prediction models.
SEVAL calculates the pseudo-label probability as qi = σ(ˆzU
i −log π∗), which is expected to have
less bias towards the majority class.
C.5.2
MixMatch
MixMatch calculates ˆyi based on multiple transformed version of an unlabelled sample ui. SEVAL
adjusts each one of them with π∗, separately.
C.5.3
ReMixMatch
ReMixMatch proposes to refine pseudo-label probability qi with distribution alignment to match
the marginal distributions. SEVAL adjusts the the probability using qi = σ(ˆzU
i −log π∗) before
ReMixMatch’s process including distribution alignment and temperature sharpening.
Appendix D. Classes of Different Performance
Here we summarize the number of classes of different performance, as demonstrated in Fig. 1. We
report the model performance of FixMatch trained on different datasets. Here we consider a class
to have high Recall when its performance surpasses the average Recall of the classes. The same
principle applies to Precision. We observe that Case1 is frequently encountered in the majority
class, while Case2 is commonly observed in the minority class. Case3 and Case4 also widely exist in
different scenarios. The threshold learning strategies in SEVAL fits Case3 and Case4, thus perform
better.
CIFAR10-LT
CIFAR100-LT
STL10-LT
Semi-Aves
γl = γu = 100
γl = γu = 10
γl = 20, γu: unknown
Model Performance
n1 = 500
n1 = 1500
n1 = 50
n1 = 150
n1 = 150
n1 = 450
U = Uin
U =
m1 = 4000
m1 = 3000
m1 = 400
m1 = 300
M = 100, 000
Uin + Uout
Case 1: High Recall & Low Precision
4
5
24
29
2
4
36
45
Case 2: Low Recall & High Precision
4
4
18
20
2
2
35
45
Case 3: High Recall & High Precision
1
1
33
29
3
2
69
58
Case 4: Low Recall & Low Precision
1
0
25
22
3
2
60
52
Total classes
10
10
100
100
10
10
200
200
Table 11: The number of classes of different performance when trained with FixMatch. We
demonstrate the prevalent occurrence of the four class performance scenarios in
existing semi-supervised learning tasks.
28

[CAPTION] Table 11: The number of classes of different performance when trained with FixMatch. We


<!-- page 29 -->
SEVAL
References
Peter L Bartlett and Shahar Mendelson. Rademacher and gaussian complexities: Risk bounds and
structural results. Journal of Machine Learning Research, 3(Nov):463–482, 2002.
David Berthelot, Nicholas Carlini, Ekin D Cubuk, Alex Kurakin, Kihyuk Sohn, Han Zhang, and
Colin Raffel. Remixmatch: Semi-supervised learning with distribution alignment and augmenta-
tion anchoring. arXiv preprint arXiv:1911.09785, 2019a.
David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver, and Colin A
Raffel. Mixmatch: A holistic approach to semi-supervised learning. Advances in neural informa-
tion processing systems, 32, 2019b.
John Bridle, Anthony Heading, and David MacKay. Unsupervised classifiers, mutual information
and’phantom targets. Advances in neural information processing systems, 4, 1991.
Kaidi Cao, Colin Wei, Adrien Gaidon, Nikos Arechiga, and Tengyu Ma.
Learning imbalanced
datasets with label-distribution-aware margin loss. Advances in neural information processing
systems, 32, 2019.
Olivier Chapelle, Bernhard Scholkopf, and Alexander Zien. Semi-supervised learning (chapelle, o.
et al., eds.; 2006)[book reviews]. IEEE Transactions on Neural Networks, 20(3):542–542, 2009.
Nitesh V Chawla, Kevin W Bowyer, Lawrence O Hall, and W Philip Kegelmeyer. Smote: synthetic
minority over-sampling technique. Journal of artificial intelligence research, 16:321–357, 2002.
Hao Chen, Yue Fan, Yidong Wang, Jindong Wang, Bernt Schiele, Xing Xie, Marios Savvides, and
Bhiksha Raj. An embarrassingly simple baseline for imbalanced semi-supervised learning. arXiv
preprint arXiv:2211.11086, 2022.
Hao Chen, Ran Tao, Yue Fan, Yidong Wang, Jindong Wang, Bernt Schiele, Xing Xie, Bhiksha Raj,
and Marios Savvides. Softmatch: Addressing the quantity-quality trade-off in semi-supervised
learning. arXiv preprint arXiv:2301.10921, 2023.
Adam Coates, Andrew Ng, and Honglak Lee. An analysis of single-layer networks in unsupervised
feature learning. In Proceedings of the fourteenth international conference on artificial intelligence
and statistics, pages 215–223. JMLR Workshop and Conference Proceedings, 2011.
Ekin D Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V Le. Randaugment: Practical automated
data augmentation with a reduced search space. In Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition workshops, pages 702–703, 2020.
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale
hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition,
pages 248–255. Ieee, 2009.
Yue Fan, Dengxin Dai, Anna Kukleva, and Bernt Schiele. Cossl: Co-learning of representation and
classifier for imbalanced semi-supervised learning. In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pages 14574–14584, 2022.
Saurabh Garg, Sivaraman Balakrishnan, Zachary C Lipton, Behnam Neyshabur, and Hanie
Sedghi. Leveraging unlabeled data to predict out-of-distribution performance. arXiv preprint
arXiv:2201.04234, 2022.
29


<!-- page 30 -->
SEVAL
Shizhan Gong, Cheng Chen, Yuqi Gong, Nga Yan Chan, Wenao Ma, Calvin Hoi-Kwan Mak, Jill
Abrigo, and Qi Dou. Diffusion model based semi-supervised learning on brain hemorrhage images
for efficient midline shift quantification. In International Conference on Information Processing
in Medical Imaging, pages 69–81. Springer, 2023.
Yves Grandvalet and Yoshua Bengio. Semi-supervised learning by entropy minimization. Advances
in neural information processing systems, 17, 2004.
Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger.
On calibration of modern neural
networks. In International conference on machine learning, pages 1321–1330. PMLR, 2017.
Lan-Zhe Guo and Yu-Feng Li. Class-imbalanced semi-supervised learning with adaptive threshold-
ing. In International Conference on Machine Learning, pages 8082–8094. PMLR, 2022.
Ju He, Adam Kortylewski, Shaokang Yang, Shuai Liu, Cheng Yang, Changhu Wang, and Alan Yuille.
Rethinking re-sampling in imbalanced semi-supervised learning. arXiv preprint arXiv:2106.00209,
2021.
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recog-
nition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages
770–778, 2016.
Daniel Ho, Eric Liang, Xi Chen, Ion Stoica, and Pieter Abbeel. Population based augmentation: Effi-
cient learning of augmentation policy schedules. In International conference on machine learning,
pages 2731–2741. PMLR, 2019.
Ahmet Iscen, Giorgos Tolias, Yannis Avrithis, and Ondrej Chum. Label propagation for deep semi-
supervised learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern
recognition, pages 5070–5079, 2019.
Konstantinos Kamnitsas, Daniel Castro, Loic Le Folgoc, Ian Walker, Ryutaro Tanno, Daniel Rueck-
ert, Ben Glocker, Antonio Criminisi, and Aditya Nori. Semi-supervised learning via compact latent
space clustering. In International conference on machine learning, pages 2459–2468. PMLR, 2018.
Bingyi Kang, Saining Xie, Marcus Rohrbach, Zhicheng Yan, Albert Gordo, Jiashi Feng, and Yannis
Kalantidis. Decoupling representation and classifier for long-tailed recognition. arXiv preprint
arXiv:1910.09217, 2019.
Jaehyung Kim, Youngbum Hur, Sejun Park, Eunho Yang, Sung Ju Hwang, and Jinwoo Shin. Dis-
tribution aligning refinery of pseudo-label for imbalanced semi-supervised learning. Advances in
neural information processing systems, 33:14567–14579, 2020.
Alex Krizhevsky, Geoffrey Hinton, et al.
Learning multiple layers of features from tiny images.
Technical report, 2009.
Zhengfeng Lai, Chao Wang, Sen-ching Cheung, and Chen-Nee Chuah. Sar: Self-adaptive refine-
ment on pseudo labels for multiclass-imbalanced semi-supervised learning. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4091–4100, 2022a.
Zhengfeng Lai, Chao Wang, Henrry Gunawan, Sen-Ching S Cheung, and Chen-Nee Chuah.
Smoothed adaptive weighting for imbalanced semi-supervised learning: Improve reliability against
unknown distribution data. In International Conference on Machine Learning, pages 11828–11843.
PMLR, 2022b.
30


<!-- page 31 -->
SEVAL
Samuli Laine and Timo Aila. Temporal ensembling for semi-supervised learning. arXiv preprint
arXiv:1610.02242, 2016.
Justin Lazarow, Kihyuk Sohn, Chen-Yu Lee, Chun-Liang Li, Zizhao Zhang, and Tomas Pfister.
Unifying distribution alignment as a loss for imbalanced semi-supervised learning. In Proceedings
of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 5644–5653,
2023.
Dong-Hyun Lee et al. Pseudo-label: The simple and efficient semi-supervised learning method for
deep neural networks. In Workshop on challenges in representation learning, ICML, volume 3,
page 896. Atlanta, 2013.
Hyuck Lee, Seungjae Shin, and Heeyoung Kim.
Abc:
Auxiliary balanced classifier for class-
imbalanced semi-supervised learning. Advances in Neural Information Processing Systems, 34:
7082–7094, 2021.
Chongxuan Li, Taufik Xu, Jun Zhu, and Bo Zhang. Triple generative adversarial nets. Advances in
neural information processing systems, 30, 2017.
Muyang Li, Runze Wu, Haoyu Liu, Jun Yu, Xun Yang, Bo Han, and Tongliang Liu.
Instant:
Semi-supervised learning with instance-dependent thresholds. Advances in Neural Information
Processing Systems, 36, 2024.
Zeju Li, Konstantinos Kamnitsas, and Ben Glocker. Analyzing overfitting under class imbalance in
neural networks for image segmentation. IEEE transactions on medical imaging, 40(3):1065–1077,
2020.
Zeju Li, Konstantinos Kamnitsas, Mobarakol Islam, Chen Chen, and Ben Glocker.
Estimating
model performance under domain shifts with class-specific confidence scores. In International
Conference on Medical Image Computing and Computer-Assisted Intervention, pages 693–703.
Springer, 2022.
Zachary Lipton, Yu-Xiang Wang, and Alexander Smola. Detecting and correcting for label shift
with black box predictors. In International conference on machine learning, pages 3122–3130.
PMLR, 2018.
Tongliang Liu and Dacheng Tao. Classification with noisy labels by importance reweighting. IEEE
Transactions on pattern analysis and machine intelligence, 38(3):447–461, 2015.
Ziwei Liu, Zhongqi Miao, Xiaohang Zhan, Jiayun Wang, Boqing Gong, and Stella X Yu. Large-scale
long-tailed recognition in an open world. In Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, pages 2537–2546, 2019.
Charlotte Loh, Rumen Dangovski, Shivchander Sudalairaj, Seungwook Han, Ligong Han, Leonid
Karlinsky, Marin Soljacic, and Akash Srivastava.
On the importance of calibration in semi-
supervised learning. arXiv preprint arXiv:2210.04783, 2022.
Aditya Krishna Menon, Sadeep Jayasumana, Ankit Singh Rawat, Himanshu Jain, Andreas Veit, and
Sanjiv Kumar. Long-tail learning via logit adjustment. arXiv preprint arXiv:2007.07314, 2020.
Alexander Mey and Marco Loog. Improved generalization in semi-supervised learning: A survey
of theoretical results. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(4):
4747–4767, 2022.
31


<!-- page 32 -->
SEVAL
Takeru Miyato, Shin-ichi Maeda, Masanori Koyama, and Shin Ishii. Virtual adversarial training: a
regularization method for supervised and semi-supervised learning. IEEE transactions on pattern
analysis and machine intelligence, 41(8):1979–1993, 2018.
Mehryar Mohri, Afshin Rostamizadeh, and Ameet Talwalkar. Foundations of machine learning.
MIT press, 2018.
Nagarajan Natarajan, Inderjit S Dhillon, Pradeep K Ravikumar, and Ambuj Tewari. Learning with
noisy labels. Advances in neural information processing systems, 26, 2013.
Youngtaek Oh, Dong-Jin Kim, and In So Kweon.
Daso: Distribution-aware semantics-oriented
pseudo-label for imbalanced semi-supervised learning. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition, pages 9786–9796, 2022.
Mamshad Nayeem Rizve, Kevin Duarte, Yogesh S Rawat, and Mubarak Shah. In defense of pseudo-
labeling: An uncertainty-aware pseudo-label selection framework for semi-supervised learning.
arXiv preprint arXiv:2101.06329, 2021.
Marco Saerens, Patrice Latinne, and Christine Decaestecker. Adjusting the outputs of a classifier to
new a priori probabilities: a simple procedure. Neural computation, 14(1):21–41, 2002.
Henry Scudder. Probability of error of some adaptive pattern-recognition machines. IEEE Trans-
actions on Information Theory, 11(3):363–371, 1965.
Kihyuk Sohn, David Berthelot, Nicholas Carlini, Zizhao Zhang, Han Zhang, Colin A Raffel, Ekin Do-
gus Cubuk, Alexey Kurakin, and Chun-Liang Li. Fixmatch: Simplifying semi-supervised learning
with consistency and confidence. Advances in neural information processing systems, 33:596–608,
2020.
Jong-Chyi Su and Subhransu Maji. The semi-supervised inaturalist-aves challenge at fgvc7 workshop,
2021.
Jong-Chyi Su, Zezhou Cheng, and Subhransu Maji. A realistic evaluation of semi-supervised learning
for fine-grained classification. In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 12966–12975, 2021.
Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged con-
sistency targets improve semi-supervised deep learning results. Advances in neural information
processing systems, 30, 2017.
Junjiao Tian, Yen-Cheng Liu, Nathaniel Glaser, Yen-Chang Hsu, and Zsolt Kira.
Posterior re-
calibration for imbalanced datasets.
Advances in Neural Information Processing Systems, 33:
8101–8113, 2020.
Jesper E Van Engelen and Holger H Hoos. A survey on semi-supervised learning. Machine learning,
109(2):373–440, 2020.
Slobodan Vucetic and Zoran Obradovic. Classification on data with biased class distribution. In
Machine Learning: ECML 2001: 12th European Conference on Machine Learning Freiburg, Ger-
many, September 5–7, 2001 Proceedings 12, pages 527–538. Springer, 2001.
Renzhen Wang, Xixi Jia, Quanziang Wang, Yichen Wu, and Deyu Meng.
Imbalanced semi-
supervised learning with bias adaptive classifier. arXiv preprint arXiv:2207.13856, 2022a.
32


<!-- page 33 -->
SEVAL
Yidong Wang, Hao Chen, Yue Fan, Wang Sun, Ran Tao, Wenxin Hou, Renjie Wang, Linyi Yang, Zhi
Zhou, Lan-Zhe Guo, et al. Usb: A unified semi-supervised learning benchmark for classification.
Advances in Neural Information Processing Systems, 35:3938–3961, 2022b.
Yidong Wang, Hao Chen, Qiang Heng, Wenxin Hou, Yue Fan, Zhen Wu, Jindong Wang, Marios
Savvides, Takahiro Shinozaki, Bhiksha Raj, et al.
Freematch: Self-adaptive thresholding for
semi-supervised learning. arXiv preprint arXiv:2205.07246, 2022c.
Chen Wei, Kihyuk Sohn, Clayton Mellina, Alan Yuille, and Fan Yang. Crest: A class-rebalancing
self-training framework for imbalanced semi-supervised learning. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, pages 10857–10866, 2021.
Tong Wei and Kai Gan. Towards realistic long-tailed semi-supervised learning: Consistency is all you
need. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
pages 3469–3478, 2023.
Qizhe Xie, Zihang Dai, Eduard Hovy, Thang Luong, and Quoc Le. Unsupervised data augmentation
for consistency training. Advances in neural information processing systems, 33:6256–6268, 2020.
Yi Xu, Lei Shang, Jinxing Ye, Qi Qian, Yu-Feng Li, Baigui Sun, Hao Li, and Rong Jin. Dash:
Semi-supervised learning with dynamic thresholding. In International Conference on Machine
Learning, pages 11525–11536. PMLR, 2021.
Zhuoran Yu, Yin Li, and Yong Jae Lee. Inpl: Pseudo-labeling the inliers first for imbalanced semi-
supervised learning. arXiv preprint arXiv:2303.07269, 2023.
Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. arXiv preprint arXiv:1605.07146,
2016.
Bowen Zhang, Yidong Wang, Wenxin Hou, Hao Wu, Jindong Wang, Manabu Okumura, and
Takahiro Shinozaki. Flexmatch: Boosting semi-supervised learning with curriculum pseudo label-
ing. Advances in Neural Information Processing Systems, 34:18408–18419, 2021.
Dengyong Zhou, Olivier Bousquet, Thomas Lal, Jason Weston, and Bernhard Sch¨olkopf. Learning
with local and global consistency. Advances in neural information processing systems, 16, 2003.
Zhi-Hua Zhou and Xu-Ying Liu. Training cost-sensitive neural networks with methods addressing
the class imbalance problem. IEEE Transactions on knowledge and data engineering, 18(1):63–77,
2005.
Barret Zoph and Quoc V Le. Neural architecture search with reinforcement learning. arXiv preprint
arXiv:1611.01578, 2016.
33