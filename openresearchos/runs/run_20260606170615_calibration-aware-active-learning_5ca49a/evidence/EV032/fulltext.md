<!-- page 1 -->
Information-Theoretic Generalization Analysis for
Expected Calibration Error
Futoshi Futami∗
Osaka University / RIKEN AIP
futami.futoshi.es@osaka-u.ac.jp
Masahiro Fujisawa∗†
RIKEN AIP
masahiro.fujisawa@riken.jp
Abstract
While the expected calibration error (ECE), which employs binning, is widely
adopted to evaluate the calibration performance of machine learning models, the-
oretical understanding of its estimation bias is limited. In this paper, we present
the first comprehensive analysis of the estimation bias in the two common binning
strategies, uniform mass and uniform width binning. Our analysis establishes up-
per bounds on the bias, achieving an improved convergence rate. Moreover, our
bounds reveal, for the first time, the optimal number of bins to minimize the esti-
mation bias. We further extend our bias analysis to generalization error analysis
based on the information-theoretic approach, deriving upper bounds that enable
the numerical evaluation of how small the ECE is for unknown data. Experiments
using deep learning models show that our bounds are nonvacuous thanks to this
information-theoretic generalization analysis approach.
1
Introduction
Ensuring reliable predictions from machine learning models holds paramount importance in risk-
sensitive applications such as medical diagnosis [18]. To achieve this, it is essential not only to
evaluate the accuracy of the point predictions of models but also to precisely quantify the associated
uncertainty. One effective approach to accomplishing this is to leverage the concept of calibration.
In the classification context, the calibration performance is evaluated by how well predictive proba-
bilities of a model align with the actual frequency of true labels, and a close correspondence between
them indicates that the model is well calibrated [7, 42]. Unfortunately, machine learning models are
often not well calibrated [11, 23], prompting extensive research on their calibration performance
both theoretically and numerically. In this paper, we assume a binary classification problem.
To evaluate the calibration performance, we often use the calibration error or true calibration error
(TCE) [12, 31, 10]. This evaluates the disparity between the predicted probability of a model and the
conditional expectation of the label given the model prediction, instead of the true label frequency.
However, analytically computing the TCE is challenging, primarily due to the intractability of the
conditional expectation. One way to address this issue is by using the binning method [49]. This
method enables the estimation of conditional expectations by dividing the probability range [0, 1]
into B small intervals called bins and comparing the empirical mean of predictive probabilities
and label frequencies within each bin, utilizing the finite test dataset denoted as Ste. The obtained
estimator of the TCE is termed the (binned) expected calibration error (ECE).
Given that the ECE estimates the TCE, it is crucial to theoretically explore the bias between them,
termed total bias in this paper, to confirm the accuracy of calibration evaluation using the ECE. Fur-
thermore, it is vital to identify the conditions under which our training algorithm achieves a low ECE
or TCE for unknown test datasets. This can be paraphrased as the importance of conducting gen-
eralization error analysis under the ECE and TCE. Nevertheless, research on these aspects remains
∗Equal contribution.
†Corresponding author.
38th Conference on Neural Information Processing Systems (NeurIPS 2024).
arXiv:2405.15709v2  [cs.LG]  26 May 2025


<!-- page 2 -->
scant. Existing studies have only shown that the ECE underestimates the TCE [23] and have only
analyzed the bias caused by a finite sample under specific conditions such as using uniform-mass
binning (UMB) [13, 12]. Consequently, there remains a significant gap in the comprehensive the-
oretical understanding of the biases introduced by binning (termed binning bias) and the statistical
bias resulting from the use of finite test data samples. While these studies have concentrated on sce-
narios utilizing UMB, there has been no corresponding analysis for uniform-width binning (UWB),
which is also frequently employed in practice. This limitation could be due to the challenges posed
by UWB, where the equal partitioning of probability intervals can lead to bins without any samples,
making bias analysis difficult. Unfortunately, to the best of our knowledge, there are also no existing
generalization analyses on the basis of the ECE and TCE.
To address the challenges outlined above, in this paper, we comprehensively analyze the ECE for
both UWB and UMB. We derive the upper bounds of the total bias of the ECE using a newly derived
concentration inequality (Corollary 1). Our bounds improve the order of convergence regarding the
bin size. Furthermore, the optimal bin size that minimizes the total bias is successfully derived from
these results. With this optimal bin size, the total bias of the ECE exhibits a rate of O(1/n1/3
te ) for
both UWB and UMB, where nte is the number of test samples (Eq. (13)).
This bias analysis leads to our second novel contribution, providing the generalization error analysis
for the ECE and TCE (Theorems 4 and 5) using the information-theoretic (IT) analysis [44, 15, 17].
Directly applying the existing IT analysis is, however, challenging because the ECE on the training
dataset is no longer represented by the sum of independent and identically distributed (i.i.d.) random
variables w.r.t. the trained model. We circumvent this problem by applying a novel exponential
moment inequality derived in the process of our bias analysis described above. We further connect
our results to classical uniform convergence theory using the metric entropy of function classes,
which allows us to discuss the convergence rate of our bounds under a broader range of models
(Theorem 6). Using our generalization bounds, we theoretically explore the existing conjecture [11,
23] that recalibration with the reuse of training data leads to severe overfitting. We then show that
our analysis successfully characterizes such an additional bias (Theorem 7). Numerical experiments
using deep learning models confirm that our bound is nonvacuous and validate our findings.
2
Preliminaries
For a random variable denoted in capital letters, we express its realization with corresponding lower-
case letters. Let PX denote the distribution of X, and let PY |X represent the conditional distribution
of Y given X. We express the expectation of a random variable X as EX. The symbol I(X; Y )
represents the mutual information (MI) between X and Y , while I(X; Y |Z) is the conditional MI
(CMI) between X and Y given Z. We further define [n] = {1, . . . , n} for n ∈N.
We consider binary classification in this paper. Let Z = X × Y be the domain of data, where
X and Y = {0, 1} are input and label spaces, respectively. Suppose D represents an unknown
data distribution, and let Str := {Zm}n
m=1 denote the training dataset consisting of n samples
drawn i.i.d. from D. We also define the test dataset comprising nte samples as Ste ∼Dnte. Let
fw : X →[0, 1] be a parametric probabilistic classifier that outputs a prediction of the probability
Y = 1, and we denote its parameters as w ∈W ⊂Rd. We consider a randomized algorithm
A : Zn × R →W, where R ∈R is the randomness of an algorithm, independent of all other
random variables. For fixed R = r and Str = s, A(s, r) is a deterministic function and fA(s,r)(x)
is the prediction at point x given s and r. We evaluate the accuracy of the trained predictor fw
using the loss function l : W × Z →[0, 1], where l(A(s, r), z) denotes the loss incurred by the
prediction fw(x) for label y. For example, the zero-one loss is commonly used to evaluate the
accuracy. Then, the training loss is given by ˆLStr := 1
n
Pn
m=1 l(A(Str, R), Zm) and the test loss
is given as LZ := l(A(Str, R), Z) where Z ∼D. We also define the expected version of them as
LS := EStr,R ˆLStr and LD := EStr,Z,RLZ.
2.1
Calibration error and its estimator
In this section, we introduce a calibration metric and its corresponding estimator. The most widely
known metric is the true calibration error (TCE) [31, 10, 12] defined as
TCE(fw) := E [|E[Y |fw(X)] −fw(X)|] ,
(1)
2


<!-- page 3 -->
conditioned on W
= w.
Unfortunately, evaluating the TCE directly is challenging due to
the intractable calculation of E[Y |fw(X)].
To avoid this issue, we often use the binning
method [11, 49, 48]. This method estimates the TCE by partitioning the prediction probability range
[0, 1] into B intervals I = {Ii}B
i=1 (called bins) and averaging within each bin using the evaluation
dataset Se := {Zm}ne
m=1 ∈Zne, where we assume ne ≥2B. For instance, we have Se = Ste when
the test dataset is used for evaluation. The TCE estimator on the basis of I, with Se, is defined as
ECE(fw, Se) :=
B
X
i=1
pi| ¯fi,Se −¯yi,Se|,
(2)
where |Ii| := Pne
m=1 1fw(xm)∈Ii, pi := |Ii|
ne , ¯fi,Se :=
1
|Ii|
Pne
m=1 1fw(xm)∈Iifw(xm), and ¯yi,Se :=
1
|Ii|
Pne
m=1 1fw(xm)∈Iiym. This estimator is called the expected calibration error (ECE) 3.
There are two common methods to construct I. One is uniform width binning (UWB) [11], which
divides the [0, 1] interval into equal widths as follows: Ii = ((i −1)/B, i/B] for i in [B]. The
other approach is uniform mass binning (UMB) [49], which sets I so that each bin contains an equal
number of samples. That is, we calculate predicted probabilities as fm = fw(xm) for xm ∈Se, let
f(m) be the m-th order statistics of (f1, . . . , fne), and then set I1 = (0, u1], I2 = (u1, u2], . . . , IB =
(uB−1, uB] for b ∈[B−1] and ub := f(⌊neb/B⌋) with uB = 1. Here, ⌊x⌋:= max{m ∈Z : m ≤x}.
2.2
Biases of ECE and limitation of existing work
Given that the ECE is an estimator of the TCE, it is of practical importance to understand the nature
of the bias defined as |TCE(fw) −ECE(fw, Se)|. We call this bias as the total bias. To facilitate
the total bias analysis, we adopt the following definition of the binned function of fw [23]:
fI(x) :=
B
X
i=1
E[fw(X)|fw(X) ∈Ii] · 1fw(x)∈Ii,
(3)
which represents the conditional expectation within each bin.
When we evaluate the ECE on
Se = Ste, we expect that ECE(fw, Ste) will converge to TCE(fI) = E|E[Y |fI(x)] −fI(x)|
with increasing nte. However, TCE(fI) underestimates TCE(fw) [23, 10], that is,
TCE(fI) ≤TCE(fw),
(4)
which implies that ECE(fw) is a biased estimator of TCE(fw). Therefore, comprehending the
extent of this bias is crucial to an accurate calibration performance evaluation. Nevertheless, pre-
vious studies [23, 13, 12] have exclusively focused on the statistical bias in UMB, defined as
|TCE(fI) −ECE(fw, Ste)| as discussed in Section 1. This brings us to an analysis of the total
bias for both UWB and UMB.
2.3
Information-theoretic generalization error analysis
We now briefly outline the IT analysis using the evaluated CMI (eCMI) [34] that we utilize in our
study. Consider ˜Z ∈Zn×2 as an n × 2 matrix, where each entry is drawn i.i.d. from D. We
refer to this matrix as a supersample. Each column of ˜Z has indexes {0, 1} associated with U =
(U1, . . . , Un) ∼Uniform({0, 1}n) independent of ˜Z. We denote the m-th row as ˜Zm with entries
( ˜Zm,0, ˜Zm,1). In this setting, we consider ˜ZU := ( ˜Zm,Um)n
m=1 as the training dataset and ˜Z ¯U :=
( ˜Zm, ¯Um)n
m=1 as the test dataset with nte = n, where ¯Um = 1−Um. With these notations, we can see
that ˆL ˜
Z := 1
n
Pn
m=1 l(A( ˜ZU, R), ˜Zm,Um) corresponds to the training error since LS = E ˜
Z,R,U ˆL ˜
Z.
Also, L ˜
Z := 1
n
Pn
m=1 l(A( ˜
ZU, R), ˜Zm, ¯Um) corresponds to the test error, LD = E ˜
Z,R,UL ˜
Z. The
described settings, called the CMI setting [17], lead to the following theorem.
Theorem 1 (Theorem 6.7 in Steinke and Zakynthinou [34]). Under the CMI setting, we have
E ˜
Z,R,U|ˆL ˜
Z −L ˜
Z| ≤
r
2
n(eCMI(l) + log 2),
(5)
3 Although some existing studies refer to Eq. (1) as the ECE, in this study, we follow the definitions of the
TCE and ECE outlined by Roelofs et al. [31] and Gruber and Buettner [10] to make a clear distinction from the
estimator obtained through binning.
3


<!-- page 4 -->
where eCMI(l) := I(l(A( ˜ZU, R), ˜Z); U| ˜Z) and l(A( ˜ZU, R), ˜Z) is an n × 2 loss matrix obtained
by applying l(A( ˜ZU, R), ·) elementwise to ˜Z.
The reason we focus on IT analysis is that it enables algorithm-dependent analysis. The conventional
uniform convergence theory [37] focuses solely on function classes to derive bounds. However,
recent findings suggest that models trained by some algorithms are not well calibrated but show
high accuracy [11, 23]. Therefore, it seems essential to incorporate information about not only
the function class but also the algorithm in the ECE analysis. Hence, in this paper, we adopt the
eCMI framework, which is the generalized analysis approach that maximizes the use of algorithmic
information. Furthermore, because eCMI-based bounds can be estimated using training and test
data, the generalization performance of the model can be evaluated numerically, making it desirable
from a practical standpoint.
3
Proposed analysis of total bias in binned ECE
Here, we present our first main analyses of the bias analysis of the ECE as the estimator of the TCE.
Our analysis primarily focuses on the total bias defined as follows:
Biastot(fw, Ste) := |TCE(fw) −ECE(fw, Ste)|.
(6)
We can derive the following upper bound of Eq. (6) by using the triangle inequality,
Biastot(fw, Ste) ≤Biasbin(fw, fI) + Biasstat(fw, Ste),
(7)
where Biasbin(fw, fI)
:=
|TCE(fw) −TCE(fI)| and Biasstat(fw, Ste)
:=
|TCE(fI) −
ECE(fw, Ste)|. We call the former as the binning bias, which arises from nonparametric estimation
via binning, and the latter as the statistical bias caused by estimation on finite data points.
Before showing our results, we introduce the following assumption that is also used by Gupta and
Ramdas [12] and Sun et al. [35]:
Assumption 1. Given W = w, fw(x) is absolutely continuous w.r.t. the Lebesgue measure.
This assumption means that fw(x) has a probability density, and it is satisfied without loss of gen-
erality as elaborated in Appendix C in Gupta and Ramdas [12].
From Eq. (7), we can obtain an upper bound on the total bias by analyzing the binning and statistical
biases separately. First, we present the following results of our statistical bias analysis:
Theorem 2 (Statistical bias analysis). Given W = w, under Assumption 1, we have
TCE(fI) ≤ESteECE(fw, Ste),
(8)
ESteBiasstat(fw, Ste) ≤



q
2B log 2
nte
(for UWB),
q
2B log 2
nte−B +
2B
nte−B
(for UMB).
(9)
Proof sketch. First, we reformulate the ECE as ECE(fw, Ste) = PB
i=1 |E(X,Y )∼ˆSte(Y −fw(X)) ·
1fw(X)∈Ii| and the TCE as TCE(fI) = PB
i=1 |E(X,Y )∼D(Y −fw(X)) · 1fw(X)∈Ii|, where ˆSte is
the empirical distribution of Ste. The proof of this reformulation is shown in Appendix C.1. Thanks
to these transformations, our analysis does not have the problem that the UWB method can lead
to bins without any samples. By evaluating the exponential moment for UWB using McDiarmid’s
inequality under these reformulation, we have, for any λ ≥0,
λESteBiasstat(fw, Ste) ≤log ESteeλ|TCE(fI)−ECE(fw,Ste)| ≤B log 2 + λ2/(2nte).
(10)
Using this, we can derive both the bias and the high probability bound. We can derive a similar
bound for UMB. The complete proof is provided in Appendix D.1.
Eq. (8) shows that ECE(fw, Ste) overestimates TCE(fI) in expectation. Combined with Eq. (4),
we can see that ECE(fW , Ste) cannot be the upper or lower bound of TCE(fw) in expectation.
This emphasizes the importance of the rigorous bias analysis of |TCE(fw) −ECE(fw, Ste)|.
4


<!-- page 5 -->
Comparison with existing work: Eq. (9) provides better generality and a tighter bound than prior
results. Our bound exhibits O(
p
B/nte) in expectation (and Op(
p
B/nte) in high probability
w.r.t. Ste proved in Appendix D.3.). In contrast, the existing analysis [13, 12, 23] provided a similar
bound focused on UMB scale as O(B/√nte) in expectation (and Op(
p
B log B/nte) in high prob-
ability). In terms of generality, our derivation techniques can be applied to both UMB and UWB,
whereas existing bounds are limited to UMB.
Pros of our proof technique: The proof procedure in existing work [13, 12, 23] involves (i) showing
that the samples assigned to each bin are i.i.d., (ii) applying the Hoeffding inequality to derive
concentration bounds separately for each bin, and (iii) summing up these error bounds across all
bins. This approach results in slow convergence and only applicable to UMB. On the other hand,
our approach simultaneously handles all bins by utilizing the concentration inequality in Eq. (10)
and provides the improved upper bound and can be used for both UWB and UMB. We offer a more
detailed explanation of this in Appendix D.6.
Next, we show the results of our binning bias analysis under the following common assumption in
the nonparametric estimation context [36].
Assumption 2. Given W = w, E[Y |fw(x)] satisfies L-Lipschitz continuity.
Theorem 3 (Binning bias analysis). Given W = w, under Assumptions 1 and 2, we have
ESteBiasbin(fw, fI) ≤
( 1+L
B
(for UWB),
(1 + L)( 1
B +
q
2B log 2
nte−B +
2B
nte−B )
(for UMB).
(11)
Proof sketch. In Appendix D.4, we show that
Biasbin(fw, fI) ≤E|E[Y |fw(X)] −E[Y |fI(X)]| + E|fw(X) −fI(X)|.
We then derive the upper bound of the right-hand side from the definitions of bins in Section 2.1. For
UWB, the upper bound is O(1/B) because UWB divides the interval into equal widths. For UMB,
we need to evaluate how samples are split by bins. The complete proof is in Appendix D.4.
Substituting the results from Theorems 2 and 3 into Eq. (7) yields the following upper bound for the
total bias.
Corollary 1. Given W = w, under Assumptions 1 and 2, we have
ESteBiastot(fw, Ste) ≤



1+L
B
+
q
2B log 2
nte
(for UWB),
1+L
B
+ (2 + L)(
q
2B log 2
nte−B +
2B
nte−B )
(for UMB).
(12)
The above result evidently implies a trade-off concerning B. Intuitively, this indicates that while
a larger number of bins, B, improves the precision of fw estimation, accurately estimating the
conditional expectation requires a greater sample size. We further determine the optimal number of
bins by minimizing the upper bound of Eq. (12) w.r.t. B, which results in B = O(n1/3
te ) and gives
ESteBiastot(fW , Ste) = O(1/n1/3
te ).
(13)
Since the bin size has been tuned heuristically in practice, this result sheds light on how to choose it
theoretically for both UMB and UWB rigorously.
Regarding tightness of Eq. (12): As we mentioned in Section 2.1, we use binning methods to
estimate intractable E[Y |fw(x)] in the TCE evaluation. Thus, the TCE evaluation can be viewed
as nonparametric estimation of a one-dimensional function on [0, 1]. According to Tsybakov [36],
the error in such nonparametric regression cannot be smaller than O(1/n1/3
te ) under Assumption 2.
Our bound is convincing because its order aligns with that in Tsybakov [36]. We provide a detailed
discussion in Appendix F.6 and F.7.
We finally remark that the total bias of binning using UWB and UMB cannot be improved even
assuming the H¨older continuity for E[Y |fw(x)] instead of Assumption 2. This is because the binning
bias includes the error term E|fw(X)−fI(X)| = O(1/B) even under the H¨older continuity. Thus,
we suffer from the slow converge ESteBiastot(fw, Ste) = O(1/n1/3
te ) under the optimal bin size
5


<!-- page 6 -->
B = O(n1/3
te ) (see Appendix D.5 for this proof). According to Tsybakov [36], the lower bound of
the nonparametric estimation is O(1/nβ/(2β+1)
te
) under β-H¨older continuity. This implies that the
binning method cannot leverage the underlying smoothness of the data distribution. Thus, the slow
convergence is the fundamental limitation of the binning scheme for both UMB and UWB.
4
Generalization error analysis in calibration error
Another goal of our study is to identify the conditions under which a training algorithm achieves a
low ECE or TCE on unknown data by analyzing the generalization error, which has been overlooked
in previous studies. In this section, we present our theoretical analysis regarding these points.
4.1
Information-theoretic analysis of generalization error in ECE and TCE
The expected generalization error between the ECE and TCE can be defined through the total bias
notion, that is, ER,StrBiastot(fW , Str) := ER,Str|TCE(fW ) −ECE(fW , Str)|. In this section, we
derive the upper bound of ER,StrBiastot(fW , Str) by analyzing the statistical and binning biases in
the same manner as in Section 3. First, we derive the following upper bound of the statistical bias,
Biasstat(fw, Str) := |TCE(fI) −ECE(fw, Str)|, using a similar proof technique as in Theorem 2.
Theorem 4 (Generalization error bound of the ECE). Under the CMI setting and under Assump-
tion 1, for both UWB and UMB, we have
ER,StrBiasstat(fW , Str) ≤ER,Str,Ste|ECE(fW , Ste) −ECE(fW , Str)| ≤
s
8(eCMI(˜l) + B log 2)
n
,
(14)
where eCMI(˜l) = I(˜l; U| ˜Z) and
˜l(U, R, ˜Z) := |ECE(fA( ˜
ZU,R), ˜Z ¯U)−ECE(fA( ˜
ZU,R), ˜ZU)|.
(15)
Proof sketch. We reformulate the ECE similarly to the proof outline in Theorem 2. Errors between
the losses evaluated on the training and test data are similar to the left-hand side of Eq. (5); however,
directly applying Eq. (5) leads to a suboptimal rate of O(B/√n). Therefore, we derive the extended
version of Eq. (5) by correlating B bins according to Eq. (10) and combining this with the Donsker–
Varadhan lemma. The complete proof can be found in Appendix E.1.
Comparing Eq. (14) with Eq. (9) in Theorem 2, we find that eCMI measures the additional bias
that arises when evaluating the ECE using training data that are dependent on the trained model fw
instead of test data, which are independent of it. In other words, the term ER,Str,Ste|ECE(fW , Ste)−
ECE(fW , Str)| can be regarded as the expected generalization error of the ECE, and eCMI is the
dominant term of the generalization gap. Therefore, if the trained model has a sufficiently low eCMI,
it achieves good generalization performance in terms of the ECE. The behavior of eCMI clearly
affects the convergence rate of this bound, which is discussed in Section 4.2. Moreover, our bound
and eCMI are numerically evaluable and we confirm that our bound is numerically nonvacuous (see
Section 6). We also show the application of our bound in the setting of recalibration in Section 4.3.
In Appendix E, we derive the binning bias under the training data similar to Theorem 3. By com-
bining this result with Theorem 4, we obtain the following generalization error bound for the TCE.
Theorem 5 (Generalization error bound of the TCE). Under the CMI setting and under Assump-
tions 1 and 2, we have
ER,StrBiastot(fW , Str)≤



1+L
B
+
q
8(eCMI(˜l)+B log 2)
n
(for UWB),
1+L
B +
q
8(eCMI(˜l)+B log 2)
n
+(1+L)
q
2(fCMI+B log 2)
n
(for UMB).
(16)
In the above, eCMI(˜l) is defined as Eq. (15) and
fCMI := I(fA( ˜
ZU,R)( ˜X); U| ˜Z),
(17)
where ˜x denotes the n × 2 matrix obtained by projecting each element of ˜z, and fA( ˜
ZU,R)( ˜X) is the
n × 2 matrix calculated by the elementwise application of fA( ˜
ZU,R)(·) to ˜x.
6


<!-- page 7 -->
When comparing Eq. (12) with the above results, it is observed that an additional bias, eCMI (in-
cluding fCMI in UMB), derived from training data arises. This implies that the trained model
shows a low TCE when it sufficiently reduces these additional biases and achieves a small ECE.
From a practical viewpoint, this implies that our bound can potentially be used as a theoretical
guarantee for some recent training algorithms, which directly control the ECE under the training
dataset [24, 29, 38]. Our theory might guarantee the ECE under test dataset for them.
Another interesting implication from our bounds is that we can derive the optimal bin size to mini-
mize the upper bound in Theorem 5. If eCMI(˜l) and fCMI are sufficiently small compared with n,
for example, O(log n) (we discuss this in Section 4.2), then, the optimal bin size can be derived as
B = O(n1/3) by minimizing Eq. (16) w.r.t. B. Such an optimal B leads to
ER,StrBiastot(fw, Str) = O(log n/n1/3).
(18)
According to Eq. (13) and the above result, we can anticipate that ER,StrBiastot(fw, Str) is much
smaller than ESteBiastot(fW , Ste) because the number of training data is often much larger than
that of test data (n ≫nte). This implies that if the model generalizes well, evaluating the ECE
using the training dataset may better reduce the total bias than that using test dataset. Although
proposing such a new TCE estimation method is beyond the scope of this paper, this represents an
important direction for future research.
4.2
On the behavior of eCMI and the order of total bias on metric entropy
In this section, we analyze how additional biases, i.e., eCMI(˜l) and fCMI, behave. The initial ob-
servation is that the following relation holds [17]: eCMI(˜l) ≤fCMI ≤I(W; S). Furthermore, we
can see that I(W; S) = O(log n) under certain constrained conditions, such as the conditionally
i.i.d. setting when fw(x) is the underlying probability model for p(y|x; w) with W being com-
pact and holding appropriate smoothness assumptions [14, 43]. Furthermore, fCMI can be upper
bounded when the algorithms satisfy the various notions of stability [34]. For example, differential
private algorithms and stochastic gradient Langevin dynamics (SGLD) [41] algorithms are included
in this argument. A more detailed discussion can be found in Appendix F.4.
These arguments, however, hold true only for specific models and algorithms. Therefore, we extend
Theorem 5 by utilizing the concept of metric entropy to overcome this issue.
Theorem 6 (Metric entropy). Let F be the function class fw belongs to. Suppose that F with the
metric ∥· ∥∞has the metric entropy, log N(F, ∥· ∥∞, δ), with the parameter δ (> 0). That is, there
exists a set of functions Fδ := {f1, . . . , fN(F,∥·∥∞,δ)} that consists of δ-cover of F. Then, under
the CMI setting and under Assumptions 1 and 2, for any δ ∈(0, 1/B] and for UWB, we have
ER,StrBiastot(fW , Str) ≤1 + L
B
+ (2 + L)δ +
r
8B log 2N(F, ∥· ∥∞, δ/B)
n
.
(19)
See Appendix E.3 for the proof. Theorem 6 connects the IT-based bound to the uniform convergence
theory. With this result, we can discuss the optimal number of bins across a broad spectrum of mod-
els. For example, we can obtain N(F, ∥· ∥∞, δ) ≍
 L0
δ
d when fw is a d-dimensional parametric
function that is L0-Lipschitz continuous (L0 > 0) [37], leading to the following upper bound:
ER,StrBiastot(fW , Str) ≲3 + 2L
B
+
r
8dB log(2L0B2)
n
,
(20)
where we set δ = O(1/B). This bound is minimized when B = O(n1/3), resulting in a bias of
O(log n/n1/3), which is consitent with Eq. (18).
The drawback of the above bound is that it depends on the model’s dimensionality explicitly, making
them unsuitable for large models such as neural networks. To address this issue, we can use the
different combinatorial properties, such as the fat-shattering dimension. See Appendix E.3 for the
detail.
4.3
Generalized error analysis on recalibration and bias due to reuse of training data
As an application of our generalization error bound, we analyze recalibration using a post-hoc re-
calibration function, which is used when the trained model is not well calibrated. We focus on the
7


<!-- page 8 -->
recalibration using UMB with recalibration data [35, 12]. In this setting, we first split overall data
into the training, recalibration, and test datasets (see Appendix B for details of this splitting strat-
egy). After training fw using the training dataset, we construct the recalibrated function h using the
recalibration dataset Sre as
hI,Sre(x) :=
B
X
i=1
¯yi,Sre · 1fw(x)∈Ii,
(21)
where ¯yi,Sre is the empirical mean of {ym}nre
m=1 ∈Sre in the i-th bin defined as in Eq. (2) and nre
is the number of the recalibration dataset. In short, Eq. (21) provides an estimator of the conditional
expectation of Y given fw(x) by setting Se = Sre. Gupta and Ramdas [12] clarified that the
statistical bias of Eq. (21) is given by ESreTCE(hI,Sre) = E[|E[Y |hI,Sre(X)] −hI,Sre(X)]| =
Op(
p
B log B/|Sre|). Since we need to split the overall data into three datasets, this approach
could be sample-inefficient and could result in a very loose bound. Although reusing the training
dataset may solve this problem to some extent, it has been suggested that this method may cause
performance degradation due to overfitting [23, 12].
Our contribution here is quantifying the bias caused by overfitting due to the reuse of training data
by utilizing our generalization error analysis in Section 4.1 as follows.
Theorem 7 (Recalibration reusing the training dataset). Replacing Sre with Str in Eq. (21), under
the CMI setting and under Assumptions 1 and 2, we have
ER,StrTCE(hI,Str) = ER,StrE[|E[Y |hI,Str(X)] −hI,Str(X)] ≤2
r
2(fCMI + B log 2)
n
,
where fCMI is defined in Eq. (17).
The complete proof is provided in Appendix E.4. In the above, fCMI corresponds to the additional
bias caused by overfitting due to the reuse of Str. This indicates that reusing Str does not negatively
affect the order of the bias if fCMI is smaller than other terms, as discussed in Sections 4.1 and 4.2.
Since the size of Str is much larger than that of Sre, the recalibration function hI,Str may exhibit a
much smaller bias compared to hI,Sre. We investigate this possibility numerically in Section 6 by
using the tighter version of Theorem 7 provided in Appendix E.4 (Corollary 4).
5
Related work
We have presented the results of our analyses of the total bias in the ECE and the generalization error
for both the ECE and the TCE. Existing studies have primarily focused on the statistical bias, with
little attention given to the binning bias. Gupta et al. [13] and Gupta and Ramdas [12] examined the
statistical bias associated with UMB, but they did not address the binning bias as we did. In contrast,
Kumar et al. [23] studied the binning bias but did not specify how this bias depends on n and B.
Moreover, most analyses have concentrated on UMB and UWB has not been thoroughly analyzed.
As outlined in the proof of Theorem 2, our approach allows us to analyze UWB even in cases where
some bins do not have any data points by employing our reformulation and concentration inequality.
It is important to note that Roelofs et al. [31] studied the numerical behavior of the total bias in some
practical models, whereas we focus on the theoretical aspect of the total bias. Recently, Sun et al.
[35] have derived the optimal number of bins under the recalibration with UMB. Compared with
this, we derived the optimal number of bins for UMB and UWB without recalibration under a similar
Lipschitz assumption. This leads to the discussion of estimating the TCE from the nonparametric
estimation. An additional discussion is summarized in Appendix F.
We have extended the existing eCMI bound [34, 15, 17, 40], which is used for analyzing general-
ization performance in terms of prediction accuracy, to calibration analysis. In addition, whereas
existing eCMI bounds numerically evaluated eCMI and fCMI for discrete random variables such
as zero-one loss, our analysis is conducted on continuous random variables as shown in Eq. (15).
We show in the next section that our bounds are still nonvacuous even for the continuous random
variables. The IT analysis was also utilized by Russo and Zou [33] to study the bias caused by data
reuse. Our analysis can be seen as an extension of this approach to the ECE and recalibration.
8


<!-- page 9 -->
10
2
10
3
10
4
10
5
10
6
n
10
−2
10
−1
Bound & Gap (less calibrate)
UWB; B = ⌊(2n(1 + L)2/(log 2))1/3⌋
n^{-1/3}
TCE gap; B = ⌊(2n(1 + L)2/(log 2))1/3⌋
10
2
10
3
10
4
10
5
10
6
n
10
−2
10
−1
Bound & Gap (better calibrate)
10
2
10
3
10
4
10
5
10
6
n
10
1
10
2
B
n vs B (less calibrate)
10
2
10
3
10
4
10
5
10
6
n
10
1
10
2
B
n vs B (better calibrate)
n^{-1/3}
Optimal B
Experimental best
Figure 1: Behavior of the upper bound in Eq. (12) as n increases when UWB is used. The following
two terms: less calibrate and better calibrate refer to β = (0.5, −1.5) and β = (0.2, −1.9), respec-
tively, where the former setting produces a worse value of the TCE estimator.
75
1000
4000
n
0.0
0.2
0.4
0.6
MNIST (Adam)
UMB; B = 6
UMB; B = 9
UMB; B = 14
UMB; B = 22
UMB; B = 34
UMB; B = 52
UMB; B = ⌊n1/3⌋
ECE gap; B = ⌊n1/3⌋
75
1000
4000
n
0.0
0.5
1.0
1.5
2.0
MNIST (SGLD)
500
5000
20000
n
0.0
0.2
0.4
0.6
CIFAR-10 (SGD)
500
5000
20000
n
0.0
0.2
0.4
0.6
CIFAR-10 (SGLD)
Figure 2: Behavior of the upper bound in Eq. (14) for various B as n increases (mean ± std.). For
clarity, only the results using UMB are shown. The ECE gap is shown for B = ⌊n1/3⌋since the
change in B did not result in significant differences. We refer to Figure 5 in Appendix H.3 for a
detailed analysis of the relationship between (log-scaled) ECE gap values and bound values across
different bin settings.
6
Experiments
In this section, we present experimental results validating our bounds (Section 6.1) and the additional
bias arising from reusing the training dataset for recalibration (Section 6.2).
6.1
Verification of our bounds
In this section, we empirically validate our theoretical findings in Eq. (12), the nonvacuous nature
of our bounds in Eq. (14), and confirm the efficiency of the optimal number of bins as discussed in
Section 4.1.
Experiments on synthetic datasets:
We first conducted simple experiments on synthetic datasets
following Zhang et al. [50]. In this experiment, we assume the distribution of Y as P(Y = 1) =
P(Y = 0) =
1
2, and we adopt fw(x) = P(Y = 1|X = x) = 1/(1 + exp(−β0 −β1x)) as
the prediction model, where w = {β0, β1} are parameters. Under these settings, we can calculate
the closed-form of E[Y |fw(X)] in Eq. (1), which allows us to estimate the TCE through Monte
Carlo integration. Next, we empirically evaluated the TCE gap, which is the empirical estimator
of ESteBiastot(fw, Ste), by calculating the difference between the TCE estimator and the ECE
using UWB. Here, the optimal B that minimizes the upper bound of Eq. (14) is B = ⌊2n(1 +
L)2/(log 2)1/3⌋, where L is estimated to be the maximum value of the gradient of the closed-form
of E[Y |fw(X)]. We provide the details of the experimental settings in Appendix G.1.
Figure 1 shows the results. The two leftmost figures show that the TCE gap is O(n−1/3) when our
optimal B is used. The other two figures show that the number of bins achieving the smallest upper
bound is closest to the optimal B and its order is O(n−1/3), where the candidates for B are set
as {n−1/2, n−1/3, n−1/4, 2n−1/2, 2n−1/3, 2n−1/4, 3n−1/2, 3n−1/3, 3n−1/4}. These observations
show the validity of our theoretical findings through Corollary 1.
Experiments on image datasets:
We further conducted two binary classification tasks on
MNIST [25] using a convolutional neural network (CNN) and on CIFAR-10 [21] using ResNet.
These models were trained using SGD with momentum for ResNet, Adam for CNN, and SGLD for
both, following the strategy of Hellstr¨om and Durisi [17]. The details of our experimental settings
are summarized in Appendix G.2. We initially evaluated the sum of the right-hand side terms of
9


**[Table p9.1]**
| 1 |  |  |  |  |
| --- | --- | --- | --- | --- |
| 2 |  |  |  |  |
|  |  |  |  |  |


**[Table p9.2]**
| n^{-1/3} Optima Experim | l B ental best |  |  |
| --- | --- | --- | --- |
|  |  |  |  |
|  |  |  |  |

[CAPTION] Figure 1: Behavior of the upper bound in Eq. (12) as n increases when UWB is used. The following

[CAPTION] Figure 2: Behavior of the upper bound in Eq. (14) for various B as n increases (mean ± std.). For

[CAPTION] Figure 1 shows the results. The two leftmost figures show that the TCE gap is O(n−1/3) when our


<!-- page 10 -->
Table 1:
Comparison of our method with existing recalibration in terms of the ECE
gap and its upper bound in Theorem 7 (mean ± std.).
Lower values are better.
We
adopted B
=
⌊n1/3⌋.
The bound values for the existing recalibration method origi-
nate from Corollary 4 in Appendix E.5.
Here, we report the ECE gap as TCE because
ER,SreE[|E[Y |hI,Sre(X)]−hI,Sre(X)] = ER,Sre,SteECE(hI,Sre, Ste) (existing recalibration meth-
ods) ER,StrE[|E[Y |hI,Str(X)] −hI,Str(X)] = ER,Str,SteECE(hI,Str, Ste) (our recalibration
method) from the definition of recalibration.
Dataset
Optimizer
Methods
TCE
Bound value
MNIST (n = 4000)
Adam
Recalib.
.0085 ± .0016
.8475
Our recalib.
.0058 ± .0026
.1444 ± .0000
SGLD
Recalib.
.0101 ± .0025
.8475
Our recalib.
.0072 ± .0007
.1444 ± .0000
CIFAR-10 (n = 20000)
SGD
Recalib.
.0139 ± .0010
1.455
Our recalib.
.0197 ± .0044
.0865 ± .0000
SGLD
Recalib.
.0109 ± .0012
1.455
Our recalib.
.0089 ± .0006
.0865 ± .0000
Eq. (14) and the ECE estimated using the training dataset, aiming to ascertain whether the disparity
from the ECE estimated using the test dataset was adequately minimal. We call this disparity as
the ECE gap, which is the estimator of ER,Str,Ste[|ECE(fW , Ste) −ECE(fW , Str)|]. We show the
results obtained when using UMB in Figure 2. These results show that our bound value becomes
less than 1 with an appropriate setting of B. We also observed that the bound values decrease with
n, whereas these values sometimes become vacuous for small n when B is large. Adjusting B could
pose challenges; however, a notable trend towards acquiring relatively stable nonvacuous bounds
can be observed when adopting B = ⌊n1/3⌋, even though this is the optimal choice only at the
upper bound of TCE, as discussed in Theorems 5 and 6 in Sections 4.1 and 4.2. Similar results are
obtained when using UWB (see Figure 3 in Appendix H).
6.2
Confirming additional bias due to reusing training dataset in recalibration
In this section, we empirically confirm the efficiency of the method when using the complete training
dataset for recalibration, referred to here as the reusing method. Table 1 illustrates that the reusing
method reduces the statistical bias of the ECE more effectively than existing methods using inde-
pendently created recalibration datasets (nre = 100). We also compared the tighter version of our
bound in Theorem 7, Corollary 4, with the bound of the existing recalibration methods presented
in Corollary 5 of Appendix E.5. Moreover, our bound values are lower than those for the exist-
ing recalibration methods on the test dataset. These results suggest that reusing training data could
be beneficial if the trained model generalizes well and eCMI is sufficiently small, as discussed in
Section 4.3.
7
Conclusion and limitations
We provided the first comprehensive analysis of the bias associated with the ECE when using the
test and training datasets. This leads to the derivation of the optimal bin size to minimize the total
bias. Numerical experiments show that our upper bound of the bias is nonvacuous for deep learning
models thanks to the IT generalization error analysis. Despite rigorous analysis, our analysis still
has limitations. Firstly, we focus on the binary classification; thus, the extension of our analysis to
the multiclass classification setting is an important future direction. However, the application of our
analytical techniques to this setting seems not clear. Additionally, our analysis cannot be applied
to the higher-order TCE, in which we use the p-th norm in Eq. (1). These limitations should be
addressed in future work to develop a more principled understanding of uncertainty.
10

[CAPTION] Table 1:
Comparison of our method with existing recalibration in terms of the ECE


<!-- page 11 -->
Acknowledgments and Disclosure of Funding
We sincerely appreciate the anonymous reviewers for their insightful feedback. FF was supported by
JSPS KAKENHI Grant Number JP23K16948. FF was supported by JST, PRESTO Grant Number
JPMJPR22C8, Japan. MF was supported by RIKEN Special Postdoctoral Researcher Program. MF
was supported by JST, ACT-X Grant Number JPMJAX210K, Japan.
References
[1] Noga Alon, Shai Ben-David, Nicolo Cesa-Bianchi, and David Haussler. Scale-sensitive di-
mensions, uniform convergence, and learnability. Journal of the ACM (JACM), 44(4):615–631,
1997.
[2] Peter L Bartlett and Wolfgang Maass. Vapnik-chervonenkis dimension of neural nets. The
handbook of brain theory and neural networks, pages 1188–1192, 2003.
[3] St´ephane Boucheron, G´abor Lugosi, and Pascal Massart.
Concentration Inequalities:
A Nonasymptotic Theory of Independence.
Oxford University Press, 02 2013.
ISBN
9780199535255. doi: 10.1093/acprof:oso/9780199535255.001.0001. URL https://doi.
org/10.1093/acprof:oso/9780199535255.001.0001.
[4] A. M. Carrell, N. Mallinar, J. Lucas, and P. Nakkiran. The calibration generalization gap. arXiv
preprint arXiv:2210.01964, 2022.
[5] Bertrand S. Clarke and Andrew R. Barron. Jeffreys’ prior is asymptotically least favorable
under entropy risk. Journal of Statistical Planning and Inference, 41:37–60, 1994.
[6] T. M. Cover and J. A. Thomas. Elements of Information Theory. John Wiley & Sons, 2012.
[7] A. P. Dawid. The well-calibrated Bayesian. Journal of the American Statistical Association,
77(379):605–610, 1982. doi: 10.1080/01621459.1982.10477856.
[8] Tyler Farghly and Patrick Rebeschini. Time-independent generalization bounds for sgld in
non-convex settings. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wort-
man Vaughan, editors, Advances in Neural Information Processing Systems, volume 34, pages
19836–19846. Curran Associates, Inc., 2021.
[9] Futoshi Futami and Masahiro Fujisawa. Time-independent information-theoretic generaliza-
tion bounds for sgld. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine,
editors, Advances in Neural Information Processing Systems, volume 36, pages 8173–8185.
Curran Associates, Inc., 2023.
[10] Sebastian Gruber and Florian Buettner. Better uncertainty calibration via proper scores for
classification and beyond. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho,
and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages
8618–8632. Curran Associates, Inc., 2022.
[11] C. Guo, G. Pleiss, Y. Sun, and K. Q Weinberger. On calibration of modern neural networks. In
International conference on machine learning, pages 1321–1330, 2017.
[12] C. Gupta and A. Ramdas. Distribution-free calibration guarantees for histogram binning with-
out sample splitting. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th Inter-
national Conference on Machine Learning, volume 139 of Proceedings of Machine Learning
Research, pages 3942–3952. PMLR, 18–24 Jul 2021.
[13] Chirag Gupta, Aleksandr Podkopaev, and Aaditya Ramdas. Distribution-free binary classifi-
cation: prediction sets, confidence intervals and calibration. In H. Larochelle, M. Ranzato,
R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Sys-
tems, volume 33, pages 3711–3723. Curran Associates, Inc., 2020.
[14] H. Hafez-Kolahi, B. Moniri, S. Kasaei, and M. S. Baghshah. Rate-distortion analysis of min-
imum excess risk in Bayesian learning. In International Conference on Machine Learning,
pages 3998–4007, 2021.
11


<!-- page 12 -->
[15] Hrayr Harutyunyan, Maxim Raginsky, Greg Ver Steeg, and Aram Galstyan.
Information-
theoretic generalization bounds for black-box learning algorithms. In M. Ranzato, A. Beygelz-
imer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, Advances in Neural Informa-
tion Processing Systems, volume 34, pages 24670–24682. Curran Associates, Inc., 2021.
[16] David Haussler and Manfred Opper. Mutual information, metric entropy and cumulative rela-
tive entropy risk. The Annals of Statistics, 25(6):2451–2492, 1997.
[17] Fredrik Hellstr¨om and Giuseppe Durisi. A new family of generalization bounds using sample-
wise evaluated cmi. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh,
editors, Advances in Neural Information Processing Systems, volume 35, pages 10108–10121.
Curran Associates, Inc., 2022.
[18] Xiaoqian Jiang, Melanie Osl, Jihoon Kim, and Lucila Ohno-Machado. Calibrating predictive
model estimates to support personalized medicine. Journal of the American Medical Informat-
ics Association, 19(2):263–274, 2012.
[19] L. F. Kozachenko and N. N. Leonenko. Sample estimate of the entropy of a random vector.
Problemy Peredachi Informatsii, 23:9–16, 1987.
[20] A. Kraskov, H. St¨ogbauer, and P. Grassberger. Estimating mutual information. Physical Review
E, 69:066138, 2004.
[21] A. Krizhevsky. Learning multiple layers of features from tiny images. Technical report, Uni-
versity of Toronto, 2009.
[22] Bogdan Kulynych, Yao-Yuan Yang, Yaodong Yu, Jarosł aw Bł asiok, and Preetum Nakkiran.
What you see is what you get: Principled deep learning via distributional generalization. In
S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances
in Neural Information Processing Systems, volume 35, pages 2168–2183. Curran Associates,
Inc., 2022.
[23] Ananya Kumar, Percy S Liang, and Tengyu Ma. Verified uncertainty calibration. In H. Wallach,
H. Larochelle, A. Beygelzimer, F. d'Alch´e-Buc, E. Fox, and R. Garnett, editors, Advances in
Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.
[24] Aviral Kumar, Sunita Sarawagi, and Ujjwal Jain. Trainable calibration measures for neural net-
works from kernel mean embeddings. In Jennifer Dy and Andreas Krause, editors, Proceed-
ings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of
Machine Learning Research, pages 2805–2814. PMLR, 10–15 Jul 2018.
[25] Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel.
Backpropagation applied to handwritten zip code recognition. Neural Computation, 1(4):541–
551, 1989.
[26] Michael Li, Matey Neykov, and Sivaraman Balakrishnan. Minimax optimal conditional density
estimation under total variation smoothness. Electronic Journal of Statistics, 16(2):3937–3972,
2022.
[27] D. O. Loftsgaarden and C. P. Quesenberry. A Nonparametric Estimate of a Multivariate Density
Function. The Annals of Mathematical Statistics, 36(3):1049–1051, 1965.
[28] W. Mou, L. Wang, X. Zhai, and K. Zheng. Generalization bounds of SGLD for non-convex
learning: Two theoretical viewpoints. In Proceedings of the 31st Conference on Learning
Theory, volume 75, pages 605–638, 2018.
[29] Teodora Popordanoska, Raphael Sayer, and Matthew Blaschko. A consistent and differentiable
lp canonical calibration error estimator. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave,
K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35,
pages 7933–7946. Curran Associates, Inc., 2022.
[30] J. Rissanen. Universal coding, information, prediction, and estimation. IEEE Trans. Inf. Theor.,
30(4):629–636, sep 2006. ISSN 0018-9448.
12


<!-- page 13 -->
[31] Rebecca Roelofs, Nicholas Cain, Jonathon Shlens, and Michael C. Mozer. Mitigating bias in
calibration error estimation. In Proceedings of The 25th International Conference on Artificial
Intelligence and Statistics, pages 4036–4054, 2022.
[32] B. C. Ross. Mutual information between discrete and continuous data sets. PLOS ONE, 9(2):
1–101, 02 2014.
[33] D. Russo and J. Zou. Controlling bias in adaptive data analysis using information theory.
In Proceedings of the 19th International Conference on Artificial Intelligence and Statistics,
volume 51, pages 1232–1240, 2016.
[34] Thomas Steinke and Lydia Zakynthinou. Reasoning About Generalization via Conditional
Mutual Information. In Jacob Abernethy and Shivani Agarwal, editors, Proceedings of Thirty
Third Conference on Learning Theory, volume 125 of Proceedings of Machine Learning Re-
search, pages 3437–3452. PMLR, 09–12 Jul 2020.
[35] Zeyu Sun, Dogyoon Song, and Alfred Hero. Minimum-risk recalibration of classifiers. In
A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in
Neural Information Processing Systems, volume 36, pages 69505–69531. Curran Associates,
Inc., 2023.
[36] Alexandre B. Tsybakov. Introduction to Nonparametric Estimation. Springer Publishing Com-
pany, Incorporated, 1st edition, 2008. ISBN 0387790519.
[37] M. J. Wainwright. High-Dimensional Statistics: A Non-Asymptotic Viewpoint. Cambridge
Series in Statistical and Probabilistic Mathematics. Cambridge University Press, 2019.
[38] Deng-Bao Wang, Lei Feng, and Min-Ling Zhang. Rethinking calibration of deep neural net-
works: Do not be afraid of overconfidence. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S.
Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems,
volume 34, pages 11809–11820. Curran Associates, Inc., 2021.
[39] H. Wang, R. Gao, and F. P. Calmon. Generalization bounds for noisy iterative algorithms
using properties of additive noise channels. Journal of Machine Learning Research, 24(26):
1–43, 2023.
[40] Ziqiao Wang and Yongyi Mao. Tighter information-theoretic generalization bounds from su-
persamples. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan
Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on
Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 36111–
36137. PMLR, 23–29 Jul 2023.
[41] M. Welling and Y. W. Teh. Bayesian learning via stochastic gradient Langevin dynamics. In
Proceedings of the 28th International Conference on Machine Learning, pages 681–688, 2011.
[42] D. Widmann, F. Lindsten, and D. Zachariah. Calibration tests beyond classification. In Inter-
national Conference on Learning Representations, 2021. URL https://openreview.net/
forum?id=-bxf89v3Nx.
[43] A. Xu and M. Raginsky. Minimum excess risk in Bayesian learning. IEEE Transactions on
Information Theory, 68(12):7935–7955, 2022.
[44] Aolin Xu and Maxim Raginsky. Information-theoretic analysis of generalization capability
of learning algorithms.
In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus,
S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems,
volume 30. Curran Associates, Inc., 2017.
[45] Yuhong Yang. Minimax nonparametric classification. i. rates of convergence. IEEE Transac-
tions on Information Theory, 45(7):2271–2284, 1999.
[46] Yuhong Yang and Andrew Barron.
Information-theoretic determination of minimax rates
of convergence.
The Annals of Statistics, 27(5):1564 – 1599, 1999.
doi: 10.1214/aos/
1017939142. URL https://doi.org/10.1214/aos/1017939142.
13


<!-- page 14 -->
[47] Yannis G. Yatracos. A Lower Bound on the Error in Nonparametric Regression Type Problems.
The Annals of Statistics, 16(3):1180 – 1187, 1988.
[48] B. Zadrozny and C. Elkan. Transforming classifier scores into accurate multiclass probability
estimates. In Proceedings of the eighth ACM SIGKDD international conference on Knowledge
discovery and data mining, pages 694–699, 2002.
[49] Bianca Zadrozny and Charles Elkan. Obtaining calibrated probability estimates from decision
trees and naive Bayesian classifiers. In Proceedings of the Eighteenth International Conference
on Machine Learning, pages 609–616, 2001.
[50] J. Zhang, B. Kailkhura, and T. Y.-J. Han. Mix-n-match : Ensemble and compositional methods
for uncertainty calibration in deep learning. In Proceedings of the 37th International Confer-
ence on Machine Learning, volume 119, pages 11117–11128, 2020.
14


<!-- page 15 -->
A
Remarks about the order expressions
Let f, g : R →R. We say f(x) ≍g(x) when there exist positive constants a, b and n0 such that
∀n > n0, we have ag(n) ≤f(x) ≤bg(x) holds. Moreover f(n) ≲g(n) means that there exists a
positive constant c and n0 such that ∀n > n0, we have |f(n)| ≤cg(n) holds. This is equivalent to
f(n) = O(g(n)).
B
The detailed explanation of how the data is split and used in our bounds
Here we remark how the data is prepared and used in our analysis.
B.1
The binning ECE and its evaluation
In the standard supervised learning settings assume that Nall data is obtained i.i.d from data gener-
ating distribution D. We express this as Sall = {(xi, yi)}Nall
i=1. Then the dataset is divided to
Sall = Str ∪Ste,
Str ∩Ste = ϕ,
where Str is the training data which is n data points and Ste is the test data points which is nte data
points. Thus Nall = n + nte.
B.1.1
Evaluation of the ECE under the test dataset in Section 3
Here we discuss the evaluation of ECE(fw, Ste), which uses Ste to calculate the ECE in Section 3.
This is the most common common approach in practice. We remark that as for the UWB, since we
do not use Ste when preparing bins, those Ste are i.i.d. inside each bin.
As for UMB, the situation is a bit complicated. Since we use Ste to construct bins, it seems that Ste
are no more i.i.d inside each bin. Surprisingly, Gupta and Ramdas [12] have shown that the samples
allocated in each bin are i.i.d. under UMB method. So using the same Ste for constructing bins and
evaluation of the binning ECE does not result in a large bias.
In these ways, we can calculate ECE(fw, Ste). We also remark that the training samples Str are
only used to learn the parameter W.
B.1.2
Evaluation of the ECE under the training dataset in Section 4
ere we discuss the evaluation of ECE(fw, Str), which uses Str to calculate the ECE in Section 3.
Thus, we use the training dataset Str for learning W and calculating ECE.
We need to carefully consider how the data is used when considering the result of the UMB in The-
orem 4. This theorem provides us the generalization guarantee of the ECE between ECE(fw, Str)
and ECE(fw, Ste). As for ECE(fw, Str) using UMB, we first train fW with Str and construct bins
with Str and calculate the empirical mean of each bin with Str. On the other hand, when calculating
ECE(fw, Ste), we calculate the empirical mean of each bin with Ste and we use the same bins with
ECE(fw, Str), so bins are constructed using Str in Theorem 4. In this sense, Theorem 4 provides
us with the generalization gap, where we regard that the bins constructed with the training dataset
are regarded as part of our trained model in our theoretical analysis.
B.2
The recalibration in Section 4.3
When the recalibration is performed, we further split the test data into
Sall = Str ∪Sre ∪Ste,
any common part sets are empty,
where Str is the training datasets used for learning W, and Sre is the dataset used for the recali-
bration. The most widely used approach is the UMB-based recalibration. First, we construct bins
following the UMB approach using Sre. Then let us express Sre = {(xi, yi)}nre
i=1. The recalibrated
function
hI,Sre(x) =
B
X
i=1
ˆµi,Sre · 1fw(x)∈Ii,
ˆµi,Sre :=
Pnre
m=1 ym · 1fw(xm)∈Ii
Pnre
m=1 1fw(xm)∈Ii
.
(22)
15


<!-- page 16 -->
Then Ste is used for evaluating the ECE or test accuracy.
However, since preparing both Ste and Sre is sample inefficient, our idea is reusing training sample
Str with size n even for the recalibration. In our setting, we split the data
Sall = Str ∪Ste,
S ∩Ste = ϕ,
and we construct bins following UMB approach using Str and calculate recalibration function by
using Str
hI,Str(x) =
B
X
i=1
ˆµi,Str · 1fw(x)∈Ii,
ˆµi,Str :=
Pn
m=1 ym · 1f(xm)∈Ii
Pn
m=1 1f(xm)∈Ii
.
(23)
We then finally evaluate the ECE using Ste. So our approach is more sample-efficient.
C
Auxiliary lemma and facts
Here we introduce auxiliary lemma and facts, which we will use repeatedly in our proofs. In this
section, we express fw as f to simplify the notation.
C.1
Binning and bias
First, from the definition of ECE in Eq. (2), we can immediately reformulate it as
ECE(f, Se) :=
B
X
i=1
|E(X,Y )∼ˆSe(Y −f(X)) · 1fw(X)∈Ii|,
(24)
where ˆSe is the empirical distribution of Se.
Next we introduce the binned function of f as fI, which is the conditional expectation given bins:
fI(x) :=
B
X
i=1
fIi(x) · 1f(x)∈Ii =
B
X
i=1
E[f(X)|f(X) ∈Ii] · 1f(x)∈Ii,
(25)
fIi(x) = E[f(X)|f(x) ∈Ii].
(26)
The following relation holds:
Lemma 1. We define the test-binned risk as
ECEBin(f) :=
B
X
i=1
|E(X,Y )∼D(Y −f(X)) · 1f(X)∈Ii|,
(27)
then we have
ECEBin(f) = TCE(fI),
(28)
where TCE(fI) means the TCE of the function fI.
Proof. By definition, we have
ECEBin(f) =
B
X
i=1
|E(X,Y )∼D[(Y −f(X)) · 1f(X)∈Ii]|
=
B
X
i=1
P(f(X) ∈Ii)E|E[Y |f(X) ∈Ii] −E[f(X)|f(X) ∈Ii]|,
(29)
16


<!-- page 17 -->
where we used the definition of the conditional expectation. On the other hand, We have
TCE(fI) = E|E[Y |fI(x)] −fI(x)|
=
B
X
i=1
E

|E[Y |fI(x)] −fI(x)| · 1fI(X)∈Ii

=
B
X
i=1
P(fI(x) ∈Ii)E [|E[Y |fI(x)] −fI(x)|fI(X) ∈Ii]
=
B
X
i=1
P(f(X) ∈Ii)E|E[Y |f(X) ∈Ii] −E[f(X) ∈Ii]|,
(30)
where we used the tower property. This concludes the proof.
Thus, we can transform the loss and ECEs by Eqs. (24) and (27)
C.2
Useful inequalities
Here we introduce lemmas, which we will use repeatedly in our proofs.
Lemma 2 (Corollary 3.2 in Boucheron et al. [3]). We say that a function f : X →R has the
bounded differences property if for some nonnegative constants c1, . . . , cn,
sup
x1,...,xn,x′
i∈X
|f(x1, . . . , xn) −f(x1, . . . , xi−1, x′
i, xi+1, . . . , xn)| ≤ci,
1 ≤i ≤n.
(31)
If X1, . . . , Xn are independent random variables taking values in X, we define the real-valued
random variable as
Z = f(X1, . . . , Xn).
(32)
If f has the bounded difference property with constants c1, . . . , cn, then we have
Var[Z] ≤1
4
n
X
i=1
c2
i .
(33)
Combining this lemma with Holder inequality, we have the following relation,
E|Z −E[Z]| ≤
v
u
u
t1
4
n
X
i=1
c2
i .
(34)
We often consider the case where −1 ≤f ≤1. This implies that ci = 2 for all i. Then
E|Z −E[Z]| ≤1
(35)
Another situation is that given a function g
:
X
→
[−1, 1], consider f(x1, . . . , xn)
:=
1
n
Pn
i=1 f(xi). , where f corresponds to the empirical mean of some function g. This implies
that ci = 2/n for all i. Then
E|f −E[f]| ≤
1
√n.
(36)
Lemma 3 (Used in the proof of McDiarmid’s inequality). Given a bounded difference function f,
for any t ∈R, we have
E
h
et(f(X1,...,Xn)−E[f(X1,...,Xn)])i
≤e
t2
8
Pn
i=1 c2
i .
(37)
17


<!-- page 18 -->
D
Proofs of Section 3
D.1
Proofs of Theorem 2
Proof. Here we express the samples in Ste as {Z′
m}nte
m=1 = {(X′
m, Y ′
m)}.
As for the first inequality, it is the consequence of the Jensen inequality, as follows;
TCE(fI) =
B
X
i=1
EZ′=(X′,Y ′)

(Y ′ −fW (X′) · 1fW (X′)∈Ii

=
B
X
i=1
E{Z′m=(X′m,Y ′m)}nte
m=1
1
nte
nte
X
m=1
(Y ′
m −fW (X′
m)) · 1fW (X′m)∈Ii

≤E{Z′m=(X′m,Y ′m)}nte
m=1
B
X
i=1

1
nte
nte
X
m=1
(Y ′
m −fW (X′
m)) · 1fW (X′m)∈Ii

= ESteECE(fW , Ste),
(38)
where we used the Jensen inequality.
As for the second inequality, we separately prove UWB and UMB.
D.1.1
Proof for the UWB
We start from UWB. Conditioned on W, using the relation between the loss and ECEs by Eqs. (24)
and (27), we have
|TCE(fI) −ECE(fW , Ste)|
=

B
X
i=1

E
Z′′=(X′′,Y ′′)

(Y ′′ −fW (X′′))·1fW (X′′)∈Ii
−
B
X
i=1

1
nte
nte
X
m=1
(Y ′
m −fW (X′
m))·1fW (X′m)∈Ii


≤
B
X
i=1

E
Z′′=(X′′,Y ′′)

(Y ′′ −fW (X′′)) · 1fW (X′′)∈Ii

−1
nte
nte
X
m=1
(Y ′
m −fW (X′
m))·1fW (X′m)∈Ii

≤
B
X
i=1
EZ′′li(Z′′) −1
nte
nte
X
m=1
li(Z′
m)
 ,
(39)
where we used the triangle inequality ||a| −|b|| ≤|a −b| for the first inequality and set li(z) =
(y −fW (x)) · 1fW (x)∈Ii.
We use the following relation: for the one-dimensional real random variable X, by Jensen inequality,
we have
tE[|X|] ≤log E[et|X|].
(40)
Then combining with the above, we have
ESte|TCE(fI) −ECE(fW , Ste)| ≤1
t ESteet PB
i=1|EZ′′li(Z′′)−
1
nte
Pnte
m=1 li(Z′
m)|.
(41)
18


<!-- page 19 -->
By setting g(i, Ste) := EZ′′li(Z′′) −
1
nte
Pnte
m=1 li(Z′
m), we have
ESteet PB
i=1 |g(i,Ste)| = ESte
B
Y
i=1
et|g(i,Ste)|
≤ESte
B
Y
i=1

etg(i,Ste) + e−tg(i,Ste)
≤ESte
X
v1,...,vB=0,1
et PB
i=1(−1)vig(i,Ste)
(42)
=
X
v1,...,vB=0,1
ESteet PB
i=1(−1)vig(i,Ste)
=
X
v1,...,vB=0,1
ESteet PB
i=1(−1)vi[EZ′′li(Z′′)−
1
nte
Pnte
m=1 li(Z′
m)],
(43)
where P
v1,...,vB=0,1 is all the combinations that will be generated by expanding QB
i=1 in Eq. (42)
and it has 2B combinations.
We would like to upper bound ESteet PB
i=1(−1)vi[EZ′′li(Z′′)−
1
nte
Pnte
m=1 li(Z′
m)] using Lemma 3. For
that purpose, here we evaluate cis of Lemma 3. By focusing on the exponent, we can estimate cis
by
sup
{z′m}nte
m=1,˜zm′∈Z
B
X
i=1
t(−1)vi ·
"
EZ′′li(Z′′) −1
nte
nte
X
m=1
li(z′
m)
#
−t(−1)vi ·

EZ′′li(Z′′) −1
nte
nte
X
m̸=m′
li(z′
m) −1
nte
li(˜zm′)


=
sup
z′m,˜zm′∈Z
B
X
i=1
t(−1)vi
nte
· [−li(z′
m′) + li(˜zm′)]
=
sup
z′m,˜zm′∈Z
t(−1)v1
nte
·
−
(y′
m′ −fW (x′
m′)) · 1fW (x′
m′)∈I1

+
(˜y′
m′ −fW (˜x′
m′))·1fW (˜x′
m′)∈I1

+
...
+ t(−1)vB
nte
·

−
(y′
m′ −fW (x′
m′)) · 1fW (x′
m′)∈IB

+
(˜ym′ −fW (˜xm′)) · 1fW (˜xm′)∈IB

(44)
≤2t
nte
,
(45)
where the last inequality is derived as follows; Here by definition of the binning, each data point is
allocated to a single bin. This means that for the input x′
m′, one of {1fW (x′
m′)∈Ii}B
i=1 is not zero.
We refer to such index as b. Then 1fW (x′
m′)∈Ib ̸= 0 at the b-th bin and 1fw(x′
m′)∈Ib′̸=b = 0 holds. A
similar argument holds for the input ˜x′
m′ and we refer to the index that the indicator function is not
zero as ˜b, which implies 1fW (˜x′
m′)∈I˜b ̸= 0 and 1fw(x′
m′)∈Ib′̸=˜b = 0. Note that such b and ˜b can be
equal and can be different. Thus, although there are 2B indicator functions in Eq. (44), at most only
two indicator functions are not zero.
Combined with the fact that |y′
m′ −fw(x′
m′)| ≤1, we obtain Eq. (45). Note that by Assumption 1,
{fw(xm)}nte
m=1 in xm ∈Ste takes the distinct values almost surely and in the above discussion, we
do not consider the case when b/B = fw(xm) for some b holds, which means that the predicted
probability is just the value of the boundary of bins.
When we do not assume that Assumption 1, there may be a possibility that b/B = fw(xm) for some
b holds, which means that the predicted probability is just the value of the boundary of bins. Then,
19


<!-- page 20 -->
at most only four indicator functions are not zero. This results in a worse bound
sup
{zm}nte
m=1,˜zm∈Z
B
X
i=1
t(−1)vi ·
"
EZ′li(Z′) −1
nte
nte
X
m=1
li(z′
m)
#
−t(−1)vi ·

EZ′li(Z′) −1
nte
nte
X
m̸=m′
li(z′
m) −1
nte
li(˜zm′)


≤4t
nte
.
(46)
Combined with Lemma 3, we have that
ESteet PB
i=1 |g(i,Z′
m)| ≤
X
v1,...,vB=0,1
nte
Y
m=1
EZ′
met PB
i=1(−1)vi[EZ′′li(Z′′)−
1
nte
Pnte
m=1 li(Z′
m)]
X
v1,...,vB=0,1
nte
Y
m=1
EZ′met PB
i=1(−1)vi[EZ′′li(Z′′)−
1
nte
Pnte
m=1 li(Z′
m)]
≤
X
v1,...,vB=0,1
e(t2/8)nte(
2
nte )
2
= 2Be
2t2
nte .
(47)
Combining this with Eq. (41), we have
ESte|TCE(fI) −ECE(fW , Ste)| ≤1
t log ESteet PB
i=1|EZ′′li(Z′′)−
1
nte
Pnte
m=1 li(Z′
m)|
≤log 2Be
t2
2nte
t
= B log 2
t
+
t
2nte
≤
r
2B log 2
nte
.
(48)
D.1.2
Proofs for the UMB
The proof goes similarly as in the case of UWB up to Eq.(41). Then we need special care to bound
the exponential moment since in the UMB, bins are constructed using Ste, and thus the samples are
no longer i.i.d. Recall the definitions that f(m) is the m-th order statistics of (f1, . . . , fnte) and the
boundaries of bins are defined by such order statistics ub := f(⌊nteb/B⌋).
We define the set S(B), which is the set of test data points used for defining the boundaries of bins.
We then define ˜Ste as Ste −S(B), and thus ˜Ste is the set of test data points, which is not used for the
boundaries. We express fw(S(B)) = {fw(x)|x ∈S(B)}.
We define kb := ⌊nteb/B⌋, which is used for defining the b-th bin. Let fix b ∈[B] and denote
u = kb and l = kb−1. Then Gupta and Ramdas [12] showed that f(l+1), . . . , f(u−1) are independent
and identically distributed given boundary points {f(⌊nteb/B⌋)}B−1
b=0 in Lemma 1 [12]. Moreover,
Lemma 2 in Gupta and Ramdas [12] showed that let p be the density induced by the distribution
P(fw(X)), then
p(f(l+1), . . . , f(u−1)|fw(S(B)))
= p( ˜fl+1, . . . , ˜fu−1|fw(S(B)), for every i ∈[l + 1, u −1], f(l) < ˜fi < f(u)),
(49)
where each ˜fi is independent random variables ˜fi ∼P(fw(X)). This implies that given the bound-
ary points defining bins, the data points inside the boundary are i.i.d.
20


<!-- page 21 -->
In order to use this result, we need to eliminate f(u) from the empirical mean of UMB. This is
evaluated as follows; using this from the result of UWB, we have
|TCE(fI) −ECE(fW , Ste)|
≤
B
X
i=1
EZ′′li(Z′′) −1
nte
nte
X
m=1
li(Z′
m)

≤
B
X
i=1

EZ′′li(Z′′) −
1
| ˜Ste|
| ˜Ste|
X
m=1
li(Z′
m)

+

1
| ˜Ste|
| ˜Ste|
X
m=1
li(Z′
m) −1
nte
nte
X
m=1
li(Z′
m)

.
(50)
This partition eliminates the boundary point from the empirical estimation. We can upper bound the
second term as

1
| ˜Ste|
| ˜Ste|
X
m=1
li(Z′
m) −1
nte
nte
X
m=1
li(Z′
m)

≤
2B
nte −B ,
(51)
which follows directly by definition (see also Corollary 1 in Gupta and Ramdas [12].). Then we
have
ESte|TCE(fI) −ECE(fW , Ste)|
≤1
t ESteet PB
i=1|EZ′′li(Z′′)−
1
nte
Pnte
m=1 li(Z′
m)|
≤1
t ES(B)

E ˜Stee
t PB
i=1
EZ′′li(Z′′)−
1
| ˜
Ste|
P
m∈˜
Ste li(Z′
m)
+ 2tB
n−B

.
(52)
Given the boundary point, the above exponential moment satisfies the condition of Lemma 3, since
the li(Z′
m) are i.i.d, given in each bin by Lemma 2 in Gupta and Ramdas [12]. This can also be
confirmed that for random variables (f(1), . . . , f(i), . . . , f(nte)), given f(i), (f(1), . . . , f(i−1)) and
f(i+1), . . . , f(nte)) are conditionally independent (this is proved in Gupta and Ramdas [12], espe-
cially the proof of Lemma 2). Combined with Eq. (49), given the boundary points, ˜Ste are i.i.d and
the size of which is nte −B. Then we only need to upper bound
1
t ES(B)

E ˜Stee
t PB
i=1
EZ′′li(Z′′)−
1
| ˜
Ste|
P
m∈˜
Ste li(Z′
m)


.
(53)
We can upper bound this in a similar way as in the case of UWB, replacing nte with nte −B under
Assumption 1. Thus, we have
ESte|TCE(fI) −ECE(fW , Ste)| ≤1
t ESteet PB
i=1|EZ′′li(Z′′)−
1
nte
P
m=1 li(Z′
m)|
≤
r
2 B log 2
nte −B +
2B
nte −B .
(54)
This concludes the proof.
D.2
Comparison with the trivial bound
We remark that for UWB, we can also upper bound in the following way;
ESte|TCE(fI) −ECE(fW , Ste)| ≤ESte
B
X
i=1
EZ′′li(Z′′) −1
nte
nte
X
m=1
li(Z′
m)

≤ESte
B
X
i=1
v
u
u
tVar
"
1
nte
nte
X
m=1
li(Z′m)
#
≤ESte
B
X
i=1
r
1
nte
=
B
√nte
,
(55)
where we used the triangle inequality ||a| −|b|| ≤|a −b| for the first inequality and set li(z) =
(y −fW (x)) · 1fW (x)∈Ii. Note that since −1 ≤li(z) ≤1, we can use Eq. (36). However, since we
did not use the property of the indicator function, this suffers from the slow convergence of B.
21


<!-- page 22 -->
D.3
High probability bound
In the main paper, we present the expectation bound. On the other hand, as shown in the above proof,
we evaluated the exponential moment. Thus, we can obtain the high probability bound directly.
Corollary 2. Under the same assumptions in Theorem 2, for any δ ∈(0, 1), we have
PSte

|TCE(fI) −ECE(fW , Ste)| ≤
s
2B log 2 + log 1
δ
nte

≥1 −δ.
(56)
This means the statistical bias is Op(
p
B/nte).
Proof. Using the proof of Theorem 2, and Chernoff-bounding technique, for any t > 0, we have
PSte (|TCE(fI) −ECE(fW , Ste)| ≥ε) ≤e−tεESteet PB
i=1|EZ′′li(Z′′)−
1
nte
Pnte
m=1 li(Z′
m)|
≤2Be−nε2
2nte + (t−nε)2
2nte .
(57)
By setting t = nε then
PSte (|TCE(fI) −ECE(fW , Ste)| ≥ε) ≤2Be−nteε2
2nte ,
(58)
and setting δ := 2Be−nε2
2nte , we have that
PSte

|TCE(fI) −ECE(fW , Ste)| ≥
s
2B log 2 + log 1
δ
nte

≤δ.
(59)
D.4
Proofs of Theorem 3
Proof. We use the following lemma to study the binning bias.
Lemma 4.
TCE(fI) ≤TCE(fw) ≤TCE(fI) + E||E[Y |fw(X)] −E[Y |fI(X)]| + E|fw(X) −fI(X)|.
(60)
This implies that
|TCE(fw) −TCE(fI)| ≤E||E[Y |fw(X)] −E[Y |fI(X)]| + E|fw(X) −fI(X)|.
(61)
Proof. The first inequality has been proved in Proposition 3.3 in Kumar et al. [23].
22


<!-- page 23 -->
The second inequality is proved as follows;
TCE(fw) = E [|E[Y |fw(X)] −fw(X)|]
=
B
X
i=1
E[1fw(X)∈Ii · |E[Y |fw(X)] −fw(X)|]
=
B
X
i=1
P(fw(X) ∈Ii)E[|E[Y |fw(X)] −fw(X)||fw(X) ∈Ii]
=
B
X
i=1
P(fw(X) ∈Ii)E[|E[Y |fw(X)] −E[fw(X)|fw(X) ∈Ii]
+ E[fw(X)|fw(X) ∈Ii] −fw(X)||fw(X) ∈Ii]
≤
B
X
i=1
P(fw(X) ∈Ii)E||E[Y |fw(X)] −E[Y |fw(X) ∈Ii]|
+
B
X
i=1
P(fw(X) ∈Ii)E|E[Y |fw(X) ∈Ii] −E[fw(X)|fw(X) ∈Ii]|
+
B
X
i=1
P(fw(X) ∈Ii)E[|E[fw(X)|fw(X) ∈Ii] −fw(X)||fw(X) ∈Ii].
(62)
In the above, the second term corresponds to TCE(fI).
As for UWB, from Lemma 4, we have
|TCE(fw) −TCE(fI)| ≤E||E[Y |fw(X)] −E[Y |fI(X)]| + E|fw(X) −fI(X)| ≤L
B + 1
B
(63)
where we used the fact that with UWB, we split the function with equal width 1/B and used the
Lipschitz continuity of the function.
Next, we focus on UMB. To analyze the binning bias of this, we focus on Eq. (50). We replace the
loss lm(z) in that equation with
lm(z) = 1fW (x)∈Im
nte
.
(64)
Then, the first line and second line of Eq. (50) can be rewritten as
B
X
i=1
|P(fw(x) ∈Im)) −ˆP(Ii)| ≤
B
X
i=1
EZ′′li(Z′′) −1
nte
nte
X
m=1
li(Z′
m)

(65)
where ˆP(Ii) is the empirical estimate of the binning probability P(Im). The right-hand side can be
bounded in the same way as Appendix. D.1.2, which requires evaluating the exponential moment.
The proof goes the same way, that is, we utilize the results of Gupta and Ramdas [12] and the upper
bound of the exponential moment. The procedure is exactly the same. Thus, we have
B
X
i=1
|P(fw(x) ∈Im)) −ˆP(Ii)| ≤
s
2B log 2
(nte −B) +
2B
nte −B .
(66)
By definition, we put equal mass in any bin, thus,
ˆP(Ii) = u −l + 1
nte
≤1
B .
from the proof of Theorem 3 in Gupta and Ramdas [12].
23


<!-- page 24 -->
Thus, by the Jensen inequality, we have for any m ∈[B],
P(fw(x) ∈Im)) ≤1
B +
s
2B log 2
(nte −B) +
2B
nte −B .
(67)
Combining these results, the binning bias is upper bounded by
E|fw(X) −fI(X)|
=
B
X
i=1
P(fw(X) ∈Ii)E[|E[fw(X)|fw(X) ∈Ii] −fw(X)||fw(X) ∈Ii]
≤
 1
B +
s
2B log 2
(nte −B) +
2B
nte −B

B
X
i=1

E[|E[fW (X)|fW (X) ∈Ii] −fW (X)||fW (X) ∈Ii]

(68)
We use the fact that E[|fw(X)−E[fw(X)|Ii]||Ii] ≤f(⌊ni/B⌋)−f(⌊n(i−1)/B⌋) holds by the definition
of the UMB, which is the largest difference of the bins. Then
B
X
i=1

E[|E[fW (X)|fW (X) ∈Ii] −fW (X)||fW (X) ∈Ii]

≤
B
X
i=1
f(⌊ni/B⌋) −f(⌊n(i−1)/B⌋) ≤1.
(69)
Combining the above, Then we have
E|fw(X) −fI(X)| ≤1
B +
s
2B log 2
(nte −B) +
2B
nte −B .
(70)
As for the E||E[Y |fw(X)]−E[Y |fI(X)]|, by the assumption of the Lipschitz continuity, we simply
multiply L to the above and obtain
E||E[Y |fw(X)] −E[Y |fI(X)]| ≤LE|fw(X) −fI(X)| ≤L
 
1
B +
s
2B log 2
(nte −B) +
2B
nte −B
!
.
(71)
This concludes the proof.
D.5
H¨older continuity does not improve the total bias
In the nonparametric estimation, imposing the higher order smoothness improves the bias order.
According to Tsybakov [36], the lower bound is O(n
−
β
β+1
te
) if we assume β-H¨older continuity.
In our total bias analysis, the statistical bias is not improved by this assumption. As for the binning
bias, if we assume that E[Y |fw(X)] satisfies β-H¨older continuity with constant M for all the order,
then we obtain that for UWB,
|TCE(fw) −TCE(fI)| ≤E|E[Y |fw(X)] −E[Y |fI(X)]| + E|fw(X) −fI(X)| ≤M
Bβ + 1
B
(72)
Combined with the statistical bias we have that
Biastot(fw, Ste) ≤M
Bβ + 1
B +
r
2B log 2
nte
(73)
and the optimal bin size is again O(n1/3
te ) and resulting bias is Biastot(fw, Ste) = O(n−1/3
te
), which
does not improve the bias. This is because of the error term of E|fw(X)−fI(X)|. This term cannot
be improved by 1/B, thus we cannot leverage the underlying smoothness of the data. A similar
discussion holds for the UMB setting.
24


<!-- page 25 -->
D.6
Additional comparison with existing work
In Gupta and Ramdas [12], the error bound of ECE is derived through the following three steps:
(i) Firstly, showing that the samples assigned to each bin are i.i.d., (ii) using Hoeffding inequal-
ity, deriveving |E[Y |fI(x)] −fI(x)| = Op(
p
B/nte) for each bin, and (iii) finally, summing up
these error bounds for all bins, resulting in Op(
p
B log B/nte) (in expectation O(B/√nte). This
slow convergence is attributed to the separated analysis for each bin, which necessitates multiple
applications of concentration inequalities.
In addition to the slow convergence, it is difficult to derive the error bound for UWB using this
approach. The difficulty lies in demonstrating convergence for specific bins. For instance, in existing
UMB studies Gupta and Ramdas [12], it becomes inevitable to address the allocation of samples to
each bin when attempting to discuss the convergence of sample means for each bin. In UMB, an
equal number of samples, nte/B, are allocated to each bin to achieve equal mass across all bins.
In UWB, however, there is no guarantee about the number of samples entering each bin (in the
worst case, all samples might be assigned to a single bin) because the widths of all bins are set
equally. This necessitates discussions about the number of samples allocated to intervals under
strong assumptions regarding E[Y |fw(x)], requiring stronger assumptions compared to both this
study and existing research.
E
Proofs of Section 4
First, we introduce notations, which are used in the IT-based analysis. We express the super-samples
as
z = (x, y),
(74)
˜z = (˜x, ˜y),
(75)
˜zm = (˜xm, ˜ym),
(76)
˜zU = (˜xU, ˜yU),
(77)
˜zm,Um = (˜xm,Um, ˜ym,Um),
(78)
˜zm, ¯Um = (˜xm, ¯Um, ˜ym, ¯Um).
(79)
We also define the total bias when using Str as
Biastot(fw, Str) := |TCE(fw) −ECE(fw, Str)|.
(80)
We then decompose this bias into two biases as follows:
Biastot(fw, Str) ≤Biasbin(fw, fI)+Biasstat(fw, Str),
(81)
where
Biasbin(fw, fI) := |TCE(fw) −TCE(fI)|,
(82)
Biasstat(fw, Str) := |TCE(fI) −ECE(fw, Str)|.
(83)
We remark that the bins used in fI of UMB are constructed using Str and thus, TCE(fI) also
depends on Str.
E.1
Proof of Theorem 4 (The statistical bias when reusing the training dataset)
Proof. We start with the case of UWB. The proofs goes almost similar way as the standard
information-theoretic generalization error bounds.
Using the relation between the loss and ECEs by Eqs. (24) and (27), first we reformulate the ECEs
as follows;
ECE(fA( ˜
ZU,R), ˜Z ¯U) =
B
X
i=1

1
n
n
X
m=1
( ˜Ym, ¯Um −fA( ˜
ZU,R)( ˜Xm, ¯Um)) · 1fW ( ˜
Xm, ¯
Um)∈Ii

(84)
ECE(fA( ˜
ZU,R), ˜ZU) =
B
X
i=1

1
n
n
X
m=1
( ˜Ym,Um −fA( ˜
ZU,R)( ˜Xm,Um)) · 1fW ( ˜
Xm,Um)∈Ii

(85)
25


<!-- page 26 -->
To simplify the notation, we also introduce the loss as
l(A( ˜ZU, R), Z, i) := ((Y −fA( ˜
ZU,R)(X)) · 1fA( ˜
ZU ,R)(X)∈Ii,
(86)
where Z = (X, Y ). Then we obtain
∆(U, ˜Z, A( ˜ZU, R)) := |ECE(fA( ˜
ZU,R), ˜Z ¯U) −ECE(fA( ˜
ZU,R), ˜ZU)|
=

B
X
i=1

1
n
n
X
m=1
l(A( ˜ZU, R), ˜Zm, ¯Um, i)
 −
B
X
i=1

1
n
n
X
m=1
l(A( ˜ZU, R), ˜Zm,Um, i)


≤
B
X
i=1

1
n
n
X
m=1
l(A( ˜ZU, R), ˜Zm, ¯Um,i)−1
n
n
X
m=1
l(A( ˜ZU, R), ˜Zm,Um,i)
 ,
(87)
where W in the second line should be W = A( ˜ZU, R), but to make the presentation simpler, we
used W. We also used the triangle inequality ||a| −|b|| < |a −b|.
With this notation, by using the Donsker–Varadhan lemma, we have
ER, ˜
Z,U∆(U, ˜Z, A( ˜ZU, R))
≤inf
t>0
I(∆(U, ˜Z, A( ˜ZU, R)); U| ˜Z) + E ˜
Z log ER,U ′,U et∆(U ′, ˜
Z,A( ˜
ZU,R))
t
≤inf
t>0
I(∆(U, ˜Z, A( ˜ZU, R)); U| ˜Z)
t
+ E ˜
Z log ER,U ′,U e
t PB
i=1

1
n
Pn
m=1 l(A( ˜
ZU,R), ˜
Zm, ¯
U′
m,i)−1
n
Pn
m=1 l(A( ˜
ZU,R), ˜
Zm,U′
m,i)

t
= inf
t>0
I(∆(U, ˜Z, A( ˜ZU, R)); U| ˜Z) + E ˜
Z log ER,U ′,Uet PB
i=1|g( ˜
Z,U,R,U ′,i)|
t
,
(88)
where we introduced
g(˜z, u, r, U ′, i) := 1
n
n
X
m=1
l(A(˜zu, r), ˜zm, ¯U ′m, i) −1
n
n
X
m=1
l(A(˜zu, r), ˜zm,U ′m, i).
(89)
Our first observation is that conditioned on ˜Z = ˜z, R = r, and U = u, the expectation of the
exponent is
EU ′ t
B
B
X
i=1
g(˜z, u, r, U ′, i) = 0,
(90)
by definition. Then similarly to Appendix D.1, we upper bound the exponential moment as follows;
EU ′et PB
i=1 |g(˜z,u,r,U ′,i)| = EU ′
B
Y
i=1
et|g(˜z,u,r,U ′,i)|
≤EU ′
B
Y
i=1

etg(˜z,u,r,U ′,i) + e−tg(˜z,u,r,U ′,i)
= EU ′
X
v1,...,vB=0,1
et PB
i=1(−1)vig(˜z,u,r,U ′,i)
(91)
=
X
v1,...,vB=0,1
EU ′et PB
i=1(−1)vig(˜z,u,r,U ′,i)
=
X
v1,...,vB=0,1
EU ′
n
Y
m=1
e
t
n
PB
i=1(−1)vi
h
l(w,˜zm, ¯
U′m,i)−l(w,˜zm,U′m,i)
i
=
X
v1,...,vB=0,1
n
Y
m=1
EU ′me
t
n
PB
i=1(−1)vi
h
l(w,˜zm, ¯
U′m,i)−l(w,˜zm,U′m,i)
i
,
(92)
26


<!-- page 27 -->
where P
v1,...,vB=0,1 is all the combinations that will be generated by expanding QB
i=1 in Eq. (91)
and it has 2B combinations.
We would like to upper bound EU ′me
t
n
Pn
i=1(−1)vi
h
l(w,˜zm, ¯
U′m,i)−l(w,˜zm,U′m,i)
i
using Lemma 3 condi-
tioned on all other random variables. For that purpose, here we evaluate ci of Lemma 3. To estimate
it let us focus on U ′
m and it takes value U ′
m = {0, 1}. So let us consider how the exponent changes
by changing U ′
m = 1 to U ′
m = 0. Then the difference of the exponent is written as
B
X
i=1
t(−1)vi
n
· [l(w, ˜zm,1, i) −l(w, ˜zm,0, i)] −t(−1)vi
n
· [l(w, ˜zm,0, i) −l(w, ˜zm,1, i)]
= 2t(−1)v1
n
·
[ym,1 −fw(xm,1)] · 1fw(xm,1)∈I1

−
[ym,0 −fw(xm,0)] · 1fw(xm,0)∈I1

+
...
+ 2t(−1)vB
n
·
[ym,1 −fw(xm,1)] · 1fw(xm,1)∈IB

−
[ym,0 −fw(xm,0)] · 1fw(xm,0)∈IB

.
(93)
To evaluate the indicator function, we repeat the same discussion in Eqs. 44 and (45). On the
basis of that discussion, by the definition of binning, for the input xn,0, exactly one of the indica-
tors {1fw(xn,0)∈Ii}B
i=1 is non-zero, denoted as b. Consequently, all other indicators are zero, i.e.,
1fw(xn,0)∈Ib′̸=b = 0. Similarly, for input xn,1, the corresponding non-zero bin index is denoted as
˜b, so 1fw(xn,1)∈I˜b is nonzero and others are zero. It should be noted that b and ˜b may be the same or
different.
Thus, although there are 2B indicator functions in Eq. (93), at most only two indicator functions are
not zero. Combined with the fact that |ym,1 −fw(xm,1)| ≤1 and |ym,0 −fw(xm,0)| ≤1, Eq. (93)
is upper bounded by 4t
n .
Note that by Assumption 1, {fw(xm,Um)}n
m=1 takes the distinct values almost surely and in the
above discussion, we do not consider the case when b/B = fw(xm,Um) for some b holds, which
means that the predicted probability is just the value of the boundary of bins. In conclusion, we
obtain the upper bound of Eq. (93) as

B
X
i=1
t(−1)vi
n
· [l(w, ˜zm,1, i) −l(w, ˜zm,0, i)] −t(−1)vi
n
· [l(w, ˜zm,0, i) −l(w, ˜zm,1, i)]
 ≤4t
n .
(94)
Then by Lemma 3, we have that
X
v1,...,vB=0,1
n
Y
m=1
EU ′me
t
n
Pn
i=1(−1)vi
h
l(w,˜zm, ¯
U′m,i)−l(w,˜zm,U′m,i)
i
≤
X
v1,...,vB=0,1
n
Y
m=1
e
2t2
n2 = 2Be
2t2
n .
(95)
Thus
EU ′et PB
i=1 |g(˜z,u,r,U ′,i)| = EU ′
B
Y
i=1
et|g(˜z,u,r,U ′,i)| ≤2Be
2t2
n ,
(96)
and combining with Eq. (88), we have
ER, ˜
Z,U

B
X
i=1

1
n
n
X
m=1
l(A( ˜ZU, R), ˜Zm, ¯Um, i)
 −
B
X
i=1

1
n
n
X
m=1
l(A( ˜ZU, R), ˜Zm,Um, i)


≤inf
t>0
I(∆(U, ˜Z, A( ˜ZU, R)); U| ˜Z) + B log 2 + 2t2
n
t
≤
s
8(I(∆(U, ˜Z, A( ˜ZU, R)); U| ˜Z) + B log 2)
n
.
(97)
27


<!-- page 28 -->
This concludes the proof of UWB.
We next prove the case of UMB. The key difference lies in the fact that the bins are dependent on
the training samples.
∆(U, ˜Z, A( ˜ZU, R)) := |ECE(fW , ˜Z ¯U) −ECE(fW , ˜ZU)|
=

B
X
i=1

1
n
n
X
m=1
(˜ym, ¯Um −fA( ˜
ZU,R)(˜xm, ¯Um)) · 1fw(˜xm, ¯
Um)∈Ii( ˜
ZU)

−
B
X
i=1

1
n
n
X
m=1
(ym, ¯Um −fA( ˜
ZU,R)(xm, ¯Um)) · 1fw(xm, ¯
Um)∈Ii( ˜
ZU)

,
where we expressed the dependency of bins on the training samples as Ii( ˜ZU). However, this
dependency does not change the proof in the above; we use the Donsker–Varadhan lemma. We upper
bound of the exponential moment. When upper bounding the exponential moment, we conditioned
on U, which means we conditioned on the bins. So we can exactly use the same derivation. So we
can proceed with the proof exactly in the same way as UWB.
Finally, we can bound the statistical bias using the Jensen inequality as follows: First following the
proof of Theorem 2, using Eqs. (24) and (27), we have
E
R,Str
|TCE(fI) −ECE(fW , Str)|
=
E
R,Str

B
X
i=1

E
Z′′=(X′′,Y ′′)

(Y ′′−fW (X′′)) · 1fW (X′′)∈Ii
−
B
X
i=1

1
n
n
X
m=1
(Ym−fW (Xm))·1fW (Xm)∈Ii


≤
E
R,Str
B
X
i=1

E
Z′′=(X′′,Y ′′)

(Y ′′ −fW (X′′)) · 1fW (X′′)∈Ii

−1
n
n
X
m=1
(Ym −fW (Xm)) · 1fW (Xm)∈Ii

=
E
R,Str
B
X
i=1

E
{Z′′
m}n
m=1
1
n
n
X
m=1

(Y ′′
m −fW (X′′
m)) · 1fW (X′′
m)∈Ii

−1
n
n
X
m=1
(Ym−fW (Xm))·1fW (Xm)∈Ii

≤
E
R,Str,{Z′′
m}n
m=1
B
X
i=1

1
n
n
X
m=1

(Y ′′
m −fW (X′′
m)) · 1fW (X′′
m)∈Ii

−1
n
n
X
m=1
(Ym−fW (Xm))·1fW (Xm)∈Ii

= ER,Str,Ste|ECE(fW , Ste) −ECE(fW , Str)|,
(98)
where the first inequality is the triangle inequality and the second inequality is the Jensen inequality.
Note that the above reformulation is possible for both UWB and UMB. Although in the case of
UMB, bins of the TCE still depend on Str, it makes no difference in the above inequalities. We then
use Theorem 4.
Remark 1. In the above proof of UWB, instead of Eq. (88), it is possible to consider the following
type Donsker–Varadhan inequality
ER, ˜
Z,U

B
X
i=1

1
n
n
X
m=1
l(A( ˜ZU, R), ˜Zm, ¯Um, i)
 −
B
X
i=1

1
n
n
X
m=1
l(A( ˜ZU, R), ˜Zm,Um, i)


≤inf
t>0
I(l(A( ˜ZU, R), ˜Z, B); U| ˜Z)
t
+ E ˜
Z log ER,U ′,U e
t PB
i=1

1
n
Pn
m=1 l(A( ˜
ZU,R), ˜
Zm, ¯
U′
m,i)−1
n
Pn
m=1 l(A( ˜
ZU,R), ˜
Zm,U′
m,i)

t
= inf
t>0
I(l(A( ˜ZU, R), ˜Z, B); U| ˜Z) + E ˜
Z log ER,U ′,Uet PB
i=1|g( ˜
Z,U,R,U ′,i)|
t
,
(99)
which results in a looser bound than the above proof, which can be confirmed by the data processing
inequality.
28


<!-- page 29 -->
E.2
Proof of Theorem 5 (The total bias)
Before presenting the proof of the total bias, we first provide the following binning bias analysis for
the UMB.
Theorem 8. For UMB, Under the CMI setting and under Assumptions 1 and 2, we have
ER,Str|TCE(fW )−TCE(fI)| ≤(1 + L)
 
1
B +
r
2
n (fCMI+B log 2)
!
,
(100)
where fCMI is defined in Eq. (17).
Proof of Theorem 8. The proof is similar in Appendix D.4. The difference is that in the current
setting, we reuse the training dataset, so we need to evaluate the bias for that. First, in the same way
as in Appendix D.4, we have that
ER,StrE|fW (X) −fI(X)|
= ER,Str
B
X
i=1
P(fW (X) ∈Ii)E|E[fW (X)|fW (X) ∈Ii] −fW (X)||fW (X) ∈Ii]
≤
B
X
i=1

E
R,Str
|P(fW (X) ∈Ii)|

E
R,Str
|E[|E[fW (X)|fW (X) ∈Ii] −fW (X)||fW (X) ∈Ii]|∞

,
(101)
where we used H¨older inequality in the second line and E| · |∞is the maximum of the integrand.
We want to estimate P(Ii) := P(f(X) ∈Ii). For this purpose, we use Eqs. (87) and (86). We
re-define the loss of Eq. (86)
l(A( ˜ZU, R), z, i) = 1fA( ˜
ZU ,R)(x)∈Ii.
(102)
and substitute it into Eq. (87), then we have that
E
R, ˜
Z,U
B
X
i=1
 ˜P(Ii) −ˆP(Ii)
 =
E
R, ˜
Z,U
B
X
i=1

n
X
m=1
1fA( ˜
ZU ,R)( ˜
Xm, ¯
Um)∈Ii
n
−
n
X
m=1
1fA( ˜
ZU ,R)( ˜
Xm,Um)∈Ii
n

,
(103)
where ˆP(Ii) is the empirical estimate of the binning probability using supersample ˜Xm,Um and
˜P(Ii) is that of obtained by ˜Xm, ¯Um. Then, to obtain the upper bound of the right-hand side of the
above, we repeat the proof of Theorem 4 in Appendix E.1. Let us define
∆(U, ˜Z, A( ˜ZU, R)) :=
B
X
i=1

1
n
n
X
m=1
(1fA( ˜
ZU ,R)( ˜
Xm, ¯
Um)∈Ii −1fA( ˜
ZU ,R)( ˜
Xm,Um)∈Ii)

(104)
29


<!-- page 30 -->
With this notation, by using the Donsker–Varadhan lemma, we have
ER, ˜
Z,U
B
X
i=1
 ˜P(Ii) −ˆP(Ii)

= ER, ˜
Z,U
B
X
i=1

Pn
m=1 1fA( ˜
ZU ,R)( ˜
Xm, ¯
Um)∈Ii
n
−
Pn
m=1 1fA( ˜
ZU ,R)( ˜
Xm,Um)∈Ii
n

ER, ˜
Z,U∆(U, ˜Z, A( ˜ZU, R))
≤inf
t>0
I(∆(U, ˜Z, A( ˜ZU, R)); U| ˜Z) + E ˜
Z log ER,U ′,U et∆(U ′, ˜
Z,A( ˜
ZU,R))
t
≤inf
t>0
I(fA( ˜
ZU,R)( ˜X); U| ˜Z)
t
+ E ˜
Z log ER,U ′,U et PB
i=1 |
Pn
m=1 1fA( ˜
ZU ,R)( ˜
Xm, ¯
U′m
)∈Ii
n
−
Pn
m=1 1fA( ˜
ZU ,R)( ˜
Xm,U′m
)∈Ii
n
|
t
= inf
t>0
I(fA( ˜
ZU,R)( ˜X); U| ˜Z) + E ˜
Z log ER,U ′,Uet PB
i=1|g( ˜
Z,U,R,U ′,i)|
t
,
(105)
where we introduced
g(˜z, u, r, U ′, i) := 1
n
n
X
m=1
l(A(˜zu, r), ˜zm, ¯U ′m, i) −1
n
n
X
m=1
l(A(˜zu, r), ˜zm,U ′m, i).
(106)
Here we use the fCMI := I(fA( ˜
ZU,R)( ˜X); U| ˜Z) and I(∆(U, ˜Z, A( ˜ZU, R)); U| ˜Z) ≤fCMI by the
data processing inequality since the indicator functions depend on fw.
Then, we can estimate this using Lemma 3 in a similar way. The difference is the estimation of the
upper-bound in Eq. (94), which is used for the evaluation of the exponential moment. Since we use
the indicator function as a loss, the coefficient cis for Lemma 3 is upper-bounded by 2t/n, not 4t/n.
Then we repeat the proof strategy replacing the exponential moment evaluation by 2t/n, not 4t/n.
With this difference,
EU ′et PB
i=1 |g(˜z,u,r,U ′,i)| = EU ′
B
Y
i=1
et|g(˜z,u,r,U ′,i)| ≤2Be
t2
2n ,
(107)
and Eq. (97) can be rewritten in the following way
ER, ˜
Z,U
B
X
i=1
 ˜P(Ii) −ˆP(Ii)
 ≤
r
2(fCMI + B log 2)
n
,
(108)
and clearly, by fixing some i, we have that
ER, ˜
Z,U
 ˜P(Ii) −ˆP(Ii)
 ≤ER, ˜
Z,U
B
X
i=1
 ˜P(Ii) −ˆP(Ii)
 ≤
r
2(fCMI + B log 2)
n
.
(109)
Since we put an equal mass for each bin for the training dataset with UMB, we have
ˆP(Ii) = u −l + 1
n
≤1
B .
Combined with Jensen inequality, we have
ER,StrP(Ii) ≤1
B +
r
2(fCMI + B log 2)
n
.
(110)
30


<!-- page 31 -->
Then we have
ER,StrE|fW (X) −fI(X)|
=
B
X
i=1
 1
B +
r
2(fCMI+B log 2)
n

E
R,Str
|E|E[fW (X)|fW (X) ∈Ii] −fW (X)||fW (X) ∈Ii]|∞
=
 1
B +
r
2(fCMI+B log 2)
n

B
X
i=1
E
R,Str
|E[|E[fW (X)|fW (X) ∈Ii] −fW (X)||fW (X) ∈Ii]|∞.
(111)
Finally, we use the fact that E[|f(X) −E[f(X)|Ii]||Ii] ≤f(⌊ni/B⌋) −f(⌊n(i−1)/B⌋) holds by the
definition of the UMB, which is the largest difference of the bins. Then
B
X
i=1
E
R,Str
|E[|E[fW (X)|fW (X) ∈Ii] −fW (X)||fW (X) ∈Ii]|∞
≤
B
X
i=1
f(⌊ni/B⌋) −f(⌊n(i−1)/B⌋) ≤1.
(112)
where E| · |∞is the maximum of the integrand.
Combining the above, Then we have
E
R,Str
E|fW (X) −fI(X)|
=
 1
B +
r
2(fCMI + B log 2)
n

B
X
i=1
E
R,Str
|E[|E[fW (X)|fW (X) ∈Ii] −fW (X)||fW (X) ∈Ii]|∞
≤1
B +
r
2(fCMI + B log 2)
n
.
(113)
This concludes the proof.
Using this we provide the proof of the total bias as follows;
Proof of Theorem 5. We use the triangle inequality,
ER,Str|TCE(fW ) −ECE(fW , Str)|
= ER,Str|TCE(fW ) −TCE(fI) + TCE(fI) −ECE(fW , Str)|
≤ER,Str|TCE(fW ) −TCE(fI)| + ER,Str|TCE(fI) −ECE(fW , Str)|.
(114)
The first term is the binning bias and the second term is the statistical bias.
We start from the UMB; we can bound the binning bias in the first term by Theorem 8 and the
statistical bias in the second term by Theorem 4 of the UMB.
As for the UWB, the binning bias is simply (1 + L)/B, which can be derived similarly as in Ap-
pendix D.4. As for the second term, we can bound it by Theorem 4 of the UWB.
This concludes the proof of Eq. (16).
Provided in the proof of Theorem 8 (especially in Eq. (105)), we can obtain the tighter version of
the binning bias bound as follows;
Corollary 3. For UMB, Under the CMI setting and under Assumptions 1 and 2, we have
ER,Str|TCE(fW )−TCE(fI)| ≤(1 + L)
 
1
B +
r
2
n

I(∆(U, ˜Z, A( ˜ZU, R)); U| ˜Z) +B log 2
!
,
(115)
where
∆(U, ˜Z, A( ˜ZU, R)) :=
B
X
i=1

1
n
n
X
m=1
(1fA( ˜
ZU ,R)( ˜
Xm, ¯
Um)∈Ii −1fA( ˜
ZU ,R)( ˜
Xm,Um)∈Ii)
 .
(116)
31


<!-- page 32 -->
In the proof of Theorem 8, we used the fact that I(∆(U, ˜Z, A( ˜ZU, R)); U| ˜Z) ≤fCMI by
the data processing inequality.
Thus, fCMI appearing in Theorem 5 can be replaced with
I(∆(U, ˜Z, A( ˜ZU, R)); U| ˜Z), which results in a tighter bound.
E.3
Proof of Theorem 6 (metric entropy)
Proof. Recall the setting, where we assume that fw ∈F has the metric entropy, log N(F, ∥·∥∞, δ),
with parameter δ (> 0). That is, there exists a set of functions Fδ := {f1, . . . , fN(F,∥·∥∞,δ)} that
consists δ-cover of F. We will consider to replace fw with the functions from the δ-cover.
Using the δ-cover, we want to construct a set of functions ˜F that satisfies the following property;
there exists a function f ∈˜F such that for any input x and that x is allocated to i-th bin, that is,
fw(x) ∈Ii, then f satisfies f(x) ∈Ii and ∥f −fw∥∞< δ.
The original δ cover F may not satisfy this property as follows; If we simply consider that h =
arg min
f∈Fδ
∥fw −f∥∞, then there is a possibility that h(x) /∈Ii for x ∈X such that fw(x) ∈I. This
will cause a problem when approximating the ECE using h. If h(x) /∈Ii, it significantly changes
the estimation of the conditional expectation in each bin, leading to a larger change of the ECE. To
avoid this, we consider a set of functions such that for each i = 1, . . . , B, for any f ∈Fδ, we define
f(x) := max

min

f(x), i
B

, i −1
B
+ ϵ

,
(117)
where 0 < ϵ < 1/2δ. We refer to this set of clipped functions as Fi for i = 1, . . . , B. The parameter
ϵ is introduced so that the clipped value does not take the boundary value between the i-th bin and
(i −1)-th bin. Since we set 0 < ϵ < 1/2δ, the parameter ϵ does not affect the bias analysis below.
Then, we define the function
f(x) :=
B
X
i=1
fi(x)1fw(x)∈Ii,
(118)
where fi := arg min
f∈Fi
supx∈Xi |fw(x) −f(x)| and Xi := {x ∈X|fw(x) ∈Ii}. (if Xi = ϕ, we
do not need to consider fi and any function in Fi can be used.) Note that under this definition,
supx∈Xi |fw(x) −fi(x)| < δ holds for each i = 1, . . . , B. This can be confirmed as follows;
given any fw, we assume that there exists a point x∗∈X (the following discussion still holds when
there are multiple such x∗and we refer to them as X ∗) that achieves supx |fw −h|∞. If fw(x∗) ∈
Ii, by the definition of Fi, |fi(x∗) −fw(x∗)| ≤max

min
h(x∗), i
B

, i−1
B + ϵ

−fw(x∗)| ≤
|h(x∗) −fw(x∗)| ≤δ. For x, which does not achieves supx |fw(x) −h(x)∥∞(that is, x /∈X ∗) and
satisfies fw(x) ∈Ii, then we have |fi(x) −fw(x)| ≤max

min
h(x), i
B

, i−1
B + ϵ

−fw(x)| ≤
|h(x) −fw(x)| ≤δ. So for any x ∈Xi, |fw(x) −fi(x)| < δ holds. Thus the function f defined in
Eq. (117) satisfies that for any x, if fw(x) ∈Ii, then f satisfies f(x) ∈Ii and ∥f −fw∥∞< δ.
With these settings, we consider replacing the functions in the total bias defined by above. Here in
after, we set δ < 1/B. Then the error in the TCE by replacing fw with f is given as
|E|E[Y |fw] −fw| −E|E[Y |f] −f|| ≤(1 + L)δ,
(119)
which is obtained by the Lipschitz assumption.
32


<!-- page 33 -->
Next, we consider the error in the ECE by replacing fw with f is given as follows;
|ECE(fw, Str) −ECE(f, Str)|
= |
B
X
i=1
|E(X,Y )∼ˆStr(Y −fw(X)) · 1fw(X)∈Ii| −
B
X
i=1
|E(X,Y )∼ˆStr(Y −f(X)) · 1f(X)∈Ii||
= |
B
X
i=1
|E(X,Y )∼ˆStr(Y −fw(X)) · 1fw(X)∈Ii| −
B
X
i=1
|E(X,Y )∼ˆStr(Y −f(X)) · 1fw(X)∈Ii||
≤|
B
X
i=1
|E(X,Y )∼ˆStr(f(X) −fw(X)) · 1fw(X)∈Ii||
=
B
X
i=1
|Ii|
ne

1
|Ii|
n
X
m=1
1fw(xm)∈Iifw(xm) −1
|Ii|
n
X
m=1
1fw(xm)∈Iif(xm)

≤
B
X
i=1
|Ii|
n
δ|Ii|
|Ii|
≤δ.
(120)
where |Ii| := Pn
m=1 1fw(xm)∈Ii and we used the fact that P
i |Ii| = n and 1fw(X)∈Ii = 1f(X)∈Ii
for any x by definition.
Then the total bias is upper bounded by using f as follows;
ER,Str|TCE(fW ) −ECE(fW , Str)|
= ER,Str|TCE(fW ) −TCE(f) + TCE(f) −ECE(fW , Str)|
≤(1 + L)δ + ER,Str|TCE(f) −ECE(f, Str)| + |ECE(f, Str) −ECE(fW , Str)|
≤(2 + L)δ + ER,Str|TCE(f) −ECE(f, Str)|.
(121)
Since the second term in the above represents the total bias of f, using Theorem 5
ER,Str|TCE(f) −ECE(f, Str)| ≤1 + L
B
+
v
u
u
t8

eCMI(˜l) + B log 2

n
,
(122)
where we replace fw appearing in the bound appearing Theorem 5 with f defined above.
From the data processing inequality, we can upper bound the eCMI(˜l) by the fCMI of f. Then such
fCMI is bounded by the entropy of f as follows
eCMI(˜l) ≤fCMI(f) = I(f( ˜X); U| ˜Z) ≤H[f( ˜X)| ˜Z] −H[f( ˜X)|U, ˜Z] ≤H[f( ˜X)| ˜Z],
(123)
where we used the definition of mutual information and the conditional entropy of f given U
(H[f( ˜X)|U, ˜Z]) is larger than 0 because f( ˜X) is the discrete random variable and the entropy of
the discrete random variable is always larger than 0. We can confirm that f( ˜X) is a discrete random
variable since f belongs to the function which is defined by the δ covering and the input ˜X is 2n.
Then we upper bound H[f( ˜X)| ˜Z] by the log of the number of distinct values that are represented
by Eq. (118) given 2n inputs of supersamples ˜Z.
We refer to it as N.
Define d2n(f, g) :=
maxi∈[2n] |f(xi) −g(xi)| for f, g ∈F.
The δ-covering number of F with respect to d2n is
denoted as N(F, d2n, δ), and we define N(F, δ, 2n) := supxn∈X n N(F, d2n, δ). It is known
that N(F, δ, 2n) ≤N(F, ∥· ∥∞, δ) in general [37].
We focus on the fact that f defined in
Eq. (118) is the element of F1 + F2 + . . . , +FB, where F1 + F2 implies the set of functions
that is {f + g|f ∈F1, g ∈F2}. Note that the covering number of sum of two functions are upper
bounded by the multiplication of each covering number, that is
N(F1 + F2, δ, 2n) ≤N(F1, δ/2, 2n)N(F2, δ/2, 2n)
(124)
and thus, we have
N(F1 + F2 + . . . , +FB, δ, 2n) ≤
B
Y
i=1
N(Fi, δ/B, 2n) ≤(N(Fi, δ/B, 2n))B.
(125)
33


<!-- page 34 -->
Thus we have
N ≤
B
Y
i=1
(N(F, δ, 2n)) ≤(N(F, ∥· ∥∞, δ/B))B.
(126)
In conclusion, we have that
eCMI(˜l) ≤log N ≤B log N(F, ∥· ∥∞, δ/B).
(127)
Combining these we have
ER,StrBiastot(fW , Str) ≤1 + L
B
+ (2 + L)δ +
r
8 (B log N(F, ∥· ∥∞, δ/B) + B log 2)
n
= 1 + L
B
+ (2 + L)δ +
r
8B log 2N(F, ∥· ∥∞, δ/B)
n
.
(128)
Finally, as for the order discussion, for example, we can obtain N(F, ∥· ∥∞, δ) ≍
 L0
δ
d when fw
is a d-dimensional parametric function that is L0-Lipschitz continuous (L0 > 0) [37], leading to the
following upper bound:
ER,StrBiastot(fW , Str) ≲3 + 2L
B
+
r
8dB log 2L0B2
n
,
(129)
where we set δ = O(1/B). This bound is minimized when B = O(n1/3), resulting in a bias of
O(log n/n1/3).
Here we provide additional discussion about the metric entropy. The bound based on the metric
entropy depends on the model’s dimensionality, making them unsuitable for large models such as
neural networks. Some existing studies [15, 17] present the upper bound of the eCMI and fCMI
by the dimension-independent complexities, such as the VC dimension for binary classification and
connecting IT theory to UC theory.
Inspired by these results, here we provide the upper bound of eCMI and fCMI using such dimension-
independent complexities. As provided in the lower bound analysis above, since TCE estimation is
similar to the nonparametric regression, we use the fat-shattering dimension [1] to upper bound the
eCMI. Specifically, from Lemma 3.5 in Alon et al. [1] in if our model class fw(·) satisfies δ/4-fat
dimension with dδ/4 for δ ∈[0, 1], we have
eCMI ≤fCMI = O

dδ/4 log
n
dδ/4δ log
 n
δ2

(130)
which results in the dimension-independent upper bound. To evaluate the fat-shattering dimension
for specific models, see Bartlett and Maass [2] for the details.
E.4
Proof of Theorem 7 (recalibration)
Proof. Recall the definition of the recalibration. Here we show the expression when we use the
training dataset Str:
hI,Str(x) =
B
X
i=1
ˆµi,Str · 1fw(x)∈Ii,
(131)
ˆµi,Str :=
Pn
m=1 ym · 1fw(xm)∈Ii
Pn
m=1 1fw(xm)∈Ii
.
(132)
The proof is similar to the proof of Theorem 8. We use Eqs. (87) and (86), let Str = {Zm}n
m=1,
then we have
34


<!-- page 35 -->
ER,StrE[|E[Y |hI,Str(x)] −hI,Str(x)]
= ER,Str

B
X
i=1
EZ′′=(X′′,Y ′′)

(Y ′′ −ˆµi,Str) · 1fW (X′′)∈Ii


= ER,Str
B
X
i=1
E{Z′
m}n
m=1
1
n
n
X
m=1

(Y ′
m −ˆµi,Str) · 1fW (X′
m)∈Ii


≤
E
R,{Zm}n
m=1,{Z′m}n
m=1
B
X
i=1

1
n
n
X
m=1

(Y ′
m·1fW (X′m)∈Ii −ˆµi,Str · 1fW (Xm)∈Ii +ˆµi,Str · 1fW (Xm)∈Ii −ˆµi,Str · 1fW (X′m)∈Ii)


=
E
R,{Zm}n
m=1,{Z′m}n
m=1
B
X
i=1

1
n
n
X
m=1

(Y ′
m·1fW (X′m)∈Ii −Ym · 1fW (Xm)∈Ii)+ˆµi,Str(1fW (Xm)∈Ii −1fW (X′m)∈Ii)


≤
E
R,{Zm}n
m=1,{Z′m}n
m=1
B
X
i=1

1
n
n
X
m=1

(Y ′
m · 1fW (X′m)∈Ii −Ym · 1fW (Xm)∈Ii)


+
E
R,{Zm}n
m=1,{Z′m}n
m=1
B
X
i=1

1
n
n
X
m=1

(1fW (Xm)∈Ii −1fW (X′m)∈Ii)


(133)
where we used the Jensen inequality first and the used the triangle inequality and used the fact that
|ˆµi,Str| ≤1 by the definition of UMB in the next inequality.
We then rewrote the above in the CMI setting, we have
ER,StrE[|E[Y |hI,Str(x)] −hI,Str(x)]
≤ER, ˜
Z,U
B
X
i=1

1
n
n
X
m=1
( ˜Ym, ¯Um · 1fA( ˜
ZU ,R)( ˜
Xm, ¯
Um)∈Ii −˜Ym,Um · 1fA( ˜
ZU ,R)( ˜
Xm,Um)∈Ii

+ ER, ˜
Z,U
B
X
i=1

1
n
n
X
m=1
1fA( ˜
ZU ,R)( ˜
Xm, ¯
Um)∈Ii −1fA( ˜
ZU ,R)( ˜
Xm,Um)∈Ii

≤
s
2(I(∆1(U, ˜Z, A( ˜ZU, R)); U| ˜Z) + B log 2)
n
+
s
2(I(∆2(U, ˜Z, A( ˜ZU, R)); U| ˜Z) + B log 2)
n
.
(134)
where
∆1(U, ˜Z, A( ˜ZU, R)) :=
B
X
i=1

1
n
n
X
m=1
( ˜Ym, ¯Um ·1fA( ˜
ZU ,R)( ˜
Xm, ¯
Um)∈Ii −˜Ym,Um ·1fA( ˜
ZU ,R)( ˜
Xm,Um)∈Ii)

(135)
∆2(U, ˜Z, A( ˜ZU, R)) :=
B
X
i=1

1
n
n
X
m=1
(1fA( ˜
ZU ,R)( ˜
Xm, ¯
Um)∈Ii −1fA( ˜
ZU ,R)( ˜
Xm,Um)∈Ii)

(136)
where the last line is we repeat the proof of Theorem 8. The proof of Theorem 8 has discussed the
loss composed of the indicator function, we can use exactly the same proof procedure. The only
difference is the CMI; here we consider the eCMI which uses above ∆1 and ∆2 as the random
variables, and their CMIs are the conditional mutual information between ∆1 and ∆2.
35


<!-- page 36 -->
From Eq. (134), we can further simplify the upper bound using the data processing inequality,
I(∆1(U, R, ˜Z); U| ˜Z), I(∆2(U, R, ˜Z); U| ˜Z) ≤eCMI
(137)
eCMI := I(l(A( ˜ZU, R), ˜Z, B); U| ˜Z),
(138)
l(A( ˜ZU, R), z, B) := (1fA( ˜
ZU ,R)(x)∈I1, . . . , 1fA( ˜
ZU ,R)(x)∈IB).
(139)
We then have
ER,StrE[|E[Y |hI,Str(x)] −hI,Str(x)] ≤2
r
2(eCMI + B log 2)
n
≤2
r
2(fCMI + B log 2)
n
.
(140)
In the numerical experiments, we use the tighter version, which appears in the proof above;
Corollary 4. Under the same setting and assumptions as Theorem 7, we have
ER,StrE[|E[Y |hI,Str(x)] −hI,Str(x)]
≤
s
2(I(∆1(U, ˜Z, A( ˜ZU, R)); U| ˜Z) + B log 2)
n
+
s
2(I(∆2(U, ˜Z, A( ˜ZU, R)); U| ˜Z) + B log 2)
n
,
(141)
where
∆1(U, ˜Z, A( ˜ZU, R)) :=
B
X
i=1

1
n
n
X
m=1
( ˜Ym, ¯Um ·1fA( ˜
ZU ,R)( ˜
Xm, ¯
Um)∈Ii −˜Ym,Um ·1fA( ˜
ZU ,R)( ˜
Xm,Um)∈Ii)

(142)
∆2(U, ˜Z, A( ˜ZU, R)) :=
B
X
i=1

1
n
n
X
m=1
1fA( ˜
ZU ,R)( ˜
Xm, ¯
Um)∈Ii −1fA( ˜
ZU ,R)( ˜
Xm,Um)∈Ii)
 .
(143)
In the main paper, we further upper bound eCMIs by fCMI, which is followed by the data processing
inequality.
E.5
Recalibration when using test data
Here we show the bias of the recalibration when the test (recalibration data) is used.
Corollary 5. Conditioned on W = w, under the same assumption as Theorem 2, we have
ESreE[|E[Y |hI,Sre(X)] −hI,Sre(X)] ≤
p
2B log 2/(nre −B) + 2B/(nre −B)
(144)
Proof. The recalibration bias corresponds to the statistical bias because from Theorem 2, we have
ESreBiasstat(h, Sre) = ESre|TCE(hI) −ECE(hw, Sre)|
≤
p
2B log 2/(nre −B) + 2B/(nre −B).
(145)
where hI is the conditional expectation of hI,Sre given bins I. Note that by the definition of
hI,Sre, from the tower property hI(x) = hI,Sre(x) holds since we take the expectation in each
bin. Thus, by definition ESreTCE(hI) = ESreE[|E[Y |hI,Sre(X)] −hI,Sre(X)] holds. Moreover
ESreECE(hw, Sre) = 0 by definition, since this is the definition of the recalibrated function. Thus,
we obtain the result.
F
Further discussion
We have presented the results of our analyses of (i) the total bias in estimating the TCE and (ii)
the generalization error analysis for the ECE and the TCE thus far. In this section, we explain the
difference between our study and the existing work in the calibration context.
36


<!-- page 37 -->
F.1
Discussion about the assumption
Here we discuss the necessity of Assumption 1. The reasons of using this assumption are two-fold:
i) : The first purpose is that we want to use the results in Gupta and Ramdas [12], which analyzes the
statistical bias of the UMB. Their proofs use the existence of the density of fw(x), so Assumption 1
cannot be eliminated. ii) : The other purpose is that by Assumption 1, we want to use the fact
that {fw(xm)}nte
m=1 in xm ∈Ste takes the distinct values almost surely (same for {fw(xm)}n
m=1 in
xm ∈S).
Regarding i), we used results in Gupta and Ramdas [12] to prove the result of UMB in Eq. (9) of
Theorem 2 and the result of UMB of Theorem 3. Thus, the results using these theorems require
Assumption 1. They correspond to the results related to the bias of UMB. So the results of UWB
essentially do not require this assumption.
The situation becomes complicated when considering the generalization error analysis. Our analysis
uses the IT-based approach and does not use the results in Gupta and Ramdas [12], so we do not
need Assumption 1 regarding i). However, if all {fw(xm)}n
m=1 in xm ∈S takes the same value, we
cannot construct the bins in UMB. So when considering the training data reuse, we need Assump-
tion 1 to construct the bins of UMB. However, we remark that we can replace Assumption 1 with
the assumption that “we assume that {fw(xm)}n
m=1 in xm ∈S takes distinct values almost surely”.
When considering the UWB, we do not suffer from such troubles since we simply split the interval
[0, 1] with equal width as the b-th interval is given as ((b −1)/B, b/B]. However, there might be
a chance that all the {fw(xm)}nte
m=1 in xm ∈Ste takes b/B, then the coefficients of the bound
changes. Recall that our proof uses the bounded difference property when upper bounding the
exponential moment, for example, in Eq. (93) and Eq. (94). That estimation is based on the fact
that {fw(xm)}n
m=1 in xm ∈Str takes different values. So if all the f takes the same value, the
upper bound of the bounded difference will change, which results in the different coefficients in our
bound, although we can proceed with the proof in the same way.
For these reasons, we decided to impose Assumption 1 for all the statements. As we discussed
above, if we focused on the specific setting, such as UWB and UMB, then there is room to eliminate
or replace the assumption.
Next, we discuss the necessity of Assumption 2. Estimating TCE involves nonparametric regres-
sion of E[Y |f(X) = v]. For finite samples, smoothness assumptions like Lipschitz continuity are
required; without them, small changes in v could cause large variations in label outcomes, making
estimation impossible [26]. Such smoothness assumptions are standard in nonparametric regres-
sion, including kernel-based ECE. Therefore, without these assumptions, increasing the sample size
would not ensure that training (or test) ECE converges to TCE. As noted in our minimax lower
bound discussion, the absence of smoothness (β →0) leads to increasing bias.
In practice, Assumption 2 is reasonably mild. For example, if label distributions follow a Bernoulli
distribution with the mean depending on the input x, Assumption 2 is satisfied [47]. This is a rel-
atively weak assumption in binary classification. Moreover, existing studies have shown that many
common benchmark datasets are consistent with this assumption (e.g., [48]). Additionally, our nu-
merical experiments with the data used in this study confirmed that the conditions for smoothness,
such as non-diverging first derivatives, are indeed satisfied (see Figure 6 in Appendix H.4). This
supports the robustness of our assumptions and the applicability of our methods in real-world sce-
narios.
Finally, regarding the assumption ne ≥2B, it is crucial for UMB, as it guarantees the proper
construction of bins. In UMB, we first use B samples to build the bins. We then partition the
remaining ne −B samples evenly across these bins. If ne ≥2B, we have ne −B ≥B, preventing
equal distribution of samples, thereby rendering UMB inapplicable. Conversely, UWB does not
require this assumption since it divides the [0, 1] interval into equal-width bins.
F.2
Discussion about our proof techniques
Here we discuss our proof techniques. In our proof for UWB, our proof technique does not heavily
depend on the binning construction method, so we can apply our technique to other than UWB
and UMB. The important ingredients are the boundedness of y and f(x) and the property of the
indicator function. However, our proof builds on the reformulation of Eqs. (24) and (27) this can be
37


<!-- page 38 -->
a restriction for some settings. For example, when we consider higher-order ECEs defined as
ECE(fw, Se) :=
B
X
i=1
pi| ¯fi,Se −¯yi,Se|p,
(146)
with p > 1, which can not be reformulated like Eqs. (24) and (27), and thus our proof technique
cannot be applicable.
On the other hand, as we introduce in the above, the technique of Gupta and Ramdas [12] can apply
to ECEs with p > 1. However, the drawback is that their technique can only apply to UWB without
training data reuse.
Next, we discuss our results with Wang et al. [39]. The eCMI appearing in Theorem 4 closely aligns
with the ∆L bound (loss difference) of Theorem 1 of reference Wang et al. [39]. Specifically, the
eCMI term in Theorem 4 is not based on the value of ECE itself. Instead, it is derived from the
difference between the test data ECE and the training data ECE. This approach aligns with the ∆L
(loss difference) as shown in Theorem 1 of reference Wang et al. [39].
However, extending our bounds using the techniques of Wang et al. [39] presents significant chal-
lenges. The ∆L bound in Wang et al. [39] defines the loss gap for a single data index i as ∆Li and
utilizes the symmetry of each individual index to derive fast rate bounds, as demonstrated in Theo-
rem 4.3. In contrast, our bound requires treating all n indices simultaneously. This necessity arises
because ECE is a nonparametric estimator that uses all n indices, unlike usual losses such as the 0-1
loss, where an estimator can be constructed using a single index. Consequently, the techniques from
Wang et al. [39] that utilize the symmetry of a single index are not applicable to our context, making
it difficult to employ the methods from Wang et al. [39].
F.3
Comparison of our bound with existing and trivial bounds
Here we discuss the order of our generalization error bias in more depth. Recall that our Theorem 4
is
ER,Str,Ste|ECE(fW , Ste)−ECE(fW , Str)| ≤
r
8(eCMI + B log 2)
n
,
(147)
and the important property is that the bound is of order O(
p
B/n) if we neglect the order of eCMI.
Here we see that when we use existing Theorem 1 directly, the resulting bound is O(B/√n). Recall
that
ECE(f, Str) =
B
X
i=1
|E(X,Y )∼ˆStr(Y −fw(X)) · 1fw(X)∈Ii|,
TCE(fI) =
B
X
i=1
|E(X,Y )∼D(Y −fw(X)) · 1fw(X)∈Ii|.
where ˆStr is the empirical distribution of the training dataset. Thus
EStr,Ste,R|
B
X
i=1
|E(X,Y )∼D(Y −fW (X)) · 1fW (X)∈Ii| −
B
X
i=1
|E(X,Y )∼ˆStr(Y −fW (X)) · 1fW (X)∈Ii||
≤EStr,Ste,R|
B
X
i=1
|E(X,Y )∼D(Y −fW (X)) · 1fW (X)∈Ii −E(X,Y )∼ˆStr(Y −fW (X)) · 1fW (X)∈Ii||
= EStr,Ste,R
B
X
i=1
|E(X,Y )∼D(Y −fW (X)) · 1fW (X)∈Ii −E(X,Y )∼ˆStr(Y −fW (X)) · 1fW (X)∈Ii|
≤
B
X
i=1
r
2
n(eCMI(˜li) log 2)
≤B
r
2
n(fCMI + log 2),
(148)
38


<!-- page 39 -->
where we used the triangle inequality in the second line and from the third line to the fourth line,
we applied Eq. (5) in Theorem 1 by fixing the binning index i and eCMI is where eCMI(˜li) =
I(˜li; U| ˜Z)
˜li(U, R, ˜Z)
:= | 1
n
n
X
m=1
( ˜Ym, ¯Um −fA( ˜
ZU,R)( ˜Xm, ¯Um)) · 1fA( ˜
ZU ,R)( ˜
Xm,Um)∈Ii
−1
n
n
X
m=1
( ˜Ym,Um −fA( ˜
ZU,R)( ˜Xm,Um)) · 1fA( ˜
ZU ,R)( ˜
Xm,Um)∈Ii|
(149)
Thus, this proof is simple compared to our proof of Theorem 4, but results in a worse dependency
on B. In our proof, we used the property of the binning and indicator function of the loss function
explicitly, which results in better dependency. On the other hand, when deriving Eq. (148), we do
not use such properties, and thus results in worse dependency on B.
F.4
Discussion about the order of eCMI and fCMI
Here we discuss when eCMI and fCMI can be controlled theoretically.
As discussed in Section 4, from data processing inequality [6], we have that eCMI ≤fCMI ≤
I(W; S). Since fCMI does not depend on B, and the dependency on B of fCMI, eCMI is not a
problem.
Here, we cite the classical result about I(W; S). Clarke and Barron [5] (see also Rissanen [30],
Haussler and Opper [16]) clarified that the growth rate of MI can be controlled as follows: if w takes
a value in a d-dimensional compact subset of Rd and p(y|x; w) is smooth in w, then as n →∞, we
have
I(W; S) = d
2 log n
2πe + h(W) + E log detJ + o(1),
where h(W) is the differential entropy of W, and J is the Fisher information matrix of p(Y |X; W).
Moreover, Steinke and Zakynthinou [34] introduced the CMI that satisfies fCMI ≤CMI discussed
the CMI is upper bounded by the various notions of stability. For example, if the training algorithm
satisfies
√
2ϵ-differentially private (DP) algorithm, then CMI is upper-bounded by ϵn. So this ϵ is
controlled by the DP algorithm, then our eCMI can also be controlled appropriately. For example,
Xu and Raginsky [44] discussed that the Gibbs algorithm satisfies O(1/n)-DP when the loss takes
value [0, 1]. Thus, such Gibbs algorithms can control our eCMI moderately.
Steinke and Zakynthinou [34] also clarified that if the algorithm is δ stable in total variation distance,
then CMI is upper bounded by δn. From the stability perspective, Mou et al. [28] showed that
SGLD satisfies O( T
n2 ) stability in the Hellinger distance and T is the iteration number of the SGLD
algorithm. Thus, this implies that when T is small, the eCMI of SGLD can be very small. Recently,
Farghly and Rebeschini [8] and Futami and Fujisawa [9] showed that under the certain non-convexity
assumption, SGLD satisfies the Wasserstein and KL stability of the order of O(1/n), which also
result in the eCMI of SGLD.
F.5
Relation to the existing study of calibration
Here we discuss other existing work, which is not shown in the main paper mainly due to the
space limitation. First, we compare our result and existing analysis by Gupta and Ramdas [12] in
Appendix F.2 in detail.
We discuss in Appendix D.2 how our proof technique improves the trivial dependency of O(B)
to our O(
√
B) for the ECE with the test dataset. We show a similar discussion for the ECE with
training dataset reuse.
Kulynych et al. [22] also discussed the relation between generalization and calibration. However,
there are two distinct differences from theirs; One is that they only discuss the statistical bias not
consider the binning bias. The other is that their statistical bias is of O(B) while ours is O(
√
B),
which is a significant improvement.
39


<!-- page 40 -->
Carrell et al. [4] numerically evaluated the generalization gap of the calibration, which is close to
the statistical bias in our training reuse setting. They focused on the numerical aspects and statistical
bias, while ours focused on the theoretical analysis and focuses on both the binning and statistical
biases.
Gruber and Buettner [10] studied the various statistics related to calibration error. While our study
rigorously analyzes both the binning and statistical bias, their work focuses on the asymptotic set-
tings and has not derived the dependency of B and n.
There is an additional comparison of our analysis with Gupta and Ramdas [12] in Appendix D.6
F.6
Discussion about the lower bound
Here, we discuss the lower bound of the bias when estimating the TCE from the following two
viewpoints; the TCE estimation can be seen as (i) estimating a parameter in each bin, and (ii)
estimating a one-dimensional function in a pointwise. To understand this, we start deriving the
following lower bound by using Jensen inequality:
TCE(fw) ≥|E[Y −fw(X)]|.
(150)
This bound suggests that estimating E[Y ] = fw(x) achieves the small TCE. From the classical the-
ory, for any distribution D, the lower bound of a parameter estimation bias is 1/√n [37]. Eq. (150)
corresponds to the setting of B = 1 bin to estimate the conditional expectation in the ECE. With
these observations, we discuss the statistical bias of UMB. In UMB, we estimate the conditional
expectation using m = n/B samples in each bin and this is a parameter estimation problem. Thus
this leads to O(1/√m) bias from the classical theory. On the other hand, we derived the statistical
bias O(
p
B/n) for UMB, thus it is optimal when viewed as the parameter estimation.
However, using a constant function fw(x) = E[Y ], which achieves the small TCE, is useless in
practice. Our original motivation is to measure the perfect calibration, which requires estimating
the conditional expectation E[Y |fw(x) = p] for the interval p ∈[0, 1] in a pointwise, and this is a
function estimation problem. Thus, the bins used in the ECE adjust that whether estimating ECE is
close to the parameter estimation or the function estimation. Then this trade-off is captured by the
binning bias in our analysis. So the total bias represents such a trade-off whether the problem is the
parameter or function estimation.
From the classical theory of Fano’s method [37], when we estimate the Lipschitz function with a
closed interval, it achieves a lower bound O(1/nd+2), with d as the input dimension of the function.
In the calibration, the input is the probability p and thus d = 1. Since we derived that the bias
of the ECE is O(1/n1/3), it achieves the minimax rate if the underground conditional expectation
E[Y |v = p] satisfies Lipsthitz continuity.
Moreover, when the d-dimensional target function satisfies stronger smoothness assumption, β-
H¨older continuity, we suffer the bias of O(1/n
β
2β+d ) [36]. So if E[Y |v = p] satisfies such condi-
tions, the lower bound of the bias should be O(1/n
β
2β+1 ), however as discussed in Appendix D.5,
the binning method cannot achieve this rate and thus cannot utilize the smoothness of the data dis-
tribution.
F.7
Additional discussion about the lower bound
Here we discuss additional points regarding the TCE bias estimation. Conditioned on W = w, let
v = fw(x) and the distribution induced by p(x) by fw(x) over v as p(v). We express the support of
p(v) as V ⊂[0, 1]. Let g(v) = E[Y = 1|fw(x) = v] = Pr[Y = 1|v]. Let G be a class of candidate
conditional probability functions over V and every candidate g ∈G satisfies 0 ≤g(v) ≤1 for all
v ∈V. Let us write the L2 minimax error as
R(G; n) := inf
ˆg sup
g∈G
E|ˆg(V ) −g(V )|2,
(151)
where ˆg is over all valid estimators for g using n samples (Xm, Ym)n
m=1 and the expectation is taken
with respect to true g. The L2 and higher order minimax rate has been shown in [45] by using the
Yang-Barron method [46], which is the mutual information-based approach stemming from Fano’s
method. To cite this result, we need additional assumptions that control the mutual information [45];
40


<!-- page 41 -->
Assumption 3. Assume that G has at least one member g∗that is bounded away from 0 and 1, i.e.,
there exist constants 0 < c1 ≤c2 < 1 such that c1 ≤g∗≤c2.
Here we assume that G is a set of functions that satisfies the β-Lipschitzness; for any g ∈G, there
exists positive constant C that satisfies |g(v) −g(v′)| ≤C|v −v′|β for any v, v′ ∈V.
According to Lemma 1 in [45], if β-Lipschitzness and we have
R(G; n) ⪰n−2β/(2β+1).
(152)
Thus, if we consider the histogram-based estimator for ˆg
ˆgbin(v) :=
B
X
i=1
¯yi,Ste · 1v∈Ii,
(153)
which is used in the definition of the binning ECE, then
sup
g∈G
E|ˆgbin(V ) −g(V )|2 ≥inf
ˆg sup
g∈G
E|ˆg(V ) −g(V )|2 ⪰n−2β/(2β+1).
(154)
Thus we can obtain the lower bound of the recalibrated function from true conditional expectation.
Here we discuss how this lower bound is related to the total bias of the ECE. Let us define the
semi-metric
ρ(g, g′) := |E[|g(V ) −V |] −E[|g′(V ) −V |]|.
(155)
We can easily confirm that ρ satisfies the positivity and triangle inequality, so this is the semi-metric.
We can show the following relation for the total bias and this ρ as follows; recall that we can express
the TCE as
TCE(fw) = E[|g(V ) −V |]
(156)
and
ESteECE(fw, Ste) = E[|(ˆgbin(V ) −¯V (V ) + V ) −V |]
(157)
where
¯V (v) =
B
X
i=1
¯vi,Ste · 1v∈Ii,
¯vi,Ste :=
1
|Ii|
nte
X
m=1
1vm∈Iivm
where vm = fw(xm). Thus, the total bias is given as the total bias
ESteBiastot(fw, Ste) = ESte|TCE(fw) −ECE(fw, Ste)|
≥|TCE(fw) −ESteECE(fw, Ste)|
= |EV [|g(V ) −V |] −EV [|(ˆgbin(V ) −¯V (V ) + V ) −V |]| = ρ(g, ˜g) (158)
where ˜g := ˆgbin(V ) −¯V (V ) + V . Thus, by studying the risk under ρ, we can study the total bias.
On the other hand, this ρ is smaller than the L1 distance
E |ˆg(V ) −g(V )| ≥|E[|ˆg(V ) −V |] −E[|V −g(V )|]| = ρ(g, ˆg)
(159)
by the triangle inequality. Note that L1 distance is smaller than L2 distance, the above minimax
result for ˆg and g in Eq. (154) is insufficient to understand the bias of TCE and ECE from the lower
bound. Instead, we introduce different lower bound given as follows. Conditioned on W = w,
under the β-Lipschitzness for E[Y |fw(x) = v] and Assumption 3, we have
sup
g∈G
E[
|g(V ) −V | −|ˆgbin(V ) −¯V (V )|
2] ⪰n−2β/(2β+1).
(160)
And We can further obtain
sup
g∈G
E[
|g(V ) −V | −|ˆgbin(V ) −¯V (V )|

∞] ⪰n−β/(2β+1).
(161)
where E| · |∞is the maximum of the integrand. These lower bounds imply the pointwise lower
bound of the difference of the conditional expectation and V . Using these lower bounds, we study
the difficulty of estimating the TCE at each V = v = fw(x).
41


<!-- page 42 -->
Table 2: Model architecture of CNN on the MNIST experiments.
Model architecture of CNN (same as Harutyunyan et al. [15])
(1st layer) Convolutional
32 filters, 4 × 4 kernels, stride 2, padding 1, batch normalization, ReLU
(2nd layer) Convolutional
32 filters, 4 × 4 kernels, stride 2, padding 1, batch normalization, ReLU
(3rd layer) Convolutional
64 filters, 3 × 3 kernels, stride 2, padding 0, batch normalization, ReLU
(4th layer) Convolutional
256 filters, 3 × 3 kernels, stride 1, padding 0, batch normalization, ReLU
Fully connected
128 units, ReLU
Fully connected
2 units, Linear activation
Proof. First, we focus on the relation
sup
g∈G
E[
|g(V ) −V | −|ˆgbin(V ) −¯V (V )|
2] ≥inf
ˆg sup
g∈G
E[
|g(V ) −V | −|ˆg(V ) −¯V (V )|
2]. (162)
We then estimate the minimax L2 estimation error under the semimetric ˜ρ(g, g′) := E||g(V )−V |−
|g′(V ) −V ||2. The proof is the same as that of Lemma 1 in [45], which uses Yang Barron method.
The difference is to derive the packing number under the semimetric ˜ρ not the L2 distance. However
˜ρ is nothing but the shifted version of the L2 distance. Thus the order of the packing number is
the same as that of L2 distance. Note that since G is the set of positive functions thus, taking the
absolute of g(V ) −V does not change the order. Then by using Fano’s method, we obtain the result
for L2 lower bound. The version of ∥· ∥∞can be proved in the same way.
We finally remark that the above pointwise gap is larger than the total bias
q
E[
|g(V ) −V | −|ˆgbin(V ) −¯V (V )|
2]
(163)
≥ESte[
E|g(V ) −V | −E|ˆgbin(V ) −¯V (V )|
] = ESteBiastot(fw, Ste)
(164)
where we used the relation of that L2 is larger than L1 and used the Jensen inequality.
F.8
Relation to the multiclass settings
Although our study focuses on binary classification, we can extend it to multi-class settings. In
existing work [23, 10], the top-label calibration error (top ECE) has been proposed as a measure
for multi-class calibration. For instance, in a K-class classification problem, we obtain predictions
of each label by the final softmax layer in neural networks. We assume that fw(x) ∈RK predicts
the label by C := argmaxkfw(X)k, where fw(X)k represents the model’s confidence of the label
k ∈[K]. Then top ECE is defined using the highest prediction probability output by fw: E|P(Y =
C|fw(X)C) −fw(X)C|. By considering binning only for the top score, we can compute the ECE
in a similar way as binary classification. In this case, since we focus only on the top label, we can
treat top-binning ECE in the same way as binary classification, leading to the same generalization
and total bias bounds. Our results, therefore, offer flexibility to analyze the widely used top-label
calibration error in multi-class settings.
G
Experimental settings
In this section, we summarize the details of our experiments conduced in Sections 3 and 6.
G.1
Experiments on the synthetic dataset
For the experiments on the synthetic dataset, we follow Zhang et al. [50] and assume that the distri-
butions of the label Y and the input data X are as follows:
P(Y = 1) = P(Y = 0) = 1
2,
(165)
and
P(X = x|Y = 1) = N(x; −1, 1),
P(X = x|Y = 0) = N(x; 1, 1),
(166)
42


**[Table p42.1]**
| (1st layer) Convolutional | 32 filters, 4 × 4 kernels, stride 2, padding 1, batch normalization, ReLU |
| --- | --- |
| (2nd layer) Convolutional | 32 filters, 4 × 4 kernels, stride 2, padding 1, batch normalization, ReLU |
| (3rd layer) Convolutional | 64 filters, 3 × 3 kernels, stride 2, padding 0, batch normalization, ReLU |
| (4th layer) Convolutional | 256 filters, 3 × 3 kernels, stride 1, padding 0, batch normalization, ReLU |
| Fully connected | 128 units, ReLU |
| Fully connected | 2 units, Linear activation |

[CAPTION] Table 2: Model architecture of CNN on the MNIST experiments.


<!-- page 43 -->
Table 3: Experimental settings on MNIST [25].
Experimental setup for MNIST experiments
Task
4 vs 9 classification
Model
CNN with four layers
Optimizer
Adam with 0.001 learning rate and β1 = 0.9
SGLD with 0.004 learning rate (decaying by a factor 0.9 after each 100 iterations)
Batch size
128 (for Adam) or 100 (for SGLD)
Num. of training samples
[75, 250, 1000, 4000]
Num. of epochs
200
Num. of samples for CMI estimation
5
Num. of samplings for U
10
Num. of recalibration dataset (existing methods)
100
where N(x; m, σ) is the Gaussian distribution with mean m and standard deviation σ. Then, the
probability of Y = 1 given x can be expressed as
P(Y = 1|X = x) =
1
1 + exp(2x).
We also define the prediction models as follows:
z = fw(x) = (z1, z2) =

1
1 + exp(−β0 −β1x),
exp(−β0 −β1x)
1 + exp(−β0 −β1x)

,
where w = {β0, β1} are parameters.
Under these settings, we can calculate the closed-form of the canonical calibration function π(z) =
(π1(z), π2(z)), where
π1(z) =

1 + exp

−2β0 + log(1/z1 −1)
β1
−1
,
π2(z) = 1 −π1(z).
Due to this closed-form calibration function, we can estimate the TCE based on Monte Carlo inte-
gration. In this paper, we use 106 random samples generated from Eqs. (165) and (166) and evaluate
the sample average value of |z1 −π1(z)| as the estimator of TCE. Furthermore, we estimated the
Lipschitz constant L by taking the maximum values of the gradient of π1(z).
G.2
MNIST and CIFAR experiments
Model architectures, datasets, model training process, and implementation:
We summarize
the details of model architectures for CNN, datasets, and model training process in Tables 2-4. Our
experiments were conducted by adapting the code from Harutyunyan et al. [15] 4 to suit our exper-
imental configurations. Consequently, the datasets utilized in this study were normalized in accor-
dance with the implementation provided in the referenced repository. We used NVIDIA GPUs with
32GB memory (NVIDIA DGX-1 with Tesla V100 and DGX-2) for MNIST (SGLD) and CIFAR-10
experiments. We also used CPU (Apple M1) with 16GB memory for the other experiments.
Table 4: Experimental settings on CIFAR-10 [21].
Experimental setup for CIFAR experiments
Task
dog-or-not classification
Model
ResNet-50 pretrained on ImageNet
Optimizer
SGD with 0.01 learning rate and 0.9 momentum
SGLD with 0.01 learning rate (decaying by a factor 0.9 after each 300 iterations)
Batch size
64
Num. of training samples
[500, 1000, 5000, 20000]
Num. of epochs
40
Num. of samples for CMI estimation
2
Num. of samplings for U
5
Num. of recalibration dataset (existing methods)
100
Mutual
information
estimation:
We
cannot
estimate
the
mutual
information
I(l(A( ˜ZS, R), ˜Z, B); S| ˜Z) in our bounds using the approach of Harutyunyan et al. [15] and
4 https://github.com/hrayrhar/f-CMI
43


**[Table p43.1]**
| Task | 4 vs 9 classification |
| --- | --- |
| Model | CNN with four layers |
| Optimizer | Adam with 0.001 learning rate and β1 = 0.9 SGLD with 0.004 learning rate (decaying by a factor 0.9 after each 100 iterations) |
| Batch size | 128 (for Adam) or 100 (for SGLD) |
| Num. of training samples | [75, 250, 1000, 4000] |
| Num. of epochs | 200 |
| Num. of samples for CMI estimation | 5 |
| Num. of samplings for U | 10 |
| Num. of recalibration dataset (existing methods) | 100 |


**[Table p43.2]**
| Task | dog-or-not classification |
| --- | --- |
| Model | ResNet-50 pretrained on ImageNet |
| Optimizer | SGD with 0.01 learning rate and 0.9 momentum SGLD with 0.01 learning rate (decaying by a factor 0.9 after each 300 iterations) |
| Batch size | 64 |
| Num. of training samples | [500, 1000, 5000, 20000] |
| Num. of epochs | 40 |
| Num. of samples for CMI estimation | 2 |
| Num. of samplings for U | 5 |
| Num. of recalibration dataset (existing methods) | 100 |

[CAPTION] Table 3: Experimental settings on MNIST [25].

[CAPTION] Table 4: Experimental settings on CIFAR-10 [21].


<!-- page 44 -->
Hellstr¨om and Durisi [17]. This is because our loss function l(A( ˜ZS, R), ˜Z, B) assumes con-
tinuous values, while these works specifically focus on discrete random variables, such as the
output values of 0-1 loss or the predicted labels of classifiers. Hence, we developed a plug-in
estimator for I(l(A( ˜ZS, R), ˜Z, B); S| ˜Z) [19, 20, 32], which is computed using estimators for the
probability density of l(A( ˜ZS, R)) and Str, as well as their joint probability density, employing
k-nearest-neighbor-based density estimation [27].
The estimation strategy is incorporated into
the sklearn.feature selection.mutual info classif function (we refer to the follow-
ing link:
https://scikit-learn.org/stable/modules/generated/sklearn.feature_
selection.mutual_info_classif.html). We set k = 3 following the default setting of this
function and Kraskov et al. [20], Ross [32].
Standard deviation evaluation of our bounds:
The standard deviation in Table 1 and Figures 2-
5, which is almost unrecognizable due to its small value, are attributed to the randomness inherent in
various experimental settings during model training, i.e., randomness of the training dataset and the
initial model parameters. For example, in the MNIST experiments in Table 3, the standard deviation
of our bound was evaluated under the 5 × 10 models.
H
Additional experimental results
In this section, we show the additional results obtained from our experiments.
H.1
Bound plot on UWB
We show the results of our bound in Eq.(14) using UWB, as shown in Figure3, which was omitted
from the main paper due to page limitations. As we discussed in Section 6, we can see the importance
of the choice of B to obtain nonvacuous bound values. We also observed that our optimal choice,
B = ⌊n1/3⌋, is effective in obtaining nonvacuous bounds.
75
1000
4000
n
0.0
0.5
1.0
1.5
2.0
MNIST (Adam)
UWB; B = 6
UWB; B = 9
UWB; B = 14
UWB; B = 22
UWB; B = 34
UWB; B = 52
UWB; B = ⌊n1/3⌋
ECE gap; B = ⌊n1/3⌋
75
1000
4000
n
0.0
0.5
1.0
1.5
2.0
MNIST (SGLD)
500
5000
20000
n
0.0
0.2
0.4
0.6
CIFAR-10 (SGD)
500
5000
20000
n
0.0
0.2
0.4
0.6
CIFAR-10 (SGLD)
Figure 3: Behavior of the upper bound in Eq. (14) for various B as n increases (mean ± std.).
For clarity, only the results using UWB are shown.
The ECE gap is evaluated by estimating
ER,Str,Ste[|ECE(fW , Ste) −ECE(fW , Str)|]. The ECE gap is shown for B = ⌊n1/3⌋since the
change in B did not result in significant differences.
H.2
Bound plot on recalibration reusing training dataset
We further show the plots of our bound for the recalibration scenario in Figure 4. The relationship
between n, B, and bound values is similar to that observed in the non-recalibration case. Inter-
estingly, the choice of optimal B is crucial for obtaining a small bound value when we conduct
recalibration.
75
1000
4000
n
0.0
0.5
1.0
1.5
2.0
MNIST (Adam)
UMB; B = 6
UMB; B = 9
UMB; B = 14
UMB; B = 22
UMB; B = 34
UMB; B = 52
UMB; B = ⌊n1/3⌋
ECE gap; B = ⌊n1/3⌋
75
1000
4000
n
0.0
0.5
1.0
1.5
2.0
MNIST (SGLD)
500
5000
20000
n
0.0
0.2
0.4
0.6
0.8
CIFAR-10 (SGD)
500
5000
20000
n
0.0
0.2
0.4
0.6
0.8
CIFAR-10 (SGLD)
Figure 4: Behavior of the upper bound in Eq. (141) as n increases for different number of bins (mean
± std.) when using UMB after recalibration.
44

[CAPTION] Figure 3: Behavior of the upper bound in Eq. (14) for various B as n increases (mean ± std.).

[CAPTION] Figure 4: Behavior of the upper bound in Eq. (141) as n increases for different number of bins (mean


<!-- page 45 -->
75
1000
4000
n
10
−5
10
−4
10
−3
10
−2
10
−1
10
0
MNIST (Adam)
UMB; B = 6
ECE gap; B = 6
UMB; B = 9
ECE gap; B = 9
UMB; B = 14
ECE gap; B = 14
UMB; B = 22
ECE gap; B = 22
UMB; B = 34
ECE gap; B = 34
UMB; B = 52
ECE gap; B = 52
UMB; B = ⌊n1/3⌋
ECE gap; B = ⌊n1/3⌋
75
1000
4000
n
10
−3
10
−2
10
−1
10
0
MNIST (SGLD)
500
5000
20000
n
10
−3
10
−2
10
−1
10
0
CIFAR-10 (SGD)
500
5000
20000
n
10
−3
10
−2
10
−1
10
0
CIFAR-10 (SGLD)
Figure 5: Behavior of the upper bound in Eq. (14) for various B as n increases (mean ± std.; log-
scale) when UMB is used. The ECE gap is evaluated by estimating ER,Str,Ste[|ECE(fW , Ste) −
ECE(fW , Str)|]. These results show that the variance of the ECE gap obtained in non-optimal B
settings is large, while the ECE gap in settings based on the optimal B is stable.
H.3
Bound plot for the various number of bins
We examined the ECE gap for various bin sizes using the same setup as Figure 2, and these results
are presented in Figure 5. We plotted them on a log scale to illustrate how the ECE gap and upper
bound behave with different bin sizes. We found that sometimes bins other than the optimal can
yield a better generalization gap. However, the optimal bin size minimizes the total bias as stated in
Theorem 5, not necessarily the generalization gap as in Theorem 4. On the other hand, the optimal
was found to be numerically stable, although, in certain models, high variance was observed for
certain bin sizes, with the ECE gap occasionally not decreasing as increases.
H.4
Empirical verification of Lipschitz continuous for E[Y |f(X)].
As discussed in Appendix F, Assumption2 is generally mild in practice. To assess the empirical
plausibility of the Lipschitz continuity assumption in Assumption 2, we performed a numerical
evaluation using ResNet experiments. Specifically, we checked whether the value of E[Y |f(X)],
estimated via binning, exhibits relatively smooth variations. These results are presented in Figure 6.
The findings indicate that the estimated values fluctuate smoothly to a significant degree, provid-
ing empirical support for the Lipschitz continuity assumption. This strengthens the validity of our
assumptions and confirms the applicability of our methods in practical, real-world scenarios.
0.0
0.2
0.4
0.6
0.8
1.0
f(x)
0.0
0.2
0.4
0.6
0.8
1.0
Estimated E[Y|f(x)]
CIFAR10 (ResNet50; SGD; B = 50)
E[Y|f(x)] estimated
Polynomial Fit
0.0
0.2
0.4
0.6
0.8
1.0
f(x)
0.0
0.2
0.4
0.6
0.8
1.0
Estimated E[Y|f(x)]
CIFAR10 (ResNet50; SGLD; B = 50)
E[Y|f(x)] estimated
Polynomial Fit
Figure 6: Behavior of the estimator of E[Y |f(X)] in the ResNet experiments. The red line is the
(third-order) Polynomial function fitted to the estimated values of E[Y |f(x)].
45


**[Table p45.1]**
| or of the upper boun B is used. The ECE These results show while the ECE gap in t for the various nu ECE gap for variou igure 5. We plotted th different bin size eralization gap. How ecessarily the gener numerically stable, with the ECE gap oc verification of Lips Appendix F, Assum e Lipschitz continui ResNet experiments ning, exhibits relativ cate that the estima port for the Lipschit confirms the applica |  |
| --- | --- |
| CIFAR10 (ResNet50; 1.0 E[Y|f(x)] estimated Polynomial Fit 0.8 E[Y|f(x)] 0.6 Estimated 0.4 0.2 0.0 0.0 0.2 0.4 0 f(x) | SGD; B = 50) CIFAR10 (ResNet50; SGLD; B = 50) 1.0 E[Y|f(x)] estimated Polynomial Fit 0.8 E[Y|f(x)] 0.6 Estimated 0.4 0.2 0.0 .6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 f(x) |


**[Table p45.2]**
| E P | [Y|f(x)] estimated olynomial Fit |  |  |  |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |


**[Table p45.3]**
| E[Y|f(x)] estimated Polynomial Fit |  |  |  |
| --- | --- | --- | --- |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |

[CAPTION] Figure 5: Behavior of the upper bound in Eq. (14) for various B as n increases (mean ± std.; log-

[CAPTION] Figure 6: Behavior of the estimator of E[Y |f(X)] in the ResNet experiments. The red line is the


<!-- page 46 -->
NeurIPS Paper Checklist
1. Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: The claims in the abstract and Section 1 match our theoretical and numerical
claims in the paper.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims
made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the
contributions made in the paper and important assumptions and limitations. A No or
NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how
much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these
goals are not attained by the paper.
2. Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes] .
Justification: Assumptions for each theorem are explicitly shown. Following each theo-
rem, there is a discussion about the theorem’s limitations and implications. Additionally,
Section 7 includes a discussion on the limitations of the entire paper.
Guidelines:
• The answer NA means that the paper has no limitation while the answer No means
that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate ”Limitations” section in their paper.
• The paper should point out any strong assumptions and how robust the results are to
violations of these assumptions (e.g., independence assumptions, noiseless settings,
model well-specification, asymptotic approximations only holding locally). The au-
thors should reflect on how these assumptions might be violated in practice and what
the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was
only tested on a few datasets or with a few runs. In general, empirical results often
depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the ap-
proach. For example, a facial recognition algorithm may perform poorly when image
resolution is low or images are taken in low lighting. Or a speech-to-text system might
not be used reliably to provide closed captions for online lectures because it fails to
handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms
and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to ad-
dress problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by
reviewers as grounds for rejection, a worse outcome might be that reviewers discover
limitations that aren’t acknowledged in the paper. The authors should use their best
judgment and recognize that individual actions in favor of transparency play an impor-
tant role in developing norms that preserve the integrity of the community. Reviewers
will be specifically instructed to not penalize honesty concerning limitations.
3. Theory Assumptions and Proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?
46


<!-- page 47 -->
Answer: [Yes] .
Justification: Complete proofs for all theorems are provided in the Appendix. The location
of each proof in the Appendix is clearly indicated for each theorem in the paper.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and cross-
referenced.
• All assumptions should be clearly stated or referenced in the statement of any theo-
rems.
• The proofs can either appear in the main paper or the supplemental material, but if
they appear in the supplemental material, the authors are encouraged to provide a
short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be comple-
mented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
4. Experimental Result Reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main
experimental results of the paper to the extent that it affects the main claims and/or conclu-
sions of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes] .
Justification: The experimental setup for reproducing our results is detailed in Section 6
and Appendix G. We submitted our source codes through OpenReview.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived
well by the reviewers: Making the paper reproducible is important, regardless of
whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps
taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture
fully might suffice, or if the contribution is a specific model and empirical evaluation,
it may be necessary to either make it possible for others to replicate the model with
the same dataset, or provide access to the model. In general. releasing code and data
is often one good way to accomplish this, but reproducibility can also be provided via
detailed instructions for how to replicate the results, access to a hosted model (e.g., in
the case of a large language model), releasing of a model checkpoint, or other means
that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all sub-
missions to provide some reasonable avenue for reproducibility, which may depend
on the nature of the contribution. For example
(a) If the contribution is primarily a new algorithm, the paper should make it clear
how to reproduce that algorithm.
(b) If the contribution is primarily a new model architecture, the paper should describe
the architecture clearly and fully.
(c) If the contribution is a new model (e.g., a large language model), then there should
either be a way to access this model for reproducing the results or a way to re-
produce the model (e.g., with an open-source dataset or instructions for how to
construct the dataset).
(d) We recognize that reproducibility may be tricky in some cases, in which case au-
thors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in
some way (e.g., to registered users), but it should be possible for other researchers
to have some path to reproducing or verifying the results.
47


<!-- page 48 -->
5. Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: [Yes] .
Justification: We only used the popular benchmark datasets (MNIST and CIFAR-10) that
can be easily obtained. As for the source code, we denoted that our experiments are imple-
mented by adapting the source codes of Harutyunyan et al. [15] (https://github.com/
hrayrhar/f-CMI).
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/
public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not
be possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
including code, unless this is central to the contribution (e.g., for a new open-source
benchmark).
• The instructions should contain the exact command and environment needed to run to
reproduce the results. See the NeurIPS code and data submission guidelines (https:
//nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how
to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new
proposed method and baselines. If only a subset of experiments are reproducible, they
should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized
versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the
paper) is recommended, but including URLs to data and code is permitted.
6. Experimental Setting/Details
Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
results?
Answer: [Yes] .
Justification: We explained all details of our experimental settings in Section 6 and Ap-
pendix G.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of
detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental
material.
7. Experiment Statistical Significance
Question: Does the paper report error bars suitably and correctly defined or other appropri-
ate information about the statistical significance of the experiments?
Answer: [Yes] .
Justification: We reported the mean ± std. of our bound values for all experiments in
Section 6 and Appendix H. The reason why we cannot see the range of std. in the plot is
that its values are very small (less than 1e −4; see Table 1 for example.).
Guidelines:
• The answer NA means that the paper does not include experiments.
48


<!-- page 49 -->
• The authors should answer ”Yes” if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula,
call to a library function, bootstrap, etc.)
• The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error
of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should prefer-
ably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of
Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or
figures symmetric error bars that would yield results that are out of range (e.g. negative
error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how
they were calculated and reference the corresponding figures or tables in the text.
8. Experiments Compute Resources
Question: For each experiment, does the paper provide sufficient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?
Answer: [Yes] .
Justification: We used NVIDIA GPUs with 32GB memory (NVIDIA DGX-1 with Tesla
V100 and DGX-2) for MNIST (SGLD) and CIFAR-10 experiments. We also used CPU
(Apple M1) with 16GB memory for the other experiments (see Appendix G).
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster,
or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual
experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute
than the experiments reported in the paper (e.g., preliminary or failed experiments
that didn’t make it into the paper).
9. Code Of Ethics
Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] .
Justification: We confirmed that our paper does not have issues concerning the NeurIPS
Code of Ethics, although the primary emphasis of this paper is on theoretical analysis.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a
deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consid-
eration due to laws or regulations in their jurisdiction).
10. Broader Impacts
Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer: [Yes] .
49


<!-- page 50 -->
Justification: Although the primary focus of this paper is theoretical analysis, discussions
on the potential impacts of our research are presented in Sections 1 and 7.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal
impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact spe-
cific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied
to particular applications, let alone deployments. However, if there is a direct path to
any negative applications, the authors should point it out. For example, it is legitimate
to point out that an improvement in the quality of generative models could be used to
generate deepfakes for disinformation. On the other hand, it is not needed to point out
that a generic algorithm for optimizing neural networks could enable people to train
models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is
being used as intended and functioning correctly, harms that could arise when the
technology is being used as intended but gives incorrect results, and harms following
from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitiga-
tion strategies (e.g., gated release of models, providing defenses in addition to attacks,
mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
feedback over time, improving the efficiency and accessibility of ML).
11. Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?
Answer: [NA] .
Justification: The primary focus of this paper is theoretical analysis, and although it in-
cludes experiments, their purpose is to numerically validate the theory. Therefore, the
concerns raised in the question do not apply.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with
necessary safeguards to allow for controlled use of the model, for example by re-
quiring that users adhere to usage guidelines or restrictions to access the model or
implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors
should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do
not require this, but we encourage authors to take this into account and make a best
faith effort.
12. Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?
Answer: [Yes] .
Justification: We provide citations or reference URLs for all of the code, data, and models
used in our experiments (see Appendix G). We also declared the name of the licence is
CC-BY 4.0 in our submission page of OpenReview.
Guidelines:
50


<!-- page 51 -->
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a
URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of
service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the pack-
age should be provided. For popular datasets, paperswithcode.com/datasets has
curated licenses for some datasets. Their licensing guide can help determine the li-
cense of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of
the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13. New Assets
Question: Are new assets introduced in the paper well documented and is the documenta-
tion provided alongside the assets?
Answer: [NA] .
Justification: The primary focus of this paper is theoretical analysis, and although it in-
cludes experiments, their purpose is to numerically validate the theory. Therefore, the
concerns raised in the question do not apply.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their
submissions via structured templates. This includes details about training, license,
limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose
asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can
either create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and Research with Human Subjects
Question: For crowdsourcing experiments and research with human subjects, does the pa-
per include the full text of instructions given to participants and screenshots, if applicable,
as well as details about compensation (if any)?
Answer: [NA] .
Justification: We do not utilize such services, so the concerns raised in the question are not
applicable to us.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research
with human subjects.
• Including this information in the supplemental material is fine, but if the main contri-
bution of the paper involves human subjects, then as much detail as possible should
be included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, cura-
tion, or other labor should be paid at least the minimum wage in the country of the
data collector.
15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human
Subjects
Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?
51


<!-- page 52 -->
Answer: [NA] .
Justification: The primary focus of this paper is theoretical analysis, and it has been con-
firmed that the concerns raised in the question are not applicable.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research
with human subjects.
• Depending on the country in which research is conducted, IRB approval (or equiva-
lent) may be required for any human subjects research. If you obtained IRB approval,
you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity
(if applicable), such as the institution conducting the review.
52