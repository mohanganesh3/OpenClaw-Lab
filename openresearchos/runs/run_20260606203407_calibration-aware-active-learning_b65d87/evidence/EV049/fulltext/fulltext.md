<!-- page 1 -->
arXiv:2502.04807v2  [stat.ML]  15 Jun 2025
Robust Conformal Outlier Detection under Contaminated Reference Data
Meshi Bashari 1 Matteo Sesia 2 3 Yaniv Romano 1 4
Abstract
Conformal prediction is a flexible framework for
calibrating machine learning predictions, provid-
ing distribution-free statistical guarantees. In out-
lier detection, this calibration relies on a reference
set of labeled inlier data to control the type-I er-
ror rate. However, obtaining a perfectly labeled
inlier reference set is often unrealistic, and a more
practical scenario involves access to a contami-
nated reference set containing a small fraction of
outliers. This paper analyzes the impact of such
contamination on the validity of conformal meth-
ods. We prove that under realistic, non-adversarial
settings, calibration on contaminated data yields
conservative type-I error control, shedding light
on the inherent robustness of conformal methods.
This conservativeness, however, typically results
in a loss of power. To alleviate this limitation,
we propose a novel, active data-cleaning frame-
work that leverages a limited labeling budget and
an outlier detection model to selectively annotate
data points in the contaminated reference set that
are suspected as outliers. By removing only the
annotated outliers in this “suspicious” subset, we
can effectively enhance power while mitigating
the risk of inflating the type-I error rate, as sup-
ported by our theoretical analysis. Experiments
on real datasets validate the conservative behav-
ior of conformal methods under contamination
and show that the proposed data-cleaning strategy
improves power without sacrificing validity.
1Department of Electrical and Computer Engineering, Technion
IIT, Haifa, Israel 2Department of Data Sciences and Operations,
University of Southern California, Los Angeles, California, USA
3Department of Computer Science, University of Southern Cali-
fornia, Los Angeles, California, USA 4Department of Computer
Science, Technion IIT, Haifa, Israel. Correspondence to: Meshi
Bashari <meshi.b@campus.technion.ac.il>.
Proceedings of the 42 nd International Conference on Machine
Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025
by the author(s).
1. Introduction
1.1. Background and Motivation
This paper studies the problem of outlier detection: given a
reference dataset (e.g., a collection of legitimate financial
transactions) and an unlabeled test point (a new transaction),
our goal is to determine whether the test point is an outlier
(a fraudulent transaction) by assessing its deviation from the
reference data distribution. Naturally, we aim to maximize
the detection of outliers by harnessing the capabilities of
complex machine learning (ML) models. However, these
models typically lack type-I error rate control, potentially re-
sulting in unreliable detections. In our running example, the
type-I error is the probability of falsely flagging a legitimate
transaction as fraudulent. As such, uncontrolled error rates
can lead to costly unnecessary investigations of legitimate
transactions and negatively impact customer experience.
The broad need for reliable ML systems has sparked a surge
of interest in conformal prediction—a versatile framework
that can provide statistical guarantees for any “black-box”
predictive model (Vovk et al., 2005). This framework for-
mulates the outlier detection task as a statistical test, where
the null hypothesis is that the new data point is not an out-
lier (Laxhammar & Falkman, 2015; Bates et al., 2023). To
derive a decision rule guaranteeing type-I error control, con-
formal inference relies on a reference (calibration) set of
inlier data points. These points are assumed to be sampled
i.i.d. from an unknown distribution, independent of the data
used to train the outlier detection model.
In practice, however, it is often difficult to obtain a perfectly
clean reference dataset that contains no outliers (Park et al.,
2021; Zhao et al., 2019; Chalapathy & Chawla, 2019; Jiang
et al., 2022). In our example, a more realistic scenario would
assume instead access to a slightly contaminated reference
set, mostly legitimate transactions with a few unnoticed
outliers (Zhao et al., 2019). But this setting poses new
challenges for conformal prediction methods, potentially
invalidating the error control guarantees or, as we shall see,
often reducing the power to detect true outliers at test time.
1.2. Outline and Contributions
While type-I error control in conformal inference theoret-
ically requires perfectly clean reference data, in practice
1


<!-- page 2 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
contaminated data often makes these methods overly conser-
vative, reducing the power to detect true outliers rather than
inflating the type-I error rate. This empirical observation
motivates the first question explored in this paper:
Q1: When does conformal outlier detection with contami-
nated reference data yield valid type-I error control?
In Section 2.3, we present the first contribution of this pa-
per: a novel theoretical analysis that identifies common
conditions under which this conservative behavior arises.
Unfortunately, this conservativeness often comes at the cost
of reduced detection power, particularly when targeting
low type-I error rates. To address this issue, we investi-
gate data-driven cleaning strategies aimed at mitigating the
contamination in the reference dataset.
A straightforward approach to cleaning the contaminated
set is to remove all data points flagged as likely outliers by
the detection model. However, this method is unsatisfactory,
as it risks inadvertently removing inliers along with outliers,
resulting in an ”overly clean” reference set. This, in turn,
distorts the inlier distribution and inflates the type-I error
rate above the desired nominal level.
This challenge motivates our second and main contribu-
tion. In Section 3.3, we introduce an approach to clean the
contaminated reference set by leveraging a limited labeling
budget (e.g., 50 annotations). The outlier detection model is
first used to identify suspected outliers within the contami-
nated reference set. The limited budget is then strategically
allocated to annotate these points, thereby avoiding the un-
intended removal of inliers. While this is a practical and
intuitive approach, it naturally prompts a critical question:
Q2: How does the selective annotation and partial removal
of outliers from a contaminated reference set affect the va-
lidity of conformal inferences?
We analyze the validity of this active labeling approach for
trimming outliers in the contaminated set. Our theoretical
results identify the conditions required to achieve approxi-
mate type-I error control, even when the data are selectively
annotated and not all outliers are removed. This analysis
also highlights key factors that can inflate the error rate, of-
fering practitioners guiding principles to enhance the power
of conformal methods in the presence of contaminated data.
Finally, in Section 4, we empirically validate our theory and
proposed data-cleaning approach through comprehensive
experiments on real-world datasets. The experiments con-
firm that conformal inference with contaminated data tends
to be conservative. Furthermore, they demonstrate that our
method significantly boosts power, particularly when the
target type-I error rate is low and the number of outliers in
the contaminated set is small.
A software that implements the proposed method is available
at https://github.com/Meshiba/robust-conformal-od.
1.3. Related Work
Recently, there has been growing interest in studying the
statistical properties of conformal inference methods under
more realistic scenarios, moving beyond the idealized as-
sumption of perfectly clean and exchangeable observations
to account for various types of distribution shift (Tibshirani
et al., 2019; Einbinder et al., 2024; Sesia et al., 2024; Barber
et al., 2023; Gibbs & Cand`es, 2021; Zaffran et al., 2022;
Feldman et al., 2023; Gibbs & Cand`es, 2024; Podkopaev &
Ramdas, 2021; Si et al., 2023; Prinster et al.). This paper
draws inspiration from several prior works in this area.
Tibshirani et al. (2019) introduced a weighted conformal pre-
diction approach to address covariate shift between calibra-
tion and test data, later extended by Podkopaev & Ramdas
(2021) to accommodate label shift. Both settings, however,
involve a different form of distribution shift from the one
we study here. Barber et al. (2023) extend this line of work
by analyzing the effects of general distribution shifts on
the validity of conformal methods, focusing however on
worst-case scenarios; see also Farinhas et al. (2024).
In contrast, our work moves away from this worst-case
perspective. We aim to explain why conformal outlier de-
tection with contaminated reference data often results in
a conservative type-I error rate, rather than investigating
type-I error inflation, which, while theoretically possible in
adversarial settings, appears less common in practice. Fur-
thermore, we focus on developing methods to address this
over-conservativeness, boosting detection power.
A more closely related line of work investigates the robust-
ness of conformal prediction to label noise (Einbinder et al.,
2024; Sesia et al., 2024; Clarkson et al., 2024; Penso et al.,
2025) or other forms of data contamination (Zaffran et al.,
2023; 2024; Feldman & Romano, 2024). Specifically, Ein-
binder et al. (2024) and Sesia et al. (2024) show that, under
certain assumptions, conformal prediction for classification
with noisy labels often results in conservative type-I error
rates. Furthermore, Sesia et al. (2024) proposes a method
to address this conservativeness by leveraging an explicit
“label noise model” that captures the relationship between
the true and contaminated labels in the calibration dataset.
In contrast, this paper avoids relying on an explicit model
for the contaminated data, as such models can be difficult to
estimate in practice within our context. Instead, we utilize a
pre-trained black-box outlier detection model and a limited
annotation budget to selectively and reliably trim outliers
from the contaminated set. Furthermore, it is important to
emphasize that the method proposed by Sesia et al. (2024)
is primarily designed for classification tasks with relatively
balanced data, whereas outlier detection naturally involves
2


<!-- page 3 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
extreme class imbalance. This distinction underscores the
need for solutions specifically tailored to outlier detection.
2. Setup and Preliminary Results
2.1. Inference with Clean Calibration Data
Conformal inference for outlier detection requires a refer-
ence (or calibration) set, Dcal = [n] := {1, . . . , n}, con-
taining n data points. The reference set is typically assumed
to be clean, consisting solely of inliers, which are i.i.d. sam-
ples from an unknown distribution P0 (exchangeability may
sometimes suffice, but this work assumes i.i.d. inliers). Un-
der this assumption, Dcal may be referred to as Dinlier.
The goal is to determine whether a new observation, Xn+1,
is an inlier—independently sampled from P0—or an outlier,
sampled from a different distribution P1 ̸= P0. This can be
formulated as a hypothesis testing problem, where the null
hypothesis H0 claims that Xn+1 is an inlier:
Xi
i.i.d.
∼P0, ∀i ∈Dinlier,
Dinlier = Dcal = [n],
H0 : Xn+1
ind.
∼P0 .
(1)
The split-conformal method, a simple and computation-
ally efficient approach, uses a pre-trained outlier detection
model—potentially any machine learning model—to com-
pute nonconformity scores that quantify how different a data
point is from the reference distribution. The model, repre-
sented by a score function s, is trained on a separate dataset
Dtrain, which is similar to but independent of Dcal. Typi-
cally, a larger dataset of inliers, assumed to be i.i.d. samples
from P0, is randomly split into Dtrain and Dcal.
The model tries to learn a score function s such that larger
values of s(Xn+1) indicate stronger evidence that the test
point may be an outlier. Conformal inference rigorously
quantifies this evidence, providing a principled decision
rule for rejecting H0 when the evidence is strong enough,
while controlling the type-I error rate—the probability of
incorrectly rejecting H0 when Xn+1 is actually an inlier.
This statistical evidence is quantified by computing a con-
formal p-value, defined as:
ˆpn+1 = 1 + Pn
i=1 I[s(Xi) ≥s(Xn+1)]
1 + n
.
(2)
Thus, larger values of s(Xn+1) correspond to smaller val-
ues of ˆpn+1, and the test point Xn+1 can be confidently
classified as an outlier (rejecting H0) when ˆpn+1 is smaller
than a given significance level α ∈(0, 1).
Proposition 2.1 (from Vovk et al. (2005)). Under (1), if
the null hypothesis H0 is true, then for any α ∈(0, 1):
P (ˆpn+1 ≤α) ≤α. Further, if s(X) has a continuous
distribution under P0, then P (ˆpn+1 ≤α) ≥α −1/(n + 1).
Proposition 2.1 intuitively states that the conformal p-value
defined in (2) provides a well-calibrated rule for flagging
a new data point as a likely outlier. Rejecting H0 when
ˆpn+1 ≤α ensures type-I error control at level α while
avoiding excessive conservatism. Specifically, the type-I
error rate closely matches α when the sample size n is large
and the nonconformity scores have a continuous distribu-
tion with no ties—a mild condition that can be achieved in
practice by adding small random noise to the scores.
What remains unclear, and serves as the starting point of
this paper, is how conformal p-values behave when the cali-
bration dataset is contaminated, containing not only inliers
but also a fraction of misplaced outliers.
2.2. Inference with Contaminated Calibration Data
In this paper, we consider a more general setting where the
calibration dataset, indexed by Dcal = [n], may contain
both inliers (Dinlier), sampled i.i.d. from a distribution P0,
and outliers (Doutlier), sampled i.i.d. from a different dis-
tribution P1 ̸= P0. Thus, Dcal = Dinlier ∪Doutlier. The
numbers of inliers and outliers, respectively n0 = | Dinlier |
and n1 = | Doutlier |, are treated as fixed, with n = n0 + n1.
The goal remains to test the null hypothesis H0 that a new
data point Xn+1 is an inlier, independently sampled from
P0. Formally, this setup can be written as:
Xi
i.i.d.
∼P0, ∀i ∈Dinlier,
Xi
i.i.d.
∼P1, ∀i ∈Doutlier,
Dinlier ∪Doutlier = Dcal = [n],
H0 : Xn+1
ind.
∼P0 .
(3)
In the following, we first analyze the behavior of standard
conformal p-values, computed as in (2), when applied to
contaminated data scenarios described by (3). Subsequently,
we will propose a novel method for computing more re-
fined conformal p-values by approximately cleaning the
calibration set to remove undesired outliers.
2.3. Explaining the Conservativeness
Empirical results suggest that contamination by outliers
in the calibration data often makes standard conformal p-
values overly conservative, resulting in a type-I error rate
significantly lower than the desired nominal level α.
We begin by examining Figure 1, which provides some in-
sight into this behavior based on the analysis of the “shuttle”
dataset (shu), as detailed in Section 4. In this example, the
nonconformity scores of outlier data points in the contam-
inated calibration set are typically larger than those of the
inliers. This pattern reflects the goal of a well-designed out-
lier detection model: to differentiate outliers from inliers by
assigning higher scores to the former. Consequently, confor-
mal p-values computed using (2) will be inflated relative to
3


<!-- page 4 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
the ideal scenario in which all calibration points are inliers,
reducing our power to detect true outliers at test time.
0.1
0.0
0.1
0.2
Scores
0
200
400
600
Inliers
Outliers
Qcal
1
Qoutlier
1
Qinlier
1
Figure 1: Histogram of nonconformity scores for inliers and
outliers in a contaminated calibration subset of the “shuttle”
data, with a contamination rate of 5%. The vertical lines
indicate the (1 −α) empirical quantile of all calibration
scores (black), as well as separately for inliers (blue) and
outliers (red), with α = 0.02.
This phenomenon is further corroborated by extensive
numerical experiments presented in Section 4 and Ap-
pendix B.2, which consistently demonstrate this conser-
vative behavior across nine different datasets.
While the conservativeness of conformal prediction meth-
ods in the presence of contaminated calibration data has
already been observed and studied theoretically in different
contexts (Einbinder et al., 2024; Sesia et al., 2024; Clarkson
et al., 2024), prior works did not focus on outlier detection.
Therefore, it is helpful to introduce a new theoretical result
that precisely quantifies the inflation of standard conformal
p-values that we often observe in practice. This will serve
as a foundation for the novel method of computing adaptive
conformal p-values presented in the next section.
Let ˆF1 denote the empirical cumulative distribution function
(CDF) of the scores in Doutlier and ˆQcal
1−α represent the
⌈(1 −α)(1 + n)⌉-th smallest score in the calibration set.
Lemma 2.2. Under the setup defined in (3), if H0 is true,
then for any α ∈(0, 1),
P (ˆpn+1 ≤α) ≤α −
n1
n0 + 1
 
1 −α −E
h
ˆF1
 
ˆQcal
1−α
 i 
.
This result is related to Theorem 1 in Sesia et al. (2024),
which studies the behavior of conformal prediction sets for
multi-class classification (Lei et al., 2013; Romano et al.,
2020) calibrated with contaminated data. The key distinc-
tion is in our treatment of the calibration set: we assume
that n0 and n1 are fixed, whereas Sesia et al. (2024) con-
sider a mixture model where the observed proportions of
data points from different classes in the calibration set are
random. While treating n0 and n1 as fixed is convenient
for this paper, we also include an additional result (Corol-
lary A.3) in Appendix A, which reaches qualitatively similar
conclusions by adopting an approach more closely aligned
with Theorem 1 from Sesia et al. (2024).
A direct corollary of Lemma 2.2 is that standard conformal
p-values are conservative when outlier scores are typically
larger than inlier scores. Formally, this condition is:
Assumption 2.3. E[ ˆF1( ˆQcal
1−α)] < 1 −α.
If Assumption 2.3 fails—for instance, when outlier scores
are smaller than inlier scores—data contamination may in-
validate standard conformal p-values, inflating the type-I
error rate. However, it is more common in practice that
Assumption 2.3 holds, in which case contamination tends
to reduce calibration power, and more so if n1 is large. In
particular, Assumption 2.3 holds if: (i) the outlier detection
model is relatively accurate, and (ii) the outlier distribution
P1 is not adversarial. Our experiments will show this power
loss can be substantial, motivating the need for new methods
that can approximately “clean up” the calibration data.
3. Methods
3.1. Key Idea: Boosting Power by Cleaning the Data
Ideally, we would like to remove all n1 outliers from Dcal,
restoring the ideal behavior of conformal p-values calibrated
on a clean dataset, as described in Proposition 2.1, and
likely boosting power. However, manually labeling the
entire contaminated calibration set Dcal is often impractical,
especially when n = | Dcal | is large. At the same time,
utilizing only a small calibration set is not always desirable.
A large calibration set is often needed because the small-
est conformal p-value obtainable through (2) scales as 1/n.
Thus, a large n is critical for achieving high confidence
in identifying outliers, especially in “needle-in-a-haystack”
scenarios (Bates et al., 2023), where a few outliers must be
detected in a large test set dominated by inliers. In such
cases, the ability to obtain very small p-values is essen-
tial to achieve non-trivial power while controlling the false
discovery rate (Benjamini & Hochberg, 1995).
3.2. A Simple but Unsatisfactory Approach: Naive-Trim
The above challenge underscores the need for a method to
mitigate the impact of outliers in the calibration dataset with-
out requiring exhaustive annotation. An intuitive approach
is to forgo annotations and simply remove all “suspicious”
data points with large nonconformity scores. For instance,
one could remove the top m scores from Dcal, where m
is a fixed guess of the true number of outliers n1 in the
calibration set. We refer to this approach as Naive-Trim.
While Naive-Trim can reduce conservativeness by re-
4

[CAPTION] Figure 1: Histogram of nonconformity scores for inliers and


<!-- page 5 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
moving some outliers, it is not a satisfactory solution as it
risks “over-compensating”. By potentially removing also
true inliers with large nonconformity scores, it can signif-
icantly skew the inlier score distribution to the left. This
side effect is problematic, as it tends to invalidate conformal
p-values and inflate the type-I error rate, over-correcting the
conservativeness of standard conformal p-values.
This issue is particularly pronounced when m > n1 or
in noisy settings where the outlier detection model cannot
perfectly distinguish between inliers and outliers. For ex-
ample, as shown in Section 4, applying Naive-Trim to
the dataset illustrated in Figure 1 results in uncontrolled
inflation of the type-I error rate.
To address this challenge, we will now present a more so-
phisticated method, which we refer to as Label-Trim.
This approach utilizes a limited labeling budget to remove
outliers from Dcal in a more reliable manner, mitigating the
risk of over-correcting the conformal p-value.
3.3. The Label-Trim Method
Consider having a limited budget to label m < n calibration
samples, where m is much smaller than n. We aim to utilize
this budget to remove as many outliers as possible from the
calibration set without altering the inlier score distribution.
A practical approach is to annotate the m largest scores in
Dcal, as these are most likely outliers based on the model.
Denote these annotated samples as Dlabeled ⊆Dcal, and let
Doutlier
labeled denote the subset of annotated data points that are
true outliers. Removing these outliers from Dcal yields a
smaller, cleaner calibration set, which we call DLT
cal ⊆Dcal.
The Label-Trim method then calculates a refined con-
formal p-value, now denoted as ˆpLT
n+1, following the same
procedure as in (2) with Dcal replaced by the (partially)
cleaned calibration set DLT
cal:
ˆpLT
n+1 =
1 + P
i∈DLT
cal I[s(Xi) ≥s(Xn+1)]
1 + | DLT
cal |
.
(4)
Algorithms 1 and 2 summarize this procedure, which in-
tuitively offers advantages over both the standard method
for computing ˆpn+1 in (2), by potentially increasing power,
and the Naive-Trim approach, by mitigating the risk of
over-correcting ˆpn+1. A schematic illustration of the con-
struction of DLT
cal by Algorithm 1 is shown in Figure 2.
The
following
theorem
provides
justification
for
Label-Trim,
demonstrating
that
ˆpLT
n+1
is
an
ap-
proximately valid p-value under relatively mild conditions.
While our method is intuitive, this result is nontrivial for
two reasons. First, Label-Trim cannot guarantee the
removal of all outliers from the calibration set, as it may be
that m < n1 or some outliers are not among the m largest
scores. Second, it involves annotating the m largest scores,
Outlier detection
model
Sorted scores
labeling budget 
Figure 2: A schematic illustration of the proposed active
data-cleaning of the contaminated reference set (Algo-
rithm 1). The approach begins by computing nonconfor-
mity scores of the contaminated reference data Dcal using a
pretrained outlier detection model, where blue circles denote
inliers and red circles denote outliers. The scores are then
sorted in increasing order, and the top m samples—those
most likely to be outliers—are selected for annotation. Af-
ter removing the annotated outliers, the resulting (partially)
cleaned set DLT
cal is used for calibration.
Algorithm 1 Label-trim calibration (construction phase)
1: Input: labeling budget m; contaminate calibration-set
Dcal = {Xi}n
i=1; score function s(·), obtained by a
pre-trained outlier detection model;
2: Compute the calibration scores Si = s(Xi), ∀i ∈Dcal.
3: Sort the calibration scores, such that Sπ(1) ≤· · · ≤
Sπ(n) where π : [n] →[n] is the corresponding permu-
tation of the indices.
4: Annotate
the
m
largest
scores
Dlabeled:=
{(Sπ(i), Yπ(i)) : i > n −m}, with Yπ(i) = 0 if
Xπ(i) is an inlier and Yπ(i) = 1 otherwise.
5: Construct the trimmed calibration set DLT
cal
=
{π(i) : i ≤n −m} ∪{j : j ∈Dlabeled and Yj = 0}.
6: Output: trimmed calibration set DLT
cal.
Algorithm 2 Label-trim calibration (testing phase)
1: Input: test point Xn+1; score function s; trimmed
calibration set DLT
cal; type-I error level α;
2: Compute the conformal p-value ˆpLT
n+1 according to (4).
3: Output: reject the null hypothesis H0 if ˆpLT
n+1 ≤α,
classifying Xn+1 as an outlier.
revealing the true labels of some calibration points but not
others, which could disrupt the exchangeability typically
assumed among inlier data points in conformal inference.
Therefore, this justification requires novel proof techniques
and does not follow directly from existing results.
Following a notation similar to that of Lemma 2.2, let ˆF LT
1
denote the empirical CDF of the scores in DLT
outlier. Define
also ˆQLT
1−α as the ˆiLT-th smallest element in {Si}i∈DLT
cal ∪
{∞}, with ˆiLT := ⌈(1 −α)(nLT + 1)⌉and nLT :=
  DLT
cal
  .
Theorem 3.1. Consider the setup in (3), with H0 being true.
5

[CAPTION] Figure 2: A schematic illustration of the proposed active


<!-- page 6 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
For any fixed α ∈(0, 1), assume that m ≤α(n + 1). Then,
P
 ˆpLT
n+1 ≤α
 
≤α +
1
n0 + 1
−E
  ˆnLT
1
n0 + 1
 
(1 −α) −ˆF LT
1
 
ˆQLT
1−α
   
.
The upper bound on the type-I error rate provided by Theo-
rem 3.1 resembles that of Lemma 2.2 and can be interpreted
as follows: Label-Trim produces approximately valid
conformal p-values if: (i) the labeling budget is small rela-
tive to the calibration set size, i.e., m ≤α(n + 1); and (ii)
the calibration set contains a large number of inliers, n0.
However, the upper bound in Theorem 3.1 also suggests
that Label-Trim may remain overly conservative, sim-
ilar to standard conformal p-values, if (i) not all outliers
are removed from the calibration set (ˆnLT
1
> 0 with high
probability), and (ii) the remaining outlier scores are gener-
ally larger than the remaining inlier scores, consistent with
E[ ˆF LT
1 ( ˆQLT
1−α)] < 1 −α, akin to Assumption 2.3.
This potential conservative behavior arises naturally from
the use of a limited labeling budget, especially when the
model guiding the construction of the set Dlabeled fails to
effectively detect true outliers. Nevertheless, as we will see
in the next section, Label-Trim often enhances power.
4. Experiments
We turn to evaluate the performance of conformal outlier de-
tection methods under contaminated data. The experiments
presented in this section are conducted on nine benchmark
datasets: three tabular datasets, listed in Section 4.1, and six
visual datasets, listed in Section 4.2.
Methods
We compare the following methods:
• Standard: The basic conformal method that uses
the contaminated reference set Dcal = Dinlier ∪Doutlier.
• Oracle: An infeasible benchmark method where the
reference set contains only inliers, i.e., Dcal = Dinlier.
• Naive-Trim: The baseline method from Section 3.2,
which removes the top r% non-conformity scores from
Dcal, where r = n1/(n0 + n1).
• Label-Trim: Our proposed reliable data-cleaning
method from Section 3.3, applied with a labeling bud-
get of m = 50 annotations to label the m data points
with the largest non-conformity scores from Dcal.
• Small-Clean: A baseline method that uses the la-
beling budget to construct a small, clean reference set
by (i) randomly selecting m data points from Dcal and
(ii) extracting the true inliers from this subset.
Setup and performance metrics
In all experiments, we
randomly split a given dataset into disjoint training Dtrain,
calibration Dcal, and test sets of inliers Dinlier
test
and outliers
Doutlier
test
. To simulate a realistic setting, we construct the
training and contaminated calibration sets with the same
contamination rate of r%. The inlier Dinlier
test
and outlier
Doutlier
test
test sets are used to compute the type-I error and
power of the outlier detection model, respectively. To ensure
fair comparisons, all conformal methods use the same outlier
detection model, trained on Dtrain. Performance metrics are
evaluated across 100 random splits of the data. The size of
each dataset, along with the details of how Dtrain, Dcal, and
Dtest are constructed are provided in Appendix B.1.
4.1. Tabular Data
We now compare the performance of the different methods
on three benchmark tabular datasets for outlier detection,
previously used in the conformal literature (Bates et al.,
2023). Since conclusions are similar across datasets, we
focus here on results for the shuttle dataset (shu). Results
for the credit card (cre) and KDDCup99 (KDD) datasets are
presented in Appendix B.2.1. For all conformal methods,
we use Isolation Forest (Liu et al., 2008) as the base outlier
detection model, implemented using scikit-learn with
default hyperparameters (Buitinck et al., 2013).
Figure 3 presents the performance metrics of each method
as a function of the contamination rate r. Following the left
panel in that figure, we can see that the Standard con-
formal method results in conservative type-I error control,
with a decrease in the error rate as the outlier proportion
increases—a behavior that is aligned with Lemma 2.2. No-
tably, the type-I error of the Oracle method is tightly
centered around α, as guaranteed by Proposition 2.1. The
Naive-Trim method does not control the type-I error rate,
emphasizing the need for reliable data-cleaning procedures.
In striking contrast, our Label-Trim method achieves a
valid type-I error rate. At lower outlier proportions, the
empirical type-I error is close to α, but the method becomes
more conservative as the outlier proportion increases. This
observation aligns with the upper bound on the error rate
derived in Theorem 3.1. Notably, as the contamination rate
in the training data increases, the outlier detection model’s
ability to distinguish between inliers and outliers weakens.
This, in turn, adversely affects the effectiveness of forming
a subset of data points for annotation, as demonstrated in
the right panel of Figure 3. The Small-Clean method
also controls the type-I error but is more conservative than
Label-Trim due to its much smaller reference set, which
becomes even smaller as the contamination rate increases.
Observe how the power of the Small-Clean method is
lower than that of the Standard approach, despite the
latter using a contaminated reference set. By contrast, our
proposed Label-Trim method significantly improves the
power of the Standard method and even achieves near-
oracle performance when the outlier proportion is low.
6

[CAPTION] Figure 3 presents the performance metrics of each method


<!-- page 7 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
0.0
0.01
0.02
0.03
0.04
0.05
Contamination rate
0.00
0.01
0.02
0.03
0.04
Type-1-Error
0.0
0.01
0.02
0.03
0.04
0.05
Contamination rate
0.1
0.2
0.3
0.4
0.5
0.6
Power
0.0
0.01
0.02
0.03
0.04
0.05
Contamination rate
0
25
50
75
100
125
Trimmed
Standard
Oracle (infeasible)
Naive-Trim (invalid)
Small-Clean
Label-Trim
Target type-I level
Labeling budget
# outliers
Figure 3: Comparison of conformal outlier detection methods on a tabular dataset (“shuttle”) as a function of the contamina-
tion rate r. The target type-I error rate is α = 0.02. Left: Empirical type-I error. Middle: Average detection rate (power),
where higher values indicate better performance. Right: Number of outliers trimmed by the Label-Trim method. Results
are averaged across 100 random splits of the data.
10
20
30
40
50
60
70
80
90
100
Labeling budget
0.000
0.005
0.010
0.015
0.020
0.025
Type-1-Error
10
20
30
40
50
60
70
80
90
100
Labeling budget
0.0
0.2
0.4
0.6
Power
10
20
30
40
50
60
70
80
90
100
Labeling budget
20
40
60
80
100
Trimmed
Oracle (infeasible)
Small-Clean
Label-Trim
Target type-I level
Labeling budget
# outliers
Figure 4: Comparison of conformal outlier detection methods on a tabular dataset (“shuttle”) as a function of the labeling
budget m. The contamination rate is fixed to r = 0.03. Other details are as in Figure 3.
Next, we study the effect of the labeling budget on the perfor-
mance of our Label-Trim method. As shown in Figure 4,
increasing the labeling budget brings the Label-Trim
method closer to the Oracle in terms of both type-I error
and power. Notably, even with a modest budget of 40–50
annotations, the power of Label-Trim is nearly indis-
tinguishable from that of the Oracle. This is attributed
to the method’s effective trimming of outliers, as shown
in the right panel. Notably, for labeling budgets m > 50,
the condition in Theorem 3.1 no longer holds, and yet the
Label-Trim method still achieves valid type-I error con-
trol at level α in practice. This highlights the robustness
of the proposed method to the choice of m beyond the
restrictions specified in Theorem 3.1, where we attribute
this robustness to the non-adversarial nature of the outlier
distribution and the underlying detection model.
Figure 4 also illustrates that the Small-Clean method
lags behind Label-Trim both in terms of power and con-
servativeness. For small labeling budgets of m < 45, the
coarse granularity of conformal p-values (2) renders the
method powerless; the smallest achievable p-value in this
case is 1/(m + 1) > 0.02 = α. Even for slightly larger
labeling budgets, the conservative nature of the conformal
p-value—specifically, the ‘plus 1’ term in (2)—continues to
have a significant impact. This effect is rigorously quanti-
fied by the lower bound on type-I error provided in Propo-
sition 2.1. For instance, with m = 80 and α = 0.02, the
lower bound is approximately α −1/(m + 1) ≈0.0076,
which aligns closely with the empirical error rate shown in
the left panel of Figure 4. Overall, these results highlight
the benefits of selectively cleaning a relatively large contam-
inated set compared to relying on a small clean reference
set, offering both improved stability and higher power.
Next, we examine how the target error level α affects the
performance of different methods. Figure 5 shows that
our Label-Trim method performs particularly well at
low type-I error rates, especially when α is smaller than
the contamination rate (r = 3%). This behavior can be
explained as follows. For a relatively accurate model, the
outliers primarily distort the tail of the empirical distribution
of nonconformity scores—see Figure 1. Consequently, the
influence of these outliers on the rejection rule ˆpn+1 ≤α
from (2), or ˆpLT
n+1 ≤α from (4), diminishes as α increases.
Additional experiments
In Appendix B.2.1, we extend
the experiments presented above. These include evalua-
tions under higher contamination levels and additional out-
lier detection models, such as One-Class SVM (Sch¨olkopf
et al., 2001) and Local Outlier Factor (Breunig et al., 2000).
We also examine the effect of different outlier injection
strategies and the robustness of the proposed method under
test-time distribution shifts in the outlier population.
7


**[Table p7.1]**
| 0.000.010.020.030.04 Type-1-Error 0.0 0.01 0.02 Contamin | 0.10.20.30.40.50.6 Power 0.03 0.04 0.05 0.0 0.01 0.02 ation rate Contamin | 75100125 Trimmed 50 25 0 0.03 0.04 0.05 0.0 0.01 0.02 0.03 0.04 0.05 ation rate Contamination rate | Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim Target type-I level Labeling budget # outliers |
| --- | --- | --- | --- |
|  |  |  |  |


**[Table p7.2]**
| 0.0000.0050.0100.0150.0200.025 Type-1-Error 10 20 30 40 50 Labelin | 0.6 0.4 Power 0.2 0.0 60 70 80 90 100 10 20 30 40 50 g budget Labeling | 80100 Trimmed 60 40 20 60 70 80 90 100 10 20 30 40 50 60 70 80 90 100 budget Labeling budget | Oracle (infeasible) Small-Clean Label-Trim Target type-I level Labeling budget # outliers |
| --- | --- | --- | --- |
|  |  |  |  |

[CAPTION] Figure 3: Comparison of conformal outlier detection methods on a tabular dataset (“shuttle”) as a function of the contamina-

[CAPTION] Figure 4: Comparison of conformal outlier detection methods on a tabular dataset (“shuttle”) as a function of the labeling

[CAPTION] Figure 4 also illustrates that the Small-Clean method


<!-- page 8 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
0.01
0.02
0.03
0.04
0.05
0.00
0.02
0.04
0.06
Type-1-Error
0.01
0.02
0.03
0.04
0.05
0.0
0.2
0.4
0.6
Power
Standard
Oracle (infeasible)
Naive-Trim (invalid)
Small-Clean
Label-Trim
Target type-I level
Figure 5: Comparison of conformal outlier detection methods on a tabular dataset (“shuttle”) as a function of the target
type-I error rate α. The contamination rate r is fixed to 3%. Other details are as in Figure 3.
4.2. Visual Data
In what follows, we compare all methods using benchmark
visual datasets for outlier detection. Similar to Zhang et al.
(2024), we construct six datasets, where the inlier samples
are always images from CIFAR10 (Krizhevsky & Hinton,
2009; Krizhevsky et al., 2009) and the outlier samples vary
across datasets. Specifically, the outliers are drawn from
(1) MNIST (Deng, 2012), (2) SVHN (Netzer et al., 2011),
(3) Texture (Cimpoi et al., 2014), (4) Places365 (Cimpoi
et al., 2014), (5) TinyImageNet (Torralba et al., 2008), and
(6) CIFAR100 (Krizhevsky et al., 2009). For all datasets,
we use the outlier detection model proposed by Sun et al.
(2021), ReAct, which operates on feature representations
extracted by a pre-trained ResNet-18 model. More details
are in Appendix B.1.
Table 1 summarizes the results for all six datasets. Overall,
we can see a trend similar to that of the tabular data: the
Standard and Small-Clean methods are valid but con-
servative, the Naive-Trim fails to control the type-I error,
and our Label-Trim achieves a significant boost in power
while practically controlling the type-I error. Notably, our
Label-Trim method attains near-oracle performance for
low contamination rates. Detailed results for each dataset,
along with experiments with additional outlier detection
models—specifically, the ReAct method (Sun et al., 2021)
with a pre-trained VGG-19 model and the SCALE method
(Xu et al., 2024) with a ResNet-18 model—are provided in
Appendix B.2.2.
5. Discussion
In this work, we studied the robustness of conformal predic-
tion under contaminated reference data. Motivated by empir-
ical evidence, we characterized the conditions under which
conformal outlier detection methods become too conserva-
tive. To improve power, we proposed the Label-Trim
method, which leverages an outlier detection model and a
limited labeling budget to remove outliers from the con-
taminated reference set. We also provided a theoretical
justification for this approach, employing novel proof tech-
niques. Numerical experiments with real data confirmed
that standard conformal outlier detection methods are con-
servative under contaminated data and demonstrated that our
Label-Trim method can significantly enhance power.
However, the experiments also reveal a limitation of our
Label-Trim method: while it improves power compared
to standard conformal inference, it often remains too conser-
vative, particularly when the labeling budget is very limited,
leaving room for further improvement. A promising direc-
tion for future research is to enhance Label-Trim with
active learning strategies (Makili et al., 2012; Fannjiang
et al., 2022; Prinster et al.), enabling the removal of more
outliers without increasing the labeling budget.
While we provide robustness results for calibration under
contaminated reference data, our analysis assumes that the
inlier calibration and test points are drawn i.i.d. from the
same distribution. An important direction for future work
is to extend this setting beyond the i.i.d. assumption, pos-
sibly by building on ideas from Tibshirani et al. (2019);
Podkopaev & Ramdas (2021); Sesia et al. (2024); Barber
et al. (2023).
Another limitation of the Label-Trim approach is its
reliance on actively collecting new annotations. In scenarios
where a flexible labeling budget is unavailable but access
to a small, clean reference set is feasible, this dependency
becomes restrictive. As our experiments demonstrate, the
limited sample size imposes a fundamental constraint on the
power of conformal outlier detection methods. This raises
an intriguing question for future research: given a small
clean reference set and a larger, contaminated reference set,
how can we effectively and safely clean the contaminated
data to enhance detection power at test time?
One potential solution could involve using the small clean
reference set to calibrate a base outlier detection model.
This calibrated model could then be employed to clean
the larger contaminated set by removing detected outliers,
while carefully accounting for inliers mistakenly classified
as outliers. Exploring such a semi-supervised data-cleaning
approach represents a promising direction for future work,
8


**[Table p8.1]**
| Type-1-Error 0.00 0.02 0.04 0.06 0.01 0.02 0.03 0.04 0.05 | Power 0.0 0.2 0.4 0.6 0.01 0.02 0.03 0.04 0.05 | Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim Target type-I level |
| --- | --- | --- |
|  |  |  |

[CAPTION] Figure 5: Comparison of conformal outlier detection methods on a tabular dataset (“shuttle”) as a function of the target

[CAPTION] Table 1 summarizes the results for all six datasets. Overall,


<!-- page 9 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
Table 1: Comparison of conformal outlier detection methods on six visual datasets for varying contamination rate r and
target type-I error level α. The empirical type-I error values are averaged across all datasets. The empirical power is
presented relative to the Standard method (higher is better), and averaged across all datasets. Results are averaged across
100 random splits of the data, with standard errors presented in parentheses.
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0317)
0.008 (± 0.0003)
1.0 (± 0.0354)
0.005 (± 0.0003)
1.0 (± 0.0408)
0.004 (± 0.0002)
Oracle (infeasible)
1.166 (± 0.0336)
0.01 (± 0.0003)
1.549 (± 0.0425)
0.01 (± 0.0003)
1.961 (± 0.0531)
0.009 (± 0.0004)
Naive-Trim (invalid)
1.659 (± 0.0342)
0.017 (± 0.0004)
2.79 (± 0.045)
0.027 (± 0.0006)
4.16 (± 0.0596)
0.036 (± 0.0007)
Small-Clean
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
Label-Trim
1.166 (± 0.0336)
0.01 (± 0.0003)
1.517 (± 0.042)
0.01 (± 0.0003)
1.786 (± 0.0498)
0.008 (± 0.0003)
(a) Target type-I error rate α = 0.01
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0174)
0.027 (± 0.0006)
1.0 (± 0.0189)
0.019 (± 0.0005)
1.0 (± 0.0212)
0.015 (± 0.0005)
Oracle (infeasible)
1.062 (± 0.0176)
0.03 (± 0.0006)
1.235 (± 0.0192)
0.029 (± 0.0006)
1.448 (± 0.023)
0.03 (± 0.0007)
Naive-Trim (invalid)
1.146 (± 0.0175)
0.035 (± 0.0006)
1.487 (± 0.0186)
0.043 (± 0.0007)
1.882 (± 0.0224)
0.052 (± 0.0008)
Small-Clean
0.714 (± 0.0448)
0.02 (± 0.0021)
0.869 (± 0.0501)
0.02 (± 0.002)
1.033 (± 0.0613)
0.021 (± 0.0023)
Label-Trim
1.041 (± 0.0177)
0.029 (± 0.0006)
1.139 (± 0.019)
0.025 (± 0.0006)
1.215 (± 0.0226)
0.021 (± 0.0006)
(b) Target type-I error rate α = 0.03
though we anticipate that establishing the theoretical validity
of such a method may not be straightforward.
Another direction related to the above discussion is how to
account for uncertainty in the labeling process. In our cur-
rent formulation, we assume that all calibration points are
labeled as inliers, but some may in fact be outliers. Impor-
tantly, we have no indication of which points are mislabeled,
nor any signal of uncertainty in the labels. Exploring how
to incorporate such uncertainty into our framework could
enhance its practical utility. The line of work presented
in Stutz et al. (2023); Javanmardi et al. (2024); Caprio et al.
(2025) may offer a valuable starting point for such an ex-
tension: it introduces techniques for handling ambiguous
labels, though in the context of multi-class classification
rather than outlier detection.
Notably, our paper also provides practical guidance on how
to annotate data in situations where there is uncertainty
about whether a point is an inlier or an outlier. As indicated
by Lemma 2.2 and supported by our empirical results, when
there is uncertainty about a point’s label, treating it as an
inlier is a conservative strategy that preserves type-I error
control. This observation suggests a compelling connection
between label ambiguity (Stutz et al., 2023; Javanmardi
et al., 2024; Caprio et al., 2025) and contamination in the
reference set, which merits further investigation.
Acknowledgments
M. S. was partly supported by NSF grant DMS 2210637 and
by a Capital One CREDIF Research Award. Y. R. and M. B.
were funded by the European Union (ERC, SafetyBounds,
101163414). Views and opinions expressed are however
those of the authors only and do not necessarily reflect those
of the European Union or the European Research Council
Executive Agency (ERCEA). Neither the European Union
nor the granting authority can be held responsible for them.
This research was also partially supported by the Israel
Science Foundation (ISF grant 729/21). Y. R. acknowledges
additional support from the Career Advancement Fellowship
at the Technion.
References
KDD Cup 1999 Data Set.
https://www.kaggle.
com/mlg-ulb/creditcardfraud.
Not normal-
ized, without duplicates, categorial attributes removed.
Accessed: January, 2021.
Credit Card Fraud Detection Data Set. https://www.
kaggle.com/mlg-ulb/creditcardfraud. Ac-
cessed: January, 2021.
Statlog
(Shuttle)
Data
Set.
http://odds.cs.
stonybrook.edu/shuttle-dataset. Accessed:
January, 2021.
Barber, R. F., Cand`es, E. J., Ramdas, A., and Tibshirani,
R. J. Conformal prediction beyond exchangeability. Ann.
Stat., 51(2):816–845, 2023.
Bates, S., Cand`es, E., Lei, L., Romano, Y., and Sesia, M.
Testing for outliers with conformal p-values. Ann. Stat.,
51(1):149 – 178, 2023.
9


**[Table p9.1]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0317) | 0.008 (± 0.0003) | 1.0 (± 0.0354) | 0.005 (± 0.0003) | 1.0 (± 0.0408) | 0.004 (± 0.0002) |
|  | 1.166 (± 0.0336) | 0.01 (± 0.0003) | 1.549 (± 0.0425) | 0.01 (± 0.0003) | 1.961 (± 0.0531) | 0.009 (± 0.0004) |
|  | 1.659 (± 0.0342) | 0.017 (± 0.0004) | 2.79 (± 0.045) | 0.027 (± 0.0006) | 4.16 (± 0.0596) | 0.036 (± 0.0007) |
|  | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) |
|  | 1.166 (± 0.0336) | 0.01 (± 0.0003) | 1.517 (± 0.042) | 0.01 (± 0.0003) | 1.786 (± 0.0498) | 0.008 (± 0.0003) |


**[Table p9.2]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0174) | 0.027 (± 0.0006) | 1.0 (± 0.0189) | 0.019 (± 0.0005) | 1.0 (± 0.0212) | 0.015 (± 0.0005) |
|  | 1.062 (± 0.0176) | 0.03 (± 0.0006) | 1.235 (± 0.0192) | 0.029 (± 0.0006) | 1.448 (± 0.023) | 0.03 (± 0.0007) |
|  | 1.146 (± 0.0175) | 0.035 (± 0.0006) | 1.487 (± 0.0186) | 0.043 (± 0.0007) | 1.882 (± 0.0224) | 0.052 (± 0.0008) |
|  | 0.714 (± 0.0448) | 0.02 (± 0.0021) | 0.869 (± 0.0501) | 0.02 (± 0.002) | 1.033 (± 0.0613) | 0.021 (± 0.0023) |
|  | 1.041 (± 0.0177) | 0.029 (± 0.0006) | 1.139 (± 0.019) | 0.025 (± 0.0006) | 1.215 (± 0.0226) | 0.021 (± 0.0006) |

[CAPTION] Table 1: Comparison of conformal outlier detection methods on six visual datasets for varying contamination rate r and


<!-- page 10 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
Benjamini, Y. and Hochberg, Y. Controlling the false dis-
covery rate: a practical and powerful approach to multiple
testing. J. R. Stat. Soc. Series B, 57(1):289–300, 1995.
Breunig, M. M., Kriegel, H.-P., Ng, R. T., and Sander, J. Lof:
identifying density-based local outliers. In Proceedings
of the 2000 ACM SIGMOD international conference on
Management of data, pp. 93–104, 2000.
Buitinck, L., Louppe, G., Blondel, M., Pedregosa, F.,
Mueller, A., Grisel, O., Niculae, V., Prettenhofer, P.,
Gramfort, A., Grobler, J., Layton, R., VanderPlas, J.,
Joly, A., Holt, B., and Varoquaux, G. API design for ma-
chine learning software: experiences from the scikit-learn
project. In ECML PKDD Workshop: Languages for Data
Mining and Machine Learning, pp. 108–122, 2013.
Caprio, M., Stutz, D., Li, S., and Doucet, A.
Confor-
malized credal regions for classification with ambigu-
ous ground truth.
Transactions on Machine Learn-
ing Research, 2025. ISSN 2835-8856. URL https:
//openreview.net/forum?id=L7sQ8CW2FY.
Chalapathy, R. and Chawla, S. Deep learning for anomaly
detection: A survey. arXiv preprint arXiv:1901.03407,
2019.
Chen, Y. Pytorch cifar models. https://github.com/
chenyaofo/pytorch-cifar-models. Accessed:
2025-5-17.
Cimpoi, M., Maji, S., Kokkinos, I., Mohamed, S., , and
Vedaldi, A. Describing textures in the wild. In Proceed-
ings of the IEEE Conf. on Computer Vision and Pattern
Recognition (CVPR), 2014.
Clarkson, J., Xu, W., Cucuringu, M., and Reinert, G. Split
conformal prediction under data contamination. In Pro-
ceedings of the Thirteenth Symposium on Conformal and
Probabilistic Prediction with Applications, volume 230
of Proceedings of Machine Learning Research, pp. 5–27.
PMLR, 2024.
Deng, L. The MNIST database of handwritten digit images
for machine learning research. IEEE Signal Processing
Magazine, 2012.
Einbinder, B.-S., Feldman, S., Bates, S., Angelopoulos,
A. N., Gendler, A., and Romano, Y. Label noise ro-
bustness of conformal prediction. Journal of Machine
Learning Research, 25(328):1–66, 2024.
Fannjiang, C., Bates, S., Angelopoulos, A. N., Listgarten, J.,
and Jordan, M. I. Conformal prediction under feedback
covariate shift for biomolecular design. Proceedings of
the National Academy of Sciences, 119(43):e2204569119,
2022.
Farinhas, A., Zerva, C., Ulmer, D. T., and Martins, A. Non-
exchangeable conformal risk control. In International
Conference on Learning Representations, 2024.
Feldman, S. and Romano, Y. Robust conformal prediction
using privileged information. In Conference on Neural
Information Processing Systems, 2024.
Feldman, S., Ringel, L., Bates, S., and Romano, Y. Achiev-
ing risk control in online learning settings. Transactions
on Machine Learning Research, 2023. ISSN 2835-8856.
Gibbs, I. and Cand`es, E. Adaptive conformal inference
under distribution shift. Advances in Neural Information
Processing Systems, 34:1660–1672, 2021.
Gibbs, I. and Cand`es, E. J. Conformal inference for online
prediction with arbitrary distribution shifts. Journal of
Machine Learning Research, 25(162):1–36, 2024.
He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learn-
ing for image recognition. In Proceedings of the IEEE
conference on computer vision and pattern recognition,
pp. 770–778, 2016.
Javanmardi, A., Stutz, D., and H¨ullermeier, E. Conformal-
ized credal set predictors. Advances in Neural Informa-
tion Processing Systems, 37:116987–117014, 2024.
Jiang, X., Liu, J., Wang, J., Nie, Q., Wu, K., Liu, Y., Wang,
C., and Zheng, F. Softpatch: Unsupervised anomaly de-
tection with noisy data. Advances in Neural Information
Processing Systems, 35:15433–15445, 2022.
Krizhevsky, A. and Hinton, G. Learning multiple layers of
features from tiny images. Technical report, University
of Toronto, (0), 2009.
Krizhevsky, A., Nair, V., and Hinton, G. CIFAR-10 and
CIFAR-100 datasets.
URl: https://www. cs. toronto.
edu/kriz/cifar. html, 6(1):1, 2009.
Laxhammar, R. and Falkman, G.
Inductive conformal
anomaly detection for sequential detection of anomalous
sub-trajectories. Annals of Mathematics and Artificial
Intelligence, 74(1-2):67–94, 2015.
Lei, J., Robins, J., and Wasserman, L. Distribution-free
prediction sets. J. Am. Stat. Assoc., 108(501):278–287,
2013.
Liu, F. T., Ting, K. M., and Zhou, Z.-H. Isolation forest.
In 2008 eighth IEEE international conference on data
mining, pp. 413–422. IEEE, 2008.
Makili, L. E., S´anchez, J. A. V., and Dormido-Canto, S.
Active learning using conformal predictors: application
to image classification. Fusion Science and Technology,
62(2):347–355, 2012.
10


<!-- page 11 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
Netzer, Y., Wang, T., Coates, A., Bissacco, A., Wu, B.,
and Ng, A. Y. Reading digits in natural images with
unsupervised feature learning. In NIPS workshop on
deep learning and unsupervised feature learning, volume
2011, pp. 4. Granada, 2011.
Park, J., Moon, J.-H., Ahn, N., and Sohn, K.-A. What
is wrong with one-class anomaly detection?
In Inter-
national Conference on Learning Representations Work-
shops, 2021.
Penso, C., Goldberger, J., and Fetaya, E. Estimating the
conformal prediction threshold from noisy labels. arXiv
preprint arXiv:2501.12749, 2025.
Podkopaev, A. and Ramdas, A. Distribution-free uncertainty
quantification for classification under label shift. In Un-
certainty in artificial intelligence, pp. 844–853. PMLR,
2021.
Prinster, D., Stanton, S. D., Liu, A., and Saria, S. Conformal
validity guarantees exist for any data distribution (and
how to find them). In Forty-first International Conference
on Machine Learning.
Romano, Y., Sesia, M., and Cand`es, E. Classification with
valid and adaptive coverage. Adv. Neural Inf. Process.
Syst., 33, 2020.
Sch¨olkopf, B., Platt, J. C., Shawe-Taylor, J., Smola, A. J.,
and Williamson, R. C. Estimating the support of a high-
dimensional distribution.
Neural computation, 13(7):
1443–1471, 2001.
Sesia, M., Wang, Y. R., and Tong, X. Adaptive conformal
classification with noisy labels. J. R. Stat. Soc. Series B,
pp. qkae114, 2024.
Si, W., Park, S., Lee, I., Dobriban, E., and Bastani, O.
PAC prediction sets under label shift. arXiv preprint
arXiv:2310.12964, 2023.
Stutz, D., Roy, A. G., Matejovicova, T., Strachan, P., Cemgil,
A. T., and Doucet, A. Conformal prediction under am-
biguous ground truth. Transactions on Machine Learn-
ing Research, 2023. ISSN 2835-8856. URL https:
//openreview.net/forum?id=CAd6V2qXxc.
Sun, Y., Guo, C., and Li, Y. React: Out-of-distribution
detection with rectified activations. Advances in Neural
Information Processing Systems, 34:144–157, 2021.
Tibshirani, R. J., Foygel Barber, R., Cand`es, E., and Ramdas,
A. Conformal prediction under covariate shift. Advances
in neural information processing systems, 32, 2019.
Torralba, A., Fergus, R., and Freeman, W. T. 80 million tiny
images: A large data set for nonparametric object and
scene recognition. TPAMI, 2008.
Vovk, V., Gammerman, A., and Shafer, G. Algorithmic
learning in a random world. Springer, 2005.
Xu, K., Chen, R., Franchi, G., and Yao, A. Scaling for
training time and post-hoc out-of-distribution detection
enhancement. In The Twelfth International Conference
on Learning Representations, 2024.
Yang, J., Wang, P., Zou, D., Zhou, Z., Ding, K., Peng, W.,
Wang, H., Chen, G., Li, B., Sun, Y., et al. Openood:
Benchmarking generalized out-of-distribution detection.
Advances in Neural Information Processing Systems, 35:
32598–32611, 2022.
Zaffran, M., F´eron, O., Goude, Y., Josse, J., and Dieuleveut,
A. Adaptive conformal predictions for time series. In In-
ternational Conference on Machine Learning, pp. 25834–
25866. PMLR, 2022.
Zaffran, M., Dieuleveut, A., Josse, J., and Romano, Y.
Conformal prediction with missing values. In Interna-
tional Conference on Machine Learning, volume 202, pp.
40578–40604. PMLR, 2023.
Zaffran, M., Josse, J., Romano, Y., and Dieuleveut, A. Pre-
dictive uncertainty quantification with missing covariates.
arXiv preprint arXiv:2405.15641, 2024.
Zhang, J., Yang, J., Wang, P., Wang, H., Lin, Y., Zhang,
H., Sun, Y., Du, X., Li, Y., Liu, Z., Chen, Y., and Li,
H.
Openood v1.5: Enhanced benchmark for out-of-
distribution detection. Journal of Data-centric Machine
Learning Research, 2024.
Zhao, Z., Cerf, S., Birke, R., Robu, B., Bouchenak, S.,
Mokhtar, S. B., and Chen, L. Y. Robust anomaly de-
tection on unreliable data. In IEEE/IFIP International
Conference on Dependable Systems and Networks, pp.
630–637. IEEE, 2019.
11


<!-- page 12 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
A. Mathematical Proofs
A.1. Auxiliary Technical Results
In this section, we begin by introducing two useful propositions, Proposition A.1 and Proposition A.2, which will be used
later in the proofs presented here.
Proposition A.1. Let D be a dataset containing n scores, and define the threshold ˆQ1−α as
ˆQ1−α := ˆi-th smallest element in D ∪{∞},
where
ˆi := ⌈(1 −α)(n + 1)⌉.
For any test point Xn+1, the following holds:
s(Xn+1) > ˆQ1−α
if and only if
ˆpn+1 ≤α,
where ˆpn+1 is the conformal p-value (2).
Proof of Proposition A.1. The proof follows the definition of conformal p-value from (2), and its relation to the empirical
quantile function:
ˆpn+1 = 1 + Pn
i=1 I[s(Xi) ≥s(Xn+1)]
n + 1
≤α
(i)
⇐⇒
ˆpn+1 = 1 + Pn
i=1 I[s(Xi) ≥s(Xn+1)]
n + 1
≤⌊α(n + 1)⌋
n + 1
⇐⇒
1 +
n
X
i=1
I[s(Xi) ≥s(Xn+1)] ≤⌊α(n + 1)⌋
⇐⇒
1 + n −
n
X
i=1
I[s(Xi) < s(Xn+1)] ≤⌊α(n + 1)⌋
⇐⇒
n
X
i=1
I[s(Xi) < s(Xn+1)] ≥n + 1 −⌊α(n + 1)⌋
(ii)
⇐⇒
n
X
i=1
I[s(Xi) < s(Xn+1)] ≥⌈(1 −α)(n + 1)⌉
(5)
The labeled steps above can be explained as follows.
• (i) The values of ˆpn+1 are discrete, taking values from {
1
n+1,
2
n+1, . . . , 1}. Therefore, ˆpn+1 =
k
n+1 for some
k ∈[n + 1]. We explicitly prove that ˆpn+1 ≤α iff ˆpn+1 ≤⌊α(n+1)⌋
n+1
as follows:
⇐Assume ˆpn+1 ≤⌊α(n+1)⌋
n+1
. Therefore, ˆpn+1 ≤⌊α(n+1)⌋
n+1
≤α(n+1)
n+1
= α.
⇒Assume ˆpn+1 ≤α, then
k
n+1 ≤α. This implies that k ≤α(n + 1). Since k is an integer, it follows that
k ≤⌊α(n + 1)⌋. Therefore, ˆpn+1 =
k
n+1 ≤⌊α(n+1)⌋
n+1
.
• (ii) This step follows directly from the equality n + 1 = ⌈(1 −α)(n + 1)⌉+ ⌊α(n + 1)⌋. We explicitly prove this
equality as follows:
– The term ⌈(1 −α)(n + 1)⌉represents the smallest integer greater than or equal to (1 −α)(n + 1). Hence, we can
write:
⌈(1 −α)(n + 1)⌉= (1 −α)(n + 1) + δ1,
where 0 ≤δ1 < 1.
12


<!-- page 13 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
– Similarly, the term ⌊α(n + 1)⌋represents the largest integer less than or equal to α(n + 1). Thus:
⌊α(n + 1)⌋= α(n + 1) −δ2,
where 0 ≤δ2 < 1.
– Adding these two terms gives:
⌊α(n + 1)⌋+ ⌈(1 −α)(n + 1)⌉= α(n + 1) −δ2 + (1 −α)(n + 1) + δ1 = n + 1 + (δ1 −δ2).
Since ⌊α(n + 1)⌋+ ⌈(1 −α)(n + 1)⌉must be an integer and δ1, δ2 ∈[0, 1), it follows that δ1 −δ2 = 0.
Therefore, ⌊α(n + 1)⌋+ ⌈(1 −α)(n + 1)⌉= n + 1.
To complete the proof, we now show that (5) holds if and only if s(Xn+1) > ˆQ1−α.
⇐Assume s(Xn+1) > ˆQ1−α. By definition, Pn
i=1 I[s(Xi) ≤ˆQ1−α] = ⌈(1 −α)(n + 1)⌉. Then, (5) holds since
n
X
i=1
I[s(Xi) < s(Xn+1)] ≥
n
X
i=1
I[s(Xi) ≤ˆQ1−α] = ⌈(1 −α)(n + 1)⌉.
⇒We prove this direction by contradiction, assuming that (5) holds. Now, suppose that s(Xn+1) ≤ˆQ1−α also holds,
implying that
n
X
i=1
I[s(Xi) < s(Xn+1)] ≤
n
X
i=1
I[s(Xi) < ˆQ1−α]
(i)
< ⌈(1 −α)(n + 1)⌉,
which contradicts the assumption (5). Therefore, we conclude that s(Xn+1) > ˆQ1−α. The last step above can be
explained as follows.
– (i) Recall that by definition, ˆQ1−α is a specific value in {s(Xi)}n
i=1 and Pn
i=1 I[s(Xi) ≤ˆQ1−α] = ⌈(1 −α)(n +
1)⌉. This implies that
n
X
i=1
I[s(Xi) < ˆQ1−α] = ⌈(1 −α)(n + 1)⌉−
n
X
i=1
I[s(Xi) = ˆQ1−α] < ⌈(1 −α)(n + 1)⌉.
In sum, ˆpn+1 ≤α holds if and only if (5) holds, and the latter holds if and only if s(Xn+1) > ˆQ1−α. This completes the
proof.
Proposition A.2. Let D be a dataset containing n scores, and let Sn+1 be a test score. Define the following thresholds:
ˆQ1−α := ˆi-th smallest element in D ∪{∞},
where
ˆi := ⌈(1 −α)(n + 1)⌉.
Similarly, let
ˆQn+1
1−α := ˆi-th smallest element in D ∪{Sn+1}.
It follows that
ˆQ1−α ≥ˆQn+1
1−α almost surely.
Proof of Proposition A.2. Since the largest possible score is ∞, the set D ∪{∞} almost surely contains scores that are
greater or equal to those in D ∪{Sn+1}. Consequently, ˆQ1−α := ˆi-th smallest element in D ∪{∞} is almost surely greater
than or equal to ˆQn+1
1−α := ˆi-th smallest element in D ∪{Sn+1}.
13


<!-- page 14 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
A.2. Explaining the Conservativeness of Standard Conformal p-Values
A.2.1. PROOF OF LEMMA 2.2
Proof of Lemma 2.2. To simplify the notation define the random score Si := s(Xi) for all i ∈Dcal ∪{n + 1}. Throughout
the proof, we refer to the calibration set as the set of nonconformity scores corresponding to the calibration points. Without
loss of generality, assume that the inliers in Dcal are located at the first n0 indices. Let Dinlier = [n0] denote the set of
indices corresponding to the inlier scores. Consequently, define Doutlier = {n0 + 1, n0 + 2, . . . , n} as the set of indices
corresponding to the outlier scores in Dcal. We assume the scores have no ties (which can always be achieved by adding a
negligible random noise to the scores output by any model).
Given a fixed realization of the score vector (s1, . . . , sn, sn+1) ∈Rn+1, define the following two events:
• Ein: the unordered set of inlier scores, including the test score, is {S1, . . . , Sn0, Sn+1} = {s1, . . . , sn0, sn+1};
• Eout: the unordered set of outlier scores is {Sn0+1, . . . , Sn} = {sn0+1, . . . , sn}.
Under the setup defined in (3), when H0 is true, the test score Sn+1 and the inlier scores in the calibration set are i.i.d. from
P0. Therefore, by exchangeability, the following holds for each inlier index i ∈Dinlier ∪{n + 1}:
P (Sn+1 = si | Ein, Eout) =
1
n0 + 1.
(6)
Since the calibration scores are almost-surely distinct, the probability of a null test point obtaining any outlier score is zero.
Therefore, for each outlier index j ∈Doutlier:
P (Sn+1 = sj | Ein, Eout) = 0.
(7)
To obtain an upper bound on the type-I error rate, P (ˆpn+1 ≤α), we use the equivalence established in Proposition A.1.
According to this result, the following holds:
P (ˆpn+1 ≤α) = P
 
Sn+1 > ˆQcal
1−α
 
,
where ˆQcal
1−α is the ˆical-th smallest element in {Si}n
i=1 ∪{∞} and ˆical := ⌈(1 −α)(n + 1)⌉.
Moreover, define ˆQn+1
1−α as the ˆical-th smallest score in {Si}n+1
i=1 . By Proposition A.2, ˆQcal
1−α ≥ˆQn+1
1−α almost surely.
Now, we obtain an upper bound for P
 
Sn+1 > ˆQcal
1−α | Ein, Eout
 
, where the probability is taken over random permutations
14


<!-- page 15 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
of the scores conditional on Ein, Eout.
P
 
Sn+1 > ˆQcal
1−α | Ein, Eout
 
≤P
 
Sn+1 > ˆQn+1
1−α | Ein, Eout
 
= E
h
I
h
Sn+1 > ˆQn+1
1−α
i
| Ein, Eout
i
=
X
i∈Dinlier ∪{n+1}
E
h
I [Sn+1 = si] I
h
si > ˆQn+1
1−α
i
| Ein, Eout
i
(i)
=
X
i∈Dinlier ∪{n+1}
I
h
si > ˆQn+1
1−α
i
P (Sn+1 = si | Ein, Eout)
=
1
n0 + 1
X
i∈Dinlier ∪{n+1}
I
h
si > ˆQn+1
1−α
i
(ii)
≤
1
n0 + 1
 
α(n + 1) −
X
i∈Doutlier
I
h
si > ˆQn+1
1−α
i!
= α +
1
n0 + 1
 
αn1 −
X
i∈Doutlier
I
h
si > ˆQn+1
1−α
i!
= α −
n1
n0 + 1
 
1 −α −1
n1
X
i∈Doutlier
I
h
si ≤ˆQn+1
1−α
i!
(iii)
≤α −
n1
n0 + 1
 
1 −α −1
n1
X
i∈Doutlier
I
h
si ≤ˆQcal
1−α
i!
= α −
n1
n0 + 1
 
1 −α −ˆF1
 
ˆQcal
1−α
  
,
where ˆF1 is the empirical CDF of the outlier scores. The labeled steps above can be explained as follows.
• (i) I
h
si > ˆQn+1
1−α
i
is measurable with respect to the σ-algebra generated by Ein, Eout. This follows because ˆQn+1
1−α is
the ˆical-th smallest element of {s1, . . . , sn+1}, which is fully determined by these variables. Thus, we can pull it out of
the expectation.
• (ii) ˆQn+1
1−α is the ˆical-th smallest score in {s1, . . . , sn, sn+1} and [n + 1] = Doutlier ∪Dinlier ∪{n + 1}. By definition,
X
i∈Dinlier ∪{n+1}
I
h
si > ˆQn+1
1−α
i
+
X
i∈Doutlier
I
h
si > ˆQn+1
1−α
i
=
n+1
X
i=1
I
h
si > ˆQn+1
1−α
i
= ⌊α(n + 1)⌋≤α(n + 1)
and therefore,
X
i∈Dinlier ∪{n+1}
I
h
si > ˆQn+1
1−α
i
≤α(n + 1) −
X
i∈Doutlier
I
h
si > ˆQn+1
1−α
i
• (iii) Since ˆQcal
1−α ≥ˆQn+1
1−α almost surely, increasing the threshold (i.e., using ˆQcal
1−α) results in an equal or larger value
of the sum.
15


<!-- page 16 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
Now, we can derive an upper bound for P (ˆpn+1 ≤α) as follows:
P (ˆpn+1 ≤α) = P
 
Sn+1 > ˆQcal
1−α
 
= E
h
P
 
Sn+1 > ˆQcal
1−α | Ein, Eout
 i
≤E
 
α −
n1
n0 + 1
 
1 −α −ˆF1
 
ˆQcal
1−α
    
= α −
n1
n0 + 1
 
1 −α −E
h
ˆF1
 
ˆQcal
1−α
 i 
,
with this expectation being taken over different realizations of the inlier and outlier nonconformity scores.
A.2.2. AN ALTERNATIVE VIEW BASED ON MIXTURE DISTRIBUTIONS
Next, we provide an additional theoretical result concerning the conservativeness of conformal outlier detection methods, to
supplement the result presented in Section 2.3 from a point of view closer to that of Sesia et al. (2024).
Specifically, we consider a contaminated calibration set, Dcal, which may include both inliers (samples i.i.d. from P0) and
outliers (samples i.i.d. from P1 ̸= P0). The goal remains to test the null hypothesis H0 that a new data point Xn+1 is an
inlier, independently sampled from P0.
This setup differs from (3) in that the calibration set is drawn from a mixed distribution, where the proportion of outliers in
the population is denoted by δ ∈[0, 1). Hence, the numbers of inliers and outliers in the calibration set are random, rather
than fixed. Formally, this setup is expressed as:
Xi
i.i.d.
∼Pmixed = (1 −δ) · P0 +δ · P1,
∀i ∈Dcal,
H0 : Xn+1
ind.
∼P0 .
(8)
Let F0 and F1 denote the CDFs of P0 and P1, respectively, and ˆQcal
1−α represent the ⌈(1 −α)(n + 1)⌉-th smallest score in
the calibration set.
Corollary A.3 (Conservativeness). Under the setup defined in (8), if H0 is true, then, for any α ∈(0, 1),
P (ˆpn+1 ≤α) ≤α −δ E
h
F0( ˆQcal
1−α) −F1( ˆQcal
1−α)
i
.
Corollary A.3 reformulates Theorem 1 in Sesia et al. (2024) under the setup in (8). This result quantifies the behavior of
conformal outlier detection methods in the presence of contaminated data and establishes guarantees on the type-I error rate.
This result complements our analysis of the conservativeness of these methods.
Proof of Corollary A.3. The proof adapts the argument of Theorem 1 in Sesia et al. (2024) to the outlier detection setting
considered here. Specifically, we follow the structure of the original proof, making adjustments to account for the presence
of inliers and outliers in the calibration set.
By Proposition A.1, we have
ˆpn+1 ≤α ⇐⇒Sn+1 > ˆQcal
1−α.
16


<!-- page 17 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
Under the null, we upper bound P (ˆpn+1 ≤α) as follows:
P0 (ˆpn+1 ≤α) = P0 (ˆpn+1 ≤α) + Pmixed (ˆpn+1 ≤α) −Pmixed (ˆpn+1 ≤α)
= Pmixed (ˆpn+1 ≤α) −[Pmixed (ˆpn+1 ≤α) −P0 (ˆpn+1 ≤α)]
≤α −[Pmixed (ˆpn+1 ≤α) −P0 (ˆpn+1 ≤α)]
= α −[((1 −δ) P0 (ˆpn+1 ≤α) + δ P1 (ˆpn+1 ≤α)) −P0 (ˆpn+1 ≤α)]
= α −δ [P1 (ˆpn+1 ≤α) −P0 (ˆpn+1 ≤α)]
= α −δ
h
P1
 
Sn+1 > ˆQcal
1−α
 
−P0
 
Sn+1 > ˆQcal
1−α
 i
= α −δ
h
P0
 
Sn+1 ≤ˆQcal
1−α
 
−P1
 
Sn+1 ≤ˆQcal
1−α
 i
= α −δ E
h
P0
 
Sn+1 ≤ˆQcal
1−α | Dcal
 
−P1
 
Sn+1 ≤ˆQcal
1−α | Dcal
 i
= α −δ E
h
F0
 
ˆQcal
1−α
 
−F1
 
ˆQcal
1−α
 i
.
A.3. Validity of the Label-Trim Method
A.3.1. PROOF OF THEOREM 3.1 — MAIN STEPS
Proof of Theorem 3.1. As in the proof of Lemma 2.2, define the random score Si := s(Xi) for all i ∈Dcal ∪{n + 1}.
By Proposition A.1, for any fixed α ∈(0, 1), the probability of a type-I error, P
 ˆpLT
n+1 ≤α
 
, can be expressed as
P
 ˆpLT
n+1 ≤α
 
= P
 
Sn+1 > ˆQLT
1−α
 
.
Consider the augmented set {Si}i∈DLT
cal ∪{Sn+1}, which includes the test score Sn+1. Define ˆQLT,n+1
1−α
as follows:
ˆQLT,n+1
1−α
:= ˆiLT-th smallest element in {Si}i∈DLT
cal ∪{Sn+1}.
By Proposition A.2, it holds that ˆQLT
1−α ≥ˆQLT,n+1
1−α
almost surely.
Now, consider an imaginary “mirror” version of this method that applies the label-trim algorithm with two key differences:
• it uses a larger labeling budget, ˜m = m + 1;
• it treats {Si}n+1
i=1 as the calibration set instead of {Si}n
i=1—that is, it includes the test point in the annotation process,
preserving the exchangeability with the calibration inliers.
Let ˜D
LT
cal ∪{n + 1} denote the indices of the trimmed augmented calibration set produced by the mirror procedure, let
˜Dlabeled denote the indices of the corresponding labeled data points, and define ˜D
inlier
labeled, ˜D
outlier
labeled as the corresponding
subsets of inliers and outliers, respectively. Under this mirror procedure, the empirical quantile ˆQLT,n+1
1−α
corresponds to
˜QLT,n+1
1−α
:= ˜iLT-th smallest element in {Si}i∈˜D
LT
cal ∪{Sn+1},
where ˜iLT := ⌈(1 −α)(˜nLT + 1)⌉and ˜nLT := |˜D
LT
cal|.
By construction of this mirror procedure, ˜iLT ≤ˆiLT almost surely, because ˜nLT ≤nLT almost surely and thus
˜iLT =
 
(1 −α)(˜nLT + 1)
 
≤⌈(1 −α)(nLT + 1)⌉= ˆiLT.
Using the fact that ˜iLT ≤ˆiLT almost surely, we prove in Appendix A.3.2 that, almost surely,
˜QLT,n+1
1−α
≤ˆQLT,n+1
1−α
.
(9)
17


<!-- page 18 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
Since we already knew that ˆQLT
1−α ≥ˆQLT,n+1
1−α
, this implies:
ˆQLT
1−α ≥ˆQLT,n+1
1−α
≥˜QLT,n+1
1−α
.
(10)
Therefore, the type-I error rate of the Label-Trim approach can be bounded from above by the type-I error rate of the mirror
procedure, which can be studied with an approach similar to that of the proof of Lemma 2.2.
Let ˜D
LT
outlier denote the outlier indices remaining in ˜D
LT
cal, with ˜nLT
1
= |˜D
LT
outlier|. As in the proof of Lemma 2.2, define Ein
and Eout as two unordered realizations of the inlier and outlier scores in {Si}n+1
i=1 , respectively. Appendix A.3.3 proves that
P
 
Sn+1 > ˆQLT
1−α | Ein, Eout, ˜Dlabeled, ˜D
inlier
labeled
 
≤α +
1
n0 + 1 −
ˆnLT
1
n0 + 1
 
(1 −α) −ˆF LT
1 ( ˆQLT
1−α)
 
,
(11)
from which it follows immediately, by marginalizing over Ein, Eout, ˜Dlabeled, and ˜D
inlier
labeled, that
P
 
Sn+1 > ˆQLT
1−α
 
≤E
 
α +
1
n0 + 1 −
ˆnLT
1
n0 + 1
 
(1 −α) −ˆF LT
1
 
ˆQLT
1−α
   
.
A.3.2. PROOF OF THEOREM 3.1 — PROOF OF EQUATION (9)
Proof. We prove (9) by analyzing two distinct cases, depending on whether Sn+1 is among the m + 1 largest scores or not.
Recall that m ≤α(n + 1), by assumption, and nLT ≤n. It is easy to see that this implies that, almost surely,
ˆiLT = ⌈(1 −α)(nLT + 1)⌉≤n + 1 −(m + 1).
(12)
The mirror procedure labels and potentially removes the largest m + 1 scores out of the n + 1 total scores. Thus, ˆQLT,n+1
1−α
is smaller than all the outliers removed during trimming, i.e., ˆQLT,n+1
1−α
< Sj for all Sj ∈˜D
outlier
labeled.
• Suppose Sn+1
is among the m + 1 largest scores in {Si}n+1
i=1 .
This case is illustrated below:
In this scenario, the mirror trimming approach ad-
ditionally labels the test score Sn+1, which, under
the null hypothesis, is an inlier and is thus not
removed. As a result, both trimming procedures
yield the same set; i.e., ˜D
LT
cal = DLT
cal. Conse-
quently, ˆQLT,n+1
1−α
= ˜QLT,n+1
1−α
.
• Suppose Sn+1 is not among the m + 1 largest scores in {Si}n+1
i=1 . Within this case, there are two sub-cases to consider.
– The (m + 1)-th largest score in an inlier:
In this case, the mirror trimming approach addi-
tionally labels an inlier, which is not removed.
As a result, both trimming procedures yield the
same set; i.e., ˜D
LT
cal = DLT
cal.
Consequently,
ˆQLT,n+1
1−α
= ˜QLT,n+1
1−α
.
18


<!-- page 19 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
– The (m + 1)-th largest score in an outlier:
This is the interesting case where ˜nLT = nLT −1
and the set ˜D
LT
cal contains one fewer outlier score
than DLT
cal. It follows from (12) that the threshold
ˆQLT,n+1
1−α
is smaller than the m + 1 largest scores.
Then, in this region, DLT
cal and ˜D
LT
cal contain the
same scores. Therefore, the ˆiLT-th smallest score
corresponds to the same score in both DLT
cal ∪{n +
1} and ˜D
LT
cal∪{n+1}. Since˜iLT ≤ˆiLT, it follows
that ˜QLT,n+1
1−α
≤ˆQLT,n+1
1−α
.
A.3.3. PROOF OF THEOREM 3.1 — PROOF OF EQUATION (11)
Proof.
P
 
Sn+1 > ˆQLT
1−α | Ein, Eout, ˜Dlabeled, ˜D
inlier
labeled
 
≤P
 
Sn+1 > ˜QLT,n+1
1−α
| Ein, Eout, ˜Dlabeled, ˜D
inlier
labeled
 
=
X
i∈Dinlier ∪{n+1}
E
h
I
h
si > ˜QLT,n+1
1−α
i
· I [Sn+1 = si] | Ein, Eout, ˜Dlabeled, ˜D
inlier
labeled
i
(i)
=
X
i∈Dinlier ∪{n+1}
I
h
si > ˜QLT,n+1
1−α
i
· P
 
Sn+1 = si | Ein, Eout, ˜Dlabeled, ˜D
inlier
labeled
 
(ii)
=
X
i∈Dinlier ∪{n+1}
I
h
si > ˜QLT,n+1
1−α
i
· P (Sn+1 = si | Ein, Eout)
=
1
n0 + 1
X
i∈Dinlier ∪{n+1}
I
h
si > ˜QLT,n+1
1−α
i
(iii)
≤
1
n0 + 1


α(˜nLT + 1) −
X
i∈˜D
LT
outlier
I
h
si > ˜QLT,n+1
1−α
i



(iv)
≤
1
n0 + 1

α(nLT + 1) −
X
i∈DLT
outlier
I
h
si > ˜QLT,n+1
1−α
i
+ 1


(v)
≤
1
n0 + 1

α(nLT + 1) −
X
i∈DLT
outlier
I
h
si > ˆQLT
1−α
i
+ 1


=
1
n0 + 1

α(nLT + 1) −ˆnLT
1
+
X
i∈DLT
outlier
I
h
si ≤ˆQLT
1−α
i
+ 1


= α +
1
n0 + 1 −
ˆnLT
1
n0 + 1
 
(1 −α) −ˆF LT
1 ( ˆQLT
1−α)
 
.
The labeled steps above can be explained as follows.
• (i) I
h
si > ˜QLT,n+1
1−α
i
is measurable with respect to the σ-algebra generated by Ein, Eout, ˜Dlabeled, and ˜D
inlier
labeled since
˜QLT,n+1
1−α
is the ˜iLT-th smallest element of {s1, . . . , sn+1} \
 
˜Dlabeled \ ˜D
inlier
labeled
 
.
19


<!-- page 20 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
• (ii) The mirror procedure is applied on {Si}n+1
i=1 , preserving the exchangeability of the test score Sn+1 with the
calibration inliers. Hence, the resulting labeled sets ˜Dlabeled and ˜D
inlier
labeled contain no additional information about
Sn+1 beyond Ein, Eout.
• (iii) By definition, ˜QLT,n+1
1−α
is the ˜iLT-th smallest element of {Si}i∈˜D
LT
cal ∪{Sn+1}, where ˜iLT = ⌈(1 −α)(˜nLT + 1)⌉.
Consequently, ⌊α(˜nLT + 1)⌋scores in {Si}i∈˜D
LT
cal ∪{Sn+1} are larger than ˜QLT,n+1
1−α
.
• (iv) The set ˜D
LT
outlier is either equal to DLT
outlier or contains one fewer outlier, and ˜nLT ∈{nLT, nLT −1} almost surely.
• (v) Recall from (10) that ˆQLT
1−α ≥˜QLT,n+1
1−α
almost surely.
20


<!-- page 21 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
B. Supplementary Experiments and Implementation Details
B.1. Datasets
Table 2 summarizes details of the three tabular benchmark datasets. For all tabular datasets, we perform random subsampling
to construct contaminated train, calibration, and test sets. Specifically, for the shuttle and KDDCup99 datasets, the train set
contains 5,000 samples, and the calibration set contains 2,500 samples, both with a contamination rate of r = 3%, unless
stated otherwise. The inlier and outlier test sets consist of 950 and 50 samples, respectively. For the credit-card dataset, the
train set contains 2,000 samples, while the calibration and test sets follow the same setup as the Shuttle and KDDCup99
datasets.
Table 2: Summary of tabular datasets
Dataset
Shuttle (shu)
Credit-card (cre)
KDDCup99 (KDD)
Total Samples
58,000
284,807
494,020
Number of Outliers
12,414
492
396,743
Number of Features
9
29
41
For visual datasets, we use the OpenOOD benchmark (Zhang et al., 2024; Yang et al., 2022). Specifically, for each
dataset and contamination rate, we perform a one-time training of an outlier detection model, which operates on feature
representations extracted from a pre-trained backbone. We include results with the following model configurations:
• ReAct with ResNet-18: ReAct (Sun et al., 2021) operates on feature representations extracted from a pre-trained
ResNet-18 model (Zhang et al., 2024; He et al., 2016). The model applies a percentile-based threshold (set to 90%)
to truncate activations, where the threshold is computed on the contaminated train set. These truncated activations
then pass through the fully connected layer of the pre-trained ResNet-18. The outlier score is computed using an
energy-based log-sum-exp function applied to these truncated activations.
• ReAct with VGG-19: Same as above, but with a pre-trained VGG-19 backbone (Chen) instead of ResNet-18.
• SCALE with ResNet-18: SCALE (Xu et al., 2024) operates on feature representations extracted from a pre-trained
ResNet-18 model (Zhang et al., 2024; He et al., 2016). The model rescales the activations using a sample-specific
factor, defined as the sum of all activations divided by the sum of activations below a certain percentile (set to 65%).
Similar to ReAct, the outlier score is computed using an energy-based log-sum-exp function applied to the rescaled
activations.
After training, we save the outlier scores for the remaining outlier samples and the CIFAR-10 test set. We randomly
subsample this pool of scores to construct the calibration and test sets, ensuring that all sets are disjoint.
The sizes of the train and calibration sets are 2,000 and 3,000, respectively, with the same contamination rate. The inlier and
outlier test sets consist of 950 and 50 samples, respectively.
B.2. Additional Experiments
B.2.1. TABULAR DATASETS
In this section, we provide additional real-data experiments conducted on the credit card (cre) and KDDCup99 datasets,
complementing the analysis provided for the shuttle dataset in the main manuscript. Figures 6 to 12 corresponds to and
extends the figures presented in the main text.
We further report supplementary results across all three tabular datasets considered in this work, including: additional outlier
detection models (Figures 13 and 14); higher contamination levels (Figure 15); strategic outlier injection (Figures 16 to 19);
and test-time distributional shifts in the outlier population (Figure 20).
Results as a function of the contamination rate
In Figure 3 of the main manuscript, we analyze the performance of
conformal outlier methods as a function of the contamination rate r. Here, we repeat the same experiment on the credit-card
21

[CAPTION] Table 2 summarizes details of the three tabular benchmark datasets. For all tabular datasets, we perform random subsampling

[CAPTION] Table 2: Summary of tabular datasets


<!-- page 22 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
and KDDCup99 datasets (Figures 6 and 7). The performance trends are similar to the one presented in the main manuscript:
both the Standard and Small-Clean methods achieve valid type-I error rate but exhibit conservative behavior. In
contrast, the Naive-Trim method fails to control the type-I error rate. The Label-Trim method attains improved power
while practically controlling the type-I error at level α.
0.0
0.01
0.02
0.03
0.04
0.05
Contamination rate
0.01
0.02
0.03
Type-1-Error
0.0
0.01
0.02
0.03
0.04
0.05
Contamination rate
0.2
0.4
0.6
0.8
Power
0.0
0.01
0.02
0.03
0.04
0.05
Contamination rate
0
25
50
75
100
125
Trimmed
Standard
Oracle (infeasible)
Naive-Trim (invalid)
Small-Clean
Label-Trim
Target type-I level
Labeling budget
# outliers
Figure 6: Comparison of conformal outlier detection methods on real dataset “credit-card” as a function of the contamination
rate r. Other details are as in Figure 3.
0.0
0.01
0.02
0.03
0.04
0.05
Contamination rate
0.01
0.02
0.03
0.04
0.05
Type-1-Error
0.0
0.01
0.02
0.03
0.04
0.05
Contamination rate
0.2
0.4
0.6
0.8
Power
0.0
0.01
0.02
0.03
0.04
0.05
Contamination rate
0
25
50
75
100
125
Trimmed
Standard
Oracle (infeasible)
Naive-Trim (invalid)
Small-Clean
Label-Trim
Target type-I level
Labeling budget
# outliers
Figure 7: Comparison of conformal outlier detection methods on real dataset “KDDCup99” as a function of the contamination
rate r. Other details are as in Figure 3.
Results as a function of the labeling budget
In Figure 4 of the main manuscript, we evaluate the performance of the
Label-Trim, Small-Clean and, Oracle methods as a function of the labeling budget m. Figures 8 and 9 extend
this analysis to the credit-card and KDDCup99 datasets, respectively. Consistent with the trends observed in Figure 4,
increasing the labeling budget improves the performance of the Label-Trim method in terms of both type-I error and
power. Notably, although the condition in Theorem 3.1 is no longer satisfied for m > 50, the Label-Trim method still
maintains valid type-I error control at the desired level α.
For the KDDCup99 dataset, however, we observe that the Small-Clean method shows higher power than the Oracle
across several labeling budgets (m ≥55). This is due to the high variability in the dataset and the small sample size used by
this method, resulting in significant variance in type-I error. To illustrate this, Figure 10 presents a box plot showing the
variability and instability of the Small-Clean method in this regime. While this leads to a higher average power, this
variability is undesirable in practice.
10
20
30
40
50
60
70
80
90
100
Labeling budget
0.000
0.005
0.010
0.015
0.020
0.025
Type-1-Error
10
20
30
40
50
60
70
80
90
100
Labeling budget
0.0
0.2
0.4
0.6
Power
10
20
30
40
50
60
70
80
90
100
Labeling budget
20
40
60
80
100
Trimmed
Oracle (infeasible)
Small-Clean
Label-Trim
Target type-I level
Labeling budget
# outliers
Figure 8: Performance on real dataset “credit-card” as a function of the labeling budget m. The contamination rate is
r = 0.03. Other details are as in Figure 3.
22


**[Table p22.1]**
| Type-1-Error 0.03 0.02 0.01 0.0 0.01 0.02 Contamin | 0.8 0.6 Power 0.4 0.2 0.03 0.04 0.05 0.0 0.01 0.02 0.03 0.04 0.0 ation rate Contamination rate |  | 75100125 Trimmed 50 25 0 5 0.0 0.01 0.02 0.03 0.04 0.05 Contamination rate | Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim Target type-I level Labeling budget # outliers |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |


**[Table p22.2]**
| 0.010.020.030.040.05 Type-1-Error 0.0 0.01 0.02 Contamin | 0.8 Power 0.6 0.4 0.2 0.03 0.04 0.05 0.0 0.01 0.02 0.03 0.04 0.0 ation rate Contamination rate |  | 75100125 Trimmed 50 25 0 5 0.0 0.01 0.02 0.03 0.04 0.05 Contamination rate | Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim Target type-I level Labeling budget # outliers |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |


**[Table p22.3]**
| 0.0000.0050.0100.01 Type-1- 10 20 30 40 50 Labelin | Pow 0.4 0.2 0.0 60 70 80 90 100 10 20 30 40 50 60 70 80 90 100 g budget Labeling budget | Trimm 60 40 20 10 20 30 40 50 60 70 80 90 100 Labeling budget | Target type-I level Labeling budget # outliers |
| --- | --- | --- | --- |

[CAPTION] Figure 6: Comparison of conformal outlier detection methods on real dataset “credit-card” as a function of the contamination

[CAPTION] Figure 7: Comparison of conformal outlier detection methods on real dataset “KDDCup99” as a function of the contamination

[CAPTION] Figure 8: Performance on real dataset “credit-card” as a function of the labeling budget m. The contamination rate is


<!-- page 23 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
10
20
30
40
50
60
70
80
90
100
Labeling budget
0.000
0.005
0.010
0.015
0.020
0.025
Type-1-Error
10
20
30
40
50
60
70
80
90
100
Labeling budget
0.0
0.1
0.2
0.3
Power
10
20
30
40
50
60
70
80
90
100
Labeling budget
20
40
60
80
100
Trimmed
Oracle (infeasible)
Small-Clean
Label-Trim
Target type-I level
Labeling budget
# outliers
Figure 9: Performance on real dataset “KDDCup99” as a function of the labeling budget m. The contamination rate is
r = 0.03. Results are averages across 400 random splits of the data. Other details are as in Figure 3.
55
65
75
85
95
105
Labeling budget
0.00
0.02
0.04
0.06
0.08
0.10
Type-1-Error
55
65
75
85
95
105
Labeling budget
0.2
0.4
0.6
0.8
1.0
Power
Oracle (infeasible)
Small-Clean
Label-Trim
Target type-I level
Labeling budget
# outliers
Figure 10: Performance on real dataset “KDDCup99” as a function of the labeling budget m. The contamination rate is
r = 0.03. Results are averages across 400 random splits of the data. Other details are as in Figure 3.
Results as a function of the target type-I error level
In Figure 5 we examine the performance of conformal outlier
detection methods as a function of the target type-I error level α. Here, we replicate the experiments on the credit-card and
KDDCup99 datasets (Figures 11 and 12). In line with the trends observed in Figure 5, the Label-Trim method performs
well at low type-I error rates. Notably, for α = 0.01, the Label-Trim method outperforms the baselines, while practically
controlling the type-I error at level α, even though the condition of Theorem 3.1 is not satisfied in this case. This highlights
the robustness of our approach.
0.01
0.02
0.03
0.04
0.05
0.00
0.02
0.04
0.06
Type-1-Error
0.01
0.02
0.03
0.04
0.05
0.0
0.2
0.4
0.6
0.8
Power
Standard
Oracle (infeasible)
Naive-Trim (invalid)
Small-Clean
Label-Trim
Target type-I level
Figure 11: Comparison of conformal outlier detection methods on real dataset “credit-card” as a function of the target type-I
error rate α. The contamination rate r is fixed to 3%; other details are as in Figure 3.
23


**[Table p23.1]**
| 0.0000.0050.0100.0150.0200.025 Type-1-Error 10 20 30 40 50 60 70 Labeling bu | 0.3 Power 0.2 0.1 0.0 80 90 100 10 20 30 40 50 60 dget Labeling b | 80100 Trimmed 60 40 20 70 80 90 100 10 20 30 40 50 60 70 80 90 100 udget Labeling budget |  |
| --- | --- | --- | --- |
|  |  |  |  |


**[Table p23.2]**
| Power 75 85 95 105 Labeling budget | 1.0 0.8 0.6 0.4 0.2 55 65 75 85 95 Labeling budget | Oracle (infeasible) Small-Clean Label-Trim Target type-I level Labeling budget # outliers 105 |
| --- | --- | --- |


**[Table p23.3]**
| 0.01 0.02 0.03 0.04 0.05 Power 0 0 0 0 0 | .2 .4 .6 .8 .0 0.01 0.02 0.03 0.04 | Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim Target type-I level 0.05 |
| --- | --- | --- |

[CAPTION] Figure 9: Performance on real dataset “KDDCup99” as a function of the labeling budget m. The contamination rate is

[CAPTION] Figure 10: Performance on real dataset “KDDCup99” as a function of the labeling budget m. The contamination rate is

[CAPTION] Figure 11: Comparison of conformal outlier detection methods on real dataset “credit-card” as a function of the target type-I


<!-- page 24 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
0.01
0.02
0.03
0.04
0.05
0.00
0.02
0.04
0.06
Type-1-Error
0.01
0.02
0.03
0.04
0.05
0.0
0.2
0.4
0.6
Power
Standard
Oracle (infeasible)
Naive-Trim (invalid)
Small-Clean
Label-Trim
Target type-I level
Figure 12: Comparison of conformal outlier detection methods on real dataset “KDDCup99” as a function of the target
type-I error rate α. The contamination rate r is fixed to 3%; other details are as in Figure 3.
Results with additional outlier detection models
Here, we consider two additional outlier detection models: Local
Outlier Factor (LOF) with 100 estimators and One-Class Support Vector Machine (OC-SVM) with an RBF kernel, both
implemented via scikit-learn. Figures 13 and 14 present the performance of all methods when applied with LOF and
OC-SVM, respectively. As can be seen, the proposed Label-Trim controls the type-I error across both models. When the
outlier detection model is less discriminative (e.g., LOF or OC-SVM on the KDDCup99 dataset), the candidate set becomes
less effective and power decreases. Nevertheless, our method improves over the baselines, including the Small-Clean
method.
0.0
0.01
0.02
0.03
0.04
0.05
Contamination rate
0.01
0.02
0.03
0.04
Type-1-Error
0.0
0.01
0.02
0.03
0.04
0.05
Contamination rate
0.2
0.4
0.6
0.8
Power
0.0
0.01
0.02
0.03
0.04
0.05
Contamination rate
0
25
50
75
100
125
Trimmed
(a) shuttle
0.0
0.01
0.02
0.03
0.04
0.05
Contamination rate
0.01
0.02
0.03
0.04
Type-1-Error
0.0
0.01
0.02
0.03
0.04
0.05
Contamination rate
0.2
0.4
0.6
Power
0.0
0.01
0.02
0.03
0.04
0.05
Contamination rate
0
25
50
75
100
125
Trimmed
(b) credit-card
0.0
0.01
0.02
0.03
0.04
0.05
Contamination rate
0.02
0.04
0.06
Type-1-Error
0.0
0.01
0.02
0.03
0.04
0.05
Contamination rate
0.00
0.05
0.10
0.15
0.20
0.25
Power
0.0
0.01
0.02
0.03
0.04
0.05
Contamination rate
0
25
50
75
100
125
Trimmed
Standard
Oracle (infeasible)
Naive-Trim (invalid)
Small-Clean
Label-Trim
Target type-I level
Labeling budget
# outliers
(c) KDDCup99
Figure 13: Comparison of conformal outlier detection methods on real datasets as a function of the contamination rate r. All
methods utilize a Local Outlier Factor model. Other details are as in Figure 3 in the main manuscript.
24


**[Table p24.1]**
| Type-1-Error 0.00 0.02 0.04 0.06 0.01 0.02 0.03 0.04 0.05 | Power 0.0 0.2 0.4 0.6 0.01 0.02 0.03 0.04 0.05 | Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim Target type-I level |
| --- | --- | --- |
|  |  |  |


**[Table p24.2]**
| 0.04 Type-1-Error 0.03 0.02 0.01 0.0 0.01 0.02 0.03 0.04 0.05 Contamination rate | 0.8 Power 0.6 0.4 0.2 0.0 0.01 0.02 0.03 0.04 0.05 Contamination rate | 75100125 Trimmed 50 25 0 0.0 0.01 0.02 0.03 0.04 0.05 Contamination rate |
| --- | --- | --- |


**[Table p24.3]**
| 0.04 Type-1-Error 0.03 0.02 0.01 0.0 0.01 0.02 0.03 0.04 0.05 Contamination rate | 0.6 Power 0.4 0.2 0.0 0.01 0.02 0.03 0.04 0.05 Contamination rate | 75100125 Trimmed 50 25 0 0.0 0.01 0.02 0.03 0.04 0.05 Contamination rate |
| --- | --- | --- |


**[Table p24.4]**
| 0.06 Type-1-Error 0.04 0.02 0.0 0.01 0.02 0.03 0.04 0.05 Contamination rate | 0.000.050.100.150.200.25 Power 0.0 0.01 0.02 0.03 0.04 0.05 Contamination rate | 75100125 Trimmed 50 25 0 0.0 0.01 0.02 0.03 0.04 0.05 Contamination rate | Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim Target type-I level Labeling budget # outliers |
| --- | --- | --- | --- |
|  |  |  |  |

[CAPTION] Figure 12: Comparison of conformal outlier detection methods on real dataset “KDDCup99” as a function of the target

[CAPTION] Figure 13: Comparison of conformal outlier detection methods on real datasets as a function of the contamination rate r. All


<!-- page 25 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
0.0
0.01
0.02
0.03
0.04
0.05
Contamination rate
0.01
0.02
0.03
0.04
Type-1-Error
0.0
0.01
0.02
0.03
0.04
0.05
Contamination rate
0.1
0.2
0.3
0.4
0.5
0.6
Power
0.0
0.01
0.02
0.03
0.04
0.05
Contamination rate
0
25
50
75
100
125
Trimmed
(a) shuttle
0.0
0.01
0.02
0.03
0.04
0.05
Contamination rate
0.02
0.04
0.06
Type-1-Error
0.0
0.01
0.02
0.03
0.04
0.05
Contamination rate
0.025
0.050
0.075
0.100
0.125
Power
0.0
0.01
0.02
0.03
0.04
0.05
Contamination rate
0
25
50
75
100
125
Trimmed
(b) credit-card
0.0
0.01
0.02
0.03
0.04
0.05
Contamination rate
0.00
0.02
0.04
0.06
Type-1-Error
0.0
0.01
0.02
0.03
0.04
0.05
Contamination rate
0.000
0.002
0.004
0.006
0.008
0.010
Power
0.0
0.01
0.02
0.03
0.04
0.05
Contamination rate
0
25
50
75
100
125
Trimmed
Standard
Oracle (infeasible)
Naive-Trim (invalid)
Small-Clean
Label-Trim
Target type-I level
Labeling budget
# outliers
(c) KDDCup99
Figure 14: Comparison of conformal outlier detection methods on real datasets as a function of the contamination rate r. All
methods utilize a One-Class SVM model. Other details are as in Figure 3 in the main manuscript.
25


**[Table p25.1]**
| Type-1-Error 0.010.020.030.04 0.0 0.01 0.02 0.03 0.04 0.05 Contamination rate | 0.10.20.30.40.50.6 Power 0.0 0.01 0.02 0.03 0.04 0.05 Contamination rate | 75100125 Trimmed 50 25 0 0.0 0.01 0.02 0.03 0.04 0.05 Contamination rate |
| --- | --- | --- |


**[Table p25.2]**
| 0.06 Type-1-Error 0.04 0.02 0.0 0.01 0.02 0.03 0.04 0.05 Contamination rate | 0.0250.0500.0750.1000.125 Power 0.0 0.01 0.02 0.03 0.04 0.05 Contamination rate | 75100125 Trimmed 50 25 0 0.0 0.01 0.02 0.03 0.04 0.05 Contamination rate |
| --- | --- | --- |


**[Table p25.3]**
| Type-1-Error 0.06 0.04 0.02 0.00 0.0 0.01 0.02 0.03 0.04 0.05 Contamination rate | 0.0000.0020.0040.0060.0080.010 Power 0.0 0.01 0.02 0.03 0.04 0.05 Contamination rate | 75100125 Trimmed 50 25 0 0.0 0.01 0.02 0.03 0.04 0.05 Contamination rate | Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim Target type-I level Labeling budget # outliers |
| --- | --- | --- | --- |
|  |  |  |  |

[CAPTION] Figure 14: Comparison of conformal outlier detection methods on real datasets as a function of the contamination rate r. All


<!-- page 26 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
Results with higher contamination rate
Figure 15 shows the performance of conformal outlier detection methods
as a function of the contamination rate r, evaluated at higher levels than those considered in the main manuscript. The
performance trend aligns with that observed at lower contamination rates, across all datasets. Both the Standard
and Small-Clean methods control the type-I error at the nominal level but remain overly conservative. In contrast,
Naive-Trim fails to control the type-I error. The proposed Label-Trim method achieves improved power while
controlling the type-I error at level α.
0.05
0.07
0.09
0.11
0.13
0.15
Contamination rate
0.00
0.02
0.04
0.06
0.08
0.10
Type-1-Error
0.05
0.07
0.09
0.11
0.13
0.15
Contamination rate
0.0
0.2
0.4
0.6
Power
0.05
0.07
0.09
0.11
0.13
0.15
Contamination rate
100
200
300
Trimmed
(a) shuttle
0.05
0.07
0.09
0.11
0.13
0.15
Contamination rate
0.00
0.02
0.04
0.06
Type-1-Error
0.05
0.07
0.09
0.11
0.13
0.15
Contamination rate
0.0
0.2
0.4
0.6
0.8
Power
0.05
0.07
0.09
0.11
0.13
0.15
Contamination rate
50
75
100
125
150
Trimmed
(b) credit-card
0.05
0.07
0.09
0.11
0.13
0.15
Contamination rate
0.000
0.025
0.050
0.075
0.100
0.125
Type-1-Error
0.05
0.07
0.09
0.11
0.13
0.15
Contamination rate
0.0
0.1
0.2
0.3
0.4
Power
0.05
0.07
0.09
0.11
0.13
0.15
Contamination rate
100
200
300
Trimmed
Standard
Oracle (infeasible)
Naive-Trim (invalid)
Small-Clean
Label-Trim
Target type-I level
Labeling budget
# outliers
(c) KDDCup99
Figure 15: Comparison of conformal outlier detection methods on real tabular datasets as a function of the contamination
rate r. Other details are as in Figure 3 in the main manuscript.
Results with strategic outlier injection
We extend the experimental setting from the main manuscript by evaluating
different outlier injection strategies. Instead of injecting outliers at random, we selected outliers that more closely resemble
inliers—specifically, outliers whose nonconformity scores fall below a given score percentile. Figure 16 illustrates the
nonconformity scores for both outliers and inliers under different outlier injection strategies for the “shuttle” dataset,
highlighting that lower-percentile outliers increasingly resemble inliers. This sets the stage to evaluate the performance of
our proposed method under these challenging settings. Figure 17 presents the performance of all conformal methods on the
same dataset and outlier injection strategies, as a function of the target type-I error level. Following that figure, we see that
Label-Trim controls the type-I error while improving power. Similar trends hold on the credit-card and KDDCup99
datasets (Figures 18 and 19).
26


**[Table p26.1]**
| 0.000.020.040.060.080.10 Type-1-Error 0.05 0.07 0.09 0.11 0.13 0.15 Contamination rate | 0.6 Power 0.4 0.2 0.0 0.05 0.07 0.09 0.11 0.13 0.15 Contamination rate | 300 Trimmed 200 100 0.05 0.07 0.09 0.11 0.13 0.15 Contamination rate |
| --- | --- | --- |


**[Table p26.2]**
| 0.06 Type-1-Error 0.04 0.02 0.00 0.05 0.07 0.09 0.11 0.13 0.15 Contamination rate | 0.8 0.6 Power 0.4 0.2 0.0 0.05 0.07 0.09 0.11 0.13 0.15 Contamination rate | 75100125150 Trimmed 50 0.05 0.07 0.09 0.11 0.13 0.15 Contamination rate |
| --- | --- | --- |


**[Table p26.3]**
| 0.0000.0250.0500.0750.1000.125 Type-1-Error 0.05 0.07 0.09 0.11 0.13 0.15 Contamination rate | 0.4 Power 0.3 0.2 0.1 0.0 0.05 0.07 0.09 0.11 0.13 0.15 Contamination rate | 300 Trimmed 200 100 0.05 0.07 0.09 0.11 0.13 0.15 Contamination rate | Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim Target type-I level Labeling budget # outliers |
| --- | --- | --- | --- |
|  |  |  |  |

[CAPTION] Figure 15 shows the performance of conformal outlier detection methods

[CAPTION] Figure 15: Comparison of conformal outlier detection methods on real tabular datasets as a function of the contamination


<!-- page 27 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
0.1
0.0
0.1
0.2
Scores
0
250
500
750
1000
(a) Random outlier injection
0.1
0.0
0.1
0.2
Scores
0
250
500
750
1000
(b) 70th percentile
0.1
0.0
0.1
0.2
Scores
0
250
500
750
1000
(c) 50th percentile
Inliers
Outliers
Qcal
1
Qoutlier
1
Qinlier
1
Figure 16: Histogram of nonconformity scores for inliers and outliers in a contaminated calibration subset of the “shuttle”
data, with a contamination rate of 5% for different outlier injection strategies.
0.01
0.02
0.03
0.04
0.05
0.00
0.02
0.04
0.06
Type-1-Error
0.01
0.02
0.03
0.04
0.05
0.0
0.2
0.4
0.6
Power
(a) Random outlier injection
0.01
0.02
0.03
0.04
0.05
0.00
0.02
0.04
0.06
Type-1-Error
0.01
0.02
0.03
0.04
0.05
0.0
0.2
0.4
Power
(b) 70th percentile
0.01
0.02
0.03
0.04
0.05
0.00
0.02
0.04
0.06
Type-1-Error
0.01
0.02
0.03
0.04
0.05
0.0
0.1
0.2
0.3
0.4
Power
Standard
Oracle (infeasible)
Naive-Trim (invalid)
Small-Clean
Label-Trim
Target type-I level
(c) 50th percentile
Figure 17: Performance on a real dataset “shuttle” as a function of the target type-I error rate α for different outlier injection
strategies. Other details are as in Figure 5 in the main manuscript.
27


**[Table p27.1]**
| 1000 750 500 250 0 0.1 0.0 0.1 0.2 Scores | 1000 750 500 250 0 0.1 0.0 0.1 0.2 Scores |  |  |  |  |  |  | 1000 750 500 250 0 0.1 0.0 0.1 0.2 Scores |  |  |  |  |  |  | Inliers Outliers Qcal 1 Qoutlier 1 Qinlier 1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |


**[Table p27.2]**
| Type-1-Error 0.02 0.04 0.00 0.06 0.01 0.02 0.03 0.04 0.05 | Power 0.2 0.4 0.0 0.6 0.01 0.02 0.03 0.04 0.05 |  |
| --- | --- | --- |


**[Table p27.3]**
| Type-1-Error 0.02 0.04 0.00 0.06 0.01 0.02 0.03 0.04 0.05 | Power 0.2 0.4 0.0 0.01 0.02 0.03 0.04 0.05 |  |
| --- | --- | --- |


**[Table p27.4]**
| Type-1-Error 0.00 0.06 0.02 0.04 0.01 0.02 0.03 0.04 0.05 | Power 0.0 0.3 0.1 0.2 0.4 0.01 0.02 0.03 0.04 0.05 |  | Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim Target type-I level |
| --- | --- | --- | --- |

[CAPTION] Figure 16: Histogram of nonconformity scores for inliers and outliers in a contaminated calibration subset of the “shuttle”

[CAPTION] Figure 17: Performance on a real dataset “shuttle” as a function of the target type-I error rate α for different outlier injection


<!-- page 28 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
0.01
0.02
0.03
0.04
0.05
0.00
0.02
0.04
Type-1-Error
0.01
0.02
0.03
0.04
0.05
0.0
0.2
0.4
0.6
0.8
Power
(a) Random outlier injection
0.01
0.02
0.03
0.04
0.05
0.00
0.02
0.04
0.06
Type-1-Error
0.01
0.02
0.03
0.04
0.05
0.0
0.2
0.4
0.6
0.8
Power
(b) 70th percentile
0.01
0.02
0.03
0.04
0.05
0.00
0.02
0.04
0.06
Type-1-Error
0.01
0.02
0.03
0.04
0.05
0.0
0.2
0.4
0.6
Power
Standard
Oracle (infeasible)
Naive-Trim (invalid)
Small-Clean
Label-Trim
Target type-I level
(c) 50th percentile
Figure 18: Performance on a real dataset “credit-card” as a function of the target type-I error rate α for different outlier
injection strategies. Other details are as in Figure 5 in the main manuscript.
28


**[Table p28.1]**
| Type-1-Error 0.02 0.04 0.00 0.01 0.02 0.03 0.04 0.05 | Power 0.2 0.4 0.6 0.8 0.0 0.01 0.02 0.03 0.04 0.05 |  |
| --- | --- | --- |


**[Table p28.2]**
| Type-1-Error 0.02 0.04 0.00 0.06 0.01 0.02 0.03 0.04 0.05 | Power 0.0 0.2 0.4 0.6 0.8 0.01 0.02 0.03 0.04 0.05 |  |
| --- | --- | --- |


**[Table p28.3]**
| Type-1-Error 0.02 0.04 0.00 0.06 0.01 0.02 0.03 0.04 0.05 | Power 0.2 0.4 0.0 0.6 0.01 0.02 0.03 0.04 0.05 |  | Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim Target type-I level |
| --- | --- | --- | --- |

[CAPTION] Figure 18: Performance on a real dataset “credit-card” as a function of the target type-I error rate α for different outlier


<!-- page 29 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
0.01
0.02
0.03
0.04
0.05
0.00
0.02
0.04
0.06
Type-1-Error
0.01
0.02
0.03
0.04
0.05
0.0
0.2
0.4
0.6
Power
(a) Random outlier injection
0.01
0.02
0.03
0.04
0.05
0.00
0.02
0.04
0.06
Type-1-Error
0.01
0.02
0.03
0.04
0.05
0.0
0.2
0.4
0.6
Power
(b) 90th percentile
0.01
0.02
0.03
0.04
0.05
0.00
0.02
0.04
0.06
Type-1-Error
0.01
0.02
0.03
0.04
0.05
0.0
0.1
0.2
0.3
0.4
0.5
Power
(c) 80th percentile
0.01
0.02
0.03
0.04
0.05
0.00
0.02
0.04
0.06
Type-1-Error
0.01
0.02
0.03
0.04
0.05
0.0
0.1
0.2
Power
Standard
Oracle (infeasible)
Naive-Trim (invalid)
Small-Clean
Label-Trim
Target type-I level
(d) 70th percentile
Figure 19: Performance on a real dataset “KDDCup99” as a function of the target type-I error rate α for different outlier
injection strategies. Other details are as in Figure 5 in the main manuscript.
29


**[Table p29.1]**
| Type-1-Error 0.00 0.06 0.02 0.04 0.01 0.02 0.03 0.04 0.05 | Power 0.0 0.2 0.4 0.6 0.01 0.02 0.03 0.04 0.05 |  |
| --- | --- | --- |


**[Table p29.2]**
| Type-1-Error 0.02 0.04 0.00 0.06 0.01 0.02 0.03 0.04 0.05 | Power 0.2 0.4 0.0 0.6 0.01 0.02 0.03 0.04 0.05 |  |
| --- | --- | --- |


**[Table p29.3]**
| Type-1-Error 0.02 0.04 0.00 0.06 0.01 0.02 0.03 0.04 0.05 | Power 0.00.10.20.30.40.5 0.01 0.02 0.03 0.04 0.05 |  |
| --- | --- | --- |


**[Table p29.4]**
| Type-1-Error 0.00 0.02 0.04 0.06 0.01 0.02 0.03 0.04 0.05 | Power 0.0 0.1 0.2 0.01 0.02 0.03 0.04 0.05 |  | Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim Target type-I level |
| --- | --- | --- | --- |

[CAPTION] Figure 19: Performance on a real dataset “KDDCup99” as a function of the target type-I error rate α for different outlier


<!-- page 30 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
Results with test-time drifting outliers
We further investigate the robustness of the proposed method under test-time
distribution shift in the outlier population. Using the injection strategies described earlier, we simulate a shift on the “shuttle”
dataset where the outlier distribution gradually changes over time. Specifically, while the training and calibration sets
contain high-percentile outliers, the test set progressively includes more challenging, low-percentile outliers. As shown in
Figure 20, Label-Trim maintains type-I error control throughout the distribution shift, outperforming all baselines and
approaching oracle performance.
(30-100)
(20-90)
(10-80)
(0-70)
(0-60)
(0-50)
(0-40)
(0-30)
(0-80)
(0-90)
(0-100)
(10-100)
(20-100)
(40-100)
(50-100)
(60-100)
(70-100)
Outlier test distribution (percentile range)
0.005
0.010
0.015
0.020
0.025
Type-1-Error
(30-100)
(20-90)
(10-80)
(0-70)
(0-60)
(0-50)
(0-40)
(0-30)
(0-80)
(0-90)
(0-100)
(10-100)
(20-100)
(40-100)
(50-100)
(60-100)
(70-100)
Outlier test distribution (percentile range)
0.0
0.2
0.4
0.6
0.8
1.0
Power
Standard
Oracle (infeasible)
Naive-Trim (invalid)
Small-Clean
Label-Trim
Target type-I level
Figure 20: Performance on a real dataset “shuttle” as a function of drift in the outlier test distribution. Outliers in the training
and calibration sets are drawn from the 30th to 100th percentile of the outlier score distribution, while the test set varies
across different percentile ranges. Other details are as in Figure 3 in the main manuscript.
30


**[Table p30.1]**
| 0.0050.0100.0150.0200.025 Type-1-Error (30-100) (20-90) (10-80) (0-70) (0-60) (0-50) (0-40) (0-30) (0-80) (0-90) (0-100) (((((( 765421 000000 ------ 111111 000000 000000 )))))) Outlier test distribution (percentile range) |  |
| --- | --- |
| 1.0 0.8 Power 0.6 0.4 0.2 0.0 (30-100) (20-90) (10-80) (0-70) (0-60) (0-50) (0-40) (0-30) (0-80) (0-90) (0-100) (((((( 765421 000000 ------ 111111 000000 000000 )))))) Outlier test distribution (percentile range) | Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim Target type-I level |
|  |  |

[CAPTION] Figure 20: Performance on a real dataset “shuttle” as a function of drift in the outlier test distribution. Outliers in the training


<!-- page 31 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
B.2.2. VISUAL DATASETS
Table 3 summarizes the results across all six datasets for a target type-I error rate of α = 0.02, showing trends consistent
with those observed in Table 1 of the main manuscript.
Table 3: Comparison of conformal outlier detection methods on six visual datasets for varying contamination rate r and
target type-I error level α = 0.02. The empirical type-I error values are averaged across all datasets. The empirical power is
presented relative to the Standard method (higher is better), and averaged across all datasets. Results are averaged across
100 random splits of the data, with standard errors presented in parentheses.
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0204)
0.017 (± 0.0004)
1.0 (± 0.0238)
0.012 (± 0.0004)
1.0 (± 0.0272)
0.009 (± 0.0003)
Oracle (infeasible)
1.096 (± 0.0214)
0.021 (± 0.0005)
1.33 (± 0.025)
0.02 (± 0.0005)
1.674 (± 0.0311)
0.02 (± 0.0006)
Naive-Trim (invalid)
1.249 (± 0.0219)
0.026 (± 0.0005)
1.777 (± 0.0254)
0.035 (± 0.0006)
2.452 (± 0.0324)
0.044 (± 0.0007)
Small-Clean
0.819 (± 0.0592)
0.018 (± 0.002)
0.613 (± 0.0753)
0.011 (± 0.0018)
0.406 (± 0.0769)
0.006 (± 0.0013)
Label-Trim
1.079 (± 0.0212)
0.02 (± 0.0005)
1.23 (± 0.0247)
0.017 (± 0.0004)
1.381 (± 0.0298)
0.014 (± 0.0005)
(a) Target type-I error rate α = 0.02
We also report performance averaged across the six datasets for two additional models: ReAct (Sun et al., 2021) with a
VGG-19 backbone and SCALE (Xu et al., 2024) with a ResNet-18 backbone, shown in Tables 4 and 5, respectively. These
results align with our main findings and further demonstrate the effectiveness of the proposed Label-Trim method.
Following this, we provide detailed results for each dataset, reporting type-I error rates and the power of each method, with
the power reported relative to the Standard method (normalized to 1). The actual power of the Standard method is
included in the last row of each table. Specifically, Tables 6 to 11 present the detailed per-dataset results corresponding
to Tables 1 and 3. Additionally, detailed per-dataset results for the additional models are provided in Tables 12 to 17 and
Tables 18 to 23, corresponding to Table 4 and Table 5, respectively.
31


**[Table p31.1]**
| 1% |  | 3% |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Power Type-I Error |  | Power Type-I Error |  |  |  |
| 1.0 (± 0.0204) | 0.017 (± 0.0004) | 1.0 (± 0.0238) | 0.012 (± 0.0004) | 1.0 (± 0.0272) | 0.009 (± 0.0003) |
| 1.096 (± 0.0214) | 0.021 (± 0.0005) | 1.33 (± 0.025) | 0.02 (± 0.0005) | 1.674 (± 0.0311) | 0.02 (± 0.0006) |
| 1.249 (± 0.0219) | 0.026 (± 0.0005) | 1.777 (± 0.0254) | 0.035 (± 0.0006) | 2.452 (± 0.0324) | 0.044 (± 0.0007) |
| 0.819 (± 0.0592) | 0.018 (± 0.002) | 0.613 (± 0.0753) | 0.011 (± 0.0018) | 0.406 (± 0.0769) | 0.006 (± 0.0013) |
| 1.079 (± 0.0212) | 0.02 (± 0.0005) | 1.23 (± 0.0247) | 0.017 (± 0.0004) | 1.381 (± 0.0298) | 0.014 (± 0.0005) |

[CAPTION] Table 3 summarizes the results across all six datasets for a target type-I error rate of α = 0.02, showing trends consistent

[CAPTION] Table 3: Comparison of conformal outlier detection methods on six visual datasets for varying contamination rate r and


<!-- page 32 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
Table 4: Comparison of conformal outlier detection methods on six visual datasets for varying contamination rate r and
target type-I error level α. All methods utilize the ReAct (Sun et al., 2021) method with a pretrained VGG-19. Other details
are as in Table 1 in the main manuscript.
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0317)
0.009 (± 0.0004)
1.0 (± 0.0377)
0.006 (± 0.0003)
1.0 (± 0.0422)
0.004 (± 0.0002)
Oracle (infeasible)
1.095 (± 0.033)
0.01 (± 0.0004)
1.368 (± 0.0418)
0.01 (± 0.0004)
1.633 (± 0.0488)
0.009 (± 0.0003)
Naive-Trim (invalid)
1.464 (± 0.0373)
0.018 (± 0.0005)
2.5 (± 0.0531)
0.031 (± 0.0006)
3.447 (± 0.0592)
0.04 (± 0.0008)
Small-Clean
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
Label-Trim
1.095 (± 0.033)
0.01 (± 0.0004)
1.364 (± 0.0418)
0.01 (± 0.0004)
1.606 (± 0.0487)
0.009 (± 0.0003)
(a) Target type-I error rate α = 0.01
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0251)
0.018 (± 0.0005)
1.0 (± 0.027)
0.014 (± 0.0004)
1.0 (± 0.0291)
0.01 (± 0.0004)
Oracle (infeasible)
1.068 (± 0.0247)
0.021 (± 0.0005)
1.221 (± 0.0297)
0.02 (± 0.0005)
1.37 (± 0.0323)
0.019 (± 0.0006)
Naive-Trim (invalid)
1.252 (± 0.0256)
0.028 (± 0.0006)
1.759 (± 0.0326)
0.039 (± 0.0007)
2.212 (± 0.0345)
0.048 (± 0.0008)
Small-Clean
0.813 (± 0.0593)
0.017 (± 0.0019)
0.593 (± 0.0737)
0.011 (± 0.0018)
0.352 (± 0.0663)
0.005 (± 0.0013)
Label-Trim
1.056 (± 0.0244)
0.02 (± 0.0005)
1.177 (± 0.0292)
0.019 (± 0.0005)
1.253 (± 0.0312)
0.016 (± 0.0005)
(b) Target type-I error rate α = 0.02
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0202)
0.028 (± 0.0006)
1.0 (± 0.0238)
0.022 (± 0.0005)
1.0 (± 0.0247)
0.017 (± 0.0005)
Oracle (infeasible)
1.074 (± 0.02)
0.031 (± 0.0006)
1.194 (± 0.0256)
0.03 (± 0.0007)
1.318 (± 0.0271)
0.029 (± 0.0007)
Naive-Trim (invalid)
1.186 (± 0.0202)
0.037 (± 0.0007)
1.51 (± 0.0248)
0.048 (± 0.0008)
1.848 (± 0.0272)
0.056 (± 0.0009)
Small-Clean
0.709 (± 0.045)
0.019 (± 0.0018)
0.85 (± 0.0541)
0.021 (± 0.002)
0.932 (± 0.0568)
0.019 (± 0.002)
Label-Trim
1.048 (± 0.0202)
0.03 (± 0.0006)
1.118 (± 0.0248)
0.027 (± 0.0006)
1.167 (± 0.026)
0.023 (± 0.0006)
(c) Target type-I error rate α = 0.03
32


**[Table p32.1]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0317) | 0.009 (± 0.0004) | 1.0 (± 0.0377) | 0.006 (± 0.0003) | 1.0 (± 0.0422) | 0.004 (± 0.0002) |
|  | 1.095 (± 0.033) | 0.01 (± 0.0004) | 1.368 (± 0.0418) | 0.01 (± 0.0004) | 1.633 (± 0.0488) | 0.009 (± 0.0003) |
|  | 1.464 (± 0.0373) | 0.018 (± 0.0005) | 2.5 (± 0.0531) | 0.031 (± 0.0006) | 3.447 (± 0.0592) | 0.04 (± 0.0008) |
|  | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) |
|  | 1.095 (± 0.033) | 0.01 (± 0.0004) | 1.364 (± 0.0418) | 0.01 (± 0.0004) | 1.606 (± 0.0487) | 0.009 (± 0.0003) |


**[Table p32.2]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0251) | 0.018 (± 0.0005) | 1.0 (± 0.027) | 0.014 (± 0.0004) | 1.0 (± 0.0291) | 0.01 (± 0.0004) |
|  | 1.068 (± 0.0247) | 0.021 (± 0.0005) | 1.221 (± 0.0297) | 0.02 (± 0.0005) | 1.37 (± 0.0323) | 0.019 (± 0.0006) |
|  | 1.252 (± 0.0256) | 0.028 (± 0.0006) | 1.759 (± 0.0326) | 0.039 (± 0.0007) | 2.212 (± 0.0345) | 0.048 (± 0.0008) |
|  | 0.813 (± 0.0593) | 0.017 (± 0.0019) | 0.593 (± 0.0737) | 0.011 (± 0.0018) | 0.352 (± 0.0663) | 0.005 (± 0.0013) |
|  | 1.056 (± 0.0244) | 0.02 (± 0.0005) | 1.177 (± 0.0292) | 0.019 (± 0.0005) | 1.253 (± 0.0312) | 0.016 (± 0.0005) |


**[Table p32.3]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0202) | 0.028 (± 0.0006) | 1.0 (± 0.0238) | 0.022 (± 0.0005) | 1.0 (± 0.0247) | 0.017 (± 0.0005) |
|  | 1.074 (± 0.02) | 0.031 (± 0.0006) | 1.194 (± 0.0256) | 0.03 (± 0.0007) | 1.318 (± 0.0271) | 0.029 (± 0.0007) |
|  | 1.186 (± 0.0202) | 0.037 (± 0.0007) | 1.51 (± 0.0248) | 0.048 (± 0.0008) | 1.848 (± 0.0272) | 0.056 (± 0.0009) |
|  | 0.709 (± 0.045) | 0.019 (± 0.0018) | 0.85 (± 0.0541) | 0.021 (± 0.002) | 0.932 (± 0.0568) | 0.019 (± 0.002) |
|  | 1.048 (± 0.0202) | 0.03 (± 0.0006) | 1.118 (± 0.0248) | 0.027 (± 0.0006) | 1.167 (± 0.026) | 0.023 (± 0.0006) |

[CAPTION] Table 4: Comparison of conformal outlier detection methods on six visual datasets for varying contamination rate r and


<!-- page 33 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
Table 5: Comparison of conformal outlier detection methods on six visual datasets for varying contamination rate r and
target type-I error level α. All methods utilize the SCALE (Xu et al., 2024) method with a pretrained ResNet-18. Other
details are as in Table 1 in the main manuscript.
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0358)
0.009 (± 0.0003)
1.0 (± 0.0375)
0.006 (± 0.0003)
1.0 (± 0.0407)
0.004 (± 0.0002)
Oracle (infeasible)
1.144 (± 0.037)
0.01 (± 0.0003)
1.472 (± 0.0474)
0.01 (± 0.0003)
1.708 (± 0.0522)
0.01 (± 0.0004)
Naive-Trim (invalid)
1.736 (± 0.0414)
0.017 (± 0.0004)
2.93 (± 0.0553)
0.028 (± 0.0006)
4.144 (± 0.0614)
0.038 (± 0.0007)
Small-Clean
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
Label-Trim
1.144 (± 0.0371)
0.01 (± 0.0003)
1.456 (± 0.0469)
0.01 (± 0.0003)
1.627 (± 0.0501)
0.009 (± 0.0003)
(a) Target type-I error rate α = 0.01
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0236)
0.018 (± 0.0004)
1.0 (± 0.0267)
0.013 (± 0.0004)
1.0 (± 0.0288)
0.01 (± 0.0003)
Oracle (infeasible)
1.076 (± 0.0244)
0.021 (± 0.0005)
1.306 (± 0.0279)
0.02 (± 0.0005)
1.583 (± 0.0316)
0.02 (± 0.0005)
Naive-Trim (invalid)
1.247 (± 0.0253)
0.027 (± 0.0006)
1.86 (± 0.0294)
0.036 (± 0.0006)
2.532 (± 0.0335)
0.047 (± 0.0008)
Small-Clean
0.831 (± 0.0598)
0.018 (± 0.0019)
0.609 (± 0.0765)
0.011 (± 0.0017)
0.348 (± 0.0744)
0.005 (± 0.0014)
Label-Trim
1.062 (± 0.0243)
0.02 (± 0.0005)
1.226 (± 0.0284)
0.018 (± 0.0005)
1.368 (± 0.0315)
0.015 (± 0.0004)
(b) Target type-I error rate α = 0.02
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0201)
0.027 (± 0.0006)
1.0 (± 0.0212)
0.021 (± 0.0005)
1.0 (± 0.022)
0.016 (± 0.0004)
Oracle (infeasible)
1.082 (± 0.0204)
0.031 (± 0.0006)
1.233 (± 0.0227)
0.029 (± 0.0006)
1.398 (± 0.0237)
0.03 (± 0.0006)
Naive-Trim (invalid)
1.19 (± 0.0197)
0.036 (± 0.0006)
1.547 (± 0.0229)
0.044 (± 0.0007)
1.9 (± 0.0239)
0.055 (± 0.0009)
Small-Clean
0.712 (± 0.045)
0.019 (± 0.0019)
0.862 (± 0.055)
0.02 (± 0.002)
0.971 (± 0.0606)
0.021 (± 0.0021)
Label-Trim
1.05 (± 0.0204)
0.029 (± 0.0006)
1.116 (± 0.022)
0.025 (± 0.0006)
1.178 (± 0.0224)
0.022 (± 0.0005)
(c) Target type-I error rate α = 0.03
33


**[Table p33.1]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0358) | 0.009 (± 0.0003) | 1.0 (± 0.0375) | 0.006 (± 0.0003) | 1.0 (± 0.0407) | 0.004 (± 0.0002) |
|  | 1.144 (± 0.037) | 0.01 (± 0.0003) | 1.472 (± 0.0474) | 0.01 (± 0.0003) | 1.708 (± 0.0522) | 0.01 (± 0.0004) |
|  | 1.736 (± 0.0414) | 0.017 (± 0.0004) | 2.93 (± 0.0553) | 0.028 (± 0.0006) | 4.144 (± 0.0614) | 0.038 (± 0.0007) |
|  | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) |
|  | 1.144 (± 0.0371) | 0.01 (± 0.0003) | 1.456 (± 0.0469) | 0.01 (± 0.0003) | 1.627 (± 0.0501) | 0.009 (± 0.0003) |


**[Table p33.2]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0236) | 0.018 (± 0.0004) | 1.0 (± 0.0267) | 0.013 (± 0.0004) | 1.0 (± 0.0288) | 0.01 (± 0.0003) |
|  | 1.076 (± 0.0244) | 0.021 (± 0.0005) | 1.306 (± 0.0279) | 0.02 (± 0.0005) | 1.583 (± 0.0316) | 0.02 (± 0.0005) |
|  | 1.247 (± 0.0253) | 0.027 (± 0.0006) | 1.86 (± 0.0294) | 0.036 (± 0.0006) | 2.532 (± 0.0335) | 0.047 (± 0.0008) |
|  | 0.831 (± 0.0598) | 0.018 (± 0.0019) | 0.609 (± 0.0765) | 0.011 (± 0.0017) | 0.348 (± 0.0744) | 0.005 (± 0.0014) |
|  | 1.062 (± 0.0243) | 0.02 (± 0.0005) | 1.226 (± 0.0284) | 0.018 (± 0.0005) | 1.368 (± 0.0315) | 0.015 (± 0.0004) |


**[Table p33.3]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0201) | 0.027 (± 0.0006) | 1.0 (± 0.0212) | 0.021 (± 0.0005) | 1.0 (± 0.022) | 0.016 (± 0.0004) |
|  | 1.082 (± 0.0204) | 0.031 (± 0.0006) | 1.233 (± 0.0227) | 0.029 (± 0.0006) | 1.398 (± 0.0237) | 0.03 (± 0.0006) |
|  | 1.19 (± 0.0197) | 0.036 (± 0.0006) | 1.547 (± 0.0229) | 0.044 (± 0.0007) | 1.9 (± 0.0239) | 0.055 (± 0.0009) |
|  | 0.712 (± 0.045) | 0.019 (± 0.0019) | 0.862 (± 0.055) | 0.02 (± 0.002) | 0.971 (± 0.0606) | 0.021 (± 0.0021) |
|  | 1.05 (± 0.0204) | 0.029 (± 0.0006) | 1.116 (± 0.022) | 0.025 (± 0.0006) | 1.178 (± 0.0224) | 0.022 (± 0.0005) |

[CAPTION] Table 5: Comparison of conformal outlier detection methods on six visual datasets for varying contamination rate r and


<!-- page 34 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
Table 6: Comparison of conformal outlier detection methods on Texture dataset (outliers) and CIFAR-10 dataset (inliers) for
varying contamination rate r and target type-I error level α. All methods utilize the ReAct (Sun et al., 2021) method with a
pretrained ResNet-18. The empirical power is presented relative to the Standard method (higher is better). Results are
averaged across 100 random splits of the data, with standard errors presented in parentheses.
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0295)
0.008 (± 0.0003)
1.0 (± 0.0369)
0.006 (± 0.0003)
1.0 (± 0.0405)
0.004 (± 0.0002)
Oracle (infeasible)
1.173 (± 0.0295)
0.01 (± 0.0003)
1.455 (± 0.0432)
0.01 (± 0.0003)
1.824 (± 0.051)
0.009 (± 0.0004)
Naive-Trim (invalid)
1.722 (± 0.035)
0.017 (± 0.0004)
2.668 (± 0.0428)
0.028 (± 0.0006)
4.008 (± 0.0599)
0.037 (± 0.0007)
Small-Clean
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
Label-Trim
1.173 (± 0.0295)
0.01 (± 0.0003)
1.448 (± 0.0428)
0.01 (± 0.0003)
1.73 (± 0.0477)
0.009 (± 0.0003)
Standard Power
0.194 (± 0.0057)
0.159 (± 0.0059)
0.123 (± 0.005)
(a) Target type-I error rate α = 0.01
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0202)
0.017 (± 0.0005)
1.0 (± 0.0239)
0.012 (± 0.0004)
1.0 (± 0.0264)
0.009 (± 0.0003)
Oracle (infeasible)
1.109 (± 0.0199)
0.021 (± 0.0005)
1.315 (± 0.0259)
0.02 (± 0.0005)
1.623 (± 0.0346)
0.02 (± 0.0006)
Naive-Trim (invalid)
1.24 (± 0.0193)
0.026 (± 0.0006)
1.781 (± 0.026)
0.035 (± 0.0007)
2.431 (± 0.0334)
0.045 (± 0.0007)
Small-Clean
0.819 (± 0.0599)
0.018 (± 0.002)
0.702 (± 0.0824)
0.014 (± 0.0025)
0.478 (± 0.0842)
0.007 (± 0.0016)
Label-Trim
1.089 (± 0.0199)
0.02 (± 0.0005)
1.219 (± 0.0253)
0.017 (± 0.0004)
1.353 (± 0.0303)
0.014 (± 0.0005)
Standard Power
0.337 (± 0.0068)
0.272 (± 0.0065)
0.224 (± 0.0059)
(b) Target type-I error rate α = 0.02
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0155)
0.027 (± 0.0006)
1.0 (± 0.0197)
0.02 (± 0.0005)
1.0 (± 0.0215)
0.015 (± 0.0005)
Oracle (infeasible)
1.068 (± 0.0152)
0.03 (± 0.0006)
1.224 (± 0.0191)
0.03 (± 0.0006)
1.427 (± 0.0249)
0.03 (± 0.0007)
Naive-Trim (invalid)
1.15 (± 0.0152)
0.036 (± 0.0006)
1.514 (± 0.0178)
0.043 (± 0.0007)
1.919 (± 0.0229)
0.052 (± 0.0008)
Small-Clean
0.743 (± 0.0441)
0.02 (± 0.002)
0.883 (± 0.0549)
0.021 (± 0.0025)
1.062 (± 0.0665)
0.024 (± 0.0027)
Label-Trim
1.046 (± 0.0153)
0.029 (± 0.0006)
1.13 (± 0.019)
0.025 (± 0.0005)
1.215 (± 0.0245)
0.021 (± 0.0005)
Standard Power
0.421 (± 0.0065)
0.357 (± 0.007)
0.309 (± 0.0067)
(c) Target type-I error rate α = 0.03
34


**[Table p34.1]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0295) | 0.008 (± 0.0003) | 1.0 (± 0.0369) | 0.006 (± 0.0003) | 1.0 (± 0.0405) | 0.004 (± 0.0002) |
|  | 1.173 (± 0.0295) | 0.01 (± 0.0003) | 1.455 (± 0.0432) | 0.01 (± 0.0003) | 1.824 (± 0.051) | 0.009 (± 0.0004) |
|  | 1.722 (± 0.035) | 0.017 (± 0.0004) | 2.668 (± 0.0428) | 0.028 (± 0.0006) | 4.008 (± 0.0599) | 0.037 (± 0.0007) |
|  | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) |
|  | 1.173 (± 0.0295) | 0.01 (± 0.0003) | 1.448 (± 0.0428) | 0.01 (± 0.0003) | 1.73 (± 0.0477) | 0.009 (± 0.0003) |
| Standard Power | 0.194 (± 0.0057) |  | 0.159 (± 0.0059) |  | 0.123 (± 0.005) |  |


**[Table p34.2]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0202) | 0.017 (± 0.0005) | 1.0 (± 0.0239) | 0.012 (± 0.0004) | 1.0 (± 0.0264) | 0.009 (± 0.0003) |
|  | 1.109 (± 0.0199) | 0.021 (± 0.0005) | 1.315 (± 0.0259) | 0.02 (± 0.0005) | 1.623 (± 0.0346) | 0.02 (± 0.0006) |
|  | 1.24 (± 0.0193) | 0.026 (± 0.0006) | 1.781 (± 0.026) | 0.035 (± 0.0007) | 2.431 (± 0.0334) | 0.045 (± 0.0007) |
|  | 0.819 (± 0.0599) | 0.018 (± 0.002) | 0.702 (± 0.0824) | 0.014 (± 0.0025) | 0.478 (± 0.0842) | 0.007 (± 0.0016) |
|  | 1.089 (± 0.0199) | 0.02 (± 0.0005) | 1.219 (± 0.0253) | 0.017 (± 0.0004) | 1.353 (± 0.0303) | 0.014 (± 0.0005) |
| Standard Power | 0.337 (± 0.0068) |  | 0.272 (± 0.0065) |  | 0.224 (± 0.0059) |  |


**[Table p34.3]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0155) | 0.027 (± 0.0006) | 1.0 (± 0.0197) | 0.02 (± 0.0005) | 1.0 (± 0.0215) | 0.015 (± 0.0005) |
|  | 1.068 (± 0.0152) | 0.03 (± 0.0006) | 1.224 (± 0.0191) | 0.03 (± 0.0006) | 1.427 (± 0.0249) | 0.03 (± 0.0007) |
|  | 1.15 (± 0.0152) | 0.036 (± 0.0006) | 1.514 (± 0.0178) | 0.043 (± 0.0007) | 1.919 (± 0.0229) | 0.052 (± 0.0008) |
|  | 0.743 (± 0.0441) | 0.02 (± 0.002) | 0.883 (± 0.0549) | 0.021 (± 0.0025) | 1.062 (± 0.0665) | 0.024 (± 0.0027) |
|  | 1.046 (± 0.0153) | 0.029 (± 0.0006) | 1.13 (± 0.019) | 0.025 (± 0.0005) | 1.215 (± 0.0245) | 0.021 (± 0.0005) |
| Standard Power | 0.421 (± 0.0065) |  | 0.357 (± 0.007) |  | 0.309 (± 0.0067) |  |

[CAPTION] Table 6: Comparison of conformal outlier detection methods on Texture dataset (outliers) and CIFAR-10 dataset (inliers) for


<!-- page 35 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
Table 7: Comparison of conformal outlier detection methods on SVHN dataset (outliers) and CIFAR-10 dataset (inliers) for
varying contamination rate r and target type-I error level α. All methods utilize the ReAct (Sun et al., 2021) method with a
pretrained ResNet-18. The empirical power is presented relative to the Standard method (higher is better). Results are
averaged across 100 random splits of the data, with standard errors presented in parentheses.
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0255)
0.008 (± 0.0003)
1.0 (± 0.0316)
0.004 (± 0.0002)
1.0 (± 0.0388)
0.002 (± 0.0002)
Oracle (infeasible)
1.192 (± 0.0262)
0.01 (± 0.0003)
1.654 (± 0.0368)
0.01 (± 0.0003)
2.208 (± 0.052)
0.009 (± 0.0004)
Naive-Trim (invalid)
1.573 (± 0.0271)
0.016 (± 0.0004)
2.689 (± 0.0353)
0.025 (± 0.0006)
4.073 (± 0.0533)
0.033 (± 0.0007)
Small-Clean
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
Label-Trim
1.192 (± 0.0262)
0.01 (± 0.0003)
1.601 (± 0.0367)
0.009 (± 0.0003)
1.895 (± 0.0448)
0.007 (± 0.0003)
Standard Power
0.271 (± 0.0069)
0.191 (± 0.006)
0.137 (± 0.0053)
(a) Target type-I error rate α = 0.01
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0168)
0.016 (± 0.0004)
1.0 (± 0.0218)
0.01 (± 0.0003)
1.0 (± 0.0245)
0.007 (± 0.0003)
Oracle (infeasible)
1.096 (± 0.0173)
0.021 (± 0.0005)
1.404 (± 0.0211)
0.02 (± 0.0005)
1.828 (± 0.0304)
0.02 (± 0.0006)
Naive-Trim (invalid)
1.2 (± 0.0175)
0.026 (± 0.0005)
1.721 (± 0.0216)
0.032 (± 0.0006)
2.42 (± 0.0281)
0.041 (± 0.0007)
Small-Clean
0.833 (± 0.053)
0.02 (± 0.0026)
0.598 (± 0.0754)
0.01 (± 0.0016)
0.509 (± 0.0929)
0.008 (± 0.0019)
Label-Trim
1.08 (± 0.0176)
0.02 (± 0.0005)
1.277 (± 0.0228)
0.016 (± 0.0004)
1.469 (± 0.0288)
0.013 (± 0.0004)
Standard Power
0.429 (± 0.0072)
0.332 (± 0.0072)
0.25 (± 0.0061)
(b) Target type-I error rate α = 0.02
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0145)
0.026 (± 0.0005)
1.0 (± 0.0169)
0.017 (± 0.0004)
1.0 (± 0.0204)
0.012 (± 0.0004)
Oracle (infeasible)
1.063 (± 0.014)
0.03 (± 0.0006)
1.246 (± 0.0158)
0.029 (± 0.0006)
1.513 (± 0.0212)
0.03 (± 0.0007)
Naive-Trim (invalid)
1.115 (± 0.0137)
0.035 (± 0.0006)
1.395 (± 0.0148)
0.041 (± 0.0007)
1.825 (± 0.0199)
0.049 (± 0.0008)
Small-Clean
0.73 (± 0.0419)
0.021 (± 0.0027)
0.948 (± 0.0447)
0.022 (± 0.0019)
1.092 (± 0.0625)
0.022 (± 0.0023)
Label-Trim
1.042 (± 0.0143)
0.029 (± 0.0006)
1.144 (± 0.0151)
0.024 (± 0.0005)
1.266 (± 0.0209)
0.019 (± 0.0006)
Standard Power
0.517 (± 0.0075)
0.44 (± 0.0074)
0.355 (± 0.0072)
(c) Target type-I error rate α = 0.03
35


**[Table p35.1]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0255) | 0.008 (± 0.0003) | 1.0 (± 0.0316) | 0.004 (± 0.0002) | 1.0 (± 0.0388) | 0.002 (± 0.0002) |
|  | 1.192 (± 0.0262) | 0.01 (± 0.0003) | 1.654 (± 0.0368) | 0.01 (± 0.0003) | 2.208 (± 0.052) | 0.009 (± 0.0004) |
|  | 1.573 (± 0.0271) | 0.016 (± 0.0004) | 2.689 (± 0.0353) | 0.025 (± 0.0006) | 4.073 (± 0.0533) | 0.033 (± 0.0007) |
|  | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) |
|  | 1.192 (± 0.0262) | 0.01 (± 0.0003) | 1.601 (± 0.0367) | 0.009 (± 0.0003) | 1.895 (± 0.0448) | 0.007 (± 0.0003) |
| Standard Power | 0.271 (± 0.0069) |  | 0.191 (± 0.006) |  | 0.137 (± 0.0053) |  |


**[Table p35.2]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0168) | 0.016 (± 0.0004) | 1.0 (± 0.0218) | 0.01 (± 0.0003) | 1.0 (± 0.0245) | 0.007 (± 0.0003) |
|  | 1.096 (± 0.0173) | 0.021 (± 0.0005) | 1.404 (± 0.0211) | 0.02 (± 0.0005) | 1.828 (± 0.0304) | 0.02 (± 0.0006) |
|  | 1.2 (± 0.0175) | 0.026 (± 0.0005) | 1.721 (± 0.0216) | 0.032 (± 0.0006) | 2.42 (± 0.0281) | 0.041 (± 0.0007) |
|  | 0.833 (± 0.053) | 0.02 (± 0.0026) | 0.598 (± 0.0754) | 0.01 (± 0.0016) | 0.509 (± 0.0929) | 0.008 (± 0.0019) |
|  | 1.08 (± 0.0176) | 0.02 (± 0.0005) | 1.277 (± 0.0228) | 0.016 (± 0.0004) | 1.469 (± 0.0288) | 0.013 (± 0.0004) |
| Standard Power | 0.429 (± 0.0072) |  | 0.332 (± 0.0072) |  | 0.25 (± 0.0061) |  |


**[Table p35.3]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0145) | 0.026 (± 0.0005) | 1.0 (± 0.0169) | 0.017 (± 0.0004) | 1.0 (± 0.0204) | 0.012 (± 0.0004) |
|  | 1.063 (± 0.014) | 0.03 (± 0.0006) | 1.246 (± 0.0158) | 0.029 (± 0.0006) | 1.513 (± 0.0212) | 0.03 (± 0.0007) |
|  | 1.115 (± 0.0137) | 0.035 (± 0.0006) | 1.395 (± 0.0148) | 0.041 (± 0.0007) | 1.825 (± 0.0199) | 0.049 (± 0.0008) |
|  | 0.73 (± 0.0419) | 0.021 (± 0.0027) | 0.948 (± 0.0447) | 0.022 (± 0.0019) | 1.092 (± 0.0625) | 0.022 (± 0.0023) |
|  | 1.042 (± 0.0143) | 0.029 (± 0.0006) | 1.144 (± 0.0151) | 0.024 (± 0.0005) | 1.266 (± 0.0209) | 0.019 (± 0.0006) |
| Standard Power | 0.517 (± 0.0075) |  | 0.44 (± 0.0074) |  | 0.355 (± 0.0072) |  |

[CAPTION] Table 7: Comparison of conformal outlier detection methods on SVHN dataset (outliers) and CIFAR-10 dataset (inliers) for


<!-- page 36 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
Table 8: Comparison of conformal outlier detection methods on Places365 dataset (outliers) and CIFAR-10 dataset (inliers)
for varying contamination rate r and target type-I error level α. All methods utilize the ReAct (Sun et al., 2021) method
with a pretrained ResNet-18. The empirical power is presented relative to the Standard method (higher is better). Results
are averaged across 100 random splits of the data, with standard errors presented in parentheses.
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.033)
0.009 (± 0.0003)
1.0 (± 0.0364)
0.006 (± 0.0003)
1.0 (± 0.0431)
0.004 (± 0.0002)
Oracle (infeasible)
1.139 (± 0.0339)
0.01 (± 0.0003)
1.47 (± 0.0423)
0.01 (± 0.0003)
1.839 (± 0.0558)
0.009 (± 0.0004)
Naive-Trim (invalid)
1.624 (± 0.0339)
0.017 (± 0.0004)
2.73 (± 0.0487)
0.028 (± 0.0006)
4.12 (± 0.069)
0.039 (± 0.0007)
Small-Clean
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
Label-Trim
1.139 (± 0.0339)
0.01 (± 0.0003)
1.466 (± 0.0421)
0.01 (± 0.0003)
1.707 (± 0.0531)
0.009 (± 0.0003)
Standard Power
0.193 (± 0.0064)
0.148 (± 0.0054)
0.115 (± 0.005)
(a) Target type-I error rate α = 0.01
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0209)
0.017 (± 0.0005)
1.0 (± 0.0241)
0.013 (± 0.0004)
1.0 (± 0.0298)
0.009 (± 0.0003)
Oracle (infeasible)
1.091 (± 0.0224)
0.021 (± 0.0005)
1.271 (± 0.0258)
0.02 (± 0.0005)
1.599 (± 0.0308)
0.02 (± 0.0006)
Naive-Trim (invalid)
1.246 (± 0.0225)
0.027 (± 0.0006)
1.753 (± 0.0277)
0.036 (± 0.0007)
2.462 (± 0.0376)
0.046 (± 0.0007)
Small-Clean
0.799 (± 0.0632)
0.018 (± 0.0021)
0.549 (± 0.0716)
0.01 (± 0.0016)
0.255 (± 0.0595)
0.003 (± 0.0009)
Label-Trim
1.071 (± 0.0219)
0.02 (± 0.0005)
1.187 (± 0.0247)
0.017 (± 0.0004)
1.361 (± 0.0306)
0.015 (± 0.0005)
Standard Power
0.316 (± 0.0066)
0.261 (± 0.0063)
0.213 (± 0.0064)
(b) Target type-I error rate α = 0.02
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.018)
0.027 (± 0.0006)
1.0 (± 0.0204)
0.02 (± 0.0005)
1.0 (± 0.021)
0.015 (± 0.0005)
Oracle (infeasible)
1.055 (± 0.0189)
0.03 (± 0.0006)
1.231 (± 0.0213)
0.029 (± 0.0006)
1.403 (± 0.025)
0.03 (± 0.0007)
Naive-Trim (invalid)
1.145 (± 0.0199)
0.036 (± 0.0006)
1.496 (± 0.0217)
0.044 (± 0.0007)
1.883 (± 0.0261)
0.054 (± 0.0008)
Small-Clean
0.746 (± 0.0469)
0.021 (± 0.002)
0.833 (± 0.0504)
0.018 (± 0.0018)
1.029 (± 0.0593)
0.022 (± 0.0022)
Label-Trim
1.038 (± 0.0188)
0.029 (± 0.0006)
1.149 (± 0.0211)
0.025 (± 0.0006)
1.186 (± 0.0228)
0.022 (± 0.0006)
Standard Power
0.395 (± 0.0071)
0.335 (± 0.0068)
0.302 (± 0.0063)
(c) Target type-I error rate α = 0.03
36


**[Table p36.1]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.033) | 0.009 (± 0.0003) | 1.0 (± 0.0364) | 0.006 (± 0.0003) | 1.0 (± 0.0431) | 0.004 (± 0.0002) |
|  | 1.139 (± 0.0339) | 0.01 (± 0.0003) | 1.47 (± 0.0423) | 0.01 (± 0.0003) | 1.839 (± 0.0558) | 0.009 (± 0.0004) |
|  | 1.624 (± 0.0339) | 0.017 (± 0.0004) | 2.73 (± 0.0487) | 0.028 (± 0.0006) | 4.12 (± 0.069) | 0.039 (± 0.0007) |
|  | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) |
|  | 1.139 (± 0.0339) | 0.01 (± 0.0003) | 1.466 (± 0.0421) | 0.01 (± 0.0003) | 1.707 (± 0.0531) | 0.009 (± 0.0003) |
| Standard Power | 0.193 (± 0.0064) |  | 0.148 (± 0.0054) |  | 0.115 (± 0.005) |  |


**[Table p36.2]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0209) | 0.017 (± 0.0005) | 1.0 (± 0.0241) | 0.013 (± 0.0004) | 1.0 (± 0.0298) | 0.009 (± 0.0003) |
|  | 1.091 (± 0.0224) | 0.021 (± 0.0005) | 1.271 (± 0.0258) | 0.02 (± 0.0005) | 1.599 (± 0.0308) | 0.02 (± 0.0006) |
|  | 1.246 (± 0.0225) | 0.027 (± 0.0006) | 1.753 (± 0.0277) | 0.036 (± 0.0007) | 2.462 (± 0.0376) | 0.046 (± 0.0007) |
|  | 0.799 (± 0.0632) | 0.018 (± 0.0021) | 0.549 (± 0.0716) | 0.01 (± 0.0016) | 0.255 (± 0.0595) | 0.003 (± 0.0009) |
|  | 1.071 (± 0.0219) | 0.02 (± 0.0005) | 1.187 (± 0.0247) | 0.017 (± 0.0004) | 1.361 (± 0.0306) | 0.015 (± 0.0005) |
| Standard Power | 0.316 (± 0.0066) |  | 0.261 (± 0.0063) |  | 0.213 (± 0.0064) |  |


**[Table p36.3]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.018) | 0.027 (± 0.0006) | 1.0 (± 0.0204) | 0.02 (± 0.0005) | 1.0 (± 0.021) | 0.015 (± 0.0005) |
|  | 1.055 (± 0.0189) | 0.03 (± 0.0006) | 1.231 (± 0.0213) | 0.029 (± 0.0006) | 1.403 (± 0.025) | 0.03 (± 0.0007) |
|  | 1.145 (± 0.0199) | 0.036 (± 0.0006) | 1.496 (± 0.0217) | 0.044 (± 0.0007) | 1.883 (± 0.0261) | 0.054 (± 0.0008) |
|  | 0.746 (± 0.0469) | 0.021 (± 0.002) | 0.833 (± 0.0504) | 0.018 (± 0.0018) | 1.029 (± 0.0593) | 0.022 (± 0.0022) |
|  | 1.038 (± 0.0188) | 0.029 (± 0.0006) | 1.149 (± 0.0211) | 0.025 (± 0.0006) | 1.186 (± 0.0228) | 0.022 (± 0.0006) |
| Standard Power | 0.395 (± 0.0071) |  | 0.335 (± 0.0068) |  | 0.302 (± 0.0063) |  |

[CAPTION] Table 8: Comparison of conformal outlier detection methods on Places365 dataset (outliers) and CIFAR-10 dataset (inliers)


<!-- page 37 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
Table 9: Comparison of conformal outlier detection methods on MNIST dataset (outliers) and CIFAR-10 dataset (inliers) for
varying contamination rate r and target type-I error level α. All methods utilize the ReAct (Sun et al., 2021) method with a
pretrained ResNet-18. The empirical power is presented relative to the Standard method (higher is better). Results are
averaged across 100 random splits of the data, with standard errors presented in parentheses.
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0248)
0.008 (± 0.0003)
1.0 (± 0.0283)
0.004 (± 0.0002)
1.0 (± 0.0363)
0.003 (± 0.0002)
Oracle (infeasible)
1.241 (± 0.0291)
0.01 (± 0.0003)
1.758 (± 0.0352)
0.01 (± 0.0003)
2.554 (± 0.0601)
0.009 (± 0.0004)
Naive-Trim (invalid)
1.629 (± 0.0292)
0.016 (± 0.0004)
2.722 (± 0.0363)
0.023 (± 0.0005)
4.718 (± 0.0552)
0.03 (± 0.0006)
Small-Clean
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
Label-Trim
1.241 (± 0.0291)
0.01 (± 0.0003)
1.636 (± 0.0331)
0.009 (± 0.0003)
2.113 (± 0.0555)
0.007 (± 0.0003)
Standard Power
0.282 (± 0.007)
0.208 (± 0.0059)
0.131 (± 0.0048)
(a) Target type-I error rate α = 0.01
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0173)
0.016 (± 0.0004)
1.0 (± 0.0205)
0.01 (± 0.0003)
1.0 (± 0.027)
0.007 (± 0.0003)
Oracle (infeasible)
1.117 (± 0.0179)
0.021 (± 0.0005)
1.452 (± 0.0209)
0.02 (± 0.0005)
1.971 (± 0.0297)
0.02 (± 0.0006)
Naive-Trim (invalid)
1.231 (± 0.0179)
0.025 (± 0.0005)
1.759 (± 0.0207)
0.031 (± 0.0006)
2.531 (± 0.0273)
0.038 (± 0.0007)
Small-Clean
0.844 (± 0.0514)
0.019 (± 0.0019)
0.617 (± 0.0738)
0.01 (± 0.0017)
0.438 (± 0.0817)
0.005 (± 0.0011)
Label-Trim
1.096 (± 0.0181)
0.02 (± 0.0005)
1.308 (± 0.0206)
0.015 (± 0.0004)
1.493 (± 0.0295)
0.012 (± 0.0004)
Standard Power
0.465 (± 0.008)
0.36 (± 0.0074)
0.264 (± 0.0071)
(b) Target type-I error rate α = 0.02
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0147)
0.025 (± 0.0005)
1.0 (± 0.0151)
0.016 (± 0.0004)
1.0 (± 0.0201)
0.011 (± 0.0004)
Oracle (infeasible)
1.072 (± 0.0149)
0.03 (± 0.0006)
1.295 (± 0.0155)
0.029 (± 0.0006)
1.617 (± 0.0186)
0.03 (± 0.0007)
Naive-Trim (invalid)
1.116 (± 0.0144)
0.034 (± 0.0006)
1.424 (± 0.0142)
0.039 (± 0.0007)
1.851 (± 0.0178)
0.046 (± 0.0007)
Small-Clean
0.699 (± 0.0394)
0.019 (± 0.0019)
0.884 (± 0.0464)
0.019 (± 0.0018)
1.164 (± 0.0576)
0.02 (± 0.002)
Label-Trim
1.049 (± 0.0148)
0.029 (± 0.0006)
1.158 (± 0.016)
0.023 (± 0.0005)
1.273 (± 0.0218)
0.017 (± 0.0005)
Standard Power
0.576 (± 0.0085)
0.483 (± 0.0073)
0.382 (± 0.0077)
(c) Target type-I error rate α = 0.03
37


**[Table p37.1]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0248) | 0.008 (± 0.0003) | 1.0 (± 0.0283) | 0.004 (± 0.0002) | 1.0 (± 0.0363) | 0.003 (± 0.0002) |
|  | 1.241 (± 0.0291) | 0.01 (± 0.0003) | 1.758 (± 0.0352) | 0.01 (± 0.0003) | 2.554 (± 0.0601) | 0.009 (± 0.0004) |
|  | 1.629 (± 0.0292) | 0.016 (± 0.0004) | 2.722 (± 0.0363) | 0.023 (± 0.0005) | 4.718 (± 0.0552) | 0.03 (± 0.0006) |
|  | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) |
|  | 1.241 (± 0.0291) | 0.01 (± 0.0003) | 1.636 (± 0.0331) | 0.009 (± 0.0003) | 2.113 (± 0.0555) | 0.007 (± 0.0003) |
| Standard Power | 0.282 (± 0.007) |  | 0.208 (± 0.0059) |  | 0.131 (± 0.0048) |  |


**[Table p37.2]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0173) | 0.016 (± 0.0004) | 1.0 (± 0.0205) | 0.01 (± 0.0003) | 1.0 (± 0.027) | 0.007 (± 0.0003) |
|  | 1.117 (± 0.0179) | 0.021 (± 0.0005) | 1.452 (± 0.0209) | 0.02 (± 0.0005) | 1.971 (± 0.0297) | 0.02 (± 0.0006) |
|  | 1.231 (± 0.0179) | 0.025 (± 0.0005) | 1.759 (± 0.0207) | 0.031 (± 0.0006) | 2.531 (± 0.0273) | 0.038 (± 0.0007) |
|  | 0.844 (± 0.0514) | 0.019 (± 0.0019) | 0.617 (± 0.0738) | 0.01 (± 0.0017) | 0.438 (± 0.0817) | 0.005 (± 0.0011) |
|  | 1.096 (± 0.0181) | 0.02 (± 0.0005) | 1.308 (± 0.0206) | 0.015 (± 0.0004) | 1.493 (± 0.0295) | 0.012 (± 0.0004) |
| Standard Power | 0.465 (± 0.008) |  | 0.36 (± 0.0074) |  | 0.264 (± 0.0071) |  |


**[Table p37.3]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0147) | 0.025 (± 0.0005) | 1.0 (± 0.0151) | 0.016 (± 0.0004) | 1.0 (± 0.0201) | 0.011 (± 0.0004) |
|  | 1.072 (± 0.0149) | 0.03 (± 0.0006) | 1.295 (± 0.0155) | 0.029 (± 0.0006) | 1.617 (± 0.0186) | 0.03 (± 0.0007) |
|  | 1.116 (± 0.0144) | 0.034 (± 0.0006) | 1.424 (± 0.0142) | 0.039 (± 0.0007) | 1.851 (± 0.0178) | 0.046 (± 0.0007) |
|  | 0.699 (± 0.0394) | 0.019 (± 0.0019) | 0.884 (± 0.0464) | 0.019 (± 0.0018) | 1.164 (± 0.0576) | 0.02 (± 0.002) |
|  | 1.049 (± 0.0148) | 0.029 (± 0.0006) | 1.158 (± 0.016) | 0.023 (± 0.0005) | 1.273 (± 0.0218) | 0.017 (± 0.0005) |
| Standard Power | 0.576 (± 0.0085) |  | 0.483 (± 0.0073) |  | 0.382 (± 0.0077) |  |

[CAPTION] Table 9: Comparison of conformal outlier detection methods on MNIST dataset (outliers) and CIFAR-10 dataset (inliers) for


<!-- page 38 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
Table 10: Comparison of conformal outlier detection methods on CIFAR-100 dataset (outliers) and CIFAR-10 dataset
(inliers) for varying contamination rate r and target type-I error level α. All methods utilize the ReAct (Sun et al., 2021)
method with a pretrained ResNet-18. The empirical power is presented relative to the Standard method (higher is better).
Results are averaged across 100 random splits of the data, with standard errors presented in parentheses.
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0393)
0.009 (± 0.0003)
1.0 (± 0.0391)
0.007 (± 0.0003)
1.0 (± 0.042)
0.005 (± 0.0003)
Oracle (infeasible)
1.116 (± 0.0417)
0.01 (± 0.0003)
1.463 (± 0.05)
0.01 (± 0.0003)
1.588 (± 0.0477)
0.009 (± 0.0004)
Naive-Trim (invalid)
1.733 (± 0.0397)
0.018 (± 0.0005)
3.06 (± 0.0585)
0.03 (± 0.0006)
4.054 (± 0.0631)
0.041 (± 0.0007)
Small-Clean
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
Label-Trim
1.116 (± 0.0417)
0.01 (± 0.0003)
1.463 (± 0.05)
0.01 (± 0.0003)
1.587 (± 0.0468)
0.009 (± 0.0004)
Standard Power
0.145 (± 0.0057)
0.118 (± 0.0046)
0.104 (± 0.0044)
(a) Target type-I error rate α = 0.01
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.023)
0.018 (± 0.0005)
1.0 (± 0.0265)
0.014 (± 0.0004)
1.0 (± 0.0264)
0.011 (± 0.0004)
Oracle (infeasible)
1.082 (± 0.0256)
0.021 (± 0.0005)
1.265 (± 0.0299)
0.02 (± 0.0005)
1.476 (± 0.0317)
0.02 (± 0.0006)
Naive-Trim (invalid)
1.294 (± 0.0266)
0.027 (± 0.0006)
1.831 (± 0.0288)
0.038 (± 0.0006)
2.464 (± 0.0372)
0.049 (± 0.0008)
Small-Clean
0.844 (± 0.0709)
0.019 (± 0.0022)
0.686 (± 0.0797)
0.012 (± 0.0017)
0.392 (± 0.078)
0.006 (± 0.0014)
Label-Trim
1.064 (± 0.0247)
0.02 (± 0.0005)
1.193 (± 0.0293)
0.018 (± 0.0004)
1.288 (± 0.0296)
0.016 (± 0.0005)
Standard Power
0.255 (± 0.0059)
0.225 (± 0.006)
0.19 (± 0.005)
(b) Target type-I error rate α = 0.02
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0204)
0.027 (± 0.0006)
1.0 (± 0.0214)
0.022 (± 0.0005)
1.0 (± 0.0228)
0.018 (± 0.0005)
Oracle (infeasible)
1.054 (± 0.0212)
0.031 (± 0.0006)
1.184 (± 0.0228)
0.03 (± 0.0006)
1.33 (± 0.0246)
0.03 (± 0.0007)
Naive-Trim (invalid)
1.184 (± 0.0213)
0.036 (± 0.0006)
1.523 (± 0.0222)
0.046 (± 0.0007)
1.94 (± 0.026)
0.056 (± 0.0009)
Small-Clean
0.709 (± 0.0527)
0.021 (± 0.0023)
0.842 (± 0.0545)
0.021 (± 0.0019)
0.947 (± 0.0628)
0.02 (± 0.002)
Label-Trim
1.029 (± 0.0212)
0.029 (± 0.0006)
1.114 (± 0.0222)
0.026 (± 0.0006)
1.16 (± 0.0229)
0.023 (± 0.0006)
Standard Power
0.332 (± 0.0068)
0.302 (± 0.0064)
0.263 (± 0.006)
(c) Target type-I error rate α = 0.03
38


**[Table p38.1]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0393) | 0.009 (± 0.0003) | 1.0 (± 0.0391) | 0.007 (± 0.0003) | 1.0 (± 0.042) | 0.005 (± 0.0003) |
|  | 1.116 (± 0.0417) | 0.01 (± 0.0003) | 1.463 (± 0.05) | 0.01 (± 0.0003) | 1.588 (± 0.0477) | 0.009 (± 0.0004) |
|  | 1.733 (± 0.0397) | 0.018 (± 0.0005) | 3.06 (± 0.0585) | 0.03 (± 0.0006) | 4.054 (± 0.0631) | 0.041 (± 0.0007) |
|  | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) |
|  | 1.116 (± 0.0417) | 0.01 (± 0.0003) | 1.463 (± 0.05) | 0.01 (± 0.0003) | 1.587 (± 0.0468) | 0.009 (± 0.0004) |
| Standard Power | 0.145 (± 0.0057) |  | 0.118 (± 0.0046) |  | 0.104 (± 0.0044) |  |


**[Table p38.2]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.023) | 0.018 (± 0.0005) | 1.0 (± 0.0265) | 0.014 (± 0.0004) | 1.0 (± 0.0264) | 0.011 (± 0.0004) |
|  | 1.082 (± 0.0256) | 0.021 (± 0.0005) | 1.265 (± 0.0299) | 0.02 (± 0.0005) | 1.476 (± 0.0317) | 0.02 (± 0.0006) |
|  | 1.294 (± 0.0266) | 0.027 (± 0.0006) | 1.831 (± 0.0288) | 0.038 (± 0.0006) | 2.464 (± 0.0372) | 0.049 (± 0.0008) |
|  | 0.844 (± 0.0709) | 0.019 (± 0.0022) | 0.686 (± 0.0797) | 0.012 (± 0.0017) | 0.392 (± 0.078) | 0.006 (± 0.0014) |
|  | 1.064 (± 0.0247) | 0.02 (± 0.0005) | 1.193 (± 0.0293) | 0.018 (± 0.0004) | 1.288 (± 0.0296) | 0.016 (± 0.0005) |
| Standard Power | 0.255 (± 0.0059) |  | 0.225 (± 0.006) |  | 0.19 (± 0.005) |  |


**[Table p38.3]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0204) | 0.027 (± 0.0006) | 1.0 (± 0.0214) | 0.022 (± 0.0005) | 1.0 (± 0.0228) | 0.018 (± 0.0005) |
|  | 1.054 (± 0.0212) | 0.031 (± 0.0006) | 1.184 (± 0.0228) | 0.03 (± 0.0006) | 1.33 (± 0.0246) | 0.03 (± 0.0007) |
|  | 1.184 (± 0.0213) | 0.036 (± 0.0006) | 1.523 (± 0.0222) | 0.046 (± 0.0007) | 1.94 (± 0.026) | 0.056 (± 0.0009) |
|  | 0.709 (± 0.0527) | 0.021 (± 0.0023) | 0.842 (± 0.0545) | 0.021 (± 0.0019) | 0.947 (± 0.0628) | 0.02 (± 0.002) |
|  | 1.029 (± 0.0212) | 0.029 (± 0.0006) | 1.114 (± 0.0222) | 0.026 (± 0.0006) | 1.16 (± 0.0229) | 0.023 (± 0.0006) |
| Standard Power | 0.332 (± 0.0068) |  | 0.302 (± 0.0064) |  | 0.263 (± 0.006) |  |

[CAPTION] Table 10: Comparison of conformal outlier detection methods on CIFAR-100 dataset (outliers) and CIFAR-10 dataset


<!-- page 39 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
Table 11: Comparison of conformal outlier detection methods on TinyImageNet dataset (outliers) and CIFAR-10 dataset
(inliers) for varying contamination rate r and target type-I error level α. All methods utilize the ReAct (Sun et al., 2021)
method with a pretrained ResNet-18. The empirical power is presented relative to the Standard method (higher is better).
Results are averaged across 100 random splits of the data, with standard errors presented in parentheses.
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0381)
0.009 (± 0.0003)
1.0 (± 0.0398)
0.006 (± 0.0003)
1.0 (± 0.0439)
0.004 (± 0.0002)
Oracle (infeasible)
1.137 (± 0.0409)
0.01 (± 0.0003)
1.492 (± 0.0474)
0.01 (± 0.0003)
1.754 (± 0.0518)
0.009 (± 0.0004)
Naive-Trim (invalid)
1.675 (± 0.0401)
0.018 (± 0.0004)
2.872 (± 0.0485)
0.028 (± 0.0006)
3.986 (± 0.0568)
0.039 (± 0.0007)
Small-Clean
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
Label-Trim
1.137 (± 0.0409)
0.01 (± 0.0003)
1.488 (± 0.0472)
0.01 (± 0.0003)
1.681 (± 0.0511)
0.009 (± 0.0003)
Standard Power
0.168 (± 0.0064)
0.133 (± 0.0053)
0.115 (± 0.005)
(a) Target type-I error rate α = 0.01
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0239)
0.018 (± 0.0004)
1.0 (± 0.0262)
0.013 (± 0.0004)
1.0 (± 0.0293)
0.01 (± 0.0004)
Oracle (infeasible)
1.084 (± 0.0253)
0.021 (± 0.0005)
1.275 (± 0.0263)
0.02 (± 0.0005)
1.544 (± 0.0293)
0.02 (± 0.0006)
Naive-Trim (invalid)
1.284 (± 0.0274)
0.027 (± 0.0006)
1.819 (± 0.0277)
0.036 (± 0.0006)
2.406 (± 0.0308)
0.046 (± 0.0008)
Small-Clean
0.777 (± 0.0566)
0.015 (± 0.0015)
0.525 (± 0.0687)
0.009 (± 0.0015)
0.366 (± 0.0653)
0.005 (± 0.0011)
Label-Trim
1.074 (± 0.0249)
0.02 (± 0.0005)
1.193 (± 0.0254)
0.017 (± 0.0004)
1.322 (± 0.03)
0.015 (± 0.0005)
Standard Power
0.285 (± 0.0068)
0.243 (± 0.0064)
0.209 (± 0.0061)
(b) Target type-I error rate α = 0.02
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0213)
0.027 (± 0.0006)
1.0 (± 0.0202)
0.02 (± 0.0005)
1.0 (± 0.0216)
0.016 (± 0.0005)
Oracle (infeasible)
1.062 (± 0.0216)
0.03 (± 0.0006)
1.23 (± 0.0205)
0.03 (± 0.0006)
1.397 (± 0.0236)
0.03 (± 0.0007)
Naive-Trim (invalid)
1.168 (± 0.0203)
0.036 (± 0.0006)
1.569 (± 0.0207)
0.045 (± 0.0007)
1.877 (± 0.0217)
0.054 (± 0.0009)
Small-Clean
0.656 (± 0.0438)
0.017 (± 0.0017)
0.825 (± 0.0496)
0.019 (± 0.0019)
0.904 (± 0.0589)
0.02 (± 0.0024)
Label-Trim
1.04 (± 0.0219)
0.029 (± 0.0006)
1.139 (± 0.0208)
0.026 (± 0.0006)
1.19 (± 0.0225)
0.022 (± 0.0006)
Standard Power
0.368 (± 0.0078)
0.318 (± 0.0064)
0.29 (± 0.0063)
(c) Target type-I error rate α = 0.03
39


**[Table p39.1]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0381) | 0.009 (± 0.0003) | 1.0 (± 0.0398) | 0.006 (± 0.0003) | 1.0 (± 0.0439) | 0.004 (± 0.0002) |
|  | 1.137 (± 0.0409) | 0.01 (± 0.0003) | 1.492 (± 0.0474) | 0.01 (± 0.0003) | 1.754 (± 0.0518) | 0.009 (± 0.0004) |
|  | 1.675 (± 0.0401) | 0.018 (± 0.0004) | 2.872 (± 0.0485) | 0.028 (± 0.0006) | 3.986 (± 0.0568) | 0.039 (± 0.0007) |
|  | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) |
|  | 1.137 (± 0.0409) | 0.01 (± 0.0003) | 1.488 (± 0.0472) | 0.01 (± 0.0003) | 1.681 (± 0.0511) | 0.009 (± 0.0003) |
| Standard Power | 0.168 (± 0.0064) |  | 0.133 (± 0.0053) |  | 0.115 (± 0.005) |  |


**[Table p39.2]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0239) | 0.018 (± 0.0004) | 1.0 (± 0.0262) | 0.013 (± 0.0004) | 1.0 (± 0.0293) | 0.01 (± 0.0004) |
|  | 1.084 (± 0.0253) | 0.021 (± 0.0005) | 1.275 (± 0.0263) | 0.02 (± 0.0005) | 1.544 (± 0.0293) | 0.02 (± 0.0006) |
|  | 1.284 (± 0.0274) | 0.027 (± 0.0006) | 1.819 (± 0.0277) | 0.036 (± 0.0006) | 2.406 (± 0.0308) | 0.046 (± 0.0008) |
|  | 0.777 (± 0.0566) | 0.015 (± 0.0015) | 0.525 (± 0.0687) | 0.009 (± 0.0015) | 0.366 (± 0.0653) | 0.005 (± 0.0011) |
|  | 1.074 (± 0.0249) | 0.02 (± 0.0005) | 1.193 (± 0.0254) | 0.017 (± 0.0004) | 1.322 (± 0.03) | 0.015 (± 0.0005) |
| Standard Power | 0.285 (± 0.0068) |  | 0.243 (± 0.0064) |  | 0.209 (± 0.0061) |  |


**[Table p39.3]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0213) | 0.027 (± 0.0006) | 1.0 (± 0.0202) | 0.02 (± 0.0005) | 1.0 (± 0.0216) | 0.016 (± 0.0005) |
|  | 1.062 (± 0.0216) | 0.03 (± 0.0006) | 1.23 (± 0.0205) | 0.03 (± 0.0006) | 1.397 (± 0.0236) | 0.03 (± 0.0007) |
|  | 1.168 (± 0.0203) | 0.036 (± 0.0006) | 1.569 (± 0.0207) | 0.045 (± 0.0007) | 1.877 (± 0.0217) | 0.054 (± 0.0009) |
|  | 0.656 (± 0.0438) | 0.017 (± 0.0017) | 0.825 (± 0.0496) | 0.019 (± 0.0019) | 0.904 (± 0.0589) | 0.02 (± 0.0024) |
|  | 1.04 (± 0.0219) | 0.029 (± 0.0006) | 1.139 (± 0.0208) | 0.026 (± 0.0006) | 1.19 (± 0.0225) | 0.022 (± 0.0006) |
| Standard Power | 0.368 (± 0.0078) |  | 0.318 (± 0.0064) |  | 0.29 (± 0.0063) |  |

[CAPTION] Table 11: Comparison of conformal outlier detection methods on TinyImageNet dataset (outliers) and CIFAR-10 dataset


<!-- page 40 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
Table 12: Comparison of conformal outlier detection methods on Texture dataset (outliers) and CIFAR-10 dataset (inliers)
for varying contamination rate r and target type-I error level α. All methods utilize the ReAct (Sun et al., 2021) method
with a pretrained VGG-19. The empirical power is presented relative to the Standard method (higher is better). Results
are averaged across 100 random splits of the data, with standard errors presented in parentheses.
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.026)
0.008 (± 0.0003)
1.0 (± 0.0312)
0.005 (± 0.0002)
1.0 (± 0.0342)
0.003 (± 0.0002)
Oracle (infeasible)
1.119 (± 0.0263)
0.01 (± 0.0004)
1.421 (± 0.032)
0.01 (± 0.0004)
1.833 (± 0.0437)
0.009 (± 0.0003)
Naive-Trim (invalid)
1.368 (± 0.0278)
0.017 (± 0.0005)
2.168 (± 0.0379)
0.029 (± 0.0006)
3.383 (± 0.0481)
0.037 (± 0.0007)
Small-Clean
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
Label-Trim
1.119 (± 0.0263)
0.01 (± 0.0004)
1.401 (± 0.0321)
0.01 (± 0.0003)
1.712 (± 0.0429)
0.008 (± 0.0003)
Standard Power
0.239 (± 0.0062)
0.008 (± 0.0003)
0.188 (± 0.0059)
0.005 (± 0.0002)
0.139 (± 0.0048)
0.003 (± 0.0002)
(a) Target type-I error rate α = 0.01
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0202)
0.017 (± 0.0005)
1.0 (± 0.0223)
0.012 (± 0.0004)
1.0 (± 0.0249)
0.008 (± 0.0003)
Oracle (infeasible)
1.083 (± 0.0186)
0.021 (± 0.0005)
1.253 (± 0.0246)
0.02 (± 0.0005)
1.502 (± 0.0292)
0.019 (± 0.0006)
Naive-Trim (invalid)
1.223 (± 0.0201)
0.027 (± 0.0006)
1.605 (± 0.0241)
0.037 (± 0.0007)
2.15 (± 0.0295)
0.045 (± 0.0008)
Small-Clean
0.841 (± 0.052)
0.018 (± 0.0018)
0.752 (± 0.0805)
0.016 (± 0.0025)
0.437 (± 0.071)
0.006 (± 0.0012)
Label-Trim
1.072 (± 0.0185)
0.02 (± 0.0005)
1.2 (± 0.0234)
0.018 (± 0.0005)
1.323 (± 0.0276)
0.015 (± 0.0005)
Standard Power
0.331 (± 0.0067)
0.017 (± 0.0005)
0.283 (± 0.0063)
0.012 (± 0.0004)
0.237 (± 0.0059)
0.008 (± 0.0003)
(b) Target type-I error rate α = 0.02
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0164)
0.027 (± 0.0006)
1.0 (± 0.02)
0.02 (± 0.0005)
1.0 (± 0.0202)
0.015 (± 0.0005)
Oracle (infeasible)
1.075 (± 0.0153)
0.031 (± 0.0006)
1.179 (± 0.0201)
0.03 (± 0.0007)
1.356 (± 0.022)
0.029 (± 0.0007)
Naive-Trim (invalid)
1.146 (± 0.0155)
0.036 (± 0.0007)
1.408 (± 0.0187)
0.045 (± 0.0008)
1.75 (± 0.0228)
0.053 (± 0.0009)
Small-Clean
0.801 (± 0.0382)
0.022 (± 0.0018)
0.937 (± 0.0569)
0.025 (± 0.0028)
1.016 (± 0.0548)
0.02 (± 0.0021)
Label-Trim
1.053 (± 0.016)
0.03 (± 0.0006)
1.117 (± 0.02)
0.027 (± 0.0006)
1.208 (± 0.0221)
0.022 (± 0.0006)
Standard Power
0.408 (± 0.0067)
0.027 (± 0.0006)
0.353 (± 0.0071)
0.02 (± 0.0005)
0.312 (± 0.0063)
0.015 (± 0.0005)
(c) Target type-I error rate α = 0.03
40


**[Table p40.1]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.026) | 0.008 (± 0.0003) | 1.0 (± 0.0312) | 0.005 (± 0.0002) | 1.0 (± 0.0342) | 0.003 (± 0.0002) |
|  | 1.119 (± 0.0263) | 0.01 (± 0.0004) | 1.421 (± 0.032) | 0.01 (± 0.0004) | 1.833 (± 0.0437) | 0.009 (± 0.0003) |
|  | 1.368 (± 0.0278) | 0.017 (± 0.0005) | 2.168 (± 0.0379) | 0.029 (± 0.0006) | 3.383 (± 0.0481) | 0.037 (± 0.0007) |
|  | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) |
|  | 1.119 (± 0.0263) | 0.01 (± 0.0004) | 1.401 (± 0.0321) | 0.01 (± 0.0003) | 1.712 (± 0.0429) | 0.008 (± 0.0003) |
| Standard Power | 0.239 (± 0.0062) | 0.008 (± 0.0003) | 0.188 (± 0.0059) | 0.005 (± 0.0002) | 0.139 (± 0.0048) | 0.003 (± 0.0002) |


**[Table p40.2]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0202) | 0.017 (± 0.0005) | 1.0 (± 0.0223) | 0.012 (± 0.0004) | 1.0 (± 0.0249) | 0.008 (± 0.0003) |
|  | 1.083 (± 0.0186) | 0.021 (± 0.0005) | 1.253 (± 0.0246) | 0.02 (± 0.0005) | 1.502 (± 0.0292) | 0.019 (± 0.0006) |
|  | 1.223 (± 0.0201) | 0.027 (± 0.0006) | 1.605 (± 0.0241) | 0.037 (± 0.0007) | 2.15 (± 0.0295) | 0.045 (± 0.0008) |
|  | 0.841 (± 0.052) | 0.018 (± 0.0018) | 0.752 (± 0.0805) | 0.016 (± 0.0025) | 0.437 (± 0.071) | 0.006 (± 0.0012) |
|  | 1.072 (± 0.0185) | 0.02 (± 0.0005) | 1.2 (± 0.0234) | 0.018 (± 0.0005) | 1.323 (± 0.0276) | 0.015 (± 0.0005) |
| Standard Power | 0.331 (± 0.0067) | 0.017 (± 0.0005) | 0.283 (± 0.0063) | 0.012 (± 0.0004) | 0.237 (± 0.0059) | 0.008 (± 0.0003) |


**[Table p40.3]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0164) | 0.027 (± 0.0006) | 1.0 (± 0.02) | 0.02 (± 0.0005) | 1.0 (± 0.0202) | 0.015 (± 0.0005) |
|  | 1.075 (± 0.0153) | 0.031 (± 0.0006) | 1.179 (± 0.0201) | 0.03 (± 0.0007) | 1.356 (± 0.022) | 0.029 (± 0.0007) |
|  | 1.146 (± 0.0155) | 0.036 (± 0.0007) | 1.408 (± 0.0187) | 0.045 (± 0.0008) | 1.75 (± 0.0228) | 0.053 (± 0.0009) |
|  | 0.801 (± 0.0382) | 0.022 (± 0.0018) | 0.937 (± 0.0569) | 0.025 (± 0.0028) | 1.016 (± 0.0548) | 0.02 (± 0.0021) |
|  | 1.053 (± 0.016) | 0.03 (± 0.0006) | 1.117 (± 0.02) | 0.027 (± 0.0006) | 1.208 (± 0.0221) | 0.022 (± 0.0006) |
| Standard Power | 0.408 (± 0.0067) | 0.027 (± 0.0006) | 0.353 (± 0.0071) | 0.02 (± 0.0005) | 0.312 (± 0.0063) | 0.015 (± 0.0005) |

[CAPTION] Table 12: Comparison of conformal outlier detection methods on Texture dataset (outliers) and CIFAR-10 dataset (inliers)


<!-- page 41 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
Table 13: Comparison of conformal outlier detection methods on SVHN dataset (outliers) and CIFAR-10 dataset (inliers)
for varying contamination rate r and target type-I error level α. All methods utilize the ReAct (Sun et al., 2021) method
with a pretrained VGG-19. The empirical power is presented relative to the Standard method (higher is better). Results
are averaged across 100 random splits of the data, with standard errors presented in parentheses.
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0297)
0.009 (± 0.0003)
1.0 (± 0.0355)
0.006 (± 0.0003)
1.0 (± 0.0463)
0.004 (± 0.0002)
Oracle (infeasible)
1.085 (± 0.0305)
0.01 (± 0.0004)
1.287 (± 0.0384)
0.01 (± 0.0004)
1.607 (± 0.0452)
0.009 (± 0.0003)
Naive-Trim (invalid)
1.355 (± 0.0356)
0.018 (± 0.0005)
2.261 (± 0.0511)
0.031 (± 0.0006)
3.202 (± 0.053)
0.04 (± 0.0007)
Small-Clean
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
Label-Trim
1.085 (± 0.0305)
0.01 (± 0.0004)
1.287 (± 0.0384)
0.01 (± 0.0004)
1.573 (± 0.0467)
0.009 (± 0.0003)
Standard Power
0.182 (± 0.0054)
0.009 (± 0.0003)
0.145 (± 0.0052)
0.006 (± 0.0003)
0.12 (± 0.0056)
0.004 (± 0.0002)
(a) Target type-I error rate α = 0.01
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.026)
0.018 (± 0.0005)
1.0 (± 0.0274)
0.014 (± 0.0004)
1.0 (± 0.0285)
0.01 (± 0.0003)
Oracle (infeasible)
1.038 (± 0.026)
0.021 (± 0.0005)
1.167 (± 0.0298)
0.02 (± 0.0005)
1.289 (± 0.0294)
0.019 (± 0.0006)
Naive-Trim (invalid)
1.222 (± 0.0269)
0.028 (± 0.0006)
1.77 (± 0.0354)
0.039 (± 0.0007)
2.198 (± 0.0315)
0.048 (± 0.0008)
Small-Clean
0.894 (± 0.0609)
0.019 (± 0.0019)
0.504 (± 0.07)
0.009 (± 0.0015)
0.389 (± 0.0722)
0.006 (± 0.0015)
Label-Trim
1.035 (± 0.0256)
0.02 (± 0.0005)
1.146 (± 0.0293)
0.019 (± 0.0005)
1.217 (± 0.0301)
0.017 (± 0.0005)
Standard Power
0.249 (± 0.0065)
0.018 (± 0.0005)
0.216 (± 0.0059)
0.014 (± 0.0004)
0.196 (± 0.0056)
0.01 (± 0.0003)
(b) Target type-I error rate α = 0.02
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0219)
0.028 (± 0.0006)
1.0 (± 0.0251)
0.023 (± 0.0005)
1.0 (± 0.0248)
0.018 (± 0.0005)
Oracle (infeasible)
1.074 (± 0.0225)
0.031 (± 0.0006)
1.206 (± 0.0281)
0.03 (± 0.0007)
1.302 (± 0.0261)
0.029 (± 0.0007)
Naive-Trim (invalid)
1.224 (± 0.0216)
0.037 (± 0.0007)
1.626 (± 0.0277)
0.048 (± 0.0008)
1.962 (± 0.0257)
0.056 (± 0.0009)
Small-Clean
0.788 (± 0.0485)
0.02 (± 0.0019)
0.878 (± 0.0536)
0.019 (± 0.0018)
0.986 (± 0.0625)
0.02 (± 0.0021)
Label-Trim
1.05 (± 0.0224)
0.03 (± 0.0006)
1.134 (± 0.0265)
0.028 (± 0.0006)
1.16 (± 0.0247)
0.024 (± 0.0006)
Standard Power
0.307 (± 0.0067)
0.028 (± 0.0006)
0.266 (± 0.0067)
0.023 (± 0.0005)
0.241 (± 0.006)
0.018 (± 0.0005)
(c) Target type-I error rate α = 0.03
41


**[Table p41.1]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0297) | 0.009 (± 0.0003) | 1.0 (± 0.0355) | 0.006 (± 0.0003) | 1.0 (± 0.0463) | 0.004 (± 0.0002) |
|  | 1.085 (± 0.0305) | 0.01 (± 0.0004) | 1.287 (± 0.0384) | 0.01 (± 0.0004) | 1.607 (± 0.0452) | 0.009 (± 0.0003) |
|  | 1.355 (± 0.0356) | 0.018 (± 0.0005) | 2.261 (± 0.0511) | 0.031 (± 0.0006) | 3.202 (± 0.053) | 0.04 (± 0.0007) |
|  | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) |
|  | 1.085 (± 0.0305) | 0.01 (± 0.0004) | 1.287 (± 0.0384) | 0.01 (± 0.0004) | 1.573 (± 0.0467) | 0.009 (± 0.0003) |
| Standard Power | 0.182 (± 0.0054) | 0.009 (± 0.0003) | 0.145 (± 0.0052) | 0.006 (± 0.0003) | 0.12 (± 0.0056) | 0.004 (± 0.0002) |


**[Table p41.2]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.026) | 0.018 (± 0.0005) | 1.0 (± 0.0274) | 0.014 (± 0.0004) | 1.0 (± 0.0285) | 0.01 (± 0.0003) |
|  | 1.038 (± 0.026) | 0.021 (± 0.0005) | 1.167 (± 0.0298) | 0.02 (± 0.0005) | 1.289 (± 0.0294) | 0.019 (± 0.0006) |
|  | 1.222 (± 0.0269) | 0.028 (± 0.0006) | 1.77 (± 0.0354) | 0.039 (± 0.0007) | 2.198 (± 0.0315) | 0.048 (± 0.0008) |
|  | 0.894 (± 0.0609) | 0.019 (± 0.0019) | 0.504 (± 0.07) | 0.009 (± 0.0015) | 0.389 (± 0.0722) | 0.006 (± 0.0015) |
|  | 1.035 (± 0.0256) | 0.02 (± 0.0005) | 1.146 (± 0.0293) | 0.019 (± 0.0005) | 1.217 (± 0.0301) | 0.017 (± 0.0005) |
| Standard Power | 0.249 (± 0.0065) | 0.018 (± 0.0005) | 0.216 (± 0.0059) | 0.014 (± 0.0004) | 0.196 (± 0.0056) | 0.01 (± 0.0003) |


**[Table p41.3]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0219) | 0.028 (± 0.0006) | 1.0 (± 0.0251) | 0.023 (± 0.0005) | 1.0 (± 0.0248) | 0.018 (± 0.0005) |
|  | 1.074 (± 0.0225) | 0.031 (± 0.0006) | 1.206 (± 0.0281) | 0.03 (± 0.0007) | 1.302 (± 0.0261) | 0.029 (± 0.0007) |
|  | 1.224 (± 0.0216) | 0.037 (± 0.0007) | 1.626 (± 0.0277) | 0.048 (± 0.0008) | 1.962 (± 0.0257) | 0.056 (± 0.0009) |
|  | 0.788 (± 0.0485) | 0.02 (± 0.0019) | 0.878 (± 0.0536) | 0.019 (± 0.0018) | 0.986 (± 0.0625) | 0.02 (± 0.0021) |
|  | 1.05 (± 0.0224) | 0.03 (± 0.0006) | 1.134 (± 0.0265) | 0.028 (± 0.0006) | 1.16 (± 0.0247) | 0.024 (± 0.0006) |
| Standard Power | 0.307 (± 0.0067) | 0.028 (± 0.0006) | 0.266 (± 0.0067) | 0.023 (± 0.0005) | 0.241 (± 0.006) | 0.018 (± 0.0005) |

[CAPTION] Table 13: Comparison of conformal outlier detection methods on SVHN dataset (outliers) and CIFAR-10 dataset (inliers)


<!-- page 42 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
Table 14: Comparison of conformal outlier detection methods on Places365 dataset (outliers) and CIFAR-10 dataset (inliers)
for varying contamination rate r and target type-I error level α. All methods utilize the ReAct (Sun et al., 2021) method
with a pretrained VGG-19. The empirical power is presented relative to the Standard method (higher is better). Results
are averaged across 100 random splits of the data, with standard errors presented in parentheses.
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0289)
0.009 (± 0.0004)
1.0 (± 0.0383)
0.006 (± 0.0003)
1.0 (± 0.0382)
0.005 (± 0.0003)
Oracle (infeasible)
1.084 (± 0.0295)
0.01 (± 0.0004)
1.444 (± 0.0407)
0.01 (± 0.0004)
1.602 (± 0.0488)
0.009 (± 0.0003)
Naive-Trim (invalid)
1.454 (± 0.0346)
0.018 (± 0.0005)
2.623 (± 0.0521)
0.03 (± 0.0006)
3.405 (± 0.0592)
0.04 (± 0.0008)
Small-Clean
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
Label-Trim
1.084 (± 0.0295)
0.01 (± 0.0004)
1.441 (± 0.0404)
0.01 (± 0.0003)
1.578 (± 0.0476)
0.009 (± 0.0003)
Standard Power
0.176 (± 0.0051)
0.009 (± 0.0004)
0.134 (± 0.0051)
0.006 (± 0.0003)
0.12 (± 0.0046)
0.005 (± 0.0003)
(a) Target type-I error rate α = 0.01
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0236)
0.018 (± 0.0005)
1.0 (± 0.0251)
0.014 (± 0.0004)
1.0 (± 0.0296)
0.01 (± 0.0004)
Oracle (infeasible)
1.074 (± 0.0238)
0.021 (± 0.0005)
1.247 (± 0.0291)
0.02 (± 0.0005)
1.35 (± 0.0332)
0.019 (± 0.0006)
Naive-Trim (invalid)
1.236 (± 0.0236)
0.028 (± 0.0006)
1.781 (± 0.0308)
0.039 (± 0.0007)
2.166 (± 0.0346)
0.048 (± 0.0009)
Small-Clean
0.71 (± 0.0577)
0.015 (± 0.0017)
0.569 (± 0.0713)
0.011 (± 0.0016)
0.268 (± 0.0606)
0.004 (± 0.0013)
Label-Trim
1.058 (± 0.0241)
0.02 (± 0.0005)
1.198 (± 0.0281)
0.019 (± 0.0005)
1.231 (± 0.0315)
0.016 (± 0.0005)
Standard Power
0.259 (± 0.0061)
0.018 (± 0.0005)
0.226 (± 0.0057)
0.014 (± 0.0004)
0.205 (± 0.0061)
0.01 (± 0.0004)
(b) Target type-I error rate α = 0.02
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0188)
0.028 (± 0.0006)
1.0 (± 0.0221)
0.022 (± 0.0005)
1.0 (± 0.0261)
0.017 (± 0.0005)
Oracle (infeasible)
1.088 (± 0.0187)
0.031 (± 0.0006)
1.197 (± 0.0233)
0.03 (± 0.0007)
1.32 (± 0.0288)
0.029 (± 0.0007)
Naive-Trim (invalid)
1.186 (± 0.0185)
0.037 (± 0.0007)
1.499 (± 0.0249)
0.048 (± 0.0008)
1.83 (± 0.026)
0.056 (± 0.0009)
Small-Clean
0.652 (± 0.045)
0.017 (± 0.0017)
0.85 (± 0.0541)
0.021 (± 0.0018)
0.921 (± 0.054)
0.02 (± 0.002)
Label-Trim
1.056 (± 0.019)
0.03 (± 0.0006)
1.12 (± 0.0232)
0.027 (± 0.0006)
1.168 (± 0.0276)
0.023 (± 0.0006)
Standard Power
0.321 (± 0.006)
0.028 (± 0.0006)
0.294 (± 0.0065)
0.022 (± 0.0005)
0.263 (± 0.0069)
0.017 (± 0.0005)
(c) Target type-I error rate α = 0.03
42


**[Table p42.1]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0289) | 0.009 (± 0.0004) | 1.0 (± 0.0383) | 0.006 (± 0.0003) | 1.0 (± 0.0382) | 0.005 (± 0.0003) |
|  | 1.084 (± 0.0295) | 0.01 (± 0.0004) | 1.444 (± 0.0407) | 0.01 (± 0.0004) | 1.602 (± 0.0488) | 0.009 (± 0.0003) |
|  | 1.454 (± 0.0346) | 0.018 (± 0.0005) | 2.623 (± 0.0521) | 0.03 (± 0.0006) | 3.405 (± 0.0592) | 0.04 (± 0.0008) |
|  | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) |
|  | 1.084 (± 0.0295) | 0.01 (± 0.0004) | 1.441 (± 0.0404) | 0.01 (± 0.0003) | 1.578 (± 0.0476) | 0.009 (± 0.0003) |
| Standard Power | 0.176 (± 0.0051) | 0.009 (± 0.0004) | 0.134 (± 0.0051) | 0.006 (± 0.0003) | 0.12 (± 0.0046) | 0.005 (± 0.0003) |


**[Table p42.2]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0236) | 0.018 (± 0.0005) | 1.0 (± 0.0251) | 0.014 (± 0.0004) | 1.0 (± 0.0296) | 0.01 (± 0.0004) |
|  | 1.074 (± 0.0238) | 0.021 (± 0.0005) | 1.247 (± 0.0291) | 0.02 (± 0.0005) | 1.35 (± 0.0332) | 0.019 (± 0.0006) |
|  | 1.236 (± 0.0236) | 0.028 (± 0.0006) | 1.781 (± 0.0308) | 0.039 (± 0.0007) | 2.166 (± 0.0346) | 0.048 (± 0.0009) |
|  | 0.71 (± 0.0577) | 0.015 (± 0.0017) | 0.569 (± 0.0713) | 0.011 (± 0.0016) | 0.268 (± 0.0606) | 0.004 (± 0.0013) |
|  | 1.058 (± 0.0241) | 0.02 (± 0.0005) | 1.198 (± 0.0281) | 0.019 (± 0.0005) | 1.231 (± 0.0315) | 0.016 (± 0.0005) |
| Standard Power | 0.259 (± 0.0061) | 0.018 (± 0.0005) | 0.226 (± 0.0057) | 0.014 (± 0.0004) | 0.205 (± 0.0061) | 0.01 (± 0.0004) |


**[Table p42.3]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0188) | 0.028 (± 0.0006) | 1.0 (± 0.0221) | 0.022 (± 0.0005) | 1.0 (± 0.0261) | 0.017 (± 0.0005) |
|  | 1.088 (± 0.0187) | 0.031 (± 0.0006) | 1.197 (± 0.0233) | 0.03 (± 0.0007) | 1.32 (± 0.0288) | 0.029 (± 0.0007) |
|  | 1.186 (± 0.0185) | 0.037 (± 0.0007) | 1.499 (± 0.0249) | 0.048 (± 0.0008) | 1.83 (± 0.026) | 0.056 (± 0.0009) |
|  | 0.652 (± 0.045) | 0.017 (± 0.0017) | 0.85 (± 0.0541) | 0.021 (± 0.0018) | 0.921 (± 0.054) | 0.02 (± 0.002) |
|  | 1.056 (± 0.019) | 0.03 (± 0.0006) | 1.12 (± 0.0232) | 0.027 (± 0.0006) | 1.168 (± 0.0276) | 0.023 (± 0.0006) |
| Standard Power | 0.321 (± 0.006) | 0.028 (± 0.0006) | 0.294 (± 0.0065) | 0.022 (± 0.0005) | 0.263 (± 0.0069) | 0.017 (± 0.0005) |

[CAPTION] Table 14: Comparison of conformal outlier detection methods on Places365 dataset (outliers) and CIFAR-10 dataset (inliers)


<!-- page 43 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
Table 15: Comparison of conformal outlier detection methods on MNIST dataset (outliers) and CIFAR-10 dataset (inliers)
for varying contamination rate r and target type-I error level α. All methods utilize the ReAct (Sun et al., 2021) method
with a pretrained VGG-19. The empirical power is presented relative to the Standard method (higher is better). Results
are averaged across 100 random splits of the data, with standard errors presented in parentheses.
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0315)
0.009 (± 0.0004)
1.0 (± 0.0376)
0.007 (± 0.0003)
1.0 (± 0.0428)
0.005 (± 0.0003)
Oracle (infeasible)
1.089 (± 0.0325)
0.01 (± 0.0004)
1.349 (± 0.0414)
0.01 (± 0.0004)
1.547 (± 0.0488)
0.009 (± 0.0003)
Naive-Trim (invalid)
1.493 (± 0.0352)
0.018 (± 0.0005)
2.499 (± 0.0571)
0.031 (± 0.0006)
3.315 (± 0.0664)
0.041 (± 0.0008)
Small-Clean
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
Label-Trim
1.089 (± 0.0325)
0.01 (± 0.0004)
1.347 (± 0.0414)
0.01 (± 0.0004)
1.549 (± 0.0492)
0.009 (± 0.0003)
Standard Power
0.162 (± 0.0051)
0.009 (± 0.0004)
0.134 (± 0.005)
0.007 (± 0.0003)
0.116 (± 0.005)
0.005 (± 0.0003)
(a) Target type-I error rate α = 0.01
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0234)
0.018 (± 0.0005)
1.0 (± 0.0283)
0.014 (± 0.0004)
1.0 (± 0.0298)
0.01 (± 0.0004)
Oracle (infeasible)
1.068 (± 0.0225)
0.021 (± 0.0005)
1.219 (± 0.0313)
0.02 (± 0.0005)
1.36 (± 0.0343)
0.019 (± 0.0006)
Naive-Trim (invalid)
1.246 (± 0.0245)
0.028 (± 0.0006)
1.76 (± 0.0341)
0.039 (± 0.0007)
2.167 (± 0.0395)
0.049 (± 0.0009)
Small-Clean
0.781 (± 0.0592)
0.017 (± 0.0019)
0.617 (± 0.0743)
0.012 (± 0.0018)
0.302 (± 0.0582)
0.005 (± 0.0011)
Label-Trim
1.053 (± 0.0224)
0.02 (± 0.0005)
1.176 (± 0.0318)
0.019 (± 0.0005)
1.242 (± 0.0326)
0.016 (± 0.0005)
Standard Power
0.244 (± 0.0057)
0.018 (± 0.0005)
0.214 (± 0.0061)
0.014 (± 0.0004)
0.194 (± 0.0058)
0.01 (± 0.0004)
(b) Target type-I error rate α = 0.02
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0195)
0.028 (± 0.0006)
1.0 (± 0.0247)
0.023 (± 0.0006)
1.0 (± 0.0262)
0.018 (± 0.0005)
Oracle (infeasible)
1.062 (± 0.0196)
0.031 (± 0.0006)
1.189 (± 0.0279)
0.03 (± 0.0007)
1.296 (± 0.0305)
0.029 (± 0.0007)
Naive-Trim (invalid)
1.178 (± 0.0208)
0.037 (± 0.0007)
1.501 (± 0.0257)
0.049 (± 0.0008)
1.799 (± 0.0319)
0.057 (± 0.0009)
Small-Clean
0.648 (± 0.0456)
0.017 (± 0.0018)
0.888 (± 0.0515)
0.023 (± 0.002)
0.964 (± 0.0563)
0.021 (± 0.0018)
Label-Trim
1.039 (± 0.0196)
0.03 (± 0.0006)
1.109 (± 0.0258)
0.027 (± 0.0006)
1.145 (± 0.0268)
0.024 (± 0.0006)
Standard Power
0.305 (± 0.0059)
0.028 (± 0.0006)
0.276 (± 0.0068)
0.023 (± 0.0006)
0.25 (± 0.0066)
0.018 (± 0.0005)
(c) Target type-I error rate α = 0.03
43


**[Table p43.1]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0315) | 0.009 (± 0.0004) | 1.0 (± 0.0376) | 0.007 (± 0.0003) | 1.0 (± 0.0428) | 0.005 (± 0.0003) |
|  | 1.089 (± 0.0325) | 0.01 (± 0.0004) | 1.349 (± 0.0414) | 0.01 (± 0.0004) | 1.547 (± 0.0488) | 0.009 (± 0.0003) |
|  | 1.493 (± 0.0352) | 0.018 (± 0.0005) | 2.499 (± 0.0571) | 0.031 (± 0.0006) | 3.315 (± 0.0664) | 0.041 (± 0.0008) |
|  | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) |
|  | 1.089 (± 0.0325) | 0.01 (± 0.0004) | 1.347 (± 0.0414) | 0.01 (± 0.0004) | 1.549 (± 0.0492) | 0.009 (± 0.0003) |
| Standard Power | 0.162 (± 0.0051) | 0.009 (± 0.0004) | 0.134 (± 0.005) | 0.007 (± 0.0003) | 0.116 (± 0.005) | 0.005 (± 0.0003) |


**[Table p43.2]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0234) | 0.018 (± 0.0005) | 1.0 (± 0.0283) | 0.014 (± 0.0004) | 1.0 (± 0.0298) | 0.01 (± 0.0004) |
|  | 1.068 (± 0.0225) | 0.021 (± 0.0005) | 1.219 (± 0.0313) | 0.02 (± 0.0005) | 1.36 (± 0.0343) | 0.019 (± 0.0006) |
|  | 1.246 (± 0.0245) | 0.028 (± 0.0006) | 1.76 (± 0.0341) | 0.039 (± 0.0007) | 2.167 (± 0.0395) | 0.049 (± 0.0009) |
|  | 0.781 (± 0.0592) | 0.017 (± 0.0019) | 0.617 (± 0.0743) | 0.012 (± 0.0018) | 0.302 (± 0.0582) | 0.005 (± 0.0011) |
|  | 1.053 (± 0.0224) | 0.02 (± 0.0005) | 1.176 (± 0.0318) | 0.019 (± 0.0005) | 1.242 (± 0.0326) | 0.016 (± 0.0005) |
| Standard Power | 0.244 (± 0.0057) | 0.018 (± 0.0005) | 0.214 (± 0.0061) | 0.014 (± 0.0004) | 0.194 (± 0.0058) | 0.01 (± 0.0004) |


**[Table p43.3]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0195) | 0.028 (± 0.0006) | 1.0 (± 0.0247) | 0.023 (± 0.0006) | 1.0 (± 0.0262) | 0.018 (± 0.0005) |
|  | 1.062 (± 0.0196) | 0.031 (± 0.0006) | 1.189 (± 0.0279) | 0.03 (± 0.0007) | 1.296 (± 0.0305) | 0.029 (± 0.0007) |
|  | 1.178 (± 0.0208) | 0.037 (± 0.0007) | 1.501 (± 0.0257) | 0.049 (± 0.0008) | 1.799 (± 0.0319) | 0.057 (± 0.0009) |
|  | 0.648 (± 0.0456) | 0.017 (± 0.0018) | 0.888 (± 0.0515) | 0.023 (± 0.002) | 0.964 (± 0.0563) | 0.021 (± 0.0018) |
|  | 1.039 (± 0.0196) | 0.03 (± 0.0006) | 1.109 (± 0.0258) | 0.027 (± 0.0006) | 1.145 (± 0.0268) | 0.024 (± 0.0006) |
| Standard Power | 0.305 (± 0.0059) | 0.028 (± 0.0006) | 0.276 (± 0.0068) | 0.023 (± 0.0006) | 0.25 (± 0.0066) | 0.018 (± 0.0005) |

[CAPTION] Table 15: Comparison of conformal outlier detection methods on MNIST dataset (outliers) and CIFAR-10 dataset (inliers)


<!-- page 44 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
Table 16: Comparison of conformal outlier detection methods on CIFAR-100 dataset (outliers) and CIFAR-10 dataset
(inliers) for varying contamination rate r and target type-I error level α. All methods utilize the ReAct (Sun et al., 2021)
method with a pretrained VGG-19. The empirical power is presented relative to the Standard method (higher is better).
Results are averaged across 100 random splits of the data, with standard errors presented in parentheses.
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0388)
0.009 (± 0.0004)
1.0 (± 0.0466)
0.007 (± 0.0003)
1.0 (± 0.0472)
0.005 (± 0.0003)
Oracle (infeasible)
1.089 (± 0.0409)
0.01 (± 0.0004)
1.34 (± 0.0508)
0.01 (± 0.0004)
1.554 (± 0.0539)
0.009 (± 0.0003)
Naive-Trim (invalid)
1.558 (± 0.0455)
0.018 (± 0.0005)
2.816 (± 0.0647)
0.032 (± 0.0007)
3.58 (± 0.0691)
0.042 (± 0.0008)
Small-Clean
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
Label-Trim
1.089 (± 0.0409)
0.01 (± 0.0004)
1.34 (± 0.0508)
0.01 (± 0.0004)
1.572 (± 0.053)
0.009 (± 0.0003)
Standard Power
0.137 (± 0.0053)
0.009 (± 0.0004)
0.111 (± 0.0052)
0.007 (± 0.0003)
0.1 (± 0.0047)
0.005 (± 0.0003)
(a) Target type-I error rate α = 0.01
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0288)
0.019 (± 0.0005)
1.0 (± 0.0308)
0.015 (± 0.0005)
1.0 (± 0.0307)
0.012 (± 0.0004)
Oracle (infeasible)
1.07 (± 0.0289)
0.021 (± 0.0005)
1.219 (± 0.0332)
0.02 (± 0.0005)
1.316 (± 0.0362)
0.019 (± 0.0006)
Naive-Trim (invalid)
1.293 (± 0.0301)
0.028 (± 0.0006)
1.811 (± 0.0395)
0.04 (± 0.0007)
2.24 (± 0.0396)
0.051 (± 0.0009)
Small-Clean
0.831 (± 0.0636)
0.018 (± 0.002)
0.646 (± 0.0826)
0.013 (± 0.0021)
0.317 (± 0.0665)
0.005 (± 0.0016)
Label-Trim
1.057 (± 0.0281)
0.02 (± 0.0005)
1.17 (± 0.0327)
0.019 (± 0.0005)
1.233 (± 0.0343)
0.017 (± 0.0006)
Standard Power
0.217 (± 0.0063)
0.019 (± 0.0005)
0.192 (± 0.0059)
0.015 (± 0.0005)
0.175 (± 0.0054)
0.012 (± 0.0004)
(b) Target type-I error rate α = 0.02
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.023)
0.028 (± 0.0006)
1.0 (± 0.0272)
0.023 (± 0.0006)
1.0 (± 0.0269)
0.019 (± 0.0005)
Oracle (infeasible)
1.061 (± 0.0233)
0.031 (± 0.0006)
1.183 (± 0.0278)
0.03 (± 0.0007)
1.292 (± 0.0306)
0.029 (± 0.0007)
Naive-Trim (invalid)
1.184 (± 0.0249)
0.038 (± 0.0007)
1.512 (± 0.0294)
0.05 (± 0.0008)
1.868 (± 0.0324)
0.059 (± 0.0009)
Small-Clean
0.706 (± 0.0461)
0.02 (± 0.002)
0.816 (± 0.0583)
0.021 (± 0.0022)
0.856 (± 0.0579)
0.018 (± 0.0021)
Label-Trim
1.039 (± 0.0226)
0.03 (± 0.0006)
1.111 (± 0.0275)
0.028 (± 0.0006)
1.154 (± 0.0291)
0.024 (± 0.0006)
Standard Power
0.282 (± 0.0065)
0.028 (± 0.0006)
0.254 (± 0.0069)
0.023 (± 0.0006)
0.228 (± 0.0062)
0.019 (± 0.0005)
(c) Target type-I error rate α = 0.03
44


**[Table p44.1]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0388) | 0.009 (± 0.0004) | 1.0 (± 0.0466) | 0.007 (± 0.0003) | 1.0 (± 0.0472) | 0.005 (± 0.0003) |
|  | 1.089 (± 0.0409) | 0.01 (± 0.0004) | 1.34 (± 0.0508) | 0.01 (± 0.0004) | 1.554 (± 0.0539) | 0.009 (± 0.0003) |
|  | 1.558 (± 0.0455) | 0.018 (± 0.0005) | 2.816 (± 0.0647) | 0.032 (± 0.0007) | 3.58 (± 0.0691) | 0.042 (± 0.0008) |
|  | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) |
|  | 1.089 (± 0.0409) | 0.01 (± 0.0004) | 1.34 (± 0.0508) | 0.01 (± 0.0004) | 1.572 (± 0.053) | 0.009 (± 0.0003) |
| Standard Power | 0.137 (± 0.0053) | 0.009 (± 0.0004) | 0.111 (± 0.0052) | 0.007 (± 0.0003) | 0.1 (± 0.0047) | 0.005 (± 0.0003) |


**[Table p44.2]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0288) | 0.019 (± 0.0005) | 1.0 (± 0.0308) | 0.015 (± 0.0005) | 1.0 (± 0.0307) | 0.012 (± 0.0004) |
|  | 1.07 (± 0.0289) | 0.021 (± 0.0005) | 1.219 (± 0.0332) | 0.02 (± 0.0005) | 1.316 (± 0.0362) | 0.019 (± 0.0006) |
|  | 1.293 (± 0.0301) | 0.028 (± 0.0006) | 1.811 (± 0.0395) | 0.04 (± 0.0007) | 2.24 (± 0.0396) | 0.051 (± 0.0009) |
|  | 0.831 (± 0.0636) | 0.018 (± 0.002) | 0.646 (± 0.0826) | 0.013 (± 0.0021) | 0.317 (± 0.0665) | 0.005 (± 0.0016) |
|  | 1.057 (± 0.0281) | 0.02 (± 0.0005) | 1.17 (± 0.0327) | 0.019 (± 0.0005) | 1.233 (± 0.0343) | 0.017 (± 0.0006) |
| Standard Power | 0.217 (± 0.0063) | 0.019 (± 0.0005) | 0.192 (± 0.0059) | 0.015 (± 0.0005) | 0.175 (± 0.0054) | 0.012 (± 0.0004) |


**[Table p44.3]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.023) | 0.028 (± 0.0006) | 1.0 (± 0.0272) | 0.023 (± 0.0006) | 1.0 (± 0.0269) | 0.019 (± 0.0005) |
|  | 1.061 (± 0.0233) | 0.031 (± 0.0006) | 1.183 (± 0.0278) | 0.03 (± 0.0007) | 1.292 (± 0.0306) | 0.029 (± 0.0007) |
|  | 1.184 (± 0.0249) | 0.038 (± 0.0007) | 1.512 (± 0.0294) | 0.05 (± 0.0008) | 1.868 (± 0.0324) | 0.059 (± 0.0009) |
|  | 0.706 (± 0.0461) | 0.02 (± 0.002) | 0.816 (± 0.0583) | 0.021 (± 0.0022) | 0.856 (± 0.0579) | 0.018 (± 0.0021) |
|  | 1.039 (± 0.0226) | 0.03 (± 0.0006) | 1.111 (± 0.0275) | 0.028 (± 0.0006) | 1.154 (± 0.0291) | 0.024 (± 0.0006) |
| Standard Power | 0.282 (± 0.0065) | 0.028 (± 0.0006) | 0.254 (± 0.0069) | 0.023 (± 0.0006) | 0.228 (± 0.0062) | 0.019 (± 0.0005) |

[CAPTION] Table 16: Comparison of conformal outlier detection methods on CIFAR-100 dataset (outliers) and CIFAR-10 dataset


<!-- page 45 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
Table 17: Comparison of conformal outlier detection methods on TinyImageNet dataset (outliers) and CIFAR-10 dataset
(inliers) for varying contamination rate r and target type-I error level α. All methods utilize the ReAct (Sun et al., 2021)
method with a pretrained VGG-19. The empirical power is presented relative to the Standard method (higher is better).
Results are averaged across 100 random splits of the data, with standard errors presented in parentheses.
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0353)
0.009 (± 0.0004)
1.0 (± 0.037)
0.007 (± 0.0003)
1.0 (± 0.0446)
0.005 (± 0.0002)
Oracle (infeasible)
1.104 (± 0.0383)
0.01 (± 0.0004)
1.37 (± 0.0474)
0.01 (± 0.0004)
1.653 (± 0.0523)
0.009 (± 0.0003)
Naive-Trim (invalid)
1.556 (± 0.0452)
0.018 (± 0.0005)
2.635 (± 0.0555)
0.031 (± 0.0006)
3.796 (± 0.0594)
0.04 (± 0.0008)
Small-Clean
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
Label-Trim
1.104 (± 0.0383)
0.01 (± 0.0004)
1.37 (± 0.0474)
0.01 (± 0.0004)
1.651 (± 0.0526)
0.009 (± 0.0003)
Standard Power
0.151 (± 0.0053)
0.009 (± 0.0004)
0.128 (± 0.0047)
0.007 (± 0.0003)
0.105 (± 0.0047)
0.005 (± 0.0002)
(a) Target type-I error rate α = 0.01
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0284)
0.019 (± 0.0005)
1.0 (± 0.0279)
0.014 (± 0.0004)
1.0 (± 0.0313)
0.011 (± 0.0004)
Oracle (infeasible)
1.074 (± 0.0281)
0.021 (± 0.0005)
1.221 (± 0.0303)
0.02 (± 0.0005)
1.4 (± 0.0317)
0.019 (± 0.0006)
Naive-Trim (invalid)
1.295 (± 0.0281)
0.028 (± 0.0006)
1.826 (± 0.0314)
0.039 (± 0.0007)
2.351 (± 0.0323)
0.049 (± 0.0008)
Small-Clean
0.823 (± 0.0624)
0.017 (± 0.0018)
0.471 (± 0.0636)
0.008 (± 0.0012)
0.401 (± 0.0691)
0.005 (± 0.001)
Label-Trim
1.06 (± 0.0275)
0.02 (± 0.0005)
1.173 (± 0.0302)
0.019 (± 0.0005)
1.272 (± 0.0311)
0.016 (± 0.0005)
Standard Power
0.239 (± 0.0068)
0.019 (± 0.0005)
0.213 (± 0.0059)
0.014 (± 0.0004)
0.187 (± 0.0059)
0.011 (± 0.0004)
(b) Target type-I error rate α = 0.02
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0217)
0.028 (± 0.0006)
1.0 (± 0.0239)
0.023 (± 0.0005)
1.0 (± 0.0239)
0.018 (± 0.0005)
Oracle (infeasible)
1.086 (± 0.0207)
0.031 (± 0.0006)
1.209 (± 0.0264)
0.03 (± 0.0007)
1.344 (± 0.0247)
0.029 (± 0.0007)
Naive-Trim (invalid)
1.196 (± 0.0201)
0.037 (± 0.0007)
1.512 (± 0.0223)
0.048 (± 0.0008)
1.879 (± 0.0245)
0.057 (± 0.0009)
Small-Clean
0.662 (± 0.0464)
0.018 (± 0.0018)
0.733 (± 0.05)
0.017 (± 0.0016)
0.848 (± 0.0556)
0.017 (± 0.0018)
Label-Trim
1.054 (± 0.0218)
0.03 (± 0.0006)
1.12 (± 0.0256)
0.027 (± 0.0006)
1.167 (± 0.0254)
0.023 (± 0.0006)
Standard Power
0.313 (± 0.0068)
0.028 (± 0.0006)
0.278 (± 0.0066)
0.023 (± 0.0005)
0.252 (± 0.006)
0.018 (± 0.0005)
(c) Target type-I error rate α = 0.03
45


**[Table p45.1]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0353) | 0.009 (± 0.0004) | 1.0 (± 0.037) | 0.007 (± 0.0003) | 1.0 (± 0.0446) | 0.005 (± 0.0002) |
|  | 1.104 (± 0.0383) | 0.01 (± 0.0004) | 1.37 (± 0.0474) | 0.01 (± 0.0004) | 1.653 (± 0.0523) | 0.009 (± 0.0003) |
|  | 1.556 (± 0.0452) | 0.018 (± 0.0005) | 2.635 (± 0.0555) | 0.031 (± 0.0006) | 3.796 (± 0.0594) | 0.04 (± 0.0008) |
|  | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) |
|  | 1.104 (± 0.0383) | 0.01 (± 0.0004) | 1.37 (± 0.0474) | 0.01 (± 0.0004) | 1.651 (± 0.0526) | 0.009 (± 0.0003) |
| Standard Power | 0.151 (± 0.0053) | 0.009 (± 0.0004) | 0.128 (± 0.0047) | 0.007 (± 0.0003) | 0.105 (± 0.0047) | 0.005 (± 0.0002) |


**[Table p45.2]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0284) | 0.019 (± 0.0005) | 1.0 (± 0.0279) | 0.014 (± 0.0004) | 1.0 (± 0.0313) | 0.011 (± 0.0004) |
|  | 1.074 (± 0.0281) | 0.021 (± 0.0005) | 1.221 (± 0.0303) | 0.02 (± 0.0005) | 1.4 (± 0.0317) | 0.019 (± 0.0006) |
|  | 1.295 (± 0.0281) | 0.028 (± 0.0006) | 1.826 (± 0.0314) | 0.039 (± 0.0007) | 2.351 (± 0.0323) | 0.049 (± 0.0008) |
|  | 0.823 (± 0.0624) | 0.017 (± 0.0018) | 0.471 (± 0.0636) | 0.008 (± 0.0012) | 0.401 (± 0.0691) | 0.005 (± 0.001) |
|  | 1.06 (± 0.0275) | 0.02 (± 0.0005) | 1.173 (± 0.0302) | 0.019 (± 0.0005) | 1.272 (± 0.0311) | 0.016 (± 0.0005) |
| Standard Power | 0.239 (± 0.0068) | 0.019 (± 0.0005) | 0.213 (± 0.0059) | 0.014 (± 0.0004) | 0.187 (± 0.0059) | 0.011 (± 0.0004) |


**[Table p45.3]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0217) | 0.028 (± 0.0006) | 1.0 (± 0.0239) | 0.023 (± 0.0005) | 1.0 (± 0.0239) | 0.018 (± 0.0005) |
|  | 1.086 (± 0.0207) | 0.031 (± 0.0006) | 1.209 (± 0.0264) | 0.03 (± 0.0007) | 1.344 (± 0.0247) | 0.029 (± 0.0007) |
|  | 1.196 (± 0.0201) | 0.037 (± 0.0007) | 1.512 (± 0.0223) | 0.048 (± 0.0008) | 1.879 (± 0.0245) | 0.057 (± 0.0009) |
|  | 0.662 (± 0.0464) | 0.018 (± 0.0018) | 0.733 (± 0.05) | 0.017 (± 0.0016) | 0.848 (± 0.0556) | 0.017 (± 0.0018) |
|  | 1.054 (± 0.0218) | 0.03 (± 0.0006) | 1.12 (± 0.0256) | 0.027 (± 0.0006) | 1.167 (± 0.0254) | 0.023 (± 0.0006) |
| Standard Power | 0.313 (± 0.0068) | 0.028 (± 0.0006) | 0.278 (± 0.0066) | 0.023 (± 0.0005) | 0.252 (± 0.006) | 0.018 (± 0.0005) |

[CAPTION] Table 17: Comparison of conformal outlier detection methods on TinyImageNet dataset (outliers) and CIFAR-10 dataset


<!-- page 46 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
Table 18: Comparison of conformal outlier detection methods on Texture dataset (outliers) and CIFAR-10 dataset (inliers)
for varying contamination rate r and target type-I error level α. All methods utilize the SCALE (Xu et al., 2024) method
with a pretrained ResNet-18. The empirical power is presented relative to the Standard method (higher is better). Results
are averaged across 100 random splits of the data, with standard errors presented in parentheses.
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0416)
0.009 (± 0.0003)
1.0 (± 0.0373)
0.006 (± 0.0003)
1.0 (± 0.036)
0.005 (± 0.0002)
Oracle (infeasible)
1.129 (± 0.0436)
0.01 (± 0.0003)
1.399 (± 0.0502)
0.01 (± 0.0003)
1.545 (± 0.0506)
0.01 (± 0.0004)
Naive-Trim (invalid)
1.757 (± 0.0508)
0.018 (± 0.0004)
2.978 (± 0.061)
0.029 (± 0.0006)
4.043 (± 0.0598)
0.04 (± 0.0007)
Small-Clean
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
Label-Trim
1.129 (± 0.0436)
0.01 (± 0.0003)
1.399 (± 0.0502)
0.01 (± 0.0003)
1.526 (± 0.0482)
0.009 (± 0.0003)
Standard Power
0.146 (± 0.0061)
0.009 (± 0.0003)
0.125 (± 0.0047)
0.006 (± 0.0003)
0.107 (± 0.0038)
0.005 (± 0.0002)
(a) α = 0.01
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.029)
0.018 (± 0.0004)
1.0 (± 0.0269)
0.014 (± 0.0004)
1.0 (± 0.0286)
0.011 (± 0.0003)
Oracle (infeasible)
1.089 (± 0.0286)
0.021 (± 0.0005)
1.292 (± 0.0312)
0.02 (± 0.0005)
1.518 (± 0.0278)
0.02 (± 0.0005)
Naive-Trim (invalid)
1.295 (± 0.0312)
0.027 (± 0.0006)
1.904 (± 0.0338)
0.037 (± 0.0006)
2.51 (± 0.0339)
0.048 (± 0.0008)
Small-Clean
0.792 (± 0.0585)
0.017 (± 0.0018)
0.636 (± 0.0789)
0.012 (± 0.0019)
0.288 (± 0.0601)
0.005 (± 0.0013)
Label-Trim
1.069 (± 0.0285)
0.02 (± 0.0005)
1.228 (± 0.0309)
0.018 (± 0.0005)
1.333 (± 0.0291)
0.015 (± 0.0004)
Standard Power
0.259 (± 0.0075)
0.018 (± 0.0004)
0.226 (± 0.0061)
0.014 (± 0.0004)
0.188 (± 0.0054)
0.011 (± 0.0003)
(b) α = 0.02
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.024)
0.028 (± 0.0006)
1.0 (± 0.0222)
0.022 (± 0.0005)
1.0 (± 0.0205)
0.017 (± 0.0005)
Oracle (infeasible)
1.089 (± 0.0227)
0.031 (± 0.0006)
1.228 (± 0.0248)
0.029 (± 0.0006)
1.394 (± 0.0218)
0.03 (± 0.0006)
Naive-Trim (invalid)
1.204 (± 0.023)
0.036 (± 0.0006)
1.557 (± 0.0251)
0.045 (± 0.0007)
1.919 (± 0.0242)
0.056 (± 0.0009)
Small-Clean
0.659 (± 0.0441)
0.019 (± 0.0018)
0.809 (± 0.0541)
0.019 (± 0.002)
0.837 (± 0.0579)
0.017 (± 0.002)
Label-Trim
1.057 (± 0.0231)
0.03 (± 0.0006)
1.115 (± 0.0238)
0.026 (± 0.0006)
1.173 (± 0.021)
0.023 (± 0.0005)
Standard Power
0.338 (± 0.0081)
0.028 (± 0.0006)
0.307 (± 0.0068)
0.022 (± 0.0005)
0.263 (± 0.0054)
0.017 (± 0.0005)
(c) α = 0.03
46


**[Table p46.1]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0416) | 0.009 (± 0.0003) | 1.0 (± 0.0373) | 0.006 (± 0.0003) | 1.0 (± 0.036) | 0.005 (± 0.0002) |
|  | 1.129 (± 0.0436) | 0.01 (± 0.0003) | 1.399 (± 0.0502) | 0.01 (± 0.0003) | 1.545 (± 0.0506) | 0.01 (± 0.0004) |
|  | 1.757 (± 0.0508) | 0.018 (± 0.0004) | 2.978 (± 0.061) | 0.029 (± 0.0006) | 4.043 (± 0.0598) | 0.04 (± 0.0007) |
|  | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) |
|  | 1.129 (± 0.0436) | 0.01 (± 0.0003) | 1.399 (± 0.0502) | 0.01 (± 0.0003) | 1.526 (± 0.0482) | 0.009 (± 0.0003) |
| Standard Power | 0.146 (± 0.0061) | 0.009 (± 0.0003) | 0.125 (± 0.0047) | 0.006 (± 0.0003) | 0.107 (± 0.0038) | 0.005 (± 0.0002) |


**[Table p46.2]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.029) | 0.018 (± 0.0004) | 1.0 (± 0.0269) | 0.014 (± 0.0004) | 1.0 (± 0.0286) | 0.011 (± 0.0003) |
|  | 1.089 (± 0.0286) | 0.021 (± 0.0005) | 1.292 (± 0.0312) | 0.02 (± 0.0005) | 1.518 (± 0.0278) | 0.02 (± 0.0005) |
|  | 1.295 (± 0.0312) | 0.027 (± 0.0006) | 1.904 (± 0.0338) | 0.037 (± 0.0006) | 2.51 (± 0.0339) | 0.048 (± 0.0008) |
|  | 0.792 (± 0.0585) | 0.017 (± 0.0018) | 0.636 (± 0.0789) | 0.012 (± 0.0019) | 0.288 (± 0.0601) | 0.005 (± 0.0013) |
|  | 1.069 (± 0.0285) | 0.02 (± 0.0005) | 1.228 (± 0.0309) | 0.018 (± 0.0005) | 1.333 (± 0.0291) | 0.015 (± 0.0004) |
| Standard Power | 0.259 (± 0.0075) | 0.018 (± 0.0004) | 0.226 (± 0.0061) | 0.014 (± 0.0004) | 0.188 (± 0.0054) | 0.011 (± 0.0003) |


**[Table p46.3]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.024) | 0.028 (± 0.0006) | 1.0 (± 0.0222) | 0.022 (± 0.0005) | 1.0 (± 0.0205) | 0.017 (± 0.0005) |
|  | 1.089 (± 0.0227) | 0.031 (± 0.0006) | 1.228 (± 0.0248) | 0.029 (± 0.0006) | 1.394 (± 0.0218) | 0.03 (± 0.0006) |
|  | 1.204 (± 0.023) | 0.036 (± 0.0006) | 1.557 (± 0.0251) | 0.045 (± 0.0007) | 1.919 (± 0.0242) | 0.056 (± 0.0009) |
|  | 0.659 (± 0.0441) | 0.019 (± 0.0018) | 0.809 (± 0.0541) | 0.019 (± 0.002) | 0.837 (± 0.0579) | 0.017 (± 0.002) |
|  | 1.057 (± 0.0231) | 0.03 (± 0.0006) | 1.115 (± 0.0238) | 0.026 (± 0.0006) | 1.173 (± 0.021) | 0.023 (± 0.0005) |
| Standard Power | 0.338 (± 0.0081) | 0.028 (± 0.0006) | 0.307 (± 0.0068) | 0.022 (± 0.0005) | 0.263 (± 0.0054) | 0.017 (± 0.0005) |

[CAPTION] Table 18: Comparison of conformal outlier detection methods on Texture dataset (outliers) and CIFAR-10 dataset (inliers)


<!-- page 47 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
Table 19: Comparison of conformal outlier detection methods on SVHN dataset (outliers) and CIFAR-10 dataset (inliers)
for varying contamination rate r and target type-I error level α. All methods utilize the SCALE (Xu et al., 2024) method
with a pretrained ResNet-18. The empirical power is presented relative to the Standard method (higher is better). Results
are averaged across 100 random splits of the data, with standard errors presented in parentheses.
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0308)
0.009 (± 0.0003)
1.0 (± 0.0386)
0.006 (± 0.0003)
1.0 (± 0.0401)
0.004 (± 0.0002)
Oracle (infeasible)
1.144 (± 0.0321)
0.01 (± 0.0003)
1.526 (± 0.0466)
0.01 (± 0.0003)
1.868 (± 0.058)
0.01 (± 0.0004)
Naive-Trim (invalid)
1.743 (± 0.033)
0.017 (± 0.0004)
2.838 (± 0.0519)
0.027 (± 0.0005)
4.211 (± 0.0669)
0.038 (± 0.0007)
Small-Clean
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
Label-Trim
1.144 (± 0.0321)
0.01 (± 0.0003)
1.513 (± 0.0462)
0.01 (± 0.0003)
1.749 (± 0.0529)
0.009 (± 0.0003)
Standard Power
0.191 (± 0.0059)
0.009 (± 0.0003)
0.143 (± 0.0055)
0.006 (± 0.0003)
0.115 (± 0.0046)
0.004 (± 0.0002)
(a) α = 0.01
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0189)
0.017 (± 0.0004)
1.0 (± 0.027)
0.012 (± 0.0003)
1.0 (± 0.0299)
0.01 (± 0.0003)
Oracle (infeasible)
1.082 (± 0.0204)
0.021 (± 0.0005)
1.342 (± 0.0262)
0.02 (± 0.0005)
1.642 (± 0.0331)
0.02 (± 0.0005)
Naive-Trim (invalid)
1.214 (± 0.021)
0.027 (± 0.0006)
1.858 (± 0.0271)
0.036 (± 0.0006)
2.462 (± 0.0359)
0.046 (± 0.0008)
Small-Clean
0.788 (± 0.0566)
0.018 (± 0.002)
0.662 (± 0.0764)
0.011 (± 0.0015)
0.46 (± 0.0819)
0.007 (± 0.0016)
Label-Trim
1.068 (± 0.0207)
0.02 (± 0.0005)
1.253 (± 0.0275)
0.017 (± 0.0005)
1.403 (± 0.0332)
0.014 (± 0.0004)
Standard Power
0.336 (± 0.0064)
0.017 (± 0.0004)
0.256 (± 0.0069)
0.012 (± 0.0003)
0.216 (± 0.0064)
0.01 (± 0.0003)
(b) α = 0.02
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0173)
0.027 (± 0.0006)
1.0 (± 0.0196)
0.02 (± 0.0005)
1.0 (± 0.0229)
0.015 (± 0.0004)
Oracle (infeasible)
1.075 (± 0.017)
0.031 (± 0.0006)
1.247 (± 0.0215)
0.029 (± 0.0006)
1.39 (± 0.0246)
0.03 (± 0.0006)
Naive-Trim (invalid)
1.162 (± 0.0163)
0.036 (± 0.0006)
1.519 (± 0.0195)
0.044 (± 0.0007)
1.805 (± 0.0245)
0.054 (± 0.0008)
Small-Clean
0.686 (± 0.0437)
0.019 (± 0.002)
0.857 (± 0.0528)
0.019 (± 0.0017)
0.913 (± 0.0586)
0.018 (± 0.0018)
Label-Trim
1.046 (± 0.0176)
0.029 (± 0.0006)
1.122 (± 0.0212)
0.025 (± 0.0005)
1.179 (± 0.0225)
0.022 (± 0.0005)
Standard Power
0.411 (± 0.0071)
0.027 (± 0.0006)
0.342 (± 0.0067)
0.02 (± 0.0005)
0.311 (± 0.0071)
0.015 (± 0.0004)
(c) α = 0.03
47


**[Table p47.1]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0308) | 0.009 (± 0.0003) | 1.0 (± 0.0386) | 0.006 (± 0.0003) | 1.0 (± 0.0401) | 0.004 (± 0.0002) |
|  | 1.144 (± 0.0321) | 0.01 (± 0.0003) | 1.526 (± 0.0466) | 0.01 (± 0.0003) | 1.868 (± 0.058) | 0.01 (± 0.0004) |
|  | 1.743 (± 0.033) | 0.017 (± 0.0004) | 2.838 (± 0.0519) | 0.027 (± 0.0005) | 4.211 (± 0.0669) | 0.038 (± 0.0007) |
|  | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) |
|  | 1.144 (± 0.0321) | 0.01 (± 0.0003) | 1.513 (± 0.0462) | 0.01 (± 0.0003) | 1.749 (± 0.0529) | 0.009 (± 0.0003) |
| Standard Power | 0.191 (± 0.0059) | 0.009 (± 0.0003) | 0.143 (± 0.0055) | 0.006 (± 0.0003) | 0.115 (± 0.0046) | 0.004 (± 0.0002) |


**[Table p47.2]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0189) | 0.017 (± 0.0004) | 1.0 (± 0.027) | 0.012 (± 0.0003) | 1.0 (± 0.0299) | 0.01 (± 0.0003) |
|  | 1.082 (± 0.0204) | 0.021 (± 0.0005) | 1.342 (± 0.0262) | 0.02 (± 0.0005) | 1.642 (± 0.0331) | 0.02 (± 0.0005) |
|  | 1.214 (± 0.021) | 0.027 (± 0.0006) | 1.858 (± 0.0271) | 0.036 (± 0.0006) | 2.462 (± 0.0359) | 0.046 (± 0.0008) |
|  | 0.788 (± 0.0566) | 0.018 (± 0.002) | 0.662 (± 0.0764) | 0.011 (± 0.0015) | 0.46 (± 0.0819) | 0.007 (± 0.0016) |
|  | 1.068 (± 0.0207) | 0.02 (± 0.0005) | 1.253 (± 0.0275) | 0.017 (± 0.0005) | 1.403 (± 0.0332) | 0.014 (± 0.0004) |
| Standard Power | 0.336 (± 0.0064) | 0.017 (± 0.0004) | 0.256 (± 0.0069) | 0.012 (± 0.0003) | 0.216 (± 0.0064) | 0.01 (± 0.0003) |


**[Table p47.3]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0173) | 0.027 (± 0.0006) | 1.0 (± 0.0196) | 0.02 (± 0.0005) | 1.0 (± 0.0229) | 0.015 (± 0.0004) |
|  | 1.075 (± 0.017) | 0.031 (± 0.0006) | 1.247 (± 0.0215) | 0.029 (± 0.0006) | 1.39 (± 0.0246) | 0.03 (± 0.0006) |
|  | 1.162 (± 0.0163) | 0.036 (± 0.0006) | 1.519 (± 0.0195) | 0.044 (± 0.0007) | 1.805 (± 0.0245) | 0.054 (± 0.0008) |
|  | 0.686 (± 0.0437) | 0.019 (± 0.002) | 0.857 (± 0.0528) | 0.019 (± 0.0017) | 0.913 (± 0.0586) | 0.018 (± 0.0018) |
|  | 1.046 (± 0.0176) | 0.029 (± 0.0006) | 1.122 (± 0.0212) | 0.025 (± 0.0005) | 1.179 (± 0.0225) | 0.022 (± 0.0005) |
| Standard Power | 0.411 (± 0.0071) | 0.027 (± 0.0006) | 0.342 (± 0.0067) | 0.02 (± 0.0005) | 0.311 (± 0.0071) | 0.015 (± 0.0004) |

[CAPTION] Table 19: Comparison of conformal outlier detection methods on SVHN dataset (outliers) and CIFAR-10 dataset (inliers)


<!-- page 48 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
Table 20: Comparison of conformal outlier detection methods on Places365 dataset (outliers) and CIFAR-10 dataset (inliers)
for varying contamination rate r and target type-I error level α. All methods utilize the SCALE (Xu et al., 2024) method
with a pretrained ResNet-18. The empirical power is presented relative to the Standard method (higher is better). Results
are averaged across 100 random splits of the data, with standard errors presented in parentheses.
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0354)
0.009 (± 0.0003)
1.0 (± 0.0381)
0.006 (± 0.0003)
1.0 (± 0.0462)
0.004 (± 0.0002)
Oracle (infeasible)
1.164 (± 0.0361)
0.01 (± 0.0003)
1.489 (± 0.0456)
0.01 (± 0.0003)
1.773 (± 0.0555)
0.01 (± 0.0004)
Naive-Trim (invalid)
1.729 (± 0.0387)
0.017 (± 0.0004)
2.944 (± 0.0532)
0.028 (± 0.0006)
4.136 (± 0.0545)
0.038 (± 0.0007)
Small-Clean
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
Label-Trim
1.164 (± 0.0361)
0.01 (± 0.0003)
1.483 (± 0.0452)
0.01 (± 0.0003)
1.704 (± 0.0533)
0.009 (± 0.0003)
Standard Power
0.174 (± 0.0061)
0.009 (± 0.0003)
0.133 (± 0.0051)
0.006 (± 0.0003)
0.112 (± 0.0052)
0.004 (± 0.0002)
(a) α = 0.01
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0221)
0.018 (± 0.0004)
1.0 (± 0.0258)
0.013 (± 0.0004)
1.0 (± 0.0296)
0.01 (± 0.0003)
Oracle (infeasible)
1.068 (± 0.0237)
0.021 (± 0.0005)
1.291 (± 0.0272)
0.02 (± 0.0005)
1.564 (± 0.0325)
0.02 (± 0.0005)
Naive-Trim (invalid)
1.233 (± 0.0249)
0.027 (± 0.0006)
1.782 (± 0.0278)
0.036 (± 0.0006)
2.483 (± 0.0308)
0.046 (± 0.0008)
Small-Clean
0.823 (± 0.0588)
0.018 (± 0.0019)
0.595 (± 0.074)
0.011 (± 0.0016)
0.331 (± 0.0746)
0.005 (± 0.0014)
Label-Trim
1.058 (± 0.0236)
0.02 (± 0.0005)
1.212 (± 0.0272)
0.018 (± 0.0005)
1.357 (± 0.0312)
0.015 (± 0.0004)
Standard Power
0.302 (± 0.0067)
0.018 (± 0.0004)
0.25 (± 0.0065)
0.013 (± 0.0004)
0.207 (± 0.0061)
0.01 (± 0.0003)
(b) α = 0.02
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0201)
0.028 (± 0.0006)
1.0 (± 0.0203)
0.021 (± 0.0005)
1.0 (± 0.0225)
0.016 (± 0.0004)
Oracle (infeasible)
1.084 (± 0.0218)
0.031 (± 0.0006)
1.229 (± 0.0211)
0.029 (± 0.0006)
1.383 (± 0.0228)
0.03 (± 0.0006)
Naive-Trim (invalid)
1.186 (± 0.0217)
0.036 (± 0.0006)
1.534 (± 0.0216)
0.044 (± 0.0007)
1.866 (± 0.0213)
0.055 (± 0.0009)
Small-Clean
0.712 (± 0.0449)
0.019 (± 0.0018)
0.832 (± 0.0525)
0.019 (± 0.0018)
1.025 (± 0.0644)
0.023 (± 0.0022)
Label-Trim
1.044 (± 0.0206)
0.029 (± 0.0006)
1.121 (± 0.0208)
0.026 (± 0.0005)
1.167 (± 0.0219)
0.022 (± 0.0005)
Standard Power
0.374 (± 0.0075)
0.028 (± 0.0006)
0.327 (± 0.0066)
0.021 (± 0.0005)
0.291 (± 0.0065)
0.016 (± 0.0004)
(c) α = 0.03
48


**[Table p48.1]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0354) | 0.009 (± 0.0003) | 1.0 (± 0.0381) | 0.006 (± 0.0003) | 1.0 (± 0.0462) | 0.004 (± 0.0002) |
|  | 1.164 (± 0.0361) | 0.01 (± 0.0003) | 1.489 (± 0.0456) | 0.01 (± 0.0003) | 1.773 (± 0.0555) | 0.01 (± 0.0004) |
|  | 1.729 (± 0.0387) | 0.017 (± 0.0004) | 2.944 (± 0.0532) | 0.028 (± 0.0006) | 4.136 (± 0.0545) | 0.038 (± 0.0007) |
|  | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) |
|  | 1.164 (± 0.0361) | 0.01 (± 0.0003) | 1.483 (± 0.0452) | 0.01 (± 0.0003) | 1.704 (± 0.0533) | 0.009 (± 0.0003) |
| Standard Power | 0.174 (± 0.0061) | 0.009 (± 0.0003) | 0.133 (± 0.0051) | 0.006 (± 0.0003) | 0.112 (± 0.0052) | 0.004 (± 0.0002) |


**[Table p48.2]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0221) | 0.018 (± 0.0004) | 1.0 (± 0.0258) | 0.013 (± 0.0004) | 1.0 (± 0.0296) | 0.01 (± 0.0003) |
|  | 1.068 (± 0.0237) | 0.021 (± 0.0005) | 1.291 (± 0.0272) | 0.02 (± 0.0005) | 1.564 (± 0.0325) | 0.02 (± 0.0005) |
|  | 1.233 (± 0.0249) | 0.027 (± 0.0006) | 1.782 (± 0.0278) | 0.036 (± 0.0006) | 2.483 (± 0.0308) | 0.046 (± 0.0008) |
|  | 0.823 (± 0.0588) | 0.018 (± 0.0019) | 0.595 (± 0.074) | 0.011 (± 0.0016) | 0.331 (± 0.0746) | 0.005 (± 0.0014) |
|  | 1.058 (± 0.0236) | 0.02 (± 0.0005) | 1.212 (± 0.0272) | 0.018 (± 0.0005) | 1.357 (± 0.0312) | 0.015 (± 0.0004) |
| Standard Power | 0.302 (± 0.0067) | 0.018 (± 0.0004) | 0.25 (± 0.0065) | 0.013 (± 0.0004) | 0.207 (± 0.0061) | 0.01 (± 0.0003) |


**[Table p48.3]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0201) | 0.028 (± 0.0006) | 1.0 (± 0.0203) | 0.021 (± 0.0005) | 1.0 (± 0.0225) | 0.016 (± 0.0004) |
|  | 1.084 (± 0.0218) | 0.031 (± 0.0006) | 1.229 (± 0.0211) | 0.029 (± 0.0006) | 1.383 (± 0.0228) | 0.03 (± 0.0006) |
|  | 1.186 (± 0.0217) | 0.036 (± 0.0006) | 1.534 (± 0.0216) | 0.044 (± 0.0007) | 1.866 (± 0.0213) | 0.055 (± 0.0009) |
|  | 0.712 (± 0.0449) | 0.019 (± 0.0018) | 0.832 (± 0.0525) | 0.019 (± 0.0018) | 1.025 (± 0.0644) | 0.023 (± 0.0022) |
|  | 1.044 (± 0.0206) | 0.029 (± 0.0006) | 1.121 (± 0.0208) | 0.026 (± 0.0005) | 1.167 (± 0.0219) | 0.022 (± 0.0005) |
| Standard Power | 0.374 (± 0.0075) | 0.028 (± 0.0006) | 0.327 (± 0.0066) | 0.021 (± 0.0005) | 0.291 (± 0.0065) | 0.016 (± 0.0004) |

[CAPTION] Table 20: Comparison of conformal outlier detection methods on Places365 dataset (outliers) and CIFAR-10 dataset (inliers)


<!-- page 49 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
Table 21: Comparison of conformal outlier detection methods on MNIST dataset (outliers) and CIFAR-10 dataset (inliers)
for varying contamination rate r and target type-I error level α. All methods utilize the SCALE (Xu et al., 2024) method
with a pretrained ResNet-18. The empirical power is presented relative to the Standard method (higher is better). Results
are averaged across 100 random splits of the data, with standard errors presented in parentheses.
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0286)
0.008 (± 0.0003)
1.0 (± 0.0284)
0.004 (± 0.0002)
1.0 (± 0.0366)
0.003 (± 0.0002)
Oracle (infeasible)
1.174 (± 0.0308)
0.01 (± 0.0003)
1.625 (± 0.0389)
0.01 (± 0.0003)
1.961 (± 0.045)
0.01 (± 0.0004)
Naive-Trim (invalid)
1.641 (± 0.0326)
0.016 (± 0.0004)
2.781 (± 0.0382)
0.025 (± 0.0006)
3.78 (± 0.0441)
0.032 (± 0.0006)
Small-Clean
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
Label-Trim
1.174 (± 0.0308)
0.01 (± 0.0003)
1.555 (± 0.0366)
0.01 (± 0.0003)
1.701 (± 0.041)
0.008 (± 0.0003)
Standard Power
0.251 (± 0.0072)
0.008 (± 0.0003)
0.181 (± 0.0051)
0.004 (± 0.0002)
0.152 (± 0.0056)
0.003 (± 0.0002)
(a) α = 0.01
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0195)
0.017 (± 0.0004)
1.0 (± 0.0216)
0.011 (± 0.0003)
1.0 (± 0.0244)
0.008 (± 0.0003)
Oracle (infeasible)
1.095 (± 0.0198)
0.021 (± 0.0005)
1.416 (± 0.0218)
0.02 (± 0.0005)
1.776 (± 0.0279)
0.02 (± 0.0005)
Naive-Trim (invalid)
1.212 (± 0.0195)
0.026 (± 0.0006)
1.84 (± 0.0223)
0.032 (± 0.0006)
2.456 (± 0.0243)
0.04 (± 0.0007)
Small-Clean
0.849 (± 0.0589)
0.02 (± 0.0021)
0.605 (± 0.0715)
0.009 (± 0.0015)
0.366 (± 0.0772)
0.005 (± 0.0014)
Label-Trim
1.076 (± 0.0197)
0.02 (± 0.0005)
1.3 (± 0.0227)
0.016 (± 0.0005)
1.424 (± 0.0272)
0.013 (± 0.0004)
Standard Power
0.415 (± 0.0081)
0.017 (± 0.0004)
0.317 (± 0.0069)
0.011 (± 0.0003)
0.257 (± 0.0063)
0.008 (± 0.0003)
(b) α = 0.02
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0161)
0.026 (± 0.0006)
1.0 (± 0.0164)
0.018 (± 0.0005)
1.0 (± 0.0191)
0.012 (± 0.0004)
Oracle (infeasible)
1.086 (± 0.0154)
0.031 (± 0.0006)
1.272 (± 0.0167)
0.029 (± 0.0006)
1.518 (± 0.02)
0.03 (± 0.0006)
Naive-Trim (invalid)
1.161 (± 0.0144)
0.035 (± 0.0006)
1.483 (± 0.0168)
0.041 (± 0.0007)
1.852 (± 0.0172)
0.048 (± 0.0008)
Small-Clean
0.778 (± 0.0434)
0.022 (± 0.002)
0.848 (± 0.0471)
0.018 (± 0.0018)
1.075 (± 0.0562)
0.02 (± 0.0021)
Label-Trim
1.056 (± 0.0161)
0.029 (± 0.0006)
1.138 (± 0.0162)
0.024 (± 0.0005)
1.237 (± 0.0195)
0.019 (± 0.0005)
Standard Power
0.505 (± 0.0081)
0.026 (± 0.0006)
0.43 (± 0.007)
0.018 (± 0.0005)
0.362 (± 0.0069)
0.012 (± 0.0004)
(c) α = 0.03
49


**[Table p49.1]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0286) | 0.008 (± 0.0003) | 1.0 (± 0.0284) | 0.004 (± 0.0002) | 1.0 (± 0.0366) | 0.003 (± 0.0002) |
|  | 1.174 (± 0.0308) | 0.01 (± 0.0003) | 1.625 (± 0.0389) | 0.01 (± 0.0003) | 1.961 (± 0.045) | 0.01 (± 0.0004) |
|  | 1.641 (± 0.0326) | 0.016 (± 0.0004) | 2.781 (± 0.0382) | 0.025 (± 0.0006) | 3.78 (± 0.0441) | 0.032 (± 0.0006) |
|  | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) |
|  | 1.174 (± 0.0308) | 0.01 (± 0.0003) | 1.555 (± 0.0366) | 0.01 (± 0.0003) | 1.701 (± 0.041) | 0.008 (± 0.0003) |
| Standard Power | 0.251 (± 0.0072) | 0.008 (± 0.0003) | 0.181 (± 0.0051) | 0.004 (± 0.0002) | 0.152 (± 0.0056) | 0.003 (± 0.0002) |


**[Table p49.2]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0195) | 0.017 (± 0.0004) | 1.0 (± 0.0216) | 0.011 (± 0.0003) | 1.0 (± 0.0244) | 0.008 (± 0.0003) |
|  | 1.095 (± 0.0198) | 0.021 (± 0.0005) | 1.416 (± 0.0218) | 0.02 (± 0.0005) | 1.776 (± 0.0279) | 0.02 (± 0.0005) |
|  | 1.212 (± 0.0195) | 0.026 (± 0.0006) | 1.84 (± 0.0223) | 0.032 (± 0.0006) | 2.456 (± 0.0243) | 0.04 (± 0.0007) |
|  | 0.849 (± 0.0589) | 0.02 (± 0.0021) | 0.605 (± 0.0715) | 0.009 (± 0.0015) | 0.366 (± 0.0772) | 0.005 (± 0.0014) |
|  | 1.076 (± 0.0197) | 0.02 (± 0.0005) | 1.3 (± 0.0227) | 0.016 (± 0.0005) | 1.424 (± 0.0272) | 0.013 (± 0.0004) |
| Standard Power | 0.415 (± 0.0081) | 0.017 (± 0.0004) | 0.317 (± 0.0069) | 0.011 (± 0.0003) | 0.257 (± 0.0063) | 0.008 (± 0.0003) |


**[Table p49.3]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0161) | 0.026 (± 0.0006) | 1.0 (± 0.0164) | 0.018 (± 0.0005) | 1.0 (± 0.0191) | 0.012 (± 0.0004) |
|  | 1.086 (± 0.0154) | 0.031 (± 0.0006) | 1.272 (± 0.0167) | 0.029 (± 0.0006) | 1.518 (± 0.02) | 0.03 (± 0.0006) |
|  | 1.161 (± 0.0144) | 0.035 (± 0.0006) | 1.483 (± 0.0168) | 0.041 (± 0.0007) | 1.852 (± 0.0172) | 0.048 (± 0.0008) |
|  | 0.778 (± 0.0434) | 0.022 (± 0.002) | 0.848 (± 0.0471) | 0.018 (± 0.0018) | 1.075 (± 0.0562) | 0.02 (± 0.0021) |
|  | 1.056 (± 0.0161) | 0.029 (± 0.0006) | 1.138 (± 0.0162) | 0.024 (± 0.0005) | 1.237 (± 0.0195) | 0.019 (± 0.0005) |
| Standard Power | 0.505 (± 0.0081) | 0.026 (± 0.0006) | 0.43 (± 0.007) | 0.018 (± 0.0005) | 0.362 (± 0.0069) | 0.012 (± 0.0004) |

[CAPTION] Table 21: Comparison of conformal outlier detection methods on MNIST dataset (outliers) and CIFAR-10 dataset (inliers)


<!-- page 50 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
Table 22: Comparison of conformal outlier detection methods on CIFAR-100 dataset (outliers) and CIFAR-10 dataset
(inliers) for varying contamination rate r and target type-I error level α. All methods utilize the SCALE (Xu et al., 2024)
method with a pretrained ResNet-18. The empirical power is presented relative to the Standard method (higher is better).
Results are averaged across 100 random splits of the data, with standard errors presented in parentheses.
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0399)
0.009 (± 0.0003)
1.0 (± 0.0432)
0.007 (± 0.0003)
1.0 (± 0.047)
0.006 (± 0.0003)
Oracle (infeasible)
1.113 (± 0.0396)
0.01 (± 0.0003)
1.355 (± 0.0496)
0.01 (± 0.0003)
1.478 (± 0.0566)
0.01 (± 0.0004)
Naive-Trim (invalid)
1.848 (± 0.0507)
0.018 (± 0.0005)
3.161 (± 0.0652)
0.03 (± 0.0006)
4.447 (± 0.0722)
0.042 (± 0.0007)
Small-Clean
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
Label-Trim
1.116 (± 0.04)
0.01 (± 0.0003)
1.355 (± 0.0496)
0.01 (± 0.0003)
1.491 (± 0.0582)
0.01 (± 0.0004)
Standard Power
0.124 (± 0.0049)
0.009 (± 0.0003)
0.103 (± 0.0045)
0.007 (± 0.0003)
0.091 (± 0.0043)
0.006 (± 0.0003)
(a) α = 0.01
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.027)
0.019 (± 0.0005)
1.0 (± 0.027)
0.014 (± 0.0004)
1.0 (± 0.0329)
0.012 (± 0.0004)
Oracle (infeasible)
1.057 (± 0.0281)
0.021 (± 0.0005)
1.24 (± 0.0302)
0.02 (± 0.0005)
1.507 (± 0.0364)
0.02 (± 0.0005)
Naive-Trim (invalid)
1.305 (± 0.0289)
0.028 (± 0.0006)
1.957 (± 0.0336)
0.038 (± 0.0006)
2.747 (± 0.0395)
0.05 (± 0.0008)
Small-Clean
0.932 (± 0.0653)
0.02 (± 0.0018)
0.533 (± 0.0766)
0.01 (± 0.0017)
0.38 (± 0.0889)
0.007 (± 0.0018)
Label-Trim
1.046 (± 0.0276)
0.021 (± 0.0005)
1.178 (± 0.0299)
0.018 (± 0.0005)
1.344 (± 0.0349)
0.016 (± 0.0005)
Standard Power
0.232 (± 0.0063)
0.019 (± 0.0005)
0.197 (± 0.0053)
0.014 (± 0.0004)
0.163 (± 0.0053)
0.012 (± 0.0004)
(b) α = 0.02
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0218)
0.028 (± 0.0006)
1.0 (± 0.0244)
0.023 (± 0.0005)
1.0 (± 0.0251)
0.018 (± 0.0005)
Oracle (infeasible)
1.088 (± 0.0232)
0.031 (± 0.0006)
1.231 (± 0.0255)
0.029 (± 0.0006)
1.376 (± 0.0272)
0.03 (± 0.0006)
Naive-Trim (invalid)
1.224 (± 0.0223)
0.037 (± 0.0007)
1.658 (± 0.0276)
0.047 (± 0.0007)
2.043 (± 0.0274)
0.058 (± 0.0009)
Small-Clean
0.758 (± 0.0461)
0.021 (± 0.0017)
0.897 (± 0.0659)
0.023 (± 0.0024)
0.97 (± 0.0695)
0.021 (± 0.0022)
Label-Trim
1.051 (± 0.0231)
0.03 (± 0.0006)
1.11 (± 0.0244)
0.026 (± 0.0006)
1.173 (± 0.0261)
0.024 (± 0.0005)
Standard Power
0.304 (± 0.0066)
0.028 (± 0.0006)
0.261 (± 0.0064)
0.023 (± 0.0005)
0.237 (± 0.0059)
0.018 (± 0.0005)
(c) α = 0.03
50


**[Table p50.1]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0399) | 0.009 (± 0.0003) | 1.0 (± 0.0432) | 0.007 (± 0.0003) | 1.0 (± 0.047) | 0.006 (± 0.0003) |
|  | 1.113 (± 0.0396) | 0.01 (± 0.0003) | 1.355 (± 0.0496) | 0.01 (± 0.0003) | 1.478 (± 0.0566) | 0.01 (± 0.0004) |
|  | 1.848 (± 0.0507) | 0.018 (± 0.0005) | 3.161 (± 0.0652) | 0.03 (± 0.0006) | 4.447 (± 0.0722) | 0.042 (± 0.0007) |
|  | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) |
|  | 1.116 (± 0.04) | 0.01 (± 0.0003) | 1.355 (± 0.0496) | 0.01 (± 0.0003) | 1.491 (± 0.0582) | 0.01 (± 0.0004) |
| Standard Power | 0.124 (± 0.0049) | 0.009 (± 0.0003) | 0.103 (± 0.0045) | 0.007 (± 0.0003) | 0.091 (± 0.0043) | 0.006 (± 0.0003) |


**[Table p50.2]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.027) | 0.019 (± 0.0005) | 1.0 (± 0.027) | 0.014 (± 0.0004) | 1.0 (± 0.0329) | 0.012 (± 0.0004) |
|  | 1.057 (± 0.0281) | 0.021 (± 0.0005) | 1.24 (± 0.0302) | 0.02 (± 0.0005) | 1.507 (± 0.0364) | 0.02 (± 0.0005) |
|  | 1.305 (± 0.0289) | 0.028 (± 0.0006) | 1.957 (± 0.0336) | 0.038 (± 0.0006) | 2.747 (± 0.0395) | 0.05 (± 0.0008) |
|  | 0.932 (± 0.0653) | 0.02 (± 0.0018) | 0.533 (± 0.0766) | 0.01 (± 0.0017) | 0.38 (± 0.0889) | 0.007 (± 0.0018) |
|  | 1.046 (± 0.0276) | 0.021 (± 0.0005) | 1.178 (± 0.0299) | 0.018 (± 0.0005) | 1.344 (± 0.0349) | 0.016 (± 0.0005) |
| Standard Power | 0.232 (± 0.0063) | 0.019 (± 0.0005) | 0.197 (± 0.0053) | 0.014 (± 0.0004) | 0.163 (± 0.0053) | 0.012 (± 0.0004) |


**[Table p50.3]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0218) | 0.028 (± 0.0006) | 1.0 (± 0.0244) | 0.023 (± 0.0005) | 1.0 (± 0.0251) | 0.018 (± 0.0005) |
|  | 1.088 (± 0.0232) | 0.031 (± 0.0006) | 1.231 (± 0.0255) | 0.029 (± 0.0006) | 1.376 (± 0.0272) | 0.03 (± 0.0006) |
|  | 1.224 (± 0.0223) | 0.037 (± 0.0007) | 1.658 (± 0.0276) | 0.047 (± 0.0007) | 2.043 (± 0.0274) | 0.058 (± 0.0009) |
|  | 0.758 (± 0.0461) | 0.021 (± 0.0017) | 0.897 (± 0.0659) | 0.023 (± 0.0024) | 0.97 (± 0.0695) | 0.021 (± 0.0022) |
|  | 1.051 (± 0.0231) | 0.03 (± 0.0006) | 1.11 (± 0.0244) | 0.026 (± 0.0006) | 1.173 (± 0.0261) | 0.024 (± 0.0005) |
| Standard Power | 0.304 (± 0.0066) | 0.028 (± 0.0006) | 0.261 (± 0.0064) | 0.023 (± 0.0005) | 0.237 (± 0.0059) | 0.018 (± 0.0005) |

[CAPTION] Table 22: Comparison of conformal outlier detection methods on CIFAR-100 dataset (outliers) and CIFAR-10 dataset


<!-- page 51 -->
Robust Conformal Outlier Detection under Contaminated Reference Data
Table 23: Comparison of conformal outlier detection methods on TinyImageNet dataset (outliers) and CIFAR-10 dataset
(inliers) for varying contamination rate r and target type-I error level α. All methods utilize the SCALE (Xu et al., 2024)
method with a pretrained ResNet-18. The empirical power is presented relative to the Standard method (higher is better).
Results are averaged across 100 random splits of the data, with standard errors presented in parentheses.
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0383)
0.009 (± 0.0003)
1.0 (± 0.0394)
0.006 (± 0.0003)
1.0 (± 0.0383)
0.005 (± 0.0002)
Oracle (infeasible)
1.138 (± 0.0399)
0.01 (± 0.0003)
1.436 (± 0.0534)
0.01 (± 0.0003)
1.625 (± 0.0475)
0.01 (± 0.0004)
Naive-Trim (invalid)
1.696 (± 0.0424)
0.018 (± 0.0004)
2.88 (± 0.0621)
0.029 (± 0.0006)
4.248 (± 0.0707)
0.041 (± 0.0007)
Small-Clean
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
0.0 (± 0.0)
Label-Trim
1.138 (± 0.0399)
0.01 (± 0.0003)
1.433 (± 0.0535)
0.01 (± 0.0003)
1.591 (± 0.0467)
0.009 (± 0.0004)
Standard Power
0.159 (± 0.0061)
0.009 (± 0.0003)
0.122 (± 0.0048)
0.006 (± 0.0003)
0.102 (± 0.0039)
0.005 (± 0.0002)
(a) α = 0.01
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0248)
0.018 (± 0.0004)
1.0 (± 0.0317)
0.014 (± 0.0004)
1.0 (± 0.0271)
0.011 (± 0.0004)
Oracle (infeasible)
1.063 (± 0.0255)
0.021 (± 0.0005)
1.254 (± 0.0309)
0.02 (± 0.0005)
1.49 (± 0.0318)
0.02 (± 0.0005)
Naive-Trim (invalid)
1.223 (± 0.0263)
0.028 (± 0.0006)
1.816 (± 0.0321)
0.037 (± 0.0006)
2.536 (± 0.0365)
0.049 (± 0.0008)
Small-Clean
0.803 (± 0.0609)
0.017 (± 0.0018)
0.625 (± 0.0816)
0.013 (± 0.0022)
0.263 (± 0.064)
0.003 (± 0.001)
Label-Trim
1.053 (± 0.0255)
0.02 (± 0.0005)
1.183 (± 0.0325)
0.018 (± 0.0005)
1.344 (± 0.0334)
0.016 (± 0.0004)
Standard Power
0.273 (± 0.0068)
0.018 (± 0.0004)
0.227 (± 0.0072)
0.014 (± 0.0004)
0.187 (± 0.0051)
0.011 (± 0.0004)
(b) α = 0.02
Contamination rate
1%
3%
5%
Method
Power
Type-I Error
Power
Type-I Error
Power
Type-I Error
Standard
1.0 (± 0.0215)
0.028 (± 0.0006)
1.0 (± 0.024)
0.022 (± 0.0005)
1.0 (± 0.0221)
0.017 (± 0.0005)
Oracle (infeasible)
1.07 (± 0.0222)
0.031 (± 0.0006)
1.188 (± 0.0264)
0.029 (± 0.0006)
1.328 (± 0.0256)
0.03 (± 0.0006)
Naive-Trim (invalid)
1.202 (± 0.0203)
0.037 (± 0.0006)
1.533 (± 0.0266)
0.046 (± 0.0007)
1.916 (± 0.0288)
0.057 (± 0.0009)
Small-Clean
0.678 (± 0.0476)
0.017 (± 0.0018)
0.932 (± 0.0576)
0.025 (± 0.0025)
1.005 (± 0.0573)
0.024 (± 0.0022)
Label-Trim
1.046 (± 0.0222)
0.03 (± 0.0006)
1.09 (± 0.0256)
0.026 (± 0.0005)
1.14 (± 0.0234)
0.023 (± 0.0005)
Standard Power
0.336 (± 0.0072)
0.028 (± 0.0006)
0.295 (± 0.0071)
0.022 (± 0.0005)
0.266 (± 0.0059)
0.017 (± 0.0005)
(c) α = 0.03
51


**[Table p51.1]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0383) | 0.009 (± 0.0003) | 1.0 (± 0.0394) | 0.006 (± 0.0003) | 1.0 (± 0.0383) | 0.005 (± 0.0002) |
|  | 1.138 (± 0.0399) | 0.01 (± 0.0003) | 1.436 (± 0.0534) | 0.01 (± 0.0003) | 1.625 (± 0.0475) | 0.01 (± 0.0004) |
|  | 1.696 (± 0.0424) | 0.018 (± 0.0004) | 2.88 (± 0.0621) | 0.029 (± 0.0006) | 4.248 (± 0.0707) | 0.041 (± 0.0007) |
|  | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) | 0.0 (± 0.0) |
|  | 1.138 (± 0.0399) | 0.01 (± 0.0003) | 1.433 (± 0.0535) | 0.01 (± 0.0003) | 1.591 (± 0.0467) | 0.009 (± 0.0004) |
| Standard Power | 0.159 (± 0.0061) | 0.009 (± 0.0003) | 0.122 (± 0.0048) | 0.006 (± 0.0003) | 0.102 (± 0.0039) | 0.005 (± 0.0002) |


**[Table p51.2]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0248) | 0.018 (± 0.0004) | 1.0 (± 0.0317) | 0.014 (± 0.0004) | 1.0 (± 0.0271) | 0.011 (± 0.0004) |
|  | 1.063 (± 0.0255) | 0.021 (± 0.0005) | 1.254 (± 0.0309) | 0.02 (± 0.0005) | 1.49 (± 0.0318) | 0.02 (± 0.0005) |
|  | 1.223 (± 0.0263) | 0.028 (± 0.0006) | 1.816 (± 0.0321) | 0.037 (± 0.0006) | 2.536 (± 0.0365) | 0.049 (± 0.0008) |
|  | 0.803 (± 0.0609) | 0.017 (± 0.0018) | 0.625 (± 0.0816) | 0.013 (± 0.0022) | 0.263 (± 0.064) | 0.003 (± 0.001) |
|  | 1.053 (± 0.0255) | 0.02 (± 0.0005) | 1.183 (± 0.0325) | 0.018 (± 0.0005) | 1.344 (± 0.0334) | 0.016 (± 0.0004) |
| Standard Power | 0.273 (± 0.0068) | 0.018 (± 0.0004) | 0.227 (± 0.0072) | 0.014 (± 0.0004) | 0.187 (± 0.0051) | 0.011 (± 0.0004) |


**[Table p51.3]**
|  | Contamination rate |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | 1% |  | 3% |  | 5% |  |
| Method | Power Type-I Error |  | Power Type-I Error |  | Power Type-I Error |  |
| Standard Oracle (infeasible) Naive-Trim (invalid) Small-Clean Label-Trim | 1.0 (± 0.0215) | 0.028 (± 0.0006) | 1.0 (± 0.024) | 0.022 (± 0.0005) | 1.0 (± 0.0221) | 0.017 (± 0.0005) |
|  | 1.07 (± 0.0222) | 0.031 (± 0.0006) | 1.188 (± 0.0264) | 0.029 (± 0.0006) | 1.328 (± 0.0256) | 0.03 (± 0.0006) |
|  | 1.202 (± 0.0203) | 0.037 (± 0.0006) | 1.533 (± 0.0266) | 0.046 (± 0.0007) | 1.916 (± 0.0288) | 0.057 (± 0.0009) |
|  | 0.678 (± 0.0476) | 0.017 (± 0.0018) | 0.932 (± 0.0576) | 0.025 (± 0.0025) | 1.005 (± 0.0573) | 0.024 (± 0.0022) |
|  | 1.046 (± 0.0222) | 0.03 (± 0.0006) | 1.09 (± 0.0256) | 0.026 (± 0.0005) | 1.14 (± 0.0234) | 0.023 (± 0.0005) |
| Standard Power | 0.336 (± 0.0072) | 0.028 (± 0.0006) | 0.295 (± 0.0071) | 0.022 (± 0.0005) | 0.266 (± 0.0059) | 0.017 (± 0.0005) |

[CAPTION] Table 23: Comparison of conformal outlier detection methods on TinyImageNet dataset (outliers) and CIFAR-10 dataset