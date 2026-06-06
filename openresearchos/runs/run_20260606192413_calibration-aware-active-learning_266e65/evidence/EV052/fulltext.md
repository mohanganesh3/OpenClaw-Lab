<!-- page 1 -->
Unsupervised Calibration under Covariate Shift
Anusri Pampari
Computer Science Department
Stanford University
Stefano Ermon
Computer Science Department
Stanford University
Abstract
A probabilistic model is said to be calibrated
if its predicted probabilities match the corre-
sponding empirical frequencies.
Calibration
is important for uncertainty quantiﬁcation and
decision making in safety-critical applications.
While calibration of classiﬁers has been widely
studied, we ﬁnd that calibration is brittle and
can be easily lost under minimal covariate
shifts. Existing techniques, including domain
adaptation ones, primarily focus on prediction
accuracy and do not guarantee calibration nei-
ther in theory nor in practice. In this work,
we formally introduce the problem of calibra-
tion under domain shift, and propose an im-
portance sampling based approach to address
it. We evaluate and discuss the efﬁcacy of our
method on both real-world datasets and syn-
thetic datasets.
1
INTRODUCTION
Machine learning models are increasingly being en-
trusted with complex decisions in many applications
such as medical diagnosis (Triantafyllidis & Tsanas,
2019), justice system (Berk & Hyatt, 2015), ﬁnancial de-
cisions (Heaton et al., 2017), human robot interaction
(Modares et al., 2015), etc.
In all these applications,
models must not only be accurate, but should also in-
dicate conﬁdence in their own predictions. Uncertainity
quantiﬁcation is important for safety-critical applications
and in decision making. This will better inform when the
model’s predictions are likely to be incorrect, and help
in building trust with the user. For example, in medical
diagnosis if the model is not conﬁdent about it’s predic-
tion, then the decision making should be passed on to a
doctor. Additionally, humans have a natural cognitive in-
Figure 1:
Reliability diagram for a LeNet-5 model
trained using CDAN (SOTA domain adaptation tech-
nique) on MNIST and tested on USPS as target data.
tuition for probabilities (Cosmides & Tooby, 1996). Cal-
ibrated probabilities provide an intuitive explanation to a
model’s predictions, making them interpretable.
Ideally, the conﬁdence or probability associated with the
predicted class label should reﬂect its ground truth occur-
rence likelihood. For example, suppose a diabetes risk
prediction model predicts a chance of 70% for a speciﬁc
patient proﬁle. Then, we expect that out of 100 similar
patients, about 70 should have diabetes. Such a model
is said to be calibrated. Many existing machine learning
models, such as SVMs, Gaussian processes, and Neural
Networks, are not naturally calibrated (Guo et al., 2017;
Bella et al., 2010), thus producing unreliable conﬁdence
estimates. This can, in turn, lead to bad decision making
and reduce the trust in using these models.
Existing literature (Platt et al., 1999; Zadrozny & Elkan,
2001; 2002; Bella et al., 2010; Guo et al., 2017) intro-
duces many post processing techniques to correct these
arXiv:2006.16405v1  [cs.LG]  29 Jun 2020

[CAPTION] Figure 1:
Reliability diagram for a LeNet-5 model


<!-- page 2 -->
miscalibrated models. However, they assume the avail-
ability of labeled held-out validation data drawn from the
same distribution as the test data to achieve calibration.
This assumption is violated in many real world scenar-
ios in the following two ways. Firstly, the test dataset
can have a different distribution due to covariate shift.
This can happen, e.g., when the operating conditions at
test time are slightly different. Secondly, labelled test
data is often unavailable if distribution shift occurs after
training. While several unsupervised domain adaptation
methods (Chu & Wang, 2018; Kouw & Loog, 2019) pro-
pose solutions for correcting the accuracy of the models,
there is no existing work to correct the effects of these
circumstances on the conﬁdence. For example, in Fig-
ure. 1, we show how an existing calibration method tem-
perature scaling (t-scaling) (Guo et al., 2017) can fail to
calibrate a LeNet-5 model trained using CDAN (a SOTA
domain adaptation model (Long et al., 2018)) under do-
main shift. Here the model is trained on MNIST and has
a prediction accuracy of 70% on USPS dataset.
In this work, we introduce and investigate the problem
of miscalibration under covariate shift. We demonstrate
that existing models learnt using domain adaptation are
poorly calibrated, showing that while current domain
adaptation techniques account for accuracy, they do not
consider calibration of the models.
We then propose
a modiﬁcation to the calibration optimization objective
used by existing techniques. Our solution employs im-
portance sampling to account for the difference in the
training and testing distributions, thereby overcoming the
inherent assumptions of the existing methods. Our pro-
posed method can adapt any existing calibration method
under covariate shift assumption without requiring any
labeled data from the test distribution. In Figure. 1, we
show how our method (weighted t-scaling) adapts the use
of t-scaling on source validation data to work under do-
main shift. We achieve close performance to perfect cal-
ibration or calibration obtained using labeled target data.
To summarize our contributions,
• We introduce the problem of miscalibration un-
der covariate shift, and show how existing domain
adapted models such as CDAN (Long et al., 2018)
remain uncalibrated in the target domain on using
existing calibration methods;
• We propose an importance sampling based solution
to address the problem of calibration under covari-
ate shift. Our method requests no additional labels
from the test distribution and can be used to adapt
any calibration method;
• We use a discriminator trained on a domain-
invariant feature layer of source and target to get
density ratios for importance sampling.
2
RELATED WORK
Background and Notation Calibration can be described
mathematically as follows. Suppose that we have some
data, comprising of inputs X ∈Rd and labels Y
∈
1, . . . , K, which follows the ground truth joint distribu-
tion π(X, Y ) = π(Y |X)π(X). Let h(.) be a classiﬁer
learned for this data, i.e. h : X →[0, 1]K which for an
input x ∼X, is the probability distribution over the K
classes in Y . The class with the maximum probability
of occurrence is the prediction ˆY , and its corresponding
probability is the conﬁdence prediction ˆP. The classiﬁer
h is said to be calibrated (Guo et al., 2017) when,
P( ˆY = Y | ˆP = p) = p ∀p ∈[0, 1]
(1)
Many existing classiﬁers do not naturally satisfy these
requirements (Guo et al., 2017; Bella et al., 2010), and
are therefore said to be miscalibrated. This error is in-
evitable because of many reasons such as using ﬁnitely
many samples to learn the classiﬁer h, model mis-
match from the true distribution, optimization issues,
etc. Existing calibration techniques (Platt et al., 1999;
Zadrozny & Elkan, 2001; 2002; Bella et al., 2010; Guo
et al., 2017) reduce this error in post-processing steps
to produce calibrated probabilities. A calibration model
(parametrized by β) is applied over the uncalibrated clas-
siﬁer. Each method deﬁnes an approximate variant of
the calibration error using a loss function of the form
E(x,y)∼π(X,Y ) l(h(x), y; β) and learns the parameters β
as the minimizer of this loss. We discuss some of the
popular calibration methods and their corresponding ex-
pected loss function brieﬂy.
Platt Scaling: (Platt et al., 1999) is a parametric approach
to calibration. The multi-class predictions of a classiﬁer
h are used as features for a multionomial logistic regres-
sion model fβ, which is trained on the validation set to
return probabilities. The accuracy of the classiﬁer h can
change when using the calibrated probabilities for pre-
diction (Guo et al., 2017).
Temperature Scaling (t-scaling): This method proposed
by (Guo et al., 2017) is popularly used for neural network
calibration. It uses a single scalar parameter β called the
temperature for all classes. Here, the calibrated probabil-
ities do not affect the accuracy of the classiﬁer h.
Parameters β of the calibration model fβ in both the
above methods is optimized by using the NLL loss over
the validation set. Hence,
E(x,y)∼π(X,Y ) l(h(x), y, ; β) =
n
X
i=1
K
X
j=1
yj
i log(fβ(h(xi)j)
(2)
where n is the number of samples drawn from the joint


<!-- page 3 -->
distribution (h(X), Y ) and Y is represented as one hot
vector of size K.
Quantifying miscalibration: The common metrics used
to report calibration performance are Expected Calibra-
tion Error or ECE (Guo et al., 2017) and reliability di-
agrams (DeGroot & Fienberg, 1983; Niculescu-Mizil &
Caruana, 2005). We brieﬂy describe both these measures
and use it in our work to report calibration performance.
We start with grouping conﬁdence predictions ˆpi into M
interval bins (each of size 1/M). Let Bm be the set of
indices of samples whose prediction conﬁdence falls into
the interval Im = ( m−1
M , m
M ]. We deﬁne accuracy of bin
Bm as,
acc(Bm) =
1
|Bm|
X
i∈Bm
1(ˆyi = yi)
where ˆyi and yi are the predicted and true class labels for
sample i. We also deﬁne the average conﬁdence within
bin Bm as,
conf(Bm) =
1
|Bm|
X
i∈Bm
ˆpi
where ˆpi is the conﬁdence for sample i.
Expected Calibration Error (ECE): (Guo et al., 2017)
deﬁne ECE as a weighted average of the bins accu-
racy/conﬁdence difference.
ECE =
M
X
m=1
|Bm|
n
|acc(Bm) −conf(Bm)|
where n is the number of samples. Lower ECE indicates
better calibration.
Reliability diagrams are visual representation of model
calibration (DeGroot & Fienberg, 1983; Niculescu-Mizil
& Caruana, 2005) as shown in Figure. 1. These diagrams
plot accuracy or the empirical frequency acc(Bm) as a
function of conﬁdence conf(Bm) for each bin Bm. So
the x-axis here ranges from [0, 1] and is divided into M
intervals. If the model is perfectly calibrated the diagram
should plot the identity function. Any deviation from a
perfect diagonal represents miscalibration.
Limitations of existing work:
Existing calibration
methods as discussed above rely on the evaluation of
the loss function E(x,y)∼π(X,Y ) l(h(x), y, ; β), which ne-
cessitates the need of labeled held-out validation data.
Here the methods inherently assume that the train, vali-
dation and test data is drawn from the same distribution
π(X, Y ). This is violated in many real world scenarios
where covariate shift occurs (discussed in Section. 3).
(Snoek et al., 2019) empirically show that deep NN are
uncalibrated under domain shift.
The effect of covariate shift on classiﬁers predictive per-
formance and various solutions to address it has been
studied under unsupervised domain adaptation literature
(Chu & Wang, 2018; Kouw & Loog, 2019). They as-
sume availability of labeled train data and assume no la-
bels on the test data. The performance of these models is
measured using accuracy, while not considering the cal-
ibration of the models. Our work introduces this issue
for domain adaptation models. We show how existing
methods fail to calibrate domain adaptation models and
provide a simple modiﬁcation to adapt any existing cali-
bration technique to dataset shift.
3
MISCALIBRATION UNDER
COVARIATE SHIFT
In this section, we discuss covariate shift and how it af-
fects calibration using a synthetic example shown in Fig-
ure. 2. We organize this section as follows- (1) we for-
malize the assumption of covariate shift (2) we consider
a case of a miscalibrated classiﬁer (3) We attempt to cal-
ibrate the classiﬁer using existing techniques on source
validation data and (4) We assume the availability of tar-
get labels and show how the calibration performance can
differ from doing calibration using source as in (3).
Covariate shift assumption: Consider that πtr(X, Y )
represents the joint distribution of inputs and outputs of
the training data and πte(Y |X) represents the same for
the testing data. Under covariate shift, we assume that
πtr(X) ̸= πte(X)
and
πtr(Y |X) = πte(Y |X),
i.e the input distribution changes between train and test
data (covariate denotes input), while the conditional dis-
tribution of the outputs given the inputs π(Y |X) remains
unchanged.
This is illustrated in Figure.
2(a) using
two multivariate Gaussian distributions with different co-
variance matrices as our initial distribution for πtr(X)
for source and πte(X) for target. We consider a binary
classiﬁcation task where π(Y = 1|X) is the same for
both source and target and changes as a function of the
x-coordinate. This results in the difference of joint distri-
bution between train and test, πtr(X, Y ) ̸= πte(X, Y ).
The resulting labeled source and target data are also high-
lighted as source 1, source 0 and target 1, target 0.
Classiﬁer mis-calibration: To emulate a realistic set-
ting, we mis-specify the classiﬁer h by training a non-
linear MLP classiﬁer on ﬁnite samples from the source
data. This allows us to have a situation for which proba-
bility distribution learnt by h signiﬁcantly deviates from


<!-- page 4 -->
(a)
(b)
(c)
(d)
(e)
(f)
Figure 2: Comparing true probability distribution π(Y = 1|X) with πθ(Y = 1|X) (output of classiﬁer h) obtained
after post-hoc calibration. (a) true π(X) and π(Y |X) (b) π(X) of source and πθ(Y |X) after calibration using source
(c) π(X) of target and πθ(Y |X) after calibration using source (d) π(X) of source and πθ(Y |X) after calibration using
target (e) π(X) of source and πθ(Y |x) after calibration using our weighted method. (f) Reliability diagram on target
the true π(Y |X). If h was capable of learning the true re-
lationship, the model would be calibrated both on source
and target.
Correcting the calibration with existing techniques:
We then attempt to calibrate h using an existing cali-
bration technique called isotonic regression (Zadrozny
& Elkan, 2002) resulting in the calibrated probability
distribution shown in Figure.
2(b) for source and in
2(c) for target. Here we notice that h is calibrated on
source, but not on target. For example consider the 0.7-
0.8 probability band. Here, 70-80% points are positive
in source showing calibration whereas nearly 100% of
points in the target are positive (red), showing miscali-
bration.. The loss function used in the current calibration
methods (as discussed in Section. 2) is deﬁned on a held
out source validation data assumed to have the same dis-
tribution as the train data, i.e E(x,y)∼πtr
(X,Y ) l(h(x), y, ; β)
is the evaluated loss function. These methods are ideal
when the train and test distributions are identical. How-
ever, if πtr(X, Y ) ̸= πte(X, Y ), it follows that the ex-
pected loss function to be minimized is different for train
and test distribution, i.e.
E(x,y)∼πtr
(X,Y )[l(h(x), y, ; β)]
̸= E(x,y)∼πte
(X,Y )[l(h(x), y, ; β)].
On using calibration
methods derived using the train data over the shifted test
data, the conﬁdence estimates given by the model are no
longer reliable.
Calibration using target data:
One can solve this
by obtaining labeled data under the test distribution
πte(X, Y ) and directly computing the expected loss
function over the test data, E(x,y)∼πte
(X,Y )[l(h(x), y, ; β).
However, we often do not not have access to the label
information on the test data. Here for illustration, we as-
sume the availability of labeled test data for calibration
and show the resulting probability distribution in Figure.
2(d) for the target data. The difference in the resulting
probability distribution in Figure. 2(c) and Figure. 2(d),
clearly show how the calibrated probability distributions
differ when using source or target data for calibration.
Further, Figure. 2(f) shows quantitatively using reliabil-
ity diagram that using source data for calibration on tar-
get can perform worse than an uncalibrated model mov-
ing it further away from perfect calibration.
Mis-calibration in domain adapted classiﬁer:
The
above discussion also extends to the case when h is learnt
using existing domain adaptation techniques. In addition
to labeled source data, these techniques also use the un-
labeled target data to learn the classiﬁer h. This reduces
overﬁtting of the learnt classiﬁer on the source labeled

[CAPTION] Figure 2: Comparing true probability distribution π(Y = 1|X) with πθ(Y = 1|X) (output of classiﬁer h) obtained


<!-- page 5 -->
data, and hence improves the generalization (or predic-
tive accuracy) on the unseen target data. However, these
models can still remain uncalibrated in the target domain.
This is inevitable because we learn h from ﬁnite source
data, or simply due to optimization issues. In Figure. 1
we show a reliability diagram of domain adapted classi-
ﬁer (CDAN on LeNet-5) on USPS dataset, the classiﬁer
is trained on labeled MNIST and achieves an accuracy
of 70% on USPS. We notice that uncalibrated classiﬁer
is far from perfect. Even after using existing calibrations
methods like t-scaling on source validation data, we no-
tice that the model still remains uncalibrated.
Both the synthetic and domain adapted examples dis-
cussed above, show the performance gap between per-
fect calibration (or reference calibration obtained using
labeled target) and calibration obtained by using source
data. We seek to close this performance gap without re-
questing new labeled data from the target.
4
IMPORTANCE SAMPLING FOR
CALIBRATION UNDER COVARIATE
SHIFT
To address the problem of miscalibration under covari-
ate shift discussed in Section. 3, we introduce an im-
portance sampling approach for estimating the calibra-
tion loss. For this, we assume access to labeled train-
ing data (x, y) ∼πtr(X, Y ), and unlabeled test data
x ∼πte(X). A classiﬁer h is assumed to be trained
either using only the labeled train data, or by using ex-
isting unsupervised domain adaptation techniques. Our
objective is to ensure that the classiﬁer h is calibrated on
the test distribution. We describe our approach and the
intuition behind it.
Consider the calibration loss deﬁned over the test dis-
tribution E(x,y)∼πte(x,y) l(h(x), y, ; β). This cannot be
computed using samples drawn from πte(X, Y ) since
we do not have access to the labels Y from the test dis-
tribution. However, note that we have access to sam-
ples drawn from πtr(X, Y ) and hence the calibration
loss over the training distribution can be computed as
E(x,y)∼πtr(X,Y ) l(h(x), y, ; β). Hence, we seek to adapt
the calibration loss deﬁned on the training distribution
to formulate the calibration error on the test distribution.
This can be done using importance sampling in the fol-
lowing way:-
E(x,y)∼πte(X,Y ) l(h(x), y, ; β)
=
R
x
R
y l(h(x), y; β)πte(x, y)dxdy
=
R
x
R
y l(h(x), y; β) πte(x,y)
πtr(x,y)πtr(x, y)dxdy
=
R
x
R
y l(h(x), y; β) πte(x)πte(y|x)
πtr(x)πtr(y|x)πtr(x, y)dxdy
Using the covariate shift assumption, we have πtr(X) ̸=
πte(X) and πtr(Y |X) = πte(Y |X). From these as-
sumptions, it follows that:
E(x,y)∼πte(X,Y )[l(h(x), y, ; β)]
=
R
x
R
y l(h(x), y, ; β) πte(x)
πtr(x)πtr(x, y)dxdy
= E(x,y)∼πtr(X,Y )
πte(x)
πtr(x)l(h(x), y, ; β)
The above result is summarized in Theorem 4.1,
Theorem 4.1 The calibration loss with covariate shift
correction on the test data is equivalent to the density
ratio weighted calibration loss on the training data, i.e
E(x,y)∼πte(X,Y )[l(h(x), y, ; β)]
= E(x,y)∼πtr(X,Y )[γ(x)l(h(x), y, ; β)],
where γ(x) = πte(x)
πtr(x) is the density ratio and supp(πte)
⊃supp(πtr) where supp(πtr) = {x|πtr(x) = 0}
Weighting the train data with density ratios given by γ(x)
is an importance sampling approach. By increasing the
relative weight of those regions of the training distribu-
tion which also have a high density under the test dis-
tribution, we adapt πtr to represent πte.
We can ob-
serve the qualitative behaviour of calibration when us-
ing our method in the following way. Consider the syn-
thetic data example and the isotonic regression calibrator
trained on the source data in Section. 3. We incorporate
the weighted calibration loss for isotonic regression to
optimize the calibrator on the source data. Here, we use
the ground truth density ratios computed from the known
distributions. The resulting probability distribution ob-
tained in Figure. 2(e) using our method is similar to the
probability distribution obtained by using target labels in
Figure. 2(d). We also note from the reliability diagram
in Figure. 2(e) that the performance after using weighted
calibration loss is closer to perfect-calibration.
In order to estimate density ratios γ(x), we require
knowledge of the true data distribution π(x) for train and
test data, which is unknown. However, we typically have
sampling access to π(x) via ﬁnite datasets which we use
to estimate the density ratios in a likelihood-free (LF)
way. Examples of some LF estimators include nearest
neighbour (Kremer et al., 2015), discriminative estima-
tion (Bickel et al., 2007) etc. We can, in principle, esti-
mate this ratio directly in the original input space. How-
ever, when the inputs are high dimensional, the estimated
loss may suffer from large estimation variances because
of greater divergence between the distributions πte and
πtr (Snoek et al., 2019). We further elaborate on this ob-
servation and discuss solutions to address it in the subse-
quent subsections.


<!-- page 6 -->
4.1
FEATURE REPRESENTATION FOR
IMPORTANCE SAMPLING
In this section, we discuss practical difﬁculties in ap-
plying importance weighted calibration and introduce a
method to address some of these difﬁculties by using a
suitable feature representation. There are two primary
concerns in using importance weighted calibration on
real train πtr and test πte distributions.
1 Accuracy of estimation: The variance of calibra-
tion loss estimate in Theorem 4.1, and hence the
accuracy of the calibration is affected by the diver-
gence between πte and πtr. This relation is dis-
cussed in (Cortes et al., 2010) and summarized in
Lemma 4.2 as follows:
Lemma 4.2 The variance of importance weighted
calibration loss is bounded by the Renyi divergence
dα,
V arπtr[lγ] = Eπtr[(lγ)2]) −(Eπtr[lγ])2
≤dα+1(πte||πtr)(Eπte[lγ)])1−1
α −(Eπte[lγ))2
where
Renyi
divergence
dα+1(πte||πtr)
=
[P
x
πte(x)α+1
πtr(x)α ]
1
α where hyperparameter α > 0.
From Lemma. 4.2 it is clear that the smaller the
divergence, the better the chance of getting an
accurate estimate of the calibration loss, in turn
affecting the accuracy of the ﬁnal calibrator.
2 Unbounded/ undeﬁned density ratios: The sup-
port of the train distribution might not contain the
support of the test distribution as required by The-
orem 4.1. When this is violated, the density ratios
can grow to inﬁnity thus resulting in a undeﬁned or
incorrect estimate.
We address both these concerns using a method similar to
(You et al., 2019), by estimating importance weights us-
ing domain-invariant features instead of the original co-
variates. Let πtr
f and πte
f be the domain-invariant feature
distributions of the train and the test data respectively,
we step from the input space to the feature space and es-
timate γf(x) =
πte
f (x)
πtr
f (x) instead of γ(x) = πte(x)
πtr(x). This
ensures that the variance of calibration loss estimate is
bounded, because by using domain-invariant features we
have dα+1(πte
f ||πtr
f ) smaller than dα+1(πte||πtr). Fur-
thermore, the assumption on the support of πtr in πte
can hold well in the learned feature space because of in-
creased overlap in the distributions compared to the co-
variate space.
Note that by estimating the importance weights in the
domain-invariant feature space we can only reduce the
distribution divergence, and never completely eliminate
it to zero. Hence we can only expect to improve over the
bias created by original unweighted calibrator. Perfect
calibration close to using the target labels is not guaran-
teed.
4.2
DENSITY RATIO ESTIMATION
To compute density ratios we adopt an approach similar
in (Bickel et al., 2007; You et al., 2019) where a discrim-
inator is used to distinguish or classify source samples
(with label d=1) from target samples (with label d=0).
Under this model γf(x) =
πte
f (x)
πtr
f (x) =
P (d=1)
P (d=0)
P (d=0|x)
P (d=1|x)
where density ratio estimation is decomposed into two
parts - (1) P (d=0|x)
P (d=1|x) which can be estimated by a discrim-
inative model to distinguish source and target samples.
The model here is trained on the domain-invariant fea-
ture representation of source and target. and (2) P (d=1)
P (d=0) -
is a constant value that can be estimated with the sample
sizes of both domains.
Practical considerations:
The importance weights
learnt by the discriminator may differ from the true den-
sity ratios. This can happen because, (1) the divergence
between train and test is not completely zero leading
to high variance (Lemma. 4.2) and (2) training on ﬁ-
nite samples from source and target data leads to over-
ﬁtting of the discriminator on some features. This results
in highly conﬁdent predictions and hence small impor-
tance weights. We follow (Grover et al., 2019) to offset
these challenges using the following techniques - self-
normalization, ﬂattening, and clipping.
5
EXPERIMENTAL SETUP
In this section, we evaluate the efﬁcacy of our proposed
importance weighting technique in adapting two post-
hoc calibration methods, Platt scaling and temperature
scaling (t-scaling), to handle domain shifts. We use the
Expected Calibration Error (ECE) discussed in Sec. 2 to
measure the calibration performance on the target data.
We compare the performance of our calibration method
(Weighted) to three baselines:
(1) Uncalibrated, i.e. the source classiﬁer as is without
any post-hoc calibration;
(2) Unweighted, i.e. the post-hoc calibrator is trained on
the source domain;
(3) Using target or target-calibrated i.e. the post-hoc
calibrator is trained on the labeled target domain. This
can be considered as a gold standard (requiring labels
from target domain), i.e. a lower-bound on the calibra-
tion error for the target data.


<!-- page 7 -->
(a)
(b)
(c)
Figure 3: Figure showing different parameters that effect calibration. (a) Increasing divergence between source and
target (b) Increasing number of samples used for calibration (c) Increasing noise in the ground truth importance weights
Table 1: ECE scores of Platt (top) and t-scaling (bot-
tom), comparing baselines and the proposed method on
pseudo-synthetic datasets. The weighted method uses
ground truth density ratios for calibration.
Dataset
CIFAR-10 classes source ratio target ratio
S1 →T1
2&7
1:4
4:1
S2 →T2
1&8
2:5
3:4
S3 →T3
3&4
5:1
1:3
S4 →T4
6&9
2:3
5:1
Method
S1 →T1 S2 →T2 S3 →T3 S4 →T4
Uncalibrated 0.134
0.019
0.142
0.020
Unweighted 0.163
0.018
0.204
0.026
Weighted
0.040
0.023
0.042
0.017
Using target 0.037
0.024
0.041
0.016
Unweighted 0.124
0.007
0.144
0.013
Weighted
0.030
0.005
0.030
0.007
Using target 0.027
0.005
0.029
0.007
The rest of the section is organized as follows. First,
we study the behaviour of the calibration on pseudo-real
datasets. This paradigm allows us to control the density
ratios and analyze how the performance is affected by
it. Then, we apply this technique on real world datasets.
Here, we derive the importance weights using the dis-
criminative density ratio estimation method (Sec. 4.2)
and use them for weighted calibration. The classiﬁers
used here include both ImageNet pre-trained ResNet50
(He et al., 2016) models trained only on the labeled
source data and popular domain adapted models such as
CDAN (Long et al., 2018) which use both labeled source
data and unlabeled target data.
5.1
PSEUDO-REAL WORLD EXPERIMENTS
We construct synthetic datasets using the CIFAR-10
dataset which consists of 60K color images distributed
equally across ten object classes (Krizhevsky et al.,
2009). We randomly pick two classes and collect sam-
ples from these two to deﬁne a binary classiﬁcation task.
We vary the mixing ratio of the two classes, thereby cre-
ating datasets with covariate shift. Sample values used
in experiments for source and target domain are docu-
mented in Table 1 as (Si →Ti). For example, if the
source consists of 1 : 4 ratio of class 1 and class 2,
then the target with ratio of 4 : 1 for the classes can
be seen to have a covariate shift. To create ratio val-
ues greater than one, we duplicate the data points of the
class. Our method of construction automatically gives us
the ground truth importance weights from the mixing ra-
tios of source and target data. In the previous example,
the source points from class 1 have an importance weight
of 4 and the source points in class 2 have an importance
weight of (1/4) =0.25 for the given target.
Using these ground truth importance weights we perform
a weighted calibration of a LeNet-5 classiﬁer trained on
the labeled source data. We use 70% of the source data
for training the classiﬁer and 30% as validation data for
calibration. For testing, we use 70% of the target data to
compute the ECE and 30% as validation data for target-
calibrated model. We consider 10 different train, val-
idation, and test splits and report the mean in Table.
1 (standard deviation in supplementary). We note that
the weighted calibration signiﬁcantly outperforms un-
weighted and uncalibrated models except in S2 →T2
where the source ratio is close to the target ratio. We
further make use of this setting to empirically study the
effect of the following parameters on the calibration per-
formance (ECE) in the target domain.
Domain shift between source and target: We consider

[CAPTION] Figure 3: Figure showing different parameters that effect calibration. (a) Increasing divergence between source and

[CAPTION] Table 1: ECE scores of Platt (top) and t-scaling (bot-


<!-- page 8 -->
Table 2: ECE scores of Platt scaling (top) and t-scaling (bottom), comparing the baselines and the proposed method.
Importance weight are estimated using discriminator.
(a) Ofﬁce-31 dataset trained using pre-trained Resnet50.
Method
A→D A→W D→A D→W
W→A
W→D
Uncalibrated 0.038
0.036
0.147
0.158
0.045
0.096
Unweighted
0.082
0.06
0.278
0.093
0.199
0.085
Weighted
0.125
0.051
0.06
0.041
0.052
0.083
Using target
0.105
0.062
0.057
0.06
0.026
0.013
Unweighted
0.040
0.045
0.145
0.028
0.136
0.023
Weighted
0.047
0.081
0.134
0.031
0.039
0.02
Using target
0.047
0.030
0.031
0.029
0.028
0.013
(b) Digits dataset, domain adapted using CDAN.
Method
U→M
M→U
uncalibrated
0.309
0.206
unweighted
0.134
0.258
weightes
0.132
0.077
using target
0.035
0.023
unweighted
0.154
0.196
weighted
0.108
0.13
using target
0.140
0.042
Table 3: Ofﬁce-home, domain adapted using using CDAN. ECE scores of Platt scaling (top) and t-scaling (bottom),
comparing baselines and the proposed method. Importance weight are estimated using discriminator.
Method
Ar→Cl Ar→Pr Ar→Rw Cl→Ar Cl→Pr Cl→Rw Pr→Ar Pr→Cl Pr→Rw Rw→Ar Rw→Cl Rw→Pr
Uncalibrated 0.114
0.091
0.119
0.135
0.133
0.113
0.098
0.158
0.021
0.124
0.102
0.027
Unweighted 0.109
0.098
0.093
0.104
0.077
0.099
0.306
0.197
0.133
0.083
0.159
0.148
Weighted
0.093
0.081
0.093
0.071
0.051
0.068
0.085
0.173
0.081
0.067
0.132
0.059
Using target 0.092
0.060
0.029
0.052
0.076
0.053
0.028
0.134
0.069
0.035
0.074
0.05
Unweighted 0.145
0.033
0.038
0.041
0.079
0.078
0.038
0.287
0.104
0.117
0.167
0.027
Weighted
0.128
0.015
0.064
0.069
0.064
0.057
0.085
0.268
0.108
0.104
0.173
0.038
Using target 0.046
0.034
0.043
0.048
0.026
0.040
0.049
0.038
0.029
0.019
0.036
0.026
two classes of CIFAR-10 and ﬁx the source class ratio
to 8 : 1 and the calibration method to t-scaling. We
then change the target ratios as N1 : N2. Here as the
value of N1 decreases and N2 increases the domain shift
of target compared to the source increases. In Figure.
3(a), we see that performance gap in ECE between un-
weighted source calibration and using target for calibra-
tion increases as the datashift increases. The weighted
calibration performs as well as using target labels for cal-
ibration (lower bound), even though our method doesn’t
have access to any target labels.
Number of validation samples used for training the
calibrator: We vary the number of validation points
used for t-scaling in Figure. 3(b) and keep the remaining
parameters ﬁxed. We note that the weighted ECE perfor-
mance may worsen compared to unweighted calibration
at signiﬁcantly smaller validation sample size. Also, the
performance of target calibration and unweighted source
calibration itself may degrade with decreasing validation
samples. The threshold of sample size for this degrada-
tion maybe differ based on the complexity of the dataset,
e.g., the number of classes.
Quality of importance weights: In reality, empirically
estimated density ratios are noisy and may deviate from
the ground truth ratios. In Figure. 3(c) we simulate this
setting by increasing the average amount of normal noise
added to the importance weights (rest of the parame-
ters are constant) and observing its effect on the calibra-
tion performance. We notice that with increasing noise,
weighted calibrations performance performance can de-
grade to below uncalibrated. The sensitivity of the cali-
brator to noise can change with extent of domain shift.
5.2
EXPERIMENTS ON REAL WORLD DATA
In this section, we evaluate calibration performance on
real world datasets using classiﬁers trained only on
source (using pre-trained ResNet50) and a range of do-
main adapted classiﬁers (LeNet-5 and ResNet50 trained
using CDAN1, pre-trained ResNet-50). The accuracy of
all the models is reported in the supplementary. We di-
vide both source and target data into 70/30 splits (we
directly use standard train/test when available). For the
source, the larger split is used for training and the smaller
split is used as validation for the post-hoc calibration
method. For the target, the larger split is used for testing
and the smaller split is used to train the target-calibrated
model. To obtain importance weights for our method we
1We train using publicly available codes given by authors

[CAPTION] Table 2: ECE scores of Platt scaling (top) and t-scaling (bottom), comparing the baselines and the proposed method.

[CAPTION] Table 3: Ofﬁce-home, domain adapted using using CDAN. ECE scores of Platt scaling (top) and t-scaling (bottom),


<!-- page 9 -->
train a discriminator (2-hidden layer MLP) on domain in-
variant features as discussed in Section. 4.2. We perform
normalization on the obtained weights, and leave exper-
imentation with ﬂattening and clipping for future work.
We experiment with different discriminator and calibra-
tor initializations keeping the classiﬁer and dataset ﬁxed
and report the mean performance for 5 iterations (stan-
dard deviation in supplementary).
Classiﬁers trained only on source: We use the Ofﬁce-
31 dataset (Saenko et al., 2010) which is concerned with
the task of object recognition. This dataset has images
from four domains: Amazon images (A), Webcam (W)
(low-resolution) and DSLR (high-resolution) (D), with
4,652 images and 31 categories.
We evaluate on six
source to target transfer tasks A →W, A →D, D →A, D
→W, W →A and W →D. We use Imagenet pre-trained
ResNet-50 as our initial classiﬁer with the ﬁnal layer re-
placed to output 31 classes. We re-train it on the labeled
source and test it on the target. The domain-invariant fea-
ture representation for the discriminator is obtained form
the ﬁnal layer of the pre-trained Resnet-50 model.
Domain adapted classiﬁers: We use Conditional Do-
main Adversarial Network (CDAN) (Long et al., 2018),
a recent domain adaptation technique to train two differ-
ent classiﬁers on different datasets mentioned here - (1)
Digits dataset consists of images form MNIST (M) and
USPS (U) (Ganin et al., 2016) comprising of 10 classes,
here we apply CDAN on a LeNet-5 classiﬁer. We evalu-
ate two source to target transfer tasks M →U and U →
M. (2) Ofﬁce-Home dataset (Venkateswara et al., 2017)
consists of images form Art (Ar, 2427), Clipart (Cl,
4365), Product (Pr, 4439) and Realworld (Rw, 4357)
(size in parenthesis) comprising of 65 classes. Here we
apply CDAN on Resnet50 classiﬁer.
We evaluate 12
source to target transfer tasks shown in Table. 3, explor-
ing all the permutations of the four datasets. In both the
datasets, we use the features obtained from the domain
adapted layer of CDAN to train our discriminator.
Discussion: In Table. 2(a), 2(b) and 3. we compare the
ECE scores of our weighted calibration methods with the
baselines. The models here span a range of accuracy’s
from 30% to 97% on the target data and still remain
uncalibrated. This shows that accounting for accuracy
alone does not gurantee calibration. We use bold font to
highlight results where weighted calibration outperforms
the uncalibrated ECE. In italics we highlight weighted
calibration which reduces the bias in unweighted cali-
bration but still performs worse than uncalibrated ECE.
This is in agreement with our discussion in Section. 4.2,
where we note that we can only reduce the bias in using
the source data for calibration but not completely elimi-
nate. From these experiments, we observe that our pro-
posed method helps in increasing the calibration perfor-
mance considerably in number of cases such as D →A in
Ofﬁce-31 dataset where the ECE performance improves
from 14.7% to 6% , M →U in MNIST-USPS dataset
where the ECE performance improves from 20.6% to
7.7% and Cl →Pr in Ofﬁce-home dataset where the ECE
performance improves from 13.3% to 6.4%.
To explain the poor performance of weighted calibra-
tion on the remaining datasets, we refer to the analy-
sis in Figure.3. For example, in experiments involving
ofﬁce-31 datasets consider A →D or A →W where
A has considerably larger data compared to D or W.
This could have affected the importance of weight es-
timation (leading to overﬁtting of the discriminator and
hence resulting in poor importance weights) or the low
samples used in validation data could have itself affected
both the weighted and using target labels calibration per-
formance. In general, performance of calibration can be
affected by multiple factors.
6
CONCLUSION
In this work, we identiﬁed that neural models, including
domain adapted models, are miscalibrated under covari-
ate shift. This indicates that existing domain adaption
techniques focus on accuracy and not calibration. Exist-
ing calibration techniques fail to calibrate them or even
sometimes worsen the calibration performance. This is
a result of the inherent assumptions made by these tech-
niques which fail to hold true when domain shift occurs.
We propose a new method that overcomes the limita-
tions of the existing techniques and adapts any calibra-
tion technique to work under domain shift using impor-
tance sampling. We show that with ground truth density
ratios our method signiﬁcantly improves the calibrator.
We further implement the proposed method on real world
datasets by employing a binary classiﬁer to estimate the
density ratios. We demonstrate performance improve-
ments on different datasets and analyze the effects of the
different parameters involved. We also note that the ef-
ﬁcacy of our method on real world datasets is limited by
accuracy of the density ratio estimation process. There-
fore, we observe that improving density ratio estimation
is a crucial future direction of research which will help
in improving calibration performance.
References
Bella, A., Ferri, C., Hern´andez-Orallo, J., and Ram´ırez-
Quintana, M. J. Calibration of machine learning mod-
els. In Handbook of Research on Machine Learning
Applications and Trends: Algorithms, Methods, and
Techniques, pp. 128–146. IGI Global, 2010.


<!-- page 10 -->
Berk, R. and Hyatt, J. Machine learning forecasts of risk
to inform sentencing decisions.
Federal Sentencing
Reporter, 27(4):222–228, 2015.
Bickel, S., Br¨uckner, M., and Scheffer, T. Discriminative
learning for differing training and test distributions. In
Proceedings of the 24th international conference on
Machine learning, pp. 81–88, 2007.
Chu, C. and Wang, R.
A survey of domain adapta-
tion for neural machine translation.
arXiv preprint
arXiv:1806.00258, 2018.
Cortes, C., Mansour, Y., and Mohri, M. Learning bounds
for importance weighting. In Advances in neural in-
formation processing systems, pp. 442–450, 2010.
Cosmides, L. and Tooby, J.
Are humans good intu-
itive statisticians after all? rethinking some conclu-
sions from the literature on judgment under uncer-
tainty. cognition, 58(1):1–73, 1996.
DeGroot, M. H. and Fienberg, S. E.
The comparison
and evaluation of forecasters.
Journal of the Royal
Statistical Society: Series D (The Statistician), 32(1-
2):12–22, 1983.
Ganin, Y., Ustinova, E., Ajakan, H., Germain, P.,
Larochelle, H., Laviolette, F., Marchand, M., and
Lempitsky, V. Domain-adversarial training of neural
networks. The Journal of Machine Learning Research,
17(1):2096–2030, 2016.
Grover, A., Song, J., Kapoor, A., Tran, K., Agarwal,
A., Horvitz, E. J., and Ermon, S. Bias correction of
learned generative models using likelihood-free im-
portance weighting. In Advances in Neural Informa-
tion Processing Systems, pp. 11056–11068, 2019.
Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q.
On calibration of modern neural networks.
In Pro-
ceedings of the 34th International Conference on Ma-
chine Learning-Volume 70, pp. 1321–1330. JMLR.
org, 2017.
He, K., Zhang, X., Ren, S., and Sun, J.
Deep resid-
ual learning for image recognition. In Proceedings of
the IEEE conference on computer vision and pattern
recognition, pp. 770–778, 2016.
Heaton, J., Polson, N., and Witte, J. H. Deep learning for
ﬁnance: deep portfolios. Applied Stochastic Models in
Business and Industry, 33(1):3–12, 2017.
Kouw, W. M. and Loog, M. A review of domain adapta-
tion without target labels. IEEE transactions on pat-
tern analysis and machine intelligence, 2019.
Kremer, J., Gieseke, F., Pedersen, K. S., and Igel, C.
Nearest neighbor density ratio estimation for large-
scale applications in astronomy. Astronomy and Com-
puting, 12:67–72, 2015.
Krizhevsky, A., Hinton, G., et al. Learning multiple lay-
ers of features from tiny images. 2009.
Long, M., Cao, Z., Wang, J., and Jordan, M. I. Con-
ditional adversarial domain adaptation. In Advances
in Neural Information Processing Systems, pp. 1640–
1650, 2018.
Modares, H., Ranatunga, I., Lewis, F. L., and Popa, D. O.
Optimized assistive human–robot interaction using re-
inforcement learning. IEEE transactions on cybernet-
ics, 46(3):655–667, 2015.
Niculescu-Mizil, A. and Caruana, R.
Predicting good
probabilities with supervised learning.
In Proceed-
ings of the 22nd international conference on Machine
learning, pp. 625–632, 2005.
Platt, J. et al. Probabilistic outputs for support vector
machines and comparisons to regularized likelihood
methods. Advances in large margin classiﬁers, 10(3):
61–74, 1999.
Saenko, K., Kulis, B., Fritz, M., and Darrell, T. Adapting
visual category models to new domains. In European
conference on computer vision, pp. 213–226. Springer,
2010.
Snoek, J., Ovadia, Y., Fertig, E., Lakshminarayanan, B.,
Nowozin, S., Sculley, D., Dillon, J., Ren, J., and Nado,
Z. Can you trust your model’s uncertainty? evaluating
predictive uncertainty under dataset shift. In Advances
in Neural Information Processing Systems, pp. 13969–
13980, 2019.
Triantafyllidis, A. K. and Tsanas, A. Applications of ma-
chine learning in real-life digital health interventions:
Review of the literature. Journal of medical Internet
research, 21(4):e12286, 2019.
Venkateswara, H., Eusebio, J., Chakraborty, S., and Pan-
chanathan, S. Deep hashing network for unsupervised
domain adaptation. In (IEEE) Conference on Com-
puter Vision and Pattern Recognition (CVPR), 2017.
You, K., Wang, X., Long, M., and Jordan, M. Towards
accurate model selection in deep unsupervised domain
adaptation. In International Conference on Machine
Learning, pp. 7124–7133, 2019.
Zadrozny, B. and Elkan, C. Obtaining calibrated proba-
bility estimates from decision trees and naive bayesian
classiﬁers. In Icml, volume 1, pp. 609–616. Citeseer,
2001.
Zadrozny, B. and Elkan, C.
Transforming classiﬁer
scores into accurate multiclass probability estimates.
In Proceedings of the eighth ACM SIGKDD interna-
tional conference on Knowledge discovery and data
mining, pp. 694–699, 2002.