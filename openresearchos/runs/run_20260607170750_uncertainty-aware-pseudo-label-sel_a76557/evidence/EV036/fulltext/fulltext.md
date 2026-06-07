<!-- page 1 -->
Complementing Semi-Supervised Learning with Uncertainty Quantiﬁcation
Ehsan Kazemi1
1Computer Science Dept., University of Central Florida, Orlando, Florida, USA
Abstract
The problem of fully supervised classiﬁcation is
that it requires a tremendous amount of annotated
data, however, in many datasets a large portion of
data is unlabeled. To alleviate this problem semi-
supervised learning (SSL) leverages the knowl-
edge of the classiﬁer on the labeled domain and
extrapolates it to the unlabeled domain which has
a supposedly similar distribution as annotated data.
Recent success on SSL methods crucially hinges
on thresholded pseudo labeling and thereby con-
sistency regularization for the unlabeled domain.
However, the existing methods do not incorporate
the uncertainty of the pseudo labels or unlabeled
samples in the training process which are due to
the noisy labels or out of distribution samples ow-
ing to strong augmentations. Inspired by the re-
cent developments in SSL, our goal in this paper
is to propose a novel unsupervised uncertainty-
aware objective that relies on aleatoric and epis-
temic uncertainty quantiﬁcation. Complementing
the recent techniques in SSL with the proposed
uncertainty-aware loss function our approach out-
performs or is on par with the state-of-the-art over
standard SSL benchmarks while being computa-
tionally lightweight. Our results outperform the
state-of-the-art results on complex datasets such as
CIFAR-100 and Mini-ImageNet.
1
INTRODUCTION
Deep learning has seen unprecedented success by provid-
ing a remarkable performance on tasks such as computer
vision, natural language processing, and virtual reality. One
of the biggest challenges for deep models is the lack of
annotated data, while the amount of annotated data for train-
ing is the major factor in the performance of deep models.
Therefore, the performance boost gained by the use of large
datasets is at a cost for data annotation. However, the large
labeled dataset is not always available due to the difﬁculty
or infeasibility of data annotation mechanisms. Further, an-
notating the labels in many domains is either expensive or
infeasible because of accessibility and privacy constraints.
Semi-supervised learning (SSL) is a paradigm to ﬁll this
void. Semi-supervised training is becoming signiﬁcantly
important as abundant unlabeled data is used to improve the
generalization performance of deep models. The ultimate
goal of SSL is to have better performance in scarce-label
regimes as it eases the requirement of annotated data. Since
unlabeled data can be obtained at minimal cost, any perfor-
mance beneﬁt from SSL is attractive. Therefore, recently
deep learning has seen an excess of SSL methods Miyato
et al. [2018], Sajjadi et al. [2016b], Laine and Aila [2016],
Tarvainen and Valpola [2017], Sajjadi et al. [2016a], Verma
et al. [2021], Kuo et al. [2020], Sohn et al. [2020], Berthelot
et al. [2019a], Hu et al. [2021]. SSL leverages the infor-
mation on the labeled domain to use the unlabeled data in
the training process. The main part of SSL is the way the
information from the labeled domain is extrapolated to the
unlabeled domain. The basic techniques in SSL employ the
continuity assumption by taking into account that the data
which are close in a proper metric would probably share
the same labels. Following this path, many approaches in
SSL have recently been developed Jaakkola and Szummer
[2002], Zhou et al. [2004], Douze et al. [2018], Chapelle
et al. [2009], Iscen et al. [2019]. The introduced approaches
for SSL mainly hinge on consistency regularization and
pseudo labeling on the unannotated domain. Pseudo label-
ing is a mechanism for labeling the unannotated data by the
model itself. These pseudo labels are used to supervise train-
ing of the model. A common practice for pseudo labeling is
to apply a conﬁdence threshold to select only the unlabeled
samples with labels of high probability predictions Lee et al.
[2013], French et al. [2017], Sohn et al. [2020]. The conﬁ-
dence in the context of SSL refers to the highest probability
across the class predictions. By applying the conﬁdence
threshold for the pseudo labels we abstain from labeling the
1
arXiv:2207.12131v1  [cs.LG]  22 Jul 2022


<!-- page 2 -->
Figure 1: Estimation of epistemic uncertainty (a) for a model
trained in a fully supervised manner on 4000 labeled ex-
amples of CIFAR-10 dataset; and (c) for a model trained
using our SSL approach over CIFAR-10 with 4000 labeled
samples. The 1 quantile of each distribution is shown by
a dashed line. The joint t-SNE of the corresponding (b)
supervised model; and (d) SSL model, for the embedding
of weakly augmented labeled training samples (dimmer
color) and strongly augmented unannotated training sam-
ples (brighter color). The ﬁgures show that our proposed
model clusters unlabeled domains consistently with the clus-
ters of labeled data. The distributions of epistemic certainty
of labeled and unlabeled samples in the histogram (c) are
similar, while the epistemic uncertainty distributions for the
supervised model for labeled and unlabeled samples in (a)
are signiﬁcantly distinguished.
samples that are close to the decision boundaries. The core
of recent SSL methods relies on unsupervised losses to train
on both the labeled and unlabeled datasets annotated with
artiﬁcial labels. Similarly, consistency regularization gener-
ates artiﬁcial labels on randomly modiﬁed inputs or models
Laine and Aila [2016], Sajjadi et al. [2016b]. Recently de-
veloped SSL techniques exploit consistency regularization
to learn from a large amount of data by contrasting the
augmented unlabeled data and quantifying the consistency
among the unlabeled samples Hu et al. [2021], Kuo et al.
[2020]. The results for image classiﬁcation using SSL based
on consistency regularization have shown a front edge per-
formance Berthelot et al. [2019b,a], Sohn et al. [2020], Hu
et al. [2021].
Traditionally, SSL methods do not account for the uncer-
tainty for the pseudo labeling and assume that the model
would not discriminate between labeled and unlabeled do-
mains. Incorporating the uncertainties in SSL is crucial as
pointed out in Iscen et al. [2019], Tagasovska and Lopez-Paz
[2019], since it abstains the sample to contribute to the loss
in the highly uncertain regimes. We believe that it is very
important for the model to quantify uncertainty when facing
unlabeled data and to let the uncertainty contribute to the
unsupervised loss formulation. Accurate quantiﬁcation of
uncertainties from unlabeled data and pseudo labels allow
for the full potential of unlabeled data in SSL. In particular,
when the model tries to extrapolate its knowledge to the
unlabeled domain it is important to quantify and minimize
the uncertainty to ensure robustness in distribution learning.
The uncertainty attribution to the loss compensates for the
weaknesses of the pseudo labeling mechanism and lets the
model learn a distribution over the entire dataset without
discriminating between labeled and unlabeled datasets.
In this paper, we propose a method to leverage uncer-
tainty to improve the performance of SSL methods. For
this purpose, we mainly focus on heteroscedastic uncer-
tainty which stands for estimating the varying uncertainty
for each sample. We employ the existing approaches to quan-
tify heteroscedastic and multivariate uncertainties for the
SSL problem in order to improve their overall performance.
Heteroscedastic uncertainties in deep learning have two
main sources. The ﬁrst one is the epistemic uncertainty or
model uncertainty which is associated with the uncertainty
of model parameters. Measuring epistemic uncertainty is
an intractable problem and typically Bayesian inference is
applied to estimate this type of uncertainty Kendall and
Gal [2017]. Accurate quantiﬁcation of epistemic uncertainty
results in more reliable performance of the machine learn-
ing models in out of distribution domains. The second one
aleatoric uncertainty is related to the stochasticity in the
distribution of data and is not reducible by increasing the
model capacity or the training data. Accurate quantiﬁcation
of aleatoric uncertainty would maximize the performance
of the model when fusing deep model prediction with con-
ventional models by employing for example Kalman ﬁlter
method. A practical example of aleatoric uncertainty in
SSL is generating pseudo labels using the neural model
itself which are noisy by construction. As in Kendall and
Gal [2017], in this work, we decouple the uncertainties
of aleatoric and epistemic sources and neglect the cross
correlations between these two types of uncertainties. The
assumption of independence of aleatoric and epistemic un-
certainties is valid when epistemic uncertainty is small.
In this work, we leverage the mechanisms in literature to
quantify the uncertainty inherited from the unlabeled do-
main and pseudo labels and incorporate it into the loss func-
tion. The uncertainty quantiﬁcation is performed in an end-
to-end approach by harnessing the Gaussian likelihood and
certiﬁcate regression for estimating the deep epistemic un-
certainty Tagasovska and Lopez-Paz [2019]. Our uncertainty
estimation is only a one-shot forward pass which is com-
putationally efﬁcient. Employing uncertainty quantiﬁcation
techniques, we develop an uncertainty-aware unsupervised
loss that minimizes the aleatoric and epistemic uncertainty
of both the distribution of unlabeled data and the knowl-
edge extrapolated to the unlabeled domain to extract pseudo
labels. In our proposed method the epistemic uncertainty
2

[CAPTION] Figure 1: Estimation of epistemic uncertainty (a) for a model


<!-- page 3 -->
presumably accounts for uncertainty due to knowledge ex-
trapolation to low-density areas, while aleatoric uncertainty
represents the uncertainty and noise owing to pseudo label
generation and out of distribution unannotated examples.
The distribution of epistemic uncertainty of two models
which are trained over 4000 labeled data of CIFAR-10
dataset is depicted in Figure 1. The ﬁgure shows the di-
rect correlation of aligning the uncertainty distribution for
labeled and unlabeled data to the clustering performance
of the model. As it is observed from the ﬁgures when the
uncertainty distribution of annotated data is distinguished
from the uncertainty distribution of unlabeled samples, the
model performs poorly on the unannotated domain. By min-
imizing the epistemic uncertainty we implicitly align the
distribution of the model over labeled and unlabeled do-
mains while training progresses. Minimizing the epistemic
uncertainty pushes the decision boundaries of clusters to
the low-density areas, which is also recommended in the
low-density separation assumption Chapelle et al. [2009].
To precisely integrate our approach to SSL, the proposed
algorithm leverages the state-of-the-art SSL works in Mix-
Match family Berthelot et al. [2019b,a], Sohn et al. [2020]
for thresholded pseudo labeling and consistency regulariza-
tion for unannotated examples. Since we build our frame-
work on the existing methods for SSL, it bears substan-
tial similarities to the contemporary SSL algorithms. Fol-
lowing consistency regularization paradigm Berthelot et al.
[2019b,a], Sohn et al. [2020] our approach uses thresholded
pseudo labeling by exploiting the model’s prediction on
weakly augmented unannotated images and thereby apply-
ing consistency regularization on strongly augmented exam-
ples. The proposed method does not use a pre-trained model
to infer labels and the pseudo-labels are generated as the
training progresses. Therefore, in our framework the uncer-
tainty quantiﬁcation of pseudo labels is crucial. The overall
architecture of our method is shown in Figure 2. We let the
deep model assess its uncertainties when unlabeled data is
presented to the model. In particular, the neural model is
trained to output the aleatoric uncertainty and the certiﬁ-
cate for approximation of epistemic uncertainty. Afterward,
the unsupervised loss function is optimized based on uncer-
tainty estimations from the model. We develop a method
that is simple but more accurate or on par with the state-
of-the-art methods. Our approach is complementary to the
current approaches in semi-supervised learning. In the ex-
periments, we demonstrate the performance of the proposed
uncertainty-aware algorithm in complementing the state-of-
the-art SSL techniques. Our framework is computationally
attractive as it does not demand to form label propagation
graphs and is free of computing similarities across the unla-
beled examples as suggested in Hu et al. [2021], Iscen et al.
[2019], while providing competitive results. To the best of
our knowledge, this work is the ﬁrst effort to estimate and
align the uncertainty distribution of labeled and unlabeled
data. The previous efforts in SSL were mainly devoted to
enhancing the consistency across and within the strongly
and weakly augmented unlabeled samples.
Our contribution is summarized as follows: 1- We propose a
novel uncertainty-aware objective that incorporates both the
aleatoric and epistemic uncertainty to the SSL to align the
uncertainty distributions of labeled and unlabeled domains.
2- We implement our approach by complementing the con-
temporary work with the proposed uncertainty-aware frame-
work. 3- We evaluate our method across a variety of standard
SSL benchmarks. We show that our method achieves state-
of-the-art performance on standard SSL benchmarks. For
example, our method obtains 79.32% accuracy on CIFAR-
100 dataset with 10000 labeled samples compared to the
state-of-the-art of 78.11% Hu et al. [2021].
2
RELATED WORKS
The requirement of huge annotated data for learning deep
models hampered their application in different domains. Al-
though visual data is abundant, the data which is reliably
annotated is scarce. Nevertheless annotating unlabeled data
is not practical, expensive, and prone to errors and noisy
labels. The literature on SSL is wealthy and different and
distinct approaches to this topic were recently developed. In
this review, we focus mainly on the works which employ
Augmentation Anchoring and Consistency Regularization
that bears the most resemblance to our implementation of
the baseline SSL algorithm upon which we build our frame-
work. Classical works in SSL are focused on transductive
learning. In Zhou et al. [2004], Zhu et al. [2003] transductive
learning by diffusion is applied to use the entire dataset and
annotated data to infer labels for the unlabeled examples.
Applying a classiﬁer trained on annotated data can produce
high-quality pseudo labels, which is providing a form of
supervision for classiﬁer training. The pseudo label data is
used to augment the training data and the model is retrained
until convergence. In Iscen et al. [2019] a label propagation
method based on transductive label propagation is offered
where entropy is used as a measure of uncertainty being
incorporated into the diffusion matrix for pseudo labeling.
In SSL the fully supervised performance is providing the
upper bound.
Thresholded pseudo labeling Rosenberg et al. [2005] refers
to a distinct version of SSL where conﬁdence-based thresh-
olding applies to model predictions to retain labels for unan-
notated examples only when the model is sufﬁciently conﬁ-
dent. Pseudo labeling has recently seen notable success as a
part of SSL algorithms Arazo et al. [2020]. Using the soft
pseudo labels to train the unlabeled data, pseudo labeling
can be seen as an entropy minimization Grandvalet et al.
[2005] which has been efﬁciently employed in SSL methods
Miyato et al. [2018]. Random augmentations such as data
augmentation French et al. [2017], stochastic regularization
(e.g. Dropout Srivastava et al. [2014]) Sajjadi et al. [2016b],
3


<!-- page 4 -->
Laine and Aila [2016], and adversarial perturbations Miyato
et al. [2018] are employed in the pseudo labeling method.
In Sajjadi et al. [2016a], Laine and Aila [2016], Tarvainen
and Valpola [2017], Qiao et al. [2018], Miyato et al. [2018]
a so-called consistency loss is applied to both labeled and
unlabeled examples and the consistency under the different
transformations of samples or models is enhanced by mini-
mizing the discrepancies in a proper metric. These methods
are based on the fact that the model response to input should
remain the same under the transformations that result in
small perturbation in a proper metric. Consistency regular-
ization was ﬁrst introduced in Bachman et al. [2014] and
early versions of it use the exponential moving average of
model parameters Tarvainen and Valpola [2017] and previ-
ous model checkpoint for generating pseudo labels Laine
and Aila [2016]. The recent results Sohn et al. [2020] show
that strong augmentations in consistency matching provide
better classiﬁcation accuracy. In Hu et al. [2021] and Kuo
et al. [2020] the consistency across the unlabeled samples is
leveraged to improve the consistency regularization based
SSL methods.
Uncertainty estimation and incorporating only highly con-
ﬁdent predictions are very crucial in SSL. Neglecting the
uncertainty in pseudo labeling may result in drifting from
optimal parameters on noisy pseudo labels. The thresholded
pseudo labels by conﬁdence were used previously in sam-
ple selections and label propagation Lee et al. [2013], Hu
et al. [2021], Iscen et al. [2019], Sohn et al. [2020], Is-
cen et al. [2019]. Despite the importance of uncertainty
assessment in machine learning models, the uncertainty
quantiﬁcation for deep models is still an open question,
where the prior efforts are mainly attributed to Bayesian
methods Hernández-Lobato and Adams [2015], Blundell
et al. [2015], Gal and Ghahramani [2016], Kendall and
Gal [2017], Khan et al. [2018], Teye et al. [2018], being
computationally an intractable approach. Several sample-
based approaches have been developed to estimate epistemic
uncertainty using Bayesian inference. The most practical
approach is to apply dropout Monte Carlo Gal and Ghahra-
mani [2016], which is computationally expensive. Other
approaches include Bayesian ensemble for uncertainty es-
timation Lakshminarayanan et al. [2017]. In this work, we
exploit the uncertainty of the model over unlabeled data to
efﬁciently learn the actual distribution over the entire dataset
constituting from labeled and unlabeled data. Our approach
for quantifying uncertainty does not require the few-shot
mechanism with an ensemble of models and therefore com-
putationally it is efﬁcient.
3
PRELIMINARIES
In this section, we discuss the classiﬁer, the pseudo labeling
mechanism, and the supervised loss for pseudo labels that
our method is based on. We consider a set of n samples
X := {xi}n
i=1, of a h class set X with L annotated samples
denoted by XL with labels YL = {yi}L
i=1. The remaining
of u := n −L samples are unlabeled set denoted by XU :=
{xi}n
i=L+1. Our goal in SSL is to leverage the entire set X to
learn a model which maps unseen samples to the labels. We
let fθ : X →Rh, where θ is the model parameter. We divide
the network for classiﬁcation into two parts. The ﬁrst part
of the model is the feature extractor network φθ : X →Rd,
projecting the input sample to the feature vector. The second
part maps the feature vector to the conﬁdence score. This
part typically consists of a block of fully connected layers.
Therefore, the model fθ is the mapping from input space to
conﬁdence scores. The prediction of the model for input x
is the one that provides a maximum conﬁdence score
ˆyj = arg max
j
fθ(x)j
(1)
where j denotes the j-th element of vector fθ(x).
3.1
PSEUDO LABELING
Pseudo labeling is a mechanism to assign labels to unlabeled
examples such that artiﬁcial labels are further used for train-
ing. In the typical pseudo labeling process using consistency
regularization, two types of augmentations are used, weak
and strong augmentations, denoted by α(.) and A(.), respec-
tively. Similar to Berthelot et al. [2019a], Sohn et al. [2020]
we apply augmentation anchoring Berthelot et al. [2019a]
where pseudo labels are obtained from weakly augmented
samples and use the output as an anchor to align the output
from a strongly augmented sample to the anchor. During
the training, the model is used to predict pseudo labels from
strongly augmented unannotated images. The pseudo labels
with high conﬁdence for unlabeled inputs are used to train
the model. In this approach, weak augmentation of the input
images could be as simple as random crop and random hori-
zontal ﬂip, while strong augmentations might include image
augmentations methods such as random afﬁne and color
jitter with variable intensities. We employ the following loss
term
L(A(XU), ˆYU; θ)
:=
n
X
i=L+1
I(max(fθ(xi))>τc)ls(fθ(A(xi)), ˆyi)
(2)
where we let ˆY denote the set of pseudo labels and ls is any
supervised loss such as cross-entropy loss or MSE loss in
the case of soft labels ˆYL. We let τc denote the threshold for
the conﬁdence of model predictions and the loss is calcu-
lated over the sample outputs with conﬁdence higher than τc.
In contrast to the approach of Iscen et al. [2019] where the
model is ﬁrstly trained on annotated data and afterward the
labels are assigned according to (1), our proposed approach
predicts the pseudo labels during the training process. To
generate the pseudo labels we use the average of model
4


<!-- page 5 -->
prediction on K weakly augmented versions of the input
sample. In this way, the pseudo label guessing is a more con-
ﬁdent process that results in a more stable training process.
More technically, for each unlabeled sample xu, the model
output produces a pseudo label using equation (1). We use
the exponential moving average over a sequence of model
updates to compute the predictions.
4
METHOD
We develop an uncertainty-aware loss function that concen-
trates on the uncertainty from pseudo labels, the uncertainty
of the out of distribution unannotated examples, and the
uncertainty of the low-density regions due to the inclusion
of unlabeled data in the training set. In the following, we
demonstrate the proposed uncertainty-aware SSL and de-
scribe the major elements of our construction.
4.1
LOSS
Our loss function L has two components corresponding to
supervised loss and unsupervised loss. We use the super-
vised loss LS for labeled data and the unsupervised loss LU
for unlabelled data
L = LS + LU
= LS + αUA LUA + αUE LUE
(3)
where LU = αUA LUA+αUE LUE is the breakdown of un-
supervised loss, accounting for the aleatoric and epistemic
uncertainty, respectively. αUA and αUE are ﬁxed hyperpa-
rameters denoting the weights of the unsupervised losses.
In supervised learning, the model is trained by a supervised
loss over the annotated data. Our supervised loss LS em-
ploys the labeled data to minimize the loss and is in the form
LS(XL, YL; θ) :=
l
X
i=1
ls(fθ(xi), yi)
(4)
A typical choice for ls is cross-entropy loss. The unsuper-
vised loss LU is composed of two parts LUA and LUE
taking into account the aleatoric uncertainty and epistemic
uncertainty, respectively. The unsupervised loss LUA which
is aware of aleatoric uncertainty is the Gaussian likelihood
loss with a nontrivial variance to quantify the aleatoric un-
certainty. Nevertheless, the epistemic uncertainty loss LUE
measures the uncertainty of the model due to the low-density
regions in the feature space and out of distribution unlabeled
examples. The output of our model consists of three com-
ponents as shown in Figure 2: the prediction probabilities,
the uncertainty of the predictions which predict the diagonal
elements of aleatoric uncertainty, and ﬁnally the certiﬁcate
for measuring the epistemic uncertainty. For predicting the
two latter components we extend the model after the feature
extractor using fully-connected blocks.
Incorporating the uncertainties in SSL allow the hypothesis
over the labeled and unlabeled domains to extend its limit by
efﬁciently evolving the model distribution when observing
new data and incorporating the uncertainty from the unla-
beled domain and generated pseudo labels. We decouple the
aleatoric and epistemic uncertainties in the model predic-
tions and treat each uncertainty with a different loss term.
We do not compute aleatoric uncertainty over the annotated
dataset with artiﬁcial labels, however, the certiﬁcates for
epistemic uncertainty are trained over the entire dataset.
4.2
ALEATORIC UNCERTAINTY-AWARE LOSS
We start by incorporating the aleatoric uncertainty to the un-
supervised loss function, i.e., the uncertainty related to the
conditional distribution of pseudo labels conditioned on the
input unlabeled feature sample. To estimate the aleatoric un-
certainty we estimate the covariance of uncertainty estima-
tion of the prediction. We train a deep uncertainty covariance
matrix using a multivariate Gaussian density loss function.
The probability of a pseudo label ˆy given the model input x
can be estimated with the multivariate Gaussian distribution
p(ˆy|x) =
1
q
(2π)h|ˆΣ(x)|
×
exp
 
−1
2(ˆy −fθ(x))T ˆΣ(x)−1(ˆy −fθ(x))
 
(5)
where f is the deep model which provides the mean of dis-
tribution, i.e., E[ˆy|x]. In this formulation, ˆy is the pseudo
label for the unlabeled sample x. The covariance matrix pa-
rameters in the log-likelihood loss describe the distribution
of model predictions and pseudo labels of unlabeled data
with respect to the model prediction. To compute the pseudo
labels we follow Sohn et al. [2020] and we ﬁrst compute the
prediction output using weakly augmented versions of the
input sample and then we use the output as a pseudo label
for the strongly augmented version of the same sample.
In contrast to the traditional approaches in SSL which only
use predictive meanwhile neglecting the predictive variance,
we accounted for the variance of prediction with the log-
likelihood Gaussian loss. To learn f and Σ, the optimal
parameters minimize the negative log-likelihood equation.
The Maximum Likelihood loss is deﬁned as the negative of
log-likelihood of predictions using the predicted covariance
of Gaussian distribution
LUA(XU, ˆYU; θ) =
1
n −L
n
X
i=L+1
I(max(fθ(xi))>τc)
 1
2(ˆyi −fθ(xi))T ˆΣ−1
i ( ˆyi −fθ(xi)) + 1
2 ln |ˆΣi|
 
(6)
5


<!-- page 6 -->
Figure 2: The proposed method for optimizing the model parameters over unlabeled data. The aleatoric uncertainty-aware
loss LUA which minimizes the negative log of Gaussian likelihood loss over pseudo labels generated from weakly augmented
data; the epistemic loss LUE which maps the projection of certiﬁcates generated by the classiﬁcation model to zero.
where {ˆΣi}L+1≤i≤n which are h × h covariance matrices
for i −th sample, corresponding to the network uncertainty
for the output vector fθ(xi) and the pseudo label ˆyi. We
let τc denote the conﬁdence threshold for the model predic-
tion. Therefore we only incorporate the pseudo labels with
the conﬁdence more than the threshold τc. ˆΣi has h(h+1)
2
degrees of freedom due to the symmetry of covariance ma-
trix. We assume a diagonal form for the covariance matrix
Σi across all the samples. I.e., we consider the covariance
matrix as the following
ˆΣi = diag(e2u1i, e2u2i, . . . , e2uhi),
(7)
where we use the Sigmoid function in the model out-
put of covariance to ensure the elements of the vector
(u1i, u2i, . . . , uhi) are between 0 and 1. The diagonal as-
sumption for the covariance matrix decouples the prediction
for each class and neglects the correlation across the class
predictions. Although the prediction across classes can be
highly correlated we simplify the covariance matrix by ne-
glecting the full multivariate uncertainties which account
for cross variances. Regressing the covariance matrix in
exponential form removes the singularity around zero in the
loss function. In fact, our model learns to regress the uncer-
tainty of the predictions, which further are used to calibrate
the MSE loss for unsupervised data during the training. We
could also enhance conﬁdent learning by accounting for the
variance of the predictions using a threshold parameter that
could be the focus of future studies.
4.3
EPISTEMIC UNCERTAINTY ESTIMATION
LOSS
As we described earlier we estimate epistemic uncertainty
independent from aleatoric uncertainty. Epistemic uncer-
tainty which is referred also as model uncertainty reﬂects
the uncertainty in the model parameters being mainly due to
the low-density regions. Hence, the epistemic uncertainty is
reduced by increasing the data and model parameters in the
training process. Epistemic uncertainty can be large when
the model tries to extrapolate its knowledge on a domain
that is signiﬁcantly different from the data which has been
previously presented to the model. For instance, in SSL the
strongly augmented unlabeled samples which are used in
consistency regularization are signiﬁcantly different from
the weakly augmented examples and therefore the model
provides high epistemic uncertainty for these samples. The
heavily augmented images are potentially out of distribution,
which is in fact beneﬁcial in boosting the performance of
SSL Dai et al. [2017]. Nevertheless, in Sohn et al. [2020]
it is found that distribution alignment has a critical impact
on achieving state-of-the-art accuracy for datasets such as
CIFAR-100. The hypothesis of the model over the distri-
bution of labeled and unlabeled domains can dynamically
change when heavily augmented examples are presented
to the model. Our ultimate target is to mainly take account
of the epistemic uncertainty and minimize the uncertainty
when the strongly augmented unlabeled data are fed to the
model. In this way, we align the epistemic uncertainty dis-
tribution of labeled and unlabeled data and make the model
extrapolate its knowledge to the unlabeled data.
To estimate epistemic uncertainty we follow Tagasovska
and Lopez-Paz [2019] and leverage orthogonal certiﬁcates
(OCs) which constitute a set of diverse functions that project
samples to zero. OCs map out of distribution samples to non-
zero values, therefore forcing the output of OCs to be zero
during the training aligns the distributions of labeled and
unlabeled domains. To generate the orthogonal certiﬁcates,
we consider the deep neural model y = fθ(φθ(x)) where
φθ is a feature extractor from the input data x and fθ is
the neural model clustering the features in the embedding
space. Given the n samples, we construct the k certiﬁcates
C = (C1, C2, . . . , Ck) on the d dimensional features of
φθ(x) of n training samples, i.e., Φθ = {φθ(xi)}n
i=1. Each
certiﬁcate Ci is a shallow neural model which is a fully-
6

[CAPTION] Figure 2: The proposed method for optimizing the model parameters over unlabeled data. The aleatoric uncertainty-aware


<!-- page 7 -->
connected layer in our work mapping the features to zero by
minimizing the Mean Square Error (MSE) loss. If we use a
fully-connected layer for certiﬁcates, the certiﬁcate is d × k
dimensional layer to predict the zero vector of dimension k
for examples drawn from the data distribution. To impose
diversity across the certiﬁcates, the weights of certiﬁcates
are trained to be orthogonal. Therefore, the loss function
for training the certiﬁcate over labeled and unlabeled data is
formulated by
LUE(XU; θ) = 1
n
n
X
i=1
lc(CT φθ(xi), 0)
+ λ||CCT −Ik||.
(8)
Letting lc be the MSE loss encourages the certiﬁcates to
be identical to the null space of the feature vector of the
samples. As indicated in Tagasovska and Lopez-Paz [2019]
the certiﬁcates are able to distinguish the out of distribu-
tion feature samples φθ(x) which are drawn from Gaussian
distribution with the covariance matrix of small associated
eigenvalues or small ∥CV ∥where V is an eigenvector of
the covariance matrix. In SSL the high value of epistemic
uncertainty shows that the model is highly confused about
the label of unannotated example xu. By incorporating epis-
temic uncertainty into the loss function, we minimize the
discrimination for distribution of unlabeled data with artiﬁ-
cial labels when exposing unlabeled data to the neural model.
This means that the certiﬁcates will be able to discriminate
the unlabeled data which has a different distribution than
the seen data and minimize the margin among the distribu-
tions leading the model to better extrapolate its knowledge
over the low-density regions during the training. We use a
single neural model that outputs the mean, Σ for aleatoric
uncertainty-aware loss, and certiﬁcate C for computing epis-
temic uncertainty.
4.4
IMPLEMENTATION
In this section, we demonstrate the performance and efﬁcacy
of our proposed uncertainty-aware semi-supervised learn-
ing on several standard SSL benchmarks. We describe the
datasets and illustrate the SSL setup for our experiments. We
also provide the details of our method implementation and
the baseline methods. For the experiments, we follow the
benchmarks and the evaluation settings in Hu et al. [2021],
Iscen et al. [2019]. Our proposed method is the combination
of two approaches, aleatoric and epistemic uncertainty quan-
tiﬁcation for unlabeled data and pseudo labeling combined
with consistency regularization. We use the pseudo anno-
tating method following Sohn et al. [2020] by applying the
model average and taking the average of model predictions
of weakly augmented versions of the same unlabeled sam-
ple for label guessing. Averaging the outputs from weakly
augmented samples makes the label guessing more stable.
Similar to Berthelot et al. [2019b], exponential moving av-
erage (EMA) is used for averaging the model parameters
to predict the model outputs for label guessing. In all our
experiments for generating pseudo labels weak augmenta-
tions incorporate standard ﬂip-and-shift augmentation and
strong augmentations, comprise of RandAugment Cubuk
et al. [2020] where we randomly select a transformation
from a predeﬁned set of strong augmentations for each sam-
ple.
4.5
DATASETS
We evaluate our framework for image classiﬁcation task
on three different datasets: CIFAR-10 Krizhevsky et al.
[2009], SVHN Netzer et al. [2011], CIFAR-100 Krizhevsky
et al. [2009] and Mini-ImageNet Vinyals et al. [2016]. Each
dataset is considered under the conﬁguration that the dataset
is only partially labeled and unlabeled data is used for SSL.
We report the accuracy of classiﬁcations on the test sets.
We split the training set into labeled and unlabeled sets by
randomly selecting the samples from each class without re-
placement. We perform experiments with varying amounts
of labeled data for CIFAR-10 and SVHN. We evaluate these
datasets with 1000 and 4000 labeled images which are cor-
respondingly equal to 100 and 400 images per class for
CIFAR-10. For CIFAR-100 and Mini-ImageNet we use
4000 labeled data. Wide-ResNet and ResNet are used as the
backbone architectures in our experiments. Following Berth-
elot et al. [2019b], we used a Wide ResNet-28-2 (WRN-28-
2) Zagoruyko and Komodakis [2016] with 1.5M parame-
ters for CIFAR-10 and SVHN, and WRN-28-8 with 23.5M
parameters for CIFAR-100. We also use WRN-28-2 and
ResNet-18 for Mini-ImageNet. In our implementation, we
use the feature extractor combined with a fully connected
layer to predict the uncertainty covariance matrix. We also
use a linear layer on top of the feature extractor to out-
put the certiﬁcates for epistemic uncertainty. We normalize
the images for all model inputs to have zero channel-wise
mean and unit variance. The performance of deep neural
networks trained for classiﬁcation is heavily determined
by the optimizer, scheduler, learning rate, and architecture.
These details are not stressed enough in the literature for
SSL methods when reporting the empirical evaluations. In
our experiments, the initial learning rate is set to 0.01 for
all the datasets. For CIFAR-10, SVHN and CIFAR-100 we
use SGD with the initial learning of 0.03 and weight decay
of 5e-4 and for Mini-ImageNet we use AdamW Loshchilov
and Hutter [2018] with the initial learning rate of 0.002
and weight decay of 2e-2. For SGD optimization we use
cosine scheduler decay with the cosine factor 0.5. We apply
the validation set to select the model for the test evalua-
tion. The conﬁdence threshold for the pseudo labeling is
set to 0.95. The weights for the loss function in (3) use an
identical set of hyperparameters αUE = 1, and λ = 0.1
across all the datasets. We set the weight parameters in the
7


<!-- page 8 -->
Dataset
CIFAR-10
SVHN
CIFAR-100
1000 labels
4000 labels
1000 labels
4000 labels
10000 labels
MeanTeacher Berthelot et al. [2019a]
82.68%
89.64%
96.25%
96.61%
–
MixMatch Berthelot et al. [2019b]
92.25%
93.76%
96.73%
97.11%
71.69%
ReMixMatch Berthelot et al. [2019a]
94.27%
94.86%
97.17%
97.58%
76.97%
FixMatch Sohn et al. [2020]
–
95.69%
97.64%
–
77.40%
SimPLE Hu et al. [2021]
94.84%
94.95%
97.54%
97.31%
78.11%
Ours
94.92%
95.46%
97.50%
97.57%
79.23%
Table 1: CIFAR-10 and CIFAR-100 Top-1 Test Accuracy.
Dataset
Mini-ImageNet
4000 labels
architecture
MixMatch
55.47%
WRN 28-2
SimPLE
66.55%
WRN 28-2
Ours
66.81%
WRN 28-2
MeanTeacher
27.49%
ResNet-18
Label Propagation
29.71%
ResNet-18
SimPLE
49.39%
ResNet-18
Ours
50.75%
ResNet-18
Table 2: Mini-ImageNet Top-1 Test Accuracy.
loss function αUA = 75 for CIFAR-10 and SVHN datasets.
We select αUA = 150 and αUA = 300 for CIFAR-100
and Mini-ImageNet, respectively. The weight selections for
αUA is consistent with the weights of unsupervised loss in
Hu et al. [2021]. A key advantage of our method is that it
does not require many hyperparameters similar to the pre-
viously studied SSL method FixMatch Sohn et al. [2020]
and in particular SimPLE Hu et al. [2021] which requires
setting the threshold parameter for the paired loss across
unannotated samples.
4.6
EVALUATION
We compare our results with the state-of-the-art results
in SSL which include MixMatch Berthelot et al. [2019b],
ReMixMatch Berthelot et al. [2019a], FixMatch Sohn et al.
[2020], Label Propagation Iscen et al. [2019] and SimPLE
Hu et al. [2021]. Our method is computationally attractive
compared to SimPLE and ReMixMatch since it does not
use the paired loss and distribution alignment, while our
results are comparable and on par with these approaches.
We present a concise comparison of the SSL algorithms in
Table 1. Our results are comparable with the state-of-the-
art results from baseline models. Our results for CIFAR-10
and SVHN datasets are on par with the best accuracy from
baseline and the result for CIFAR-100 improves the state-of-
the-art by 1.12. We improved the state-of-the-art accuracy
for CIFAR-100 while we did not use the distribution align-
ment term which is found in Sohn et al. [2020] to be critical
to obtaining the best accuracy. We examine the performance
and the scalability of our uncertainty-aware SSL for Mini-
Dataset
CIFAR-10
CIFAR-100
4000 labels
architecture
10000 labels
architecture
Ours
95.46%
WRN 28-2
79.23%
WRN 28-8
w/o LUE
95.23%
WRN 28-2
75.36%
WRN 28-8
w/o LUA
94.90%
WRN 28-2
73.06%
WRN 28-8
w/o LUE – w/o LUA
94.97%
WRN 28-2
72.72%
WRN 28-8
λ = 0.5
95.99
WRN 28-2
79.44
WRN 28-8
Table 3: Ablation study on CIFAR-10 and CIFAR-100.
ImageNet in Table 2 where we listed the performance of our
method along the baselines for WRN 28-2 and ResNet18
architectures. We also reported the results from the base-
line Iscen et al. [2019] for a fair comparison. Our proposed
approach outperforms other approaches by a large margin
which shows the efﬁciency of capturing uncertainties due
to the acquisition of the unlabeled domain for gaining per-
formance. Our method outperforms SimPLE despite being
simpler due to removing the paired loss which is the main
contribution of SimPLE.
4.7
ABLATION STUDY
In the following, we perform an ablation study on the pro-
posed method over the CIFAR-10 and CIFAR-100 datasets
with WRN 28-2 and WRN 28-8 backbones, respectively.
Our aim is to study the effect of different components of
loss function (3) in performance improvements. In Table 3
we listed the results from different loss components. The
results show that our uncertainty-aware method outperforms
the baseline method (without uncertainty losses) by a sig-
niﬁcant margin. The reason is that we incorporate the un-
certainty from the unlabeled domain directly to the loss
function, therefore the SSL is robust to the inherited noise
and uncertainty from pseudo labels and out of distribution
unlabeled samples. It is observed from Table 3 that the
aleatoric uncertainty loss has a signiﬁcant impact on accu-
racy. This result pronounces that the performance gain is
mainly due to accounting for the aleatoric uncertainty when
the model’s knowledge is extrapolated from labeled samples
through noisy pseudo labels to the unlabeled datasets. It also
shows that the performance enhancements using a higher
weight for the orthogonality regularization in the epistemic
loss (8) which naturally encourages the diversity across the
certiﬁcates.
8

[CAPTION] Table 1: CIFAR-10 and CIFAR-100 Top-1 Test Accuracy.

[CAPTION] Table 2: Mini-ImageNet Top-1 Test Accuracy.

[CAPTION] Table 1. Our results are comparable with the state-of-the-

[CAPTION] Table 3: Ablation study on CIFAR-10 and CIFAR-100.


<!-- page 9 -->
5
CONCLUSION
We have proposed an approach that relies on aleatoric and
epistemic uncertainty quantiﬁcation to account for uncer-
tainties of unlabeled data when the network’s knowledge
is extrapolated to the unlabeled domain. We show that the
proposed uncertainty-aware objective could contribute to
gaining performance and improve the state-of-the-art meth-
ods for semi-supervised learning. Our method outperforms
the results from state-of-the-art SSL baseline models on
Mini-ImageNet, CIFAR-100, and CIFAR-10 datasets and
is on par with the recent results for SVHN while being
computationally more efﬁcient. Our method can be applied
as complementary to SSL methods to further improve the
performance.
References
Eric Arazo, Diego Ortego, Paul Albert, Noel E O’Connor, and
Kevin McGuinness. Pseudo-labeling and conﬁrmation bias in
deep semi-supervised learning. In 2020 International Joint
Conference on Neural Networks (IJCNN), pages 1–8. IEEE,
2020.
Philip Bachman, Ouais Alsharif, and Doina Precup. Learning with
pseudo-ensembles. Advances in neural information processing
systems, 27:3365–3373, 2014.
David Berthelot, Nicholas Carlini, Ekin D Cubuk, Alex Kurakin,
Kihyuk Sohn, Han Zhang, and Colin Raffel. Remixmatch:
Semi-supervised learning with distribution matching and aug-
mentation anchoring. In International Conference on Learning
Representations, 2019a.
David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Paper-
not, Avital Oliver, and Colin A Raffel. Mixmatch: A holistic
approach to semi-supervised learning. Advances in Neural
Information Processing Systems, 32, 2019b.
Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and Daan
Wierstra. Weight uncertainty in neural network. In International
Conference on Machine Learning, pages 1613–1622. PMLR,
2015.
Olivier Chapelle, Bernhard Scholkopf, and Alexander Zien. Semi-
supervised learning (chapelle, o. et al., eds.; 2006)[book re-
views]. IEEE Transactions on Neural Networks, 20(3):542–542,
2009.
Ekin D Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V Le.
Randaugment: Practical automated data augmentation with a
reduced search space. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition Workshops,
pages 702–703, 2020.
Zihang Dai, Zhilin Yang, Fan Yang, William W Cohen, and Russ R
Salakhutdinov. Good semi-supervised learning that requires a
bad gan. Advances in Neural Information Processing Systems,
30, 2017.
Matthijs Douze, Arthur Szlam, Bharath Hariharan, and Hervé Jé-
gou. Low-shot learning with large-scale diffusion. In Proceed-
ings of the IEEE Conference on Computer Vision and Pattern
Recognition, pages 3349–3358, 2018.
Geoffrey French, Michal Mackiewicz, and Mark Fisher. Self-
ensembling for visual domain adaptation.
arXiv preprint
arXiv:1706.05208, 2017.
Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian ap-
proximation: Representing model uncertainty in deep learning.
In international conference on machine learning, pages 1050–
1059. PMLR, 2016.
Yves Grandvalet, Yoshua Bengio, et al. Semi-supervised learning
by entropy minimization. CAP, 367:281–296, 2005.
José Miguel Hernández-Lobato and Ryan Adams. Probabilistic
backpropagation for scalable learning of bayesian neural net-
works. In International conference on machine learning, pages
1861–1869. PMLR, 2015.
Zijian Hu, Zhengyu Yang, Xuefeng Hu, and Ram Nevatia. Simple:
Similar pseudo label exploitation for semi-supervised classiﬁca-
tion. In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 15099–15108, 2021.
Ahmet Iscen, Giorgos Tolias, Yannis Avrithis, and Ondrej Chum.
Label propagation for deep semi-supervised learning. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 5070–5079, 2019.
Martin Szummer Tommi Jaakkola and Martin Szummer. Partially
labeled classiﬁcation with markov random walks. Advances
in neural information processing systems (NIPS), 14:945–952,
2002.
Alex Kendall and Yarin Gal. What uncertainties do we need in
bayesian deep learning for computer vision? Advances in Neural
Information Processing Systems, 30:5574–5584, 2017.
Mohammad Khan, Didrik Nielsen, Voot Tangkaratt, Wu Lin, Yarin
Gal, and Akash Srivastava. Fast and scalable bayesian deep
learning by weight-perturbation in adam. In International Con-
ference on Machine Learning, pages 2611–2620. PMLR, 2018.
Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers
of features from tiny images. 2009.
Chia-Wen Kuo, Chih-Yao Ma, Jia-Bin Huang, and Zsolt Kira.
Featmatch: Feature-based augmentation for semi-supervised
learning. In European Conference on Computer Vision, pages
479–495. Springer, 2020.
Samuli Laine and Timo Aila. Temporal ensembling for semi-
supervised learning. arXiv preprint arXiv:1610.02242, 2016.
Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell.
Simple and scalable predictive uncertainty estimation using
deep ensembles. Advances in neural information processing
systems, 30, 2017.
Dong-Hyun Lee et al. Pseudo-label: The simple and efﬁcient
semi-supervised learning method for deep neural networks. In
Workshop on challenges in representation learning, ICML, vol-
ume 3, page 896, 2013.
9


<!-- page 10 -->
Ilya Loshchilov and Frank Hutter. Decoupled weight decay regu-
larization. In International Conference on Learning Represen-
tations, 2018.
Takeru Miyato, Shin-ichi Maeda, Masanori Koyama, and Shin
Ishii. Virtual adversarial training: a regularization method for
supervised and semi-supervised learning. IEEE transactions
on pattern analysis and machine intelligence, 41(8):1979–1993,
2018.
Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco,
Bo Wu, and Andrew Y Ng. Reading digits in natural images
with unsupervised feature learning. 2011.
Siyuan Qiao, Wei Shen, Zhishuai Zhang, Bo Wang, and Alan
Yuille. Deep co-training for semi-supervised image recognition.
In Proceedings of the european conference on computer vision
(eccv), pages 135–152, 2018.
Chuck Rosenberg, Martial Hebert, and Henry Schneiderman. Semi-
supervised self-training of object detection models. 2005.
Mehdi Sajjadi, Mehran Javanmardi, and Tolga Tasdizen. Mutual
exclusivity loss for semi-supervised deep learning. In 2016
IEEE International Conference on Image Processing (ICIP),
pages 1908–1912. IEEE, 2016a.
Mehdi Sajjadi, Mehran Javanmardi, and Tolga Tasdizen. Regu-
larization with stochastic transformations and perturbations for
deep semi-supervised learning. Advances in neural information
processing systems, 29:1163–1171, 2016b.
Kihyuk Sohn, David Berthelot, Chun-Liang Li, Zizhao Zhang,
Nicholas Carlini, Ekin D Cubuk, Alex Kurakin, Han Zhang,
and Colin Raffel.
Fixmatch: Simplifying semi-supervised
learning with consistency and conﬁdence.
arXiv preprint
arXiv:2001.07685, 2020.
Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya
Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way
to prevent neural networks from overﬁtting. The journal of
machine learning research, 15(1):1929–1958, 2014.
Natasa Tagasovska and David Lopez-Paz. Single-model uncer-
tainties for deep learning. Advances in Neural Information
Processing Systems, 32:6417–6428, 2019.
Antti Tarvainen and Harri Valpola. Mean teachers are better role
models: Weight-averaged consistency targets improve semi-
supervised deep learning results. In Proceedings of the 31st
International Conference on Neural Information Processing
Systems, pages 1195–1204, 2017.
Mattias Teye, Hossein Azizpour, and Kevin Smith.
Bayesian
uncertainty estimation for batch normalized deep networks. In
International Conference on Machine Learning, pages 4907–
4916. PMLR, 2018.
Vikas Verma, Kenji Kawaguchi, Alex Lamb, Juho Kannala, Arno
Solin, Yoshua Bengio, and David Lopez-Paz. Interpolation con-
sistency training for semi-supervised learning. Neural Networks,
2021.
Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Daan Wierstra,
et al. Matching networks for one shot learning. Advances in
neural information processing systems, 29:3630–3638, 2016.
Sergey Zagoruyko and Nikos Komodakis. Wide residual networks.
In British Machine Vision Conference 2016. British Machine
Vision Association, 2016.
Dengyong Zhou, Olivier Bousquet, Thomas N Lal, Jason Weston,
and Bernhard Schölkopf. Learning with local and global con-
sistency. In Advances in neural information processing systems,
pages 321–328, 2004.
Xiaojin Zhu, John Lafferty, and Zoubin Ghahramani.
Semi-
supervised learning: From Gaussian ﬁelds to Gaussian pro-
cesses. School of Computer Science, Carnegie Mellon Univer-
sity, 2003.
10