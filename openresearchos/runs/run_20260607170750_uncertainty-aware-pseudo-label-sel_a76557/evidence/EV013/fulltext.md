<!-- page 1 -->
1
Exploring Feature Representation Learning for
Semi-supervised Medical Image Segmentation
Huimin Wu, Xiaomeng Li, Member, IEEE, and Kwang-Ting Cheng, Fellow, IEEE
Abstract—This paper presents a simple yet effective two-
stage framework for semi-supervised medical image segmenta-
tion. Unlike prior state-of-the-art semi-supervised segmentation
methods that predominantly rely on pseudo supervision directly
on predictions, such as consistency regularization and pseudo
labeling, our key insight is to explore the feature representation
learning with labeled and unlabeled (i.e., pseudo labeled) images
to regularize a more compact and better-separated feature space,
which paves the way for low-density decision boundary learning
and therefore enhances the segmentation performance. A stage-
adaptive contrastive learning method is proposed, containing a
boundary-aware contrastive loss that takes advantage of the
labeled images in the first stage, as well as a prototype-aware
contrastive loss to optimize both labeled and pseudo labeled
images in the second stage. To obtain more accurate proto-
type estimation, which plays a critical role in prototype-aware
contrastive learning, we present an aleatoric uncertainty-aware
method, namely AUA, to generate higher-quality pseudo labels.
AUA adaptively regularizes prediction consistency by taking
advantage of image ambiguity, which, given its significance, is
under-explored by existing works. Our method achieves the best
results on three public medical image segmentation benchmarks.
Index
Terms—Semi-supervised
segmentation,
contrastive
learning, aleatoric uncertainty, consistency regularization, pseudo
labeling
I. INTRODUCTION
Medical image segmentation is a foundational task for
computer-aided diagnosis and computer-aided surgery. In re-
cent years, considerable efforts have been devoted to designing
neural networks for medical image segmentation, such as U-
Net [1], DenseUNet [2], nnUNet [3], HyperDenseNet [4].
However, training these models requires a large number of la-
beled images. Unlike natural images, the professional expertise
required for pixel-wise manual annotation of medical images
makes such labeling tasks challenging and time-consuming,
H. Wu is with the Department of Computer Science and Engineering, Hong
Kong University of Science and Technology, Hong Kong, SAR, China (e-mail:
hwubl@connect.ust.hk).
X. Li is the corresponding author of this work. X. Li is with the Department
of Electronic and Computer Engineering, The Hong Kong University of
Science and Technology, Hong Kong, SAR, China, and also with The Hong
Kong University of Science and Technology Shenzhen Research Institute,
Shenzhen 518057, China (e-mail: eexmli@ust.hk).
K.-T. Cheng is with the Department of Electronic and Computer Engi-
neering and Department of Computer Science and Engineering, Hong Kong
University of Science and Technology, Hong Kong, SAR, China (e-mail:
timcheng@ust.hk).
This research was partially supported by HKSAR RGC General Research
Fund (GRF) #16203319. This work is supported in part by Foshan HKUST
Projects under FSUST21-HKUST10E & FSUST21-HKUST11E and in part by
the Shenzhen Municipal Central Government Guides Local Science and Tech-
nology Development Special Funded Projects under Grant 2021Szvup139.
(a)
(b)
Labeled data
Unlabeled data
Class centroids
Linear classifier
Figure 1: Two toy examples which (a) visualizes the fea-
ture space of an indiscriminative semi-supervised model, and
(b) visualizes the feature space of a well-clustered semi-
supervised model.
resulting in the difficulty of obtaining a large labeled dataset.
Hence, semi-supervised learning, which enables training using
labeled and unlabeled data, becomes an active research area
for medical image segmentation.
A common assumption of semi-supervised learning is that
the decision boundary should not pass through high-density
regions. Consistency regularization-based techniques [5–7]
achieve a decision boundary at a low-density area by penal-
izing prediction variation under different input perturbations.
Entropy minimization-based methods aim to achieve high-
confidence predictions for unlabeled data either in an explicit
manner [8] or an implicit manner [9–12]. As shown in
Figure 1, an ideal model should pull together data points of the
same class and push apart data points from different classes in
the feature space. As the training set of semi-supervised learn-
ing includes labeled and unlabeled images, it is challenging to
directly optimize the unlabeled images in the feature space
without explicit guidance. We observe that with unlabeled
images, most semi-supervised methods [5–7] can achieve more
accurate segmentation results than the model trained with only
labeled data. Therefore, the pseudo segmentation predicted by
a semi-supervised model on unlabeled data could possibly be
made even more stable and precise.
Motivated by this observation, we present a simple yet
effective two-stage framework for semi-supervised medical
image segmentation with the key idea to explore representation
learning for segmentation from both labeled and unlabeled
images. The first stage aims to generate high-quality pseudo
labels, and the second stage aims to use pseudo labels to
retrain the network to regularize features for both labeled and
unlabeled images. Existing uncertainty-based semi-supervised
methods [5, 13–15] have achieved stunning results by con-
arXiv:2111.10989v2  [cs.CV]  30 Jul 2023

[CAPTION] Figure 1: Two toy examples which (a) visualizes the fea-


<!-- page 2 -->
2
sidering the reliability of the supervision for the unlabeled
images. These methods exploit the epistemic uncertainty, a
kind of uncertainty about the model’s parameters arising from
a lack of data, either in the output space [5, 13, 14] or in
the feature space [14], as guidance for identifying trustworthy
supervision. Medical images are often noisy, and the bound-
aries between tissue types may not be well defined, leading
to a disagreement among human experts [16–18]. However,
aleatoric uncertainty that represents the ambiguity about the
input data and is irreducible by obtaining more data, is ignored
in these methods.
To obtain high-quality pseudo labels for unlabeled images,
we present an Aleatoric Uncertainty Adaptive method, namely
AUA, for semi-supervised medical image segmentation. Under
the framework of the mean teacher model [19], to obtain
reliable target supervision for unlabeled data, instead of es-
timating the model’s epistemic uncertainty [5, 13, 14], we
explore the aleatoric uncertainty of the model for noisy input
data. AUA first measures the spatially correlated aleatoric
uncertainty by modeling a multivariate normal distribution
over the logit space. To effectively utilize unlabeled im-
ages, AUA encourages the prediction consistency between the
teacher model and the student model by adaptively considering
the aleatoric uncertainty for each image. Specifically, the
consistency regularization automatically emphasizes the input
images with lower aleatoric uncertainty, i.e., input images with
less ambiguity.
In the second stage, we retrain the network with pseudo
labels. To effectively regularize feature representation learning
in both stages, we propose stage-adaptive feature regulariza-
tion, including a boundary-aware contrastive loss in the first
stage and a prototype-aware contrastive loss in the second
stage. The main idea of boundary-aware contrastive loss is
to fully leverage labeled images for representation learning.
A straightforward solution is to pull together the pixels to
the same class and push away pixels from different classes
using a contrastive loss. However, medical images usually
contain a large number of pixels. Simply utilizing contrastive
loss would lead to a high computational cost and memory
consumption. To this end, we present a boundary-aware con-
trastive loss, where only randomly sampled pixels from the
segmentation boundary are optimized. In the second stage,
to effectively utilize both labeled and pseudo-labeled images,
i.e., unlabeled images for representation learning, we present
a prototype-aware contrastive loss with each pixel’s feature
pulled closer to its class centroid, i.e., prototype, and pushed
further away from the class centroids it does not belong to. The
main intuition is that the trained model can generate pseudo
labels for unlabeled images in the second stage. Compared
with the boundary-aware contrastive loss, the prototype-aware
contrastive loss better leverages the pseudo labels, especially
those that may not occur at the segmentation boundaries.
In summary, this paper makes the following contributions:
• We introduce stage-adaptive contrastive losses (i.e.,
boundary-aware contrastive loss and prototype-aware
contrastive loss) to regularize a more compact and better-
separated feature space, which eases the learning of a
segmentation decision boundary.
• We present AUA, an aleatoric uncertainty adaptive con-
sistency regularization method that paves the way for
prototype-aware contrastive loss by improving pseudo
label quality and prototype estimation.
• Our method achieves state-of-the-art performance on
three public datasets. The ablation study validates the ef-
fectiveness of our proposed method. Our code is available
at GitHub https://github.com/Huiimin5/AUA now.
II. RELATED WORK
We briefly discuss related works in semi-supervised medical
image segmentation, including pseudo labeling and consis-
tency regularization. We also discuss some techniques related
to contrastive learning and uncertainty estimation.
A. Semi-supervised Medical Image Segmentation
Semi-supervised learning (SSL) refers to training the model
with both labeled and unlabeled images. A wide span of tasks
has been explored such as segmentation [7], classification [20–
23], and crowd counting [24]. For medical image segmenta-
tion, Early work used graph-based methods [25, 26] for semi-
supervised segmentation. Recently, semi-supervised medical
image segmentation has featured deep learning. The existing
methods can be broadly classified into two categories: pseudo
labeling-based [9, 12, 27–29] and consistency regularization-
based methods [5–7, 13, 14, 30–37].
Pseudo Labeling-based Methods. Pseudo labeling-based
methods handle label scarcity by estimating pseudo labels on
unlabeled data and using all the labeled and pseudo labeled
data to train the model. Self-training is one of the most
straightforward solutions [9, 27, 28] and has been extended
to the biomedical domain for segmentation [10, 11, 38]. The
main idea of self-training is that the model is first trained
with labeled data only and then generates pseudo labels for
unlabeled data. By retraining the model with both labeled
and pseudo labeled images, the model performance can be
enhanced. The model can be trained iteratively with these
two processes until the model performance becomes stable
and satisfactory. To reduce the noise in pseudo labels, dif-
ferent methods have been developed, including identifying
trustworthy pseudo labels by uncertainty estimation [10], using
a Conditional Random Field (CRF) [39] to refine pseudo
labels [38] or using pseudo labels only for fine-tuning [11].
In addition to such an offline pseudo label generation strategy,
online self-training methods [40, 41, 12] have been developed
recently where pseudo labels are generated after each forward
propagation and used as immediate supervision.
Another pseudo labeling-based method is co-training [42–
44] where multiple learners are trained and their disagreement
on unlabeled data is exploited for improving the accuracy
of pseudo labels. The basic idea is that each learner could
learn different and complementary information from the other
learners. In some self-training methods, more than one learner
can be used, such as in [12] and the supervision on unlabeled
data is unidirectional. For example, the teacher model [19]
generates pseudo labels to supervise the student model, while
in a dual-model co-training method such as [29], supervision


<!-- page 3 -->
3
is bidirectional. Specifically, each base model’s supervision of
unlabeled data is based on the fused predictions from the other
base models, weighted by the confidence of each model.
However, these methods ignore the class-aware feature
regularization, which is a key focus of this study. We will
demonstrate the importance of feature representation learning
in learning with labeled and pseudo labeled images.
Consistency regularization-based Methods. The goal of
consistency regularization-based semi-supervised methods [19,
45, 46] is to find the model that is not only accurate in
predictions but also invariant to input perturbations to en-
force the decision boundary traverse the low-density region
of the feature space. One line of these methods considers
invariance to input domain perturbations. For example, the
temporal ensembling model [45] achieves promising results by
accumulating soft pseudo labels on randomly perturbated input
images. An extension work with soft pseudo label accumu-
lation guided by epistemic uncertainty was proposed in [13].
When the epistemic uncertainty of the prediction is high, it will
contribute less to pseudo label accumulation. The mean teacher
model [19] achieves invariance to input perturbations by
promoting consistency between the predictions of the teacher
and the student models where input images fed to the teacher
model are added with noises. Extensions have also been made
from the perspective of reliability evaluation [5, 14, 15] to
provide reliable supervision from the teacher model to the stu-
dent model or considering structural information of foreground
objects [30, 15, 47]. In addition to input domain perturbation,
other perturbations that would not change the semantics of
the prediction have also been designed and used to promote
consistency.
For example, consistency among predictions
given by differently designed decoders [31, 32, 48, 49], or
at different scales [6] or with different modalities [33] where
perturbations are beyond the input level is maintained. The
distribution-level consistency between predicted segmentation
maps on labeled data with those on unlabeled ones [36, 50]
has also been proved effective.
Aside from perturbations
that lead to invariance in output, there is another line of
studies [34, 7, 35] that promotes equivariance between the
input and the output because some input space transform,
especially spatial transform such as rotations, should lead to
the same transform in the output space.
Unlike these existing methods that are based on consistency
regularization, our method is a two-stage framework, which
improves the overall framework by regularizing the feature
representation. Moreover, we introduce AUA, an aleatoric
uncertainty-aware method, to represent inherent ambiguities
in medical images and enhance the segmentation performance
by encouraging consistency for images with low ambiguity.
B. Contrastive Learning in Semi-supervised Image Segmenta-
tion.
Note that we exclude self-supervised learning methods
where unlabeled data are only used for task-agnostic pur-
poses, i.e., pretraining such as in [51], even though perfor-
mance under semi-supervised setting is also reported. We only
consider contrastive learning for task-specific use [52–54].
Among these works, only [54]’s goal is to promote inter-
class separation and intra-class compactness. However, in [54],
pseudo labels are obtained from the model trained with labeled
data only, whose performance is inferior to our first stage
model, where pseudo labels are obtained from a model that
takes advantage of consistency regularization on unlabeled
data and feature regularization on labeled data. In [53], inter-
class separation is considered by taking pixels with different
pseudo labels as negative pairs, but intra-class compactness is
ignored since the positive pair is built on the same pixels from
different crops, which essentially is an extension of instance
discrimination for the segmentation task. To the best of our
knowledge, ours is the first study with pixel-level feature
regularization aiming at intra-class compactness and inter-class
separation for semi-supervised medical image segmentation.
C. Uncertainty Estimation in Semi-supervised Medical Image
Segmentation.
Uncertainties generally fall into two categories: epistemic
and aleatoric. Epistemic uncertainty is about a model’s pa-
rameters caused by a lack of data while aleatoric uncer-
tainty is caused by intrinsic ambiguities or randomness of
input data and cannot be reduced by introducing more data.
Early methods measure uncertainty using particle filtering
and Conditional Random Fields (CRFs) [55, 56]. More re-
cently, in Bayesian networks, epistemic uncertainty is usually
estimated with Monte Carlo Dropout [57], which has been
extended for the semi-supervised medical image segmentation
task [5, 13, 14]. Aleatoric uncertainty is estimated either
without considering correlations between pixels [57] or with
a limited ability to model spatial correlation since it is
captured by uncorrelated latent variables from multivariate
normal distribution [16, 17]. Monteiro et al. [18] proposed an
aleatoric uncertainty estimation technique where correlations
between pixels are considered. Given the ubiquitous existence
of noises or ambiguities in medical images, aleatoric uncer-
tainty has been overlooked for semi-supervised medical image
segmentation. In this work, we propose an aleatoric uncertainty
adaptive consistency regularization technique, where correla-
tions between pixels are considered when measuring aleatoric
uncertainty.
III. METHOD
Figure 2 visualizes the overview of the proposed two-
stage framework. The input image is first fed into the AUA
module to get a segmentation model which generates a high-
quality pseudo label. Then, we introduce the stage-adaptive
contrastive learning method, consisting of a boundary-aware
contrastive loss (BCL) on labeled data only in the first stage
and a prototype-aware contrastive loss (PCL) on all data in
the second stage. By sequential training through the first and
second stages, we generate the final segmentation results.
A. Preliminaries: Deterministic Medical Image Segmentation
Here, we consider a C-class segmentation task on 3D
volumes with size H × W × D × C, where H, W, and D

[CAPTION] Figure 2 visualizes the overview of the proposed two-


<!-- page 4 -->
4
Labeled image
Unlabeled image
+Noise
Projection head
Segmentation head
Uncertainty head
A set of realistic predictions on unlabeled data
Input images
Foreground pixel features
Background pixel features
Attract
Attract
Dispel
Dispel
First stage training
Second stage training
…
…
Segmentation model
!"#$%&#'
Sample
A set of realistic predictions on unlabeled data
A set of realistic predictions on labeled data
…
Segmentation model
!(")*#+"
Sample
Attract
Near boundary
foreground pixel features
Attract
Near boundary
background pixel features
Dispel
Feature extraction
Ground truth
Pseudo labels
Segmentation
model
!
Segmentation
model
!(")*#+"
Initialize
Supervised loss
Information flow
of unlabeled data
Information flow
of labeled data
Information flow
of all data
Loss calculation
Training
Inference
Sample
,-.'#/'.)+*,
1-.'#/'.)+*
,2$%3/'.)+*,
12$%3/'.)+*
Class-wise statistics
BCL
PCL
AUA
Minimize
Figure 2: Overview of our method. The proposed loss functions (AUA, BCL and PCL) are boxed out in brown. Firstly, we
propose AUA under a mean-teacher framework which consists of a student model (parameterized by θstudent) and a teacher
model (parameterized by θteacher). AUA is composed of an aleatoric uncertainty estimation module (boxed out in green) and
an adaptive consistency regularization module (boxed out in orange). Secondly, stage-wise contrastive learning is proposed,
which consists of BCL on boundary pixels of labeled data in the first stage, as well as a prototype-aware contrastive loss which
is applied to all pixels in the second stage.
denote the height, width, and depth, respectively. Given an
image x ∈RH×W ×D×C and its ground truth y with the same
size, the loss function of a general segmentation network is
designed to minimize the negative log-likelihood, formulated
as:
Lsup = −log p(y|x) = −log
Z
p(y|g)p(g|x)dg,
(1)
where g denotes logits.
In a deterministic segmentation network, i.e., assuming
p(g|x) = δ(f(g|x; θ)) and independence of each pixel’s pre-
diction on the other, where f is a neural network parameterized
by θ and δ denotes the Dirac delta function, the loss function
in Eq. 1 can be rewritten as:
Lsupce = −log p(y|g) = −
V
X
i=1
C
X
c=1
yiclog softmax(gi)c.
(2)
For simplicity, we use a one-dimensional scalar i to index
each pixel out of a whole set of V = H ∗W ∗D pixels in a
3D volume. The above equation is the cross-entropy function
commonly used in segmentation models.
B. Preliminaries: Aleatoric Uncertainty Estimation for Seg-
mentation
To model inherent ambiguities of input data, we follow [18]
and assume a multi-variant Gaussian distribution around log-
its, i.e., g|x ∼N(µ(x), σ(x)), with µ(x) ∈RH×W ×D×C
and σ(x) ∈R(H×W ×D×C)2. Monte-Carlo integration of S
samples is applied to approximate the intractable integral
operation, leading us from Eq. 1 to:
Lsupau = −log 1
S
S
X
s=1
p(y|g(s))
(3)
= −logsumexpS
s=1 log p(y|g(s)) + log(S).
(4)
The logsumexp (LSE) operation is defined as LSE(l1, ..., lS)=
log(exp(l1)+...+exp(lS)) where ls = log p(y|g(s)). We refer
the calculation of log p(y|g(s)) to Eq. 2, where g(s) is a sample
out of g|x ∼N(µ(x), σ(x)). As pointed out by [18], the full-
rank covariance matrix σ(x) is computationally infeasible, so
we also adopt a low-rank (specifically, r-rank) approximation
defined as:
σ(x) = eF eF T + eD,
(5)
where eF ∈RH×W ×D×C×r denotes the factor part of a low-
rank form of the covariance matrix and eD ∈RH×W ×D×C
denotes the diagonal part. Compared with a full-rank param-
eterization, where (H × W × D × C)2 parameters should be
estimated, which is beyond what a GPU card can accommo-
date unless for very small images, its low-rank approximation
is more computationally feasible since the complexity reduces
from quadratic to linear, i.e., H × W × D × C × r (for eF)
+H × W × D × C (for eD). This approximation might lead
to a compromise of estimated aleatoric uncertainty but still
can bring effective guidance for semi-supervised learning, as
shown in our experiments.
C. Aleatoric Uncertainty Adaptive Consistency Regularization
(AUA)

[CAPTION] Figure 2: Overview of our method. The proposed loss functions (AUA, BCL and PCL) are boxed out in brown. Firstly, we


<!-- page 5 -->
5
It is desirable if the semi-supervised segmentation model
can be aware of its chance of making mistakes on unlabeled
data. We resort to aleatoric uncertainty that captures input
ambiguities, to guide how much the student model should
learn from the teacher model. A consistency regularization
technique adaptive to aleatoric uncertainty is proposed.
Given an unlabeled image xu, the predicted distribution
by the student model parameterized by θs is denoted as
ps = p(yu|xu; θs). Similarly, we can obtain the teacher
model’s prediction pt = p(yu|xu′; θt) over the perturbed
version of the same input xu′ by Gaussian noise injection,
where parameters of the teacher model, denoted as θt, are
updated with an exponential moving average of the parameters
of the student model. The consistency between the teacher
model’s predictions and the student model’s predictions on
unlabeled data is encouraged by minimizing the generalized
energy distance [16, 58], which is defined as:
L
′
con = 2Eys∼ps,yt∼ptd(ys, yt)
−Eys1∼ps,ys2∼psd(ys1, ys2)
−Eyt1∼pt,yt2∼ptd(yt1, yt2).
(6)
To approximate the intractable expectation operation in Eq. 6,
we take S samples out of ps and pt, respectively. The
consistency regularization loss function can be reformulated
as:
Lcon = 2
XS
is=0
XS
it=0 d(y(is)
s
, y(it)
t
)
−
XS
is1=0
XS
is2=0 d(y(is1)
s
, y(is2)
s
)
−
XS
it1=0
XS
it2=0 d(y(it1)
t
, y(it2)
t
),
y(is)
s
, y(is1)
s
, y(is2)
s
∼ps,
y(it)
t
, y(it1)
t
, y(it2)
t
∼pt.
(7)
In Eq. 7, d is defined as the Generalized Dice loss [59]:
d(yi, yj) = 1−
PH×W ×D
k=1
PC
c=1(yi
kc · yj
kc)
PH×W ×D
k=1
PC
c=1(yi
kc · yi
kc) + PH×W ×D
k=1
PC
c=1(yj
kc · yj
kc)
,
(8)
where k indexes each pixel out of a whole set of V = H ∗
W ∗D pixels in a 3D volume and c indexes each class out of
a total of C classes.
The optimum of Eq. 7 is 0, which means the optimum of
the first term is the sum of the last two. This consistency
regularization metric is adaptive to aleatoric uncertainty in the
sense that if the diversity between samples of the student (or
the teacher) model is high, i.e., the values of the last two terms
of Eq. 7 are large, indicating a high aleatoric uncertainty, the
pairwise similarity of samples from the student and the teacher
models, denoted by the first term of Eq. 7, would be less
strictly constrained. On the contrary, on input data where a low
diversity is estimated, implying the aleatoric uncertainty is low
and the model is more likely to generalize well, the student
model automatically learns more from the teacher model by
optimizing the first term to a smaller value.
To summarize, AUA loss is defined as follows:
LAUA = Lsupau + λgLcon,
(9)
where λg is the scaling weight to balance the uncertainty
estimation loss and the generalized energy distance loss.
D. Stage-adaptive feature regularization
We introduce a stage-adaptive feature learning method con-
sisting of a boundary-aware contrastive loss and a prototype-
aware contrastive loss, to enhance the representation learning
with only labeled images and both labeled and pseudo labeled
images, respectively. A natural solution is a contrastive loss
with features of pixels belonging to the same class (i.e.,
both foreground pixels or both background pixels) as positive
pairs and features belonging to different classes (i.e., one
from foreground the other from background) as negative pairs.
This strategy allows us to perform pixel-wise regularization
but consumes memory quadratically to the number of pixels,
so we propose a stage-adaptive contrastive learning method
with these concerns properly handled. To reduce the compu-
tational cost, at the first stage, we only optimize the feature
representation for pixels around the segmentation boundaries,
using a boundary-aware contrastive loss (BCL). At the second
stage, with more accurate pseudo labels on unlabeled data,
we introduce a prototype-aware contrastive loss (PCL) to
fully leverage both labeled and pseudo labeled images for
representation learning.
1) Boundary-aware contrastive learning: As a balance of
benefits of pixel-wise feature level regularization and com-
putational costs, we build positive and negative pairs based
on a random subset of near-boundary pixels, arriving at the
boundary-aware contrastive loss formally defined as:
LBCL =
X
i∈NB
−1
P(i)
X
pi∈P (i)
log
exp(f 1
i · f 1
pi/τ1)
P
o∈O(i)
exp(f 1
i · f 1o /τ1),
(10)
where NB contains indexes of randomly sampled near bound-
ary pixels from an input image, O(i) contains indexes of
the other pixels except pixel i and P(i) contains indexes of
pixels in O(i) belonging to the same class as pixel i. The
feature vectors f 1
i , f 1
o and f 1
pi are obtained from a 3-layer
convolutional projection head, which is connected after the
layer before the last layer. The temperature τ1 is set to be
0.07. By sub-sampling, BCL reduces the computational cost
from (H ×W ×D)2 (i.e., pixel-wise contrastive loss) to NB2.
2) Prototype-aware contrastive learning: In the second
stage, the way to regularize an indiscriminative feature space
as in Figure 1(a) is to encourage each feature to be closer to
any other pixels that share the same label and further away
from the centroid of opposite class so that forming a feature
space in Figure 1(b) is encouraged, which is defined as:
L′
P CL = −1
|P|
X
i∈P
−1
|P(i)\{i}|
X
pi∈P (i)
log
exp(f 2
i · f 2
pi/τ)
exp(f 2
i · f 2
pi/τ) +
1
|N(i)|
P
ni∈N(i) exp(fi · f 2
ni/τ),
(11)
where P contains indexes of all pixels. P(i) and N(i) contains
the indexes of positive pixels, i.e., those sharing the same class,


<!-- page 6 -->
6
and negative pixels, i.e., those with different labels to pixel i
, respectively. Features extracted from the second stage model
are denoted as f 2
∗where ∗can be an index of any pixel.
In [60], by assuming a Gaussian distribution for features
belonging to each class, the computational cost of Eq. 11 can
be reduced from quadratic to linear, leading to a regularization
formulated as:
LP CL = f 2
i
⊤σpf 2
i −
1
|P|
X
i∈P
log
exp( f 2
i ·µp
τ2
+ f 2
i
⊤σpf 2
i
2τ 2
2
)
exp( f 2
i ·µp
τ2
+ f 2
i
⊤σpf 2
i
2τ 2
2
) + exp( f 2
i ·µn
τ2
+ f 2
i
⊤σnf 2
i
2τ 2
2
)
,
(12)
where µp and σp are the mean and covariance matrix of the
positive class to pixel i and similarly, µn and σn are the mean
and covariance matrix of the negative class corresponding to
pixel i. These prototype statistics for each class c are estimated
from the first stage model with an moving average update of
extracted features with each update at t-step formulated as:
µc
t =
Nc
t−1µc
t−1 + nc
tµ
′c
t
Nc
t−1 + nc
t
,
σc
t =
Nc
t−1σc
t−1 + nc
tσ
′c
t
Nc
t−1 + nc
t
+
Nc
t−1nc
t(µc
t−1 −µ
′c
t )(µc
t−1 −µ
′c
t )⊤
(Nc
t−1 + nc
t)2
,
(13)
where Nt−1 denotes the total number of pixels belonging to
class c seen before time step t, and nt denotes the pixel number
of class c in the loaded image at time step t. µ
′c
t and σ
′c
t denote
the mean and covariance, respectively, of features belonging
to class c in images at t. The final prototypes are estimated
after 3000 iterations and the temperature τ2 is set to be 100.
By utilizing prototypes, BCL reduces the computational cost
from (H × W × D)2 (i.e., pixel-wise contrastive loss) to H ×
W × D × C.
E. Stage-wise Training as a Unified Framework
To summarize, in the first stage, the loss function is defined
as:
Lstage1 = LAUA + λcLBCL
(14)
where λc is the scaling weight for BCL loss. To this end,
pseudo labels on unlabeled data with higher quality can
be obtained thanks to joint prediction regularization (with
AUA) and feature regularization (with BCL), which enables
retaining a stronger segmentation model at the second stage
by regularizing both predictions and features over the whole
dataset in a label-aware manner. The loss function in the
second stage is as follows:
Lstage2 = Lsupced + λrLP CL
(15)
where Lsupced is defined as the average of cross-entropy loss
and Dice loss as a common practice in segmentation, which
serves as pseudo labeling and λr is the weight for PCL loss.
IV. EXPERIMENTAL RESULTS
A. Datasets and Preprocessing
Pancreas CT dataset. Pancreas CT dataset [61] is a public
dataset containing 80 scans with a resolution of 512×512
pixels and slice thickness between 1.5 and 2.5 mm. Each image
has a corresponding pixel-wise label, which is annotated by
an expert and verified by a radiologist.
Colon cancer segmentation dataset. Colon cancer dataset
is a subset from Medical Segmentation Decathlon (MSD)
datasets [62], consisting of 190 colon cancer CT volumes.
Pixel-level label annotations are given on 126 CT volumes.
Among these volumes, we randomly split 26 CT volumes as
a test set and use the rest for training.
The Left Atrium (LA) MR dataset. LA dataset contains
100 MR image scans with an isotropic resolution of 0.625 ×
0.625×0.625mm3. This dataset is fully annotated with pixel-
level supervision, among which 80 scans are used for training
and the remaining 20 are used for validation.
Preprocessing. To fairly compare with other methods, we
follow preprocessing in [33] by clipping CT images to a range
of [−125, 275] HU values, resampling images to 1×1×1mm
resolution, center-cropping both raw images and annotations
around foreground area with a margin of 25 voxels and finally
normalizing raw images to zero mean and unit variance. On the
Pancreas dataset, we apply random crop as an augmentation
on the fly, and the Colon dataset is augmented with random
rotation, random flip, and random crop. On both CT datasets,
96×96×96 sub-volumes are randomly cropped from raw data
and fed to the segmentation model for training. On LA dataset,
we apply center crop as well as normalizing the intensities
to zero mean and unit variance for pre-processing. During
training, we adopt random crop to 112 × 113 × 80 for on-
the-fly augmentation.
B. Implementation Details
Environment. All experiments in this work are implemented
in Pytorch 1.6.0 and conducted under python 3.7.4 running on
an NVIDIA TITAN RTX.
Backbone. VNet [63] is used as our backbone where the
last convolutional layer is replaced by a 3D 1×1×1 convo-
lutional layer. On top of that, a projection module and an
aleatoric uncertainty module are built for feature regularization
and aleatoric uncertainty estimation, respectively. Similar to
[54], the projection head constitutes 3 convolutional layers,
each followed by ReLU activations and batch normaliza-
tion, except for the last layer, which is followed by a unit-
normalization layer. The channel size of each convolutional
layer is set as 16. The aleatoric uncertainty module is com-
prised of three 1-layer branches predicting means, covari-
ance factors, and covariance diagonals respectively.
Output
sampling is implemented by calling the sample function of
torch.distributions.LowRankMultivariateNormal 1. The teacher
model parameters are updated by taking a moving average of
the student model parameters. For BCL, the near-boundary
1https://pytorch.org/docs/stable/distributions.html#torch.distributions.
lowrank_multivariate_normal.LowRankMultivariateNormal


<!-- page 7 -->
7
Table I: A comparison with state-of-the-art on pancreas dataset with 20% labeled data. The up arrow implies that the larger
the number, the better the performance. The down arrow implies that a lower number indicates a better performance.
Method
## scans used
Metrics
Labeled
Unlabeled
Dice[%]↑
Jaccard[%]↑
ASD[voxel]↓
95HD[voxel]↓
V-Net
12
0
70.63
56.72
6.29
22.54
V-Net
62
0
81.78
69.65
1.34
5.13
MT [19]
12
50
75.85
61.98
3.40
12.59
DAN [50]
12
50
76.74
63.29
2.97
11.13
Entropy Mini [40]
12
50
75.31
61.73
3.88
11.72
UA-MT [5]
12
50
77.26
63.82
3.06
11.90
CCT [49]
12
50
76.58
62.76
3.69
12.92
SASSNet [36]
12
50
77.66
64.08
3.05
10.93
DTC [33]
12
50
78.27
64.75
2.25
8.36
URPC [6]
12
48
78.83
66.01
1.96
6.92
MC-Net [48]
12
48
79.27
66.24
1.90
7.07
Ours
12
48
79.81
66.82
1.64
5.90
Table II: A comparison with state-of-the-art on Pancreas dataset with 5% labeled data.
Method
## scans used
Metrics
Labeled
Unlabeled
Dice[%]↑
Jaccard[%]↑
ASD[voxel]↓
95HD[voxel]↓
V-Net
3
0
30.74
18.84
6.97
26.45
V-Net
60
0
81.46
69.18
1.31
5.09
MT [19]
3
57
31.09
18.77
28.14
59.22
DAN [50]
3
57
46.37
30.84
16.87
42.89
Entropy Mini[40]
3
57
50.71
35.13
16.14
42.45
UA-MT[5]
3
57
34.46
21.24
25.73
57.40
CCT [49]
3
57
43.89
28.72
20.81
52.58
DTC [33]
3
57
48.47
32.71
17.03
42.61
SASSNet [36]
3
57
51.96
36.03
16.08
45.36
MC-Net [48]
3
57
50.26
35.41
11.27
30.05
URPC [6]
3
57
55.00
39.23
6.11
22.40
Ours
3
57
56.18
40.05
12.47
34.85
pixels are obtained from the difference set of original fore-
ground pixels and resulting foreground pixels after morphol-
ogy dilation of 1-pixel radius.
Training details. Our model is trained with an SGD optimizer
with 0.9 momentum and 0.0001 weight decay for 6000 itera-
tions. A step decay learning rate schedule is applied where the
initial learning rate is set to be 0.01 and dropped by 0.1 every
2500 iterations. For each iteration, a training batch containing
two labeled and two unlabeled sub-volumes is fed to the
proposed model, with each sub-volume randomly cropped with
the size of 96×96×96 for CT volumes and 112×112×80 for
MRI. On the test set, predictions on sub-volumes with the
same size using a sliding window strategy with a stride of
16×16×16 (for CT volumes) or 18×18×4 (for MRI on LA
dataset) are fused to obtain the final results.
Evaluation metrics. We use Dice (DI), Jaccard (JA), the
average surface distance (ASD), and the 95% Hausdorff
Distance (95HD) to evaluate the effectiveness of our semi-
supervised segmentation method. DI and JA mainly measure
the amount of overlap between output segmentation maps and
human annotations. The latter two metrics, ASD and 95HD,
measure surface distance and are more sensitive to errors over
the segmentation boundary.
C. Results on Pancreas Dataset
Our settings. Since the predictions on unlabeled data may
be inaccurate in the early stage of training, we follow common
practices [5, 33] and use a Gaussian ramping up function
λg(t) = 0.15 ∗e−5(1−
t
tmax )2 to control the strength of
consistency regularization, where t denotes current time step
and tmax denotes the maximal training step, i.e., 6000 as
introduced previously. The constant used to scale BCL, i.e.,
λc is set to be 0.09 given 20% labeled data and 0.01 given 5%
labeled data. In the second stage of training, the PCL weight
λr is always set to be 0.1.
Compared methods. Table I shows the results on the
pancreas dataset.
We compare with recent algorithms in-
cluding mean teacher (MT) [19], Deep Adversarial Network
(DAN) [50], Entropy Minimization (Entropy Mini) [40], Un-
certainty Aware Mean Teacher (UAMT) [5], cross-consistency
training method (CCT) [49], shapeaware adversarial network
(SASSNet) [36], Dual-task Consistency (DTC) [33], Uncer-
tainty Rectified Pyramid Consistency (URPC) [6] and Mutual
Consistency Network (MC-Net) [48].
Previous methods are
mainly benchmarked on the first version of the Pancreas
dataset with 12 labeled volumes and 50 unlabeled volumes,
where, however, 2 duplicates of scan #2 are found. In case
some of these three samples are in the training set and the rest
are in the test set after a random split, we use version 2 where
two duplicates are removed, leaving us the same number of
labeled data but 2 less, i.e., 48 unlabeled data. Even under
a more strict scenario, our proposed model achieves the best
performance among existing works.
Results analysis. The first row, i.e., a fully supervised
baseline on the partial dataset, shows the lower bound of the
semi-supervised segmentation methods. Whereas the second
row, i.e., a fully supervised model on a fully labeled dataset,


**[Table p7.1]**
| ## scans used Metrics Method Labeled Unlabeled Dice[%]↑ Jaccard[%]↑ ASD[voxel]↓ 95HD[voxel]↓ |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | Labeled | Unlabeled | Dice[%]↑ | Jaccard[%]↑ | ASD[voxel]↓ | 95HD[voxel]↓ |
| V-Net V-Net | 12 62 | 0 0 | 70.63 81.78 | 56.72 69.65 | 6.29 1.34 | 22.54 5.13 |
| MT [19] DAN [50] Entropy Mini [40] UA-MT [5] CCT [49] SASSNet [36] DTC [33] URPC [6] MC-Net [48] | 12 12 12 12 12 12 12 12 12 | 50 50 50 50 50 50 50 48 48 | 75.85 76.74 75.31 77.26 76.58 77.66 78.27 78.83 79.27 | 61.98 63.29 61.73 63.82 62.76 64.08 64.75 66.01 66.24 | 3.40 2.97 3.88 3.06 3.69 3.05 2.25 1.96 1.90 | 12.59 11.13 11.72 11.90 12.92 10.93 8.36 6.92 7.07 |
| Ours | 12 | 48 | 79.81 | 66.82 | 1.64 | 5.90 |


**[Table p7.2]**
| ## scans used Metrics Method Labeled Unlabeled Dice[%]↑ Jaccard[%]↑ ASD[voxel]↓ 95HD[voxel]↓ |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | Labeled | Unlabeled | Dice[%]↑ | Jaccard[%]↑ | ASD[voxel]↓ | 95HD[voxel]↓ |
| V-Net V-Net | 3 60 | 0 0 | 30.74 81.46 | 18.84 69.18 | 6.97 1.31 | 26.45 5.09 |
| MT [19] DAN [50] Entropy Mini[40] UA-MT[5] CCT [49] DTC [33] SASSNet [36] MC-Net [48] URPC [6] | 3 3 3 3 3 3 3 3 3 | 57 57 57 57 57 57 57 57 57 | 31.09 46.37 50.71 34.46 43.89 48.47 51.96 50.26 55.00 | 18.77 30.84 35.13 21.24 28.72 32.71 36.03 35.41 39.23 | 28.14 16.87 16.14 25.73 20.81 17.03 16.08 11.27 6.11 | 59.22 42.89 42.45 57.40 52.58 42.61 45.36 30.05 22.40 |
| Ours | 3 | 57 | 56.18 | 40.05 | 12.47 | 34.85 |


<!-- page 8 -->
8
Table III: A comparison with state-of-the-art on Colon tumor dataset with 5% labeled data.
Method
# scans used
Metrics
Labeled
Unlabeled
Dice[%]↑
Jaccard[%]↑
ASD[voxel]↓
95HD[voxel]↓
V-Net
5
0
34.07
23.09
10.12
26.52
V-Net
100
0
62.31
49.47
2.14
13.49
MT [19]
5
95
38.64
26.38
14.41
33.08
DAN [50]
5
95
34.02
24.14
14.21
32.42
Entropy Mini [40]
5
95
38.62
26.78
18.02
39.01
UA-MT[5]
5
95
40.61
28.01
15.31
34.92
CCT [49]
5
95
43.74
30.64
11.23
26.21
SASSNet [36]
5
95
41.64
30.07
11.93
28.96
DTC [33]
5
95
43.29
29.84
10.62
26.22
MC-Net [48]
5
95
38.71
26.90
12.19
28.52
URPC [6]
5
95
46.43
33.01
9.31
24.57
Ours
5
95
49.00
35.15
9.04
22.32
Table IV: A comparison with state-of-the-art on LA dataset with 20% labeled data.
Method
# scans used
Metrics
Labeled
Unlabeled
Dice[%]↑
Jaccard[%]↑
ASD[voxel]↓
95HD[voxel]↓
V-Net
16
0
86.03
76.06
3.51
14.26
V-Net
80
0
91.14
83.82
1.52
5.75
MT[19]
16
64
88.42
79.45
2.73
13.07
DAN [50]
16
64
87.52
78.29
2.42
9.01
Entropy Mini[40]
16
64
88.45
79.51
3.72
14.14
UA-MT [5]
16
64
88.88
80.21
2.26
7.32
ICT [64]
16
64
89.02
80.34
1.97
10.38
SASSNet [36]
16
64
89.27
80.82
3.13
8.83
DTC [33]
16
64
89.42
80.98
2.10
7.32
Chaitanya et al. [65]
16
64
89.94
81.82
2.66
7.23
SimCVD [47]
16
64
90.85
83.80
1.86
6.03
MC-Net [48]
16
64
91.07
83.67
1.67
5.84
Ours
16
64
91.08
83.67
1.80
5.60
shows the upper bound performance. We can observe that our
method achieves 79.81% on Dice, surpassing the current state-
of-the-art by 0.54%. Notably, our method is very close to the
fully-supervised model that employs all volumes supervised by
human annotations, showing the effectiveness of the proposed
semi-supervised method.
A more challenging setting with 5% labeled data. To
further validate our method under a more challenging scenario,
we reduce the number of labeled data to only 5% and use the
rest 95% as unlabeled. As shown in Table II, in such a small-
data regime, a performance drop of every semi-supervised
learning method is observed compared to its counterpart in
a big-data regime in Table I where 20% labeled are avail-
able, which confirms common sense. It is observed that our
method consistently outperforms other methods. Specifically,
our method surpasses all the other semi-supervised methods
and outperforms the current state-of-the-art by 1.18% on Dice,
which demonstrates that the effectiveness of our method is
more obvious in a more challenging setting.
Comparison of computational cost. Due to a two-stage
pipeline, our method takes around 2x hours to finish training
compared with existing single-stage works. However, during
inference, the proposed method does not introduce heavy
computational overhead. Existing works use V-Net as the
backbone for inference and their computational time costs are
very similar. As mentioned in Section IV-B, we only append
one more layer after V-Net and the time cost is very close:
4.70 (ours) vs. 4.67 (V-Net), measured by seconds per sample.
It means that in practical use, our method is as efficient as
existing works.
D. Results on Colon Dataset
Table III shows the results on the colon dataset. To get a
result, we set λg(t) to be 0.15∗e−5(1−
t
tmax )2, and the scaling
weight of BCL, i.e., λc is set to be 0.03. In the second stage
of training, the weight PCL is set to be 0.1. We compare our
method with several state-of-the-art methods using 5% data
as labeled and the rest as unlabeled. Again, we tune hyper-
parameters for previous methods so that these methods can
reach the best performance on this dataset. In Table III, by
comparing the second row with Table II, we notice under
a fully supervised setting using a full dataset, the perfor-
mance on the Colon dataset is lower than Pancreas dataset,
indicating Colon dataset is more challenging. By comparing
semi-supervised segmentation methods with a fully supervised
setting using the partial dataset, i.e., the result in the first
row of Table III, we observe stronger performance, showing
that leveraging unlabeled data can improve the segmentation
performance, which confirms common sense. Our method
achieves superior performance compared with all previous
works by a large margin (3.43%), which indicates that our
method can make better use of unlabeled data.
E. Results on LA Dataset
To demonstrate that our method is generalizable to differ-
ent medical image modalities, we also conduct comparative
experiments on the LA dataset, as shown in Table IV. To


**[Table p8.1]**
| # scans used Metrics Method Labeled Unlabeled Dice[%]↑ Jaccard[%]↑ ASD[voxel]↓ 95HD[voxel]↓ |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | Labeled | Unlabeled | Dice[%]↑ | Jaccard[%]↑ | ASD[voxel]↓ | 95HD[voxel]↓ |
| V-Net V-Net | 5 100 | 0 0 | 34.07 62.31 | 23.09 49.47 | 10.12 2.14 | 26.52 13.49 |
| MT [19] DAN [50] Entropy Mini [40] UA-MT[5] CCT [49] SASSNet [36] DTC [33] MC-Net [48] URPC [6] | 5 5 5 5 5 5 5 5 5 | 95 95 95 95 95 95 95 95 95 | 38.64 34.02 38.62 40.61 43.74 41.64 43.29 38.71 46.43 | 26.38 24.14 26.78 28.01 30.64 30.07 29.84 26.90 33.01 | 14.41 14.21 18.02 15.31 11.23 11.93 10.62 12.19 9.31 | 33.08 32.42 39.01 34.92 26.21 28.96 26.22 28.52 24.57 |
| Ours | 5 | 95 | 49.00 | 35.15 | 9.04 | 22.32 |


**[Table p8.2]**
| # scans used Metrics Method Labeled Unlabeled Dice[%]↑ Jaccard[%]↑ ASD[voxel]↓ 95HD[voxel]↓ |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | Labeled | Unlabeled | Dice[%]↑ | Jaccard[%]↑ | ASD[voxel]↓ | 95HD[voxel]↓ |
| V-Net V-Net | 16 80 | 0 0 | 86.03 91.14 | 76.06 83.82 | 3.51 1.52 | 14.26 5.75 |
| MT[19] DAN [50] Entropy Mini[40] UA-MT [5] ICT [64] SASSNet [36] DTC [33] Chaitanya et al. [65] SimCVD [47] MC-Net [48] | 16 16 16 16 16 16 16 16 16 16 | 64 64 64 64 64 64 64 64 64 64 | 88.42 87.52 88.45 88.88 89.02 89.27 89.42 89.94 90.85 91.07 | 79.45 78.29 79.51 80.21 80.34 80.82 80.98 81.82 83.80 83.67 | 2.73 2.42 3.72 2.26 1.97 3.13 2.10 2.66 1.86 1.67 | 13.07 9.01 14.14 7.32 10.38 8.83 7.32 7.23 6.03 5.84 |
| Ours | 16 | 64 | 91.08 | 83.67 | 1.80 | 5.60 |


<!-- page 9 -->
9
Table V: Ablation study on Pancreas dataset. BCL refers to boundary-aware contrastive learning and PCL refers to prototype-
aware contrastive learning. Pseudo labeling refers to directly retraining the network with pseudo labels without PCL.
Method
Metrics
Dice[%] ↑
Jaccard[%] ↑
ASD[voxel]↓
95HD[voxel]↓
Supervised baseline
70.63
56.72
6.29
22.54
AUA
76.13
62.19
2.25
9.35
AUA + BCL (First stage)
77.15
63.34
2.04
7.00
AUA + BCL + Pseudo labeling
79.08
65.91
1.91
6.69
AUA + BCL + Pseudo labeling + PCL (Full)
79.81
66.82
1.64
5.90
Table VI: Ablation study on Colon dataset. BCL refers to boundary-aware contrastive learning and PCL refers to prototype-
aware contrastive learning. Pseudo labeling refers to directly retraining the network with pseudo labels without PCL.
Method
Metrics
Dice[%] ↑
Jaccard[%] ↑
ASD[voxel]↓
95HD[voxel]↓
Supervised baseline
34.07
23.09
10.12
26.52
AUA
42.74
30.20
15.00
35.43
AUA + BCL (First stage)
43.70
30.92
14.74
33.34
AUA + BCL + Pseudo labeling
46.75
33.62
12.39
28.49
AUA + BCL + Pseudo labeling + PCL (Full)
49.00
35.15
9.04
22.32
AUA
Supervised 
baseline
AUA+BCL      
+Pseudo labeling+PCL
AUA+BCL
AUA+BCL 
+Pseudo labeling
Figure 3: A visualized ablation study. Regions highlighted in red are true positive areas, i.e., pixels correctly predicted. Green
and blue regions represent false negatives, i.e., foreground pixels incorrectly predicted as background, and false positives, i.e.,
background pixels incorrectly predicted as foreground.
get a result, we set λg(t) to be 0.15 ∗e−5(1−
t
tmax )2, and the
scaling weight of BCL, i.e., λc is set to be 0.09. In the second
stage of training, the weight PCL is set to 0.1. We compare
with state-of-the-arts benchmarked in [47] using 20% data
as labeled and the rest as unlabeled. As a sanity check, all
semi-supervised methods can outperform a fully-supervised
baseline on the partial dataset (i.e., 16 labeled MR images),
demonstrating a meaningful utilization of unlabeled data. In
Table IV, compared with previous works, the proposed method
achieves the best results, closing the performance gap with a
fully-supervised upper bound (i.e., trained with a full dataset).
F. Ablation Studies
Here we ablate each component of our proposed framework
on the Pancreas dataset with 20% as labeled (Table V) and on
the Colon dataset with 5% as labeled (Table VI). We gradually
add our proposed component and showcase their performances
in terms of four metrics.
Firstly, we validate the effectiveness of the adaptive super-
vision fitting scheme: AUA. On both datasets, as shown in
the second row of Table V and VI, applying AUA achieves
superior performance over the fully-supervised model, i.e., V-
Net. This performance gain mainly comes from its ability
to identify and adaptively learn from trustworthy supervi-
sion. Specifically, AUA automatically estimates aleatoric un-
certainty of input data and down-weights supervision from
low-quality images, so that the student model learns from
more accurate supervision of the teacher model. Secondly,
we demonstrate the effectiveness our stage-aware contrastive
learning. BCL can boost performance on top of AUA further
by a margin of 1.02% and 0.96% on Pancreas and Colon
datasets, respectively, and PCL improves its baseline (as shown
in the fourth row) by 0.73% and 2.25%, respectively. Both
techniques are designed to pull features of pixels belonging to
the same class closer and push features belonging to opposite
classes further, which shapes a more compact (inside each
class) and better separated (across different classes) feature
space, leading to a more robust and effective semi-supervised
method.
To get a qualitative sense of the effectiveness of each
component of our pipeline, in Figure 3, we plot segmen-
tation results by gradually adding the proposed technique.
We illustrate a failure case of the baseline model where the
foreground pixels are mislabeled as background. Adding our
technique one by one is able to recall more foreground pixels
and output segmentation maps with gradually larger overlap
with the ground truth.
Additionally, we ablate the effect of λc, the weight balanc-
ing AUA loss and BCL loss in the first stage of training. Its
values are chosen from 0.03, 0.05, 0.07, and 0.09. Table VII


**[Table p9.1]**
| Metrics Method Dice[%] ↑ Jaccard[%] ↑ ASD[voxel]↓ 95HD[voxel]↓ |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Dice[%] ↑ | Jaccard[%] ↑ | ASD[voxel]↓ | 95HD[voxel]↓ |
| Supervised baseline AUA AUA + BCL (First stage) AUA + BCL + Pseudo labeling AUA + BCL + Pseudo labeling + PCL (Full) | 70.63 76.13 77.15 79.08 79.81 | 56.72 62.19 63.34 65.91 66.82 | 6.29 2.25 2.04 1.91 1.64 | 22.54 9.35 7.00 6.69 5.90 |


**[Table p9.2]**
| Dice[%] ↑ | Jaccard[%] ↑ | ASD[voxel]↓ |
| --- | --- | --- |
| 34.07 42.74 43.70 46.75 49.00 | 23.09 30.20 30.92 33.62 35.15 | 10.12 15.00 14.74 12.39 9.04 |

[CAPTION] Figure 3: A visualized ablation study. Regions highlighted in red are true positive areas, i.e., pixels correctly predicted. Green


<!-- page 10 -->
10
demonstrates that the final results are robust to various choices
of λc.
Table VII: Ablation study on the effect of hyperparameter λc
on the Pancreas dataset.
Method
Metrics
Dice[%] ↑
Jaccard[%] ↑
ASD[voxel]↓
95HD[voxel]↓
0.03
79.26
66.03
1.84
6.45
0.05
79.61
66.49
2.03
7.53
0.07
79.45
66.31
2.04
6.44
0.09
79.81
66.82
1.64
5.90
Finally, we ablate the robustness of the proposed method
to the amount of unlabeled data. in VIII, we demonstrate the
performance of our method by increasing the unlabeled data
number from a small split (i.e., 1/4) to full. We can observe a
growing trend in performance, and it is safe to conjecture that
with more in-domain unlabeled data, our method can obtain
extra performance gain.
Table VIII: Ablation study on an increasing number of unla-
beled data.
#Unlabeled
Metrics
Dice[%]↑
Jaccard[%]↑
ASD[voxel]↓
95HD[voxel]↓
12 (1/4)
76.77
62.95
1.76
8.47
24 (1/2)
78.63
65.37
1.81
6.79
48 (Full)
79.81
66.82
1.64
5.90
G. A Qualitative Comparison with the State-of-the-arts
We visualize the segmentation predictions obtained from
other state-of-the-art methods and ours in Figure 4. We high-
light true positive, false negative, and false positive pixels in
red, green, and blue, respectively. We can observe that for
the other state-of-the-art works, they either achieve a lower
recall, such as MT [19], DAN [50], Entropy Mini [40], UA-
MT [5], SASSNet [36], DTC [33] and MC-Net [48], or suffer
from more false positives, such as DAN [50] and URPC [6].
However, the prediction of our method has a greater overlap
with the ground truth.
V. DISCUSSION
In this paper, to alleviate heavy reliance on pixel-wise
labels, which requires considerable human efforts, we propose
a novel semi-supervised learning method by taking advan-
tage of aleatoric uncertainty estimation and exploring feature
representation learning. Firstly, on top of the mean teacher
framework, we present AUA that estimates each image’s
aleatoric uncertainty and automatically down-weights super-
vision on ambiguous images so that the trained model is
able to generate more reliable pseudo labels. However, image
ambiguity is an under-explored aspect in semi-supervised
learning. Secondly, we explore representation learning and
propose a state-adaptive contrastive learning method. In the
first stage, a boundary-aware contrastive loss is designed to
regularize labeled image features and in the second stage,
we use a prototype-aware contrastive loss to regularize both
labeled and unlabeled features. Superior performance across
Pancreas-CT, Colon cancer, and LA datasets validates the
superior performance of our method as well well its generality
to different data modalities.
This study has comprehensive applicability. Firstly, the pro-
posed method can be used to automate downstream diagnosis.
It is found in recent research [66, 67] that combining segmen-
tation results benefits disease diagnosis classification tasks.
The proposed method allows for training a segmentation model
that achieves satisfactory results without relying on large-scale
human labels and thus can be applied to downstream disease
diagnosis tasks. Secondly, in clinical practice, our method can
serve as another expert for doctors’ reference. For example, a
doctor may take into consideration colon cancer segmentation
results of the proposed method and make a better diagnosis of
cancer staging.
The main limitation of this study is lacking an automatic
mechanism to differentiate incorrect pseudo labels from cor-
rect ones in the second stage. In this work, we put more effort
into generating more accurate pseudo labels prior to their use.
But given pseudo labels, how to make better use is also a non-
trivial question. Online confidence thresholding [68, 69] can
be a potential solution to identify noisy pseudo labels out of
clean ones. In addition, developing a more noisy label-tolerant
loss function on our design could also get an extra performance
gain.
VI. CONCLUSION
This paper presents a simple yet effective two-stage frame-
work for semi-supervised medical image segmentation, with
the key idea of exploring the feature representation from
labeled and unlabeled images. We propose a stage-adaptive
contrastive learning method, including a boundary-aware con-
trastive loss and a prototype-aware contrastive loss. In the
first stage, BCL loss regularizes features by pulling features
sharing the same labels closer and pushing features with
opposite labels further, arriving at a more compact and well-
separated feature space. This loss function, together with AUA,
which adaptively encourage consistency by considering the
ambiguity of medical images, enhances pseudo label quality
after the first stage of training. Improved pseudo labels not
only provide higher quality supervision for the segmentation
head but also generate more accurate prototypes, which allows
PCL to regularize a well-separated feature space further.
Specifically, the feature of each pixel is pulled closer to
its class centroid and pushed away from its opposite class
centroid, which translates to a more accurate segmentation
model. Our method achieves the best results on three public
medical image segmentation benchmarks, and the ablation
study validates the effectiveness of our proposed method. Our
future works include extending this work to different types
of medical data, such as X-ray images, fundus images, and
surgical videos.
REFERENCES
[1] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional
networks for biomedical image segmentation,” in International
Conference on Medical image computing and computer-assisted
intervention, pp. 234–241, Springer, 2015.


**[Table p10.1]**
| Metrics Method Dice[%] ↑ Jaccard[%] ↑ ASD[voxel]↓ 95HD[voxel]↓ |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Dice[%] ↑ | Jaccard[%] ↑ | ASD[voxel]↓ | 95HD[voxel]↓ |
| 0.03 0.05 0.07 0.09 | 79.26 79.61 79.45 79.81 | 66.03 66.49 66.31 66.82 | 1.84 2.03 2.04 1.64 | 6.45 7.53 6.44 5.90 |


**[Table p10.2]**
| Metrics #Unlabeled Dice[%]↑ Jaccard[%]↑ ASD[voxel]↓ 95HD[voxel]↓ |  |  |  |  |
| --- | --- | --- | --- | --- |
|  | Dice[%]↑ | Jaccard[%]↑ | ASD[voxel]↓ | 95HD[voxel]↓ |
| 12 (1/4) 24 (1/2) 48 (Full) | 76.77 78.63 79.81 | 62.95 65.37 66.82 | 1.76 1.81 1.64 | 8.47 6.79 5.90 |


<!-- page 11 -->
11
Entropy Mini [54]
DAN [53]
MC-Net [42]
CCT [55]
URPC [6]
UA-MT [5]
MT [19]
Ours
SASSNet [31]
DTC [28]
Figure 4: A comparison between visualized segmentation maps obtained by the state-of-the-art and our method. Regions
highlighted in red are true positive areas, i.e., pixels correctly predicted. Regions highlighted in green and blue are false
negatives, i.e., foreground pixels incorrectly predicted as background, and false positives, i.e., background pixels incorrectly
predicted as foreground.
[2] X. Li, H. Chen, X. Qi, Q. Dou, C.-W. Fu, and P.-A. Heng, “H-
denseunet: hybrid densely connected unet for liver and tumor
segmentation from ct volumes,” IEEE transactions on medical
imaging, vol. 37, no. 12, pp. 2663–2674, 2018.
[3] F. Isensee, P. F. Jaeger, S. A. Kohl, J. Petersen, and K. H. Maier-
Hein, “nnu-net: a self-configuring method for deep learning-
based biomedical image segmentation,” Nature Methods, pp. 1–
9, 2020.
[4] J. Dolz, K. Gopinath, J. Yuan, H. Lombaert, C. Desrosiers,
and I. B. Ayed, “Hyperdense-net: a hyper-densely connected
cnn for multi-modal image segmentation,” IEEE transactions
on medical imaging, vol. 38, no. 5, pp. 1116–1126, 2018.
[5] L. Yu, S. Wang, X. Li, C.-W. Fu, and P.-A. Heng, “Uncertainty-
aware self-ensembling model for semi-supervised 3d left atrium
segmentation,” in International Conference on Medical Image
Computing and Computer-Assisted Intervention, pp. 605–613,
Springer, 2019.
[6] X. Luo, W. Liao, J. Chen, T. Song, Y. Chen, S. Zhang,
N. Chen, G. Wang, and S. Zhang, “Efficient semi-supervised
gross target volume of nasopharyngeal carcinoma segmentation
via uncertainty rectified pyramid consistency,” MICCAI, 2021.
[7] X. Li, L. Yu, H. Chen, C.-W. Fu, L. Xing, and P.-A. Heng,
“Transformation-consistent self-ensembling model for semisu-
pervised medical image segmentation,” IEEE Transactions on
Neural Networks and Learning Systems, vol. 32, no. 2, pp. 523–
534, 2020.
[8] Y. Grandvalet and Y. Bengio, “Semi-supervised learning by
entropy minimization,” in Proceedings of the 17th Interna-
tional Conference on Neural Information Processing Systems,
pp. 529–536, 2004.
[9] D.-H. Lee et al., “Pseudo-label: The simple and efficient semi-
supervised learning method for deep neural networks,” in Work-
shop on challenges in representation learning, ICML, vol. 3,
2013.
[10] S. Sedai, B. Antony, R. Rai, K. Jones, H. Ishikawa, J. Schu-
man, W. Gadi, and R. Garnavi, “Uncertainty guided semi-
supervised segmentation of retinal layers in oct images,” in
International Conference on Medical Image Computing and
Computer-Assisted Intervention, pp. 282–290, Springer, 2019.
[11] D. Fan, T. Zhou, G. Ji, Y. Zhou, G. Chen, H. Fu, J. Shen,
and L. Shao, “Inf-net: Automatic COVID-19 lung infection
segmentation from CT images,” IEEE Trans. Medical Imaging,
vol. 39, no. 8, pp. 2626–2637, 2020.
[12] S. Reiß, C. Seibold, A. Freytag, E. Rodner, and R. Stiefelhagen,
“Every annotation counts: Multi-label deep supervision for med-
ical image segmentation,” in Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pp. 9532–
9542, 2021.
[13] X. Cao, H. Chen, Y. Li, Y. Peng, S. Wang, and L. Cheng, “Un-
certainty aware temporal-ensembling model for semi-supervised
ABUS mass segmentation,” IEEE Trans. Medical Imaging,
vol. 40, no. 1, pp. 431–443, 2021.
[14] Y. Wang, Y. Zhang, J. Tian, C. Zhong, Z. Shi, Y. Zhang,
and Z. He, “Double-uncertainty weighted method for semi-
supervised learning,” in International Conference on Med-
ical Image Computing and Computer-Assisted Intervention,
pp. 542–551, Springer, 2020.
[15] K. Wang, B. Zhan, C. Zu, X. Wu, J. Zhou, L. Zhou, and
Y. Wang, “Semi-supervised medical image segmentation via a
tripled-uncertainty guided mean teacher model with contrastive
learning,” Medical Image Analysis, vol. 79, p. 102447, 2022.
[16] S. Kohl, B. Romera-Paredes, C. Meyer, J. De Fauw, J. R.
Ledsam, K. Maier-Hein, S. Eslami, D. Jimenez Rezende, and
O. Ronneberger, “A probabilistic u-net for segmentation of
ambiguous images,” Advances in Neural Information Processing
Systems, vol. 31, 2018.
[17] C. F. Baumgartner, K. C. Tezcan, K. Chaitanya, A. M. Hötker,
U. J. Muehlematter, K. Schawkat, A. S. Becker, O. Donati, and
E. Konukoglu, “Phiseg: Capturing uncertainty in medical image
segmentation,” in International Conference on Medical Image
Computing and Computer-Assisted Intervention, pp. 119–127,
Springer, 2019.
[18] M. Monteiro, L. Le Folgoc, D. Coelho de Castro, N. Pawlowski,
B. Marques, K. Kamnitsas, M. van der Wilk, and B. Glocker,
“Stochastic segmentation networks: Modelling spatially corre-
lated aleatoric uncertainty,” in Advances in Neural Information
Processing Systems, vol. 33, pp. 12756–12767, 2020.
[19] A. Tarvainen and H. Valpola, “Mean teachers are better role
models: Weight-averaged consistency targets improve semi-
supervised deep learning results,” in Proceedings of the 31st
International Conference on Neural Information Processing
Systems, pp. 1195–1204, 2017.
[20] Y. Wang, S. Chen, and Z.-H. Zhou, “New semi-supervised
classification method based on modified cluster assumption,”
IEEE Transactions on Neural Networks and Learning Systems,
vol. 23, no. 5, pp. 689–702, 2012.
[21] Y. Duan, Z. Zhao, L. Qi, L. Wang, L. Zhou, Y. Shi, and
Y. Gao, “Mutexmatch: Semi-supervised learning with mutex-
based consistency regularization,” IEEE Transactions on Neural
Networks and Learning Systems, pp. 1–15, 2022.

[CAPTION] Figure 4: A comparison between visualized segmentation maps obtained by the state-of-the-art and our method. Regions


<!-- page 12 -->
12
[22] Y. Yang, X. Tang, X. Zhang, J. Ma, F. Liu, X. Jia, and L. Jiao,
“Semi-supervised multiscale dynamic graph convolution net-
work for hyperspectral image classification,” IEEE Transactions
on Neural Networks and Learning Systems, pp. 1–15, 2022.
[23] M. Gong, H. Zhou, A. K. Qin, W. Liu, and Z. Zhao, “Self-
paced co-training of graph neural networks for semi-supervised
node classification,” IEEE Transactions on Neural Networks and
Learning Systems, pp. 1–14, 2022.
[24] P. Zhu, J. Li, B. Cao, and Q. Hu, “Multi-task credible pseudo-
label learning for semi-supervised crowd counting,” IEEE
Transactions on Neural Networks and Learning Systems, pp. 1–
13, 2023.
[25] H. Su, Z. Yin, S. Huh, T. Kanade, and J. Zhu, “Interactive cell
segmentation based on active and semi-supervised learning,”
IEEE Trans. Medical Imaging, vol. 35, no. 3, pp. 762–777,
2016.
[26] M. Borga, T. Andersson, and O. D. Leinhard, “Semi-supervised
learning of anatomical manifolds for atlas-based segmentation
of medical images,” in 2016 23rd International Conference on
Pattern Recognition (ICPR), pp. 3146–3149, IEEE, 2016.
[27] Q. Xie, M.-T. Luong, E. Hovy, and Q. V. Le, “Self-training with
noisy student improves imagenet classification,” in Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pp. 10687–10698, 2020.
[28] R. Ke, A. I. Avilés-Rivero, S. Pandey, S. Reddy, and C. Schön-
lieb, “A three-stage self-training framework for semi-supervised
semantic segmentation,” CoRR, vol. abs/2012.00827, 2020.
[29] Y. Xia, D. Yang, Z. Yu, F. Liu, J. Cai, L. Yu, Z. Zhu, D. Xu,
A. L. Yuille, and H. Roth, “Uncertainty-aware multi-view co-
training for semi-supervised medical image segmentation and
domain adaptation,” Medical Image Anal., vol. 65, p. 101766,
2020.
[30] W. Hang, W. Feng, S. Liang, L. Yu, Q. Wang, K.-S. Choi,
and J. Qin, “Local and global structure-aware entropy regu-
larized mean teacher model for 3d left atrium segmentation,”
in International Conference on Medical Image Computing and
Computer-Assisted Intervention, pp. 562–571, Springer, 2020.
[31] K. Fang and W.-J. Li, “Dmnet: Difference minimization net-
work for semi-supervised segmentation in medical images,” in
International Conference on Medical Image Computing and
Computer-Assisted Intervention, pp. 532–541, Springer, 2020.
[32] Y. Wu, M. Xu, Z. Ge, J. Cai, and L. Zhang, “Semi-supervised
left atrium segmentation with mutual consistency training,”
CoRR, vol. abs/2103.02911, 2021.
[33] X. Luo, J. Chen, T. Song, and G. Wang, “Semi-supervised
medical image segmentation through dual-task consistency,”
AAAI Conference on Artificial Intelligence, 2021.
[34] X. Li, L. Yu, H. Chen, C.-W. Fu, and P.-A. Heng, “Semi-
supervised skin lesion segmentation via transformation consis-
tent self-ensembling model,” in BMVC, 2018.
[35] G. Bortsova, F. Dubost, L. Hogeweg, I. Katramados, and
M. de Bruijne, “Semi-supervised medical image segmentation
via learning consistency under transformations,” in Interna-
tional Conference on Medical Image Computing and Computer-
Assisted Intervention, pp. 810–818, Springer, 2019.
[36] S. Li, C. Zhang, and X. He, “Shape-aware semi-supervised
3d semantic segmentation for medical images,” in Interna-
tional Conference on Medical Image Computing and Computer-
Assisted Intervention, pp. 552–561, Springer, 2020.
[37] H. Yang, C. Shan, A. F. Kolen, et al., “Deep q-network-driven
catheter segmentation in 3d us by hybrid constrained semi-
supervised learning and dual-unet,” in International Conference
on Medical Image Computing and Computer-Assisted Interven-
tion, pp. 646–655, Springer, 2020.
[38] W. Bai, O. Oktay, M. Sinclair, H. Suzuki, M. Rajchl, G. Tar-
roni, B. Glocker, A. King, P. M. Matthews, and D. Rueckert,
“Semi-supervised learning for network-based cardiac mr image
segmentation,” in International Conference on Medical Image
Computing and Computer-Assisted Intervention, pp. 253–260,
Springer, 2017.
[39] P. Krähenbühl and V. Koltun, “Efficient inference in fully
connected crfs with gaussian edge potentials,” Advances in
neural information processing systems, vol. 24, pp. 109–117,
2011.
[40] T.-H. Vu, H. Jain, M. Bucher, M. Cord, and P. Pérez, “Advent:
Adversarial entropy minimization for domain adaptation in
semantic segmentation,” in Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pp. 2517–
2526, 2019.
[41] Y. Li, J. Chen, X. Xie, K. Ma, and Y. Zheng, “Self-loop uncer-
tainty: A novel pseudo-label for semi-supervised medical image
segmentation,” in International Conference on Medical Image
Computing and Computer-Assisted Intervention, pp. 614–623,
Springer, 2020.
[42] A. Blum and T. Mitchell, “Combining labeled and unlabeled
data with co-training,” in Proceedings of the eleventh annual
conference on Computational learning theory, pp. 92–100,
1998.
[43] S. Qiao, W. Shen, Z. Zhang, B. Wang, and A. Yuille, “Deep co-
training for semi-supervised image recognition,” in Proceedings
of the european conference on computer vision (eccv), pp. 135–
152, 2018.
[44] W. Dong-DongChen and Z.-H. WeiGao, “Tri-net for semi-
supervised deep learning,” in Proceedings of twenty-seventh in-
ternational joint conference on artificial intelligence, pp. 2014–
2020, 2018.
[45] L. Samuli and A. Timo, “Temporal ensembling for semi-
supervised learning,” in International Conference on Learning
Representations (ICLR), vol. 4, p. 6, 2017.
[46] T. Miyato, S. Maeda, M. Koyama, and S. Ishii, “Virtual adver-
sarial training: A regularization method for supervised and semi-
supervised learning,” IEEE Trans. Pattern Anal. Mach. Intell.,
vol. 41, no. 8, pp. 1979–1993, 2019.
[47] C. You, Y. Zhou, R. Zhao, L. Staib, and J. S. Duncan, “Simcvd:
Simple contrastive voxel-wise representation distillation for
semi-supervised medical image segmentation,” IEEE Transac-
tions on Medical Imaging, 2022.
[48] Y. Wu, Z. Ge, D. Zhang, M. Xu, L. Zhang, Y. Xia, and J. Cai,
“Mutual consistency learning for semi-supervised medical im-
age segmentation,” Medical Image Analysis, vol. 81, p. 102530,
2022.
[49] Y. Ouali, C. Hudelot, and M. Tami, “Semi-supervised semantic
segmentation with cross-consistency training,” in Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pp. 12674–12684, 2020.
[50] Y. Zhang, L. Yang, J. Chen, M. Fredericksen, D. P. Hughes, and
D. Z. Chen, “Deep adversarial networks for biomedical image
segmentation utilizing unannotated images,” in International
conference on medical image computing and computer-assisted
intervention, pp. 408–416, Springer, 2017.
[51] K. Chaitanya, E. Erdil, N. Karani, and E. Konukoglu, “Con-
trastive learning of global and local features for medical image
segmentation with limited annotations,” in Advances in Neural
Information Processing Systems, vol. 33, pp. 12546–12558,
Curran Associates, Inc., 2020.
[52] J. Iwasawa, Y. Hirano, and Y. Sugawara, “Label-efficient multi-
task segmentation using contrastive learning,” in Brainlesion:
Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries
- 6th International Workshop, BrainLes 2020, Held in Conjunc-
tion with MICCAI 2020, Lima, Peru, October 4, 2020, Revised
Selected Papers, Part I, pp. 101–110, 2020.
[53] X. Lai, Z. Tian, L. Jiang, S. Liu, H. Zhao, L. Wang, and
J. Jia, “Semi-supervised semantic segmentation with directional
context-aware consistency,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pp. 1205–1214, 2021.
[54] X. Zhao, R. Vemulapalli, P. Mansfield, B. Gong, B. Green,
L. Shapira, and Y. Wu, “Contrastive learning for label-efficient


<!-- page 13 -->
13
semantic
segmentation,”
arXiv
preprint
arXiv:2012.06985,
2020.
[55] A. Blake, R. Curwen, and A. Zisserman, “A framework for
spatiotemporal control in the tracking of visual contours,” In-
ternational Journal of Computer Vision, vol. 11, no. 2, pp. 127–
145, 1993.
[56] X. He, R. S. Zemel, and M. A. Carreira-Perpinán, “Multiscale
conditional random fields for image labeling,” in Proceedings
of the 2004 IEEE Computer Society Conference on Computer
Vision and Pattern Recognition, 2004. CVPR 2004., vol. 2,
pp. II–II, IEEE, 2004.
[57] A. Kendall and Y. Gal, “What uncertainties do we need in
bayesian deep learning for computer vision?,” Advances in
Neural Information Processing Systems, vol. 30, pp. 5574–5584,
2017.
[58] G. J. Székely and M. L. Rizzo, “Energy statistics: A class of
statistics based on distances,” Journal of statistical planning and
inference, vol. 143, no. 8, pp. 1249–1272, 2013.
[59] C. H. Sudre, W. Li, T. Vercauteren, S. Ourselin, and M. J.
Cardoso, “Generalised dice overlap as a deep learning loss func-
tion for highly unbalanced segmentations,” in Deep learning in
medical image analysis and multimodal learning for clinical
decision support, pp. 240–248, Springer, 2017.
[60] S. Li, B. Xie, B. Zang, C. H. Liu, X. Cheng, R. Yang, and
G. Wang, “Semantic distribution-aware contrastive adaptation
for semantic segmentation,” arXiv preprint arXiv:2105.05013,
2021.
[61] H. R. Roth, L. Lu, A. Farag, H.-C. Shin, J. Liu, E. B.
Turkbey, and R. M. Summers, “Deeporgan: Multi-level deep
convolutional networks for automated pancreas segmentation,”
in International conference on medical image computing and
computer-assisted intervention, pp. 556–564, Springer, 2015.
[62] A. L. Simpson, M. Antonelli, S. Bakas, M. Bilello, K. Farahani,
B. Van Ginneken, A. Kopp-Schneider, B. A. Landman, G. Lit-
jens, B. Menze, et al., “A large annotated medical image dataset
for the development and evaluation of segmentation algorithms,”
arXiv preprint arXiv:1902.09063, 2019.
[63] F. Milletari, N. Navab, and S.-A. Ahmadi, “V-net: Fully convo-
lutional neural networks for volumetric medical image segmen-
tation,” in 2016 fourth international conference on 3D vision
(3DV), pp. 565–571, IEEE, 2016.
[64] V. Verma, A. Lamb, J. Kannala, Y. Bengio, and D. Lopez-Paz,
“Interpolation consistency training for semi-supervised learn-
ing,” in International Joint Conference on Artificial Intelligence,
pp. 3635–3641, 2019.
[65] K. Chaitanya, E. Erdil, N. Karani, and E. Konukoglu, “Con-
trastive learning of global and local features for medical image
segmentation with limited annotations,” Advances in Neural
Information Processing Systems, vol. 33, pp. 12546–12558,
2020.
[66] J. Wu, H. Fang, F. Shang, D. Yang, Z. Wang, J. Gao, Y. Yang,
and Y. Xu, “Seatrans: Learning segmentation-assisted diag-
nosis model via transformer,” in International Conference on
Medical Image Computing and Computer-Assisted Intervention,
pp. 677–687, Springer, 2022.
[67] Y. Zhou, X. He, L. Huang, L. Liu, F. Zhu, S. Cui, and
L. Shao, “Collaborative learning of semi-supervised segmen-
tation and classification for medical images,” in Proceedings
of the IEEE/CVF conference on computer vision and pattern
recognition, pp. 2079–2088, 2019.
[68] P. Cascante-Bonilla, F. Tan, Y. Qi, and V. Ordonez, “Curricu-
lum labeling: Revisiting pseudo-labeling for semi-supervised
learning,” in Proceedings of the AAAI Conference on Artificial
Intelligence, vol. 35, pp. 6912–6920, 2021.
[69] L.-Z. Guo and Y.-F. Li, “Class-imbalanced semi-supervised
learning with adaptive thresholding,” in International Confer-
ence on Machine Learning, pp. 8082–8094, PMLR, 2022.