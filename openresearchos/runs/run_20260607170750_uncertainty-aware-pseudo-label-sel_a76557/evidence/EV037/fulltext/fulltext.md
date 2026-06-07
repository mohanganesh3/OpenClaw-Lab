<!-- page 1 -->
Uncertainty-Aware Semi-Supervised Few Shot Segmentation
Soopil Kim, Philip Chikontwe, Sang Hyun Park
Department of Robotics Engineering, DGIST
soopilkim, philipchicco, shpark13135@dgist.ac.kr
Abstract
Few shot segmentation (FSS) aims to learn pixel-level classi-
ﬁcation of a target object in a query image using only a few
annotated support samples. This is challenging as it requires
modeling appearance variations of target objects and the di-
verse visual cues between query and support images with lim-
ited information. To address this problem, we propose a semi-
supervised FSS strategy that leverages additional prototypes
from unlabeled images with uncertainty guided pseudo label
reﬁnement. To obtain reliable prototypes from unlabeled im-
ages, we meta-train a neural network to jointly predict seg-
mentation and estimate the uncertainty of predictions. We
employ the uncertainty estimates to exclude predictions with
high degrees of uncertainty for pseudo label construction to
obtain additional prototypes based on the reﬁned pseudo la-
bels. During inference, query segmentation is predicted using
prototypes from both support and unlabeled images including
low-level features of the query images. Our approach is end-
to-end and can easily supplement existing approaches without
the requirement of additional training to employ unlabeled
samples. Extensive experiments on PASCAL-5i and COCO-
20i demonstrate that our model can effectively remove un-
reliable predictions to reﬁne pseudo labels and signiﬁcantly
improve upon state-of-the-art performances.
Introduction
While deep-learning based segmentation models have
shown
impressive
performance
in
various
applica-
tions (Chen et al. 2017; Ronneberger et al. 2015), the
need for large-scale labeled training data strongly limits
scalability and performance to novel unseen classes and
tasks with a different distribution from the training data.
To address this, few-shot segmentation (FSS) has been
proposed to train deep models in the low data setting
using meta-learning to learn transferable knowledge across
various tasks. FSS models perform dense pixel-level predic-
tion of unseen images (queries) guided by limited labeled
samples (support images). This is challenging due to the
limited data samples for the unseen target objects and the
large diverse appearances between support and queries,
particularly if the training and testing classes present a large
distributional shift.
Copyright © 2022, Association for the Advancement of Artiﬁcial
Intelligence (www.aaai.org). All rights reserved.
For FSS, prototype-based models have been mainly pro-
posed. Here, foreground/background (FG/BG) prototypes
are deﬁned using extracted feature maps from a support
image and its pixel-annotation. To segment unseen query
images, one predicts similarity between the query feature
and the obtained prototypes. For example, (Wang et al.
2019) employs a single prototype strategy but was limited
in the ability to represent different parts of the target ob-
ject. Follow-up works address this by deﬁning multiple pro-
totypes via clustering algorithms to better model feature dis-
crepancy and reduce semantic ambiguity (Liu et al. 2020b;
Li et al. 2021; Yang et al. 2020). Despite the progress, per-
formance is still limited and suffers from scarce context in
support data, particularly if FG/BG appearance in the sup-
port and query vary, leading to inaccurate segmentation.
Recent works have demonstrated that employing FG/BG
features from unlabeled images can mitigate the aforemen-
tioned issues. For example, PPNet (Liu et al. 2020b) pro-
posed a semi-supervised model that supplements part-aware
prototypes using unlabeled images with superpixels. How-
ever, their approach does not make full use of the unlabeled
image information, as they just reﬁne support prototypes
based on attention mechanism between support prototypes
and superpixels from unlabeled images. Consequently, only
a few superpixels with high similarity to the support proto-
types are mainly used, but they do not make good use of
various FG/BG features with lower similarities during infer-
ence.
Instead, we propose a novel semi-supervised approach
that leverages additional prototypes from unlabeled images
via pseudo labels. Using pseudo label predictions from unla-
beled data can further boost model performance, yet inaccu-
rate predictions may equally deteriorate performance. To ad-
dress this, our approach further reﬁnes initial pseudo labels
by excluding unreliable predictions based on uncertainty es-
timation. Based on gaussian assumption on pixel predic-
tions per class following (Kendall and Gal 2017; Lakshmi-
narayanan et al. 2017), we integrate uncertainty estimation
in FSS by training a neural network to model the mean and
variance of outputs from a query and prototype feature pair.
Our intuition is that uncertainty in prototype-based FSS ap-
proaches may arise from varied observations between query
and prototype feature pairs. Thus, we exclude unreliable pre-
dictions from pseudo labels of unlabeled images by only in-
arXiv:2110.08954v1  [cs.CV]  18 Oct 2021


<!-- page 2 -->
cluding those with high mean and low uncertainty predic-
tions as pseudo labels. This also enables the model to learn
better FG/BG features not present in the support data for
improved segmentation. Notably, our approach can estimate
uncertainty without degrading existing prototype-based FSS
models and can be trained end-to-end without an additional
learning process for unlabeled samples. During inference,
we jointly employ the additional prototypes from unlabeled
images with the existing support prototypes to segment an
unseen query image. Our contributions are summarized as
follows:
• We propose an uncertainty estimation method for
prototype-based FSS which captures uncertainty of sim-
ilarity between query feature and prototype pairs. Our
method can reliably quantify uncertainty without degrad-
ing the baseline performance of existing FSS models.
• We propose a semi-supervised FSS method that em-
ploys additional prototypes from unlabeled images using
pseudo labels. Our approach is robust to the number of
unlabeled samples employed despite the varied appear-
ance between samples.
• We empirically demonstrate the beneﬁt of uncertainty-
based pseudo-label reﬁnement in the semi-supervised
scenario with several ablations and report improve-
ments over state-of-the-art on two FSS benchmarks, i.e.,
PASCAL-5i and COCO-20i.
Related Works
Few Shot Semantic Segmentation
Existing few-shot segmentation (FSS) models use the meta-
learning framework via task-based episodic training of sup-
port and query images. OSLSM (Shaban et al. 2017) is the
ﬁrst method to address FSS and predicts weights of a lin-
ear classiﬁer from support data to discriminate the target ob-
ject in a query image. Several follow-up works have pro-
posed to segment the target object based on the similarity
between query feature and class prototypes (Dong and Xing
2018; Wang et al. 2019). Since then, various models have
been proposed to deﬁne better prototypes. To better leverage
different parts of target objects, PPNet (Liu et al. 2020b),
ASGNet (Li et al. 2021), and RPMMs (Yang et al. 2020)
proposed to use multiple prototypes obtained via K-means
clustering, superpixel-guided clustering, and gaussian mix-
ture models, respectively. VPI (Wang et al. 2021) suggested
using a probabilistic prototype rather than deterministic. On
the other hand, others proposed different strategies based
on single prototype to improve performance (Wang et al.
2019; Liu et al. 2020a; Zhang et al. 2019a; Wang et al.
2020; Xie et al. 2021). Notably, CANet (Zhang et al. 2019b)
and PFENet (Tian et al. 2020) argued that FSS models can
predict better segmentation using low-level features from
the encoder. Departing from the meta-learning framework,
(Boudiaf et al. 2021) introduced a transductive approach to
learn task-speciﬁc knowledge for each task with an impres-
sive performance over prior methods.
As for the semi-supervised methods, PPNet also leverages
unlabeled images for FSS and is closely related to our work.
It divides the unlabeled image into superpixels and uses
some superpixels to supplement support prototypes with a
GNN. However, because only a few superpixels similar to
the support prototypes are used, some unlabeled data infor-
mation is discarded. Moreover, it requires a training process
of the GNN to utilize unlabeled images. In this paper, we in-
stead deﬁne additional prototypes from the pseudo label pre-
dictions of unlabeled images while avoiding any additional
training as in PPNet.
Pseudo Labels in Semi-Supervised Segmentation
Pseudo labels are commonly used in semi-supervised learn-
ing methods, e.g., the teacher-student network setting. In
this scenario, a trained teacher network makes predictions
(pseudo-labels) to guide student network training. However,
incorrect/noisy predictions can affect student learning. To
address this, consistency regularization between the teacher
and student networks has been popularly used. (Feng et al.
2020; Ke et al. 2019) Meanwhile, several works suggested
reﬁning pseudo labels using estimated uncertainty. (Sedai
et al. 2019) quantiﬁed uncertainty as entropy of the aver-
aged probabilities obtained by randomly applying dropout
several times following the work of (Gal and Ghahramani
2016), and trained the model with soft labels guided by en-
tropy. However, this method is computationally expensive
with performance highly inﬂuenced by dropout sampling.
Thus, (Li et al. 2020) proposed to estimate uncertainty using
multiple losses from several jigsaw puzzle sub-tasks. On the
other hand, (Saporta et al. 2020) directly used the entropy
of pixel-level probability as uncertainty for an unsupervised
domain adaptation (UDA) task without resorting to prior en-
semble methods. Though impressive, we believe improve-
ments in UDA are due to the use of large-scale data, which
makes entropy estimates feasible. Thus, the direct use of en-
tropy in FSS may be error prone and challenging given a
few data samples. Consequently, we employ an alternative
formulation for uncertainty estimation applicable to the FSS
task for pseudo-label reﬁnement.
Uncertainty Estimation in Neural Network
Although modern neural networks (NNs) show good per-
formance on several tasks, (Guo et al. 2017) reports that
the predicted probability is often different from the actual
observed likelihood, i.e., most of the expected probabilities
are close to 0 or 1, and thus highly overconﬁdent. In order
to quantify uncertainty of model prediction, bayesian neural
networks (BNNs) have been proposed. BNN models calcu-
late posterior probability when a prior of weights is given
and the uncertainty can be quantiﬁed based on variational in-
ference of output. Since the posterior of NNs is intractable,
various approximations have been proposed (Louizos et al.
2016; Blundell et al. 2015). For example, dropout-based
methods are popular and frequently used in several appli-
cations (Gal 2016; Kendall and Gal 2017; Kendall et al.
2018). On the other hand, non-bayesian approaches employ
a gaussian distribution-based method, i.e., where the output
is assumed to follow a gaussian distribution and the model
estimates the mean and variance (Lakshminarayanan et al.


<!-- page 3 -->
2017). Nevertheless, we argue the above approaches are dif-
ﬁcult to correctly optimize NN parameters for the FSS task
with only a small number of data samples. Thus, we con-
sider gaussian process inspired techniques; along this line
of work, gaussian process regression (GPR) can estimate
the mean and variance of gaussian distribution, but requires
a predeﬁned kernel and incurs heavy computation in the
order of O(n + m)3 with n and m being the number of
observations and target data, respectively. To address this,
CNP (Garnelo et al. 2018) trained a neural network that ag-
gregates information from given data samples and estimates
mean and variance. As a result, it could reduce the compu-
tation of GPR and perform ﬂexible tasks such as half-image
completion. Inspired by CNP, we propose an uncertainty es-
timation module in our FSS framework. Our module esti-
mates mean and variance of the gaussian distribution from
a query feature and its nearest prototype. To the best of our
knowledge, we are the ﬁrst to propose an uncertainty esti-
mation method in FSS.
Methods
Problem Setup
A few shot segmentation model FSSθ parameterized by θ
learns to segment an unseen target object in a query im-
age Iq when K support image and label pairs
 
I1
s, L1
s
	
,
 
I2
s, L2
s
	
, ...,
 
IK
s , LK
s
	
are given. The model learns trans-
ferable knowledge from the training task set Ttrain and
is later applied on the test task set Ttest containing novel
classes. Ttrain and Ttest are sampled from the base task set
Tbase where each element has images and pixel-wise anno-
tations of different class, with no overlap between the sets,
i.e., Ttrain ∩Ttest = ∅. Existing standard FSS methods ad-
dress the following supervised learning problem:
Lq = FSSθ(
 
Ik
s , Lk
s
	K
k=1 , Iq).
(1)
In this work, we extend this setting to a semi-supervised
learning problem using unlabeled images. Unlabeled images
are relatively easy to obtain and the scarcity of support im-
ages can be complemented by the unlabeled images. Thus,
given a set of unlabeled samples I1
u, I2
u, ...IM
u , the semi-
supervised FSS problem can be formulated as:
Lq = FSSθ(
 
Ik
s , Lk
s
	K
k=1 , {Im
u , Lm
u }M
m=1 , Iq),
(2)
where M is the number of unlabeled images.
In Fig. 1 and Fig. 2, we present an overview of the train-
ing and inference pipelines of our approach. We jointly
train FSSθ and the uncertainty estimation module follow-
ing a standard meta-training strategy without any unlabeled
images. During inference, we directly employ the trained
model to estimate uncertainty and reﬁne pseudo labels for
prototype generation on the unlabeled image features. Fi-
nally, we employ both the initial support and additional pro-
totypes for segmentation of a query image. In particular,
we cluster per-class support features into several clusters
via K-means clustering. Here, the prototype nearest to the
query feature is selected and the µ and σ2 of the gaussian
distribution are estimated using the nearest prototype-query
𝜇
𝜎
2
  𝑁௦
Query Feature
FG Prototype
BG Prototype
2
  𝑁௦
2
  𝑁௦
Linear Projection
Fully 
Connected 
Layer
𝜎ிீ
𝜎஻ீ
CNN
K-means 
clustering
1
2
 𝑁௦
1
2
𝑁௦
Support Prototypes
…
…
Support Image
Support Label
FG
BG
CNN
Query Image
𝝈
𝝁
Uncertainty 
Estimation
ℒ= −𝒩(𝐿௤; 𝜇, 𝜎ଶ)
Figure 1: Proposed uncertainty-aware FSS model training
strategy. Blue box shows a detail of uncertainty estimation
module.
feature pair to deﬁne pseudo labels. Following, additional
prototypes are deﬁned from the unlabeled images using the
pseudo labels. For precise query segmentation, we leverage:
(i) support and unlabeled prototypes, (ii) low-level features
of support and query images, and (iii) the initial query pre-
diction in a reﬁnement module that learns cross-relations for
improved segmentation.
Prototype-Based Few Shot Segmentation
This work builds upon that of PPNet (Liu et al. 2020b),
a multiple prototype-based approach with a simple design.
However, clear distinctions are shown by the novel modules
and inference strategies introduced in our work.
Formally, part-aware prototypes can be obtained using
K-means clustering on the extracted CNN features, and
later used to segment a query image based on the simi-
larity between extracted query feature and part-aware pro-
totypes, respectively. In particular, given a support image
Is ∈RW ×H×3, we obtain a feature map fs ∈RW ′×H′×C
by feeding Is to an encoder Eθ, where C is channel size,
(W, H) and (W ′, H′) are spatial resolutions of the origi-
nal image and feature maps, respectively. Here, (W ′, H′) is
halved as many times as the number of max-pooling oper-
ations in Eθ, e.g., W ′ = W/8, H′ = H/8 with 3 max-
pooling operations. At the same time, a support label Ls is
resized to the same size as fs and we later use this mask to
separate foreground/background (FG/BG) features into f fg
s
and f bg
s . Using K-means clustering, features are divided into
Ns clusters, where Ns is the number of clusters for per class-

[CAPTION] Figure 1: Proposed uncertainty-aware FSS model training


<!-- page 4 -->
CNN
K-means 
clustering
1
2
 𝑁௦
1
2
𝑁௦
Support Prototypes
…
…
Support Image
Support Label
𝝈
Pseudo Label
Query Prediction
Query Image
FG
BG
Unlabeled Images
CNN
𝝁
1
2
 𝑁௨
1
2
𝑁௨
…
…
K-means 
clustering
Unlabeled Prototypes
Refinement 
Module
CNN
Query Low Level 
Feature
Support Low Level 
FG Prototype
Initial Prediction
ASPP
Prediction
FG
BG
Figure 2: Overview of the proposed semi-supervised FSS inference strategy. Blue box shows a detail of reﬁnement module.
support images. Finally, the mean vectors of the features be-
longing to each cluster are deﬁned as part-aware prototypes
 
˜pi ∈R1×1×C	Ns
i=1:
˜pi =
1
|Gi|
X
j∈Gi
fs,j, fs,j ∈fs,
(3)
where Gi contains indices of pixels of the ith cluster. In ad-
dition, these prototypes are augmented to reﬂect global con-
text based on an attention mechanism. Formally, part-aware
prototypes of a class c from support images are deﬁned as
Pc
s = {pi}Ns
i=1:
pi = ˜pi + λp
Ns
X
j=1∧j̸=i
ai,j ˜pj, ai,j =
d( ˜pi, ˜pj)
P
j̸=i d( ˜pi, ˜pj),
(4)
where λp is a hyperparameter that adjusts the degree of
global context reﬂection and d(·, ·) calculates similarity.
After deﬁning prototypes, a query image Iq is segmented
based on the similarity between the query features fq ∈
RW ′×H′×C and Pc
s. Each pixel fq,i,j ∈R1×1×C from fq
calculates cosine-similarity to the prototypes in Pc
s and se-
lects the nearest one for each class. By aggregating the simi-
larity of each pixel to the nearest, we obtain a similarity map
for each class. We then resize this map to the original in-
put size and obtain a softmax probability map. Subsequently,
each pixel of Iq is classiﬁed to the class of max probability.
Neural Uncertainty Estimation
Our intuition is that uncertainty of prototype-based FSS
models mainly stems from various observations of query
feature and nearest prototype pairs. To leverage uncer-
tainty in the segmentation task which regresses the proba-
bility of pixels belonging to each class, we assume that the
probability follows a gaussian distribution similar to prior
work (Kendall and Gal 2017; Lakshminarayanan et al. 2017;
Garnelo et al. 2018). Formally,
FSSθ(
 
Ik
s , Lk
s
	K
k=1 , Iq) ∼N(µ, σ2).
(5)
We estimate µ ∈RW ×H×2 and σ ∈RW ×H×2 of the gaus-
sian distribution based on the query and nearest prototype
feature pairs. Here, µ is estimated as a similarity-based soft-
max probability map similar to predictions in PPNet and
provides us with a strong baseline. However, as µ is some-
times overconﬁdent with high probability even though the
prototype of a class is not close to the query feature, the
model needs a mechanism to capture uncertainty of simi-
larity between the features to produce more reliable predic-
tions. Thus, we propose an uncertainty estimation module
Uw (see. Fig. 1), that learns a parameter w to estimate σ
from various observations. This module uses FG/BG proto-
types pfg
s and pbg
s and the query feature fq,i,j as inputs when
pfg
s
and pbg
s are the nearest to fq,i,j:
σi,j = Uw(pfg
s , pbg
s , fq,i,j),
(6)
where i ∈[1, W] and j ∈[1, H]. Speciﬁcally, the channel
sizes of FG/BG prototypes and the query features are ﬁrst
reduced by a linear projection layer, then concatenated and
fed into a fully-connected layer block consisting of several
linear layers with ReLU activation. Moreover, σ is predicted
pixel-by-pixel with the ﬁnal uncertainty map for each class
obtained via aggregation of all predictions. The parameters
of Eθ and Uw are simultaneously optimized to minimize the
negative log-likelihood (NLL) loss.
θ, w = argmin
θ,w
−
W
X
i=1
H
X
j=1
N(Lq,i,j; µi,j, σ2
i,j)
(7)
N(Lq,i,j; µi,j, σ2
i,j) =
1
σ
√
2π exp
 
−1
2
 Lq,i,j −µi,j
σi,j
 2 !
(8)

[CAPTION] Figure 2: Overview of the proposed semi-supervised FSS inference strategy. Blue box shows a detail of reﬁnement module.


<!-- page 5 -->
Semi-Supervised Few Shot Segmentation
In this work, our uncertainty-aware semi-supervised FSS
model utilizes pseudo labels of unlabeled images to boost
performance. After training Eθ and Uw, we deﬁne a pseudo
label ˆLu of an unlabeled image given the estimates µ and σ
and deﬁne additional unlabeled data prototypes Pu from ˆLu.
The new prototypes provide additional FG/BG information
and complement the limited representations of Ps by cap-
turing varied object part semantics not presented in support
images. Even though pseudo-labels are commonly used in
semi-supervised approaches, incorrect predictions can dete-
riorate performance, especially in the FSS task where noisy
predictions can lead to using unintended prototypes. To ad-
dress this, we exclude unreliable predictions from pseudo
labels based on uncertainty estimate σ.
Speciﬁcally, given some unlabeled images I1
u, I2
u,..., IM
u
and Ps obtained from support data, µ and σ of the gaus-
sian distribution are estimated for each Iu. Though a pseudo
label can be simply deﬁned as ˆLu = round(µ), it may con-
tain incorrect predictions. Thus, to exclude unreliable pre-
dictions from ˆLu, we deﬁne an uncertainty-aware probabil-
ity µ′ ranging from 0 to 1.0, because both µ and σ have the
same range. Herein,
µ′ = µ × (1 −σ).
(9)
The obtained probability considers both the initial predic-
tion and uncertainty estimate together. Even though the ini-
tial probability of a pixel is high, if its uncertainty is also
high, we can obtain µ′ with lower values and vice-versa.
Therefore, such pixels will not be included in the uncertainty
reﬁned ˆLu = round(µ′). Consequently, the newly deﬁned
pseudo labels only include the pixels with high µ and low σ
values. In this way, we effectively reduce the number of in-
correct predictions in ˆLu. Finally, ˆLu is then used to deﬁne
prototypes from unlabeled images.
Herein, we proceed to deﬁne additional prototypes Pu us-
ing the earlier approach that deﬁnes prototypes for support
samples. After ˆLu is resized to the same size of the feature
map fu, features of FG/BG classes are separated using ˆLu
with Nu clusters obtained via K-means clustering. Follow-
ing, we obtain the mean vector of features belonging to each
cluster and consider it as a prototype. For query image seg-
mentation, we use the entire set of prototypes P = Ps ∪Pu,
and compute the similarity between fq and each prototype
in P to produce a softmax probability map as segmentation.
Implementation Details
Starting from PPNet as a baseline, we observed that predic-
tion boundaries tend to be inaccurate since it uses the re-
duced feature map of the last layer of the encoder. To mit-
igate this, our model additionally trains a reﬁnement mod-
ule R which reﬁnes initial predictions using low-level fea-
tures similar to CANet (Zhang et al. 2019b). R intakes three
inputs, i.e., global low-level support prototype, low-level
query features and initial soft-prediction, which are appro-
priately resized before concatenation. In particular, R reﬁnes
Methods
M
1-shot
0
3
6
12
PPNet w Lsem
51.50
52.14
53.39
51.83
PPNet wo Lsem
51.78
52.37
53.95
52.16
PPNet + PL
51.78
53.52
54.12
54.55
PPNet + PL + Iq
-
53.79
54.42
54.47
PPNet + PLH
51.78
53.26
54.00
54.50
PPNet + PLH + Iq
-
54.16
54.63
54.88
Ours
52.11
53.75
54.60
54.92
Ours + Iq
-
54.88
55.35
55.41
Ours + Iq + R
-
56.27
56.70
57.13
5-shot
PPNet w Lsem
62.00
62.55
63.04
62.55
PPNet wo Lsem
62.02
61.40
62.14
61.36
PPNet + PL
62.02
61.77
62.34
62.34
PPNet + PL + Iq
-
60.82
61.89
62.06
PPNet + PLH
62.02
62.40
62.73
62.79
PPNet + PLH + Iq
-
62.21
62.87
62.90
Ours
62.24
62.49
63.10
63.38
Ours + Iq
-
63.20
63.89
63.90
Ours + Iq + R
-
64.73
65.27
65.38
Table 1: Mean-IoU comparison of the proposed model with
different number of unlabeled images against PPNet on
PASCAL-5i. RN50 was used as a backbone. PL and PLH
denotes a model using pseudo label and modiﬁed pseudo la-
bel using H as uncertainty, respectively. Iq denotes results
using query image as an additional Iu. Boldface represents
the best accuracy without using R.
the predictions via several convolution layers and a subse-
quent ASPP module (Chen et al. 2017) without multiple it-
erations. To effectively use available GPU resources, R was
trained separately.
We closely follow the public implementation of PPNet
and set the hyperparameters of our model as λp = 0.8,
and the number of iterations in K-means clustering as 10.
As the authors reported the best performance with 5 clus-
ters in PPNet, we also used 5 clusters in our model, i.e.,
Ns = Nu = 5.
Experiments
Experimental Setting
We evaluated the proposed model on commonly used FSS
benchmarks, PASCAL-5i (Shaban et al. 2017) and COCO-
20i (Nguyen et al. 2019). PASCAL-5i and COCO-20i have
20 and 80 classes split into 4 folds with 5 and 20 classes
each, respectively. We validated our model on the stan-
dard 4-fold cross-validation setting. Moreover, every image
and its annotation were resized to (417, 417) for training
and testing. ImageNet (Russakovsky et al. 2015) pre-trained
Resnet-50 (RN50) and Resnet-101 (RN101) (He et al. 2016)
backbones were used for the encoder. We follow the evaluta-
tion setting in (Wang et al. 2019) which uses mean-IoU and
binary-IoU as evaluation metrics.
We evaluated our model in both supervised and semi-
supervised 1-way 1,5-shot settings. In the supervised setting,
the model only uses support images to segment a query im-


**[Table p5.1]**
| Methods | M |
| --- | --- |
| 1-shot | 0 3 6 12 |
| PPNet w L sem PPNet wo L sem PPNet + PL PPNet + PL + I q PPNet + PLH PPNet + PLH + I q Ours Ours + I q Ours + I + R q | 51.50 52.14 53.39 51.83 51.78 52.37 53.95 52.16 51.78 53.52 54.12 54.55 - 53.79 54.42 54.47 51.78 53.26 54.00 54.50 - 54.16 54.63 54.88 52.11 53.75 54.60 54.92 - 54.88 55.35 55.41 - 56.27 56.70 57.13 |
| 5-shot |  |
| PPNet w L sem PPNet wo L sem PPNet + PL PPNet + PL + I q PPNet + PLH PPNet + PLH + I q Ours Ours + I q Ours + I + R q | 62.00 62.55 63.04 62.55 62.02 61.40 62.14 61.36 62.02 61.77 62.34 62.34 - 60.82 61.89 62.06 62.02 62.40 62.73 62.79 - 62.21 62.87 62.90 62.24 62.49 63.10 63.38 - 63.20 63.89 63.90 - 64.73 65.27 65.38 |

[CAPTION] Table 1: Mean-IoU comparison of the proposed model with


<!-- page 6 -->
Methods
Eθ
Mean-IoU
Binary-IoU
1-shot 5-shot 1-shot 5-shot
CANet (Zhang et al. 2019b)
RN
50
55.4
57.1
66.2
69.6
PGNet (Zhang et al. 2019a)
56.0
58.5
69.9
70.5
PMMs (Yang et al. 2020)
55.2
56.8
-
-
PPNet* (Liu et al. 2020b)
52.3
63.0
-
-
PFENet (Tian et al. 2020)
60.8
61.9
73.3
73.9
SAGNN (Xie et al. 2021)
62.1
62.8
73.2
73.3
ASGNet (Li et al. 2021)
59.3
64.4
69.2
74.2
Ours
53.6
64.3
71.0
77.3
Ours*
56.7
65.3
71.6
77.4
FWB (Nguyen et al. 2019)
RN
101
56.2
59.9
-
-
DAN (Wang et al. 2020)
58.2
60.5
71.9
72.3
VPI (Wang et al. 2021)
57.3
60.4
-
-
PPNet* (Liu et al. 2020b)
55.2
65.1
-
-
ASGNet (Li et al. 2021)
59.3
64.4
71.7
75.2
Ours
55.2
65.3
72.4
78.5
Ours*
57.4
67.2
73.1
79.5
Table 2: Comparison of the proposed model against state-
of-the-art FSS models on PASCAL-5i. Scores of the com-
parison methods are taken from literature. “*” denotes semi-
supervised result using 6 unlabeled images.
age without unlabeled images, i.e., the estimated µ was used
as the ﬁnal predicted probability. In the semi-supervised set-
ting, 6 unlabeled images were used for comparison against
state-of-the-art methods. As our proposed model deﬁnes
prototypes similar to PPNet, we reproduced PPNet experi-
ments on PASCAL-5i dataset using public code and consid-
ered it as a baseline. Moreover, since one can also use Iq as
part of the unlabeled images set, we equally verify whether
this setting further boosts performance.
Results
Comparison with PPNet
In Table 1, we present the
mean-IoU comparison of our method against the reproduced
baseline PPNet which is the only semi-supervised FSS
model to the best of our knowledge. In the semi-supervised
scenario, PPNet was trained once using M = 6 and tested
with different M, and later compared to our model without
using R. Without unlabeled images, our model reports per-
formance on par with PPNet with slight improvements in
both 1- and 5-shot settings. As opposed to the cross-entropy
loss used in PPNet that forces probabilities to tend to be
either 0.0 or 1.0, our formulation better handles ambigious
predictions by allowing soft probabilities. It is worth noting
that semantic regularization Lsem proposed by PPNet did
not report consistent improvements with different M. Thus,
we omitted Lsem in our framework. Though the best per-
formance reported in the PPNet paper was obtained using
M = 6, improvements were limited as M increases.
Further, we tested a model that uses additional prototypes
based on pseudo labels ˆLu = round(P) where P is the
PPNet prediction. In this case, additional unlabeled proto-
types Pu were obtained using the proposed method. We ob-
served that pseudo labels obtained using PPNet to deﬁne Pu
could not improve performance as incorrect predictions are
included in pseudo labels. We also evaluated whether en-
Methods
Eθ
Mean-IoU
Binary-IoU
1-shot 5-shot 1-shot 5-shot
PPNet* (Liu et al. 2020b)
RN
50
29.0
38.5
-
-
ASGNet (Li et al. 2021)
34.6
42.5
60.4
67.0
Ours
27.9
38.7
65.4
70.5
Ours*
31.1
41.0
66.2
70.6
FWB (Nguyen et al. 2019)
RN
101
21.2
23.7
-
-
PMMs (Yang et al. 2020)
29.6
34.3
-
-
DAN (Wang et al. 2020)
24.4
29.6
62.3
63.9
PFENet (Tian et al. 2020)
32.4
37.4
58.6
61.9
SAGNN (Xie et al. 2021)
37.2
42.7
60.9
63.4
Ours
31.0
40.5
66.7
71.7
Ours*
32.8
43.9
66.6
72.6
Table 3: Comparison of the proposed model against state-
of-the-art FSS models on COCO-20i. Scores of the compar-
ison methods are taken from literature. “*” denotes semi-
supervised result using 6 unlabeled images.
tropy H = −P(P log P) is comparable to the proposed
uncertainty estimation as in (Saporta et al. 2020). Herein,
the pseudo labels predicted by PPNet were modiﬁed using
H instead of σ, i.e., ˆLu = round(P × (1 −H)). In this
case, we noted marginal improvements over vanilla PPNet
w/Lsem. This shows that using H as uncertainty in FSS is
helpful but is non-trivial to remove overconﬁdent incorrect
predictions in the pseudo labels. Interestingly, all models re-
port better performance when Iq is used together with Iu.
Overall, our proposed method reports +3.29% and +1.66%
in 1-shot and 5-shot with M = 12 using Iq, and shows a
continual trend as more Iu samples were employed. Further,
we obtained higher performances using R.
Comparison with State-of-the-Art Models
In Table 2
and Table 3, we report the overall mean-IoU and binary-
IoU comparison of our model against other state-of-the-art
approaches on PASCAL-5i and COCO-20i. All reported
scores of our model include the reﬁnement module R using
Iq as additional Iu. On PASCAL-5i, our model with RN50
beats the baseline (PPNet*) even without using unlabeled
images, i.e., +1.3% mean-IoU in both 1-shot and 5-shot set-
tings (Ours). When 6 unlabeled images were employed, we
observed a further boost, i.e., +2.2% and +1.9% mean-IoU in
1-shot and 5-shot with RN101, with similar observations on
COCO-20i (Ours*). Interestingly, our method achieved the
best scores in the 5-shot setting for both backbones. Though
mean-IoU scores of our 1-shot model was second to that of
SAGNN and DAN with different backbones (Table. 2), we
report the best 1-shot binary-IoU score on COCO-20i. Rel-
atively lower performance of our 1-shot model may be at-
tributed to the weak baseline model. Thus, we believe that
the 1-shot model will achieve higher scores if a better base-
line is used.
Qualitative Results
Fig. 3 shows intermediate results of
our proposed model. We compare the quality of pseudo la-
bels from µ and µ′ considering uncertainty. We normalize
σ between [0, 0.5] to be a heatmap since the pixels with σ
larger than 0.5 are excluded from the pseudo label regardless
of µ using Eq. (9). Results show that σ is high on some am-


**[Table p6.1]**
| Methods | E θ | Mean-IoU |  | Binary-IoU |  |
| --- | --- | --- | --- | --- | --- |
|  |  | 1-shot | 5-shot | 1-shot | 5-shot |
| CANet (Zhang et al. 2019b) PGNet (Zhang et al. 2019a) PMMs (Yang et al. 2020) PPNet* (Liu et al. 2020b) PFENet (Tian et al. 2020) SAGNN (Xie et al. 2021) ASGNet (Li et al. 2021) Ours Ours* | RN 50 | 55.4 56.0 55.2 52.3 60.8 62.1 59.3 53.6 56.7 | 57.1 58.5 56.8 63.0 61.9 62.8 64.4 64.3 65.3 | 66.2 69.9 - - 73.3 73.2 69.2 71.0 71.6 | 69.6 70.5 - - 73.9 73.3 74.2 77.3 77.4 |
| FWB (Nguyen et al. 2019) DAN (Wang et al. 2020) VPI (Wang et al. 2021) PPNet* (Liu et al. 2020b) ASGNet (Li et al. 2021) Ours Ours* | RN 101 | 56.2 58.2 57.3 55.2 59.3 55.2 57.4 | 59.9 60.5 60.4 65.1 64.4 65.3 67.2 | - 71.9 - - 71.7 72.4 73.1 | - 72.3 - - 75.2 78.5 79.5 |


**[Table p6.2]**
| Methods | E θ | Mean-IoU |  | Binary-IoU |  |
| --- | --- | --- | --- | --- | --- |
|  |  | 1-shot | 5-shot | 1-shot | 5-shot |
| PPNet* (Liu et al. 2020b) ASGNet (Li et al. 2021) Ours Ours* | RN 50 | 29.0 34.6 27.9 31.1 | 38.5 42.5 38.7 41.0 | - 60.4 65.4 66.2 | - 67.0 70.5 70.6 |
| FWB (Nguyen et al. 2019) PMMs (Yang et al. 2020) DAN (Wang et al. 2020) PFENet (Tian et al. 2020) SAGNN (Xie et al. 2021) Ours Ours* | RN 101 | 21.2 29.6 24.4 32.4 37.2 31.0 32.8 | 23.7 34.3 29.6 37.4 42.7 40.5 43.9 | - - 62.3 58.6 60.9 66.7 66.6 | - - 63.9 61.9 63.4 71.7 72.6 |

[CAPTION] Table 2: Comparison of the proposed model against state-

[CAPTION] Table 3: Comparison of the proposed model against state-

[CAPTION] Fig. 3 shows intermediate results of


<!-- page 7 -->
Support
Query
GT
𝜇
𝜎௙௚
𝑃𝐿௙௚
𝑃𝐿௕௚
𝜇′௙௚
𝑃𝐿′௙௚
𝑃𝐿′௕௚
Figure 3: Visualization of intermediate results in our proposed 1-shot model on PASCAL-5i. Color in σ changes from blue to
red with higher intensity. PLfg and PLbg denote pseudo labels based on µ while PL′
fg and PL′
bg denote pseudo labels based
on µ′.
biguous pixels due to the limited context in the support data.
For example, in the ﬁrst row, the man’s leg is falsely classi-
ﬁed as FG because its position is near the saddle in the sup-
port image. However, in µ′, we were able to suppress such
spurious activations for better segmentation in ˆLfg
u . These
results verify that our uncertainty-aware learning model is
accurately estimating µ and σ.
Moreover, we show a t-SNE visualization (Van der
Maaten and Hinton 2008) of query features, and prototypes
from support and unlabeled images in Fig. 4. Here, f fg
q
and
f bg
q were separated using the true label. As shown in the ﬁg-
ure, Pu provides rich representations relavent to the query
features in metric space and supplements the limited context
in Ps. In particular, we observed that the decision boundary
(dotted line) moves to include more f fg
q
by utilizing Pfg
u
(orange arrow) while the original decision boundary calcu-
lated based on Pfg
s
(red arrow) and Pbg
s (navy arrow) causes
signiﬁcant errors. Besides, Pbg
u (blue arrow) provides useful
information to classify ambiguous f bg
q
which are far from
Pbg
s . This result shows that newly deﬁned prototypes from
unlabeled images are appropriately used for better predic-
tion.
Conclusion
In this paper, we introduced a novel semi-supervised FSS
model which deﬁnes additional prototypes from unlabeled
images. Our approach also incorporates an uncertainty esti-
mation module tailored for FSS using representations of the
𝒒
𝒃𝒈
𝒒
𝒇𝒈
𝒔
𝒇𝒈
𝒖
𝒇𝒈
𝒔
𝒃𝒈
𝒖
𝒃𝒈
Figure 4: A t-SNE visualization of query features and proto-
types from support and unlabeled images.
query and its nearest prototype pairs. Based on uncertainty
estimation, we show that noisy/overconﬁdent pseudo labels
obtained from unlabeled data can be reﬁned using estimates
for better FSS performance. Extensive quantitative and qual-
itative results on popular benchmarks show the effectiveness
of our approach over state-of-the-art models. We believe that
our semi-supervised learning concept can be generally used
in prototype-based FSS models to further improve perfor-
mance.

[CAPTION] Figure 3: Visualization of intermediate results in our proposed 1-shot model on PASCAL-5i. Color in σ changes from blue to

[CAPTION] Figure 4: A t-SNE visualization of query features and proto-


<!-- page 8 -->
References
Blundell, C.; Cornebise, J.; Kavukcuoglu, K.; and Wierstra,
D. 2015.
Weight uncertainty in neural network.
In In-
ternational Conference on Machine Learning, 1613–1622.
PMLR.
Boudiaf, M.; Kervadec, H.; Masud, Z. I.; Piantanida, P.;
Ben Ayed, I.; and Dolz, J. 2021. Few-Shot Segmentation
Without Meta-Learning: A Good Transductive Inference Is
All You Need? In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, 13979–13988.
Chen, L.-C.; Papandreou, G.; Schroff, F.; and Adam, H.
2017.
Rethinking atrous convolution for semantic image
segmentation. arXiv preprint arXiv:1706.05587.
Dong, N.; and Xing, E. P. 2018. Few-shot semantic segmen-
tation with prototype learning. In BMVC, volume 3.
Feng, Z.; Zhou, Q.; Gu, Q.; Tan, X.; Cheng, G.; Lu, X.; Shi,
J.; and Ma, L. 2020.
Dmt: Dynamic mutual training for
semi-supervised learning. arXiv preprint arXiv:2004.08514.
Gal, Y. 2016. Uncertainty in Deep Learning.
Gal, Y.; and Ghahramani, Z. 2016. Dropout as a bayesian ap-
proximation: Representing model uncertainty in deep learn-
ing. In international conference on machine learning, 1050–
1059. PMLR.
Garnelo, M.; Rosenbaum, D.; Maddison, C.; Ramalho, T.;
Saxton, D.; Shanahan, M.; Teh, Y. W.; Rezende, D.; and
Eslami, S. A. 2018. Conditional neural processes. In In-
ternational Conference on Machine Learning, 1704–1713.
PMLR.
Guo, C.; Pleiss, G.; Sun, Y.; and Weinberger, K. Q. 2017.
On calibration of modern neural networks. In International
Conference on Machine Learning, 1321–1330. PMLR.
He, K.; Zhang, X.; Ren, S.; and Sun, J. 2016. Deep resid-
ual learning for image recognition. In Proceedings of the
IEEE conference on computer vision and pattern recogni-
tion, 770–778.
Ke, Z.; Wang, D.; Yan, Q.; Ren, J.; and Lau, R. W. 2019.
Dual student: Breaking the limits of the teacher in semi-
supervised learning. In Proceedings of the IEEE/CVF In-
ternational Conference on Computer Vision, 6728–6736.
Kendall, A.; and Gal, Y. 2017. What uncertainties do we
need in bayesian deep learning for computer vision? arXiv
preprint arXiv:1703.04977.
Kendall, A.; et al. 2018. Multi-task learning using uncer-
tainty to weigh losses for scene geometry and semantics. In
Proceedings of the IEEE conference on computer vision and
pattern recognition, 7482–7491.
Lakshminarayanan, B.; et al. 2017. Simple and Scalable Pre-
dictive Uncertainty Estimation using Deep Ensembles. Ad-
vances in Neural Information Processing Systems, 30.
Li, G.; Jampani, V.; Sevilla-Lara, L.; Sun, D.; Kim, J.;
and Kim, J. 2021.
Adaptive Prototype Learning and Al-
location for Few-Shot Segmentation.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 8334–8343.
Li, Y.; Chen, J.; Xie, X.; Ma, K.; and Zheng, Y. 2020. Self-
loop uncertainty: A novel pseudo-label for semi-supervised
medical image segmentation. In International Conference
on Medical Image Computing and Computer-Assisted Inter-
vention, 614–623. Springer.
Liu, W.; Zhang, C.; Lin, G.; and Liu, F. 2020a. Crnet: Cross-
reference networks for few-shot segmentation. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 4165–4173.
Liu, Y.; Zhang, X.; Zhang, S.; and He, X. 2020b.
Part-
aware prototype network for few-shot semantic segmenta-
tion. In European Conference on Computer Vision, 142–
158. Springer.
Louizos, C.; et al. 2016.
Structured and efﬁcient varia-
tional deep learning with matrix gaussian posteriors. In In-
ternational Conference on Machine Learning, 1708–1716.
PMLR.
Nguyen, K.; et al. 2019. Feature weighting and boosting for
few-shot segmentation.
In Proceedings of the IEEE/CVF
International Conference on Computer Vision, 622–631.
Ronneberger, O.; et al. 2015. U-net: Convolutional networks
for biomedical image segmentation. In International Con-
ference on Medical image computing and computer-assisted
intervention, 234–241. Springer.
Russakovsky, O.; Deng, J.; Su, H.; Krause, J.; Satheesh, S.;
Ma, S.; Huang, Z.; Karpathy, A.; Khosla, A.; Bernstein, M.;
et al. 2015. Imagenet large scale visual recognition chal-
lenge.
International journal of computer vision, 115(3):
211–252.
Saporta, A.; Vu, T.-H.; Cord, M.; and P´erez, P. 2020.
Esl: Entropy-guided self-supervised learning for domain
adaptation in semantic segmentation.
arXiv preprint
arXiv:2006.08658.
Sedai, S.; Antony, B.; Rai, R.; Jones, K.; Ishikawa, H.;
Schuman, J.; Gadi, W.; and Garnavi, R. 2019. Uncertainty
guided semi-supervised segmentation of retinal layers in
OCT images. In International Conference on Medical Image
Computing and Computer-Assisted Intervention, 282–290.
Springer.
Shaban, A.; Bansal, S.; Liu, Z.; Essa, I.; and Boots, B. 2017.
One-shot learning for semantic segmentation. arXiv preprint
arXiv:1709.03410.
Tian, Z.; Zhao, H.; Shu, M.; Yang, Z.; Li, R.; and Jia, J. 2020.
Prior guided feature enrichment network for few-shot seg-
mentation. IEEE Transactions on Pattern Analysis & Ma-
chine Intelligence, (01): 1–1.
Van der Maaten, L.; and Hinton, G. 2008. Visualizing data
using t-SNE. Journal of machine learning research, 9(11).
Wang, H.; Yang, Y.; Cao, X.; Zhen, X.; Snoek, C.; and Shao,
L. 2021. Variational prototype inference for few-shot se-
mantic segmentation. In Proceedings of the IEEE/CVF Win-
ter Conference on Applications of Computer Vision, 525–
534.
Wang, H.; Zhang, X.; Hu, Y.; Yang, Y.; Cao, X.; and Zhen,
X. 2020. Few-shot semantic segmentation with democratic
attention networks. In Computer Vision–ECCV 2020: 16th


<!-- page 9 -->
European Conference, Glasgow, UK, August 23–28, 2020,
Proceedings, Part XIII 16, 730–746. Springer.
Wang, K.; Liew, J. H.; Zou, Y.; Zhou, D.; and Feng, J. 2019.
Panet: Few-shot image semantic segmentation with proto-
type alignment. In Proceedings of the IEEE/CVF Interna-
tional Conference on Computer Vision, 9197–9206.
Xie, G.-S.; Liu, J.; Xiong, H.; and Shao, L. 2021. Scale-
Aware Graph Neural Network for Few-Shot Semantic Seg-
mentation. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 5475–5484.
Yang, B.; Liu, C.; Li, B.; Jiao, J.; and Ye, Q. 2020. Prototype
mixture models for few-shot semantic segmentation. In Eu-
ropean Conference on Computer Vision, 763–778. Springer.
Zhang, C.; Lin, G.; Liu, F.; Guo, J.; Wu, Q.; and Yao, R.
2019a.
Pyramid graph networks with connection atten-
tions for region-based one-shot semantic segmentation. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision, 9587–9595.
Zhang, C.; Lin, G.; Liu, F.; Yao, R.; and Shen, C. 2019b.
Canet: Class-agnostic segmentation networks with iterative
reﬁnement and attentive few-shot learning. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition, 5217–5226.