<!-- page 1 -->
IEEE TRANSACTIONS ON MEDICAL IMAGING, VOL. XX, NO. XX, XXXX 2020
1
Less is More: Surgical Phase Recognition from
Timestamp Supervision
Xinpeng Ding, Xinjian Yan, Zixun Wang, Wei Zhao, Jian Zhuang, Xiaowei Xu and Xiaomeng Li
Abstract— Surgical phase recognition is a fundamental
task in computer-assisted surgery systems. Most existing
works are under the supervision of expensive and time-
consuming full annotations, which require the surgeons
to repeat watching videos to ﬁnd the precise start and
end time for a surgical phase. In this paper, we introduce
timestamp supervision for surgical phase recognition to
train the models with timestamp annotations, where the
surgeons are asked to identify only a single timestamp
within the temporal boundary of a phase. This annotation
can signiﬁcantly reduce the manual annotation cost com-
pared to the full annotations.
To make full use of such
timestamp supervisions, we propose a novel method called
uncertainty-aware temporal diffusion (UATD) to generate
trustworthy pseudo labels for training. Our proposed UATD
is motivated by the property of surgical videos, i.e., the
phases are long events consisting of consecutive frames.
To be speciﬁc, UATD diffuses the single labelled timestamp
to its corresponding high conﬁdent ( i.e., low uncertainty)
neighbour frames in an iterative way. Our study uncovers
unique insights of surgical phase recognition with times-
tamp supervisions: 1) timestamp annotation can reduce
74% annotation time compared with the full annotation, and
surgeons tend to annotate those timestamps near the mid-
dle of phases; 2) extensive experiments demonstrate that
our method can achieve competitive results compared with
full supervision methods, while reducing manual annota-
tion cost; 3) less is more in surgical phase recognition, i.e.,
less but discriminative pseudo labels outperform full but
containing ambiguous frames; 4) the proposed UATD can
be used as a plug and play method to clean ambigu-
ous labels near boundaries between phases, and improve
the performance of the current surgical phase recogni-
tion methods. Code and annotations obtained from sur-
geons are available at https://github.com/xmed-lab/
TimeStamp-Surgical.
Index Terms— Surgical phase recognition, timestamp su-
pervision, uncertainty estimation
I. INTRODUCTION
Manuscript received XX XX, 2021. X. Ding, Z. Wang and X. Li are
with the Department of Electronic and Computer Engineering, The Hong
Kong University of Science and Technology, Hong Kong SAR, China.
W. Zhao is with the School of Physics, Beihang University, Beijing,
China, and also with Beihang Hangzhou Innovation Institute Yuhang
Xixi Octagon City, Yuhang District, Hangzhou, China (email: craddy-
wagn@gmail.com; xpding.xidian@gmail.com; zhaow20@buaa.edu.cn;
eexmli@ust.hk)
Copyright (c) 2021 IEEE. Personal use of this material is permitted.
Permission from IEEE must be obtained for all other uses, including
reprinting/republishing this material for advertising or promotional pur-
poses, collecting new collected works for resale or redistribution to
servers or lists, or reuse of any copyrighted component of this work in
other works.
78
80
82
84
86
88
90
92
94
20
30
40
50
60
70
80
90
100
110
26
Full annotation
1.
2.
Annotating precise boundaries
Phase: Preparation
start: 00:00:00.00 end: 00:00:21.00
Timestamp annotation
Once identify a phase
Roll back and clearly 
find start and end time
3.
1.
2.
Phase: Preparation
timestamp: 00:00:15.00
Once identify a phase
Annotating timestamp
Preparation
CalotTriangleDissection
Accuracy(%)
Annotation cost (%)
(a)
(b)
Phase: CalotTriangleDissection
start: 00:00:21.04 end: 00:11:13.00
Phase: CalotTriangleDissection
timestamp: 00:05:20.00
Full annotation
Semi-supervision annotation
Timestamp annotation
EndoNet
TCN
PhaseNet
SurgSSL
LRTD
PSPNet
Casual TCN
TMRNet
TCN
Casual TCN
562.83s
148.53s
Annotation
time
Fig. 1.
(a) Comparison of the full annotation and our proposed
timestamp annotation. When labelling a phase in full annotation, the
annotator needs to roll back and ﬁnd the precise start and end time. In
our timestamp annotation, only a single timestamp is labelled without
identifying the start and end time, which can save annotation cost
and is much faster than the full annotation. We invite two surgeons to
conduct full and timestamp annotations, and record their annotation
times. We ﬁnally observe that they took an average of 562.83 and
148.53 seconds per video for full annotation and timestamp annotation,
respectively.
(b) The trade-off between manual annotation cost and
accuracy for different methods. Compared with existing methods, our
method achieves the competitive performance while using only 26%
manual annotation cost compared with the full supervision.
C
OMPUTER-ASSISTED surgery systems can improve
the surgery’s quality and ensure the patients’ safety in
modern operating rooms [1], [2]. Surgical phase recognition
is one key component of computer-assisted surgery systems,
which aims to predict which phase is occurring at the current
frame [3], [4]. It can be used for automatic indexing of
surgical video databases [5], monitoring surgical process [6],
scheduling surgeons [7] and assessing surgeons’ skills [8].
In recent years, automated surgical phase recognition has
featured deep learning [9]–[11] and has reached promising
recognition performance [5], [12], [13]. Most current surgical
arXiv:2202.08199v2  [cs.CV]  1 Dec 2022


**[Table p1.1]**
| 1. 2. | Roll back and clea find start and end ti | rly me |  | Phase: Preparation start: 00:00:00.00 end: 00:00:21.00 Phase: CalotTriangleDissection start: 00:00:21.04 end: 00:11:13.00 |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |
| Once identify a phase |  | notating precise boundaries |  |  |

[CAPTION] Fig. 1.
(a) Comparison of the full annotation and our proposed


<!-- page 2 -->
2
IEEE TRANSACTIONS ON MEDICAL IMAGING, VOL. XX, NO. XX, XXXX 2020
phase recognition approaches require full annotations from
surgeons, i.e., the surgeons need to ﬁnd the precise start
and end time for a surgical phase. To this end, the surgeon
should repeat watching the video at a very slow speed to
ﬁnd a speciﬁc time for the start of the phase. Then, the
surgeon needs to continue to watch the video and ﬁnd the
precise end time of the phase. As shown in Fig. I (a), this
full annotation is very time-consuming, e.g., surgeons need
to spend an average of 562.83 seconds to annotate a video.
Furthermore, the boundaries between different phases are
usually ambiguous [12].
Due to the subjective of different
surgeons, they would provide inconsistent annotations for the
same video [14].
To address the limitation of the full annotation, this paper
introduces the timestamp supervision to surgical phase recog-
nition which trains the model from the timestamp annotation as
shown in Fig. I (a). In timestamp annotation, the surgeons only
annotate the phase class and a single timestamp for each phase,
instead of start and end times. Once identifying the phase,
the surgeon records the current timestamp (e.g., 00:05:20.00),
no need to roll back and repeat watching the video to ﬁnd
precise start time. After recording this single timestamp, since
there is no need to ﬁnd the end time, the surgeon would
continue to go through the video quickly to ﬁnd another phase.
Hence, the timestamp annotation signiﬁcantly reduces manual
annotation cost compared to the full annotations; see the
detailed annotation analysis in Section IV-B. Given timestamp
supervision, i.e., only a single label for each phase, the total
number of positive frames is quite small, and the naive way
that training with annotated labels may be difﬁcult to learn a
robust model; see results in Table. II. To generate more pseudo
labels, some researchers propose to detect the action changes
between two consecutive labeled frames for action recognition
in natural videos with timestamp supervision [14]. However,
this method displays limited performance to surgical videos
because surgical videos contain more ambiguous boundaries,
leading to the noisy and inconsistent pseudo labels; see
Sec. IV-E for detailed discussion.
To address the above problems, we leverage the the property
of surgical videos to generate more trustworthy pseudo labels
from timestamp supervision. The property we observed is that
phases in the surgical video are long events consisting of
continuous frames, which shows a desirable temporal property
that the closer the frames to the annotated timestamp, the
more likely they are to be classiﬁed to the same label as the
annotated one. Frames far from the annotated timestamp are
difﬁcult to have correct pseudo labels. Based on the above
property, a Uncertainty-Aware Temporal Diffusion (UATD)
module is proposed to diffuse the annotated timestamps to
their adjacent low-uncertainty frames in the temporal axis.
In this way, only frames with high conﬁdence and near
the annotated timestamps would be considered for adding
into pseudo-labels for training. Furthermore, the duration of
the surgical videos generally lasts tens of minutes or even
hours, making it hard to train the model in the end-to-end
manner. Current works [3], [4], [15] generally sample a few
consequent frames from the long videos, and optimize the
combined spatial-temporal model in the end-to-end manner.
This can be implemented in the full annotations, since all
sampled consequent frames have labels. However, in times-
tamp annotation, most of the sampled frames have no labels,
resulting in the imbalance of positive and negative samples.
This imbalance training would degrade the performance; see
details in Table III. To this end, we propose Loop Training
(LP), which optimizes the spatial and temporal model in an
independent and iterative way.
We conduct empirical studies based on the proposed UATD
and LP, and discover important insights of surgical phase
recognition from timestamp supervision as follow: 1) Times-
tamp annotation can reduce 74% annotation time compared
with the full annotation, and surgeons tend to annotate those
timestamps that are near the middle of phases; see details in
Fig. 3. 2) Extensive experiments demonstrate that our method
can achieve competitive results compared with full supervision
methods, while reducing manual annotation cost; see details
in Table I. 3) Less is more in surgical phase recognition, i.e.,
less but discriminative pseudo labels outperform full but
containing ambiguous frames; see details in -Table. I. 4) The
proposed UATD can be used as a plug and play method
to clean ambiguous labels near boundaries between phases,
and improve the performance of the current surgical phase
recognition methods; see details in Fig 10. The reason is that
training with our method would help to decrease intra-class
distance and increase inter-class distance simultaneously; see
details in Table. IX. The main contributions of this work can
be summarized as the following:
• We study surgical phase recognition with a new time-
stamp supervision, which is the the most efﬁcient an-
notation setting in current surgical works. We invite
two surgeons with rich clinical experience to annotate
timestamp annotations and record their annotation time,
and ﬁnd that the timestamp annotation can reduce 74%
annotation cost compared with the full annotation.
• We introduce UATD to generate the trustworthy pseudo
labels from the timestamp annotation, and LP to train the
model from the generated pseudo labels in an iterative
way.
• We conduct in-depth empirical studies of the proposed
UATD and LP based on timestamp supervision, and
discover four deep insights which may boost the future
development of surgical phase recognition. Our code and
timestamp annotations obtained from surgeons will be
released at GitHub upon acceptance.
II. RELATED WORK
A. Surgical Phase Recognition
We broadly classiﬁed related methods for surgical phase
recognition into two categories including fully-supervised
learning and label-efﬁcient learning.
Fully-supervised Learning. In fully-supervised learning, each
frame in a surgical video is labeled. Early works [16]–[18]
use hand-crafted features such as color and texture to perform
recognition, which achieves limited performance and poor
generalization. With the development of neural networks,

[CAPTION] Fig. 3. 2) Extensive experiments demonstrate that our method


<!-- page 3 -->
DING et al.: PREPARATION OF PAPERS FOR IEEE TRANSACTIONS ON MEDICAL IMAGING
3
Video feature 𝐅
Video
𝐗= {𝐱!}!"#
$
(a) Feedforward
Predictions {𝐲3!}!"#
$
𝑓(#)
Spatial feature
extractor
𝑇
𝐷
g(#)
Temporal feature
extractor
(b) Uncertainty-aware temporal diffusion (UATD)
Timestamp annotation 𝐘!%
𝐘5 = {𝐲6!}!"#
$
Uncertainty score
Uncertainty
estimation
Diffusion
𝐗 or 𝐅
Pseudo labels
g(#)
𝐅
UATD
𝐘"
𝑓(# )
𝐘#
𝐗
𝐅
𝐘#
𝐘#
𝐗
𝐘!"
ℎ(#)
𝐗
𝐘"
𝐅
g(#)
𝑓(# )
𝑓(# )
ℎ(#)
ℎ(#)
UATD
(c) Loop training
Forward
Backward
(1)
(2)
(3)
(4)
𝐘!"
𝐘!"
𝜇
Fig. 2.
Overview of our proposed framework. (a) The feedforward process of mapping a video to the phase predictions. A video is ﬁrst fed into a
spatial feature extractor (normally a CNN) to obtain the video feature, followed by a temporal feature extractor to obtain the frame-wise prediction. (b)
Uncertainty-aware temporal diffusion (UATD). To generate trustworthy pseudo labels based on the timestamp supervision, videos or video features
is fed into the uncertainty estimation to obtain the uncertainty scores for each frame. Based on the uncertainty scores, we diffuse the timestamp
annotation to the new pseudo labels. (c) Loop training. Due to the computation and memory cost, the loop training is introduced to optimize the
spatial feature extractor and temporal feature extractor by generated pseudo labels in an iterative way.
recent deep learning based methods achieve the great suc-
cess [3]–[5], [12], [13], [15], [19]–[22]. ZIBNET [23] a state-
preserving Long Short Term Memory (LSTM) to utilize the
long-term evolution of tool usage within complete surgical
phases. EndoNet [5] ﬁrst uses a convolutional neural network
to automatically learn features and prove its effectiveness for
surgical phase recognition. SV-RCNet [3] integrates convolu-
tional neural networks (CNN) and long short-term memory
(LSTM) to learn both spatial and temporal representations
in an end-to-end way. To capture the long-range temporal
relationship, TMRNet [4] introduces a memory bank and
TeCNO [24] uses dilated temporal convolutional network to
get a large receptive ﬁeld. Recently, Yi et al. [20] realize
the negative effect of hard frames and propose data cleansing
and online hard frames mapper to detect and handle them
respectively. Yi et al. [21] ﬁnd that simply applying multi-
stage architecture e.g. multi-stage TCN makes the reﬁnement
fall short and thus design not end-to-end training manner to
alleviate this problem. OperA [25] leverages attention weight
to yield further insights into the decision-making process.
Trans-SVNet [13] proposes a hybrid embedding aggregation
Transformer to fuse spatial and temporal embedding. Ding et
al. [12] emphasize the importance of segment-level semantics
and extract semantic-consistent segments to reﬁne the erro-
neous predictions. Notably, some related methods [5], [15],
[24] utilize additional tool presence labels to perform a multi-
task learning to facilitate surgical phase recognition.
Semi-supervised Learning. Despite the great success the
above methods get, they require a large amount of annotated
videos, which is very costly. some researchers [26]–[30]
explore the methods for semi-supervision, where only parts
of videos in the dataset are fully annotated, and others are
unlabelled. For example, LRTD [29] use active learning to
this context. It captures the long-range temporal dependency
among continuous frames in the unlabeled data pool and se-
lects the clips with weak dependencies to annotate. Yengera et
al. [27] introduce self-supervised pre-training ensuring all
available laparoscopic videos can be utilized. Yu et al. [26]
propose a teacher/student approach where the teacher is trained
on a small set of labeled videos and generates pseudo labels
on the rest of unlabeled videos for student model learning.
Furthermore, SurgSSL [30] uses consistency regularization
and pseudo labeling to leverage the knowledge in unlabeled
data, which progressively leverages the inherent knowledge
held in the unlabeled data to a larger extent.
Comparison of the manual annotation cost for different
supervision setting. Here, we compare our proposed times-
tamp annotation compared with the above methods including
full supervision methods and semi-supervision methods. In
full supervised methods [12], [24], annotators are required to
repeat watching the video and roll back to ﬁnd the precise start
and end time for each phase, which is vert time-consuming. As
shown in Fig. I (a), the average annotation time of each video
for full supervision is 562.83s. In semi-supervision [26]–[30],
the authors are required to only label full annotations for a
few parts of all videos. Generally, in semi-supervised surgical
phase recognition methods, 50% of videos are required to be
annotated for achieving the competitive results compared with
full supervision methods, as shown in Fig. I (b). However,
the annotation times of the introduced timestamp supervision
is only 148.53s for each video, i.e., 26% annotation time of
the full supervision, and achieve the competitive results. For
clarity, using the same network TCN [14], our methods achieve
91.9% accuracy in Cholec80 with only 26% annotation time,

[CAPTION] Fig. 2.
Overview of our proposed framework. (a) The feedforward process of mapping a video to the phase predictions. A video is ﬁrst fed into a


<!-- page 4 -->
4
IEEE TRANSACTIONS ON MEDICAL IMAGING, VOL. XX, NO. XX, XXXX 2020
while the full supervision achieves 91.1% accuracy using
100% annotation time. Meanwhile, the SOTA semi-supervised
method SurgSSL [30] achieves 87.0% accuracy using 30%
annotation times. Hence, our proposed method is the best
trade-off between accuracy and manual annotation cost.
B. Weak Supervision for Video Understanding
Weakly supervision has received widespread attention in
some video understanding tasks, such as temporal action lo-
calization [31]–[37] and action segmentation [38]–[40]. Some
of them use video-level supervision, i.e., a set of action
categories, while some use transcript-level supervision, i.e.,
an ordered list of actions. For example, Richard et al. [41]
leverage text-based grammar from unordered action sets. Al-
though they signiﬁcantly reduce the annotation effort, the
performance is quite limited. To trade-off the annotation-
efﬁcient and performance, timestamp supervision [14], [42]–
[44] is proposed for action recognition. For example, SF-
Net [44] designs an action frame mining and a background
frame mining strategy to introduce more negative frames into
the training process. However, the above methods aiming at
temporal action localization task generate very limited pseudo
labels, is not suitable for surgical phase recognition, i.e.,
frame-wise recognition. In the action segmentation task, to
generate frame-wise pseudo labels, Li et al. [14] detect the
action change between two consecutive timestamps by stamp-
to-stamp energy function and generate full pseudo labels.
However, in surgical videos, the frames near boundaries are
generally ambiguous, the generated pseudo labels may be
noisy annotations, which degrades the performance. Compared
with previous approaches, our proposed method generates as
many conﬁdent pseudo labels as possible by considering the
temporal relationships among frames, while discarding pseudo
labels with a large uncertainty.
C. Uncertainty Estimation
In deep learning, neural networks may generate false pre-
diction with a high probability, which is called epistemic un-
certainty resulting from the model itself [45]. To estimate the
uncertainty of the deep networks, Monte Carlo Dropout [46]
is proposed to approximate the posterior distribution for un-
certainty estimation. Ensembles [47] trains multiple networks
independently on the entire dataset using random, and the
predictions of multiple networks are averaged over an en-
semble. Follow-up researchers majorly focus on improving
the quality of the predicted uncertainty scores by inference-
based methods [48], [49] or auto-encoder based methods [50],
[51]. Estimation of uncertainty has also been investigated
for medical image classiﬁcation and segmentation. Laves et
al. [52] leverages Monte Carlo dropout at test time, and shows
that error prediction is correlated with higher uncertainty in
OCT classiﬁcation. Leibig [53] uses Monte Carlo dropout
to conduct uncertainty estimation and shows that uncertainty
informed decision can improve the diagnostic performance.
Wang et al [54] utilize Monte Carlo dropout and test data
augmentation to reduce overconﬁdent error predictions in 3D
brain tumor and 2D brain segmentation. Different from current
methods [53], [54] that directly use Monte Carlo Dropout to
estimate each sample individually, in our proposed UATD, the
uncertainty of each frame is estimated based on the relation of
itself, its nearby timestamp annotations and its adjacent frames
in the temporal axis, which is motivated by the property of the
surgical phase.
III. METHOD
A. Problem Deﬁnition
Let X = {xt}T
t=1 be a surgical video with T frames, where
xt is the t-th frame. Each surgical video is divided into several
phases, and there is no overlapping among phases. Our goal is
to learn a spatial feature extractor network f(·) and a temporal
feature extractor g(·) that maps the frame xt to a phase label,
which is presented in Fig. 2 (a). In the full supervision, the
frame-wise labels Y = {y1, . . . , yT } are available. However,
in our timestamp supervision, given a video consisting of N
phases, where N << T, only a single timestamp in each
phase are annotated as Yts = {yt1, . . . , ytN }, where ti is in
the i-th phase, yti ∈{1, 2, . . . , C}, and C is the total number
of classes.
To perform surgical phase recognition with timestamp su-
pervision, we propose an uncertainty-aware temporal diffusion
(UATD) to generate trustworthy pseudo labels, denoted as Y,
from the timestamp supervision Yts to optimize f(·) and g(·).
The proposed UATD is shown in Fig. 2 (b); see Sec. III-
B for details. Furthermore, we introduce the loop training,
which optimizes f(·) and g(·) in an iterative way to reduce
the memory cost and imbalance optimization; see Fig. 2 (c)
and Section III-C for details.
B. Uncertainty-aware Temporal Diffusion
In timestamp supervision Yts, i.e., only a single label for
each phase, the total number of positive frames is quite small
and may be difﬁcult to learn a robust model. Although we
do not have full annotations, it is clear that the phases are
long events consisting of consecutive frames. Motivated by this
property of surgical videos, we propose the uncertainty-aware
temporal diffusion (UATD) to diffuse the single labelled frame
to its corresponding high conﬁdent (i.e., low uncertainty)
neighbour frames. In this way, we can introduce more frames
acted as pseudo labels into the training process. Furthermore,
the diffusion of frames is stopped by low conﬁdent frames,
which can avoid the ambiguous annotations. The proposed
UATD consists two components: uncertainty estimation and
temporal diffusion. In the following, we describe the two
components respectively.
Uncertainty estimation. In UATD, we ﬁrst need to estimated
the uncertainty of each frame to ﬁnd the high conﬁdent ones
for the single annotated frame. To this end, we introduce
Monte Carlo Dropout [46], a simple yet efﬁcient way, to eval-
uate the uncertainty of each frame. In Monte Carlo Dropout,
given a input denoted as z and a network denoted as o(·), we
feed z into o(·) with different dropout K times and obtain a
set of class probabilities. This process can be formulated as:
P = {pk = o(z)}K
k=1,
(1)


<!-- page 5 -->
DING et al.: PREPARATION OF PAPERS FOR IEEE TRANSACTIONS ON MEDICAL IMAGING
5
Algorithm 1 Temporal Diffusion
Input:Uncertainty scores {µt}T
t=1, prediction ˆY = {ˆyt}T
t=1,
timestamp annotation Yts
=
{yt1, . . . , ytN }, uncertainty
threshold τ.
Output:Pseudo labels Y = {yt}T
t=1 for the next iteration.
1: ▷Diffusion for each phase
2: for i = 1 to N do
3:
▷Diffusion for the left side
4:
for t = ti−1 to ti do
5:
yt = ˆyt · 1(ut < τ) · 1(ˆyt = yti)
6:
end for
7:
▷Diffusion for the right side
8:
for t = ti to ti+1 do
9:
yt = ˆyt · 1(ut < τ) · 1(ˆyt = yti)
10:
end for
11: end for
where pk ∈RCand P ∈RK×C, C is the total number
classes. Then, we average these K vectors of probability,
which can be formulated as µ(P) ∈RC, where µ(·) is the
mean function. After that, we obtain the class label for the
input by:
c = argmax µ(P).
(2)
Finally, we use the standard deviation to measure the uncer-
tainty of the obtain class label, i.e., c, which can be formulated
as:
u = σ (Pc),
(3)
where u is the uncertainty score for o(·) with the input of z.
The higher u indicates that the model o(·) predicts z to class
c with lower conﬁdence, and vice versa. In this paper, we
need evaluate the uncertainty of both the spatial and temporal
feature extractors, which are deﬁned in Section III-A.
To conduct the uncertainty estimation for the spatial feature
extractor f(·), we add an extra classiﬁcation head h(·) to
f(·) as shown in Fig. 2 (c), denoted as h(f(·)) to obtain the
classiﬁcation prediction for each frame xt. Let o(·) = h(f(·))
and z = xt, and then we can obtain the uncertainty score µt for
each frame xt by using Eq. 1 to Eq. 3. Similarly, to conduct the
uncertainty estimation for the spatial feature extractor g(·), we
can easily set o(·) = g(·) and z = ft, obtaining the uncertainty
score for each frame feature ft.
Temporal diffusion. After obtaining the uncertainty score µt,
we use the temporal diffusion module to diffuse the current
labels to more pseudo labels for training in the next iteration;
see the iterative training details in Sec. III-C. To be speciﬁc,
we treat the labeled frames as anchors and start diffusion from
anchors to the adjacent frames on either sides of them in
temporal dimension, which is illustrated in Algorithm 1. By
the temporal diffusion, one frame would be introduced into
next iteration training only if the uncertainty score of it is
lower than a threshold τ and the predicted class label equals
to its nearby timestamp frame. In this way, the generated
pseudo label would be high conﬁdence, avoid introducing
noisy annotations. Note that in the obtained pseudo labels
Y = {yt}T
t=1, yt = 0 means the t-th frame is not labelled.
Algorithm 2 Loop training
Input: Video X, timestamp annotation Yts, initial spatial fea-
ture extractor f(·), initial spatial classiﬁer h(·), initial temporal
feature extractor g(·), uncertainty threshold τ, forward times
K, times of temporal diffusion n, times of loop training m.
Output: Well optimized spatial feature extractor f(·) and
temporal feature extractor g(·).
1: Y ←Yts
▷Set the initial pseudo labels
2: for i = 1 to n do
3:
▷Optimizing the spatial feature extractor
4:
f(·), h(·) ←OptimS(f(·), h(·), X, Y, Lce)
5:
▷Use UATD to generate the new pseudo labels
6:
Y ←UATD(h(f(·)), X, Yts, τ, K).
7: end for
8: for j = 1 to m do
9:
F ←f(X)
10:
for i = 1 to n do
11:
▷Optimizing the spatial feature extractor
12:
g(·) ←OptimT(g(·), F, Y, Lce, Lsmooth)
13:
▷Use UATD to generate the new pseudo labels
14:
Y ←UATD(g(·), F, Yts, τ, K)
15:
end for
16:
▷Optimizing the spatial feature extractor
17:
f(·), h(·) ←OptimS(f(·), h(·), X, Y, Lce)
18: end for
For clarity, we formulate the overall process of UATD as
Y ←UATD(o(·), Z, ˆY, τ, K), where Z is the input ( e.g., X
or F), o(·) is the network ( e.g., h(f(·)) or g(·)).
C. Loop Training
The duration of the surgical videos generally lasts tens of
minutes or even hours, making it hard to train the model in the
end-to-end manner. In previous full supervised methods [3],
[4], [15], a few consequent frames are sampled from the long
videos for training the spatial and temporal networks in the
end-to-end manner. However, in timestamp annotation, most of
the sampled frames have no labels, resulting in the imbalance
of positive and negative samples. This imbalance training
would degrade the performance; see details in Table III.
To address this problem, we decouple the optimization of
spatial and temporal feature extractors via loop training, as
shown in Fig. 2 (c). In the loop training, we only sample
labelled frames (annotated timestamp or generated pseudo
labels) to optimize the spatial feature extractor or temporal
feature extractor, which can not be achieved in previous jointly
training. Formally, there four main steps in our loop training:
(a) Optimizing the spatial feature extractor: f(·), h(·) ←
OptimS(f(·), h(·), X, Y, Lce). To be speciﬁc, the input video
X is fed into the spatial feature extractor f to obtain the video
feature F = f(X). Then a classiﬁer h(·) is used to obtain the
prediction ˆY = h(F), where ˆY = { ˆyt}T
t=1. Given the target
labels (timestamp annotation or pseudo labels) Y = {yt}T
t1,
the objective for the spatial feature extractor can be formulated


<!-- page 6 -->
6
IEEE TRANSACTIONS ON MEDICAL IMAGING, VOL. XX, NO. XX, XXXX 2020
as:
Lce = −1
T
T
X
t=1,yt̸=0
yt log(ˆyt),
(4)
where yt ̸= 0 indicates the t-th frame is not labelled.
(b) Extracting the spatial features: F = f(X); see details
in step (1).
(c) Optimizing the temporal feature extractor: g(·)
←
OptimT(g(·), F, Y, Lce, Lsmooth). Speciﬁcally, the video fea-
ture F is fed into g(·) to capture the temporal relation of
frames and obtain the corresponding predictions ˆY. We use the
CrossEntropy loss to train the g(·), similar as f(·). Compared
with the spatial feature extractor, to encourage a smooth
transition between frames, we use the truncated mean squared
error as a Smoothing Loss following [14], [55]:
Lsmooth =
1
TC
X
t,c
˜∆2
t,c,
(5)
˜∆2
t,c =
(
∆2
t,c,
∆t,c < γ
γ,
otherwise ,
(6)
∆t,c = |log(ˆyt,c) −log(ˆyt−1,c)|,
(7)
where T is the video length and C is the number of action
classes. This loss function explicitly penalize the difference of
each two adjacent frames and we suggest readers refer to [55]
for more details. The ﬁnal loss function is the weighted sum
of these two losses:
L = Lce + λLsmooth,
(8)
where λ is a hyper-parameter to balance the contribution of
each loss and is set to 0.015 for all of our experiments.
(d) Generating the pseudo labels based on UATD: Y ←
UATD(o(·), Z, Yts, τ, K); see details in Section III-B.
After the deﬁniation of the four steps, we illustrate the loop
training in Algorithm 2.
IV. EXPERIMENTS
A. Datasets and Metrics
M2CAI16. The M2CAI16 dataset [57] consists of 41 laparo-
scopic videos that are acquired at 25fps of cholecystectomy
procedures, and each frame has a resolution of 1920×1080.
Following [21], 27 videos are used for training while the rest
14 are used for testing. These videos are segmented into 8
phases by experienced surgeons.
Cholec80. The cholec80 dataset [5] contains 80 videos of
cholecystectomy surgeries performed by 13 surgeons. All the
videos are recorded at 25 fps, and the frames in them have
the resolution of 1920 × 1080 or 854 × 480. The dataset is
divided into two subsets of equal size, with the ﬁrst 40 videos
as a training set and the other 40 as a testing set.
Evaluation metrics. Following previous works [3]–[5], [20],
we utilize four metrics, i.e., accuracy (AC), precision (PR),
recall (RE), and Jaccard (JA), to evaluate the phase prediction
accuracy. Among them, accuracy and Jaccard index are used
to evaluate the submission of M2CAI Workﬂow Challenge,
while precision and recall are also commonly used metrics
for video-based phase recognition.
(a)
(b)
0
5
10
15
20
25
[0.0, 0.1] (0.1, 0.2] (0.2, 0.3] (0.3, 0.4] (0.4, 0.5] (0.5, 0.6] (0.6, 0.7] (0.7, 0.8] (0.8, 0.9] (0.9, 1.0]
Cholec80
M2CAI16
Annotation position
Percentage (%)
0
100
200
300
400
500
600
700
1
2
3
4
Timestamp
Full
Cholec80
M2CAI16
Surgeon1
Surgeon2
Surgeon1
Surgeon2
Annotation time (s)
73%
69%
75%
78%
0
100
200
300
400
500
600
700
800
900
1
2
3
4
5
6
7
8
9
10
Full-1
Timestamp-1
Full-2
Timestamp-2
0
200
400
600
800
1000
1
2
3
4
5
6
7
8
9
10
11
12
Full-1
Timestamp-1
Full-2
Timestamp-2
Cholec80
MI2CAI16
Annotation time (s)
Annotation time (s)
Video index
Video index
Timestamp
Full
Timestamp
Full
(c)
Fig. 3.
(a) Comparison of the annotation time of timestamp and full
annotations. We show the annotation times of two different surgeons for
Cholec80 and M2CAI16 datasets, which are reported as seconds/video
for each dataset. (b) Statistics of positions of manually annotated
timestamps on two datasets. The horizontal axis indicates the relative
temporal portion of the whole phase. For example, (0.1, 0.2] indicates
the annotated timestamp is inside the temporal period from 0.1 to 0.2
of the phase. The vertical axis represents the percentage of annotated
timestamps. It shows that the timestamps would appear in the arbitrary
position of the phase, and surgeons prefer to label timestamp near the
middle of phases. (c) Statistics of annotation time of manually annotated
timestamps on two datasets. The horizontal axis indicates the video
index, and the vertical axis represents annotation time. It shows that our
timestamp annotation consistently takes less time than full annotation
for labeling each video.
B. Annotation Analysis
To obtain the timestamp annotations, we invite two
surgeons to label a single timestamp for each phase on two
datasets. Speciﬁcally, they are asked to label one timestamp for
each phase while watching the video, as shown in Fig. I (a). To
compare the annotation cost of different types of annotations,
we also ask them to ﬁnd the precise the start and end time for
each phase, i.e., full annotation. In Fig. 3 (a), we report the
average time they spend on each video when using timestamp
and full annotations. “Surgeon1” and “Surgeon2” indicates the
ﬁrst surgeon and the second surgeon respectively. To obtain
annotation times for full or timestamp annotations, we ﬁrst
let the surgeon prepare a timer. When conducting full or
timestamp annotations, the surgeon ﬁrst turned on the timer,
then immediately watched the video and annotated it. After
completing the annotation of a video, the surgeon stopped the
timer immediately, and record the time it takes to annotate
the video. When all videos are annotated and their annotation
time are recorded, we calculate the average annotation time
for all videos. It is clear that our introduced timestamp
annotation can largely reduce the annotation time compared
with the full annotation, e.g.,Surgeon2 can reduce 78% time
in Cholec80 dataset. On average, our proposed timestamp
annotation only requires 26% annotation time compared with
the full annotation.
Furthermore, we also show the distribution of the relative
position of timestamp annotation to the corresponding phase
on two datasets. As shown in Fig. 3 (b), the labeled timestamps
would appear in arbitrary position of the phase. Surgeons


**[Table p6.1]**
|  | 69 | % |
| --- | --- | --- |
| 3% 78 | % | 75 |

[CAPTION] Fig. 3.
(a) Comparison of the annotation time of timestamp and full


<!-- page 7 -->
DING et al.: PREPARATION OF PAPERS FOR IEEE TRANSACTIONS ON MEDICAL IMAGING
7
TABLE I
COMPARISON WITH THE STATE-OF-THE-ART ON CHOLEC80 AND M2CAI16 DATASETS. “∗" INDICATES THE OFFLINE PREDICTION.
Method
Cholec80
M2CAI16
AC (%)
PR (%)
RE (%)
JA (%)
AC (%)
PR (%)
RE (%)
JA (%)
Fully Supervised Methods - 100% annotation time
PhaseNet [56]
78.8 ± 4.7
71.3 ± 15.6
76.6 ± 16.6
-
79.5 ± 12.1
-
-
64.1 ± 10.3
EndoNet [5]
81.7 ± 4.2
-
79.6 ± 7.9
-
-
-
-
-
SV-RCNet [3]
85.3 ± 7.3
80.7 ± 7.0
83.5 ± 7.5
-
81.7 ± 8.1
81.0 ± 8.3
81.6 ± 7.2
65.4 ± 8.9
OHFM [20]
87.3 ± 5.7
-
-
67.0 ± 13.3
85.2 ± 7.5
-
-
68.8 ± 10.5
Casual TCN [14]
87.9 ± 8.2
86.4 ± 7.7
84.8 ± 12.9
72.4 ± 9.4
81.9 ± 11.3
84.8 ± 5.2
82.2 ± 9.0
68.1 ± 8.5
TeCNO [24]
88.6 ± 7.8
86.5 ± 7.0
88.8 ± 17.4
75.1 ± 6.9
-
-
-
-
TMRNet [4]
90.1 ± 7.6
90.3 ± 3.3
89.5 ± 5.0
79.1 ± 5.7
87.0 ± 8.6
87.8 ± 6.9
88.4 ± 5.3
75.1 ± 6.9
Trans-SVNet [13]
90.3 ± 7.1
90.7 ± 5.0
88.8 ± 7.4
79.3 ± 6.6
87.2 ± 9.3
88.0 ± 6.7
87.5 ± 5.5
74.7 ± 7.7
TCN∗[55]
91.1 ± 6.7
90.8 ± 4.5
87.6 ± 11.7
79.1 ± 8.5
82.9 ± 10.8
85.8 ± 5.4
82.7 ± 9.0
69.7 ± 8.7
Not end-to-end [21]
91.5 ± 7.1
-
87.2 ± 8.2
77.2 ± 11.2
88.2 ± 8.5
-
91.4 ± 11.2
75.1 ± 10.6
Semi Supervised Methods - 50% annotation time
LRTD [29]
82.5 ± 8.4
79.7 ± 9.0
80.9 ± 8.1
64.2 ± 10.2
72.1 ± 13.7
74.1 ± 14.9
74.0 ± 10.4
54.4 ± 12.9
SurgSSL [30]
87.0 ± 7.4
84.2 ± 8.9
85.2 ± 11.1
70.5 ± 12.6
79.6 ± 9.4
80.2 ± 11.3
79.6 ± 11.5
62.0 ± 11.1
Timestamp Supervised Methods - 26% annotation time
Casual TCN+Ours
88.6 ± 6.7
86.1 ± 6.7
88.0 ± 10.1
73.7 ± 10.2
86.0 ± 7.8
85.0 ± 6.2
87.1 ± 7.7
71.4 ± 10.4
TCN∗+Ours
91.9 ± 5.6
89.5 ± 4.4
90.5 ± 5.9
79.9 ± 8.5
87.6 ± 8.7
88.2 ± 7.4
87.9 ± 9.6
75.7 ± 9.5
prefer to label timestamp near the middle of phases, which
reveals that the surgeons can identify a phase without watching
the whole phase. That is to say, the surgeons can skip the left
part of the phase after the timestamp annotation. Of course,
there is no need for the surgeons to repeat watching videos
to ﬁnd precise the temporal window for each phase. Hence,
the annotation time can largely be reduced compared with the
full annotation. In the implementation, one second of video is
converted to 25 frames. To save memory and computation cost,
we sample one frame every second. Hence, during annotation,
the surgeon labels the second, and during implementation,
we set the frame belonging to the timestamp second as the
annotation. Finally, we show the statistics of annotation time
of manually annotated timestamps on two datasets in Fig. 3 (c).
The results show that the annotation times of timestamp are
much less than those of full for all videos.
C. Implementation Details
Our code is based on PyTorch using an NVIDIA GeForce
RTX 3090 GPU. We downsample the video to 1fps for training
in all experiments following previous works [3]–[5]. All the
frames are resized to a resolution of 250 × 250, and data aug-
mentations including 224 × 224 cropping, random mirroring,
and color jittering are applied during the training stage. We get
a pre-trained inception-v3 [58] on ImageNet [59]. The batch
size is set to 8, and an Adam optimizer with an initial learning
rate of 1e −4 and weight decay of 1e −5 is used. We further
use a step learning rate scheduler where the step size is two
epochs and decay rate is 0.5 for fune-tuning by 5 epochs. To
train TCN, we use Adam optimizer with an initial learning rate
of 1e−3 and cosine annealing for learning rate decay. For all
experiments, we set a dropout rate of 0.5 and an uncertainty
threshold τ = 0.1; the detailed analysis is shown in Table IV.
The uncertainty is estimated by 5 forward times Monte Carlo
Dropout. [46]. The numbers of rounds of uncertainty-aware
temporal diffusion and loop training are set to m = 4 and
n = 2, respectively. Furthermore, the timestamp annotations
are simulated by randomly selecting one frame from each
action phase in the training videos.
D. Comparison with the State-of-the-Arts
We compare our less is more method with the state-of-
the-arts on the Cholec80 and M2CAI16 datasets, and report
their results in Table I.
The numbers in Table I are the
mean and standard deviation of performance of all phases.
For example, in Cholec80, there are 7 phases. To obtain
the accuracy (AC) for each method, we ﬁrst obtain AC for
each phase, the mean of AC of 7 phases is computed to
obtain the ﬁrst number in Table I. After that, we compute
the standard deviation of AC numbers of 7 phases to obtain
the number after “+/-”. The computation of PR, RE and JA
is like AC. It is clear that our method outperforms previous
data-efﬁcient methods, i.e., semi-supervised ones, on both data
efﬁciency and phase recognition performance. For example,
our timestamp supervision only requires 26% annotation time
of the full supervision [44], while semi-supervision needs 50%
annotation time [29]. Moreover, our method with the casual
TCN [14] achieves 88.6% of accuracy on Cholec80 dataset,
achieving the competitive performance compared to semi-
supervised methods. We can also ﬁnd that our method can even
achieve the competitive performance compared with the fully
supervised methods, with only 26% annotation time of them.
Notably, the improvements of our method are more signiﬁcant
in M2CAI16 than in Cholec80. This is because M2CAI16
contains more ambiguous frames [12], which degrades the
performance. The details why our methods can outperform
corresponding backbones in fully supervised setup will be
discussed in Sec. IV-F.
E. Comparison with Different Timestamp Supervision
Methods
To evaluate the efﬁciency of our proposed uncertainty-aware
temporal diffusion (UATD) for surgical video timestamp su-


**[Table p7.1]**
| Method | Cholec80 M2CAI16 AC (%) PR (%) RE (%) JA (%) AC (%) PR (%) RE (%) JA (%) |
| --- | --- |


**[Table p7.2]**
| PhaseNet [56] EndoNet [5] SV-RCNet [3] OHFM [20] Casual TCN [14] TeCNO [24] TMRNet [4] Trans-SVNet [13] TCN∗ [55] Not end-to-end [21] | 78.8 ± 4.7 71.3 ± 15.6 76.6 ± 16.6 - 81.7 ± 4.2 - 79.6 ± 7.9 - 85.3 ± 7.3 80.7 ± 7.0 83.5 ± 7.5 - 87.3 ± 5.7 - - 67.0 ± 13.3 87.9 ± 8.2 86.4 ± 7.7 84.8 ± 12.9 72.4 ± 9.4 88.6 ± 7.8 86.5 ± 7.0 88.8 ± 17.4 75.1 ± 6.9 90.1 ± 7.6 90.3 ± 3.3 89.5 ± 5.0 79.1 ± 5.7 90.3 ± 7.1 90.7 ± 5.0 88.8 ± 7.4 79.3 ± 6.6 91.1 ± 6.7 90.8 ± 4.5 87.6 ± 11.7 79.1 ± 8.5 91.5 ± 7.1 - 87.2 ± 8.2 77.2 ± 11.2 | 79.5 ± 12.1 - - 64.1 ± 10.3 - - - - 81.7 ± 8.1 81.0 ± 8.3 81.6 ± 7.2 65.4 ± 8.9 85.2 ± 7.5 - - 68.8 ± 10.5 81.9 ± 11.3 84.8 ± 5.2 82.2 ± 9.0 68.1 ± 8.5 - - - - 87.0 ± 8.6 87.8 ± 6.9 88.4 ± 5.3 75.1 ± 6.9 87.2 ± 9.3 88.0 ± 6.7 87.5 ± 5.5 74.7 ± 7.7 82.9 ± 10.8 85.8 ± 5.4 82.7 ± 9.0 69.7 ± 8.7 88.2 ± 8.5 - 91.4 ± 11.2 75.1 ± 10.6 |
| --- | --- | --- |


**[Table p7.3]**
| LRTD [29] SurgSSL [30] | 82.5 ± 8.4 79.7 ± 9.0 80.9 ± 8.1 64.2 ± 10.2 87.0 ± 7.4 84.2 ± 8.9 85.2 ± 11.1 70.5 ± 12.6 | 72.1 ± 13.7 74.1 ± 14.9 74.0 ± 10.4 54.4 ± 12.9 79.6 ± 9.4 80.2 ± 11.3 79.6 ± 11.5 62.0 ± 11.1 |
| --- | --- | --- |


**[Table p7.4]**
| Casual TCN+Ours TCN∗+Ours | 88.6 ± 6.7 86.1 ± 6.7 88.0 ± 10.1 73.7 ± 10.2 91.9 ± 5.6 89.5 ± 4.4 90.5 ± 5.9 79.9 ± 8.5 | 86.0 ± 7.8 85.0 ± 6.2 87.1 ± 7.7 71.4 ± 10.4 87.6 ± 8.7 88.2 ± 7.4 87.9 ± 9.6 75.7 ± 9.5 |
| --- | --- | --- |


<!-- page 8 -->
8
IEEE TRANSACTIONS ON MEDICAL IMAGING, VOL. XX, NO. XX, XXXX 2020
(d) Video 13
GT
ours
Li et al.
(a) Video 03
(b) Video 12
GT
ours
Li et al.
GT
ours
Li et al.
(c) Video 04
GT
ours
Li et al.
GT
ours
(f) Video 25
GT
ours
(e) Video 14
Fig. 4.
Comparison of the visualization of pseudo labels generated by ours and Li et al. [14]. “GT” indicates the ground-truth. We sample four
videos, i.e., from Cholec80 ((a)-(b)) and M2CAI16 ((c)-(d)). It is clear that our method can generate more accurate pseudo labels compared with
Li et al. [14]; see red boxes. We also illustrate the worst pseudo labels generated by our method shown in (e)-(f). The uncertainty frames inside the
phases would make the model to stop temporal diffusion (see black boxes).
TABLE II
COMPARISON WITH DIFFERENT TIMESTAMP SUPERVISION METHODS.
Method
AC (%)
PR (%)
RE (%)
JA (%)
Cholec80
Naive
66.9 ± 5.6
62.3 ± 6.5
74.8 ± 6.5
48.2 ± 4.4
Uniform
58.7 ± 7.8
55.5 ± 6.1
65.9 ± 5.4
39.0 ± 6.5
Li et al. [14]
79.4 ± 5.5
78.7 ± 6.5
85.4 ± 5.5
64.0 ± 5.6
Ours
88.6 ± 6.7
86.1 ± 6.7
88.0 ± 10.1
73.7 ± 10.2
M2CAI16
Naive
67.5 ± 7.2
58.7 ± 6.5
61.7 ± 6.5
44.8 ± 6.7
Uniform
56.5 ± 8.7
56.7 ± 7.7
57.0 ± 7.9
38.2 ± 5.6
Li et al. [14]
72.7 ± 8.8
76.5 ± 7.1
80.5 ± 6.9
59.9 ± 10.1
Ours
86.0 ± 7.8
85.0 ± 6.2
87.1 ± 7.7
71.4 ± 10.4
pervision, we compare our methods with two baseline models
following [14], i.e., Naive and Uniform, and report the results
in Table II. Speciﬁcally, in Naive, we only use the annotated
timestamp labels to supervise the model training, without
generating any pseudo labels. In Uniform, the pseudo labels
are generated by a uniform way, i.e., the action labels change
at the center frame between two timestamp annotations. For
example, assuming two timestamps yt1 and yt2 with t1 < t2,
then the pseudo labels can be generated as:
ˆyt =
(
yt1,
t ∈(t1, t1 + (t2 −t1)/2]
yt2,
t ∈(t1 + (t2 −t1)/2, t2) .
(9)
It is clear that our method outperforms other two methods with
a clear margin. Furthermore, we also compare Li et al. [14],
which is the SOTA in action segmentation under this setting.
It uses the middle output of model i.e., features of frames to
detect action change and generate frame-wise pseudo labels
However, using feature similarity to detect action change could
be confused when the boundaries are generally ambiguous. As
shown in Fig. 4, our methods give accurate pseudo labels by
stopping diffusion near boundaries while [14] attempts to give
TABLE III
ABLATIVE STUDY OF KEY COMPONENTS ON CHOLEC80 DATASET.
‘UATD (S)’ AND ‘UATD (T)’ INDICATE USING UATD IN THE SPATIAL
FEATURE AND TEMPORAL FEATURE EXTRACTORS. ‘LP’ INDICATES THE
LOOP TRAINING WHICH IS DEFINED IN SEC. III-C.
UATD (S)
UATD (T)
LP
AC (%)
PR (%)
RE (%)
JA (%)
%
%
%
66.9 ± 9.7
62.3 ± 6.7
74.8 ± 7.8
48.2 ± 7.6
"
%
%
82.3 ± 7.6
78.1 ± 8.8
86.9 ± 6.5
66.0 ± 7.4
%
"
%
77.6 ± 5.3
77.3 ± 6.7
81.0 ± 7.3
61.3 ± 5.2
%
%
"
68.5 ± 4.8
63.7 ± 6.2
75.2 ± 3.7
50.2 ± 6.1
"
"
%
85.6 ± 7.4
83.5 ± 6.5
86.6 ± 6.1
70.9 ± 8.2
"
"
"
88.6 ± 6.7
86.1 ± 6.7
88.0 ± 10.1
73.7 ± 10.2
TABLE IV
QUANTITATIVE RESULTS OF DIFFERENT UNCERTAINTY THRESHOLDS.
AC (%)
PR (%)
RE (%)
JA (%)
Labelling
Labelling
Rate (%)
Accuracy (%)
τ = ∞
84.65
86.34
84.65
70.43
93.49
92.04
τ = 0.2
85.32
86.71
85.06
71.14
92.72
94.13
τ = 0.1
85.95
84.96
87.05
71.43
87.51
96.99
τ = 0.05
85.49
86.19
86.42
71.20
60.76
99.04
unappealing labels. From Table. II, we can also see that our
method obtains 7% −13% improvements over all metrics.
F. Ablation Study
Effect of UATD and LP. There are two key components,
i.e., uncertainty-aware temporal diffusion (UATD) and loop
training (LP), in our method. We ablate the effect of them in
Table. III. It is clear that the proposed UATD can improve
the timestamp supervision with a clear margin, e.g., combined
with UATD, the model achieves 85.59% accuracy, outperform-
ing 18.67% over the baseline model. Also, we could ﬁnd that
loop training contributes to around 3% improvements.
Impact of the uncertainty threshold τ. The quality of pseudo
labels is depended on pseudo labeling rate and pseudo labels


**[Table p8.1]**
| Method | AC (%) PR (%) RE (%) JA (%) |
| --- | --- |


**[Table p8.2]**
| Naive Uniform Li et al. [14] Ours | 66.9 ± 5.6 62.3 ± 6.5 74.8 ± 6.5 48.2 ± 4.4 58.7 ± 7.8 55.5 ± 6.1 65.9 ± 5.4 39.0 ± 6.5 79.4 ± 5.5 78.7 ± 6.5 85.4 ± 5.5 64.0 ± 5.6 88.6 ± 6.7 86.1 ± 6.7 88.0 ± 10.1 73.7 ± 10.2 |
| --- | --- |


**[Table p8.3]**
| UATD (S) UATD (T) LP | AC (%) PR (%) RE (%) JA (%) |
| --- | --- |
| % % % " % % % " % % % " " " % " " " | 66.9 ± 9.7 62.3 ± 6.7 74.8 ± 7.8 48.2 ± 7.6 82.3 ± 7.6 78.1 ± 8.8 86.9 ± 6.5 66.0 ± 7.4 77.6 ± 5.3 77.3 ± 6.7 81.0 ± 7.3 61.3 ± 5.2 68.5 ± 4.8 63.7 ± 6.2 75.2 ± 3.7 50.2 ± 6.1 85.6 ± 7.4 83.5 ± 6.5 86.6 ± 6.1 70.9 ± 8.2 88.6 ± 6.7 86.1 ± 6.7 88.0 ± 10.1 73.7 ± 10.2 |


**[Table p8.4]**
| Naive Uniform Li et al. [14] Ours | 67.5 ± 7.2 58.7 ± 6.5 61.7 ± 6.5 44.8 ± 6.7 56.5 ± 8.7 56.7 ± 7.7 57.0 ± 7.9 38.2 ± 5.6 72.7 ± 8.8 76.5 ± 7.1 80.5 ± 6.9 59.9 ± 10.1 86.0 ± 7.8 85.0 ± 6.2 87.1 ± 7.7 71.4 ± 10.4 |
| --- | --- |


**[Table p8.5]**
|  | AC (%) PR (%) RE (%) JA (%) | Labelling Labelling Rate (%) Accuracy (%) |
| --- | --- | --- |
| τ = ∞ τ = 0.2 τ = 0.1 τ = 0.05 | 84.65 86.34 84.65 70.43 85.32 86.71 85.06 71.14 85.95 84.96 87.05 71.43 85.49 86.19 86.42 71.20 | 93.49 92.04 92.72 94.13 87.51 96.99 60.76 99.04 |

[CAPTION] Fig. 4.
Comparison of the visualization of pseudo labels generated by ours and Li et al. [14]. “GT” indicates the ground-truth. We sample four


<!-- page 9 -->
DING et al.: PREPARATION OF PAPERS FOR IEEE TRANSACTIONS ON MEDICAL IMAGING
9
TABLE V
COMPARISON OF LABELLING RATE AND LABELLING ACCURACY OF
PSEUDO LABELS GENERATED BY UATD IN DIFFERENT ITERATIONS.
“TS" INDICATES THE INITIAL TIMESTAMP ANNOTATIONS.
Iteration
TS
1-st
2-nd
3-rd
Labelling Rate (%)
0.33
67.70
76.82
84.45
Labelling Accuracy (%)
100.00
98.69
97.95
97.42
Training step
GT
1-𝑠𝑡
2-𝑛𝑑
3-𝑟𝑑
Time
Fig. 5.
Visualization of the different iterations of the pseudo labels
generated by our method. “GT” indicates the ground truth. “1-st”, “2-nd”
and “3-rd” indicate generated pseudo labels in the ﬁrst, second and third
iterations respectively.
accuracy, which is controlled by the uncertainty threshold τ in
Algorithm 1. In order to evaluate the effect of τ, we compare
the performance of the models with different τ and report the
results in Table IV. “Labelling Rate” indicates the ratio of the
frames annotated by our method to all frames. To evaluate the
accuracy of our generated annotations, i.e., pseudo labels, we
compare the generated pseudo labels with the ground-truth.
Speciﬁcally, for a frame, if the annotated label generated by
our method is equal to the ground-truth, the frame is regarded
as the correct annotated frame, and vice versa. We can ﬁnd
that the higher uncertainty threshold would lead to the higher
pseudo labeling rate and the lower accuracy of pseudo labels,
and vice versa. For example, with inﬁnity threshold, i.e., ﬁrst
row in Table IV, pseudo labeling rate can reach 93.49% while
accuracy of pseudo labels is only 92.04%. Such higher labeling
rate would introduce more noisy labels, which degrades the
labeling accuracy. Furthermore, with different τ, i.e., 0.2,
0.1 and 0.05, the performance of our method is very stable.
For example, the variance for accuracy values with different
thresholds is only 0.11% In our paper, we set τ to 0.1 for the
best trade-off.
Analysis of pseudo labels in different iterations.
Given
only a single manual labeled annotations, we show that our
model can generate more and more reliable pseudo-labels
step by step in Table V. “Labelling Rate” and “Labelling
Accuracy” are the same meaning as Table IV. It shows that
our method can generate more and more pseudo labels the
number of iterations increases. This is because each iteration
of temporal diffusion gives temporal model extra information,
the model can generate more pseudo labels next time. Also, the
accuracy of generated pseudo labels is very trustworthy. Since
the frames show very similar appearances to their adjacent
frames, the network can easily generate correct predictions
for the neighbor frames of the annotated frame. We also show
45
55
65
75
85
0
1
2
3
4
ACC
PR
RE
JA
45
55
65
75
85
0
1
2
3
4
ACC
PR
RE
JA
(a)
(b)
Accuracy
Number of iterations
Accuracy
Number of iterations
Fig. 6.
Analysis of the number of iterations for loop training on (a)
Cholec80 and (b) M2CAI16. We show the results of ACC, PR, RE and
JA of models with different numbers of iterations.
(a) Cholec80
(b) M2CAI16
Fig. 7.
Box plots of performance of random timestamp annotations on
Cholec80 and M2CAI16 datasets.
the visualization of the different iterations of the pseudo labels
generated by our method in Fig. 5.
Effect the number of iterations for loop training.
We
conduct the analysis of the number of iterations for loop
training in Fig. 6. The results show that more iterations
for loop training can improve the performance, since more
trustworthy labels are introduced to training. We also ﬁnd that
there is no signiﬁcant performance improvement after more
than two iterations. Hence, in this paper, we set the number
of iterations for loop training as two.
Robust to different timestamp annotations. In our experi-
ments, the timestamp annotations are generated by randomly
selecting one frames to be annotated of each phase. In order
to evaluate our method is robust to the different timestamps,
we random 10 different timestamps by different random seeds
and analyse their impacts to the performance, which is shown
in Fig. 7. Speciﬁcally, we report the box plots of 10 random
timestamp annotations on Cholec80 and M2CAI16 datasets.
The short and ﬂat boxes indicate that our proposed method is
robust to different timestamp annotations, e.g., the difference
between the maximum and minimum is 2.3%. What’s more,
our method can outperform most of methods in Table. I with
even the worst timestamp annotations.
Effect of timestamps in different phase positions.
In order
to explore the effect of timestamps in different phase positions,
we enforce the random timestamp annotations inside start,
end or middle region of each phase. More speciﬁcally, we
regard the ﬁrst 10% frames, the middle 10% frames and the
last 10% frames of each phase as the start, middle and end
regions. As shown in Table. VI, annotating at the start and end
frames of each phase would degrade the performance. This is
because that frames near boundaries are generally ambiguous,
which can be hard to act as an anchor of temporal diffusion.
In the contrast, the middle frames are more discriminative to


**[Table p9.1]**
| Iteration | TS 1-st 2-nd 3-rd |
| --- | --- |
| Labelling Rate (%) Labelling Accuracy (%) | 0.33 67.70 76.82 84.45 100.00 98.69 97.95 97.42 |

[CAPTION] Fig. 5.
Visualization of the different iterations of the pseudo labels

[CAPTION] Fig. 6.
Analysis of the number of iterations for loop training on (a)

[CAPTION] Fig. 7.
Box plots of performance of random timestamp annotations on


<!-- page 10 -->
10
IEEE TRANSACTIONS ON MEDICAL IMAGING, VOL. XX, NO. XX, XXXX 2020
TABLE VI
QUANTITATIVE RESULTS OF START, END, MIDDLE AND RANDOM
TIMESTAMP POSITIONS ON CHOLEC80 DATASET.
Timestamp
AC (%)
PR (%)
RE (%)
JA (%)
Position
Start
90.64
87.92
88.37
76.75
End
90.17
88.35
82.24
70.75
Middle
92.59
90.13
89.60
80.04
Random
91.86
89.51
90.52
79.90
TABLE VII
COMPARISON OF ANNOTATION TIME BETWEEN A SINGLE TIMESTAMP
AND TWO TIMESTAMPS.
Video Index
01
05
Single Timestamp
222s
155s
Two Timestamps
331s
279s
represent current phases and thus can generate more correct
pseudo labels. Actually, the surgeons ,i.e., the annotators, tend
to label the discriminative frames because they can easily
recognize them when seeing through the whole video [44],
which ensure timestamp annotations efﬁcient and effective.
Comparison between a single and two timestamps. During
the timestamp annotation, once a phase is identiﬁed and
current timestamp is recorded, the surgeon could choose to
record another timestamp for the phase. Here, we compare
the annotation cost between a single and two timestamps in
Table VII. Two videos, i.e.,“01” and “05”, are sampled from
Cholec80 and M2CAI16 respectively. The result shows that
two timestamp annotation would cost more time than a single
timestamp annotation, e.g., the surgeon would spend 331s for
“01” while annotating a single timestamp only requires 222s.
We also conduct experiments to compare the performance
of the models training with a single timestamp and two
timestamps, as shown in Table VIII. The results show that
two timestamp annotation cannot achieve clear improvement
but bring additional annotation cost. Hence, annotating a single
timestamp is much efﬁcient than two timestamps, and we use
the best efﬁcient way to solve surgical phase recognition in
this paper.
Comparison of generated pseudo label and ground-truth.
In our experiments, we ﬁnd that our method only generates
pseudo labels for discriminative frames while ignores the
ambiguous ones near boundaries. As shown in Fig. 8, our
generated pseudo labels discard ambiguous or less informative
frames compared to the ground-truth. More importantly, the
model trained with our generated pseudo labels outperforms
the model trained with the ground-truth; see details in Table.
I. This indicates that ambiguous boundary of two adjacent
actions would degree the performance.
G. Incorporate UATD into Current Methods
As analyzed in Fig. 8, we ﬁnd that our method can
only generate labels for discriminative frames, instead of
ambiguous frames. It comes up that if masking ambiguous
frames from the ground-truth by our UATD can improve
the performance. To this end, as shown in Fig. 9 (a), we
TABLE VIII
COMPARISON OF PERFORMANCE OF MODELS TRAINING WITH A SINGLE
TIMESTAMP AND TWO TIMESTAMPS.
Method
Annotation
AC (%)
PR (%)
RE (%)
JA (%)
Cholec80
Casual TCN
Single
88.56
86.05
88.00
73.72
Two
88.79
89.61
88.12
73.80
TCN
Single
91.86
89.51
90.52
79.90
Two
91.91
89.66
90.81
79.93
M2CAI16
Casual TCN
Single
86.03
85.02
87.08
71.43
Two
86.07
85.11
87.14
71.50
TCN
Single
87.62
88.25
87.91
75.72
Two
87.70
88.30
87.98
75.81
Ours
GT
Ours
GT
Ambiguous
Less informative
(a)
(b)
Fig. 8.
Comparison of pseudo labels generated by ours and ground-
truth. It is clear that our method avoids annotating the frames near
boundaries, where frames are generally (a) ambiguous or (b) less
informative. In our paper, we regard ambiguous frames as the frames
that shows similar appearance in different phases following [12]. Less
informative frames indicate the frames that provide little information
to identify different phases, such as phases containing no actions or
instruments.
mask some ground-truth labels near boundaries, based on the
pseudo labels generated by our methods. To be speciﬁc, we
use UATD to generate pseudo labels, and record the indexes
of unlabelled frames that are with high uncertainty. Then,
we remove those frames with the recorded indexes from the
ground-truth, and obtained a clean ground-truth to supervise
the model. We compare the performance of the models training
with (a) ground-truth (GT), (b) pseudo labels generated by
UATD and (c) GT masked by UATD, and report the results
in Table IX. To show the generalization of our proposed
method, we also conduct experiments on JIGSAW [60], which
a simulated dataset with a clear domain gap. From Table IX, it
is clear that the model training with GT masked by UATD can
achieve the best results, even outperforms the current SOTA;
see Table I for comparison. We further conduct experiments
on the models training with GT masked by the ﬁxed width,
as shown in Fig. 9 (b). As illustrated in Table. X, masking
some frames near boundary during training outperform the
model without masking over around 1% −3% in all metrics.
However, it will introduce a new hyper-parameter, i.e., the
width of mask, which is critical to the performance. Hence,
in order to achieve the good performance, we need to conduct
many experiments to ﬁnd the best choice, which is very time-
consuming. On the contrary, our method can be used as
an approach to clean the noisy labels in the ground-truth
automatically, without the need of hand-designed width. To
further explain this phenomenon, we visualize the feature


**[Table p10.1]**
| Timestamp Position | AC (%) PR (%) RE (%) JA (%) |
| --- | --- |
| Start End Middle Random | 90.64 87.92 88.37 76.75 90.17 88.35 82.24 70.75 92.59 90.13 89.60 80.04 91.86 89.51 90.52 79.90 |


**[Table p10.2]**
| S Casual TCN S TCN | ingle Two ingle Two | 88.56 86.05 88.00 73.72 88.79 89.61 88.12 73.80 91.86 89.51 90.52 79.90 91.91 89.66 90.81 79.93 |
| --- | --- | --- |


**[Table p10.3]**
| S Casual TCN S TCN | ingle Two ingle Two | 86.03 85.02 87.08 71.43 86.07 85.11 87.14 71.50 87.62 88.25 87.91 75.72 87.70 88.30 87.98 75.81 |
| --- | --- | --- |


**[Table p10.4]**
| Video Index | 01 05 |
| --- | --- |
| Single Timestamp Two Timestamps | 222s 155s 331s 279s |

[CAPTION] Fig. 8.
Comparison of pseudo labels generated by ours and ground-


<!-- page 11 -->
DING et al.: PREPARATION OF PAPERS FOR IEEE TRANSACTIONS ON MEDICAL IMAGING
11
(a) Ours
(b) Fixed width
Fig. 9.
(a) Masking boundaries by using UATD to detect ambiguous
frames. (b) Masking boundaries by ﬁxed width.
TABLE IX
EFFECTIVENESS OF INCORPORATING UATD INTO CURRENT METHODS.
‘TIMESTAMP’ IS USING OUR GENERATED PSEUDO LABELS BY UATD
FROM TIMESTAMP ANNOTATIONS AND ‘GT W/ UATD’ INDICATES THE
GROUND-TRUTH LABELS MASKED BY UATD; SEE SEC. IV-G FOR
DETAILS.
Method
Annotation
AC (%)
PR (%)
RE (%)
JA (%)
Cholec80
Casual TCN
GT
87.94
86.40
84.81
72.40
Timestamp
88.56
86.05
88.00
73.72
GT w/ UATD
91.18
89.88
90.93
79.76
TCN
GT
91.14
90.84
87.64
79.14
Timestamp
91.86
89.51
90.52
79.90
GT w/ UATD
92.75
91.23
93.10
83.89
M2CAI16
Casual TCN
GT
81.91
84.82
82.24
68.06
Timestamp
86.03
85.02
87.08
71.43
GT w/ UATD
87.01
88.23
88.81
76.26
TCN
GT
82.94
85.82
82.69
69.71
Timestamp
87.62
88.25
87.91
75.72
GT w/ UATD
88.32
89.03
89.23
78.81
JIGSAW
Casual TCN
GT
80.12
82.02
81.16
69.11
Timestamp
81.73
84.91
84.82
71.55
GT w/ UATD
83.27
85.51
85.27
72.81
TCN
GT
81.43
84.29
83.71
70.18
Timestamp
83.18
85.19
85.72
72.12
GT w/ UATD
84.28
86.13
86.16
73.15
similarity matrix in Fig. 10. Each red box in each similarity
matrix indicates each phase in a video. It is clear that training
with our generate pseudo labels i.e., removing ambiguous
labels near boundaries between two phases, would help to
decrease intra-class distance and increase inter-class distance
simultaneously.
V. DISCUSSION
Surgical phase recognition is one key component of
computer-assisted surgery systems, which advances context-
awareness in modern operating rooms. However, most existing
works require full annotations which are expensive, expertise-
required and error-prone [28]. In contrast, we introduce times-
tamp supervision which only requires one timestamp anno-
tated by human for each phase in a video. We invite two
surgeons to conduct both full and timestamp annotations and
record the time cost for these two annotations. To leverage
this supervision, we propose Uncertainty-Aware Temporal
Diffusion (UATD) to generate trustworthy pseudo labels for
those unlabeled frames, which is based on the property of
surgical phases. Furthermore, loop training is also introduce to
address the imbalance training and memory cost in timestamp
surgical phase recognition. The in-depth empirical studies of
the proposed UATD and LP based on timestamp supervision
discovers four deep insights: 1) Timestamp annotation can
TABLE X
EFFECTIVENESS OF BOUNDARY MASK ON CHOLEC80 DATASET.
Mask width
AC (%)
PR (%)
RE (%)
JA (%)
0
91.14
90.84
87.64
79.14
3
92.04
91.87
89.07
81.44
5
92.31
92.26
89.52
82.12
10
92.75
92.86
90.57
83.32
20
92.68
93.20
90.40
82.66
Training with Ours
Training with GT
time
time
time
time
time
time
time
time
Fig. 10.
Feature similarity matrix visualization. The horizontal and verti-
cal axes represent the time indexes. We use cosine similarity to measure
the degree of similarity between two arbitrary frame-level feature vectors
within the same video. Each red box indicates each phase in a video.
Note that, frame-level features of the same phase should be as similar
as possible while separating one from others. Compared to the model
trained with the ground-truth (GT), better representations of the features
can be learned by our generated pseudo labels (right).
reduce 74% annotation time compared with the full annotation,
and surgeons tend to annotate those timestamps that are near
the middle of phases; 2) Extensive experiments demonstrate
that our method can achieve competitive results compared with
full supervision methods, while reducing manual annotation
cost; 3) Less is more in surgical phase recognition, i.e., less
but discriminative pseudo labels outperform full but containing
ambiguous frames; 4) The proposed UATD can be used as
a plug and play method to clean ambiguous labels near
boundaries between phases, and improve the performance of
the current surgical phase recognition methods; see details in
Table IX.
Although our method achieves promising results, there are
some limitations. First, the temporal property we consider is
not overall yet. The diffusion in our method assumes that the
workﬂow is smooth without dramatic change and hardly any
ambiguous frame occurs in the internal of phase ; see Fig. 4
(e)-(f). But such assumption may be false for other datasets
and in the future we will study more comprehensive temporal
relationship to handle the intra-phase discontinuity. Moreover,
the training process we propose is time-consuming containing
several iterations of training model from scratch. And we will
design more elegant training process to link up the optimal
learning from different annotations, i.e., different rounds of


**[Table p11.1]**
| Casual TCN TCN | GT Timestamp GT w/ UATD GT Timestamp GT w/ UATD | 87.94 86.40 84.81 72.40 88.56 86.05 88.00 73.72 91.18 89.88 90.93 79.76 91.14 90.84 87.64 79.14 91.86 89.51 90.52 79.90 92.75 91.23 93.10 83.89 |
| --- | --- | --- |


**[Table p11.2]**
| Casual TCN TCN | GT Timestamp GT w/ UATD GT Timestamp GT w/ UATD | 81.91 84.82 82.24 68.06 86.03 85.02 87.08 71.43 87.01 88.23 88.81 76.26 82.94 85.82 82.69 69.71 87.62 88.25 87.91 75.72 88.32 89.03 89.23 78.81 |
| --- | --- | --- |


**[Table p11.3]**
| Casual TCN TCN | GT Timestamp GT w/ UATD GT Timestamp GT w/ UATD | 80.12 82.02 81.16 69.11 81.73 84.91 84.82 71.55 83.27 85.51 85.27 72.81 81.43 84.29 83.71 70.18 83.18 85.19 85.72 72.12 84.28 86.13 86.16 73.15 |
| --- | --- | --- |

[CAPTION] Fig. 9.
(a) Masking boundaries by using UATD to detect ambiguous

[CAPTION] Fig. 10.
Feature similarity matrix visualization. The horizontal and verti-


<!-- page 12 -->
12
IEEE TRANSACTIONS ON MEDICAL IMAGING, VOL. XX, NO. XX, XXXX 2020
temporal diffusion in our methods.
Finally, we expect the community to focus more on label-
efﬁcient surgical video analysis. The weakly setting of videos,
such as transcripts [39] and timestamp supervision, deserve
further attention and exploiting. And the related ideas can be
further investigated in other medical image analysis problems
in CT [61]–[64], MRI [65]–[67].
VI. CONCLUSION
In this paper, we introduce the most annotation-saving
setting, namely timestamp supervision, for surgical phase
recognition. With timestamp supervision, we propose a novel
uncertainty-aware temporal diffusion (UATD) method to gen-
erate trustworthy pseudo labels according to the labeled
frames. Our main idea is to generate pseudo labels by con-
sidering the relationship among video frames. Results on two
datasets show that our method can achieve the competitive
performance compared with the fully supervised setup. More-
over, we also ﬁnd that our method can be used as a labeling
clean approach to remove the noisy labels near boundaries
to improve the generalization of the current surgical phase
recognition, which reveals an interesting phenomenon less
is more in this task. This paper provides some insights for
label-efﬁcient surgical phase recognition and hopefully inspire
researchers to design label-efﬁcient surgical video analysis
algorithms.
REFERENCES
[1] L. Maier-Hein, S. S. Vedula, S. Speidel, N. Navab, R. Kikinis, A. Park,
M. Eisenmann, H. Feussner, G. Forestier, S. Giannarou, et al., “Surgi-
cal data science for next-generation interventions,” Nature Biomedical
Engineering, vol. 1, no. 9, pp. 691–696, 2017.
[2] A. Moglia, V. Ferrari, L. Morelli, M. Ferrari, F. Mosca, and A. Cuschieri,
“A systematic review of virtual reality simulators for robot-assisted
surgery,” European urology, vol. 69, no. 6, pp. 1065–1080, 2016.
[3] Y. Jin, Q. Dou, H. Chen, L. Yu, J. Qin, C.-W. Fu, and P.-A. Heng,
“Sv-rcnet: workﬂow recognition from surgical videos using recurrent
convolutional network,” IEEE transactions on medical imaging, vol. 37,
no. 5, pp. 1114–1126, 2017.
[4] Y. Jin, Y. Long, C. Chen, Z. Zhao, Q. Dou, and P.-A. Heng, “Temporal
memory relation network for workﬂow recognition from surgical video,”
IEEE Transactions on Medical Imaging, 2021.
[5] A. P. Twinanda, S. Shehata, D. Mutter, J. Marescaux, M. De Mathelin,
and N. Padoy, “Endonet: a deep architecture for recognition tasks on
laparoscopic videos,” IEEE transactions on medical imaging, vol. 36,
no. 1, pp. 86–97, 2016.
[6] N. Bricon-Souf and C. R. Newman, “Context awareness in health care:
A review,” international journal of medical informatics, vol. 76, no. 1,
pp. 2–12, 2007.
[7] B. Bhatia, T. Oates, Y. Xiao, and P. Hu, “Real-time identiﬁcation of
operating room state from video,” in AAAI, vol. 2, pp. 1761–1766, 2007.
[8] D. Liu, Q. Li, T. Jiang, Y. Wang, R. Miao, F. Shan, and Z. Li, “Towards
uniﬁed surgical skill assessment,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pp. 9522–
9531, 2021.
[9] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image
recognition,” in Proceedings of the IEEE conference on computer vision
and pattern recognition, pp. 770–778, 2016.
[10] K. Simonyan and A. Zisserman, “Very deep convolutional networks for
large-scale image recognition,” arXiv preprint arXiv:1409.1556, 2014.
[11] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and
B. Guo, “Swin transformer: Hierarchical vision transformer using shifted
windows,” arXiv preprint arXiv:2103.14030, 2021.
[12] X. Ding and X. Li, “Exploring segment-level semantics for online
phase recognition from surgical videos,” IEEE Transactions on Medical
Imaging, 2022.
[13] X. Gao, Y. Jin, Y. Long, Q. Dou, and P.-A. Heng, “Trans-svnet:
Accurate phase recognition from surgical videos via hybrid embedding
aggregation transformer,” arXiv preprint arXiv:2103.09712, 2021.
[14] Z. Li, Y. Abu Farha, and J. Gall, “Temporal action segmentation from
timestamp supervision,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, pp. 8365–8374, 2021.
[15] Y. Jin, H. Li, Q. Dou, H. Chen, J. Qin, C.-W. Fu, and P.-A. Heng, “Multi-
task recurrent convolutional network with correlation loss for surgical
video analysis,” Medical image analysis, vol. 59, p. 101572, 2020.
[16] T. Blum, H. Feußner, and N. Navab, “Modeling and segmentation of
surgical workﬂow from laparoscopic video,” in International Conference
on Medical Image Computing and Computer-Assisted Intervention,
pp. 400–407, Springer, 2010.
[17] F. Lalys, L. Riffaud, D. Bouget, and P. Jannin, “A framework for the
recognition of high-level surgical tasks from video images for cataract
surgeries,” IEEE Transactions on Biomedical Engineering, vol. 59, no. 4,
pp. 966–976, 2011.
[18] A. Graves, “Practical variational inference for neural networks,” Ad-
vances in neural information processing systems, vol. 24, 2011.
[19] I. Funke, S. Bodenstedt, F. Oehme, F. von Bechtolsheim, J. Weitz,
and S. Speidel, “Using 3d convolutional neural networks to learn
spatiotemporal features for automatic surgical gesture recognition in
video,” in International Conference on Medical Image Computing and
Computer-Assisted Intervention, pp. 467–475, Springer, 2019.
[20] F. Yi and T. Jiang, “Hard frame detection and online mapping for
surgical phase recognition,” in International Conference on Medical
Image Computing and Computer-Assisted Intervention, pp. 449–457,
Springer, 2019.
[21] F. Yi and T. Jiang, “Not end-to-end: Explore multi-stage architecture
for online surgical phase recognition,” arXiv preprint arXiv:2107.04810,
2021.
[22] B. Zhang, A. Ghanem, A. Simes, H. Choi, A. Yoo, and A. Min, “Swnet:
Surgical workﬂow recognition with deep convolutional network,” in
Medical Imaging with Deep Learning, 2021.
[23] M. Sahu, A. Szengel, A. Mukhopadhyay, and S. Zachow, “Surgical
phase recognition by learning phase transitions,” Current Directions in
Biomedical Engineering, vol. 6, no. 1, 2020.
[24] T. Czempiel, M. Paschali, M. Keicher, W. Simson, H. Feussner, S. T.
Kim, and N. Navab, “Tecno: Surgical phase recognition with multi-
stage temporal convolutional networks,” in International Conference
on Medical Image Computing and Computer-Assisted Intervention,
pp. 343–352, Springer, 2020.
[25] T. Czempiel, M. Paschali, D. Ostler, S. T. Kim, B. Busam, and
N. Navab, “Opera: Attention-regularized transformers for surgical phase
recognition,” in International Conference on Medical Image Computing
and Computer-Assisted Intervention, pp. 604–614, Springer, 2021.
[26] T. Yu, D. Mutter, J. Marescaux, and N. Padoy, “Learning from a tiny
dataset of manual annotations: a teacher/student approach for surgical
phase recognition,” arXiv preprint arXiv:1812.00033, 2018.
[27] G. Yengera, D. Mutter, J. Marescaux, and N. Padoy, “Less is more:
Surgical phase recognition with less annotations through self-supervised
pre-training of cnn-lstm networks,” arXiv preprint arXiv:1805.08569,
2018.
[28] R. DiPietro and G. D. Hager, “Automated surgical activity recognition
with one labeled sequence,” in International conference on medical
image computing and computer-assisted intervention, pp. 458–466,
Springer, 2019.
[29] X. Shi, Y. Jin, Q. Dou, and P.-A. Heng, “Lrtd: long-range temporal
dependency based active learning for surgical workﬂow recognition,”
International Journal of Computer Assisted Radiology and Surgery,
vol. 15, no. 9, pp. 1573–1584, 2020.
[30] X. Shi, Y. Jin, Q. Dou, and P.-A. Heng, “Semi-supervised learning
with progressive unlabeled data excavation for label-efﬁcient surgical
workﬂow recognition,” Medical Image Analysis, vol. 73, p. 102158,
2021.
[31] K. K. Singh and Y. J. Lee, “Hide-and-seek: Forcing a network to
be meticulous for weakly-supervised object and action localization,”
in 2017 IEEE international conference on computer vision (ICCV),
pp. 3544–3553, IEEE, 2017.
[32] L. Wang, Y. Xiong, D. Lin, and L. Van Gool, “Untrimmednets for
weakly supervised action recognition and detection,” in Proceedings
of the IEEE conference on Computer Vision and Pattern Recognition,
pp. 4325–4334, 2017.
[33] P. Nguyen, T. Liu, G. Prasad, and B. Han, “Weakly supervised action
localization by sparse temporal pooling network,” in Proceedings of
the IEEE Conference on Computer Vision and Pattern Recognition,
pp. 6752–6761, 2018.


<!-- page 13 -->
DING et al.: PREPARATION OF PAPERS FOR IEEE TRANSACTIONS ON MEDICAL IMAGING
13
[34] S. Paul, S. Roy, and A. K. Roy-Chowdhury, “W-talc: Weakly-supervised
temporal activity localization and classiﬁcation,” in Proceedings of the
European Conference on Computer Vision (ECCV), pp. 563–579, 2018.
[35] X. Ding, N. Wang, X. Gao, J. Li, X. Wang, and T. Liu, “Weakly
supervised temporal action localization with segment-level labels,” arXiv
preprint arXiv:2007.01598, 2020.
[36] X. Ding, N. Wang, X. Gao, J. Li, X. Wang, and T. Liu, “Kfc: An
efﬁcient framework for semi-supervised temporal action localization,”
IEEE Transactions on Image Processing, vol. 30, pp. 6869–6878, 2021.
[37] X. Ding, N. Wang, S. Zhang, D. Cheng, X. Li, Z. Huang, M. Tang, and
X. Gao, “Support-set based cross-supervision for video grounding,” in
Proceedings of the IEEE/CVF International Conference on Computer
Vision, pp. 11573–11582, 2021.
[38] P. Bojanowski, R. Lajugie, F. Bach, I. Laptev, J. Ponce, C. Schmid, and
J. Sivic, “Weakly supervised action labeling in videos under ordering
constraints,” in European Conference on Computer Vision, pp. 628–643,
Springer, 2014.
[39] D.-A. Huang, L. Fei-Fei, and J. C. Niebles, “Connectionist temporal
modeling for weakly supervised action labeling,” in European Confer-
ence on Computer Vision, pp. 137–153, Springer, 2016.
[40] J. Li, P. Lei, and S. Todorovic, “Weakly supervised energy-based
learning for action segmentation,” in Proceedings of the IEEE/CVF
International Conference on Computer Vision, pp. 6243–6251, 2019.
[41] A. Richard, H. Kuehne, and J. Gall, “Action sets: Weakly supervised
action segmentation without ordering constraints,” in Proceedings of
the IEEE conference on Computer Vision and Pattern Recognition,
pp. 5987–5996, 2018.
[42] P. Mettes, J. C. Van Gemert, and C. G. Snoek, “Spot on: Action
localization from pointly-supervised proposals,” in European conference
on computer vision, pp. 437–453, Springer, 2016.
[43] D. Moltisanti, S. Fidler, and D. Damen, “Action recognition from single
timestamp supervision in untrimmed videos,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
pp. 9915–9924, 2019.
[44] F. Ma, L. Zhu, Y. Yang, S. Zha, G. Kundu, M. Feiszli, and Z. Shou,
“Sf-net: Single-frame supervision for temporal action localization,” in
European conference on computer vision, pp. 420–437, Springer, 2020.
[45] A. Kendall and Y. Gal, “What uncertainties do we need in bayesian deep
learning for computer vision?,” arXiv preprint arXiv:1703.04977, 2017.
[46] Y. Gal and Z. Ghahramani, “Dropout as a bayesian approximation:
Representing model uncertainty in deep learning,” in international
conference on machine learning, pp. 1050–1059, PMLR, 2016.
[47] B. Lakshminarayanan, A. Pritzel, and C. Blundell, “Simple and scalable
predictive uncertainty estimation using deep ensembles,” Advances in
neural information processing systems, vol. 30, 2017.
[48] R. Tanno, D. E. Worrall, A. Ghosh, E. Kaden, S. N. Sotiropoulos,
A. Criminisi, and D. C. Alexander, “Bayesian image quality transfer
with cnns: exploring uncertainty in dmri super-resolution,” in Interna-
tional Conference on Medical Image Computing and Computer-Assisted
Intervention, pp. 611–619, Springer, 2017.
[49] A. Jungo, F. Balsiger, and M. Reyes, “Analyzing the quality and
challenges of uncertainty estimations for brain tumor segmentation,”
Frontiers in neuroscience, p. 282, 2020.
[50] D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” arXiv
preprint arXiv:1312.6114, 2013.
[51] K. Sohn, H. Lee, and X. Yan, “Learning structured output represen-
tation using deep conditional generative models,” Advances in neural
information processing systems, vol. 28, 2015.
[52] M.-H. Laves, S. Ihler, and T. Ortmaier, “Uncertainty quantiﬁcation in
computer-aided diagnosis: Make your model say” i don’t know” for
ambiguous cases,” arXiv preprint arXiv:1908.00792, 2019.
[53] C. Leibig, V. Allken, M. S. Ayhan, P. Berens, and S. Wahl, “Lever-
aging uncertainty information from deep neural networks for disease
detection,” Scientiﬁc reports, vol. 7, no. 1, pp. 1–14, 2017.
[54] G. Wang, W. Li, M. Aertsen, J. Deprest, S. Ourselin, and T. Ver-
cauteren, “Aleatoric uncertainty estimation with test-time augmentation
for medical image segmentation with convolutional neural networks,”
Neurocomputing, vol. 338, pp. 34–45, 2019.
[55] Y. A. Farha and J. Gall, “Ms-tcn: Multi-stage temporal convolutional
network for action segmentation,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pp. 3575–
3584, 2019.
[56] A. P. Twinanda, D. Mutter, J. Marescaux, M. de Mathelin, and N. Padoy,
“Single-and multi-task architectures for surgical workﬂow challenge at
m2cai 2016,” arXiv preprint arXiv:1610.08844, 2016.
[57] A. Twinanda, S. Shehata, D. Mutter, J. Marescaux, M. De Mathelin,
and N. Padoy, “Miccai modeling and monitoring of computer assisted
interventions challenge,” 2016.
[58] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, “Rethinking
the inception architecture for computer vision,” in Proceedings of the
IEEE conference on computer vision and pattern recognition, pp. 2818–
2826, 2016.
[59] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classiﬁcation
with deep convolutional neural networks,” Advances in neural informa-
tion processing systems, vol. 25, pp. 1097–1105, 2012.
[60] N. Ahmidi, L. Tao, S. Sefati, Y. Gao, C. Lea, B. B. Haro, L. Zappella,
S. Khudanpur, R. Vidal, and G. D. Hager, “A dataset and benchmarks
for segmentation and recognition of gestures in robotic surgery,” IEEE
Transactions on Biomedical Engineering, vol. 64, no. 9, pp. 2025–2041,
2017.
[61] X. Li, H. Chen, X. Qi, Q. Dou, C.-W. Fu, and P.-A. Heng, “H-denseunet:
hybrid densely connected unet for liver and tumor segmentation from
ct volumes,” IEEE transactions on medical imaging, vol. 37, no. 12,
pp. 2663–2674, 2018.
[62] E. Gibson, F. Giganti, Y. Hu, E. Bonmati, S. Bandula, K. Gurusamy,
B. Davidson, S. P. Pereira, M. J. Clarkson, and D. C. Barratt, “Automatic
multi-organ segmentation on abdominal ct with dense v-networks,” IEEE
transactions on medical imaging, vol. 37, no. 8, pp. 1822–1834, 2018.
[63] T. Heimann, B. Van Ginneken, M. A. Styner, Y. Arzhaeva, V. Aurich,
C. Bauer, A. Beck, C. Becker, R. Beichel, G. Bekes, et al., “Comparison
and evaluation of methods for liver segmentation from ct datasets,” IEEE
transactions on medical imaging, vol. 28, no. 8, pp. 1251–1265, 2009.
[64] X. Li, L. Yu, H. Chen, C.-W. Fu, L. Xing, and P.-A. Heng,
“Transformation-consistent self-ensembling model for semisupervised
medical image segmentation,” IEEE Transactions on Neural Networks
and Learning Systems, vol. 32, no. 2, pp. 523–534, 2020.
[65] T. Wang, X. Xu, J. Xiong, Q. Jia, H. Yuan, M. Huang, J. Zhuang, and
Y. Shi, “Ica-unet: Ica inspired statistical unet for real-time 3d cardiac
cine mri segmentation,” in International conference on medical image
computing and computer-assisted intervention, pp. 447–457, Springer,
2020.
[66] X. Li, Q. Dou, H. Chen, C.-W. Fu, X. Qi, D. L. Belav`y, G. Armbrecht,
D. Felsenberg, G. Zheng, and P.-A. Heng, “3d multi-scale fcn with
random modality voxel dropout learning for intervertebral disc localiza-
tion and segmentation from multi-modality mr images,” Medical image
analysis, vol. 45, pp. 41–54, 2018.
[67] Y. Yu, Y. Xie, T. Thamm, E. Gong, J. Ouyang, C. Huang, S. Christensen,
M. P. Marks, M. G. Lansberg, G. W. Albers, et al., “Use of deep learning
to predict ﬁnal ischemic stroke lesions from initial magnetic resonance
imaging,” JAMA network open, vol. 3, no. 3, pp. e200772–e200772,
2020.