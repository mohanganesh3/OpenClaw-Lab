<!-- page 1 -->
arXiv:2505.12418v1  [eess.IV]  18 May 2025
Mutual Evidential Deep Learning for Medical
Image Segmentation
1st Yuanpeng He
Key Laboratory of High Confidence
Software Technologies (MOE)
School of Computer Science
Peking University, Beijng, China
heyuanpeng@stu.pku.edu.cn
2nd Yali Bi
College of Computer and Information Science
School of Software
Southwest University, Chongqing, China
biyali812@outlook.com
3rd Lijian Li
Department of Computer and
Information Science
University of Macau, Macau, China
mc35305@umac.mo
4th Chi-Man Pun
Department of Computer and
Information Science
University of Macau, Macau, China
cmpun@um.edu.mo
5th Wenpin Jiao
Key Laboratory of High Confidence
Software Technologies (MOE)
School of Computer Science
Peking University, Beijng, China
jwp@pku.edu.cn
6th Zhi Jin∗
Key Laboratory of High Confidence
Software Technologies (MOE)
School of Computer Science
Peking University, Beijng, China
zhijin@pku.edu.cn
Abstract—Existing semi-supervised medical segmentation co-
learning frameworks have realized that model performance can
be diminished by the biases in model recognition caused by
low-quality pseudo-labels. Due to the averaging nature of their
pseudo-label integration strategy, they fail to explore the relia-
bility of pseudo-labels from different sources. In this paper, we
propose a mutual evidential deep learning (MEDL) framework
that offers a potentially viable solution for pseudo-label gener-
ation in semi-supervised learning from two perspectives. First,
we introduce networks with different architectures to generate
complementary evidence for unlabeled samples and adopt an
improved class-aware evidential fusion to guide the confident syn-
thesis of evidential predictions sourced from diverse architectural
networks. Second, utilizing the uncertainty in the fused evidence,
we design an asymptotic Fisher information-based evidential
learning strategy. This strategy enables the model to initially
focus on unlabeled samples with more reliable pseudo-labels,
gradually shifting attention to samples with lower-quality pseudo-
labels while avoiding over-penalization of mislabeled classes in
high data uncertainty samples. Additionally, for labeled data,
we continue to adopt an uncertainty-driven asymptotic learning
strategy, gradually guiding the model to focus on challenging
voxels. Extensive experiments on five mainstream datasets have
demonstrated that MEDL achieves state-of-the-art performance.
Index Terms—Semi-supervised medical segmentation, Mutual
evidential deep learning, Class-aware evidential fusion
I. INTRODUCTION
Medical image segmentation (MIS) [1] is a formidable un-
dertaking at the intersection of computer vision and healthcare,
characterized by intricate challenges that demand meticulous
This work is supported by National Key R&D Program of China
(2023YFC3502900),
National
Natural
Science
Foundation
of
China
(granted No. 62192731) and National Key R&D Program of China
(2021YFF1201100). Corresponding author: Zhi Jin.
consideration. These challenges encompass the inherent mul-
timodality of medical data, evident disparities from natural
images, and the difficulty of manual annotation [2]. Mul-
timodality arises due to the diverse imaging modalities in
use, such as CT scans and MRI, each marked by unique
and complex image attributes. The substantial divergence
from natural images is evident in the wide range of pixel
values, the prevalence of artifacts and noise, and the lack
of well-defined object boundaries, all of which significantly
complicate conventional segmentation methods. Furthermore,
high-quality annotated medical image datasets are notably
constrained in scale, rendering semi-supervised learning a
pragmatic and scalable solution for addressing the intrinsic
challenges to MIS.
The current landscape of MIS methods can be broadly
categorized into two main approaches. The first emphasizes
uncertainty estimation to improve model robustness and pre-
diction quality [3]. In this context, uncertainty-aware methods
[4] play a pivotal role by quantifying the model’s confidence
in its predictions, thus enabling reliable refinement and error
correction. Many recent works further advance this line of
research by integrating evidential theory into segmentation
models [5]–[7], offering a probabilistic framework for mod-
eling aleatoric and epistemic uncertainties. The second cate-
gory leverages semi-supervised strategies by jointly training
on labeled and unlabeled data [8], often incorporating data
augmentation, consistency regularization, and pseudo-labeling.
Techniques such as prototype-based embedding refinement [9],
prototype consistency [10], and federated knowledge transfer
[11] have shown significant improvements in performance
and generalization. Nevertheless, a persistent challenge across
existing approaches is the effective mitigation of recognition
biases and the optimal exploitation of uncertainty information.


<!-- page 2 -->
To address these issues, the mutual evidential deep learning
(MEDL) framework [12] introduces an advanced methodology
that combines evidential deep learning (EDL) with evidential
fusion mechanisms [5], [6]. Specifically, MEDL is designed
to model voxel-wise uncertainty and adjust the learning strat-
egy per voxel by fusing evidence from two complementary
networks, each with its own architectural bias. Such fusion
draws inspiration from generalized evidence theory [13], or-
dinal belief entropy [14], and conflict management methods
in belief assignment [15]–[17], enhancing both robustness
and adaptability. In the MEDL framework, two segmentation
networks are employed to generate evidential predictions on
both labeled and unlabeled data. For the unlabeled portion,
class-aware evidential fusion (CAEF) integrates predictions
from both networks to produce reliable pseudo-labels and
associated uncertainty estimates [7], [18]. Reliability-based
masking, inspired by entropy theory and evidential entropy
[17], is applied to suppress noisy predictions. A progressive
Fisher Information-based Evidential Learning (FIE) module
then facilitates a curriculum-style training regime, emphasiz-
ing confident voxels and gradually incorporating uncertain
ones. This design also benefits from recent studies on temporal
evidence fusion in time series forecasting [19], matrix-based
decision methods [20], and adaptive entropy-driven clustering
[21].
Additionally, the MEDL pipeline borrows from attention-
based weakly-supervised localization [22], neural structure
modeling [23], and fuzzy forecasting principles [24], integrat-
ing them into a coherent evidential framework. The use of
feature-reutilization networks [25] further boosts segmentation
accuracy by enhancing spatial contextual representation. Inter-
estingly, MEDL’s iterative pseudo-label refinement and pro-
gressive training paradigm resemble self-debugging techniques
in code generation systems [26], where a model generates
intermediate outputs and continuously self-corrects through
structured evaluations. Moreover, its ability to leverage het-
erogeneous predictions across modalities can be analogized to
multi-temporal image classification frameworks [27], which
address spatio-temporal variability across complex datasets.
In summary, the MEDL approach not only leverages the
strengths of evidential modeling and uncertainty estimation
but also inherits and expands upon recent developments in
fuzzy systems, deep evidential learning, and semi-supervised
segmentation. As such, it presents a promising direction for
future research on robust, scalable, and uncertainty-aware
medical image segmentation. The main contributions of this
work are given as follows:
• The proposed method designs CAEF to combine com-
plementary evidences from two heterogeneous segmenta-
tion models. This technique enhances the confidence of
labeled data predictions and generates reliable pseudo-
labels by filtering out uncertain information.
• The method proposes an innovative strategy for asymp-
totic learning using uncertainty-based estimations. By
integrating the evidential predictions from two networks
and their built-in uncertainties, which guides the easy-to-
hard and certain-to-uncertain learning process.
• Experiments strongly demonstrate the effectiveness of
MEDL which far outperforms previous state of the arts.
II. METHOD
The details of the proposed method are provided in the
Figure 1. Specifically, the mutual evidential deep learning
framework consists of two networks, N1(Θ1) and N2(Θ2),
where Θ1 and Θ2 are parameters. For the semi-supervised
medical image segmentation task, the training dataset can be
divided into two parts, labeled data Dl = {(X l
k, Yl
k)}A
i=1 and
unlabeled data Du = {(X u
k )}A+B
j=A+1 where A ≪B. Besides,
the volume of 3D medical image and ground truth can be
defined as X ∈RH×W ×L, and Y ∈{0, 1, ..., K −1}H×W ×L
which indicates where targets and backgrounds are in X.
A. Backgoround of Evidential Deep Learning
Dempster-Shafer evidence theory’s notion of belief mass
over a frame of discernment as a Dirichlet Distribution is
formalized by Subjective logic [28]. Therefore, it enables the
application of evidence theory principles to quantify belief
masses and uncertainty within a clearly defined theoretical
framework. Specifically, in the field of classification task, the
belief mass of each category, overall uncertainty mass and
corresponding predicted probability can be given as [29]:
bn = en/S,
u = K/S,
pn = αn/S,
K−1
X
n=0
bn + u = 1
(1)
where u ≥0, bn ≥0, αn = en + 1, α = [α0, ..., αK−1)] and
S = PK−1
n=0 αn, n = 0, ..., K −1.
B. Generalized Probabilistic Framework
In this work, we extend EDL by integrating multiple
objective sets in traditional evidence theory [30]. We map
the inherent uncertainty of evidential deep learning onto the
universal set corresponding to the discernment framework
of traditional evidence theory [31]. This approach provides
greater flexibility in the process of evidence-based feature
fusion and uncertainty measurement. Specifically, let b(x,y,z)
N1,l
,
b(x,y,z)
N2,l
, b(x,y,z)
N1,u
and b(x,y,z)
N2,u
∈RC represent the belief mass
from the evidential classifier predictive results [32] for a voxel
at position (x, y, z) by the segmentation output of sub-network
N1 and N2 on labeled and unlabeled images respectively.
Moreover, K represents the number of classes. We define
the basic probability mass assignments for a voxel as M =
{{b(x,y,z)
(Cn) }K−1
n=0 , u(x,y,z)}, with u(x,y,z) = 1 −PK−1
n=0 b(x,y,z)
(Cn)
signifying the original uncertainty. The generalized probability
mass assignments which maps the u(x,y,z) to multiple objec-
tive sets are then expressed as:
M(x,y,z) = {{b(x,y,z)
(Cn) }K−1
n=0 , b(x,y,z)
(CK) }, CK = {C0, ..., CK−1}
(2)
where CK denotes a multiple sets whose cardinality is K −1.

[CAPTION] Figure 1. Specifically, the mutual evidential deep learning


<!-- page 3 -->
CAEF
Reliability 
Map
Reliability 
Map
Uncertainty 
Average
Uncertainty 
Average
UMAL
Uncertainty 
Sorting
Uncertainty 
Measure
Uncertainty 
Measure
Uncertainty 
Sorting
Uncertainty 
Sorting
UMAL-FIE
UMAL-FIE
UMAL
Uncertainty 
Sorting
 "!   #  
 "!   #    
 "!   #  
1
1- Ξ
1+ Ξ
=
Fig. 1: The overview of Mutual Evidential Deep Learning framework. We use two different segmentation models to predict
evidence for labeled and unlabeled data. For unlabeled data, class-aware evidential fusion (CAEF) combines complementary
evidence from the two models to generate pseudo-labels and reliability measures. Reliability measures mask the original
evidential predictions from each network, and evidential deep learning (EDL) uncertainty with weighted average estimates
rank each voxel, applying asymptotic Fisher information-based EDL (FIE). For labeled data, a similar strategy guides learning
without the need for reliability-based masking and FIE (⊙represents the fusion operation).
C. Mutual Evidential Interaction for Pseudo-label Generation
In the field of semi-supervised medical segmentation, most
methods adopt a teacher-student network structure, where
the teacher network is responsible for generating pseudo-
labels for unlabeled data to guide the student network in the
corresponding feature learning [33]. However, some related
work has abandoned this structure and instead uses two sub-
networks to independently output pseudo-labels to guide the
learning for unlabeled data, achieving quite excellent results
[34]. However, such a design may lead to potential problems
like error accumulation and increased model bias. These issues
could be caused by incorrect pseudo-labels generated by one
of the networks, or when both networks have similar structures
or biases, they may produce similar errors, leading to a lack of
diversity in pseudo-labels [35]. In these cases, the model may
not be able to fully learn from the data, thus failing to achieve
optimal performance. Therefore, we propose a probability in-
teraction strategy evolving from Dempster’s combination rule
[30], where it allows the pseudo-label probability distributions
generated by the two sub-networks to be reallocated and
interact with each other, which can be defined as (the subscript
u is omitted for simplicity):
b(x,y,z)
(Cn)
= b(x,y,z)
N1,(Cn)b(x,y,z)
N2,(Cn) +
|Cn|
|Cn| + |CN|
· (b(x,y,z)
N1,(Cn)b(x,y,z)
N2,(CN ) + b(x,y,z)
N2,(Cn)b(x,y,z)
N1,(CK))
(3)
where b(x,y,z)
N1,(Cn) and b(x,y,z)
N2,(Cn) represent prediction of sub-
network N1 and N2 on unlabeled data for class Cn. Besides,
the fused probability mass assignments are supposed to be
normalized, b(x,y,z)
(CK)
= b(x,y,z)
N1,(CK)b(x,y,z)
N2,(CK), |Cn| = 1 and
|CK| = K. After the fusion of the predictions made by
two sub-networks on unlabeled data, we can obtain the fused
pseudo-labels for the unlabeled data. The significance of the
coefficient
|Ck|
|Ck|+|CN| is to prevent the values corresponding to
the multiple objective set associated with uncertainty from be-
ing too trivial. We will also proceed to dynamically adjust the
learning at the voxel level for the optimized uncertainty, which
is illustrated in the next subsection. For contentious voxels,
the corresponding pseudo-labels do not explicitly indicate the
categories of voxel segmentation. Instead, the segmentation of
these voxels is determined by the model learning the potential
semantic connections among the surrounding voxels. Utilizing
Eq. 3, we can integrate the probability distributions of pseudo-
labels generated by the two sub-networks to obtain the final
pseudo-labels for the unlabeled data. The predictions of the
two networks for the unlabeled data undergo the same loss
calculation process as in the training phase with labeled data.
Unlike the training phase with labeled data, we consider the
significant uncertainty that may exist in the pseudo-labeled
data [36], and recognize that further reinforcing the model’s
learning of pseudo-label patterns may not aid in enhancing
the model’s segmentation performance on test data. Therefore,
based on the integrated pseudo-labels, we propose using the
uncertainty obtained during interaction for each voxel to
compute its reliability. Prior to the loss calculation between
pseudo-labels and predictions, we perform an operation on
the pseudo-labels akin to an attention-based mask mechanism.
Given the effectiveness of Shannon entropy [37], the defini-
tion of reliability synthesizing fused uncertainty for voxel at
position (x, y, z) can be given as:
R(x,y,z) = eb(x,y,z)
(CK )
PN−1
n=0 (ζnlog2ζn)
(4)
where ζn represents b(x,y,z)
(Cn)
and e denotes natural constant.
We adopt the same strategy as related work [38], utilizing
reliability to mask the generated pseudo-labels, thereby further
optimizing the quality of the pseudo-labels and reducing the
likelihood of misleading the model during the learning process
to generated the final pseudo-labels ˆy for each voxel.
D. Asymptotic Fisher Evidential Deep Learning
We propose optimizing the model labeled learning process
by utilizing the way of uncertainties estimation designed for
unlabeled learning part and the model’s voxel-wise evidential

[CAPTION] Fig. 1: The overview of Mutual Evidential Deep Learning framework. We use two different segmentation models to predict


<!-- page 4 -->
Fig. 2: Visualization results on the LA, Pancreas, and TBAD datasets. The first, second, and third rows show the results of
the comparison method, the proposed method, and the ground truth (GT).
TABLE I: Comparisons with SOTA models on TBAD dataset in labeled ratio 5%, 10% and 20%
Model
Scans Used
Metrics
Dice↑
Jaccard↑
95HD↓
ASD↓
Labeled
Unlabled
TL
FL
Mean
TL
FL
Mean
TL
FL
Mean
TL
FL
Mean
V-Net
(20%)
(80%)
55.51
48.98
52.25
39.81
34.79
37.30
7.24
10.17
8.71
1.27
3.19
2.23
(100%)
(0%)
75.98
64.02
70.00
61.89
50.05
55.97
3.16
7.56
5.36
0.48
2.44
1.46
MT [39]
(20%)
(80%)
57.62
49.95
53.78
41.57
35.52
38.54
6.00
8.98
7.49
0.97
2.77
1.87
UA-MT [40]
70.91
60.66
65.78
56.15
46.24
51.20
4.44
7.94
6.19
0.83
2.37
1.60
FUSSNet [41]
79.73
65.32
72.53
67.31
51.74
59.52
3.46
7.87
5.67
0.61
2.93
1.77
URPC [42]
81.84
69.15
75.50
70.35
57.00
63.68
4.41
9.13
6.77
0.93
1.11
1.02
UPCoL [43]
82.65
69.74
76.19
71.49
57.42
64.45
2.82
6.81
4.82
0.43
2.22
1.33
Ours
(5%)
(95%)
81.83
73.51
77.67
69.78
60.56
65.17
2.54
4.77
3.66
0.30
1.32
0.81
(10%)
(90%)
83.76
75.16
79.46
72.53
62.69
67.61
2.11
4.67
3.39
0.33
1.32
0.83
(20%)
(80%)
84.16
75.45
79.80
73.21
62.88
68.05
2.25
4.57
3.41
0.30
1.22
0.76
predictions of labeled data. For learning with labeled data, we
find that integrating the predictions of the two sub-networks
in the same manner as with pseudo-labels, results in certain
performance gains. The measurement of uncertainty for each
voxel in the labeled data learning phase is the weighted mean
of the original and fused uncertainty values, which can be
given as (the subscript l is omitted for simplicity):
bfinal
Ni,CK = λ1b(x,y,z)
Ni,CK + λ2b(x,y,z)
(CK) ,
λ1 + λ2 = 1
(5)
where i = 1, 2, bfinal
Ni,CK and b(x,y,z)
(CK)
represent final uncertainty
measure and the one from combined evidence for labeled data
voxel at position (x, y, z). Based on the optimized uncertainty
measurement of voxel from sub-networks on labeled data
and original uncertainty measure b(x,y,z)
(CK)
in pseudo-labels for
unlabeled data from evidential predictive results, we propose
to evaluate uncertainty of the model’s prediction and pseudo-
labels for each voxel in labeled and unlabeled data to guide the
model for targeted learning. We aim for the model to initially
focus on voxels and pseudo-labels with smaller uncertainties,
and then gradually shift towards understanding features of
voxels with higher uncertainties and the ones corresponding
to more unreliable pseudo-labels as the training progresses.
Drawing inspiration from curriculum learning [44], the pro-
posed approach enhances the learning of complex information,
building on a firm grasp of simpler characteristics. The voxel-
level weight adjustment function is defined as:
ω(q, v) = Ξ · Tanh(ψ(h(v))ζ(q)) + 1
(6)
where Ξ is devised to control the change amplitude of dynamic
weights. ψ(h(v)) = 2h(v)
V
−1 ∈[−1, 1], v = 1, ..., V , h(v)
represents the rank number of voxel v through sorting uncer-
tainties of voxels in labeled data and pseudo-labels in a de-
scending order. Besides, ζ(q) = 2q
Q −1 ∈[−1, 1], q = 1, ..., Q,
where Q denotes the total number of total training epochs and
q is the current epoch index. For unlabeled data, considering
the importance of modeling uncertainties contained in classes,
we propose to integrate the voxel-level weight for pseudo-
labels with the KL term-free optimization objective presented
by fisher information-based evidential deep learning I-EDL
[45] for a more robust learning pattern on voxel features from
unlabeled data [46], which can be defined as (the subscript Cn
is replaced by n and superscript (x, y, z) is omitted):
LI
j,v = ω(q, v)(
K−1
X
n=0
((ˆyvn −αvn
Sv
)2 + αvn(Sv −αvn)
S2v(Sv + 1)
)
ψ1(αvn) −λ2 log |I(αv)|),
(7)
where ˆyv represents one-hot encoded ground-truth of vth voxel
and Lfiel
j,v denotes the loss between vth voxel in jth unlabeled
sample and corresponding integrated pseudo-labels.
E. Optimization Objective
Update of model parameters is divided into learning from
labeled and unlabeled data [34], [43]. For labeled data, the
sub-networks’ predictions are compared with the labels to
compute Dice and Cross-Entropy loss, resulting in LN1
l
and
LN2
l
. Utilizing the both kinds of uncertainties from predictions
of labeled data and generated pseudo-labels, we can apply
weighting to the loss calculation for each voxel of labeled
data and unlabeled data. The process generating asymptotic
loss for labeled and unlabeled data can be given as:
Lϵ
w,l =
A
X
i=1
V
X
vl=1
ω(q, vl)Lϵ
i,vl/V,
Lϵ,I
u
=
B
X
j=1
V ′
X
vu=1
LI
j,vu/V ′
(8)


**[Table p4.1]**
|  | Scans Used |  |  |  |
| --- | --- | --- | --- | --- |
|  |  | Dice↑ | Jaccard↑ | 95HD↓ |
|  | Labeled Unlabled | TL FL Mean | TL FL Mean | TL FL Mean |
|  | (20%) (80%) (100%) (0%) | 55.51 48.98 52.25 75.98 64.02 70.00 | 39.81 34.79 37.30 61.89 50.05 55.97 | 7.24 10.17 8.71 3.16 7.56 5.36 |
|  |  | 57.62 49.95 53.78 70.91 60.66 65.78 | 41.57 35.52 38.54 56.15 46.24 51.20 | 6.00 8.98 7.49 4.44 7.94 6.19 |
| FUSSNet [41] URPC [42] UPCoL [43] Ours | (20%) (80%) | 79.73 65.32 72.53 81.84 69.15 75.50 82.65 69.74 76.19 | 67.31 51.74 59.52 70.35 57.00 63.68 71.49 57.42 64.45 | 3.46 7.87 5.67 4.41 9.13 6.77 2.82 6.81 4.82 |
|  | (5%) (95%) (10%) (90%) (20%) (80%) | 81.83 73.51 77.67 83.76 75.16 79.46 84.16 75.45 79.80 | 69.78 60.56 65.17 72.53 62.69 67.61 73.21 62.88 68.05 | 2.54 4.77 3.66 2.11 4.67 3.39 2.25 4.57 3.41 |

[CAPTION] Fig. 2: Visualization results on the LA, Pancreas, and TBAD datasets. The first, second, and third rows show the results of


<!-- page 5 -->
TABLE II: Comparisons with SOTA models on three dataset in labeled ratio 5%, 10% and 20%
Pancreas-CT
dataset
Labeled Ratio
Metrics
UA-MT
SASSNet
DTC
URPC
MC-Net
SS-Net
Co-BioNet
BCP
A&D
Ours
(5%)
Dice↑
47.03
56.05
49.83
52.05
54.99
56.35
79.74
80.33
81.65
82.14
Jaccard↑
32.79
41.56
34.47
36.47
40.65
43.41
65.66
67.65
69.11
69.86
95HD↓
35.31
36.61
41.16
34.02
16.03
22.75
5.43
11.78
15.01
8.75
ASD↓
4.26
4.90
16.53
13.16
3.87
5.39
2.79
4.32
4.53
2.91
(10%)
Dice↑
66.96
66.69
67.28
64.73
69.07
67.40
82.49
81.54
82.25
84.14
Jaccard↑
51.89
51.66
52.86
49.62
54.36
53.06
67.88
69.29
70.17
72.86
95HD↓
21.65
18.88
17.74
21.90
14.53
20.15
6.51
12.21
14.44
9.90
ASD↓
6.25
5.76
1.97
7.73
2.28
3.47
3.26
3.80
4.53
2.72
(20%)
Dice↑
77.26
77.66
78.27
79.09
78.17
79.74
84.01
82.91
82.56
84.93
Jaccard↑
63.82
64.08
64.75
65.99
65.22
65.42
70.00
70.97
70.69
73.95
95HD↓
11.90
10.93
8.36
11.68
6.90
12.44
5.35
6.43
11.78
6.24
ASD↓
3.06
3.05
2.25
3.31
1.55
2.69
2.75
2.25
3.42
2.21
LA
dataset
Labeled Ratio
Metrics
UA-MT
SASSNet
DTC
URPC
MC-Net
SS-Net
Co-BioNet
BCP
A&D
Ours
(5%)
Dice↑
82.26
81.60
81.25
86.92
87.62
86.33
76.88
88.02
89.93
90.49
Jaccard↑
70.98
69.63
69.33
77.03
78.25
76.15
66.76
78.72
81.82
82.75
95HD↓
13.71
16.16
14.90
11.13
10.03
9.97
19.09
7.90
5.25
5.95
ASD↓
3.82
3.58
3.99
2.28
1.82
2.31
2.30
2.15
1.86
1.83
(10%)
Dice↑
87.79
87.54
87.51
86.92
87.62
88.55
89.20
89.62
90.31
91.55
Jaccard↑
78.39
78.05
78.17
77.03
78.25
79.62
80.68
81.31
82.40
84.46
95HD↓
8.68
9.84
8.23
11.13
10.03
7.49
6.44
6.81
5.58
5.65
ASD↓
2.12
2.59
2.36
2.28
1.82
1.90
1.90
1.76
1.64
1.60
(20%)
Dice↑
88.88
89.54
89.42
88.43
90.12
89.25
91.26
90.34
90.42
91.95
Jaccard↑
80.21
81.24
80.98
81.15
82.12
81.62
83.99
82.50
82.72
85.14
95HD↓
7.32
8.24
7.32
8.21
11.28
6.45
5.17
6.75
6.33
5.11
ASD↓
2.26
1.99
2.10
2.35
2.30
1.80
1.64
1.77
1.57
1.37
ACDC
dataset
Labeled Ratio
Metrics
UA-MT
SASSNet
DTC
URPC
MC-Net
SS-Net
Co-BioNet
BCP
A&D
Ours
(5%)
Dice↑
46.04
57.77
56.90
55.87
62.85
65.83
87.46
87.59
86.51
89.60
Jaccard↑
35.97
46.14
45.67
44.64
52.29
55.38
77.93
78.67
76.61
81.67
95HD↓
20.08
20.05
23.36
13.60
7.62
6.67
1.11
1.90
2.13
4.80
ASD↓
7.75
6.06
7.39
3.74
2.33
2.28
0.41
0.67
0.84
1.31
(10%)
Dice↑
81.65
84.50
84.29
83.10
86.44
86.78
88.49
88.84
88.12
90.40
Jaccard↑
70.64
74.34
73.92
72.41
77.04
77.67
79.76
80.62
79.39
82.94
95HD↓
6.88
5.42
12.81
4.84
5.50
6.07
3.70
3.98
13.03
3.38
ASD↓
2.02
1.86
4.01
1.53
1.84
1.40
1.14
1.17
3.21
0.83
(20%)
Dice↑
85.61
86.45
87.10
85.44
87.04
87.41
89.51
89.12
88.85
90.95
Jaccard↑
75.49
77.20
78.15
76.36
78.01
78.82
81.64
81.03
80.62
83.80
95HD↓
5.91
6.63
6.76
5.93
5.35
4.79
4.72
3.40
4.26
3.84
ASD↓
1.79
1.98
1.99
1.70
1.67
1.48
1.52
0.97
1.39
0.85
Fig. 3: Visualization results on the ACDC dataset. The first,
second, and third rows are the results of the comparison
method, the proposed method, and the ground truth (GT).
where ϵ = N1, N2, V
and V ′ represent a total number
of voxels in a labeled and unlabeled sample. Besides, Lϵ
i,v
denotes the loss of vth voxel in ith sample and h(v) denotes
corresponding ordinal number of voxel in ith sample. For
the unlabeled data part, the prediction of each sub-network
is compared with the optimized generated pseudo-labels to
compute the losses consisting of Dice and Cross-Entropy loss,
denoted as LN1
u
and LN2
u . For the update of each sub-network’s
parameters, the final optimization objective can be given as:
Lϵ
medl = Lϵ
l + Lϵ
u + Lϵ
w,l + λGW ULϵ,I
u
(9)
where λGW U are self-adaptive hyper-parameters.
III. EXPERIMENTS
In our framework, subnet N1 is based on the VNet ar-
chitecture, which is widely recognized for its effectiveness
in medical image segmentation. To facilitate error correction
between subnets, it’s essential that the performance disparity
among the diverse subnets is minimal. To this end, we have
substituted VNet’s encoder with a 3D ResNet34, creating a
second subnet N2 known as 3D-ResVNet. When processing
the data for final output, the system computes the mean of
both subnet outputs. PyTorch is employed to run this system,
leveraging the computational power of an NVIDIA 4090
GPU. Our parameter configurations align well with established
benchmarks for comparison. We employ an SGD optimizer
to modify network parameters, setting the weight decay to
0.0001 and the momentum to 0.9. The learning rate starts at
0.01, reducing by a factor of 10 at intervals of 2500 iterations,
up to 6000 iterations in total. We utilize a batch size of
four, comprising two labeled and two unlabeled data volumes.
Following [47], Gaussian warming up function is utilized to
control the weight λGW U.


**[Table p5.1]**
| Labeled Ratio | Metrics UA-MT SASSNet DTC URPC MC-Net SS-Net Co-BioNet BCP A&D Ours |  |
| --- | --- | --- |
| (5%) | Dice↑ 47.03 56.05 49.83 52.05 54.99 56.35 79.74 80.33 81.65 82.14 Jaccard↑ 32.79 41.56 34.47 36.47 40.65 43.41 65.66 67.65 69.11 69.86 95HD↓ 35.31 36.61 41.16 34.02 16.03 22.75 5.43 11.78 15.01 8.75 ASD↓ 4.26 4.90 16.53 13.16 3.87 5.39 2.79 4.32 4.53 2.91 |  |
| (10%) | Dice↑ 66.96 66.69 67.28 64.73 69.07 67.40 82.49 81.54 82.25 84.14 Jaccard↑ 51.89 51.66 52.86 49.62 54.36 53.06 67.88 69.29 70.17 72.86 95HD↓ 21.65 18.88 17.74 21.90 14.53 20.15 6.51 12.21 14.44 9.90 ASD↓ 6.25 5.76 1.97 7.73 2.28 3.47 3.26 3.80 4.53 2.72 |  |
| (20%) | Dice↑ 77.26 77.66 78.27 79.09 78.17 79.74 84.01 82.91 82.56 84.93 Jaccard↑ 63.82 64.08 64.75 65.99 65.22 65.42 70.00 70.97 70.69 73.95 95HD↓ 11.90 10.93 8.36 11.68 6.90 12.44 5.35 6.43 11.78 6.24 ASD↓ 3.06 3.05 2.25 3.31 1.55 2.69 2.75 2.25 3.42 2.21 |  |
| Labeled Ratio | Metrics UA-MT SASSNet DTC URPC MC-Net SS-Net Co-BioNet BCP A&D Ours |  |
| (5%) | Dice↑ 82.26 81.60 81.25 86.92 87.62 86.33 76.88 88.02 89.93 90.49 Jaccard↑ 70.98 69.63 69.33 77.03 78.25 76.15 66.76 78.72 81.82 82.75 95HD↓ 13.71 16.16 14.90 11.13 10.03 9.97 19.09 7.90 5.25 5.95 ASD↓ 3.82 3.58 3.99 2.28 1.82 2.31 2.30 2.15 1.86 1.83 |  |
| (10%) | Dice↑ 87.79 87.54 87.51 86.92 87.62 88.55 89.20 89.62 90.31 91.55 Jaccard↑ 78.39 78.05 78.17 77.03 78.25 79.62 80.68 81.31 82.40 84.46 95HD↓ 8.68 9.84 8.23 11.13 10.03 7.49 6.44 6.81 5.58 5.65 ASD↓ 2.12 2.59 2.36 2.28 1.82 1.90 1.90 1.76 1.64 1.60 |  |
| (20%) | Dice↑ 88.88 89.54 89.42 88.43 90.12 89.25 91.26 90.34 90.42 91.95 Jaccard↑ 80.21 81.24 80.98 81.15 82.12 81.62 83.99 82.50 82.72 85.14 95HD↓ 7.32 8.24 7.32 8.21 11.28 6.45 5.17 6.75 6.33 5.11 ASD↓ 2.26 1.99 2.10 2.35 2.30 1.80 1.64 1.77 1.57 1.37 |  |
| Labeled Ratio | Metrics UA-MT SASSNet DTC URPC MC-Net SS-Net Co-BioNet BCP A&D Ours |  |
| (5%) | Dice↑ 46.04 57.77 56.90 55.87 62.85 65.83 87.46 87.59 86.51 89.60 Jaccard↑ 35.97 46.14 45.67 44.64 52.29 55.38 77.93 78.67 76.61 81.67 95HD↓ 20.08 20.05 23.36 13.60 7.62 6.67 1.11 1.90 2.13 4.80 ASD↓ 7.75 6.06 7.39 3.74 2.33 2.28 0.41 0.67 0.84 1.31 |  |
| (10%) | Dice↑ 81.65 84.50 84.29 83.10 86.44 86.78 88.49 88.84 88.12 90.40 Jaccard↑ 70.64 74.34 73.92 72.41 77.04 77.67 79.76 80.62 79.39 82.94 95HD↓ 6.88 5.42 12.81 4.84 5.50 6.07 3.70 3.98 13.03 3.38 ASD↓ 2.02 1.86 4.01 1.53 1.84 1.40 1.14 1.17 3.21 0.83 |  |
|  |  |  |
| (20%) | Dice↑ 85.61 86.45 87.10 85.44 87.04 87.41 89.51 89.12 88.85 Jaccard↑ 75.49 77.20 78.15 76.36 78.01 78.82 81.64 81.03 80.62 95HD↓ 5.91 6.63 6.76 5.93 5.35 4.79 4.72 3.40 4.26 ASD↓ 1.79 1.98 1.99 1.70 1.67 1.48 1.52 0.97 1.39 | 90.95 83.80 3.84 0.85 |
|  |  | ive ca GW U amete ed on for it itate e erfor To thi ResNe et. W mpute d to r f an well y an |

[CAPTION] Fig. 3: Visualization results on the ACDC dataset. The first,


<!-- page 6 -->
TABLE III: Comparison with other methods on the ACDC test
set. DSC (%) and ASSD (mm) are reported with selected 14 la-
beled scans and 126 unlabeled scans [48] for semi-supervised
training. The bold font represents the best performance.
Method
Scans
RV
Myo
L/U
DSC ↑
JAC ↑
ASD ↓
95HD ↓
DSC ↑
JAC ↑
ASD ↓
95HD ↓
CPS [49]
14/126
83.51
71.30
0.46
4.20
85.91
74.64
1.17
5.34
ASE-Net [50]
14/126
83.05
72.72
0.46
4.20
85.61
75.18
1.17
5.34
CoraNet [51]
14/126
83.05
72.72
0.46
4.20
85.61
75.18
1.17
5.34
MCF [34]
14/126
83.76
74.16
1.53
2.74
85.40
75.18
1.08
6.35
ETC-Net [48]
14/126
86.52
77.28
1.48
5.52
85.66
75.26
0.82
3.50
Ours
14/126
89.01
80.80
7.28
1.70
86.43
76.40
2.87
0.98
Method
Scans
LV
Avg
L/U
DSC ↑
JAC ↑
ASD ↓
95HD ↓
DSC ↑
JAC ↑
ASD ↓
95HD ↓
CPS [49]
14/126
89.86
80.28
2.14
6.42
86.43
75.40
1.26
5.32
ASE-Net [50]
14/126
89.86
80.28
2.14
6.42
86.18
76.06
1.26
5.32
CoraNet [51]
14/126
89.86
80.28
2.14
6.42
86.18
76.06
1.26
5.32
MCF [34]
14/126
89.15
81.48
2.35
7.08
86.77
76.27
1.62
5.39
ETC-Net [48]
14/126
92.07
85.68
1.43
4.25
88.08
79.40
1.24
4.42
Ours
14/126
93.34
87.81
4.25
1.23
90.40
82.94
3.38
0.83
A. Datasets Description
The Left Atrium Dataset (LA).
The LA dataset [52]
consists
of
100
3D
gadolinium-enhanced
MR
imaging
volumes. Its spatial resolution is 0.625 × 0.625 × 0.625mm3.
The dataset is divided into 5 folds, each with 20 volumes.
For the stage of preprocessing, all volumes are normalized to
zero mean and unit variance, and then the extensive edges of
each 3D MRI volume are cropped according to the targets. In
the process of training, we randomly crop the training volume
to 112 × 112 × 80 and use it as the input. For the inference
stage, we utilize the same size sliding window and a stride of
18 × 18 × 4 to obtain the final segmentation results.
The
Pancreas-CT
Dataset.
Pancreas-CT dataset [53]
contains 82 contrastk-enhanced abdominal annotated 3D CT
volumes. The size of each CT volume is 512 × 512 × D
where D ranges from 181 to 466. The Pancreas-CT dataset is
divided into four folds and the number corresponding to each
fold is 20, 20, 21, 21. For the stage of preprocessing, a soft
tissue [54] CT window of [-120,240] HU is utilized and the
CT scan is cropped centered on the pancreatic region while
the extensive edges of 25 voxels are magnified. We crop the
training volume to 96 × 96 × 96 randomly, with a stride of
16 × 16 × 16 at the inference stage.
TBAD Dataset. The Multi-center type B aortic dissection
(TBAD) dataset comprises 124 CTA scans [55]. Experienced
radiologists meticulously label the TBAD dataset, dividing
it into 100 training scans and 24 test scans. This dataset
includes both publicly accessible data and contributions from
UPCoL. It standardizes to a resolution of 1mm3 and resizes
to dimensions of 128×128×128. In each dataset, a maximum
of 20% of the labeled training data is used, with voxel
intensities normalized to achieve zero mean and unit variance.
ACDC Dataset. The Automated Cardiac Diagnosis Challenge
dataset [56] encompasses 200 scans of 100 patients, which
belong to four classes: background, right ventricle, left
ventricle, and myocardium. We select 100 scans [57] and split
them into 70% for training, 10% for validation, and 20% for
testing for experiments in Table II. A 2D U-Net serves as the
backbone, accepting inputs of size 256 × 256. The zero-value
region of the mask M has a dimension of 170 × 170.
BraTs2018 dataset. The BraTs2018 brain tumor dataset
comprises 285 patient cases for training and 66 for validation.
Each patient has four types of brain MRI volumes: Flair, T1,
T2, and T1ce, along with the tumor segmentation ground
truth. Each volume is sized at 155 × 240 × 240 voxels. To
facilitate the training process, we first sliced the 3D volumes
into 2D images and normalized each modality for the brain
region using the mean and standard deviation. Subsequently,
the images were center-cropped to 160 × 160 pixels.
Metrics. There are four metrics which are used for evaluating
our
proposed
method:
edge-sensitive
indicators:
95%
Hausdorff Distance (95HD) and Average Surface Distance
(ASD); regional-sensitive metrics: Dice similarity coefficient
(Dice) and Jaccard similarity coefficient (Jaccard).
B. Comparison on the four benchmark dataset
We compare the performance of the proposed model with
previous SOTA methods including UA-MT [40], SASSNet
[47], DTC [54], MC-Net [58], SS-Net [59], Co-BioNet [60],
BCP [57] and A& D [61] on Pancreas-CT, LA and ACDC
datasets. Besides, the performance of our model is also
compared with methods like MT, UA-MT, FUSSNet [41],
URPC [42] and UPCoL [43] on TBAD dataset. This paper
presents experiments conducted on five datasets and the results
are provided in Table I and II. For the TBAD dataset, the
proposed method achieves superior performance using only
5% of labeled data compared to other methods using 20% of
labeled data. With 10% and 20% of labeled data, the proposed
approach demonstrates significant performance improvements.
For the remaining datasets, the proposed method still outper-
forms others in most cases demonstrating its effectiveness.
TABLE IV: The comparison of different methods on ACDC
dataset on different semi-supervised labeled data ratio settings.
Method
10%
20%
DSC ↑
95HD ↓
ASD ↓
DSC ↑
95HD ↓
ASD ↓
ICT [62]
83.54
8.42
2.46
86.52
5.65
1.90
CPS [49]
84.70
8.25
2.35
87.47
5.98
1.24
URPC [42]
82.07
5.62
1.88
85.87
5.71
1.75
EVIL [63]
85.91
3.91
1.36
88.22
4.01
1.21
Ours
90.40
3.38
0.83
90.95
3.84
0.85
TABLE V: Comparison with the state-of-the art methods (Dice
score) on the BraTS 2018 validation dataset.
Methods
Dice
Dice
WT
TC
ET
mean
ELUnet [64]
86.16
90.27
85.15
87.19
No-new-Net [65]
90.62
84.54
80.12
85.09
DMFNet [66]
91.26
86.34
80.87
86.15
NVDLMED [67]
90.68
86.02
81.73
86.14
C-VNet [68]
90.48
83.64
77.68
83.93
Ours
90.78
90.59
86.24
89.20
C. Ablation Studies
In this section, we will explore the contributions of various
components to the overall performance of the proposed method
and regard MCF [34] as baseline. CAEF (U) denotes utilizing


**[Table p6.1]**
| Method | Scans L/U | RV | Myo |
| --- | --- | --- | --- |
|  |  | DSC ↑ JAC ↑ ASD ↓ 95HD ↓ | DSC ↑ JAC ↑ ASD ↓ 95HD ↓ |
| CPS [49] ASE-Net [50] CoraNet [51] MCF [34] ETC-Net [48] Ours | 14/126 14/126 14/126 14/126 14/126 14/126 | 83.51 71.30 0.46 4.20 83.05 72.72 0.46 4.20 83.05 72.72 0.46 4.20 83.76 74.16 1.53 2.74 86.52 77.28 1.48 5.52 89.01 80.80 7.28 1.70 | 85.91 74.64 1.17 5.34 85.61 75.18 1.17 5.34 85.61 75.18 1.17 5.34 85.40 75.18 1.08 6.35 85.66 75.26 0.82 3.50 86.43 76.40 2.87 0.98 |
| Method | Scans L/U | LV | Avg |
|  |  | DSC ↑ JAC ↑ ASD ↓ 95HD ↓ | DSC ↑ JAC ↑ ASD ↓ 95HD ↓ |
| CPS [49] ASE-Net [50] CoraNet [51] MCF [34] ETC-Net [48] Ours | 14/126 14/126 14/126 14/126 14/126 14/126 | 89.86 80.28 2.14 6.42 89.86 80.28 2.14 6.42 89.86 80.28 2.14 6.42 89.15 81.48 2.35 7.08 92.07 85.68 1.43 4.25 93.34 87.81 4.25 1.23 | 86.43 75.40 1.26 5.32 86.18 76.06 1.26 5.32 86.18 76.06 1.26 5.32 86.77 76.27 1.62 5.39 88.08 79.40 1.24 4.42 90.40 82.94 3.38 0.83 |


**[Table p6.2]**
| Method | 10% | 20% |
| --- | --- | --- |
|  | DSC ↑ 95HD ↓ ASD ↓ | DSC ↑ 95HD ↓ ASD ↓ |
| ICT [62] CPS [49] URPC [42] EVIL [63] Ours | 83.54 8.42 2.46 84.70 8.25 2.35 82.07 5.62 1.88 85.91 3.91 1.36 90.40 3.38 0.83 | 86.52 5.65 1.90 87.47 5.98 1.24 85.87 5.71 1.75 88.22 4.01 1.21 90.95 3.84 0.85 |


<!-- page 7 -->
TABLE VI: Component Ablation Study of the MEDL.
Method
Dice↑
Jaccard↑
95HD↓
ASD↓
Baseline
87.06
77.83
2.67
7.81
+ CAEF (U)
89.04
80.37
7.62
1.95
+ CAEF (L)
90.11
82.10
6.47
1.71
+ UMAL (U)
90.71
83.14
6.02
1.52
+ UMAL (L)
91.71
84.75
5.30
1.42
+ FIE
91.95
85.14
5.11
1.37
TABLE VII: Ablation Study of Varying λ Values.
λ1, λ2
Dice↑
Jaccard↑
95HD↓
ASD↓
(0.3, 0.7)
90.37
82.88
3.28
0.86
(0.4, 0.6)
90.31
82.85
3.75
0.79
(0.5, 0.5)
90.95
83.80
3.84
0.85
(0.6, 0.4)
90.77
83.56
3.68
0.81
(0.7, 0.3)
90.70
83.45
3.82
0.87
evidential fusion to generate the pseudo-labels for unlabeled
data. CAEF (L) denotes the fusion of evidential predictions
from different networks for labeled data. FIE denotes fisher
evidential deep learning. Besides, UMAL (U) and UMAL (L)
denote the uncertainty measure-based asymptotic learning for
unlabeled data and labeled data, respectively. To be specific,
each proposed component will sequentially be added to the
baseline. Table VI. provides the experimental results. Accord-
ing to the table, it can be observed that both EF (U) and EF
(L) bring remarkable performance improvement (an increase
of 3.05%, 4.27% and 6.1 on metrics of Dice, Jaccard and
ASD). Subsequently, when the components of UMAL (U)
and UMAL (L) are integrated into the baseline method, the
performance also appears great improvements (The Dice score
increases to 91.71 and the Jaccard score to 84.75, with 95HD
and ASD reducing to 5.30 and 1.42). It can be proved that the
uncertainty measure-based asymptotic learning is capable of
boosting performance. Finally, when the fisher evidential deep
learning is integrated into baseline, the model yields the best
performance. Additionally, the impact of different λ values on
model’s performance is also explored. Table VII. demonstrates
the experimental results. According to the table, based on the
Dice and Jaccard indices, λ1, λ2 = (0.5, 0.5) appears to be the
optimal value, providing the best overlap measures. However,
there is a clear trade-off between overlap measures (Dice and
Jaccard) and boundary accuracy measures (95HD and ASD).
D. Differences between the MEDL and EDL-based solutions
Several studies have applied EDL to semi-supervised medi-
cal image segmentation. Nevertheless, our work fundamentally
EF
CAEF
75
80
85
90
95
100
Dice
79.04
79.46
83.89
84.14
91.26
91.55
90.11
90.4
Performance Metrics Comparison
TBAD
Pancreas-CT
LA
ACDC
Fig. 4: 10% labeled ratio performance comparison on the
usage of EF and CAEF.
differs from ETC-Net [48], EVIL [63] (EL), and ELUnet
[64] in key aspects (perforamnce comparison is provided in
Table III, IV and V): EL adopts the original EDL loss and
probability definitions, substituting these into the dice loss,
whereas we propose using belief entropy to re-evaluate voxel-
level evidence, tightly integrating EDL loss with uncertainty
to reduce feature confusion. While ETC-Net and ELUnet
use evidential fusion (EF) for generating pseudo labels and
combining predictions, we utilize an improved class-aware ev-
idential fusion (CAEF) formula to integrate pseudo labels from
different network architectures as also evidenced by recent
work [69], and combine predictions from both networks on
labeled data enhancing robustness and predictive performance.
E. Ablation study of EF and CAEF
Furthermore, we use a class-aware evidential fusion strategy
that adjusts according to the number of segmentation classes
corresponding to each voxel. The performance comparison
between conventional evidential fusion (EF) and CAEF is
shown in Figure 4. It can be concluded that CAEF provides
better performance compared to EF. We argue this is because,
as the total number of classes corresponding to the voxels
increases, the interaction between confidence and uncertainty
associated with the voxels should be suppressed during the
fusion process. Without discounting this process, uncertain
information is likely to dominate, potentially leading to con-
fusion in some scenarios and performance degradation.
IV. CONCLUSION
The proposed MEDL framework effectively enables mod-
els to assess the uncertainty of both labeled and unlabeled
data. Through class-aware evidential fusion, we achieve the
synthesis of pseudo-labels generated by heterogeneous net-
work models while integrating their evidential predictions
for labeled data, resulting in more robust predictive results.
The uncertainty-based asymptotic learning strategy further
enhances the model’s ability to learn reliable and manage-
able features, asymptotically shifting attention to more chal-
lenging features as training progresses. Moreover, extensive
experimental results on five mainstream medical segmentation
datasets strongly validate the effectiveness of MEDL.
REFERENCES
[1] M. Yuan, Y. Xia, H. Dong, Z. Chen, J. Yao, M. Qiu, K. Yan, X. Yin,
Y. Shi, X. Chen, Z. Liu, B. Dong, J. Zhou, L. Lu, L. Zhang, and
L. Zhang, “Devil is in the queries: Advancing mask transformers for real-
world medical image segmentation and out-of-distribution localization,”
in IEEE/CVF Conference on Computer Vision and Pattern Recognition,
CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023.
[2] R. Najjar, “Redefining radiology: A review of artificial intelligence
integration in medical imaging,” Diagnostics, 2023.
[3] P. Tang, P. Yang, D. Nie, X. Wu, J. Zhou, and Y. Wang, “Unified
medical image segmentation by learning from uncertainty in an end-
to-end manner,” Knowledge-Based Systems, 2022.
[4] F. O. Catak, T. Yue, and S. Ali, “Uncertainty-aware prediction validator
in deep learning models for cyber-physical system data,” ACM Trans-
actions on Software Engineering and Methodology (TOSEM), 2022.
[5] Y. He and L. Li, “Uncertainty-aware evidential fusion-based learn-
ing for semi-supervised medical image segmentation,” arXiv preprint
arXiv:2404.06177, 2024.


**[Table p7.1]**
|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |
|  | E | F |  |  |  | C | AEF |  |

[CAPTION] Fig. 4: 10% labeled ratio performance comparison on the


<!-- page 8 -->
[6] Y. He, L. Li, T. Zhan, C.-M. Pun, W. Jiao, and Z. Jin, “Co-evidential
fusion with information volume for semi-supervised medical image
segmentation,” Pattern Recognition, vol. 166, p. 111639, 2025.
[7] Y. He, “Epl: Evidential prototype learning for semi-supervised medical
image segmentation,” arXiv preprint arXiv:2404.06181, 2024.
[8] K. Wang, B. Zhan, C. Zu, X. Wu, J. Zhou, L. Zhou, and Y. Wang,
“Semi-supervised medical image segmentation via a tripled-uncertainty
guided mean teacher model with contrastive learning,” Medical Image
Analysis, 2022.
[9] Y. Bi, E. Che, Y. Chen, Y. He, and J. Qu, “Multi-prototype-based
embedding refinement for medical image segmentation,” in ICASSP
2025-2025 IEEE International Conference on Acoustics, Speech and
Signal Processing (ICASSP).
IEEE, 2025, pp. 1–5.
[10] L. Li, Y. He, and C.-M. Pun, “Efficient prototype consistency learning
in semi-supervised medical image segmentation via joint uncertainty
and data augmentation,” in 2024 IEEE International Conference on
Bioinformatics and Biomedicine (BIBM).
IEEE, 2024, pp. 2114–2121.
[11] C.-j. Huang, Y. He, X. Han, W. Jiao, Z. Jin, and L. Wang, “Unitrans: A
unified vertical federated knowledge transfer framework for enhancing
cross-hospital collaboration,” arXiv preprint arXiv:2501.11388, 2025.
[12] Y. He, Y. Bi, L. Li, C.-M. Pun, W. Jiao, and Z. Jin, “Mutual evidential
deep learning for semi-supervised medical image segmentation,” in
2024 IEEE International Conference on Bioinformatics and Biomedicine
(BIBM).
IEEE, 2024, pp. 2010–2017.
[13] Y. He and Y. Deng, “Mmget: a markov model for generalized evidence
theory,” Computational and Applied Mathematics, vol. 41, pp. 1–41,
2022.
[14] Y. He and Y. Deng, “Ordinal belief entropy,” Soft Computing, vol. 27,
no. 11, pp. 6973–6981, 2023.
[15] Y. He and F. Xiao, “Conflicting management of evidence combination
from the point of improvement of basic probability assignment,” Inter-
national Journal of Intelligent Systems, vol. 36, no. 5, pp. 1914–1942,
2021.
[16] Y. He and F. Xiao, “A new base function in basic probability assignment
for conflict management,” Applied Intelligence, vol. 52, no. 4, pp. 4473–
4487, 2022.
[17] Y. He and Y. Deng, “Ordinal fuzzy entropy,” Iranian Journal of Fuzzy
Systems, vol. 19, no. 3, pp. 171–186, 2022.
[18] Y. He and Y. Deng, “Tdqmf: Two-dimensional quantum mass function,”
Information Sciences, vol. 621, pp. 749–765, 2023.
[19] T. Zhan, Y. He, Y. Deng, Z. Li, W. Du, and Q. Wen, “Time evidence
fusion network: Multi-source view in long-term time series forecasting,”
arXiv preprint arXiv:2405.06419, 2024.
[20] Y. He, L. Li, and T. Zhan, “A matrix-based distance of pythagorean
fuzzy set and its application in medical diagnosis,” arXiv preprint
arXiv:2102.01538, 2021.
[21] L. Li, Y. He, and C.-M. Pun, “An adaptive framework for multi-
view clustering leveraging conditional entropy optimization,” in ICASSP
2025-2025 IEEE International Conference on Acoustics, Speech and
Signal Processing (ICASSP), 2025.
[22] Y. He, L. Li, T. Zhan, W. Jiao, and C.-M. Pun, “Generalized uncertainty-
based evidential fusion with hybrid multi-head attention for weak-
supervised temporal action localization,” in ICASSP 2024-2024 IEEE
International Conference on Acoustics, Speech and Signal Processing
(ICASSP).
IEEE, 2024, pp. 3855–3859.
[23] L. Li, Y. He, and L. Li, “Nndf: A new neural detection network
for aspect-category sentiment analysis,” in International Conference
on Knowledge Science, Engineering and Management.
Springer
International Publishing Cham, 2022, pp. 339–355.
[24] T. Zhan, Y. He, Y. Deng, and Z. Li, “Differential convolutional fuzzy
time series forecasting,” IEEE Transactions on Fuzzy Systems, vol. 32,
no. 3, pp. 831–845, 2023.
[25] Y. He, W. Song, L. Li, T. Zhan, and W. Jiao, “Residual feature-
reutilization inception network,” Pattern Recognition, p. 110439, 2024.
[26] X. Chen, Z. Tao, K. Zhang, C. Zhou, W. Gu, Y. He, M. Zhang, X. Cai,
H. Zhao, and Z. Jin, “Revisit self-debugging with self-generated tests
for code generation,” arXiv preprint arXiv:2501.12793, 2025.
[27] T. Xu, K. Yan, Y. He, S. Gao, K. Yang, J. Wang, J. Liu, and Z. Liu,
“Spatio-temporal variability analysis of vegetation dynamics in china
from 2000 to 2022 based on leaf area index: A multi-temporal image
classification perspective,” Remote Sensing, vol. 15, no. 12, p. 2975,
2023.
[28] A. Jøsang, Subjective Logic - A Formalism for Reasoning Under
Uncertainty, 2016.
[29] M. Sensoy, L. M. Kaplan, and M. Kandemir, “Evidential deep learning
to quantify classification uncertainty,” in Advances in Neural Information
Processing Systems 31: Annual Conference on Neural Information Pro-
cessing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montr´eal,
Canada, 2018.
[30] A. P. Dempster, “Upper and lower probabilities induced by a multivalued
mapping,” in Classic Works of the Dempster-Shafer Theory of Belief
Functions, 2008.
[31] Z. Han, C. Zhang, H. Fu, and J. T. Zhou, “Trusted multi-view classifica-
tion with dynamic evidential fusion,” IEEE Trans. Pattern Anal. Mach.
Intell., 2023.
[32] Z. Han, C. Zhang, H. Fu, and J. T. Zhou, “Trusted multi-view classi-
fication,” in 9th International Conference on Learning Representations,
ICLR 2021, Virtual Event, Austria, May 3-7, 2021, 2021.
[33] A. Tarvainen and H. Valpola, “Mean teachers are better role mod-
els: Weight-averaged consistency targets improve semi-supervised deep
learning results,” Advances in neural information processing systems,
2017.
[34] Y. Wang, B. Xiao, X. Bi, W. Li, and X. Gao, “Mcf: Mutual correction
framework for semi-supervised medical image segmentation,” in Pro-
ceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2023.
[35] X. Zhang, Z. Peng, P. Zhu, T. Zhang, C. Li, H. Zhou, and L. Jiao,
“Adaptive affinity loss and erroneous pseudo-label refinement for weakly
supervised semantic segmentation,” in Proceedings of the 29th ACM
international conference on multimedia, 2021.
[36] Y. Wang, J. Peng, and Z. Zhang, “Uncertainty-aware pseudo label
refinery for domain adaptive semantic segmentation,” in Proceedings
of the IEEE/CVF international conference on computer vision, 2021.
[37] C. E. Shannon, “A mathematical theory of communication,” Bell Syst.
Tech. J., 1948.
[38] W. Lu, J. Lei, P. Qiu, R. Sheng, J. Zhou, X. Lu, and Y. Yang,
“Upcol: Uncertainty-informed prototype consistency learning for semi-
supervised medical image segmentation,” in International Conference
on Medical Image Computing and Computer-Assisted Intervention.
Springer, 2023.
[39] A. Tarvainen and H. Valpola, “Mean teachers are better role mod-
els: Weight-averaged consistency targets improve semi-supervised deep
learning results,” in 5th International Conference on Learning Repre-
sentations, ICLR 2017, Toulon, France, April 24-26, 2017, Workshop
Track Proceedings, 2017.
[40] L. Yu, S. Wang, X. Li, C. Fu, and P. Heng, “Uncertainty-aware self-
ensembling model for semi-supervised 3d left atrium segmentation,” in
Medical Image Computing and Computer Assisted Intervention - MIC-
CAI 2019 - 22nd International Conference, Shenzhen, China, October
13-17, 2019, Proceedings, Part II, 2019.
[41] J. Xiang, P. Qiu, and Y. Yang, “Fussnet: Fusing two sources of un-
certainty for semi-supervised medical image segmentation,” in Medical
Image Computing and Computer Assisted Intervention - MICCAI 2022
- 25th International Conference, Singapore, September 18-22, 2022,
Proceedings, Part VIII, 2022.
[42] X. Luo, G. Wang, W. Liao, J. Chen, T. Song, Y. Chen, S. Zhang, D. N.
Metaxas, and S. Zhang, “Semi-supervised medical image segmentation
via uncertainty rectified pyramid consistency,” Medical Image Anal.,
2022.
[43] W. Lu, J. Lei, P. Qiu, R. Sheng, J. Zhou, X. Lu, and Y. Yang,
“Upcol: Uncertainty-informed prototype consistency learning for semi-
supervised medical image segmentation,” in Medical Image Computing
and Computer Assisted Intervention - MICCAI 2023 - 26th International
Conference, Vancouver, BC, Canada, October 8-12, 2023, Proceedings,
Part IV, 2023.
[44] Y. Bengio, J. Louradour, R. Collobert, and J. Weston, “Curriculum
learning,” in Proceedings of the 26th annual international conference
on machine learning, 2009.
[45] D. Deng, G. Chen, Y. Yu, F. Liu, and P. Heng, “Uncertainty estimation
by fisher information-based evidential deep learning,” in International
Conference on Machine Learning, ICML 2023, 23-29 July 2023, Hon-
olulu, Hawaii, USA.
[46] Y. Yu, D. Deng, F. Liu, Q. Dou, Y. Jin, G. Chen, and P. Heng, “ANEDL:
adaptive negative evidential deep learning for open-set semi-supervised
learning,” in Thirty-Eighth AAAI Conference on Artificial Intelligence,
AAAI 2024, Thirty-Sixth Conference on Innovative Applications of
Artificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational


<!-- page 9 -->
Advances in Artificial Intelligence, EAAI 2014, February 20-27, 2024,
Vancouver, Canada, 2024.
[47] S. Li, C. Zhang, and X. He, “Shape-aware semi-supervised 3d semantic
segmentation for medical images,” in Medical Image Computing and
Computer Assisted Intervention - MICCAI 2020 - 23rd International
Conference, Lima, Peru, October 4-8, 2020, Proceedings, Part I, 2020.
[48] Z. Zhang, H. Zhou, X. Shi, R. Ran, C. Tian, and F. Zhou, “An evidential-
enhanced tri-branch consistency learning method for semi-supervised
medical image segmentation,” CoRR, 2024.
[49] X. Chen, Y. Yuan, G. Zeng, and J. Wang, “Semi-supervised semantic
segmentation with cross pseudo supervision,” in IEEE Conference on
Computer Vision and Pattern Recognition, CVPR 2021, virtual, June
19-25, 2021.
[50] T. Lei, D. Zhang, X. Du, X. Wang, Y. Wan, and A. K. Nandi, “Semi-
supervised medical image segmentation using adversarial consistency
learning and dynamic convolution network,” IEEE Trans. Medical
Imaging, 2023.
[51] Y. Shi, J. Zhang, T. Ling, J. Lu, Y. Zheng, Q. Yu, L. Qi, and
Y. Gao, “Inconsistency-aware uncertainty estimation for semi-supervised
medical image segmentation,” IEEE transactions on medical imaging,
2022.
[52] Z. Xiong, Q. Xia, Z. Hu, N. Huang, C. Bian, Y. Zheng, S. Vesal,
N. Ravikumar, A. K. Maier, X. Yang, P. Heng, D. Ni, C. Li, Q. Tong,
W. Si, ´E. Puybareau, Y. Khoudli, T. G´eraud, and J. Zhao, “A global
benchmark of algorithms for segmenting the left atrium from late
gadolinium-enhanced cardiac magnetic resonance imaging,” Medical
Image Anal., 2021.
[53] H. R. Roth, L. Lu, A. Farag, H. Shin, J. Liu, E. B. Turkbey, and
R. M. Summers, “Deeporgan: Multi-level deep convolutional networks
for automated pancreas segmentation,” in Medical Image Computing
and Computer-Assisted Intervention - MICCAI 2015 - 18th International
Conference Munich, Germany, October 5-9, 2015, Proceedings, Part I,
2015.
[54] X. Luo, J. Chen, T. Song, and G. Wang, “Semi-supervised medical
image segmentation through dual-task consistency,” in Thirty-Fifth AAAI
Conference on Artificial Intelligence, AAAI 2021, Thirty-Third Confer-
ence on Innovative Applications of Artificial Intelligence, IAAI 2021, The
Eleventh Symposium on Educational Advances in Artificial Intelligence,
EAAI 2021, Virtual Event, February 2-9, 2021, 2021.
[55] Z. Yao, W. Xie, J. Zhang, Y. Dong, H. Qiu, H. Yuan, Q. Jia, T. Wang,
Y. Shi, J. Zhuang et al., “Imagetbad: A 3d computed tomography
angiography image dataset for automatic segmentation of type-b aortic
dissection,” Frontiers in Physiology, 2021.
[56] O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, X. Yang, P.-A.
Heng, I. Cetin, K. Lekadir, O. Camara, M. A. G. Ballester et al.,
“Deep learning techniques for automatic mri cardiac multi-structures
segmentation and diagnosis: is the problem solved?” IEEE transactions
on medical imaging, 2018.
[57] Y. Bai, D. Chen, Q. Li, W. Shen, and Y. Wang, “Bidirectional copy-
paste for semi-supervised medical image segmentation,” in IEEE/CVF
Conference on Computer Vision and Pattern Recognition, CVPR 2023,
Vancouver, BC, Canada, June 17-24, 2023.
[58] Y. Wu, Z. Ge, D. Zhang, M. Xu, L. Zhang, Y. Xia, and J. Cai, “Mutual
consistency learning for semi-supervised medical image segmentation,”
Medical Image Anal., 2022.
[59] Y. Wu, Z. Wu, Q. Wu, Z. Ge, and J. Cai, “Exploring smoothness
and class-separation for semi-supervised medical image segmentation,”
in Medical Image Computing and Computer Assisted Intervention -
MICCAI 2022 - 25th International Conference, Singapore, September
18-22, 2022, Proceedings, Part V, 2022.
[60] H. Peiris, M. Hayat, Z. Chen, G. F. Egan, and M. Harandi, “Uncertainty-
guided dual-views for semi-supervised volumetric medical image seg-
mentation,” Nat. Mac. Intell., 2023.
[61] H. Wang and X. Li, “Towards generic semi-supervised framework
for volumetric medical image segmentation,” in Advances in Neural
Information Processing Systems 36: Annual Conference on Neural
Information Processing Systems 2023, NeurIPS 2023, New Orleans,
LA,
USA,
December
10
-
16,
2023,
A.
Oh,
T.
Naumann,
A. Globerson, K. Saenko, M. Hardt, and S. Levine, Eds., 2023.
[Online].
Available:
http://papers.nips.cc/paper files/paper/2023/hash/
05dc08730e32441edff52b0fa6caab5f-Abstract-Conference.html
[62] V. Verma, A. Lamb, J. Kannala, Y. Bengio, and D. Lopez-Paz, “Interpo-
lation consistency training for semi-supervised learning,” in Proceedings
of the Twenty-Eighth International Joint Conference on Artificial Intel-
ligence, IJCAI 2019, Macao, China, August 10-16, 2019, 2019.
[63] Y. Chen, Z. Yang, C. Shen, Z. Wang, Y. Qin, and Y. Zhang, “EVIL: evi-
dential inference learning for trustworthy semi-supervised medical image
segmentation,” in 20th IEEE International Symposium on Biomedical
Imaging, ISBI 2023, Cartagena, Colombia, April 18-21, 2023.
[64] L. Huang, S. Ruan, and T. Denoeux, “Belief function-based semi-
supervised learning for brain tumor segmentation,” in 18th IEEE Inter-
national Symposium on Biomedical Imaging, ISBI 2021, Nice, France,
April 13-16, 2021.
[65] F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-
Hein, “No new-net,” in Brainlesion: Glioma, Multiple Sclerosis, Stroke
and Traumatic Brain Injuries - 4th International Workshop, BrainLes
2018, Held in Conjunction with MICCAI 2018, Granada, Spain, Septem-
ber 16, 2018, Revised Selected Papers, Part II, 2018.
[66] C. Chen, X. Liu, M. Ding, J. Zheng, and J. Li, “3d dilated multi-fiber
network for real-time brain tumor segmentation in MRI,” in Medical
Image Computing and Computer Assisted Intervention - MICCAI 2019
- 22nd International Conference, Shenzhen, China, October 13-17, 2019,
Proceedings, Part III, 2019.
[67] A. Myronenko, “3d MRI brain tumor segmentation using autoencoder
regularization,” in Brainlesion: Glioma, Multiple Sclerosis, Stroke and
Traumatic Brain Injuries - 4th International Workshop, BrainLes 2018,
Held in Conjunction with MICCAI 2018, Granada, Spain, September
16, 2018, Revised Selected Papers, Part II, 2018.
[68] M. I. Sharif, J. P. Li, M. A. Khan, and M. A. Saleem, “Active deep
neural network features selection for segmentation and recognition of
brain tumors using MRI images,” Pattern Recognit. Lett., 2020.
[69] K. Wang, C. Zhang, Y. Geng, and H. Ma, “Evidential pseudo-label en-
semble for semi-supervised classification,” Pattern Recognition Letters,
2024.