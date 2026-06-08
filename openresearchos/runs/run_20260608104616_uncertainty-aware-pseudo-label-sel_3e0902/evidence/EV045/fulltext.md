<!-- page 1 -->
Scribble-based 3D Multiple Abdominal Organ
Segmentation via Triple-branch Multi-dilated
Network with Pixel- and Class-wise Consistency
Meng Han1, Xiangde Luo1, Wenjun Liao2, Shichuan Zhang2,
Shaoting Zhang1,3, and Guotai Wang1,3,∗
1 School of Mechanical and Electrical Engineering, University of Electronic Science
and Technology of China, Chengdu, China
2 Department of Radiation Oncology, Sichuan Cancer Hospital & Institute,
University of Electronic Science and Technology of China, Chengdu, China
3 Shanghai Artificial Intelligence Laboratory, Shanghai, China
* Corresponding author: guotai.wang@uestc.edu.cn
Abstract. Multi-organ segmentation in abdominal Computed Tomog-
raphy (CT) images is of great importance for diagnosis of abdominal
lesions and subsequent treatment planning. Though deep learning based
methods have attained high performance, they rely heavily on large-
scale pixel-level annotations that are time-consuming and labor-intensive
to obtain. Due to its low dependency on annotation, weakly supervised
segmentation has attracted great attention. However, there is still a large
performance gap between current weakly-supervised methods and fully
supervised learning, leaving room for exploration. In this work, we pro-
pose a novel 3D framework with two consistency constraints for scribble-
supervised multiple abdominal organ segmentation from CT. Specifi-
cally, we employ a Triple-branch multi-Dilated network (TDNet) with
one encoder and three decoders using different dilation rates to cap-
ture features from different receptive fields that are complementary to
each other to generate high-quality soft pseudo labels. For more stable
unsupervised learning, we use voxel-wise uncertainty to rectify the soft
pseudo labels and then supervise the outputs of each decoder. To fur-
ther regularize the network, class relationship information is exploited
by encouraging the generated class affinity matrices to be consistent
across different decoders under multi-view projection. Experiments on
the public WORD dataset show that our method outperforms five exist-
ing scribble-supervised methods.
Keywords: Weakly-supervised learning · Scribble annotation · Uncer-
tainty · Consistency.
1
Introduction
Abdominal organ segmentation from medical images is an essential work in clin-
ical diagnosis and treatment planning of abdominal lesions [17]. Recently, deep
arXiv:2309.09730v1  [cs.CV]  18 Sep 2023


<!-- page 2 -->
2
M. Han et al.
learning methods based on Convolution Neural Network (CNN) have achieved
impressive performance in medical image segmentation tasks [2, 24]. However,
their success relies heavily on large-scale high-quality pixel-level annotations
that are too expensive and time-consuming to obtain, especially for multiple
organs in 3D volumes. Weakly supervised learning with a potential to reduce
annotation costs has attracted great attention. Commonly-used weak annota-
tions include dots [6,11], scribbles [1,11,13,15], bounding boxes [5], and image-
level tags [20, 25]. Compared with the other weak annotations, scribbles can
provide more location information about the segmentation targets, especially
for objects with irregular shapes [1]. Therefore, this work focuses on exploring
high-performance models for multiple abdominal organ segmentation based on
scribble annotations.
Training CNNs for segmentation with scribble annotations has been increas-
ingly studied recently. Existing methods are mainly based on pseudo label learn-
ing [11, 15], regularized losses [10, 18, 22] and consistency learning [7, 13, 26].
Pseudo label learning methods deal with unannotated pixels by generating fake
semantic labels for learning. For example, Luo et al. [15] introduced a network
with two slightly different decoders that generate dynamically mixed pseudo la-
bels for supervision. Liang et al. [11] proposed to leverage minimum spanning
trees to generate low-level and high-level affinity matrices based on color in-
formation and semantic features to refine the pseudo labels. Arguing that the
pseudo label learning may be unreliable, Tang et al. [22] introduced the Condi-
tional Random Field (CRF) regularization loss for image segmentation directly.
Obukhov et al. [18] proposed to incorporate the gating function with CRF loss
considering the directionality of unsupervised information propagation. Recently,
consistency strategies that encourage consistent outputs of the network for the
same input under different perturbations have achieved increasing attentions.
Liu et al. [13] introduced transformation-consistency based on an uncertainty-
aware mean teacher [4] model. Zhang et al. [26] proposed a framework composed
of mix augmentation and cycle consistency. Although these scribble-supervised
methods have achieved promising results, their performance is still much lower
than that of fully-supervised training, leaving room for improvement.
Differently from most existing weakly supervised methods that are designed
for 2D slice segmentation with a single or few organs, we propose a highly opti-
mized 3D triple-branch network with one encoder and three different decoders,
named TDNet, to learn from scribble annotations for segmentation of multiple
abdominal organs. Particularly, the decoders are assigned with different dilation
rates [25] to learn features from different receptive fields that are complementary
to each other for segmentation, which also improves the robustness of dealing
with organs at different scales as well as the feature learning ability of the shared
encoder. Considering the features at different scales learned in these decoders,
we fuse these multi-dilated predictions to obtain more accurate soft pseudo la-
bels rather than hard labels [15] that tend to be over-confidence predictions. For
more stable unsupervised learning, we use voxel-wise uncertainty to rectify the
soft pseudo labels and then impose consistency constraints on the output of each


<!-- page 3 -->
Scribble-based 3D Multiple Abdominal Organ Segmentation
3
ℒ𝑈𝑈𝑈𝑈𝑈𝑈𝑈𝑈
ℒ𝑝𝑝𝑝𝑝𝑝𝑝
uncertainty 
rectified
ℒ𝑈𝑈𝑈𝑈𝑈𝑈𝑈𝑈
ℒ𝑈𝑈𝑈𝑈𝑈𝑈𝑈𝑈
Image 𝑋𝑋
Scribble 𝑆𝑆
𝜉𝜉1
𝜉𝜉2
Encoder 𝜃𝜃𝑒𝑒
Decoder 𝜃𝜃𝑑𝑑𝑑, 𝑑𝑑𝑑𝑑= 1
Decoder 𝜃𝜃𝑑𝑑𝑑, 𝑑𝑑𝑑𝑑= 6
Decoder 𝜃𝜃𝑑𝑑𝑑, 𝑑𝑑𝑑𝑑= 3
𝑃𝑃1
𝑃𝑃2
𝑃𝑃3
ത𝑃𝑃
𝑄𝑄2
𝑄𝑄1
𝑄𝑄3
ℒ𝑀𝑀𝑀𝑀𝑀𝑀𝑀𝑀
ℒ𝑀𝑀𝑀𝑀𝑀𝑀𝑀𝑀
ℒ𝑀𝑀𝑀𝑀𝑀𝑀𝑀𝑀
ത𝑄𝑄
×
𝑃𝑃𝑛𝑛
𝑄𝑄𝑛𝑛
∗× 𝐶𝐶
𝐶𝐶× 𝐶𝐶
𝐶𝐶× (∗)
𝑇𝑇
Project in sagittal view
𝑆𝑆
Reshape matrix
×
Multiply operation
𝑇𝑇
Transpose matrix 
(b) Class Affinity Calculation
(a) The Proposed TDNet
Project in coronal view
Project in axial view
𝑆𝑆
class affinity calculation
Fig. 1. Overview of the proposed Triple-branch multi-Dilated Network (TDNet) that
uses different dilation rates at three decoders. The TDNet is optimized by Uncertainty-
weighted Soft Pseudo label Consistency (USPC) using the mixed soft pseudo labels and
Multi-view Projection-based Class-similarity Consistency (MPCC). The class affinity
calculation process is shown in (b). Best viewed in color.
branch. In addition, we extend the consistency to the class-related information
level [23] to constrain inter-class affinity for better distinguishing them. Specifi-
cally, we generate the class affinity matrices in different decoders and encourage
them to be consistent after projection in different views.
The contributions of this paper are summarized as follows: 1) We propose a
novel 3D Triple-branch multi-Dilated network called TDNet for scribble-supervised
segmentation. By equipping with varying dilation rates, the network can better
leverage multi-scale context for dealing with organs at different scales. 2) We pro-
pose two novel consistency loss functions, i.e., Uncertainty-weighted Soft Pseudo
label Consistency (USPC) loss and Multi-view Projection-based Class-similarity
Consistency (MPCC) loss, to regularize the prediction from the pixel-wise and
class-wise perspectives respectively, which helps the segmentation network ob-
tain reliable predictions on unannotated pixels. 3) Experiments results show our
proposed method outperforms five existing scribble-supervised methods on the
public dataset WORD [17] for multiple abdominal organ segmentation.
2
Method
Fig. 1 shows the the proposed framework for scribble-supervised medical image
segmentation. We introduce a network with one encoder and three decoders with


**[Table p3.1]**
|  | ℒ𝑈𝑈𝑈 u ℒ𝑈𝑈𝑈 |
| --- | --- |
| 𝜉1𝜉 Decoder 𝜃𝑑𝜃𝑑, 𝑑𝑑 = 3 𝑃2𝑃 𝜉2𝜉 |  |

[CAPTION] Fig. 1. Overview of the proposed Triple-branch multi-Dilated Network (TDNet) that

[CAPTION] Fig. 1 shows the the proposed framework for scribble-supervised medical image


<!-- page 4 -->
4
M. Han et al.
different dilation rates to learn multi-scale features. The decoders’ outputs are
averaged to generate a soft pseudo label that is rectified by uncertainty and then
used to supervise each branch. To better deal with multi-class segmentation, a
class similarity consistency loss is also used for regularization.
For the convenience of following description, we first define several mathe-
matical symbols. Let X, S be a training image and the corresponding scribble
annotation, respectively. Let C denote the number of classes for segmentation,
and Ω= ΩS ∪ΩU denote the whole set of voxels in X, where ΩS is the set of
labeled pixels annotated in S, and ΩU is the unlabeled pixel set.
2.1
Triple-branch Multi-dilated Network (TDNet)
As shown in Fig. 1(a), the proposed TDNet consists of a shared encoder (θe) and
three independent decoders (θd1, θd2, θd3) with different dilation rates to mine
unsupervised context from different receptive fields. Specifically, decoders using
convolution with small dilation rates can extract detailed local features but their
receptive fields are small for understanding a global context. Decoders using con-
volution with large dilation rates can better leverage the global information but
may lose some details for accurate segmentation. In this work, our TDNet is
implemented by introducing two auxiliary decoders into a 3D UNet [3]. The di-
lation rate in the primary decoder and the two auxiliary decoders are 1, 3 and
6 respectively, with the other structure parameters (e.g., kernel size, channel
number etc.) being the same in the three decoders. To further introduce per-
turbations for obtaining diverse outputs, the three branches are initialized with
Kaiming initialization, Xavier and Normal initialization methods, respectively.
In addition, the bottleneck’s output features are randomly dropped out before
sending into the auxiliary decoders. The probability prediction maps obtained
by the three decoders are denoted as P1, P2 and P3, respectively.
2.2
Pixel-wise and Class-wise Consistency
Uncertainty-weighted Soft Pseudo label Consistency (USPC) As the
three decoders capture features at different scales that are complementary to
each other, an ensemble of them would be more robust than a single branch.
Therefore, we take an average of P1, P2, P3 to get a better soft pseudo label
¯P = (P1 + P2 + P3)/3 that is used to supervise each branch during training.
However, ¯P may also contain noises and be inaccurate, and it is important
to highlight reliable pseudo labels while suppressing unreliable ones. Thus, we
propose a regularization term named Uncertainty-weighted Soft Pseudo label
Consistency (USPC) between Pn (n= 1, 2, 3) and ¯P:
LUSP C = 1
3
X
n=1,2,3
P
i wiKL(Pn,i∥¯Pi)
P
i wi
(1)
where ¯Pi refers to the prediction probability at voxel i in ¯P, and ¯Pn,i is the cor-
responding prediction probability at voxel i in ¯Pn. KL() is the Kullback–Leibler


<!-- page 5 -->
Scribble-based 3D Multiple Abdominal Organ Segmentation
5
divergence. wi is the voxel-wise weight based on uncertainty estimation:
wi = e
P
c ¯
P c
i log( ¯
P c
i )
(2)
where the uncertainty is estimated by entropy. c is the class index, and ¯P c
i means
the probability for class c at voxel i in the pseudo label. Note that a higher
uncertainty leads to a lower weight. With the uncertainty-based weighting, the
model will be less affected by unreliable pseudo labels.
Multi-view Projection-based Class-similarity Consistency (MPCC) For
multi-class segmentation tasks, it is important to learn inter-class relationship
for better distinguishing them. In addition to using LUSP C for pixel-wise super-
vision, we consider making consistency on class relationship across the outputs
of the decoders as illustrated in Fig. 1. In order to save computing resources, we
project the soft pseudo labels along each dimension and then calculate the affin-
ity matrices, which also strengthens the class relationship information learning.
We first project the soft prediction map of the n-th decoder Pn ∈RC×D×H×W
in axial view to a tensor with the shape of C × 1 × H × W. It is reshaped
into C × (WH) and multiplied by its transposed version, leading to a class
affinity matrix Q′axial
n
∈RC×C. A normalized version of Q′axial
n
is denoted as
Qaxial
n
= Q′axial
n
/||Q′axial
n
||. Similarly, Pn is projected in the sagittal and coro-
nal views, respectively, and the corresponding normalized class affinity matrices
are denoted as Qsagittal
n
and Qcoronal
n
, respectively. Here, the affinity matrices
represents the relationship between any pair of classes along the dimensions.
Then we constraint the consistency among the corresponding affinity matrices
by Multi-view Projection-based Class-similarity Consistency (MPCC) loss:
LMP CC =
1
3 × 3
X
v
X
n=1,2,3
KL(Qv
n∥¯Qv)
(3)
where v ∈{axial, sagittal, coronal} is the view index, and ¯Qv is the average class
affinity matrix in a certain view obtained by the three decoders.
2.3
Overall Loss Function
To learn from the scribbles, the partially Cross-Entropy (pCE) loss is used to
train the network, where the labeled pixels are considered to calculate the gra-
dient and the other pixels are ignored [21]:
Lsup = −
1
3 |ΩS|
X
n=1,2,3
X
i∈ΩS
X
c
Sc
i log P c
n,i
(4)
where S represents the one-hot scribble annotation, and ΩS is the set of labeled
pixels in S. The total object function is summarized as:
Ltotal = Lsup + αtLUSP C + βtLMP CC
(5)


<!-- page 6 -->
6
M. Han et al.
where αt and βt are the weights for the unsupervised losses. Following [13],
we define αt based on a ramp-up function: αt = α · e(−5(1−t/tmax)2), where t
denotes the current training step and tmax is the maximum training step. We
define βt = β · e(−5(1−t/tmax)2) in a similar way. In this way, the model can learn
accurate information from scribble annotations, which also avoids getting stuck
in a degenerate solution due to low-quality pseudo labels at an early stage.
3
Experiments and Results
3.1
Dataset and Implementation Details
We used the publicly available abdomen CT dataset WORD [17] for experiments,
which consists of 150 abdominal CT volumes from patients with rectal cancer,
prostate cancer or cervical cancer before radiotherapy. Each CT volume contains
159-330 slices of 512×512 pixels, with an in-plane resolution of 0.976 × 0.976 mm
and slice spacing of 2.5-3.0 mm. We aimed to segment seven organs: the liver,
spleen, left kidney, right kidney, stomach, gallbladder and pancreas. Following
the default settings in [17], the dataset was split into 100 for training, 20 for
validation and 30 for testing, respectively, where the scribble annotations for
foreground organs and background in the axial view of the training volumes had
been provided and were used in model training. For pre-processing, we cut off
the Hounsfield Unit (HU) values with a fixed window/level of 400/50 to focus
on the abdominal organs, and normalized it to [0, 1]. We used the commonly-
adopted Dice Similarity Coefficient (DSC), 95% Hausdorff Distance(HD95) and
the Average Surface Distance (ASD) for quantitative evaluation.
Our framework was implemented in PyTorch [19] on an NVIDIA 2080Ti
with 11GB memory. We employed the 3D UNet [3] as the backbone network
for all experiments, and extended it with three decoders by embedding two
auxiliary decoders with different dilation rates, as detailed in Section 2.1. To
introduce perturbations, different initializations were applied to each decoder,
and random perturbations (ratio = (0, 0.5)) were introduced in the bottleneck
before the auxiliary decoders. The Stochastic Gradient Descent (SGD) optimizer
with momentum of 0.9 and weight decay of 10−4 was used to minimize the overall
loss function formulated in Eq.5, where α=10.0 and β=1.0 based on the best
performance on the validation set. The poly learning rate strategy [16] was used
to decay learning rate online. The batch size, patch size and maximum iterations
tmax were set to 1, [80, 96, 96] and 6 × 104 respectively. The final segmentation
results were obtained by using a sliding window strategy. For a fair comparison,
we used the primary decoder’s outputs as the final results during the inference
stage and did not use any post-processing methods. Note that all experiments
were conducted in the same experimental setting. The existing methods are
implemented with the help of open source codebase from [14].


<!-- page 7 -->
Scribble-based 3D Multiple Abdominal Organ Segmentation
7
Table 1. Quantitative comparison between our method and existing weakly supervised
methods on WORD testing set. ∗denotes p-value < 0.05 (paired t-test) when comparing
with the second place method [15]. The best values are highlighted in bold.
Organ
FullySup [3]
pCE
TV [9]
USTM [13]
EM [8]
DMPLS [15]
Ours
Liver
96.37±0.74
86.53±3.61 87.22±3.07 80.57±3.87 89.60±1.72
88.89±2.54
93.31±0.85
∗
spleen
95.42±1.55
86.81±5.75 82.95±7.30 86.49±4.71 87.76±5.65
89.19±3.93
91.77±3.19
∗
kidney(L)
94.95±1.58
86.25±4.46 83.78±4.60 87.15±4.21 87.29±3.88
90.14±2.98
92.34±2.30
∗
kidney(R)
95.33±1.34
89.41±2.97 89.38±3.15 89.26±2.35 81.32±4.79
90.93±2.15
92.54±1.95
∗
stomach
90.08±4.42
61.09±12.20 62.64±12.74 77.33±6.19 77.74±7.34
77.06±7.39
85.82±4.19
∗
gallbladder
75.33±13.21 56.61±20.12 44.06±19.92 63.94±17.17 65.83±16.80 70.25±15.57 69.01±16.53
pancreas
80.90±7.67
61.55±10.45 65.31±9.20 67.24±9.49 73.52±7.54
75.02±7.53
75.40±7.51
avg DSC(%)
89.77±4.36
75.46±8.51 73.62±8.57 78.85±6.86 80.44±6.82
83.07±6.01
85.74±5.22
∗
avg ASD(mm)
1.60±1.56
25.11±11.59 31.01±12.41 18.24±9.29 16.17±8.35
7.77±6.33
2.33±1.76
∗
avg HD95(mm)
5.71±5.36
77.56±36.80 98.61±39.93 61.43±34.79 50.90±29.92 24.00±20.08
7.84±5.84
∗
Ground Truth
FS[3]
pCE
USTM[13]
DMPLS[15]
TV[9]
EM[8]
Ours
Background
Liver
Gallbladder
Spleen
Kidney (L)
Kidney (R)
Stomach
Pancreas
Images
Fig. 2. Visual comparison between our method and other weakly supervised methods.
Best view in color.
3.2
Comparison with other methods
We compared our method with five weakly supervised segmentation methods
with the same set of scribbles, including pCE only [12], Total Variation Loss
(TV) [9], Uncertainty-aware Self-ensembling and Transformation-consistent Model
(USTM) [13], Entropy Minimization (EM) [8] and Dynamically Mixed Pseudo
Labels Supervision (DMPLS) [15]. They were also compared with the upper
bound by using dense annotation to train models (FullySup) [3]. The results in
Table 1 show that our method leads to the best DSC, ASD and HD95. Compared
with the second best method DMPLS [15], the average DSC was increased by
2.67 percent points, and the average ASD and HD95 were decreased by 5.44 mm
and 16.16 mm, respectively. It can be observed that TV [9] obtained a worse
performance than pCE, which is mainly because that method classifies pixels
by minimizing the intra-class intensity variance, making it difficult to achieve
good segmentation due to the low contrast. Fig. 2 shows a visual comparison
between our method and the other weakly supervised methods on the WORD


**[Table p7.1]**
| Background Liver Spleen Kidney (R) Kidney (L) Stomach Gallbladder Pancreas Images FS[3] pCE TV[9] USTM[13] EM[8] DMPLS[15] Ours Ground Truth |  | Background |  | Liver |  | Spleen |  | Kidney (R) |  | Kidney (L) |  | Stomach |  | Gallbladder |  | Pancreas |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

[CAPTION] Table 1. Quantitative comparison between our method and existing weakly supervised

[CAPTION] Fig. 2. Visual comparison between our method and other weakly supervised methods.

[CAPTION] Table 1 show that our method leads to the best DSC, ASD and HD95. Compared


<!-- page 8 -->
8
M. Han et al.
Table
2. Ablation study of our proposed
method on WORD validation set. N(s) and
N(d) means N decoders with the same and dif-
ferent dilation rates, respectively. Lsup is used
by default. The best values are highlighted in
bold.
Decoder
Loss
DSC(%)
ASD(mm) HD95(mm)
1(s)
74.70±8.68 25.51±10.12 79.98±30.39
3(s)
LUSP C(−ω)
81.92±8.04
9.40±6.79
31.11±20.92
3(d)
LUSP C(−ω)
82.57±7.28
3.34±2.67
9.26±7.26
3(d)
LUSP C
84.21±6.99
2.82±2.71
8.25±6.36
3(d)
LUSP C + LMP CC 84.75±7.01 2.64±2.46
7.91±5.93
2(d)
LUSP C + LMP CC 84.18±6.84
2.85±2.38
8.56±6.36
4(d)
LUSP C + LMP CC 83.51±7.01
2.88±2.49
8.58±6.53
(a)  3 𝑠𝑠+ ℒ𝑈𝑈𝑈𝑈𝑈𝑈𝑈𝑈(−𝜔𝜔)
(b)  3 𝑑𝑑+ ℒ𝑈𝑈𝑈𝑈𝑈𝑈𝑈𝑈(−𝜔𝜔)
(c)  3 𝑑𝑑+ ℒ𝑈𝑈𝑈𝑈𝑈𝑈𝑈𝑈
(d) Ground Truth
Fig. 3. Visualization of the improve-
ment obtained by using different dila-
tion rates and uncertainty rectifying.
Best viewed in color.
dataset (word_0014.nii). It can be obviously seen that the results obtained by
our method are closer to the ground truth, with less mis-segmentation in both
slice level and volume level.
3.3
Ablation experiment
We then performed ablation experiments to investigate the contribution of each
part of our method, and the quantitative results on the validation set are shown
in Table 2, where LUSP C(−ω) means using LUSP C without pixel-wise uncer-
tainty rectifying. Baseline refers to a triple-branch model with different initial-
izations and random feature-level dropout in the bottleneck, supervised by pCE
only. It can be observed that by using LUSP C(−ω) with mutiple decoders, the
model segmentation performance is greatly enhanced with average DSC increas-
ing by 7.70%, ASD and HD95 decreasing by 16.11 mm and 48.87 mm, respec-
tively. By equipping each decoders with different dilation rates, the model’s
performance is further improved, especially in terms of ASD and HD95, which
proves our hypothesis that learning features from different scales can improve
the segmentation accuracy. Replacing LUSP C(−ω) with LUSP C further improved
the DSC to 84.21%, and reduced the ASD and HD95 by 0.52 mm and 1.01 mm
through utilizing the uncertainty information. Visual comparison in Fig. 3 demon-
strates that over-segmentation can be mitigated by using different dilation rates
in the three decoders, and using the uncertainty-weighted pseudo labels can
further improve the segmentation accuracy with small false positive regions re-
moving.
Additionally, Table 2 shows that combining LUSP C and LMP CC obtained the
best performance, where the average DSC, ASD and HD95 were 84.75%, 2.64 mm
and 7.91 mm, respectively, which demonstrates the effectiveness of the proposed
class similarity consistency. In order to find the optimal number of decoders, we
set the decoder number to 2, 3 and 4 respectively. The quantitative results in the
last three rows of Table 2 show that using three decoders outperformed using
two and four decoders.

[CAPTION] Table
2. Ablation study of our proposed

[CAPTION] Fig. 3. Visualization of the improve-


<!-- page 9 -->
Scribble-based 3D Multiple Abdominal Organ Segmentation
9
4
Conclusion
In this paper, we proposed a scribble-supervised multiple abdominal organ seg-
mentation method consisting of a 3D triple-branch multi-dilated network with
two-level consistency constraints. By equipping each decoder with different di-
lation rates, the model leverages features at different scales to obtain high-
quality soft pseudo labels. In addition to mine knowledge from unannotated
pixels, we also proposed USPC Loss and MPCC Loss to learn unsupervised
information from the uncertainty-rectified soft pseudo labels and class affinity
matrix information respectively. Experiments on a public abdominal CT dataset
WORD demonstrated the effectiveness of the proposed method, which outper-
forms five existing scribble-based methods and narrows the performance gap
between weakly-supervised and fully-supervised segmentation methods. In the
future, we will explore the effect of our method on sparser labels, such as a
volumetric data with scribble annotations on one or few slices.
Acknowledgements This work was supported by the National Natural Sci-
ence Foundation of China (No.62271115), Science and Technology Department
of Sichuan Province, China (2022YFSY0055) and Radiation Oncology Key Lab-
oratory of Sichuan Province Open Fund (2022ROKF04).
References
1. Chen, Q., Hong, Y.: Scribble2D5: Weakly-supervised volumetric image segmenta-
tion via scribble annotations. In: MICCAI. pp. 234–243. Springer (2022)
2. Chen, X., Sun, S., Bai, N., Han, K., Liu, Q., Yao, S., Tang, H., Zhang, C., Lu,
Z., Huang, Q., et al.: A deep learning-based auto-segmentation system for organs-
at-risk on whole-body computed tomography images for radiation therapy. Radio-
therapy and Oncology 160, 175–184 (2021)
3. Çiçek, Ö., Abdulkadir, A., Lienkamp, S.S., Brox, T., Ronneberger, O.: 3D U-Net:
learning dense volumetric segmentation from sparse annotation. In: MICCAI. pp.
424–432. Springer (2016)
4. Cui, W., Liu, Y., Li, Y., Guo, M., Li, Y., Li, X., Wang, T., Zeng, X., Ye, C.:
Semi-supervised brain lesion segmentation with an adapted mean teacher model.
In: IPMI. pp. 554–565. Springer (2019)
5. Dai, J., He, K., Sun, J.: Boxsup: Exploiting bounding boxes to supervise convolu-
tional networks for semantic segmentation. In: ICCV. pp. 1635–1643 (2015)
6. En, Q., Guo, Y.: Annotation by clicks: A point-supervised contrastive vari-
ance method for medical semantic segmentation. arXiv preprint arXiv:2212.08774
(2022)
7. Gao, F., Hu, M., Zhong, M.E., Feng, S., Tian, X., Meng, X., Huang, Z., Lv, M.,
Song, T., Zhang, X., et al.: Segmentation only uses sparse annotations: Unified
weakly and semi-supervised learning in medical images. Medical Image Analysis
80, 102515 (2022)
8. Grandvalet, Y., Bengio, Y.: Semi-supervised learning by entropy minimization. In:
NeurIPS. pp. 1–17 (2004)


<!-- page 10 -->
10
M. Han et al.
9. Javanmardi, M., Sajjadi, M., Liu, T., Tasdizen, T.: Unsupervised total variation
loss for semi-supervised deep learning of semantic segmentation. arXiv preprint
arXiv:1605.01368 (2016)
10. Kim, B., Ye, J.C.: Mumford–shah loss functional for image segmentation with deep
learning. IEEE Transactions on Image Processing 29, 1856–1866 (2019)
11. Liang, Z., Wang, T., Zhang, X., Sun, J., Shen, J.: Tree energy loss: Towards sparsely
annotated semantic segmentation. In: CVPR. pp. 16907–16916 (2022)
12. Lin, D., Dai, J., Jia, J., He, K., Sun, J.: Scribblesup: Scribble-supervised convolu-
tional networks for semantic segmentation. In: CVPR. pp. 3159–3167 (2016)
13. Liu, X., Yuan, Q., Gao, Y., He, K., Wang, S., Tang, X., Tang, J., Shen, D.: Weakly
supervised segmentation of COVID19 infection with scribble annotation on CT
images. Pattern Recognition 122, 108341 (2022)
14. Luo, X.: WSL4MIS. https://github.com/Luoxd1996/WSL4MIS (2021)
15. Luo, X., Hu, M., Liao, W., Zhai, S., Song, T., Wang, G., Zhang, S.: Scribble-
supervised medical image segmentation via dual-branch network and dynamically
mixed pseudo labels supervision. In: MICCAI. pp. 528–538. Springer (2022)
16. Luo, X., Liao, W., Chen, J., Song, T., Chen, Y., Zhang, S., Chen, N., Wang, G.,
Zhang, S.: Efficient semi-supervised gross target volume of nasopharyngeal carci-
noma segmentation via uncertainty rectified pyramid consistency. In: MICCAI. pp.
318–329. Springer (2021)
17. Luo, X., Liao, W., Xiao, J., Chen, J., Song, T., Zhang, X., Li, K., Metaxas, D.N.,
Wang, G., Zhang, S.: WORD: A large scale dataset, benchmark and clinical ap-
plicable study for abdominal organ segmentation from ct image. Medical Image
Analysis 82, 102642 (2022)
18. Obukhov, A., Georgoulis, S., Dai, D., Van Gool, L.: Gated CRF loss for weakly
supervised semantic image segmentation. arXiv preprint arXiv:1906.04651 (2019)
19. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen,
T., Lin, Z., Gimelshein, N., Antiga, L., et al.: Pytorch: An imperative style, high-
performance deep learning library. NeurIPS 32 (2019)
20. Ru, L., Zhan, Y., Yu, B., Du, B.: Learning affinity from attention: end-to-end
weakly-supervised semantic segmentation with transformers. In: CVPR. pp. 16846–
16855 (2022)
21. Tang, M., Djelouah, A., Perazzi, F., Boykov, Y., Schroers, C.: Normalized cut loss
for weakly-supervised cnn segmentation. In: CVPR. pp. 1818–1827 (2018)
22. Tang, M., Perazzi, F., Djelouah, A., Ben Ayed, I., Schroers, C., Boykov, Y.: On
regularized losses for weakly-supervised cnn segmentation. In: ECCV. pp. 507–522
(2018)
23. Tung, F., Mori, G.: Similarity-preserving knowledge distillation. In: ICCV. pp.
1365–1374 (2019)
24. Wang, Y., Zhou, Y., Shen, W., Park, S., Fishman, E.K., Yuille, A.L.: Abdomi-
nal multi-organ segmentation with organ-attention networks and statistical fusion.
Medical Image Analysis 55, 88–102 (2019)
25. Wei, Y., Xiao, H., Shi, H., Jie, Z., Feng, J., Huang, T.S.: Revisiting dilated convo-
lution: A simple approach for weakly-and semi-supervised semantic segmentation.
In: CVPR. pp. 7268–7277 (2018)
26. Zhang, K., Zhuang, X.: Cyclemix: A holistic strategy for medical image segmenta-
tion from scribble supervision. In: CVPR. pp. 11656–11665 (2022)