<!-- page 1 -->
Uncertainty-Guided Mutual Consistency Learning for Semi-Supervised Medical
Image Segmentation
Yichi Zhanga, Rushi Jiaoa, Qingcheng Liaoa, Dongyang Lia, Jicong Zhanga,b,c,∗
aSchool of Biological Science and Medical Engineering, Beihang University, Beijing, China
bHefei Innovation Research Institute, Beihang University, Hefei, China
cBeijing Advanced Innovation Centre for Biomedical Engineering, Beijing, China
Abstract
Medical image segmentation is a fundamental and critical step in many clinical approaches. Semi-supervised learn-
ing has been widely applied to medical image segmentation tasks since it alleviates the heavy burden of acquiring
expert-examined annotations and takes the advantage of unlabeled data which is much easier to acquire. Although
consistency learning has been proven to be an eﬀective approach by enforcing an invariance of predictions under
diﬀerent distributions, existing approaches cannot make full use of region-level shape constraint and boundary-level
distance information from unlabeled data. In this paper, we propose a novel uncertainty-guided mutual consistency
learning framework to eﬀectively exploit unlabeled data by integrating intra-task consistency learning from up-to-date
predictions for self-ensembling and cross-task consistency learning from task-level regularization to exploit geomet-
ric shape information. The framework is guided by the estimated segmentation uncertainty of models to select out
relatively certain predictions for consistency learning, so as to eﬀectively exploit more reliable information from un-
labeled data. Experiments on two publicly available benchmark datasets showed that: 1) Our proposed method can
achieve signiﬁcant performance improvement by leveraging unlabeled data, with up to 4.13% and 9.82% in Dice
coeﬃcient compared to supervised baseline on left atrium segmentation and brain tumor segmentation, respectively.
2) Compared with other semi-supervised segmentation methods, our proposed method achieve better segmentation
performance under the same backbone network and task settings on both datasets, demonstrating the eﬀectiveness and
robustness of our method and potential transferability for other medical image segmentation tasks.
Keywords: Medical Image Segmentation, Semi-Supervised Learning, Uncertainty Estimation, Mutual Consistency
Learning
1. Introduction
Medical imaging have been widely used in clinical researches, which improves the quality of healthcare by discov-
ering potential lesion and providing diagnostic opinions. Among the various tasks of medical image analysis, medical
image segmentation aims to understand images in pixel-level and label each pixel into a certain class, which plays an
∗Corresponding author. E-mail address: jicongzhang@buaa.edu.cn
Preprint submitted to Elsevier
August 29, 2022
arXiv:2112.02508v2  [eess.IV]  25 Aug 2022


<!-- page 2 -->
important role in diagnostic analysis, surgical planning and postoperative analysis, and has attracted the attention of
researchers [1, 2, 3]. Based on accurate and robust segmentation results, the morphological attributes of physiological
and pathological structures can be quantitatively analyzed, so as to provide useful basis for clinicians to diagnose
diseases. Recently, deep learning-based methods have shown signiﬁcant improvements and achieved state-of-the-art
performances in many medical image segmentation tasks like cardiac segmentation [4, 5, 6], abdominal segmentation
[7, 8], etc. However, the success of most existing deep learning-based methods relies on a large amount of labeled
training data to ease the sub-optimal performance caused by over-ﬁtting and ensure reliable generalization perfor-
mance on test set, while it is hard and expensive to obtain large-amount well-annotated data in the medical imaging
domain where only experts can provide reliable annotations. Besides, many commonly used medical images like
computed tomography (CT) and magnetic resonance imaging (MRI) scans are in 3D volumes, which further increase
the burden of manual annotation [9]. According to the statistics in [10], it takes about 400±45 minutes for experts to
delineate one CT scan with 250 slices for lung infection segmentation.
To ease the manual labeling burden, signiﬁcant eﬀorts have been devoted to utilize available annotations eﬃ-
ciently and improve the segmentation performance with low labeling cost [11, 12, 13]. Compared with acquiring
expert-examined annotations, unlabeled medical images are relatively easier to obtain. Therefore, implementing med-
ical image segmentation models with only a few labeled images has become an active research topic for clinical
applications [14]. In this work, we focus on semi-supervised learning for medical image segmentation by encour-
aging models to learn from a limited amount of expert-examined labeled data and a large amount of unlabeled data,
which is a fundamental, challenging problem and has a high impact on real-world clinical applications. To utilize
unlabeled data, a simple and intuitive method is to assign pseudo annotations to unlabeled data and then train the seg-
mentation model using both labeled and pseudo labeled data. However, model-generated annotations can be noisy and
have detrimental eﬀects to the subsequent segmentation model [15]. Recent impressive progress in semi-supervised
medical image segmentation has been focused on incorporating unlabeled data into the training procedure with an
unsupervised loss function. Speciﬁcally, the mean teacher (MT) model [16] has achieved great success by enforcing
the consistency of predictions from perturbed inputs between student and teacher models. Following [16], many con-
sistency learning methods [17, 18, 19] have been proposed. Besides, another line of researches [20, 21, 22] focus on
building task-level regularization by adding auxiliary task to leverage boundary-based surface mismatch. However,
these consistency-based semi-supervised medical segmentation methods cannot make full use of reliable region-level
shape constraint and boundary-level distance information from unlabeled cases.
In this paper, we propose a novel uncertainty-guided mutual consistency learning framework to eﬀectively exploit
unlabeled data for semi-supervised medical image segmentation. We use dual-task backbone network with two out-
put branches to generate segmentation probabilistic maps and signed distance maps simultaneously. For consistency
learning, our framework focuses on integrating both the intra-task consistency learning of up-to-date predictions for
self-ensembling and cross-task consistency learning from task-level regularization to exploit geometric shape infor-
mation. Besides, our proposed framework is guided by the estimated segmentation uncertainty of models to select out
2


<!-- page 3 -->
Input  Volume
Same  Architecture
EMA
Student Model  θ
Teacher Model  θ’
MC Dropout
Segmentation Branch
Regression Branch
guide
Uncertainty Map
Segmentation Branch
Regression Branch
Figure 1: The overview of our proposed uncertainty-guided mutual consistency learning framework for semi-supervised medical image segmen-
tation. The backbone network consists of two diﬀerent branches for diﬀerent tasks, where the ﬁrst branch aims for pixel-wise classiﬁcation and
generates the segmentation probabilistic maps as output, and the second branch aims at level set function regression and regresses the signed dis-
tance maps. Following the design of mean teacher framework, the student model is optimized by minimizing the supervised loss on labeled data
DL and cross-task consistency loss on both labeled data DL and unlabeled data DU. The estimated uncertainty from the teacher model is used to
guide the learning of the student model so as to learn more reliable information from unlabeled data for semi-supervised learning.
relatively certain predictions for consistency learning, so as to eﬀectively exploit more reliable information from un-
labeled data. We conduct extensive experiments on two publicly available medical image segmentation datasets: Left
Atrium Segmentation (LA) and Brain Tumor Segmentation (BraTS). Experimental results show that our framework
can largely improve the segmentation performance by leveraging the unlabeled images and outperform state-of-the-art
semi-supervised segmentation methods.
2. Related Work
2.1. CNNs for Medical Image Segmentation
Image segmentation aims to understand the image in pixel level and classify each pixel into a certain class. Re-
cently, convolutional neural networks (CNNs) have achieved state-of-the-art performance in many medical image
segmentation tasks. Fully convolutional networks (FCN) [23] is a landmark in image segmentation. It applied classic
3

[CAPTION] Figure 1: The overview of our proposed uncertainty-guided mutual consistency learning framework for semi-supervised medical image segmen-


<!-- page 4 -->
convolution neural network to dense prediction and ﬁrst realized end to end segmentation by replacing fully connected
layers with convolutional layers. Most of widely used methods for medical image segmentation are inspired by U-
Net [24] based on an encoder-decoder structure to extract features at multiple scales. The network architecture fused
the features of diﬀerent scales by concatenating the feature maps of the downsampling layers and the corresponding
upsampling layers for subsequent learning. For segmentation of medical volumes, 3D segmentation networks like 3D
U-Net [25] and V-Net [26] are proposed to use 3D convolution kernels to extract volumetric features. Besides, many
variants of U-Net have been proposed to improve it by designing novel structures and have been applied to many
medical image segmentation tasks. [27, 28, 29, 30, 31]. Isensee et al. [32] proposed nnU-Net to automatically adapt
training strategies and network architectures to a given medical dataset and achieved state-of-the-art performances on
many segmentation tasks [33], which demonstrated that the basic encoder-decoder structure is still diﬃcult to surpass
if the corresponding pipeline is designed adequately.
2.2. Semi-supervised Medical Image Segmentation
To reduce the burden of annotation cost, many semi-supervised learning medical image segmentation methods
have been proposed by using a limited number of labeled data and an arbitrary amount of unlabeled data. Existing
semi-supervised methods mainly have two categories. The ﬁrst category is based on pseudo labels, which is an in-
tuitive method by assigning pseudo annotations for unlabeled images, and then using the pseudo labeled images to
update the segmentation model. Bai et al. [34] iteratively updated the pseudo segmentation labels and network pa-
rameters and used conditional random ﬁeld (CRF) to reﬁne the pseudo labels. Zhang et al. [35] introduced adversarial
learning for biomedical image segmentation by encouraging the segmentation output of unlabeled data to be similar
to annotations of labeled data. However, this category ignores the quality of pseudo labels, where model-generated
annotations can be noisy and may have detrimental eﬀects to the subsequent segmentation model [15].
Another category for semi-supervised learning aims to learn from labeled and unlabeled images simultaneously,
with supervised loss for labeled images and unsupervised images for all images. Among these methods, consistency
learning is widely used by enforcing an invariance of predictions of input images under diﬀerent distributions. For
instance, Samuli et al. [36] proposed temporal ensembling strategy to use exponential moving average (EMA) pre-
dictions for unlabeled data as the consistency targets. However, maintaining the EMA predictions during the training
process is a heavy burden. To issue the problem, Tarvainen et al. [16] proposed to use a teacher model with the EMA
weights of the student model for training. Li et al. [37] applied perturbations like Gaussian noise, randomly rotation
and scaling to the input images and encourage the network to be transformation-consistent for unlabeled data. Yu et
al. [17] extended the mean teacher paradigm with an uncertainty estimation strategy to improve the performance of
consistency-based model so as to learn from more meaningful and reliable targets during training. Luo et al. [38]
proposed to learn from multi-scale consistency between outputs from diﬀerent scales for semi-supervised gross target
volume segmentation. Chaitanya et al. [39] proposed novel contrasting strategies to leverage structural similarity
and learn distinctive representations of local regions. Meyer et al. [40] proposed an uncertainty-aware temporal self-
4


<!-- page 5 -->
learning for semi-supervised segmentation of prostate zones and beyond. Instead of perturbing networks or data for
consistency learning, another line of researches focus on building task-level regularization by adding auxiliary task
to leverage geometric information with distance maps. Li et al. [20] developed a multi-task network to build shape-
aware constraints with adversarial regularization. Luo et al. [21] combined the regression task with the segmentation
task to form a dual-task consistency learning. Zhang et al. [22] extended the learning of cross-task consistency with
mutual learning of dual-task networks and obtain further performance improvement.
2.3. Uncertainty Estimation
For deep neural networks, reliable quantiﬁcation of uncertainty plays a crucial role in evaluating the conﬁdence
of predictions due to the capability to tell when and where the model is likely to make false predictions, especially
for medical imaging area. Recently, many methods have been developed for uncertainty estimation. Uncertainties
for image segmentation are derived from general considerations of the statistical model, from resampling training
data sets in ensemble approaches [41], or from modiﬁcations like Monte Carlo dropout of the predictive procedure
[42]. For semi-supervised learning, the uncertainty can be used to judge whether the model provides accurate and
conﬁdent prediction, which can be leveraged to further exploit the unlabeled data and has been applied to many semi-
supervised medical image segmentation tasks [17, 18]. Wang et al. [43] found that Monte Carlo dropout perform
better for uncertainty estimation. In this work, we incorporate the estimated segmentation uncertainty for consistency
learning, so as to encourage the model to focus on relatively more certain and reliable information learned from
unlabeled data.
3. Methodology
An overview of our proposed framework for semi-supervised medical image segmentation is shown in Fig 1. To
ease the description of the methodology, we formulate the problem of our task as follows. Given a dataset D for
training, we denote the labeled set with M labeled cases as DL = {xl
i, yi}M
i=1, and the unlabeled set with N unlabeled
cases as DU = {xu
i }N
i=1, where xl
i and xu
i denote the input images and yi denotes the corresponding ground truth of
labeled data. For semi-supervised segmentation settings, we aim at building a data-eﬃcient deep learning model with
the combination of DL and DU and making the performance to be comparable to an optimal model trained over fully
labeled dataset D.
3.1. Backbone Network Architecture and Supervised Learning
In most existing approaches, medical image segmentation can be regarded as a pixel-level classiﬁcation task to
generate segmentation probabilistic maps and assign each pixel into a certain class. Other than pixel-wise classiﬁca-
tion using binary or multi-label masks, another line of researches focus on using signed distance maps by transforming
binary masks to gray-level images where the intensities of pixels are changed according to the distance to the closest
boundary [44]. As a traditional method to embed object contours in a higher dimensional space, signed distance
5


<!-- page 6 -->
function (SDF) has recently been incorporated with segmentation CNNs to capture geometric distance information
and obtain further improvements [45, 46, 47]. Speciﬁcally, we introduce the transformation from binary ground truth
to signed distance maps T (x) as follows:
T (x) =

−inf
y∈∂G ∥x −y∥2,
x ∈Gin
0,
x ∈∂G
+ inf
y∈∂G ∥x −y∥2,
x ∈Gout
(1)
where ∥x −y∥2 is the Euclidian distance between voxels x and y. Gin, ∂G, Gout represent the inside, boundary and
outside of the segmentation target, respectively. Generally, SDF takes negative values inside the object and positive
values outside the object, while the absolute value of each pixel is deﬁned by the distance to the closest boundary
point. Following the design of classic encoder-decoder architecture like [24, 25, 26], as shown in Fig 2, an auxiliary
regression branch is added to generate the signed distance maps composed by a 3D convolution block followed by
a hyperbolic tangent activation, in parallel with the classic segmentation branch to generate the segmentation proba-
bilistic maps. The task-level diﬀerence of two branches can lead to inherent prediction perturbation and encourage the
segmentation model to learn diﬀerent representations of segmentation targets from diﬀerent perspectives. For labeled
data, supervision from both segmentation branch and regression branch can be utilized for training. Therefore, the
supervised segmentation loss can be deﬁned as follows:
Lsup(θ; DL) = Ldice( fseg, y) + Lce( fseg, y) + Ldis( fdis, T (y))
(2)
where fseg and fdis represent the output predictions of segmentation branch and regression branch, respectively.
3.2. Intra-Task Consistency Regularization
For semi-supervised medical image segmentation tasks, the improvement of performance beneﬁts from learning
unsupervised knowledge from unlabeled data by generating a supervision signal through unsupervised loss function.
We adopt the mean teacher [16] as the main framework of our semi-supervised segmentation approach. The frame-
work consists of a student model and a teacher model with the same network architecture. Since ensembling the
predictions of the network at diﬀerent training stages can further bootstrap the quality of learning representations
[36], we update the weights of teacher model θ
′ as an exponential moving average (EMA) of the weights of student
model θ as θ
′
t = αθ
′
t−1 + (1 −α)θt, where α is a hyper-parameter named EMA decay.
For consistency learning, the student model learns from the teacher model by minimizing the combination of
supervised segmentation loss and unsupervised consistency loss with respect to the output targets of the teacher
model. In our work, we integrate the consistency loss for both segmentation task and regression task. Following
6


<!-- page 7 -->
Segmentation branch
Regression branch
3D Conv
Tanh
Input Volume
Segmentation Maps
Signed Distance Maps
Segmentation Network
Figure 2: Overview of the backbone network for dual-task learning, where the segmentation branch generates the segmentation probabilistic maps
as output and the regression branch generates the signed distance maps as output.
the design in [17], we estimate the uncertainty with Monte Carlo Dropout [48] by performing T stochastic forward
passes on the teacher model under random dropout. Therefore, the predictive entropy can be summarized as following
equation to approximate the segmentation uncertainty of the model.
ˆpi = 1
T
T
X
t=1
pt
i
u = −
C
X
i=1
ˆpilog( ˆpi)
(3)
where pt
i is the prediction logits of class i at the tth time in the forward pass, C is the number of classes in the
segmentation tasks, ˆpi is the average softmax probability of T stochastic passes from teacher model, and u is the
estimated segmentation uncertainty. For a given input, the overall segmentation uncertainty U is the combination of
voxel-wise uncertainty u. With the guidance of the estimated uncertainty U, we can ﬁlter out the relatively unreliable
predictions with higher uncertainty and use only the certain predictions for the student model to learn from. Therefore,
the intra-task consistency loss can be deﬁned as follows:
Litc(θ; θ
′; D) = β
P I(u < uth)∥fseg −f
′
seg∥2
P I(u < uth)
+ (1 −β)
P I(u < uth)∥fdis −f
′
dis∥2
P I(u < uth)
(4)
where (fseg, fdis) and (f
′
seg, f
′
dis) represent the output of segmentation and regression branch of the student model and
the teacher model, respectively. I(·) is the indicator function used to select out relatively certain predictions, and β is
the balance weight of consistency learning between segmentation task and regression task.
3.3. Cross-Task Consistency Regularization
Since the task-level diﬀerence of two branches can lead to inherent prediction perturbation, diﬀerent tasks can
guide the model to learn diﬀerent representations of segmentation targets from diﬀerent perspectives. Under the
7

[CAPTION] Figure 2: Overview of the backbone network for dual-task learning, where the segmentation branch generates the segmentation probabilistic maps


<!-- page 8 -->
Algorithm 1 Training procedure of our proposed uncertainty-guided mutual consistency learning framework.
Input: A batch of {xl, yl} from labeled dataset DL and {xu} from unlabeled dataset DU.
Output: Trained network N with θt
1: fseg and fdis represent the output predictions of segmentation branch and regression branch to generate segmenta-
tion probabilistic maps and signed distance maps, respectively.
2: while not stopping criterion do
3:
(xl
i, yi), (xu
i ) ←sampled from DL and DU
4:
Generate output segmentation maps fseg, output distance maps fdis and estimated uncertainty U
5:
Calculate supervised segmentation loss Lsup as Eq. (2)
6:
Calculate intra-task consistency losses Litc as Eq. (4)
7:
Calculate cross-task consistency losses Lctc as Eq. (6)
8:
Update the student model’s weights θ with
L = Lsup + λiLitc + λcLctc
9:
Update the teacher model’s weights θ
′ with exponential moving average (EMA) of the student model’s weights
as θ
′
t = αθ
′
t−1 + (1 −α)θt
10: end while
assumption that predictions of the same input data from diﬀerent tasks should be consistent when mapped to the same
predeﬁned space, in contrast to data-level ensembling consistency, we also regularize cross-task consistency between
the outputs of segmentation branch and regression branch to further utilize unlabeled data. To transform the output of
distance maps back to binary segmentation output, we utilize a smooth approximation to the inverse transform as in
[21, 22], which can be deﬁned by
T −1(z) =
1
1 + e−k·z , z ∈GS DF
(5)
where z is the value of signed distance maps at voxel x, and k is a transform factor selected as large as possible to
approximate the transform. Therefore, the cross-task consistency loss for semi-supervised learning can be deﬁned as
follows:
Lctc(θ; D) = ∥fseg −T −1( fdis)∥2
(6)
3.4. Overall Training Procedure
The overall training objective of our proposed framework is to minimize the weighted sum of supervised segmen-
tation loss Lsup, intra-task consistency loss Litc and cross-task consistency loss Lctc. The student model ﬁrst explicitly
learns from labeled data DL via the supervised segmentation loss Lsup. Meanwhile, the student model also acquires
useful information from unlabeled data DL with the guidance of estimated uncertainty. Therefore, the task can be
8


<!-- page 9 -->
formulated as training the network by minimizing the following functions.
min
θ Lsup(θ; DL) + λiLitc(θ; θ
′; D) + λcLctc(θ; D)
(7)
where λi and λc are ramp-up weighting coeﬃcients that control the trade-oﬀbetween supervised and unsupervised
loss, so as to mitigate the disturbance of consistency loss at early training stage. The training objective of our proposed
uncertainty-guided mutual consistency learning framework can be formulated as Algorithm 1.
4. Experiments
4.1. Datasets and Experimental Setup
We extensively evaluate our proposed method on two public datasets. The ﬁrst dataset is Left Atrium (LA) dataset
from Atrial Segmentation Challenge1 [6]. The dataset contains 100 3D gadolinium-enhanced MR imaging scans
(GE-MRIs) and corresponding segmentation masks of left atrium for training and validation. These scans have an
isotropic resolution of 0.625 × 0.625 × 0.625 mm3. Following the same task setting in [17], we split the 100 scans
into 80 scans for training and 20 scans for testing, and apply the same pre-processing methods. Out of the 80 training
scans, we use the same 20%/16 scans as labeled data and the remaining 80%/64 scans as unlabeled data for semi-
supervised segmentation task. The second dataset is Brain Tumor Segmentation (BraTS) 2019 dataset 2 [49]. The
dataset contains multi-institutional preoperative MRI of 335 glioma patients, where each patient has four modalities
of MRI scans including T1, T1Gd, T2 and T2-FLAIR with neuroradiologist-examined labels. We use T2-FLAIR
for whole tumor segmentation since such modality can better manifest the malignant tumors [50]. All the scans are
resampled to the same resolution of 1 × 1 × 1 mm3 with intensity normalized to zero mean and unit variance. In our
experiments, we split the dataset into 250 scans for training, 25 scans for validation and the remaining 60 scans for
testing. Among the 250 training scans, we conduct experiments under two diﬀerent settings with 10%/25 and 20%/50
scans as labeled data and the remaining scans as unlabeled data.
4.2. Implementing Details and Evaluation Metrics
All of our experiments are implemented in Python with PyTorch, using an NVIDIA Tesla V100 GPU with 32GB
memory. We adopt the same V-Net [26] as the backbone structure for all experiments to ensure a fair comparison.
To control the balance between supervised segmentation loss and unsupervised consistency loss, we use a Gaussian
ramp-up function λ(t) = 0.1∗e−5(1−t/tmax) as [16] in our experiments, where t represents the current number of iterations
and tmax represents the maximum number of iterations. We use the Stochastic Gradient Descent (SGD) optimizer to
update the network parameters with an initial learning rate of 0.01 decayed by 0.1 every 2500 iterations. The maximum
training iterations is set to 6,000. The batch size is set to 4, consisting of 2 labeled images and 2 unlabeled images in
1http://atriaseg2018.cardiacatlas.org/data/
2https://ieee-dataport.org/competitions/brats-miccai-brain-tumor-dataset
9


<!-- page 10 -->
Table 1: Ablation analysis of our mutual consistency learning framework on LA dataset. The arrows of evaluation metrics indicate which direction
is better.
Method
Supervised Loss
Consistency Loss
Metrics
Lseg
Ldis
Litc
Lctc
Dice[%] ↑
Jaccard[%] ↑
ASD[voxel] ↓
95HD[voxel] ↓
(1)
✓
-
-
-
86.03
76.06
3.51
14.26
(2)
✓
✓
-
-
87.88
78.77
2.81
10.25
(3)
✓
-
✓
-
88.68
79.90
2.71
9.07
(4)
✓
✓
✓
-
89.15
80.58
2.03
8.15
(5)
✓
✓
-
✓
89.42
80.89
2.10
7.32
(6)
✓
✓
✓
✓
90.16
82.18
1.98
6.50
Table 2: Experimental results of diﬀerent balance weight β of segmentation task and regression task for self-ensembling on LA dataset.
β
Metrics
Dice[%] ↑
Jaccard[%] ↑
ASD[voxel] ↓
95HD[voxel] ↓
0
89.60
81.30
2.51
8.66
0.25
89.86
81.75
2.22
7.18
0.5
90.07
82.03
2.17
7.57
0.75
90.16
82.18
1.98
6.50
1
89.99
81.93
2.74
7.95
each mini-batch. We randomly crop 112×112×80 sub-volumes as the network input and the ﬁnal segmentation results
are obtained using a sliding window strategy. We use the standard data augmentation techniques on-the-ﬂy to avoid
overﬁtting during the training procedure [51], including randomly ﬂipping, and rotating with 90, 180 and 270 degrees
along the axial plane. To quantitatively evaluate the segmentation results, we use four complementary evaluation
metrics. Dice similarity coeﬃcient (Dice) and Jaccard Index (Jaccard), two region-based metrics, are used to measure
the region mismatch. Average surface distance (ASD) and 95% HausdorﬀDistance (95HD), two boundary-based
metrics, are used to evaluate the boundary errors between the segmentation results and the ground truth.
4.3. Ablation Experiments
Since the classic consistency learning procedure of self-ensembling is based on the segmentation predictions, for
our segmentation network with dual-task outputs , we use a balance weight β to control the consistency learning
between segmentation task and regression task. We conduct experiments to evaluate the selection of β in our mutual
consistency learning framework. The quantitative results are shown in Table 2. When β = 1 or 0, the model achieves
lower performance because the ensembling of predictions is only based on segmentation or regression branch, while
another branch is neglected. It can be found that when β = 0.75, the framework can achieve the best performance.
10


**[Table p10.1]**
| Method | Supervised Loss | Consistency Loss | Metrics |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | L L seg dis | L L itc ctc | Dice[%] ↑ | Jaccard[%] ↑ | ASD[voxel] ↓ | 95HD[voxel] ↓ |
| (1) (2) | ✓ - ✓ ✓ | - - - - | 86.03 87.88 | 76.06 78.77 | 3.51 2.81 | 14.26 10.25 |
| (3) (4) (5) (6) | ✓ - ✓ ✓ ✓ ✓ ✓ ✓ | ✓ - ✓ - - ✓ ✓ ✓ | 88.68 89.15 89.42 90.16 | 79.90 80.58 80.89 82.18 | 2.71 2.03 2.10 1.98 | 9.07 8.15 7.32 6.50 |


**[Table p10.2]**
| β | Metrics |  |  |  |
| --- | --- | --- | --- | --- |
|  | Dice[%] ↑ | Jaccard[%] ↑ | ASD[voxel] ↓ | 95HD[voxel] ↓ |
| 0 0.25 0.5 0.75 1 | 89.60 89.86 90.07 90.16 89.99 | 81.30 81.75 82.03 82.18 81.93 | 2.51 2.22 2.17 1.98 2.74 | 8.66 7.18 7.57 6.50 7.95 |

[CAPTION] Table 1: Ablation analysis of our mutual consistency learning framework on LA dataset. The arrows of evaluation metrics indicate which direction

[CAPTION] Table 2: Experimental results of diﬀerent balance weight β of segmentation task and regression task for self-ensembling on LA dataset.


<!-- page 11 -->
DSC 89.48%
DSC 89.93%
DSC 90.16%
DSC 90.07%
DSC 90.03%
Figure 3: Comparison of Dice performance with diﬀerent uncertainty threshold uth on left atrium segmentation dataset.
Table 3: Quantitative comparison between our methods and other semi-supervised methods on LA dataset. All the models use the same V-Net
as the backbone. The ﬁrst and second rows are upper-bound performance and fully supervised baseline. * and ** denote the p-value ¡ 0.05 and
p-value ¡ 0.01 based on paired t-test when comparing the proposed method with others.
Method
Scans used
Metrics
Labeled
Unlabeled
Dice[%] ↑
Jaccard[%] ↑
ASD[voxel] ↓
95HD[voxel] ↓
Supervised baseline
16
0
86.03∗∗
76.06∗∗
3.51∗∗
14.26∗∗
TCSE [37]
16
64
88.15∗∗
79.20∗∗
2.44∗
9.57∗
MT [16]
16
64
88.23∗∗
79.29∗∗
2.73∗
10.64∗∗
UA-MT [17]
16
64
88.88∗
80.21∗
2.26
7.32
Entropy Mini [52]
16
64
88.45∗
79.51∗∗
3.72∗∗
14.14∗∗
SASS [20]
16
64
89.54
81.24
2.20
8.24∗
DUWM [18]
16
64
89.65
81.35
2.03
7.04
URPC [38]
16
64
88.74∗
79.93∗
3.66∗∗
12.73∗∗
DTC [21]
16
64
89.42∗
80.89∗
2.10
7.32
UG-MCL (Ours)
16
64
90.16
82.18
1.98
6.50
Supervised upper-bound
80
0
91.14
83.82
1.52
5.75
Therefore, we use β = 0.75 in our framework in the following experiments. Besides, following the selection of
uncertainty threshold in [17] with Gaussian ramp-up paradigm, we conduct experiments to show the inﬂuence of
diﬀerent choices for uncertainty threshold. The uncertainty threshold selection with corresponding Dice score is
11


**[Table p11.1]**
| Method | Scans used |  | Metrics |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | Labeled | Unlabeled | Dice[%] ↑ | Jaccard[%] ↑ | ASD[voxel] ↓ | 95HD[voxel] ↓ |
| Supervised baseline TCSE [37] MT [16] UA-MT [17] Entropy Mini [52] SASS [20] DUWM [18] URPC [38] DTC [21] UG-MCL (Ours) | 16 16 16 16 16 16 16 16 16 16 | 0 64 64 64 64 64 64 64 64 64 | 86.03∗∗ 88.15∗∗ 88.23∗∗ 88.88∗ 88.45∗ 89.54 89.65 88.74∗ 89.42∗ 90.16 | 76.06∗∗ 79.20∗∗ 79.29∗∗ 80.21∗ 79.51∗∗ 81.24 81.35 79.93∗ 80.89∗ 82.18 | 3.51∗∗ 2.44∗ 2.73∗ 2.26 3.72∗∗ 2.20 2.03 3.66∗∗ 2.10 1.98 | 14.26∗∗ 9.57∗ 10.64∗∗ 7.32 14.14∗∗ 8.24∗ 7.04 12.73∗∗ 7.32 6.50 |
| Supervised upper-bound | 80 | 0 | 91.14 | 83.82 | 1.52 | 5.75 |

[CAPTION] Figure 3: Comparison of Dice performance with diﬀerent uncertainty threshold uth on left atrium segmentation dataset.

[CAPTION] Table 3: Quantitative comparison between our methods and other semi-supervised methods on LA dataset. All the models use the same V-Net


<!-- page 12 -->
Figure 4: Comparison of segmentation performance of our proposed method (in red) with comparison to fully supervised baseline (in black),
UA-MT (in blue) and DTC (in green) using diﬀerent percentages of labeled data (5%, 10%, 15%, 20%, 30% and 50%) on LA dataset.
illustrated in Figure 3. Speciﬁcally, the upper one in black with uncertainty threshold ﬁxed to Umax for all iterations.
Based on the results, we use uncertainty threshold in red ramping from 3
4Umax to Umax in our framework.
To evaluate the eﬀectiveness of our proposed framework, we conduct ablation studies by removing diﬀerent com-
ponents in our method. All the experiments are performed on LA dataset using 16 labeled scans and 64 labeled
scans for comparison. The testing results of ablation experiments are shown in Table 1. Experiment (1) and (2) are
the supervised baseline of V-Net trained with only labeled data. We can observe that by utilizing unlabeled data, all
semi-supervised methods can signiﬁcantly improve the segmentation performance compared with supervised baseline
results. In experiment (3), we only activate the classic segmentation branch for consistency learning, while the regres-
sion branch is removed. In experiment (4), the supervision based on the transformed distance maps of ground truth
is utilized to regularize the training. We can observe that adding supervision on distance maps further improves the
segmentation by 0.47% in Dice and 0.68% in Jaccard. In experiment (5), when removing the intra-task consistency
learning of teacher-student framework, the model degenerates into a dual-task V-Net with only cross-task consistency
learning. From the results, we can observe that with the integration of intra-task consistency and cross-task con-
12

[CAPTION] Figure 4: Comparison of segmentation performance of our proposed method (in red) with comparison to fully supervised baseline (in black),


<!-- page 13 -->
Image
Supervised
Ours
GT
UA-MT
DTC
Figure 5: Visual comparison of the left atrium segmentation results of our proposed method with comparison to fully supervised baseline and other
state-of-the-art semi-supervised methods using 20% labeled data.
sistency for mutual consistency learning, the performance of our framework is further promoted. Besides, we also
conduct paired t-test to validate the signiﬁcance of the improvements between our method and the ablation studies.
The results show that all the improvements are statistically signiﬁcant at p < 0.05, demonstrating the eﬀectiveness of
our framework.
4.4. Performance of Using Diﬀerent Percentages of Labeled Data
We performed a study on data utilization eﬃciency of our semi-supervised framework with comparison to fully
supervised baseline and other two semi-supervised methods including uncertainty-aware mean teacher (UA-MT) [17]
and dual task consistency learning (DTC) [21]. We conduct experiments of using diﬀerent percentages of labeled
data of LA dataset. The visualization of segmentation results are presented in Figure 4. It can be observed that all
semi-supervised methods consistently perform better than the supervised baseline in diﬀerent labeled data settings.
Besides, our method outperforms other semi-supervised segmentation methods consistently with diﬀerent percentages
of labeled data, demonstrating the superiority of our proposed framework. Compared with the supervised baseline,
our proposed method obtains improvement of 21.07%, 6.95%, 5.78%, 4.13%, 2.05% and 1.36% in Dice by using 5%,
10%, 15%, 20%, 30% and 50% of labeled training data, respectively. We notice that the performance improvement of
semi-supervised learning gradually narrows with the increase of labeled data, which is in line with the common sense.
4.5. Comparison Experiments with Other Semi-Supervised Segmentation Methods on LA Dataset
To demonstrate the eﬀectiveness of our method, a comprehensive comparison with existing methods is conducted
on LA dataset. We evaluate our method with comparisons to several recent state-of-the-art semi-supervised segmenta-
tion methods, including transformation-consistent self-ensembline (TCSE) [37], mean teacher (MT) [16], uncertainty-
aware mean teacher (UA-MT) [17], entropy minimization (Entropy Mini) [52], shape-aware semi-supervised segmen-
tation (SASS) [20], double uncertainty weighted method (DUWM) [18], dual task consistency learning (DTC) [21]
13

[CAPTION] Figure 5: Visual comparison of the left atrium segmentation results of our proposed method with comparison to fully supervised baseline and other


<!-- page 14 -->
Table 4: Quantitative comparison between our methods and other semi-supervised methods on BraTS 2019 dataset. All the models use the same
V-Net as the backbone. The ﬁrst and second rows are upper-bound performance and fully supervised baseline. * and ** denote the p-value ¡ 0.05
and p-value ¡ 0.01 based on paired t-test when comparing the proposed method with others.
Method
Scans used
Metrics
Labeled
Unlabeled
Dice[%] ↑
Jaccard[%] ↑
ASD[voxel] ↓
95HD[voxel] ↓
Supervised baseline
25
0
73.00∗∗
59.96∗∗
2.96∗∗
42.64∗∗
MT [16]
25
225
81.21∗∗
70.83∗∗
2.45
14.72∗∗
UA-MT [17]
25
225
80.85∗∗
70.32∗∗
2.57∗
14.61∗
Entropy Mini [52]
25
225
81.74∗
71.42∗
2.37
13.71∗
URPC [38]
25
225
81.80∗
71.63
2.48
11.50
DTC [21]
25
225
81.96
71.84
2.43
12.08
UG-MCL (Ours)
25
225
82.82
72.77
2.30
11.29
Supervised baseline
50
0
76.14∗∗
64.15∗∗
2.70∗∗
36.01∗∗
MT [16]
50
200
82.38∗
72.32∗∗
2.21
14.54
UA-MT [17]
50
200
81.57∗∗
71.42∗∗
2.49∗
13.98
Entropy Mini [52]
50
200
82.37∗
72.28∗
2.30
15.83∗
URPC [38]
50
200
82.80
72.72∗
2.72∗∗
12.48
DTC [21]
50
200
82.78∗
72.47∗∗
2.20
13.43
UG-MCL (Ours)
50
200
83.61
73.98
2.26
11.44
and uncertainty rectiﬁed pyramid consistency (URPC) [38]. To ensure a fair comparison, we used the same network
backbone in these methods. Fig 5 presents some visualization of segmentation results from diﬀerent semi-supervised
segmentation methods. It can be observed that our proposed method generates more accurate and complete segmen-
tation than other methods. Table 3 shows the quantitative results on the LA dataset. As a contrast, we also conduce
experiments of V-Net under fully-supervised settings with 20% and all labeled data as the lower-bound and upper-
bound references for the task. Compared with semi-supervised learning settings, only labeled scans are used for the
lower-bound subtask, while both labeled and unlabeled scans with annotations are used for the upper-bound sub-
task. It can be observed that by eﬃciently leveraging unlabeled data for training, our proposed method can achieve
signiﬁcant performance gains from 86.03% to 90.16% of Dice and 76.06% to 82.18% of Jaccard. Meanwhile, our
method obtains comparable results of 90.16% in Dice compared with the upper-bound performance of 91.14%, and
signiﬁcantly outperforms other semi-supervised segmentation methods in terms of all the four evaluation metrics.
4.6. Comparison Experiments with Other Semi-Supervised Segmentation Methods on BraTS Dataset
To further validate the eﬀectiveness and generalization ability of our proposed method, we conduct experiments on
BraTS 2019 dataset with comparison to existing methods under 10% and 20% labeled data settings. Table 4 presents
14


**[Table p14.1]**
| Method | Scans used |  | Metrics |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | Labeled | Unlabeled | Dice[%] ↑ | Jaccard[%] ↑ | ASD[voxel] ↓ | 95HD[voxel] ↓ |
| Supervised baseline MT [16] UA-MT [17] Entropy Mini [52] URPC [38] DTC [21] UG-MCL (Ours) | 25 25 25 25 25 25 25 | 0 225 225 225 225 225 225 | 73.00∗∗ 81.21∗∗ 80.85∗∗ 81.74∗ 81.80∗ 81.96 82.82 | 59.96∗∗ 70.83∗∗ 70.32∗∗ 71.42∗ 71.63 71.84 72.77 | 2.96∗∗ 2.45 2.57∗ 2.37 2.48 2.43 2.30 | 42.64∗∗ 14.72∗∗ 14.61∗ 13.71∗ 11.50 12.08 11.29 |
| Supervised baseline MT [16] UA-MT [17] Entropy Mini [52] URPC [38] DTC [21] UG-MCL (Ours) | 50 50 50 50 50 50 50 | 0 200 200 200 200 200 200 | 76.14∗∗ 82.38∗ 81.57∗∗ 82.37∗ 82.80 82.78∗ 83.61 | 64.15∗∗ 72.32∗∗ 71.42∗∗ 72.28∗ 72.72∗ 72.47∗∗ 73.98 | 2.70∗∗ 2.21 2.49∗ 2.30 2.72∗∗ 2.20 2.26 | 36.01∗∗ 14.54 13.98 15.83∗ 12.48 13.43 11.44 |

[CAPTION] Table 4: Quantitative comparison between our methods and other semi-supervised methods on BraTS 2019 dataset. All the models use the same


<!-- page 15 -->
Image
Supervised
Ours
GT
UA-MT
DTC
Figure 6: Visual comparison of the brain tumor segmentation results of our proposed method with comparison to fully supervised baseline and
other state-of-the-art semi-supervised methods using 10% labeled data.
the segmentation results of our proposed method and other semi-supervised segmentation methods on the dataset.
We can observe that comparing with left atrium segmentation, the brain tumor segmentation is a more challenging
task due to the irregular shape and boundary of tumors. With 10% and 20% labeled data for training, the supervised
baseline only yields an average Dice of 73.00% and 76.14%, respectively. For semi-supervised segmentation settings,
although the performances are comparable among all the comparing methods, our proposed method outperforms other
state-of-the-art semi-supervised methods with diﬀerent amounts of labeled data, demonstrating the superiority of our
framework to eﬀectively exploit the information from unlabeled data. In Fig 6, we present some of the brain tumor
segmentation results using 10% labeled data. We can observe that our proposed method generates more accurate
predictions compared with other methods, which further demonstrates the eﬀectiveness of our proposed method.
5. Conclusion
Despite existing deep learning-based medical image segmentation methods have achieved great success, it is also
limited by requiring large amount of expert-examined annotations. Semi-supervised segmentation by encouraging
segmentation models to utilize unlabeled data which is much easier to acquire havs shown the potential to deal with
this challenge [14]. In this paper, we present a novel uncertainty-guided mutual consistency learning framework for
semi-supervised medical image segmentation. We use dual-task backbone network with two output branches to gener-
ate segmentation probabilistic maps and signed distance maps simultaneously. To eﬀectively exploit unlabeled data for
training, our framework integrates intra-task consistency learning from up-to-date predictions for self-ensembling and
cross-task consistency learning from task-level regularization to exploit geometric shape information. Our proposed
framework is guided by the estimated segmentation uncertainty of models to select out relatively certain predictions
for consistency learning, so as to eﬀectively exploit more reliable information from unlabeled data. Extensive ex-
periments on two public medical image segmentation datasets demonstrate the superiority of our proposed method
15

[CAPTION] Figure 6: Visual comparison of the brain tumor segmentation results of our proposed method with comparison to fully supervised baseline and


<!-- page 16 -->
over other semi-supervised learning methods. Based on the experimental results, our proposed method can achieve
signiﬁcant performance improvement by leveraging unlabeled data, with up to 4.13% and 9.82% in Dice coeﬃcient
compared to supervised baseline on left atrium segmentation and brain tumor segmentation, respectively. Besides,
our proposed method achieve better segmentation performance compared with other semi-supervised segmentation
methods under the same backbone network and task settings on both datasets. The proposed method has the potential
to be applied to further clinical applications to ease the burden of annotation cost. In this work, one limitation is we
only focus on single-class segmentation tasks on relatively small datasets. In the future work, we aim to focus on
more challenging multi-class segmentation tasks on more diverse datasets like [7] to further improve and validate the
performance of semi-supervised medical image segmentation methods in real-world clinical applications.
Declaration of Competing Interest
There are no conﬂicts of interest.
Acknowledgement
This work is supported in part by the National Key Research and Development Program of China (2016YFF0201002),
and in part by the University Synergy Innovation Program of Anhui Province (GXXT-2019-044).
References
[1] B. Van Ginneken, C. M. Schaefer-Prokop, M. Prokop, Computer-aided diagnosis: how to move from the laboratory to the clinic, Radiology
261 (3) (2011) 719–732.
[2] W. J. Niessen, Mr brain image analysis in dementia: From quantitative imaging biomarkers to ageing brain models and imaging genetics,
Medical Image Analysis 33 (2016) 107–113.
[3] G. Litjens, T. Kooi, B. E. Bejnordi, A. A. A. Setio, F. Ciompi, M. Ghafoorian, J. A. Van Der Laak, B. Van Ginneken, C. I. S´anchez, A survey
on deep learning in medical image analysis, Medical image analysis 42 (2017) 60–88.
[4] O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, X. Yang, P.-A. Heng, I. Cetin, K. Lekadir, O. Camara, M. A. G. Ballester, et al., Deep
learning techniques for automatic mri cardiac multi-structures segmentation and diagnosis: is the problem solved?, IEEE Transactions on
Medical Imaging 37 (11) (2018) 2514–2525.
[5] A. Lalande, Z. Chen, T. Pommier, T. Decourselle, A. Qayyum, M. Salomon, D. Ginhac, Y. Skandarani, A. Boucher, K. Brahim, et al., Deep
learning methods for automatic evaluation of delayed enhancement-mri. the results of the emidec challenge, Medical Image Analysis 79
(2022) 102428.
[6] Z. Xiong, Q. Xia, Z. Hu, N. Huang, C. Bian, Y. Zheng, S. Vesal, N. Ravikumar, A. Maier, X. Yang, et al., A global benchmark of algorithms for
segmenting the left atrium from late gadolinium-enhanced cardiac magnetic resonance imaging, Medical Image Analysis 67 (2021) 101832.
[7] J. Ma, Y. Zhang, S. Gu, C. Zhu, C. Ge, Y. Zhang, X. An, C. Wang, Q. Wang, X. Liu, et al., Abdomenct-1k: Is abdominal organ segmentation
a solved problem, IEEE Transactions on Pattern Analysis and Machine Intelligence (2021).
[8] N. Heller, F. Isensee, K. H. Maier-Hein, X. Hou, C. Xie, F. Li, Y. Nan, G. Mu, Z. Lin, M. Han, et al., The state of the art in kidney and kidney
tumor segmentation in contrast-enhanced ct imaging: Results of the kits19 challenge, Medical Image Analysis (2020) 101821.
[9] Y. Zhang, Q. Liao, L. Ding, J. Zhang, Bridging 2d and 3d segmentation networks for computation-eﬃcient volumetric medical image
segmentation: An empirical study of 2.5 d solutions, Computerized Medical Imaging and Graphics (2022) 102088.
16


<!-- page 17 -->
[10] J. Ma, Y. Wang, X. An, C. Ge, Z. Yu, J. Chen, Q. Zhu, G. Dong, J. He, Z. He, et al., Towards data-eﬃcient learning: A benchmark for
covid-19 ct lung and infection segmentation, Medical physics (2020).
[11] N. Tajbakhsh, L. Jeyaseelan, Q. Li, J. N. Chiang, Z. Wu, X. Ding, Embracing imperfect datasets: A review of deep learning solutions for
medical image segmentation, Medical Image Analysis 63 (2020) 101693.
[12] Y. Zhang, Q. Liao, L. Yuan, H. Zhu, J. Xing, J. Zhang, Exploiting shared knowledge from non-covid lesions for annotation-eﬃcient covid-19
ct lung infection segmentation, IEEE Journal of Biomedical and Health Informatics (2021).
[13] V. Cheplygina, M. de Bruijne, J. P. Pluim, Not-so-supervised: a survey of semi-supervised, multi-instance, and transfer learning in medical
image analysis, Medical image analysis 54 (2019) 280–296.
[14] R. Jiao, Y. Zhang, L. Ding, R. Cai, J. Zhang, Learning with limited annotations: A survey on deep semi-supervised learning for medical
image segmentation, arXiv preprint arXiv:2207.14191 (2022).
[15] S. Min, X. Chen, Z.-J. Zha, F. Wu, Y. Zhang, A two-stream mutual attention network for semi-supervised biomedical segmentation with noisy
labels, in: Proceedings of the AAAI Conference on Artiﬁcial Intelligence, Vol. 33, 2019, pp. 4578–4585.
[16] A. Tarvainen, H. Valpola, Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning
results, in: Proceedings of the 31st International Conference on Neural Information Processing Systems, 2017, pp. 1195–1204.
[17] L. Yu, S. Wang, X. Li, C.-W. Fu, P.-A. Heng, Uncertainty-aware self-ensembling model for semi-supervised 3d left atrium segmentation, in:
International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer, 2019, pp. 605–613.
[18] Y. Wang, Y. Zhang, J. Tian, C. Zhong, Z. Shi, Y. Zhang, Z. He, Double-uncertainty weighted method for semi-supervised learning, in:
International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer, 2020, pp. 542–551.
[19] W. Hang, W. Feng, S. Liang, L. Yu, Q. Wang, K.-S. Choi, J. Qin, Local and global structure-aware entropy regularized mean teacher model
for 3d left atrium segmentation, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer,
2020, pp. 562–571.
[20] S. Li, C. Zhang, X. He, Shape-aware semi-supervised 3d semantic segmentation for medical images, in: International Conference on Medical
Image Computing and Computer-Assisted Intervention, Springer, 2020, pp. 552–561.
[21] X. Luo, J. Chen, T. Song, G. Wang, Semi-supervised medical image segmentation through dual-task consistency, in: Proceedings of the
AAAI Conference on Artiﬁcial Intelligence, Vol. 35, 2021, pp. 8801–8809.
[22] Y. Zhang, J. Zhang, Dual-task mutual learning for semi-supervised medical image segmentation, in: Pattern Recognition and Computer
Vision, 2021, pp. 548–559.
[23] J. Long, E. Shelhamer, T. Darrell, Fully convolutional networks for semantic segmentation, in: Proceedings of the IEEE conference on
computer vision and pattern recognition, 2015, pp. 3431–3440.
[24] O. Ronneberger, P. Fischer, T. Brox, U-net: Convolutional networks for biomedical image segmentation, in: International Conference on
Medical image computing and computer-assisted intervention, Springer, 2015, pp. 234–241.
[25]
¨O. C¸ ic¸ek, A. Abdulkadir, S. S. Lienkamp, T. Brox, O. Ronneberger, 3d u-net: learning dense volumetric segmentation from sparse annotation,
in: International conference on medical image computing and computer-assisted intervention, Springer, 2016, pp. 424–432.
[26] F. Milletari, N. Navab, S.-A. Ahmadi, V-net: Fully convolutional neural networks for volumetric medical image segmentation, in: 2016 fourth
international conference on 3D vision (3DV), IEEE, 2016, pp. 565–571.
[27] Z. Zhou, M. M. R. Siddiquee, N. Tajbakhsh, J. Liang, Unet++: Redesigning skip connections to exploit multiscale features in image segmen-
tation, IEEE transactions on medical imaging 39 (6) (2019) 1856–1867.
[28] X. Li, H. Chen, X. Qi, Q. Dou, C.-W. Fu, P.-A. Heng, H-denseunet: hybrid densely connected unet for liver and tumor segmentation from ct
volumes, IEEE transactions on medical imaging 37 (12) (2018) 2663–2674.
[29] Y. Zhang, L. Yuan, Y. Wang, J. Zhang, Sau-net: eﬃcient 3d spine mri segmentation using inter-slice attention, in: Medical Imaging with
Deep Learning, PMLR, 2020, pp. 903–913.
[30] Q. Jin, Z.-P. Meng, C. Sun, L. Wei, R. Su, Ra-unet: A hybrid deep attention-aware network to extract liver and tumor in ct scans, Frontiers in
Bioengineering and Biotechnology 8 (2020).
17


<!-- page 18 -->
[31] O. Oktay, J. Schlemper, L. L. Folgoc, M. Lee, M. Heinrich, K. Misawa, K. Mori, S. McDonagh, N. Y. Hammerla, B. Kainz, et al., Attention
u-net: Learning where to look for the pancreas, arXiv preprint arXiv:1804.03999 (2018).
[32] F. Isensee, P. F. Jaeger, S. A. A. Kohl, J. Petersen, K. Maier-Hein, nnu-net: a self-conﬁguring method for deep learning-based biomedical
image segmentation., Nature methods (2020).
[33] M. Antonelli, A. Reinke, S. Bakas, K. Farahani, A. Kopp-Schneider, B. A. Landman, G. Litjens, B. Menze, O. Ronneberger, R. M. Summers,
et al., The medical segmentation decathlon, Nature Communications 13 (1) (2022) 1–13.
[34] W. Bai, O. Oktay, M. Sinclair, H. Suzuki, M. Rajchl, G. Tarroni, B. Glocker, A. King, P. M. Matthews, D. Rueckert, Semi-supervised
learning for network-based cardiac mr image segmentation, in: International Conference on Medical Image Computing and Computer-
Assisted Intervention, Springer, 2017, pp. 253–260.
[35] Y. Zhang, L. Yang, J. Chen, M. Fredericksen, D. P. Hughes, D. Z. Chen, Deep adversarial networks for biomedical image segmentation
utilizing unannotated images, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer,
2017, pp. 408–416.
[36] S. Laine, T. Aila, Temporal ensembling for semi-supervised learning, arXiv preprint arXiv:1610.02242 (2016).
[37] X. Li, L. Yu, H. Chen, C.-W. Fu, L. Xing, P.-A. Heng, Transformation-consistent self-ensembling model for semisupervised medical image
segmentation, IEEE Transactions on Neural Networks and Learning Systems (2020).
[38] X. Luo, W. Liao, J. Chen, T. Song, Y. Chen, S. Zhang, N. Chen, G. Wang, S. Zhang, Eﬃcient semi-supervised gross target volume of nasopha-
ryngeal carcinoma segmentation via uncertainty rectiﬁed pyramid consistency, in: International Conference on Medical Image Computing
and Computer-Assisted Intervention, Springer, 2021, pp. 318–329.
[39] K. Chaitanya, E. Erdil, N. Karani, E. Konukoglu, Contrastive learning of global and local features for medical image segmentation with
limited annotations, Advances in Neural Information Processing Systems 33 (2020) 12546–12558.
[40] A. Meyer, S. Ghosh, D. Schindele, M. Schostak, S. Stober, C. Hansen, M. Rak, Uncertainty-aware temporal self-learning (uats): Semi-
supervised learning for segmentation of prostate zones and beyond, Artiﬁcial Intelligence in Medicine 116 (2021) 102073.
[41] B. Lakshminarayanan, A. Pritzel, C. Blundell, Simple and scalable predictive uncertainty estimation using deep ensembles, Advances in
Neural Information Processing Systems 30 (2017).
[42] Y. Gal, Z. Ghahramani, Dropout as a bayesian approximation: Representing model uncertainty in deep learning, in: international conference
on machine learning, PMLR, 2016, pp. 1050–1059.
[43] G. Wang, S. Zhai, G. Lasio, B. Zhang, B. Yi, S. Chen, T. J. MacVittie, D. N. Metaxas, J. Zhou, S. Zhang, Semi-supervised segmentation
of radiation-induced pulmonary ﬁbrosis from lung ct scans with multi-scale guided dense attention, IEEE transactions on medical imaging
(2021).
[44] J. Ma, Z. Wei, Y. Zhang, Y. Wang, R. Lv, C. Zhu, C. Gaoxiang, J. Liu, C. Peng, L. Wang, et al., How distance transform maps boost
segmentation cnns: an empirical study, in: Medical Imaging with Deep Learning, PMLR, 2020, pp. 479–492.
[45] F. Navarro, S. Shit, I. Ezhov, J. Paetzold, A. Gaﬁta, J. C. Peeken, S. E. Combs, B. H. Menze, Shape-aware complementary-task learning for
multi-organ segmentation, in: International Workshop on Machine Learning in Medical Imaging, Springer, 2019, pp. 620–627.
[46] S. Dangi, C. A. Linte, Z. Yaniv, A distance map regularized cnn for cardiac cine mr image segmentation, Medical physics 46 (12) (2019)
5637–5651.
[47] Y. Wang, X. Wei, F. Liu, J. Chen, Y. Zhou, W. Shen, E. K. Fishman, A. L. Yuille, Deep distance transform for tubular structure segmentation
in ct scans, 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2020) 3832–3841.
[48] A. Kendall, Y. Gal, What uncertainties do we need in bayesian deep learning for computer vision?, Advances in Neural Information Processing
Systems 30 (2017) 5574–5584.
[49] S. S. Bakas, Brats miccai brain tumor dataset (2020). doi:10.21227/hdtd-5j88.
URL https://dx.doi.org/10.21227/hdtd-5j88
[50] R. A. Zeineldin, M. E. Karar, J. Coburger, C. R. Wirtz, O. Burgert, Deepseg: deep neural network framework for automatic brain tumor
segmentation using magnetic resonance ﬂair images, International journal of computer assisted radiology and surgery 15 (6) (2020) 909–920.
18


<!-- page 19 -->
[51] L. Yu, J.-Z. Cheng, Q. Dou, X. Yang, H. Chen, J. Qin, P.-A. Heng, Automatic 3d cardiovascular mr segmentation with densely-connected
volumetric convnets, in: International conference on medical image computing and computer-assisted intervention, Springer, 2017, pp. 287–
295.
[52] T.-H. Vu, H. Jain, M. Bucher, M. Cord, P. P´erez, Advent: Adversarial entropy minimization for domain adaptation in semantic segmentation,
in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 2517–2526.
19