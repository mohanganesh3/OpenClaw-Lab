<!-- page 1 -->
SAM-DRIVEN WEAKLY SUPERVISED NODULE SEGMENTATION WITH
UNCERTAINTY-AWARE CROSS TEACHING
Xingyue Zhao1, Peiqi Li1, Xiangde Luo2,3, Meng Yang5, Shi Chang4∗, Zhongyu Li1∗
1School of Software Engineering, Xi’an Jiaotong University, Xi’an, China
2University of Electronic Science and Technology of China, Chengdu, China
3Shanghai Artificial Intelligence Laboratory, Shanghai 200030, China
4Department of General Surgery, Xiangya Hospital, Central South University, China
5Hunan Frontline Medical Technology Co., Ltd
ABSTRACT
Automated nodule segmentation is essential for computer-
assisted diagnosis in ultrasound images. Nevertheless, most
existing methods depend on precise pixel-level annotations by
medical professionals, a process that is both costly and labor-
intensive.
Recently, segmentation foundation models like
SAM have shown impressive generalizability on natural im-
ages, suggesting their potential as pseudo-labelers. However,
accurate prompts remain crucial for their success in medical
images. In this work, we devise a novel weakly supervised
framework that effectively utilizes the segmentation foun-
dation model to generate pseudo-labels from aspect ration
annotations for automatic nodule segmentation. Specifically,
we develop three types of bounding box prompts based on
scalable shape priors, followed by an adaptive pseudo-label
selection module to fully exploit the prediction capabilities
of the foundation model for nodules.
We also present a
SAM-driven uncertainty-aware cross-teaching strategy. This
approach integrates SAM-based uncertainty estimation and
label-space perturbations into cross-teaching to mitigate the
impact of pseudo-label inaccuracies on model training. Ex-
tensive experiments on two clinically collected ultrasound
datasets demonstrate the superior performance of our pro-
posed method.
Index Terms— Weakly supervised learning, Ultrasound
nodule segmentation, Aspect ratio annotations
1. INTRODUCTION
Deep learning has significantly advanced automatic medical
image segmentation, key for computer-assisted diagnosis.
Most of the existing methods need extensive, precise pixel-
level annotations, which is a laborious process, especially
for ultrasound images with their lower resolution and irreg-
ular lesions.
Recently, researchers have explored weakly
supervised learning to reduce the need for extensive data
∗Correspondence: zhongyuli@xjtu.edu.cn, changshi@csu.edu.cn
Clinical Label
Bounding Box
SAM’s Output
Fig. 1. Examples for aspect ratio annotation and prediction
results of segment anything model in different bounding box
prompts setting.
annotation.
Grandvalet et al. [1] proposed a method that
diminishes uncertainty in unannotated data, allowing the
model to produce high-confidence predictions. Javanmardi
et al. [2] emphasized promoting spatial smoothness by re-
ducing pixel variation among neighboring regions. Kim et
al. [3] introduced a novel loss function that balances seg-
mentation smoothness with data fidelity, thereby promoting
consistent segmentations. Additionally, several researchers
have explored strategies to train models using sparse annota-
tions, including image-level annotations [4], scribbles [5, 6],
bounding boxes [7], and point annotations [8]. For example,
in the realm of tumor segmentation, some researchers have
tried to train models using RECIST (Response Evaluation
Criteria in Solid Tumors) annotations [9, 10, 11, 12].
Despite recent advancements, current methods still ex-
hibit significant challenges. One primary issue is the insuf-
ficient detail provided by weak annotations, especially con-
cerning the location and shape of lesions. The training pro-
cess is frequently misled by this limitation, and the diverse
shapes and textures of nodules in ultrasound images further
intensify this issue. Furthermore, while weak annotations are
more cost-effective, they still require medical professionals
to manually annotate the training set. Many existing meth-
ods depend on unsupervised segmentation algorithms, such
as GrabCut [13], to generate pseudo-labels. Often, these ap-
proaches [11, 12] necessitate the use of custom trimaps and
arXiv:2407.13553v1  [cs.CV]  18 Jul 2024

[CAPTION] Fig. 1. Examples for aspect ratio annotation and prediction


<!-- page 2 -->
iterative updates, leading to unstable performance and ex-
tended computational times, particularly with low-resolution
ultrasound images. Some researchers [9, 10] have tried to di-
rectly construct pseudo-labels from weak annotations. How-
ever, these static pseudo-labels are not adaptive enough to
cater to the diverse shapes of lesions.
Distinct from other forms of sparse annotation, aspect ra-
tio annotations (clinical annotations) from clinical ultrasound
provide valuable nodule shape and location information (see
Fig. 1). Readily accessible in hospital PACS, they offer a con-
venient source of weak annotations for nodule segmentation.
The main challenge is to devise a method that can use these
annotations to generate useful pseudo-labels for training mod-
els efficiently. In recent developments, foundational models
like SAM [14], trained on extensive datasets, aim to reduce re-
liance on detailed manual annotations for image segmentation
tasks. While segmentation foundational models excel with
natural images, they struggle with medical images, particu-
larly those with low contrast like ultrasound images. Using
bounding boxes as prompts can improve segmentation per-
formance, but results heavily depend on the bounding boxes’
precision.
In this study, prompted by the identified challenges, we
introduce a novel framework for automated nodule segmen-
tation utilizing aspect ratio annotations.
Specifically, we
leverage SAM for pseudo-label generation. However, there
is a particular concern that bounding boxes generated di-
rectly from clinical annotations may not fully encompass
nodules, especially since nodules often have a convex shape
[15] (see Fig. 1). To overcome the limitations of bounding
boxes, we propose a Box Prompts Generation (BPG) mod-
ule that creates bounding box prompts with clinical shape
priors.
These prompts are then used in conjunction with
SAM to generate pseudo-labels. Subsequently, we designed
a pseudo-label selection module that conducts pixel-level
selection from the three potential pseudo-labels produced
by SAM, resulting in two final pseudo-labels. Additionally,
we have developed an SAM-driven uncertainty estimation
method that utilizes the inconsistencies in SAM’s outputs for
the BPG-generated bounding boxes. Ultimately, we employ a
SAM-driven Uncertainty-Aware Cross Teaching strategy for
model training. This involves two models trained separately
with the two final pseudo-labels engaging in cross-supervised
training under the guidance of uncertainty maps.
Exten-
sive experiments on two clinical ultrasound datasets, thyroid
and breast, reveal that our framework outperforms existing
approaches in clinical label settings.
2. METHOD
2.1. Overview
The overall framework of our proposed method is illustrated
in Figure 2, comprising two primary steps. Initially, the Box
Prompts Generation module is employed to create bound-
ing box prompts with shape prior, which serve to enhance
SAM’s segmentation capabilities. Subsequently, a pixel-level
pseudo-label selection module is utilized to generate two
pseudo-labels. In the second step, these pseudo-labels from
the prior stage are used to train two fully automated segmen-
tation models. We introduced a new uncertainty estimation
method and devised a SAM-driven Uncertainty-Aware Cross
Teaching strategy to achieve improved segmentation results.
2.2. Generating pseudo labels from SAM
2.2.1. Generating bounding box prompts
We define the dataset with clinical labels as D, with each la-
beled data pair represented by (X, C) ∈D, where X is the
raw image, and C is the corresponding clinical label. A pre-
cise prompt is instrumental in enhancing the segmentation
performance of the Segment Anything Model (SAM). For
each clinical label ci, we initially generate a tight bounding
box as a prompt, denoted by b1. Given that nodules often ex-
hibit a convex shape, a tight bounding box may not fully cap-
ture the lesion’s extent. To address this, we propose the gen-
eration of bounding boxes with shape priors. Specifically, we
construct an approximate ellipse by connecting curved lines
between each pair of clinically annotated adjacent endpoints
and derive a minimum enclosing circle based on the clinical
annotations. Subsequently, we generate two bounding boxes
based on the approximate ellipse and the minimum enclosing
circle, denoted as b2 and b3, respectively.
2.2.2. Pseudo label selection
For the i-th image in our dataset, three bounding box prompts
have been generated, denoted as Bi =
 
b1
i , b2
i , b3
i
	
. Subse-
quently, each bounding box prompt is used to guide SAM in
generating a corresponding pseudo-label. This process yields
three pseudo-labels, expressed as m1
i , m2
i and m3
i . Under the
guidance of shape-prior bounding box prompts, we derive the
minimum and maximum predicted masks from SAM’s out-
puts by calculating the intersection and union of m1
i , m2
i and
m3
i . Formally, we define the minimum predicted mask as
Yint
i
and the maximum predicted mask as Yuni
i
. These can
be computed respectively as follows:
Yint
i
= m1
i ∩m2
i ∩m3
i ,
Yuni
i
= m1
i ∪m2
i ∪m3
i ,
(1)
where ∩denotes the intersection operation and ∪signifies the
union operation.
2.3. Learning from pseudo labels
2.3.1. Uncertainty Estimation
SAM’s performance is sensitive to the position and size of
bounding box. Despite the incorporation of shape priors into


<!-- page 3 -->
Model
Prediction 
Model
Prediction
Input Image
ȢǠ
Ȣǡ
ȢǠ(ȴȿ)
Ȣǡ(ȴȿ)
ȴȿ
Foundation
filter
Pseudo  Label
 
Ȩ풔ɋɆ
Ȩ풔ɋɆ
Intersection   
 Union
   Mask   
  Mask
  Map
 Uncertainty
 Clinical Label
࡯ȿ
ȸȿ
Ǡ
ȸȿ
ǡ
ȸȿ
Ǣ
Ȩ풄Ɋ−ɋ
Ƀȿ
Ǡ
Ƀȿ
ǡ
Ƀȿ
Ǣ
    Box  Prompts       
        
Step1 : Generating pseudo labels 
from SAM with Box prompts
Step2 : Nodule Segmentation via 
SAM-driven Uncertainty-Aware 
Cross Teaching
ȵȿ
ȿɄɊ
ȵȿ
ɋɄȿ
     Generation      
        
    Model
Selection
 
The XOR operator
Multiplication operator
Fig. 2. Overview of our proposed framework. The framework includes two stages. In the initial stage, three distinct sets of
prior bounding boxes are created based on clinical annotations and fed as prompts into the SAM, generating pseudo labels. In
the subsequent stage, a selection is made from these pseudo labels. A strategy named SAM-driven Uncertainty-Aware Cross
Teaching is then introduced, leveraging these selected pseudo labels for nodule segmentation.
our bounding box generation, ensuring absolute precision re-
mains a challenge. To address this, we propose a novel SAM-
driven Uncertainty Estimation approach to inform and en-
hance the model training process. Specifically, regions where
predictions m1
i , m2
i and m3
i concur are considered certain,
while discrepancies among these predictions denote areas of
uncertainty. The uncertainty map of the i-th image, denoted
as Ui, can be computed as follows:
Ui = Yint
i
⊕Yuni
i
,
(2)
where ⊕represents the pixel-wise XOR operator to indicate
the inconsistency region between Yint
i
and Yuni
i
.
2.3.2. SAM-driven Uncertainty-Aware Cross Teaching
The proposed methodology incorporates a dual-model archi-
tecture for segmentation, where F1 and F2 represent the two
segmentation models.
Perturbations in label-space are in-
troduced to enhance model robustness. Specifically, F1 is
trained using the intersection of pseudo-labels Yint
i
, while F2
is trained using the union of pseudo-labels Yuni
i
. The super-
vised loss function is a composite of two commonly employed
loss functions in segmentation tasks, which are defined as fol-
lows:
Lsup =Lce(F1(Xi), Yint
i
) + LDice(F1(Xi), Yint
i
)
+Lce(F2(Xi), Yuni
i
) + LDice(F2(Xi), Yuni
i
).
(3)
Owing to the distinct output-level properties resulting from
label-space perturbations, we introduce the cross teaching be-
tween F1 and F2. Specifically, the prediction of a network is
used as the pseudo label to supervise the other network. The
pseudo labels can be computed as follows:
Ypl1
i
= argmax(F1(Xi)),
Ypl2
i
= argmax(F2(Xi)).
(4)
Given that label-space perturbations originate from SAM, we
incorporate the SAM-based uncertainty estimation into the
cross teaching process. Specifically, cross teaching is con-
ducted solely within regions of uncertainty. Then, the SAM-
driven cross teaching loss is defined as:
Lct−u = Lce(F1(Xi), Ypl2
i
, Ui)
+Lce(F2(Xi), Ypl1
i
, Ui),
(5)
where Lce is cross entropy loss function. Finally, the overall
objective function Ltotal can be defined as:
Ltotal = Lsup + λLct−u.
(6)
where λ is the weight factor. The determination of λ was
based on experimentation.
3. EXPERIMENT AND RESULTS
3.1. Datasets
We conducted the evaluation of the proposed method on two
clinical ultrasound datasets for thyroid and breast nodules.
The Thyroid Ultrasound Dataset consists of 422 left and 422
right thyroid ultrasound images, and the Breast Ultrasound
Dataset comprises 604 left and 151 right breast ultrasound
images. All the nodules were manually segmented by two


**[Table p3.1]**
| ȵȿȿɄɊ Intersection Mask |  |
| --- | --- |
|  |  |

[CAPTION] Fig. 2. Overview of our proposed framework. The framework includes two stages. In the initial stage, three distinct sets of


<!-- page 4 -->
radiologists, each with over a decade of experience. For fur-
ther validation, these initial annotations were also reviewed
by a third radiologist with over twenty years of experience.
The dataset was randomly split into training and test subsets,
following a ratio of approximately 4:1. During the training
phase, we ensured that each image in the training set was ac-
companied by its respective clinical aspect ratio annotation.
Table 1. Comparison results of different methods on the two
test datasets.
Method
Thyroid Ultrasound
Breast Ultrasound
DSC(%)
HD95(pixel)
DSC(%)
HD95(pixel)
EM [1]
71.7±23.6
35.6±68.4
72.4±21.0
62.5±43.3
TV [2]
70.7±23.5
53.4±95.8
73.4±21.0
59.3±43.0
Mumford-Shah [3] 71.7±22.6
36.3±71.4
72.3±24.1
57.7±45.0
GatedCRF [16]
68.6±25.2
39.1±74.6
73.2±20.3
57.7±39.6
WSSS [12]
73.3±21.3
37.2±67.1
74.7±19.5
59.8±41.9
RECISTSup [11]
72.5±21.9
37.5±72.9
72.5±20.8
61.6±41.1
CoTraining [9]
73.4±23.7
39.2±74.0
68.3±22.0
58.9±37.5
CoTeaching [17]
71.2±24.0
38.7±72.5
72.7±21.8
53.4±37.3
TriNet [18]
70.8±23.9
35.9±72.4
69.5±22.6
58.2±38.8
MTCL [19]
70.2±25.0
40.1±76.3
70.5±22.1
60.1±38.6
Ours
73.5±22.2
32.8±71.5
75.5±19.8
52.3±36.5
GrabCut [13]
68.6±25.3
48.4±85.2
71±23.0
59.2±42.7
SAM [14]
72.1±23.5
42.0±78.7
74.6±20.0
59.2±42.4
Fully Supervised
74.7±23.8
35.3±75.3
76.4±20.6
53.2±43.2
3.2. Implementation Details
All compared methods were implemented in PyTorch [20]
utilizing an NVIDIA RTX 3090 GPU. To maintain consis-
tency and ensure fairness in our comparisons, we adopted the
UNet [21] as the common backbone network for all the meth-
ods involved in our experiments. All images were uniformly
resized to dimensions of 256 × 256 pixels and each exper-
iment underwent an identical data augmentation process in-
cluding random rotation and flipping. The batch size was set
to 24. The learning rate was initially set at 0.01, and the poly
learning rate strategy [22] was employed for adaptive fine-
tuning of the learning rate. We evaluated performance us-
ing the Dice similarity coefficient (DSC) and 95th percentile
Hausdorff Distance (HD95).
3.3. Results
Comparison With State-of-the-Arts. To assess the perfor-
mance of our framework, we compared it with several state-
of-the-art weakly supervised approaches including GrabCut
only [13], Entropy Minimization (EM) [1], Total Variation
(TV) loss [2], Mumford-Shah loss [3], GatedCRF Loss [16],
WSSS [12], RECISTSup [11] and CoTraining [9]. Addition-
ally, as the pseudo-labels may contain noise, we evaluated
our method alongside other popular approaches for learning
from noisy annotations: 1) CoTeaching [17], 2) TriNet [18],
3) MTCL [19]. All compared methods, except for CoTraining
[9], use GrabCut to generate pseudo-labels for training. The
results of these methods are tabulated in Table 1.
Table 2. Ablation studies of our proposed framework. We
compared our proposed method with different dual model su-
pervision strategies on the Thyroid Ultrasound dataset.
Method
DSC
HD95
Fsingle
72.1±23.5
42.0±78.7
Dual Model+CR [9, 23]
72.1±24.1
38.9±71.5
Dual Model+CPS [24]
71.7±24.7
42.2±80.6
Dual Model+DMPLS [5]
73.1±23.0
37.9±73.2
UAMT [25]
71.7±24.2
40.9±81.9
Ours
73.5±22.2
32.8±71.5
Ablation Study. Given the utilization of two networks in our
framework, we compared the effects of various dual-model
supervision methods with our proposed method. These ap-
proaches include: 1) Baseline (single model Fsingle), 2) Con-
sistency Regularization (CR) [9, 23], 3) Cross Pseudo Super-
vision (CPS) [24], 4) Dynamically Mixed Pseudo Labels Su-
pervision (DMPLS) [5], 5) SAM-driven Uncertainty-Aware
Cross Teaching (Ours). Additionally, we also compared our
SAM-driven uncertainty estimation with traditional method
using Monte Carlo simulation (UAMT) [25]. The quantita-
tive evaluation results are presented in Table 2.
Hyper-parameters Experiments Results.
The proposed
framework has a hyper-parameter λ. we conducted exper-
iments to investigate the impact of different values of the
hyper-parameter λ. Specifically, we evaluated our method
across different settings of λ ∈{0.1, 0.3, 0.5, 1}. Addition-
ally, we also explored a dynamic approach to setting λ by
employing a Gaussian warming-up function dependent on
time [25]. As shown in Fig. 3, the best performance was
achieved when λ was set to 0.1.
62.0
64.0
66.0
68.0
70.0
72.0
74.0
76.0
0.1
0.3
0.5
1
Ramp-up
DSC
𝜆
0.0
10.0
20.0
30.0
40.0
50.0
60.0
0.1
0.3
0.5
1
Ramp-up
Hd95
𝜆
Fig. 3. The impact of varying λ on the segmentation perfor-
mance of the proposed framework, assessed on the Thyroid
Ultrasound dataset using Dice similarity coefficient (DSC)
and 95th percentile Hausdorff Distance (HD95).
4. CONCLUSION
In this paper, we propose a novel weakly supervised frame-
work for nodule segmentation using clinical annotations,
which effectively employs the zero-shot segmentation capa-
bilities of the foundational segmentation model with a shape-
prior box prompts generation module and a SAM-driven
uncertainty-aware cross-teaching strategy. Extensive exper-
iments conducted demonstrate that our framework achieves
state-of-the-art performance.


**[Table p4.1]**
| Method | DSC HD95 |
| --- | --- |
| F single Dual Model+CR [9, 23] Dual Model+CPS [24] Dual Model+DMPLS [5] UAMT [25] | 72.1±23.5 42.0±78.7 72.1±24.1 38.9±71.5 71.7±24.7 42.2±80.6 73.1±23.0 37.9±73.2 71.7±24.2 40.9±81.9 |
| Ours | 73.5±22.2 32.8±71.5 |

[CAPTION] Table 1. Comparison results of different methods on the two

[CAPTION] Table 2. Ablation studies of our proposed framework. We

[CAPTION] Fig. 3. The impact of varying λ on the segmentation perfor-


<!-- page 5 -->
5. ACKNOWLEDGMENTS
This work is supported by the Key Research and Development
Program of Shaanxi Province under Grant. 2021GXLH-Z-
097.
6. REFERENCES
[1] Yves Grandvalet and Yoshua Bengio, “Semi-supervised
learning by entropy minimization,” NeurIPS, vol. 17,
2004.
[2] Mehran Javanmardi et al., “Unsupervised total varia-
tion loss for semi-supervised deep learning of semantic
segmentation,” arXiv preprint arXiv:1605.01368, 2016.
[3] Boah Kim and Jong Chul Ye,
“Mumford–shah loss
functional for image segmentation with deep learning,”
IEEE Transactions on Image Processing, vol. 29, pp.
1856–1866, 2019.
[4] George Papandreou et al.,
“Weakly-and semi-
supervised learning of a deep convolutional network
for semantic image segmentation,” in ICCV, 2015, pp.
1742–1750.
[5] Xiangde Luo et al., “Scribble-supervised medical image
segmentation via dual-branch network and dynamically
mixed pseudo labels supervision,” in MICCAI. Springer,
2022, pp. 528–538.
[6] Di Lin et al., “Scribblesup: Scribble-supervised convo-
lutional networks for semantic segmentation,” in CVPR,
2016, pp. 3159–3167.
[7] Jifeng Dai et al., “Boxsup: Exploiting bounding boxes
to supervise convolutional networks for semantic seg-
mentation,” in ICCV, 2015, pp. 1635–1643.
[8] Shuwei Zhai et al., “Pa-seg: Learning from point an-
notations for 3d medical image segmentation using con-
textual regularization and cross knowledge distillation,”
IEEE Transactions on Medical Imaging, 2023.
[9] Lianyu Zhou et al.,
“Recist weakly supervised le-
sion segmentation via label-space co-training,”
arXiv
preprint arXiv:2303.00205, 2023.
[10] Lianyu Zhou et al., “Recist-induced reliable learning:
Geometry-driven label propagation for universal lesion
segmentation,” IEEE Transactions on Medical Imaging,
2023.
[11] Han Wang et al., “Recistsup: Weakly-supervised lesion
volume segmentation using recist measurement,” IEEE
Transactions on Medical Imaging, vol. 41, no. 7, pp.
1849–1861, 2022.
[12] Jinzheng Cai et al., “Accurate weakly-supervised deep
lesion segmentation using large-scale clinical annota-
tions: Slice-propagated 3d mask generation from 2d re-
cist,” in MICCAI. Springer, 2018, pp. 396–404.
[13] Carsten Rother et al., “” grabcut” interactive foreground
extraction using iterated graph cuts,” ACM transactions
on graphics (TOG), vol. 23, no. 3, pp. 309–314, 2004.
[14] Alexander Kirillov et al., “Segment anything,” arXiv
preprint arXiv:2304.02643, 2023.
[15] Erik K Alexander et al.,
“Thyroid nodule shape and
prediction of malignancy,” Thyroid, vol. 14, no. 11, pp.
953–958, 2004.
[16] Anton Obukhov et al., “Gated crf loss for weakly su-
pervised semantic image segmentation,” arXiv preprint
arXiv:1906.04651, 2019.
[17] Bo Han et al., “Co-teaching: Robust training of deep
neural networks with extremely noisy labels,” NeurIPS,
vol. 31, 2018.
[18] Tianwei Zhang et al., “Robust medical image segmen-
tation from non-expert annotations with tri-network,” in
MICCAI. Springer, 2020, pp. 249–258.
[19] Zhe Xu et al.,
“Anti-interference from noisy labels:
Mean-teacher-assisted confident learning for medical
image segmentation,”
IEEE Transactions on Medical
Imaging, vol. 41, no. 11, pp. 3062–3073, 2022.
[20] Adam Paszke et al.,
“Pytorch: An imperative style,
high-performance deep learning library,” Advances in
neural information processing systems, vol. 32, 2019.
[21] Olaf Ronneberger et al.,
“U-Net: Convolutional net-
works for biomedical image segmentation,” in MICCAI,
Cham, Switzerland, 2015, Springer, pp. 234–241.
[22] Xiangde Luo et al., “Efficient semi-supervised gross tar-
get volume of nasopharyngeal carcinoma segmentation
via uncertainty rectified pyramid consistency,” in MIC-
CAI. Springer, 2021, pp. 318–329.
[23] Jose Dolz et al.,
“Teach me to segment with mixed
supervision: Confident students become masters,”
in
IPMI. Springer, 2021, pp. 517–529.
[24] Xiaokang Chen et al., “Semi-supervised semantic seg-
mentation with cross pseudo supervision,”
in CVPR,
2021, pp. 2613–2622.
[25] Lequan Yu et al., “Uncertainty-aware self-ensembling
model for semi-supervised 3d left atrium segmentation,”
in MICCAI. Springer, 2019, pp. 605–613.