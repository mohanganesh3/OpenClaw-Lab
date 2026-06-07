<!-- page 1 -->
EVIL: EVIDENTIAL INFERENCE LEARNING FOR TRUSTWORTHY SEMI-SUPERVISED
MEDICAL IMAGE SEGMENTATION
Yingyu Chen1
Ziyuan Yang1
Chenyu Shen1
Zhiwen Wang1
Yang Qin1
Yi Zhang2,1, Senior Member, IEEE
1College of Computer Science, Sichuan University, Chengdu, China
2 School of Cyber Science and Engineering, Sichuan University, Chengdu, China
ABSTRACT
Recently, uncertainty-aware methods have attracted in-
creasing attention in semi-supervised medical image seg-
mentation.
However, current methods usually suffer from
the drawback that it is difficult to balance the computational
cost, estimation accuracy, and theoretical support in a uni-
fied framework.
To alleviate this problem, we introduce
the Dempster–Shafer Theory of Evidence (DST) into semi-
supervised medical image segmentation, dubbed EVidential
Inference Learning (EVIL). EVIL provides a theoretically
guaranteed solution to infer accurate uncertainty quantifica-
tion in a single forward pass. Trustworthy pseudo labels on
unlabeled data are generated after uncertainty estimation. The
recently proposed consistency regularization-based training
paradigm is adopted in our framework, which enforces the
consistency on the perturbed predictions to enhance the gen-
eralization with few labeled data. Experimental results show
that EVIL achieves competitive performance in comparison
with several state-of-the-art methods on the public dataset.
Index Terms— Medical Image Segmentation, Semi-
Supervised Learning, Evidential Learning
1. INTRODUCTION
Medical image segmentation plays an essential role in sub-
sequent clinical or computer-aided diagnosis and fully-
supervised learning has achieved great success in the field
of automatic image segmentation [1]. However, annotating
medical images is laborious and requires rich professional
knowledge [2].
Semi-supervised learning (SSL) has shown great poten-
tial to alleviate this problem by leveraging a large set of un-
labeled data accompanied with a limited number of labeled
data. These methods can be roughly categorized into two
types: (1) pseudo-label retraining, which incorporates pseudo
labels on unlabeled data for retraining [3, 4, 5]; and (2) con-
sistency regularization, which enforces the prediction consis-
tency to enhance generalization with various perturbations,
Yi Zhang is the corresponding author
such as input perturbation, feature perturbation, and network
perturbation [6, 7, 8].
However, since these methods rely heavily on the predic-
tion of pseudo label, false predictions will severely degrade
the segmentation performance.
To improve the quality of
pseudo labels, some uncertainty-aware methods have been
proposed, including Monte Carlo dropout (MC-dropout)-
based [9], Information-Entropy-based [10], and Prediction
Variance-based [11] methods. However, these methods suffer
from some problems: (1) Although MC-dropout is mathe-
matically guaranteed by Bayesian theory, its training process
is costly due to the multiple sampling operations; (2) Due
to the limited sampling times, MC-dropout can’t obtain ac-
curate uncertainty quantification; (3) Other two uncertainty
estimation methods have advantages in computational cost,
but they lack theoretical support, leading to unstable pseudo
label generation.
To handle the above issues, we introduce the Demp-
ster–Shafer Theory of Evidence (DST) into semi-supervised
medical image segmentation, providing a theoretically guar-
anteed single-pass solution for uncertainty quantification
inference, dubbed EVidential Inference Learning (EVIL).
Following the training paradigm proposed in [7], EVIL be-
longs to the consistency regularization method with network
perturbation, which imposes the prediction consistency on
two networks perturbed with different initialization. In par-
ticular, the two networks play different roles. One is a vanilla
segmentation network (S-Net) which directly generates the
segmentation result.
The other network called evidential-
network (E-Net) is built from the perspective of DST, which
is theoretically guaranteed for reliable predictions. Different
from S-Net, the output of E-Net is regarded as the evidence
and parameterized into a Dirichlet distribution on segmen-
tation probabilities.
Subjective Logic theory (SL) [12] is
employed to quantify the predictions and uncertainties of
different categories with the Dirichlet distribution in a sin-
gle inference, which significantly reduces the training time.
Then, the trustworthy pseudo labels on unlabeled data are
generated. In summary, there are three merits for our pro-
posed EVIL: lower computation cost due to the single-pass
arXiv:2307.08988v1  [cs.CV]  18 Jul 2023


<!-- page 2 -->
𝑥𝑥
E-Net  ℱ1
𝛼𝛼𝑘𝑘𝑘𝑘=1
𝐾𝐾
Dirichlet
Distribution (𝛼𝛼)
𝑃𝑃1
𝑌𝑌1
⨀
𝑃𝑃2
𝑌𝑌2
S-Net  ℱ2
Fig. 1. The overview of our EVidential Inference Learning
framework (EVIL), where M denotes uncertainty map esti-
mated by E-Net and ⊙denotes element-wise product. ‘→’
presents forward operation, ‘99K’ presents supervision loss
operation and ‘//’ on ‘→’ presents stop-gradient.
operation, accurate uncertainty estimation based on SL and
theoretical guarantee based on DST.
The main contributions of this work are summarized as:
1) we introduce DST into SSL and provide a fast accurate
uncertainty estimation with theoretical guarantee in a unified
framework; 2) a novel network perturbation strategy is pro-
posed, which allows different initialized network optimized
with different objectives; and 3) extensive experiments are
conducted to validate the effectiveness of our proposed EVIL.
2. METHOD
Given a labeled set Dl = {(xi, yi)}Nl
i=1 with Nl samples and
an unlabeled set Du = {xi}Nu
i=1 with Nu samples, where
Nu ≫Nl in semi-supervised task.
As illustrated in Fig. 1, EVIL has two differently initial-
ized networks, E-Net F1 with parameter set θ1 and S-Net F2
with parameter set θ2. For labeled data, S-Net is optimized
with traditional joint cross-entropy loss and dice loss, while
E-Net models a Dirichlet distribution and is optimized with
evidential segmentation loss. P1, P2 are the segmentation
predictions and Y1, Y2 are the corresponding pseudo labels
generated by argmax function. For unlabeled data, E-Net
generates pseudo labels Y1 and accurate uncertainty estima-
tions M simultaneously. Then, the trustworthy pseudo labels
are calculated by M ⊙Y1 and used to guide the training of
S-Net. Reversely, the pseudo labels Y2 generated by S-Net
is leveraged for E-Net to explore more potential evidence to
improve the generation of pseudo labels from unlabeled data.
2.1. Uncertainty Modeling
In this section, we utilize DST to model the segmentation un-
certainty and generate trustworthy prediction. For a K-class
segmentation task, given an input xi, the evidence vector ei
is obtained with a transform function g, which is defined in
[13]:
ei = g(F1(xi)) = exp(tanh F1(xi)/τ),
(1)
𝑏𝑏𝑖𝑖
1
…
𝑢𝑢𝑖𝑖
Belief mass 𝑏𝑏𝑖𝑖
𝑘𝑘
𝑘𝑘=1
𝐾𝐾
Uncertainty 𝑢𝑢𝑖𝑖
𝑏𝑏𝑖𝑖
2
𝑏𝑏𝑖𝑖
𝐾𝐾
Fig. 2. Subjective Logic model, where ui + PK
k=1 bk
i = 1.
where 0 < τ < 1 is a scaling parameter set to 1/K. F1(xi)
is the output of E-Net with input xi. Subjective Logic [12]
computes the belief mass for category k and uncertainty as:
bk
i = ek
i
Si
= αk
i −1
Si
and
ui = K
Si
,
(2)
where Si = PK
k=1(ek
i + 1), ui + PK
k=1 bk
i = 1 and αk
i =
ek
i +1. αi =
 
α1
i , . . . , αK
i
 
can be regarded as the parameters
of Dirichlet distribution, which models the density of segmen-
tation probability and uncertainty [14]. The density function
is defined as:
D(pi | αi) = {
1
B(αi)
QK
k=1 pαk
i −1
i
for pi ∈SK,
0
otherwise,
(3)
where pi is the segmentation probability, B(αi) is the K-
dimensional multinomial beta function for parameter αi, and
SK is the K-dimensional simplex.
2.2. Evidential Net (E-Net)
We follow [14] and use cross-entropy loss to make the seg-
mentation probabilities pi approach the ground-truth yi. No-
tably, the density of pi follows the Dirichlet distribution pa-
rameterized with αi. The loss can be formulated as:
Ldig =
Z " K
X
k=1
−yk
i log
 pk
i
 
#
D(pi | αi)dpi
=
K
X
k=1
yk
i
 ψ (Si) −ψ
 αk
i
  
,
(4)
where ψ(·) is the digamma function. By optimizing Ldig, the
evidence of different classes for positive samples is generated.
However, Ldig cannot guarantee that negative samples gener-
ate evidence as close as zero. Therefore, Kullback-Leibler
(KL) divergence is incorporated into our loss function to pe-
nalize the divergence from negative samples, which is defined
as:
LKL = KL [D (pi | ˜αi) ∥D (pi | 1)]
= log


Γ
 PK
k=1 eαk
i
 
Γ(K) PK
k=1 Γ
 eαk
i
 


+
K
X
k=1
 eαk
i −1
 
"
ψ
 eαk
i
 
−ψ
 K
X
k=1
eαk
i
!#
,
(5)
where Γ(·) is the gamma function, D(pi|1) is the uniform
Dirichlet distribution, and ˜αi = yi + (1 −yi) ⊙αi.


**[Table p2.1]**
|  |  |
| --- | --- |
| E-Net ℱ 𝛼 1 D Distr 𝑥𝑥 S-Net ℱ 𝑃𝑃 2 Fig. 1. The overview of our framework (EVIL), where M mated by E-Net and ⊙ deno | 𝛼 𝐾𝐾 𝑃𝑃 𝑌𝑌 Belief m 𝑘𝑘 𝑘𝑘=1 1 1 irichlet 𝑏𝑏1 𝑖𝑖 ibution (𝛼𝛼) ⨀ Fig. 2. Subject where 0 < τ < 𝑌𝑌 2 2 is the output of computes the be bk = EVidential Inference Learning i denotes uncertainty map esti- tes element-wise product. ‘→’ where S i = P |
| presents forward operation, operation and ‘//’ on ‘→’ pr operation, accurate uncertain theoretical guarantee based o The main contributions o 1) we introduce DST into S uncertainty estimation with t framework; 2) a novel netwo posed, which allows differen with different objectives; an conducted to validate the effe |  |


**[Table p2.2]**
|  | 𝛼 | 𝛼 𝐾𝐾 𝑘𝑘 𝑘𝑘=1 |
| --- | --- | --- |
| D Distr |  | irichlet ibution (𝛼𝛼) |


**[Table p2.3]**
|  | Belief m |  | ass 𝑏𝑏𝑘𝑘 𝐾𝐾 𝑖𝑖 𝑘𝑘=1 |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 𝑏𝑏1 𝑖𝑖 |  |  | 𝑏𝑏2 𝑖𝑖 | … | 𝑏𝑏𝐾𝐾 𝑖𝑖 |  |  | 𝑢𝑢 𝑖𝑖 |


**[Table p2.4]**
| 𝑃𝑃 | 2 |
| --- | --- |

[CAPTION] Fig. 1. The overview of our EVidential Inference Learning

[CAPTION] Fig. 2. Subjective Logic model, where ui + PK


<!-- page 3 -->
(a) image & GT
(b) MT
(c) UAMT
(d) ICT
(e) CPS
(f) URPC
(g) EVIL (ours)
Fig. 3. Visual comparison of segmentation results with different methods with 10% labeled images.
For segmentation task, the evidence ei is obtained with
xi. Then, αi = ei+1 is parameterized into the corresponding
Dirichlet distribution and the evidential loss is:
Levi = Ldig + βLKL,
(6)
where β is a annealing coefficient and is set to β(t) =
min(1.0,
t
0.5tmax ). t is the current epoch and tmax is the
total number of training epochs.
As shown in Fig. 2, the Subjective Logic model has two
parts, the certain part called belief mass bi and the uncertain
part ui. The evidential loss generates evidence to reduce the
uncertainty. However, since the cross-entropy based eviden-
tial loss is based on pixel level, which ignores the relation-
ships between pixels in segmentation task, we use the Dice
loss on the certain part and the certain loss is defined as:
Lcertain = 1 −2 PK
k=1 yk
i
PK
k=1 ˆpk
i
PK
k=1 yk
i + PK
k=1 ˆpk
i
,
(7)
where ˆpi = softmax(bi) presents a simplex transformed
from the belief mass bi with a softmax function.
Then, our overall evidential segmentation loss is defined:
LEseg = Levi + γLcertain,
(8)
where Levi and Lcertain denote the evidential loss and the
certain loss, respectively. γ denotes the weighting parame-
ter, which is set to 1. By optimizing Levi, E-Net generates
the evidence for positive samples, while reduces the evidence
for negative samples. Lcertain is leveraged to constrain the
relationship between different predicted pixels.
2.3. EVIL Framework
The total loss L for our whole framework contains two com-
ponents: supervised loss Lsup on labeled data and consis-
tency loss Lcon on unlabeled data:
L = Lsup + λLcon,
(9)
where λ is the balancing parameter. We use Gaussian ramp-
up function λ(t) = λmax ∗e−5(1.0−
t
tmax )
2
and λmax = 0.1.
The supervision loss is formulated as:
Lsup = LEseg(F1(x), y) + LSseg(F2(x), y),
(10)
where LSseg = 1
2(Lce + Ldice) denotes the loss component
for S-Net. Lce and Ldice are the cross-entropy loss and dice
loss, respectively.
The pseudo label can be calculated as Y1 = argmax(bi)
for E-Net and Y2 = argmax(F2(x)) for S-Net. The consis-
tency loss on the unlabeled data is written as:
Lcon = Levi(F1(x), Y2) + Lce(F2(x), M ⊙Y1).
(11)
where M = u < T is the mask to filter out high uncertain
results with threshold T = 0.2. We only use the evidential
and cross-entropy losses in consistency loss term due to the
mask operation which preserves only the reliable pseudo pixel
labels. The consistency loss encourages E-Net to generate
potential evidence from S-Net using Levi and S-Net to learn
the reliable pseudo labels using Lce from E-Net.
3. EXPERIMENT
3.1. Experiment Setup
We evaluate our method on the Automated Cardiac Diagno-
sis Challenge (ACDC) [15] dataset which contains 200 anno-
tated short-axis cardiac MR-cine images from 100 patients.
We leverage 70 patients (140 scans) for training, 10 patients
(20 scans) for validation and 20 patients (40 scans) for test-
ing. All short-axis slices within 3D scans are resized to 256
× 256 as 2D images. See SSL4MIS 1 for more details. For
semi-supervised experiments, images from 7 patients, 14 pa-
tients and 21 patients are set as labeled ratio 10%, 20% and
30% in the training set, respectively. Standard data augmen-
tation, including random cropping, random rotating, and ran-
dom flipping, is used to enlarge the training set. Three widely
1https://github.com/HiLab-git/SSL4MIS

[CAPTION] Fig. 3. Visual comparison of segmentation results with different methods with 10% labeled images.


<!-- page 4 -->
Table 1. The comparison of different methods on ACDC dataset on different semi-supervised labeled data ratio settings.
Method
10%
20%
30%
DSC ↑
HD95 ↓
ASD ↓
DSC ↑
HD95 ↓
ASD ↓
DSC ↑
HD95 ↓
ASD ↓
Unet
80.05
7.41
2.38
84.90
8.94
2.52
87.07
6.61
1.95
E-Net (ours)
81.05
11.17
3.26
85.68
7.39
2.12
87.45
8.12
2.23
MT
81.06
10.17
2.64
86.01
8.13
2.40
87.37
4.81
1.49
UA-MT
80.81
11.73
3.52
85.38
7.77
2.70
87.53
6.32
2.05
ICT
83.54
8.42
2.46
85.28
5.65
1.64
87.49
8.25
2.23
CPS
84.70
8.25
2.35
87.47
5.98
1.74
88.21
6.49
1.90
URPC
82.07
5.62
1.88
85.13
5.71
1.75
86.99
4.43
1.31
EVIL (ours)
85.91
3.91
1.36
88.22
4.01
1.21
89.43
3.84
1.07
Table 2. The comparison of training time.
Method
Num
Uncertainty
Time
Cost
Unet
1
×
0.076 s
-
ICT
1
×
0.090 s
+ 18.42 %
URPC
1
√
0.089 s
+ 17.11 %
E-Net (ours)
1
√
0.085 s
+ 11.84 %
MT
2
×
0.101 s
-
CPS
2
×
0.137 s
+ 35.64 %
UA-MT
2
√
0.337 s
+ 233.66%
EVIL (ours)
2
√
0.148 s
+ 46.53 %
used metrics, Dice Coefficient (DSC), Hausdorff Distance 95
(HD95) and Average Surface Distance (ASD) are employed
to evaluate the performance of our method.
For the sake of fairness, Unet [1] is chosen as the back-
bone in all methods, and SGD is adopted as the optimizer.
The initial learning rate is set to 0.01, and polynomial sched-
uler strategy is employed to update the learning rate. We im-
plement the proposed framework with PyTorch, using a single
NVIDIA GTX 1080Ti GPU. The batch size is set to 24, where
12 images are labeled. All methods perform 30000 iterations
during training.
3.2. Experimental Results
Several recently proposed semi-supervised segmentation
methods are compared, including: Mean-Teacher (MT) [6],
Uncertainty-Aware Mean Teacher (UA-MT) [9], Interpola-
tion Consistency Training (ICT) [16], Cross Pseudo Super-
vision (CPS) [7], and Uncertainty Rectified Pyramid Con-
sistency (URPC) [17]. For all competing methods, official
parameter settings are adopted.
Tab. 1 illustrates the quantitative results on ACDC. The
first and second rows list the quantitative results of super-
vised Unet and E-Net. In different labeled data ratio settings,
EVIL outperforms all the other methods. When only 10% of
data are labeled, our method improves DSC by more than
3% compared with other SOTA uncertainty-aware methods
(UAMT and URPC). Moreover, we achieve 4 points improve-
ment in HD95 and 1 point in ASD compared with CPS. Es-
pecially, we can see that the performance of EVIL using 20%
labeled data has surpassed all compared methods using 30%
labeled data.
(a) Input
(b) Label
(c) Ours
(d) EVIL
(e) S=2
(f) S=8
(g) S=64
(h) UA-MT
Fig. 4. Visualization of uncertainty estimation. ‘S’ denotes
the MC-dropout sampling times.
Fig. 3 visualizes the segmentation results of two cases us-
ing different methods with 10% labeled data. It is easy to see
that the compared methods mis-classify many pixels while
EVIL obtains more accurate prediction. As shown in Fig.
4, sampling times affect the uncertainty estimation quality of
MC-dropout and our E-Net has best accurate estimation.
Tab. 2 shows the training time with fixed batch size =
24, where ‘Num’, ‘Uncertainty’, ‘Time’, ‘Cost’ denotes the
network number, uncertainty-based or not, time consuming,
and the additional time consuming cost respectively. We treat
Unet as the upper bound of single network method and MT
as the baseline of the multi-network framework since it is the
fastest method compared to others. Specially, we can see that
the proposed method improve significantly without introduc-
ing too much computation overhead.
4. CONCLUSION
In this paper, we propose a novel uncertainty-aware semi-
supervised medical image segmentation framework. The pro-
posed EVIL introduces DST into the consistency regulariza-
tion training paradigm and achieves fast accurate uncertainty
estimation with solid theoretical guarantee. Extensive experi-
ments demonstrate that EVIL achieves state-of-the-art perfor-
mance on the widely used ACDC dataset.
5. CONFLICTS OF INTEREST
The authors declare that they have no conflicts of interest.


**[Table p4.1]**
| Method | 10% DSC ↑ HD ↓ ASD ↓ 95 | 20% DSC ↑ HD ↓ ASD ↓ 95 | 30% DSC ↑ HD ↓ ASD ↓ 95 |
| --- | --- | --- | --- |
| Unet E-Net (ours) | 80.05 7.41 2.38 81.05 11.17 3.26 | 84.90 8.94 2.52 85.68 7.39 2.12 | 87.07 6.61 1.95 87.45 8.12 2.23 |
| MT UA-MT ICT CPS URPC EVIL (ours) | 81.06 10.17 2.64 80.81 11.73 3.52 83.54 8.42 2.46 84.70 8.25 2.35 82.07 5.62 1.88 85.91 3.91 1.36 | 86.01 8.13 2.40 85.38 7.77 2.70 85.28 5.65 1.64 87.47 5.98 1.74 85.13 5.71 1.75 88.22 4.01 1.21 | 87.37 4.81 1.49 87.53 6.32 2.05 87.49 8.25 2.23 88.21 6.49 1.90 86.99 4.43 1.31 89.43 3.84 1.07 |


**[Table p4.2]**
| Method | Num | Uncertainty | Time | Cost |
| --- | --- | --- | --- | --- |
| Unet ICT URPC E-Net (ours) | 1 1 1 1 | × × √ √ | 0.076 s 0.090 s 0.089 s 0.085 s | - + 18.42 % + 17.11 % + 11.84 % |
| MT CPS UA-MT EVIL (ours) | 2 2 2 2 | × × √ √ | 0.101 s 0.137 s 0.337 s 0.148 s | - + 35.64 % + 233.66% + 46.53 % |

[CAPTION] Table 1. The comparison of different methods on ACDC dataset on different semi-supervised labeled data ratio settings.

[CAPTION] Table 2. The comparison of training time.

[CAPTION] Fig. 4. Visualization of uncertainty estimation. ‘S’ denotes

[CAPTION] Fig. 3 visualizes the segmentation results of two cases us-


<!-- page 5 -->
6. COMPLIANCE WITH ETHICAL STANDARDS
This research study was conducted retrospectively using real
clinical exams acquired at the University Hospital of Dijon.
Ethical approval was not required as confirmed by the license
attached with the open access data.
7. ACKNOWLEDGEMENT
This work was supported in part by the National Natural Sci-
ence Foundation of China under Grant 62271335; in part
by the Sichuan Science and Technology Program under
Grant 2021JDJQ0024; and in part by the Sichuan Univer-
sity “From 0 to 1” Innovative Research Program under Grant
2022SCUH0016.
8. REFERENCES
[1] Olaf Ronneberger, Philipp Fischer, and Thomas Brox,
“U-net: Convolutional networks for biomedical image
segmentation,” in International Conference on Medical
Image Computing and Computer-Assisted Intervention
(MICCAI). Springer, 2015, pp. 234–241.
[2] Ke Zou, Xuedong Yuan, Xiaojing Shen, Meng Wang,
and Huazhu Fu, “Tbrats: Trusted brain tumor segmen-
tation,” in International Conference on Medical Image
Computing and Computer-Assisted Intervention (MIC-
CAI). Springer, 2022, pp. 503–513.
[3] Barret Zoph, Golnaz Ghiasi, Tsung-Yi Lin, Yin Cui,
Hanxiao Liu, Ekin Dogus Cubuk, and Quoc Le, “Re-
thinking pre-training and self-training,” in Advances in
Neural Information Processing Systems, 2020, vol. 33,
pp. 3833–3845.
[4] Zhengyang Feng, Qianyu Zhou, Guangliang Cheng,
Xin Tan, Jianping Shi, and Lizhuang Ma,
“Semi-
supervised semantic segmentation via dynamic self-
training and classbalanced curriculum,” arXiv preprint
arXiv:2004.08514, vol. 1, no. 2, pp. 5, 2020.
[5] Mostafa S Ibrahim, Arash Vahdat, Mani Ranjbar, and
William G Macready, “Semi-supervised semantic im-
age segmentation with self-correcting networks,”
in
IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), 2020, pp. 12715–12725.
[6] Antti Tarvainen and Harri Valpola, “Mean teachers are
better role models: Weight-averaged consistency targets
improve semi-supervised deep learning results,” in Ad-
vances in Neural Information Processing Systems, 2017,
vol. 30.
[7] Xiaokang Chen, Yuhui Yuan, Gang Zeng, and Jingdong
Wang,
“Semi-supervised semantic segmentation with
cross pseudo supervision,”
in IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR).
IEEE, 2021, pp. 2613–2622.
[8] Yassine Ouali, C´eline Hudelot, and Myriam Tami,
“Semi-supervised semantic segmentation with cross-
consistency training,”
in IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR),
2020, pp. 12674–12684.
[9] Lequan Yu, Shujun Wang, Xiaomeng Li, Chi-Wing
Fu, and Pheng-Ann Heng,
“Uncertainty-aware self-
ensembling model for semi-supervised 3d left atrium
segmentation,” in International Conference on Medical
Image Computing and Computer-Assisted Intervention
(MICCAI). Springer, 2019, pp. 605–613.
[10] Tao Wang, Jianglin Lu, Zhihui Lai, Jiajun Wen, and
Heng Kong,
“Uncertainty-guided pixel contrastive
learning for semi-supervised medical image segmenta-
tion,” in International Joint Conferences on Artificial
Intelligence, 2022.
[11] Zhedong Zheng and Yi Yang, “Rectifying pseudo label
learning via uncertainty estimation for domain adaptive
semantic segmentation,” International Journal of Com-
puter Vision, vol. 129, pp. 1106–1120, 2021.
[12] Audun Jsang, Subjective Logic: A formalism for rea-
soning under uncertainty, Springer, 2018.
[13] Yang Qin, Dezhong Peng, Xi Peng, Xu Wang, and Peng
Hu,
“Deep evidential learning with noisy correspon-
dence for cross-modal retrieval,” in ACM International
Conference on Multimedia, 2022, p. 4948–4956.
[14] Murat Sensoy, Lance Kaplan, and Melih Kandemir,
“Evidential deep learning to quantify classification un-
certainty,” in Advances in Neural Information Process-
ing Systems, 2018, vol. 31.
[15] Olivier Bernard, Alain Lalande, Clement Zotti, Fred-
erick Cervenansky, Xin Yang, Pheng-Ann Heng, Irem
Cetin, Karim Lekadir, Oscar Camara, Miguel An-
gel Gonzalez Ballester, et al., “Deep learning techniques
for automatic mri cardiac multi-structures segmentation
and diagnosis: is the problem solved?,” IEEE Trans-
actions on Medical Imaging, vol. 37, pp. 2514–2525,
2018.
[16] Vikas Verma, Kenji Kawaguchi, Alex Lamb, Juho Kan-
nala, Arno Solin, Yoshua Bengio, and David Lopez-Paz,
“Interpolation consistency training for semi-supervised
learning,” Neural Networks, vol. 145, pp. 90–106, 2022.
[17] Xiangde Luo, Guotai Wang, Wenjun Liao, Jieneng
Chen, Tao Song, Yinan Chen, Shichuan Zhang, Dim-
itris N Metaxas, and Shaoting Zhang, “Semi-supervised
medical image segmentation via uncertainty rectified
pyramid consistency,” Medical Image Analysis, vol. 80,
pp. 102517, 2022.