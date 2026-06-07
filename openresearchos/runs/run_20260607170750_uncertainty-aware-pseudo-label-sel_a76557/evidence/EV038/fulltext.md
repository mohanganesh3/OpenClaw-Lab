<!-- page 1 -->
IN PREPARATION FOR SUBMITTING TO IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS
1
Uncertainty-Aware Distillation for Semi-Supervised
Few-Shot Class-Incremental Learning
Yawen Cui, Wanxia Deng, Haoyu Chen, and Li Liu
Abstract—Given a model well-trained with a large-scale base
dataset, Few-Shot Class-Incremental Learning (FSCIL) aims at
incrementally learning novel classes from a few labeled samples
by
avoiding
overﬁtting,
without
catastrophically
forgetting
all encountered classes previously. Currently, semi-supervised
learning technique that harnesses freely-available unlabeled data
to compensate for limited labeled data can boost the performance
in numerous vision tasks, which heuristically can be applied to
tackle issues in FSCIL, i.e., the Semi-supervised FSCIL (Semi-
FSCIL). So far, very limited work focuses on the Semi-FSCIL
task, leaving the adaptability issue of semi-supervised learning
to the FSCIL task unresolved. In this paper, we focus on this
adaptability issue and present a simple yet efﬁcient Semi-FSCIL
framework named Uncertainty-aware Distillation with Class-
Equilibrium (UaD-CE), encompassing two modules UaD and
CE. Speciﬁcally, when incorporating unlabeled data into each
incremental session, we introduce the CE module that employs
a class-balanced self-training to avoid the gradual dominance
of
easy-to-classiﬁed
classes
on
pseudo-label
generation.
To
distill reliable knowledge from the reference model, we further
implement the UaD module that combines uncertainty-guided
knowledge reﬁnement with adaptive distillation. Comprehensive
experiments on three benchmark datasets demonstrate that our
method can boost the adaptability of unlabeled data with the
semi-supervised learning technique in FSCIL tasks. The code is
available at https://github.com/yawencui/UaD-CE.
Index Terms—Few-shot learning, class-incremental learning,
semi-supervised learning, knowledge distillation, uncertainty
estimation
I. INTRODUCTION
Deep learning has been successfully applied to a broad
range of computer vision tasks, such as image classiﬁcation
[1], [2], [3], object detection [4], [5], [6] and scene
segmentation [7], [8], etc. However, when generalizing a
trained model to unseen new classes, we need to retrain it
from scratch with a large number of labeled samples together
with the data from old classes. Otherwise, the discriminative
ability of old classes will be undermined if we only ﬁnetune it
with novel samples. Differently, humans can constantly receive
and learn new concepts without forgetting old ones even
This work was partially supported by National Key Research and
Development Program of China No. 2021YFB3100800, the Academy of
Finland under grant 331883 and the National Natural Science Foundation of
China under Grant 61872379, 62022091, and the China Scholarship Council
(CSC) under grant 201903170129.
Li Liu is with the College of System Engineering, National University of
Defense Technology (NUDT), Changsha, Hunan, China. She is also with the
Center for Machine Vision and Signal analysis, University of Oulu, Oulu, Fin-
land. Li Liu is the corresponding author. (email:dreamliu2010@gmail.com)
Yawen Cui and Haoyu Chen are with CMVS, University of Oulu, Oulu,
Finland. (email: yawen.cui@oulu.ﬁ; chen.haoyu@oulu.ﬁ)
Wanxia Deng is with the School of Meteorology and Oceanography, NUDT,
Changsha, Hunan, China. (email: dengwanxia14@nudt.edu.cn)
with limited supervised information of new concepts, which
stimulates the research interest in Few-Shot Class-Incremental
Learning (FSCIL).
Based on a model well-trained by a large-scale base
dataset, FSCIL [9], [10], [11], [12], [13], [14], [15], [16]
aims to incrementally learn new classes in limited labeled
data regime without forgetting previously seen categories.
This emerging research topic faces the following challenges:
(1) easy overﬁtting on the novel categories due to limited
labeled samples; and (2) catastrophic forgetting [17] on old
categories. Though existing FSCIL works tend to deploy
intricate representation structures [10], [12] or extra feature
reﬁnement module [11], [9] to eliminate the forgetting
and overﬁtting issues, they still easily suffer from severe
performance deterioration and classiﬁcation bias problem due
to the limited labeled data regime and data imbalance between
the base and novel categories.
The motivation of this work relies on the intuition that utiliz-
ing freely-available unlabelled data can endue the construction
of better learning procedures [18] and alleviate the above-
mentioned issues in FSCIL. Our previous works [19], [20]
ﬁrstly introduce semi-supervised learning into the FSCIL task
by straightforwardly leveraging easily accessible unlabeled
data. Although these works shed light on the Semi-Supervised
FSCIL (Semi-FSCIL) solution, they only provide the baseline
with existing methods and the straightforward solution, while
the role of unlabeled data still cannot be maximized. Besides,
there still remain some unsolved issues brought by unlabeled
data, such as the high model uncertainty with the reliance on
smoothness assumptions [18] and the issue of the robustness
to perturbations brought by unlabeled instances. We refer
to them collectively as the adaptability issue of semi-
supervised learning for the FSCIL task. In this work, we
focus on the adaptability issue and activate the potential
value of unlabeled data to FSCIL tasks. Our experimental
results argue that the adaptability issue can signiﬁcantly
affect the upper bound of the performance when harnessing
unlabeled data in FSCIL tasks. Furthermore, the current FSCIL
frameworks [9], [11] are not compatible with the Semi-
FSCIL task, since experimental results illustrate no expected
remarkable performance improvement when we incorporate
unlabeled data, which also emphasizes that the endowing of
models’ adaptability of the unlabeled data is signiﬁcant.
Toward the unresolved adaptability issues in Semi-FSCIL,
we propose a simple yet efﬁcient Semi-FSCIL framework
named Uncertainty-aware Distillation with Class Equilib-
rium (UaD-CE). Figure 1 describes the distribution of
embedding features from our proposed method and the
arXiv:2301.09964v1  [cs.CV]  24 Jan 2023


<!-- page 2 -->
IN PREPARATION FOR SUBMITTING TO IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS
2
(a) Baseline
Base classes
Catastrophic forgetting
&
Overfitting
(c) 
Base classes + Novel classes
...
Base classes + Novel classes
FSCIL
Standard distillation
Semi-FSCIL
Uncertainty-aware 
distillation
Old knowledge preserved
&
New knowledge obtained
...
Base classes
Labeled data
Labeled data
&
Unlabeled data
(a) 
Base classes + Novel classes
Base classes
Labeled data
&
Unlabeled data
Severe  bias
(b) 
...
Self-training
Class-balanced
Self-training
Fig. 1. The t-SNE visualization of classiﬁcation features in the FSCIL. (a)
The top: the baseline method [21] fails to model the distribution of novel
categories with limited samples, causing severe overﬁtting. It also weakens
the previous discriminative ability, i.e., the catastrophic forgetting of old
categories. (b) The middle: when implementing Semi-FSCIL with standard
knowledge distillation and self-training, the classiﬁer biases to base classes
with large dataset and easy-to-classiﬁed novel classes. (c) The bottom: with
the help of unlabeled samples, our proposed framework UaD-CE can preserve
the discriminative ability on base categories and obtain novel knowledge by
mitigating the overﬁtting issue.
baseline method [21]. As illustrated, our method can preserve
the discriminative ability on previously seen categories
and obtain novel knowledge by mitigating the overﬁtting
issue concurrently. The UaD-CE encompasses two modules
corresponding to the dual challenges in FSCIL: Class
Equilibrium (CE) module, and Uncertainty-aware Distillation
(UaD) module.
The data imbalance exists between the base session and
the incremental session, thus the classiﬁer always biases
to base classes with a large amount of labeled data [10].
Moreover, experiments in the work [20] illustrate that there
is gradual dominance of easy-to-classiﬁed novel classes on
pseudo-label generation, which means that classes with a
good learning status tend to be better at being recognized. To
tackle the overﬁtting issue on novel categories with limited
labeled data and mitigate the classiﬁcation bias to base
categories and easy-to-classiﬁed categories, CE module is
implemented with two procedures. A substantial of easily
accessible unlabeled data is provided ﬁrst to alleviate the
data imbalance between the base and novel categories, which
constitutes the Semi-FSCIL. Secondly, inspired by curriculum
pseudo labeling [22], class-balanced self-training is employed
based on the learning status of categories to avoid the
gradual dominance of easy-to-classiﬁed classes on pseudo-
label generation in each incremental session. The UaD module
focuses on the catastrophic forgetting issue of FSCIL and
alleviating the performance deterioration, where we conduct an
uncertainty-guided reﬁnement for a more efﬁcacious exemplar
set to serve the adaptive distillation process.
We naturally unify the CE module and UaD module into one
single framework, which is trained end-to-end. To summarize,
our main contributions include:
• This paper focuses on the adaptability issue of harnessing
unlabeled samples with the semi-supervised learning
technique for FSCIL task, and explore the upper bound
of the performance improvement with unlabeled data.
• To delicately conduct Semi-FSCIL, Class Equilibrium
(CE) module is proposed to address overﬁtting and
classiﬁcation bias issues.
• We introduce an efﬁcient module named Uncertainty-
aware Distillation (UaD) to distill reliable knowledge
for memorizing previous categories and eliminate the
ambiguity between previous and novel categories.
• We provide a comprehensive assessment of UaD-CE
framework on three benchmark datasets to validate the
effectiveness concerning three evaluation indicators.
II. RELATED WORK
A. Few-Shot Learning
Few-Shot Learning (FSL) [23], [24], [25], [26], [27],
[28] aims to solve the target task with limited labeled
instances per class, and there is usually a related source
task whose knowledge can be transferred to the few-shot
target task. Traditional FSL can be categorized into data
augmentation-based methods, transfer learning-based methods,
and meta-learning-based methods. Data augmentation-based
methods [25] target at enlarging the limited labeled dataset
in the instance level or the feature level. The mechanism
of transfer learning-based methods pretrain a model with a
large-scale dataset ﬁrst and then further ﬁnetune the model
on the FSL task with the strategies of alleviating overﬁtting.
Recent efforts on meta-learning based methods mainly follow
the three directions: metric learning-based methods [26], [27]
employing metrics to evaluate the similarity among support
images and query images, Optimization-based methods [28],
[29] searching for better parameter conﬁgurations of the
model such that it can effectively adapt to FSL tasks
with a few gradient-descent update steps, and memory-based
methods [30] aiming at forcing the query samples to match
with the previously obtained knowledge.
B. Class-Incremental Learning
Class-Incremental Learning (CIL) [21], [31], [32], [33],
[34], [35] targets at continually learning a uniﬁed classiﬁer
to recognize all seen categories so far. Great efforts have
been devoted to the following two directions: identifying and
preserving signiﬁcant parameters of the original model [32],
and memorizing the knowledge of the old model through
some strategies like knowledge distillation [31]. Recently,
some works focused on generalizing CIL to a limited-data
regime and led to a new practical scenario, i.e., Few-Shot
CIL (FSCIL) [9], [10], [11], [12], [13], [14], [15], [16].
Existing methods for FSCIL mainly employ two strategies.
The one is knowledge representation and reﬁnement. Tao et

[CAPTION] Fig. 1. The t-SNE visualization of classiﬁcation features in the FSCIL. (a)


<!-- page 3 -->
IN PREPARATION FOR SUBMITTING TO IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS
3
al. [10] propose TOPIC to model the topology of the feature
space using neural gas. Zhang et al. [11] adopt a simple
but effective decoupled learning strategy of representations,
and Continually Evolved Classiﬁer (CEC) is proposed by
employing a graph model to propagate context information
between classiﬁers for adaptation. Zhu et al. [9] offer a
novel incremental prototype learning scheme to solve the
FSCIL task. Another strategy is via knowledge distillation.
Cheraghian et al. [36] employ the semantic information during
training, and an attention mechanism on multiple parallel
embeddings of visual data is proposed to align visual and
semantic vectors, which reduces issues related to catastrophic
forgetting. Recently, we benchmark the Semi-FSCIL task
ﬁrst and provide a detailed conﬁguration and straightforward
solution [19]. Then, we further consider the uncertainty in
semi-supervised learning process to promote the performance,
while the adaptability issues can not be solved better since
the gain obtained with unlabeled data is limited and the
performance imbalance among categories is severe [20]. In
this paper, we still focus on the unresolved adaptive issues
of Semi-FSCIL task and discuss the effect of uncertainty on
knowledge distillation procedure.
C. Semi-Supervised Learning
Semi-Supervised Learning [37], [38], [39], [40], [41]
is to promote the supervised learning performance with
unlabeled samples. Diverse methods focus on semi-supervised
learning, which can be divided into consistency-regularization
methods [38], [40], [22] and pseudo-labeling [39], [42].
Consistency-regularization methods always rely on an ex-
tensive set of data augmentations requiring domain-speciﬁc
knowledge that is insufﬁcient for the limited-data regime
(one or few labeled examples per category). FixMatch [40]
generates the pseudo-label for a weakly-augmented unlabeled
image ﬁrst, then the model is trained to predict the pseudo-
label of a strongly-augmented version of the same image.
Pseudo-labeling methods aim to assign pseudo labels for
unlabeled data by using the model trained on labeled
ones. Speciﬁcally, the pseudo labels can be created by the
predictions of trained neural network [39], [42] or assigned
based on neighborhood graph [41], [43], [44]. Zhu et al. [43]
represent labeled and unlabeled data as vertices in a weighted
graph with edge weights encoding the similarity between
instances. In this paper, we employ class-balanced self-training
to avoid the bias to easy-to-classiﬁed classes on pseudo-label
generation.
D. Uncertainty Estimation
There are mainly two types of uncertainty [45]: aleatoric
uncertainty which models the noise inherent in the data, and
epistemic uncertainty which accounts for the uncertainty in
the model. Aleatoric uncertainty can not be lessened even
if we collect more data, while epistemic uncertainty can
be explained away when enough data is provided. In this
paper, we mainly focus on epistemic uncertainty. Traditionally,
approximate Bayesian inference methods [46] can be applied
to obtain the model uncertainties. However, due to the costly
computation and the implementation hardness of Bayesian
neural network, Gal et al. [47] prove that dropout together
with its variants can be seemed as a Bayesian approximation
to represent model uncertainty in deep learning. The test time
data augmentation [48], [49] is one of the simpler predictive
uncertainty estimation techniques, which are also applied in
this paper.
III. METHODOLOGY
In this section, we introduce the semi-supervised learning
framework UaD-CE for Semi-FSCIL as shown in Fig. 2.
Our UaD-CE framework contains two key components: Class
Equilibrium (CE) module for tackling the overﬁtting issue with
large-scale unlabeled samples via semi-supervised learning,
and Uncertainty-aware Distillation (UaD) module for distilling
reliable knowledge from large-scale yet unlabeled samples and
eliminating the ambiguity between the current categories and
previous categories. We ﬁrst give the problem formation with
learning targets, then the overview of the framework, and
lastly the details of each component in the framework will
be introduced.
A. Problem Formulation
We present a sequence of disjoint datasets by D
=
{D1, D2, ..., Dn}, where D1 is the large-scale base dataset
used in the ﬁrst base session and the followings are all
novel few-shot datasets. To be speciﬁc, we deﬁne D1 =
{(xj, yj)}|D1|
j=1 where yj ∈C1 and C1 denotes the base category
set. In the i-th session where i > 1, the novel/new category
dataset is set as Di = Dl
i ∪Du
i with Dl
i and Du
i denoting the
labeled training data and unlabeled training data, respectively.
The labeled training data Dl
i = {(xj, yj)}N×K
j=1
consists of N
classes Ci with K labeled examples per class, i.e., a N-way
K-shot problem. The unlabeled training data Du
i = {xj}M
j=1
comprises unlabeled samples, where M ≫K. j is the index
of a speciﬁc sample in Du and Dl. Noteworthy, there is
no overlap between the categories of different sessions, i.e.,
Ci ∩Ci′ = ∅, where i ̸= i′. In this paper, we implement and
validate our proposed method on object classiﬁcation task. As
for the model F (·), it usually contains the backbone Θ (·) for
extracting features and the classiﬁcation head Γ (·).
The aim of Semi-FSCIL is to train a uniﬁed classiﬁcation
model with a sequence of disjoint datasets delicately, and it
requires the model to incrementally learn the new categories
without forgetting all the learned categories so far, i.e., in
session i, after being trained on the Di, F (·) needs to classify
the samples from categories of C1 ∪C2∪, ..., Ci.
B. Overview of UaD-CE Framework
Our UaD-CE framework arranged by data ﬂow is illustrated
in Figure. 2. First, F1 is trained with base dataset D1 by
computing the classiﬁcation loss. For the sake of preserving
the current classiﬁcation performance later, an exemplar set
E1 ﬁltered from D1 is stored in an extra memory. When
it comes to the second session, F2 is initialized by the
reference model F1, i.e., the teacher model. Then, class-
balanced self-training is conducted with Dl
2 and Du
2 for


<!-- page 4 -->
IN PREPARATION FOR SUBMITTING TO IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS
4
Session 1
Dataset
Session 2
Model
Class-balanced 
self-training
Class 
Equilibrium
Session n
Exemplar set
Class-balanced 
self-training
Labeled dataset
Unlabeled dataset
Labeled dataset
Unlabeled dataset
UaD
UaD
Uncertainty-aware Distillation 
(UaD) 
Session i
Exemplar 
set
Uncertainty-guided 
selection
Computing adaptive distillation loss
UaD
Fig. 2.
Illustration of the proposed UaD-CE framework. First, the model is trained on the large-scale D1. When it comes to the following sessions, we
incorporate the unlabeled set into the labeled training set to alleviate the inadequate of novel samples and propose to utilize the class-balanced self-training as
the semi-supervised learning method. In order to solve the catastrophic forgetting problem efﬁciently, the proposed UaD module is conducted by eliminating
the negative effect of unlabeled samples.
Algorithm 1 Semi-FSCIL with UaD-CE.
Input: D1, D2, ..., F (·), session number n.
Output:F (·) that can classify all seen categories so far.
1: for i in n do
2:
if n==1 then
3:
Train Θ (·), Γ (·) by Dl
1 to form F1;
4:
Sample exemplars E1 from D1;
5:
else
6:
Fref = Fi−1;
7:
Ftarget = Fref;
8:
for supervised epochs do
9:
Ftarget updates by learning E ∪Dl
i;
10:
for unlabeled iterations do
11:
Train Ftarget on Dl
i and Du
i by class-balanced
self-training;
12:
Do uncertainty-aware distillation with Fref
and Ftarget;
13:
Update Ei−1 to Ei by sampling from Dl
i and Du
i ;
14:
Fi = Ftarget;
15:
Conduct the classiﬁcation by NME;
16: return F (·) obtained after n sessions.
learning novel categories by mitigating the overﬁtting issue,
which is the Class-Equilibrium (CE) module. While for the
categories encountered in the previous training, Uncertainty-
aware Distillation (UaD) is implemented by distilling reliable
knowledge from the reference with the exemplar set E1. At the
end of the current session, E1 is updated into E2 by adding
the ﬁltered samples from the current training dataset. The
procedure of the second session will be iteratively executed
for the rest sessions to accomplish the Semi-FSCIL task. The
pseudo code of our proposed methods is shown in Algorithm 1.
C. Class Equilibrium
Through the evaluation of existing methods [11], [9]
for FSCIL, we ﬁnd that their overall performances are
mainly contributed by the high classiﬁcation accuracy of base
categories that possess massive labeled samples. However,
the overﬁtting issue of novel categories is still severe due
to the few-shot learning regime, which causes the huge
performance discrepancy between base and novel categories.
Semi-FSCIL [19] can resolve this bias issue mainly caused by
the imbalance among datasets by incorporating a freely-used
unlabeled dataset in each incremental session, which is also
adopted in CE module ﬁrst.
When harnessing unlabeled data, this work [19] utilizes
the self-training [50] as the semi-supervised learning method.
Since self-training selects pseudo labels based on the predic-
tion probability with the smoothness assumptions [18], it tends
to bias to easy-to-classiﬁed classes, ignoring other classes that
have inferior classiﬁcation performances. In this paper, we take
this bias and the learning status into consideration and apply
the class-balanced self-training for Semi-FSCIL which is the
critical component of CE module.
The CE module is conducted from the second session
since a large-scale dataset is available in the ﬁrst session. As
shown in Figure 2, in session i (i>1), an unlabeled dataset
Du
i = {xj}M
j=1 is introduced into each incremental session
to alleviate the class-imbalance between the base and novel
categories ﬁrst. In this way, the model can learn these novel
categories in a semi-supervised manner. The common self-
training process contains two procedures: supervised epochs
and unlabeled iteration. In supervised epochs, the model
is trained with labeled samples ﬁrst. After that, unlabeled
iterations are conducted L times for selecting unlabeled
samples together with obtained pseudo labels from the current
F into the training process. For each unlabeled iteration, the


**[Table p4.1]**
| Labeled dataset Unlabeled dataset Class-balanced self-training UaD | Labeled dataset Unlabeled dataset Class-balanced self-training |
| --- | --- |
|  |  |
|  | Class-balanced self-training |


**[Table p4.2]**
| UaD |  |
| --- | --- |
|  |  |


**[Table p4.3]**
| UaD |  |
| --- | --- |
|  |  |

[CAPTION] Fig. 2.
Illustration of the proposed UaD-CE framework. First, the model is trained on the large-scale D1. When it comes to the following sessions, we


<!-- page 5 -->
IN PREPARATION FOR SUBMITTING TO IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS
5
obtained pseudo labels are termed eYi = {eyj}M
j=1, in which eyj
obtained by the following:
eyj = 1
 
pc
j == max(p(xj))
 N
c=1 ,
(1)
where pc
j is the probability that this unlabeled sample belongs
to the cth category of the total N categories in session i, and
max(p(xj)) is the maximum of p(xj).
Typically, a threshold γ for max(p(x)) is deﬁned for
selecting the unlabeled samples
bDu
i,t together with their
pseudo labels. Inspired by curriculum pseudo labeling [22],
the learning status of a speciﬁc class can be reﬂected by the
numbers of unlabeled samples whose predictions fall into this
class and are above γ. Classes that are well-learned by the
model or easy-to-classiﬁed tends to have higher prediction
probability. In this way, these classes do not require overmuch
unlabeled data. To avoid the gradual dominance of easy-to-
classiﬁed classes on pseudo-label generation, we propose to
select unlabeled samples from class level.
First, we divide the bDu
i,t based on the acquired pseudo
labels, and for each speciﬁc category, we obtain a subset of
unlabeled dataset Du
i,t = {(xj, eyj)} representing the unlabeled
dataset assigned the pseudo label by the tth category in session
i. Due to the N-way K-shot problem in each incremental
session, we will obtain N subsets after the division process.
When choosing unlabeled data with the consideration of
the learning status, we introduce a separate parameter pi,t
determining the proportion of selected instances for a speciﬁc
category. Consequently, the selected unlabeled set in each
unlabeled iteration is
eDu
i = { eDu
i,t|t = 1, 2, ..., N},
(2)
where eDu
i,t is made of the ﬁrst pi,t partition of bDu
i,t ranked by
prediction probabilities. The setting of p should fulﬁll that
∀| bDu
i,a| ≤| bDu
i,b|, pi,a ≥pi,b,
(3)
where a and b stand for the ath and bth categories in session
i. This restriction guarantees that more unlabeled samples for
hard-to-classiﬁed categories are incorporated into the training
process.
D. Uncertainty-Aware Distillation
Another challenge of FSCIL is catastrophic forgetting on
previously seen categories. The distillation-based framework
is an effective solution for maintaining previous abilities by
transferring the related knowledge from the reference model
to the target model, which is veriﬁed in large-scale CIL tasks.
This work [19] for Semi-FSCIL also learns from knowledge
distillation technique by simply implementing this task on
existing CIL methods [31], [21]. However, The work [19]
ignores that the challenges faced by large-scale CIL tasks
and FSCIL tasks are not exactly the same, and the high
model uncertainty with the combination of unlabeled samples
results in the unstable distillation process. In this section, we
introduce the Uncertainty-aware Distillation module to solve
the issues in the distillation-based framework for Semi-FSCIL.
The mechanism of the standard knowledge distillation
technique is to evaluate prediction variations on old categories
after the model is updated by new categories of the current
session. Typically, a distillation loss is introduced to the
standard classiﬁcation loss. Distillation loss occurs between
two models used in a pair of neighboring sessions. The model
in the previous session is termed the reference model, and
another one in the current session is the target model. Apart
from the preserved reference model, part of old class samples,
termed exemplars, are required to store in an extra memory
for access to the current session. To this end, the incremental
learning loss function in a speciﬁc session i is deﬁned as
follows:
L(Di, Ei, F) = Lce(Di, Ei, F) + Ldl(Ei, F),
(4)
where Lce means the cross-entropy loss and Ldl denotes the
distillation loss, and Ei is old class exemplars drawn from the
datasets in previous i −1 sessions.
Due to unlabeled data being introduced into the incremental
learning session, the extra unlabeled data may be preserved as
the exemplars and participate knowledge distillation process
in the following sessions. Precisely, Ei contains labeled
samples and unlabeled samples at the same time. According
to Equation 4, the reference model generates objectives for
the target model, i.e., the reference model makes predictions
on exemplars. However, the predictions on exemplars from
the unlabeled dataset are uncertain because of the unstable
training process with limited labeled samples. Accordingly,
the distillation loss is unreliable for memorizing the previous-
seen categories. To transfer trustworthy knowledge from the
reference model, we propose an Uncertainty-aware Distillation
for the Semi-FSCIL conﬁguration.
The proposed UaD contains two components: uncertainty-
guided reﬁnement and adaptive distillation loss. To promote
the efﬁcacy of knowledge distillation, the uncertainty-guided
reﬁnement acts on the exemplar set to obtain more reliable
instances with more certain objectives obtained with the
reference model. First, the uncertainty of predictions on the
exemplar set
[49] is estimated with data augmentation.
Assuming that the uncertainty is termed λi,q for the qth
exemplar in session i, the updated exemplar set is deﬁned
as
eEi = Ei
h
1 [λi,q > τ]|Ei|
q=0
i
,
(5)
where τ is the uncertainty threshold. After the reﬁnement, ex-
emplars with stable predictions are remained, which represents
that old knowledge in the reference model can be effectively
transferred by computing losses on reliable exemplars.
Considering the quality of distilled knowledge, the reﬁne-
ment is conducted to remove uncertain exemplars. With the
quantity concern, we deﬁne the adaptive distillation procedure
with a dynamically changeable weight ζi:
ζi = ζbase × |Ei|/|eEi| ×
q
|Cold
i
|/|Cnew
i
|,
(6)
where ζbase is a ﬁxed term decided by the dataset. |Ei|/|eEi|
adjusts the amount of distilled old knowledge within a
particular incremental session. |Cold
i
| and |Cnew
i
| are the
numbers of seen categories previously and novel encountered


<!-- page 6 -->
IN PREPARATION FOR SUBMITTING TO IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS
6
TABLE I
THE DATASET CONFIGURATIONS FOR FSCIL. #CATEGORIES AND
#SAMPLES STAND FOR THE NUMBER OF CATEGORIES AND THE NUMBER
OF SAMPLES, RESPECTIVELY. THE LEARNING PATTERN REPRESENTS THE
SETTING OF NOVEL TASKS IN EACH INCREMENTAL LEARNING SESSION.
Base session
Incremental session
#Categories
#Samples
#Categories
#Samples
Incremental pattern
CIFAR100
60
500
40
5
5-way 5-shot
miniImageNet
60
500
40
5
5-way 5-shot
CUB200
100
30
100
5
10-way 5-shot
categories in session i.
p
|Cold
i
|/|Cnew
i
| considers the ratio of
the previous class number and the novel class number in the
current session, which is also appeared in [31]. To this end,
Equation 4 is redeﬁned as
L(Di, eEi, F) = Lce(Di, eEi, F) + ζiLdl(eEi, F),
(7)
where we use F to represent the model set. When computing
distillation loss, the F contains the reference model and the
target model. The modiﬁed distillation loss ensures that the old
knowledge can be memorized efﬁciently and balanced with
new knowledge to eliminate the ambiguity between old and
novel categories in the overall classiﬁcation process.
IV. EXPERIMENTS
A. Dataset and Evaluation Indicators
We conducted comprehensive experiments on three bench-
mark datasets for FSCIL: CIFAR100 [51], miniImageNet [26]
and CUB200 [52]. The dataset conﬁgurations are illustrated
in Table I. CIFAR100 [51] is commonly used in CIL. This
dataset concludes 100 categories with 600 RGB images per
class. For each category, 500 images are used for training and
100 images for testing. The size of the image is 32 × 32.
miniImageNet [26] is a subset of the ImageNet with a
smaller number of classes. It includes 600 images for each
of 100 classes. These images are of the size of 84 × 84.
CUB200 [52] contains about 6, 000 training images and 6, 000
test images of over 200 bird categories. The images are
resized to 256 × 256 and then cropped to 224 × 224 for
training. By following the dataset conﬁguration in [10], the
dataset conﬁguration for FSCIL is illustrated in Table I. For
CIFAR100 and miniImageNet, we set 60 and 40 classes as
the base and novel categories, respectively, and chose a 5-
way 5-shot setting in each incremental session. In total, we
had 9 training sessions, i.e.one session for base classes and 8
sessions for novel classes. While for CUB200, we chose 100
classes as base classes and split the remaining 100 classes
into 10 incremental sessions with the 10-way 5-shot setting.
Notably, except for labeled samples used in each incremental
learning session, the rest was regarded as unlabeled dataset,
which is followed by the work [19].
We conducted on three evaluation indicators: (1) the ﬁnal
overall accuracy (%) in the last session; (2) the performance
dropping rate (PD) (%) [11] that measures the absolute
accuracy drops in the last session w.r.t. the accuracy in the
ﬁrst session; (3) the average accuracy (%) of all the sessions.
B. Model Conﬁgurations
In our experiment, ResNet-18 [2] was employed as the
backbone for CIFAR100, miniImageNet and CUB200. As
for the backbone, we froze the parameters of the front four
layers after the ﬁrst session. During training, the model
was optimized by SGD [53] (with lr=0.1 and wd=5e-4).
When selecting exemplar set incorporated into the following
sessions for memorizing the previous session categories by
knowledge distillation, we used the method proposed in [21]
based on herding selection. Moreover, we applied the nearest-
mean-of-exemplars classiﬁcation, donated as NME. The whole
framework was implemented using Pytorch and trained on
GeForce RTX 3080 GPUs.
C. Parameter Conﬁgurations and Training Details
For the ﬁrst session of CIFAR100 and miniImageNet, the
learning rate started from 0.1 and was divided by 10 after 80
and 120 epochs (160 epochs in total). For the rest sessions, the
learning rate was 0.001 in 100 epochs. For CUB200, the base
learning rate in the ﬁrst session was 0.001, and divided by 10
after 80 and 120 epochs (160 epochs in total). The learning
rate of the following sessions was 0.0005 used in a total of 60
supervised epochs. The model was trained with the training
batch size of 128 for miniImageNet, and 32 for CIFAR100
and CUB200. The test batch size was 100 for miniImageNet,
and 50 for CIFAR100 and CUB200.
For each unlabeled iteration, we chose 10 unlabeled samples
incorporated into the training procedure with a class-balanced
manner. As for the the selection proportion p, this parameter
ensures that the numbers of unlabeled samples added to each
class are equal. In each unlabeled iteration, the model was
trained on the labeled dataset and chosen unlabeled dataset
with pseudo labels for extra 10 epochs for CIFAR100 and
miniImageNet, 20 epochs for CUB200.
To guarantee that labeled samples contributed more to the
training process, the model was not trained from the reference
model with more epochs in each unlabeled iteration, instead
continually trained after labeled epochs. Moreover, during
the extra epochs, the training dataset was not set randomly
to ensure that the model was trained on labeled data ﬁrst,
and losses could be generated again from unlabeled samples.
Finally, when computing the class means, the dimension was
set as 512.
For CIFAR100, we added 350 unlabeled samples in 35
unlabeled iterations, while for miniImageNet and CUB200,
we incorporated 160 unlabeled samples in 16 unlabeled
iterations. In the uncertainty-guided selection part, we added
random Gaussian noise to the input for obtaining the
model uncertainty. For each sample, 10 forward passes with
random Gaussian noise were conducted, and the variance on
predictions was regarded as the uncertainty. Then, we ranked
the exemplars in an ascending order based on the uncertainty.
Notably, we selected the top three-quarter exemplars which
then were used for computing the adaptive distillation loss.
In the adaptive weight, ζbase was assigned 1 to CIFAR100, 2
to miniImageNet and CUB200. As for the exemplar set, we


**[Table p6.1]**
| Base session #Categories #Samples CIFAR100 60 500 miniImageNet 60 500 CUB200 100 30 | Incremental session |
| --- | --- |
|  | #Categories #Samples Incremental pattern |
|  | 40 5 5-way 5-shot 40 5 5-way 5-shot 100 5 10-way 5-shot |


<!-- page 7 -->
IN PREPARATION FOR SUBMITTING TO IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS
7
selected 20 samples per class of previous sessions. The number
is equal to that of large-scale CIL setting [31].
D. Construction Details of the Comparative Study on Semi-
FSCIL Task.
For the fair comparison, we also implemented Semi-FSCIL
setting based on existing methods: SPPR [9], and CEC [11],
which is regarded as Semi-SPPR and Semi-CEC. For SPPR
and CEC, the backbone is frozen after the ﬁrst session, then
the average feature mean (i.e., prototypes) of a particular
novel category is computed by obtaining features with this
backbone in the following sessions. In the ﬁrst session, base
classes are also represented by prototypes. To further build a
uniﬁed classiﬁer, prototypes of all categories encountered so
far are fed into a graph network [11] or an attention module
[9]. When incorporating unlabeled data, we ﬁrst compute
prototypes based on limited labeled samples, then update them
with unlabeled data according to the distance between the
current prototypes and features of unlabeled data.
E. Comparative Studies
The comparative experimental results encompass three
perspectives: (1) To certify that our proposed UaD-CE for
Semi-FSCIL can beneﬁt from harnessing unlabeled samples,
we compared UaD-CE with the state of the arts for FSCIL
setting; (2) For the fairness issue, we compared with the
exist Semi-FSCIL method [19] and implemented Semi-FSCIL
setting with existing new FSCIL methods to elaborate that our
proposed framework can gain more from unlabeled samples;
(3) To illustrate that our UaD-CE possesses the superior
performance on mitigating overﬁtting issue and the bias to
base categories, we present the accuracy of base and novel
categories respectively, and compare them with the state of
arts. It is worth noting that the evaluation process for each
dataset was repeated 5 times, and we reported the average
overall accuracy.
Comparative
study
on
the
FSCIL
task. The results
of comparison FSCIL methods are directly quoted from
original papers to facilitate fair comparison. CIFAR100 and
miniImageNet results are presented in Figure 3 (a) and
Figure 3 (b), respectively. Besides, Table II presents the
results of CUB200. When comparing UaD-CE with the state
of the arts on CIFAR100, miniImageNet and CUB200 for
FSCIL setting, we conducted experiments with regard to
three evaluation indicators: (1) The ﬁnal overall accuracy
(%). Our proposed framework exceeds all other methods on
three datasets and even outperforms the “Joint-CNN” methods
to a large extent, which is deﬁned as the upper bound in
[10]. Especially, our proposed UaD-CE achieves 54.50% on
CIFAR100, 50.52% on miniImageNet and 60.72% on CUB200
in the ﬁnal session, surpassing the state of the arts by around
4% and 3%. (2) For the performance dropping rate (PD), the
proposed UaD-CE also gains remarkable capability. Although
for the ﬁrst session, our method cannot exceed the state of the
arts on miniImageNet and CUB200, the accuracies obtained
by our framework go through slight descending curves, which
indicates that old knowledge can be better preserved in our
proposed UaD-CE. (3) The average accuracy (%) of all the
sessions. The UaD-CE obtains advanced results on CIFAR100
(63.93%) and CUB200 (65.70%). As for miniImageNet, the
best performance of average accuracy is shown in [9], which
mainly results from the higher accuracy of the ﬁrst session. As
CIL aims to mitigate catastrophic forgetting on old categories,
the attenuation of accuracy (i.e., PD) is more convincing than
the average accuracy for evaluating CIL methods. Though the
result of the ﬁrst session in our proposed framework is inferior
to that in [9], our UaD-CE outpaces it as novel categories
arriving in the following sessions.
Comparative study on the Semi-FSCIL task. To explain
that unlabeled samples can contribute more to FSCIL with
our proposed framework, we present the comparative results
with the Semi-FSCIL method [19]. Furthermore, the same
unlabeled instances were fairly incorporated into existing
FSCIL methods, referred to Semi-FSCIL. CEC [11] and SPPR
[9] are advanced works for FSCIL, and they also release
codes
12. These codes can reduce deviations brought by
implementation details from original papers and be reliant on
conducting Semi-CEC and Semi-SPPR. Results of Semi-CEC
and Semi-SPPR are presented on Table II for CUB200, and
Table III for CIFAR100 and miniImageNet. In view of the
results, the performance of FSCIL boosts to a greater extent by
harnessing unlabeled samples with our UaD-CE framework.
However, when the same number of unlabeled samples are
combined to Semi-CEC and Semi-SPPR, a slight enhancement
is obtained. The reason is that CEC and SPPR heavily rely
on reﬁning the acquired latent features by the graph structure
[11] or attention modules [9]. This particularity undermines
the role of the latent representative capability, where the
semi-supervised learning technique fundamentally contributes.
Compared with Us-KD [20], UaD-CE outperforms it regarding
three evaluation indicators, which illustrates that considering
the uncertainty issue during distillation is more applicable and
efﬁcient than that of in semi-supervised learning process.
Comparative results of base and novel categories. To
reveal that UaD-CE is efﬁcacious to conquer the overﬁtting
challenges and mitigate the bias to base categories, we
separately exhibit the accuracy of base categories (i.e., C1)
and novel categories (i.e., C2∪, ..., Cn) in Table IV. For
CEC and SPPR, we use the released codes to conduct
comparison experiments on base and novel categories. Models
can better remember the performance of base categories with
the large-scale dataset, while the overﬁtting problem is still
severe among novel classes with limited labeled samples.
Although the overall performance is roughly satisfactory, these
methods can not resolve two issues in the FSCIL task at
the same time, which bring out the severe classiﬁcation bias
to base categories. Surprisingly, our proposed UaD-CE can
mitigate overﬁtting on novel categories while maintaining the
performance of the base ones.
Further remark. We present the visualized t-SNE [57]
results in Figure 4 (a)-(c). The t-SNE results of our methods
1https://github.com/icoz69/CEC-CVPR2021
2https://github.com/zhukaii/SPPR

[CAPTION] Figure 3 (b), respectively. Besides, Table II presents the


<!-- page 8 -->
IN PREPARATION FOR SUBMITTING TO IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS
8
(a) CIFAR100
(b) miniImageNet
Fig. 3. Comparative study on the FSCIL task with CIFAR100 (a) and miniImageNet (b). For CIFAR100, our proposed UaD-CE outperforms the state of the
arts regarding three evaluation indicators. For miniImageNet, though the result of the ﬁrst session in our proposed framework is inferior to that of [9], our
framework outpaces it as novel categories arriving in the following sessions.
(the right) have clearer decision boundaries of the old and
new categories than that of iCaRL [21] (the left) used the
standard distillation technique (i.e., Equation 4). In addition,
we visualize the loss of the training on CIFAR100 to illustrate
the superiority of the UaD module in Figure 5. From the
loss curves, it can be shown that curve (b) representing UaD
experiences relatively smaller ﬂuctuations. This fact indicates
that the gradients generated by computing distillation losses
on the uncertainty-guided reﬁned exemplar set have similar
decline directions, i.e., in most cases, valid old knowledge
from the reference model is transferred to the target model.
F. Ablation Studies
To demonstrate the efﬁcacy of UaD-CE framework, we
conducted extensive ablation experiments on the proposed
modules and hyperparameters.
Efﬁcacy
of
UaD
module. The UaD-CE framework in
this paper is set up based on the knowledge distillation
technique for the class-incremental learning purpose, and the
proposed UaD module is proposed to make this technique
more applicable to Semi-FSCIL. The common form of
knowledge distillation usually follows the distillation loss
deﬁned in Equation 4
[58], and we term it standard
knowledge distillation (S DL). We evaluated our framework
by replacing UaD module with S DL, and the comparative
result is illustrated in Figure 6 (a). With the UaD module,
the overall classiﬁcation accuracy exceeds that of the standard
knowledge distillation in all incremental sessions, proving the
prominent adaptability of the UaD module to Semi-FSCIL.
This adaptability is achieved due to the quality and quantity
of old knowledge we consider in the distillation process. With
the UaD module, the negative effect brought by unlabeled data
Novel categories
Old categories
(a)
(b)
(c)
Fig. 4.
(a)-(c) The t-SNE visualization of the features for CIFAR100 (a),
miniImageNet (b), and CUB200 (c).
can be mitigated to some extent, i.e., unlabeled data can better
assist the knowledge distillation process in memorizing the old
knowledge.
Efﬁcacy
of
CE
module. As for effectively harnessing
unlabeled data in each incremental session to alleviate the
overﬁtting issue, we propose the CE module in which the
main component is class-balance self-training. We analyzed

[CAPTION] Fig. 3. Comparative study on the FSCIL task with CIFAR100 (a) and miniImageNet (b). For CIFAR100, our proposed UaD-CE outperforms the state of the

[CAPTION] Fig. 4.
(a)-(c) The t-SNE visualization of the features for CIFAR100 (a),


<!-- page 9 -->
IN PREPARATION FOR SUBMITTING TO IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS
9
TABLE II
COMPARATIVE STUDY ON FSCIL AND SEMI-FSCIL TASKS WITH CUB200. PD IS THE PERFORMANCE DROPPING RATE FROM THE FIRST SESSION TO
THE LAST SESSION. AVERAGE ACC. IS THE AVERAGE PERFORMANCE OF ALL THE ENCOUNTERED SESSIONS.
Task
Method
Session ID
PD↓
Average
Acc.
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
FSCIL
Ft-CNN [10]
68.68
44.81
32.26
25.83
25.62
25.22
20.84
16.77
18.82
18.25
17.18
51.50
28.57
Joint-CNN [10]
68.68
62.43
57.23
52.80
49.50
46.10
42.80
40.10
38.70
37.10
35.60
33.08
48.28
iCaRL,2017 [21]
68.68
52.65
48.61
44.16
36.62
29.52
27.83
26.26
24.01
23.89
21.16
47.52
36.67
EEIL,2018 [54]
68.68
53.63
47.91
44.20
36.30
27.46
25.93
24.70
23.95
24.13
22.11
46.57
36.27
NCM,2019 [31]
68.68
57.12
44.21
28.78
26.71
25.66
24.62
21.52
20.12
20.06
19.87
48.81
32.49
TOPIC,2020 [10]
68.68
62.49
54.81
49.99
45.25
41.40
38.35
35.36
32.22
28.31
26.28
42.40
43.92
SAKD,2021 [36]
68.23
60.45
55.70
50.45
45.72
42.90
40.89
38.77
36.51
34.87
32.96
35.27
46.13
SPPR,2021 [9]
68.68
61.85
57.43
52.68
50.19
46.88
44.65
43.07
40.17
39.63
37.33
31.35
49.32
CEC,2021 [11]
75.85
71.94
68.50
63.50
62.43
58.27
57.73
55.81
54.83
53.52
52.28
23.57
61.33
ERDIL,2021 [55]
73.52
71.09
66.13
63.25
59.49
59.89
58.64
57.72
56.15
54.75
52.28
21.24
61.17
SFMS,2021 [56]
68.78
59.37
59.32
54.96
52.58
49.81
48.09
46.32
44.33
43.43
43.23
25.55
51.84
IDLVQ-C,2021 [12]
77.37
74.72
70.28
67.13
65.34
63.52
62.10
61.54
59.04
58.68
57.81
19.56
65.18
FACT, 2022 [13]
75.90
73.23
70.84
66.13
65.56
62.15
61.74
59.83
58.41
57.89
56.94
18.96
64.42
MetaFSCIL, 2022 [14]
75.90
72.41
68.78
64.78
62.96
59.99
58.30
56.85
54.78
53.82
52.64
23.26
61.92
Liu et al., 2022 [15]
75.90
72.14
68.64
63.76
62.58
59.11
57.82
55.89
54.92
53.58
52.39
23.51
61.52
Semi-FSCIL
SS-iCaRL [19]
69.89
61.24
55.81
50.99
48.18
46.91
43.99
39.78
37.50
34.54
31.33
38.56
47.29
SS-NCM [19]
69.89
61.91
55.51
51.71
49.68
46.11
42.19
39.03
37.96
34.05
32.60
37.24
47.33
SS-NCM-CNN [19]
69.89
64.87
59.82
55.14
52.48
49.60
47.87
45.10
40.47
38.10
35.25
34.64
50.78
Semi-SPPR
68.44
61.66
57.11
53.41
50.15
46.68
44.93
43.21
40.61
39.21
37.43
31.01
49.34
Semi-CEC
75.82
71.91
68.52
63.53
62.45
58.27
57.62
55.81
54.85
53.52
52.26
23.56
61.32
Us-KD [20]
74.69
71.71
69.04
65.08
63.60
60.96
59.06
58.68
57.01
56.41
55.54
19.15
62.89
UaD-CE (ours)
75.17
73.27
70.87
67.14
65.49
63.66
62.42
62.55
60.99
60.48
60.72
14.45
65.70
TABLE III
COMPARATIVE STUDY ON THE SEMI-FSCIL TASK WITH CIFAR100 AND miniIMAGENET DATASET.
Dataset
Task
Method
Session ID
PD↓
Average
Acc.
1
2
3
4
5
6
7
8
9
CIFAR100
FSCIL
SPPR,2021 [9]
76.68
72.69
67.61
63.52
59.18
55.82
53.08
50.89
48.12
28.56
60.84
CEC,2021 [11]
73.03
70.86
65.20
61.27
58.03
55.53
53.17
51.19
49.06
23.97
59.70
UaD-CE (ours)
75.55
71.78
65.47
62.83
55.56
55.08
50.11
46.35
40.46
35.09
58.13
Semi-FSCIL
Semi-SPPR
76.68
72.63
67.59
63.69
59.24
56.02
53.23
50.46
48.29
28.39
60.87
Semi-CEC
73.03
70.72
65.79
61.91
58.64
55.84
53.70
51.37
49.37
23.66
60.04
SS-iCaRL [19]
64.13
56.02
51.16
50.93
43.46
41.69
38.41
39.25
34.80
29.33
46.65
SS-NCM-CNN [19]
64.13
62.29
61.31
57.96
54.26
50.95
49.02
45.85
44.59
19.54
54.51
Us-KD [20]
76.85
69.87
65.46
62.36
59.86
57.29
55.22
54.91
54.42
22.43
61.80
UaD-CE (ours)
75.55
72.17
68.57
65.35
62.80
60.27
59.12
57.05
54.50
21.05
63.93
miniImageNet
FSCIL
SPPR,2021 [9]
80.27
74.22
68.89
64.43
60.54
56.82
53.81
51.22
48.54
31.73
62.08
CEC,2021 [11]
72.22
67.06
63.17
59.79
56.96
53.91
51.36
49.32
47.60
24.62
57.93
UaD-CE (ours)
72.35
66.83
61.94
58.48
55.77
52.20
49.96
47.96
46.81
25.54
56.92
Semi-FSCIL
Semi-SPPR
80.10
74.21
69.31
64.83
60.53
57.36
53.70
52.01
49.61
30.49
62.41
Semi-CEC,2021
71.91
66.81
63.87
59.41
56.42
53.83
51.92
49.57
47.58
24.33
57.92
SS-iCaRL [19]
62.98
51.64
47.43
43.92
41.69
38.74
36.67
34.54
33.92
29.06
43.50
SS-NCM-CNN [19]
62.98
60.88
57.63
52.8
50.66
48.28
45.27
41.65
40.51
22.47
51.26
Us-KD [20]
72.35
67.22
62.41
59.85
57.81
55.52
52.64
50.86
50.47
21.88
58.79
UaD-CE (ours)
72.35
66.91
62.13
59.89
57.41
55.52
53.26
51.46
50.52
21.83
58.82
the function of class-balanced operation in CE module by
comparing the performance with traditional self-training [50],
which is presented in Figure 6 (b). The curve with CE module
experiences a slight decline. It demonstrates that the CE
module can enhance the overall classiﬁcation performance
by taking the model’s learning status into consideration. In
each incremental session, the difﬁculty of learning varies for
different classes, and it also refers to the learning status of
the model. Therefore, in our proposed CE module, when
selecting unlabeled data based on the predictions, we set
a speciﬁc threshold for each class, which can mitigate the
performance imbalance. The difﬁculty of learning also varies
among different incremental sessions. This is why the curve
experiences a slight increase in the eighth session.
Impact of number of incorporated unlabeled samples.
Since we discuss the adaptability issue of semi-supervised
learning to the FSCIL task, the extreme value of incorporated
unlabeled data is explored in our experiments so that the
performance can achieve the greatest improvement. The
ablation results on the number of incorporated unlabeled
samples are exhibited in Figure 6 (c). With different numbers
of unlabeled samples, the overall classiﬁcation accuracy
can increase to varying degrees when compared with the
performance obtained without extra unlabeled data. In this


<!-- page 10 -->
IN PREPARATION FOR SUBMITTING TO IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS
10
TABLE IV
PREDICTION ACCURACY OF BASE AND NOVEL CLASSES ON THREE BENCHMARK DATASETS.
Dataset
Task
Method
Classes
Session ID
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
CIFAR100
FSCIL
SPPR,2021 [9]
Base
76.68
77.48
77.32
76.95
60.25
76.42
75.95
75.83
75.30
-
-
Novel
-
1.52
9.40
9.80
7.45
7.52
7.33
8.14
7.35
-
-
All
76.68
72.69
67.61
63.52
59.18
55.82
53.08
50.89
48.12
-
-
CEC,2021 [11]
Base
73.03
73.77
71.52
70.78
70.07
69.28
68.70
68.45
67.78
-
-
Novel
-
36.00
27.30
22.67
21.90
22.52
22.10
21.60
20.98
-
-
All
73.03
70.86
65.20
61.27
58.03
55.53
53.17
51.19
49.06
-
-
Semi-FSCIL
SS-iCaRL [19]
Base
64.13
56.97
53.77
55.30
48.07
47.48
44.18
45.84
40.97
-
-
Novel
-
44.57
35.48
33.47
27.40
27.39
25.71
27.97
26.10
-
-
All
64.13
56.02
51.16
50.93
43.46
41.69
38.41
39.25
34.80
-
-
SS-NCM-CNN [19]
Base
64.13
63.34
64.43
62.93
60.01
58.02
56.39
53.55
52.49
Novel
-
49.56
42.51
38.08
34.21
33.99
32.81
32.67
33.44
-
-
All
64.13
62.29
61.31
57.96
54.26
50.95
49.02
45.85
44.59
-
-
UaD-CE (ours)
Base
75.55
73.20
70.06
68.94
67.63
66.82
65.81
65.25
61.99
-
-
Novel
-
59.80
59.21
53.00
48.30
46.60
45.73
43.00
42.27
-
-
All
75.55
72.17
68.57
65.35
62.80
60.27
59.12
57.05
54.50
-
-
miniImageNet
FSCIL
SPPR,2021 [9]
Base
80.27
80.22
80.28
80.20
80.08
79.77
79.98
79.58
79.87
-
-
Novel
-
2.20
0.50
1.33
1.90
1.76
1.47
2.60
1.55
-
-
All
80.27
74.22
68.89
64.43
60.54
56.82
53.81
51.22
48.54
-
-
CEC,2021 [11]
Base
72.22
70.92
70.17
69.65
69.32
68.98
68.68
68.25
67.87
-
-
Novel
-
20.80
21.20
20.33
19.90
17.72
16.70
16.86
17.20
-
-
All
72.22
67.06
63.17
59.79
56.96
53.91
51.36
49.32
47.60
-
-
Semi-FSCIL
SS-iCaRL [19]
Base
62.98
53.63
49.85
47.69
46.11
44.10
42.18
40.34
39.93
-
-
Novel
-
27.66
32.89
28.86
26.28
25.83
24.54
24.61
25.44
-
-
All
62.98
51.64
47.43
43.92
41.69
38.74
36.67
34.54
33.92
-
-
SS-NCM-CNN [19]
Base
62.98
63.23
60.57
57.33
56.03
54.98
52.08
48.64
47.69
-
-
Novel
-
32.61
39.96
34.69
31.94
32.21
30.30
29.68
30.38
-
-
All
62.98
60.88
57.63
52.80
50.66
48.28
45.27
41.65
40.51
-
-
UaD-CE (ours)
Base
72.35
70.27
67.36
66.06
64.26
63.14
61.78
60.71
59.20
-
-
Novel
-
27.03
30.73
35.20
36.85
37.24
36.23
35.60
36.70
-
-
All
72.35
66.91
62.13
59.89
57.41
55.52
53.26
51.46
50.52
-
-
CUB200
FSCIL
SPPR,2021 [9]
Base
68.16
60.67
59.87
60.13
57.82
55.85
53.77
53.57
52.95
52.43
51.73
Novel
-
60.39
45.55
30.24
31.07
27.75
26.67
27.51
25.16
24.76
23.88
All
68.16
60.32
57.11
52.79
49.68
45.95
43.19
42.39
40.17
38.93
37.33
CEC,2021 [11]
Base
75.82
74.34
73.94
73.59
72.73
72.39
71.90
71.28
71.14
70.90
70.66
Novel
-
46.61
41.74
33.34
37.46
32.77
34.78
35.06
33.23
34.98
34.17
All
75.82
71.91
68.52
63.53
62.45
58.27
57.62
55.81
54.85
53.52
52.26
Semi-FSCIL
SS-iCaRL [19]
Base
69.89
62.32
60.62
58.99
58.59
57.77
59.88
56.21
54.46
50.54
46.11
Novel
-
53.22
32.38
24.07
22.76
23.34
17.58
16.40
16.39
16.13
16.32
All
69.89
61.24
55.81
50.99
48.18
46.91
43.99
39.78
37.50
34.54
31.33
SS-NCM-CNN [19]
Base
69.89
65.80
64.97
63.79
63.81
61.08
65.24
63.73
58.77
55.74
51.88
Novel
-
56.37
34.70
26.03
24.04
24.68
19.14
18.60
17.70
17.79
18.36
All
69.89
64.87
59.82
55.14
52.48
49.60
47.87
45.10
40.47
38.10
35.25
UaD-CE (ours)
Base
75.17
74.58
73.78
72.97
71.33
70.88
69.76
69.48
68.26
68.82
68.47
Novel
-
59.86
56.18
47.80
51.12
49.52
50.40
52.86
52.09
51.14
53.14
All
75.17
73.27
70.87
67.14
65.49
63.66
62.42
62.55
60.99
60.48
60.72
Loss
40
45
50
55
60
0
20
40
60
80
100
Epoch
S_DL 
UaD
Fig. 5. Loss curves of the second session in CIFAR100. S DL means standard
knowledge distillation.
way, it can be summarized that the performance of FSCIL
can proﬁt from the combination of unlabeled data. With
160 unlabeled samples, the proposed UaD-CE obtains the
surpassing performance. There are mainly two reasons why
more unlabeled data is not better: (1) Unlabeled data may
bring the noise to the model training due to its quality and
the uncertainty of obtaining its pseudo label. Consequently,
the trade-off between the proﬁt and the noise brought by
unlabeled data exists in Semi-FSCIL tasks. In our future work,
we will seek for advanced technique to mitigate the negative
effects brought by unlabeled data in Semi-FSCIL. (2) For
distillation-based incremental learning methods, the trade-off
between the performance of old and new classes should be
taken into consideration. In particular, Tao et al. [10] point
out that this trade-off is more challenging for FSCIL because
the large learning rate was required to learn novel classes with
limited labeled data. Though unlabeled data is combined into
each incremental learning session to relax this requirement,
we still need to handle the trade-off issue carefully. Moreover,
we also illustrate the ablation study results in Table V with

[CAPTION] Fig. 5. Loss curves of the second session in CIFAR100. S DL means standard


<!-- page 11 -->
IN PREPARATION FOR SUBMITTING TO IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS
11
(a)
(b)
(c)
(e)
(f)
(g)
(h)
(d)
Fig. 6. Ablation study on modules and hyperparameters in UaD-CE with CUB200.
TABLE V
ABLATION STUDY RESULTS ON THE NUMBER OF UNLABELED SAMPLES INCORPORATED IN EACH INCREMENTAL LEARNING SESSION. #Unlabeled
REPRESENTS THE NUMBER OF UNLABELED SAMPLES INCORPORATED INTO EACH INCREMENTAL SESSION. Prop (%) REFERS TO THE PROPORTION OF
THE NUMBER OF LABELED SAMPLES TO THE NUMBER OF ALL SAMPLES.
Dataset
#Unlabeled
Prop (%)
Session ID
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
CIFAR100
250
9.10
75.55
71.49
67.76
64.00
61.02
58.35
57.23
55.46
53.65
-
-
350
6.70
75.55
72.17
68.57
65.35
62.80
60.27
59.12
57.05
54.50
-
-
450
5.30
75.55
71.67
67.67
64.20
60.70
58.51
56.78
55.27
53.67
-
-
miniImageNet
100
20.00
72.35
61.89
58.83
56.39
53.14
51.01
49.56
48.62
-
-
160
13.51
72.35
66.91
62.13
59.89
57.41
55.52
53.26
51.46
50.52
-
-
200
11.11
72.35
66.62
61.86
58.76
56.27
53.81
51.83
50.83
49.78
-
-
CUB200
100
33.33
75.17
72.86
69.59
66.04
64.23
62.70
61.08
60.72
58.18
57.75
57.28
160
23.81
75.17
73.27
70.87
67.14
65.49
63.66
62.42
62.55
60.99
60.48
60.72
200
20.00
75.21
72.10
69.15
66.40
64.28
62.40
60.97
60.34
58.59
59.34
58.47
the numerical representation and the proportion (%) of labeled
samples. The best proportion varies for different datasets, and
it may result from the nature of the dataset itself and the
incremental pattern. We will continue exploring the root causes
affecting this proportion in future work.
Impact of classiﬁcation head. In the default setting of
UaD-CE framework, we apply the nearest-mean-of-exemplars
(NME) classiﬁcation. In this ablation, we consider the CNN
predictions and give the comparative result in Figure 6
(d). With increasing novel categories with limited labeled
instances, the NME classiﬁcation outperforms the CNN
prediction signiﬁcantly from the seventh session. NME
classiﬁcation is to measure the distance between the features of
test images and prototypes (i.e., the class mean in the feature
level), where the computation is only related to parameters
of the backbone. However, the CNN prediction also requires
putting the features into the fully-connected layer, and the
prediction is then conducted based on the ﬁnal outputs. Aiming
at the incremental learning pattern, the fully-connected layer
has to expand to ﬁt the increase of total classes. For a speciﬁc
session, the parameters of the backbone are optimized in all
encountered sessions, while the parameters related to the new
neurons of the fully-connected layers are only updated in the
current session, which can explain why the NME classiﬁcation
achieves the superior performance to the CNN prediction.
Impact of ζ in UaD module. One of the challenges for
the distillation-based framework in FSCIL is the trade-off
between old and new classes. With the number of seen
classes increasing, the model should memorize more old
knowledge for the overall classiﬁcation performance. The
weight ζ tradeoffs the contribution of previous knowledge
to the current task for enabling the overall classiﬁcation
ability. Figure 6 (e) illustrates the performance without
this dynamically changeable weight (Without AW). We can
conclude that the adaptive weight for distillation loss is
effective for Semi-FSCIL.
Impact of τ in UaD module. Hyperparameter τ is the
threshold of the uncertainty, and it determines how many
reliable exemplars of previous classes are reserved totally in
a speciﬁc following session for the knowledge distillation.
This value also determines the quantity of old knowledge
considered when we conduct the adaptive weights. We increase
this number from 0.5 to 0.8, and report the results in Figure 6
(f). The superior overall classiﬁcation performance is achieved


**[Table p11.1]**
|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

[CAPTION] Fig. 6. Ablation study on modules and hyperparameters in UaD-CE with CUB200.


<!-- page 12 -->
IN PREPARATION FOR SUBMITTING TO IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS
12
Fig. 7. Ablation results of CIFAR100 on semi-supervised learning method.
when we remain the ﬁrst three-quarter of exemplars based on
the uncertainty.
Study
of
noise
in
uncertainty
estimation
part. We
apply the test-time augmentation method for the uncertainty
estimation, and Gaussian noise is added to the original
images for the augmentation. Besides the Gaussian noise used
in the uncertainty estimation part, we conduct the ablation
experiments with random augmentation, i.e., random crop and
random ﬂip. As shown in Figure 6 (g), using Gaussian noise
for uncertainty estimation is more valid in UaD-CE. Since
the model is trained with a large-scale dataset, the model
can have a relatively robust discriminative capability. In this
way, random augmentation is too weak to generate the diverse
predictions of a speciﬁc image, thus making the uncertainty-
aware module less salient to the model performance.
Impact of ζbase in UaD module. The ablation study result
on ζbase is given in Figure 6 (h). Hou et al. [31] also apply
the adaptive weight of the distillation loss and point out
that ζbase is constant for a speciﬁc dataset. We also follow
this conﬁguration in our experiment. However, in UaD-CE
framework, ζbase is not sensitive to this parameter, which
demonstrates that the last two components in ζ are more
important than ζbase. In future work, we will explore better
forms of adaptive weight.
Comparison with more semi-supervised learning methods.
In the UaD-CE framework, we apply class-balanced self-
training (CB ST) when harnessing unlabeled data in a semi-
supervised manner. To illustrate the efﬁcacy of class-balanced
self-training, we replace it with the consistency regularization-
based methods, i.e., MixMatch [38] and FixMatch [40]. These
ablation study results are presented in Figure 7. It can
be seen that the performance obtained with MixMatch and
FixMatch is inferior to that of class-balanced self-training.
Since our proposed framework is knowledge distillation-based,
the performance trade-off between old and new classes in
FSCIL is hard to handle, which is pointed out in [10].
For consistency regularization-based methods, the consistency
loss is generated on unlabeled samples. When this loss is
combined with distillation loss, it may generate gradients
for learning new classes in different directions from that of
distillation loss, which is not conducive to preserving the
old knowledge. We will seek the solutions that can make
advanced consistency regularization-based semi-supervised
learning methods efﬁcacious in Semi-FSCIL.
V. CONCLUSION AND FUTURE WORK
In this paper, we focus on tackling the unresolved
adaptability issue of semi-supervised learning to the FSCIL
task and propose a simple yet efﬁcient Semi-FSCIL framework
named UaD-CE, which includes CE and UaD modules.
CE is presented to address the overﬁtting and bias issues
in FSCIL by incorporating unlabeled samples with class-
balanced self-training. Furthermore, to efﬁciently address the
catastrophic forgetting problem, we conduct the UaD module
with uncertainty-guided reﬁnement and adaptive distillation.
Comprehensive experiments on three benchmark datasets
achieve superior results, demonstrating that introducing the
semi-supervision into FSCIL by our framework can achieve
prominent and robust enhancement.
In spite of the remarkable performance achieved by the
UaD-CE framework, some aspects still need to be improved:
(1) In each incremental learning session, the best proportion of
unlabeled data in the semi-supervised learning process varies
for different datasets. This proportion can affect not only the
learning status of the current session but also the memorization
of old knowledge. In the future, we will explore the root
causes that affec the best proportion of labeled data. (2) The
trade-off issue between old and new classes is challenging
in distillation-based methods for FSCIL, and the adaptive
weight of distillation loss is proposed to tackle this issue.
More valid solutions will be explored in our future work. (3)
Compared with pseudo-labeling-based methods, consistency
regularization-based semi-supervised learning methods have
achieved dominant performance. For future work, we will
discuss the compatibility issue of this series of semi-supervised
learning methods and distillation-based incremental learning
framework.
REFERENCES
[1] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classiﬁcation
with deep convolutional neural networks,” in NeurIPS, 2012, pp. 1097–
1105. 1
[2] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image
recognition,” in CVPR, 2016, pp. 770–778. 1, 6
[3] K. Tang, Y. Ma, D. Miao, P. Song, Z. Gu, Z. Tian, and W. Wang,
“Decision fusion networks for image classiﬁcation,” IEEE Transactions
on Neural Networks and Learning Systems, 2022. 1
[4] L. Liu, W. Ouyang, X. Wang, P. Fieguth, J. Chen, X. Liu, and
M. Pietik¨ainen, “Deep learning for generic object detection: A survey,”
International Jornal of Computer Vision, vol. 128, no. 2, pp. 261–318,
2020. 1
[5] S. Ren, K. He, R. Girshick, and J. Sun, “Faster r-cnn: Towards real-time
object detection with region proposal networks,” in NeurIPS, 2015, pp.
91–99. 1
[6] W. Zhou, Q. Guo, J. Lei, L. Yu, and J.-N. Hwang, “Irfr-net: interactive
recursive feature-reshaping network for detecting salient objects in rgb-d
images,” IEEE Transactions on Neural Networks and Learning Systems,
2021. 1
[7] J. Fu, J. Liu, H. Tian, Y. Li, Y. Bao, Z. Fang, and H. Lu, “Dual attention
network for scene segmentation,” in Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, 2019, pp. 3146–
3154. 1
[8] J. Fu, J. Liu, J. Jiang, Y. Li, Y. Bao, and H. Lu, “Scene segmentation with
dual relation-aware attention network,” IEEE Transactions on Neural
Networks and Learning Systems, vol. 32, no. 6, pp. 2547–2560, 2020.
1

[CAPTION] Fig. 7. Ablation results of CIFAR100 on semi-supervised learning method.


<!-- page 13 -->
IN PREPARATION FOR SUBMITTING TO IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS
13
[9] K. Zhu, Y. Cao, W. Zhai, J. Cheng, and Z.-J. Zha, “Self-promoted
prototype reﬁnement for few-shot class-incremental learning,” in CVPR,
2021, pp. 6801–6810. 1, 2, 3, 4, 7, 8, 9, 10
[10] X. Tao, X. Hong, X. Chang, S. Dong, X. Wei, and Y. Gong, “Few-shot
class-incremental learning,” in CVPR, 2020, pp. 12 183–12 192. 1, 2, 3,
6, 7, 9, 10, 12
[11] C. Zhang, N. Song, G. Lin, Y. Zheng, P. Pan, and Y. Xu, “Few-shot
incremental learning with continually evolved classiﬁers,” in CVPR,
2021, pp. 12 455–12 464. 1, 2, 3, 4, 6, 7, 9, 10
[12] K. Chen and C.-G. Lee, “Incremental few-shot learning via vector
quantization in deep embedded space,” in ICLR, 2021. 1, 2, 9
[13] D.-W. Zhou, F.-Y. Wang, H.-J. Ye, L. Ma, S. Pu, and D.-C.
Zhan, “Forward compatible few-shot class-incremental learning,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2022, pp. 9046–9056. 1, 2, 9
[14] Z. Chi, L. Gu, H. Liu, Y. Wang, Y. Yu, and J. Tang, “Metafscil:
A meta-learning approach for few-shot class incremental learning,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2022, pp. 14 166–14 175. 1, 2, 9
[15] H. Liu, L. Gu, Z. Chi, Y. Wang, Y. Yu, J. Chen, and J. Tang, “Few-shot
class-incremental learning via entropy-regularized data-free replay,” in
European Conference on Computer Vision.
Springer, 2022, pp. 146–
162. 1, 2, 9
[16] T. Chen, S. Wu, X. Yang, Y. Xu, and H.-S. Wong, “Semantic
regularized class-conditional gans for semi-supervised ﬁne-grained
image synthesis,” IEEE Transactions on Multimedia, 2021. 1, 2
[17] G. I. Parisi, R. Kemker, J. L. Part, C. Kanan, and S. Wermter, “Continual
lifelong learning with neural networks: A review,” Neural Networks, vol.
113, pp. 54–71, 2019. 1
[18] J. E. Van Engelen and H. H. Hoos, “A survey on semi-supervised
learning,” Machine Learning, vol. 109, no. 2, pp. 373–440, 2020. 1,
4
[19] Y. Cui, W. Xiong, M. Tavakolian, and L. Liu, “Semi-supervised few-
shot class-incremental learning,” in ICIP.
IEEE, 2021, pp. 1239–1243.
1, 3, 4, 5, 6, 7, 9, 10
[20] Y. Cui, W. Deng, X. Xu, Z. Liu, Z. Liu, M. Pietik¨ainen, and
L. Liu, “Uncertainty-guided semi-supervised few-shot class-incremental
learning with knowledge distillation,” IEEE Transactions on Multimedia,
2022. 1, 2, 3, 7, 9
[21] S.-A. Rebufﬁ, A. Kolesnikov, G. Sperl, and C. H. Lampert, “icarl:
Incremental classiﬁer and representation learning,” in CVPR, 2017, pp.
2001–2010. 2, 5, 6, 8, 9
[22] B. Zhang, Y. Wang, W. Hou, H. Wu, J. Wang, M. Okumura, and
T. Shinozaki, “Flexmatch: Boosting semi-supervised learning with
curriculum pseudo labeling,” NeurIPS, vol. 34, 2021. 2, 3, 5
[23] N. Lai, M. Kan, C. Han, X. Song, and S. Shan, “Learning to learn
adaptive classiﬁer–predictor for few-shot learning,” IEEE transactions
on neural networks and learning systems, vol. 32, no. 8, pp. 3458–3470,
2020. 2
[24] H.-G.
Jung
and
S.-W.
Lee,
“Few-shot
learning
with
geometric
constraints,” IEEE Transactions on Neural Networks and Learning
Systems, vol. 31, no. 11, pp. 4660–4672, 2020. 2
[25] X. Yu and Y. Aloimonos, “Attribute-based transfer learning for object
categorization with zero/one training example,” in ECCV, 2010, pp. 127–
140. 2
[26] O. Vinyals, C. Blundell, T. Lillicrap, D. Wierstra et al., “Matching
networks for one shot learning,” in NeurPIS, 2016, pp. 3630–3638. 2,
6
[27] J. Snell, K. Swersky, and R. Zemel, “Prototypical networks for few-shot
learning,” in NeurIPS, 2017, pp. 4080–4090. 2
[28] C. Finn, P. Abbeel, and S. Levine, “Model-agnostic meta-learning for
fast adaptation of deep networks,” in ICML, 2017, pp. 1126–1135. 2
[29] M. A. Jamal and G.-J. Qi, “Task agnostic meta-learning for few-shot
learning,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2019, pp. 11 719–11 727. 2
[30] A. Santoro, S. Bartunov, M. Botvinick, D. Wierstra, and T. Lillicrap,
“Meta-learning with memory-augmented neural networks,” in ICML,
2016, pp. 1842–1850. 2
[31] S. Hou, X. Pan, C. C. Loy, Z. Wang, and D. Lin, “Learning a uniﬁed
classiﬁer incrementally via rebalancing,” in CVPR, 2019, pp. 831–839.
2, 5, 6, 7, 9, 12
[32] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins,
A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska
et al., “Overcoming catastrophic forgetting in neural networks,”
Proceedings of the national academy of sciences, vol. 114, no. 13, pp.
3521–3526, 2017. 2
[33] H. Zhao, H. Wang, Y. Fu, F. Wu, and X. Li, “Memory efﬁcient class-
incremental learning for image classiﬁcation,” IEEE Transactions on
Neural Networks and Learning Systems, 2021. 2
[34] D.-W. Zhou, Y. Yang, and D.-C. Zhan, “Learning to classify with
incremental new class,” IEEE Transactions on Neural Networks and
Learning Systems, 2021. 2
[35] Y. Liu, X. Hong, X. Tao, S. Dong, J. Shi, and Y. Gong, “Model behavior
preserving for class-incremental learning,” IEEE Transactions on Neural
Networks and Learning Systems, 2022. 2
[36] A. Cheraghian, S. Rahman, P. Fang, S. K. Roy, L. Petersson, and
M. Harandi, “Semantic-aware knowledge distillation for few-shot class-
incremental learning,” in CVPR, 2021, pp. 2534–2543. 3, 9
[37] Z. Song, X. Yang, Z. Xu, and I. King, “Graph-based semi-supervised
learning: A comprehensive review,” IEEE Transactions on Neural
Networks and Learning Systems, 2022. 3
[38] D. Berthelot, N. Carlini, I. Goodfellow, N. Papernot, A. Oliver, and C. A.
Raffel, “Mixmatch: A holistic approach to semi-supervised learning,” in
NeurIPS, 2019. 3, 12
[39] D.-H. Lee et al., “Pseudo-label: The simple and efﬁcient semi-supervised
learning method for deep neural networks,” in ICML Workshop, vol. 3,
no. 2, 2013. 3
[40] K. Sohn, D. Berthelot, N. Carlini, Z. Zhang, H. Zhang, C. A. Raffel,
E. D. Cubuk, A. Kurakin, and C.-L. Li, “Fixmatch: Simplifying semi-
supervised learning with consistency and conﬁdence,” NeurIPS, vol. 33,
pp. 596–608, 2020. 3, 12
[41] A. Iscen, G. Tolias, Y. Avrithis, and O. Chum, “Label propagation for
deep semi-supervised learning,” in CVPR, 2019, pp. 5070–5079. 3
[42] M. N. Rizve, K. Duarte, Y. S. Rawat, and M. Shah, “In defense of
pseudo-labeling: An uncertainty-aware pseudo-label selection frame-
work for semi-supervised learning,” arXiv preprint arXiv:2101.06329,
2021. 3
[43] X. Zhu, Z. Ghahramani, and J. D. Lafferty, “Semi-supervised learning
using gaussian ﬁelds and harmonic functions,” in ICML, 2003, pp. 912–
919. 3
[44] B. Liu, Z. Wu, H. Hu, and S. Lin, “Deep metric transfer for label
propagation with limited annotated data,” in ICCV Workshop, 2019, pp.
0–0. 3
[45] A. Kendall and Y. Gal, “What uncertainties do we need in bayesian
deep learning for computer vision?” Advances in neural information
processing systems, vol. 30, 2017. 3
[46] C. Louizos and M. Welling, “Structured and efﬁcient variational deep
learning with matrix gaussian posteriors,” in ICML, 2016, pp. 1708–
1716. 3
[47] Y. Gal and Z. Ghahramani, “Dropout as a bayesian approximation:
Representing model uncertainty in deep learning,” in ICML, 2016, pp.
1050–1059. 3
[48] G. Wang, W. Li, S. Ourselin, and T. Vercauteren, “Automatic brain
tumor segmentation using convolutional neural networks with test-
time augmentation,” in International MICCAI Brainlesion Workshop.
Springer, 2018, pp. 61–72. 3
[49] J. Gawlikowski, C. R. N. Tassi, M. Ali, J. Lee, M. Humt, J. Feng,
A. Kruspe, R. Triebel, P. Jung, R. Roscher et al., “A survey of uncertainty
in deep neural networks,” arXiv preprint arXiv:2107.03342, 2021. 3, 5
[50] Y. Grandvalet and Y. Bengio, “Semi-supervised learning by entropy
minimization,” in NeurIPS, 2004, pp. 529–536. 4, 9
[51] A. Krizhevsky, G. Hinton et al., “Learning multiple layers of features
from tiny images,” 2009. 6
[52] A. Chaudhry, M. Ranzato, M. Rohrbach, and M. Elhoseiny, “Efﬁcient
lifelong learning with a-gem,” in ICLR, 2018. 6
[53] H. Robbins and S. Monro, “A stochastic approximation method,” The
annals of mathematical statistics, pp. 400–407, 1951. 6
[54] F. M. Castro, M. J. Mar´ın-Jim´enez, N. Guil, C. Schmid, and K. Alahari,
“End-to-end incremental learning,” in ECCV, 2018, pp. 233–248. 9
[55] S. Dong, X. Hong, X. Tao, X. Chang, X. Wei, and Y. Gong, “Few-shot
class-incremental learning via relation knowledge distillation,” in AAAI,
2021, pp. 1255–1263. 9
[56] A. Cheraghian, S. Rahman, S. Ramasinghe, P. Fang, C. Simon,
L. Petersson, and M. Harandi, “Synthesized feature based few-shot class-
incremental learning on a mixture of subspaces,” in ICCV, 2021, pp.
8661–8670. 9
[57] V. D. M. Laurens and G. Hinton, “Visualizing data using t-sne,” The
Journal of Machine Learning Research, vol. 9, p. 2579–2605, 2008. 7
[58] G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural
network,” in NeurIPS Workshop, 2015. 8