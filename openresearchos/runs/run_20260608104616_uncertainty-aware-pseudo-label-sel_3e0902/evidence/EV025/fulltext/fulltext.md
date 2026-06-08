<!-- page 1 -->
Graphical Abstract
CRMSP: A Semi-supervised Approach for Key Information Ex-
traction with Class-Rebalancing and Merged Semantic Pseudo-
Labeling
Qi Zhang, Yonghong Song, Pengcheng Guo, Yangyang Hui
arXiv:2407.15873v1  [cs.LG]  19 Jul 2024


<!-- page 2 -->
Highlights
CRMSP: A Semi-supervised Approach for Key Information Ex-
traction with Class-Rebalancing and Merged Semantic Pseudo-
Labeling
Qi Zhang, Yonghong Song, Pengcheng Guo, Yangyang Hui
• We propose a semi-supervised approach for KIE with Class-Rebalancing
and Merged Semantic Pseudo-Labeling (CRMSP), utilizing a large
number of unlabeled documents, significantly reducing the annotation
costs, and improving the generalizability of the model.
• To solve the problem of underestimation on the confidence of tail classes
in the long-tailed distribution, we proposed the Class-Rebalancing Pseudo-
Labeling (CRP) module.
• We propose the Merged Semantic Pseudo-Labeling (MSP) module to
fickle the difficulty in achieving intra-class compactness and inter-class
separability of tail classes in unlabeled feature space.


<!-- page 3 -->
CRMSP: A Semi-supervised Approach for Key
Information Extraction with Class-Rebalancing and
Merged Semantic Pseudo-Labeling
Qi Zhang, Yonghong Song, Pengcheng Guo, Yangyang Hui
aSchool of Software Engineering, Xi’an Jiaotong University, No.28, Xianning West
Road, Xi’an City, 710049, Shaanxi Province, China
Abstract
There is a growing demand in the field of KIE (Key Information Extraction)
to apply semi-supervised learning to save manpower and costs, as training
document data using fully-supervised methods requires labor-intensive man-
ual annotation. The main challenges of applying SSL in the KIE are (1) un-
derestimation of the confidence of tail classes in the long-tailed distribution
and (2) difficulty in achieving intra-class compactness and inter-class separa-
bility of tail features. To address these challenges, we propose a novel semi-
supervised approach for KIE with Class-Rebalancing and Merged Semantic
Pseudo-Labeling (CRMSP). Firstly, the Class-Rebalancing Pseudo-Labeling
(CRP) module introduces a reweighting factor to rebalance pseudo-labels,
increasing attention to tail classes. Secondly, we propose the Merged Se-
mantic Pseudo-Labeling (MSP) module to cluster tail features of unlabeled
data by assigning samples to Merged Prototypes (MP). Additionally, we de-
signed a new contrastive loss specifically for MSP. Extensive experimental
results on three well-known benchmarks demonstrate that CRMSP achieves
state-of-the-art performance. Remarkably, CRMSP achieves 3.24% f1-score
improvement over state-of-the-art on the CORD.
Keywords:
semi-supervised learning, key information extraction,
long-tailed distribution, semantic pseudo-labeling
1. Introduction
Key Information Extraction (KIE) as the downstream task of Optical
Character Recognition (OCR) is the process of extracting structured infor-
Preprint submitted to NEUROCOMPUTING
July 24, 2024


<!-- page 4 -->
(a) Class distribution
(b) Dilemma of tail classes
Figure 1: Analysis on class distribution, and the dilemma of tail classes.
mation from documents. KIE generally includes tasks such as named entity
recognition and relation extraction, structured information extraction, and
document classification. KIE has various applications in real-life scenarios,
including bill processing, medical record handling, contract analysis, and re-
sume processing. KIE is a challenging task since documents involve different
types of information, including images, text, and layout. Recently, many
multimodal pre-trained methods [1, 2, 3] for KIE have been proposed to
fickle this problem. However, these multimodal pre-trained methods require
annotation for multiple types of information, which further increases time
and manpower costs.
Semi-supervised learning (SSL) [4] tackles situations with limited labeled
and abundant unlabeled data [5, 6, 7, 8, 9], bridging the gap between su-
pervised and unsupervised learning for enhanced model performance. Exist-
ing SSL approaches [5, 10] are to perform consistency regularization between
weakly and strongly augmented views of unlabeled data based on the pseudo-
labels predicted by the model as targets, thereby mitigating the model’s sen-
sitivity to small variations in similar samples within the input space. The
performance of these SSL methods based on consistency regularization de-
pends on whether the classes are balanced and the intra-class compactness
and inter-class separability of the model in the feature space.
Specifically, the first one is that the confidence of tail classes in the long-
tailed distribution [11] is underestimated, leading to the model exhibiting
higher confidence in predicting samples from the head classes. As shown in
Fig. 1(a), both labeled and unlabeled data exhibit a long-tailed distribution.
This phenomenon implies that pseudo-labels are more likely to belong to head
2

[CAPTION] Figure 1: Analysis on class distribution, and the dilemma of tail classes.


<!-- page 5 -->
{C0-C9}
{C10-C19}
{C20-C29}
Class index
0.0
0.2
0.4
0.6
0.8
1.0
Precision
FixMatch(avg:0.86)
CRMSP(avg:0.89)
(a) Precision of PL
{C0-C9}
{C10-C19}
{C20-C29}
Class index
0.0
0.2
0.4
0.6
0.8
1.0
Recall
FixMatch(avg:0.86)
CRMSP(avg:0.90)
(b) Recall of PL
{C0-C9}
{C10-C19}
{C20-C29}
Class index
0.0
0.2
0.4
0.6
0.8
1.0
F1-score
FixMatch(avg:0.86)
CRMSP(avg:0.89)
(c) F1-score of PL
Figure 2: Comparison of precision, recall and f1-score of pseudo-labels generated by Fix-
Match and CRMSP. ”PL” represents Pseudo-Labels.
classes with higher probabilities and less likely to belong to tail classes with
lower probabilities. As the number of iterations increases, this imbalance in
the long-tailed distribution tends to worsen.
Secondly, it is hard to achieve intra-class compactness and inter-class
separability of tail classes in unlabeled feature space [12]. To help the model
learn richer representations, prototypes [13, 14] are commonly introduced into
the model. Each prototype can be seen as the representative features of a
specific class of samples. The model calculates semantic pseudo-labels based
on these prototypes. Following this, consistency regularization is applied to
learn and enhance features. As shown in Fig. 1(b), when classifying yellow
unlabeled samples of tail classes, the sample points are closer to the ”green”
prototype than to ”yellow”. This results in ”yellow” sample points being
pushed away from the true direction and towards the wrong direction in
the feature space. Therefore, the existence of the deviation angle θ causes
semantic pseudo-labels to be biased to head classes over the tail classes, and
this imbalance will increase as training progresses, leading to a degradation
in model performance.
To address these challenges, we propose a semi-supervised approach for
KIE with Class-Rebalancing and Merged Semantic Pseudo-Labeling (CRMSP).
Firstly, to augment the model’s attention toward tail classes, we introduce the
Class-Rebalancing Pseudo-Labeling (CRP) module that enhances the weight
of pseudo-labels of tail classes and reduces the weight of pseudo-labels of head
classes with a reweighting factor. It improves the recall of tail classes com-
pared to the classical SSL method FixMatch [5] while maintaining a high
precision of head classes, resulting in a higher f1-score, as shown in Fig. 2.
Secondly, to enhance the intra-class compactness and separability from
3


**[Table p5.1]**
|  |  | FixMatch(avg:0.86) CRMSP(avg:0.89) |
| --- | --- | --- |


**[Table p5.2]**
|  |  | FixMatch(avg:0.86) CRMSP(avg:0.90) |
| --- | --- | --- |


**[Table p5.3]**
|  |  | FixMatch(avg:0.86) CRMSP(avg:0.89) |
| --- | --- | --- |

[CAPTION] Figure 2: Comparison of precision, recall and f1-score of pseudo-labels generated by Fix-


<!-- page 6 -->
other classes of tail classes, we propose the Merged Semantic Pseudo-Labeling
(MSP) module. This module sorts the generated pseudo-labels in descending
order, identifies the Top-K classes, and aggregates the features of these K
classes in the memory bank into a super-class. The merged prototype (MP)
is then calculated.
By utilizing the clustering of merged prototypes, the
semantic pseudo-labels generated in this way push the features of tail samples
closer to the prototype of the super-class, rather than pushing the features
closer to prototypes of head classes.
Additionally, we designed a new contrastive loss specifically for the merged
semantic pseudo-labels, whose effectiveness is demonstrated in Table 4. Ex-
tensive experimental results indicate that the MSP module improves the
performance of tail classes.
To the best of our knowledge, CRMSP is the first semi-supervised learn-
ing method in the field of KIE. Based on a multi-modal model, CRMSP
has designed a semi-supervised learning method that fully utilizes text, im-
age, and layout information, which is different from previous semi-supervised
methods that only utilize image or text information from CV or NLP models.
The main contributions are summarized as follows:
• We propose a semi-supervised approach for KIE with Class-Rebalancing
and Merged Semantic Pseudo-Labeling (CRMSP), utilizing a large
number of unlabeled documents, significantly reducing the annotation
costs, and improving the generalizability of the model.
• To solve the problem of underestimation of the confidence of tail classes
in the long-tailed distribution, we proposed the Class-Rebalancing Pseudo-
Labeling (CRP) module.
• We propose the Merged Semantic Pseudo-Labeling (MSP) module to
fickle the difficulty in achieving intra-class compactness and inter-class
separability of tail classes in unlabeled feature space.
2. Related Work
2.1. Key information extraction
Transformer-based pre-training has demonstrated success across various
KIE tasks, where extensive unlabeled document datasets are leveraged for
model pre-training, preceding fine-tuning on downstream tasks. Numerous
existing frameworks [15, 1, 2, 3] have investigated pre-training approaches
4


<!-- page 7 -->
on documents. LayoutLM [15] achieved significant improvements in various
document understanding tasks by jointly pre-training text and layout. Lay-
outLMv2 [1] greatly enhanced the model’s image understanding capability
by integrating visual feature information into the pre-training process. Lay-
outLMv3 [2] overcame the differences between text and image in pre-training
objectives and promoted multi-modal representation learning. Our approach
utilizes these multi-modal models based on image, text, and layout as en-
coders, extending the scope of semi-supervised methods to the KIE domain.
2.2. Semi-supervised learning
SSL is a learning approach focused on building models that leverage both
labeled and unlabeled data. While unlabeled data is crucial for SSL, gener-
ating pseudo-labels from model predictions remains a challenge. Existing ap-
proaches, including pseudo-labeling [16], consistency regularization [17, 18],
generative methods [19, 20] and hybrid methods [21, 5, 6, 7, 8, 9]. However,
pseudo-labels can introduce bias, particularly in the presence of imbalanced
data, adversely affecting model performance. To mitigate this issue, pre-
vious works have explored various strategies such as threshold adjustment
[6, 7, 9], incorporating additional classifiers [22, 14]. However, designing dy-
namic thresholds is complex and computationally intensive. In our work, we
directly incorporate an additional branch for semantic pseudo-label classi-
fication, which effectively promotes intra-class compactness and inter-class
separability for imbalanced classes, without the need for designing complex
dynamic threshold strategies.
2.3. Imbalanced learning
Class-imbalanced supervised learning is of great interest both in theory
and in practice. Recent works include resampling [23, 24] and reweighting
[25] which rebalance the contribution of each class, while others focus on
reweighting the given loss function by a factor inversely proportional to the
sampling frequency in a class-wise manner. [25] proposed a suppressed con-
sistency loss to suppress the loss on minority classes.[26] proposed Distribu-
tion Aligning Refinery (DARP) to refine pseudo-labels for SSL under assum-
ing class-imbalanced training distributions. CReST proposed a re-sampling
method to iteratively refine the model by supplementing the labeled set with
high-quality pseudo-labels, where minority classes are updated more aggres-
sively than majority classes. DASO adaptively blends the linear and seman-
tic pseudo-labels within each class to mitigate the overall bias across the
5


<!-- page 8 -->
class for imbalanced semi-supervised learning. In our work, we alleviate the
class-imbalanced problem by directly rebalancing pseudo-labels according to
distributions between head and tail classes instead of designing complicated
reweighting losses.
3. Proposed Method
3.1. Preliminaries
For a C-class semi-supervised classification problem, let X = {(xb, yb)}B
b=1
be a batch of B labeled samples, where xb are the training samples and yb
are the ground-truth, yb ∈Y = {1, . . . , C}. Meanwhile, let U = {ub}µB
b=1
be a batch of µB unlabeled samples, where the hyperparameter µ is used
to control the batch size of unlabeled samples. Note that the underlying
ground truth ˆy of unlabeled data may be different from labeled data, ˆy ∈Y,
Y = {1, . . . , C}.
For the labeled data, the input xb is paired with the label yb to train the
base model f(·) through calculating supervised loss Lsup, generating features
zb. For the unlabeled data, unlabeled samples are sent to the base model
f(·) as inputs after weak augmentation Aw and strong augmentation As.
Both are followed by a classification head h(·) and a projection head g(·) to
get pw = h ◦f(Aw(u)), zw = g ◦f(Aw(u)), ps = h ◦f(As(u)) and zs =
g ◦f(As(u)). The Class-Rebalancing Pseudo-labeling module is employed to
alleviate the imbalance problem of pseudo-labels. The rebalanced pseudo-
labels ˆp ∈RC are then assigned to calculate the unsupervised loss Lun.
The Merged Semantic Pseudo-Labeling module generates merged semantic
pseudo-labels of unlabeled features with the Merged Prototypes ˆC, which is
used to compute the contrastive loss Lctr. The overall framework is shown
in Fig. 3.
3.2. Class-Rebalancing Pseudo-Labeling
Due to the smaller sample size in the tail classes compared to the head
classes, the model tends to generate lower confidence when predicting tail
data.
The approach in FixMatch [5], which filters out samples based on
a fixed threshold applied to the highest confidence, overlooks the numerical
disadvantage of the tail data. Experiments show that the predictive distribu-
tion of labeled samples is generally positively correlated with the distribution
of unlabeled samples. We estimate the approximate sample distribution by
calculating the exponential moving average (EMA) of the model’s confidence
6


<!-- page 9 -->
Labeled
Unlabeled
Strongly-
aug
Weakly-
aug
p
zw
y
......
......
pw
zs
ps
CRP Module
MSP Module
pˆ
Lctr
Lctr
Lun
Lun
Lsup
Lsup
Reweight
Reverse
       
       
top-K
......
Merged 
Prototype
Merged 
Prototype
top-K
Merged 
Prototype
top-K
transformer
encoder
transformer
encoder
transformer
encoder
{                
"bbox": {
"x2": 456,
"y3": 799,
…,
"x4": 325,
"y2": 765
},
...,
"text": "Bremer"
"label": "menu.nm",
}
label, bbox, 
image, …, 
text
bbox, image, 
…, text
bbox, image, 
…, text
Cls
Cls
Proj
Proj
Proj
Proj
Cls
Cls
Lsup: supervised loss
Lctr: contrastive loss
Lun: unsupervised loss
MSP: Merged Semantic Pseudo-Labeling
CRP: Class-Rebalancing Pseudo-Labeling
Lsup: supervised loss
Lctr: contrastive loss
Lun: unsupervised loss
MSP: Merged Semantic Pseudo-Labeling
CRP: Class-Rebalancing Pseudo-Labeling
Labeled
Unlabeled
Strongly-
aug
Weakly-
aug
p
zw
y
......
pw
zs
ps
CRP Module
MSP Module
pˆ
Lctr
Lun
Lsup
Reweight
Reverse
       
top-K
...
Merged 
Prototype
top-K
transformer
encoder
transformer
encoder
transformer
encoder
{                
"bbox": {
"x2": 456,
"y3": 799,
…,
"x4": 325,
"y2": 765
},
...,
"text": "Bremer"
"label": "menu.nm",
}
label, bbox, 
image, …, 
text
bbox, image, 
…, text
bbox, image, 
…, text
Cls
Proj
Proj
Cls
Lsup: supervised loss
Lctr: contrastive loss
Lun: unsupervised loss
MSP: Merged Semantic Pseudo-Labeling
CRP: Class-Rebalancing Pseudo-Labeling
Figure 3: Framework of the proposed Class-Rebalancing and Merged Semantic Pseudo-
Labeling (CRMSP). Labeled and unlabeled samples are from the training data mini-batch.
predictions for labeled data p, which we call ˜p. Note that EMA preserves
previous information through a weighted average, smoothing the process of
updating data. At the t-th iteration, we compute ˜pt as:
˜pt =
(
1
K,
if t = 0,
λ˜pt−1 + (1 −λ) 1
B
PB
b=1 pb,
otherwise,
(1)
where λ is a smoothing factor.
3.2.1. Reweighting pseudo-labels
We observed that the pseudo-labels produced by the model are biased
toward the head classes.
To augment the model’s attention towards tail
classes, we introduced a reweighting factor β that enhances the weight of tail
classes while correspondingly reducing that of head classes.
We first perform the reverse operation on ˜pt to obtain the reweighting
7

[CAPTION] Figure 3: Framework of the proposed Class-Rebalancing and Merged Semantic Pseudo-


<!-- page 10 -->
factor β:
β = Normalize(M(˜pt)),
(2)
M is a monotonically decreasing mapping function (a minimum value not less
than 0) that ensures a higher weight is assigned to classes with a smaller pro-
portion in the predicted distribution, while classes with a larger proportion
in the predicted distribution receive a lower weight. e.g., M(x) = 1 −x/T ,
where T is a temperature hyperparameter, Normalize(·) is the normalized
operation defined as x′
i = xi/Pn
j=1xj, i ∈(1, . . . , n).
Then the model’s confidence predictions pw
b for weakly-augmented data
is multiplied by β to get more balanced confidence predictions p′
b in a batch:
p′
b = Normalize(pw
b × β)
(3)
The rebalanced pseudo-labels ˆpb are generated by argmax(·) in a batch:
ˆpb = argmax(p′
b).
(4)
By rebalancing the pseudo-labels, tail labels are more likely to be chosen
when filtering pseudo-labels with a fixed threshold for prediction.
3.3. Merged Semantic Pseudo-Labeling
To obtain semantic pseudo-labels from a feature perspective, DASO [14]
involves prototype clustering, which updates the dynamic memory bank with
features and ground-truths of labeled data. However, due to the significantly
smaller number of tail samples compared to head classes, the tail features in
this memory bank are inherently limited. Consequently, the computed tail
prototypes lack representation, and it is inappropriate to assume that all tail
features are concentrated around this prototype. The semantic pseudo-labels
are computed by merely comparing tail samples to this unrepresentative
prototype push tail features close to prototypes of other classes, which is
detrimental to achieving intra-class compactness and inter-class separability
of tail classes.
3.3.1. EMA Model
The basic assumption in SSL is the smoothness assumption: if two data
points are close in high-density regions, their corresponding outputs should
also be close.
Mean Teacher [17] utilizes this assumption by using unla-
beled data. In practice, augmented samples are generated by adding small
8


<!-- page 11 -->
perturbations to the original samples, and they should have consistent pre-
dictions in both Teacher and Student models, achieved through consistency
constraints. The Teacher model is essentially an EMA model of the Student.
The EMA model provides guidance for updating the parameters of the base
model. Therefore, their weights are tightly coupled. The parameter (θ′) of
the EMA model is updated with a weighted average of the current parameter
(θ) of the base model:
θ′
t = α · θ′
t−1 + (1 −α) · θ
(5)
where α is a smoothing factor (0 < α < 1).
3.3.2. Merged Prototypes (MP) generation
We first build a set of basic prototypes C = {ci}C
i=1 from X. The basic
prototype ci for every class is efficiently calculated by averaging the feature
representations in the dynamic memory bank Q = {Qi}C
i=1, Qi = {zj}maxsize
j=1
,
where Qi is a queue with a max size. We update Q every iteration by pushing
new features zb and labels yb from a batch of labeled data.
Then we determine to construct the super-class for each batch. Based
on a common understanding: for an unlabeled sample, if the confidences of
several classes are close, their corresponding feature representations in the
feature space are close.
In such cases, we merge these top-K proximate
classes. We achieve this by sorting the confidence predictions pw
b obtained
from the weak augmentation branch in descending order, resulting in the
sorted confidence predictions sw
b :
sw
b = sort(pw
b , descending) = {pw
σ(1), . . . , pw
σ(K), . . . , pw
σ(C)}
(6)
where {σ(1), . . . , σ(C)} represents the order in which classes are arranged in
descending order.
After obtaining this order, we merge the features corresponding to the
top-K classes Qσ(K) in the dynamic memory bank C to get the new dynamic
memory bank is ˆQ = { ˆQi}N
i=1, where N = C −K + 1:
ˆQi =
(
Qσ(1) ∪Qσ(2) ∪. . . ∪Qσ(K),
if i = 1
Qσ(K+i−1),
otherwise
(7)
For Merged prototypes ˆC = {ˆci}N
i=1, each ˆci is computed by taking the
average of the features in the queue ˆQi.
9


<!-- page 12 -->
3.3.3. Semantic pseudo-labels
In a batch, the merged semantic predictions qw
b and qs
b of the super class
space is computed from zw
b , zs
b with the merged prototype ˆC:
qw
b = Sim(zw
b , ˆC) = < zw
b , ˆC >
∥zw
b ∥∥ˆC ∥
/Tproto
(8)
qs
b = Sim(zs
b, ˆC) = < zs
b, ˆC >
∥zs
b ∥∥ˆC ∥
/Tproto
(9)
where Tproto is a temperature hyperparameter, and Sim(·) denotes cosine
similarity, < ·, · > represents the dot product operation, and ∥· ∥represents
the L2 norm.
The merged semantic pseudo-labels ˆqb are generated by argmax(·) in a
batch:
ˆqb = argmax(qw
b ).
(10)
3.4. Loss function
Following the SSL paradigm, the first two items are supervised loss Lsup
and unsupervised loss Lun, respectively. In addition, we include a contrastive
loss Lctr to compute the distance of two semantic similarities.
The loss
minimized by CRMSP is simply:
Ltotal = Lsup + λunLun + λctrLctr,
(11)
Lsup = 1
B
B
X
b=1
H(pb, yb),
(12)
Lun =
1
µB
µB
X
b=1
1(max(p′
b) ≥τ)H(ˆpb, ps
b),
(13)
Lctr =
1
µB
µB
X
b=1
H(ˆqb, qs
b).
(14)
where λun and λctr are fixed scalar hyperparameters denoting the relative
weight of the unsupervised loss and contrastive loss, and 1 is all-one vector,
τ represents a fixed threshold. H represents cross-entropy.
10


<!-- page 13 -->
Data: A batch of labeled and unlabeled samples X = {(xb, yb)}B
b=1
and U = {ub}µB
b=1, base model f(·), classification head and
projection head: h(·) and g(·)
Input
: λun: weight of the unsupervised loss, λctr: weight of the
contrastive loss, τ: a fixed threshold
Output: Trained model parameters
1 for epoch in range(num epochs) do
2
for batch in unlabeled data batches do
3
{Supervised Learning with labeled data};
4
pb = h ◦f(xb), ˜pt =
(
1
K,
if t = 0,
λ˜pt−1 + (1 −λ) 1
B
PB
b=1 pb,
otherwise,;
5
Lsup = 1
B
PB
b=1 H(pb, yb);
6
{Unsupervised Learning with unlabeled data};
7
pw
b = h ◦f(Aw(ub)), ps
b = h ◦f(As(ub));
8
β = Normalize(M(˜pt)), p′
b = Normalize(pw
b × β);
9
ˆpb = argmax(p′
b);
10
Lun =
1
µB
PµB
b=1 1(max(p′
b) ≥τ)H(ˆpb, ps
b);
11
{Contrastive Learning};
12
zw
b = g ◦f(Aw(ub)), zs
b = g ◦f(As(ub));
13
Q = {Qi}C
i=1, Qi ←(zb, yb);
14
C = {ˆci}N
i=1 ←the average of the features Q;
15
sw
b = sort(pw
b , descending) = {pw
σ(1), . . . , pw
σ(K), . . . , pw
σ(C)},
{σ(1), . . . , σ(C)} represents the descending order;
16
ˆQ = { ˆQi}N
i=1, ˆC = {ˆci}N
i=1 ←the average of the features ˆQ;
17
qw
b =
<zw
b ,ˆC>
∥zw
b ∥∥ˆC∥/Tproto, qs
b =
<zs
b,ˆC>
∥zs
b∥∥ˆC∥/Tproto;
18
ˆqb = argmax(qw
b );
19
Lctr =
1
µB
PµB
b=1 H(ˆqb, qs
b);
20
Ltotal = Lsup + λunLun + λctrLctr;
21
update f(·), h(·), g(·) with AdamW to minimize Ltotal;
22
end
23 end
Algorithm 1: CRMSP algorithm
11


<!-- page 14 -->
3.5. Pseudo Code
Algorithm 1 presents the algorithm of the entire CRMSP during the train-
ing phase.
4. Experiments
4.1. Datasets and Compared Methods
4.1.1. Datasets
FUNSD [27] is a comprehensive collection of real, fully annotated, scanned
forms with 149 samples for training and 50 samples for testing. The docu-
ments are noisy and vary widely in appearance, making form understanding
a challenging task. The proposed dataset can be used for various tasks, in-
cluding text detection, optical character recognition, spatial layout analysis,
and entity. CORD [28] is typically utilized for receipt KIE, which consists of
800/100/100 receipts for training/validation/testing. The dataset consists of
thousands of Indonesian receipts, which contain images and box/text anno-
tations for OCR, and multi-level semantic labels for parsing. The proposed
dataset can be used to address various OCR and parsing tasks.
We construct long-tailed versions of CIFAR10 (CIFAR10-LT), CIFAR100
(CIFAR100-LT), and STL10 (STL10-LT) separately. Nk represents the num-
ber of examples in class k for labeled data and unlabeled data. γl and γu
are the imbalance ratios for labeled data and unlabeled data. The number
of examples for each class except the head class is based on the formula
Nk = N1 · γ
−k−1
K−1
l
. It is important to note that within each Nk, examples in
class k are arranged in descending order (i.e., N1 ≥· · · ≥NK).
4.1.2. Compared Methods
Our approach is compared with both classical and imbalanced SSL meth-
ods. Classical SSL methods include FixMatch [5], Dash [6], FlexMatch [7],
SimMatch [8] and FreeMatch [9]. Imbalanced SSL methods include DARP
[26], CReST [24] and DASO [14].
4.2. Experiments settings
Our experiments are conducted on NVIDIA Tesla V100 GPU. For KIE
datasets, the split ratios (i.e., the proportions of labeled data) are 5% and
10%, and the batch size of labeled data is 4, µ is set to 1.0. To validate the
effectiveness of our proposed SSL approach, we use Transformer-based models
12


<!-- page 15 -->
Base Model
Method
FUNSD
CORD
5%
10%
5%
10%
P
R
F1
P
R
F1
P
R
F1
P
R
F1
LayoutLMv2
Fully-supervised
77.91
81.38
79.61
77.91
81.38
79.61
95.81
95.02
95.41
95.81
95.02
95.41
FixMatch [5]
61.42
70.55
65.67
65.27
73.36
69.08
83.96
85.36
84.66
87.01
87.47
87.24
Dash [6]
62.71
74.51
68.10
64.77
70.65
67.58
77.99
82.64
80.25
87.47
88.53
88.00
FlexMatch [7]
59.14
75.66
66.39
62.85
72.40
67.29
84.60
84.60
84.60
86.84
87.17
87.01
SimMatch [8]
63.29
70.85
66.86
70.06
77.72
73.69
82.83
83.40
83.11
88.21
89.21
88.71
FreeMatch [9]
64.06
70.65
67.19
68.32
74.46
71.26
82.79
84.60
83.69
88.81
89.28
89.05
CRMSP(Ours)
63.39
71.85
67.36
71.49
78.63
74.89
84.47
84.98
84.73
90.46
90.19
90.33
LayoutLMv3
Fully-supervised
77.74
80.33
79.01
77.74
80.33
79.01
93.09
93.51
93.30
93.09
93.51
93.30
FixMatch [5]
57.40
67.46
62.02
62.74
73.87
67.85
82.73
83.17
82.95
86.72
87.25
86.98
Dash [6]
53.53
63.34
58.02
61.91
67.41
64.54
82.11
83.47
82.78
83.77
86.87
85.29
FlexMatch [7]
60.04
70.54
64.87
70.85
73.77
72.28
80.06
82.42
81.22
86.10
86.49
86.30
SimMatch [8]
64.78
66.68
65.72
69.40
77.40
73.18
82.07
82.57
82.32
87.72
87.85
87.78
FreeMatch [9]
60.37
72.88
66.04
67.17
71.14
69.10
82.06
82.87
82.46
87.08
89.06
88.06
CRMSP(Ours)
63.89
71.98
67.69
67.68
74.91
71.12
82.41
84.15
83.27
91.51
91.09
91.30
Table 1: Evaluation results with 5% and 10% of labeled training samples on three KIE
benchmarks based on LayoutLMv2 and LayoutLMv3.
The best result is in bold, the
second-best result is in underline.
such as LayoutLMv2 and LayoutLMv3 as base models. For all methods, we
employ the Adam optimizer with a fixed learning rate of 1e-5.
we take
precision, recall and f1-score as our evaluation metrics. Referring to [29], we
set weak augmentation to none and strong augmentation to random swap,
which means randomly swapping two words in the sentence n times.
In
training, temperature hyperparameter Tproto is set to 1.0. The EMA model
is used for testing with a momentum factor 0.999. For two unsupervised
supervised losses, both λun and λctr are set to 0.1, τ is set to 0.95. For both
LayoutLMv2 and LayoutLMv3, K was fine-tuned with a setting of 5.
For CV datasets, we train the CIFAR10-LT/CIFAR100-LT and STL10-
LT on the Wide ResNet-28-2 [26] with 1.5M parameters for 250k iterations.
The optimizer is SGD with a learning rate of 0.03 and a weight decay of 5e-4.
We set the training epoch to 256 and the batch size to 64. The imbalance
ratio includes the following settings: γ = γl = γu = 100, γ = γl = γu = 10
and γl = 10, γu : unknown. The number of examples for the head class of
labeled data N1 is set to {500, 1500, 50, 150, 450}, and the number of examples
for the head class of unlabeled data M1 is set to {4000, 3000, 400, 300, 100k}.
4.3. Results
4.3.1. Results on FUNSD and CORD
To validate the effectiveness of CRMSP, we perform experiments on FUNSD
and CORD for the token classification task.
Table 1 shows comparative
13

[CAPTION] Table 1: Evaluation results with 5% and 10% of labeled training samples on three KIE

[CAPTION] Table 1 shows comparative


<!-- page 16 -->
Method
Dataset
CIFAR10-LT
CIFAR100-LT
STL10-LT
imb ratio
γ = γl = γu = 100
γ = γl = γu = 10
γl = 10, γu : unknown
#Label
N1 = 500
N1 = 1500
N1 = 50
N1 = 150
N1 = 150
N1 = 450
#Unlabel
M1 = 4000
M1 = 3000
M1 = 400
M1 = 300
M1 = 100k
M1 = 100k
Supervised
46.75
62.78
31.11
49.02
45.39
62.09
Classical
SSL
FixMatch [5]
73.60
77.60
48.52
57.85
65.80
77.85
Dash [6]
70.63
75.45
42.93
56.01
73.40
82.01
FlexMatch [7]
62.18
73.69
38.99
53.88
82.75
85.55
SimMatch [8]
76.03
78.68
47.30
57.86
82.85
85.75
FreeMatch [9]
70.63
76.41
44.16
57.20
82.34
86.06
Imbalanced
SSL
DARP [26]
75.92
78.62
49.05
57.88
65.38
75.70
CReST [24]
71.21
75.74
44.37
55.77
63.58
71.70
DASO [14]
71.12
76.79
48.35
57.71
68.37
79.03
CRMSP(Ours)
71.38
78.91
44.53
57.90
83.01
86.35
Table 2: Top-1 accuracy (%) on CIFAR10-LT, CIFAR100-LT and STL10-LT. The best
result is in bold, the second-best result is in underline.
results with 5% and 10% labeled samples based on LayoutLMv2 and Lay-
outLMv3. For all methods, we observe that the f1-score increases as the
ratio of labeled data increases. In contrast, CRMSP improves the f1-score in
the vast majority of experimental settings. Based on LayoutLMv3, it works
particularly well on the CORD with 10% labeled data and achieves 4.32%
and 3.24% f1-score gain compared with FixMatch and suboptimal method
FreeMatch, respectively. On the FUNSD, our proposed CRMSP achieved
an improvement of f1-score that were 5.81% and 1.20% higher compared
to FixMatch and the suboptimal model SimMatch based on LayoutLMv2,
respectively.
This indicates that CRMSP can more effectively utilize labeled data to
reduce model bias under long-tailed distribution. Furthermore, we observe
that our method even achieves an f1-score of 91.30%, which is close to the
f1-score of 93.30% achieved by the fully-supervised LayoutLMv3 on the KIE
task, while using only 10% of the labeled data.
4.3.2. Results on CIFAR10/100-LT and STL10-LT
To illustrate the generalization of CRMSP, we also conducted experi-
ments on the CIFAR10/100-LT and STL10-LT, as shown in Table 2. We
consider rebalancing biased pseudo-labels by matching (e.g., γ = γl = γu)
or mismatching (e.g., γl = 10, γu : unknown) distributions between imbal-
anced labeled and unlabeled data (X and U) in Table 2. When γl = γu, we
compare the proposed CRMSP with several classical (i.e., FixMatch [5] and
imbalanced (i.e., DARP [26], CReST [24] and DASO [14]) SSL baseline meth-
14

[CAPTION] Table 2: Top-1 accuracy (%) on CIFAR10-LT, CIFAR100-LT and STL10-LT. The best


<!-- page 17 -->
Method
Class
0
1
2
3
4
5
6
7
8
9
Supervised
86.1
61.0
81.5
49.4
38.8
23.8
42.9
18.5
29.0
23.0
Classical SSL
FixMatch [5]
73.8
4.7
0.2
48.2
80.4
53.9
0.2
1.1
42.1
38.1
Dash [6]
94.1
92.0
96.1
60.0
86.3
70.5
83.8
50.0
69.0
32.3
FlexMatch [7]
95.4
83.1
94.6
54.4
85.8
69.9
89.6
80.3
94.0
80.5
SimMatch [8]
92.0
85.0
94.4
58.0
87.9
73.0
90.4
69.0
95.0
84.3
FreeMatch [9]
94.5
84.3
94.1
46.3
90.3
67.8
88.9
77.6
95.8
84.0
Imbalanced SSL
DARP [26]
93.3
82.6
93.3
54.4
81.0
34.9
79.9
26.4
64.5
43.6
CReST [24]
55.3
37.3
58.6
59.5
68.4
20.9
73.5
53.1
78.6
82.1
DASO [14]
90.8
75.3
92.3
52.9
82.6
53.8
82.8
31.0
72.1
50.4
CRMSP(Ours)
93.0
82.6
94.3
56.0
89.0
73.8
90.5
68.9
95.0
87.1
Table 3: Per-class top-1 accuracy (%) on the balanced test dataset of STL10-LT (γl =
10, γu : unknown, N1 = 150, M1 = 100k). Our method shows a significant improvement
in pseudo-labeling for tail classes. The best result is in bold, the second-best result is in
underline.
ods. In the supervised scenario, the performance is relatively constrained
compared to other semi-supervised learning methods.
Notably, CRMSP demonstrates comparable or even superior performance
across most settings, exhibiting substantial improvements compared to the
baseline methods SimMatch and DARP. Compared to the baseline FixMatch
and baseline DARP, the accuracy has increased by 0.04%-0.60% and 0.02%-
17.63%, respectively.
Overall, Table 2 demonstrates that our proposed CRMSP is not only
effective in mitigating imbalance issues in the SSL domain of KIE but also
applicable in the CV domain.
4.3.3. Per-class performance
By comparing the top-1 accuracy of different methods on the STL10-LT
across per class, we designate classes C0-C4 as the head classes and C5-
C9 as the tail classes. In the head classes, it can be observed that Dash
performs well on some head classes (C1, C2, C3), while our method achieves
the second-best accuracy in C4 with 89.0%. In the tail classes, our method
achieves an accuracy of 95.0% for C8, which is second-best compared to
FreeMatch. Remarkably, our method CRMSP achieves the best accuracy
of 73.8%, 90.5%, and 87.1% in C5, C6, and C9, respectively, representing
improvements of 0.8%, 0.1%, and 2.8% over the second-best class SimMatch.
This table highlights the enhancement provided by our method for tail classes
in long-tailed distribution.
15

[CAPTION] Table 3: Per-class top-1 accuracy (%) on the balanced test dataset of STL10-LT (γl =


<!-- page 18 -->
#
RP
Lctr
MP
FUNSD
CORD
0
66.56
84.05
1
✓
67.21
84.76
2
✓
✓
67.75
85.36
3
✓
68.96
87.56
4
✓
✓
69.91
88.08
5
✓
✓
✓
71.12
89.59
Table 4: Ablation study on FUNSD and CORD. “RP”, “Lctr” and “MP” mean Reweight-
ing Pseudo-Labels, Contrastive Loss and Merged Prototypes, respectively. We use super-
vised loss as baseline.
Class
menu.sub price
menu.sub cnt
total.creditcardprice
sub total.service price
menu.num
sub total.etc
sub total.discount price
support
20
17
16
12
11
8
7
w/o RP
0.72
0.42
0.46
0.88
0.40
0.33
0.73
w RP
0.88
0.78
0.73
0.96
0.74
0.62
0.88
Table 5: The influence of the RP on the f1-score of tail classes on the CORD, ”support”
denotes the sample size.
4.4. Ablation study
To verify the effectiveness of each component of our proposed method,
We conduct extensive ablation studies on the CORD. The results are shown
in Table 4. And the evaluation metric for all experiments was the f1-score.
We utilize LayoutLMv3 as the base model.
4.4.1. Effectiveness of Reweighting Pseudo-Labels
To verify the effectiveness of RP, By comparing Experiment 2 and Ex-
periment 5 in Table 4, we can observe that the RP improves the f1-score by
3.37% and 4.23% on the FUNSD and CORD. Table 5 We also compared the
performance of tail classes without and with RP. It is found that by adding
RP, there is a significant improvement in the results of the tail classes, espe-
cially for num.sub cnt (0.42→0.78). Note that LayoutLMv2 is used as the
base model.
To demonstrate the detailed effectiveness of our proposed RP, we present
confusion matrixes of the predictions on the test dataset of FUNSD. As de-
picted in Fig. 4, the pseudo-labels for the tail classes without RP (e.g., C4, C5
and C6) are underestimated, while the accuracy between the pseudo-labels
and the true labels for the head classes is higher. Our proposed RP improves
the generation of more balanced pseudo-labels for tail classes, alleviating the
issue of long-tailed distribution.
16

[CAPTION] Table 4: Ablation study on FUNSD and CORD. “RP”, “Lctr” and “MP” mean Reweight-

[CAPTION] Table 5: The influence of the RP on the f1-score of tail classes on the CORD, ”support”


<!-- page 19 -->
4.4.2. Effectiveness of Contrastive Loss
When incorporating contrastive loss, CRMSP can further boost the per-
formance on all settings by another few points, resulting in 0.52% to 0.66%
absolute accuracy improvement by comparing Experiment 3 and Experiment
4 in Table 4.
4.4.3. Effectiveness of Merged Prototypes
Comparing Experiment 4 and Experiment 5 in Table 4, we observed that
f1-score improved by 1.21% and 1.51% on the FUNSD and CORD, respec-
tively.
To demonstrate the effectiveness of our proposed MP, we present the
comparison of t-SNE visualization of unlabeled data. As shown in Fig. 5,
the MP helps the tail class (e.g., C6) to be separated from the confusion class
and better clustering is achieved. Other confusing features (e.g., C0 and C1)
are also clustered more compactly. Fig. 5 effectively promotes intra-class
compactness and inter-class separability of unlabeled tail classes in feature
space.
Figure 4: Confusion matrixes of the predic-
tions on the test dataset of FUNSD.
Figure 5: Comparison of t-SNE visualiza-
tion of unlabeled data on the FUNSD.
4.4.4. Ablation study on λun
In Fig. 6, we study the effect of the temperature hyper-parameter λun
to compute the weights for unsupervised loss described in Eq. 11. We em-
pirically find that, for both FUNSD and CORD, λun = 0.1 shows the best
performance.
4.4.5. Ablation study on λctr
In Fig. 7, we investigate the impact of the temperature hyper-parameter
λctr on computing the weights for the contrastive loss described in Eq. 11.
λctr = 0.1 yields the optimal performance on the FUNSD and CORD.
17

[CAPTION] Figure 4: Confusion matrixes of the predic-

[CAPTION] Figure 5: Comparison of t-SNE visualiza-


<!-- page 20 -->
4.4.6. Ablation study on K
The influence of different K on the f1-score on the FUNSD and CORD
is illustrated in Fig. 8. We notice that K = 5 provides the best f1-score
among all tested values. When K is set to a small value, prototypes for some
tail samples lack representation. Comparing these tail samples with non-
representative prototypes results in semantic pseudo-labels that push the
feature in the wrong direction in the feature space, leading to classification
errors. On the other hand, if K is set too large, although the new sample
features are effectively separated from classes not belonging to this super-
class, the super feature range extends far beyond the range of variations in
tail features, causing internal confusion within the super-class.
Figure 6:
The influence of
different λun on the f1-score
on the FUNSD and CORD.
Figure 7:
The influence of
different λctr on the f1-score
on the FUNSD and CORD.
Figure 8:
The influence of
different K on the f1-score on
the FUNSD and CORD.
4.4.7. Case Study
We present the output of samples for Ground-truth, FixMatch, and CRMSP
on both the CORD and FUNSD. On the CORD, as depicted in Fig. 9, the
tail class sub total.discount price in the ground-truth is incorrectly classified
as total.total etc by FixMatch. This misclassification is corrected by our pro-
posed CRMSP approach. The tail classes menu.sub cnt and menu.sub nm
are erroneously associated with their respective classes menu.cnt and menu.nm,
but our proposed CRMSP method adeptly distinguishes between them.
On the FUNSD, as shown in Fig. 10, FixMatch misclassifies B-QUESTION
and I-QUESTION as B-ANSWER and I-ANSWER, respectively. However,
CRMSP correctly identifies these tail classes.
18

[CAPTION] Figure 6:
The influence of

[CAPTION] Figure 7:
The influence of

[CAPTION] Figure 8:
The influence of


<!-- page 21 -->
·
(a) Ground-truth
(b) FixMatch
(c) CRMSP
Figure 9: Example output of Ground-truth, FixMatch, and CRMSP for CORD.
19

[CAPTION] Figure 9: Example output of Ground-truth, FixMatch, and CRMSP for CORD.


<!-- page 22 -->
(a) Ground-truth
(b) FixMatch
(c) CRMSP
Figure 10: Example output of Ground-truth, FixMatch, and CRMSP for FUNSD.
5. Conclusion
In this paper, we propose a novel semi-supervised approach for key in-
formation extraction with Class-Rebalancing and Merged Semantic Pseudo-
Labeling (CRMSP). Firstly, the Class-Rebalancing Pseudo-Labeling (CRP)
module is proposed to directly rebalance pseudo-labels with a reweighting
factor, increasing attention to tail classes. Secondly, the Merged Semantic
Pseudo-Labeling (MSP) module is proposed to achieve intra-class compact-
ness and inter-class separability of unlabeled tail classes in feature space
by assigning samples to Merged Prototypes (MP). We even achieved close
to fully-supervised learning in the semi-supervised setting.
Extensive ex-
20

[CAPTION] Figure 10: Example output of Ground-truth, FixMatch, and CRMSP for FUNSD.


<!-- page 23 -->
perimental results have demonstrated the proposed CRMSP surpasses other
state-of-the-art methods on three benchmarks. Our findings suggest that
the proposed approach can obtain high-quality pseudo-labels from a larger
amount of unlabeled data, which provides a good solution for semi-supervised
learning.
References
[1] Y. Xu, Y. Xu, T. Lv, L. Cui, F. Wei, G. Wang, Y. Lu, D. Florencio,
C. Zhang, W. Che, et al., Layoutlmv2: Multi-modal pre-training for
visually-rich document understanding, arXiv preprint arXiv:2012.14740
(2020).
[2] Y. Huang, T. Lv, L. Cui, Y. Lu, F. Wei, Layoutlmv3: Pre-training for
document ai with unified text and image masking, in: ACM MM, 2022,
pp. 4083–4091.
[3] P. Guo, Y. Song, Y. Deng, K. Xie, M. Xu, J. Liu, H. Ren, Dcmai: A
dynamical cross-modal alignment interaction framework for document
key information extraction, IEEE TCSVT (2023).
[4] X. J. Zhu, Semi-supervised learning literature survey (2005).
[5] K. Sohn, D. Berthelot, N. Carlini, Z. Zhang, H. Zhang, C. A. Raf-
fel, E. D. Cubuk, A. Kurakin, C.-L. Li, Fixmatch: Simplifying semi-
supervised learning with consistency and confidence, NeurIPS 33 (2020)
596–608.
[6] Y. Xu, L. Shang, J. Ye, Q. Qian, Y.-F. Li, B. Sun, H. Li, R. Jin, Dash:
Semi-supervised learning with dynamic thresholding, in: ICML, PMLR,
2021, pp. 11525–11536.
[7] B. Zhang, Y. Wang, W. Hou, H. Wu, J. Wang, M. Okumura, T. Shi-
nozaki, Flexmatch: Boosting semi-supervised learning with curriculum
pseudo labeling, NeurIPS 34 (2021) 18408–18419.
[8] M. Zheng, S. You, L. Huang, F. Wang, C. Qian, C. Xu, Simmatch: Semi-
supervised learning with similarity matching, in: Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2022, pp. 14471–14481.
21


<!-- page 24 -->
[9] Y. Wang, H. Chen, Q. Heng, W. Hou, Y. Fan, Z. Wu, J. Wang, M. Sav-
vides, T. Shinozaki, B. Raj, et al., Freematch: Self-adaptive thresholding
for semi-supervised learning, arXiv preprint arXiv:2205.07246 (2022).
[10] Q. Xie, Z. Dai, E. Hovy, T. Luong, Q. Le, Unsupervised data augmen-
tation for consistency training, NeurIPS 33 (2020) 6256–6268.
[11] Y. Zhang, B. Kang, B. Hooi, S. Yan, J. Feng, Deep long-tailed learning:
A survey, IEEE TPAMI (2023).
[12] X. Huang, C. Zhu, W. Chen, Semi-supervised domain adaptation via
prototype-based multi-level learning, arXiv preprint arXiv:2305.02693
(2023).
[13] M. Caron, I. Misra, J. Mairal, P. Goyal, P. Bojanowski, A. Joulin, Unsu-
pervised learning of visual features by contrasting cluster assignments,
NeurIPS 33 (2020) 9912–9924.
[14] Y. Oh, D.-J. Kim, I. S. Kweon, Daso: Distribution-aware semantics-
oriented pseudo-label for imbalanced semi-supervised learning, in: Pro-
ceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2022, pp. 9786–9796.
[15] Y. Xu, M. Li, L. Cui, S. Huang, F. Wei, M. Zhou, Layoutlm: Pre-
training of text and layout for document image understanding, in: Pro-
ceedings of the 26th ACM SIGKDD International Conference on Knowl-
edge Discovery & Data Mining, 2020, pp. 1192–1200.
[16] Q. Xie, M.-T. Luong, E. Hovy, Q. V. Le, Self-training with noisy stu-
dent improves imagenet classification, in: Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, 2020, pp. 10687–
10698.
[17] A. Tarvainen, H. Valpola, Mean teachers are better role models: Weight-
averaged consistency targets improve semi-supervised deep learning re-
sults, Advances in neural information processing systems 30 (2017).
[18] S. Laine, T. Aila, Temporal ensembling for semi-supervised learning,
arXiv preprint arXiv:1610.02242 (2016).
22


<!-- page 25 -->
[19] H. Fang, W. Deng, Y. Zhong, J. Hu, Triple-gan: Progressive face aging
with triple translation loss, in: Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition workshops, 2020, pp. 804–
805.
[20] Z. Liu, J. Wang, Z. Liang, Catgan: Category-aware generative adver-
sarial networks with hierarchical evolutionary learning for category text
generation, in: Proceedings of the AAAI Conference on Artificial Intel-
ligence, Vol. 34, 2020, pp. 8425–8432.
[21] D. Berthelot,
N. Carlini,
E. D. Cubuk,
A. Kurakin,
K. Sohn,
H. Zhang, C. Raffel, Remixmatch:
Semi-supervised learning with
distribution alignment and augmentation anchoring, arXiv preprint
arXiv:1911.09785 (2019).
[22] C.-W. Kuo, C.-Y. Ma, J.-B. Huang, Z. Kira, Featmatch: Feature-based
augmentation for semi-supervised learning, in: Computer Vision–ECCV
2020: 16th European Conference, Glasgow, UK, August 23–28, 2020,
Proceedings, Part XVIII 16, Springer, 2020, pp. 479–495.
[23] N. V. Chawla, K. W. Bowyer, L. O. Hall, W. P. Kegelmeyer, Smote:
synthetic minority over-sampling technique, Journal of artificial intelli-
gence research 16 (2002) 321–357.
[24] C. Wei, K. Sohn, C. Mellina, A. Yuille, F. Yang, Crest:
A class-
rebalancing self-training framework for imbalanced semi-supervised
learning, in: CVPR, 2021, pp. 10857–10866.
[25] M. Hyun, J. Jeong, N. Kwak, Class-imbalanced semi-supervised learn-
ing, arXiv preprint arXiv:2002.06815 (2020).
[26] J. Kim, Y. Hur, S. Park, E. Yang, S. J. Hwang, J. Shin, Distribution
aligning refinery of pseudo-label for imbalanced semi-supervised learn-
ing, NeurIPS 33 (2020) 14567–14579.
[27] G. Jaume, H. K. Ekenel, J.-P. Thiran, Funsd: A dataset for form un-
derstanding in noisy scanned documents, in: ICDARW, Vol. 2, IEEE,
2019, pp. 1–6.
23


<!-- page 26 -->
[28] S. Park, S. Shin, B. Lee, J. Lee, J. Surh, M. Seo, H. Lee, Cord: a consol-
idated receipt dataset for post-ocr parsing, in: Workshop on Document
Intelligence at NeurIPS 2019, 2019.
[29] Y. Wang, H. Chen, Y. Fan, W. Sun, R. Tao, W. Hou, R. Wang, L. Yang,
Z. Zhou, L.-Z. Guo, et al., Usb: A unified semi-supervised learning
benchmark for classification, NeurIPS 35 (2022) 3938–3961.
24