<!-- page 1 -->
Semi-supervised learning for medical image
classiﬁcation using imbalanced training data
Tri Huynha,∗, Aiden Nibalia, Zhen Hea
aDepartment of Computer Science and Information Technology, La Trobe University,
Melbourne, Australia
Abstract
Medical image classiﬁcation is often challenging for two reasons: a lack of la-
belled examples due to expensive and time-consuming annotation protocols, and
imbalanced class labels due to the relative scarcity of disease-positive individu-
als in the wider population. Semi-supervised learning (SSL) methods exist for
dealing with a lack of labels, but they generally do not address the problem of
class imbalance. In this study we propose Adaptive Blended Consistency Loss
(ABCL), a drop-in replacement for consistency loss in perturbation-based SSL
methods. ABCL counteracts data skew by adaptively mixing the target class
distribution of the consistency loss in accordance with class frequency. Our ex-
periments with ABCL reveal improvements to unweighted average recall on two
diﬀerent imbalanced medical image classiﬁcation datasets when compared with
existing consistency losses that are not designed to counteract class imbalance.
Keywords:
Semi supervised learning; Medical Imaging; Class Imbalance
∗Corresponding author
Email address: t.huynh@latrobe.edu.au (Tri Huynh)
Preprint submitted to Elsevier
August 23, 2021
arXiv:2108.08956v1  [cs.CV]  20 Aug 2021


<!-- page 2 -->
1. Introduction
Computer-aided diagnosis is an important ﬁeld of research, as well-written
algorithms can improve diagnostic eﬃciency and reduce the chance of misdiag-
nosis. For example, at the early stages tumors or lesions might be very small
and easy for a radiologist to miss, but by using an algorithm to automatically
highlight such instances the number of false negatives can be reduced. Recently,
many supervised machine learning methods, especially those using convolutional
neural networks (CNNs) have achieved promising results in medical image clas-
siﬁcation, an important component of computer-aided diagnosis (Rezvantalab
et al., 2018; Han et al., 2018; Bi et al., 2017; Hekler et al., 2019; Iizuka et al.,
2020; Ausawalaithong et al., 2018; Rajpurkar et al., 2017). However supervised
learning requires a large number of fully labelled medical images. In the real
world, annotating medical images is typically very time-consuming, especially
when the consensus of multiple experts is required. Fortunately, in many prac-
tical situations there are often many unlabelled images available that can be
leveraged to boost model accuracy. Semi-supervised learning (SSL) (Chapelle
et al., 2009) makes use of a small amount of labelled and a large amount of
unlabelled data to train machine learning models. The goal of SSL is to learn
patterns from unlabelled data to enhance the accuracy of a model that is trained
using a small amount of labelled data. There are several variants of SSL deal-
ing with medical image classiﬁcation tasks which have achieved promising re-
sults such as Categorical Generative Adversarial Networks (Springenberg, 2015),
Denoising Adversarial AutoEncoders (Creswell et al., 2018), Dual-path Semi-
supervised Conditional Generative Adversarial Networks (Yang et al., 2019),
and the MeanTeacher with Label Propagation algorithm (Su et al., 2019). Re-
cently researchers have found perturbation based methods give the best per-
formance (Xie et al., 2019; Tarvainen and Valpola, 2017; Laine and Aila, 2016;
Miyato et al., 2018; Berthelot et al., 2019). Hence in this study we will focus
on perturbation based methods. Most perturbation based algorithms augment
unlabelled data and then apply a consistency loss to make the predicted class
2


<!-- page 3 -->
distribution from the original and augmented (perturbed) unlabelled samples
similar.
There are several techniques based on this approach which recently achieved
state of the art results, mostly on various classiﬁcation tasks, such as Tempo-
ral Ensembling (Laine and Aila, 2016), Mean Teacher (Tarvainen and Valpola,
2017), Virtual Adversarial Training (Miyato et al., 2018), Unsupervised Data
Augmentation (UDA) (Xie et al., 2019) and MixMatch (Berthelot et al., 2019).
Medical image datasets often have a very skewed data distribution, usually
due to a large number of negative disease cases versus a small number of positive
disease cases. It is critically important to address this class imbalance problem
since failure to address this problem can lead to false negatives for the minor
class (the disease-positive case) which can have potentially fatal consequences.
This motivates us to study techniques for dealing with skewed data in SSL.
Almost all existing approaches for handling skewed data distributions are
only for labelled datasets including: various oversampling or undersampling
approaches (Mani and Zhang, 2003; Kubat et al., 1997; Barandela et al., 2004;
Chawla et al., 2002; Han et al., 2005; Bunkhumpornpat et al., 2009); modiﬁed
loss functions (Wang et al., 2016; Lin et al., 2017; Phan et al., 2017); cost
sensitive learning (Wang et al., 2018; Khan et al., 2017; Zhang et al., 2016);
decision threshold moving (Krawczyk and Wo´zniak, 2015; Chen et al., 2006;
Yu et al., 2016). In our case we would like to develop eﬀective methods for
addressing the skewed class distribution of unlabelled data for SSL.
We focus our study on modifying the consistency loss (loss on unlabelled
data) on one of the best performing recent SSL methods, Unsupervised Data
Augmentation (UDA) (Xie et al., 2019).
Although our study is focused on
UDA, our proposed method can be applied to any perturbation based SSL
which uses the consistency loss such as Temporal Ensembling (Laine and Aila,
2016), Mean Teacher (Tarvainen and Valpola, 2017) and Virtual Adversarial
Training (Miyato et al., 2018).
The standard consistency loss (CL) used in
UDA has two shortcomings: 1) it degrades the classiﬁcation performance of the
minor classes, and 2) it always sets the target as the original sample instead of
3


<!-- page 4 -->
a blend of the augmented and original sample. Hence, we propose the Adaptive
Blended Consistency Loss (ABCL) method which tackles both shortcomings
of standard consistency loss by generating a target class distribution which is a
blend of the original and augmented class distributions. The blended target class
distribution skews towards either the original or augmented sample depending
on which predicted the minor class.
To our knowledge the only existing work that addresses the class distri-
bution skew problem for SSL perturbation based methods is the Suppressed
Consistency Loss (SCL) method (Hyun et al., 2020). Their observation is that
for class imbalanced datasets, the training is likely to move the decision bound-
ary into the low-density areas of the minor class which will in turn cause the
model to misclassify minor class samples. Therefore, SCL suppresses the con-
sistency loss of minor classes which will help move the decision boundary away
from the low-density areas that the minor classes occupy. However SCL, like
the standard consistency loss of UDA, also sets its target as the original sample
instead of a blend of the augmented and original samples. This will in general
favor the major class since the models are more likely to predict the major class
as a result of the bias inherent in the unbalanced dataset itself. In contrast, our
ABCL uses a blended target class distribution that skews towards either the
original or augmented sample depending on which predicts the minor class.
We have conducted an extensive experimental study on a skin cancer HAM10000
dataset (Tschandl et al., 2018) and a retinal fundus glaucoma REFUGE dataset (Or-
lando et al., 2020). Our experiments show that when our ABCL method is used
to replace the consistency loss of UDA, the unweighted average recall (UAR) on
HAM10000 increases from 0.59 to 0.67. Furthermore, ABCL also signiﬁcantly
outperformed the state-of-the-art SCL method for the skin cancer dataset. For
the retinal fundus glaucoma dataset, ABCL signiﬁcantly outperformed SCL,
increasing UAR from 0.57 to 0.67. These results show that ABCL is able to
improve performance of SSL for diﬀerent datasets.
This paper makes the following key contributions:
4


<!-- page 5 -->
• We identify the importance of handling class imbalance for semi-supervised
classiﬁcation of medical images. In contrast, no existing work explicitly
addresses this problem for medical images.
• We propose the Adaptive Blended Consistency Loss (ABCL) as a replace-
ment for the consistency loss of perturbation based SSL algorithms such
as UDA in order to tackle the class imbalance problem in SSL.
• We conduct extensive experiments on two diﬀerent datasets to demon-
strate the advantages of ABCL over standard consistency loss and other
existing methods designed for addressing the class imbalance problem in
both supervised learning and SSL.
2. Material and methods
2.1. Semi-supervised learning architecture
We base our research on perturbation based SSL methods since all state
of the art SSL methods (Xie et al., 2019; Tarvainen and Valpola, 2017; Laine
and Aila, 2016; Miyato et al., 2018; Berthelot et al., 2019) use this approach.
The solution we developed for this paper can be applied to any perturbation
based SSL that uses the consistency loss. To simplify our analysis we will focus
on a particular perturbation based SSL called the Unsupervised Data Augmen-
tation (Xie et al., 2019) (UDA) method. UDA is one of the best performing
recent SSL methods. Fig. 1 shows a diagram of how UDA works. The model
is trained using two losses: a supervised loss (cross-entropy loss) and an unsu-
pervised loss (consistency loss). The aim of consistency loss is to enforce the
consistency of two prediction distributions. The key idea of UDA is to use op-
timal data augmentation on unlabelled samples to increase the eﬀectiveness of
the consistency loss. To obtain optimal data augmentation they applied an al-
gorithm called RandAugmentation (Cubuk et al., 2019) on the labelled dataset.
5


<!-- page 6 -->
Fig. 1: The architecture used by the Unsupervised Data Augmentation (UDA)
(Xie et al.,
2019) perturbation SSL method. In the diagram M is the shared CNN model used for classi-
fying both the labelled and unlabelled images.
The total loss for the UDA architecture consists of two terms (see
Fig. 1):
supervised loss for labelled data and consistency loss for unlabelled data. The
loss formula can be summarized as follows:
L = LS(pθ(y|l)) + Lcon(pθ(y|u), pθ(y|ˆu))
(1)
Where LS is a supervised cross entropy loss function that takes input as the
predicted probability distribution pθ(y|l) of y for a labelled sample l produced
by the model M with parameters θ. Lcon is a consistency loss function that uses
the Kullback-Leibler divergence to steer the predicted class distribution of the
augmented unlabelled image ˆu towards the target predicted class distribution
of the original unlabelled image u.
A key assumption of perturbation based methods such as UDA is the smooth-
ness assumption. Under the smoothness assumption, two data points that are
close in the input space should have the same label. That means, an unlabelled
sample that is close to a labelled example will have the label information propa-
gated to the unlabelled sample. Another important property of the smoothness
assumption is that the original input and the augmented input should be close
6

[CAPTION] Fig. 1: The architecture used by the Unsupervised Data Augmentation (UDA)


<!-- page 7 -->
to each other in the embedded space and hence should be assigned the same la-
bel. This idea is captured by the consistency loss Lcon which ensures the model
produces the same predicted probabilities from both the original and augmented
input, leading to the model being robust to noise.
There have been various existing approaches (Mani and Zhang, 2003; Ku-
bat et al., 1997; Barandela et al., 2004; Chawla et al., 2002; Han et al., 2005;
Bunkhumpornpat et al., 2009; Wang et al., 2016; Lin et al., 2017; Phan et al.,
2017; Wang et al., 2018; Khan et al., 2017; Zhang et al., 2016; Krawczyk and
Wo´zniak, 2015; Chen et al., 2006; Yu et al., 2016) to tackle class imbalance
for the supervised learning problem. These approaches are complementary to
our solutions since they work on improving supervised loss while ours works
to improve the unsupervised loss component of the overall loss. The unsuper-
vised loss is particularly important since the number of unlabelled examples is
typically much larger than labelled examples. As shown in Fig. 1, UDA uses
the consistency loss to exploit the information from the unlabelled loss by mak-
ing the class distribution of the augmented unlabelled data match the original
unlabelled data. Hence to alleviate the class imbalance problem in UDA, we
focus on modifying UDA’s consistency loss (shown in
Eq. 1). In particular,
UDA’s consistency loss is replaced by our new novel loss function, Adaptive
Blended Consistency Loss (ABCL). Hence the total loss is reformulated with
ABCL replacing the unsupervised term:
L = LS(pθ(y|l)) + ABCL(pθ(y|u), pθ(y|ˆu))
(2)
2.2. Issues with existing consistency loss formulations
In this section, we analyse problems with standard consistency loss (CL)
in UDA (Xie et al., 2019) and the state-of-the-art suppressed consistency loss
(SCL) (Hyun et al., 2020) when training on datasets with imbalanced class
distributions.
The problems will be analysed in the context of the original
sample’s prediction (OSP) and augmented sample’s prediction (ASP) which
represent the probability distributions pθ(y|u) and pθ(y|ˆu) respectively in Eq. 1
7


<!-- page 8 -->
Fig. 2: An illustration showing how CL and SCL works. Note that the target class distribution
is always the class distribution for OSP. In addition, SCL down-weights the loss if the predicted
class of OSP is the minor class.
Fig. 2 illustrates how CL and SCL works. In UDA, the CL is a function
that sets the OSP as the target for ASP. That is the CL always pushes the class
distribution of ASP towards the class distribution of OSP. The idea behind SCL
is to suppress the minor class‘s consistency loss to move the decision boundary
such that it passes through a low-density region of the latent space. In practical
terms, SCL suppresses the CL when the OSP is the minor class and applies the
CL when the OSP is the major class. Like CL, SCL uses the OSP’s class dis-
tribution as the target irrespective of whether OSP and ASP class distributions
belong to the major or minor class.
Both standard CL and SCL have two shortcomings. Firstly they are both
biased towards targeting the major class in the presence of imbalanced training
data. Secondly they do not target a blend of OSP and ASP but instead always
target OSP only, and thus do not exploit the augmented example to improve
the model’s behaviour for the original example.
Shortcoming 1: CL and SLC are more likely to target the ma-
jor class. When in doubt the model will more often predict the major class
since the model is trained on labelled data which is skewed towards major class
samples. Consequently, samples of the minor class are more likely to be mispre-
8

[CAPTION] Fig. 2: An illustration showing how CL and SCL works. Note that the target class distribution

[CAPTION] Fig. 2 illustrates how CL and SCL works. In UDA, the CL is a function


<!-- page 9 -->
dicted as the major class than vice versa. In particular, when the original sample
is incorrectly predicted as major class and the augmented sample is correctly
predicted as minor class, the CL and SCL erroneously encourages the model to
predict the sample as major class. As a result, the model’s performance will
degrade for minor class samples.
Shortcoming 2: CL and SLC do not target a blend of OSP and
ASP but instead always targets OSP only. Targeting a blend of OSP
and ASP reduces the harmful eﬀects of targeting OSP only when OSP predicts
the wrong class. Unlike existing methods, we do not make the assumption that
the model’s prediction for the original sample is more accurate than for the
augmented sample. Instead we set the target as a blend of OSP and ASP to
potentially average out some of the harmful eﬀects of a wrong OSP prediction.
This is in some ways similar to the beneﬁt of ensembling the prediction of
multiple models to minimize prediction error and the established technique of
test-time data augmentation (Krizhevsky et al., 2012).
The shortcomings mentioned above are all centered around what we should
set as the target distribution for the CL as a function of OSP and ASP. Hence,
to provide a more detailed analysis of the desired target class distribution we
divide the analysis into four separate cases depending on whether major or
minor classes were predicted by the OSP and ASP. See Tab. 1 and Fig. 3 for
an illustration of the 4 diﬀerent OSP and ASP cases. In cases 1 and 2, OSP
and ASP are in agreement. Unlike CL and SCL, we forgo the assumption that
the OSP is in any way more valid than the ASP. As a result, we posit that
it may be better to move the desired target class distribution to be a blend
of OSP and ASP instead of just to OSP. This is because we would like the
target to contain information in the predictions of both OSP and ASP. In case
3 CL and SCL may unintentionally encourage the bias towards the major class
to be stronger since it is moving the desired target class distribution towards
the major class although there is no consensus between the two predictions.
There is already a natural tendency to predict the major class, a bias which is
induced by the dataset skew. CL/SCL does nothing to counteract this bias as
9


<!-- page 10 -->
Case
OSP
ASP
CL target
SCL target
ABCL tar-
get
1
Major class
Major class
OSP (major
class)
OSP (major
class)
Near
the
middle
be-
tween
OSP
and ASP
2
Minor class
Minor class
OSP (minor
class)
OSP (minor
class)
Near
the
middle
be-
tween
OSP
and ASP
3
Major class
Minor class
OSP (major
class)
OSP (major
class)
Closer
to-
wards ASP
4
Minor class
Major class
OSP (minor
class)
OSP (minor
class)
Closer
to-
wards OSP
Tab. 1: The analysis of the target class distribution of CL (standard consistency loss), SCL
(suppressed consistency loss), and our ABCL (adaptive blended consistency loss) for 4 pre-
diction cases.
Fig. 3 gives a diagrammatic illustration of the 4 cases.
Fig. 3: An illustration of the target class distribution of CL, SCL and ABCL for the 4 cases
shown in Tab. 1.
10


**[Table p10.1]**
| Case | OSP | ASP | CL target | SCL target | ABCL tar- get |
| --- | --- | --- | --- | --- | --- |
| 1 | Major class | Major class | OSP (major class) | OSP (major class) | Near the middle be- tween OSP and ASP |
| 2 | Minor class | Minor class | OSP (minor class) | OSP (minor class) | Near the middle be- tween OSP and ASP |
| 3 | Major class | Minor class | OSP (major class) | OSP (major class) | Closer to- wards ASP |
| 4 | Minor class | Major class | OSP (minor class) | OSP (minor class) | Closer to- wards OSP |

[CAPTION] Fig. 3 gives a diagrammatic illustration of the 4 cases.

[CAPTION] Fig. 3: An illustration of the target class distribution of CL, SCL and ABCL for the 4 cases


<!-- page 11 -->
it does not consider the frequencies of the predicted classes. Case 3 presents an
opportunity to counteract this bias, as the model has already indicated via the
ASP that the minor class is likely correct since it made that prediction despite
its natural tendency to predict the major class. Unfortunately, using CL/SCL
in this situation is likely to encourage the model to mispredict the minor class
sample as the major class. It would be preferable to instead move the desired
target class distribution towards ASP, thus counteracting the dataset bias. In
case 4, OSP is the minor class, so there is no bias towards the major class and
so both CL and SCL do a good job of rewarding the minor class prediction in
this case.
2.3. Adaptive Blended Consistency Loss (ABCL)
To address the drawbacks of CL and SCL, the desired target class distri-
bution for the consistency loss should be adaptively adjusted to be somewhere
between OSP and ASP depending on which of the 4 cases in Tab. 1 and Fig. 3
has occurred. In cases 1 and 2, OSP and ASP both are either the major or
minor class. In this case the desired target class distribution should be in the
middle of OSP and ASP. In case 3 and 4, to discourage the bias towards pre-
dicting the major class, the desired target distribution should be closer to the
minor class depending on whether OSP or ASP is the minor class.
We proposed a new consistency loss function called Adaptive Blended Con-
sistency Loss (ABCL) which captures the desirable properties listed above. Fig. 4
illustrates how it works. ABCL uses the following loss function to generate a
new target class distribution which is a blend of OSP and ASP.
ABCL(z, ˆz) = Lcon(zblended, z) + Lcon(zblended, ˆz)
(3)
z = pθ(y|u); ˆz = pθ(y|ˆu)
(4)
Lcon is the Kullback-Leibler divergence between the blended target probability
distribution zblended and either OSP (z) or ASP (ˆz). Hence ABCL pulls both the
11


<!-- page 12 -->
original and augmented predictions towards the target distribution. For a model
with parameters θ, pθ(y|u) and pθ(y|ˆu) denote the predicted class probability
distribution for an original unlabelled example u and its augmented version ˆu
respectively. The gradient of the loss is not be back-propagated through zblended
during parameter optimisation.
The blended target class distribution zblended is deﬁned as follows:
zblended = (1 −k) ∗z + k ∗ˆz
(5)
Where k is the weighting value [0,1] that determines the proportion to which the
blended target moves towards OSP versus ASP. When k equals 0, the target will
become OSP meaning ABCL will become the same as CL. On the other hand,
as k approaches 1, the target approaches ASP. k is calculated based on predicted
class frequencies with respect to the training dataset using the following formula:
k = max(0, min(γ ∗(Noriginal −Naugmented) + 0.5, 1))
(6)
Noriginal and Naugmented are the class frequencies of the predicted class (class
with the highest predicted probability) for the OSP and ASP respectively. γ ∈
(0, 1] is the class imbalance compensation strength that controls how strong the
new blended target class distribution skews towards either OSP or ASP. The
value of k can be interpreted as follows:
• When the OSP is the minor class and the ASP is the major class (case 4
in Tab. 1 and Fig. 3), the value of Noriginal will be smaller than the value
of Naugmented so the value of k will be smaller than 0.5. This indicates
that the new blended target class distribution will skew towards the minor
class side (OSP).
• When the OSP is the major class and the ASP is the minor class (case 3
in Tab. 1 and Fig. 3), the value of Noriginal will be larger than the value
of Naugmented so the value of k will be larger than 0.5. This indicates that
the new blended target distribution will skew towards the minor class side
(ASP).
12


<!-- page 13 -->
Fig. 4: Diagram showing how ABCL works. The target distribution is blended more towards
the minor class, although it still retains some of OSP’s distribution.
• When OSP and ASP both are the major class or both are the minor class
(cases 1 and 2 in Tab. 1 and Fig. 3), the value of Noriginal and Naugmented
are similar, so the value of k will be around 0.5. In this case OSP and
ASP contribute equally to the blended target distribution.
Class imbalance compensation strength γ. In ABCL ( Eq. 6), the
γ term is used to control how strongly the new blended target distribution is
pushed towards either OSP or ASP. As γ approaches 0, the impact of Noriginal−
Naugmented on the value of k will be smaller, and k will therefore be close to
0.5.
This indicates that the new blended target class distribution will skew
only slightly to either OSP or ASP. On the other hand, as γ approaches 1, the
impact that Noriginal −Naugmented has on k will be bigger. This indicates that
the new blended target class distribution will skew strongly to either the OSP
or ASP. When γ is set to a high value, ABCL approaches the standard CL in
the case that Naugmented >> Noriginal, since zblended ≈z. This corresponds to
CL, where OSP is wholly responsible for deﬁning the target of the unsupervised
loss term.
13

[CAPTION] Fig. 4: Diagram showing how ABCL works. The target distribution is blended more towards


<!-- page 14 -->
Skin cancer dataset
MEL
NV
BCC
AKIEC
BKL
DF
VASC
Total
1113
6705
514
327
1099
115
142
10015
0.11
0.67
0.06
0.03
0.11
0.01
0.01
1
Retinal fundus glaucoma dataset
Glaucoma
Non-Glaucoma)
Total
120
1080
1200
0.1
0.9
1
Tab. 2: The class distribution of 2 experimental classiﬁcation datasets. Both datasets are very
imbalanced. The range of class frequencies frequency is (0.67,0.01) for skin cancer dataset and
(0.9,0.1) for retinal fundus glaucoma dataset.
3. Experiment Setup
3.1. Dataset
We evaluate our proposed methodology on skin cancer and retinal fundus
glaucoma datasets:
• HAM10000 (Tschandl et al., 2018) (main dataset) : contains 10015 RGB
dermatoscopic skin lesions images of size (450, 600) multiclass, divided into
7 classes: Pigmented Bowen’s (AKIEC), Basal Cell Carcinoma (BCC),
Pigmented Benign Keratoses (BKL), Dermatoﬁbroma (DF), Melanoma
(MEL), Nevus (NV), Vascula (VASC).
• REFUGE Challenge (Orlando et al., 2020) (supporting dataset) : contains
1200 RGB retinal fundus glaucoma images of various sizes. Each image is
assigned a binary label of glaucoma or non-glaucoma.
Fig. 5 shows example images for each class. As is shown in Tab. 2, the class
distributions of both datasets are highly imbalanced.
We split the datasets as follows: 70% training images, 20% test images and
10% validation images. We use the validation dataset to decide when to perform
early stopping when training the model. All the results reported in the paper are
from the test dataset. Notably, the unlabelled training dataset is deliberately
14


**[Table p14.1]**
| Skin cancer dataset | MEL | NV | BCC | AKIEC | BKL | DF | VASC | Total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | 1113 | 6705 | 514 | 327 | 1099 | 115 | 142 | 10015 |
|  | 0.11 | 0.67 | 0.06 | 0.03 | 0.11 | 0.01 | 0.01 | 1 |


**[Table p14.2]**
| Retinal fundus glaucoma dataset | Glaucoma | Non-Glaucoma) | Total |
| --- | --- | --- | --- |
|  | 120 | 1080 | 1200 |
|  | 0.1 | 0.9 | 1 |

[CAPTION] Fig. 5 shows example images for each class. As is shown in Tab. 2, the class


<!-- page 15 -->
(a)
(b)
Fig. 5: Example images of the dermatoscopic skin lesions and retinal fundus glaucoma dataset
for each class.
15

[CAPTION] Fig. 5: Example images of the dermatoscopic skin lesions and retinal fundus glaucoma dataset


<!-- page 16 -->
larger than the labelled training dataset in order to simulate the problem we
described earlier. We ensure the class distribution across each of the data splits
are the same.
3.2. Data processing
For the HAM10000 dataset, the model is trained with the original image
size of (450, 600), in contrast, images of the REFUGE challenge dataset are
resized to (512,512). All images of both datasets have their color normalised.
In particular, the red and blue color channel will be normalised based on the
green color channel. The formula is described as follows:
R = R ∗(Gmean
Rmean
); B = B ∗(Gmean
Bmean
)
3.3. Evaluation metrics
Due to the class imbalance in the testing dataset, the model usually per-
forms really well on the major classes and much worse on the minor classes. By
looking at the overall accuracy, we might be tempted to say that the model has
good performance, but actually it only achieves high accuracy on major classes.
However, minor class accuracy is really important, especially for medical image
analysis since often the samples with disease belong to the minor class. Hence,
the Skin Lesion Analysis towards Melanoma Detection (ISIC 2018) (Gutman
et al., 2016) competition used an evaluation metric that gives equal weight to
all classes (called unweighted average recall (UAR)) to rank the performance
of the diﬀerent models. Therefore, we also use UAR to measure our model’s
performance across all classes in a fair way. In addition we report results using
the G-mean and average Area Under the ROC curve (AUC) metric since they
are also commonly used in other studies involving classiﬁcation of imbalanced
classes (Kaur et al., 2019).
3.4. Model training
All models were trained for 200 epochs with the static learning rate of
10−4. To implement early stopping we use the model that gave the best vali-
16


<!-- page 17 -->
dation UAR for measuring results on the test set. Furthermore, the Stochastic
Gradient Descent (SGD) optimizer with momentum 0.9 and weight decay 5 x
10−4 was used. During training we use a batch size of 30 samples consisting of
8 labelled samples and 22 unlabelled samples.
3.5. Data Augmentation
Strong data augmentation via RandAugmentation (Cubuk et al., 2019)
is a key component of the UDA method, and the authors show that it can sig-
niﬁcantly improve the sample eﬃciency by encouraging consistency on a diverse
set of augmented examples. However, experiments presented in Sec. 4.3.2 show
ABCL performs poorly when strong data augmentation is used. We attribute
the poor performance of ABCL in the presence of strong data augmentation
to increased distance between the augmented and unaugmented examples in
embedding space. ABCL assumes in cases 3 and 4 of Tab. 1 (one of the pre-
dictions belongs to a major class and the other belongs to a minor class), it
is more likely the actual class is the minor class. However, when using strong
data augmentation, the augmented major class sample may be moved far away
from the original example in feature space. As a consequence, there is a risk the
augmented sample may be predicted to be a diﬀerent class. That is, samples
belonging to the major classes are more likely to be predicted wrongly since the
strong augmentation can make very signiﬁcant changes to the appearance of
the images. Consequently, the main assumption behind ABCL is violated when
strong data augmentation is used. Hence, we use weak data augmentation for
both labelled and unlabelled samples which include random horizontal ﬂips,
rotations (between 0 and 180 degree), erasing (Zhong et al., 2017) (a random
proportion (0.02-0.33) of input image will be erased) and color augmentation
(jitter brightness, saturation and contrast with a random value in range of 0.9
to 1.1).
17


<!-- page 18 -->
3.6. Algorithms used in experimental study
UDA baseline. The original UDA method is implemented to obtain
the baseline method. This baseline is not designed to handle class imbalance.
All other methods implemented modify this baseline UDA method. As default
we use weak data augmentation for all algorithms implemented in this paper
including UDA baseline. However in Sec. 4.3.2 we show an experiment where
we compare the performance of strong versus weak data augmentation for UDA
and UDA+ABCL. The results show that ABCL with weak data augmentation
is able to outperform UDA using strong data augmentation. This shows that
beneﬁts of using ABCL outweigh the performance loss from using weak data
augmentation.
Sampling based method for the labelled dataset.
We will resam-
ple the skewed labelled dataset by using the intelligent oversampling method
SMOTE (Chawla et al., 2002) and random undersampling.
Weighted Cross Entropy Loss (WeightedCE). This is an extension of
the Cross Entropy loss function which weights the loss of each class based on a
given set of weights. The objective of WeightedCE is to give higher loss for the
minor class and lower loss for the major class. Intuitively, it helps the model
reduce the eﬀect of the major class. We deﬁne the set of class weights as follows:
• Class frequency : [0.11,0.67,0.06,0.03,0.11,0.01,0.01] for the HAM10000
dataset and [0.1,0.9] for the REFUGE dataset
• weight = 1 - Class frequency
Focal Loss (Lin et al., 2017). This loss function is used for supervised
learning to reduce the eﬀect of the “easy samples” (samples belonging to the
major class). The default γ value used was 1 because we found this value for γ
gave the best UAR result when we did a hyperparameter search.
Consistency loss suppression (SCL) (Hyun et al., 2020). This is
the state-of-the-art SSL method used to address class skew that suppresses
the consistency loss when the minor class is predicted. We implemented this
18


<!-- page 19 -->
approach on top of UDA. The default β value used was 0.8 because we found
this value gave the best UAR result when we did a hyperparameter search.
Adaptive Blended Consistency Loss (ABCL). This is our method for
addressing the class imbalance problem for SSL implemented on top of UDA.
Our new method has only one hyperparameter: the class imbalance compensa-
tion factor γ. The default γ value used was 0.4 because we found this value gave
the best overall result when considering UAR and the individual recall results
for each class.
UDA-WeightedCE-SCL and UDA-WeightedCE-ABCL. This method
combines supervised and unsupervised class imbalance methods.
We com-
bine WeightedCE with SCL and ABCL. We chose WeightedCE since we found
WeightedCE performs the best among all the supervised class imbalance meth-
ods.
4. Experiment Results
4.1. Results for the skin cancer (HAM10000) dataset
Tab. 3 shows the test set results (UAR as the main metric and G-mean,
average AUC as supporting metrics) comparing our ABCL with other methods
for the HAM10000 dataset. All semi-supervised methods are based on the UDA
method.
Our ABCL achieved the highest UAR, G-mean and average AUC
values of 0.67, 0.62 and 0.95 respectively among all experimented methods. Ex-
isting class imbalance methods designed for labelled data (Sampling and Focal)
performed worse than the baseline UDA method for the UAR and G-mean met-
ric meaning they were not eﬀective at addressing the class imbalance problem
in SSL. This is because these methods can only make a small contribution to
overall model quality since they can only be applied to the labelled data which
in SSL is just a small portion of the total dataset. On the other hand, class
19


<!-- page 20 -->
imbalance methods designed for unlabelled data (SCL and our ABCL) out-
performed the baseline UDA for the UAR and G-mean metrics meaning these
methods are more eﬀective than methods which are only applicable to labelled
data. This is because these methods modify the consistency loss which is used
for the unlabelled data (occupies a larger proportion of all data for SSL) .
When WeightedCE was used with ABCL and SCL, the UAR performance
was boosted from 0.67 to 0.68 and from 0.61 to 0.65 respectively. This can be
explained by the fact addressing class imbalance for the labelled data helps the
model produce lower bias of the major class for the pseudo target distribution
in the unsupervised consistency loss. Additionally, although the SCL outper-
formed the baseline UDA, its performance was still worse than our ABCL in
all experiments that involved both methods. This is because as discussed in
Sec. 2.2, SCL has the two problems of bias towards major class and ignoring
the augmented sample prediction when determining the target class distribu-
tion.
To further understand the performance of ABCL against the rival meth-
ods, we also reported the test set recall of each class in
Tab. 4. This allows
us to see how the major and minor classes contribute to the UAR. ABCL
performed the best among all methods for almost all of the minor classes
(MEL,BCC,BKL,VASC) and performed worse than most methods for the major
class NV. This result shows ABCL is doing what it was designed to do, namely,
it alleviates the bias towards the major class unlike CL and SCL.
The importance of high recall for minor classes. In the medical
domain it is usually better to mistakenly report a false positive than to miss a
true positive disease diagnosis. This is because manual assessment or further
testing can be used to correct the misdiagnosis. However, missing a positive
disease diagnosis may leave a potentially fatal condition untreated. Additionally,
in medical data, the amount of healthy cases is usually higher than the amount
of diseased cases implying the major class is usually the healthy case. In the skin
cancer dataset, the NV class is the benign class, which is also the only major
class and the rest are minor classes. Based on this principle, ABCL gives better
20


<!-- page 21 -->
Algorithms
UAR
G-mean
Average AUC
Supervised learning
0.75
0.71
0.98
UDA baseline
0.59
0.56
0.91
UDA-Sampling
0.51
0.44
0.89
UDA-WeightedCE
0.59
0.55
0.92
UDA-Focal
0.55
0.50
0.92
UDA-SCL
0.61
0.58
0.92
UDA-WeightedCE-SCL
0.65
0.64
0.94
UDA-ABCL(ours)
0.67
0.62
0.95
UDA-WeightedCE-ABCL(ours)
0.68
0.66
0.96
Tab. 3: Test set results of supervised learning, UDA baseline and various methods designed
for handling class imbalance on top of UDA for the HAM10000 dataset. The results in bold
show the best result for each column among the unsupervised methods.
Algorithms
MEL
(0.11)
NV
(0.67)
BCC
(0.06)
AKIEC
(0.03)
BKL
(0.11)
DF
(0.01)
VASC
(0.01)
UAR
UDA baseline
0.43
0.93
0.81
0.35
0.55
0.50
0.56
0.59
UDA-Sampling
0.44
0.84
0.69
0.52
0.53
0.36
0.12
0.51
UDA-Weighted
Loss
0.39
0.94
0.83
0.40
0.53
0.41
0.59
0.59
UDA-Focal
0.34
0.96
0.81
0.32
0.51
0.32
0.62
0.55
UDA-SCL
0.40
0.94
0.78
0.46
0.57
0.45
0.65
0.61
UDA-
ABCL(ours)
0.73
0.88
0.83
0.46
0.64
0.36
0.76
0.67
UDA-
WeightedCE-
SCL
0.5
0.96
0.73
0.58
0.59
0.55
0.65
0.65
UDA-
WeightedCE-
ABCL(ours)
0.68
0.9
0.79
0.51
0.67
0.45
0.74
0.68
Tab. 4: Test set recall results of the algorithms for each class of the HAM10000 dataset. The
number in brackets under each class name shows the fraction of all samples belonging to that
class. Therefore the major class with most examples is NV.
21


**[Table p21.1]**
| MEL NV BCC AKIEC BKL DF VASC Algorithms (0.11) (0.67) (0.06) (0.03) (0.11) (0.01) (0.01) | UAR |
| --- | --- |


**[Table p21.2]**
| UDA baseline 0.43 0.93 0.81 0.35 0.55 0.50 0.56 UDA-Sampling 0.44 0.84 0.69 0.52 0.53 0.36 0.12 UDA-Weighted 0.39 0.94 0.83 0.40 0.53 0.41 0.59 Loss UDA-Focal 0.34 0.96 0.81 0.32 0.51 0.32 0.62 UDA-SCL 0.40 0.94 0.78 0.46 0.57 0.45 0.65 UDA- 0.73 0.88 0.83 0.46 0.64 0.36 0.76 ABCL(ours) UDA- WeightedCE- 0.5 0.96 0.73 0.58 0.59 0.55 0.65 SCL UDA- WeightedCE- 0.68 0.9 0.79 0.51 0.67 0.45 0.74 ABCL(ours) | 0.59 0.51 0.59 0.55 0.61 0.67 0.65 0.68 |
| --- | --- |


<!-- page 22 -->
Algorithms
MEL
(0.11)
NV
(0.67)
BCC
(0.06)
AKIEC
(0.03)
BKL
(0.11)
DF
(0.01)
VASC
(0.01)
Average
AUC
UDA
base-
line
0.85
0.91
0.96
0.91
0.89
0.89
0.97
0.91
UDA-
ABCL(ours)
0.93
0.95
0.98
0.96
0.95
0.95
0.99
0.95
UDA-SCL
0.86
0.91
0.97
0.92
0.89
0.88
0.98
0.92
Tab. 5: Test set AUC results for each class of the HAM10000 dataset and its average using
UDA baseline, SCL and ABCL methods. The number in the brackets next to the class name
is the fraction of examples that belong to that class.
performance than all its competitors. The test set results in Tab. 4 shows that
ABCL when compared to its nearest non-ABCL rival achieves higher recall for
the minor classes MEL, BCC, BKL and VASC. This means ABCL is less likely
to miss disease diagnosis than alternative losses.
Fig. 6 shows our ABCL method compared against the SCL and UDA base-
line using the ROC and the corresponding AUC. ABCL’s AUC results are better
than the UDA baseline and SCL for all classes according to Tab. 5. This means
ABCL is better at separating between the positive and negative classes than
the alternatives.
4.2. The eﬀects of varying γ value
In this section we analyse the important γ parameter of ABCL according
to reported recall results for each class in
Tab. 6.
Important observations
include the following:
• As γ approaches 1, the recall of the major class NV decreases and the
recall of almost all minor classes increases.
• As γ approaches 0, the recall of the major class NV increases and the
recall of almost all minor classes decreases.
This implies that as γ is increasing, the model is compensating more towards
the minor classes than the major class, leading to the degradation of major
22


**[Table p22.1]**
| MEL NV BCC AKIEC BKL DF VASC Algorithms (0.11) (0.67) (0.06) (0.03) (0.11) (0.01) (0.01) | Average AUC |
| --- | --- |


**[Table p22.2]**
| UDA base- 0.85 0.91 0.96 0.91 0.89 0.89 0.97 line UDA- 0.93 0.95 0.98 0.96 0.95 0.95 0.99 ABCL(ours) UDA-SCL 0.86 0.91 0.97 0.92 0.89 0.88 0.98 | 0.91 0.95 0.92 |
| --- | --- |

[CAPTION] Fig. 6 shows our ABCL method compared against the SCL and UDA base-


<!-- page 23 -->
Fig. 6: Test set ROC and AUC results for each class of the HAM10000 dataset using UDA
baseline, SCL and ABCL methods. The number in the brackets next to the class name is the
fraction of examples that belong to that class.
23

[CAPTION] Fig. 6: Test set ROC and AUC results for each class of the HAM10000 dataset using UDA


<!-- page 24 -->
γ
value
MEL
(0.11)
NV
(0.67)
BCC
(0.06)
AKIEC
(0.03)
BKL
(0.11)
DF
(0.01)
VASC
(0.01)
UAR
0.2
0.66
0.92
0.82
0.47
0.65
0.32
0.68
0.65
0.4
0.73
0.88
0.83
0.46
0.64
0.36
0.76
0.67
0.5
0.74
0.86
0.84
0.51
0.64
0.41
0.71
0.67
0.6
0.73
0.83
0.84
0.51
0.66
0.36
0.76
0.67
0.8
0.75
0.78
0.86
0.49
0.63
0.45
0.76
0.68
1
0.78
0.76
0.83
0.53
0.64
0.41
0.82
0.68
Tab. 6: Test set recall results for each class of the HAM10000 dataset when the γ hyper
parameter value of ABCL is varied.
class performance. This can be explained by the case when the model correctly
predicts the OSP as the major class and mispredicts the ASP as the minor
class, the blended target class distribution is then skewed towards the ASP.
As γ approaches 1, the blended target class distribution moves closer to ASP.
Therefore the γ term in ABCL can be used to tradeoﬀdecreased major class
performance for increased minor class performance.
From Tab. 6 we decided to choose a γ value of 0.4 as our default value since
it provided a good balance of recall for the minor classes while retaining most
of the recall for the major class NV.
4.3. Ablation study
4.3.1. Selective versus always-on target blending
Here we compare ABCL with selective target blending versus ABCL with
always-on target blending using the HAM10000 dataset. In our experiments we
use ABCL with always-on target blending as our default method. This means
even when the original and augmented samples both are predicted as the minor
or major class (cases 1 and 2 of
Tab. 1), we still blend the two samples to
produce the targets. However in selective target blending we do not blend the
original and augmented predictions when they both predict the minor or major
class, instead in this situation we resort to the standard UDA method of just
setting the target as the predicted output from the original sample.
24


**[Table p24.1]**
| 0.2 0.66 0.92 0.82 0.47 0.65 0.32 0.68 0.4 0.73 0.88 0.83 0.46 0.64 0.36 0.76 0.5 0.74 0.86 0.84 0.51 0.64 0.41 0.71 0.6 0.73 0.83 0.84 0.51 0.66 0.36 0.76 0.8 0.75 0.78 0.86 0.49 0.63 0.45 0.76 1 0.78 0.76 0.83 0.53 0.64 0.41 0.82 | 0.65 0.67 0.67 0.67 0.68 0.68 |
| --- | --- |


<!-- page 25 -->
Algorithms
γ
value
MEL
(0.11)
NV
(0.67)
BCC
(0.06)
AKIEC
(0.03)
BKL
(0.11)
DF
(0.01)
VASC
(0.01)
UAR
ABCL
(always-
on
blending)
0.2
0.66
0.92
0.82
0.47
0.65
0.32
0.68
0.65
0.4
0.73
0.88
0.83
0.46
0.64
0.36
0.76
0.67
0.5
0.74
0.86
0.84
0.51
0.64
0.41
0.71
0.67
0.6
0.73
0.83
0.84
0.51
0.66
0.36
0.76
0.67
0.8
0.75
0.78
0.86
0.49
0.63
0.45
0.76
0.68
1
0.78
0.76
0.83
0.53
0.64
0.41
0.82
0.68
ABCL
(se-
lec-
tive
blending)
0.2
0.52
0.91
0.77
0.49
0.60
0.45
0.65
0.63
0.4
0.65
0.83
0.83
0.49
0.62
0.55
0.74
0.67
0.5
0.64
0.80
0.85
0.50
0.61
0.53
0.72
0.66
0.6
0.65
0.69
0.85
0.46
0.55
0.45
0.76
0.63
0.8
0.69
0.53
0.86
0.53
0.58
0.32
0.76
0.61
1
0.59
0.50
0.82
0.56
0.59
0.41
0.71
0.60
Tab. 7: Test set recall results selective target blending versus always-on blending for each
class of the HAM10000 dataset.
The results in Tab. 7 show that the version of ABCL that always blends
targets gives higher UAR (between 0.65 and 0.68) across the entire range of γ
values. In contrast, ABCL with selective target blending performs poorly for
high γ values, especially for the major class NV. ABCL improves the model’s
performance on minor classes by compensating more towards minor classes. As
a consequence, in case 2 of Tab. 1 (the original and augmented samples both
are predicted as the minor class), there is a harmful eﬀect to major classes that
the actual major class might be mispredicted as the minor class. Therefore,
in this case, always selecting the original distribution as the target distribution
might exacerbate this harmful eﬀect. On the other hand, ABCL with always-
on blending can mitigate this harmful eﬀect, leading to less degradation in the
recall for the major class.
4.3.2. Weak versus strong data augmentation
In this section we compare the eﬀects of diﬀerent data augmentation
strategies for the baseline UDA method and our ABCL method on the HAM10000
dataset. Tab. 8 shows the results of this experiment. The results show that for
25


**[Table p25.1]**
| γ MEL NV BCC AKIEC BKL DF VASC Algorithms value (0.11) (0.67) (0.06) (0.03) (0.11) (0.01) (0.01) | UAR |
| --- | --- |


**[Table p25.2]**
| 0.2 0.66 0.92 0.82 0.47 0.65 0.32 0.68 0.4 0.73 0.88 0.83 0.46 0.64 0.36 0.76 ABCL 0.5 0.74 0.86 0.84 0.51 0.64 0.41 0.71 (always- 0.6 0.73 0.83 0.84 0.51 0.66 0.36 0.76 on 0.8 0.75 0.78 0.86 0.49 0.63 0.45 0.76 blending) 1 0.78 0.76 0.83 0.53 0.64 0.41 0.82 | 0.65 0.67 0.67 0.67 0.68 0.68 |
| --- | --- |


**[Table p25.3]**
| 0.2 0.52 0.91 0.77 0.49 0.60 0.45 0.65 ABCL 0.4 0.65 0.83 0.83 0.49 0.62 0.55 0.74 (se- 0.5 0.64 0.80 0.85 0.50 0.61 0.53 0.72 lec- 0.6 0.65 0.69 0.85 0.46 0.55 0.45 0.76 tive 0.8 0.69 0.53 0.86 0.53 0.58 0.32 0.76 blending) 1 0.59 0.50 0.82 0.56 0.59 0.41 0.71 | 0.63 0.67 0.66 0.63 0.61 0.60 |
| --- | --- |


<!-- page 26 -->
Algorithms
UAR
G-mean
Average AUC
UDA baseline + WeakAug
0.59
0.56
0.91
UDA baseline + StrongAug
0.61
0.56
0.92
ABCL + WeakAug
0.67
0.62
0.95
ABCL + StrongAug
0.50
0.42
0.91
Tab. 8: Test set UAR, G-mean and average AUC results of the HAM10000 dataset comparing
UDA baseline and ABCL with strong data augmentation and weak data augmentation.
the UAR, G-mean and average AUC metrics, strong data augmentation works
better than weak data augmentation for the UDA baseline whereas the oppo-
site is true for ABCL. As discussed in Sec. 3.5, we believe the reason for this is
when using strong data augmentation, even samples belonging to major classes
may be predicted wrongly since the augmentation can make very signiﬁcant
changes to the appearance of the images. Consequently, the main assumption
behind ABCL is violated when strong data augmentation is used. The results
also show that ABCL with weak data augmentation is able to outperform the
baseline UDA using the strong data augmentation. This is an important result
since it shows ABCL is able to overcome any negative consequence of not being
able to use strong data augmentation.
4.4. Results for the retinal fundus glaucoma (REFUGE)
dataset
Tab. 9 shows the test set results of comparing ABCL with the other com-
peting methods for the glaucoma (REFUGE) dataset. ABCL does not improve
the UDA-baseline model’s performance when the standard cross entropy loss is
used for the supervised loss. However, using a combination of WeightedCE for
the supervised loss and ABCL for unsupervised loss, UDA-WeightedCE-ABCL
is able to signiﬁcantly outperform UDA-WeightedCE and UDA-WeightedCE-
SCL. As explained in Sec. 4.1, WeightedCE can help the model produce lower
bias of towards the major class for the pseudo target distribution in the un-
26


<!-- page 27 -->
Algorithms
UAR
G-mean
Average AUC
UDA baseline
0.55
0.37
0.82
UDA-WeightedCE
0.57
0.43
0.83
UDA-Focal
0.57
0.43
0.82
UDA-SCL
0.55
0.37
0.81
UDA-WeightedCE-SCL
0.55
0.37
0.82
UDA-ABCL(ours)
0.55
0.37
0.82
UDA-WeightedCE-ABCL(ours)
0.67
0.61
0.83
Tab. 9: Test set UAR, G-mean and AUC results for the REFUGE dataset.
supervised consistency loss which is of beneﬁt to both ABCL and SCL meth-
ods. However, UDA-WeightedCE-ABCL outperforms UDA-WeightedCE-SCL
because ABCL is able to adaptively blend the target distribution between the
original and augmented samples’s predicted distributions depending on which
predicted the minor class. This helps ABCL better address the class imbalance
problem in the unlabelled data.
5. Conclusion
In this study, we identiﬁed an important gap in existing literature: class
imbalance in the context of semi-supervised learning for medical images. This is
an important problem to study since medical image datasets often have skewed
distributions and missing positive disease diagnosis can have fatal consequences.
We address this gap by proposing a new consistency loss function called Adap-
tive Blended Consistency Loss (ABCL). To demonstrate the eﬀectiveness of
ABCL, we applied it to the perturbation based semi-supervised learning al-
gorithm UDA. ABCL extends standard consistency loss by generating a new
blended target class distribution from the mix of original and augmented sam-
ple’s class distribution. Extensive experiments showed ABCL consistently out-
performs alternative approaches designed to address the class imbalance prob-
27


<!-- page 28 -->
lem.
Our proposed ABCL method was able to improve the performance of
UDA baseline from 0.59 to 0.67 UAR, outperform methods that address the
class imbalance problem for labelled data (between 0.51 and 0.59 UAR) and
for unlabelled data (0.61 UAR) on the imbalanced skin cancer dataset. On the
imbalanced retinal fundus glaucoma dataset, combining with Weighted Cross
Entropy loss, ABCL achieved 0.67 UAR as compared to 0.57 to its nearest
rival. The results on both datasets show that ABCL can be combined with
methods designed to address class imbalance for supervised learning to further
boost performance on semi-supervised learning. As future work we plan to eval-
uate ABCL on other medical imaging tasks, such as semi-supervised semantic
segmentation.
Declaration of Competing Interest
The authors declare that they have no known competing ﬁnancial inter-
ests or personal relationships that could have appeared to inﬂuence the work
reported in this paper.
Acknowledgements
This research did not receive any speciﬁc grant from funding agencies in
the public, commercial, or not-for-proﬁt sectors.
References
Ausawalaithong, W., Thirach, A., Marukatat, S., Wilaiprasitporn, T., 2018.
Automatic lung cancer prediction from chest x-ray images using the deep
learning approach, in: 2018 11th Biomedical Engineering International Con-
ference (BMEiCON), IEEE. pp. 1–5.
28


<!-- page 29 -->
Barandela, R., Valdovinos, R.M., S´anchez, J.S., Ferri, F.J., 2004.
The im-
balanced training sample problem: Under or over sampling?, in: Joint IAPR
international workshops on statistical techniques in pattern recognition (SPR)
and structural and syntactic pattern recognition (SSPR), Springer. pp. 806–
814.
Berthelot, D., Carlini, N., Goodfellow, I., Papernot, N., Oliver, A., Raﬀel, C.A.,
2019. Mixmatch: A holistic approach to semi-supervised learning, in: Ad-
vances in Neural Information Processing Systems, pp. 5050–5060.
Bi, L., Kim, J., Ahn, E., Feng, D., 2017. Automatic skin lesion analysis using
large-scale dermoscopy images and deep residual networks. arXiv preprint
arXiv:1703.04197 .
Bunkhumpornpat, C., Sinapiromsaran, K., Lursinsap, C., 2009.
Safe-level-
smote: Safe-level-synthetic minority over-sampling technique for handling the
class imbalanced problem, in: Paciﬁc-Asia conference on knowledge discovery
and data mining, Springer. pp. 475–482.
Chapelle, O., Scholkopf, B., Zien, A., 2009. Semi-supervised learning (chapelle,
o. et al., eds.; 2006)[book reviews]. IEEE Transactions on Neural Networks
20, 542–542.
Chawla, N.V., Bowyer, K.W., Hall, L.O., Kegelmeyer, W.P., 2002.
Smote:
synthetic minority over-sampling technique. Journal of artiﬁcial intelligence
research 16, 321–357.
Chen, J., Tsai, C.A., Moon, H., Ahn, H., Young, J., Chen, C.H., 2006. Decision
threshold adjustment in class prediction. SAR and QSAR in Environmental
Research 17, 337–352.
Creswell, A., Pouplin, A., Bharath, A.A., 2018. Denoising adversarial autoen-
coders: classifying skin lesions using limited labelled training data. IET Com-
puter Vision 12, 1105–1111.
29


<!-- page 30 -->
Cubuk, E.D., Zoph, B., Shlens, J., Le, Q.V., 2019. Randaugment: Practical
automated data augmentation with a reduced search space. arXiv preprint
arXiv:1909.13719 .
Gutman, D., Codella, N.C., Celebi, E., Helba, B., Marchetti, M., Mishra,
N., Halpern, A., 2016. Skin lesion analysis toward melanoma detection: A
challenge at the international symposium on biomedical imaging (isbi) 2016,
hosted by the international skin imaging collaboration (isic). arXiv preprint
arXiv:1605.01397 .
Han, H., Wang, W.Y., Mao, B.H., 2005. Borderline-smote: a new over-sampling
method in imbalanced data sets learning, in: International conference on
intelligent computing, Springer. pp. 878–887.
Han, S.S., Kim, M.S., Lim, W., Park, G.H., Park, I., Chang, S.E., 2018. Clas-
siﬁcation of the clinical images for benign and malignant cutaneous tumors
using a deep learning algorithm. Journal of Investigative Dermatology 138,
1529–1538.
Hekler, A., Utikal, J.S., Enk, A.H., Hauschild, A., Weichenthal, M., Maron,
R.C., Berking, C., Haferkamp, S., Klode, J., Schadendorf, D., et al., 2019.
Superior skin cancer classiﬁcation by the combination of human and artiﬁcial
intelligence. European Journal of Cancer 120, 114–121.
Hyun, M., Jeong, J., Kwak, N., 2020. Class-imbalanced semi-supervised learn-
ing. arXiv preprint arXiv:2002.06815 .
Iizuka, O., Kanavati, F., Kato, K., Rambeau, M., Arihiro, K., Tsuneki, M.,
2020. Deep learning models for histopathological classiﬁcation of gastric and
colonic epithelial tumours. Scientiﬁc Reports 10, 1–11.
Kaur, H., Pannu, H.S., Malhi, A.K., 2019. A systematic review on imbalanced
data challenges in machine learning: Applications and solutions. ACM Com-
puting Surveys (CSUR) 52, 1–36.
30


<!-- page 31 -->
Khan, S.H., Hayat, M., Bennamoun, M., Sohel, F.A., Togneri, R., 2017. Cost-
sensitive learning of deep feature representations from imbalanced data. IEEE
transactions on neural networks and learning systems 29, 3573–3587.
Krawczyk, B., Wo´zniak, M., 2015. Cost-sensitive neural network with roc-based
moving threshold for imbalanced classiﬁcation, in: International Conference
on Intelligent Data Engineering and Automated Learning, Springer. pp. 45–
52.
Krizhevsky, A., Sutskever, I., Hinton, G.E., 2012. Imagenet classiﬁcation with
deep convolutional neural networks, in: Advances in neural information pro-
cessing systems, pp. 1097–1105.
Kubat, M., Matwin, S., et al., 1997. Addressing the curse of imbalanced training
sets: one-sided selection, in: Icml, Citeseer. pp. 179–186.
Laine, S., Aila, T., 2016. Temporal ensembling for semi-supervised learning.
arXiv preprint arXiv:1610.02242 .
Lin, T.Y., Goyal, P., Girshick, R., He, K., Doll´ar, P., 2017. Focal loss for dense
object detection, in: Proceedings of the IEEE international conference on
computer vision, pp. 2980–2988.
Mani, I., Zhang, I., 2003. knn approach to unbalanced data distributions: a
case study involving information extraction, in: Proceedings of workshop on
learning from imbalanced datasets.
Miyato, T., Maeda, S.i., Koyama, M., Ishii, S., 2018. Virtual adversarial train-
ing: a regularization method for supervised and semi-supervised learning.
IEEE transactions on pattern analysis and machine intelligence 41, 1979–
1993.
Orlando, J.I., Fu, H., Breda, J.B., van Keer, K., Bathula, D.R., Diaz-Pinto,
A., Fang, R., Heng, P.A., Kim, J., Lee, J., et al., 2020. Refuge challenge: A
uniﬁed framework for evaluating automated methods for glaucoma assessment
from fundus photographs. Medical image analysis 59, 101570.
31


<!-- page 32 -->
Phan, H., Krawczyk-Becker, M., Gerkmann, T., Mertins, A., 2017. Dnn and
cnn with weighted and multi-task loss functions for audio event detection.
arXiv preprint arXiv:1708.03211 .
Rajpurkar, P., Irvin, J., Zhu, K., Yang, B., Mehta, H., Duan, T., Ding, D.,
Bagul, A., Langlotz, C., Shpanskaya, K., et al., 2017. Chexnet: Radiologist-
level pneumonia detection on chest x-rays with deep learning. arXiv preprint
arXiv:1711.05225 .
Rezvantalab, A., Saﬁgholi, H., Karimijeshni, S., 2018. Dermatologist level der-
moscopy skin cancer classiﬁcation using diﬀerent deep learning convolutional
neural networks algorithms. arXiv preprint arXiv:1810.10348 .
Springenberg, J.T., 2015. Unsupervised and semi-supervised learning with cat-
egorical generative adversarial networks. arXiv preprint arXiv:1511.06390 .
Su, H., Shi, X., Cai, J., Yang, L., 2019. Local and global consistency regularized
mean teacher for semi-supervised nuclei classiﬁcation, in: International Con-
ference on Medical Image Computing and Computer-Assisted Intervention,
Springer. pp. 559–567.
Tarvainen, A., Valpola, H., 2017. Mean teachers are better role models: Weight-
averaged consistency targets improve semi-supervised deep learning results,
in: Advances in neural information processing systems, pp. 1195–1204.
Tschandl, P., Rosendahl, C., Kittler, H., 2018. The ham10000 dataset, a large
collection of multi-source dermatoscopic images of common pigmented skin
lesions. Scientiﬁc data 5, 180161.
Wang, H., Cui, Z., Chen, Y., Avidan, M., Abdallah, A.B., Kronzer, A., 2018.
Predicting hospital readmission via cost-sensitive deep learning. IEEE/ACM
transactions on computational biology and bioinformatics 15, 1968–1978.
Wang, S., Liu, W., Wu, J., Cao, L., Meng, Q., Kennedy, P.J., 2016. Training
deep neural networks on imbalanced data sets, in: 2016 international joint
conference on neural networks (IJCNN), IEEE. pp. 4368–4374.
32


<!-- page 33 -->
Xie, Q., Dai, Z., Hovy, E., Luong, M.T., Le, Q.V., 2019. Unsupervised data
augmentation. arXiv preprint arXiv:1904.12848 .
Yang, W., Zhao, J., Qiang, Y., Yang, X., Dong, Y., Du, Q., Shi, G., Zia,
M.B., 2019.
Dscgans: Integrate domain knowledge in training dual-path
semi-supervised conditional generative adversarial networks and s3vm for ul-
trasonography thyroid nodules classiﬁcation, in: International Conference on
Medical Image Computing and Computer-Assisted Intervention, Springer. pp.
558–566.
Yu, H., Sun, C., Yang, X., Yang, W., Shen, J., Qi, Y., 2016. Odoc-elm: Optimal
decision outputs compensation-based extreme learning machine for classifying
imbalanced data. Knowledge-Based Systems 92, 55–70.
Zhang, C., Tan, K.C., Ren, R., 2016. Training cost-sensitive deep belief networks
on imbalance data problems, in: 2016 international joint conference on neural
networks (IJCNN), IEEE. pp. 4362–4367.
Zhong, Z., Zheng, L., Kang, G., Li, S., Yang, Y., 2017. Random erasing data
augmentation. arXiv preprint arXiv:1708.04896 .
33