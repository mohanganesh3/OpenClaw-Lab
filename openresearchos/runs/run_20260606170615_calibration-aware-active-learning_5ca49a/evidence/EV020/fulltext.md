<!-- page 1 -->
arXiv:2505.05163v2  [cs.CV]  4 Jul 2025
Probabilistic Embeddings for Frozen Vision-Language Models: Uncertainty
Quantification with Gaussian Process Latent Variable Models
Aishwarya Venkataramanan1
Paul Bodesheim1
Joachim Denzler1
1Computer Vision Group, Friedrich Schiller University Jena, Germany
Abstract
Vision-Language Models (VLMs) learn joint rep-
resentations by mapping images and text into a
shared latent space. However, recent research high-
lights that deterministic embeddings from stan-
dard VLMs often struggle to capture the uncer-
tainties arising from the ambiguities in visual and
textual descriptions and the multiple possible cor-
respondences between images and texts. Exist-
ing approaches tackle this by learning probabilis-
tic embeddings during VLM training, which de-
mands large datasets and does not leverage the
powerful representations already learned by large-
scale VLMs like CLIP. In this paper, we propose
GroVE, a post-hoc approach to obtaining proba-
bilistic embeddings from frozen VLMs. GroVE
builds on Gaussian Process Latent Variable Model
(GPLVM) to learn a shared low-dimensional latent
space where image and text inputs are mapped to
a unified representation, optimized through single-
modal embedding reconstruction and cross-modal
alignment objectives. Once trained, the Gaussian
Process model generates uncertainty-aware proba-
bilistic embeddings. Evaluation shows that GroVE
achieves state-of-the-art uncertainty calibration
across multiple downstream tasks, including cross-
modal retrieval, visual question answering, and
active learning.
1
INTRODUCTION
Deep learning has seen remarkable success over the last
decade, yet its practical applicability, especially in safety-
critical areas is limited by unreliable, overconfident predic-
tions [Abdar et al., 2021]. This has motivated the devel-
opment of methods to quantify uncertainty in model pre-
dictions, including stochastic [Blundell et al., 2015, Gal
and Ghahramani, 2016], deterministic [Van Amersfoort
et al., 2020, Mukhoti et al., 2023, Venkataramanan et al.,
2023a], evidential [Sensoy et al., 2018], and post-hoc ap-
proaches [Corbiere et al., 2021], with the aim to produce
calibrated confidence values that better reflect the model’s
actual performance [Guo et al., 2017]. While these meth-
ods have shown strong performance in tasks involving data
from a single modality, they often struggle in multi-modal
settings, such as vision language models (VLMs), where
inputs come from different domains, such as images and
text [Jung et al., 2022]. The challenge arises because these
single-modal approaches fail to capture the uncertainties
that emerge from interactions between the different modali-
ties.
VLMs typically encode images and their corresponding text
descriptions into vector representations within a joint em-
bedding space. While combining modalities enriches seman-
tics and boosts performance on various tasks [Zhang et al.,
2024], it also introduces additional uncertainties. Beyond
the inherent uncertainty of each modality, there is an uncer-
tainty due to the ambiguous relationships between images
and text. This is illustrated in Figure 1, where each image
can correspond to multiple text descriptions, and each text
description can be associated with multiple images. Deter-
ministic embeddings from VLMs often fail to capture these
uncertainties, motivating the development of probabilistic
embeddings [Ji et al., 2023, Chun et al., 2021, Chun, 2023].
Probabilistic embeddings represent a distribution, thereby
capturing a range of possible representations for ambiguous
or uncertain data. Typically, the embeddings are modeled
as Gaussian distributions, and deep neural networks are
trained to maximize their likelihood, learning the distribu-
tion parameters. However, these methods require training
the VLMs from scratch, which requires large-scale datasets,
and does not effectively leverage the strong multi-modal
representations already provided by the pre-trained large-
scale VLMs [Radford et al., 2021, Li et al., 2022b, Singh
et al., 2022].
In this work, we introduce GroVE, a method to generate


<!-- page 2 -->
Figure 1: Illustration of uncertainty arising from multiple
correspondences between image and text descriptions. De-
terministic embeddings represent the instances as fixed
points. In contrast, probabilistic embeddings capture un-
certainty by modeling text and images as distributions, al-
lowing for multiple reasonable matches.
probabilistic embeddings for VLMs in a post-hoc man-
ner that builds on Gaussian Process Latent Variable Model
(GPLVM) [Lawrence, 2003]. GroVE stands for Gaussian
Process for Probabilistic VLM Embeddings. A GPLVM
models the relationship between a low-dimensional latent
space and a high-dimensional observational space using
Gaussian Processes (GPs). Traditionally, the latent space is
used for dimensionality reduction [Lawrence, 2003, Lalc-
hand et al., 2022], and less commonly for more task-specific
applications, such as classification [Eleftheriadis et al.,
2014] and cross-modal retrieval [Song et al., 2017]. In our
approach, we adopt an extension of the GPLVM frame-
work to a multi-modal context, and show that it provides a
principled approach for obtaining probabilistic image and
text embeddings from the deterministic embeddings of the
large-scale frozen VLMs. To achieve this, we learn a joint
low-dimensional latent space, where each pair of image
and text embeddings derived from a VLM is represented
as a single unified point. The mapping between the latent
space and the observed VLM embeddings is established
through two GPs: one for image embeddings and one for
text embeddings. Our training objective consists of an em-
bedding reconstruction loss to learn this mapping, and a
cross-modal alignment that regularizes the latent space to
preserve the semantic structure of the data. Once the latent
space is learned, the trained GP models are used to obtain
probabilistic embeddings for images and texts.
We evaluate GroVE for uncertainty calibration in cross-
modal retrieval using CLIP [Radford et al., 2021] and
BLIP [Li et al., 2022b] on the following standard bench-
marks: common objects datasets MS-COCO [Chen et al.,
2015] and Flickr30k[Young et al., 2014], as well as fine-
grained datasets CUB-200-2011 [Wah et al., 2011] and
Oxford Flowers 102 [Nilsback and Zisserman, 2008]. We
further demonstrate the applicability of our approach in an
active learning setting. We also evaluate its ability to provide
calibrated uncertainty estimates in visual question answer-
ing (VQA) using the VQA 2.0 dataset [Goyal et al., 2017].
Our results show that GroVE effectively learns probabilistic
embeddings that provide calibrated uncertainty estimates.
The contributions are summarized as follows: i) We
propose GroVE, which extends GPLVM and provides
a principled approach to obtain probabilistic VLM em-
beddings for both images and text. ii) We show that
GroVE produces calibrated uncertainty estimates for cross-
modal retrieval and VQA, and demonstrate its practi-
cal utility in active learning. iii) We design GroVE to
work in a post-hoc manner on frozen VLMs, avoiding
the need for retraining large-scale models from scratch.
Code is available: https://github.com/cvjena/
GroVE-Probabilistic_VLM_embeddings.git
2
RELATED WORK
Vision Language Models. Early vision-language ap-
proaches used textual data to embed images in a seman-
tic space, capturing semantic relationships and improv-
ing zero-shot capabilities Frome et al. [2013], Barz and
Denzler [2019], Venkataramanan et al. [2023b], Li et al.
[2017], Zhang and Saligrama [2015]. The advent of trans-
formers Vaswani et al. [2017] revolutionized the landscape
of vision-language modeling. Models like VisualBERT Li
et al. [2019], ViLBERT Lu et al. [2019] and LXMERT Tan
and Bansal [2019] extended the BERT architecture Kenton
and Toutanova [2019] to model complex relationships be-
tween image regions and text tokens. CLIP Radford et al.
[2021] is a prominent VLM trained on 400 million web-
sourced image-caption pairs using a contrastive learning ob-
jective Gutmann and Hyvärinen [2010], Oord et al. [2018]
to align images with their textual descriptions in a shared
embedding space while separating unrelated pairs. CLIP
demonstrates strong zero-shot performance across diverse
tasks, including image classification Li et al. [2023], Qian
and Hu [2024], object detection Lin and Gong [2023], Wu
et al. [2023], cross-modal retrieval Xia et al. [2023], Li et al.
[2024], and visual question answering Xing et al. [2024],
Parelli et al. [2023]. BLIP Li et al. [2022b] leverages noisy
data through bootstrapping, combined with contrastive learn-
ing to achieve state-of-the-art performance. However, these
methods rely on deterministic embeddings that do not cap-
ture modality uncertainty. In contrast, our approach converts
deterministic embeddings into probabilistic representations,
with proper uncertainty estimates from GP models.
Uncertainty Quantification in VLMs. Input data ambi-
guities in VLMs are often addressed by replacing tradi-
tional deterministic embeddings with probabilistic embed-

[CAPTION] Figure 1: Illustration of uncertainty arising from multiple


<!-- page 3 -->
dings Li et al. [2022a]. PCME [Chun et al., 2021] mod-
els image and text embeddings as Gaussians with learned
means and variances, optimizing the joint embedding space
with a soft cross-modal contrastive loss. PCME++ [Chun,
2023] introduces Closed-Form Sampled Distance (CSD)
to compute Gaussian embeddings of images and text for
faster uncertainty estimation compared to PCME. MAP [Ji
et al., 2023] introduces a Probability Distribution Encoder
to model multi-modal representations as probabilistic distri-
butions. However, all these methods require training from
scratch, and do not effectively leverage the strong multi-
modal representations already learned by the pre-trained
large-scale VLMs. ProbVLM [Upadhyay et al., 2023] is a
post-hoc approach that trains neural networks to estimate the
parameters of Generalized Gaussian distribution for image
and text embeddings. Although being straightforward, the
prediction of distribution parameters lacks proper probabilis-
tic modeling of statistical processes underlying the sampling
of data. Furthermore, neural networks are prone to uncali-
brated predictions when presented with out-of-distribution
(OOD) data or limited training samples [Guo et al., 2017].
In contrast, our approach leverages GPs, a Bayesian method
that inherently incorporates probabilistic reasoning with
reliable and theoretically sound uncertainty quantification
as well as distance-awareness through the covariance func-
tion, which has proven effective in calibrated uncertainty
estimation [Liu et al., 2020, Jung et al., 2022].
Post-hoc approaches for uncertainty quantification.
Some of the widely used post-hoc calibration techniques for
data from a single modality are temperature scaling [Guo
et al., 2017] and Platt scaling [Platt et al., 1999], which
adjust the model’s predicted probabilities after training to
better align predicted confidence scores with actual perfor-
mance. Test-Time Data Augmentation (TTDA) [Ayhan and
Berens, 2018, Wang et al., 2019] quantifies uncertainty by
applying various transformations to input data during infer-
ence, generating multiple predictions, and measuring the
variability among them to assess the uncertainty. A line of
work [Corbiere et al., 2021, Yu et al., 2021, Hornauer et al.,
2023, Shi and Jain, 2019] focuses on training auxiliary mod-
els to quantify uncertainty in the primary model, allowing
for uncertainty estimation without impacting the perfor-
mance of the primary model. Unlike these single-modal
approaches, our method captures uncertainty from the re-
lationship between visual and textual modalities, which
is crucial for obtaining accurate uncertainty estimates in
VLMs [Jung et al., 2022].
3
METHOD
GroVE builds on the GPLVM framework to learn a shared
latent space for image and text inputs using GPs. It optimizes
this space through single-modal reconstruction and cross-
modal alignment loss, generating probabilistic embeddings
Image Encoder
Text Encoder
Gaussian 
Process
Gaussian 
Process
Figure 2: Method overview of GroVE. Given deterministic
image and text embeddings from a frozen VLM, GroVE
learns a joint low dimensional latent space, where each
image-text pair is represented by a single point. Two GP
models learn to reconstruct the image and text embeddings
from the latent space points through single-modal recon-
struction and cross-modal alignment objectives. The GP
models act as probabilistic mappings that model the uncer-
tainty in both the image and text modalities.
from deterministic VLM embeddings to capture uncertainty.
Figure 2 illustrates the overall pipeline of GroVE.
3.1
PROBLEM DESCRIPTION
Let D = {(In, Tn)}N
n=1 ⊂I × T represent a dataset of
N paired samples, where In ∈I is an image sampled
from the image space I, and Tn ∈T is the corresponding
text description sampled from the text space T . The VLM
maps an image I and a text T into a shared embedding
space Z ⊆RD. To achieve this, the VLM consists of an
image encoder f θI
I
: I →Z with parameters θI, and a text
encoder f θT
T
: T →Z with parameters θT . We assume that
the VLM has already been trained on a large-scale dataset,
and the parameters of f θI
I
and f θT
T
are fixed as θ∗
I and θ∗
T ,
respectively. The encoders have been trained such that, for
a given image-text pair (I, T), the resulting embeddings
zI = f θ∗
I
I (I) and zT = f θ∗
T
T (T) are positioned close to one
another in Z, so that semantically related visual and textual
information is aligned.
While deterministic VLMs provide fixed embeddings, they
lack the ability to represent the uncertainty associated with
these embeddings. To address this, we propose GroVE, a
method that leverages GPLVM to obtain probabilistic em-
beddings in a post-hoc manner to model the uncertainties.

[CAPTION] Figure 2: Method overview of GroVE. Given deterministic

[CAPTION] Figure 2 illustrates the overall pipeline of GroVE.


<!-- page 4 -->
3.2
GROVE MODEL
We obtain the image and text embeddings from the frozen
VLM on D:
n
zIn, zTn

=

f θ∗
I
I (In), f θ∗
T
T (Tn)
oN
n=1 ,
(1)
where zIn, zTn ∈RD are D-dimensional image and text
embeddings, respectively.
To derive probabilistic embeddings using GPLVM, we as-
sume that zIn and zTn are generated from a shared low-
dimensional latent space X ⊆RQ with Q ≪D, where
each image-text pair (zIn, zTn) is associated with a com-
mon latent point xn ∈RQ. We define two GPLVM models
GPI and GPT , one for each modality (images from I and
text from T ), to learn the mappings GI : X →ZI and
GT : X →ZT from xn to the high-dimensional embed-
dings zIn and zTn, respectively. During GP model training,
the latent points xn are optimized to maximize the likeli-
hood of the observed embeddings zIn and zTn
GP model definitions. For describing the GPLVM mod-
els, we define the matrix X ∈RN×Q as the collection of
the N latent inputs xn. Image embeddings zIn and text
embeddings zTn are supposed to be computed from latent
functions GI and GT :
zIn = GI(xn) + ϵI;
zTn = GT (xn) + ϵT ,
(2)
with noise terms ϵI and ϵT and a GP prior such that for
each dimension d of the embeddings zIn and zTn, the latent
function values gd
I, gd
T ∈RN of the N samples follow a
multivariate Gaussian distribution:
gd
I ∼N(mI(X), kI(X, X)),
gd
T ∼N(mT (X), kT (X, X)).
(3)
These distributions are parameterized by a mean func-
tion m(·) and a covariance function k(·, ·), which defines
the covariance matrix between pairs of points in X. For
both GPLVM models, we use a constant mean function
m(X) = m and a radial basis function (RBF) kernel
k(xi, xj) = exp

−∥xi−xj∥2
2ℓ2

with length-scale hyperpa-
rameter ℓ. However, optimal values for m and ℓare learnt
separately for each modality I and T . The likelihood func-
tions are defined as:
p
zd
I|gd
I

=
N
Y
n=1
p(zd
In|gd
In) = N
gd
I, σ2
II

,
p
zd
T |gd
T

=
N
Y
n=1
p(zd
Tn|gd
Tn) = N
gd
T , σ2
T I

,
(4)
where σ2
I, σ2
T are the parameters of the Gaussian noise
model, which are learned along with the model parameters
during the training.
Embedding Reconstruction Objective. Given the prior
and the likelihood, our goal is to estimate the posterior
distribution. While the exact inference is possible, it is com-
putationally expensive, with cost O(N 3). In this work, we
adopt a sparse GP with inducing points and variational infer-
ence Titsias [2009]. We introduce M inducing points in X
for each modality I and T , where M ≪N. Each inducing
point corresponds to an inducing variable, represented as
the latent function values ud
I ∈RM and ud
T ∈RM, which
capture the latent function values at these locations. The
key idea is to approximate the true posterior distribution
over the latent function values at the observed data points
by conditioning on the inducing variables. This reduces the
computational complexity of the model to O(NM 2).
To achieve this, we introduce a variational distribution over
the inducing variables as:
q(ud
I) = N(ud
I|µd
I, Sd
I); q(ud
T ) = N(ud
T |µd
T , Sd
T ), (5)
where µd
I and µd
T , Sd
I and Sd
T are variational parameters
that are optimized during training. These variational param-
eters, the inducing points, along with the model parameters
mI, mT , lI, lT , σ2
I and σ2
T are learned by maximizing the
lower bound on the marginal likelihood of the data i.e. the
evidence lower bound (ELBO), given by
Ld
ELBO = Eq(gd
I)[log p(zd
I|gd
I)] −DKL(q(ud
I)||p(ud
I))
+Eq(gd
T )[log p(zd
T |gd
T )] −DKL(q(ud
T )||p(ud
T )),
(6)
where DKL is the Kullback-Leibler (KL) divergence, and
is measured between the variational distributions and their
corresponding priors obtained by the GP prior evaluated at
the inducing points. The embedding reconstruction objective
is given by:
Lemb = −
D
X
d=1
Ld
ELBO
(7)
Cross-modal Alignment Objective. In addition to this re-
construction objective, we introduce a regularization term,
so that the predicted distributions of the corresponding im-
age and text embeddings from the GPs match. Aligning
these distributions encourages the latent space to learn a
shared underlying structure between the modalities, so that
semantically related data points are represented by similar
latent variables. To enforce this, we define a KL divergence
loss function between the distributions of the image and
text embeddings from the GP models, which take the forms
N(ˆµI, ˆΣI) and N(ˆµT , ˆΣT ) respectively (refer Sec. 3.3
for inference using GP). The resulting objective LKL is the
mean of the KL divergence in both directions (image-to-text
and text-to-image):
LKL = 1
2[DKL(N(ˆµI, ˆΣI)∥N(ˆµT , ˆΣT ))+
DKL(N(ˆµT , ˆΣT )∥N(ˆµI, ˆΣI))].
(8)


<!-- page 5 -->
Final Objective. The overall objective function is the
weighted sum of the embedding reconstruction loss and
the cross-modal alignment loss:
Ltotal = λ1Lemb + λ2LKL
(9)
where λ1 and λ2 are trade-off parameters.
3.3
PROBABILISTIC EMBEDDINGS
Once the latent space representation X is learned, we use
GPI and GPT to predict the probabilistic image and text
embeddings. Given a new embedding z∗(image or text) ob-
tained from the VLM, we first infer its latent representation
x∗by randomly initializing x∗and iteratively optimizing it
with the ELBO. This approximates the posterior distribution
p(x∗|z∗, zM), where M denotes the modality (either I or
T ). From x∗, the probabilistic embedding can be inferred
using the respective GP.
Inference using GP. The predictive distribution, which de-
fines the predicted probabilistic embedding is given by:
p(gd
∗) =
Z
p(gd
∗|ud
M)q(ud
M)dud
M
(10)
Evaluating the integral results in a Gaussian distribu-
tion Hensman et al. [2015]:
p(gd
∗) = N(gd
∗|ˆµd
∗, ˜Σ
d
∗)
(11)
where the mean ˆµd
∗and covariance ˆΣ
d
∗of the embedding is:
ˆµd
∗= mM + A(µd
M −mvM)
(12)
ˆΣ
d
∗= k(x∗, x∗) −A(Sd
M −k(vM, vM))AT ,
(13)
where vM refers to the inducing points of the respective
modality, A = k(x∗, vM)k(vM, vM)−1, with dimensions
k(vM, vM) ∈RM×M and k(x∗, vM) ∈RM and mvM is
the prior mean evaluated at vM.
Uncertainty Quantification. When an embedding z∗be-
longs to an ambiguous input, the uncertainty associated with
the posterior distribution p(x∗|z∗, zM) increases. This un-
certainty is propagated to the predictive distribution, which
can be written as: p(g∗) =
R
p(g∗|x∗)p(x∗|z∗, zM)dx∗.
Here, p(g∗|x∗) is a multivariate Gaussian distribution that
describes the function values at the fixed point x∗. Thus, a
large uncertainty in x∗increases the variance of the predic-
tive distribution p(g∗|z∗, zM). The uncertainty is captured
by ˆΣ∗, which accounts for variance contributions from both
the latent space uncertainty and inherent noise in the model’s
predictions. The final uncertainty is obtained by averaging
the uncertainty values across all dimensions in ˆΣ∗.
4
EXPERIMENTS
4.1
EXPERIMENTAL SETUP
Baselines and Datasets. We evaluate GroVE against six
baseline methods: Deterministic, TTDA [Ayhan and Berens,
2018], PFE [Shi and Jain, 2019], PCME [Chun et al., 2021],
PCME++ [Chun, 2023], and ProbVLM [Upadhyay et al.,
2023], using two VLMs—CLIP [Radford et al., 2021] and
BLIP [Li et al., 2022b]—with a focus on uncertainty cali-
bration for downstream tasks. In the deterministic approach,
uncertainty is quantified by the cosine distance between the
image and text embeddings derived from the VLM. While
PFE, PCME and PCME++ are methods to learn probabilis-
tic embeddings for pre-training VLMs, we follow Upadhyay
et al. [2023], and adapt them to work in a post-hoc man-
ner. The similarity ranking between probabilistic image and
text embeddings is determined by the Wasserstein distance,
with embeddings ranked based on the increasing distance,
while for the deterministic embeddings, the cosine similar-
ity is used. The implementation details for these methods
are provided in Appendix A.2. The methods are evaluated
on MS-COCO [Chen et al., 2015], Flickr30k [Young et al.,
2014], CUB-200-2011 [Wah et al., 2011] and Oxford Flow-
ers 102 [Nilsback and Zisserman, 2008] for cross-modal re-
trieval, and VQA2.0 [Goyal et al., 2017] for visual question
answering. The captions for the CUB and Flowers datasets
were obtained from Reed et al. [2016].
Evaluation Metrics. The cross-modal retrieval is evalu-
ated using the Recall@1 metric. For evaluating uncertainty
calibration, we adopt the metrics used in Upadhyay et al.
[2023], which computes the Spearman rank correlation (S)
between different uncertainty levels and Recall@1, R2 value
for the regression between the uncertainty levels and the
Recall@1 performance, and their product −SR2. For an
ideal model, the Recall@1 score should decrease with in-
creasing uncertainty, resulting in a S value of -1. A higher
R2 score indicates that with increasing uncertainty levels,
the model’s performance declines linearly. A higher −SR2
score implies better uncertainty calibration, reflecting both
a strong negative correlation and a monotonic decrease in
performance with increasing uncertainty. VQA is evalu-
ated using the soft voting accuracy of 10 human-annotated
answers [Goyal et al., 2017]. Calibration is evaluated by
the Expected Calibration Error (ECE) score between the
model’s confidence and the soft voting accuracy. Model
confidence is computed by first predicting an uncertainty
score u(a) for each candidate answer a, and then applying a
softmax function over these uncertainty scores. The model
confidence is given by conf(a) = 1 −softmax(u(a)).
Implementation details. The experiments on CLIP were
conducted using the ViT-B/32 model as the image encoder,
with D = 512. For BLIP, we adopt the ViT-B architecture
as the image encoder. We trained the GPs with a latent space


<!-- page 6 -->
Method
Flickr
COCO
CUB
Flowers
S ↓
R2 ↑
−SR2 ↑
S ↓
R2 ↑
−SR2 ↑
S ↓
R2 ↑
−SR2 ↑
S ↓
R2 ↑
−SR2 ↑
Image to Text
Deterministic
-0.80±0.00
0.66±0.00
0.52±0.00
-0.80±0.00
0.64±0.00
0.51±0.00
-0.10±0.00
0.05±0.00
0.00±0.00
-0.10±0.00
0.00±0.00
0.00±0.00
TTDA
0.12±0.03
0.32±0.07
-0.03±0.01
-0.36±0.05
0.38±0.08
0.17±0.05
-0.60±0.00
0.36±0.07
0.21 ±0.04
-0.78±0.04
0.37±0.07
0.28±0.06
PFE
-0.34±0.06
0.45±0.04
0.13±0.03
0.63±0.05
0.72±0.07
-0.46±0.05
-0.13±0.04
0.28±0.03
0.02±0.01
-0.11±0.05
0.29±0.04
0.04±0.01
PCME
0.61±0.06
0.18±0.02
-0.11±0.02
-0.63±0.00
0.50±0.03
0.31±0.02
-0.19±0.05
0.13±0.03
0.03±0.01
0.12±0.07
0.04±0.03
0.00±0.01
PCME++
-0.08±0.04
0.33±0.04
0.04±0.02
-0.30±0.07
0.37±0.04
0.10±0.03
-0.62±0.05
0.67±0.05
0.38±0.05
-0.61±0.11
0.55±0.04
0.32±0.08
ProbVLM
-0.79±0.05
0.52±0.04
0.38±0.04
-0.72±0.04
0.21±0.02
0.14±0.02
-0.33±0.05
0.46±0.04
0.15±0.02
-0.78±0.03
0.47±0.03
0.36±0.03
GroVE
-0.87±0.06
0.85±0.04
0.77±0.05
-0.90±0.03
0.88±0.04
0.79±0.02
-0.61±0.07
0.75±0.04
0.46±0.06
-0.88±0..04
0.81±0.01
0.70±0.03
Text to Image
Deterministic
-0.90±0.00
0.80±0.00
0.73±0.00
-0.80±0.00
0.76±0.00
0.61±0.00
0.60±0.00
0.12±0.00
-0.06±0.00
-0.30±0.00
0.17±0.00
0.05±0.00
TTDA
0.08±0.06
0.02±0.06
0.01±0.01
-0.61±0.06
0.20±0.05
0.12±0.04
-0.53±0.05
0.64±0.03
0.32±0.02
0.04±0.01
0.04±0.00
0.00±0.02
PFE
-0.68±0.07
0.56±0.05
0.38±0.06
0.33±0.04
0.52±0.02
-0.16±0.02
-0.32±0.07
0.34±0.02
0.09±0.02
0.21±0.04
0.43±0.02
-0.10±0.02
PCME
0.18±0.08
0.42±0.02
-0.07±0.04
0.86±0.04
0.84±0.03
-0.74±0.05
0.57±0.04
0.05±0.00
-0.03±0.00
0.72±0.05
0.45±0.03
-0.29±0.03
PCME++
-0.13±0.04
0.06±0.03
0.01±0.00
0.02±0.07
0.38±0.02
0.01±0.03
-0.28±0.05
0.02±0.01
0.01±0.00
0.12±0.07
0.47±0.02
-0.06±0.04
ProbVLM
-0.54±0.03
0.68±0.07
0.34±0.03
0.09±0.02
0.11±0.04
-0.01±0.00
-0.92±0.03
0.52±0.05
0.48±0.04
-0.60±0.03
0.16±0.06
0.10±0.03
GroVE
-0.92±0.04
0.74±0.04
0.66±0.04
-0.81±0.02
0.81±0.01
0.65±0.02
-0.78±0.07
0.60±0.02
0.49±0.05
-0.62±0.08
0.66±0.04
0.43±0.06
Table 1: Uncertainty calibration for cross-modal retrieval using CLIP. GroVE demonstrates superior performance in
uncertainty calibration in majority cases compared to baseline models. The best scores are highlighted in bold and the
second-best scores are underlined.
Method
Flickr
COCO
CUB
Flowers
S ↓
R2 ↑
−SR2 ↑
S ↓
R2 ↑
−SR2 ↑
S ↓
R2 ↑
−SR2 ↑
S ↓
R2 ↑
−SR2 ↑
Image to Text
Deterministic
-0.70±0.00
0.78±0.00
0.55±0.00
-0.80±0.00
0.84±0.00
0.67±0.00
0.50±0.00
0.13±0.00
-0.07±0.00
-0.20±0.00
0.05±0.00
0.01±0.00
TTDA
-0.68±0.04
0.27±0.03
0.19±0.02
-0.72±0.05
0.48±0.04
0.32±0.04
-0.70±0.05
0.50±0.02
0.33±0.03
-0.63±0.04
0.24±0.03
0.13±0.02
PFE
0.12±0.06
0.37±0.02
-0.04±0.02
0.04±0.00
0.32±0.05
0.00±0.00
0.56±0.06
0.53±0.04
0.32±0.03
0.13±0.04
0.02±0.03
-0.01±0.03
PCME
-0.31±0.06
0.17±0.04
0.05±0.03
-0.62±0.03
0.24±0.02
0.14±0.02
-0.64±0.03
0.63±0.03
0.38±0.03
0.08±0.03
0.25±0.04
-0.03±0.02
PCME++
-0.68±0.03
0.26±0.03
0.18±0.03
-0.69±0.04
0.50±0.04
0.34±0.03
-0.71±0.04
0.57±0.03
0.40±0.03
-0.69±0.06
0.53±0.02
0.37±0.02
ProbVLM
0.03±0.07
0.48±0.02
0.02±0.02
-0.61±0.03
0.50±0.04
0.30±0.03
-0.68±0.06
0.60±0.03
0.42±0.04
-0.67±0.00
0.65±0.02
0.46±0.01
GroVE
-0.72±0.03
0.74±0.02
0.51±0.03
-0.93±0.05
0.76±0.03
0.68±0.03
-0.89±0.04
0.60±0.04
0.54±0.02
-0.72±0.07
0.72±0.06
0.50±0.05
Text to Image
Deterministic
-0.90±0.00
0.88±0.00
0.79±0.00
-0.90±0.00
0.88±0.00
0.80±0.00
0.40±0.00
0.06±0.00
0.02±0.00
-0.10±0.00
0.00±0.00
0.00±0.00
TTDA
-0.37±0.03
0.35±0.04
0.14±0.03
0.41±0.06
0.00±0.01
0.00±0.03
-0.68±0.05
0.48±0.06
0.34±0.05
0.09±0.03
0.43±0.02
-0.04±0.04
PFE
-0.58±0.04
0.50±0.03
0.30±0.04
0.11±0.05
0.15±0.04
-0.02±0.02
-0.78±0.03
0.58±0.02
0.47±0.02
-0.23±0.06
0.01±0.03
0.00±0.03
PCME
-0.12±0.04
0.50±0.02
0.05±0.01
0.62±0.03
0.42±0.06
-0.25±0.03
-0.68±0.04
0.58±0.03
0.41±0.02
-0.31±0.03
0.26±0.03
0.08±0.04
PCME++
-0.72±0.06
0.30±0.04
0.21±0.04
-0.48±0.03
0.31±0.02
0.15±0.03
-0.12±0.08
0.00±0.04
0.00±0.02
-0.20±0.07
0.06±0.06
0.01±0.03
ProbVLM
-0.56±0.04
0.50±0.03
0.31±0.03
-0.12±0.05
0.48±0.04
0.05±0.04
-0.43±0.03
0.50±0.02
0.18±0.03
0.38±0.03
0.02±0.04
-0.01±0.04
GroVE
-0.92±0.04
0.90±0.03
0.81±0.04
-0.62±0.03
0.36±0.06
0.22±0.04
-0.89±0.02
0.75±0.03
0.74±0.03
-0.73±0.03
0.62±0.04
0.44±0.02
Table 2: Uncertainty calibration for cross-modal retrieval using BLIP. GroVE demonstrates superior performance in
uncertainty calibration in majority cases compared to baseline models. The best scores are highlighted in bold and the
second-best scores are underlined.
dimension of Q = 5 for MS-COCO, Flickr30k and VQA2.0,
and Q = 10 for CUB and Flowers alongside trade-off pa-
rameters λ1 = 0.01 and λ2 = 400 and 250 inducing points,
determined through grid search. The models were imple-
mented with GPyTorch Gardner et al. [2018], and trained for
200 epochs using the Adam optimizer with a learning rate of
1e−5 and a batch size of 64. The detailed implementation,
including data processing and hyper-parameter tuning is
provided in Appendix A.1 and A.3 respectively.
4.2
UNCERTAINTY CALIBRATION IN
CROSS-MODAL RETRIEVAL
Quantitative Results. The uncertainty calibration results
for CLIP and BLIP is provided in Table 1 and Table 2 respec-
tively. We observe that GroVE demonstrates superior per-
formance across all four datasets in both image-to-text and
text-to-image retrieval tasks, outperforming other methods
in most cases. A high −SR2 value for GroVE indicates that
the model maintains strong performance when uncertainty is
low, and the decline in performance is well-aligned with in-
creasing uncertainty scores, indicating effective uncertainty
calibration. Interestingly, the Deterministic baseline also
performs competitively on the Flickr30k and MS-COCO
datasets. This is because the VLMs were trained on datasets
with common real-world objects, well-represented in these
datasets, allowing the deterministic approach to benefit from
familiar image-text pair contexts. However, on fine-grained
datasets like CUB and Flowers, which are less represented
in the training data, it exhibits a noticeable drop in perfor-
mance. In these cases, the probabilistic methods outperform
the deterministic approach, with GroVE consistently leading
across both common object and fine-grained datasets.
Qualitative Results. Given a query image from MS-COCO,
we obtain its probabilistic embedding using GroVE. Using
the distribution of this embedding, we compute the likeli-
hood of each image in the Flickr30k dataset. Figure 3 shows
a t-SNE plot of the mean embeddings on Flickr30k, col-
ored by likelihood scores. The query image depicts children
playing on a field. We observe that the images with the high-
est likelihood scores, share similar semantic content, such
as scenes of people playing in fields. In contrast, images
with lower likelihood values (close to 0.0) show little to no
semantic or visual similarity to the query.

[CAPTION] Table 1: Uncertainty calibration for cross-modal retrieval using CLIP. GroVE demonstrates superior performance in

[CAPTION] Table 2: Uncertainty calibration for cross-modal retrieval using BLIP. GroVE demonstrates superior performance in


<!-- page 7 -->
Query
Likelihood
Figure 3: Given a probabilistic query image embedding from
COCO, the plot shows a t-SNE visualization of Flickr30k
embeddings, colored by their likelihood of belonging to
the query distribution. Sample images are shown in colored
boxes, where images with high likelihoods share similar
semantic and visual content to the query.
Additional results containing the retrieval performance, zero-
shot performance, calibration plots and qualitative analysis
is provided in Appendix. C.1 and C.2.
4.3
ACTIVE LEARNING
The objective of this experiment is to fine-tune the CLIP
model on the CUB dataset with limited labeled data. We
estimate the uncertainty of image and text embeddings to
identify the most uncertain samples from the unlabeled CUB
dataset, which are labeled for fine-tuning. For methods us-
ing auxiliary models, we derive uncertainty estimates from
models trained on COCO. We sample the top 500 uncer-
tain samples at each step for fine-tuning with contrastive
loss. A random sampling baseline is also included. Figure 4
provides the Recall@1 scores achieved in relation to the
number of samples used for fine-tuning the CLIP model.
GroVE achieves consistently better performance compared
to others, demonstrating that its uncertainty estimates effec-
tively identify the informative samples for active learning.
4.4
UNCERTAINTY IN FEW-SHOT SETTING
In this experiment, we explore a practical scenario where
labeled training data is scarce. To simulate this, we cre-
ate a few-shot dataset by randomly selecting three images
and their corresponding text descriptions from 150 classes
of the CUB dataset as done by Verma et al. [2021]. The
probabilistic adapters were trained on this dataset using em-
beddings obtained from CLIP, and the uncertainty calibra-
tion was evaluated for cross-modal retrieval. Table 3 shows
the −SR2 scores obtained for the baselines and GroVE
Random
Deterministic
TTDA
PFE
PCME
PCME++
ProbVLM
GroVE
500
1000 1500 2000 2500 3000 3500 4000 4500
Number of samples
0.18
0.20
0.22
0.24
0.26
0.28
Recall@1
500
1000 1500 2000 2500 3000 3500 4000 4500
Number of samples
0.12
0.13
0.14
0.15
0.16
0.17
0.18
0.19
Recall@1
Figure 4: Active Learning. The results highlight GroVE’s
ability to effectively leverage uncertainty estimates to guide
sample selection, outperforming the baselines on both
image-to-text (left) and text-to-image (right) retrieval.
Method
Image to Text
Text to Image
TTDA
0.03±0.06
0.01±0.04
PFE
-0.29±0.02
-0.22±0.03
PCME
0.04±0.03
0.14±0.02
PCME++
0.27±0.03
0.01±0.04
ProbVLM
-0.12±0.04
-0.43±0.04
GroVE (M=50)
0.24±0.03
0.22±0.03
GroVE (M=150)
0.36±0.04
0.31±0.02
GroVE (M=250)
0.35±0.03
0.31±0.03
GroVE (exact GP)
0.39±0.03
0.36±0.02
Table 3: Few-shot uncertainty calibration. GroVE out-
performs other baselines, achieving superior uncertainty
calibration in few-shot settings in terms of -SR2 (↑).
with different numbers of inducing points as well as ex-
act GP models, where training and inference is performed
without approximations [Williams and Rasmussen, 2006].
The results show that while the calibration performance im-
proves as the number of inducing points increases, GroVE
consistently outperforms the baselines in terms of calibra-
tion quality. The best performance was achieved with exact
GP models and no approximation. A comparison of the re-
trieval performance and the inference time is provided in
Appendix C.3.
4.5
UNCERTAINTY CALIBRATION FOR VQA
Table 4 shows the accuracy and ECE scores obtained for
VQA2.0 using BLIP as the VLM. All the baselines achieve
similar accuracy scores, with deterministic achieving the


**[Table p7.1]**
| 0.19 0.18 0.17 Recall@1 0.16 0.15 0.14 0.13 0.12 500 1000 1500 2000 2500 3000 3500 4000 4500 Number of samples |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |

[CAPTION] Figure 3: Given a probabilistic query image embedding from

[CAPTION] Figure 4: Active Learning. The results highlight GroVE’s

[CAPTION] Table 3: Few-shot uncertainty calibration. GroVE out-

[CAPTION] Table 4 shows the accuracy and ECE scores obtained for


<!-- page 8 -->
Method
Accuracy ↑
ECE ↓
Determinsitic
78.20
0.56
TTDA
77.67±2.23
0.48±0.06
PFE
76.34±1.98
0.65±0.02
PCME
77.25±1.76
0.64±0.01
PCME++
77.53±1.71
0.64±0.02
ProbVLM
76.66±1.13
0.69±0.01
GroVE
77.48±2.15
0.24±0.04
Table 4: Results for VQA. While all methods achieve simi-
lar accuracy (with the deterministic model performing best),
GroVE reaches the best calibration performance in terms of
ECE (↓).
Kernel
Image to Text
Text to Image
COCO
CUB
COCO
CUB
RBF
0.79±0.02
0.46±0.06
0.65±0.02
0.49±0.05
Matérn (ν = 1.5)
0.27±0.03
0.47±0.05
0.41±0.04
0.22±0.04
Matérn (ν = 2.5)
0.52±0.05
0.38±0.04
0.43±0.04
0.12±0.05
Cosine Similarity
0.46±0.04
0.39±0.03
0.35±0.03
0.30±0.02
Table 5: Ablation on choice of GP kernel. GroVE achieves
the best performance on MS-COCO and CUB-200-2011
with the RBF kernel.
best accuracy. When evaluated for confidence calibration,
GroVE achieves the lowest ECE score.
4.6
ABLATION ANALYSIS
GP Kernel. We evaluate the performance of the RBF,
Matérn (ν = 1.5 and 2.5, where ν is the smoothness param-
eter) and the cosine similarity kernel on GroVE’s perfor-
mance on the MS-COCO and CUB data. From Table 5, the
RBF kernel achieves superior performance compared to the
other kernels across both datasets, with improvements up to
53%. The kernels are defined in Appendix B.1.
Latent Space Dimension. We investigate the influence
0
5
10
15
20
Dimension
0.4
0.5
0.6
0.7
0.8
SR2
Image to Text
Text to Image
0.2
0.4
0.6
0.8
1.0
Fraction of full data
0.3
0.4
0.5
0.6
0.7
0.8
SR2
Image to Text
Text to Image
Figure 5: Ablation using MS-COCO: (i) latent space di-
mension (left). Low latent space dimensions results in loss
of information, while higher dimensions results in perfor-
mance degradation due to over-fitting. (ii) dataset size for
training (right). GroVE achieves good performance with
just 60% of the total training dataset.
0
200
400
600
800
1000
Weight
0.50
0.55
0.60
0.65
0.70
0.75
0.80
SR2
KL-div Image to Text
KL-div Text to Image
MSE Image to Text
MSE Text to Image
0.2
0.4
0.6
0.8
1.0
Masking fraction
5
10
15
20
25
Uncertainty
ViT B32
ViT B16
ResNet50
Text encoder
Figure 6: Ablation using MS-COCO: (i) trade-off parame-
ter (left). KL-divergence improves uncertainty calibration
with optimal performance at λ2 = 400, with λ1 = 0.01. (ii)
noisy data (right). With increasing amount of noise in the
input data, GroVE predicts higher uncertainty.
of the latent space dimension Q on GroVE’s performance
using the MS-COCO dataset. Figure 5 (left) presents the
−SR2 scores for various values of Q. Low values of Q lead
to information loss, which compromises the model’s ability
to capture complex patterns in the data. Conversely, high
values of Q result in overfitting and make the model more
challenging to optimize, resulting in a performance decline.
The optimal performance was observed when Q = 5.
Dataset Size. We study the impact of the dataset size on
GroVE’s performance by training it on various fractions of
the COCO training dataset. As shown in Figure 5 (right),
the model achieves good performance when trained on 60%
of the full dataset. While the uncertainty calibration per-
formance for text-to-image retrieval plateaus beyond this
point, the performance for image-to-text retrieval continues
to improve almost linearly as more data is utilized.
Cross-modal Alignment.
We compare GroVE’s KL-
divergence-based alignment loss with the MSE loss-based
regularization used in Song et al. [2017]. The authors use
GPLVM for cross-modal retrieval, regularizing the latent
space with the loss function ||kI −S||2 +||kT −S||2, where
kI and kT are the GP covariance matrices, and S is the latent
space similarity matrix. For comparison, we replace LKL in
our method with the MSE loss and experiment with different
values of the trade-off parameter λ2, maintaining λ1 = 0.01.
As shown in Figure 6 (left), the KL-divergence alignment
loss improves uncertainty calibration performance by up
to 23%, with the best performance at λ2 = 400. Addition-
ally, we evaluate the cross-modal alignment KL-divergence
loss against other widely-used probabilistic distance metrics:
Jensen-Shannon (JS) divergence and Wasserstein-2 distance.
The results in Table 6 indicate that while all metrics perform
similarly, KL-divergence offers a slight edge. The distance
metrics are defined in Appendix B.2.
Noisy Data. To evaluate the performance of GroVE against
noisy inputs, we systematically introduce increasing levels


**[Table p8.1]**
| 0.80 0.75 0.70 SR2 0.65 0.60 0.55 0.50 0 200 400 600 800 1000 Weight KL-div Image to Text MSE Image to Text KL-div Text to Image MSE Text to Image | 25 20 Uncertainty 15 10 5 0.2 0.4 0.6 0.8 1.0 Masking fraction ViT B32 ResNet50 ViT B16 Text encoder |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |


**[Table p8.2]**
| 0.8 Image to Text Text to Image 0.7 SR2 0.6 0.5 0.4 0 5 10 15 20 Dimension |  |  |  |  | 0.8 Image to Text Text to Image 0.7 0.6 SR2 0.5 0.4 0.3 0.2 0.4 0.6 0.8 1.0 Fraction of full data |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  | Ima Text | ge to Text to Image |  | Ima Tex | ge to Tex t to Imag | t e |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |

[CAPTION] Table 4: Results for VQA. While all methods achieve simi-

[CAPTION] Table 5: Ablation on choice of GP kernel. GroVE achieves

[CAPTION] Figure 5: Ablation using MS-COCO: (i) latent space di-

[CAPTION] Figure 6: Ablation using MS-COCO: (i) trade-off parame-


<!-- page 9 -->
Kernel
Image to Text
Text to Image
COCO
CUB
COCO
CUB
KL-Divergence
0.79±0.02
0.46±0.06
0.65±0.02
0.49±0.05
JS-Divergence
0.70±0.04
0.48±0.02
0.59±0.03
0.44±0.05
Wasserstein-2
0.59±0.04
0.39±0.04
0.60±0.02
0.43±0.04
Table 6: Ablation on probabilistic distance metric. GroVE
performs better for cross-modal alignment using KL-
Divergence compared to other metrics.
of masking to both the input images and texts. This analysis
employs several CLIP image encoder backbones, including
ViT-B/32, ViT-B/16, and ResNet50, along with CLIP’s text
encoder. The results, presented in Figure 6 (right), indicate
that as the noise level increases, the uncertainty predicted
by GroVE rises steadily as desired.
5
CONCLUSION
This paper introduces GroVE, a post-hoc approach for gen-
erating probabilistic embeddings from frozen, pre-trained
VLMs to model input data ambiguities. GroVE leverages
the GPLVM framework, utilizing GP models to learn a
shared, low-dimensional latent space that aligns visual and
textual representations. By mapping into this latent space,
the GP models generate probabilistic embeddings that pro-
vide a measure of uncertainty in the predictions. GroVE
demonstrates state-of-the-art performance in uncertainty
calibration for cross-modal retrieval, active learning and
VQA. One limitation of GroVE is the it is computationally
expensive compared to the neural network based methods
(see Appendix. C.3). In latency-sensitive scenarios, such
as real-time applications, neural network-based stochastic
models like Neural Processes [Garnelo et al., 2018] offer a
viable alternative to GPs. Future work will focus on assess-
ing their uncertainty calibration performance for VLMs.
References
Moloud Abdar, Farhad Pourpanah, Sadiq Hussain, Dana
Rezazadegan, Li Liu, Mohammad Ghavamzadeh, Paul
Fieguth, Xiaochun Cao, Abbas Khosravi, U Rajendra
Acharya, et al. A review of uncertainty quantification in
deep learning: Techniques, applications and challenges.
Information fusion, 76:243–297, 2021.
Murat Seckin Ayhan and Philipp Berens. Test-time data
augmentation for estimation of heteroscedastic aleatoric
uncertainty in deep neural networks. In Medical Imaging
with Deep Learning, 2018.
Björn Barz and Joachim Denzler. Hierarchy-based image
embeddings for semantic image retrieval. In 2019 IEEE
winter conference on applications of computer vision
(WACV), pages 638–647. IEEE, 2019.
Charles Blundell, Julien Cornebise, Koray Kavukcuoglu,
and Daan Wierstra. Weight uncertainty in neural network.
In International conference on machine learning, pages
1613–1622. PMLR, 2015.
Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedan-
tam, Saurabh Gupta, Piotr Dollár, and C Lawrence Zit-
nick. Microsoft coco captions: Data collection and evalu-
ation server. arXiv preprint arXiv:1504.00325, 2015.
Sanghyuk Chun. Improved probabilistic image-text repre-
sentations. arXiv preprint arXiv:2305.18171, 2023.
Sanghyuk Chun, Seong Joon Oh, Rafael Sampaio
De Rezende, Yannis Kalantidis, and Diane Larlus. Proba-
bilistic embeddings for cross-modal retrieval. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 8415–8424, 2021.
Charles Corbiere, Nicolas Thome, Antoine Saporta, Tuan-
Hung Vu, Matthieu Cord, and Patrick Perez. Confidence
estimation via auxiliary models. IEEE Transactions on
Pattern Analysis and Machine Intelligence, 44(10):6043–
6055, 2021.
John Duchi. Derivations for linear algebra and optimization.
Berkeley, California, 3(1):2325–5870, 2007.
Stefanos Eleftheriadis, Ognjen Rudovic, and Maja Pantic.
Discriminative shared gaussian processes for multiview
and view-invariant facial expression recognition. IEEE
transactions on image processing, 24(1):189–204, 2014.
Andrea Frome, Greg S Corrado, Jon Shlens, Samy Bengio,
Jeff Dean, Marc’Aurelio Ranzato, and Tomas Mikolov.
Devise: A deep visual-semantic embedding model. Ad-
vances in neural information processing systems, 26,
2013.
Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian
approximation: Representing model uncertainty in deep
learning. In international conference on machine learn-
ing, pages 1050–1059. PMLR, 2016.
Jacob Gardner, Geoff Pleiss, Kilian Q Weinberger, David
Bindel, and Andrew G Wilson.
Gpytorch: Blackbox
matrix-matrix gaussian process inference with gpu ac-
celeration. Advances in neural information processing
systems, 31, 2018.
Marta Garnelo, Jonathan Schwarz, Dan Rosenbaum, Fabio
Viola, Danilo J Rezende, SM Eslami, and Yee Whye
Teh. Neural processes. arXiv preprint arXiv:1807.01622,
2018.
Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv
Batra, and Devi Parikh. Making the v in vqa matter:
Elevating the role of image understanding in visual ques-
tion answering. In Proceedings of the IEEE conference
on computer vision and pattern recognition, pages 6904–
6913, 2017.

[CAPTION] Table 6: Ablation on probabilistic distance metric. GroVE


<!-- page 10 -->
Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger.
On calibration of modern neural networks. In Interna-
tional conference on machine learning, pages 1321–1330.
PMLR, 2017.
Michael Gutmann and Aapo Hyvärinen. Noise-contrastive
estimation: A new estimation principle for unnormalized
statistical models. In Proceedings of the thirteenth inter-
national conference on artificial intelligence and statis-
tics, pages 297–304. JMLR Workshop and Conference
Proceedings, 2010.
James Hensman, Alexander Matthews, and Zoubin Ghahra-
mani. Scalable variational gaussian process classification.
In Artificial Intelligence and Statistics, pages 351–360.
PMLR, 2015.
Julia Hornauer, Adrian Holzbock, and Vasileios Belagiannis.
Out-of-distribution detection for monocular depth esti-
mation. In Proceedings of the IEEE/CVF International
Conference on Computer Vision, pages 1911–1921, 2023.
Yatai Ji, Junjie Wang, Yuan Gong, Lin Zhang, Yanru Zhu,
Hongfa Wang, Jiaxing Zhang, Tetsuya Sakai, and Yu-
jiu Yang. Map: Multimodal uncertainty-aware vision-
language pre-training model.
In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 23262–23271, 2023.
Myong Chol Jung, He Zhao, Joanna Dipnall, Belinda Gabbe,
and Lan Du. Uncertainty estimation for multi-view data:
The power of seeing the whole picture. Advances in
Neural Information Processing Systems, 35:6517–6530,
2022.
Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina
Toutanova. Bert: Pre-training of deep bidirectional trans-
formers for language understanding. In Proceedings of
naacL-HLT, volume 1, page 2, 2019.
Vidhi Lalchand, Aditya Ravuri, and Neil D Lawrence. Gen-
eralised gaussian process latent variable models (gplvm)
with stochastic variational inference.
arXiv preprint
arXiv:2202.12979, 2022.
Neil Lawrence. Gaussian process latent variable models
for visualisation of high dimensional data. Advances in
neural information processing systems, 16, 2003.
Dong Li, Hsin-Ying Lee, Jia-Bin Huang, Shengjin Wang,
and Ming-Hsuan Yang.
Learning structured seman-
tic embeddings for visual recognition. arXiv preprint
arXiv:1706.01237, 2017.
Hao Li, Jingkuan Song, Lianli Gao, Pengpeng Zeng, Haonan
Zhang, and Gongfu Li. A differentiable semantic met-
ric approximation in probabilistic embedding for cross-
modal retrieval. Advances in Neural Information Process-
ing Systems, 35:11934–11946, 2022a.
Jiaxing Li, Wai Keung Wong, Lin Jiang, Xiaozhao Fang,
Shengli Xie, and Yong Xu. Ckdh: Clip-based knowledge
distillation hashing for cross-modal retrieval. IEEE Trans-
actions on Circuits and Systems for Video Technology,
2024.
Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi.
Blip: Bootstrapping language-image pre-training for uni-
fied vision-language understanding and generation. In In-
ternational conference on machine learning, pages 12888–
12900. PMLR, 2022b.
Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh,
and Kai-Wei Chang. Visualbert: A simple and perfor-
mant baseline for vision and language. arXiv preprint
arXiv:1908.03557, 2019.
Xiang Li, Congcong Wen, Yuan Hu, and Nan Zhou. Rs-
clip: Zero shot remote sensing scene classification via
contrastive vision-language supervision. International
Journal of Applied Earth Observation and Geoinforma-
tion, 124:103497, 2023.
Jiayi Lin and Shaogang Gong. Gridclip: One-stage object
detection by grid-level clip representation learning. arXiv
preprint arXiv:2303.09252, 2023.
Jeremiah Liu, Zi Lin, Shreyas Padhy, Dustin Tran, Tania
Bedrax Weiss, and Balaji Lakshminarayanan. Simple
and principled uncertainty estimation with deterministic
deep learning via distance awareness. Advances in neural
information processing systems, 33:7498–7512, 2020.
Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vil-
bert: Pretraining task-agnostic visiolinguistic representa-
tions for vision-and-language tasks. Advances in neural
information processing systems, 32, 2019.
Jishnu Mukhoti, Andreas Kirsch, Joost van Amersfoort,
Philip HS Torr, and Yarin Gal. Deep deterministic un-
certainty: A new simple baseline. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 24384–24394, 2023.
Maria-Elena Nilsback and Andrew Zisserman. Automated
flower classification over a large number of classes. In
2008 Sixth Indian conference on computer vision, graph-
ics & image processing, pages 722–729. IEEE, 2008.
Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Rep-
resentation learning with contrastive predictive coding.
arXiv preprint arXiv:1807.03748, 2018.
Maria Parelli, Alexandros Delitzas, Nikolas Hars, Geor-
gios Vlassis, Sotirios Anagnostidis, Gregor Bachmann,
and Thomas Hofmann. Clip-guided vision-language pre-
training for question answering in 3d scenes. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 5607–5612, 2023.


<!-- page 11 -->
John Platt et al. Probabilistic outputs for support vector ma-
chines and comparisons to regularized likelihood meth-
ods. Advances in large margin classifiers, 10(3):61–74,
1999.
Qi Qian and Juhua Hu. Online zero-shot classification with
clip. arXiv preprint arXiv:2408.13320, 2024.
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sas-
try, Amanda Askell, Pamela Mishkin, Jack Clark, et al.
Learning transferable visual models from natural lan-
guage supervision. In International conference on ma-
chine learning, pages 8748–8763. PMLR, 2021.
Scott Reed, Zeynep Akata, Honglak Lee, and Bernt Schiele.
Learning deep representations of fine-grained visual de-
scriptions. In Proceedings of the IEEE conference on
computer vision and pattern recognition, pages 49–58,
2016.
Murat Sensoy, Lance Kaplan, and Melih Kandemir. Eviden-
tial deep learning to quantify classification uncertainty.
Advances in neural information processing systems, 31,
2018.
Yichun Shi and Anil K Jain. Probabilistic face embeddings.
In Proceedings of the IEEE/CVF International Confer-
ence on Computer Vision, pages 6902–6911, 2019.
Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guil-
laume Couairon, Wojciech Galuba, Marcus Rohrbach,
and Douwe Kiela. Flava: A foundational language and
vision alignment model. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 15638–15650, 2022.
Guoli Song, Shuhui Wang, Qingming Huang, and Qi Tian.
Multimodal gaussian process latent variable models with
harmonization. In Proceedings of the IEEE international
conference on computer vision, pages 5029–5037, 2017.
Hao Tan and Mohit Bansal.
Lxmert: Learning cross-
modality encoder representations from transformers.
arXiv preprint arXiv:1908.07490, 2019.
Michalis Titsias. Variational learning of inducing variables
in sparse gaussian processes. In Artificial intelligence
and statistics, pages 567–574. PMLR, 2009.
Uddeshya Upadhyay, Shyamgopal Karthik, Massimiliano
Mancini, and Zeynep Akata.
Probvlm: Probabilistic
adapter for frozen vison-language models. In Proceedings
of the IEEE/CVF International Conference on Computer
Vision, pages 1899–1910, 2023.
Joost Van Amersfoort, Lewis Smith, Yee Whye Teh, and
Yarin Gal. Uncertainty estimation using a single deep
deterministic neural network. In International conference
on machine learning, pages 9690–9700. PMLR, 2020.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszko-
reit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
Illia Polosukhin. Attention is all you need. Advances in
Neural Information Processing Systems, 30, 2017.
Aishwarya Venkataramanan, Martin Laviale, Cécile Figus,
Philippe Usseglio-Polatera, and Cédric Pradalier. Tack-
ling inter-class similarity and intra-class variance for mi-
croscopic image-based classification. In International
conference on computer vision systems, pages 93–103.
Springer, 2021.
Aishwarya Venkataramanan, Assia Benbihi, Martin Laviale,
and Cédric Pradalier. Gaussian latent representations
for uncertainty estimation using mahalanobis distance in
deep classifiers. In Proceedings of the IEEE/CVF Interna-
tional Conference on Computer Vision, pages 4488–4497,
2023a.
Aishwarya Venkataramanan, Martin Laviale, and Cédric
Pradalier.
Integrating visual and semantic similarity
using hierarchies for image retrieval. In International
Conference on Computer Vision Systems, pages 422–431.
Springer, 2023b.
Vinay Kumar Verma, Ashish Mishra, Anubha Pandey,
Hema A Murthy, and Piyush Rai. Towards zero-shot
learning with fewer seen class examples. In Proceedings
of the IEEE/CVF Winter Conference on Applications of
Computer Vision, pages 2241–2251, 2021.
Catherine Wah, Steve Branson, Peter Welinder, Pietro Per-
ona, and Serge Belongie. The caltech-ucsd birds-200-
2011 dataset. 2011.
Guotai Wang, Wenqi Li, Michael Aertsen, Jan Deprest,
Sébastien Ourselin, and Tom Vercauteren.
Aleatoric
uncertainty estimation with test-time augmentation for
medical image segmentation with convolutional neural
networks. Neurocomputing, 338:34–45, 2019.
Christopher KI Williams and Carl Edward Rasmussen.
Gaussian processes for machine learning, volume 2. MIT
press Cambridge, MA, 2006.
Xiaoshi Wu, Feng Zhu, Rui Zhao, and Hongsheng Li. Cora:
Adapting clip for open-vocabulary detection with region
prompting and anchor pre-matching. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 7031–7040, 2023.
Xinyu Xia, Guohua Dong, Fengling Li, Lei Zhu, and Xi-
aomin Ying. When clip meets cross-modal hashing re-
trieval: A new strong baseline. Information Fusion, 100:
101968, 2023.
Fengchuang Xing, Mingjie Li, Yuan-Gen Wang, Guopu Zhu,
and Xiaochun Cao. Clipvqa: Video quality assessment
via clip. arXiv preprint arXiv:2407.04928, 2024.


<!-- page 12 -->
Peter Young, Alice Lai, Micah Hodosh, and Julia Hocken-
maier. From image descriptions to visual denotations:
New similarity metrics for semantic inference over event
descriptions. Transactions of the Association for Compu-
tational Linguistics, 2:67–78, 2014.
Xuanlong Yu, Gianni Franchi, and Emanuel Aldea. Slurp:
Side learning uncertainty for regression problems. arXiv
preprint arXiv:2110.11182, 2021.
Jingyi Zhang, Jiaxing Huang, Sheng Jin, and Shijian Lu.
Vision-language models for vision tasks: A survey. IEEE
Transactions on Pattern Analysis and Machine Intelli-
gence, 2024.
Ziming Zhang and Venkatesh Saligrama. Zero-shot learning
via semantic similarity embedding. In Proceedings of the
IEEE international conference on computer vision, pages
4166–4174, 2015.


<!-- page 13 -->
Probabilistic Embeddings for Frozen Vision-Language Models: Uncertainty
Quantification with Gaussian Process Latent Variable Models
(Supplementary Material)
Aishwarya Venkataramanan1
Paul Bodesheim1
Joachim Denzler1
1Computer Vision Group, Friedrich Schiller University Jena, Germany
A
ADDITIONAL IMPLEMENTATION DETAILS
This section provides details on the data processing steps for training both the baseline models and GroVE, implementation
details for each baseline, and the hyper-parameter tuning procedure applied for GroVE.
A.1
DATASETS
For the experiments, we use MS-COCO, Flickr30k, CUB-200-2011, and Oxford Flowers 102 dataset.
MS-COCO Chen et al. [2015] is a widely used cross-modal retrieval dataset includes 123,287 images, each image annotated
with 5 captions describing common objects. The training set comprises 113,287 images, while both the validation and test
sets contain 5,000 images each. Different papers apply varying evaluation protocols on the 5,000 test images in the COCO
dataset. Some cross-modal retrieval papers report results on the full 5,000 test set, while others use 1,000 unique test images,
averaging results over 5 random splits. In our study, we follow the former approach, presenting results based on the entire
5,000 test set.
Flickr30k Young et al. [2014] is a widely used cross-modal retrieval dataset comprising 31,783 images, each image
annotated with 5 captions describing common objects. The dataset is split into 29,783 training images, with 1,000 images
each in the validation and test sets.
CUB-200-2011 Wah et al. [2011] is a fine-grained bird species dataset comprising 11,788 images across 200 categories,
with each image paired with 10 captions sourced from Reed et al. [2016]. Following the split protocol in Chun et al. [2021],
the dataset includes 7,067 training images, 1,754 validation images, and 2,967 test images.
Oxford Flowers 102 Nilsback and Zisserman [2008] is a fine-grained flowers dataset comprising 8,189 images across
102 categories, with each image paired with 10 captions sourced from Reed et al. [2016]. Following the split protocol in
Upadhyay et al. [2023], the dataset includes 7,034 training images, 750 validation and 805 test images.
All images are resized to 224 × 224, suitable for input to VLMs. All methods are trained and evaluated using the same
dataset splits for the comparison.
A.2
BASELINE METHODS
In this section, we provide the implementation and training details for the baseline methods.
TTDA Ayhan and Berens [2018]. During inference, data augmentations are applied to the input, generating multiple
variations to estimate prediction uncertainty. For each augmented version, a prediction is generated, and the variance across
these predictions reflects the model’s uncertainty. Image augmentations include random resized cropping and horizontal
flipping (applied with a probability of 0.3), while text data undergoes random word masking with a 0.3 probability. The
model is run for 10 passes on these augmentations, to obtain the image and text uncertainty.


<!-- page 14 -->
PFE Shi and Jain [2019], PCME Chun et al. [2021] and PCME++ Chun [2023]. During training, we adapt these
methods to process image and text embeddings derived from a frozen VLM. Following Upadhyay et al. [2023], the model
architecture consists of two Multi-Layer Perceptrons (MLPs)—one for images and one for text. Each MLP has an input
layer that reduces the embedding dimension to 256, a hidden layer of 256 units, and an output layer that maps from 256 back
to the original embedding dimension. We apply the respective loss functions to learn covariances for a Gaussian distribution,
with mean values matching the VLM’s deterministic embeddings. Training is conducted for 200 epochs using the Adam
optimizer with a learning rate of 10−8 and batch size of 64. The learning rate was fixed using a grid search over values
{1e−4, 1e−5, 1e−6, 1e−7, 1e−8}
ProbVLM Upadhyay et al. [2023]. We follow the training procedure outlined in the original paper. The model architecture
consists of two MLPs—one for image embeddings and one for text embeddings—similar to previous methods. Training is
conducted with the Adam optimizer for 100 epochs, using a learning rate of 10−4 and a batch size of 64.
A.3
HYPER-PARAMETER TUNING
GroVE introduces the following hyper-parameters which were obtained using grid-search: latent space dimension Q, and the
trade-off parameters λ1 and λ2. For Q, we evaluated values Q ∈{2, 5, 10, 20, 50, 128, 256}. For the trade-off parameters,
we used λ1 ∈{1, 0.1, 0.01, 0.001}, and λ2 ∈{0, 200, 400, 600, 800, 1000}. based on the grid-search results, the optimal
setting for Q was Q = 5 for MS-COCO and Flickr30k, and Q = 10 for CUB-200-2011 and Oxford Flowers 102. The
trade-off parameters that achieved the best performance were λ1 = 0.01 and λ2 = 400. The number of inducing points was
selected from {100, 150, 200, 250, 300, 350}, with model performance plateauing beyond 250 points. Finally, the learning
rate of 1e−5 was selected based on a grid-search over values {1e−1, 1e−2, 1e−3, 1e−4, 1e−5, 1e−6}.
B
DEFINITIONS
This section presents the definitions of the GP kernels and the probabilistic distance metrics used in the ablation study.
B.1
GP KERNELS
In Table. 5, we presented the results for ablation study of various kernels for GP. In this section, we provide the definitions
and formulas for the kernels used.
Radial Basis Function (RBF). The RBF kernel, also known as the Gaussian kernel, is a popular choice in GPs. It assumes
that closer data points in input space have higher similarity. The RBF kernel between two points xi and xj is defined as:
k(xi, xj) = exp

−∥xi −xj∥
l2

(14)
where l is the length scale parameter, controlling how quickly the similarity decreases with distance in input space.
Matérn. The Matérn kernel generalizes the RBF kernel, defined as
k(xi, xj) =
1
Γ(ν)2ν−1
 √
2ν
l
∥xi −xj∥
!ν
Kν
 √
2ν
l
∥xi −xj∥
!
(15)
where ν controls the smoothness of the resulting function, l is the length scale parameter, Γ is the Gamma function and Kν
is a modified Bessel function.
Cosine Similarity. This is a linear kernel with normalized inputs, and is defined as:
k(xi, xj) =
xiT xj
∥xi∥∥xj∥
(16)
B.2
PROBABILISTIC DISTANCES
Sec. 4.6 presented an ablation study on the choice of the probability distance metric for cross-modal alignment (refer Table. 6).
The definitions of the probabilistic distance metrics for two multivariate Gaussians p = N(ˆµI, ˆΣI) and q = N(ˆµT , ˆΣT ),
are as follows:


<!-- page 15 -->
Kullback-Liebler Divergence. The KL Divergence quantifies the difference between two probability distributions, and is
defined as Duchi [2007]:
DKL(p∥q) = 1
2
"
tr( ˆΣ
−1
T ˆΣI) + (ˆµT −ˆµI)T ˆΣ
−1
T (ˆµT −ˆµI) −D + log
 
det( ˆΣT )
det( ˆΣI)
! #
,
(17)
where tr(.) is the trace and det(.) is the determinant of a matrix. Note that the KL-Divergence is asymmetric; thus, we
calculate the total cross-alignment loss as the mean of the KL divergences in both directions (refer Eq. 8): 1
2[DKL(p∥q) +
DKL(q∥p)].
Jensen-Shannon (JS) Divergence. The JS Divergence is obtained by averaging the KL divergences between each distribution
and the average distribution. The JS divergence is defined as:
DJS(p∥q) = 1
2 (DKL(p∥m) + DKL(q∥m))
(18)
where m = 1
2(p + q) is the mean distribution of p and q.
Wasserstein-2 distance. The Wasserstein-2 distance quantifies the cost of transforming one distribution into another. This is
defined as:
W 2
2 (p, q) = ∥ˆµI −ˆµT ∥2 + tr

ˆΣI + ˆΣT −2

ˆΣ
1/2
I
ˆΣT ˆΣ
1/2
I
1/2
(19)
C
ADDITIONAL RESULTS
This section presents additional quantitative and qualitative results.
C.1
CROSS-MODAL RETRIEVAL
C.1.1
Calibration plots
Figure 7 and Figure 8 show the calibration plots for the CLIP and BLIP models, respectively. Calibration plots are obtained
by binning uncertainty values, referred to as uncertainty levels and computing Recall@1 for each bin. From the plots, GroVE
maintains a more consistent alignment between decreasing uncertainty and increasing Recall@1.
C.1.2
Retrieval performance
Table 7 presents the Recall@1 scores for various baselines using CLIP. The score for the Deterministic baseline was
computed by retrieving the nearest image/text embedding based on cosine similarity to the query text/image from the
deterministic embeddings generated by the CLIP model. For the other baselines, retrieval was performed by selecting the
image/text embedding with the minimum Wasserstein distance to the query, using the probabilistic image/text embeddings.
Results show that GroVE achieves a good performance on the fine-grained CUB and Flowers dataset, whereas deterministic
achieves the best scores in MS-COCO and Flickr30k dataset.
C.1.3
Qualitative Analysis
A t-SNE visualization of the probabilistic embeddings from GroVE on a subset of the CUB dataset is provided in Figure 9,
where the uncertainty corresponds to the area of the embedding. The plot shows that images and texts with similar semantic
content are clustered together, and the probabilistic embeddings are able to capture the uncertainty arising from the data
ambiguities. Figure 10 illustrates a scenario from the CUB-200-2011 dataset where incorrect predictions sometimes occur
due to high inter-class similarity Venkataramanan et al. [2021] between the image and text descriptions of two distinct bird
species. We also include a scenario where either the image or text is masked, introducing ambiguity. In such cases, GroVE
assigns a distribution with higher variance, reflecting increased uncertainty.

[CAPTION] Figure 7 and Figure 8 show the calibration plots for the CLIP and BLIP models, respectively. Calibration plots are obtained

[CAPTION] Table 7 presents the Recall@1 scores for various baselines using CLIP. The score for the Deterministic baseline was


<!-- page 16 -->
Flickr30k
MS-COCO
CUB
Flowers
Image to Text
1
2
3
4
5
Uncertainty level
0.0
0.2
0.4
0.6
0.8
1.0
Recall@1
1
2
3
4
5
Uncertainty level
0.0
0.2
0.4
0.6
0.8
1.0
Recall@1
1
2
3
4
5
Uncertainty level
0.00
0.05
0.10
0.15
0.20
0.25
0.30
Recall@1
1
2
3
4
5
Uncertainty level
0.0
0.2
0.4
0.6
0.8
1.0
Recall@1
Text to Image
1
2
3
4
5
Uncertainty level
0.0
0.2
0.4
0.6
0.8
1.0
Recall@1
1
2
3
4
5
Uncertainty level
0.0
0.2
0.4
0.6
0.8
1.0
Recall@1
1
2
3
4
5
Uncertainty level
0.000
0.025
0.050
0.075
0.100
0.125
0.150
0.175
0.200
Recall@1
1
2
3
4
5
Uncertainty level
0.0
0.1
0.2
0.3
0.4
0.5
0.6
Recall@1
Deterministic
TTDA
PFE
PCME
PCME++
ProbVLM
GroVE
Figure 7: Evaluation of uncertainty calibration for embeddings obtained from CLIP on Image-to-Text retrieval. For
perfect calibration, the Recall@1 should show a monotonic decrease as uncertainty levels increase. GroVE exhibits a more
consistent relationship between increasing uncertainty and performance degradation compared to the baseline methods.
Flickr
MS-COCO
CUB
Flowers
Image to Text
1
2
3
4
5
Uncertainty level
0.0
0.2
0.4
0.6
0.8
1.0
Recall@1
1
2
3
4
5
Uncertainty level
0.0
0.2
0.4
0.6
0.8
1.0
Recall@1
1
2
3
4
5
Uncertainty level
0.00
0.05
0.10
0.15
0.20
0.25
Recall@1
1
2
3
4
5
Uncertainty level
0.0
0.2
0.4
0.6
0.8
Recall@1
Text to Image
1
2
3
4
5
Uncertainty level
0.0
0.2
0.4
0.6
0.8
1.0
Recall@1
1
2
3
4
5
Uncertainty level
0.0
0.2
0.4
0.6
0.8
1.0
Recall@1
1
2
3
4
5
Uncertainty level
0.0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
Recall@1
1
2
3
4
5
Uncertainty level
0.0
0.2
0.4
0.6
0.8
1.0
Recall@1
Deterministic
TTDA
PFE
PCME
PCME++
ProbVLM
GroVE
Figure 8: Evaluation of uncertainty calibration for embeddings obtained from BLIP on Image-to-Text (top) and Text-
to-Image (bottom) retrieval tasks. For perfect calibration, the Recall@1 should show a monotonic decrease as uncertainty
levels increase. GroVE exhibits a more consistent relationship between increasing uncertainty and performance degradation
compared to the baseline methods.


**[Table p16.1]**
| 1.0 0.8 Recall@1 0.6 0.4 0.2 0.0 1 2 3 4 Uncertainty level |  |  |  |  | 1.0 0.8 Recall@1 0.6 0.4 0.2 0.0 5 1 2 3 4 Uncertainty level |  |  |  |  | 0.30 0.25 Recall@1 0.20 0.15 0.10 0.05 0.00 5 1 2 3 4 Uncertainty level |  |  |  |  | 1.0 0.8 Recall@1 0.6 0.4 0.2 0.0 5 1 2 3 4 Uncertainty level |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 1.0 0.8 Recall@1 0.6 0.4 0.2 0.0 1 2 3 4 Uncertainty level |  |  |  |  | 1.0 0.8 Recall@1 0.6 0.4 0.2 0.0 5 1 2 3 4 Uncertainty level |  |  |  |  | 0.200 0.175 0.150 Recall@1 0.125 0.100 0.075 0.050 0.025 0.000 5 1 2 3 4 Uncertainty level |  |  |  |  | 0.6 0.5 0.4 Recall@1 0.3 0.2 0.1 0.0 5 1 2 3 4 Uncertainty level |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |


**[Table p16.2]**
| 1.0 0.8 Recall@1 0.6 0.4 0.2 0.0 1 2 3 4 5 Uncertainty level |  |  |  |  | 1.0 0.8 Recall@1 0.6 0.4 0.2 0.0 1 2 3 4 5 Uncertainty level |  |  |  |  | 0.25 0.20 Recall@1 0.15 0.10 0.05 0.00 1 2 3 4 5 Uncertainty level |  |  |  |  | 0.8 Recall@1 0.6 0.4 0.2 0.0 1 2 3 4 Uncertainty level |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |


**[Table p16.3]**
| 1.0 0.8 Recall@1 0.6 0.4 0.2 0.0 1 2 3 4 5 Uncertainty level |  |  |  |  | 1.0 0.8 Recall@1 0.6 0.4 0.2 0.0 1 2 3 4 5 Uncertainty level |  |  |  |  | 0.8 0.7 0.6 Recall@1 0.5 0.4 0.3 0.2 0.1 0.0 1 2 3 4 5 Uncertainty level |  |  |  |  | 1.0 0.8 Recall@1 0.6 0.4 0.2 0.0 1 2 3 4 5 Uncertainty level |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| e 8: E | Deterministic TTDA PFE PCME PCME++ ProbVLM GroVE valuation of uncertainty calibration for embeddings obtained from BLIP on Image-to-Text (to |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | p) and |  |

[CAPTION] Figure 7: Evaluation of uncertainty calibration for embeddings obtained from CLIP on Image-to-Text retrieval. For

[CAPTION] Figure 8: Evaluation of uncertainty calibration for embeddings obtained from BLIP on Image-to-Text (top) and Text-


<!-- page 17 -->
Method
Flickr
COCO
CUB
Flowers
Image to Text
Deterministic
0.801
0.715
0.532
0.357
TTDA
0.423
0.326
0.133
0.289
PFE
0.238
0.213
0.101
0.102
PCME
0.392
0.246
0.129
0.134
PCME++
0.423
0.397
0.124
0.111
ProbVLM
0.491
0.303
0.136
0.245
GroVE
0.569
0.512
0.307
0.402
Text to Image
Deterministic
0.543
0.515
0.141
0.109
TTDA
0.202
0.139
0.046
0.057
PFE
0.298
0.219
0.023
0.024
PCME
0.092
0.102
0.099
0.029
PCME++
0.133
0.125
0.087
0.058
ProbVLM
0.104
0.156
0.005
0.102
GroVE
0.241
0.288
0.343
0.379
Table 7: Retrieval performance using CLIP. Table shows the Recall@1 scores obtained using the different baselines.
GroVE achieves the best scores for the fine-grained datasets.
Figure 9: t-SNE visualization of the probabilistic representations generated by GroVE on a subset of the CUB-200-2011
dataset. Starting from deterministic embeddings provided by frozen VLMs, GroVE produces corresponding probabilistic
representations that capture input ambiguities.

[CAPTION] Table 7: Retrieval performance using CLIP. Table shows the Recall@1 scores obtained using the different baselines.

[CAPTION] Figure 9: t-SNE visualization of the probabilistic representations generated by GroVE on a subset of the CUB-200-2011


<!-- page 18 -->
A bird <unk> <unk> 
<unk> <unk> <unk>.
this is a large bird with a 
white throat and breast 
and the rest of the body 
is grey
a small brown 
and white bird 
with a small 
black sharp bill
Figure 10: Illustration of failure case of GroVE where the model makes incorrect predictions of the CUB-200-2011 dataset
due to the high inter-class similarity.

[CAPTION] Figure 10: Illustration of failure case of GroVE where the model makes incorrect predictions of the CUB-200-2011 dataset


<!-- page 19 -->
Method
Flickr
Flowers
CUB
Image to Text
PFE
0.01±0.03
0.38±0.04
0.02±0.02
PCME
0.04±0.02
0.13±0.04
0.09±0.06
PCME++
0.01±0.02
0.48±0.03
0.03±0.02
ProbVLM
0.55±0.03
0.19±0.04
0.15±0.04
GroVE
0.74±0.03
0.69±0.02
0.41±0.03
Text to Image
PFE
0.41±0.03
0.02±0.03
0.04±0.01
PCME
0.24±0.03
-0.01±0.02
0.02±0.03
PCME++
-0.43±0.03
0.05±0.03
0.03±0.02
ProbVLM
0.14±0.05
0.01±0.03
0.00±0.01
GroVE
0.42±0.02
0.09±0.03
0.04±0.02
Table 8: Zero-shot uncertainty calibration - MS-COCO. GroVE outperforms other baselines in most cases, achieving
superior uncertainty calibration in zero-shot settings. The best scores are highlighted in bold and the second-best scores are
underlined.
Method
Flickr
COCO
Flowers
Image to Text
PFE
0.00±0.04
0.02±0.03
-0.13±0.03
PCME
0.46±0.02
0.01±0.05
0.02±0.04
PCME++
0.40±0.03
0.10±0.02
0.44±0.03
ProbVLM
0.15±0.02
0.38±0.03
0.18±0.03
GroVE
0.59±0.03
0.45±0.03
0.50±0.04
Text to Image
PFE
-0.01±0.02
0.31±0.04
-0.12±0.03
PCME
0.14±0.02
-0.19±0.03
0.15±0.04
PCME++
0.13±0.02
0.52±0.03
0.36±0.03
ProbVLM
0.01±0.03
0.01±0.02
0.02±0.03
GroVE
0.76±0.03
0.42±0.02
0.37±0.03
Table 9: Zero-shot uncertainty calibration - CUB-200-2011. GroVE outperforms other baselines in most cases, achieving
superior uncertainty calibration in zero-shot settings. The best scores are highlighted in bold and the second-best scores are
underlined.
C.2
ZERO-SHOT UNCERTAINTY CALIBRATION
We evaluate the generalization of uncertainty calibration across methods that use auxiliary models for probabilistic
embeddings on out-of-distribution datasets. Two CLIP experiments are conducted by training on MS-COCO and CUB, then
evaluating on their respective unseen datasets. Table 8 and 9 presents the −SR2 scores with models trained on MS-COCO
and CUB respectively. The models trained on MS-COCO show a strong performance on Flickr30k due to its similarity to
MS-COCO, thereby exhibiting better generalization. There is a drop in performance on the more fine-grained Flowers and
CUB datasets, particularly for text-to-image retrieval. GroVE, however, demonstrates better generalization than the baseline
methods for both the experiments.
The calibration results for the experiments are presented in Figure 11 and Figure 12, respectively, where GroVE maintains a
more consistent alignment between decreasing uncertainty and increasing Recall@1.

[CAPTION] Table 8: Zero-shot uncertainty calibration - MS-COCO. GroVE outperforms other baselines in most cases, achieving

[CAPTION] Table 9: Zero-shot uncertainty calibration - CUB-200-2011. GroVE outperforms other baselines in most cases, achieving


<!-- page 20 -->
Flickr
Flowers
CUB
Image to Text
1
2
3
4
5
Uncertainty level
0.0
0.2
0.4
0.6
0.8
1.0
Recall@1
1
2
3
4
5
Uncertainty level
0.0
0.2
0.4
0.6
0.8
1.0
Recall@1
1
2
3
4
5
Uncertainty level
0.00
0.05
0.10
0.15
0.20
0.25
0.30
0.35
0.40
Recall@1
Text to Image
1
2
3
4
5
Uncertainty level
0.0
0.2
0.4
0.6
0.8
1.0
Recall@1
1
2
3
4
5
Uncertainty level
0.0
0.2
0.4
0.6
0.8
1.0
Recall@1
1
2
3
4
5
Uncertainty level
0.00
0.01
0.02
0.03
0.04
0.05
0.06
0.07
0.08
Recall@1
PFE
PCME
PCME++
ProbVLM
GroVE
Figure 11: Evaluation of zero-shot uncertainty calibration using MS-COCO for embeddings obtained from CLIP
on Image-to-Text (top) and Text-to-Image (bottom) retrieval tasks. For perfect calibration, the Recall@1 should show
a monotonic decrease as uncertainty levels increase. GroVE exhibits a more consistent relationship between increasing
uncertainty and performance degradation compared to the baseline methods.


**[Table p20.1]**
| 1.0 0.8 Recall@1 0.6 0.4 0.2 0.0 1 2 3 4 5 Uncertainty level | 1.0 0.8 Recall@1 0.6 0.4 0.2 0.0 1 2 3 4 5 Uncertainty level | 0.40 0.35 0.30 Recall@1 0.25 0.20 0.15 0.10 0.05 0.00 1 2 3 4 5 Uncertainty level |
| --- | --- | --- |


**[Table p20.2]**
| 1.0 0.8 Recall@1 0.6 0.4 0.2 0.0 1 2 3 4 5 Uncertainty level | 1.0 0.8 Recall@1 0.6 0.4 0.2 0.0 1 2 3 4 5 Uncertainty level | 0.08 0.07 0.06 Recall@1 0.05 0.04 0.03 0.02 0.01 0.00 1 2 3 4 5 Uncertainty level |
| --- | --- | --- |

[CAPTION] Figure 11: Evaluation of zero-shot uncertainty calibration using MS-COCO for embeddings obtained from CLIP


<!-- page 21 -->
Flickr
MS-COCO
Flowers
Image to Text
1
2
3
4
5
Uncertainty level
0.0
0.2
0.4
0.6
0.8
1.0
Recall@1
1
2
3
4
5
Uncertainty level
0.0
0.2
0.4
0.6
0.8
1.0
Recall@1
1
2
3
4
5
Uncertainty level
0.0
0.2
0.4
0.6
0.8
1.0
Recall@1
Text to Image
1
2
3
4
5
Uncertainty level
0.0
0.2
0.4
0.6
0.8
1.0
Recall@1
1
2
3
4
5
Uncertainty level
0.0
0.2
0.4
0.6
0.8
1.0
Recall@1
1
2
3
4
5
Uncertainty level
0.00
0.05
0.10
0.15
0.20
0.25
0.30
0.35
Recall@1
PFE
PCME
PCME++
ProbVLM
GroVE
Figure 12: Evaluation of zero-shot uncertainty calibration using CUB-200-2011 for embeddings obtained from CLIP
on Image-to-Text (top) and Text-to-Image (bottom) retrieval tasks. For perfect calibration, the Recall@1 should show
a monotonic decrease as uncertainty levels increase. GroVE exhibits a more consistent relationship between increasing
uncertainty and performance degradation compared to the baseline methods.


**[Table p21.1]**
| 1.0 0.8 Recall@1 0.6 0.4 0.2 0.0 1 2 3 4 5 Uncertainty level | 1.0 0.8 Recall@1 0.6 0.4 0.2 0.0 1 2 3 4 5 Uncertainty level | 1.0 0.8 Recall@1 0.6 0.4 0.2 0.0 1 2 3 4 5 Uncertainty level |
| --- | --- | --- |


**[Table p21.2]**
| 1.0 0.8 Recall@1 0.6 0.4 0.2 0.0 1 2 3 4 5 Uncertainty level | 1.0 0.8 Recall@1 0.6 0.4 0.2 0.0 1 2 3 4 5 Uncertainty level | 0.35 0.30 Recall@1 0.25 0.20 0.15 0.10 0.05 0.00 1 2 3 4 5 Uncertainty level |
| --- | --- | --- |


**[Table p21.3]**
|  |  |  |  |
| --- | --- | --- | --- |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |

[CAPTION] Figure 12: Evaluation of zero-shot uncertainty calibration using CUB-200-2011 for embeddings obtained from CLIP


<!-- page 22 -->
C.3
FEW-SHOT UNCERTAINTY CALIBRATION
Table 10 shows the Recall@1 scores for the cross-modal retrieval task for the auxiliary models trained using limited data
from the synthetic CUB dataset. The performance of the neural network based methods drop, which is expected given the
insufficient number of data points for the model to generalize. Note that deterministic and TTDA are agnostic to the few shot
setting since they work directly on the VLM embeddings for the prediction. Among the methods using auxiliary models,
GroVE achieves a higher score, leveraging the ability of GPs to generalize well even with limited data because of their
distance awareness property by capturing structure through kernel functions. Moreover, as the number of inducing points
increases, GroVE’s performance improves, with the best results achieved when performing exact GP. However, GroVE is
computationally expensive compared to the neural network based approaches, with longer inference time as the number of
inducing points increases.
Method
Image to Text
Text to Image
Time (ms/example) (↓)
Determinsitic
0.532
0.141
29.98
TTDA (10 passes)
0.133±0.003
0.046±0.011
288.51
PFE
0.062±0.001
0.026±0.010
31.59
PCME
0.074±0.002
0.031±0.005
31.60
PCME++
0.063±0.003
0.031±0.003
31.55
ProbVLM
0.081±0.001
0.034±0.005
32.80
GroVE (M=50)
0.062±0.002
0.035±0.009
47.62
GroVE (M=150)
0.084±0.004
0.049±0.004
142.85
GroVE (M=250)
0.086±0.003
0.056±0.004
392.16
GroVE (exact GP)
0.103±0.002
0.182±0.002
1130.09
Table 10: Retrieval performance using Recall@1 scores and inference speed per instance for few-shot experiment using
CUB-200-2011. The best results are highlighted in bold and the second best are underlined.

[CAPTION] Table 10 shows the Recall@1 scores for the cross-modal retrieval task for the auxiliary models trained using limited data

[CAPTION] Table 10: Retrieval performance using Recall@1 scores and inference speed per instance for few-shot experiment using