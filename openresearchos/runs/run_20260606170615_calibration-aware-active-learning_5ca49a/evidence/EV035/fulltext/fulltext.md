<!-- page 1 -->
arXiv:2310.15952v5  [cs.LG]  29 Jun 2025
IEEE TRANSACTIONS ON MEDICAL IMAGING, VOL. XX, NO. XX, XXXX 2020
1
Improving Robustness and Reliability in Medical
Image Classification with Latent-Guided
Diffusion and Nested-Ensembles
Xing Shen, Hengguan Huang, Brennan Nichyporuk, and Tal Arbel
Abstract— Once deployed, medical image analysis meth-
ods are often faced with unexpected image corruptions
and noise perturbations. These unknown covariate shifts
present significant challenges to deep learning based
methods trained on “clean” images. This often results in
unreliable predictions and poorly calibrated confidence,
hence hindering clinical applicability. While recent methods
have been developed to address specific issues such as
confidence calibration or adversarial robustness, no single
framework effectively tackles all these challenges simulta-
neously. To bridge this gap, we propose LaDiNE, a novel
ensemble learning method combining the robustness of
Vision Transformers with diffusion-based generative mod-
els for improved reliability in medical image classification.
Specifically, transformer encoder blocks are used as hierar-
chical feature extractors that learn invariant features from
images for each ensemble member, resulting in features
that are robust to input perturbations. In addition, diffusion
models are used as flexible density estimators to estimate
member densities conditioned on the invariant features,
leading to improved modeling of complex data distributions
while retaining properly calibrated confidence. Extensive
experiments on tuberculosis chest X-rays and melanoma
skin cancer datasets demonstrate that LaDiNE achieves
superior performance compared to a wide range of state-
© 2025 IEEE. Personal use of this material is permitted. Permission
from IEEE must be obtained for all other uses, in any current or future
media, including reprinting/republishing this material for advertising or
promotional purposes, creating new collective works, for resale or
redistribution to servers or lists, or reuse of any copyrighted component
of this work in other works.
This work was supported in part by the Natural Sciences and Engi-
neering Research Council of Canada, in part by the Canadian Institute
for Advanced Research (CIFAR) Artificial Intelligence Chairs Program,
in part by the Mila—Quebec Artificial Intelligence Institute Technology
Transfer Program, in part by the Mila—Google Research Grant, in part
by Calcul Quebec, in part by the Digital Research Alliance of Canada,
and in part by the Canada First Research Excellence Fund, awarded to
the Healthy Brains, Healthy Lives initiative at McGill University.
Xing Shen is at the Centre for Intelligent Machines, McGill University,
Montreal, QC H3A 0G4 Canada, and a student at Mila – Quebec AI Insti-
tute, Montreal, QC H2S 3H1 Canada (e-mail:xing.shen@mail.mcgill.ca).
Hengguan Huang is an Assistant Professor at Department of Pub-
lic Health, Section for Health Data Science and AI, University of
Copenhagen, 1172 København, Denmark. Corresponding author (e-
mail:hengguan.huang@sund.ku.dk).
Brennan Nichyporuk is a Research Scientist at Mila – Quebec
AI Institute, Montreal, QC H2S 3H1 Canada. He is also an affiliate
member of McGill University, Montreal, QC H3A 0G4 Canada (e-
mail:nichypob@mila.quebec).
Tal Arbel is a Professor at McGill University, and a member of
the Centre for Intelligent Machines, McGill University, Montreal, QC
H3A 0G4 Canada. She is also a Fellow of the Canadian Academy of
Engineering, a CIFAR AI Chair, and Core Member of Mila – Quebec AI
Institute, Montreal, QC H2S 3H1 Canada. Corresponding author (e-mail:
tal.arbel@mcgill.ca).
of-the-art methods by simultaneously improving prediction
accuracy and confidence calibration under unseen noise,
adversarial perturbations, and resolution degradation.
Index Terms— Medical Image Classification, Uncertainty
Quantification, Diffusion-based Generative Models, Ensem-
ble Methods.
I. INTRODUCTION
I
N the rapidly evolving domain of medical imaging analysis,
deep learning has led to enormous advances in many
clinical domains of interest [1]–[9], notably in tasks such as
detection of diabetic retinopathy in eye fundus images [2],
classification of skin cancer [1], and identification of cancerous
regions in mammograms [7]. Despite the fact that recent
methods have achieved unprecedented success in controlled
experimental settings, their fundamental building blocks –
deep neural networks (DNNs) – are known to be sensitive to
slight distribution changes or covariate shifts, and vulnerable
to attacks [10]–[12]. As a result, their application to real-world
clinical contexts often results in significant performance degra-
dation, including inaccurate predictions and poorly calibrated
confidence estimates.
Data augmentation is a widely used tool for improving
the generalization and robustness of DNNs. In the field of
medical imaging, however, where datasets are typically smaller
than natural image datasets especially for rare diseases [13],
conventional data augmentation strategies may not always be
suitable and can even lead to degradation in the performance of
DNNs [14]. Designing suitable augmentation strategies from
small medical imaging datasets can be challenging, requir-
ing careful design choices in order to incorporate suitable
inductive biases that lead to robust and generalizable DNN
architectures and learning algorithms.
Real-world medical imaging applications, such as medical
image classification, often face unpredictable and complex
covariate shifts that are difficult to anticipate during train-
ing [15], leading to significant challenges in maintaining
reliable performance of DNN-based methods. In the field of
machine learning, ensemble methods are popular choices to
enhance robustness and generalization, as they combine the
predictions of multiple models, effectively reducing variance
and mitigating overfitting [16]. By leveraging the strengths
of diverse models, ensemble techniques such as bagging,
boosting, and stacking can achieve better and more robust
predictive performance over a single model [17]. In addition,


<!-- page 2 -->
2
IEEE TRANSACTIONS ON MEDICAL IMAGING, VOL. XX, NO. XX, XXXX 2020
deep ensemble methods have been shown to improve both
predictive performance and the quality of uncertainty estimates
by training multiple DNNs independently and averaging their
predictions [18]. Other frameworks involve developing algo-
rithms to select better ensemble members and distribute input-
dependent weights to each member, aiming to reduce the effect
of weak members and providing performance gains [19], [20].
Despite the potential robustness offered by ensemble methods,
their application in medical image classification to address
covariate shifts and confidence calibration remains limited.
While in natural image classification, existing ensemble meth-
ods often rely on simplified component distribution assump-
tions, such as Gaussian distributions or deterministic mappings
to the parameters of a categorical distribution [18], which may
not adequately capture the complexity and heteroscedasticity
of real-world medical imaging data.
This paper introduces LaDiNE, Latent-guided Diffusion
Nested-Ensembles, a robust ensemble learning model designed
for mitigating the aforementioned challenges in medical image
classification. LaDiNE is a parametric mixture model that
incorporates transformers and diffusion models, leveraging
their recent success in medical imaging contexts [21], [22]. It
encodes invariant and informative features as latent variables,
and performs functional-form-free inference to estimate the
predictive distribution. Specifically, a subset of transformer
encoder blocks of a Vision Transformer (ViT) is used as a
hierarchical feature extractor that learn invariant features from
images for each mixture component. The diffusion models are
used as flexible density estimators to estimate the component
densities conditioned on the invariant features. In our formu-
lation, each mixture component is interpreted equivalently as
a Bayesian network that encodes the dynamics of observed
(e.g., images) and latent variables. LaDiNE is specifically
designed to be: (i) Robust to covariate shift; (ii) Provide
calibrated confidence estimates; (iii) Be resilient to gradient-
based adversarial attacks.
Extensive experiments are performed on the Tuberculosis
chest X-ray classification benchmark [23] and the ISIC skin
cancer classification benchmark [24]. We split the original
datasets into training, validation, and testing sets. Models are
trained on the original training set and then evaluated on a
perturbed version of the test set, one with complex simu-
lated unseen covariate shifts. Results indicate that LaDiNE
performs substantially better than popular baselines in terms
of prediction accuracy and prediction confidence calibration,
under a variety of challenging covariate shift conditions. Our
contribution is threefold:
1. This paper introduces LaDiNE, a novel ensemble learning
method that encodes invariant features and estimates the
predictive component distribution without specific assump-
tions regarding its component functional form using, for
the first time, diffusion models. This additional component
enables flexible and expressive modeling of complex dis-
tributions, especially important in the with limited medical
data.
2. Extensive
evaluations
demonstrate
that
the
proposed
method outperforms a wide variety of state-of-the-art
(SOTA) methods in terms of classification accuracy under
diverse covariate shift scenarios, including (i) images with
Gaussian noise injection, (ii) images with lower resolution,
(iii) images with lower color contrast, and (iv) images
with adversarial perturbation. Empirical evidence shows
significant improvements in classification accuracy over
baseline methods. Detailed ablation studies further justify
the design choices made in the paper.
3. Extensive experimentation indicates that the proposed
method provides better calibrated predictions than com-
peting methods. Instance-level prediction uncertainties are
evaluated under severe perturbations of the input images
and are shown to be correct when more certain, as desired.
II. RELATED WORK
This section summarizes existing work on robustness learn-
ing in medical imaging, focusing on methods based on trans-
formers and diffusion models.
Transformers in Medical Imaging. While transformer-
based models have been less extensively studied in medical
image analysis compared to their widespread application in
computer vision, they hold significant potential for capturing
complex spatial relationships and the variability inherent in
clinical data [21]. Some recent works have shown how in-
corporating transformers helps to improve prediction accuracy
in various medical imaging tasks. Peiris et al. [25] proposes
a transformer-based architecture that can encode local and
global spatial cues for 3D tumor segmentation and exhibits
robust performance against the presence of image artifacts.
Chen et al. [26] integrates multi-scale feature extraction into
the transformer to achieve improved performance in image-
based gastric cancer detection, with the ability to be robust
to noise. Wang et al. [27] uses transformers to capture X-ray
sinograms’ global characteristics and achieve enhanced per-
formance against artifacts in sparse-view CT reconstruction.
Almalik et al. [28] extends this direction by leveraging inter-
mediate representations from ViT blocks to improve robust-
ness against adversarial attacks. Although these transformer-
based methods demonstrate some degree of robustness to
specific forms of noise and artifacts, they do not explicitly
address confidence calibration and uncertainty quantification
under noisy conditions or distribution shifts in input images.
As a result, their ability to assist in informed clinical decision-
making under unknown covariate shifts remains limited.
Diffusion Models in Medical Imaging. Diffusion models
have recently emerged as powerful generative models for
medical imaging applications due to their ability to generate
high-quality images, while remaining robust to distribution
shifts [22], [29]. While most of the current work is focused
on medical image generation and reconstruction with diffusion
models [30]–[34], some methods have been developed for
medical image segmentation via generating the segmentation
mask [35], [36]. In other work, Li et al. [37] uses frequency-
domain filters to guide the diffusion model for structure-
preserving image translation, achieving robust generalization
capability. Kim et al. [38] incorporates diffusion models into
a representation learning framework for vessel segmentation
and shows superior results on noisy data. In the field of


<!-- page 3 -->
SHEN et al.: IMPROVING ROBUSTNESS AND RELIABILITY IN MEDICAL IMAGE CLASSIFICATION WITH LATENT-GUIDED DIFFUSION AND NESTED-
ENSEMBLES
3
machine learning, recent work by Clark and Jaini demonstrates
that a text-to-image diffusion model can function as a zero-
shot classifier [39]. Han et al. [40] further introduces a
diffusion-based conditional generative framework tailored for
classification and regression on natural images and tabular
data, showing strong performance in uncertainty estimation.
Despite their robustness and ability to provide instance-level
predictive uncertainty [40], [41], which can aid in informed
clinical decision-making, their application in medical image
classification has yet to be explored.
III. THE PROPOSED METHOD: LADINE
This work focuses on establishing a robust and generalizable
model for medical image classification, where a model trained
on “clean” images would be required to be robust to substan-
tial, unseen covariate shifts. The proposed framework consists
of several important components: (i) transformer encoders
(TEs) derived from one Vision Transformer (ViT) [42], (ii)
conditional diffusion models (CDM), and (iii) feed-forward
networks (FFNs). In this section, we first give a high-level
overview of the proposed method. Then we describe the
notation of variables and the computational paths involved
in each neural network. Finally, we introduce the proposed
probabilistic model in Sec. III-A and the training procedure
for those neural networks in Sec. III-B.
Overview. LaDiNE is an ensemble deep learning model
specifically designed to be robust to covariate shifts, while
providing high-quality prediction confidence. To this end,
LaDiNE (i) leverages early transformer encoders from only
one ViT and mapping networks in order to learn image
representations that are robust across different environments
and (ii) estimates component distributions through conditional
diffusion models, free from fixed distributional assumptions
(see Fig. 2 (a) for an illustration of the proposed method).
Notations. An image input is denoted x ∈RH×W ×C,
where (H × W) is the resolution of the image and C is the
number of channels. Its corresponding label is denoted as y
(class index) and y (one-hot encoded vector). We assume that
the observational data pair ⟨x, y⟩is sampled from the joint
distribution ptrain(x, y), which serves as the training data for
the model. In a covariate shift setting, the test data points
⟨x′, y′⟩are sampled from a different distribution, denoted
as ptest(x′, y′). The covariate shift assumption implies that
the conditional distribution of labels given the input remains
unchanged, i.e., ptrain(y|x) = ptest(y′|x′), but the marginal
distribution of the inputs differs, i.e., ptrain(x) ̸= ptest(x′). As a
result, we have a divergence measure D(ptrain(x), ptest(x′)) ̸=
0.
Vision Transformer and Transformer Encoders.
We
follow the architecture described in [42], and introduce some
additional notation used here. The Vision Transformer (ViT) is
the stack of L transformer encoder (TE) blocks, where each TE
block acts as a deterministic mapping from the input sequence
to the output sequence. We define TE : Rn×d →Rn×d to
represent this mapping. When computing TE, the input image
x ∈RH×W ×C is first divided into patches. Those patches
are then flattened and projected into a lower-dimensional
1 2 3 4 5 6 7 8 9 10 11 12
Block k
0
5
10
15
20
25
30
35
L2 Norm
(a)
1 2 3 4 5 6 7 8 9 10 11 12
Block k
0
10
20
30
40
L2 Norm
(b)
1 2 3 4 5 6 7 8 9 10 11 12
Block k
0
10
20
30
40
50
L2 Norm
(c)
1 2 3 4 5 6 7 8 9 10 11 12
Block k
0
20
40
60
80
100
120
L2 Norm
(d)
Fig. 1.
The Euclidean distance between the token sequence of image
variants (under a range of conditions) and its original copy increases as
going deeper into the encoder block hierarchy under (a) noisy, (b) lower-
resolution, (c) lower-contrast contexts, and (d) adversarial attack.
embedding space. This process, including the addition of
positional encoding, can be simplified into a single embedding
function e0 = Emb(x). Before feeding into the first TE block,
a class token (with its own positional encoding) is appended
to the embedding e0. For simplifying the whole process, we
denote another function Te(·) as:
Te(ek) := TE({class token, ek}),
(1)
here ek is the embedding output (excluding the class token)
generated by the (k−1)-th TE block. The class token generated
by the last TE block is used for classification with a feed-
forward network (FFN).
In Almalik et al. and Walmer et al., the authors presented
an investigation of the invariance to adversarial noise [28]
and informative hidden layers in ViTs [43]. Extending these
findings, we investigate whether the representations extracted
from the early TE blocks in ViTs remain robust under a range
of image perturbations, beyond adversarial noise considered
in prior work [28]. To this end, a number of experiments are
performed. Fig. 1 depicts the results where the L2 norms of
the differences between clean input and noisy input represen-
tations are shown, under four types of covariate shifts. The
results indicate a clear pattern where, under all conditions,
early TE blocks learn more invariant features than the deeper
blocks. This finding motivates the use of early TE blocks in
the predictions in order to improve robustness performance.
Specifically, here TEk is included where k = 1, 2, . . . , K and
K < L.
Mapping to Latents. In addition to using FFNs within the
ViT, FFNs are used to map the embedding e (e.g. e1 defined
previously) to a latent z for subsequent computations. As such,
it is named as a mapping network. In order not to confuse this
mapping with the FFNs used in the ViT, it is denoted as a

[CAPTION] Fig. 1.
The Euclidean distance between the token sequence of image


<!-- page 4 -->
4
IEEE TRANSACTIONS ON MEDICAL IMAGING, VOL. XX, NO. XX, XXXX 2020
function g : RN×d →Rdlatent as follows:
z = g(e) = FFN(e).
(2)
Estimating Distribution with Diffusion Models. In [40],
the diffusion model (DM) models a conditional distribution
free of predefined functional-forms with a single covariate.
Here a conditional DM (CDM) is defined with several co-
variates: the latent z and the image input x, together with the
response y. This enables sampling from the probability density
function p(y|z, x).
A. Probabilistic Predictive Model
The proposed method is visually depicted in Fig. 2 (b),
which provides a graphical representation of the dependencies
among variables. This representation enables the factorization
of the predictive distribution into conditional components. In
this section, we describe the parametrization of each factorized
conditional distribution.
Predictive Distribution. The proposed predictive model is
then defined as a mixture model composed of K components,
or ensemble members:
p(y|x, Θ) =
K
X
k=1
πk
Z
· · ·
Z
p(y, zk, e1:k|x) dzkde1:k
|
{z
}
pk(y|x)
(3)
where Θ denotes all parameters in the mixture, πk is the mix-
ture weight for the k-th component distribution pk(y|x) with
PK
k=1 πk = 1 and πk ≥0. The k-th predictive component can
be further factorized as:
pk(y|x) =
Z
· · ·
Z
p(y|zk, x)p(zk|ek)
k
Y
i=2
p(ei|ei−1)p(e1|x) dzkde1:k.
(4)
This factorization allows us to compute the predictive density
by breaking it down into a series of conditional distributions.
The next step is to parameterize these conditional distributions
so that the model can learn from the data.
Parameterization. When there is no particular prior belief
in the contribution of each mixture component, each compo-
nent is equally weighted, such that πk = K−1. With deter-
ministic functions Te(·) and g(·), the component distribution
becomes:
pk(y|x) =
Z
· · ·
Z
pθk(y|zk, x)δ(zk −gϕk(ek))
k
Y
i=2
δ(ei −Tei(ei−1))δ(e1 −Te1(Emb(x))) dzkde1:k, (5)
where δ(·) is the Dirac Delta function, and Emb(x) is the
embedding step to produce e0. Here, the conditional distribu-
tion pθk(y|zk, x) modeled by the DM is parameterized by θk,
and the mapping network gϕk(·) is parameterized by ϕk. The
notation of the parameters in Emb(·) and Te(·) are omitted
here as they are packed in the TE blocks’ parameters, which
are estimated along the training of the ViT. After simplification
of the δ distribution, the model becomes:
pk(y|x) = pθk(y|zk = gϕk(ek = Te1:k(Emb(x))), x),
(6)
where Te1:k(·) denotes the composite function (Tek ◦Tek−1 ◦
· · · ◦Te1)(·).
pθk(y|zk, x) is modeled with a CDM based on an exten-
sion of the original denoising diffusion probabilistic model
(DDPM) [44] that includes additional covariates. For simplic-
ity, the subscript k in θk and zk is omitted, and a probability
density function pθ(y|z, x) is assumed. Here, we consider a
diffusion process that is fixed to a Markov chain with T states,
the joint probability given the covariates z, x and the response
y is as follows 1:
q(y1:T | y0, z, x) =
T
Y
t=1
q(yt | yt−1, z, x).
(7)
The parametric form of the forward transition density func-
tion is represented as a Gaussian density function with a static
variance schedule {β1, β2, . . . , βT }. αt := 1 −βt such that:
q(yt|yt−1, z, x) := N(yt; √αtyt−1
+ (1 −√αt) (z + Enc(x)), βtI).
(8)
The encoder Enc(·) maps the image x to an embedding with
the same dimensionality as z. Later sections will provide more
details for the CDM, specifically pertaining to inference and
sampling.
B. Training
This section describes the process of estimating the param-
eters of the predictive density p(y|x, Θ) during training. Note
that the mixture weight π does not need to be estimated as
it is set to be uniform. In order to estimate the parameters of
the TE blocks, θk, ϕk, a 3-step training procedure is followed
as illustrated in Fig. 3. Steps 2 and 3 are performed for
k = 1, 2, . . . , K, so as to train K ensemble members.
Step 1 Training TE Blocks. The parameters of the TE
blocks are learned during the training of the ViT. As discussed
previously, the input x is initially transformed into an em-
bedding and subsequently processed through L Transformer
Encoder (TE) blocks. The class token generated by the final
TE block is used for classifying x. For simpler notation, we
denote a function Vit : X →Y to summarize all computations
involved in the ViT including the last softmax function, where
x ∈X and y ∈Y . The parameters of the ViT are estimated
using maximum likelihood estimation (MLE), or equivalently,
through minimizing the cross entropy (CEloss) between the
prediction and the ground truth:
LViT(ϑ) :=
E
⟨x,y⟩[CEloss(y, Vitϑ(x))] .
(9)
Step 2 Training Mapping Network. Next, the parameters
ϕk of the mapping network g(·) are estimated. In the proposed
model, the latent variable z serves as a conditioning signal
1Here we denote yt=a as ya for simplicity, and y0 is equivalent to the
response y.


<!-- page 5 -->
SHEN et al.: IMPROVING ROBUSTNESS AND RELIABILITY IN MEDICAL IMAGE CLASSIFICATION WITH LATENT-GUIDED DIFFUSION AND NESTED-
ENSEMBLES
5
𝑧!
Te"
Te#
Te!
…
Extract Invariant 
Features
Early 𝑘 Layers of the ViT 
x
𝑔<!
Infer
 Latent 
Variable
Embedding
𝜖A!
Conditional Diffusion 
Model (CDM)
The 𝑘−th Component 
Distribution p! y|x
Draw 𝑀 
Samples
y!,#$%,… ,y!,#$&
Iterate 
Through 
𝑘= 1, … ,𝐾
𝐾×𝑀 
Samples
y!$%,#$%,… , y!$',#$&
Average
Map to 
Probabilities 
Pr 𝑦= 𝑎|x
Final Prediction
𝑒!
𝑒"
𝑧"
x
𝑒U
𝑧U
𝑒#
𝑧#
…
…
y
y
y
x
x
…
 a  
 b  
Phase 2
Phase 3
Mapping Network 
Phase 1
Fig. 2.
An illustration of the proposed model from two perspectives: (a) The flowchart shows the workflow of the proposed model in three phases.
In Phase 1, the transformer encoders and the mapping network gϕk compute the latent variable z from the image x. In Phase 2, a conditional
diffusion model estimates the predictive component distribution. M samples are drawn from this distribution. In Phase 3, M samples are extracted
from each of the K ensemble members. These samples are aggregated to form the final prediction. (b) This directed acyclic graph shows the
dependency of each variable within the (unrolled) probabilistic model. Here x (observed in grey) and y are the input image and its predicted label,
respectively. ek denote the image embedding and zk denotes the latent variable in the k-th ensemble member. The mixture weight variables are
omitted as the components are equally weighted. In this graph, every directed edge shows dependencies. For example, x →e1 means that e1
depends on x, and the local Markovian property holds.
!
Te!
Te"
Te#
…
$ Transformer 
Encoder Blocks 5
Vision Transformer (ViT)
Embedding
x
Projection
?@
?
ℒViT 5
Obtain the 
Latent 
with Trained 
Te!:#      /$!
x
Embedding
"3
❄
#
One-hot
$
Add Noise
%9!
Noise 
Estimator 
%
"
Conditional 
Diffusion Model (CDM)
Train with
 ℒCDM C#
𝐾< 𝐿    Iterate through 𝑘= 1,2,… ,𝐾
Step 2 
Step 3 
Step 1 
🔥: Parameters are active for optimization
❄: Parameters are frozen
x
!
Te!
Te"
Te#
…
Extract Invariant 
Features
Early 5 Layers of the ViT 
Embedding
❄
$#
%@!
Infer 
Invariant 
and 
Informative 
Latent
&#
Softmax
'(
'
ℒg C!
Mapping
Network
Fig. 3.
An illustration of the 3-step training procedure. Note that the training data point ⟨x, y⟩is sampled from ptrain. In Step 1, the Vision
Transformer (ViT) is trained to estimate its parameters ϑ in an end-to-end fashion using a cross-entropy loss LViT(ϑ) (see (9)). In Step 2, the ViT
is frozen and the embedding ek is produced from the k-th transformer encoder block (in orange). The extracted embedding ek is passed through
the mapping network (in red, with parameters ϕk) to produce the latent zk. The mapping network is trained by minimizing a cross-entropy loss
Lg(ϕk) (see (10)) with the ground-truth label y and the softmax-ed zk. In Step 3, all transformer encoder blocks and the mapping network are
frozen. The diffusion model is trained with parameters θk conditioned on x and zk to predict the noise term, and thus predict the denoised y. This
diffusion model is trained with the simplified objective LCDM(θk) (equation (15)). The system iterates k from 1 to K in Step 2 and Step 3, resulting
in a total of K ensemble members.
in the diffusion process. z is required to (i) provide relevant
information about the ground truth label to facilitate the
estimation of the predictive distribution (which is estimated by
the CDM), while (ii) be robust to covariate shifts. Recall that
the mediator e ∈{e1:K} encodes the image into an embedding
space that is insensitive to image distribution shift (under the
constraint shown in Fig. 1). Thus, we identify the latent z as
a non-linear transformation of the mediator e (realised by a
FFN g(·), see (2)). Specifically, it represents the unnormalized
probability (logit) when optimizing the cross-entropy loss with
respect to the parameter ϕk:
Lg(ϕk) := E
⟨e,y⟩[CEloss(y, softmax(zk = gϕk(ek)))] .
(10)
Step 3 Training Conditional Diffusion Model. In the final
step, the parameter θk of the noise estimator is learned in the
conditional diffusion model. For simpler subscript notation, k
is omitted in θk and zk here. The parameter θ parameterizes the
probability density function pθ(y|z, x), which is estimated via
minimizing the variational bound on the negative logarithmic
likelihood (VBNL) of the conditional distribution pθ(y|z, x):


**[Table p5.1]**
| The 𝑘−th Co |  | Dra Sam mponent |
| --- | --- | --- |


**[Table p5.2]**
| Mapping Early 5 Layers of the ViT Network ❄ ! Extract Invariant Embedding Features $ # Infer … Te Te Te Invariant ! " # x and Informative Latent % @! Softmax ' ℒg C! '( & # |  | ❄ Obtain the Embedding Latent with Trained " 3 Te!:# /$! x Train with " ℒCDM C# Noise One-hot Add # $ Estimator % Noise % 9! Conditional Diffusion Model (CDM) |
| --- | --- | --- |

[CAPTION] Fig. 2.
An illustration of the proposed model from two perspectives: (a) The flowchart shows the workflow of the proposed model in three phases.

[CAPTION] Fig. 3.
An illustration of the 3-step training procedure. Note that the training data point ⟨x, y⟩is sampled from ptrain. In Step 1, the Vision


<!-- page 6 -->
6
IEEE TRANSACTIONS ON MEDICAL IMAGING, VOL. XX, NO. XX, XXXX 2020
LVBNL(θ) := E[−log pθ(y|z, x)] ≤E
q
 pθ(y0:T |z, x)
q(y1:T |y0, z, x)

= E
q
"
L0 +
T
X
t=2
Lt−1(t) + LT
#
,
(11)
where we have:
L0 := −log pθ(y0|y1, z, x))
|
{z
}
reconstruction term
,
(12)
LT := DKL
q(yT |y0, z, x)
 p(yT |z, x)

|
{z
}
prior matching term
,
(13)
Lt−1(t) := DKL
q(yt−1|yt, y0, z, x)
 pθ(yt−1|yt, z, x)

|
{z
}
consistency term
.
(14)
Similar to the evidence lower bound in DDPM, this VBNL
bound can be interpreted as three terms: (i) the reconstruction
term, (ii) the prior matching term, and (iii) the consistency
term. Among these, the prior matching term (LT ) does not
depend on any parameter and thus can be omitted during
optimization.
In practice, a simplified variant of the VBNL is used for
optimization (akin to the simplified objective in DDPM). Here,
the noise term ϵ ∼N(0, I) is estimated to define yt from y0
for t ∼Uniform(1, 2, . . . , T). Note that here we again omit
the subscript k in θk and zk for simplicity:
LCDM(θ) :=
E
⟨x,y0⟩,ϵ,t

∥ϵ −ϵθ(yt, z, x, t)∥2
2

.
(15)
yt is computed by applying the reparameterization trick to the
forward transition density function conditioned on y0, that is,
q(yt|y0, z, x). Its functional form can be computed from the
reparameterized q(yt|yt−1, z, x) in a recursive fashion. The
derived yt is:
yt = √¯αty0 + (1 −√¯αt)(z + Enc(x)) +
√
1 −¯αtϵ,
(16)
where ¯αt := Qt
i=1 αi.
C. Test-time Predictions
In this subsection, the probabilistic model proposed in
Sec. III-A is used to predict the class given an image x
sampled from ptest(x). Once trained, a 3-phase procedure is
proposed to obtain the final prediction as illustrated in Fig. 2
(a).
Formally, predicting the response given the proposed mix-
ture model p(y|x, Θ) is defined as follows: Given an image
input x, the class label is predicted by computing the condi-
tional expectation E[y|x]. Note that each component density
pk(y|x) does not have a trivial closed form. However, the
reverse diffusion process allows us to sample from it. In
practice, the expectation is computed by using the Monte Carlo
(MC) method due to the intractability of
R
yp(y|x, Θ) dy:
E[y|x] =
Z
yp(y|x, Θ) dy
(17)
=
Z
yK−1
K
X
k=1
pk(y|x) dy
(18)
≈(MK)−1
M
X
m=1
K
X
k=1
yk,m,
yk,m ∼pk(y|x), (19)
where M ∈Z+ should be as large as possible to achieve an
accurate estimation.
To estimate the expected response (or class), one needs
to sample ym,k from the predictive component distribution
pk(y|x). As indicated in (6), the component distribution is
equivalent to the conditional distribution estimated by the
CDM. To this end, a 3-phase procedure is proposed in order to
sample from the CDM, and to estimate the expected response.
Phase 1. The value of the latent variable zk given the input
x is computed through the trained transformer encoders and
the mapping network. The computed value will be used as an
informative and invariant conditioning signal for the CDM.
Phase 2. Once zk is computed, M samples are drawn
from the CDM’s probability density function, pθk(y|zk, x).
Each sample is seen as a candidate for the final prediction. In
this work, drawing a sample from the CDM is performed by
first drawing a noisy sample from the Gaussian prior and then
gradually denoising it through the reverse diffusion process to
obtain a clean sample. (For ease of reading, the subscripts for
θk and zk, and in ym,k are omitted.) If θ is properly modeled,
the consistency term Lt−1(t) is minimized. In the consistency
term, the posterior of the forward transition density function
is derived as:
q(yt−1|yt, y0, z, x) = N(yt−1; µq(yt, y0, z, x), Σq(t)),
(20)
where its parameters are:
µq(yt, y0, z, x) =
√αt(1 −¯αt−1)
1 −¯αt
yt + βt
√¯αt−1
1 −¯αt
y0
−
√αt(1 −¯αt−1) + βt
√¯αt−1
1 −¯αt
−1

(z + Enc(x)), (21)
and
Σq(t) = σ2
q(t)I = βt(1 −¯αt−1)
1 −¯αt
I.
(22)
Applying the reparameterization trick yields the following:
yt−1 = µq(yt, y0, z, x) + σq(t)ϵ.
(23)
To calculate yt−1, we require the value of y0 estimated at time
step t. Recalling the forward transition density function at an
arbitrary time step q(yt|y0, z, x), y0 given yt is estimated as:
y0 =
1
√¯αt

yt −(1 −√¯αt)(z + Enc(x)) −
√
1 −¯αtϵθ

.
(24)
A step-by-step algorithm summarizing the entire CDM sam-
pling procedure can be found in Algorithm 1.


<!-- page 7 -->
SHEN et al.: IMPROVING ROBUSTNESS AND RELIABILITY IN MEDICAL IMAGE CLASSIFICATION WITH LATENT-GUIDED DIFFUSION AND NESTED-
ENSEMBLES
7
Algorithm 1 Drawing a sample from the CDM
Require: Image input x, latent variable z, and learned parameter θ including Enc(·)
Ensure: A sample y given x and z
1: Draw yT ∼N(z, I)
2: for t in {T, T −1, . . . , 1} do
3:
Compute ˜y0 =
1
√¯αt

yt −(1 −√¯αt)(z + Enc(x)) −√1 −¯αtϵθ(yt, z, x, t)

4:
if t > 1 then
5:
Draw ϵ ∼N(0, I)
6:
Compute yt−1 =
√αt(1−¯αt−1)
1−¯αt
yt + βt
√¯αt−1
1−¯αt
˜y0 −
 √αt(1−¯αt−1)+βt
√¯αt−1
1−¯αt
−1

(z + Enc(x)) +
q
βt(1−¯αt−1)
1−¯αt
ϵ
7:
end if
8: end for
9: Let y = y0
10: return y
Recall that the specification of the component pk(y|x)
enables us to draw a sample from its distribution via sampling
from the CDM. Specifically, y ∼pk(y|x) are drawn through:
ek = Te1:k(Emb(x)), zk = gϕk(ek), y ∼pθk(y|z = zk, x).
(25)
Phase 3. Iterating k from 1 to K results in a set of K ×M
samples. Note that since each sample is a A-dimensional
vector (A is the number of classes) rather than a scalar
class label. Therefore, each sample is mapped to a probability
simplex after averaging them.
Mapping Sample Space to Probability Simplex.
The
CDM guides the mixture model to treat y as a vector sampled
from a real-valued set rather than a categorical distribution.
This is due to the fact that, in the context of denoising score
matching, the loss function used during CDM’s optimization
effectively becomes the squared error (i.e. the Brier score)
between the estimated denoised y and the actual clean y∗from
the data distribution [45], [46]. Consequently, it is crucial to
map the sampled y onto the probability simplex.
Given the estimated expected response vector y ∈RA ob-
tained by averaging all K×M samples from the mixture model
p(y|x, Θ), let ya represent its value in the a-th dimension. The
probability of the final prediction being the class a is then
calculated in the weighted-softmax form of the Brier score
[47]. Specifically, we follow the mapping function introduced
in diffusion-based classifiers by Han et al. [40], where the
hyperparameter ι controls the sharpness of the probability
distribution:
Pr(y = a|x) =
exp
−ι−1(ya −1)2
PA
i=1 exp (−ι−1(yi −1)2)
.
(26)
IV. EXPERIMENTS AND RESULTS
We evaluate the proposed method on two medical imaging
benchmarking datasets: Tuberculosis chest X-ray dataset [23]
and the ISIC Melanoma skin cancer dataset [24]. The Tu-
berculosis chest X-ray dataset consists of X-ray images of
3500 patients with Tuberculosis and 3500 patients without
Tuberculosis. The ISIC Melanoma skin cancer dataset contains
lesion images of 5085 patients with malignant skin cancer
and 5480 patients with benign tumor. A range of baseline
Original
Noisy (Gaussian noise)
Low Resolution
Low Contrast
Adversarial Attack
Fig. 4.
Illustration of the Tuberculosis chest X-ray dataset under
different perturbations (noisy ð = 0.10, low resolution w = 8.00,
contrast r = 0.70, FGSM adversarial attack ε = 0.03).
methods are chosen for comparisons and these cover a variety
of different architectures:
• Comparison with Non-ensemble Methods: To evaluate
the advantages of the ensembling framework over using
individual models, several widely used non-ensemble meth-
ods are included: ResNets [48] and ViTs [42]. Additionally,
comparisons with models based on hybrid architectures are
included, such as MedViT [49].
• Comparison with Existing Ensemble Methods: To pro-
vide a comprehensive evaluation, the proposed ensemble
method is compared against state-of-the-art ensemble deep
learning methods specifically designed for medical image
classification, including deep tree training of convolutional
ensembles (DTT) [19], improved convolutional ensembles
(ICNN-Ensemble) [50] and dynamic-weighted ensembles
(DWE) [20].
Experiment Configuration Details.
For the chest X-
ray dataset, the split for the image-label pairs in train-
ing/validation/testing set is 5670/630/700. For the ISIC skin
cancer dataset, the split for the image-label pairs in train-
ing/validation/testing set is 7605/1000/1960. In both datasets,
all images have binary labels. The test sets are balanced.
Several empirical choices were made: 5 mixture components

[CAPTION] Fig. 4.
Illustration of the Tuberculosis chest X-ray dataset under


<!-- page 8 -->
8
IEEE TRANSACTIONS ON MEDICAL IMAGING, VOL. XX, NO. XX, XXXX 2020
(K = 5), and each were sampled 20 times (M = 20).
For the diffusion model, 1000 time-steps (T = 1000) were
chosen, with a noise schedule of β1 = 10−4, βT = 0.02. For
the probability simplex mapping, we determine the optimal
value of ι by minimizing the ECE on the validation set,
using the Nelder-Mead method [51], specifically, ι = 0.1737
was set for the X-ray dataset, and ι = 0.3162 for the skin
cancer dataset. The transformer encoder blocks in LaDiNE
are extracted from ViT-B [42], and all mapping networks
are implemented by multilayer perceptrons (MLPs) with 3
hidden layers. All CDMs and Enc(·) are implemented with
MLPs with 3 hidden layers, the conditioning in the CDM
(time and image x conditioning) is implemented with element-
wise product. All baseline models and LaDiNE were trained
from scratch until loss convergence. For the baseline methods
that require selecting ensemble members, the procedures in
the original papers were followed exactly as described. The
implementation of LaDiNE, including the code for models as
well as training/evaluation scripts, is made publicly available2.
A. Results without Perturbations
When no perturbations are performed on the input images,
performance is very good for all methods. For the chest X-
ray dataset, when presented with clean inputs (i.e., without
simulated covariate shifts), all methods achieve classification
accuracies that exceed 99.00%. For the skin cancer dataset,
the proposed method achieves accuracies of 94.18% on clean
inputs, on par with the best-performing method ResNet-18
which attains accuracies of 95.02%3.
B. Results Under Perturbations
The robustness of LaDiNE against competing methods is
examined by providing the network with images at test time
that have been perturbed in ways that were not previously
seen during training. To simulate significant covariate shifts, a
variety of perturbations were performed on the clean test set
images. These perturbations included adding Gaussian noise,
altering image resolution, and adjusting contrast levels. Specif-
ically, the following transformation functions are defined:
• Gaussian
Noise.
The function Tgn
:
RH×W ×C
→
RH×W ×C is defined such that
Tgn(x; ð) := x + ðϵ,
(27)
where x is an image in the test set, and ϵ ∼N(0, I). ð ∈R
is a scalar that controls the scale of the injected noise.
• Low Image Resolution.
Reduced-resolution images are
produced by defining a function Tlr
:
RH×W ×C
→
RH×W ×C that
Tlr(x; w) := Resize(DownSample(x, w), (H, W, C)), (28)
where x is an image in the test set, and w is the down-
sampling factor. This function reduces the resolution of the
2The code repository: https://github.com/xingbpshen/nested-diffusion
3For the clean chest X-ray dataset, ViT-B achieves 99.75% accuracy,
LaDiNE achieves 99.90% accuracy. For the clean skin cancer dataset, ViT-B
achieves 92.93% accuracy.
image and then resizes it back to the original dimensions,
simulating a low-resolution effect.
• Image Color Contrast. The color contrast of the images
are manipulated through the function Tc : RH×W ×C →
RH×W ×C that
Tc(x; r) := r(x −¯x) + ¯x,
(29)
where x is an image in the test set, ¯x is the mean value of
all pixels in the image x for each channel, and r ∈R is a
scalar that controls the contrast level. This function adjusts
the contrast of the image by scaling the variance of the pixel
values, enhancing or reducing the overall contrast according
to the value of r.
Results. The experimental results for the robustness exper-
iments on both datasets can be found in Table. I. The results
presented indicate the means and standard deviations of the
classification accuracies (in percentages) over three runs. The
overall trend indicates that LaDiNE consistently outperforms
other methods across almost all perturbations, highlighting its
effectiveness and robustness in handling noisy and perturbed
images. In particular, LaDiNE shows superior performance
on both datasets under high noise levels (ð = 1.00), and
demonstrates the highest robustness among all tested models
overall. Traditional models such as ResNet-18 and ResNet-
50 exhibit poor performance under Gaussian noise, with
accuracies dropping to 50% 4, indicating a failure to generalize
under noisy conditions.
When handling lower-resolution images, LaDiNE achieves
the highest accuracies, particularly in the chest X-ray dataset
(98.90%). Although model performance typically declines
with lower resolution input images, the transformer-based
models (i.e. the second group of models in Table. I), such as
ViT-B and DeiT-B, maintain relatively high accuracies as com-
pared to CNN-based models. When handling lower-contrast
input images, LaDiNE and ResNet-50 lead in robustness on
the ISIC dataset, with LaDiNE achieving 93.14% accuracy.
Transformer-based models generally show robust performance
across varying contrast levels, with SEViT and ConViT-B also
performing well.
C. Results Under Adversarial Attacks
Adversarial attacks can seriously compromise the reliability
and safety of medical imaging models deployed in clinical set-
tings. Several medical machine learning papers have illustrated
how these attacks could result in incorrect diagnoses, inap-
propriate treatments, and even financial exploitation through
insurance fraud [56]–[58].
Adversarial attacks can be formulated as a covariate shift
context where the adversarially perturbed inputs xadv are
sampled from a distribution padv(xadv) that differs from the
original distribution p(x), while the conditional distribution
of labels given the input remains unchanged, i.e., p(y|x) =
padv(y|xadv). In this setting, the adversarial attack induces a
shift in the marginal distribution of the input, creating a context
where the model encounters input distributions during testing
4Note that ResNet-50 consistently provides predictions of “Healthy” for all
input images perturbed with Gaussian noise (ð = 1.00).


<!-- page 9 -->
SHEN et al.: IMPROVING ROBUSTNESS AND RELIABILITY IN MEDICAL IMAGE CLASSIFICATION WITH LATENT-GUIDED DIFFUSION AND NESTED-
ENSEMBLES
9
TABLE I
COMPARISON IN CLASSIFICATION ACCURACY (%) WITH STATE-OF-THE-ART METHODS ON TWO BENCHMARK DATASETS WITH UNSEEN INPUT
PERTURBATIONS. METHODS ARE CATEGORIZED INTO FOUR GROUPS, FROM TOP TO BOTTOM: (I) MODELS BASED ON CONVOLUTIONAL NEURAL
NETWORKS (CNNS). (II) MODELS BASED ON TRANSFORMERS. (III) HYBRID MODELS WITH CNNS AND TRANSFORMERS (CONVIT-B, MEDVIT-B),
AND DIFFUSION MODELS (CARD). (IV) ENSEMBLE LEARNING METHODS.
Methods
Chest X-ray
ISIC
Gaussian noise
Low resolution
Contrast
Gaussian noise
Low resolution
Contrast
ð = 0.50
ð = 1.00
w = 4.00
r = 0.70
ð = 0.50
ð = 1.00
w = 4.00
r = 0.70
ResNet-18 [48]
50.00 ± 0.00
50.00 ± 0.00
50.00 ± 0.00
99.57 ± 0.11
50.56 ± 2.23
49.64 ± 0.94
81.48 ± 2.88
92.37 ± 0.28
ResNet-50 [48]
50.00 ± 0.00
50.38 ± 0.54
50.00 ± 0.00
99.86 ± 0.20
51.02 ± 0.00
54.34 ± 4.69
80.48 ± 3.40
92.42 ± 0.09
EfficientNetV2-L [52]
50.00 ± 0.00
50.00 ± 0.00
96.86 ± 0.42
93.10 ± 0.70
48.98 ± 0.00
48.98 ± 0.00
89.73 ± 0.31
88.45 ± 1.50
DeiT-B [53]
68.86 ± 5.36
57.57 ± 6.25
94.43 ± 2.89
99.57 ± 0.00
69.32 ± 2.04
64.69 ± 1.92
87.75 ± 0.15
92.54 ± 0.67
ViT-B [42]
74.34 ± 2.89
57.76 ± 4.18
94.71 ± 0.71
97.14 ± 0.12
71.90 ± 8.86
55.42 ± 4.00
89.72 ± 0.37
91.58 ± 0.40
Swin-B [54]
59.81 ± 0.04
50.00 ± 0.00
59.29 ± 3.25
98.29 ± 2.42
67.43 ± 1.72
63.93 ± 0.83
88.44 ± 0.60
91.34 ± 0.25
ConViT-B [55]
76.57 ± 4.69
55.00 ± 2.53
94.62 ± 1.67
99.33 ± 0.29
70.86 ± 1.69
60.83 ± 4.08
90.62 ± 0.90
92.93 ± 0.04
MedViT-B [49]
73.00 ± 3.05
52.95 ± 2.20
96.24 ± 0.57
94.67 ± 0.49
61.63 ± 2.02
47.58 ± 2.15
91.46 ± 0.46
90.92 ± 0.37
CARD [40]
75.38 ± 2.86
57.79 ± 3.25
94.86 ± 0.65
97.95 ± 0.13
72.06 ± 8.25
55.41 ± 3.67
90.20 ± 0.34
91.80 ± 0.36
Deep Ensembles [18]
50.00 ± 0.00
50.00 ± 0.00
83.86 ± 0.23
92.71 ± 0.31
50.00 ± 0.00
50.00 ± 0.00
88.32 ± 0.08
91.09 ± 0.06
DWE [20]
74.14 ± 0.58
40.52 ± 1.58
68.14 ± 0.12
72.62 ± 0.13
58.18 ± 0.23
55.14 ± 1.29
71.02 ± 0.07
71.68 ± 0.11
DTT [19]
75.67 ± 0.64
62.90 ± 0.47
97.71 ± 0.42
95.14 ± 0.51
50.60 ± 0.09
50.02 ± 0.10
92.52 ± 0.06
91.85 ± 0.05
ICNN-Ensemble [50]
75.86 ± 0.42
61.43 ± 0.93
97.05 ± 0.41
96.00 ± 0.65
50.34 ± 0.15
50.12 ± 0.10
87.57 ± 0.30
89.52 ± 0.11
SEViT [28]
69.19 ± 3.27
62.24 ± 4.36
97.76 ± 0.33
97.15 ± 0.20
67.04 ± 2.59
54.42 ± 3.64
90.16 ± 0.72
92.00 ± 0.14
LaDiNE (proposed)
78.33 ± 0.69 66.33 ± 2.07 98.90 ± 0.49
97.86 ± 0.23
73.16 ± 2.66 69.93 ± 2.35
91.17 ± 0.36
93.14 ± 0.24
(or deployment) that deviate from those it was trained on, yet
where the underlying relationship between the input and the
label remains consistent.
To complement existing studies on adversarial robustness in
medical image classification, this work presents experimental
results testing adversarial robustness of ensemble learning
methods. Following the procedure described in [28], adver-
sarial perturbation Tadv(·) is applied to image x based on
the gradient of the backbone model (e.g. ViT or ResNet)
Mbase(·), such that ∥x −Tadv(x)∥∞≤ε, where ε ∈R is
a divergence threshold, and Mbase(x) ̸= Mbase(Tadv(x)). In
this work, the top performing methods (for robustness against
noisy conditions) are chosen in order to assess their robustness
to adversarial attacks. For a maximally comprehensive assess-
ment, three gradient-based methods are deployed to generate
adversarial images with a threshold value of ε = 0.03: (i) Fast
Gradient Sign Method (FGSM) [59]; (ii) Projected Gradient
Descent (PGD) [60]; (iii) Auto-PGD [61].
Results.
The results presented in Table. II demonstrate
the performance of various methods under adversarial attacks
using FGSM, PGD, and AutoPGD algorithms. LaDiNE, con-
sistently outperforms other models across both datasets and
all attack types, indicating superior robustness to adversarial
attacks.
For the chest X-ray dataset, LaDiNE achieves the high-
est classification accuracy for FGSM, PGD, and AutoPGD
attacks, respectively. In comparison, SEViT shows strong
performance but falls short of LaDiNE, especially under PGD
and AutoPGD attacks. This shortfall can be attributed to
SEViT’s reliance on the final prediction of the ViT, which is
particularly vulnerable to adversarial perturbations (in contrast,
LaDiNE does not rely on ViT’s final prediction). Traditional
deep learning models such as ResNet-50 and EfficientNetV2-
L, as well as ensemble methods, perform poorly under these
adversarial conditions, with accuracies often dropping to near
zero under PGD and AutoPGD attacks.
For the skin cancer dataset, LaDiNE also leads with ac-
curacies of 60.15%, 61.60%, and 61.30% for FGSM, PGD,
and AutoPGD attacks, respectively. While SEViT performs
relatively well, with accuracy scores in the mid-50s, other
methods like ViT-B and MedViT-B show vulnerabilities to ad-
versarial perturbations, with accuracies dropping substantially
under more sophisticated attacks like PGD and AutoPGD.
D. Results on the Quality of Prediction Confidence
In high-stakes domains such as clinical decision-making, it
is crucial to assess whether a model’s predicted confidence
aligns with its actual performance. One common metric for
this is Expected Calibration Error (ECE), which measures the
discrepancy between confidence scores and observed accu-
racy [62]–[65]. In this work, we evaluate confidence calibra-
tion using ECE across various covariate shift scenarios. Proper
calibration is vital in clinical settings to ensure the model’s
confidence reliably reflects its true performance, reducing the
risk of over-confident and potentially erroneous predictions.
The ECE measures the weighted average of the differences
between predicted confidence and accuracy, over all confi-
dence levels. To compute ECE, the predictions are divided into
several bins based on their confidence scores. For each bin,
the accuracy and confidence are calculated, and the absolute
difference between them is weighted by the number of samples
in the bin. The formula for ECE with b bins is given by:
ECEb :=
b
X
i=1
|Bi|
u
acc(Bi) −conf(Bi)
,
(30)
where Bi denotes the set of indices of predictions that fall
into bin i, u is the total number of predictions, acc(Bi) is
the empirical accuracy for bin i, i.e., the fraction of correct


**[Table p9.1]**
|  | Gaussian noise Low resolution Contrast ð = 0.50 ð = 1.00 w = 4.00 r = 0.70 | Gaussian noise Low resolution Contrast ð = 0.50 ð = 1.00 w = 4.00 r = 0.70 |
| --- | --- | --- |


**[Table p9.2]**
| ResNet-18 [48] ResNet-50 [48] EfficientNetV2-L [52] | 50.00 ± 0.00 50.00 ± 0.00 50.00 ± 0.00 99.57 ± 0.11 50.00 ± 0.00 50.38 ± 0.54 50.00 ± 0.00 99.86 ± 0.20 50.00 ± 0.00 50.00 ± 0.00 96.86 ± 0.42 93.10 ± 0.70 | 50.56 ± 2.23 49.64 ± 0.94 81.48 ± 2.88 92.37 ± 0.28 51.02 ± 0.00 54.34 ± 4.69 80.48 ± 3.40 92.42 ± 0.09 48.98 ± 0.00 48.98 ± 0.00 89.73 ± 0.31 88.45 ± 1.50 |
| --- | --- | --- |


**[Table p9.3]**
| DeiT-B [53] ViT-B [42] Swin-B [54] | 68.86 ± 5.36 57.57 ± 6.25 94.43 ± 2.89 99.57 ± 0.00 74.34 ± 2.89 57.76 ± 4.18 94.71 ± 0.71 97.14 ± 0.12 59.81 ± 0.04 50.00 ± 0.00 59.29 ± 3.25 98.29 ± 2.42 | 69.32 ± 2.04 64.69 ± 1.92 87.75 ± 0.15 92.54 ± 0.67 71.90 ± 8.86 55.42 ± 4.00 89.72 ± 0.37 91.58 ± 0.40 67.43 ± 1.72 63.93 ± 0.83 88.44 ± 0.60 91.34 ± 0.25 |
| --- | --- | --- |


**[Table p9.4]**
| ConViT-B [55] MedViT-B [49] CARD [40] | 76.57 ± 4.69 55.00 ± 2.53 94.62 ± 1.67 99.33 ± 0.29 73.00 ± 3.05 52.95 ± 2.20 96.24 ± 0.57 94.67 ± 0.49 75.38 ± 2.86 57.79 ± 3.25 94.86 ± 0.65 97.95 ± 0.13 | 70.86 ± 1.69 60.83 ± 4.08 90.62 ± 0.90 92.93 ± 0.04 61.63 ± 2.02 47.58 ± 2.15 91.46 ± 0.46 90.92 ± 0.37 72.06 ± 8.25 55.41 ± 3.67 90.20 ± 0.34 91.80 ± 0.36 |
| --- | --- | --- |


**[Table p9.5]**
| Deep Ensembles [18] DWE [20] DTT [19] ICNN-Ensemble [50] SEViT [28] LaDiNE (proposed) | 50.00 ± 0.00 50.00 ± 0.00 83.86 ± 0.23 92.71 ± 0.31 74.14 ± 0.58 40.52 ± 1.58 68.14 ± 0.12 72.62 ± 0.13 75.67 ± 0.64 62.90 ± 0.47 97.71 ± 0.42 95.14 ± 0.51 75.86 ± 0.42 61.43 ± 0.93 97.05 ± 0.41 96.00 ± 0.65 69.19 ± 3.27 62.24 ± 4.36 97.76 ± 0.33 97.15 ± 0.20 78.33 ± 0.69 66.33 ± 2.07 98.90 ± 0.49 97.86 ± 0.23 | 50.00 ± 0.00 50.00 ± 0.00 88.32 ± 0.08 91.09 ± 0.06 58.18 ± 0.23 55.14 ± 1.29 71.02 ± 0.07 71.68 ± 0.11 50.60 ± 0.09 50.02 ± 0.10 92.52 ± 0.06 91.85 ± 0.05 50.34 ± 0.15 50.12 ± 0.10 87.57 ± 0.30 89.52 ± 0.11 67.04 ± 2.59 54.42 ± 3.64 90.16 ± 0.72 92.00 ± 0.14 73.16 ± 2.66 69.93 ± 2.35 91.17 ± 0.36 93.14 ± 0.24 |
| --- | --- | --- |


<!-- page 10 -->
10
IEEE TRANSACTIONS ON MEDICAL IMAGING, VOL. XX, NO. XX, XXXX 2020
TABLE II
COMPARISON IN CLASSIFICATION ACCURACY (%) WITH STATE-OF-THE-ART METHODS ON TWO BENCHMARKS WITH ADVERSARIAL ATTACKS.
Methods
Chest X-ray
ISIC
FGSM [59]
PGD [60]
AutoPGD [61]
FGSM
PGD
AutoPGD
ResNet-50 [48]
46.72 ± 3.40
0.00 ± 0.00
0.00 ± 0.00
55.32 ± 0.24
0.00 ± 0.00
0.00 ± 0.00
EfficientNetV2-L [52]
34.28 ± 1.02
0.19 ± 0.07
0.14 ± 0.12
19.75 ± 6.24
0.00 ± 0.00
0.00 ± 0.00
DeiT-B [53]
35.28 ± 1.76
0.00 ± 0.00
0.00 ± 0.00
26.33 ± 7.15
0.00 ± 0.00
0.00 ± 0.00
ViT-B [42]
15.38 ± 3.68
0.14 ± 0.12
0.00 ± 0.00
22.20 ± 8.22
0.29 ± 0.17
0.00 ± 0.00
ConViT-B [55]
20.52 ± 2.66
0.00 ± 0.00
0.00 ± 0.00
35.95 ± 0.74
0.02 ± 0.02
0.00 ± 0.00
MedViT-B [49]
10.29 ± 4.50
2.95 ± 2.10
0.38 ± 0.36
23.93 ± 6.28
0.00 ± 0.00
0.00 ± 0.00
CARD [40]
14.81 ± 2.01
0.00 ± 0.00
0.00 ± 0.00
18.50 ± 4.13
0.20 ± 0.13
0.00 ± 0.00
Deep Ensembles [18]
34.67 ± 0.07
0.00 ± 0.00
0.00 ± 0.00
20.71 ± 0.07
0.00 ± 0.00
0.00 ± 0.00
DTT [19]
10.29 ± 0.47
0.76 ± 0.07
0.57 ± 0.03
26.46 ± 0.06
5.83 ± 0.03
2.48 ± 0.02
ICNN-Ensemble [50]
49.86 ± 0.31
0.00 ± 0.00
0.00 ± 0.00
41.46 ± 0.28
1.11 ± 0.02
0.00 ± 0.00
SEViT [28]
85.90 ± 3.39
92.76 ± 0.86
94.24 ± 1.36
54.15 ± 3.56
51.52 ± 4.99
57.30 ± 8.74
LaDiNE (proposed)
94.86 ± 0.20
96.10 ± 0.94
96.05 ± 1.48
60.15 ± 4.93
61.60 ± 4.53
61.30 ± 2.70
ResNet-50
EfficientNetV2-L
DeiT-B
ViT-B
ConViT-B
MedViT-B
Deep Ensembles
DTT
ICNN-Ensemble
SEViT
LaDiNE (ours)
Methods
0.0
0.1
0.2
0.3
0.4
0.5
ECE
0.45
0.50
0.24
0.30 0.32
0.45
0.22 0.22
0.37
0.26
0.20
(a)
ResNet-50
EfficientNetV2-L
DeiT-B
ViT-B
ConViT-B
MedViT-B
Deep Ensembles
DTT
ICNN-Ensemble
SEViT
LaDiNE (ours)
Methods
0.0
0.2
0.4
0.6
0.8
ECE
0.46
0.66
0.71
0.83
0.77 0.81
0.28
0.87
0.50
0.07
0.02
(b)
ResNet-50
EfficientNetV2-L
DeiT-B
ViT-B
ConViT-B
MedViT-B
Deep Ensembles
DTT
ICNN-Ensemble
SEViT
LaDiNE (ours)
Methods
0.0
0.1
0.2
0.3
0.4
0.5
ECE
0.49 0.51
0.26
0.46
0.40
0.51
0.05
0.50
0.47 0.47
0.14
(c)
ResNet-50
EfficientNetV2-L
DeiT-B
ViT-B
ConViT-B
MedViT-B
Deep Ensembles
DTT
ICNN-Ensemble
SEViT
LaDiNE (ours)
Methods
0.0
0.2
0.4
0.6
0.8
ECE
0.41
0.90
0.69
0.75
0.33
0.86
0.46
0.59
0.51
0.39
0.26
(d)
Fig. 5.
Plot of expected calibration error (ECE) with a uniform set of ten bins: (a) Gaussian noise injection (ð = 1.00) in the chest X-ray dataset.
(b) FGSM attack (ε = 0.03) in the chest X-ray dataset. (c) Gaussian noise injection (ð = 1.00) in the skin cancer dataset, (d) FGSM attack
(ε = 0.03) for the skin cancer dataset.
predictions in the bin, and conf(Bi) is the average confidence
score for bin i.
Results. All methods are tested under covariate shifts on
both datasets. Accurately expressing their prediction confi-
dence is important in order to avoid being over-confident in
incorrect predictions. Overall, LaDiNE and Deep Ensembles,
a scalable method for improving predictive uncertainty es-
timation [18], show stronger capabilities in providing well-
calibrated confidence scores among all methods in both
datasets. Furthermore, LaDiNE achieves a lower ECE than
Deep Ensembles (i) under adversarial perturbations in both
datasets and (ii) under Gaussian noise injections in the chest
X-ray dataset, as illustrated in Fig. 5. Under Gaussian noise
injections, Deep Ensembles reaches the lowest ECE in the skin
cancer dataset, however, its classification accuracy under this
condition is 50.00% which indicates low predictive power as
compared to LaDiNE’s accuracy of 73.16% in this case.
E. Quantifying Instance-level Uncertainty
Quantifying the reliability of a model’s predictions is critical
in clinical settings, where the consequences of presenting
incorrect predictions can be significant. In order to maintain
trust in the system, the model should quantify the level of
uncertainty in each prediction, with the goal of being correct
when confident, and uncertain when incorrect [66]. In this
fashion, clinicians can focus their review on cases where the
model is less certain, thereby improving decision-making and
fostering trust in the system. This section shows results in the
quantification of LaDiNE’s prediction uncertainties given an
image instance under covariate shifts.
Recall that LaDiNE provides K × M prediction vectors ˆy
for a given image x (see Sec. III-C). These prediction vectors
are denoted as a set S and a set of scalars are defined: Sa :=
{ˆya
i | ˆyi ∈S} (note that ˆya
i denotes the value of ˆyi in the
a-th dimension). Uncertainties are measured with respect to
the consistency among the samples predictions. To this end,
two methods are used:
• Class-wise Prediction Interval Width (CPIW).
The
CPIW, defined by Han et al., measures the uncertainties of
the predictions provided by a diffusion model [40]:
CPIWa := Q97.5(Sa) −Q2.5(Sa),
(31)
where Qn(·) calculates the n-th percentile. CPIW measures
the spread of the model’s predictions for a specific class
a. A smaller CPIW indicates that the predictions are more
tightly clustered, suggesting higher certainty in the model’s
prediction for that class. Conversely, a larger CPIW suggests
greater variability, and thus greater uncertainty.
• Class-wise Normalized Prediction Variance (CNPV).
Calculating the prediction variance is a common method
for quantifying uncertainty in medical imaging [67]. The


**[Table p10.1]**
|  | FGSM [59] PGD [60] AutoPGD [61] | FGSM PGD AutoPGD |
| --- | --- | --- |


**[Table p10.2]**
| ResNet-50 [48] EfficientNetV2-L [52] DeiT-B [53] ViT-B [42] ConViT-B [55] MedViT-B [49] CARD [40] Deep Ensembles [18] DTT [19] ICNN-Ensemble [50] SEViT [28] LaDiNE (proposed) | 46.72 ± 3.40 0.00 ± 0.00 0.00 ± 0.00 34.28 ± 1.02 0.19 ± 0.07 0.14 ± 0.12 35.28 ± 1.76 0.00 ± 0.00 0.00 ± 0.00 15.38 ± 3.68 0.14 ± 0.12 0.00 ± 0.00 20.52 ± 2.66 0.00 ± 0.00 0.00 ± 0.00 10.29 ± 4.50 2.95 ± 2.10 0.38 ± 0.36 14.81 ± 2.01 0.00 ± 0.00 0.00 ± 0.00 34.67 ± 0.07 0.00 ± 0.00 0.00 ± 0.00 10.29 ± 0.47 0.76 ± 0.07 0.57 ± 0.03 49.86 ± 0.31 0.00 ± 0.00 0.00 ± 0.00 85.90 ± 3.39 92.76 ± 0.86 94.24 ± 1.36 94.86 ± 0.20 96.10 ± 0.94 96.05 ± 1.48 | 55.32 ± 0.24 0.00 ± 0.00 0.00 ± 0.00 19.75 ± 6.24 0.00 ± 0.00 0.00 ± 0.00 26.33 ± 7.15 0.00 ± 0.00 0.00 ± 0.00 22.20 ± 8.22 0.29 ± 0.17 0.00 ± 0.00 35.95 ± 0.74 0.02 ± 0.02 0.00 ± 0.00 23.93 ± 6.28 0.00 ± 0.00 0.00 ± 0.00 18.50 ± 4.13 0.20 ± 0.13 0.00 ± 0.00 20.71 ± 0.07 0.00 ± 0.00 0.00 ± 0.00 26.46 ± 0.06 5.83 ± 0.03 2.48 ± 0.02 41.46 ± 0.28 1.11 ± 0.02 0.00 ± 0.00 54.15 ± 3.56 51.52 ± 4.99 57.30 ± 8.74 60.15 ± 4.93 61.60 ± 4.53 61.30 ± 2.70 |
| --- | --- | --- |


**[Table p10.3]**
| 0.50 |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.45 |  | 0.45 |  |  |  |  |  |  |  |  |
|  |  | 0.30 0.32 |  |  |  |  |  | 0.37 | 0.26 |  |
|  |  | 0.24 |  |  |  | 0.22 0.22 |  |  |  | 0.20 |
|  |  |  |  |  |  |  |  |  |  |  |


**[Table p10.4]**
| 0.87 0.83 |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.71 0.66 0.46 |  |  |  | 0.77 | 0.81 |  |  | 0.50 |  |
|  | 0.66 |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  | 0.28 |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  | 0.07 0.02 |
|  |  |  |  |  |  |  |  |  |  |


**[Table p10.5]**
| 0.49 0.51 0.51 0.50 0.47 |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  | 0.46 0.40 |  |  |  |  |  |  | 0.47 |  |
|  |  | 0.26 |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  | 0.14 |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  | 0.05 |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |


**[Table p10.6]**
| 0.90 0.86 |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  | 0.75 0.69 |  |  |  | 0.59 0.51 0.46 |  |  |  |  |
|  |  | 0.69 |  |  |  |  |  |  |  |  |
| 0.41 |  |  |  | 0.33 |  |  |  |  | 0.39 0.26 |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |

[CAPTION] Fig. 5.
Plot of expected calibration error (ECE) with a uniform set of ten bins: (a) Gaussian noise injection (ð = 1.00) in the chest X-ray dataset.


<!-- page 11 -->
SHEN et al.: IMPROVING ROBUSTNESS AND RELIABILITY IN MEDICAL IMAGE CLASSIFICATION WITH LATENT-GUIDED DIFFUSION AND NESTED-
ENSEMBLES
11
TABLE III
RESULTS OF UNCERTAINTY MEASURES UNDER COVARIATE SHIFT
(IMAGES WITH GAUSSIAN NOISE INJECTION ð = 1.00).
Class
Evaluated Predictions
CPIW
CNPV
Tuberculosis
Correct
0.4330
0.2520
Incorrect
0.8600
0.7196
Healthy
Correct
0.9998
0.9112
Incorrect
0.9997
0.8940
Tuberculosis
Healthy
Fig. 6.
Two images from chest X-ray dataset with Gaussian noise
injection (ð = 1.00): (i) Patient with tuberculosis (left). (ii) Healthy
patient (right).
CNPV is defined as follows:
CNPVa := |Sa|−1
|Sa|
X
i=1
4(ˆya
i −¯ya)2,
(32)
where |Sa| denotes the cardinality of the set Sa, and ¯ya
denotes the mean of all values in Sa. CNPV quantifies the
variability of the predictions by calculating the normalized
variance of the samples in Sa. A lower CNPV value
indicates that the predictions are consistent and the model
is confident in its decision for that class. Higher CNPV
values suggest more uncertainty, as the predictions vary
more significantly around the mean.
In order to examine the power of the method in challenging
contexts, the input images are perturbed significantly with
Gaussian noise (ð = 1.00), see example images in Fig. 6.
This level of Gaussian noise results in it being challenging to
differentiate healthy images from tuberculosis images. Results
shown in Table. III illustrate the instance-level uncertainty
estimates for the LaDiNE’s predictions on different classes,
specifically focusing on the distinction between correct and
incorrect predictions. The first thing to note is that the results
reflect how the challenging context results in high uncertain-
ties for the healthy class predictions. LaDiNE appropriately
expresses the uncertainty in these challenging cases, informing
clinicians to review those uncertain instances more carefully.
On the other hand, LaDiNE is more certain when correctly
predicting tuberculosis, which demonstrates the model’s ro-
bustness in detecting true unhealthy cases.
F. Ablation Studies
The effect on classification performance is examined when
a variety of other design choices are implemented.
Results on Element-wise Ablation Studies. To further
justify the design choices made for each component of the
framework, the proposed model is tested on four configu-
rations of (i) a clean chest X-ray testing dataset and (ii) a
noise injected chest X-ray testing dataset (Gaussian noise with
ð = 1.00):
1. In this configuration, CDM is removed from LaDiNE
and instead a deterministic mapping to the parameters of
the final categorical distribution is used. Specifically, the
softmaxed latent variable z serves as the predicted class
probabilities.
2. Instead of learning the distribution with CDM, the CDM is
replaced with a Gaussian distribution parameterized by a
two-head neural network to estimate the mean and variance
of y conditioned on the latent variable z. 20 samples
are drawn from the Gaussian distribution per ensemble
member and the average confidence is estimated.
3. To investigate the effectiveness of the inferred latent vari-
able z, z is replaced with the output logits from the ViT. 20
samples are drawn from the CDM per ensemble member
and the average confidence is estimated.
4. The entire LaDiNE is examined, where 20 samples are
drawn from the CDM per ensemble member and the
average confidence is estimated.
As shown in Table. IV, the complete version of LaDiNE
(design 4) achieves the highest accuracy for all testing sets, the
lowest relative accuracy drop under noisy conditions, and the
lowest Expected Calibration Error (ECE), providing support
and justification for the design elements chosen.
In Design 1 which does not make use of a CDM, the
accuracy drops especially under Gaussian noise, and the
ECE increases, indicating that the CDM is crucial for robust
performance and reliable uncertainty estimation under input
image perturbation. Replacing the CDM with a Gaussian
distribution (Design 2) also leads to a notable performance
degradation, suggesting that the flexibility of the CDM in
modeling complex distributions contributes to better accuracy
and calibration. On the other hand, Design 2 achieves a
lower ECE than Design 1, indicating that encoding predictive
confidence (the Gaussian distribution in Design 2) can help
mitigate issues with over-confidence.
When the latent variable z is removed (Design 3), the model
experiences a severe drop in performance under input image
perturbation, emphasizing the importance of z in capturing
informative and invariant representations that are critical for
generalization and robustness.
Overall, the ablation studies clearly demonstrate that both
the CDM and the latent variable z play essential roles in the
superior performance of LaDiNE, especially in handling of
noisy data from outside the learned distribution.
Results on Selection of Mixture Components. Another
key design choice in our method is the selection of mixture
components, in other words, the TE hierarchy. Table. V shows
the performance of the proposed method with K = 3, 4, . . . , 7
(we draw 20 samples per ensemble member). When K = 5,
the classification accuracy is the highest for both datasets.
When K is smaller, there is insufficient discrimination in the
extracted features which therefore results in low classification
accuracy. On the other hand, as K increases, the inner structure
of feature representations from the different hierarchies be-


**[Table p11.1]**
| Healthy | Correct 0.9998 0.9112 Incorrect 0.9997 0.8940 |
| --- | --- |

[CAPTION] Fig. 6.
Two images from chest X-ray dataset with Gaussian noise


<!-- page 12 -->
12
IEEE TRANSACTIONS ON MEDICAL IMAGING, VOL. XX, NO. XX, XXXX 2020
TABLE IV
COMPARISON IN CLASSIFICATION ACCURACY (ACC. %) AND EXPECTED CALIBRATION ERROR (ECE) WITH DIFFERENT DESIGNS. M INDICATES
THE NUMBER OF SAMPLING TIMES FROM THE DISTRIBUTION. WE INVESTIGATE THE EFFECTIVENESS OF INFERRED LATENT VARIABLE AND
DIFFUSION MODEL IN IMPROVING CLASSIFICATION ACCURACY AND CONFIDENCE CALIBRATION.
Design
(K = 5)
With
latent
variable z
With
CDM
Functional
assumption
Clean
Gaussian noise
Acc.
ECE
Acc.
(- % drop compare to Clean)
ECE
1
✓
✗
Dirac delta (deterministic) (M=1)
99.71
0.3614
62.29 (-37.53%)
0.3732
2
✓
✗
Gaussian (M=20, avg. conf.)
98.86
0.2331
61.43 (-37.86%)
0.2485
3
✗
✓
Any (M=20, avg. conf.)
99.77
0.0031
57.42 (-42.44%)
0.2273
4
✓
✓
Any (M=20, avg. conf.)
99.90
0.0030
66.33 (-33.60%)
0.1960
TABLE V
COMPARISON IN CLASSIFICATION ACCURACY (%) WITH DIFFERENT
CHOICE OF MIXTURE COMPONENTS.
Chest X-ray
ISIC
K = 3
96.47 ± 0.29
87.84 ± 0.13
K = 4
98.63 ± 0.60
91.24 ± 0.43
K = 5
99.90 ± 0.14
94.18 ± 0.24
K = 6
97.59 ± 0.25
92.40 ± 0.54
K = 7
98.31 ± 0.38
92.14 ± 0.41
comes too complex and may therefore result in a performance
drop.
Examining the results in Table. V and the findings in Fig. 1,
one can observe a trade-off between informativeness and
invariance in selecting early TE blocks’ representation. Specif-
ically:
• Invariance. Shallow blocks tend to capture more invariant
features across different environments, providing a stable
representation that is less sensitive to specific variations in
the input data. These invariant features are beneficial for
maintaining consistency and robustness, especially under
covariate shifts. However, these representations may encode
less direct information for classifying the input data.
• Informative.
Higher blocks, on the other hand, encode
more informative and discriminative features. These fea-
tures capture more detailed and specific characteristics of
the input data, which can enhance classification accuracy.
However, this increased specificity can also lead to reduced
invariance, and can result poor generalization due to over-
parameterization of the mixture.
V. LIMITATIONS AND FUTURE WORKS
One of the limitations of our approach is the increased com-
putational cost associated with diffusion models, which require
iterative denoising processes. Although diffusion models are
expensive, the denoising process is on RA instead of on the
whole image space (A is the number of classes). Inference
time is still reasonable for many medical imaging tasks. In
our experiments, the inference time (on one NVIDIA A100
GPU) per image is approximately 98.23 ms for LaDiNE (we
sequentially sample K×M times, where K = 5 and M = 20),
compared to 0.54 ms for the ViT-B baseline. Future work
could explore faster sampling techniques, such as Denoising
Diffusion Implicit Models (DDIM) [68] or consistency models
[69], which can reduce the number of required iterations
and computational overhead while maintaining comparable
performance. In addition to incorporating accelerated sampling
techniques, the computation of ensemble members (imple-
mented by K mapping networks and denoising networks)
can be parallelized to further reduce the latency and improve
efficiency. By distributing the ensemble’s workload across
multiple processing units (e.g., multiple graphical processing
units), inference times can be significantly reduced, making the
approach more efficient and practical for various applications.
Another limitation is the slight variability in performance
depending on the neural network initialization compared to
other methods. This sensitivity can lead to inconsistencies in
model outputs. Future research could integrate Bayesian deep
learning techniques, which explicitly model uncertainty in
the network parameters. Approaches such as Bayesian neural
networks (BNNs) or approximate Bayesian inference [70] can
provide more reliable uncertainty estimates and help stabilize
performance across different initializations.
VI. CONCLUSION
In this work, we present a novel ensemble learning ap-
proach, LaDiNE, designed to improve the robustness and
reliability of medical image classification under covariate
shifts. By learning invariant features and modeling the pre-
dictive distribution with a functional-form-free mixture, the
proposed approach effectively addresses the challenges of
image perturbations and adversarial attacks on the inputs,
and achieving calibrated confidence levels in its predictions.
Extensive experiments on benchmark datasets demonstrate the
superiority of LaDiNE in achieving high classification accu-
racy and well-calibrated prediction confidence under various
challenging conditions. This work underscores the importance
of robust and reliable models in clinical decision-making,
providing a pathway for future advancements in trustworthy
artificial intelligence for medical image analysis.
REFERENCES
[1] A. Esteva, B. Kuprel, R. A. Novoa, J. Ko, S. M. Swetter, H. M. Blau,
and S. Thrun, “Dermatologist-level classification of skin cancer with
deep neural networks,” Nature, vol. 542, no. 7639, pp. 115–118, 2017.
[2] V.
Gulshan,
L.
Peng,
M.
Coram,
M.
C.
Stumpe,
D.
Wu,
A. Narayanaswamy, S. Venugopalan, K. Widner, T. Madams, J. Cuadros
et al., “Development and validation of a deep learning algorithm for
detection of diabetic retinopathy in retinal fundus photographs,” Journal
of the American Medical Association, vol. 316, no. 22, pp. 2402–2410,
2016.


**[Table p12.1]**
| variable z | Acc. ECE | Acc. ECE (- % drop compare to Clean) |
| --- | --- | --- |


**[Table p12.2]**
| 1 ✓ ✗ Dirac delta (deterministic) (M=1) 2 ✓ ✗ Gaussian (M=20, avg. conf.) 3 ✗ ✓ Any (M=20, avg. conf.) | 99.71 0.3614 98.86 0.2331 99.77 0.0031 | 62.29 (-37.53%) 0.3732 61.43 (-37.86%) 0.2485 57.42 (-42.44%) 0.2273 |
| --- | --- | --- |


**[Table p12.3]**
| K = 3 K = 4 K = 5 K = 6 K = 7 | 96.47 ± 0.29 98.63 ± 0.60 99.90 ± 0.14 97.59 ± 0.25 98.31 ± 0.38 | 87.84 ± 0.13 91.24 ± 0.43 94.18 ± 0.24 92.40 ± 0.54 92.14 ± 0.41 |
| --- | --- | --- |


<!-- page 13 -->
SHEN et al.: IMPROVING ROBUSTNESS AND RELIABILITY IN MEDICAL IMAGE CLASSIFICATION WITH LATENT-GUIDED DIFFUSION AND NESTED-
ENSEMBLES
13
[3] D. Ardila, A. P. Kiraly, S. Bharadwaj, B. Choi, J. J. Reicher, L. Peng,
D. Tse, M. Etemadi, W. Ye, G. Corrado et al., “End-to-end lung
cancer screening with three-dimensional deep learning on low-dose chest
computed tomography,” Nature Medicine, vol. 25, no. 6, pp. 954–961,
2019.
[4] B. E. Bejnordi, M. Veta, P. J. Van Diest, B. Van Ginneken, N. Karsse-
meijer, G. Litjens, J. A. Van Der Laak, M. Hermsen, Q. F. Manson,
M. Balkenhol et al., “Diagnostic assessment of deep learning algorithms
for detection of lymph node metastases in women with breast cancer,”
Journal of the American Medical Association, vol. 318, no. 22, pp. 2199–
2210, 2017.
[5] J. De Fauw, J. R. Ledsam, B. Romera-Paredes, S. Nikolov, N. Tomasev,
S. Blackwell, H. Askham, X. Glorot, B. O’Donoghue, D. Visentin et al.,
“Clinically applicable deep learning for diagnosis and referral in retinal
disease,” Nature Medicine, vol. 24, no. 9, pp. 1342–1350, 2018.
[6] G. Litjens, T. Kooi, B. E. Bejnordi, A. A. A. Setio, F. Ciompi,
M. Ghafoorian, J. A. Van Der Laak, B. Van Ginneken, and C. I. S´anchez,
“A survey on deep learning in medical image analysis,” Medical Image
Analysis, vol. 42, pp. 60–88, 2017.
[7] Y. Qiu, Y. Wang, S. Yan, M. Tan, S. Cheng, H. Liu, and B. Zheng,
“An initial investigation on developing a new method to predict short-
term breast cancer risk based on deep learning technology,” in Medical
Imaging 2016: Computer-Aided Diagnosis, vol. 9785.
SPIE, 2016, pp.
517–522.
[8] N. Tajbakhsh, J. Y. Shin, S. R. Gurudu, R. T. Hurst, C. B. Kendall, M. B.
Gotway, and J. Liang, “Convolutional neural networks for medical image
analysis: Full training or fine tuning?” IEEE Transactions on Medical
Imaging, vol. 35, no. 5, pp. 1299–1312, 2016.
[9] F. Milletari, N. Navab, and S.-A. Ahmadi, “V-net: Fully convolutional
neural networks for volumetric medical image segmentation,” in Inter-
national Conference on 3D Vision.
IEEE, 2016, pp. 565–571.
[10] D. Hendrycks and T. Dietterich, “Benchmarking neural network ro-
bustness to common corruptions and perturbations,” in International
Conference on Learning Representations, 2018.
[11] C. Szegedy, W. Zaremba, I. Sutskever, J. Bruna, D. Erhan, I. Good-
fellow, and R. Fergus, “Intriguing properties of neural networks,” in
International Conference on Learning Representations, 2014.
[12] F. Navarro, C. Watanabe, S. Shit, A. Sekuboyina, J. C. Peeken, S. E.
Combs, and B. H. Menze, “Evaluating the robustness of self-supervised
learning in medical imaging,” arXiv preprint arXiv:2105.06986, 2021.
[13] J. R¨oglin, K. Ziegeler, J. Kube, F. K¨onig, K.-G. Hermann, and S. Ort-
mann, “Improving classification results on a small medical dataset using
a gan; an outlook for dealing with rare disease datasets,” Frontiers in
Computer Science, vol. 4, p. 858874, 2022.
[14] T. Takase, R. Karakida, and H. Asoh, “Self-paced data augmentation
for training neural networks,” Neurocomputing, vol. 442, pp. 296–306,
2021.
[15] S. Matta, M. Lamard, P. Zhang, A. Le Guilcher, L. Borderie, B. Coch-
ener, and G. Quellec, “A systematic review of generalization research in
medical image classification,” Computers in biology and medicine, vol.
183, p. 109256, 2024.
[16] L. Breiman, “Bagging predictors,” Machine Learning, vol. 24, pp. 123–
140, 1996.
[17] I. D. Mienye and Y. Sun, “A survey of ensemble learning: Concepts,
algorithms, applications, and prospects,” IEEE Access, vol. 10, pp.
99 129–99 149, 2022.
[18] B. Lakshminarayanan, A. Pritzel, and C. Blundell, “Simple and scalable
predictive uncertainty estimation using deep ensembles,” in Advances in
Neural Information Processing Systems, vol. 30, 2017.
[19] Y. Yang, Y. Hu, X. Zhang, and S. Wang, “Two-stage selective ensemble
of cnn via deep tree training for medical image classification,” IEEE
Transactions on Cybernetics, vol. 52, no. 9, pp. 9194–9207, 2021.
[20] A. G. C. Pacheco, T. Trappenberg, and R. Krohling, “Learning dynamic
weights for an ensemble of deep models applied to medical imaging
classification,” 2020 International Joint Conference on Neural Networks
(IJCNN), pp. 1–8, 2020.
[21] F. Shamshad, S. Khan, S. W. Zamir, M. H. Khan, M. Hayat, F. S. Khan,
and H. Fu, “Transformers in medical imaging: A survey,” Medical Image
Analysis, p. 102802, 2023.
[22] A. Kazerouni, E. K. Aghdam, M. Heidari, R. Azad, M. Fayyaz,
I. Hacihaliloglu, and D. Merhof, “Diffusion models in medical imaging:
A comprehensive survey,” Medical Image Analysis, p. 102846, 2023.
[23] T. Rahman, A. Khandakar, M. A. Kadir, K. R. Islam, K. F. Islam,
R. Mazhar, T. Hamid, M. T. Islam, S. Kashem, Z. B. Mahbub et al.,
“Reliable tuberculosis detection using chest x-ray with deep learning,
segmentation and visualization,” IEEE Access, vol. 8, pp. 191 586–
191 601, 2020.
[24] V. Rotemberg, N. Kurtansky, B. Betz-Stablein, L. Caffery, E. Chousakos,
N. Codella, M. Combalia, S. Dusza, P. Guitera, D. Gutman et al.,
“A patient-centric dataset of images and metadata for identifying
melanomas using clinical context,” Scientific Data, vol. 8, no. 1, p. 34,
2021.
[25] H. Peiris, M. Hayat, Z. Chen, G. Egan, and M. Harandi, “A robust
volumetric transformer for accurate 3d tumor segmentation,” in Interna-
tional conference on medical image computing and computer-assisted
intervention.
Springer, 2022, pp. 162–172.
[26] H. Chen, C. Li, G. Wang, X. Li, M. M. Rahaman, H. Sun, W. Hu,
Y. Li, W. Liu, C. Sun et al., “Gashis-transformer: A multi-scale visual
transformer approach for gastric histopathological image detection,”
Pattern Recognition, vol. 130, p. 108827, 2022.
[27] C. Wang, K. Shang, H. Zhang, Q. Li, Y. Hui, and S. K. Zhou,
“Dudotrans: dual-domain transformer provides more attention for sino-
gram restoration in sparse-view ct reconstruction,” arXiv preprint
arXiv:2111.10790, 2021.
[28] F. Almalik, M. Yaqub, and K. Nandakumar, “Self-ensembling vision
transformer (sevit) for robust medical image classification,” in Medical
Image Computing and Computer Assisted Intervention–MICCAI 2022:
25th International Conference, Singapore, September 18–22, 2022,
Proceedings, Part III.
Springer, 2022, pp. 376–386.
[29] P. Dhariwal and A. Nichol, “Diffusion models beat gans on image
synthesis,” Advances in neural information processing systems, vol. 34,
pp. 8780–8794, 2021.
[30] W. H. Pinaya, P.-D. Tudosiu, J. Dafflon, P. F. Da Costa, V. Fernandez,
P. Nachev, S. Ourselin, and M. J. Cardoso, “Brain imaging generation
with latent diffusion models,” in MICCAI Workshop on Deep Generative
Models.
Springer, 2022, pp. 117–126.
[31] P. A. Moghadam, S. Van Dalen, K. C. Martin, J. Lennerz, S. Yip,
H. Farahani, and A. Bashashati, “A morphology focused diffusion prob-
abilistic model for synthesis of histopathology images,” in Proceedings
of the IEEE/CVF winter conference on applications of computer vision,
2023, pp. 2000–2009.
[32] J. S. Yoon, C. Zhang, H.-I. Suk, J. Guo, and X. Li, “Sadm: Sequence-
aware diffusion model for longitudinal medical image generation,” in
International Conference on Information Processing in Medical Imag-
ing.
Springer, 2023, pp. 388–400.
[33] Y. Song, L. Shen, L. Xing, and S. Ermon, “Solving inverse problems in
medical imaging with score-based generative models,” in International
Conference on Learning Representations, 2021.
[34] Y. Xie and Q. Li, “Measurement-conditioned denoising diffusion prob-
abilistic model for under-sampled medical image reconstruction,” in
International Conference on Medical Image Computing and Computer-
Assisted Intervention.
Springer, 2022, pp. 655–664.
[35] J. Wu, R. Fu, H. Fang, Y. Zhang, Y. Yang, H. Xiong, H. Liu, and Y. Xu,
“Medsegdiff: Medical image segmentation with diffusion probabilistic
model,” in Medical Imaging with Deep Learning.
PMLR, 2024, pp.
1623–1639.
[36] J. Wolleb, R. Sandk¨uhler, F. Bieder, P. Valmaggia, and P. C. Cattin,
“Diffusion models for implicit image segmentation ensembles,” in Inter-
national Conference on Medical Imaging with Deep Learning.
PMLR,
2022, pp. 1336–1348.
[37] Y. Li, H.-C. Shao, X. Liang, L. Chen, R. Li, S. Jiang, J. Wang, and
Y. Zhang, “Zero-shot medical image translation via frequency-guided
diffusion models,” IEEE Transactions on Medical Imaging, 2023.
[38] B. Kim, Y. Oh, and J. C. Ye, “Diffusion adversarial representation learn-
ing for self-supervised vessel segmentation,” in International Conference
on Learning Representations, 2022.
[39] K. Clark and P. Jaini, “Text-to-image diffusion models are zero shot
classifiers,” Advances in Neural Information Processing Systems, vol. 36,
2024.
[40] X. Han, H. Zheng, and M. Zhou, “Card: Classification and regression
diffusion models,” in Advances in Neural Information Processing Sys-
tems, vol. 35, 2022, pp. 18 100–18 115.
[41] H. Chen, Y. Dong, S. Shao, Z. Hao, X. Yang, H. Su, and J. Zhu,
“Diffusion models are certifiably robust classifiers,” in The Thirty-eighth
Annual Conference on Neural Information Processing Systems, 2024.
[42] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai,
T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al.,
“An image is worth 16x16 words: Transformers for image recognition at
scale,” in International Conference on Learning Representations, 2021.
[43] M. Walmer, S. Suri, K. Gupta, and A. Shrivastava, “Teaching matters:
Investigating the role of supervision in vision transformers,” in Pro-
ceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2023, pp. 7486–7496.


<!-- page 14 -->
14
IEEE TRANSACTIONS ON MEDICAL IMAGING, VOL. XX, NO. XX, XXXX 2020
[44] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,”
in Advances in Neural Information Processing Systems, vol. 33, 2020,
pp. 6840–6851.
[45] C. Luo, “Understanding diffusion models: A unified perspective,” arXiv
preprint arXiv:2208.11970, 2022.
[46] T. Karras, M. Aittala, T. Aila, and S. Laine, “Elucidating the design
space of diffusion-based generative models,” Advances in Neural Infor-
mation Processing Systems, vol. 35, pp. 26 565–26 577, 2022.
[47] G. W. Brier, “Verification of forecasts expressed in terms of probability,”
Monthly Weather Review, vol. 78, no. 1, pp. 1–3, 1950.
[48] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image
recognition,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2016, pp. 770–778.
[49] O. N. Manzari, H. Ahmadabadi, H. Kashiani, S. B. Shokouhi, and
A. Ayatollahi, “Medvit: a robust vision transformer for generalized
medical image classification,” Computers in Biology and Medicine, vol.
157, p. 106791, 2023.
[50] J. Musaev, A. Anorboev, Y.-S. Seo, N. T. Nguyen, and D. Hwang, “Icnn-
ensemble: An improved convolutional neural network ensemble model
for medical image classification,” IEEE Access, 2023.
[51] J. A. Nelder and R. Mead, “A simplex method for function minimiza-
tion,” The Computer Journal, vol. 7, no. 4, pp. 308–313, 1965.
[52] M. Tan and Q. Le, “Efficientnetv2: Smaller models and faster training,”
in International Conference on Machine Learning.
PMLR, 2021, pp.
10 096–10 106.
[53] H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, and
H. J´egou, “Training data-efficient image transformers & distillation
through attention,” in International Conference on Machine Learning.
PMLR, 2021, pp. 10 347–10 357.
[54] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and
B. Guo, “Swin transformer: Hierarchical vision transformer using shifted
windows,” in Proceedings of the IEEE/CVF International Conference on
Computer Vision, 2021, pp. 10 012–10 022.
[55] S. d’Ascoli, H. Touvron, M. L. Leavitt, A. S. Morcos, G. Biroli,
and L. Sagun, “Convit: Improving vision transformers with soft con-
volutional inductive biases,” in International Conference on Machine
Learning.
PMLR, 2021, pp. 2286–2296.
[56] S. G. Finlayson, J. D. Bowers, J. Ito, J. L. Zittrain, A. L. Beam, and I. S.
Kohane, “Adversarial attacks on medical machine learning,” Science,
vol. 363, no. 6433, pp. 1287–1289, 2019.
[57] S. Kaviani, K. J. Han, and I. Sohn, “Adversarial attacks and defenses
on ai in medical imaging informatics: A survey,” Expert Systems with
Applications, vol. 198, p. 116815, 2022.
[58] G. Bortsova, C. Gonz´alez-Gonzalo, S. C. Wetstein, F. Dubost, I. Ka-
tramados, L. Hogeweg, B. Liefers, B. van Ginneken, J. P. Pluim,
M. Veta et al., “Adversarial attack vulnerability of medical image
analysis systems: Unexplored factors,” Medical Image Analysis, vol. 73,
p. 102141, 2021.
[59] I. J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and harnessing
adversarial examples,” in International Conference on Learning Repre-
sentations, 2015.
[60] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, “Towards
deep learning models resistant to adversarial attacks,” in International
Conference on Learning Representations, 2018.
[61] F. Croce and M. Hein, “Reliable evaluation of adversarial robustness
with an ensemble of diverse parameter-free attacks,” in International
Conference on Machine Learning.
PMLR, 2020, pp. 2206–2216.
[62] M. P. Naeini, G. Cooper, and M. Hauskrecht, “Obtaining well calibrated
probabilities using bayesian binning,” in Proceedings of the AAAI
Conference on Artificial Intelligence, vol. 29, no. 1, 2015.
[63] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration
of modern neural networks,” in International Conference on Machine
Learning.
PMLR, 2017, pp. 1321–1330.
[64] A. Jungo, F. Balsiger, and M. Reyes, “Analyzing the quality and
challenges of uncertainty estimations for brain tumor segmentation,”
Frontiers in Neuroscience, vol. 14, p. 282, 2020.
[65] C. Shui, J. Szeto, R. Mehta, D. L. Arnold, and T. Arbel, “Mitigating
calibration bias without fixed attribute grouping for improved fairness
in medical imaging analysis,” in International Conference on Medical
Image Computing and Computer-Assisted Intervention. Springer, 2023,
pp. 189–198.
[66] R. Mehta, A. Filos, U. Baid, C. Sako, R. McKinley, M. Rebsamen,
K. D¨atwyler, R. Meier, P. Radojewski, G. K. Murugesan et al., “Qu-
brats: Miccai brats 2020 challenge on quantifying uncertainty in brain
tumor segmentation-analysis of ranking scores and benchmarking re-
sults,” The journal of machine learning for biomedical imaging, vol.
2022, 2022.
[67] J. M. Gomes, J. Kong, T. Kurc¸, A. C. Melo, R. Ferreira, J. Saltz, and
G. Teodoro, “Building robust pathology image analyses with uncertainty
quantification,” Computer Methods and Programs in Biomedicine, vol.
208, p. 106291, 2021.
[68] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,”
in International Conference on Learning Representations, 2021.
[69] Y. Song, P. Dhariwal, M. Chen, and I. Sutskever, “Consistency models,”
in International Conference on Machine Learning.
PMLR, 2023, pp.
32 211–32 252.
[70] Y. Gal and Z. Ghahramani, “Dropout as a bayesian approximation:
Representing model uncertainty in deep learning,” in International
Conference on Machine Learning.
PMLR, 2016, pp. 1050–1059.