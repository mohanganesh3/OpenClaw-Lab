<!-- page 1 -->
Proceedings of Machine Learning Research 189, 2022
ACML 2022
Out of Distribution Detection via Neural Network Anchoring
Rushil Anirudh
anirudh1@llnl.gov
Jayaraman J. Thiagarajan
jjayaram@llnl.gov
Center for Applied Scientiﬁc Computing (CASC),
Lawrence Livermore National Laboratory
Editors: Emtiyaz Khan and Mehmet G¨onen
Abstract
Our goal in this paper is to exploit heteroscedastic temperature scaling as a calibration
strategy for out of distribution (OOD) detection. Heteroscedasticity here refers to the fact
that the optimal temperature parameter for each sample can be diﬀerent, as opposed to
conventional approaches that use the same value for the entire distribution. To enable
this, we propose a new training strategy called anchoring that can estimate appropriate
temperature values for each sample, leading to state-of-the-art OOD detection performance
across several benchmarks. Using NTK theory, we show that this temperature function
estimate is closely linked to the epistemic uncertainty of the classiﬁer, which explains its
behavior. In contrast to some of the best-performing OOD detection approaches, our method
does not require exposure to additional outlier datasets, custom calibration objectives, or
model ensembling. Through empirical studies with diﬀerent OOD detection settings – far
OOD, near OOD, and semantically coherent OOD - we establish a highly eﬀective OOD
detection approach. Code to reproduce our results is available at github.com/LLNL/AMP
Keywords: OOD Detection, Temperature Scaling, Calibration, Anchoring, Uncertainty
1. Introduction
The task of using a trained model to accurately distinguish between samples from the dataset
used for training – i.e., the in-distribution (ID), and any other external dataset with diﬀerent
semantic characteristics is broadly referred to as OOD (out-of-distribution) detection. To
solve this challenging problem, one needs to obtain an eﬀective characterization of the ID
data manifold, such that the discrepancy between test data and the inferred manifold can
be used to recognize the model’s lack of knowledge about OOD data. This is commonly
achieved by learning a scoring function: S : X →R that can score both ID and OOD
samples appropriately. A simple scoring function can be based on the maximum softmax
probability (MSP) of a prediction, with the expectation that the model will be more
conﬁdent about an ID sample compared to OOD samples. However, in practice, such simple
prediction conﬁdence scores are poorly calibrated, and as a result, several novel scoring
functions have emerged – predictive entropy (Guo et al., 2017), energy (Liu et al., 2020),
uncertainty estimates (Gal and Ghahramani, 2016; Lakshminarayanan et al., 2017), latent
space deviation (Van Amersfoort et al., 2020), class-speciﬁc deviations (Lee et al., 2018b;
Sastry and Oore, 2020) etc. Though these scoring functions often perform better than
MSP, many state-of-the-art formulations (Hendrycks et al., 2019; Liu et al., 2020; Yang
et al., 2021) rely on additional unlabeled data for calibrating model predictions to better
© 2022 R. Anirudh & J.J. Thiagarajan.
arXiv:2207.04125v2  [cs.LG]  1 Dec 2022


<!-- page 2 -->
Anirudh Thiagarajan
reject OOD data. In addition to requiring sophisticated training strategies (e.g., outlier
exposure), this approach can be sub-optimal when the calibration dataset is not strictly
OOD, i.e., and they contain shared semantics with the ID set (Yang et al., 2021). Further,
the calibration strategy used in many of these methods relies on temperature scaling (Guo
et al., 2017), which essentially scales the logits by a scalar called the temperature. When the
temperature parameter is greater (or lower) than 1, the entropy of the resulting prediction
distribution increases (or decreases). Consequently, with an appropriate temperature value
(chosen with either external or additional validation data), even this simple scaling leads to
much improved OOD detection performance.
Heteroscedastic temperature scaling with anchoring.
In this paper, we explore the
idea of heteroscedastic temperature scaling, i.e., instead of using the same temperature
scalar for all the samples, we construct a temperature function that produces sample-speciﬁc
temperature values. Our hypothesis is that by appropriately tempering the predictions
for ID and OOD samples, any existing scoring function can eﬀectively between distinguish
them. We achieve this using a novel training procedure called neural network anchoring.
In a nutshell, anchoring involves ﬁrst transforming the input image, x, into a tuple using
the transformation E : x →[c, x −c], where c is another randomly chosen image (“anchor”)
from the training set, and predicting the label for x using this tuple. We also propose an
additional consistency training strategy by perturbing the anchor before encoding, which
boosts the performance further. During inference, we obtain predictions from multiple
random anchors and propose to estimate the temperature based on standard deviation of
these predictions. Using neural tangent kernel theory (Jacot et al., 2018), we show that our
heteroscedastic temperature estimate is closely related to the epistemic uncertainty of the
model.
We use this temperature estimate to calibrate the predictions for a test sample, using
which we can compute an OOD score using existing scoring functions (e.g., entropy). See
Fig. 1(A) for an illustration of the process, and Figs. 2 and 3 for the pseudo-codes. Fig. 1(B)
illustrates the improvement over conventional temperature scaling using the standard CIFAR-
10/SVHN OOD benchmark. Through extensive empirical analysis, we demonstrate that
the proposed approach produces state-of-the-art detection performance across multiple
benchmarks and models (summarized in Table 1).
Base Model
Benchmark (IN →OOD)
Year
Reference
WRN-40-2
CIFAR-10/100 →6 Datasets
(Liang et al., ICLR’17)
Table 2
ResNet-34
CIFAR-10/100 →7 Datasets
(Sastry and Oore, ICML’20)
Table 3
ResNet-34
CIFAR-10 ↔CIFAR-100
(Near OOD)
Table 4
ResNet-50
ImageNet-1K →ImageNet-C (Krishnan and Tickoo, NeurIPS’20)
Table 5, Figure 5
ResNet-18
Semantically Coherent OOD
(Yang et al., ICCV’21)
Table 6
ResNet-34
Robustness to resizing artifacts
(this paper)
Table 7
WRN-40-2
Ablation study
Table 8
Table 1: Summary of experiments in this paper.


**[Table p2.1]**
| WRN-40-2 ResNet-34 ResNet-34 ResNet-50 ResNet-18 ResNet-34 WRN-40-2 | CIFAR-10/100 → 6 Datasets (Liang et al., ICLR’17) CIFAR-10/100 → 7 Datasets (Sastry and Oore, ICML’20) CIFAR-10 ↔ CIFAR-100 (Near OOD) ImageNet-1K → ImageNet-C (Krishnan and Tickoo, NeurIPS’20) Semantically Coherent OOD (Yang et al., ICCV’21) Robustness to resizing artifacts (this paper) Ablation study | Table 2 Table 3 Table 4 Table 5, Figure 5 Table 6 Table 7 Table 8 |
| --- | --- | --- |

[CAPTION] Table 2
ResNet-34

[CAPTION] Table 3
ResNet-34

[CAPTION] Table 4
ResNet-50

[CAPTION] Table 6
ResNet-34

[CAPTION] Table 7
WRN-40-2

[CAPTION] Table 8
Table 1: Summary of experiments in this paper.


<!-- page 3 -->
Out of Distribution Detection via Neural Network Anchoring
TNR@FPR95 (↑): 71.17
AUROC (↑): 91.94 
TNR@FPR95 : 81.10
AUROC : 96.50
TNR@FPR95 : 88.99
AUROC : 98.03 
Homoscedastic Temp. Scaling
NN Anchoring
NN Anchoring +Consistency
misclassification
Heteroscedastic Temp. Scaling (Proposed)
0
0
0
0
0
0
1
0
0
0
0
0
0
0 .1 0 .9 0
0
0
0 .1 .1 0
0
0 .8 0
0
0
𝐶1
𝑿−𝑅!
𝐶3
𝑿
Ground Truth 𝐗→𝑌
𝐶!, 𝑿−𝐶! →(𝑌!
.1 0 .1 0
0
0 .8 0
0
0
𝐶", 𝑿−𝐶" →(𝑌"
𝐶#, 𝑿−𝐶# →(𝑌#
Training Manifold
𝐶$
𝑿−𝐶"
𝑿−𝐶#
Anchored
Predictions
Anchors are drawn at 
random from
The mean of the predictions is scaled by its standard 
deviation, that acts as a heteroscedastic temp. scaling.
Test 
sample
𝑃(𝑿)
B
A
Figure 1: Improving existing OOD detectors via heteroscedastic temperature scaling.
(A) We propose a new training procedure called neural network anchoring to estimate
the temperature parameter for any test sample, and show that it can be leveraged to
improve conventional OOD detectors (e.g., entropy-based). (B) We also introduce a new
consistency training objective to further improve the ﬁdelity of detectors.
2. Background and Related Work
We use training data D = {(xi, yi)}n
i=1, where xi ∈PI and yi ∈CI := {1, 2, · · · , Nclass}, to
train a model f(θ) ∈H with randomly initialized weights θ0 and hypothesis space H.We
train a classiﬁer f(θ) : x →y, parameterized by θ using D. While the learned model is
required to generalize to the test dataset Dt = {(xt
i, yt
i)}Nt
i=1 when xt
i ∈PI and yt
i ∈CI, it is
also critical to recognize out-of-distribution samples, i.e., xt
i ∈PO and yt
i ∈CO, on which
the model could fail. Without loss of generality, this formulation encompasses scenarios
with unknown distribution shifts to the input images, i.e., PO ̸= PI, CO ⊆CI as well as the
presence of additional, unknown classes CO ⊃CI.
Scoring functions
In summary, the goal of OOD detection is to use labeled data D to
train a classiﬁer that has the capacity to reject samples from PO, while also accurately
classifying samples from PI. This is typically achieved by deﬁning a scalar scoring function,
such that Sx∈PI ̸= Sx∈PO, i.e., they are suﬃciently distinct that an out of distribution sample
is easily classiﬁed. Choosing the appropriate scoring function has been focus of the last
several years of research, with most techniques choosing the score as some function of the
prediction logits obtained from the trained classiﬁer – such as maximum softmax probability,
entropy (Guo et al., 2017), and energy (Liu et al., 2020).
Uncertainties for OOD detection
Uncertainty estimation has been a popular choice
for OOD detection since epistemic (or model) uncertainties are supposed to be indicative of
the OOD-ness of a test sample. For example, DUQ (Van Amersfoort et al., 2020) uses a


**[Table p3.1]**
| 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |


**[Table p3.2]**
| 0 | 0 | 0 | 0 | .1 | 0 | .9 | 0 | 0 | 0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |


**[Table p3.3]**
| .1 | 0 | .1 | 0 | 0 | 0 | .8 | 0 | 0 | 0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |


**[Table p3.4]**
| 0 | .1 | .1 | 0 | 0 | 0 | .8 | 0 | 0 | 0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

[CAPTION] Figure 1: Improving existing OOD detectors via heteroscedastic temperature scaling.


<!-- page 4 -->
Anirudh Thiagarajan
kernel distance to a set of class-speciﬁc centroids deﬁned in the feature of a deep network as
the measure for uncertainty. Another recent technique is DEUP (Jain et al., 2021) which
trains an explicit epistemic uncertainty estimator for a pre-trained model. More generally,
Bayesian methods (Neal, 2012) are among the most common kinds of uncertainty estimators
today, but they are not easily scalable to large datasets and are known to be outperformed
by model ensembling (Lakshminarayanan et al., 2017). Monte Carlo Dropout (Gal and
Ghahramani, 2016) is a scalable alternative to Bayesian methods in that it approximates
the posterior distribution on the weights via dropout to estimate uncertainties.
Prediction calibration for OOD detection
In general uncertainty-based OOD detec-
tors have so far not been able to improve performance over more traditional scoring based
methods. As a result, a lot of focus has been on calibrating classiﬁer predictions to make the
scoring function more eﬀective (often relying on outlier examples) such as Mahalanobis (Lee
et al., 2018a), ODIN (Liang et al., 2017), Gram matrices (Sastry and Oore, 2020), AVuC
(Krishnan and Tickoo, 2020), and outlier exposure (Hendrycks et al., 2019). The simplest,
and often most eﬀective, strategy for calibration is temperature scaling (Guo et al., 2017)
where the logits are scaled by a temperature parameter (τ). When τ > 1 the prediction
is made less conﬁdent (i.e., higher entropy), and if not it is made more peaky (i.e., lower
entropy). Notably, most techniques deﬁne a single τ value to calibrate predictions on an
entire test set.
The underlying assumption made by such approaches is that the predictions are ho-
moscedastic – i.e., the errors (and uncertainties) are uniform everywhere and therefore a
single scalar for all samples suﬃces. However, in practice, most models are heteroscedastic –
i.e., the errors around a prediction can vary for diﬀerent test samples. In this paper, we
propose a novel calibration strategy that estimates a speciﬁc temperature for every sample
related to the unreliability or uncertainty for that sample. Unlike existing approaches, this
takes the heteroscedasticity of the model into account and therefore, subsequently improves
OOD detection.
3. Heteroscedastic Temperature Scaling via Neural Network Anchoring
In this section, we ﬁrst outline the proposed scoring function, followed by theoretical and
empirical justiﬁcation of its eﬀectiveness in visual OOD detection. As previously stated, we
are interested in learning a temperature function τ : X →R, deﬁned in the image domain
X, that is used to calibrate a model’s predictions. The temperature parameter for a sample
x is denoted as τ(x) (ﬁxed to be the same for all classes). We refer to this process as
heteroscedastic scaling, since the model predictions for any test sample can be adjusted with
the learned function. Next, we discuss the proposed approach for constructing τ.
3.1. Neural network anchoring
First, let us randomly choose a training image from the dataset D, denoted by c and refer
to it as an anchor. Using this anchor c, we deﬁne a simple coordinate transformation on
the input domain as E : x →[c, x −c]. That is, we represent an image as a combination of
the anchor, and the residual between the anchor and the image. Note that, this deﬁnition
allows the use of multiple transformations (w.r.t. many anchors) to obtain predictions for


<!-- page 5 -->
Out of Distribution Detection via Neural Network Anchoring
a given sample x, i.e., fA([c1, x −c1]) = fA([c2, x −c2]) = · · · = fA([ck, x −ck]), where fA
refers to the model that takes the tuple ([ck, x −ck]) and predicts the target y.
Training.
During training, for every input sample xi, we use one random anchor to perform
the coordinate transformation, which is implemented as a simple concatenation along the
channel dimension. Due to the randomness over the choice of the anchor, over the course
of training each xi gets combined with a large number of anchors. Since the prediction
for the sample xi – regardless of the choice c – is expected to be the same (label yi), this
enforces an implicit consistency in the predictions across diﬀerent anchors. The optimization
of this anchored model is similar to that of a standard network, e.g., cross entropy loss-based
training.
Consistency via standard image augmentations.
Inspired by the recent successes of
data augmentation strategies in improving model generalization, we exploit an additional
consistency during training to further improve anchored models. More speciﬁcally, we modify
the input to be as follows – ¯x = [T (c), x −c], where T (.) refers to a pre-deﬁned image
augmentation and it returns a perturbed anchor. Intuitively, we encourage the model to
learn invariances between c and T (c) using an asymmetry in the coordinate transformation.
While diﬀerent choices currently exist to implement T for natural images, we ﬁnd that using
a composition of multiple standard augmentation strategies already used in training the
model (random crops, random ﬂips, color jitter etc.) are suﬃcient in practice.
Figure 2: Pseudocode for training
def train loop(trainloader ,T):
for inputs, targets in trainloader:
A = Shuffle(inputs)
D = inputs−A
X d = torch.cat([T(A), D],axis=1)
y d = model(X d)
loss = criterion(y d ,targets)
....
return
Figure 3: Pseudocode for inference
def inference(inputs ,anchors):
for A in anchors:
D = inputs−A
X test = torch.cat([A, D],axis=1)
y test = model(X test)
preds.append(y test)
P = torch.cat(preds ,0)
H = torch.mean(P,0)
tau = P.sigmoid().std(0).sum(1)
ood score = AMP(H/tau)
return H, ood score
Inference.
As discussed in the section, this anchoring process leads to diﬀerent hypotheses
for diﬀerent anchor choices, so we propose to marginalize out the eﬀect of the anchor,
c, to both obtain the predictions as well as the temperature function for performing the
heteroscedastic calibration. The temperature value for a sample is estimated as the standard
deviation of the predictions, obtained using multiple random anchor choices, as shown below:
H(y|x) = MEAN [fA([ck, x −ck])]K
k=1 ; τ(x) =
X
all classes
STD-DEV
 
σ
 
fA([ck, x −ck])
  K
k=1
.
(1)

[CAPTION] Figure 2: Pseudocode for training

[CAPTION] Figure 3: Pseudocode for inference


<!-- page 6 -->
Anirudh Thiagarajan
Here, for convenience we use MEAN and STD-DEV to denote the mean and standard deviation
over predictions obtained using K diﬀerent anchors during inference.
Further, H(y|x)
indicates the set of logits for each of the classes. To scale the standard deviation appropriately,
we compute it after passing the logits through a sigmoid activation layer. Note that, we
compute the class-speciﬁc uncertainties by performing this aggregation directly using the
logits obtained from the model. Essentially, we marginalize out the eﬀect of the randomly
chosen anchor to obtain the ﬁnal prediction and the temperature value for a test sample.
Heteroscedastic temperature scaling.
Having estimated the temperature for the sam-
ple in (1), the heteroscedastic calibration process can be expressed as Hc(y|x) = H(y|x)
τ(x) .
The OOD score for this calibrated sample is given by using any pre-speciﬁed scoring
function S.
In particular, we ﬁrst obtain class likelihoods from the calibrated logits
P c(y|x) = SOFTMAX(Hc(y|x)), and compute the negative log likelihood score for OOD detec-
tion. We refer to this score as Anchor Marginalized Prediction score (AMP). Formally,
the AMP scoring function can be deﬁned as:
AMP :
S(x) = −1
N
X
all classes
log(SOFTMAX(Hc(y|x))).
(2)
For smaller datasets like CIFAR-10/100 the variance tends to be small, and to avoid scaling
issues we instead use Hc(y|x) =
H(y|x)
1+exp(τ(x)). In Figures 2 and 3 we demonstrate pytorch-like
pseudocode for training and inference.
4. Intuition Behind AMP
AMP is eﬀective in distinguishing OOD samples due to the proposed heteroscedastic
temperature scaling strategy – by deﬁning the temperature value as a function of the sample,
we are able to automatically adjust the scaling to be sensitive to the OOD-ness of the sample.
Naturally, this works better than using a single temperature value for the entire dataset. A
strong candidate for such a scaling strategy is the epistemic uncertainty of a model for a
given sample. Intuitively, scaling by uncertainty should improve OOD performance since,
by deﬁnition, an OOD sample has high epistemic uncertainty, compared to an inlier, which
translates to a higher temperature that scales the logits more aggressively. This leads to
increased entropy in the resulting prediction probabilities. During inference, AMP estimates
the temperature of a sample based on the standard deviation of predictions obtained via
multiple anchors. In the following section, we justify why this estimator expressed in (1) is,
in fact, related to the epistemic uncertainty.
We utilize neural tangent kernel (NTK) theory (Jacot et al., 2018; Arora et al., 2019;
Bietti and Mairal, 2019; Lee et al., 2019), as it provides a convenient framework for analyzing
the eﬀect of the modiﬁed training proposed in AMP . The basic idea of NTK is that, when
the width of a neural network tends to inﬁnity and the learning rate of SGD tends to zero,
the function f(x; θ) converges to a solution obtained by kernel regression using the NTK:
Kxixj = Eθ
 ∂f(xi, θ)
∂θ
, ∂f(xj, θ)
∂θ
 
.
(3)


<!-- page 7 -->
Out of Distribution Detection via Neural Network Anchoring
When the samples xi, xj ∈Sd−1, i.e., points on the hypersphere and have unit norm, the
NTK for a simple 2 layer ReLU MLP can be simpliﬁed as a dot product kernel (Arora et al.,
2019; Bietti and Mairal, 2019; Lee et al., 2019):
Kxixj = hNTK(x⊤
i xj) = 1
2πx⊤
i xj(π −cos−1(x⊤
i xj)).
(4)
Let us also consider the prediction on a test sample xt in the limit as the inner layer widths
grow to inﬁnity. It has been shown that (c.f. (Lee et al., 2019; Bietti and Mairal, 2019)):
f∞(xt) = f0(xt) −KxtXK−1
XX(f0(X) −Y),
(5)
where f0 is the network with the initial random weights, θ0, and X is the matrix of all
training data samples.
NTK with neural network anchoring.
Recall, the anchoring process involves trans-
forming the input into a tuple: [c, x −c], for a randomly chosen anchor c. In the following,
we examine the impact of anchoring on the NTK. Without loss of generality, we assume
[c, xi −c] and [c, xj −c] are unit norm, so we can simplify hNTK([c, xi −c]⊤[c, xj −c]). Further,
we use a Taylor series approximation for cos−1: cos−1(u −c) ≈cos−1(u) +
c
√
1−(u−c)2 .
Each anchor results in a different decision boundary
NTK Spectrum of an MLP (width = 512, depth = 8)
NTK Spectrum of the same MLP with anchoring
Figure 4: NTK Spectra for a vanilla model and one with neural network anchoring (left). In the
simple classiﬁcation setup on the right, we demonstrate the eﬀect of diﬀerent anchors
on the classiﬁer’s predictions. We observe that each anchor produces a slightly diﬀerent
NTK (due to (6)), resulting in meaningful inconsistencies in the classiﬁer’s predictions.

[CAPTION] Figure 4: NTK Spectra for a vanilla model and one with neural network anchoring (left). In the


<!-- page 8 -->
Anirudh Thiagarajan
Expanding [c, xi −c]⊤[c, xj −c] = x⊤
i xj −c⊤(xi + xj −2c) and letting v = (xi + xj −2c),
we obtain the expression for hNTK under a shifted domain (from (4)) as follows:
Kanc = 1
2π(x⊤
i xj −c⊤v)(π −cos−1(x⊤
i xj −c⊤v))
≈1
2πx⊤
i xj(π −cos−1(x⊤
i xj)) −1
2πc⊤v(π −cos−1(x⊤
i xj)) −
c(x⊤
i xj −c⊤v)
2π
q
1 −(x⊤
i xj −c⊤v)2
= Kxixj −Γxi,xj,c,
(6)
where we combine all terms dependent on c into Γxi,xj,c, which also behaves as a dot product
kernel. We can see that combining this stochastic NTK (in c) into (5), the prediction on a test
sample is correspondingly stochastic under a ﬁxed initialization θ0. In other words, neural
network anchoring eﬀectively perturbs the hypothesis space of the neural network resulting
in an ensembling-like behavior, that eﬀectively produces diﬀerent predictions, conditioned
on the anchor choice. We illustrate the process in Figure 1, and compute the NTK spectrum
(as done in Tancik et al.) along with a demonstrative toy classiﬁcation problem in Fig. 4.
Our training objective (with randomly chosen anchors) forces the predictions on a sample
across anchors to be the same for in distribution samples, and therefore the disagreement
(measured by sum of standard deviation, (1)) is larger when the sample is OOD. This
is similar, in principle, to deep ensembles (Ovadia et al., 2019), that measure prediction
variance by training multiple models with diﬀerent random initializations θ0.
5. Experiments and Results
In this section we evaluate the proposed anchoring-based OOD detection on a wide variety
of OOD benchmarks, with several diﬀerent model architectures, and OOD settings.
Setup.
Our modiﬁcation to any standard neural network architecture is minimal – we
only change the ﬁrst convolutional layer to accept a 6 channel input (3 channels for anchor
and residual each) instead of the original 3 channels. We use standard hyper-parameters
to train all our models. For our experiments with CIFAR10/100 (Krizhevsky et al., 2009)
datasets, we trained a wide ResNet (WRN) model for 200 epochs, with an initial learning
rate of 0.1, and a decay of 0.2 at 60, 120, and 160 epochs. In addition to WRN, we also
experimented with ResNet-18 and ResNet-34 architectures. For our ImageNet (Russakovsky
et al., 2015) experiments we trained a ResNet-50 model with an initial learning rate of 0.1
and a decay of 0.1 every 30 epochs. Note that, we used the same normalization scheme
for pre-processing both in- and out- distribution data sets (at train and test times). As
discussed in the previous section, for our method, we used a single randomly chosen anchor
for every input image in a mini-batch. From our extensive empirical studies, we observed no
signiﬁcant diﬀerence in the top-1 accuracy of the anchored models on any of the benchmarks.
For example, with the WRN-40-2 on CIFAR-10, we obtained an accuracy of 95.1 on average,
and on CIFAR-100 a top-1 accuracy of 76.1. On ImageNet, our ResNet-50 model trained for
120 epochs resulted in a top-1 accuracy of 76.0. More details of the benchmarks and the
exact hyper-parameter settings are provided in the supplement.


<!-- page 9 -->
Out of Distribution Detection via Neural Network Anchoring
Baselines.
We performed comparisons with several widely-adopted OOD detection ap-
proaches: (a) Maximum Softmax Probability (MSP); (b) ODIN (Liang et al., 2017); (c)
Energy (Liu et al., 2020); (d) Gram Matrices GM (Sastry and Oore, 2020) that uses latent
space deviation to detect OOD, but has been shown to perform better than scoring functions
such as Mahalanobis detection (Lee et al., 2018b) without requiring exposure to outlier
data. We also evaluated our approach against uncertainty-based OOD detectors for the
ImageNet experiments, namely MC-dropout (Gal and Ghahramani, 2016), deep ensembles
(Lakshminarayanan et al., 2017) and techniques that require post-hoc calibration on a
validation set like temperature scaling (Guo et al., 2017), SVI (Blundell et al., 2015), and
AVuC (Krishnan and Tickoo, 2020).
Metrics: We used 4 standard metrics to evaluate OOD detection performance across our
benchmarks – (a) FPR95: False positive rate of examples from the OOD set when the
true positive rate (TPR) of the in-distribution is set at 95%; (b) AUROC: Area under the
receiver operating characteristic curve; (3) AUPR: Area under the precision-recall curve for
both In/Out sets depending on which one is considered positive; (4) DTACC: Detection
accuracy measures the maximum possible OOD detection accuracy across all thresholds as
proposed in (Lee et al., 2018b).
OOD
FPR95 ↓
AUROC ↑
AUPR ↑
MSP / ODIN / Energy / AMP (ours)
CIFAR-10
iSUN
56.03 / 32.05 / 33.68 / 16.59
89.83 / 93.50 / 92.62 / 97.16
97.74 / 98.54 / 98.27 / 97.83
LSUN (R)
52.15 / 26.62 / 27.58 / 13.73
91.37 / 94.57 / 94.24 / 97.77
98.12 / 98.77 / 98.67 / 97.77
LSUN (C)
30.80 / 15.52 /
8.26 /
1.50
95.65 / 97.04 / 98.35 / 99.55
99.13 / 99.33 / 99.66 / 99.57
Places365
59.48 / 57.40 / 40.14 / 19.89
88.20 / 84.49 / 89.89 / 95.79
97.10 / 95.82 / 97.30 / 87.28
Texture
59.28 / 49.12 / 52.79 / 35.43
88.50 / 84.97 / 85.22 / 93.61
97.16 / 95.28 / 95.41 / 96.14
SVHN
48.49 / 33.55 / 35.59 /
5.19
91.89 / 91.96 / 90.96 / 98.10
98.27 / 98.00 / 97.64 / 97.51
Average
42.71 / 35.71 / 33.01 / 15.39
90.91 / 91.01 / 91.88 / 96.99
97.91 / 97.62 / 97.82 / 96.02
CIFAR-100
iSUN
82.80 / 68.51 / 81.10 / 67.15
75.46 / 82.69 / 78.91 / 83.79
94.06 / 95.80 /94.91 / 85.84
LSUN (R)
82.42 / 71.96 / 79.47 / 61.73
75.38 / 81.82 / 79.23 / 85.64
94.06 / 95.65 / 94.96 / 84.30
LSUN (C)
66.54 / 55.55 / 35.32 /
4.16
83.79 / 87.73 / 93.53 / 99.18
96.35 / 97.22 / 98.62 / 99.19
Places365
82.84 / 87.88 / 80.56 / 65.18
73.78 / 71.63 / 75.44 / 85.78
93.29 / 92.56 / 93.45 / 69.65
Texture
83.29 / 79.27 / 79.41 / 81.81
73.34 / 73.45 / 76.28 / 71.36
92.89 / 92.75 / 93.63 / 79.72
SVHN
84.59 / 84.66 / 85.82 / 12.57
71.44 / 67.26 / 73.99 / 97.85
92.93 / 91.38 / 93.65 / 95.25
Average
80.41 / 74.64 / 73.61 / 48.77
75.53 / 77.43 / 79.56 / 87.27
93.93 / 94.23 / 94.87 / 85.66
Table 2: OOD Detection with WideResNet-40-2: We compare with a range of diﬀerent
OOD detection methods on the commonly used OOD benchmark. We see a consistent
improvement in performance when using AMP , over existing baselines – across both
CIFAR-10/100 models. AMP is particularly good in suppressing false positives as reﬂected
by the FPR95 metric.
5.1. Performance on OOD Benchmarks
We begin by evaluating AMP on a commonly used OOD detection benchmark ﬁrst introduced
by Liang et al. (Liang et al., 2017). In this experiment, predictive models were trained with
CIFAR-10 or CIFAR-100 as the in-distribution, and then used to detect out-of-distribution
data chosen from one of the six datasets (details in the supplement).
We follow the

[CAPTION] Table 2: OOD Detection with WideResNet-40-2: We compare with a range of diﬀerent


<!-- page 10 -->
Anirudh Thiagarajan
Method
FPR95 ↓
AUROC ↑
DTACC ↑
CIFAR-10
MSP
55.12
90.37
84.65
ODIN
33.40
92.70
85.28
Energy
28.89
94.47
88.58
Mahal.*
14.8
97.33
93.15
GM
14.83
96.33
92.14
AMP (ours)
12.33
97.20
92.84
CIFAR-100
MSP
80.22
77.22
70.95
ODIN
60.82
85.41
78.61
Energy
70.96
82.44
75.67
Mahal.*
24.36
94.07
88.51
GM
49.87
89.96
83.50
AMP (ours)
49.70
89.97
82.50
Table 3: OOD detection performance with ResNet-
34:. Averaged across 7 datasets following Sastry
and Oore (2020). Here, ‘Mahal.’ indicates the
Mahalanobis score, which uses additional outlier
data during training.
Method
ResNet-34
FPR95 ↓/ AUROC ↑
C10 →C100
ODIN
58.0 / 88.2
Energy
47.5 / 88.4
GM
59.8 / 83.6
Mahal.∗
58.4 / 88.2
AMP (ours)
43.5 / 90.2
C100 →C10
ODIN
81.3 / 77.2
Energy
80.9 / 77.0
GM
83.1 / 74.5
Mahal.∗
79.8 / 77.5
AMP
82.5 / 79.9
Table 4: AMP
is
eﬀective
on
near
OOD detection with CIFAR-
10↔100 respectively.
experimental protocol from (Liu et al., 2020), and used the WideResNet architecture WRN-
40-2 (Zagoruyko and Komodakis, 2016). In Table 2, we show results across all the six datasets
for both CIFAR-10/100, using the three metrics. On average, we ﬁnd that AMP signiﬁcantly
improves on the challenging FPR95 metric (nearly 50% lower than the next best), while
also providing substantial gains in the AUROC metric. We notice that, these gains persist
even with CIFAR-100, which is known to be a much harder setting (as reﬂected by the
higher false positive rates). We do not assume access to additional OOD data (for model
ﬁnetuning), unlike both energy- and ODIN-based detection that perform similarly, while
our approach signiﬁcantly improves upon them (≈25% drop in the FPR metric).
Next, in Table 3, we evaluate on the benchmark introduced in (Sastry and Oore, 2020)
using ResNet-34. This consists of 7 diﬀerent OOD datasets – iSUN, LSUN (R), LSUN (C),
TinyImageNet (R), TinyImageNet (C), CIFAR10/100, SVHN. We observe that AMP comes
second to Mahalanobis on CIFAR-100 while matching it on CIFAR-10, even though we
do not use outlier data for ﬁne-tuning. In order to make a fair comparison on the same
OOD samples for all datasets, we use GM without the validation data (and corresponding
normalization) – the performance improvements were consistently observed in most cases.
Detailed results for this benchmark are available in the supplement.
Uncertainty based ImageNet-OOD detection. The prediction variance from AMP can
be interpreted as a measure of unreliability in the model’s predictions. Hence, a natural
evaluation is to leverage this unreliability estimate as a score for OOD detection, since it is
expected to be high as we move away from the original data manifold (i.e., OOD), while being
low for in-distribution data. We analyzed this performance on ImageNet-C data (Hendrycks
and Dietterich, 2019) using a ResNet-50 architecture trained on ImageNet-1K, as shown
in Table 5. Speciﬁcally, we used the OOD dataset obtained via Gaussian blur corruption
at intensity 5. We ﬁnd that, by leveraging our unreliability score to perform adaptive
temperature scaling, AMP is highly eﬀective for OOD detection. In fact, this improves over
several state-of-the-art uncertainty estimators on this benchmark.


**[Table p10.1]**
|  | ResNet-34 Method FPR95 ↓/ AUROC ↑ |
| --- | --- |
| ODIN 58.0 / 88.2 →C100 Energy 47.5 / 88.4 GM 59.8 / 83.6 Mahal.∗ 58.4 / 88.2 C10 AMP (ours) 43.5 / 90.2 ODIN 81.3 / 77.2 →C10 Energy 80.9 / 77.0 GM 83.1 / 74.5 C100 Mahal.∗ 79.8 / 77.5 AMP 82.5 / 79.9 | ODIN 58.0 / 88.2 Energy 47.5 / 88.4 GM 59.8 / 83.6 Mahal.∗ 58.4 / 88.2 AMP (ours) 43.5 / 90.2 |
|  | ODIN 81.3 / 77.2 Energy 80.9 / 77.0 GM 83.1 / 74.5 Mahal.∗ 79.8 / 77.5 AMP 82.5 / 79.9 |

[CAPTION] Table 3: OOD detection performance with ResNet-

[CAPTION] Table 4: AMP


<!-- page 11 -->
Out of Distribution Detection via Neural Network Anchoring
Method
AUROC ↑DTACC ↑
AUPR-in/out ↑
ResNet-50 (He et al., 2016)
93.36
86.08
92.82 / 93.71
Temp-Scal (Guo et al., 2017)
93.71
86.47
93.21 / 94.01
Deep Ens (Lakshminarayanan et al., 2017)
95.49
88.82
95.31 / 95.64
MCD (Gal and Ghahramani, 2016)
96.38
89.98
96.16 / 96.67
SVI (Blundell et al., 2015)
96.40
90.03
95.97 / 96.83
SVI-AvUC (Krishnan and Tickoo, 2020)
97.60
92.07
97.39 / 97.85
AMP (ours)
99.07
95.14
99.10 / 99.08
Table 5: UQ-based detection of ImageNet-C with ResNet-50: We evaluate AMP on the
benchmark introduced by (Krishnan and Tickoo, 2020) where we use Gaussian Blur of level
5 intensity as the OOD, and the clean ImageNet validation as the ID. For comparisons
with OOD methods on this benchmark, refer to Figure 5(b).
(a) Densities of scores between in
and out distribution ImageNet-
C.
(b) AUROC (↑) across corruption levels for AMP
Figure 5: From near to far OOD: AMP outperforms baselines consistently at all the intensity
levels (Fig. 5(b)), and produces meaningful scores (Fig. 5(a)) when moving from near
(level 1) to far (level 5) OOD sets.
Near OOD detection. Here, we consider the more challenging “near OOD” detection
task that seeks to separate CIFAR-10 and CIFAR-100 datasets, by using a model trained on
one of them while treating the other as the OOD set. Since both are drawn from the same
image distribution but have mutually exclusive classes, this is extremely challenging and
often causes OOD detectors to fail. We measure OOD performance using both FPR and
AUROC as shown in Table 4. We see that methods like Mahalanobis detector (Lee et al.,
2018b), Gram Matrices (Sastry and Oore, 2020), which are otherwise very competitive tends
to fail in this task – whereas AMP is able to separate them better, indicating its eﬀectiveness
for both near and far OOD tasks.
We also study how AMP performs as the OOD set is artiﬁcially made to go farther from the
ImageNet validation set using the Gaussian blur corruption of varying intensity. We expect


**[Table p11.1]**
| Method | AUROC ↑ DTACC ↑ AUPR-in/out ↑ |
| --- | --- |
| ResNet-50 (He et al., 2016) Temp-Scal (Guo et al., 2017) Deep Ens (Lakshminarayanan et al., 2017) MCD (Gal and Ghahramani, 2016) SVI (Blundell et al., 2015) SVI-AvUC (Krishnan and Tickoo, 2020) | 93.36 86.08 92.82 / 93.71 93.71 86.47 93.21 / 94.01 95.49 88.82 95.31 / 95.64 96.38 89.98 96.16 / 96.67 96.40 90.03 95.97 / 96.83 97.60 92.07 97.39 / 97.85 |
| AMP (ours) | 99.07 95.14 99.10 / 99.08 |

[CAPTION] Table 5: UQ-based detection of ImageNet-C with ResNet-50: We evaluate AMP on the

[CAPTION] Figure 5: From near to far OOD: AMP outperforms baselines consistently at all the intensity


<!-- page 12 -->
Anirudh Thiagarajan
that an eﬀective scoring mechanism must reﬂect this well in its scores. We demonstrate
this in Fig. 5(a) that shows the kernel density plot of AMP scores for both in- and out-
distribution datasets across 5 intensity levels. Since the score is designed so as to be higher
for the in-distribution set, we ﬁnd a gradual and meaningful transition between the clean
in-distribution set to the farthest OOD set (tight in the low conﬁdence area). This is reﬂected
in the OOD detection performance as measured by AUROC shown in Fig. 5(b), where all
three variants of scoring functions with AMP outperform competing approaches consistently
across diﬀerent corruption levels.
In-distribution
Method
Needs OOD
Exposure?
FPR95 ↓
AUROC ↑
AUPR(In/Out) ↑
CIFAR-10
(ResNet-18)
ODIN (Liang et al., 2017)
 
52.00
82.00
73.13 / 85.12
Energy (Liu et al., 2020)
 
50.03
83.83
77.15 / 85.11
OE (Hendrycks et al., 2019)
 
50.53
88.93
87.55 / 87.83
MCD (Yu and Aizawa, 2019)
 
73.02
83.89
83.39 / 80.53
UDG (Yang et al., 2021)
 
36.22
93.78
93.61 / 92.61
AMP
 
36.82
92.40
91.23 / 90.91
CIFAR-100
(ResNet-18)
ODIN (Liang et al., 2017)
 
81.89
77.98
78.54 / 72.56
Energy (Liu et al., 2020)
 
81.66
79.31
80.54 / 72.82
OE (Hendrycks et al., 2019)
 
80.06
78.46
80.22 / 71.83
MCD (Yu and Aizawa, 2019)
 
85.14
74.82
75.93 / 69.14
UDG (Yang et al., 2021)
 
75.45
79.63
80.69 / 74.10
AMP
 
70.34
82.22
84.14 / 76.20
Table 6: SCOOD benchmark with ResNet-18. AMP outperforms even the best performing
techniques on the recent semantically coherent OOD bencmark (Yang et al., 2021) on
both CIFAR-10 and CIFAR-100 inspite of not requiring outlier exposure. The methods
OE (Hendrycks et al., 2019), MCD (Yu and Aizawa, 2019), and UDG use Tiny-ImageNet
during training.
5.2. Semantically Coherent OOD (SCOOD)
The current suite of OOD detection benchmarks rely on separating the dataset on which
a model is trained from another dataset marked as OOD, for example, CIFAR-SVHN.
However, considering that dataset speciﬁc biases are prominent, it is likely that many of
the OOD detection algorithms tend to overemphasize dataset-speciﬁc noise in metrics such
as FPR or AUROC. Further, it is likely there are factors other than semantic content that
are contributing to very high OOD performance. To address this, we consider the recently
proposed SCOOD (semantically coherent OOD detection) benchmark (Yang et al., 2021),
which eﬀectively re-samples the in- and out-distribution datasets such that the only truly
distinguishing factor between the two is the semantic content. This also includes transferring
semantically similar images from the OOD dataset into the in-distribution set (e.g., cats
from TinyImageNet into CIFAR-100), and getting rid of resizing artifacts so that the OOD
performance reﬂects the true performance.
In Table 6 we report average OOD detection performance with AMP on the SCOOD
benchmark, which is comprised of 6 diﬀerent resampled/resized datasets – Texture, SVHN,
CIFAR-10/100, Tiny-ImageNet, LSUN, Places365. We trained ResNet-18 models on modiﬁed
CIFAR-10/100 training sets provided by SCOOD (Yang et al., 2021) and test on their custom
OOD sets for fair comparison. We compare the diﬀerent methods on FPR95, AUROC, and

[CAPTION] Table 6: SCOOD benchmark with ResNet-18. AMP outperforms even the best performing


<!-- page 13 -->
Out of Distribution Detection via Neural Network Anchoring
AUPR (In/Out) metrics. The current best performing approach on SCOOD, UDG (Yang
et al., 2021) additionally also uses outlier data from TinyImageNet for training, similar to
other approaches like OE. We ﬁnd that AMP is the best performing method on CIFAR-100
across all the metrics, while on CIFAR-10 AMP performs comparably to UDG in terms of
FPR95, while being a close second on the other metrics – in spite of not having any access
to outlier data, while being signiﬁcantly better than other comparable baselines.
In Distribution
Method
Pillow Resizing from original LSUN
Average
nearest*
bilinear
bicubic
lanczos
CIFAR-10
(ResNet-34)
MSP
41.5 / 94.0
47.8 / 91.6
45.5 / 92.2
45.3 / 92.4
46.9 / 92.2
Energy
28.6 / 98.4
34.5 / 92.9
33.0 / 93.4
32.0 / 93.8
30.4 / 94.1
GM
1.8 / 99.2
46.2 / 90.7
49.0 / 90.6
46.3 / 91.3
25.6 / 94.9
AMP
7.1 / 98.4
13.0 / 97.4
13.9 / 97.2
14.3 / 97.2
9.6 / 98.1
CIFAR-100
(ResNet-34)
MSP
69.0 / 83.8
80.0 / 79.6
84.0 / 79.9
84.0 / 80.5
79.9 / 79.2
Energy
52.4 / 89.3
84.9 / 74.9
83.3 / 75.7
81.4 / 76.8
73.9 / 79.6
GM
38.7 / 94.4
78.1 / 79.0
77.6 / 80.0
76.0 / 81.5
60.9 / 86.5
AMP
51.1 / 90.5
75.4 / 81.2
74.0 / 81.7
71.8 / 82.7
58.3 / 86.9
∗aliasing artifacts are likely
Performance measures are FPR95 ↓/ AUROC ↑
Table 7: Resizing methods vs OOD benchmarks: We study the eﬀect of resizing on OOD
performance. Here we use ResNet-34 trained on CIFAR-100 as our “in” dataset, to detect
10K LSUN test images. These are resized using diﬀerent interpolation algorithms from
the Pillow package. We evaluate the performance of diﬀerent OOD detection algorithms
on these various datasets, and observe that AMP consistently performs well across all the
variants of LSUN.
5.3. Robustness to Resizing Artifacts
As stated previously, OOD detection benchmarks can be hard to interpret when OOD datasets
are laden with their dataset-speciﬁc noise/biases, often reﬂecting in overly optimistic OOD
performance. A key source for such kinds of noise is artifacts obtained from resizing –
since most datasets have diﬀerent sized images, they are resized typically using a nearest
neighbor interpolation algorithm before running any OOD detection algorithm. The issue
with resizing packages like OpenCV, or native resizing in Pytorch or Tensorﬂow was recently
studied in detail in (Parmar et al., 2021) with its impact on FID scores for generative models
– they concluded that the Pillow resizing with a bicubic interpolation scheme was the most
reliable and gets rid of aliasing artifacts the best. Designing speciﬁc resizing frameworks has
also been shown to have an impact on classiﬁcation accuracy (Talebi and Milanfar, 2021).
This issue is further exaggerated in the current OOD setup relying on CIFAR-10/100
because the images are very small (32 × 32) when compared to other datasets. Speciﬁcally
for OOD, this was pointed out in the SCOOD benchmark (Yang et al., 2021), as being one of
the reasons to re-design these benchmarks from scratch, however there is no ablation on how
resizing alone actually aﬀects performance. Here, we evaluate how some of the best OOD
detection algorithms perform on the same in-out distribution experiment – CIFAR-10/100
→LSUN. Since the original LSUN images are of a much larger size, we resize them using
the Pillow resizing library using diﬀerent types of interpolations.


**[Table p13.1]**
| In Distribution | Method | Pillow Resizing from original LSUN nearest* bilinear bicubic lanczos | Average |
| --- | --- | --- | --- |


**[Table p13.2]**
| MSP Energy GM | 41.5 / 94.0 47.8 / 91.6 45.5 / 92.2 45.3 / 92.4 28.6 / 98.4 34.5 / 92.9 33.0 / 93.4 32.0 / 93.8 1.8 / 99.2 46.2 / 90.7 49.0 / 90.6 46.3 / 91.3 | 46.9 / 92.2 30.4 / 94.1 25.6 / 94.9 |
| --- | --- | --- |


**[Table p13.3]**
| MSP Energy GM | 69.0 / 83.8 80.0 / 79.6 84.0 / 79.9 84.0 / 80.5 52.4 / 89.3 84.9 / 74.9 83.3 / 75.7 81.4 / 76.8 38.7 / 94.4 78.1 / 79.0 77.6 / 80.0 76.0 / 81.5 | 79.9 / 79.2 73.9 / 79.6 60.9 / 86.5 |
| --- | --- | --- |

[CAPTION] Table 7: Resizing methods vs OOD benchmarks: We study the eﬀect of resizing on OOD


<!-- page 14 -->
Anirudh Thiagarajan
We report the results of this study in Table 7. The ﬁrst striking observation is the
amount of variance in OOD performance across the diﬀerent LSUN variants – across the
baselines, and metrics we used. In particular – the case of nearest interpolation is the most
similar to the LSUN (R) benchmark , and it is expected to introduce the most amount of
aliasing artifacts, and existing approach produce unusually low FPR scores. However, the
results the more sophisticated interpolation methods are expected to be more indicative
of the true OOD performance, since they are not prone to aliasing. Interestingly, we note
that AMP performs the best on all of these cases. Our trend across datasets is similar to
energy-based OOD (Liu et al., 2020) and on average, AMP signiﬁcantly outperforms the
other baselines on both the metrics considered.
5.4. Ablation Studies
Finally, we study the two important aspects of AMP here and their impact on OOD
performance using the WRN-40-2 on the CIFAR-10/SVHN benchmark. In Table 8, we
report FPR95 as our metric of choice since its the most sensitive and reﬂective of the
performance. We ablate on two factors – the type of transformation used during training to
anchor the neural network, and the number of anchors needed during inference time. We see
that simple functions like Gaussian blur or color jitter drastically improve the performance.
In all our experiments, we used a combination of all the ﬁve corruptions, with 5 anchors.
Corruption
# Anchors
2
5
10
20
None (trivial)
50.71
50.73
50.85
50.89
+ ColorJitter
+ GaussianBlur
6.53
6.31
6.36
6.28
+ HorizontalFlip
+ Grayscale
9.96
9.85
10.11
9.89
+ ResizedCrop
5.32
5.19
5.35
5.22
Table 8: Ablation
studies
demonstrating
FPR95
for
CIFAR-10-SVHN
OOD
experiment using types of transforma-
tions and varying number of anchors.
All these transformations are applied at
random every time they are executed.
As can be seen, there is not a signiﬁcant
diﬀerence in performance while increasing
the number of anchors, but a big boost in
using consistency training using any of the
corruption functions.
6. Discussion
In this paper, we introduced anchoring as
a strategy to achieve eﬀective heterscedastic
temperature scaling for state-of-the-art OOD
detection on a large suite of benchmarks in-
cluding near and semantically coherent-OOD
problems. Using NTK theory, we show that
our temperature estimates are closely linked
to epistemic uncertainty of the classiﬁer, ex-
plaining it superior performance. We also
introduced a new benchmark that evaluates the robustness of OOD detection methods
against resizing artifacts. Anchoring is a powerful new mechanism to estimate conﬁdence
or reliability of a model, and shows promise well beyond OOD itself including studying its
properties as an uncertainty estimator in more general settings.
Acknowledgments
This work was performed under the auspices of the U.S. Department of Energy by Lawrence
Livermore National Laboratory under Contract DE-AC52-07NA27344. Supported by the


**[Table p14.1]**
| # Anchors Corruption | 2 5 10 20 |
| --- | --- |
| None (trivial) | 50.71 50.73 50.85 50.89 |
| + ColorJitter + GaussianBlur | 6.53 6.31 6.36 6.28 |
| + HorizontalFlip + Grayscale | 9.96 9.85 10.11 9.89 |
| + ResizedCrop | 5.32 5.19 5.35 5.22 |

[CAPTION] Table 8: Ablation


<!-- page 15 -->
Out of Distribution Detection via Neural Network Anchoring
LDRD Program under projects 21-ERD-028, 22-ERD-006 and released under LLNL-JRNL-
829478.
Appendix
We include experimental details, additional results, pseudo-code in the appendix. Code to
reproduce our results is available at github.com/LLNL/AMP
References
Sanjeev Arora, Simon Du, Wei Hu, Zhiyuan Li, and Ruosong Wang. Fine-grained analysis of
optimization and generalization for overparameterized two-layer neural networks. In International
Conference on Machine Learning, pages 322–332. PMLR, 2019.
Alberto Bietti and Julien Mairal. On the inductive bias of neural tangent kernels. Advances in
Neural Information Processing Systems, 32, 2019.
Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and Daan Wierstra. Weight uncertainty in
neural network. In International Conference on Machine Learning, pages 1613–1622. PMLR, 2015.
Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describ-
ing textures in the wild. In Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition, pages 3606–3613, 2014.
Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model
uncertainty in deep learning. In international conference on machine learning, pages 1050–1059.
PMLR, 2016.
Chuan Guo, GeoﬀPleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural
networks. In International Conference on Machine Learning, pages 1321–1330. PMLR, 2017.
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
Deep residual learning for image
recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition,
pages 770–778, 2016.
Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common
corruptions and perturbations. In International Conference on Learning Representations, 2019.
URL https://openreview.net/forum?id=HJz6tiCqYm.
Dan Hendrycks, Mantas Mazeika, and Thomas Dietterich. Deep anomaly detection with outlier
exposure. In ICLR, 2019.
Arthur Jacot, Franck Gabriel, and Cl´ement Hongler. Neural tangent kernel: Convergence and
generalization in neural networks. Advances in neural information processing systems, 31, 2018.
Moksh Jain, Salem Lahlou, Hadi Nekoei, Victor Butoi, Paul Bertin, Jarrid Rector-Brooks, Maksym
Korablyov, and Yoshua Bengio. Deup: Direct epistemic uncertainty prediction. arXiv preprint
arXiv:2102.08501, 2021.
Ranganath Krishnan and Omesh Tickoo.
Improving model calibration with accuracy versus
uncertainty optimization.
In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and
H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 18237–
18248. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper/2020/file/
d3d9446802a44259755d38e6d163e820-Paper.pdf.


<!-- page 16 -->
Anirudh Thiagarajan
Alex Krizhevsky, Geoﬀrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.
Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive
uncertainty estimation using deep ensembles. Advances in neural information processing systems,
30, 2017.
Jaehoon Lee, Lechao Xiao, Samuel Schoenholz, Yasaman Bahri, Roman Novak, Jascha Sohl-Dickstein,
and Jeﬀrey Pennington. Wide neural networks of any depth evolve as linear models under gradient
descent. Advances in neural information processing systems, 32, 2019.
Kimin Lee, Kibok Lee, Honglak Lee, and Jinwoo Shin. A simple uniﬁed framework for detecting
out-of-distribution samples and adversarial attacks. Advances in neural information processing
systems, 31, 2018a.
Kimin Lee, Kibok Lee, Honglak Lee, and Jinwoo Shin. A simple uniﬁed framework for detecting
out-of-distribution samples and adversarial attacks. In NeurIPS, 2018b.
Shiyu Liang, Yixuan Li, and Rayadurgam Srikant. Enhancing the reliability of out-of-distribution
image detection in neural networks. In ICLR, 2017.
Weitang Liu, Xiaoyun Wang, John Owens, and Yixuan Li. Energy-based out-of-distribution detection.
In NeurIPS, 2020.
Radford M Neal. Bayesian learning for neural networks, volume 118. Springer Science & Business
Media, 2012.
Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y Ng. Reading
digits in natural images with unsupervised feature learning. 2011.
Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, David Sculley, Sebastian Nowozin, Joshua
Dillon, Balaji Lakshminarayanan, and Jasper Snoek. Can you trust your model’s uncertainty?
evaluating predictive uncertainty under dataset shift. Advances in neural information processing
systems, 32, 2019.
Gaurav Parmar, Richard Zhang, and Jun-Yan Zhu. On buggy resizing libraries and surprising
subtleties in ﬁd calculation. arXiv preprint arXiv:2104.11222, 2021.
Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang,
Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition
challenge. International journal of computer vision, 115(3):211–252, 2015.
Chandramouli Shama Sastry and Sageev Oore. Detecting out-of-distribution examples with gram
matrices. In International Conference on Machine Learning, pages 8491–8501. PMLR, 2020.
Hossein Talebi and Peyman Milanfar. Learning to resize images for computer vision tasks. In
Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 497–506, 2021.
Matthew Tancik, Pratul Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh
Singhal, Ravi Ramamoorthi, Jonathan Barron, and Ren Ng. Fourier features let networks learn
high frequency functions in low dimensional domains. Advances in Neural Information Processing
Systems, 33:7537–7547, 2020.
Joost Van Amersfoort, Lewis Smith, Yee Whye Teh, and Yarin Gal. Uncertainty estimation using a
single deep deterministic neural network. In International Conference on Machine Learning, pages
9690–9700. PMLR, 2020.


<!-- page 17 -->
Out of Distribution Detection via Neural Network Anchoring
Pingmei Xu, Krista A Ehinger, Yinda Zhang, Adam Finkelstein, Sanjeev R Kulkarni, and Jianxiong
Xiao. Turkergaze: Crowdsourcing saliency with webcam based eye tracking. arXiv preprint
arXiv:1504.06755, 2015.
Jingkang Yang, Haoqi Wang, Litong Feng, Xiaopeng Yan, Huabin Zheng, Wayne Zhang, and Ziwei Liu.
Semantically coherent out-of-distribution detection. In Proceedings of the IEEE/CVF International
Conference on Computer Vision, pages 8301–8309, 2021.
Fisher Yu, Ari Seﬀ, Yinda Zhang, Shuran Song, Thomas Funkhouser, and Jianxiong Xiao. Lsun:
Construction of a large-scale image dataset using deep learning with humans in the loop. arXiv
preprint arXiv:1506.03365, 2015.
Qing Yu and Kiyoharu Aizawa. Unsupervised out-of-distribution detection by maximum classiﬁer
discrepancy. In ICCV, 2019.
Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. arXiv preprint arXiv:1605.07146,
2016.
Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva, and Antonio Torralba. Places: A 10
million image database for scene recognition. IEEE transactions on pattern analysis and machine
intelligence, 40(6):1452–1464, 2017.


<!-- page 18 -->
Anirudh Thiagarajan
Appendix A. Details of Benchmark Datasets
We use commonly used benchmarks to evaluate AMP, these include the following datasets – iSUN
(Xu et al., 2015), LSUN (R), LSUN (C) (Yu et al., 2015), Places365 (Zhou et al., 2017), Texture
(Cimpoi et al., 2014), and SVHN (Netzer et al., 2011)
Consistency training details
The transformation T was applied to the anchors using a pre-
speciﬁed schedule, every 5th batch for CIFAR-10/100 and every 10th batch for ImageNet, while the
clean anchors were used directly in the other batches. However, from our experiments, we found
that the choice of this schedule is not sensitive and the detection performance was similar even with
other schedules. During the inference step, we did not utilize any transformation T , and ﬁxed the
number of anchors K = 5 while making predictions for a test image. We performed an ablation on
the number of anchors (reported at the end of the section), and observed that even a small number
of random anchors was suﬃcient to obtain good detection performance, thus making our approach
eﬃcient in practice.
During training we always use K = 1 anchor, which is typically chosen by randomly shuﬄing the
current batch so that every input sample is assigned a random anchor from that batch. During
training we use RandomCrop, RandomHorizontalFlip augmentations in Pytorch. For the test set and
the OOD set, we normalize data to the same mean and standard deviation as the training set without
any additional transformations.
Appendix B. Hyperparameter settings
CIFAR-10/100: We use standard training protocol for both CIFAR-10/100 datasets using all our
networks – WideResNet, ResNet-18, ResNet-34 (He et al., 2016). We use an SGD optimizer with an
initial learning rate of 0.1, momentum of 0.9, and weight decay of 5e −4. This learning rate is scaled
down by a γ = 0.2 using a schedule of [60, 120, 160] epochs out of the total 200 epochs for training.
We use a batch size of 128 in all our training experiments for CIFAR datasets. ImageNet: We also
follow standard training protocol for ResNet-50 on ImageNet as well. We use an SGD optimizer with
a learning rate of 0.1, weight decay of 1e −4, momentum of 0.9. We decay the learning rate by 0.1
every 30 epochs, and train for a total of 120 epochs. We use a batch size of 128 to train the model.
Method
AUROC ↑
ResNet-18 (He et al., 2016)
91.77 ± 1.85
DUQ (Van Amersfoort et al., 2020)
92.70 ± 1.30
Deep Ens (Lakshminarayanan et al., 2017)
94.70
AMP
97.41 ± 0.72
Table 9: OOD Detection with uncertainties on CIFAR-SVHN with ResNet-18.
Appendix C. Modiﬁcation to anchor a model
We demonstrate in 1, the simple modiﬁcation to be able to train with anchoring
Appendix D. Additional Results
We report detailed results for individual datasets on various benchmarks used in the paper here.
Table 11 and Table 10 report 4 performance metrics for the SCOOD benchmark (Yang et al., 2021),
where we use the re-sampled OOD set following the SCOOD protocol. We observe competitive


**[Table p18.1]**
| Method | AUROC ↑ |
| --- | --- |


**[Table p18.2]**
| ResNet-18 (He et al., 2016) DUQ (Van Amersfoort et al., 2020) Deep Ens (Lakshminarayanan et al., 2017) | 91.77 ± 1.85 92.70 ± 1.30 94.70 |
| --- | --- |

[CAPTION] Table 9: OOD Detection with uncertainties on CIFAR-SVHN with ResNet-18.

[CAPTION] Table 11 and Table 10 report 4 performance metrics for the SCOOD benchmark (Yang et al., 2021),


<!-- page 19 -->
Out of Distribution Detection via Neural Network Anchoring
Algorithm 1: PyTorch-style pseudo-code for anchoring.
def create anchored model(model):
model.conv1 = nn.Conv2d(in channels=6, 64)
return model
Tx = transforms.Compose([
transforms.RandomResizedCrop(size=224),
transforms.RandomHorizontalFlip(),
transforms.RandomApply([color jitter ,blurr], p=0.8),
])
## load model and change the first conv layer
model basic = ResNet50(pre trained=False,n class =1000)
model = create anchored model(model basic)
## load datasets, setup optimizer, define criterion etc.
for images, targets in train loder:
batch order = np.arange(images.shape[0])
np.random.shuffle(batch order)
anchors = images[batch order ,:,:,:]
diff = images−anchors
if i % 10 ==0:
tx anchors = Tx(anchors)
else:
tx anchors = anchors
batch = torch.cat([tx anchors ,diff],axis=1)
output = model(batch)
loss = criterion(output, target)
optimizer.zero grad()
loss.backward()
optimizer.step()
performance on CIFAR-10 and state-of-the-art on CIFAR-100 with AMP. Next, Table 12, we report
detailed performance numbers on the second OOD benchmark used in the paper. We note that
our method consistently performs either the best or second best as compared to GM (Sastry and
Oore, 2020), while being better on average across the various datasets. In particular, we see that on
challenging datasets like near-OOD AMP is signiﬁcantly better than all competing baselines. Finally,
in Table 9 we show uncertainty based OOD on a CIFAR-10 vs SVHN benchmark, compared to other
uncertainty based approaches. We see once again that AMP is signiﬁcantly better than sophisticated
methods including Deep Ensembles that requires multiple models to be trained.


<!-- page 20 -->
Anirudh Thiagarajan
Method
Dataset
FPR95 ↓
AUROC ↑
AUPR(In/Out) ↑
ODIN
Texture
42.52
84.06
86.01 / 80.73
SVHN
52.27
83.26
63.76 / 92.60
CIFAR-100
56.34
78.40
73.21 / 80.99
Tiny-ImageNet
59.09
79.69
79.34 / 77.52
LSUN
47.85
84.56
81.56 / 85.58
Places365
53.94
82.01
54.92 / 93.30
Mean
52.00
82.00
73.13 / 85.12
EBO
Texture
52.11
80.70
83.34 / 75.20
SVHN
30.56
92.08
80.95 / 96.28
CIFAR-100
56.98
79.65
75.09 / 81.23
Tiny-ImageNet
57.81
81.65
81.80 / 78.75
LSUN
50.56
85.04
82.80 / 85.29
Places365
52.16
83.86
58.96 / 93.90
Mean
50.03
83.83
77.15 / 85.11
MCD
Texture
83.92
81.59
90.20 / 63.27
SVHN
60.27
89.78
85.33 / 94.25
CIFAR-100
74.00
82.78
83.97 / 79.16
Tiny-ImageNet
78.89
80.98
85.63 / 72.48
LSUN
68.96
84.71
85.74 / 81.50
Places365
72.08
83.51
69.44 / 92.52
Mean
73.02
83.89
83.39 / 80.53
OE
Texture
51.17
89.56
93.79 / 81.88
SVHN
20.88
96.43
93.62 / 98.32
CIFAR-100
58.54
86.22
86.17 / 84.88
Tiny-ImageNet
58.98
87.65
90.9 / 82.16
LSUN
57.97
86.75
87.69 / 85.07
Places365
55.64
87.00
73.11 / 94.67
Mean
50.53
88.93
87.55 / 87.83
UDG
Texture
20.43
96.44
98.12 / 92.91
SVHN
13.26
97.49
95.66 / 98.69
CIFAR-100
47.20
90.98
91.74 / 89.36
Tiny-ImageNet
50.18
91.91
94.43 / 86.99
LSUN
42.05
93.21
94.53 / 91.03
Places365
44.22
92.64
87.17 / 96.66
Mean
36.22
93.78
93.61 / 92.61
AMP (ours)
Texture
52.43
88.74
91.91 / 80.48
SVHN
12.53
97.60
95.58 / 98.83
CIFAR-100
48.10
89.61
88.99 / 88.47
Tiny-ImageNet
50.40
90.26
92.01 / 85.74
LSUN
23.01
95.17
94.94 / 94.78
Places365
34.45
93.25
83.95 / 97.19
Mean
36.82
92.40
91.23 / 90.91
Table 10: Detailed results on SCOOD benchmark (Yang et al., 2021) using
CIFAR-10/ResNet-18. AMP performs very close to methods that use outlier
exposure, while outperforming all the baselines that do not. We use results for
baselines as reported in (Yang et al., 2021)


**[Table p20.1]**
| Texture SVHN CIFAR-100 Tiny-ImageNet LSUN Places365 | 42.52 84.06 86.01 / 80.73 52.27 83.26 63.76 / 92.60 56.34 78.40 73.21 / 80.99 59.09 79.69 79.34 / 77.52 47.85 84.56 81.56 / 85.58 53.94 82.01 54.92 / 93.30 |
| --- | --- |


**[Table p20.2]**
| Texture SVHN CIFAR-100 Tiny-ImageNet LSUN Places365 | 52.11 80.70 83.34 / 75.20 30.56 92.08 80.95 / 96.28 56.98 79.65 75.09 / 81.23 57.81 81.65 81.80 / 78.75 50.56 85.04 82.80 / 85.29 52.16 83.86 58.96 / 93.90 |
| --- | --- |


**[Table p20.3]**
| Texture SVHN CIFAR-100 Tiny-ImageNet LSUN Places365 | 83.92 81.59 90.20 / 63.27 60.27 89.78 85.33 / 94.25 74.00 82.78 83.97 / 79.16 78.89 80.98 85.63 / 72.48 68.96 84.71 85.74 / 81.50 72.08 83.51 69.44 / 92.52 |
| --- | --- |


**[Table p20.4]**
| Texture SVHN CIFAR-100 Tiny-ImageNet LSUN Places365 | 51.17 89.56 93.79 / 81.88 20.88 96.43 93.62 / 98.32 58.54 86.22 86.17 / 84.88 58.98 87.65 90.9 / 82.16 57.97 86.75 87.69 / 85.07 55.64 87.00 73.11 / 94.67 |
| --- | --- |


**[Table p20.5]**
| Texture SVHN CIFAR-100 Tiny-ImageNet LSUN Places365 | 20.43 96.44 98.12 / 92.91 13.26 97.49 95.66 / 98.69 47.20 90.98 91.74 / 89.36 50.18 91.91 94.43 / 86.99 42.05 93.21 94.53 / 91.03 44.22 92.64 87.17 / 96.66 |
| --- | --- |


**[Table p20.6]**
| Texture SVHN CIFAR-100 Tiny-ImageNet LSUN Places365 | 52.43 88.74 91.91 / 80.48 12.53 97.60 95.58 / 98.83 48.10 89.61 88.99 / 88.47 50.40 90.26 92.01 / 85.74 23.01 95.17 94.94 / 94.78 34.45 93.25 83.95 / 97.19 |
| --- | --- |

[CAPTION] Table 10: Detailed results on SCOOD benchmark (Yang et al., 2021) using


<!-- page 21 -->
Out of Distribution Detection via Neural Network Anchoring
Method
Dataset
FPR95 ↓
AUROC ↑
AUPR(In/Out) ↑
ODIN
Texture
79.47
77.92
86.69 / 62.97
SVHN
90.33
75.59
65.25 / 84.49
CIFAR-10
81.82
77.90
79.93 / 73.39
Tiny-ImageNet
82.74
77.58
86.26 / 61.38
LSUN
80.57
78.22
86.34 / 63.44
Places365
76.42
80.66
66.77 / 89.66
Mean
81.89
77.98
78.54 / 72.56
EBO
Texture
84.29
76.32
85.87 / 59.12
SVHN
78.23
83.57
75.61 / 90.24
CIFAR-10
81.25
78.95
80.01 / 74.44
Tiny-ImageNet
83.32
78.34
87.08 / 62.13
LSUN
84.51
77.66
86.42 / 61.40
Places365
78.37
80.99
68.22 / 89.60
Mean
81.66
79.31
80.54 / 72.82
MCD
Texture
83.97
73.46
83.11 / 56.79
SVHN
85.82
76.61
65.50 / 85.52
CIFAR-10
87.74
73.15
76.51 / 67.24
Tiny-ImageNet
84.46
75.32
85.11 / 59.49
LSUN
86.08
74.05
84.21 / 58.62
Places365
82.74
76.30
61.15 / 87.19
Mean
85.14
74.82
75.93 / 69.14
OE
Texture
86.56
73.89
84.48 / 54.84
SVHN
68.87
84.23
75.11 / 91.41
CIFAR-10
79.72
78.92
81.95 / 74.28
Tiny-ImageNet
83.41
76.99
86.36 / 60.56
LSUN
83.53
77.10
86.28 / 60.97
Places365
78.24
79.62
67.13 / 88.89
Mean
80.06
78.46
80.22 / 71.83
UDG
Texture
75.04
79.53
87.63 / 65.49
SVHN
60.00
88.25
81.46 / 93.63
CIFAR-10
83.35
76.18
78.92 / 71.15
Tiny-ImageNet
81.73
77.18
86.00 / 61.67
LSUN
78.70
76.79
84.74 / 63.05
Places365
73.86
79.87
65.36 / 89.60
Mean
75.45
79.63
80.69 / 74.10
AMP (ours)
Texture
68.39
83.76
90.69 / 72.16
SVHN
34.12
94.21
90.11 / 97.24
CIFAR-10
80.47
78.74
81.36 / 74.07
Tiny-ImageNet
80.70
78.34
86.95 / 63.03
LSUN
83.60
76.64
85.80 / 60.63
Places365
74.77
81.67
69.97 / 90.09
Mean
70.34
82.22
84.14 / 76.20
Table 11: Detailed results on SCOOD benchmark (Yang et al., 2021) using
CIFAR-100/ResNet-18. AMP consistently outperforms all methods including
those that use outlier exposure. We use results for baselines as reported in (Yang
et al., 2021)


**[Table p21.1]**
| Texture SVHN CIFAR-10 Tiny-ImageNet LSUN Places365 | 79.47 77.92 86.69 / 62.97 90.33 75.59 65.25 / 84.49 81.82 77.90 79.93 / 73.39 82.74 77.58 86.26 / 61.38 80.57 78.22 86.34 / 63.44 76.42 80.66 66.77 / 89.66 |
| --- | --- |


**[Table p21.2]**
| Texture SVHN CIFAR-10 Tiny-ImageNet LSUN Places365 | 84.29 76.32 85.87 / 59.12 78.23 83.57 75.61 / 90.24 81.25 78.95 80.01 / 74.44 83.32 78.34 87.08 / 62.13 84.51 77.66 86.42 / 61.40 78.37 80.99 68.22 / 89.60 |
| --- | --- |


**[Table p21.3]**
| Texture SVHN CIFAR-10 Tiny-ImageNet LSUN Places365 | 83.97 73.46 83.11 / 56.79 85.82 76.61 65.50 / 85.52 87.74 73.15 76.51 / 67.24 84.46 75.32 85.11 / 59.49 86.08 74.05 84.21 / 58.62 82.74 76.30 61.15 / 87.19 |
| --- | --- |


**[Table p21.4]**
| Texture SVHN CIFAR-10 Tiny-ImageNet LSUN Places365 | 86.56 73.89 84.48 / 54.84 68.87 84.23 75.11 / 91.41 79.72 78.92 81.95 / 74.28 83.41 76.99 86.36 / 60.56 83.53 77.10 86.28 / 60.97 78.24 79.62 67.13 / 88.89 |
| --- | --- |


**[Table p21.5]**
| Texture SVHN CIFAR-10 Tiny-ImageNet LSUN Places365 | 75.04 79.53 87.63 / 65.49 60.00 88.25 81.46 / 93.63 83.35 76.18 78.92 / 71.15 81.73 77.18 86.00 / 61.67 78.70 76.79 84.74 / 63.05 73.86 79.87 65.36 / 89.60 |
| --- | --- |


**[Table p21.6]**
| Texture SVHN CIFAR-10 Tiny-ImageNet LSUN Places365 | 68.39 83.76 90.69 / 72.16 34.12 94.21 90.11 / 97.24 80.47 78.74 81.36 / 74.07 80.70 78.34 86.95 / 63.03 83.60 76.64 85.80 / 60.63 74.77 81.67 69.97 / 90.09 |
| --- | --- |

[CAPTION] Table 11: Detailed results on SCOOD benchmark (Yang et al., 2021) using


<!-- page 22 -->
Anirudh Thiagarajan
In-dist
(model)
OOD
TNR at TPR 95% ↑
AUROC ↑
Detection Acc. ↑
MSP / ODIN / Gram Matrices / Ours
CIFAR-10
(ResNet-34)
iSUN
44.6 / 73.2 / 97.3 / 91.8
91.0 / 94.0 / 99.1 / 98.2
85.0 / 86.5 / 96.2 / 93.8
LSUN (R)
49.8 / 82.1 / 98.2 / 92.4
91.0 / 94.1 / 99.2 / 98.7
85.3 / 86.7 / 96.7 / 94.9
LSUN (C)
48.6 / 62.0 / 91.7 / 98.5
91.9 / 91.2 / 98.3 / 99.5
86.3 / 82.4 / 94.1 / 97.0
TinyImgNet (R)
41.0 / 67.9 / 95.9 / 88.8
91.0 / 94.0 / 98.9 / 97.0
85.1 / 86.5 / 95.6 / 92.1
TinyImgNet (C)
46.4 / 68.7 / 77.6 / 94.5
91.4 / 93.1 / 96.2 / 98.7
85.4 / 85.2 / 90.8 / 94.9
SVHN
50.5 / 70.3 / 95.3 / 91.2
89.9 / 96.7 / 99.0 / 98.1
85.1 / 91.1 / 95.2 / 93.7
CIFAR-100
33.3 / 42.0 / 40.2 / 56.5
86.4 / 85.8 / 83.6 / 90.2
80.4 / 78.6 / 76.4 / 83.5
CIFAR-100
(ResNet-34)
iSUN
16.9 / 45.2 / 66.2 / 48.7
75.8 / 85.5 / 94.6 / 90.2
70.1 / 78.5 / 88.3 / 82.6
LSUN (R)
18.8 / 23.2 / 61.4 / 54.2
75.8 / 85.6 / 94.4 / 91.7
69.9 / 78.3 / 88.6 / 84.3
LSUN (C)
18.7 / 44.1 / 43.7 / 67.8
75.5 / 82.7 / 89.7 / 94.0
69.2 / 75.9 / 82.4 / 86.4
TinyImgNet (R)
20.4 / 36.1 / 66.8 / 45.9
77.2 / 87.6 / 94.7 / 89.2
70.8 / 80.1 / 88.6 / 81.3
TinyImgNet (C)
24.3 / 44.3 / 41.4 / 61.5
79.7 / 85.4 / 89.7 / 92.9
72.5 / 78.3 / 82.8 / 85.4
SVHN
20.3 / 62.7 / 54.5 / 56.5
79.5 / 93.9 / 92.1 / 91.9
73.2 / 88.0 / 84.9 / 83.7
CIFAR-10
19.1 / 18.7 / 16.9 / 17.5
77.1 / 77.2 / 74.5 / 79.9
71.0 / 71.2 / 68.9 / 73.8
Table 12: Detailed results on the OOD detection benchmark with ResNet-34. Note, diﬀerent
from the main paper we report TNR here (instead of FPR95) which is 100-FPR95,
as this was used in (Sastry and Oore, 2020). We observe that AMP performs
comparably to Gram Matrices, while being better on average. Our method has
signiﬁcant advantages on more challenging datasets like near OOD.

[CAPTION] Table 12: Detailed results on the OOD detection benchmark with ResNet-34. Note, diﬀerent