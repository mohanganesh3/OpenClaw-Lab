<!-- page 1 -->
Beyond the Loss Curve: Scaling Laws, Active Learning, and the Limits of
Learning from Exact Posteriors
Arian Khorasani 1 Nathaniel Chen * 2 Yug D Oswal * 3 Akshat Santhana Gopalan 4 Egemen Kolemen 2
Ravid Shwartz-Ziv 5
Abstract
How close are neural networks to the best they
could possibly do? Standard benchmarks can-
not answer this because they lack access to the
true posterior p(y|x). We use class-conditional
normalizing flows as oracles that make exact pos-
teriors tractable on realistic images (AFHQ, Im-
ageNet). This enables five lines of investigation.
Scaling laws: Prediction error decomposes into ir-
reducible aleatoric uncertainty and reducible epis-
temic error; the epistemic component follows a
power law in dataset size, continuing to shrink
even when total loss plateaus. Limits of learn-
ing: The aleatoric floor is exactly measurable,
and architectures differ markedly in how they ap-
proach it: ResNets exhibit clean power-law scal-
ing while Vision Transformers stall in low-data
regimes. Soft labels: Oracle posteriors contain
learnable structure beyond class labels: training
with exact posteriors outperforms hard labels and
yields near-perfect calibration. Distribution shift:
The oracle computes exact KL divergence of con-
trolled perturbations, revealing that shift type mat-
ters more than shift magnitude: class imbalance
barely affects accuracy at divergence values where
input noise causes catastrophic degradation. Ac-
tive learning: Exact epistemic uncertainty dis-
tinguishes genuinely informative samples from
inherently ambiguous ones, improving sample ef-
ficiency. Our framework reveals that standard
metrics hide ongoing learning, mask architectural
differences, and cannot diagnose the nature of
distribution shift.
*Equal contribution 1Mila-Quebec AI Institute, Canada 2Plasma
Physics Lab, Princeton University, USA 3School of Computer Sci-
ence and Engineering, Vellore Institute of Technology, India 4High
School Student, John P. Stevens High School, USA 5Center of Data
Science, New York University, New York, USA. Correspondence
to: Arian Khorasani <Arian.Khorasani@mila.quebec>.
Preprint. February 13, 2026.
1. Introduction
Every paper reporting test accuracy implicitly asks: how
good is this model? But good compared to what? Without
access to the true posterior p(y|x), we cannot tell whether a
model is at 50% or 99% of the theoretical maximum. When
a loss curve flattens, we cannot distinguish irreducible noise
(aleatoric uncertainty) from gaps in the model’s knowledge
(epistemic uncertainty) (Gal & Ghahramani, 2016; Guo
et al., 2017). Neural Scaling laws (Kaplan et al., 2020) tell
us loss decreases as N −α, but loss conflates both (Hoffmann
et al., 2022). When performance degrades under distribu-
tion shift, we cannot tell whether the shift was large or
small (Hendrycks & Dietterich, 2019) (Gal & Ghahramani,
2016).
A class-conditional normalizing flow (Dinh et al., 2017;
Kingma & Dhariwal, 2018) trained on real images provides
a concrete reference point. The flow defines an explicit
pθ(x | y) from which we can both sample and evaluate
exact likelihoods. We treat this not as an approximation
to nature, but as the complete specification of a synthetic
world in which Bayes-optimal posteriors are computable in
closed form (Dinh et al., 2017). In this world, the expected
cross-entropy of any classifier decomposes exactly:
L(qθ) = Ex[H(p(y|x))]
|
{z
}
aleatoric (irreducible)
+ Ex[KL(p(y|x)∥qθ(y|x))]
|
{z
}
epistemic (reducible)
This decomposition, intractable on any standard benchmark,
is a direct consequence of Bayes-optimal risk decomposition
and is the basis for all our experiments. As illustrated in
Figure 1, we train state-of-the-art normalizing flows on
realistic image datasets and conduct a thorough validation
to confirm the generated images match real data statistics
and are not memorized (Section 2.3). We find:
• Scaling laws beyond the loss curve. Epistemic error
follows a power law KL ∝N −α that continues to
shrink even when total loss plateaus. The exponents
differ across architectures.
• The limits of learning are exactly measurable. The
aleatoric floor is a hard limit no architecture can beat.
1
arXiv:2602.00315v2  [cs.LG]  12 Feb 2026


<!-- page 2 -->
Decomposing Neural Network Error via Flow-Based Oracles
Figure 1. Normalizing flows enable exact posterior computation on realistic images. (Left) We train class-conditional flows on
AFHQ/ImageNet, creating a frozen oracle with tractable likelihoods. (Center) The oracle decomposes prediction error into aleatoric
(irreducible) and epistemic (reducible) components, quantities hidden in standard benchmarks. (Right) We apply this framework to
diagnose scaling laws, quantify distribution shift with exact KL divergence, and train models with exact soft labels.
Architectures differ markedly in how they approach it:
ResNets show clean power-law scaling while Vision
Transformers stall without pretraining.
• Soft labels from exact posteriors enable better learn-
ing. Training with p(y|x) instead of argmax labels
outperforms hard labels and yields near-perfect calibra-
tion (Szegedy et al., 2015).
• Distribution shift is exactly quantifiable. The oracle
computes KL[pshifted∥pbaseline] exactly, revealing that
shift type matters far more than magnitude: class im-
balance barely affects accuracy at divergence values
where input noise causes collapse.
• Active learning with ground-truth uncertainty. Ex-
act epistemic uncertainty distinguishes genuinely in-
formative samples from inherently ambiguous ones,
improving sample efficiency over standard acquisition
functions that confound aleatoric and epistemic uncer-
tainty (Gal et al., 2017).
Our oracle defines a synthetic world, not nature. But the phe-
nomena it reveals (hidden epistemic progress, architectural
scaling gaps, misleading divergence measures) are proper-
ties of the models and metrics, not the data source (Belghazi
et al., 2021).
2. The Oracle Framework
We build an oracle benchmark by training class-conditional
normalizing flows that define tractable densities pθ(x | y) on
high-dimensional inputs (Papamakarios et al., 2021). Once
the flows are trained and frozen, they specify a synthetic
world in which the posterior is available by Bayes’ rule,
p(y | x) =
pθ(x | y) πy
P
y′ pθ(x | y′) πy′ .
(1)
Because likelihoods are tractable, we can compute Bayes-
optimal posteriors exactly under the oracle model (up to
floating-point arithmetic) via Bayes’ rule (Dinh et al., 2017).
This in turn makes our loss decomposition and controlled
scaling, shift, and active-learning experiments possible.
Practical details on training and scaling to many classes
(ImageNet) are provided in the appendix (Deng et al., 2009).
2.1. Flows as Oracles
A normalizing flow defines an invertible mapping between
inputs x and latent variables z (Dinh et al., 2017; Kingma &
Dhariwal, 2018): the forward map fθ sends x 7→z, while
the inverse f −1
θ
generates samples by mapping z 7→x for
z ∼N(0, I). The key property is that densities are tractable
(Papamakarios et al., 2021):
log pθ(x) = log pZ(fθ(x)) + log |det Jfθ(x)| .
(2)
2

[CAPTION] Figure 1. Normalizing flows enable exact posterior computation on realistic images. (Left) We train class-conditional flows on


<!-- page 3 -->
Decomposing Neural Network Error via Flow-Based Oracles
We train one flow per class, yielding class-conditional densi-
ties pθ(x | y), and then freeze the parameters. For any input
x, we compute the oracle posterior via Bayes’ rule (Eq. 1).
Unlike typical Bayesian methods that rely on sampling or
variational approximations, this computation is exact under
the oracle model (Alemi et al., 2019) since the likelihoods
and class prior are explicitly known: given the trained flows
and the chosen class prior π, p(y | x) is computed in closed
form (implemented in log space with log-sum-exp normal-
ization for numerical stability). The oracle therefore returns
the true posterior for the synthetic world defined by the flow.
2.2. What Does “Truth” Mean Here?
Here, “true” means Bayes-optimal with respect to the oracle
world induced by the trained flows. The oracle world is
defined by (i) the per-class flow densities pθ(x | y), (ii) the
exact preprocessing pipeline used during flow training and
sampling, and (iii) a class prior πy. All “exact” statements
are with respect to this specification: for the oracle world,
we can compute the Bayes posterior p(y | x) (Eq. 1) and
therefore quantities such as the aleatoric term E[H(p(y |
x))] and epistemic term E[KL(p ∥qθ)] exactly under the
oracle model (Section 3).
Unless otherwise stated, we use a uniform prior over classes
(πy = 1/K). In the class-imbalance shift experiments, we
modify the class prior π while keeping pθ(x | y) fixed
(Chawla et al., 2002), isolating prior shift from conditional
shift.
The main limitation is external validity: conclusions are
only useful insofar as samples from the oracle world match
the statistical and semantic properties of the real data used
to fit the flows.
2.3. Oracle Validation
Bottom line: Generated images are statistically similar
to the original data across six metrics, and classifiers
trained on oracle samples approach the Bayes-optimal
bound (Salimans et al., 2016).
Because the oracle provides exact posteriors, we can com-
pute the Bayes-optimal accuracy for the oracle world:
99.77% on AFHQ (Choi et al., 2020) (Bayes error 0.23%).
If classifiers trained on oracle data approach this bound,
the oracle is producing learnable, well-structured sam-
ples. We trained five architectures on 50K oracle samples.
ConvNeXt (Liu et al., 2022) reaches 98.0%, ResNet (He
et al., 2015) 97.7%, Swin (Liu et al., 2021) 97.7%, Mo-
bileNet (Howard et al., 2019) 97.7%, and ViT-Base (Doso-
vitskiy et al., 2021) 97.0% (Table 1).
The remaining
1.8–2.8% gap closes with longer training. On ImageNet-
64 (Chrabaszcz et al., 2017), the Bayes error is 4.031%.
Table 1. Classifiers approach the Bayes-optimal bound on
oracle data. Self-validation on AFHQ: all architectures reach 97–
98% accuracy, within 2–3% of the theoretical optimum (99.8%).
Architecture
Accuracy (%)
Error (%)
Gap (%)
ConvNeXt
98.03
1.97
1.75
ResNet-18
97.73
2.27
2.05
Swin
97.68
2.32
2.10
MobileNet
97.65
2.35
2.12
ViT-Base
96.98
3.02
2.80
We validated distributional quality using six metrics.
FID (Heusel et al., 2018) measures feature distribution dis-
tance (28.44 on AFHQ, 13.48 on ImageNet-64), higher
than diffusion models because flows trade sharpness for ex-
act likelihoods. Inception Score (Salimans et al., 2016)
evaluates quality and diversity (6.24 ± 3.57 on AFHQ,
32.57 ± 4.47 on ImageNet-64). Manifold coverage con-
firms broad support (90% on AFHQ, 89.9% on ImageNet-
64). Feature variance match shows semantic diversity
is preserved (83–92% on AFHQ, 72–93% on ImageNet-
64). Memorization check: only 36% of AFHQ samples
(6.5% on ImageNet-64) have a training neighbor within fea-
ture distance 10, and visual inspection confirms these share
pose/lighting but depict distinct individuals. Full metrics
and protocol in Appendix A, Table 4.
2.4. Implementation
We use TarFlow (Zhai et al., 2025), a recent normalizing
flow that achieves strong likelihood scores on AFHQ and Im-
ageNet. Training follows standard practice: dequantization,
logit preprocessing, and maximum likelihood optimization.
We train separate flows per class on AFHQ (3 classes: cat,
dog, wild) and ImageNet (Deng et al., 2009) (up to 1000
classes at 64×64 and 128×128 resolution). After training,
we freeze the flows and generate 50,000 labeled samples
with oracle posteriors. We compute posteriors in log space
and normalize with log-sum-exp for numerical stability. Full
architecture and systems details (including throughput and
scaling across many classes) appear in the appendix.
3. Decomposing Prediction Error
Why does this decomposition matter? In practice, a flatten-
ing loss curve is ambiguous: it could mean the model has
learned everything learnable, or it could mean the model is
stuck while reducible error remains. Distinguishing these
cases determines whether more data, more capacity, or a
different task formulation is needed. Similarly, calibration
methods aim to match model confidence to true probabil-
ities, but without access to true posteriors, calibration can
only be evaluated against empirical frequencies (Guo et al.,
3

[CAPTION] Table 1. Classifiers approach the Bayes-optimal bound on


<!-- page 4 -->
Decomposing Neural Network Error via Flow-Based Oracles
2017). Access to the true posterior p(y|x) enables a decom-
position that resolves both questions. Consider a classifier
qθ(y|x) and measure its expected cross-entropy:
L(qθ) = Ex
"
−
X
y
p(y|x) log qθ(y|x)
#
(3)
This loss splits cleanly into two terms:
L(qθ) = Ex [H(p(y|x))]
|
{z
}
aleatoric
+ Ex [KL(p(y|x)∥qθ(y|x))]
|
{z
}
epistemic
(4)
The aleatoric term is the entropy of the true posterior aver-
aged over inputs. It reflects irreducible uncertainty, or the
ambiguity inherent in the data that no classifier can resolve.
An image of a cat-like dog has high H(p(y|x)) regardless
of model quality.
The epistemic term measures the gap between the model’s
beliefs and truth (Gal & Ghahramani, 2016). A perfect
model achieves KL = 0; any deviation indicates something
learnable that the model has not yet captured.
On standard benchmarks, we observe only total loss. When
it plateaus, we cannot tell whether the model has reached the
aleatoric floor or whether epistemic error remains. Our ora-
cle makes both terms measurable. This distinction matters:
if epistemic error dominates, more data or capacity should
help; if aleatoric error dominates, improvements require
changing the task itself.
4. Experiments
Setup.
We trained TarFlow (Zhai et al., 2025) on
AFHQ (Choi et al., 2020) (3 classes: cat, dog, wild; ∼15K
images) and ImageNet (Deng et al., 2009) at 64×64 res-
olution. Oracle quality is validated in Section 2.3. We
generated labeled samples with exact posteriors for all exper-
iments. We trained classifiers (ResNet-50 (He et al., 2015),
ViT-Base (Dosovitskiy et al., 2021), ConvNeXt (Liu et al.,
2022), Swin (Liu et al., 2021), MobileNetV3 (Howard et al.,
2019)) using default hyperparameters on varying amounts of
oracle data, from N=100 to N=10,000 (up to N=45, 000
for ImageNet-64) samples. All results are averaged over 3
seeds with standard deviation shown.
4.1. Scaling Laws Beyond the Loss Curve
Bottom line: Epistemic error follows a clean power law
in dataset size, continuing to shrink even when total loss
plateaus. Standard metrics hide this ongoing learning.
We trained classifiers on varying amounts of oracle
data and decomposed prediction error into aleatoric
and epistemic components (Kaplan et al., 2020) (Sec-
tion 3). For each model, we computed total cross-entropy
loss (what standard benchmarks measure), the aleatoric
component E[H(p(y|x))], and the epistemic component
E[KL(p(y|x)∥qθ(y|x))].
Figure 2 shows the result. Total cross-entropy loss decreases
with dataset size on a log-log scale, consistent with Kaplan
et al. (2020). MobileNet exhibits the highest loss throughout,
ResNet and ConvNeXt the lowest. But the total loss curve
obscures what is actually happening.
The epistemic component behaves differently. On a log-log
plot, it traces a nearly perfect straight line: KL ∝N −α
(dashed lines in Figure 2b) (Hoffmann et al., 2022). This
decay continues even when total loss appears to plateau,
because the aleatoric floor dominates the total at large N.
Standard metrics miss this ongoing learning because they
conflate the two error sources.
The scaling exponents α differ across architectures (Fig-
ure 2d): ResNet-50 (α=0.039 ± 0.007), ViT-Base (0.032 ±
0.010), ConvNeXt (0.030 ± 0.008), Swin (0.030 ± 0.005),
and MobileNetV3 (0.135 ± 0.025). These quantify how
efficiently each architecture converts additional data into
reduced uncertainty; MobileNet’s much higher α indicates
faster epistemic decay rate, yet it starts from a higher error
floor, suggesting a less efficient initial representation. Cali-
bration error (ECE) also decreases with N, but at a different
rate than epistemic error, confirming that calibration and pos-
terior approximation quality are distinct phenomena (Guo
et al., 2017).
The aleatoric floor.
On AFHQ, the aleatoric uncertainty
is exactly 0.012 nats (Bayes error 0.23%; see Section 2.3),
with mutual information I(X; Y ) = 1.577 bits (normalized
MI = 99.5%). On ImageNet-64, the aleatoric floor is 0.10
nats (Bayes error 4.031%), higher due to the greater number
of classes and inherent inter-class ambiguity.
ImageNet replication.
Figure 6 shows scaling laws on
ImageNet-64 up to 45K samples, a substantially more chal-
lenging dataset than AFHQ due to its fine-grained structure
and 1000-way label space. Despite this higher intrinsic en-
tropy, the total cross-entropy loss decreases rapidly with
dataset size, at a faster absolute rate than in AFHQ, rein-
forcing the classical finding that larger datasets continue to
yield measurable gains even when the absolute loss remains
dominated by intrinsic class complexity.
The aleatoric component remains effectively constant at
approximately 0.10 nats across all dataset sizes, confirming
that all models converge to the same irreducible uncertainty
induced by the oracle conditional distribution. As in AFHQ,
this floor is architecture-independent and invariant to dataset
size, reflecting the Bayes error of the oracle world rather
than model capacity. The higher value relative to AFHQ
(0.012 nats) is consistent with ImageNet-64’s greater inter-
4

[CAPTION] Figure 2 shows the result. Total cross-entropy loss decreases

[CAPTION] Figure 6 shows scaling laws on


<!-- page 5 -->
Decomposing Neural Network Error via Flow-Based Oracles
Figure 2. Epistemic error follows a power law even when total loss plateaus. (a) Total cross-entropy decreases with dataset size;
MobileNet shows the highest loss, ResNet/ConvNeXt the lowest. (b) Epistemic uncertainty (KL from oracle) follows N −α; dashed
lines are power-law fits. This decay continues even when total loss appears flat. (c) Accuracy improves from 92–96% at N=100 toward
97–98% at N=10,000. (d) Scaling exponents reveal architectural differences: MobileNet improves fastest (α=0.135), ViT stalls without
pretraining (α=0.032).
class overlap and multimodality.
Epistemic uncertainty decays with dataset size and cor-
roborates the power-law behaviour observed on AFHQ.
ImageNet-64, however, exhibits a brief plateau in the mid-
range (2,000-10,000 samples), reflecting a representation-
transition regime in which models do not immediately con-
vert additional data into reduced epistemic uncertainty. Be-
yond this regime, all architectures resume clear power-law
decay, reinforcing that epistemic uncertainty, alongside total
loss, is a distinct and reliable indicator of continued learning
at scale.
The resulting exponents show both similarities and diver-
gences relative to AFHQ. ResNet-50 exhibits the highest
scaling rate (α=0.019 ± 0.006), closely followed by ViT-
Base (0.018 ± 0.004), mirroring their behaviour on AFHQ.
ConvNeXt achieves only a weak decay rate (0.003 ± 0.003,
lower than ResNet and ViT-Base as in AFHQ), indicating
limited epistemic improvement per additional sample, likely
due to architectural biases that mismatch the ImageNet-64
feature geometry. Swin-T displays the strongest positive
exponent (0.027 ± 0.01), consistent with hierarchical trans-
formers benefiting disproportionately from larger, more di-
verse datasets.
MobileNet-V3 is an exception: its epistemic curve exhibits
a pronounced non-monotonicity. We attribute this to Mo-
bileNet’s compressed architecture and depthwise-separable
convolutions, which yield an inefficient initial representa-
tion incapable of faithfully capturing ImageNet-64’s high
inter-class diversity. At small dataset sizes, the model over-
fits and becomes overconfident, producing an artificially
low epistemic floor. As the dataset expands into the mid-
scale regime (≈2k–10k samples), this brittle representation
is forced to reorganize, triggering a spike in loss, calibra-
tion error, and epistemic uncertainty. Once past this regime,
MobileNet resumes clean power-law decay, confirming that
the underlying epistemic scaling law still holds once an
adequate representation is established.
Together with the monotonic rise in test accuracy, these
trends show that accuracy, calibration, and epistemic uncer-
tainty respond differently to data scaling, with calibration
showing partial signatures of the behaviour made explicit
by epistemic uncertainty.
Architectures differ in how they approach optimality.
Table 2 shows that ResNet-18 exhibits clean power-law
scaling: epistemic error drops from 0.16 to 0.026 as data
increases from N=100 to N=40,000 (a 6× reduction). ViT-
Base behaves differently: it starts competitive at N=100 but
5

[CAPTION] Figure 2. Epistemic error follows a power law even when total loss plateaus. (a) Total cross-entropy decreases with dataset size;

[CAPTION] Table 2 shows that ResNet-18 exhibits clean power-law


<!-- page 6 -->
Decomposing Neural Network Error via Flow-Based Oracles
Table 2. ResNets reduce epistemic error 6×; ViTs stall with-
out pretraining. Epistemic KL (nats) vs. dataset size on AFHQ.
ResNet exhibits clean power-law scaling from 0.16 to 0.026; ViT
improves by only 1.2× over the same range.
Dataset size N
ResNet-18
ViT-B-16
100
0.160
0.131
1,000
0.090
0.141
5,000
0.058
0.099
10,000
0.033
0.117
40,000
0.026
0.117
then stalls. By N=40,000, ViT has improved by less than
1.2×. Both architectures achieve similar total loss at large
N; the difference is in how they approach optimality. ViTs,
lacking pretraining, fail to extract geometric structure from
small samples (Dosovitskiy et al., 2021). This is consistent
with Dosovitskiy et al. (2021), but our framework quantifies
the gap in information-theoretic terms.
4.2. Soft Labels Enable Better Learning
Our oracle can generate exact posterior distributions p(y|x)
as training labels, not just the argmax class. Training with
these soft labels outperforms hard-label training at 4 out of
5 dataset sizes, with accuracy gains up to ∼1%. Models
trained on soft labels also achieve near-perfect calibration
(ECE = 0.018) (Guo et al., 2017), confirming the posteriors
encode learnable structure beyond class labels (Szegedy
et al., 2015; Hinton et al., 2015). Full results in Appendix D.
4.3. Distribution Shift: Exact Quantification via Oracle
Bottom line: The oracle computes exact KL divergence
of controlled perturbations. Class imbalance barely af-
fects accuracy at KL values where input noise causes
collapse. Shift type matters far more than shift magni-
tude.
On standard benchmarks, distribution shift is observed but
never measured: we see accuracy drop but cannot quantify
how much the distribution actually changed. Our oracle
computes KL[pshifted∥pbaseline] exactly by evaluating the true
log-likelihood log p(x|y) under both distributions. This
enables controlled experiments that are impossible on real
benchmarks: we introduce perturbations of known type and
magnitude, measure the exact divergence, and observe how
models degrade.
We introduce two types of controlled perturbation to the
oracle distribution:
• Class imbalance: Shifting the class prior from uni-
form to skewed ratios (e.g., 40/35/25 up to 70/20/10),
which increases KL divergence through the marginal
p(y) (Chawla et al., 2002).
• Gaussian noise: Adding noise with standard deviation
σ ∈{0.05, 0.10, 0.15} to generated images, which in-
creases KL divergence through the conditional p(x|y).
The results (Figure 3) show a clear asymmetry. Class im-
balance produces KL divergences from 0.018 to 0.293, yet
accuracy remains flat at 97.4–97.7%. Gaussian noise at
σ=0.15 produces a comparable KL divergence but causes
accuracy to collapse to ∼77%, a 20-percentage-point drop.
The same KL magnitude leads to dramatically different
outcomes depending on where the shift occurs.
Class imbalance shifts the marginal p(y) but leaves the
class-conditional p(x|y) intact, so learned features remain
discriminative. Gaussian noise corrupts p(x|y) directly, de-
stroying the features that classifiers depend on (Hendrycks
& Dietterich, 2019). A linear fit of KL divergence against
accuracy drop yields R2 = 0.042 (Figure 3f): aggregate di-
vergence does not predict robustness. What shifted matters
more than how much.
4.4. Active Learning with Ground-Truth Uncertainty
Active learning aims to select the most informative training
samples (Sener & Savarese, 2018). On standard bench-
marks, informativeness must be estimated via heuristics
(e.g., entropy of model predictions) (Lewis & Gale, 1994;
Gal et al., 2017). Our oracle provides exact epistemic un-
certainty, enabling a critical distinction: separating samples
that are epistemically uncertain (informative) from those
that are aleatorically uncertain (inherently ambiguous). On
standard benchmarks, a high-entropy sample could be either;
the oracle tells us which.
We compare three acquisition strategies on ImageNet-64:
Random, Max Entropy, and Max Epistemic (selecting by
exact epistemic uncertainty). Max Epistemic achieves a
1.6× larger accuracy gain than Max Entropy under the same
labeling budget, and requires 47.8% fewer labeled queries
to reach the same mid-regime performance. Max Epistemic
consistently outperforms both baselines, confirming that
predictive entropy is a poor proxy for information gain when
aleatoric noise is substantial. Full results and learning curves
appear in Appendix (Figure 9).
6

[CAPTION] Table 2. ResNets reduce epistemic error 6×; ViTs stall with-


<!-- page 7 -->
Decomposing Neural Network Error via Flow-Based Oracles
Figure 3. What shifts matters more than how much: KL magnitude alone poorly predicts performance (R2=0.04). (a) Test accuracy
vs. exact KL divergence: class imbalance (blue) maintains ∼97% accuracy; Gaussian noise (low points) collapses to ∼77% at comparable
KL. (b) Accuracy drop vs. KL: imbalance (red diamonds) clusters near zero regardless of divergence magnitude. (c) Prior shift alone
barely affects accuracy (97.4–97.7% across all imbalance levels). (d) Covariate shift causes exponential degradation (σ=0.15 →77%).
(e) Aggregated comparison confirms noise dominates. (f) Linear regression: R2 = 0.04 shows aggregate KL is uninformative.
5. Discussion
Implications for practice.
First, validation loss is a poor
proxy for learning progress: epistemic error continues to
shrink long after loss curves flatten. Practitioners using
early stopping may be leaving performance on the table.
Tracking epistemic proxies (e.g., ensemble disagreement) is
a better strategy, though in real settings these must be esti-
mated (Lakshminarayanan et al., 2017; Gal & Ghahramani,
2016).
Second, in small-data regimes without pretraining, convo-
lutional architectures are the better choice: ResNets reduce
epistemic error by 6× from N=100 to N=40K, while ViTs
improve by less than 1.2×.
Third, the soft label results suggest that access to posterior
information, even approximate, can meaningfully improve
classification. This connects to knowledge distillation (Hin-
ton et al., 2015): a strong teacher’s soft outputs contain struc-
ture beyond hard labels. Our oracle provides a controlled
setting to study this phenomenon with exact posteriors.
Fourth, a shift in p(x|y) (feature corruption) is far more
damaging than a shift in p(y) (prior change) (Hendrycks
& Dietterich, 2019; Chawla et al., 2002) at comparable KL
values. Robustness evaluations should characterize what
shifted, not just how much.
Limitations.
Our oracle defines a synthetic world, not
nature. Findings transfer to natural images only insofar as
the flow captures relevant statistical structure. Key concerns:
(1) domain gap between flow samples and real photographs;
(2) scale limitations (AFHQ has 3 classes; full ImageNet
experiments ongoing); (3) architectural conclusions may
depend on hyperparameters. We view this framework as
complementary to standard benchmarks: it provides ground
truth within a controlled setting, not claims about nature’s
true Bayes error.
6. Related Work
Neural scaling laws.
Kaplan et al. (2020) established that
loss decreases as a power law in data, compute, and parame-
ters. Subsequent work refined these laws for vision (Zhai
et al., 2022) and explored their limits. All such studies mea-
sure total loss, conflating aleatoric and epistemic error. Our
decomposition reveals that epistemic error alone follows a
power law, even when total loss appears flat.
Flow-based synthetic benchmarks with information-
theoretic control.
Hashmani et al. (2025) construct syn-
thetic multimodal datasets with controllable mutual informa-
7

[CAPTION] Figure 3. What shifts matters more than how much: KL magnitude alone poorly predicts performance (R2=0.04). (a) Test accuracy


<!-- page 8 -->
Decomposing Neural Network Error via Flow-Based Oracles
tion by combining causal latent-variable models with invert-
ible flow-based transformations, enabling rigorous bench-
marking of MI estimators and multimodal self-supervised
learning methods. Our work shares the use of normalizing
flows to build benchmarks with exact information-theoretic
quantities, but differs in focus: rather than controlling inter-
modality mutual information, we use flows as oracles to
decompose single-modality classifier error into aleatoric
and epistemic components via exact Bayes-optimal posteri-
ors.
Uncertainty estimation.
Bayesian deep learning and en-
semble methods aim to quantify epistemic uncertainty, but
without ground truth, evaluation is indirect (Alemi et al.,
2019). Calibration metrics compare to empirical frequen-
cies, not true posteriors (Guo et al., 2017). Our framework
provides the missing ground truth, enabling evaluation of
both calibration and posterior quality.
Knowledge distillation and soft labels.
Hinton et al.
(2015) showed that training on a teacher’s soft predictions
transfers “dark knowledge” beyond hard labels. Our soft-
label experiments connect to this literature but with a key
difference: our soft labels are exact posteriors, not approx-
imations from a teacher model. This provides an upper
bound on what soft-label training can achieve.
Distribution
shift.
Benchmarks
like
ImageNet-
C (Hendrycks & Dietterich, 2019) evaluate robustness to
distribution shift, but the magnitude of shift is unknown.
Our oracle enables controlled experiments with exact
divergence, revealing that shift type matters more than
magnitude.
Active learning.
Uncertainty-based acquisition func-
tions (Gal et al., 2017; Kirsch et al., 2019) select samples by
estimated informativeness. Our oracle separates epistemic
from aleatoric uncertainty exactly, providing an upper bound
on what uncertainty-based active learning can achieve.
Normalizing flows.
Flows have matured from Real-
NVP (Dinh et al., 2017) and Glow (Kingma & Dhariwal,
2018) to TarFlow (Zhai et al., 2025), which achieves strong
likelihoods on ImageNet. We use flows differently: not
as models to evaluate, but as oracles that define evaluation
itself.
Synthetic benchmarks.
Researchers have long used toy
distributions (Gaussians, moons, spirals) with known poste-
riors. These lack realism. Our approach combines realistic
images from AFHQ and ImageNet with the exact posterior
access of synthetic benchmarks.
7. Conclusion
We introduced a framework that treats normalizing flows
as oracles rather than models, enabling exact computation
of quantities that standard benchmarks can only estimate.
When we know the data-generating process, we can directly
measure what classifiers learn versus what remains funda-
mentally uncertain.
Our experiments reveal that epistemic error follows a clean
power law (KL ∝N −α) even when total loss plateaus:
models keep learning, but standard metrics miss it. This
power-law decay holds across architectures and scales to Im-
ageNet with 1000 classes. The decomposition also reveals
marked architectural differences: ResNets reduce epistemic
error by 6× over the data range where ViTs improve by
only 1.2×, suggesting convolutional inductive biases matter
most in low-data regimes.
Our distribution shift experiments show that what shifts
matters more than how much: class imbalance barely
affects accuracy at KL values where input noise causes
20-point drops. This finding implies that robustness eval-
uations should characterize the nature of shift, not just its
magnitude.
The oracle framework is not a replacement for real-world
benchmarks; it complements them by providing ground
truth within a controlled setting. This approach can inform
architecture selection, training decisions, and robustness
evaluation in ways that aggregate loss curves cannot. Code
and oracle models will be made available soon.
Impact Statement
This paper presents work whose goal is to advance the field
of Machine Learning by providing benchmarks with ex-
act information-theoretic ground truth.
Our framework
enables more rigorous evaluation of learning algorithms,
which should lead to better scientific understanding of deep
learning. The oracle datasets we generate are synthetic and
do not raise privacy concerns. There are many potential so-
cietal consequences of our work, none which we feel must
be specifically highlighted here.
Acknowledgments
This research was supported in part by the Digital Research
Alliance of Canada (DRAC), Calcul Qu´ebec, and the Prince-
ton Laboratory for Artificial Intelligence, under Award No.
2025-97. We also acknowledge Vellore Institute of Tech-
nology (VIT) for providing access to eight NVIDIA V100
GPUs, which supported the computational experiments re-
ported in this work. We are also gratful to Alexander A.
Alemi for a valuable discussion and suggestions.
8


<!-- page 9 -->
Decomposing Neural Network Error via Flow-Based Oracles
8. Contributions
Arian organized, planned, and led the project; developed
and implemented the core idea and codebase; refined the
architecture by improving key components and training
strategies; iterated on the dataset; and ran and iterated on all
experiments.
Nathaniel integrated all the experiments for the ImageNet-
128 dataset and contributed meaningfully to writing the
paper.
Yug integrated all the experiments for the ImageNet-64
dataset and contributed meaningfully to writing the paper.
Akshat contributed to the writing of the paper.
Egemen Advised the project in high-level.
Ravid advised the whole project from scratch, with feed-
back in all the stages, contributing to the writing of the paper,
and conceiving and pushing forward the research direction
in the all stages of the project.
References
Alemi, A. A., Fischer, I., Dillon, J. V., and Murphy,
K.
Deep Variational Information Bottleneck, Octo-
ber 2019.
URL http://arxiv.org/abs/1612.
00410. arXiv:1612.00410 [cs].
Belghazi, M. I., Baratin, A., Rajeswar, S., Ozair, S., Bengio,
Y., Courville, A., and Hjelm, R. D. MINE: Mutual Infor-
mation Neural Estimation, August 2021. URL http://
arxiv.org/abs/1801.04062. arXiv:1801.04062
[cs].
Chawla, N. V., Bowyer, K. W., Hall, L. O., and Kegelmeyer,
W. P. SMOTE: Synthetic Minority Over-sampling Tech-
nique. Journal of Artificial Intelligence Research, 16:
321–357, June 2002. ISSN 1076-9757. doi: 10.1613/jair.
953. URL http://arxiv.org/abs/1106.1813.
arXiv:1106.1813 [cs].
Choi, Y., Uh, Y., Yoo, J., and Ha, J.-W.
StarGAN
v2: Diverse Image Synthesis for Multiple Domains,
April 2020. URL http://arxiv.org/abs/1912.
01865. arXiv:1912.01865 [cs].
Chrabaszcz, P., Loshchilov, I., and Hutter, F. A Downsam-
pled Variant of ImageNet as an Alternative to the CIFAR
datasets, August 2017. URL http://arxiv.org/
abs/1707.08819. arXiv:1707.08819 [cs].
Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei,
L. ImageNet: A large-scale hierarchical image database.
In 2009 IEEE Conference on Computer Vision and Pat-
tern Recognition, pp. 248–255, June 2009. doi: 10.1109/
CVPR.2009.5206848. URL https://ieeexplore.
ieee.org/document/5206848.
ISSN: 1063-
6919.
Dinh, L., Sohl-Dickstein, J., and Bengio, S. Density esti-
mation using Real NVP, February 2017. URL http://
arxiv.org/abs/1605.08803. arXiv:1605.08803
[cs].
Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn,
D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer,
M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby,
N. An Image is Worth 16x16 Words: Transformers for
Image Recognition at Scale, June 2021. URL http://
arxiv.org/abs/2010.11929. arXiv:2010.11929
[cs].
Gal, Y. and Ghahramani, Z. Dropout as a Bayesian Approx-
imation: Representing Model Uncertainty in Deep Learn-
ing. In Proceedings of The 33rd International Conference
on Machine Learning, pp. 1050–1059. PMLR, June 2016.
URL https://proceedings.mlr.press/v48/
gal16.html.
Gal,
Y.,
Islam,
R.,
and Ghahramani,
Z.
Deep
Bayesian Active Learning with Image Data, March
2017.
URL
http://arxiv.org/abs/1703.
02910. arXiv:1703.02910 [cs].
Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q.
On Calibration of Modern Neural Networks, Au-
gust 2017. URL http://arxiv.org/abs/1706.
04599. arXiv:1706.04599 [cs].
He, K., Zhang, X., Ren, S., and Sun, J.
Deep
Residual Learning for Image Recognition, Decem-
ber 2015.
URL http://arxiv.org/abs/1512.
03385. arXiv:1512.03385 [cs].
Hendrycks, D. and Dietterich, T. Benchmarking Neural
Network Robustness to Common Corruptions and Per-
turbations, March 2019. URL http://arxiv.org/
abs/1903.12261. arXiv:1903.12261 [cs].
Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., and
Hochreiter, S. GANs Trained by a Two Time-Scale Up-
date Rule Converge to a Local Nash Equilibrium, Jan-
uary 2018. URL http://arxiv.org/abs/1706.
08500. arXiv:1706.08500 [cs].
Hinton, G., Vinyals, O., and Dean, J. Distilling the Knowl-
edge in a Neural Network, March 2015. URL http://
arxiv.org/abs/1503.02531. arXiv:1503.02531
[stat].
Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E.,
Cai, T., Rutherford, E., Casas, D. d. L., Hendricks, L. A.,
Welbl, J., Clark, A., Hennigan, T., Noland, E., Millican,
9


<!-- page 10 -->
Decomposing Neural Network Error via Flow-Based Oracles
K., Driessche, G. v. d., Damoc, B., Guy, A., Osindero,
S., Simonyan, K., Elsen, E., Rae, J. W., Vinyals, O., and
Sifre, L. Training Compute-Optimal Large Language
Models, March 2022.
URL http://arxiv.org/
abs/2203.15556. arXiv:2203.15556 [cs].
Howard, A., Sandler, M., Chu, G., Chen, L.-C., Chen, B.,
Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V.,
Le, Q. V., and Adam, H. Searching for MobileNetV3,
November 2019. URL http://arxiv.org/abs/
1905.02244. arXiv:1905.02244 [cs].
Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B.,
Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and
Amodei, D. Scaling Laws for Neural Language Mod-
els, January 2020. URL http://arxiv.org/abs/
2001.08361. arXiv:2001.08361 [cs].
Kingma, D. P. and Dhariwal, P. Glow: Generative Flow with
Invertible 1x1 Convolutions, July 2018. URL http://
arxiv.org/abs/1807.03039. arXiv:1807.03039
[stat].
Kirsch, A., Amersfoort, J. v., and Gal, Y. BatchBALD: Effi-
cient and Diverse Batch Acquisition for Deep Bayesian
Active Learning, October 2019. URL http://arxiv.
org/abs/1906.08158. arXiv:1906.08158 [cs].
Lakshminarayanan, B., Pritzel, A., and Blundell, C. Sim-
ple and Scalable Predictive Uncertainty Estimation us-
ing Deep Ensembles, November 2017. URL http://
arxiv.org/abs/1612.01474. arXiv:1612.01474
[stat].
Lewis, D. D. and Gale, W. A. A Sequential Algorithm for
Training Text Classifiers, July 1994. URL http://
arxiv.org/abs/cmp-lg/9407020.
arXiv:cmp-
lg/9407020.
Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z.,
Lin, S., and Guo, B.
Swin Transformer: Hierarchi-
cal Vision Transformer using Shifted Windows, Au-
gust 2021. URL http://arxiv.org/abs/2103.
14030. arXiv:2103.14030 [cs].
Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Dar-
rell, T., and Xie, S.
A ConvNet for the 2020s,
March 2022.
URL http://arxiv.org/abs/
2201.03545. arXiv:2201.03545 [cs].
Papamakarios, G., Nalisnick, E., Rezende, D. J., Mo-
hamed, S., and Lakshminarayanan, B.
Normaliz-
ing Flows for Probabilistic Modeling and Inference,
April 2021. URL http://arxiv.org/abs/1912.
02762. arXiv:1912.02762 [stat].
Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Rad-
ford, A., and Chen, X. Improved Techniques for Training
GANs, June 2016. URL http://arxiv.org/abs/
1606.03498. arXiv:1606.03498 [cs].
Sener, O. and Savarese, S.
Active Learning for Con-
volutional Neural Networks:
A Core-Set Approach,
June 2018. URL http://arxiv.org/abs/1708.
00489. arXiv:1708.00489 [stat].
Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., and Wojna,
Z. Rethinking the Inception Architecture for Computer
Vision, December 2015. URL http://arxiv.org/
abs/1512.00567. arXiv:1512.00567 [cs].
Zhai, S., Zhang, R., Nakkiran, P., Berthelot, D., Gu, J.,
Zheng, H., Chen, T., Bautista, M. A., Jaitly, N., and
Susskind, J. Normalizing Flows are Capable Generative
Models, June 2025. URL http://arxiv.org/abs/
2412.06329. arXiv:2412.06329 [cs].
Zhai, X., Kolesnikov, A., Houlsby, N., and Beyer, L. Scaling
Vision Transformers, June 2022. URL http://arxiv.
org/abs/2106.04560. arXiv:2106.04560 [cs].
Hashmani, R. K., Merz, G. W., et al.
Multimodal
Datasets with Controllable Mutual Information, Octo-
ber 2025.
URL http://arxiv.org/abs/2510.
21686. arXiv:2510.21686 [cs].
10


<!-- page 11 -->
Decomposing Neural Network Error via Flow-Based Oracles
A. Oracle Validation Details
We validated the oracle along five axes on both AFHQ and ImageNet-64. This appendix provides full details for the summary
in Section 2. We report quantitative checks verifying our Oracle provides (i) high-quality samples, (ii) broad support over
the data manifold, and (iii) stable posteriors for uncertainty decomposition.
A.1. Distribution Quality (FID)
We measured Fr´echet Inception Distance (FID) (Heusel et al., 2018), which compares the mean and covariance of Inception-
v3 features between real and generated images. Our AFHQ oracle achieves FID 28.44 and ImageNet-64 achieves FID 13.48,
comparable to state-of-the-art generative models on these datasets.
A.2. Manifold Coverage
A generative model might produce high-quality samples that only capture part of the true distribution (e.g., generating
realistic dogs but missing rare breeds). We computed manifold coverage by embedding both real and generated samples in a
pretrained feature space and measuring what fraction of real samples have a nearby generated sample within a distance
threshold. In our implementation, distances are computed in ResNet feature space using k-nearest neighbors and an adaptive
threshold (set to the 90th percentile of real-to-synthetic distances). Our oracle achieves 90% coverage on AFHQ and 89.9%
on ImageNet, suggesting it captures broad distributional structure rather than collapsing to a few high-density modes.
A.3. Diversity (Inception Score)
We evaluated diversity using Inception Score (IS), which measures the KL divergence between the conditional label distribu-
tion p(y|x) and the marginal p(y) averaged over generated samples. Higher scores indicate both confident classifications
and diverse outputs. We achieve 6.24 ± 3.57 on AFHQ and 32.57 ± 4.47 on ImageNet. Per-class diversity ratios (variance
ratio and distance ratio between synthetic and real) range from 0.75 to 0.96 on AFHQ, confirming reasonable diversity that
is slightly below real data. Across the three classes, the synthetic-to-real feature variance ratios are {0.746, 0.854, 0.921}
and the average pairwise distance ratios are {0.891, 0.944, 0.965}, indicating slightly reduced diversity relative to real data,
but without severe collapse
A.4. Semantic Feature Alignment
Pixel-level metrics do not capture whether images have the right semantic content. We extracted features from a pretrained
ResNet, which encode high-level structure like shape and pose, and compared statistics between real and generated images.
Feature variance matches at 83–92% across classes on AFHQ, with Class 0 (cat) performing best (92%) and Classes 1–2
(dog, wild) showing larger distributional separation but still acceptable overlap. ImageNet-64 results show 72–93% feature
variance match across classes.
A.5. Memorization Check
If the flow memorizes training images, our benchmark would be meaningless. We computed nearest-neighbor distances
in Inception feature space between each generated image and the training set. 36% of generated samples on AFHQ have
a training neighbor within distance 10 in feature space; 6.53% on ImageNet. This thresholded “near-neighbor” rate is a
conservative proxy for potential memorization rather than direct evidence of exact duplication. Visual inspection confirms
that “close” pairs share high-level attributes (pose, lighting) but depict different individuals. The nearest-neighbor distance
distribution is well-separated from zero, with the memorization threshold clearly above the bulk of the distribution; (See
Figure 4).
A.6. Texture Analysis
Generated images are slightly smoother than real photographs: pixel-level texture variance is 56–69% of real images across
all classes. This is a known property of flow-based models, which tend to soften fine-grained details while preserving global
structure. For our purposes, this is acceptable because our downstream classifiers primarily rely on semantic cues (shape,
pose, object parts) rather than fine-grained pixel texture. Since this smoothing effect is systematic across the oracle-generated
dataset, comparative analyses across architectures remain valid.
11


<!-- page 12 -->
Decomposing Neural Network Error via Flow-Based Oracles
Dataset
FID↓
IS↑
Coverage↑
Mem. rate
NN dist. (med)
Post. stab. KL↓
AFHQ-256
28.44
6.24 ± 3.57
0.900
0.363
11.01
1.51 × 10−9
Table 3. Oracle passes all validation checks: high coverage, low memorization, stable posteriors. FID measures distributional
quality; coverage confirms broad manifold support (90%); memorization rate shows most samples are novel; posterior stability KL (10−9)
confirms robustness to small perturbations.
Table 4. Oracle samples match real data statistics across multiple metrics. FID, Inception Score, and manifold coverage confirm
distributional quality; low memorization rates verify sample novelty. Dashes indicate metrics not yet computed.
Metric
AFHQ
ImageNet-64
FID ↓
28.44
13.48
Inception Score ↑
6.24 ± 3.57
32.57 ± 4.47
Manifold coverage
90%
89.9%
Feature variance match
83–92%
72–93%
Texture variance ratio
56–69%
—
Feature variance ratio match
—
71.6–93.3%
Memorization rate
36%
6.5%
Figure 4. Oracle samples are not memorized from training data. Distribution of nearest-neighbor distances (ResNet feature space)
between generated and training images. The dashed line marks the memorization threshold (d=10); the bulk of distances lie well above,
confirming sample novelty.
12

[CAPTION] Table 3. Oracle passes all validation checks: high coverage, low memorization, stable posteriors. FID measures distributional

[CAPTION] Table 4. Oracle samples match real data statistics across multiple metrics. FID, Inception Score, and manifold coverage confirm

[CAPTION] Figure 4. Oracle samples are not memorized from training data. Distribution of nearest-neighbor distances (ResNet feature space)


<!-- page 13 -->
Decomposing Neural Network Error via Flow-Based Oracles
B. TarFlow Architecture
We train TarFlow models from scratch on AFHQ at 256×256 and class-conditional ImageNet at 64×64 and 128×128,
following the architecture introduced by Zhai et al. (2025). All configurations use an autoregressive Transformer backbone
with 8 blocks and 8 flow layers per block. For all datasets, we use the same optimizer settings (learning rate 10−4) and
label-dropout of 0.1 for class-conditional training. Input dequantization is performed by adding Gaussian noise with
dataset-specific standard deviation σ (Table 5). We compute and cache dataset statistics required for FID evaluation using
the corresponding ground-truth training distribution at each resolution prior to sampling and evaluation.
Training and evaluation protocol.
We train using distributed data-parallelism with dataset-dependent GPU counts (8
GPUs for AFHQ 2562 and ImageNet 642; 32 GPUs for ImageNet 1282). During training, we periodically generate samples
and evaluate FID using cached ground-truth statistics computed at the matching resolution. For conditional generation at
evaluation time, we additionally report classifier-free guidance (CFG) results by sampling with a nonzero guidance scale
(e.g., γ = 2.3 for ImageNet 642), while keeping guidance disabled during training. We emphasize that all reported ImageNet
and AFHQ models are trained from scratch using the above protocol.
13


<!-- page 14 -->
Decomposing Neural Network Error via Flow-Based Oracles
Hyperparameter
AFHQ 2562
ImageNet 642
ImageNet 1282
Conditioning
Class-conditional
Class-conditional
Class-conditional
Image size
256
64
128
Channels (C)
768
768
1024
Patch size
8
2
4
# Transformer blocks
8
8
8
Flow layers / block
8
8
8
Gaussian noise std (σ)
0.07
0.05
0.15
Learning rate
1 × 10−4
1 × 10−4
1 × 10−4
Batch size
256
256
768
Epochs
4000
200
320
Label dropout (p)
0.1
0.1
0.1
CFG during training
0 (disabled)
0 (disabled)
0 (disabled)
Table 5. TarFlow hyperparameters for reproducibility. All models use 8 Transformer blocks with 8 flow layers each. Gaussian noise
refers to dequantization noise during training.
C. Full Scaling Results
Figure 5. Full scaling results on AFHQ. Epistemic uncertainty follows power-law decay across all architectures, with MobileNet showing
the steepest decline and ViT the shallowest.
14

[CAPTION] Table 5. TarFlow hyperparameters for reproducibility. All models use 8 Transformer blocks with 8 flow layers each. Gaussian noise

[CAPTION] Figure 5. Full scaling results on AFHQ. Epistemic uncertainty follows power-law decay across all architectures, with MobileNet showing


<!-- page 15 -->
Decomposing Neural Network Error via Flow-Based Oracles
Figure 6. Scaling laws extend to ImageNet-64 with 1000 classes. Power-law epistemic decay holds at scale, though MobileNet shows a
mid-range transition regime before resuming clean scaling.
D. Soft Labels: Full Results
Our oracle can generate exact posterior distributions p(y|x) as training labels, not just the argmax class. We compare
training with hard labels (one-hot from the most likely class) versus soft labels (the full posterior vector) across dataset sizes
from N=100 to N=5,000.
Figure 7 shows that training with oracle soft labels outperforms hard-label training at 4 out of 5 dataset sizes, with accuracy
gains up to ∼1%. The one exception is N=500, where hard labels slightly outperform.
This result validates oracle quality from a different angle: the soft posteriors contain learnable information beyond the
class label. If the oracle were merely assigning noisy labels, soft training would not consistently outperform. Instead, the
posteriors encode genuine uncertainty structure (inter-class similarities, ambiguous regions, confidence gradients) that
classifiers can exploit. This form of supervision is unavailable on any standard benchmark, where labels are always hard.
Models trained on soft labels also achieve near-perfect calibration (ECE = 0.018), learning the full uncertainty landscape
rather than just decision boundaries.
15

[CAPTION] Figure 6. Scaling laws extend to ImageNet-64 with 1000 classes. Power-law epistemic decay holds at scale, though MobileNet shows a

[CAPTION] Figure 7 shows that training with oracle soft labels outperforms hard-label training at 4 out of 5 dataset sizes, with accuracy


<!-- page 16 -->
Decomposing Neural Network Error via Flow-Based Oracles
Figure 7. Exact soft labels outperform hard labels at most dataset sizes. Top: Training curves show soft labels (blue) tracking above
hard labels (red) across three dataset sizes. Bottom left: Final accuracy vs. N; soft labels win at 4 of 5 sizes. Bottom right: Accuracy
gain reaches ∼1%, confirming oracle posteriors encode learnable structure beyond class labels.
E. Full Active Learning Results
Figure 8. Epistemic-based acquisition reduces uncertainty fastest on AFHQ. (a) Learning curves: Max Epistemic (red) initially dips
while selecting challenging samples, then recovers. (b) Epistemic KL reduction: Max Epistemic achieves the steepest decline, confirming
it targets genuinely informative samples. (c) Final accuracy: all methods converge to ∼97–98% on this 3-class dataset.
16

[CAPTION] Figure 7. Exact soft labels outperform hard labels at most dataset sizes. Top: Training curves show soft labels (blue) tracking above

[CAPTION] Figure 8. Epistemic-based acquisition reduces uncertainty fastest on AFHQ. (a) Learning curves: Max Epistemic (red) initially dips


<!-- page 17 -->
Decomposing Neural Network Error via Flow-Based Oracles
Figure 9. Max Epistemic consistently outperforms entropy-based selection on ImageNet-64. Selecting by exact epistemic uncertainty
(orange) beats both random (gray) and max entropy (blue), confirming that predictive entropy conflates aleatoric and epistemic components
while our decomposition isolates the informative signal.
F. Computing Posteriors at Scale
The bijective invertability of normalizing flow models allow us to independently calculate samples in parallel. Posteriors are
obtained by iteratively cycling through smaller batches of classes per batch of image across for all possible classes. Since
the results can quickly accumulate, we store the outputs of the image batch until a given size, then save the combined output
as a shard for each GPU. For ImageNet64, this process has been split across 7 V100 GPUs for a total of 4,200 samples. And
for ImageNet128, this process has been split across 4 H200 GPUs for a total of 16,800 samples. For the Imagenet datasets,
we iteratively accumulate 50 class batches per image batch which we find to be a good balance for memory efficiency.
One notable difference between the models is that OracleFlow using the AFHQ 256x256 model uses a smoothing temperature
of 500 across 3 classes, compared to ImageNet 64x64 and ImageNet 128x128 which uses a smoothing temperature of 1 due
to ImageNet’s much higher class size of 1000.
G. Distribution Shift Experimental Details
We are providing the full specification of controlled distribution-shift protocol used to stress-test supervised learners trained
on oracle-generated data. We generate training-time shifts via two orthogonal knobs: (i) label-prior shift (class imbalance)
and (ii) covariate shift via additive Gaussian noise. All models are evaluated on the same held-out baseline test set to isolate
the impact of training distribution mismatch.
G.1. Base dataset and notation
Let D0 = {(xi, yi)}Npool
i=1 denote a fixed pool of labeled images stored in the oracle raw range x ∈[−1, 1] with K = 3 classes.
We also fix a baseline test set Dtest
0
(size Ntest = 2000) sampled once from the same source and used for all evaluations.
Baseline label distribution.
We define the baseline label prior as uniform:
P0(y = k) = 1
K
(K = 3).
(5)
G.2. Shift family: label-prior shift and noise perturbations
Each shifted training distribution is parameterized by (π, σ), where π = (π0, π1, π2) specifies the target class prior and
σ ≥0 controls Gaussian covariate noise.
(A) Label-prior (class-imbalance) shift.
We sample class counts via multinomial draw:
(n0, n1, n2) ∼Multinomial(Ntrain, π) ,
(6)
then construct the shifted training set by sampling nk examples from the pool restricted to class k. If nk exceeds the
available pool size for class k, we sample with replacement (this matches the implementation).
17

[CAPTION] Figure 9. Max Epistemic consistently outperforms entropy-based selection on ImageNet-64. Selecting by exact epistemic uncertainty


<!-- page 18 -->
Decomposing Neural Network Error via Flow-Based Oracles
Table 6. Distribution-shift training protocol. All experiments use identical hyperparameters; only the training distribution varies.
Component
Setting
Base architecture
ResNet-50 pretrained on ImageNet-1K
Classifier head
Replace final FC with K = 3 outputs
Training size
Ntrain = 5000 (shifted)
Validation split
80/20 split of the shifted training set
Test set
Baseline test set Dtest
0 , Ntest = 2000
Optimizer
AdamW (weight decay 0.01)
Learning rate
10−4
Schedule
Cosine annealing over 40 epochs
Batch size
32
Seeds
S = 3
Metric
Test accuracy on Dtest
0 ; error rate = 1 −acc
(B) Covariate shift via additive Gaussian noise.
Given a sampled image x ∈[−1, 1], we generate a noised view:
x′ = clip (x + ε, −1, 1) ,
ε ∼N(0, σ2I).
(7)
Noise is applied after selecting examples according to π and before normalization.
Normalization for training.
Models are trained using ImageNet-style normalization. Concretely, we map x′ ∈[−1, 1] to
[0, 1] via (x′ + 1)/2 and then apply per-channel mean/std normalization.
G.3. KL computation methodology
We report a label-marginal KL that quantifies the strength of the prior shift:
KLy(Pshift(y) ∥P0(y)) =
K
X
k=1
ˆπk log ˆπk
1/K ,
(8)
where ˆπk is the empirical class frequency in the constructed shifted training set (i.e., computed from the realized multinomial
sample and any resampling-with-replacement), and log is the natural logarithm (units: nats).
Important implication.
KLy is insensitive to pure covariate shifts induced by σ when class priors remain balanced.
Therefore, we always report both (σ, KLy): KLy measures label shift strength, while σ measures the covariate-noise shift
strength.
G.4. Per-experiment protocol (fixed across all shift settings)
Each shift setting (π, σ) defines one training distribution. We repeat each experiment over S = 3 random seeds (affecting
multinomial draw, sampling, and optimization).
G.5. Shift configurations and measured KLy
Table 7 enumerates all perturbations. We report the target KLy implied by π (using Eq. 8 with ˆπ = π), and the empirical
KLy computed from realized label frequencies (mean ± std over seeds).
G.6. Results: performance degradation under training-time shift
Table 8 reports accuracy on the baseline test set Dtest
0
for each perturbation, averaged over three seeds. We also report
∆TestAcc (percentage points, pp) relative to the balanced baseline, and the corresponding test error rate.
Interpretation (high-level).
Across the tested range, label-prior shift alone (up to KLy ≈0.29 nats) causes only minor
changes in baseline test accuracy (sub-0.3pp on average). In contrast, covariate noise drives substantially larger degradation,
with σ = 0.10 inducing ≈2pp drops and σ = 0.15 leading to severe and high-variance failures. This separation is expected
because KLy captures label shift only, while σ controls covariate shift strength.
18

[CAPTION] Table 6. Distribution-shift training protocol. All experiments use identical hyperparameters; only the training distribution varies.

[CAPTION] Table 7 enumerates all perturbations. We report the target KLy implied by π (using Eq. 8 with ˆπ = π), and the empirical

[CAPTION] Table 8 reports accuracy on the baseline test set Dtest


<!-- page 19 -->
Decomposing Neural Network Error via Flow-Based Oracles
Table 7. Controlled shift configurations with exact KL values. We vary class prior π (label shift) and noise σ (covariate shift)
independently. Empirical KL matches target values closely.
Configuration
π = (π0, π1, π2)
σ
KLy (target)
KLy (empirical)
Balanced (Baseline)
(0.33, 0.33, 0.34)
0.00
1.00×10−4
1.27×10−4 ± 1.56×10−4
Mild Imbalance (40/35/25)
(0.40, 0.35, 0.25)
0.00
0.017
0.018 ± 0.001
Moderate Imbalance (50/30/20)
(0.50, 0.30, 0.20)
0.00
0.069
0.070 ± 0.003
Strong Imbalance (60/25/15)
(0.60, 0.25, 0.15)
0.00
0.161
0.158 ± 0.007
Very Strong Imbalance (70/20/10)
(0.70, 0.20, 0.10)
0.00
0.296
0.293 ± 0.009
Balanced + Noise σ=0.05
(0.33, 0.33, 0.34)
0.05
1.00×10−4
1.27×10−4 ± 1.56×10−4
Balanced + Noise σ=0.10
(0.33, 0.33, 0.34)
0.10
1.00×10−4
1.27×10−4 ± 1.56×10−4
Balanced + Noise σ=0.15
(0.33, 0.33, 0.34)
0.15
1.00×10−4
1.27×10−4 ± 1.56×10−4
Imbalance (50/30/20) + Noise σ=0.05
(0.50, 0.30, 0.20)
0.05
0.069
0.070 ± 0.003
Imbalance (70/20/10) + Noise σ=0.10
(0.70, 0.20, 0.10)
0.10
0.296
0.293 ± 0.009
Table 8. Noise causes 20-point accuracy drops; imbalance causes <0.3 points. All models evaluated on the same baseline test set.
Label-prior shift (up to KL=0.29) barely affects accuracy; covariate noise (σ=0.15) causes catastrophic degradation.
Configuration
Test Acc (%)
∆TestAcc (pp)
Test Error (%)
Val Acc (%)
Balanced (Baseline)
97.67 ± 0.10
+0.00
2.33 ± 0.10
98.20 ± 0.26
Mild Imbalance (40/35/25)
97.58 ± 0.10
-0.08
2.42 ± 0.10
98.10 ± 0.44
Moderate Imbalance (50/30/20)
97.45 ± 0.13
-0.22
2.55 ± 0.13
98.40 ± 0.60
Strong Imbalance (60/25/15)
97.40 ± 0.09
-0.27
2.60 ± 0.09
98.70 ± 0.20
Very Strong Imbalance (70/20/10)
97.38 ± 0.20
-0.28
2.62 ± 0.20
98.80 ± 0.26
Balanced + Noise σ=0.05
97.58 ± 0.12
-0.08
2.42 ± 0.12
98.13 ± 0.35
Balanced + Noise σ=0.10
95.70 ± 0.87
-1.97
4.30 ± 0.87
98.00 ± 0.35
Balanced + Noise σ=0.15
76.92 ± 13.02
-20.75
23.08 ± 13.02
98.03 ± 0.42
Imbalance (50/30/20) + Noise σ=0.05
97.03 ± 0.38
-0.63
2.97 ± 0.38
98.80 ± 0.44
Imbalance (70/20/10) + Noise σ=0.10
95.65 ± 1.03
-2.02
4.35 ± 1.03
99.27 ± 0.58
Table 9. Isolating shift axes: prior shift is benign, covariate shift is catastrophic. Left: varying class imbalance with no noise. Right:
varying noise with balanced classes. The asymmetry is stark.
Prior shift only (σ=0)
Noise shift only (balanced priors)
Config
KLy
Test Acc (%)
∆(pp)
σ
Test Acc (%)
∆(pp)
Balanced
≈0
97.67 ± 0.10
+0.00
0.05
97.58 ± 0.12
-0.08
40/35/25
0.018
97.58 ± 0.10
-0.08
0.10
95.70 ± 0.87
-1.97
50/30/20
0.070
97.45 ± 0.13
-0.22
0.15
76.92 ± 13.02
-20.75
60/25/15
0.158
97.40 ± 0.09
-0.27
—
—
—
70/20/10
0.293
97.38 ± 0.20
-0.28
—
—
—
19


**[Table p19.1]**
| Prior shift only (σ=0) Config KL Test Acc (%) ∆(pp) y | Noise shift only (balanced priors) σ Test Acc (%) ∆(pp) |
| --- | --- |


**[Table p19.2]**
| Balanced ≈0 97.67 ± 0.10 +0.00 40/35/25 0.018 97.58 ± 0.10 -0.08 50/30/20 0.070 97.45 ± 0.13 -0.22 60/25/15 0.158 97.40 ± 0.09 -0.27 70/20/10 0.293 97.38 ± 0.20 -0.28 | 0.05 97.58 ± 0.12 -0.08 0.10 95.70 ± 0.87 -1.97 0.15 76.92 ± 13.02 -20.75 — — — — — — |
| --- | --- |

[CAPTION] Table 7. Controlled shift configurations with exact KL values. We vary class prior π (label shift) and noise σ (covariate shift)

[CAPTION] Table 8. Noise causes 20-point accuracy drops; imbalance causes <0.3 points. All models evaluated on the same baseline test set.

[CAPTION] Table 9. Isolating shift axes: prior shift is benign, covariate shift is catastrophic. Left: varying class imbalance with no noise. Right: