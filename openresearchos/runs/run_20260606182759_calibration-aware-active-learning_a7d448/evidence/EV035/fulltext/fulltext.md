<!-- page 1 -->
Generative and Nonparametric Approaches for
Conditional Distribution Estimation: Methods,
Perspectives, and Comparative Evaluations
Yen-Shiu Chin1 , Zhi-Yu Jou1, Toshinari Morimoto2, Chia-Tse Wang3,
Ming-Chung Chang1, Tso-Jung Yen1, Su-Yun Huang1, and Tailen Hsing4
1Institute of Statistical Science, Academia Sinica, Taipei, Taiwan
2Department of Mathematics, National Taiwan University, Taipei, Taiwan
3Data Science Degree Program, National Taiwan University and Academia Sinica, Taipei, Taiwan
4Department of Statistics, University of Michigan, Ann Arbor, USA
Abstract
The inference of conditional distributions is a fundamental problem in statistics, essential
for prediction, uncertainty quantification, and probabilistic modeling.
A wide range of
methodologies have been developed for this task. This article reviews and compares several
representative approaches spanning classical nonparametric methods and modern generative
models. We begin with the single-index method of Hall and Yao (2005), which estimates the
conditional distribution through a dimension-reducing index and nonparametric smoothing
of the resulting one-dimensional cumulative conditional distribution function.
We then
examine the basis-expansion approaches, including FlexCode (Izbicki and Lee, 2017) and
DeepCDE (Dalmasso et al., 2020), which convert conditional density estimation into a
set of nonparametric regression problems.
In addition, we discuss two recent generative
simulation-based methods that leverage modern deep generative architectures: the generative
conditional distribution sampler (Zhou et al., 2023) and the conditional denoising diffusion
probabilistic model (Fu et al., 2024; Yang et al., 2025). A systematic numerical comparison
of these approaches is provided using a unified evaluation framework that ensures fairness
and reproducibility. The performance metrics used for the estimated conditional distribution
include the mean-squared errors of conditional mean and standard deviation, as well as the
Wasserstein distance. We also discuss their flexibility and computational costs, highlighting
the distinct advantages and limitations of each approach.
Key words and phrases: Conditional distribution estimation, Deep learning, Diffusion
models, Dimension reduction, Generative modeling, Wasserstein distance.
1
arXiv:2601.22650v1  [stat.ML]  30 Jan 2026


<!-- page 2 -->
1
Introduction
The inference of conditional distributions and
related functionals (e.g., mean, quantiles, vari-
ance, etc.) is a fundamental statistical task un-
derlying prediction, uncertainty quantification,
and probabilistic modeling. Numerous classical
and modern methods exist for the general prob-
lem of conditional distribution inference. How-
ever, comparisons across the diverse approaches,
particularly between the more traditional non-
parametric and the newer generative simulation-
based methods remain limited. This article re-
views some representative approaches and pro-
vides a systematic comparison.
Let (X, Y ) denote a random pair with X ∈
X
⊆Rp and Y
∈Y
⊆Rq for p,
q ≥
1.
We assume that the joint distribution of
(X, Y ) is FX,Y , and denote the corresponding
marginal distributions by FX and FY .
Let
FY |X(· | x) represent the distribution of Y
conditional on X = x, and fY |X(· | x) denote
its corresponding density.
We shall use the
non-boldface notation Y in place of Y when
referring explicitly to the univariate case Y ⊆R.
A key theoretical foundation for conditional
distribution estimation is the so-called Noise-
Outsourcing
Lemma
(see,
e.g.,
Kallenberg
(2021, Lemma 4.22 and Chapter 8), Austin
(2015, Lemma 3.1)), which states that if X
and Y are standard Borel spaces, then on an
enlarged probability space there exist a ran-
dom variable U ∼Uniform[0, 1) and a Borel-
measurable function G : [0, 1) × X →Y, where
U and X are independent, such that
Y = G(U, X) a.s.
(1)
Thus, the conditional distribution of Y | X =
x can be represented as the distribution of
G(U, x), where U encodes the variability in Y
that remains unexplained by X. Note, however,
that both G and U are generally non-unique.
This lemma also reveals a close connection to
the classical nonparametric regression model:
Y = g(X) + ε,
(2)
where randomness beyond the conditional mean
is absorbed by a zero-mean noise term ε, and
the main goal is to infer g(x) = E(Y | X =
x). Although the study of (2) occupies a cen-
tral place in the statistical literature, our focus
here is not on nonparametric regression itself.
Since the conditional mean is a statistical func-
tional of the conditional distribution, a condi-
tional distribution estimation method can, in
principle, be adapted to conditional mean esti-
mation; see FlexCode (Izbicki and Lee, 2017)
and DeepCDE (Dalmasso et al., 2020) in Sec-
tion 2 for examples on this for the univariate
case Y ∈R. Importantly, the representation (1)
provides the theoretical underpinning for gener-
ative simulation-based approaches, which learn
the mapping G(U, X) directly to model the con-
ditional distribution of Y | X. These ideas mo-
tivate the two generative methods reviewed in
Section 2.
The traditional literature on the conditional
distribution inference problem primarily fo-
cused on settings with small p and q.
Hall
et al. (1999) considered an adjusted Nadaraya-
Watson estimator of the conditional cumulative
distribution function. Yu and Jones (1998) and
Spokoiny et al. (2013) considered the inference
of the quantile function of the conditional distri-
bution. Dalmasso et al. (2020) introduced sev-
eral approaches on the estimation of conditional
density, including a nearest-neighbor-based ker-
nel density estimator and an approach based on
basis expansion that turns the estimation of the
conditional density into a series of nonparamet-
ric regression problems. These approaches are
kernel based and as such are sensitive to the
ambient dimensions of both X and Y.
Some recent studies
have developed
ap-
proaches that are more flexible with respect to
p, the dimension of X. For example, Hall and
Yao (2005) introduced a dimension-reduction
approach for models where the conditional dis-
tribution of a scalar random variable Y given
X = x depends approximately on a single in-
dex v⊤x for some direction v, thereby poten-
2


<!-- page 3 -->
tially overcoming some of the challenges posed
by large p.
This idea has the same flavor
as some classical dimension-reduction strategies
in regression; see also Henzi et al. (2023) for
a related approach that combines the ideas
of single-index models and isotonic regression.
The basis-expansion and nonparametric regres-
sion approach of Dalmasso et al. (2020) does not
directly address the curse of dimensionality due
to large p. However, their deep-neural-network
implementation (DeepCDE) may achieve a simi-
lar effect, as deep networks can implicitly learn
lower-dimensional representations of the predic-
tor space (see, e.g., Hinton and Salakhutdinov,
2006; Bauer and Köhler, 2019; Schmidt-Hieber,
2020).
Recently, two generative simulation-based ap-
proaches were proposed in Zhou et al. (2023),
and Fu et al. (2024) and Yang et al. (2025).
Both methods take advantage of recent ad-
vances in deep neural networks. The method
of Zhou et al. (2023), known as the Genera-
tive Conditional Distribution Sampler (GCDS),
estimates the mapping G in (1) by training a
neural network bG that minimizes the empiri-
cal Kullback–Leibler divergence between a can-
didate joint distribution and the observed data
distribution.
Then, for a given x, the condi-
tional distribution of Y | X = x is simulated
by generating samples of a latent noise variable
(e.g., a multivariate Gaussian vector η; see Sec-
tion 2.3) and evaluating bG(η, x). In contrast,
Fu et al. (2024) and Yang et al. (2025) con-
sidered the simulation of the conditional dis-
tribution using the conditional diffusion model,
which is a conditional extension of the denoising
diffusion probabilistic model (DDPM; Ho et al.,
2020) and will be referred to as the conditional
DDPM. Both of these offer greater flexibility
with respect to the dimensions p and q than
kernel-based approaches and are capable of ef-
ficient estimation of various functionals of the
conditional distribution.
In this article, we review and provide a sys-
tematic numerical comparison of four represen-
tative approaches for conditional distribution es-
timation: (i) the single-index model approach
(Hall and Yao, 2005), (ii) basis-expansion meth-
ods including FlexCode (Izbicki and Lee, 2017)
and DeepCDE (Dalmasso et al., 2020), (iii) the
distribution-matching generative model GCDS
(Zhou et al., 2023), and (iv) the conditional
diffusion-based generative model (Fu et al.,
2024; Yang et al., 2025). All four approaches
can potentially be applied in the setting of high-
dimensional predictors. The approaches (i) and
(ii) are selected for their superior flexibility and
effectiveness among kernel and related nonpara-
metric estimation methods. For (iii) and (iv),
since a substantial body of research has already
examined how well generative models perform
in ultra high-dimensional data contexts, our
aim here is to study their performance in scenar-
ios where the dimensionality is more moderate
and kernel-based methods have been commonly
used.
The remainder of the article is organized as
follows. In Section 2, we briefly review the four
representative approaches, (i)-(iv), mentioned
above, In Section 3, we conduct a comprehen-
sive set of simulation studies to compare the
effectiveness of these methods. Section 4 offers
additional discussion and concluding remarks.
To facilitate systematic evaluations and en-
sure reproducibility, we developed a unified
Python framework that integrates our imple-
mentations with the original software of the au-
thors when available, including packages pre-
viously released only in R or not publicly re-
leased. The full codebase is publicly accessible
on GitHub.1
2
A review of four classes
of methods
This section reviews several methods for es-
timating conditional distributions or densities,
1https://github.com/chiatsewang/
generative-nonparametric-cde
3


<!-- page 4 -->
whose empirical performance will be compared
in the simulation studies of Section 3.
The
approaches considered here include those in-
troduced in Hall and Yao (2005), Izbicki and
Lee (2017), Dalmasso et al. (2020), Zhou et al.
(2023), and Fu et al. (2024), and Yang et al.
(2025). One of the criteria for selecting these
methods is their potential to accommodate high-
dimensional predictors.
2.1
Hall and Yao: a
dimension-reduction
approach
We begin with the dimension-reduction ap-
proach of Hall and Yao (2005), developed for
scalar responses Y ∈R with predictors X ∈Rp.
The basic idea is to approximate the conditional
cumulative distribution function FY |X(y | x) by
a single-index model of the form
F(y | v⊤x) := P(Y ≤y | v⊤X = v⊤x)
for some unit vector v that is not known a pri-
ori. This unknown v represents the direction in
the predictor space that captures the dominant
dependence of Y on X. For certain models, this
approximation is exact, while for others it pro-
vides a useful low-dimensional approximation of
the conditional distribution. A key question is
how to estimate the direction vector v. Hall and
Yao (2005) proposed minimizing a discrepancy
measure between the true conditional probabil-
ity and its single-index approximation, briefly
explained below.
Let A be a Borel set of Rp and define
F(A, y) = P{(X, Y ) ∈A × (−∞, y]};
also, let
Hv(A, y) =
Z
A
F
 y | v⊤x
 
dFX(x).
If for some v and all x, the single-index approx-
imation F(y | v⊤x) ≈FY |X(y | x) holds well,
then F(A, y) and Hv(A, y) should be close for a
rich class of A’s. Hall and Yao (2005) focused on
the class Q = {δ : Aδ ⊆R}, where each Aδ is a
p-dimensional sphere contained within a given
fixed set R ⊆Rp, indexed by a (p+1)-vector δ,
whose first p components denote the center and
whose last component specifies the radius.
Suppose
we
know
FY (y), F(A, y),
and
Hv(A, y). We can compute the optimal v by
minimizing the following criterion over v ∈Θ,
where Θ denotes the set of p-dimensional
unit vectors whose first nonzero component is
positive:
S0(v) =
Z
Q
Z ∞
−∞
{F (Aδ, y) −Hv (Aδ, y)}2 dFY (y) dδ.
(3)
However, in general, the functions in (3) are
unknown. In practice, S0(v) is approximated
by
S(v) =
Z
Q
S (v, Aδ) dδ,
where S(v, A) is defined, based on the observed
data (Xi, Yi), i = 1, . . . , n, as
S(v, A) := 1
n
n
X
j=1
(
bF−j (A, Yj)
−
1
n −1
X
i:i̸=j,Xi∈A
bF−i,−j
 Yj | v⊤Xi
 
)2
.
Here, bF−j(A, Yj) is the empirical estimate of
F(A, Yj) based on all data except (Xj, Yj), and
bF−i,−j(Yj | v⊤Xi) is a leave-two-out local linear
estimator of F(Yj | v⊤Xi).
Thus, S(v, Aδ)
provides an estimate of
Z ∞
−∞
{F (Aδ, y) −Hv (Aδ, y)}2 dFY (y).
We then define bv as the minimizer of S(v) over
v ∈Θ.
Finally, the conditional cumulative distribu-
tion function FY |X(y
|
x) is estimated by
bF
 y | bv⊤x
 
, which is a local linear estimator of
F
 y | bv⊤x
 
and constructed in the same way
4


<!-- page 5 -->
as bF−i,−j but using the full sample (without
leaving two out).
Different bandwidths may
be employed for estimating v and F
 y | bv⊤x
 
.
Hall and Yao (2005) provided an empirical rule
for choosing the bandwidths and developed a
convergence rate for the estimated conditional
cumulative distribution function under suitable
conditions.
Some advantages and disadvan-
tages of the Hall and Yao approach are as fol-
lows.
Advantages:
• The method mitigates the curse of dimen-
sionality by reducing the predictor space
from Rp to one dimension through a single
index model.
• A rigorous asymptotic theory is available.
Disadvantages:
• Single-index limitation: The approxima-
tion of
FY |X(y | x) by F(y | v⊤x) may suffer
from model misspecification. Extensions
to multiple-index settings are nontrivial,
and to our knowledge no such extensions
have been developed.
• Multi-dimensional Y : It seems possible
to consider the case where Y is a vector
by focusing on its joint cdf. However, the
implementation in that case will undoubt-
edly add an extra layer of challenges.
• Computational challenges: Estimating v
involves minimizing a nonconvex, high-
dimensional objective function based on
kernel estimators of conditional distribu-
tions, which is computational demanding.
2.2
FlexCode and DeepCDE:
nonparametric estimation
using orthogonal series
expansion
Next, we review the basis-expansion approaches
of Izbicki and Lee (2017) and Dalmasso et al.
(2020), which considered the estimation of the
conditional density fY |X(y | x) of a scalar
response Y
∈R given a random predictor
X ∈Rp. Let {φj}∞
j=1 be an orthonormal basis
in L2(R), such as a cosine, Fourier or wavelet
basis. Then, we can write
fY |X(y | x) =
∞
X
j=1
βj(x)φj(y),
(4)
where the coeﬀicients are
βj(x) =
Z
fY |X(y|x)φj(y)dy
= E (φj(Y ) | X = x) .
Thus, the problem of estimating f(y | x) be-
comes a nonparametric regression problem of
estimating the functions βj(x).
This can, in
principle, be accomplished using any suitable
nonparametric regression method applied to the
data pair {Xi, φj(Yi), i = 1, . . . , n}.
In prac-
tice, the basis expansion must be truncated,
with the number of retained terms serving as a
tuning parameter that balances bias and vari-
ance in the final density estimate.
Izbicki
and Lee (2017) considered several nonpara-
metric regression techniques for estimating the
βj(x), depending on the data type, includ-
ing sparse additive models, nearest-neighbor re-
gression, random forests, and support distri-
bution machines.
Dalmasso et al. (2020) fur-
ther extended this framework by employing
neural networks for the estimation of βj(x).
The FlexCode approach has implementations
in both R (https://github.com/rizbicki/
FlexCoDE) and Python (https://github.com/
lee-group-cmu/FlexCode), and DeepCDE is
available in Python at https://github.com/
lee-group-cmu/DeepCDE/tree/master.
5


<!-- page 6 -->
Advantages:
• The formulation of expressing fY |X(· |
x) by a basis expansion is conceptually
clean.
The coeﬀicients estimation can
be regarded as nonparametric regression
problems, which can be solved using a
wide range of methods, allowing flexibility
and adaptability to different data types
and data structures.
• Training for FlexCode is computationally
eﬀicient. Moreover, although paralleliza-
tion is not essential in typical problem
sizes, the method is naturally paralleliz-
able over basis indices j because since
each regression problem can be handled
independently.
Disadvantages:
• FlexCode and DeepCDE rely on the as-
sumption that the conditional density ad-
mits a well-behaved orthogonal series rep-
resentation of the form (4), so that a fi-
nite number of basis coeﬀicients can ap-
propriately capture the relevant structure
of fY |X. In more complex scenarios, for
example those involving heteroscedastic
noise—such as M6 and M7 in Section 3—
the performance of FlexCode and Deep-
CDE tends to deteriorate.
• The estimated conditional density may
take negative values or fail to integrate
to one.
Additional steps, such as post-
processing or normalization, are often re-
quired to ensure that the estimate satis-
fies the basic properties of a valid density
function.
• The series expansion must be truncated
at some finite J, and the performance
of the approach depends on this choice.
The cutoff J serves as a tuning parameter
that balances the bias-variance tradeoff in
the resulting density estimator. Typically,
smoother densities can be adequately ap-
proximated with a smaller J.
2.3
GCDS: a distribution matching
generative approach
The third approach we consider is proposed
by Zhou et al. (2023), which is fundamentally
different from the previous two methods. It is a
simulation-based generative approach grounded
in the Noise Outsourcing Lemma introduced
in (1).
In this framework, the uniform noise
variable U is replaced by a latent noise vector
η ∼N (0, Im) that is independent of X. The
vector η introduces stochasticity into the gener-
ative mechanism and represents the randomness
in Y not explained by X. Although the state-
ment of the Noise Outsourcing Lemma uses a
scalar uniform variable, it is common in prac-
tice to employ a multivariate Gaussian noise
vector, which provides richer latent variability
and enables the model to capture complex or
multimodal conditional distributions.
The la-
tent dimension m governs the expressive capac-
ity of the model: a larger m allows more flexi-
ble approximations but may also lead to greater
stochastic variation.
It is worth noting that
m need not equal q, the dimension of Y . In-
tuitively, GCDS trains a generator to produce
synthetic pairs (X, G(η, X)) whose joint distri-
bution matches the observed joint distribution
of (X, Y ).
In the approach of Zhou et al. (2023), the
measurable mapping G is parameterized by a
neural network Gψ with parameters ψ. Follow-
ing the ideas of generative adversarial networks
(Goodfellow et al., 2014), Zhou et al. (2023)
adopted a distribution-matching approach that
estimates the conditional generator Gψ by min-
imizing the Kullback-Leibler (KL) divergence,
DKL
 FX,Gψ∥FX,Y
 
, between the joint distribu-
tion of the generated pair, (X, Gψ(η, X)), and
the true joint distribution of (X, Y ). Let
D(w) = log fX,Gψ(w)
fX,Y (w) ,
w ∈Rp+q,
(5)
which is the logarithm of the corresponding den-
sity ratio.
Here D(w) denotes the ideal log-
6


<!-- page 7 -->
density ratio in (5), while Dϕ(w) is its neural-
network approximation parameterized by ϕ.
Using the variational representation (Nguyen
et al., 2010), the objective KL divergence can be
reformulated as a minimax optimization prob-
lem over a generator Gψ and a discriminator
Dϕ. Thus,
DKL
 FX,Gψ∥FX,Y
 
= E(X,η)
 
log fX,Gψ(X, Gψ(η, X))
fX,Y (X, Gψ(η, X))
 
= sup
Dϕ
n
E(X,η)∼FXFη[Dϕ(X, Gψ(η, X))]
−E(X,Y )∼FX,Y [exp(Dϕ(X, Y ) −1)]
o
= sup
Dϕ
n
E(X,η)∼FXFη[Dϕ(X, Gψ(η, X))]
−E(X,Y )∼FX,Y [exp(Dϕ(X, Y ))]
o
+ 1,
where Fη denotes the distribution function of η.
Define
L(Gψ, Dϕ) = E(X,η)∼FXFη[Dϕ(X, Gψ(η, X))]
−E(X,Y )∼FX,Y [exp(Dϕ(X, Y ))].
Thus, Gψ and Dϕ are estimated by minimizing
a sample version of L(Gψ, Dϕ).
Given an
i.i.d. sample {(Xi, Yi), i = 1, . . . , n} from FX,Y
and an independent sample {ηi, i = 1, . . . , n}
generated from Fη, define
bL(Gψ, Dϕ)
=1
n
n
X
i=1
{Dϕ(Xi, Gψ(ηi, Xi)) −exp(Dϕ(Xi, Yi))} .
The first term in
bL(Gψ, Dϕ) uses samples
(Xi, ηi) drawn from FX × Fη, where the noise
variables ηi are independently re-drawn at each
training epoch. The second term uses the ob-
served pairs (Xi, Yi) ∼FX,Y to approximate
the expectation under FX,Y .
Here, Gψ, Dϕ
are two separate feedforward neural networks
(Goodfellow et al., 2016). For network parame-
ter selection, we solve
( bψ, bϕ) = argmin
ψ
argmax
ϕ
bL(Gψ, Dϕ),
and set bG = G bψ. As shown in Zhou et al. (2023),
under suitable conditions, the distribution of
bG(η, x) provides a consistent estimator of the
conditional distribution of Y given X = x.
Advantages:
• Sampling
during
inference
is
speedy.
Once the model is trained, generating
samples requires only a single forward
pass of the generator, making it much
faster at inference time than diffusion-
based methods such as DDPM, which
involve multiple iterative denoising steps.
Disadvantages:
• The dimension m of the latent noise vec-
tor η ∼N(0, Im) critically affects the ex-
pressiveness of the model.
If m is too
small, the generator Gψ(η, X) may lack
suﬀicient stochastic degrees of freedom to
represent the full variability in Y given
X = x. If m is too large, training may
become more diﬀicult, and the generator
may suffer from greater stochastic vari-
ability. Zhou et al. (2023) does not pro-
vide a general rule for selecting the dimen-
sion m; instead, it noted that “The value
of m should be chosen on a case-by-case
basis in practice.”
• Although
GCDS
offers
much
faster
computation
in
inference
(sampling)
compared to the other simulation-based
DDPM method reviewed in this article,
this comes at the cost of reduced flexi-
bility in capturing complex conditional
structures.
In addition,
its training
time is considerably slower than that of
DDPM.
2.4
Conditional DDPM: a
diffusion-based generative
approach
The second generative simulation–based ap-
proach we consider is the conditional denoising
7


<!-- page 8 -->
diffusion probabilistic model.
Although both
DDPM and the Noise Outsourcing Lemma rely
on the idea of generating randomness exter-
nally and transforming it into samples of a tar-
get conditional distribution, DDPM operates in
a fundamentally different manner from GCDS.
Rather than learning a one-step transformation
Y = Gψ(η, X), DDPM constructs a multi-step
stochastic diffusion process that gradually con-
verts data into noise and then learns a reverse
denoising process capable of synthesizing new
samples.
Recent works, including Fu et al.
(2024) and Yang et al. (2025), study condi-
tional sampling via diffusion models by defining
a forward diffusion process and learning the cor-
responding reverse-time denoising process via
score matching under a continuous-time diffu-
sion model.
For ease of exposition, and to
avoid the technical overhead of continuous-time
stochastic differential equations, we present a
discrete-time formulation that aligns with this
approach and suﬀices for our purposes.
The
DDPM framework consists of two components:
• a forward diffusion process that gradually
transforms the training data into random
noise, and
• a reverse denoising process that generates
data samples by progressively transform-
ing noise back toward the data distribu-
tion.
In the forward process, given a training sample
Yi, Gaussian noise is progressively injected
so that, after many steps, the distribution
of the corrupted sample becomes close to a
standard normal distribution.
For notational
simplicity, we drop the sample index i, with
the understanding that the following procedure
is applied to each training sample.
Let T ∈
N be the number of diffusion steps, and let
{αt ∈(0, 1)}T
t=1 be a sequence of pre-specified
noise schedule.
Set βt = 1 −αt, and draw
i.i.d. noise vectors εt ∼N(0, Iq). The forward
diffusion process, which takes the form of an
AR(1) model, is given by
Yt = √αt Yt−1 +
p
βt εt,
t = 1, . . . , T.
(6)
Recursively applying (6), it leads to the follow-
ing closed-form representation of the forward
process:
Yt = √αt
 √αt−1 Yt−2 +
p
βt−1εt−1
 
+
p
βtεt
= √αtαt−1 Yt−2 +
p
αtβt−1 εt−1 +
p
βt εt
...
...
...
= √αt · · · α1 Y0 +
t
X
j=1
p
αt · · · αj+1βj εj
with the convention αtαt+1 = 1 when j = t. Let
αt := Qt
s=1 αs. Then the variance contributed
by the noise terms is Pt
j=1 αt · · · αj+1βj = 1 −
αt · · · α1 = 1 −αt. Thus the forward process
admits the compact DDPM form:
Yt = √αt Y0 +
√
1 −αt εt,
(7)
where
εt = (1−αt)−1/2
t
X
j=1
p
αt · · · αj+1βj εj ∼N(0, Iq).
Here εt denotes the Gaussian noise injected in
the forward AR(1) process, whereas εt is the
aggregated noise appearing in the closed-form
expression (7). In contrast, bεt in (8) denotes
the neural network’s estimate of this aggregated
noise. Although each εt is standard Gaussian,
the sequence {εt} is not independent. Owing
to (7), the conditional distribution of Yt given
Y0 = y0 converges weakly to N(0, Iq) as t →∞.
Up to this point, the conditioning variable X
has not yet appeared in the formulation.
It
appears naturally in the reverse denoising step
through the noise-prediction model.
While the forward process diffuses the data
toward a standard normal distribution, the re-
verse process seeks to invert this diffusion. The
reverse (denosing) process requires estimating
8


<!-- page 9 -->
the noise component εt given Yt = yt and
X = x. This estimation step can be viewed
as a nonparametric regression task, in which a
neural network noise model τθ is trained to pre-
dict the noise εi,t from (yi,t, xi, t), with θ denot-
ing the network parameters and i indexing the
training samples. Specifically, we solve
bθ := argmin
{θ:τθ∈F}
n
X
i=1
T
X
t=1
∥εi,t −τθ(yi,t, xi, t)∥2 ,
where F is a suitable neural network function
class. With the resulting estimate bθ, we denote
the trained noise estimation model by τbθ. For
notational convenience, we write the noise esti-
mate for the input (yt, x, t) as
bεt := τbθ(yt, x, t).
(8)
To describe the reverse process, we introduce
the backward transition.
For simplicity, we
again suppress the sample index i. Under the
diffusion process, we have
p(yt−1|yt, y0) = p(yt|yt−1, y0)p(yt−1|y0)
p(yt|y0)
= p(yt|yt−1)p(yt−1|y0)
p(yt|y0)
,
where the second equality follows from the
Markovian property of the forward diffusion
process. Combining (6) and (7), together with
some straightforward calculation, yields:
Yt−1|yt, y0 ∼N
 µ(yt, y0), σ2(t)Iq
 
,
where
µ(yt, y0) =
√αt(1 −αt−1)yt + √αt−1(1 −αt)y0
1 −αt
and
σ2(t) = (1 −αt)(1 −αt−1)
1 −αt
.
To remove dependence on the unknown y0, we
substitute it with the noise expression from (7):
y0 = yt −√1 −αt εt
√αt
,
(9)
and we can estimate this noise term using the
trained model (8). Substituting the expression
(9) into the backward mean yields the noise-
parameterized posterior mean
µ(yt, y0) =
1
√αt
 
yt −1 −αt
√1 −αt
εt
 
.
Replacing εt by its estimator bεt from (8) gives
the sampling rule for the reverse transition.
Thus, a sample yt−1 given yt can be generated
according to
yt−1 =
1
√αt
 
yt −1 −αt
√1 −αt
bεt
 
+ σ(t)z,
where z ∼N(0, Iq).
Among the methods reviewed, we provide
pseudocode only for the DDPM approach, as
it is the most flexible and broadly applica-
ble across a wide range of settings.
We sum-
marize the above discussion by presenting the
method in two stages: Algorithm 1 (Training)
and Algorithm 2 (Sampling), which describe
the procedures for learning the noise-estimation
model and generating conditional samples, re-
spectively. For computational eﬀiciency, train-
ing does not simulate the full forward diffusion
sequence over all timesteps. Instead, for each
training sample indexed by i, a single random
timestep ti is selected, and a Gaussian noise vari-
able, denoted by εi, is drawn to construct the
noisy input yi,ti.
This εi serves solely as the
synthetic noise used in the single-step training
scheme and is not part of the forward AR(1)
diffusion sequence discussed earlier.
Advantages:
• DDPM achieves better generative quality
and diversity compared with GCDS, ow-
ing to its probabilistic formulation and
fine-grained denoising steps that progres-
sively refine samples.
• Training is relatively stable compared
with GCDS, since the DDPM objective
reduces to a simple mean-squared error
9


<!-- page 10 -->
between the true and predicted noise.
In contrast, GCDS relies on distribution
matching through a min–max adversarial
optimization, which is more susceptible
to instability.
As a result, DDPM typ-
ically exhibits more stable training and
less stochastic behavior at inference than
GAN-style methods such as GCDS.
Disadvantages:
• Sampling (inference) is computationally
expensive because the reverse diffusion
process requires many iterative denois-
ing steps.
Consequently, generation is
substantially slower than GCDS in both
computation time and memory usage.
Faster approximate sampling methods
(e.g., Song et al., 2021) are available, but
they may involve trade-offs in accuracy or
stability.
Algorithm 1 Conditional DDPM Training
1: Input: dataset {(xi, yi,0)}n
i=1, number of diffusion
steps T
2: Output: trained noise estimation model τbθ(yt, x, t)
3: Initialize network parameters θ of τθ(yt, x, t)
4: while not converged do
5:
Partition index set {1, . . . , n} into mini-batches
B1, . . . , Bm
6:
for each mini-batch B do
7:
for each i ∈B do
8:
Draw ti ∼Uniform{1, . . . , T} and
εi ∼N(0, Iq)
9:
Compute yi,ti = √αti yi,0 + √1 −αti εi
10:
Predict bεi = τθ(yi,ti, xi, ti)
11:
end for
12:
Compute the loss L =
1
|B|
P
i∈B
∥εi −bεi∥2
13:
Update θ ←θ −γ ∇θL
14:
end for
15: end while
16: Return: τbθ
3
Numerical studies
This section presents a systematic compari-
son of the approaches introduced in Section 2
Algorithm 2 Conditional DDPM Sampling
1: Input: predictor x and trained noise model τbθ
2: Output: conditional sample y0
3: Draw yT ∼N(0, Iq)
4: for t = T, . . . , 1 do
5:
Compute predicted noise bεt = τbθ(yt, x, t)
6:
Draw z ∼N(0, Iq) if t > 1, else z = 0
7:
Update by yt−1 =
1
√αt
 
yt −
1−αt
√1−αt bεt
 
+ σ(t)z
8: end for
9: return y0
through controlled simulation experiments de-
signed to evaluate their empirical performance
under various settings.
3.1
Simulation models
In order to conduct a broad and informative
comparison among approaches grounded in dis-
tinct methodologies, we consider multiple simu-
lation scenarios that exhibit a wide range of con-
ditional behaviors. These models cover a wide
spectrum of conditional behaviors, ranging from
smooth, homoscedastic, and Gaussian settings
to scenarios exhibiting heteroscedasticity, high
skewness, heavy tails, covariate-dependent mix-
tures, and latent mixture structures. By these
different features of simulation models, we illu-
minate the types of distributional characteris-
tics that each method is able to capture, as well
as those that present greater challenges, thereby
providing a more balanced assessment of their
respective strengths and limitations.
We consider the following simulation models.
M1 This is Example 2 of Hall and Yao (2005).
Let
Y = 0.5
4
X
j=1
sin Xj + ε,
(10)
where Xj, j = 1, . . . , 4, and ε are i.i.d.
N(0, 1).
This model features a smooth
conditional
mean
and
homoscedastic
Gaussian noise.
10


<!-- page 11 -->
M2 This model is identical to M1 except that
the number of predictors is increased to
10; that is, X = (X1, . . . , X10)⊤∈R10
with Xj
i.i.d.
∼N(0, 1) and independent ε ∼
N(0, 1). The response Y is still generated
by (10), so X5, . . . , X10 are redundant pre-
dictors.
This setting examines whether
the methods remain stable in the presence
of irrelevant predictors.
M3 This model is based on M1, except that
X = (X1, . . . , X4)⊤follows a multivariate
normal distribution N(0, Σ) with Σij =
0.5|i−j| for i, j = 1, . . . , 4, independent of
the error term ε ∼N(0, 1). This model
introduces correlation among predictors.
M4 This model extends M1 by introduc-
ing latent sign variables.
Let Z
=
(Z1, . . . , Z4)⊤, where Zj are i.i.d.
with
Pr(Zj = 1) = Pr(Zj = −1) = 1/2, in-
dependent of X and ε. Define
Y = 0.5
4
X
j=1
sin(ZjXj) + ε.
(11)
Z serves as an unobserved latent factor
that randomly flips the sign of each Xj.
The marginal distribution of Y in (11)
remains the same as that of Y in (10);
however, their conditional distributions of
Y
| X differ.
Since Z is latent and
unavailable to the estimation procedures,
this model provides a useful setting for
assessing the robustness of different meth-
ods to unobserved heterogeneity.
M5 This model extends M1 by introducing
predictor-specific latent scale variables.
The response is generated by the same
formula as in (11), but with Zj
i.i.d.
∼
Uniform(0, 1), independent of X and ε.
The latent variables Zj introduce random
continuous modulation of the predictor ef-
fects, creating feature-wise nonlinear dis-
tortions. As a result, the conditional dis-
tribution of Y | X becomes more hetero-
geneous and distinctly non-Gaussian, ex-
hibiting varying local curvature and scale
depending on the latent realization of Z.
M6 This model corresponds to Example 1 of
Yang et al. (2025) and Example 2 of Zhou
et al. (2023). Let
Y = X2
1 + exp(X2 + X3/3) + X4 −X5
+ (0.5 + X2
2/2 + X2
5/2) × ε,
where ε ∼N(0, 1). This model exhibits
heteroscedasticity arising from nonlinear
covariate effects on the noise scale.
M7 This model is a milder counterpart to
M8 described below. It retains the same
multiplicative structure but uses a milder
and covariate-dependent noise, thereby
yielding an easier problem. For X ∈R30,
let
m(X) = 5 + X2
1/3 + X2
2 + X2
3 + X4 + X5,
and define
Y = m(X) × exp(0.25 ε),
where ε | X1 ∼π(X1) × N(−1, 1) + {1 −
π(X1)}×N(1, 1), and π(x) = logit−1(κx).
The parameter κ > 0 governs the sensitiv-
ity of the mixing weights to X1: larger val-
ues produce sharper changes in the mix-
ing weight π(X1) and hence stronger devi-
ations from the pure scaling structure in
M8, whereas smaller values yield milder
shape differences and a problem closer to
the scaling model. In other words, κ tunes
how non-scaling and how heterogeneous
the conditional density is. Here we take
κ = 0.5. Compared with M8, the noise
variation is milder.
M8 This model follows Example 2 of Yang
et al. (2025) and Example 3 of Zhou et al.
11


<!-- page 12 -->
(2023). Using the same function m(X) as
in M7, define
Y = m(X) × exp(0.5 ε),
where ε ∼I(U < 0.5)×N(−2, 1)+I(U >
0.5) × N(2, 1) with U ∼Uniform(0, 1)
and X ∈R30.
The latent mixture in-
side the exponential induces stronger het-
eroscedasticity and shape variation than
in M7.
M9 Let X1 and ε be i.i.d. N(0, 1), and let
c > 0 control the oscillation frequency of
sin(cX1 + ε) in X1. Define
Y = g (sin(cX1 + ε)) with g(u) = exp (u),
and set c
=
20.
The composition
of
a
highly
oscillating
sinusoid
with
the exponential mapping produces highly
skewed and heavy-tailed conditional dis-
tributions.
M10 This model evaluates multivariate condi-
tional distribution estimation for Y ∈R7.
Let
Y =
 
X2
1, X2
2, X2
3, X2
4, X2
5,
exp(X2 + X5/3), sin(X4 + X5)
 ⊤
+ ε,
where X ∼N(0, I5) and ε ∼N(0, I7).
The heterogeneous nonlinear components
in Y create a nontrivial multivariate con-
ditional structure useful for comparing
GCDS and DDPM.
To illustrate the behavior of the synthetic
models (excluding M10, which is multivariate),
we display their true conditional densities in
Figures 1-2.
For each model, four predictor
values x are drawn at random, and the true
conditional density is evaluated directly from
the data-generating mechanism.
Among the
proposed synthetic models, M1–M4 and M6–
M8 admit closed-form conditional densities. In
contrast, the conditional densities of M5 and M9
are unavailable for closed form and are therefore
evaluated approximately. Each panel shows the
true conditional density (black curve), together
with the true conditional mean marked by a red
dashed line.
3.2
Simulation design
We conducted 10 independent simulation runs
for each algorithm to assess performance vari-
ability.
The number of runs was limited to
10 due to the high computational cost and the
large number of methods being compared. Nev-
ertheless, the standard deviations reported in
Table 1 indicate modest variability across runs,
suggesting that this choice is reasonable.
In
each run, the model was trained on 5,000 con-
ditional sample pairs, validated on 2,000 sam-
ples, and evaluated on 2,000 previously unseen
test samples. For each test input, we generated
2,000 samples from the estimated conditional
distribution and computed the evaluation met-
rics described in Section 3.4. Note that while
it is possible to compute the metrics numeri-
cally for both the Hall and Yao and FlexCode
methods, we chose to standardize the process
by conducting simulations for all methods.
The normalization schemes used in the exper-
iments are described below. For FlexCode and
DeepCDE, we follow the normalization settings
in the original authors’ implementations. For
GCDS and DDPM, which are implemented in
this work, standard normalization is applied to
facilitate stable training. All generated samples
are rescaled to their original physical scale prior
to computing the evaluation metrics.
• FlexCode:
the response Y is scaled to
[0, 1].
• DeepCDE: the response Y is scaled to
[0, 1], and the predictors X are standard-
ized coordinate-wise to have mean zero
and unit variance.
12

[CAPTION] Table 1 indicate modest variability across runs,


<!-- page 13 -->
• GCDS: the response Y is standardized to
have mean zero and unit variance.
• DDPM: the response Y is standardized to
have mean zero and unit variance.
The configurations for each method are sum-
marized as follows.
• For Hall and Yao, we set the radius
to r∗
= 1.
For bandwidth selection
in each model, we carried out a grid
search to choose (h, H), corresponding
to the bandwidths for estimating v and
F
 y | bv⊤x
 
, over a coarse grid, where h ∈
{0.1, 0.3, . . . , 1.1} and H ∈{h + 0.1, h +
0.3, h + 0.5, . . . , 1.2}.
The grid search
was conducted using a single training
sample of size 5,000, and the selected pair
was chosen by minimizing the Wasserstein
distance on a validation sample of size
2,000.
After obtaining the optimal pair
(bh, bH), we fixed it and applied the Hall
and Yao method to independent training
and test samples for evaluation. Although
the selected pair (bh, bH) may vary across
different data realizations, the resulting
inferential performance averaged over 10
independent runs remains very similar.
• For FlexCode, we employed a cosine ba-
sis together with random forest regression.
The contributing basis functions were se-
lected, based on their importance in mini-
mizing the empirical conditional density
estimation loss, from a maximum of 31
candidate basis functions.
• For DeepCDE, we used a set of 31 cosine
basis functions and a learning rate of 10−4
for all models, with early stopping based
on the validation loss and a patience of 20
epochs.
• For GCDS, each model was trained for
500 epochs, using a learning rate of 3 ×
10−4 for both the generator and discrimi-
nator in M10, and 10−4 for both networks
in all remaining models.
• For DDPM, the diffusion process em-
ployed a linear noise schedule. The mod-
els were trained for 50 epochs with an ini-
tial learning rate of 10−2 and a drop factor
of 0.5 every 10 epochs.
3.3
Model implementation
details
We next describe the implementation details
of the learning-based methods, focusing on the
neural network architectures and the implemen-
tation environment.
3.3.1
Neural network architectures
DeepCDE network architecture. For DeepCDE,
we employ a three-hidden-layer MLP (with
width 32-64-32 and GELU activations) to ex-
tract nonlinear representations from the predic-
tors. A final linear layer (the CDE layer) maps
the 32-dimensional representation to J basis co-
eﬀicients {βj(x)}J
j=1 corresponding to the cho-
sen basis system. The entire network is trained
by minimizing the CDE loss. This compact mul-
tilayer perceptron enables DeepCDE to capture
nonlinear dependencies between predictors and
eﬀiciently estimate the conditional density func-
tion bfY |X(y | x) = PJ
j=1 bβj(x) φj(y), where
{φj}J
j=1 denotes the chosen orthonormal basis
in L2(R). The number of basis functions J is
a tunable hyperparameter.
In the simulation
studies, we set J = 31.
In addition, we use
orthonormal cosine basis for φ.
GCDS network architecture. The generator
takes as input the concatenation of the predic-
tors and a latent noise vector.
This input is
passed through a single hidden-layer MLP with
width 50 and a ReLU activation to generate a
conditional sample of dimension q. The discrim-
inator receives the concatenation of the predic-
tors and a response, and processes it through a
13


<!-- page 14 -->
two-hidden-layer MLP with widths 50 and 25,
each followed by ReLU activations. A final lin-
ear layer maps the 25-dimensional representa-
tion to a single scalar output, which serves as
the discriminator’s score indicating how likely
the predictor–response pair is to have come
from the true conditional distribution.
DDPM network architecture. We use a con-
ditional MLP that takes as input the concate-
nation of the predictors, a noised version of the
response, and the scalar diffusion time. The net-
work consists of two hidden layers with widths
50 and 25, each followed by a ReLU activation,
and ends with a linear output layer that maps to
a vector in Rq. The network parameters are es-
timated by minimizing the mean squared error
between the true noise and the predicted noise.
3.3.2
Implementation environment
All learning-based methods were implemented
in a common software environment using stan-
dard deep learning libraries. To ensure fair com-
parisons, all experiments were conducted on the
same computing platform and executed exclu-
sively on the CPU, with no GPU acceleration.
Specifically, experiments were run on a server
equipped with an AMD EPYC 7742 processor
(64 cores, 128 threads) and 1 TiB of RAM.
3.4
Evaluation metrics
The first issue to address is the choice of
discrepancy measures between the estimated
and the true conditional distributions. There
are many possibilities depending on what aspect
the conditional distribution is being evaluated.
For example, Zhou et al. (2023) focus on how
well the true conditional mean and conditional
variance can be estimated by the simulated
conditional samples, using the following mean
squared errors:
MSE (mean) = 1
k
k
X
i=1
 
bE(Y | xi) −E(Y | xi)
 2
,
MSE (sd) = 1
k
k
X
i=1
 
c
SD(Y | xi) −SD(Y | xi)
 2
,
where {x1, . . . , xk} is a test set, and, bE(Y | xi),
c
SD(Y | xi), i = 1, . . . , k, are computed from
simulated samples of Y at each xi. While the
mean and standard deviation fully determine a
normal distribution, they are not ideal measures
for judging the discrepancy between two distri-
butions in general. A broader discussion of pos-
sible distance measures between distributions is
provided in Ramdas et al. (2017).
In addition to the MSE of the conditional
mean and standard deviation, we also adopt
the Wasserstein distance as our primary dis-
crepancy measure.
For r ∈[1, ∞) and Borel
probability measures µ1, µ2 on Rq with finite
r-moments, their r-Wasserstein distance is de-
fined as
Wr(µ1, µ2) =
 
inf
π∈Γ(µ1,µ2)
Z
Rq×Rq ∥y −y′∥rdπ(y, y′)
 1/r
,
where Γ(µ1, µ2) is the set of all joint probability
measures π on Rq × Rq whose marginals are µ1
and µ2. That is, for all subsets A∗⊂Rq, we
have π (A∗× Rq) = µ1(A∗) and π (Rq × A∗) =
µ2(A∗).
If q = 1, then the r-Wasserstein
distance simplifies to
Wr(µ1, µ2) =
 Z 1
0
|F −1
1 (u) −F −1
2 (u)|rdu
 1/r
,
where F −1
1
and F −1
2
are the quantile func-
tions corresponding to µ1 and µ2,
respec-
tively.
In
practice,
the
Wasserstein
dis-
tance
Wr(µ1, µ2)
can
be
approximated
by
Wr(µ1,n, µ2,n), where µ1,n and µ2,n are empir-
ical measures based on finite samples.
The
scipy.stats
library
provides
the
function
14


<!-- page 15 -->
wasserstein_distance for computing the 1-
Wasserstein distance for univariate distribu-
tions, as well as wasserstein_distance_nd for
multivariate distributions. However, as noted
by Ramdas et al. (2017), the empirical estimate
Wr(µ1,n, µ2,n) may converge slowly to the true
Wasserstein distance Wr(µ1, µ2) when q > 1.
To obtain a computationally eﬀicient approxi-
mation, we instead compute the sliced Wasser-
stein distance between the empirical distribu-
tions using the function sliced_wasserstein_
distance from the POT (Python Optimal Trans-
port) library (Flamary et al., 2021). The sliced
Wasserstein distance approximates the multi-
variate Wasserstein distance by averaging 1D
Wasserstein distances of the distributions pro-
jected onto multiple random directions (Bon-
neel et al., 2015), which is fast to compute and
effective in practice.
3.5
Results
To assess the comparative performance of condi-
tional distribution estimation, we evaluate the
three metrics described in Section 3.4 for each
method across a series of simulated examples.
Table 1 summarizes the results based on 10 sim-
ulation runs.
Boldface indicates the method
that achieves the best overall performance in
terms of distributional accuracy, as reflected
primarily by the Wasserstein distance together
with the conditional mean MSE. In some set-
tings, a method may exhibit smaller variability
(e.g., lower MSE of the conditional standard de-
viation) while remaining farther from the target
conditional distribution.
M1 serves as a simple baseline model with
smooth nonlinear main effects and homoscedas-
tic Gaussian noise. M2-M5 extend M1 by in-
troducing additional structural complexity: M2
adds redundant predictors, M3 introduces cor-
related predictors, M4 incorporates latent sign
variables that give rise to multiple regression-
function branches, and M5 introduces contin-
uous latent scales. These settings form an in-
creasingly challenging sequence for conditional
distribution estimation. With the exception of
the Hall and Yao method on M1–M3 and M5,
all approaches perform well on M1–M5.
No-
tably, Hall and Yao performs particularly well
on M4.
Among these, DDPM achieves the
strongest overall performance on M1–M3, and
the Hall and Yao method and DeepCDE ex-
hibit the strongest performance on M4 and M5,
respectively, according to the three evaluation
metrics.
M6-M9 represent settings with heteroscedas-
tic noise, and M7-M9 additionally produce non-
Gaussian conditional distributions.
M6 al-
lows the noise scale to vary with the predic-
tors. In M7, the conditional distribution is a
non-Gaussian mixture induced by a covariate-
dependent Gaussian mixture in the noise term.
M8 employs a latent-weight Gaussian mixture,
which produces more complex conditional den-
sities, often exhibiting multimodality and pro-
nounced shape variation.
In M9, the non-
linear transformation leads to highly skewed,
heavy-tailed, or bimodal conditional distribu-
tions. Relative to M8 and M9, the settings in
M6 and M7 are comparatively simpler for condi-
tional distribution estimation. DDPM and Flex-
Code yield the strongest performance on M6-
M7 and M8-M9, respectively.
M10 is a multivariate example designed to
compare GCDS and DDPM; see the next two
paragraphs for the reasons why the other meth-
ods are not considered. In this setting, DDPM
performs better than GCDS. More broadly,
across all simulation settings except M9, DDPM
generally outperforms GCDS under the three
evaluation metrics considered for conditional
distribution estimation.
In M9,
the two
methods exhibit comparable performance, with
GCDS performing slightly better. A key reason
is that DDPM employs a multi-step diffusion-
based generative mechanism, which yields more
stable and effective learning.
By contrast,
GCDS involves a min-max adversarial optimiza-
tion, which is known to be more sensitive to
15

[CAPTION] Table 1 summarizes the results based on 10 sim-


<!-- page 16 -->
training instability (see, e.g., Goodfellow et al.,
2014; Dhariwal and Nichol, 2021; empirical ev-
idence in the conditional settings was also re-
ported in Han et al., 2022).
The Hall and Yao method is a dimension-
reduction approach and, as a result, offers less
flexibility than the other methods for condi-
tional distribution estimation.
Note that the
results based on the Hall and Yao method are
not reported for models M7, M8, and M10.
This is because it was specifically developed
for the scalar-response setting, and the compu-
tational burden grows rapidly with the predic-
tor dimension p, making it practically infeasi-
ble for higher-dimensional settings due to mem-
ory limitations.
In our simulation study, be-
cause no publicly available implementation of
this method exists, we developed our own based
on the original paper. Following their formula-
tion, the algorithm requires constructing a full
p-dimensional Cartesian grid of centers {Aδ}
within the hypercube [−r∗, r∗]p with grid spac-
ing h. The exponential growth of the grid size
renders the full grid computationally infeasible
even for moderate dimensions.
For instance,
in M7 and M8 with dimension p = 30, this
would require on the order of 8 × 1014 bytes
of memory.
To alleviate this issue, we exper-
imented with a random sampling scheme that
replaced the exhaustive mesh construction by
drawing a fixed number of centers uniformly
from [−r∗, r∗]p.
While this strategy substan-
tially reduced the memory requirement, empir-
ical results indicated that it led to degraded
performance, suggesting that the random sam-
pling strategy was insuﬀicient to capture the
fine structural dependencies represented by the
full grid.
Results based on FlexCode, and DeepCDE
are also not reported for M10 since the pub-
licly available implementations of FlexCode and
DeepCDE currently support only the univari-
ate response case. FlexCode, which estimates
each coeﬀicient βj(x) by regressing φj(Y ) on
X using methods such as random forests or k-
nearest neighbors, performs poorly under condi-
tional heteroscedasticity, where the variability
of φj(Y ) | X = x depends strongly on x. Deep-
CDE, in contrast, learns all coeﬀicients simul-
taneously through the optimization of a single
neural-network objective, allowing it to better
adapt to heteroscedastic structure in the data.
Similar behavior is observed under M6 and M7.
However, FlexCode is not constrained by a
fixed neural network architecture and remains
a reasonably flexible approach when a moder-
ate number of basis functions is employed to-
gether with an appropriate regression method,
even for estimating conditional densities that
exhibit complex shapes (e.g.
highly skewed,
multimodal, or heavy-tailed).
This flexibility
helps to explain its stronger performance rel-
ative to the neural-network-based methods in
M8 and M9.
We suggest that increasing the
architectural complexity of DeepCDE, GCDS
and DDPM would likely improve their ability
to learn the latent mixtures and more complex
distributional structures present in these mod-
els.
From a computational perspective, Table 2 re-
ports average epoch times for DeepCDE, GCDS,
and DDPM. Overall, DeepCDE achieves the
fastest per-epoch training times, while GCDS
is consistently the slowest due to its adversarial
min-max optimization. DDPM lies in between:
its per-epoch training cost is moderate and con-
siderably lower than that of GCDS, but its sam-
pling procedure is much more expensive, as re-
flected in the sampling times reported in Ta-
ble 1. This gap arises because each conditional
sample from DDPM requires running the full re-
verse diffusion chain, whereas GCDS generates
samples in a single forward pass of the generator
network. These findings are consistent with the
methodological discussion in Section 2: DDPM
often attains the best or near-best statistical
performance, but its generation is substantially
slower than GCDS, unless one resorts to faster
approximate samplers (e.g., Song et al., 2021)
that may trade off some accuracy or stability.
16


<!-- page 17 -->
Table 1: Performance comparison of five methods across selected data models. Values shown as
mean (standard deviation) over 10 simulation runs. Lower values indicate better performance for
all metrics.
Model Method
MSE Mean
MSE Std
W-1
Training Time
Sampling Time
M1
Hall & Yao
0.3023 (0.1289)
0.0257 (0.0155)
0.4503 (0.1187)
38.1 (39.1)
18.0 (0.6)
FlexCode
0.0415 (0.0041)
0.0262 (0.0065)
0.2042 (0.0113)
32.1 (0.3)
12.5 (0.5)
DeepCDE
0.0422 (0.0064)
0.0173 (0.0027)
0.1744 (0.0119)
34.6 (10.8)
4.1 (0.6)
GCDS
0.0459 (0.0094)
0.0049 (0.0015)
0.1667 (0.0154)
287.2 (2.5)
5.9 (0.3)
DDPM
0.0281 (0.0149)
0.0056 (0.0033)
0.1386 (0.0286)
22.2 (0.6)
211.1 (1.5)
M2
Hall & Yao
0.3887 (0.0545)
0.0338 (0.0074)
0.5262 (0.0410)
371.7 (411.1)
18.1 (0.6)
FlexCode
0.0475 (0.0037)
0.0298 (0.0039)
0.2208 (0.0068)
40.1 (4.7)
13.5 (1.6)
DeepCDE
0.0705 (0.0035)
0.0327 (0.0092)
0.2415 (0.0087)
28.3 (7.6)
4.2 (0.4)
GCDS
0.1185 (0.0136)
0.0059 (0.0016)
0.2686 (0.0156)
287.2 (3.0)
6.2 (0.4)
DDPM
0.0694 (0.0127)
0.0071 (0.0015)
0.2139 (0.0186)
22.0 (0)
216.4 (1.7)
M3
Hall & Yao
0.4392 (0.3412)
0.0615 (0.0557)
0.5230 (0.2635)
74.5 (70.5)
27.2 (5.1)
FlexCode
0.0395 (0.0023)
0.0397 (0.0082)
0.2119 (0.0069)
38.0 (3.0)
13.1 (0.7)
DeepCDE
0.0314 (0.0046)
0.0259 (0.0077)
0.1613 (0.0102)
27.7 (4.8)
3.9 (0.3)
GCDS
0.0530 (0.0090)
0.0076 (0.0060)
0.1944 (0.0129)
285.5 (2.4)
5.7 (0.5)
DDPM
0.0256 (0.0092)
0.0040 (0.0010)
0.1353 (0.0174)
22.2 (0.4)
212.3 (1.5)
M4
Hall & Yao
0.0028 (0.0009)
0.0061 (0.0007)
0.0826 (0.0047)
33.8 (30.8)
19.2 (0.6)
FlexCode
0.0060 (0.0017)
0.0096 (0.0031)
0.1047 (0.0097)
36.1 (7.0)
16.3 (0.7)
DeepCDE
0.0027 (0.0013)
0.0139 (0.0031)
0.1003 (0.0089)
13.3 (1.9)
2.5 (0.5)
GCDS
0.0191 (0.0160)
0.0076 (0.0014)
0.1413 (0.0337)
273.0 (2.5)
4.8 (0.4)
DDPM
0.0105 (0.0062)
0.0073 (0.0012)
0.1113 (0.0183)
23.2 (0.9)
213.5 (1.8)
M5
Hall & Yao
0.1176 (0.0481)
0.0047 (0.0020)
0.2793 (0.0682)
47.0 (31.1)
19.0 (1.6)
FlexCode
0.0198 (0.0026)
0.0107 (0.0027)
0.1458 (0.0051)
32.1 (0.3)
12.6 (0.5)
DeepCDE
0.0090 (0.0024)
0.0072 (0.0028)
0.1027 (0.0099)
21.6 (5.2)
4.0 (0)
GCDS
0.0235 (0.0186)
0.0033 (0.0014)
0.1353 (0.0415)
291.4 (3.9)
5.3 (0.5)
DDPM
0.0179 (0.0133)
0.0057 (0.0045)
0.1234 (0.0362)
22.1 (0.7)
212.0 (1.5)
M6
Hall & Yao
9.6668 (0.9152)
1.2568 (0.0583)
2.2461 (0.1026)
58.4 (59.7)
18.7 (0.5)
FlexCode
4.2328 (0.4169)
34.3715 (6.5531)
2.6339 (0.1467)
34.0 (2.0)
13.5 (2.0)
DeepCDE
0.9187 (0.3077)
4.0832 (1.2943)
0.7417 (0.0692)
71.0 (3.8)
3.0 (0)
GCDS
0.5603 (0.2113)
0.4041 (0.1037)
0.5153 (0.0581)
285.2 (0.9)
5.4 (0.5)
DDPM
0.2875 (0.0994)
0.0785 (0.0288)
0.3007 (0.0367)
22.0 (0.5)
212.8 (2.1)
M7
FlexCode
2.3556 (0.1946)
4.6753 (0.8336)
1.5303 (0.0627)
42.0 (0.5)
14.0 (0)
DeepCDE
1.8959 (0.2948)
3.4071 (0.6394)
1.2471 (0.0580)
34.6 (4.8)
3.1 (0.3)
GCDS
2.3605 (0.2883)
1.2614 (0.4157)
1.5737 (0.0968)
311.4 (47.5)
4.1 (0.7)
DDPM
0.9465 (0.0984)
0.3384 (0.0483)
0.8245 (0.0281)
23.1 (0.7)
216.0 (3.4)
M8
FlexCode
6.9075 (1.7249)
18.1483 (2.7374)
2.7234 (0.1746)
42.2 (1.9)
13.3 (0.8)
DeepCDE
26.9664 (3.7139) 68.4103 (25.1083)
4.5726 (0.3623)
19.1 (3.1)
3.0 (0)
GCDS
18.0615 (2.6195)
14.3314 (0.9252)
4.0671 (0.2768)
292.2 (3.3)
4.8 (0.6)
DDPM
12.3189 (1.3913)
11.7034 (1.4097)
3.3847 (0.1341)
22.1 (0.3)
211.2 (6.6)
M9
Hall & Yao
0.2168 (0.0157)
0.3093 (0.1157)
0.5684 (0.0545)
5.2 (0.4)
15.0 (0.7)
FlexCode
0.1125 (0.0054)
0.0383 (0.0024)
0.2955 (0.0106)
32 (0.5)
15.8 (0.4)
DeepCDE
0.2376 (0.0025)
0.0503 (0.0024)
0.4424 (0.0028)
13.9 (2.4)
3.8 (0.4)
GCDS
0.2377 (0.0028)
0.0449 (0.0063)
0.4428 (0.0027)
275.9 (1.5)
5.1 (0.3)
DDPM
0.2393 (0.0053)
0.0466 (0.0050)
0.4477 (0.0046)
22.6 (0.5)
224.8 (61.0)
M10
GCDS
0.6226 (0.1277)
0.1414 (0.0136)
0.6171 (0.0648)
283.8 (2.1)
5.8 (0.4)
DDPM
0.1244 (0.0469)
0.0282 (0.0044)
0.2157 (0.0144)
22.5 (0.5)
324.5 (10.5)
17

[CAPTION] Table 1: Performance comparison of five methods across selected data models. Values shown as


<!-- page 18 -->
Table 2: Comparison of training eﬀiciency across DeepCDE, GCDS, and DDPM. Average (standard
deviation) epoch time is reported in CPU seconds.
Model Method
Epoch Time
Model
Method
Epoch Time
M1
DeepCDE
0.2199 (0.1300)
M6
DeepCDE
0.2580 (0.2973)
GCDS
0.8235 (1.2695)
GCDS
0.5647 (0.2618)
DDPM
0.3657 (0.0293)
DDPM
0.3674 (0.0399)
M2
DeepCDE
0.2199 (0.1941)
M7
DeepCDE
0.2373 (0.1728)
GCDS
2.4470 (4.6249)
GCDS
0.6129 (0.6331)
DDPM
0.3688 (0.0335)
DDPM
0.3774 (0.0918)
M3
DeepCDE
0.1981 (0.1175)
M8
DeepCDE
0.1397 (0.0091)
GCDS
0.6978 (0.9915)
GCDS
0.6756 (0.8536)
DDPM
0.3651 (0.0343)
DDPM
0.3661 (0.0341)
M4
DeepCDE
0.1369 (0.0100)
M9
DeepCDE
0.1688 (0.0669)
GCDS
0.5366 (0.0273)
GCDS
0.5414 (0.0381)
DDPM
0.3584 (0.0420)
DDPM
0.3636 (0.0365)
M5
DeepCDE
0.1962 (0.1112)
M10
DeepCDE
–
GCDS
1.0545 (0.8006)
GCDS
0.5580 (0.0284)
DDPM
0.3679 (0.0438)
DDPM
0.3800 (0.0327)
4
Discussion
Conditional distribution estimation is a central
theme in modern statistics, and the landscape of
available methodologies is remarkably diverse.
This article has surveyed four representative
approaches: single-index dimension reduction,
basis-expansion regression, and two classes of
generative simulation-based methods, and eval-
uated their performance across various scenar-
ios in which the predictor dimension ranges
from low to moderately high.
The main em-
pirical findings from our simulation study can
be summarized as follows.
For
conditional
distribution
estimation,
DDPM, an autoregressive generative model,
appears to be the most robust approach among
the five methods considered based on the
simulation study.
FlexCode and DeepCDE, which are nonpara-
metric estimation approaches using basis expan-
sions, offer a certain degree of flexibility for con-
ditional density estimation when an appropri-
ate basis type is chosen and a suﬀiciently large
number of basis functions are used. However,
their performance may still be less competitive
relative to DDPM, which employs suﬀiciently
expressive model architectures.
FlexCode estimates the basis coeﬀicient func-
tions using nonparametric regression methods;
hence, it can perform poorly when the underly-
ing scenario is unfavorable for the chosen non-
parametric regression approach, such as in the
presence of heteroscedastic noise. In such cases,
using DeepCDE may lead to improved condi-
tional distribution estimation.
DeepCDE, GCDS, and DDPM all employ
neural networks and therefore require proper
hyperparameter tuning, which can be chal-
lenging in complex conditional distribution set-
tings. In particular, GCDS relies on a min-max
distribution-matching objective and is often sen-
sitive to hyperparameter choices and prone to
training instability, based on our empirical ex-
periences.
The classical dimension-reduction method for
conditional distribution estimation proposed by
Hall and Yao is generally less competitive than
the other four approaches considered in our
study.
18

[CAPTION] Table 2: Comparison of training eﬀiciency across DeepCDE, GCDS, and DDPM. Average (standard


<!-- page 19 -->
References
Austin, T. (2015). Exchangeable random mea-
sures.
Ann. Inst. Henri Poincaré Probab.
Stat., 51(3):842–861.
Bauer, B. and Köhler, M. (2019). On deep learn-
ing as a remedy for the curse of dimensional-
ity in nonparametric regression. The Annals
of Statistics, 47(4):2261–2285.
Bonneel, N., Rabin, J., Peyré, G., and Pfis-
ter, H. (2015). Sliced and Radon Wasserstein
barycenters of measures. Journal of Mathe-
matical Imaging and Vision, 51(1):22–45.
Dalmasso, N., Pospisil, T., Lee, A. B., Izbicki,
R.,
Freeman,
P.
E.,
and
Malz,
A.
I.
(2020). Conditional density estimation tools
in python and r with applications to photo-
metric redshifts and likelihood-free cosmolog-
ical inference.
Astronomy and Computing,
30:100362.
Dhariwal, P. and Nichol, A. (2021). Diffusion
models beat gans on image synthesis.
Ad-
vances in neural information processing sys-
tems, 34:8780–8794.
Flamary, R., Courty, N., Gramfort, A., Alaya,
M. Z., Boisbunon, A., Chambon, S., Chapel,
L., Corenflos, A., Fatras, K., Fournier, N.,
et al. (2021).
Pot: Python optimal trans-
port. Journal of Machine Learning Research,
22(78):1–8.
Fu, H., Yang, Z., Wang, M., and Chen, M.
(2024).
Unveil conditional diffusion models
with classifier-free guidance: A sharp statisti-
cal theory. arXiv preprint arXiv:2403.11968.
Goodfellow, I., Bengio, Y., and Courville, A.
(2016).
Deep Learning.
MIT Press.
http:
//www.deeplearningbook.org.
Goodfellow, I. J., Pouget-Abadie, J., Mirza, M.,
Xu, B., Warde-Farley, D., Ozair, S., Courville,
A., and Bengio, Y. (2014). Generative adver-
sarial nets. Advances in neural information
processing systems, 27.
Hall, P., Wolff, R. C., and Yao, Q. (1999). Meth-
ods for estimating a conditional distribution
function. Journal of the American Statistical
association, 94(445):154–163.
Hall, P. and Yao, Q. (2005).
Approximat-
ing conditional distribution functions using
dimension reduction.
Annals of Statistics,
33:1404–1421.
Han, X., Zheng, H., and Zhou, M. (2022). Card:
Classification and regression diffusion models.
Advances in Neural Information Processing
Systems, 35:18100–18115.
Henzi, A., Kleger, G.-R., and Ziegel, J. F.
(2023). Distributional (single) index models.
Journal of the American Statistical Associa-
tion, 118(541):489–503.
Hinton, G. E. and Salakhutdinov, R. R. (2006).
Reducing the dimensionality of data with
neural networks. science, 313(5786):504–507.
Ho, J., Jain, A., and Abbeel, P. (2020).
De-
noising diffusion probabilistic models.
Ad-
vances in neural information processing sys-
tems, 33:6840–6851.
Izbicki, R. and Lee, A. B. (2017).
Con-
verting high-dimensional regression to high-
dimensional conditional density estimation.
Electron. J. Statist., 11(2):2800–2831.
Kallenberg, O. (2021). Foundations of Modern
Probability. Springer.
Nguyen, X., Wainwright, M. J., and Jordan,
M. I. (2010). Estimating divergence function-
als and the likelihood ratio by convex risk
minimization. IEEE Transactions on Infor-
mation Theory, 56(11):5847–5861.
19


<!-- page 20 -->
Ramdas, A., García Trillos, N., and Cuturi, M.
(2017).
On Wasserstein two-sample testing
and related families of nonparametric tests.
Entropy, 19(2):47.
Schmidt-Hieber, J. (2020). Nonparametric re-
gression using deep neural networks with
ReLU activation function.
The Annals of
Statistics, 48(4):1875–1897.
Song, J., Meng, C., and Ermon, S. (2021).
Denoising diffusion implicit models.
Pro-
ceedings
of
the
International
Conference
on Learning Representations.
Available at
arXiv:2010.02502.
Spokoiny, V., Wang, W., and Härdle, W. K.
(2013).
Local quantile regression.
Jour-
nal of Statistical Planning and Inference,
143(7):1109–1129.
Yang, Y., Li, S., Zhang, Y., Sun, Z., Shu, H.,
Chen, Z., and Zhang, R. (2025).
Condi-
tional diffusion models based conditional in-
dependence testing.
In Proceedings of the
AAAI Conference on Artificial Intelligence,
volume 39, pages 22020–22028.
Yu, K. and Jones, M. (1998).
Local linear
quantile regression. Journal of the American
statistical Association, 93(441):228–237.
Zhou, X., Jiao, Y., Liu, J., and Huang, J. (2023).
A deep generative approach to conditional
sampling. Journal of the American Statistical
Association, 118(543):1837–1848.
20


<!-- page 21 -->
Figure 1: True conditional densities for models M1-M5.
21

[CAPTION] Figure 1: True conditional densities for models M1-M5.


<!-- page 22 -->
Figure 2: True conditional densities for models M6-M9.
22

[CAPTION] Figure 2: True conditional densities for models M6-M9.