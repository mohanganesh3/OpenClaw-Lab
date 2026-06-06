<!-- page 1 -->
DDPM Score Matching and Distribution Learning
Sinho Chewi
Alkis Kalavasis
Anay Mehrotra
Omar Montasser
Yale University
Abstract
Score estimation is the backbone of score-based generative models (SGMs), and particularly
denoising diffusion probabilistic models (DDPMs). A fundamental theoretical result in this area
is that, given access to accurate score estimates, SGMs can efficiently generate from any realistic
data distribution (Chen, Chewi, Li, Li, Salim, and Zhang, ICLR’23; Lee, Lu, and Tan, ALT’23).
This can be viewed as a result on distribution learning, where the learned distribution is implicit
as the law of the output of a sampler. However, it is unclear how score estimation relates to more
classical forms of distribution learning, such as parameter estimation and density estimation.
The contribution of this paper is a framework that reduces from score estimation to the other
two forms of distribution learning. This framework has various implications in statistical and
computational learning theory:
▶(Parameter Estimation)
Recent work has shown that for estimation in parametric models,
a variant of score matching known as implicit score matching is provably statistically
inefficient for multimodal densities that are common in practice (Koehler, Heckett, and
Risteski, ICLR’23). In contrast, under mild conditions, we show that denoising score
matching in DDPMs is asymptotically efficient, i.e., the DDPM estimator is asymptotically
normal with a covariance matrix given by the inverse Fisher information.
▶(Density Estimation)
Given the reduction from generation to score estimation, there is
a large volume of work providing statistical and computational guarantees for learning
the score of a distribution. Using our framework, we can lift the estimated scores to a
(ε, δ)-PAC density estimator, i.e., a function that ε-approximates the target log-density in
all but a δ-fraction of the space. To illustrate our framework, we provide two results: (i)
minimax rates for density estimation over H¨older classes of densities in the standard L1 risk
and (ii) a quasi-polynomial PAC density estimation algorithm for the classical Gaussian
location mixture model. For the latter result, our result builds on and answers an open
problem in the recent work of Gatmiry, Kelner, and Lee (arXiv’24).
▶(Lower Bounds for Score Estimation)
Our framework provides the first principled way to
prove computational lower bounds for score estimation for general families of distributions.
As an application, we prove cryptographic lower bounds for score estimation of general
Gaussian mixture models, conceptually recovering the results of Song (NeurIPS’24) and
making important progress to Song’s key open problem.
arXiv:2504.05161v1  [stat.ML]  7 Apr 2025


<!-- page 2 -->
Contents
1
Introduction
1
1.1
Framework and first main tool
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
2
1.1.1
Likelihood identity . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
2
1.2
Applications to parameter estimation . . . . . . . . . . . . . . . . . . . . . . . . . . .
3
1.2.1
DDPM is an asymptotically efficient parameter estimator . . . . . . . . . . .
4
1.3
Applications to density estimation
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
5
1.3.1
PAC density estimation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
5
1.3.2
Minimax optimal density estimation over H¨older classes . . . . . . . . . . . .
6
1.3.3
Algorithms for PAC density estimation via score matching . . . . . . . . . . .
6
1.4
Applications to lower bounds for score estimation . . . . . . . . . . . . . . . . . . . .
8
1.4.1
Cryptographic lower bounds for score estimation . . . . . . . . . . . . . . . .
8
1.5
Second main tool: Reduction from score estimation to density estimation
. . . . . .
9
1.6
Other related work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
10
1.7
Discussion and open problems . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
11
1.8
Notation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
12
2
Main tools
13
2.1
Connection between log-likelihood and DDPM score estimation (Lemma 1)
. . . . .
13
2.2
Score estimation implies PAC density estimation . . . . . . . . . . . . . . . . . . . .
13
2.2.1
Relevant oracles
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
14
2.2.2
Sub-Gaussianity of the score
. . . . . . . . . . . . . . . . . . . . . . . . . . .
14
2.2.3
Score estimation implies integrated score estimation
. . . . . . . . . . . . . .
16
2.2.4
Integrated score estimation implies PAC density estimation . . . . . . . . . .
20
2.2.5
Completing the reduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
22
2.3
Early stopping
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
23
2.4
Application to estimating the differential entropy . . . . . . . . . . . . . . . . . . . .
23
3
DDPM is an asymptotically efficient parameter estimator
23
3.1
Implications of the likelihood identity for parameter estimation . . . . . . . . . . . .
24
3.2
The proof of Informal Theorem 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
24
4
Minimax optimal density estimation over the H¨older class
26
5
PAC density estimation for Gaussian location mixtures
29
5.1
Proof of Theorem 5.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
31
6
Cryptographic lower bounds for score estimation
34
6.1
PAC density estimation for GMMs implies homogeneous CLWE . . . . . . . . . . . .
35
6.2
Lower bounds for score estimation for GMMs . . . . . . . . . . . . . . . . . . . . . .
38
6.2.1
Score estimation for GMMs implies (homogeneous) CLWE
. . . . . . . . . .
39
6.2.2
LWE-hardness of score estimation for GMMs . . . . . . . . . . . . . . . . . .
40
A Background on denoising diffusion probabilistic modeling
51


<!-- page 3 -->
B Background on LWE and Continuous LWE
52
B.1
Learning with errors . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
52
B.2
Background on lattices and discrete Gaussians
. . . . . . . . . . . . . . . . . . . . .
53
B.2.1
Discrete Gaussian sampling problem . . . . . . . . . . . . . . . . . . . . . . .
53
B.3
Continuous LWE . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
54
B.3.1
CLWE distributions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
54
B.3.2
CLWE decision problems
. . . . . . . . . . . . . . . . . . . . . . . . . . . . .
54
B.3.3
Hardness of CLWE . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
55
C Auxiliary lemmas
56
C.1
Standard facts about sub-Gaussianity
. . . . . . . . . . . . . . . . . . . . . . . . . .
56
C.2
Integral estimates . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
56
C.3
Facts about the H¨older class . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
57
D Further related work
57
D.1 Importance of density estimation . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
58
D.2 History of evaluators and generators for GMMs . . . . . . . . . . . . . . . . . . . . .
58


<!-- page 4 -->
1
Introduction
Score-based generative models (SGMs), also known as diffusion models, have emerged as a popular
approach to generate samples from complex data distributions. These models leverage learned score
functions—that is, the logarithmic gradients of the probability density—to progressively transform
white noise into samples from the target data distribution by following a stochastic differential
equation (SDE). The remarkable empirical success of SGMs has not only led to impressive practical
applications but has also spurred significant interest within the theoretical computer science (e.g.,
[PRSLMR23; SCK23; CKS24; GKL24]) and the machine learning and statistics communities (e.g.,
[CCLLLS23; CCLLSZ23; KHR23; OAS23; DKXZ24; KV24; QR24; WWY24]) toward establishing
rigorous theoretical foundations for SGMs.
More specifically, SGMs evolve the data distribution along a noising process, producing a
family of distributions (Pt)t∈[0,T].
In this work, for concreteness, we focus exclusively on the
Ornstein–Uhlenbeck (OU) process, for which
Pt := Law(Xt) := Law(e−t X0 + (1 −e−2t) Zt) ,
X0 ∼P , Zt ∼N(0, Id) , X0 ⊥⊥Zt .
Then, to generate a sample from P = P0, SGMs numerically discretize a certain SDE (obtained as
the time reversal of the noising process), the implementation of which crucially requires estimation
of the score functions {∇log Pt}t∈[0,T]. We refer the reader to Appendix A for background.
A central component underlying this procedure is score estimation [Hyv05], which transforms the
problem of learning the score function into a regression objective amenable to first-order optimization.
It relies on the following key identity, valid for any vector field st : Rd →Rd:
E
 
∥st(Xt) −∇log Pt(Xt)∥2    X0
 
= E
h
∥st(Xt)∥2 +
2
√
1 −e−2t ⟨st(Xt), Zt⟩
    X0
i
+ C(P, X0) , (1)
where C(P, X0) is a constant that does not depend on st.1 Since we can freely generate Zt (and
thus Xt) given X0, the right-hand side of the identity above is readily turned into an empirical loss
that can be minimized over the choice of st.
Despite a number of recent works investigating its efficacy [KLV24; KV24; QR24], a complete
statistical understanding of score matching remains to be developed. In this work, we aim to clarify
the relationships between this problem and the well-studied problem of distribution learning [Hau92;
KMRRSS94]. Concretely, we study the following general question.
Main Question: Given a family P of probability distributions and samples from P ∈P,
how does the complexity (both computational and statistical) of learning the score functions
{∇log Pt}t∈[0,T] relate to the complexity of learning the distribution P?
In order to make this question precise, we must specify the sense in which we learn the distribution.
Classically, the three most common forms of distribution learning are the following.
▶Parameter estimation: If P = {Pθ : θ ∈Θ} is indexed by a finite-dimensional parameter
space Θ ⊆Rp, then we can ask to output an estimate of the true value of the parameter.
▶Density estimation: We can also ask to output an evaluator, that is, a function (or algorithm)
bP which, given an input x, yields an estimate bP(x) of the density of P evaluated at x. When
1The expectation above is often replaced with E[∥st(Xt)+(Zt/
p
1 −exp(−2t))∥2 | X0], which is formally equivalent
by completing the square. However, we prefer to write the identity as above since
R T
0 E[∥Zt/
p
1 −exp(−2t)∥2] dt = ∞.
1


<!-- page 5 -->
computational considerations are at play, we further ask that bP runs in polynomial time
(with respect to various problem parameters). This setting is especially well-established in the
statistics literature [Tsy09].
▶Learning a sampler: A more recent paradigm2 instead asks to learn a generator, that
is, a function (or algorithm) which, given a random seed, outputs a sample b
X such that
Law( b
X) ≈P. Again, when computational resources are a concern, we ask that the generator
runs in polynomial time.3
SGMs most naturally correspond to the last paradigm since they are typically viewed as samplers.
The prior works [CCLLSZ23; LLT23] made this connection rigorous by showing that, provided the
score functions for P are Lipschitz, score estimation implies that one can learn a sampler.
More precisely, if ∇log Pt is L-Lipschitz for all t ≥0 and P ∈P, and we have score estimates
{bst}t≥0 satisfying the guarantee supt≥0 ∥bst −∇log Pt∥2
L2(Pt) ≤ε2, then SGMs can learn a sampler
up to error eO(ε) in total variation distance using a number of score evaluations and additional
computation time that scales polynomially in L, the dimension d, and the desired accuracy 1/ε.4
1.1
Framework and first main tool
Informally, the main message of this paper is that the other two main settings for distribution
learning, namely parameter estimation and density estimation, can both be reduced to score
estimation as well (see Figure 1).
Figure 1: Landscape of reductions between score estimation and distribution learning (see Sec-
tion 1.3.1 for a definition of PAC density estimation). Prior to our work, the only known reduction
was from generation to score estimation. Our work shows that score estimation has strong impli-
cations for parameter recovery and density estimation. We introduce the notion of DDPM score
estimation in Definition 1 and provide relevant background in Appendix A.
Given the above landscape of computationally efficient reductions from density estimation and
parameter estimation to score estimation, we can obtain several new results in statistical and
computational learning theory, which we review in the upcoming sections. Before that, we present
the following tool, which is a core part of all of our results.
1.1.1
Likelihood identity
The starting point for this work is the following identity for the likelihood. This identity relates
the log-likelihood of any point x0 under the target density P and a certain integrated score
2We note, however, that the definition goes back at least to [KMRRSS94].
3If we ignore computational aspects, then evaluators and generators are equivalent objects: given a generator
specified by, e.g., a circuit, one can estimate the relative frequency of each point in its support to sufficient accuracy
to estimate the density at each point; and given an evaluator, one can generate a sample via rejection sampling.
4See Section 1.6 for further developments.
2

[CAPTION] Figure 1: Landscape of reductions between score estimation and distribution learning (see Sec-


<!-- page 6 -->
matching formula corresponding to DDPM score matching where the OU process is run until time
T starting from x0. Here, we use Qt|0 to denote the transition semigroup of the OU process, i.e.,
Qt|0(· | x0) := N(e−t x0, (1 −e−2t) Id).
Lemma 1 (Likelihood identity). Let P be a continuous density over Rd with finite second moment.
Then, for all x0 ∈Rd,
Z
log PT dQT|0(· | x0) −
log P(x0)
|
{z
}
log-density at x0
=
Z T
0
Z  
∥∇log Pt∥2 −2 ⟨∇log Pt, ∇log Qt|0(· | x0)⟩
	
dQt|0(· | x0) dt
|
{z
}
integrated DDPM score matching objective at x0
+
d · T
| {z }
known constant
.
Variants of this identity have been observed previously in the literature, e.g., in [LY24] or in [CLT22]
in the context of the diffusion Schr¨odinger bridge. In fact, a similar formula was already put forth in
the work of Song, Durkan, Murray, and Ermon [SDME21], where it was presented as a variational
lower bound on the log-likelihood, although the connection likely dates back even earlier to ideas
by Jarzynski [Jar97]. For completeness, we prove Lemma 1 in Section 2.1. In any case, while we
do not claim novelty for Lemma 1, our contribution is to thoroughly explore its consequences for
distribution learning.
The power of the identity is that, by the convergence of the OU process, the first term on
the left-hand side converges exponentially fast, as T →∞, to a known constant (see Lemma 7
below). Moreover, the integral on the right-hand side exactly corresponds to the score-matching loss
from (1) which is the (single-sample version of the) DDPM objective evaluated at x0 (for details,
we also refer to Definition 1). We call this quantity the integrated DDPM score matching objective.
Therefore, Lemma 1 shows that the negative log-likelihood −log P is precisely related, up to a
known constant and a vanishing error, to the integrated score-matching objective.
In the upcoming sections, we discuss some applications of Lemma 1 to parameter and density
estimation.
1.2
Applications to parameter estimation
In this section, we assume that our class of distributions is a parametric family P = {Pθ : θ ∈Θ}
with parameter space Θ ⊆Rp. In this setting, the prior work of Koehler, Heckett, and Risteski
[KHR23] investigated the performance of a variant of score matching known as implicit score
matching (ISM): Given i.i.d. samples X(1), . . . , X(n) from Pθ⋆, for some parameter θ⋆∈Θ, the ISM
estimator is
bθ ISM
n
:= arg min
θ∈Θ
1
n
Xn
i=1
 
∥∇log Pθ(X(i))∥2 + 2 ∆log Pθ(X(i))
	
.
Under appropriate regularity conditions, they proved that bθ ISM
n
is asymptotically normal, i.e.,
√n (bθ ISM
n
−θ⋆)
d−→N(0, ΣISM(θ⋆)). Moreover, they bounded the operator norm of ΣISM(θ⋆) in terms
of the asymptotic covariance of the maximum likelihood estimator (MLE)—i.e., the inverse Fisher
information matrix—and the so-called restricted Poincar´e constant of the model. This shows that
the aforementioned parameter estimator based on minimizing the implicit score matching loss can
achieve the asymptotic efficiency of MLE up to a constant factor for distributions whose restricted
Poincar´e constant is O(1). While their result is insightful for many families, the dependency on the
restricted Poincar´e constant is not harmless: it can be arbitrarily large for multimodal distributions,
3


<!-- page 7 -->
which frequently arise in practice. Unfortunately, this dependence is also unavoidable, since Koehler,
Heckett, and Risteski [KHR23] also exhibited examples in which implicit score matching is
provably inefficient compared to MLE; see also Diao [Dia23].
Follow-up work by Qin and Risteski [QR24] generalized this asymptotic efficiency result to
generalized (implicit) score matching estimators [Lyu09] by establishing a connection between the
mixing time of broad classes of Markov processes, and the statistical efficiency of an appropriately
chosen generalized score matching loss (GISM). Under this framework, they managed to show that
for Gaussian mixtures in d dimensions, the generalized score estimator is asymptotically normal
with covariance matrix ΣGISM(θ⋆) which has an operator norm that is, roughly speaking, at most
poly(d) times the (squared) operator norm of the inverse Fisher information (bypassing the lower
bounds of [KHR23]).
In short, both works [KHR23; QR24] indicated strong statistical properties of (generalized) ISM.
That said, they still cannot match the performance of MLE or come within a constant factor of it
for general families P, and they left open whether some diffusion-based estimator can achieve the
statistical efficiency of MLE under mild assumptions on P.
Here, we consider denoising diffusion probabilistic models (DDPMs)—arguably the most popular
variant used in practice—which interestingly do not rely on (generalized) implicit score matching.
Instead, DDPMs employ an alternative known as denoising score matching [Hyv08; Vin11] and
extend the idea by applying score matching at many different noise levels [SE19; HJA20; Yan+23],
leading to the following risk:
E
hZ T
0
 
∥st(Xt)∥2 +
2
√
1 −e−2t ⟨st(Xt), Zt⟩
	
dt
    X0
i
.
(2)
1.2.1
DDPM is an asymptotically efficient parameter estimator
We consider the following DDPM estimator, which precisely amounts to minimizing the DDPM risk
in Equation (2) over samples from P.
Below Pθ,t denotes the law of Xt := e−t X0 +
√
1 −e−2t Zt, where X0 ∼Pθ and Zt ∼N(0, Id)
are independent. We provide relevant background on the DDPM objective in Appendix A.
Definition 1. Fix a terminal time T > 0. Given samples X(1)
0 , . . . , X(n)
0
and a family P = {Pθ :
θ ∈Θ ⊆Rp}, the DDPM estimator is bθ DDPM
n
:= arg minθ∈Θ bRDDPM
n
(θ), where
bRDDPM
n
(θ) := 1
n
n
X
i=1
Z T
0
E
h
∥∇log Pθ,t(X(i)
t )∥2 +

∇log Pθ,t(X(i)
t ),
2Z(i)
t
√
1 −e−2t
      X(i)
0
i
dt
and for each i ∈[n] and t ∈[0, T], we draw Z(i)
t
∼N(0, Id) independently from X(i)
0
and define the
noised sample X(i)
t
:= e−t X(i)
0
+
√
1 −e−2t Z(i)
t .
Our main result for this application is that, under mild regularity assumptions on the distribution
family P (essentially the same conditions needed for the asymptotic normality of the MLE, see As-
sumption 3) and by choosing the terminal time T = Tn to grow sufficiently rapidly with the number
of samples n (namely, Tn −1
2 log n →∞), the DDPM estimator bθ DDPM
n
converges in distribution to
a Gaussian centered at θ⋆with covariance exactly equal to the inverse Fisher information. Recall
that the inverse Fisher information is also the asymptotic covariance of the MLE and is the best
possible for any unbiased estimator (by the Cram´er–Rao or information inequality) [Vaa98], so this
statement can be interpreted as a form of asymptotic optimality; furthermore, by comparison of
experiments, the MLE can be shown to be locally asymptotically minimax [H´aj72; Vaa98].
4


<!-- page 8 -->
To state the result formally, let bθ DDPM
n
denote the DDPM estimator as defined in Definition 1
on n i.i.d. samples X(1)
0 , . . . , X(n)
0
∼Pθ⋆.
Informal Theorem 1 (DDPM is asymptotically efficient; informal, see Theorem 3.1). Under
standard assumptions,
√n
 bθ DDPM
n
−θ⋆  d−→N(0, I(θ⋆)−1)
as
n →∞,
where I(θ⋆) denotes the Fisher information matrix at θ⋆.
Informal Theorem 1 provides a principled explanation for the statistical power of the DDPM
estimator in the asymptotic regime and has immediate implications for parameter recovery. For
more details, we refer to Section 3.
1.3
Applications to density estimation
A long line of works showed that, under minimal regularity assumptions, access to a score estimation
oracle for P is sufficient for learning to generate from the target density; see Section 1.6 for a
discussion of this literature. However, the precise connection between score estimation and density
estimation remains elusive. One of the main results of our framework is an efficient reduction from
a particular notion of density estimation to score estimation, which we define below.
1.3.1
PAC density estimation
To formally define our notion of density estimation, we need the following evaluation oracle:
Definition 2 (Evaluation oracle). An evaluation oracle for a function f : Rd →R is a primitive
that, given a point x ∈Rd, outputs f(x).
Density estimation is extremely well-studied in statistics and computer science. Prior works on
density estimation (see e.g., [BC91; KMRRSS94; FSO06; KMV10; MV10; DK14; SOAJ14; Dia16;
LS17; ABHLMP18; DKKLMS19; DK20; BDJKKV22]) focus on finding a model bP (i.e., in the form
of an evaluation oracle for the target density) such that, with high probability over the training set,
it estimates the target density P with high accuracy. There are many variants of density estimation
which differ in the specific evaluation metric used; e.g., it can be the total variation distance or KL
divergence [KMRRSS94; Dia16; DK20].
We are now ready to define the notion of probably approximately correct (PAC) density
estimation, which is a slight relaxation of the above requirement.
The goal in PAC density
estimation is to output a model bP that fits the target density P everywhere except for a δ-fraction
of points (according to the probability mass of the target distribution P), with high probability.
Definition 3 (PAC density estimation algorithm). Let P be a class of distributions over Rd. An
(ε, δ)-PAC density estimation algorithm for P is an algorithm which, for any P ∈P, given ε, δ > 0
and poly(d, 1/ε, 1/δ) i.i.d. samples drawn from distribution P, outputs a representation (in the form
of an evaluation oracle) of a possibly randomized function bP such that with probability at least 9/10
over the randomness of the samples and the algorithm,
E P{x ∈Rd : e−ε P(x) ≤bP(x) ≤eε P(x)} ≥1 −δ ,
where E denotes the expectation over the randomness of bP.
5


<!-- page 9 -->
In the above definition, we often call ε the accuracy of the algorithm and δ its coverage. If additionally
the (ε, δ)-PAC learner runs in sample polynomial time, we call it efficient.5
Some further remarks are in order. We do not require the function bP to be a density. In this
sense, bP multiplicatively estimates the density of P on most of the domain, but is not a density
itself. For parametric families, this means that PAC density estimation is weaker, perhaps strictly,
than parameter estimation which would provide density estimation on the entire domain. Omitting
this requirement is standard in both theoretical computer science (e.g., [CDSS14; ADLS17]) and
(non-parametric) statistics [DL01; Sco15] (e.g., kernel density estimators can output functions that
neither integrate to 1 nor have a non-negative range). The PAC density estimators that we design
are not densities (i.e., they do not integrate to 1) but they approximate the target density P
everywhere except for a δ-fraction of the points and take non-negative values.
To further motivate the non-triviality of getting such a PAC guarantee, we give evidence that
this problem is essentially as hard as standard density estimation for standard families: We show
that for H¨older classes, existing minimax lower bounds also hold against PAC density estimation
(Section 4), and that PAC density estimation for Gaussian mixtures can be used to solve (C)LWE
and is therefore cryptographically hard in some regime (Section 6.1). Such a reduction was known
for the stronger notion of density estimation in total variation and is an important reason why
density estimation is believed to be computationally challenging [BRST21; GVV22].
The success probability of 9/10 in Definition 3 is arbitrary and can straightforwardly be boosted
to achieve (ε, 3δ)-PAC density estimation with a success probability (9/10)k by computing the median
of k (ε, δ)-PAC density estimators with success probability 9/10.
1.3.2
Minimax optimal density estimation over H¨older classes
There is a long line of works that studied statistical rates for score estimation and thereby obtained
minimax optimal generators. Using our tools, we can directly translate these results to PAC density
estimators (we briefly overview this below and refer the reader to Section 4 for more details).
To see an application, we consider a representative result on score estimation coming from the
recent work of Dou, Kotekal, Xu, and Zhou [DKXZ24]. Their work considered the H¨older class of
densities (see Definition 7) and showed that the score ∇log Pt can be approximated in L2(Pt) at
a certain rate for any P in that class. Prior to our work, such a result had only implications to
generation; the next result converts such a guarantee to a density estimation result. We mention
that for this section, we actually achieve a stronger guarantee compared to PAC density estimation:
we give minimax optimal rates for estimation in the standard L1 risk using an estimator based on
DDPM score matching.
Informal Theorem 2 (Density estimation for H¨older classes; informal, see Theorem 4.1). Let Hs
be the H¨older class of densities P supported on [−1, 1] with smoothness parameter s > 0. Given a
rate function r : N →R and a number n ≥1, define the risk of an estimator bP using n samples to
be Rn( bP, P) :=
R
[−1,1] EP | bP(x0) −P(x0)| dx0. Then, an estimator based on DDPM score estimation
achieves the minimax risk n−s/(2s+1) over Hs up to a √log n factor.
1.3.3
Algorithms for PAC density estimation via score matching
Apart from statistical implications, a key conceptual message for our density-to-score estimation
reduction is that it is also computationally efficient. We believe that this can lead to novel algorithms
5As is standard, efficiency has two aspects: both (i) producing the output bP and (ii) evaluating bP(x) at any given
point x at inference time should run in time that is polynomial in the number of samples.
6


<!-- page 10 -->
for PAC density estimation. To illustrate this, we demonstrate a perhaps surprising application to
PAC density estimation for the following Gaussian location mixture, which goes back to Kiefer and
Wolfowitz [KW56] and was studied from an algorithmic perspective in a recent work by Gatmiry,
Kelner, and Lee [GKL24]: Given distribution Q, the corresponding Gaussian location mixture is
P = Q ∗N(0, σ2 Id) ,
for some variance parameter σ2 (controlling the smoothness of the target density). For instance,
when Q is supported on k discrete points, then P is a spherical Gaussian mixture model. In general,
there is a long list of works where the mixing distribution Q is non-parametric (see, e.g., [GW00;
GV01; Zha09; Kim14; SG20; PS25]). Following Gatmiry, Kelner, and Lee [GKL24], it is assumed
that Q satisfies the following structural properties (see Definition 8):
1. The support of Q is contained in k ℓ2-balls of radius R.
2. The ℓ2-ball of radius R around any point of the support of Q has mass at least 1/k.
3. The support of Q is a subset of the ℓ2-ball of radius D centered at the origin.
Our starting point is the important recent work by Gatmiry, Kelner, and Lee [GKL24] that provides
a generator for this family of multi-modal densities. Their algorithm is based on score estimation,
runs in time roughly dpoly(log(1/ε),R/σ) when ε < min {σ/R, 1/D, 1/d, 1/k}, and learns a sampler for the
target density, using the well-known reduction from generation to score estimation.
There are several reasons why this is interesting. First, [GKL24] learn to generate from various
non-parametric models for which no sub-exponential-in-d algorithms were previously known. In
particular, this applies to the case of Gaussian convolutions of distributions on low-dimensional
manifolds or, more generally, sets with a small covering number. Moreover, their algorithm provides
an alternative to the algebraic toolbox employed for the density estimation of spherical GMMs.
Indeed, the algorithm of Diakonikolas and Kane [DK20] outputs a density estimator in time
poly(dk/ε) + (k/ε)O(log2 k) for mixtures of k spherical Gaussians using tools from algebraic geometry
(tailored to GMMs) and it does not seem to extend to the more general distribution class of Gaussian
location mixtures. Even providing a generator for such a general problem with qualitatively similar
runtime when specialized to the GMM setting is a surprising algorithmic result.
However, it does not have any implications for the density estimation problem studied by the
majority of works on GMMs. Gatmiry, Kelner, and Lee [GKL24] mention that it may be possible to
upgrade their algorithm’s guarantee from generation to density estimation but leave it as an open
problem. We make progress on their open question by establishing the following result.
Informal Theorem 3 (PAC density estimation for Gaussian location mixtures; see Theorem 5.1
for the precise statement). Let M be an (k, R, D)-Gaussian location mixture over Rd with σ ∈(0, 1].
Fix ε ≤min{1/2, σ/R, 1/D, 1/d, 1/k}. There is an algorithm that, given accuracy parameter ε, instance
parameters (σ, k, R, D), and sample access to M, draws N i.i.d. samples from M, runs in poly(N)
time, and, for any coverage parameter δ ∈(0, 1), returns an (ε/δ, δ)-PAC density estimator for M,
where
N =
 
d log 1
ε
 polylog(1/ε)+poly(R/σ)
.
In other words, we match the sample complexity and runtime of the algorithm of [GKL24], but
instead of a generator for the class of interest, we return a PAC density estimator.
7


<!-- page 11 -->
1.4
Applications to lower bounds for score estimation
Up to now, we have shown that score estimation implies powerful results for distribution learning,
both from a statistical and a computational lens, beyond the ability to generate samples. In this last
application, we ask the following natural question: Do our reductions lead to impossibility results
for score estimation itself?
1.4.1
Cryptographic lower bounds for score estimation
In this section, we show that our reduction, from PAC density estimation to score estimation,
allows us to deduce computational bottlenecks to score estimation from the hardness of the latter.
Importantly, prior to this reduction, there was no principled method to demonstrate the hardness of
score estimation; in particular, the existing reductions from score estimation to generation did not
imply hardness for score estimation as we do not yet have tools for showing complexity-theory-based
hardness for generators (see Open Problem 5 in Section 1.7). Since our reduction from PAC density
to score estimation is efficient, we obtain the following abstract tool to show computational hardness
for score estimation.
Blueprint for computational lower bounds for score estimation under certain condition C
1. Show that PAC density estimation for P is computationally hard under condition C.
2. Show that P has Lipschitz scores and bounded second moments.
As an illustration of the type of hardness results that can be obtained from our reduction, we
show that it implies the following hardness result, which (conceptually) recovers the very recent
lower bound of Song [Son24].6 In particular, we show that, under standard hardness of Learning
with Errors (LWE), score estimation for general GMMs with at least d ε components (or even
(log d)1+ε under a stronger condition, following Gupte, Vafa, and Vaikuntanathan [GVV22]) requires
super-polynomial time in the dimension for any constant ε.
In order to get this result, it suffices to apply our blueprint for GMMs: we must show that
PAC density estimation for GMMs is computationally intractable under some standard complexity
assumption. Following Bruna, Regev, Song, and Tang [BRST21], we show that CLWE reduces
to it. To complete the reduction, we have to show that our “hard” GMM instance satisfies the
assumptions of our density-to-score estimation reduction, leading to the following result.
Informal Theorem 4 (Hardness of score estimation for Gaussian mixtures; see Theorem 6.2).
Under polynomial hardness of LWE, score estimation for Gaussian mixtures with k ≳d ε components
for any constant ε > 0 cannot be done in poly(d, k) time with accuracy equal to 1/√d log d.
A result similar in spirit to Informal Theorem 4 is the main result of Song [Son24], with a very
different technique, specialized to the Gaussian pancakes distribution. A comparison with [Son24]
appears in Section 1.6. Our reduction, however, is rooted in a general blueprint for deriving lower
bounds for score estimation, leaving open the possibility of extensions to other distribution families.
Moreover, it already has implications for Song [Son24]’s open problem on finding natural
assumptions on data distributions that eliminate Gaussian pancakes while allowing for rich data
6The work of Song [Son24] showed a reduction from distinguishing a Gaussian pancake and the standard Gaussian
to score estimation of a Gaussian pancake. Since Gaussian pancakes are morally behind cryptographic lower bounds
for GMMs [BRST21], one can obtain a series of cryptographic hardness results for GMM score estimation (which are
not explicitly stated in Song [Son24]). We recover these cryptographic hardness results for GMM score estimation.
8


<!-- page 12 -->
distributions encountered in practice. Our results make significant progress on this by showing that
L2 score estimation is computationally hard for any family of distributions for which evaluating the
density is computationally hard, in the sense of PAC density estimation.
The formal version of Informal Theorem 4 (Theorem 6.2) and its proof appear in Section 6.
1.5
Second main tool: Reduction from score estimation to density estimation
In this section, we present our second main tool (apart from the likelihood identity presented earlier
in Lemma 1), which is our key technical contribution and will be used for our applications in density
estimation (Section 1.3) and computational hardness results for score estimation (Section 1.4). In
particular, we present a reduction from PAC density estimation to score estimation under mild
assumptions on the target density P, which is the analogue of the standard reduction from generation
to score estimation [CCLLSZ23; LLT23].
Before delving into the details, let us first introduce the score estimation oracle.
Definition 4 (Score estimation oracle; informal, see Definition 5). Let P be a density on Rd. A
score estimation oracle for P gets as input t and outputs a model bst with
R
∥bst −∇log Pt∥2 dPt ≤ε2
t .
We define the error of the oracle with early stopping τ > 0 and terminal time T as ε2
∗, where
ε2
∗=
Z T
τ
ε2
t dt .
In our reduction, we obtain a PAC density estimator given access to a score estimation oracle.
Informal Theorem 5 (Score estimation to PAC density estimation; informal, see Theorems 2.3
and 2.4). Let P be a distribution on Rd, and let ε > 0 be the desired accuracy. Assume access to a
score estimation oracle with early stopping τ and error ε∗.
1. Assume that P has second moment bounded by M2 ≤poly(d) and L-Lipschitz score function.
There is an algorithm that outputs a function bP (in the form of an evaluation oracle) such that
Z
E
  log
bP(x0)
P(x0)
   P(dx0) ≲ε .
The algorithm makes N = eΘ(Ld2/ε2) calls to the score estimation oracle with accuracy ε∗=
eΘ(ε/√d log L) and runs in poly(N) time. The above hold when τ ≲ε2/Ld2.
2. Assume only that P has second moment bounded by M2 ≤poly(d). For any 0 < τ ≤1, there
is an algorithm that outputs a function bPτ (in the form of an evaluation oracle) such that
Z
E
  log
bPτ(x)
Pτ(x)
   Pτ(dx) ≲ε .
The algorithm makes N = eΘ((d2+1/τ)/ε2) calls to the score estimation oracle with accuracy
ε∗= eΘ(ε/√
d log(1/τ)) and runs in poly(N) time.
The details of the above key reductions appear in Section 2.2.
Some remarks are in order. The above result draws parallels with standard results reducing
sample generation to score estimation. Item 1 in Informal Theorem 5 requires bounded second
moment and Lipschitz scores and can be seen as the analogue of the result of [CCLLSZ23; LLT23] in
9


<!-- page 13 -->
the context of density estimation. In Item 2, P is only assumed to have a bounded second moment.
Since it can even be discrete, we can only get guarantees slightly before time 0. Hence, we provide
PAC density estimation guarantees for the early stopped distribution Pτ, which is also common in
the sample generation literature.
The reduction is efficient: given access to the score estimation oracle, the algorithm makes
polynomially many calls to the oracle and runs in polynomial time. Moreover, at inference time,
given any point x0, the estimation of the log-density at x0 takes polynomial time. We remind the
reader that the outputs bP (and bPτ) may not integrate to 1. Hence, the expected value is not an
upper bound for a KL divergence.
The proof of Item 1 proceeds in two steps. The starting point is the likelihood identity of
Lemma 1 which, roughly speaking (for large enough T) says that at any point x0 ∈Rd:
−log P(x0) ≈
Z T
0
Z  
∥∇log Pt∥2 −2 ⟨∇log Pt, ∇log Qt|0(· | x0)⟩
	
dQt|0(· | x0) dt + const .
(3)
Hence, if we could estimate the right-hand side of the above equation, ignoring the absolute constant
term, we would be able to get an estimation for the log-density at x0. For details, we refer to
Section 2.2.4. We call the problem of estimating this double integral integrated (DDPM) score
estimation (see also Figure 1). Converting score estimation to integrated score estimation is the
most technical part of our reduction and appears in Section 2.2.3. The key technical ingredient for
this reduction builds on the recent work of Altschuler and Chewi [AC24] and is likely of independent
interest. Details about the proof appear in Section 2.2.2.
Lemma 2 (Lipschitz score implies sub-Gaussian score). Let P be a distribution on Rd such that
the score ∇log P is L-Lipschitz. Then, for every t ≥0, ∇log Pt is √Lt-sub-Gaussian under Pt,
where Lt ≤2L is explicit (see Lemma 3).
The above discussion sketches the main steps of the reduction for Item 1: the condition that the
score is Lipschitz is used for the transformation from score to integrated score estimation, while the
bound on the second moment is used to convert the integrated score to a PAC density estimator.
For the proof of Item 2, it suffices to apply Item 1 with P equal to Pτ, since its score is sub-Gaussian
with parameter 1/(1 −e−2τ) (see Section 2.3).
1.6
Other related work
Score estimation for generation. There is a vast literature on convergence guarantees for diffu-
sion models, and here we provide a brief discussion on the implications for learning a sampler. The
first works that obtained polynomial-time guarantees for generation from general data distributions
are [CCLLSZ23; LLT23]. These works assumed that the score functions are Lipschitz continuous
uniformly in time and are learned accurately in L2, and obtained TV guarantees. When the data
distribution does not admit a Lipschitz score, one can still obtain TV guarantees for generating
from an early stopped distribution with polynomial complexity. The subsequent works of [CLL23;
BDDD24; CDG25] sharpened the guarantees, replacing the assumption of Lipschitz scores with
the assumption that the initial distribution has finite Fisher information relative to the Gaussian.
The current state-of-the-art runtime guarantee is [LY25], although there have been extensions in
numerous directions, e.g., deterministic samplers, low-dimensional adaptation, and parallelization,
and we do not survey them all here.
Score estimation for parameter recovery.
Closest to our paper is the work of Koehler,
Heckett, and Risteski [KHR23] that studied the implicit score-matching estimator, as discussed
10


<!-- page 14 -->
in the introduction. To the best of our knowledge, the first appearance of an objective such
as Definition 1 for the purpose of point estimation is the work of Shah, Chen, and Klivans [SCK23],
in the context of Gaussian mixture models, which also showed how to algorithmically minimize the
DDPM objective (at carefully selected noise levels). We are not aware of general statistical theory
for bθ DDPM
n
. Most works studying score estimation in DDPM instead considered estimating the score
functions at different times separately (as opposed to bθ DDPM
n
, which finds the value of the parameter
that optimizes an objective using all of the scores). In particular, a line of work showed that score
estimation can achieve minimax rates for density estimation; see Section 4. Finally, we note that
variants of Proposition 1 have appeared in the literature, e.g., Song, Durkan, Murray, and Ermon
[SDME21] showed that the DDPM loss can be pointwise lower bounded in terms of the MLE loss,
Chen, Liu, and Theodorou [CLT22] proved an analogous result for the Schr¨odinger bridge, and Li
and Yan [LY24] established essentially the same formula along a slightly different process. Variants
of Proposition 1 also appeared in a line of works aiming at estimating partition functions [DGMS22;
GTC25] and is related to Jarzynski’s equality from statistical physics [Jar97; VJ08; HRSZ17].
Computational aspects of score estimation. Apart from statistical questions regarding score
matching, there is increasing interest in computational aspects of score estimation. In particular,
Pabbaraju, Rohatgi, Sevekari, Lee, Moitra, and Risteski [PRSLMR23] gave an example of an
exponential family of distributions such that implicit score matching is computationally efficient to
optimize (i.e., finding bθISM
n
can be done efficiently), and has a comparable statistical efficiency to MLE,
while the MLE objective is intractable to optimize using gradient-based methods. Meanwhile, Chen,
Kontonis, and Shah [CKS24] and Gatmiry, Kelner, and Lee [GKL24] used score estimation to
establish new algorithmic results for generating samples from certain families of Gaussian mixtures.
In terms of lower bounds, Song [Son24] reduced the problem of distinguishing Gaussian pancakes
from a standard Gaussian to the problem of score estimation in the L2-norm (with error 1/√log d
which is larger – better – than the error in our hardness result (Informal Theorem 4)). Since the
problem of distinguishing Gaussian pancakes from a Gaussian is cryptographically hard [BRST21;
GVV22], this establishes the cryptographic hardness for L2-score estimation for this specific family.
Due to this and the fact that Song [Son24]’s result implies hardness for larger score estimation
errors, the above result is implied by the work of Song [Son24], however, the techniques in our
work and theirs are quite different: we obtain the result as a natural application of our general
score-to-density framework, while the previous reduction is arguably ad hoc.
Other related work. We discuss further related works specific to each application in the respective
sections, and we provide additional discussions in Appendix D.
1.7
Discussion and open problems
A key contribution of our work is to bridge the extensive literature on score estimation, which
has traditionally focused on generative modeling, with the literature on density and parameter
estimation. We believe that there is ample room to more thoroughly explore this connection and in
light of this perspective, we identify and leave several open problems.
Open problem 1. Using score estimation, is it possible to output a proper density estimator, i.e.,
a function bP that is non-negative and integrates to 1?
In general, it seems intractable to evaluate
R bP to normalize our estimator, and even if we could
compute the integral, we cannot show that it is close to 1 (hence, normalization may destroy the
PAC density estimation guarantee). For specific families, such as mixtures of Gaussians, it may be
more feasible to post-process our estimator to output a legitimate density.
11


<!-- page 15 -->
Open problem 2. Our work focuses on a particular generative process, namely the Ornstein–
Uhlenbeck process over finite-dimensional Euclidean spaces. Are there analogous results in
other domains, for example, in discrete domains, infinite-dimensional spaces, or manifolds,
when the noising process is suitably adapted to the domain?
It is well-established that generative modeling via score matching extends to other (e.g., discrete)
domains and processes. Generalizing our results to such settings could yield new implications for
methods such as estimation via pseudo-likelihood [KLV24].
Open problem 3. Can score estimation lead to algorithms for PAC density estimation of well-
conditioned Gaussian mixture models (in the sense of Chen, Kontonis, and Shah [CKS24])
with minimum weight ≥1/poly(k) that run in time singly exponential in k?
Roughly speaking, the guarantee of [CKS24] is that learning the score of a well-conditioned (non-
spherical) GMM with k components to accuracy ζ can be done with d poly(k/ζ) samples and compute.
Unfortunately, our reduction requires taking ζ ≍ε/
√
d, which trivializes the runtime guarantee, so it
seems that new ideas are needed.
Open problem 4. Can one boost the coverage probability δ in the definition of PAC density
estimation?
A weakness of our reduction is that through the use of Markov’s inequality in our (ε, δ)-PAC density
estimation guarantee, ε scales polynomially with 1/δ, rather than with log(1/δ). This could perhaps
be mitigated by assuming access to a stronger score estimation oracle, or via a generic “boosting”
procedure in analogy to classical learning theory. We believe the latter is unlikely to exist, but it
would be useful to formalize this.
Open problem 5. We now know that both density estimation and score estimation are crypto-
graphically hard for Gaussian mixtures with many components. Is it also cryptographically
hard to learn a sampler?
More broadly, it would be interesting to prove computational hardness results for score estimation
for other natural distributions, utilizing cryptographic tools different from Gaussian pancakes.
1.8
Notation
We focus on continuous distributions over Rd that are absolutely continuous with respect to the
Lebesgue measure. Given a distribution P, for each point x ∈Rd, we abuse notation by using
P(x) to denote its Lebesgue density evaluated at x. We use standard definitions of distances
and divergences between distributions. Namely, for two distributions P and Q over Rd, the total
variation distance between P and Q is dTV(P, Q) := (1/2)
R
|dP −dQ|, the KL divergence of P
with respect to Q is KL(P ∥Q) :=
R
log dP
dQ dP (provided P ≪Q), and the 2-Wasserstein distance
between P and Q is W2(P, Q) = infγ∈C(µ,ν) (
R
∥x −y∥2 γ(dx, dy))1/2, where the infimum is over the
set C(µ, ν) of all couplings of P and Q. We use f ≲g to denote f = O(g), f ≳g to denote f = Ω(g),
and f ≍g to denote f = Θ(g). We also use the notation f = eO(g) to hide polylogarithmic factors,
namely, f = O(g logO(1) g). We also use ∧and ∨to denote min and max respectively.
We say that a random vector X in Rd is σ2-sub-Gaussian if for all vectors v ∈Rd, ⟨v, X⟩is a
σ2 ∥v∥2-sub-Gaussian random variable (see [Wai19, Definition 2.1]). See Appendix C.1 for further
discussion of sub-Gaussianity.
Given a probability measure P over Rd, we let Pt denote the law at time t of the Ornstein–
Uhlenbeck (OU) process started at P, and Qt|0 the transition density of the OU process.
12


<!-- page 16 -->
2
Main tools
In this section, we present the formal statements and proofs of the two tools crucial to our applications
establishing connections between score estimation and different notions of distribution learning.
2.1
Connection between log-likelihood and DDPM score estimation (Lemma 1)
First, we give a proof of Lemma 1 that establishes a precise link between the log-likelihood and a
certain integrated score matching objective.
Recall that given a probability measure P over Rd, we let Pt denote the law of the Ornstein–
Uhlenbeck (OU) process started from P0 = P at time t ∈[0, T]. We further denote by Qt|0 the
transition density of at time t.
Proof of Lemma 1. Let (Bt)t≥0 be standard Brownian motion and let (Xt)t≥0 denote the OU process
started at X0 = x0. By parabolic regularity (or direct computation with the OU semigroup), the
mapping (t, x) 7→Pt(x) is strictly positive and smooth on R>0 × Rd, with Pt →P pointwise as
t ↘0. Therefore, the Fokker–Planck equation implies
∂t log Pt = ∆Pt + div(Pt xt)
Pt
= ∆log Pt + ∥∇log Pt∥2 + d + ⟨∇log Pt, xt⟩.
By Itˆo’s formula,
d log Pt(Xt)
=
 
∂t log Pt(Xt) −⟨∇log Pt(Xt), Xt⟩+ ∆log Pt(Xt)
	
dt +
√
2 ⟨∇log Pt(Xt), dBt⟩
=
 
∥∇log Pt(Xt)∥2 + 2 ∆log Pt(Xt) + d
	
dt +
√
2 ⟨∇log Pt(Xt), dBt⟩.
Integrating over time and taking expectations, for ε > 0,
E
 
log PT (XT ) −log Pε(Xε)
 
= d (T −ε) +
Z T
ε
E
 
∥∇log Pt(Xt)∥2 + 2 ∆log Pt(Xt)
	
dt ,
where we used the fact that {
R t
ε ⟨∇log Ps(Xs), dBs⟩}t∈[ε,T] is a martingale which, in turn, can be
deduced because E[∥∇log Pt(Xt)∥2] = O(1/t2) [cf. OV01]. On the other hand, for any t > 0, we
note that
Z
⟨∇log Pt(xt), ∇log Qt|0(xt | x0)⟩Qt|0(dxt | x0) =
Z
⟨∇log Pt(xt), ∇Qt|0(xt | x0)⟩dxt
= −
Z
∆log Pt(xt) Qt|0(dxt | x0) .
Substituting this in and taking ε ↘0 completes the proof.
2.2
Score estimation implies PAC density estimation
A long line of works shows that, under minimal regularity assumptions, access to a score estimation
oracle is sufficient for learning to sample; see Appendix D for a discussion of this literature. However,
the precise connection between score estimation and density estimation remains elusive. Next, we
prove that access to a score estimation oracle is sufficient for PAC density estimation, in the sense
of Definition 3, under essentially the weakest regularity assumptions as required for generation
obtained by the above line of works.
13


<!-- page 17 -->
2.2.1
Relevant oracles
We begin by introducing the two oracles relevant to our reduction. The first oracle formalizes our
notion of score estimation.
Definition 5 (Score estimation oracle). A score estimation oracle for a density P on Rd is a
primitive that receives as inputs a time t ≥0 and a point xt ∈Rd, and outputs bst(xt) ∈Rd. The
error of the oracle with early stopping τ > 0 and terminal time T is defined to be ε2
∗:=
R T
τ ε2
t dt,
where for each t ∈[τ, T],
Z
∥bst(xt) −∇log Pt(xt)∥2 Pt(dxt) ≤ε2
t .
This definition only requires good score estimation for times bounded away from 0 (i.e., t ≥τ). The
latter is particularly useful since the regularity of the score function typically degrades as t ↘0, so
it becomes more difficult to estimate the score at small times. Indeed, early stopping is a commonly
used device in the literature to circumvent this issue (see, e.g., [CCLLSZ23]). This weakening only
makes the oracle easier to implement.
Our reduction from PAC density estimation to score estimation passes through the following
intermediate oracle.
Definition 6 (Integrated score estimation oracle). An integrated score estimation oracle for a
density P on Rd is a primitive that receives as inputs a point x0 ∈Rd and a terminal time T, and
outputs a (possibly random) value bv(x0) ∈R. The oracle is said to have error ε if
Z
E|bv(x0) −v(x0)| P(dx0) ≤ε ,
where E is over the randomness of bv and
v(x0) :=
Z T
0
Z n
∥∇log Pt(xt)∥2 −2

∇log Pt(xt), ∇log Qt|0(xt | x0)
 o
Qt|0(dxt | x0) dt .
The motivation for this oracle comes from Lemma 1, which implies that for any point x0, the output
bv(x0) of the integrated score estimation oracle is close to the target log-density −log P(x0). Note
that this oracle only requires a bound on the average error across the draw of the initial sample
x0 ∼P. In contrast to the score estimation oracle, the above oracle does not allow early stopping.
Nevertheless, we will show that under a mild regularity assumption (see Section 2.2.2), the score
estimation oracle with early stopping is sufficient to implement the above integrated oracle.
Outline of this section. First, in Section 2.2.2, we prove a result about the sub-Gaussianity of
the score along the OU process. This result is a key technical ingredient in our subsequent reduction
and likely of independent interest. Next, in Section 2.2.3, we show that under mild assumptions
on P, the score estimation oracle can be efficiently transformed to an integrated score estimation
oracle (Theorem 2.1) in polynomial time with polynomially many calls to the score estimation oracle.
Then, in Section 2.2.4, we show how to transform the integrated score estimation oracle to a PAC
density estimation oracle using Lemma 1.
2.2.2
Sub-Gaussianity of the score
The key technical ingredient in the subsequent reduction is the following lemma, which ensures that
the score function remains sub-Gaussian along the OU process. It builds upon diffusion estimates
recently developed in [AC23].
14


<!-- page 18 -->
Lemma 3 (Sub-Gaussianity of the score). Assume that for X0 ∼P0, the score ∇log P(X0) is
√
L-sub-Gaussian. Then, for all t ≥0 and Xt ∼Pt, ∇log Pt(Xt) is √Lt-sub-Gaussian, where
Lt := min
n
L exp(2t),
1
1 −exp(−2t)
o
≤2L .
Proof. The main fact that we use is that by [AC23, Theorem 1.2], the score of P is
√
L-sub-Gaussian
under P if and only if P satisfies the local gradient-entropy (LGE) inequality with constant L:
∥
R
∇f dP∥2
R
f dP
≤2L EntP (f)
for all smooth, compactly supported f : Rd →R .
Actually, [AC23, Theorem 1.2] only states one direction of this implication (namely, LGE implies
sub-Gaussianity of the score), but one can see from the proof that it is an equivalence. In light of
this, our goal is, therefore, to verify that Pt satisfies the LGE inequality with parameter Lt, where
Lt = min
n
L exp(2t),
1
1 −exp(−2t)
o
.
We combine two different bounds. The first bound is effective for small t. Let f be such that
R
f dPt = 1 and let (Qt)t≥0 denote the OU semigroup. Then,
Z
∇f dPt =
Z
Qt∇f dP = exp(t)
Z
∇Qtf dP .
By applying the LGE inequality for P, since
R
Qtf dP =
R
f dPt = 1,

Z
∇f dPt

2
≤2L exp(2t) EntP (Qtf) ≤2L exp(2t) EntPt(f) ,
where the last inequality follows from the entropy decomposition
EntPt(f) = EntP (Qtf) +
Z
EntQt|0(·|x0)(f) P(dx0) .
Next, we consider a bound for large t. Since (a, b) 7→∥a∥2/b is jointly convex,
∥
R
∇f dPt∥2
R
f dPt
≤
Z ∥Qt∇f∥2
Qtf
dP ≤
2
1 −exp(−2t)
Z
EntQt|0(·|x0)(f) P(dx0)
≤
2
1 −exp(−2t) EntPt(f) ,
where we applied the LGE inequality along the OU semigroup (see [AC23, Theorem 1.1]).
Putting the two cases together, we have shown that
∥
R
∇f dPt∥2
R
f dPt
≤2 min
n
L exp(2t),
1
1 −exp(−2t)
o
EntPt(f) ≤4L EntPt(f) .
In order to apply Lemma 3 for our purposes, we must verify that the initial score is sub-Gaussian.
Toward that end, we provide two tools for checking this assumption.
Lemma 4 (Lipschitz score implies sub-Gaussian score). Let P be a probability distribution over Rd
such that the score ∇log P is L-Lipschitz. Then, ∇log P is
√
L-sub-Gaussian under P.
15


<!-- page 19 -->
Proof. This fact was established in [Neg22]; the simple argument is reproduced as [AC23, Remark
5.4]. Alternatively, it follows from [AC23, Corollary 5.3 and Theorem 1.2].
Lemma 5 (Score of a mixture). Let µ be a probability measure over a space X, and let P be a
Markov kernel from X to Rd. Let X ∼µ and conditionally on X, let Y ∼P(X, ·); thus, the marginal
law of Y is the mixture µP. Then, the score of µP can be expressed as
∇log µP(y) = E[∇log P(X, ·) | Y = y] .
Proof. Since
µP(y) =
Z
P(x, y) µ(dx) ,
then
∇log µP(y) =
R
∇y log P(x, y) P(x, y) µ(dx)
R
P(x, y) µ(dx)
= E[∇log P(X, ·) | Y = y] .
Lemma 6 (Sub-Gaussianity of the score of a mixture). In the setting of Lemma 5, suppose that for
each x ∈X, ∇log P(x, ·) is σ2-sub-Gaussian under P(x, ·). Then, ∇log µP is also σ2-sub-Gaussian
under µP.
Proof. For any vector v ∈Rd, Jensen’s inequality implies
E exp ⟨v, ∇log µP(Y )⟩= E exp ⟨v, E[∇log P(X, Y ) | Y ]⟩≤E exp ⟨v, ∇log P(X, Y )⟩
≤exp σ2 ∥v∥2
2
,
where the last inequality follows by first conditioning on X.
For example, we use these facts to verify that Gaussian location mixtures satisfy the assumptions
for our reduction (Lemma 8).
2.2.3
Score estimation implies integrated score estimation
We are now ready to show that there is a polynomial-time reduction from score estimation to
integrated score estimation whenever the distribution P has a sub-Gaussian score.
Assumption 1. There is a constant L ≥1 such that the distribution P over Rd has a score function
∇log P which is
√
L-sub-Gaussian under P.
The main result of this section is the following.
Theorem 2.1 (Score estimation implies integrated score estimation). Let P be a distribution on Rd
that satisfies Assumption 1 with parameter L. There is an algorithm that, given accuracy ε ∈(0, 1),
constant L, terminal time T ≥1, and query access to a score estimation oracle for P with early
stopping parameter τ, implements an (ε, T)-integrated score estimation oracle for P. The algorithm
makes N calls to the score estimation oracle with accuracy ε∗for
N = eO
 (L + T) Td2
ε2
 
and
ε∗= eO
 
ε
p
d (log L + T)
 
,
and the early stopping parameter τ of the score estimation oracle is required to satisfy τ ≲ε2/Ld2.
16


<!-- page 20 -->
Recall that the integrated score oracle is a primitive that aims at estimating the values
v(x0) :=
Z T
0
Z n
∥∇log Pt(xt)∥2 −2

∇log Pt(xt), ∇log Qt|0(xt | x0)
 o
Qt|0(dxt | x0) dt
in expectation over x0 ∼P. We are now ready to define the (randomized) output bv(x0) of the
integrated score estimation oracle on input x0 ∈Rd:
bv(x0) := T −τ
m
X
i∈[m]
 
∥bsti(xi
ti)∥2 −2 ⟨bsti(xi
ti), ∇log Qti|0(xi
ti | x0)⟩
	
.
where for i = 1, . . . , m, the pairs (ti, xi
ti) are i.i.d. and drawn as follows: first, ti ∼Unif([τ, T]), and
then, conditionally on ti, xi
ti ∼Qti|0(· | x0). Here, we set τ ≍ε2/Ld2; even if the score estimation
oracle provides score estimates for smaller times, we do not use them.
Proof of Theorem 2.1. Our goal is to control the error
Z
E|bv(x0) −v(x0)| P(dx0) ,
where E denotes the expectation over the randomness of bv. Throughout the proof, we repeatedly
use the sub-Gaussianity of the score (Lemma 3).
The first step is to bound this error by (I) + (II) + (III), where
(I) := T −τ
m
Z
E
   
X
i∈[m]
 
∥bsti(xi
ti)∥2 −∥∇log Pti(xi
ti)∥2	    P(dx0) ,
(II) := 2 (T −τ)
m
Z
E
   
X
i∈[m]
⟨bsti(xi
ti) −∇log Pti(xi
ti), ∇log Qti|0(xi
ti | x0)⟩
    P(dx0) ,
(III) :=
Z
E|(III0)(x0)| P(dx0) ,
(III0)(x0) := T −τ
m
X
i∈[m]
 
∥∇log Pti(xi
ti)∥2 −2 ⟨∇log Pti(xi
ti), ∇log Qti|0(xi
ti | x0)⟩
	
−v(x0) .
We also define the quantities
L∗:=
Z T
τ
Lt dt ,
L∗,2 :=
 Z T
τ
L2
t dt
 1/2
,
L∗,3 :=
Z T
τ
Lt
1 −e−2t dt ,
where Lt is the constant from Lemma 3.
Control of term I.
We start by controlling term I. The error of term I relies on how well the
squared norm of the score oracle approximates the squared norm of the actual score.
Claim 1 (Controlling term (I)). Let ε2
∗=
R T
τ ε2
t dt. It holds that
T −τ
m
Z
E
   
X
i∈[m]
 
∥bsti(xi
ti)∥2 −∥∇log Pti(xi
ti)∥2	    P(dx0) ≲ε2
∗+
p
L∗d ε∗.
17


<!-- page 21 -->
Proof of Claim 1. We can bound term I by
(I) ≤T −τ
m
Z
E
X
i∈[m]
∥bsti(xi
ti) −∇log Pti(xi
ti)∥
  ∥bsti(xi
ti)∥+ ∥∇log Pti(xi
ti)∥
   P(dx0)
≤T −τ
m
Z
E
X
i∈[m]
∥bsti(xi
ti) −∇log Pti(xi
ti)∥2 P(dx0)
+ 2 (T −τ)
m
Z
E
X
i∈[m]
∥bsti(xi
ti) −∇log Pti(xi
ti)∥∥∇log Pti(xi
ti)∥P(dx0)
≤T −τ
m
E
X
i∈[m]
ε2
ti + 2 (T −τ)
m
E
X
i∈[m]
εti
 Z
∥∇log Pti(xi
ti)∥2 Pti(dxi
ti)
 1/2
≲ε2
∗+ (T −τ)
√
d
m
E
X
i∈[m]
εti
p
Lti ≲ε2
∗+
p
L∗d ε∗.
The second inequality follows by observing that ∥a−b∥·|∥a∥−∥b∥+ 2∥b∥| ≤∥a−b∥2 + 2 ∥a−b∥∥b∥
for any vectors a, b, the third inequality follows by Cauchy–Schwarz and the property of the score
estimation oracle at times {ti}i∈[m], and the fourth inequality follows by sub-Gaussianity of the
score and the definition of the integrated error ε∗.
Control of term II. Similarly, we can control term II, which involves again a difference between
the score oracle and the actual score function. In contrast to term I which contained the difference
of the norms, Term II (roughly speaking) involves the difference of the two vectors in the direction
of the associated OU process.
Claim 2 (Controlling term (II)). Let ε2
∗=
R T
τ ε2
t dt. It holds that
2 (T −τ)
m
Z
E
   
X
i∈[m]
⟨bsti(xi
ti) −∇log Pti(xi
ti), ∇log Qti|0(xi
ti | x0)⟩
    P(dx0) ≲ε∗
r
d
 T + log 1
τ
 
.
Proof of Claim 2. We have the following for Term II:
(II) ≤2 (T −τ)
m
X
i∈[m]
Z
E
 
∥bsti(xi
ti) −∇log Pti(xi
ti)∥∥∇log Qti|0(xi
ti | x0)∥
 
P(dx0)
≲(T −τ)
 Z
E
 
∥bst1(x1
t1) −∇log Pt1(x1
t1)∥2 
P(dx0)
Z
E
 
∥∇log Qt1|0(x1
t1 | x0)∥2 
P(dx0)
 1/2
≤(T −τ)
 
E[ε2
t1] E
 
d
1 −exp(−2t1)
  1/2
=
 Z T
τ
ε2
t dt
Z T
τ
d
1 −exp(−2t) dt
 1/2
≲ε∗
r
d
 T + log 1
τ
 
.
The second inequality follows by Cauchy–Schwarz and the fact that we use i.i.d. samples and the
third inequality uses the property of the score estimation oracle at time t1 and the sub-Gaussianity
of the OU process.
18


<!-- page 22 -->
Control of term III. The last step is to control term (III). Recall the quantities
L∗,2 :=
 Z T
τ
L2
t dt
 1/2
,
L∗,3 :=
Z T
τ
Lt
1 −e−2t dt ,
where Lt is the constant from Lemma 3.
Claim 3 (Controlling term (III)). It holds that
Z
E
   T −τ
m
X
i∈[m]
 
∥∇log Pti(xi
ti)∥2 −2 ⟨∇log Pti(xi
ti), ∇log Qti|0(xi
ti | x0)⟩
	
−v(x0)
    P(dx0)
≲Ldτ +
√
Ld√τ
|
{z
}
early stopping
+
L∗,2
√
Td + d
p
L∗,3T
√m
|
{z
}
variance
.
For term (III), we perform a bias-variance decomposition. First, the estimator in the above expression
is biased since it only integrates from time τ to T, while v(x0) integrates from time 0. Hence, the
bias term corresponds to the error due to early stopping. On the other side, we also have to deal
with the variance of the estimator.
Proof of Claim 3. We bound term (III) by (IV) + (V), where
(IV) =
Z
|E(III0)(x0)| P(dx0)
=
Z    
Z τ
0
Z  
∥∇log Pt(xt)∥2 −2 ⟨∇log Pt(xt), ∇log Qt|0(xt | x0)⟩
	
Qt|0(dxt | x0) dt
    P(dx0) ,
(V) ≤T −τ
√m
Z q
Var
 ∥∇log Pt1(x1
t1)∥2 −2 ⟨∇log Pt1(x1
t1), ∇log Qt|0(x1
t1 | x0)⟩
 
P(dx0) .
Note that term (IV) is the early stopping error, which we control as follows:
(IV) ≤
Z τ
0
ZZ  
∥∇log Pt(xt)∥2 + 2 ∥∇log Pt(xt)∥∥∇log Qt|0(xt | x0)∥
	
Qt|0(dxt | x0) P(dx0) dt
≲
Z τ
0
 
Ld +
√
Ld
p
1 −exp(−2t)
	
dt ≲Ldτ +
√
Ld√τ .
Note that the second inequality follows immediately by noting that ∥∇log Pt∥2 is
√
2Ld-sub-Gaussian
and that the OU process ∥∇log Qt|0∥contributes another
√
d/
√
1 −e−2t factor.
Finally, for the variance term (V), by the triangle inequality, we bound it by (VI) + (VII), where
(VI) = T −τ
√m
Z q
Var
 ∥∇log Pt1(x1
t1)∥2 
P(dx0) ,
(VII) = 2 (T −τ)
√m
Z q
Var
 ⟨∇log Pt1(x1
t1), ∇log Qt1|0(x1
t1 | x0)⟩
 
P(dx0) .
By sub-Gaussianity of the score,
(VI) ≤T −τ
√m
sZ
E[∥∇log Pt1(x1
t1)∥4] P(dx0) ≲(T −τ) d
√m
q
E[L2
t1] ≲L∗,2
√
Td
√m
.
19


<!-- page 23 -->
For the last term,
(VII) ≤2 (T −τ)
√m
Z q
E
 
∥∇log Pt1(x1
t1)∥2 ∥∇log Qt|0(x1
t1 | x0)∥2 
P(dx0)
≤2 (T −τ)
√m
sZ
E
 
∥∇log Pt1(x1
t1)∥2 ∥∇log Qt|0(x1
t1 | x0)∥2 
P(dx0)
≤2 (T −τ)
√m
 Z
E
hsZ
∥∇log Pt1(xt1)∥4 Qt1|0(dxt1 | x0)
×
sZ
∥∇log Qt1|0(xt1 | x0)∥4 Qt1|0(dxt1 | x0)
i
P(dx0)
 1/2
≲T −τ
√m
 
E
h
d
1 −exp(−2t1)
Z sZ
∥∇log Pt1(xt1)∥4 Qt1|0(dxt1 | x0) P(dx0)
i 1/2
≤T −τ
√m
 
E
h
d
1 −exp(−2t1)
sZZ
∥∇log Pt1(xt1)∥4 Qt1|0(dxt1 | x0) P(dx0)
i 1/2
≲(T −τ) d
√m
 
E
h
Lt1
1 −exp(−2t1)
i 1/2
≲d
p
L∗,3T
√m
.
In the above, the second and the fifth inequality use that E
√
X ≤
√
EX. The third inequality
follows by Cauchy–Schwarz, the fourth by properties of the OU process, and the sixth one by
sub-Gaussianity of the score.
Putting everything together. We now apply the integral estimates from Lemma 12, combine
the bounds of Claims 1 to 3, and simplify to obtain
Z
E|bv(x0) −v(x0)| P(dx0) ≲Ldτ +
√
Ld√τ
|
{z
}
early stopping
+ ε2
∗+ ε∗
r
d
 T + log 1
τ
 
|
{z
}
score error
+ d
p
T (T + L log 1/Lτ)
√m
|
{z
}
sampling error
.
To make the overall error at most ε, it suffices to take
τ ≍ε2
Ld2 ,
ε∗≲
ε
p
d (T + log(Ld2/ε2))
,
and
m ≳d2 (T 2 + LT log(d2/ε2))
ε2
.
2.2.4
Integrated score estimation implies PAC density estimation
In this section, we reduce the problem of PAC density estimation to that of implementing the
integrated score estimation oracle introduced in Definition 6.
We need the following mild assumption on the density P.
Assumption 2 (Second moment bound). There exists M2 > 0 such that the density P has second
moment at most M2, i.e.,
R
∥x∥2 P(dx) ≤M2.
In this section, we prove the following.
20


<!-- page 24 -->
Theorem 2.2 (Integrated score estimation implies PAC density estimation). Consider a density P
satisfying Assumption 2 with parameter M2. For any ε ∈(0, 1), if there is an efficient ε-integrated
score estimation oracle for P with terminal time T ≥1
2 log(1 + 2M2/ε), then, there exists an efficient
algorithm outputting a function bP : Rd →R+ (as an evaluation oracle) such that
Z
E
   log
bP(x0)
P(x0)
    P(dx0) ≤2ε .
We remark that while the output bP is non-negative, it does not necessarily integrate to 1 and is
therefore not a valid probability density. Thus, despite appearances, Theorem 2.2 does not provide
a KL divergence guarantee.
Proof of Theorem 2.2. By the existence of the integrated score estimation oracle, we know that
there is an efficient algorithm that outputs a function bv: Rd →R such that
Z
E|bv(x0) −v(x0)| P(dx0) ≤ε ,
where v(·) is given in Definition 6. Since Assumption 2 holds, P has finite second moment, due to
which Lemmas 1 and 7 and Equation (5) are applicable. First, from Lemma 1, we know that for all
x0 ∈Rd:
Z
log PT dQT|0(· | x0) −log P(x0)
=
Z T
0
Z  
∥∇log Pt∥2 −2 ⟨∇log Pt, ∇log Qt|0(· | x0)⟩
	
dQt|0(· | x0) dt
|
{z
}
=v(x0)
+dT .
This means that we can use the integrated score oracle as an estimator for the negative log-likelihood
−log P by defining the function ℓ: Rd →R with
ℓ(x0) := bv(x0) + d
 T + 1
2 log(2πe (1 −exp(−2T)))
 
.
Next, from Equation (5), we know that for any x0,
|−log P(x0) −ℓ(x0)| = KL(QT|0(· | x0) ∥PT ) .
It remains to show that for T sufficiently large the above error is negligible. Toward this, we use
the following bound from Lemma 7:
KL(QT|0(· | x0) ∥PT ) ≤
1
exp(2T) −1
 
∥x0∥2 +
Z
∥x∥2 P(dx)
 
.
It follows from Assumption 2 that
Z
|−log P(x0) −ℓ(x0)| P(dx0) ≤
2M2
exp(2T) −1 .
This is made at most ε if T ≥1
2 log(1 + 2M2/ε). So, if we let bP := exp(−ℓ), we have shown that
Z
E
  log
bP(x0)
P(x0)
   P(dx0) ≤2ε .
21


<!-- page 25 -->
Remark 1. To explain why the theorem implies PAC density estimation, suppose that the score
estimation oracle is implemented on the basis of samples and yields score estimates satisfying
E
R T
τ ∥st −∇log Pt∥2
L2(Pt) dt ≤ε2
∗. By Markov’s inequality, with probability at least 9/10 over
the samples, it holds that
R T
τ ∥st −∇log Pt∥2
L2(Pt) dt ≤10ε2
∗. Conditioned on this event, we can
apply Theorem 2.2 and Markov’s inequality to deduce that
EP{x ∈Rd : bP(x) /∈[e−2ε/δ P(x), e2ε/δ P(x)]} ≤δ ,
i.e., it yields a (2ε/δ, δ)-PAC density estimator.
2.2.5
Completing the reduction
Combining Theorems 2.1 and 2.2, we obtain the following result.
Theorem 2.3 (Reduction from score estimation to PAC density estimation). Let P be a distribution
on Rd that satisfies Assumption 1 with parameter L and Assumption 2 with parameter M2 ≥1.
There is an efficient algorithm that, given access to a score estimation oracle for P, outputs a
function bP : Rd →R+ (as an evaluation oracle) such that
Z
E
  log
bP(x0)
P(x0)
   P(dx0) ≤2ε .
The algorithm makes N calls to the score estimation oracle with accuracy ε∗for
N = eO
 Ld2 log2(M2)
ε2
 
and
ε∗= eO
 
ε
p
d log(LM2)
 
,
and the early stopping parameter τ of the score estimation oracle is required to satisfy τ ≲ε2/Ld2.
Moreover, the algorithm takes poly(N) time.
Proof. We set T ≍log M2/ε2.
A more precise choice of parameters, obtained from the proof
of Theorem 2.1, is given by
ε∗≲
ε
p
d (log M2/ε2 + log Ld2/ε2)
and
N ≳d2 (log2(M2/ε2) + L log(M2/ε2) log(d2/ε2))
ε2
.
Remark 2. It is interesting to compare this reduction with the one for generation. If we ignore
algorithmic considerations and solely focus on how the score estimation error ε∗translates into the
error for distribution learning, then existing works on sampling from diffusion models (or simply
Girsanov’s theorem) imply the following statement. If bPgen denotes the law of the output of the
diffusion model with estimated scores, then
2 dTV( bPgen, P)2 ≤KL(P ∥bPgen) ≲ε2
∗.
Thus, ε∗score estimation error leads to ε∗error in total variation. On the other hand, our reduction
shows that for density estimation,
Z
E
  log
bP
P
   dP ≲ε2
∗+ eO(ε∗
√
d) .
Since this performance metric greatly resembles a KL divergence, it is natural to wonder if the
extra term eO(ε∗
√
d) is superfluous. Perhaps surprisingly, the answer is no: the right-hand side must
contain a term scaling linearly with ε∗, or else it would violate minimax lower bounds for density
estimation; see Section 4.
22


<!-- page 26 -->
2.3
Early stopping
The reduction in Section 2.2 requires the assumption that the initial distribution P has a sub-
Gaussian score. In this section, we show that even if this assumption is removed, we can still output
an estimate of the density Pτ with early stopping. This is used for our result on density estimation
over H¨older classes in Section 4.
Theorem 2.4 (PAC density estimation with early stopping). Let P be a distribution on Rd that
satisfies Assumption 2 with parameter M2 ≥1. Let 0 < τ, ε < 1. Then, given access to a score
estimation oracle with early stopping τ, there is an algorithm that outputs a function bPτ : Rd →R+
(as an evaluation oracle) such that
Z
E
  log
bPτ(x0)
Pτ(x0)
   Pτ(dx0) ≤2ε .
The algorithm makes N calls to the score estimation oracle with accuracy ε∗for
N ≍d2 (log2(M2/ε2) + τ −1 log(M2/ε2) log(d2/ε2))
ε2
and
ε∗≍
ε
p
d (log M2/ε2 + log d2/τε2)
,
Moreover, the algorithm takes poly(N) time.
Proof. We apply the reduction in Section 2.2, after replacing P with Pτ. By Lemma 3, Pτ has a
sub-Gaussian score with parameter L = 1/(1 −e−2τ) = O(1/τ).
2.4
Application to estimating the differential entropy
In this paper, we focus on applications of our reduction from PAC density estimation. However, here
we briefly mention that our guarantee immediately implies that a score estimation oracle can be
used to estimate the differential entropy of the distribution P, which is also a well-studied problem
(see [HJWW20] and references therein).
To see how to estimate differential entropy using our tools, assume that we have access to
bP satisfying the guarantee of Theorem 2.2. The estimator is simply defined by drawing n i.i.d.
samples X(1)
0 , . . . , X(n)
0
from P and outputting the median of the values {−log bP(X(i)
0 )}i∈[n]. By
the guarantee for bP,
E
    1
n
n
X
i=1
log bP(X(i)
0 ) −
Z
log P dP
    = E
    1
n
n
X
i=1
log
bP(X(i)
0 )
P(X(i)
0 )
+
n
X
i=1
log P(X(i)
0 ) −
Z
log P dP
   
≤2ε +
r
VarP log P
n
.
3
DDPM is an asymptotically efficient parameter estimator
In this section, we study the use of DDPM score estimation for estimating parameters over a
parametric family P ∈P = {Pθ : θ ∈Θ}. We consider the following idealized DDPM estimator,
which is equivalent to selecting the parameter that minimizes the DDPM risk in Lemma 1 over
samples x0 from P.
23


<!-- page 27 -->
Definition 1. Fix a terminal time T > 0. Given samples X(1)
0 , . . . , X(n)
0
and a family P = {Pθ :
θ ∈Θ ⊆Rp}, the DDPM estimator is bθ DDPM
n
:= arg minθ∈Θ bRDDPM
n
(θ), where
bRDDPM
n
(θ) := 1
n
n
X
i=1
Z T
0
E
h
∥∇log Pθ,t(X(i)
t )∥2 +

∇log Pθ,t(X(i)
t ),
2Z(i)
t
√
1 −e−2t
      X(i)
0
i
dt
and for each i ∈[n] and t ∈[0, T], we draw Z(i)
t
∼N(0, Id) independently from X(i)
0
and define the
noised sample X(i)
t
:= e−t X(i)
0
+
√
1 −e−2t Z(i)
t .
The main result of this section is that, under mild regularity assumptions on the distribution family P
(essentially the same conditions needed for the asymptotic normality of the MLE, see Assumption 3)
and by choosing the terminal time T = Tn to grow sufficiently rapidly with the number of samples
n (namely, Tn −1
2 log n →∞), the DDPM estimator bθ DDPM
n
converges in distribution to a Gaussian
centered at θ⋆with covariance exactly equal to the inverse Fisher information.
3.1
Implications of the likelihood identity for parameter estimation
We begin by noting the following immediate consequence of
Lemma 1: When specialized to
a parametric family P ∈P = {Pθ : θ ∈Θ}, set x0 = X(i)
0 , and sum over i ∈[n], where
X(1)
0 , . . . , X(n)
0
i.i.d.
∼
Pθ⋆, Lemma 1 implies that the empirical risk for the maximum likelihood
estimator (MLE) coincides with the empirical score matching loss, up to a known constant and a
vanishing error. More precisely:
Proposition 1 (Tight connection between DDPM and MLE). The DDPM objective bRDDPM
n
and
the maximum likelihood objective bRMLE
n
satisfy:
bRMLE
n
(θ) = bRDDPM
n
(θ) + Cd,T + 1
n
n
X
i=1
KL
 QT|0(· ∥X(i)
0 )
 Pθ,T
 
where bRMLE
n
(θ) := −1
n
Pn
i=1 log Pθ(X(i)
0 ), bRDDPM
n
is given in Definition 1, and Cd,T = d (T +
1
2 log(2πe (1 −e−2T ))) is a fixed constant.
Since we show later in this section (Lemma 7) that the final term above decays as exp(−2T), it
is intuitive from Proposition 1 that the DDPM estimator inherits the favorable properties of the
MLE, including its statistical efficiency. We make this precise in Section 3.2 under the assumption
that the family {Pθ : θ ∈Θ} is differentiable in quadratic mean, which is essentially the weakest
regularity condition under which the Fisher information is well-defined.
3.2
The proof of Informal Theorem 1
We now proceed with the proof of Informal Theorem 1 regarding the asymptotic efficiency of DDPM
score matching. Let Qt|0(· | x0) denote the transition density of the OU process run until time t
started at time 0 at x0.
Step 1 (Likelihood identity). As a first step, Lemma 1 implies that for any θ ∈Θ,
Z
log Pθ,T dQT|0(· | x0) −log Pθ(x0)
=
Z T
0
Z  
∥∇log Pθ,t∥2 −2 ⟨∇log Pθ,t, ∇log Qt|0(· | x0)⟩
	
dQt|0(· | x0) dt + dT .
24


<!-- page 28 -->
Step 2 (Relating MLE and DDPM). Therefore, if we consider the one-sample empirical risks
(where Equation (4) is proved in Appendix A),
bRMLE(θ) = −log Pθ(x0) ,
bRDDPM(θ) =
Z T
0
Z
{∥∇log Pθ,t∥2 −2 ⟨∇log Pθ,t, ∇log Qt|0(· | x0)⟩} dQt|0(· | x0) dt ,
(4)
we can rewrite the identity above as
bRMLE(θ) = bRDDPM(θ) + dT −
Z
log Pθ,T dQT|0(· | x0)
= bRDDPM(θ) + d
 T + 1
2 log(2πe (1 −exp(−2T)))
 
+ KL(QT|0(· | x0) ∥Pθ,T ) ,
(5)
where the last line follows by adding and subtracting
R
log QT|0(· | x0) dQT|0(· | x0) and using the
formula for the differential entropy of a Gaussian. This proves Proposition 1.
Step 3 (Exponential decay of KL). We observe that the last term in (5) is controlled by the
following lemma.
Lemma 7. For any probability measure P with finite second moment and any x0 ∈Rd,
KL(QT|0(· | x0) ∥PT ) ≤
1
exp(2T) −1
 
∥x0∥2 +
Z
∥x∥2 P(dx)
 
.
Proof of Lemma 7. By the dimension-free log-Harnack inequality [see, e.g., BGL01; Wan06; AC24],
KL(QT|0(· | x0), PT ) ≤W 2
2 (δx0, P)/{2 (exp(2T)−1)}. The result follows from the triangle inequality
for W2.
Statement and proof of Informal Theorem 1. Given steps I–III, we are now ready to prove
Informal Theorem 1. To state our asymptotic normality result for the DDPM score matching
estimator, we build on the following standard conditions for asymptotic normality of the MLE.
Note that it is implicitly assumed that the MLE exists for sufficiently large n.7
Assumption 3 (Conditions for asymptotic normality of MLE [Vaa98]). The family {Pθ}θ∈Θ is
differentiable in quadratic mean (DQM) at an interior point θ⋆∈Θ ⊆Rp. Furthermore, there
exists a function L such that for all θ, θ′ in a neighborhood of Θ, |log Pθ −log Pθ′| ≤L ∥θ −θ′∥with
R
L2 dPθ⋆< ∞. The Fisher information matrix Iθ⋆is positive definite. Finally, the MLE bθ MLE
n
is
consistent: bθ MLE
n
→θ⋆in probability as n →∞.
Here, the DQM condition weakens the classical assumptions for asymptotic normality of the
MLE, which require the existence of a third derivative of θ 7→log Pθ, and instead asks for the
existence of a derivative of θ 7→√Pθ at θ⋆in L2(Pθ⋆). This covers non-differentiable examples such
as the two-sided exponential location family. Under Assumption 3, it is shown in [Vaa98, Theorem
5.39] that √n (bθ MLE
n
−θ⋆) d−→N(0, I(θ⋆)−1). We prove the following result.
Theorem 3.1 (Asymptotic normality of the DDPM estimator). Adopt Assumption 3. Consider the
DDPM estimator bθ DDPM
n
where the time Tn of the diffusion satisfies Tn −1
2 log n →∞. Assume also
that for some neighborhood Θ′ of θ⋆, it holds that supθ∈Θ′
R
∥x∥2 Pθ(dx) < ∞, and that the DDPM
estimator is consistent. Then, the DDPM estimator is asymptotically efficient: √n (bθ DDPM
n
−θ⋆) d−→
N(0, I(θ⋆)−1).
7This assumption could also be relaxed.
25


<!-- page 29 -->
Proof of Theorem 3.1. We modify the proof of [Vaa98, Theorem 5.39], which relies on Theorem
5.23 therein. For θ ∈Θ, let mθ := log pθ and Pn := (1/n) Pn
i=1 δXi. In order to invoke Theorem 5.23,
it suffices to show that Pnmbθ DDPM
n
≥Pnmbθ MLE
n
−oPθ⋆(n−1). By (5),
−Pnmbθ DDPM
n
= bRDDPM
n
(bθ DDPM
n
) + cd,T + Pnerr(bθ DDPM
n
)
≤bRDDPM
n
(bθ MLE
n
) + cd,T + Pnerr(bθ DDPM
n
)
= −Pnmbθ MLE
n
+ Pn[err(bθ DDPM
n
) −err(bθ MLE
n
)] ,
where cd,T is a constant and err(θ, x) := KL(QT|0(· | x) ∥Pθ,T ). Since err is non-negative, it yields
Pnmbθ DDPM
n
≥Pnmbθ MLE
n
−Pnerr(bθ DDPM
n
). Since the DDPM estimator is consistent, Lemma 7 and
our assumptions imply Pθ⋆err(bθ DDPM
n
) ≤2 (exp(2T) −1)−1 supθ′∈Θ
R
∥x∥2 Pθ(dx) = o(1/n). By
Markov’s inequality, Pnerr(bθ DDPM
n
) = oPθ⋆(1/n). The rest of the proof is unchanged.
The assumption of consistency for the MLE and the DDPM estimators is typically mild and can be
handled by standard tools, e.g., [Vaa98, §5.2].
4
Minimax optimal density estimation over the H¨older class
Recently, many works have studied the statistical rates of estimation over non-parametric classes
of densities, both for the score function (along the OU process) and its implications for learning a
sampler, together with matching minimax lower bounds. For example, [BMR20] obtained rates for
estimating score functions based on Rademacher complexity, and [WWY24] established the minimax
rate for estimating a Lipschitz score for a sub-Gaussian density. The works [OAS23; DKXZ24;
ZYLL24] showed that DDPM score estimation can lead to minimax optimal rates for distribution
learning. However, we emphasize that these prior works only showed that one can learn a sampler
using the existing reduction (Section 1.6), whereas our goal is to show that DDPM score estimation
leads to density estimators.
In this subsection, we start with a representative result on score estimation from the literature,
namely the result of Dou, Kotekal, Xu, and Zhou [DKXZ24]. Their work considered the following
H¨older class of densities.
Definition 7 (H¨older class). For C > 2 and s, L > 0, let Hs(C, L) denote the class of probability
densities P supported on [−1, 1] with the following properties:
• P is continuous on [−1, 1], admits ⌊s⌋derivatives on (−1, 1), and
|D⌊s⌋P(x) −D⌊s⌋P(y)| ≤L |x −y|s−⌊s⌋,
for all x, y ∈(−1, 1) .
• On the domain [−1, 1], the density P is bounded away from 0 and ∞, i.e., C−1 ≤P ≤C.
We note that the restriction to one dimension is purely for ease of exposition (as is common in the
literature). All of the ideas below can be adapted to the higher-dimensional case by replacing the
rate n−s/(2s+1) with n−s/(2s+d).
For P ∈Hs(C, L), [DKXZ24] proved that the score ∇log Pt can be estimated in L2(Pt) at a
certain rate (see Remark 3 below). Our goal is to convert this into a result for density estimation.
Although this is ultimately a consequence of our framework in Section 2, the main effort here is to
provide a common setting in which we can apply the reductions in Section 2, the score estimation
rates of [DKXZ24], and the lower bounds in the density estimation literature so that the final result
26


<!-- page 30 -->
is minimax optimal. This is not entirely trivial. For example, since the densities in Definition 7
are compactly supported, they do not have globally Lipschitz scores, and for s < 2 they do not
even have Lipschitz scores in the interior (−1, 1). Hence, to apply our reductions, we utilize early
stopping. Moreover, since our approach leads to PAC density estimation, which is a relatively weak
solution concept, it is not immediately clear that it is compatible with existing notions of risk in the
literature on density estimation.
Our notion of risk is defined as follows. Given an estimator bP using n samples and a probability
density P, we define the L1 risk
Rn( bP, P) :=
Z
[−1,1]
EP | bP(x0) −P(x0)| dx0 .
Note that if bP were a probability density on [−1, 1], this would correspond to twice the total
variation distance. Here, we use the subscript on EP to indicate that the estimator is based on n
i.i.d. samples from P. Henceforth, all of the asymptotic notation (e.g., ≲, O(·), . . . ) suppresses
constants which do not depend on n.
Our main result of this section is stated below.
Theorem 4.1 (Density estimation for H¨older classes). Let C > 2, s, L > 0.
1. The following minimax lower bound holds:
inf
bP
sup
P∈Hs(C,L)
Rn( bP, P) ≳n−s/(2s+1) .
2. There is an estimator bP based on DDPM score estimation such that
sup
P∈Hs(C,L)
Rn( bP, P) ≲n−s/(2s+1)p
log n .
Before proceeding to the proof, we need the following remark.
Remark 3. The paper [DKXZ24] actually considers estimation of the score function along the heat
flow, rather than the OU process: ePt := P ∗N(0, t Id). From Tweedie’s identity [Rob56],
−(1 −e−2t) ∇log Pt(xt) = xt −E[X0 | e−t X0 +
p
1 −e−2t Z = xt] ,
−t ∇log ePt(ext) = xt −E[X0 | X0 +
√
t Z = ext] ,
one can relate the two score functions:
∇log Pt(xt) =
et −1
1 −e−2t xt + e2t ∇log ePe2t−1(etxt) .
Hence, if est is an estimator for ∇log ePt and we set st(xt) :=
et−1
1−e−2t xt + e2t est(etxt), then
∥st −∇log Pt∥L2(Pt) ≤e2t ∥est −∇log ePe2t−1∥L2( ePt) .
Applying this to the score estimator of [DKXZ24] for the class Hs(C, L), we obtain
∥st −∇log Pt∥2
L2(Pt) ≲1
n ∧
1
nt3/2 ∧(n−2(s−1)/(2s+1) + ts−1) .
27


<!-- page 31 -->
Proof of Theorem 4.1. We divide the proof into two parts corresponding to the lower bound and
the upper bound.
Lower bound.
The lower bound is classical, see, e.g., Yang and Barron [YB99] (which also
considers the more general Besov classes, among others).
Upper bound. For the upper bound, we apply Theorem 2.4 with the score estimator in Remark 3.
Let ϕσ2 denote the density of the Gaussian N(0, σ2). For x0 ∈[−1, 1] and τ ≲1, since P is H¨older
continuous on [−1, 1] with exponent s ∧1 := min{s, 1} (see Lemma 13) and the density P is upper
bounded by C,
|Pτ(x0) −P(x0)|
=
   
Z
eτP(eτx) ϕ1−e−2τ (x0 −x) dx −P(x0)
   
≤C (eτ −1) +
Z
|P(eτx) −P(x0)| ϕ1−e−2τ (x0 −x) dx
≤C (eτ −1) +
Z
|x|≤e−τ |P(eτx) −P(x0)| ϕ1−e−2τ (x0 −x) dx + C
Z
|x|≥e−τ ϕ1−e−2τ (x0 −x) dx
≲τ +
Z
|eτx −x0|s∧1 ϕ1−e−2τ (x0 −x) dx + C
Z
|x|≥e−τ ϕ1−e−2τ (x0 −x) dx
≲τ s∧1 +
Z
|x −x0|s∧1 ϕ1−e−2τ (x0 −x) dx + C
Z
|x|≥e−τ ϕ1−e−2τ (x0 −x) dx
≲τ (s∧1)/2 + C
Z
|x|≥e−τ ϕ1−e−2τ (x0 −x) dx .
In the above, the first inequality follows by adding and subtracting
R
P(eτx) ϕ1−e−2τ (x0 −x) dx and
the triangle inequality, and the third inequality by H¨older continuity. The last inequality follows by
standard tail bounds on Gaussian random variables with variance 1 −e−2τ (Theorem C.1). The
last term is bounded by the probability that a centered Gaussian with variance 1 −e−2τ exceeds
1 −e−τ −|x0|. By standard Gaussian tail estimates,
Z
|x|≥e−τ ϕ1−e−2τ (x0 −x) dx ≲
(
τ ,
|x0| ≤1 −Ω(
p
τ log 1/τ) ,
1 ,
otherwise .
This leads to the integrated estimate
Z
[−1,1]
|Pτ(x0) −P(x0)| dx0 ≲τ (s∧1)/2 + τ +
p
τ log 1/τ ≲τ (s∧1)/2p
log 1/τ .
We choose τ so that this quantity is bounded by n−s/(2s+1), so that log 1/τ ≲log n.
Our estimator is a clipped version of the early stopped PAC density estimator, i.e., we set
bP := max{1/C′, min{ bPτ, C′}} .
Here, C′ > 0 is a constant not depending on n, bPτ is the output of early stopping (see Theorem 2.4),
and bP is not to be confused with the PAC density estimator without early stopping. We choose C′
so that 1/C′ ≤Pτ ≤C′ on [−1, 1]; such a constant exists by [DKXZ24, Lemma 11].
28


<!-- page 32 -->
From Theorem 2.4, since ε∗≍r∗
n := n−s/(2s+1), we obtain ε ≍ε∗
p
log(1/ε∗), i.e., ε ≍r∗
n
√log n.
Hence, we have the guarantee
Z
EP
  log
bPτ(x0)
Pτ(x0)
   Pτ(dx0) ≲r∗
n
p
log n .
Recall that EP corresponds to the expectation over the n i.i.d. samples used for the estimator bPτ.
Since Pτ ≳1 on [−1, 1] (see [DKXZ24, Lemma 11]), it implies
Z
[−1,1]
EP
  log
bPτ(x0)
Pτ(x0)
   dx0 ≲r∗
n
p
log n .
Since 1/C′ ≤Pτ ≤C′ on [−1, 1] and bP is bPτ clipped to [1/C′, C′], it follows that
Z
[−1,1]
EP
  log
bP(x0)
Pτ(x0)
   dx0 ≲r∗
n
p
log n .
Now, since bP/Pτ is bounded away from 0 and ∞, Taylor expansion of the logarithm shows that
Z
[−1,1]
EP
   bP(x0)
Pτ(x0) −1
   dx0 ≲r∗
n
p
log n .
Finally, since Pτ is lower bounded on [−1, 1], it implies
Z
[−1,1]
EP | bP(x0) −Pτ(x0)| dx0 ≲r∗
n
p
log n .
Combining this with the upper bound on
R
[−1,1] |Pτ(x0) −P(x0)| dx0 finishes the proof.
Some more remarks are in order.
Remark 4. Regarding the computational cost of our estimator, once the score estimates have been
computed, the number of evaluations of the estimated scores is eO(n4s/(2s+1)) ≤eO(n2).
Remark 5. We believe that the same strategy yields density estimators for other settings, such as
the one considered in [YP25]. For brevity, we do not pursue such results here.
5
PAC density estimation for Gaussian location mixtures
In this section, we study density estimation for the classical family of Gaussian location mixtures.
This family is parameterized by a distribution Q (of the means): given Q, the corresponding
Gaussian Location Mixture (GLM) is
M = Q ∗N(0, σ2 Id) .
(Gaussian location mixture)
This is a (possibly) continuous mixture of spherical Gaussians where Q is the distribution of means.
In the special case where Q is discrete, say Q = Pk
i=1 wiδµi, then the above is exactly a mixture of
k spherical-covariance Gaussians with means µ1, . . . , µk. However, in general, Q can be continuous,
and then the above spherical GLM family is non-parametric.
This family can also be seen as the smoothening of an underlying family of distributions.
Smoothness is a very natural property of real-world distributions (which are subject to independent
29


<!-- page 33 -->
errors) and a huge body of work in theoretical computer science studies algorithms in the presence
of smoothed data; a partial list is [ST04; HRS20; HHSY22; BRS24; CKKMS24; HRS24] and we
refer the reader to [Rou21, Chapter 13] and [BV04] for an overview of these works.
Apart from computer science, this family has also appeared in the statistics literature at least
as early as the work of Kiefer and Wolfowitz [KW56], where it is called the Gaussian location
mixture [SG20; KG22]. These works consider arbitrary mixing measures Q and focus on the sample
complexity (without computational considerations). Saha and Guntuboyina [SG20] studied the
finite sample complexity bounds for non-parametric maximum likelihood estimation in squared
Hellinger distance, while Kim and Guntuboyina [KG22] gave a minimax bound for estimation in
squared Hellinger distance using kernel density estimation.
Recently, Gatmiry, Kelner, and Lee [GKL24], in a surprising result developed a quasi-polynomial
time generator for a subset of this family satisfying the following locality assumption which restricts
the choice of Q; but still allows it to be continuous and non-parametric.
Definition 8 ((k, R, D, wmin)-locality [GKL24]). Given parameters R ≥1, D > 0, k ∈N, and
wmin ∈(0, 1/k), the GLM with distribution Q and variance σ2 is said to be (k, R, D, wmin)-local if
the following hold:
• For every point x in the support of Q, Q(B(x, R)) ≥wmin.
• There exist points x1, x2, . . . , xk such that the support of P is a subset of Sk
i=1 B(xi, R).
• Q(B(0, D)) = 1.
However, they left the problem of obtaining a density estimation algorithm for this family open.
The main result of this section is a quasi-polynomial time PAC density estimator for this family.
Theorem 5.1 (PAC density estimator for Gaussian location mixtures). Let M be a (k, R, D, wmin)-
local GLM with variance σ2 ∈(0, 1]. Fix ε ≤min {1/2, σ/R, 1/D, 1/d, wmin}. Define
N =
 
d log 1
ε
 O
 (log 1
ε )7+( R
σ log 1
ε )4 
.
There is an algorithm that, given accuracy parameter ε, instance parameters (σ, k, R, D, wmin),
and sample access to M, draws N i.i.d. samples from M, runs in poly(N) time, and returns an
(ε/δ, δ)-PAC density estimator for M for any coverage parameter δ ∈(0, 1).
Note that due to the requirement on ε, the exponent has an implicit dependence on log (dk) (since
ε ≤wmin ≤1/k).
To the best of our knowledge, this is the first sub-exponential PAC density estimation algorithm
for such a general and non-parametric family of distributions.
To gain some intuition about the above result, consider the special case of spherical Gaussian
mixture models with k components, which have significantly more structure than the (continuous)
general Gaussian location mixture. In this case, we can improve the dependence on d to min {d, k}
by using an SVD-based pre-processing scheme by incurring an additive cost of poly(d) in the running
time (see [VW04]). This results in a time and sample complexity poly(dk/ε) + kpolylog(dk), which
comes very close to the state-of-the-art running time of poly(dk/ε) + (k/ε)O(log2 k) by [DK20]. While
the result has poorer polynomial-dependence in the exponent, the power of the result comes by its
generality in extending beyond Gaussian mixture models – which prohibits the use of specialized
algebraic tools.8
8As noted in [GKL24], the algorithm of [DK20] relies on finding ε-covers of plausible parameters. Since the ε-cover
of even a constant radius ball has size exponential in the dimension, it seems unlikely that their methods will extend
to the more general spherical GLMs that we study.
30


<!-- page 34 -->
Finally, as mentioned before, the guarantees of Theorem 5.1 go beyond spherical Gaussian
mixture models by allowing Q to be a continuous distribution, provided it satisfies Definition 8. In
particular, as a corollary, we obtain a PAC density estimator for the family of distributions satisfying
a weak manifold assumption introduced by [GKL24]. In particular, this assumption requires the
support S of the distribution to be coverable by Cℓℓ2-balls of radius R where ℓ> 0 is a parameter
controlling the sample complexity.
Corollary 1. Fix GLM parameters σ = 1 and C, R > 1 and accuracy parameter 0 < ε < 1/2.
Suppose Q belongs to the family of distributions satisfying the following: Each distribution is
supported on some set SQ such that SQ has radius D, SQ can be covered with Cℓℓ2-balls of radius
R, and every point µ ∈S satisfies Q(B(µ, R)) ≥ε · C−ℓ. Let M be the resulting d-dimensional
spherical Gaussian location mixture, i.e., M = Q ∗N(0, σ2 Id).
Then, there is an algorithm that, given ε, C, ℓ, R, D and sample access to M, draws N =
(d)O(ℓ)+O(log (dD)/ε)7 samples from P, runs in poly(N) time, and outputs an (ε/δ, δ)-PAC density
estimator for M for any coverage parameter δ ∈(0, 1).
As noted in [GKL24], this is a family of distributions for which diffusion models can perform
density estimation while standard methods (such as binning or kernel density estimation) do not
work. In particular, while binning can learn the distribution Q, the above algorithm gives a method
to learn Q convolved with a spherical Gaussian, a more challenging problem.
5.1
Proof of Theorem 5.1
The algorithm in Theorem 5.1 combines our results reducing density estimation to score estimation
(which, in turn, utilizes the connection between DDPM score estimation and MLE) (see Section 2.2)
with the following score estimation algorithm that is implicit in §5.5 of [GKL24].
Theorem 5.2 (General Gaussian mixture score estimator; implicit in §5.5 in [GKL24]). Let M be
an (k, R, D, wmin)-local spherical GLM with variance σ ∈(0, 1]. For ζ ≤min {1/2, σ/R, 1/D, 1/d, wmin}
and η ∈(0, 1), define
Nζ,η =
 
d log 1
η
 O
 (log 1
ζ )7+( R
σ log 1
ζ )4 
.
There is an algorithm that, given accuracy and confidence parameters (ζ, η), time ζσ2 ≲t ≲
log ((d+D)/ζ),9 instance parameters (σ, k, R, D, wmin), and sample access to M, draws Nζ,η i.i.d.
samples from M, runs in poly(Nζ,η) time and, returns a score function bst that satisfies
∥bst −∇log Mt∥2
L2(Mt)
≤
eO
  ζ2 (1 + t)
log(d + D)
 
≤
eO
 ζ2 
,
with probability 1 −η over the samples generated from the mixture M. Here, Mt is the distribution
obtained by running the OU process for time t starting from M0 = M.
Some remarks are in order. First, the above result can be extended to times t larger than log((d+D)/ζ),
although we do not require this guarantee and, hence, to simplify presentation, we omit this. Second,
the specific polynomial dependence of the score estimation error on ζ is not very important as it
only affects constant factors in the exponent. Further, the exponent in the definition of Nζ,η has an
implicit dependence on log d, log D, and log 1/wmin, since we require ζ ≤min {1/2, σ/R, 1/D, 1/d, wmin}.
Further, it also has an implicit dependence on log k since wmin ≤1/k.
9The guarantee in [GKL24] holds for a larger range of times t. We restrict to this range as it is sufficient for our
use and simplifies presentation.
31


<!-- page 35 -->
Next, while the above result only provides a bound on the score estimation error at time t, the
algorithm used computes the score on a list of times t1, . . . , tN, which contains t. This is necessary
because Gatmiry, Kelner, and Lee [GKL24]’s algorithm iterates over times t1, . . . , tN and to estimate
the score at time ti, it requires the clustering at time ti−1, which, in turn, requires an estimate of
the score at time ti−1.
Proof of Theorem 5.2. Let M2 :=
R
∥x∥2 M(dx). Since Q satisfies Definition 8 and M is a convolu-
tion of Q by a spherical Gaussian with variance σ2 ≤1, M2 ≲D2 + d. In §5.5 of [GKL24], they
showed that for any sequence of times 0 < t1 < · · · < tN satisfying Properties P1 and P2 below,
their algorithm satisfies the following score estimation guarantee: for each 1 ≤i ≤N, the score sti
computed for the i-th noise level is ζi-accurate in L2(Mti) for ζ2
i = ζ2 (σ2+t+1)
log(tN+1) , i.e.,
∥sti −∇log Mti∥2
L2(Mti) ≤ζ2
i .
They needed the following properties on the time sequence (t1, . . . , tN):
1. P1 (Start and end points):
t1 ≍ζ2σ2
2
√
d and tN ≍d+M2
ζ2
.
2. P2 (Recurrence):
For each 1 ≤i ≤N −1,
tk + 1 = (tk+1 + 1) · max
 
e−2α, (tk+1 + 1)−α	
where
α :=
ζ2
M2 + d log T + 1 .
The above theorem follows by constructing a sequence 0 < t1 < · · · < tN satisfying the above
properties and containing the noise scale t provided in the theorem (i.e., ensuring there is an index
i with ti = t).
We are now ready to employ our polynomial-time reduction to produce a PAC density estimator.
To do this, we first have to verify that the well-conditioned model M satisfies the mild assumptions
(see Assumptions 1 and 2) required by our reduction from PAC density estimation to score estimation.
Recall that Assumptions 1 and 2 are parameterized by constants L (sub-Gaussian score) and M2
(second moment bound). To bound these, we use the following lemma.
Lemma 8. Let M be an (k, R, D, wmin)-local GLM with variance σ ∈(0, 1]. Let L and M2 be the
following constants:
L = 1
σ
and
M2 = D2 + σ2d .
Then, ∇log M is
√
L-sub-Gaussian under M, and
R
∥x∥2 M(dx) ≤M2. Hence, the mixture M
satisfies Assumptions 1 and 2 with the constants above.
Proof of Lemma 8. Recall that M = Q ∗N(0, σ2 Id). We divide the proof into two parts.
Sub-Gaussian score.
By Lemma 6, it suffices to show that Nσ2 = N(0, σ2 Id) has a σ−1-sub-
Gaussian score, and this follows from Lemma 4 since ∇log Nσ2(x) = σ−2x is σ−2-Lipschitz.
Bound on the second moment. Since M is a convex combination of N(µ, σ2 Id) for µ in the
support of Q, it suffices to bound
R
∥· ∥2 dN(µ, σ2 Id) = ∥µ∥2 + σ2d ≤D2 + σ2d.
Now, we are ready to prove Theorem 5.1.
32


<!-- page 36 -->
Proof of Theorem 5.1. Theorem 2.3 implies that to obtain an (ε/δ, δ) PAC density estimator for any
coverage probability δ > 0, it is sufficient to make C = C(ε, L, M2) calls to a score estimation oracle
with aggregated error ε∗= ε∗(ε, L, M2), terminal time T ≍log(1 + 2M2/ε), where using the values of
L and M2 from Lemma 8, we have
C = eO
 d2 log2 D
σε2
 
,
ε∗= eO
 ε
p
d log D/σ
 
,
and
T ≍log
 D + d
ε
 
.
(6)
Moreover, the calls to the score estimation oracle are made at times t satisfying τ ≤t ≤T, where
the starting time τ is τ ≍
ε2
Ld2 and, substituting L = 1/σ, is
τ ≍σε2
d2 .
(7)
It remains to show that Theorem 5.2 with a suitably small ζ implies an efficient score estimation
oracle with the desired aggregate error. First, note that to obtain an aggregate error
R T
δ ε2
t dt ≤ε2
∗, it
suffices to let accuracy ζ ≤eO(ε∗/
√
T) for each time t it is queried. Next, the reduction in Theorem 2.3
makes N queries with timescale τ ≤t ≤T (for τ as in (7)), the requirement on timescales is satisfied
for
poly
 ε
d + D
 
≲ζ ≲σε2
d2 ,
The above observations allow us to use the score estimation oracle in Theorem 5.2 at these times.
For each call 1 ≤i ≤C, we query the score estimation oracle in Theorem 5.2 with
ζ = min
nσε2
d2 , eO
 ε
p
d log D/σ
 o
,
and confidence δ/N. Observe that this choice of ζ satisfies the requirements that poly(ε/(d+D)) ≲
ζ ≲ε2/(Ld2) and ζ ≤eO(ε∗/
√
T) mentioned above; in fact, it satisfies ζ ≥Ω(ε5) as L = 1/σ and
ε ≤min {1/d, σ/R, 1/D} (Lemma 8).
Taking a union bound over all C calls implies that with probability 1 −δ, the construction
in Theorem 2.1 outputs a valid (ε, T)-integrated score estimation oracle as required. It remains
to bound the total number of samples from M used. Since each call to the score estimation in
Theorem 5.2 requires
 
d log N
δ
 O
 (log 1
ζ )7+( R
σ log 1
ζ )4 
samples from M ,
and we ensured ζ ≥Ω(ε5), ε ≤min {σ/R, 1/d, 1/D}, and Equation (6), and the total samples used are
C
 
d log C
δ
 O
 (log 1
ε )7+( R
σ log 1
ε )4 
=
eO
 d2 log2 D
σε2
   
d log C
δ
 O
 (log 1
ε )7+( R
σ log 1
ε )4 
=
 
d log 1
δε
 O
 (log 1
ε )7+( R
σ log 1
ε )4 
.
The running time follows since the score estimation oracle and the reduction in Theorem 2.1 run in
sample-polynomial time.
33


<!-- page 37 -->
6
Cryptographic lower bounds for score estimation
In this section, we discuss how our connections between score estimation and PAC density estimation
imply computational bottlenecks for score estimation. In Section 6.1, we show that an algorithm
that performs PAC density estimation for Gaussian mixtures implies an algorithm for distinguishing
homogeneous Continuous Learning with Errors (hCLWE). In Section 6.2, using standard reductions
from [BRST21; GVV22], we show that this implies an algorithm for the Continuous Learning with
Errors (CLWE) and Learning with Errors (LWE) problems. Combined with our reduction from
PAC density estimation to score estimation, these establish cryptographic hardness results for score
estimation. See Figure 2 for a summary of the reductions.
Remark 6 (SQ lower bounds). A natural question is whether it is meaningful to derive lower
bounds for score estimation in a restricted computational model, such as the Statistical Query (SQ)
model [Kea98]. Notably, existing SQ lower bounds for density estimation, specifically, lower bounds
for evaluators, directly imply SQ lower bounds for score estimation. This follows from the fact that,
information theoretically, generators and evaluators for densities (in the sense of [KMRRSS94])
are equivalent objects and, since the SQ model permits arbitrary additional computation beyond
the SQ queries themselves, this equivalence holds within the SQ framework as well. Consequently,
known reductions from generation to score estimation already yield SQ hardness results; that is, SQ
lower bounds for density evaluation (e.g., [DKS17]) imply corresponding SQ lower bounds for score
estimation. Therefore, we restrict our attention to lower bounds based on complexity theory. To
obtain such computational lower bounds, it seems to be more difficult to use the existing works on
generation because generators and evaluators are not computationally equivalent. However, as we
will see, our framework can be used in a principled way to derive computational bottlenecks for
score estimation (based on standard complexity-theoretic assumptions).
Figure 2: Illustration of reductions between different cryptographic problems, density estimation,
PAC density estimation, and score estimation. We use these reductions to obtain the hardness of
GMM score estimation (Informal Theorem 4 and Theorem 6.2).
34

[CAPTION] Figure 2: Illustration of reductions between different cryptographic problems, density estimation,


<!-- page 38 -->
6.1
PAC density estimation for GMMs implies homogeneous CLWE
Our starting point is the seminal work of Bruna, Regev, Song, and Tang [BRST21], which introduced
CLWE. One of their most important ideas is a reduction from hCLWE (i.e., homogeneous CLWE) to
GMM density estimation. For background on CLWE, we refer the reader to Appendix B.3.
In this section, we modify this reduction and show that PAC density estimation also implies an
algorithm for hCLWE; note that this is not immediate from the reduction in [BRST21] since PAC
density estimation is an easier task than density estimation.
We say that an algorithm A is a (c, ε, δ)-PAC density estimator of the density P if, with
probability at least c over S ∼P n, the output bP ∼A(S) satisfies
E P{x ∈Rd : e−εP(x) ≤bP(x) ≤eεP(x)} ≥1 −δ .
Originally, in Definition 3, we stated this definition with c = 9/10 because this parameter can be
boosted, but it is convenient to keep c as a free parameter for some of the results in this section. A
model bP that satisfies the above guarantee is simply called a (c, ε, δ)-PAC density estimator for P.
Next, we show that an algorithm that solves PAC density estimation for d-dimensional GMMs
with k components can be used to solve hCLWEβ,γ better than the na¨ıve algorithm that selects one
option uniformly at random for some choice of k, β, and γ.
Before stating the result, let us shortly recall hCLWE. For details, see Appendix B.3.
Definition 9 (hCLWE [BRST21]). For parameters β, γ > 0, the average-case decision problem
hCLWEβ,γ is to distinguish with probability > 1/2 the following two distributions over Rd:
(H0) The Gaussian distribution in d dimensions with mean 0 and covariance Id/(2π).
(H1) the hCLWE distribution Hw,β,γ (see Definition 17) for some uniformly random unit vector
w ∈Rd (which is fixed for all samples).
This distinguishing problem is considered to be computationally hard for some parameters β(d), γ(d)
even for advantage 1/poly(d). For instance, the main result of Bruna, Regev, Song, and Tang
[BRST21] is that when β(d) ∈(0, 1) and γ(d) ≥2
√
d (such that the ratio γ/β is polynomially
bounded), then there is a polynomial-time quantum reduction from standard lattice problems
such as GapSVPα [Reg09] (for some approximation factor α ≈d/β) to hCLWEβ,γ. In other words,
an efficient algorithm for hCLWE would imply an efficient quantum algorithm that approximates
worst-case lattice problems within polynomial factors. We prove the following reduction from hCLWE
to PAC density estimation for Gaussian mixture models.
Proposition 2 (PAC density estimation for GMMs implies hCLWE). Let β = β(d) ∈(0, 1/32),
γ = γ(d) ≥1, and g(d) ≥4π. For k ≳γ
p
g(d), if there is an exp(g(d))-time algorithm that solves
(9/10, ε, δ)-PAC density estimation for mixtures of 2k + 1 Gaussians in d dimensions for sufficiently
small absolute constants ε, δ, then there is a O(exp(g(d)))-time algorithm that solves hCLWEβ,γ.
Since the reduction of [BRST21] from lattice-based problems to hCLWE holds in the regime where
β ∈(0, 1) and γ ≥2
√
d, the above result (taking g(d) = O(log d)), implies that (9/10, ε, δ)-PAC
density estimation for GMMs with Ω(√d log d) components requires super-polynomial time for some
absolute constants ε, δ; otherwise there is a polynomial-time quantum algorithm for standard lattice
problems. Before proving this result, we need the following two intermediate lemmas.
Lemma 9 (PAC density estimation solves simple vs. composite testing). Fix ε, δ ∈(0, 1). Let
H0 = {P0} and H1 be families of probability distributions. Let bP be a (c, ε, δ)-PAC density estimator
over H0 ∪H1. Then, there is a tester that draws m (fresh) i.i.d. samples from P ∈H0 ∪H1, makes
m queries to bP, and has the following properties:
35


<!-- page 39 -->
1. If P = P0, then the tester outputs H0 with probability at least c (1 −δ)m.
2. If P ∈H1, dTV(P0, P) ≥1/10, ε ≤1/160, and δ ≤1/80, then the tester outputs H1 with probability
at least c (1 −(79/80)m).
In particular, if c > 1/2 and we take m, 1/ε, and 1/δ to be sufficiently large absolute constants, then
in both cases the probability of success is strictly larger than 1/2.
Proof. The testing algorithm draws m points x1, . . . , xm i.i.d. from P and evaluates each one of
them using the model bP. Then it outputs H1 if there exists an 1 ≤i ≤m such that
bP(xi)
P0(xi) /∈[e−ε, eε] .
(8)
Otherwise, it outputs H0. Throughout, we always work conditionally on the event of probability at
least c that the PAC density estimation succeeds.
1. Let us assume that P = P0. By the guarantee of bP, for each i, the probability that condition (8)
is satisfied is at least 1−δ. Thus, the probability that the tester outputs H0 is at least (1 −δ)m.
2. Now, let us assume that P ∈H1 and dTV(P0, P) > 1/10. Let S = {x ∈Rd : P(x) > P0(x)}.
By the definition of total variation distance, P(S) −P0(S) > 1/20. For η ∈(0, 1), consider the
set T = {x ∈S : P(x)/P0(x) > eη}. We show that P(T) ≥1/40 provided that η ≤1/80.
For any x ∈S \ T, we have P(x) ≤eηP0(x), which means that P(S \ T) −P0(S \ T) ≤
(eη −1) P0(S \ T) ≤2η. Hence, we can write
1
20 ≤P(S) −P0(S) = P(T) + P(S \ T) −P0(T) −P0(S \ T) ≤P(T) + 2η .
Choosing η = 1/80, we obtain P(T) ≥1/40.
Next, consider the event G := { bP(x1)/P(x1) ≥e−ε}. Note that on the event G ∩{x1 ∈T},
bP(x1) ≥e−εP(x1) ≥e
1/80−εP0(x1) ≥eεP0(x1) ,
provided ε ≤1/160, and in this case the tester outputs H1. Also, by the PAC guarantee, the
event G ∩{x1 ∈T} has probability at least 1/40 −δ ≥1/80, provided δ ≤1/80. The probability
that the tester fails to reject on any of the m samples is therefore bounded by (79/80)m.
The next lemma is used to show that PAC density estimation over mixtures of finitely many
Gaussians is enough to perform PAC density estimation over hCLWE distributions, which technically
are Gaussian mixtures with infinitely many components.
Lemma 10 (hCLWE from mixtures of Gaussians). Assume that there is a (9/10, ε, δ)-PAC density
estimation algorithm for mixtures of 2k+1 Gaussians in d dimensions that uses exp(g(d)) samples for
sufficiently small absolute constants ε, δ. Then, there is a (9/10−2e−4π, ε+4e−4π/δ, 2δ +2e−4π)-PAC
density estimation algorithm for the distribution Hw,β,γ that uses exp(g(d)) samples.
We recall that the success probability above can be increased from 9/10 −2e−4π via boosting (by,
e.g., computing the median of several PAC density estimators).
36


<!-- page 40 -->
Proof. The idea for this lemma follows from the work of Bruna, Regev, Song, and Tang [BRST21].
From [BRST21, Proposition 5.2], we know that if we truncate the distribution Hw,β,γ to its first
2k + 1 central mixture components to form a GMM H(k) with 2k + 1 components, it holds that
dTV(Hw,β,γ, H(k)) ≤2 exp(−πk2/(2γ2)) ,
when β = β(d) ∈(0, 1) and γ = γ(d) ≥1.
The total variation distance between the joint distribution of exp(g(d)) samples from Hw,β,γ and
that of exp(g(d)) samples from H(k) is bounded by
exp(g(d)) · 2 exp(−πk2/(2γ2)) .
By picking k = 2γ
p
g(d)/π and since g(d) ≥4π, we get that the total variation is of order
2 exp(−g(d)) ≤2 exp(−4π) .
Hence, we can condition on the event that the exp(g(d)) samples are drawn from H(k) by reducing
the success of the algorithm to 9/10 −2e−4π. We now have to deal with the density ratio. Let us
assume that we have an (ε, δ)-PAC density estimator bP for H(k) and write
bP(x)
Hw,β,γ(x) =
bP(x)
H(k)(x) · H(k)(x)
Hw,β,γ(x) .
Using the above calculations, we know that
dTV(H(k), Hw,β,γ) ≤2 exp(−g(d)) ≤2 exp(−4π) .
Let ε0 ∈(0, 1) and B := {x ∈Rd : |H(k)(x)/Hw,β,γ(x) −1| ≤ε0}. We know that
Z    H(k)(x)
Hw,β,γ(x) −1
   Hw,β,γ(dx) ≤2 exp(−4π) .
Hence, by Markov’s inequality, Hw,β,γ(Bc) = Hw,β,γ{|H(k)/Hw,β,γ −1| > ε0} ≤2 exp(−4π)/ε0. Hence,
there exists a set B with mass at least 1 −2 exp(−4π)/ε0 such that for any x ∈B, the density ratio
is in the interval [1 −ε0, 1 + ε0]. Let us take ε0 = 2 exp(−4π)/δ. Since bP is an (ε, δ)-PAC density
estimator for H(k), we obtain
EH(k)n
x ∈Rd : e−ε−ε0 ≤
bP(x)
Hw,β,γ(x) ≤eε+2ε0o
≥1 −2δ .
This further implies that
EHw,β,γ
n
x ∈Rd : e−ε−ε0 ≤
bP(x)
Hw,β,γ(x) ≤eε+2ε0o
≥1 −2δ −2e−4π .
Hence, bP is a (9/10 −2e−4π, ε + 2ε0, 2δ + 2e−4π)-PAC density estimator for Hw,β,γ.
We are now ready to complete the reduction from hCLWE to PAC density estimation.
37


<!-- page 41 -->
Proof of Proposition 2. We apply the PAC density estimation algorithm to the unknown given
distribution P, which is either DRd or Hw,β,γ. Let bP be the output of the (ε, δ)-PAC learner using
exp(g(d)) samples from P.
We claim that bP is a PAC density estimator for P. Since the guarantee
of the algorithm is that it is a (9/10, ε, δ)-PAC density estimator for mixtures of 2k + 1 Gaussians
in d dimensions, we directly get the guarantee when P = DRd. When P = Hw,β,γ, we instead
apply Lemma 10, which implies that bP is a (9/10 −2e−4π, ε0, δ0)-PAC density estimator for Hw,β,γ
with slightly worsened parameters ε0, δ0.
Next, to obtain a tester for hCLWE, we apply Lemma 9, where P0 = DRd and H1 = {Hw,β,γ :
w ∈Rd , ∥w∥= 1}. It is known that for every Hw,β,γ ∈H1, dTV(DRd, Hw,β,γ) > 1/2. We check the
numerical constants to ensure that the probability of success for the tester is > 1/2 for both cases.
For the case where P = Hw,β,γ, since c = 9/10 −2e−4π > 0.89, it suffices to take m = 66. And for
the case where P = DRd, it suffices to have ce−δ0m > 1/2, so we require δ0 ≤0.0087 (which satisfies
δ0 ≤1/80). Since δ0 = 2δ + 2e−4π, we take δ = 0.0043. This leads to ε0 = ε + 4e−4π/δ ≤1/160
provided that ε ≤0.003. All of the conditions of Lemma 9 are thus met.
6.2
Lower bounds for score estimation for GMMs
In this section, we use our reduction from PAC density estimation to score estimation (Section 2.2)
in combination with our reduction from hCLWE to GMM PAC density estimation to prove compu-
tational hardness results for score estimation for GMMs. The root of computational intractability
will be problems like CLWE and LWE, which are assumed to be intractable due to reductions from
standard problems on lattices [Reg09; BRST21; GVV22]. For the required background on LWE, we
refer the reader to Appendix B.1, for basic properties on lattices to Appendix B.2, and for details
on CLWE, we refer to Appendix B.3.
To illustrate the flavor of our hardness reductions, using the results of Bruna, Regev, Song,
and Tang [BRST21] on CLWE, combined with our reduction, we can exclude score estimation
for Gaussian mixtures with k ≥√d log d components in poly(d, k) time with error smaller than
O(1/√d log d).
This result follows because we can efficiently transform a score estimation oracle to a PAC density
estimator and, using Section 6.1, such a PAC density estimator for GMMs can be used to efficiently
solve the CLWE problem. However, from [BRST21], we know that there is a polynomial-time
quantum reduction from standard lattice problems such as GapSVP and SIVP [Reg09]10 to CLWE.
Note that we can only exclude eO(1/
√
d)-accurate score estimation oracles for GMMs and this is
inherent to our density-to-score estimation reduction (since getting an (ε, δ)-PAC density estimator
for absolute constants ε, δ in Rd requires calling the score estimation oracle with accuracy 1/√d log d).
In the above reduction, we use the hard instance of Bruna, Regev, Song, and Tang [BRST21].
To make the above chain of reductions fully rigorous, we also have to verify that the hard GMM
satisfies the assumptions of our density-to-score estimation reduction.
As we will see later, we can use the follow-up work of [GVV22] and improve the above result
showing that score estimation is computationally intractable for a smaller number of components in
the GMM (see Section 6.2.2).
10GapSVP and SIVP are among the main computational problems on lattices and are believed to be computationally
hard (even with quantum computation) for polynomial approximation factors.
38


<!-- page 42 -->
6.2.1
Score estimation for GMMs implies (homogeneous) CLWE
First, we combine the reduction of Section 6.1 with our PAC density to score estimation reduction
to show:
Theorem 6.1 (GMM Score estimation implies hCLWE). Let β = β(d) ∈(0, 1/32), γ = γ(d) ≥1,
and g(d) ≥4π. For k ≳γ
p
g(d), if there is an exp(g(d))-time algorithm that returns an ε∗-score
estimation oracle for 2k + 1 mixtures of Gaussians in d dimensions that satisfy Assumptions 1
and 2 with parameters max{L, M2} = poly(d) for some ε∗≤C/√d log d for some absolute constant
C > 0, then there is a poly(d) · exp(g(d))-time algorithm that solves hCLWEβ,γ.
Given this result, we can invoke known hardness results for hCLWE [BRST21; GVV22] (see e.g.,
Appendix B.3.3) to show the hardness of score estimation for mixtures of Gaussians.
Proof. Assume that there is an exp(g(d))-time algorithm that returns an ε-score estimation oracle
for 2k + 1 mixtures of Gaussians in d dimensions for ε∗≤C/√d log d. Since the target family
satisfies our assumptions with max{L, M2} = poly(d), we can employ the PAC density estimation
to score estimation reduction: For any sufficiently small constants ε1, δ1, this reduction implies
that there is an (ε1, δ1)-PAC density estimator for 2k + 1 mixtures of Gaussians in dimension
d with N = eO(ε−2
1 Ld2 log2(M2)) = poly(d) calls to the score estimation oracle with accuracy
O(ε1/
p
d log(LM2)) = C/√d log d for some absolute constant C > 0. This means that we can
get, in time poly(d) eg(d), a (9/10, ε1, δ1)-PAC density estimation algorithm for 2k + 1 mixtures of
Gaussians in d dimensions for some sufficiently small absolute constants ε1, δ1. Since k ≳γ
√
d, we
can use Proposition 2 to complete the proof.
Let us now take γ(d) = 2
√
d and g(d) = O(log d). Corollary 4.2 in [BRST21] shows that hCLWEβ,γ
is hard for that choice of γ assuming hardness for worst-case lattice problems. Then, under the same
hardness assumption, Theorem 6.1 excludes a poly(d, k)-time algorithm for GMM score estimation
when k ≳√d log d and the error of the oracle is as small as 1/√d log d.
The last step in order to complete the proof, is to show that the instance that realizes the above
reduction satisfies the assumptions of our density-to-score reduction with parameters L, M2 that
are poly(d, k). The key observation of Bruna, Regev, Song, and Tang [BRST21] is that hCLWE
has a natural interpretation as an instance of mixtures of Gaussians. This mixture has infinitely
many components, but they manage to reduce hCLWE to a truncated version of hCLWE, which
contains only the first 2k + 1 central ones [BRST21, Proposition 5.2]. Finally, using this truncation,
they show that any distinguisher between the hCLWE distribution and the standard multivariate
Gaussian can be used to solve CLWE. Therefore, an algorithm for density estimation for Gaussian
mixtures implies a solver for CLWE. We now explicitly compute the parameters of our assumptions
for the truncated hCLWE distribution with parameters β, γ.
Definition 10 (Truncated hCLWE distribution). Let s be the hidden direction (with ∥s∥= 1). The
density of the truncated hCLWE distribution with parameters β, γ at x is proportional to
k
X
i=−k
ρ√
β2+γ2(i) · ρ(x⊥s) · ρβ/√
β2+γ2
 
⟨s, x⟩−
γ
β2 + γ2 i
 
,
where ρr(x) = exp(−π∥x∥2/r2) and ρ = ρ1 = N(0, Id/(2π)).
The above distribution is a mixture of Gaussians with 2k + 1 components of width β/
p
β2 + γ2
in the secret direction and unit width in the orthogonal space. We next compute the parameters
L, M2 of our assumptions as a function of β, γ, and d.
39


<!-- page 43 -->
Proposition 3. The truncated hCLWE distribution with parameters β, γ has a
√
L-sub-Gaussian
score and second moment bounded by M2 with L = 1 + γ/β and M2 = d + k2/γ2 .
Proof. First, let us state the mean and covariance of each component of the GMM in Definition 10.
Consider the i-th component, which has density
∝ρ(x⊥s) · ρβ/√
β2+γ2
 
⟨s, x⟩−
γ
β2 + γ2 i
 
.
Here, the first term is the standard Gaussian distribution in the directions orthogonal to s, and
the second component is a Gaussian in the direction s, with mean
γ
β2+γ2 · i and variance
β
β2+γ2 .
Therefore, the i-th component itself is also a Gaussian with mean and covariance
µi =
γ · i
β2 + γ2 s
and
Σi =
"
Id
0
0
β2
β2+γ2
#
= Id −
 
β2
β2 + γ2 −1
 
ss⊤.
Observe the following inequality, which will be useful in the subsequent proof:
Σ−1
i
 = λmin(Σi) ≤1 + γ2
β2 .
(9)
Now, we are ready to bound L and M2.
Sub-Gaussian score.
By Lemma 6, it suffices to show each component has a
p
1 + γ2/β2-
sub-Gaussian score, and this follows from Lemma 4 since the i-th component has distribution
Nµi,Σi = N(µi, Σi) and its score is ∇log Nµi,Σi(x) = −Σ−1
i (x −µi) is (1 + γ2/β2)-Lipschitz (see
Equation (9)).
Bound on the second moment. Observe that any −k ≤i ≤k, we have the following bound:
Z
∥· ∥2 dN(µi, Σi) = ∥µi∥2 + Tr(Σi) ≤
(γ · i)2
(β2 + γ2)2 + (d −1) +
β2
β2 + γ2 ≤
(γ · i)2
(β2 + γ2)2 + d
|i|≤k
≤
k2
β2 + γ2 + d .
Since γ ≥1 and β ∈(0, 1), this further simplifies to
Z
∥· ∥2 dN(µi, Σi) ≤k2
γ2 + d .
Since the truncated hCLWE distribution is a convex combination of {N(µi, Σi) : |i| ≤k}, we conclude
that M2 ≤d + k2
γ2 .
6.2.2
LWE-hardness of score estimation for GMMs
The work of Bruna, Regev, Song, and Tang [BRST21] managed to show that hCLWEβ,γ is hard
when γ ≥2
√
d and β is a small constant (and the ratio γ/β is polynomially bounded). Proposition 3
implies that the truncated hCLWE distribution (which is a GMM) with these parameters satisfies
our assumptions with poly(d)-bounded parameters. Hence, our Theorem 6.1 then implies that score
estimation for Gaussian mixtures with eΩ(
√
d) components and error smaller than 1/√d log d would
40


<!-- page 44 -->
imply the existence of an efficient quantum algorithm that approximates worst-case lattice problems
within polynomial factors, which is believed to be hard.
The follow-up work of Gupte, Vafa, and Vaikuntanathan [GVV22] showed a direct reduction from
LWE to CLWE (and hence hCLWE), allowing us to extend the above hardness result to smaller values
of γ(d) (and hence fewer Gaussian components)11. Gupte, Vafa, and Vaikuntanathan [GVV22]
show that assuming the polynomial-hardness assumption on LWE, we get the hardness of CLWE for
γ(d) = dε for any arbitrary small ε > 0 (see [GVV22, Corollary 1, 2] and β = o(1). This can be
used to show the following:
Theorem 6.2 (Cryptographic hardness of score estimation for general GMMs). Let α > 1 be
an absolute constant. Let ℓbe the dimension of LWE, d be the dimension of the GMMs, and
let d = ℓα. Fix some modulus q = ℓ2 and error parameter σ =
√
ℓ. Let also m be such that
ℓ≤m ≤poly(ℓ). Assuming that the LWEq,σ problem (see Definition 12) is hard to distinguish for
poly(ℓ)-time algorithms with advantage Ω(1/m3) and d samples, then there exists no algorithm
implementing the c-score estimation oracle for k Gaussians over Rd in poly(d) time with m samples
for k = d1/(2α) log d and accuracy c ≤O(1/√d log d).
The constant in the exponent of k = d1/(2α) log d, which controls the number of Gaussian components
we want to exclude, leads to some specific regime for LWE. For the exact details on the polynomial
factors in the above statement, we refer to [GVV22, Corollary 9].
The key idea of Gupte, Vafa, and Vaikuntanathan [GVV22] is a reduction from LWE to a sparse12
version of CLWE (which is enabled by a reduction from LWE to LWE with sparse secrets [Mic18;
GVV22]). This sparsity parameter, which we denote by ∆below, will allow us to show hardness for
CLWEβ,γ, roughly speaking even for smaller values of γ compared to the range γ(d) ≥2
√
d, shown
by Bruna, Regev, Song, and Tang [BRST21] (essentially the sparsity level ∆will appear in the
right-hand side of the inequality). We will now sketch the ideas for showing Theorem 6.2, which
can be proved using the reductions from LWE to sparse-CLWE [GVV22, Lemma 20], from CLWE to
hCLWE [BRST21], our reduction from hCLWE to GMM PAC density estimation, and finally, our
reduction from PAC density estimation to score estimation.
Using the first three reductions, one can show the following result for PAC density estimation.
Lemma 11 (Analogue of Corollary 7 in Gupte, Vafa, and Vaikuntanathan [GVV22]). Assume that
the following conditions hold for the parameters m, d, ℓ, σ, q, ∆:
1. 10
√
ln m + ln d ≤σ,
2. ω(σ
√
∆) ≤q ≤poly(ℓ),
3. ∆log2(d/∆) = (1 + Θ(1)) ℓlog2 q,
4. q ≤m2,
5. m ≤poly(d).
Assuming that the LWEq,σ problem in ℓdimensions is hard to distinguish for (T(ℓ) + poly(d))-
time algorithms with advantage Ω(1/m3), there is no algorithm solving PAC density estimation for
Gaussian mixtures in d dimensions with m samples for k = O(√∆log m log d) components.
Combining the above with our reduction from PAC density estimation to score estimation and setting
∆= 4ℓ/(α −1) = 4d1/α/(α −1) for α > 1, ℓ≤m ≤poly(d), q = ℓ2, σ =
√
ℓ, and T(ℓ) = poly(ℓ)
gives Theorem 6.2 with the number of components k = d1/(2α) log d.
11Moreover, the result of Gupte, Vafa, and Vaikuntanathan [GVV22] provides hardness of CLWE under the classical
(instead of quantum) worst-case hardness of GapSVP.
12Sparsity with parameter ∆means that the secret vector of LWE or CLWE has exactly ∆non-zero entries.
41


<!-- page 45 -->
Acknowledgments
We thank Harrison H. Zhou for numerous helpful conversations. Alkis Kalavasis was supported by
the Institute for Foundations of Data Science at Yale.
References
[ABBKS24]
Prashanti Anderson, Mitali Bafna, Rares-Darius Buhai, Pravesh K. Kothari, and
David Steurer. “Dimension reduction via sum-of-squares and improved clustering
algorithms for non-spherical mixtures”. In: arXiv preprint arXiv:2411.12438 (2024)
(cit. on pp. 58, 59).
[ABHLMP18]
Hassan Ashtiani, Shai Ben-David, Nicholas Harvey, Christopher Liaw, Abbas Mehra-
bian, and Yaniv Plan. “Nearly tight sample complexity bounds for learning mixtures
of Gaussians via sample compression schemes”. In: Advances in Neural Information
Processing Systems 31 (2018) (cit. on pp. 5, 58).
[AC23]
Jason M. Altschuler and Sinho Chewi. “Shifted composition II: shift Harnack
inequalities and curvature upper bounds”. In: arXiv preprint 2401.00071 (2023)
(cit. on pp. 14–16).
[AC24]
Jason M. Altschuler and Sinho Chewi. “Shifted composition I: Harnack and reverse
transport inequalities”. In: IEEE Transactions on Information Theory (2024), pp. 1–
1 (cit. on pp. 10, 25).
[ADLS17]
Jayadev Acharya, Ilias Diakonikolas, Jerry Li, and Ludwig Schmidt. “Sample-optimal
density estimation in nearly-linear time”. In: Proceedings of the 2017 Annual ACM-
SIAM Symposium on Discrete Algorithms (SODA). 2017, pp. 1278–1289 (cit. on
p. 6).
[BC91]
Andrew R Barron and Thomas M Cover. “Minimum complexity density estimation”.
In: IEEE transactions on information theory 37.4 (1991), pp. 1034–1054 (cit. on
p. 5).
[BDDD24]
Joe Benton, Valentin De Bortoli, Arnaud Doucet, and George Deligiannidis. “Linear
convergence bounds for diffusion models via stochastic localization”. In: The Twelfth
International Conference on Learning Representations. 2024 (cit. on p. 10).
[BDJKKV22]
Ainesh Bakshi, Ilias Diakonikolas, He Jia, Daniel M. Kane, Pravesh K. Kothari,
and Santosh S. Vempala. “Robustly learning mixtures of k arbitrary Gaussians”. In:
Proceedings of the 54th Annual ACM SIGACT Symposium on Theory of Computing.
STOC 2022. Rome, Italy: Association for Computing Machinery, 2022, pp. 1234–
1247 (cit. on pp. 5, 58).
[BGL01]
Sergey G. Bobkov, Ivan Gentil, and Michel Ledoux. “Hypercontractivity of Hamilton–
Jacobi equations”. In: J. Math. Pures Appl. (9) 80.7 (2001), pp. 669–696 (cit. on
p. 25).
[BGL14]
Dominique Bakry, Ivan Gentil, and Michel Ledoux. Analysis and geometry of Markov
diffusion operators. Vol. 348. Grundlehren der Mathematischen Wissenschaften [Fun-
damental Principles of Mathematical Sciences]. Springer, Cham, 2014, pp. xx+552
(cit. on p. 51).
[BK20]
Ainesh Bakshi and Pravesh Kothari. “Outlier-robust clustering of non-spherical
mixtures”. In: arXiv preprint arXiv:2005.02970 (2020) (cit. on pp. 58, 59).
42


<!-- page 46 -->
[BMR20]
Adam Block, Youssef Mroueh, and Alexander Rakhlin. “Generative modeling with
denoising auto-encoders and Langevin sampling”. In: arXiv preprint 2002.00107
(2020) (cit. on p. 26).
[BRS24]
Adam Block, Alexander Rakhlin, and Abhishek Shetty. “On the performance of
empirical risk minimization with smoothed data”. In: Proceedings of Thirty Seventh
Conference on Learning Theory. Ed. by Shipra Agrawal and Aaron Roth. Vol. 247.
Proceedings of Machine Learning Research. PMLR, June 2024, pp. 596–629 (cit. on
p. 30).
[BRST21]
Joan Bruna, Oded Regev, Min Jae Song, and Yi Tang. “Continuous LWE”. In:
Proceedings of the 53rd Annual ACM SIGACT Symposium on Theory of Computing.
2021, pp. 694–707 (cit. on pp. 6, 8, 11, 34, 35, 37–41, 53–55, 59).
[BS10]
Mikhail Belkin and Kaushik Sinha. “Polynomial learning of distribution families”.
In: 2010 IEEE 51st Annual Symposium on Foundations of Computer Science. IEEE.
2010, pp. 103–112 (cit. on p. 58).
[BS23]
Rares-Darius Buhai and David Steurer. “Beyond parallel pancakes: quasi-polynomial
time guarantees for non-spherical Gaussian mixtures”. In: The Thirty Sixth Annual
Conference on Learning Theory. PMLR. 2023, pp. 548–611 (cit. on pp. 58, 59).
[BV04]
Rene Beier and Berthold V¨ocking. “Typical properties of winners and losers in
discrete optimization”. In: Proceedings of the Thirty-Sixth Annual ACM Symposium
on Theory of Computing. STOC ’04. Chicago, IL, USA: Association for Computing
Machinery, 2004, pp. 343–352 (cit. on p. 30).
[BV08]
Spencer C. Brubaker and Santosh S. Vempala. “Isotropic PCA and affine-invariant
clustering”. In: 2008 49th Annual IEEE Symposium on Foundations of Computer
Science. IEEE. 2008, pp. 551–560 (cit. on p. 58).
[CCLLLS23]
Sitan Chen, Sinho Chewi, Holden Lee, Yuanzhi Li, Jianfeng Lu, and Adil Salim.
“The probability flow ODE is provably fast”. In: Advances in Neural Information
Processing Systems. Ed. by A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt,
and S. Levine. Vol. 36. Curran Associates, Inc., 2023, pp. 68552–68575 (cit. on p. 1).
[CCLLSZ23]
Sitan Chen, Sinho Chewi, Jerry Li, Yuanzhi Li, Adil Salim, and Anru R Zhang.
“Sampling is as easy as learning the score: theory for diffusion models with minimal
data assumptions”. In: International Conference on Learning Representations. 2023
(cit. on pp. 1, 2, 9, 10, 14, 51).
[CDG25]
Giovanni Conforti, Alain Durmus, and Marta Gentiloni Silveri. “KL convergence
guarantees for score diffusion models under minimal data assumptions”. In: SIAM
J. Math. Data Sci. 7.1 (2025), pp. 86–109 (cit. on p. 10).
[CDSS14]
Siu-On Chan, Ilias Diakonikolas, Rocco A. Servedio, and Xiaorui Sun. “Efficient
density estimation via piecewise polynomial approximation”. In: Proceedings of the
Forty-Sixth Annual ACM Symposium on Theory of Computing. 2014, pp. 604–613
(cit. on p. 6).
[CKKMS24]
Gautam Chandrasekaran, Adam Klivans, Vasilis Kontonis, Raghu Meka, and Kon-
stantinos Stavropoulos. “Smoothed analysis for learning concepts with low intrinsic
dimension”. In: Proceedings of Thirty Seventh Conference on Learning Theory.
Ed. by Shipra Agrawal and Aaron Roth. Vol. 247. Proceedings of Machine Learning
Research. PMLR, June 2024, pp. 876–922 (cit. on p. 30).
43


<!-- page 47 -->
[CKS24]
Sitan Chen, Vasilis Kontonis, and Kulin Shah. “Learning general Gaussian mixtures
with efficient score matching”. In: arXiv preprint 2404.18893 (2024) (cit. on pp. 1,
11, 12, 57–59).
[CKVZ24]
Hugo Cui, Florent Krzakala, Eric Vanden-Eijnden, and Lenka Zdeborova. “Analysis
of learning a flow-based generative model from limited sample complexity”. In: The
Twelfth International Conference on Learning Representations. 2024 (cit. on p. 57).
[CLL23]
Hongrui Chen, Holden Lee, and Jianfeng Lu. “Improved analysis of score-based
generative modeling: user-friendly bounds under minimal smoothness assumptions”.
In: International Conference on Machine Learning, ICML 2023. Ed. by Andreas
Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and
Jonathan Scarlett. Vol. 202. Proceedings of Machine Learning Research. PMLR,
2023, pp. 4735–4763 (cit. on p. 10).
[CLT22]
Tianrong Chen, Guan-Horng Liu, and Evangelos A. Theodorou. “Likelihood train-
ing of Schr¨odinger bridge using forward-backward SDEs theory”. In: The Tenth
International Conference on Learning Representations, ICLR. 2022 (cit. on pp. 3,
11).
[Das99]
Sanjoy Dasgupta. “Learning mixtures of Gaussians”. In: 40th Annual Symposium on
Foundations of Computer Science (Cat. No. 99CB37039). IEEE. 1999, pp. 634–644
(cit. on p. 58).
[DGMS22]
Arnaud Doucet, Will Grathwohl, Alexander G. Matthews, and Heiko Strathmann.
“Score-based diffusion meets annealed importance sampling”. In: Advances in Neural
Information Processing Systems 35 (2022), pp. 21482–21494 (cit. on p. 11).
[DHKK20]
Ilias Diakonikolas, Samuel B. Hopkins, Daniel M. Kane, and Sushrut Karmalkar. “Ro-
bustly learning any clusterable mixture of Gaussians”. In: arXiv preprint arXiv:2005.06417
(2020) (cit. on pp. 58, 59).
[Dia16]
Ilias Diakonikolas. “Learning structured distributions.” In: Handbook of Big Data
267 (2016), pp. 10–1201 (cit. on p. 5).
[Dia23]
Michael Diao. “Challenges of score matching”. Blog post available at mzydiao.com/
posts/score-matching. 2023 (cit. on p. 4).
[DK14]
Constantinos Daskalakis and Gautam Kamath. “Faster and sample near-optimal
algorithms for proper learning mixtures of Gaussians”. In: Conference on Learning
Theory. PMLR. 2014, pp. 1183–1213 (cit. on pp. 5, 58).
[DK20]
Ilias Diakonikolas and Daniel M. Kane. “Small covers for near-zero sets of polyno-
mials and learning latent variable models”. In: 2020 IEEE 61st Annual Symposium
on Foundations of Computer Science (FOCS). IEEE. 2020, pp. 184–195 (cit. on
pp. 5, 7, 30, 58).
[DK23]
Ilias Diakonikolas and Daniel M. Kane. Algorithmic high-dimensional robust statis-
tics. Cambridge University Press, 2023 (cit. on p. 59).
[DKKLMS19]
Ilias Diakonikolas, Gautam Kamath, Daniel M. Kane, Jerry Li, Ankur Moitra, and
Alistair Stewart. “Robust estimators in high-dimensions without the computational
intractability”. In: SIAM Journal on Computing 48.2 (2019), pp. 742–864 (cit. on
pp. 5, 58).
44


<!-- page 48 -->
[DKS17]
Ilias Diakonikolas, Daniel M. Kane, and Alistair Stewart. “Statistical query lower
bounds for robust estimation of high-dimensional Gaussians and Gaussian mixtures”.
In: 2017 IEEE 58th Annual Symposium on Foundations of Computer Science
(FOCS). IEEE. 2017, pp. 73–84 (cit. on pp. 34, 59).
[DKXZ24]
Zehao Dou, Subhodh Kotekal, Zhehao Xu, and Harrison H. Zhou. “From optimal
score matching to optimal sampling”. In: arXiv preprint 2409.07032 (2024) (cit. on
pp. 1, 6, 26–29, 57).
[DL01]
Luc Devroye and G´abor Lugosi. Combinatorial methods in density estimation.
Springer Science & Business Media, 2001 (cit. on p. 6).
[FGRVX17]
Vitaly Feldman, Elena Grigorescu, Lev Reyzin, Santosh S. Vempala, and Ying Xiao.
“Statistical algorithms and a lower bound for detecting planted cliques”. In: Journal
of the ACM (JACM) 64.2 (2017), pp. 1–37 (cit. on p. 59).
[FSO06]
Jon Feldman, Rocco A. Servedio, and Ryan O’Donnell. “PAC learning axis-aligned
mixtures of Gaussians with no separation assumption”. In: International Conference
on Computational Learning Theory. Springer. 2006, pp. 20–34 (cit. on pp. 5, 58).
[GKL24]
Khashayar Gatmiry, Jonathan Kelner, and Holden Lee. “Learning mixtures of
Gaussians using diffusion models”. In: arXiv preprint arXiv:2404.18869 (2024)
(cit. on pp. 1, 7, 11, 30–32, 57–59).
[GTC25]
Wei Guo, Molei Tao, and Yongxin Chen. “Complexity analysis of normalizing
constant estimation: from Jarzynski equality to annealed importance sampling and
beyond”. In: arXiv preprint arXiv:2502.04575 (2025) (cit. on p. 11).
[GV01]
Subhashis Ghosal and Aad W. Van der Vaart. “Entropies and rates of convergence
for maximum likelihood and Bayes estimation for mixtures of normal densities”. In:
Annals of Statistics (2001), pp. 1233–1263 (cit. on p. 7).
[GVV22]
Aparna Gupte, Neekon Vafa, and Vinod Vaikuntanathan. “Continuous LWE Is
as hard as LWE & applications to learning Gaussian mixtures”. In: 2022 IEEE
63rd Annual Symposium on Foundations of Computer Science (FOCS). IEEE. 2022,
pp. 1162–1173 (cit. on pp. 6, 8, 11, 34, 38, 39, 41, 55, 59).
[GW00]
Christopher R. Genovese and Larry Wasserman. “Rates of convergence for the
Gaussian mixture sieve”. In: The Annals of Statistics 28.4 (2000), pp. 1105–1127
(cit. on p. 7).
[H´aj72]
Jaroslav H´ajek. “Local asymptotic minimax and admissibility in estimation”. In: Pro-
ceedings of the sixth Berkeley symposium on mathematical statistics and probability.
Vol. 1. 1972, pp. 175–194 (cit. on p. 4).
[Hau92]
David Haussler. “Decision theoretic generalizations of the PAC model for neural net
and other learning applications”. In: Information and Computation 100.1 (1992),
pp. 78–150 (cit. on p. 1).
[HHSY22]
Nika Haghtalab, Yanjun Han, Abhishek Shetty, and Kunhe Yang. “Oracle-efficient
online learning for smoothed adversaries”. In: Advances in Neural Information
Processing Systems. Ed. by S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K.
Cho, and A. Oh. Vol. 35. Curran Associates, Inc., 2022, pp. 4072–4084 (cit. on
p. 30).
45


<!-- page 49 -->
[HJA20]
Jonathan Ho, Ajay Jain, and Pieter Abbeel. “Denoising diffusion probabilistic mod-
els”. In: Advances in Neural Information Processing Systems. Ed. by H. Larochelle,
M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin. Vol. 33. Curran Associates, Inc.,
2020, pp. 6840–6851 (cit. on pp. 4, 51).
[HJWW20]
Yanjun Han, Jiantao Jiao, Tsachy Weissman, and Yihong Wu. “Optimal rates of
entropy estimation over Lipschitz balls”. In: The Annals of Statistics 48.6 (2020),
pp. 3228–3250 (cit. on p. 23).
[HL18]
Samuel B. Hopkins and Jerry Li. “Mixture models, robustness, and sum of squares
proofs”. In: Proceedings of the 50th Annual ACM SIGACT Symposium on Theory
of Computing. 2018, pp. 1021–1034 (cit. on pp. 58, 59).
[HRS20]
Nika Haghtalab, Tim Roughgarden, and Abhishek Shetty. “Smoothed analysis
of online and differentially private learning”. In: Advances in Neural Information
Processing Systems. Ed. by H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan,
and H. Lin. Vol. 33. Curran Associates, Inc., 2020, pp. 9203–9215 (cit. on p. 30).
[HRS24]
Nika Haghtalab, Tim Roughgarden, and Abhishek Shetty. “Smoothed analysis with
adaptive adversaries”. In: J. ACM 71.3 (June 2024) (cit. on p. 30).
[HRSZ17]
Carsten Hartmann, Lorenz Richter, Christof Sch¨utte, and Wei Zhang. “Variational
characterization of free energy: theory and algorithms”. In: Entropy 19.11 (2017),
p. 626 (cit. on p. 11).
[Hyv05]
Aapo Hyv¨arinen. “Estimation of non-normalized statistical models by score match-
ing”. In: Journal of Machine Learning Research 6.24 (2005), pp. 695–709 (cit. on
p. 1).
[Hyv08]
Aapo Hyv¨arinen. “Optimal approximation of signal priors”. In: Neural Computation
20.12 (Dec. 2008), pp. 3087–3110 (cit. on p. 4).
[Jar97]
Christopher Jarzynski. “Nonequilibrium equality for free energy differences”. In:
Physical Review Letters 78.14 (1997), p. 2690 (cit. on pp. 3, 11).
[Kea98]
Michael Kearns. “Efficient noise-tolerant learning from statistical queries”. In:
Journal of the ACM (JACM) 45.6 (1998), pp. 983–1006 (cit. on pp. 34, 59).
[KG22]
Arlene K. H. Kim and Adityanand Guntuboyina. “Minimax bounds for estimating
multivariate Gaussian location mixtures”. In: Electronic Journal of Statistics 16.1
(2022), pp. 1461–1484 (cit. on p. 30).
[KHR23]
Frederic Koehler, Alexander Heckett, and Andrej Risteski. “Statistical efficiency
of score matching: the view from isoperimetry”. In: The Eleventh International
Conference on Learning Representations. 2023 (cit. on pp. 1, 3, 4, 10).
[Kim14]
Arlene K. H. Kim. “Minimax bounds for estimation of normal mixtures”. In:
Bernoulli 20.4 (2014), pp. 1802–1818 (cit. on p. 7).
[KLV24]
Frederic Koehler, Holden Lee, and Thuy-Duong Vuong. “Efficiently learning and
sampling multimodal distributions with data-based initialization”. In: arXiv preprint
arXiv:2411.09117 (2024) (cit. on pp. 1, 12, 57, 58).
[KMRRSS94]
Michael Kearns, Yishay Mansour, Dana Ron, Ronitt Rubinfeld, Robert E. Schapire,
and Linda Sellie. “On the learnability of discrete distributions”. In: Proceedings of
the Twenty-Sixth Annual ACM Symposium on Theory of Computing. 1994, pp. 273–
282 (cit. on pp. 1, 2, 5, 34, 58, 59).
46


<!-- page 50 -->
[KMV10]
Adam T. Kalai, Ankur Moitra, and Gregory Valiant. “Efficiently learning mixtures
of two Gaussians”. In: Proceedings of the Forty-Second ACM Symposium on Theory
of Computing. 2010, pp. 553–562 (cit. on pp. 5, 58, 59).
[KSV05]
Ravindran Kannan, Hadi Salmasian, and Santosh S. Vempala. “The spectral method
for general mixture models”. In: International Conference on Computational Learn-
ing Theory. Springer. 2005, pp. 444–457 (cit. on p. 58).
[KV24]
Frederic Koehler and Thuy-Duong Vuong. “Sampling multimodal distributions with
the vanilla score: benefits of data-based initialization”. In: The Twelfth International
Conference on Learning Representations. 2024 (cit. on p. 1).
[KW56]
Jack Kiefer and Jacob Wolfowitz. “Consistency of the maximum likelihood estima-
tor in the presence of infinitely many incidental parameters”. In: The Annals of
Mathematical Statistics (1956), pp. 887–906 (cit. on pp. 7, 30).
[Lin95]
Bruce G. Lindsay. “Mixture models: theory, geometry and applications”. In: NSF-
CBMS Regional Conference Series in Probability and Statistics 5 (1995), pp. i–163
(cit. on p. 58).
[LL22]
Allen Liu and Jerry Li. “Clustering mixtures with almost optimal separation in
polynomial time”. In: Proceedings of the 54th Annual ACM SIGACT Symposium
on Theory of Computing. 2022, pp. 1248–1261 (cit. on pp. 58, 59).
[LLT23]
Holden Lee, Jianfeng Lu, and Yixin Tan. “Convergence of score-based generative
modeling for general data distributions”. In: International Conference on Algorith-
mic Learning Theory. Ed. by Shipra Agrawal and Francesco Orabona. Vol. 201.
Proceedings of Machine Learning Research. PMLR, 2023, pp. 946–985 (cit. on pp. 2,
9, 10).
[LM22]
Allen Liu and Ankur Moitra. “Learning GMMs with nearly optimal robustness
guarantees”. In: Conference on Learning Theory. PMLR. 2022, pp. 2815–2895 (cit.
on pp. 58, 59).
[LS17]
Jerry Li and Ludwig Schmidt. “Robust and proper learning for mixtures of Gaussians
via systems of polynomial inequalities”. In: Conference on Learning Theory. PMLR.
2017, pp. 1302–1382 (cit. on pp. 5, 58).
[LY24]
Gen Li and Yuling Yan. “A score-based density formula, with applications in
diffusion generative models”. In: arXiv preprint arXiv:2408.16765 (2024) (cit. on
pp. 3, 11).
[LY25]
Gen Li and Yuling Yan. “O(d/T) convergence theory for diffusion probabilistic
models under minimal assumptions”. In: arXiv preprint 2409.18959 (2025) (cit. on
p. 10).
[Lyu09]
Siwei Lyu. “Interpretation and generalization of score matching”. In: Proceedings
of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence. UAI ’09.
Montreal, Quebec, Canada: AUAI Press, 2009, pp. 359–366 (cit. on p. 4).
[Mic18]
Daniele Micciancio. “On the hardness of learning with errors with binary secrets”.
In: Theory of Computing 14.1 (2018), pp. 1–17 (cit. on p. 41).
[MV10]
Ankur Moitra and Gregory Valiant. “Settling the polynomial learnability of mixtures
of Gaussians”. In: 2010 IEEE 51st Annual Symposium on Foundations of Computer
Science. IEEE. 2010, pp. 93–102 (cit. on pp. 5, 58, 59).
47


<!-- page 51 -->
[MW25]
Song Mei and Yuchen Wu. “Deep networks as denoising algorithms: sample-efficient
learning of diffusion models in high-dimensional graphical models”. In: IEEE Trans.
Inform. Theory 71.4 (2025), pp. 2930–2954 (cit. on p. 57).
[Neg22]
Jeffrey Negrea. Approximations and scaling limits of Markov chains with applica-
tions to MCMC and approximate inference. Thesis (Ph.D.)–University of Toronto
(Canada). Ann Arbor, MI: ProQuest LLC, 2022, p. 204 (cit. on p. 16).
[OAS23]
Kazusato Oko, Shunta Akiyama, and Taiji Suzuki. “Diffusion models are minimax
optimal distribution estimators”. In: Proceedings of the 40th International Confer-
ence on Machine Learning. Ed. by Andreas Krause, Emma Brunskill, Kyunghyun
Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett. Vol. 202. Proceed-
ings of Machine Learning Research. PMLR, July 2023, pp. 26517–26582 (cit. on
pp. 1, 26, 57).
[OV01]
Felix Otto and C´edric Villani. “Comment on: “Hypercontractivity of Hamilton–
Jacobi equations” [J. Math. Pures Appl. (9) 80 (2001), no. 7, 669–696] by S. G.
Bobkov, I. Gentil and M. Ledoux”. In: J. Math. Pures Appl. (9) 80.7 (2001), pp. 697–
700 (cit. on p. 13).
[Pea94]
Karl Pearson. “Contributions to the mathematical theory of evolution”. In: Philo-
sophical Transactions of the Royal Society of London. A 185 (1894), pp. 71–110
(cit. on p. 58).
[PRSLMR23]
Chirag Pabbaraju, Dhruv Rohatgi, Anish Prasad Sevekari, Holden Lee, Ankur
Moitra, and Andrej Risteski. “Provable benefits of score matching”. In: Advances in
Neural Information Processing Systems. Ed. by A. Oh, T. Naumann, A. Globerson, K.
Saenko, M. Hardt, and S. Levine. Vol. 36. Curran Associates, Inc., 2023, pp. 61306–
61326 (cit. on pp. 1, 11).
[PS25]
Yury Polyanskiy and Mark Sellke. “Nonparametric MLE for Gaussian location mix-
tures: certified computation and generic behavior”. In: arXiv preprint arXiv:2503.20193
(2025) (cit. on p. 7).
[QR24]
Yilong Qin and Andrej Risteski. “Fit like you sample: sample-efficient general-
ized score matching from fast mixing diffusions”. In: The Thirty Seventh Annual
Conference on Learning Theory. PMLR. 2024, pp. 4413–4457 (cit. on pp. 1, 4).
[Reg09]
Oded Regev. “On lattices, learning with errors, random linear codes, and cryptog-
raphy”. In: Journal of the ACM (JACM) 56.6 (2009), pp. 1–40 (cit. on pp. 35, 38,
52, 53).
[Rob56]
Herbert Robbins. “An empirical Bayes approach to statistics”. In: Proceedings of the
Third Berkeley Symposium on Mathematical Statistics and Probability, 1954–1955,
vol. I. Univ. California Press, Berkeley-Los Angeles, Calif., 1956, pp. 157–163 (cit. on
p. 27).
[Rou21]
Tim Roughgarden. Beyond the worst-case analysis of algorithms. Cambridge Uni-
versity Press, 2021 (cit. on p. 30).
[RW84]
Richard A. Redner and Homer F. Walker. “Mixture densities, maximum likelihood
and the EM algorithm”. In: SIAM Review 26.2 (1984), pp. 195–239 (cit. on p. 58).
[SCK23]
Kulin Shah, Sitan Chen, and Adam Klivans. “Learning mixtures of Gaussians using
the DDPM objective”. In: Advances in Neural Information Processing Systems 36
(2023), pp. 19636–19649 (cit. on pp. 1, 11, 57).
48


<!-- page 52 -->
[Sco15]
David W. Scott. Multivariate density estimation: theory, practice, and visualization.
John Wiley & Sons, 2015 (cit. on p. 6).
[SDME21]
Yang Song, Conor Durkan, Iain Murray, and Stefano Ermon. “Maximum likelihood
training of score-based diffusion models”. In: Advances in Neural Information
Processing Systems. Ed. by M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang,
and J. Wortman Vaughan. Vol. 34. Curran Associates, Inc., 2021, pp. 1415–1428
(cit. on pp. 3, 11).
[SE19]
Yang Song and Stefano Ermon. “Generative modeling by estimating gradients of
the data distribution”. In: Advances in neural information processing systems 32
(2019) (cit. on p. 4).
[SG20]
Sujayam Saha and Adityanand Guntuboyina. “On the nonparametric maximum
likelihood estimator for Gaussian location mixture densities with application to
Gaussian denoising”. In: The Annals of Statistics 48.2 (2020), pp. 738–762 (cit. on
pp. 7, 30).
[SME21]
Jiaming Song, Chenlin Meng, and Stefano Ermon. “Denoising diffusion implicit
models”. In: International Conference on Learning Representations. 2021 (cit. on
p. 51).
[SOAJ14]
Ananda Theertha Suresh, Alon Orlitsky, Jayadev Acharya, and Ashkan Jafarpour.
“Near-optimal-sample estimators for spherical Gaussian mixtures”. In: Advances in
Neural Information Processing Systems 27 (2014) (cit. on pp. 5, 58).
[Son24]
Min Jae Song. “Cryptographic hardness of score estimation”. In: The Thirty-Eighth
Annual Conference on Neural Information Processing Systems. 2024 (cit. on pp. 8,
11).
[ST04]
Daniel A. Spielman and Shang-Hua Teng. “Smoothed analysis of algorithms: why
the simplex algorithm usually takes polynomial time”. In: J. ACM 51.3 (May 2004),
pp. 385–463 (cit. on p. 30).
[TSM85]
D. Michael Titterington, Adrian F. M. Smith, and Udi E. Makov. Statistical anal-
ysis of finite mixture distributions. Wiley Series in Probability and Mathematical
Statistics: Applied Probability and Statistics. John Wiley & Sons, Ltd., Chichester,
1985, pp. x+243 (cit. on p. 58).
[Tsy09]
Alexandre B. Tsybakov. Introduction to nonparametric estimation. Springer Series
in Statistics. Revised and extended from the 2004 French original, Translated by
Vladimir Zaiats. Springer, New York, 2009, pp. xii+214 (cit. on p. 2).
[Vaa98]
Aad W. van der Vaart. Asymptotic statistics. Vol. 3. Cambridge Series in Statistical
and Probabilistic Mathematics. Cambridge University Press, Cambridge, 1998,
pp. xvi+443 (cit. on pp. 4, 25, 26).
[Vin11]
Pascal Vincent. “A connection between score matching and denoising autoencoders”.
In: Neural Computation 23.7 (2011), pp. 1661–1674 (cit. on p. 4).
[VJ08]
Suriyanarayanan Vaikuntanathan and Christopher Jarzynski. “Escorted free energy
simulations: improving convergence by reducing dissipation”. In: Physical Review
Letters 100.19 (2008), p. 190601 (cit. on p. 11).
[VW02]
Santosh S. Vempala and Grant Wang. “A spectral algorithm for learning mixtures of
distributions”. In: The 43rd Annual IEEE Symposium on Foundations of Computer
Science, 2002. Proceedings. IEEE. 2002, pp. 113–122 (cit. on p. 58).
49


<!-- page 53 -->
[VW04]
Santosh S. Vempala and Grant Wang. “A spectral algorithm for learning mixture
models”. In: Journal of Computer and System Sciences 68.4 (2004). Special Issue
on FOCS 2002, pp. 841–860 (cit. on p. 30).
[Wai19]
Martin J. Wainwright. High-dimensional statistics. Vol. 48. Cambridge Series in
Statistical and Probabilistic Mathematics. A non-asymptotic viewpoint. Cambridge
University Press, Cambridge, 2019, pp. xvii+552 (cit. on p. 12).
[Wan06]
Feng-Yu Wang. “Dimension-free Harnack inequality and its applications”. In: Front.
Math. China 1.1 (2006), pp. 53–72 (cit. on p. 25).
[WWY24]
Andre Wibisono, Yihong Wu, and Kaylee Y. Yang. “Optimal score estimation
via empirical Bayes smoothing”. In: Proceedings of Thirty Seventh Conference on
Learning Theory. Ed. by Shipra Agrawal and Aaron Roth. Vol. 247. Proceedings of
Machine Learning Research. PMLR, June 2024 (cit. on pp. 1, 26, 57).
[Yan+23]
Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao,
Wentao Zhang, Bin Cui, and Ming-Hsuan Yang. “Diffusion models: a comprehensive
survey of methods and applications”. In: ACM Comput. Surv. 56.4 (Nov. 2023)
(cit. on p. 4).
[YB99]
Yuhong Yang and Andrew Barron. “Information-theoretic determination of minimax
rates of convergence”. In: Ann. Statist. 27.5 (1999), pp. 1564–1599 (cit. on p. 28).
[YP25]
Konstantin Yakovlev and Nikita Puchkin. “Generalization error bound for denoising
score matching under relaxed manifold assumption”. In: arXiv preprint 2502.13662
(2025) (cit. on p. 29).
[Zha09]
Cun-Hui Zhang. “Generalized maximum likelihood estimation of normal mixture
densities”. In: Statistica Sinica (2009), pp. 1297–1318 (cit. on p. 7).
[ZYLL24]
Kaihong Zhang, Heqi Yin, Feng Liang, and Jingbo Liu. “Minimax optimality of
score-based diffusion models: beyond the density lower bound assumptions”. In:
Proceedings of the 41st International Conference on Machine Learning. Ed. by
Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver,
Jonathan Scarlett, and Felix Berkenkamp. Vol. 235. Proceedings of Machine Learning
Research. PMLR, July 2024, pp. 60134–60178 (cit. on p. 26).
50


<!-- page 54 -->
A
Background on denoising diffusion probabilistic modeling
In this section, we provide standard background on denoising diffusion probabilistic models (DDPMs);
see Chen, Chewi, Li, Li, Salim, and Zhang [CCLLSZ23] for further details on sampling.
DDPM employs two Markov chains. The first Markov chain iteratively adds noise to the data
and the second Markov chain reverses this process, converting noise back to the original data. The
first Markov chain is usually handcrafted, the most prevalent choice being the addition of standard
Gaussian noise. The second Markov chain, i.e., the reverse process, is parameterized by learned
neural networks. Below, we present the continuous time extension of DDPM with standard Gaussian
noise, which corresponds to the Ornstein–Uhlenbeck (OU) process in continuous time.
Forward process arising from OU. The forward process arising from the OU process is the
following stochastic differential equation (SDE):
dXt = −Xt dt +
√
2 dBt ,
X0 ∼P ,
(10)
where (Bt)t≥0 is a standard Brownian motion in Rd. The forward process transforms samples from
the data distribution P into standard Gaussian noise N(0, 1). Namely, if we denote by Pt the law of
Xt, then limt→∞Pt = N(0, 1). In fact, the convergence is exponentially fast in many metrics and
divergences [BGL14].
Reversing the OU process. The ultimate goal of generative modeling is to generate samples
from P. To this end, we must reverse the process (10) in time, which yields the reverse process that
transforms pure noise back to samples from the target distribution.
Fix a terminal time T > 0. Denote
X←
t
= XT−t ,
t ∈[0, T] .
It turns out that the time reversal of (10) is
dX←
t
= {X←
t
+ 2 ∇log PT−t(X←
t )} dt +
√
2 dBt ,
X←
0 ∼PT ,
(11)
where now (Bt)t≥0 is the reversed Brownian motion.
Denoising score matching with DDPM
To implement the reverse process, one needs to learn the unknown score functions ∇log PT−t for
t ∈[0, T]. This is where denoising score matching (DSM) is utilized in the DSM–DDPM estimator
[HJA20; SME21]. We start with the objective
arg min
s={st}t∈[0,T ]∈F
Z T
0
∥st −∇log Pt∥2
L2(Pt) dt .
(12)
This objective is not amenable to empirical risk minimization since it involves the unknown score
function ∇log Pt. Instead, we note that for fixed t ∈(0, T],
Z
∥st −∇log Pt∥2 dPt
=
Z
∥st∥2 dPt −2
Z
⟨st, ∇log Pt⟩dPt + constant .
51


<!-- page 55 -->
In this derivation, constant refers to any term that does not depend on the optimization variable
st. Continuing, the second term above can be written
−2
Z
⟨st, ∇log Pt⟩dPt = −2
Z
⟨st(xt), ∇Pt(xt)⟩dxt
= −2
Z D
st(xt), ∇xt
Z
Qt|0(xt | x0) P(dx0)
E
dxt
= −2
ZZ
⟨st(xt), ∇Qt|0(xt | x0)⟩P(dx0) dxt
= −2
ZZ
⟨st(xt), ∇log Qt|0(xt | x0)⟩Qt|0(dxt | x0) P(dx0) .
We deduce that the original problem (12) is equivalent to
arg min
s∈F
Z
ℓDDPM(s; x0) P(dx0) ,
where
ℓDDPM(s; x0) :=
Z T
0
Z  
∥st∥2 −2 ⟨st, ∇log Qt|0(· | x0)⟩
	
dQt|0(· | x0) dt .
This is now amenable to empirical risk minimization, and hence we define the empirical version
bRDDPM
n
(s) := 1
n
n
X
i=1
ℓDDPM(s; x(i)
0 )
where x(1)
0 , . . . , x(n)
0
are samples.
Finally, to see that this is equivalent to Definition 1, we take F = {{∇log Pθ,t}t∈[0,T] : θ ∈Θ}
and we note that by well-known properties of the OU semigroup, if zt ∼N(0, Id) is independent of
x0 and xt := exp(−t) x0 +
p
1 −exp(−2t) zt, then xt ∼Qt|0(· | x0). Then, we can write
∇log Qt|0(xt | x0) = −xt −exp(−t) x0
1 −exp(−2t)
= −
zt
p
1 −exp(−2t)
.
B
Background on LWE and Continuous LWE
B.1
Learning with errors
The Learning with Errors (LWE) problem, introduced by Regev [Reg09], is a fundamental problem
in lattice-based cryptography and theoretical computer science. The problem is inspired by the
classical learning parity with noise (LPN) problem and is believed to be computationally hard, even
for quantum computers.
Let d and q be positive integers, and σ > 0 be the error parameter. We denote the quotient ring
of integers modulo q as Zq = Z/qZ. Central to LWE is the following distribution.
Definition 11 (LWE distribution). For dimension d, modulus q ≥2, and secret vector s ∈Zd,
the LWE distribution As,σ over Zd
q × R/qZ is sampled by independently choosing uniformly random
a ∈Zd
q and e ∼N(0, σ2), and outputting (a, (⟨a, s⟩+ e) mod q).
In words, LWE asks to efficiently distinguish (from polynomially many samples) between the LWE
distribution and the uniform distribution over Zd
q ×R/qZ with strict advantage over random guessing.
52


<!-- page 56 -->
Definition 12 (LWE). Let ε > 0 be the advantage13. For dimension d, number of samples m,
modulus q = q(d) ≥2, and error parameter σ = σ(d) > 0, the average-case decision problem LWEq,σ
is to distinguish with advantage ε the following two distributions over Zd
q × R/qZ using m i.i.d.
samples: (1) the LWE distribution As,σ for some uniformly random s ∈Zd (which is fixed for all
samples), or (2) the uniform distribution over Zd
q × R/qZ.
LWE is conjectured to be computationally hard even for error σ = poly(d), modulus q = poly(d),
and non-negligible advantage, i.e., ε should be at least d−c for some c > 0 [Reg09]. Worst-case
hardness reductions show that breaking LWE is at least as hard as solving certain lattice problems
in the worst case [Reg09]. LWE has since become a cornerstone of post-quantum cryptography, as
its hardness remains robust even against quantum adversaries.
B.2
Background on lattices and discrete Gaussians
A lattice is a discrete additive subgroup of Rd. We also define the discrete Gaussian supported on a
lattice L as follows.
Definition 13 (Discrete Gaussian). For lattice L ⊂Rd, vector y ∈Rd, and parameter r > 0, the
discrete Gaussian distribution Dy+L,r on coset y + L with width r is defined to have support y + L
and probability mass function proportional to ρr, where ρr(y) := exp(−π∥y∥2/r2).
For y = 0, we simply denote the discrete Gaussian distribution on lattice L with width r by DL,r.
An important lattice parameter induced by discrete Gaussians is the smoothing parameter,
defined as follows.
Definition 14 (Smoothing parameter). For lattice L and real ε > 0, we define the smoothing
parameter ηε(L) as
ηε(L) = inf{s > 0 | ρ1/s(L∗\ {0}) ≤ε} ,
where L∗corresponds to the dual lattice of L, defined as L∗:= {y ∈Rd | ⟨x, y⟩∈Z for all x ∈L} .
Intuitively, this parameter is the width beyond which the discrete Gaussian distribution behaves
like a continuous Gaussian [Reg09].
B.2.1
Discrete Gaussian sampling problem
The hardness for CLWE proved in Bruna, Regev, Song, and Tang [BRST21] is based on the hardness
of the following standard problem, which asks for sampling from a discrete Gaussian.
Definition 15 (DGS). For a function φ that maps lattices to non-negative reals, an instance of
DGSφ is given by a lattice L and a parameter r ≥φ(L). The goal is to output an independent
sample whose distribution is within a negligible statistical distance of DL,r.
This problem becomes interesting when one relates φ with the smoothness parameter of the lattice.
It is well-known that the main computational problems on lattices, namely GapSVP and SIVP,
which are believed to be computationally hard (even with quantum computation) for polynomial
approximation factor, can be reduced to DGS (see Regev [Reg09, Section 3.3]) for some choice of φ.
Roughly speaking, if ε < 1/10 and φ(L) ≥
√
2ηε(L), there is a polynomial-time reduction from (a
generalization of) the shortest independent vector problem to DGSφ.
13For a distinguisher A running on two distributions P, Q, we say that A has advantage ε if |Px∼P,A[A(x) =
1] −Px∼Q,A[A(x) = 1]| = ε.
53


<!-- page 57 -->
B.3
Continuous LWE
Our cryptographic hardness result for score estimation requires some background on continuous
LWE. The exposition follows the work of Bruna, Regev, Song, and Tang [BRST21]. The Learning
with Errors (LWE) problem has served as a foundation for many lattice-based cryptographic schemes.
Bruna, Regev, Song, and Tang [BRST21] introduced continuous LWE (CLWE), the continuous
analogue of LWE. In what follows, we let
ρσ(x) = exp(−π∥x∥2/σ2) .
(13)
Following Bruna, Regev, Song, and Tang [BRST21], we also let ρ(x) = exp(−π∥x∥2).
B.3.1
CLWE distributions
We first define the CLWE distribution.
Definition 16 (CLWE distribution [BRST21]). For unit vector w ∈Rd and parameters β, γ > 0,
define the CLWE distribution Aw,β,γ over Rd+1 to have density at (y, z) proportional to
ρ(y) ·
X
i∈Z
ρβ(z + i −γ ⟨w, y⟩) .
The next key distribution is the homogeneous CLWE (hCLWE) distribution, which, roughly speaking,
can be obtained from CLWE by essentially conditioning on z ≈0.
Definition 17 (Homogeneous CLWE distribution [BRST21]). For unit vector w ∈Rd and parameters
β, γ > 0, define the hCLWE distribution Hw,β,γ over Rd to have density at y proportional to
ρ(y) ·
X
i∈Z
ρβ(i −γ ⟨w, y⟩) .
(14)
The hCLWE distribution can be equivalently defined as a mixture of Gaussians. To see this, notice
that Equation (14) is equal to
X
i∈Z
ρ√
β2+γ2(i) · ρ(y⊥w) · ρβ/√
β2+γ2
 
⟨w, y⟩−
γ
β2 + γ2 i
 
,
(15)
where y⊥w denotes the projection of y on the orthogonal space to w. Hence, Hw,β,γ can be viewed
as a mixture of (infinitely many) Gaussian components of width β/
p
β2 + γ2 (which is roughly β/γ
for β ≪γ) in the secret direction, and width 1 in the orthogonal space. The components are equally
spaced, with a separation of γ/(β2 + γ2) between them (which is roughly 1/γ for β ≪γ).
B.3.2
CLWE decision problems
Given the above distributions, we can introduce the decision problems which, under standard
assumptions and for some regime of (β, γ), will be the basis of our density estimation hardness
results. We state the average-case version of these problems but it can be shown that the average
case is as hard as the worst-case decision problem [BRST21].
Following [BRST21], we will denote by
• DRd the Gaussian distribution in d dimensions with 0 mean and covariance Id/(2π).
• T is the quotient group of reals modulo the integers, i.e., T = R/Z = [0, 1).
54


<!-- page 58 -->
• U is the uniform distribution on T.
Given the above, we are now ready to define the decision version of CLWE.
Definition 18 (Decision CLWE [BRST21]). For parameters β, γ > 0, the average-case decision
problem CLWEβ,γ is to distinguish with probability > 1/2 the following two distributions over Rd × T:
(H0) DRd × U, or
(H1) the CLWE distribution Aw,β,γ for some uniformly random unit vector w ∈Rd (which is fixed
for all samples).
Similarly, we define the hCLWE problem as:
Definition 19 (Decision hCLWE [BRST21]). For parameters β, γ > 0, the average-case decision
problem hCLWEβ,γ is to distinguish with probability > 1/2 the following two distributions over Rd:
(H0) DRd, or
(H1) the hCLWE distribution Hw,β,γ for some uniformly random unit vector w ∈Rd (which is fixed
for all samples).
B.3.3
Hardness of CLWE
Now, we present the main hardness results of [BRST21] for CLWE and hCLWE. The hardness results
apply in the regime where γ ≥2
√
d and β(d) is a small constant (but the ratio γ/β is polynomially
bounded). This allows [BRST21] to exclude poly(d, k)-time algorithms for density estimation of
d-dimensional Gaussian mixtures with k ≳√d log d components, assuming the hardness of standard
lattice-based problems.
Theorem B.1 (Hardness of CLWE [BRST21]). Let β = β(d) ∈(0, 1) and γ = γ(d) ≥2
√
d
such that γ/β is polynomially bounded. Then, there is a polynomial-time quantum reduction from
DGS2
√
dηε(L)/β to CLWEβ,γ.
Bruna, Regev, Song, and Tang [BRST21] also show the hardness of hCLWE by reducing from CLWE.
The idea of the reduction is to transform CLWE samples to hCLWE samples using rejection sampling.
Theorem B.2 (Hardness of hCLWE [BRST21]). For any β = β(d) ∈(0, 1) and γ = γ(d) ≥2
√
d such
that γ/β is polynomially bounded, there is a polynomial-time quantum reduction from DGS2
√
2dηε(L)/β
to hCLWEβ,γ.
Finally, using standard reductions from GapSVP to DGS, one can obtain reductions from GapSVP
to CLWE (and hence hCLWE).
Follow-up work by Gupte, Vafa, and Vaikuntanathan [GVV22] has shown improved hardness
results for CLWE by giving a direct reduction from LWE.
For instance, for γ = eO(
√
d) and
β = O(σ
√
d/q), there is a polynomial in d time reduction from LWE in dimension ℓ, with d samples,
modulus q and error parameter σ to CLWEβ,γ in dimension d, as long as d ≫ℓlog2 q and σ ≫1.
55


<!-- page 59 -->
C
Auxiliary lemmas
C.1
Standard facts about sub-Gaussianity
In this section, we define sub-Gaussianity and state standard facts about sub-Gaussian random
variables that are used in our proofs.
Definition 20 (Sub-Gaussian random variable). Given a constant σ > 0, a random variable X ∈R
is said to be σ-sub-Gaussian if for all λ ∈R,
Eeλ (X−EX) ≤eλ2σ2/2 .
We note that there are several definitions of sub-Gaussian random variables which are all equivalent
up to constant factors. In particular, this equivalence implies the following fact.
Fact C.1. Given a constant σ > 0 and a σ-sub-Gaussian random variable X ∈R, for any p ≥1,
E[|X|p]1/p ≲σ√p .
Next, we define a sub-Gaussian random vector.
Definition 21 (Sub-Gaussian random vector). For constant σ > 0, a random vector X ∈Rd is
said to be σ-sub-Gaussian if, for every unit vector v ∈Rd, ⟨v, X⟩is σ-sub-Gaussian.
Hence, in particular, if X ∈Rd is σ-sub-Gaussian, then each coordinate of X is σ-sub-Gaussian.
Further, the Euclidean norm ∥X∥is also sub-Gaussian.
Fact C.2 (Sub-Gaussianity of the Euclidean norm). For any σ > 0 and a random vector X ∈Rd
which is σ-sub-Gaussian (with possibly dependent coordinates), the Euclidean norm of X, i.e., ∥X∥,
is O(σ
√
d)-sub-Gaussian.
C.2
Integral estimates
Lemma 12 (Integral estimates). Let L ≥1 and 0 < τ ≤1 ≤T. For each t > 0, let Lt be as
in Lemma 3. Then, the following estimates hold.
1.
R T
τ Lt dt ≲T + log 1/(τ+1/L).
2.
R T
τ L2
t dt ≲T + 1/(τ+1/L).
3.
R T
τ
Lt/(1−e−2t) dt ≲T + log+(1/Lτ)/(τ + 1/L).
Proof. Let t∗:= log(1 + 1/L)/2 and note that Lt ≤1/(1 −e−2t) for t ≥t∗. For the first integral,
Z T
τ
Lt dt ≤2L (t∗−τ)+ +
Z T
t∗∨τ
1
1 −e−2t dt = 2L (t∗−τ)+ + 1
2 log
e2T −1
e2 (t∗∨τ) −1
≲T + log
1
τ + 1/L .
For the second integral,
Z T
τ
L2
t dt ≤4L2 (t∗−τ)+ +
Z T
t∗∨τ
1
(1 −e−2t)2 dt ≲T +
1
τ + 1/L .
For the third integral,
Z T
τ
Lt
1 −e−2t dt ≲L log+
t∗
τ +
Z T
t∗∨τ
1
(1 −e−2t)2 dt ≲T + log+ 1/Lτ
τ + 1/L .
56


<!-- page 60 -->
C.3
Facts about the H¨older class
The following fact is classical, but we prove it for completeness.
Lemma 13 (Lipschitz estimate for the H¨older class). Let P ∈Hs(C, L) with s > 1. Then, P is
Lipschitz continuous on [−1, 1] with a constant that only depends on s, C, and L.
Proof. We prove via induction that for all k ≤⌊s⌋and all intervals [a, a + ℓ] ⊆[−1, 1], there exists
x ∈[a, a + ℓ] with |DkP(x)| ≤C′, where C′ denotes a constant which only depends on s, C, L,
and ℓand can change from line to line. The base case k = 0 follows from the definition of C
in Definition 7. Next, assuming that the statement holds for an integer k < ⌊s⌋, we split [a, a + ℓ]
into three sub-intervals and note that there must exist x ∈[a, a + ℓ/3], y ∈[a + 2ℓ/3, a + ℓ] such
that |DkP(x)| ∨|DkP(y)| ≤C′. Now, if |Dk+1P| is always large on [a, a + ℓ], say Dk+1P ≥C′′,
this leads to a contradiction for C′′ > 6C′/ℓ, and thus the inductive step holds.
Now we perform backward induction on k and argue that in fact |DkP| ≤C′ on all of [−1, 1].
When s is an integer and k = s, this is true by definition. Otherwise, for k = ⌊s⌋, we know that there
exists x ∈[−1, 1] with |DkP(x)| ≤C′, and that Dk is (s −k)-H¨older continuous, so |DkP| ≤C′ on
[−1, 1]. This verifies the base case. Similarly, assuming that the statement holds for an integer k > 1,
we know that there exists x ∈[−1, 1] such that |Dk−1P(x)| ≤C′ and that Dk−1P is C′-Lipschitz
by the inductive hypothesis; hence, |Dk−1P| ≤C′ on [−1, 1].
D
Further related work
In this section, we present further related work.
Statistical guarantees for diffusion models for learning.
Recent works have established
rigorous statistical guarantees for diffusion models and related score matching estimators. For
example, Oko, Akiyama, and Suzuki [OAS23] bounded the estimation error of the empirical risk
minimizer over a neural network class and demonstrated that diffusion models are nearly minimax-
optimal generators in both the total variation and the Wasserstein distance of order one, provided
that the target density belongs to the Besov space. Cui, Krzakala, Vanden-Eijnden, and Zdeborova
[CKVZ24] employed a two-layer neural network to learn score functions and, in the special case where
the target distribution is a mixture of two Gaussians, they established an error guarantee of Θ(1/n)
for the estimated mean. Further, Wibisono, Wu, and Yang [WWY24] considered sub-Gaussian
densities with Lipschitz-continuous score functions and provided optimal rates for estimating the
scores in the L2-norm. In a related direction, Koehler, Lee, and Vuong [KLV24] showed that
pseudo-likelihood methods can be used to learn low-rank Ising models, which is an example of
using score matching for designing learning methods with provable statistical guarantees. Also,
Mei and Wu [MW25] studied the statistical efficiency of neural networks for approximating score
functions, focusing on the setting of graphical models and variational inference algorithms. Moreover,
Dou, Kotekal, Xu, and Zhou [DKXZ24] studied the score matching (SM) estimator in detail and
established the sharp minimax rate of score estimation for smooth, compactly supported densities
using sophisticated techniques.
Computational properties of the DDPM estimator. Beyond the immense practical success
of DDPM estimators and the growing interest from statisticians, surprisingly, DDPM estimators are
also leading to new provably efficient algorithms for sampling and distribution learning; see e.g.,
the works of Shah, Chen, and Klivans [SCK23], Chen, Kontonis, and Shah [CKS24], and Gatmiry,
Kelner, and Lee [GKL24].
57