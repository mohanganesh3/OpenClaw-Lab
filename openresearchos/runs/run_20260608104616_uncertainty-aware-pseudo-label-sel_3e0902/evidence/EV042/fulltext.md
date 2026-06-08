<!-- page 1 -->
Wasserstein Gradient Boosting: A Framework for
Distribution-Valued Supervised Learning
Takuo Matsubara
The University of Edinburgh
takuo.matsubara@ed.ac.uk
Abstract
Gradient boosting is a sequential ensemble method that fits a new weaker learner to pseudo residuals at
each iteration. We propose Wasserstein gradient boosting, a novel extension of gradient boosting that fits
a new weak learner to alternative pseudo residuals that are Wasserstein gradients of loss functionals of
probability distributions assigned at each input. It solves distribution-valued supervised learning, where
the output values of the training dataset are probability distributions for each input. In classification
and regression, a model typically returns, for each input, a point estimate of a parameter of a noise
distribution specified for a response variable, such as the class probability parameter of a categorical
distribution specified for a response label. A main application of Wasserstein gradient boosting in this
paper is tree-based evidential learning, which returns a distributional estimate of the response parameter
for each input. We empirically demonstrate the superior performance of the probabilistic prediction by
Wasserstein gradient boosting in comparison with existing uncertainty quantification methods.
1
Introduction
Gradient boosting is a celebrated machine learning algorithm that has achieved considerable success with
tabular data [1]. Gradient boosting has been extensively used for point forecasts and probabilistic classification,
yet a relatively small number of studies have been concerned with the predictive uncertainty of gradient
boosting. Predictive uncertainty of machine learning models plays a growing role in today’s real-world
production systems [2]. It is vital for safety-critical systems, such as medical diagnoses [3] and autonomous
driving [4], to assess the potential risk of their actions that partially or entirely rely on predictions from their
models. Gradient boosting has already been applied in a diverse range of real-world applications, including
click prediction [5], ranking systems [6], scientific discovery [7], and data competition [8]. There is a pressing
need for methodology to harness the power of gradient boosting to predictive uncertainty quantification.
In classification and regression, we typically specify a noise distribution p(y | θ) of a response variable y and
use a model to return a point estimate θ(x) of the response parameter for each input x. In recent years, the
importance of capturing uncertainty in the model output θ(x) has increasingly been emphasised [2]. A variety
of approaches have been proposed to obtain a distributional estimate p(θ | x) of the response parameter for
each input x [e.g. 9, 10, 11]. For example, Bayesian neural networks (BNNs) quantify uncertainty in network
weights and propagate it to the space of network outputs. Marginalising the predictive distribution p(y | θ)
over the distributional estimate p(θ | x) has been demonstrated to confer enhanced predictive accuracy and
robustness against adversarial attacks [11]. Furthermore, the dispersion of the distributional estimate has
been used as a powerful indicator for out-of-distribution (OOD) detection [12].
In this context, a line of research based on the concept of evidential learning has recently gained significant
attention [11, 13, 14, 15]. The idea can be interpreted as making use of the ‘individual-level’ posterior
p(θ | yi) of the response parameter θ conditional on each individual datum yi, which arises from the
response-distribution likelihood p(yi | θ) and a user-specified prior p(θ). If each individual-level posterior
1
arXiv:2405.09536v2  [stat.ME]  29 Aug 2024


<!-- page 2 -->
(a) 0 weak learner trained
(b) 15 weak learners trained
(c) 100 weak learners trained
Figure 1: Illustration of inputs and outputs of WGBoost trained on a training set {xi, µi}10
i=1 whose inputs are
10 grid points in [−3.5, 3.5] and output distributions are each a normal distribution µi(θ) = N(θ | sin(xi), 0.5)
over θ ∈R. The blue area indicates the 95% high probability region of the output distribution for each point.
WGBoost returns N particles (red lines) that predicts the output distribution for each input, where this
illustration selects N = 10 and uses a Gaussian kernel regressor as each weaker learner of WGBoost.
falls into a closed form characterised by some hyperparameter, neural networks can be trained by using the
hyperparameter of the individual-level posterior as a target value to predict for each input. Outstanding
performance and computational efficiency of the existing approaches have been delivered in a wide spectrum
of engineering and medical applications [16, 17, 18, 19]. However, the existing approaches are limited to neural
networks and to the case where every individual-level posterior is in closed form so that the finite-dimensional
hyperparameter can be predicted by proxy. In general, posterior distributions are known only up to their
normalising constants and, therefore, require an approximation typically by particles [20].
Without closed-form expression, each individual-level posterior needs to be treated as an infinite-dimensional
output for each input. This challenge poses the following fundamental question:
Consider supervised learning whose outputs are probability distributions. Given a training
set of input values and output distributions {xi, µi}D
i=1, can we build a model that receives
an input x and returns a nonparametric prediction of the output distribution?
Motivated by this question, we formulate a general framework of Wasserstein gradient boosting (WGBoost).
WGBoost receives an input and returns a particle-based prediction of the output probability distribution.
Figure 1 illustrates inputs and outputs of WGBoost. This paper considers application of WGBoost to
evidential learning, where the individual-level posterior p(θ | yi) of the response distribution p(y | θ) is used as
the output distribution µi for each input xi of the training set. Figure 2 compares the pipeline of evidential
learning with Bayesian learning. To the author’s knowledge, WGBoost is the first framework that enables
evidential learning (i) for boosted tree models and (ii) without closed form of individual-level posteriors.
Contributions Our contributions are summarised as follows:
• Section 2 establishes the general framework of WGBoost. It is a novel family of gradient boosting that
returns a set of particles that approximates an output distribution assigned at each input. In contrast
to standard gradient boosting that fits a weak learner to the gradient of a loss function, WGBoost fits
a weak learner to the estimated Wasserstein gradient of a loss functional over probability distributions.
• Section 3 establishes tree-based evidential learning based on WGBoost, with the loss functional specified
by the Kullback–Leibler (KL) divergence. Following modern gradient-boosting libraries [21, 22] that
use second-order gradient boosting (c.f. Section 2.2), we use a second-order WGBoost algorithm built
on an approximate Wasserstein gradient and Hessian of the KL divergence (c.f. Sections 3.3 and 3.4).
• Section 4 demonstrates the performance of probabilistic regression and classification with OOD detection
on real-world tabular datasets in comparison with common uncertainty quantification methods.
2

[CAPTION] Figure 1: Illustration of inputs and outputs of WGBoost trained on a training set {xi, µi}10

[CAPTION] Figure 1 illustrates inputs and outputs of WGBoost. This paper considers application of WGBoost to


<!-- page 3 -->
(a) Bayesian learning of a model f(x, w)
(b) Evidential learning based on WGBoost
Figure 2: Comparison of the pipeline of (a) Bayesian learning and (b) evidential learning based on WGBoost.
The former uses the (global) posterior p(w | {xi, yi}D
i=1) of the model parameter w conditional on all data,
and samples multiple models from it. The latter uses the individual-level posterior p(θ | yi) of the response
parameter θ conditional on each individual datum yi as the output distribution in the training set, and trains
WGBoost to directly returns a particle-based distributional estimate p(θ | x) of θ for each input x.
2
General Formulation of Wasserstein Gradient Boosting
This section presents the general formulation of WGBoost. Section 2.1 recaps the notion of Wasserstein
gradient flows, a ‘gradient’ system of probability distributions that minimises an objective functional in the
Wasserstein space. Section 2.2 recaps the notion of gradient boosting, a sequential ensemble method that fits
a new weak learner to the ‘gradient’ of the remaining loss at each iteration. Section 2.3 combines the above
two notions to establish WGBoost, a novel family of gradient boosting whose output is a set of particles that
approximates an output distribution assigned at each input.
Notation and Setting Let X and Y denote the space of inputs and responses in classification and regression.
Suppose Θ = Rd. Let P2 be the 2-Wasserstein space, that is, a set of all probability distributions on Θ with
finite second moment equipped with the Wasserstein metric [23]. We identify a probability distribution in P2
with its density whenever it exits. Denote by ⊙and ⊘, respectively, elementwise multiplication and division
of two vectors in Rd. Let ∇be the gradient operator. Let ∇2
d be a second-order gradient operator that takes
the second derivative at each coordinate i.e. ∇2
df(θ) = [∂2f(θ)/∂θ2
1, . . . , ∂2f(θ)/∂θ2
d]T ∈Rd.
2.1
Wasserstein Gradient Flow
In the Euclidean space, a gradient flow of a function f means a curve of points xt that solves a differential
equation (d/dt)xt = −∇f(xt) from an initial value x0. That is the continuous-time limit of gradient descent,
which minimises the function f as t →∞. A Wasserstein gradient flow means a curve of probability
distributions µt minimising a given functional F on the 2-Wasserstein space P2. The Wasserstein gradient
flow µt is characterised as a solution of a partial differential equation, known as the continuity equation:
d
dtµt = −∇· (µt∇W F(µt))
given
µ0 ∈P2,
(1)
where ∇W F(µ) : Θ →Θ denotes the Wasserstein gradient of F at µ [24, 25]. Appendix A recaps the
derivation of the Wasserstein gradient and presents the examples for several functionals.
One of the elegant properties of the Wasserstein gradient flow is casting the infinite-dimensional optimisation of
the functional F as a finite-dimensional particle update [23]. The continuity equation (1) can be reformulated
as a dynamical system of a random variable θt ∼µt, such that
d
dtθt = −[∇W F(µt)] (θt)
given
θ0 ∼µ0,
(2)
in the sense that the law µt of the random variable θt is a weak solution of the continuity equation. Consider
the case where the initial measure µ0 is set to the empirical distribution ˆµ0 of N particles {θn
0 }N
n=1. Discretising
3

[CAPTION] Figure 2: Comparison of the pipeline of (a) Bayesian learning and (b) evidential learning based on WGBoost.


<!-- page 4 -->
the continuous-time system (2) by the Euler method with a small step size ν > 0 yields an iterative update
scheme of N particles {θn
m}N
n=1 from step m = 0:


θ1
m+1
...
θN
m+1

=


θ1
m
...
θN
m

+ ν


−[∇W F(ˆµm)](θ1
m)
...
−[∇W F(ˆµm)](θN
m)

,
(3)
where ˆµm denotes the empirical distribution of the particles {θn
m}N
n=1 at step m.
In practice, it is common that the Wasserstein gradient of a chosen functional F is not well-defined for
empirical distributions. In such cases, the particle update scheme (3) is not directly applicable because it
depends on the Wasserstein gradient ∇W F(ˆµm) at the empirical distribution ˆµm. For example, the KL
divergence F(µ) = KL(µ | π) with a reference distribution π has such a Wasserstein gradient [∇W F(µ)](θ) =
−(∇log π(θ) −∇log µ(θ)) ill-defined when µ is an empirical distribution. Hence, one often uses an estimate
or approximation of the Wasserstein gradient of a functional F that is well-defined for empirical distributions,
in order to perform the particle update scheme (3) approximately [e.g. 26, 27, 28, 29, 30]. Our application of
WGBoost in Section 3 uses the ‘smoothed’ Wasserstein gradient of the KL divergence [26] recapped later.
2.2
Gradient Boosting
Gradient boosting [31] is a sequential ensemble method of M multiple weak learners f1, . . . , fM. It iteratively
constructs an ensemble Fm of m weak learners f1, . . . , fm from step m = 0 to M. Given the current ensemble
Fm at step m, it trains a new weak learner fm+1 and constructs the next ensemble Fm+1 by
Fm+1(x) = Fm(x) + νfm+1(x)
(4)
where ν is a shrinkage hyperparameter called a learning rate. The initial state of the ensemble F0(x) at step
m = 0 is typically set to a constant that best fits the data. Any learning algorithm can be used as a weak
learner in principle, although tree-based algorithms are most used [32].
The fundamental idea of gradient boosting is to train the new weak learner fm+1 to approximate the negative
gradient of the remaining error of the current ensemble Fm. Suppose that outputs are vectors in Rd and that
a loss function L measures the remaining error at each data point Ri(Fm(xi)) := L(Fm(xi), yi). The new
weak learner fm+1 is fitted to the set {xi, gi}D
i=1, where the target variable gi is specified as
gi = −∇Ri(Fm(xi)) ∈Rd.
The target variable gi is often called a pseudo residual. At every fixed input xi, the boosting scheme (4)
updates the output of the current ensemble Fm(xi) in the steepest descent direction of the error Ri(Fm(xi)).
Although [31] originally suggested performing an additional line search to determine a scaling constant for
each weak learner, the line search has been reported to have a negligible influence on performance [33].
In modern gradient-boosting libraries such as XGBoost [21] and LightGBM [22], the standard practice is
to use the diagonal (coordinatewise) Newton direction of the remaining error for the target variable of the
new weak learner fm+1. In this case, the new base leaner fm+1 is fitted to the set {xi, gi ⊘hi}n
i=1, where the
negative gradient gi is divided elementwise by the Hessian diagonal hi specified as
hi = ∇2
dRi(Fm(xi)) ∈Rd.
The target variable gi ⊘hi is the diagonal Newton direction that minimises the second-order Taylor approxi-
mation of the remaining error for each coordinate independently. Combining second-order gradient boosting
with tree-based weak learners has demonstrated exceptional scalability and performance [34, 35]. Although it
is possible to use the ‘full’ Newton direction as the target variable of each weak learner, the impracticality of
the full Newton direction has been pointed out [e.g. 36, 37]. In addition, the coordinatewise computability of
the diagonal Newton direction is suitable for popular gradient-boosting tree algorithms [36].
4


<!-- page 5 -->
2.3
Wasserstein Gradient Boosting
Now we consider the setting of ‘distribution-valued’ supervised learning, where we are given a training set of
input vectors and output distributions {xi, µi}D
i=1 ⊂X × P2. Our goal is to construct a model that receives
an input and returns a set of N particles whose empirical distribution approximates the output distribution.
We specify a loss functional D(· | ·) between two probability distributions—such as the KL divergence—to
measure the remaining error Fi(·) = D(· | µi) of the model output for each i-th training output distribution
µi. Our idea is to combine gradient boosting with Wasserstein gradient, where we iteratively construct a set
of N boosting ensembles F 1
m, . . . , F N
m —each of which consists of m weak learners—from step m = 0 to M.
Here, the output F n
m(x) of each n-th boosting ensemble represents the n-th output particle for an input x.
Given the current set of N ensembles F 1
m, . . . , F N
m at step m, WGBoost trains a set of N new weak learners
f 1
m+1, . . . , f N
m+1 and computes the next set of N ensembles F 1
m+1, . . . , F N
m+1 by


F 1
m+1(x)
...
F N
m+1(x)

=


F 1
m(x)
...
F N
m (x)

+ ν


f 1
m+1(x)
...
f N
m+1(x)


(5)
where ν is a learning rate. Similarly to standard gradient boosting, we set the initial state of N ensembles
F 1
0 , . . . , F N
0
at step m = 0 to a set of given constants. Throughout, denote by ˆµm,i the empirical distribution
of the N output particles F 1
m(xi), . . . , F N
m (xi) for each i-th training input xi.
As discussed in Section 2.1, the Wasserstein gradient often needs to be estimated when a distribution is an
empirical distribution. For better presentation, let Gi(µ) denote an estimate of the Wasserstein gradient of
the i-th remaining error Fi(µ) at arbitrary µ. If the original Wasserstein gradient is well-defined for all µ, it is
a trivial estimate to use as Gi(µ). Otherwise, any suitable estimate can be used as Gi(µ). The foundamental
idea of WGBoost is to train the n-th new learner f n
m+1 to approximate the estimated Wasserstein gradient
−Gi(ˆµm,i) evaluated at the n-th boosting output F n
m(xi) for each xi, so that,


f 1
m+1(xi)
...
f N
m+1(xi)

≈


−[Gi (ˆµm,i)]
 F 1
m(xi)
 
...
−[Gi (ˆµm,i)]
 F N
m (xi)
 

.
At every fixed xi, the boosting scheme (5) approximates the particle update scheme (3) for the output
particles F 1
m(xi), . . . , F N
m (xi) under the estimated Wasserstein gradint ∇W Fi(ˆµm,i), by which each boosting
output is updated in the direction to decrease the remianing error Fi(ˆµm,i) = D(ˆµm,i | µi) at step m.
The general procedure of WGBoost is summarised in Algorithm 1. For our application, we focus on the
KL divergence as a choice of the loss functional D(· | ·) and use the Wasserstein gradient estimate based on
kernel smoothing in Section 3. Appendix A presents examples of Wasserstein gradients of several divergences.
Figure 1 illustrates the output of WGBoost using a toy output distribution µi(·) = N(· | sin(xi), 0.5).
Remark 1 (Stochastic WGBoost). Stochastic gradient boosting [38] uses only a randomly sampled subset
of data to fit a new weak learner at each step m to reduce the computational cost. The same subsampling
approach can be applied for WGBoost whenever the dataset is large.
Remark 2 (Second-Order WGBoost). If an estimate of the Wasserstein ‘Hessian’ of the remaining
error Fi is available, the Newton direction of Fi may also be computable [e.g. 39, 40]. We can immediately
implement a second-order WGBoost algorithm by plugging such a Newton direction into Gi(µ) in Algorithm
1. Our default WGBoost algorithm for tree-based evidential learning is built on a diagonal approximate
Newton direction of the KL divergence, aligning with the standard practice in modern gradient-boosting
libraries to use the diagonal Newton direction.
5

[CAPTION] Figure 1 illustrates the output of WGBoost using a toy output distribution µi(·) = N(· | sin(xi), 0.5).


<!-- page 6 -->
Algorithm 1: Wasserstein Gradient Boosting
Input: training set {xi, µi}D
i=1 of input xi ∈X and output distribution µi ∈P2
Parameter : loss functional D(· | ·), estimate Gi(µ) of the Wasserstein gradient of D(µ | µi), particle
number N, iteration M, learning rate ν, weak learner f, initial constants (ϑ1
0, . . . , ϑN
0 )
Output: set of N boosting ensembles (F 1
M, . . . , F N
M) at final step M
(F 1
0 (·), . . . , F N
0 (·)) ←(ϑ1
0, . . . , ϑN
0 )
▷set initial state of N boosting ensembles
for m ←0, . . . , M −1 do
for i ←1, . . . , D do
ˆµm,i ←empirical distribution of set of N output values (F 1
m(xi), . . . , F N
m (xi)) for input xi
for n ←1, . . . , N do
gn
i ←−[Gi(ˆµm,i)] (F n
m(xi))
▷Wasserstein gradient evaluated at n-th output value for xi
end
end
for n ←1, . . . , N do
f n
m+1 ←fit
 
{xi, gn
i }D
i=1
 
▷fit n-th new weak learner to Wasserstein gradients
F n
m+1(·) ←F n
m(·) + νf n
m+1(·)
▷get next state of n-th boosting ensemble
end
end
3
Default Setting for Tree-Based Evidential Learning
This section provides the default setting to implement a concrete WGBoost algorithm for evidential learning,
which enables classification and regression with predictive uncertainty. The individual-level posterior p(θ | yi)
of a response distribution p(y | θ) is used as the output distribution µi in the training set {xi, µi}D
i=1 of
WGBoost. Section 3.1 recaps derivation of the individual-level posterior p(θ | yi), followed by the default
choice of the prior discussed in Section 3.2. We choose the KL divergence as a loss functional D(· | ·) of
WGBoost. Section 3.3 recaps a widely-used approximation of the Wasserstein gradient of the KL divergence
based on kernel smoothing [26]. A further advantage of the kernel smoothing approach is that the approximate
Wasserstein Hessian is available, with which Section 3.4 establishes a second-order WGBoost algorithm
similarly to modern gradient-boosting libraries.
3.1
Individual-Level Posteriors as Output Distributions
Suppose that a response distribution p(y | θ) is specified for a response variable y of one’s classification
or regression problem, as is typically done. Suppose further that a prior distribution pi(θ) of the response
parameter θ is specified at each individual observed input xi. At each individual observation (xi, yi), the
response-distribution likelihood p(yi | θ) and the prior pi(θ) determine the individual-level posterior
p(θ | yi) ∝p(yi | θ)pi(θ).
The individual-level posterior is used as the output distribution µi for each input xi in the training set
{xi, µi}D
i=1 of WGBoost. We apply the framework of WGBoost to construct a model that returns a set of
particles that approximates the output distribution µi(·) = p(· | yi) for each observed input xi.
For a new input x, the trained WGBoost model returns a set of particles (θ1(x), . . . , θN(x)) that is a
nonparametric distributional estimate p(θ | x) of the response parameter. Based on the output particles, a
predictive distribution p(y | x) of the response y for each new input x can be defined via marginalisation:
p(y | x) =
Z
Θ
p(y | θ)p(θ | x)dθ = 1
N
N
X
i=1
p
 y | θi(x)
 
.
(6)
6


<!-- page 7 -->
A point prediction ˆy for each new input x can also be defined via the ‘individual-level’ Bayes action:
ˆy = argminy∈Y
Z
Θ
U(y, θ)p(θ | x)dθ = argminy∈Y
1
N
N
X
i=1
U(y, θi(x)),
which is the minimiser of the average risk of a given utility U : Y × Θ →R. For example, if the utility is a
quadratic function U(y, θ) = (y −θ)2, the Bayes action is the mean of the output particles (θ1(x), . . . , θN(x)).
In general, the explicit form of the individual-level posterior p(θ | yi) is known only up to the normalising
constant. The WGBoost algorithm for evidential learning, provided in Section 3.4, requires no normalising
constant of the individual-level posterior p(θ | yi). It depends only on the log-gradient of the individual-level
posterior ∇log p(θ | yi) = ∇p(θ | yi)/p(θ | yi), cancelling the normalising constant by fraction. Hence,
knowing the form of the response-distribution likelihood p(yi | θ) and the prior pi(θ) suffices.
Remark 3 (Difference from Bayesian Learning). Bayesian learning of a given model f(x, w) uses the
posterior p(w | {xi, yi}D
i=1) of the model parameter w conditional on all data. The predictive distribution
p(y | x) of the response y in Bayesian learning is defined via marginalisation of the model parameter w, that
is, p(y | x) =
R
Θ p(y | θ = f(x, w))p(w | {xi, yi}D
i=1)dw. In contrast, WGBoost learns the nonparametric
distributional estimate p(θ | x) directly from the training set {xi, µi}D
i=1, involving no marginalisation of the
model parameter w that is often exceedingly high dimensional in machine learning.
3.2
Choice of Individual-Level Priors
The prior pi(θ) of the response parameter θ is specified at each individual observed input xi. The approach to
specifying the prior may differ depending on whether past data are available. When past data are available,
past data can be utilised in any possible way to elicit a reasonable prior for future data. When no past
data are available, we recommend the use of a noninformative prior that have been developed as a sensible
choice of prior in the absence of past data; see [e.g. 41] for the introduction. To avoid numerical errors, if
a noninformative prior is improper (nonintegrable) as is often the case, we recommend the use of a proper
probability distribution that approximates the noninformative prior sufficiently well.
Example 1 (Normal Location-Scale). Consider regression with a scalar-valued response variable y ∈R.
A normal location-scale distribution N(y | m, σ) has the mean and scale parameters m ∈R and σ ∈(0, ∞).
A typical noninformative prior of m and σ are given by 1 and 1/σ respectively, which are improper. At every
observation (xi, yi), we use a normal prior N(m | 0, σ0) over m and an inverse gamma prior IG(σ | α0, β0)
over σ, with the hyperparameters σ0 = 10 and α0 = β0 = 0.01, which approximate the non-informative priors.
Example 2 (Categorical). Consider classification with a k-class label response variable y ∈{1, . . . , k}.
A categorical distribution C(y | q) has a class probability parameter q = (q1, . . . , qk) in the k-dimensional
simplex ∆k. If k = 2, it corresponds to the Bernoulli distribution. A typical noninformative prior of q is given
by 1/ Qk
i=1 qi. At every observation (xi, yi), we use the logistic normal prior—a multivariate generalisation of
the logit normal distribution [42]—over q with the mean 0 and identity covariance matrix scaled by 10.
In Section 2, we have supposed that Θ = Rd for some dimension d. Any parameter that lies in a subset of
the Euclidean space (e.g. σ) can be reparametrised as one in the Euclidean space (e.g. log σ). Appendix D
details the reparametrisation used for the experiment. If a dataset has scalar outputs and they have a low or
high order of magnitude, we also recommend standardising the outputs to adjust the magnitude.
3.3
Approximate Wasserstein Gradient of KL Divergence
The loss functional D(µ | µi) considered in this setting is the KL divergence KL(µ | µi).
A com-
putational challenge of the KL divergence is that the associated Wasserstein gradient
 
GKL
i
(µ)
 
(θ) :=
−(∇log πi(θ) −∇log µ(θ)) is not well-defined for empirical distributions. A particularly successful approach
to finding a well-defined approximation of the Wasserstein gradient—which originates in [43] and has been
7


<!-- page 8 -->
applied in wide contexts [26, 44, 45]—is to smooth the original Wasserstein gradient through a kernel integral
operator
R
Θ[GKL
i
(µ)](θ∗)k(θ, θ∗)dµ(θ∗) [46]. By integration-by-part (see [e.g. 43]), the smoothed Wasserstein
gradient—denoted G∗
i (µ)—falls into the following form that is well-defined for any distribution µ:
[G∗
i (µ)] (θ) := −Eθ∗∼µ
h
∇log µi(θ∗)k(θ, θ∗) + ∇k(θ, θ∗)
i
∈Rd,
(7)
where ∇k(θ, θ∗) denotes the gradient of k with respect to the first argument θ. An approximate Wasserstein
gradient flow based on the smoothed Wasserstein gradient G∗
i (µ) is called the Stein variational gradient
descent [43] or kernelised Wasserstein gradient flow [47]. In most cases, the kernel k is set to the Gaussian
kernel k(θ, θ∗) = exp(−∥θ −θ∗∥2/h) with the scale hyperparameter h > 0. We use the Gaussian kernel with
the scale hyperparameter h = 0.1 throughout this work.
Another common approach to approximating the Wasserstein gradient flow of the KL divergence is the
Langevin diffusion approach [48]. The discretised algorithm, called the unadjusted Langevin algorithm [49],
is a stochastic particle update scheme that adds a Gaussian noise at every iteration. However, several known
challenges, such as asymptotic bias and slow convergence, often necessitate an ad-hoc adjustment of the
algorithm [48]. Appendix B discusses a variant of WGBoost built on the Langevin algorithm, although it is
not considered the default implementation.
3.4
Second-Order Implementation of WGBoost
Following the standard practice in modern gradient-boosting libraries [21, 22] to use the diagonal Newton
direction, we further consider a diagonal (coordinatewise) approximate Wasserstein Newton direction of the
KL divergence. In a similar manner to the smoothed Wasserstein gradient (7), the approximate Wasserstein
Hessian of each KL divergence KL(µ | µi) can be obtained by the kernel smoothing. The diagonal of the
approximate Wasserstein Hessian, denoted H∗
i (µ), is defined by
[H∗
i (µ)] (θ) := Eθ∗∼µ
h
−∇2
d log µi(θ∗)k(θ, θ∗)2 + ∇k(θ, θ∗) ⊙∇k(θ, θ∗)
i
∈Rd.
(8)
The diagonal approximate Wasserstein Newton direction of each KL divergence is then defined by −[G∗
i (µ)] (·)⊘
[H∗
i (µ)] (·). Appendix C provides the derivation based on [39] who derived the Newton direction of the KL
divergence in the context of nonparametric variational inference. The second-order WGBoost algorithm is
established by plugging it into Gi(µ) in Algorithm 1 i.e. setting
[Gi(µ)] (·) = [G∗
i (µ)] (·) ⊘[H∗
i (µ)] (·).
(9)
Algorithm 1 under the setting (9) is considered our default WGBoost algorithm for evidential learning. We
refer this algorithm to as the Wasserstein-boosted evidential learning (WEvidential). The explicit pseudocode
is provided in Algorithm 2 for full clarity.
Remark 4 (Computation). The diagonal Newton direction has a clear computational benefit in that
only elementwise division is involved. The computational complexity is the same as that for the smoothed
Wasserstein gradient, scaling linearly to both the particle number N and the particle dimension d. Hence,
there is essentially no reason not to use the diagonal Newton direction instead of the smoothed Wasserstein
gradient. Although it is possible to use the full Newton direction with no diagonal approximation, the inverse
and product of (N ×d)×(N ×d) matrices are required at every computation of the direction (c.f. Appendix D).
Appendix D presents a simulation study to compare computational time and convergence speed of WGBoost
algorithms implemented with four different estimates of the Wasserstein gradient.
4
Applications with Real-world Tabular Data
We empirically demonstrate the performance of the WGBoost algorithm through three applications using
real-world tabular data. The first application illustrates the output of WGBoost through a simple conditional
8


<!-- page 9 -->
Algorithm 2: Wasserstein-Boosted Evidential Learning
Input: dataset {xi, yi}D
i=1 of input xi and response yi of classification or regression
Parameter : individual-level posterior p(θ | yi) of response distribution p(y | θ) conditional on each yi,
particle number N, iteration M, learning rate ν, weak learner f, initial constants {ϑn
0}N
n=1
Output: set of N boosting ensembles (F 1
M, . . . , F N
M) at final step M
(F 1
0 (·), . . . , F N
0 (·)) ←(ϑ1
0, . . . , ϑN
0 )
▷set initial state of N boostings
for m ←0, . . . , M −1 do
for i ←1, . . . , D do
ˆµm,i ←empirical distribution of set of N output values (F 1
m(xi), . . . , F N
m (xi)) for input xi
for n ←1, . . . , N do
gn
i ←Eθ∗∼ˆµm,i[∇log p(θ∗|yi)k(F n
m(xi), θ∗) + ∇k(F n
m(xi), θ∗)]
hn
i ←Eθ∗∼ˆµm,i[−∇2
d log p(θ∗|yi)k(F n
m(xi), θ∗)2 + ∇k(F n
m(xi), θ∗) ⊙∇k(F n
m(xi), θ∗)]
end
end
for n ←1, . . . , N do
f n
m+1 ←fit
 
{xi, gn
i ⊘hn
i }D
i=1
 
▷fit n-th new tree regressor to approximate Newton directions
F n
m+1(·) ←F n
m(·) + νf n
m+1(·)
▷set next state of n-th boosting
end
end
density estimation. The second application benchmarks the regression performance on nine real-world datasets
[50]. The third application examines the classification and OOD detection performance on the real-world
datasets used in [14]. The source code is available in https://github.com/takuomatsubara/WGBoost.
Common Hyperparameters: Throughout, we set the number of output particles N to 10 and set each
weak learner f to the decision tree regressor [51] with maximum depth 1 for the first application and 3 for the
rest. We set the learning rate ν to 0.1 for regression and 0.4 for classification. Appendix E contains further
details, including a choice of the initial constant {ϑn
0}N
n=1.
4.1
Illustrative Conditional Density Estimation
This section illustrates the output of the WGBoost algorithm by estimating a conditional density p(y | x)
from one-dimensional scalar inputs and outputs {xi, yi}D
i=1. The normal output distribution N(y | m, σ) and
the prior pi(m, σ) in Example 1 were used to define the individual-level posterior p(m, σ | yi), in which case
the output of the WGBoost algorithm is a set of 10 particles {(mn(x), σn(x))}10
n=1 of the mean and scale
parameters for each input x. We set the number of weak learners M to 500.
The conditional density is estimated using the predictive distribution (6) by the WGBoost algorithm. We
used two real-world datasets, bone mineral density [52] and old faithful geyser [53]. Figure 3 depicts the
result for the former dataset, demonstrating that the WGBoost algorithm captures the heterogeneity of the
conditional density on each input well. Similarly, Figure 4 depicts the result for the latter dataset.
4.2
Probabilistic Regression Benchmark
This section examines the regression performance of the WGBoost algorithm using a standard benchmark
protocol that originated in [50] and has been used in a number of subsequent works [10, 9, 32]. The benchmark
protocol uses real-world tabular datasets from the UCI machine learning repository [54], each with one-
dimensional scalar responses. As in Section 4.1, the normal response distribution N(y | m, σ) and the prior
pi(m, σ) in Example 1 were used to define the individual-level posterior p(m, σ | yi).
We randomly held out 10% of each dataset as a test set, following the data splitting protocol in [50]. The
9


<!-- page 10 -->
Figure 3: Conditional density estimation for the bone mineral density dataset (grey dots) by WEvidential,
where the normal response distribution N(y | m, σ) is specified for the response variable y. Left: distributional
estimate (10 particles) of the location parameter {mn(x)}10
n=1 for each input. Right: estimated density (6)
based on the normal response distribution averaged over the output particles {(mn(x), σn(x))}10
n=1.
Figure 4: Conditional density estimation for the old faithful geyser dataset (grey dots) by WEvidential. Left:
distributional estimate (10 particles) of the location parameter for each input. Right: estimated density by
the predictive distribution (6) based on the output particles.
negative log likelihood (NLL) is measured by using the predictive distribution (6) by the WGBoost algorithm.
The room mean squared error (RMSE) is measured by using the point prediction produced by taking the
mean value of the predictive distribution. For this benchmark, we followed an approach used in [32] to choose
the number of weak learners M by early-stopping, where we held out 20% of the training set as a validation
set to choose the number 1 ≤M ≤4000 achieving the least validation error. Once the number M was chosen,
the WGBoost algorithm was trained again using all the entire training set. We repeated this procedure 20
times for each dataset, except the protein dataset for which we repeated five times.
We compared the performance of WEvidential with five other methods: Monte Carlo Dropout (MCDropout)
[9], Deep Ensemble (DEnsemble) [10], Concrete Dropout (CDropout) [55], Natural Gradient Boosting
(NGBoost) [32], and Deep Evidential Regression (DEvidential) [13]. Appendix E briefly describes each
algorithm and provides further details on the experiment. Table 1 summarises the NLLs and RMSEs of the
six algorithms. The WGBoost algorithm achieves the best score or a score sufficiently close to the best score
for the majority of the datasets.
4.3
Classification and Out-of-Distribution Detection
This section examines the classification and anomaly OOD detection performance of the WGBoost algorithm
on two real-world tabular datasets, segment and sensorless, following the protocol used in [14]. The categorical
output distribution C(y | q) and the prior pi(q) in Example 2 were used to define the individual-level posterior
p(q | yi), in which case the output of the WGBoost algorithm is a set of 10 particles {qn}10
n=1 of the class
probability parameter q in the simplex ∆k for each input x. We set the number of weak learners M to 4000.
The dispersion of the output particles of the WGBoost algorithm was used for OOD detection [56]. If a test
input was an in-distribution sample from the same distribution as the data, we expected the output particles
10

[CAPTION] Figure 3: Conditional density estimation for the bone mineral density dataset (grey dots) by WEvidential,

[CAPTION] Figure 4: Conditional density estimation for the old faithful geyser dataset (grey dots) by WEvidential. Left:


<!-- page 11 -->
Table 1: The NLLs and RMSEs with the standard deviation. The best score is underlined for each dataset,
and the scores whose standard deviation ranges include the best score are in bold. Results of MCDropout,
DEnsembles, CDropout, NGBoost, and DEvidential were reported in [9], [10], [55], [32] and [13] respectively.
Dataset
Criteria
WEvidential
MCDropout
DEnsemble
CDropout
NGBoost
DEvidential
boston
NLL
2.47 ± 0.16
2.46 ± 0.06
2.41 ± 0.25
2.72 ± 0.01
2.43 ± 0.15
2.35 ± 0.06
concrete
2.83 ± 0.11
3.04 ± 0.02
3.06 ± 0.18
3.51 ± 0.00
3.04 ± 0.17
3.01 ± 0.02
energy
0.53 ± 0.08
1.99 ± 0.02
1.38 ± 0.22
2.30 ± 0.00
0.60 ± 0.45
1.39 ± 0.06
kin8nm
-0.44 ± 0.03
-0.95 ± 0.01
-1.20 ± 0.02
-0.65 ± 0.00
-0.49 ± 0.02
-1.24 ± 0.01
naval
-5.47 ± 0.03
-3.80 ± 0.01
-5.63 ± 0.05
-5.87 ± 0.05
-5.34 ± 0.04
-5.73 ± 0.07
power
2.60 ± 0.04
2.80 ± 0.01
2.79 ± 0.04
2.75 ± 0.01
2.79 ± 0.11
2.81 ± 0.07
protein
2.70 ± 0.01
2.89 ± 0.00
2.83 ± 0.02
2.81 ± 0.00
2.81 ± 0.03
2.63 ± 0.00
wine
0.95 ± 0.08
0.93 ± 0.01
0.94 ± 0.12
1.70 ± 0.00
0.91 ± 0.06
0.89 ± 0.05
yacht
0.16 ± 0.24
1.55 ± 0.03
1.18 ± 0.21
1.75 ± 0.00
0.20 ± 0.26
1.03 ± 0.19
boston
RMSE
2.78 ± 0.60
2.97 ± 0.19
3.28 ± 1.00
2.65 ± 0.17
2.94 ± 0.53
3.06 ± 0.16
concrete
4.15 ± 0.52
5.23 ± 0.12
6.03 ± 0.58
4.46 ± 0.16
5.06 ± 0.61
5.85 ± 0.15
energy
0.42 ± 0.07
1.66 ± 0.04
2.09 ± 0.29
0.46 ± 0.02
0.46 ± 0.06
2.06 ± 0.10
kin8nm
0.15 ± 0.00
0.10 ± 0.00
0.09 ± 0.00
0.07 ± 0.00
0.16 ± 0.00
0.09 ± 0.00
naval
0.00 ± 0.00
0.01 ± 0.00
0.00 ± 0.00
0.00 ± 0.00
0.00 ± 0.00
0.00 ± 0.00
power
3.19 ± 0.25
4.02 ± 0.04
4.11 ± 0.17
3.70 ± 0.04
3.79 ± 0.18
4.23 ± 0.09
protein
4.09 ± 0.02
4.36 ± 0.01
4.71 ± 0.06
3.85 ± 0.02
4.33 ± 0.03
4.64 ± 0.03
wine
0.61 ± 0.05
0.62 ± 0.01
0.64 ± 0.04
0.62 ± 0.00
0.63 ± 0.04
0.61 ± 0.02
yacht
0.48 ± 0.18
1.11 ± 0.09
1.58 ± 0.48
0.57 ± 0.05
0.50 ± 0.20
1.57 ± 0.56
to concentrate on some small region in ∆k indicating a high probability of the correct class. If a test input
was an OOD sample, we expected the output particles to disperse over ∆k because the model ought to be
less certain about the correct class.
The segment and sensorless datasets have 7 and 11 classes in total. For the segment dataset, the data subset
that belongs to the last class was kept as the OOD samples. For the sensorless dataset, the data subset that
belongs to the last two classes was kept as the OOD samples. For each dataset, 20% of the non-OOD samples
is held out as a test set to measure the classification accuracy. Several approaches can define the OOD score
of each input [56]. We focused on an approach that uses the variance of the output particles as the OOD
score. For the WGBoost algorithm, we employed the inverse of the maximum norm of the variance as the
OOD score. Given the OOD score, we measured the OOD detection performance by the area under the
precision recall curve (PR-AUC), viewing non-OOD test data as the positive class and OOD data as the
negative class. We repeated this procedure five times.
We compared the WGBoost algorithm with four other methods: MCDropout, DEnsemble, and Distributional
Distillation (DDistillation) [57], and Posterior Network (PNetwork) [14]. Appendix E briefly describes each
algorithm and provides further details on the experiment. Table 2 summarises the classification and OOD
detection performance of the five algorithms. The WGBoost algorithm demonstrates a high classification and
Table 2: The classification accuracies and OOD detection PR-AUCs with the standard deviation. For each
dataset, the best score is underlined and in bold. The results other than WEvidential were reported in [14].
Dataset
Criteria
WEvidential
MCDropout
DEnsemble
DDistillation
PNetwork
segment
Accuracy
96.57 ± 0.6
95.25 ± 0.1
97.27 ± 0.1
96.21 ± 0.1
96.92 ± 0.1
OOD
99.67 ± 0.2
43.11 ± 0.6
58.13 ± 1.7
35.83 ± 0.4
96.74 ± 0.9
sensorless
Accuracy
99.54 ± 0.1
89.32 ± 0.2
99.37 ± 0.0
93.66 ± 1.5
99.52 ± 0.0
OOD
81.13 ± 5.3
40.61 ± 0.7
50.62 ± 0.1
31.17 ± 0.2
88.65 ± 0.4
11

[CAPTION] Table 1: The NLLs and RMSEs with the standard deviation. The best score is underlined for each dataset,

[CAPTION] Table 2: The classification accuracies and OOD detection PR-AUCs with the standard deviation. For each


<!-- page 12 -->
OOD detection accuracy simultaneously. Although PNetwork has the best OOD detection performance for
the sensorless dataset, the performance of the WGBoost algorithm also exceeds 80%, which is distinct from
MCDropout, DEnsemble, and DDistillation.
5
Discussion
This work established the general framework of WGBoost for ’distribution-valued’ supervised learning, which
receives a particle-based approximation of an output distribution assigned at each input. We focused on
application of WGBoost to evidential learning, for which we provided the setting to implement a second-order
WGBoost algorithm, aligning with the standard practice in modern gradient-boosting libraries. We empirically
demonstrated that the probabilistic forecast by WGBoost leads to better predictive accuracy and OOD
detection performance.
The established framework of WGBoost offers exciting avenues for future research. Important directions for
future study include (i) investigating the convergence properties, (ii) evaluating the robustness to misspecified
response distributions, and (iii) exploring alternative loss functionals to the KL divergence. A limitation of
WGBoost may arise when data are not tabular, as in the case of standard gradient boosting. These questions
require careful examination and are critical for future work.
References
[1] Ravid Shwartz-Ziv and Amitai Armon. Tabular data: Deep learning is not all you need. Information
Fusion, 81:84–90, 2022.
[2] Moloud Abdar, Farhad Pourpanah, Sadiq Hussain, Dana Rezazadegan, Li Liu, Mohammad Ghavamzadeh,
Paul Fieguth, Xiaochun Cao, Abbas Khosravi, U. Rajendra Acharya, Vladimir Makarenkov, and Saeid
Nahavandi. A review of uncertainty quantification in deep learning: Techniques, applications and
challenges. Information Fusion, 76:243–297, 2021.
[3] Eric Topol. High-performance medicine: the convergence of human and artificial intelligence. Nature
Medicine, 25:44–56, 2019.
[4] Sorin Grigorescu, Bogdan Trasnea, Tiberiu Cocias, and Gigel Macesanu. A survey of deep learning
techniques for autonomous driving. Journal of Field Robotics, 37(3):362–386, 2020.
[5] Matthew Richardson, Ewa Dominowska, and Robert Ragno. Predicting clicks: estimating the click-
through rate for new ads. In Proceedings of the 16th International Conference on World Wide Web, page
521–530, 2007.
[6] Christopher Burges. From ranknet to lambdarank to lambdamart: An overview. Learning, 11, 2010.
[7] Byron P. Roe, Hai-Jun Yang, Ji Zhu, Yong Liu, Ion Stancu, and Gordon McGregor. Boosted decision
trees as an alternative to artificial neural networks for particle identification. Nuclear Instruments
and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated
Equipment, 543(2):577–584, 2005.
[8] James Bennett and Stan Lanning. The Netflix prize. In Proceedings of the KDD Cup Workshop 2007,
pages 3–6, 2007.
[9] Yarin Gal and Zoubin Ghahramani.
Dropout as a Bayesian approximation: Representing model
uncertainty in deep learning. In Proceedings of The 33rd International Conference on Machine Learning,
volume 48 of Proceedings of Machine Learning Research, pages 1050–1059. PMLR, 2016.
[10] Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive
uncertainty estimation using deep ensembles. In Advances in Neural Information Processing Systems,
volume 30, 2017.
12


<!-- page 13 -->
[11] Murat Sensoy, Lance Kaplan, and Melih Kandemir. Evidential deep learning to quantify classification
uncertainty. In Advances in Neural Information Processing Systems, volume 31, 2018.
[12] Jakob Gawlikowski, Cedrique Rovile Njieutcheu Tassi, Mohsin Ali, Jongseok Lee, Matthias Humt,
Jianxiang Feng, Anna Kruspe, Rudolph Triebel, Peter Jung, Ribana Roscher, Muhammad Shahzad, Wen
Yang, Richard Bamler, and Xiao Xiang Zhu. A survey of uncertainty in deep neural networks. Artificial
Intelligence Review, 56:1513–1589, 2023.
[13] Alexander Amini, Wilko Schwarting, Ava Soleimany, and Daniela Rus. Deep evidential regression. In
Advances in Neural Information Processing Systems, volume 33, pages 14927–14937, 2020.
[14] Bertrand Charpentier, Daniel Zügner, and Stephan Günnemann.
Posterior network: Uncertainty
estimation without OOD samples via density-based pseudo-counts. In Advances in Neural Information
Processing Systems, volume 33, pages 1356–1367. Curran Associates, Inc., 2020.
[15] Dennis Thomas Ulmer, Christian Hardmeier, and Jes Frellsen. Prior and posterior networks: A survey
on evidential deep learning methods for uncertainty estimation. Transactions on Machine Learning
Research, 2023.
[16] Edouard Capellier, Franck Davoine, Veronique Cherfaoui, and You Li. Evidential deep learning for
arbitrary lidar object classification in the context of autonomous driving. In 2019 IEEE Intelligent
Vehicles Symposium (IV), pages 1304–1311, 2019.
[17] Patrick Hemmer, Niklas Kühl, and Jakob Schöffer. Deal: Deep evidential active learning for image
classification. In 2020 19th IEEE International Conference on Machine Learning and Applications
(ICMLA), pages 865–870, 2020.
[18] Ava P. Soleimany, Alexander Amini, Samuel Goldman, Daniela Rus, Sangeeta N. Bhatia, and Connor W.
Coley. Evidential deep learning for guided molecular property prediction and discovery. ACS Central
Science, 7(8):1356–1367, 2021.
[19] Jakob Gawlikowski, Sudipan Saha, Anna Kruspe, and Xiao Xiang Zhu. An advanced Dirichlet prior
network for out-of-distribution detection in remote sensing. IEEE Transactions on Geoscience and
Remote Sensing, 60:1–19, 2022.
[20] Andrew Gelman, John B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, and Donald B. Rubin.
Bayesian Data Analysis. Chapman and Hall/CRC, 3rd ed. edition, 2013.
[21] Tianqi Chen and Carlos Guestrin. XGBoost: A scalable tree boosting system. In Proceedings of the
22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’16,
page 785–794, 2016.
[22] Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan
Liu. LightGBM: A highly efficient gradient boosting decision tree. In Advances in Neural Information
Processing Systems, volume 30, 2017.
[23] C’edric Villani. Topics in Optimal Transportation. Americal Mathematical Society, 2003.
[24] Luigi Ambrosio, Nicola Gigli, and Giuseppe Savaré. Gradient Flows In Metric Spaces and in the Space
of Probability Measures. Birkhäuser Basel, 2005.
[25] Filippo Santambrogio. {Euclidean, metric, and Wasserstein} gradient flows: an overview. Bulletin of
Mathematical Sciences, 7:87–154, 2017.
[26] Qiang Liu. Stein variational gradient descent as gradient flow. In Advances in Neural Information
Processing Systems, volume 30, 2017.
[27] Jos’e Antonio Carrillo, Katy Craig, and Francesco S. Patacchini. A blob method for diffusion. Calculus
of Variations and Partial Differential Equations, 58(53), 2019.
13


<!-- page 14 -->
[28] Yifei Wang, Peng Chen, and Wuchen Li. Projected Wasserstein gradient descent for high-dimensional
Bayesian inference. SIAM/ASA Journal on Uncertainty Quantification, 10(4):1513–1532, 2022.
[29] Dimitra Maoutsa, Sebastian Reich, and Manfred Opper. Interacting particle solutions of Fokker-Planck
equations through gradient-log-density estimation. Entropy (Basel), 22(8):802, 2020.
[30] Ye He, Krishnakumar Balasubramanian, Bharath K. Sriperumbudur, and Jianfeng Lu. Regularized Stein
variational gradient flow. arXiv:2211.07861, 2022.
[31] Jerome H. Friedman. Greedy function approximation: A gradient boosting machine. The Annals of
Statistics, 29(5):1189–1232, 2001.
[32] Tony Duan, Avati Anand, Daisy Yi Ding, Khanh K. Thai, Sanjay Basu, Andrew Ng, and Alejandro
Schuler. NGBoost: Natural gradient boosting for probabilistic prediction. In Proceedings of the 37th
International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research,
pages 2690–2700. PMLR, 2020.
[33] Peter Bühlmann and Torsten Hothorn. Boosting Algorithms: Regularization, Prediction and Model
Fitting. Statistical Science, 22(4):477 – 505, 2007.
[34] Leo Grinsztajn, Edouard Oyallon, and Gael Varoquaux. Why do tree-based models still outperform deep
learning on typical tabular data? In Advances in Neural Information Processing Systems, volume 35,
pages 507–520, 2022.
[35] Piotr Florek and Adam Zagdański. Benchmarking state-of-the-art gradient boosting algorithms for
classification. arXiv:2305.17094, 2023.
[36] Zhendong Zhang and Cheolkon Jung. GBDT-MO: Gradient-boosted decision trees for multiple outputs.
IEEE Transactions on Neural Networks and Learning Systems, 32(7):3156–3167, 2021.
[37] Tianqi Chen, Sameer Singh, Ben Taskar, and Carlos Guestrin. Efficient Second-Order Gradient Boosting
for Conditional Random Fields. In Proceedings of the Eighteenth International Conference on Artificial
Intelligence and Statistics, volume 38 of Proceedings of Machine Learning Research, pages 147–155.
PMLR, 2015.
[38] Jerome H. Friedman. Stochastic gradient boosting. Computational Statistics & Data Analysis, 38(4):367–
378, 2002.
[39] Gianluca Detommaso, Tiangang Cui, Youssef Marzouk, Alessio Spantini, and Robert Scheichl. A Stein
variational Newton method. In Advances in Neural Information Processing Systems, volume 31, 2018.
[40] Yifei Wang and Wuchen Li. Information Newton’s flow: second-order optimization method in probability
space. arXiv:2001.04341, 2020.
[41] Malay Ghosh. Objective priors: An introduction for frequentists. Statistical Science, 26(2):187–202,
2011.
[42] J. Aitchison and S. M. Shen. Logistic-normal distributions: Some properties and uses. Biometrika,
67(2):261–272, 1980.
[43] Qiang Liu and Dilin Wang. Stein variational gradient descent: A general purpose Bayesian inference
algorithm. In Advances in Neural Information Processing Systems, volume 29, 2016.
[44] Dilin Wang, Zhe Zeng, and Qiang Liu. Stein variational message passing for continuous graphical models.
In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of
Machine Learning Research, pages 5219–5227. PMLR, 2018.
[45] Alexander Lambert, Fabio Ramos, Byron Boots, Dieter Fox, and Adam Fishman. Stein variational model
predictive control. In Proceedings of the 2020 Conference on Robot Learning, volume 155 of Proceedings
of Machine Learning Research, pages 1278–1297. PMLR, 2021.
14


<!-- page 15 -->
[46] Anna Korba, Adil Salim, Michael Arbel, Giulia Luise, and Arthur Gretton. A non-asymptotic analysis
for Stein variational gradient descent. In Advances in Neural Information Processing Systems, volume 33,
pages 4672–4682, 2020.
[47] Sinho Chewi, Thibaut Le Gouic, Chen Lu, Tyler Maunu, and Philippe Rigollet. SVGD as a kernelized
Wasserstein gradient flow of the chi-squared divergence. In Advances in Neural Information Processing
Systems, volume 33, pages 2098–2109, 2020.
[48] Andre Wibisono. Sampling as optimization in the space of measures: The Langevin dynamics as a
composite optimization problem. In Proceedings of the 31st Conference On Learning Theory, volume 75
of Proceedings of Machine Learning Research, pages 2093–3027. PMLR, 2018.
[49] Gareth O. Roberts and Richard L. Tweedie. Exponential convergence of Langevin distributions and
their discrete approximations. Bernoulli, 2(4):341 – 363, 1996.
[50] Jose Miguel Hernandez-Lobato and Ryan Adams. Probabilistic backpropagation for scalable learning of
Bayesian neural networks. In Proceedings of the 32nd International Conference on Machine Learning,
volume 37 of Proceedings of Machine Learning Research, pages 1861–1869. PMLR, 2015.
[51] Leo Breiman, Jerome Friedman, R.A. Olshen, and Charles J. Stone. Classification and Regression Trees.
Chapman and Hall/CRC, 1984.
[52] Trevor Hastie, Robert Tibshirani, and Jerome Friedman. The Elements of Statistical Learning. Springer
New York, 2009.
[53] Sanford Weisberg. Applied Linear Regression. John Wiley & Sons, 1985.
[54] Dheeru Dua and Casey Graff. UCI machine learning repository, 2017.
[55] Yarin Gal, Jiri Hron, and Alex Kendall. Concrete dropout. In Advances in Neural Information Processing
Systems, volume 30, 2017.
[56] Jingkang Yang, Kaiyang Zhou, Yixuan Li, and Ziwei Liu. Generalized out-of-distribution detection: A
survey. arXiv:2110.11334, 2024.
[57] Andrey Malinin, Bruno Mlodozeniec, and Mark Gales. Ensemble distribution distillation. In International
Conference on Learning Representations, 2020.
[58] Filippo Santambrogio. Optimal Transport for Applied Mathematicians. Birkhäuser Cham, 2015.
[59] Mingxuan Yi and Song Liu. Bridging the gap between variational inference and Wasserstein gradient
flows, 2023.
[60] Michael Arbel, Anna Korba, Adil Salim, and Arthur Gretton. Maximum mean discrepancy gradient
flow. In Advances in Neural Information Processing Systems, volume 32, 2019.
[61] Richard Jordan, David Kinderlehrer, and Felix Otto. The variational formulation of the Fokker–Planck
equation. SIAM Journal on Mathematical Analysis, 29(1):1–17, 1998.
[62] Grigorios A. Pavliotis. Stochastic Processes and Applications. Springer New York, 2014.
[63] Vern I. Paulsen and Mrinal Raghupathi. An Introduction to the Theory of Reproducing Kernel Hilbert
Spaces. Cambridge University Press, 2016.
[64] Alex Leviyev, Joshua Chen, Yifei Wang, Omar Ghattas, and Aaron Zimmerman. A stochastic Stein
variational Newton method. arXiv:2204.09039, 2022.
[65] Alex Smola, Arthur Gretton, Le Song, and Bernhard Schölkopf.
A Hilbert space embedding for
distributions.
[66] Shun ichi Amari. Information Geometry and Its Applications. Springer Tokyo, 2016.
15


<!-- page 16 -->
Appendix
This appendix contains the technical and experiment details referred to in the main text. Appendix A recaps
the derivation of the Wasserstein gradient and presents several examples. Appendix B discusses a variant
of WGBoost for the KL divergence built on the unadjusted Langevin algorithm. Appendix C derives the
diagonal approximate Wasserstein Newton direction used for WEvidential. Appendix D provides a simulation
study to compare four different WGBoost algorithms. Appendix E describes the additional details of the
experiment in the main text.
A
Derivation and Example of Wasserstein Gradient
This section recaps the derivation of the Wasserstein gradient of a functional F, with examples of common
divergences. The Wasserstein gradient depends on a function on Θ called the first variation [24]. The first
variation δF(µ)/δµ of the functional F at µ is a function on Θ that satisfies
lim
ϵ→0+
F(µ + ϵν) −F(µ)
ϵ
=
Z
Θ
δF(µ)
δµ
(θ)ν(θ)dθ
for all signed measure ν s.t. µ + ϵν ∈P2 for all ϵ sufficiently small. The Wasserstein gradient ∇W F(µ) of the
functional F at µ is derived as the gradient of the first variation (see [e.g. 24]):
[∇W F(µ)](θ) := ∇δF(µ)
δµ
(θ).
It is common to suppose that the functional F consists of three energies, which are determined by functions
U : R →R, V : Θ →R, and W : Θ →R respectively, such that
F(µ) =
Z
Θ
U(µ(θ))dθ
|
{z
}
internal energy
+
Z
Θ
V (θ)µ(θ)dθ
|
{z
}
potential energy
+ 1
2
Z
Θ×Θ
W(θ −θ′)µ(θ)dθµ(θ′)dθ′
|
{z
}
interaction energy
.
For a functional F that falls into the above form, the Wasserstein gradient is derived as
[∇W F(µ)] (θ) = ∇U ′(µ(θ)) + ∇V (θ) +
Z
Θ
∇W(θ −θ′)µ(θ′)dθ′
where U ′ is the derivative of U : R →R [23]. The KL divergence F(µ) = KL(µ | π) of a distribution π falls
into the form with U(x) = x log x, V (θ) = −log π(θ), and W(θ) = 0, where
KL(µ | π) =
Z
Θ
log µ(θ)µ(θ)dθ +
Z
Θ
−log π(θ)µ(θ)dθ.
Table 3 presents examples of Wasserstein gradients of common divergences F(µ) = D(µ | π).
In the context of Bayesian inference, the KL divergence is particularly useful among many divergences. The
Wasserstein gradient of the KL divergence requires no normalising constant of a posterior distribution π. This
is because the Wasserstien gradient depends only on the log-gradient of the posterior ∇log π(θ) = ∇π(θ)/π(θ)
of the target π, in which case the normalising constant of the target π is cancelled out by fraction. Hence,
any posterior known only up to the normalising constant can be used as the target distribution π in the
Wasserstein gradient of the KL divergence.
16

[CAPTION] Table 3 presents examples of Wasserstein gradients of common divergences F(µ) = D(µ | π).


<!-- page 17 -->
Table 3: Wasserstein gradients of four divergences: the KL divergence [58], the chi-squared divergence [47],
the alpha divergence [59], and the maximum mean discrepancy [60].
Divergence F(µ) = D(µ | π)
Wasserstein gradient [∇W F(µ)] (θ)
KL(µ | π)
−(∇log π(θ) −∇log µ(θ))
Chi2(µ | π)
2∇(µ(θ)/π(θ))
Alpha(µ | π)
(µ(θ)/π(θ))α−1∇(µ(θ)/π(θ))
MMD(µ | π)
R
Θ ∇k(θ, θ′)µ(θ)dθ −
R
Θ ∇k(θ, θ′)π(θ)dθ
B
Langevin Gradient Boosting for KL Divergence
If a chosen functional F on P2 is the KL divergence F(µ) = KL(µ | π) of a target distribution π, the
continuity equation (1) admits an equivalent representation as the Fokker-Planck equation [61]:
d
dtµt = ∇· (µt∇log π) + ∆µt
given
µ0 ∈P2
(10)
where ∆denotes the Laplacian operator. Recall that the original continuity equation (1) can be reformulated
as the deterministic differential equation (2) of a random variable θt ∼µt. In contrast, the Fokker-Planck
equation (10) can be reformulated as a stochastic differential equation of a random variable θt ∼µt, known
as the overdamped Langevin dynamics [62]:
dθt = ∇log π(θt)dt +
√
2dBt
given
θ0 ∼µ0,
(11)
where Bt denotes a standard Brownian motion. Note that the deterministic system (2) in the case of the KL
divergence and the above stochastic system (11) are equivalent at population level, in a sense that the law of
the random variable θt in both the systems solves the two equivalent equations.
At the algorithmic level, however, discretisation of each system leads to different particle update schemes. Set
the initial distribution µ0 in (11) to the empirical distribution ˆµ0 of N initial particles {θn
0 }N
n=1. Discretising
the stochastic system (11) by the Euler-Maruyama method with a step size ν > 0 yields a stochastic update
scheme of particles {θn
m}N
n=1 from step m = 0:


θ1
m+1
...
θN
m+1

=


θ1
m
...
θN
m

+ ν


∇log π(θ1
m) +
p
2/ν ξ1
...
∇log π(θN
m) +
p
2/ν ξN

,
where each ξn denotes a realisation from a standard normal distribution on Rd. The above updating scheme of
each n-th particle is known as the unadjusted Langevin algorithm [49]. We can define a variant of WGBoost
by replacing the term Gi(µ) in Algorithm 1 with ∇log µi(·) +
p
2/ν ξi where µi is an output distribution
at each xi and ξi is a realisation from a standard normal distribution. The procedure is summarised in
Algorithm 3, which we call Langevin gradient boosting (LGBoost).
C
Derivation of Approximate Wasserstein Newton Direction
This section derives the diagonal approximate Wasserstein Newton direction based on the kernel smoothing.
The approximate Wasserstein Newton direction of the KL divergence was derived in [39] under a different
terminology—simply, the Newton direction—from a viewpoint of nonparametric variational inference. We
place their result in the context of approximate Wasserstein gradient flows. Appendix C.1 shows the derivation
of the smoothed Wasserstein gradient and Hessian. Appendix C.2 defines the Newton direction built upon
the smoothed Wasserstein gradient and Hessian, following the derivation in [39]. Appendix C.3 derives the
diagonal approximation of the Newton direction.
17

[CAPTION] Table 3: Wasserstein gradients of four divergences: the KL divergence [58], the chi-squared divergence [47],


<!-- page 18 -->
Algorithm 3: Langevin Gradient Boosting
Input: training set {xi, µi}D
i=1 of input xi ∈X and output distribution µi ∈P2
Parameter : particle number N, iteration M, rate ν, weak learner f, initial constants (ϑ1
0, . . . , ϑN
0 )
Output: set of N boosting ensembles (F 1
M, . . . , F N
M) at final step M
(F 1
0 (·), . . . , F N
0 (·)) ←(ϑ1
0, . . . , ϑN
0 )
for m ←0, . . . , M −1 do
for n ←1, . . . , N do
for i ←1, . . . , D do
gn
i ←∇log µi(F n
m(xi)) +
p
2/ν ξn
i
where
ξn
i ∼N(0, Id)
end
f n
m+1 ←fit
 
{xi, gn
i }D
i=1
 
F n
m+1(·) ←F n
m(·) + νf n
m+1(·)
end
end
C.1
Smoothed Wasserstein Gradient and Hessian
Consider the one-dimensional case Θ = R for simplicity. For a map T : R →R and a distribution µ ∈P2,
let µt be the pushforward of µ under the transform θ 7→θ + tT(θ) defined with a time-variable t ∈R. This
means that µt is a distribution obtained by change-of-variable applied for µ. The Wasserstein gradient of a
functional F(µ) can be associated with the time derivative (d/dt)F(µt) [23]. In what follows, we focus on the
KL divergence F(µ) = KL(µ | π) as a loss functional. Under a condition T ∈L2(µ), the time derivative at
t = 0 satisfies the following equality
d
dt KL(µt | π)
   
t=0 =
Z
Θ
T(θ)
 
GKL(µ)
 
(θ) dµ(θ) =

T, GKL(µ)
 
L2(µ) ,
(12)
where GKL(µ) denotes the Wasserstein gradient of F(µ) = KL(µ | π) with the target distribution π made
implicit. It gives an interpretation of the Wasserstein gradient as the steepest-descent direction because the
decay of the KL divergence at t = 0 is maximised when T = −GKL(µ).
The ‘smoothed’ Wasserstein gradient can be derived by restricting the transform map T to a more regulated
Hilbert space than L2(µ). A reproducing kernel Hilbert space (RKHS) H associated with a kernel function
k : R × R →R is the most common choice of such a Hilbert space [e.g. 26]. An important property of
the RKHS H is that any function f ∈H satisfies the reproducing property f(θ) = ⟨f(·), k(·, θ)⟩H under
the associated kernel k and inner product ⟨·, ·⟩H [63]. As discussed in [e.g. 46], applying the reproducing
property in (12) under the condition T ∈H and exchanging the integral order, the time derivative satisfies
an alternative equality as follows:
d
dt KL(µt | π)
   
t=0 =
Z
Θ
⟨T(·), k(·, θ)⟩H
 
GKL(µ)
 
(θ) dµ(θ)
=
 
T(·),
Z
Θ
 
GKL(µ)
 
(θ)k(·, θ)dµ(θ)
 
H
= ⟨T, G∗(µ)⟩H
(13)
where [G∗(µ)](·) :=
R
Θ[GKL(µ)](θ)k(·, θ)dµ(θ) corresponds to the smoothed Wasserstein gradient used in the
main text. The decay of the KL divergence at t = 0 is maximised by T = −G∗(µ).
Similarly, the Wasserstein Hessian of the functional F(µ) can be associated with the second time derivative
(d2/dt2)F(µt) [23]. As discussed in [e.g. 46], the Wasserstein Hessian of the KL divergence, denoted Hess(µ),
is an operator over functions T ∈L2(µ) that satisfies
d2
dt2 KL(µt | π)
   
t=0 = ⟨T, Hess(µ)T⟩L2(µ) .
(14)
18


<!-- page 19 -->
See [46] for the explicit form of the Wasserstein Hessian. In the same manner as the smoothed Wasserstein
gradient, applying the reproducing property in (14) under the condition T ∈H and exchanging the integral
order, the second time derivative satisfies an alternative equality as follows:
d2
dt2 KL(µt | π)
   
t=0 =
D
T(⋆1),

[Hess∗(µ)] (⋆1, ⋆2), T(⋆2)
 
H
E
H
(15)
where [Hess∗(µ)](⋆1, ⋆2) := ⟨k(⋆1, ·), Hess(µ)k(⋆2, ·)⟩L2(µ) is the smoothed Wasserstein Hessian and the
symbols ⋆1 and ⋆2 denote the variables to which each of the two inner products is taken.
In the multidimensional case Θ = Rd, the transport map T is a vector-valued function T : Rd →Rd, where a
similar derivation can be repeated by replacing L2(µ) and H with the product space of d independent copies
of L2(µ) and H. It follows from Proposition 1 and Theorem 1 in [39]—which derives the explicit form of
(13) and (15) under their terminology, first and second variations—that the explicit form of the smoothed
Wasserstein gradient and Hessian is given by
[G∗(µ)] (·) = Eθ∼µ
h
−∇log π(θ)k(·, θ) −∇k(·, θ)
i
∈Rd,
[Hess∗(µ)] (⋆1, ⋆2) = Eθ∼µ
h
−∇2 log π(θ)k(⋆1, θ)k(⋆2, θ) + ∇k(⋆1, θ) ⊗∇k(⋆2, θ)
i
∈Rd×d
where ∇2 denotes an operator to take the Jacobian of the gradient—i.e., ∇2f(θ) is the Hessian matrix of f
at θ—and ⊗denotes the outer product of two vectors. Note that both the smoothed Wasserstein gradient
and Hessian are well-defined for any distribution µ including empirical distributions.
C.2
Approximate Wasserstein Newton Direction
In the Euclidean space, the Newton direction of an objective function is a direction s.t. the second-order Taylor
approximation of the function is minimised. Similarly, [39] characterised the Newton direction T ∗: Rd →Rd
of the KL divergence KL(µ | π) as a solution of the following equation
D
[Hess∗(µ)](⋆1, ⋆2), T ∗(⋆2)
 
H + [G∗(µ)](⋆1), V (⋆1)
E
H = 0
for all
V ∈H.
Here Θ = Rd and H is the product space of d independent copies of the RKHS of a kernel k. To obtain a closed-
form solution, [39] supposed that the Newton direction T ∗can be expressed in a form T ∗(·) = Pn
i=1 W nk(·, θn)
dependent on a set of each particle θn ∈Θ and associated vector-valued coefficient W n ∈Rd. Once the set of
the particles is given, the set of the associated vector-valued coefficients is determined by solving the following
simultaneous linear equation


PN
n=1[Hess∗(µ)](θ1, θn) · W n
...
PN
n=1[Hess∗(µ)](θN, θn) · W n

=


−[G∗(µ)](θ1)
...
−[G∗(µ)](θN)

.
(16)
These equations (16) can be rewritten in a matrix form [64]. Let K := N × d. Define a block matrix
H ∈RK×K and a block vector G ∈RK by the following partitioning
H =



H11
· · ·
H1N
...
...
...
HN1
· · ·
HNN



and
G =



G1
...
GN



with each block specified as Hij := [Hess∗(µ)](θi, θj) ∈Rd×d and Gi := [G∗(µ)](θi) ∈Rd. Define a block
matrix K ∈RK×K and a block vector W ∈RK by the following partitioning
K :=



K11
· · ·
K1N
...
...
...
KN1
· · ·
KNN



and
W :=



W 1
...
W N



19


<!-- page 20 -->
with each block of K specified as Kij := Id × k(θi, θj) ∈Rd×d, where Id denotes the d × d identity matrix.
Notice that W is a block vector that aligns the vector-valued coefficients {W n}N
n=1. Using these notations,
the optimal coefficients that solve (16) is simply written as W = −H−1G [64].
Given the optimal coefficients W = −H−1G, the Newton direction T ∗(θn) evaluated at the given particle θn
for each n = 1, . . . , N can be written in the following block vector form



T ∗(θ1)
...
T ∗(θN)


= −



K11
· · ·
K1N
...
...
...
KN1
· · ·
KNN






H11
· · ·
H1N
...
...
...
HN1
· · ·
HNN



−1 


G1
...
GN



(17)
To distinguish from the standard Newton direction in the Euclidean space, we call (17) the approximate
Wasserstein Newton direction. The approximate Wasserstein Newton direction yields a second-order particle
update scheme. Suppose we have particles {θn
m}N
n=1 to be updated at each step m. At each step m, define
the above matrices H and G with the empirical distribution µ = ˆπm of the particles {θn
m}N
n=1. Replacing the
Wasserstein gradient in the particle update scheme (3) by the approximate Wasserstein Newton direction
(17) provides the second-order update scheme in [39].
C.3
Diagonal Approximate Wasserstein Newton Direction
We derive the diagonal approximation of the approximate Wasserstein Newton direction, which we used
for our second-order WGBoost algorithm. A few approximations of the approximate Wasserstein Newton
direction were discussed in [39] for better performance of their particle algorithm. We derive the diagonal
approximation so that no matrix product and inversion will be involved. Specifically, we replace the matrices
K and H in (17) by the diagonal approximations ˆK and ˆH, that is,
ˆK =



Id
· · ·
0
...
...
...
0
· · ·
Id



and
ˆH =



h11
· · ·
0
...
...
...
0
· · ·
hNN


,
where Knn = Id × k(θn, θn) = Id for the Gaussian kernel k used in this work, and the matrix hnn ∈Rd×d
denotes the diagonal approximation of the diagonal block Hnn of H.
Recall that Hnn = [Hess∗(µ)](θn, θn). Denote by Diag(A) the diagonal of a square matrix A. The diagonal
approximation hnn is a diagonal matrix whose diagonal is Diag(Hnn). We plug the diagonal approximations ˆK
and ˆH in (17). It follows from inverse and multiplication properties of diagonal matrices that the approximate
Wasserstein Newton direction turns into a form



T ∗(θ1)
...
T ∗(θN)


= −



h11
· · ·
0
...
...
...
0
· · ·
hNN



−1 


G1
...
GN


=



−G1 ⊘Diag (H11)
...
−GN ⊘Diag (HNN)


.
(18)
At an arbitrary particle location θ, denote by [H∗(µ)](θ) the diagonal of the smoothed Wasserstein Hessian
[Hess∗(µ)](θ, θ). It is straightforward to see that the diagonal can be written as
[H∗(µ)] (·) = Eθ∼µ
h
−∇2
d log π(θ)k(·, θ)2 + ∇k(·, θ) ⊙∇k(·, θ)
i
.
Notice that Diag (Hnn) = [H∗(µ)] (θn) by definition.
It therefore follows from the formula (18) with
Gn = [G∗(µ)](θn) and Diag (Hnn) = [H∗(µ)] (θn) that the diagonal approximate Wasserstein Newton
direction at an arbitrary particle location θ can be independently computed by
−[G∗(µ)](θ) ⊘[H∗(µ)] (θ).
We used this direction in Section 3. In the main text, the diagonal approximate Wasserstein Newton direction
is defined for each loss functional Fi(·) = D(· | µi), with π = µi, using the smoothed Wasserstein gradient
G∗
i (µ) and the diagonal of the smoothed Wasserstein Hessian H∗
i (µ) for each i-th output distribution µi.
20


<!-- page 21 -->
D
Comparison of Different WGBoost Algorithms
We compare four different algorithms of WGBoost for the KL divergence through a simulation study. The
first three algorithms are defined by setting the term Gi(µ) in Algorithm 1 to, respectively,
1. the smoothed Wasserstein gradient in (7);
2. the diagonal approximate Wasserstein Newton direction in (8);
3. the full approximate Wasserstein Newton direction in (17).
The fourth algorithm, which is rather a variant of WGBoost, is LGBoost in Appendix B. The first and
third WGBoost algorithms is called, respectively, first-order WEvidential and full-Newton WEvidential. The
second WGBoost algorithm is WEvidential presented in Section 3. We fit the four algorithms to a synthetic
dataset {xi, µi}D
i=1 whose inputs are 200 gird points on the interval [−3.5, 3.5] and output distributions are
normal distributions µi(θ) = N(θ | sin(xi), 0.5) conditional on each xi.
The first-order WEvidential is implemented by Algorithm 2 removing hn
i and replacing gn
i ⊘hn
i with gn
i .
The full-Newton WEvidential is implemented by Algorithm 2 replacing gn
i ⊘hn
i with vn
i computed by the
following Algorithm 4, where ∇2f(θ) denotes the Hessian matrix of a function f : Θ →R at θ.
Algorithm 4: Computation of Approximate Wasserstein Newton Direction
Input: input xi, output distribution µi, outputs of N boostings (F 1
m(xi), . . . , F N
m (xi)) for input xi
Output: Wasserstein Newton direction (v1
i , . . . , vN
i ) evaluated at (F 1
m(xi), . . . , F N
m (xi)) for input xi
ˆµm,i ←empirical distribution of set of N outputs (F 1
m(xi), . . . , F N
m (xi)) for input xi
for n ←1, . . . , N do
gn
i ←Eθ∗∼ˆµm,i[∇log µ(θ∗)k(F n
m(xi), θ∗) + ∇k(F n
m(xi), θ∗)]
for k ←1, . . . , N do
Hnk
i
←Eθ∗∼ˆµm,i[−∇2 log µi(θ∗)k(F n
m(xi), θ∗)k(F k
m(xi), θ∗) + ∇k(F n
m(xi), θ∗) ⊗∇k(F k
m(xi), θ∗)]
Knk
i
←Id · k(F n
m(xi), F k
m(xi))
end
end



v1
i
...
vN
i


←



K11
i
· · ·
K1N
i
...
...
...
KN1
i
· · ·
KNN
i






H11
i
· · ·
H1N
i
...
...
...
HN1
i
· · ·
HNN
i



−1 


g1
i
...
gN
i



Figure 5 shows the performance and computational time of each algorithm on the synthetic data with respect
to the number of weak learners. We computed the output of each algorithm for 500 grid points in the
interval [−3.5, 3.5]. We used the maximum mean discrepancy (MMD) [65] to measure the approximation
error between the empirical distribution ˆµi of the output particles and the output distribution µi at each
input xi:
MMD2 (ˆµi, µi) = Eθ∼ˆµi,θ′∼ˆµi[k(θ, θ′)] −2Eθ∼ˆµi,θ′∼µi[k(θ, θ′)] + Eθ∼µi,θ′∼µi[k(θ, θ′)]
where k is a Gaussian kernel k(θ, θ′) = exp(−(θ −θ′)2/h) with scale hyperparameter h = 0.025. The total
approximation error was measured by the MMD averaged over all the inputs. We set the initial constant
{ϑn}10
n=1 of each algorithm to 10 grid points in the interval [−10, 10], which sufficiently differs from the output
distributions to observe the decay of the approximation error. The decision tree regressor with maximum
depth 3 was used as weak learners for all the algorithm.
Figure 5 demonstrates that WEvidential and full-Newton WEvidential reduce the approximation error
most efficiently, while full-Newton WEvidential takes the longest computational time among others. As in
Algorithm 4, the computation of the full approximate Wasserstein Newton direction requires the inverse and
21


**[Table p21.1]**
|  v1   K11 i i . .  . . ← . .    vN KN1 i i | · · · | K1N  H11 i i . . . .  . .  KNN HN1 i i | · · · | H1N −1  g1  i i . . . .   . .     HNN gN i i |
| --- | --- | --- | --- | --- |
|  | ... |  | ... |  |
|  | · · · |  | · · · |  |

[CAPTION] Figure 5 shows the performance and computational time of each algorithm on the synthetic data with respect

[CAPTION] Figure 5 demonstrates that WEvidential and full-Newton WEvidential reduce the approximation error


<!-- page 22 -->
Figure 5: The approximation error and computational time of the four algorithms. Left: approximation error
of each algorithm measured by the MMD averaged over the inputs. Right: computational time with respect
to the weak learner number in common logarithm scale.
product of the (N × d) × (N × d) block matrices, where N denotes the particle number N and d denotes the
particle dimension. The computation of the diagonal approximate Wasserstein Newton direction requires only
elementwise division of the d-dimensional vectors. The error decay of LGBoost is not as fast as the other
WGBoost algorithms, and also shows stochasticity due to the Gaussian noise used in the algorithm. We
therefore recommend to use WEvidential for better performance and efficient computation. Figure 6 depicts
the output of each algorithm with 100 weak learners trained.
(a) First-order WEvidential
(b) WEvidential
(c) Full-Newton WEvidential
(d) LGBoost
Figure 6: Illustration of the output of the four algorithms (red line) with 100 weak learners trained. The blue
area corresponds to the 95% high probability region of each output distribution.
E
Additional Detail of Application
This section describes additional details of the applications in Section 4. All the experiments were performed
with x86-64 CPUs, where some of them were parallelised up to 10 CPUs and the rest uses 1 CPU. The scripts
to reproduce all the experiments are provided in the source code. Appendices E.1 to E.3 describe additional
details of the applications in, respectively, Sections 4.1 to 4.3. Appendix E.4 describes a choice of initial
constants {ϑn
0}N
n=1 for the WGBoost algorithm used in Section 4.
E.1
Detail of Section 4.1
The normal response distribution N(y | m, σ) in Example 1 was used in Section 4.1. The normal output
distribution has the scale parameter σ that lies only in the positive domain of the Euclidean space R.
We reparametrised it as one in the Euclidean space R by the log transform σ′ := log(σ), which is the
standard practice in Bayesian computation [20]. The inverse of the log transform is the exponential transform
σ = exp(σ′). It follows from change of variable that the individual-level posterior on (m, σ′) conditional on
22

[CAPTION] Figure 5: The approximation error and computational time of the four algorithms. Left: approximation error

[CAPTION] Figure 6: Illustration of the output of the four algorithms (red line) with 100 weak learners trained. The blue


<!-- page 23 -->
each individual observed response yi, with the prior in Example 1, is given by
µi(m, σ′) = p(m, σ′ | yi) ∝exp
 
−1
2
(yi −m)2
exp(σ′)2
 
× exp
 
−1
2
m2
102
 
×
1
exp(σ′)1.01 exp
 
−
0.01
exp(σ′)
 
up to the normalising constant, where we used the Jacobian determinant |dσ/dσ′| = exp(σ′).
E.2
Detail of Section 4.2
The same reparametrisation of the normal output distribution as Appendix E.2 was used in Section 4.2. For
test data {xi, yi}D
i=1, the NLL and RMSE of each algorithm were computed by
NLL = −1
D
D
X
i=1
log p(yi | xi)
and
RMSE =
v
u
u
t 1
D
D
X
i=1
(yi −ˆyi)2
for the obtained provided predictive distribution p(yi | xi) and the point prediction ˆyi. For WEvidential, the
observed responses in training data were standardised. Accordingly, the test responses were standardised as
y′
i = (yi −ytrain
mean)/ytrain
std
using the mean ytrain
mean and standard deviation ytrain
std
of the training responses. The
predictive distribution p(y′
i | xi) and point prediction ˆy′
i of WEvidential were provided for the standardised
responses y′
i. The NLL and RMSE for the original responses yi can be computed as follows:
NLL = −1
D
D
X
i=1
log p(yi | xi) = −1
D
D
X
i=1
log p(y′
i | xi) + log ytrain
std ,
RMSE =
v
u
u
t 1
D
D
X
i=1
(yi −(ytrain
mean + ytrain
std
× ˆy′
i))2 = ytrain
std
v
u
u
t 1
D
D
X
i=1
(y′
i −ˆy′
i)2
where the equality of the NLL follows from change of variable p(yi | xi) = p(y′
i | xi)/ytrain
std
and the equality of
the RMSE follows from rearranging the terms.
We provide a brief description of each algorithm used for the comparison. For each algorithm, the normal
response distribution p(y | m, σ) was specified for the response variable and the algorithm produces a point
or distributional estimate of the response parameter (m, σ) at each input x.
• MCDropout [9] trains a single neural network F while dropping out each network weight with some
Bernoulli probability. It can be interpreted as a variational approximation of a Bayesian neural network.
It generates multiple subnetworks {F n}N
n=1 by subsampling the network weights by the dropout. The
predictive distribution p(y | x) is given by the model averaging (1/N) PN
i=1 p(y | (m, σ) = F n(x)) for
each input x.
• DEnsemble [10] simply trains independent copies {F n}N
n=1 of a neural network F in parallel. It is one
of the mainstream approaches to uncertainty quantification based on deep learning. The predictive
distribution is given by the model averaging as in MCDropout.
• CDropout [55] consider a continuous relaxation of the Bernoulli random variable used in MCDropout
to optimise the Bernoulli probability of the dropout. It generates multiple subnetworks {F n}N
n=1
by subsampling the network weights by the dropout with the optimised probability. The predictive
distribution is the same as MCDropout.
• NGBoosting [32] is a family of gradient booting that use the natural gradient [66] of the response
distribution as a target variable of each weak learner. In contrast to other methods, NGBoost outputs
a single value F(x) to be plugged into the response-distribution parameter. The predictive distribution
p(y | x) is given by p(y | (m, σ) = F(x)) for each input x.
23


<!-- page 24 -->
• DEvidential [13] extends deep evidential learning [11], originally proposed in classification settings, to
regression settings. It considers the case where the individual-level posterior of the response distribution
falls into a conjugate parametric form, and predicts the hyperparameter of the individual-level posterior
by a neural network. The predictive distribution is also given in a conjugate closed-form.
E.3
Detail of Section 4.3
Similarly to the normal response distribution used in Sections 4.1 and 4.2, we reparametrised the parameter
of the categorical response distribution used in Section 4.3. The categorical response distribution C(y | q)
in Example 2 has a class probability parameter q = [q1, . . . , qk] in the simplex ∆k. We reparametrised
the parameter q by the log-ratio transform q′ := [log(q1/qk), . . . , log(qk−1/qk)] ∈Rk−1 that maps from the
simplex ∆k to the Euclidean space Rk−1 [42]. The inverse is the logistic transform
q =
 exp(q′
1)
zk
, . . . , exp(q′
k−1)
zk
, 1
zk
 
∈∆k
where
zk = 1 +
k−1
X
j=1
exp(q′
j).
The logistic normal distribution on the original parameter q corresponds to a normal distribution on the
transformed parameter q′ by change of variable [42]. By change of variable, the individual-level posterior on
q′ conditional on each individual observed response yi, with the prior in Example 2, is
µi(q′) = p(q′ | yi) ∝
  1
zk
 [yi=k]
×
k−1
Y
j=1
 exp(q′
j)
zk
 [yi=j]
× exp
 
−1
2
∥q′∥2
102
 
up to the normalising constant, where [yi = j] is 1 if yi is the j-th class label and 0 otherwise.
We provide a brief description of each algorithm used in comparison with WGBoost. MCDropout and
DEnsemble are described in Appendix E.2.
• DDistillation [57] predicts the parameter of a Dirichlet distribution over the simplex ∆k by a neural
network using the output of DEnsemble. The output of multiple networks in DEnsemble is distilled
into the a Dirichlet distribution controlled by one single network.
• PNetwork [14] considers the case where the individual-level posterior of the categorical response
distribution falls into a Dirichlet distribution similarly to deep evidential learning [11]. It predicts the
hyperparameter of the individual-level posterior given in the form of the Dirichlet distribution.
E.4
Choice of Initial State of WGBoost
In standard gradient boosting, the initial state at step m = 0 is specified by a constant that most fits given
data. Similarly, we specified the initial state {ϑn
0}N
n=1 of WEvidential by a set of constants that most fits the
output distributions in average. We find such a set of constants by performing an approximate Wasserstein
gradient flow averaged over all the output distributions. Specifically, given the term Gi(µ) in Algorithm 1, we
define ¯G(µ) := (1/D) PD
i=1 Gi(µ) and perform the update scheme of a set of N particles {¯ϑn
m}N
n=1:


¯ϑ1
m+1
...
¯ϑN
m+1

=


¯ϑ1
m
...
¯ϑN
m

+ ν0


−[ ¯G(ˆπm)](¯ϑ1
m)
...
−[ ¯G(ˆπm)](¯ϑN
m)


with the learning rate ν0 = 0.01 up to the maximum step number m = 5000. The initial particle locations for
this update scheme were sampled from a standard normal distribution. We specified the initial state {ϑn
0}N
n=1
by the set of particles {¯ϑn
m}N
n=1 obtained though this scheme at m = 5000.
24