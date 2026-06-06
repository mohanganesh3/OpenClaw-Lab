<!-- page 1 -->
Data Space Inversion for Efficient Predictions and Uncertainty
Quantification for Geothermal Models
Alex de Beer∗1, Andrew Power2, Daniel Wong2, Ken Dekkers2, Michael Gravatt2,
Elvar K. Bjarkason3, John P. O’Sullivan2, Michael J. O’Sullivan2, Oliver J. Maclaren2, and
Ruanui Nicholson2
1School of Mathematics and Statistics, University of Sydney, New South Wales 2006, Australia
2Department of Engineering Science and Biomedical Engineering, University of Auckland,
Auckland 1010, New Zealand
3Graduate School of International Resource Sciences, Akita University, Akita 010-8502, Japan
ABSTRACT
The ability to make accurate predictions with quantified uncertainty provides a cru-
cial foundation for the successful management of a geothermal reservoir.
Conventional
approaches for making predictions using geothermal reservoir models involve estimating
unknown model parameters using field data, then propagating the uncertainty in these es-
timates through to the predictive quantities of interest. However, the unknown parameters
are not always of direct interest; instead, the predictions are of primary importance. Data
space inversion (DSI) is an alternative methodology that allows for the efficient estimation
of predictive quantities of interest, with quantified uncertainty, that avoids the need to
estimate model parameters entirely. In this paper, we illustrate the applicability of DSI to
geothermal reservoir modelling. We first review the processes of model calibration, predic-
tion and uncertainty quantification from a Bayesian perspective, and introduce data space
inversion as a simple, efficient technique for approximating the posterior predictive distri-
bution. We then introduce a modification of the typical DSI algorithm that allows us to
sample directly and efficiently from the DSI approximation to the posterior predictive distri-
bution, and apply the algorithm to two model problems in geothermal reservoir modelling.
We evaluate the accuracy and efficiency of our DSI algorithm relative to other common
methods for uncertainty quantification and study how the number of reservoir model sim-
ulations affects the resulting approximation to the posterior predictive distribution. Our
results demonstrate that data space inversion is a robust and efficient technique for mak-
ing predictions with quantified uncertainty using geothermal reservoir models, providing a
useful alternative to more conventional approaches.
1
INTRODUCTION
Computational models are widely used in geothermal reservoir engineering to facilitate effective
decision making (O’Sullivan and O’Sullivan, 2016). One of the key features of these models is
the ability to make predictions with quantified uncertainty. Computing accurate predictions and
uncertainty estimates generally requires calibration of the model; that is, the estimation of model
parameters, such as the subsurface permeability structure and the strength and location of the
deep mass upflows at the base of the system, using observations such as downhole temperature
and pressure measurements. In many situations, the parameters themselves are not of direct
∗Corresponding author. Email: adeb0907@uni.sydney.edu.au.
1
arXiv:2407.15401v3  [stat.AP]  2 Feb 2025


<!-- page 2 -->
interest; instead, the predictions are of primary importance. However, the calibration process
is typically the most computationally demanding step in the process of making predictions.
Here we discuss the application of the data space inversion (DSI) methodology (Sun and
Durlofsky, 2017; Sun et al., 2017) for making predictions using geothermal reservoir models,
as well as providing associated estimates of uncertainty. The DSI framework provides several
computational advantages, the most significant being the ability to effectively circumvent the
model calibration process; instead, it simply estimates the values of predictive quantities of
interest conditioned on measured data. This idea is sometimes referred to as direct forecasting.
Furthermore, the DSI approach does not require access to model derivatives (in fact, the method
can be applied to non-differentiable models) while much of the required computation can be
carried out in parallel. The DSI approach has been used successfully in a variety of applications,
including subsurface hydrology (Delottier et al., 2023), petroleum engineering (Jiang et al., 2020;
Jiang and Durlofsky, 2021; Lima et al., 2020; Liu et al., 2021), and carbon storage (Sun and
Durlofsky, 2019; Jiang and Durlofsky, 2024). The application of the framework to geothermal
reservoir modelling, however, is largely unexplored and potentially more challenging as the
governing equations are generally highly nonlinear in the geothermal context, with simulation
non-convergence being a common issue (see, e.g., Croucher et al., 2020; O’Sullivan et al., 2013).
We note that the idea of direct forecasting is not exclusive to DSI; in particular, the Bayesian
evidential learning (BEL) framework (Scheidt et al., 2018) also involves direct forecasting. Like
DSI, the BEL framework has been applied in a variety of subsurface modelling applications
(Hermans et al., 2018; Michel et al., 2020; Pradhan and Mukerji, 2020); most notably, Athens
and Caers (2019) demonstrate the application of BEL to predict the temperature in a geothermal
target area of a synthetic model based on Dixie Valley, Nevada. We note, however, that only
single-phase, natural state simulations are considered in this study; by contrast, we apply the
DSI framework to a two-phase problem, and consider both natural state and production history
simulations.
In this paper, we build on our previous work (Power et al., 2022) to illustrate the applicability
of the DSI methodology to geothermal reservoir modelling. The present work makes a number
of new contributions to the DSI literature. First, we introduce a modification of the typical DSI
algorithm that allows us to sample from the resulting approximation to the posterior predictive
distribution directly and efficiently, avoiding the need to characterise this distribution using
techniques such as randomised maximum likelihood (Sun and Durlofsky, 2017; Sun et al., 2017)
or ensemble methods (Lima et al., 2020), which add an additional degree of approximation to the
procedure. Second, we illustrate the applicability of the resulting DSI algorithm to geothermal
reservoir modelling through two synthetic model problems (outlined in Section 3); one based on a
simplified two-dimensional reservoir and one based on a large-scale, three-dimensional reservoir.
Through these model problems, we provide a numerical comparison between DSI and other
methods for uncertainty quantification in subsurface modelling, investigate how the number
of reservoir model simulations affects the resulting approximation to the posterior predictive
distribution, and illustrate how one can evaluate the accuracy of the approximate mapping
constructed during the DSI algorithm.
2
METHODOLOGY
In this section, we first introduce some key notation used throughout the remainder of the pa-
per. We then briefly recall the key concepts and steps involved in a typical (Bayesian) statistical
approach to geothermal model calibration and prediction, and the associated uncertainty quan-
tification. Finally, we introduce data space inversion as a technique that allows us to circumvent
some of the difficulties associated with classical approaches to model calibration and prediction
in the Bayesian setting.
2


<!-- page 3 -->
2.1
Notation
Throughout the paper, we use bold lowercase letters to denote vectors and bold uppercase
letters to denote matrices. We use In to denote the identity matrix of dimension n, and 0n to
denote the zero vector of dimension n. We use the notation {x(i)}n
i=1 as a shorthand for the set
{x(1), x(2), . . . , x(n)}. For a symmetric positive definite matrix G ∈Rn×n and vector v ∈Rn, we
let ∥v∥G :=
√
v⊤Gv denote the Euclidean norm weighted by G. Finally, we use the notation
u ∼N(u0, Γ) to indicate that a random variable, u ∈Rn, has a Gaussian distribution with
mean u0 ∈Rn and covariance matrix Γ ∈Rn×n.
2.2
Bayesian Model Calibration and Prediction
Uncertainty quantification (UQ) for geothermal reservoir models typically uses the Bayesian
framework (Kaipio and Somersalo, 2006; Tarantola, 2005), which naturally allows for incorpo-
ration and quantification of various sources and types of uncertainty. The standard procedure
is as follows:
1. Calibrate (i.e., estimate) the model parameters using data.
2. Approximately (i.e., linearly) quantify the posterior uncertainty in the parameters.
3. Propagate the posterior parameter uncertainty to the predictive quantities of interest.
We now describe each of these steps in greater detail.
2.2.1
Calibration
Calibration of geothermal reservoir models is typically based on the assumption that the pa-
rameters and data are linked by a setup of the form
y = f(k) + e,
(1)
where y ∈Rd denotes the data, k ∈Rn denotes the unknown parameters, f : Rn →Rd represents
the forward model (or parameter to observable mapping), and e ∈Rd denotes measurement error
(as well as possible model error). Generally, application of the forward model f(·) involves first
solving for the the dynamics of the system using a reservoir simulator (for further details, see
Section 3.1), then applying an observation operator which extracts the simulation output at the
times and locations corresponding to the available data.
In a Bayesian setting, the model calibration problem is recast as a problem of statistical
inference, where the goal is to estimate the (parameter) posterior distribution, π(k|yobs); that
is, the conditional distribution of the parameters given a particular realisation of the data, yobs.
The posterior distribution can be expressed using Bayes’ theorem, as
π(k|yobs) = π(yobs|k)π(k)
π(yobs)
∝π(yobs|k)π(k).
(2)
In Equation (2), π(k) denotes the prior distribution, which describes our beliefs about the
parameters before (i.e., prior to) considering the data, while the likelihood, π(yobs|k), encodes
the likelihood of observing the data, yobs, under a given set of parameters.
The marginal
probability π(yobs) acts as a normalising constant, and is unimportant in most cases.
Assuming the errors, e, are independent of the parameters, k, it is well known that the
likelihood inherits the distribution of the errors; that is, π(yobs|k) = πe(yobs −f(k)) (see, e.g.,
3


<!-- page 4 -->
Calvetti and Somersalo, 2007; Kaipio and Somersalo, 2006). We can therefore rewrite Equation
(2) as
π(k|yobs) ∝π(yobs|k)π(k) = πe(yobs −f(k))π(k).
(3)
Assuming a-priori, as is standard, that the parameters and errors are normally distributed—
that is, k ∼N(k0, Γk) and e ∼N(0d, Γe)—the posterior can be expressed as (Calvetti and
Somersalo, 2007; Kaipio and Somersalo, 2006)
π(k|yobs) ∝exp
 
−1
2
 
∥yobs −f(k)∥2
Γ−1
e
+ ∥k −k0∥2
Γ−1
k
  
.
(4)
Computing a full characterisation of the posterior generally requires the use of sampling-based
approaches such as Markov chain Monte Carlo (MCMC) (Cui et al., 2011; Cui et al., 2019;
Maclaren et al., 2020; Scott et al., 2022). However, for most geothermal models these methods
are computationally infeasible.
As a computationally cheaper alternative, it is common to
compute the maximum a posteriori (MAP) estimate, kMAP; that is, the point in parameter
space which maximises the posterior density, defined as
kMAP := arg min
k∈Rn
 1
2∥yobs −f(k)∥2
Γ−1
e
+ 1
2∥k −k0∥2
Γ−1
k
 
.
(5)
The definition of the MAP estimate reveals a link between the Bayesian approach to model cali-
bration and classical, optimisation-based approaches; computing the MAP estimate is equivalent
to solving a regularised least-squares problem, where the form of the regularisation term follows
from the specification of the prior.
2.2.2
Approximate Parameter Uncertainty Quantification
After computing the MAP estimate, it is common to quantify the posterior uncertainty ap-
proximately, using a local Gaussian approximation (see, e.g., Omagbon et al., 2021; Zhang et
al., 2014); that is, π(k|yobs) ≈N(kMAP, Γpost). A common approximation to the posterior
covariance matrix is given by
Γpost = (F ⊤Γ−1
e F + Γ−1
k )−1,
(6)
where F denotes the sensitivity (Jacobian) matrix of the model with respect to the parameters
(i.e., F ij = ∂fi/∂kj, for i = 1, 2, . . . , d and j = 1, 2, . . . , n), evaluated at the MAP estimate. A
sample, ki, from the Gaussian approximation to the posterior, N(kMAP, Γpost) can be generated
as
ki = kMAP + Lpostηi,
(7)
where LpostL⊤
post = Γpost, and ηi ∼N(0n, In) is a sample from the n-dimensional standard
Gaussian distribution. Because this approximation uses a linearisation of the forward model, it
is often referred to as linearisation about the MAP estimate (LMAP).
We note that there exist several additional classes of methods that have been used to approx-
imate the parameter posterior and posterior predictive distributions in geothermal settings. One
such method is randomised maximum likelihood (RML; Kitanidis, 1995; Oliver et al., 1996),
in which a set of stochastic optimisation problems are solved to obtain samples distributed in
regions of high posterior density (geothermal applications include Bjarkason et al., 2020; Tian et
al., 2024; T¨ureyen et al., 2014; Zhang et al., 2014). These optimisation problems take the same
form as Equation (5), but the data is perturbed using a sample from the distribution of the error
4


<!-- page 5 -->
and the prior mean is replaced by a sample from the prior. While RML generally produces a
more accurate approximation to the posterior than linearisation about the MAP estimate, the
computational expense is amplified given that each sample from the (approximate) posterior
incurs a similar computational cost to computing the MAP estimate. An alternative class of
methods are ensemble methods (Chen and Oliver, 2013; Emerick and Reynolds, 2013; Iglesias
et al., 2013), which are based on sampling methods such as RML, but employ an ensemble
(sample-based) approximation of any required derivative or covariance information (geothermal
applications include B´ek´esi et al., 2020; Bjarkason et al., 2020). These are simpler to implement
and can be less computationally intensive than methods such as RML. However, the calibration
process is still iterative and potentially requires many forward simulations.
2.2.3
Approximate Predictive Uncertainty Quantification
Carrying out uncertainty quantification for the predictive quantities of interest, denoted here
by p ∈Rm, relies on having a predictive model, q : Rn →Rm, relating the parameters to the
predictions; that is,
p = q(k).
(8)
In various settings, the forward model f(·) and predictive model q(·) may be represented by the
same model but evaluated at different locations in space and/or time. In any case, to propagate
the uncertainty from the parameters through to the predictions, typically one of two methods
is applied (Omagbon et al., 2021):
1. Samples from the (approximate) parameter posterior are generated (see Equation (7)) and
run through q to give prediction samples; that is, q(ki), where ki ∼N(kMAP, Γpost).
2. A Gaussian approximation of the posterior predictive distribution is made; that is, p ∼
N(pMAP, Γpred), with
kMAP = q(kMAP),
Γpred = QΓpostQ⊤,
(9)
where Q denotes the sensitivity (Jacobian) matrix of the predictive model with respect
to parameters (i.e., Qij = ∂qi/∂kj) for i = 1, 2, . . . , m and j = 1, 2, . . . , n, evaluated at the
MAP estimate.
The latter approach is often termed linear(-ised) uncertainty propagation, and may not be
suitable in all situations; see Omagbon et al. (2021) for a comparison and discussion of these
two approaches within a geothermal context. We emphasise that both of the aforementioned
approaches only approximate the posterior predictive distribution (with the exception of the case
where both the forward and predictive models are linear and the prior and error distributions
are Gaussian).
2.2.4
Accurate Predictive Uncertainty Quantification
Although infeasible in most geothermal settings, it is theoretically possible to accurately char-
acterise the posterior predictive distribution. Specifically, this involves first generating samples
from the true posterior using (for example) MCMC, then running these samples through the
predictive model. In this work, we use this approach to provide a benchmark for the simplified
two-dimensional reservoir (see Section 3.2).
5


<!-- page 6 -->
2.3
Data Space Inversion
The main computational bottleneck associated with the standard procedure for (approximate)
uncertainty quantification is computation of the MAP estimate, kMAP. Due to the scale and
computational complexity of a typical geothermal model, solving the optimisation problem (5)
can require a significant amount of time, even when efficient methods such as adjoint-based
approaches are used (Bjarkason et al., 2018; Bjarkason et al., 2019; Gonzalez-Gutierrez et al.,
2018). By contrast, data space inversion procedures (Sun and Durlofsky, 2017; Sun et al., 2017)
essentially (approximately) marginalise over the uncertain parameters, k, using a surrogate
model, to focus on the posterior predictive distribution. This is natural when the predictions
and associated uncertainties are of primary interest, as is common in geothermal settings, and
avoids the need to compute the MAP estimate altogether. DSI procedures work by building an
approximation to the joint distribution of the data and predictive QoIs, π(y, p), using samples
from this distribution generated by simulating the forward model using sets of parameters drawn
from the prior. This approximation is then conditioned on the observed data, yobs, to form an
approximation to the posterior predictive distribution, π(p|yobs).
In the simplest form of DSI, the joint distribution of the data and predictive quantities of
interest is approximated using a multivariate Gaussian distribution (Sun and Durlofsky, 2017);
the conditional distribution associated with a given instance of the data, yobs, is then available
analytically. In many settings, however, the actual joint distribution is highly non-Gaussian,
and so this approach can give poor results. In such cases, it is common to instead build a
surrogate mapping that transforms samples from a simple “reference” distribution (typically a
multivariate Gaussian distribution) to samples distributed (approximately) according to π(y, p).
The associated approximation to the posterior predictive distribution, under this surrogate, is
then characterised using techniques such as RML (Jiang et al., 2020; Sun and Durlofsky, 2017;
Sun et al., 2017) or ensemble methods (Delottier et al., 2023; Lima et al., 2020).
In the remainder of this section, we outline the DSI procedure we use in this work. In Sec-
tion 2.3.1, we describe how we construct a surrogate that maps between a standard multivariate
Gaussian distribution and the joint distribution of the data and predictive quantities of interest.
Then, in Section 2.3.2, we outline how we can utilise the structure of this surrogate to sample
directly and efficiently from the associated approximation to the posterior predictive distribu-
tion, without the use of techniques such as RML or ensemble methods. Then, in Section 2.3.3,
we discuss techniques for evaluating the quality of the DSI surrogate. Finally, in Section 2.3.4,
we discuss how our approach is related to other variants of the DSI algorithm, as well as other
sampling techniques.
2.3.1
Construction of the DSI Surrogate
In this section, we outline how we construct a surrogate which transforms samples distributed
according to the standard multivariate Gaussian distribution of the appropriate dimension to
samples distributed (approximately) according to π(y, p). To do this, we first draw a set of
samples, {k(j)}J
j=1, from the prior, and a set of samples, {e(j)}J
j=1, from the distribution of the
errors. We then run the forward and predictive models to generate the corresponding samples
from π(y, p), denoted by {(y(j), p(j))}J
j=1, where
y(j) = f(k(j)) + e(j),
p(j) = q(k(j)).
(10)
We use these samples to construct an approximate mapping between each element of the data
and predictive QoIs, and a standard Gaussian random variable. To do this, we begin by noting
that a mapping from a standard Gaussian random variable, ξy
i ∼N(0, 1), to the ith element of
the data, yi, is given by
yi = ((Fy
i )−1 ◦Φ)(ξy
i ),
(11)
6


<!-- page 7 -->
where Φ : R →(0, 1) denotes the cumulative distribution function (CDF) of the standard
Gaussian distribution, and Fy
i : X y
i ⊆R →[0, 1] denotes the CDF of yi, which we assume is
continuous and strictly increasing (and therefore invertible) on X y
i , which denotes the range of
possible values taken by yi. Similarly, a mapping from a standard Gaussian random variable,
ξp
i , to the ith predictive QoI, pi, is given by
pi = ((Fp
i )−1 ◦Φ)(ξp
i ),
(12)
where Fp
i : X p
i
⊆R →[0, 1] denotes the CDF of pi. We denote the mappings in Equation
(11) and (12) as Py
i := (Fy
i )−1 ◦Φ and Pp
i := (Fp
i )−1 ◦Φ, respectively. Then, by defining the
mapping P : Rd+m →Rd+m, as
P(ξ) :=
 Py(ξy)
Pp(ξp)
 
,
where
Py(ξy) :=


Py
1 (ξy
1 )
Py
2 (ξy
2 )
...
Py
d (ξy
d )

,
Pp(ξp) :=


Pp
1 (ξp
1 )
Pp
2 (ξp
2 )
...
Pp
m(ξp
m)

,
(13)
it follows that
 
y
p
 
= P(ξ),
(14)
where ξ = [ξy
1 , ξy
2 , . . . , ξy
d , ξp
1 , ξp
2 , . . . , ξp
m]⊤∈Rd+m is a vector of standard Gaussian random
variables.
Of course, we do not have access to the CDFs {Fy
i (·)}d
i=1 and {Fp
i (·)}m
i=1 used to construct
the mapping P; instead, we approximate each of these functions using the previously-drawn
samples. The resulting procedure is referred to as (empirical) Gaussian anamorphosis in the
geostatistics literature (Wackernagel, 2003), and is a common element in many applications of
DSI (see, e.g., Sun et al., 2017; Sun and Durlofsky, 2017; Sun and Durlofsky, 2019). In this
work, we use a piecewise linear construction of each CDF; Figure 1 shows an example of this.
Of note is the fact that the construction of the CDF shown in Figure 1 is continuous and strictly
increasing (and therefore invertible) on the interval X y
i
:= (ymin
i
, ymax
i
), which represents the
range of possible values for yi.
We note that, while the marginal distributions of each element of ξ are Gaussian, the joint
distribution is not, in general. Nevertheless, the assumption that the joint distribution of ξ is
Gaussian is often reasonable in practice; we illustrate this using the case studies in Section 3.
Additionally, we note that this distribution generally possesses a non-trivial covariance structure,
which we need to estimate. To do this, we first apply the inverse of transformation P to each
of the previously-generated samples from π(y, p), to obtain a set of realisations of ξ; that is,
ξ(j) = P−1(y(j), p(j)),
j = 1, 2, . . . , J.
(15)
Next, we compute the empirical covariance matrix of these samples, which is given by
C = 1
J
J
X
j=1
(ξ(j))⊤ξ(j).
(16)
Note that, by construction, the mean of the samples {ξ(j)}J
j=1 is equal to 0.
Finally, we compute the Cholesky factorisation of C, C = LL⊤(see, e.g., Golub and Van
Loan, 2013)1. Given the Cholesky factor L, the final DSI surrogate, T : Rd+m →X y×X p, where
1We note that for such a factorisation to exist, C must be full rank, which may not always be the case (note
that the rank of C is always bounded above by the number of samples, J, used to construct it, which may be
less than d + m, the dimension of ξ). To avoid this, we can simply add a small (positive) multiple of the identity
matrix to C to ensure it is full rank.
7


<!-- page 8 -->
ymin
i
y(2)
i
y(5)
i
y(4)
i
y(3)
i
y(1)
i
ymax
i
yi
0.00
0.25
0.50
0.75
1.00
Fy
i (yi)
Figure 1
A piecewise linear approximation to the CDF, Fy
i (·), of a given element of the data,
yi, using five samples, {y(j)
i
}5
j=1. The values ymin
i
and ymax
i
represent bounds on the
range of possible values for yi.
X y := X y
1 ×X y
2 ×· · ·×X y
d ⊆Rd and X p := X p
1 ×X p
2 ×· · ·×X p
m ⊆Rm, which transforms samples
of the Gaussian random variable η ∼N(0d+m, Id+m) to samples distributed approximately
according to π(y, p), can be expressed as
T (η) = (P ◦L)(η).
(17)
Note that this transformation is composed of two key steps. First, the application of L to η
acts to give η the required correlation structure. Next, the application of P acts to transform
the marginal distributions of each element of the resulting vector to the desired form.
2.3.2
Posterior Predictive Simulation
Once the mapping T (·) has been constructed, it is easy to draw samples from the approximation,
under T (·), to the posterior predictive distribution, π(p|yobs). To illustrate how this can be done,
we first note that the mapping is endowed with a particular structure. Because the Cholesky
factor L is a lower triangular matrix, it follows that T (·) is a lower triangular transformation;
that is, each component Ti(·) is a function of the first i input variables only. Additionally, each
component Ti(·) is strictly increasing as a function of the ith input variable; this is a consequence
of the fact that diagonal entry Lii of the Cholesky factor is strictly positive by definition, and
the mapping Pi(·) is strictly increasing by construction.
The triangular structure of T (·) means that it can be expressed in the form
T (η) =
  T y(ηy)
T p(ηy, ηp)
 
,
(18)
where (ηy, ηp) is a partitioning of the random variable η into its first d components and its last
m components, and T y : Rd →X y and T p : Rd+m →X p. Then, owing to the monotonicity
property of T , it follows that the mapping
ηp 7→T p(ηobs
y , ηp),
(19)
8

[CAPTION] Figure 1
A piecewise linear approximation to the CDF, Fy


<!-- page 9 -->
where
ηobs
y
= (T y)−1(yobs),
(20)
transforms a vector, ηp, of independent samples from the standard normal distribution to sam-
ples distributed according to the approximation, under T (·), of the conditional density of p | yobs;
that is, the posterior predictive distribution π(p|yobs). For further details and a proof of this
property, we refer the reader to Marzouk et al. (2016, Sec. 7).
The above gives us a simple and efficient way to generate samples from the posterior predic-
tive distribution. We first apply the inverse of T y to the observations, yobs, to obtain ηobs
y . We
then generate a sample by computing
T p(ηobs
y , ηp),
where
ηp ∼N(0m, Im).
(21)
2.3.3
Evaluating the Quality of the DSI Surrogate
The construction of the DSI surrogate outlined in Section 2.3.1 makes use of two approximations;
the quality of the DSI approximation to the posterior predictive distribution depends on the
degree to which these approximations are accurate. The first is the approximation of the CDF
of each element of the data and predictive quantities of interest using the set of samples drawn
from the prior. This approximation will, in general, improve as the number of samples, J, used
to construct each CDF is increased. The second is the approximation of the joint density of
the random variable ξ, obtained by applying the inverse of the mapping P(·) to the data and
predictive quantities of interest (see Eq. 14) as Gaussian.
There are several qualitative checks we can carry out to evaluate the quality of the DSI
surrogate, T (·), after its construction.
First, we can plot the joint densities of elements of
the samples of the data and predictive quantities of interest after applying the inverse of the
surrogate mapping to each sample, to evaluate their degree of Gaussianity. Ideally, the resulting
samples should appear similar to draws from a standard multivariate Gaussian distribution.
However, the data and predictive quantities of interest are generally high-dimensional, which
makes this check challenging to carry out.
A second approach is to plot samples from the
approximation to the distribution of the data and/or predictive quantities of interest under
the DSI surrogate, by applying the mapping to sets of independent samples from the standard
normal distribution (see, e.g., Sun et al., 2017; Jiang et al., 2021). Ideally, the resulting samples
should appear qualitatively similar to the samples used to construct the surrogate. Finally, we
can withhold a set of “validation” samples of the data and predictive quantities of interest when
constructing the DSI surrogate. We can then condition on each validation dataset using DSI
to form the associated approximation to the posterior predictive distribution, and determine
whether the corresponding values of the predictive quantities of interest fall within the range
of the posterior predictions. We note that the construction of the DSI surrogate (including the
required simulations of the forward model) can be done offline; that is, without knowledge of
the data. As a result, this process can be carried out in a computationally efficient manner. We
use the case studies in Section 3 to illustrate the second and third approaches to evaluating the
quality of the DSI surrogate.
Additionally, we note that, like many other inference techniques that use surrogate models,
DSI may not provide accurate results when used to condition on observed data that differs
meaningfully to the samples of the data used to build the DSI surrogate. To deal with this, we
can make use of prior predictive checks (see, e.g, Gelman et al., 2013; Gelman et al., 2020)—
that is, visualisations of samples of the data (and predictive quantities of interest, if the data
comprise a subset of these) drawn from the prior—to ensure that these samples encompass the
full range of expected behaviour of the system under consideration. If this is not the case, the
prior can be re-characterised and the process repeated. This will help to reduce the risk that
9


<!-- page 10 -->
the prior is unrepresentative of the data that is collected. We note that similar ideas are also
part of the Bayesian evidential learning framework (Scheidt et al., 2018).
Finally, we emphasise that the DSI surrogate may generate samples that do not adhere to the
physics of the particular problem under consideration. It is, therefore, important that results
obtained using the DSI algorithm (or any other form of surrogate model) are interpreted with
caution, by people with a high level of domain knowledge. If a significant number of the samples
generated using DSI do not appear to be physically plausible, this suggests that an alternative
approach to inference may be required.
2.3.4
Related Work
We note that the DSI algorithm we have described shares similarities to the variant of DSI
introduced by Sun and Durlofsky (2017, Sec. 3), which we briefly outline. In this variant of
DSI, the authors first perform principal component analysis (PCA) on a set of samples of the
data and predictive quantities of interest obtained by simulating the forward model using a set
of samples drawn from the prior. Then, sampling from the resulting approximation to the prior
predictive distribution of the data and predictive quantities of interest involves the projection of
samples from the standard multivariate Gaussian distribution of the appropriate dimension onto
the previously-computed principal components, followed by an empirical Gaussian anamorphosis
step which acts to transform the marginal distributions of each quantity to the desired shape. We
note that the use of PCA to re-parametrise the observations and predictive quantities of interest
means that the resulting mapping is not lower triangular, making it challenging to sample from
directly; instead, Sun and Durlofsky (2017) characterise the approximation to the posterior
predictive distribution associated with a particular instance of the data using a randomised
maximum likelihood (RML) procedure. Our variant of the DSI algorithm, by contrast, allows
for direct simulation from the posterior predictive distribution, avoiding the need for sampling
methods such as RML, which add additional computation and an extra degree of approximation
to the procedure.
Additionally, we note that the DSI mapping described in this work can be thought of as
a simple instance of a larger class of lower triangular, strictly increasing transformations that
provide an (approximate) mapping between a simple “reference” random variable and a complex
“target” random variable. For further discussion of these ideas, we refer the reader to Marzouk
et al. (2016).
3
COMPUTATIONAL EXAMPLES
We now demonstrate the DSI approach by applying it to a simplified two-dimensional problem,
as well as a large-scale, three-dimensional reservoir model. Both problems are adapted from those
presented in de Beer (2024b). We first outline the governing equations for general geothermal
reservoir modelling, before presenting the results of each model problem.
3.1
Governing Equations
The dynamics of a geothermal reservoir are described (mathematically) by a non-isothermal,
multi-phase version of Darcy’s law, enforcing conservation of mass and energy (Croucher et al.,
2020; O’Sullivan and O’Sullivan, 2016). In what follows, we denote by Ω∈R3 the domain of
interest, with boundary ∂Ωand outward-facing normal vector n. Furthermore, we let γ denote
the number of each component (e.g., water, air, energy). The governing equations, expressed in
integral form, are then
d
dt
Z
Ω
M γ dx = −
Z
∂Ω
F γ · n dσ +
Z
Ω
qγ dx,
γ = 1, 2, . . . , N + 1,
(22a)
10


<!-- page 11 -->
where components γ = 1, 2, . . . , N denote the mass components and component γ = N + 1
denotes the energy component. For each mass component, M γ denotes mass density (kg m−3),
F γ denotes mass flux (kg m−2 s−1), and qγ denotes mass sources or sinks (kg m−3 s−1). For the
energy component, M γ denotes energy density (J m−3), F γ denotes energy flux (J m−2 s−1), and
qγ denotes energy sources or sinks (J m−3 s−1). The mass and energy densities can be expressed
as
M γ =
(
ϕ(ρlSlXγ
l + ρvSvXγ
v ),
γ < N + 1,
(1 −ϕ)ρrurT + ϕ(ρlulSl + ρvuvSv),
γ = N + 1,
(22b)
where ϕ denotes porosity (dimensionless), Sl and Sv denote liquid saturation and vapour satura-
tion (dimensionless) respectively, ρl, ρv and ρr denote the density of the liquid, vapour and rock
(kg m−3) respectively, Xγ
l and Xγ
v denote the liquid and vapour mass fractions (dimensionless)
of component γ respectively, ul and uv denote the internal energy of the liquid and vapour
(J kg−1) respectively, ur denotes the specific heat of the rock (J kg−1 K−1), and T denotes tem-
perature (K). Next, the mass (γ < N + 1) fluxes are given by the sum of the mass flux of liquid
and the mass flux of vapour,
F γ = F γ
l + F γ
v,
F γ
l = −kkrl
νl
Xγ
l (∇P −ρlg),
F γ
v = −kkrv
νv
Xγ
v (∇P −ρvg).
(22c)
Here, κ represents the permeability tensor (m2), P denotes pressure (Pa), νl and νv denote
the kinematic viscosity of liquid and vapour (m2 s−1) respectively, κrl and κrv denote relative
permeabilities (dimensionless), and g denotes gravitational acceleration (m s−2). Finally, the
energy (γ = N + 1) flux is given by
F γ = −K∇T +
N
X
m=1
X
χ
hm
χ F m
χ ,
(22d)
where hm
χ denotes the specific enthalpy (J kg−1) of mass component m in phase χ, and K denotes
thermal conductivity (J s−1 m−1 K−1).
3.2
Two-Dimensional Single-Phase Model
Our first model problem serves to provide a comparison between the posterior predictions pro-
duced using MCMC, linearisation about the MAP estimate (see Section 2.2.2), and DSI, when
applied to a high-dimensional subsurface flow problem. Problems of a similar nature are often
used as benchmarks when evaluating uncertainty quantification algorithms (see, e.g., Aristoff
and Bangerth, 2023; Christie and Blunt, 2001).
3.2.1
Problem Setup
For this problem, we make several simplifications to the governing equations introduced in
Section 3.1. Namely, we consider a two-dimensional reservoir with domain Ω= (0 m, 1000 m)2,
containing single-phase, isothermal, slightly compressible fluid (see, e.g., Chen, 2007), with a
set of nw production wells. We further assume that the permeability tensor, κ, is isotropic. In
this case, Equation (22) simplifies to
cϕ∂P
∂t −1
µ∇· (κ∇P) =
nw
X
i=1
qiδ(x −xi),
x ∈Ω, t ∈(0, τ],
(23)
where c denotes fluid compressibility (assumed to be 2.9 × 10−8 Pa−1), µ denotes dynamic
viscosity (assumed to be 0.5 mPa s), qi denotes the extraction rate of well i, and δ(x −xi)
11


<!-- page 12 -->
0
500
1000
x1 [m]
0
500
1000
x2 [m]
0
500
1000
x1 [m]
0
500
1000
x2 [m]
WELL 7
WELL 4
WELL 1
WELL 8
WELL 5
WELL 2
WELL 9
WELL 6
WELL 3
0
80
160
Time [Days]
16
18
20
Pressure [MPa]
−31
−30
−29
ln(Perm) [ln(m2)]
Figure 2
The setup for the simplified two-dimensional reservoir model.
Left:
the true log-
permeability field. Centre: the locations of the production wells. Right: the pressure in
well 8; the solid line denotes the true pressure, the dots denote the noisy data collected
during the production history period, and the grey region denotes the forecast period.
0
500
1000
x1 [m]
0
500
1000
x2 [m]
t = 40 Days
0
500
1000
x1 [m]
t = 80 Days
0
500
1000
x1 [m]
t = 120 Days
0
500
1000
x1 [m]
t = 160 Days
15
16
17
18
19
20
Pressure [MPa]
Figure 3
The true reservoir pressure at t = 40, t = 80, t = 120 and t = 160 days.
denotes a Dirac delta mass centred at well i, the location of which is indicated by xi.
We
assume a reservoir porosity of ϕ = 0.3, and impose the boundary and initial conditions
−κ∇P · n = 0,
x ∈∂Ω,
t ∈(0, τ],
P = P0,
x ∈Ω,
t = 0.
(24)
In the above, P0 denotes the initial reservoir pressure, which we assume to be 20 MPa. The sole
uncertain parameter is the (spatially varying, isotropic) permeability of the reservoir, κ.
We consider a setup in which a set of nw = 9 production wells operate over a period of
τ = 160 days. The first 80 days act as the production period, while the final 80 days act as the
forecast period. The location of each well is indicated in Figure 2. For the first 40 days, each of
the odd-numbered wells extracts fluid at a rate of 50 m3 day−1, while the even-numbered wells
are turned off. For the next 40 days, this is reversed; the even-numbered wells extract fluid at
a rate of 50 m3 day−1, while the odd-numbered wells are turned off. For the next 40 days, all
wells are turned off, before operating at a rate of 25 m3 day−1 for the final 40 days. Figure 3
shows the true reservoir pressure at the end of each 40-day period.
3.2.2
Prior Parametrisation
As is standard, when solving the inverse problem we work in terms of the log-permeability,
u := ln(κ), which ensures that the resulting estimates of the permeability are positive. We
parametrise the log-permeability of the reservoir using a Gaussian random field (GRF) with a
mean function of m(x) = −31 ln(m2), and a squared-exponential covariance function (Williams
and Rasmussen, 2006), given by
C(x, x′) = σ2 exp
 
−1
2ℓ2 ∥x −x′∥2
 
.
(25)
12

[CAPTION] Figure 2
The setup for the simplified two-dimensional reservoir model.

[CAPTION] Figure 3
The true reservoir pressure at t = 40, t = 80, t = 120 and t = 160 days.


<!-- page 13 -->
0
500
1000
x1 [m]
0
500
1000
x2 [m]
0
500
1000
x1 [m]
0
500
1000
x1 [m]
0
500
1000
x1 [m]
−32
−31
−30
ln(Perm) [ln(m2)]
Figure 4
Samples from the prior distribution of the reservoir permeability.
We use a standard deviation of σ = 0.75 ln(m2) and a characteristic lengthscale of ℓ= 250 m.
To reduce the dimension of the parameter space and accelerate the convergence of our MCMC
sampler, we approximate this GRF using a truncated Karhunen–Lo`eve expansion; that is,
u ≈
n
X
i=1
p
λiviηi,
(26)
where ηi ∼N(0, 1).
In Equation (26), {(λi, vi)}n
i=1 denote the n largest eigenpairs of the
covariance matrix of the (discretised) GRF, where n is typically small compared to the dimension
of the field. Under this parametrisation, the values of the coefficients {ξi}n
i=1 become the targets
of inference. Here, we retain n = 50 coefficients. Figure 4 shows several draws from the prior.
3.2.3
Data
We assume that the pressure at each well is recorded every 8 days throughout the production
history period (i.e., the first 80 days); this gives a total of 90 measurements. We add indepen-
dent Gaussian noise with a standard deviation of 1% of the initial reservoir pressure to each
observation. Figure 2 shows the data collected at well 8.
3.2.4
Simulation
We discretise the system using a cell-centred finite difference scheme (Chen, 2007; Haber and
Hanson, 2007), and use the backward Euler method to solve for the dynamics of the system
over time. The permeability distribution of the true system, shown in Figure 2, is generated
using a draw from the prior. To avoid the “inverse crime” of generating the synthetic data
and solving the inverse problem using the same numerical discretisation (Kaipio and Somersalo,
2006; Kaipio and Somersalo, 2007), we use an 80 × 80 grid when simulating the dynamics of the
true system, but a 50 × 50 grid when carrying out each inversion.
3.2.5
Inference Methods
We aim to use the data collected at each well over the production period to estimate how the
pressure at each well will change over the forecast period.
We compute a complete characterisation of the posterior and posterior predictive distribu-
tions using the preconditioned Crank-Nicolson MCMC sampler (Chen et al., 2019; Cotter et al.,
2013), which is commonly used to solve high-dimensional inverse problems. We run four Markov
chains, each initialised at a random draw from the prior, for 500,000 iterations, and discard the
first half of each chain as burn in. These results provide a reference to which we can compare
the posterior predictive distributions produced using linearisation about the MAP estimate and
DSI.
13

[CAPTION] Figure 4
Samples from the prior distribution of the reservoir permeability.


<!-- page 14 -->
10
15
20
Prior
Pressure [MPa]
WELL 2
WELL 5
WELL 8
WELL 9
0
80
160
Time [Days]
10
15
20
DSI (Unconditional)
Pressure [MPa]
0
80
160
Time [Days]
0
80
160
Time [Days]
0
80
160
Time [Days]
Figure 5
Sets of 100 samples from the prior predictive distribution (top row) and the DSI ap-
proximation to the prior predictive distribution (bottom row) of the pressure in wells
2, 5, 8 and 9. In all plots, the grey region denotes the forecast period.
When characterising the posterior using LMAP (as outlined in Section 2.2.2), we compute the
MAP estimate using a matrix-free inexact Gauss-Newton conjugate gradient method (see, e.g.,
Haber and Hanson, 2007; Petra and Stadler, 2011). The process of computing the MAP estimate
and forming the approximate posterior covariance matrix requires 244 “forward-like” solves
(these include both forward and adjoint solves, which are associated with similar computational
costs), though this could, of course, be reduced through the use of an improved optimisation
method. We then run samples from the approximate posterior through the predictive model to
obtain samples from the corresponding (approximate) posterior predictive distribution. Note
that this corresponds to the first method of approximate predictive uncertainty quantification
outlined in Section 2.2.3.
When approximating the posterior predictive distribution using DSI, we use an initial set of
1000 samples from the prior to build the mapping outlined in Section 2.3.1. We then examine
the differences in the results when the number of samples used to estimate the DSI mapping is
varied.
3.2.6
Validation
Before discussing the results obtained using each inference technique, we evaluate the quality of
the DSI surrogate, T (·), by plotting a set of (unconditional) realisations from the DSI approxi-
mation to the prior predictive distribution of the pressure of several of the wells of the system,
as discussed in Section 2.3.3. These results are shown in Figure 5; we note that this visualisation
also functions as an example of a prior predictive check. We observe that in general, the DSI
predictions exhibit similar behaviour to the prior predictions, with the exception of a handful
of samples of the pressure in wells 5 and 9 which exhibit some oscillatory behaviour near the
end of the production period that is not present in the corresponding samples from the prior.
14

[CAPTION] Figure 5
Sets of 100 samples from the prior predictive distribution (top row) and the DSI ap-


<!-- page 15 -->
15.5
18.0
20.5
WELL 1
Pressure [MPa]
Prior
MCMC
LMAP
DSI
15.5
18.0
20.5
WELL 6
Pressure [MPa]
0
80
160
Time [Days]
15.5
18.0
20.5
WELL 8
Pressure [MPa]
0
80
160
Time [Days]
0
80
160
Time [Days]
0
80
160
Time [Days]
Figure 6
Sets of 1000 samples from the prior predictive distribution (left) and the posterior
predictive distributions generated using MCMC (centre left), LMAP (centre right),
and DSI (right), for wells 1, 6 and 8. In all plots, the blue lines indicate the samples,
the black line denotes the true well pressure, the black dots denote the observations
collected during the production history period, and the grey region denotes the forecast
period.
3.2.7
Results
We now compare the results obtained using each inference technique. Figure 6 shows a set of
1000 samples of the pressure at wells 1, 6 and 8, drawn from the posterior predictive distributions
produced using MCMC, LMAP, and DSI. In each case, a set of 1000 samples from the prior
is also presented for comparison.
We observe that in all cases, the posterior uncertainty is
significantly reduced in comparison to the prior uncertainty, and the true pressure at each well
is contained within the predictions. The predictions of the pressure at well 1 obtained using
DSI appear to have a slightly greater variance in comparison to those generated using MCMC
and LMAP. Those for well 6 and well 8, by contrast, appear very similar.
Figure 7 shows
the estimates of the marginal densities of the pressure in each well at the end of the forecast
period (t = 160 days) obtained using each method (we note that the DSI densities are known
analytically). These plots largely reinforce these conclusions.
Figure 8 shows how the posterior predictions change as the number of samples used to
estimate the DSI surrogate T (·) varies. The posterior predictive distribution generated using
100 samples is often significantly different to the approximations generated using larger numbers
of samples (see, e.g., well 2 and well 3). However, after the number of samples reaches 500, the
predictive distributions begin to look very similar to one another. This suggests that J = 500
15

[CAPTION] Figure 6
Sets of 1000 samples from the prior predictive distribution (left) and the posterior

[CAPTION] Figure 7 shows

[CAPTION] Figure 8 shows how the posterior predictions change as the number of samples used to


<!-- page 16 -->
15.5
16.5
0
9
Probability Density
WELL 1
16
17
0.0
8.5
WELL 2
16
17
0.0
10.5
WELL 3
15.0
16.5
0
7
Probability Density
WELL 4
15
16
0
9
WELL 5
16
17
0
8
WELL 6
16.0
16.5
Pressure [MPa]
0
9
Probability Density
WELL 7
16
17
Pressure [MPa]
0
8
WELL 8
16.0
16.5
Pressure [MPa]
0
9
WELL 9
Truth
Prior
MCMC
LMAP
DSI
Figure 7
The prior predictive distribution and posterior predictive distributions of the pressure
in each well at the end of the forecast period (t = 160 days) obtained using MCMC,
LMAP, and DSI.
16

[CAPTION] Figure 7
The prior predictive distribution and posterior predictive distributions of the pressure


<!-- page 17 -->
16
17
0
17
Probability Density
WELL 1
16
17
0
14
WELL 2
16
17
0
17
WELL 3
15
17
0
8
Probability Density
WELL 4
15.0
16.5
0
10
WELL 5
15.5
17.0
0
10
WELL 6
15
17
Pressure [MPa]
0
10
Probability Density
WELL 7
16
17
Pressure [MPa]
0
10
WELL 8
16
17
Pressure [MPa]
0
10
WELL 9
Truth
Prior
MCMC
DSI (J = 100)
DSI (J = 250)
DSI (J = 500)
DSI (J = 1000)
DSI (J = 2000)
DSI (J = 5000)
DSI (J = 10000)
Figure 8
The prior predictive distribution and posterior predictive distributions of the pressure
in each well at the end of the production period (t = 160 days) obtained using MCMC,
and DSI with varying numbers of samples.
or more samples is an appropriate number to use when applying DSI to this problem.
We
note, however, that there is no guarantee a sample size that provides acceptable results in one
context will provide acceptable results in another. The number of samples after which the DSI
estimate of the predictive QoIs begins to stabilise will depend on a variety of factors, including
the characteristics of the forward model, the prior parametrisation, and the dimensions of the
predictive quantities of interest and the data. In future work, it would be valuable to investigate
how the required number of samples for a stable DSI estimate of the predictive QoIs varies when
these characteristics of the problem are modified.
3.3
Three-Dimensional Reservoir Model
The second test case we consider is a synthetic three-dimensional reservoir model.
3.3.1
Problem Setup
The model domain, shown in Figure 9, spans 6000 m in the horizontal (x1 and x2) directions,
and extends to a depth of 3000 m in the vertical direction. Figure 10 shows the true subsurface
17

[CAPTION] Figure 8
The prior predictive distribution and posterior predictive distributions of the pressure


<!-- page 18 -->
0
3000
6000
x1 [m]
0
3000
6000
x2 [m]
WELL 1
WELL 2
WELL 3
WELL 4
WELL 5
WELL 6
WELL 7
WELL 8
WELL 9
Figure 9
The mesh of the synthetic reservoir model, and the locations of each production well.
Existing wells are denoted using blue and new wells are denoted using red.
0
50
100
150
200
250
Temperature [±C]
0
°17
°16
°15
°14
°13
log10(Perm) [log10(m2)]
0
6000
x1 [m]
0
6000
x2 [m]
0.0
0.5
1.0
1.5
2.0
2.5
Upﬂow [kg s°1 m°2]
£10°4
Figure 10
The true permeability structure (left), mass upflows (centre) and natural state con-
vective plume (right) of the synthetic reservoir model.
permeability structure, mass upflows, and natural state convective plume of the system.
We assume that we have been extracting fluid at each of the seven existing wells (wells 1–7
in Figure 9) of the system at a rate of 0.25 kg s−1 over a production history period of one year.
We then wish to estimate the downhole temperature profiles associated with each of the existing
wells, as well as two new wells (well 8 and well 9 in Figure 9), and to predict how the pressure
and enthalpy of the fluid extracted at each well will change if we operate all wells at the increased
rate of 0.5 kg s−1 for a forecast period of an additional year. Each well has a single feedzone at
a depth of 1200 m. As is standard in geothermal reservoir modelling, we consider a combined
natural state and production history simulation (O’Sullivan and O’Sullivan, 2016); that is, we
simulate the dynamics of the system until steady-state conditions are reached, then use the
resulting state of the system as the initial condition for the subsequent production simulation.
3.3.2
Prior Parametrisation
When parametrising the prior, we consider uncertainty in the subsurface permeability structure
(modelled as isotropic, for simplicity), the reservoir porosity, and the location and magnitude
of the hot mass upflow at the base of the reservoir. All other reservoir properties are assumed
known. The rock of the reservoir is assumed to have a thermal conductivity of 2.5 W m−1 K−1
and a specific heat of 1000 J kg−1 K−1. The top boundary of the model is set to a constant
18


**[Table p18.1]**
|  | 6000 WELL 7 WELL 5 WELL 2 WELL 8 WELL 4 WELL 6 [m] 3000 x 2 WELL 9 WELL 3 WELL 1 0 0 3000 6000 x [m] 1 |
| --- | --- |


**[Table p18.2]**
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  | W WEL |  |  |  |  |  |  | WELL |  | 5 |  | WE 6 |  | LL |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 7 |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  | W | E | LL |  | 2 |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  | L | L | 8 |  |  | WEL | L | 4 | WE | LL |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  | W | EL |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  | L 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  | W | ELL |  | 9 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  | WE | LL | 3 |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |


**[Table p18.3]**
| [log10(m2)] °13 °14 °15 log10(Perm) °16 °17 0 | £10°4 6000 2.5 m°2] 2.0 s°1 [m] 1.5 [kg x2 1.0 Upflow 0.5 0 0.0 0 6000 x1 [m] | 250 [±C] 200 Temperature 150 100 50 0 |
| --- | --- | --- |


**[Table p18.4]**
|  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |

[CAPTION] Figure 9
The mesh of the synthetic reservoir model, and the locations of each production well.

[CAPTION] Figure 10
The true permeability structure (left), mass upflows (centre) and natural state con-


<!-- page 19 -->
0
6000
x1 [m]
0
6000
x2 [m]
0
6000
x1 [m]
0
6000
x1 [m]
0
6000
x1 [m]
0
1
2
Upﬂow [kg s−1 m−2]
×10−4
Figure 11
Mass upflows sampled from the prior.
Figure 12
The top surfaces (top row) and bottom surfaces (bottom row) of clay cap geometries
sampled from the prior.
pressure of 1 bar and a temperature of 20◦C; this represents an atmospheric boundary condition.
The side boundaries are closed. We impose a background heat flux of 200 mW m−2 through all
cells on the bottom boundary except for those which are associated with a mass upflow. All
of the mass upflow entering through the bottom boundary is associated with an enthalpy of
1500 kJ kg−1.
We assume that there is a single linear, vertical fault running through the reservoir from
east to west. However, we treat the exact location of the fault, as well as the magnitude of
the mass upflow contained within it, as unknown. We assume that the points at which the
fault intersects the eastern and western boundaries of the model domain are independent and
uniformly distributed between 1500 m and 4500 m, and that the upflow within each cell along the
fault is modelled using a Gaussian random field with a squared-exponential covariance function
(see Eq. 25) and a mean and standard deviation that reduce as the horizontal distance to the
centre of the model domain increases, reflecting a prior belief that the upflow tends to be greatest
near the centre of the model domain. The parameters of this GRF are chosen such that the total
mass upflow entering the model domain tends to be between 80 kg s−1 and 120 kg s−1. Figure
11 shows several samples of mass upflows drawn from the prior.
To parametrise the permeability and porosity of the reservoir, we first partition the domain
of the model into three regions with variable interfaces: the fault (which has a high permeability
and porosity), a clay cap (which has a low permeability and porosity), and a background region
(which has a moderate permeability and porosity). We model the clay cap as the deformation of
a star-shaped set, with a boundary represented using a truncated Fourier series with uncertain
coefficients. Figure 12 shows several clay cap geometries drawn from the prior; for a complete
description of this parametrisation, the reader is referred to de Beer (2024b). Where the clay
cap and fault intersect, the clay cap takes priority.
19

[CAPTION] Figure 11
Mass upflows sampled from the prior.

[CAPTION] Figure 12
The top surfaces (top row) and bottom surfaces (bottom row) of clay cap geometries


<!-- page 20 -->
0
°17
°16
°15
°14
°13
log10(Perm) [log10(m2)]
Figure 13
Permeability structures sampled from the prior.
To model the permeability and porosity within each region, we use the level set method
(Iglesias et al., 2016), which is often used in the modelling of subsurface systems to generate
distinct regions with common geophysical characteristics (see, e.g., Muir and Tsai, 2020; Tso
et al., 2021). These regions are defined using the contours of a continuous function, referred to
as the level set function, which we denote using φ(·). In each region, we first specify a set of
rock types with varying permeabilities and porosities, then select a set of constants at which to
threshold the level set function to produce each rock type. For instance, the permeability and
porosity within the fault region are given by
(κ(x), ϕ(x)) =





(10−13.5 m2, 0.20),
φ(x) < −0.5,
(10−13.0 m2, 0.25),
−0.5 ≤φ(x) < 0.5,
(10−12.5 m2, 0.30),
0.5 ≤φ(x).
(27)
The permeability and porosity of each of the other regions is defined similarly. We allow the
permeability of the clay cap to vary between 10−17 m2 and 10−16 m2, and the porosity to vary
between 0.05 and 0.10. We allow the permeability of the background region to vary between
10−15.5 m2 and 10−13.5 m2, and the porosity to vary between 0.10 and 0.20.
In all regions,
we choose the level set function to be a centred GRF with a squared-exponential covariance
function, with a standard deviation chosen such that the prior probabilities of a given location
within each region belonging to each rock type are approximately equal. The lengthscale of each
level set function in the horizontal (x1 and x2 directions) is 8000 m, while the lengthscale in the
vertical (x3) direction is 2000 m. This tends to result in the generation of layered structures.
Figure 13 shows several permeability structures sampled from the prior.
3.3.3
Data
We assume that we have access to measurements of the natural state temperature at seven
equispaced points down each of wells 1–7, as well as measurements of the pressure and enthalpy
of the fluid extracted from each well at three-month intervals over the first year of the production
period. This gives a total of 119 measurements. We add independent Gaussian noise to each
measurement, with a standard deviation equal to 2% of the maximum value of the corresponding
data type. Figure 14 shows the data collected at well 1. No data is collected at well 8 or well 9.
3.3.4
Simulation
All simulations are carried out using the open-source simulator Waiwera (Croucher et al., 2020),
which uses a finite volume discretisation of the governing equations in Section 3.1, and are
run in parallel on a high-performance computing cluster provided by New Zealand eScience
Infrastructure. As in our previous model problem, to avoid the “inverse crime” of generating
the synthetic data and solving the inverse problem using the same numerical discretisation
(Kaipio and Somersalo, 2006; Kaipio and Somersalo, 2007), we use a mesh comprised of 13,383
cells when simulating the dynamics of the true system (generated using a draw from the prior),
20


**[Table p20.1]**
|  | [log10(m2)] °13 °14 °15 log10(Perm) °16 °17 0 |
| --- | --- |
|  |  |

[CAPTION] Figure 13
Permeability structures sampled from the prior.

[CAPTION] Figure 13 shows several permeability structures sampled from the prior.


<!-- page 21 -->
0
200
Temperature [◦C]
−2600
−450
Elevation [m]
0
1
2
Time [Years]
5
8
Pressure [MPa]
0
1
2
Time [Years]
300
600
Enthalpy [kJ kg−1]
Figure 14
The natural state downhole temperature data (left), transient pressure data (centre)
and transient enthalpy data (right) collected at well 2. In all plots, the solid line
denotes the true reservoir state and the dots denote noisy observations. In the pressure
and enthalpy plots, the grey region denotes the forecast period.
but a mesh comprised of 8788 cells (plotted in Figure 9) when carrying out the simulations
required to approximate the posterior predictive distribution using DSI.
For the DSI algorithm, we run 2000 simulations using sets of parameters drawn from the
prior. As is common in geothermal reservoir modelling, some of these converge to physically
unreasonable values (for example, the reservoir pressure reduces below atmospheric pressure),
while others do not converge entirely. After discarding these, we are left with 1413 simulations.
We use 1300 of these as part of the DSI algorithm. Of the remaining samples, 100 are used to
evaluate the quality of the DSI mapping; we elaborate on this further in the next section.
3.3.5
Validation
As in the two-dimensional case study, we first evaluate the quality of the DSI mapping, T (·), by
plotting a set of (unconditional) realisations from the DSI approximation to the prior predictive
distribution of the downhole temperature profiles at the end of the production history period,
and the transient feedzone pressure and enthalpy of the system, at wells 6 and 8. These results
are shown in Figure 15, which also functions as a prior predictive check. Again, we observe that
the DSI predictions exhibit similar qualitative behaviour to the prior predictions.
As an additional form of validation, we evaluate the quality of the DSI surrogate using
100 of the samples from the prior predictive distribution of the data and predictive quantities of
interest that were not used to construct it. For each of these “validation” datasets, we draw 1000
samples from the DSI approximation to the posterior predictive distribution. We then compute
the proportion of each of the true values of the predictive quantities of interest (comprised of the
downhole temperature profiles down each well at the end of the production history period, and
the transient pressure and enthalpy at each feedzone over the forecast period) that are contained
within the central 95 percent of the predictions. We find that, on average, 92.0 percent of each
of the predicted temperature profiles, 93.2 percent of each of the predicted pressure profiles, and
92.3 percent of the predicted enthalpy profiles are contained within the central 95 percent of
the predictions. This gives us confidence that the DSI algorithm is able to reduce our level of
uncertainty in the values of these predictive quantities of interest without discounting the true
values.
3.3.6
Results
We now discuss the results obtained when DSI is applied to the “true” system plotted in Figure
10. Figure 16 shows samples from the prior predictive distribution, and the approximation to the
posterior predictive distribution obtained using DSI, for the downhole temperature profiles at
the end of the production history period, and the transient feedzone pressure and enthalpy over
21

[CAPTION] Figure 14
The natural state downhole temperature data (left), transient pressure data (centre)


<!-- page 22 -->
20
170
320
Temperature [◦C]
−2600
−1525
−450
Elevation [m]
Prior
20
170
320
Temperature [◦C]
DSI (Unconditional)
20
170
320
Temperature [◦C]
Prior
20
170
320
Temperature [◦C]
DSI (Unconditional)
0
1
2
Time [Years]
0.5
5.5
10.5
Pressure [MPa]
0
1
2
Time [Years]
0
1
2
Time [Years]
0
1
2
Time [Years]
0
1
2
Time [Years]
100
800
1500
Enthalpy [kJ kg−1]
0
1
2
Time [Years]
0
1
2
Time [Years]
0
1
2
Time [Years]
WELL 3
WELL 6
Figure 15
A set of 100 samples of the downhole temperatures at the end of the production
history period, and the transient feedzone pressure and enthalpy, in well 6 (first and
second columns) and well 8 (third and fourth columns). For both wells, samples from
the prior predictive distribution and the DSI approximation to the prior predictive
distribution are shown.
22

[CAPTION] Figure 15
A set of 100 samples of the downhole temperatures at the end of the production


<!-- page 23 -->
the production history and forecast periods, in well 3 and well 4. Results for other wells at which
data is collected are similar. In all cases, applying the DSI algorithm gives a significant reduction
in uncertainty, and the state of the true system generally has high posterior probability.
Figure 17 shows samples of the same quantities as Figure 16, but for well 8 and well 9, at
which no data is collected, and which only operate during the forecast period. In both cases, the
uncertainty in each predictive quantity of interest is reduced after applying the DSI algorithm,
and the state of the true system has high posterior probability. However, these reductions in
uncertainty are significantly less than the corresponding reductions in uncertainty for well 3 and
well 4; this is expected, given that we do not have access to any direct information on the state
of the reservoir down these wells.
Finally, Figure 18 shows how the posterior predictive distributions for the temperature at
the bottom of each well at the end of the production history period, and the feedzone pressure
and enthalpy at the end of the forecast period change as the number of samples used to estimate
the mapping T (·) varies, for wells 3, 4, 8 and 9. As in the two-dimensional setting, we observe
that, when 100 samples are used, the resulting estimates appear significantly different to those
obtained using a greater number of samples. Additionally, they often fail to capture the truth
with non-negligible probability. When 500 samples are used, the resulting estimates appear to
be fairly consistent with those obtained using larger sample sizes, suggesting that 500 or more
samples is likely to be an appropriate number to use for this particular problem.
4
CONCLUSIONS AND FUTURE WORK
In this work, we have introduced a simple variant of the data space inversion methodology that
allows for efficient, approximate characterisation of the posterior predictive distribution. We
have demonstrated that the resulting methodology can provide an efficient, derivative-free means
of making geothermal reservoir model predictions, with quantified uncertainty, conditioned on
observed data.
We have also illustrated how the approximation to the posterior predictive
distribution generated using DSI changes as the number of samples from the prior that are used
as part of the algorithm is varied, and provided a systematic comparison of DSI and linearisation
about the MAP estimate, another commonly-used technique for approximate Bayesian inference.
We emphasise that there are a number of limitations of the DSI framework. Like most sur-
rogate modelling techniques, there exists little in the way of theoretical guarantees surrounding
the quality of the DSI approximation to the posterior predictive distribution when applied to
general inverse problems, and there is no guarantee that results obtained using DSI will re-
spect the physics of the problem under consideration. There are also some elements of the DSI
framework that warrant further study, including the sensitivity of the method to the choice of
prior used, the dimension of the data, and the particular characteristics of the forward problem
under consideration. Nonetheless, the empirical results we have obtained in this work appear
promising.
An obvious next step will be to demonstrate the application of the DSI methodology to a real-
world case study arising in geothermal reservoir modelling. A model of a real-world geothermal
system is likely to require a more complex prior than we have used in our model problems; for
instance, it is likely to need to account for the full, anisotropic permeability structure of the
reservoir. There may also be additional data to consider, such as CO2 fractions or information
on surface features. However, we anticipate that the application of the DSI framework will
remain much the same.
Additionally, we have identified a variety of extensions of the DSI framework that would be
valuable to explore in a geothermal setting. First, it is worth noting that while it is generally
predictive quantities that are of most interest in a geothermal setting, it is also possible to
use DSI to approximate the solution to the calibration problem (see Sec. 2.2.1); that is, the
problem of estimating the model parameters. This simply amounts to using the samples of the
23

[CAPTION] Figure 17 shows samples of the same quantities as Figure 16, but for well 8 and well 9, at


<!-- page 24 -->
20
170
320
Temperature [◦C]
−2600
−1525
−450
Elevation [m]
Prior
20
170
320
Temperature [◦C]
DSI
20
170
320
Temperature [◦C]
Prior
20
170
320
Temperature [◦C]
DSI
0
2
Time [Years]
0.5
5.5
10.5
Pressure [MPa]
0
2
Time [Years]
0
2
Time [Years]
0
2
Time [Years]
0
2
Time [Years]
100
800
1500
Enthalpy [kJ kg−1]
0
2
Time [Years]
0
2
Time [Years]
0
2
Time [Years]
WELL 3
WELL 4
Figure 16
The downhole temperatures at the end of the production history period, and the
transient feedzone pressure and enthalpy, in well 3 (first and second columns) and
well 4 (third and fourth columns). For both wells, samples from the prior predictive
distribution and the approximation to the posterior predictive distribution computed
using DSI are shown. In all plots, the black line denotes the true reservoir state and
the grey lines indicate the central 95% of the samples. In the pressure and enthalpy
plots, the black dots denote the noisy observations and the grey region denotes the
forecast period.
24

[CAPTION] Figure 16
The downhole temperatures at the end of the production history period, and the


<!-- page 25 -->
20
170
320
Temperature [◦C]
−2600
−1475
−350
Elevation [m]
Prior
20
170
320
Temperature [◦C]
DSI
20
170
320
Temperature [◦C]
Prior
20
170
320
Temperature [◦C]
DSI
1
2
Time [Years]
0.5
5.5
10.5
Pressure [MPa]
1
2
Time [Years]
1
2
Time [Years]
1
2
Time [Years]
1
2
Time [Years]
100
800
1500
Enthalpy [kJ kg−1]
1
2
Time [Years]
1
2
Time [Years]
1
2
Time [Years]
WELL 8
WELL 9
Figure 17
The downhole temperatures at the end of the production history period, and the
transient feedzone pressure and enthalpy, in well 8 (first and second columns) and well
9 (third and fourth columns), over the forecast period. For both wells, samples from
the prior predictive distribution and the approximation to the posterior predictive
distribution computed using DSI are shown. In all plots, the black line denotes the
true reservoir state and the grey lines indicate the central 95% of the samples.
25

[CAPTION] Figure 17
The downhole temperatures at the end of the production history period, and the


<!-- page 26 -->
20
80
Temperature [◦C]
0.0
0.3
Probability Density
WELL 3
220
300
Temperature [◦C]
0.0
0.1
WELL 4
50
280
Temperature [◦C]
0.00
0.05
WELL 8
0
100
Temperature [◦C]
0.0
0.1
WELL 9
3
8
Pressure [MPa]
0
2
Probability Density
5
10
Pressure [MPa]
0
5
2
8
Pressure [MPa]
0
3
3
8
Pressure [MPa]
0
1
400
700
Enthalpy [kJ kg−1]
0.00
0.05
Probability Density
1000
1200
Enthalpy [kJ kg−1]
0.0
0.1
300
800
Enthalpy [kJ kg−1]
0.00
0.01
200
800
Enthalpy [kJ kg−1]
0.00
0.02
Prior
Truth
DSI (J = 100)
DSI (J = 250)
DSI (J = 500)
DSI (J = 1000)
DSI (J = 1300)
Figure 18
The prior predictive distribution and posterior predictive distributions of the temper-
ature at the bottom of the well at the end of the production history period, and the
feedzone pressure and enthalpy at the end of the forecast period, for wells 3, 4, 8 and
9. The black line in each plot denotes the true reservoir state.
26


**[Table p26.1]**
|  |  |
| --- | --- |

[CAPTION] Figure 18
The prior predictive distribution and posterior predictive distributions of the temper-


<!-- page 27 -->
parameters instead of the samples of the predictive quantities of interest when building the DSI
surrogate; the process of drawing samples from the resulting approximation to the (parameter)
posterior remain the same. We note that these ideas are used in the work of Park and Caers
(2020), which employs elements of the Bayesian evidential learning framework. It would be of
interest to apply ideas from DSI to estimate the parameters of a geothermal reservoir model, and
to compare the results obtained to those generated using other surrogate modelling approaches
(see, e.g., Han et al., 2024; Han et al., 2025).
Additionally, the DSI framework is likely to be well-suited to solving optimal experimental
design problems, in which one is interested in identifying a data collection plan that minimises a
measure of the expected posterior uncertainty in the model parameters or predictive quantities
of interest (Alexanderian, 2021).
Solving an OED problem generally requires the repeated
computation of the parameter posterior or posterior predictive distribution associated with sets
of possible data one could expect to collect; because the DSI framework uses the same set
of simulations from the prior when approximating these distributions, regardless of the data
collected, solving an OED problem would require no more reservoir model simulations than
solving a single inverse problem. For similar ideas within the context of the Bayesian evidential
learning framework, see Thibaut et al. (2021).
A final area of interest is optimal control. The application of DSI to optimal control problems
is studied in Jiang et al. (2020), which uses an extension of the DSI framework to optimise the
management of oil reservoirs, by treating user-specified well controls as “data” to be conditioned.
This allows for the efficient approximation of the posterior predictive distribution under various
management scenarios.
Overall, we believe that our results demonstrate that data space inversion has the potential
to be a useful tool in geothermal reservoir modelling, and should be investigated further.
ACKNOWLEDGMENTS
The authors wish to acknowledge the use of New Zealand eScience Infrastructure (NeSI; www.
nesi.org.nz) high performance computing facilities as part of this research. These facilities
are funded jointly by NeSI’s collaborator institutions and through the Ministry of Business,
Innovation & Employment’s Research Infrastructure programme. The authors would like to
thank the MBIE research programme “Empowering Geothermal” which has in part funded this
research. Finally, the authors wish to thank John Doherty for several fruitful discussions on
the data space inversion methodology, and the two anonymous reviewers, whose comments have
improved this paper significantly.
OPEN RESEARCH SECTION
The code and models used to carry out the experiments and generate the figures in this paper
are archived on Zenodo (de Beer, 2024a), and are also available on GitHub (https://github.
com/alexgdebeer/GeothermalDSI) under the MIT license.
REFERENCES
Alexanderian, A (2021). Optimal experimental design for infinite-dimensional Bayesian inverse prob-
lems governed by PDEs: A review. Inverse Problems 37, 043001. doi: https://doi.org/10.1088/
1361-6420/abe10c.
Aristoff, D & Bangerth, W (2023). A benchmark for the Bayesian inversion of coefficients in partial
differential equations. SIAM Review 65, 1074–1105. doi: https://doi.org/10.1137/21M1399464.
27


<!-- page 28 -->
Athens, ND & Caers, JK (2019). A Monte Carlo-based framework for assessing the value of information
and development risk in geothermal exploration. Applied Energy 256, 113932. doi: https://doi.
org/10.1016/j.apenergy.2019.113932.
B´ek´esi, E, Struijk, M, Bont´e, D, Veldkamp, H, Limberger, J, Fokker, PA, Vrijlandt, M, &
Wees, JD van (2020). An updated geothermal model of the Dutch subsurface based on inversion
of temperature data. Geothermics 88, 101880. doi: https://doi.org/10.1016/j.geothermics.
2020.101880.
Bjarkason, EK, Maclaren, OJ, Nicholson, R, Yeh, A, & O’Sullivan, MJ (2020). Uncertainty
quantification of highly-parameterized geothermal reservoir models using ensemble-based methods.
Proc. World Geothermal Congress. url: https://hdl.handle.net/2292/65029.
Bjarkason, EK, Maclaren, OJ, O’Sullivan, JP, & O’Sullivan, MJ (2018). Randomized truncated
SVD Levenberg-Marquardt approach to geothermal natural state and history matching. Water Re-
sources Research 54, 2376–2404. doi: https://doi.org/10.1002/2017WR021870.
Bjarkason, EK, O’Sullivan, JP, Yeh, A, & O’Sullivan, MJ (2019). Inverse modeling of the natural
state of geothermal reservoirs using adjoint and direct methods. Geothermics 78, 85–100. doi:
https://doi.org/10.1016/j.geothermics.2018.10.001.
Calvetti, D & Somersalo, E (2007). An introduction to Bayesian scientific computing: ten lectures
on subjective computing. Springer New York. doi: https://doi.org/10.1007/978-0-387-73394-4.
Chen, V, Dunlop, MM, Papaspiliopoulos, O, & Stuart, AM (2019). Dimension-robust MCMC in
Bayesian inverse problems. arXiv: 1803.03344.
Chen, Y & Oliver, DS (2013). Levenberg–Marquardt forms of the iterative ensemble smoother for
efficient history matching and uncertainty quantification. Computational Geosciences 17, 689–703.
doi: https://doi.org/10.1007/s10596-013-9351-5.
Chen, Z (2007). Reservoir simulation: mathematical techniques in oil recovery. SIAM. doi: https:
//doi.org/10.1137/1.9780898717075.
Christie, MA & Blunt, MJ (2001). Tenth SPE comparative solution project: A comparison of upscal-
ing techniques. SPE Reservoir Evaluation & Engineering 4, 308–317. doi: https://doi.org/10.
2118/72469-PA.
Cotter, SL, Roberts, GO, Stuart, AM, & White, D (2013). MCMC methods for functions: modi-
fying old algorithms to make them faster. Statistical Science 28, 424–446. doi: https://doi.org/
10.1214/13-STS421.
Croucher, A, O’Sullivan, M, O’Sullivan, J, Yeh, A, Burnell, J, & Kissling, W (2020). Waiwera:
A parallel open-source geothermal flow simulator. Computers & Geosciences 141, 104529. doi:
https://doi.org/10.1016/j.cageo.2020.104529.
Cui, T, Fox, C, & O’Sullivan, MJ (2011). Bayesian calibration of a large-scale geothermal reser-
voir model by a new adaptive delayed acceptance Metropolis Hastings algorithm. Water Resources
Research 47. doi: https://doi.org/10.1029/2010WR010352.
Cui, T, Fox, C, & O’Sullivan, MJ (2019). A posteriori stochastic correction of reduced models in
delayed-acceptance MCMC, with application to multiphase subsurface inverse problems. Interna-
tional Journal for Numerical Methods in Engineering 118, 578–605. doi: https://doi.org/10.
1002/nme.6028.
de Beer, A (2024a). alexgdebeer/GeothermalDSI. Version 0.5.0. Software. doi: https://doi.org/10.
5281/zenodo.12193947.
de Beer, A (2024b). Ensemble Methods for Geothermal Inverse Problems. Master’s thesis. University
of Auckland. url: https://hdl.handle.net/2292/68150.
Delottier, H, Doherty, J, & Brunner, P (2023). Data space inversion for efficient uncertainty
quantification using an integrated surface and subsurface hydrologic model. Geoscientific Model
Development Discussions 16, 1–30. doi: https://doi.org/10.5194/gmd-16-4213-2023.
Emerick, AA & Reynolds, AC (2013). Ensemble smoother with multiple data assimilation. Computers
& Geosciences 55, 3–15. doi: https://doi.org/10.1016/j.cageo.2012.03.011.
Gelman, A, Carlin, JB, Stern, HS, Dunson, DB, Vehtari, A, & Rubin, DB (2013). Bayesian data
analysis. 3rd Edition. CRC Press. doi: https://doi.org/10.1201/b16018.
Gelman, A, Vehtari, A, Simpson, D, Margossian, CC, Carpenter, B, Yao, Y, Kennedy, L,
Gabry, J, B¨urkner, PC, & Modr´ak, M (2020). Bayesian workflow. arXiv: 2011.01808.
28


<!-- page 29 -->
Golub, GH & Van Loan, CF (2013). Matrix computations. 4th Edition. JHU press. doi: https:
//doi.org/10.1137/1.9781421407944.
Gonzalez-Gutierrez, B, Sung, S, Nicholson, R, O’Sullivan, JP, O’Sullivan, MJ, & Maclaren,
O (2018). Accelerating the solution of geothermal inverse problems using adjoint methods in Wai-
wera: A progress report. Proc. 40th New Zealand Geothermal Workshop.
Haber, E & Hanson, L (2007). Model problems in PDE-constrained optimization. Tech. rep. Atlanta,
Georgia: Emory University.
Han, Y, Hamon, FP, & Durlofsky, LJ (2025). Accelerated training of deep learning surrogate models
for surface displacement and flow, with application to MCMC-based history matching of CO2 storage
operations. Geoenergy Science and Engineering 246, 213589. doi: https://doi.org/10.1016/j.
geoen.2024.213589.
Han, Y, Hamon, FP, Jiang, S, & Durlofsky, LJ (2024). Surrogate model for geological CO2 storage
and its use in hierarchical MCMC history matching. Advances in Water Resources 187, 104678.
doi: https://doi.org/10.1016/j.cageo.2020.104567.
Hermans, T, Nguyen, F, Klepikova, M, Dassargues, A, & Caers, J (2018). Uncertainty quantifica-
tion of medium-term heat storage from short-term geophysical experiments using Bayesian evidential
learning. Water Resources Research 54, 2931–2948. doi: https://doi.org/10.1002/2017WR022135.
Iglesias, MA, Law, KJ, & Stuart, AM (2013). Ensemble Kalman methods for inverse problems.
Inverse Problems 29, 045001. doi: https://doi.org/10.1088/0266-5611/29/4/045001.
Iglesias, MA, Lu, Y, & Stuart, AM (2016). A Bayesian level set method for geometric inverse
problems. Interfaces and Free Boundaries 18, 181–217. doi: https://doi.org/10.4171/IFB/362.
Jiang, S & Durlofsky, LJ (2021). Treatment of model error in subsurface flow history matching using
a data-space method. Journal of Hydrology 603, 127063. doi: https://doi.org/10.1016/j.
jhydrol.2021.127063.
Jiang, S & Durlofsky, LJ (2024). History matching for geological carbon storage using data-space
inversion with spatio-temporal data parameterization. International Journal of Greenhouse Gas
Control 134, 104124. doi: https://doi.org/10.1016/j.ijggc.2024.104124.
Jiang, S, Hui, MH, & Durlofsky, LJ (2021). Data-space inversion with a recurrent autoencoder
for naturally fractured systems. Frontiers in Applied Mathematics and Statistics 7, 686754. doi:
https://doi.org/10.3389/fams.2021.686754.
Jiang, S, Sun, W, & Durlofsky, LJ (2020). A data-space inversion procedure for well control op-
timization and closed-loop reservoir management. Computational Geosciences 24, 361–379. doi:
https://doi.org/10.1007/s10596-019-09853-4.
Kaipio, J & Somersalo, E (2006). Statistical and computational inverse problems. Springer New York.
doi: https://doi.org/10.1007/b138659.
Kaipio, J & Somersalo, E (2007). Statistical inverse problems: Discretization, model reduction and
inverse crimes. Journal of Computational and Applied Mathematics 198, 493–504. doi: https :
//doi.org/10.1016/j.cam.2005.09.027.
Kitanidis, PK (1995). Quasi-linear geostatistical theory for inversing. Water Resources Research 31,
2411–2419. doi: https://doi.org/10.1029/95WR01945.
Lima, MM, Emerick, AA, & Ortiz, CE (2020). Data-space inversion with ensemble smoother. Com-
putational Geosciences 24, 1179–1200. doi: https://doi.org/10.1007/s10596-020-09933-w.
Liu, D, Rao, X, Zhao, H, Xu, YF, & Gong, RX (2021). An improved data space inversion method to
predict reservoir state fields via observed production data. Petroleum Science 18, 1127–1142. doi:
https://doi.org/10.1016/j.petsci.2021.07.008.
Maclaren, OJ, Nicholson, R, Bjarkason, EK, O’Sullivan, JP, & O’Sullivan, MJ (2020). Incor-
porating posterior-informed approximation errors into a hierarchical framework to facilitate out-of-
the-box MCMC sampling for geothermal inverse problems and uncertainty quantification. Water
Resources Research 56, e2018WR024240. doi: https://doi.org/10.1029/2018WR024240.
Marzouk, Y, Moselhy, T, Parno, M, & Spantini, A (2016). Sampling via measure transport: An
introduction. Handbook of Uncertainty Quantification. Springer, 1–41. doi: https://doi.org/10.
1007/978-3-319-11259-6_23-1.
Michel, H, Nguyen, F, Kremer, T, Elen, A, & Hermans, T (2020). 1D geological imaging of the
subsurface from geophysical data with Bayesian Evidential Learning. Computers & Geosciences
138, 104456. doi: https://doi.org/10.1016/j.cageo.2020.104456.
29


<!-- page 30 -->
Muir, JB & Tsai, VC (2020). Geometric and level set tomography using ensemble Kalman inversion.
Geophysical Journal International 220, 967–980. doi: https://doi.org/10.1093/gji/ggz472.
O’Sullivan, MJ & O’Sullivan, JP (2016). “Reservoir modeling and simulation for geothermal resource
characterization and evaluation”. Geothermal power generation. Elsevier, 165–199. doi: https://
doi.org/10.1016/B978-0-08-100337-4.00007-3.
O’Sullivan, J, Croucher, A, Yeh, A, & O’Sullivan, M (2013). Improved convergence for air-water
and CO2-water TOUGH2 simulations. Proc. 35th New Zealand Geothermal Workshop.
Oliver, DS, He, N, & Reynolds, AC (1996). Conditioning permeability fields to pressure data. Proc.
5th European conference on the mathematics of oil recovery. European Association of Geoscientists
& Engineers. doi: https://doi.org/10.3997/2214-4609.201406884.
Omagbon, J, Doherty, J, Yeh, A, Colina, R, O’Sullivan, J, McDowell, J, Nicholson, R, Ma-
claren, OJ, & O’Sullivan, M (2021). Case studies of predictive uncertainty quantification for
geothermal models. Geothermics 97, 102263. doi: https://doi.org/10.1016/j.geothermics.
2021.102263.
Park, J & Caers, J (2020). Direct forecasting of global and spatial model parameters from dynamic
data. Computers & Geosciences 143, 104567. doi: https://doi.org/10.1016/j.cageo.2020.
104567.
Petra, N & Stadler, G (2011). Model variational inverse problems governed by partial differential
equations. Tech. rep. Austin, Texas: Institute for Computational Engineering and Sciences, Univer-
sity of Texas at Austin.
Power, A, Wong, D, Dekkers, K, Gravatt, M, Maclaren, OJ, O’Sullivan, JP, O’Sullivan, MJ,
& Nicholson, R (2022). Data-space inversion for efficient geothermal reservoir model predictions
and uncertainty quantification. Proc. 43rd New Zealand Geothermal Workshop.
Pradhan, A & Mukerji, T (2020). Seismic Bayesian evidential learning: Estimation and uncertainty
quantification of sub-resolution reservoir properties. Computational Geosciences 24, 1121–1140. doi:
https://doi.org/10.1007/s10596-019-09929-1.
Scheidt, C, Li, L, & Caers, J (2018). Quantifying uncertainty in subsurface systems. John Wiley &
Sons. doi: https://doi.org/10.1002/9781119325888.
Scott, SW, O’Sullivan, JP, Maclaren, OJ, Nicholson, R, Covell, C, Newson, J, & Gudj´onsd´ott´ır,
MS (2022). Bayesian calibration of a natural state geothermal reservoir model, Krafla, North Iceland.
Water Resources Research 58, e2021WR031254. doi: https://doi.org/10.1029/2021WR031254.
Sun, W & Durlofsky, LJ (2017). A new data-space inversion procedure for efficient uncertainty
quantification in subsurface flow problems. Mathematical Geosciences 49, 679–715. doi: https:
//doi.org/10.1007/s11004-016-9672-8.
Sun, W & Durlofsky, LJ (2019). Data-space approaches for uncertainty quantification of CO2 plume
location in geological carbon storage. Advances in water resources 123, 234–255. doi: https://doi.
org/10.1016/j.advwatres.2018.10.028.
Sun, W, Hui, MH, & Durlofsky, LJ (2017). Production forecasting and uncertainty quantification
for naturally fractured reservoirs using a new data-space inversion procedure. Computational Geo-
sciences 21, 1443–1458. doi: https://doi.org/10.1007/s10596-017-9633-4.
Tarantola, A (2005). Inverse problem theory and methods for model parameter estimation. SIAM.
doi: https://doi.org/10.1137/1.9780898717921.
Thibaut, R, Laloy, E, & Hermans, T (2021). A new framework for experimental design using Bayesian
Evidential Learning: The case of wellhead protection area. Journal of Hydrology 603, 126903. doi:
https://doi.org/10.1016/j.jhydrol.2021.126903.
Tian, X, Volkov, O, & Voskov, D (2024). An advanced inverse modeling framework for efficient
and flexible adjoint-based history matching of geothermal fields. Geothermics 116, 102849. doi:
https://doi.org/10.1016/j.geothermics.2023.102849.
Tso, CHM, Iglesias, M, Wilkinson, P, Kuras, O, Chambers, J, & Binley, A (2021). Efficient
multiscale imaging of subsurface resistivity with uncertainty quantification using ensemble Kalman
inversion. Geophysical Journal International 225, 887–905. doi: https://doi.org/10.1093/gji/
ggab013.
T¨ureyen, ¨O˙I, Kırmacı, A, & Onur, M (2014). Assessment of uncertainty in future performance
predictions by lumped-parameter models for single-phase liquid geothermal systems. Geothermics
51, 300–311. doi: https://doi.org/10.1016/j.geothermics.2014.01.015.
30


<!-- page 31 -->
Wackernagel, H (2003). Multivariate geostatistics: An introduction with applications. Springer Science
& Business Media. doi: https://doi.org/10.1007/978-3-662-05294-5.
Williams, CK & Rasmussen, CE (2006). Gaussian processes for machine learning. MIT Press. doi:
https://doi.org/10.7551/mitpress/3206.001.0001.
Zhang, Z, Jafarpour, B, & Li, L (2014). Inference of permeability heterogeneity from joint inversion
of transient flow and temperature data. Water Resources Research 50, 4710–4725. doi: https:
//doi.org/10.1002/2013WR013801.
31