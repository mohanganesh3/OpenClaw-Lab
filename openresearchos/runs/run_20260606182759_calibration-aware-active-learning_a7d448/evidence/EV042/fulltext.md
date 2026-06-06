<!-- page 1 -->
arXiv:2001.10827v3  [math.NA]  12 Sep 2020
Stochastic Analysis of an Adaptive Cubic Regularisation Method under
Inexact Gradient Evaluations and Dynamic Hessian Accuracy
Stefania Bellaviaa and Gianmarco Guriolib
a Dipartimento di Ingegneria Industriale, Universit`a degli Studi, Firenze, Italy; b Dipartimento di
Matematica e Informatica “Ulisse Dini”, Universit`a degli Studi, Firenze, Italy.
ARTICLE HISTORY
Compiled September 15, 2020
ABSTRACT
We here adapt an extended version of the adaptive cubic regularisation method with dynamic
inexact Hessian information for nonconvex optimisation in [3] to the stochastic optimisation
setting. While exact function evaluations are still considered, this novel variant inherits the
innovative use of adaptive accuracy requirements for Hessian approximations introduced
in [3] and additionally employs inexact computations of the gradient. Without restrictions
on the variance of the errors, we assume that these approximations are available within a
suﬃciently large, but ﬁxed, probability and we extend, in the spirit of [18], the deterministic
analysis of the framework to its stochastic counterpart, showing that the expected number of
iterations to reach a ﬁrst-order stationary point matches the well known worst-case optimal
complexity. This is, in fact, still given by O(ǫ−3/2), with respect to the ﬁrst-order ǫ tolerance.
Finally, numerical tests on nonconvex ﬁnite-sum minimisation conﬁrm that using inexact ﬁrst
and second-order derivatives can be beneﬁcial in terms of the computational savings.
KEYWORDS
Adaptive cubic regularization methods; inexact derivatives evaluations; stochastic
nonconvex optimization; worst-case complexity analysis; ﬁnite-sum minimization.
1. Introduction
Adaptive Cubic Regularisation (ARC) methods are Newton-type procedures for solving un-
constrained optimisation problems of the form
min
x∈Rn f(x),
(1.1)
in which f : Rn →R is a suﬃciently smooth, bounded below and, possibly, nonconvex
function. In the seminal work by [26] the iterative scheme of the method is based on the min-
imisation of a cubic model, relying on the Taylor series, for predicting the objective function
values, and is a globally convergent second-order procedure. The main reason to consider the
ARC framework in place of other globalisation strategies, such as Newton-type methods em-
bedded into a linesearch or a trust-region scheme, lies on its optimal complexity. In fact, given
the ﬁrst-order ǫ tolerance and assuming Lipschitz continuity of the Hessian of the objective
function, an ǫ-approximate ﬁrst-order stationary point is reached, in the worst-case, in at
CONTACT: Stefania Bellavia, Email: stefania.bellavia@uniﬁ.it


<!-- page 2 -->
2
most O(ǫ−3/2) iterations, instead of the O(ǫ−2) bound gained by trust-region and linesearch
methods [12,17]. More in depth, an (ǫ, ǫH)-approximate ﬁrst- and second-order critical point
is found in at most O(max(ǫ−3/2, ǫ−3
H )) iterations, where ǫH is the positive preﬁxed second-
order optimality tolerance [12,14,16,26]. We observe that, in [9] it has been shown that the
bound O(ǫ−3/2) for computing an ǫ-approximate ﬁrst-order stationary point is optimal among
methods operating on functions with Lipschitz continuous Hessian. Experimentally, second-
order methods can be more eﬃcient than ﬁrst-order ones on badly scaled and ill-conditioned
problems, since they take advantage of curvature information to easily escape from saddle
points to search for local minima ([11,12,33]) and this feature is in practice quite robust to
the use of inexact Hessian information. On the other hand, their per-iteration cost is expected
to be higher than ﬁrst-order procedures, due to the computation of the Hessian-vector prod-
ucts. Consequently, literature has recently focused on ARC variants with inexact derivative
information, starting from schemes employing Hessian approximations [3,21,32] though con-
serving optimal complexity. ARC methods with inexact gradient and Hessian approximations
and still preserving optimal complexity are given in [4,18,25,33–35]. These approaches have
mostly been applied to large-scale ﬁnite-sum minimisation problems
min
x∈Rn f(x) = 1
N
N
X
i=1
ϕi(x),
(1.2)
widely used in machine learning applications. In this setting, the objective function f is the
mean of N component functions ϕi : Rn →R and, hence, the evaluation of the exact deriva-
tives might be, for larger values of N, computationally expensive. In the papers cited above
the derivatives approximations are required to fulﬁl given accuracy requirements and are com-
puted by random sampling. The size of the sample is determined as to satisfy the prescribed
accuracy with a suﬃciently large preﬁxed probability exploiting the operator Bernstein in-
equality for tensors (see [30]). To deal with the nondeterministic aspects of these algorithms,
in [18,35] probabilistic models are considered and it is proved that, in expectation, optimal
complexity applies as in the deterministic case; in [3,4,21,32,34] high probability results are
given and it is shown that the optimal complexity result is restored in probability. Neverthe-
less, this latter analysis does not provide information on the behaviour of the method when
the desired accuracy levels in derivatives approximations are not fulﬁlled. With the aim of
ﬁlling this gap, we here perform the stochastic analysis of the framework in [3], where ap-
proximated Hessians are employed. To make the method more general, inexactness is allowed
in ﬁrst-order information, too. The analysis aims at bounding the expected number of itera-
tions required by the algorithm to reach a ﬁrst-order stationary point, under the assumption
that gradient and Hessian approximations are available within a suﬃciently large, but ﬁxed,
probability, recovering optimal complexity in the spirit of [18].
The rest of the paper is organised as follows. In section 1.1 we brieﬂy survey the related
works and in section 1.2 we summarise our contributions. In Section 2 we introduce a stochas-
tic ARC algorithm with inexact gradients and dynamic Hessian accuracy and state the main
assumptions on the stochastic process induced by the algorithm. Relying on several existing
results and deriving some additional outcomes, Section 3 is then devoted to perform the com-
plexity analysis of the framework, while Section 4 proposes a practical guideline to apply the
method for solving ﬁnite-sum minimisation problems. Numerical results for nonconvex ﬁnite-
sum minimisation problems are discussed in Section 5 and concluding remarks are ﬁnally
given in Section 6.
Notations. The Euclidean vector and matrix norm is denoted as ∥· ∥. Given the scalar or


<!-- page 3 -->
3
vector or matrix v, and the non-negative scalar χ, we write v = O(χ) if there is a constant
g such that ∥v∥≤gχ. Given any set S, |S| denotes its cardinality. As usual, R+ denotes the
set of positive real numbers.
1.1. Related works
The interest in ARC methods with inexact derivatives has been steadily increasing. We are
here interested in computable accuracy requirements for gradient and Hessian approximations,
preserving optimal complexity of these procedures. Focusing on the Hessian approximation, in
[16] it has been proved that optimal complexity is conserved provided that, at each iteration
k, the Hessian approximation ∇2f(xk) satisﬁes
∥(∇2f(xk) −∇2f(xk))sk∥≤χ∥sk∥2,
(1.3)
where ∇2f(xk) denotes the true Hessian at xk. The method in [25], speciﬁcally designed to
minimise ﬁnite-sum problems, assume that ∇2f(xk) satisﬁes
∥∇2f(xk) −∇2f(xk)∥≤χ∥sk∥
(1.4)
with χ a positive constant, leading to (1.3). Unfortunately, the upper bound in use depends
on the steplength ∥sk∥which is unknown when forming the Hessian approximation ∇2f(xk).
Finite-diﬀerences versions of ARC method have been investigated in [13]. The Hessian ap-
proximation satisﬁes (1.4) and its computation requires an inner-loop to meet the accuracy
requirement. This mismatch is circumvented, in practical implementations of the method in
[25], by taking the step length at the previous iteration. Hence, this approach is unreliable
when the norm of the step varies signiﬁcantly from an iteration to the other, as also noticed
in the numerical tests of [3]. To overcome this practical issue, Xu and others replace in [32]
the accuracy requirement (1.4) with
∥∇2f(xk) −∇2f(xk)∥≤χǫ,
(1.5)
where ǫ is the ﬁrst-order tolerance. This provides them with ∥(∇2f(xk) −∇2f(xk))sk∥≤
χǫ∥sk∥, used to prove optimal complexity. In this situation, the estimate ∇2f(xk) is practically
computable, independently of the step length, but at the cost of a very restrictive accuracy
requirement (it is deﬁned in terms of the ǫ tolerance) to fulﬁl at each iteration of the method.
We further note that, in [31], optimal complexity results for a cubic regularisation method
employing the implementable condition
∥∇2f(xk) −∇2f(xk)∥≤χ∥sk−1∥
(1.6)
are given under the assumption that the constant regularisation parameter is greater than
the Hessian Lipschitz constant. Then, the knowledge of the Lipschitz constant is assumed.
Such an assumption can be quite stringent, especially when minimising nonconvex objective
functions. On the contrary, adaptive cubic regularisation frameworks get rid of the Lipschitz
constant, trying to overestimate it by an adaptive procedure that is well deﬁned provided
that the approximated Hessian is accurate enough. To our knowledge, accuracy requirements
depending on the current step, as those in (1.3)-(1.5), are needed to prove that the step
acceptance criterion is well-deﬁned and the regularisation parameter is bounded above.


<!-- page 4 -->
4
Regarding the gradient approximation, the accuracy requirement in [13,25] has the following
form
∥∇f(xk) −∇f(xk)∥≤µ∥sk∥2,
(1.7)
where ∇f(xk) denotes the gradient approximation and µ is a positive constant. Then, the
accuracy requirement depends on the norm of the step again.
In [34], as for the Hessian approximation, in order to get rid of the norm of the step, a very
tight accuracy requirement in used as the absolute error has to be of the order of ǫ2 at each
iteration, i.e.
∥∇f(xk) −∇f(xk)∥≤µǫ2.
(1.8)
As already noticed, in [32,34], a complexity analysis in high probability is carried out in
order to cover the situation where accuracy requirements (1.5) and (1.8) are satisﬁed only
with a suﬃciently large probability. While the behaviour of cubic regularisation approaches
employing approximated derivatives is analysed in expectation in [18], assuming that (1.3)
and (1.7) are satisﬁed with high probability. In the ﬁnite-sum minimisation context, accuracy
requirements (1.3), (1.4) and (1.7) can be enforced with high probability by subsampling via
an inner iterative process. Namely, the approximated derivative is computed using a predicted
accuracy, the step sk is computed and, if the predicted accuracy is larger than the required
accuracy, the predicted accuracy is progressively decreased (and the sample size progressively
increased) until the accuracy requirement is satisﬁed.
The cubic regularisation variant proposed in [3] employs exact gradient and ensures condition
(1.3), avoiding the above vicious cycle, requiring that
∥∇2f(xk) −∇2f(xk)∥≤ck,
(1.9)
where the guideline for choosing ck is as follows:
ck ≤
 
c,
c > 0,
if
∥sk∥≥1,
α(1 −β)∥∇f(xk)∥,
if
∥sk∥< 1,
(1.10)
with 0 ≤α < 2
3 and 0 < β < 1. Note that, for a suﬃciently large constant c, the accuracy
requirement ck can be less stringent than ǫ when ∥sk∥≥1 or, otherwise, as long as α(1 −
β)∥∇f(xk)∥> ǫ. Despite condition (1.10) still involves the norm of the step, the accuracy
requirement (1.9) can be implemented without requiring an inner loop (see, [3] and Algorithm
2.1).
We ﬁnally mention that regularisation methods employing inexact derivatives and also inexact
function values are proposed in [4] and the complexity analysis carried out covers arbitrary
optimality order and arbitrary degree of the available approximate derivatives. Also in this
latter approach, the accuracy requirement in derivatives approximation depends on the norm
of the step and an inner loop is needed in order to increase the accuracy and meet the accuracy
requirements. A diﬀerent approach based on the Inexact Restoration framework is given in [8]
where, in the context of ﬁnite-sums problems, the sample size rather than the approximation
accuracy is adaptively chosen.


<!-- page 5 -->
5
1.2. Contributions
In light of the related works the main contributions of this paper are the following:
• We generalise the method given in [3]. In particular, we kept the practical adaptive
criterion (1.9), which is implemented without including an inner loop, allowing inex-
actness in the gradient as well. Namely, inspired by [4], we require that the gradient
approximation satisﬁes the following relative implicit condition:
∥∇f(xk) −∇f(xk)∥≤ζk∥∇f(xk)∥2,
(1.11)
where ζk is an iteration-dependent nonnegative parameter. Unlike [18] and [25] (see
(1.7)), this latter condition does not depend on the norm of the step. Thus, its practical
implementation calls for an inner loop that can be performed before the step com-
putation and extra-computations of the step are not needed. A detailed description
of a practical implementation of this accuracy requirement in subsampling scheme for
ﬁnite-sum minimisation is given in Section 4.
• We assume that the accuracy requirements (1.9) and (1.11) are satisﬁed with high proba-
bility and we perform, in the spirit of [18], the stochastic analysis of the resulting method,
showing that the expected number of iterations needed to reach an ǫ-approximate ﬁrst-
order critical point is, in the worst-case, of the order of ǫ−3/2. This analysis also applies
to the method given in [3].
2. A stochastic cubic regularisation algorithm with inexact derivatives
evaluations
Before introducing our stochastic algorithm, we consider the following hypotheses on f.
Assumption 2.1. With reference to problem (1.1), the objective function f is assumed to
be:
(i) bounded below by flow, for all x ∈IRn;
(ii) twice continuously diﬀerentiable, i.e. f ∈C2(Rn);
Moreover,
(iii) the Hessian is globally Lipschitz continuous with Lipschitz constant LH > 0, i.e.,
∥∇2f(x) −∇2f(y)∥≤LH∥x −y∥,
(2.1)
for all x, y ∈Rn.
The iterative method we are going to introduce is, basically, the stochastic counterpart of
an extension of the one proposed in [3], based on ﬁrst and second-order inexact information.
More in depth at iteration k, given the trial step s, the value of the objective function at
xk +s is predicted by mean of a cubic model mk(xk, s, σk) deﬁned in terms of an approximate


<!-- page 6 -->
6
Taylor expansion of f centered at xk with increment s, truncated to the second order, namely
mk(xk, s, σk) = f(xk) + ∇f(xk)T s + 1
2sT ∇2f(xk)s + σk
3 ∥s∥3 def
= T 2(xk, s) + σk
3 ∥s∥3,
(2.2)
in which both the gradient ∇f(xk) and the Hessian matrix ∇2f(xk) represent approximations
of ∇f(xk) and ∇2f(xk), respectively. According to the basic ARC framework in [15], the main
idea is to approximately minimise, at each iteration, the cubic model and to adaptively search
for a regulariser σk such that the following overestimation property is satisﬁed:
f(xk + s) ≤mk(xk, s, σk),
in which s denotes the approximate minimiser of mk(xk, s, σk). Within these requirements, it
follows that
f(xk) = mk(xk, 0, σk) ≥mk(xk, s, σk) ≥f(xk + s),
so that the objective function is not increased when moving from xk to xk + s. To get more
insight, the cubic model (2.2) is approximately minimised in the sense that the minimiser sk
satisﬁes
mk(xk, sk, σk) < mk(xk, 0, σk),
(2.3)
∥∇smk(xk, sk, σk)∥≤βk∥∇f(xk)∥,
(2.4)
for all k ≥0 and some 0 ≤βk ≤β, β ∈[0, 1). Practical choices for βk are, for instance,
βk = β min
 
1,
∥sk∥2
∥∇f(xk)∥
 
or βk = β min(1, ∥sk∥) (see, e.g., [3]), leading to
∥∇smk(xk, sk, σk)∥≤β min  ∥sk∥2, ∥∇f(xk)∥ 
,
(2.5)
and
∥∇smk(xk, sk, σk)∥≤β min(1, ∥sk∥)∥∇f(xk)∥,
(2.6)
respectively. We notice that, if the overestimation property f(xk + s) ≤mk(xk, s, σk) is
satisﬁed, the requirement (2.3) implies that f(xk) = mk(xk, 0, σk) > mk(xk, s, σk) ≥f(xk+s),
resulting in a decrease of the objective. The trial point xk + sk is then used to compute the
relative decrease [7]
ρk =
f(xk) −f(xk + sk)
T 2(xk, 0) −T 2(xk, sk).
(2.7)
If ρk ≥η, with η ∈(0, 1) a prescribed decrease fraction, then the trial point is accepted, the
iteration is declared successful, the regularisation parameter is decreased by a factor γ and we
go on recomputing the approximate model at the updated iterate; otherwise, an unsuccessful
iteration occurs: the point xk + sk is rejected, the regulariser is increased by a factor γ, a
new approximate model at xk is computed and a new trial step sk is recomputed. At each
iteration, the model mk(xk, s, σk) involved relies on inexact quantities, that can be considered
as realisations of random variables. Hereafter, all random quantities are denoted by capital


<!-- page 7 -->
7
letters, while the use of small letters is reserved for their realisations. In particular, let us
denote a random model at iteration k as Mk, while we use the notation mk = Mk(ω) for
its realisation, with ω a random sample taken from a context-dependent probability space
Ω. In particular, we denote by ∇f(Xk) and ∇2f(Xk) the random variables for ∇f(xk) and
∇2f(xk), respectively. Consequently, the iterates Xk, as well as the regularisers Σk and the
steps Sk are the random variables such that xk = Xk(ω), σk = Σk(ω) and sk = Sk(ω).
The focus of this paper is to derive the expected worst-case complexity bound to approach a
ﬁrst-order optimality point, that is, given a tolerance ǫ ∈(0, 1), the number of steps k (in the
worst-case) such that an iterate xk satisfying
∥∇f(xk)∥≤ǫ
is reached. To this purpose, after the description of the algorithm, we state the main deﬁni-
tions and hypotheses needed to carry on with the analysis up to the complexity result. Our
algorithm is reported below.
Algorithm 2.1: Stochastic ARC algorithm with inexact gradient and dynamic
Hessian accuracy
Step 0: Initialisation. An initial point x0 ∈Rn and an initial regularisation parameter
σ0 > 0 are given. The constants β, α, η, γ, σmin and c are also given such that
0 < β < 1, α ∈
 
0, 2
3
 
, σmin ∈(0, σ0], 0 < η < 2 −3α
2
, γ > 1, c > 0.
(2.8)
Compute f(x0) and set k = 0, ﬂag = 1.
Step 1: Gradient approximation.
Compute an approximate gradient ∇f(xk)
Step 2: Hessian approximation (model costruction).
If ﬂag = 1 set ck = c, else
set ck = α(1 −β)∥∇f(xk)∥.
Compute an approximate Hessian ∇2f(xk) that satisﬁes condition (1.9) with a
preﬁxed probability. Form the model mk(xk, s, σk) deﬁned in (2.2).
Step 3: Step calculation.
Choose βk ≤β. Compute the step sk satisfying (2.3)-(2.4).
Step 4: Check on the norm of the trial step.
If ∥sk∥< 1 and ﬂag = 1 and c >
α(1 −β)∥∇f(xk)∥
set xk+1 = xk, σk+1 = σk, ﬂag = 0
(unsuccessful iteration)
set k = k + 1 and go to Step 1.
Step 5: Acceptance of the trial point and parameters update.
Compute
f(xk + sk) and the relative decrease deﬁned in (2.7).
If ρk ≥η
deﬁne xk+1 = xk + sk, set σk+1 = max[σmin, 1
γ σk].
(successful iteration)
If ∥sk∥≥1 set ﬂag = 1, otherwise set ﬂag = 0.
else
deﬁne xk+1 = xk, σk+1 = γσk.
(unsuccessful iteration)
Set k = k + 1 and go to Step 1.


<!-- page 8 -->
8
Some comments on this algorithm are useful at this stage. We ﬁrst note that the Algorithm
2.1 generates a random process
{Xk, Sk, Mk, Σk, Ck},
(2.9)
where Ck = ck(ω) refers to the random variable for the dynamic Hessian accuracy ck, that is
adaptively deﬁned in Step 2 of Algorithm 2.1. Since its deﬁnition relies on random quantities,
ck constitutes a random variable too. We recall that, in the deterministic counterpart given
in [3], the Hessian approximation ∇2f(xk) computed at iteration k has to satisfy the absolute
accuracy requirement (1.9). Here, this condition is assumed to be satisﬁed only with a certain
probability (see, e.g., Assumption 2.2).
The main goal is thus to prove that, if Mk is suﬃciently accurate with a suﬃciently high
probability conditioned to the past, then the stochastic process preserves the expected op-
timal complexity. To this scope, the next section is devoted to state the basic probabilistic
accuracy assumptions and deﬁnitions. In what follows, we use the notation E[X] to indicate
the expected value of a random variable X. In addition, given a random event A, Pr(A) de-
notes the probability of A, while
1A refers to the indicator of the random event A occurring
(i.e.
1A(a) = 1 if a ∈A, otherwise
1A(a) = 0). The notation Ac indicates the complement of
the event A.
2.1. Main assumptions on the stochastic ARC algorithm
For k ≥0, to formalise the conditioning on the past, let FM
k−1 denote the ˆσ-algebra induced
by the random variables M0, M1,..., Mk−1, with FM
−1 = ˆσ(x0).
We ﬁrst consider the following deﬁnitions for measuring the accuracy of the model estimates.
Deﬁnition 2.1 (Accurate model). A sequence of random models {Mk} is said to be p-
probabilistically suﬃciently accurate for Algorithm 2.1, with respect to the corresponding se-
quence {Xk, Sk, Σk, Ck}, if the event Ik = I(1)
k
∩I(2)
k
∩I(3)
k , with
I(1)
k
=
(
∥∇f(Xk) −∇f(Xk)∥≤κ(1 −β)2
 ∥∇f(Xk)∥
Σk
 2
,
κ > 0
)
,
(2.10)
I(2)
k
=
n
∥∇2f(Xk) −∇2f(Xk)∥≤Ck
o
,
(2.11)
I(3)
k
=
n
∥∇f(xk)∥≤κg,
∥∇2f(xk)∥≤κB,
κg > 0, κB > 0
o
,
(2.12)
satisﬁes
Pr(Ik|FM
k−1) = E[1Ik|FM
k−1] ≥p.
(2.13)
What follows is an assumption regarding the nature of the stochastic information used by
Algorithm 2.1.
Assumption 2.2. We assume that the sequence of random models {Mk}, generated by Al-
gorithm 2.1, is p-probabilistically suﬃciently accurate for some suﬃciently high probability
p ∈(0, 1].


<!-- page 9 -->
9
3. Complexity analysis of the algorithm
For a given level of tolerance ǫ, the aim of this section is to derive a bound on the expected
number of iterations E[Nǫ] which is needed, in the worst-case, to reach an ǫ-approximate
ﬁrst-order stationary point. Speciﬁcally, Nǫ denotes a random variable corresponding to the
number of steps required by the process until ∥∇f(Xk)∥≤ǫ occurs for the ﬁrst time, namely
Nǫ = inf{k ≥0 | ∥∇f(Xk)∥≤ǫ};
(3.14)
indeed, Nǫ can be seen as a stopping time for the stochastic process generated by Algorithm
2.1 (see [10, Deﬁnition 2.1]). The analysis follows the path of [18], but some results need to be
proved as for the adopted accuracy requirements for gradient and Hessian and failures in the
sense of Step 4. It is preliminarly useful to sum up a series of existing lemmas from [18] and
[3] and to derive some of their suitable extensions, which will be of paramount importance to
perform the complexity analysis of our stochastic method. These lemmas are recalled in the
following subsection.
3.1. Existing and preliminary results
We observe that each iteration k of Algorithm 2.1 such that
1Ik = 1 corresponds to an
iteration of the ARC Algorithm 3.1 in [3], before termination, except for the fact that in
Algorithm 2.1 the model (2.2) is deﬁned not only using inexact Hessian information, but also
considering an approximate gradient. In particular, the nature of the accuracy requirement
for the gradient approximation given by (2.10) is diﬀerent from the one for the Hessian
approximation, namely (2.11). In fact, a realisation ck of the upper bound Ck in (2.11), needed
to obtain an approximate Hessian ∇2f(xk), is determined by the mechanism of the algorithm
and is available when forming the Hessian approximation ∇2f(xk). On the other hand, (2.10)
is an implicit condition and can be practically gained computing the gradient approximation
within a prescribed absolute accuracy level, that is eventually reduced to recompute the
inexact gradient ∇f(xk); but, in contrast with [18, Algorithm 4.1], without additional step
computation, which is performed only once per iteration at Step 3 of Algorithm 2.1. We will
see that, for any realisation of the algorithm, if the model is accurate, i.e.
1Ik = 1, then there
exist δ ≥0 and ξk > 0 such that
∥(∇f(xk) −∇f(xk))sk∥≤δ∥sk∥3,
∥(∇2f(xk) −∇2f(xk))sk∥≤ξk∥sk∥2,
which will be fundamental to recover optimal complexity. At this regard, let us consider the
following deﬁnitions and state the lemma below.
Deﬁnition 3.1. With reference to Algorithm 2.1, for all 0 ≤k ≤l, l ∈{0, ..., Nǫ −1}, we
deﬁne the events
• Sk = {iteration k is successful};
• Uk,1 = {iteration k is unsuccessful in the sense of Step 5};
• Uk,2 = {iteration k is unsuccessful in the sense of Step 4}.
We underline that if k ∈Uk,1 then ρk < η, while k ∈Uk,2 if and only if ∥sk∥< 1, ﬂag = 1
and c > α(1 −β)∥∇f(xk)∥. Moreover, if ρk < η and a failure in Step 4 does not occur, then


<!-- page 10 -->
10
k ∈Uk,1.
Lemma 3.1. Consider any realisation of Algorithm 2.1. Then, at each iteration k such
that
1I
(1)
k ∩I
(3)
k
= 1 (accurate gradient and bounded inexact derivatives) we have
∥∇f(xk) −∇f(xk)∥≤δ∥sk∥2,
δ def
= κ
  κB
σmin
+ 1
 
max
  κg
σmin
, κB
σmin
+ 1
 
,
(3.15)
and, thus,
∥(∇f(xk) −∇f(xk))sk∥≤δ∥sk∥3.
(3.16)
Proof. Let us consider k such that
1I(1)
k ∩I(3)
k
= 1. Using (2.4) we obtain
β∥∇f(xk)∥
≥
∥∇sm(xk, sk, σk)∥=
 ∇f(xk) + ∇2f(xk)sk + σksk∥sk∥

≥
∥∇f(xk)∥−∥∇2f(xk)∥∥sk∥−σk∥sk∥2.
(3.17)
We can then distinguish between two diﬀerent cases. If ∥sk∥≥1, from (3.17) and (2.12) we
have that
β∥∇f(xk)∥≥∥∇f(xk)∥−∥∇2f(xk)∥∥sk∥2 −σk∥sk∥2 ≥∥∇f(xk)∥−(κB + σk)∥sk∥2
which is equivalent to
∥sk∥2 ≥(1 −β)∥∇f(xk)∥
κB + σk
.
Consequently, by (2.10) and (2.12)
∥∇f(xk) −∇f(xk)∥
≤
κ
 1 −β
σk
 2
∥∇f(xk)∥2 ≤κκg(1 −β)2∥∇f(xk)∥
σ2
k∥sk∥2
∥sk∥2
≤
κκg(1 −β)κB + σk
σ2
k
∥sk∥2 ≤κ κg
σmin
  κB
σmin
+ 1
 
∥sk∥2, (3.18)
where in the last inequality we have used that β ∈(0, 1) and σk ≥σmin. If, instead, ∥sk∥< 1,
inequality (3.17) and (2.12) lead to
β∥∇f(xk)∥≥∥∇f(xk)∥−∥∇2f(xk)∥∥sk∥−σk∥sk∥≥∥∇f(xk)∥−(κB + σk)∥sk∥,
obtaining that
∥sk∥≥(1 −β)∥∇f(xk)∥
κB + σk
.
(3.19)


<!-- page 11 -->
11
Hence, by squaring both sides in the above inequality and using (2.10), β ∈(0, 1) and σk ≥
σmin, we obtain
∥∇f(xk) −∇f(xk)∥
≤
κ
 1 −β
σk
 2
∥∇f(xk)∥2 = κ(1 −β)2∥∇f(xk)∥2
σ2
k∥sk∥2
∥sk∥2
≤
κ
 κB + σk
σk
 2
∥sk∥2 ≤κ
  κB
σmin
+ 1
 2
∥sk∥2.
(3.20)
Inequality (3.15) then follows by virtue of (3.18) and (3.20), while (3.16) stems from (3.15)
by means of the triangle inequality.
The following Lemma is a slight modiﬁcation of [3, Lemma 3.1].
Lemma 3.2. Consider any realisation of Algorithm 2.1 and assume that c ≥α(1−β)κg.
Then, at each iteration k such that
1I
(2)
k ∩I
(3)
k (1 −
1Uk,2) = 1 (successful or unsuccessful in
the sense of Step 5, with accurate Hessian and bounded inexact derivatives) we have
∥∇2f(xk) −∇2f(xk)∥≤ck ≤ξk∥sk∥,
ξk
def
= max[c, α(κB + σk)],
(3.21)
and, thus,
∥(∇2f(xk) −∇2f(xk))sk∥≤ξk∥sk∥2.
(3.22)
Proof. Let us consider k such that
1I(2)
k ∩I(3)
k (1 −
1Uk,2) = 1. Algorithm 2.1 ensures that, if
∥sk∥≥1, then ck = c or
ck = α(1 −β)∥∇f(xk)∥.
(3.23)
Trivially, (3.23), ∥sk∥≥1 and (2.12) give
∥∇2f(xk)−∇2f(xk)∥≤ck ≤max[c, α(1−β)∥∇f(xk)∥] ≤max[c, α(1−β)κg] ≤c∥sk∥, (3.24)
where we have considered the assumption c ≥α(1−β)κg. On the other hand, Step 4 guarantees
the choice
ck ≤α(1 −β)∥∇f(xk)∥,
(3.25)
when ∥sk∥< 1. In this case, inequality (3.19) still holds. Thus,
∥∇2f(xk) −∇2f(xk)∥≤ck =
ck
∥sk∥∥sk∥≤
ck(κB + σk)
(1 −β)∥∇f(xk)∥∥sk∥≤α(κB + σk)∥sk∥, (3.26)
where the last inequality is due to (3.25). Finally, (3.24) and (3.26) imply (3.21), while (3.22)
follows by (3.21) using the triangle inequality.


<!-- page 12 -->
12
The next lemma bounds the decrease of the objective function on successful iterations, irre-
spectively of the satisfaction of the accuracy requirements for gradient and Hessian approxi-
mations.
Lemma 3.3. Consider any realisation of Algorithm 2.1. At each iteration k we have
T 2(xk, 0) −T 2(xk, sk) > σk
3 ∥sk∥3 ≥σmin
3
∥sk∥3 > 0.
(3.27)
Hence, on every successful iteration k:
f(xk) −f(xk+1) > ησk
3 ∥sk∥3 ≥ησmin
3
∥sk∥3 > 0.
(3.28)
Proof. We ﬁrst notice that, by (2.3), we have that ∥sk∦= 0. Moreover, Lemma 2.1 in [7]
coupled with (2.2) yields (3.27). The second part of the thesis is easily proved taking into
account that, if k is successful, then (3.27) implies
f(xk) −f(xk+1) ≥η(T 2(xk, 0) −T 2(xk, sk)) > ησk
3 ∥sk∥3.
As a corollary, because of the fact that xk+1 = xk on each unsuccessful iteration k, for any
realisation of Algorithm 2.1 we have that
f(xk) −f(xk+1) ≥0.
We now show that, if the model is accurate, there exists a constant σ > 0 such that an
iteration is successful or unsuccessful in the sense of Step 4 (1Ik(1 −
1Uk,1) = 1), whenever
σk ≥σ. In other words, it is an iteration at which the regulariser is not increased.
Lemma 3.4. Let Assumption 2.1 (ii) hold. Let δ be given in (3.15), assume c ≥α(1 −
β)κg and the validity of (2.1). For any realisation of Algorithm 2.1, if the model is
accurate and
σk ≥σ def
= max
 6δ + 3ακB + LH
2(1 −η) −3α , 6δ + 3c + LH
2(1 −η)
 
> 0,
(3.29)
then the iteration k is successful or a failure in the sense of Step 4 occurs.
Proof. Let us consider an iteration k such that
1Ik(1 −
1Uk,1) = 1 and the deﬁnition of ρk
in (2.7). Assume that a failure in the sense of Step 4 does not occur. If ρk −1 ≥0, then
iteration k is successful by deﬁnition. We can thus focus on the case in which ρk −1 < 0. In


<!-- page 13 -->
13
this situation, the iteration k is successful provided that 1 −ρk ≤1 −η. From (2.1) and the
Taylor expansion of f centered at xk with increment s it ﬁrst follows that
f(xk + s) ≤f(xk) + ∇f(xk)⊤s + 1
2s⊤∇2f(xk)s + LH
6 ∥s∥3.
(3.30)
Therefore, since
1Ik = 1,
f(xk + sk) −T 2(xk, sk)
≤
(∇f(xk) −∇f(xk))⊤sk + 1
2s⊤
k (∇2f(xk) −∇2f(xk))sk + LH
6 ∥sk∥3
≤
∥∇f(xk) −∇f(xk)∥∥sk∥+ 1
2∥∇2f(xk) −∇2f(xk)∥∥sk∥2 + LH
6 ∥sk∥3
≤
 
δ + LH
6 + ξk
2
 
∥sk∥3,
(3.31)
where we have used (3.15) and (3.21). Thus, by (3.31) and (3.27),
1 −ρk = f(xk + sk) −T 2(xk, sk)
T 2(xk, 0) −T 2(xk, sk) < (6δ + 3ξk + LH) ∥sk∥3
2σk∥sk∥3
= 6δ + 3ξk + LH
2σk
.
Depending on the maximum in the deﬁnition of ξk in (3.21), two diﬀerent cases can then
occur. If ξk = c, 1 −ρk ≤1 −η, provided that
σk ≥6δ + 3c + LH
2(1 −η)
.
Otherwise, if c < α(κB + σk), so that ξk = α(κB + σk), then
1 −ρk < 6δ + 3α(κB + σk) + LH
2σk
≤1 −η,
provided that
σk ≥6δ + 3ακB + LH
2(1 −η) −3α .
In conclusion, iteration k is successful if (3.29) holds. Note that σ is a positive lower bound
for σk because of the ranges for the values of η and α in (2.8).
Using some of the results from the proof of the previous lemma, we can now prove the
following, giving a crucial relation between the step length ∥sk∥and the true gradient norm
∥∇f(xk + sk)∥at the next iteration.


<!-- page 14 -->
14
Lemma 3.5. Let Assumption 2.1 (ii)-(iii) hold and assume c ≥α(1 −β)κg. For any
realisation of Algorithm 2.1, at each iteration k such that
1Ik(1 −
1Uk,2) = 1 (accurate in
which the iteration is successful or a failure in the sense of Step 5 occurs), we have
∥sk∥≥
p
νk∥∇f(xk + sk)∥,
(3.32)
for some positive νk, whenever sk satisﬁes (2.5). Moreover, (3.32) holds even in case sk
satisﬁes (2.6) provided that there exists Lg > 0 such that
∥∇f(x) −∇f(y)∥≤Lg∥x −y∥,
(3.33)
for all x, y ∈Rn.
Proof. Let us consider an iteration k such that
1Ik(1 −
1Uk,2) = 1. From the Taylor series of
∇f(x) centered at xk with increment s, and the deﬁnition of the model (2.2), proceeding as
in the proof of Lemma 4.1 in [3] we obtain
∥∇f(xk + sk) −∇sT 2(xk + sk)∥
≤
∥∇f(xk) −∇f(xk)∥+ ∥(∇2f(xk) −∇2f(xk))sk∥
+
Z 1
0
∥∇2f(xk + τs) −∇2f(xk)∥∥sk∥dτ
≤
 
δ + ξk + LH
2
 
∥sk∥2,
(3.34)
where we have used (3.15), (3.22) and (2.1). Moreover, since ∇sm(xk, sk, σk) = ∇sT 2(xk, sk)+
σk∥sk∥sk, it follows:
∥∇f(xk + sk)∥≤∥∇f(xk + sk) −∇sT 2(xk, sk)∥+ ∥∇sm(xk, sk, σk)∥+ σk∥sk∥2.
(3.35)
As a consequence, the thesis follows from (3.34)–(3.35) with
ν−1
k
=
 
δ + ξk + LH
2 + β + σk
 
> 0,
(3.36)
when the stopping criterion (2.5) is considered. Assume now that (2.6) is used for Step 3 of
Algorithm 2.1. Inequalities (3.15) and (3.33) imply that
∥∇f(xk)∥
≤
∥∇f(xk) −∇f(xk)∥+ ∥∇f(xk) −∇f(xk + sk)∥+ ∥∇f(xk + sk)∥
≤
δ∥sk∥2 + Lg∥sk∥+ ∥∇f(xk + sk)∥.
(3.37)
By using (3.34)–(3.35) and plugging (3.37) into (2.6), we ﬁnally have
∥∇f(xk + sk)∥(1 −β) ≤
 
(1 + β)δ + ξk + LH
2 + βLg + σk
 
∥sk∥2,


<!-- page 15 -->
15
which is equivalent to (3.32), with
νk =
1 −β
(1 + β)δ + ξk + LH/2 + βLg + σk
> 0.
(3.38)
It is worth noticing that the global Lipschitz continuity of the gradient, namely, (3.33), is
needed only when condition (2.6) is used in Step 3 of Algorithm 2.1. We ﬁnally recall a result
from [18] that will be of key importance to carry out the complexity analysis addressed in
the following two subsections.
Lemma 3.6. [18, Lemma 2.1] Let Nǫ be the hitting time deﬁned as in (3.14). For all
k < Nǫ, let {Ik} be the sequence of events in Deﬁnition 2.1 so that (2.13) holds. Let
1Wk
be a nonnegative stochastic process such that ˆσ(1Wk) ⊆FM
k−1, for any k ≥0. Then,
E
"Nǫ−1
X
k=0
1Wk
1Ik
#
≥pE
"Nǫ−1
X
k=0
1Wk
#
.
Similarly,
E
"Nǫ−1
X
k=0
1Wk(1 −
1Ik)
#
≤(1 −p)E
"Nǫ−1
X
k=0
1Wk
#
.
3.2. Bound on the expected number of steps with Σk ≥σ
In this section we derive an upper bound for the expected number of steps in the process
generated by Algorithm 2.1 with Σk ≥σ. Given l ∈{0, ..., Nǫ −1}, for all 0 ≤k ≤l, let us
deﬁne the event
Λk = {iteration k is such that Σk < σ}
and let
Nσ
def
=
Nǫ−1
X
k=0
(1 −
1Λk),
N
C
σ
def
=
Nǫ−1
X
k=0
1Λk,
(3.39)
be the number of steps, in the stochastic process induced by Algorithm 2.1, with Σk ≥σ
and Σk < σ, before Nǫ is met, respectively. In what follows we consider the validity of
Assumption 2.1, Assumption 2.2 and the following assumption on Σ0.


<!-- page 16 -->
16
Assumption 3.1. With reference to the stochastic process generated by Algorithm 2.1 and
the deﬁnition of σ in (3.29), we assume that
Σ0 = γ−iσ,
(3.40)
for some positive integer i. We additionally assume that c ≥α(1 −β)κg.
By referring to Lemma 3.6 and some additional lemmas from [18], we can ﬁrst obtain an
upper bound on E[Nσ]. In particular, rearranging [18, Lemma 2.2], given a generic iteration
l, we derive a bound on the number of iterations successful and unsuccessful in the sense of
Step 4, in terms of the overall number of iterations l + 1. At this regard, we underline that in
case of unsuccessful iterations in Step 4, the value of Σk is not modiﬁed and such an iteration
occurs at most once between two successful iterations (not necessary consecutive) with the
ﬁrst one having the norm of the step not smaller than one or once before the ﬁrst successful
iteration of the process (since ﬂag is initially 1). In fact, a failure in the sense of Step 4 may
occur only if ﬂag=1; except for the ﬁrst step, ﬂag is reassigned only at the end of a successful
iteration and can be set to one only in case of successful iteration with ∥sk∥≥1 (see Step
5 of Algorithm 2.1), except for the ﬁrst iteration. If the case ﬂag = 1 and ∥sk∥< 1 occurs
then ﬂag is set to zero, preventing a failure in Step 4 at the subsequent iteration, and it is
not further changed until a subsequent successful iteration.
Lemma 3.7. Assume that Σ0 < σ. Given l ∈{0, ..., Nǫ −1}, for all realisations of
Algorithm 2.1,
l
X
k=0
(1 −
1Λk)
1Sk∪Uk,2 ≤2
3(l + 1).
Proof. Each iteration k such that (1−
1Λk)1Sk∪Uk,2 = 1 is an iteration with Σk ≥σ that can
be a successful iteration, leading to Σk+1 = max[σmin, 1
γ Σk] (Σk is decreased), or an unsuc-
cessful iteration in the sense of Step 4. In the latter case, Σk is left unchanged (Σk+1 = Σk).
Moreover, Σk in successful/unsuccessful in the sense of Step 5 iterations is decreased/increased
by the same factor γ. More in depth, since Σ0 < σ, we have two possible scenarios. In the
ﬁrst one we have Σk < σ, k = 0, . . . , l and the thesis obviously follows. In the second scenario
there exists at least one index k such that Σk ≥σ and at least one unsuccessful iteration at
iteration j ∈{0, . . . , k −1} in which Σk has been increased by the factor γ. In case
1Uk,2 = 1,
Σk is left unchanged, ﬂag is set to 0 and
1Uk+1,2 = 0. Then, at any iteration j such that
1Uj,1 = 1 corresponds at most one successful iteration and one unsuccessful iteration in the
sense of Step 4, with Σk ≥σ and this yields the thesis.
We note that in the stochastic ARC method in [18] each iteration can be successful or un-
successful according to the satisfaction of the decrease condition ρk ≥η. On the contrary, in
Algorithm 2.1 also failures in Step 4 may occur and this yields the bound 2/3(l+1) in Lemma
3.7, while the corresponding bound in [18] is 1/2(l + 1).
As in [18], we note that ˆσ(1Λk) ⊆FM
k−1, that is the variable Λk is fully determined by the
ﬁrst k −1 iterations of the Algorithm 2.1. Then, setting l = Nǫ −1 we can rely on Lemma


<!-- page 17 -->
17
3.6 (with Wk = Λc
k) to deduce that
E
"Nǫ−1
X
k=0
(1 −
1Λk)1Ik
#
≥pE
"Nǫ−1
X
k=0
(1 −
1Λk)
#
.
(3.41)
Considering the bound in Lemma 3.7 and the fact that Lemma 3.4 and the mechanism of
Step 4 in Algorithm 2.1 ensure that each iteration k such that
1Ik = 1 with σk ≥σ can be
successful or unsuccessful in the sense of Step 4 (i.e.,
1Sk∪Uk,2 = 1), we have that
Nǫ−1
X
k=0
(1 −
1Λk)1Ik ≤
Nǫ−1
X
k=0
(1 −
1Λk)1Sk∪Uk,2 ≤2
3Nǫ.
Taking expectation in the above inequality and recalling the deﬁnition of Nσ in (3.39), from
(3.41) we conclude that:
E[Nσ] ≤2
3pE[Nǫ].
(3.42)
The remaining bound for E
 
N
C
σ
 
will be derived in the next section.
3.3. Bound on the expected number of steps with Σk < σ
Let us now obtain an upper bound for E
 
N
C
σ
 
, with N
C
σ deﬁned in (3.39). To this purpose,
the following additional deﬁnitions are needed.
Deﬁnition 3.2. Let Uk,1, Uk,2 and Sk be as deﬁned in Deﬁnition 3.1. With reference to the
process (2.9) generated by Algorithm 2.1 let us deﬁne:
• the event Λk = {iteration k is such that Σk ≤σ}, i.e. Λk is the closure of Λk.
• M1 = PNǫ−1
k=0
1Λk(1 −
1Ik): number of inaccurate iterations with Σk ≤σ;
• M2 = PNǫ−1
k=0
1Λk
1Ik: number of accurate iterations with Σk ≤σ;
• N1 = PNǫ−1
k=0
1Λk
1Ik
1Sk: number of accurate successful iterations with Σk ≤σ;
• N2 = PNǫ−1
k=0
1Λk
1Ik
1Uk,2: number of accurate unsuccessful iterations, in the sense of
Step 4, with Σk ≤σ;
• N3 = PNǫ−1
k=0
1Λk
1Ik
1Uk,1: number of accurate unsuccessful iterations, in the sense of
Step 5, with Σk < σ;
• M3 = PNǫ−1
k=0
1Λk(1 −
1Ik)1Sk: number of inaccurate successful iterations, with Σk ≤σ;
• S = PNǫ−1
k=0
1Λk
1Sk: number of successful iterations, with Σk ≤σ;
• H = PNǫ−1
k=0
1Uk,2: number of unsuccessful iterations in the sense of Step 4;
• U = PNǫ−1
k=0
1Λk
1Uk,1: number of unsuccessful iterations, in the sense of Step 5, with
Σk < σ.


<!-- page 18 -->
18
It is worth noting that an upper bound on E 
N
C
σ
 
is given, once an upper bound on E[M1] +
E[M2] is provided, since
E
 
N
C
σ
 
≤
E
"Nǫ−1
X
k=0
1Λk
#
=
E
"Nǫ−1
X
k=0
1Λk(1 −
1Ik) +
Nǫ−1
X
k=0
1Λk
1Ik
#
= E[M1] + E[M2].
(3.43)
where M1 and M2 are given in Deﬁnition 3.2. Following [18], to bound E[M1] we can still
refer to the central Lemma 3.6 (with Wk = Λk), of which the result stated below is a direct
consequence.
Lemma 3.8. [18, Lemma 2.6] With reference to the stochastic process (2.9) generated
by Algorithm 2.1 and the deﬁnitions of M1, M2 in Deﬁnition 3.2,
E[M1] ≤1 −p
p
E[M2].
(3.44)
Concerning the upper bound for E[M2] we observe that
E[M2] =
3
X
i=1
E[Ni] ≤E[N1] + E[N2] + E[U].
(3.45)
In the following Lemma we provide upper bounds for N1 and N2, given in Deﬁnition 3.2.
Lemma 3.9. Let Assumption 2.1 hold and that the stopping criterion (2.5) is used to
perform each Step 3 of Algorithm 2.1. With reference to the stochastic process (2.9)
induced by the Algorithm there exists κs > 0 such that
N1 ≤κs(f0 −flow)ǫ−3/2 + 1.
(3.46)
Moreover, in case the stopping criterion (2.6) is used in Step 3, (3.46) still holds provided
that there exists Lg > 0 such that (3.33) is satisﬁed for all x, y ∈Rn.
Finally, let Assumption 2.1 (i)-(ii) hold, independently of the stopping criterion used to
perform Step 3 there exists κu > 0 such that
N2 ≤κu(f0 −flow).
(3.47)
Proof. Taking into account that (3.28) holds for each realisation of Algorithm 2.1, (3.32) is
valid for each realisation of Algorithm 2.1 with
1Ik(1 −
1Uk,2) = 1, recalling that f(Xk) =


<!-- page 19 -->
19
f(Xk+1) for all k ∈Uk,1 ∪Uk,2 and setting f0
def
= f(X0), it follows:
f0 −flow ≥f0 −f(XNǫ) =
Nǫ−1
X
k=0
(f(Xk) −f(Xk+1))1Sk ≥
Nǫ−1
X
k=0
>0
z
}|
{
ησmin
3
∥Sk∥3
1Sk
≥
Nǫ−2
X
k=0
ησmin
3
∥Sk∥3
1Sk
1Ik ≥
Nǫ−2
X
k=0
ησmin
3
ν3/2
k
∥∇f(Xk+1)∥3/2
1Sk
1Ik
≥
Nǫ−2
X
k=0
ησmin
3
ν3/2∥∇f(Xk+1)∥3/2
1Sk
1Ik
1Λk
≥(N1 −1)κ−1
s ǫ3/2,
in which νk is deﬁned in (3.36) when sk satisﬁes (2.5) and in (3.38) when sk satisﬁes (2.6)
and
κ−1
s
def
= ησmin
3
ν3/2
(3.48)
where
ν =
1
δ + max[c, α(κB + σ)] + LH/2 + β + σ > 0,
in case (2.5) is used and
ν =
1 −β
(1 + β)δ + max[c, α(κB + σ)] + LH/2 + βLg + σ > 0,
whenever (2.6) is adopted. Hence, (3.46) holds.
Moreover, an upper bound for N2 can be obtained taking into account that, as already
noticed, an iteration k ≥1 in the process such that
1Uk,2 = 1 occurs at most once between
two successful iterations with the ﬁrst one having the norm of the trial step not smaller than
1, plus at most once before the ﬁrst successful iteration in the process (since in Algorithm 2.1
ﬂag is initialised at 1). Therefore, by means of (3.28),
f0 −flow ≥f0 −f(XNǫ) =
Nǫ−1
X
k=0
(f(Xk) −f(Xk+1))1Sk ≥
Nǫ−1
X
k = 0
∥Sk∥≥1
(f(Xk) −f(Xk + Sk))1Sk
≥ησmin
3
Nǫ−1
X
k = 0
∥Sk∥≥1
1Sk∥Sk∥3 ≥κ−1
u H,
where H denotes (see Deﬁnition 3.2) the number of unsuccessful iterations in the sense of
Step 4. Then, since H ≥N2, (3.47) follows.


<!-- page 20 -->
20
An upper bound for U can be still derived using [18, Lemma 2.5], provided that (3.40) holds.
This is because the process induced by Algorithm 2.1 ensures that Σk is decreased by a factor
γ on successful steps, increased by the same factor on unsuccessful ones in the sense of Step
5 and left unchanged if an unsuccessful iteration in the sense of Step 4 occurs.
Lemma 3.10. [18, Lemma 2.5] Consider the validity of (3.40). For any l ∈{0, ..., Nǫ−1}
and for all realisations of Algorithm 2.1, we have that
l
X
k=0
1Λk
1Uk,1 ≤
l
X
k=0
1Λk
1Sk + logγ
  σ
σ0
 
.
Consequently, considering l = Nǫ −1 and Deﬁnition 3.2,
U ≤S + logγ
  σ
σ0
 
= N1 + M3 + logγ
  σ
σ0
 
.
(3.49)
We underline that the right-hand side in (3.49) involves M3, that has not been bounded yet.
To this aim we can proceed as in [18], obtaining that
E[M3] ≤1 −p
2p −1
 
2E[N1] + E[N2] + logγ
  σ
σ0
  
.
(3.50)
In fact, recalling the deﬁnition of M3 and (3.45), the inequality (3.44) implies that
E[M3] ≤E[M1] ≤1 −p
p
E[M2] ≤1 −p
p
(E[N1] + E[N2] + E[U]) .
(3.51)
Indeed, taking expectation in (3.49) and plugging it into (3.51),
E[M3] ≤1 −p
p
 
2E[N1] + E[N2] + E[M3] + logγ
  σ
σ0
  
,
which yields(3.50). The upper bound on E[M2] then follows:
E[M2]
≤
E[N1] + E[N2] + E[U] ≤2E[N1] + E[N2] + E[M3] + logγ
  σ
σ0
 
≤
  1 −p
2p −1 + 1
   
2E[N1] + E[N2] + logγ
  σ
σ0
  
=
p
2p −1
 
2E[N1] + E[N2] + logγ
  σ
σ0
  
≤
p
2p −1
 
(f0 −flow)
 
2κsǫ−3/2 + κu
 
+ logγ
  σ
σ0
 
+ 2
 
,
(3.52)
in which we have used (3.45), (3.47), (3.46), (3.49) and (3.50). Therefore, recalling (3.43) and


<!-- page 21 -->
21
(3.44), we obtain that
E
 
N
C
σ
 
≤1
pE[M2] ≤
1
2p −1
 
(f0 −flow)
 
2κsǫ−3/2 + κu
 
+ logγ
  σ
σ0
 
+ 2
 
,
(3.53)
where the last inequality follows from (3.52). We are now in the position to state our ﬁnal
result, providing the complexity of the stochastic method associated with Algorithm 2.1,
in accordance with the complexity bounds given by the deterministic analysis of an ARC
framework with exact [7] and inexact [3,4,12,14,16] function and/or derivatives evaluations.
Theorem 3.11. Let Assumptions 2.1 and 3.1 hold. Assume that Assumption 2.2 holds with
p > 2/3 and that the stopping criterion (2.5) is used to perform each Step 3 of Algorithm 2.1.
Then, the hitting time Nǫ for the stochastic process generated by Algorithm 2.1 satisﬁes
E[Nǫ] ≤
3p
(3p −2)(2p −1)
 
(f0 −flow)
 
2κsǫ−3/2 + κu
 
+ logγ
  σ
σ0
 
+ 2
 
.
(3.54)
Moreover, in case the stopping criterion (2.6) is used to perform Step 3, (3.54) still holds
provided that there exists Lg > 0 such that (3.33) is satisﬁed for all x, y ∈Rn.
Proof. By deﬁnition (see (3.39)), E[Nǫ] = E
 
Nσ
 
+ E
 
N
C
σ
 
. Thus, considering (3.42),
E[Nǫ] ≤2
3pE[Nǫ] + E
 
N
C
σ
 
,
and, hence, by (3.53),
E[Nǫ] ≤
3p
3p −2E
 
N
C
σ
 
=
3p
(3p −2)(2p −1)
 
(f0 −flow)
 
2κsǫ−3/2 + κu
 
+ logγ
  σ
σ0
 
+ 2
 
,
which concludes the proof.
4. Subsampling scheme for ﬁnite-sum minimisation
We now consider the solution of large-scale instances of the ﬁnite-sum minimisation problems
arising in machine learning and data analysis, modelled by (1.2). In this context, the approx-
imations ∇f(xk) and ∇2f(xk) to the gradient and the Hessian used at Step 1 and Step 2
of Algorithm 2.1, respectively, are obtained by subsampling, using subsets of indexes Dj,k,
j ∈{1, 2}, randomly and uniformly chosen from {1, ..., N}. I.e., for j ∈{1, 2},
∇jf(xk) =
1
|Dj,k|
X
i∈Dj,k
∇jϕi(xk),
(4.55)
are used in place of ∇jf(xk) =
1
N
PN
i=0 ∇jϕi(xk). Speciﬁcally, if we want ∇jf(xk) to be
within an accuracy τj,k with probability at least pj, j ∈{1, 2}, i.e.,
Pr
 
∥∇jf(xk) −∇jf(xk)∥≤τj,k
 
≥pj,


<!-- page 22 -->
22
the sample size |Dj,k| can be determined by using the operator-Berstein inequality introduced
in [30], so that ∇jf(xk) takes the form (see [4]) given by (4.55), with
|Dj,k| ≥min
 
N,
 4κϕ,j(xk)
τj,k
 2κϕ,j(xk)
τj,k
+ 1
3
 
log
 
dj
1 −pj
   
,
(4.56)
where
dj =
 
n + 1,
if j = 1,
2n,
if j = 2,
and under the assumption that, for any x ∈Rn, there exist non-negative upper bounds
{κϕ,j}2
j=1 such that
max
i∈{1,...,N} ∥∇jϕi(x)∥≤κϕ,j(x),
j ∈{1, 2}.
Let us assume that there exist κg > 0 and κB > 0 such that κϕ,1(x) ≤κg and κϕ,2(x) ≤κB
for any x ∈Rn. Since the subsampling procedures used at iteration k to get D1,k and D2,k are
independent, it follows that when {τj,k}2
j=1 are chosen as the right-hand sides in (2.10) and
(2.11), respectively, the builded model (2.2) is p-probabilistically δ-suﬃciently accurate with
p = p1p2. Therefore, a practical version of Algorithm 2.1 is for instance given by adding a
suitable termination criterion and modifying the ﬁrst three steps of Algorithm 2.1 as reported
in Algorithm 4.1 below.
Algorithm 4.1: Modiﬁed Steps 0 −2 of Algorithm 2.1
Step 0: Initialisation. An initial point x0 ∈Rn and an initial regularisation parameter
σ0 > 0 are given, as well as an accuracy level ǫ ∈(0, 1). The constants β, α, η, γ,
σmin, κ, τ0, κτ and c are also given such that
0 < β, κτ < 1,
0 ≤α < 2
3,
σmin ∈(0, σ0],
0 < η < 2−3α
2
,
γ > 1,
κ ∈[0, 1),
τ0 > 0,
c > 0.
Compute f(x0) and set k = 0, ﬂag = 1.
Step 1: Gradient approximation.
Set i = 0 and initialise τ (i)
1,k = τ0. Do
1.1 compute ∇f(xk) such that (4.55)–(4.56) are satisﬁed with j = 1, τ1,k = τ (i)
1,k;
1.2 if τ (i)
1,k ≤κ(1 −β)2  
∥∇f(xk)∥
σk
 2
, go to Step 2;
else, set τ (i+1)
1,k
= κττ (i)
1,k, increment i by one and go to Step 1.1;
Step 2: Hessian approximation (model costruction).
If ﬂag = 1 set ck = c, else
set ck = α(1 −β)∥∇f(xk)∥.
Compute ∇2f(xk) using (4.55)–(4.56) with j = 2, τ2,k = ck and form the model
mk(s) deﬁned in (2.2).


<!-- page 23 -->
23
Concerning the gradient estimate, the scheme computes (Step 1) an approximation ∇f(xk)
satisfying the accuracy criterion
∥∇f(xk) −∇f(xk)∥≤κ(1 −β)2
 ∥∇f(xk)∥
σk
 2
,
(4.57)
which is independent of the step computation and based on the knowable quantities κ,
β and σk. This is done by reducing the accuracy τ (i)
1,k and repeating the inner loop at Step
1, until the fulﬁllment of the inequality at Step 1.2. We underline that condition (4.57) is
guaranteed by the algorithm, since (4.56) is a continuous and increasing function with respect
to τj,k, for ﬁxed j = 1, k, pj and N; hence, there exists a suﬃciently small τ 1,k such that the
right-hand side term in (4.56) will reach, in the worst-case, the full sample size N, yielding
∇f(xk) = ∇f(xk). Moreover, if the stopping criterion ∥∇f(xk)∥≤ǫ is used, the loop is
ensured to terminate also whenever the predicted accuracy requirement τ (i)
1,k becomes smaller
than κ(1−β
σk )2ǫ2. On the other hand, in practice, we expect to use a small number of samples in
the early stage of the iterative process, when the norm of the approximated gradient is not yet
small. To summarise, if without loss of generality we assume that τ 1,k ≥ˆτ at each iteration
k, we conclude that, in the worst case, Step 1 will lead to at most ⌊log(ˆτ)/ log(κττ0)⌋+ 1
computations of ∇f(xk). The Hessian approximation ∇2f(xk) is, instead, deﬁned at Step 2
and its computation relies on the reliable value of ck. We remark that at iteration k we have
that:
• ∇2f(xk) is computed only once, irrespectively of the approximate gradient computation
considered at Step 1;
• a ﬁnite loop is considered at Step 1 to obtain a gradient approximation satisfying (4.57),
where the right-hand side is independent of the step length ∥sk∥, thou implying (3.15)–
(3.16). Hence, the gradient approximation is fully determined at the end of Step 1 and
further recomputations due to the step calculation (see Algorithm 2.1, Step 3) are not
required.
We conclude this section by noticing that each iteration k of Algorithm 2.1 with the modiﬁed
steps introduced in Algorithm 4.1 can indeed be seen as an iteration of Algorithm 2.1 where
the sequence of random models {Mk} is p-probabilistically suﬃciently accurate in the sense of
Deﬁnition 2.1, with p = p1p2, and an iteration of [3, Algorithm 3.1], when κ = 0 is considered
in (2.10) (exact gradient evaluations).
5. Numerical tests
In this section we analyse the behaviour of the Stochastic ARC Algorithm (Algorithm (2.1)).
Inexact gradient and Hessian evaluations are performed as sketched in modiﬁed Steps 0-2
of Algorithm 4.1. The performance of the proposed algorithm is compared with that of the
corresponding version in [3] employing exact gradient, with the aim to provide numerical
evidence that adding a further source of inexactness in gradient computation is beneﬁcial in
terms of computational cost saving. We consider nonconvex ﬁnite-sum minimisation problems.
This is, in fact, a highly frequent scenario when dealing with binary classiﬁcation tasks arising
in machine learning applications. More in depth, given a training set of N features ai ∈Rn


<!-- page 24 -->
24
and corresponding labels yi, i = 1, . . . , N, we solve the following minimisation problem:
min
x∈Rn f(x) = min
x∈Rn
1
N
N
X
i=1
ϕi(x) = min
x∈Rn
1
N
N
X
i=1
 
yi −σ
 
a⊤
i x
  2
,
(5.58)
where
σ(a⊤w) =
1
1 + e−a⊤w ,
a, w ∈Rn.
(5.59)
That is we use the sigmoid function (5.59) as the model for predicting the values of the
labels and the least-squares loss as a measure of the error on such predictions, that has
to be minimised by approximately solving (5.58) in order to come out with the parameter
vector x, to be used for label predictions on new untested data. Moreover, a number NT
of testing data {ai, yi}NT
i=1 is used to validate the computed model. The values σ(a⊤
i x) are
used to predict the testing labels yi, i ∈{1, ..., NT }, and the corresponding error, measured
by
1
NT
PNT
i=1
 yi −σ
 aT
i x
  2 , is computed. Implementation issues concerning the considered
procedures are the object of Subsection 5.1, while statistics of our runs are discussed in
Subsection 5.2.
5.1. Implementation issues
The implementation of the main phases of Algorithm (2.1), equipped with the modiﬁed steps
in Algorithm (4.1), respects the following speciﬁcations.
According to [3, Algorithm 3.1], the cubic regularisation parameter is initially σ0 = 10−1, its
minimum value is σmin = 10−5 and the initial guess vector x0 = (0, ..., 0)⊤∈Rn is considered
for all runs. Moreover, the probability of success pj in (4.56) is set equal to 0.8, for j ∈{1, 2},
while the parameters α, β, ǫ, η and γ are ﬁxed as α = 0.1, β = 0.5, ǫ = 5 · 10−3, η = 0.8 and
γ = 2. The latter two correspond to the values of η2 and γ3 considered in [3, Algorithm 3.1],
respectively. The minimisation of the cubic model at Step 3 of Algorithm 2.1 is performed by
the Barzilai-Borwein gradient method [2] combined with a nonmonotone linesearch following
the proposal in [6]. The major per iteration cost of such Barzilai-Borwein process is one
Hessian-vector product, needed to compute the gradient of the cubic model. The threshold
used in the termination criterion (2.4) is βk = 0.5, k ≥0. As for [3, Algorithm 3.1], we impose
a maximum of 500 iterations and a successful termination is declared when the following
condition is met:
∥∇f(xk)∥≤ǫ,
k ≥0.
In case ∥∇f(xk)∥≤ǫ and the model is accurate, then by (2.10)
∥∇f(xk)∥≤∥∇f(xk)∥+ ∥∇f(xk) −∇f(xk)∥≤ǫ := ǫ + κ[(1 −β)/σmin]2ǫ2
and, hence, xk is an ǫ-approximate ﬁrst-order optimality point. Since the model is accurate
with probability at least p, xk is an ǫ-approximate ﬁrst-order optimality point with probability
at least p. We further note that the exact gradient and the Hessian of the component functions


<!-- page 25 -->
25
ϕi(x), i ∈{1, ..., N}, are given by:
∇ϕi(x) = −2e−a⊤
i x  
1 + e−a⊤
i x −2  
yi −
 
1 + e−a⊤
i x −1 
ai,
(5.60)
∇2ϕi(x) = −2e−a⊤
i x  
1 + e−a⊤
i x −4  
yi
  
e−a⊤
i x 2
−1
 
+ 1 −2e−a⊤
i x
 
aia⊤
i .(5.61)
Then, the gradient and the Hessian approximations ∇jf(xk), j ∈{1, 2}, computed at Step 1
and Step 2 of Algorithm 4.1 according to (4.55)–(4.56), involve the constants
κϕ,1(xk)
=
max
i∈{1,...,N}
 
2e−a⊤
i xk  
1 + e−a⊤
i xk −2     yi −
 
1 + e−a⊤
i xk −1     ∥ai∥
 
,
κϕ,2(xk)
=
max
i∈{1,...,N}
 
2e−a⊤
i xk  
1 + e−a⊤
i xk −4     yi
  
e−a⊤
i xk 2
−1
 
+ 1 −2e−a⊤
i xk
     ∥ai∥2
 
,
whose computations can indeed be an issue in theirselves. Nevertheless, thank to the exactness
and the speciﬁc form (see (5.58)) of the function evaluation f(xk), the values a⊤
i xk, 1 ≤i ≤N,
are available at iteration k and, hence, κϕ,j(xk), j ∈{1, 2}, can be determined at the (oﬄine)
extra cost of computing ∥ai∥j, j ∈{1, 2}, for 1 ≤i ≤N. As in [3, Subsection 8.2], the value of
c used in (1.10), in order to reduce the iteration computational cost whenever ∥sk∥≥1, is such
that |D2,0| computed via (4.56) for j = 2, with τ2,0 = c (ﬁrst approximation of the Hessian),
satisﬁes |D2,0|/N = 0.1. We indeed start using the 10% of the examples to approximate the
Hessian. Concerning the gradient approximation performed at Step 1 of Algorithm 4.1, the
value of τ0 is chosen in order to use a prescribed percentage of the number of training samples
N to obtain ∇f(x0). In all runs, such a percentage has been set to 0.4. Then, we proceeded as
follows. We computed ∇f(x0), via (4.55), with j = 1 and |D1,0|/N = 0.4. Then, we compute
τ0 so that (4.56), with τ1,0 = τ0 is satisﬁed as an equality. Finally, the value of κ at Step 1.2
of Algorithm 4.1 has been correspondingly set to 4τ (0)
1,0
 σ0/∥∇f(x0)∥
 2, with τ (0)
1,0 = τ0. This
way, the acceptance criterion of Step 1.2 is satisﬁed without further inner iterations (i.e., for
i = 0), when k = 0, and τ0 is indeed considered as the starting accuracy level for gradient
approximation at each execution of Step 1 of Algorithm 4.1. We will hereafter refer to such
implementation of Algorithm (2.1) coupled with Algorithm 4.1 as SARC. The numerical
tests of this section compare SARC with the corresponding variant in [3, Algorithm 3.1],
namely ARC-Dynamic, employing exact gradient evaluations, with γ1 = 1/γ, γ2 = γ3 = γ
and η1 = η2 = η. It is worth noticing that the problem (5.58) arises in the training of
an artiﬁcial neural network with no hidden layers and zero bias. Nevertheless, to cover the
general situation where SARC algorithm is applied to more complex neural networks, we
have followed the approach in [33] for what concerns the cost measure. Going into more
details, at the generic iteration k, we count the N forward propagations needed to evaluate
the objective in (5.58) at xk has a unit Cost Measure (CM), while the evaluation of the
approximated gradient at the same point requires |D1,k| additional backward propagations at
the weighed cost |D1,k|/N CM. Moreover, each vector-product ∇2f(xk)v (v ∈Rn), needed
at each iteration of the Barzilai-Borwein method used to minimise the cubic model at Step 3
of Algorithm 2.1, is performed via ﬁnite-diﬀerences, leading to additional |D2,k| forward and
backward propagations to compute ∇f(xk + hv), (h ∈R+), at the price of the weighted cost
2|D2,k|/N CM and a potential extra-cost |D2,k ∖(D1,k ∩D2,k)|/N CM to approximate ∇f(xk)
via uniform subsampling using the samples in D2,k. This latter approximation is computed


<!-- page 26 -->
26
once at the beginning of the Barzilai-Borwein procedure. Therefore, denoting by r the number
of Barzilai-Borwein iterations at iteration k, the increase of the CM at the k-th iteration of
ARC-Dynamic and SARC related to the derivatives computation is reported in Table 1.
ARC-Dynamic
SARC
1 + 2|D2,k|r/N
(|D1,k| + 2|D2,k|r + |D2,k ∖(D1,k ∩D2,k)|) /N
Table 1.
Increase of the CM at the k-th iteration of ARC-Dynamic and SARC related to the derivatives computation;
r denotes the number of performed Barzilai-Borwein iterations.
We will refer to the Cost Measure at Termination (CMT) as the main parameter to evaluate
the eﬃciency of the method within the numerical tests of the next section. The algorithms
have been implemented in Fortran language and run on an Intel Core i5, 1.8 GHz × 1 CPU,
8 GB RAM.
5.2. Numerical results
In this section we ﬁnally report statistics of the numerical tests performed by SARC and
ARC-Dynamic on the set of synthetic datasets from [3,5], whose main characteristics are
recalled in Table 2. They provide moderately ill-conditioned problems (see, e.g., Table 2) and
motivate the use of second order methods.
Dataset
Training N
n
Testing NT
cond
Synthetic1
9000
100
1000
2.5 · 104
Synthetic2
9000
100
1000
1.4 · 105
Synthetic3
9000
100
1000
4.2 · 107
Synthetic4
90000
100
10000
4.1 · 104
Synthetic6
90000
100
10000
5.0 · 106
Table 2.
Number of training samples (N), feature dimension (n), number of testing samples (NT ), 2-norm condition
number of the Hessian matrix at computed solution (cond).
For fair comparisons, the values of c used for each dataset in Table 2 to build the Hessian
approximation according to Step 2 of Algorithm 4.1 are chosen as in [3, Table 8.1].
In Table 3 we report, for both SARC and ARC-Dynamic algorithms, the total number of iter-
ations (n-iter), the value of Cost Measure at Termination (CMT) and the mean percentage of
saving (Save-M) obtained by SARC with respect to ARC-Dynamic on the synthetic datasets
listed in Table 2. Since the selection of the subsets Dj,k, j ∈{1, 2}, in (4.56) is uniformly
and randomly made at each iteration of the method, statistics in the forthcoming tables are
averaged over 20 runs.
Table 3 shows that the novel adaptive strategy employed by SARC results more eﬃcient than
ARC-Dynamic, reaching an ǫ-approximate ﬁrst-order stationary point at a lower CMT, in all
cases except from Synthetic6. This is obtained without aﬀecting the classiﬁcation accuracy
on the testing sets as it is shown in Table 4, where the average binary accuracy on the testing
sets achieved by methods under comparison is reported.

[CAPTION] Table 1.
Increase of the CM at the k-th iteration of ARC-Dynamic and SARC related to the derivatives computation;

[CAPTION] Table 2.
Number of training samples (N), feature dimension (n), number of testing samples (NT ), 2-norm condition

[CAPTION] Table 3 shows that the novel adaptive strategy employed by SARC results more eﬃcient than


<!-- page 27 -->
27
Dataset
ARC-Dynamic
SARC
n-iter
CMT
n-ter
CMT
Save-M
Synthetic1
11.1
130.84
10.0
95.27
27%
Synthetic2
10.6
109.56
10.2
93.08
15%
Synthetic3
11.2
109.64
10.0
97.52
11%
Synthetic4
11.0
124.07
10.4
100.48
19%
Synthetic6
10.0
84.18
10.1
106.31
−26%
Table 3.
Synthetic datasets. The columns are divided in two diﬀerent groups. ARC-Dynamic: average number of
iterations (n-iter) and CMT. SARC : average number of iterations (n-iter), CMT and mean percentage of saving (Save-
M) obtained by SARC over ARC-Dynamic. Mean values over 20 runs.
Method
Synthetic1
Synthetic2
Synthetic3
Synthetic4
Synthetic6
ARC-Dynamic
94.34%
92.68%
94.64%
95.52%
93.82%
SARC
93.18%
92.44%
93.62%
94.61%
93.70%
Table 4.
Synthetic datasets. Binary classiﬁcation rate at termination on the testing set employed by ARC-Dynamic
and SARC, mean values over 20 runs.
To give more evidence of the gain in terms of CMT provided by SARC on Synthetic1-
Synthetic4 along the iterative process, we display in Figure 1 the decrease of the training and
the testing loss versus the adopted cost measure CM, while Figure 2 is reserved to the plot of
the gradient norm versus CM. For such ﬁgures, a representative plot is considered among each
series of 20 runs obtained by SARC and ARC-Dynamic on each of the considered dataset.
In all cases, Figure 1 shows the savings gained by SARC in terms of the overall computational
cost, as well as the improvements in the training phase and the testing accuracy under the
same cost measure. More in general, we stress that second order methods show their strength
on these ill-conditioned datasets since all the tested procedures manage to reduce the norm
of the gradient and reach high accuracies in the classiﬁcation rate. Even if we believe that
reporting binary classiﬁcations accuracy obtained by each of the considered methods at ter-
mination is relevant in itself, we remark that the higher accuracy obtained at termination
by ARC-Dynamic (see Table 4) is just due to the fact the SARC stops earlier. This should
not be confused with a better performance of ARC-Dynamic, since Figure 1 highlights that,
along all datasets, when SARC stops its testing loss is sensibly below the corresponding one
performed by ARC-Dynamic at the same CMT value.
In Figure 3, we ﬁnally analyse the adaptive choices of the sample sizes Dj,k, j ∈{1, 2}, in
(4.56). As expected, the two strategies are more or less comparable when selecting the sample
sizes for Hessian approximations, while the number of samples used to compute gradient
approximations by SARC oscillates across all iterations, always remaining far below the full
sample size. In so doing, we outline that too small values of τ0 seem to have a bad inﬂuence
on the performance of SARC, while as τ0 increases it generally produces frequent saving in
the CMT, once that it is above a certain threshold value. In support of this observation, we
report in Figure 4 the variation of CMT against τ0 on Synthetic1 and Synthetic4. We ﬁnally
notice that, except for a few iterations at the ﬁrst stage of the iterative process, the sample

[CAPTION] Table 3.
Synthetic datasets. The columns are divided in two diﬀerent groups. ARC-Dynamic: average number of

[CAPTION] Table 4.
Synthetic datasets. Binary classiﬁcation rate at termination on the testing set employed by ARC-Dynamic


<!-- page 28 -->
28
size for Hessian approximation is lower than that used for gradient approximation. This is
in line with the theory as the gradient is eventually required to be more accurate than the
Hessian. In fact, the error in gradient approximation has to be of the order of ∥sk∥2, while
that in Hessian approximation has to be of the order of ∥sk∥, see Lemma 3.1 and 3.2.
6. Conclusion and perspectives
We have proposed the stochastic analysis of the process generated by an ARC algorithm
for solving unconstrained, nonconvex, optimisation problems under inexact derivatives in-
formation. The algorithm is an extension of the one in [3], since it employs approximated
evaluations of the gradient with the main feature of mantaining the dynamic rule for building
Hessian approximations, introduced and numerically tested in [3]. This kind of accuracy re-
quirement is always reliable and computable when an approximation of the exact Hessian is
needed by the scheme and, in contrast to other strategies such that the one in [18], does not
require the inclusion of additional inner loops to be satisﬁed. With respect to the framework
in [3], where in the ﬁnite-sum setting optimal complexity is restored with high probability, we
have here provided properties of the method when the adaptive accuracy requirements of the
derivatives involved in the model deﬁnition are not accomplished, with a view to search for
the number of expected steps that the process takes to reach the prescribed accuracy level.
The stochastic analysis is thus performed exploiting the theoretical framework given in [18],
showing that the expected complexity bound matches the worst-case optimal complexity of
the ARC framework. The possible lack of accuracy of the model has just the eﬀect of scaling
the optimal complexity we would derive from the deterministic analysis of the framework
(see, e.g., [3, Theorem 4.2]), by a factor which depends on the probability p of the model
being suﬃciently accurate. Numerical results conﬁrm the theoretical achievements and high-
light the improvements of the novel strategy on the computational cost in most of the tests
with no worsening of the binary classiﬁcation accuracy. This paper does not cover the case of
noisy functions ([20,22,28]), as well as the second-order complexity analysis. The stochastic
second-order complexity analysis of ARC methods with derivatives and function estimates
will be a challenging line of investigation for future work. Concerning the latter point, we
remark that a recent advance in [10], based on properties of supermartingales, has tackled
with the second-order convergence rate analysis of a stochastic trust-region method.
Funding: the authors are member of the INdAM Research Group GNCS and partially sup-
ported by INdAM-GNCS through Progetti di Ricerca 2019.
Acknowledgements. The authors dedicate this paper, in honor of his 70th birthday, to
Alfredo Iusem. Thanks are due to Coralia Cartis, Benedetta Morini and Philippe Toint for
fruitful discussion on stochastic complexity analysis and to two anonymous referees whose
comments signiﬁcantly improved the presentation of this paper.
References
[1] A. Bandeira, K. Scheinberg, L. Vicente. Convergence of Trust-Region Methods Based on Proba-
bilistic Models. SIAM Journal on Optimization 24(3), 1238–1264, 2014.


<!-- page 29 -->
29
[2] J. Barzilai, J. M. Borwein. Two-Point Step Size Gradient Methods. IMA Journal of Numerical
Analysis 8, 14–148, 1998.
[3] S. Bellavia, G. Gurioli, B. Morini. Adaptive Cubic Regularization Methods with Dynamic Inexact
Hessian Information and Applications to Finite-Sum Minimization. IMA Journal of Numerical
Analysis, https://doi.org/10.1093/imanum/drz076, 2019.
[4] S. Bellavia, G. Gurioli, B. Morini, Ph. L. Toint. Adaptive Regularization Algorithms with Inexact
Evaluations for Nonconvex Optimization. SIAM Journal on Optimization 29(4), 2281–2915, 2019.
[5] A.S. Berahas, R. Bollapragada, J. Nocedal.
An investigation of Newton-sketch and subsampled
Newton methods. Optimization Methods and Software, 35, 661–680, 2020.
[6] T. Bianconcini, G. Liuzzi, B. Morini, M. Sciandrone. On the use of iterative methods in cubic
regularization for unconstrained optimization. Computational Optimization and Applications 60,
35–57, 2015.
[7] E. G. Birgin, J. L. Gardenghi, J. M. Mart´ınez, S. A. Santos, Ph. L. Toint. Worst-case evalu-
ation complexity for unconstrained nonlinear optimization using high-order regularized models.
Mathematical Programming, Ser. A, 163(1-2), 359-368, 2017.
[8] E. G. Birgin, N. Kreji´c, J. M. Mart´ınez. Iteration and evaluation complexity for the minimization
of functions whose computation is intrinsically inexact. Mathematics of Computation 89 (321),
253–278, 2020.
[9] Y. Carmon, J.C. Duchi, O. Hinder, A. Sidford. Lower bounds for ﬁnding stationary points I.
Mathematical Programming, Series A, 1-50, 2019.
[10] J. Blanchet, C. Cartis, M. Menickelly, K. Scheinberg. Convergence Rate Analysis of a Stochastic
Trust-Region Method via Supermartingales. INFORMS Journal on Optimization 1(2), 92–119,
2019.
[11] L. Bottou, F. E. Curtis, J. Nocedal. Optimization Methods for Large-Scale Machine Learning.
SIAM Review 60(2), 223-311, 2018.
[12] C. Cartis, N. I. M. Gould, Ph. L. Toint. Complexity bounds for second-order optimality in uncon-
strained optimization. Journal of Complexity 28(1), 93–108, 2012.
[13] C. Cartis, N. I. M. Gould, Ph. L. Toint. On the oracle complexity of ﬁrst-order and derivative-free
algorithms for smooth nonconvex minimization
SIAM Journal on Optimization 22(1), 66–86,
2012.
[14] C. Cartis, N. I. M. Gould, Ph. L. Toint. An adaptive cubic regularisation algorithm for nonconvex
optimization with convex constraints and its function-evaluation complexity.
IMA Journal of
Numerical Analysis 32(4), 1662–1695, 2012.
[15] C. Cartis, N. I. M. Gould, Ph. L. Toint. Adaptive cubic overestimation methods for unconstrained
optimization. Part I: motivation, convergence and numerical results. Mathematical Programming,
Ser. A, 127, 245–295, 2011.
[16] C. Cartis, N. I. M. Gould, Ph. L. Toint. Adaptive cubic overestimation methods for unconstrained
optimization. Part II: worst-case function and derivative-evaluation complexity. Mathematical
Programming, Ser. A, 130(2), 295–319, 2011.
[17] C. Cartis, N. I. M. Gould, Ph. L. Toint. On the complexity of steepest descent, Newton’s and
regularized Newton’s method for nonconvex unconstrained optimization. SIAM Journal on Opti-
mization 20(6), 2833–2852, 2010.
[18] C. Cartis, K. Scheinberg. Global convergence rate analysis of unconstrained optimization methods
based on probabilistic models. Mathematical Programming, Ser. A, 159(2), 337–375, 2018.
[19] K. H. Chang, M. K. Li, H. Wan. Stochastic Trust-Region Response-Surface Method (STRONG) –
A New Response-Surface Framework for Simulation Optimization. INFORMS Journal on Com-
puting 25(2), 193–393, 2013.
[20] R. Chen. Stochastic Derivative-Free Optimization of Noisy Functions. Theses and Dissertations
2548, 2015.
[21] X. Chen, B. Jiang, T. Lin, S. Zhang. On Adaptive Cubic Regularized Newton’s Methods for Convex
Optimization via Random Sampling, arXiv:1802.05426, 2019.
[22] R. Chen, M. Menickelly, K. Scheinberg. Stochastic optimization using a trust-region method and
random models. Mathematical Programming 169(2), 447–487, 2018.


<!-- page 30 -->
30
[23] S. Gratton, C. W. Royer, L. N. Vicente, Z. Zhang. Complexity and global rates of trust-region
methods based on probabilistic models. IMA Journal of Numerical Analysis 38(3), 1579–1597,
2018.
[24] S. Gratton, C. W. Royer, L. N. Vicente, Z. Zhang. Direct Search Based on Probabilistic Descent.
SIAM Journal on Optimization 25(3), 1515–1541, 2015.
[25] J. M. Kohler, A. Lucchi. Sub-sampled cubic regularization for non-convex optimization. Proceed-
ings of the 34th International Conference on Machine Learning 70, 1895–1904, 2017.
[26] Y. Nesterov, B.T. Polyak. Cubic regularization of Newton method and its global performance.
Mathematical Programming, Ser. A, 108, 177–205, 2006.
[27] Y. Nesterov, V. Spokoiny. Random Gradient-Free Minimization of Convex Functions. Foundations
of Computational Mathematics 17(2), 527–566, 2017.
[28] C. Paquette, K. Scheinberg. A Stochastic Line Search Method with Expected Complexity Analysis.
SIAM Journal on Optimization 30(1), 349–376, 2020.
[29] S. Shashaani, F. S. Hashemi, R. Pasupathy. ASTRO-DF: A Class of Adaptive Sampling Trust-
Region Algorithms for Derivative-Free Simulation Optimization. SIAM Journal on Optimization
28(4), 3145–3176, 2018.
[30] J. Tropp.
An Introduction to Matrix Concentration Inequalities.
Foundations and Trends in
Machine Learning 8(1-2), 1–230, 2015.
[31] Z. Wang, Y. Zhou, Y. Liang, G. Lan. A note on inexact gradient and Hessian conditions for cubic
regularized Newton’s method. Operations Research Letters 47, 146–149, 2019.
[32] P.
Xu,
F.
Roosta-Khorasani,
M.
W.
Mahoney.
Newton-type
methods
for
non-
convex
optimization
under
inexact
Hessian
information.
Mathematical
Programming,
https://doi.org/10.1007/s10107-019-01405-z, 2019.
[33] P. Xu, F. Roosta-Khorasani, M. W. Mahoney. Second-Order Optimization for Non-Convex Ma-
chine Learning: An Empirical Study. Proceedings of the 2020 SIAM International Conference on
Data Mining, 2020.
[34] Z. Yao, P. Xu, F. Roosta-Khorasani, M. W. Mahoney. Inexact Non-Convex Newton-type Methods,
arXiv:1802.06925, 2018.
[35] D. Zhou, P. Xu, Q. Gu. Stochastic Variance-Reduced Cubic Regularization Methods. Journal of
Machine Learning Research 20, 1–47, 2019.


<!-- page 31 -->
31
20
40
60
80
100
120
Cost Measure
0.08
0.1
0.12
0.14
0.16
0.18
0.2
0.22
0.24
Training loss
SYNTHETIC1
ARC-Dynamic
SARC
20
40
60
80
100
120
Cost Measure
0.08
0.1
0.12
0.14
0.16
0.18
0.2
0.22
0.24
Testing loss
SYNTHETIC1
ARC-Dynamic
SARC
10
20
30
40
50
60
70
80
90
100
Cost Measure
0.08
0.1
0.12
0.14
0.16
0.18
0.2
0.22
0.24
Training loss
SYNTHETIC2
ARC-Dynamic
SARC
10
20
30
40
50
60
70
80
90
100
Cost Measure
0.08
0.1
0.12
0.14
0.16
0.18
0.2
0.22
0.24
Testing loss
SYNTHETIC2
ARC-Dynamic
SARC
10
20
30
40
50
60
70
80
90
100
Cost Measure
0.08
0.1
0.12
0.14
0.16
0.18
0.2
0.22
0.24
Training loss
SYNTHETIC3
ARC-Dynamic
SARC
10
20
30
40
50
60
70
80
90
100
Cost Measure
0.08
0.1
0.12
0.14
0.16
0.18
0.2
0.22
0.24
Testing loss
SYNTHETIC3
ARC-Dynamic
SARC
20
40
60
80
100
120
Cost Measure
0.1
0.15
0.2
0.25
Training loss
SYNTHETIC4
ARC-Dynamic
SARC
20
40
60
80
100
120
Cost Measure
0.1
0.15
0.2
0.25
Testing loss
SYNTHETIC4
ARC-Dynamic
SARC
Figure 1.
Synthetic datasets. Comparison of SARC (continuous line with asteriks) and ARC-Dynamic (dashed line
with triangles) against the considered cost measure CM. Each row corresponds to a diﬀerent synthetic dataset. Training
loss (left) and testing loss (right) against CM with logarithmic scale on the y axis.


**[Table p31.1]**
|  |  |  |  |  | ARC-Dynam SARC |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |


**[Table p31.2]**
|  |  |  |  |  | ARC-Dynam SARC |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |

[CAPTION] Figure 1.
Synthetic datasets. Comparison of SARC (continuous line with asteriks) and ARC-Dynamic (dashed line


<!-- page 32 -->
32
20
40
60
80
100
120
Cost Measure
10-2
Euclidean Gradient Norm
SYNTHETIC1
ARC-Dynamic
SARC
10
20
30
40
50
60
70
80
90
100
Cost Measure
10-2
Euclidean Gradient Norm
SYNTHETIC2
ARC-Dynamic
SARC
10
20
30
40
50
60
70
80
90
100
Cost Measure
10-2
Euclidean Gradient Norm
SYNTHETIC3
ARC-Dynamic
SARC
20
40
60
80
100
120
Cost Measure
0.006
0.008
0.01
0.012
0.014
0.016
0.018
Euclidean Gradient Norm
SYNTHETIC4
ARC-Dynamic
SARC
Figure 2.
Synthetic datasets. Euclidean norm of the gradient against CM (training set) with logarithmic scale on the
y axis. SARC (continuous line with asteriks), ARC-Dynamic (dashed line with triangles).


**[Table p32.1]**
|  |  |  |  |  | ARC-Dy SARC | namic |
| --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |


**[Table p32.2]**
|  |  |  |  |  |  |  |  |  | ARC-Dyn SARC | amic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |


**[Table p32.3]**
|  |  |  |  |  |  |  |  |  | SARC |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |


**[Table p32.4]**
|  |  |  |  |  | SARC |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |

[CAPTION] Figure 2.
Synthetic datasets. Euclidean norm of the gradient against CM (training set) with logarithmic scale on the


<!-- page 33 -->
33
1
2
3
4
5
6
7
8
9
10
11
Iterations
10
20
30
40
50
60
70
80
90
100
Sample size (percentage)
SYNTHETIC1
ARC-Dynamic (Hessian)
SARC (Hessian)
SARC (gradient)
1
2
3
4
5
6
7
8
9
10
11
Iterations
10
20
30
40
50
60
70
80
90
100
Sample size (percentage)
SYNTHETIC2
ARC-Dynamic (Hessian)
SARC (Hessian)
SARC (gradient)
1
2
3
4
5
6
7
8
9
10
11
Iterations
10
20
30
40
50
60
70
80
90
100
Sample size (percentage)
SYNTHETIC3
1
2
3
4
5
6
7
8
9
10
11
Iterations
10
20
30
40
50
60
70
80
90
100
Sample size (percentage)
SYNTHETIC4
ARC-Dynamic (Hessian)
SARC (Hessian)
SARC (gradient)
Figure 3.
Synthetic datasets. Sample size for Hessian approximations employed by ARC-Dynamic (dashed line with
triangles) and SARC (dashed line with asteriks), together with the sample size for gradient approximations considered
by SARC (dotted dashed line with asteriks) against iterations.
20
30
40
50
60
70
80
90
100
0
90
100
110
120
130
140
150
160
170
180
190
CMT
SYNTHETIC1
ARC-Dynamic
SARC
20
30
40
50
60
70
80
90
100
0
100
150
200
250
CMT
SYNTHETIC4
ARC-Dynamic
SARC
Figure 4.
Cost Measure at Termination (CMT) against τ0 among SARC (continuous line) and ARC-Dynamic (dashed
line) on Synthetic1 and Synthetic4.


**[Table p33.1]**
|  |  |  |  |  |  |  |  |  |  | essian) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  | ARC- SARC SARC | Dynamic (H (Hessian) (gradient) |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |


**[Table p33.2]**
|  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  | ARC- SARC SARC | Dynamic ( (Hessian) (gradient) | Hessian) |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |


**[Table p33.3]**
|  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  | ARC- SARC | Dynamic ( (Hessian) | Hessian) |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  | SARC | (gradient) |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |


**[Table p33.4]**
|  |  |  |  |  |  |  |  | ynamic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  | ARC-D SARC |  |
|  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |


**[Table p33.5]**
|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  | ARC-D SARC | ynamic |
|  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |

[CAPTION] Figure 3.
Synthetic datasets. Sample size for Hessian approximations employed by ARC-Dynamic (dashed line with

[CAPTION] Figure 4.
Cost Measure at Termination (CMT) against τ0 among SARC (continuous line) and ARC-Dynamic (dashed