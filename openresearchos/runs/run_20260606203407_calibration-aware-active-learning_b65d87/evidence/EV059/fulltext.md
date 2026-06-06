<!-- page 1 -->
Murphy’s Laws of AI Alignment:
Why the
Gap Always Wins
Madhava Gaikwad
mgaikwad@microsoft.com
Microsoft
∗
Abstract
We study reinforcement learning from human feedback under misspecifica-
tion. Contexts are drawn i.i.d. from a distribution µ over X. There exists
a measurable subset Xhard ⊂X with mass α = µ(Xhard) on which the
feedback channel is systematically biased. We consider pairwise preference
feedback and allow adaptive query selection with a budget of n queries.
We construct two environments w ∈{+, −} with reward functions rw that
differ only on Xhard. Let πw denote an optimal policy for rw, and define
the separation parameter
γ = Ex∼µ
   r+(x, π+(x)) −r+(x, π−(x))
   · 1{x ∈Xhard}
 
> 0.
On Xhard the pairwise label favors the truly better action with probability
1
2 + wε for some ε ∈(0, 1
2). Outside Xhard feedback is symmetric. For any
learner using at most n queries, Le Cam’s two point method combined with
a transcript level Kullback–Leibler decomposition yields
inf
bπ
sup
w∈{+,−}
 
Vw(πw)−Ew[Vw(bπ)]
 
≥γ
4 exp
 −n α κ(ε)
 
,
κ(ε) = 4ε atanh(2ε).
With access to a calibration oracle h(x) = 1{x ∈Xhard} that flags misspec-
ified contexts, an adaptive procedure that concentrates queries on Xhard
achieves expected gap at most η with
Q ≤
1
2αε2 logγ
η
queries, using a simple majority test on the flagged contexts. The constants
arise from per hit Bernoulli KL and standard testing bounds. The results
quantify how α and ε govern the sample complexity required to resolve
misspecification.
1
Introduction
Reinforcement learning from human feedback (RLHF) is often instantiated via pairwise
comparisons or scalar ratings that train a reward model used for policy optimization. In
many deployments the feedback channel is imperfect and the number of informative queries
is limited. This paper analyzes the fundamental limitations imposed by misspecification
under bounded query budgets and identifies minimal additional structure that removes
these limitations.
Setting.
Let (X, µ) be a context space with i.i.d. draws x ∼µ and a finite action set
A. A policy π : X →A has value Vw(π) = Ex∼µ[rw(x, π(x))] under world w. There is a
measurable subset Xhard ⊂X with α = µ(Xhard) on which feedback is misspecified. Two
worlds w ∈{+, −} are defined by reward functions rw that differ only on Xhard. Let πw
denote an optimal policy for rw. Define the separation parameter
γ = Ex∼µ
   r+(x, π+(x)) −r+(x, π−(x))
   · 1{x ∈Xhard}
 
> 0.
∗This is a working paper open to collab, revision and critique. This work was done in individual
capacity and does not represent the views of my employer.
1
arXiv:2509.05381v3  [cs.AI]  15 Sep 2025


<!-- page 2 -->
On x ∈Xhard the pairwise label equals the indicator that the truly better action is preferred,
flipped with Massart style bias of magnitude ε ∈(0, 1
2):
Pr
w {label favors the truly better action | x ∈Xhard} =
1
2 + wε.
On x ∈Xeasy = X \ Xhard the channel is symmetric with probability 1
2. The learner may
adaptively select which queries to issue, up to a budget of n queries.
Lower bound.
Let Pw be the joint law of the full transcript under world w. The transcript
level Kullback–Leibler divergence satisfies
DKL(P+ ∥P−) =
n
X
t=1
EP+ 
DKL
 P+(Yt | Ht−1) ∥P−(Yt | Ht−1)
  
≤n α κ(ε),
where each term is nonzero only when xt ∈Xhard and
κ(ε) = DKL
 Ber
  1
2 + ε
   Ber
  1
2 −ε
  
= 4ε atanh(2ε).
A standard testing inequality converts the KL budget into a lower bound on the Bayes error
for distinguishing the two worlds, which implies the finite sample value gap
inf
bπ
sup
w∈{+,−}
 
Vw(πw) −Ew[Vw(bπ)]
 
≥γ
4 exp
 −n α κ(ε)
 
.
The bound holds for adaptive procedures since the chain rule is applied conditionally on the
observed history.
Upper bound with a calibration oracle.
Suppose an oracle h : X →{0, 1} indicates mem-
bership in Xhard. Draw contexts until h(x) = 1. The expected number of draws per hit
is 1/α.
On each hit, collect a bounded preference bit.
A majority test over T hits on
Xhard errs with probability at most exp(−2Tε2) by a bounded difference inequality. Setting
T =
1
2ε2 log(γ/η) yields an expected value gap at most η with
Q = T
α ≤
1
2αε2 logγ
η
total queries. This matches the dependence on α and ε in the lower bound up to constants.
Discussion.
The analysis isolates three parameters that govern difficulty under misspeci-
fication: prevalence α, bias magnitude ε, and separation γ. The lower bound follows from
indistinguishability of two worlds on Xhard, and the upper bound shows that the ability to
route queries to that set is sufficient to overcome the barrier. The same blueprint extends
to scalar ratings when the per hit log likelihood ratio is bounded by a quantity of order ε2.
2
Formal Setup and Main Results
We study a contextual decision problem with a possibly misspecified preference channel.
Contexts are drawn i.i.d. from a probability space (X, F, µ). The action set A is finite.
For world w ∈{+, −}, a reward function rw : X × A →[0, 1] induces the value of a policy
π : X →A,
Vw(π) = Ex∼µ
 
rw(x, π(x))
 
.
2.1
Environment family
There exists a measurable subset Xhard ⊂X with mass α := µ(Xhard) ∈(0, 1), and Xeasy =
X \ Xhard. The two worlds share the same reward on Xeasy and differ only on Xhard. Let
πw be an optimal policy for rw. Define the separation
γ = Ex∼µ
   r+(x, π+(x)) −r+(x, π−(x))
   · 1{x ∈Xhard}
 
> 0.
2


<!-- page 3 -->
2.2
Feedback channel
A round t presents a context xt ∼µ. If a pairwise query is issued, a binary label Yt ∈{0, 1}
is returned. On x ∈Xeasy, the label is symmetric with mean 1/2. On x ∈Xhard, the label
has Massart type bias of fixed magnitude ε ∈(0, 1
2):
Pr
w {label favors the truly better action | x ∈Xhard} =
1
2 + wε.
Scalar ratings admit the same analysis when the per query log likelihood ratio on Xhard is
bounded by a quantity of order ε2.
2.3
Learning protocol
The learner interacts for at most n queries. At each t, after observing xt and history Ht−1,
it decides whether to issue a query. Let It ∈{0, 1} indicate that a query is issued and write
the decision as a measurable function qt : X × Ht−1 →[0, 1] with Pr(It = 1 | xt, Ht−1) =
qt(xt, Ht−1).
The transcript law under world w is Pw.
The learner outputs a policy bπ
adapted to the transcript. We evaluate the minimax gap
∆n = inf
alg
sup
w∈{+,−}
 
Vw(πw) −Ew[Vw(bπ)]
 
.
2.4
Per hit information
Lemma 1 (Bernoulli KL on opposite biases). For |ε| < 1
2,
κ(ε) := DKL
 Ber
  1
2 + ε
   Ber
  1
2 −ε
  
= 4ε atanh(2ε) ≤
8ε2
1 −4ε2 .
Proof. Write p = 1
2 + ε and q = 1
2 −ε. Then
DKL(Ber(p)∥Ber(q)) = p log p
q + (1 −p) log 1 −p
1 −q = (2p −1) log 1 + 2ε
1 −2ε = 2ε log 1 + 2ε
1 −2ε.
Use atanh(z) = 1
2 log 1+z
1−z and atanh(z) ≤z/(1 −z2) for |z| < 1. □
2.5
Hit count bound
Lemma 2 (Expected queried hits on Xhard). For any predictable query policy with at most
n queries,
n
X
t=1
EP+ 
It1{xt ∈Xhard}
 
≤n α.
The same bound holds under P−. Proof. Condition on Ht−1. Since xt ∼µ is independent
of Ht−1,
E[It1{xt ∈Xhard} | Ht−1] =
Z
qt(x, Ht−1)1{x ∈Xhard} µ(dx) ≤α · sup
x qt(x, Ht−1) ≤α.
Summing over t and taking expectation gives the claim. □
2.6
Lower bound
Theorem 1 (Finite sample impossibility). For any adaptive learner using at most n queries,
∆n ≥γ
4 exp
 −n α κ(ε)
 
.
Proof. Apply the KL chain rule to the transcript:
DKL(P+ ∥P−) =
n
X
t=1
EP+ 
DKL
 P+(Yt | Ht−1, xt, It)
 P−(Yt | Ht−1, xt, It)
  
.
The inner KL is zero unless It = 1 and xt ∈Xhard. In that case it is at most κ(ε) by Lemma
1. Taking expectation and using Lemma 2 yields DKL(P+ ∥P−) ≤n α κ(ε). A standard
testing inequality gives that the Bayes error for distinguishing the two worlds under equal
priors is at least 1
4 exp(−DKL). A testing mistake induces value gap at least γ. Combine
the two relations. □
3


<!-- page 4 -->
2.7
Oracle upper bound
Theorem 2 (Calibration oracle suffices). Assume access to h(x) = 1{x ∈Xhard}. For any
η ∈(0, γ) there exists a procedure that uses
Q ≤
1
2 α ε2 logγ
η
queries and returns bπ such that supw∈{+,−}
 Vw(πw) −Ew[Vw(bπ)]
 
≤η.
Proof.
Draw
contexts until h(x) = 1. The expected number of draws per hit is 1/α. On each hit issue a
pairwise query and record the binary outcome Zi with mean 1
2 ± ε depending on w. After T
hits, decide bw by the sign of PT
i=1(Zi −1
2) and output π b
w. A bounded difference inequality
gives Pr( bw ̸= w) ≤exp(−2Tε2). Setting T =
1
2ε2 log(γ/η) makes the expected value gap at
most η. The expected total queries equal Q = T/α. □
2.8
Noisy oracle and minimality
Theorem 3 (Minimality within x-only binary oracles). Let ˜h(x) ∈{0, 1} satisfy Pr(˜h = 1 |
x ∈Xhard) = τ and Pr(˜h = 1 | x ∈Xeasy) = ϕ. Let the procedure keep only ˜h(x) = 1
contexts and run the same majority test on hits. To obtain T true hits in expectation one
needs at least T/(ατ) kept draws, hence
Q ≥
1
2 α τ ε2 logγ
η .
Therefore h(x) = 1{x ∈Xhard} with τ = 1 and ϕ = 0 is minimal in this class. Proof.
The fraction of kept draws that are true hits equals
ατ
ατ+(1−α)ϕ ≤τ. The stated lower bound
follows. □
2.9
Remark on scalar ratings
If on Xhard the scalar rating R ∈[0, 1] satisfies log dP+
dP−(R) with variance proxy bounded by
cε2 uniformly in the history, then the per hit KL is ≤Cε2 for a constant C. The proofs
above apply with κ(ε) replaced by Cε2.
3
Empirical Indications
This section specifies synthetic protocols that reflect the parameters α, ε, and γ, and diag-
nostics that probe the mechanisms in Theorems 1 and 2. The goal is to visualize the gap
predicted by the bound and the effect of concentrating queries on Xhard.
3.1
Synthetic environment
Fix dimension d. Draw contexts x ∈Rd i.i.d. from a mixture µ = (1 −α) N(0, Id) +
α N(µh, Id). Define Xhard as the support of the second component, so µ(Xhard) = α. Let
A = {a0, a1}. Choose unit vectors θ, υ ∈Rd with θ ⊥υ. Set
r+(x, a1) −r+(x, a0) = sign(θ⊤x) · 1{x ∈Xeasy} + sign(υ⊤x) · 1{x ∈Xhard}.
Define r−by flipping the sign on Xhard. This induces π+(x) = a1 if r+(x, a1) ≥r+(x, a0)
and π−analogously. The separation γ is controlled by the margin distribution on Xhard.
Pairwise feedback is generated by the Massart model with bias ε on Xhard and symmetry
on Xeasy.
3.2
Preference learner and proxy
Collect n pairwise comparisons with adaptive selection. Fit a logistic preference model bs(x)
that predicts the advantage of a1 over a0. Define the proxy reward br(x, a1)−br(x, a0) = bs(x).
4


<!-- page 5 -->
For a temperature parameter λ ≥0, define a stochastic policy
πλ(x) =
 a1
with prob. σ
 λ bs(x)
 
,
a0
otherwise,
σ(u) = (1 + e−u)−1.
Let Vw(πλ) be the true value under world w and bV (πλ) = Ex∼µ[br(x, πλ(x))] the proxy value.
3.3
Diagnostics
D1. Gap versus optimization pressure.
Compute G(λ) = V+(π+) −V+(πλ) and the proxy
bG(λ) = bV (π+) −bV (πλ) for λ on a grid. The theory predicts bV (πλ) increases with λ while
V+(πλ) can plateau or decrease once the policy mass shifts toward Xhard.
D2.
In distribution to shifted distribution.
Construct a shifted test distribution µρ by
increasing the hard mass to αρ = (1 + ρ)α and renormalizing. Measure Gρ(λ) under µρ.
The gap should increase with ρ, consistent with the dependence on α in Theorem 1.
D3. Query routing.
Implement a calibrated flagger bh(x) ∈{0, 1} that predicts membership
in Xhard using a small held out audit set. Restrict data collection to bh(x) = 1 and compare
the number of queries required to drive G(λ) ≤η with and without routing. The observed
query counts should follow the 1/α factor in Theorem 2. When bh has true positive rate τ
and false positive rate ϕ, the query count scales like 1/(ατ).
3.4
Tilting calculus
Define the tilted sampling law qλ on X by
dqλ
dµ (x) =
exp
 λ bs(x)
 
Eµ[exp(λ bs(X))].
Let H(x) = 1{x ∈Xhard}. The hard mass under qλ is
qλ(Xhard) = Eµ
 
H(X) exp(λ bs(X))
 
Eµ[exp(λ bs(X))]
.
Proposition 1. The derivative satisfies
d
dλ log qλ(Xhard) = Covqλ
 H(X), bs(X)
  .
qλ(Xhard).
In particular, if Covqλ(H(X), bs(X)) > 0 for λ in an interval, then qλ(Xhard) is strictly
increasing on that interval. Proof. Differentiate numerator and denominator and apply the
quotient rule. Use that
d
dλEµ[f(X) exp(λ bs(X))] = Eqλ[f(X)bs(X)]. □
When bs inherits bias from the preference channel, the covariance in Proposition 1 is positive,
which explains the observed increase of hard mass under stronger optimization.
3.5
Protocol summary
Choose parameters α ∈{0.01, 0.05, 0.1}, ε ∈{0.05, 0.1}, d ∈{10, 50}, and set n on a grid.
For each setting:
1. Generate a training transcript with at most n queries using an adaptive but oracle
free policy as in Section 2.3.
2. Fit bs and evaluate V+(πλ) and bV (πλ) for λ on a grid.
3. Repeat with query routing using bh trained on a small audit set and record the total
queries needed to reach a target gap η.
Report medians over 10 seeds with 90 percent intervals.
5


<!-- page 6 -->
3.6
Figures
If graphicx is available, include three panels:
• Gap versus λ: plot V+(πλ) and bV (πλ) for several (α, ε).
• Gap under shift: plot Gρ(λ) for ρ ∈{0, 0.5, 1}.
• Query routing: plot empirical query count against the target η with and without
bh; overlay the curve Q = (2αε2)−1 log(γ/η) for reference.
Placeholders:
4
Catalogue of Alignment Laws
This section states concrete properties that follow from the formal setup.
Each item is
expressed as a definition and a sufficient condition or bound. The quantities α, ε, γ, the
score bs, the tilted law qλ, and the policy family πλ are as introduced earlier.
4.1
Optimization drift under proxy tilting
Let H(x) = 1{x ∈Xhard}. Define ρ(λ) = qλ(Xhard).
Proposition 2 (Drift of hard mass).
d
dλ log ρ(λ) = Covqλ
 H(X), bs(X)
 
ρ(λ)
.
In particular, if Covqλ(H(X), bs(X)) > 0 on an interval, then ρ(λ) is strictly increasing on
that interval.
Proof. Differentiate ρ(λ) using the tilt definition and the quotient rule, as in Proposition 1.
4.2
Proxy improvement with true value saturation
Consider a fixed decision rule π and the true value under distribution tilting V dist
w
(λ) =
Eqλ[rw(X, π(X))]. Similarly define the proxy value bV dist(λ) = Eqλ[br(X, π(X))].
Proposition 3 (Sign of local change).
d
dλV dist
w
(λ) = Covqλ
 rw(X, π(X)), bs(X)
 
,
d
dλ
bV dist(λ) = Covqλ
 br(X, π(X)), bs(X)
 
.
If the second covariance is positive and the first is nonpositive on an interval, then the proxy
increases while the true value saturates or decreases on that interval.
Proof. Differentiate Eqλ[f(X)] and use the standard covariance identity under exponential
tilting.
4.3
Distribution shift sensitivity
Let µρ be a shift that changes the hard mass to αρ = (1 + ρ)α with ρ ≥0 and keeps the
conditional laws fixed. For a fixed policy π,
Proposition 4 (Monotonicity in hard mass).
Vw,ρ(π) = Eµρ[rw(X, π(X))] = (1 −αρ) E[rw | Xeasy] + αρ E[rw | Xhard].
If E[rw | Xhard] < E[rw | Xeasy] for π, then Vw,ρ(π) is nonincreasing in ρ.
Proof. Linear decomposition in αρ.
6


<!-- page 7 -->
4.4
Preference bias and sign estimation error
Let bw be any estimator of the world sign based on a transcript of at most n queries and
suppose the output policy equals π b
w.
Proposition 5 (Lower bound on sign error).
Pr( bw ̸= w) ≥
1
4 exp
 −n α κ(ε)
 
.
Consequently,
E
h
1{π b
w(X) ̸= πw(X)} 1{X ∈Xhard}
i
≥α Pr( bw ̸= w).
Proof. The first inequality follows from the transcript KL bound and a standard testing
inequality used in Theorem 1. The second relation holds because on Xhard the optimal
actions of the two worlds disagree by construction.
4.5
Reward score divergence
Define the divergence between proxy and true values for a policy π under qλ,
∆PV(λ, π) = bV dist(λ) −V dist
w
(λ).
Proposition 6 (Local growth condition). If Covqλ(br(X, π(X)) −rw(X, π(X)), bs(X)) > 0 on
an interval, then ∆PV(λ, π) is strictly increasing on that interval.
Proof. Differentiate ∆PV and apply the covariance identity.
4.6
Multi objective trade off
Let g1, g2, g3 be three bounded objectives on X × A.
For each j let π(j) maximize
Eµ[gj(X, π(X))]. Assume there exists a subset S ⊆Xhard with µ(S) = αS > 0 on which
the three optimal actions disagree pairwise. Define
mj = essinfx∈S
  gj(x, π(j)(x)) −gj(x, a)
  
where a ranges over the two nonoptimal actions at x. Then for any policy π,
3
X
j=1
 
E[gj(X, π(j)(X))] −E[gj(X, π(X))]
 
≥αS min{m1, m2, m3}.
(1)
In particular, at least one objective loses at least 1
3αS min{m1, m2, m3}.
Proof. On S any single action can match at most one of the three optimal actions. At
each x ∈S at least two terms in the sum are lower bounded by the corresponding margins.
Integrate over S and use the definition of mj.
Relation (1) formalizes a three way tension when objectives disagree on a biased subset.
5
Vision and Outlook
This section lists technical directions suggested by the bounds and by the diagnostics.
5.1
Calibration oracles
Design x-only flaggers that approximate h(x) = 1{x ∈Xhard} with high true positive rate
τ and low false positive rate ϕ. Theorem 3 shows that τ controls the leading 1/(ατ) factor
in query complexity. Candidate constructions:
• Residual based detectors: train a preference model, compute residuals on a held
out audit set, and fit a classifier bh to predict large residual regions.
7


<!-- page 8 -->
• Disagreement based detectors: maintain two preference models trained on disjoint
views and flag contexts with large predictor disagreement.
• Intervention based detectors: where feasible, inject counterfactual label queries to
estimate whether the label channel departs from a symmetric model on the candi-
date region.
Each method yields an empirical estimate of τ and ϕ, which can be plugged into Theorem
3.
5.2
Active routing and allocation
Given an estimated bh, allocate the query budget to maximize the expected number of true
hits. A simple rule selects x with probability proportional to bh(x). When per hit costs differ
across context types, incorporate an importance weight and reuse the analysis with a cost
adjusted α.
5.3
Noisy oracle analysis
Extend the upper bound to oracles that supply a confidence score u(x) ∈[0, 1]. Under a
monotone likelihood ratio assumption for u with respect to the event x ∈Xhard, a thresh-
olding rule on u maximizes the hit rate among rules with the same keep rate. The expected
query complexity becomes Q ≈T/E[u(X)1{X ∈Xhard}].
5.4
Function approximation and continuous actions
Replace finite A with a compact action space and assume Lipschitz rewards. If the per hit
log likelihood ratio remains O(ε2) and the hard set induces a margin condition on optimal
actions, the transcript KL bound carries over and the same dependence on α and ε appears.
Formalizing the margin requirement for continuous actions is a natural next step.
5.5
Sequential testing and stopping
The majority test in Theorem 2 can be replaced by a sequential probability ratio test. This
yields an expected hit count of order ε−2 log(γ/η) with a data dependent stopping time and
preserves the 1/α factor after accounting for routing.
5.6
Estimation of α, ε, and γ
Estimate α via audit sampling and density estimation on bh(x) = 1. Estimate ε from the
empirical bias of preference labels on flagged contexts, using held out adjudicated compar-
isons. Estimate γ from the observed margin between candidate policies on Xhard. These
estimates enable empirical curves for the lower bound (γ/4) exp(−nακ(ε)) versus n.
5.7
Evaluation protocols
Report, for each experiment, the estimated (α, ε, γ), the achieved sign error rate, and the
query allocation across bh(x). Include plots of Vw(πλ) and bV (πλ) and overlay the theoretical
reference Q = (2αε2)−1 log(γ/η) for routing.
5.8
Limitations
The analysis assumes i.i.d. contexts, a fixed biased subset with constant ε, and a per hit
KL that scales as O(ε2). When the bias magnitude varies across Xhard or evolves over time,
the bounds apply with κ(ε) replaced by an average per hit KL. Extending the lower bound
to non i.i.d. context processes and to structured feedback channels is open.
8


<!-- page 9 -->
6
Related Work
RLHF foundations and variants.
RLHF fine-tunes pretrained models using human prefer-
ence data and a learned reward model (Ouyang et al., 2022). Preference-based optimization
without an explicit RL step, such as Direct Preference Optimization, replaces the policy
improvement stage with a preference-matching loss (Rafailov et al., 2023). These pipelines
differ in optimization details but share the same core ingredients: a proxy signal derived
from preferences, a finite feedback budget, and policy updates that place weight on contexts
favored by the proxy. The analysis in Sections 2 and 3 targets this shared structure rather
than any specific training recipe.
Analyses and limits of RLHF.
Surveys and critical analyses document issues arising from
reward misspecification, annotator bias, and metric alignment (Casper et al., 2023). The
lower bound in Theorem 1 complements these accounts with a finite-sample indistinguisha-
bility argument under a biased slice of the context distribution. The matching upper bound
with a calibration oracle (Theorem 2) identifies the minimal structure needed to redirect
queries to the informative slice.
Mitigation and policy shaping.
Mitigation work proposes training-time and inference-time
controls that reduce over-optimization and undesired behaviors (Lin et al., 2023). These
methods can be interpreted as adding structure to the supervision loop. Our oracle formu-
lation makes the routing requirement explicit: a flagger that identifies when the preference
channel departs from a symmetric model on a subset of contexts is sufficient to recover the
correct sign with the stated sample complexity.
Constitutional and AI-feedback approaches.
Constitutional AI and related AI-feedback
methods replace part of the human signal with rule-based or model-generated judgments
(Bai et al., 2022). These approaches instantiate specific proxy signals and objective shaping.
The lower bound applies whenever the induced proxy is biased on a subset with nonzero
mass and the query budget is bounded; the oracle upper bound gives a condition under
which additional structure overcomes the indistinguishability.
Proxy objectives and optimization pressure.
The divergence between proxy improve-
ment and true objective is classically studied through Goodhart-type effects (Manheim
& Garrabrant, 2018). The tilting view in Section 3.4 provides a mechanism-level account
in the preference-learning setting: optimization that amplifies contexts correlated with the
proxy induces a change of measure that increases the mass on biased regions, which in turn
enlarges the gap predicted by Theorem 1.
Evaluation distributions and shift.
Choice of evaluation distribution affects conclusions
about alignment quality and stability. Shifts that increase the prevalence of contexts where
the proxy is biased can mask or reveal failures. The diagnostics in Section 3.3 illustrate
this dependence and connect it to explicit changes in the hard-set mass parameter. These
observations are consistent with discussions of distributional choice and evaluation under
shift (Rastogi et al., 2025).
References
Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy
Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Consti-
tutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073, 2022.
Stephen Casper, Dylan Hadfield-Menell, Geoffrey Irving, et al. Open problems and fun-
damental limitations of reinforcement learning from human feedback.
arXiv preprint
arXiv:2307.15217, 2023.
Yong Lin, Hangyu Lin, Wei Xiong, Shizhe Diao, Jianmeng Liu, Jipeng Zhang, Rui Pan,
Haoxiang Wang, Wenbin Hu, Hanning Zhang, et al. Mitigating the alignment tax of rlhf.
arXiv preprint arXiv:2309.06256, 2023.
9


<!-- page 10 -->
David Manheim and Scott Garrabrant.
Categorizing variants of goodhart’s law.
arXiv
preprint arXiv:1803.04585, 2018.
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin,
Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language
models to follow instructions with human feedback.
Advances in neural information
processing systems, 35:27730–27744, 2022.
Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and
Chelsea Finn. Direct preference optimization: Your language model is secretly a reward
model. Advances in neural information processing systems, 36:53728–53741, 2023.
Charvi Rastogi, Tian Huey Teh, Pushkar Mishra, Roma Patel, Ding Wang, Mark D´ıaz,
Alicia Parrish, Aida Mostafazadeh Davani, Zoe Ashwood, Michela Paganini, et al. Whose
view of safety? a deep dive dataset for pluralistic alignment of text-to-image models.
arXiv preprint arXiv:2507.13383, 2025.
A
Appendix A: Proofs
A.1
Auxiliary inequalities
Lemma 3 (Bretagnolle–Huber). For probability measures P, Q on the same measurable space,
inf
φ
 
1
2P(φ = 0) + 1
2Q(φ = 1)
 
≥
1
4 exp
 −DKL(P∥Q)
 
,
where the infimum is over all tests φ ∈{0, 1}. Equivalently, with equal priors, the Bayes
error is at least 1
4 exp(−DKL).
Lemma 4 (Chain rule for adaptive transcripts). Let Ht be the sigma field generated by
the transcript up to round t, and let Zt be the round-t observation with conditional laws
P(Zt | Ht−1) and Q(Zt | Ht−1). Then
DKL(PHn∥QHn) =
n
X
t=1
EP
h
DKL
 P(Zt | Ht−1)∥Q(Zt | Ht−1)
 i
.
A.2
Proof of Lemma 1
Let p = 1
2 + ε and q = 1
2 −ε. Then
DKL(Ber(p)∥Ber(q)) = p log p
q + (1 −p) log 1 −p
1 −q = (2p −1) log 1 + 2ε
1 −2ε = 2ε log 1 + 2ε
1 −2ε.
Using atanh(z) = 1
2 log 1+z
1−z gives κ(ε) = 4ε atanh(2ε). For |z| < 1, atanh(z) ≤z/(1 −z2),
which yields κ(ε) ≤8ε2/(1 −4ε2).
A.3
Proof of Lemma 2
Condition on Ht−1. Since xt ∼µ is independent of Ht−1,
E[It1{xt ∈Xhard} | Ht−1] =
Z
qt(x, Ht−1) 1{x ∈Xhard} µ(dx) ≤α.
Summing over t and taking expectation proves the claim.
A.4
Proof of Theorem 1
Write Tn = (x1:n, I1:n, Y1:n) for the transcript. By Lemma 4,
DKL(P+∥P−) =
n
X
t=1
EP+
h
DKL
 P+(Yt | Ht−1, xt, It)∥P−(Yt | Ht−1, xt, It)
 i
.
10


<!-- page 11 -->
The inner divergence is zero unless It = 1 and xt ∈Xhard, and in that case it is at most
κ(ε) by Lemma 1. Therefore,
DKL(P+∥P−) ≤κ(ε)
n
X
t=1
EP+ 
It1{xt ∈Xhard}
 
≤n α κ(ε),
using Lemma 2.
Under equal priors, Lemma 3 implies Bayes testing error at least
1
4 exp(−DKL). A testing mistake induces value gap at least γ, hence
∆n ≥γ
4 exp
 −n α κ(ε)
 
.
A.5
Proof of Theorem 2
Query until observing T hits with h(x) = 1. The expected number of draws per hit is 1/α,
so the expected total number of queries is Q = T/α. On each hit, the observed bit has
mean 1
2 ± ε depending on w. Let ST = PT
i=1(Zi −1
2). By a bounded differences inequality,
Pr(sign(ST ) ̸= sign(w)) ≤exp(−2Tε2). If bw is the sign decision and the learner outputs
π b
w, then
sup
w
 
Vw(πw) −Ew[Vw(π b
w)]
 
≤γ Pr( bw ̸= w) ≤γ exp(−2Tε2).
Setting T =
1
2ε2 log(γ/η) yields the target gap η with Q ≤(2αε2)−1 log(γ/η).
A.6
Proof of Theorem 3
Let ˜h have true positive rate τ on Xhard and false positive rate ϕ on Xeasy. Among kept
draws with ˜h(x) = 1, the fraction of true hits equals
p =
ατ
ατ + (1 −α)ϕ ≤τ.
To obtain T true hits in expectation one needs at least T/p ≥T/(ατ) total draws. Repeating
the argument in the proof of Theorem 2 shows that achieving gap η requires
Q ≥
1
2 α τ ε2 logγ
η .
The oracle h(x) = 1{x ∈Xhard} has τ = 1 and ϕ = 0, which minimizes Q in this class.
A.7
Proof of Proposition 1
Let Zλ(f) = Eµ[f(X) exp(λbs(X))] and Zλ = Zλ(1). Then
qλ(Xhard) = Zλ(H)
Zλ
.
Differentiate and use Z′
λ(f) = Eµ[f(X)bs(X) exp(λbs(X))] = Zλ Eqλ[f(X)bs(X)]:
d
dλ log qλ(Xhard) = Z′
λ(H)
Zλ(H) −Z′
λ
Zλ
= Eqλ[H(X)bs(X)]
Eqλ[H(X)]
−Eqλ[bs(X)] = Covqλ(H(X), bs(X))
qλ(Xhard)
.
A.8
Scalar ratings
Assume that on Xhard the log likelihood ratio ℓ= log dP+
dP−(R) satisfies E[ℓ| Ht−1, xt ∈
Xhard] = m and Var(ℓ| Ht−1, xt ∈Xhard) ≤σ2 with m, σ2 = O(ε2) uniformly. Then the
per hit KL equals E[ℓ] ≤Cε2, which can be used in the chain rule in place of κ(ε).
B
Appendix B: Extended Catalogue of Alignment Laws
This appendix states extensions of the identities and bounds used in the main text. Through-
out, H = 1{X ∈Xhard}, qλ is the exponential tilt by bs, and πλ is the stochastic policy that
selects a1 with probability σ(λbs(x)).
11


<!-- page 12 -->
B.1
Heterogeneous bias and aggregated information
Suppose the bias varies across the hard set: there is a measurable ε(x) ∈(0, 1
2) on Xhard
and the per hit KL equals κ(x) = 4ε(x) atanh(2ε(x)). Let K = E[κ(X) | X ∈Xhard].
Proposition 7. For any adaptive procedure with at most n queries,
DKL(P+∥P−) ≤n α K,
∆n ≥γ
4 exp(−n α K).
Proof. Replace κ(ε) by the conditional expectation of κ(X) inside the chain rule and follow
the proof of Theorem 1.
B.2
Mixtures of hard subsets
Let Xhard = SJ
j=1 Sj be a disjoint union with masses αj and biases εj. Then the per hit
KL on Sj equals κj = 4εj atanh(2εj).
Proposition 8.
DKL(P+∥P−) ≤n
J
X
j=1
αjκj,
∆n ≥γ
4 exp
 
−n
X
j
αjκj
 
.
Proof. Apply Lemma 2 on each Sj and sum.
B.3
Tilt identities and monotonicity
For any bounded f,
d
dλEqλ[f(X)] = Covqλ(f(X), bs(X)).
In particular, for f = H this recovers Proposition 1.
For f = rw(X, π(X)) and f =
br(X, π(X)) it yields the signs in Section 4.2.
B.4
Over-optimization threshold
Assume there exists c > 0 such that
Eqλ
 
rw(X, πλ(X)) | X ∈Xhard
 
≤Eqλ
 
rw(X, πλ(X)) | X ∈Xeasy
 
−c
for λ in an interval. If ρ(λ) = qλ(Xhard) is strictly increasing on that interval, then
d
dλVw(πλ) = d
dλ
 
(1 −ρ)E[rw | Xeasy] + ρ E[rw | Xhard]
 
≤−c ρ′(λ).
Hence Vw(πλ) is strictly decreasing where ρ′(λ) > 0.
B.5
Sequential testing
Let LT = PT
i=1 log Pr(Zi|w=+)
Pr(Zi|w=−) over hits. A sequential probability ratio test that stops at the
first time T when LT /∈(a, b) with thresholds chosen for error probabilities (δ, δ) satisfies
E[T | w = ±] ≤log 1−δ
δ
κ(ε) .
With routing by h, the expected number of queries is Q ≤1
αE[T]. This preserves the 1/α
factor and replaces the fixed-sample T by a data-dependent stopping time.
12


<!-- page 13 -->
B.6
Composite objectives
Let g1, . . . , gm be bounded objectives. Suppose there is a subset S ⊆Xhard with µ(S) =
αS > 0 such that for each x ∈S the action that maximizes gj(x, ·) differs across at least k
indices j. Define
mj = essinfx∈S
min
a̸=a(j)(x)
 gj(x, a(j)(x)) −gj(x, a)
 
,
where a(j)(x) is the gj-optimal action. Then for any π,
m
X
j=1
 
E[gj(X, π(j)(X))] −E[gj(X, π(X))]
 
≥k αS min
j
mj.
At least one objective incurs a gap of at least k αS minj mj/m.
B.7
Noisy flaggers with scores
Let u : X →[0, 1] be a score with class-conditional densities that satisfy a monotone
likelihood ratio with respect to the event X ∈Xhard. For a keep rate constraint β ∈(0, 1),
a threshold rule 1{u(x) ≥t} maximizes the true hit rate among all rules with E[1{u(X) ≥
t}] = β. If τ(t) = Pr(u(X) ≥t | X ∈Xhard), then the expected query complexity to obtain
T true hits is Q = T/(ατ(t)).
B.8
Ratings with subgaussian log likelihood ratios
If on Xhard the per hit log likelihood ratio ℓis σ2-subgaussian with mean m > 0 under
w = + and −m under w = −, then DKL = m2/(2σ2) per hit. The chain rule gives
DKL(P+∥P−) ≤n α m2
2σ2 ,
∆n ≥γ
4 exp
 
−n α m2
2σ2
 
.
B.9
Shift sensitivity under mixture perturbations
Let µρ = (1 −αρ)µeasy + αρµhard with αρ = (1 + ρ)α and fixed conditionals. For any policy
π,
d
dρVw,ρ(π) = α
 
Eµhard[rw(X, π(X))] −Eµeasy[rw(X, π(X))]
 
.
If the bracket is negative, Vw,ρ(π) is strictly decreasing in ρ.
B.10
Gap decomposition under tilting
Let ∆PV(λ, π) = bV dist(λ) −V dist
w
(λ). Then
d
dλ∆PV(λ, π) = Covqλ
 br(X, π(X)) −rw(X, π(X)), bs(X)
 
.
If the covariance is positive on an interval, ∆PV is strictly increasing there.
B.11
Abstention and escalation
Augment the action set with an abstain action that escalates to a human label at cost
c ∈(0, 1).
On Xhard, choose abstain whenever the posterior over w is within a band
[ 1
2 −δ, 1
2 + δ]. A standard sequential test with a continuation region determined by δ yields
an expected number of hits of order ε−2 log(1/δ) before a definitive decision, which trades
off the expected decision loss and escalation cost. This procedure reduces the effective α
seen by the learner by routing ambiguous contexts to escalation.
13


<!-- page 14 -->
C
Appendix C: KL-Tilting Formalism
C.1
Setup and notation
Let (X, F, µ) be a base probability space and let s : X →R be a measurable score with
Eµ[exp(λs(X))] < ∞for λ in an open interval Λ ⊂R. Define the log-partition function
A(λ) = log Eµ
 
exp(λs(X))
 
,
and the exponentially tilted law qλ by
dqλ
dµ (x) = exp
 λs(x) −A(λ)
 
,
λ ∈Λ.
All expectations, variances, and covariances indexed by qλ are taken with respect to this
law. In the main text s is instantiated by the learned proxy score bs.
C.2
Basic identities
For any integrable f,
d
dλ Eqλ[f(X)] = Covqλ
 f(X), s(X)
 
.
(2)
In particular,
A′(λ) = Eqλ[s(X)],
A′′(λ) = Varqλ(s(X)) ≥0.
The Kullback–Leibler divergences between qλ and µ admit closed forms:
DKL(qλ∥µ) =
Z
log
 dqλ
dµ
 
dqλ = λA′(λ) −A(λ),
DKL(µ∥qλ) =
Z
log
  dµ
dqλ
 
dµ = A(λ) −λ Eµ[s(X)].
Both are convex and nondecreasing in |λ| whenever s is nonconstant. Moreover,
d
dλDKL(qλ∥µ) = λ Varqλ(s(X)).
C.3
Small parameter expansions
Assume A is C2 near λ = 0. Then
A(λ) = λ Eµ[s(X)] + 1
2λ2 Varµ(s(X)) + o(λ2).
For any indicator H = 1{X ∈Xhard} with α = Eµ[H],
ρ(λ) := qλ(Xhard) = α + λ Covµ(H, s) + o(λ).
(3)
Hence if Covµ(H, s) > 0 then ρ′(λ) > 0 for sufficiently small positive λ.
C.4
Bounds for mass transport
From (2) with f = H,
d
dλρ(λ) = Covqλ(H, s).
By Cauchy–Schwarz,
  ρ′(λ)
   ≤
q
Varqλ(H)
q
Varqλ(s) =
q
ρ(λ)
 1 −ρ(λ)
  p
A′′(λ).
Integrating,
  ρ(λ) −ρ(0)
   ≤
Z |λ|
0
q
ρ(t)
 1 −ρ(t)
  p
A′′(±t) dt.
This relates the growth of hard-set mass under tilting to the curvature A′′.
14


<!-- page 15 -->
C.5
I-projection interpretation
Fix m ∈{Eqλ[s] : λ ∈Λ}. Among all probability measures Q absolutely continuous with
respect to µ that satisfy EQ[s] = m,
qλ⋆= arg
min
Q: EQ[s]=m DKL(Q∥µ),
where λ⋆is the unique parameter with Eqλ⋆[s] = m.
Thus exponential tilting is the
minimum-information way to enforce a moment constraint on s.
C.6
Vector tilting
Let s = (s1, . . . , sk) and θ ∈Rk. Define
A(θ) = log Eµ

exp
 
k
X
j=1
θjsj(X)
 

,
dqθ
dµ (x) = exp
 ⟨θ, s(x)⟩−A(θ)
 
.
Then ∇A(θ) = Eqθ[s(X)] and ∇2A(θ) = Covqθ(s(X), s(X)) is the k × k covariance matrix.
For any f,
∇θ Eqθ[f(X)] = Covqθ
 f(X), s(X)
 
.
These formulas transfer the one-dimensional results to multi-signal shaping.
C.7
Interaction with policy randomization
If a stochastic policy depends on s through a logit λs(x), then for any bounded reward r,
d
dλ Eµ
 
r(X, πλ(X))
 
= Covµ
 r(X, πλ(X)), s(X)
 
.
This mirrors (2) and allows the same covariance-based reasoning for the effect of tuning λ
on performance and on the allocation of probability mass.
D
Appendix D: MAPS Interventions
D.1
Definition
Mitigation via alignment proxy shaping (MAPS) modifies the statistic used for tilting. Let
s0 be a baseline proxy score. MAPS constructs a shaped score
t(x) = w0 s0(x) +
m
X
i=1
wi si(x) −
ℓ
X
j=1
βj gj(x),
where si are auxiliary proxies and gj are penalty signals, for example predictors of misspec-
ification. The policy or data selection then tilts by t with parameter λ:
dq t
λ
dµ (x) = exp
 λ t(x) −At(λ)
 
,
At(λ) = log Eµ
 
exp(λt(X))
 
.
D.2
First-order effect on hard-set mass
Let H = 1{X ∈Xhard} and ρt(λ) = q t
λ(Xhard). Then
d
dλρt(λ) = Covq t
λ
 H, t(X)
 
,
d
dλ log ρt(λ) =
Covq t
λ(H, t)
ρt(λ)
.
In particular,
ρ′
t(0) = Covµ(H, t) = w0 Covµ(H, s0) +
X
i
wi Covµ(H, si) −
X
j
βj Covµ(H, gj).
Designing MAPS to reduce ρ′
t(0) amounts to reducing this covariance.
15


<!-- page 16 -->
D.3
Penalty by a noisy flagger
Let g(x) = bh(x) ∈{0, 1} be a flagger with true positive rate τ = Pr(bh = 1 | H = 1) and
false positive rate ϕ = Pr(bh = 1 | H = 0). Then
Covµ(H, bh) = α(1 −α) (τ −ϕ).
Consider t = g −β bh. At λ = 0,
ρ′
t(0) = Covµ(H, g) −β α(1 −α) (τ −ϕ).
Setting
β⋆=
Covµ(H, g)
α(1 −α) (τ −ϕ)
cancels the first-order drift at λ = 0. For λ ̸= 0 the exact cancellation does not persist, but
the slope reduction holds locally. When τ = ϕ the flagger carries no information and no
choice of β can reduce ρ′
t(0).
D.4
Averaging multiple proxies under a reward constraint
Let {si}m
i=0 be centered under µ. Define the covariance inner product ⟨f, g⟩= Covµ(f, g).
Let r∆be a centered surrogate for the reward difference that the designer wishes to preserve.
Solve
min
w∈Rm+1 ⟨H, P
i wisi⟩
subject to
⟨r∆, P
i wisi⟩≥τ0,
∥w∥2 ≤R.
The solution lies in the span of {H, r∆}. In particular, the choice
X
i
wisi = r∆−βH,
β = ⟨H, r∆⟩
⟨H, H⟩
makes ⟨H, P
i wisi⟩= 0 and preserves the component of r∆orthogonal to H. This cancels
ρ′
t(0) while keeping the reward-correlated direction.
D.5
Temperature control
Let t be fixed and consider the tilt parameter λ. The divergence to the base law satisfies
DKL(q t
λ∥µ) = λ A′
t(λ) −At(λ) ≈
1
2λ2 Varµ(t(X))
for small λ.
Reducing |λ| reduces both the change of measure and, via ρ′
t(λ) = Covq t
λ(H, t), the rate at
which mass moves into Xhard. Temperature control cannot guarantee ρ′
t(λ) ≤0 without
information correlated with H.
D.6
Limits without a correlated penalty
Let t be any measurable statistic built from proxies that are mean independent of H under µ,
so Covµ(H, t) = 0. Then by (3) the first-order drift vanishes at λ = 0, but nothing enforces
Covq t
λ(H, t) = 0 for λ ̸= 0. In contrast, if Covµ(H, t) > 0 and t is fixed, then ρ′
t(λ) > 0 for all
sufficiently small positive λ. Therefore, reducing drift uniformly over a range of λ requires
a penalty with positive covariance with H under the relevant laws. Perfect cancellation for
all λ requires the oracle H itself.
D.7
Sequential allocation with MAPS
Combine MAPS shaping t with sequential testing on flagged contexts. Let bh be a flagger
and let the learner retain contexts with bh(x) = 1. If τ and ϕ are the flagger rates, the
expected number of draws per true hit is 1/(ατ). With a sequential probability ratio test on
hits targeting error δ, the expected number of hits is O(ε−2 log(1/δ)), hence the expected
query count is O((ατ)−1ε−2 log(1/δ)). This preserves the 1/(ατ) factor and quantifies the
gain from improving τ through better penalties.
16


<!-- page 17 -->
D.8
Design summary
The following rules follow from the identities above.
• To reduce drift at small λ, enforce Covµ(H, t) ≤0. This can be achieved by sub-
tracting a penalty proportional to a flagger bh correlated with H.
• To preserve reward correlation, project t onto the subspace orthogonal to H while
keeping the component aligned with a reward surrogate.
• Temperature scaling reduces the magnitude of drift but does not change its sign
without a correlated penalty.
17