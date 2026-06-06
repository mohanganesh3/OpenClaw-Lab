<!-- page 1 -->
Low Rank for Rank: Uncertainty-Aware Task-Specific
LLM Ranking under Sparse Pairwise Comparisons
Jiachun Li
Laboratory for Information and Decision Systems, MIT, jiach334@mit.edu
David Simchi-Levi
Laboratory for Information and Decision Systems, MIT, dslevi@mit.edu
Will Wei Sun
Daniels School of Business, Purdue University, sun244@purdue.edu
Abstract. Pairwise human-preference platforms such as Chatbot Arena have become central to large lan-
guage model evaluation, yet reliable task-specific ranking remains challenging. Global leaderboards can mask
substantial task heterogeneity, while independently estimating rankings for each fine-grained task is statisti-
cally unstable under sparse and imbalanced comparisons. We propose a low-rank framework for task-specific
LLM ranking from sparse pairwise comparisons, modeling the task-by-model ability matrix as low rank
so that information can be shared across related tasks while preserving task-specific differences. We first
develop a max-norm accurate estimator for the latent score matrix, which yields task-wise top-K recovery
guarantees under sparse sampling. Our main contribution is an uncertainty quantification framework for
task-specific ranking. Beyond point estimation, we construct debiased one-step estimators for fixed score con-
trasts, such as task-specific gaps between two models, yielding asymptotically valid confidence intervals that
attain the semiparametric efficiency lower bound. We further extend this inference to the high-dimensional
ranking regime, where ranks and top-K membership are determined by many dependent score-gap hypothe-
ses. Using Gaussian and multiplier-bootstrap calibration, we obtain simultaneous confidence sets for per-task
ranks and valid tests for top-K membership across many tasks and models. Experiments on synthetic data
and Arena comparisons show that low-rank sharing improves sample efficiency over independent task-wise
Bradley–Terry estimation and enables uncertainty-aware task-specific certification.
1.
Introduction
Pairwise human-preference evaluation has become a central tool for comparing large language mod-
els (LLMs). Platforms such as Chatbot Arena [Chiang et al., 2024] collect side-by-side comparisons
of model responses and aggregate them into public leaderboards, providing a scalable alternative
to fixed benchmark scores. At the same time, modern LLM evaluation is increasingly task-specific,
1
arXiv:2605.29395v1  [stat.ME]  28 May 2026


<!-- page 2 -->
with benchmarks and evaluation platforms reporting performance separately for coding, mathe-
matical reasoning, instruction following, multilingual tasks, creative writing, and other fine-grained
categories [White et al., 2025, Frick et al., 2025, Moslem and Kelleher, 2026]. This task-specific
view is essential because model strengths are heterogeneous: a model that is strong overall may
not be the best choice for a particular task, user group, or deployment domain.
The central statistical problem is therefore not only to estimate model abilities, but to make
reliable task-specific ranking decisions. Practitioners often want to know which models can be
trusted as top performers for a task, whether an apparent difference between two models is statisti-
cally significant, and which leaderboard claims remain valid after many comparisons are considered
simultaneously. A point leaderboard alone cannot answer these questions. When comparisons are
sparse or imbalanced, especially within fine-grained task categories, small estimated score gaps near
the top-K boundary may simply reflect sampling noise. Thus task-specific LLM evaluation requires
uncertainty-aware ranking: confidence statements for ranks, top-K membership, and task-specific
model comparisons.
Existing approaches leave a gap. Independent task-wise Bradley-Terry-Luce (BTL) estimation
[Bradley and Terry, 1952, Luce, 1959] respects task heterogeneity, but can be statistically unstable
when each task receives limited or uneven comparisons. Fully pooling data across tasks reduces
variance, but erases the task-specific heterogeneity that motivates fine-grained evaluation in the
first place. Recent low-rank approaches provide an attractive compromise by sharing information
across related tasks, but smooth score estimation or inference for fixed functionals does not by itself
solve the ranking problem. Ranks and top-K membership are nonsmooth functionals determined
by many dependent score-gap signs. Valid task-specific leaderboards therefore require new tools for
boundary-sensitive top-K recovery, multiple testing over correlated score gaps, and simultaneous
rank certification.
We develop a statistical framework for certified task-specific leaderboards from sparse pairwise
comparisons. Let Θ⋆∈Rdt×dm denote the latent task-by-model ability matrix, where Θ⋆
t,m is the
score of model m on task t. Each observation consists of a task t, two models m,m′, and a binary
preference outcome following a BTL-type model that depends on the score difference Θ⋆
t,m −Θ⋆
t,m′.
We assume that Θ⋆is approximately low rank, reflecting shared latent capabilities, such as reason-
ing, instruction following, or style sensitivity, across tasks. This setting is more challenging than
standard low-rank matrix completion [Cand`es and Recht, 2009, Negahban and Wainwright, 2012,
Davenport et al., 2014], because each observation is a binary, within-task comparison depending
only on a score difference, rather than a noisy observation of an individual matrix entry. More
importantly, the main target is not merely low-rank score estimation, but statistically valid ranking
and certification under sparse, non-uniform, and dependent pairwise comparisons.
2


<!-- page 3 -->
Moving from score estimation to ranking certification introduces additional difficulties. First,
top-K accuracy requires entrywise control of task-model scores, rather than only Frobenius or
prediction-error accuracy. Second, rank and top-K decisions are determined by the signs of many
score gaps, so uncertainty quantification naturally becomes a multiple hypothesis testing problem.
Third, these score-gap statistics are strongly dependent: they share tasks, models, comparisons,
and low-rank latent factors. As a result, pointwise confidence intervals or independent testing
corrections are not sufficient for valid leaderboard-level guarantees.
Sparse task-specific
pairwise comparisons
{ti,mi,m′
i,Yi}n
i=1
Shared low-rank
ability matrix
Θ⋆∈Rdt×dm
Entrywise error control
∥bΘ −Θ⋆∥∞
Multiple testing for
score-gap hypotheses
Sec. 4
Task-wise top-K
accuracy
Sec. 3
Simultaneous rank and
top-K certification
Sec. 5
Figure 1
Paper outline: from sparse pairwise comparisons to certified task-specific leaderboards.
Figure 1 summarizes how we turn low-rank preference estimates into certified task-specific leader-
board decisions. Our contributions can be summarized as follows.
• Task-wise top-K accuracy from sparse comparisons. We derive max-norm estimation
guarantees tailored to task-specific ranking and show how they imply top-K Hamming accu-
racy and exact recovery. The resulting theory characterizes task-specific ranking error through
the number of models near the K-boundary, extending single-task top-K recovery principles
to many coupled task-specific rankings under shared low-rank structure.
• Efficient score-gap inference for ranking hypotheses. We construct debiased one-step
estimators for task-specific score gaps, which are the basic pairwise comparisons underlying
rank and top-K decisions. The construction uses the low-rank tangent space and the restricted
Fisher information operator to obtain semiparametrically efficient inference, together with a
joint covariance characterization for correlated ranking hypotheses.
• Multiple testing and simultaneous rank certification. We convert score-gap inference
into valid ranking by calibrating the maximum of studentized debiased gap statistics using
Gaussian and multiplier-bootstrap approximation. This yields simultaneous confidence sets
for task-specific ranks and valid top-K membership tests across many tasks and models,
separating models into certified top-K, non-top-K, and statistically unresolved categories.
• Empirical validation on task-specific leaderboards. Synthetic and Arena experiments
show that low-rank sharing improves top-K recovery and produces shorter, better-calibrated
rank and membership certificates than independent task-wise BTL estimation.
3

[CAPTION] Figure 1
Paper outline: from sparse pairwise comparisons to certified task-specific leaderboards.

[CAPTION] Figure 1 summarizes how we turn low-rank preference estimates into certified task-specific leader-


<!-- page 4 -->
1.1.
Related work and positioning.
Arena-style LLM evaluation. Arena-style platforms have made pairwise human-preference com-
parison a standard tool for evaluating LLMs [Chiang et al., 2024, Arena Team, 2026a,b]. Modern
leaderboards increasingly report category-specific results for coding, math, creative writing, and
instruction following, and hard prompts [Arena Team, 2026b], reflecting substantial task-level het-
erogeneity. Recent work further studies prompt-dependent leaderboards and routing: Frick et al.
[2025] learn prompt-dependent Bradley-Terry coefficients, while Avelar Menendez et al. [2026]
provide uncertainty quantification for prompt-dependent rankings. These works motivate fine-
grained and uncertainty-aware evaluation, but independent category- or prompt-wise ranking does
not share information across related tasks and can be inefficient under sparse comparisons. Our
framework uses low-rank structure to share information across tasks while preserving task-specific
rankings and providing simultaneous uncertainty statements for ranks and top-K membership.
Pairwise ranking, uncertainty quantification, and low-rank inference. Our theory builds
on pairwise ranking and top-K recovery under BTL-type models [Bradley and Terry, 1952, Luce,
1959, Hunter, 2004, Chen and Suh, 2015, Chen et al., 2022], as well as recent uncertainty quan-
tification for sparse, covariate-assisted, and heterogeneous ranking models [Fan et al., 2024, 2025,
2026]. Related semiparametric ideas have also begun to appear in LLM evaluation, for example
through the use of auxiliary comparison signals to improve efficiency [Dong et al., 2026]. Closest to
our work, Li et al. [2026] study low-rank LLM evaluation and semiparametric efficiency for smooth
functionals of a latent score tensor. In contrast, our paper focuses on task-specific ranks and top-K
membership, which are nonsmooth ranking functionals determined by many dependent score-gap
signs. Thus the main new ingredients are ranking-specific: boundary-based top-K recovery, multi-
ple testing for correlated score-gap hypotheses, and simultaneous rank/top-K certification. These
ingredients allow us to move from efficient score estimation to certified task-specific leaderboards.
2.
Problem Setup
We formalize task-specific LLM evaluation as a sparse pairwise-comparison problem. There are
dt task categories and dm candidate models. For each task-model pair, let Θ⋆
t,m denote the latent
ability score of model m on task t, and collect these scores in the matrix Θ⋆∈Rdt×dm. The row index
t ∈[dt] represents a task category and the column index m ∈[dm] represents an LLM. We assume
that Θ⋆is approximately low rank, with rank r ≪min{dt,dm} and singular value decomposition
Θ⋆= U ⋆Σ⋆(V ⋆)⊤. Let σ⋆
1 ≥··· ≥σ⋆
r > 0 denote the nonzero singular values collected in Σ⋆, and
define the condition number κ := σ⋆
1/σ⋆
r. This low-rank structure allows information to be shared
across related tasks while preserving task-specific model rankings. Let et ∈Rdt and em ∈Rdm denote
4


<!-- page 5 -->
the standard basis vectors for task t and model m, respectively. We assume the singular vectors
are µ-incoherent:
max
t∈[dt]∥e⊤
t U ⋆∥2
2 ≤µr
dt
,
max
m∈[dm]∥e⊤
mV ⋆∥2
2 ≤µr
dm
.
Observation model. For i = 1,...,n, a task ti ∈[dt] is sampled from a distribution ν. Con-
ditional on ti, a pair of distinct models (mi,m′
i) is sampled from a task-dependent distribution
πti. We let Yi = 1 indicate that model mi is preferred to model m′
i. Define the signed comparison
design matrix Xi := eti(emi −em′
i)⊤. It has a single nonzero row, corresponding to task ti, with
+1 in column mi and −1 in column m′
i. Hence ⟨Xi,Θ⋆⟩= Θ⋆
ti,mi −Θ⋆
ti,m′
i. Conditional on Xi, the
preference follows a BTL model [Bradley and Terry, 1952, Luce, 1959]
Pr(Yi = 1 | Xi) = σ(⟨Xi,Θ⋆⟩) = σ
 
Θ⋆
ti,mi −Θ⋆
ti,m′
i
 
,
σ(x) = (1 + e−x)−1.
Because pairwise comparisons depend only on score differences, the matrix Θ⋆is identifiable only
up to task-specific additive shifts. We fix a representative by imposing the row-centering constraint
Θ⋆1dm = 0.
Sampling design. We allow the task and model-pair sampling distributions to be non-uniform,
reflecting the uneven traffic patterns of real evaluation platforms. To obtain clean theoretical rates,
we assume this imbalance is controlled: there exist constants 0 < cν ≤Cν < ∞and 0 < cπ ≤Cπ < ∞,
independent of dt,dm,n, such that for all t ∈[dt] and unordered pairs {m,m′} ⊂[dm],
cν
dt
≤νt ≤Cν
dt
,
cπ
 dm
2
  ≤πt({m,m′}) ≤Cπ
 dm
2
 .
Together with the bounded-signal condition ∥Θ⋆∥∞≤B, the above sampling assumptions ensure
that every task and every model pair receives comparable statistical information. Indeed, writing
η = Θ⋆
t,m −Θ⋆
t,m′ for a generic within-task score difference, the bounded-signal condition implies
|η| ≤2B. Hence the BTL Fisher information I(η) := σ(η){1−σ(η)} is bounded away from zero and
infinity, as in standard bounded dynamic-range assumptions for BTL-type ranking models [Chen
and Suh, 2015, Chen et al., 2022, Fan et al., 2024, 2025]. The lower bounds on νt and πt({m,m′})
prevent any task or pairwise comparison direction from being asymptotically unobserved, in the
same spirit as standard sampling conditions in low-rank matrix completion and one-bit matrix
estimation [Cand`es and Recht, 2009, Negahban and Wainwright, 2012, Davenport et al., 2014].
Ranking targets. Built on the entrywise error bound for the latent score matrix, Section 3
studies task-wise top-K accuracy for S⋆
K(t) := {m : rkt(m) ≤K}, where rkt(m) is model m’s rank
on task t. Section 4 develops efficient inference and multiple testing tools for task-specific score
gaps Θ⋆
t,m −Θ⋆
t,m′, and Section 5 converts these gap inferences into simultaneous rank confidence
sets and top-K membership certificates.
5


<!-- page 6 -->
3.
Entrywise Estimation and Task-Wise Top-K Accuracy
This section establishes the first step toward certified task-specific leaderboards: task-wise top-K
accuracy from sparse pairwise comparisons.
Low-rank score estimator. Let Xi = eti(emi −em′
i)⊤and ℓ(y,η) = log(1 + exp(η)) −yη. We
compute a nuclear-norm penalized BTL initializer
bΘ0 ∈arg minΘ∈CB
n
1
|I0|
P
i∈I0 ℓ(Yi,⟨Xi,Θ⟩) + λ∥Θ∥∗
o
,
CB = {Θ : Θ1dm = 0, ∥Θ∥∞≤B}.
This convex program provides a Frobenius-accurate initializer. In particular, Theorem B.5 in
Appendix shows that, with high probability, ∥bΘ0 −Θ⋆∥F ≲
p
r ¯d3 polylog( ¯d)/n0, where ¯d :=
max{dt,dm} and n0 = |I0|. Such a global error guarantee is sufficient for initialization, but it is
not enough for ranking: top-K recovery depends on individual score gaps, especially near the K-
versus-(K + 1) boundary, and therefore requires entrywise control of the score matrix. To obtain
this stronger guarantee, starting from the rank-r SVD of bΘ0, we apply a row-wise pairwise-logistic
refinement,
bΘ = Refiner(bΘ0).
At a high level, the refinement first constructs an estimated right factor from the initializer, then
updates each task-side latent vector by solving a pairwise-logistic score equation conditional on this
right factor. After re-centering to enforce the row-sum gauge, it performs an analogous model-side
update and returns the resulting rank-r, row-centered matrix. Thus the refinement converts the
global Frobenius initializer into an entrywise-accurate estimator tailored to task-specific ranking.
The full refinement steps and proof are deferred to Appendix B.
Theorem 3.1 (Uniform entrywise estimation). Under the model and near-uniform sam-
pling assumptions in Section 2, suppose Θ⋆has rank r, is µ-incoherent, satisfies ∥Θ⋆∥∞≤B,
and has condition number κ. Denote ¯d := max{dt,dm}. If the pairwise comparison sample size
n ≳poly(µ,r,κ,B) ¯d logc(n ¯d), then, for some large constant a, with probability at least 1 −n−a,
∥bΘ −Θ⋆∥∞≤εn,
with εn := C poly(a,µ,r,κ,B)
q
¯d logc(n ¯d)
n
.
Theorem 3.1 is the main estimation result. The nuclear-norm optimization provides a Frobenius-
accurate initializer, while the refinement step upgrades this global accuracy to uniform max-norm
control over all task-model scores. For constant rank and condition number, the required sample
size is near-linear in ¯d, so the method learns all task-specific scores jointly rather than fitting dt
unrelated BTL models. In the balanced regime dt ≍dm ≍d, this is ∥bΘ−Θ⋆∥∞≲
p
dpolylog(nd)/n.
Task-wise top-K accuracy. We next translate the entrywise error bound into a task-specific
top-K recovery guarantee. Denote the estimated top-K set for task t as c
S⋆
K(t) = {m : brkt(m) ≤K},
6


<!-- page 7 -->
where the rank brkt(m) is computed from the refined estimate bΘt,m. To measure the discrepancy
between the estimated and true top-K sets, define the normalized Hamming error
HamK,t := 1
2K | c
S⋆
K(t)△S⋆
K(t)|.
Here △denotes symmetric difference: it counts models that are included in one top-K set but not
the other. The normalization by 2K makes HamK,t ∈[0,1].
The difficulty of top-K recovery depends on how many models have scores close to the top-K
cutoff. Let Θ⋆
t,(1) ≥··· ≥Θ⋆
t,(dm) be the sorted true scores for task t, and define the midpoint between
the K-th and (K + 1)-st scores as τK(t) := (Θ⋆
t,(K) + Θ⋆
t,(K+1))/2. We call τK(t) the top-K decision
boundary. For a resolution level δ > 0, define the boundary profile
RK,t(δ;Θ⋆) := 1
2K
  {m : |Θ⋆
t,m −τK(t)| ≤δ}
  .
This quantity measures the fraction of models lying within distance δ of the top-K boundary. If
many models lie near τK(t), then the task is intrinsically hard to rank because small estimation
errors can swap models across the top-K cutoff. If few models lie near the boundary, top-K recovery
is easier.
Proposition 3.2 (Task-wise top-K Hamming accuracy). On the event ∥bΘ −Θ⋆∥∞≤εn,
for every t ∈[dt], HamK,t ≤RK,t(2εn;Θ⋆). Therefore, under Theorem 3.1, the above Hamming
bound holds simultaneously for all tasks with probability at least 1 −n−a.
Proposition 3.2 shows that top-K mistakes can only occur for models whose true scores lie within
the statistical resolution 2εn of the top-K boundary. This extends the boundary-resolution principle
from single-task BTL top-K ranking [Chen and Suh, 2015, Chen et al., 2022] to many task-specific
rankings coupled through shared low-rank structure. Consequently, exact recovery follows under a
task-specific margin condition. Define the K-gap ∆K(t) := Θ⋆
t,(K) −Θ⋆
t,(K+1). If ∆K(t) > 4εn, then
no model can cross the top-K boundary, so c
S⋆
K(t) = S⋆
K(t). Thus, with high probability, exact
top-K recovery holds simultaneously for every task whose K-gap exceeds 4εn. Hence, low-rank
sharing enables simultaneous task-wise top-K recovery at the same entrywise resolution as score
estimation, without requiring each task to be estimated independently.
4.
Score-Gap Inference and Multiple Testing Foundations
Task-specific ranking decisions are built from score-gap signs. For example, deciding whether model
m outranks model m′ on task t requires inference on Θ⋆
t,m −Θ⋆
t,m′. Similarly, ranks and top-K
membership are determined by many such pairwise gaps. We therefore first develop efficient infer-
ence for score gaps and characterize their joint dependence, which provides the multiple-testing
foundations used later for simultaneous ranking inference.
7


<!-- page 8 -->
Score-gap contrasts. We write a generic linear contrast as ψΓ(Θ⋆) = ⟨Γ, Θ⋆⟩. The canonical
example is Γ = et(em −em′)⊤, for which ψΓ(Θ⋆) = Θ⋆
t,m −Θ⋆
t,m′. Testing whether model m is better
than model m′ on task t is therefore a test on the sign of ψΓ(Θ⋆). Multiple ranking claims correspond
to testing many such contrasts jointly.
Efficient one-step estimator. In the low-rank model, local perturbations of Θ⋆must lie in
the tangent space T of the rank-r, row-centered manifold. Therefore only the projected contrast
direction PTΓ is locally identifiable. Define the Fisher operator G by
⟨GH1, H2⟩= E[I(⟨X, Θ⋆⟩)⟨H1, X⟩⟨H2, X⟩],
I(η) = σ(η){1 −σ(η)}.
It measures how much information the pairwise-comparison design carries about directions H1,H2.
Closely matched comparisons have larger I(η), while lopsided comparisons carry less information.
Define the restricted Fisher information operator A, the efficient direction H⋆
Γ, and the corre-
sponding efficient variance Veff(Γ) by
A := PTGPT,
H⋆
Γ := A−1PTΓ,
Veff(Γ) :=

PTΓ, A−1PTΓ
 
.
The operator A describes how much information the observed pairwise comparisons carry about
locally admissible low-rank perturbations of Θ⋆. The efficient direction H⋆
Γ is the optimal weight-
ing direction for converting comparison residuals into an estimate of the target contrast ψΓ. The
variance Veff(Γ) is the resulting semiparametric efficiency bound: it is the smallest achievable asymp-
totic variance for regular estimators of ψΓ, accounting for the sampling design, the BTL Fisher
information, and the low-rank constraint.
Given an entrywise-accurate estimator bΘ from Section 3, we estimate PT, G, and H⋆
Γ by bPT, bG,
and bHΓ, where ( bPT bG bPT) bHΓ = bPTΓ. This leads to the final one-step efficient estimator
bψΓ =
D
Γ, bΘ
E
+ 1
n
Pn
i=1 s(Yi, bηi)
D
bHΓ, Xi
E
, with bηi =
D
Xi, bΘ
E
and s(y,η) = y −σ(η).
The second term of bψΓ debiases the plug-in estimator
D
Γ, bΘ
E
and yields efficient score-gap inference.
Joint inference for multiple score gaps. For ranking, we need joint inference for many
score gaps, not just one. For a fixed collection Γ1,...,Γq, let ψj = ψΓj(Θ⋆). The efficient covariance
between the corresponding one-step estimators is
Σjk =

PTΓj, A−1PTΓk
 
.
This covariance is generally non-diagonal because two score gaps may share a task, a model,
observed comparisons, or low-rank latent factors. Capturing this dependence is essential for mul-
tiple testing: treating correlated score-gap tests as independent can miscalibrate leaderboard-level
uncertainty.
8


<!-- page 9 -->
Theorem 4.1 (Efficient joint score-gap inference). Under
the
assumptions
of
Theo-
rem 3.1 and the regularity conditions in Appendix D, there exists ZΓ ∼N(0,Σ) such that
supB∈Rq
   Pr
n√n( bψ1 −ψ1,..., bψq −ψq) ∈B
o
−Pr{ZΓ ∈B}
    ≲CA
q
¯d logc(n ¯d)
n
.
Here Rq is the class of rectangles in Rq, and CA is the inverse-information stability factor control-
ling the ℓ∞→ℓ∞size of A−1. Consequently, if CA
p ¯d logc(n ¯d)/n →0, then
√n( bψ1 −ψ1,..., bψq −ψq) ⇝N(0,Σ).
Moreover, the following empirical influence-function covariance is consistent for Σjk
bΣjk = n−1 X
i
bϕj(Wi)bϕk(Wi),
bϕj(Wi) = s(Yi, bηi)
D
bHΓj, Xi
E
.
Theorem 4.1 yields confidence intervals and joint tests for fixed collections of score-gap hypothe-
ses. For a single gap, with c
SEΓ = (bVΓ/n)1/2, a pointwise (1−α)-confidence interval is bψΓ±z1−α/2 c
SEΓ.
For multiple gaps, bΣ captures their dependence, which is crucial for valid joint testing of over-
lapping ranking claims. The covariance Σ is also the semiparametric efficiency lower bound: any
regular asymptotically Gaussian estimator has limiting covariance no smaller than Σ in the positive-
semidefinite order, and the proposed one-step estimator attains this bound.
For ranking, however, the relevant family of score gaps grows with the number of models and
tasks. Thus fixed-dimensional inference must be strengthened to a uniform expansion over growing
collections of score-gap contrasts.
Uniform expansion for ranking tests. For growing collections of score-gap contrasts, our
analysis gives the uniform expansion
√n( bψj −ψj) =
1
√n
Pn
i=1 ϕj(Wi) + rj,
maxj |rj| ≲CA
q
¯d logc(n ¯d)
n
,
with high probability. This expansion reduces simultaneous testing of correlated score gaps to a
high-dimensional Gaussian approximation problem. Together with Theorem 4.1, it provides the
technical foundation for the rank and top-K certification procedure developed next.
5.
Simultaneous Rank and Top-K Certification
We now convert score-gap inference into leaderboard-level certification. Fix a task t and a model
m. The rank of m on task t is determined by the signs of the dm −1 score gaps ∆(m)
t,ℓ:= Θ⋆
t,ℓ−Θ⋆
t,m =
ψΓ(m)
t,ℓ(Θ⋆), ℓ̸= m, with Γ(m)
t,ℓ:= et(eℓ−em)⊤, since rkt(m) = 1 + P
ℓ̸=m 1{∆(m)
t,ℓ> 0}. Thus rank
inference is a multiple-testing problem over the competitors of m: each positive gap corresponds
to one model ranked above m. The goal is not only to test one gap at a time, but to produce a
confidence set for the rank and a certified top-K membership decision.
9


<!-- page 10 -->
Simultaneous score-gap bands. For each ℓ̸= m, denote the one-step estimator from Section 4
as b∆(m)
t,ℓ:= bψΓ(m)
t,ℓ, with estimated influence-function summand and standard error
bϕ(m)
t,ℓ(Wi) := s(Yi, bηi)
 
bHΓ(m)
t,ℓ, Xi
 
,
bσ(m)
t,ℓ:=
n
1
n
Pn
i=1
 bϕ(m)
t,ℓ(Wi)
 2o1/2
.
A pointwise interval for each gap is not enough: the rank depends on all dm −1 gap signs jointly,
and these gap estimators are correlated through shared tasks, models, comparisons, and low-rank
factors. We need a joint calibration that accounts for this covariance structure. Equivalently, simul-
taneous confidence bands require controlling the maximum studentized error over all competitors,
maxℓ̸=m
    
√n(b∆(m)
t,ℓ−∆(m)
t,ℓ)
bσ(m)
t,ℓ
    . Draw i.i.d. multipliers ξi ∼N(0,1) and define the multiplier-bootstrap
statistic
T ∗
t,m := maxℓ̸=m
    
1
√n
Pn
i=1 ξi
bϕ(m)
t,ℓ(Wi)
bσ(m)
t,ℓ
    .
Conditional on the data, this bootstrap process preserves the empirical dependence among the
gap statistics. Let ct,m(1 −α) be the conditional (1 −α)-quantile of T ∗
t,m. We form simultaneous
confidence bands
bI(m)
t,ℓ:=
 
b∆(m)
t,ℓ± ct,m(1 −α)
bσ(m)
t,ℓ
√n
 
= [bL(m)
t,ℓ, bU (m)
t,ℓ].
The critical value is the quantile of the maximum of many correlated studentized errors, not
the usual pointwise Gaussian quantile. Under the uniform expansion from Section 4, standard
high-dimensional Gaussian approximation and multiplier bootstrap theory justify this calibration
[Chernozhukov et al., 2013, 2017].
Rank confidence band. On the simultaneous coverage event, the score-gap bands can be
inverted into a rank confidence band. Define At(m) := #{ℓ̸= m : bL(m)
t,ℓ> 0},
Bt(m) := #{ℓ̸=
m : bU (m)
t,ℓ
< 0}. Here At(m) is the number of competitors certified to be above m, and Bt(m) is
the number certified to be below m. The remaining competitors have confidence bands crossing
zero and are therefore unresolved. Thus a valid confidence band for the rank is bRt(m) := [1 +
At(m), dm −Bt(m)].
Theorem 5.1 (Rank confidence band for one task). Under the assumptions of Theo-
rem 4.1, the uniform remainder condition from Section 4, and the high-dimensional Gaussian
approximation conditions in Appendix E,
Pr{rkt(m) ∈bRt(m)} ≥1 −α −o(1).
Top-K membership certification. The rank band directly yields a three-way top-K decision.
If dm −Bt(m) ≤K, then even the worst rank compatible with the confidence bands is at most
K, so we certify m ∈S⋆
K(t). If 1 + At(m) > K, then even the best compatible rank is larger than
10


<!-- page 11 -->
K, so we certify m /∈S⋆
K(t). Otherwise, the membership decision is statistically unresolved. In
LLM evaluation, this three-way output: {certified top-K, certified non-top-K, unresolved} separates
reliable leaderboard claims from comparisons that remain too noisy to certify.
Simultaneous inference across tasks. The same construction can be applied simultaneously
across tasks. For a fixed model m, replace the one-task family {ℓ: ℓ̸= m} by
J (m) := {(t,ℓ) : t ∈[dt], ℓ̸= m},
|J (m)| = dt(dm −1).
We compute the multiplier-bootstrap critical value for the maximum over J (m), and use the
resulting bands to construct bRt(m) for every task t. Since |J (m)| is polynomial in ¯d, the same
high-dimensional Gaussian approximation applies up to logarithmic factors.
Corollary 5.2 (Simultaneous task-wise rank inference). Under the conditions of Theo-
rem 5.1, with the bootstrap maximum taken over J (m),
Pr
n
rkt(m) ∈bRt(m) for all t ∈[dt]
o
≥1 −α −o(1).
Consequently, the certified top-K, certified non-top-K, and unresolved decisions for model m are
simultaneously valid across all tasks.
Appendix E.10 extends the same score-gap band inversion to the entire task-specific top-K set,
producing inner and outer confidence sets satisfying c
S⋆
Kin(t) ⊆S⋆
K(t) ⊆c
S⋆
Kout(t), simultaneously
over tasks.
6.
Experiments
We complement our theory with simulation studies and an LM Arena case study. Throughout,
the joint estimator uses the convex initializer of Section 3 followed by alternating-minimization
refinement, and inference uses the cross-fitted one-step debiased estimator combined with a Gaus-
sian multiplier bootstrap calibrated against the high-dimensional CLT of Section 5; We compare
against per-task Bradley–Terry (BTL), which fits each column independently and uses the anal-
ogous Wald-type plug-in influence functions for its multiplier bootstrap. Each experimental cell
reports a Monte Carlo summary over N = 200 trials, formed by bootstrap resampling from the runs
we executed; uncertainty intervals in tables are 95% bootstrap intervals for each metric. For both
simulation and LM Arena we report initialization accuracy, top-K recovery, the joint asymptotic
distribution of contrast estimators, and rank inference both for a single task and simultaneously
across tasks.
11


<!-- page 12 -->
6.1.
Simulation
We use a square setting with dt = dm = 50, true rank r⋆= 5, and amplitude α = 5, generating
T ⋆= ΘA⊤from i.i.d. Gaussian factors and rescaling to ∥T ⋆∥∞≤α. Pairwise comparisons are
sampled uniformly over (task, model-pair) and generated from the BTL model with temperature
τ = 1. We sweep the total number of comparisons n ∈{4,000, 8,000, 16,000, 32,000}. Cross-fitting
uses K = 6 folds.
Estimation error decay. Theorem 3.1 predicts that the joint AltMin initialization attains
bΘ −Θ⋆
F ≲1/√n and
bΘ −Θ⋆
∞≲1/√n up to polylog factors when n ≳r ¯dlog ¯d. Figure 2 plots
both errors versus n on a log–log scale and overlays a 1/√n reference. The joint estimator’s empir-
ical errors track the predicted rate, while per-task BTL — which fits each column independently
using only n/dt comparisons — has a far larger constant. The gap is largest at n = 4,000 (per-task
BTL Frobenius error exceeds 1000, almost two orders of magnitude above the joint estimator) and
shrinks as n grows.
4k
8k
16k
32k
number of comparisons n
101
102
103
T
T
F
Frobenius error
Joint AltMin
Per-task BTL
1/ n
4k
8k
16k
32k
number of comparisons n
100
101
T
T
Sup-norm error
Joint AltMin
Per-task BTL
1/ n
Figure 2
Estimation error vs. n at dt = dm = 50, r⋆= 5, α = 5. The joint estimator’s Frobenius and sup-norm
errors decay at the predicted 1/√n rate; per-task BTL, which does not pool across tasks, incurs a much larger
constant.
Top-K Recovery. Table 1 reports the per-task top-K set Hamming distance, averaged over
the dt tasks, for K ∈{5,10}. The joint estimator yields uniformly lower Hamming distance than
per-task BTL across all n, with the gap shrinking as n grows.
Joint asymptotic Gaussianity of two contrasts. Theorem 4.1 predicts that for any fixed
collection of score-gap contrasts, the cross-fitted one-step estimator ( bψ1,..., bψq) is jointly asymp-
totically Gaussian, with covariance equal to the inverse of the semiparametric information. We
verify this prediction directly with two contrasts. Pick items a,b,c ∈[dm] on a single task t ∈[dt]
and form ψ1 = Θ⋆
a,t −Θ⋆
b,t and ψ2 = Θ⋆
a,t −Θ⋆
c,t; the shared item a induces nontrivial correlation
between the two contrast estimators. Over N = 500 Monte Carlo trials at n = 16,000 we compute
12


**[Table p12.1]**
|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  |  | Joint AltM Per-task | in BTL |  |  |  |
|  |  | 1/ n |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |


**[Table p12.2]**
|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  | Joint AltMin Per-task BT | L |  |
|  |  |  |  | 1/ n |  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |

[CAPTION] Figure 2
Estimation error vs. n at dt = dm = 50, r⋆= 5, α = 5. The joint estimator’s Frobenius and sup-norm


<!-- page 13 -->
Table 1
Simulation, mean per-task top-K Hamming distance across dt = 50 tasks, N = 200 trials. Smaller is
better; entries are mean (95% CI).
K = 5
K = 10
n
Joint
Per-task BTL
Joint
Per-task BTL
4,000
0.482 (0.478, 0.486) 0.730 (0.728, 0.733)
0.388 (0.385, 0.390)
0.596 (0.594, 0.598)
8,000
0.339 (0.335, 0.342) 0.617 (0.613, 0.620)
0.257 (0.254, 0.259)
0.489 (0.487, 0.491)
16,000 0.237 (0.234, 0.239) 0.479 (0.476, 0.482)
0.181 (0.179, 0.182)
0.366 (0.363, 0.368)
32,000 0.167 (0.164, 0.169) 0.360 (0.357, 0.362)
0.129 (0.127, 0.130)
0.269 (0.267, 0.271)
( bψ1, bψ2) and the per-trial plug-in covariance bΣ. The empirical covariance Σemp of ( bψ1, bψ2) across
trials and the mean-of-plug-in Σthy are reported in Table 2. The two are close in entrywise scale
and correlation, and the 95% confidence ellipse derived from Σthy achieves nominal coverage on
the empirical samples (0.950 vs. target 0.95). Figure 3 shows the centered Monte Carlo samples
together with the theoretical 95% Gaussian ellipse and the standardized marginals overlaid on
N(0,1), both consistent with bivariate Gaussianity.
Table 2
Empirical vs. theoretical (mean plug-in) covariance of the two-contrast cross-fitted one-step estimators
across N = 500 trials at dt = dm = 50, r⋆= 5, α = 5, n = 16,000.
Σemp
Σthy
bψ1
bψ2
bψ1
bψ2
bψ1
0.276
0.114
0.343
0.161
bψ2
0.114
0.356
0.161
0.385
correlation
ρemp = 0.365
ρthy = 0.443
95% ellipse cov.
0.950
(target 0.95)
2
1
0
1
1
1
1
0
1
2
2
2
Joint distribution
MC samples
95\% Gaussian (
thy)
4
2
0
2
4
1
1  (standardized)
0.0
0.1
0.2
0.3
0.4
Marginal of contrast 1
(0, 1)
4
2
0
2
4
2
2  (standardized)
0.0
0.1
0.2
0.3
0.4
Marginal of contrast 2
(0, 1)
Figure 3
Joint asymptotic Gaussianity at n = 16,000. Left: centered Monte Carlo samples ( bψ1 −ψ⋆
1, bψ2 −ψ⋆
2) and
the theoretical 95% Gaussian ellipse derived from Σthy. Middle/Right: standardized marginals overlaid on N(0,1).
Single-task rank inference. Theorem 5.1 provides a multiplier-bootstrap rank confidence set
for a single model on a single task. To exercise the procedure for an arbitrary target, we draw
one model mt uniformly at random for each task t ∈[dt] (fixed across trials) and form the 1 −α
13


**[Table p13.1]**
|  | 95\% | Gaussian (thy) |  |
| --- | --- | --- | --- |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |


**[Table p13.2]**
|  |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |


**[Table p13.3]**
|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |

[CAPTION] Table 1
Simulation, mean per-task top-K Hamming distance across dt = 50 tasks, N = 200 trials. Smaller is

[CAPTION] Table 2
Empirical vs. theoretical (mean plug-in) covariance of the two-contrast cross-fitted one-step estimators

[CAPTION] Figure 3
Joint asymptotic Gaussianity at n = 16,000. Left: centered Monte Carlo samples ( bψ1 −ψ⋆


<!-- page 14 -->
rank confidence set for mt on task t at α = 0.05, Ktop = 10, and n = 16,000. Table 3 reports the
per-task coverage of the true rank, the fraction of trials at which the rule correctly certifies in/out
of top-K, and the mean rank-CI width. The joint method preserves coverage at the nominal level
and correctly certifies ≈29% of (task, trial) instances; per-task BTL correctly certifies only ≈8%
since at this sample size each task has roughly n/dt = 320 comparisons — too sparse for per-task
inference to produce informative rank CIs.
Table 3
Simulation, single-task rank confidence set for a randomly chosen model on each task at n = 16,000,
Ktop = 10, N = 200 trials.
Joint (low-rank)
Per-task BTL
Coverage of true rank
1.000 (1.000, 1.000)
0.967 (0.964, 0.970)
Correct top-K certification
0.289 (0.284, 0.293)
0.081 (0.077, 0.086)
Mean rank-CI width
36.7 (36.6, 36.8)
44.6 (44.5, 44.8)
Simultaneous inference. We also show simultaneous inference across tasks for a fixed model.
Fixing one model m⋆chosen at random, we form a single multiplier-bootstrap critical value over
the joint family {(t,ℓ) : ℓ̸= m⋆, t ∈[dt]} at n = 32,000 and Ktop = 10, yielding simultaneous rank-
CIs for m⋆across all dt = 50 tasks. Table 4 reports the fraction of (trial, task) instances on which
the procedure produces a non-trivial decision (resolved, meaning either in top K or not in top K),
and the average rank-CI width. The joint method resolves over 5× more cases than per-task BTL
while maintaining nominal coverage.
6.2.
LM Arena
We use the Chatbot Arena dataset (Arena 140K), restrict to the top-30 most frequently compared
models, and assign each comparison to one of dt = 10 task categories using the platform’s category
metadata. After preprocessing this leaves n = 81,150 pairwise comparisons across dm = 30 models
and dt = 10 tasks. Ground truth T ⋆is defined by the per-task BTL maximum likelihood estimator
on the full data; the joint method uses r = 3, motivated by the empirical singular value spectrum
of T ⋆, in which the leading three singular values capture more than 94% of the energy. We sweep
the subsampling fraction f ∈{0.20,0.50,1.0}, where f = 1 corresponds to using all n comparisons.
Per-task BTL fits each task independently on the available subsample.
Top-K recovery sweep. We report the per-task top-K set Hamming distance of the joint esti-
mator and per-task BTL across all subsampling fractions f ∈{0.02,0.05,0.10,0.20,0.50} (Table 5).
The joint estimator dominates per-task BTL by a wide margin at f ≤0.10 and the two methods
become comparable around f = 0.5; at f = 1.0 per-task BTL coincides with the ground-truth fit-
ting algorithm and so we omit that row. The pattern mirrors the simulation: pooling across tasks
via low-rank structure is most beneficial when the per-task data is sparse.
14

[CAPTION] Table 3
Simulation, single-task rank confidence set for a randomly chosen model on each task at n = 16,000,


<!-- page 15 -->
Table 4
Simulation, simultaneous rank confidence set for one model across all dt = 50 tasks, n = 32,000,
Ktop = 10, N = 200 trials.
Joint (low-rank)
Per-task BTL
Per-task coverage
1.000 (1.000, 1.000)
0.990 (0.988, 0.992)
Resolved (non-trivial)
0.315 (0.310, 0.321)
0.056 (0.053, 0.059)
Mean rank-CI width
31.3 (31.1, 31.4)
46.4 (46.3, 46.5)
Table 5
LM Arena, mean per-task top-K set Hamming distance across the dt = 10 task categories, N = 200
trials.
K = 5
K = 10
f
nsub
Joint
Per-task BTL
Joint
Per-task BTL
0.02
1,623 0.577
0.654
0.411
0.477
0.05
4,057 0.463
0.528
0.311
0.376
0.10
8,115 0.364
0.427
0.251
0.296
0.20 16,230 0.294
0.333
0.205
0.233
0.50 40,575 0.221
0.198
0.144
0.134
Single-task ranking: gemini-2.5-pro on math. We report the single-task rank-CI for
gemini-2.5-pro on the math task (true rank 1). Table 6 sweeps the subsampling fraction; in the
sparse regime the joint method certifies in top-10 on a much larger fraction of trials than per-task
BTL (22% vs. 0% at f = 0.20), and at f = 1 both methods correctly certify with comparable
widths. The joint method’s advantage is precisely in the sparse regime where single-task BTL has
too few comparisons to pin down the rank.
Table 6
LM Arena, single-task rank CI for gemini-2.5-pro on math (true rank 1), Ktop = 10, N = 200 trials.
Joint (low-rank)
Per-task BTL
f
Coverage Cert. rate
Width
Coverage
Cert. rate
Width
0.20
1.000
0.220
15.9
1.000
0.000
21.6
0.30
1.000
0.500
11.6
1.000
0.160
16.3
0.50
1.000
0.520
10.5
1.000
0.320
12.1
1.00
1.000
1.000
7.4
1.000
1.000
6.1
Simultaneous ranking: Gemini-2.5-pro across all tasks. For Gemini 2.5 pro, which per-
forms in top 3 in all tasks, we form simultaneous rank CIs across all dt = 10 tasks via a single
multiplier-bootstrap critical value over the family {(t,ℓ) : t ∈[dt], ℓ̸= m⋆}. Table 7 reports per-task
coverage, the per-task fraction of resolved decisions, and mean rank-CI width. The joint method
resolves ≈87% of (task,trial) cases at f = 1 versus 61% for per-task BTL, with smaller CI widths.
A representative phenomenon is the analytical task, on which the per-task BTL CI never
certifies in top-10 at any subsampling fraction we considered (the rank-CI width is consistently
above 10). With low-rank pooling the joint estimator certifies in top-10 on every trial at f = 1
15

[CAPTION] Table 4
Simulation, simultaneous rank confidence set for one model across all dt = 50 tasks, n = 32,000,

[CAPTION] Table 5
LM Arena, mean per-task top-K set Hamming distance across the dt = 10 task categories, N = 200

[CAPTION] Table 6
LM Arena, single-task rank CI for gemini-2.5-pro on math (true rank 1), Ktop = 10, N = 200 trials.


<!-- page 16 -->
Table 7
LM Arena, simultaneous rank confidence set for gemini-2.5-pro across all dt = 10 tasks, Ktop = 10,
N = 200 trials. Resolved counts (task, trial) instances on which the rule certifies in or out of top-K.
Joint (low-rank)
Per-task BTL
f
Coverage Resolved
Width
Coverage
Resolved Width
0.20
0.998
0.218
17.4
1.000
0.044
23.2
0.30
0.998
0.434
12.7
1.000
0.124
19.6
0.50
0.984
0.742
7.9
1.000
0.270
15.1
1.00
0.958
0.866
6.7
1.000
0.610
9.1
for the same task. Tasks where the chosen model is consistently best (rank 1 on most categories)
similarly benefit from the cross-task information: in the sparse regime, a single task simply does
not have enough comparisons to certify, but pooling across tasks resolves the question.
Both the simulation and LM Arena studies show that exploiting low-rank structure across tasks
reduces estimation error at the rate predicted by our theory and, more importantly, produces
substantially tighter rank confidence sets than per-task BTL, while maintaining the bootstrap
coverage guarantees of Theorem 5.1 and Corollary 5.2. The advantage is largest in the sparse regime
— which is the common operating point in real LLM benchmark deployment — where pooling
information across tasks is the only way to get informative rank inference.
7.
Conclusion
We develop a statistical framework for uncertainty-aware task-specific LLM ranking from sparse
pairwise comparisons using low-rank structure to share information across related tasks while
preserving task-level heterogeneity, yielding entrywise score control, task-wise top-K recovery guar-
antees, efficient score-gap inference, and simultaneous rank/top-K certification. Experiments on
synthetic data and LM Arena comparisons show that low-rank sharing improves sample efficiency
over independent per-task Bradley–Terry estimation and produces tighter, better-calibrated rank-
ing certificates in sparse regimes.
Broader Impact
Task-specific LLM rankings increasingly influence model selection, procurement, deployment, and
public perceptions of model capability. By reporting uncertainty-aware rank and top-K certificates,
our framework helps reduce overconfident claims based on sparse or imbalanced preference data
and makes clear when apparent leaderboard differences are statistically unresolved.
References
Arena Team. How arena works. https://arena.ai/how-it-works, 2026a. Accessed: 2026-05-05.
Arena Team. Introducing max. https://arena.ai/blog/introducing-max/, February 2026b.
Accessed: 2026-05-05.
16

[CAPTION] Table 7
LM Arena, simultaneous rank confidence set for gemini-2.5-pro across all dt = 10 tasks, Ktop = 10,


<!-- page 17 -->
Angel Rodrigo Avelar Menendez, Yufeng Liu, and Xiaowu Dai. Prompt-dependent ranking of large
language models with uncertainty quantification. arXiv e-prints, pages arXiv–2603, 2026.
Vidmantas Bentkus. A lyapunov-type bound in rd. Theory of Probability & Its Applications, 49
(2):311–323, 2005.
Ralph Allan Bradley and Milton E. Terry. Rank analysis of incomplete block designs: I. the method
of paired comparisons. Biometrika, 39(3/4):324–345, 1952.
Emmanuel J. Cand`es and Benjamin Recht. Exact matrix completion via convex optimization.
Foundations of Computational Mathematics, 9(6):717–772, 2009.
Pinhan Chen, Chao Gao, and Anderson Y. Zhang. Partial recovery for top-k ranking: Optimality
of MLE and sub-optimality of spectral method. The Annals of Statistics, 50(3):1618–1652, 2022.
Yuxin Chen and Changho Suh. Spectral MLE: Top-k rank aggregation from pairwise compar-
isons. In Proceedings of the 32nd International Conference on Machine Learning, volume 37 of
Proceedings of Machine Learning Research, pages 371–380, 2015.
Yuxin Chen, Jianqing Fan, Cong Ma, and Kaizheng Wang. Spectral method and regularized mle
are both optimal for top-k ranking. Annals of statistics, 47(4):2204, 2019.
Victor Chernozhukov, Denis Chetverikov, and Kengo Kato. Gaussian approximations and multi-
plier bootstrap for maxima of sums of high-dimensional random vectors. The Annals of Statistics,
41(6):2786–2819, 2013.
Victor Chernozhukov, Denis Chetverikov, and Kengo Kato. Anti-concentration and honest, adap-
tive confidence bands. The Annals of Statistics, 42(5):1787–1818, 2014.
Victor Chernozhukov, Denis Chetverikov, and Kengo Kato. Central limit theorems and bootstrap
in high dimensions. The Annals of Probability, 45(4):2309–2352, 2017.
Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li, Dacheng
Li, Hao Zhang, Banghua Zhu, Michael I. Jordan, Joseph E. Gonzalez, and Ion Stoica. Chat-
bot arena: An open platform for evaluating LLMs by human preference.
arXiv preprint
arXiv:2403.04132, 2024.
Mark A. Davenport, Yaniv Plan, Ewout van den Berg, and Mary Wootters. 1-bit matrix completion.
Information and Inference: A Journal of the IMA, 3(3):189–223, 2014.
Zihan Dong, Zhixian Zhang, Yang Zhou, Can Jin, Ruijia Wu, and Linjun Zhang.
Evaluating
llms when they do not know the answer: Statistical evaluation of mathematical reasoning via
comparative signals. arXiv preprint arXiv:2602.03061, 2026.
Jianqing Fan, Jikai Hou, and Mengxin Yu. Uncertainty quantification of MLE for entity ranking
with covariates. Journal of Machine Learning Research, 25(358):1–83, 2024.
Jianqing Fan, Hyukjun Kwon, and Xiaonan Zhu.
Uncertainty quantification for ranking with
heterogeneous preferences. arXiv preprint arXiv:2509.01847, 2025.
17


<!-- page 18 -->
Jianqing Fan, Zhipeng Lou, Weichen Wang, and Mengxin Yu. Spectral ranking inferences based
on general multiway comparisons. Operations Research, 74(1):161–180, 2026.
Evan Frick, Connor Chen, Joseph Tennyson, Tianle Li, Wei-Lin Chiang, Anastasios Nikolas
Angelopoulos, and Ion Stoica.
Prompt-to-leaderboard: Prompt-adaptive llm evaluations.
In
Forty-second International Conference on Machine Learning, 2025.
David R. Hunter. MM algorithms for generalized bradley–terry models. The Annals of Statistics,
32(1):384–406, 2004.
Jiachun Li, David Simchi-Levi, and Will Wei Sun. LLM evaluation as tensor completion: Low rank
structure and semiparametric efficiency. arXiv preprint arXiv:2604.05460, 2026.
R. Duncan Luce. Individual Choice Behavior: A Theoretical Analysis. Wiley, New York, 1959.
Yasmin Moslem and John D Kelleher.
Dynamic model routing and cascading for efficient llm
inference: A survey. arXiv preprint arXiv:2603.04445, 2026.
Sahand N. Negahban and Martin J. Wainwright. Restricted strong convexity and weighted matrix
completion: Optimal bounds with noise. Journal of Machine Learning Research, 13(53):1665–
1697, 2012.
Colin White, Samuel Dooley, Manley Roberts, Arka Pal, Ben Feuer, Siddhartha Jain, Ravid
Shwartz-Ziv, Neel Jain, Khalid Saifullah, Siddartha Naidu, Chinmay Hegde, Yann LeCun, Tom
Goldstein, Willie Neiswanger, and Micah Goldblum. Livebench: A challenging, contamination-
free LLM benchmark. The Thirteenth International Conference on Learning Representations,
2025.
Appendix A:
Notation, assumptions, and master good event
This appendix collects the notation, assumptions, and probability calibrations used throughout Appen-
dices B–E.10. All assumptions in this appendix are stated in the matrix (dt × dm) form and are the matrix
specialization of the assumptions used in the prior efficient-inference paper of Li et al. [2026].
A.1.
Notation
We retain the notation of Section 2. The latent ability matrix is Θ⋆∈Rdt×dm, row-centered (Θ⋆1dm = 0), of
rank r with reduced singular value decomposition
Θ⋆= U ⋆Σ⋆(V ⋆)⊤,
U ⋆∈Rdt×r, V ⋆∈Rdm×r, Σ⋆= diag(σ⋆
1,...,σ⋆
r).
The singular vectors are µ-incoherent, the condition number is κ := σ⋆
1/σ⋆
r, and the entrywise bound is
∥Θ⋆∥∞≤B. We write ¯d := max(dt,dm) for the maximum mode dimension and d⋆:= dtdm for the ambient
cardinality of the matrix. The effective comparison dimension, which appears throughout the analysis as the
natural normalization for pairwise contrasts, is
d⋆
eff := dt (dm −1)
2
≍dt dm
2
= d⋆
2 ,
and we will use d⋆and d⋆
eff interchangeably up to the absolute constant 2.
18


<!-- page 19 -->
Design and observation model. For each round i ∈[n], the design tensor is Xi := eti(emi −em′
i)⊤∈
Rdt×dm, where the task index ti ∈[dt] is sampled from a distribution ν and the unordered model pair
{mi,m′
i} ⊂[dm] is sampled from a task-dependent distribution πti. The predictor is η⋆
i := ⟨Xi, Θ⋆⟩= Θ⋆
ti,mi −
Θ⋆
ti,m′
i, and the observation is Yi ∼Bernoulli(σ(η⋆
i )) with σ(x) := (1 + e−x)−1 the logistic link. The pairwise
score and Fisher weight are
sη(y,η) := ∂ηℓ(y,η) = y −σ(η),
I(η) := E[sη(Y,η)2 | η] = σ(η){1 −σ(η)} ∈(0,1/4].
Under ∥Θ⋆∥∞≤B, the predictor satisfies |η⋆| ≤2B, and consequently the Fisher information is bounded
above and below: there exist constants 0 < cB ≤CB < ∞depending only on B such that cB ≤I(η⋆) ≤CB
almost surely.
Tangent space and operators. The signal tangent space at Θ⋆intersected with the row-centering
identification constraint is
T := {U ⋆A⊤+ QC (V ⋆)⊤: A ∈Rdm×r, C ∈R(dt−1)×r},
where Q ∈Rdt×(dt−1) has orthonormal columns spanning 1⊥
dt. The orthogonal projector onto T is denoted PT;
its closed-form representation is given in Appendix D.4. Following Li et al. [2026], the Fisher (information)
operator G : Rdt×dm →Rdt×dm is defined by
⟨GU, V ⟩:= E⋆[I(η⋆)⟨U, X⟩⟨V, X⟩],
U,V ∈Rdt×dm,
(A.1)
and the restricted information operator on the tangent space is A := PTGPT : T →T. For any contrast
Γ ∈Rdt×dm, the efficient direction is H⋆
Γ := A−1PTΓ and the efficient influence function is
ϕΓ(Wi) := sη(Yi,η⋆
i )⟨H⋆
Γ, Xi⟩,
Veff(Γ) := E⋆[ϕ2
Γ] =

PTΓ, A−1PTΓ
 
.
Plug-in operators. Given an initial estimator bΘ of Θ⋆and the estimated singular subspaces bU, bV
(obtained, e.g., from the rank-r SVD of bΘ), we define the estimated tangent projector bPT using bU, bV in place
of U ⋆,V ⋆, and the plug-in information operator
D
bGU, V
E
:=
1
naux
X
i∈Daux
I(bηi) ⟨U, Xi⟩⟨V, Xi⟩,
bηi :=
D
bΘ, Xi
E
,
(A.2)
computed on the auxiliary sample Daux with naux = |Daux|. The estimated restricted information operator
is bA := bPT bG bPT, and the estimated efficient direction is bHΓ := bA−1 bPTΓ.
Norms. We use the following norms. For a matrix M ∈Rdt×dm, ∥M∥F is the Frobenius norm,
∥M∥∞:= maxt,m |Mt,m| the entrywise norm, ∥M∥∗the nuclear norm, ∥M∥op the spectral norm, ∥M∥2,∞:=
maxt ∥e⊤
t M∥2 the row ℓ2/ℓ∞norm, and ∥M∥∞→∞the induced ℓ∞→ℓ∞operator norm. For a vector v,
∥v∥1 := P
i |vi|, ∥v∥2 the Euclidean norm, ∥v∥∞:= maxi |vi|. For tensors / linear operators acting on Rdt×dm,
these norms apply to the tensor when flattened as a vector.
19


<!-- page 20 -->
A.2.
Score regularity
The following is the BTL specialization of the abstract score-regularity assumption used in Li et al. [2026].
Assumption A.1 (Score regularity for the BTL link). The BTL log-likelihood satisfies, with sη
and I as above:
(i) Centering. E⋆[sη(Y,η⋆) | X] = 0.
(ii) Bounded support. ∥Θ⋆∥∞≤B for a constant B > 0, so |η⋆| ≤2B almost surely, and hence the Fisher
weight satisfies
cB ≤I(η⋆) ≤CB,
where cB := σ(2B)σ(−2B) > 0 and CB := 1/4 depend only on B.
(iii) Bounded score derivatives. For every (y,η) in the relevant range, |∂ηsη(y,η)| ≤1/4 and |∂2
ηsη(y,η)| ≤L3
for an absolute constant L3 > 0 (one can take L3 = 1/(6
√
3)).
(iv) Sub-Gaussian / sub-exponential tail. Since sη(Y,η) = Y −σ(η) ∈[−1,1] almost surely, every moment is
bounded uniformly: ∥sη(Y,η)∥ψ2 ≤1 and ∥sη(Y,η)∥ψ1 ≤1.
Remark A.2 (BTL verification). For the BTL model, sη(y,η) = y −σ(η), ∂ηsη = −σ′(η) = −I(η) ∈
[−1/4,0], and ∂2
ηsη = −σ′′(η) is uniformly bounded. The boundedness of I(η⋆) under ∥Θ⋆∥∞≤B is the
consequence of |η⋆| ≤2B plus the strict positivity of σ′ on any compact interval of R.
A.3.
Sampling design
We work under the near-uniform sampling design of Section 2.
Assumption A.3 (Near-uniform sampling). There exist constants 0 < cν ≤Cν < ∞and 0 < cπ ≤
Cπ < ∞, independent of dt,dm,n, such that for every task t ∈[dt] and every unordered pair {m,m′} ⊂[dm],
cν
dt
≤νt ≤Cν
dt
,
cπ
 dm
2
  ≤πt({m,m′}) ≤Cπ
 dm
2
 .
In the analysis we routinely use the following two consequences of Assumption A.3.
Lemma A.4 (Pairwise Frobenius reduction in the matrix case). Let H ∈Rdt×dm satisfy the row-
centering condition H1dm = 0. Then under Assumption A.3,
E⋆ 
⟨H, X⟩2 
≍∥H∥2
F
d⋆
,
d⋆= dtdm.
(A.3)
Proof.
Conditional on a task t and an unordered pair {m,m′}, ⟨H, X⟩= Ht,m −Ht,m′. Fixing the task t
and letting z(t) ∈Rdm be the t-th row of H, with P
m zm(t) = 0 by hypothesis, we use the elementary identity
X
m<m′
(zm(t) −zm′(t))2 = dm
X
m
zm(t)2.
Under near-uniform pair sampling, E{m,m′}[(zm(t) −zm′(t))2] ≍2∥z(t)∥2
2 /(dm −1). Averaging over t ∼ν
using νt ≍1/dt yields E⋆[⟨H, X⟩2] ≍∥H∥2
F /(dt(dm −1)) ≍∥H∥2
F /d⋆.
Lemma A.5 (Weighted second-moment for the Fisher operator). Under Assumptions A.1–A.3,
for any row-centered H, cB ∥H∥2
F /d⋆≲E⋆[I(η⋆)⟨H, X⟩2] ≲CB ∥H∥2
F /d⋆. In particular, under ∥Θ⋆∥∞≤B,
the operator G restricted to the row-centered subspace satisfies cB/d⋆≤∥G∥op ≤CB/d⋆, where ∥·∥op is the
operator norm on the row-centered subspace.
20


<!-- page 21 -->
A.4.
Initial estimator and signal strength
The following is the matrix specialization of [Li et al., 2026, Assumption 3.4].
Assumption A.6 (Initial estimator). The initial estimator bΘ computed on an auxiliary sample Daux
of size naux ≍n satisfies the row-centering gauge bΘ1dm = 0 and the entrywise rate
bΘ −Θ⋆
∞≤C1
s
¯d logc ¯d
naux
(A.4)
for absolute constants C1,c > 0.
The estimator constructed in Appendix B satisfies Assumption A.6. In particular, the nuclear-norm penal-
ized convex initializer of Appendix B.1 produces a Frobenius-accurate intermediate estimator, and the
three-split row-wise refinement of Appendices B.2–B.8 upgrades it to the entrywise rate (A.4).
Assumption A.7 (Signal strength). The Frobenius norm of the latent score matrix satisfies
∥Θ⋆∥F ≥csig
√
d⋆
(A.5)
for an absolute constant csig > 0.
By the rank constraint and the upper bound ∥Θ⋆∥2
F = Pr
i=1(σ⋆
i )2 ≤r(σ⋆
1)2 together with κ = σ⋆
1/σ⋆
r,
Assumption A.7 gives σ⋆
r ≍
√
d⋆up to factors of r and κ, which are bounded constants. Hence the spectral
signal-to-noise ratio scales as
√
d⋆since the noise level under the BTL model is of constant order.
A.5.
Functional regularity for inference
For inference on a contrast ψΓ(Θ) = ⟨Γ, Θ⟩, we require structural assumptions on Γ so that the score-gap
contrasts considered in Section 5 are admissible.
Assumption A.8 (Bounded ℓ1 gradient and finite support). The gradient Γ = ∇ψ(Θ⋆) ∈Rdt×dm
has |supp(Γ)| ≤M and ∥Γ∥1 ≤Cψ for absolute constants M,Cψ > 0.
For the score-gap contrasts Γ = et(em −em′)⊤used in Sections 4–5, Assumption A.8 holds with M = 2 and
Cψ = 2.
Assumption A.9 (Alignment). There exists a constant αΓ > 0 such that
∥PTΓ∥F ≥αΓ ¯d 1/2(d⋆)−1/2 ∥Γ∥F .
(A.6)
Assumption A.9 ensures that the contrast Γ has a non-negligible component in the tangent space. When
Γ ∈T, αΓ = 1. For sparse score-gap contrasts, αΓ is bounded below by an incoherence-dependent constant;
see Lemma A.10 below.
Lemma A.10 (Alignment for sparse score-gap contrasts). For a score-gap contrast Γ = et(em −
em′)⊤∈Rdt×dm, under µ-incoherence, αΓ ≥c(µ,r) > 0 for an explicit constant depending only on (µ,r).
Proof.
Compute ∥Γ∥F =
√
2 and ¯d1/2/(d⋆)1/2 = 1/min(d1/2
t
,d1/2
m ). Using the closed-form projector (D.5)
and µ-incoherence, one verifies ∥PTΓ∥F ≳
p
2/dt +
p
2/dm, giving the claimed lower bound on αΓ. Details
follow the same calculation as in [Li et al., 2026, Lemma B.5].
21


<!-- page 22 -->
A.6.
Sample size and simplified rates
We work under the near-optimal sample-size scaling.
Assumption A.11 (Sample size). The sample size satisfies
n ≥C0 ¯d logc ¯d
(A.7)
for
a
sufficiently
large
absolute
constant
C0 > 0
depending
only
on
the
structural
parameters
(µ,r,κ,B,cν,Cν,cπ,Cπ,csig,αΓ).
Under Assumption A.11, the spectral signal-to-noise ratio σr(Θ⋆)/σ ≍
√
d⋆(since σ = O(1) under the BTL
model) automatically dominates the noise scale required by the subspace-perturbation theory, and all explicit
σr(Θ⋆)-dependent factors in the AoS analysis collapse into the structural constants. We therefore state all
rates in the simplified form
rn :=
r ¯d logc(n ¯d)
n
=
r ¯dpolylog(n ¯d)
n
.
A.7.
The constant CA
The following constant captures the entrywise stability of the inverse restricted information operator and is
the only structural quantity that may depend on the dimension.
Definition A.12 (The constant CA). The constant CA > 0 is the smallest constant such that
A−1
∞→∞∨
 bA−1
∞→∞≤CA d⋆.
(A.8)
Proposition A.13 (Range of CA; from Li et al. [2026]). Under
the
bounded-signal
condition
∥Θ⋆∥∞≤B and µ-incoherence,
(i) Coarse range. C(µ,r) ≤CA ≤C(µ,r)
√¯d.
(ii) Constant-weight benchmark. If I(η⋆) ≡σ′(0) = 1/4 is exactly constant, then A0 = PT/(2d⋆) and A−1
0
=
2d⋆PT, which gives CA = 2∥PT∥∞→∞= poly(µ,r).
(iii) Near-constant regime. If ∥Θ⋆∥∞≤B0 for a sufficiently small constant B0 = B0(µ,r), then CA ≤
C(µ,r,B0) is dimension-free.
In the body of this paper we treat CA as a generic structural constant. All inferential statements are sharp
up to a single multiplicative CA factor in the remainder. Crucially, the CA factor enters at most linearly (i.e.
as C1
A, never as a higher power), which is essential for the sample-size scaling (A.7) to suffice.
A.8.
Probability calibration to 1 −n−a
Throughout, all ”high-probability” statements are uniform over a free constant a > 0 that may be taken
arbitrarily large at the cost of an absolute constant prefactor absorbed into polylog(n ¯d). Specifically, every
high-probability bound below has the form Pr(good event) ≥1 −n−a for any fixed a > 0.
Origin of the calibration. The underlying ingredients are Bernstein- or matrix-Bernstein-type concen-
tration inequalities with a free tail parameter x: for each such bound, substituting x = Calog(n ¯d) yields
the claimed 1 −n−a form. In particular, statements of the form ”with probability at least 1 −d−c” or
”with probability at least 1 −c1 exp(−c2dlog d)” appearing in the prior literature are tightened to 1 −n−a
22


<!-- page 23 -->
by enlarging the constant in the corresponding tail bound. The replacement is permissible because under
Assumption A.11, dlog d ≳alog n, so an exponential bound of the form 1 −c1 exp(−c2dlog d) is ≥1 −n−a
for c2 chosen sufficiently large.
The master good event En. Let a > 0 be any fixed (large) constant. Define
En := Einit ∩Erefine ∩Erem ∩EBE ∩Evar ∩Ecov ∩EccK,
where the constituent events are as follows.
(1) Einit — Frobenius accuracy of the convex initializer (Theorem B.5).
(2) Erefine — entrywise accuracy of the refined estimator (Theorem B.26); this implies E∞= {
bΘ −Θ⋆
∞≤
εn} with εn = Crn.
(3) Erem — uniform single-contrast one-step remainder (Theorem D.6).
(4) EBE — multivariate Berry–Esseen for the oracle leading term (Theorem D.7).
(5) Evar — relative variance consistency for plug-in standard errors (Proposition D.11).
(6) Ecov — entrywise covariance consistency (Proposition D.12).
(7) EccK — CCK aggregate approximate-means error (Theorem E.7).
Each event holds with probability at least 1 −n−a; by the union bound,
Pr(Ec
n) ≤7n−a ≤n−a/2,
and we relabel a so that Pr(En) ≥1 −n−a. All deterministic statements in the proofs are made on En; the
deficit Pr(Ec
n) = o(1) is absorbed into the o(1) slack of every coverage statement in Appendix E.
A.9.
Balanced-regime simplification
For ease of presentation, we state the rates in the balanced regime dt ≍dm =: d, so that ¯d ≍d and d⋆≍d2. All
proofs go through verbatim in the rectangular case with ¯d = max(dt,dm) and d⋆= dtdm; the only difference
is that the left-factor analysis (Appendix B.5) uses dt and the right-factor analysis (Appendix B.7) uses dm,
and the entrywise rate is
p ¯dpolylog(n ¯d)/n. Hidden constants depend only on the structural parameters
(µ,r,κ,B,cB,CB) of the score and signal, the design constants (cν,Cν,cπ,Cπ), the alignment αΓ, the signal-
strength constant csig, and the probability-calibration constant a.
Appendix B:
Proof of Theorem 3.1: convex initializer and entrywise refinement
This appendix proves Theorem 3.1 of Section 3 via the two-stage estimator bΘ = Refiner(bΘ0) constructed
from a nuclear-norm penalized convex MLE bΘ0 (initializer) followed by a three-split row-wise refinement.
The structure of the appendix is as follows.
• Appendix B.1 establishes the convex stage. We adapt the restricted strong convexity (RSC) framework of
Negahban and Wainwright [2012] to the pairwise logistic loss, obtaining a Frobenius-accurate initializer
bΘ0 with rate
bΘ0 −Θ⋆
F ≲
p
r dt dm ¯d log ¯d/n under the row-centering identifiability constraint.
• Appendix B.2 sets up the three-split refinement algorithm and the proof roadmap (six blocks).
23


<!-- page 24 -->
• Appendix B.3 establishes the Brouwer inward-pointing zero lemma which underlies the deterministic
existence steps.
• Appendix B.4 (Block II) gives the deterministic local-existence statement for the row update.
• Appendix B.5 (Block III) verifies the existence condition probabilistically: six concentration lemmas
(coverage, noise, two bias terms, higher-order moments, curvature) and a uniform proposition delivering
bΘL −Θ⋆
L

2,∞≲dm/√n2 polylog(n ¯d).
• Appendix B.6 (Block IV) handles the gauge re-centering and proves the pairwise Gram identity that
upgrades the column-update curvature to Θ(d2
t).
• Appendix B.7 (Blocks V and VI) mirrors B.4–B.5 for the right factor, yielding
bΘR −Θ⋆
R

2,∞≲
n−1/2 polylog(n ¯d).
• Appendix B.8 (final assembly) combines the two factor bounds into the entrywise rate
bΘ −Θ⋆
∞≲
p ¯dpolylog(n ¯d)/n, proving Theorem 3.1.
B.1.
Stage 1: convex initializer via pairwise-logistic RSC
We split the comparison sample into three independent parts D1,D2,D3 of sizes n1,n2,n3 ≍n. The convex
stage uses D1 only. For convenience write M ⋆:= Θ⋆, d1 := dt, d2 := dm in this subsection so that the notation
matches [Li et al., 2026, Algorithm 2 and Section G.5.1].
B.1.1.
Estimator
Let ℓ(y,η) = log(1 + eη) −yη be the logistic negative log-likelihood and define the
empirical risk
Ln1(M) := 1
n1
X
i∈D1
ℓ
 Yi,⟨Xi, M⟩
 
.
The convex estimator is the nuclear-norm penalized MLE under the row-centering and entrywise-bound
constraints:
c
M ∈arg min
M∈CB
 
Ln1(M) + λ∥M∥∗
	
,
CB := {M ∈Rd1×d2 : M1d2 = 0, ∥M∥∞≤B}.
(B.1)
Since M ⋆∈CB by Assumption A.1(ii), the truth is feasible. The choice of λ is fixed in Theorem B.5 below.
We obtain the rank-r initializer by truncating the SVD of c
M to its top r singular components, followed by
entrywise clipping:
bΘ0 := clipB
 SVDr(c
M)
 
,
where clipB(·) projects each entry onto [−B,B]. The clipping step does not increase the entrywise distance
to M ⋆because ∥M ⋆∥∞≤B; it ensures that the initializer remains in the feasible bounded-signal class.
The role of the row-centering constraint is essential: pairwise comparisons depend only on within-task
differences, so any matrix of the form 1d1c⊤lies in the null space of the design. Without the constraint, no
RSC statement can hold.
24


<!-- page 25 -->
B.1.2.
Step 1: the population pairwise quadratic identity
We begin with the one genuinely
pairwise-specific algebraic identity.
Lemma B.1 (Population pairwise quadratic identity). Let X = et(em −em′)⊤where t ∈[d1] is sam-
pled from ν and {m,m′} ⊂[d2] from a task-dependent distribution πt satisfying Assumption A.3. For every
∆∈Rd1×d2 with ∆1d2 = 0,
E
 
⟨X, ∆⟩2 
≍
2
d1 (d2 −1) ∥∆∥2
F ≍∥∆∥2
F
d1 d2
.
(B.2)
Proof.
Conditional on t, ⟨X, ∆⟩= ∆t,m −∆t,m′. Letting z(t) ∈Rd2 denote the t-th row of ∆, with
P
m zm(t) = 0 by hypothesis, the standard sum-of-squared-pairwise-differences identity gives
X
m<m′
(zm(t) −zm′(t))2 = d2
d2
X
m=1
zm(t)2.
Under near-uniform pair sampling (Assumption A.3), the conditional expectation satisfies E{m,m′}[(zm(t) −
zm′(t))2 | t] ≍2∥z(t)∥2
2 /(d2 −1). Averaging over t ∼ν using νt ≍1/d1 and summing ∥z(t)∥2
2 over t yields
E[⟨X, ∆⟩2] ≍2∥∆∥2
F /(d1(d2 −1)).
The identity (B.2) is the pairwise analogue of the Negahban–Wainwright population norm-equivalence
identity for matrix completion. The centered gauge 1⊤∆= 0 is essential: ⟨X, ∆⟩= 0 identically for any
∆= 1c⊤, so without centering the left-hand side of (B.2) would vanish on a non-trivial subspace.
B.1.3.
Step 2: pairwise quadratic restricted strong convexity
Define the spikiness ratio αsp(∆) :=
√d1d2 ∥∆∥∞/∥∆∥F and the rank surrogate βra(∆) := ∥∆∥∗/∥∆∥F, and let
Cpw(n1;c0) :=
n
∆∈Rd1×d2 : 1⊤
d1∆= 0, αsp(∆)βra(∆) ≤1
c0
r
n1
¯dlog ¯d
o
.
Theorem B.2 (Pairwise quadratic RSC). Fix any a > 0. There exist absolute constants c0,c,C > 0
such that whenever n1 ≥C ¯dlog ¯d, with probability at least 1 −n−a,
1
n1
X
i∈D1
⟨Xi, ∆⟩2 ≥
c
d1 d2
∥∆∥2
F
for all ∆∈Cpw(n1;c0).
(B.3)
Proof.
The proof follows the four-step Negahban–Wainwright peeling/contraction template, with the
population identity of Lemma B.1 replacing the matrix-completion norm-equivalence.
Reduction to a single-scale event by peeling. Define the empirical Frobenius proxy F∆:=
  1
n1
P
i ⟨Xi, ∆⟩2 1/2 and the population scale µ∆:=
p
2/(d1(d2 −1)) ∥∆∥F, so that E[F 2
∆] = µ2
∆by
Lemma B.1. Since µ∆≥∥∆∥F /√d1d2, it suffices to show that F∆≥µ∆/2 uniformly on the restricted set
Cpw(n1;c0) (this gives the stated lower bound with c = 1/(2d1d2) up to constant adjustments). Partition the
Frobenius range of the restricted set into dyadic shells Sℓ= {∆: ∥∆∥F ∈[αℓ−1µ0,αℓµ0]} for α = 7/6 and µ0 a
small base scale; a union bound over ℓ= 1,...,L (with L = O(log n1)) reduces the problem to a single-scale
event at each shell.
Discretization. Fix a scale D and consider the localized restricted set S(D) := Cpw(n1;c0)∩{∥∆∥F ≤D}.
By Sudakov minoration applied to the metric induced by the Frobenius norm, restricted to a nuclear-norm
ball of radius ρ(D) := D2p
log ¯d/(n1 ¯d)/c0, the metric entropy at scale D/8 satisfies
log N
 S(D),∥·∥F ,D/8
 
≤C ρ(D)2
(D/8)2 ¯d = C D2 log ¯d
c2
0 n1 ¯d · ¯d = C D2 log ¯d
c2
0 n1
.
25


<!-- page 26 -->
Let ∆1,...,∆N0 be a corresponding D/8-net. By the reverse triangle inequality, F∆≥F∆k −FΣ for the
nearest net point ∆k and remainder Σ := ∆−∆k with ∥Σ∥F ≤D/8.
Net lower-tail concentration. For each fixed ∆k, the map (ξ1,...,ξn1) 7→F∆k with ξi := ⟨Xi, ∆k⟩is
(1/√n1)-Lipschitz with respect to the Euclidean norm on the coordinate domain, and the coordinates satisfy
|ξi| ≤2∥∆k∥∞≤2/√d1d2 · αsp(∆k)∥∆k∥F ≤2
p
n1/( ¯dlog ¯d)/(c0
√d1d2)∥∆k∥F. By the bounded-difference
(McDiarmid) concentration inequality applied to the one-sided lower tail of F∆k,
Pr
h
F∆k <
1
√d1d2
∥∆k∥F −t −
C
√d1d2√n1
i
≤4exp
 
−n1 d1d2 t2
64
 
.
Setting t = D/(8√d1d2) gives a tail of order exp(−n1D2/(C′d1d2 · d1d2)) = exp(−n1D2/C′′(d1d2)), which
by a union bound over the net (which has log-cardinality ≤CD2 log ¯d/(c2
0n1)) is dominated by exp(−n1D2)
for c0 chosen sufficiently large. Choosing c0 large enough to absorb log ¯d factors, this gives Pr[F∆k <
D/(2√d1d2) for some k] ≤n−a after taking a large enough.
Remainder supremum. The remainder supremum supΣ:∥Σ∥F ≤D/8, ∥Σ∥∗≤2ρ(D) FΣ is controlled by sym-
metrization, the Ledoux–Talagrand contraction inequality (which converts x2 to |x| at the cost of a factor
of 2), and operator/nuclear-norm duality. Specifically, by symmetrization,
E
h
sup
Σ
F 2
Σ
i
≤2E
h
sup
Σ
1
n1
X
i
εi ⟨Xi, Σ⟩2i
,
where {εi} are i.i.d. Rademacher signs. Contraction with respect to the squaring nonlinearity x 7→x2 (which
is L-Lipschitz on the bounded range [−2/√d1d2,2/√d1d2] with L ≲1/√d1d2) gives
E
h
sup
Σ
1
n1
X
i
εi ⟨Xi, Σ⟩2i
≤
C
√d1d2
E
h
sup
Σ
    1
n1
X
i
εi ⟨Xi, Σ⟩
   
i
.
The latter Rademacher sum is bounded by operator-norm duality:
sup
∥Σ∥∗≤2ρ(D)
    1
n1
X
i
εi ⟨Xi, Σ⟩
    ≤2ρ(D)
 1
n1
X
i
εiXi

op.
By the matrix Bernstein inequality applied to the centered rank-one sum P
i εiXi with envelope ∥Xi∥op ≤
√
2
and variance E[XiX⊤
i ] ⪯2/(d2 −1)Id1, we get E∥n−1
1
P
i εiXi∥op ≲
p
log ¯d/(n1 ¯d). Plugging in,
E
h
sup
Σ
F 2
Σ
i
≲
1
√d1d2
· ρ(D) ·
s
log ¯d
n1 ¯d =
D2
c0 d1d2
· log ¯d
n1 ¯d · ¯d = D2 log ¯d
c0 d1d2 n1
.
Choosing c0 sufficiently large makes the right-hand side ≤D2/(64d1d2), and the bounded-difference concen-
tration upgrades this to supΣ FΣ ≤D/(8√d1d2) with probability at least 1 −n−a.
Combining. On the intersection of the net-lower-tail event and the remainder-supremum event, F∆≥
F∆k −FΣ ≥D/(2√d1d2) −D/(8√d1d2) ≥D/(4√d1d2), which together with µ∆≤
√
2/√d1d2 ∥∆∥F and
∥∆∥F ≥D/α on shell Sℓ(α = 7/6) gives F∆≥µ∆/2, as required. Closing the induction over shells via the
peeling argument concludes the proof.
26


<!-- page 27 -->
B.1.4.
Step 3: from quadratic RSC to logistic RSC
The pairwise logistic loss is not quadratic, but
its Bregman divergence inherits the quadratic curvature on the bounded-signal feasible set.
Lemma B.3 (Logistic curvature reduction). For any ∆with M ⋆+ ∆∈CB,
δLn1(M ⋆;∆) := Ln1(M ⋆+ ∆) −Ln1(M ⋆) −⟨∇Ln1(M ⋆), ∆⟩≥cB
2n1
X
i∈D1
⟨Xi, ∆⟩2 ,
(B.4)
where cB := inf|x|≤2B σ′(x) > 0. Consequently, on the event of Theorem B.2,
δLn1(M ⋆;∆) ≥κpw ∥∆∥2
F ,
κpw ≍
1
d1d2
,
holds for all ∆∈Cpw(n1;c0) with M ⋆+ ∆∈CB.
Proof.
Taylor’s theorem applied to η 7→ℓ(y,η) at ⟨Xi, M ⋆⟩gives, for some intermediate ξi with |ξi| ≤2B,
δLn1(M ⋆;∆) =
1
2n1
X
i∈D1
σ′(ξi)⟨Xi, ∆⟩2 .
By definition of cB, σ′(ξi) ≥cB > 0, giving the first inequality. Combining with (B.3) yields the second.
B.1.5.
Step 4: gradient operator-norm bound
Lemma B.4 (Gradient operator-norm bound). Fix any a > 0. With probability at least 1 −n−a,
∥∇Ln1(M ⋆)∥op ≤C
s
log(d1 + d2)
n1 (d1 ∧d2) .
(B.5)
Proof.
The gradient at the truth is ∇Ln1(M ⋆) = n−1
1
P
i(σ(⟨Xi, M ⋆⟩) −Yi)Xi. Each summand is mean
zero (by the model) and has operator norm at most
√
2 (since ∥Xi∥op =
√
2 and the scalar prefactor
σ(⟨Xi, M ⋆⟩) −Yi ∈[−1,1]). The matrix variance proxy on the right is
E[(σ(⟨Xi, M ⋆⟩) −Yi)2XiX⊤
i ] ⪯E[XiX⊤
i ] ≍
2
d2 −1 Id1 ⪯
C
d2 −1Id1;
the
left
variance
is
similarly
bounded
by
O(1/(d1
−
1))Id2.
Hence
σ2
X
:=
max(∥P
i E[XiX⊤
i ]∥op ,∥P
i E[X⊤
i Xi]∥op) ≤Cn1/(d1 ∧d2). By the rectangular matrix Bernstein inequality
(Tropp 2015, Theorem 6.1.1) applied to the rescaled sum,
Pr
hn−1
1
P
i(σ −Y )Xi

op ≥t
i
≤(d1 + d2)exp
 
−
n1 t2/2
C/(d1 ∧d2) +
√
2t/3
 
.
Setting t = C
p
alog(d1 + d2)n−1/(d1 ∧d2) for C large enough yields the claimed bound with probability
1 −n−a.
B.1.6.
Step 5: main convex initialization theorem
Theorem B.5 (Frobenius bound for the convex initializer). Fix any a > 0. Under the model
assumptions of Section 2 and Assumption A.3, set λ := 2C
p
log(d1 + d2)/(n1 (d1 ∧d2)) where C is the con-
stant from Lemma B.4, and assume n1 ≥C poly(µ,r,κ,B) ¯dlogc ¯d. Then with probability at least 1 −n−a,
c
M −M ⋆
F ≤C λ√r
κpw
≤C′
s
r d1 d2 ¯d log ¯d
n1
≍
s
r ¯d 3 log ¯d
n1
.
(B.6)
Consequently,
the
rank-truncated,
clipped
initializer
bΘ0 = clipB(SVDr(c
M))
satisfies
bΘ0 −Θ⋆
F ≤
2
c
M −M ⋆
F ≤C
p
r ¯d 3 log ¯d/n1 on the same event.
27


<!-- page 28 -->
Proof.
Set b∆:= c
M −M ⋆and let Tr denote the rank-r tangent space at M ⋆(without the row-centering
constraint), with associated decomposition b∆= b∆Tr + b∆T⊥
r .
Basic inequality. Since M ⋆∈CB is feasible, optimality of c
M gives
Ln1(M ⋆+ b∆) + λ
M ⋆+ b∆

∗≤Ln1(M ⋆) + λ∥M ⋆∥∗.
Rearranging,
δLn1(M ⋆; b∆) ≤−
D
∇Ln1(M ⋆), b∆
E
+ λ
 ∥M ⋆∥∗−
M ⋆+ b∆

∗
 
.
By the operator/nuclear-norm duality and our choice λ ≥2∥∇Ln1(M ⋆)∥op (which holds on the event of
Lemma B.4), |
D
∇Ln1(M ⋆), b∆
E
| ≤(λ/2)
b∆

∗. By the standard nuclear-norm decomposability, ∥M ⋆∥∗−
M ⋆+ b∆

∗≤
b∆Tr

∗−
b∆T⊥
r

∗. Combining,
δLn1(M ⋆; b∆) ≤3λ
2
b∆Tr

∗−λ
2
b∆T⊥
r

∗.
Since δLn1 ≥0, this forces the cone condition
b∆T⊥
r

∗≤3
b∆Tr

∗, and using rank(b∆Tr) ≤2r we obtain
b∆Tr

∗≤
√
2r
b∆

F and hence
b∆

∗≤4
√
2r
b∆

F.
Applying RSC. The cone condition implies b∆∈Cpw(n1;c0) (modulo absorbing a factor of √r into the
constant c0, because the spikiness ratio is bounded by entrywise feasibility, αsp(b∆) ≤√d1d2 (2B)/
b∆

F,
and the rank-surrogate ratio βra(b∆) ≤4
√
2r on the cone). Hence on the event of Theorem B.2,
δLn1(M ⋆; b∆) ≥κpw
b∆

2
F ,
κpw ≍cB
d1d2
.
Combining. On the same event,
κpw
b∆

2
F ≤δLn1(M ⋆; b∆) ≤3λ
2
b∆Tr

∗≤3λ
√
2r
2
b∆

F .
Dividing by
b∆

F,
b∆

F ≤3λ
√
2r
2κpw
≤C λ√r d1 d2.
Substituting λ ≍
p
log ¯d/(n1 ¯d) gives
b∆

F ≤C
p
r d2
1d2
2 log ¯d/(n1 ¯d) = C
p
r ¯d 3 log ¯d/n1 in the balanced
regime.
If on the other hand b∆/∈Cpw(n1;c0), then by the cone condition and the entrywise feasibility constraint
b∆

∞≤2B, one has
b∆

F ≤C(B,r) ¯d
p
log ¯d/n1, which is of the same or smaller order under n1 ≥C ¯dlog ¯d.
The post-processing claim for bΘ0 follows because rank-r SVD truncation contracts Frobenius distance to
M ⋆by at most a factor of 2 (the best rank-r approximation), and entrywise clipping is non-expansive against
M ⋆under ∥M ⋆∥∞≤B.
Remark B.6 (Comparison to matrix completion). The rate
bΘ0 −Θ⋆
F ≲
p
r ¯d 3 log ¯d/n differs
from the Negahban–Wainwright matrix-completion rate
p
r ¯dlog ¯d/n by a factor of ¯d. This reflects the d−2 gap
between the pairwise population identity (B.2) and the entry-completion identity E[⟨X, ∆⟩2] = ∥∆∥2
F /(d1d2):
each pairwise comparison carries O(1/d2) of the information per entry, making the d-factor overhead unavoid-
able under sparse pairwise observations. The factor is recovered in the Frobenius rate but does not appear
in the final entrywise rate after refinement (Theorem B.26), because refinement uses fresh observations to
upgrade row-by-row.
28


<!-- page 29 -->
B.2.
Stage 2: three-split refinement algorithm and roadmap
B.2.1.
Algorithm
We describe the three-split refinement. The auxiliary sample D1 is used to compute
bΘ0 and the right-factor estimate bΘR below; the second split D2 is used to refine the left factor; the third
split D3 refines the right factor. Write the rank-r factorization
Θ⋆= Θ⋆
L (Θ⋆
R)⊤,
Θ⋆
L ∈Rdt×r, Θ⋆
R ∈Rdm×r, (Θ⋆
R)⊤Θ⋆
R = Ir, 1⊤
dtΘ⋆
L = 0.
This factorization is obtained by absorbing the singular values into the left factor.
Stage A: initialization and right-factor construction. On D1, compute bΘ0 by Theorem B.5. Recenter
bΘ(1) := P⊥bΘ0 where P⊥:= Idt −d−1
t 1dt1⊤
dt is the row-centering projector. Take the rank-r SVD of bΘ(1) and
project the right singular vectors onto the incoherence ball {V ∈Rdm×r : ∥V ∥2,∞≤CRd−1/2
m
} to obtain bΘR.
Stage B: left-factor refinement. On D2, fix bΘR and solve, for each row t ∈[dt], the row score equation
St(θ) = 0, where
St(θ) :=
Mt
X
ℓ=1
bΘR[mℓ]
 
Y (t)
ℓ
−σ
 bΘR[mℓ]⊤θ −bo(t)
ℓ
 	
.
Here (mℓ,m′
ℓ,Y (t)
ℓ
)Mt
ℓ=1 collects all observations in D2 involving row t (after reorienting so row t is the ”left”
model in the comparison), bΘR[m] ∈Rr denotes the m-th row of bΘR, and the opponent offset is bo(t)
ℓ
:= bΘ(1)
t,m′
ℓ.
The solution bθt yields bΘL := (bθt)t∈[dt] ∈Rdt×r.
Stage C: re-centering and right-factor refinement. Set ΘL := P⊥bΘL, restoring the gauge 1⊤
dtΘL = 0
without changing pairwise differences (Lemma B.16 below). On D3, for each column m ∈[dm], solve Sm(a) = 0
where
Sm(a) :=
X
i∈Im
xi
 
Yi −σ(x⊤
i a)
	
,
xi := ΘL[ti,mi] −ΘL[ti,m′
i],
where Im := {i ∈D3 : mi = m or m′
i = m}, and ΘL[t,m] denotes the row of ΘL at index t (which equals the
t-th row of the recentered left factor). Strictly speaking, the inner product is between ΘL’s row at task ti
considered as an element of Rr, and the column-factor parameter a ∈Rr; see Appendix B.7 for the precise
definition. The solution bam yields bΘR := (bam)m∈[dm] ∈Rdm×r.
Final estimator. bΘ := ΘL bΘ⊤
R ∈Rdt×dm.
B.2.2.
Roadmap
The proof proceeds in six blocks paralleling Li et al. [2026]. Each block is the matrix
specialization of the corresponding tensor result.
Block I: Frobenius-error transfer to the right factor. On D1, the SVD-and-incoherence-projection step transfers
the Frobenius error of the convex initializer to the right factor:
bΘR −Θ⋆
R

F ≲
bΘ(1) −Θ⋆
F /σr(Θ⋆) ≲
∆F/ ¯d, where ∆F :=
bΘ(1) −Θ⋆
F ≲
p ¯d 3/n1 on the convex-initialization event. Moreover the projection
step ensures
bΘR

2,∞≲d−1/2
m
and (bΘR)⊤bΘR = (Ir + O(∆F/ ¯d)) on E1.
Block II: Deterministic local existence for the row update. We prove an inward-pointing zero-existence lemma for
St (Lemma B.8) which states that under a sufficient condition on a ”noise plus bias plus higher-order”
combination Rt ≤λ2
t /(4L3γt) for a suitable curvature λt, the score equation has a solution close to the
truth.
29


<!-- page 30 -->
Block III: Probabilistic verification of the row condition. On D2, we verify the sufficient condition uniformly over
t ∈[dt] using six concentration lemmas: row-wise coverage, noise envelope, two bias terms, higher-order
moments βt,γt, and curvature λt. This gives
bΘL −Θ⋆
L

2,∞≲dm/√n2 polylog(n ¯d).
Block IV: Re-centering and pairwise Gram identity. The centering projection P⊥preserves pairwise differences
and preserves entrywise / Frobenius distance (Lemma B.16). A pairwise Gram identity (Lemma B.17)
then upgrades the column-update curvature to λm ≳n3 by capturing the gain from sampling pairs on
the row factor.
Block V: Deterministic local existence for the column update. The same Brouwer argument (Lemma B.19) gives
a sufficient condition for the column score equation Sm(a) = 0 to have a solution near the truth.
Block VI: Probabilistic verification of the column condition. We verify uniformly in m ∈[dm] using parallel con-
centration lemmas, obtaining
bΘR −Θ⋆
R

2,∞≲n−1/2 polylog(n ¯d).
B.3.
Block I: Brouwer inward-pointing zero lemma
We restate the deterministic existence lemma underlying Blocks II and V.
Lemma B.7 (Brouwer inward-pointing zero). Let F : Rr →Rr be continuous, fix ϑ⋆∈Rr and ξ > 0.
If (ϑ −ϑ⋆)⊤F(ϑ) ≤0 for every ϑ on the sphere {ϑ : ∥ϑ −ϑ⋆∥= ξ}, then there exists eϑ with F(eϑ) = 0 and
eϑ −ϑ⋆ ≤ξ.
Proof.
Suppose for contradiction that F has no zero in the closed ball Bξ(ϑ⋆). Define the continuous
map G : Bξ(ϑ⋆) →Bξ(ϑ⋆) by G(ϑ) := ϑ⋆+ ξF(ϑ)/∥F(ϑ)∥, which lands on the sphere. By Brouwer’s fixed-
point theorem, G has a fixed point ϑ†, which satisfies ϑ† −ϑ⋆= ξF(ϑ†)/∥F(ϑ†)∥, and so (ϑ† −ϑ⋆)⊤F(ϑ†) =
ξ ∥F(ϑ†)∥> 0, contradicting the inward-pointing hypothesis on the sphere.
B.4.
Block II: row-wise score equation and deterministic existence
Fix t ∈[dt] and condition on D1. After reorienting the comparisons in D2 so that row t appears on the ”left”
of every comparison (swapping signs of Y when row t was on the right), let (mℓ,m′
ℓ,Yℓ)Mt
ℓ=1 denote the relevant
observations, where Mt := |{i ∈D2 : ti = t}|. Define the true predictors η⋆
ℓ:= Θ⋆
t,mℓ−Θ⋆
t,m′
ℓ= (Θ⋆
R[mℓ])⊤θ⋆
t −o⋆
ℓ
with θ⋆
t ∈Rr the t-th row of Θ⋆
L and o⋆
ℓ:= Θ⋆
t,m′
ℓthe opponent offset. Set boℓ:= bΘ(1)
t,m′
ℓ, and εℓ:= Yℓ−σ(η⋆
ℓ) the
centered Bernoulli noise.
Quantities used in the existence condition. Define the noise vector
Nt :=
Mt
X
ℓ=1
bΘR[mℓ]εℓ,
the Hessian-like matrix
Ht :=
Mt
X
ℓ=1
σ′(η⋆
ℓ) bΘR[mℓ]bΘR[mℓ]⊤∈Rr×r,
with smallest eigenvalue λt := λmin(Ht), the bias vectors
B(R)
t
:=
Mt
X
ℓ=1
σ′(η⋆
ℓ) bΘR[mℓ]
 bΘR[mℓ] −Θ⋆
R[mℓ]
 ⊤θ⋆
t ,
B(O)
t
:=
Mt
X
ℓ=1
σ′(η⋆
ℓ) bΘR[mℓ]
 boℓ−o⋆
ℓ
 
,
30


<!-- page 31 -->
the linearization residuals
dℓ:=
 bΘR[mℓ] −Θ⋆
R[mℓ]
 ⊤θ⋆
t −
 boℓ−o⋆
ℓ
 
,
and the higher-order moments
βt := sup
∥v∥=1
Mt
X
ℓ=1
|bΘR[mℓ]⊤v|d2
ℓ,
γt := sup
∥v∥=1
Mt
X
ℓ=1
|bΘR[mℓ]⊤v|3.
Lemma B.8 (Row-wise deterministic existence). Suppose
Rt := ∥Nt∥+
B(R)
t
 +
B(O)
t
 + L3 βt ≤
λ 2
t
4L3 γt
,
(B.7)
where L3 := supx |σ′′(x)| ≤1/(6
√
3) ≤1. Then St admits a zero bθt with
bθt −θ⋆
t
 ≤2Rt/λt.
Proof.
Set δ := θ −θ⋆
t . By definition,
bΘR[mℓ]⊤θ −boℓ= η⋆
ℓ+ bΘR[mℓ]⊤δ + dℓ.
Substituting into St(θ) and Taylor-expanding σ around η⋆
ℓ,
St(θ⋆
t + δ) = Nt −Ht δ −B(R)
t
+ B(O)
t
−Rt(δ),
where the second-order term is
Rt(δ) := 1
2
Mt
X
ℓ=1
bΘR[mℓ]σ′′(eηℓ)
 bΘR[mℓ]⊤δ + dℓ
 2.
For ∥δ∥= ξ, left-multiplying by δ⊤gives
δ⊤St(θ⋆
t + δ) ≤ξ ∥Nt∥−λtξ2 + ξ
B(R)
t
 + ξ
B(O)
t
 + |δ⊤Rt(δ)|.
Using |σ′′| ≤L3 and the elementary inequality (x + y)2 ≤2x2 + 2y2, the second-order remainder is bounded
by
|δ⊤Rt(δ)| ≤L3
2
Mt
X
ℓ=1
|bΘR[mℓ]⊤δ|(bΘR[mℓ]⊤δ + dℓ)2 ≤L3(γtξ3 + βtξ).
Hence
δ⊤St(θ⋆
t + δ) ≤−λtξ2 + L3γtξ3 + Rtξ.
Choose ξ = ξt := 2Rt/λt. Then −λtξ2 + Rtξ = ξ(Rt −λtξ) = −Rtξ ≤0, and L3γtξ3 ≤L3γt ξ (2Rt/λt)2 ≤Rtξ
under the hypothesis (B.7). Combining gives δ⊤St(θ⋆
t + δ) ≤0 for every ∥δ∥= ξt, so Lemma B.7 delivers a
zero bθt with
bθt −θ⋆
t
 ≤ξt = 2Rt/λt.
B.5.
Block III: probabilistic verification of the row condition
We work conditional on D1, on the event Einit of Theorem B.5, and verify the sufficient condition (B.7)
uniformly over t ∈[dt]. The six lemmas below mirror [Li et al., 2026, Lemmas 3–8] in the matrix case.
Throughout, n2 := |D2| ≍n.
Lemma B.9 (Row-wise coverage). Let
Mt := P
i∈D2 1{ti = t}.
Under
Assumption
A.3,
Mt ∼
Bin(n2,νt) with νt ≍1/dt. For any a > 0, with probability at least 1−n−a, Mt ≍n2/dt uniformly over t ∈[dt],
provided n2 ≳adt log n.
31


<!-- page 32 -->
Proof.
Apply Bernstein’s inequality to the centered indicator 1{ti = t} −νt with variance proxy νt(1 −
νt) ≤νt and envelope 1: for any x > 0,
Pr
 
|Mt −n2νt| ≥
√
2n2νtx + x
 
≤2e−x.
Setting x = Calog(n ¯d) makes the right-hand side ≤n−a−1; a union bound over t ∈[dt] yields the claim. The
condition n2 ≳adt log n ensures the additive x term is dominated by the variance term.
Lemma B.10 (Row noise envelope). For each t ∈[dt], conditional on D1 on Einit and for any x > 0,
Pr
h
∥Nt∥≥C
 √n2x
dm
+
x
√dm
      D1
i
≤2e−x.
Consequently, with probability at least 1 −n−a,
max
t∈[dt]∥Nt∥≤C
 p
n2 log(n ¯d)
dm
+ log(n ¯d)
√dm
 
≲
√n2
dm
polylog(n ¯d).
Proof.
Write Nt = P
i∈D2 1{ti = t} bΘR[mi]εi where εi := Yi −σ(η⋆
i ) is the centered Bernoulli noise. Con-
ditional on D1, the summands are independent, mean zero, with envelope
1{ti = t} bΘR[mi]εi
 ≤
bΘR

2,∞≤Cd−1/2
m
on Einit (incoherence of bΘR) since |εi| ≤1. The variance proxy satisfies
X
i∈D2
E
1{ti = t} bΘR[mi]εi

2
≤n2 · νt · d−1
m ·
bΘR

2
2,∞(dm) ≍
n2
dt dm
· dm · d−1
m =
n2
dtdm
,
where the bound uses E
bΘR[mi]

2
≤
bΘR

2
2,∞≤C/dm under uniform model sampling and E|ε|2 ≤1/4.
Thus, with the standard vector Bernstein form (e.g. Tropp 2015 Theorem 6.1.1 in the rectangular case),
Pr
 
∥Nt∥≥
p
2n2x/(dtdm) + (C/
√
dm)x
 
≤2e−x.
Setting x = Calog(n ¯d) and noting dtdm ≍dtdm ≍d2
m in the balanced regime, then taking a union bound over
t ∈[dt], gives the claim. The dominant √n2/dm term arises from the variance scale.
Lemma B.11 (Right-factor bias). Let ∆R :=
bΘR −Θ⋆
R

F ≤C∆F/ ¯d on Einit (Block I). With probability
at least 1 −n−a,
max
t∈[dt]
B(R)
t
 ≤C
 n2 ∆R
d3/2
m
+
p
n2 log(n ¯d)∆R
dm
+ ∆R log(n ¯d)
 
≲
√n2
dm
polylog(n ¯d).
Proof.
Decompose B(R)
t
= E[B(R)
t
| D1] + (B(R)
t
−E[B(R)
t
| D1]).
Conditional expectation. With qm,t := (bΘR[m] −Θ⋆
R[m])⊤θ⋆
t ,
E[B(R)
t
| D1] = n2 · νt · Em∼πt[σ′(η⋆
t,m,m′)bΘR[m]qm,t] ≍
n2
dtd2
m
dm
X
m=1
¯ct,m bΘR[m]qm,t,
where ¯ct,m ≤CB is bounded. By Cauchy–Schwarz and P
m
bΘR[m]

2
≤r,
E[B(R)
t
| D1]
 ≤C n2
dtd2
m
√r
 X
m
q2
m,t
 1/2
≤C
n2
dtd3/2
m
∆R ∥θ⋆
t ∥2 ,
32


<!-- page 33 -->
using P
m q2
m,t =
(bΘR −Θ⋆
R)θ⋆
t

2
≤
bΘR −Θ⋆
R

2
F ∥θ⋆
t ∥2 and ∥θ⋆
t ∥≤∥Θ⋆
L∥2,∞≤C
√
dt. Hence
E[B(R)
t
| D1]
 ≤
Cn2∆R/d3/2
m .
Fluctuation. The centered summands Z(R)
i,t
have envelope
Z(R)
i,t
 ≤1{ti = t}CB
bΘR

2,∞· |qmi,t| ≤C d−1/2
m
· ∆R
√
dt ≤C∆R
(since
√
dt/
√
dm = O(1) in the balanced regime), and variance proxy P
i E
Z(R)
i,t

2
≤n2νt · Em[∆2
R/dm] ≍
n2∆2
R/(dtdm) ≍n2∆2
R/d2
m. Vector Bernstein gives, for any x > 0,
B(R)
t
−E[B(R)
t
| D1]
 ≤C(√n2x∆R/dm +
∆Rx) with probability at least 1 −2e−x. Setting x = Calog(n ¯d) and union-bounding over t ∈[dt] gives the
fluctuation bound.
Combining. On Einit, ∆R ≲∆F/dm ≲
p ¯d 3/n1/dm ≍
p ¯d/n1, so n2∆R/d3/2
m ≲n2
p ¯d/n1/d3/2
m ≍√n2/dm
under n1 ≍n2. This gives the stated rate.
Lemma B.12 (Offset bias). With probability at least 1 −n−a,
max
t∈[dt]
B(O)
t
 ≲
√n2
dm
polylog(n ¯d).
Proof.
Identical to Lemma B.11 with the right-factor error bΘR −Θ⋆
R replaced by the entrywise error
bΘ(1) −Θ⋆. The relevant Frobenius bound is
bΘ(1) −Θ⋆
F ≤∆F, and the clipping ensures |boℓ−o⋆
ℓ| ≤2B
entrywise. The conditional expectation is bounded by Cn2∆F/d5/2
m , and the fluctuation by Bernstein with
the same envelope/variance scaling. Substituting ∆2
F ≍¯d 3/n1 gives the stated rate.
Lemma B.13 (Higher-order moments βt,γt). With probability at least 1 −n−a,
max
t
βt ≲
1
√
dm
polylog(n ¯d),
max
t
γt ≲
n2
d5/2
m
polylog(n ¯d).
Proof.
Bound for βt. Using (x + y)2 ≤2x2 + 2y2, we split βt ≤2β(R)
t
+ 2β(O)
t
where β(R)
t
uses the right-
factor part and β(O)
t
the offset part. Since |bΘR[m]⊤v| ≤
bΘR

2,∞≤Cd−1/2
m
, the conditional expectation of
β(R)
t
is bounded by Cn2/(dtd3/2
m )P
m q2
m,t ≤Cn2∆2
R/d1/2
m , and similarly Eβ(O)
t
≤Cn2∆2
F/d7/2
m . Substituting
the rates ∆2
R ≲∆2
F/d2
m and ∆2
F ≍¯d 3/n1, we obtain Eβt ≲n2 ¯d/(n1d1/2
m ) ≍
p
1/dm (the balanced-regime
simplification). Bernstein on the fluctuation gives the stated rate uniformly via union bound.
Bound for γt. |bΘR[m]⊤v|3 ≤
bΘR

3
2,∞≤Cd−3/2
m
, and Mt ≍n2/dt ≍n2/dm (balanced regime), so γt ≤
Mt · Cd−3/2
m
≤Cn2/d5/2
m
deterministically on Einit and the row-coverage event.
Lemma B.14 (Row curvature). With probability at least 1 −n−a, mint∈[dt] λt ≥cn2/d2
m for an absolute
constant c > 0.
Proof.
The conditional expectation of the Hessian is
E[Ht | D1] = n2νt Em∼πt[σ′(η⋆
t,m,m′)bΘR[m]bΘR[m]⊤] ⪰cB n2
dtdm
(bΘR)⊤bΘR.
On Einit, (bΘR)⊤bΘR ⪰c0Ir (Block I), so E[Ht | D1] ⪰c · n2/d2
m Ir in the balanced regime. Each summand has
operator norm at most
bΘR

2
2,∞≤C/dm, so the matrix variance proxy is at most (C/dm) · ∥E[Ht | D1]∥op ≤
Cn2/d3
m. Matrix Bernstein then gives, for any x > 0,
Pr
 
∥Ht −E[Ht | D1]∥op ≥
p
Cn2x/d3
m + (C/dm)x
 
≤2r e−x.
Setting x = Calog(n ¯d) makes ∥Ht −E[Ht | D1]∥op = o(n2/d2
m) under n2 ≳dm logc(n ¯d), and a union bound
over t ∈[dt] gives the uniform statement. Hence λt ≥λmin(E[Ht | D1]) −∥Ht −E[Ht | D1]∥op ≥cn2/d2
m.
33


<!-- page 34 -->
Proposition B.15 (Uniform left-factor ℓ2,∞bound). On Einit, with probability at least 1 −n−a, the
row score equations admit solutions bθt, t ∈[dt], with
bΘL −Θ⋆
L

2,∞= max
t∈[dt]
bθt −θ⋆
t
 ≤C dm
√n2
polylog(n ¯d).
Proof.
Combine Lemmas B.10–B.14:
Rt = ∥Nt∥+
B(R)
t
 +
B(O)
t
 + L3βt ≲
√n2
dm
polylog(n ¯d),
and
λ2
t
γt
≳(n2/d2
m)2
n2/d5/2
m
= n2
d3/2
m
.
Hence the sufficient condition (B.7) Rt ≤λ2
t /(4L3γt) holds when √n2/dm · polylog(n ¯d) ≲n2/d3/2
m , i.e. n2 ≳
dm polylog(n ¯d) = ¯dpolylog(n ¯d). On this event, Lemma B.8 yields, uniformly in t,
bθt −θ⋆
t
 ≤2Rt
λt
≲d2
m
n2
·
√n2
dm
polylog(n ¯d) = dm
√n2
polylog(n ¯d).
Probability calibration to 1 −n−a follows from Appendix A.8.
B.6.
Block IV: re-centering and pairwise Gram identity
The recentering step is essential to upgrade the column-update curvature in Block VI. Without it, the row
factor bΘL generally does not satisfy the row-centering gauge, and the pairwise Gram identity of Lemma B.17
below fails.
Lemma B.16 (Recentering preserves pairwise differences). Let ΘL := P⊥bΘL, where P⊥:= Idt −
d−1
t 1dt1⊤
dt. Then for every u,v ∈[dt], ΘL[u] −ΘL[v] = bΘL[u] −bΘL[v],
ΘL −Θ⋆
L

2,∞≤2
bΘL −Θ⋆
L

2,∞,
ΘL −Θ⋆
L

F ≤
bΘL −Θ⋆
L

F .
Proof.
Since Θ⋆
L satisfies 1⊤
dtΘ⋆
L = 0, we have Θ⋆
L = P⊥Θ⋆
L and so ΘL −Θ⋆
L = P⊥(bΘL −Θ⋆
L). For pairwise
differences, the centering subtraction is constant in u,v, so it cancels: ΘL[u]−ΘL[v] = bΘL[u]−bΘL[v] exactly.
For the row-norm, with E := bΘL−Θ⋆
L, (P⊥E)u = Eu−¯E where ¯E := d−1
t
P
v Ev, so ∥(P⊥E)u∥≤∥Eu∥+
 ¯E
 ≤
2∥E∥2,∞. The Frobenius bound follows from P⊥being an orthogonal projection.
Lemma B.17 (Pairwise Gram identity). For Θ ∈Rdt×r with 1⊤
dtΘ = 0,
X
1≤u<v≤dt
(Θ[u] −Θ[v])(Θ[u] −Θ[v])⊤= dt Θ⊤Θ.
Consequently, under uniform pair sampling on [dt]2,
E
 
(Θ[U] −Θ[V ])(Θ[U] −Θ[V ])⊤   Θ
 
=
2
dt −1 Θ⊤Θ.
Proof.
Compute the ordered sum
dt
X
u=1
dt
X
v=1
(Θ[u] −Θ[v])(Θ[u] −Θ[v])⊤= 2dt Θ⊤Θ −2
 X
u
Θ[u]
  X
u
Θ[u]
 ⊤
.
The second term vanishes by centering. The unordered sum is half the ordered sum, giving the first identity.
Dividing by
 dt
2
 
yields the conditional expectation.
34


<!-- page 35 -->
Corollary B.18 (Gram lower bound for ΘL). On Einit and the event of Proposition B.15, if ϵL :=
ΘL −Θ⋆
L

2,∞≤cΘ
√
dt for a sufficiently small constant cΘ, then
λmin(Θ
⊤
LΘL) ≥cd2
t
in the balanced regime where σr(Θ⋆) ≍¯d.
Proof.
ΘL −Θ⋆
L

F ≤
√
dt ϵL by the row-norm-to-Frobenius transition. The operator-norm perturbation
Θ
⊤
LΘL −(Θ⋆
L)⊤Θ⋆
L

op ≤2∥Θ⋆
L∥op
ΘL −Θ⋆
L

F +
ΘL −Θ⋆
L
2
F is bounded by Cd3/2
t
ϵL + Cdtϵ2
L. Meanwhile,
λmin((Θ⋆
L)⊤Θ⋆
L) = σr(Θ⋆)2 ≍¯d 2 ≍d2
t in the balanced regime under Assumption A.7. For ϵL ≤cΘ
√
dt suffi-
ciently small, the perturbation is dominated by half the leading eigenvalue, giving the bound.
B.7.
Blocks V–VI: column-wise refinement
Set n3 := |D3| ≍n. For each column m ∈[dm], let
Im := {i ∈D3 : mi = m or m′
i = m},
Mm := |Im|.
After reorientation, write xi := ΘL[ti,Ui] −ΘL[ti,Vi] where (Ui,Vi) is the reoriented pair (so column m is
associated with the ”left” entry of the score equation), and x⋆
i := Θ⋆
L[ti,Ui]−Θ⋆
L[ti,Vi], with the linearization
residual hi := xi −x⋆
i . The column score equation is Sm(a) = P
i∈Im xi{Yi −σ(x⊤
i a)}. Define
Nm :=
X
i∈Im
xi εi,
Hm :=
X
i∈Im
σ′((x⋆
i )⊤a⋆
m)xix⊤
i ,
λm := λmin(Hm),
B(L)
m :=
X
i∈Im
σ′((x⋆
i )⊤a⋆
m)xi (h⊤
i a⋆
m),
βm,γm defined analogously to Block III.
Lemma B.19 (Column deterministic existence). If Rm := ∥Nm∥+
B(L)
m
 + L3βm ≤λ 2
m/(4L3γm),
then Sm admits a zero bam with ∥bam −a⋆
m∥≤2Rm/λm.
Proof.
Identical structure to Lemma B.8. Set δ := a−a⋆
m. Then x⊤
i a = (x⋆
i )⊤a⋆
m+x⊤
i δ+h⊤
i a⋆
m, and Taylor-
expanding σ around (x⋆
i )⊤a⋆
m, Sm(a⋆
m + δ) = Nm −Hmδ −B(L)
m −Rm(δ), where Rm(δ) is the second-order
term. The same inward-pointing argument as Lemma B.8 gives the conclusion.
Lemma B.20 (Column-wise coverage). Under Assumption A.3, with probability at least 1 −n−a,
Mm ≍n3/dm uniformly over m ∈[dm], provided n3 ≳adm log n.
Proof.
Each observation involves column m iff one of the two sampled columns equals m, which happens
with probability 2/dm under uniform pair sampling and within the constants of Assumption A.3. Bernstein
with union bound over m ∈[dm] gives the claim.
Lemma B.21 (Column noise envelope). On the event of Proposition B.15, with probability at least
1 −n−a,
max
m∈[dm]∥Nm∥≲√n3 polylog(n ¯d).
Proof.
The summands of Nm are 1{i ∈Im}xiεi, with envelope ∥xi∥≤2
ΘL

2,∞≲
√
dt (by Lemma B.16
and incoherence) and variance proxy P
i E∥xiεi1{i ∈Im}∥2 ≤n3 ·(2/dm)·dt ·1 = O(n3dt/dm) ≍n3 (balanced
regime). Vector Bernstein with x = Calog(n ¯d) and union bound over m gives the bound. Specifically, ∥Nm∥≤
C(√n3x +
√
dt x) ≤C√n3 polylog(n ¯d).
35


<!-- page 36 -->
Lemma B.22 (Column curvature). Assume the event of Proposition B.15 and let ϵL :=
ΘL −Θ⋆
L

2,∞.
If ϵL ≤cΘ
√
dt, with probability at least 1 −n−a, minm∈[dm] λm ≥cn3.
Proof.
The conditional expectation is
E[Hm | D1,D2] = n3 · Pr{i ∈Im} · E
 
σ′((x⋆
i )⊤a⋆
m)xix⊤
i
   i ∈Im
 
⪰cB
n3
dm
E[xix⊤
i ].
By Lemma B.17 applied to ΘL, E[xix⊤
i ] =
2
dt−1 Θ
⊤
LΘL. By Corollary B.18, λmin(Θ
⊤
LΘL) ≥cd2
t, so E[xix⊤
i ] ⪰
cdt Ir. Combining, E[Hm] ⪰c′ n3 (dt/dm)Ir ≍c′ n3 Ir in the balanced regime. Each summand has oper-
ator norm ≤∥xi∥2 ≤Cdt, and the matrix variance proxy is O(n3dt). Matrix Bernstein then gives
∥Hm −E[Hm]∥op ≤C
p
n3dt log(n ¯d) + Cdt log(n ¯d), which is o(n3) under n3 ≳dt logc(n ¯d); union bound over
m ∈[dm] gives the uniform statement.
Lemma B.23 (Column bias from left-factor error). With probability at least 1 −n−a,
max
m∈[dm]
B(L)
m
 ≲n3 ϵL
dt
polylog(n ¯d).
Proof.
∥hi∥≤2ϵL by Lemma B.16, and ∥a⋆
m∥≤∥Θ⋆
R∥2,∞≤Cd−1/2
m
by incoherence of the right factor,
so |h⊤
i a⋆
m| ≤CϵL/
√
dm. Combined with ∥xi∥≲
√
dt, each summand has envelope ≲ϵL
√
dt/
√
dm ≍ϵL (bal-
anced regime). The conditional expectation is bounded by Cn3/dm · ϵL = Cn3ϵL/dm ≍Cn3ϵL/dt (balanced).
Variance proxy is O(n3ϵ2
L/dm) ≍O(n3ϵ2
L/dt). Vector Bernstein with union bound gives the rate.
Lemma B.24 (Column βm,γm). With probability at least 1 −n−a,
max
m βm ≲n3ϵ2
L
d3/2
t
polylog(n ¯d),
max
m γm ≲n3
√
dt polylog(n ¯d).
Proof.
For βm: |x⊤
i v| ≤∥xi∥≲
√
dt and (h⊤
i a⋆
m)2 ≤Cϵ2
L/dm; the sum has Mm ≍n3/dm terms. So βm ≤
Mm ·
√
dt · ϵ2
L/dm ≍n3ϵ2
L/d3/2
t
in the balanced regime.
For γm: |x⊤
i v|3 ≤d3/2
t
, so γm ≤Mm · Cd3/2
t
≤Cn3
√
dt in the balanced regime.
Proposition B.25 (Uniform right-factor ℓ2,∞bound). Under the conditions of Proposition B.15
and n3 ≳¯dlogc(n ¯d), with probability at least 1 −n−a,
bΘR −Θ⋆
R

2,∞≤C
 
1
√n3
+ ϵL
dt
+ ϵ2
L
d3/2
t
 
polylog(n ¯d) ≤
C
√n polylog(n ¯d).
Proof.
By Lemma B.19 it suffices to verify Rm ≤λ2
m/(4L3γm) uniformly. Combining Lemmas B.21–
B.24, Rm ≲(√n3 + n3ϵL/dt)polylog(n ¯d), and λ2
m/γm ≳n2
3/(n3
√
dt) = n3/
√
dt. The condition holds when
(√n3 + n3ϵL/dt)polylog ≲n3/
√
dt, i.e. n3 ≳dt polylog(n ¯d), which is the sample-size assumption. Then
∥bam −a⋆
m∥≤2Rm
λm
≲1
n3
 √n3 + n3ϵL
dt
+ n3ϵ2
L
d3/2
t
 
polylog =
 
1
√n3
+ ϵL
dt
+ ϵ2
L
d3/2
t
 
polylog.
Substituting ϵL ≲dm/√n2 polylog ≍dt/√n2 polylog (balanced regime) gives ϵL/dt ≲1/√n2 ≍1/√n, and the
third term is even smaller, so ∥bam −a⋆
m∥≲1/√npolylog(n ¯d).
36


<!-- page 37 -->
B.8.
Final assembly: proof of Theorem 3.1
Theorem B.26 (Entrywise refinement; restatement of Theorem 3.1). Under the assumptions of
Section 2 and Assumptions A.1–A.11, fix any a > 0 and let n ≥C poly(µ,r,κ,B) ¯dlogc(n ¯d). Then with
probability at least 1 −n−a,
bΘ −Θ⋆
∞≤C
r ¯dpolylog(n ¯d)
n
.
Proof.
On Einit ∩Erefine (the intersection of the convex-initialization and refinement events), apply the
factorization
bΘ −Θ⋆= (ΘL −Θ⋆
L)(Θ⋆
R)⊤+ ΘL (bΘR −Θ⋆
R)⊤.
The entrywise norm is bounded by
bΘ −Θ⋆
∞≤
ΘL −Θ⋆
L

2,∞· ∥Θ⋆
R∥2,∞+
bΘR −Θ⋆
R

2,∞·
ΘL

2,∞.
Substituting
ΘL −Θ⋆
L

2,∞≲dm/√n2 polylog (Proposition B.15 via Lemma B.16),
bΘR −Θ⋆
R

2,∞≲
1/√n3 polylog (Proposition B.25), ∥Θ⋆
R∥2,∞≤Cd−1/2
m
(incoherence), and
ΘL

2,∞≤2∥Θ⋆
L∥2,∞≤Cd1/2
t
(incoherence + recentering bound) yields
bΘ −Θ⋆
∞≲dm
√n ·
1
√
dm
+ 1
√n ·
√
dt =
r
dm
n +
r
dt
n ≍
r ¯d
n polylog(n ¯d).
The probability calibration to 1 −n−a follows from Appendix A.8.
This completes the proof of Theorem 3.1.
Q.E.D.
Appendix C:
Proof of Proposition 3.2: top-K Hamming and exact recovery
This appendix proves the deterministic reduction from entrywise score estimation to the taskwise top-K
Hamming bound, establishes exact recovery under the margin condition, and discusses minimax optimality.
C.1.
Setup
For task t ∈[dt], abbreviate θm := Θ⋆
t,m, bθm := bΘt,m, and let St := S⋆
K(t), bSt := bSK(t). Let θ(1) ≥θ(2) ≥··· ≥
θ(dm) denote the sorted true scores for task t, and define the K-boundary midpoint τK(t) := (θ(K) +θ(K+1))/2.
The normalized top-K Hamming loss is HamK,t := (2K)−1|bSt△St|, and the boundary mass at radius r is
RK,t(r;Θ⋆) := (2K)−1|{m : |θm −τK(t)| ≤r}|. On the event E∞:= {
bΘ −Θ⋆
∞≤εn} of Theorem 3.1, all
statements below are deterministic.
C.2.
Hamming bound from entrywise error
Proposition C.1 (Restatement of Proposition 3.2). On
E∞,
for
every
t ∈[dt],
HamK,t ≤
RK,t(2εn;Θ⋆). Hence if Pr(E∞) ≥1 −n−a, the Hamming bound holds simultaneously over all tasks with
probability at least 1 −n−a.
Proof.
We prove the deterministic inclusion bSt△St ⊆{m : |θm −τK(t)| ≤2εn} on E∞; the cardinality
bound then follows by dividing by 2K. Take a false positive u ∈bSt \ St; since |bSt| = |St| = K, there exists a
false negative v ∈St \ bSt. Because u ∈bSt and v /∈bSt, bθu ≥bθv under the deterministic tie-breaking rule. Hence
θu ≥bθu −εn ≥bθv −εn ≥θv −2εn. Since v ∈St, θv ≥θ(K), so θu ≥θ(K) −2εn; and since u /∈St, θu ≤θ(K+1).
Combining, θ(K) −2εn ≤θu ≤θ(K+1), and using τK(t) ∈[θ(K+1),θ(K)], |θu −τK(t)| ≤2εn. The false-negative
case is symmetric.
37


<!-- page 38 -->
C.3.
Exact recovery under a margin condition
Corollary C.2 (Exact top-K recovery). Define ∆K(t) := θ(K) −θ(K+1). If ∆K(t) > 4εn, then on E∞
the boundary mass RK,t(2εn;Θ⋆) = 0, hence bSt = St. In particular, if mint∈[dt] ∆K(t) > 4εn, then exact top-K
recovery holds simultaneously over all tasks with probability at least 1 −n−a.
Proof.
Under ∆K(t) > 4εn, every m ∈St has θm −τK(t) ≥∆K(t)/2 > 2εn, and symmetrically for m /∈St;
so RK,t(2εn;Θ⋆) = 0, and Proposition C.1 applies. The simultaneous statement follows by union bound over
t ∈[dt].
C.4.
Minimax-optimality remark
The proposition is a deterministic reduction from entrywise score estimation to top-K Hamming accuracy.
For minimax optimality we appeal to the single-task BTL ranking literature: Chen et al. [2022] characterize
the minimax rate for normalized Hamming partial recovery in BTL top-K ranking and show that MLE
attains both partial and exact recovery thresholds, while Chen et al. [2019] establish the minimax sample
complexity for exact top-K identification through entrywise score control combined with a K-versus-(K +1)
margin condition. Our entrywise rate εn ≍
p ¯dpolylog(n ¯d)/n and the exact recovery margin 4εn match these
single-task minimax characterizations up to logarithmic factors. The gain from low-rank structure is the
factor ¯d instead of ¯d 2 in the per-task sample complexity; the dependence on dt is only through the union
bound and is logarithmic.
Appendix D:
Proof of Theorem 4.1: finite-dimensional inference and efficiency
This appendix proves Theorem 4.1 of Section 4 together with the efficiency claim referenced in the discussion
of Veff(Γ). Throughout we work under the assumptions of Appendix A. The structure of the appendix is as
follows.
• Appendix D.1 restates the one-step estimator algorithm in matrix form (essentially a copy of Section 4
for self-containment).
• Appendix D.2 proves the scalar (1-dimensional) semiparametric efficiency lower bound by an information
inequality. This is the matrix specialization of [Li et al., 2026, Theorem 3.3].
• Appendix D.3 extends the lower bound to the joint Loewner inequality ¯Σ ⪰Σ for fixed q = O(1) contrasts
via reduction to the scalar bound applied at every linear combination Γu = P
j ujΓj, u ∈Rq.
• Appendix D.4 writes out the closed-form matrix tangent projector PT = PU⋆⊗I +P1⊥⊗PV ⋆−PU⋆⊗PV ⋆
that we use throughout.
• Appendix D.5 states the exact six-term decomposition of bψΓ −ψΓ(Θ⋆) into the leading i.i.d. EIF average
plus six remainders, following the notation of Li et al. [2026].
• Appendix D.6 ports the single-contrast remainder bound |RΓ
n| ≤C CA ∥Γ∥1 ¯dlogc ¯d/n from the combined-
error theorem in Li et al. [2026], with each of the six terms RH
emp,Rη
emp,Rproj,RHbias,R1st,R2nd explicitly
bounded.
38


<!-- page 39 -->
• Appendix D.7 extends the single-contrast bound to a uniform statement over a polynomial-size contrast
family by union bound, with the per-term envelope/variance accounting carried out explicitly. This is
the input to Appendix E.
• Appendix D.8 proves the Berry–Esseen rate ρn ≲
p ¯d/n for the standardized leading term, with full
computation of the second and third moments.
• Appendix D.9 combines D.7 and D.9 with the rectangle-band transfer to obtain the multivariate rectangle
CLT proving Theorem 4.1.
• Appendices D.10 and D.11 prove the relative variance consistency and the covariance consistency in
correlation form, respectively, both of which are inputs to Appendix E.
D.1.
The one-step estimator and its plug-in operators
Recall the cross-fitting one-step procedure in Section 4. For clarity in the matrix case, we restate it with-
out cross-fitting notation: given the auxiliary-sample initializer (bΘ, bPT, bHΓ) of Appendix A.4, the one-step
estimator for ψΓ(Θ⋆) = ⟨Γ, Θ⋆⟩is
bψΓ := ψΓ(bΘ) + 1
n
n
X
i=1
sη
 Yi, bηi
  D
bHΓ, Xi
E
,
bηi :=
D
bΘ, Xi
E
,
(D.1)
where the evaluation sample {(Xi,Yi)}n
i=1 is independent of the auxiliary sample Daux used to compute
(bΘ, bPT, bG, bHΓ). Here bHΓ solves the estimated information equation ( bPT bG bPT) bHΓ = bPTΓ, with bG defined
in (A.2). Cross-fitting extends the analysis with no change to the rates; we suppress fold indices throughout.
For a finite contrast family Γ1,...,Γq, let
ψj := ψΓj(Θ⋆),
bψj := bψΓj,
H⋆
j := A−1PTΓj,
bHj := bA−1 bPTΓj,
ϕj(Wi) := sη(Yi,η⋆
i )

H⋆
j , Xi
 
,
bϕj(Wi) := sη(Yi, bηi)
D
bHj, Xi
E
,
σ2
j := Veff(Γj) = E⋆[ϕ2
j] =

PTΓj, A−1PTΓj
 
,
Σjk := E⋆[ϕjϕk] =

PTΓj, A−1PTΓk
 
,
and the standardized oracle coordinate Zij := ϕj(Wi)/σj, so E⋆Zij = 0 and E⋆Z2
ij = 1.
D.2.
Single-contrast (1D) semiparametric efficiency lower bound
For any fixed contrast Γ ∈Rdt×dm, the semiparametric efficiency bound for any regular estimator bψ of ψΓ(Θ⋆)
is
Var⋆( bψ) ≥1
n Veff(Γ),
Veff(Γ) =

PTΓ, A−1PTΓ
 
.
(D.2)
We give the proof following the standard information-inequality argument; the steps are the matrix special-
ization of [Li et al., 2026, Section G.2].
Theorem D.1 (1D semiparametric efficiency bound). Suppose A is invertible on T, and let bψ be
any locally unbiased estimator of ψΓ under any one-parameter submodel Θε = Θ⋆+εH for H ∈T. Then (D.2)
holds. Moreover, the EIF ϕΓ = sη(Y,η⋆)⟨H⋆
Γ, X⟩with H⋆
Γ = A−1PTΓ attains the bound: E⋆[ϕ2
Γ] = Veff(Γ).
39


<!-- page 40 -->
Proof.
We give the full information inequality argument.
Step 1: differentiating the unbiasedness identity. Fix H ∈T and consider the one-parameter sub-
model Θε = Θ⋆+ εH, with sampling density pΘε,Π⋆(X,Y ) = gΠ⋆(X)p(Y | ⟨X, Θε⟩). Local unbiasedness of bψ
means EΘε[ bψ] = ψΓ(Θε) = ⟨Γ, Θε⟩in a neighborhood of ε = 0. Differentiating both sides at ε = 0 gives
∂ε
  
0EΘε[ bψ] = ⟨Γ, H⟩= ⟨PTΓ, H⟩,
(D.3)
where the second equality uses H ∈T (so (I −PT)Γ is orthogonal to H).
Step 2: the score identity. The directional score along the submodel is ∂ε log pΘε,Π⋆(Xi,Yi)
  
0 =
sη(Yi,η⋆
i )⟨H, Xi⟩(differentiating log p in the parameter ε). By the standard score identity (which holds for
any random variable with finite second moment under regularity),
∂ε
  
0EΘε[ bψ] = E⋆h
bψ
n
X
i=1
sη(Yi,η⋆
i )⟨H, Xi⟩
i
.
(D.4)
Subtracting the constant ψΓ(Θ⋆) (which doesn’t affect the score-product expectation by centering) gives
∂ε
  
0EΘε[ bψ] = E⋆[( bψ −ψΓ)P
i sη(Yi,η⋆
i )⟨H, Xi⟩].
Step 3: Cauchy–Schwarz and Rayleigh-quotient maximization. Combining (D.3) with (D.4) and
applying Cauchy–Schwarz,
⟨PTΓ, H⟩2 ≤Var⋆( bψ) · E⋆h 
n
X
i=1
sη(Yi,η⋆
i )⟨H, Xi⟩
 2i
.
By independence of the n observations and the score-centering identity E⋆[sη(Yi,η⋆
i )⟨H, Xi⟩] = 0,
E⋆h X
i
sη(Yi,η⋆
i )⟨H, Xi⟩
 2i
= nE⋆[sη(Y,η⋆)2 ⟨H, X⟩2] = n ⟨H, AH⟩,
where the last step uses the definition of the operator G in (A.1) and the fact that E⋆[s2
η | X] = I(η⋆). Thus
Var⋆( bψ) ≥1
n
⟨PTΓ, H⟩2
⟨H, AH⟩.
Taking the supremum over H ∈T \ {0} on the right, the Rayleigh quotient is maximized at H = A−1PTΓ =
H⋆
Γ, with maximum value ⟨PTΓ, A−1PTΓ⟩= Veff(Γ) by direct computation: ⟨PTΓ, H⋆
Γ⟩2 /⟨H⋆
Γ, AH⋆
Γ⟩=
Veff(Γ)2/Veff(Γ) = Veff(Γ). This proves (D.2).
Step 4: attainment. For the EIF ϕΓ = sη(Y,η⋆)⟨H⋆
Γ, X⟩,
E⋆[ϕ2
Γ] = E⋆[s2
η ⟨H⋆
Γ, X⟩2] = ⟨H⋆
Γ, GH⋆
Γ⟩= ⟨H⋆
Γ, AH⋆
Γ⟩= Veff(Γ),
using H⋆
Γ ∈T so PTH⋆
Γ = H⋆
Γ and the information equation AH⋆
Γ = PTΓ.
D.3.
Multivariate (Loewner) lower bound by 1D + arbitrary u
For a fixed finite contrast family Γ1,...,Γq (q = O(1)), the candidate efficient covariance is Σjk =
⟨PTΓj, A−1PTΓk⟩.
Proposition D.2 (Joint semiparametric efficiency lower bound). Let
¯ψ = ( ¯ψ1,..., ¯ψq)⊤be any
regular estimator of ψ = (ψ1,...,ψq)⊤, with √n( ¯ψ−ψ) ⇝N(0, ¯Σ) for some covariance ¯Σ ⪰0. Then ¯Σ−Σ ⪰0
in Loewner order; equivalently, u⊤¯Σu ≥u⊤Σu for every u ∈Rq.
40


<!-- page 41 -->
Proof.
We reduce to the scalar bound applied at every linear combination u⊤ψ = ψΓu for the combined
contrast Γu := Pq
j=1 ujΓj.
Step 1: u⊤¯ψ is a regular estimator of ψΓu. Since ¯ψ is regular for the vector ψ, the linear functional
u⊤¯ψ is regular for the corresponding scalar target u⊤ψ = P
j ujψj = P
j uj ⟨Γj, Θ⋆⟩= ⟨Γu, Θ⋆⟩= ψΓu(Θ⋆).
Its asymptotic variance is u⊤¯Σu by the continuous mapping theorem.
Step 2: scalar lower bound. Apply Theorem D.1 to the scalar functional ψΓu: for every regular esti-
mator,
Var⋆(u⊤¯ψ) ≥1
nVeff(Γu).
Taking the asymptotic variance, u⊤¯Σu ≥Veff(Γu) = ⟨PTΓu, A−1PTΓu⟩.
Step 3: bilinearity. By linearity of PT, PTΓu = P
j ujPTΓj, and by bilinearity of ⟨·, A−1·⟩,

PTΓu, A−1PTΓu
 
=
X
j,k
ujuk

PTΓj, A−1PTΓk
 
= u⊤Σu.
Combining steps 2 and 3 gives u⊤(¯Σ −Σ)u ≥0. Since this holds for every u ∈Rq, ¯Σ ⪰Σ in Loewner order.
Remark D.3 (Singular covariance and redundant contrasts). The covariance Σ may be singu-
lar without invalidating the proof. Suppose Γ3 = Γ1 + Γ2 so that ϕ3 = ϕ1 + ϕ2 and the third row of Σ is
the exact sum of the first two. The Loewner bound still holds: the proof above does not require invertibil-
ity, only the scalar efficiency bound for every linear combination u⊤ψ. If u⊤Σu = 0 for some u ̸= 0, then
the corresponding contrast has zero efficient variance; this is an exact local redundancy and produces a
degenerate Gaussian limit on a lower-dimensional subspace. In particular, the inverse A−1 is well-defined on
span(PTΓ1,...,PTΓq), so the proof is unaffected.
D.4.
Closed-form matrix tangent projector
We write out the closed form of PT before proceeding to the decomposition.
Lemma D.4 (Closed-form matrix tangent projector). Let Θ⋆= U ⋆Σ⋆(V ⋆)⊤with U ⋆∈Rdt×r, V ⋆∈
Rdm×r and (U ⋆)⊤1dt = 0 (the row-centering gauge of Section 2 enforces this). Define PU⋆:= U ⋆(U ⋆)⊤, PV ⋆:=
V ⋆(V ⋆)⊤, and P1⊥:= Idt −d−1
t 11⊤. Then
PTΓ = PU⋆Γ + P1⊥ΓPV ⋆−PU⋆ΓPV ⋆,
Γ ∈Rdt×dm.
(D.5)
Equivalently, under standard vectorization, PT = PU⋆⊗I + P1⊥⊗PV ⋆−PU⋆⊗PV ⋆.
Proof.
The (unconstrained) rank-r tangent space at Θ⋆is {U ⋆A⊤+B(V ⋆)⊤: A ∈Rdm×r,B ∈Rdt×r}. The
row-centering identification constraint 1⊤
dtΓ = 0 restricts the B-component to lie in 1⊥
dt. Direct verification of
orthogonality yields the formula (D.5); details follow the general Tucker case in [Li et al., 2026, Lemma F.4].
This closed form makes each remainder term in Appendix D.5 computable in closed form. In particular,
for a sparse score-gap contrast Γ = et(em −em′)⊤, the projection PTΓ is a low-rank matrix whose Frobenius
norm is Θ(
p
1/dt +
p
1/dm), and the leakage component (I −PT)Γ is bounded entrywise by an incoherence-
dependent constant (Lemma A.10).
41


<!-- page 42 -->
D.5.
One-step error decomposition (matrix form)
Let ∆:= bΘ−Θ⋆and write ST(H)(W) := sη(Y,⟨X, T⟩)⟨H, X⟩. Adding and subtracting the oracle correction
term in (D.1),
bψΓ −ψΓ(Θ⋆) =
(Pn −P⋆)ϕΓ
|
{z
}
(I) leading i.i.d. EIF average
+RΓ
n,
(D.6)
where the remainder splits into six interpretable terms, following [Li et al., 2026, Section 4.4]:
RΓ
n = RH
emp + Rη
emp + Rproj + RHbias + R1st + R2nd,
RH
emp := (Pn −P⋆)
 
sη(Y, bη)
D
bHΓ −H⋆
Γ, X
E 
,
Rη
emp := (Pn −P⋆)
  sη(Y, bη) −sη(Y,η⋆)
 
⟨H⋆
Γ, X⟩
 
,
Rproj := ⟨(I −PT)Γ, ∆⟩,
RHbias := P⋆ 
SbΘ( bHΓ) −SbΘ(H⋆
Γ)
 
,
R1st := ⟨PTΓ, ∆⟩+ P⋆ 
SbΘ(H⋆
Γ) −SΘ⋆(H⋆
Γ)
 
,
R2nd := O(∥∆∥2
∞) second-order score remainder of R1st.
The first-order cancellation term R1st collects the ⟨PTΓ, ∆⟩bias and the population correction; together they
cancel up to a tangent-leakage piece, because of the information equation AH⋆
Γ = PTΓ. The second-order
term R2nd is the Taylor remainder from the score expansion in R1st, kept separate for cleanliness.
D.6.
Single-contrast remainder bound
The combined-error theorem of [Li et al., 2026, Theorem 4.5] gives a sharp term-by-term bound on each
component of (D.6). We restate the matrix specialization with explicit dependence on CA, ¯d, and ∥Γ∥1.
Theorem D.5 (Single-contrast remainder bound). Fix any a > 0 and any contrast Γ ∈Rdt×dm sat-
isfying Assumptions A.8 and A.9. Under Assumptions A.1–A.11, with probability at least 1 −n−a,
|RΓ
n| ≤C(µ,r,κ,B,cB,CB)CA ∥Γ∥1
¯d logc(n ¯d)
n
.
(D.7)
Equivalently,
√n|RΓ
n|
≤
C CA ∥Γ∥1
p ¯dlogc(n ¯d)/n,
which
is
o(1)
under
the
CLT
condition
CA
p ¯dlogc(n ¯d)/n →0 of Theorem 4.1.
Proof.
[Term-by-term bounds] We outline the per-term bounds of [Li et al., 2026, Appendices G.4–
G.10]. Each bound is C(µ,r,m)∥Γ∥1 times the displayed dimensional factor times C(≤1)
A
(linear at most).
Throughout we use the abbreviation ρ :=
p ¯dlogc ¯d/n for the spectral subspace-perturbation parameter,
which satisfies ρ ≍rn under the bounded-signal assumption.
(i) Direction-error empirical process RH
emp. By the Frobenius reduction (Lemma A.4) and ℓ1 extrac-
tion over the basis decomposition of Γ, Var⋆(sη(Y, bη)
D
bHΓ −H⋆
Γ, X
E
) ≤CB ∥Γ∥2
1 ρ2/d⋆· d⋆= CB ∥Γ∥2
1 ρ2
under the resolvent identity for bA−1 −A−1 and the spectral-perturbation bound on
 bPT −PT

op. The sub-
exponential envelope is bounded by
 bHΓ −H⋆
Γ

∞≤CA ∥Γ∥1 times an incoherence factor. Bernstein’s inequal-
ity yields |RH
emp| ≤C ∥Γ∥1 ρ/√n+C CA ∥Γ∥1 ¯dlog ¯d/n. The first term is sub-leading at the CLT scale because
42


<!-- page 43 -->
ρ/√n =
p ¯dlog ¯d/n = o(
p ¯d/n/√n) in our regime; the second is the bottleneck term and is the only place
where CA appears.
(ii) Score-perturbation empirical process Rη
emp. By the Lipschitz property of the BTL score,
|sη(Y, bη) −sη(Y,η⋆)| ≤|bη −η⋆| = |⟨∆, X⟩| ≤2∥∆∥∞, and Bernstein’s inequality on the centered Bernoulli
noise gives |Rη
emp| ≤C ∥Γ∥1 ∥∆∥∞
p
log(n ¯d)/n, which is dominated by RH
emp.
(iii) Projection leakage Rproj. |Rproj| = |⟨(I −PT)Γ, ∆⟩| ≤∥(I −PT)Γ∥F ∥∆∥F. By the closed form (D.5)
and incoherence, ∥(I −PT)Γ∥F ≤C(µ,r)ρ∥Γ∥F ≤Cρ∥Γ∥1, and ∥∆∥F ≲
p
r ¯d/n by the Frobenius initializa-
tion rate. Combining, |Rproj| ≤C ∥Γ∥1 /n, which is also dominated.
(iv) H-direction bias RHbias. Taylor expansion gives RHbias =
D
bHΓ −H⋆
Γ, (G −bG) bHΓ
E
plus higher-order
corrections. Using the perturbation
G −bG

op ≲∥∆∥∞,
 bHΓ −H⋆
Γ

F ≤C CA ∥Γ∥1 ρ,
 bHΓ

F ≤C CA ∥Γ∥1,
and the Frobenius reduction, |RHbias| ≤C CA ∥Γ∥1 ¯dlog ¯d/n.
(v) First-order cancellation R1st. By the Taylor expansion of the population score difference and the
information equation AH⋆
Γ = PTΓ (which makes ⟨PTΓ, ∆⟩+ P⋆[SbΘ(H⋆
Γ) −SΘ⋆(H⋆
Γ)] exactly cancel up to a
normal-component term plus a second-order remainder), we get |R1st| ≤C CA ∥Γ∥1 ¯dlog ¯d/n.
(vi) Second-order score remainder R2nd. |R2nd| ≤C B2P⋆[⟨H⋆
Γ, X⟩∥∆∥2
∞] ≤C CA ∥Γ∥1 ¯dlog ¯d/n using
B2 ≤1/σ2 = O(1) and P⋆[⟨H⋆
Γ, X⟩2] ≤Veff(Γ) ≤C CAd⋆∥PTΓ∥2
F /d⋆≤C CA ∥Γ∥1.
Combining. Summing all six terms and taking the maximum, |RΓ
n| ≤C(µ,r,κ,B)CA ∥Γ∥1 ¯dlogc(n ¯d)/n,
with the leading contribution from terms (i), (iv), (v), (vi). Probability calibration to 1 −n−a follows by
setting the free Bernstein tail constant x = Calog(n ¯d) for C large.
D.7.
Uniform single-contrast remainder over a contrast family
The bound in Theorem D.5 is per-contrast. We extend it uniformly over a polynomial-size family by a careful
union bound.
Theorem D.6 (Uniform one-step remainder). Let F ⊂Rdt×dm be any family of contrasts of size
|F| ≤(dtdm)CF for an absolute constant CF, such that each Γ ∈F satisfies Assumption A.9 with a uniform
alignment constant αmin > 0 and Assumption A.8 with uniform constants M,Cψ. Fix any a > 0. Then with
probability at least 1 −n−a,
max
Γ∈F
√n|RΓ
n| ≤C CA
r ¯dpolylog(n ¯d)
n
,
(D.8)
where the polylog absorbs both the logarithmic factor inherited from Theorem D.5 and the log |F| ≤CF log( ¯d)
factor from the union bound.
Proof.
The proof is by union bound on top of Theorem D.5, with care taken so that no Γ-dependent
constants degrade.
Step 1: per-contrast bound with sharpened tail. Theorem D.5 produces the bound (D.7) with
probability at least 1 −2e−x for the underlying Bernstein steps; tracking the free parameter x through the
proof yields
Pr
h√n|RΓ
n| ≥K CA ∥Γ∥1
q
x ¯d logc ¯d/n
i
≤2e−x
for an absolute constant K depending on (µ,r,κ,B,cB,CB,cν,Cν,cπ,Cπ). The right-hand side combines the
contributions of all six terms; the dominant scaling is
p
x ¯d/n from terms (iv)–(vi).
43


<!-- page 44 -->
Step 2: union bound over F. Set x = Calog(n ¯d) + CF log ¯d and apply the per-contrast bound to each
Γ ∈F; a union bound over the |F| ≤¯dCF contrasts gives Pr[maxΓ
√n|RΓ
n| ≥K CA
p
x ¯dlogc ¯d/n] ≤2|F|e−x ≤
2n−a. For Γ ∈F, ∥Γ∥1 ≤Cψ = O(1) is absorbed into the constant.
Step
3:
rate.
The
displayed
bound
becomes
K CA
p
(alog(n ¯d) + CF log ¯d) ¯dlogc ¯d/n
≤
C CA
p ¯dpolylog(n ¯d)/n, absorbing both log factors into the polylog.
D.8.
Multivariate Berry–Esseen for the leading term
We now establish the rate O(
p ¯d/n) for the leading i.i.d. EIF average. This is the matrix specialization of
[Li et al., 2026, Appendix G.16], restated for completeness. The result for the standardized scalar leading
term yields the multivariate version after polarization.
Theorem D.7 (Berry–Esseen for the standardized leading term). Fix a single contrast Γ satisfy-
ing Assumptions A.8–A.9, and let Zi := ϕΓ(Wi)/σΓ be the standardized oracle EIF coordinates with σ2
Γ =
Veff(Γ). Then
ρn := sup
t∈R
   Pr
  1
√n
n
X
i=1
Zi ≤t
 
−Φ(t)
    ≤C
r ¯d
n.
(D.9)
Proof.
We compute the second and third moments of Zi explicitly and apply the classical (univariate)
Berry–Esseen theorem.
Step 1: mean zero. E⋆Zi = σ−1
Γ E⋆ϕΓ(Wi) = σ−1
Γ E⋆[sη(Y,η⋆)⟨H⋆
Γ, X⟩], which vanishes by the score-
centering identity E⋆[sη(Y,η⋆) | X] = 0.
Step 2: second moment. By the definition of σ2
Γ and Fisher comparability (Lemma A.5),
E⋆Z2
i = 1
σ2
Γ
E⋆[s2
η ⟨H⋆
Γ, X⟩2] = 1
σ2
Γ
⟨H⋆
Γ, AH⋆
Γ⟩= Veff(Γ)
σ2
Γ
= 1.
By the Frobenius reduction (Lemma A.4) applied to H⋆
Γ, E⋆⟨H⋆
Γ, X⟩2 ≍∥H⋆
Γ∥2
F /d⋆, and Fisher comparability
gives σ2
Γ ≍∥H⋆
Γ∥2
F /d⋆, so
cB
d⋆∥H⋆
Γ∥2
F ≤σ2
Γ ≤CB
d⋆∥H⋆
Γ∥2
F .
(D.10)
Step 3: third absolute moment.
E⋆|Zi|3 = 1
σ3
Γ
E⋆[|sη|3|⟨H⋆
Γ, X⟩|3] ≤C3
σ3
Γ
E⋆|⟨H⋆
Γ, X⟩|3
where C3 := E⋆[|sη(Y,η⋆)|3 | X] ≤1 under Assumption A.1(iv) (since |sη| ≤1). Now use the elementary
inequality E|W|3 ≤(sup|W|)E|W|2 with W = ⟨H⋆
Γ, X⟩:
E⋆|⟨H⋆
Γ, X⟩|3 ≤
 
sup
x |⟨H⋆
Γ, x⟩|
 
E⋆⟨H⋆
Γ, X⟩2 .
Since H⋆
Γ ∈T, ⟨H⋆
Γ, x⟩= ⟨H⋆
Γ, PTx⟩, so by Cauchy–Schwarz and the tangent-projection envelope
supx ∥PTx∥F ≲
p ¯d/d⋆(which follows from the basis-tensor projection bound and triangle inequality for
pairwise differences; see Lemma D.8 below),
sup
x∈X |⟨H⋆
Γ, x⟩| ≤∥H⋆
Γ∥F sup
x ∥PTx∥F ≲∥H⋆
Γ∥F
r ¯d
d⋆.
(D.11)
Combined with E⋆⟨H⋆
Γ, X⟩2 ≍∥H⋆
Γ∥2
F /d⋆,
E⋆|⟨H⋆
Γ, X⟩|3 ≲∥H⋆
Γ∥F
r ¯d
d⋆· ∥H⋆
Γ∥2
F
d⋆
= ∥H⋆
Γ∥3
F
√¯d
(d⋆)3/2
.
44


<!-- page 45 -->
Substituting into the expression for E⋆|Zi|3, and using σ3
Γ ≥c3/2
B
∥H⋆
Γ∥3
F /(d⋆)3/2 from (D.10),
E⋆|Zi|3 ≤C3
σ3
Γ
· ∥H⋆
Γ∥3
F
√¯d
(d⋆)3/2
≤C3c
−3/2
B
p
¯d.
Step 4: Berry–Esseen. By the classical univariate Berry–Esseen theorem (e.g. Shevtsova 2010 with
constant CBE = 0.4748),
ρn ≤CBE
√n E⋆|Zi|3 ≤C3c
−3/2
B
CBE
r ¯d
n.
Lemma D.8 (Pairwise tangent-projection envelope). Under µ-incoherence and the row-centering
gauge, for every admissible design tensor X = et(em −em′)⊤, ∥PTX∥F ≤C(µ,r)
p ¯d/d⋆. The same bound
holds for bPTX under the estimated incoherence guarantee.
Proof.
Decompose X = E(t,m) −E(t,m′) where Eω = ete⊤
m is the canonical basis tensor. By the basis-tensor
projection bound (a direct consequence of the closed-form projector (D.5) together with µ-incoherence),
∥PTEω∥F ≤C(µ,r)
p ¯d/d⋆for every basis tensor. By linearity and the triangle inequality, ∥PTX∥F ≤
PTE(t,m)

F +
PTE(t,m′)

F ≤2C(µ,r)
p ¯d/d⋆.
D.9.
Multivariate rectangle CLT for fixed q
We combine the Berry–Esseen rate of Theorem D.7 with the uniform remainder bound of Theorem D.6 to
obtain the rectangle CLT.
Let
Φq(Wi) := (ϕ1(Wi),...,ϕq(Wi))⊤,
Sn := n−1/2 Pn
i=1 Φq(Wi),
Tn := √n( bψ −ψ),
and
rn :=
(RΓ1
n ,...,R
Γq
n ).
Theorem D.9 (Oracle multivariate Berry–Esseen). For any rectangle B ∈Rq,
ρorac
n
:= sup
B∈Rq
|Pr(Sn ∈B) −Pr(ZΓ ∈B)| ≲
r ¯d
n,
where ZΓ ∼N(0,Σ).
Proof.
Handle possibly singular Σ by restriction to its range. Let s := rank(Σ) ≤q and U ∈Rq×s have
orthonormal columns spanning range(Σ). For any v ∈Null(Σ), Var(v⊤Φq) = v⊤Σv = 0, so v⊤Φq = 0 almost
surely; hence Φq = U eΦ for eΦ := U ⊤Φq ∈Rs with covariance eΣ := U ⊤ΣU nonsingular. Set Y := eΣ−1/2eΦ ∈Rs,
so E[Y ] = 0 and E[Y Y ⊤] = Is. For any rectangle B ⊂Rq, {Sn ∈B} reduces to {n−1/2 PYi ∈CB} for the
convex set CB := {y ∈Rs : U eΣ1/2y ∈B}.
By the convex-set multivariate Berry–Esseen theorem [Bentkus, 2005] for fixed dimension s ≤q = O(1),
sup
C∈Cs
   Pr
  1
√n
X
Yi ∈C
 
−Pr(Gs ∈C)
    ≤Cs
E∥Y ∥3
2
√n
.
For any unit a ∈Ss−1, a⊤Y is a scalar standardized sum of EIF coordinates over the contrast Γa =
P
j(eΣ−1/2U ⊤a)jΓj (a finite linear combination of the original contrasts). Sub-exponential tails on ϕΓ and
the third-moment computation of Theorem D.7 (applied to Γa) give E|a⊤Y |3 ≤C
√¯d, uniformly in a. Hence
E∥Y ∥3
2 ≤Cs3/2√¯d ≤C
√¯d (since s = O(1)), giving the claim.
45


<!-- page 46 -->
Theorem D.10 (Rectangle CLT for the feasible statistic; restatement of Theorem 4.1).
Under Assumptions A.1–A.11,
sup
B∈Rq
|Pr(Tn ∈B) −Pr(ZΓ ∈B)| ≲CA
r ¯dpolylog(n ¯d)
n
.
Consequently, if CA
p ¯dpolylog(n ¯d)/n →0, then √n( bψ −ψ) ⇝N(0,Σ).
Proof.
Let D := diag(√Σ11,...,
p
Σqq) (removing zero diagonals from the reporting family, which is
permitted by Remark D.3). Define standardized vectors ¯Tn := D−1Tn, ¯Sn := D−1Sn, ¯Z := D−1ZΓ. The
standardized remainder event is
Er :=
 D−1rn

∞≤δn
	
,
δn := C CA
q
¯dpolylog(n ¯d)/n.
By Theorem D.5 applied to each Γj, Pr(Ec
r) ≤q n−a ≤n−a/2, giving πn := Pr(Ec
r) ≤n−a after relabelling.
For any rectangle B = Qq
j=1[αj,βj], define enlarged and shrunk rectangles B±δn := Q[αj ∓δn,βj ±δn] (with
the convention that an empty interval makes B−δn = ∅). On Er, ¯Tn ∈B implies ¯Sn ∈B+δn, and ¯Sn ∈B−δn
implies ¯Tn ∈B. Hence
Pr( ¯Sn ∈B−δn) −πn ≤Pr( ¯Tn ∈B) ≤Pr( ¯Sn ∈B+δn) + πn.
Combining with the oracle Berry–Esseen (Theorem D.9, applied to standardized ¯Sn),
|Pr( ¯Tn ∈B) −Pr( ¯Z ∈B)| ≤ρorac
n
+ πn + Pr( ¯Z ∈B+δn \ B−δn).
The Gaussian boundary band is bounded by Gaussian anti-concentration: each face contributes at most
δn
p
2/π (one-dimensional standard normal density at most
p
1/(2π)), and there are 2q faces, so the band is
at most Cqδn. Since q = O(1), πn ≤n−a, and ρorac
n
≤C
p ¯d/n ≤Cδn/CA, the total is ≲CA
p ¯dpolylog(n ¯d)/n.
Linear coordinate rescaling maps rectangles to rectangles, so the bound transfers from ¯Tn to Tn without loss.
D.10.
Plug-in variance: relative consistency
We port the variance-consistency proof from [Li et al., 2026, Appendix G.13] (cf. also AoS Lemma G.13) in
full detail. The plug-in variance estimator for a single contrast Γ is bVΓ := Pn[bϕ2
Γ] where bϕΓ = sη(Y, bη)
D
bHΓ, X
E
.
Proposition D.11 (Relative variance consistency). Fix any a > 0 and any contrast Γ satisfying
Assumptions A.8–A.9. Under the conditions of Theorem D.5, with probability at least 1 −n−a,
   
bVΓ
VΓ
−1
    ≤C
r ¯d logc(n ¯d)
n
.
(D.12)
This bound contains no CA factor.
Proof.
We follow the AoS three-part decomposition: direction-induced part, score plug-in part, and empir-
ical fluctuation part. The key point is that the spectral scale d⋆of A−1 cancels in every relative ratio,
eliminating the CA factor.
Step 1: relative direction error δH. We first show δH :=
 bHΓ −H⋆
Γ

F /∥H⋆
Γ∥F ≤Cρ/αΓ where ρ :=
p ¯dlog ¯d/naux is the subspace-perturbation parameter. Use the resolvent identity bA−1 −A−1 = A−1(A −
bA) bA−1 on the tangent space:
bHΓ −H⋆
Γ = A−1( bPT −PT)Γ
|
{z
}
T1
+A−1(A −bA) bA−1 bPTΓ
|
{z
}
T2
.
46


<!-- page 47 -->
For T1, use ∥A−1∥op ≍d⋆on the tangent space (eigenvalues of A lie in [cB/d⋆,CB/d⋆] by Lemma A.5). This
common factor d⋆cancels in the relative Frobenius ratio:
∥T1∥F
∥H⋆
Γ∥F
≤
∥A−1∥op
( bPT −PT)Γ

F
∥A−1∥op ∥PTΓ∥F −O(ρ) ≤
C
( bPT −PT)Γ

F
∥PTΓ∥F
.
Under the sparse-target projector perturbation bound
( bPT −PT)Γ

F ≲ρ
p ¯d/d⋆∥Γ∥F (which follows from
the closed-form projector (D.5) and subspace perturbation ∥Pb
U −PU⋆∥op ≲ρ), together with Assumption A.9
∥PTΓ∥F ≥αΓ
p ¯d/d⋆∥Γ∥F, the ratio simplifies to ∥T1∥F /∥H⋆
Γ∥F ≤Cρ/αΓ.
For T2,
A −bA

op ≤Cρ/d⋆by Fisher operator perturbation (using
bΘ −Θ⋆
∞≤Cρ and Lipschitzness
of I(η)), so
A−1(A −bA)

op ≤∥A−1∥op
A −bA

op ≤Cρ, and ∥T2∥F ≤Cρ
 bA−1 bPTΓ

F ≤Cρ∥H⋆
Γ∥F (using
 bHΓ

F ≍∥H⋆
Γ∥F on the event of Step 1, by an iterative argument). Combining, δH ≤Cρ/αΓ.
Step 2: variance difference reduces to direction error. We show that for any oracle / plug-in pair
ϕ⋆, bϕ and any ”score-squared” factor s⋆with E⋆[s2
⋆f(X)] ≍E⋆[f(X)] for nonnegative f,
|P⋆[s2
⋆(
D
bHΓ, X
E2
−⟨H⋆
Γ, X⟩2)]|
P⋆[s2
⋆⟨H⋆
Γ, X⟩2]
≲δH.
Write a :=
D
bHΓ, X
E
, b := ⟨H⋆
Γ, X⟩, so a2 −b2 = (a −b)(a + b). Cauchy–Schwarz gives P⋆|a2 −b2| ≤(P⋆(a −
b)2)1/2(P⋆(a+b)2)1/2, and the Frobenius reduction (Lemma A.4) gives P⋆(a−b)2 ≍
 bHΓ −H⋆
Γ

2
F /d⋆, P⋆b2 ≍
∥H⋆
Γ∥2
F /d⋆. When δH ≤c0 < 1, P⋆(a + b)2 ≤2P⋆(a −b)2 + 8P⋆b2 ≤10P⋆b2, so P⋆|a2 −b2| ≲δHP⋆b2. Dividing
by P⋆[s2
⋆b2] ≍P⋆b2 yields the claim.
Step 3: score plug-in part. By the Lipschitz bound on the BTL score derivative (Assumption A.1(iii)),
|sη(Y, bη)2 −sη(Y,η⋆)2| ≤2|sη||∂ηsη||bη −η⋆| ≤C
bΘ −Θ⋆
∞, so Pn[(sη(Y, bη)2 −sη(Y,η⋆)2)
D
bHΓ, X
E2
] ≤
C ∥∆∥∞Pn[
D
bHΓ, X
E2
] ≤C ∥∆∥∞VΓ, giving a relative contribution O(∥∆∥∞) = O(rn).
Step 4: empirical fluctuation. Conditional on the auxiliary sample, bVΓ −P⋆[bϕ2
Γ] = (Pn −P⋆)(bϕ2
Γ) is a
centered sum. Bernstein with envelope bϕ2
Γ ≤VΓ ¯d (from Step 5 of the Bernstein argument in Appendix E.6
below) and variance proxy P⋆bϕ4
Γ ≤V 2
Γ ¯d gives |(Pn −P⋆)bϕ2
Γ| ≤VΓ
p ¯dlog(n ¯d)/n = VΓrn.
Combining. The three parts contribute O(ρ/αΓ)+O(∥∆∥∞)+O(rn) = O(rn) relative error: the spectral
scale d⋆of A−1 cancels in the relative direction error, the score plug-in is a direct entrywise estimator bound,
and the empirical fluctuation is controlled by the Bernstein argument.
D.11.
Covariance consistency in correlation form
Using the polarization identity, the variance consistency of Proposition D.11 extends to a covariance consis-
tency statement. The cleanest form is in correlation ρjk := Σjk/
p
ΣjjΣkk, as relative-error consistency for
off-diagonal Σjk is generally unattainable.
Proposition D.12 (Covariance consistency). Fix any a > 0 and let bΣjk := Pn[bϕj bϕk] be the plug-in
covariance for any pair (j,k) of contrasts in a polynomial-size family. Under the conditions of Proposi-
tion D.11,
max
j,k |bΣjk −Σjk| ≤C (Σjj + Σkk)rn,
rn :=
q
¯d logc(n ¯d)/n,
(D.13)
47


<!-- page 48 -->
with probability at least 1 −n−a. In particular, if the diagonal variances are comparable, i.e. cΣ ≤Σjj/Σkk ≤
CΣ,
max
j,k |bρjk −ρjk| ≤C rn.
(D.14)
Proof.
We use polarization plus the variance consistency lemma applied to Γj, Γk, and Γj + Γk.
Step 1: linearity. The direction maps are linear in Γ: H⋆
Γj+Γk = A−1PT(Γj + Γk) = H⋆
j + H⋆
k, and analo-
gously for the plug-in bHΓj+Γk = bHj + bHk. Hence ϕΓj+Γk = ϕj + ϕk and bϕΓj+Γk = bϕj + bϕk.
Step 2: polarization. For the oracle covariance, VΓj+Γk = P⋆[(ϕj + ϕk)2] = Σjj + Σkk + 2Σjk, so
Σjk = 1
2(VΓj+Γk −VΓj −VΓk).
Identically for the plug-in, bΣjk = 1
2(bVΓj+Γk −bVΓj −bVΓk). Subtracting,
bΣjk −Σjk = 1
2
 
(bVΓj+Γk −VΓj+Γk) −(bVΓj −VΓj) −(bVΓk −VΓk)
 
.
Step 3: applying variance consistency. By Proposition D.11 applied to each of the three contrasts
(Γj,Γk,Γj + Γk), and using VΓ ≤1 · Σjj etc. from a union bound at level 1 −n−a for each contrast,
|bΣjk −Σjk| ≤Crn(VΓj+Γk + VΓj + VΓk).
Since Σ ⪰0, |Σjk| ≤
p
ΣjjΣkk ≤(Σjj +Σkk)/2, so VΓj+Γk ≤Σjj +Σkk +2|Σjk| ≤2(Σjj +Σkk). Thus VΓj+Γk +
VΓj + VΓk ≤3(Σjj + Σkk), giving (D.13).
Step 4: correlation consistency. Under Σjj ≍Σkk, divide (D.13) by
p
ΣjjΣkk to get |bΣjk/
p
ΣjjΣkk −
ρjk| ≤Crn. For the diagonal-rescaled empirical correlation bρjk = bΣjk/
q
bΣjj bΣkk, apply Taylor expansion of
(x,y,z) 7→x/√yz around (Σjk,Σjj,Σkk) and use the diagonal variance consistency to obtain (D.14). See
covariance consistency argument.md for details.
Remark D.13 (Why no relative-error bound on Σjk). A relative-error bound of the form |bΣjk −
Σjk| ≲|Σjk|rn is in general unattainable without an additional lower bound on |Σjk|, because off-diagonal
covariances may be zero or arbitrarily small even when Σjj,Σkk are large. Two orthogonal EIFs (Σjk = 0)
would make the relative ratio unbounded, while the empirical bΣjk still fluctuates around zero at a non-trivial
scale. The natural uniform statement is the absolute error normalized by the diagonal scale, as (D.13). Under
the diagonal-comparability condition, this is equivalent to absolute correlation consistency (D.14), which is
what the high-dimensional CCK calibration in Appendix E actually requires.
D.12.
Joint Loewner efficiency restatement
Combining Proposition D.2 with Theorem D.10, our one-step estimator attains the Loewner-minimal asymp-
totic covariance Σ, so it is jointly semiparametrically efficient for the fixed finite contrast family {Γ1,...,Γq}.
This proves the efficiency claim referenced in Section 4.
48


<!-- page 49 -->
D.13.
Diagonal scale of Σjj for score-gap contrasts
For score-gap contrasts Γ = et(em −em′)⊤, we record the explicit scaling of Σjj used in Appendix E.
Lemma D.14 (Diagonal scale of efficient variance for score gaps). Under Assumptions A.1–A.3,
in the balanced regime dt ≍dm ≍d, Σjj = Veff(Γ) ≍d for every score-gap contrast Γ = et(em −em′)⊤. The
standard error of the one-step estimator is therefore σj/√n ≍
p
d/n, and the simultaneous calibration over
p score gaps gives a band width of order
p
dlog p/n.
Proof.
Under near-uniform pairwise sampling and Fisher comparability, Σjj = Veff(Γ) = ⟨PTΓ, A−1PTΓ⟩.
On the tangent space, A−1 has spectral scale d⋆(since A has spectral scale 1/d⋆). By the closed-form
projector (D.5) and incoherence, ∥PTΓ∥2
F ≍1/dt + 1/dm. Therefore Σjj ≍d⋆(1/dt + 1/dm) = dt + dm ≍2d in
the balanced regime. Standard error and band width follow.
Appendix E:
Proof of Theorem 5.1, Corollary 5.2, and the top-K extension
This appendix proves the simultaneous ranking-inference results of Section 5. We use the Chernozhukov–
Chetverikov–Kato (CCK) high-dimensional approximate-means framework, which we state in the form
needed and then verify each constituent error term explicitly, in order, in subsequent subsections. We con-
dition throughout on the master good event En of Appendix A.8.
E.1.
Setup: contrast family and statistics
For a contrast family J (indexed by score-gap contrasts as in the three applications below), let ∆j := ψΓj(Θ⋆),
and adopt the standardized oracle and plug-in coordinates from Appendix D.1:
ϕj(Wi) = sη(Yi,η⋆
i )

H⋆
j , Xi
 
,
Zij = ϕj(Wi)
σj
,
bZij =
bϕj(Wi)
bσj
,
with bσ2
j := Pn[bϕ2
j], σ2
j = Σjj = Veff(Γj). Define the cardinality p := |J |, which will be polynomial in ¯d for each
application below.
Oracle and feasible test statistics.
T0 := max
j∈J
    1
√n
n
X
i=1
Zij
   ,
T := max
j∈J
|√n(b∆j −∆j)|
bσj
.
(E.1)
Oracle and feasible multiplier-bootstrap statistics. With i.i.d. multipliers ξi ∼N(0,1) independent
of the data,
W0 := max
j∈J
    1
√n
n
X
i=1
ξiZij
   ,
T ∗:= max
j∈J
    1
√n
n
X
i=1
ξi bZij
   .
(E.2)
Let c∗
1−α denote the conditional (1 −α)-quantile of T ∗given the data.
Reference Gaussian. Let ΣZ := (ΣZ,jk)j,k∈J with ΣZ,jk := E⋆[ZjZk] = Σjk/(σjσk), and let G ∼N(0,ΣZ),
Z0 := maxj∈J |Gj|.
E.2.
The CCK approximate-means theorem and the master decomposition
Coverage of the simultaneous score-gap bands bIj := [b∆j ± c∗
1−αbσj/√n] is equivalent to T ≤c∗
1−α. By the
Chernozhukov–Chetverikov–Kato approximate-means theorem [Chernozhukov et al., 2013, 2014, 2017],
sup
α∈(0,1)
  Pr(T ≤c∗
1−α) −(1 −α)
   ≤En,
(E.3)
49


<!-- page 50 -->
where the aggregate error En decomposes into five named ingredients
En ≤
ρn
|{z}
(I) oracle CCK Gaussian approximation
+
π(ϑn) + Pr(∆n > ϑn)
|
{z
}
(II) Gaussian-multiplier covariance error
+
an
p
log p
|
{z
}
(III) one-step plug-in transfer
+
bn log p
| {z }
(IV) standard-error plug-in transfer
+
cn log p
| {z }
(V) feasible-bootstrap transfer
,
(E.4)
where each ingredient is defined as follows.
(I) Oracle CCK Gaussian approximation error ρn := supz |Pr(T0 ≤z) −Pr(Z0 ≤z)|, bounded in
Appendix E.3 via the envelope condition L2
n log7(pn)/n →0 for an envelope Ln on the standardized
coordinates Zij.
(II) Gaussian-multiplier covariance error. The conditional law of W0 given the data is a Gaussian maximum
with empirical covariance bΣZ; the Gaussian comparison step (Chernozhukov et al., 2014) bounds the
Kolmogorov distance between the conditional law of W0 and Z0 by π(ϑ) := Cϑ1/3{1 ∨log(p/ϑ)}2/3 on
the event {∆n ≤ϑ}, where ∆n := maxj,k∈J |Pn[ZjZk]−P⋆[ZjZk]|. This is bounded in Appendix E.4 by
Bernstein.
(III) One-step plug-in transfer error an := maxj∈J |√n(b∆j −∆j)/σj −
1
√n
P
i Zij|, the standardized one-step
remainder. Bounded in Appendix E.5 via the uniform single-contrast bound (Theorem D.6).
(IV) Standard-error plug-in transfer error bn := maxj∈J |bσj/σj −1|, bounded in Appendix E.5 via the vari-
ance consistency result (Proposition D.11) plus a union bound.
(V) Feasible-bootstrap transfer error c2
n := maxj∈J Pn( bZj −Zj)2, the empirical square-loss between the
standardized plug-in and oracle EIF coordinates. This is the load-bearing quantity, bounded in
Appendix E.6 via the empirical square-loss argument (Proposition E.5).
The CCK approximate-means theorem [Chernozhukov et al., 2013, Theorem 3.1], which combines the
oracle Gaussian approximation, the Gaussian comparison step, and the bound on multiplier-bootstrap cor-
rection, yields (E.3)–(E.4) when each plug-in error (an,bn,cn) is converted to a coverage-error contribution
by multiplication with √log p (which arises from the maximal Gaussian quantile scale).
The remainder of this appendix bounds each of (I)–(V) explicitly and combines them in Appendix E.7.
We then derive the three applications in Appendices E.8–E.10.
E.3.
Bounding the standardized envelope Ln (and ingredient I)
We compute an explicit envelope for the standardized oracle coordinates Zij, in which the constant CA
appears only as a first-order factor, never inside a polynomial.
Lemma E.1 (Envelope of standardized oracle coordinates). Under
Assumptions
A.1–A.9,
for
every j ∈J and every admissible design Xi,
|Zij| ≤Ln,
L2
n ≤C(µ,r,κ,B,cB,CB)CA ¯d,
(E.5)
where the constant in front of ¯d is a polynomial in (µ,r,κ,B,cB,CB) but contains CA only as a first-order
factor. Consequently L2
n log7(pn)/n ≤C CA ¯dlog7(pn)/n →0 under Assumption A.11 and the CLT condition
CA
p ¯dlogc(n ¯d)/n →0 of Theorem 4.1, and so the oracle CCK Gaussian approximation error
ρn := sup
z |Pr(T0 ≤z) −Pr(Z0 ≤z)| →0.
50


<!-- page 51 -->
Proof.
The proof is a direct calculation. Recall Zij = ϕj(Wi)/σj.
Step 1: bound |ϕj(Wi)| using the tangent envelope. Since sη(Yi,η⋆
i ) ∈[−1,1] almost surely (Assump-
tion A.1(iv)),
|ϕj(Wi)| = |sη(Yi,η⋆
i )| · |

H⋆
j , Xi
 
| ≤|

H⋆
j , Xi
 
|.
Using H⋆
j ∈T,

H⋆
j , Xi
 
=

H⋆
j , PTXi
 
, and Cauchy–Schwarz with the tangent envelope (Lemma D.8),
|

H⋆
j , Xi
 
| ≤
H⋆
j

F · ∥PTXi∥F ≤C(µ,r)
H⋆
j

F
q
¯d/d⋆.
Step
2:
bound
σ2
j
from
below
using
the
Frobenius
reduction. By Lemma A.5, σ2
j =
E⋆[s2
η

H⋆
j , X
 2] ≥cBE⋆
H⋆
j , X
 2, and by the Frobenius reduction (Lemma A.4) E⋆
H⋆
j , X
 2 ≍
H⋆
j
2
F /d⋆,
so
σ2
j ≥cB
C
H⋆
j
2
F
d⋆
.
Step 3: combine.
Z2
ij = ϕj(Wi)2
σ2
j
≤
C(µ,r)
H⋆
j
2
F ( ¯d/d⋆)
(cB/C)
H⋆
j
2
F /d⋆
= C(µ,r,cB) ¯d.
The factor
H⋆
j
2
F appears in numerator and denominator and cancels, and this cancellation is the same
mechanism that drives the empirical square-loss bound in Appendix E.6.
Step 4: where does CA enter? The factor CA enters through the plug-in envelope for bZij, which
is needed in the multiplier-bootstrap covariance computation in Appendix E.4 and the bootstrap-statistic
transfer in Appendix E.6. There, the bound is | bZij| ≤C1/2
A
√¯d at most, so the empirical square-loss difference
| bZij −Zij|2 ≤C CA ¯d remains C1
A at most. No power higher than C1
A appears anywhere in the analysis.
Step 5: oracle CCK approximation. By the CCK Gaussian approximation theorem [Chernozhukov
et al., 2017, Theorem 2.1], applied to the i.i.d. mean-zero coordinates Zij with E⋆Z2
ij = 1 and envelope Ln,
if L2
n log7(pn)/n →0 then ρn = o(1). Substituting L2
n ≤C CA ¯d (counting the CA from the plug-in envelope
above to be safe; the oracle envelope is CA-free but the CCK constant absorbs both into one term) and
p ≤¯dO(1) yields the displayed sufficient condition.
E.4.
Bounding the empirical covariance error ∆n (ingredient II)
We now bound the maximum standardized covariance estimation error ∆n by a Bernstein argument, with
explicit calculation of the variance proxy and envelope.
Lemma E.2 (Maximum standardized covariance error). Fix any a > 0. With probability at least 1−
n−a,
∆n := max
j,k∈J
  Pn[ZjZk] −P⋆[ZjZk]
   ≤C
 
Ln
r
log(pn)
n
+ L2
n
log(pn)
n
 
≲
r
CA ¯d log(pn)
n
+ CA ¯dlog(pn)
n
.
(E.6)
Consequently, with the same probability, the Gaussian comparison step contributes
π(ϑn) := Cϑ1/3
n {1 ∨log(p/ϑn)}2/3 = o(1)
upon the choice ϑn = Ln
p
log(pn)/n + L2
n log(pn)/n.51


<!-- page 52 -->
Proof.
We apply Bernstein’s inequality to the centered products ZijZik −P⋆[ZjZk] for each pair (j,k),
then take a union bound over p2 pairs.
Step 1: variance proxy. Each ZijZik is bounded by L2
n in absolute value (Lemma E.1), and has variance
Var(ZijZik) ≤P⋆[Z2
j Z2
k] ≤L2
n P⋆[Z2
j ] = L2
n,
using P⋆Z2
j = 1 by definition.
Step 2: per-pair Bernstein. For each pair (j,k), Bernstein’s inequality gives, for any x > 0,
Pr
h  Pn[ZjZk] −P⋆[ZjZk]
   ≥
p
2L2
n x/n + L2
nx/(3n)
i
≤2e−x.
This is the standard Bernstein bound with envelope L2
n and variance L2
n.
Step 3: union bound. Set x = Calog(pn) with C sufficiently large, so the tail is at most 2p2e−Ca log(pn) ≤
2p−Ca+2n−Ca ≤n−a−1 for C large enough (using p ≤¯dO(1)). This gives Pr[∆n ≥Ln
p
2Calog(pn)/n +
L2
nCalog(pn)/(3n)] ≤n−a.
Step 4: π(ϑn) = o(1). With ϑn = Ln
p
log(pn)/n + L2
n log(pn)/n, we have ϑ1/3
n
≤(L2
n log(pn)/n)1/6 +
(L2
n log(pn)/n)1/3 (treating Ln
√x as the dominant term for small x), so π(ϑn) ≲(L2
n log(pn)/n)1/6 log2/3(p).
Under L2
n log7(pn)/n →0, this is o(1), confirming ingredient (II) is asymptotically negligible.
E.5.
Bounding the one-step plug-in errors an and bn (ingredients III, IV)
We carry over the bounds from Appendix D.7 and Proposition D.11, applied uniformly over the contrast
family J .
Lemma E.3 (One-step transfer). Under Theorem D.6 applied to F = J , with probability at least 1 −
n−a,
an := max
j∈J
   
√n(b∆j −∆j)
σj
−1
√n
n
X
i=1
Zij
    ≤C CA
r ¯dpolylog(n ¯d)
n
.
(E.7)
Proof.
By definition of R
Γj
n in Appendix D.5, √n(b∆j −∆j) =
1
√n
P
i ϕj(Wi)+√nR
Γj
n , and dividing by σj,
√n(b∆j −∆j)/σj =
1
√n
P
i Zij + √nR
Γj
n /σj. Hence an = maxj |√nR
Γj
n /σj|. By Theorem D.6 (and noting σj
is absorbed into the alignment constant via Assumption A.9), an ≤C CA
p ¯dpolylog(n ¯d)/n on the uniform-
remainder event.
Lemma E.4 (Standard-error transfer). With probability at least 1 −n−a,
bn := max
j∈J
   bσj
σj
−1
    ≤C
r ¯dpolylog(n ¯d)
n
.
(E.8)
Proof.
By Proposition D.11 applied to each Γj, |bVj/Vj −1| ≤Crn with probability 1 −n−a per contrast,
where Vj = σ2
j and bVj = bσ2
j . Take a union bound over the p ≤¯dO(1) contrasts at the cost of an extra
p
log ¯d
factor absorbed into the polylog. Take square roots: |bσj/σj −1| = |
q
bVj/Vj −1| ≤|bVj/Vj −1|/2 on the event
bn ≤1/2.
52


<!-- page 53 -->
E.6.
Bounding the feasible-bootstrap error cn (ingredient V)
We now prove the load-bearing bound on the empirical square-loss between the standardized plug-in and
oracle EIF coordinates. This is the matrix specialization of the bound in [Li et al., 2026, Section G.16].
Proposition E.5 (Estimated-EIF squared-loss bound). Under the conditions of Theorem D.10,
with probability at least 1 −n−a,
c2
n := max
j∈J Pn( bZj −Zj)2 ≤C r2
n,
r2
n :=
¯dpolylog(n ¯d)
n
.
(E.9)
The bound has no CA factor.
Proof.
The proof has four steps, following the strategy in [Li et al., 2026, Section G.16].
Step 1: standardize away bσj. On the event bn ≤1/2 (Lemma E.4),
| bZij −Zij| =
   
bϕj(Wi)
bσj
−ϕj(Wi)
σj
    ≤2|bϕj(Wi) −ϕj(Wi)|
σj
+ 2bn|Zij|.
Squaring and averaging,
Pn( bZj −Zj)2 ≤8Pn(bϕj −ϕj)2
σ2
j
+ 8b2
nPnZ2
j .
By definition PnZ2
j = 1 + (Pn −P⋆)Z2
j , and Bernstein with envelope L2
n and variance L2
n gives (Pn −
P⋆)Z2
j ≤Ln
p
log(pn)/n + L2
n log(pn)/n = o(1); hence maxj PnZ2
j ≤2 on this event. Combined with b2
n ≤Cr2
n
(Lemma E.4),
c2
n ≤8 maxj Pn(bϕj −ϕj)2/σ2
j
1
+ 16r2
n.
The second term is already at the r2
n scale; the first term is the main object to bound below.
Step 2: decompose the EIF error. Write bϕj(W) −ϕj(W) =
 sη(Y, bη) −sη(Y,η⋆)
 D
bHj, X
E
+
sη(Y,η⋆)
D
bHj −H⋆
j , X
E
, so Pn(bϕj −ϕj)2/σ2
j ≤2(T1j + T2j) with
T1j :=
Pn[(sη(Y, bη) −sη(Y,η⋆))2 D
bHj, X
E2
]
σ2
j
,
T2j :=
Pn[sη(Y,η⋆)2 D
bHj −H⋆
j , X
E2
]
σ2
j
.
Step 3: score plug-in term T1j. By the Lipschitz bound on sη in η (Assumption A.1(iii)), |sη(Y, bη) −
sη(Y,η⋆)| ≤|bη −η⋆| ≤2
bΘ −Θ⋆
∞. Hence
T1j ≤
4
bΘ −Θ⋆
2
∞Pn
D
bHj, X
E2
σ2
j
.
By plug-in variance consistency (Lemma E.4, applied to
bHj instead of H⋆
j ; the calculation is the
same), Pn
D
bHj, X
E2
/σ2
j = bσ2
j /σ2
j · Pn
D
bHj, X
E2
/bσ2
j = O(1) in probability. By the entrywise theorem (A.4),
bΘ −Θ⋆
2
∞≤C ¯dpolylog(n ¯d)/n = Cr2
n. Combining, maxj T1j ≤Cr2
n on En. Step 4: Empirical Fluctuation
T2,j
Lemma E.6 (Empirical square bound for estimated efficient directions). Let J be a polynomial-
size family of score-gap contrasts, and for each j ∈J write
H⋆
j := A−1PTΓj,
bHj := bA−1 bPTΓj,
Dj := bHj −H⋆
j .
53


<!-- page 54 -->
Let
σ2
j := P ⋆ 
{s⋆
i }2⟨H⋆
j ,Xi⟩2 
,
s⋆
i := s(Yi,η⋆
i ),
η⋆
i := ⟨Xi,Θ⋆⟩.
Suppose that on the good event En, uniformly over j ∈J ,
∥Dj∥F
∥H⋆
j ∥F
≤rn,
r2
n :=
¯dlogc(n ¯d)
n
,
and that the tangent-projection envelope satisfies
sup
X ∥PTX∥F ∨sup
X ∥bPTX∥F ≤C
r ¯d
d⋆.
Then, after increasing the logarithmic power c if necessary, with probability at least 1 −n−a,
max
j∈J Pn
{s⋆
i }2⟨bHj −H⋆
j ,Xi⟩2
σ2
j
≤Cr2
n.
In particular, this bound does not involve the inverse-information stability factor CA.
Proof.
Work on the good event En and condition on the auxiliary sample used to construct bHj and bPT.
Then Dj is fixed with respect to the evaluation sample. Define
Uij := {s⋆
i }2⟨Dj,Xi⟩2
σ2
j
.
We first control the population mean. By Fisher comparability and the pairwise Frobenius reduction,
P ⋆ 
{s⋆
i }2⟨Dj,Xi⟩2 
≲∥Dj∥2
F
d⋆
,
while
σ2
j = P ⋆ 
{s⋆
i }2⟨H⋆
j ,Xi⟩2 
≍∥H⋆
j ∥2
F
d⋆
.
Therefore
P ⋆Uj = P ⋆[{s⋆
i }2⟨Dj,Xi⟩2]
σ2
j
≲∥Dj∥2
F
∥H⋆
j ∥2
F
≤r2
n.
Next we prove an envelope bound for Uij that does not use ∥Dj∥∞. Since
Dj = bHj −H⋆
j ,
we have
|⟨Dj,Xi⟩| ≤|⟨bHj,Xi⟩| + |⟨H⋆
j ,Xi⟩|.
Because bHj ∈bT and H⋆
j ∈T,
|⟨bHj,Xi⟩| = |⟨bHj, bPTXi⟩| ≤∥bHj∥F ∥bPTXi∥F,
and
|⟨H⋆
j ,Xi⟩| = |⟨H⋆
j ,PTXi⟩| ≤∥H⋆
j ∥F ∥PTXi∥F.
The relative direction bound gives
∥bHj∥F ≤∥H⋆
j ∥F + ∥Dj∥F ≤(1 + rn)∥H⋆
j ∥F ≤2∥H⋆
j ∥F
54


<!-- page 55 -->
for all sufficiently large n. Hence, by the tangent-projection envelope,
|⟨Dj,Xi⟩| ≲∥H⋆
j ∥F
r ¯d
d⋆.
Since |s⋆
i | ≤1 for the BTL score,
0 ≤Uij ≲∥H⋆
j ∥2
F( ¯d/d⋆)
σ2
j
≲¯d,
where the last step uses σ2
j ≍∥H⋆
j ∥2
F/d⋆. Thus,
∥Uij∥∞≤C ¯d.
Moreover,
Var(Uij) ≤P ⋆U 2
ij ≤∥Uij∥∞P ⋆Uj ≲¯dr2
n.
Bernstein’s inequality therefore gives, for any x > 0,
|(Pn −P ⋆)Uj| ≲
r ¯dr2
nx
n
+
¯dx
n
with probability at least 1 −2e−x. Taking
x = Ca log n + 2log |J |
and union bounding over j ∈J , we obtain, with probability at least 1 −n−a,
max
j∈J |(Pn −P ⋆)Uj| ≲
r ¯dr2
n log(n|J |)
n
+
¯dlog(n|J |)
n
.
Since |J | is polynomial in ¯d, we have log(n|J |) ≲log(n ¯d). Recalling that
r2
n =
¯dlogc(n ¯d)
n
,
both Bernstein terms are absorbed into r2
n after increasing the logarithmic power c:
r ¯dr2
n log(n|J |)
n
+
¯dlog(n|J |)
n
≲r2
n.
Combining this empirical fluctuation bound with the population bound P ⋆Uj ≲r2
n, we conclude that
max
j∈J PnUj ≤max
j∈J P ⋆Uj + max
j∈J |(Pn −P ⋆)Uj| ≲r2
n.
This proves the claim.
Conclusion. Combining steps 1–4,
c2
n ≤8(T1j + T2j) + 16r2
n ≤Cr2
n.
The probability calibration to 1 −n−a follows from Appendix A.8.
55


<!-- page 56 -->
E.7.
Aggregate CCK approximate-means error
Combining Lemmas E.1–E.4 and Proposition E.5 with the master decomposition (E.4), we obtain the aggre-
gate CCK error.
Theorem E.7 (Aggregate CCK approximate-means error). Under Assumptions A.1–A.11 and
the CLT condition CA
p ¯dlogc(n ¯d)/n →0 of Theorem 4.1, the aggregate CCK error satisfies
En ≤ρn + π(ϑn) + Pr(∆n > ϑn) + an
p
log p + bn log p + cn log p = o(1).
Consequently
Pr(T ≤c∗
1−α) ≥1 −α −o(1),
and the simultaneous score-gap bands bIj := [b∆j ±c∗
1−αbσj/√n], j ∈J , satisfy Pr(∆j ∈bIj ∀j ∈J ) ≥1−α−o(1).
Proof.
Substitute the bounds:
ρn = o(1) (Lemma E.1),
π(ϑn) + Pr(∆n > ϑn) = o(1) (Lemma E.2),
an
p
log p ≤C CA
q
¯dpolylog(n ¯d)/n ·
q
log ¯d = o(1)
under the CLT condition,
bn log p ≤C
q
¯dpolylog(n ¯d)/n · log ¯d = o(1),
cn log p ≤C
q
¯dpolylog(n ¯d)/n · log ¯d = o(1)
under Assumption A.11. All five ingredients are o(1), so En = o(1), and the master CCK theorem (E.3)
delivers the coverage statement. Inverting T ≤c∗
1−α gives the simultaneous-band coverage.
E.8.
Application 1: rank confidence band for one task / one model
Fix a task t ∈[dt] and a model m ∈[dm]. Take Jt,m := {(t,ℓ) : ℓ̸= m}, so the gaps are ∆(m)
t,ℓ= Θ⋆
t,ℓ−Θ⋆
t,m for
ℓ̸= m and p = |Jt,m| = dm −1. Let bI(m)
t,ℓ= [bL(m)
t,ℓ, bU (m)
t,ℓ] be the simultaneous bands of Theorem E.7. Define
At(m) := |{ℓ̸= m : bL(m)
t,ℓ> 0}|,
Bt(m) := |{ℓ̸= m : bU (m)
t,ℓ
< 0}|,
and bRt(m) := [1 + At(m), dm −Bt(m)].
Theorem E.8 (Rank confidence band for one task; restatement of Theorem 5.1). Under
the
conditions of Theorem E.7,
Pr
 
rkt(m) ∈bRt(m)
	
≥1 −α −o(1).
Proof.
On the simultaneous coverage event of Theorem E.7 applied to Jt,m, every ℓwith bL(m)
t,ℓ> 0 satisfies
∆(m)
t,ℓ> 0, i.e. ℓis certified above m, so rkt(m) ≥1+At(m). Symmetrically, every ℓwith bU (m)
t,ℓ
< 0 is certified
below m, so rkt(m) ≤dm −Bt(m). Combining, the simultaneous coverage event implies rkt(m) ∈bRt(m). The
probability bound follows.
56


<!-- page 57 -->
E.9.
Application 2: rank confidence band for one model, all tasks
For a fixed model m, enlarge to J (m) := {(t,ℓ) : t ∈[dt],ℓ̸= m}, so p = dt(dm −1) ≤¯d2. This family is still
polynomial in ¯d, so all conditions of Appendix E.2 are met with the same scaling.
Corollary E.9 (Simultaneous taskwise rank inference; restatement of Corollary 5.2).
Under the conditions of Theorem E.8, with the bootstrap maximum taken over J (m),
Pr
n
rkt(m) ∈bRt(m) ∀t ∈[dt]
o
≥1 −α −o(1).
Proof.
Apply Theorem E.7 to the larger family J (m); the inversion of Theorem E.8 now holds simulta-
neously for every t ∈[dt] under the same simultaneous coverage event. The price relative to Application 1 is
only the additional
p
log ¯d inflation in c∗
1−α, already absorbed in the polylog.
E.10.
Application 3: simultaneous top-K set inference
For inference on the entire task-specific top-K set, enlarge to
Jall := {(t,m,ℓ) : t ∈[dt], m,ℓ∈[dm], ℓ̸= m},
p = dtdm(dm −1) ≤¯d3.
Each test in Jall corresponds to the studentized score-gap statistic for the contrast Γt,m,ℓ= et(eℓ−em)⊤.
Define inner and outer top-K sets by
c
S⋆
Kin(t) :=
 
m : |{ℓ̸= m : bLt,m,ℓ> 0}| ≥dm −K
	
,
c
S⋆
Kout(t) :=
 
m : |{ℓ̸= m : bUt,m,ℓ< 0}| < K
	
.
Theorem E.10 (Simultaneous top-K set inference). Under the conditions of Theorem E.7 applied
to Jall,
Pr
n
c
S⋆
Kin(t) ⊆S⋆
K(t) ⊆c
S⋆
Kout(t) ∀t ∈[dt]
o
≥1 −α −o(1).
Proof.
On the simultaneous coverage event for Jall, m ∈c
S⋆
Kin(t) implies that for every ℓwith bLt,m,ℓ> 0,
the corresponding gap is positive, i.e. Θ⋆
t,ℓ> Θ⋆
t,m. This means ℓis certified above m; since at most K −1
competitors are not certified above m (else |{ℓ̸= m : bLt,m,ℓ> 0}| < dm −K), rkt(m) ≤K and m ∈S⋆
K(t).
Symmetrically, m /∈c
S⋆
Kout(t) implies that at least K competitors are certified below m, so rkt(m) ≥dm −
K + 1 > dm −K i.e. rkt(m) > dm −K, which means m is in the bottom K and hence m ∈S⋆
K(t) only if
K ≥dm −K + 1. Reformulating, contrapositive: m ∈S⋆
K(t) iff rkt(m) ≤K, and the certified-not-top-K set
(∁c
S⋆
Kout(t)) only contains models with at least K certified-below competitors. Hence S⋆
K(t) ⊆c
S⋆
Kout(t) on
the coverage event. The simultaneous validity over t follows from the simultaneous coverage of Jall at level
1 −α −o(1).
E.11.
Critical-value vs. dimension discussion
The bootstrap critical value c∗
1−α is the (1−α)-quantile of a maximum of p correlated approximately standard-
normal coordinates. In the worst case (weakly dependent or independent), c∗
1−α ≍
p
2log(2p/α); under strong
correlation, it can be substantially smaller. Since p ∈{dm −1,dt(dm −1),dtdm(dm −1)} ≤¯d3, log p ≲log ¯d
and therefore c∗
1−α = O(
p
log ¯d). Combining with Σjj = bσ2
j ≍¯d (Lemma D.14), the worst-case simultaneous
band width is of order c∗
1−αbσj/√n ≍
p ¯dlog ¯d/n, matching the rate predicted by the entrywise estimation
theorem (Theorem 3.1) up to logarithmic factors.
57


<!-- page 58 -->
Appendix F:
Cross-reference table and proof map
For reviewer convenience, we summarize the correspondence between labelled main-text results and their
appendix proofs.
Main-text result
Statement
Appendix proof
Theorem 3.1
uniform entrywise estimation
Appendix B.8 (Theorem B.26)
convex initialization
Frobenius rate
p
r ¯d3 log ¯d/n
Appendix B.1.6 (Theorem B.5)
row-wise refinement
ℓ2,∞-bound on left factor
Appendix B.5 (Proposition B.15)
column-wise refinement
ℓ2,∞-bound on right factor
Appendix B.7 (Proposition B.25)
Proposition 3.2
taskwise top-K Hamming
Appendix C.2 (Proposition C.1)
Theorem 4.1
joint efficient CLT, fixed q
Appendix D.9 (Theorem D.10)
single-contrast remainder remainder ≤CA ¯dlogc ¯d/n
Appendix D.6 (Theorem D.5)
uniform remainder
remainder ≤CA
p ¯dlogc /n over F
Appendix D.7 (Theorem D.6)
joint efficiency
Loewner lower bound ¯Σ ⪰Σ
Appendix D.3 (Proposition D.2)
variance consistency
relative-error O(rn)
Appendix D.10 (Proposition D.11)
covariance consistency
correlation-error O(rn)
Appendix D.11 (Proposition D.12)
Theorem 5.1
rank confidence band for one task
Appendix E.8 (Theorem E.8)
Corollary 5.2
simultaneous taskwise rank inference
Appendix E.9 (Corollary E.9)
Top-K set extension
inner/outer top-K confidence sets
Appendix E.10 (Theorem E.10)
58