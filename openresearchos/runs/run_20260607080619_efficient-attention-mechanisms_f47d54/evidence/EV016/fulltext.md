<!-- page 1 -->
Robust Filter Attention: Self-Attention as Precision-Weighted State Estimation
Peter Racioppo 1
Abstract
We introduce Robust Filter Attention (RFA), a
formulation of self-attention as a robust state es-
timator. Each token is treated as a noisy obser-
vation of a latent trajectory governed by a lin-
ear stochastic differential equation (SDE), and
attention weights are determined by consistency
under this model rather than static feature similar-
ity. Under isotropic noise and decay assumptions,
RFA matches the computational complexity of
standard attention. On language modeling bench-
marks, RFA achieves lower perplexity than RoPE
within the training window while remaining stable
under zero-shot extrapolation to longer contexts.
The framework also provides a dynamical inter-
pretation of standard positional mechanisms, con-
necting rotational embeddings and recency biases
to transport and uncertainty propagation induced
by stochastic dynamics.
1. Introduction
Self-attention has become the dominant paradigm for se-
quence modeling due to its parallelism and scalability
(Vaswani et al., 2017). Unlike recurrent architectures (El-
man, 1990), however, it does not explicitly propagate latent
states through shared temporal dynamics — each token at-
tends independently to all others, with no constraint that
states evolve consistently over time. Temporal structure
is therefore encoded through positional mechanisms rather
than state evolution.
We introduce Robust Filter Attention (RFA), which grounds
such structure in an explicit stochastic dynamical prior: past
tokens are propagated to the query position under learned
linear dynamics, and attention weights are determined by
their predicted reliability under this model, coupling trans-
port and uncertainty propagation.
1Independent Researcher, Los Angeles, CA, USA. Correspon-
dence to: Peter Racioppo <pcracioppo@gmail.com>.
Proceedings of the 43 rd International Conference on Machine
Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026
by the author(s).
In RFA, each token is modeled as a noisy observation of
a latent trajectory. The query token serves as a reference
observation of the current state, while past keys are propa-
gated to the query position under learned dynamics, each
yielding a prediction of the current latent state with an as-
sociated precision determined by the Differential Lyapunov
Equation (DLE). Attention weights are then computed from
prediction errors measured under these precisions, so that
tokens are weighted according to their consistency with the
dynamical model. In this view, attention arises as precision-
weighted state estimation under a stochastic dynamical prior.
Unlike recursive filters such as the Kalman filter, RFA per-
forms this estimation independently at each query position,
yielding a parallel batch estimator compatible with standard
attention-style computation.
Signal magnitude and reliability are governed separately:
dynamical decay attenuates transported states, while DLE-
derived precision tracks accumulated uncertainty. Depend-
ing on the learned balance between process and measure-
ment uncertainty, different heads may specialize into dis-
tinct temporal filtering behaviors, ranging from short-range
recency-biased filtering to stable long-range integration.
Common positional encodings can be understood as impos-
ing implicit dynamical models on how information evolves
across tokens. RoPE (Su et al., 2024) encodes transport
through norm-preserving rotations but assigns no explicit
reliability to tokens as a function of distance. ALiBi (Press
et al., 2022) imposes a distance-based reliability penalty
but carries no corresponding transport operator. In RFA,
transport and reliability are not independent design choices
but consequences of a shared dynamical model. RFA un-
der noiseless, decay-free dynamics recovers RoPE, while
pure diffusion with no decay yields a logarithmic distance
penalty analogous to ALiBi.
Finally, we introduce Spectrally-Coupled RFA (SC-RFA).
Standard rotational encodings such as RoPE apply no decay,
so high-frequency components persist indefinitely. As a re-
sult, tokens far apart in the sequence can appear positionally
similar to nearby ones, degrading long-context discrimina-
tion. SC-RFA addresses this by partitioning the frequency
spectrum across attention heads and coupling each head’s
decay rate to its maximum frequency, so that high-frequency
heads act as short-range filters while low-frequency heads
1
arXiv:2509.04154v6  [cs.LG]  22 May 2026


<!-- page 2 -->
Robust Filter Attention
behave as stable long-range integrators.
Our contributions are as follows:
(i) A derivation of self-attention as a tractable robust batch
estimator for a linear SDE.
(ii) A scalable isotropic formulation with analytic uncer-
tainty modeling at standard attention cost.
(iii) A recovery of RoPE and ALiBi as limiting cases, cor-
responding respectively to noiseless, zero-decay dynamics
and Brownian diffusion.
(iv) A spectrally-coupled decay prior enabling multi-
resolution temporal filtering across heads, with principled
suppression of long-range phase interference.
2. Related Work
2.1. Probabilistic and Kernel Views of Attention
The Transformer architecture (Vaswani et al., 2017) com-
putes attention scores via a scaled dot-product between
queries and keys. While originally motivated by its effi-
ciency and ability to capture long-range dependencies, a
growing body of work has sought to interpret these weights
as probabilities derived from latent statistical models.
Probabilistic Transformers (Gabbur et al., 2021) show that
dot-product attention arises as a constrained limit of MAP
inference in a Gaussian mixture model. The Bayesian At-
tention Mechanism (BAM) (Bianchessi et al., 2026) treats
positional embeddings as explicit priors over token indices,
while the Correlated Gaussian Process Transformer (CGPT)
(Bui et al., 2024) interprets asymmetric projections through
correlated Gaussian process inference. These approaches
introduce probabilistic structure, but rely on static feature-
space similarity or fixed prior covariances.
Other work has interpreted attention as a kernel regression
estimator, modifying similarity geometry in feature space to
improve robustness or variance properties (Tsai et al., 2019;
Han et al., 2023; Nielsen et al., 2024; Liu et al., 2020; Goel
& Bartlett, 2024). From this perspective, dot-product atten-
tion is a kernel smoother in which the kernel is determined
by learned feature similarity, treating all tokens as equally
reliable regardless of temporal lag. In contrast, RFA is not
constructed by specifying or learning a similarity function.
Instead, attention weights arise from a state estimation for-
mulation under a shared dynamical model — the kernel over
token pairs is a consequence of the underlying stochastic
dynamics rather than the primary object of design.
2.2. Filtering, Continuous Dynamics, and SSMs
Continuous-time sequence models often parameterize la-
tent dynamics using Neural Ordinary Differential Equations
(Neural ODEs) (Chen et al., 2018) and their stochastic ex-
tensions, Neural SDEs (Li et al., 2020; Shen & Cheng,
2025), which learn drift and diffusion functions from data.
Several architectures integrate attention with continuous
dynamics to handle irregular sampling or time-dependent
relevance, including Continuous-Time Attention (Chien &
Chen, 2021), Attentive Neural Processes (Kim et al., 2019),
and ACE-NODE (Jhin et al., 2021). Self-Modulating Atten-
tion (SMA) (Chen et al., 2021) adjusts attention weights as
a function of temporal distance.
Other work integrates neural networks with classical filter-
ing frameworks by learning components of the Kalman filter,
such as gains, noise models, or update rules (Jahanshahi &
Zhu, 2026; Revach et al., 2022; Liu et al., 2023; Cohen &
Klein, 2025; Shen et al., 2025). In contrast, RFA assumes
a structured linear SDE whose DLE admits a closed-form
solution, enabling parallel precision-weighted aggregation
without learning Kalman gains or full covariance updates.
State space models (SSMs) provide another approach to se-
quence modeling by assuming linear time-invariant (LTI) dy-
namics and converting recurrence into convolution. Frame-
works such as HiPPO (Gu et al., 2020) and S4 (Gu et al.,
2022) achieve efficiency by restricting the dynamics to struc-
tured forms (e.g., diagonalizable or diagonal-plus-low-rank),
reducing the cost of state updates from O(d2) to O(d) and
enabling fast convolutional implementations. Recent work
shows that causal linear attention can be viewed as a special
case of LTI convolution (Dao & Gu, 2024).
Concurrent with our work, Kalman Linear Attention (KLA)
(Shaj et al., 2026) develops a probabilistic filtering inter-
pretation for linear attention and state-space sequence mod-
els using parallelized information-form Kalman updates.
Whereas KLA and recurrent SSMs propagate a compressed
hidden state through recursive updates, RFA instead formu-
lates self-attention as a parallel precision-weighted batch
estimation problem, where pairwise uncertainties are ana-
lytically derived from stochastic dynamics rather than re-
cursively propagated through posterior state updates, while
preserving explicit content-based token interactions through
attention.
2.3. Positional Encodings and Complex Geometry
Modeling relative temporal structure in Transformers has
been approached with several geometric methods. RoPE
(Su et al., 2024) encodes relative position through determin-
istic complex rotations of queries and keys, but introduces
no explicit notion of decay or uncertainty. ALiBi (Press
et al., 2022) applies a linear distance-based bias to atten-
tion logits, improving length extrapolation by suppressing
distant interactions. xPos (Sun et al., 2023b;a) generalizes
RoPE by combining rotations with dimension-wise decay
to stabilize long-range behavior.
2


<!-- page 3 -->
Robust Filter Attention
These methods impose useful geometric or monotonic struc-
ture, but are not derived from an explicit model of latent state
evolution and measurement. As a result, transport and relia-
bility are treated as separate design choices. Each method
can be interpreted as corresponding to an implicit dynamical
assumption. RoPE implements deterministic phase trans-
port without uncertainty accumulation, corresponding to a
noiseless LTI system with zero decay. ALiBi’s linear dis-
tance penalty approximates RFA’s DLE-derived bias under
pure Brownian diffusion, giving it an implicit noise model
but no transport operator. xPos introduces decay without
deriving it from a covariance model, leaving transport and
reliability decoupled.
RFA derives both the transport operator and the precision
weighting from the same underlying SDE, so that these
are coupled by construction rather than introduced indepen-
dently. This also implies a rotate–aggregate–counter-rotate
structure on the value stream, required for aggregation to
correspond to fusion of state estimates in a shared temporal
frame. While methods such as RoPER (Harik & Jayasiri,
2022) have applied value rotations previously, RFA derives
this structure as a necessity of the dynamical model rather
than a geometric heuristic.
Methods such as YaRN (Peng et al., 2024) address RoPE’s
extrapolation failure by compressing position indices at in-
ference, mitigating out-of-distribution phase rotations post-
hoc while introducing a train-inference mismatch. SC-RFA
instead prevents the failure by coupling each head’s decay
rate to its maximum frequency during training, so that high-
frequency modes attenuate before accumulating spurious
phase matches. Selective RoPE (Movahedi et al., 2026) also
combines rotation with decay via a spectral leakage analogy,
but lacks a precision kernel, value rotation structure, and
principled frequency-decay coupling — components that in
RFA are necessary consequences of the dynamical model
rather than design choices.
3. Methods
RFA models each token as a noisy observation of a latent tra-
jectory evolving under a linear SDE. Keys are transported to
the query position under the dynamics model, while the Dif-
ferential Lyapunov Equation provides an analytic estimate
of the uncertainty accumulated during transport. Attention
weights are then determined by prediction consistency under
this uncertainty model, yielding a robust precision-weighted
estimator. Under isotropic noise and decay assumptions,
the uncertainty model reduces to a scalar function of tempo-
ral lag, recovering standard O(N 2d) attention complexity.
Further details are provided in Appendix A and Appendix B.
3.1. Attention as Factorized State Estimation
Generative model.
We model each token embedding zi ∈
Rd as a noisy observation of a latent state evolving under a
linear time-invariant SDE:
dx(t) = A x(t) dt + G dw(t),
zi = Cx(ti) + vi,
vi ∼N(0, R),
(1)
where w(t) is a standard Wiener process, vi is Gaussian
measurement noise with covariance R, and Q := GG⊤is
the process noise covariance. We assume C = I without
loss of generality, since any invertible observation map may
be absorbed into a change of basis; non-invertible maps
correspond to partially observed systems, which we do not
consider here.
This model is not intended to represent the true data-
generating process. Rather, it defines a shared dynamical
prior that specifies both how key embeddings are transported
to the query position and how their reliability evolves. We
adopt an LTI structure because it admits closed-form propa-
gation of both state means and covariances.
Estimation setup.
Given a query at position i, our goal
is to estimate the latent state x(ti) from past observations
{zj}j≤i. This estimation problem is solved independently
for each query position, yielding a parallel batch estimator
rather than a recursive filter. At this stage, queries and keys
are not separate representations, but roles assigned to the
same embeddings. Each past embedding zj serves as a key:
it is propagated forward via the state transition matrix to
form a prediction of the current latent state,
ˆzij = eA∆tijzj,
where ∆tij = ti −tj. The query embedding zi serves as
a reference observation against which these predictions are
compared.
Under the SDE, the transported key is distributed as:
ˆzij ∼N
 x(ti), ˆV ij
 
,
where the covariance captures both accumulated process
noise and the measurement noise of the source token:
ˆV ij = V (∆tij) + eA∆tijR eA⊤∆tij.
Here, V (∆t) is the solution of the Differential Lyapunov
Equation (DLE):
˙V (s) = AV (s) + V (s)A⊤+ Q,
V (0) = 0.
(2)
Query-key residual.
Because the query serves as a ref-
erence rather than a transported prediction, we assign it a
3


<!-- page 4 -->
Robust Filter Attention
separate covariance RΓ, independent of R, yielding a two-
sided measurement model with distinct noise on the query
and key sides.
The residual between query zi and transported key ˆzij is:
rij := zi −ˆzij.
The residual covariance is the sum of the key-side and query-
side uncertainties:
rij ∼N
 
0, Σij
 
,
Σij =
ˆV ij
|{z}
key-side
+
RΓ
|{z}
query-side
,
The positive definite query-side term RΓ ≻0 ensures that
the precision P ij := Σ−1
ij remains bounded as ∆t →0.
The similarity between query zi and key zj is measured by
the squared Mahalanobis distance:
d2
ij = r⊤
ijP ijrij,
(3)
which replaces dot-product similarity with a consistency
test under the precision prior given by the DLE.
Latent state estimation.
The optimal estimator for xi
under the full joint model requires inverting a dense cross-
token covariance matrix, which is not parallelizable. To
obtain a tractable estimator, we adopt a conditional indepen-
dence approximation, neglecting cross-token correlations in-
duced by shared process noise while retaining the marginal
covariance of each transported observation (Appendix A.3).
We then estimate the latent state by minimizing a sum of
squared Mahalanobis residuals:
¯zi = arg min
z
X
j≤i
(z −ˆzij)⊤P ij(z −ˆzij).
This yields the closed-form estimator:
¯zi =
  X
j≤i
P ij
 −1 X
j≤i
P ij ˆzij.
(4)
This approximation is exact when process noise is absent.
In general, shared process noise induces cross-token corre-
lations that the factorized likelihood ignores, causing corre-
lated evidence to be overcounted. However, the marginal
covariance of each transported observation is preserved, so
the resulting precision weights remain individually consis-
tent with the uncertainty predicted by the SDE. The approx-
imation therefore sacrifices globally consistent inference in
exchange for a tractable parallel attention-style estimator
with structured uncertainty propagation.
Robust reweighting.
The above estimator assumes resid-
uals are well-described by the predicted covariance. To
improve robustness under model mismatch, we instead use
a heavy-tailed likelihood over the Mahalanobis distance d2
ij,
yielding a robust penalty ρ(d2
ij), as in M-estimation.
Minimizing this loss yields data-dependent influence
weights:
wij ∝
∂
∂d2
ij
ρ(d2
ij),
which re-weight the precisions:
P ij →wij P ij,
reducing the influence of tokens whose residuals are unex-
pectedly large under the predicted covariance.
Two standard choices are:
wij ∝





exp
 
−
d2
ij
ν
 
(exponential)
 
1 +
d2
ij
ν
 −κ
(power law)
(5)
where ν governs the tail weight of the influence function.
The exponential form recovers the standard dot-product at-
tention structure, while the power-law form yields a heavier-
tailed robust variant.
Closed-form parallel computation.
To evaluate P ij in
closed form for all token pairs simultaneously, we require
that the system matrices be simultaneously diagonalizable
by some S ∈Cd×d: A = SΛS−1, Q = SΛQS†, R =
SΛRS†, RΓ = SΛΓS†. Under this assumption, the DLE
decouples into independent scalar ODEs for each eigenmode
Σij = SΛΣ,ijS†, yielding a diagonal covariance in this
basis. Letting λ(·) = diag(Λ(·)),
λΣ,ij = λQ ⊙1 −e2 Re(λ) ∆tij
−2 Re(λ)
+ e2 Re(λ) ∆tij ⊙λR + λΓ.
The precision is obtained by diagonal inversion, λP,ij =
1 ⊘λΣ,ij. Since the precision is diagonal in this basis, the
Mahalanobis distance decomposes into independent scalar
contributions per mode. Letting zs,i = S−1zi, the robust
precision-weighted aggregation takes the form:
¯zs,i =
X
j≤i
Aij ⊙ˆzs,ij,
ˆzs,ij = eΛ∆tijzs,j,
Aij := wijλP,ij ⊘
  X
j′≤i
wij′λP,ij′
 
,
When the eigenvalues of A are purely imaginary, eΛ∆t
reduces to element-wise rotations, as in RoPE.
3.2. Robust Filter Attention Mechanism
We instantiate the robust state estimator as a complex-valued
attention layer by identifying the abstract diagonalization
4


<!-- page 5 -->
Robust Filter Attention
matrices with learned linear projections. The input pro-
jections W q, W k, W v ∈Cd×d learn the transformation
into the diagonalizing basis of the SDE’s system matrices,
absorbing the inverse diagonalizing matrix S−1, while the
output matrix W o absorbs S, mapping the filtered estimates
back to the original basis:
Q = W q Z,
K = W k Z,
V = W v Z
∈Cd×N.
To preserve the O(N 2 + Nd) memory complexity of stan-
dard attention, we impose isotropic decay and noise in the
learned eigenbasis (per head):
Λ = −µI + iΛΩ, ΛQ = σ2I, ΛR = η2I, ΛΓ = γ2I,
where µ, σ2, η2, γ2 ∈R+ and ΛΩ∈Rd×d is diagonal with
kth diagonal entry ωk. This removes the ability to model
dimension-dependent noise, but preserves the temporal de-
pendence of uncertainty on lag. These definitions ensure
marginally stable dynamics and positive semi-definite noise
covariances.
Under isotropic decay and noise, each eigenmode follows
independent exponentially decaying rotations with decay
rate µ and angular frequency ωk. This yields simple element-
wise rotation factors for forward/backward propagation, and
a decay kernel that depends only on the time lag ∆tij:
˜Φ
−[k, i] := e−iωkti, ˜Φ
+[k, i] := eiωkti, E[i, j] := e−µ∆tij.
We define rotated queries, keys, and values:
˜
Q := ˜Φ
−⊙Q,
˜
K := ˜Φ
−⊙K,
˜V := ˜Φ
−⊙V .
The isotropic constraints cause the variance to become inde-
pendent of the feature dimension:
Σ∆t[i, j] := ˜σ2 1 −e−2µ∆tij 
+ η2e−2µ∆tij + γ2. (6)
Here, ˜σ2 := σ2
2µ, η2, and γ2 are learned scalar parameters
(per head), corresponding respectively to steady-state pro-
cess uncertainty, historical measurement noise (key-side),
and uncertainty in the query observations; Σ∆t denotes the
scalar variance as a function of lag, distinct from the full
covariance matrix.
Collecting terms,
Σ∆t[i, j] = αe−2µ∆tij+β,
α := η2−˜σ2,
β := γ2+˜σ2.
The behavior of the variance with temporal lag is governed
by the sign of α. When α < 0, process noise dominates
and precision decreases monotonically, yielding a diffusive
regime. When α > 0, measurement noise dominates and
precision increases with lag, corresponding to an integrative
regime. Because α and β are determined by learned scalar
parameters per head, different heads can specialize into
distinct filtering behaviors.
The isotropic constraint allows the Mahalanobis distance for
all pairs (i, j) to be computed by element-wise multiplying
a matrix of scalar precisions P ∆t[i, j] := 1/Σ∆t[i, j] by a
matrix of squared residual norms ∥Rqk[i, j]∥2:
D2[i, j] = P ∆t[i, j] ·
Rqk[i, j]
2,
where the ijth residual is:
Rqk[i, j] := ˜
Qi −E[i, j] · ˜
Kj.
The squared residual norm decomposes into a query magni-
tude term, a decayed key magnitude term, and a cross-term
containing the complex inner product:
Rqk[i, j]
2 = ∥Qi∥2 + E[i, j]2 · ∥Kj∥2
−2 E[i, j] · Re
 
˜
Q
†
i ˜
Kj
 
.
(7)
Choosing the power-law form for wij in Eq. 5 yields the
following attention logits:
L = log(P ∆t) −κ log
 
1 + 1
ν P ∆t ⊙
Rqk
2
 
,
(8)
where setting κ = ν+d
d , and ν = νsd for νs ∈R+ yields a
dimension-free logit equivalent to a dimension-normalized
Student-t log-likelihood.
The attention matrix is then ˆ
A = A ⊙E, where:
A = Softmaxj
 βsL + M causal
 
,
where βs is an additional inverse temperature parameter and
M causal ∈{0, −∞}N×N is a causal mask.
The filtered value estimate is computed by aggregating the
rotated values and rotating the result back into the original
value frame:
¯V = ˜Φ
+ ⊙
  ˜V ˆ
A
⊤ 
.
This rotate–aggregate–counter-rotate structure is required
by the dynamical model: values from different time steps
must be brought into a common temporal frame before
aggregation, and the counter-rotation restores the output to
the original frame.
The attention layer then computes an innovation step in the
value basis,
∆V = ¯V −V ,
which represents a correction from the current value toward
the filtered estimate. This correction is projected back into
the original basis and added to the residual stream:
Z+ = Z + W o∆V .
Under the isotropic constraint, RFA preserves the asymp-
totic complexity of standard attention. The dominant op-
eration remains a single O(N 2d) matrix multiplication to
5


<!-- page 6 -->
Robust Filter Attention
compute the cross-term Re( ˜
Q
† ˜
K). The remaining com-
ponents—the decay kernel E[i, j] and precision kernel
P ∆t[i, j]—are computed via elementwise operations with
O(N 2) cost, and do not change the asymptotic complexity.
3.3. Real-valued implementation
Although RFA is formulated over Cd, all operations reduce
to standard real arithmetic, as detailed in Appendix D.1.
The complex projections W q, W k, W v ∈Cd×d are imple-
mented as real d × 2d matrices and W o as a 2d × d matrix,
while complex rotations reduce to the standard RoPE opera-
tion on paired real and imaginary channels (Appendix D.1).
RoPE corresponds to restricting the eigenvalues of A to the
imaginary axis (µ = 0), so that the complex exponential
eΛ∆t reduces to pure rotation with no decay.
The complete implementation of the Isotropic RFA mecha-
nism is formalized in Algorithm 1 in Appendix D.3. 1
3.4. Iterative refinement across layers
The robust M-estimator defines a fixed-point problem: the
weights wij depend on residuals computed relative to the
unknown latent state, which must be approximated by the
current estimate. A single attention layer performs one such
reweighting step using the previous layer’s output as the
current state estimate, while residual connections implement
a partial step toward each reweighted estimate.
Stacking layers therefore yields an iteratively reweighted
least squares (IRLS)-like procedure, in which observations
are progressively reweighted according to their consistency
with the evolving estimate. This provides an interpretation
of both depth and residual connections in RFA as successive
refinement of the state estimate (Appendix C.3).
3.5. Filtering Behaviors
The attention weights in RFA are determined by two distinct
mechanisms that play complementary roles. The additive
bias B∆t := log(P ∆t) acts as a prior budget allocated
to tokens at each lag, while the multiplicative gate P ∆t
controls the selectivity of the attention — how sharply the
model discriminates between consistent and inconsistent
tokens at that lag. Both are determined by the learned noise
parameters (˜σ2, η2, γ2) through the scalar variance Σ∆t =
αe−2µ∆t + β.
The sign of α = η2 −˜σ2 defines a phase transition between
two qualitatively distinct filtering behaviors:
Diffusive Regime (α < 0). When process noise dominates
(˜σ2 > η2), uncertainty accumulates monotonically with lag.
1A multi-head implementation is available at https://
github.com/PCR-git/Robust-Filter-Attention.
The precision P ∆t acts as a closing gate: selectivity is max-
imal near the diagonal and degrades as lag increases. The
additive bias decays toward a floor of −log(β), implement-
ing a forgetting prior that suppresses distant tokens. These
heads implement a recency bias, analogous to the linear
distance penalties used in ALiBi (Appendix B.4).
Integrative Regime (α > 0). When measurement noise
dominates (η2 > ˜σ2), the precision acts as an opening
gate: selectivity is low near the diagonal and increases with
lag as the initial measurement noise on the transported key
dissipates under the stable dynamics. The additive bias cor-
respondingly starts low and curves upward, implementing a
settling prior that delays commitment until the transported
observation becomes reliable. These heads function as lag-
selective denoising filters, suppressing transient noise to
identify stable historical structure.
Because α is determined by learned scalars per head, dif-
ferent heads can self-organize into different regimes during
training. Explicit functional forms for the bias and gate in
each regime are derived in Appendix B.4.
3.6. Recovery of Standard Positional Encodings
RoPE corresponds to the special case in which dynamics
are noiseless (σ2 = 0, η2 = 0), decay is absent (µ = 0),
and value rotation is omitted. Under these conditions, the
state transition matrix reduces to a complex rotation and the
precision prior is uniform.
If the queries and keys are normalized, the Mahalanobis
distance reduces to a dot product between queries and keys.
In the zero-decay, short-lag limit, the RFA additive bias
reduces to:
B∆t ≈−log(η2 + γ2) −
σ2
η2 + γ2 ∆t,
recovering a linear distance penalty with learned slope and
intercept, consistent with the form of ALiBi’s fixed linear
bias (Appendix B.3.5).
3.7. Spectrally Coupled RFA (SC-RFA)
Standard positional mechanisms such as RoPE utilize a fixed
frequency bank across all heads, allowing high-frequency
oscillations to persist indefinitely. At long horizons, this
leads to phase wrap-around: tokens separated by a full oscil-
lation period produce similar phase configurations, making
them difficult to distinguish based on position alone.
To address this, we introduce Spectrally Coupled RFA (SC-
RFA), in which decay rates are coupled to frequencies across
heads. We partition a global frequency bank Ωmonotoni-
cally across heads, assigning each head h a spectral band
[ωh,min, ωh,max], and couple each head’s decay rate to its
6


<!-- page 7 -->
Robust Filter Attention
maximum frequency:
µh = b · ωh,max,
where b ∈R+ is a dimensionless damping coefficient (Ap-
pendix B.5). This enforces a fixed decay per oscillation
cycle: over one period, the signal is attenuated by a factor
of e−b, directly controlling the trade-off between spectral
resolution and long-range stability.
Each attention head induces a prior temporal response given
by the product of a decay term and a precision term:
P ∆t · e−µ∆t ∝
e−µ∆t
αe−2µ∆t + β ,
where α = η2 −˜σ2 and β = γ2 + ˜σ2.
In the integrative regime (α > 0), precision initially in-
creases with lag as key-side measurement noise dissipates,
competing with exponential decay. The product peaks at a
characteristic lag:
∆t∗= 1
2µ log
 α
β
 
.
Since µh varies across heads, peak locations ∆t∗
h ∝1/µh
span a range of temporal scales, forming a bank of lag-
selective filters. For sufficiently small µ, the peak may lie
beyond the training context length, causing the integrative
profile to appear as a monotonically increasing function
over the training window and inducing an anti-recency bias
that extrapolates poorly. Diffusive heads (α < 0) provide
a fallback by enforcing monotonic decay and stabilizing
long-range behavior.
4. Experimental Evaluation and Ablations
We evaluate whether explicitly modeling uncertainty growth
improves long-context stability while preserving short-
range accuracy, comparing RFA against two widely used
positional baselines derived from deterministic geometry
(RoPE) (Su et al., 2024) and monotonic recency biasing
(ALiBi) (Press et al., 2022), respectively. The per-head
scalar parameters (˜σ2, η2, γ2, νs, βs) are learned entirely
within the training window (L = 512), and evaluated
under zero-shot extrapolation at longer context lengths
(L ∈{512, 1024, 2048, 4096}).
4.1. Experimental Setup
Architecture. All models use a 6-layer Transformer with
h = 8 heads and embedding dimension d = 256. To
ensure comparable model capacity, we apply identical
d →2d →d projections in both RFA and the RoPE/ALiBi
baselines, which represent the mapping between the real and
complex domains (Appendix D.1). RFA introduces only
a small number of additional scalar parameters per head
for noise and robustness, increasing total parameter count
by approximately 0.02%. We employ a pre-norm architec-
ture with an FFN expansion factor of 4. Models are trained
for 15 epochs until convergence using Adam with a cosine
learning rate schedule.
Datasets. We evaluate on WikiText-103, a large-scale word-
level language modeling benchmark derived from Wikipedia
articles and used to measure perplexity and long-context
extrapolation (Merity et al., 2017), and on BabyLM-2025
(Strict), a curated English language modeling corpus used
as a complementary benchmark under the same training and
evaluation protocol (Charpentier et al., 2025).
Ablations. We compare against standard positional base-
lines:
RoPE (B1) and ALiBi (B2), and include two
geometry-only decay variants to isolate the effect of damp-
ing in rotational embeddings: Decayed RoPE (B3), which
applies exponential decay with distance, as in RFA, and
SC-RoPE (B4), which couples decay rates to head-wise fre-
quency bands, as in SC-RFA. These baselines test whether
decay and spectral coupling alone can explain extrapolation
gains, without modeling uncertainty.
We evaluate RFA (M1) and two variants of SC-RFA: one
optimized for in-window performance (M2), with b = 0.05,
and one optimized for long-context extrapolation via in-
creased damping (M3), with b = 5.0. We also include
structural ablations relative to M2, designed to isolate the ef-
fect of its components when removed: the power-law robust
weight wij, replacing it with an exponential weight (M2.1);
the DLE-derived precision prior (M2.2); the multiplicative
gating term P ∆t (M2.3); value rotations (M2.4); all rota-
tions (M2.5); finally, we test a purely rotational, zero decay
and zero noise variant, analogous to RoPER (M2.6).
Full architectural and ablation details are provided in Ap-
pendix E. Analysis of attention maps and noise parameters
are provided in Appendix F.
4.2. Results on Wikitext-103
We evaluate extrapolation by measuring test perplexity on
WikiText-103 at increasing context lengths, after training all
models with a fixed context window of 512 tokens. Results
are shown in Table 1.
RFA variants achieve both stronger local performance and
improved extrapolation relative to RoPE. In particular, SC-
RFA (M2) improves over RoPE by 0.94 PPL at L = 512
and reduces degradation at long horizons, reaching 37.19
PPL at L = 4096 compared to RoPE’s 72.69. M2 (SC-RFA)
outperforms M1 (RFA) across all context lengths.
Unlike RoPE, which degrades monotonically outside the
training window, RFA exhibits a non-monotonic extrap-
7


<!-- page 8 -->
Robust Filter Attention
Table 1. Long-context extrapolation on WikiText-103 (Test PPL).
All models were trained with a fixed context window of 512 tokens.
Model
L=512
L=1024 L=2048 L=4096
RoPE (B1)
28.48
30.94
44.21
72.69
ALiBi (B2)
28.59
27.30
26.54
26.30
Decayed RoPE (B3)
28.45
30.45
40.03
64.54
SC-RoPE (B4)
28.44
30.49
41.06
60.08
RFA (M1)
28.01
27.58
29.99
38.46
SC-RFA (M2)
27.54
26.73
29.46
37.19
SC-RFA (M3)
27.91
26.68
26.37
28.16
Structural Ablations (Relative to M2)
Exp. Weight (M2.1)
27.98
27.16
28.95
33.51
Flat Prior (M2.2)
27.69
28.71
38.11
62.83
No Mult. Gate (M2.3)
27.65
29.01
39.18
57.30
No Value Rot. (M2.4)
30.24
92.08
187.29
463.29
No Rotations (M2.5)
28.58
27.25
26.61
26.83
Pure Rotation (M2.6)
27.97
35.59
69.39
131.29
olation profile: perplexity initially decreases beyond the
training horizon before increasing at longer context lengths.
With higher damping coefficient, SC-RFA (M3) further im-
proves long-context stability, achieving nearly flat perplexity
at up to L = 4096 while maintaining competitive perfor-
mance within the training window. This behavior emerges
under a fixed training protocol without requiring length-
dependent scaling rules or curriculum schedules.
Introducing decay into rotational embeddings (B3) and spec-
trally coupling decay across heads (B4) slows the long-range
degradation of RoPE. However, both geometry-only vari-
ants underperform RFA across all context lengths, indicating
that decay alone is insufficient without explicit uncertainty
modeling. B4 does not improve substantially over B3, indi-
cating that coupling decay to the frequency spectrum alone
provides little additional benefit.
The exponential weight variant of SC-RFA (M2.1) under-
performs the power-law robust weighting at the training
horizon, but achieves lower perplexity at extreme extrapo-
lation lengths. This is consistent with Gaussian likelihoods
imposing stronger quadratic penalties on residuals, which
suppress extreme deviations more aggressively but reduce
sensitivity to small errors when uncertainty is low.
Removing the DLE-derived precision prior (M2.2) leads to
degradation at long horizons, with perplexity increasing to
62.83 at L = 4096, demonstrating that explicit uncertainty
propagation via the DLE is necessary for stable long-context
behavior. Removing the precision term P ∆t from the Ma-
halanobis distance (M2.3) causes degradation within the
training window and worsens extrapolation, indicating that
both the additive and multiplicative precision terms con-
tribute to stability.
Table 2. Sensitivity analysis of the damping coefficient b in SC-
RFA (M2), with RoPE (B1) and ALiBi (B2) as baselines. Results
show Test PPL on WikiText-103 across increasing context lengths.
Damping (b)
L=512
L=1024
L=2048
L=4096
RoPE (B1)
28.48
30.94
44.21
72.69
ALiBi (B2)
28.59
27.30
26.54
26.30
5 × 10−4
27.60
28.88
37.34
51.48
5 × 10−3
27.60
28.71
35.35
43.90
5 × 10−2
27.54
26.73
29.46
37.19
5 × 10−1
27.61
26.38
26.37
29.72
5 × 100
27.91
26.68
26.37
28.16
Eliminating value-space rotation and counter-rotation
(M2.4) causes severe degradation at long context, reach-
ing 463.29 PPL at L = 4096. This is consistent with ag-
gregation no longer corresponding to fusion of latent state
estimates in a shared temporal frame. Removing all rota-
tions (M2.5) degrades short-context performance but yields
strong long-range stability. In this setting, RFA reduces to
a purely distance-dependent bias with logarithmic scaling,
closely matching the behavior and performance of ALiBi.
In the zero-noise, zero-decay, pure rotational setting (M2.6),
perplexity increases sharply with context length. Despite
outperforming RoPE within the training window, value ro-
tations without decay and uncertainty modeling accelerate
long-range degradation rather than preventing it.
Compared to ALiBi, SC-RFA achieves lower perplexity at
the training length (L = 512) and at moderate extrapola-
tion (L = 1024), suggesting improved utilization of fine-
grained temporal structure when uncertainty remains low.
At longer horizons, ALiBi attains lower perplexity by enforc-
ing strict locality, effectively suppressing long-range interac-
tions, while SC-RFA continues to integrate distant context
with attenuated but nonzero precision. This reflects a trade-
off between aggressive locality and uncertainty-weighted
long-range integration.
Table 2 shows that this tradeoff is continuously controlled
in SC-RFA by the damping coefficient b. Smaller values
of b yield slower decay, improving short-context perfor-
mance but leading to faster degradation as context increases.
Larger values of b produce stronger attenuation and more
stable long-range behavior at the cost of reduced short-
range performance. Notably, for sufficiently strong damping
(b ≥0.5), SC-RFA outperforms ALiBi at intermediate hori-
zons (L = 2048), with ALiBi retaining an advantage only
at the largest tested context length.
4.3. Results on BabyLM-2025
We use the same architectures, hyperparameters, and train-
ing protocol as on WikiText-103. On BabyLM-2025, where
8


**[Table p8.1]**
| RFA (M1) | 28.01 | 27.58 | 29.99 | 38.46 |
| --- | --- | --- | --- | --- |
| SC-RFA (M2) | 27.54 | 26.73 | 29.46 | 37.19 |
| SC-RFA (M3) | 27.91 | 26.68 | 26.37 | 28.16 |

[CAPTION] Table 1. Long-context extrapolation on WikiText-103 (Test PPL).

[CAPTION] Table 2. Sensitivity analysis of the damping coefficient b in SC-

[CAPTION] Table 2 shows that this tradeoff is continuously controlled


<!-- page 9 -->
Robust Filter Attention
Table 3. Long-context extrapolation on BabyLM-2025 (Test PPL).
All models were trained with a fixed context window of 512 tokens.
Model
L=512
L=1024
L=2048
L=4096
RoPE (B1)
17.70
18.78
23.33
33.29
ALiBi (B2)
17.70
17.20
17.06
17.51
RFA (M1)
17.51
17.71
20.61
31.04
SC-RFA (M2)
17.36
16.99
18.33
22.25
SC-RFA (M3)
17.51
17.07
17.26
18.74
language modeling performance is more strongly dominated
by short-range context, differences between positional mech-
anisms are smaller at short context lengths. The RFA vari-
ants outperform RoPE at all evaluated context lengths and
outperform ALiBi within the training window. SC-RFA
also achieves lower perplexity than ALiBi at intermedi-
ate context (L = 1024), while ALiBi remains strongest at
the longest horizons due to its strict recency bias. These
results mirror the trade-off observed on WikiText-103: pre-
cision weighting improves robustness over purely rotational
embeddings while retaining stronger short- and mid-range
performance than aggressively local recency biases.
4.4. Learning Dynamics and Head Specialization
RFA variants achieve lower validation perplexity earlier in
training than RoPE and ALiBi, indicating that the SDE-
based prior provides an effective inductive bias for latent
state estimation (Appendix F.1). Analysis of learned noise,
decay, and robustness parameters reveals systematic special-
ization across heads into distinct uncertainty and selectivity
regimes (Appendix F.2).
Attention map visualizations at long context lengths reveal
structured multi-scale behavior in RFA. Compared to RoPE,
attention maps exhibit reduced checkerboard interference
from high-frequency oscillations and more coherent peri-
odic structure, reflecting aggregation in a dynamically con-
sistent frame. Some heads exhibit an integrative regime, in
which attention is initially suppressed near the diagonal and
peaks at a characteristic lag, indicating delayed aggregation
until the latent state estimate stabilizes (Appendix F.3).
SC-RFA sharpens and organizes this structure through spec-
tral coupling, inducing a clear ordering of temporal spe-
cialization across heads. High-decay heads concentrate
attention near the diagonal, while lower-decay heads shift
their mass toward progressively longer temporal offsets, pro-
ducing distinct lag bands. This results in fewer and sharper
periodic bands and a clearer separation between local and
long-range interactions. Together, these patterns support the
interpretation of RFA as learning a structured, uncertainty-
aware temporal filter bank rather than relying solely on
geometric positional bias.
5. Conclusion
We reformulate self-attention as a tractable approximation to
state estimation under a linear stochastic dynamical model,
yielding Robust Filter Attention (RFA). In this formulation,
attention weights reflect uncertainty-aware agreement be-
tween dynamically transported representations rather than
static feature similarity, while preserving the computational
structure of standard attention. RFA recovers existing posi-
tional mechanisms as limiting cases and improves temporal
consistency and long-context behavior.
Tractability requires four structural constraints — a condi-
tional independence approximation, linear time-invariant dy-
namics, simultaneous diagonalizability, and isotropic noise
and decay per head. A central challenge for future work is
relaxing these constraints while retaining tractable covari-
ance propagation and parallel computation.
Other important directions include understanding how these
dynamical and uncertainty-based mechanisms interact with
architectural components such as normalization and depth,
as well as how similar uncertainty-aware formulations can
be incorporated into other sequence modeling architectures.
Impact Statement
This work introduces an uncertainty-aware formulation of
self-attention. We do not identify any ethical concerns be-
yond those generally associated with advances in machine
learning methodology.
References
Bianchessi, A. S., Aguirre, Y. C., Barros, R. C., and
Kupssinsk¨u, L. S. Bayesian attention mechanism: A
probabilistic framework for positional encoding and
context length extrapolation.
In The Fourteenth In-
ternational Conference on Learning Representations,
2026. URL https://openreview.net/forum?
id=dXJB9O8fLd.
Bui, L. M., Huu, T. T., Dinh, D., Nguyen, T. M., and Hoang,
T. N. Revisiting kernel attention with correlated Gaussian
process representation. In Proceedings of the Fortieth
Conference on Uncertainty in Artificial Intelligence, UAI
’24. JMLR.org, 2024.
Charpentier, L., Choshen, L., Cotterell, R., Gul, M. O.,
Hu, M. Y., Liu, J., Jumelet, J., Linzen, T., Mueller,
A., Ross, C., Shah, R. S., Warstadt, A., Wilcox, E. G.,
and Williams, A. Findings of the third BabyLM chal-
lenge: Accelerating language modeling research with
cognitively plausible data. In Charpentier, L., Choshen,
L., Cotterell, R., Gul, M. O., Hu, M. Y., Liu, J.,
Jumelet, J., Linzen, T., Mueller, A., Ross, C., Shah,
9


**[Table p9.1]**
| RFA (M1) | 17.51 | 17.71 | 20.61 | 31.04 |
| --- | --- | --- | --- | --- |
| SC-RFA (M2) | 17.36 | 16.99 | 18.33 | 22.25 |
| SC-RFA (M3) | 17.51 | 17.07 | 17.26 | 18.74 |

[CAPTION] Table 3. Long-context extrapolation on BabyLM-2025 (Test PPL).


<!-- page 10 -->
Robust Filter Attention
R. S., Warstadt, A., Wilcox, E. G., and Williams, A.
(eds.), Proceedings of the First BabyLM Workshop, pp.
399–420, Suzhou, China, November 2025. Association
for Computational Linguistics. doi: 10.18653/v1/2025.
babylm-main.28.
URL https://aclanthology.
org/2025.babylm-main.28/.
Chen, C., Geng, H., Yang, N., Yan, J., Xue, D., Yu, J.,
and Yang, X. Learning self-modulating attention in con-
tinuous time space with applications to sequential rec-
ommendation. In Meila, M. and Zhang, T. (eds.), Pro-
ceedings of the 38th International Conference on Ma-
chine Learning, volume 139 of Proceedings of Machine
Learning Research, pp. 1606–1616. PMLR, 18–24 Jul
2021. URL https://proceedings.mlr.press/
v139/chen21h.html.
Chen, R. T. Q., Rubanova, Y., Bettencourt, J., and Duvenaud,
D. Neural ordinary differential equations. In Proceedings
of the 32nd International Conference on Neural Informa-
tion Processing Systems, NIPS’18, pp. 6572–6583, Red
Hook, NY, USA, 2018. Curran Associates Inc.
Chien, J.-T. and Chen, Y.-H. Continuous-time self-attention
in neural differential equation. In ICASSP 2021 - 2021
IEEE International Conference on Acoustics, Speech and
Signal Processing (ICASSP), pp. 3290–3294, 2021. doi:
10.1109/ICASSP39728.2021.9414104.
Cohen, N. and Klein, I.
Adaptive Kalman-informed
Transformer.
Engineering Applications of Artificial
Intelligence, 146:110221, 2025.
ISSN 0952-1976.
doi:
https://doi.org/10.1016/j.engappai.2025.110221.
URL
https://www.sciencedirect.com/
science/article/pii/S0952197625002210.
Dao, T. and Gu, A. Transformers are SSMs: generalized
models and efficient algorithms through structured state
space duality. In Proceedings of the 41st International
Conference on Machine Learning, ICML’24. JMLR.org,
2024.
Elman,
J. L.
Finding structure in time.
Cog-
nitive
Science,
14(2):179–211,
1990.
doi:
https://doi.org/10.1207/s15516709cog1402\ 1.
URL
https://onlinelibrary.wiley.com/doi/
abs/10.1207/s15516709cog1402_1.
Gabbur, P., Bilkhu, M., and Movellan, J.
Probabilistic
attention for interactive segmentation. In Beygelzimer,
A., Dauphin, Y., Liang, P., and Vaughan, J. W. (eds.),
Advances in Neural Information Processing Systems,
2021. URL https://openreview.net/forum?
id=JpDlWGTBHB.
Goel, G. and Bartlett, P.
Can a Transformer represent
a Kalman filter?
In Abate, A., Cannon, M., Margel-
los, K., and Papachristodoulou, A. (eds.), Proceedings
of the 6th Annual Learning for Dynamics &; Control
Conference, volume 242 of Proceedings of Machine
Learning Research, pp. 1502–1512. PMLR, 15–17 Jul
2024. URL https://proceedings.mlr.press/
v242/goel24a.html.
Gu, A., Dao, T., Ermon, S., Rudra, A., and R´e, C. HiPPO:
recurrent memory with optimal polynomial projections.
In Proceedings of the 34th International Conference on
Neural Information Processing Systems, NIPS ’20, Red
Hook, NY, USA, 2020. Curran Associates Inc. ISBN
9781713829546.
Gu, A., Goel, K., and Re, C.
Efficiently modeling
long sequences with structured state spaces.
In In-
ternational Conference on Learning Representations,
2022. URL https://openreview.net/forum?
id=uYLFoz1vlAC.
Han, X., Ren, T., Nguyen, T. M., Nguyen, K., Ghosh, J., and
Ho, N. Designing robust Transformers using robust kernel
density estimation. In Thirty-seventh Conference on Neu-
ral Information Processing Systems, 2023. URL https:
//openreview.net/forum?id=BqTv1Mtuhu.
Harik, G. and Jayasiri, V.
Rotary positional embed-
dings with relative distances, 2022.
URL http://
research.labml.ai/RoPER.html.
Jahanshahi, H. and Zhu, G. Uncertainty propagation net-
works for neural ordinary differential equations. Neuro-
computing, 677:133134, 02 2026. doi: 10.1016/j.neucom.
2026.133134.
Jhin, S. Y., Jo, M., Kong, T., Jeon, J., and Park, N. ACE-
NODE: Attentive co-evolving neural ordinary differential
equations. In Proceedings of the 27th ACM SIGKDD Con-
ference on Knowledge Discovery & Data Mining, KDD
’21, pp. 736–745, New York, NY, USA, 2021. Associa-
tion for Computing Machinery. ISBN 9781450383325.
doi: 10.1145/3447548.3467419. URL https://doi.
org/10.1145/3447548.3467419.
Kim, H., Mnih, A., Schwarz, J., Garnelo, M., Eslami,
A., Rosenbaum, D., Vinyals, O., and Teh, Y. W. At-
tentive neural processes. In International Conference
on Learning Representations, 2019. URL https://
openreview.net/forum?id=SkE6PjC9KX.
Li, X., Wong, T.-K. L., Chen, R. T. Q., and Duvenaud,
D. K. Scalable gradients and variational inference for
stochastic differential equations. In Zhang, C., Ruiz, F.,
Bui, T., Dieng, A. B., and Liang, D. (eds.), Proceed-
ings of The 2nd Symposium on Advances in Approxi-
mate Bayesian Inference, volume 118 of Proceedings of
Machine Learning Research, pp. 1–28. PMLR, 08 Dec
2020. URL https://proceedings.mlr.press/
v118/li20a.html.
10


<!-- page 11 -->
Robust Filter Attention
Liu, H., Lu, J., Zhao, X., Xu, S., Peng, H., Liu, Y., Zhang,
Z., Li, J., Jin, J., Bao, Y., and Yan, W. Kalman filtering
attention for user behavior modeling in CTR prediction.
In Proceedings of the 34th International Conference on
Neural Information Processing Systems, NIPS ’20, Red
Hook, NY, USA, 2020. Curran Associates Inc. ISBN
9781713829546.
Liu, W., Lai, Z., Bacsa, K., and Chatzi, E. Neural extended
Kalman filters for learning and predicting dynamics of
structural systems. Structural Health Monitoring, 23(2):
1037–1052, June 2023. ISSN 1741-3168. doi: 10.1177/
14759217231179912. URL http://dx.doi.org/
10.1177/14759217231179912.
Merity, S., Xiong, C., Bradbury, J., and Socher, R. Pointer
sentinel mixture models. In International Conference
on Learning Representations, 2017. URL https://
openreview.net/forum?id=Byj72udxe.
Movahedi, S., Carstensen, T., Afzal, A., Hutter, F., Orvieto,
A., and Cevher, V. Selective rotary position embedding.
In The Fourteenth International Conference on Learning
Representations, 2026. URL https://openreview.
net/forum?id=AQo1SEElNb.
Nielsen, S., Abdullaev, L., Teo, R., and Nguyen, T. M.
Elliptical attention. In The Thirty-eighth Annual Con-
ference on Neural Information Processing Systems,
2024. URL https://openreview.net/forum?
id=Ejg4d4FVrs.
Peng, B., Quesnelle, J., Fan, H., and Shippole, E. YaRN: Ef-
ficient context window extension of large language mod-
els. In The Twelfth International Conference on Learning
Representations, 2024. URL https://openreview.
net/forum?id=wHBfxhZu1u.
Press, O., Smith, N., and Lewis, M. Train short, test long:
Attention with linear biases enables input length extrapo-
lation. In International Conference on Learning Represen-
tations, 2022. URL https://openreview.net/
forum?id=R8sQPpGCv0.
Revach, G., Shlezinger, N., Ni, X., Escoriza, A. L., van
Sloun, R. J. G., and Eldar, Y. C. KalmanNet: Neural
network aided Kalman filtering for partially known dy-
namics. IEEE Transactions on Signal Processing, 70:
1532–1547, 2022.
ISSN 1941-0476.
doi: 10.1109/
tsp.2022.3158588. URL http://dx.doi.org/10.
1109/TSP.2022.3158588.
Shaj, V., Barker, C., Scannell, A., Szecsenyi, A., Crowley,
E. J., and Storkey, A. Kalman linear attention: Parallel
Bayesian filtering for efficient language modelling and
state tracking, 2026. URL https://arxiv.org/
abs/2602.10743.
Shen, M. and Cheng, C. Neural SDEs as a unified approach
to continuous-domain sequence modeling, 2025. URL
https://arxiv.org/abs/2501.18871.
Shen, S., Chen, J., Yu, G., Zhai, Z., and Han, P. Kalman-
Former: using Transformer to model the Kalman gain in
Kalman filters. Frontiers in Neurorobotics, 18:1460255,
2025. doi: 10.3389/fnbot.2024.1460255. URL https:
//doi.org/10.3389/fnbot.2024.1460255.
Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., and
Liu, Y.
Roformer:
Enhanced transformer with
rotary
position
embedding.
Neurocomputing,
568:127063,
2024.
ISSN
0925-2312.
doi:
https://doi.org/10.1016/j.neucom.2023.127063.
URL
https://www.sciencedirect.com/
science/article/pii/S0925231223011864.
Sun, Y., Dong, L., Huang, S., Ma, S., Xia, Y., Xue, J.,
Wang, J., and Wei, F. Retentive Network: A successor
to Transformer for large language models, 2023a. URL
https://arxiv.org/abs/2307.08621.
Sun, Y., Dong, L., Patra, B., Ma, S., Huang, S., Benhaim,
A., Chaudhary, V., Song, X., and Wei, F.
A length-
extrapolatable Transformer. In Rogers, A., Boyd-Graber,
J., and Okazaki, N. (eds.), Proceedings of the 61st An-
nual Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), pp. 14590–14604,
Toronto, Canada, July 2023b. Association for Compu-
tational Linguistics.
doi: 10.18653/v1/2023.acl-long.
816. URL https://aclanthology.org/2023.
acl-long.816/.
Tsai, Y.-H. H., Bai, S., Yamada, M., Morency, L.-P., and
Salakhutdinov, R. Transformer dissection: A unified
understanding for Transformer’s attention via the lens
of kernel. In Inui, K., Jiang, J., Ng, V., and Wan, X.
(eds.), Proceedings of the 2019 Conference on Empir-
ical Methods in Natural Language Processing and the
9th International Joint Conference on Natural Language
Processing (EMNLP-IJCNLP), pp. 4344–4353, Hong
Kong, China, November 2019. Association for Compu-
tational Linguistics. doi: 10.18653/v1/D19-1443. URL
https://aclanthology.org/D19-1443/.
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones,
L., Gomez, A. N., Kaiser, L. u., and Polosukhin, I.
Attention is all you need. In Guyon, I., Luxburg, U. V.,
Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S.,
and Garnett, R. (eds.), Advances in Neural Information
Processing Systems, volume 30. Curran Associates, Inc.,
2017.
URL https://proceedings.neurips.
cc/paper_files/paper/2017/file/
3f5ee243547dee91fbd053c1c4a845aa-Paper.
pdf.
11


<!-- page 12 -->
Robust Filter Attention
Appendix Table of Contents
1. Appendix A: Attention from a Factorized State Estimation Formulation
13
• Derives a precision-weighted attention mechanism from a state estimation formulation.
2. Appendix B: Robust Filter Attention Mechanism
17
• Covers the simplification from the full anisotropic tensor formulation of RFA to the scalable isotropic variant.
Also provides a physical interpretation of common positional embeddings, showing how RoPE and ALiBi arise as
limiting cases of the RFA framework.
3. Appendix C: Model Extensions
24
• Describes several extensions, including models with inhomogeneous dynamics, time-dependent noise, and itera-
tively reweighted least squares.
4. Appendix D: Implementation Details
26
• Provides the complex-to-real isomorphism for hardware-efficient computation and full pseudocode for the Isotropic
RFA layer.
5. Appendix E: Experimental Details and Ablations
29
• Defines all ablation variants, training setup, and hyperparameters.
6. Appendix F: Additional Experimental Results
31
• Analyzes learned noise parameters, head specialization, and long-context attention maps to illustrate how RFA
implements multi-scale filtering behavior in practice.
12


<!-- page 13 -->
Robust Filter Attention
A. Attention from a Factorized State Estimation Formulation
Here, we derive the RFA mechanism introduced in Section 3 in greater detail.
We briefly recall the generative model from Section 3.1. The latent state evolves under a linear SDE,
dx(t) = Ax(t) dt + G dw(t),
zi = x(ti) + vi,
where w(t) is a standard Wiener process, vi is Gaussian measurement noise with covariance R, and Q := GG⊤is the
process noise covariance.
Each past token zj is transported to time ti via:
ˆzij = eA∆tijzj,
∆tij = ti −tj.
Under the SDE, this transported observation satisfies:
ˆzij | x(ti) ∼N
 x(ti), ˆV ij
 
,
ˆV ij = V (∆tij) + eA∆tijR eA⊤∆tij,
where the accumulated process noise V (∆tij) is the solution of the DLE (Section A.1). Figure 1 illustrates propagation
through a stable LTI in 2D.
(a) Latent trajectory and noisy observations
(b) Propagated measurements at 5 target points
Figure 1. Illustration of an LTI SDE in 2D. (a) The true trajectory (black) is observed through additive Gaussian measurement noise
vt ∼N(0, R); noisy measurements zi are shown as blue points. (b) For five target times ti, the plot shows the ensemble of transported
observations ˆzij mapped through the transition eA∆tij from all noisy measurements zj. Both forward and backward transported ˆzij are
shown for visualization; estimation is causal and uses only j ≤i.
A.1. Analytical Solution of the Differential Lyapunov Equation (DLE)
For parallel aggregation across all token pairs, we must efficiently solve the DLE for all i, j ∈[1, N]. To obtain an
analytically tractable solution, we assume the system matrices are simultaneously diagonalizable by an invertible S ∈Cd×d,
where A = SΛS−1 and Q = SΛQS†. This assumption corresponds to learning dynamics in a basis of decoupled modes.
The propagated state covariance, V (∆tij), is the solution to the DLE (Eq. 2):
V (∆tij) =
Z ∆tij
0
eAsQeA⊤s ds.
13

[CAPTION] Figure 1. Illustration of an LTI SDE in 2D. (a) The true trajectory (black) is observed through additive Gaussian measurement noise


<!-- page 14 -->
Robust Filter Attention
Transforming to the eigenbasis, the covariance becomes:
V (∆tij) = SΛV (∆tij)S†.
where each diagonal entry of ΛV (∆tij) satisfies the scalar integral:
λV,k(∆tij) = λQ,k
Z ∆tij
0
e(λk+λ∗
k)sds = λQ,k
Z ∆tij
0
e2Re(λk)sds
(where ΛV = diag(λV ) and ΛQ = diag(λQ)). Each mode accumulates noise according to its real decay rate Re(λk).
Evaluating this integral yields the analytical solution φ(λ, λQ, ∆t) (for the causal case ∆t ≥0):
ΛV (∆tij) = diag
 φ(λk, λQ,k, ∆tij)
 d
k=1,
φ(λ, λQ, ∆t) =





λQ
1 −e2Re(λ) ∆t
−2 Re(λ)
,
Re(λ) ̸= 0,
λQ ∆t,
Re(λ) = 0.
A.2. Residual Covariance and Precision
The residual rij = zi−ˆzij has covariance Σij = ˆV ij+RΓ (Section 3.1). Assuming that R = SΛRS† and RΓ = SΛΓS†,
this becomes:
Σij = SΛΣ,ijS†,
ΛΣ,ij = ΛV (∆tij) + e2Re(Λ)∆tijΛR + ΛΓ.
This covariance is bounded for all ∆tij ≥0 if and only if Re(λk) < 0 for all k, i.e., the dynamics are stable. The precision
is then obtained by diagonal inversion:
P ij = S−†ΛP,ijS−1,
ΛP,ij = Λ−1
Σ,ij.
A.3. Precision-Weighted Consensus State Estimation
To estimate the latent state at time ti, we consider the transported observations {ˆzij}j≤i, which are jointly Gaussian under
the linear SDE with cross-covariances induced by shared process noise along the trajectory. The optimal estimator under
this model requires inversion of a dense covariance matrix or sequential inference (e.g., Kalman filtering or smoothing),
both of which retain temporal coupling and therefore do not yield the factorized attention form considered here.
Parallel aggregation requires factorizing the joint likelihood across observations. We adopt a mean-field (conditionally
independent) approximation in which each transported observation is treated as an independent noisy measurement of the
latent state, with its marginal covariance matched to the SDE-derived uncertainty.
Specifically, we treat the latent state xi as a parameter x ∈Rd and model each transported observation as conditionally
independent given x:
p(ˆzij | x) = N(ˆzij; x, Σij) ,
where Σij is the SDE-derived marginal covariance of the residual, capturing both accumulated process noise and observation
noise at the query position, and P ij = Σ−1
ij is the corresponding precision.
Under this approximation, the likelihood factorizes:
p({ˆzij}j≤i | x) ≈
Y
j≤i
p(ˆzij | x),
yielding the precision-weighted least-squares problem:
¯zi = arg min
x
X
j≤i
(x −ˆzij)⊤P ij(x −ˆzij).
14


<!-- page 15 -->
Robust Filter Attention
Setting the gradient to zero gives the precision-weighted batch estimator:
¯zi =
  X
j≤i
P ij
 −1 X
j≤i
P ij ˆzij.
This approximation is exact when Q = 0. As process noise increases, uncertainty accumulates along the trajectory and
induces correlations between transported observations, causing independent evidence to be over-counted. Nevertheless, the
approximation preserves the correct marginal uncertainty for each transported observation, ensuring tokens are weighted
according to their individual predicted reliability under the dynamics.
A.4. Robust Precision Reweighting
The above estimator corresponds to a Gaussian likelihood over residuals. To account for model mismatch, we replace this
with an M-estimator in which the precision prior P ij is reweighted by a scalar function of the Mahalanobis distance:
˜P ij = w(d2
ij) P ij,
d2
ij = r⊤
ijP ijrij.
The reweighted estimator becomes:
¯zi =
  X
j≤i
˜
P ij
 −1 X
j≤i
˜
P ij ˆzij.
The choice of w(·) determines the robustness profile of the estimator. Two standard choices are an exponential, which
corresponds to a Gaussian likelihood and recovers softmax-style attention, and a power-law, which corresponds to a Student-t
likelihood and provides heavier tails with increased robustness to outliers:
wij ∝





exp
 
−
d2
ij
ν
 
(exponential)
 
1 +
d2
ij
ν
 −κ
(power law)
Since d2
ij depends on the unknown latent state, the weights wij are implicitly functions of ¯zi, yielding a fixed-point equation
interpretable as one step of iteratively reweighted least squares (IRLS). This connection is developed in Appendix C.3,
where stacked attention layers are interpreted as unrolling this iterative procedure across depth.
A.5. Parallel Aggregation via Diagonalization
To obtain a scalable implementation, we transform the robust precision-weighted average to the diagonalized basis. We
define the state and propagated measurements in this basis as:
zs,i := S−1zi,
ˆzs,ij := eΛ∆tij zs,j,
and the corresponding residual:
rs,ij = zs,i −ˆzs,ij.
Using the simultaneous diagonalization, P ij = S−† ΛP,ij S−1, and the Mahalanobis distance decomposes into independent
scalar components:
d2
ij = r†
s,ijΛP,ijrs,ij =
d
X
k=1
λP,ij,k
  rs,ij,k
  2.
where λP,ij,k and rs,ij,k are the kth diagonal components of ΛP,ij and rs,ij, respectively.
This allows the robust weights wij = w(d2
ij) to be computed efficiently for all token pairs.
Applying the aggregation in this basis yields:
¯zs,i =
  X
j≤i
wij ΛP,ij
 −1 X
j≤i
wij ΛP,ij ˆzs,ij.
Since all matrices are diagonal, both the sum and inverse are element-wise operations.
15


<!-- page 16 -->
Robust Filter Attention
Writing λP,ij := diag(ΛP,ij), we define attention weights as:
˜
Aij := wij λP,ij,
Aij = ˜
Aij ⊘
  X
j′≤i
˜
Aij′
 
,
where ⊘denotes element-wise division. The aggregation then takes the form:
¯zs,i =
X
j≤i
Aij ⊙ˆzs,ij,
where, instead of scalar attention weights, each Aij is a vector that element-wise multiplies ˆzs,ij.
Finally, the output in the original coordinate system is recovered by ¯zi = S ¯zs,i. All operations are O(d) per token pair,
yielding an overall complexity of O(N 2d) with no matrix inversions.
Equivalently, this normalization can be written in Softmax form by defining dimension-wise attention logits. For exponential
reweighting wij ∝exp(−d2
ij/ν):
Aij,k =
exp(Lij,k)
P
j′≤i exp(Lij′,k),
Lij,k := log λP,ij,k −1
ν λP,ij,k
  rs,ij,k
  2.
For the power law influence function wij ∝(1 + d2
ij/ν)−κ, the corresponding logit becomes:
Lij,k := log λP,ij,k −κ log
 
1 + 1
ν λP,ij,k|rs,ij,k|2 
,
In the isotropic case, where λP,ij,k = λP,ij is a shared scalar across dimensions, both kernels reduce to scalar logits. In the
exponential case,
Lij = log(λP,ij) −1
ν λP,ij∥rs,ij∥2,
which is a Softmax normalization over scalar similarity scores. This shows that attention weights arise as normalized
likelihood scores under the SDE-induced uncertainty model.
Remark. The normalization term
  P
j′≤i ˜
Aij′
 −1
corresponds to the approximate posterior covariance in the diagonalizing
basis under the CI assumption, providing a measure of the model’s confidence in the aggregated state estimate. This quantity
is a byproduct of the formulation that goes unused in standard attention, and may be useful beyond weighting — for instance,
to gate updates or signal when the context provides insufficient evidence. We leave exploration of such uses to future work.
A.5.1. STUDENT-T LIKELIHOOD ATTENTION (ISOTROPIC CASE)
Under an isotropic covariance constraint Σij = Σ2(∆tij)Id, where Σ2(∆tij) ∈R+, the robust precision-reweighted
attention score may be written as:
Lij = −log
 Σ2(∆tij)
 
−κ log
 
1 + 1
ν d2
ij
 
,
where d2
ij = ∥rij∥2/Σ2(∆tij) is the isotropic Mahalanobis distance.
Choosing κ = ν+d
d
makes this expression proportional (up to additive constants) to the dimension-normalized negative
log-likelihood of an isotropic multivariate Student’s t distribution:
Lij = −log
 Σ2(∆tij)
 
+ ν+d
d
log
 
1 + 1
ν d2
ij
 
.
Setting ν = νsd, this becomes:
Lij = −log
 Σ2(∆tij)
 
+ (νs + 1) log
 
1 +
1
νsdd2
ij
 
.
The dimension normalization is critical for stability: in high-dimensional spaces, squared Mahalanobis distances concentrate
around their expected value of d, causing unnormalized likelihoods to produce overly sharp, near-deterministic weights. The
exponent κ = (ν + d)/d is the unique value ensuring that the influence function treats a residual of typical size d2
ij ∼d
consistently regardless of dimension, preserving sensitivity to relative consistency rather than absolute norm. Under this
choice, robust precision-weighted filtering is equivalent, up to additive constants that are absorbed by softmax normalization,
to using dimension-normalized Student-t log-likelihoods as attention logits.
16


<!-- page 17 -->
Robust Filter Attention
B. Robust Filter Attention Mechanism
We now instantiate the state estimation formulation as an attention mechanism, proceeding from the full anisotropic tensor
formulation implied by the derivation to the memory-efficient isotropic variant used in practice.
B.1. Anisotropic Tensor RFA
Under diagonalizable dynamics, the most general RFA formulation propagates and weights each feature dimension
independently, yielding an O(N 2d) attention tensor. This is not memory-scalable but serves as the reference from which the
isotropic variant is derived.
Learned Change-of-Basis Projections.
The transformation to the decoupled eigenbasis is learned through complex-valued
projections. We define:
W q, W k, W v, W o ∈Cd×d,
where d is the embedding dimension. The input projections {W q, W k, W v} parameterize the learned diagonalizing basis
S−1, mapping the input into the eigenbasis of A where the DLE is analytically solvable, while the output projection W o
parameterizes S, mapping the filtered state estimate back into the original embedding space.
Given an input sequence Z ∈Rd×N, we obtain query, key, and value representations as usual:
Q = W qZ,
K = W kZ,
V = W vZ.
Key and Value Propagation.
In this reference model, every feature k possesses its own complex eigenvalue λk. We
define the propagation tensors E and the resulting propagated keys and values:
E[k, i, j] = eλk(ti−tj),
ˆK[k, i, j] = E[k, i, j] · K[k, j],
ˆV[k, i, j] = E[k, i, j] · V [k, j].
Residuals & Precision.
We compute the residual tensor Rqk:
Rqk[k, i, j] = Q[k, i] −ˆK[k, i, j],
This is weighted by the analytic precision tensor P, which is defined element-wise for each channel k using the DLE
solution, where µk := −Re(λk):
P[k, i, j] =
 
˜σ2
k
 1 −e−2µk∆tij 
+ η2
ke−2µk∆tij + γ2
k
 −1
.
Aggregation.
Unlike standard attention, which applies a single scalar score per head, tensor RFA computes an attention
tensor A ∈Rd×N×N. The logit tensor is then:
L[k, i, j] = log
 P[k, i, j]
 
−κ log
 
1 + 1
ν
X
k′
P[k′, i, j] ·
  Rqk[k′, i, j]
  2
!
.
The estimate in the eigenbasis is computed via a row-wise Softmax over the logits, followed by a weighted sum:
A[k, i, j] = Softmaxj(L[k, i, j]),
¯V [k, i] =
X
j≤i
A[k, i, j] · ˆV[k, i, j].
The time complexity remains O(N 2d), but storing the propagated keys and values, residuals, and attention tensors requires
O(N 2d) memory, limiting scalability. We therefore derive a memory-efficient implementation that avoids storing tensors.
B.2. Factorization and Complexity Reduction
We introduce the following factorizations to simplify the computation:
17


<!-- page 18 -->
Robust Filter Attention
B.2.1. TOEPLITZ KERNEL FOR PRECISION
If the measurements occur at equal time intervals δt, the analytic precision kernel P[k, i, j] depends only on the channel k
and the time lag τ = |i −j|. This induces a Toeplitz structure along the temporal dimensions for each channel.
Letting ∆tij = τδt, we can thus pre-compute a 1D covariance kernel: KΣ ∈Rd×N:
KΣ[k, τ] = ˜σ2
k
 1 −e−2µkδt τ 
+ η2
k e−2µkδt τ + γ2
k,
The full precision tensor is then simply the element-wise inverse of this kernel:
P[k, i, j] = KP [k, |i −j|] := 1/KΣ[k, |i −j|].
B.2.2. FACTORIZING THE PROPAGATED MEASUREMENTS
Because the dynamics are LTI, we can avoid decompose the propagation tensor E into separate forward and backward terms
for each dimension k:
E[k, i, j] = Φ+[k, i] · Φ−[k, j],
where:
Φ+[k, i] := eλkti,
Φ−[k, i] := e−λkti.
We can then define stationary representations:
ˆ
Q[k, j] := Φ−[k, j] · Q[k, j],
ˆ
K[k, j] := Φ−[k, j] · K[k, j],
ˆV [k, j] := Φ−[k, j] · V [k, j].
The propagated keys and values are then refactored as products:
ˆK[k, i, j] = Φ+[k, i] · ˆ
K[k, j],
ˆV[k, i, j] = Φ+[k, i] · ˆV [k, j]
Since Φ+[k, i] does not depend on j, we can pull it outside the sum:
¯V [k, i] :=
X
j≤i
A[k, i, j] · ˆV[k, i, j] = Φ+[k, i] ·
X
j≤i
A[k, i, j] · ˆV [k, j]
B.2.3. MEMORY EFFICIENCY AND NUMERICAL STABILITY
Recall that:
Rqk[k, i, j] = Q[k, i] −ˆK[k, i, j]
Plugging in the factorizations for ˆ
Q[k, j] and ˆK[k, i, j], the residual becomes:
Rqk[k, i, j] = Q[k, i] −Φ+[k, i] · ˆ
K[k, j] = Φ+[k, i] ·
  ˆ
Q[k, i] −ˆ
K[k, j]
 
.
The matrix of Mahalanobis distances now becomes:
D2[i, j] =
X
k
KP [k, |i −j|]
|
{z
}
Precision kernel
· |Φ+[k, i]|2
|
{z
}
Forward decay
·
"
   ˆ
Q[k, i]
  2
|
{z
}
Stationary Query
+
   ˆ
K[k, j]
  2
|
{z
}
Stationary Key
−2 Re
  ˆ
Q
∗[k, i] ˆ
K[k, j]
 
|
{z
}
Stationary Cross-term
#
(where ∗denotes the complex conjugate). The remaining bottleneck is the k-dependence of the precision kernel KP in the
evaluation of the cross-term:
X
k
KP [k, |i −j|] · 2 Re
  ˆ
Q
∗[k, i] ˆ
K[k, j]
 
.
In standard attention, scores are computed with a single matrix multiplication (QK⊤). Here, however, the precision kernel
KP [k, |i −j|] weights each feature differently as a function of time lag, so the summation over k cannot be expressed as a
single matmul. Achieving O(N 2 + Nd) memory therefore requires the precision kernel to be independent of the feature
index k, allowing it to factor outside the summation.
A degenerate case occurs in the zero-noise limit, where the precision kernel is constant. This recovers a memory-efficient
formulation with anisotropic (feature-wise) decay, similar to xPos.
18


<!-- page 19 -->
Robust Filter Attention
However, for stable dynamics with µk > 0, the backward transition factor Φ−[k, j] = e(µk−iωk)tj grows exponentially with
sequence length. When decay rates vary across features, the stationary representations ˆ
Q, ˆ
K, ˆV grow exponentially with
sequence length, making fully parallel computation numerically unstable because forward and backward factors cancel only
after multiplication, allowing intermediate values to overflow.
Therefore, retaining a non-constant precision kernel while ensuring numerical stability under extrapolation requires restricting
decay to be isotropic within each head. This allows decay to be factored at the head level rather than per feature, enabling
stable, fully parallel attention with O(N 2 + Nd) memory. This motivates the Isotropic RFA variant introduced next.
B.3. Isotropic RFA
B.3.1. ISOTROPIC DECAY AND NOISE ASSUMPTIONS
All assumptions in this section are applied per attention head. In particular, the real part of the eigenvalues within a head is
taken to be a shared scalar −µ:
λk = −µ + iωk,
µ ∈R+, ωk ∈R
This corresponds to a system with an isotropic plus skew-symmetric state matrix:
A = −µI + Ω,
Ω= −Ω⊤∈Rd×d.
We also assume that the noise is isotropic, i.e. that the noise covariances are scalar multiples of identity:
ΛQ = σ2I,
ΛR = η2I,
ΛΓ = γ2I
Under this constraint, the covariance kernel simplifies to a scalar function:
Σ2(|i −j|) = ˜σ2  1 −e−2µδt|i−j| 
+ η2e−2µδt|i−j| + γ2
Hence, the precision kernel becomes a scalar function of the time lag τ = |i −j|, allowing it to be pulled outside the feature
summation. Defining Σ∆t[i, j] := Σ2(|i −j|) and P ∆t[i, j] := 1/Σ∆t[i, j], the matrix of Mahalanobis distances become:
D2[i, j] = P ∆t[i, j] ·
  X
k
  Rqk[k, i, j]
  2 
=: P ∆t[i, j] ·
Rqk[i, j]
2,
(Note that
Rqk
 denotes a matrix of vector norms, not a matrix norm.)
B.3.2. SIMPLIFYING THE SQUARED RESIDUAL NORM
The isotropic constraint allows the dynamics to be factored into a stable decay kernel and complex-valued forward/backward
rotations:
E[i, j] = e−µ|ti−tj|,
˜Φ
+[k, i] := eiωkti,
˜Φ
−[k, i] := e−iωkti,
We can then define backward-rotated queries, keys, and values:
˜
Q := ˜Φ
−⊙Q,
˜
K := ˜Φ
−⊙K,
˜V := ˜Φ
−⊙V ,
Note that:
Φ+[k, i] = e−µti ˜Φ
+[k, i],
ˆ
Q[k, i] = eµti ˜
Q[k, i],
ˆ
K[k, j] = eµtj ˜
K[k, j].
Plugging this into the expression for the Mahalanobis distance, and using the fact that complex rotation preserves magnitude:
Rqk[i, j]
2 =
X
k
e−2µti  ˜Φ
+[k, i]
  2 ·
h
e2µti   ˜
Q[k, i]
  2 + e2µtj   ˜
K[k, j]
  2 −2eµtieµtj Re
  ˜
Q
∗[k, i] ˜
K[k, j]
 i
.
=
X
k
h   ˜
Q[k, i]
  2 + e−2µ(ti−tj)   ˜
K[k, j]
  2 −2e−µ(ti−tj) Re
  ˜
Q[k, i]∗˜
K[k, j]
 
.
Or, in vectorized form:
Rqk[i, j]
2 = ∥Qi∥2
| {z }
Query Norm
+ E[i, j]2 · ∥Kj∥2
|
{z
}
Decayed Key Norm
−2 E[i, j] · Re
 
˜
Q
†
i ˜
Kj
 
|
{z
}
Propagated Cross-term
19


<!-- page 20 -->
Robust Filter Attention
(since ∥Qi∥2 = ∥˜
Qi∥2 and ∥Kj∥2 = ∥˜
Kj∥2).
Hence, under the isotropic assumption, the cross-term Re( ˜
Q
† ˜
K) can be computed using one O(N 2d) matrix multiplication,
achieving the required memory efficiency.
B.3.3. THE ATTENTION MATRIX AND ESTIMATE
The logit matrix L is defined using a Student-t–inspired robust loss:
L = log
 P ∆t
 
−(νs + 1) log
 
1 +
1
νsdP ∆t ⊙
Rqk
2 
,
where νs ∈R+ and d is the head dimension.
Defining a causal mask M causal and adding an inverse temperature parameter βs, we can then express the row-normalization
using row-wise Softmax:
A[i, j] = Softmaxj
 βsL[i, j] + M causal
 
.
The value aggregation is refactored for stability:
¯V [k, i] =
 
e−µti ˜Φ
+[k, i]
 
·
X
j≤i
A[i, j] ·
 
eµtj ˜V [k, j]
 
= ˜Φ
+[k, i] ·
X
j≤i
 
A[i, j] · E[i, j]
 
· ˜V [k, j]
Hence, defining a decayed attention matrix ˆ
A := A⊙E, the filtered estimate ¯V is computed by transforming the aggregation
back into the original frame:
¯V = ˜Φ
+ ⊙
  ˜V ˆ
A
⊤ 
.
Or, in a form more typical for attention:
¯V
⊤= (˜Φ
+)⊤⊙( ˆ
A ˜V
⊤)
This rotate–aggregate–counter-rotate structure aligns representations in a shared dynamical frame before aggregation (Fig. 2).
In practice, multi-head attention is implemented by applying this procedure independently across multiple heads, each with
its own dynamical and uncertainty parameters. The resulting update steps are concatenated and projected in the standard
Transformer manner.
Figure 2. Rotate, aggregate, counter-rotate structure of Isotropic RFA. Queries, keys, and values are rotated into a common frame to
compute attention and aggregate values. The resulting estimate is then rotated back to the initial frame, yielding a state that preserves
relative phase while remaining equivariant to absolute position.
B.3.4. RESIDUAL CONNECTION AND OUTPUT PROJECTION
The filtered value ¯V should not be interpreted as a complete replacement for the current representation, but rather as a
refined estimate under the robust filtering objective. Accordingly, the attention layer produces a correction relative to the
current value representation:
∆V := ¯V −V .
20

[CAPTION] Figure 2. Rotate, aggregate, counter-rotate structure of Isotropic RFA. Queries, keys, and values are rotated into a common frame to


<!-- page 21 -->
Robust Filter Attention
The residual connection then applies a partial step toward this estimate:
V + = V + α∆V ,
where α ∈(0, 1] controls the step size. Mapping back to the original basis yields:
Z+ = W oV + = W oV + αW o( ¯V −V ).
Since the value projection maps residual representations into the latent dynamical space while the output projection maps
the resulting corrections back into the residual stream, we expect W oW v ≈I. Under this assumption, the update may be
written directly in the residual stream:
Z+ ≈Z + αW o∆V .
Hence, each attention layer contributes an incremental correction to the current representation, analogous to an iterative
refinement step in state estimation.
B.3.5. THE ZERO-DECAY LIMIT AND ALIBI
If the queries and keys are normalized, the matrix of squared residual norms becomes:
Rqk
2 = 1 + E2 −2 E ⊙Re
 
˜
Q
† ˜
K
 
.
In the zero-decay limit (µ →0), the relative decay vanishes (E = 1), and the residual simplifies to the chordal distance on
the unit-norm hypersphere:
Rqk
2 = 2
 
1 −Re
  ˜
Q
† ˜
K
  
.
Substituting this into the NLL,
L = log(P ∆t) −κ log
 
1 + 2
ν P ∆t −2
ν P ∆t ⊙Re
  ˜
Q
† ˜
K
  
.
In the zero-decay limit, the covariance grows linearly with temporal lag:
Σ2(∆t) = σ2 ∆t + (η2 + γ2),
P ∆t = 1/Σ2(∆t).
The resulting additive bias is dominated by the log-precision term :
B∆t ≈log(P ∆t) = −log
 η2 + γ2 
−log
 
1 +
σ2
η2 + γ2 ∆t
 
.
For small ∆t, this admits the linear approximation:
B∆t ≈−log
 η2 + γ2 
−
σ2
η2 + γ2 ∆t.
Thus, in the short-lag regime, the RFA prior induces an approximately linear distance-dependent bias, providing a local
approximation to commonly used linear recency biases such as ALiBi.
B.4. Diffusive and Integrative Filtering Regimes
Here we derive the explicit functional forms of the additive bias B∆t := log(P ∆t) and multiplicative gate P ∆t in each
regime. The behavior of each regime is illustrated in Fig. 3, and the effect of µ on the speed of the phase transition is shown
in Fig. 4.
Diffusive Regime (α < 0).
Letting α′ = −α > 0, the bias follows a logarithmic decay:
B∆t = −log(β) −log
 
1 −α′
β e−2µ∆t
 
,
starting at its maximum at ∆t = 0 and decaying toward −log(β). The precision decays as:
P ∆t =
 β −α′e−2µ∆t −1,
with selectivity maximal near the diagonal and blurring out at longer lags.
21


<!-- page 22 -->
Robust Filter Attention
Integrative Regime (α > 0).
The bias follows a mirrored Softplus:
B∆t = −log(β) −Softplus(ln(α/β) −2µ∆t) ,
starting low and curving upward as key-side measurement noise dissipates. The precision follows a sigmoid:
P ∆t = 1
β · sigmoid(2µ∆t −ln(α/β)) ,
with selectivity initially low, opening as the transported observation becomes reliable.
Figure 3. By varying the ratio of steady-state process uncertainty ˜σ2 to measurement noise η2, RFA heads can specialize into distinct
physical regimes: a diffusive regime that favors local recency (top row) and an integrative regime (bottom row) that filters transient
noise to identify stable historical trends. The multiplicative gate P ∆t controls the selectivity (adaptive gain) of the attention, while the
additive bias B∆t defines the prior budget allocated to tokens at a given temporal lag.
Figure 4. The decay rate µ dictates the speed of the phase transition. As µ →0, the model recovers Brownian diffusion, where precision
drops linearly with time. As µ increases, the model enforces stationarity, where the attention bias saturates to a learned global noise floor
β, providing a principled mechanism for long-range context retention.
22

[CAPTION] Figure 3. By varying the ratio of steady-state process uncertainty ˜σ2 to measurement noise η2, RFA heads can specialize into distinct

[CAPTION] Figure 4. The decay rate µ dictates the speed of the phase transition. As µ →0, the model recovers Brownian diffusion, where precision


<!-- page 23 -->
Robust Filter Attention
B.5. Spectrally Coupled RFA
In standard RFA, all heads share the full frequency range with a uniform decay rate, so high- and low-frequency components
are damped equally within each head (Fig. 5, left). SC-RFA partitions the spectrum across heads and couples decay rate to
the maximum frequency per head (Fig. 5, right), inducing an ordered separation of temporal scales: high-frequency heads
decay rapidly and act as short-range filters, while low-frequency heads decay slowly and preserve long-range structure. The
resulting eigenvalue distribution in the complex plane is shown in Fig. 6.
Figure 5. Eigenvalue spectra of standard isotropic RFA (left) and SC-RFA (right), with b = 1.0. Standard RFA uses the full frequency
range with uniform decay per head. SC-RFA assigns each head a distinct spectral band with decay rate coupled to its maximum frequency.
Figure 6. SC-RFA eigenvalue distribution in log-log space (b = 1.0). The boundary µh = b · ωh,max appears as a straight line of slope
b. Each head’s eigenvalues form a vertical strip, with the highest-frequency eigenvalue on the boundary and lower-frequency eigenvalues
falling below it.
23

[CAPTION] Figure 5. Eigenvalue spectra of standard isotropic RFA (left) and SC-RFA (right), with b = 1.0. Standard RFA uses the full frequency

[CAPTION] Figure 6. SC-RFA eigenvalue distribution in log-log space (b = 1.0). The boundary µh = b · ωh,max appears as a straight line of slope


<!-- page 24 -->
Robust Filter Attention
C. Extensions
This section presents some natural generalizations of the RFA framework.
C.1. Inhomogeneous Dynamics
We show that constant drift does not alter the structure of RFA. Instead, it induces a deterministic shift in the propagated
mean, which can be absorbed into learned bias terms in the query, key, and value projections.
Consider an inhomogeneous linear SDE with a constant drift u:
dx(t) =
 Ax(t) + u
 
dt + G dw(t),
zk = x(tk) + v(tk).
The drift term can be eliminated via state augmentation, but this introduces a singular system that breaks simultaneous
diagonalization. We therefore treat the inhomogeneous case directly.
We assume A is Hurwitz. Identifying the equilibrium point µ, the SDE can be rewritten as:
dx(t) = A
 x(t) −µ
 
dt + G dw(t),
µ = −A−1u.
The solution to the SDE, propagating the state forward from x(tj) to x(ti), is:
x(ti) = eA∆tijx(tj) +
 Z ∆tij
0
eA(∆tij−τ) dτ
!
u +
Z ∆tij
0
eA(∆tij−τ)G dw(τ).
Letting Gu(∆tij) =
R ∆tij
0
e−Aτ dτ, the deterministic part is:
ˆxij = eA∆tij 
x(tj) + Gu(∆tij) u
 
.
Letting us := S−1u, the drift term is:
Gu(∆tij)u = S
  Z ∆tij
0
e−Λτ dτ
 
us = S
 I −e−Λ∆tij
Λ
 
us
Hence, the propagated measurement in the eigenbasis becomes:
ˆzs,ij = eΛ∆tij 
zs,j +
 I −e−Λ∆tij
Λ
 
us
 
= eΛ∆tij 
zs,j + us
Λ
 
−us
Λ ,
(where division is element-wise). Thus, drift induces a constant offset −us/Λ in the diagonalized coordinates. Because the
drift contributes only a deterministic shift, the covariance evolution remains identical to the homogeneous case.
Since this shift is constant across time, it can be absorbed into the learned linear projections by defining bias terms
bq, bk, bv ∈Cd×1 in the input projections defining the queries, keys, and values:
Qu[k, i] := Q[k, i] + bq[k],
Ku[k, i] := K[k, i] + bk[k],
V u[k, i] := V [k, i] + bv[k],
bℓ[k] := uℓ[k]
λℓ,k
,
ℓ∈{q, k, v},
where uℓand λℓ,k denote the k-th diagonal element of the drift and eigenvalue vector associated with the projection ℓ.
These bias terms correspond to the steady-state offset induced by constant drift in the diagonalized dynamics. This allows
the residual tensor to maintain the same form as the homogeneous case, using the biased tensors:
Rqk[k, i, j] = Qu[k, i] −Eqk[k, i, j] · Ku[k, j].
The attention output is:
¯V [k, i] = Φv[k, i] ·
X
j≤i
A[k, i, j] · ˆV u[k, j] −bv[k] ·
X
j≤i
A[k, i, j]
24


<!-- page 25 -->
Robust Filter Attention
= Φv[k, i] ·
X
j≤i
A[k, i, j] · ˆV u[k, j] −bv[k],
where we have used the fact that P
j≤i A[k, i, j] = 1 due to softmax normalization. The bias in this final expression can be
absorbed into the bias of the output projection: bo := W obv.
Hence, the inhomogeneous SDE with constant drift u is structurally equivalent to the homogeneous RFA mechanism,
provided the deterministic effects are absorbed into constant bias vectors in the input and output projections (bq, bk, bv, bo).
C.2. Generalized Analytic Priors via Time-Structured Noise
The derivation in Section A.1 assumed white process noise. However, each diagonal DLE is a linear ODE, and allowing
the noise injection rate qk(t) to vary in time yields a richer class of analytic priors. For each mode k with decay rate
µk = −Re(λk), the covariance satisfies:
d
d∆tλV,k(∆t) = −2µkλV,k(∆t) + qk(∆t),
λV,k(0) = 0.
The unique solution is given by the convolution of the mode-specific noise source qk(s) with the system’s exponential
impulse response:
λV,k(∆t) =
Z ∆t
0
e−2µk(∆t−s)qk(s) ds.
To ensure λV,k(∆t) can be solved in closed-form, we restrict the noise source to the class of functions closed under
exponential convolution: the complex exponentials. Letting qk(s) = P
j cjeγjs for cj, γj ∈C, the integral yields a
weighted sum of exponential differences:
λV,k(∆t) = Re
  X
j
cj
 eγj∆t −e−2µk∆t
2µk + γj
  
.
This characterizes the most general class of precision kernels that remain analytically tractable under a scalar DLE.
C.3. Stacked Attention Layers as an Unrolled Iterative State Estimator
The robust M-estimator is defined implicitly: the weights wij depend on residuals computed against the unknown latent
state xi, which must itself be approximated from the current iterate. Each attention layer can therefore be interpreted as one
step of an Iteratively Reweighted Least Squares (IRLS)-like procedure: given the previous layer’s state estimate, the current
layer recomputes residuals, updates weights, and produces a refined precision-weighted average.
Working in the eigenbasis zs,i := S−1zi, the procedure is initialized with each position’s own embedding as the zeroth
estimate: ˆz(1)
s,ii = zs,i. At each iteration, transported predictions are recomputed from the current state estimates:
ˆz(k)
s,ij = eΛ∆tij ˆz(k)
s,jj.
Weights are recomputed from the Mahalanobis residuals:
w(k)
ij
:=
 
1 + (λP,ij)⊤|r(k)
s,ij|2/ν
 −κ
,
r(k)
s,ij := ˆz(k)
s,ii −ˆz(k)
s,ij.
A single refinement step k computes the precision-weighted estimate
¯z(k)
s,i =
  X
j≤i
w(k)
ij λP,ij
 −1
⊙
X
j≤i
w(k)
ij λP,ij ⊙ˆz(k)
s,ij.
where λP,ij := λP (∆tij) is the diagonal precision vector at lag ∆tij (Appendix A).
The estimate may then be updated via an innovation step:
ˆz(k+1)
s,ii
= ˆz(k)
s,ii + αi
 ¯z(k)
s,i −ˆz(k)
s,ii
 
,
where αi ∈(0, 1] controls the correction step size. This mirrors the residual updates used in standard Transformers,
which may similarly be interpreted as iterative correction steps. Stacking L attention layers with shared parameters can be
interpreted as unrolling L steps of this iterative estimation procedure.
25


<!-- page 26 -->
Robust Filter Attention
D. Implementation
D.1. Complex-valued Computations
RFA is formulated in a complex latent space. In practice, this is implemented by lifting real-valued representations into a
2d-dimensional space using a linear projection (corresponding to Cd ∼= R2d), performing complex rotations and attention in
this space, and then projecting the result back to Rd.
A complex-valued linear transformation can be represented in the real domain by operating on paired real and imaginary
channels. For an input x = [xr, xi]⊤with xr, xi ∈Rd, this corresponds to:
L(x) =
"
W r
−W i
W i
W r
#
x +
"
br
bi
#
,
Here W r, W i ∈Rd×d are the real and imaginary components of the weight matrix and br, bi ∈Rd the bias. This is
equivalent to multiplication by a complex matrix W = W r + iW i with bias b = br + ibi.
Assuming the inputs and outputs are purely real, only the real-input columns of the input projections and the real-output
columns of the output projections are required:
Ld×2d(xr) :=
"
W r
W i
#
xr +
"
br
bi
#
.
L2d×d(x) :=
 W r
−W i
 
x + br.
Hence, both projections may be implemented using standard real-valued linear layers in R2d.
We define queries, keys, and values using:
Q = Ld×2d
q
(Z),
K = Ld×2d
k
(Z),
V = Ld×2d
v
(Z)
We define cosine and sine matrices:
C[k, i] = cos(ωkti),
S[k, i] = sin(ωkti)
Complex rotations are applied as:
˜
Q
⊤=
"
˜
Q
⊤
r
˜
Q
⊤
i
#
= (˜Φ
−)⊤⊙Q⊤=
 C ⊙Q⊤
r + S ⊙Q⊤
i
C ⊙Q⊤
i −S ⊙Q⊤
r
 
,
and likewise for ˜
K and ˜V . This is algebraically identical to RoPE.
To ensure the underlying system matrix A is real-valued, we enforce that its eigenvalues appear in complex conjugate pairs:
ω =
 
ω1, −ω1, . . . , ωd/2, −ωd/2
	
,
The Mahalanobis distance requires the real part of the complex inner product,
Re
 
˜
Q
† ˜
K
 
= ˜
Q
⊤
r ˜
Kr + ˜
Q
⊤
i ˜
Ki =
h
˜
Q
⊤
r
˜
Q
⊤
i
i   ˜
Kr
˜
Ki
 
.
This is implemented as a single real matrix multiplication in R2d.
Value aggregation, ¯V , is computed in the R2d domain. The real-valued attention matrix ˆ
A is applied identically to both the
real and imaginary components of the complex-rotated values:
M ⊤= ˆ
A
"
˜V
⊤
r
˜V
⊤
i
#
.
26


<!-- page 27 -->
Robust Filter Attention
The inverse rotation yields:
¯
Z
⊤
v = (˜Φ
+)⊤⊙( ˆ
A ˜V
⊤) =
 C ⊙M ⊤
r −S ⊙M ⊤
i
C ⊙M ⊤
i + S ⊙M ⊤
r
 
The final output is projected back to the real domain using the L2d×d layer:
¯
Z = L2d×d
o
( ¯V ) =
 W r
−W i
  ¯V + br ∈Rd×N.
All components of RFA are therefore implemented using standard real-valued operations.
D.2. Initialization
Isotropic Complex Projections.
Complex weights W = W r + iW i are initialized isotropically:
W ij = Mij
 cos(ϕij)
sin(ϕij)
 
,
Mij ∼Rayleigh
 r
1
din + dout
 
,
ϕij ∼U(0, 2π).
Output projections (W o) are scaled by 1/
√
2 to preserve variance when converting back to real space.
Noise and Robustness.
We initialize a constant steady-state uncertainty ˜σ across heads by scaling the process noise with
the decay rate, σ = 0.1 µ. This ensures that the variance floor remains comparable across heads despite differences in
temporal persistence. We initialize measurement noise (η2, γ2) such that such that the model begins in the integrative regime
(η2 > ˜σ2), to preserve long-range gradient flow early in training. We enforce positive query-side measurement noise γ2 > 0
to ensure finite precision.
The Student-t degrees of freedom ν are initialized as a positive multiple of the head dimension: ν = νsd. We initialized
νs = 4, placing the model in a quasi-Gaussian regime during the initial phase of training. This provides a broad prior that
prevents the premature rejection of tokens while the Query-Key representations are still unoptimized.
Remark. In our implementation, σ2 was learned directly. An equivalent and often more numerically stable parameterization
is obtained by learning the steady-state variance ˜σ2 := σ2/(2µ) directly. This decouples the variance floor from the decay
rate µ, improving conditioning when µ varies across heads. We drop the update step size α, since it may be absorbed into
the output projection W o.
27


<!-- page 28 -->
Robust Filter Attention
D.3. Algorithm
Algorithm 1 details the implementation of Isotropic RFA.
(Note: We use ⊕to denote broadcast addition.)
Algorithm 1 Robust Filter Attention (Isotropic; Single Head)
Input: Input sequence Z ∈Rd×N
Definitions:
Linear layers: Ld×2d
q
, Ld×2d
k
, Ld×2d
v
, L2d×d
o
.
Scalar parameters: Noise variance parameters: σ′, η′, γ′; robustness parameter νs; Softmax inverse temperature βs.
Constants: Causal mask M causal ∈{0, −∞}N×N; angular frequencies ω; decay rate µ ∈R+.
Enforce Conjugate Symmetry: ω ∈{ω′
1, −ω′
1, . . . , ω′
d/2, −ω′
d/2}.
Ensure positive noise/decay parameters:
{˜σ2, η2, γ2} ←Softplus({σ′, η′, γ′})
Input projections:
(Re(Q), Im(Q)) ←Lq(Z)
(Re(K), Im(K)) ←Lk(Z)
(Re(V ), Im(V )) ←Lv(Z)
Decay and rotation kernels: E[i, j] = e−µ|ti−tj|,
˜Φ
+[k, i] = eiωkti,
˜Φ
−[k, i] = e−iωkti
Covariance kernel: Σ∆t[i, j] = ˜σ2 1 −E[i, j]2 
+ η2E[i, j]2 + γ2
Query/Key/Value Rotations: ˜
Q[k, i] = ˜Φ
−⊙Q[k, i],
˜
K[k, j] = ˜Φ
−⊙K[k, j]
˜V [k, i] = ˜Φ
−⊙V [k, i]
Squared residuals: ∥Rqk[i, j]∥2 = ∥Qi∥2 + E[i, j]2 · ∥Kj∥2 −2E[i, j] · Re( ˜
Q
†
i ˜
Kj)
Logits: L = −log(Σ∆t) −(νs + 1) log
 
1 +
1
νsd
Rqk
2 ⊘Σ∆t
 
.
Attention matrix: A[i, j] = Softmaxj
 βsL[i, j] + M causal
 
,
ˆ
A = A ⊙E
Value estimate: ¯V = ˜Φ
+ ⊙( ˜V ˆ
A
⊤)
Value step: ∆V = ¯V −V
Output projection: ∆Z ←Lo
 ∆V
 
Residual connection: Z+ = Z + ∆Z
Return: Z+
Our current implementation is written in high-level PyTorch and incurs an approximately 2× training overhead relative
to PyTorch’s optimized scaled dot-product attention backend; we expect this gap to be reduced with kernel fusion and
optimized implementations.
28


<!-- page 29 -->
Robust Filter Attention
E. Experimental Details and Ablations
E.1. Experimental Setup
Architecture and Model Configuration.
All experiments were conducted using a 6-layer decoder-only Transformer
architecture. We set the model dimension to dmodel = 256 with h = 8 attention heads. The attention mechanism maps the
model dimension to a total latent dimension of 512 via the d × 2d query, key, and value projections (split into dh = 64 per
head), while the 2d × d output projection maps back down to 256.
We employ a Pre-Norm configuration using Layer Normalization. The Feed-Forward Network utilizes an expansion factor
of 4. To optimize the parameter budget, we implement weight tying between the token embedding layer and the final linear
output head. We use the GPT-2 byte-pair encoding (BPE) tokenizer with a vocabulary size of 50,257 for all language
modeling experiments.
To ensure a fair comparison, RFA models and the baselines (RoPE and ALiBi) were designed with near-identical parameter
counts. RFA introduces only a small set of scalar coefficients per head to parameterize noise variances (˜σ2, η2, γ2) and
robustness (ν, βs). Hence, the RFA models match the baseline parameter count (19.36M), with only a 0.02% increase due to
additional scalar coefficients.
Training and Optimization Protocol.
Models were trained for 15 epochs using the Adam optimizer. We utilized a
OneCycleLR scheduler with cosine annealing and a 5% warmup period, and trained until convergence. For RFA models, we
adopted a decoupled optimization strategy to ensure the stability of the SDE coefficients:
Feature weights use a peak LR of 1 × 10−3 with β1 = 0.9, while SDE coefficients use 5 × 10−4 with β1 = 0.0 and
ϵ = 10−7. We apply global gradient clipping at 1.0, with a stricter 1 × 10−4 threshold for RFA-specific parameters.
All models were trained on the WikiText-103 and BabyLM-2025 datasets using a standard causal language modeling
objective.
E.2. Model Variants and Ablation Design
This section presents a series of ablations designed to isolate the contributions of RFA’s core components. The ablations
consist of:
1. Baseline models;
2. RFA variants evaluated in Section 4 (M1–M3); and
3. Structural diagnostic ablations (M2.1–M2.6), which progressively remove components of the filtering formulation
to test necessity and failure modes. These models are not intended as competitive models, but rather as mechanistic
probes of stability and extrapolation behavior.
Baselines:
• B1: Standard Transformer + RoPE. Dot-product attention with rotary positional embeddings (Su et al., 2024).
Applies d →2d →d projections to match RFA parameterization.
• B2: Standard Transformer + ALiBi. Dot-product attention with linear distance bias (Press et al., 2022). Tests whether
static geometric penalties are sufficient for stability. Applies d →2d →d projections to match RFA parameterization.
• B3: Decayed RoPE. Identical to B1 but with an additional exponential decay applied to attention scores per-head, as
in RFA, testing whether decay alone suffices in the absence of uncertainty modeling.
• B4: Spectrally Coupled RoPE (SC-RoPE). Identical to B3 but with frequency-partitioned RoPE with head-wise
decay schedules, testing whether decay with spectral coupling can recover SC-RFA’s stability.
Primary RFA Models:
29


<!-- page 30 -->
Robust Filter Attention
• M1: Isotropic RFA. Isotropic RFA as described in Algorithm D.3, replacing the attention module in a standard
Transformer. The first two heads are reserved with µh = 0, and ˜σh is initialized to 0.1µh.
• M2: Spectrally Coupled RFA (SC-RFA), optimized for near-field performance. Identical to M1 except with
explicit coupling between rotation frequencies and decay rates, µh = b · ωh,max, using light damping (b = 0.05).
• M3: Spectrally Coupled RFA (SC-RFA), optimized for extrapolation. Identical to M2 but with stronger damping
(b = 5.0), and ˜σh initialized to 0.5µh.
Structural Diagnostic Ablations:
These ablations progressively remove components of the filtering formulation, starting
from the full SC-RFA model (M2) and simplifying toward standard attention. Their purpose is to isolate which mechanisms
are required for stable extrapolation.
• M2.1: Exponential Kernel. M2 with Student’s t influence function replaced by an exponential weighting, i.e.,
wij = exp(−d2
ij/ν). This isolates the effect of heavy-tailed robust reweighting under the same dynamical precision
prior.
• M2.2: Flat Precision Prior. M2 with noise parameters removed so that P ∆t is constant across time lag. Tests whether
dynamics alone suffice without uncertainty accumulation.
• M2.3: No Multiplicative Gate. M2 with the multiplicative gating term P ∆t set to a constant, to test the impact of the
additive bias B∆t in isolation.
• M2.4: No Value Frame Alignment. M2 without value rotation and counter-rotation, testing the necessity of
aggregating in a shared temporal frame.
• M2.5: No Rotational Dynamics. M2 without rotations applied to queries, keys, or values, isolating the effect of
decay-only dynamics.
• M2.6: Unitary Dot-Product Limit. No decay, no process or measurement noise, and no query and key normalization
terms, so that attention weights reduce to normalized complex dot products between rotated embeddings. This yields a
purely rotational positional encoding analogous to RoPER (Harik & Jayasiri, 2022).
30


<!-- page 31 -->
Robust Filter Attention
F. Additional Experimental Results
F.1. Training Dynamics and Extrapolation Behavior
To assess learning efficiency, we track validation perplexity throughout training (Fig. 7a). All RFA variants (M1-M3) achieve
consistently lower validation perplexity than the baselines (B1 and B2) over the course of training. This suggests that the
SDE-based prior provides a more informative inductive bias than purely geometric positional encodings.
Figure 7b visualizes validation PPL and length extrapolation trends corresponding to the tabulated results in Section 4. RFA
variants converge faster and degrade more gradually with context length than RoPE, while ALiBi remains stable due to its
enforced locality, at the cost of worse performance within the training window.
(a) Validation perplexity over training epochs.
(b) Test perplexity under length extrapolation beyond the training window (512 tokens).
Figure 7. Training dynamics and length extrapolation on WikiText-103. RFA variants converge faster during training and degrade
more gradually with increasing context length than RoPE, while ALiBi remains stable due to enforced locality.
Figure 8 shows the sensitivity analysis over damping values b reported in Table 2. Increasing damping improves long-range
stability by suppressing high-frequency propagation, but excessively large damping degrades short-range modeling.
31

[CAPTION] Figure 7. Training dynamics and length extrapolation on WikiText-103. RFA variants converge faster during training and degrade

[CAPTION] Figure 8 shows the sensitivity analysis over damping values b reported in Table 2. Increasing damping improves long-range


<!-- page 32 -->
Robust Filter Attention
Figure 8. Impact of the Damping Coefficient b on Length Extrapolation. Perplexity curves for varying b demonstrate that higher
damping coefficient values effectively stabilize long-range integration.
F.2. Parameter Dynamics in RFA
Learned measurement and process noise parameters over the course of training are shown in Fig. 9 for the last layer of
both RFA (M1) and SC-RFA (M2). Distinct trajectories in query and key noise parameters indicate that different heads
self-organize into separate signal-to-noise regimes. We plot robustness parameter νs = ν/d and inverse temperature βs in
Fig. 10.
In general, lower-decay heads tend to converge to lower measurement noise variances η2, γ2, lower robustness parameter νs,
and higher inverse temperature βs, consistent with stable long-range integration, while higher-decay heads tend to tolerate
larger measurement noise.
Intermediate heads tend to converge to the highest measurement noise variance, lowest steady-state process uncertainty,
and strongest robustness, consistent with modeling heterogeneous and noisy mid-range structure, while extreme short- and
long-range heads tend to remain more tolerant to outliers.
The spectrally coupled model (M2, SC-RFA) exhibits lower average query and key noise variance and more clustered
trajectories across heads.
When initialized in the diffusive regime, we observed that higher-decay heads consistently transitioned into the integrative
regime (α > 0), while the lowest-decay heads remained diffusive (Fig. 11).
32

[CAPTION] Figure 8. Impact of the Damping Coefficient b on Length Extrapolation. Perplexity curves for varying b demonstrate that higher

[CAPTION] Fig. 10.
In general, lower-decay heads tend to converge to lower measurement noise variances η2, γ2, lower robustness parameter νs,


<!-- page 33 -->
Robust Filter Attention
(a) M1 Query Measurement Noise Variance (γ2)
(b) M2 Query Measurement Noise Variance (γ2)
(c) M1 Key Measurement Noise Variance (η2)
(d) M2 Key Measurement Noise Variance (η2)
(e) M1 Steady State Process Variance (σ2/2µ)
(f) M2 Steady State Process Variance (σ2/2µ)
Figure 9. Measurement and Process Noise Parameters Comparison. Query and key measurement noise variance and state process
variance for M1 and M2, over the course of training. (Note that ˜σ2 is undefined for heads 0 and 1, with µ = 0.)
33

[CAPTION] Figure 9. Measurement and Process Noise Parameters Comparison. Query and key measurement noise variance and state process


<!-- page 34 -->
Robust Filter Attention
(a) M1 Robustness Parameter (νs = ν/d)
(b) M2 Robustness Parameter (νs = ν/d)
(c) M1 Inverse Temperature (βs)
(d) M2 Inverse Temperature (βs)
Figure 10. Robustness and inverse temperature. Robustness parameter and inverse temperature for M1 and M2, over the course of
training.
(a) Phase parameter α by head (M2)
(b) Diffusive →integrative transition during training
Figure 11. Integrative dynamics in SC-RFA. (a) Phase parameter (α) under standard initialization, showing specialization across heads.
(b) When initialized in the diffusive regime (α < 0), most heads transitioned into the integrative regime (α > 0) during training, while
the two lowest-decay heads remained diffusive. (Note that α is undefined for heads 0 and 1, with µ = 0.)
34

[CAPTION] Figure 10. Robustness and inverse temperature. Robustness parameter and inverse temperature for M1 and M2, over the course of

[CAPTION] Figure 11. Integrative dynamics in SC-RFA. (a) Phase parameter (α) under standard initialization, showing specialization across heads.


<!-- page 35 -->
Robust Filter Attention
F.3. Analysis of Attention Matrices
We plot attention matrices at a context length of 4096 to visualize long-range behaviors induced by each positional prior: the
baselines RoPE (B1) (Fig 12) and ALiBi (B2) (Fig 13); and the RFA (M1) (Fig 14) and SC-RFA (M2) (Fig 15) models. We
use attention matrices from the last layer of each model.
Note: For the RFA models, for improved visualization, we plot the unattenuated attention matrix A rather than the decayed
attention matrix ˆ
A := A ⊙E.
Figure 12. Baseline RoPE Transformer (B1) at L = 4096: Attention map exhibits persistent checkerboard structure.
Figure 13. ALiBi Transformer (B2) at L = 4096: Attention maps remain tightly localized to the diagonal across all heads, with only
modest widening in higher heads. Long-range structure is suppressed rather than integrated.
35

[CAPTION] Figure 12. Baseline RoPE Transformer (B1) at L = 4096: Attention map exhibits persistent checkerboard structure.

[CAPTION] Figure 13. ALiBi Transformer (B2) at L = 4096: Attention maps remain tightly localized to the diagonal across all heads, with only


<!-- page 36 -->
Robust Filter Attention
Figure 14. Robust Filter Attention (M1) at L = 4096: Periodic bands are clearly visible. High-decay heads concentrate focus on the
local diagonal, while low-decay heads exhibit the integrative regime: the bottom-right corner near the diagonal is suppressed as the model
waits for the SDE dynamics to suppress initial measurement noise before assigning high precision to the state estimate.
Figure 15. Spectrally-Coupled RFA (M2, b = 0.05) at L = 4096. Frequency-dependent damping (µh = b · ωh,max) substantially alters
long-range attention structure. SC-RFA has fewer periodic bands than RFA. Heads 3-5 each have only a single band, which become
narrower and moves closer to the diagonal as decay increases. Heads 1 and 2 act as stable long-range integrators.
36

[CAPTION] Figure 14. Robust Filter Attention (M1) at L = 4096: Periodic bands are clearly visible. High-decay heads concentrate focus on the

[CAPTION] Figure 15. Spectrally-Coupled RFA (M2, b = 0.05) at L = 4096. Frequency-dependent damping (µh = b · ωh,max) substantially alters


<!-- page 37 -->
Robust Filter Attention
The attention maps for RoPE exhibit persistent checkerboard structure and high-frequency oscillations that remain visible
even at large temporal offsets. In the absence of decay, these oscillations introduce non-local interference and unstable
long-range patterns.
In ALiBi, attention remains tightly localized to the diagonal across all heads, reflecting its fixed distance-based bias. While
higher heads show modestly broader receptive fields, long-range context is suppressed rather than integrated.
In contrast, RFA (M1) produces clearer periodic bands by aggregating in a stationary frame, preserving phase relationships
and reducing interference. Some heads exhibit an “opening gate” behavior, where attention is suppressed near the diagonal
and peaks at a characteristic lag, indicating delayed aggregation until the state estimate stabilizes.
Spectral coupling in SC-RFA (M2) sharpens and organizes this structure. Coupling decay to frequency (µh = b · ωh,max)
induces a specialization of heads to temporal lags: high-decay heads concentrate near the diagonal, while low-decay heads
shift toward longer lags, forming distinct, narrow bands consistent with ∆t∗∝1/µh (Sec. 3.7).
In SC-RFA, the first two (zero-decay) heads exhibit vertical structures corresponding to stable long-range retrieval. These
heads effectively perform global key–value lookup rather than temporal filtering, assigning similar weight to salient tokens
across all query positions and producing clean, vertically aligned patterns.
Together, these patterns support the view that RFA learns a structured multi-scale filtering behavior rather than relying on
fixed geometric positional biases.
37