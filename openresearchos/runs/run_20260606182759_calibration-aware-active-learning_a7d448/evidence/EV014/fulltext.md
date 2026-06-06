<!-- page 1 -->
Bayesian E(3)-Equivariant Interatomic Potential
with Iterative Restratification of Many-body
Message Passing
Soohaeng Yoo Willow1†, Tae Hyeon Park1,2†, Gi Beom Sim1,2,
Sung Wook Moon3, Seung Kyu Min2,3, Sangjae Seo4,
Jaewook Kim4, D. ChangMo Yang1, Hyun Woo Kim5,
Juho Lee6*, Chang Woo Myung1,2,7*
1Department of Energy Science, Sungkyunkwan University, Seobu-ro
2066, Suwon, 16419, Republic of Korea.
2Center for 2D Quantum Heterostructures, Institute for Basic Science
(IBS), Suwon, 16419, Republic of Korea.
3Department of Chemistry, School of Natural Science, Ulsan National
Institute of Science and Technology (UNIST), 50 UNIST-gil, Ulju-gun,
Ulsan, 44919, Republic of Korea.
4Department of Supercomputing Acceleration Research, Korea Institute
of Science and Technology Information, Daejeon, 34141, Republic of
Korea.
5Department of Chemistry, Gwangju Institute of Science and
Technology, Gwangju, 61005, Republic of Korea.
6Kim Jaechul Graduate School of AI, KAIST, Daejeon, Republic of
Korea.
7Department of Energy, Sungkyunkwan University, Seobu-ro 2066,
Suwon, 16419, Republic of Korea.
*Corresponding author(s). E-mail(s): juholee@kaist.ac.kr;
cwmyung@skku.edu;
†These authors contributed equally to this work.
Abstract
Machine learning potentials (MLPs) have become essential for large-scale atom-
istic simulations, enabling ab initio-level accuracy with computational efficiency.
1
arXiv:2510.03046v2  [cs.LG]  8 Apr 2026


<!-- page 2 -->
However, current MLPs struggle with uncertainty quantification, limiting their
reliability for active learning, calibration, and out-of-distribution (OOD) detec-
tion. We address these challenges by developing Bayesian E(3) equivariant MLPs
with iterative restratification of many-body message passing. Our approach intro-
duces the joint energy-force negative log-likelihood (NLLJEF) loss function, which
explicitly models uncertainty in both energies and interatomic forces, yielding
substantially improved accuracy compared to conventional NLL losses. We sys-
tematically benchmark multiple Bayesian approaches, including deep ensembles
with mean-variance estimation, stochastic weight averaging Gaussian, improved
variational online Newton, and Laplace approximation by evaluating their per-
formance on uncertainty prediction, OOD detection, calibration, and active
learning tasks. We further demonstrate that NLLJEF facilitates efficient active
learning by quantifying energy and force uncertainties. Using Bayesian active
learning by disagreement (BALD), our framework outperforms random sampling
and energy-uncertainty-based sampling. Our results demonstrate that Bayesian
MLPs achieve competitive accuracy with state-of-the-art models while enabling
uncertainty-guided active learning, OOD detection, and energy/forces calibra-
tion. This work establishes Bayesian equivariant neural networks as a powerful
framework for developing uncertainty-aware MLPs for atomistic simulations at
scale.
Keywords: Bayesian Machine Learning Potential, Uncertainty Quantification,
Out-of-Distribution Detection, Calibration, Atomistic Simulation
1 Introduction
Machine learning has become a cornerstone of materials science, enabling data-driven
predictions that accelerate the discovery of novel materials and the prediction of
their properties [1–8]. However, neural networks often behave as black boxes, mak-
ing it challenging to assess their reliability on unseen inputs [9, 10]. In high-stakes
applications, understanding a model’s confidence is as critical as the prediction itself,
driving interest in uncertainty quantification (UQ) for material properties [11–14]. UQ
enables active learning, calibration, and out-of-distribution (OOD) detection, which
are essential capabilities for reliable materials modeling.
Bayesian neural networks (BNNs) offer a principled UQ framework by treating
weights as random variables with prior distributions [15–22]. Early work showed
that BNNs naturally exhibit higher uncertainty in data-sparse regions [23] and that
infinitely wide networks with appropriate priors are equivalent to Gaussian pro-
cesses [24]. Formally, a BNN model defines a prior distribution over network weights
and uses Bayes’ theorem to update this to the posterior p(w|D) after observing train-
ing data D. Predictions for a property y of a new input x feature are obtained by
marginalizing over the weight posterior, p(y|x, D) =
R
p(y|x, w)p(w|D)dw. While this
integral is generally intractable for modern NNs, advances in approximation tech-
niques have made it feasible to take advantage of the Bayesian approach in practice.
Although exact Bayesian inference remains challenging for modern NNs, approximate
2


<!-- page 3 -->
methods such as Hamiltonian Monte Carlo (HMC) [25], stochastic variational infer-
ence like Bayes by Backprop [26], and Monte Carlo (MC) dropout [27] have proven
effective in practice. Blundell et al. introduced Bayes by Backprop, an efficient algo-
rithm to learn a weight distribution using variational Bayes, fully compatible with
standard backpropagation training [26]. This allowed NNs to learn the mean and vari-
ance of each weight, demonstrating that reasonably good posterior estimates can be
obtained without expensive sampling.
Another notable class of approaches includes Bayesian approximation meth-
ods such as deep ensembles (DEs) [28], stochastic weight averaging Gaussian
(SWAG) [13], improved variational online Newton (IVON) [29], and Laplace approx-
imation (LA) [30]. These methods are all used to quantify predictive uncertainty
in neural network-based models but differ in how they incorporate Bayesian infer-
ence and in the mathematical structure they employ. DEs train multiple independent
mean-variance estimators (MVEs) [31] and aggregate their predictions to estimate
the uncertainty. It is intuitive, easy to implement, and often yields strong predictive
performance. SWAG constructs a low-dimensional Gaussian approximation by com-
puting the mean and covariance of the model weights collected during the later stages
of training, allowing sampling of diverse predictions from a single training run [13].
Both methods estimate the statistical properties of the weight space without explicitly
computing the posterior distribution, making them practical and widely applicable
in various domains. In contrast, IVON and LA treat the weights of neural networks
as random variables and explicitly define a mathematical structure for the posterior
approximation. IVON uses curvature information in the parameter space to approx-
imate a high-dimensional Gaussian distribution through variational inference and
allows the posterior to be updated online during training [29]. LA, on the other hand,
approximates the posterior as a Gaussian centered around the mode of the loss land-
scape after training. Recent developments, including Kronecker and block-diagonal
approximations, have made this approach scalable to large models [30].
Early MLPs relied on hand-crafted descriptors to ensure rotational, transla-
tional, and permutational invariance, such as Atom-Centered Symmetry Functions
(ACSF) [32], Smooth Overlap of Atomic Positions (SOAP), and the Atomic Cluster
Expansion (ACE) [33–36]. In 2019, Drautz introduced ACE which is a systemati-
cally improvable basis expansion of atomic environments. ACE represents the local
density as an orthonormal polynomial basis, guaranteeing invariance and complete-
ness in the limit of high expansion order. It provides a unifying framework that can
reproduce other descriptors as special cases and has been used both in linear mod-
els and as input to neural networks. In recent years, there has been a shift toward
learning descriptors automatically via deep neural network architectures that oper-
ate directly on atomic coordinates or graphs. Instead of explicit fingerprint vectors,
these models learn an internal representation of atomic structures through many
layers, often using invariant or equivariant neural network designs. Early examples
include Behler–Parrinello networks (which summed atomic contributions computed
from ACSFs) and graph convolutional models like SchNet [37]. SchNet introduced
continuous-filter convolution layers to operate on interatomic distances, producing
an architecture inherently invariant to translations, rotations, and atom indexing.
3


<!-- page 4 -->
Many subsequent models (TensorMol [38], PhysNet [39], DeepPot-SE [40], etc.) fol-
lowed this paradigm of encoding physics invariance by input features (distances,
angles) and by pooling/readout operations that sum or average over atoms. Recent
architectures explicitly retain directional information through features that transform
covariantly under rotations [41–44]. Notably, SE(3)-equivariant graph networks (e.g.
Tensor Field Networks(TFN) [45], Cormorant [46], and NequIP [47, 48]) use tensorial
features expanded in spherical harmonics that rotate with the atomic geometry. The
NequIP, SevenNet [49], Allegro [50], and MACE [1] models are prominent examples
that map atomic coordinates to latent feature vectors and iteratively update them
with rotation-equivariant message passing, finally outputting atomic energy contribu-
tions. More recent models such as GotenNet [51], MGNN [52], and the universal model
UMA [44] have further advanced predictive accuracy and transferability across diverse
chemical systems.
Uncertainty quantification for MLIPs has attracted growing attention as a dis-
tinct research direction. Several methods have been developed specifically for atomistic
systems beyond the general Bayesian approaches described above. Musil et al. [53] pre-
sented a resampling-based uncertainty estimation scheme for sparse Gaussian process
regression models, demonstrating that subsampled ensembles can provide calibrated
uncertainties at negligible additional inference cost. Zhu et al. [54] proposed fast uncer-
tainty estimates by fitting a Gaussian mixture model to the learned features of a
single neural network, achieving uncertainty quality comparable to deep ensembles at
a fraction of the computational cost. Bigi et al. [55] introduced the prediction rigidity
formalism and its last-layer approximation (LLPR), which provides post-hoc uncer-
tainties for pre-trained neural networks by treating the last-layer features as a linear
Gaussian process, with theoretical justification from the neural tangent kernel the-
ory. Zaverkin et al. [56] demonstrated that molecular dynamics biased by the MLIP’s
energy uncertainty can simultaneously explore extrapolative regions and rare events,
employing calibrated gradient-based uncertainties as an alternative to ensemble meth-
ods. Kellner and Ceriotti [57] investigated uncertainty propagation through shallow
ensembles with particular attention to the size-extensivity of uncertainty predictions.
Swinburne and Perez [58] developed POPS, a misspecification-aware uncertainty quan-
tification method designed for the regime where model errors dominate over data
scarcity; Perez et al. [59] subsequently applied POPS to propagate uncertainties across
diverse material properties in tungsten and to foundational MLIPs including MACE-
MPA-0. More recently, Xu et al. [60] applied evidential deep learning to MLIPs with
a physics-inspired design incorporating locality and directionality, providing single-
model uncertainty estimates without ensemble overhead. While these prior works
address various aspects of uncertainty estimation, they typically model force uncer-
tainties either as scalar per-atom quantities, as independent per-component variances,
or derive them indirectly through energy differentiation or ensemble propagation. In
contrast, the present work explicitly parameterizes a full 3 × 3 per-atom force covari-
ance matrix via Cholesky decomposition within a joint energy-force NLL framework,
and systematically compares multiple approximate Bayesian methods (DE, SWAG,
IVON, LA) for equivariant neural network potentials.
4


<!-- page 5 -->
Despite these advances, MLPs face a critical limitation: they are unreliable for
configurations outside their training distribution, where they may produce unphysi-
cal results [61]. Bayesian methods address this by quantifying uncertainty to guide
active learning loops that retrain on new ab initio calculations when entering high-
uncertainty regimes [54, 57, 62]. Although fully Bayesian models produce well-aligned
uncertainty estimates [61], they are computationally expensive. Conversely, ensemble
methods are easier to train but often yield overconfident predictions, requiring careful
calibration.
To address these challenges, we make three key contributions (Figure 1). First, we
develop the joint energy-force negative logarithmic likelihood (NLLJEF) loss function,
which provides a systematic way to quantify uncertainties in both energies and forces
simultaneously. Unlike conventional losses that treat forces as deterministic quanti-
ties, NLLJEF explicitly models force uncertainties, leading to substantially improved
accuracy and more reliable uncertainty quantification essential for active learning,
calibration, and OOD detection [63, 64]. Second, we develop comprehensive BNN mod-
els based on our RACE architecture, an equivariant message-passing neural network
that iteratively restratifies many-body interactions to reduce computational overhead.
We design an eight-headed MVE module that integrates with various approximate
Bayesian frameworks, including DE, SWAG, IVON, and LA. These models enable
efficient UQ while maintaining the expressiveness of equivariant architectures. Third,
we evaluated the performance of active learning with BALD-based sample selection.
By comparing strategies that consider energy, forces, and both simultaneously, our
experiments highlight an efficient approach to data selection in active learning.
We validate our methods on established benchmarks, including QM9 [65, 66],
rMD17 [67], PSB3 [68], 3BPA [69], and we introduce an OOD test set, oBN25. Our
work demonstrates that combining Bayesian deep learning with NLLJEF functions and
comprehensive evaluation metrics significantly improves the reliability and practical
applicability of Bayesian MLPs for demanding atomistic simulations.
2 Results
2.1 Predictive accuracy of the RACE architecture
We first establish that the base RACE architecture, trained with standard MSE loss,
achieves competitive point prediction accuracy compared to established equivariant
models. The following benchmarks on QM9 and PSB3 evaluate RACE using only MSE
training without any uncertainty quantification components, focusing purely on the
architectural capability of the model.
2.1.1 QM9 benchmark
We evaluate our model in the QM9 dataset [65, 66], which is a widely used benchmark
that contains approximately 134,000 small organic molecules composed of CHONF
atoms. Each molecule is annotated with quantum chemical properties calculated using
DFT at the B3LYP/6-31G(2df,p) level of theory. The dataset includes a diverse
set of molecular properties, such as dipole moment, isotropic polarizability, frontier
5


<!-- page 6 -->
Base
1-Body Emb.
𝐴0
{𝑍𝑖}
Angular 
Emb.
Radial 
Emb.
Layer 1
Layer 2
𝐴1
𝐸𝑖
(2)
𝐴2
𝐸𝑖
(3)
𝐴𝜂−1
𝐸𝑖
(𝜂)
…
…
Layer η
𝐸𝑖
(𝜂+1)
𝑎1
{𝒓𝒊𝒋}
{|𝒓𝒊𝒋|}
𝑒𝑖𝑗
ǁ𝑒𝑖𝑗
𝑅𝐵𝐹
Linear
Local Chemical Environment
{|𝒓𝒊𝒋|} , {𝑍𝑖, 𝑍𝑗} , {𝒓𝒊𝒋}
𝑹𝒊𝒋
Improved Variational Online Newton
Kulback-Leibler
divergence
Second-order optimization
weight
Loss
Approximate
Posterior distribution
𝜃s
ID
ID
ID
Training
Inference
Out-of-distribution Detection
Mean-variance Estimator
Deep Ensemble
Ensemble
Combination
⋮
⋮
Recalibration
Expected Confidence Level
Observed Confidence Level
Uncalibrated
Calibrated
Calibration Plot
True
Predicted
Uncalibrated Confidence Intervals
Calibrated Confidence Intervals
Active Learning
Inference
Training
Unlabeled
Pool
Low 
Uncertainty
High 
Uncertainty
`
`
Laplace Approximation
Hessian
𝐻= ∇𝜃
2ℒ(𝐷; 𝜃) ቚ
𝜃MAP
𝑝𝐷𝜃
𝑝(𝜃)
𝜃MAP
Approximate
Posterior distribution
𝜃s
Sampling
Sampling
Sampling
a
b
Residual
Self-
Interaction
Linear
Linear
𝐴𝜂−1
𝐸𝑖
(𝜂)
𝐴𝜂−2
𝑎1
Layer η-1
𝐵𝑖
𝜂−1
Rotationally 
equivariant
𝑒𝑖𝑗
ℳ
ǁ𝑒𝑖𝑗
𝑅𝐵𝐹
∑
Aggregation
Linear
Gate
Linear
: Concatenation
: Tensor product
: Multiplication
: Sum
c
d
e
Stochastic Weight Averaging Gaussian
𝜃1, 𝜃2, 𝜃3, ⋯, 𝜃𝑇
Collecting Weights
Training
Approximate
Posterior distribution
𝜃s
𝑞𝜃
𝑝𝜃
OOD
True
Predicted
𝜇𝐸
𝜎𝐸
𝜎𝐿6
𝜇𝐸
𝜎𝐸𝜎𝐿1
Fig. 1: Overview of the proposed Bayesian E(3)-equivariant machine learn-
ing potential framework. (a) Two model variants: the base model predicts only
energies and forces, while the Mean-variance Estimation model additionally outputs
predictive uncertainties for energy and forces. (b) The RACE model architecture, con-
sisting of an embedding layer that initializes node features A(0)
i
from atomic numbers
{Zi} and encodes local environments via pair-wise vectors rij, angular edge features
eij, and radial basis functions ˜eRBF(rij). (c) Each interaction layer updates node
features in a ResNet-like scheme and predicts per-atom energies Ei through a read-
out block; in Bayesian variants, this block also outputs uncertainties for energies and
forces. (d) Bayesian neural network approaches, including deep ensemble, stochastic
weight averaging Gaussian, improved variational online Newton, and Laplace approx-
imation, were used to obtain predictive distributions. (e) Downstream applications
of uncertainty quantification, including out-of-distribution detection, active learning
with uncertainty-based sample selection, and model recalibration using reliability plots
and confidence intervals.
6


**[Table p6.1]**
| Residual Li Self- Interaction 𝑎1 |  |  |
| --- | --- | --- |
| Residual Self- Interaction |  |  |
|  |  |  |
| 𝐵𝑖𝜂−1 Li | 𝐵𝑖𝜂−1 |  |
|  |  | Linear Gate Linear |


**[Table p6.2]**
|  |  |  |
| --- | --- | --- |
| 𝐴2 … 𝜂−1 | 𝐸𝑖(3) … (𝜂) |  |

[CAPTION] Fig. 1: Overview of the proposed Bayesian E(3)-equivariant machine learn-


<!-- page 7 -->
orbital energies (HOMO/LUMO), electronic spatial extent, thermodynamic quanti-
ties (internal energy, enthalpy, free energy), and vibrational properties (ZPVE, heat
capacity). Following the same dataset split as in the EquiformerV2 benchmark [43],
we use 110,000 molecules for training, 10,000 for validation, and 11,000 for testing
to allow for direct comparison with existing models. Further details on architecture,
training procedure, and hyperparameters can be found in Supplementary Section B.1
(Table S1).
Table 1 reports the mean absolute error (MAE) for each of the 12 regression tasks
on the test set. RACE, trained with standard MSE loss, achieved competitive per-
formance in general. In particular, it achieved the third-lowest MAE for zero-point
vibrational energy and comparable heat capacity accuracy to several established mod-
els including ViSNet, Equiformer, and EquiformerV2. While recent architectures such
as GotenNet and MGNN achieve lower errors across most properties, RACE remains com-
petitive in thermodynamic quantities (H, U0, G) and provides the foundation for the
Bayesian extensions presented in this work.
Table 1: MAE of various models on the QM9 test dataset across 12 quantum
chemical properties. Bold and underline indicate the best and second-best results,
respectively.
Task
α
∆ε εHOMO εLUMO
µ
CV
G
H
R2
U
U0
ZPVE
Model
Units
α3
0
meV
meV
meV
D
cal/mol K meV meV
α2
0
meV meV meV
DimeNet++ [70]
.044
33
25
20
.030
.023
8
7
.331
6
6
1.21
EGNN [71]
.071
48
29
25
.029
.031
12
12
.106
12
11
1.55
PaiNN [72]
.045
46
28
20
.012
.024
7.35 5.98 .066 5.83 5.85
1.28
TorchMD-NET [73]
.059
36
20
18
.011
.026
7.62 6.16 .033 6.38 6.15
1.84
SphereNet [74]
.046
32
23
18
.026
.021
8
6
.292
7
6
1.12
SEGNN [75]
.060
42
24
21
.023
.031
15
16
.660
13
15
1.62
EQGAT [76]
.053
32
20
16
.011
.024
23
24
.382
25
25
2.00
ViSNet [77]
.041
32
17
15
.010
.023
5.86 4.25 .030 4.25 4.23
1.56
Equiformer [42]
.046
30
15
14
.011
.023
7.63 6.63 .251 6.74 6.59
1.26
EquiformerV2 [43]
.050
29
14
13
.010
.023
7.57 6.22 .186 6.49 6.17
1.47
MGNN [52]
.041
30
23.2
17
.010
.023
5.7
4.1
.040
4.2
4.1
1.17
GotenNet [51]
.028
20
13.4
12.2
.007
.019
4.98 3.30 .024 3.41 3.37 1.08
RACE
.048
64
39
32
.027
.023
8.22 5.89 .311 5.99 5.61
1.15
2.1.2 PSB3 benchmark
We evaluate our model on the PSB3 dataset [68], which consists of 48,750 molecular
geometries of the penta-2,4-dieniminium cation sampled from 50 trajectories gen-
erated using exact factorization-based surface hopping dynamics (SHXF) [78]. The
simulations were performed using the SSR(2,2) formalism at the ωPBEh/6-31G* level
of theory [79, 80], with a time step of 0.24 fs over a total duration of 300 fs. The
dataset provides reference energies and forces for perfectly spin-paired singlet (PPS)
and open-shell singlet (OSS) configurations, as well as a phaseless coupling term ∆2,
7

[CAPTION] Table 1 reports the mean absolute error (MAE) for each of the 12 regression tasks

[CAPTION] Table 1: MAE of various models on the QM9 test dataset across 12 quantum


<!-- page 8 -->
Table 2: MAE, Root-Mean-Square Error (RMSE), and R2 values of SchNet,
NequIP, and RACE on the PSB3 test dataset. Energy errors (EP P S and EOSS) are
given in kcal/mol, and force errors in kcal/mol/˚A. For the coupling term, SchNet
is evaluated on ∆(in kcal/mol), whereas NequIP and RACE are evaluated on ∆2
(in (kcal/mol)2). Bold indicates the best result.
EP P S
EOSS
∆/ ∆2
Model
MAE
RMSE
R2
MAE
RMSE
R2
MAE
RMSE
R2
SchNet
E
0.05
0.08
-
0.06
0.10
-
0.08
0.30
-
F
0.19
0.39
-
0.22
0.45
-
0.30
1.10
-
NequIP
E
0.03
0.07
0.99
0.03
0.07
0.99
0.47
0.78
0.99
F
0.08
0.33
0.99
0.09
0.28
0.99
1.15
3.09
0.99
RACE
E
0.04
0.07
0.99
0.03
0.07
0.99
0.62
1.00
0.99
F
0.06
0.25
0.99
0.06
0.26
0.99
1.53
3.31
0.99
*NequIP denotes the implementation of NequIP available in the BAM package.
which describes the state interaction between these configurations. We use 40,000 con-
figurations for training, 4,375 for validation, and 4,375 for testing. Further details on
architecture, training procedure, and hyperparameters can be found in Supplementary
Section C.1 (Table S2 and Figure S2).
Table 2 summarizes the performances of SchNet, NequIP, RACE on the PSB3
test set. RACE achieves low energy MAEs of 0.04 kcal/mol for PPS and 0.03 kcal/-
mol for OSS, and 0.62 (kcal/mol)2 for ∆2. The corresponding force MAEs are 0.06
kcal/mol/˚A for both PPS and OSS, and 1.53 (kcal/mol)2/˚A for ∆2. All R2 scores
exceed 0.997, indicating excellent agreement between predictions and reference data.
Compared to SchNet, RACE produces substantially lower errors for both energies
and forces in PPS and OSS. Compared to NequIP, RACE demonstrated superior accu-
racy in PPS and OSS energy and force predictions, whereas NequIP achieved lower
errors for ∆2, particularly in force predictions. These results indicate that RACE pro-
vides accurate and reliable predictions of energies and forces relevant to excited-state
molecular dynamics in PSB3.
2.2 Effect of NLL-based training on predictive accuracy
Having established the competitive accuracy of the base RACE model, we now inves-
tigate the effect of NLL-based training strategies on predictive performance. Training
with NLL objectives introduces a dual optimization target—simultaneously minimiz-
ing prediction error and learning calibrated uncertainties—which inherently involves
a trade-off with point prediction accuracy compared to standard MSE training. A key
question is whether this trade-off can be mitigated. In this section, we compare MSE-
trained RACE, its ensemble variant, and deep ensembles trained with energy-only
NLL (NLLE) and joint energy-force NLL (NLLJEF) losses. We demonstrate that while
NLLE training leads to substantial accuracy degradation, NLLJEF largely recovers
the predictive performance of MSE-trained models, providing calibrated uncertainty
estimates with minimal sacrifice in accuracy.
8


**[Table p8.1]**
| Model | EP P S MAE RMSE R2 | EOSS MAE RMSE R2 | ∆ / ∆2 MAE RMSE R2 |
| --- | --- | --- | --- |
| E SchNet F | 0.05 0.08 - 0.19 0.39 - | 0.06 0.10 - 0.22 0.45 - | 0.08 0.30 - 0.30 1.10 - |
| E NequIP F | 0.03 0.07 0.99 0.08 0.33 0.99 | 0.03 0.07 0.99 0.09 0.28 0.99 | 0.47 0.78 0.99 1.15 3.09 0.99 |
| E RACE F | 0.04 0.07 0.99 0.06 0.25 0.99 | 0.03 0.07 0.99 0.06 0.26 0.99 | 0.62 1.00 0.99 1.53 3.31 0.99 |

[CAPTION] Table 2: MAE, Root-Mean-Square Error (RMSE), and R2 values of SchNet,

[CAPTION] Table 2 summarizes the performances of SchNet, NequIP, RACE on the PSB3


<!-- page 9 -->
2.2.1 rMD17 benchmark
We evaluate our models on the revised MD17 (rMD17) dataset [67], a recomputed
version of the original MD17 benchmark that provides high-accuracy energies and
forces for ten small organic molecules. The data were generated using DFT at the
PBE/def2-SVP level of theory. Each molecule contains 100,000 molecular dynamics
configurations. Following the data split used in MACE, we use 950 configurations for
training, 50 for validation, and 1,000 configurations for testing per molecule. Further
details on the architecture, training procedure, and hyperparameters can be found in
Supplementary Section D.1 (Table S3).
Table 3 summarizes the MAEs for the energy and force predictions across all
molecules. Although RACE does not reach the accuracy of state-of-the-art (SOTA)
equivariant models such as GotenNet, MGNN, and VisNet, it consistently surpasses
other baselines and achieves energy MAEs comparable to those of SOTA models.
We first examine RACE-Ensemble, obtained by averaging multiple independently
trained single head RACE models. This ensemble approach consistently improves the
single-model RACE across all molecules, reducing MAE in both energy and forces. For
example, in Aspirin, the force MAE decreases from 9.6 to 8.0 meV/˚A, while the energy
MAE improves from 3.3 to 2.9 meV. Averaged across all molecules, the ensemble yields
an error reduction of approximately 15 %.
Although NLL-based training shows lower performance compared to MSE-based
training for point estimates, the NLLJEF loss significantly bridges this gap for
Bayesian learning. As shown in Table 3, incorporating NLLJEF leads to substantial
improvements in test errors in all evaluated systems.
When employing deep ensembles trained with the NLLJEF loss (RACE-DE-JEF),
this consistently outperforms deep ensembles trained with energy-only NLL loss
(RACE-DE-E). In Aspirin, for example, RACE-DE-JEF reduces the energy MAE from 7.8
to 3.7 meV and the force MAE from 21.6 to 9.5 meV/˚A.
Both RACE-DE-E and RACE-DE-JEF show lower predictive accuracy than base
RACE or RACE-Ensemble, reflecting the dual objective of NLL-based training of opti-
mizing both predictive accuracy and UQ. However, a critical observation is that
while RACE-DE-E shows substantial compromise in test accuracy compared to MSE-
based single head models, RACE-DE-JEF achieves test errors remarkably close to those
of MSE-trained RACE. This demonstrates that NLLJEF enables effective training of
uncertainty-aware MLPs with minimal sacrifice in predictive accuracy, which provides
reliable uncertainty quantification almost “for free.” The benefits of this balanced
approach become particularly evident in Section 2.3, where we demonstrate supe-
rior OOD detection and calibration capabilities enabled by the improved uncertainty
estimates.
To quantify this trade-off more rigorously, we analyze predictive variability across
independently trained models (Supplementary Tables S4 and S6). Using ±1σ confi-
dence intervals, the error ranges of RACE (MSE) and RACE-DE-JEF overlap in 4 out of
14 molecule–property pairs for rMD17 and 2 out of 6 temperature–property pairs for
3BPA. For the non-overlapping cases, the average separation between the ±1σ inter-
vals remains small, with 0.44 meV for energy and 0.77 meV/˚A for forces in rMD17,
9

[CAPTION] Table 3 summarizes the MAEs for the energy and force predictions across all


<!-- page 10 -->
and 3.74 meV for energy and 1.16 meV/˚A for forces in 3BPA, indicating that the abso-
lute accuracy difference is minor. In contrast, RACE-DE-E shows errors 2–5× larger,
along with substantially higher variance. This suggests that the observed accuracy–
uncertainty trade-off is predominantly attributable to energy-only NLL training rather
than an inherent limitation of the Bayesian framework.
Table 3: MAE on the rMD17 test dataset. Energy (E, meV) and force (F, meV/˚A)
errors of different models trained on 950 configurations and validated on 50. Bold
and underline indicate the best and second-best results, respectively.
Aspirin
Ethanol
Malonaldehyde
Naphthalene
Salicylic acid
Toluene
Uracil
Model
E
F
E
F
E
F
E
F
E
F
E
F
E
F
GotenNet [51]
1.6
5.7
0.3
2.1
0.6
3.6
0.2
1.0
0.6
3.0
0.2
1.1
0.3
1.8
MGNN [52]
3.1
9.1
0.3
2.7
0.8
5.1
0.5
2.6
1.3
6.4
0.4
2.4
0.4
2.7
ViSNet [77]
1.9
6.6
0.3
2.3
0.6
3.9
0.2
1.3
0.7
3.4
0.3
1.1
0.3
2.1
MACE [1]
2.2
6.6
0.4
2.1
0.8
4.1
0.5
1.6
0.9
3.1
0.5
1.5
0.5
2.1
Allegro [50]
2.3
7.3
0.4
2.1
0.6
3.6
0.2
0.9
0.9
2.9
0.4
1.8
0.6
1.8
BOTNet [81]
2.3
8.5
0.4
3.2
0.8
5.8
0.2
1.8
0.8
4.3
0.3
1.9
0.4
3.2
NequIP [47]
2.3
8.2
0.4
2.8
0.8
5.1
0.9
1.3
0.7
4.0
0.3
1.6
0.4
3.1
ACE [69]
6.1
17.9
1.2
7.3
1.7
11.1
0.9
5.1
1.8
9.3
1.1
6.5
1.1
6.6
FCHL [82]
6.2
20.9
0.9
6.2
1.5
10.3
1.2
6.5
1.8
9.5
1.7
8.8
0.6
4.2
GAP [36]
17.7
44.9
3.5
18.1
4.8
26.4
3.8
16.5
5.6
24.7
4.0
17.8
3.0
17.6
ANI [83]
16.6
40.6
2.5
13.4
4.6
24.5
11.3
29.2
9.2
29.7
7.7
24.3
5.1
21.4
PaiNN [72]
6.9
16.1
2.7
10.0
3.9
13.8
5.1
3.6
4.9
9.1
4.2
4.4
4.5
6.1
RACE
3.3
9.6
0.8
3.6
1.4
7.2
0.9
2.6
1.1
5.4
0.6
2.7
0.7
4.4
RACE-Ensemble
2.9
8.0
0.8
3.0
1.2
6.1
0.9
2.1
1.1
4.7
0.5
2.2
0.7
3.7
RACE-DE-E
7.8
21.6
2.4
10.6
3.9
16.9
4.0
12.6
4.2
16.9
3.3
12.9
2.9
13.8
RACE-DE-JEF
3.7
9.5
1.4
4.4
2.2
8.7
1.8
3.5
2.4
7.6
1.5
3.4
1.6
4.7
2.2.2 3BPA benchmark
We evaluate our model on the 3BPA dataset [69], which contains molecular dynam-
ics trajectories of 3-(benzyloxy)pyridin-2-amine (3BPA), a flexible drug-like molecule
with three rotatable bonds and a complex dihedral potential energy surface. The
dataset is designed to assess both in-distribution (ID) and OOD generalization, with
molecular configurations sampled at three different temperatures: 300 K, 600 K, and
1200 K. Following previous benchmarks, we use 450 configurations for training and
50 for validation, sampled from the 300 K trajectory. The test set consists of 1,669
ID configurations at 300 K and 2,138 and 2,139 OOD configurations at 600 K and
1200 K, respectively. Further details on the architecture, training procedure, and
hyperparameters can be found in Supplementary Section E.1 (Table S5).
Table 4 presents the root-mean-square errors (RMSEs) for energy and force pre-
dictions. At 300 K (ID), RACE delivered competitive accuracy, achieving 3.4 meV in
energy, although it was less accurate in force prediction compared to SOTA models.
Under OOD conditions, predictive accuracy degraded, yet a single RACE achieved the
10


**[Table p10.1]**
| Model | Aspirin Ethanol Malonaldehyde Naphthalene Salicylic acid Toluene Uracil E F E F E F E F E F E F E F |
| --- | --- |
| GotenNet [51] MGNN [52] ViSNet [77] MACE [1] Allegro [50] BOTNet [81] NequIP [47] ACE [69] FCHL [82] GAP [36] ANI [83] PaiNN [72] | 1.6 5.7 0.3 2.1 0.6 3.6 0.2 1.0 0.6 3.0 0.2 1.1 0.3 1.8 3.1 9.1 0.3 2.7 0.8 5.1 0.5 2.6 1.3 6.4 0.4 2.4 0.4 2.7 1.9 6.6 0.3 2.3 0.6 3.9 0.2 1.3 0.7 3.4 0.3 1.1 0.3 2.1 2.2 6.6 0.4 2.1 0.8 4.1 0.5 1.6 0.9 3.1 0.5 1.5 0.5 2.1 2.3 7.3 0.4 2.1 0.6 3.6 0.2 0.9 0.9 2.9 0.4 1.8 0.6 1.8 2.3 8.5 0.4 3.2 0.8 5.8 0.2 1.8 0.8 4.3 0.3 1.9 0.4 3.2 2.3 8.2 0.4 2.8 0.8 5.1 0.9 1.3 0.7 4.0 0.3 1.6 0.4 3.1 6.1 17.9 1.2 7.3 1.7 11.1 0.9 5.1 1.8 9.3 1.1 6.5 1.1 6.6 6.2 20.9 0.9 6.2 1.5 10.3 1.2 6.5 1.8 9.5 1.7 8.8 0.6 4.2 17.7 44.9 3.5 18.1 4.8 26.4 3.8 16.5 5.6 24.7 4.0 17.8 3.0 17.6 16.6 40.6 2.5 13.4 4.6 24.5 11.3 29.2 9.2 29.7 7.7 24.3 5.1 21.4 6.9 16.1 2.7 10.0 3.9 13.8 5.1 3.6 4.9 9.1 4.2 4.4 4.5 6.1 |
| RACE RACE-Ensemble RACE-DE-E RACE-DE-JEF | 3.3 9.6 0.8 3.6 1.4 7.2 0.9 2.6 1.1 5.4 0.6 2.7 0.7 4.4 2.9 8.0 0.8 3.0 1.2 6.1 0.9 2.1 1.1 4.7 0.5 2.2 0.7 3.7 7.8 21.6 2.4 10.6 3.9 16.9 4.0 12.6 4.2 16.9 3.3 12.9 2.9 13.8 3.7 9.5 1.4 4.4 2.2 8.7 1.8 3.5 2.4 7.6 1.5 3.4 1.6 4.7 |

[CAPTION] Table 3: MAE on the rMD17 test dataset. Energy (E, meV) and force (F, meV/˚A)

[CAPTION] Table 4 presents the root-mean-square errors (RMSEs) for energy and force pre-


<!-- page 11 -->
second-best energy at 1200 K, demonstrating robustness in energy prediction across
distributions. RACE-Ensemble further improved performance. At 300 K, it achieved
the lowest energy error with competitive force accuracy. However, its advantages under
OOD were limited, providing little improvement over the single model at 600 K and
1200 K.
We investigate DE models with two NLL losses: energy-only NLL objective
(RACE-DE-E) and joint energy-force NLL loss (RACE-DE-JEF). While conventional neu-
ral networks trained with MSE loss benefit from focusing solely on point predictions,
Bayesian models face an inherent disadvantage by simultaneously optimizing for both
accuracy and UQ. This dual objective typically results in a compromised trade-off
clearly evident in RACE-DE-E, which shows substantially degraded accuracy compared
to baseline models.
However, RACE-DE-JEF fundamentally alters this picture. By incorporating force
uncertainties through our NLLJEF loss, it reduces energy errors by 3–4 times and force
errors by 2–3 times compared to RACE-DE-E. More importantly, RACE-DE-JEF achieves
test accuracies comparable to those of MSE-trained models, effectively closing the per-
formance gap between Bayesian and conventional approaches. This demonstrates that
the NLLJEF objective enables uncertainty-aware models to match the performance
of deterministic baselines, essentially providing UQ with minimal sacrifice in predic-
tive accuracy. Although absolute RMSEs remain slightly above SOTA models such
as MACE and NequIP, this marginal difference is a reasonable trade-off for obtaining
reliable uncertainty estimates that are crucial for active learning, uncertainty-driven
sampling, calibration, and OOD detection.
The Laplace approximation variant (RACE-LA), which is based on MSE-trained
models, outperformed RACE-DE-E but did not achieve the substantial accuracy
improvements of RACE-DE-JEF, particularly under OOD conditions. This further
underscores the importance of our joint energy-force uncertainty modeling approach.
2.3 Uncertainty Quantification
Before evaluating UQ performance, we first confirm that RACE achieves compet-
itive point prediction accuracy on the oBN25 dataset compared to NequIP and
MACE (Supplementary Table S8). On liquid-phase (ID) test data, all three models
achieve comparable accuracy, with MACE showing the lowest energy RMSE. On solid-
phase (OOD) data, RACE demonstrates the strongest generalization with the lowest
energy and force RMSE. The previous sections established that the RACE architec-
ture is competitive in point prediction accuracy and that NLLJEF training enables
uncertainty-aware models with minimal accuracy loss. We now evaluate whether these
uncertainty estimates are practically useful by assessing calibration, OOD detection,
and active learning performance on the oBN25 and 3BPA datasets.
2.3.1 A unified validation score metric for Bayesian machine
learning potential
We evaluate BNN models on the boron nitride (oBN25) dataset, comprising 65,000
configurations in four phases, including hexagonal BN (h-BN), cubic BN (c-BN),
11


<!-- page 12 -->
Table 4: RMSE on the 3BPA test dataset. We present the errors in energy (E, meV)
and force (F, meV/˚A) for models trained on ID (T = 300 K) configurations and
tested on both ID (T = 300K) and OOD (T = 600 K, 1200 K) configurations of the
flexible drug-like molecule 3BPA. Bold and underline indicate the best and second-
best results, respectively.
300 K
600 K
1200 K
Model
E
F
E
F
E
F
MACE [1]
3.0
8.8
9.7
21.8
29.8
62.0
Allegro [50]
3.8
13.0
12.1
29.2
42.6
83.0
BOTNet [81]
3.1
11.0
11.5
26.7
39.1
81.1
NequIP [47]
3.3
10.8
11.2
26.4
38.5
76.2
MGNN [52]
5.5
15.7
17.8
39.6
74.3
142.6
RACE
3.4
12.1
11.7
31.8
37.5
115.3
RACE-Ensemble
2.9
10.2
11.1
30.4
37.7
119.2
RACE-DE-E
17.5
52.7
43.7
98.0
171.9
232.8
RACE-DE-JEF
5.0
14.8
14.6
37.0
51.1
120.8
RACE-LA
4.8
18.2
15.3
51.0
60.8
171.8
high-pressure sp3-like liquid BN, and low-pressure sp-like liquid BN. These configura-
tions were generated under high temperature and pressure conditions, with energies
and forces computed using density functional theory (DFT) at the PBE level. To
assess OOD detection capabilities, we employ a phase-based data splitting. MLPs were
trained exclusively on 48,000 liquid-phase geometries and validated on 6,000 liquid
samples. The test set has two distinct classes, 6,000 liquid-phase configurations for ID
evaluation and 5,000 solid-phase configurations for OOD evaluation. This ensures that
solid phases remain entirely unseen during training, providing a test for the models’
ability to detect uncertain local atomic environments.
Unless otherwise stated, all Bayesian models discussed in this section are trained
using the NLLJEF loss. More architectural details and training setup are provided in
Supplementary Section F.4. For comparison, results obtained with the conventional
NLLE loss, where uncertainty is modeled only for energies, along with the correspond-
ing training details, are provided in Supplementary Section F.1 and F.3 (Tables S7
and S9).
Unlike conventional MLPs trained with MSE loss, Bayesian MLPs present unique
challenges in performance evaluation due to their dual objective of prediction accu-
racy and UQ. A comprehensive assessment requires metrics beyond traditional energy
and force test errors to capture the full spectrum of Bayesian MLPs’ capabilities.
To address this, we propose a unified validation evaluation metric that accounts for
multiple normalized criteria, including RMSE for energy and forces, calibration error
(CE) for energy and forces, and AUROC for OOD detection. Although this score does
not provide an absolute measure of model performance of Bayesian MLPs, it enables
systematic hyperparameter optimization and validation by balancing the various capa-
bilities of Bayesian MLPs. Details of the unified scoring framework are provided in
Supplementary Section F.4 (Tables S10–S13).
12


**[Table p12.1]**
| Model | 300 K 600 K 1200 K E F E F E F |
| --- | --- |
| MACE [1] Allegro [50] BOTNet [81] NequIP [47] MGNN [52] | 3.0 8.8 9.7 21.8 29.8 62.0 3.8 13.0 12.1 29.2 42.6 83.0 3.1 11.0 11.5 26.7 39.1 81.1 3.3 10.8 11.2 26.4 38.5 76.2 5.5 15.7 17.8 39.6 74.3 142.6 |
| RACE RACE-Ensemble RACE-DE-E RACE-DE-JEF RACE-LA | 3.4 12.1 11.7 31.8 37.5 115.3 2.9 10.2 11.1 30.4 37.7 119.2 17.5 52.7 43.7 98.0 171.9 232.8 5.0 14.8 14.6 37.0 51.1 120.8 4.8 18.2 15.3 51.0 60.8 171.8 |

[CAPTION] Table 4: RMSE on the 3BPA test dataset. We present the errors in energy (E, meV)


<!-- page 13 -->
Table 5: Evaluation results on the oBN25 test dataset using different UQ methods.
Values are reported for RMSE and CE of energy and force, and AUROC. Bold and
underline indicate the best and second-best results, respectively.
ERMSE
FRMSE
ECE
FCE
AUROC
Model
ID
OOD
ID
OOD
ID
OOD
ID
OOD
RACE-MVE
0.20
7.39
0.62
0.53
0.01
0.33
0.06 × 10−3
1.25 × 10−2
0.54
RACE-DE
0.14
6.94
0.53
0.37
0.03
0.21
8.49 × 10−3
5.17 × 10−2
1.00
RACE-SWAG
0.23
8.18
0.62
0.50
0.02
0.33
0.03 × 10−3
0.74 × 10−2
0.58
RACE-IVON
0.97
10.79
0.62
0.52
0.02
0.33
2.14 × 10−3
0.36 × 10−2
1.00
RACE-LLPR
0.20
2.69
0.62
0.45
0.00
0.33
3.07 × 10−1
2.97 × 10−1
0.71
We evaluate four Bayesian MLP architectures, including MVE, DE, SWAG, and
IVON, as well as LLPR, a post-hoc last-layer uncertainty method applied to the
trained RACE model (Table 5 and Figure S3). Among these, RACE-DE achieved
the most balanced performance by combining accurate energy/force predictions with
robust UQ. Specifically, it recorded the lowest energy and force RMSEs while maintain-
ing excellent OOD detection capability (AUROC = 1.00). This balanced performance
demonstrates the effectiveness of the DE architecture when coupled with the NLLJEF
loss, although the ensemble of 10 MVEs incurs a substantial computational cost. In
contrast, RACE-MVE exhibited mixed results. While it achieved competitive energy pre-
diction accuracy (second-best) and the best force calibration, it failed catastrophically
at OOD detection (AUROC = 0.54), performing no better than random chance. This
critical limitation undermines MVE’s applicability in practical scenarios where OOD
configurations are inevitable.
RACE-SWAG achieved a balanced profile, providing exceptional energy calibration
(CE = 0.02) and near-optimal force calibration while maintaining reasonable OOD
detection capability (AUROC = 0.58). Although its prediction errors slightly exceeded
those of RACE-DE, SWAG’s superior calibration makes it well-suited for applications
that prioritize uncertainty calibration over raw predictive accuracy. RACE-IVON pre-
sented a distinct trade-off profile. Despite exhibiting the highest prediction errors
among all models, it matched RACE-DE’s excellent OOD detection performance
(AUROC = 1.00). This result suggests that variational inference approaches can excel
in identifying distributional shifts even when predictive accuracy is compromised, high-
lighting a potential advantage for safety-critical applications where detecting OOD
samples is paramount.
RACE-LLPR provides a complementary perspective as a post-hoc method requiring
no additional training. It achieved the lowest OOD energy RMSE (2.69 vs. 6.94 for
RACE-DE) and the best energy calibration in the ID regime (CE = 0.00), demonstrating
that last-layer feature-space distances are effective for detecting distributional shifts in
energy predictions. However, LLPR showed force calibration of CE ∼3×10−1, roughly
two orders of magnitude worse than the NLLJEF-trained models. This is because
LLPR’s last-layer ensemble perturbs only the energy readout weights, propagating
uncertainty to forces only indirectly through differentiation, which fails to capture the
full directional covariance structure of atomic forces. This limitation highlights the
13


**[Table p13.1]**
| Model | E F E F RMSE RMSE CE CE AUROC ID OOD ID OOD ID OOD ID OOD |
| --- | --- |
| RACE-MVE RACE-DE RACE-SWAG RACE-IVON RACE-LLPR | 0.20 7.39 0.62 0.53 0.01 0.33 0.06 × 10−3 1.25 × 10−2 0.54 0.14 6.94 0.53 0.37 0.03 0.21 8.49 × 10−3 5.17 × 10−2 1.00 0.23 8.18 0.62 0.50 0.02 0.33 0.03 × 10−3 0.74 × 10−2 0.58 0.97 10.79 0.62 0.52 0.02 0.33 2.14 × 10−3 0.36 × 10−2 1.00 0.20 2.69 0.62 0.45 0.00 0.33 3.07 × 10−1 2.97 × 10−1 0.71 |

[CAPTION] Table 5: Evaluation results on the oBN25 test dataset using different UQ methods.


<!-- page 14 -->
advantage of explicitly modeling the 3 × 3 force covariance via NLLJEF, particularly
for applications where force uncertainty is critical such as molecular dynamics stability
assessment and active learning with force-based acquisition.
a
c
b
d
Fig. 2: Predicted–empirical error scatter plots for the oBN25 dataset using a
RACE-MVE, b RACE-DE, c RACE-SWAG, and d RACE-IVON. Blue dots represent liquid
BN (ID), and red dots represent solid BN (OOD). Black dashed line represents the
reference diagonal of |∆y| = σpred, and the colored bands indicate the reference quan-
tile intervals derived from the ideal folded-normal distribution. Data points that are
well aligned with the quantile bands imply good consistency between uncertainty and
error, while points located above or below the bands indicate overconfidence or under-
confidence, respectively.
These performance differences become more apparent through UQ analysis.
Figure 2 presents predicted-versus-empirical error scatter plots for the oBN25 test set.
RACE-DE and RACE-IVON demonstrate well-calibrated uncertainties, with errors closely
following the expected quantile bands in both the ID and OOD regimes, indicating
that their predicted uncertainties reliably correlate with actual errors. Conversely,
RACE-MVE and RACE-SWAG exhibit a systematic overconfidence which is mild in the ID
regime (points shifted upward from ideal bands) and severe in the OOD regime due
to dramatic uncertainty underestimation.
14

[CAPTION] Fig. 2: Predicted–empirical error scatter plots for the oBN25 dataset using a

[CAPTION] Figure 2 presents predicted-versus-empirical error scatter plots for the oBN25 test set.


<!-- page 15 -->
a
c
b
d
Fig. 3: Calibration plots for energy predictions in a liquid phase and b solid phase,
and for force predictions in c liquid phase and d solid phase of the oBN25 dataset.
Methods compared: RACE-MVE, RACE-DE, RACE-SWAG, and RACE-IVON. Curves above
the diagonal indicate underconfidence, whereas curves below indicate overconfidence.
Calibration analysis (Figure 3) further elucidates the behavior of the model in dif-
ferent prediction tasks. For liquid-phase energies (ID), RACE-DE and RACE-IVON show a
slight underconfidence (observed confidence exceeding expected), while RACE-MVE and
RACE-SWAG show overconfidence. This overconfidence intensifies dramatically for solid-
phase energies (OOD), where all models struggle with calibration, although RACE-DE
maintains relatively better performance. Interestingly, all models consistently exhibit
underconfidence in force predictions regardless of phase, suggesting that quantifica-
tion of force uncertainty remains challenging even with NLLJEF (detailed results in
Supplementary Table S14).
2.3.2 Bayesian Active Learning by Disagreement
We investigate the data efficiency of uncertainty-guided active learning based on
NLLJEF by examining how strategically selected active learning data impact model
performance. Starting with RACE-DE-JEF trained on only 500 configurations from the
300 K trajectory of the 3BPA training dataset, we evaluate the MLP’s ability to
improve when augmented with limited additional data (10, 20, 50, 100, or 200) from
OOD regimes (600 K and 1200 K trajectories). This setup mimics realistic scenarios
15

[CAPTION] Fig. 3: Calibration plots for energy predictions in a liquid phase and b solid phase,


<!-- page 16 -->
where computational resources for generating new training data through expensive ab
initio calculations are limited.
To assess uncertainty-guided selection, we compared four data acquisition strate-
gies: random sampling, farthest point sampling (FPS) [84, 85], high uncertainty (max
variance) selection, and Bayesian active learning by disagreement (BALD). The FPS
selects configurations farthest distant in sorted pairwise-distance descriptor space as
a simple heuristic. The BALD acquisition function quantifies the mutual information
between the model predictions and model parameters. We derived the BALD score
function for the DE of MVE (Supplementary Section H), which is given as
αBALD(x) = I[y, θ|x, D] = H[y|x, D] −Eθ∼p(θ|D)[H[y|x, θ]]
= 1
2
"
log(σ2
total(x)) −1
M
M
X
m=1
log(σ2
m(x))
#
.
(1)
This metric identifies the configurations where the model exhibits high epistemic
uncertainty. By selecting points with maximum BALD scores, we prioritize data that
maximize information gain, thus accelerating model improvement relative to random
sampling.
We compared random sampling against BALD-based uncertainty-guided selection
across varying data budgets (10, 20, 50, 100, and 200 configurations). We implemented
a balanced acquisition strategy, selecting half of the configurations based on maxi-
mum energy BALD scores and half based on maximum force BALD scores. Table 6
demonstrates that uncertainty-guided selection outperforms random sampling, partic-
ularly in the low-data regime. With only 10 additional configurations from the 1200 K
trajectory, BALD selection reduced energy RMSE by 11 % and force RMSE by 17
% compared to random selection. The high uncertainty selection also outperforms
random sampling in most regime, confirming that uncertainty-guided selection is gen-
erally beneficial. In contrast, farthest point sampling performed worse than random
sampling, indicating that geometric diversity alone is insufficient for active learning.
BALD-guided selection consistently achieves the best performance across all budgets.
Notably, BALD-guided active learning achieves test errors comparable to random sam-
pling with twice as many configurations, effectively halving the data requirements for
equivalent performance. This advantage persists across all data budgets but gradually
diminishes as more data become available. At 200 configurations, both strategies con-
verge to similar performance in the 600K test set, although BALD maintains a clear
advantage in the more challenging 1200 K OOD test set.
We also evaluate active learning protocols enabled by NLLJEF’s joint uncertainty
modeling. We compared four data selection strategies using a fixed budget of 10
configurations, including BALDE (selecting configurations with the highest energy
uncertainty), BALDF (the highest force uncertainty), BALDEF (50/50 split between
energy and force uncertainties), and random sampling (Table 7).
The balanced BALDEF strategy proved to be the most effective, achieving superior
energy and force accuracy at both temperatures. Single-objective selection strategies
showed predictable trade-offs. BALDE minimized energy errors but offered marginal
16


<!-- page 17 -->
Table 6: Active learning results on the 3BPA test dataset with
varying data budgets. RMSE for energy (E, meV) and force
(F, meV/˚A). R: Random, F: Farthest Point, U: High Uncer-
tainty, B: High BALD score. Bold indicates the best result per
budget.
600 K
1200 K
R
F
U
B
R
F
U
B
Baseline
E: 14.6
F: 37.0
E: 51.1
F: 120.8
+10
E
13.9
13.6
12.2
12.2
40.8
39.6
36.3
36.3
F
33.3
34.2
31.8
30.1
102.4
108.6
97.1
85.2
+20
E
13.5
12.9
12.1
11.5
36.2
37.1
30.8
31.9
F
31.0
33.0
28.8
28.8
87.4
102.7
74.4
78.3
+50
E
12.2
11.8
11.1
11.1
30.5
32.9
25.9
26.0
F
27.5
29.7
26.0
25.8
70.7
84.4
60.9
60.4
+100
E
11.0
11.2
10.2
10.1
25.3
28.0
21.6
21.9
F
23.8
26.5
23.7
23.3
55.6
68.8
51.4
49.7
+200
E
8.2
10.1
8.7
8.8
20.7
24.7
18.8
18.8
F
20.0
23.9
20.5
20.0
45.6
59.9
42.7
41.3
force improvements, while BALDF showed the inverse pattern. This demonstrates that
NLLJEF’s dual uncertainty quantification enables more comprehensive refinement of
the model than traditional energy-only approaches. Crucially, all uncertainty-guided
strategies significantly outperformed random sampling.
Beyond substantially improving the accuracy of NLL-based training and closing
the performance gap with MSE-trained models, NLLJEF also enables highly efficient
active learning workflows through BALD-based UQ. This dual capability positions
NLLJEF as a practical advance for Bayesian MLP, addressing the accuracy-uncertainty
trade-off while providing principled guidance for data-efficient model refinement.
Table 7: Active learning results with different data selection
strategies (random vs. high BALD score) on the 3BPA test
dataset. RMSE for energy (E, meV) and force (F, meV/˚A) on
600 K and 1200 K test sets. Bold and underline indicate the best
and second-best results, respectively.
Baseline
+10
BALDE
BALDF
BALDEF
Random
600 K
E
14.6
12.2
12.5
12.2
13.9
F
37.0
32.6
30.7
30.1
33.3
1200 K
E
51.1
40.2
36.9
36.3
40.8
F
120.8
103.4
87.7
85.2
102.4
17


**[Table p17.1]**
|  | 600 K R F U B | 1200 K R F U B |
| --- | --- | --- |
| Baseline | E: 14.6 F: 37.0 | E: 51.1 F: 120.8 |
| E +10 F | 13.9 13.6 12.2 12.2 33.3 34.2 31.8 30.1 | 40.8 39.6 36.3 36.3 102.4 108.6 97.1 85.2 |
| E +20 F | 13.5 12.9 12.1 11.5 31.0 33.0 28.8 28.8 | 36.2 37.1 30.8 31.9 87.4 102.7 74.4 78.3 |
| E +50 F | 12.2 11.8 11.1 11.1 27.5 29.7 26.0 25.8 | 30.5 32.9 25.9 26.0 70.7 84.4 60.9 60.4 |
| E +100 F | 11.0 11.2 10.2 10.1 23.8 26.5 23.7 23.3 | 25.3 28.0 21.6 21.9 55.6 68.8 51.4 49.7 |
| E +200 F | 8.2 10.1 8.7 8.8 20.0 23.9 20.5 20.0 | 20.7 24.7 18.8 18.8 45.6 59.9 42.7 41.3 |


**[Table p17.2]**
|  | Baseline | +10 BALD BALD BALD Random E F EF |
| --- | --- | --- |
| E 600 K F | 14.6 37.0 | 12.2 12.5 12.2 13.9 32.6 30.7 30.1 33.3 |
| E 1200 K F | 51.1 120.8 | 40.2 36.9 36.3 40.8 103.4 87.7 85.2 102.4 |

[CAPTION] Table 6: Active learning results on the 3BPA test dataset with

[CAPTION] Table 7: Active learning results with different data selection


<!-- page 18 -->
2.3.3 Recalibration Result
While BNNs provide predictive uncertainties, these estimates are often miscalibrated,
especially under distributional shift. To address this, we applied a post-hoc recalibra-
tion method following Kuleshov et al. [12], which learns a monotonic mapping from
the nominal confidence levels to calibrated ones. This procedure is model-agnostic and
can be applied to any Bayesian predictor without retraining.
Figure 4 and Table 8 present CEs before and after recalibration for both liquid
and solid BN phases. Recalibration improves calibration across most models and con-
ditions, with particularly pronounced improvements under severe distribution shifts.
For solid-phase energy predictions, where OOD effects are most severe, RACE-DE
and RACE-IVON achieve substantial CE reductions, indicating that their recalibrated
predictive variances now properly scale with actual prediction errors. In contrast,
RACE-MVE exhibits inconsistent recalibration behavior. Its CE of solid-phase energy
remains unchanged, whereas its CE of liquid-phase forces deteriorates after recalibra-
tion. This suggests fundamental limitations in MVE’s uncertainty modeling framework
that prevent effective post-hoc calibration.
Before recalibration (Figure 4e), despite the wide predicted intervals, none of the
test points fell within the nominal 90 % confidence bounds, which is a severe miscali-
bration. After recalibration (Figure 4f), the intervals become correctly calibrated, with
precisely 9 of 10 samples falling within the 90 % interval. This transformation demon-
strates that recalibration not only improves the numerical metrics but fundamentally
restores the statistical reliability of the uncertainty estimates.
Among the recalibrated models, RACE-DE consistently achieves the lowest CE in
both energy and force predictions, followed by RACE-IVON. Furthermore, the relative
error reduction ratio between before and after recalibration is the highest for RACE-DE
and the second highest for RACE-IVON, underscoring their robustness to OOD set-
tings. In contrast, RACE-MVE and RACE-SWAG yield marginal improvements only in
liquid-phase (ID) predictions and remain poorly calibrated in the solid phase (OOD).
Detailed per-model values and relative reduction ratios and Calibration plot are
provided in Supplementary Section F.7 (Table S15 and Figures S5–S8).
Table 8: CE on the oBN25 test dataset after post-hoc recalibration.
Results are shown separately for energy and force predictions in the
liquid and solid phases. Bold and underline indicate the best and second-
best results, respectively.
Liquid
Solid
Model
Energy
Force
Energy
Force
RACE-MVE
1.1 × 10−4
6.5 × 10−5
3.3 × 10−1
6.9 × 10−5
RACE-DE
6.0 × 10−5
3.8 × 10−5
3.3 × 10−4
4.9 × 10−5
RACE-SWAG
1.8 × 10−4
3.3 × 10−5
3.3 × 10−1
1.4 × 10−4
RACE-IVON
6.4 × 10−5
2.3 × 10−4
1.4 × 10−4
9.0 × 10−5
18


**[Table p18.1]**
| Model | Liquid Energy Force | Solid Energy Force |
| --- | --- | --- |
| RACE-MVE RACE-DE RACE-SWAG RACE-IVON | 1.1 × 10−4 6.5 × 10−5 6.0 × 10−5 3.8 × 10−5 1.8 × 10−4 3.3 × 10−5 6.4 × 10−5 2.3 × 10−4 | 3.3 × 10−1 6.9 × 10−5 3.3 × 10−4 4.9 × 10−5 3.3 × 10−1 1.4 × 10−4 1.4 × 10−4 9.0 × 10−5 |

[CAPTION] Figure 4 and Table 8 present CEs before and after recalibration for both liquid

[CAPTION] Table 8: CE on the oBN25 test dataset after post-hoc recalibration.


<!-- page 19 -->
a
c
b
d
e
f
Fig. 4: CE of Bayesian models before and after post-hoc recalibration on the oBN25
test dataset. Results are reported separately for energy and force in liquid-phase (ID)
and solid-phase (OOD). Panels: a RACE-MVE, b RACE-DE, c RACE-SWAG, d RACE-IVON.
Panels e and f illustrate 90% confidence intervals for RACE-DE on liquid BN before
and after recalibration, respectively, showing how post-hoc adjustment corrects the
raw Bayesian intervals.
19

[CAPTION] Fig. 4: CE of Bayesian models before and after post-hoc recalibration on the oBN25


<!-- page 20 -->
3 Discussion
This work addresses fundamental challenges in machine learning potentials by devel-
oping a comprehensive Bayesian framework that achieves both accurate predictions
and reliable uncertainty quantification. Through extensive benchmarking across five
datasets (QM9, rMD17, PSB3, 3BPA, and oBN25), we demonstrate three key contri-
butions: (i) the joint energy-force NLL (NLLJEF ) loss function that systematically
models uncertainties in both quantities, (ii) RACE-based approximate Bayesian archi-
tectures (DE, SWAG, IVON, LA) with an 8-head MVE module for comprehensive
uncertainty estimation, and (iii) highly efficient active learning protocols using energy-
and force-based BALD scores for optimal data selection.
Our RACE architecture delivers competitive performance on standard bench-
marks. In the QM9 dataset, RACE achieves competitive MAEs for thermodynamic
properties, including the third-lowest ZPVE and comparable heat capacity(CV ) among
a wide range of equivariant and invariant baseline MLPs.
The NLLJEF loss function represents a practical and effective extension of equivari-
ant MLIPs, explicitly modeling the full 3×3 force covariance structure alongside energy
uncertainty within a multi-task Gaussian NLL framework. In the rMD17, RACE-DE-JEF
reduces errors by 50–60% compared to energy-only NLL training (RACE-DE-E),
achieving accuracies comparable to MSE-trained models while providing calibrated
uncertainty estimates. This substantial reduction of the accuracy-uncertainty trade-off
is particularly significant because force predictions dominate both the computa-
tional cost and numerical stability of molecular dynamics simulations. The systematic
quantification of force uncertainties enables the reliable detection of unphysical pre-
dictions and potentially unstable simulation regimes which are critical capabilities for
trustworthy autonomous simulations.
Our systematic evaluation of multiple Bayesian MLPs on oBN25 reveals distinct
advantages for different applications. Despite its computational cost, DE achieves the
most balanced performance with excellent OOD detection (AUROC = 1.00) and supe-
rior prediction accuracy. IVON matches DE’s OOD detection and calibration despite
higher prediction errors, suggesting that variational approaches excel at identifying dis-
tributional shifts even when accuracy is compromised. These complementary strengths
enable practitioners to select methods optimized for their specific requirements.
The practical utility of our framework is exemplified through active learning, where
BALD-guided selection achieves equivalent performance using only half the training
data compared to random sampling. The balanced BALDEF strategy, uniquely enabled
by NLLJEF’s dual UQ for energy and forces, outperforms single-objective approaches
(energy- or force-only) by 10–17% in force prediction accuracy on 3BPA. Furthermore,
post-hoc recalibration enhances model reliability, reducing calibration errors by orders
of magnitude for well-performing models like RACE-DE and RACE-IVON.
An important direction for condensed-phase applications is the extension of uncer-
tainty quantification to stress predictions, which are essential for cell optimization and
constant-pressure molecular dynamics. Our framework naturally accommodates this
extension: analogous to the 3 × 3 force covariance modeling presented in this work,
20


<!-- page 21 -->
stress uncertainties can be captured through a covariance matrix over the six indepen-
dent components of the symmetric stress tensor, with positive definiteness enforced
via Cholesky decomposition. We leave this extension as future work.
This work demonstrates that the principled UQ in MLPs does not compromise
predictive accuracy. By integrating architectural innovations with the novel NLLJEF
loss function, we demonstrate that Bayesian MLPs (DE, SWAG, LA, IVON) can
achieve competitive accuracy while providing the uncertainty estimates essential for
reliable materials modeling. Accurate predictions, calibrated uncertainties, efficient
active learning, and robust OOD detection position our E(3) equivariant Bayesian
MLPs as a foundational method for next-generation uncertainty-aware atomistic sim-
ulations. Our framework opens new possibilities for AI-driven materials research,
including autonomous materials discovery, Bayesian optimization, foundation model
development, and other applications where quantifying predictive confidence is as
critical as the predictions themselves.
4 Methods
4.1 Atomic Cluster Expansion
The ACE framework offers a systematic way to create high-order polynomial single-
edge basis functions [33]. These single-edge basis functions, also called features in
the neural network, can be computed at a fixed cost per function, regardless of the
order. The structure of a single-edge basis in ACE mirrors that of atomic orbitals
used in electronic structure calculations. It combines radial and angular components,
expressed mathematically as
ϕv(rij) =
√
4πRnl(rij)Y m
l (ˆrij).
(2)
Here, the term v = (n, l, m) encapsulates the quantum numbers n, l, and m, which
define the specific orbital being described. Rnl represents the radial basis functions
that depend on the distance rij = |rij|, where rij = rj −ri, while Y m
l
denotes the
spherical harmonics that depend on the edge direction ˆrij = rij/rij.
With the atomic density of an elemental material
ρi =
X
j
δ(r −rij),
(3)
we define the atomic base as the projection of the basis functions on the atomic density
Aiv = ⟨ρi|ϕv⟩=
X
j∈N (i)
ϕv(rij),
(4)
with a local neighborhood N(i) = {j|rij ≤rcut}.
21


<!-- page 22 -->
The atomic energy Ei with the atomic density ρi can be expressed as a polynomial
in Aiv,
Ei =
X
v1
c(1)
v Aiv1 + 1
2
X
v1v2
c(2)
v1v2Aiv1Aiv2
+ 1
3!
X
v1v2v3
c(3)
v1v2v3Aiv1Aiv2Aiv3 + · · · .
(5)
4.2 Tensor Field Networks
Tensor field networks (TFNs) are neural networks that work with point clouds [45].
These networks transform point clouds while preserving SE(3)-equivariance, which
includes 3D rotations and translations. For point clouds, the input is a vector field
A : R3 →Rd, defined as:
A(r) =
N
X
j=1
Ajδ(r −rj),
(6)
where δ is the Dirac delta function and {rj} are the 3D point coordinates. Each
Aj represents a concatenation of vectors corresponding to various rotation orders
l, where the subvector associated with a specific rotation order l is denoted as
Al
j. A TFN layer performs convolution using a learnable weight kernel, denoted as
Wlk : R3 →R(2l+1)×(2k+1), which maps a vector field in three-dimensional space to a
matrix that facilitates the transformation from rotation order k to l. Here, Rd refers
to a vector in d-dimensional space, while R represents a real number.
During convolution Φlk in the TFN layer, the interatomic interactions between
the ith atom and its neighbors N(i) are considered. Based on this, we assumed that
if the input features Ak,η
in,j to the TFN layer exhibit η-body characteristics, the output
features Al,η+1
out,i will exhibit (η+1)-body characteristics.
Al,η+1
out,i = wllAl,η
in,i +
X
k≥0
X
j∈N (i)
Φlk(rij)Ak,η
in,j.
(7)
The first term is referred to as self-interaction, when k = l and J = 0, which reduces
the basis kernel to a scalar w multiplied by the identity, Wll = wll1. Here, the
kernel Wlk lies in the span of an equivariant basis {Wlk
J }k+l
J=|k−l|. The kernel is a linear
combination of these basis kernels. Mathematically this is
Φlk(rij) =
k+l
X
J=|k−l|
Rlk
J (rij)Y lk
J (ˆrij),
(8)
22


<!-- page 23 -->
where
Ylk
J (ˆrij) =
J
X
m=−J
Y m
J (ˆrij)Qlk
Jm.
(9)
A learnable radial basis function Rlk
J
: R≥0 →R is obtained by feeding a set of
radial features that embed the radial distance rij using Bessel functions multiplied
by a smooth polynomial cutoff to a multilayer perceptron. An angular basis kernel
Ylk
J : R3 →R(2l+1)×(2k+1) is formed by taking a linear combination of Clebsch-Gordon
matrices Qlk
Jm of shape (2l + 1) × (2k + 1). Each angular basis kernel Ylk
J completely
constrains the form of the learned kernel in the angular direction.
The single-edge basis ϕv(rij) in Eq. (2) and the atomic basis Aiv in Eq. (4), defined
in the ACE framework, are estimated using the TFN basis kernel Φ provided in Eq. (8)
and the second term P
j Φ(rij) of the TFN layer in Eq. (7), respectively. Therefore,
when the input features Al,2
in,j, which have 2-body characteristics like Aiv1, are given
in the TFN, the output features Al,3
out,i are expected to show 3-body characteristics
like Aiv1Aiv2.
The atomic basis Aiv is not rotationally invariant [33]. To address this, we cre-
ate a set of functions that remain invariant under permutations and rotations. We
achieve this by averaging the atomic basis Aiv over the three-dimensional rotational
group O(3) in terms of the ACE framework:
Biv =
X
v′
Cvv′Aiv′ .
(10)
Here the matrix of Clebsch-Gordan coefficients Cvv′ is extremely sparse. Clebsch-
Gordan coefficients are used to combine atomic basis functions (or features) in a way
that ensures the resulting features transform predictably under rotations. This is key
to constructing rotationally equivariant features in models, ensuring that the physical
properties of the system are preserved under symmetry operations such as rotations.
In the TFN layer, the rotationally equivariant features Bl,η
out,i can be obtained from
Al,η
out,i via tensor product of features as
Bl,η
out,i = Al,η
out,i ⊗a(1)
i ,
(11)
where a(1)
i
is the 1-body learnable feature.
4.3 Architecture
An overview of the RACE architecture, including a detailed algorithmic flowchart
annotated with tensor dimensions at each step, is shown in Supplementary Section A
(Figure S1). The potential energy of the system, denoted Epot, is calculated by adding
the atomic energy Ei for all atoms in the system. To guarantee energy conservation
during molecular dynamics simulations, forces are obtained as the gradients of the
23


<!-- page 24 -->
predicted potential energy with respect to the atomic positions: fi = −∇iEpot with
Epot =
Natoms
X
i=1
Ei =
Natoms
X
i=1
Nlayers
X
t=1
E(t)
i .
(12)
The predicted potential energy is invariant under translation, reflection, and rotations,
whereas the forces fi and the internal features of the geometric tensors in the neural
network are equivariant to rotation and reflection. The stress of the system is obtained
as the product summation of position and forces with volume gradient with respect to
the cell as S = 1
V
 
∂Epot
∂h h⊤+ P
i ri ⊗∂Epot
∂ri
 
, where Epot is the total energy, h ∈R3×3
is the cell matrix composed of lattice vectors, ri is the position vector of the atom i,
and V is the volume of cell.
The architecture of this message-passing-based MLP model is composed of an
embedding block and multiple interaction layers. Within each interaction layer, a
readout block is incorporated to estimate the atomic energy of the node i.
4.3.1 Embedding block
Atomic features are modeled as learnable node features, denoted as (A(η)
i
∈RF ),
where F specifies the dimensionality of the node features, and (η) indicates the current
layer in the model. The initial node features, denoted A(0)
i , are established using an
embedding associated with atomic numbers {Zi}. This relationship is expressed as
A(0)
i
= aZi, where aZ represents atomic-type embeddings. These embeddings are
initially assigned random values and subsequently optimized through the training
procedure.
The single-edge basis ϕv(rij) in Eq. (2) is decomposed into radial and angular com-
ponents. The radial component for the interatomic distance rij [70] are defined using
Bessel basis functions Bn(x) =
q
2
r3
c
j0(nπx)
|j1(nπ)| and a polynomial envelope function fenv
as
˜eRBF(rij) = Bn(rij/rc)fenv(rij, rc) .
(13)
Here, j0 and j1 are spherical Bessel functions of the first kind. The edge features eij,
which encode the angular components, are obtained as Y (l)
m (ˆrij).
The 1-body learnable feature, represented as a(1), is updated through an equiv-
ariant linear layer applied to the initial node features, A(0). Whithin this linear layer,
features associated with different rotation order (l) are separately transformed using
separate weights:
a(1) = fLin(A(0)).
(14)
4.3.2 Interaction Layers
The atomic energy Ei is influenced by the local chemical environment, represented
by {eij, ˜eRBF
ij
: j ∈Ni}. To accurately reflect this dependence, the node features
24


<!-- page 25 -->
incorporate the corresponding edge information. Therefore, the interaction layer is
designed to capture the many-body interatomic interactions effectively. In the (η)-th
interaction layer, the node features are updated using the scheme of ResNet [86]:
A(η) = fSI(A(η−1)) + f(A(η−1), eij, ˜eRBF
ij
) as outlined in the TFN layer (Eq. (7)).
The function fSI denotes the self-interaction layer, with weights trained separately for
each atomic number. The function f consists of a sequence of operations, including an
equivariant linear layer fLin, an interatomic continuous-filter convolution layer fConv
that adheres to the message-passing convolution framework, and a final atom-wise
equivariant linear layer. In the convolution layer fConv, the node features are updated
according to the following equation:
A(η−1)
i
=
X
j∈Ni
h
(eij ⊗A(η−1)
j
) ⊙M(˜eRBF
ij
)
i
,
(15)
where a multilayer perceptron, denoted as M, encodes the learnable radial function
Rlk
J and operates the interatomic distance-dependent radial basis vectors ˜eRBF
ij
to cap-
ture distance-based interatomic interactions. The symbol ⊗signifies the rotationally
equivariant tensor product defined by the angular basis kernel in Eq. (9). In con-
trast, ⊙represents the element-wise multiplication between the learnable radial basis
function and the angular basis kernel, as outlined in Eq. (8).
4.3.3 Readout Layer
In the (η)-th interaction layer, the node features A(η) are improved by considering
the local chemical environment through pooling over the neighboring features, which
are updated via the message passing convolutions. However, since these updated node
features A(η) lack rotational invariance, rotationally equivariant node features B(η)
are computed using an equivariant tensor product. This tensor product combines the
(η + 1)-body node features A(η) and 1-body node features a(1), such that B(η) =
A(η) ⊗a(1). To calculate the energy value E(η+1)
i
for node i at the (η)-th interaction
layer, the invariant part of the node features is mapped to node energy via the readout
function:
E(η+1)
i
=
X
˜k
W (η)
˜k
(B0,(η)
i,˜k
),
(16)
where W (η)
˜k
denotes the readout weights, and B(η)
i,˜k represents the ˜k-th element of
the i-th node feature at the (η)-th layer. To ensure invariance of the node energy
E(η+1)
i
, the readout layer is only based on the invariant features, which correspond to
those of rotational order l = 0. These invariant features are obtained by applying the
transformation: B0,(η+1) = Gate(f l=0
Lin (B(η+1))), where Gate(·) refers to an equivariant
SiLU-based gate.
In the first (η = 1) interaction layer, the updated node features A(1) exhibit two-
body characteristics and describe the atomic basis Aiv1, as represented by the first
term in Eq. (5). During the second interaction layer, the input node features A(1)
25


<!-- page 26 -->
are pooled over neighboring features through message passing convolutions, resulting
in updated output features A(2). These updated node features possess three-body
characteristics and describe the atomic basis Aiv1Aiv2, as represented by the second
term in Eq. (5).
4.3.4 Comparison against other models
Equivariant architectures are distinguished by how they partition the total potential
energy. For example, models such as NequIP define the potential energy as the sum
of atomic contributions (E = P
i Ei), where each Ei is predicted from the features
of the final layer node A(Nlayer). Alternatively, architectures in the Allegro employ a
pairwise decomposition (E = P
ij Eij), with contributions derived directly from edge
features Aij.
The MACE architecture uses a more complex scheme, summing contributions from
multiple interaction orders ν at each layer t, expressed as E = P
itν E(ν,t)
i
. This
method constructs a full set of higher-order features, B(1) through B(Nν), at every
layer to capture interactions up to (Nν+1)-body terms. In contrast, our proposed
RACE architecture streamlines this process. At each layer t, RACE restratifies a single
higher-order feature B(t) that corresponds specifically to the (t + 1)-body interaction
energy, E(t+1)
i
. The total potential energy for RACE is then defined by summing these
restratified higher-order contributions, as detailed in Eq. (12).
4.4 Force Uncertainty and Joint Energy-Force Negative
Log-Likelihood Loss
In atomistic simulations, forces are obtained as the negative gradients of the potential
energy with respect to the atomic positions. For the atom i and the spatial component
α ∈{x, y, z}, the force is defined as:
fiα = −∂E
∂riα
,
(17)
where E is the predicted total energy and riα is the α-th component of the posi-
tion of atom i. Thus, the predicted force can be derived from the energy model by
differentiation.
To quantify the uncertainty in force predictions, we assume that the components
of the force vector fi = (fix, fiy, fiz) follow a multivariate normal distribution. Further
details are provided in Supplementary Section G. The covariance matrix Σ captures
both the variance of individual components and their correlations
Σ = LL⊤+ ϵI,
(18)
where L is a lower triangular matrix (i.e., obtained by Cholesky decomposition) and ϵI
is a small diagonal jitter term added for numerical stability. Details on the rotational
properties of the predicted force covariance are provided in Supplementary Section G.6.
26


<!-- page 27 -->
The matrix L is modeled as:
L =


σ1 0
0
σ6 σ2 0
σ5 σ4 σ3

,
(19)
resulting in the full symmetric covariance matrix:
Σ =


σ2
xx σ2
xy σ2
xz
σ2
xy σ2
yy σ2
yz
σ2
xz σ2
yz σ2
zz

.
(20)
The uncertainty in each orthogonal force component is given by the diagonal
elements of Σ:
σ2
fiα(x) = σ2
αα,
α ∈{x, y, z}.
(21)
Assuming that the true force vector is yfi, the predictive distribution becomes:
p (yfi|µfi, Σfi) =
1
p
(2π)3 det Σfi
exp
 
−1
2(yfi −µfi)⊤Σ−1
fi (yfi −µfi)
 
.
(22)
Taking the negative log-likelihood of the above gives the loss function for force
prediction of atom i:
Lforce = (yfi −µfi)⊤Σ−1
fi (yfi −µfi) + log(det Σfi).
(23)
To jointly optimize the predictions and uncertainties of both energy and forces, we
introduce the joint energy-force negative log-likelihood loss function,
Ltotal = λE · (yE −µE)2
σ2
E
+ λF ·
"
1
N
N
X
i
(yfi −µfi)⊤Σ−1
fi (yfi −µfi)
#
+ log
 σ2
E
 
+ 1
N
N
X
i
log(det Σfi),
(24)
where λE and λF are hyperparameters that control the relative contributions of the
energy and force loss terms, respectively. This NLL loss function becomes a practical
and effective multi-task Gaussian NLL framework with full force covariance modeling.
When the force uncertainty Σfi is fixed to 1, the force loss term reduces to a mean
squared error (MSE). In this case, the combination of the energy NLL loss and the
force MSE loss is denoted as NLLE loss. Furthermore, if both σ2
E and Σfi are fixed to
1, the overall loss reduces to the standard MSE loss for both energy and forces.
27


<!-- page 28 -->
4.5 Uncertainty Quantification
This section introduces two classes of UQ used in this work: (i) a non-Bayesian method
that directly models output uncertainty and (ii) approximate Bayesian approaches
that place a distribution over weights to capture epistemic uncertainty. UQ improves
predictive reliability and interpretability, allowing applications such as active learning,
OOD detection, and recalibration.
4.5.1 Mean-Variance Estimation
Uncertainty estimation is a critical capability for neural networks. Unlike BNNs that
model a posterior over parameters, MVE [31] provides a non-Bayesian alternative
by augmenting the network to output a predictive mean µ∗(x) and a predictive
variance σ2
∗(x) for each input x. At first glance, one might consider deriving force
uncertainty from the differentiated kernel, as shown in the proof in Supplementary
Section G.1. However, this approach fails due to a fundamental limitation of the
MVE model. In MVE(f), predictions at different points are independent, denoted
by f(x) ⊥f(x′) for x ̸= x′, where ⊥indicates statistical independence. This inde-
pendence implies that the corresponding Gaussian process has a degenerate kernel of
the form K(x, x′) = 1[x=x′]σ2  
x+x′
2
 
, where 1[x=x′] is the indicator function, which
equals 1 when x = x′ and 0 otherwise. Hence, the kernel is non-zero only with x = x′.
Consequently, when we apply Theorem 1.1 from Supplementary Section G.1, the gra-
dient ∇xK(x) becomes ill-defined at the discontinuity, preventing us from obtaining
meaningful force uncertainties through kernel differentiation. We provide a detailed
proof of this in Supplementary Section G.
In this work, all Bayesian approaches employ MVE modules with either 2 or 8 out-
put heads. The 2-head module outputs mean energy and energy variance (µE, σ2
E) and
is used when modeling energy uncertainty alone. The 8-head module outputs energy
predictions plus force covariance components (µE, σ2
E, σL1, σL2, σL3, σL4, σL5, σL6),
where σLi represents the i-th unique element of a (3 × 3) lower triangular matrix L.
This matrix is used to construct the full symmetric force covariance matrix for pos-
itive definiteness. The 8-head architecture is used when jointly modeling energy and
force uncertainties together, enabling comprehensive uncertainty quantification across
both quantities. To guarantee strictly positive variance predictions, we apply softplus
or exponential activation functions to the variance output.
The training process for Bayesian models utilizes an NLL loss function, mathe-
matically defined as
NLL(x, y) = 1
2 ln σ2
∗(x) + (yE −µ∗(x))2
2σ2∗(x)
.
(25)
This loss function simultaneously optimizes both the mean prediction µ(x) and the
uncertainty estimation σ2(x). The first term penalizes extreme variance predictions,
and the second term measures the discrepancy between the predicted µ(x) and actual
values yE, weighted by the predicted variance.
28


<!-- page 29 -->
4.5.2 Deep Ensemble
In the DE [28] approach, multiple neural networks are trained independently, each
initialized with different random weights. When using MVE models within a DE, each
network m predicts both the mean µθm(x) and the variance σ2
θm(x) independently,
capturing the aleatoric uncertainty at the individual model level.
For a given input x, the predictive mean is computed by averaging the means
predicted by all models in the ensemble as
µ∗(x) = 1
M
M
X
m=1
µθm(x),
(26)
where M is the number of MVE models in the ensemble and µθm(x) is the mean
predicted by the m-th model. The total predictive variance, which accounts for both
model uncertainty and data uncertainty, is computed as:
σ2
∗(x) = 1
M
M
X
m=1
 σ2
θm(x) + µ2
θm(x)
 
−µ2
∗(x).
(27)
While a single MVE accounts for aleatoric uncertainty (i.e., inherent data noise),
an ensemble of independently trained MVE models can be employed to also quantify
epistemic uncertainty, which arises from uncertainty in the model parameters them-
selves. In this framework, epistemic uncertainty is captured by the variance between
the predictions of the different models in the ensemble. This combined approach allows
for a more robust decomposition of the total predictive uncertainty, yielding more
reliable confidence estimates, and improving overall model accuracy.
4.5.3 Stochastic Weight Averaging Gaussian
SWAG [13] is a method for quantifying predictive uncertainty in neural networks by
extending Stochastic Weight Averaging (SWA) [87] to approximate a probabilistic
distribution over model weights. While traditional stochastic gradient descent (SGD)
converges to a single point estimate and thus fails to capture weight uncertainty, SWA
improves generalization by averaging weights over multiple training epochs. However,
SWA only provides a single mean estimate and does not reflect the diversity of the
weight distribution.
To address this limitation, SWAG estimates a Gaussian distribution centered at
the SWA weights θSWA, combining both a diagonal covariance matrix
Σdiag = diag( ¯θ2 −θ2
SWA)
(28)
and a low-rank covariance matrix Σlow-rank constructed from the deviations of the
most recent K weight samples. The resulting approximate posterior is given by
N
 
θSWA, 1
2Σdiag + Σlow-rank
 
.
(29)
29


<!-- page 30 -->
Weights sampled from this distribution can be used to perform approximate Bayesian
inference. Given S sampled models, the predictive mean and variance are computed as
µ∗(x) = 1
S
S
X
s=1
µθs(x),
σ2
∗(x) = 1
S
S
X
s=1
 σ2
θs(x) + µ2
θs(x)
 
−µ2
∗(x),
(30)
where µθs(x) and σ2
θs(x) denote the predictive mean and variance of the s-th model.
Through this process, SWAG not only retains the generalization benefits of SWA
but also enables Bayesian uncertainty estimation by exploring the weight space more
comprehensively.
4.5.4 Improved Variational Online Newton
IVON is a second-order optimization-based variational Bayesian learning algorithm
that enables simultaneous uncertainty quantification and regularization based on
Bayesian inference while maintaining a computational cost comparable to that of
Adaptive Moment Estimation (Adam). Traditional variational inference (VI) aims
to approximate the posterior distribution q(θ) over the model parameters θ by
minimizing the following objective:
L(q) = λ Eq(θ)
 ¯ℓ(θ)
 
+ KL (q(θ) ∥p(θ)) ,
(31)
where ¯ℓ(θ) denotes the expected loss, KL is the Kullback–Leibler divergence, and λ is
a scaling hyperparameter. However, in high-dimensional neural networks, the standard
VI approach becomes inefficient due to the noise in expectation estimation and the
difficulty of accurately estimating the Hessian.
To address these limitations, IVON introduces a second-order optimization scheme
that includes an Adam-like update strategy. The algorithm models the posterior
distribution as a diagonal Gaussian:
q(θ) = N(m, diag(σ)2),
(32)
where m and σ represent the mean and standard deviation of the parameter
distribution, respectively.
Given a sample θ ∼q(θ), the loss gradient ˆg and curvature estimate ˆh are
computed as:
ˆg = ∇θ¯ℓ(θ),
(33)
ˆh = ˆg · (θ −m)
σ2
.
(34)
To ensure numerical stability and positivity of the curvature estimate, IVON
adopts a Riemannian gradient-based update rule:
g ←β1g + (1 −β1)¯g,
(35)
30


<!-- page 31 -->
h ←(1 −ρ)h + ρˆh + 1
2ρ2 (h −ˆh)2
h + δ
,
(36)
¯g ←g/(1 −βt
1).
(37)
The mean and variance of the posterior are updated as:
m ←m −α ¯g + δm
h + δ ,
(38)
σ ←
1
p
λ(h + δ)
.
(39)
The updated parameters m and σ define the approximate posterior distribution
enabling efficient approximate Bayesian inference during both training and prediction.
Given S samples θs ∼q(θ), predictive mean and variance are computed in the
same way as in SWAG (Eq. 30).
4.5.5 Laplace Approximation
LA is an efficient Bayesian inference method that estimates predictive uncertainty
by approximating the posterior distribution with a Gaussian. In this work, we apply
LA in a post-hoc manner, where the pretrained parameters obtained from standard
training are used directly as θMAP, thus avoiding the need for retraining or variational
optimization.
We approximate the posterior as
p(θ | D) ≈N(θ; θMAP, Σ),
Σ−1 = H = ∇2
θL(D; θ)
  
θMAP
(40)
where L(D; θ) is the training loss (e.g., negative log-likelihood). Since computing the
full Hessian is infeasible for large neural networks, we adopt the generalized Gauss-
Newton (GGN) approximation to the Hessian:
G =
N
X
n=1
Jn(∇2
f log p(yn | f)
  
f=fθMAP(xn))J⊤
n
(41)
where Jn = ∇θfθ(xn) is the Jacobian of the network outputs with respect to
the parameters. This approximation preserves positive semi-definiteness and is more
scalable to deep networks.
For prediction, we draw S samples θs ∼N(θMAP, Σ) and compute the predictive
mean and variance as defined in Eq. (30).
We use a diagonal approximation of the GGN matrix over all network weights,
implemented with the CurvlinopsGGN backend. The prior precision and observation
noise are optimized jointly by maximizing the log marginal likelihood. Under the
regression likelihood considered here, the GGN matrix coincides with the Fisher infor-
mation matrix. As a result, once the MAP estimate is fixed, the posterior becomes
deterministic and does not depend on stochastic sampling.
31


<!-- page 32 -->
Data availability
The
QM9
dataset
is
available
at
https://www.kaggle.com/datasets/zaharch/
quantum-machine-9-aka-qm9. The rMD17 dataset is available at https://figshare.
com/articles/dataset/Revised MD17 dataset rMD17 /12672038. The PSB3 dataset
is available at https://github.com/myung-group/Data phaseless namd. The 3BPA
dataset is available at https://pubs.acs.org/doi/10.1021/acs.jctc.1c00647. The oBN25
dataset is available at https://github.com/myung-group/Data BAM RACE and via
Zenodo at https://doi.org/10.5281/zenodo.17404713 [88]. .
Code availability
All code necessary to run the public portion of the experiment is available
via GitHub at https://github.com/myung-group/BAM-jax and https://github.com/
myung-group/BAM-torch. The code is licensed under the GNU Lesser General Public
License v3.0.
Declarations
Funding
This research was supported by the National Research Foundation of Korea (NRF)
funded by the Korean government (Ministry of Science and ICT (MSIT)) (RS-2022-
NR072058, NRF-2023M3K5A1094813, RS-2023-00257666, RS-2024-00455131) and by
the Institute for Basic Science (IBS-R036-D1). SYW and CWM acknowledge the sup-
port from the Brain Pool program funded by the Ministry of Science and ICT through
the National Research Foundation of Korea (No. RS-2024-00407680). JL acknowl-
edges the support from the Institute of Information & Communications Technology
Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No.
RS-2019-II190075, Artificial Intelligence Graduate School Program (KAIST)).
Acknowledgements
We are grateful for the computational support from the Korea Institute of Science and
Technology Information (KISTI) for the Nurion cluster (KSC-2022-CRE-0082, KSC-
2022-CRE-0113, KSC-2022-CRE-0408, KSC-2022-CRE-0424, KSC-2022-CRE-0429,
KSC-2022-CRE-0469, KSC-2023-CRE-0050, KSC-2023-CRE-0059, KSC-2023-CRE-
0251, KSC-2023-CRE-0261, KSC-2023-CRE-0311, KSC-2023-CRE-0332, KSC-2023-
CRE-0355, KSC-2023-CRE-0454, KSC-2023-CRE-0501, KSC-2023-CRE-0502, KSC-
2024-CRE-0117, KSC-2024-CRE-0144, KSC-2024-CRE-0330, KSC-2024-CRE-0358,
KSC-2025-CRE-0161, KSC-2025-CRE-0286, KSC-2025-CRE-0316, KSC-2025-CHA-
0020) and the Neuron cluster (KSC-2023-CRE-0472, KSC-2025-CRE-0093, KSC-2025-
CRE-0122, KSC-2025-CRE-0164, KSC-2025-CRE-0341). Computational work for this
research was partially performed on the Olaf supercomputer supported by the IBS
Research Solution Center and on the GPU cluster supported by the Ministry of Science
and ICT (MSIT) and the National IT Industry Promotion Agency (NIPA).
32


<!-- page 33 -->
Author information
These authors contributed equally: Soohaeng Yoo Willow, Tae Hyeon Park.
Contributions
S.Y.W. and T.H.P. conceived and designed the research, developed the machine-
learning framework, performed the model implementation and training, and analyzed
the results. G.B.S. contributed to code development and figure preparation. S.W.M.
and S.K.M. provided datasets and contributed to result visualization. S.J.S., J.W.K.,
D.C.Y. and H.W.K. contributed to manuscript preparation and participated in dis-
cussions. J.L. and C.W.M. supervised the research and provided theoretical guidance
throughout the study.
Corresponding authors
Correspondence to Juho Lee or Chang Woo Myung.
Ethics declarations
Competing interests
The authors declare no competing interests.
Additional information
Supplementary information
See the supplementary material for a detailed compilation of the obtained results as
well as further data and analysis to support the points made throughout the text.
References
[1] Batatia, I., Kov´acs, D. P., Simm, G. N. C., Ortner, C. & Cs´anyi, G. MACE: Higher
Order Equivariant Message Passing Neural Networks for Fast and Accurate Force
Fields (2023). Preprint at https://arxiv.org/abs/2206.07697v2.
[2] Batatia, I. et al. A foundation model for atomistic materials chemistry. J. Chem.
Phys. 163, 184110 (2025).
[3] Willow, S. Y. et al.
Active sparse Bayesian committee machine potential for
isothermal–isobaric molecular dynamics simulations. Phys. Chem. Chem. Phys.
26, 22073–22082 (2024).
[4] Coley, C. W. et al. A robotic platform for flow synthesis of organic compounds
informed by AI planning. Science 365, eaax1566 (2019).
33


<!-- page 34 -->
[5] Wen, M. et al.
Chemical reaction networks and opportunities for machine
learning. Nat. Comput. Sci. 3, 12–24 (2023).
[6] Moor, M. et al. Foundation models for generalist medical artificial intelligence.
Nature 616, 259–265 (2023).
[7] R¨ockert, A., Kullgren, J. & Hermansson, K. Predicting frequency from the exter-
nal chemical environment: OH vibrations on hydrated and hydroxylated surfaces.
J. Chem. Theory Comput. 18, 7683–7694 (2022).
[8] Mishin, Y. Machine-learning interatomic potentials for materials science. Acta
Mat. 214, 116980 (2021).
[9] Varivoda, D., Dong, R., Omee, S. S. & Hu, J. Materials property prediction with
uncertainty quantification: A benchmark study.
Appl. Phys. Rev. 10, 021409
(2023).
[10] Li, L. et al. Uncertainty quantification in multivariable regression for material
property prediction with Bayesian neural networks. Sci. Rep. 14, 10543 (2024).
[11] Gawlikowski, J. et al. A survey of uncertainty in deep neural networks. Artif.
Intell. Rev. 56, 1513–1589 (2023).
[12] Kuleshov, V., Fenner, N. & Ermon, S. Accurate Uncertainties for Deep Learn-
ing Using Calibrated Regression (2018). Preprint at https://arxiv.org/abs/1807.
00263.
[13] Maddox, W., Garipov, T., Izmailov, P., Vetrov, D. & Wilson, A. G. A Simple
Baseline for Bayesian Uncertainty in Deep Learning (2019). Preprint at https:
//arxiv.org/abs/1902.02476v2.
[14] Nagler, T. & R¨ugamer, D. Uncertainty quantification for prior-data fitted net-
works using martingale posteriors (2025). Preprint at https://arxiv.org/abs/2505.
11325v2.
[15] Van De Schoot, R. et al. Bayesian statistics and modelling. Nat. Rev. Methods
Primers 1, 1 (2021).
[16] Wu, J., Poloczek, M., Wilson, A. G. & Frazier, P. I. Bayesian Optimization with
Gradients (2018). Preprint at https://arxiv.org/abs/1703.04389v3.
[17] Goan, E. & Fookes, C. Bayesian neural networks: An introduction and survey
(2020). Preprint at https://arxiv.org/abs/2006.12024.
[18] Wilson, A. G. & Izmailov, P.
Bayesian Deep Learning and a Probabilistic
Perspective of Generalization (2022).
Preprint at https://arxiv.org/abs/2002.
08791.
34


<!-- page 35 -->
[19] Gal, Y., Koumoutsakos, P., Lanusse, F., Louppe, G. & Papadimitriou, C.
Bayesian uncertainty quantification for machine-learned models in physics. Nat.
Rev. Phys. 4, 573–577 (2022).
[20] Seligmann, F., Becker, P., Volpp, M. & Neumann, G. Beyond deep ensembles: A
large-scale evaluation of bayesian deep learning under distribution shift. Advances
in Neural Information Processing Systems 36, 29372–29405 (2023).
[21] Kim, J., Nguyen, T. D., Suleymanzade, A., An, H. & Hong, S. Learning Proba-
bilistic Symmetrization for Architecture Agnostic Equivariance (2024). Preprint
at https://arxiv.org/abs/2306.02866v3.
[22] Fortuin, V. Priors in bayesian deep learning: A review. Int. Stat. Rev. 90, 563–591
(2022).
[23] MacKay, D. J. C. A practical bayesian framework for backpropagation networks.
Neural Comput. 4, 448–472 (1992).
[24] Neal, R. M. Bayesian Learning for Neural Networks (Springer Science&Business
Media, 1996).
[25] Neal, R. M. MCMC Using Hamiltonian Dynamics (Chapman and Hall/CRC,
2011).
[26] Blundell, C., Cornebise, J., Kavukcuoglu, K. & Wierstra, D. Weight uncertainty
in neural network. Proceedings of the 32nd International Conference on Machine
Learning 37, 1613–1622 (2015).
[27] Gal, Y. & Ghahramani, Z. Dropout as a bayesian approximation: Represent-
ing model uncertainty in deep learning. Proceedings of The 33rd International
Conference on Machine Learning 48, 1050–1059 (2016).
[28] Lakshminarayanan, B., Pritzel, A. & Blundell, C. Simple and Scalable Predictive
Uncertainty Estimation using Deep Ensembles (2017). Preprint at https://arxiv.
org/abs/1612.01474v3.
[29] Shen, Y. et al. Variational Learning is Effective for Large Deep Networks (2024).
Preprint at https://arxiv.org/abs/2402.17641v2.
[30] Daxberger, E. et al. Laplace redux – effortless bayesian deep learning (2022).
Preprint at https://arxiv.org/abs/2106.14806v3.
[31] Nix, D. A. & Weigend, A. S. Estimating the mean and variance of the target
probability distribution. Proceedings of 1994 IEEE International Conference on
Neural Networks 1, 55–60 (1994).
[32] Behler, J. Atom-centered symmetry functions for constructing high-dimensional
neural network potentials. J. Chem. Phys. 134, 074106 (2011).
35


<!-- page 36 -->
[33] Drautz, R. Atomic cluster expansion for accurate and transferable interatomic
potentials. Phys. Rev. B 99, 014104 (2019).
[34] Dusson, G. et al.
Atomic Cluster Expansion: Completeness, Efficiency and
Stability (2021). Preprint at https://arxiv.org/abs/1911.03550v4.
[35] Behler, J. & Parrinello, M. Generalized Neural-Network Representation of High-
Dimensional Potential-Energy Surfaces. Phys. Rev. Lett. 98, 146401 (2007).
[36] Bart´ok, A. P., Payne, M. C., Kondor, R. & Cs´anyi, G. Gaussian Approximation
Potentials: The Accuracy of Quantum Mechanics, without the Electrons. Phys.
Rev. Lett. 104, 136403 (2010).
[37] Sch¨utt, K. T., Sauceda, H. E., Kindermans, P.-J., Tkatchenko, A. & M¨uller, K.-
R. SchNet – A deep learning architecture for molecules and materials. J. Chem.
Phys. 148, 241722 (2018).
[38] Yao, K., Herr, J. E., Toth, D., Mckintyre, R. & Parkhill, J. The TensorMol-0.1
model chemistry: a neural network augmented with long-range physics. Chem.
Sci. 9, 2261–2269 (2018).
[39] Unke, O. T. & Meuwly, M. PhysNet: A neural network for predicting energies,
forces, dipole moments, and partial charges. J. Chem. Theory Comput. 15, 3678–
3693 (2019).
[40] Zhang, L. et al. End-to-end symmetry preserving inter-atomic potential energy
model for finite and extended systems (2018). Preprint at https://arxiv.org/abs/
1805.09003v2.
[41] Geiger, M. & Smidt, T. E3nn: Euclidean Neural Networks (2022). Preprint at
https://arxiv.org/abs/2207.09453.
[42] Liao, Y.-L. & Smidt, T. Equiformer: Equivariant Graph Attention Transformer
for 3D Atomistic Graphs (2023). Preprint at https://arxiv.org/abs/2206.11990v2.
[43] Liao, Y.-L., Wood, B., Das, A. & Smidt, T. EquiformerV2: Improved Equivariant
Transformer for Scaling to Higher-Degree Representations (2024). Preprint at
https://arxiv.org/abs/2306.12059v3.
[44] Wood, B. M. et al.
UMA: A Family of Universal Models for Atoms (2026).
Preprint at https://arxiv.org/abs/2506.23971v2.
[45] Thomas, N. et al. Tensor field networks: Rotation- and translation-equivariant
neural networks for 3D point clouds (2018). Preprint at https://arxiv.org/abs/
1802.08219v3.
[46] Anderson, B., Hy, T.-S. & Kondor, R. Cormorant: Covariant molecular neural
networks (2019). Preprint at https://arxiv.org/abs/1906.04015v3.
36


<!-- page 37 -->
[47] Batzner, S. et al. E(3)-equivariant graph neural networks for data-efficient and
accurate interatomic potentials. Nat. Commun. 13, 2453 (2022).
[48] Merchant, A. et al. Scaling deep learning for materials discovery. Nature 624,
80–85 (2023).
[49] Park, Y., Kim, J., Hwang, S. & Han, S. Scalable parallel algorithm for graph neu-
ral network interatomic potentials in molecular dynamics simulations. J. Chem.
Theory Comput. 20, 4857–4868 (2024).
[50] Musaelian, A. et al.
Learning local equivariant representations for large-scale
atomistic dynamics. Nat. Commun. 14, 579 (2023).
[51] Aykent, S. & Xia, T.
GotenNet: Rethinking Efficient 3D Equivariant Graph
Neural Networks. Proceedings of the 13th International Conference on Learning
Representations 40622–40646 (2025).
[52] Chang, J. & Zhu, S.
MGNN: Moment graph neural network for universal
molecular potentials. npj Comput. Mater. 11, 55 (2025).
[53] Musil, F., Willatt, M. J., Langovoy, M. A. & Ceriotti, M. Fast and accurate
uncertainty estimation in chemical machine learning. J. Chem. Theory Comput.
15, 906–915 (2019).
[54] Zhu, A., Batzner, S., Musaelian, A. & Kozinsky, B. Fast uncertainty estimates
in deep learning interatomic potentials. J. Chem. Phys. 158, 164111 (2023).
[55] Bigi, F., Chong, S., Ceriotti, M. & Grasselli, F. A prediction rigidity formalism
for low-cost uncertainties in trained neural networks. Mach. Learn.: Sci. Technol.
5, 045018 (2024).
[56] Zaverkin, V. et al. Uncertainty-biased molecular dynamics for learning uniformly
accurate interatomic potentials. npj Comput. Mater. 10, 83 (2024).
[57] Kellner, M. & Ceriotti, M. Uncertainty quantification by direct propagation of
shallow ensembles. Mach. Learn.: Sci. Technol. 5, 035006 (2024).
[58] Swinburne, T. D. & Perez, D. Parameter uncertainties for imperfect surrogate
models in the low-noise regime. Mach. Learn.: Sci. Technol. 6, 015008 (2025).
[59] Perez, D., Subramanyam, A. P. A., Maliyov, I. & Swinburne, T. D.
Uncer-
tainty quantification for misspecified machine learned interatomic potentials. npj
Comput. Mater. 11, 263 (2025).
[60] Xu, H. et al. Evidential deep learning for interatomic potentials. Nat. Commun.
17, 937 (2026).
37


<!-- page 38 -->
[61] Kahle, L. & Zipoli, F.
Quality of uncertainty estimates from neural network
potential ensembles. Phys. Rev. E 105, 015311 (2022).
[62] Wen, M. & Tadmor, E. B. Uncertainty quantification in molecular simulations
with dropout neural network potentials. npj Comput. Mater. 6, 124 (2020).
[63] Deng, B. et al. Overcoming systematic softening in universal machine learning
interatomic potentials by fine-tuning (2024). Preprint at https://arxiv.org/abs/
2405.07105.
[64] Kaur, H. et al. Data-efficient fine-tuning of foundational models for first-principles
quality sublimation enthalpies (2024). Preprint at https://arxiv.org/abs/2405.
20217.
[65] Ruddigkeit, L., Van Deursen, R., Blum, L. C. & Reymond, J.-L. Enumeration of
166 billion organic small molecules in the chemical universe database GDB-17. J.
Chem. Inf. Model. 52, 2864–2875 (2012).
[66] Ramakrishnan, R., Dral, P. O., Rupp, M. & von Lilienfeld, O. A.
Quantum
chemistry structures and properties of 134 kilo molecules. Sci. Data 1, 140022
(2014).
[67] Christensen, A. S. & Von Lilienfeld, O. A. On the role of gradients for machine
learning of molecular energies and forces. Mach. Learn.: Sci. Technol. 1, 045018
(2020).
[68] Moon, S. W., Willow, S. Y., Park, T. H., Min, S. K. & Myung, C. W. Machine
Learning Nonadiabatic Dynamics: Eliminating Phase Freedom of Nonadiabatic
Couplings with the State-Interaction State-Averaged Spin-Restricted Ensemble-
Referenced Kohn–Sham Approach.
J. Chem. Theory Comput. 21, 1521–1529
(2025).
[69] Kov´acs, D. P. et al.
Linear atomic cluster expansion force fields for organic
molecules: Beyond RMSE. J. Chem. Theory Comput. 17, 7696–7711 (2021).
[70] Gasteiger, J., Groß, J. & G¨unnemann, S.
Directional Message Passing for
Molecular Graphs (2022). Preprint at https://arxiv.org/abs/2003.03123v2.
[71] Satorras, V. G., Hoogeboom, E. & Welling, M. E(n) equivariant graph neural
networks (2022). Preprint at https://arxiv.org/abs/2102.09844v3.
[72] Sch¨utt, K. T., Unke, O. T. & Gastegger, M. Equivariant message passing for
the prediction of tensorial properties and molecular spectra (2021). Preprint at
https://arxiv.org/abs/2102.03150v4.
[73] Th¨olke, P. & Fabritiis, G. D. TorchMD-NET: Equivariant transformers for neural
network based molecular potentials (2022). Preprint at https://arxiv.org/abs/
2202.02541v2.
38


<!-- page 39 -->
[74] Liu, Y. et al. Spherical message passing for 3d graph networks (2022). Preprint
at https://arxiv.org/abs/2102.05013v5.
[75] Brandstetter, J., Hesselink, R., Pol, E. v. d., Bekkers, E. J. & Welling, M. Geo-
metric and physical quantities improve e(3) equivariant message passing (2022).
Preprint at https://arxiv.org/abs/2110.02905v3.
[76] Le, T., No´e, F. & Clevert, D.-A.
Equivariant graph attention networks for
molecular property prediction (2022). Preprint at https://arxiv.org/abs/2202.
09891v2.
[77] Wang, Y. et al.
Enhancing geometric representations for molecules with
equivariant vector-scalar interactive message passing.
Nat. Commun. 15, 313
(2024).
[78] Ha, J.-K., Lee, I. S. & Min, S. K. Surface hopping dynamics beyond nonadiabatic
couplings for quantum coherence. J. Phys. Chem. Lett. 9, 1097–1104 (2018).
[79] Rohrdanz, M. A., Martins, K. M. & Herbert, J. M.
A long-range-corrected
density functional that performs well for both ground-state properties and time-
dependent density functional theory excitation energies, including charge-transfer
excited states. J. Chem. Phys. 130, 054112 (2009).
[80] Krishnan, R., Binkley, J. S., Seeger, R. & Pople, J. A. Self-consistent molecular
orbital methods. XX. a basis set for correlated wave functions. J. Chem. Phys.
72, 650–654 (1980).
[81] Batatia, I. et al. The design space of e(3)-equivariant atom-centred interatomic
potentials. Nat. Mach. Intell. 7, 56–67 (2025).
[82] Faber, F. A., Christensen, A. S., Huang, B. & Lilienfeld, O. A. v. Alchemical and
structural distribution based representation for improved QML. J. Chem. Phys.
148, 241717 (2018).
[83] Gao, X., Ramezanghorbani, F., Isayev, O., Smith, J. S. & Roitberg, A. E. Tor-
chANI: A free and open source PyTorch-based deep learning implementation of
the ANI neural network potentials. J. Chem. Inf. Model. 60, 3408–3415 (2020).
[84] Ceriotti, M., Tribello, G. A. & Parrinello, M. Demonstrating the transferability
and the descriptive power of sketch-map. J. Chem. Theory Comput. 9, 1521–1532
(2013).
[85] Imbalzano, G. et al. Automatic selection of atomic fingerprints and reference
configurations for machine-learning potentials.
J. Chem. Phys. 148, 241730
(2018).
39


<!-- page 40 -->
[86] He, K., Zhang, X., Ren, S. & Sun, J. Deep residual learning for image recognition.
2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
770–778 (2016).
[87] Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D. & Wilson, A. G. Averaging
weights leads to wider optima and better generalization (2019). Preprint at https:
//arxiv.org/abs/1803.05407v3.
[88] Park, T. H., Willow, S. Y. & Myung, C. W. Supporting data and code for the
published paper : Bayesian e(3)-equivariant interatomic potential with iterative
restratification of many-body message passing. (2025). https://doi.org/10.5281/
zenodo.17404713.
40


<!-- page 41 -->
Supplementary Information: Bayesian E(3)-
Equivariant Interatomic Potential with Iterative
Renormalization of Many-body Message Passing
Soohaeng Yoo Willow1†, Tae Hyeon Park1,2†, Gi Beom Sim1,2,
Sung Wook Moon3, Seung Kyu Min2,3, Sangjae Seo4,
Jaewook Kim4, D. ChangMo Yang1, Hyun Woo Kim5,
Juho Lee6*, Chang Woo Myung1,2,7*
1Department of Energy Science, Sungkyunkwan University, Seobu-ro
2066, Suwon, 16419, Republic of Korea.
2Center for 2D Quantum Heterostructures, Institute for Basic Science
(IBS), Suwon, 16419, Republic of Korea.
3Department of Chemistry, School of Natural Science, Ulsan National
Institute of Science and Technology (UNIST), 50 UNIST-gil, Ulju-gun,
Ulsan, 44919, Republic of Korea.
4Department of Supercomputing Acceleration Research, Korea Institute
of Science and Technology Information, Daejeon, 34141, Republic of
Korea.
5Department of Chemistry, Gwangju Institute of Science and
Technology, Gwangju, 61005, Republic of Korea.
6Kim Jaechul Graduate School of AI, KAIST, Daejeon, Republic of
Korea.
7Department of Energy, Sungkyunkwan University, Seobu-ro 2066,
Suwon, 16419, Republic of Korea.
*Corresponding author(s). E-mail(s): juholee@kaist.ac.kr;
cwmyung@skku.edu;
†These authors contributed equally to this work.
S1


<!-- page 42 -->
Contents
A Details of RACE architecture
S4
B Details of experiments on QM9 dataset
S5
B.1
Training Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
S5
C Details of experiments on PSB3 dataset
S6
C.1 Training Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
S6
C.2 Excited State Molecular dynamics result . . . . . . . . . . . . . . . . .
S6
D Details of experiments on rMD17 dataset
S7
D.1 Training Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
S7
D.2 Variance analysis of predictive accuracy
. . . . . . . . . . . . . . . . .
S8
E Details of experiments on 3BPA dataset
S9
E.1
Training Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
S9
E.2
Variance analysis of predictive accuracy
. . . . . . . . . . . . . . . . .
S10
F Details of experiments on Boron nitride (oBN25) dataset
S11
F.1
Training Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
S11
F.2
Point prediction accuracy benchmark . . . . . . . . . . . . . . . . . . .
S12
F.3
NLL-E result . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
S13
F.4
Benchmark and Final Evaluation Procedure . . . . . . . . . . . . . . .
S14
F.4.1
Benchmark Phase
. . . . . . . . . . . . . . . . . . . . . . . . .
S14
F.4.2
Final Evaluation Phase
. . . . . . . . . . . . . . . . . . . . . .
S16
F.5
Size-extensivity of energy uncertainty . . . . . . . . . . . . . . . . . . .
S16
F.6
Calibration result . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
S18
F.7
Recalibration result . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
S18
G Proof: Force uncertainty quantification
S22
G.1 Derivatives of Gaussian Processes . . . . . . . . . . . . . . . . . . . . .
S22
G.2 For mean-variance parameterizations . . . . . . . . . . . . . . . . . . .
S22
G.3 Gaussian Process for Force Uncertainty Quantification . . . . . . . . .
S23
G.4 Multivariate Gaussian Formulation . . . . . . . . . . . . . . . . . . . .
S23
G.5 Weighted Gaussian Negative Log-Likelihoods for Energy and Force . .
S24
G.6 Rotational properties of predicted force covariance
. . . . . . . . . . .
S24
G.7 Negative Log-Likelihood for Stress Uncertainty Quantification . . . . .
S25
H Proof: Bayesian Active Learning by Disagreement for Deep Ensem-
ble Regression with Heteroscedastic Uncertainty
S26
H.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
S26
H.2 Problem Setup
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
S26
H.3 Information-Theoretic Objective
. . . . . . . . . . . . . . . . . . . . .
S26
H.4 Derivation of BALD for Deep Ensembles . . . . . . . . . . . . . . . . .
S26
H.4.1
Predictive Distribution . . . . . . . . . . . . . . . . . . . . . . .
S26
S2


<!-- page 43 -->
H.4.2
Moment Matching Approximation
. . . . . . . . . . . . . . . .
S27
H.4.3
Entropy Calculations . . . . . . . . . . . . . . . . . . . . . . . .
S28
H.4.4
Alternative Formulation . . . . . . . . . . . . . . . . . . . . . .
S29
H.5 Computational Considerations . . . . . . . . . . . . . . . . . . . . . . .
S29
S3


<!-- page 44 -->
A Details of RACE architecture
{𝑍𝑖, 𝑖= 1, 2, … , 𝑁𝑛𝑜𝑑𝑒𝑠}
(Nnodes, 1)
1-Body Embedding
(Nnodes, Dfeature)
Node features
(Nnodes, Dfeature)
Linear
(Nnodes, Dfeature)
Interaction
Layer
{𝑋𝑖, 𝑖= 1, 2, … , 𝑁𝑛𝑜𝑑𝑒𝑠}
(Nnodes, 3)
|𝑅𝑖𝑗| = ||𝑋𝑖−𝑋𝑗||
(Nedges, 1)
𝑅𝑖𝑗= 𝑋𝑗−𝑋𝑖
(Nedges, 3)
Radial Embedding
(Nedges, Nradial_basis)
Angular Embedding
(Nedges, Dspherical_harmonics)
Edge attributes
(Nedges, Dspherical_harmonics)
Edge features
(Nedges, Nradial_basis)
Output 1
(Nnodes, 1)
Output η
(Nnodes, Dout)
Node features
(Nnodes, Dhidden)
Activation
(Nnodes, 1)
Aggregation
(Nnodes, 1)
Energy
(Nbatch, 1)
X Node features
(Nnodes, Dfeatues)
x η
Auto gradient
(Nnodes, 3)
Forces
(Nnodes, 3)
NX: Number of the X
nA: Number of the representation A
DA: Dimension of the representation A
A = ۩𝑙,𝑝𝑚𝑙,𝑝(𝑙, 𝑝)
nA = σ𝑝=±1 σ𝑙𝑚𝑙,𝑝
DA = σ𝑝=±1 σ𝑙𝑚𝑙,𝑝2𝑙+ 1
Multi layer perceptron
(Nedges, Nmlp_hidden)
…
Node features
(Nnodes, Dfeature)
or (Nnodes, Dhidden)
Linear (skip)
(Nnodes, Dhidden)
Linear (up)
(Nnodes, Dfeature)
or (Nnodes, Dhidden)
Edge attributes
(Nedges, Dspherical_harmonics)
Edge features
(Nedges, Nradial_basis)
Tensor product
(Nedges, Dcg)
Concatenate
(Nedges, Dcg+ Dfeature)
or (Nedges, Dcg+ Dfeature)
Multiplication
Linear (down)
(Nnodes, Dfeature)
or (Nnodes, Dhidden)
Tensor product
(Nnodes, Dcg)
or (Nnodes, Dcg)
Linear
(Nnodes, Dhidden)
X Node features
(Nnodes, Dfeature)
Node features
(Nnodes, Dhidden)
Linear
(Nnodes, Dredout)
Linear
(Nnodes, Dout)
Gate
(Nnodes, Dredout)
Output
(Nnodes, Dout)
Messages
(Nedges, Dfeature)
or (Nedges, Dhidden)
Node features
(Nnodes, Dhidden)
Get Senders
(Nedges, Dfeature)
or (Nedges, Dhidden)
(Nedges, ncg + nfeature)
or (Nedges, ncg + nfeature)
(Nedges, Nmlp_hidden)
(Nedges, Dcg+ Dfeature)
or (Nedges, Dcg+ Dfeature)
Sum
(Nedges, Dhidden)
Aggregation
(Nnodes, Dcg+ Dfeature)
or (Nnodes, Dcg+ Dfeature)
Fig. S1: Algorithm flowchart of the RACE architecture, annotated with
tensor dimensions at each step. The blue boxes represent the input or output fea-
tures, with their corresponding shapes indicated. The yellow and purple boxes denote
computational operations, where the annotated shapes indicate the output dimensions
of the respective operations. The orange boxes represent the final desired output. X
denotes the positional information of atoms in the system (x, y, z coordinates), Z rep-
resents the atomic species, and R corresponds to the interatomic distance vectors. For
the angular embedding, we employed spherical harmonics via the irreducible represen-
tation of SO(3) with ml,p = 1 for all irreducible representations, where l denotes the
degree of the representation (l = 0, 1, · · · ), p the parity of the representation (p = ±1),
and ml,p the multiplicity of representation. For the 1-body embedding, we used the
irreducible representation of SO(3) with only l = 0 and arbitrary ml,p. In other cases,
the representation is not restricted to SO(3). Dcg represents the output dimension
of an equivariant tensor product, which corresponds directly to the decomposition of
irreducible representations via the Clebsch-Gordan coefficients. The order l of the irre-
ducible representations in the output of the equivariant tensor product is limited to
the range up to the orders of the two inputs.
S4


<!-- page 45 -->
B Details of experiments on QM9 dataset
B.1 Training Details
Table S1: Hyper-parameters for QM9 dataset
Hyper-parameters
Value or description
Optimize
AMSgrad
Learning rate scheduling
ReduceLROnPlateau
decay factor
0.9
patience
50
Maximum learning rate
0.01
Batch size
64
Number of epochs
2000
Energy coefficient λE
1
Force coefficient λF
1
Model EMA decay
0.99
Cutoff radius (˚A)
5.0
Maximum number of neighbors
28
Number of radial bases
8
number of layers
5
Hidden irreps
64x0o+64x0e+64x1o+64x1e+64x2o+64x2e
Feature dimension
128
Maximum degree Lmax
2
S5


<!-- page 46 -->
C Details of experiments on PSB3 dataset
C.1 Training Details
Table S2: Hyper-parameters for PSB3 dataset
Hyper-parameters
Value or description
Optimize
AMSgrad
Learning rate scheduling
ReduceLROnPlateau
decay factor
0.9
patience
50
Maximum learning rate
0.01
Batch size
5
Number of epochs
2000
Energy coefficient λE
1
Force coefficient λF
50
Model EMA decay
0.99
Cutoff radius (˚A)
6.0
Maximum number of neighbors
13
Number of radial bases
8
number of layers
5
Hidden irreps
64x0o+64x0e+64x1o+64x1e+64x2o+64x2e
Feature dimension
64
Maximum degree Lmax
2
C.2 Excited State Molecular dynamics result
a 
b
Fig. S2: Analysis of the dynamics of photoisomerization of PSB3: (a) Dihedral angle
of the central C=C bond of PSB3 over time in individual trajectories. Black trajecto-
ries represent ML-based results, while red trajectories represent reference SSR-based
dynamics. (b) Average electronic population with the corresponding time. The blue
and red line represents ML and reference SSR population evolution, respectively. The
solid line represents the BO population and the dashed line represents the averaged
running state.
S6


<!-- page 47 -->
D Details of experiments on rMD17 dataset
D.1 Training Details
Table S3: Hyper-parameters for rMD17 dataset
Hyper-parameters
Value or description
RACE
NLLE
NLLJEF
Optimize
AMSgrad
AMSgrad
AMSgrad
Learning rate scheduling
ReduceLROnPlateau
decay factor
0.9
0.9
0.9
patience
50
50
50
Maximum learning rate
0.01
0.01
0.01
Batch size
5
5
5
Number of epochs
10000
10000
10000
Energy coefficient λE
1
1
1
Force coefficient λF
1000
1000
1000
Model EMA decay
0.99
0.99
0.99
Cutoff radius (˚A)
5.0
5.0
5.0
Maximum number of neighbors
Aspirin:21, ethanol:9,
malonaldehyde:9, naphthalene:18,
salicylic:16, toluene:16,
uracil:16
Number of radial bases
8
8
8
number of layers
5
5
5
Hidden irreps
64x0o+64x0e+64x1o+64x1e
+64x2o+64x2e+64x3o+64x3e
Feature dimension
128
128
128
Maximum degree Lmax
3
3
3
S7


<!-- page 48 -->
D.2 Variance analysis of predictive accuracy
To assess whether the accuracy difference between MSE-trained and NLLJEF-
trained models is statistically significant, we report the mean and standard deviation of
MAE across independently trained models in Table S4. Using ±1σ confidence intervals,
the ranges of RACE (MSE) and RACE-DE-JEF overlap in 4 out of 14 molecule–property
pairs (marked with †). For the remaining 10 non-overlapping pairs, the average gap
between the ±1σ intervals is 0.44 meV for energy and 0.77 meV/˚A for forces. In
contrast, RACE-DE-E exhibits errors 2–5× larger with high variance, confirming that
the accuracy–uncertainty trade-off is predominantly attributable to energy-only NLL
training.
Table S4: Mean and standard deviation of MAE on the rMD17
test dataset across independently trained models. Energy (E,
meV) and force (F, meV/˚A). Values where the ±1σ intervals
of RACE and RACE-DE-JEF overlap are marked with †.
Molecule
RACE
RACE-DE-E
RACE-DE-JEF
Aspirin
E
3.30 ± 0.11
7.80 ± 0.70
3.70 ± 0.57†
F
9.60 ± 0.30
21.60 ± 1.90
9.50 ± 0.51†
Ethanol
E
0.80 ± 0.10
2.40 ± 0.32
1.40 ± 0.41
F
3.60 ± 0.11
10.60 ± 0.94
4.40 ± 0.50
Malonaldehyde E
1.40 ± 0.10
3.90 ± 0.33
2.20 ± 0.54
F
7.20 ± 0.43
16.90 ± 0.89
8.70 ± 1.58†
Naphthalene
E
0.60 ± 0.27
4.00 ± 0.45
1.80 ± 0.54
F
2.60 ± 0.07
12.60 ± 0.92
3.50 ± 0.17
Salicylic acid
E
1.10 ± 0.10
4.20 ± 0.62
2.40 ± 0.12
F
5.40 ± 0.17
16.90 ± 2.31
7.60 ± 0.13
Toluene
E
0.60 ± 0.04
3.30 ± 0.38
1.50 ± 0.39
F
2.70 ± 0.07
12.90 ± 1.08
3.40 ± 0.32
Uracil
E
0.70 ± 0.10
2.90 ± 0.31
1.60 ± 0.38
F
4.40 ± 0.13
13.80 ± 1.19
4.70 ± 0.32†
S8


<!-- page 49 -->
E Details of experiments on 3BPA dataset
E.1 Training Details
Table S5: Hyper-parameters for 3BPA dataset
Hyper-parameters
Value or description
RACE
NLLE
NLLJEF
Optimize
AMSgrad
AMSgrad
AMSgrad
Learning rate scheduling
ReduceLROnPlateau
decay factor
0.9
0.9
0.9
patience
50
50
50
Maximum learning rate
0.01
0.01
0.01
Batch size
2
2
2
Number of epochs
5000
5000
5000
Energy coefficient λE
1
1
1
Force coefficient λF
100
100
100
Model EMA decay
0.99
0.99
0.99
Cutoff radius (˚A)
6.0
6.0
6.0
Maximum number of neighbors
26
26
26
Number of radial bases
8
8
8
number of layers
5
5
5
Hidden irreps
64x0o+64x0e+64x1o+64x1e
+64x2o+64x2e+64x3o+64x3e
Feature dimension
128
128
128
Maximum degree Lmax
2
2
2
S9


<!-- page 50 -->
E.2 Variance analysis of predictive accuracy
As in the rMD17 analysis (Section D.2), we report the mean and standard devia-
tion of RMSE across independently trained models on the 3BPA dataset in Table S6.
Using ±1σ confidence intervals, the ranges of RACE (MSE) and RACE-DE-JEF overlap
in 2 out of 6 temperature–property pairs (marked with †). For the remaining 4 non-
overlapping pairs, the average gap between the ±1σ intervals is 3.74 meV for energy
and 1.16 meV/˚A for forces. Combined with the rMD17 results, these findings con-
firm that the accuracy–uncertainty trade-off introduced by NLLJEF is modest across
evaluation conditions.
Table S6: Mean and standard deviation of RMSE on the 3BPA
test dataset across independently trained models. Energy (E,
meV) and force (F, meV/˚A). Values where the ±1σ intervals
of RACE and RACE-DE-JEF overlap are marked with †.
Temperature
RACE
RACE-DE-E
RACE-DE-JEF
300 K
E
3.4 ± 0.71
17.5 ± 0.64
5.0 ± 1.26†
F
12.1 ± 0.95
52.7 ± 1.47
14.8 ± 0.97
600 K
E
11.7 ± 0.67
43.7 ± 2.65
14.6 ± 1.61
F
31.8 ± 1.54
98.0 ± 3.81
37.0 ± 2.13
1200 K
E
37.5 ± 2.38
171.9 ± 17.49
51.1 ± 4.36
F
115.3 ± 8.38
232.8 ± 19.04
120.8 ± 7.15†
S10


<!-- page 51 -->
F Details of experiments on Boron nitride (oBN25)
dataset
F.1 Training Details
Table S7: Hyper-parameters for oBN25 dataset
Hyper-parameters
Value or description
NLLE
Optimize
AMSgrad
Learning rate scheduling
ReduceLROnPlateau
decay factor
0.9
patience
50
Maximum learning rate
0.01
Batch size
2
Number of epochs
1000
Energy coefficient λE
1
Force coefficient λF
1
Model EMA decay
0.99
Cutoff radius (˚A)
3.0
Maximum number of neighbors
19
Number of radial bases
8
number of layers
5
Hidden irreps
32x0o+32x0e+16x1o
+16x1e+8x2o+8x2e
Feature dimension
128
Maximum degree Lmax
2
S11


<!-- page 52 -->
F.2 Point prediction accuracy benchmark
To assess the predictive accuracy of the RACE architecture on condensed-phase
systems, we trained NequIP and RACE using identical data splits and training con-
figurations within the BAM framework. MACE was trained on the same dataset using
its official codebase with hyperparameters matched as closely as possible to the BAM
setting. Results are shown in Table S8.
On liquid-phase (ID) test data, all three models achieve comparable accuracy,
with MACE showing the lowest energy RMSE. On solid-phase (OOD) data, RACE
demonstrates the strongest generalization with the lowest energy and force RMSE.
These results confirm that RACE is competitive with established equivariant models
on condensed-phase systems, particularly in OOD robustness.
Table
S8:
Point
prediction
accuracy
(RMSE) on the oBN25 test dataset. Energy
(E, eV) and force (F, eV/˚A) errors for
liquid-phase (ID) and solid-phase (OOD)
test sets. NequIP and RACE report mean ±
standard deviation across five independent
training runs. MACE was trained using the
official codebase with a single run. Bold
indicates the best result.
Liquid (ID)
Solid (OOD)
Model
E
F
E
F
MACE
0.147
0.484
11.245
0.550
NequIP
0.332
0.491
1.862
0.370
RACE
0.410
0.530
1.567
0.370
S12


**[Table p52.1]**
| Model | Liquid (ID) E F | Solid (OOD) E F |
| --- | --- | --- |
| MACE NequIP RACE | 0.147 0.484 0.332 0.491 0.410 0.530 | 11.245 0.550 1.862 0.370 1.567 0.370 |


<!-- page 53 -->
F.3 NLLE result
Table S9: Evaluation results on the BN dataset using dif-
ferent UQ methods. Values are reported for RMSE of energy
and force, CE of energy and AUROC. Bold indicates the best
result.
Boron nitride
ERMSE
FRMSE
ECE
AUROC
MVE
0.21
0.82
0.02
0.50
DE
0.14
0.68
0.02
1.00
SWAG
0.26
0.90
0.05
0.95
a
c
b
d
Fig. S3: Predicted uncertainty of energy (σE) for the BN dataset using a RACE-MVE,
b RACE-DE, c RACE-SWAG, and d RACE-IVON. Blue dots correspond to liquid BN (ID)
and orange dots to solid BN (OOD).
S13


<!-- page 54 -->
F.4 Benchmark and Final Evaluation Procedure
To quantitatively compare the performance of Bayesian MLP models, we developed a
unified benchmark score based on five evaluation metrics: energy RMSE, force RMSE,
energy calibration error (CE), force CE, and AUROC for OOD detection. Since these
metrics vary in scale and importance, we applied normalization and weighting to com-
bine them into a unified score. However, it is critical to note that this unified metric
should not be interpreted as an objective measure of absolute model performance or
used to compare models in different studies. The choice of weights inevitably introduces
subjectivity, as different weighting schemes can arbitrarily favor models that excel
in specific metrics over others. Consequently, the unified score is inherently relative
rather than absolute. In this work, we use the unified score exclusively for hyperpa-
rameter optimization during the validation phase, where it serves as a practical tool to
balance the multiple objectives of Bayesian MLPs. For transparent model evaluation
and comparison, we report all individual metrics separately in our results to assess
performance according to their specific priorities and application requirements.
F.4.1 Benchmark Phase
The metrics were normalized as follows. First, RMSE values for both energy and
force were converted from eV to kcal/mol by multiplying with a factor of 23.06. This
conversion was applied because differences in RMSE values are small in eV units,
whereas RMSE is a highly important metric that needed to be emphasized. Second,
calibration errors (CE) were normalized using min–max scaling:
x′ =
x −min(x)
max(x) −min(x).
Finally, AUROC values were inverted according to
x′ = 1 −AUROC,
so that lower values correspond to better performance.
The final composite score was computed as
Score = ωERMSE,norm · ERMSE,norm + ωFRMSE,norm · FRMSE,norm
+ ωECE,norm · ECE,norm + ωFCE,norm · FCE,norm + ωAUROC,norm · AUROC,norm,
(F.1)
where ωERMSE,norm = 0.25, ωFRMSE,norm = 0.25, ωECE,norm = 0.125, ωFCE,norm = 0.125,
and ωAUROC,norm = 0.25. This weighting scheme ensures a balanced evaluation by
assigning 50% to accuracy capability, with equal contributions from energy and force
prediction, and 50% to uncertainty quantification, with equal contributions from cali-
bration error and OOD detection. This holistic scoring framework enables systematic
comparison across different Bayesian MLPs by balancing their various capabilities
during the validation phase. For each model, we selected the optimal hyperparameter
configuration by minimizing this unified metric.
S14


<!-- page 55 -->
To reduce computational cost, we conducted the benchmark phase using models
trained up to 500 epochs. For each hyperparameter setting, we trained 5 independent
models and used their average performance for comparison.
MVE and Deep Ensemble. For MVE, we benchmarked models across five ini-
tial learning rates: 0.001, 0.01, 0.05, 0.1, and 0.5. Since Deep Ensembles rely on the
performance of their constituent MVE models, we first identified the best-performing
hyperparameters for MVE. Among them, learning rates of 0.05 and 0.1 yielded the
best results. The benchmark results are summarized in Table S10.
Table S10: Average performance of MVE for each learning rate across five indepen-
dent runs.
Learning Rate
ERMSE
FRMSE
ECE
FCE
AUROC
Score
0.001
1.16
0.71
0.01
1.43 × 10−4
0.50
10.93
0.01
0.48
0.63
0.01
5.78 × 10−5
0.69
6.51
0.05
0.28
0.62
0.01
4.60 × 10−5
0.50
5.30
0.1
0.29
0.61
0.01
5.64 × 10−5
0.60
5.29
0.5
0.47
0.59
0.01
3.92 × 10−5
1.00
6.16
SWAG. For SWAG, we evaluated a grid of hyperparameters by varying the learn-
ing rate (0.05 or 0.1), the weight collection starting point (after 50% or 60% of
training), and the rank size (20 or 40). The best performance was observed with the
configuration using a learning rate of 0.1, weights collected after 50% of training, and
a rank size of 40. The benchmark results are summarized in Table S11.
Table S11: Average performance of SWAG under various hyperparameter configura-
tions (5 runs per setting).
LR
Start
Epoch
Rank
ERMSE
FRMSE
ECE
FCE
AUROC
Score
0.05
60%
20
0.33
0.62
0.04
4.09 × 10−4
0.83
5.50
0.1
60%
20
0.58
0.63
0.10
1.07 × 10−2
0.96
6.99
0.05
50%
20
0.34
0.61
0.11
1.11 × 10−3
0.72
5.63
0.1
50%
20
0.39
0.60
0.002
9.22 × 10−5
0.91
5.77
0.05
50%
40
0.31
0.62
0.01
1.04 × 10−4
0.91
5.36
0.1
50%
40
0.29
0.60
0.001
0.97 × 10−5
0.97
5.14
IVON. IVON is highly sensitive to hyperparameter settings, and it often diverges
or fails to converge under overly aggressive training conditions. Therefore, among
various tested configurations, we report only the results from experiments in which
training progressed stably. The benchmark results are summarized in Table S12.
S15


**[Table p55.1]**
| Learning Rate | E F E F AUROC RMSE RMSE CE CE | Score |
| --- | --- | --- |
| 0.001 0.01 0.05 0.1 0.5 | 1.16 0.71 0.01 1.43 × 10−4 0.50 0.48 0.63 0.01 5.78 × 10−5 0.69 0.28 0.62 0.01 4.60 × 10−5 0.50 0.29 0.61 0.01 5.64 × 10−5 0.60 0.47 0.59 0.01 3.92 × 10−5 1.00 | 10.93 6.51 5.30 5.29 6.16 |


**[Table p55.2]**
| Start LR Rank Epoch | ERMSE FRMSE ECE FCE AUROC | Score |
| --- | --- | --- |
| 0.05 60% 20 0.1 60% 20 0.05 50% 20 0.1 50% 20 0.05 50% 40 0.1 50% 40 | 0.33 0.62 0.04 4.09 × 10−4 0.83 0.58 0.63 0.10 1.07 × 10−2 0.96 0.34 0.61 0.11 1.11 × 10−3 0.72 0.39 0.60 0.002 9.22 × 10−5 0.91 0.31 0.62 0.01 1.04 × 10−4 0.91 0.29 0.60 0.001 0.97 × 10−5 0.97 | 5.50 6.99 5.63 5.77 5.36 5.14 |


<!-- page 56 -->
Table S12: Benchmark results of IVON with different hyperparameter settings.
Only the differing hyperparameters are shown; all other settings are identical across
models.
LR
β2
ERMSE
FRMSE
ECE
FCE
AUROC
Score
0.01
0.999995
1.26
0.72
0.01
1.05 × 10−3
1.00
11.40
0.05
0.999995
0.97
0.62
0.02
2.14 × 10−3
1.00
9.16
0.05
0.999990
1.14
0.67
0.02
1.99 × 10−3
0.99
10.2
Table S13: Composite scores of all final models. MVE and SWAG each include 10
runs for LR = 0.05 and 0.1. Deep and IVON are reported individually.
Model (LR)
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
MVE (0.05)
4.79
4.78
4.82
4.83
4.94
4.78
4.85
4.79
4.81
4.80
MVE (0.1)
4.87
4.91
5.01
4.88
4.90
5.01
4.90
4.94
4.92
5.05
SWAG (0.05)
4.83
4.81
5.46
4.83
5.22
4.94
5.03
4.90
5.07
4.84
SWAG (0.1)
4.92
5.34
5.20
5.08
5.13
5.04
4.99
5.91
5.14
5.21
DE (0.05)
3.88
DE (0.1)
4.04
IVON (0.05)
9.16
F.4.2 Final Evaluation Phase
For the final results reported in the main text, we fully trained all models up to
1000 epochs using the best configurations identified in the benchmark phase. MVE
was trained with 10 models each for learning rates 0.05 and 0.1, and the single best-
performing model was selected based on the composite score. Deep Ensembles were
constructed from the 10 MVE models for each learning rate, thereby forming two
ensembles. For SWAG, we trained 10 models at each learning rate (0.05 and 0.1),
saving weights starting from 60% of the full training (1000 epochs), and selected the
single best-performing model among them. Finally, IVON was trained up to 1000
epochs using the stable and effective configuration identified during the benchmark
phase. The final performance results are summarized in Table S13.
F.5 Size-extensivity of energy uncertainty
An important requirement for uncertainty estimates in condensed-phase systems
is that they scale appropriately with system size [1]. In RACE, the total energy is
a sum of atom-wise contributions E = P
i ei, and the energy variance under the
independence assumption becomes Var(E) = P
i Var(ei), yielding σE ∝
√
N for a
homogeneous system of N atoms.
S16


**[Table p56.1]**
| Model (LR) | 1 2 3 4 5 6 7 8 9 10 |
| --- | --- |
| MVE (0.05) MVE (0.1) | 4.79 4.78 4.82 4.83 4.94 4.78 4.85 4.79 4.81 4.80 4.87 4.91 5.01 4.88 4.90 5.01 4.90 4.94 4.92 5.05 |
| SWAG (0.05) SWAG (0.1) | 4.83 4.81 5.46 4.83 5.22 4.94 5.03 4.90 5.07 4.84 4.92 5.34 5.20 5.08 5.13 5.04 4.99 5.91 5.14 5.21 |
| DE (0.05) DE (0.1) | 3.88 4.04 |
| IVON (0.05) | 9.16 |


<!-- page 57 -->
To verify this scaling empirically, we constructed supercells of the oBN25 dataset
by replicating the 64-atom unit cell to obtain systems of 128, 256, and 512 atoms.
For each system size, we computed the predicted energy uncertainty (σpred) and the
absolute prediction error (|∆y|) using the trained RACE-DE-JEF model. Figure S4 shows
a log-log scatter plot of |∆y| versus σpred for all system sizes. The systematic shift of
each cluster toward larger σpred with increasing N, together with the parallel linear
fits across system sizes, confirms that the energy uncertainty scales as σE ∝
√
N,
satisfying the size-extensivity requirement.
Fig. S4: Log-log scatter plot of absolute prediction error |∆y| versus predicted energy
uncertainty σpred for oBN25 supercells of varying size (64, 128, 256, and 512 atoms).
The systematic rightward shift with increasing system size and the parallel linear fits
confirm σE ∝
√
N scaling. The dashed black line indicates the ideal |∆y| = σpred
reference.
S17


<!-- page 58 -->
F.6 Calibration result
Table S14: CE on the oBN25 dataset. Results are shown separately for
energy and force predictions in the liquid and solid phases. Bold and
underline indicate the best and second-best results, respectively.
Liquid
Solid
Model
Energy
Force
Energy
Force
RACE-MVE
1.4 × 10−2
8.4 × 10−6
3.3 × 10−1
9.3 × 10−4
RACE-DE
3.4 × 10−2
8.4 × 10−3
2.1 × 10−1
5.1 × 10−2
RACE-SWAG
1.1 × 10−2
4.9 × 10−5
3.3 × 10−1
1.8 × 10−4
RACE-IVON
1.9 × 10−2
3.2 × 10−3
3.3 × 10−1
3.4 × 10−3
F.7 Recalibration result
a
c
b
d
Fig. S5: Calibration plots for liquid-phase energy predictions of the oBN25 dataset,
shown before (blue) and after (orange) post-hoc recalibration. Panels: a RACE-MVE, b
RACE-DE, c RACE-SWAG, d RACE-IVON.
S18


**[Table p58.1]**
|  | Liquid | Solid |
| --- | --- | --- |
| Model Energy Force Energy Force RACE-MVE 1.4 × 10−2 8.4 × 10−6 3.3 × 10−1 9.3 × 10−4 RACE-DE 3.4 × 10−2 8.4 × 10−3 2.1 × 10−1 5.1 × 10−2 RACE-SWAG 1.1 × 10−2 4.9 × 10−5 3.3 × 10−1 1.8 × 10−4 RACE-IVON 1.9 × 10−2 3.2 × 10−3 3.3 × 10−1 3.4 × 10−3 F.7 Recalibration result a b c d Fig. S5: Calibration plots for liquid-phase energy predictions of the oBN25 dataset, shown before (blue) and after (orange) post-hoc recalibration. Panels: a RACE-MVE, b RACE-DE, c RACE-SWAG, d RACE-IVON. S18 | Energy Force |  |
|  | 1.4 × 10−2 8.4 × 10−6 3.4 × 10−2 8.4 × 10−3 1.1 × 10−2 4.9 × 10−5 1.9 × 10−2 3.2 × 10−3 |  |


<!-- page 59 -->
a
c
b
d
Fig. S6: Calibration plots for solid-phase energy predictions of the oBN25 dataset,
shown before and after post-hoc recalibration. Panels: a RACE-MVE, b RACE-DE, c
RACE-SWAG, d RACE-IVON.
Table S15: Relative calibration-error reduction after post-
hoc recalibration on the oBN25 dataset. Entries report
(Before CE −After CE)/Before CE of energy and force in
the liquid and solid phases. Higher is better (1 = perfect
correction, 0 = no change; negative values indicate degrada-
tion). Bold and underline indicate the best and second-best
results, respectively.
Liquid
Solid
Model
Energy
Force
Energy
Force
MVE
0.9917
0.0000
-6.6690
0.9255
RACE-DE
0.9983
0.9984
0.9954
0.9990
RACE-SWAG
0.9843
0.0000
0.3161
0.1959
RACE-IVON
0.9966
0.9996
0.9295
0.9732
S19


<!-- page 60 -->
a
c
b
d
Fig. S7: Calibration plots for liquid-phase force predictions of the oBN25 dataset,
shown before (blue) and after (orange) post-hoc recalibration. Panels: a RACE-MVE, b
RACE-DE, c RACE-SWAG, d RACE-IVON.
S20