<!-- page 1 -->
RewardUQ: A Unified Framework for
Uncertainty-Aware Reward Models
Daniel Yang1∗
Samuel Stante1∗
Florian Redhardt1∗
Lena Libon1∗
Parnian Kassraie1
Ido Hakimi1,2
Barna Pásztor1,2
Andreas Krause1,2
1 ETH Zurich
2 ETH AI Center
Abstract
Reward models are central to aligning large language models (LLMs) with hu-
man preferences. Yet most approaches rely on pointwise reward estimates that
overlook the epistemic uncertainty in reward models arising from limited human
feedback. Recent work suggests that quantifying this uncertainty can reduce the
costs of human annotation via uncertainty-guided active learning and mitigate
reward overoptimization in LLM post-training. However, uncertainty-aware re-
ward models have so far been adopted without thorough comparison, leaving them
poorly understood. This work introduces a unified framework, RewardUQ, to
systematically evaluate uncertainty quantification for reward models. We compare
common methods along standard metrics measuring accuracy and calibration, and
we propose a new ranking strategy incorporating both dimensions for a simplified
comparison. Our experimental results suggest that model size and initialization
have the most meaningful impact on performance, and most prior work could
have benefited from alternative design choices. To foster the development and
evaluation of new methods and aid the deployment in downstream applications, we
release our open-source framework as a Python package. Our code is available at
https://github.com/lasgroup/rewarduq.
1
Introduction
Reinforcement learning from human feedback (RLHF) is a key component for aligning large
language models (LLMs) with human preferences to ensure they are safe and helpful (Ziegler et al.,
2019; Ouyang et al., 2022; Bai et al., 2022). The standard RLHF process first trains a reward model
on a dataset of pairwise comparisons to learn the underlying preferences, and then uses this model
to align the LLM policy with reinforcement learning (RL) algorithms (Christiano et al., 2017).
However, the success of RLHF heavily relies on the quality of the reward model. This poses several
challenges, as collecting high-quality human preference data is expensive and reward models trained
on limited and noisy datasets are imperfect (Casper et al., 2023). Additionally, aligning an LLM
with such an imperfect reward model can lead to reward hacking, where the LLM overoptimizes
flawed rewards rather than intended human preferences (Eisenstein et al., 2024; Coste et al., 2024;
Gao et al., 2023; Amodei et al., 2016).
Uncertainty quantification (UQ) for reward models emerged as a promising way to address these
issues by explicitly modeling epistemic uncertainty arising from limited preference data. Recent
work leverages uncertainty-aware reward models to mitigate reward hacking by penalizing (Lou et al.,
2025; Zhai et al., 2023; Yan et al., 2024; Houliston et al., 2024; Banerjee & Gopalan, 2024; Coste
et al., 2024) or filtering (Sun et al., 2025b; Lou et al., 2025) uncertain samples. Reward uncertainty
∗Equal contributions. Correspondence to {dayang,sstante,fredhardt,llibon}@ethz.ch.
Preprint.
arXiv:2602.24040v1  [cs.LG]  27 Feb 2026


<!-- page 2 -->
estimates are also leveraged to improve sample efficiency and reduce data collection costs through
active learning in the reward modeling (Melo et al., 2024; Dwaracherla et al., 2024; Das et al., 2025)
or alignment step (Mehta et al., 2025; Muldrew et al., 2024; Liu et al., 2024b; Gleave & Irving, 2022;
Liang et al., 2022; Christiano et al., 2017). However, most studies adopt a single UQ method without
systematic evaluation, leaving the impact of specific design choices largely unexplored.
In this work, we present RewardUQ, a unified framework for the design and evaluation of uncertainty-
aware reward models. This framework represents a first step towards principled UQ in preference
modeling with reward functions, with the goal to build a foundation for more reliable and sample-
efficient RLHF. Our main contributions are as follows:
• We introduce a unified framework which formalizes the UQ problem, standardizes exist-
ing methods in a consistent notation, and defines a common evaluation procedure. Our
evaluations utilize a new ranking strategy that incorporates the accuracy and calibration of
predictions under uncertainty.
• We conduct a systematic evaluation of existing uncertainty-aware reward models, analyzing
how architectural choices and training parameters affect the quality of uncertainty estimates.
• We release an open-source Python package as an accessible and extensible platform for
developing, evaluating, and deploying new UQ methods.
2
Related work
2.1
Methods for reward model uncertainty quantification
Ensembles
The predominant approach for uncertainty quantification for reward models in RLHF
are ensembles, with the uncertainty represented by the variance across ensemble members. In its
simplest form, an ensemble combines multiple reward models trained with different random seeds
(Coste et al., 2024; Eisenstein et al., 2024; Liang et al., 2022; Christiano et al., 2017) and, optionally,
with bootstrapped datasets (Lou et al., 2025; Gleave & Irving, 2022). To reduce computational
cost, others utilize a pretrained model and train only lightweight ensemble members such as linear
heads (Banerjee & Gopalan, 2024; Yan et al., 2024), multi-layer perceptron (MLP) heads (Melo
et al., 2024; Liu et al., 2024b; Dwaracherla et al., 2024), low-rank adaptation (LoRA) adapters (Zhai
et al., 2023; Sun et al., 2025b; Houliston et al., 2024; Yang et al., 2024; Zhang et al., 2024a), or apply
Monte Carlo (MC) dropout (Mehta et al., 2025; Zhang et al., 2025).
Bayesian inference with Laplace approximation
An alternative approach assumes a Gaussian
prior on the parameters of a single reward model and derives the uncertainty from the predictive
posterior based on the Laplace approximation (Cercola et al., 2025). As the Hessian is often
intractable over all model parameters, the Laplace approximation is typically applied to a subset
of the parameters of a pretrained LLM, such as the linear head (Das et al., 2025; Cercola et al., 2025)
or a LoRA adapter (Yang et al., 2024).
Mean-variance estimation
Some studies utilize reward models which predict the mean and
variance of a Gaussian reward distribution, capturing the aleatoric uncertainty under heteroscedastic
noise (Yan et al., 2024; Lou et al., 2025; Siththaranjan et al., 2024; Sun et al., 2025a).
Reward-margin-based preference uncertainty
Others leverage the margin between pointwise
rewards as a measure of uncertainty about the true preference, without modeling the uncertainty
about the true reward (Muldrew et al., 2024; Lou et al., 2025).
Our work focuses on the most common approaches and covers a selection of ensemble and Bayesian
inference methods, identifying commonalities and differences, and evaluating them side by side.
2.2
Applications for reward UQ
Uncertainty-aware alignment
Uncertainty estimates can make the alignment step in RLHF more
resilient to reward overoptimization by encouraging the LLM to avoid uncertain rewards. Common
schemes involve penalizing (Lou et al., 2025; Zhai et al., 2023; Houliston et al., 2024; Banerjee
& Gopalan, 2024; Coste et al., 2024; Sun et al., 2025a) or filtering (Sun et al., 2025b; Lou et al.,
2


<!-- page 3 -->
2025) uncertain rewards. Other approaches adopt pessimistic objectives to optimize for worst-case
performance under uncertainty (Zhang et al., 2024b; Yan et al., 2024) or apply pessimistic best-of-N
sampling (Liu et al., 2025b).
Active learning for reward modeling
To reduce the cost of collecting high-quality preference data,
uncertainty estimates can guide the label acquisition towards more informative samples, improving
the sample efficiency in the reward modeling step in RLHF (Melo et al., 2024; Dwaracherla et al.,
2024; Das et al., 2025). Others utilize uncertainty to estimate the quality of and adaptively assign
weights to preference samples (Zhang et al., 2025).
Active learning for alignment
Similarly, uncertainty in the predicted rewards can improve the
sample efficiency in the alignment step in RLHF, be it through uncertainty-based selection criteria of
alignment samples (Mehta et al., 2025; Muldrew et al., 2024; Christiano et al., 2017; Cercola et al.,
2025) or exploration bonuses (Liu et al., 2024b; Liang et al., 2022).
These directions highlight the promise of UQ methods for reward models. Yet, most studies adopt a
single method and focus on downstream applications. Even work that compare multiple methods,
such as the ensemble architecture study of Zhang et al. (2024a), limit their analysis to downstream
performance rather than a systematic analysis of the uncertainty quantification itself. In contrast, our
work follows a complementary direction by focusing on the design and evaluation of different UQ
methods, aiming to provide a clear comparison and offer insights on how to choose and use methods.
With most prior work initializing their reward models from generic pre-trained models, our results
suggest that most works could have benefited from better design choices, especially by choosing
model initializations that are tuned for reward modeling.
3
Uncertainty quantification for reward models
We introduce a unified framework for designing and evaluating uncertainty-aware reward models,
which integrates a range of existing approaches and extends them with novel contributions of our
own. We begin by formalizing the UQ problem for reward models in Section 3.1, and then introduce
our evaluation metrics in Sections 3.2 and 3.3.
3.1
Problem statement
We consider the reinforcement learning from human feedback (RLHF) problem, which aims to align
a language model π with human preferences, such that π is more likely to generate a human-preferred
completion y ∼π(· | x) for a given prompt x (Ouyang et al., 2022; Stiennon et al., 2020). We assume
preferences to be expressed as pairwise comparisons in terms of (x, y+, y−) with y+ being preferred
over y−, denoted y+ ≻y−. As standard in the literature, we assume the Bradley-Terry preference
model (Bradley & Terry, 1952) that models the comparison between two candidate completions y
and y′ as a Bernoulli distribution with probability
p(y ≻y′ | x, y, y′) = σ(r(x, y) −r(x, y′))
(1)
where σ(x) =
1
1+exp(−x) is the sigmoid function and r is a reward function assigning a scalar score
to any prompt-completion pair. Given a dataset Dtrain = {(xi, y+
i , y−
i )}n
i=1, a reward model rθ is
trained by maximizing the likelihood of the observed preferences or equivalently by minimizing the
binary cross-entropy loss
Lbase(θ; Dtrain) = 1
n
X
(x,y+,y−)∈Dtrain
−log σ(rθ(x, y+) −rθ(x, y−)).
(2)
Once trained, rθ can be used to align π via RL algorithms such as PPO (Schulman et al., 2017) or
GRPO (Shao et al., 2024), or at inference time with best-of-N sampling (Stiennon et al., 2020; Yang
et al., 2024). However, the standard RLHF framework relies on reward models which only make
pointwise predictions, thereby neglecting the epistemic uncertainty arising from training on a finite
dataset sampled from the large domain of natural language.
An uncertainty-aware reward model additionally predicts upper and lower confidence bounds rθ(x, y)
and rθ(x, y), quantifying its epistemic uncertainty about the true underlying reward in terms of a
3


<!-- page 4 -->
confidence interval Irθ(x, y) =
 
rθ(x, y), rθ(x, y)
 
. We introduce the most common methods in
detail in Section 4. Under the Bradley-Terry model assumption, the corresponding upper and lower
bounds on the preference probability are given by
pθ(y ≻y′ | x, y, y′) = σ
 rθ(x, y) −rθ(x, y′)
 
pθ(y ≻y′ | x, y, y′) = σ
 rθ(x, y) −rθ(x, y′)
 
,
(3)
which are based on the largest and smallest plausible reward margin between both candidate comple-
tions, respectively (Mehta et al., 2025).
The goal of an uncertainty-aware reward model is to predict preference probabilities and confidence
bounds, which not only accurately reflect the true binary preferences but are also statistically well-
calibrated with respect to the true preference probabilities. We introduce our precise notion of
accuracy and calibration, two complementary evaluation dimensions, along with their corresponding
metrics in Sections 3.2 and 3.3. We elaborate on the epistemic and aleatoric uncertainty decomposition
for preference classification in Appendix A.1, and discuss the differences between standard and
preference classification by focusing on the symmetry of Equation (1) in Appendix A.2.
3.2
Accuracy metrics
While accuracy is a standard performance measure for pointwise predictions, we further extend the
notion of accuracy to confidence bounds.
Accuracy of predictions
Given an evaluation dataset Deval and a reward model rθ, the predicted
rewards are correct if they assign higher rewards to preferred completions.2 Let
T(rue) = {(x, y+, y−) ∈Deval | rθ(x, y+) > rθ(x, y−)}
F(alse) = {(x, y+, y−) ∈Deval | rθ(x, y+) ≤rθ(x, y−)}
be the set of true (i.e., correct) and false (i.e., incorrect) preference predictions. The accuracy,
commonly known as win rate in the context of RLHF, is defined as
win rate = |T|
n .
↑3
(4)
Accuracy of bounds
While the win rate only captures the accuracy of pointwise predictions, we
extend this notion of accuracy to confidence intervals. To quantify the accuracy of the predicted
reward confidence intervals Irθ(x, y), we further categorize the true and false predictions into
C(onfident) = {(x, y+, y−) ∈Deval | Irθ(x, y+) ∩Irθ(x, y−) = ∅}
U(nconfident) = {(x, y+, y−) ∈Deval | Irθ(x, y+) ∩Irθ(x, y−) ̸= ∅}.
Intuitively, a prediction is confident when the predicted reward confidence intervals of the preferred
and non-preferred completion do not overlap, indicating no ambiguity in the predicted preference
even under uncertainty. By combining the correctness of the pointwise predictions with the confidence
of the predicted bounds, we define the following metrics
(confident)
(unconfident)
(true)
CT rate = |C ∩T|
n
↑
UT rate = |U ∩T|
n
↘
(false)
CF rate = |C ∩F|
n
↓
UF rate = |U ∩F|
n
.
↘
(5)
We refer to Appendix A.3 for a generalization of these metrics to the standard binary classification
setting.
2This is equivalent to pθ(y+ ≻y−| x, y+, y−) > 0.5 under the Bradley-Terry model in Equation (1).
3The arrow indicates the direction of improvement in the metric value.
4


<!-- page 5 -->
Ranking score
In order to compare models, we propose a ranking score that combines the accuracy
metrics above into a single score. Motivated by the UQ reward model applications, this score
encourages a high confident true rate to efficiently guide active learning algorithms and identify
reliable training data samples. Simultaneously, it penalizes the confident false rate that could provide
misleading signals. The ranking score is defined as
RSα =
CT rate
win rate + α · (1 −win rate) −
CF rate
(1 −win rate) + α · win rate
=
|C ∩T|
|T| + α · |F| −
|C ∩F|
|F| + α · |T| ∈[−1, 1]
↑
(6)
using a trade-off parameter α ∈[0, 1], which balances the focus on the confidence and the focus
on the accuracy. For α = 0, RS0 considers the relative rate of confidence among true and false
predictions by normalizing the CT and CF rates and represents performance in the range [−1, 1].
In other words, high RS0 scores are achievable without achieving a high win rate. For α = 1.0,
RS1 looks at the absolute difference between the confident true and false predictions and represents
performance in the range [win rate −1, win rate]. This approach assigns equal weight to CT and
CF and favors a high win rate over confidence. For our evaluations in Section 5, we choose RS0.2
as a balance between accuracy and confidence. A more detailed explanation of the rationale and
analysis of the inherent trade-offs are provided in Appendix A.4.
3.3
Calibration metrics
Calibration refers to the gap between a predicted probability and the true (or empirical) probability.
We give a formal description of calibration in Appendix A.5 while providing an overview here.
Calibration of predictions
The expected calibration error (ECE) is commonly used to measure
the calibration of predicted preference probabilities (Zhai et al., 2023; Gleave & Irving, 2022). It is
approximated based on grouping the predicted probabilities into M bins {Bm}M
m=1 and computing
ECE ≈
M
X
m=1
|Bm|
n
|P(Bm) −pθ(Bm)|
↓
(7)
with empirical probability P(Bm) and average predicted probability pθ(Bm) in each bin Bm (Guo
et al., 2017; Pavlovic, 2025).
Calibration of bounds
We extend the notion of calibration to the predicted preference probability
bounds constructed in Equation (3) by introducing the expected lower calibration error (ELCE) and
expected upper calibration error (EUCE). Analog to ECE, we group the lower and upper bounds
separately into M bins {Bm}M
m=1 and compute
ELCE ≈
M
X
m=1
|Bm|
n
max
 pθ(Bm) −P(Bm), 0
 
↓
EUCE ≈
M
X
m=1
|Bm|
n
max(P(Bm) −pθ(Bm), 0)
↓
with pθ(Bm) and pθ(Bm) denoting the average predicted lower and upper bounds in the correspond-
ing bin Bm. ELCE penalizes lower bounds that overestimate the true preference probability, and
EUCE penalizes upper bounds that underestimate the true preference probability.
Note that preference probabilities are antisymmetric in their completions argument. Accordingly, all
calibration metrics are computed on a symmetrized preference evaluation set that includes flipped
comparisons with opposite labels, ensuring that both directions of each preference pair contribute
to the binning-based approximation of the calibration errors, as further described in Appendix A.2.
Therefore, the lower bound on the probability of y ≻y′ corresponds at the same time to an upper
bound for the probability of y′ ≻y. Hence, ELCE and EUCE are identical in the context of preference
probabilities, and we subsequently only report the expected bound calibration error (EBCE)
EBCE = ELCE = EUCE.
↓
(8)
5


<!-- page 6 -->
x
y
LM
 
z
last
MLP head
MLP head
\
\
r1
rK
...
...
r
u
(a) MLP head ensemble
x
y
LM
 
LoRA adapter
LoRA adapter
\
\
linear head
linear head
\
\
z
last
z
last
r1
rK
...
...
...
r
u
(b) LoRA adapter ensemble
x
y
LM
 
z
last
linear head
\
w0
∼N(µ0, Σ0)
r
u
(c) Bayesian linear head
x
y
LM
\
z
all
dropout
dropout
LM head
\
rDPO
r1
rK
...
...
r
u
(d) DPO-based MC dropout
Figure 1: Uncertainty-aware reward model architectures compared in this work. For a given prompt
x and completion y, each model extracts an embedding z from a pretrained language model (LM)
and predicts a reward r and uncertainty estimate u. Blue components indicate the parts responsible
for estimating the uncertainty, while \ and   denote trainable and frozen components, respectively.
4
Uncertainty-aware reward models
In this work, we focus on the most common uncertainty-aware reward model architectures from
existing work. While these models differ in how they represent epistemic uncertainty, they share
several core principles as illustrated on Figure 1. Following prior work (Li et al., 2023; Ji et al., 2024;
Mehta et al., 2025), for any prompt-completion pair, (x, y), we separate pointwise prediction from
uncertainty quantification, and assume that each model predicts a reward rθ(x, y) and an uncertainty
estimate uθ(x, y), which are used to construct the symmetric confidence bounds
rθ(x, y) = rθ(x, y) + β · uθ(x, y)
rθ(x, y) = rθ(x, y) −β · uθ(x, y)
(9)
with scaling factor β > 0. The reward models are trained using the standard binary cross-entropy loss
defined in Equation (2), with modifications depending on the specific architecture as described below.
4.1
MLP head ensemble (ENS-MLP)
A common approach to estimate epistemic uncertainty is to train an ensemble of K independent
Multi-Layer Perceptron (MLP) heads using the embedding z provided by a pretrained LLM (Melo
et al., 2024; Liu et al., 2024b; Dwaracherla et al., 2024), as illustrated in Figure 1a. Each MLP head
is parametrized by θ(k) ∈Rd and predicts a pointwise reward rθ(k)(x, y). The pointwise reward
and uncertainty estimates in Equation (9) are computed as the mean and standard deviation over
the individual rewards
rθ(x, y) = 1
K
K
X
k=1
rθ(k)(x, y)
and
uθ(x, y) =
v
u
u
t
1
K −1
K
X
k=1
(rθ(k)(x, y) −rθ(x, y))2.
(10)
The model is trained by minimizing the loss
L(θ; Dtrain) = 1
K
K
X
k=1
 
Lbase(θ(k); Dtrain) + λ
d
θ(k) −θ(k)
init

2
2
+ γ
n
X
(x,y+,y−)∈Dtrain
(rθ(k)(x, y+) + rθ(k)(x, y−))2
 
,
(11)
which consists of the standard cross-entropy loss from Equation (2) and two regularization terms
applied on each head. The first regularization term controlled by λ encourages the parameters of
each head θ(k) to stay close to their random initialization θ(k)
init , which preserves diversity across the
heads in the ensemble. The second regularization term controlled by γ centers predicted rewards
around zero (Eisenstein et al., 2024). This is a crucial practical step often overseen in practice, as
the cross-entropy loss is invariant to additive constants in the reward function, which could otherwise
lead to poorly calibrated uncertainty estimates due to unintended large standard deviations.
6


**[Table p6.1]**
| x y | LM | MLP head\ r1 r z ... ... last u MLP head\ rK |
| --- | --- | --- |


**[Table p6.2]**
| x y | LoRA adapter\ LM ... LoRA adapter\ | LoRA adapter\ |  | z | linear head\ |  | r1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  |  | LoRA adapter\ |  |  | linear head\ |  | rK |
|  |  |  |  | last |  |  |  |


**[Table p6.3]**
| x y | \ LM | dropout z ... all dropout | dropout |  | \ LM head |  | rDPO |  | r1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  | dropout |  |  |  |  |  | rK |
|  |  |  |  |  |  |  |  |  |  |

[CAPTION] Figure 1: Uncertainty-aware reward model architectures compared in this work. For a given prompt


<!-- page 7 -->
4.2
LoRA adapter ensemble (ENS-LoRA)
ENS-LoRA (Mühlematter et al., 2025) extends the framework of the MLP head ensemble model
defined in Section 4.1 by training all layers of the model instead of additional MLP heads. To
overcome the computational constraints of training K models, Low-Rank Adaptation (LoRA)
method is used to reduce the number of trainable parameters (Wang et al., 2023). We denote each
LoRA adapter by the parameter vector θ(k) and initialize a linear head for each adapter to obtain
a pointwise reward rθ(k)(x, y) from embeddings z. The adapters are trained by minimizing the loss
defined in Equation (11) and the reward rθ(x, y) and uncertainty estimates uθ(x, y) are computed
following Equation (10).
4.3
DPO-based MC dropout (MCD-DPO)
Instead of training several heads of LoRA adapters, one can also leverage Monte-Carlo (MC) dropouts
before the final layer of a fine-tuned model and estimate rewards implicitly. Formally, let πθ be a
fine-tuned LLM initialized from a reference policy πref and trained to minimize the KL-regularized
loss (Christiano et al., 2017; Ouyang et al., 2022; Stiennon et al., 2020). This policy defines an
implicit reward model as
rθ(x, y) = λ log πθ(y | x)
πref(y | x) + λ log Z(x),
(12)
where λ controls the KL-regularization term and Z(x) is the partition function (Rafailov et al.,
2023). MCD-DPO (Mehta et al., 2023) quantifies the uncertainty of this implicit reward function by
introducing a dropout layer right before the language modeling head to enable MC dropout (Gal &
Ghahramani, 2016), as shown in Figure 1d.
During inference, K dropout masks m(k) are sampled and applied to the embedding z of a pretrained
LLM, providing an ensemble of completion probabilities πθ(y | x; m(k)). We obtain the implicit
rewards rθ(x, y; m(k)) using Equation (12), denoted by the rDPO layer in Figure 1d, and utilize the
mean and standard deviation over these individual rewards
rθ(x, y) = 1
K
K
X
k=1
rθ(x, y; m(k))
and
uθ(x, y) =
v
u
u
t
1
K −1
K
X
k=1
 rθ(x, y; m(k)) −rθ(x, y)
 2
as our reward and uncertainty estimate for Equation (9). In our implementation πθ is trained with the
DPO loss, derived by substituting Equation (12) into the standard cross-entropy loss in Equation (2),
L(θ; Dtrain) = 1
n
X
(x,y+,y−)∈Dtrain
m∼Pdropout
−log σ
 
λ log πθ(y+ | x; m)
πref(y+ | x)
−λ log πθ(y−| x; m)
πref(y−| x)
 
,
(13)
with a randomly sampled dropout mask m per sample.
4.4
Bayesian linear head (BAY-LIN)
Another common approach in the literature is to consider reward estimation as a Bayesian linear
regression problem (Das et al., 2025; Cercola et al., 2025). This method also computes the embedding
z for each prompt-completion pair (x, y) but applies a single linear reward head
rθ(x, y) = θ⊤z
with a Gaussian prior on the trainable parameters θ ∼N
 0, λ−1I
 
. The posterior on θ is then
approximated using a Laplace approximation, resulting in the following Gaussian distribution
θ | Dtrain
approx
∼
N
 
θMAP, H−1  
θ=θMAP
 
with mean centered at the posterior mode θMAP = argminθ −log p(θ|Dtrain) and the inverse co-
variance given by the Hessian of the negative log-posterior H = ∇2
θ −log p(θ|Dtrain) evaluated at
7


<!-- page 8 -->
UQ method
ENS-MLP
ENS-LoRA
MCD-DPO
BAY-LIN
UQ method
ENS-MLP
ENS-LoRA
MCD-DPO
BAY-LIN
Base model
Qwen 3 (solid)
Skywork Qwen 3 (dashed)
Figure 2: Ranking scores on RewardBench across different UQ methods, training datasets, pretrained
and finetuned models, and model sizes. The ranking score is defined in Equation (6).
θMAP. Intuitively, the Gaussian distribution is centered at and fitted to the local curvature around the
posterior mode. The posterior mode is obtained by equivalently minimizing
θMAP = argmin
θ
Lbase(θ; Dtrain) + λ
2 ∥θ∥2
2
(14)
corresponding to the cross-entropy loss in Equation (2) with ℓ2-regularization. The Hessian
H =
X
(x,y+,y−)∈Dtrain
w(x, y+, y−) · (z+ −z−)(z+ −z−)⊤+ λI,
(15)
with weights w(x, y+, y−) = σ′(rθ(x, y+) −rθ(x, y−)) corresponds to the empirical covariance of
the feature differences with larger weights for ambiguous predictions, i.e., rθ(x, y+) ≈rθ(x, y−).
However, these weights depend on the current parameter estimate θ. This dependence requires the
entire sum in the Hessian to be recomputed in the active learning setting, where θ is updated iteratively.
To avoid these high computational costs, Das et al. (2025) omit these weights, allowing the Hessian
H to be updated incrementally. We follow this unweighted approach to keep our evaluation practical.
The final reward and uncertainty estimate in Equation (9) are given by the predictive posterior mean
and standard deviation
rθ(x, y) = θ⊤z
and
uθ(x, y) =
√
z⊤H−1z.
5
Experiments
For a systematic comparison, we train and evaluate the model architectures outlined in Section 4 across
different datasets and base models with a unified evaluation procedure as described in Section 5.1.
Our main results are detailed in Section 5.2.
5.1
Experimental setup
Evaluation procedure
For each uncertainty quantification method, we first perform a parameter
optimization over common parameters (e.g., learning rate, base model family, and model size) and
model-specific ones (e.g., regularization parameters, dropout rate). This search is done on the training
and validation split of the UltraFeedback preference dataset4 (Cui et al., 2024) consisting of around
62K and 1K samples, respectively. We select the best parameters by first applying an upper threshold
on ECE and EBCE as introduced in Equations (7) and (8) with 0.05 and 0.01, respectively, to ensure
reasonable calibration, and then ranking according to RS0.25 from Equation (6). We report the final
4https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized
5We describe our choice of α = 0.2 in Appendix A.4.
8

[CAPTION] Figure 2: Ranking scores on RewardBench across different UQ methods, training datasets, pretrained


<!-- page 9 -->
predicted probability pθ(Bm)
true prob. P(Bm)
(a) Calibration of predictions
predicted upper bound pθ(Bm)
true prob. P(Bm)
(b) Calibration of (upper) bounds
Figure 3: Calibration diagrams for Qwen3-0.6B (top) and Qwen3-4B (bottom) trained on UltraFeed-
back and evaluated on RewardBench. The predictions are well-calibrated when they agree with the
actual probability per bin (i.e., on the diagonal), while the predicted upper bounds are well-calibrated
when they consistently exceed the actual probability per bin (i.e., below the diagonal). The calibration
metrics are defined in Equations (7) and (8). The color intensity of each bar is proportional to the
bin size. As described in Section 3.3, the calibration diagrams for the upper and lower bounds are
equivalent.
performance on the popular RewardBench dataset6 (Lambert et al., 2025b). Finally, we train each
model on two additional datasets to evaluate our results robustness to the dataset’s source and size:
the Skywork preference dataset7 (Liu et al., 2024a) with around 77K samples and the preference
dataset for the Tulu 3 8B model8 (Lambert et al., 2025a) with around 273K samples.
Models
We initialize our models from either the Qwen 3 family9 (Yang et al., 2025) with sizes
from 0.6B to 32B, and the Skywork-Reward-V2 Qwen 3 series10 (Liu et al., 2025a), which are
further finetuned for the reward modeling task on a large-scale preference dataset of around 26M
preference pairs and range from 0.6B to 8B. This provides us with a broad coverage over model
sizes and pre-training purposes.
Due to the larger computational requirements of ENS-LoRA and MCD-DPO, we consider models
only up to 4B. Further experimental details, including hyperparameters, are provided in Appendix B.
5.2
Results
Insights into accuracy
As illustrated in Figure 2, no single uncertainty quantification algorithm
consistently dominates according to the ranking score RS0.2; rather, performance is highly contingent
on model size, dataset, and pre-training. A critical determinant of performance is the base model
initialization. Methods that rely on a fixed LLM backbone to provide embeddings, such as BAY-LIN
and ENS-MLP, benefit significantly from initialization with a task-aligned reward model (e.g., the
Skywork family). Conversely, when initialized from a generic base like Qwen 3, these methods
underperform compared to ENS-LoRA and MCD-DPO, which fine-tune the full model parameters
and are thus less sensitive to the quality of the initial embeddings. Additionally, we observe dimin-
ishing returns in ranking scores as model size increases, a phenomenon we attribute to the higher
overconfidence of larger models, which is penalized by our metric. While BAY-LIN achieves the
highest performance across most settings, it lags behind ENS-MLP on the UltraFeedback dataset,
preventing a definitive recommendation. However, given that prior works typically utilize generic
initializations, our findings strongly suggest that adopting task-aligned base models offers a potential
for performance improvement.
Insights into calibration
We further analyze the calibration of our UQ methods, as defined in
Section 3.3, for two representative initialization models, namely, Qwen3-0.6B and Qwen3-4B. As
6We use the filtered split from https://huggingface.co/datasets/allenai/reward-bench
7https://huggingface.co/datasets/Skywork/Skywork-Reward-Preference-80K-v0.2
8https://huggingface.co/datasets/allenai/llama-3.1-tulu-3-8b-preference-mixture
9https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f
10https://huggingface.co/collections/Skywork/skywork-reward-v2-685cc86ce5d9c9e4be500c84
9

[CAPTION] Figure 3: Calibration diagrams for Qwen3-0.6B (top) and Qwen3-4B (bottom) trained on UltraFeed-


<!-- page 10 -->
shown in Figure 3, the different UQ methods provide similarly calibrated preference predictions and
confidence bounds with ECE rates below 0.1 and EBCE rates below 0.01. However, we observe
that the distribution of predicted preference probabilities concentrates around 0.5 for smaller models,
as seen from the color intensities in Figure 3a, indicating lower certainty on average. Similarly,
we observe that smaller models tend to be slightly overconfident if certain, since the predicted
preference probabilities for > 0.5 lie below the diagonal and for < 0.5 above the diagonal. We
discuss the unusual appearance of calibration diagrams in the context of preference classification in
Appendix A.2.
We provide detailed results and further discussion on both accuracy and calibration in Appendix C.
6
Discussion
Conclusions
Uncertainty quantification for reward models offers a promising direction for robust
and sample-efficient RLHF, ultimately improving the safety and helpfulness of language models. In
this work, we introduced RewardUQ, a unified framework to design and evaluate uncertainty-aware
reward models, which complements prior work focusing on downstream applications exclusively.
The unified formalization of existing approaches, a novel metric balancing accuracy and uncertainty,
and a common evaluation procedure enable a more systematic comparison of different methods. Our
results show that utilizing finetuned base models results in higher accuracy, but we find that the
best-performing method remains instance dependent. By releasing our framework as an open-source
library, we aim to lower the barrier to uncertainty-aware alignment research and foster applications in
active learning, safe alignment, and reward robustness.
Limitations
Our study focuses on the intrinsic evaluation of UQ methods rather than their per-
formance in downstream reinforcement learning loops. This scope was chosen intentionally to
disentangle the quality of uncertainty estimates from the confounding variables inherent in PPO or
DPO fine-tuning. We posit that identifying robust UQ signals in isolation provides a more reliable
foundation for researchers than costly end-to-end ablations. Furthermore, as detailed in Appendix A.4,
we acknowledge that our proposed ranking score (Equation (6)) entails specific trade-offs between
calibration and discrimination that may not perfectly capture every nuance of downstream utility.
Future Work
To bolster the generalizability of our findings, future comparisons should expand
to a broader range of algorithms, model families, datasets, and metrics. Additionally, while this
work provides extensive empirical benchmarks, the theoretical mechanisms governing epistemic
uncertainty in reward models for preference optimization remain under-explored. We hope this
work serves as a launchpad for rigorous theoretical analysis and the development of next-generation
applications in active preference learning and safety-constrained alignment.
Broader impact statement
The rigorous study of UQ for reward models offers promising improvements to various downstream
applications, such as reducing the costs of data collection via active learning, enhancing the safety of
LLMs via uncertainty-aware alignment, and making reward models more interpretable. However, we
acknowledge that highly accurate UQ methods could eventually serve as an additional attack vector
targeting the uncertainties of LLMs. While current methods do not yet pose a significant risk, we
emphasize the importance of monitoring dual-use risks in the future.
Acknowledgments
This work was supported as part of the Swiss AI initiative by a grant from the Swiss National
Supercomputing Centre (CSCS) under project IDs a10, a145, and infra01 on Alps. Barna Pásztor
was primarily supported by the ETH AI Center through an ETH AI Center doctoral fellowship, and
Ido Hakimi primarily supported by the ETH AI Center through an ETH AI Center postdoctoral
fellowship.
10


<!-- page 11 -->
References
Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman and Dan Mané.
Concrete Problems in AI Safety. arXiv preprint arXiv:1606.06565, 2016.
Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, et al.
Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback.
arXiv preprint arXiv:2204.05862, 2022.
Debangshu Banerjee and Aditya Gopalan. Towards Reliable Alignment: Uncertainty-aware RLHF.
arXiv preprint arXiv:2410.23726, 2024.
Ralph Allan Bradley and Milton E. Terry. Rank Analysis of Incomplete Block Designs: I. The
Method of Paired Comparisons. Biometrika, 39(3/4):324–345, 1952.
Stephen Casper, Xander Davies, Claudia Shi, Thomas Krendl Gilbert, Jérémy Scheurer, Javier Rando,
et al. Open Problems and Fundamental Limitations of Reinforcement Learning from Human
Feedback. Transactions on Machine Learning Research, 2023.
Matteo Cercola, Valeria Capretti and Simone Formentin. Efficient Reinforcement Learning from
Human Feedback via Bayesian Preference Inference. arXiv preprint arXiv:2511.04286, 2025.
Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg and Dario Amodei. Deep
Reinforcement Learning from Human Preferences. In Advances in Neural Information Processing
Systems 30 (NIPS), volume 30. Curran Associates, Inc., 2017.
Thomas Coste, Usman Anwar, Robert Kirk and David Krueger. Reward Model Ensembles Help
Mitigate Overoptimization. In Proceedings of the 12th International Conference on Learning
Representations (ICLR), 2024.
Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Bingxiang He, Wei Zhu, et al. ULTRA-
FEEDBACK: Boosting Language Models with Scaled AI Feedback. In Proceedings of the 41st
International Conference on Machine Learning (ICML), pp. 9722–9744. PMLR, 2024.
Nirjhar Das, Souradip Chakraborty, Aldo Pacchiano and Sayak Ray Chowdhury. Active Preference
Optimization for Sample Efficient RLHF. In Joint European Conference on Machine Learning
and Knowledge Discovery in Databases, pp. 96–112. Cham: Springer Nature Switzerland, 2025.
Stefan Depeweg, Jose-Miguel Hernandez-Lobato, Finale Doshi-Velez and Steffen Udluft. Decom-
position of Uncertainty in Bayesian Deep Learning for Efficient and Risk-sensitive Learning. In
Proceedings of the 35th International Conference on Machine Learning, pp. 1184–1193. PMLR,
2018.
Vikranth Dwaracherla, Seyed Mohammad Asghari, Botao Hao and Benjamin Van Roy. Efficient
Exploration for LLMs. In Proceedings of the 41st International Conference on Machine Learning
(ICML), pp. 12215–12227. PMLR, 2024.
Jacob Eisenstein, Chirag Nagpal, Alekh Agarwal, Ahmad Beirami, Alexander Nicholas D’Amour,
Krishnamurthy Dj Dvijotham, et al. Helping or Herding? Reward Model Ensembles Mitigate but
do not Eliminate Reward Hacking. In COLM 2024, 2024.
Yarin Gal and Zoubin Ghahramani. Dropout as a Bayesian Approximation: Representing Model
Uncertainty in Deep Learning. In Proceedings of The 33rd International Conference on Machine
Learning, pp. 1050–1059. PMLR, 2016.
Leo Gao, John Schulman and Jacob Hilton. Scaling Laws for Reward Model Overoptimization. In
Proceedings of the 40th International Conference on Machine Learning, pp. 10835–10866. PMLR,
2023.
Adam Gleave and Geoffrey Irving. Uncertainty Estimation for Language Reward Models. arXiv
preprint arXiv:2203.07472, 2022.
Sylvain Gugger, Lysandre Debut, Thomas Wolf, Philipp Schmid, Zachary Mueller, Sourab Man-
grulkar, et al. Accelerate: Training and inference at scale made simple, efficient and adaptable.
2022. URL https://github.com/huggingface/accelerate.
11


<!-- page 12 -->
Chuan Guo, Geoff Pleiss, Yu Sun and Kilian Q. Weinberger. On Calibration of Modern Neural
Networks. In Proceedings of the 34th International Conference on Machine Learning (ICML).
arXiv, 2017.
Sam Houliston, Alizée Pace, Alexander Immer and Gunnar Rätsch. Uncertainty-Penalized Direct
Preference Optimization. In NeurIPS 2024 Workshop on Fine-Tuning in Modern Machine Learning:
Principles and Scalability, 2024.
Eyke Hüllermeier and Willem Waegeman. Aleatoric and Epistemic Uncertainty in Machine Learning:
An Introduction to Concepts and Methods. Machine Learning, 110(3):457–506, 2021.
Kaixuan Ji, Jiafan He and Quanquan Gu. Reinforcement Learning from Human Feedback with
Active Queries. arXiv preprint arXiv:2402.09401, 2024.
Alex Kendall and Yarin Gal. What Uncertainties Do We Need in Bayesian Deep Learning for
Computer Vision? In Advances in Neural Information Processing Systems, volume 30. Curran
Associates, Inc., 2017.
Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman,
et al. Tulu 3: Pushing Frontiers in Open Language Model Post-Training. In Second Conference on
Language Modeling, 2025a.
Nathan Lambert, Valentina Pyatkin, Jacob Morrison, L. J. Miranda, Bill Yuchen Lin, Khyathi Chandu,
et al. RewardBench: Evaluating Reward Models for Language Modeling. In Findings of the
Association for Computational Linguistics: NAACL 2025, pp. 1755–1797, 2025b.
Xiang Li, Viraj Mehta, Johannes Kirschner, Ian Char, Willie Neiswanger, Jeff Schneider, et al.
Near-optimal Policy Identification in Active Reinforcement Learning. In Proceedings of the 11th
International Conference on Learning Representations (ICLR), 2023.
Xinran Liang, Katherine Shu, Kimin Lee and Pieter Abbeel. Reward Uncertainty for Exploration in
Preference-based Reinforcement Learning. In Proceedings of the 10th International Conference
on Learning Representations (ICLR), 2022.
Chris Yuhao Liu, Liang Zeng, Jiacai Liu, Rui Yan, Jujie He, Chaojie Wang, et al. Skywork-Reward:
Bag of Tricks for Reward Modeling in LLMs. arXiv preprint arXiv:2410.18451, 2024a.
Chris Yuhao Liu, Liang Zeng, Yuzhen Xiao, Jujie He, Jiacai Liu, Chaojie Wang, et al. Skywork-
Reward-V2:
Scaling Preference Data Curation via Human-AI Synergy.
arXiv preprint
arXiv:2507.01352, 2025a.
Pangpang Liu, Junwei Lu and Will Wei Sun. Uncertainty Quantification for Large Language Model
Reward Learning under Heterogeneous Human Feedback. arXiv preprint arXiv:2512.03208,
2025b.
Zichen Liu, Changyu Chen, Chao Du, Wee Sun Lee and Min Lin. Sample-Efficient Alignment for
LLMs. arXiv preprint arXiv:2411.01493, 2024b.
Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Confer-
ence on Learning Representations, 2019.
Xingzhou Lou, Dong Yan, Wei Shen, Yuzi Yan, Jian Xie and Junge Zhang. Uncertainty-aware Reward
Model: Teaching Reward Models to Know What is Unknown. arXiv preprint arXiv:2410.00847,
2025.
Viraj Mehta, Vikramjeet Das, Ojash Neopane, Yijia Dai, Ilija Bogunovic, Jeff Schneider, et al. Sample
Efficient Reinforcement Learning from Human Feedback via Active Exploration. arXiv preprint
arXiv:2312.00267, 2023.
Viraj Mehta, Syrine Belakaria, Vikramjeet Das, Ojash Neopane, Yijia Dai, Ilija Bogunovic, et al.
Sample Efficient Preference Alignment in LLMs via Active Exploration. In Second Conference on
Language Modeling, 2025.
12


<!-- page 13 -->
Luckeciano C. Melo, Panagiotis Tigas, Alessandro Abate and Yarin Gal. Deep Bayesian Active
Learning for Preference Modeling in Large Language Models. In Advances in Neural Information
Processing Systems 37 (NeurIPS), pp. 118052–118085, 2024.
Dominik J. Mühlematter, Michelle Halbheer, Alexander Becker, Dominik Narnhofer, Helge Aasen,
Konrad Schindler, et al. LoRA-Ensemble: Efficient Uncertainty Modelling for Self-Attention
Networks. arXiv preprint arXiv:2405.14438, 2025.
William Muldrew, Peter Hayes, Mingtian Zhang and David Barber. Active Preference Learning
for Large Language Models. In Proceedings of the 41st International Conference on Machine
Learning (ICML), pp. 36577–36590. PMLR, 2024.
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, et al.
Training Language Models to Follow Instructions with Human Feedback. In Advances in Neural
Information Processing Systems 35 (NeurIPS), pp. 27730–27744, 2022.
Maja Pavlovic. Understanding Model Calibration – A gentle introduction and visual exploration of
calibration and the expected calibration error (ECE). In The Fourth Blogpost Track at ICLR 2025,
2025.
Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon and Chelsea
Finn. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. In
Advances in Neural Information Processing Systems 36 (NeurIPS), pp. 53728–53741, 2023.
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford and Oleg Klimov. Proximal Policy
Optimization Algorithms. arXiv preprint arXiv:1707.06347, 2017.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, et al. DeepSeek-
Math: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv preprint
arXiv:2402.03300, 2024.
Judy Hanwen Shen, Archit Sharma and Jun Qin. Towards Data-Centric RLHF: Simple Metrics for
Preference Dataset Comparison. arXiv preprint arXiv:2409.09603, 2024.
Anand Siththaranjan, Cassidy Laidlaw and Dylan Hadfield-Menell. Distributional Preference Learn-
ing: Understanding and Accounting for Hidden Context in RLHF. In The Twelfth International
Conference on Learning Representations, 2024.
Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, et al. Learning
to summarize with human feedback. In Advances in Neural Information Processing Systems 33
(NeurIPS), volume 33, pp. 3008–3021. Curran Associates, Inc., 2020.
Wangtao Sun, Xiang Cheng, Xing Yu, Haotian Xu, Zhao Yang, Shizhu He, et al. Probabilistic
Uncertain Reward Model. arXiv preprint arXiv:2503.22480, 2025a.
Zexu Sun, Yiju Guo, Yankai Lin, Xu Chen, Qi Qi, Xing Tang, et al. Uncertainty and Influence aware
Reward Model Refinement for Reinforcement Learning from Human Feedback. In Proceedings of
the 13th International Conference on Learning Representations (ICLR), 2025b.
Leandro von Werra, Younes Belkada, Lewis Tunstall, Edward Beeching, Tristan Thrush, Nathan
Lambert, et al. TRL: Transformer Reinforcement Learning. 2020. URL https://github.com/
huggingface/trl.
Xi Wang, Laurence Aitchison and Maja Rudolph. LoRA Ensembles for Large Language Model
Fine-Tuning. arXiv preprint arXiv:2310.00035, 2023.
Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi,
et al. Transformers: State-of-the-Art Natural Language Processing. Association for Computational
Linguistics, 2020. URL https://aclanthology.org/2020.emnlp-demos.6/.
Yuzi Yan, Xingzhou Lou, Jialian Li, Yiping Zhang, Jian Xie, Chao Yu, et al. Reward-Robust RLHF
in LLMs. arXiv preprint arXiv:2409.15360, 2024.
13


<!-- page 14 -->
Adam X. Yang, Maxime Robeyns, Thomas Coste, Zhengyan Shi, Jun Wang, Haitham Bou-Ammar,
et al. Bayesian Reward Models for LLM Alignment. In ICML 2024 Workshop on Structured
Probabilistic Inference & Generative Modeling, 2024.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, et al. Qwen3 Technical
Report. arXiv preprint arXiv:2505.09388, 2025.
W. J. Youden. Index for Rating Diagnostic Tests. Cancer, 3(1):32–35, 1950.
Yuanzhao Zhai, Han Zhang, Yu Lei, Yue Yu, Kele Xu, Dawei Feng, et al. Uncertainty-Penalized
Reinforcement Learning from Human Feedback with Diverse Reward LoRA Ensembles. arXiv
preprint arXiv:2401.00243, 2023.
Rongzhi Zhang, Chenwei Zhang, Xinyang Zhang, Liang Qiu, Haoming Jiang, Yuchen Zhuang,
et al. DORM: Preference Data Weights Optimization for Reward Modeling in LLM Alignment.
In Findings of the Association for Computational Linguistics: EMNLP 2025, pp. 22721–22739.
Association for Computational Linguistics, 2025.
Shun Zhang, Zhenfang Chen, Sunli Chen, Yikang Shen, Zhiqing Sun and Chuang Gan. Improving
Reinforcement Learning from Human Feedback with Efficient Reward Model Ensemble. arXiv
preprint arXiv:2401.16635, 2024a.
Xiaoying Zhang, Jean-François Ton, Wei Shen, Hongning Wang and Yang Liu. Mitigating Reward
Overoptimization via Lightweight Uncertainty Estimation. In Advances in Neural Information
Processing Systems, volume 37, pp. 81717–81747, 2024b.
Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, et al.
Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019.
14


<!-- page 15 -->
Appendix
A Theoretical details
15
A.1
Background on the uncertainty decomposition in preference classification . . . . .
15
A.2
Background on the symmetry in preference classification . . . . . . . . . . . . . .
16
A.3
Background on the accuracy metrics . . . . . . . . . . . . . . . . . . . . . . . . .
16
A.4
Background on the ranking score . . . . . . . . . . . . . . . . . . . . . . . . . . .
17
A.5
Background on the calibration metrics . . . . . . . . . . . . . . . . . . . . . . . .
20
B
Experimental details
21
B.1
Technical setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
21
B.2
Hyperparameters
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
21
B.3
Dataset preprocessing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
22
B.4
Evaluation on RewardBench . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
22
C Supplementary results
24
A
Theoretical details
A.1
Background on the uncertainty decomposition in preference classification
Traditionally, the total uncertainty is decomposed into epistemic uncertainty, which describes the lack
of knowledge in the model, and aleatoric uncertainty, which captures the irreducible randomness
in the data (Hüllermeier & Waegeman, 2021; Kendall & Gal, 2017). The two commonly used
decompositions of the total uncertainty in terms of the variance and in terms of the entropy have been
explicitly derived by Depeweg et al. (2018) and correspond to
Var [Y | X] = Varθ[EY [Y | X; θ]] + Eθ[VarY [Y | X; θ]]
H(Y | X) = I(Y ; θ)
+ Eθ[H(Y | X, θ)],
with X denoting the input, Y the output and θ the parameters of the assumed underlying statistical
model. The first term describes the epistemic and the second term the aleatoric uncertainty.
In the context of preference classification described in Section 3.1, X denotes the sample (x, y, y′)
and Y the label 1{y≻y′}. The common statistical model, which we adopt in this work, is given by
y ≻y′ | x, y, y′ ∼Ber(σ(rθ(x, y) −rθ(x, y′)))
(16)
based on the Bradley-Terry model (Bradley & Terry, 1952). Importantly, this statistical model
makes the following assumptions: First, it assumes that the preference label contains Bernoulli
noise, which leads to aleatoric uncertainty in the preference predictions. Second, it assumes that the
preference signal comes from a deterministic reward function through the Bradley-Terry model and,
hence, is free of randomness. Hence, under this statistical model assumption, reward models trained
on preference data are free of aleatoric uncertainty and only contain epistemic uncertainty, as the
aleatoric uncertainty is fully captured by the Bernoulli noise model.
Some work (Lou et al., 2025; Yan et al., 2024) use the alternative statistical model assumption
y ≻y′ | R, R′ ∼Ber(σ(R −R′))
with
R | x, y ∼N
 rθ(x, y), s2
θ(x, y)
 
R′ | x, y′ ∼N
 rθ(x, y′), s2
θ(x, y′)
 
which assumes heteroscedastic Gaussian noise in the reward, leading to aleatoric uncertainty in the
underlying reward models. This assumption is equivalent to
y ≻y′ | x, y, y′, ∆ε ∼Ber(σ(rθ(x, y) −rθ(x, y′) + ∆ε))
with
∆ε | x, y, y′ ∼N
 0, s2
θ(x, y) + s2
θ(x, y′)
 
.
15


<!-- page 16 -->
Intuitively, the Gaussian noise assumption in the reward smooths the sigmoid function with a Gaussian
kernel with bandwidth s2
θ(x, y) + s2
θ(x, y′) as we marginalize over ∆ε. Hence, the smoothed sigmoid
function converges towards a constant function at 0.5 with increasing noise level, while the original
sigmoid function is recovered with zero noise.
In summary, we adopt the more common statistical model assumption in Equation (16) and assume
the aleatoric uncertainty to be fully captured by the Bernoulli model, while the underlying reward
model is free of aleatoric uncertainty.
A.2
Background on the symmetry in preference classification
Pairwise preference classification is a special form of binary classification, where the goal is predict
the label 1{y≻y′} ∈{0, 1} for a preference sample (x, y, y′). Unlike standard binary classification,
the label is defined through the antisymmetric relation ≻, which implies
1{y≻y′} = 1 −1{y′≻y}.
Hence, there is no distinction between positive and negative classes in pairwise preference classifi-
cation, since each sample (x, y, y′) is equivalent to its flipped counterpart (x, y′, y) with the class
label inverted. As a result, the predictive accuracy is fully characterized by the win rate defined in
Equation (4), which jointly describes the true positive (TP), true negative (TN), false positive (FP)
and false negative (FN) rate as
win rate = TP rate = TN rate
and
1 −win rate = FP rate = FN rate.
The same antisymmetry extends to preference probabilities, yielding
p(y ≻y′ | x, y, y′) = 1 −p(y′ ≻y | x, y′, y).
Similarly, an upper bound on a preference probability induces a corresponding lower bound for the
flipped comparison and vice versa, i.e.,
p(y ≻y′ | x, y, y′) = 1 −p(y′ ≻y | x, y′, y)
p(y ≻y′ | x, y, y′) = 1 −p(y′ ≻y | x, y′, y).
In theory, this is irrelevant for computing the expected calibration errors defined in Equations (7)
and (8). However, in practice, when approximating these errors via binning, it is essential to
consider both (x, y, y′) and (x, y′, y), effectively doubling the evaluation set (Shen et al., 2024).
This ensures that predictions for both y ≻y′ and y′ ≻y contribute to the empirical frequencies in
the corresponding bins. As a result, the calibration diagram for predictions is point-symmetric at
(0.5, 0.5),11 and the calibration diagrams for upper and lower bounds are equivalent, resulting in
identical calibration errors for both.
A.3
Background on the accuracy metrics
Our extension of accuracy metrics to predictions under uncertainty in Section 3.2 can be generalized
to standard binary classification metrics. Specifically, categorizing predictions into confident and
unconfident introduces an orthogonal dimension, resulting in a three-dimensional confusion tensor
along the axes
{C(onfident), U(nconfident)} × {T(rue), F(alse)} × {P(ositive), N(egative)},
where P and N denote the set of real positives and negatives, respectively. For example, the confident
true positive (CTP) rate is then defined as CTP rate = |C∩T∩P|
|P|
.
In preference classification, there is no distinction between positive and negative classes as described
in Appendix A.2 and the confusion tensor collapses into the 2 × 2 matrix in Equation (5), which
should not be confused with the classical binary confusion matrix. Accordingly, we normalize by the
total number of samples instead of by the per-class counts.
11This is why in a calibration diagram overconfidence appears as a flat line (i.e., above the diagonal on
[0.0, 0.5] and below the diagonal on [0.5, 1.0]), while underconfidence forms a sigmoid-shaped curve.
16


<!-- page 17 -->
(a) Ranking score ranges
(b) Ranking score weights
Figure 4: Background on our ranking score for different α. While the range is invariant of the win
rate for α = 0, it has a linear dependence for α = 1 as shown in Figure 4a. The inherent trade-off
underlying the choice of α is shown in Figure 4b, which visualizes the weights in our ranking score
in Equation (20). For example, with α = 0.2, when the win rate increases from 0.6 to 0.8, the
confidence among true predictions is upweighted from 0.88 to 0.95 by a factor of ≈1.08, while the
confidence among false predictions is downweighted from 0.77 to 0.56 by a factor of ≈0.73.
A.4
Background on the ranking score
Observe that all accuracy metrics introduced in Equations (4) and (5) can be expressed in terms of
the four base counts
(confident)
(unconfident)
(true)
CT = |C ∩T|
↑
UT = |U ∩T|
↘
(false)
CF = |C ∩F|
↓
UF = |U ∩F|
↘
(17)
with T = CT + UT and F = UT + UF. Since the total number of samples is fixed to the size
of the evaluation dataset n = CT + UT + CF + UF, there are only three degrees of freedom,
capturing the overall accuracy of predictions under uncertainty. A ranking strategy reduces these three
degrees of freedom to a single score, effectively compressing two dimensions along which differently
performing models are ranked equally, reflecting the inherent trade-offs made by the ranking strategy.
Recall our proposed ranking score in Equation (6), which can be expressed in terms of these counts as
RSα =
CT rate
win rate + α · (1 −win rate) −
CF rate
(1 −win rate) + α · win rate
=
CT
T + α · F −
CF
F + α · T ,
where α ∈[0, 1] balances the inherent trade-offs between the three degrees of freedom. The general
idea is to encourage confident true predictions and penalize confident false predictions. Depending
on the choice of α, the score normalizes the number of confident predictions differently and puts
a different focus on confidence and accuracy. In the following, we first discuss the two edge cases
α = 0 and α = 1 and then how our ranking score formulation unifies both cases.
17

[CAPTION] Figure 4: Background on our ranking score for different α. While the range is invariant of the win


<!-- page 18 -->
Focus on confidence (RS0)
When α = 0, the ranking score corresponds to
RS0 = CT rate
win rate −
CF rate
1 −win rate = CT
T
−CF
F
∈[−1, 1].
(18)
This ranking score considers the relative rate of confidence among true and false predictions. In
other words, it looks at the proportions of confident predictions conditioned on the correctness of the
predictions. Intuitively, it encourages confidence among true predictions and penalizes confidence
among false predictions, while it does not take into account the overall ratio between true and false
predictions. In particular, observe that the range of this ranking score is invariant of the win rate as
shown in Figure 4a. Overall, this ranking score focuses on the confidence of predictions.
Remark: This ranking score is conceptually related to Youden’s index (Youden, 1950), a summary
statistic in binary classification, which is defined as J =
T P
T P +F N −
F P
T N+F P = |T∩P|
|P|
−|F∩N|
|N| .
Focus on accuracy (RS1)
When α = 1, the ranking score corresponds to
RS1 = CT rate −CF rate = CT
n −CF
n
∈[win rate −1, win rate].
(19)
This ranking score considers the absolute rate of confidence among all samples. In other words,
it looks at the joint proportions of confidence and correctness. Intuitively, it encourages confident
correctness and penalizes confident incorrectness among all samples, while it does not take into
account the number of uncertain true and false predictions. Observe that the range of this ranking
score is determined based on this win rate as shown in Figure 4a. Overall, this ranking score focuses
on the confidence and accuracy of predictions.
Unified formulation (RSα)
The issue of RS0 is that it focuses too much on the confidence and
cannot distinguish between models with completely different win rates, as long as the relative
proportions of confidence are the same. The issue of RS1 is that it focuses too much on the accuracy
and, when setting the uncertainty globally to zero, it simplifies to RS1 = 2 · win rate −1. Hence,
we introduce α ∈[0, 1] to balance the trade-off between both ends. To better understand the effect of
α, we can rewrite our ranking score formulation into
RSα = fα(win rate) · CT rate
win rate −fα(1 −win rate) ·
CF rate
1 −win rate
(20)
which considers the relative rate of confidence, but each weighted by some factor fα(x) =
x
x+α·(1−x)
depending on the win rate. Intuitively, a higher win rate puts more weight on the bonus caused by
confidence among true predictions, while a lower win rate puts more weight on the penalty caused by
confidence among false predictions. We visualize fα(x) for different choices of α in Figure 4b. In
this work, we choose α = 0.2 as it balances well the trade-off between confidence and accuracy.
A.4.1
Invariances of RS0
Recall Equation (18), which can be expressed in terms of the base counts from Equation (17) as
RS0 =
CT
CT + UT −
CF
CF + UF .
This “difference of two ratios” introduces two invariances, as analyzed in the following.
Invariance 1 (by normalization)
One degree of freedom is lost due to normalization in the ratios.
Hence, RS0 is invariant to changes in the numerator and denominator, as long as each ratio is
preserved, e.g.,
CT
UT
CF
UF
RS0
40
60
2
8
0.2 = 0.4 −0.2
42
63
1
4
0.2 = 0.4 −0.2
18


**[Table p18.1]**
| CT UT CF UF | RS 0 |
| --- | --- |


**[Table p18.2]**
| 40 60 2 8 42 63 1 4 | 0.2 = 0.4 −0.2 0.2 = 0.4 −0.2 |
| --- | --- |


<!-- page 19 -->
Intuitively, two models with different numbers of true and false predictions (i.e., different win rates)
are ranked equally if they share the same proportion of confident predictions among true and false
predictions. Formally, we can scale the numerators and denominators in each ratio equally
(confident)
(unconfident)
(true)
CT →
 
1 + δ
T
 
CT
UT →
 
1 + δ
T
 
UT
(false)
CF →
 
1 −δ
F
 
CF
UF →
 
1 −δ
F
 
UF
based on some δ.12 The resulting ranking score
RS0(δ) =
 1 + δ
T
 
CT
 1 + δ
T
 
CT +
 1 + δ
T
 
UT −
 1 −δ
F
 
CF
 1 −δ
F
 
CF +
 1 −δ
F
 
UF
=
CT
CT + UT −
CF
CF + UF
= const.
is independent of δ and thus constant.
Invariance 2 (by taking the difference)
The other degree of freedom is eliminated by taking the
difference between the two terms. Accordingly, RS0 is invariant to changes in both terms, provided
their difference stays constant, e.g.,
CT
UT
CF
UF
RS0
40
60
2
8
0.2 = 0.4 −0.2
70
30
5
5
0.2 = 0.7 −0.5
Intuitively, two models with different degrees of confidence are ranked equally if the true and false
predictions are affected by the confidence level similarly. Formally, we can increase or decrease the
number of confident predictions for both true and false predictions
(confident)
(unconfident)
(true)
CT →CT + δ · T
UT →UT −δ · T
(false)
CF →CF + δ · F
UF →UF −δ · F
based on some δ. The resulting ranking score
RS0(δ) =
CT + δ · T
(CT + δ · T) + (UT −δ · T) −
CF + δ · F
(CF + δ · F) + (UF −δ · F)
=
CT
CT + UT −
CF
CF + UF
= const.
remains independent of δ, confirming the invariance.
A.4.2
Invariances of RS1
Recall Equation (19), which can be expressed in terms of the base counts from Equation (17) as
RS1 =
CT
CT + UT + UF + CF −
CF
CT + UT + UF + CF .
Note that both denominators are constant, as they correspond to the number of evaluation samples.
12For simplicity, we omit that δ must ensure non-negative integer counts.
19


**[Table p19.1]**
| 40 60 2 8 70 30 5 5 | 0.2 = 0.4 −0.2 0.2 = 0.7 −0.5 |
| --- | --- |


<!-- page 20 -->
Invariance 1 (by indistinction between unconfident predictions)
One degree of freedom is lost
by not considering the base counts UT and UF separately. Hence, RS1 is invariant to changes in
both counts as long as their sum is preserved, e.g.,
CT
UT
CF
UF
RS1
40
60
2
8
0.35 ≈40/110 −2/110
40
8
2
60
0.35 ≈40/110 −2/110
Intuitively, the ranking score does not distinguish between unconfident true and false predictions.
Formally, we can change the base counts of unconfident predictions
(confident)
(unconfident)
(true)
CT →CT
UT →UT + δ
(false)
CF →CF
UF →UF −δ
based on some δ. The resulting ranking score
RS1(δ) =
CT
CT + (UT + δ) + (UF −δ) + CF −
CF
CT + (UT + δ) + (UF −δ) + CF
=
CT
CT + UT + UF + CF −
CF
CT + UT + UF + CF
= const.
is independent of δ and thus constant.
Invariance 2 (by taking the difference)
The other degree of freedom is eliminated by taking the
difference between the two terms. Accordingly, RS1 is invariant to changes in both terms, provided
their difference stays constant, e.g.,
CT
UT
CF
UF
RS1
40
60
2
8
0.35 ≈40/110 −2/110
48
52
10
0
0.35 ≈48/110 −10/110
Intuitively, two models with different degrees of confidence are ranked equally if the true and false
predictions are affected by the confidence level similarly. Formally, we can increase or decrease the
number of confident predictions for both true and false predictions
(confident)
(unconfident)
(true)
CT →CT + δ
UT →UT −δ
(false)
CF →CF + δ
UF →UF −δ
based on some δ. The resulting ranking score
RS1(δ) =
(CT + δ)
(CT + δ) + (UT −δ) + (UF −δ) + (CF + δ)
−
(CF + δ)
(CT + δ) + (UT −δ) + (UF −δ) + (CF + δ)
=
CT
CT + UT + UF + CF −
CF
CT + UT + UF + CF
= const.
remains independent of δ, confirming the invariance.
A.5
Background on the calibration metrics
For readability, we use Y ≻∈{0, 1} to denote the event y ≻y′ for given x, y and y′.
20


**[Table p20.1]**
| 40 60 2 8 40 8 2 60 | 0.35 ≈40/110 −2/110 0.35 ≈40/110 −2/110 |
| --- | --- |


<!-- page 21 -->
Calibration of predictions
The predicted preference probabilities are well-calibrated if they match
the true preference probabilities, i.e.,
P(Y ≻| pθ(Y ≻) = p) = p
for all p ∈[0, 1], following Guo et al. (2017). The expected calibration error (ECE) is defined as
ECE = Ep
   P(Y ≻| pθ(Y ≻) = p) −p
   
,
which penalizes over- and underestimations of the true preference probabilities. Since the true
probabilities are unknown in practice, we measure the deviation from the empirical probabilities.
Specifically, the predicted probabilities are grouped into M bins {Bm}M
m=1 and we compute
ECE ≈
M
X
m=1
|Bm|
n
|P(Bm) −pθ(Bm)|
↓
with empirical probability P(Bm) and average predicted probability pθ(Bm) in each bin Bm (Guo
et al., 2017; Pavlovic, 2025), as stated in Equation (7).
Calibration of bounds
The predicted preference probability bounds are well-calibrated if they are
not violated by the true preference probabilities, i.e.,
P(Y ≻| pθ(Y ≻) = p) ≥p
P(Y ≻| pθ(Y ≻) = p) ≤p
for all p, p ∈[0, 1]. We introduce the expected lower calibration error (ELCE) and expected upper
calibration error (EUCE) as follows
ELCE = Ep
 
max
 p −P
 Y ≻| pθ(Y ≻ 
= p), 0
  
EUCE = Ep
 
max
 P
 Y ≻| pθ(Y ≻) = p
 
−p, 0
  
with ELCE penalizing lower bounds that overestimate the true preference probability and EUCE
penalizing upper bounds that underestimate the true preference probability. In practice, analogous to
ECE, we group the lower and upper bounds separately into M bins {Bm}M
m=1 and compute
ELCE ≈
M
X
m=1
|Bm|
n
max
 pθ(Bm) −P(Bm), 0
 
↓
EUCE ≈
M
X
m=1
|Bm|
n
max(P(Bm) −pθ(Bm), 0)
↓
with pθ(Bm) and pθ(Bm) denoting the average predicted lower and upper bounds in the correspond-
ing bin Bm, as stated in Equation (8).
B
Experimental details
B.1
Technical setup
All models were trained on a single node equipped with four NVIDIA GH200 GPUs, providing a
total of 378GB of VRAM. Our implementation is built on top of Transformers (Wolf et al., 2020) and
TRL (von Werra et al., 2020) by HuggingFace, with multi-GPU management handled by Accelerate
(Gugger et al., 2022). We use data parallelism for models that fit on a single GPU, and model and
tensor parallelism for larger models.
B.2
Hyperparameters
We used the AdamW optimizer (Loshchilov & Hutter, 2019) (with a weight decay of 0, β1 = 0.9,
β2 = 0.999, and ϵ = 10−8) and an effective batch size of 64 via gradient accumulation for all
experiments, except for the LoRA adapter ensemble, which used a batch size of 16. We trained our
models using a single epoch with a cosine learning rate scheduler and a warm-up phase of 5% of the
total number of steps. The reward uncertainty bounds in Equation (9) are constructed using β = 2 for
the MLP head ensemble and DPO-based MC dropout and β = 0.5 for the Bayesian linear head. The
best hyperparameters for each base model are given in Table 1.
21


<!-- page 22 -->
MLP head ensemble
We use an ensemble of K = 20 MLP heads. Each head is a two-layer
network with 128 nodes per layer and ReLU activation functions. They are initialized using a
Xavier uniform distribution with a gain of one. We conducted a grid search over the learning rate
η ∈{10−5, 10−4, 10−3} and regularization parameters λ ∈{0.0, 0.1, 1.0} and γ ∈{0.0, 0.01, 0.1}
from Equation (11).
LoRA adapter ensemble
We use an ensemble of K = 8 LoRA adapter with rank rLoRA = 16 and
scaling factor αLoRA = 32. We conducted a grid search over the learning rate η ∈{10−5, 10−4, 10−3}
and regularization parameters λ ∈{0.001, 0.01, 0.1} and choose γ = 0.01.
DPO-based MC dropout
The ensemble is formed by sampling K = 20 dropout masks at inference
time. We conducted a grid search over the learning rate η ∈{10−7, 10−6, 10−5}, DPO regularization
parameter λ ∈{0.01, 0.05, 0.1} and dropout rate pdropout ∈{0.01, 0.05, 0.1} from Equation (13).
Bayesian linear head
We conducted a grid search over the learning rate η ∈{10−3, 10−2, 10−1}
and ℓ2-regularization parameter λ ∈{10−3, 10−2, 10−1} from Equation (14).
B.3
Dataset preprocessing
For all datasets, we remove preference samples if the total sequence length of the prompt and one of
the completions exceeds 2048 tokens to avoid additional evaluation noise due to the truncation of the
prompt and completions.
B.4
Evaluation on RewardBench
RewardBench (Lambert et al., 2025b) consists of the four main categories “Chat”, “Chat Hard”,
“Safety”, and “Reasoning”, each with weighted subcategories. Following the standard procedure,
we first compute our metrics from Sections 3.2 and 3.3 (excluding the ranking score) within each
category using the subcategory weights, then average them across categories, and finally derive the
ranking score from these averages.
22


<!-- page 23 -->
Table 1: Best hyperparameters found according to our evaluation procedure described in Section 5.1.
base model
ENS-MLP
ENS-LoRA
MCD-DPO
BAY-LIN
η
λ
γ
η
λ
γ
η
λ
pdropout
η
λ
Qwen3-0.6B
0.001
0
0.01
0.0001
0.01
0.01
0.00001
0.01
0.05
0.01
0.01
Qwen3-1.7B
0.001
0
0.01
0.0001
0.01
0.01
0.00001
0.01
0.1
0.001
0.01
Qwen3-4B
0.001
0.1
0
0.0001
0.001
0.01
0.00001
0.01
0.1
0.001
0.001
Qwen3-8B
0.001
0.1
0
0.001
0.01
Qwen3-14B
0.001
1
0
0.001
0.01
Qwen3-32B
0.001
1
0.01
0.001
0.1
Skywork-Qwen3-0.6B
0.001
0.1
0.01
0.0001
0.01
0.01
0.00001
0.01
0.1
0.1
0.001
Skywork-Qwen3-1.7B
0.001
1
0.01
0.0001
0.01
0.01
0.00001
0.01
0.1
0.1
0.01
Skywork-Qwen3-4B
0.001
0.1
0.01
0.0001
0.001
0.01
0.00001
0.01
0.05
0.1
0.1
Skywork-Qwen3-8B
0.001
1
0.01
0.1
0.1
23

[CAPTION] Table 1: Best hyperparameters found according to our evaluation procedure described in Section 5.1.


<!-- page 24 -->
C
Supplementary results
Figure 5 provides detailed results supplementing the discussion in Section 5.2. Figure 5a provides a
granular decomposition of the ranking score trends, explicitly isolating the contributions of win rate,
Confident True rate, and Confident False rate. Validating the importance of initialization, methods
utilizing the task-aligned Skywork base model consistently outperform the generic Qwen 3 base
models across both win rate and CT rate. This advantage is particularly pronounced for fixed-head
methods like BAY-LIN, which rely heavily on high-quality embeddings to produce confident, correct
predictions. Furthermore, the diminishing returns in ranking scores observed in Section 5.2 are
elucidated by the behavior of the CF rate relative to other metrics. While win rates generally improve
or plateau with model size, the CF rate decreases at a lower rate than the CT rate increases.
Regarding calibration, Figure 5b supports the general observation that most UQ methods maintain
reasonable calibration, with Expected Calibration Error (ECE) and Expected Bound Calibration
Error (EBCE) typically remaining below 0.10 and 0.04, respectively. However, the breakdown
reveals specific instabilities that impact the aggregate performance; for example, ENS-MLP trained
on the generic Qwen 3 base exhibits a sharp spike in EBCE at the 32B scale. This degradation in
bound calibration aligns with the underperformance of ENS-MLP in the main ranking results for that
configuration, suggesting that larger model sizes can lead to overfitting instabilities.
Overall, these supplementary metrics confirm that the superior ranking of task-aligned models is
driven not just by higher accuracy, but by a more favorable balance of maximizing confident true
predictions while suppressing confident errors.
24

[CAPTION] Figure 5 provides detailed results supplementing the discussion in Section 5.2. Figure 5a provides a


<!-- page 25 -->
UQ method
ENS-MLP
ENS-LoRA
MCD-DPO
BAY-LIN
UQ method
ENS-MLP
ENS-LoRA
MCD-DPO
BAY-LIN
Base model
Qwen 3 (solid)
Skywork Qwen 3 (dashed)
(a) Accuracy metrics
UQ method
ENS-MLP
ENS-LoRA
MCD-DPO
BAY-LIN
UQ method
ENS-MLP
ENS-LoRA
MCD-DPO
BAY-LIN
Base model
Qwen 3 (solid)
Skywork Qwen 3 (dashed)
(b) Calibration metrics
Figure 5: Our base metrics on RewardBench across different UQ methods, training datasets, pretrained
and finetuned models, and model sizes. The accuracy metrics are defined in Equations (4) and (5)
and the calibration metrics in Equations (7) and (8).
25

[CAPTION] Figure 5: Our base metrics on RewardBench across different UQ methods, training datasets, pretrained