<!-- page 1 -->
Revisiting Unknowns: Towards Effective and Efficient Open-Set Active Learning
Chen-Chen Zong, Yu-Qi Chi, Xie-Yang Wang, Yan Cui, Sheng-Jun Huang*
Nanjing University of Aeronautics and Astronautics
Nanjing, 211106, China
{chencz,chiyuqi,xieyang,cuiyan,huangsj}@nuaa.edu.cn
Abstract
Open-set active learning (OSAL) aims to identify infor-
mative samples for annotation when unlabeled data may
contain previously unseen classes—a common challenge
in safety-critical and open-world scenarios. Existing ap-
proaches typically rely on separately trained open-set de-
tectors, introducing substantial training overhead and over-
looking the supervisory value of labeled unknowns1 for
improving known-class learning. In this paper, we propose
E2OAL (Effective and Efficient Open-set Active Learning),
a unified and detector-free framework that fully exploits la-
beled unknowns for both stronger supervision and more
reliable querying.
E2OAL first uncovers the latent class
structure of unknowns through label-guided clustering in
a frozen contrastively pre-trained feature space, optimized
by a structure-aware F1-product objective. To leverage la-
beled unknowns, it employs a Dirichlet-calibrated aux-
iliary head that jointly models known and unknown cate-
gories, improving both confidence calibration and known-
class discrimination. Building on this, a logit-margin pu-
rity score estimates the likelihood of known classes to con-
struct a high-purity candidate pool, while an OSAL-specific
informativeness metric prioritizes partially ambiguous yet
reliable samples. These components together form a flex-
ible two-stage query strategy with adaptive precision con-
trol and minimal hyperparameter sensitivity. Extensive ex-
periments across multiple OSAL benchmarks demonstrate
that E2OAL consistently surpasses state-of-the-art methods
in accuracy, efficiency, and query precision, highlighting its
effectiveness and practicality for real-world applications.
The code is available at github.com/chenchenzong/E2OAL.
1. Introduction
Deep learning has achieved remarkable success across a
wide range of domains, largely fueled by the availability
*Corresponding Author.
1Here, unknowns denotes samples from unknown (open-set) classes.
of large-scale datasets with high-quality annotations [9, 16,
25].
However, collecting such annotations is often pro-
hibitively expensive, requiring extensive human effort and
domain expertise [26, 30]. These limitations hinder the de-
ployment of deep models in real-world scenarios where la-
beled data is scarce or costly.
Active learning (AL) mitigates this issue by iteratively
selecting a small yet informative subset for annotation [30].
Traditional AL methods typically select samples based on
model uncertainty [31, 37, 40], sample diversity [1, 28, 36],
or combinations thereof [13, 22, 32]. Despite their suc-
cess, most AL methods assume a closed set—that all unla-
beled samples belong to known classes [20]. This assump-
tion rarely holds in practice, especially in safety-critical do-
mains such as autonomous driving [23, 35] or medical diag-
nosis [4, 33], where unlabeled data often includes samples
from previously unseen classes. Under such open-set con-
ditions, conventional AL strategies tend to over-select un-
known class samples—mistaking their high uncertainty or
feature-level novelty for informativeness—thereby degrad-
ing the overall learning process. These challenges motivate
the study of open-set active learning (OSAL), which aims to
query informative yet class-consistent samples while prop-
erly handling unknowns.
Recent advances in OSAL incorporate surrogate out-of-
distribution (OOD) detection mechanisms to guide query-
ing [6, 20, 21, 27, 38, 41, 42]. These methods typically
combine uncertainty or diversity with auxiliary OOD scores
to balance informativeness and purity. However, they often
rely on separately trained OOD detectors, which incur sub-
stantial training overhead, and further overlook the poten-
tial of labeled unknowns as valuable supervision for en-
hancing known-class learning. This observation calls for a
unified, detector-free OSAL framework that can both detect
and utilize unknowns to enhance learning and querying.
To explore this, we first conduct a pilot study examin-
ing whether labeled unknowns benefit known-class learn-
ing.
Assuming access to ground-truth labels for labeled
unknowns, we train a dual-head classifier:
a primary
head optimized for known classes and an auxiliary head
arXiv:2603.07898v1  [cs.CV]  9 Mar 2026


<!-- page 2 -->
0
2
4
6
8
10
Number of AL rounds
40
50
60
70
Test accuracy
Mean accuracy
Certainty-based sampling
Cknown
Cknown + 1
Call
0
2
4
6
8
10
Number of AL rounds
40
50
60
70
Test accuracy
Mean accuracy
Uncertainty-based sampling
0
2
4
6
8
10
Number of AL rounds
40
50
60
70
Test accuracy
Mean accuracy
Random sampling
Figure 1. Per-round and mean test accuracy on CIFAR-100 (40
known / 60 unknown) using ResNet-18. Cknown excludes labeled
unknowns, Cknown+1 collapses them into a single “unknown”
class, and Call leverages their true labels. Call consistently per-
forms best, suggesting that preserving the latent structure of un-
known classes benefits known-class learning.
jointly trained on both known and unknown categories. As
shown in Figures 1 and C1 (Appendix), incorporating la-
beled unknowns into training with their ground-truth la-
bels consistently improves test accuracy. In contrast, merg-
ing them into a single “unknown” class may hinder perfor-
mance. These results reveal that the latent structure within
unknowns provides useful inductive signals. Meanwhile,
a well-designed auxiliary head can also implicitly support
unknowns detection—without explicit OOD models.
In this paper, we present E2OAL (Effective and Efficient
Open-set Active Learning), a unified, detector-free OSAL
framework that converts unknown-class feedback into both
stronger supervision and more informative queries. At each
AL round, E2OAL estimates the number and composition
of unknown classes by clustering all labeled samples, in-
cluding both knowns and unknowns, in a frozen con-
trastively representation space. The optimal cluster count is
determined by aligning clusters with known labels and max-
imizing a structure-aware F1-product objective. To lever-
age supervision from labeled unknowns, E2OAL employs
a Dirichlet-calibrated auxiliary head that improves known-
class discrimination and provides calibrated confidence un-
der open-set conditions. For query selection, we introduce a
lightweight logit-margin purity score to estimate the likeli-
hood of known classes and a tailored informativeness metric
that favors samples with moderate uncertainty while sup-
pressing those with overly ambiguous or confident predic-
tions. These scores drive a two-stage query strategy: first
constructing a high-purity candidate pool, then selecting
the most informative samples for annotation. Crucially, the
candidate pool size is dynamically adjusted to meet a tar-
get query precision, without introducing additional hyper-
parameters. Our main contributions are summarized as:
• We propose E2OAL, a unified and detector-free OSAL
framework that transforms unknown-class feedback into
both effective supervision and informative querying.
• We introduce a label-guided clustering strategy for auto-
matic class estimation within a contrastive feature space,
enabling structure-aware discovery of unknown classes.
• We develop a Dirichlet-calibrated auxiliary head with an
associated logit-margin purity score to enhance known-
class learning and provide reliable confidence calibration.
• We design an OSAL-specific informativeness metric that
favors samples with moderate uncertainty while suppress-
ing overly ambiguous or overconfident ones.
• We present a flexible two-stage selection scheme that dy-
namically adjusts to maintain a fixed target query preci-
sion—without introducing additional hyperparameters.
• Extensive experiments on multiple OSAL benchmarks
show that E2OAL consistently outperforms state-of-the-
art methods in accuracy, efficiency, and query precision.
2. Related Work
Active learning (AL) aims to reduce annotation cost by
selectively querying the most informative samples while
maintaining model performance [26, 30]. Classical AL ap-
proaches can be broadly categorized into uncertainty-based,
diversity-based, and hybrid strategies. Uncertainty-based
methods query samples with high model uncertainty, mea-
sured by entropy [12], margin [3], confidence scores [18], or
Monte Carlo dropout variance [7]. Diversity-based methods
instead seek to represent the underlying data distribution
through clustering [19] or coreset selection [28]. Hybrid
methods, such as BADGE [2], combine both uncertainty
and diversity by clustering in the gradient space to balance
representativeness and informativeness.
Most conventional AL methods operate under a closed-
set assumption, where unlabeled and labeled data share the
same label space. This assumption, however, fails in open-
set active learning (OSAL), where the unlabeled pool may
contain out-of-distribution (OOD) or open-set samples ir-
relevant to the known classes. In such cases, standard AL
strategies often over-select OOD samples that appear uncer-
tain or diverse yet provide little learning value—much like
requesting annotations for unrelated concepts—ultimately
degrading model performance.
To address this, various OSAL methods have been de-
veloped. LfOSA [20] trains an auxiliary detector to distin-
guish known classes, guiding queries using maximum acti-
vation values. MQNet [21] adopts a meta-learning frame-
work that balances purity and informativeness via a multi-
layer perceptron. More recent works such as EOAL [27],
BUAL [42], and EAOA [41] jointly assess purity and in-
formativeness within a detector-based paradigm: EOAL

[CAPTION] Figure 1. Per-round and mean test accuracy on CIFAR-100 (40


<!-- page 3 -->
𝑆𝐼
𝑆𝐼(𝑥)
𝑆𝑃(𝑥)
Unknown class
Known class
Labeled pool
Unlabeled pool
Target
model
Feature
encoder
⋯⋯
Features
Clustering
Class 
mapping
a:
b:
c:
d:
e:
𝑘+ ො𝑢
Evaluate and adjust ො𝑢 
Initialize
 ො𝑢
Target model
Low
High
Dynamic partitioning
Oracle
𝑆𝑃
Low
High
𝒍𝒐𝒈𝒊𝒕
𝑃(𝑦|𝑥) =
𝑒𝑜𝑦
σ𝑐𝑒𝑜𝑐
𝑃(𝑦|𝑥) =
𝛼𝑦+ 𝛾
σ𝑐(𝛼𝑐+𝛾)
ℒ𝐸𝐷𝐿
ℒ𝐶𝐸
𝐷𝑖𝑟(𝜶)
𝑜1, 𝑜2, ⋯
ො𝑢
Figure 2. Overview of the proposed E2OAL framework. Each AL round consists of two stages: (1) Adaptive class estimation and
calibration-aware training, where latent unknown classes are discovered via label-guided clustering and incorporated into model learning
through Dirichlet-based auxiliary supervision; (2) Flexible two-stage query selection, where a high-purity candidate pool is first constructed
using a purity score guided by a target query precision, followed by informativeness-driven sample selection.
defines separate entropy scores for known and unknown
classes; BUAL leverages bidirectional uncertainty to refine
selection boundaries; and EAOA integrates epistemic and
aleatoric uncertainty for improved sample evaluation.
Despite recent progress, most OSAL methods still de-
pend on separately trained detector networks, which add
notable computational burden. In addition, they fail to fully
exploit labeled unknowns—annotations that could pro-
vide valuable supervision for improving known-class dis-
crimination. These limitations motivate our work, which
revisits the role of labeled unknowns and introduces a uni-
fied, detector-free OSAL framework that converts them into
effective supervision and reliable query guidance.
3. The Proposed Approach
In this section, we present E2OAL, a unified and detector-
free framework for open-set active learning (OSAL) in
multi-class image classification, where the unlabeled pool
may contain samples from previously unseen classes. An
overview of the framework is illustrated in Figure 2, with
detailed pseudocode provided in Algorithm B1 (Appendix).
3.1. Preliminaries
Notations. We consider an OSAL setting where the unla-
beled pool DU contains both known and unknown classes.
Let Ck and Cu denote the sets of known and unknown
classes, with |Ck| = k, |Cu| = u, and Ck ∩Cu = ∅. The la-
beled set DL initially includes only samples from Ck, while
DU covers Ck ∪Cu, with |DL| ≪|DU|. At each active
learning (AL) round, a batch B ⊂DU is queried and an-
notated. If a sample belongs to Ck, its true class label is
provided; otherwise, it receives a single “unknown” label,
without revealing its actual class in Cu. The newly labeled
batch is added to DL, resulting in DL = Dkno
L
∪Dunk
L
for
known and unknown subsets, respectively. The observed
query precision is ¯p∗= |Bkno|/|B|. Our goal is to train a
classifier fθ that generalizes well on Ck with minimal label-
ing cost, even under contamination from unknowns.
Overview. Each AL round in E2OAL proceeds in two
sequential stages: first, it uncovers and exploits the latent
class structure of labeled unknowns to strengthen model
training; then, it selects informative yet likely-known in-
stances through an adaptive query strategy.
In stage 1, E2OAL performs label-guided clustering on
DL within a frozen contrastive representation space (e.g.,
from CLIP [25], MoCo [10], or SimCLR [5]) to reveal la-
tent unknown-class structures.
A simple structure-aware
F1-product objective determines the optimal cluster config-
uration, enabling adaptive class discovery. The resulting
pseudo-clusters are treated as auxiliary classes for an addi-
tional classification head, trained jointly with the primary
head using a Dirichlet-based calibration loss.
This joint
training improves known-class discrimination and yields
calibrated confidence estimates. In stage 2, E2OAL esti-
mates a logit-margin purity score for each unlabeled sam-
ple to measure its likelihood of belonging to known classes.
Samples surpassing an adaptively adjusted purity margin,
set to maintain a target query precision, form a high-purity
candidate pool. Within this pool, a tailored informativeness
metric evaluates each sample’s utility under open-set uncer-
tainty, prioritizing those that are both reliable and informa-
tive. By combining these mechanisms, E2OAL adaptively
selects high-quality queries and achieves efficient, detector-
free OSAL without additional hyperparameter tuning.


**[Table p3.1]**
| a: ⋯ ⋯ b: Feature Initialize 𝑘+ 𝑢ො Clustering Class c: encoder 𝑢ො mapping d: Features e: 𝑢ො Evaluate and adjust 𝑢ො |  |  |  |  | Unlabeled pool Target model 𝑆 (𝑥) 𝑆 (𝑥) 𝑃 𝐼 High High 𝑆 𝐼 Low 𝑆 𝑃 Low Oracle Dynamic partitioning |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |  |
| Unknown class 𝛼 + 𝛾 𝑃(𝑦|𝑥) = σ (𝑦 𝛼 +𝛾) ℒ 𝐸𝐷𝐿 Target 𝐷𝑖𝑟(𝜶) 𝑐 𝑐 model Known class 𝑒𝑜𝑦 𝑜 1 𝒍, 𝑜 𝒈2 𝒊, 𝒕⋯ 𝑃(𝑦|𝑥) = σ 𝑐𝑒𝑜𝑐 ℒ 𝐶𝐸 𝒐 Labeled pool |  |  |  |  | 𝑆 (𝑥) 𝑃 |  |  | 𝑆 (𝑥) 𝐼 |  |  |
|  |  |  |  |  | High |  |  | High 𝑆 𝐼 |  |  |
|  |  | ℒ 𝐸𝐷𝐿 |  |  |  |  |  |  |  |  |
|  |  |  |  |  | 𝑆 𝑃 |  |  |  |  |  |
|  | 𝑒𝑜𝑦 |  |  |  | Low |  |  |  |  |  |
|  | 𝑃(𝑦|𝑥) = σ 𝑐𝑒𝑜𝑐 |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |


**[Table p3.2]**
|  |  | ⋯ |  | ⋯ |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |


**[Table p3.3]**
|  | 𝛼 + 𝛾 𝑦 𝑃(𝑦|𝑥) = σ (𝛼 +𝛾) 𝑐 𝑐 𝑒𝑜𝑦 |
| --- | --- |
| 𝑜 , 𝑜 , ⋯ 1 2 𝒍𝒐𝒈𝒊𝒕 |  |

[CAPTION] Figure 2. Overview of the proposed E2OAL framework. Each AL round consists of two stages: (1) Adaptive class estimation and


<!-- page 4 -->
3.2. Adaptive Class Estimation
To utilize labeled unknowns effectively, E2OAL estimates
the latent class structure of Dunk
L
, including both the num-
ber and composition of unknown categories. Unlike tra-
ditional clustering methods that assume a fixed number of
classes, E2OAL operates under the open-set assumption,
where neither the number nor the granularity of unknown
classes is known a priori.
To capture latent relationships among samples, we op-
erate in a semantically rich and noise-robust contrastively
pretrained feature space. Here, we leverage frozen repre-
sentations from CLIP [25] 2, a pre-trained vision-language
model known for providing high-quality, task-agnostic em-
beddings without additional fine-tuning. Within this fea-
ture space, we perform clustering on the labeled set DL.
For each candidate number of clusters ˆu ∈{k + 1, k +
2, . . . , ˆumax}, clustering quality is evaluated by one-to-
one aligning predicted assignments with ground-truth labels
(covering k known classes and a unified “unknown” class)
using Hungarian algorithm [15]. Assignments unmatched
to known classes are treated as unknown categories. The
clustering score is defined as the product of per-class F1
scores, jointly capturing alignment and structural fidelity.
The choice of ˆu has a strong impact on clustering qual-
ity [34]:
underestimating ˆu may merge distinct known
classes, while overestimating it can fragment coherent ones;
both reduce the F1 score and hence the product. To deter-
mine the optimal ˆu efficiently, we employ a ternary search
to maximize the F1-product score, thereby identifying the
best cluster configuration. Each labeled unknown sample is
then assigned to its nearest inferred cluster, which serves as
a proxy category for downstream training.
The detailed pseudocode is provided in Algorithm 1.
3.3. Dirichlet-Based Calibration
While incorporating labeled unknowns improves general-
ization, reliable confidence estimation in open-set settings
remains challenging. Standard softmax classifiers are noto-
riously overconfident [8, 29, 43], especially for outliers, be-
cause the softmax function is translation-invariant—adding
or subtracting a constant from all logits does not alter the
predicted probabilities. For example, logits [0, 5, 0, 0] and
[−5, 0, −5, −5] both yield a softmax confidence of roughly
0.88 for the second class, even though the latter encodes
far weaker semantic evidence. This results in misleadingly
high confidence for semantically ambiguous or abnormal
inputs, which is particularly detrimental in OSAL, where
confidence estimation plays a critical role.
To alleviate this overconfidence, we adopt a translation-
aware variant of softmax that explicitly breaks translation
2Our experiments show that MoCo and CLIP achieve similar perfor-
mance, while CLIP can be directly used without additional training.
Algorithm 1 The adaptive class estimation algorithm
Input: Labeled data pool DL = {(xi, yi)}, known class
count k, and upper limit ˆumax
Output: Estimated number of unknown classes ˆu
1: Extract CLIP features {fi} for all xi ∈DL
2: # Ternary search for optimal unknown class count
3: Initialize bounds: l ←k + 1, r ←ˆumax
4: while r −l > 2 do
5:
m1 ←⌊l + r−l
3 ⌋,
m2 ←⌊r −r−l
3 ⌋
6:
sm1 ←EVALUATE(m1, {fi}, k)
7:
sm2 ←EVALUATE(m2, {fi}, k)
8:
if sm1 < sm2 then
9:
l ←m1
10:
else
11:
r ←m2
12:
end if
13: end while
14: ˆu ←arg maxm∈{l,l+1,...,r} EVALUATE(m, {fi}, k)
Function EVALUATE(m, {fi}, k):
1: C ←k + m
2: Perform K-Means clustering on {fi} into C clusters
3: Match the k +1 clusters to k +1 classes—including all
k known classes and a unified unknown class—using
the Hungarian algorithm [15]
4: Assign the remaining (C −(k + 1)) clusters to the un-
known class as well
5: Compute class-wise F1 scores {F1c}k+1
c=1
6: return Qk+1
c=1 F1c
invariance by introducing an additive constant. Given logits
o = [o1, o2, . . . , ok+ˆu] for a sample x, the calibrated prob-
ability for class y is:
 \labe l
 {e q 1
} P(y
|x)  = \ f ra
c {e^{o_{y}} + \gamma }{\sum _{c=1}^{k+\hat {u}} (e^{o_{c}} + \gamma )}, 
(1)
where γ > 0 is a calibration constant. This modification en-
sures that predicted probabilities reflect both relative logit
magnitudes and absolute evidence.
The additive term γ
serves as a smoothing factor, suppressing overconfidence
for low-evidence samples. For instance, with γ = 1, the
probabilities for the above logits decrease from 0.88 to ap-
proximately 0.60 and 0.38, respectively, improving confi-
dence separation between confident and ambiguous cases.
However, directly optimizing this formulation with the
commonly adopted cross-entropy (CE) loss introduces a
gradient mismatch: the additive constant enlarges the logit
margin required for confident predictions, thereby requiring
gradient compensation to maintain effective optimization.
A straightforward remedy is to impose auxiliary regu-
larization (e.g., ℓ2 or KL divergence) to promote calibrated
outputs. However, such rigid constraints often conflict with


<!-- page 5 -->
CE loss, leading to unstable optimization. To address this,
we adopt Evidential Deep Learning (EDL) [17, 29, 43],
which models predictive probabilities as Dirichlet distribu-
tions and provides a principled, soft calibration mechanism
naturally aligned with the modified softmax.
In EDL, the predictive distribution p = [p1, . . . , pk+ˆu] is
drawn from a Dirichlet distribution Dir(α), parameterized
by α = [α1, . . . , αk+ˆu]. The expected class probability is:
 \labe l  {eq2} P(y|x)  
= 
\math
bb {E
}_{\boldsymbol {p} \sim \mathrm {Dir}(\boldsymbol {\alpha })}[p_y] = \frac {\alpha _y}{\sum _{c=1}^{k + \hat {u}} \alpha _c}. 
(2)
We define α via the model’s logits as α = g(o)
γ
+ 1, where
g(·) is a non-negative activation (e.g., exponential) and γ is
the calibration constant in Eq. (1), which can be implicitly
learned by the network during training. The term e = g(o)
represents the evidence—the model’s support for each class.
Substituting α = g(o)
γ
+ 1 into Eq. (2) with g(·) = exp(·)
recovers the translation-aware softmax in Eq. (1).
Training is guided by a composite objective combining
predictive accuracy and distributional regularization. The
first term, a negative log-likelihood (NLL) loss, encourages
high expected confidence on the true label:
 \la b e l { eq3} \ m a thc
al
 {L}_
{\m at
hrm {NLL}} = - \log P(y|x) = - \log \frac {\alpha _{y}}{\sum _{c=1}^{k+\hat {u}} \alpha _{c}}. 
(3)
The second term, a KL divergence, penalizes excessive
evidence for incorrect classes by regularizing the output
Dirichlet toward a non-informative uniform prior:
 \l a be
l
 {eq4} \ mathca
l
 {L}_{\mathrm {KL}} = \mathrm {KL}\big (\mathrm {Dir}(\tilde {\boldsymbol {\alpha }}) \parallel \mathrm {Dir}(\mathbf {1})\big ), 
(4)
where ˜α = y + (1 −y) ⊙α, y is the one-hot label en-
coding, 1 is a vector of ones, and ⊙denotes element-wise
multiplication. The full expansion of Eq. (4) is provided in
the Appendix (Sec. A).
Finally, the overall training loss is defined as:
 \ lab e l {e q 5} \ mathc a l {L} = \mathcal {L}_{\mathrm {CE}} + \mathcal {L}_{\mathrm {EDL}} = \mathcal {L}_{\mathrm {CE}} + (\mathcal {L}_{\mathrm {NLL}} + \mathcal {L}_{\mathrm {KL}}), 
(5)
where the auxiliary head is optimized over k + ˆu categories
using LNLL and LKL, while the primary head only focuses
on the known classes via LCE.
3.4. Flexible Information-Purity Sampling
E2OAL performs query selection by jointly balancing two
key factors: class purity and sample informativeness. High
informativeness promotes label efficiency, while high pu-
rity ensures that most queries originate from known classes,
thereby minimizing contamination from unknowns.
A fundamental challenge in OSAL is to identify un-
labeled samples that are most likely to belong to known
classes. We address this by introducing a lightweight logit-
margin purity score computed from the auxiliary head’s
calibrated logits. Under Dirichlet-based calibration, each
logit encodes class-specific evidence: positive values indi-
cate support, with larger magnitudes reflecting higher con-
fidence, while negative values indicate rejection. For an un-
labeled sample x ∈DU, let o(1) and o(2) denote the high-
est logits among known and unknown classes, respectively.
The purity score is defined as:
 \label {e q :pur i ty} S _{\
text  { p uri
ty}}( x) = o_{(1)} - o_{(2)} = \max _{c \in \mathcal {C}_k} o_c - \max _{c \in \mathcal {C}_{\hat {u}}} o_c. 
(6)
A higher Spurity(x) indicates stronger evidence for known-
class membership. Unlike softmax-based confidence mar-
gins, this metric explicitly measures evidence separation,
providing enhanced robustness under open-set conditions.
E2OAL further prioritizes samples that are both likely
from known classes and highly informative. To quantify
informativeness under open-set conditions, we propose a
Jensen–Shannon (JS) divergence–based metric between the
primary head’s output and two reference distributions:
 \label { eq
:
i n f
o
r ma
t
i v eness
} S_{\text {info}}(x) = \mathrm {JS}\big (\mathbf {p} \parallel \mathbf {u}\big ) \cdot \mathrm {JS}\big (\mathbf {p} \parallel \mathbf {p}^{\mathrm {max}}\big ), 
(7)
where p is the predicted probability vector from the pri-
mary head, u is the uniform distribution, and pmax is the
one-hot encoding of the most confident class in p. This for-
mulation emphasizes moderately uncertain samples, which
diverge from both uniform and peaked distributions, thus fa-
voring ambiguous yet informative instances while avoiding
outliers or trivial predictions.
E2OAL integrates the above two metrics into a flexible
two-stage selection scheme calibrated to meet a fixed tar-
get query precision p∗. In stage 1, a high-purity candidate
pool Cpool is constructed by fitting a three-component Gaus-
sian Mixture Model (GMM) [24] to all Spurity(x) values in
DU. Each GMM component corresponds to a distinct purity
regime: high (known-class-like), low (unknown-class-like),
and intermediate (ambiguous). Samples are ranked by their
posterior probability of belonging to the high-purity com-
ponent, and Cpool is expanded until the mean posterior prob-
ability of its bottom-|B| entries matches a calibrated query
precision ˆp∗. Notably, due to systematic estimation bias,
we use a calibrated query precision ˆp∗instead of directly
applying the target precision p∗. To compensate for such
systematic shifts, the calibrated precision ˆp∗is adaptively
updated using the observed precision ¯p∗from the previous
round and the target precision p∗as follows:
 \
hat 
{
p}^
*
_{t+1} 
{ = } \ b egi
n  { cas e
s
} \max
 \
bi g (\min (\hat {p}^*_t + (p^* - \bar {p}_t^*), 0), 1\big ) & \text {if } t {>} 0, \\ p^* & \text {if } t {=} 0. \end {cases} 
(8)
This adaptive mechanism eliminates manual threshold tun-
ing and ensures stable query precision across rounds.
In stage 2, the top-|B| samples from Cpool are selected ac-
cording to Sinfo(x). This two-stage strategy achieves a prin-
cipled balance between uncertainty-driven exploration and
label-purity control, enabling robust and adaptive querying.


<!-- page 6 -->
0
2
4
6
8
10
Number of AL rounds
80
85
90
95
100
Test accuracy
CIFAR-10, Mismatch ratio: 20%
Ours
EAOA
BUAL
EOAL
LfOSA
MQNet
BADGE
Coreset
MSP
Uncertainty
Random
0
2
4
6
8
10
Number of AL rounds
40
50
60
70
80
Test accuracy
CIFAR-100, Mismatch ratio: 20%
0
2
4
6
8
10
Number of AL rounds
25
35
45
55
65
Test accuracy
Tiny-ImageNet, Mismatch ratio: 10%
0
2
4
6
8
10
Number of AL rounds
55
65
75
85
95
Test accuracy
CIFAR-10, Mismatch ratio: 40%
0
2
4
6
8
10
Number of AL rounds
35
45
55
65
Test accuracy
CIFAR-100, Mismatch ratio: 40%
0
2
4
6
8
10
Number of AL rounds
25
35
45
55
Test accuracy
Tiny-ImageNet, Mismatch ratio: 20%
Figure 3. Test accuracy across AL rounds under varying mismatch ratios on CIFAR-10/100 and Tiny-ImageNet.
4. Experiments
Datasets. We evaluate our method E2OAL on three stan-
dard image classification benchmarks:
CIFAR-10 [14],
CIFAR-100 [14], and Tiny-ImageNet [39], which contain
10, 100, and 200 classes, respectively. To simulate open-
set active learning (OSAL) conditions, a subset of classes is
designated as “known”, with the remainder treated as “un-
known”. The proportion of known classes, referred to as the
mismatch ratio, is set to 20%, 30%, and 40% for CIFAR-
10/100, and 10%, 15%, and 20% for Tiny-ImageNet, intro-
ducing varying levels of open-set difficulty.
Training Setup. For each dataset, the labeled pool is ini-
tialized by randomly sampling 1% of known-class instances
for CIFAR-10 and 8% for CIFAR-100 and Tiny-ImageNet.
We perform 10 active learning (AL) rounds, querying 1,500
samples per round. All models adopt a ResNet-50 [9] back-
bone and are trained with SGD (momentum 0.9, weight de-
cay 5 × 10−4, batch size 128) for 200 epochs, starting with
a learning rate of 0.01 decayed by 0.1 every 60 epochs. Ex-
periments are performed on NVIDIA RTX 3090 GPUs, and
all results are averaged over three runs with random seeds
{1, 2, 3}. The target query precision is fixed at p∗= 0.6,
consistent with prior work [41], while the estimated upper
bound of unknown classes is set to ˆumax = 1000.
Compared Methods. We compare E2OAL against rep-
resentative methods from four categories: (1) Random sam-
pling; (2) Informativeness-based approaches, including Un-
certainty [18], Coreset [28], and BADGE [2]; (3) Purity-
based methods, such as MSP [11] and LfOSA [20]; (4)
Hybrid approaches, including MQNet [21], EOAL [27],
BUAL [42], and EAOA [41], which jointly consider infor-
mativeness and purity. All methods follow identical training
and evaluation protocols for fair comparison.
4.1. Performance Comparison
We assess the classification performance and query effec-
tiveness of E2OAL on CIFAR-10, CIFAR-100, and Tiny-
ImageNet. Figure 3 illustrates the progression of test accu-
racy across AL rounds, while Figure 4 presents the relation-
ship between mean query precision and mean test accuracy.
Due to space constraints, results for intermediate mismatch
ratios are provided in Appendix Figures D1 and E1.
As shown in Figures 3 and D1 (Appendix), E2OAL con-
sistently surpasses all baselines across varying mismatch ra-
tios and datasets, maintaining clear advantages throughout
the entire query process. These results demonstrate the ro-
bustness and effectiveness of our framework. Among the
baselines, hybrid approaches such as EAOA, BUAL, and
EOAL outperform single-factor methods by jointly balanc-
ing purity and informativeness. However, their reliance on
separate detectors and limited use of labeled unknowns re-
stricts overall performance. In contrast, MQNet combines
both factors but suffers from inaccurate purity estimation
and also fails to effectively leverage labeled unknowns.
Figures 4 and E1 (Appendix) further show that E2OAL
consistently lies in the top-right region, achieving both
high query precision and superior test accuracy.
Com-
pared with hybrid baselines (e.g., EAOA, BUAL, EOAL),
E2OAL achieves a better balance between precision and in-

[CAPTION] Figure 3. Test accuracy across AL rounds under varying mismatch ratios on CIFAR-10/100 and Tiny-ImageNet.


<!-- page 7 -->
0.15
0.25
0.35
0.45
0.55
Mean query precision
90.0
92.5
95.0
Mean test accuracy
CIFAR-10
Mismatch ratio: 20%
Ours
EAOA
BUAL
EOAL
LfOSA
MQNet
BADGE
Coreset
MSP
Uncertainty
Random
0.15
0.25
0.35
0.45
Mean query precision
52.5
57.5
62.5
67.5
CIFAR-100
Mismatch ratio: 20%
0.10
0.15
0.20
0.25
0.30
Mean query precision
40
45
50
55
Tiny-ImageNet
Mismatch ratio: 10%
0.3
0.4
0.5
0.6
0.7
0.8
Mean query precision
70
75
80
85
Mean test accuracy
CIFAR-10
Mismatch ratio: 40%
0.3
0.4
0.5
0.6
Mean query precision
42.5
45.0
47.5
50.0
52.5
CIFAR-100
Mismatch ratio: 40%
0.15
0.25
0.35
0.45
Mean query precision
30
35
40
45
Tiny-ImageNet
Mismatch ratio: 20%
Figure 4. Mean query precision vs. mean test accuracy across rounds under varying mismatch ratios on CIFAR-10/100 and Tiny-ImageNet.
formativeness, attributed to its adaptive sampling and effec-
tive use of labeled unknowns. Moreover, E2OAL main-
tains query precision close to the target p∗= 0.6, demon-
strating better control than EAOA, which is also guided
by a target precision.
In contrast, informativeness-only
methods (e.g., Uncertainty, Coreset, BADGE) tend to over-
sample unknowns, while purity-only methods (e.g., MSP,
LfOSA) often query redundant known-class samples. By
harmonizing these two objectives, E2OAL enables more ef-
ficient querying and improved generalization.
In addition, Figure E2 (Appendix) further illustrates how
query precision evolves over AL rounds. We observe that
even in the early stages, our method consistently maintains
high query precision, whereas the previous state-of-the-art
method, EAOA, exhibits suboptimal precision and notable
fluctuations. This instability originates from the inherent
limitation of its adaptive strategy, which updates with a
fixed step size and struggles to converge quickly to the opti-
mal value. In contrast, our method introduces no additional
hyperparameters and achieves stable, high-precision query-
ing from the outset, demonstrating clear superiority.
4.2. Ablation Studies
Impact of Each Module. We first examine a variant where
the target classifier is trained independently without using
labeled unknowns, as in prior methods. As shown in Ta-
ble 1 (see Table F1 in Appendix for complete results), our
method still achieves consistent performance gains, particu-
larly on the challenging Tiny-ImageNet, confirming the ef-
fectiveness of the proposed adaptive sampling strategy.
We then separately analyze the contributions of the pu-
Method
CIFAR-10
CIFAR-100
Tiny-ImageNet
Ours*
95.94
67.54
60.44
EAOA
95.88
67.14
57.31
BUAL
95.04
63.73
56.09
EOAL
93.64
63.69
56.13
Table 1. Final-round test accuracy (%) under fixed mismatch ratios
(30% for CIFAR-10/100 and 15% for Tiny-ImageNet). “Ours*”
denotes a variant trained without leveraging labeled unknowns.
Method
CIFAR-10
CIFAR-100
Tiny-ImageNet
LEDL
9495
7934
5844
LCE
9394
7814
5661
Table 2. Purity comparison under the most challenging mismatch
ratios: 20% for CIFAR-10/100 and 10% for Tiny-ImageNet. Sam-
pling by Eq. (6) with CE and EDL losses on the auxiliary head,
evaluated by the total number of queried known-class samples.
rity and informativeness components. As presented in Ta-
ble 2, our Dirichlet-based calibration significantly enhances
known-class discrimination, yielding a larger number of
known samples among queried instances and thereby im-
proving overall purity. For informativeness, Table 3 shows
that our moderately uncertainty-based metric consistently
outperforms its counterpart in EAOA—the previous state of
the art—achieving higher test accuracy and demonstrating
stronger capability in identifying informative samples.
Finally, Table 4 evaluates the contribution of each com-
ponent. Using only or disabling any single module notably


**[Table p7.1]**
| CIFAR-10 Mismatch ratio: 40% |  |
| --- | --- |


**[Table p7.2]**
| CIFAR-100 Mismatch ratio: 40% |  |
| --- | --- |

[CAPTION] Figure 4. Mean query precision vs. mean test accuracy across rounds under varying mismatch ratios on CIFAR-10/100 and Tiny-ImageNet.

[CAPTION] Table 1. Final-round test accuracy (%) under fixed mismatch ratios

[CAPTION] Table 2. Purity comparison under the most challenging mismatch


<!-- page 8 -->
Method
CIFAR-10
CIFAR-100
Tiny-ImageNet
Ours
95.95
65.73
48.63
EAOA
94.90
61.95
44.73
Uncertainty
95.37
61.28
44.73
Table 3. Informativeness comparison under the most challeng-
ing mismatch ratios: 20% for CIFAR-10/100 and 10% for Tiny-
ImageNet. Final test accuracy (%) when retaining only the infor-
mativeness component from “Ours”, “EAOA”, and “Uncertainty”.
Method
CIFAR-10
CIFAR-100
Tiny-ImageNet
Ours
97.52
72.10
64.02
w/o ClassExp
97.17
70.73
62.67
Spurity
96.73
72.00
61.93
Sinfo
96.00
68.20
57.60
Table 4. Final-round test accuracy (%) of different ablation vari-
ants under intermediate mismatch ratios on CIFAR-10/100 and
Tiny-ImageNet. “Sinfo” and “Spurity” denote sampling based solely
on informativeness or purity. “w/o ClassExp” disables class ex-
pansion by collapsing all labeled unknowns into a single class.
Mismatch ratio
p∗= 0.4
0.5
0.6
0.7
20%
81.47
81.67
82.20
82.18
30%
71.90
72.96
72.10
73.17
40%
66.68
68.03
67.98
66.78
Table 5. Final-round test accuracy (%) on CIFAR-100 under vary-
ing target query precisions and mismatch ratios.
Uncertainty
Coreset
MSP
Random
Ours
LfOSA
EAOA
BUAL
EOAL
BADGE
MQNet
0
5
10
15
20
Total training time (hours)
0.303 0.321 0.384 0.380
0.602
0.528 0.576
0.437 0.602 0.378
0.328
Figure 5. Total training time (hours) on CIFAR-100 under a 40%
mismatch ratio. Bars indicate actual training time with the average
query precision annotated; the dashed line shows the approximate
projection assuming a linear relationship between query precision
and time, aligned to the precision level of “Uncertainty”.
reduces performance across datasets, confirming that each
contributes complementarily to the overall framework.
Sensitivity to Target Precision p∗. In the main experi-
ments, the target query precision p∗is fixed at 0.6, following
EAOA. To examine the sensitivity of E2OAL to this param-
eter, we further evaluate it under p∗∈{0.4, 0.5, 0.6, 0.7}.
As shown in Table 5, although the optimal value may
vary slightly with the mismatch ratio, E2OAL consistently
achieves strong performance across all settings, demonstrat-
ing its robustness and adaptability.
Training Efficiency.
Figure 5 presents a comparison
of the overall and equivalent training times across different
methods. E2OAL achieves high training efficiency, with its
equivalent time comparable to lightweight baselines such
as Random, MSP, Coreset, and Uncertainty. By removing
the need for separate detectors and effectively utilizing la-
beled unknowns, E2OAL attains superior accuracy with
only marginal additional cost.
Ablation on ˆu Estimation. Figure H1 (Appendix) vi-
sualizes how the inferred number of unknown classes ˆu
evolves across rounds. Even without access to class priors,
our adaptive class estimation module exhibits stable conver-
gence and produces estimates that remain within the correct
order of magnitude under varying mismatch ratios.
Ablation on Pretrained Representations.
Table G1
(Appendix) reports the final-round performance when the
default CLIP-extracted features in Algorithm 1 are replaced
with representations from a pretrained MoCo [10] back-
bone. The results show that model accuracy remains largely
unchanged, indicating that E2OAL is robust to the choice
of pretrained representation. This consistency suggests that
the effectiveness of our adaptive estimation strategy primar-
ily arises from the proposed algorithmic design rather than
dependence on a specific feature backbone.
In practice,
CLIP can be substituted with any pretrained model capable
of providing high-quality, task-agnostic representations.
5. Conclusion
In this paper, we presented E2OAL, an effective and effi-
cient framework for open-set active learning (OSAL) that
reduces wasted annotation cost while significantly improv-
ing model performance under open-set conditions. Unlike
prior approaches that depend on separate detectors or ne-
glect the value of labeled unknowns, E2OAL leverages
them through a two-stage pipeline: (1) an adaptive class es-
timation and utilization module that uncovers latent struc-
tures of labeled unknowns in a frozen semantic space
and leverages them to enhance known-class discrimination;
and (2) a flexible query strategy that jointly balances infor-
mativeness and purity through Dirichlet-based calibration,
logit-margin scoring, and a mild uncertainty metric tailored
for open-set scenarios. Extensive experiments across mul-
tiple OSAL benchmarks demonstrate that E2OAL consis-
tently surpasses prior methods in both accuracy and query
effectiveness, while requiring fewer hyperparameters and
lower training overhead. Our analysis further highlights the
overlooked potential of labeled unknowns as a valuable
source of supervisory signal for open-world learning.


**[Table p8.1]**
|  | 0.602 |  |
| --- | --- | --- |
|  |  |  |
|  |  |  |

[CAPTION] Table 3. Informativeness comparison under the most challeng-

[CAPTION] Table 4. Final-round test accuracy (%) of different ablation vari-

[CAPTION] Table 5. Final-round test accuracy (%) on CIFAR-100 under vary-

[CAPTION] Figure 5. Total training time (hours) on CIFAR-100 under a 40%

[CAPTION] Figure 5 presents a comparison


<!-- page 9 -->
Acknowledgements
This work was supported in part by the NSFC (U2441285)
and in part by the Interdisciplinary Innovation Fund for
Doctoral Students of Nanjing University of Aeronautics and
Astronautics.
References
[1] Sharat Agarwal, Himanshu Arora, Saket Anand, and Chetan
Arora. Contextual diversity for active learning. In European
Conference on Computer Vision, pages 137–153. Springer,
2020. 1
[2] Jordan T Ash, Chicheng Zhang, Akshay Krishnamurthy,
John Langford, and Alekh Agarwal.
Deep batch active
learning by diverse, uncertain gradient lower bounds. arXiv
preprint arXiv:1906.03671, 2019. 2, 6, 5
[3] Maria-Florina Balcan, Andrei Broder, and Tong Zhang. Mar-
gin based active learning. In International Conference on
Computational Learning Theory, pages 35–50. Springer,
2007. 2
[4] Samuel Budd, Emma C Robinson, and Bernhard Kainz. A
survey on active learning and human-in-the-loop deep learn-
ing for medical image analysis. Medical image analysis, 71:
102062, 2021. 1
[5] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Ge-
offrey Hinton. A simple framework for contrastive learning
of visual representations. In International conference on ma-
chine learning, pages 1597–1607. PmLR, 2020. 3
[6] Pan Du, Suyun Zhao, Hui Chen, Shuwen Chai, Hong Chen,
and Cuiping Li.
Contrastive coding for active learning
under class distribution mismatch.
In Proceedings of the
IEEE/CVF International Conference on Computer Vision,
pages 8927–8936, 2021. 1
[7] Yarin Gal, Riashat Islam, and Zoubin Ghahramani. Deep
bayesian active learning with image data. In International
conference on machine learning, pages 1183–1192. PMLR,
2017. 2
[8] Julia Grabinski, Paul Gavrikov, Janis Keuper, and Margret
Keuper. Robust models are less over-confident. Advances
in neural information processing systems, 35:39059–39075,
2022. 4
[9] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
Deep residual learning for image recognition. In Proceed-
ings of the IEEE conference on computer vision and pattern
recognition, pages 770–778, 2016. 1, 6
[10] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross
Girshick. Momentum contrast for unsupervised visual rep-
resentation learning. In Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, pages
9729–9738, 2020. 3, 8
[11] Dan Hendrycks and Kevin Gimpel. A baseline for detect-
ing misclassified and out-of-distribution examples in neural
networks. arXiv preprint arXiv:1610.02136, 2016. 6, 1, 5
[12] Alex Holub, Pietro Perona, and Michael C Burl. Entropy-
based active learning for object recognition. In 2008 IEEE
Computer Society Conference on Computer Vision and Pat-
tern Recognition Workshops, pages 1–8. IEEE, 2008. 2
[13] Sheng-Jun Huang, Rong Jin, and Zhi-Hua Zhou.
Active
learning by querying informative and representative exam-
ples.
Advances in neural information processing systems,
23, 2010. 1
[14] A Krizhevsky. Learning multiple layers of features from tiny
images. Master’s thesis, University of Tront, 2009. 6
[15] Harold W Kuhn. The hungarian method for the assignment
problem. Naval research logistics quarterly, 2(1-2):83–97,
1955. 4
[16] Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep
learning. nature, 521(7553):436–444, 2015. 1
[17] Changbin Li, Kangshuo Li, Yuzhe Ou, Lance M Kaplan,
Audun Jøsang, Jin-Hee Cho, Dong Hyun Jeong, and Feng
Chen. Hyper evidential deep learning to quantify composite
classification uncertainty. arXiv preprint arXiv:2404.10980,
2024. 5
[18] Mingkun Li and Ishwar K Sethi. Confidence-based active
learning. IEEE transactions on pattern analysis and machine
intelligence, 28(8):1251–1261, 2006. 2, 6, 1, 5
[19] Hieu T Nguyen and Arnold Smeulders. Active learning us-
ing pre-clustering. In Proceedings of the twenty-first inter-
national conference on Machine learning, page 79, 2004. 2
[20] Kun-Peng Ning, Xun Zhao, Yu Li, and Sheng-Jun Huang.
Active learning for open-set annotation. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 41–49, 2022. 1, 2, 6, 5
[21] Dongmin Park, Yooju Shin, Jihwan Bang, Youngjun Lee,
Hwanjun Song, and Jae-Gil Lee. Meta-query-net: Resolv-
ing purity-informativeness dilemma in open-set active learn-
ing. Advances in Neural Information Processing Systems,
35:31416–31429, 2022. 1, 2, 6, 5
[22] Fengchao Peng, Chao Wang, Jianzhuang Liu, and Zhen
Yang. Active learning for lane detection: A knowledge dis-
tillation approach. In Proceedings of the IEEE/CVF Interna-
tional Conference on Computer Vision, pages 15152–15161,
2021. 1
[23] Jon Perez-Cerrolaza, Jaume Abella, Markus Borg, Carlo
Donzella, Jes´us Cerquides, Francisco J Cazorla, Cristofer
Englund, Markus Tauber, George Nikolakopoulos, and
Jose Luis Flores.
Artificial intelligence for safety-critical
systems in industrial and transportation domains: A survey.
ACM Computing Surveys, 56(7):1–40, 2024. 1
[24] Haim Permuter, Joseph Francos, and Ian Jermyn. A study
of gaussian mixture models of color and texture features for
image classification and segmentation. Pattern recognition,
39(4):695–706, 2006. 5, 2
[25] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning
transferable visual models from natural language supervi-
sion. In International conference on machine learning, pages
8748–8763. PmLR, 2021. 1, 3, 4
[26] Pengzhen Ren, Yun Xiao, Xiaojun Chang, Po-Yao Huang,
Zhihui Li, Brij B Gupta, Xiaojiang Chen, and Xin Wang.
A survey of deep active learning. ACM computing surveys
(CSUR), 54(9):1–40, 2021. 1, 2
[27] Bardia Safaei, VS Vibashan, Celso M de Melo, and Vishal M
Patel. Entropic open-set active learning. In Proceedings of


<!-- page 10 -->
the AAAI conference on artificial intelligence, pages 4686–
4694, 2024. 1, 2, 6, 5
[28] Ozan Sener and Silvio Savarese. Active learning for convolu-
tional neural networks: A core-set approach. arXiv preprint
arXiv:1708.00489, 2017. 1, 2, 6, 5
[29] Murat Sensoy, Lance Kaplan, and Melih Kandemir. Eviden-
tial deep learning to quantify classification uncertainty. Ad-
vances in neural information processing systems, 31, 2018.
4, 5
[30] Burr Settles. Active learning literature survey. Computer
Sciences Technical Report 1648, University of Wisconsin–
Madison, 2009. 1, 2
[31] Manali Sharma and Mustafa Bilgic. Evidence-based uncer-
tainty sampling for active learning. Data Mining and Knowl-
edge Discovery, 31(1):164–202, 2017. 1
[32] Shuzhou Sun, Shuaifeng Zhi, Janne Heikkil¨a, and Li Liu.
Evidential uncertainty and diversity guided active learning
for scene graph generation. In 11th International Conference
on Learning Representations, ICLR 2023. OpenReview. net,
2023. 1
[33] Jiao Tang, Yagao Yue, Peng Wan, Mingliang Wang, Dao-
qiang Zhang, and Wei Shao. Osal-nd: Open-set active learn-
ing for nucleus detection. In International Conference on
Medical Image Computing and Computer-Assisted Interven-
tion, pages 351–361. Springer, 2024. 1
[34] Sagar Vaze, Kai Han, Andrea Vedaldi, and Andrew Zisser-
man.
Generalized category discovery.
In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 7492–7501, 2022. 4
[35] Paulo Rafael Soares Oliveira Vieira.
Open-world active
learning in self-driving cars. Master’s thesis, Universidade
de Coimbra (Portugal), 2022. 1
[36] Yichen Xie, Han Lu, Junchi Yan, Xiaokang Yang, Masayoshi
Tomizuka, and Wei Zhan. Active finetuning: Exploiting an-
notation budget in the pretraining-finetuning paradigm. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition, pages 23715–23724, 2023. 1
[37] Yazhou Yang and Marco Loog. Active learning using uncer-
tainty information. In 2016 23rd International Conference
on Pattern Recognition (ICPR), pages 2646–2651. IEEE,
2016. 1
[38] Yang Yang, Yuxuan Zhang, Xin Song, and Yi Xu. Not all
out-of-distribution data are harmful to open-set active learn-
ing. Advances in Neural Information Processing Systems,
36:13802–13818, 2023. 1
[39] Leon Yao and John Miller. Tiny imagenet classification with
convolutional neural networks. CS 231N, 2(5):8, 2015. 6
[40] Donggeun Yoo and In So Kweon.
Learning loss for ac-
tive learning. In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pages 93–102,
2019. 1
[41] Chen-Chen Zong and Sheng-Jun Huang. Rethinking epis-
temic and aleatoric uncertainty for active open-set annota-
tion: An energy-based approach.
In Proceedings of the
Computer Vision and Pattern Recognition Conference, pages
10153–10162, 2025. 1, 2, 6, 5
[42] Chen-Chen Zong, Ye-Wen Wang, Kun-Peng Ning, Hai-Bo
Ye, and Sheng-Jun Huang. Bidirectional uncertainty-based
active learning for open-set annotation. In European Confer-
ence on Computer Vision, pages 127–143. Springer, 2024. 1,
2, 6, 5
[43] Chen-Chen Zong, Ye-Wen Wang, Ming-Kun Xie, and
Sheng-Jun Huang. Dirichlet-based prediction calibration for
learning with noisy labels. In Proceedings of the AAAI Con-
ference on Artificial Intelligence, pages 17254–17262, 2024.
4, 5


<!-- page 11 -->
Revisiting Unknowns: Towards Effective and Efficient Open-Set Active Learning
Supplementary Material
A. Full Expression of Equation (4)
The KL divergence term LKL between the predicted Dirich-
let distribution and the uniform prior can be derived as:
  \ b eg
i
n {alig n ed} &\
m
a
t
hcal { L}_ {\m athrm {KL
}} = \ ma th
r
m  
{
K L}\b
ig (
\
mat
hrm {D
i
r
}(\
t
ilde
 {\bo
ldsy
m
bol
 {\alp
h
a
 }
} ) \ para
l lel 
\
m
a thrm
 {Di
r
}(\
mathbf
 
{1
}
) \
big
 ) \
\
 &=
 \int 
\
m a
t
h rm {
Dir}
(
\bo
ldsymb
o
l
 {
p }; \til
d e {\ b oldsymbol 
{
\al pha }
}) \log \
f
r
a c { \mat
h rm { D
i
r
}(\
bold s ym bol {p}; \t
ilde  {\b olds ymb
o l {
\
a
l
pha }
)}{ \ma
t
hrm  {Di r}(\b
old s ymbol
 
{
p
}; \
m
ath
bf { 1 })
}
 d\bol d s
ymbol
 
{p}
 \\
 &
= \int \left (\frac {1}{\mathbb {B}\left ( \boldsymbol {\tilde {\alpha }} \right ) }{ \prod _{i=1}^{k+\hat {u}}} {p}_{i}^{\tilde {\alpha }_{i}-1}\right ) \log \left (\frac {\mathbb {B}(\boldsymbol {1})}{\mathbb {B}\left (\boldsymbol {\tilde {\alpha }}\right )} \prod _{i=1}^{k+\hat {u}} {p}_{i}^{\tilde {\alpha }_{i}-1}\right ) d \boldsymbol {p}\\ &= \log \frac {\mathbb {B}(\boldsymbol {1})}{\mathbb {B}\left (\boldsymbol {\tilde {\alpha }}\right )} \int \frac {1}{\mathbb {B}\left (\boldsymbol {\tilde {\alpha }}\right )} \prod _{i=1}^{k+\hat {u}} p_{i}^{\tilde {\alpha }_{ i}-1} d \boldsymbol {p} \\&+ \int \left (\log \prod _{i=1}^{k+\hat {u}} p_{i}^{\tilde {\alpha }_{i}-1}\right )\left (\frac {1}{\mathbb {B}\left (\tilde {\alpha }\right )} \prod _{i=1}^{k+\hat {u}} p_{i}^{\tilde {\alpha }_{i}-1}\right ) d \boldsymbol {p} \\ &= \log \frac {\mathbb {B}(\boldsymbol {1})}{\mathbb {B}\left (\boldsymbol {\tilde {\alpha }}\right )}+\mathbb {E}_{\boldsymbol {p} \sim \mathrm {Dir}(\boldsymbol {\tilde {\alpha }})} \left [\log \textstyle \prod _{i=1}^{k+\hat {u}} p_{i}^{\tilde {\alpha }_{i}-1}\right ]\\ &= \log \frac {\mathbb {B}(\boldsymbol {1})}{\mathbb {B}\left (\boldsymbol {\tilde {\alpha }}\right )}+\sum _{j=1}^C\left (\tilde {\alpha }_{ j}-1\right ) \mathbb {E}_{\boldsymbol {p}_j \sim {\mathcal {B} }\left (\tilde {\alpha }_{j}, {\sum _{i \ne j}} \tilde {\alpha } _{i}\right )} \left [\log p_{j} \right ] \\ & = \log \left [\frac {\Gamma \left (\sum _{i=1}^{k+\hat {u}} \tilde {\alpha }_{i}\right )}{\Gamma ({k+\hat {u}}) \prod _{i=1}^{k+\hat {u}} \Gamma \left (\tilde {\alpha }_{ i}\right )}\right ]\\&+\sum _{j=1}^{k+\hat {u}}\left (\tilde {\alpha }_{ j}-1\right )\left [\psi \left (\tilde {\alpha }_{j}\right )-\psi \left (\sum _{i=1}^{k+\hat {u}} \tilde {\alpha }_{i}\right )\right ], \end {aligned} 
(9)
where B(·) denotes the multivariate Beta function, B(·, ·)
is the standard Beta function, Γ(·) represents the Gamma
function, and ψ(·) denotes the digamma function. The ex-
plicit definitions of B(·) and B(·, ·) are given by:
  \ma t
hbb {
B}( \tilde
 
{\bold
sym bol
 {\alpha }}) = \frac { \prod _{i=1}^{k+\hat {u}} \Gamma (\tilde {\alpha }_i)}{\Gamma \left ( \sum _{i=1}^{k+\hat {u}} \tilde {\alpha }_i \right )}, 
(10)
and
 
 
{\mat
h
cal 
{B}
 
} \
l eft ( \
ti
lde {\a
l
p
h
a } _ {
j}, {\s
u m _{i \ne j}} \tilde {\alpha } _{i}\right )=\frac {\Gamma \left (\tilde {\alpha }_{j} \right ) \Gamma \left ({\sum _{i \ne j}} \tilde {\alpha } _{i} \right )}{\Gamma \left ( \tilde {\alpha }_{j}+ {\sum _{i \ne j}} \tilde {\alpha } _{i}\right ) } . 
(11)
B. The Pseudocode of E²OAL
We outline the full workflow of E2OAL in Algorithm B1,
which covers each stage of the active learning process.
C. Additional Results for Figure 1
We present additional results over a wider range of mis-
match ratios and network architectures beyond those in Fig-
ure 1. As illustrated in Figure C1, we evaluate three sam-
pling strategies to assess whether labeled unknowns can
enhance known-class learning: (1) random sampling; (2)
certainty-based sampling via maximum softmax probabil-
ity (MSP [11]); and (3) uncertainty-based sampling using
the least-confidence criterion [18]. Our observations indi-
cate that, regardless of sampling strategy, mismatch ratio,
or network architecture, leveraging fine-grained labels of
labeled unknowns within the auxiliary classifier consis-
tently yields the best performance—especially as network
capacity increases. Treating all labeled unknowns as a sin-
gle class typically improves over ignoring them but results
in less stable and less substantial gains, occasionally even
degrading performance. These findings highlight the signif-
icant benefits of effectively exploiting labeled unknowns
during training, motivating the in-depth study and method
proposed in this work.
D. Additional Results for Figure 3
Figure D1 illustrates the evolution of test accuracy across
rounds under intermediate mismatch ratios on three bench-
mark datasets. E2OAL consistently surpasses all baselines,
exhibiting a clearly superior accuracy trajectory through-
out the process. Notably, the performance gap widens as
the dataset complexity increases, highlighting the robust-
ness and scalability of the proposed framework.
E. Additional Results for Figure 4
Figure E1 reports the mean test accuracy and average query
precision across rounds under moderate mismatch ratios
on three benchmark datasets. E2OAL achieves the highest
query precision on CIFAR-100 and Tiny-ImageNet, while
consistently outperforming all competing methods in accu-
racy. On CIFAR-10, the query precision is slightly lower
than EOAL [27], which is expected since our purity con-
trol is explicitly regulated by a target precision of 0.6. This
observation underscores two key insights: (1) Simply maxi-
mizing query purity is not optimal for open-set active learn-
ing—although EOAL attains high precision, its queried
samples, albeit from known classes, tend to be less informa-
tive, leading to suboptimal accuracy; (2) Although E2OAL
does not achieve the absolute highest query precision, it
adheres more closely to the target value than EAOA [41],
which is also guided by the target precision, demonstrating


<!-- page 12 -->
Algorithm B1 The E2OAL algorithm
Input: Labeled pool DL = Dkno
L
∪Dunk
L
, unlabeled pool DU, known class count k, upper limit ˆumax, target classifier fθ,
query budget |B|, and target query precision p∗.
Process: (The t-th active learning round)
1: if t = 1 then
2:
# Model training
3:
Train classifier fθ using LCE for primary head (k-way) and LEDL for auxiliary head (k-way)
4:
# Purity-based candidate selection
5:
Obtain auxiliary head’s outputs for all x ∈DL ∪DU
6:
for each x ∈DL ∪DU do
7:
Compute purity margin: Spurity(x) = maxc∈Ck oc
8:
end for
9: else
10:
# Unknown class estimation
11:
Estimate ˆu based on labeled pool DL, known class count k, and upper bound ˆumax using Algorithm 1
12:
# Model training
13:
Train classifier fθ using LCE for primary head (k-way) and LEDL for auxiliary head ((k + ˆu)-way)
14:
# Purity-based candidate selection
15:
Obtain auxiliary head’s outputs for all x ∈DL ∪DU.
16:
for each x ∈DL ∪DU do
17:
Compute purity margin: Spurity(x) = maxc∈Ck oc −maxc∈Cˆu oc
18:
end for
19: end if
20: Fit a 3-component Gaussian Mixture Model (GMM) [24] on logit margins {Spurity(x)} to model different purity regimes:
high (known), low (unknown), and intermediate (ambiguous)
21: For each x ∈DU, compute its likelihood under the high-purity component
22: Sort DU in descending order of the computed likelihoods
23: Initialize the candidate pool Cpool with the top |B| samples from the sorted DU
24: # Precision-based candidate refinement
25: Compute the calibrated target query precision ˆp∗
t =
(
max
 min(ˆp∗
t−1 + (p∗−¯p∗
t−1), 0), 1
 
if t > 1
p∗
if t = 1
26: while the mean likelihood of the lowest |B| samples in Cpool is greater than ˆp∗
t do
27:
Add the next highest-likelihood sample from DU into Cpool
28: end while
29: # Information-based final query selection
30: for each x ∈Cpool do
31:
Obtain predicted probability vector p of x from the primary head
32:
Let u be a uniform distribution over all classes
33:
Let pmax be a one-hot vector with 1 at arg max(p)
34:
Compute information score: Sinfo(x) = JS
 p ∥u
 
· JS
 p ∥pmax 
35: end for
36: Select top |B| samples from Cpool with the highest Sinfo(x) scores as the query set B
37: Compute observed query precision: ¯p∗
t = |Bkno|
|B|
38: Update data pools: DU ←DU \ B, DL ←DL ∪B
39: return DL, DU, ¯p∗
t , and fθ for the next round
superior control and stability in purity regulation.
Figure E2 further illustrates how query precision evolves
over rounds. It can be observed that our method maintains
a high query precision even in the early training stages—a
desirable property, as highlighted in [21], where high-purity
samples tend to provide greater utility when the model is
still undertrained. This advantage stems from our proposed
purity metric and flexible sampling strategy, which effec-
tively constructs high-purity candidate sets.
In contrast,
the previous state-of-the-art method, EAOA, shows subop-


<!-- page 13 -->
0
2
4
6
8
10
Number of AL rounds
50
60
70
80
Test accuracy
Mean accuracy
Certainty-based sampling
Cknown
Cknown + 1
Call
0
2
4
6
8
10
Number of AL rounds
50
60
70
80
Test accuracy
Mean accuracy
Uncertainty-based sampling
0
2
4
6
8
10
Number of AL rounds
50
60
70
80
Test accuracy
Mean accuracy
Random sampling
(a) ResNet-18, 20 known / 80 unknown classes
0
2
4
6
8
10
Number of AL rounds
40
50
60
70
Test accuracy
Mean accuracy
Certainty-based sampling
Cknown
Cknown + 1
Call
0
2
4
6
8
10
Number of AL rounds
40
50
60
70
Test accuracy
Mean accuracy
Uncertainty-based sampling
0
2
4
6
8
10
Number of AL rounds
40
50
60
70
Test accuracy
Mean accuracy
Random sampling
(b) ResNet-18, 30 known / 70 unknown classes
0
2
4
6
8
10
Number of AL rounds
40
60
Test accuracy
Mean accuracy
Certainty-based sampling
Cknown
Cknown + 1
Call
0
2
4
6
8
10
Number of AL rounds
40
60
Test accuracy
Mean accuracy
Uncertainty-based sampling
0
2
4
6
8
10
Number of AL rounds
40
60
Test accuracy
Mean accuracy
Random sampling
(c) ResNet-34, 30 known / 70 unknown classes
0
2
4
6
8
10
Number of AL rounds
40
60
Test accuracy
Mean accuracy
Certainty-based sampling
Cknown
Cknown + 1
Call
0
2
4
6
8
10
Number of AL rounds
40
60
Test accuracy
Mean accuracy
Uncertainty-based sampling
0
2
4
6
8
10
Number of AL rounds
40
60
Test accuracy
Mean accuracy
Random sampling
(d) ResNet-50, 30 known / 70 unknown classes
Figure C1. Per-round and mean test accuracy on CIFAR-100 under varying class splits and network architectures. Cknown excludes labeled
unknowns, Cknown+1 collapses them into a single class, and Call leverages their ground-truth labels.
0
2
4
6
8
10
Number of AL rounds
65
75
85
95
Test accuracy
CIFAR-10, Mismatch ratio: 30%
Ours
EAOA
BUAL
EOAL
LfOSA
MQNet
BADGE
Coreset
MSP
Uncertainty
Random
0
2
4
6
8
10
Number of AL rounds
40
50
60
70
Test accuracy
CIFAR-100, Mismatch ratio: 30%
0
2
4
6
8
10
Number of AL rounds
30
40
50
60
Test accuracy
Tiny-ImageNet, Mismatch ratio: 15%
Figure D1. Test accuracy across rounds under mismatch ratio 30% on CIFAR-10/100 and 15% on Tiny-ImageNet.


**[Table p13.1]**
| 80 Certainty-based sampling accuracy Cknown 70 Cknown + 1 60 Call Mean accuracy Test 50 80 0 2 4 6 8 10 UnNcuermtabinetry o-bf aAseLd rsoaumnpdlsing accuracy 70 60 Mean accuracy Test 50 0 2 4 6 8 10 80 RaNnduommb esra mofp AlinLg rounds accuracy 70 60 Mean accuracy Test 50 0 2 4 6 8 10 Number of AL rounds | 70 Certainty-based sampling accuracy Cknown 60 Cknown + 1 50 Call Mean accuracy Test 40 0 2 4 6 8 10 70 UnNcuermtabinetry o-bf aAseLd rsoaumnpdlsing accuracy 60 50 Mean accuracy Test 40 0 2 4 6 8 10 70 RaNnduommb esra mofp AlinLg rounds accuracy 60 50 Mean accuracy Test 40 0 2 4 6 8 10 Number of AL rounds |
| --- | --- |


**[Table p13.2]**
| Certainty-based sampling accuracy Cknown 60 Cknown + 1 Call Mean accuracy Test 40 0 2 4 6 8 10 UnNcuermtabinetry o-bf aAseLd rsoaumnpdlsing accuracy 60 Mean accuracy Test 40 0 2 4 6 8 10 RaNnduommb esra mofp AlinLg rounds accuracy 60 Mean accuracy Test 40 0 2 4 6 8 10 Number of AL rounds | Certainty-based sampling accuracy 60 Cknown Cknown + 1 Call Mean accuracy Test 40 0 2 4 6 8 10 UnNcuermtabinetry o-bf aAseLd rsoaumnpdlsing accuracy 60 Mean accuracy Test 40 0 2 4 6 8 10 RaNnduommb esra mofp AlinLg rounds accuracy 60 Mean accuracy Test 40 0 2 4 6 8 10 Number of AL rounds |
| --- | --- |


<!-- page 14 -->
0.2
0.3
0.4
0.5
0.6
0.7
Mean query precision
81
86
91
Mean test accuracy
CIFAR-10
Mismatch ratio: 30%
Ours
EAOA
BUAL
EOAL
LfOSA
MQNet
BADGE
Coreset
MSP
Uncertainty
Random
0.25
0.35
0.45
0.55
Mean query precision
42.5
47.5
52.5
57.5
Mean test accuracy
CIFAR-100
Mismatch ratio: 30%
0.1
0.2
0.3
0.4
Mean query precision
35
40
45
50
Mean test accuracy
Tiny-ImageNet
Mismatch ratio: 15%
Figure E1. Mean query precision and test accuracy under mismatch ratio 30% on CIFAR-10/100 and 15% on Tiny-ImageNet.
0
2
4
6
8
10
Number of AL rounds
0.2
0.4
0.6
0.8
Query precision
Ours
EAOA
BUAL
EOAL
LfOSA
MQNet
BADGE
Coreset
MSP
Uncertainty
Random
0
2
4
6
8
10
Number of AL rounds
0.0
0.2
0.4
0.6
0.8
Query precision
0
2
4
6
8
10
Number of AL rounds
0.2
0.4
0.6
0.8
Query precision
0
2
4
6
8
10
Number of AL rounds
0.2
0.4
0.6
Query precision
0
2
4
6
8
10
Number of AL rounds
0.2
0.4
0.6
0.8
Query precision
0
2
4
6
8
10
Number of AL rounds
0.3
0.4
0.5
0.6
0.7
Query precision
0
2
4
6
8
10
Number of AL rounds
0.1
0.2
0.3
0.4
Query precision
Ours
EAOA
BUAL
EOAL
LfOSA
BADGE
Coreset
MSP
Uncertainty
Random
0
2
4
6
8
10
Number of AL rounds
0.1
0.2
0.3
0.4
0.5
Query precision
0
2
4
6
8
10
Number of AL rounds
0.2
0.3
0.4
0.5
0.6
Query precision
Figure E2. Cross-round query precision under varying mismatch ratios on three benchmarks. From top to bottom: results on CIFAR-10,
CIFAR-100, and Tiny-ImageNet; from left to right: mismatch ratios of 20%, 30%, and 40%, respectively.
timal query precision in the early rounds and exhibits sig-
nificant fluctuations, especially on CIFAR-100 with a 20%
mismatch ratio. This instability arises from its inherent lim-
itation in the adaptive sampling strategy, where the adjust-
ment of the parameter k is limited to a fixed step size per
round, hindering rapid convergence to an optimal value.
Our method, by comparison, introduces no additional hy-
perparameters and achieves stable, high-precision queries
from the outset, demonstrating clear superiority.
F. Additional Results for Table 1
Table F1 summarizes the final-round test accuracy of
all methods under different mismatch ratios across three


**[Table p14.1]**
| CIFAR-10 Mismatch ratio: 30% |  |
| --- | --- |


<!-- page 15 -->
Dataset
CIFAR-10
CIFAR-100
Tiny-ImageNet
Mismatch ratio
20%
30%
40%
20%
30%
40%
10%
15%
20%
Random
94.48
91.11
87.18
64.45
58.48
55.48
47.93
46.00
43.32
Uncertainty [18]
95.70
89.77
83.61
62.25
53.52
50.83
45.83
43.40
35.43
Coreset [28]
94.20
89.56
86.38
63.53
56.62
55.00
50.60
47.33
45.35
BADGE [2]
94.95
90.91
87.12
64.00
56.49
50.20
49.70
48.16
46.23
MSP [11]
94.15
91.51
87.21
65.33
58.69
56.68
51.43
47.78
46.57
LfOSA [20]
94.15
90.91
87.43
70.32
62.49
58.49
58.37
54.78
51.33
MQNet [21]
95.12
89.39
87.42
63.70
53.52
55.44
-
-
-
EOAL [27]
96.23
93.64
91.63
73.73
63.69
59.55
61.40
56.13
52.65
BUAL [42]
96.48
95.04
92.52
73.43
63.73
59.89
63.80
56.09
50.52
EAOA [41]
97.23
95.88
93.09
74.60
67.14
63.49
62.33
57.31
53.33
Ours*
97.33
95.94
93.13
75.90
67.54
63.85
64.23
60.44
54.73
↑over best baseline (%)
0.10
0.06
0.04
1.30
0.40
0.36
1.15
3.13
1.40
Ours
98.77
97.52
95.69
82.20
72.10
67.98
68.53
64.02
57.10
↑over best baseline (%)
1.44
1.64
2.60
7.60
4.96
4.49
4.73
6.71
3.77
Table F1. Final-round test accuracy (%) of all methods under varying mismatch ratios on CIFAR-10, CIFAR-100, and Tiny-ImageNet.
“Ours*” denotes a variant of our method where the target classifier is trained independently without leveraging labeled unknowns. The
best result in each setting is highlighted in bold, while the second best is underlined. Due to the poor performance and high training cost
of MQNet, we do not include it on Tiny-ImageNet, and thus mark it with “–”.
benchmark datasets.
We report results for both “Ours”
(the proposed E2OAL) and “Ours*” (a variant where the
target classifier is trained independently without utilizing
labeled unknowns, similar to prior baselines).
Both
E2OAL and its variant consistently outperform existing
methods by a notable margin.
The strong performance
of “Ours*” demonstrates the effectiveness of our adaptive
sampling strategy in selecting more informative known-
class samples, while the additional improvement achieved
by E2OAL further highlights the value of leveraging labeled
unknowns to enhance known-class learning.
G. Ablation on CLIP Representations
Feature source
CIFAR-10
CIFAR-100
Tiny-ImageNet
CLIP
97.52
72.10
64.02
MoCo
97.44
72.31
63.87
Table G1. Final-round test accuracy (%) under fixed mismatch
ratios (30% for CIFAR-10/100 and 15% for Tiny-ImageNet).
“CLIP” and “MoCo” respectively refer to adaptive class estima-
tion performed using features extracted from a CLIP model and
from a self-supervised MoCo pretrained model.
Table G1 presents the final-round results when the de-
fault CLIP features are replaced with self-supervised MoCo
pretrained representations for adaptive class estimation.
E2OAL exhibits stable performance across all datasets, with
only marginal differences between the two feature sources.
These results demonstrate that E2OAL is robust to the
choice of pretrained representation, and its effectiveness
does not depend on a specific feature extractor. In practice,
CLIP can be substituted with any pretrained model capable
of providing high-quality, task-agnostic features.
H. Ablation on ˆu Estimation
Figure H1 illustrates the estimated total number of classes,
k + ˆu, across rounds on CIFAR-100 under mismatch ra-
tios of 20%, 30%, and 40%, where k denotes the number of
known classes (20, 30, and 40, respectively). For reference,
the figure also includes the final number of queried samples
per class. Our adaptive class estimation module consistently
yields stable and reliable estimates across rounds and mis-
match settings. Since the true class prior of unknown sam-
ples is inaccessible, the estimated total number of classes
may not exactly match the ground truth.
This deviation
partly arises from the inherent ambiguity of class granular-
ity—for instance, CIFAR-100 can be organized into either
20 coarse-grained or 100 fine-grained categories. Never-
theless, our method consistently provides estimates within
the correct order of magnitude, with accuracy improving as
the mismatch ratio increases (i.e., as the open-set problem
becomes less challenging). In particular, under the 40% set-
ting, the estimated class count closely fluctuates around the


<!-- page 16 -->
20
40
60
80
100
Class Index
0
100
200
300
400
Sample count
20
40
60
80
100
Class Index
0
100
200
300
400
Sample count
20
40
60
80
100
Class Index
0
100
200
300
400
Sample count
0
20
40
60
80
100
120
Estimated class count
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
Number of AL rounds
0
20
40
60
80
100
120
140
Estimated class count
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
Number of AL rounds
0
20
40
60
80
100
120
Estimated class count
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
Number of AL rounds
Figure H1. Ablation results for unknown class estimation on CIFAR-100 under mismatch ratios of 20%, 30%, and 40% (from left to right).
Bar charts (bottom x-axis and left y-axis) show the total number of samples labeled per class in the final round. Green bars represent
known classes, and blue bars represent unknown classes. Line plots (top x-axis and right y-axis) illustrate the evolution of the estimated
total number of classes (k + ˆu), where the ground-truth total is 100.
true value, highlighting the effectiveness and robustness of
the proposed class estimation strategy.