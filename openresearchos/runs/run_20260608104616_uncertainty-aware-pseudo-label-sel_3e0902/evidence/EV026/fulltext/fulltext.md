<!-- page 1 -->
DRIVE: DUAL-ROBUSTNESS VIA INFORMATION VARIABILITY
AND ENTROPIC CONSISTENCY IN SOURCE-FREE UNSUPERVISED
DOMAIN ADAPTATION
Ruiqiang Xiao∗
HKUST(GZ)
ruiqiangxiao@hkust-gz.edu.cn
Songning Lai∗
HKUST(GZ)
Deep Interdisciplinary Intelligence Lab
songninglai@hkust-gz.edu.cn
Yijun Yang
HKUST(GZ)
yyang018@connect.hkust-gz.edu.cn
Jiemin Wu
HKUST(GZ)
Deep Interdisciplinary Intelligence Lab
jieminwu@hkust-gz.edu.cn
Yutao Yue
HKUST(GZ)
Institute of Deep Perception Technology, JITRI
Deep Interdisciplinary Intelligence Lab
yutaoyue@hkust-gz.edu.cn
Lei Zhu†
HKUST(GZ)
leizhu@hkust-gz.edu.cn
ABSTRACT
Adapting machine learning models to new domains without labeled data, especially when source data
is inaccessible, is a critical challenge in applications like medical imaging, autonomous driving, and
remote sensing. This task, known as Source-Free Unsupervised Domain Adaptation (SFUDA),
involves adapting a pre-trained model to a target domain using only unlabeled target data, which can
lead to issues such as overfitting, underfitting, and poor generalization due to domain discrepancies and
noise. Existing SFUDA methods often rely on single-model architectures, struggling with uncertainty
and variability in the target domain. To address these challenges, we propose DRIVE (Dual-
Robustness through Information Variability and Entropy), a novel SFUDA framework leveraging a
dual-model architecture. The two models, initialized with identical weights, work in parallel to
capture diverse target domain characteristics. One model is exposed to perturbations via projection
gradient descent (PGD) guided by mutual information, focusing on high-uncertainty regions. We
also introduce an entropy-aware pseudo-labeling strategy that adjusts label weights based on
prediction uncertainty, ensuring the model focuses on reliable data while avoiding noisy regions.
The adaptation process has two stages: the first aligns the models on stable features using a mutual
information consistency loss, and the second dynamically adjusts the perturbation level based on
the loss from the first stage, encouraging the model to explore a broader range of the target domain
while preserving existing performance. This enhances generalization capabilities and robustness
against interference. Evaluations on standard SFUDA benchmarks show that DRIVE consistently
outperforms previous methods, delivering improved adaptation accuracy and stability across complex
target domains.
∗The first two authors contributed equally to this work.
†Correspondence to Prof. Lei Zhu {leizhu@hkust-gz.edu.cn}.
arXiv:2411.15976v2  [cs.CV]  23 Dec 2024


<!-- page 2 -->
Under Review
1
Introduction
Adapting machine learning models to new domains where labeled data is unavailable—especially when the original
source data cannot be reused—represents a complex yet critical challenge in practical applications across fields
like medical imaging [1], autonomous driving [2, 3], and remote sensing [4]. This task, known as Source-Free
Unsupervised Domain Adaptation (SFUDA), aims to adapt a pretrained model to a target domain using only
unlabeled target data. Unlike traditional domain adaptation methods, which typically require access to both source
and target data for knowledge transfer [5, 6], SFUDA operates under stricter conditions due to privacy, storage, and
regulatory constraints that prohibit the reuse of source data. Thus, SFUDA has become a crucial area of research for
real-world applications where source data is inaccessible.
A major challenge in SFUDA is the high risk of model overfitting or underfitting in the target domain [7, 8], especially
when domain discrepancies are large, or when the target data exhibits significant variability and noise [9, 10]. Without
source data to guide the model, SFUDA methods must rely solely on unlabeled target data, which provides limited
and often indirect information about the distributional differences between domains. To address this, most SFUDA
techniques rely on self-supervised learning, consistency regularization, or confidence-based approaches to iteratively
refine the model’s adaptation to the target domain [11, 12, 13]. However, these approaches often use single-model
architectures that struggle to generalize effectively when the target domain is noisy or uncertain, leading to brittle
performance and suboptimal results [14, 15, 16].
Existing methods like DIFO (Distilling multimodal Foundation models) [17], which leverages CLIP [18] and a two-
stage distillation process for source-free adaptation, have shown promise. However, DIFO still faces challenges in
handling substantial uncertainty or variability in the target domain. Its reliance on a fixed distillation strategy, which
combines predictions from CLIP and the target model, often results in suboptimal pseudo-labels, especially when the
target data is noisy or uncertain. This introduces a significant challenge for reliable domain transfer, as the model may
overfit to noisy data or fail to capture the comprehensive underlying distribution of the target domain.
This limitation of single-model approaches motivates us to explore more robust and adaptive frameworks. While DIFO
utilizes a single pretrained model for adaptation, it does not fully capitalize on the potential benefits of using multiple
models to handle the inherent variability and uncertainty in the target domain. We hypothesize that incorporating a
dual-model3 architecture can address these challenges more effectively, by leveraging complementary strengths of two
models working in parallel to promote more stable and transferable adaptations.
To this end, we propose DRIVE (Dual-Robustness through Information Variability and Entropy), a novel SFUDA
framework based on DIFO, designed to overcome these limitations. Our approach builds upon the insights from DIFO
but introduces several key modifications to enhance robustness and generalizability. Rather than relying on a fixed,
weighted combination of CLIP and the target model outputs, we introduce an entropy-based strategy to determine the
weight of pseudo-labels. This entropy-aware method accounts for the uncertainty in the model’s predictions, ensuring
that the model prioritizes reliable regions of the target domain while avoiding over-reliance on noisy or uncertain labels.
At the core of DRIVE is a dual-model architecture, where two models with identical initial weights operate in parallel
throughout the adaptation process. One model is exposed to perturbations generated through projection gradient
descent (PGD) [19], guided by mutual information. These perturbations target high-uncertainty regions of the target
domain, allowing the model to handle the domain’s variability more effectively. This dual-model setup takes advantage
of both consistency—encouraging convergence toward stable, transferable features—and divergence—fostering the
exploration of diverse characteristics of the target domain. This ensures that the models are not only robust but also
capable of capturing the full spectrum of target domain variability.
The adaptation process in DRIVE occurs in two stages. In the first stage, we apply a mutual information consistency
loss to align the two models on stable, transferable features despite the presence of perturbations. This phase minimizes
the impact of noise, ensuring that the adaptation process is based on reliable features that generalize well across domains.
Crucially, the loss function in the first phase also influences the initialization of PGD perturbations in the second
phase. By grounding the perturbations in robust, consistent features, the second phase can more effectively explore the
target domain’s variability, leading to more effective adaptation. This interdependence between the stages ensures that
the exploration of the target domain in the second phase is built upon a stable foundation, leading to more effective
adaptation and preventing overfitting to noisy data.
In the second stage, we introduce a mutual information divergence loss, which encourages the models to explore
complementary aspects of the target data. This phase mitigates the risk of overfitting and enables the models to learn
complementary information, enhancing the overall robustness and adaptability of the system. The interplay between the
3While there may be ambiguity here, we generalize DIFO as a single model architecture because we treat the overall framework
of ViL (CLIP) and target model as a single model.
2


<!-- page 3 -->
Under Review
Stage 1: Task-Specific ViL model Customization with Dual-model Consistency
Target: Get Robust Learnable Prompt Context 𝑣∗
Target Model 𝜃𝑡,1
Target Domain
Images
ViL Model 𝜃𝑣,1
𝑥𝑖
S𝑡,1
S𝑣,1
𝑥𝑖+ δ 𝑝
(1)
𝑥𝑗
PGD
ViL Model 𝜃𝑣,2
[CLS]
𝑣 → 𝑣∗
δ 𝑝
(1)
Stage 2: Knowledge Adaptation with Perturbed Model Encouragement
Target: Get Enhanced Target Model 𝜃𝑡,1
Target Domain
Images
𝑥𝑖
𝑥𝑖+ δ 𝑝
(2)
𝑥𝑗
PGD
δ 𝑝
(2)
Target Model 𝜃𝑡,1
ViL Model 𝜃𝑣,1
Target Model 𝜃𝑡,2
[CLS]
𝑣∗
S𝑣′
S𝑡′
𝐿𝑇𝐶𝑉(𝑥𝑖) |
Figure 1: Overview of our DRIVE: it illustrates the two-stage adaptation process of DRIVE, including task-specific
ViL model customization and knowledge adaptation with perturbed model encouragement, enhancing robustness and
generalization in SFUDA.
two phases ensures that the first stage’s stable learning process guides the more exploratory second phase, leading to a
seamless and effective adaptation trajectory.
Our approach offers several key contributions to the SFUDA landscape:
(i) Dual-model Architecture: We introduce a dual-model framework that promotes both consistency (for stable feature
alignment) and divergence (for exploring complementary aspects of the target domain), improving the robustness and
adaptability of the model in the target domain.
(i) Entropy-Aware Pseudo-Labeling: We propose an entropy-based strategy for pseudo-label weighting, ensuring that
the model focuses on reliable regions of the target domain while mitigating the risks of overfitting to uncertain or noisy
data.
(iii) Dynamic Perturbation Adjustment for Enhanced Exploration: We propose a two-phase adaptation process,
where the first stage aligns models on stable features, and the second stage leverages the first-stage loss to guide the
initialization of PGD perturbations, enhancing the exploration of high-uncertainty regions in the target domain.
2
Related Work
Source-Free Unsupervised Domain Adaptation.
SFUDA seeks to adapt models to a new, unlabeled target domain
without access to the original source data, often restricted due to privacy or storage constraints. Recent SFUDA methods
3


**[Table p3.1]**
|  |  |  |
| --- | --- | --- |
|  | Stage 1: Task-Specific ViL model Customization with Dual-model Consistency Target: Get Robust Learnable Prompt Context 𝑣∗ |  |
| Target Model 𝜃𝑡,1 S𝑡,1 Target Domain S𝑣,1 Images 𝑥𝑖 ViL Model 𝜃𝑣,1 𝑣 → 𝑣∗ [CLS] PGD ViL Model 𝜃𝑣,2 δ 𝑝(1) 𝑥𝑗 𝑥𝑖+ δ 𝑝(1) |  |  |
|  | Stage 2: Knowledge Adaptation with Perturbed Model Encouragement Target: Get Enhanced Target Model 𝜃 𝑡,1 |  |
| 𝑣∗ [CLS] S𝑣′ ViL Model 𝜃𝑣,1 Target Domain Images S𝑡′ 𝑥𝑖 Target Model 𝜃𝑡,1 PGD 𝐿𝑇𝐶𝑉(𝑥𝑖) | Target Model 𝜃𝑡,2 δ 𝑝(2) 𝑥𝑗 𝑥𝑖+ δ 𝑝(2) |  |  |


**[Table p3.2]**
| 𝑣 → 𝑣∗ |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |
|  |  |  |  |  |


**[Table p3.3]**
| 𝑣∗ |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |

[CAPTION] Figure 1: Overview of our DRIVE: it illustrates the two-stage adaptation process of DRIVE, including task-specific


<!-- page 4 -->
Under Review
have emphasized self-supervised learning, leveraging pseudo-labeling and entropy minimization to infer target domain
structures [20, 21, 22]. Consistency regularization techniques promote model stability across augmented target data,
effectively reducing uncertainties in the target domain [23, 24, 25, 16]. Additionally, approaches such as CPGA [26]
and BAIT [27] employ contrastive learning frameworks to align samples with category-wise prototypes. NRC [28]
and LSC-SDA [29] propagate categorical semantics through neighborhood or cluster structures in the feature space.
Xia et al. [30] focus on the disagreements between target data and the source model. Litrico et al. [22] leverage a
loss reweighting strategy that brings robustness against the noise that inevitably affects the pseudo-labels. DIFO [17]
explores the integration of heterogeneous knowledge sources from vision-language models.
Despite these advancements, most SFUDA methods rely on single-model frameworks that are susceptible to noise and
uncertainty in target data, especially when domain gaps are substantial. Our proposed method addresses this limitation
by introducing a dual-model structure, incorporating both consistency and divergence regularizations. This approach
leverages mutual information to enhance robustness across diverse target conditions.
Mutual Information.
Mutual information (MI) has emerged as a valuable tool in domain adaptation, enabling
models to capture and retain essential information across domains without requiring direct alignment data. MI-based
alignment techniques aim to maximize shared information between model representations and target domain features,
thereby enhancing adaptation robustness and the quality of extracted features. Approaches such as those by Peng et al.
[31] and Singha et al. [32] focus on MI-based metrics to strengthen domain-invariant feature extraction. In vision-
language models (VLMs), MI-based techniques facilitate the alignment of cross-modal information by maximizing
dependency between visual and language representations, ensuring robust cross-modal grounding. For instance, Ma et
al. [33] introduce mutual information contrastive learning to align vision-language representations, grounding language
specifications and rewarding learning from action-free videos with text annotations. Kim et al. [34] propose negative
Gaussian cross-mutual information, coined as Mutual Information Divergence, using CLIP features as a unified metric.
DIFO [17] customize VLMs by maximizing the mutual information with the target model in a prompt learning manner.
Beyond facilitating domain adaptation and cross-modal semantic alignment, MI also enables model-specific perturba-
tions that enhance adaptability and responsiveness to target-specific uncertainties. Our approach leverages MI both as a
consistency mechanism in the initial phase and as a divergence mechanism in the later stage, enabling the framework to
harness both alignment and diversity during model learning. This dual-phase application of MI ensures robust feature
alignment while promoting model flexibility across diverse target conditions.
3
Method
3.1
Problem Statement and Overview
Problem Statement. This work addresses the Source-Free Unsupervised Domain Adaptation (SFUDA) problem, where
the objective is to adapt a pre-trained source model θs to an unlabeled target domain Xt without direct access to the
labeled source data Xs, often due to privacy or storage constraints. Formally, we consider two domains that share C
common classes: a labeled source domain Xs and an unlabeled target domain Xt comprising n target samples {xi}n
i=1,
with unknown true labels Yt = {yi}n
i=1 in the target domain. The goal is to adapt θs to a target model θt : Xt →Yt
using only Xt and θs.
Overview of DRIVE Framework.
The DRIVE framework employs a dual-model, two-stage approach with
perturbation-induced robustness. It leverages Projected Gradient Descent (PGD) noise perturbations, where the
loss computed in Stage 1 influences the initialization of the perturbation in Stage 2. Both stages utilize an entropy-based
pseudo-labeling strategy. Details of the two stages are outlined below.
3.2
Stage 1: Task-Specific ViL Model Customization with Dual-Model Consistency
In the first stage, DRIVE employs prompt learning to customize the visual-language (ViL) model θv by aligning its
predictions with those of the frozen target model θt initialized with θs. This alignment is achieved through a dual-model
approach, where Model 1 (θv,1, θt,1) processes clean target samples, while Model 2 (θv,2, θt,2) is perturbed using
Projected Gradient Descent (PGD).
A PGD perturbation is defined as an adversarial adjustment to x, designed to maximize a loss function L(θt(x), y)
with respect to x while keeping the perturbed sample within an ϵ-ball around the original input x. Formally, a PGD
perturbation is given by:
xPGD = ProjB(x,ϵ) (x + α ∇xL(θt(x), y)) ,
(1)
where ProjB(x,ϵ) denotes projection onto the ϵ-ball around x, and α is a step size.
4


<!-- page 5 -->
Under Review
The perturbations in Model 2, guided by mutual information, explore meaningful variations in the target domain, while
the weights of Models 1 and 2 are shared. Importantly, θt,1 and θt,2 remain frozen throughout Stage 1.
Entropy-Aware Predictor. At this stage, pseudo-labels for the task are generated by combining the categorical
distribution outputs of θt,1(xi) and θv,1(xi, v) with the learnable prompt context v, weighted by their respective
entropies. The pseudo-label for Stage 1 is defined as follows:
P(1)(xi) =
Sv,1(xi)
Sv,1(xi) + St,1(xi) + λ · θt,1(xi)
(2)
+
St,1(xi) + λ
Sv,1(xi) + St,1(xi) + λ · θv,1(xi, v)
Here, Sv,1(xi) and St,1(xi) represent the entropy of the categorical distribution output from θv,1(xi, v) and θt,1(xi),
respectively. λ is a prior bias used to adjust the model’s confidence in the output of the ViL model.
Task-Specific ViL Loss. To align the predictions of θv,1(xi) with P(1)(xi), we apply a mutual information-based
consistency loss. The loss is formulated as:
L(1)
TSV = min
v
 
−Exi∈Xt, I
 θv,1(xi, v), P(1)(xi)
  
(3)
where I(·, ·) denotes the mutual information between the two predictions.
Mutual Information Consistency Loss for Stage 1. In this step, we apply a mutual information-based consistency
loss to align predictions from both the clean and perturbed models. Let the predictions from Model 2 be denoted by
θv,2(xi + δ(1), v) for a target sample xi. The mutual information-based consistency loss is given by:
L(1)
MIC = min
v −
 
Exi∈Xt, I(θt,1(xi), θt,2(xi + δ(1)))
 
(4)
This loss encourages Model 1 to generalize through the learnable text prompt context v in a way that remains consistent
with the perturbed Model 2, thus aligning both models’ outputs under varying target domain conditions.
The optimization process follows the PGD methodology [19], where iterative updates to the first stage perturbation δ(1)
are performed. At the p-th iteration, the current perturbation δ∗(1)
p−1 is updated as follows:
δ(1)
p
= δ∗(1)
p−1 +
γp
|Ap−1|
X
x∈Ap−1
∇δ∗(1)
p−1
[−Exi∈Xt, I
 θv,1(xi, v), P(1)(xi)
 
]
(5)
where δ∗(1)
p
= arg min||δ(1)||≤R ||δ(1) −δ(1)
p || and Ap−1 denotes a batch of samples, γp is the step size parameter for
PGD, and R is the norm bound for the perturbation. Specifically, the initialization of δ(1)
p
−δ(1) = Random(xj) is
a random input from this batch of samples Ap−1. Once δ(1)
P
is obtained after P iterations, we update v to v∗using
batched gradients.
3.3
Stage 2: Knowledge Adaptation with Perturbed Model Encouragement
In this stage, we extend the dual-model approach from Stage 1 to construct PGD perturbations, driving the target model
θt,1 to perform more extensive exploration of the target domain based on its overall consistency loss from Stage 1.
Dynamic Perturbation Adjustment for Enhanced Exploration. To ensure that the model performs extensive
exploration in the target domain while maintaining prediction consistency for familiar samples, DRIVE introduces a
dynamic perturbation adjustment mechanism in Stage 2. This mechanism dynamically adjusts the initialization noise
magnitude of the PGD perturbations based on the consistency losses from Stage 1. The perturbation noise magnitude η
for Stage 2 is defined as:
η ∝L(1)
TSV + βL(1)
MIC
(6)
The optimization process for these perturbations follows the PGD methodology [19]. At the p-th iteration, the current
perturbation δ∗(2)
p−1 is updated similarly to the process described earlier, but with specific adjustments for the iteration
and initialization. Specifically, we compute P(2)(xi) as the pseudo-label for Stage 2 computed based on Eq. 2 and the
5


<!-- page 6 -->
Under Review
target-domain-customized context prompt embedding v∗. In addition, the initialization of δ(2)
p −δ(2)
0
= η·Random(xj)
is a random input from this batch of samples Ap−1, scaled by η.
when the target model achieves high consistency for a target domain sample xi in Stage 1, it indicates that the model has
learned the label correspondence between this sample and the source domain samples. Choosing a smaller initialization
noise helps retain this learned stable mapping in subsequent training. Conversely, if the target model’s predictions for
the same target domain sample xi are inconsistent with the existing ViL prior knowledge and the predictions after PGD
perturbations in Stage 1, it suggests that the distribution space represented by this sample remains highly uncertain
for the current target model. Increasing the magnitude of the initialization noise aids the model in performing more
extensive exploration within the neighborhood of this sample.
Mutual Information Consistency Loss for Stage 2. As in Stage 1, to ensure that the perturbed and unperturbed
models align under a wide range of perturbations, we use mutual information-based consistency losses. The mutual
information consistency loss for Stage 2 is formulated as:
L(2)
MIC = min
θt,1 −
 
Exi∈Xt, I(θt,2(xi), θt,2(xi + δ(2)))
 
(7)
This loss encourages Model 1 to generalize in a manner that remains consistent with the perturbed Model 2, aligning
both models’ outputs under varying target domain conditions.
Predictive Consistency and Category Attention Calibration. To ensure knowledge adaptation and improve model
performance, we incorporate two key components following the DIFO method: predictive consistency loss and category
attention calibration.
First, the predictive consistency loss ensures that the target model’s predictions remain consistent with those of the ViL
model. This loss is defined as:
LPC = min
θt,1 [−Exi∈XtI (θt,1 (xi) , θv,1 (xi, v∗)) + αLB]
(8)
where the category balance term LB = KL(θv,1(xi)∥1
C ) ensures the predicted label distribution matches the uniform
distribution 1
C .
Second, we employ category attention calibration to regularize the model’s predictions using pseudo-labels. Specifically,
we identify the top-N most probable categories using Ps,1(xi). The indices of these categories are denoted by
Mi = {mk}N
k=1. The regularization loss is defined as:
LMCE = min
θt Exi∈Xt log
exp (ai/τ)
P
j /∈Mi exp (bi · li,j/τ)
ai =
N
Y
k=1
li,mk,
bi =
N
X
k=1
li,mk
(9)
where li,j denotes the j-th element of the logit vector li and τ is the temperature parameter.
Together, these mechanisms ensure that the target model not only adapts to the target domain but also explores new and
challenging examples, thereby improving its generalization and robustness.
3.4
Training Procedure
The training process for DRIVE iterates between Stage 1 (ViL model customization) and Stage 2 (knowledge adaptation),
progressively adapting θt,1 and θt,2 by leveraging both clean and perturbed inputs. In each epoch, Stage 1 optimizes
prompt v with L(1)
total = L(1)
TSV + βL(1)
MIC, while Stage 2 adapts θt,1 using:
L(2)
total = LMCE + ξ1LPC + ξ2L(2)
MIC
(10)
Here, ξ1 and ξ2 are hyperparameters that control the weights of the respective loss components, LPC and L(2)
MIC. The
details of this procedure are summarized in Algorithm 1.
4
Experiments
4.1
Experimental Setup
Datasets: We evaluate our proposed framework, DRIVE, on four benchmark datasets for domain adaptation: Office-31
[35], Office-Home [36], and DomainNet-126 [31]. These datasets offer varying levels of complexity, from small-scale
6


<!-- page 7 -->
Under Review
Algorithm 1 Training Algorithm for DRIVE Framework
1: Input: Target domain Xt, ViL model θv, target model θt, perturbation magnitude η0, balancing factor β, ξ1, ξ2
2: Output: Adapted target model θt
3: Initialize target model θt and ViL model θv with pre-trained weights
4: for each epoch do
5:
Stage 1: ViL Model Customization
6:
for each sample xi ∈Xt do
7:
Sample perturbation δ with magnitude η0
8:
Compute clean prediction from target model: θt,1(xi)
9:
Compute clean prediction from ViL model: θv,1(xi, v)
10:
Compute perturbed prediction from ViL model: θv,2(xi + δ, v)
11:
Compute pseudo-label using entropy-aware predictor: P(1)(xi)
12:
Calculate mutual information consistency loss: L(1)
total = L(1)
TSV + βL(1)
MIC
13:
Update prompt embedding v using L(1)
total
14:
end for
15:
Stage 2: Knowledge Adaptation
16:
for each sample xi ∈Xt do
17:
Compute perturbation: δ with adaptive dynamical magnitude η and L(1)
total
18:
Compute pseudo-label using entropy-aware predictor: P(2)(xi)
19:
Compute mutual information-divergence loss: LMID
20:
Compute predictive consistency loss: LPC
21:
Compute category attention calibration loss LMCE
22:
Update target model θt using the combined loss:
L(2)
total = LMCE + ξ1LPC + ξ2L(2)
MIC
23:
end for
24: end for
(Office-31) to large-scale (Ofiice-Home, DomainNet-126), ensuring a comprehensive assessment across different
domain shifts and challenges. Dataset details are provided in the Supplementary Materials.
Baselines: We compare DRIVE with leading SFUDA methods across three categories:(i) Source model: Baselines
include Source (the source-only model). (ii) Multimodal UDA Methods: We include DAPL [23], PADCLIP [25],
ADCLIP [32], and DIFO [17], which leverage multimodal models for domain adaptation. (iii) SFUDA Methods: Key
SFUDA methods include SHOT [20], NRC [21], and TPDS [37], which address domain shifts without requiring source
data during adaptation.
Implementation Details: WWe present DRIVE-C-B32, which utilizes the ViT-B/32 backbone, and is designed to be
broadly similar to DIFO-C-B2 (also using ViT-B/32). Experiments follow the same settings as previous baselines for
fair comparison, with further details in the Supplementary Materials.
4.2
Results on Closed-set SFUDA
On the Office-31 dataset(Table 1), DRIVE achieves a mean accuracy of 92.7%, significantly outperforming state-of-the-
art methods. Specifically, in the challenging tasks such as A →D (97.4%) and A →W (95.7%), DRIVE demonstrates
robustness and adaptability. The entropy-aware pseudo-labeling strategy ensures that the model focuses on reliable
regions, reducing overfitting to noisy data. This is particularly important in tasks where the target domain has high
variability and noise. For the Office-Home dataset(Table 2), DRIVE achieves a mean accuracy of 83.6%. Notable
improvements are observed in tasks such as Ar →Cl (72.4%) and Cl →Pr (90.7%), which are known for their high
uncertainty. The dynamic perturbation adjustment mechanism enhances the exploration of high-uncertainty regions,
leading to better feature alignment and generalization. This is crucial for tasks where the target domain has significant
variations, and the model needs to adapt effectively to these changes.
On the DomainNet-126 dataset(Table 3), DRIVE achieves a mean accuracy of 80.6%, outperforming other methods. In
tasks such as C →R (88.1%) and R →C (81.0%), DRIVE shows strong performance, indicating its ability to handle
large domain shifts. The dual-model architecture and mutual information-driven consistency loss ensure stable feature
alignment, even in complex target domains. This robust performance across multiple datasets and tasks highlights the
effectiveness of DRIVE in addressing the challenges of closed-set SFUDA.
7


<!-- page 8 -->
Under Review
Table 1: Closed-set SFDA on Office-31 (%).
Method
Venue
A →D
A →W
D →A
D →W
W →A
W →D
Mean
Source
–
79.1
76.6
59.9
95.5
61.4
98.8
78.6
SHOT
ICML20
93.7
91.1
74.2
98.2
74.6
100.0
88.6
NRC
NIPS21
96.0
90.8
75.3
99.0
75.0
100.0
89.4
GKD
IROS21
94.6
91.6
75.1
98.7
75.1
100.0
89.2
HCL
NIPS21
94.7
92.5
75.9
98.2
77.7
100.0
89.8
AaD
NIPS22
96.4
92.1
75.0
99.1
76.5
100.0
89.9
AdaCon
CVPR22
87.7
83.1
73.7
91.3
77.6
72.8
81.0
CoWA
ICML22
94.4
95.2
76.2
98.5
77.6
99.8
90.3
SCLM
NN22
95.8
90.0
75.5
98.9
75.5
99.8
89.4
ELR
ICLR23
93.8
93.3
76.2
98.0
76.9
100.0
89.6
PLUE
CVPR23
89.2
88.4
72.8
97.1
69.6
97.9
85.8
TPDS
IJCV23
97.1
94.5
75.7
98.7
75.5
99.8
90.2
DIFO-C-B32
CVPR24
96.2
95.0
83.1
96.0
83.0
99.0
92.0
DRIVE(Ours)
–
97.4
96.0
83.5
96.9
83.1
99.4
92.7
Table 2: Closed-set SFDA on Office-home (%). SF and M means source-free and multimodal, respectively.
Method
Venue
SF
M
Office-home
Ar →Cl
Ar →Pr
Ar →Rw
Cl →Ar
Cl →Pr
Cl →Rw
Pr →Ar
Pr →Cl
Pr →Rw
Rw →Ar
Rw→Cl
Rw →Pr
Mean
Source
–
-
-
43.7
67.0
73.9
49.9
60.1
62.5
51.7
40.9
72.6
64.2
46.3
78.1
59.2
DAPL-RN
TNNLS23
✕
✓
54.1
84.3
84.8
74.4
83.7
85.0
74.5
54.6
84.8
75.2
54.7
83.8
74.5
PADCLIP-RN
ICCV23
✕
✓
57.5
84.0
83.8
77.8
85.5
84.7
76.3
59.2
85.4
78.1
60.2
86.7
76.6
ADCLIP-RN
ICCVW23
✕
✓
55.4
85.2
85.6
76.1
85.8
86.2
76.7
56.1
85.4
76.8
56.1
85.5
75.9
SHOT
ICML20
✓
✕
56.7
77.9
80.6
68.0
78.0
79.4
67.9
54.5
82.3
74.2
58.6
84.5
71.9
NRC
NIPS21
✓
✕
57.7
80.3
82.0
68.1
79.8
78.6
65.3
56.4
83.0
71.0
58.6
85.6
72.2
GKD
IROS21
✓
✕
56.5
78.2
81.8
68.7
78.9
79.1
67.6
54.8
82.6
74.4
58.5
84.8
72.2
AaD
NIPS22
✓
✕
47.2
75.1
75.5
60.7
73.3
73.2
60.2
45.2
76.6
65.6
48.3
79.1
65.0
AdaCon
CVPR22
✓
✕
56.9
78.4
81.0
69.1
80.0
79.9
67.7
57.2
82.4
72.8
60.5
84.5
72.5
CoWA
ICML22
✓
✕
56.9
78.4
81.0
69.1
80.0
79.9
67.7
57.2
82.4
72.8
60.5
84.5
72.5
SCLM
NN22
✓
✕
58.2
80.3
81.5
69.3
79.0
80.7
69.0
56.8
82.7
74.7
60.6
85.0
73.0
ELR
ICLR23
✓
✕
58.4
78.7
81.5
69.2
79.5
79.3
66.3
58.0
82.6
73.4
59.8
85.1
72.6
PLUE
CVPR23
✓
✕
49.1
73.5
78.2
62.9
73.5
74.5
62.2
48.3
78.6
68.6
51.8
81.5
66.9
TPDS
IJCV23
✓
✕
59.3
80.3
82.1
70.6
79.4
80.9
69.8
56.8
82.1
74.5
61.2
85.3
73.5
DIFO-C-B32
CVPR24
✓
✓
70.6
90.6
88.8
82.5
90.6
88.8
80.9
70.1
88.9
83.4
70.5
91.2
83.1
DRIVE(Ours)
–
✓
✓
72.4
89.9
89.2
81.8
90.7
89.3
80.9
71.2
90.0
83.6
73.2
91.5
83.6
These results substantiate DRIVE’s capability to enhance cross-domain performance in closed-set SFUDA settings,
benefiting from its dual-model architecture and mutual information-driven consistency mechanism.
Table 3: Closed-set SFDA on DomainNet-126 (%). SF and M means source-free and multimodal, respectively.
Method
Venue
SF
M
C→R
R→C
R→P
R→S
S→C
Mean
Source
-
-
-
59.8
55.3
62.7
46.4
55.1
55.9
DAPL-RN
TNNLS23
✕
✓
87.6
73.2
72.4
66.2
73.8
74.6
ADCLIP-RN
ICCVW23
✕
✓
88.1
73.6
73.0
68.4
72.3
75.1
SHOT
ICML20
✓
✕
78.2
67.7
67.6
57.8
70.2
68.3
NRC
NIPS21
✓
✕
77.1
64.7
69.4
58.7
69.4
67.9
GKD
IROS21
✓
✕
77.4
68.3
68.4
59.5
71.5
69.0
AdaCon
CVPR22
✓
✕
74.8
63.1
68.1
55.6
67.1
65.7
CoWA
ICML22
✓
✕
80.6
69.0
67.2
60.0
69.0
69.2
PLUE
CVPR23
✓
✕
74.0
61.6
65.9
53.8
67.5
64.6
TPDS
IJCV23
✓
✕
77.1
66.4
67.0
58.2
68.6
67.5
DIFO-C-B32
CVPR24
✓
✓
87.9
80.1
77.4
75.5
79.2
80.0
DRIVE(Ours)
-
✓
✓
88.1
81.0
78.2
76.0
79.8
80.6
4.3
Ablation Study
To thoroughly evaluate the contributions of each component in DRIVE, we conducted an ablation study focusing on the
following aspects: Entropy-Aware Predictor, Perturbed Model Encouragement, and Dynamic Perturbation Adjustment.
8


**[Table p8.1]**
| Method Venue |  | A →D A →W D →A D →W W →A W →D Mean |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Source – SHOT ICML20 NRC NIPS21 GKD IROS21 HCL NIPS21 AaD NIPS22 AdaCon CVPR22 CoWA ICML22 SCLM NN22 ELR ICLR23 PLUE CVPR23 TPDS IJCV23 DIFO-C-B32 CVPR24 |  | 79.1 76.6 59.9 95.5 61.4 98.8 78.6 93.7 91.1 74.2 98.2 74.6 100.0 88.6 96.0 90.8 75.3 99.0 75.0 100.0 89.4 94.6 91.6 75.1 98.7 75.1 100.0 89.2 94.7 92.5 75.9 98.2 77.7 100.0 89.8 96.4 92.1 75.0 99.1 76.5 100.0 89.9 87.7 83.1 73.7 91.3 77.6 72.8 81.0 94.4 95.2 76.2 98.5 77.6 99.8 90.3 95.8 90.0 75.5 98.9 75.5 99.8 89.4 93.8 93.3 76.2 98.0 76.9 100.0 89.6 89.2 88.4 72.8 97.1 69.6 97.9 85.8 97.1 94.5 75.7 98.7 75.5 99.8 90.2 96.2 95.0 83.1 96.0 83.0 99.0 92.0 |  |  |  |  |  |  |
| DRIVE(Ours) | – | 97.4 | 96.0 | 83.5 | 96.9 | 83.1 | 99.4 | 92.7 |


**[Table p8.2]**
| Method Venue |  | SF | M | Offcie-home Ar →Cl Ar →Pr Ar →Rw Cl →Ar Cl →Pr Cl →Rw Pr →Ar Pr →Cl Pr →Rw Rw →Ar Rw→Cl Rw →Pr Mean |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Source – |  | - | - | 43.7 67.0 73.9 49.9 60.1 62.5 51.7 40.9 72.6 64.2 46.3 78.1 59.2 |  |  |  |  |  |  |  |  |  |  |  |  |
| DAPL-RN TNNLS23 PADCLIP-RN ICCV23 ADCLIP-RN ICCVW23 |  | ✕ ✕ ✕ | ✓ ✓ ✓ | 54.1 84.3 84.8 74.4 83.7 85.0 74.5 54.6 84.8 75.2 54.7 83.8 74.5 57.5 84.0 83.8 77.8 85.5 84.7 76.3 59.2 85.4 78.1 60.2 86.7 76.6 55.4 85.2 85.6 76.1 85.8 86.2 76.7 56.1 85.4 76.8 56.1 85.5 75.9 |  |  |  |  |  |  |  |  |  |  |  |  |
| SHOT ICML20 NRC NIPS21 GKD IROS21 AaD NIPS22 AdaCon CVPR22 CoWA ICML22 SCLM NN22 ELR ICLR23 PLUE CVPR23 TPDS IJCV23 DIFO-C-B32 CVPR24 |  | ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ | ✕ ✕ ✕ ✕ ✕ ✕ ✕ ✕ ✕ ✕ ✓ | 56.7 77.9 80.6 68.0 78.0 79.4 67.9 54.5 82.3 74.2 58.6 84.5 71.9 57.7 80.3 82.0 68.1 79.8 78.6 65.3 56.4 83.0 71.0 58.6 85.6 72.2 56.5 78.2 81.8 68.7 78.9 79.1 67.6 54.8 82.6 74.4 58.5 84.8 72.2 47.2 75.1 75.5 60.7 73.3 73.2 60.2 45.2 76.6 65.6 48.3 79.1 65.0 56.9 78.4 81.0 69.1 80.0 79.9 67.7 57.2 82.4 72.8 60.5 84.5 72.5 56.9 78.4 81.0 69.1 80.0 79.9 67.7 57.2 82.4 72.8 60.5 84.5 72.5 58.2 80.3 81.5 69.3 79.0 80.7 69.0 56.8 82.7 74.7 60.6 85.0 73.0 58.4 78.7 81.5 69.2 79.5 79.3 66.3 58.0 82.6 73.4 59.8 85.1 72.6 49.1 73.5 78.2 62.9 73.5 74.5 62.2 48.3 78.6 68.6 51.8 81.5 66.9 59.3 80.3 82.1 70.6 79.4 80.9 69.8 56.8 82.1 74.5 61.2 85.3 73.5 70.6 90.6 88.8 82.5 90.6 88.8 80.9 70.1 88.9 83.4 70.5 91.2 83.1 |  |  |  |  |  |  |  |  |  |  |  |  |
| DRIVE(Ours) | – | ✓ | ✓ | 72.4 | 89.9 | 89.2 | 81.8 | 90.7 | 89.3 | 80.9 | 71.2 | 90.0 | 83.6 | 73.2 | 91.5 | 83.6 |


**[Table p8.3]**
| Method Venue | SF | M | C→R R→C R→P R→S S→C Mean |
| --- | --- | --- | --- |


**[Table p8.4]**
| SHOT ICML20 NRC NIPS21 GKD IROS21 AdaCon CVPR22 CoWA ICML22 PLUE CVPR23 TPDS IJCV23 | ✓ ✓ ✓ ✓ ✓ ✓ ✓ | ✕ ✕ ✕ ✕ ✕ ✕ ✕ | 78.2 67.7 67.6 57.8 70.2 68.3 77.1 64.7 69.4 58.7 69.4 67.9 77.4 68.3 68.4 59.5 71.5 69.0 74.8 63.1 68.1 55.6 67.1 65.7 80.6 69.0 67.2 60.0 69.0 69.2 74.0 61.6 65.9 53.8 67.5 64.6 77.1 66.4 67.0 58.2 68.6 67.5 |
| --- | --- | --- | --- |


**[Table p8.5]**
| DIFO-C-B32 CVPR24 |  | ✓ | ✓ | 87.9 80.1 77.4 75.5 79.2 80.0 |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DRIVE(Ours) | - | ✓ | ✓ | 88.1 | 81.0 | 78.2 | 76.0 | 79.8 | 80.6 |

[CAPTION] Table 1: Closed-set SFDA on Office-31 (%).

[CAPTION] Table 2: Closed-set SFDA on Office-home (%). SF and M means source-free and multimodal, respectively.

[CAPTION] Table 3: Closed-set SFDA on DomainNet-126 (%). SF and M means source-free and multimodal, respectively.


<!-- page 9 -->
Under Review
Entropy-Aware Predictor. The Entropy-Aware Predictor is designed to adjust label weights based on prediction
uncertainty, ensuring the model focuses on reliable data while avoiding noisy regions. As shown in Table 4, enabling
the Entropy-Aware Predictor alone (second row) improves performance on the A →D and D →W tasks, indicating its
effectiveness in enhancing the model’s robustness.
Perturbed Model Encouragement. The Perturbed Model Encouragement mechanism exposes one of the models
to perturbations via projection gradient descent (PGD) guided by mutual information, focusing on high-uncertainty
regions. When combined with the Entropy-Aware Predictor (third row), there is a further improvement in performance,
particularly on the A →D and D →W tasks, demonstrating the importance of exploring the feature space more
comprehensively.
Dynamic Perturbation Adjustment. The Dynamic Perturbation Adjustment mechanism dynamically adjusts the
perturbation level based on the loss from the first stage, encouraging the model to explore a broader range of the target
domain while preserving existing performance. Adding this component (fourth row) maintains the performance gains
observed in the third row, indicating that dynamic adjustment is beneficial for maintaining robustness and generalization.
Table 4: Classification results of ablation study on several settings on Office31 (%). The full results are provided in
Supplementary.
Entropy-Aware Predictor
Perturbed Model Encouragement
Dynamic Perturbation Adjustment
Office31
A →D
A →W
D →W
✕
✕
✕
96.39
95.72
95.6
✓
✕
✕
96.99
95.72
95.97
✓
✓
✕
97.39
95.85
97.74
✓
✓
✓
97.39
95.97
97.74
These findings highlight the importance of each component in DRIVE and validate the design choices made to
ensure robust and effective domain adaptation. The combined use of the Entropy-Aware Predictor, Perturbed Model
Encouragement, and Dynamic Perturbation Adjustment significantly enhances the model’s performance across different
domain adaptation tasks.
4.4
Grad-CAM Visualization of SFUDA
To gain deeper insights into how DRIVE and competing models attend to domain-relevant features, we conducted
Grad-CAM analysis on selected samples from the target domain, which shown on 2. By comparing the attention maps
generated by DRIVE against those produced by competitive baseline(DIFO), we can assess the efficacy of our method in
focusing on meaningful and domain-invariant features. The Grad-CAM visualizations thus provide empirical evidence
supporting our claim that DRIVE not only achieves quantitative performance gains but also qualitatively attends to more
meaningful and transferable features within the target domain. This characteristic is critical for achieving robust and
reliable domain adaptation, especially in real-world applications where domain shifts are common and unpredictable.
5
Conclusion
Source-Free Unsupervised Domain Adaptation (SFUDA), involves adapting a pre-trained model to a target domain
using only unlabeled target data, leading to issues such as overfitting, underfitting, and poor generalization due to
domain discrepancies and noise. Existing SFUDA methods often struggle with these challenges due to their reliance on
single-model architectures. To address these difficulties, we introduced DRIVE (Dual-Robustness through Information
Variability and Entropy), a novel SFUDA framework leveraging a dual-model architecture. DRIVE captures diverse
characteristics of the target domain by using two models initialized with identical weights, one of which is exposed to
perturbations via projection gradient descent (PGD) guided by mutual information, focusing on high-uncertainty regions.
Additionally, we introduced an entropy-aware pseudo-labeling strategy that adjusts label weights based on prediction
uncertainty, ensuring the model focuses on reliable data while avoiding noisy regions. The adaptation process consists
of two stages: the first aligns the models on stable features using a mutual information consistency loss, and the second
dynamically adjusts the perturbation level based on the loss from the first stage, encouraging the model to explore a
broader range of the target domain while preserving existing performance. This enhances generalization capabilities
and robustness against interference. Evaluations on standard SFUDA benchmarks show that DRIVE consistently
outperforms previous methods, delivering improved adaptation accuracy and stability across complex target domains.
9


**[Table p9.1]**
| Offcie31 Entropy-Aware Predictor Perturbed Model Encouragement Dynamic Perturbation Adjustment A →D A →W D →W |  |
| --- | --- |
| ✕ ✕ ✕ ✓ ✕ ✕ ✓ ✓ ✕ ✓ ✓ ✓ | 96.39 95.72 95.6 96.99 95.72 95.97 97.39 95.85 97.74 97.39 95.97 97.74 |

[CAPTION] Table 4: Classification results of ablation study on several settings on Office31 (%). The full results are provided in


<!-- page 10 -->
Under Review
Figure 2: Grad-CAM visualization of our proposed method (DRIVE) compared to the primary baseline (DIFO), and
the evolution of Grad-CAM visualizations for DRIVE as the number of iterations increases.
References
[1] Heng Li, Ziqin Lin, Zhongxi Qiu, Zinan Li, Ke Niu, Na Guo, Huazhu Fu, Yan Hu, and Jiang Liu. Enhancing
and adapting in the clinic: Source-free unsupervised domain adaptation for medical image enhancement. IEEE
Transactions on Medical Imaging, 2023.
[2] Yuqi Fang, Pew-Thian Yap, Weili Lin, Hongtu Zhu, and Mingxia Liu. Source-free unsupervised domain adaptation:
A survey. Neural Networks, page 106230, 2024.
[3] Deepti Hegde, Velat Kilic, Vishwanath Sindagi, A Brinton Cooper, Mark Foster, and Vishal M Patel. Source-
free unsupervised domain adaptation for 3d object detection in adverse weather. In 2023 IEEE International
Conference on Robotics and Automation (ICRA), pages 6973–6980. IEEE, 2023.
[4] Weixing Liu, Jun Liu, Xin Su, Han Nie, and Bin Luo. Source-free domain adaptive object detection in remote
sensing images. arXiv preprint arXiv:2401.17916, 2024.
[5] Yaroslav Ganin and Victor S. Lempitsky. Unsupervised domain adaptation by backpropagation. In ICML, 2015.
[6] Guoliang Kang, Lu Jiang, Yi Yang, and Alexander G Hauptmann. Contrastive adaptation network for unsupervised
domain adaptation. In CVPR, 2019.
[7] Liqiang Yuan. Domain adaptation in biomedical engineering: unsupervised, source-free, and black box approaches.
2024.
10

[CAPTION] Figure 2: Grad-CAM visualization of our proposed method (DRIVE) compared to the primary baseline (DIFO), and


<!-- page 11 -->
Under Review
[8] Xin Luo, Wei Chen, Zhengfa Liang, Longqi Yang, Siwei Wang, and Chen Li. Crots: Cross-domain teacher–student
learning for source-free domain adaptive semantic segmentation. International Journal of Computer Vision,
132(1):20–39, 2024.
[9] Yuchang Zhao, Shuai Feng, Chang Li, Rencheng Song, Deng Liang, and Xun Chen. Source-free domain adaptation
for privacy-preserving seizure prediction. IEEE Transactions on Industrial Informatics, 2023.
[10] Jingjing Li, Zhiqi Yu, Zhekai Du, Lei Zhu, and Heng Tao Shen. A comprehensive survey on source-free domain
adaptation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.
[11] Ning Ding, Yixing Xu, Yehui Tang, Chao Xu, Yunhe Wang, and Dacheng Tao. Source-free domain adaptation via
distribution estimation. In CVPR, 2022.
[12] Jiayi Tian, Jing Zhang, Wen Li, and Dong Xu. Vdm-da: Virtual domain modeling for source data-free domain
adaptation. IEEE Transactions on Circuits and Systems for Video Technology, 32(6):3749–3760, 2021.
[13] Jogendra Nath Kundu, Akshay R Kulkarni, Suvaansh Bhambri, Deepesh Mehta, Shreyas Anand Kulkarni, Varun
Jampani, and Venkatesh Babu Radhakrishnan. Balancing discriminability and transferability for source-free
domain adaptation. In ICML, 2022.
[14] Qicheng Lao, Xiang Jiang, and Mohammad Havaei. Hypothesis disparity regularized mutual information
maximization. In AAAI, 2021.
[15] Shiqi Yang, Yaxing Wang, Kai Wang, Shangling Jui, et al. Attracting and dispersing: A simple approach for
source-free domain adaptation. In NeurIPS, 2022.
[16] Song Tang, Yan Zou, Zihao Song, Jianzhi Lyu, Lijuan Chen, Mao Ye, Shouming Zhong, and Jianwei Zhang.
Semantic consistency learning on manifold for source data-free unsupervised domain adaptation. Neural Networks,
152, 2022.
[17] Song Tang, Wenxin Su, Mao Ye, and Xiatian Zhu. Source-free domain adaptation with frozen multimodal
foundation model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
pages 23711–23720, 2024.
[18] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language
supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.
[19] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep
learning models resistant to adversarial attacks. In International Conference on Learning Representations, 2018.
[20] Jian Liang, Dapeng Hu, and Jiashi Feng. Do we really need to access the source data? source hypothesis transfer
for unsupervised domain adaptation. In ICML, 2020.
[21] Shiqi Yang, Joost van de Weijer, Luis Herranz, Shangling Jui, et al. Exploiting the intrinsic neighborhood structure
for source-free domain adaptation. In NeurIPS, 2021.
[22] Mattia Litrico, Alessio Del Bue, and Pietro Morerio. Guiding pseudo-labels with uncertainty estimation for
source-free unsupervised domain adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 7640–7650, 2023.
[23] Chunjiang Ge, Rui Huang, Mixue Xie, Zihang Lai, Shiji Song, Shuang Li, and Gao Huang. Domain adaptation
via prompt learning. IEEE Transactions on Neural Networks and Learning Systems, 2023.
[24] Jonghyun Lee, Dahuin Jung, Junho Yim, and Sungroh Yoon. Confidence score for source-free unsupervised
domain adaptation. In International conference on machine learning, pages 12365–12377. PMLR, 2022.
[25] Zhengfeng Lai, Noranart Vesdapunt, Ning Zhou, Jun Wu, Cong Phuoc Huynh, Xuelu Li, Kah Kuen Fu, and
Chen-Nee Chuah. Padclip: Pseudo-labeling with adaptive debiasing in clip for unsupervised domain adaptation.
In ICCV, 2023.
[26] Zhen Qiu, Yifan Zhang, Hongbin Lin, Shuaicheng Niu, Yanxia Liu, Qing Du, and Mingkui Tan. Source-free
domain adaptation via avatar prototype generation and adaptation. arXiv preprint arXiv:2106.15326, 2021.
[27] Shiqi Yang, Yaxing Wang, Joost van de Weijer, Luis Herranz, and Shangling Jui. Unsupervised domain adaptation
without source data by casting a bait. arXiv preprint arXiv:2010.12427, 2020.
[28] Shiqi Yang, Yaxing Wang, Joost van de Weijer, Luis Herranz, and Shangling Jui. Exploiting the intrinsic
neighborhood structure for source-free domain adaptation. arXiv preprint arXiv:2110.04202, 2021.
[29] Shiqi Yang, Yaxing Wang, Joost van de Weijer, Luis Herranz, and Shangling Jui. Generalized source-free domain
adaptation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8978–8987,
2021.
11


<!-- page 12 -->
Under Review
[30] Haifeng Xia, Handong Zhao, and Zhengming Ding. Adaptive adversarial network for source-free domain
adaptation. In CVPR, 2021.
[31] Xingchao Peng, Qinxun Bai, Xide Xia, Zijun Huang, Kate Saenko, and Bo Wang. Moment matching for
multi-source domain adaptation. In ICCV, 2019.
[32] Mainak Singha, Harsh Pal, Ankit Jha, and Biplab Banerjee. Ad-clip: Adapting domains in prompt space using
clip. In ICCV Workshop, 2023.
[33] Yecheng Jason Ma, Vikash Kumar, Amy Zhang, Osbert Bastani, and Dinesh Jayaraman. Liv: Language-image
representations and rewards for robotic control. In International Conference on Machine Learning, pages
23301–23320. PMLR, 2023.
[34] Jin-Hwa Kim, Yunji Kim, Jiyoung Lee, Kang Min Yoo, and Sang-Woo Lee. Mutual information divergence: A
unified metric for multimodal generative models. Advances in Neural Information Processing Systems, 35:35072–
35086, 2022.
[35] Kate Saenko, Brian Kulis, Mario Fritz, and Trevor Darrell. Adapting visual category models to new domains. In
ECCV, 2010.
[36] Hemanth Venkateswara, Jose Eusebio, Shayok Chakraborty, and Sethuraman Panchanathan. Deep hashing
network for unsupervised domain adaptation. In CVPR, 2017.
[37] Song Tang, An Chang, Fabian Zhang, Xiatian Zhu, Mao Ye, and Changshui Zhang. Source-free domain adaptation
via target prediction distribution searching. International Journal of Computer Vision, pages 1–19, 2023.
12