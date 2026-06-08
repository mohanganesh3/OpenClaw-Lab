<!-- page 1 -->
Journal of Data-centric Machine Learning Research (2024)
Submitted 3/24; Revised 5/24; Published 6/24
You can’t handle the (dirty) truth:
Data-centric insights improve pseudo-labeling
Nabeel Seedat∗
ns741@cam.ac.uk
University of Cambridge, Cambridge, UK
Nicolas Huynh∗
nvth2@cam.ac.uk
University of Cambridge, Cambridge, UK
Fergus Imrie
imrie@ucla.edu
University of California, Los Angeles, CA, USA
Mihaela van der Schaar
mv472@cam.ac.uk
University of Cambridge, Cambridge, UK
Reviewed on OpenReview: https: // openreview. net/ forum? id= 2tBwcT9z55
Editor: Sergio Escalera
Abstract
Pseudo-labeling is a popular semi-supervised learning technique to leverage unlabeled data
when labeled samples are scarce. The generation and selection of pseudo-labels heavily rely
on labeled data. Existing approaches implicitly assume that the labeled data is gold standard
and “perfect”. However, this can be violated in reality with issues such as mislabeling or
ambiguity. We address this overlooked aspect and show the importance of investigating
labeled data quality to improve any pseudo-labeling method. Specifically, we introduce a
novel data characterization and selection framework called DIPS to extend pseudo-labeling.
We select useful labeled and pseudo-labeled samples via analysis of learning dynamics. We
demonstrate the applicability and impact of DIPS for various pseudo-labeling methods across
an extensive range of real-world tabular and image datasets. Additionally, DIPS improves
data efficiency and reduces the performance distinctions between different pseudo-labelers.
Overall, we highlight the significant benefits of a data-centric rethinking of pseudo-labeling
in real-world settings.
Keywords:
pseudo-labeling, semi-supervised learning, data characterization
1 Introduction
Machine learning heavily relies on the availability of large numbers of annotated training
examples. However, in many real-world settings, such as healthcare and finance, collecting
even limited numbers of annotations is often either expensive or practically impossible. Semi-
supervised learning leverages unlabeled data to combat the scarcity of labeled data (Zhu,
2005; Chapelle et al., 2006; van Engelen and Hoos, 2019). Pseudo-labeling is a prominent
semi-supervised approach applicable across data modalities that assigns pseudo-labels to
unlabeled data using a model trained on the labeled dataset. The pseudo-labeled data is then
∗. Equal Contribution
©2024 Seedat, Huynh, Imrie and van der Schaar.
arXiv:2406.13733v1  [cs.LG]  19 Jun 2024


<!-- page 2 -->
Seedat, Huynh, Imrie and van der Schaar
Assumption
Real-world settings
Pseudo-labeling
Pseudo-labeling
Class 1
Class 2
Unlabeled
Mislabeled
Figure 1: (Left) Current pseudo-labeling formulations implicitly assume that the labeled
data is the gold standard. (Right) However, this assumption is violated in real-world settings.
Mislabeled samples lead to error propagation when pseudo-labeling the unlabeled data.
combined with labeled data to produce an augmented training set. This increases the size of
the training set and has been shown to improve the resulting model. In contrast, consistency
regularization methods (Sohn et al., 2020) are less versatile and often not applicable to
settings such as tabular data, where defining the necessary semantic-preserving augmentations
proves challenging (Gidaris et al., 2018; Nguyen et al., 2022a). Given the broad applicability
across data modalities and competitive performance, we focus on pseudo-labeling approaches.
Labeled data is not always gold standard. Current pseudo-labeling methods focus
on unlabeled data selection. However, an equally important yet overlooked problem is around
labeled data quality, given the reliance of pseudo-labelers on the labeled data. In particular,
it is often implicitly assumed that the labeled data is “gold standard and perfect”.
This “gold standard” assumption is unlikely to hold in reality, where data can have issues
such as mislabeling and ambiguity (Sambasivan et al., 2021; Renggli et al., 2021; Jain et al.,
2020; Gupta et al., 2021a,b; Northcutt et al., 2021a,b).
For example, Northcutt et al. (2021b) quantified the label error rate of widely-used
benchmark datasets, reaching up to 10%, while Wei et al. (2022a) showed this can be as
significant as 20-40%. This issue is critical for pseudo-labeling, as labeled data provides the
supervision signal for pseudo-labels. Hence, issues in the labeled data will affect the pseudo-
labels and the predictive model (see Fig. 1). Mechanisms to address this issue are essential
to improve pseudo-labeling. It might appear possible to manually inspect the data to identify
errors in the labeled set. However, this requires domain expertise and is human-intensive,
especially in modalities such as tabular data where inspecting rows in a spreadsheet can be
much more challenging than reviewing an image. In other cases, updating labels might be
infeasible due to rerunning costly experiments in domains such as biology and physics, or
indeed impossible due to lack of access to either the underlying sample or equipment.
Extending the pseudo-labeling machinery. To solve this fundamental challenge,
we propose a novel framework to extend the pseudo-labeling machinery called Data-centric
Insights for Pseudo-labeling with Selection (DIPS). DIPS focuses on the labeled and pseudo-
labeled data to characterize and select the most useful samples. We instantiate DIPS based
on learning dynamics — the behavior of individual samples during training. We analyze
the dynamics by computing two metrics, confidence and aleatoric (data) uncertainty, which
enables the characterization of samples as Useful or Harmful, guiding sample selection
for model training. Sec. 5 empirically shows that this selection improves pseudo-labeling
performance in multiple real-world settings.
2

[CAPTION] Figure 1: (Left) Current pseudo-labeling formulations implicitly assume that the labeled


<!-- page 3 -->
You can’t handle the (dirty) truth: Data-centric insights improve pseudo-labeling
Beyond performance, DIPS is also specifically designed to be a flexible solution that easily
integrates with existing pseudo-labeling approaches, having the following desired properties:
(P1) Plug & Play: applicable on top of any pseudo-labeling method (to improve it).
(P2) Model-agnostic data characterization: agnostic to any class of supervised
backbone models trained in an iterative scheme (e.g. neural networks, boosting methods).
(P3) Computationally cheap: minimal computational overhead to be practically
usable.
Contributions: 1
○Conceptually, we propose a rethinking of pseudo-labeling, demon-
strating the importance of characterizing and systematically selecting data from both the
labeled and pseudo-labeled datasets, in contrast to the current focus only on the unlabeled
data.
2
○Technically, we introduce DIPS, a novel framework to characterize and select
the most useful samples for pseudo-labeling. This extends the pseudo-labeling machinery
to address the unrealistic status quo of considering the labeled data as gold standard.
3
○
Empirically, we show the value of taking into account labeled data quality, with DIPS’s
selection mechanism improving various pseudo-labeling baselines, both in terms of perfor-
mance and data efficiency, which we demonstrate across 18 real-world datasets, spanning
both tabular data and images. This highlights the usefulness and applicability of DIPS.
2 Related work
Semi-supervised learning and pseudo-labeling methods. Semi-supervised learning
leverages unlabeled data to combat the scarcity of labeled data (Zhu, 2005; Chapelle et al.,
2006; van Engelen and Hoos, 2019; Iscen et al., 2019; Berthelot et al., 2019). As mentioned
in Sec. 1, we focus on pseudo-labeling approaches, given their applicability across data
modalities and competitive performance. Recent methods have extended pseudo-labeling by
modifying the selection mechanism of unlabeled data (Lee et al., 2013; Rizve et al., 2021;
Nguyen et al., 2022a; Tai et al., 2021), using curriculum learning (Cascante-Bonilla et al.,
2020), or merging pseudo-labeling with consistency loss-focused regularization (Sohn et al.,
2020). A commonality among these works is a focus on ensuring the correct selection of
the unlabeled data, assuming a gold standard labeled data. In contrast, DIPS addresses the
question: “What if the labeled data is not gold standard?”, extending the aforementioned
approaches to be more performant.
Self-supervised learning. In addition to semi-supervised learning, there exist other
paradigms to leverage unlabeled data. For example, self-supervised learning is a popular
technique to learn representations from large unlabeled datasets. It has been widely used
in different modalities, e.g. computer vision (Oquab et al., 2023; Henaff, 2020; Chen et al.,
2020), text (Devlin et al., 2019; Radford et al., 2019) and tabular data (Yoon et al., 2020;
Lee et al., 2021; Seedat et al., 2023c). Similarly to consistency regularization, self-supervised
learning is often specific to each modality, typically requires large quantities of unlabeled
data, and only incorporates labeled data separately (e.g. during fine-tuning). This contrasts
pseudo-labeling, which is a versatile and general semi-supervised approach applicable across
modalities, and which uses labeled data in combination with unlabeled data.
Data-centric AI. Data-centric AI has emerged focused on developing systematic methods
to improve the quality of data (Liang et al., 2022; Seedat et al., 2023b). One aspect is to
3


<!-- page 4 -->
Seedat, Huynh, Imrie and van der Schaar
score data samples based on their utility for a task, or whether samples are easy or hard to
learn (Seedat et al., 2023a), then enabling the curation or sculpting of high-quality datasets
for training efficiency purposes (Paul et al., 2021) or improved performance (Liang et al.,
2022). Typically, the goal is to identify mislabeled, hard, or ambiguous examples, with
methods differing based on metrics including uncertainty (Swayamdipta et al., 2020; Seedat
et al., 2022), logits (Pleiss et al., 2020), gradient norm (Paul et al., 2021), or variance of
gradients (Agarwal et al., 2022). We note two key aspects: (1) we draw inspiration from
their success in the fully supervised setting (where there are large amounts of labeled data)
and bring the idea to the semi-supervised setting where we have unlabeled data but scarce
labeled data; (2) many of the discussed supervised methods are only applicable to neural
networks, relying on gradients or logits. Hence, they are not broadly applicable to any model
class, such as boosting methods which are predominant in tabular settings (Borisov et al.,
2021; Grinsztajn et al., 2022). This violates P2: Model-agnostic data characterization.
Learning with Noisy Labels (LNL). LNL typically operates in the supervised setting
and assumes access to a large amount of labeled data. This contrasts the semi-supervised
setting, where labeled data is scarce, and is used to output pseudo-labels for unlabeled data.
Some LNL methods alter a loss function, e.g. adding a regularization term (Cheng et al.,
2021; Wei et al., 2022b). Other methods select samples using a uni-dimensional metric, the
most common being the small-loss criterion in the supervised setting (Xia et al., 2021). DIPS
contrasts these approaches by taking into account both confidence and aleatoric uncertainty
in its selection process. While the LNL methods have not been used in the semi-supervised
setting previously, we repurpose them for this setting and experimentally highlight the value
of the curation process of DIPS in Appendix C. Interestingly, pseudo-labeling can also be used
as a tool in the supervised setting to relabel points identified as noisy by treating them as
unlabeled (Li et al., 2019); however, this contrasts DIPS in two key ways: (1) data availability:
these works operate only on large labeled datasets, whereas DIPS operates with a small
labeled and large unlabeled dataset. (2) application: these works use pseudo-labeling as a
tool for supervised learning, whereas DIPS extends the machinery of pseudo-labeling itself.
3 Background
We now give a brief overview of pseudo-labeling as a general paradigm of semi-supervised
learning. We then highlight that the current formulation of pseudo-labeling overlooks the
key notion of labeled data quality, which motivates our approach.
3.1
Semi-supervised learning via pseudo-labeling
Semi-supervised learning addresses the scarcity of labeled data by leveraging unlabeled data.
The natural question it answers is: how can we combine labeled and unlabeled data to boost
the performance of a model, compared to training on the small labeled data alone?
Notation. Consider a classification setting where we have a labeled dataset Dlab =
{(xi, yi)|i ∈[nlab]} as well as an unlabeled dataset Dunlab = {x′
j|j ∈[nunlab]}. We typically
assume that nlab ≪nunlab. Moreover, the labels take values in {0, 1}C, where C is the
number of classes. This encompasses both binary (C = 2) and multi-label classification. Our
goal is to learn a predictive model f : x →y which leverages Dunlab in addition to Dlab, such
4


<!-- page 5 -->
You can’t handle the (dirty) truth: Data-centric insights improve pseudo-labeling
that it performs better than a model trained on the small labeled dataset Dlab alone. For
all k ∈[C], the k-th coordinate of f(x) is denoted as [f(x)]k. It is assumed to be in [0, 1],
which is typically the case after a softmax layer.
Pseudo-labeling. Pseudo-labeling (PL) is a powerful and general-purpose semi-supervised
approach that answers the pressing question of how to incorporate Dunlab in the learning
procedure. PL is an iterative procedure that spans T iterations and constructs a succession
of models f(i), for i = 1, ..., T. The result of this procedure is the last model f(T), which
issues predictions at test time. The idea underpinning PL is to gradually incorporate Dunlab
into Dlab to train the classifiers f(i). At each iteration i of pseudo-labeling, two steps are
conducted in turn. Step 1: The model f(i) is first trained with supervised learning. Step 2:
f(i) then pseudo-labels unlabeled samples, a subset of which are selected to expand the train-
ing set of the next classifier f(i+1). The key to PL is the construction of these training sets.
More precisely, let us denote D(i)
train the training set used to train f(i) at iteration i. D(i)
train is
defined by an initial condition, D(1)
train = Dlab, and by the following recursive equation: for all
i = 1, ..., T −1, D(i+1)
train = D(i)
train ∪s(Dunlab, f(i)), where s is a selector function. Alternatively
stated, f(i) outputs pseudo-labels for Dunlab at iteration i and the selector function s then
selects a subset of these pseudo-labeled samples, which are added to D(i)
train to form D(i+1)
train .
Common heuristics define s with metrics of confidence and/or uncertainty (e.g. greedy-PL
(Lee et al., 2013), UPS (Rizve et al., 2021)). More details are given in Appendix A regarding
the exact formulation of s in those cases.
3.2
Overlooked aspects in the current formulation of pseudo-labeling
Having introduced the pseudo-labeling paradigm, we now show that its current formulation
overlooks several key elements that will motivate our approach.
First, the selection mechanism s only focuses on unlabeled data and ignores labeled data.
This implies that the labeled data is considered "perfect". This assumption is not reasonable
in many real-world settings where labeled data is noisy. In such situations, as shown in Fig.
1, noise propagates to the pseudo-labels, jeopardizing the accuracy of the pseudo-labeling
steps (Nguyen et al., 2022a). To see why such propagation of error happens, recall that
D(1)
train = Dlab. Alternatively stated, Dlab provides the initial supervision signal for PL and
its recursive construction of D(i)
train.
Second, PL methods do not update the pseudo-labels of unlabeled samples once they
are incorporated in one of the D(i)
train.
However, the intuition underpinning PL is that
the classifiers f(i) progressively get better over the iterations, meaning that pseudo-labels
computed at iteration T are expected to be more accurate than pseudo-labels computed at
iteration 1, since f(T) is the output of PL.
Taken together, these two observations shed light on an important yet overlooked aspect
of current PL methods: the selection mechanism s ignores labeled and previously pseudo-
labeled samples. This naturally manifests in the asymmetry of the update rule D(i+1)
train =
D(i)
train ∪s(Dunlab, f(i)), where the selection function s is only applied to unlabeled data and
ignores D(i)
train at iteration i + 1.
5


<!-- page 6 -->
Seedat, Huynh, Imrie and van der Schaar
4 DIPS: Data-centric insights for improved pseudo-labeling
In response to these overlooked aspects, we propose a new formulation of pseudo-labeling,
DIPS, with the data-centric aim to characterize the usefulness of both labeled and pseudo-
labeled samples. We then operationalize this framework with the lens of learning dynamics.
Our goal is to improve the performance of any pseudo-labeling algorithm by selecting useful
samples to be used for training.
4.1
A data-centric formulation of pseudo-labeling
Motivated by the asymmetry in the update rule of D(i)
train, as defined in Sec. 3.1, we propose
DIPS, a novel framework which explicitly focuses on both labeled and pseudo-labeled samples.
The key idea is to introduce a new selection mechanism, called r, while still retaining the
benefits of s. For any dataset D and classifier f, r(D, f) defines a subset of D to be used for
training in the current pseudo-labeling iteration. More formally, we define the new update
rule (Eq. 1) for all i = 1, ..., T −1 as:
(
D(i+1)
=
D(i) ∪s(Dunlab, f(i))
▷Original PL formulation
D(i+1)
train
=
r(D(i+1), f(i))
▷DIPS selection
(1)
Then, let D(1) = D(1)
train = r(Dlab, f(0)), where f(0) is a classifier trained on Dlab only. The
selector r selects samples from D(i+1), producing D(i+1)
train , the training set of f(i+1).
This new formulation addresses the challenges mentioned in Sec. 3.2. Indeed, for any
j < i, r selects samples in D(j) at any iteration i (which includes labeled samples), since
D(j) ⊂D(i) for all i = 0, ..., T. We investigate in Appendix C.1 alternative ways to incorporate
the selector r, showing the value of the DIPS formulation.
Finally, notice that DIPS subsumes current pseudo-labeling methods via its selector r.
To see that, we note current pseudo-labeling methods define an identity selector r, selecting
all samples, such that for any D and function f, we have r(D, f) = D. Hence, DIPS goes
beyond this status quo by permitting a non-identity selector r.
4.2
Operationalizing DIPS using learning dynamics
We now explicitly instantiate DIPS by constructing the selector r. Our key idea is to define r
using learning dynamics of samples. Before giving a precise definition, let us detail some
context. Prior works in learning theory have shown that the learning dynamics of samples
contain a useful signal about the nature of samples (and their usefulness) for a specific task
(Arpit et al., 2017; Arora et al., 2019; Li et al., 2020). Some samples may be easier for a
model to learn, whilst for other samples, a model might take longer to learn (i.e. more
variability through training) or these samples might not be learned correctly during the
training process. We build on this insight about learning dynamics, bringing the idea to the
semi-supervised setting.
Our construction of r has three steps. First, we analyze the learning dynamics of labeled
and pseudo-labeled samples to define two metrics: (i) confidence and (ii) aleatoric uncertainty,
which captures the inherent data uncertainty. Second, we use these metrics to characterize
the samples as Useful or Harmful. Third, we select Useful samples for model training, which
gives our definition of r.
6


**[Table p6.1]**
| ▷ Original PL formulation |  |
| --- | --- |
|  | ▷ DIPS selection |


<!-- page 7 -->
You can’t handle the (dirty) truth: Data-centric insights improve pseudo-labeling
Stage 1: Data characterization
Stage 2: Pseudo-labeling
⚙️
Uncurated data
Discard harmful 
samples
Improved pseudo-labels
Train model
Epochs
Probability -true label
Learning dynamics
Figure 2: Stage 1 operationalizes DIPS by leveraging learning dynamics of individual labeled
and pseudo-labeled samples to characterize them as Useful or Harmful. Only Useful samples
are then kept for Stage 2, which consists of pseudo-labeling, using any off-the-shelf method.
For any i, we assume that the classifier f(i) at iteration i of PL is trained in an iterative
scheme (e.g. neural networks or XGBoost trained over iterations), which is ubiquitous in
practice. This motivates and makes it possible to analyze the learning dynamics as a way to
characterize individual samples. For clarity of presentation, we consider binary classification
(C = 2) and denote f(i) = f.
At any pseudo-labeling iteration, f is trained from scratch and goes through e ∈[E]
different checkpoints leading to the set F = {f1, f2, . . . , fE}, such that fe is the classifier at
the e-th checkpoint. Our goal is to assess the learning dynamics of samples over these E
training checkpoints. For this, we define H, a random variable following a uniform distribution
UF over the set of checkpoints F. Specifically, given H = h and a sample (x, y), where y is
either a provided label (x ∈Dlab) or a pseudo-label (x ∈Dunlab), we define the correctness
in the prediction of H as a binary random variable ˆYF(x, y) with the following conditional:
P( ˆYF(x, y) = 1|H = h) = [h(x)]y and P( ˆYF(x, y) = 0|H = h) = 1 −P( ˆYF(x, y) = 1|H = h).
Equipped with a probabilistic interpretation of the predictions of a model, we now define
our characterization metrics: (i) average confidence and (ii) aleatoric (data) uncertainty,
inspired by (Kwon et al., 2020; Seedat et al., 2022).
Definition 4.1 (Average confidence). For any set of checkpoints F = {f1, ..., fE}, the
average confidence for a sample (x, y) is defined as the following marginal:
¯PF(x, y) := P( ˆYF(x, y) = 1) = EH∼UF[P( ˆYF(x, y) = 1|H)] = 1
E
E
X
e=1
[fe(x)]y
(2)
Definition 4.2 (Aleatoric uncertainty). For any set of checkpoints F = {f1, ..., fE}, the
aleatoric uncertainty for a sample (x, y) is defined as:
val,F(x, y) := EH∼UF[V ar( ˆYF(x, y)|H)] = 1
E
E
X
e=1
[fe(x)]y(1 −[fe(x)]y)
(3)
Intuitively, the aleatoric uncertainty for a sample x is maximized when [fe(x)]y = 1
2
for all checkpoints fe, akin to random guessing. Recall aleatoric uncertainty captures the
inherent data uncertainty, hence is a principled way to capture issues such as mislabeling.
7

[CAPTION] Figure 2: Stage 1 operationalizes DIPS by leveraging learning dynamics of individual labeled


<!-- page 8 -->
Seedat, Huynh, Imrie and van der Schaar
This contrasts epistemic uncertainty, which is model-dependent and can be reduced simply
by increasing model parameterization (Hüllermeier and Waegeman, 2021).
We emphasize that this definition of uncertainty is model-agnostic, satisfying P2: Model-
agnostic data characterization, and only relies on having checkpoints through training.
Hence, it comes for free, unlike ensembles (Lakshminarayanan et al., 2017). This fulfills P3:
Computationally cheap. Moreover, it is applicable to any iteratively trained model (e.g.
neural networks and XGBoost) unlike approaches such as MC-dropout or alternative training
dynamic metrics using gradients (Paul et al., 2021) or logits (Pleiss et al., 2020). We note
that confidence and uncertainty are defined as averages over all the training checkpoints, in
order to capture the full learning trajectories of samples. We show in Appendix C.6 that
ignoring earlier training checkpoints is suboptimal, highlighting that earlier checkpoints carry
valuable signal for data characterization.
4.3
Defining the selector r: data characterization and selection
Having defined sample-wise confidence and aleatoric uncertainty, we characterize both labeled
and pseudo-labeled samples into two categories, namely Useful and Harmful. Given a sample
(x, y), a set of training checkpoints F, and two thresholds τconf and τal, we define the category
c(x, y, F) as Useful if ¯PF(x, y) ≥τconf and val,F(x, y) < τal, and Harmful otherwise.
Hence, a Useful sample is one where we are highly confident in predicting its asso-
ciated label and for which we also have low inherent data uncertainty.
In contrast, a
harmful sample would have low confidence and/or high data uncertainty. Finally, given
a function f whose training led to the set of checkpoints F, we can define r explicitly by
r(D, f) = {(x, y) | (x, y) ∈D, c(x, y, F) = Useful}.
4.4
Combining DIPS with any pseudo-labeling algorithm
Algorithm 1 Plug DIPS into any pseudo-labeler
1: Train a network, f (0), using the samples from Dlab.
2:
Plug-in DIPS: set D(1)
train = D(1) = r(Dlab, f (0))
3: for t = 1..T do
4:
Initialize new network f (t)
5:
Train f (t) using D(t)
train.
6:
Pseudo-label Dunlab using f (t)
7:
Define D(t+1) using the PL method’s selector s
8:
Plug-in DIPS : Define D(t+1)
train = r(D(t+1), f (t))
▷Data
characterization and selection, Sec. 4.3
9: end for
10: return fT
We outline the integration of DIPS
into any pseudo-labeling algorithm
as per Algorithm 1 (see Appendix
A). A fundamental strength of
DIPS lies in its simplicity. The com-
putational overhead is also small
(no extra model training and stor-
ing checkpoints is not required) –
i.e.
satisfying P3:
Computa-
tionally cheap, with only min-
imal overhead on forward passes.
These are negligible compared to
the pseudo-labeling process in gen-
eral. Additionally, DIPS is easily integrated into any pseudo-labeling algorithm – i.e. P1:
Plug & Play, making for easier adoption.
8


<!-- page 9 -->
You can’t handle the (dirty) truth: Data-centric insights improve pseudo-labeling
5 Experiments
We now empirically investigate multiple aspects of DIPS1. We discuss the setup of each
experiment at the start of each sub-section, with further experimental details in Appendix B.
1. Characterization: Does it matter? Sec. 5.1 analyzes the effect of not characterizing
and selecting samples in all D(i)
train in a synthetic setup, where noise propagates from Dlab
to Dunlab.
2. Performance: Does it work? Sec. 5.2 shows characterizing D(i)
train using DIPS improves
performance of various state-of-the-art pseudo-labeling baselines on 12 real-world datasets.
3. Narrowing the gap: Can selection reduce performance disparities? Sec. 5.2
shows that DIPS also renders the PL methods more comparable to one other.
4. Data efficiency: Can similar performance be achieved with less labeled data?
Sec. 5.3 studies the efficiency of data usage of vanilla methods vs. DIPS on different
proportions of labeled data.
5. Selection across countries: Can selection improve performance when using
data from a different country? Sec. 5.4 assesses the role of selection of samples when
Dlab and Dunlab come from different countries in a clinically relevant task.
6. Other modalities: Sec. 5.5 shows the potential to use DIPS in image experiments.
Hence, the experiments will demonstrate the core purpose of DIPS, which is to address
the overlooked issue of labeled data quality in pseudo-labeling, validating DIPS as an effective
framework to improve different pseudo-labelers.
5.1
Synthetic example: Data characterization and unlabeled data improve test
accuracy
Goal. To motivate DIPS, we demonstrate (1) label noise harms pseudo-labeling and (2) char-
acterizing and selecting data using DIPS consequently improves pseudo-labeling performance.
Setup. We consider a synthetic setup with two quadrants (Lee et al., 2023), as illustrated
in Fig. 3b 2. We sample data in each of the two quadrants from a uniform distribution, and
each sample is equally likely to fall in each quadrant. To mimic a real-world scenario of label
noise in Dlab, we randomly flip labels with varying proportions pcorrupt ∈[0.1, 0.45]
Baselines. We compare DIPS with two baselines (i) Supervised which trains a classifier
using the initial Dlab (ii) Greedy pseudo-labeling (PL) (Lee et al., 2013) which uses
both Dlab and Dunlab. We use an XGBoost backbone for all the methods and we combine
DIPS with PL for a fair comparison. We also instantiate our DIPS framework with other
possible selectors from the noisy label literature namely the small-loss criterion (Xia et al.,
2021), Fluctuation (Wei et al., 2022b) and FINE (Kim et al., 2021), thereby contrasting
DIPS’ selector based on learning dynamics with these other selectors.
Results. Test performance over 20 random seeds with different data splits and nlab =
100, nunlab = 900 is illustrated in Fig. 3c, for varying pcorrupt. It highlights two key elements.
First, PL barely improves upon the supervised baseline. The noise in the labeled dataset
1. https://github.com/seedatnabeel/DIPS or https://github.com/vanderschaarlab/DIPS
2. Notice that the two quadrant setup satisfies the cluster assumption inherent to the success of semi-
supervised learning (Chapelle et al., 2006).
9


<!-- page 10 -->
Seedat, Huynh, Imrie and van der Schaar
Unlabeled
Class 1
Class 2
(a) PL: The inherent noise in Dlab
propagates when assigning pseudo-
labels.
(b) PL+DIPS: DIPS mitigates the issue
of noise by selecting useful samples.
0.0
0.1
0.2
0.3
0.4
Proportion of corrupted samples
60
70
80
90
100
Test accuracy
Supervised
PL
PL+DIPS (Ours)
(c) DIPS selection mechanism signifi-
cantly improves test performance un-
der label noise.
0.10
0.15
0.20
0.25
0.30
0.35
0.40
0.45
Proportion of corrupted samples
60
70
80
90
100
Test accuracy
PL+FINE
PL+Small-Loss
PL+Fluctuation
DIPS (Ours)
(d) DIPS selection mechanism outper-
forms other data-centric insights un-
der label noise.
Figure 3: (a)-(b) The colored dots illustrate the selected labeled and pseudo-labeled samples
for the last iteration of PL and PL+DIPS, with 30% label noise. Grey dots are unselected
unlabeled samples. (c) Characterizing and selecting data for the semi-supervised algorithm
yields the best results (epitomized by PL+DIPS) and makes the unlabeled data impactful.
(d) Characterizing and selecting data via DIPS outperforms other data-centric mechanisms
propagates to the unlabeled dataset, via the pseudo-labels, as shown in Fig. 3a. This
consequently negates the benefit of using the unlabeled data to learn a better classifier,
which is the original motivation of semi-supervised learning. Second, DIPS mitigates this
issue via its selection mechanism and improves performance by around +20% over the two
baselines when the amount of label noise is around 30%. We also conduct an ablation study
in Appendix C to understand when in the pseudo-labeling pipeline to apply DIPS.
Takeaway. The results emphasize the key motivation of DIPS: labeled data quality is
central to the performance of the pseudo-labeling algorithms because labeled data drives the
learning process necessary to perform pseudo-labeling. Hence, careful consideration of Dlab
is crucial to performance.
10


**[Table p10.1]**
| 100 90 accuracy 80 Test 70 Supervised 60 PL PL+DIPS (Ours) 0.0 0.1 0.2 0.3 0.4 Proportion of corrupted samples |  |  |
| --- | --- | --- |
|  |  |  |
|  |  |  |
|  | Supervised |  |
|  | Supervised |  |
|  | PL PL+DIPS (Ours) |  |


**[Table p10.2]**
| 100 90 accuracy 80 Test 70 PL+FINE PL+Small-Loss 60 PL+Fluctuation DIPS (Ours) 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 Proportion of corrupted samples |  |  |  |
| --- | --- | --- | --- |
|  |  |  |  |
|  |  |  |  |
|  | PL+FINE PL+Small-Loss |  |  |
|  | PL+FINE PL+Small-Loss |  |  |
|  | PL+Fluctuation DIPS (Ours) |  |  |

[CAPTION] Figure 3: (a)-(b) The colored dots illustrate the selected labeled and pseudo-labeled samples


<!-- page 11 -->
You can’t handle the (dirty) truth: Data-centric insights improve pseudo-labeling
5.2
DIPS improves different pseudo-labeling algorithms across 12 real-world
tabular datasets.
Goal. We evaluate the effectiveness of DIPS on 12 different real-world tabular datasets
with diverse characteristics (sample sizes, number of features, task difficulty). We aim to
demonstrate that DIPS improves the performance of various pseudo-labeling algorithms. We
focus on the tabular setting, as pseudo-labeling plays a crucial role in addressing data scarcity
issues in healthcare and finance, discussed in Sec. 1, where data is predominantly tabular
(Borisov et al., 2021; Shwartz-Ziv and Armon, 2022). Moreover, enhancing our capacity to
improve models for tabular data holds immense significance, given its ubiquity in real-world
applications. For perspective, nearly 79% of data scientists work with tabular data on a
daily basis, compared to only 14% who work with modalities such as images (Kaggle, 2017).
This underlines the critical need to advance pseudo-labeling techniques in the context of
impactful real-world tabular data.
PL
UPS Flex SLA CSA
81
82
83
Accuracy
seer
PL
UPS Flex SLA CSA
81
82
83
adult
PL
UPS Flex SLA CSA
62.5
65.0
67.5
cutract
PL
UPS Flex SLA CSA
67.5
70.0
covid
PL
UPS Flex SLA CSA
66
68
Accuracy
maggic
PL
UPS Flex SLA CSA
60
65
compas
PL
UPS Flex SLA CSA
62.5
65.0
agaricus-lepiota
PL
UPS Flex SLA CSA
67.5
70.0
German-credit
PL
UPS Flex SLA CSA
81
82
Accuracy
higgs
PL
UPS Flex SLA CSA
75.0
77.5
drug
PL
UPS Flex SLA CSA
79
80
blog
PL
UPS Flex SLA CSA
78
80
credit
Supervised
Vanilla
DIPS (Ours)
Figure 4: DIPS consistently improves the performance of all five pseudo-labeling methods
across the 12 real-world datasets. DIPS also reduces the performance gap between the different
pseudo-labelers.
Datasets. The tabular datasets are drawn from a variety of domains (e.g. healthcare,
finance), mirroring Sec. 1, where the issue of limited annotated examples is highly prevalent.
It is important to note that the vast majority of the datasets (10/12) are real-world datasets,
demonstrating the applicability of DIPS and its findings in practical scenarios. For example,
Covid-19 (Baqui et al., 2020), MAGGIC (Pocock et al., 2013), SEER (Duggan et al., 2016),
and CUTRACT (Prostate Cancer PCUK, 2019) are medical datasets. COMPAS (Angwin
et al., 2016) is a recidivism dataset. Credit is a financial default dataset from a Taiwan bank
(Yeh and Lien, 2009). Higgs is a physics dataset (Baldi et al., 2014). The datasets vary
significantly in both sample size (from 1k to 41k) and number of features (from 12 to 280).
More details on the datasets can be found in Table 1, Appendix B.
Baselines. As baselines, we compare: (i) Supervised training on the small Dlab, (ii)
five state-of-the-art pseudo-labeling methods applicable to tabular data: greedy-PL (Lee
et al., 2013), UPS (Rizve et al., 2021), FlexMatch (Zhang et al., 2021), SLA (Tai et al.,
2021), CSA (Nguyen et al., 2022a). For each of the baselines, we apply DIPS as a plug-in to
improve performance.
11


**[Table p11.1]**
| Supervised Vanilla DIPS (Ours) seer adult cutract covid 83 83 70.0 Accuracy 82 67.5 82 81 65.0 67.5 81 62.5 PL UPS Flex SLA CSA PL UPS Flex SLA CSA PL UPS Flex SLA CSA PL UPS Flex SLA CSA maggic compas agaricus-lepiota German-credit 68 65 Accuracy 65.0 70.0 66 67.5 60 62.5 PL UPS Flex SLA CSA PL UPS Flex SLA CSA PL UPS Flex SLA CSA PL UPS Flex SLA CSA higgs drug blog credit 82 77.5 80 Accuracy 80 81 75.0 79 78 PL UPS Flex SLA CSA PL UPS Flex SLA CSA PL UPS Flex SLA CSA PL UPS Flex SLA CSA |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  |  | L U G | PS F erma | lex S n-cred | LA C it | SA |
|  |  |  |  |  |  |  |
|  |  | L U | PS F cre | lex S dit | LA C | SA |
|  |  |  |  |  |  |  |
|  |  | L U | PS F | lex S | LA C | SA |


**[Table p11.2]**
|  | L U | PS F ma | lex S ggic | LA C |  |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |
|  | L U | PS F hig | lex S gs | LA C |  |
|  |  |  |  |  |  |
|  | L U | PS F | lex S | LA C |  |


**[Table p11.3]**
|  | L U | PS F com | lex S pas | LA C |  |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |
|  | L U | PS F dr | lex S ug | LA C |  |
|  |  |  |  |  |  |
|  | L U | PS F | lex S | LA C |  |


**[Table p11.4]**
|  | L U ag | PS F aricu | lex S s-lepio | LA C ta |  |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |
|  | L U | PS F bl | lex S og | LA C |  |
|  |  |  |  |  |  |
|  | L U | PS F | lex S | LA C |  |

[CAPTION] Figure 4: DIPS consistently improves the performance of all five pseudo-labeling methods


<!-- page 12 -->
Seedat, Huynh, Imrie and van der Schaar
Results We report results in Fig. 4 across 50 random seeds with different data splits
with a fixed proportion of Dlab : Dunlab of 0.1:0.9. We note several findings from Fig. 4 perti-
nent to DIPS.
■DIPS improves the performance of almost all baselines across various real-
world datasets.
We showcase the value of data characterization and selection to improve SSL performance.
We demonstrate that DIPS consistently boosts the performance when incorporated with ex-
isting pseudo-labelers. This illustrates the key motivation of our work: labeled data is of
critical importance for pseudo-labeling and calls for curation, in real-world scenarios.
■DIPS reduces the performance gap between pseudo-labelers.
Fig. 4 shows the reduction in variability of performance across pseudo-labelers by introducing
data characterization. On average, we reduce the average variance across all datasets and
algorithms from 0.46 in the vanilla case to 0.14 using DIPS. In particular, we show that the
simplest method, namely greedy pseudo-labeling (Lee et al., 2013), which is often the worst in
the vanilla setups, is drastically improved simply by incorporating DIPS, making it competitive
with the more sophisticated alternative baselines. This result of equalizing performance
is important as it directly influences the process of selecting a pseudo-labeling algorithm.
We report additional results in Appendix C.2 where we replace the selector r with sample
selectors from the LNL literature, highlighting the advantage of using learning dynamics.
Takeaway. We have empirically demonstrated improved performance by DIPS across
multiple pseudo-labeling algorithms and multiple real-world datasets.
5.3
DIPS improves data efficiency
(a) Pseudo-Labeling
(b) UPS
Figure 5: DIPS (pink) improves data efficiency of vanilla
methods (green), achieving the same level of performance
with 60-70% fewer labeled examples, as shown by the vertical
dotted lines. The results (a) Pseudo-labeling and (b) UPS are
averaged across datasets and show gains in accuracy vs. the
maximum performance of the vanilla method. Additionally,
DIPS selection generally provides additional efficiency gains
over other possible selection mechanisms.
Goal. In real-world scenar-
ios, collecting labeled data is
a significant bottleneck, hence
it is traditionally the sole fo-
cus of semi-supervised bench-
marks. The goal of this exper-
iment is to demonstrate that
data quality is an overlooked
dimension that has a direct
impact on data quantity re-
quirements to achieve a given
test performance for pseudo-
labeling.
Setup. For clarity, we fo-
cus on greedy-PL and UPS
as pseudo-labeling algorithms.
To assess data efficiency, we
consider subsets of Dlab with size p · |Dlab|, with p going from 0.1 to 1. We also instantiate
12

[CAPTION] Fig. 4 shows the reduction in variability of performance across pseudo-labelers by introducing

[CAPTION] Figure 5: DIPS (pink) improves data efficiency of vanilla


<!-- page 13 -->
You can’t handle the (dirty) truth: Data-centric insights improve pseudo-labeling
our DIPS framework with the small-loss criterion (Xia et al., 2021), Fluctuation (Wei et al.,
2022b) and FINE (Kim et al., 2021) 3.
Results. The results in Fig. 5, averaged across datasets, show the performance gain in
accuracy for all p compared to the maximum performance of the vanilla method (i.e. when
p = 1). We conclude that DIPS significantly improves the data efficiency of the vanilla pseudo-
labeling baselines, between 60-70% more efficient for UPS and greedy-PL respectively, to
reach the same level of performance.
Takeaway. We have demonstrated that data quantity is not the sole determinant of
success in pseudo-labeling. We reduce the amount of data needed to achieve a desired test
accuracy by leveraging the selection mechanism of DIPS. This highlights the significance of a
multi-dimensional approach to pseudo-labeling, where a focus on quality reduces the data
quantity requirements.
5.4
DIPS improves performance of cross-country pseudo-labeling
Goal. To further assess the real-world benefit of DIPS, we consider the clinically relevant
task of improving classifier performance using data from hospitals in different countries.
Figure 6: Curation of Dlab per-
mits us to better leverage a cross-
country Dunlab
Setup. We assess a setup using Prostate cancer data from
the UK (CUTRACT (Prostate Cancer PCUK, 2019)) to define
(Dlab, Dtest), which is augmented by Dunlab, from US data
(SEER (Duggan et al., 2016)). While coming from different
countries, the datasets have interoperable features and the task
is to predict prostate cancer mortality.
We leverage the unlabeled data from the US to augment
the small labeled dataset from the UK, to improve the classifier
when used in the UK (on Dtest). We also instantiate our DIPS
framework with the small-loss criterion (Xia et al., 2021), Fluctuation (Wei et al., 2022b)
and FINE (Kim et al., 2021)
Results. Fig. 6 illustrates that greedy-PL and UPS benefit from DIPS’s selection of
labeled and pseudo-labeled samples, resulting in improved test performance. Hence, this
result underscores that ignoring the labeled data whilst also naively selecting pseudo-labeled
samples simply using confidence scores (as in greedy-PL) yields limited benefit. We provide
further insights into the selection and gains by DIPS in Appendix C.
Takeaway. DIPS’s selection mechanism improves performance when using semi-supervised
approaches across countries.
5.5
DIPS works with other data modalities
Goal. While DIPS is mainly geared towards the important problem of pseudo-labeling for
tabular data, we explore an extension of DIPS to images, highlighting its versatility.
Setup. We investigate the use of DIPS to improve pseudo-labeling for CIFAR-10N (Wei
et al., 2022a). With realism in mind, we specifically use this dataset as it reflects noise in
image data stemming from real-world human annotations on M-Turk, rather than synthetic
noise models (Wei et al., 2022a).
3. PL+FINE is below -6% and hence not shown for clarity of visuals
13

[CAPTION] Figure 6: Curation of Dlab per-


<!-- page 14 -->
Seedat, Huynh, Imrie and van der Schaar
100k
500k
1000k
Iterations
50
55
60
65
70
75
80
85
90
Mean Accuracy
FixMatch
FixMatch+SmallLoss
FixMatch+DIPS
Figure 7:
DIPS improves Fix-
Match on CIFAR-10N, and also
outperforms the selector based
on the small-loss heuristic.
We evaluate the semi-supervised algorithm FixMatch (Sohn
et al., 2020) with a WideResNet-28 (Zagoruyko and Komodakis,
2016) for nlab = 1000 over three seeds. FixMatch combines
pseudo-labeling with consistency regularization, hence does not
apply to the previous tabular data-focused experiments.
We incorporate DIPS as a plug-in to the pseudo-labeling
component of FixMatch.
Results. Fig. 7 showcases the improved test accuracy for
FixMatch+DIPS of 85.2% over vanilla FixMatch of 82.6%. A
key reason is that DIPS discards harmful mislabeled samples
from Dlab, with example images shown in Fig. 8b. Fig. 7 also
compares DIPS with the small-loss criterion, showing the supe-
rior performance of DIPS’ selection mechanism. Furthermore,
Table 8a shows the addition of DIPS improves time efficiency
significantly, reducing the final computation time by 8 hours.
We show additional results for CIFAR-100N in Appendix C.4 and for other image datasets
in Appendix C.4.4.
Takeaway. DIPS is a versatile framework that can be extended to various data modalities.
Test acc (%)
FM + DIPS
FixMatch (FM)
65
2.3 ± 0.4
8.0 ± 2.0
70
4.6 ± 0.5
16.4 ± 3.26
75
10.8 ± 0.7
26.5 ± 1.9
80
27.8 ± 0.8
35.8 ± 0.9
85
38.5 ± 0.3
N.A.
(a)
Noisy label: automobile
True label:   airplane
Noisy label: deer
True label:  horse
Noisy label: dog
True label:  deer
Noisy label: truck
True label:  automobile
(b)
Figure 8: (a) DIPS improves the time efficiency (hours reported on a v100 GPU) of FixMatch,
by 1.5-4X for the same performance (↓better). (b) Examples of mislabeled samples in
CIFAR-10N discarded by DIPS. We note the incorrect labels and ideal ground-truth labels.
6 Discussion
We propose DIPS, a plugin designed to improve any pseudo-labeling algorithm. DIPS builds
on the key observation that the quality of labeled data is overlooked in pseudo-labeling
approaches, while it is the core signal that renders pseudo-labeling possible. Motivated by
real-world datasets and their inherent noisiness, we introduce a cleaning mechanism that
operates both on labeled and pseudo-labeled data. We showed the value of taking into account
labeled data quality – by characterizing and selecting data we improve test performance for
various pseudo-labelers across 18 real-world datasets spanning tabular data and images.
14


**[Table p14.1]**
| 90 FixMatch 85 FixMatch+SmallLoss FixMatch+DIPS 80 Accuracy 75 70 Mean 65 60 55 50 100k 500k 1000k Iterations |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |
|  | 1 | 00 | k | It | 5 er | 00 ati | k on | s | 10 | 0 | 0k |  |
| Figure 7: Match on outperfor on the sm nd for ot | m a h | D CI s ll- e | I F t l r | PS A he oss i | R m | i -1 se he a | m 0 le u g | pr N, ct ri e | o a o st d | ve n r ic at | s d b . a | Fix- also ased sets |


**[Table p14.2]**
| Test acc (%) | FM + DIPS FixMatch (FM) |
| --- | --- |
| 65 | 2.3 ± 0.4 8.0 ± 2.0 |
| 70 | 4.6 ± 0.5 16.4 ± 3.26 |
| 75 | 10.8 ± 0.7 26.5 ± 1.9 |
| 80 | 27.8 ± 0.8 35.8 ± 0.9 |
| 85 | 38.5 ± 0.3 N.A. |

[CAPTION] Figure 7:
DIPS improves Fix-

[CAPTION] Figure 8: (a) DIPS improves the time efficiency (hours reported on a v100 GPU) of FixMatch,


<!-- page 15 -->
You can’t handle the (dirty) truth: Data-centric insights improve pseudo-labeling
Acknowledgments and Disclosure of Funding
NS is supported by the Cystic Fibrosis Trust and NH by Illumina. This work was supported
by Azure sponsorship credits granted by Microsoft’s AI for Good Research Lab.
Broader Impact Statement
In this work, we delve into the essential yet often neglected aspect of labeled data quality in
the application of pseudo-labeling, a semi-supervised learning technique. Our key insights
stem from a data-centric approach that underscores the role of labeled data quality - a facet
typically overlooked due to the default assumption of labeled data being “perfect”. In stark
contrast to the traditional, algorithm-centric pseudo-labeling literature which largely focuses
on refining pseudo-labeling methods, we accentuate the critical influence of the quality of
labeled data on the effectiveness of pseudo-labeling.
By way of introducing the DIPS framework, our work emphasizes the value of characteri-
zation and selection of labeled data, consequently improving any pseudo-labeling method.
Moreover, akin to traditional machine learning problems, focusing on labeled data quality in
the context of pseudo-labeling promises to lessen risks, costs, and potentially detrimental
consequences of algorithm deployment. This perspective opens up many avenues for ap-
plications in areas where labeled data is scarce or expensive to acquire, including but not
limited to healthcare, social sciences, autonomous vehicles, wildlife conservation, and climate
modeling scenarios. Our work underscores the need for a data-centric paradigm shift in the
pseudo-labeling landscape.
15


<!-- page 16 -->
Seedat, Huynh, Imrie and van der Schaar
References
C. Agarwal, D. D’souza, and S. Hooker. Estimating example difficulty using variance of
gradients. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 10368–10378, 2022.
J.
Angwin,
J.
Larson,
L.
Kirchner,
and
S.
Mattu.
Machine
bias.
ProP-
ublica:
https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-
sentencing, May 2016.
S. Arora, S. Du, W. Hu, Z. Li, and R. Wang. Fine-grained analysis of optimization and
generalization for overparameterized two-layer neural networks. In International Conference
on Machine Learning, pages 322–332. PMLR, 2019.
D. Arpit, S. Jastrzębski, N. Ballas, D. Krueger, E. Bengio, M. S. Kanwal, T. Maharaj,
A. Fischer, A. Courville, Y. Bengio, et al. A closer look at memorization in deep networks.
In International Conference on Machine Learning, pages 233–242. PMLR, 2017.
A. Asuncion and D. Newman. UCI machine learning repository, 2007.
P. Baldi, P. Sadowski, and D. Whiteson. Searching for exotic particles in high-energy physics
with deep learning. Nature Communications, 5(1):4308, 2014.
P. Baqui, I. Bica, V. Marra, A. Ercole, and M. van Der Schaar. Ethnic and regional variations
in hospital mortality from COVID-19 in Brazil: a cross-sectional observational study. The
Lancet Global Health, 8(8):e1018–e1026, 2020.
D. Berthelot, N. Carlini, I. Goodfellow, N. Papernot, A. Oliver, and C. A. Raffel. MixMatch:
A holistic approach to semi-supervised learning. Advances in Neural Information Processing
Systems, 32, 2019.
V. Borisov, T. Leemann, K. Seßler, J. Haug, M. Pawelczyk, and G. Kasneci. Deep neural
networks and tabular data: A survey. arXiv preprint arXiv:2110.01889, 2021.
K. Buza. Feedback prediction for blogs. In Data analysis, machine learning and knowledge
discovery, pages 145–152. Springer, 2013.
P. Cascante-Bonilla, F. Tan, Y. Qi, and V. Ordonez. Curriculum labeling: Revisiting pseudo-
labeling for semi-supervised learning. In AAAI Conference on Artificial Intelligence, 2020.
O. Chapelle, B. Schölkopf, and A. Zien, editors. Semi-Supervised Learning. The MIT Press,
2006. ISBN 9780262033589. URL http://dblp.uni-trier.de/db/books/collections/
CSZ2006.html.
T. Chen, S. Kornblith, M. Norouzi, and G. Hinton. A simple framework for contrastive
learning of visual representations. In International Conference on Machine Learning, pages
1597–1607. PMLR, 2020.
H. Cheng, Z. Zhu, X. Li, Y. Gong, X. Sun, and Y. Liu. Learning with instance-dependent label
noise: A sample sieve approach. In International Conference on Learning Representations,
2021.
16


<!-- page 17 -->
You can’t handle the (dirty) truth: Data-centric insights improve pseudo-labeling
J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. Bert: Pre-training of deep bidirectional
transformers for language understanding. In North American Chapter of the Association for
Computational Linguistics, 2019. URL https://api.semanticscholar.org/CorpusID:
52967399.
M. A. Duggan, W. F. Anderson, S. Altekruse, L. Penberthy, and M. E. Sherman. The
surveillance, epidemiology and end results (SEER) program and pathology: towards
strengthening the critical relationship. The American Journal of Surgical Pathology, 40
(12):e94, 2016.
E. Fehrman, A. K. Muhammad, E. M. Mirkes, V. Egan, and A. N. Gorban. The five factor
model of personality and evaluation of drug consumption risk. In Data science: innovative
developments in data analysis and clustering, pages 231–242. Springer, 2017.
S. Gidaris, P. Singh, and N. Komodakis. Unsupervised representation learning by predicting
image rotations. In International Conference on Learning Representations, pages 3938–
3961, 2018.
L. Grinsztajn, E. Oyallon, and G. Varoquaux. Why do tree-based models still outperform
deep learning on typical tabular data? In Thirty-sixth Conference on Neural Information
Processing Systems Datasets and Benchmarks Track, 2022.
N. Gupta, S. Mujumdar, H. Patel, S. Masuda, N. Panwar, S. Bandyopadhyay, S. Mehta,
S. Guttula, S. Afzal, R. Sharma Mittal, et al. Data quality for machine learning tasks.
In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data
Mining, pages 4040–4041, 2021a.
N. Gupta, H. Patel, S. Afzal, N. Panwar, R. S. Mittal, S. Guttula, A. Jain, L. Nagalapatti,
S. Mehta, S. Hans, et al. Data quality toolkit: Automatic assessment of data quality and
remediation for machine learning datasets. arXiv preprint arXiv:2108.05935, 2021b.
P. Helber, B. Bischke, A. Dengel, and D. Borth. Eurosat: A novel dataset and deep learning
benchmark for land use and land cover classification. IEEE Journal of Selected Topics in
Applied Earth Observations and Remote Sensing, 12(7):2217–2226, 2019.
O. Henaff. Data-efficient image recognition with contrastive predictive coding. In International
conference on machine learning, pages 4182–4192. PMLR, 2020.
N. Houlsby, F. Huszár, Z. Ghahramani, and M. Lengyel.
Bayesian active learning for
classification and preference learning. arXiv preprint arXiv:1112.5745, 2011.
E. Hüllermeier and W. Waegeman. Aleatoric and epistemic uncertainty in machine learning:
An introduction to concepts and methods. Machine Learning, 110:457–506, 2021.
A. Iscen, G. Tolias, Y. Avrithis, and O. Chum. Label propagation for deep semi-supervised
learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 5070–5079, 2019.
17


<!-- page 18 -->
Seedat, Huynh, Imrie and van der Schaar
A. Jain, H. Patel, L. Nagalapatti, N. Gupta, S. Mehta, S. Guttula, S. Mujumdar, S. Afzal,
R. Sharma Mittal, and V. Munigala. Overview and importance of data quality for machine
learning tasks. In Proceedings of the 26th ACM SIGKDD International Conference on
Knowledge Discovery & Data Mining, pages 3561–3562, 2020.
Kaggle. Kaggle machine learning and data science survey, 2017. URL https://www.kaggle.
com/datasets/kaggle/kaggle-survey-2017.
T. Kim, J. Ko, J. Choi, S.-Y. Yun, et al. Fine samples for learning with noisy labels. Advances
in Neural Information Processing Systems, 34:24137–24149, 2021.
A. Kirsch, J. Van Amersfoort, and Y. Gal. Batchbald: Efficient and diverse batch acquisition
for deep bayesian active learning. Advances in neural information processing systems, 32,
2019.
Y. Kwon, J.-H. Won, B. J. Kim, and M. C. Paik. Uncertainty quantification using bayesian
neural networks in classification: Application to biomedical image segmentation. Compu-
tational Statistics & Data Analysis, 142:106816, 2020.
B. Lakshminarayanan, A. Pritzel, and C. Blundell. Simple and scalable predictive uncertainty
estimation using deep ensembles. Advances in Neural Information Processing Systems, 30,
2017.
C. Lee, F. Imrie, and M. van der Schaar. Self-supervision enhanced feature selection with
correlated gates. In International Conference on Learning Representations, 2021.
D.-H. Lee et al. Pseudo-label: The simple and efficient semi-supervised learning method for
deep neural networks. In Workshop on Challenges in Representation Learning, ICML, 2013.
Y. Lee, H. Yao, and C. Finn. Diversify and disambiguate: Out-of-distribution robustness via
disagreement. In The Eleventh International Conference on Learning Representations, 2023.
J. Li, R. Socher, and S. C. Hoi. Dividemix: Learning with noisy labels as semi-supervised
learning. In International Conference on Learning Representations, 2019.
Y. Li, T. Ma, and H. R. Zhang. Learning over-parametrized two-layer neural networks
beyond ntk. In Conference on Learning Theory, pages 2613–2682. PMLR, 2020.
W. Liang, G. A. Tadesse, D. Ho, L. Fei-Fei, M. Zaharia, C. Zhang, and J. Zou. Advances,
challenges and opportunities in creating data for trustworthy AI. Nature Machine Intelli-
gence, 4(8):669–677, 2022.
S. Mussmann and P. Liang. On the relationship between data efficiency and error for
uncertainty sampling. In International Conference on Machine Learning, pages 3674–3682.
PMLR, 2018.
V.-L. Nguyen, S. S. Farfade, and A. van den Hengel. Confident Sinkhorn allocation for
pseudo-labeling. arXiv preprint arXiv:2206.05880, 2022a.
V.-L. Nguyen, M. H. Shaker, and E. Hüllermeier. How to measure uncertainty in uncertainty
sampling for active learning. Machine Learning, 111(1):89–122, 2022b.
18


<!-- page 19 -->
You can’t handle the (dirty) truth: Data-centric insights improve pseudo-labeling
C. Northcutt, L. Jiang, and I. Chuang. Confident learning: Estimating uncertainty in dataset
labels. Journal of Artificial Intelligence Research, 70:1373–1411, 2021a.
C. G. Northcutt, A. Athalye, and J. Mueller. Pervasive label errors in test sets destabilize
machine learning benchmarks. In Thirty-fifth Conference on Neural Information Processing
Systems Datasets and Benchmarks Track (Round 1), 2021b.
M. Oquab, T. Darcet, T. Moutakanni, H. V. Vo, M. Szafraniec, V. Khalidov, P. Fernandez,
D. HAZIZA, F. Massa, A. El-Nouby, et al. Dinov2: Learning robust visual features without
supervision. Transactions on Machine Learning Research, 2023.
M. Paul, S. Ganguli, and G. K. Dziugaite. Deep learning on a data diet: Finding important
examples early in training. Advances in Neural Information Processing Systems, 34, 2021.
G. Pleiss, T. Zhang, E. Elenberg, and K. Q. Weinberger. Identifying mislabeled data using
the area under the margin ranking. Advances in Neural Information Processing Systems,
33:17044–17056, 2020.
S. J. Pocock, C. A. Ariti, J. J. McMurray, A. Maggioni, L. Køber, I. B. Squire, K. Swedberg,
J. Dobson, K. K. Poppe, G. A. Whalley, et al. Predicting survival in heart failure: a risk
score based on 39 372 patients from 30 studies. European Heart Journal, 34(19):1404–1413,
2013.
Prostate Cancer PCUK. Cutract. https://prostatecanceruk.org, 2019.
A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever. Language models are
unsupervised multitask learners. 2019.
C. Renggli, L. Rimanic, N. M. Gürel, B. Karlas, W. Wu, and C. Zhang. A data quality-
driven view of MLOps. IEEE Data Engineering Bulletin, 2021.
M. N. Rizve, K. Duarte, Y. S. Rawat, and M. Shah. In defense of pseudo-labeling: An
uncertainty-aware pseudo-label selection framework for semi-supervised learning.
In
International Conference on Learning Representations, 2021.
N. Sambasivan, S. Kapania, H. Highfill, D. Akrong, P. Paritosh, and L. M. Aroyo. “Everyone
wants to do the model work, not the data work”: Data cascades in high-stakes AI. In
Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems, pages
1–15, 2021.
L. Schmarje, V. Grossmann, C. Zelenka, S. Dippel, R. Kiko, M. Oszust, M. Pastell, J. Stracke,
A. Valros, N. Volkmann, et al. Is one annotation enough? A data-centric image classification
benchmark for noisy and ambiguous label estimation. Advances in Neural Information
Processing Systems, 35:33215–33232, 2022.
N. Seedat, J. Crabbé, I. Bica, and M. van der Schaar. Data-IQ: Characterizing subgroups with
heterogeneous outcomes in tabular data. In Advances in Neural Information Processing
Systems, 2022.
19


<!-- page 20 -->
Seedat, Huynh, Imrie and van der Schaar
N. Seedat, F. Imrie, and M. van der Schaar. Dissecting sample hardness: A fine-grained
analysis of hardness characterization methods for data-centric AI. In The Twelfth Interna-
tional Conference on Learning Representations, 2023a.
N. Seedat, F. Imrie, and M. van der Schaar. Navigating Data-Centric Artificial Intelligence
with DC-Check: Advances, Challenges, and Opportunities. IEEE Transactions on Artificial
Intelligence, 2023b.
N. Seedat, A. Jeffares, F. Imrie, and M. van der Schaar. Improving adaptive conformal pre-
diction using self-supervised learning. In International Conference on Artificial Intelligence
and Statistics, pages 10160–10177. PMLR, 2023c.
R. Shwartz-Ziv and A. Armon. Tabular data: Deep learning is not all you need. Information
Fusion, 81:84–90, 2022.
K. Sohn, D. Berthelot, N. Carlini, Z. Zhang, H. Zhang, C. A. Raffel, E. D. Cubuk, A. Kurakin,
and C.-L. Li.
FixMatch: Simplifying semi-supervised learning with consistency and
confidence. Advances in Neural Information Processing Systems, 33, 2020.
S. Swayamdipta, R. Schwartz, N. Lourie, Y. Wang, H. Hajishirzi, N. A. Smith, and Y. Choi.
Dataset cartography: Mapping and diagnosing datasets with training dynamics.
In
Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing
(EMNLP), pages 9275–9293, 2020.
K. S. Tai, P. D. Bailis, and G. Valiant. Sinkhorn label allocation: Semi-supervised classification
via annealed self-training. In International Conference on Machine Learning, pages 10065–
10075. PMLR, 2021.
J. E. van Engelen and H. H. Hoos. A survey on semi-supervised learning. Machine Learning,
109:373–440, 2019.
Y. Wang, H. Chen, Y. Fan, W. Sun, R. Tao, W. Hou, R. Wang, L. Yang, Z. Zhou, L.-Z. Guo,
et al. USB: A unified semi-supervised learning benchmark for classification. Advances in
Neural Information Processing Systems, 35:3938–3961, 2022.
Y. Wang, H. Chen, Q. Heng, W. Hou, Y. Fan, Z. Wu, J. Wang, M. Savvides, T. Shinozaki,
B. Raj, B. Schiele, and X. Xie. Freematch: Self-adaptive thresholding for semi-supervised
learning. In The Eleventh International Conference on Learning Representations, 2023.
URL https://openreview.net/forum?id=PDrUPTXJI_A.
J. Wei, Z. Zhu, H. Cheng, T. Liu, G. Niu, and Y. Liu. Learning with noisy labels revisited:
A study using real-world human annotations. In International Conference on Learning
Representations, 2022a.
Q. Wei, H. Sun, X. Lu, and Y. Yin. Self-filtering: A noise-aware sample selection for label
noise with confidence penalization. In European Conference on Computer Vision, pages
516–532. Springer, 2022b.
20


<!-- page 21 -->
You can’t handle the (dirty) truth: Data-centric insights improve pseudo-labeling
X. Xia, T. Liu, B. Han, M. Gong, J. Yu, G. Niu, and M. Sugiyama. Sample selection with
uncertainty of losses for learning with noisy labels. In International Conference on Learning
Representations, 2021.
J. Yang, R. Shi, and B. Ni.
Medmnist classification decathlon: A lightweight automl
benchmark for medical image analysis. In 2021 IEEE 18th International Symposium on
Biomedical Imaging (ISBI), pages 191–195. IEEE, 2021.
I.-C. Yeh and C.-h. Lien. The comparisons of data mining techniques for the predictive
accuracy of probability of default of credit card clients. Expert Systems with Applications,
36(2):2473–2480, 2009.
J. Yoon, Y. Zhang, J. Jordon, and M. van der Schaar. VIME: Extending the success of
self-and semi-supervised learning to tabular domain. Advances in Neural Information
Processing Systems, 33:11033–11043, 2020.
S. Zagoruyko and N. Komodakis. Wide residual networks. In British Machine Vision
Conference 2016. British Machine Vision Association, 2016.
B. Zhang, Y. Wang, W. Hou, H. Wu, J. Wang, M. Okumura, and T. Shinozaki. FlexMatch:
Boosting semi-supervised learning with curriculum pseudo labeling. Advances in Neural
Information Processing Systems, 34, 2021.
X. J. Zhu. Semi-supervised learning literature survey. Technical Report 1530, University of
Wisconsin-Madison Department of Computer Sciences, 2005.
21


<!-- page 22 -->
Seedat, Huynh, Imrie and van der Schaar
Appendix: You can’t handle the (dirty) truth:
Data-centric insights improve pseudo-labeling
Table of Contents
A
Additional details on DIPS
23
A.1 DIPS algorithm . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
23
A.2 Overview of pseudo-labeling methods . . . . . . . . . . . . . . . . . . . . .
23
A.3 Learning dynamics plot
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
24
A.4 Comparison with Schmarje et al. (2022) . . . . . . . . . . . . . . . . . . .
25
A.5 Connection to active learning . . . . . . . . . . . . . . . . . . . . . . . . .
25
B
Experimental details
27
B.1
Baselines
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
27
B.2
Datasets . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
27
B.3
Implementation details . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
27
C
Additional experiments
30
C.1 Ablation study
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
30
C.2 Impact of the selector function r
. . . . . . . . . . . . . . . . . . . . . . .
31
C.3 Insights into data selection
. . . . . . . . . . . . . . . . . . . . . . . . . .
32
C.4 Additional experiments in computer vision . . . . . . . . . . . . . . . . . .
34
C.5 Dependency between label noise level and amount of labeled data . . . . .
36
C.6 Ablation on the window of iterations for learning dynamics
. . . . . . . .
37
C.7 Analyzing DIPS performance with an alternative pseudo-labeling formulation 38
C.8 Threshold choices . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
39
C.9 Comparison to VIME
. . . . . . . . . . . . . . . . . . . . . . . . . . . . .
40
C.10 Importance of aleatoric uncertainty . . . . . . . . . . . . . . . . . . . . . .
40
C.11 Link between optimal thresholds and error rate . . . . . . . . . . . . . . .
41
C.12 DIPS in the clean label setting
. . . . . . . . . . . . . . . . . . . . . . . .
42
22


<!-- page 23 -->
You can’t handle the (dirty) truth: Data-centric insights improve pseudo-labeling
Appendix A. Additional details on DIPS
A.1
DIPS algorithm
We summarize how DIPS can be incorporated into any pseudo-labeling algorithm as per
Algorithm 1. For clarity, we highlight in red how DIPS extends the current pseudo-labeling
formulations.
Algorithm 2 Plug DIPS into any pseudo-labeler
1: Train a network, f (0), using the samples from Dlab.
2:
Plug-in DIPS: set D(1)
train = D(1) = r(Dlab, f (0))
▷Initialization of the training set
3: for t = 1..T do
4:
Initialize new network f (t)
5:
Train f (t) using D(t)
train.
6:
Pseudo-label Dunlab using f (t)
7:
Define D(t+1) using the PL method’s selector s
8:
Plug-in DIPS : Define D(t+1)
train = r(D(t+1), f (t))
▷Data characterization and selection, Sec. 4.3
9: end for
10: return fT
We emphasize a key advantage of DIPS lies in its simplicity. Beyond getting the compu-
tation almost for free (no additional models to train when instantiating DIPS with learning
dynamics) - i.e. P3: Computationally cheap, we can also plug DIPS into and augment
any pseudo-labeling algorithm. (i.e. P1: Plug & Play), which makes for easier adoption.
On checkpoint storage. From a practical perspective, DIPS does not require storing
all the model checkpoints (which would be costly from a memory perspective). Indeed, notice
that the confidence (Definition 4.1) and aleatoric uncertainty (Definition 4.2) metrics can be
expressed as averages. Hence, we can compute these quantities iteratively simply by using the
model predictions at the end of an epoch to update a running average (i.e. define µ(e+1) =
µ(e)×E+m(e+1)
e+1
, where µ(e) is the running average at epoch e, and m(e+1) is the summand in
either Definition 4.1 or Definition 4.2), which bypasses the need to store all the checkpoints.
A.2
Overview of pseudo-labeling methods
As we described in Sec. 3.1, current pseudo-labeling methods typically differ in the way the
selector function s is defined. Note that s is solely used to select pseudo-labeled samples,
among those which have not already been pseudo-labeled at a previous iteration. The general
way to define s for any set D and function f is s(D, f) = {(x, [ˆy]k)|x ∈D, ˆy = f(x), k ∈
[C], [m(x, f)]k = 1}, where m is such that m(x, f) ∈RC. Alternatively stated, a selector m
outputs the binary decision of selecting x and its associated pseudo-labels ˆy. Notice that we
allow the multi-label setting, where C > 1, hence explaining why the selector m’s output is
a vector of size C. We now give the intuition of how widely used PL methods construct the
selector m, thus leading to specific definitions of s.
Greedy pseudo-labeling (Lee et al., 2013) The intuition of greedy pseudo-labeling
is to select a pseudo-label if the classifier is sufficiently confident in it. Given two positive
23


<!-- page 24 -->
Seedat, Huynh, Imrie and van der Schaar
thresholds τp and τn with τn < τp, and a classifier f, m is defined by [m(x, f)]k = 1([f(x)]k ≥
τp) + 1([f(x)]k ≤τn) for k ∈[C].
UPS (Rizve et al., 2021) In addition to confidence, UPS considers the uncertainty
of the prediction when selecting pseudo-labels. Given two thresholds κn < κp, UPS defines
[m(x, f)]k = 1(u([f(x)]k) ≤κp)1([f(x)]k ≥τp)+1(u([f(x)]k) ≤κn)1([f(x)]k ≤τn), where u
is a proxy for the uncertainty. One could compute u using MC-Dropout (for neural networks)
or ensembles.
Flexmatch (Zhang et al., 2021) FlexMatch dynamically adjusts a class-dependent
threshold for the selection of pseudo-labels. The selection mechanism is defined by: [m(x, f)]k =
1(maxj[f(x)]j > τt(arg maxi[f(x)]i)), i.e. at iteration t, an unlabeled point is selected if and
only if the confidence of the most probable class is greater than its corresponding dynamic
threshold.
SLA (Tai et al., 2021) and CSA (Nguyen et al., 2022a) The fundamental intuition
behind SLA and CSA is to solve a linear program in order to assign pseudo-labels to unlabeled
data, based on the predictions f(x). The allocation of pseudo-labels considers both the
rows (the unlabeled samples) and the columns (the classes), hence contrasts greedy pseudo-
labeling, and incorporates linear constraints in the optimization problem. An approximate
solution is found using the Sinkhorn-Knopp algorithm.
A.3
Learning dynamics plot
Key to our instantiation of DIPS is the analysis of learning dynamics. We illustrate in Fig. 9
for a pseudo-labeling run in Sec. 5.5 the learning dynamics of 6 individual samples. DIPS uses
these dynamics to compute the metrics of confidence and aleatoric uncertainty, as explained
in Sec. 4.2. This then characterizes the samples as Useful or Harmful.
Figure 9: DIPS uses learning dynamics to compute the metrics of confidence and aleatoric
uncertainty. It then characterizes labeled (and pseudo-labeled) samples based on these
metrics. The y axis represents the probability assigned to the true label by the current
classifier h, which is P( ˆYF(x, y) = 1|H = h).
24

[CAPTION] Figure 9: DIPS uses learning dynamics to compute the metrics of confidence and aleatoric


<!-- page 25 -->
You can’t handle the (dirty) truth: Data-centric insights improve pseudo-labeling
The intuition is the following: a sample is deemed Useful if the classifiers at each checkpoint
are confident in predicting the associated (pseudo-)label of the sample, and the associated
aleatoric uncertainty is low. Failure to satisfy this criterion leads to a characterization as
Harmful.
A.4
Comparison with Schmarje et al. (2022)
In what follows, we compare DIPS with DC3 (Schmarje et al., 2022). While both DIPS and
DC3 handle the data-centric issue of issues in data and share similarities in their titles, they
tackle different data-centric problems that might arise in semi-supervised learning. The main
differences are along 4 different dimensions:
1. Problem setup/Type of data-centric issue: DIPS tackles the problems of hard
noisy labels where each sample has a single label assigned in the labeled set which could
be incorrect. In contrast, DC3 deals with the problem of soft labeling where each sample
might have multiple annotations from different annotators which may be variable.
2. Label noise modeling: DIPS aims to identify the noisy labels, whereas DC3 models
the inter-annotator variability to estimate label ambiguity.
3. Integration into SSL: DIPS is a plug-in on top of any pseudo-labeling pipeline,
selecting the labeled and pseudo-labeled data.
DC3, on the other hand, uses its
ambiguity model (learned on the multiple annotations) to either keep the pseudo-label
or use a cluster assignment.
4. Dataset applicability: DIPS has lower dataset requirements as it can be applied to any
dataset with labeled and unlabeled samples, even if there is only a single label per sample.
It does not require multiple annotations. DC3 has higher dataset requirements as it
relies on having multiple annotations per sample to estimate inter-annotator variability
and label ambiguity. Without multiple labels per sample, it cannot estimate ambiguity
and perform joint classification and clustering. Consequently, DIPS is applicable to the
standard semi-supervised learning setup of limited labeled data and abundant unlabeled
data, whereas DC3 targets the specific problem of ambiguity across multiple annotators.
A.5
Connection to active learning
Active learning primarily focuses on the iterative process of selecting data samples that, when
labeled, are expected to most significantly improve the model’s performance. This selection
is typically based on criteria such as uncertainty sampling which focuses on epistemic
uncertainty (Mussmann and Liang, 2018; Houlsby et al., 2011; Kirsch et al., 2019; Nguyen
et al., 2022b). The primary objective is to minimize labeling effort while maximizing the
model’s learning efficiency. In contrast, DIPS does both labeled and pseudo-labeled selection
and employs the term ’useful’ in a different sense. Here, ’usefulness’ refers to the capacity
of a data sample to contribute positively to the learning process based on its likelihood
of being correctly labeled. Our approach, which leverages training dynamics based on
aleatoric uncertainty and confidence, is designed to flag and exclude mislabeled data.
This distinction is critical in our methodology as it directly addresses the challenge of data
25


<!-- page 26 -->
Seedat, Huynh, Imrie and van der Schaar
quality, particularly in scenarios where large volumes of unlabeled data are integrated into
the training process. In active learning, these metrics are used to identify data points that,
if labeled, would yield the most significant insights for model training. In our approach,
they serve to identify and exclude data points that could potentially deteriorate the model’s
performance due to incorrect labeling.
26


<!-- page 27 -->
You can’t handle the (dirty) truth: Data-centric insights improve pseudo-labeling
Appendix B. Experimental details
B.1
Baselines
In our experiments, we consider the following baselines for pseudo-labeling: Greedy-PL (Lee
et al., 2013), UPS (Rizve et al., 2021), Flexmatch (Zhang et al., 2021), SLA (Tai et al., 2021)
and CSA (Nguyen et al., 2022a). To assess the performance of DIPS for computer vision, we
use FixMatch (Sohn et al., 2020).
Greedy-PL The confidence upper threshold is 0.8
UPS The confidence upper threshold is 0.8 and the threshold on the uncertainty is 0.2.
The size of the ensemble is 10.
FlexMatch The upper threshold is 0.9 (which is then normalized).
CSA and SLA The confidence upper threshold is 0.8. The size of the ensemble is 20.
We use the implementation of these algorithms provided in (Nguyen et al., 2022a).
FixMatch The threshold is set to 0.95 as in (Sohn et al., 2020).
B.2
Datasets
We summarize the different datasets we use in this paper in Table 1. The datasets vary
in number of samples, number of features, and domain. Recall we use data splits with a
proportion of Dlab : Dunlab of 0.1:0.9.
Table 1: Summary of the datasets used.
Name
n samples
n features
Domain
Adult Income (Asuncion and Newman, 2007)
30k
12
Finance
Agarius lepiota (Asuncion and Newman, 2007)
8k
22
Agriculture
Blog (Buza, 2013)
10k
280
Social media
Compas (Angwin et al., 2016)
5k
13
Criminal justice
Covid-19 (Baqui et al., 2020)
7k
29
Healthcare/Medicine
Credit (Taiwan) (Yeh and Lien, 2009)
30k
23
Finance
CUTRACT Prostate (Prostate Cancer PCUK, 2019)
2k
12
Healthcare/Medicine
Drug (Fehrman et al., 2017)
2k
27
Healthcare/Medicine
German-credit (Asuncion and Newman, 2007)
1k
24
Finance
Higgs (Baldi et al., 2014)
25k
23
Physics
MAGGIC (Pocock et al., 2013)
41k
29
Healthcare/Medicine
SEER Prostate (Duggan et al., 2016)
20k
12
Healthcare/Medicine
The private datasets are de-identified and used in accordance with the guidance of the
respective data providers.
For computer vision experiments, we use CIFAR-10N (Wei et al., 2022a). The dataset
can be accessed via its official release.4
B.3
Implementation details
The three key design decisions necessary for pseudo-labeling are:
4. https://github.com/UCSC-REAL/cifar-10-100n
27


**[Table p27.1]**
| Name | n samples n features Domain |
| --- | --- |
| Adult Income (Asuncion and Newman, 2007) Agarius lepiota (Asuncion and Newman, 2007) Blog (Buza, 2013) Compas (Angwin et al., 2016) Covid-19 (Baqui et al., 2020) Credit (Taiwan) (Yeh and Lien, 2009) CUTRACT Prostate (Prostate Cancer PCUK, 2019) Drug (Fehrman et al., 2017) German-credit (Asuncion and Newman, 2007) Higgs (Baldi et al., 2014) MAGGIC (Pocock et al., 2013) SEER Prostate (Duggan et al., 2016) | 30k 12 Finance 8k 22 Agriculture 10k 280 Social media 5k 13 Criminal justice 7k 29 Healthcare/Medicine 30k 23 Finance 2k 12 Healthcare/Medicine 2k 27 Healthcare/Medicine 1k 24 Finance 25k 23 Physics 41k 29 Healthcare/Medicine 20k 12 Healthcare/Medicine |

[CAPTION] Table 1: Summary of the datasets used.


<!-- page 28 -->
Seedat, Huynh, Imrie and van der Schaar
1. Choice of backbone model (i.e. predictive classifier f)
2. Number of pseudo-labeling iterations — recall that it is an iterative process by repeatedly
augmenting the labeled data with selected samples from the unlabeled data.
3. Compute requirements.
We describe each in the context of each experiment.
For further details on the
experimental setup and process, see each relevant section of the main paper.
■DIPS thresholds: Recall that DIPS has two thresholds τconf and τal. We set τconf = 0.8,
in order to select high confidence samples based on the mean of the learning dynamic. Note,
τal is bounded between [0,0.25]. We adopt an adaptive threshold for τal based on the dataset,
such that τal = 0.75 · (max(val(Dtrain)) −min(val(Dtrain)))
B.3.1
Synthetic example: Data characterization and unlabeled data
improve test accuracy
1. Backbone model: We use an XGBoost, with 100 estimators similar to (Nguyen et al.,
2022a). Note, XGBoost has been shown to often outperform deep learning methods
on tabular data (Borisov et al., 2021; Shwartz-Ziv and Armon, 2022). That said, our
framework is not restricted to XGBoost.
2. Iterations: we use T = 5 pseudo-labeling iterations, as in (Nguyen et al., 2022a).
3. Compute: CPU on a MacBook Pro with an Intel Core i5 and 16GB RAM.
B.3.2
Tabular data experiments: Sec 5.2, 5.3, 5.4
1. Backbone model: Some of the datasets have limited numbers, hence we have the
backbone as XGBoost, with 100 estimators similar to (Nguyen et al., 2022a). Note,
XGBoost has been shown to often outperform deep learning methods on tabular data
(Borisov et al., 2021; Shwartz-Ziv and Armon, 2022). That said, our framework is not
restricted to XGBoost.
2. Iterations: we use T=5, pseudo-labeling iterations, as in (Nguyen et al., 2022a).
3. Compute: CPU on a MacBook Pro with an Intel Core i5 and 16GB RAM.
For the tabular datasets, we use splits with a proportion of Dlab : Dunlab of 0.1:0.9.
B.3.3
Computer vision experiments: Sec 5.5
1. Backbone model: we use a WideResnet-18 (Zagoruyko and Komodakis, 2016) as in
(Sohn et al., 2020).
2. Iterations: with Fixmatch we use T=1024k iterations as in (Sohn et al., 2020).
3. Data augmentations: Strong augmentation is done with RandAugment with random
magnitude. Weak augmentation is performed with a random horizontal flip.
28


<!-- page 29 -->
You can’t handle the (dirty) truth: Data-centric insights improve pseudo-labeling
4. Training hyperparameters: we use the same hyperparameters as in the original work
(Sohn et al., 2020)
5. Compute: Nvidia V100 GPU, 6-Core Intel Xeon E5-2690 v4 with 16GB RAM.
For the experiments on CIFAR-10N, we use Dlab = 1000 samples.
29


<!-- page 30 -->
Seedat, Huynh, Imrie and van der Schaar
Appendix C. Additional experiments
C.1
Ablation study
Goal. We conduct an ablation study to characterize the importance of the different compo-
nents of DIPS.
Experiment.
These components are:
• data characterization and selection for the initialisation of D(1): DIPS defines D(1) =
D(1)
train = r(Dlab, f(0))
• data characterization and selection at each pseudo-labeling iteration: DIPS defines
D(i+1)
train = r(D(i+1), f(i))
Each of these two components can be ablated, resulting in four different possible combinations:
DIPS (data selection both at initialization and during the pseudo-labeling iterations), A1
(data selection only at initialization), A2 (data selection only during the pseudo-labeling
iterations), A3 (no data selection). Note that A3 corresponds to vanilla pseudo-labeling,
and not selecting data amounts to using an identity selector.
We consider the same experimental setup as in Sec. 5.1, generating data in two quadrants
with varying proportions of corrupted samples.
Results. The results are reported in Fig. 10. As we can see, each component in DIPS is
important to improve pseudo-labeling: 1) the initialization of the labeled data D(1) drives the
pseudo-labeling process 2) data characterization of both labeled and pseudo-labeled samples
is important.
0.10
0.15
0.20
0.25
0.30
0.35
0.40
0.45
Proportion of corrupted samples
60
70
80
90
100
Accuracy
A1
A2
A3
DIPS (ours)
Figure 10: Ablation study: data characterization and selection are important both for the
initialization of the labeled data and during the pseudo-labeling iterations, which forms the
basis of DIPS.
Takeaway. It is important to characterize and select data both for the initialization of
the labeled data and during the pseudo-labeling iterations.
30


**[Table p30.1]**
|  |  |
| --- | --- |
|  |  |
|  |  |
|  |  |
| A1 A2 |  |
| A3 DIPS (ours) |  |

[CAPTION] Figure 10: Ablation study: data characterization and selection are important both for the


<!-- page 31 -->
You can’t handle the (dirty) truth: Data-centric insights improve pseudo-labeling
C.2
Impact of the selector function r
Goal. We investigate variants of DIPS where we replace the selector function r with heuristics
used in the LNL literature. The most commonly used is the small-loss criterion (Xia et al.,
2021). Additionally, we assess Fluctuation (Wei et al., 2022b) and FINE (Kim et al., 2021)
as alternative sample selection approaches.
Experiments. We use the vanilla PL algorithm as the semi-supervised backbone. We
consider three different selectors in addition to the one presented in the main paper:
• Small-loss criterion (Xia et al., 2021): the quantity of interest is µ(x, y) = 1
E
PE
e=1 l(x, y, fe)
where l is a loss function and f1, ..., fE are the classifiers at the different checkpoints.
Intuitively, a sample with a high loss is more likely to present mislabeling issues, since
it is harder to learn.
• Fluctuation criterion (Wei et al., 2022b): the quantity of interest, for two checkpoints
e1 < e2 is β(x, y, e1, e2) = 1([fe1(x)]y > 1
2)1([fe2(x)]y < 1
2), which is equal to one if
the sample is correctly classified at the checkpoint e1 and wrongly classified at e2, 0
otherwise. Following (Wei et al., 2022b), we smooth this score with the confidence.
• FINE criterion (Kim et al., 2021): FINE creates a gram matrix of the representations in
the noisy training dataset for each class. Then, FINE computes an alignment using the
square of the inner product values between the representations and the first eigenvector
of each gram matrix. A Gaussian mixture model is then fit on these alignment values
to find clean and noisy instances.
The scores obtained by each approach are then used for sample selection, hence defining
variants of the selector r.
Results. We report the results for 12 different datasets in Table 2. As we can see, the
DIPS approach using learning dynamics outperforms the alternative LNL methods. This
highlights the importance of a multi-dimensional data characterization, where both confidence
and aleatoric uncertainty are taken into account to select samples. Moreover, the LNL methods
typically operate under the assumption of a large number of labeled samples, highlighting
that our DIPS approach tailored for the pseudo-labeling setting should indeed be preferred.
Takeaway. A key component of DIPS is its sample selector based on learning dynamics,
which outperforms methods designed for the LNL setting.
31


<!-- page 32 -->
Seedat, Huynh, Imrie and van der Schaar
Table 2: DIPS outperforms heuristics used in the LNL setting by leveraging learning dynamics.
Best performing method in bold, statistically equivalent performance underlined.
DIPS
(Ours)
Small-Loss
(Xia et al., 2021)
Fluctuation
(Wei et al., 2022b)
FINE
(Kim et al., 2021)
adult
82.66 ± 0.10
79.52 ± 0.26
80.81 ± 0.20
24.22 ± 0.35
agaricus-lepiota
65.03 ± 0.25
64.45 ± 0.28
49.21 ± 1.48
35.96 ± 3.65
blog
80.58 ± 0.10
79.90 ± 0.28
80.05 ± 0.27
73.41 ± 1.66
credit
81.39 ± 0.07
78.46 ± 0.34
79.38 ± 0.29
64.58 ± 3.30
covid
69.97 ± 0.30
70.09 ± 0.36
69.03 ± 0.53
67.70 ± 0.84
compas
65.34 ± 0.25
62.76 ± 0.60
61.31 ± 0.64
60.20 ± 1.29
cutract
68.60 ± 0.31
66.33 ± 1.09
64.35 ± 1.49
61.98 ± 2.27
drug
78.16 ± 0.26
76.84 ± 0.61
75.66 ± 0.72
74.63 ± 1.98
German-credit
69.40 ± 0.46
69.80 ± 1.00
69.60 ± 1.19
69.70 ± 1.19
higgs
81.99 ± 0.07
81.08 ± 0.12
81.50 ± 0.09
73.82 ± 0.51
maggic
67.60 ± 0.08
65.57 ± 0.20
65.94 ± 0.20
62.28 ± 0.42
seer
82.74 ± 0.08
80.74 ± 0.26
82.28 ± 0.20
77.10 ± 0.76
C.3
Insights into data selection
Goal. We wish to gain additional insight into the performance improvements provided by
DIPS when added to the vanilla pseudo-labeling method. In particular, we examine the
significant performance gain attained by DIPS for cross-country augmentation, shown in Sec.
5.4.
Experiment. The DIPS selector mechanism is the key differentiator as compared to
the vanilla methods. Hence, we examine the selected samples from DIPS and vanilla. We
then compare the samples to Dtest. Recall we select samples from Dunlab which come from
US patients, whereas Dlab and Dtest are from the UK. We posit that “matching” the test
distribution as closely as possible would lead to the best performance.
Results. We examine the most important features as determined by the XGBoost
and compare their distributions. We find that the following 4 features in order are the
most important: (1) Treatment: Primary hormone therapy, (2) PSA score, (3) Age, (4)
Comorbidities. This is expected where the treatment and PSA blood scores are important
predictors of prostate cancer mortality. We then compare these features in Dtest vs the final
Dtrain when using DIPS selection and using vanilla selection.
Fig. 11 shows that especially on the two most important features (1) Treatment: Primary
hormone therapy and (2) PSA score; that DIPS’s selection better matches Dtest — which
explains the improved performance.
Quantitatively, we then compute the Jensen-Shannon (JS) divergence between the data
selected by DIPS and vanilla as compared to Dtest. We find DIPS has a lower JSD of 0.0296
compared to vanilla of 0.0317, highlighting we better match the target domain’s features
through selection.
This behavior is reasonable, as samples that are very different will be filtered out by
virtue of their learning dynamics being more uncertain.
32

[CAPTION] Table 2: DIPS outperforms heuristics used in the LNL setting by leveraging learning dynamics.

[CAPTION] Fig. 11 shows that especially on the two most important features (1) Treatment: Primary


<!-- page 33 -->
You can’t handle the (dirty) truth: Data-centric insights improve pseudo-labeling
DIPS
Vanilla
Vanilla does 
not match 
Test well
DIPS 
matches  
Test well
Figure 11: DIPS improves performance as its selection for Dtrain better matches Dtest on the
most important features, whereas vanilla selects samples which are different to Dtest
For completeness, we also include a radar chart with vanilla PL but without any data
selection in Fig. 12.
Figure 12: DIPS is closer to the test data than vanilla PL.
Takeaway. A significant source of DIPS’s gain is that its selection mechanism selects
samples that closely approximate the distribution of Dtest. In particular, we see this holds
true for the features that are considered most important to the classifier — hence accounting
for the improved downstream model performance observed when using DIPS.
33

[CAPTION] Figure 11: DIPS improves performance as its selection for Dtrain better matches Dtest on the

[CAPTION] Figure 12: DIPS is closer to the test data than vanilla PL.


<!-- page 34 -->
Seedat, Huynh, Imrie and van der Schaar
C.4
Additional experiments in computer vision
Goal. The goal of this experiment is to assess the benefit of DIPS in additional computer
vision settings, namely:
(i) when increasing the number of classes to 100 (CIFAR-100N), with FixMatch
(ii) when the size of Dlab (nlab) is small, with FixMatch
(iii) when using a different pseudo-labeling algorithm — FreeMatch (Wang et al., 2023) and
(iv) when evaluating on other datasets.
C.4.1
DIPS improves the performance of FixMatch on CIFAR-100N
Goal. The goal of this experiment is to further demonstrate DIPS’s utility in a setting with
an increased number of classes.
Setup. We adopt the same setup as in Sec. 5.5, using the dataset CIFAR-100N (Wei
et al., 2022a), which has 100 classes.
100k
500k
1000k
Iterations
20
25
30
35
40
45
50
55
60
Mean Top 1 Accuracy
FixMatch
FixMatch+DIPS
(a) Top-1 accuracy
100k
500k
1000k
Iterations
50
55
60
65
70
75
80
Mean Top 5 Accuracy
FixMatch
FixMatch+DIPS
(b) Top-5 accuracy
Figure 13: DIPS improves FixMatch on CIFAR-100N. We report both top-1 and top-5 accuracies.
Results. We show both top-1 and top-5 accuracies in Figure 13 for three different
numbers of iterations and 3 different seeds, which highlight the performance gains obtained
by using DIPS with FixMatch.
Takeaway. DIPS improves the performance of FixMatch on CIFAR-100N.
C.4.2
DIPS improves FixMatch for smaller nlab
Setup. We consider nlab = 200, and use the same setup as in Sec. 5.5, considering the
dataset CIFAR-10N.
34


**[Table p34.1]**
| 60 FixMatch 55 FixMatch+DIPS Accuracy 50 45 40 1 Top 35 Mean 30 25 20 100k 500k 1000k Iterations |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  | 10 | 0k | It | 50 era | 0k tion | s | 10 | 00k |  |
| (a) e 13: | To | p- | 1 | a | cc | ur | ac | y |  |


**[Table p34.2]**
| 80 FixMatch 75 FixMatch+DIPS Accuracy 70 65 5 Top 60 Mean 55 50 100k 500k 1000k Iterations |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  | 10 | 0k | It | 50 era | 0k tion | s | 10 | 00k |  |
| (b) eport bot ies in F ght the | To h ig p | p- to ur er | 5 p- e fo | a 1 a 1 rm | cc n 3 a | ur d f n | ac to or ce | y p-5 t g | acc hree ains |

[CAPTION] Figure 13: DIPS improves FixMatch on CIFAR-100N. We report both top-1 and top-5 accuracies.


<!-- page 35 -->
You can’t handle the (dirty) truth: Data-centric insights improve pseudo-labeling
100k
500k
1000k
Iterations
50
55
60
65
70
75
80
85
90
Mean Accuracy
FixMatch
FixMatch+DIPS
Figure 14: DIPS improves FixMatch with nlab = 200 on CIFAR-10N
Results. We report the test accuracy in Figure 14 for three different numbers of iterations
over 3 different seeds, highlighting the performance gains with DIPS.
Takeaway. DIPS improves performance of FixMatch for different sizes of Dlab
C.4.3
DIPS improves the performance of FreeMatch
Goal. The goal of this experiment is to further demonstrate DIPS’s utility in computer
vision with an alternative pseudo-labeling method, namely FreeMatch (Wang et al., 2023).
Setup. We adopt the same setup as in Sec. 5.5, using the dataset CIFAR-10N.
100k
500k
1000k
Iterations
50
55
60
65
70
75
80
85
90
Test accuracy
FreeMatch
FreeMatch+DIPS
Figure 15: DIPS improves FreeMatch on CIFAR-10N
Results.
We show the test accuracy in Figure 15, for three different numbers of
iterations and highlight the performance gains by combining DIPS with FreeMatch to improve
performance.
Takeaway. DIPS is versatile to other data modalities, improving the performance of
FreeMatch on CIFAR-10N with the inclusion of DIPS.
C.4.4
Additional datasets
We present in Figure 16 results for 4 additional image datasets: satellite images from Eurosat
(Helber et al., 2019) and medical images from TissueMNIST which form part of the USB
benchmark (Wang et al., 2022). Additionally, we include the related OrganAMNIST, and
35


**[Table p35.1]**
| 90 FixMatch 85 FixMatch+DIPS 80 Accuracy 75 70 Mean 65 60 55 50 100k 500k 1000k Iterations |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  | 10 | 0k | It | 50 erat | 0k ion | s | 10 | 00k |  |
| mproves F ccuracy g the p erforma rforma | ix in er nc n | M fo e c | at Fi r o e | ch gu ma f F o | w re n i f | it 1 ce x F | h 4 g Ma r | n l fo ai t e | = ab r th ns ch f eMa |


**[Table p35.2]**
| 90 FreeMatch 85 FreeMatch+DIPS 80 accuracy 75 70 65 Test 60 55 50 100k 500k 1000k Iterations |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  | 10 | 0k | It | 50 era | 0k tio | ns | 100 | 0k |  |
| 15: DIPS i test accu rformance tile to ot th the inc | mp ra g he lu | rov cy ain r sio | es i s da n | Fr n by ta of | ee Fi co m D | M gu m o IP | atc re bi dal S. | h o 1 nin iti | n CIF 5, fo g DI es, i |

[CAPTION] Figure 14: DIPS improves FixMatch with nlab = 200 on CIFAR-10N

[CAPTION] Figure 15: DIPS improves FreeMatch on CIFAR-10N


<!-- page 36 -->
Seedat, Huynh, Imrie and van der Schaar
PathMNIST, which are part of the MedMNIST collection (Yang et al., 2021). Given that
these datasets are well-curated, we consider a proportion of 0.2 of symmetric label noise
added to these datasets. As is shown, DIPS consistently improves the FixMatch baseline,
demonstrating its generalizability.
DIPS+FixMatch
FixMatch
60
62
64
66
68
70
Test accuracy
EuroSAT
DIPS+FixMatch
FixMatch
44.0
44.5
45.0
45.5
46.0
46.5
47.0
47.5
48.0
TissueMNIST
DIPS+FixMatch
FixMatch
70.0
70.5
71.0
71.5
72.0
72.5
73.0
73.5
74.0
OrganAMNIST
DIPS+FixMatch
FixMatch
72
73
74
75
76
77
78
PathMNIST
Figure 16: DIPS improves FixMatch on images datasets
C.5
Dependency between label noise level and amount of labeled data
We conduct a synthetic experiment following a similar setup as in Sec. 5.1 to investigate the
dependency between label noise level and amount of labeled data. Note that the experiment
is synthetic in order to be able to control the amount of label noise.
We considered the same list of label noise proportions, ranging from 0. to 0.45. For each
label noise proportion, we consider nlab ∈{50, 100, 1000}, and fix nunlab = 1000. For each
configuration, we conduct the experiment 40 times.
We report the results in Fig. 17. As we can see on the plots, PL+DIPS consistently
outperforms the supervised baselines in almost all the configurations. When the amount of
labeled data is low (nlab = 50) and the proportion of corrupted samples is high (0.45), PL is
on par with the supervised baseline. Hence pseudo-labeling is more difficult with a very low
amount of labeled samples (and a high level of noise).
We note, though, that DIPS consistently improves the PL baseline for reasonable amounts
of label noise which we could expect in real-world settings (e.g. 0.1). The performance gap
between DIPS and PL is remarkably noticeable for nlab = 1000, i.e. when the amount of
labeled samples is high.
0.0
0.1
0.2
0.3
0.4
Proportion of corrupted samples
50
60
70
80
90
100
Test accuracy
n_lab = 50
Supervised
PL+DIPS
PL
0.0
0.1
0.2
0.3
0.4
Proportion of corrupted samples
50
60
70
80
90
100
Test accuracy
n_lab = 100
Supervised
PL+DIPS
PL
0.0
0.1
0.2
0.3
0.4
Proportion of corrupted samples
50
60
70
80
90
100
Test accuracy
n_lab = 1000
Supervised
PL+DIPS
PL
Figure 17: DIPS consistently improves upon the PL and supervised baselines, across the
different label noise levels and amounts of labeled data.
36


**[Table p36.1]**
| EuroSAT TissueMNIST OrganAMNIST PathMNIST 70 48.0 74.0 78 47.5 73.5 68 77 47.0 73.0 76 66 46.5 72.5 accuracy 46.0 72.0 75 64 45.5 71.5 Test 74 45.0 71.0 62 73 44.5 70.5 60 44.0 70.0 72 DIPS+FixMatch FixMatch DIPS+FixMatch FixMatch DIPS+FixMatch FixMatch DIPS+FixMatch FixMatch Figure 16: DIPS improves FixMatch on images datasets C.5 Dependency between label noise level and amount of labeled data We conduct a synthetic experiment following a similar setup as in Sec. 5.1 to investigate the dependency between label noise level and amount of labeled data. Note that the experiment is synthetic in order to be able to control the amount of label noise. We considered the same list of label noise proportions, ranging from 0. to 0.45. For each label noise proportion, we consider n ∈ {50, 100, 1000}, and fix n = 1000. For each lab unlab configuration, we conduct the experiment 40 times. We report the results in Fig. 17. As we can see on the plots, PL+DIPS consistently outperforms the supervised baselines in almost all the configurations. When the amount of labeled data is low (n = 50) and the proportion of corrupted samples is high (0.45), PL is lab on par with the supervised baseline. Hence pseudo-labeling is more difficult with a very low amount of labeled samples (and a high level of noise). We note, though, that DIPS consistently improves the PL baseline for reasonable amounts of label noise which we could expect in real-world settings (e.g. 0.1). The performance gap between DIPS and PL is remarkably noticeable for n = 1000, i.e. when the amount of lab labeled samples is high. |  | EuroSAT TissueMNIST OrganAMNIST PathMNIST 70 48.0 74.0 78 47.5 73.5 68 77 47.0 73.0 76 66 46.5 72.5 accuracy 46.0 72.0 75 64 45.5 71.5 Test 74 45.0 71.0 62 73 44.5 70.5 60 44.0 70.0 72 DIPS+FixMatch FixMatch DIPS+FixMatch FixMatch DIPS+FixMatch FixMatch DIPS+FixMatch FixMatch |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  | DIPS+FixMatch |  |  | FixMatch |  |  | DIPS+FixMatch |  |  | FixMatch |  |  | DIPS+FixMatch |  | FixMatch |  |  |  |  | DIPS+FixMatch |  | FixMatch |  |
|  |  |  | F ndenc a synth betwee n orde dered t roport , we c rt the the su is low the su beled thoug e whic S and les is |  | ig y e n r h i o re p ( pe sa h, h hi | ure 16 betw tic exp label to be e sam on, we nduct sults ervised n = lab rvised mples that D we cou PL is gh. | : DI een erim noise able e list con the e in F bas 50) bas (an IPS ld e rema |  | PS im label ent fo level to con of lab sider n xperi ig. 17 elines and th eline. d a hig consis xpect rkably |  | p ll a t e la m . i e H h t in | roves noise owing nd am rol th l noise ∈ { b ent 40 As w n almo propo ence p level ently i real- notice | FixM leve a si ount e am pro 50, 1 tim e can st al rtion seud of n mpro worl able |  | atch l and milar s of lab ount o portio 00, 100 es. see o l the of co o-labe oise). ves th d setti for n l | o e e f ns 0 n co rr li e n ab | n imag amou tup as led da label , rang }, and the p nfigur upted ng is PL ba gs (e.g = 10 |  | es d nt of in S ta. N noise ing f fix lots, ation sam more selin . 0.1 00, i |  |  | atasets labe ec. 5.1 ote th . rom 0. n unlab PL+ s. W ples is difficu e for r ). The .e. wh | le t a t = D he h lt ea e | d dat o inve t the e o 0.45 1000 IPS co n the igh (0 with sonabl perfor n the |  |
| 100 90 accuracy 80 Test 70 Supervise 60 PL+DIPS PL 50 0.0 0.1 Propo |  |  | n_lab = 5 |  | 0 |  |  |  | 100 90 accuracy 80 Test 70 60 50 0.0 |  |  | n | _lab = 1 |  | 00 |  | accuracy Test 4 |  | 100 90 80 70 60 50 0.0 |  |  |  | n | _lab = 100 | 0 |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  | d |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  | Supervise |  | d |  |  |  |  |  |  |  | S | upervised |  |  |  |  |  |  |  |  |  | Supervis | ed |  |  |
|  | PL+DIPS PL |  |  |  |  |  |  |  |  |  | P P | L+DIPS L |  |  |  |  |  |  |  |  |  | PL+DIPS PL |  |  |  |
|  |  |  | 0.2 rtion of corrupted |  | 0. sa | 3 0.4 mples |  |  |  |  |  | 0.1 Proportio | 0.2 n of corrupt |  | 0.3 ed samples | 0. |  |  |  |  |  | 0.1 Prop | ortio | 0.2 n of corrupted | 0.3 0.4 samples |
|  |  |  | IPS co el nois |  |  | sisten levels |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

[CAPTION] Figure 16: DIPS improves FixMatch on images datasets

[CAPTION] Figure 17: DIPS consistently improves upon the PL and supervised baselines, across the


<!-- page 37 -->
You can’t handle the (dirty) truth: Data-centric insights improve pseudo-labeling
C.6
Ablation on the window of iterations for learning dynamics
We conduct an experiment to investigate the choice of the range of iterations used to compute
the learning dynamics. We consider ignoring the first 25%/50%/75% iterations and use the
remaining iterations to compute the learning dynamics.
Figure 18 shows the mean performance difference by using the truncated iteration windows
versus using all the iterations, and averages the results over the 12 datasets used in Sec. 5.2.
As we can see, it is better to use all the iterations window, as the initial iterations
carry some informative signal about the hardness of samples. This motivates our choice of
computing the learning dynamics over the whole optimization trajectory, a choice that we
adopt for all of our experiments.
Figure 18: Computing the learning dynamics over the whole window of iterations yields the
best results
37

[CAPTION] Figure 18 shows the mean performance difference by using the truncated iteration windows

[CAPTION] Figure 18: Computing the learning dynamics over the whole window of iterations yields the


<!-- page 38 -->
Seedat, Huynh, Imrie and van der Schaar
C.7
Analyzing DIPS performance with an alternative pseudo-labeling
formulation
In Sec. 3.1, we decided to describe the common pseudo-labeling methodology adopted in the
tabular setting. As a consequence, we used the implementation provided by Nguyen et al.
(2022a), which iteratively grows the training set, by including the pseudo-labels generated
at each iteration to it, thus keeping old pseudo-labels in the training dataset in subsequent
iterations. We empirically compare it to another version of confidence-based pseudo-labeling,
hence obtaining two versions:
• Version 1: with a growing set of pseudo-labels (as followed by the implementation of
Nguyen et al. (2022a) and this work)
• Version 2: without keeping old pseudo-labels between PL iterations: at each iteration,
the training set is defined by the concatenation of the original labeled set and the
pseudo-labeled samples obtained at the current iteration. Alternatively stated, for all
i = 1, ..., T −1, D(i+1) = Dlab ∪s(Dunlab, f(i)). For example, this is done in (Rizve
et al., 2021) for image experiments.
We evaluate the test accuracy achieved by these two versions in the synthetic setup described
in Sec. 5.1 and report the results in Fig. 19. The red line corresponds to Version 1 (the
implementation we used in the manuscript), while the green line corresponds to Version 2.
As we can see, in this tabular setting, growing a training set by keeping the pseudo-labels
generated at each iteration leads to the best results, motivating our adoption of this pseudo-
labeling methodology used in the tabular setting.
Additionally, we evaluate the efficacy of DIPS with this alternative formulation of pseudo-
labeling on 12 real-world tabular datasets, when using Version 2 as the pseudo-labeling
formulation and by repeating the experiment from Sec. 5.2. The results shown in Figure 20
lead to two conclusions. First, they demonstrate that DIPS helps improve the performance
of all vanilla methods. This result highlights the benefit of DIPS across both setups. More
specifically, it shows the importance of the data-centric assessment of labeled data quality
and the subsequent data characterization and selection in improving pseudo-labeling more
generally. The second conclusion is that Version 2 does not give tangible benefits over the
supervised baseline. This is expected since there is no explicit mechanism (i.e. data growing)
to make the XGBoost model change over the different iterations, unlike in Version 1. This
justifies the choice of Version 1 formulation in our manuscript.
38


<!-- page 39 -->
You can’t handle the (dirty) truth: Data-centric insights improve pseudo-labeling
0.10
0.15
0.20
0.25
0.30
0.35
0.40
0.45
Proportion of corrupted samples
60
70
80
90
Test accuracy
Implementation
With old pseudo-labels
Without old pseudo-labels
Figure 19: Growing the training set with the pseudo-labels generated at every iteration yields
the best results
PL
UPS Flex SLA CSA
81
82
Accuracy
seer
PL
UPS Flex SLA CSA
81
82
83
adult
PL
UPS Flex SLA CSA
62.5
65.0
67.5
cutract
PL
UPS Flex SLA CSA
66
68
70
covid
PL
UPS Flex SLA CSA
66
68
maggic
PL
UPS Flex SLA CSA
60.0
62.5
65.0
compas
PL
UPS Flex SLA CSA
63
64
65
Accuracy
agaricus-lepiota
PL
UPS Flex SLA CSA
68
70
German-credit
PL
UPS Flex SLA CSA
80
81
82
higgs
PL
UPS Flex SLA CSA
74
76
78
drug
PL
UPS Flex SLA CSA
78
79
80
blog
PL
UPS Flex SLA CSA
79
80
81
credit
Supervised
Vanilla
DIPS (Ours)
Figure 20: DIPS consistently improves the performance of all five pseudo-labeling methods
across the 12 real-world datasets. DIPS also reduces the performance gap between the different
pseudo-labelers.
C.8
Threshold choices
We conduct an experiment in the synthetic setup where we vary the thresholds used for
both the confidence and the aleatoric uncertainty. In addition to our choice used in our
manuscript (confidence threshold = 0.8, and adaptive threshold on the aleatoric uncertainty),
we consider two baselines:
• confidence threshold = 0.9 and uncertainty threshold = 0.1 (aggressive filtering)
• confidence threshold = 0.5 and uncertainty threshold = 0.2 (permissive filtering)
We show the test accuracy for these baselines in Figure 21. As we can see, our configuration
achieves a good trade-off between an aggressive filtering configuration (red line) and a
permissive one (blue line), which is why we adopt it for the rest of the experiments. We
empirically notice in Sec. 5.2 that it performs well on the 12 real-world datasets we used.
39


**[Table p39.1]**
| Supervised Vanilla DIPS (Ours) seer adult cutract covid maggic compas 83 70 68 65.0 67.5 Accuracy 82 88 12 65.0 68 66 62.5 81 60.0 PL UPS Flex SLA CSA PL UPS Flex SLA CSA62.5 PL UPS Flex SLA CSA 66 PL UPS Flex SLA CSA PL UPS Flex SLA CSA PL UPS Flex SLA CSA agaricus-lepiota German-credit higgs drug blog credit 65 70 82 78 80 81 Accuracy 64 81 76 79 80 63 68 80 74 78 79 PL UPS Flex SLA CSA PL UPS Flex SLA CSA PL UPS Flex SLA CSA PL UPS Flex SLA CSA PL UPS Flex SLA CSA PL UPS Flex SLA CSA |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  | L U | PS F cre | lex S dit | LA C | SA |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  | L U | PS F | lex S | LA C | SA |


**[Table p39.2]**
|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  | L U ag | PS Fl aricus | ex S -lepio | LA C ta |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  | L U | PS Fl | ex S | LA C |  |


**[Table p39.3]**
|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |
|  | L U G | PS F erma | lex S n-cred | LA C it |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  | L U | PS F | lex S | LA C |  |


**[Table p39.4]**
|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  | L U | PS Fl hig | ex S gs | LA C |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  | L U | PS Fl | ex S | LA C |  |


**[Table p39.5]**
|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |
|  | L U | PS F dr | lex S ug | LA C |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  | L U | PS F | lex S | LA C |  |


**[Table p39.6]**
|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |
|  | L U | PS F bl | lex S og | LA C |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  | L U | PS F | lex S | LA C |  |

[CAPTION] Figure 19: Growing the training set with the pseudo-labels generated at every iteration yields

[CAPTION] Figure 20: DIPS consistently improves the performance of all five pseudo-labeling methods


<!-- page 40 -->
Seedat, Huynh, Imrie and van der Schaar
0.10
0.15
0.20
0.25
0.30
0.35
0.40
0.45
Proportion of corrupted samples
50
60
70
80
90
100
Test accuracy
Thresholds
conf. threshold: 0.9, uncert. threshold: 0.1
conf. threshold: 0.8, uncert. threshold: adaptive
conf. threshold: 0.5, uncert. threshold: 0.2
Figure 21: Our choice of thresholds performs better than an aggressive or a permissive filtering
In addition to these results, we also compute the accuracy of the pseudo-labels, which is
the proportion of pseudo-labeled samples that are assigned their ground-truth label during
the pseudo-labeling process. We report the results in Figure 22, which align with the results
in Figure 21.
0.10
0.15
0.20
0.25
0.30
0.35
0.40
0.45
Proportion of corrupted samples
50
60
70
80
90
100
Accuracy for the pseudo-labels
Thresholds
conf. threshold: 0.9, uncert. threshold: 0.1
conf. threshold: 0.8, uncert. threshold: adaptive
conf. threshold: 0.5, uncert. threshold: 0.2
Figure 22: Accuracy of the pseudo-labels is improved with our choice of thresholds
C.9
Comparison to VIME
We compare DIPS with the semi-supervised method VIME (Yoon et al., 2020). We evaluate
the two methods with the same experimental setting as in Sec. 5.2. We report the results in
Table 3. These results demonstrate that DIPS outperforms VIME across multiple real-world
tabular datasets.
C.10
Importance of aleatoric uncertainty
We conduct an ablation study where we remove the aleatoric uncertainty in DIPS and only
keep a confidence-based selection (with threshold = 0.8). We term this confidence ablation to
highlight if there is indeed value to the aleatoric uncertainty component of DIPS. We report
results in Table 4, for the 12 tabular datasets used in Sec. 5.2, which shows the benefit of
the two-dimensional selection criterion of DIPS. Of course, in some cases there might not
be a large difference with respect to our confidence ablation — however, we see that DIPS
provides a statistically significant improvement in most of the datasets. Hence, since the
40

[CAPTION] Figure 21: Our choice of thresholds performs better than an aggressive or a permissive filtering

[CAPTION] Figure 22: Accuracy of the pseudo-labels is improved with our choice of thresholds

[CAPTION] Table 3. These results demonstrate that DIPS outperforms VIME across multiple real-world


<!-- page 41 -->
You can’t handle the (dirty) truth: Data-centric insights improve pseudo-labeling
Table 3: DIPS outperforms VIME. Best performing method in bold
DIPS
(Ours)
VIME
(Yoon et al., 2020)
adult
82.66 ± 0.10
67.69 ± 0.10
agaricus-lepiota
65.03 ± 0.25
66.13 ± 0.01
blog
80.58 ± 0.10
73.52 ± 0.01
credit
81.39 ± 0.07
66.91 ± 0.02
covid
69.97 ± 0.30
68.28 ± 0.03
compas
65.34 ± 0.25
63.41 ± 0.02
cutract
68.60 ± 0.31
60.36 ± 0.04
drug
78.16 ± 0.26
74.47 ± 0.03
German-credit
69.40 ± 0.46
62.65 ± 0.05
higgs
81.99 ± 0.07
71.34 ± 0.03
maggic
67.60 ± 0.08
64.98 ± 0.01
seer
82.74 ± 0.08
80.12 ± 0.01
computation is negligible, it is reasonable to use the 2-D approach given the benefit obtained
on the noisier datasets.
Table 4: Aleatoric uncertainty is a key component of DIPS
DIPS
Confidence Ablation
adult
82.66 ± 0.10
82.13 ± 0.16
agaricus-lepiota
65.03 ± 0.25
64.38 ± 0.23
blog
80.58 ± 0.10
80.22 ± 0.33
credit
81.39 ± 0.07
79.76 ± 0.15
covid
69.97 ± 0.30
69.28 ± 0.40
compas
65.34 ± 0.25
64.69 ± 0.25
cutract
68.60 ± 0.31
66.32 ± 0.12
drug
78.16 ± 0.26
75.37 ± 0.71
higgs
81.99 ± 0.07
81.42 ± 0.16
maggic
67.60 ± 0.08
66.26 ± 0.18
seer
82.74 ± 0.08
82.02 ± 0.15
C.11
Link between optimal thresholds and error rate
In this subsection, we investigate the links between the error rate in Dlab, and the associated
optimal threshold.
We conduct an experiment following the synthetic setup described in Sec. 5.1. For different
levels of corruption pcorrupt ∈[0.1, 0.4], we find the optimal threshold on the confidence τconf
by considering the list of percentiles {0.05+k×0.1 | k ∈[6]}. We show the results in Figure 23.
41

[CAPTION] Table 3: DIPS outperforms VIME. Best performing method in bold

[CAPTION] Table 4: Aleatoric uncertainty is a key component of DIPS


<!-- page 42 -->
Seedat, Huynh, Imrie and van der Schaar
0.10
0.15
0.20
0.25
0.30
0.35
0.40
Proportion of corruption
0.20
0.25
0.30
0.35
0.40
0.45
0.50
0.55
0.60
Percentile
Figure 23: Optimal percentile for τconf as a function of the proportion of corruption
We can observe two regimes. For small error rates (25%), and as the error rate increases,
it is better to discard more samples (i.e. making τconf progressively less generous). However,
for big error rates, the learning dynamics of clean samples may be severely perturbed by
the noise of the other samples. This means that setting an aggressive threshold is not ideal,
because we run the risk of also discarding valuable clean samples, and explains why the
optimal percentile decreases as the proportion of corruption increases.
C.12
DIPS in the clean label setting
Table 5: Test accuracy in the clean label setting, for the two moons dataset.
Method
Test accuracy (%)
Supervised
82.36 ± 1.46
PL
82.40 ± 1.32
PL+DIPS
84.53 ± 0.99
For the synthetic setup with two quadrants considered in Sec. 5.1, we observe that DIPS
achieves similar performance as the supervised and PL baselines in the absence of label noise.
We perform an additional experiment on the synthetic Two Moons dataset (using a synthetic
setup permits to ensure the absence of noise in the data). We generated the dataset with 100
labeled samples per class and 800 unlabeled samples, with the standard deviation used to
generate the dataset set to 0.4. Note that we do not add any label noise in this experiment.
We report the results in Table 5 (average test accuracies and 1 standard deviation reported
for 10 seeds).
As we can see, even though we did not introduce any label noise, DIPS helps with this
dataset since it enforces that training happens on easy samples via its selection mechanism.
42

[CAPTION] Figure 23: Optimal percentile for τconf as a function of the proportion of corruption

[CAPTION] Table 5: Test accuracy in the clean label setting, for the two moons dataset.