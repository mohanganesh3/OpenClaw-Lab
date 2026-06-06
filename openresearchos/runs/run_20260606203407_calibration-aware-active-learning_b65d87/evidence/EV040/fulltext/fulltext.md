<!-- page 1 -->
Ask-n-Learn: Active Learning via Reliable Gradient Representations
for Image Classiﬁcation
Bindya Venkatesh
BVENKA15@ASU.EDU
Arizona State University
Tempe, AZ, USA
Jayaraman J. Thiagarajan
JJAYARAM@LLNL.GOV
Lawrence Livermore National Labs
Livermore, CA,USA
Abstract
Deep predictive models rely on human supervision in the form of labeled training data. Ob-
taining large amounts of annotated training data can be expensive and time consuming, and this
becomes a critical bottleneck while building such models in practice. In such scenarios, active
learning (AL) strategies are used to achieve faster convergence in terms of labeling efforts. Ex-
isting active learning employ a variety of heuristics based on uncertainty and diversity to select
query samples. Despite their wide-spread use, in practice, their performance is limited by a num-
ber of factors including non-calibrated uncertainties, insufﬁcient trade-off between data exploration
and exploitation, presence of conﬁrmation bias etc. In order to address these challenges, we pro-
pose Ask-n-Learn, an active learning approach based on gradient embeddings obtained using the
pesudo-labels estimated in each iteration of the algorithm. More importantly, we advocate the use
of prediction calibration to obtain reliable gradient embeddings, and propose a data augmentation
strategy to alleviate the effects of conﬁrmation bias during pseudo-labeling. Through empirical
studies on benchmark image classiﬁcation tasks (CIFAR-10, SVHN, Fashion-MNIST, MNIST),
we demonstrate signiﬁcant improvements over state-of-the-art baselines, including the recently
proposed BADGE algorithm.
1. Introduction
The superior performance of data-driven methods, including deep learning, comes at the price of
requiring large amounts of labeled data. This can be a critical bottleneck in applications involv-
ing time-consuming data acquisition or high labeling costs. Furthermore, fully supervised methods
assume access to samples representing the entire data distribution beforehand, thus making it chal-
lenging to handle changes in data distribution over time or adapt the learned model when diverse
samples are incrementally included into the training process. This has motivated the use of active
learning techniques that involve humans in the training loop to build predictive models that are more
data-efﬁcient.
Broadly, the goal of active learning is to select the most useful samples for expert annotation,
from an unlabeled dataset, while adhering to a given labeling budget Settles (2009). More speciﬁ-
cally, we are interested in settings where the labeling is carried out in batches Settles (2009). The
data selection process is at the core of active learning methods and can be designed based on a
variety of heuristics including prediction uncertainty Gal et al. (2017), diversity Brinker (2003);
1
arXiv:2009.14448v1  [stat.ML]  30 Sep 2020


<!-- page 2 -->
Sener and Savarese (2017) and model generalization Freytag et al. (2014); Cai et al. (2013). For a
classiﬁer, it is well known that samples near the decision boundary tend to be associated with large
uncertainties and hence including them into the training set can help improve the model’s gener-
alization Beluch et al. (2018); Settles (2009). On the other hand, to avoid sampling biases in the
learned models, one needs to ensure that the training dataset sufﬁciently captures the diversity of
the inherent data distribution Sener and Savarese (2017); Geifman and El-Yaniv (2017). Balanc-
ing between these two complementary objectives is often referred to as the exploitation-exploration
trade-off Bondu et al. (2010). Interestingly, hybrid methods that leverage this trade-off tend to out-
perform approaches that rely only on either criterion Hsu and Lin (2015); Baram et al. (2004).
While a large class of heuristics exist in the literature for measuring uncertainties and sample
diversity, recently, Ash et al. Ash et al. (2019) showed that gradient embedding of unlabeled data,
obtained using pseudo-labels from the model’s current state, is highly effective for both capturing
uncertainties and measuring diversity. Pseudo-labeling refers to the process of generating labels
for unlabeled data to drive the sample selection process. Despite their effectiveness in building
predictive models under minimal supervision Ash et al. (2019), methods based on pseudo-labeling
are known to suffer from the undesirable behavior of conﬁrmation bias when incorrect pseudo-labels
are used in the learning process. Note that, this problem arises even in semi-supervised learning and
is typically addressed by including additional regularization to the pseudo-labeling process Arazo
et al. (2019). On the other hand, existing methods that directly utilize uncertainty scores for sample
selection, e.g. entropy, are known to perform poorly when the uncertainties are not well calibrated.
In this paper, we propose a new active learning approach, Ask-n-Learn, which addresses the
inherent limitations with both pseudo-labeling and uncertainty based methods. While our approach
uses gradient embeddings, similar to BADGE Ash et al. (2019), for selecting samples, we propose to
utilize calibrated uncertainties to produce reliable gradient embeddings, and employ a data augmen-
tation strategy for avoiding conﬁrmation bias during pseudo-labeling. Using benchmark image clas-
siﬁcation tasks (SVHN, MNIST, Fashion-MNIST and CIFAR-10), and different deep architectures,
we show that the proposed approach signiﬁcantly outperforms existing active learning approaches,
including BADGE Ash et al. (2019), both in terms of test accuracy and prediction calibration.
2. Related Work
The long-standing problem of active learning (AL) is aimed at enabling faster model generalization,
in terms of the amount of supervision required. AL methods can be broadly categorized based on
a number of factors including the type of query (e.g., stream based, pool based query etc), sample
selection technique, label acquisition strategy etc. Settles (2009). More speciﬁcally, in terms of the
sample selection, there exists two main classes of AL methods, namely diversity and uncertainty
based methods. In the former approach, by promoting diversity, one expects to obtain a minimal
subset of samples that covers the variations expected in the entire data distribution. For example,
Brinker (2003); Guo and Schuurmans (2008) proposed diversity based heuristics to select a batch
of unlabeled samples in AL tasks. In Wang and Ye (2015), the authors develop a batch mode AL
strategy based on empirical risk minimization principle. The AL problem was posed as a core-set
selection task in Sener and Savarese (2017), wherein a new heuristic to select diverse samples,
speciﬁcally for convolutional networks, was designed. Finally, a more recent class of diverse sam-
2


<!-- page 3 -->
pling methods has been based on adversarial learning Sinha et al. (2019); Ducoffe and Precioso
(2018); Gissin and Shalev-Shwartz (2019); Zhu and Bento (2017).
On the other hand, uncertainty based AL methods aim at selecting data samples that the model
is most likely uncertain about. Several uncertainty heuristics including entropy, conﬁdence and
distance-measures have been proposed in the literature for both Bayesian Bondu et al. (2010); Gal
et al. (2017) and non-Bayesian settings Yang and Loog (2016). In Kirsch et al. (2019), the authors
introduced an acquisition strategy based on mutual information for Bayesian active learning. Sim-
ilarly, Bayesian model uncertainties were utilized for AL in high-dimensional data spaces in Gal
et al. (2017). AL methods based on uncertainties from ensembles techniques have also been found
to be effective Melville and Mooney (2004); Beluch et al. (2018). Given the challenges in quan-
tifying model and data uncertainties, recently, Kiyasseh et al. (2020) investigated the use of data
perturbations to design an acquisition function. A task-agnostic loss prediction framework was pro-
posed in Yoo and Kweon (2019) to select samples that the model is most likely to misclassify.
Similarly, Konyushkova et al. (2017) proposed to train a regressor to predict the expected error
reduction for samples, in AL tasks.
In addition to the uncertainty and diversity based methods, hybrid methods combine these
heuristics have also been proposed Yang et al. (2017). For instance, Li and Guo (2013) proposed
an adaptive AL approach to combine the information density and uncertainty measure fors image
classiﬁcation. Sinha et al. Sinha et al. (2019) proposed a hybrid method by combining variational au-
toencoders and adversarial learning frameworks. Several existing works have systematically studied
the trade-off between using the uncertainty and representative sampling as exploration-exploitation
for AL Loy et al. (2012); Bondu et al. (2010); Osugi et al. (2005). Hybrid methods based on quan-
tifying the model generalization have become more popular recently Freytag et al. (2014); Cai
et al. (2013); Huang et al. (2016); Bouneffouf (2016), including the state-of-the-art BADGE al-
gorithm Ash et al. (2019). When compared to existing methods, Ask-n-Learn also falls under the
category of hybrid methods and leverages gradient embeddings to perform the sample selection.
Through crucial modiﬁcations to the AL pipeline, namely uncertainty calibration and data aug-
mentation for conﬁrmation bias control, it achieves signiﬁcantly better convergence compared to
existing methods.
3. Proposed Approach
3.0.1 NOTATIONS.
Given the input data sample x ⊂X ∈Rd, we consider a K-way multi-class classiﬁcation task,
i.e., predicting the target label y ⊂Y ∈{1, 2, . . . , K}. In this paper, our goal is to select the most
informative query samples to build a classiﬁer f : X 7→Y. We assume that the initial seed dataset
of size S is drawn randomly from the train dataset and annotated, i.e., S = {(xi, yi)}S
i=1. Given
the total annotation budget B, we develop a batch selection strategy, i.e. pick a subset of b samples
from an unlabeled pool, U = {xj}N
j=1, in every iteration. Finally, we use the set L = {xi, yi}M
i=1
to denote the labeled training data, which grows by including newly annotated samples through the
algorithm. Note, in the ﬁrst iteration, L = S.
3


<!-- page 4 -->
Figure 1: An illustration of the proposed active learning approach Ask-n-Learn, that utilizes reliable
gradient representation obtained via calibrated classiﬁer models and a data-augmentation strategy
for reducing conﬁrmation bias.
3.1 Formulation
Similar to the recently proposed BADGE algorithm Ash et al. (2019), our approach utilizes the gra-
dient embeddings to design the sampling heuristic. More speciﬁcally, in every iteration, we compute
the gradient embeddings for all samples in U based on the loss function computed with respect to
the the pseudo-labels estimated using the current state of the classiﬁer f. Note that, pseudo-labels
are hard labels obtained by picking the class with the highest softmax probability. In contrast to ex-
isting active learning approaches that use explicit predictive uncertainties, the lengths of the gradient
vectors provide an estimate of the inherent uncertainties. Finally, similar to Ash et al. (2019), we
use the K-means++ seeding algorithm Arthur and Vassilvitskii (2006) on the gradient embedding to
select the query samples, thus promoting the diversity objective.
Despite outperforming existing active learning methods Ash et al. (2019), the success of this
sample selection heuristic relies heavily on the quality of the pseudo-labels and the resulting gradient
embeddings. Hence, in this work, we advocate for the use of well-calibrated predictive models to
obtain reliable gradient embeddings, and we also propose to leverage data augmentation strategies to
alleviate the effects of conﬁrmation bias arising due to incorrect pseudo-labeling. As we show in our
empirical studies, with these key modiﬁcations to the sampling process, Ask-n-Learn consistently
outperforms BADGE at all sample sizes.
Building Calibrated Classiﬁers.
Calibration metrics are typically used to measure the consis-
tency between a model’s prediction probabilities and the true likelihood (or accuracy). It has been
showed in several recent works that by utilizing an explicit calibration objective during training can
lead to highly reliable predictive models Nixon et al. (2019); Guo et al. (2017). In this paper, we
show that such a calibrated model is necessary to obtain effective gradient embeddings for subse-
quent sampling. Formally, the training objective to build a well-calibrated classiﬁer can be expressed
as
ˆθ = argmin
θ
ℓf = ℓce + λℓcalib,
(1)
where ℓf represents the overall loss that needs to be minimized using the labeled training data L.
This objective is a combination of the standard cross-entropy loss ℓce and a calibration objective
ℓcalib with a regularization weight λ.
4

[CAPTION] Figure 1: An illustration of the proposed active learning approach Ask-n-Learn, that utilizes reliable


<!-- page 5 -->
In this work, we consider two different calibration objectives, namely the Variance Weighted
Conﬁdence Calibration (VWCC) and the Likelihood Weighted Conﬁdence Calibration (LWCC).
Note this regularized optimization eliminates the need for a separate calibration dataset, as required
by post-hoc calibration strategies Guo et al. (2017).
(a) Variance Weighted Conﬁdence Calibration (VWCC): In this strategy we employ stochastic infer-
ences to capture the epistemic uncertainties to calibrate the classiﬁer’s conﬁdence in its predictions.
In particular, we utilize the objective introduced in Seo et al. (2019), where the variance across mul-
tiple stochastic inferences are used to regularize the optimization. Here, we utilize a label-smoothing
regularization with penalty λ speciﬁed as the measured variance. Mathematically, this is as follows:
ℓV WCC = 1
M
M
X
i=1
(1 −αi)ℓi
ce + ℓi
calib
= 1
M
M
X
i=1
T
X
t=1
−(1 −αi) log(p(ˆyt
i|xi))
+ αiDKL(U(y)||p(ˆyt
i|xi)).
(2)
Here, the softmax predictions are obtained via T stochastic inferences for each sample xi, i.e.,
p(ˆyt
i|xi), t = 1 · · · T, and the variance across these predictions αi is estimated via the Bhattacharyya
coefﬁcients. When the variance is high, the loss function is designed to encourage the softmax
predictions to be smoothed to an uniform distribution U, via using KL-divergence objective. This
ensures that the prediction distribution has an higher entropy when the model is more uncertain
about its prediction.
(b) Likelihood Weighted Conﬁdence Calibration (LWCC): We propose a new calibration objective
based on the estimated likelihoods, which does not require multiple stochastic inferences. Similar to
the VWCC approach, we use the likelihood estimates to control the label smoothing regularization.
In particular,
ℓLWCC =
M
X
i=1
ℓi
ce + λ[βiDKL(U(y)||p(ˆyi|xi))],
where βi =
 
1 −max(ˆyi)
 I(yi=ˆyi)
.
(3)
When the softmax prediction p(ˆyi|xi) for a sample xi is both incorrect and associated with high
conﬁdence, those overconﬁdent predictions are adjusted to produce a high-entropy distribution.
Obtaining Gradient Embeddings.
Given the unlabeled set U, let {ˆyj}N
j=1 represent the pseudo-
labels predicted by the model f(U; θ). The gradient embeddings for all x ∈U are computed as the
gradient of the loss function w.r.t the last layer of the network. Note that, even with a calibrated
model, we use only ℓce to compute the gradients. More speciﬁcally, the gradient gx of a sample x
w.r.t to the last layer θo of the network f(x, θ) can be written as
(gx)i = (pi −I(ˆy = i)) z(x; θ\θo),
(4)
5


<!-- page 6 -->
where z represents a non-linear transformation applied before the ﬁnal layer (denote by θ\θo). The
model’s uncertainty about an unlabeled sample is captured via the gradient norm, since the gradient
w.r.t to the optimization objective represents the expected model change and its direction.
Reducing Conﬁrmation Bias via Data Augmentation.
We propose to reﬁne the pseudo-label
estimates for unlabeled samples, prior to using them in gradient embedding computation. More
speciﬁcally, given our calibrated predictive model, we retain the pseudo-labels when their conﬁ-
dence (maximum softmax probability) exceeds a preset threshold. In cases when this condition is
not satisﬁed, we obtain more reliable pseudo-label estimates by averaging predictions over multiple
augmented versions of the unlabeled samples. In our implementation, we use pseudo-labels obtained
by averaging predictions from k randomly augmented versions of an unlabeled sample (details of
the augmentation are given in the experiments section). Note that, existing semi-supervised learning
approaches routinely use this idea to produce highly consistent classiﬁer models Zhu and Goldberg
(2009); Zhu (2005). Such a label regularization strategy based on calibrated predictions reduces the
conﬁrmation bias, as this does not completely rely on the model’s existing belief.
Exploration-Exploitation Trade-off.
While designing an active learning heuristic, balancing the
trade-off between exploration and exploitation of the data space plays a crucial role in selecting
informative samples. In the context of active learning, exploration refers to selecting samples to
query from an non-sampled region of data distribution, while exploitation refers to selecting query
samples from data space that is already sampled. The task of exploration requires the AL heuristic
to select diverse samples that are not already sampled, similarly the exploitation task requires to
select samples that the model is most uncertain about and the samples that are difﬁcult to learn from
(closer to the decision boundary). As showed in our results, our proposed strategies for model cali-
bration and pseudo-label estimation, combined with the K-means++ seeding achieves signiﬁcantly
better trade-off between these two objectives, thus leading to better quality classiﬁers for the same
sampling budget.
3.2 Algorithm
The steps involved in the proposed Ask-n-Learn method are detailed in Algorithm: 1. Given the
labeled seed data, the ﬁrst step is to build a well-calibrated classiﬁer (equation 2 or 3). At each
active learning iteration, the pseudo-labels for x ∈U are estimated using the proposed strategy.
These pseudo-labels are then used to compute the gradient embedding representations, from which
a batch of diverse samples are selected via K-means++. Selecting samples with high gradient norms
and different directions allows the active learning strategy to consider both the uncertainty and
diversity factors while sampling the query points. Assuming that the expert annotates the selected
query samples, the model is retrained using the updated labeled set L. This process is repeated until
the annotation budget is exhausted.
4. Empirical Studies
In this work, we consider deep image classiﬁcation tasks and compare the model generalization
performance, as a function of the number of labeled training samples utilized, from different active
learning methods. In order to evaluate the reliability of the resulting models, we also measure predic-
6


<!-- page 7 -->
Algorithm 1: Ask-n-Learn
Input: Seed data S, Unlabeled set U, Query set Q;
Labeling budget B, Batch size b, Conﬁdence threshold τ, Number of augmentations
k.
Output: Trained classiﬁer f(θ; L)
Initialize: Parameters θ of model f, Set L = S;
while budget B ̸= 0 do
for xi in U do
Train : Minimize Ex∈L [ℓf(f(x; θ), y)] using VWCC or LWCC regularization;
compute yi = f(xi; θ) ;
if max p(yi|xi; θ) ≥τ then
ˆyi = argmax p(yi|xi; θ)
else
Obtain x(j)
i
= Aug(xi), j = 1, · · · , k; ˆyi = argmax
  1
k
Pk
j=1 p(yi|x(j)
i ; θ)
 
end
compute gradient embeddings using eqn. (4) and pseudo-label ˆyi;
end
Sampling : Use K-means++ on the gradient embeddings and select a batch of b samples
from U to form a Query set Q;
Update : U ←−U −Q,
L ←−L ∪Q,
B ←−B −b
end
Return the ﬁnal trained model
5000
10000
15000
Number of Labels Queried
0.80
0.85
0.90
0.95
Accuracy
5000
10000
15000
Number of Labels Queried
0.00
0.02
0.04
0.06
0.08
ECE
5000
10000
15000
Number of Labels Queried
0.1
0.2
0.3
0.4
0.5
0.6
NLL
5000
10000
15000
Number of Labels Queried
0.05
0.10
0.15
0.20
0.25
Brier Score
Random
Entropy
Confidence
BADGE
Ours (VWCC)
Ours (LWCC)
Figure 2: MNIST: Generalization performance and reliability analysis of Ask-n-Learn. We plot the
mean and standard deviation for each of the metrics as training data is added in batches (b = 1000).
tion calibration via standard metrics. Finally, we evaluate the robustness of the proposed approach
when the oracle is noisy, a commonly encountered challenge in practice.
4.1 Experiment setup
Datasets and Architectures:
In our study, we consider four benchmark image classiﬁcation tasks,
namely CIFAR-10 Krizhevsky et al., SVHN Netzer et al. (2011), Fashion-MNIST Xiao et al. (2017)
and MNIST LeCun and Cortes (2010) datasets. We use the standard ResNet-18 He et al. (2015)
7

[CAPTION] Figure 2: MNIST: Generalization performance and reliability analysis of Ask-n-Learn. We plot the


<!-- page 8 -->
5000
10000
15000
Number of Labels Queried
0.750
0.775
0.800
0.825
0.850
0.875
0.900
Accuracy
5000
10000
15000
Number of Labels Queried
0.050
0.075
0.100
0.125
0.150
0.175
ECE
5000
10000
15000
Number of Labels Queried
0.4
0.6
0.8
1.0
1.2
NLL
5000
10000
15000
Number of Labels Queried
0.15
0.20
0.25
0.30
0.35
0.40
Brier Score
Random
Entropy
Confidence
BADGE
Ours (VWCC)
Ours (LWCC)
Figure 3: Fashion-MNIST: Performance of Ask-n-Learn in comparison to the baseline methods.
With batch size b = 1000, we show the results obtained using a 3−layer fully connected network.
10000
20000
30000
40000
Number of Labels Queried
0.4
0.5
0.6
0.7
0.8
Accuracy
10000
20000
30000
40000
Number of Labels Queried
0.10
0.15
0.20
0.25
0.30
0.35
ECE
10000
20000
30000
40000
Number of Labels Queried
1.0
1.5
2.0
2.5
NLL
10000
20000
30000
40000
Number of Labels Queried
0.3
0.4
0.5
0.6
0.7
0.8
0.9
Brier Score
Random
Entropy
Confidence
BADGE
Ours (VWCC)
Ours (LWCC)
Figure 4: CIFAR-10: Ask-n-Learn provides signiﬁcant improvements in terms of both accuracy and
calibration metrics, particularly at lower training sizes. In this case, we use the ResNet-18 architec-
ture with b = 5000.
VGG-16 Simonyan and Zisserman (2014) architectures for CIFAR-10 and SVHN classiﬁcation ex-
periments respectively. A three layer MLP with conﬁguration [256−256−256] and ReLU activation
was used for Fashion-MNIST and MNIST experiments.
Hyperparameter choices:
We implemented the proposed Ask-n-Learn active learning strategy as
shown in Algorithm 1. The batch size (b) of samples queried in each active learning cycle was ﬁxed
to 5K samples for the CIFAR-10 dataset, and 1K samples for SVHN, Fashion-MNIST and MNIST.
All the models were retrained from scratch in every active learning cycle and trained until conver-
gence. The baselines were implemented following hyperparameter settings in Ash et al. (2019) and
trained with the standard cross-entropy loss. We used the Adam optimizer with a learning rate of
0.001. All models were trained with an initial random seed data of size 100. The experiments were
repeated for three trials with different random seeds, and average performance along with their
variances are reported.
Baselines:
The proposed active learning strategy including the two calibration training variations
are compared against four baselines including the standard active learning heuristics namely the
conﬁdence based methods Wang and Shang (2014); Ash et al. (2019), entropy based methods. Most
importantly, we also compare it against the state-of-the-art BADGE algorithm Ash et al. (2019) and
8

[CAPTION] Figure 3: Fashion-MNIST: Performance of Ask-n-Learn in comparison to the baseline methods.

[CAPTION] Figure 4: CIFAR-10: Ask-n-Learn provides signiﬁcant improvements in terms of both accuracy and


<!-- page 9 -->
5000
10000
15000
Number of Labels Queried
0.75
0.80
0.85
0.90
Accuracy
5000
10000
15000
Number of Labels Queried
0.025
0.050
0.075
0.100
0.125
0.150
0.175
ECE
5000
10000
15000
Number of Labels Queried
0.4
0.6
0.8
1.0
1.2
NLL
5000
10000
15000
Number of Labels Queried
0.1
0.2
0.3
0.4
Brier Score
Random
Entropy
Confidence
BADGE
Ours (VWCC)
Ours (LWCC)
Figure 5: SVHN: While even simpler AL heuristics such as entropy and prediction conﬁdence
perform comparatively, Ask-n-Learn, in particular LWCC, provides non-trivial performance gains
(1% −2%) when the number o queried labels was low.
10000
20000
30000
40000
Number of Labels Queried
0.50
0.55
0.60
0.65
0.70
Accuracy
10000
20000
30000
40000
Number of Labels Queried
0.15
0.20
0.25
0.30
0.35
ECE
10000
20000
30000
40000
Number of Labels Queried
1.5
2.0
2.5
NLL
10000
20000
30000
40000
Number of Labels Queried
0.4
0.5
0.6
0.7
0.8
Brier Score
Random
BADGE
Ours (VWCC)
Ours (LWCC)
(a) Noise Ratio = 0.1
10000
20000
30000
40000
Number of Labels Queried
0.45
0.50
0.55
0.60
0.65
Accuracy
10000
20000
30000
40000
Number of Labels Queried
0.20
0.25
0.30
0.35
ECE
10000
20000
30000
40000
Number of Labels Queried
1.50
1.75
2.00
2.25
2.50
2.75
NLL
10000
20000
30000
40000
Number of Labels Queried
0.5
0.6
0.7
0.8
0.9
Brier Score
Random
BADGE
Ours (VWCC)
Ours (LWCC)
(b) Noise Ratio = 0.2
Figure 6: Generalization performance and reliability analysis of active learning strategies on CIFAR-
10, in the presence of a noisy oracle: (a) 10% and (b) 20% of the correct oracle labels were randomly
corrupted in each iteration of the AL algorithm.
na¨ıve random sampling. The data selection heuristic in the conﬁdence-based AL method is based on
conﬁdences from the model predictions, wherein a batch of b samples with lowest class probability
predictions are selected Wang and Shang (2014). In the entropy-based AL baseline, data samples
in an unlabeled pool are ranked according to the entropy of the model softmax predictions. The
BADGE algorithm Ash et al. (2019) aims to select batches of most informative query samples from
9

[CAPTION] Figure 5: SVHN: While even simpler AL heuristics such as entropy and prediction conﬁdence

[CAPTION] Figure 6: Generalization performance and reliability analysis of active learning strategies on CIFAR-


<!-- page 10 -->
an unlabeled pool via sampling based on the directions and lengths of the gradients. The random
baseline represents the scenario where a batch of samples are randomly selected from the pool of
unlabeled samples for the oracle (or human annotator) to label. Consequently, this represents the
conventional fully-supervised training protocol.
Evaluation Metrics:
In order to evaluate the model’s generalization performance on test data, we
use the conventional accuracy metric. The reliability analysis of the trained models are carried out
using prediction calibration metrics. In particular, we use the standard metrics namely the expected
calibration error (ECE), negative-log-likelihood (NLL) and Brier-scores Guo et al. (2017); Ovadia
et al. (2019); Nixon et al. (2019). The ECE metric meaures the discrepancy between the accuracy
and model conﬁdence. This metric is computed by dividing the predictions into equal sized bins and
averaging the difference between the accuracy in each bin with its corresponding conﬁdence.
ECE =
B
X
b=1
Nb
N |acc (Bb) −conf (Bb)| ,
where Nb represents the number of predictions falling in bin b and acc(Bb) is the accuracy and
conf(Bb) the average conﬁdence of the samples in bin b. The negative-log likelihood or the cross
entropy loss is used to measure the quality of the models probabilistic predictions and hence its
uncertainties. Finally, the Brier score is computed as square of the difference between the models’
softmax predictions and the true labels, and often used to measure the accuracy of the probabilistic
predictions.
4.2 Results and Findings
Generalization Performance.
The efﬁcacy of an active learning heuristic is measured via the
generalization performance of the trained model in terms of labeling efforts. Hence, we evaluate the
different active learning methods through convergence plots of accuracy versus the number of la-
beled training data utilized to achieve it. As observed in Figure 2, our proposed approach with both
LWCC and VWCC calibration variations performs better than the baselines on the MNIST classi-
ﬁcation task. More speciﬁcally, the convergence of our approach in terms of accuracy is achieved
faster when compared to other active learning heuristics. We observe a similar behaviour in Fashion-
MNIST (Figure 3), CIFAR-10 (Figure 4) and SVHN (Figure 5) classiﬁcation tasks, where our pro-
posed approach consistently outperforms the baselines including the BADGE. For instance, in the
CIFAR-10 experiment, test accuracy of about 70% was achieved with just around 10K samples with
our VWCC based method when compared to baselines that require nearly double the sample size to
achieve the same accuracy. This clearly evidences the effectiveness of Ask-n-Learn in selecting the
most informative samples for improved generalization. Additionally, by enabling a principled char-
acterization of prediction probabilities for computing the gradient embeddings, we observe that the
stochastic inferencing based calibration strategy (VWCC) appears to be superior to the likelihood
based calibration (LWCC), except on the SVHN data where the latter achieves faster convergence.
These observations clearly illustrate the importance of using reliable gradients combined with the
data augmentation strategies that alleviates the effects of conﬁrmation bias in the training pipeline.
Reliability Analysis.
While accuracy is a widely-adopted metric for model evaluation, it becomes
critical to analyse the reliability of deep predictive models. For instance, a model associated with
10


<!-- page 11 -->
overconﬁdent probabilities while making an incorrect prediction is undesirable. Hence, in addition
to evaluating the accuracy of AL methods, we assess the reliability of the trained models by mea-
suring the consistency between model accuracy and the prediction probabilities (ECE, NLL, Brier
score). We observe from Figures 2 to 5 that the proposed strategy consistently produces well cali-
brated predictive models across all the datasets. In terms of all three metrics, Ask-n-Learn produce
signiﬁcantly lower calibration errors compared to the baselines.
Robustness to a Noisy Oracle.
Active learning heuristics are designed with an assumption that
the training labels are annotated accurately. However, in practice, this assumption may be violated
and in those cases, the labeling noise (imperfect oracle) can impact the performance of any active
learning pipeline. Hence, it is critical to analyze the performance of AL methods in the presence of
a noisy oracle. To this end, we repeat the CIFAR-10 classiﬁcation experiments with varying levels
of labeling noise. To simulate such a scenario, a pre-speciﬁed ratio of query samples are wrongly
labeled in every iteration of the AL process. In particular, we perform experiments where 10% and
20% of the true labels are randomly replaced with incorrect ones and used for training. From Fig-
ures 6a and 6b it can be seen that the models designed using the proposed active learning heuristic,
in particular VWCC, is signiﬁcantly more robust to labeling noise. More interestingly, during the
initial few iterations, the performance of BADGE is similar to that of na¨ıve random sampling. In
comparison, Ask-n-Learn achieves > 10% higher accuracy than the baselines in both 10% and
20% noise cases. This implies that the effects of conﬁrmation bias and predictive miscalibration are
reduced through the proposed strategies, thus leading to improved sample selection.
5. Conclusions
In this work, we proposed an active learning framework Ask-n-Learn, based on reliable gradient
representations for deep image classiﬁcation tasks. We addressed the inherent limitations of both
pseudo-labeling and uncertainty scoring based active learning methods, via prediction calibration
and data-augmentation based pseudo-label modiﬁcation. We demonstrated the sample efﬁciency
of Ask-n-Learn across various benchmark classiﬁcation tasks and model architectures, in terms of
both accuracy and empirical calibration metrics. Finally, we studied the robustness our approach in
an imperfect oracle scenario, and showed that it generalized better when compared to the popular
baselines.
6. Acknowledgements
This work was performed under the auspices of the U.S. Department of Energy by Lawrence Liver-
more National Laboratory under Contract DE-AC52-07NA27344.
References
Eric Arazo, Diego Ortego, Paul Albert, Noel E O’Connor, and Kevin McGuinness.
Pseudo-labeling and conﬁrmation bias in deep semi-supervised learning.
arXiv
preprint arXiv:1908.02983, 2019.
11


<!-- page 12 -->
David Arthur and Sergei Vassilvitskii. k-means++: The advantages of careful seeding.
Technical report, Stanford, 2006.
Jordan T Ash, Chicheng Zhang, Akshay Krishnamurthy, John Langford, and Alekh
Agarwal. Deep batch active learning by diverse, uncertain gradient lower bounds.
arXiv preprint arXiv:1906.03671, 2019.
Yoram Baram, Ran El Yaniv, and Kobi Luz. Online choice of active learning algo-
rithms. Journal of Machine Learning Research, 5(Mar):255–291, 2004.
William H Beluch, Tim Genewein, Andreas N¨urnberger, and Jan M K¨ohler. The power
of ensembles for active learning in image classiﬁcation. In Proceedings of the IEEE
Conference on Computer Vision and Pattern Recognition, pages 9368–9377, 2018.
Alexis Bondu, Vincent Lemaire, and Marc Boull´e. Exploration vs. exploitation in ac-
tive learning: A bayesian approach. In The 2010 International Joint Conference on
Neural Networks (IJCNN), pages 1–7. IEEE, 2010.
Djallel Bouneffouf. Exponentiated gradient exploration for active learning. Computers,
5(1):1, 2016.
Klaus Brinker. Incorporating diversity in active learning with support vector machines.
In Proceedings of the 20th international conference on machine learning (ICML-03),
pages 59–66, 2003.
Wenbin Cai, Ya Zhang, and Jun Zhou. Maximizing expected model change for active
learning in regression. In 2013 IEEE 13th International Conference on Data Mining,
pages 51–60. IEEE, 2013.
David Cohn, Les Atlas, and Richard Ladner.
Improving generalization with active
learning. Machine learning, 15(2):201–221, 1994.
David A Cohn, Zoubin Ghahramani, and Michael I Jordan. Active learning with statis-
tical models. Journal of artiﬁcial intelligence research, 4:129–145, 1996.
Sanjoy Dasgupta. Two faces of active learning. Theoretical computer science, 412(19):
1767–1781, 2011.
Melanie Ducoffe and Frederic Precioso. Adversarial active learning for deep networks:
a margin based approach. arXiv preprint arXiv:1802.09841, 2018.
Meng Fang, Yuan Li, and Trevor Cohn. Learning how to active learn: A deep reinforce-
ment learning approach. arXiv preprint arXiv:1708.02383, 2017.
Alexander Freytag, Erik Rodner, and Joachim Denzler. Selecting inﬂuential examples:
Active learning with expected model output changes. In European Conference on
Computer Vision, pages 562–577. Springer, 2014.
Yarin Gal, Riashat Islam, and Zoubin Ghahramani. Deep bayesian active learning with
image data. arXiv preprint arXiv:1703.02910, 2017.
12


<!-- page 13 -->
Yonatan Geifman and Ran El-Yaniv. Deep active learning over the long tail. arXiv
preprint arXiv:1711.00941, 2017.
Daniel Gissin and Shai Shalev-Shwartz. Discriminative active learning. arXiv preprint
arXiv:1907.06347, 2019.
Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern
neural networks. arXiv preprint arXiv:1706.04599, 2017.
Yuhong Guo and Dale Schuurmans. Discriminative batch mode active learning. In
Advances in neural information processing systems, pages 593–600, 2008.
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for
image recognition. corr abs/1512.03385 (2015), 2015.
Wei-Ning Hsu and Hsuan-Tien Lin. Active learning by learning. In Twenty-Ninth AAAI
conference on artiﬁcial intelligence. Citeseer, 2015.
Jiaji Huang, Rewon Child, Vinay Rao, Hairong Liu, Sanjeev Satheesh, and Adam
Coates.
Active learning for speech recognition: the power of gradients.
arXiv
preprint arXiv:1612.03226, 2016.
David A Huffaker and Sandra L Calvert. The new science of learning: Active learn-
ing, metacognition, and transfer of knowledge in e-learning applications. Journal of
Educational Computing Research, 29(3):325–334, 2003.
Longlong Jing and Yingli Tian. Self-supervised visual feature learning with deep neu-
ral networks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelli-
gence, 2020.
Andreas Kirsch, Joost van Amersfoort, and Yarin Gal. Batchbald: Efﬁcient and diverse
batch acquisition for deep bayesian active learning. In Advances in Neural Informa-
tion Processing Systems, pages 7026–7037, 2019.
Dani Kiyasseh, Tingting Zhu, and David A Clifton. Alps: Active learning via pertur-
bations. arXiv preprint arXiv:2004.09557, 2020.
Ksenia Konyushkova, Raphael Sznitman, and Pascal Fua. Learning active learning
from data. In Advances in Neural Information Processing Systems, pages 4225–
4235, 2017.
Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. Cifar-10 (canadian institute for
advanced research). URL http://www.cs.toronto.edu/∼kriz/cifar.html.
Yann LeCun and Corinna Cortes. MNIST handwritten digit database. 2010. URL
http://yann.lecun.com/exdb/mnist/.
Xin Li and Yuhong Guo. Adaptive active learning for image classiﬁcation. In Proceed-
ings of the IEEE Conference on Computer Vision and Pattern Recognition, pages
859–866, 2013.
13


<!-- page 14 -->
Chen Change Loy, Timothy M Hospedales, Tao Xiang, and Shaogang Gong. Stream-
based joint exploration-exploitation active learning. In 2012 IEEE Conference on
Computer Vision and Pattern Recognition, pages 1560–1567. IEEE, 2012.
Prem Melville and Raymond J Mooney. Diverse ensembles for active learning. In Pro-
ceedings of the twenty-ﬁrst international conference on Machine learning, page 74,
2004.
Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y
Ng. Reading digits in natural images with unsupervised feature learning. 2011.
Jeremy Nixon, Michael W Dusenberry, Linchuan Zhang, Ghassen Jerfel, and Dustin
Tran. Measuring calibration in deep learning. In CVPR Workshops, pages 38–41,
2019.
Thomas Osugi, Deng Kim, and Stephen Scott. Balancing exploration and exploitation:
A new algorithm for active machine learning. In Fifth IEEE International Conference
on Data Mining (ICDM’05), pages 8–pp. IEEE, 2005.
Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, David Sculley, Sebastian Nowozin,
Joshua Dillon, Balaji Lakshminarayanan, and Jasper Snoek.
Can you trust your
model’s uncertainty? evaluating predictive uncertainty under dataset shift. In Ad-
vances in Neural Information Processing Systems, pages 13991–14002, 2019.
Jack O’Neill, Sarah Jane Delany, and Brian MacNamee. Model-free and model-based
active learning for regression. In Advances in computational intelligence systems,
pages 375–386. Springer, 2017.
Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary
DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic
differentiation in pytorch. 2017.
Ozan Sener and Silvio Savarese. Active learning for convolutional neural networks: A
core-set approach. arXiv preprint arXiv:1708.00489, 2017.
Seonguk Seo, Paul Hongsuck Seo, and Bohyung Han. Learning for single-shot conﬁ-
dence calibration in deep neural networks through stochastic inferences. In Proceed-
ings of the IEEE Conference on Computer Vision and Pattern Recognition, pages
9030–9038, 2019.
Burr Settles.
Active learning literature survey.
Technical report, University of
Wisconsin-Madison Department of Computer Sciences, 2009.
Jingyu Shao, Qing Wang, and Fangbing Liu. Learning to sample: an active learning
framework. In 2019 IEEE International Conference on Data Mining (ICDM), pages
538–547. IEEE, 2019.
Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-
scale image recognition. arXiv preprint arXiv:1409.1556, 2014.
14


<!-- page 15 -->
Samarth Sinha, Sayna Ebrahimi, and Trevor Darrell.
Variational adversarial active
learning. In Proceedings of the IEEE International Conference on Computer Vision,
pages 5972–5981, 2019.
Dan Wang and Yi Shang. A new active labeling method for deep learning. In 2014
International joint conference on neural networks (IJCNN), pages 112–119. IEEE,
2014.
Zheng Wang and Jieping Ye. Querying discriminative and representative samples for
batch mode active learning. ACM Transactions on Knowledge Discovery from Data
(TKDD), 9(3):1–23, 2015.
Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset
for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747,
2017.
Lin Yang, Yizhe Zhang, Jianxu Chen, Siyuan Zhang, and Danny Z Chen. Suggestive
annotation: A deep active learning framework for biomedical image segmentation.
In International conference on medical image computing and computer-assisted in-
tervention, pages 399–407. Springer, 2017.
Yazhou Yang and Marco Loog. Active learning using uncertainty information. In 2016
23rd International Conference on Pattern Recognition (ICPR), pages 2646–2651.
IEEE, 2016.
Donggeun Yoo and In So Kweon. Learning loss for active learning. In Proceedings of
the IEEE Conference on Computer Vision and Pattern Recognition, pages 93–102,
2019.
Jia-Jie Zhu and Jos´e Bento. Generative adversarial active learning. arXiv preprint
arXiv:1702.07956, 2017.
Xiaojin Zhu and Andrew B Goldberg. Introduction to semi-supervised learning. Syn-
thesis lectures on artiﬁcial intelligence and machine learning, 3(1):1–130, 2009.
Xiaojin Jerry Zhu. Semi-supervised learning literature survey. Technical report, Uni-
versity of Wisconsin-Madison Department of Computer Sciences, 2005.
15