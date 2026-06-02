[2304.01973] ERM++: An Improved Baseline for Domain Generalization

ERM++: An Improved Baseline for Domain Generalization

Piotr Teterwak  1  , Kuniaki Saito  1  ,Theodoros Tsiligkaridis  2  , Kate Saenko  1,3  , Bryan A. Plummer  1

1  Boston University,

2  MIT Lincoln Laboratory,

3  Meta AI

{piotrt,keisaito,saenko,bplum}@bu.edu, ttsili@ll.mit.edu

Abstract

Multi-source Domain Generalization (DG) measures a classifier’s ability to generalize to new distributions of data it was not trained on, given several training domains. While several multi-source DG methods have been proposed, they incur additional complexity during training by using domain labels. Recent work has shown that a well-tuned Empirical Risk Minimization (ERM) training procedure, that is simply minimizing the empirical risk on the source domains, can outperform most existing DG methods. We identify several key candidate techniques to further improve ERM performance, such as better utilization of training data, model parameter selection, and weight-space regularization. We call the resulting method ERM++, and show it significantly improves the performance of DG on five multi-source datasets by over 5% compared to standard ERM, and beats state-of-the-art despite being less computationally expensive. Additionally, we demonstrate the efficacy of ERM++ on the WILDS-FMOW dataset, a challenging DG benchmark. We hope that ERM++ becomes a strong baseline for future DG research. Code is released at  https://github.com/piotr-teterwak/erm_plusplus .

Figure 1:

ERM++: 
We tackle the task of Multi-Source Domain Generalization, where a model is trained on several source domains and evaluated on a different target domain. We do this by improving the classic, and already strong, ERM  [ 20 ]  algorithm with known methodologies. We verify our method on a diverse set of domain shifts, and show that it improves over the best reported numbers in the literature.

1  Introduction

Domain Generalization (DG) is a crucial problem in the field of machine learning, as it addresses the challenge of building models that perform well on unseen (target) data distributions, without using target data to update the model  [ 7 ,  39 ,  64 ] . This is important in many real-world applications, where the distribution of data may vary between settings, and it is not always feasible to collect and label a large amount of data for each new domain. Similarly, it is not always known a-priori how the distribution on which the model is deployed differs from the training distribution. In multi-source domain generalization, each training sample is labelled as being part of one of several domains. Many advanced methods leverage domain membership explicitly. For example, DANN  [ 18 ]  uses an adversarial loss to match feature distributions across source domains. Adaptive Risk Minimization  [ 62 ]  meta-learns parameters which adapt a model to newly seen distribution shift. Yet, recently DomainBed  [ 20 ]  holistically evaluated methods and found that ERM (Empirical Risk Minimization), outperforms most prior work for DG in a setting where hyper-parameters are tuned. This is all the more impressive since ERM only leverages domain labels in a very weak way; by oversampling minority domains to balance domain sizes in the training data. Advanced techniques do not beat ERM  [ 20 ]  despite strong inductive biases and additional complexities (and hyper-parameters to tune).

In this paper, our goal is to revisit the framework used to benchmark multisource domain generalization problems to ensure that we maximize the performance of baseline methods. As illustrated in Figure

1  , our new baseline, ERM++, is able to outperform the state-of-the-art without the need for domain labels, architecture changes or complex training strategies. Instead, we critically evaluate the components of the training pipeline along three major themes. First, we explore how the training data is being used, including training length and checkpoint selection. Second, we consider how we initialize network parameters such as the selection of pretraining network and whether or not to fine-tune or freeze layers. Third, we investigate weight-space regularization methods that are often used to help avoid overfitting to the training data.

Revisiting and improving baselines to address shortcomings or incorporating new training techniques can provide new insights into the state of the research on the topic. For example, SimSiam  [ 12 ]  showed that a simple siamese network can perform competitively on self-supervised learning by incorporating a stop-gradient function. Beyer

et al .

[ 5 ]  show that a few simple techniques, such as longer training or increased augmentation strength, outperform all prior work in knowledge distillation. Wightman  et al .

[ 55 ]  show that techniques such as longer training can substantially improve ImageNet  [ 14 ]  performance. These works helped provide new insights into their respective tasks, as we aim to do in our work for domain generalization.

Through a careful evaluation of the training framework used to compare DG methods across six diverse domains, we are able to make several interesting observations. For example, we find that improved performance on ImageNet  [ 14 ]  does not necessitate a gain in generalization ability. We also find that many of the hyperparameters such as training time used by many methods (such as DomainBed  [ 20 ] ) result in evaluating models before they have converged. To address this, we utilized an adaptive training procedure that would automatically determine the sufficient training length for a model to obtain the best performance. Compared to the state-of-the-art DG methods such as MIRO  [ 10 ]  and DIWA  [ 42 ] , our approach is able to obtain a 1% gain across five datasets used by DomainBed  [ 20 ] , while also reducing the required training compute by 50% compared with MIRO and 95% compared to DIWA due to reduced need for hyperparameter tuning. Critically, although we also show that using the techniques we identified boosts MIRO and DIWA’s performance, the improved DIWA is unable to outperform ERM++. This helps highlight the need for our work.

Algorithm 1

ERM++:  Components of ERM++ are annotated in the algorithm comments. We run this training loop in two passes, the first to set training length by using a validation set split from the source domains. In the second pass we train on the combination of train and val data.

M  ​  o  ​  d  ​  e  ​  l  ​  W  ​  e  ​  i  ​  g  ​  h  ​  t  ​  s

←

A  ​  u  ​  g  ​  m  ​  i  ​  x  ​  W  ​  e  ​  i  ​  g  ​  h  ​  t  ​  s

←

𝑀  𝑜  𝑑  𝑒  𝑙  𝑊  𝑒  𝑖  𝑔  ℎ  𝑡  𝑠

𝐴  𝑢  𝑔  𝑚  𝑖  𝑥  𝑊  𝑒  𝑖  𝑔  ℎ  𝑡  𝑠

ModelWeights\leftarrow AugmixWeights

▷

▷

\triangleright

Strong init

while

s  ​  t  ​  e  ​  p  ​  s

≠

L  ​  o  ​  n  ​  g  ​  T  ​  r  ​  a  ​  i  ​  n  ​  S  ​  t  ​  e  ​  p  ​  s

𝑠  𝑡  𝑒  𝑝  𝑠

𝐿  𝑜  𝑛  𝑔  𝑇  𝑟  𝑎  𝑖  𝑛  𝑆  𝑡  𝑒  𝑝  𝑠

steps\neq LongTrainSteps

do

▷

▷

\triangleright

Long Train

X  ,  Y

←

←

𝑋  𝑌

absent

X,Y\leftarrow

next(FullDataIterator)

▷

▷

\triangleright

Full Data

if

s  ​  t  ​  e  ​  p  ​  s

≤  500

𝑠  𝑡  𝑒  𝑝  𝑠

500

steps\leq 500

then

Update linear classifier

▷

▷

\triangleright

Warmstart

else

Update linear classifier and backbone

end

if

if

s  ​  t  ​  e  ​  p  ​  s

≥  600

𝑠  𝑡  𝑒  𝑝  𝑠

600

steps\geq 600

then

M  ​  o  ​  d  ​  e  ​  l  ​  W  ​  e  ​  i  ​  g  ​  h  ​  t  ​  s  ​  A  ​  v  ​  g

←

←

𝑀  𝑜  𝑑  𝑒  𝑙  𝑊  𝑒  𝑖  𝑔  ℎ  𝑡  𝑠  𝐴  𝑣  𝑔

absent

ModelWeightsAvg\leftarrow

Update

▷

▷

\triangleright

Weight Reg.

M  ​  o  ​  d  ​  e  ​  l  ​  W  ​  e  ​  i  ​  g  ​  h  ​  t  ​  s  ​  A  ​  v  ​  g  ​  B  ​  N

←

←

𝑀  𝑜  𝑑  𝑒  𝑙  𝑊  𝑒  𝑖  𝑔  ℎ  𝑡  𝑠  𝐴  𝑣  𝑔  𝐵  𝑁

absent

ModelWeightsAvgBN\leftarrow

BN stats

▷

▷

\triangleright

UBN

end

if

end

while

2  Related Works

In this work, w