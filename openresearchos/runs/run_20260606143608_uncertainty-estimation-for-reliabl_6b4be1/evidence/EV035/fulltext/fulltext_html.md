[2102.03526] Open-World Semi-Supervised Learning

Open-World Semi-Supervised Learning

Kaidi Cao  ,  Maria Brbić  ∗  ,  Jure Leskovec
 Department of Computer Science
 Stanford University

{kaidicao, mbrbic, jure}@cs.stanford.edu

The two first authors made equal contributions.

Abstract

A fundamental limitation of applying semi-supervised learning in real-world settings is the assumption that unlabeled test data contains only classes previously encountered in the labeled training data. However, this assumption rarely holds for data in-the-wild, where instances belonging to novel classes may appear at testing time. Here, we introduce a novel  open-world semi-supervised learning  setting that formalizes the notion that novel classes may appear in the unlabeled test data. In this novel setting, the goal is to solve the class distribution mismatch between labeled and unlabeled data, where at the test time every input instance either needs to be classified into one of the existing classes or a new unseen class needs to be initialized. To tackle this challenging problem, we propose ORCA, an end-to-end deep learning approach that introduces uncertainty adaptive margin mechanism to circumvent the bias towards seen classes caused by learning discriminative features for seen classes faster than for the novel classes. In this way, ORCA reduces the gap between intra-class variance of seen with respect to novel classes. Experiments on image classification datasets and a single-cell annotation dataset demonstrate that ORCA consistently outperforms alternative baselines, achieving

25  %

percent  25

25\%

improvement on seen and

96  %

percent  96

96\%

improvement on novel classes of the ImageNet dataset.

1  Introduction

With the advent of deep learning, remarkable breakthroughs have been achieved and current machine learning systems excel on tasks with large quantities of labeled data  (LeCun et al.,  2015 ; Silver et al.,  2016 ; Esteva et al.,  2017 ) . Despite the strengths, the vast majority of models are designed for the closed-world setting rooted in the assumption that training and test data come from the same set of predefined classes  (Bendale &amp; Boult,  2015 ; Boult et al.,  2019 ) . This assumption, however, rarely holds for data in-the-wild, as labeling data depends on having the complete knowledge of a given domain. For example, biologists may prelabel known cell types (seen classes), and then want to apply the model to a new tissue to identify known cell types but also to  discover novel  previously unknown cell types (unseen classes). Similarly, in social networks one may want to classify users into predefined interest groups while also discovering new unknown/unlabeled
interests of users. Thus, in contrast to the commonly assumed closed world, many real-world problems are inherently open-world — new classes can emerge in the test data that have never been seen (and
labeled) during training.

Here we introduce  open-world semi-supervised learning  (open-world SSL) setting that generalizes semi-supervised learning and novel class discovery.
Under open-world SSL, we are given a labeled training dataset as well as an unlabeled dataset. The labeled dataset contains instances that belong to a set of  seen classes , while instances in the unlabeled/test dataset belong to both the seen classes as well as to an unknown number of  unseen classes  (Figure

1  ). Under this setting, the model needs to either classify instances into one of the previously seen classes, or discover new classes and assign instances to them. In other words, open-world SSL is a transductive learning setting under class distribution mismatch in which unlabeled test set may contain classes that have never been labeled during training,  i.e. , are not part of the labeled training set.

Open-world SSL is fundamentally different but closely related to two recent lines of work: robust semi-supervised learning (SSL) and novel class discovery. Robust SSL  (Oliver et al.,  2018 ; Guo et al.,  2020 ; Chen et al.,  2020b ; Guo et al.,  2020 ; Yu et al.,  2020 )  assumes class distribution mismatch between labeled and unlabeled data, but in this setting the model only needs to be able to recognize (reject) instances belonging to novel classes in the unlabeled data as out-of-distribution instances. In contrast, instead of rejecting instances belonging to novel classes, open-world SSL aims at discovering individual novel classes and then assigning instances to them.
Novel class discovery  (Hsu et al.,  2018 ;  2019 ; Han et al.,  2019 ;  2020 ; Zhong et al.,  2021 )  is a clustering problem where one assumes unlabeled data is composed only of novel classes. In contrast, open-world SSL is more general as instances in the unlabeled data can come from seen as well as from novel classes. To apply robust SSL and novel class discovery methods to open-world SSL, one could in principle adopt a multi-step approach by first using robust SSL to reject instances from novel classes and then applying a novel class discovery method on rejected instances to discover novel classes. An alternative would be that one could treat all classes as “novel”, apply novel class discovery methods and then match some of the classes back to the seen classes in the labeled dataset. However, our experiments show that such ad hoc approaches do not perform well in practice. Therefore, it is necessary to design a method that can solve this practical problem in an end-to-end framework.

Figure 1:  In the open-world SSL, the unlabeled dataset may contain classes that have never been encountered in the labeled set. Given unlabeled test set, the model needs to either assign instances to one of the classes previously seen in the labeled set, or form a novel class and assign instances to it.

In this paper we propose ORCA (Open-woRld with unCertainty based Adaptive margin) that operates under the novel open-world SSL setting. ORCA effectively assigns examples from the unlabeled data to either previously seen classes, or forms novel classes by grouping similar instances. ORCA is an end-to-end deep learning framework, where the key to our approach is a novel uncertainty adaptive margin mechanism that gradually decreases plasticity and increases discriminability of the model during training. This mechanism effectively reduces an undesired gap between intra-class variance of seen with respect to the novel classes caused by learning seen classes faster than the novel, which we show is a critical difficulty in this setting. We then develop a special model training procedure that learns to classify data points into a set of previously seen classes while also learning to use an additional classification head for each newly discovered class. Classification heads for seen classes are used to assign the unlabeled examples to classes from the labeled set, while activating additional classification heads allows ORCA to form a novel class. ORCA does not need to know the number of novel classes ahead of time and can automatically discover them at the deployment time.

We evaluate ORCA on three benchmark image classification datasets adapted for open-world SSL setting and a single-cell annotation dataset from biology domain. Since no existing methods can operate under the open-world SSL setting we extend existing state-of-the-art SSL, open-set recognition and novel class discovery methods to the open-world SSL and compare them to ORCA. Experimental results demonstrate that ORCA effectively addresses the challenges of open-world SSL setting and consistently outperforms all baselines by a large margin. Specifically, ORCA achieves

25  %

percent  25

25\%

and

96  %

percent  96

96\%

improvements on seen and novel classes of the ImageNet dataset. Moreover, we show that ORCA is robust to unknown number of novel classes, different distributions of seen and novel classes, unbalanced data distributions, pretraining strategies and a small num