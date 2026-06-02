[2407.07668] How to Leverage Predictive Uncertainty Estimates for Reducing Catastrophic Forgetting in Online Continual Learning

How to Leverage Predictive Uncertainty Estimates for

Reducing Catastrophic Forgetting in Online Continual Learning

Giuseppe Serra

serra@med.uni-frankfurt.de

Goethe University Frankfurt

German Cancer Consortium (DKTK)

,

Ben Werner

s8939364@stud.uni-frankfurt.de

Goethe University Frankfurt

and

Florian Buettner

florian.buettner@dkfz-heidelberg.de

Goethe University Frankfurt

German Cancer Consortium (DKTK)

German Cancer Research Center

Abstract.

Many real-world applications require machine-learning models to be able to deal with non-stationary data distributions and thus learn autonomously over an extended period of time, often in an online setting. One of the main challenges in this scenario is the so-called catastrophic forgetting (CF) for which the learning model tends to focus on the most recent tasks while experiencing predictive degradation on older ones. In the online setting, the most effective solutions employ a fixed-size memory buffer to store old samples used for replay when training on new tasks. Many approaches have been presented to tackle this problem. However, it is not clear how predictive uncertainty information for memory management can be leveraged in the most effective manner and conflicting strategies are proposed to populate the memory. Are the easiest-to-forget or the easiest-to-remember samples more effective in combating CF? Starting from the intuition that predictive uncertainty provides an idea of the samples’ location in the decision space, this work presents an in-depth analysis of different uncertainty estimates and strategies for populating the memory. The investigation provides a better understanding of the characteristics data points should have for alleviating CF. Then, we propose an alternative method for estimating predictive uncertainty via the generalised variance induced by the negative log-likelihood. Finally, we demonstrate that the use of predictive uncertainty measures helps in reducing CF in different settings.

Uncertainty, Online Continual Learning, Catastrophic Forgetting

†

†  conference:  3rd Workshop on Uncertainty Reasoning and Quantification in Decision Making; August 26, 2024; Barcelona, Spain

1.  Introduction

Typical machine learning models assume to work in a single-task static scenario where multiple epochs are performed over the same data until convergence. In many real-world situations, however, this setting is too limiting. As an example, let us consider the problem of classifying new COVID-19 variants. Given the evolving nature of the virus, new variants (i.e., new classes) arise over time. In this context, a typical learning model would fail since the standard setting does not consider the dynamic increment of new classes. For this reason,  online  Continual Learning (online-CL) has been constantly more explored. In online-CL, a single model is required to learn continuously from a sequence of tasks that comes as a stream of tiny batches which can be processed only once  (Aljundi et al . ,  2019 ,  2017 ; Mai et al . ,  2022 ) . Given the overlap between old and new information, the model tends to forget about the past knowledge to focus more on the newest tasks, leading to a performance degradation on previous tasks. This challenge is usually referred to as catastrophic forgetting (CF)  (McCloskey and Cohen,  1989 ; Ratcliff,  1990 ) . Many approaches have been developed to prevent catastrophic forgetting and the most successful ones employ a memory buffer for rehearsal  (Chaudhry et al . ,  2019 ; Soutif-Cormerais et al . ,  2023 ) . Memory-based strategies tackle CF by training the model on both current samples and some old samples stored in a limited size memory. In this context, what differentiates each memory-based approach are the retrieval strategy and the update strategy  (Mai et al . ,  2022 )  — i.e., how to populate and update the memory with meaningful and representative samples, and how to efficiently sample from the memory respectively. Although many approaches have been developed in this direction – ranging from using random approaches  (Chaudhry et al . ,  2019 )  to exploiting the gradient  (Lopez-Paz and Ranzato,  2017 ; Chaudhry et al . ,  2018b ; Aljundi et al . ,  2019 )  or the loss  (Belilovsky et al . ,  2019 )  information – the effect of predictive uncertainty estimates for memory management is largely unexplored. One of the few methods in this direction is presented in  Bang et al .  ( 2021 ) , where the authors propose a combination of predictive uncertainty and data augmentation at the memory-level to generate a memory of diverse samples – from the most uncertain to the most representative ones – for each class.
Despite remarkable results, the contribution of the uncertainty score on the overall performance is not easily quantifiable. Furthermore, considering the quickly expanding landscape of recently rehearsal-based methods proposed, it is difficult to identify a clear strategy to exploit the memory at its best. In fact, contrasting strategies can be found in the literature. Some of them suggest to use the most representative samples  (Yoon et al . ,  2021 ; Hurtado et al . ,  2023 ) , while others consider the samples near the decision boundary as the most useful to reduce CF  (Kumari et al . ,  2022 ) . Hence, it is unclear which type of data points would reduce CF in a consistent manner: Are the easiest-to-forget or the easiest-to-remember samples more effective in combating CF? To answer these and other questions, this paper investigates the contribution of predictive uncertainty scores on reducing CF. Intuitively, using predictive uncertainty to populate the memory represents a solution for identifying the location of the instances in the decision space and considering either the most representative – i.e., the  k  samples with the  lowest  uncertainty (bottom-k) – or the marginal ones (i.e., the top-k with the  highest  uncertainty). In the first part of this work, we evaluate and compare different combinations of uncertainty scores and sorting on CIFAR-10
 (Krizhevsky et al . ,  2009 ) , a dataset commonly used in CL. The objective is to understand how different uncertainty estimates behave when used in different ways under same conditions. In order to achieve the intended goal, following a recent trend on research transparency and comparability  (Mundt et al . ,  2021 ) , the evaluation framework will be freed from any other ’trick’ that could create ambiguity in the assessment. The investigation will focus on describing the characteristics of the considered uncertainty scores, while providing an in-depth analysis of the effect of the metrics under a memory-based regime. In addition to the comparison of established uncertainty scores, we propose a new selection strategy based on recent advances in uncertainty quantification via the logit space. More specifically, we introduce a memory management strategy based on the Bregman Information (BI) as a generalised variance measure, which stems directly from a bias-variance decomposition of the model loss  (Gruber and Buettner,  2023 ) .
 In the second part of our work, we evaluate our findings on more challenging and realistic scenarios. In realistic setups, recent tasks have less data points available than older tasks as the time to collect instances is considerably shorter than for previous tasks. Considering this behaviour, we will focus on long-tailed (LT) datasets with this characteristic. We test our findings on two datasets with controllable degrees of data imbalance, as well as a real-world imbalanced dataset for classification of biomedical images  (Yang et al . ,  2023 ) .

The main contributions of this paper are the followings:

•

We systematically evaluate the performance of established predictive uncertainty scores in reducing CF in online-CL f