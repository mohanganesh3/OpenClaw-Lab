[2110.04202] Exploiting the Intrinsic Neighborhood Structure for Source-free Domain Adaptation

Exploiting the Intrinsic Neighborhood Structure for Source-free Domain Adaptation

Shiqi Yang  1  , Yaxing Wang  1,2  , Joost van de Weijer  1  , Luis Herranz  1  , Shangling Jui  3

1

Computer Vision Center, Universitat Autonoma de Barcelona, Barcelona, Spain

2

PCALab, Nanjing University of Science and Technology, China

3

Huawei Kirin Solution, Shanghai, China

{syang,yaxing,joost,lherranz}@cvc.uab.es ,  jui.shangling@huawei.com

Corresponding Author.

Abstract

Domain adaptation (DA) aims to alleviate the domain shift between source domain and target domain. Most DA methods require access to the source data, but often that is not possible (e.g. due to data privacy or intellectual property). In this paper, we address the challenging source-free domain adaptation (SFDA) problem, where the source pretrained model is adapted to the target domain in the absence of source data. Our method is based on the observation that target data, which might no longer align with the source domain classifier, still forms clear clusters. We capture this intrinsic structure by defining local affinity of the target data, and encourage label consistency among data with high local affinity. We observe that higher affinity should be assigned to reciprocal neighbors, and propose a self regularization loss to decrease the negative impact of noisy neighbors. Furthermore, to aggregate information with more context, we consider expanded neighborhoods with small affinity values. In the experimental results we verify that the inherent structure of the target features is an important source of information for domain adaptation. We demonstrate that this local structure can be efficiently captured by considering the local neighbors, the reciprocal neighbors, and the expanded neighborhood. Finally, we achieve state-of-the-art performance on several 2D image and 3D point cloud recognition datasets. Code is available in  https://github.com/Albert0147/SFDA_neighbors .

1  Introduction

Most deep learning methods rely on training on large amount of labeled data, while they cannot generalize well to a related yet different domain. One research direction to address this issue is Domain Adaptation (DA), which aims to transfer learned knowledge from a source to a target domain. Most existing DA methods demand labeled source data during the adaptation period, however, it is often not practical that source data are always accessible, such as when applied on data with privacy or property restrictions. Therefore, recently, there have emerged a few works  [ 16 ,  17 ,  20 ,  21 ]  tackling a new challenging DA scenario where instead of source data only the source pretrained model is available for adapting,  i.e. , source-free domain adaptation (SFDA). Among these methods, USFDA  [ 16 ]  addresses universal DA  [ 57 ]  and SF  [ 17 ]  addresses open-set DA  [ 36 ] . In both universal and open-set DA the label set is different for source and target domains. SHOT  [ 21 ]  and 3C-GAN  [ 20 ]  are for closed-set DA where source and target domains have the same categories. 3C-GAN  [ 20 ]  is based on target-style image generation with a conditional GAN, and SHOT  [ 21 ]  is based on mutual information maximization and pseudo labeling. Finally, BAIT  [ 56 ]  extends MCD  [ 35 ]  to the SFDA setting.
However, these methods ignore the intrinsic neighborhood structure of the target data in feature space which can be very valuable to tackle SFDA.

Figure 1:  ( a ) t-SNE visualization of target features by source model. ( b ) Ratio of different type of nearest neighbor features of which: the  predicted  label is the same as the feature, K is the number of nearest neighbors. The features in (a) and (b) are on task Ar

→

→

\rightarrow

Rw of Office-Home. ( c ) Illustration of our method. In the left shows we distinguish reciprocal and non-reciprocal neighbors. The adaptation is achieved by pushed the features towards reciprocal neighbors heavily.

In this paper, we focus on closed-set source-free domain adaptation. Our main observation is that current DA methods do not exploit the intrinsic neighborhood structure of the target data. We use this term to refer to the fact that, even though the target data might have shifted in the feature space (due to the covariance shift), target data of the same class is still expected to form a cluster in the embedding space. This can be implied to some degree from the t-SNE visualization of target features on the source model which suggests that significant cluster structure is preserved (see Fig.

1

(a)). This assumption is implicitly adopted by most DA methods, as instantiated by a recent DA work  [ 42 ] .
A well-established way to assess the structure of points in high-dimensional spaces is by considering the nearest neighbors of points, which are expected to belong to the same class. However, this assumption is not true for all points; the blue curve in Figure 1(b) shows that around 75% of the nearest neighbors has the correct label. In this paper, we observe that this problem can be mitigated by considering reciprocal nearest neighbors (RNN); the reciprocal neighbors of a point have the point as their neighbor. Reciprocal neighbors have been studied before in different contexts  [ 14 ,  31 ,  60 ] . The reason why reciprocal neighbors are more trustworthy is illustrated in Fig.

1  (c). Fig.

1  (b) shows the ratio of neighbors which have the  correct prediction  for different kinds of nearest neighbors. The curves show that reciprocal neighbors indeed have more chances to predict the  true  label than non-reciprocal nearest neighbors (nRNN).

The above observation and analysis motivate us to assign different weights to the supervision from nearest neighbors. Our method, called Neighborhood Reciprocity Clustering ( NRC ), achieves source-free domain adaptation by encouraging reciprocal neighbors to concord in their label prediction. In addition, we will also consider a weaker connection to the non-reciprocal neighbors. We define affinity values to describe the degree of connectivity between each data point and its neighbors, which is also utilized to encourage class-consistency between neighbors, and we propose to use a self-regularization to decrease the negative impact of potential noisy neighbors. Furthermore, inspired by recent graph based methods  [ 1 ,  3 ,  61 ]  which show that the higher order neighbors can provide relevant context, and also considering neighbors of neighbors is more likely to provide datapoints that are close on the data manifold  [ 43 ] . Thus, to aggregate wider local information, we further retrieve the expanded neighbors,  i.e , neighbor of the nearest neighbors, for auxiliary supervision.

Our contributions can be summarized as follows, to achieve source-free domain adaptation: (i) we explicitly exploit the fact that same-class data forms cluster in the target embedding space, we do this by considering the predictions of neighbors and reciprocal neighbors, (ii) we further show that considering an extended neighborhood of data points further improves results (iii) the experiments results on three 2D image datasets and one 3D point cloud dataset show that our method achieves state-of-the-art performance compared with related methods.

2  Related Work

Domain Adaptation. 
Most DA methods tackle domain shift by aligning the feature distributions. Early DA methods such as  [ 23 ,  41 ,  45 ]  adopt moment matching to align feature distributions. And in recent years, plenty of works have emerged that achieve alignment by adversarial training. DANN  [ 7 ]  formulates domain adaptation as an adversarial two-player game. The adversarial training of CDAN  [ 24 ]  is conditioned on several sources of information. DIRT-T  [ 40 ]  performs domain adversarial training with an added term that penalizes violations of the cluster assumption. Additio