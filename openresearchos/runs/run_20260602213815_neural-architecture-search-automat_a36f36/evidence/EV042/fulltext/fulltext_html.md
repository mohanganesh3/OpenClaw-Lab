[2201.12725] Generalized Global Ranking-Aware Neural Architecture Ranker for Efficient Image Classifier Search

Generalized Global Ranking-Aware Neural Architecture Ranker for Efficient Image Classifier Search

Bicheng Guo 1 , Tao Chen 2 , Shibo He 1 , Haoyu Liu 3 , Lilin Xu 1 , Peng Ye 2 , Jiming Chen 1

1 Zhejiang University,  2 Fudan University,  3 Fuxi AI Lab, NetEase Games

{guobc, s18he, cjm}@zju.edu.cn
 {eetchen, yepeng20}@fudan.edu.cn
 liuhaoyu03@corp.netease.com

Corresponding Author.

Abstract

Neural Architecture Search (NAS) is a powerful tool for automating effective image processing DNN designing. The ranking has been advocated to design an efficient performance predictor for NAS. The previous contrastive method solves the ranking problem by comparing pairs of architectures and predicting their relative performance. However, it only focuses on the rankings between two involved architectures and neglects the overall quality distributions of the search space, which may suffer generalization issues. A predictor, namely Neural Architecture Ranker (NAR) which concentrates on the global quality tier of specific architecture, is proposed to tackle such problems caused by the local perspective. The NAR explores the quality tiers of the search space globally and classifies each individual to the tier they belong to according to its global ranking. Thus, the predictor gains the knowledge of the performance distributions of the search space which helps to generalize its ranking ability to the datasets more easily. Meanwhile, the global quality distribution facilitates the search phase by directly sampling candidates according to the statistics of quality tiers, which is free of training a search algorithm, e.g., Reinforcement Learning (RL) or Evolutionary Algorithm (EA), thus it simplifies the NAS pipeline and saves the computational overheads. The proposed NAR achieves better performance than the state-of-the-art methods on two widely used datasets for NAS research. On the vast search space of NAS-Bench-101, the NAR easily finds the architecture with top 0.01‰ performance only by sampling. It also generalizes well to different image datasets of NAS-Bench-201, i.e., CIFAR-10, CIFAR-100, and ImageNet-16-120 by identifying the optimal architectures for each of them.

Figure 1 :

(a) Comparison between previous performance predictors (top and middle) and ours (bottom). Different from predicting the absolute or pair-wise comparison performance, the NAR ranks the architectures with quality tier and scores them relative metric. (b)  our NAR generalizes well to various image datasets by identifying the optimal architecture for each (CIFAR-10, CIFAR-100, and ImageNet-16-120).

1  Introduction

Deep neural networks (DNNs) have received great attention in recent years. Many DNNs that are artificially designed by researchers have been applied to many scenarios, such as image classification  [ 37 ,  18 ] , object detection  [ 36 ,  35 ] , semantic segmentation  [ 26 ,  3 ]  and other real-world applications  [ 17 ,  48 ,  49 ] . Though these artificially designed DNNs have been proved to be powerful, designing them requires rich human expertise and is labor-intensive. Furthermore, dedicated knowledge is needed in the network architecture design for many specific target domains.

Neural architecture search (NAS) offers a powerful tool for automating effective DNN designing for specific objectives. Previous studies directly apply different search and optimization methods, including Reinforcement Learning (RL)  [ 50 ,  32 ] , Evolutionary Algorithm (EA)  [ 24 ,  33 ] , and differentiable methods  [ 25 ,  2 ] , to find candidates in the search space. In order to reduce the prohibitive cost in evaluating a population of candidates, performance predictor is proposed to replace the evaluation metrics with the predicted performance of architectures  [ 29 ,  1 ,  41 ] .
However, these predictors try to approximate the absolute performance of architectures and suffer the ranking problem, i.e., the architectures with similar ground-truth performance have the incorrect predicted rankings due to the prediction bias, as shown in the top of Fig.

1

(a). As a result, the search algorithm could be misled to select the low-ranking architectures and yields the deteriorate results  [ 43 ] .

The most recent contrastive method

[ 6 ]  solves the ranking problem by comparing pairs of sampled architectures and calculating the probability that one architecture is better than the other, as shown in the middle of Fig.

1

(a). Even though it can achieve good ranking ability, such an approach may suffer from generalization issues. This is because it only focuses on the rankings between two involved architectures and neglects the overall quality distributions of the search space. As a result, this limits the predictor only to the architecture ranking instead of generalizing its ranking ability over various target datasets. To solve this, we propose to utilize the quality distributions of the search space where the architectures are classified into multiple quality tiers by their global rankings according to the ground-truth performance. In this way, we can train the predictor by learning the features of quality tiers and classifying each individual to the tier they belong to according to its global ranking. This classification enhanced ranking paradigm has also been explored previously  [ 15 ] . The rational,  class membership information is important for determining a quality ranking over a dataset

[ 20 ] , inspires us to first roughly classify the quality of the architecture and then score them. As a result, the predictor gains the global knowledge of the performance distributions of the search space which helps to generalize the ranking ability to various datasets more easily than previous local methods.

During the search phase, most of the previous studies either adopt reinforcement learning  [ 50 ,  38 ]  or evolutionary algorithms  [ 16 ,  28 ] , which requires additional training cost and complicates the NAS pipeline. Interestingly, we can benefit from collecting the distributions of the top quality tiers and focus on the outperformed architectures by directly sampling with them in the search space. This makes our method free of training an RL controller or employing EA methods, thus saves the computational overheads.

In this work, we propose a Neural Architecture Ranker (NAR) to learn the global distributions of the architectures with various quality and identify each individual’s quality level (tier) among the search space according to its performance (bottom of Fig.

1

(a)).
Specifically, we first divide the search space into five quality tiers according to the performance distributions of the architectures.
Then, each individual architectures is encoded to represent its structural and computational feature, and is matched with the embeddings of all tiers alternately to decide which tier it belongs to.
In this way, we relax the performance prediction into quality classification problem. We also leverage the extracted feature to predict the relative scores of the sampled architectures.
Consequently, the NAR is capable of ranking and scoring the candidates according to their global ranking among the search space, and generalizes well to various image datasets as shown in Fig.

1

(b). Furthermore, the distributions of the different quality tiers are collected to guide the sampling procedure in the search phase, which requires no additional computational overheads for searching and simplifies the NAS pipeline. The overall contribution is summarized as follows:

•

Different from locally comparing pairs of architectures and calculating relative probability, we propose a Neural Architecture Ranker (NAR) that ranks and scores the architectures by matching them with the representation of various quality tiers from a global perspective.

•

We propose to collect the distribut