[2106.11197] Iterative Network Pruning with Uncertainty Regularization for Lifelong Sentiment Classification

Iterative Network Pruning with Uncertainty Regularization

for Lifelong Sentiment Classification

Binzong Geng  1,2∗  , Min Yang  2†  , Fajie Yuan  3,6∗  , Shupeng Wang  2  , Xiang Ao  4  , Ruifeng Xu  5

1  University of Science and Technology of China

2  Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences

3  Westlake University

4  Key Lab of Intelligent Information Processing of Chinese Academy of Sciences, Institute of Computing Technology, CAS

5  Harbin Institute of Technology (Shenzhen)

6  Tencent

bz.geng, min.yang, wang.sp@siat.ac.cn, yuanfajie@westlake.edu.cn, aoxiang@ict.ac.cn, xuruifeng@hit.edu.cn

(2021)

Abstract.

Lifelong learning capabilities are crucial for sentiment classifiers to process continuous streams of opinioned information on the Web.
However, performing lifelong learning is non-trivial for deep neural networks as continually training of incrementally available information inevitably results in catastrophic forgetting or interference. In this paper, we propose a novel  i terative network  p runing with uncertainty  r egularization method for  l ifelong  s entiment classification (IPRLS), which leverages the principles of network pruning and weight regularization.
By performing network pruning with uncertainty regularization in an iterative manner, IPRLS can adapt a single BERT model to work with continuously arriving data from multiple domains while avoiding catastrophic forgetting and interference.
Specifically, we leverage an iterative pruning method to remove redundant parameters in large deep networks so that the freed-up space can then be employed to learn new tasks, tackling the catastrophic forgetting problem. Instead of keeping the old-tasks fixed when learning new tasks, we also use an uncertainty regularization based on the Bayesian online learning framework to constrain the update of old tasks weights in BERT, which enables positive backward transfer, i.e. learning new tasks improves performance on past tasks while protecting old knowledge from being lost.
In addition, we propose a task-specific low-dimensional residual function in parallel to each layer of BERT, which makes IPRLS less prone to losing the knowledge saved in the base BERT network when learning a new task.
Extensive experiments on 16 popular review corpora demonstrate that the proposed IPRLS method significantly outperforms the strong baselines for lifelong sentiment classification.
For reproducibility, we submit the code and data at:  https://github.com/siat-nlp/IPRLS .

Lifelong learning, sentiment classification, network pruning, uncertainty regularization

∗  This work was done when Binzong Geng interned at SIAT, Chinese Academy of Sciences. This work was done when Fajie Yuan worked at Tencent (past affiliation) and Westlake University (current affiliation).

†  Min Yang is corresponding author.

†

†  journalyear:  2021

†

†  copyright:  acmcopyright

†

†  conference:  Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval; July 11–15, 2021; Virtual Event, Canada

†

†  booktitle:  Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’21), July 11–15, 2021, Virtual Event, Canada

†

†  price:  15.00

†

†  doi:  10.1145/3404835.3462902

†

†  isbn:  978-1-4503-8037-9/21/07

†

†  ccs:  Computing Methodologies Lifelong Learning

†

†  ccs:  Applied Computing Document Management and Text Processing

1.  Introduction

With the increase of large collections of opinion-rich documents on the Web, much focus has been given to sentiment classification that targets at automatically predicting the sentiment polarity of given text. In recent years, deep learning has achieved great success and been almost dominant in the field of sentiment classification  (Tai
et al . ,  2015 ; Wang
et al . ,  2016 ; Wen et al . ,  2019 ) . Powerful deep neural networks
have to depend on large amounts of annotated
training resources. However, labeling large datasets is usually time-consuming and labor-intensive, creating significant barriers when applying the trained sentiment classifier to new domains. In addition, no matter how much data is collected and used to train a sentiment classifier, it is difficult to cover all possible domains of opinioned data on the Web. Thus, when deployed in practice, a well-trained sentiment classifier often performs unsatisfactorily.

A sentiment classifier operating in a production environment often encounters
continuous streams of information  (Parisi et al . ,  2019 )  and thereby is required to expand its knowledge to new domains  1

1  1 Note that by a task, we mean a category of products. In this paper, we use the terms domain and task interchangeably, because each of our tasks is from a distinct domain.

.
The ability to continually learn over time by grasping new knowledge and remembering previously learned experiences is referred to as lifelong or continual learning  (Parisi et al . ,  2019 ) .
Recently, there are several studies  (Chen et al . ,  2015 ; Wang
et al . ,  2018 ; Lv
et al . ,  2019 )  that utilize lifelong learning to boost the performance of sentiment classification in a changing environment.

Chen et al .

( 2015 )  proposed a lifelong learning method for sentiment classification based on Naive Bayesian framework and stochastic gradient descent.

Lv
et al .

( 2019 )  extended the work of  (Chen et al . ,  2015 )  with a neural network based approach.
However, the performances of previous lifelong sentiment classification techniques are far from satisfactory.

Lifelong learning has been a long-standing challenge for deep neural networks. It is difficult to automatically balance the trade-off between stability and plasticity in lifelong learning  (Mermillod
et al . ,  2013 ; Ahn
et al . ,  2019 ) . On the one hand, a sentiment classifier is expected to reuse previously acquired knowledge, but focusing too much on stability may hinder the classifier from quickly adapting to new tasks.
On the other hand, when the classifier pays too much attention to plasticity, it may quickly forget previously-acquired abilities.
One possible solution is to efficiently reuse previously acquired knowledge when processing new tasks, meanwhile avoiding forgetting previously-acquired abilities. That is, on the one hand, consolidated knowledge is preserved to hold the long-term durability and prevent catastrophic forgetting when learning new tasks over time. On the other hand, in certain cases such as immersive long-term experiences, old knowledge can be
modified or substituted to refine new knowledge and avoid knowledge interference  (Parisi et al . ,  2019 ) .

In this paper, we propose a novel  i terative network  p runing with uncertainty  r egularization method for  l ifelong  s entiment classification (IPRLS). IPRLS deploys BERT

(Devlin
et al . ,  2019 )  as the base model for sentiment classification given its superior performance in the recent literature. To resolve the stability-plasticity dilemma  (Ahn
et al . ,  2019 ) , we leverage the principles of network pruning and weight regularization, sequentially integrating the important knowledge from multiple sequential tasks into a single BERT model while ensuring minimal decrease in accuracy.
Specifically, in each round of pruning, we use a weight-based pruning technique to free up a certain fraction of eligible weights from each layer of BERT after it has been trained for a task, and the released parameters can be modified for learning subsequent new tasks. Instead of keeping the old-tasks weights fixed when learning new tasks as in previous works  (Rusu et al . ,  2016 ; Mallya and
Lazebnik,  2018 ) , we incorporate an uncertainty regularization based on the Bayesian online learning framework into the iterative prunin