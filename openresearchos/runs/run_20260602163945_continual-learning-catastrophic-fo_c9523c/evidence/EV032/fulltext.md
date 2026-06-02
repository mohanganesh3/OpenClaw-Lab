[2109.00328] Memory-Free Generative Replay For Class-Incremental Learning

Memory-Free Generative Replay For Class-Incremental Learning

Xiaomeng Xin

†

†

\dagger

Yiran Zhong

‡

‡

\ddagger

Yunzhong Hou

‡

‡

\ddagger

Jinjun Wang

†

†

\dagger

Liang Zheng

‡

‡

\ddagger

†

†

\dagger

Xi’an Jiaotong University

‡

‡

\ddagger

Australian National University

xmengxin@163.com, zhongyiran@gmail.com,Yunzhong.Hou@anu.edu.au, jinjun@xjtu.edu.cn, Liang.Zheng@anu.edu.au

Abstract

Regularization-based methods are beneficial to alleviate the catastrophic forgetting problem in class-incremental learning. With the absence of old task images, they often assume that old knowledge is well preserved if the classifier produces similar output on new images.
In this paper, we find that their effectiveness largely depends on the nature of old classes: they work well on classes that are easily distinguishable between each other but may fail on more fine-grained ones, e.g., boy and girl. In spirit, such methods project new data onto the feature space spanned by the weight vectors in the fully connected layer, corresponding to old classes. The resulting projections would be similar on fine-grained old classes, and as a consequence the new classifier will gradually lose the discriminative ability on these classes.
To address this issue, we propose a memory-free generative replay strategy to preserve the fine-grained old classes characteristics by generating representative old images directly from the old classifier and combined with new data for new classifier training. To solve the homogenization problem of the generated samples, we also propose a diversity loss that maximizes Kullback–Leibler (KL) divergence between generated samples.
Our method is best complemented by prior regularization-based methods proved to be effective for easily distinguishable old classes. We validate the above design and insights on CUB-200-2011, Caltech-101, CIFAR-100 and Tiny ImageNet and show that our strategy outperforms existing memory-free methods with a clear margin. Code is available at  https://github.com/xmengxin/MFGR .

1  Introduction

Incremental learning addresses a critical problem called catastrophic forgetting: a network often quickly forgets previously acquired knowledge when learning new knowledge  [ 25 ,  2 ,  8 ] . Between the two main categories,  i.e .  , memory-based and memory-free methods, we choose to study the latter. Briefly, a network is asked to learn a sequence of new tasks without accessing data from previous (old) tasks. Many memory-free methods are regularization based. That is, to consolidate previous knowledge during the new knowledge learning process with extra regularization terms in the loss functions  [ 21 ,  23 ] .

Figure 1:

A cartoon illustration of how catastrophic forgetting happens on fine-grained old classes when using regularization-based methods such as LwF  [ 21 ] . We draw weight vectors of five old classes and the feature vector of a new task sample. Among the five old classes,  boy ,  girl  and  man  have similar semantics, and thus are closer in the feature space, while  cloud  and  chair  are further apart. When projecting the new image feature onto the weight vectors, fine-grained classes will give us nearly identical coordinates or soft labels. The new task images will require the new classifier to output similar soft labels on the three dimensions. As such, it offers little incentive for the classifier to preserve the knowledge that distinguishes between  boy ,  girl  and  man .

In the absence of old task data, these regularization-based methods often require that the old and new classifiers give similar responses ( e.g .  , softmax vector on old classes) to new images. This constraint benefits old knowledge preservation, as shown by the overall improvement in old data classification.
However, this strategy still suffers from catastrophic forgetting on  old classes that are closer in the feature/semantics space ,  i.e .  , the fine-grained classes.

Consider the cartoon example in Fig.

1  . What the above regularization method essentially does is to project a new task image onto the weight vectors corresponding to the five classes (and ask the 5-dim projection vectors or soft labels to be similar between the new and old classifiers). In this example, given a new task image, its projections on weight vectors of the three fine-grained classes are nearly the same in value.
In order words, in the view of new task images, weight vectors of  boy ,  girl  and  man  carry no discriminative information among them. As a consequence, the new classifier, when using these three coordinates or soft labels as network constraints, would view them as coming from the same class; there is no longer incentive for the new classifier to distinguish them, which in effect leads to catastrophic forgetting of these fine-grained old classes.

In this paper, we aim to mitigate the above catastrophic forgetting problem through a generative method named memory-free generative replay. While Fig.

1

indirectly  preserves old knowledge using new task images, we do so by  directly  having the new classifier learn from the generated old task images. As shown in Fig.

2  , our framework contains two stages: knowledge recording and knowledge inheritance. Specifically,
we first train the generator such that the generated images exhibit desired properties,  e.g .  , a high diversity, matching the BN statistics of the old classifier, and a low domain gap with the new task data. Then, we train the new task classifier using the new task images and generated old task images that are mixed and balanced in mini-batches. Also, in practice we find that the generated samples often suffer serious homogenization problem on sample level,  i.e .  , generating similar images for one class. To avoid this issue, we maximize Kullback–Leibler (KL) divergence between generated samples to force the samples to be diverse.

Our method tackles the Achilles’ heel of the regularization-based class-incremental learning. First, we conceptually and experimentally show that the fine-grained old classes are the most vulnerable when new tasks are learned. It is because regularization-based methods provide very little guidance for the new classifier to preserve such old knowledge. Second, although our method uses images that are much less genuine than the new target images, we still provide the characteristics of the fine-grained classes. This is proved by the significant classification accuracy improvement on these classes.
Moreover, since the generated old images cannot perfectly cover the real data distributions, our method is best to work together with previous regularization-base methods. This validates our system: a combination of strengths of regularization-based methods and the proposed generative replay.
On standard CIFAR100 and Tiny ImageNet benchmarks, we report state-of-the-art performance among memory-free methods.

Figure 2:

System workflow. Given an old task classifier, we first generate old task images using a Generator. The generated images, together with the new task data, are used to train the new task model.

2  Related work

Class-incremental learning attracts more attention recently, and there are significant number of papers published on this topic  [ 9 ] . Below we only cite a few most recent methods that closely related to our work.

Regularization-based Methods. 
Regularization-based methods are often memory-free,  i.e .  , does not explicitly  [ 27 ,  14 ,  24 ]  or implicitly  [ 28 ,  31 ,  19 ]  store previous data distributions. Elastic Weight Consolidation (EWC)  [ 15 ]  estimates important weights through calculating diagonal approximation of the Fisher information matrix. Memory Aware Synapses (MAS)  [ 1 ]  evaluates important parameters by looking at the sensitivity of a learned function, which can be computed in an online manner. Note that these metho