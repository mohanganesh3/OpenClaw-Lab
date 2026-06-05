[2409.16391] Patch-Based Contrastive Learning and Memory Consolidation for Online Unsupervised Continual Learning

Patch-Based Contrastive Learning and Memory Consolidation for Online Unsupervised Continual Learning

Cameron Taylor 1

cameron.taylor@gatech.edu

Vassilis Vassiliades 2

v.vassiliades@cyens.org.cy

Constantine Dovrolis  1,3

constantine@gatech.edu

&amp; 1 Georgia Institute of Technology

Atlanta, GA

2 CYENS Centre of Excellence

Nicosia, Cyprus

3 The Cyprus Institute

Nicosia, Cyprus

Abstract

We focus on a relatively unexplored learning paradigm known as  Online Unsupervised Continual Learning  (O-UCL), where an agent receives a non-stationary, unlabeled data stream and progressively learns to identify an increasing number of classes. This paradigm is designed to model real-world applications where encountering novelty is the norm, such as exploring a terrain with several unknown and time-varying entities. Unlike prior work in unsupervised, continual, or online learning, O-UCL combines all three areas into a single challenging and realistic learning paradigm. In this setting, agents are frequently evaluated and must aim to maintain the best possible representation at any point of the data stream, rather than at the end of pre-specified offline tasks. The proposed approach, called  P atch-based  C ontrastive learning and  M emory  C onsolidation (PCMC), builds a compositional understanding of data by identifying and clustering patch-level features. Embeddings for these patch-level features are extracted with an encoder trained via patch-based contrastive learning. PCMC incorporates new data into its distribution while avoiding catastrophic forgetting, and it consolidates memory examples during “sleep” periods. We evaluate PCMC’s performance on streams created from the ImageNet and Places365 datasets. Additionally, we explore various versions of the PCMC algorithm and compare its performance against several existing methods and simple baselines. The code is publicly available  on Github .

1  Introduction

Imagine an agent navigating an unfamiliar environment with objects that have never appeared in its pre-training data. As the agent explores that environment, it must recognize new classes of objects. Furthermore, it must generalize from the observed data to other instances of the same class. Additionally, it should not forget previously learned classes, even if it does not encounter such instances often. This setup demands an efficient learning model where the agent operates in a streaming fashion, retaining only a minuscule fraction of the observed data due to storage or privacy constraints. We refer to this learning paradigm as  Online Unsupervised Continual Learning  or “O-UCL” for short.

The O-UCL learning paradigm must address the following three challenges: 1) The data stream is non-stationary, in the sense that the number of classes of objects in the stream increases with time; 2) the observed data is unlabeled, and so the agent needs to identify novel classes in real-time and without supervision; 3) the agent cannot store the observed data for future replay, necessitating online stream learning. While a simple solution might be to utilize a frozen encoder pretrained on the largest possible dataset, this is insufficient for applications in which the environment is inherently unknown and/or it includes novelty (e.g., people, animals or objects that were never previously seen). Instead, we consider a dynamic approach in which the encoder and the corresponding data representations are periodically adapted during short “sleep” periods.

Figure 1:  A toy example for an O-UCL scenario. After an offline initialization (task-0), the agent is presented with a stream of data consisting of images from various classes (say three new classes in each task). The agent is tasked with learning to identify both new and previously known classes, without forgetting classes that no longer appear in the stream. The performance is evaluated frequently during the stream, to monitor how the agent learns over time. During sleep periods, the agent retrains its encoder and adapts its stored representations. Note that a small number of labeled examples are given to the classifier only during inference – no labeled examples are available for representation learning during the stream.

For example consider the following hypothetical applications: an AI-powered drone that explores a new construction site or physical system, or a face recognition system that has to classify people it has never seen before with distinct identifiers. In addition to the three O-UCL challenges described above, such applications have one more common requirement: there is no boundary between training and inference. Instead, a successful O-UCL learner should both learn and perform its task during the data stream, exhibiting gradually better performance as it observes more data.
In Figure

1  , we demonstrate a toy scenario to exemplify O-UCL.

A similar problem to O-UCL was first introduced in  (Smith et al.,  2021 ) , referred to as “Unsupervised Progressive Learning.” That work also proposed a method called “Self-Taught Associative Memories” or STAM. The method we propose here, referred to as  P atch-based  C ontrastive learning and  M emory  C onsolidation or “PCMC,” utilizes some techniques from STAM but it also introduces several new ideas. PCMC operates in a cycle of “wake” and “sleep” periods. The model identifies and clusters incoming stream data during each wake period, while it retrains the encoder and consolidates data representations during each sleep period. PCMC utilizes a novel patch-based contrastive learning encoder along with online clustering and novelty detection techniques. We evaluate PCMC’s performance against several baselines on challenging natural image datasets such as ImageNet  (Deng et al.,  2009 )  and Places365  (Zhou et al.,  2017 ) .

2  PCMC Method

The key component of PCMC is a growing set of cluster centroids that represent distinct data features. To learn useful centroids, we utilize an encoder

F  ϕ

​

(  ⋅  )

subscript  𝐹  italic-ϕ

⋅

F_{\phi}(\cdot)

, which is a deep neural network with parameters

ϕ

italic-ϕ

\phi

that generates similar embeddings for similar inputs and dissimilar embeddings for dissimilar inputs (contrastive learning). As the distribution of the stream changes over time, the encoder must adapt to changes in the data distribution (e.g., novel classes) but also to avoid concept drift and catastrophic forgetting. PCMC accomplishes this by adopting a wake-sleep cycle. During a wake period, the model performs novelty detection (i.e., creation of new clusters) and it also adapts the centroids of previously known clusters with new data. During a sleep period, the encoder is retrained and also the representations of the learned centroids are updated.

Critically, instead of clustering entire input examples (images in this paper), PCMC breaks each incoming example into smaller patches, and maps each patch to a cluster that represents a distinct feature of the data. The use of patches is important for two reasons. The first is that as the data distribution changes, we want to leverage potential forward transfer for shared features across classes (e.g., fire truck tires and ambulance tires) at the level of patches. Secondly, clustering similar inputs is more challenging at the level of the entire input compared to the patch level. This is because the variance of a patch-level feature (e.g., a dog’s nose) is typically much smaller than the variance of the entire input (e.g., an image of a dog). By learning features at the patch level, the model can build a compositional understanding of each class, based on the class-informative features it includes.

Figure 2:  This figure summarizes the wake period of PCMC. The input

x  i

subscript  𝑥  𝑖

x_{i}

is broken up into patches, and the encoder

F

ϕ  s

subscript  𝐹

sub