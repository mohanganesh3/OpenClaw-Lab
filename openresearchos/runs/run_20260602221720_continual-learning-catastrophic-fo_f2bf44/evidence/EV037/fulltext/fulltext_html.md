[2501.11002] pMixFed: Efficient Personalized Federated Learning through Adaptive Layer-Wise Mixup

(cvpr)

Package cvpr Warning: Incorrect paper size - CVPR uses paper size ‘letter’. Please load document class ‘article’ with ‘letterpaper’ option

pMixFed: Efficient Personalized Federated Learning through Adaptive Layer-Wise Mixup

Yasaman Saadati  1,2  , Mohammad Rostami 3 , and M. Hadi Amini  1,2

1  Knight Foundation School of Computing and Information Sciences, Florida International University

2  Sustainability, Optimization, and Learning for InterDependent networks laboratory (solid lab)

3  University of Southern California

ysaadati@fiu.edu  ,  rostamim@usc.edu  ,  moamini@fiu.edu

Abstract

Traditional Federated Learning (FL) methods encounter significant challenges when dealing with heterogeneous data and providing personalized solutions for non-IID scenarios. Personalized Federated Learning (PFL) approaches aim to address these issues by balancing generalization and personalization, often through parameter decoupling or partial models that freeze some neural network layers for personalization while aggregating other layers globally. However, existing methods still face challenges of global-local model discrepancy, client drift, and catastrophic forgetting, which degrade model accuracy. To overcome these limitations, we propose  pMixFed , a dynamic, layer-wise PFL approach that integrates  mixup  between shared global and personalized local models. Our method introduces an adaptive strategy for partitioning between personalized and shared layers, a gradual transition of personalization degree to enhance local client adaptation, improved generalization across clients, and a novel aggregation mechanism to mitigate catastrophic forgetting. Extensive experiments demonstrate that pMixFed outperforms state-of-the-art PFL methods, showing faster model training, increased robustness, and improved handling of data heterogeneity under different heterogeneous settings. Our code is available for reproducing our results:  https://github.com/YasMinSdt/pMixFed .

1  Introduction

Figure 1 :

Discrepancy between personalized and global shared layers in Partial PFL:  (1)  The global model,

G  t

G^{t}

, is constructed by aggregating asynchronous local updates from clients, denoted as

L  i  t

L^{t}_{i}

,

L  j  t

L^{t}_{j}

, and

L  k  t

L^{t}_{k}

.  (2),(3)  In communication round

t  t

, available clients

i  i

and

j  j

aggregate shared parameters to produce the updated global model

G

t  +  1

G^{t+1}

, while the personalized parameters, such as

L  k  t

L^{t}_{k}

, remain unchanged for unavailable clients.  (4)  This integration of distinct models,

G

t  +  1

G^{t+1}

and

L  k  t

L^{t}_{k}

, induces inconsistencies in the overall model updates.  (Bottom)  During the joint training of generalized and personalized models, the gradient updates from the generalized layers are impacted by the gradients from personalized layers, resulting in catastrophic forgetting, performance drop and slower convergence rates.

The goal in federated learning (FL)  [  22  ]  is to facilitate collaborative learning of several machine learning (ML) models in a decentralized scheme. FL requires addressing data privacy, catastrophic forgetting, and client drift problem

1

1  1 A phenomenon where the global model fails to serve as an accurate representation because local models gradually drift apart due to high data heterogeneity.

[  38  ,

15  ,

43  ,

31  ,

36  ] .
Existing FL methods cannot address all these challenges with non-IID (non-Independent and Identically Distributed) data.
For instance, although “FedAvg”

[  32  ]  demonstrates strong generalization performance, it fails to provide personalized solutions for a cohort of clients with non-IID datasets. Hence, the global model, or one “average client” in “FedAvg”, may not adequately represent all individual local models in non-IID settings due to client-drift  [  53  ] .
Personalized FL (PFL) methods handle data heterogeneity by considering both generalization and personalization during the training stage. Since there is a trade-off between generalization and personalization in heterogeneous environments, PFL methods leverage heterogeneity and diversity as
advantages rather than adversities  [  35  ,

47  ] . A group of PFL approaches train personalized local models on each device while collaborating toward a shared global model. Partial PFL, also known as parameter decoupling, involves using a partial model sharing, where only a subset of the model is shared while other parameters remain “frozen” to balance generalization and personalization until the subsequent round of local training.

While partial PFL methods are effective in mitigating catastrophic forgetting, strengthening privacy, and reducing computation and communication overhead  [  34  ,

45  ] , there are still some unaddressed challenges. First, the question of  when, where, and how to optimally partition the full model?  remains unresolved. Recent studies  [  34  ,

45  ]  have shown that there is no “one-size-fits-all” solution; the best or optimal partitioning strategy depends on factors such as task type (e.g., next word prediction or speech recognition) and local model architecture. An improper partitioning choice can lead to issues such as underfitting, overfitting, increased bias, and catastrophic forgetting. Some studies  [  29  ]  suggest that personalized layers should reside in the base layers, while others  [  9  ,

4  ]  argue that the base layers contain more generalized information and should be shared.
Further, the use of a fixed partitioning strategy across all communication rounds for heterogeneous clients can limit the efficacy of collaborative learning. For instance, if the performance of the client suddenly drops due to new incoming data, the partitioning strategy should be changed because the client requires more frozen layers.
Another issue is catastrophic forgetting of the previously shared global knowledge after only a few rounds of local training because the shared global model can be completely overwritten by local updates leading to generalization degradation  [  31  ,

41  ,

15  ,

54  ] .
Most importantly, partial models may experience slower convergence compared to full model personalization, as frozen local model updates can diverge in an opposite direction from the globally-shared model. Since the generalized and personalized models are trained on non-IID datasets, there might also be a domain shift, leading to model discrepancy as depicted in Figure

1  . These discrepancies arise from variations in local and global objective functions, differences in initialization, and asynchronous updates  [  56  ,

24  ] . As a result, merging the shared and the personalized layers can disrupt information flow within the network, impede the learning process, and lead to a slower convergence rate or accuracy drop in partial PFL models such as FedAlt and FedSim  [  34  ]

2

2  2 More details on this is discussed in section

4  .

.
Further, while partial PFL techniques contribute to an overall improved training accuracy, they can reduce the test accuracy on some devices, particularly in devices with limited samples, leading to variations in results in terms of the performance level  [  34  ] . Hence, there is a need for novel solutions to achieve the following goals in PFL:

•

Dynamic and Adaptive Partitioning: The balance between shared and personalized layers should be dynamically and adaptively adjusted for each client during every communication round, rather than relying on a static, fixed partitioning strategy for all participants.

•

Gradual Personalization Transition: The degree of personalization should transition gradually across layers, as opposed to an ”all-or-nothing” approach that employs strict partitioning or hard splits within the model discussed in Figure

1  . This ability allows nuanced