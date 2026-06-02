[2506.05977] Mitigating Catastrophic Forgetting with Adaptive Transformer Block Expansion in Federated Fine-Tuning

Mitigating Catastrophic Forgetting with Adaptive Transformer Block Expansion in Federated Fine-Tuning

Yujia Huo,
Jianchun Liu,

Hongli Xu,

Zhenguo Ma,

Shilong Wang, 
Liusheng Huang,

Y. Huo
are with the School of Computer Science and Technology, University of Science and Technology of China, Hefei, Anhui, China, 230027, and also with Suzhou Institute for Advanced Research, University of Science and Technology of China, Suzhou, Jiangsu, China, 215123. E-mails: yujia

_  \_

huo@mail.ustc.edu.cn; jcliu17@ustc.edu.cn; xuhongli@ustc.edu.cn; shilongwang@mail.ustc.edu.cn; lshuang@ustc.edu.cn.
Z. Ma is with the School of Computer Science and Technology, China University of Mining and Technology, Xuzhou, Jiangsu, China, 221116, and also with Mine Digitization Engineering Research Center of the Ministry of Education. E-mail: cs

_  \_

zgma@cumt.edu.cn.

Abstract

Federated fine-tuning (FedFT) of large language models (LLMs) has emerged as a promising solution for adapting models to distributed data environments while ensuring data privacy.
Existing FedFT methods predominantly utilize parameter-efficient fine-tuning (PEFT) techniques to reduce communication and computation overhead.
However, they often fail to adequately address the catastrophic forgetting, a critical challenge arising from continual adaptation in distributed environments.
The traditional centralized fine-tuning methods, which are not designed for the heterogeneous and privacy-constrained nature of federated environments, struggle to mitigate this issue effectively.
Moreover, the challenge is further exacerbated by significant variation in data distributions and device capabilities across clients, which leads to intensified forgetting and degraded model generalization.
To tackle these issues, we propose FedBE, a novel FedFT framework that integrates an adaptive transformer block expansion mechanism with a dynamic trainable-block allocation strategy.
Specifically, FedBE expands trainable blocks within the model architecture, structurally separating newly learned task-specific knowledge from the original pre-trained representations.
Additionally, FedBE dynamically assigns these trainable blocks to clients based on their data distributions and computational capabilities. This enables the framework to better accommodate heterogeneous federated environments and enhances the generalization ability of the model.
Extensive experiments show that compared with existing federated fine-tuning methods, FedBE achieves 12–74% higher accuracy retention on general tasks after fine-tuning and a model convergence acceleration ratio of 1.9-3.1

×  \times

without degrading the accuracy of downstream tasks.

Index Terms:

Federated Learning, Fine-Tuning, LLM, Catastrophic Forgetting, Transformer Block Expansion

1

Introduction

Recently, transformer-based large language models (LLMs) have achieved remarkable success in NLP through the pre-training and fine-tuning paradigm  [ 1 ,  2 ] , but this approach typically requires centralized access to large-scale labeled data, raising privacy concerns under regulations like GDPR  [ 3 ] . To address this, federated fine-tuning (FedFT) has emerged as a privacy-preserving alternative, enabling decentralized clients to adapt models without sharing raw data. FedNLP  [ 4 ]  exemplifies this approach by collaboratively optimizing full model parameters. However, full-parameter tuning incurs heavy communication and computation overhead, making it unsuitable for resource-constrained clients.
To alleviate these burdens, researchers have introduced parameter-efficient fine-tuning (PEFT) techniques  [ 5 ]  into the FedFT setting. For instance, FedAdapter  [ 6 ]  and FedPETuning  [ 7 ]  incorporate lightweight modules such as adapters, LoRA, and BitFit to reduce the volume of updated parameters during training. By freezing the pre-trained LLM backbone and only updating inserted parameters, PEFT methods significantly reduce computation overhead on the client side. Moreover, as only these modules are transmitted between clients and the server, the communication cost is also greatly reduced compared to full-model updates.

Although recent advances in FedFT have yielded impressive improvements in task performance and training efficiency,  catastrophic forgetting

[ 8 ]  remains an underexplored problem.
This phenomenon manifests as the model is adapted to new tasks and forgets previously acquired knowledge, resulting in significant performance degradation on previous tasks.
While catastrophic forgetting has been extensively studied in centralized continual learning, with techniques such as memory replay  [ 9 ]  and regularization-based methods (e.g., R-EWC  [ 10 ] ), these solutions do not directly translate to the federated settings.
Memory replay mitigates forgetting by storing and revisiting samples from prior tasks, but such data retention is incompatible with privacy constraints in federated environments.
Besides, methods like R-EWC utilize Fisher information matrices computed from gradients of previous tasks to penalize deviations in critical parameters. However, in federated fine-tuning, each client typically has access only to a fragmented subset of data, resulting in inaccurate Fisher matrix estimation and weakened forgetting mitigation  [ 11 ] .
In the context of federated learning, several studies have explored mitigating catastrophic forgetting by introducing regularization mechanisms or decoupled training strategies. These approaches typically aim to preserve global knowledge while allowing local adaptation.
For instance, FedCurv  [ 12 ]  adds a Fisher-based regularization term to constrain local model updates and reduce forgetting. However, when applied to the fine-tuning of large language models, such constraints may overly limit parameter updates, leading to underfitting and poor downstream generalization.

Furthermore, we observe that the catastrophic forgetting issue is exacerbated in federated fine-tuning, primarily due to the inherent challenge of  data heterogeneity ,
which arise from differences in user behaviors, environmental contexts, or application-specific requirements  [ 13 ,  14 ] .
As a result, locally fine-tuned model parameters tend to diverge significantly from one another, impairing their ability to capture and generalize global knowledge  [ 15 ] .
Besides, such parameter divergence hinders the effectiveness of model aggregation, making it difficult to preserve the original model’s generalization capacity and exacerbating catastrophic forgetting.
As demonstrated in Section

2.2  , under identical training hyperparameters, the accuracy of general classification tasks decreases by 11–14% and 17-27% in centralized and federated settings respectively, after fine-tuning on specific downstream tasks.
Moreover, as the heterogeneity of data distribution increases, the model tends to forget more of its general knowledge, which is reflected in a more substantial drop in accuracy on general tasks.

To address the above issues, we propose an efficient FedFT framework named FedBE, which incorporates an adaptive transformer block expansion mechanism.
Specifically, FedBE expands the base model by appending additional transformer blocks as trainable modules while keeping the original parameters frozen.
This strategy structurally decouples the model’s general knowledge from downstream task adaptation, thereby isolating new knowledge from existing representations to mitigate catastrophic forgetting.
Moreover, FedBE adapts to data and resource heterogeneity across clients by employing a dynamic trainable block assignment strategy.
This strategy assigns different subsets of the expanded transformer blocks to each client based on its specific data characteristics and computational resource capacity, enabling flexible adaptation without compromising overa