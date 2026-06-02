[2504.00661] \model: Boosting Mixture of LoRA Experts Fine-Tuning with a Hybrid Routing Mechanism

\model : Boosting Mixture of LoRA Experts Fine-Tuning with a Hybrid Routing Mechanism

Dengchun Li 1 , Naizheng Wang 1 , Zihao Zhang 1 , Haoyang Yin 1 , Lei Duan 1 , Meng Xiao 2

Mingjie Tang  1

1

School of Computer Science, Sichuan University, Chengdu, China.

2 Computer Network Information Center, Chinese Academy of Sciences, Beijing, China.

mikecovlee@163.com, pherenice1125@gmail.com, zzzzh@stu.scu.edu.cn ,

filtee0812@gmail.com, leiduan@scu.edu.cn, shaow@cnic.cn ,

tangrock@gmail.com

Abstract

Instruction-based fine-tuning of large language models (LLMs) has achieved remarkable success in various natural language processing (NLP) tasks. Parameter-efficient fine-tuning (PEFT) methods, such as Mixture of LoRA Experts (MoLE), combine the efficiency of Low-Rank Adaptation (LoRA) with the versatility of Mixture of Experts (MoE) models, demonstrating significant potential for handling multiple downstream tasks. However, the existing routing mechanisms for MoLE often involve a trade-off between computational efficiency and predictive accuracy, and they fail to fully address the diverse expert selection demands across different transformer layers.
In this work, we propose  \model , a hybrid routing strategy that dynamically adjusts expert selection based on the Tsallis entropy of the router’s probability distribution. This approach mitigates router uncertainty, enhances stability, and promotes more equitable expert participation, leading to faster convergence and improved model performance. Additionally, we introduce an auxiliary loss based on Tsallis entropy to further guide the model toward convergence with reduced uncertainty, thereby improving training stability and performance.
Our extensive experiments on commonsense reasoning benchmarks demonstrate that  \model  achieves substantial performance improvements, outperforming LoRA by 9.6% and surpassing the state-of-the-art MoLE method, MoLA, by 2.3%. We also conduct a comprehensive ablation study to evaluate the contributions of  \model ’s key components.

1  Introduction

Instruction-based fine-tuning of large language models  (Brown et al.,  2020 ; Chowdhery et al.,  2022 ; Touvron et al.,  2023a ;  b )  for various downstream tasks has achieved remarkable proficiency in natural language processing tasks  (Chung et al.,  2022 ; Iyer et al.,  2022 ; Zheng et al.,  2024 ) .
To significantly reduce the computational and memory resources required for full parameter fine-tuning, parameter-efficient fine-tuning methods have emerged  (Houlsby et al.,  2019 ; Li &amp; Liang,  2021 ; Lester et al.,  2021 ; Ben-Zaken et al.,  2021 ; Liu et al.,  2022 ) .
Among these, LoRA  (Hu et al.,  2021 )  has gained popularity due to its ability to reduce substantial computational costs.
To maintain both cross-task generalization and computational efficiency, a promising solution  (Yang et al.,  2024 ; Luo et al.,  2024 ; Feng et al.,  2024 )  is to design an architecture that combines the resource-efficient features of LoRA with the versatility of Mixture of Experts (MoE) models  (Wu et al.,  2024a ; Dou et al.,  2024 ; Gou et al.,  2023 ; Liu et al.,  2023 ; Feng et al.,  2024 ) .
These methods are often referred to as Mixture of LoRA Experts (MoLE). The routing mechanisms of these MoLE methods are mostly derived from standard MoE models, where a fixed number of expert networks are activated.
However, recent studies indicate that the requirements for experts vary across different transformer layers  (Gao et al.,  2024 ; Zeng et al.,  2024 ) , suggesting that the routing mechanism requires further modifications to account for these factors.

Current routing mechanisms  (Cai et al.,  2024 )  can be broadly classified into two categories:
 1) Soft Routing : These methods activate all expert networks for each input token, which typically leads to improvement in prediction accuracy  (Ma et al.,  2018 ; Nie et al.,  2021 ; Wu et al.,  2024c ; Dou et al.,  2024 ; Pan et al.,  2024 ) .
However, this comes at the cost of significant computational overhead, as all experts are involved in the computation  (Shazeer et al.,  2017 ) .
 2) Sparse Routing : These approaches enhance model efficiency by activating only a subset of experts  (Shazeer et al.,  2017 ) .
Some techniques route each token to a single expert  (Fedus et al.,  2022 ) , while others activate multiple experts, such as using Top-K  (Zhou et al.,  2022 )  or Top-P  (Huang et al.,  2024 ) , or employing uncertainty-based routing  (Wu et al.,  2024b ) .
Although sparse routing improves parameter efficiency, it often results in an imbalanced workload among experts, making it necessary to include an auxiliary loss functions to ensure the balance.
Though both of these routing techniques aim to select the optimal set of experts for each input token, neither provides a fully comprehensive solution that accounts for the diverse and complex factors affecting model performance, which raises a key question:  How can we design a  hybrid routing approach  that considers these factors holistically to provide a more complete solution for MoE and MoLE models?

Figure 1:  Visualized motivation of  \model . We propose a  hybrid routing  mechanism for  \model  to address and solve these critical challenges.

To answer this question, several critical challenges emerge:
 1) Inconsistent Expert Selection  occurs when flat probability distributions lead to similar inputs activating different experts, resulting in unstable expert training.
 2) Varied Expert Requirements  across the model, as noted by  Gao et al. ( 2024 ); Zeng et al. ( 2024 ) , leads to uneven expert loads when the number of activated experts is fixed across all layers.
Finally,  3) Fluctuating Gradient Updates  resulting from uncertain routing decisions, cause fluctuations in gradient flows, which adversely affect convergence speed and stability during model training.
These challenges are illustrated in Figure

1  .

To address these challenges, we propose  \model  ( Dyn amic Routing for  M ixture  o f  L oRA  E xperts), a hybrid routing approach designed to reduce router uncertainty in Mixture of LoRA Experts adapters for parameter-efficient fine-tuning of large language models.
Our approach leverages the mathematical properties of Tsallis entropy  (Tsallis,  1988 ) , a generalized entropy measure, to develop adaptive routing strategies that effectively minimize router uncertainty.
Furthermore, we introduce an auxiliary loss based on Tsallis entropy to guide the model towards convergence with reduced uncertainty, thus improving training stability and performance.
By preventing over-reliance on certain experts and promoting more equitable engagement across all experts, this method fosters a diverse and robust set of expert contributions. This approach not only optimizes computational resource allocation but also enhances overall model performance by improving decision consistency and stability during training.

Summary of Contributions :

1.

We identify the uncertainty problem in MoE routers and theoretically derive their optimal probability distribution, which we term the  Peaked Distribution . Through formal reasoning, we prove that Tsallis entropy provides a more effective quantification of routing uncertainty compared to traditional measures.

2.

We propose a hybrid strategy, called  \model , which enables the routing mechanism to dynamically adjust based on the entropy of the routing distribution for each token, making expert selection more flexible and efficient. Additionally, we introduce an auxiliary loss for  \model  based on T sallis entropy, to guide the model toward convergence with reduced uncertainty, improving training stability and performance.

3.

We validate the effectiveness of  \model  using widely recognized benchmarks, as used in prior works. The results demonstrate that  \model  achieves rem