[2401.06066] DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models

\reportnumber

001

\correspondingauthor Contribution during internship at DeepSeek-AI.

DeepSeekMoE: Towards Ultimate Expert Specialization in

Mixture-of-Experts Language Models

Damai Dai  ∗1,2

Chengqi Deng  1

Chenggang Zhao  ∗1,3

R.X. Xu  1

Huazuo Gao  1

Deli Chen  1

Jiashi Li  1

Wangding Zeng  1

Xingkai Yu  ∗1,4

Y. Wu  1

Zhenda Xie  1

Y.K. Li  1

Panpan Huang  1

Fuli Luo  1

Chong Ruan  1

Zhifang Sui  2

Wenfeng Liang  1

1

DeepSeek-AI

2 National Key Laboratory for Multimedia Information Processing

Peking University

3

Institute for Interdisciplinary Information Sciences

Tsinghua University

4

National Key Laboratory for Novel Software Technology

Nanjing University

{daidamai, szf}@pku.edu.cn

{wenfeng.liang}@deepseek.com

https://github.com/deepseek-ai/DeepSeek-MoE

Abstract

In the era of large language models, Mixture-of-Experts (MoE) is a promising architecture for managing computational costs when scaling up model parameters.
However, conventional MoE architectures like GShard, which activate the top-

K

𝐾

K

out of

N

𝑁

N

experts, face challenges in ensuring expert specialization, i.e. each expert acquires non-overlapping and focused knowledge.
In response, we propose the  DeepSeekMoE  architecture towards ultimate expert specialization.
It involves two principal strategies:
(1) finely segmenting the experts into

m  ​  N

𝑚  𝑁

mN

ones and activating

m  ​  K

𝑚  𝐾

mK

from them, allowing for a more flexible combination of activated experts;
(2) isolating

K  s

subscript  𝐾  𝑠

K_{s}

experts as shared ones, aiming at capturing common knowledge and mitigating redundancy in routed experts.
Starting from a modest scale with 2B parameters, we demonstrate that DeepSeekMoE 2B achieves comparable performance with GShard 2.9B, which has 1.5

×

\times

expert parameters and computation.
In addition, DeepSeekMoE 2B nearly approaches the performance of its dense counterpart with the same number of total parameters, which set the upper bound of MoE models.
Subsequently, we scale up DeepSeekMoE to 16B parameters and show that it achieves comparable performance with LLaMA2 7B, with only about 40% of computations.
Further, our preliminary efforts to scale up DeepSeekMoE to 145B parameters consistently validate its substantial advantages over the GShard architecture, and show its performance comparable with DeepSeek 67B, using only 28.5% (maybe even 18.2%) of computations.

Figure 1:

Comparison between DeepSeekMoE 16B and open source models on the Open LLM Leaderboard.
The red dashed line is linearly fitted from data points of all models except DeepSeekMoE 16B.
DeepSeekMoE 16B consistently outperforms models with a similar number of activated parameters by a large margin, and achieves comparable performance with LLaMA2 7B, which has approximately 2.5 times the activated parameters.

1  Introduction

Recent research and practices have empirically demonstrated that, with sufficient training data available, scaling language models with increased parameters and computational budgets can yield remarkably stronger models  (Brown et al.,  2020 ; OpenAI,  2023 ; Touvron et al.,  2023a ; Hoffmann et al.,  2022 ) .
It is imperative to acknowledge, however, that the endeavor to scale models to an extremely large scale is also associated with exceedingly high computational costs.
Considering the substantial costs, the Mixture-of-Experts (MoE) architecture  (Jacobs et al.,  1991 ; Jordan and Jacobs,  1994 ; Shazeer et al.,  2017 )  has emerged as a popular solution.
It can enable parameter scaling, while concurrently keeping computational costs at a modest level.
Recent applications of MoE architectures in Transformers  (Vaswani et al.,  2017 )  have yielded successful attempts at scaling language models to a substantial size  (Fedus et al.,  2021 ; Lepikhin et al.,  2021 ; Du et al.,  2022 ; Zoph,  2022 ) , accompanied with remarkable performance.
These achievements underscore the considerable potential and promise of MoE language models.

Despite the promising potential of MoE architectures, existing MoE architectures potentially suffer from issues of knowledge hybridity and knowledge redundancy, which limit the expert specialization, i.e., each expert acquires non-overlapping and focused knowledge.
Conventional MoE architectures substitute the Feed-Forward Networks (FFNs) in a Transformer with MoE layers.
Each MoE layer consists of multiple experts, with each structurally identical to a standard FFN, and each token is assigned to one  (Fedus et al.,  2021 )  or two  (Lepikhin et al.,  2021 )  experts.
This architecture manifests two potential issues:
(1)
 Knowledge Hybridity : existing MoE practices often employ a limited number of experts (e.g., 8 or 16), and thus tokens assigned to a specific expert will be likely to cover diverse knowledge.
Consequently, the designated expert will intend to assemble vastly different types of knowledge in its parameters, which are hard to utilize simultaneously.
(2)
 Knowledge Redundancy : tokens assigned to different experts may require common knowledge.
As a result, multiple experts may converge in acquiring shared knowledge in their respective parameters, thereby leading to redundancy in expert parameters.
These issues collectively hinder the expert specialization in existing MoE practices, preventing them from reaching the theoretical upper-bound performance of MoE models.

In response to the aforementioned issues, we introduce  DeepSeekMoE , an innovative MoE architecture specifically designed towards ultimate expert specialization.
Our architecture involves two principal strategies:
(1)  Fine-Grained Expert Segmentation: 
while maintaining the number of parameters constant, we segment the experts into a finer grain by splitting the FFN intermediate hidden dimension.
Correspondingly, keeping a constant computational cost, we also activate more fine-grained experts to enable a more flexible and adaptable combination of activated experts.
Fine-grained expert segmentation allows diverse knowledge to be decomposed more finely and be learned more precisely into different experts, where each expert will retain a higher level of specialization.
In addition, the increased flexibility in combining activated experts also contributes to a more accurate and targeted knowledge acquisition.
(2)  Shared Expert Isolation: 
we isolate certain experts to serve as shared experts that are always activated, aiming at capturing and consolidating common knowledge across varying contexts.
Through compressing common knowledge into these shared experts, redundancy among other routed experts will be mitigated.
This can enhance the parameter efficiency and ensure that each routed expert retains specialized by focusing on distinctive aspects.
These architectural innovations in DeepSeekMoE offer opportunities to train a parameter-efficient MoE language model where each expert is highly specialized.

Starting from a modest scale with 2B parameters, we validate the advantages of the DeepSeekMoE architecture.
We conduct evaluations on 12 zero-shot or few-shot benchmarks spanning diverse tasks.
Empirical results indicate that DeepSeekMoE 2B surpasses GShard 2B  (Lepikhin et al.,  2021 )  by a substantial margin, and even matches GShard 2.9B, a larger MoE model with 1.5

×

\times

expert parameters and computation.
Remarkably, we find that DeepSeekMoE 2B nearly approaches the performance of its dense counterpart with an equivalent number of parameters, which sets the strict upper bound of MoE language models.
In pursuit of deeper insights, we conduct elaborate ablation studies and analysis on the expert specialization for DeepSeekMoE.
These studies validate the effectiveness of fine-grained expert segmentation and shared expert isolation, and provide empirical evidence supporting the assertion that DeepSeekMoE ca