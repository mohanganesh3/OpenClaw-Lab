[2406.16554] LLaMA-MoE: Building Mixture-of-Experts from LLaMA with Continual Pre-training

LLaMA-MoE: Building Mixture-of-Experts

from LLaMA with Continual Pre-training

Tong Zhu  1

,  Xiaoye Qu  2

,  Daize Dong  2

,  Jiacheng Ruan  3

,  Jingqi Tong  4

,

Conghui He  2

,  Yu Cheng

5  ​

✉

5

✉

{}^{5\text{ {\char 0\relax}}}

1

Soochow University

2

Shanghai AI Laboratory

3

Shanghai Jiao Tong University

4

Fudan University

5

The Chinese University of Hong Kong

tzhu7@stu.suda.edu.cn,

{quxiaoye,dongdaize.d,heconghui}@pjlab.org.cn,

jackchenruan@sjtu.edu.cn,

jqtong23@m.fudan.edu.cn,

chengyu@cse.cuhk.edu.hk

Work was done during an internship at Shanghai AI Laboratory.

Abstract

Mixture-of-Experts (MoE) has gained increasing popularity as a promising framework for scaling up large language models (LLMs).
However, training MoE from scratch in a large-scale setting still suffers from data-hungry and instability problems.
Motivated by this limit, we investigate building MoE models from existing dense large language models.
Specifically, based on the well-known LLaMA-2 7B model, we obtain an MoE model by: (1)  Expert Construction , which partitions the parameters of original Feed-Forward Networks (FFNs) into multiple experts; (2)  Continual Pre-training , which further trains the transformed MoE model and additional gate networks.
In this paper, we comprehensively explore different methods for expert construction and various data sampling strategies for continual pre-training.
After these stages, our LLaMA-MoE models could maintain language abilities and route the input tokens to specific experts with part of the parameters activated.
Empirically, by training 200B tokens, LLaMA-MoE-3.5B models significantly outperform dense models that contain similar activation parameters.
The source codes and models are available at  https://github.com/pjlab-sys4nlp/llama-moe  .

1  Introduction

Large language models (LLMs)  (OpenAI,  2023 ; Touvron et al.,  2023a ; Cai et al.,  2024 )  have presented remarkable understanding and reasoning capabilities on a wide range of tasks  (Lu et al.,  2024 ; Su et al.,  2024a ,  b ; Tao et al.,  2024 ) .
Nowadays, scaling model size has become the de facto approach to further boost the overall performances  (Wei et al.,  2022 ; Hoffmann et al.,  2022 ) .
However, the immense model size is unsustainable due to the computational costs  (Fedus et al.,  2022a ) .
Motivated by the trade-off between scaling the model size and reducing the inference cost, sparse Mixture-of-Experts (MoE) is further proposed to activate a part of the model parameters  (Shazeer et al.,  2017 ; Lepikhin et al.,  2020 ) .
Inspired by this, we focus on sparsely activated MoE models that decouple model size from computation costs.

However, training MoEs from scratch  (Fedus et al.,  2022b ; Zoph et al.,  2022 ; Xue et al.,  2024 ; Dai et al.,  2024 )  leads to a significant overall budget.
In this work, we reduce the training costs by investigating building MoE models from existing dense LLMs.
Moreover, starting from the dense model provides flexible structure design choices for MoE.
In other words, we can place the MoE block in any transformer layers.
In this paper, we dedicate to building a full MoE model from LLaMA  (Touvron et al.,  2023b ) , where each layer contains an MoE block.

To build strong LLaMA-MoE models, we identify two important challenges.
First,  how to effectively construct experts  from the Feed-Forward Networks (FFNs) in the existing LLMs.
There are works exploring splitting FFN parameters to construct experts on T5  (Zhang et al.,  2021 )  or BERT  (Zuo et al.,  2022 )  models.
Conversely,  Komatsuzaki et al. ( 2022 )  directly copy the FFNs to form experts.
However, there is no existing work exploring it for decoder-only models.
Notably, the Swish-based FFN  (Ramachandran et al.,  2017 ; Shazeer,  2020 )  does not bring natural sparsity as ReLU  (Nair and Hinton,  2010 )  in T5 and BERT.
Second,  overcoming the performance decrease entailed by
changing the network structure  from dense to sparse remains challenging.
Due to the reduction in the amount of activated parameters and the newly introduced gate network for expert routing, we observe a significant performance decline between the LLaMA-MoE models and the original dense LLaMA models.

To solve the above issues, we comprehensively explore four different methods for expert construction.
Among them, the non-overlapping randomly splitting method achieves the best performance.
Subsequently, we continue training the transformed MoE models and additional gate networks.
In this stage, we also carefully study both dynamic and static data sampling strategies for obtaining the fastest convergence and performance improvement.
Finally, with a static domain weight proportion corresponding to the activated parameters, the LLaMA-MoE models can quickly converge to a decent level with 200B tokens.

In summary, our contributions are as follows:

•

We propose a framework to develop mixture-of-experts from existing decoder-style LLMs by splitting FFNs and continual pre-training, which has never been explored before.

•

We comprehensively explore different splitting methods for expert construction. Meanwhile, we comprehensively investigate both dynamic and static data sampling strategy for continual pre-training.

•

Our extensive experiments on a variety of tasks validate the effectiveness of our proposed LLaMA-MoE series models. Notably, all our model construction processes and training data are transparent.

2  Related Work

Mixture-of-Experts (MoE). 
Traditionally, dense models feed all parameters to each input token. In this way, the growing model capacity brings increased computational cost. To alleviate this issue, sparse models attempt to activate a subset of parameters for each input and these activated parameters are referred as experts.
In  Shazeer et al. ( 2017 ) , MoE was first proven effective in modern deep learning.
This work added an MoE layer which was stacked between LSTM, resulting in state-of-the-art results in language modeling and machine translation benchmarks.
Subsequently, the MoE layer is introduced to the transformer architecture as a substitute for the FFN layers.
GShard  (Lepikhin et al.,  2020 )  applied the MoE to the Transformer and significantly improved machine translation across 100 languages.
Switch Transformers  (Fedus et al.,  2022b )  further scales the language model’s size to the trillion-level parameter with a simple and effective MoE layer design.
Naively trained MoE models is prone to load imbalance, e.g., only a few experts are frequently used while the others are scarcely activated. For optimizing the training, BASE layer  (Lewis et al.,  2021 ) , HASH layer  (Roller et al.,  2021 ) , and Expert Choice  (Zhou et al.,  2022 )  study how to build MoE models to fully utilize the model capacity.
Recently, for model architecture,  Xue et al. ( 2024 )  explore training a decoder-only MoE with a modified UL2 training objective.
Mixtral is another decoder-style MoE model that selects two out of eight experts with token-choice routing  (AI,  2023b ) .

Expert Construction.  There are two lines of works constructing MoE from dense checkpoints. The first category splits the parameters of the FFNs and ensures that the total model parameters remain unchanged.
MoEBERT  (Zuo et al.,  2022 )  propose an
importance-based method to adapt the FFNs into experts. Considering that some neurons in the FFNs contribute more to the model performance, they share the most important neurons (i.e., the ones with the highest scores) among the experts, and the other neurons are distributed evenly.
MoEfication  (Zhang et al.,  2021 )  study the activation patterns of FFNs in Transformer models and find a sparse activation
phenomenon. Then, they discover the functional partitions (experts) in FFNs and build
routers for selecting experts. It is worth noting that the