[2402.12656] HyperMoE: Paying Attention to Unselected Experts in Mixture of Experts via Dynamic Transfer.

HyperMoE: Paying Attention to Unselected Experts in Mixture of Experts via Dynamic Transfer.

Hao Zhao 1  Zihan Qiu 3  Huijia Wu 1  Zili Wang 4  Zhaofeng He 1  {NoHyper}  Jie Fu 2

1

1  footnotemark:

1

1 Beijing University of Posts and Telecommunications  2 Hong Kong University of Science and Technology

3 Tsinghua University

4 INF Technology

{haozhao,huijiawu,zhaofenghe}@bupt.edu.cn

jiefu@ust.hk

qzh11628@gmail.com

ziliwang.do.gmail.com

Corresponding authors.

Abstract

The Mixture of Experts (MoE) for language models has been proven effective in augmenting the capacity of models by dynamically routing each input token to a specific subset of experts for processing.
Despite the success, most existing methods face a challenge for balance between sparsity and the availability of expert knowledge: enhancing performance through increased use of expert knowledge often results in diminishing sparsity during expert selection.
To mitigate this contradiction, we propose HyperMoE, a novel MoE framework built upon Hypernetworks.
This framework integrates the computational processes of MoE with the concept of knowledge transferring in multi-task learning.
Specific modules generated based on the information of unselected experts serve as supplementary information, which allows the knowledge of experts not selected to be used while maintaining selection sparsity.
Our comprehensive empirical evaluations across multiple datasets and backbones establish that HyperMoE significantly outperforms existing MoE methods under identical conditions concerning the number of experts.
We provide some weights for HyperMoE at  https://github.com/Bumble666/HyperMoE_early_version

1  Introduction

The accelerated advancement of large language models has culminated in their widespread application across various domains, including healthcare, education, and social interactions  (Brown et al.,,  2020 ; Achiam et al.,,  2023 ; Touvron et al.,,  2023 ) . The remarkable capabilities of these models are attributed to the enhancements in their scale. Nevertheless, the scaling of dense models is often hampered by significant computational demands, posing a challenge to developing the Natural Language Processing (NLP) community.
In response, sparse activation models have emerged as a solution  (Artetxe et al.,,  2022 ; Du et al.,,  2022 ) , activating only a subset of parameters for different inputs, thus mitigating computational costs.
One of the most representative methods is the Mixture of Experts (MoE,  Shazeer et al., ( 2017 ) ), which routers different inputs to specific groups of experts, thereby enlarging the model’s capacity without increasing computational burdens.

Figure 1:  A trade-off in MoE: (a) A small number of selectable experts can maintain sparsity but limits the availability of expert knowledge. (b) Increasing the number of selectable experts can improve performance but decrease sparsity. (c) Transferring partial knowledge from the unselected experts

E

2  ,  3

subscript  𝐸

2  3

E_{2,3}

to the selected experts

E  1

subscript  𝐸  1

E_{1}

can improve the availability of expert knowledge while maintaining sparsity.

The key to effectively reducing computational costs lies in the sparsity of expert selection, with the number of experts selected for each token being kept at a lower level.
In practical applications or experiments, existing works  ( Roller et al., 2021a,  ; Fedus et al.,,  2022 ; Rajbhandari et al.,,  2022 ; Xue et al.,,  2023 )  usually select only one or two experts per input.
However, increasing the number of selected experts per token can enhance the availability of expert knowledge and improve the performance of downstream tasks  Yang et al., ( 2019 ); Shazeer et al., ( 2017 ); He et al., ( 2023 ) .
This scenario positions the MoE model in a predicament akin to a zero-sum game: a choice between increasing the number of available experts to improve performance or preserving a lower level of available experts to ensure sparsity, as depicted in Figure

1  .

To mitigate this contradiction, one solution would be to use the knowledge of other experts to assist the sparsely selected experts.
This is similar to multi-task learning, which transfers knowledge among related tasks.
Some works  (Karimi Mahabadi et al.,,  2021 ; Ivison and Peters,,  2022 ; Zhao et al.,,  2023 )  suggest using hypernetworks  (Ha et al.,,  2017 )  to generate task-specific knowledge to enhance positive transfer between tasks.
Inspired by this, we aim to increase the availability of expert knowledge by transferring the knowledge of unselected experts while sparsely selecting experts.

In this paper, we propose  HyperMoE , a novel MoE framework built upon hypernetworks, which captures the information from every expert by leveraging expert-shared hypernetwork while achieving positive expert transfer by generating conditioned modules individually.
We refer to the information as  cross-expert  information.
Specifically, a HyperMoE consists of HyperExperts, which are generated based on the information of unselected experts and serve as supplementary information for selected experts while maintaining sparsity.

We further improve upon this by introducing the concept of  cross-layer  Hypernetworks: A hypernetwork is shared among all transformer layers, which enables information flow among MoEs in different layers.
This brings additional efficiency in terms of parameters and computational costs: Despite the additional computation, our method only experienced a decrease  1

1  1 The degree of decline in speed is related to the scale of the Hypernetworks and the bottleneck size in the generated HyperExpert (similar to

r

𝑟

r

in LoRA  (Hu et al.,,  2022 ) ). For various tasks, these hyperparameters can be dynamically adjusted to control the delay.

of approximately 15% in training speed and 10% in inference speed compared to the standard MoE.

We evaluate HyperMoE on 20 representative NLP datasets across diverse tasks: sequence classification, extractive question answering, summarization, and text generation.
Extensive experimental results show that HyperMoE outperforms baselines, including Switch Transformer  (Fedus et al.,,  2022 )  with MoE architecture.
This demonstrates the effectiveness of our method in transferring knowledge to experts, which increases the utilization of expert knowledge while keeping the number of experts selected at a low level.

To summarise, our core contributions are:

•

We propose a novel HyperMoE architecture with HyperExpert for the MoE framework, which resolves the inherent tension between maintaining sparse expert selection and ensuring sufficient expert availability within MoE.

•

HyperMoE outperforms strong baselines based on Switch Transformer across a diverse set of NLP tasks, confirming our approach’s effectiveness.

•

We show the relevance between selection embeddings, which are based on the context of unselected experts, and selected experts, indicating that the selection embeddings effectively encode the information of knowledge that the currently selected experts need.

2  Background

2.1  Mixture of Expert

A Mixture of Experts (MoE) typically consists of two parts: the gate model

G

𝐺

G

and a set of expert models

E  1

,

E  2

,  ⋯  ,

E  N

subscript  𝐸  1

subscript  𝐸  2

⋯

subscript  𝐸  𝑁

E_{1},E_{2},\cdots,E_{N}

. The gate model is used to dynamically select and combine the outputs of the expert models based on the input

x

𝑥

x

. As a result, each input will be determined by the collective participation of multiple expert models to obtain the output

y

𝑦

y

:

y  =

∑

i  =  1

N

G  ​

(  x  )

i

​

E  i

​

(  x  )

.

𝑦

superscript

subscript

𝑖  1

𝑁

𝐺

subscript  𝑥  𝑖

subscript  𝐸  𝑖

𝑥

y=\sum_{i=1}^{N}G(x)_{i}E_{i}(x).

(1)

The gate model

G  ​

(  ⋅  )

𝐺  ⋅

G(\cdot)

is a