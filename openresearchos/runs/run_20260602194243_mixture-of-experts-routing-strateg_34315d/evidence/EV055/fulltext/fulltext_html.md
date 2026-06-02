[2504.09265] Mixture of Group Experts for Learning Invariant Representations

Mixture of Group Experts for Learning Invariant Representations

Lei Kang

Beijing Normal University  Beijing  China

leikang@mail.bnu.edu.cn

,

Jia Li

Beijing Normal University  Beijing  China

jiali@bnu.edu.cn

,

Mi Tian

TAL Education Group  Beijing  China

tianmi@tal.com

and

Hua Huang

Beijing Normal University  Beijing  China

huahuang@bnu.edu.cn

(2025)

Abstract.

Sparsely activated Mixture-of-Experts (MoE) models effectively increase the number of parameters while maintaining consistent computational costs per token. However, vanilla MoE models often suffer from limited diversity and specialization among experts, constraining their performance and scalability, especially as the number of experts increases. In this paper, we present a novel perspective on vanilla MoE with top-

k  k

routing inspired by sparse representation. This allows us to bridge established theoretical insights from sparse representation into MoE models. Building on this foundation, we propose a group sparse regularization approach for the input of top-

k  k

routing, termed Mixture of Group Experts (MoGE). MoGE indirectly regularizes experts by imposing structural constraints on the routing inputs, while preserving the original MoE architecture. Furthermore, we organize the routing input into a 2D topographic map, spatially grouping neighboring elements. This structure enables MoGE to capture representations invariant to minor transformations, thereby significantly enhancing expert diversity and specialization. Comprehensive evaluations across various Transformer models for image classification and language modeling tasks demonstrate that MoGE substantially outperforms its MoE counterpart, with minimal additional memory and computation overhead. Our approach provides a simple yet effective solution to scale the number of experts and reduce redundancy among them. The source code is included in the supplementary material and will be publicly released.

Mixture-of-Experts, Group Sparsity, Invariant Representations

†

†  copyright:  acmlicensed

†

†  journalyear:  2025

†

†  doi:  XXXXXXX.XXXXXXX

†

†  conference:  the 33rd ACM International Conference on Multimedia; October 27–31, 2025; Dublin, Ireland

†

†  isbn:  978-1-4503-XXXX-X/2018/06

†

†  submissionid:  4041

1.  Introduction

Transformer models  (Vaswani et al.,  2017 )  have consistently demonstrated performance improvements as their parameter counts increase  (Kaplan et al.,  2020 ) . However, the growth in model size comes with significant computational costs, making further scaling increasingly challenging. Sparsely activated Mixture-of-Experts (MoE)  (Shazeer et al.,  2017 )  offers a promising solution by utilizing a sparse architecture that activates only a subset of parameters during both training and inference. As shown in Fig.

1

(a), an MoE layer comprises multiple experts, activating only a few experts for each input token. A gating network is trained to route each token to the most suitable experts. This conditional computation enables MoE models to significantly increase their capacity without a rise in computation overhead. By integrating MoE layers into Transformer models, researchers have successfully scaled large foundation models to impressive sizes, achieving outstanding performance in tasks such as image classification (Riquelme et al.,  2021 ; Hwang et al.,  2023 )  and language modeling  (Teo and Nguyen,  2024 ; Csordás et al.,  2024 ) .

(a)

Vanilla top-2 routing MoE layer

(b)

Our proposed top-2 routing MoGE layer

Figure 1 .

A high-level comparison between the vanilla MoE layer and the proposed MoGE layer. (a) illustrates the vanilla MoE layer with the top-2 routing, while (b) demonstrates the application of group sparse regularization in MoGE. It is important to note that the using of group sparse regularization does not alter the MoE architecture. In the vanilla MoE, each element in the input of top-2 routing is independent and considered part of a single group. In contrast, MoGE introduces a grouping mechanism that partitions the input of top-2 routing into two groups: the first two elements in one group and the remaining three elements in the other. This grouping effect naturally extends to the subsequent experts

{

E  1

,  ⋯  ,

E  5

}

\{E_{1},\cdots,E_{5}\}

, resulting in their corresponding organization into same groups.

Despite their advantages, MoE models face a critical challenge: the lack of diversity and specialization among experts  (Zhou et al.,  2022 ; Sun et al.,  2024 ; Wang et al.,  2024 ; Feng et al.,  2025 ; Nakamura et al.,  2025 ) . In an MoE layer, a gating network determines token-to-expert mappings, ideally routing similar or relevant tokens to the same group of experts. However, the vanilla MoE layer often produces redundant experts that fail to achieve sufficient diversity and specialization, thereby undermining the performance of MoE models. Promoting diversity and specialization among experts remains a significant challenge for MoE models, especially as the number of experts scales.

Contributions:  In this paper, we reinterpret the combination of the most popular top-

k  k

routing and experts as a form of sparse representation. Building on this perspective, we propose a novel regularization technique that is applied to the routing inputs, rather than directly constraining the experts themselves, to promote greater diversity and specialization. Our approach does not require modifications to the MoE architecture and can be directly applied to any MoE model. The main contributions are highlighted below.

Firstly, we propose investigating MoE employing top-

k  k

routing from the perspective of sparse representation. Sparse representation aims to find the sparse code that effectively approximates the input when multiplied with the dictionary  (Donoho and Elad,  2003 ; Tropp,  2004 ) . For the MoE with top-

k  k

routing, the routing output can be viewed as the sparse code, while the output of each expert corresponds to a column in the dictionary. This perspective enables the application of the rich theoretical results of sparse representation to MoE.

Secondly, we propose introducing group sparse regularization to the inputs of top-

k  k

routing to indirectly impose structural constraints on experts. Group sparsity assumes that the input elements of top-

k  k

routing are organized into predefined groups, where all elements within a group tend to be either all zeros or all nonzeros  (Yuan and Lin,  2006 ; Eldar et al.,  2010 ) . These groups are typically formed based on an arbitrary partition, informed by prior knowledge specific to the task. As illustrated in Fig.

1

(b), we refer to the application of group sparse regularization in MoE as the Mixture of Group Experts (MoGE). Inspired by previous works  (Hyvärinen and Hoyer,  2001 ; Hyvärinen and Köster,  2007 ; Kavukcuoglu et al.,  2009 ) , we further organize the routing input into a 2D matrix, forming a topographic map where neighboring elements are grouped together to activate similar experts. This topographically organized input enables MoGE to extract locally invariant representations, encouraging diversity and specialization among experts.

Finally, we evaluate our MoGE across various image classification and language modeling tasks. Consistently, our MoGE demonstrates superior performance compared to its MoE counterpart, with only a negligible increase in memory consumption and computational cost. Additionally, our MoGE effectively promotes diversity and specialization among experts through an easily implementable approach, enabling the training of a greater number of experts and larger

k  k

values in top-

k  k

routing.

2.  Related Work

2.1.  Mixture-of-Experts

Mixture-of-Experts (MoE) was first introduced in the early 1990s  (Jacobs et al.,  1991 ; Jordan and Jacobs,  1994 ) . Later, Sh