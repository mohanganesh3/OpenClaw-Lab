[2503.22996] Sparse Mixture of Experts as Unified Competitive Learning

Sparse Mixture of Experts as Unified Competitive Learning

Giang Do  Hung Le  Truyen Tran

Applied Artificial Intelligence Institute (A2I2), Deakin University

{s224363215,thai.le,truyen.tran}@deakin.edu.au

Corresponding author

Abstract

Sparse Mixture of Experts (SMoE) improves the efficiency of large language model training by directing input tokens to a subset of experts. Despite its success in generation tasks, its generalization ability remains an open question. In this paper, we demonstrate that current SMoEs, which fall into two categories: (1) Token Choice ;and (2) Expert Choice, struggle with tasks such as the Massive Text Embedding Benchmark (MTEB). By analyzing their mechanism through the lens of competitive learning, our study finds that the Token Choice approach may overly focus on irrelevant experts, while the Expert Choice approach risks discarding important tokens, potentially affecting performance. Motivated by this analysis, we propose Unified Competitive Learning SMoE (USMoE), a novel and efficient framework designed to improve the performance of existing SMoEs in both scenarios: with and without training. Extensive experiments across various tasks show that USMoE achieves up to a  10%  improvement over traditional approaches or reduces computational inference costs by  14%  while maintaining strong performance.

Sparse Mixture of Experts as Unified Competitive Learning

Giang Do  †

†  thanks:  Corresponding author

Hung Le  Truyen Tran

Applied Artificial Intelligence Institute (A2I2), Deakin University

{s224363215,thai.le,truyen.tran}@deakin.edu.au

1  Introduction

Sparse Mixture of Experts (SMoE) models have achieved notable success in natural language processing (NLP) and visual representation learning tasks  (Du et al.,  2022 ; Fedus et al.,  2022 ; Riquelme et al.,  2021a ; Shen et al.,  2023 ) . These advancements build on the Transformer architecture  (Vaswani et al.,  2017 )  and its variants  (Child et al.,  2019 ; Dai et al.,  2019b ) , which leverage large datasets and significant compute resources. However, training large Transformer models can be prohibitively expensive, requiring extensive compute hours  (Kaddour et al.,  2023 ) . To overcome this issue, SMoE models activate only a subset of experts for each input, reducing inference time compared to dense models  (Shazeer et al.,  2017a ; Zoph et al.,  2022 ; Artetxe et al.,  2022 ; Krajewski et al.,  2024 ) . The SMoE architecture can be categorized into two variants:  Token Choice , which assigns experts to each token  Dai et al. ( 2024 ); Team ( 2024 ); Muennighoff et al. ( 2024 ); Jiang et al. ( 2024a ) , and  Expert Choice , which assigns tokens to each expert  Zhou et al. ( 2022b ) . The advantage of Token Choice lies in its ability to dynamically select experts for each token, while Expert Choice ensures a more balanced token distribution across experts.

Figure 1:  We compare the performance of USMoE (ours) with the Expert Choice and Token Choice approaches on the Massive Text Embedding Benchmark (MTEB). The results demonstrate that our method outperforms traditional approaches across six tasks using OLMoE-7B

Muennighoff et al. ( 2024 )  without additional training.

Despite their promising results, SMoE models have several limitations. The Expert Choice approach suffers from token dropping

Zhou et al. ( 2022b ) , while the Token Choice approach struggles with unbalanced expert loading

Shazeer et al. ( 2017b ) . Additionally, both approaches are prone to representation collapse, where either a few experts dominate the routing or all experts learn similar representations  Chi et al. ( 2022a ); Chen et al. ( 2022 ) . Recent research has explored improving router policies  Chi et al. ( 2022b ); Chen et al. ( 2023a ); Do et al. ( 2023a )  to mitigate these issues. However, existing methods face two key challenges: (1) the use of auxiliary losses requires balancing router loss and task loss, leading to trade-offs, and (2) they still struggle with the fundamental limitations of either the Token Choice or Expert Choice approach.

Which approach is better for SMoE -  Token Choice  or  Expert Choice  - in terms of generalization across multiple tasks, both with and without training?

In this paper, we address this question by reexamining SMoE through the lens of Competitive Learning  Rumelhart and Zipser ( 1985 ); Kaski and Kohonen ( 1994 ); Srivastava et al. ( 2013 ); Pham et al. ( 2024a ) . From this perspective, Token Choice can be seen as horizontal competitive learning, where the most similar expert is selected for each token, while Expert Choice represents vertical competitive learning, where each expert selects the most similar tokens. This viewpoint reveals a key trade-off: Expert Choice risks dropping important tokens, whereas Token Choice must process both relevant and irrelevant tokens.

Building on this analysis, we propose  Unified Competitive Learning SMoE (USMoE) , a robust and efficient framework comprising two key components: (1)  Unified Competitive Score  and (2)  Unified Competitive Mechanism . These components enable the SMoE model to dynamically prioritize tokens or experts while ensuring the selection of the most similar token-expert pair, enhancing both robustness and effectiveness. To demonstrate the effectiveness of our approach, we evaluate USMoE across multiple scenarios, including pretraining and both fine-tuned and non-fine-tuned settings. USMoE consistently outperforms baseline methods across these scenarios, with particularly strong gains in tasks that require deep input understanding, such as semantic textual similarity, classification, and clustering. Extensive experiments across various benchmarks show that USMoE achieves up to a  10%  improvement over traditional approaches or reduces inference computational costs by  14% , all while maintaining high performance.

In summary, this paper makes the following key contributions:

•

We introduce a  Competitive Learning  perspective on SMoE, highlighting the weaknesses of existing approaches.

•

We propose  USMoE , a robust and efficient framework that addresses the limitations of both Token Choice and Expert Choice.

•

We  theoretically demonstrate  that USMoE effectively mitigates representation collapse, outperforming baseline methods.

•

We conduct  extensive experiments  on large language models, covering pretraining and both fine-tuned and non-fine-tuned settings, providing a detailed analysis of USMoE’s performance and effectiveness.

2  Related Work

Sparse Mixture of Experts (SMoE).

Sparse Mixture of Experts (SMoE), an extension of the Mixture of Experts framework  (Jacobs et al.,  1991 ; Jordan and Jacobs,  1994 ) , has gained traction with large language models and has since been applied in various domains, including computer vision and speech recognition  (Zhou et al.,  2022c ; Riquelme et al.,  2021b ) . The SMoE architecture consists of two main variants:  Token Choice , where experts are assigned to each token  (Shazeer et al.,  2017b ; Fedus et al.,  2022 ; Jiang et al.,  2024b ; Do et al.,  2024b ) , and  Expert Choice , where tokens are assigned to specific experts  (Zhou et al.,  2022b ) .

Token Choice treats all tokens equally, which has raised concerns among researchers  (Wu et al.,  2021 ; Hou et al.,  2022 ; Lin et al.,  2025 ) , while Expert Choice suffers from token-dropping issues. Additionally, SMoE faces the challenge of representation collapse, where experts produce similar outputs. Various solutions have been proposed, such as XMoE, which employs low-dimensional routing scores  (Chi et al.,  2022b ) , and SMoE-dropout, which gradually activates more experts  (Chen et al.,  2023a ) . Other approaches, including HyperRouter  (Do et al.,  2023a )  and StableMoE  (Dai et al.,  2022a ) , focus on enhancing router stability and robustness. Although these advancements have imp