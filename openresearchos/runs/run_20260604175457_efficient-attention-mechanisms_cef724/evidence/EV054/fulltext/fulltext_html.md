[2602.02159] Focus-dLLM: Accelerating Long-Context Diffusion LLM Inference via Confidence-Guided Context Focusing

Focus-dLLM: Accelerating Long-Context Diffusion LLM Inference via Confidence-Guided Context Focusing

Lingkun Long  1  ,
Yushi Huang  2,3  ,
Shihao Bai  3  ,
Ruihao Gong  1,3  ,

Jun Zhang  2  ,
Ao Zhou  1  ,
Jianlei Yang  1

1 Beihang University

2 Hong Kong University of Science and Technology

3 SenseTime Research

Abstract

Diffusion Large Language Models (dLLMs) deliver strong long-context processing capability in a non-autoregressive decoding paradigm. However, the considerable computational cost of bidirectional full attention limits the inference efficiency. Although sparse attention is promising, existing methods remain ineffective. This stems from the need to estimate attention importance for tokens yet to be decoded, while the unmasked token positions are unknown during diffusion. In this paper, we present  Focus-dLLM , a novel training-free attention sparsification framework tailored
for accurate and efficient long-context dLLM inference. Based on the finding that token confidence strongly correlates across adjacent steps, we first design a  past confidence-guided indicator  to predict unmasked regions. Built upon this, we propose a  sink-aware pruning strategy  to accurately estimate and remove redundant attention computation, while preserving highly influential attention sinks. To further reduce overhead, this strategy reuses identified sink locations across layers, leveraging the observed cross-layer consistency. Experimental results show that our method offers more than

29  ×

29\times

lossless speedup under

32  ​  K

32K

context length. The code is publicly available at:  https://github.com/Longxmas/Focus-dLLM .

Focus-dLLM: Accelerating Long-Context Diffusion LLM Inference via Confidence-Guided Context Focusing

Lingkun Long  1  ,
Yushi Huang  2,3  ,
Shihao Bai  3  ,
Ruihao Gong  1,3  ,

Jun Zhang  2  ,
Ao Zhou  1  ,
Jianlei Yang  1

1 Beihang University

2 Hong Kong University of Science and Technology

3 SenseTime Research

1  Introduction

Diffusion large language models (dLLMs)  Bie  et al.  ( 2025 ); Gong  et al.  ( 2025 ); Arriola  et al.  ( 2025 )  have recently emerged as a compelling non-autoregressive paradigm for text generation, replacing left-to-right token emission with iterative denoising over a fixed-length sequence  Li  et al.  ( 2022 ); Gong  et al.  ( 2022 ); Austin  et al.  ( 2021 ); Lou  et al.  ( 2023 ); He  et al.  ( 2023 ) .
By updating multiple positions in parallel and leveraging bidirectional attention, dLLMs offer an appealing path toward higher decoding throughput while retaining strong generation quality.
Moreover, recent studies have substantially extended the context length of dLLMs  Liu  et al.  ( 2025a ); He  et al.  ( 2025 ) , demonstrating effective long-context extrapolation and scaling to long inputs.

Nevertheless, efficient long-context inference remains a key obstacle for the dLLM due to its  non-autoregressive  decoding and bidirectional  full  attention nature. Prior methods  Wu  et al.  ( 2025 ); Liu  et al.  ( 2025b ); Ma  et al.  ( 2025 )  to address this challenge fall into two categories: ( i )  Approximated  KV cache and ( ii ) sparse attention. The former selectively refreshes KV states by exploiting strong redundancy between adjacent steps. However, attention computation is still costly over the  full  cached context. On the other hand, sparse attention  Tang  et al.  ( 2024 ); Xiao  et al.  ( 2024a ); Xu  et al.  ( 2025 ); Yuan  et al.  ( 2025 )  offers a practical solution, but it often requires token importance estimation using the  currently decoded  token as a query  Zhang  et al.  ( 2023 ); Xiao  et al.  ( 2024a ) . Since the positions to be decoded (unmasked) are not known in advance for dLLMs, recent works  Song  et al.  ( 2025 ); Huang  et al.  ( 2025 )  leverage inaccurate coarse-grained estimation, leading to suboptimal performance and limited efficiency. This paper, therefore, asks:  Can we accurately predict the positions of the unmasked tokens and only retain necessary computation to achieve more effective long-context inference acceleration for dLLMs?

To tackle this challenge, we first make an in-depth analysis to investigate the predictability of the unmasked tokens. In particular, we discover that the confidence scores at the same positions in two consecutive steps exhibit a strong positive correlation, and the positions of currently unknown tokens largely overlap with those that had the highest-confidence tokens in the previous step. Thus, unmasked positions for the current steps can be inferred from previous-step confidence. Besides, we also analyze the redundancy of attention patterns and observe that attention sink  Xiao  et al.  ( 2023 ); Ruscio  et al.  ( 2025 ) , which contributes significantly to the attention score in LLMs  Bai  et al.  ( 2023 ); Touvron  et al.  ( 2023 ) , displays notable cross-layer consistency for dLLMs. This phenomenon suggests sink tokens can be identified at an intermediate depth. Therefore, we can directly reuse them without re-identification in deeper layers.

Motivated by the above findings, we propose Focus-dLLM, a training-free sparse attention framework with approximated KV cache, to accelerate long-context dLLM inference. To begin with, we introduce a  past confidence-guided indicator  that uses confidence scores from step

t  −  1

t\!-\!1

to predict the unmasked positions at step

t  t

, and then window-expands them to preserve semantic coherence. Next, we design a  sink-aware pruning strategy  for diffusion decoding: Using the tokens within the positions predicted before as queries, we select only the most relevant tokens for attention while retaining step-wise attention sinks. Moreover, this approach shares the identified sink tokens across layers to further reduce additional overhead. Leveraging these novel techniques, our framework computes attention over the predicted unmasked queries and the selected necessary key-value pairs. As a result, it achieves considerable inference speedups without compromising performance throughout the dynamic decoding process.

Our contributions are summarized as follows:

•

We analyze diffusion inference dynamics and reveal a strong positive correlation of token confidence across adjacent denoising steps, together with dynamic and structured attention patterns in dLLMs.

•

We propose Focus-dLLM, a novel training-free acceleration framework that consists of a past confidence-guided indicator for predicting the next unmasked positions with a sink-aware dynamic token pruning strategy for efficient sparse attention.

•

Experiments show that Focus-dLLM achieves substantial speedups over baselines while preserving accuracy. For instance, it attains better-than-vanilla performance and delivers

2.05  ×

2.05\times

speedup over Fast-dLLM for UltraLLaDA at

32  ​  K

32K

context length.

2  Related Work

Diffusion large language models. 
Diffusion large language models (dLLMs)  Li  et al.  ( 2025 ); You  et al.  ( 2025 ); Chen  et al.  ( 2025 )  have emerged as a promising non-autoregressive paradigm that enables parallel token generation via iterative denoising. Prior works explore both continuous-space diffusion for text  Li  et al.  ( 2022 ); Gong  et al.  ( 2022 )  and discrete-token diffusion formulations  Austin  et al.  ( 2021 ); Lou  et al.  ( 2023 ); He  et al.  ( 2023 ) . Recent masked diffusion LMs  Nie  et al.  ( 2025 ); Zhu  et al.  ( 2025 ); Ye  et al.  ( 2025 )  have been successfully scaled up, demonstrating competitive performance against autoregressive counterparts at billion-parameter scales. Besides, long-context capability  Liu  et al.  ( 2025a ); He  et al.  ( 2025 )  for dLLMs has also been explored, which pushes the context window up to

≥

16  ​  K

{\geq}16K

tokens.

KV cache for dLLMs. 
Due to bidirectional attention and token states evolving