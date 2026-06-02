[2406.19598] Mixture of In-Context Experts Enhance LLMs’ Long Context Awareness

Mixture of In-Context Experts

Enhance LLMs’ Long Context Awareness

Hongzhan Lin  1

Ang Lv  1

1

1  footnotemark:

1

Yuhan Chen  2

1

1  footnotemark:

1

Chen Zhu  3

Yang Song  4

Hengshu Zhu  3

Rui Yan  1

1

Gaoling School of Artificial Intelligence, Renmin University of China

2

XiaoMi AI Lab

3

Career Science Lab, BOSS Zhipin

4

NLP Center, BOSS Zhipin

{linhongzhan, anglv, ruiyan}@ruc.edu.cn

{chenyuhan5}@xiaomi.com

Equal contribution.
Hongzhan Lin and Ang Lv proposed the idea of MoICE.
Hongzhan Lin and Yuhan Chen designed the MoICE router architecture and implemented efficient code.
Experiments were conducted by Hongzhan Lin, while Ang Lv led the writing.
Code is available at  https://github.com/p1nksnow/MoICE .
Corresponding author: Rui Yan ( ruiyan@ruc.edu.cn )

Abstract

Many studies have revealed that large language models (LLMs) exhibit uneven awareness of different contextual positions.
Their limited context awareness can lead to overlooking critical information and subsequent task failures.
While several approaches have been proposed to enhance LLMs’ context awareness, achieving both effectiveness and efficiency remains challenging.
In this paper, for LLMs utilizing RoPE as position embeddings, we introduce a novel method called “Mixture of In-Context Experts” (MoICE) to address this challenge.
MoICE comprises two key components: a router integrated into each attention head within LLMs and a lightweight router-only training optimization strategy:
(1) MoICE views each RoPE angle as an ‘in-context’ expert, demonstrated to be capable of directing the attention of a head to specific contextual positions.
Consequently, each attention head flexibly processes tokens using multiple RoPE angles dynamically selected by the router to attend to the needed positions.
This approach mitigates the risk of overlooking essential contextual information.
(2) The router-only training strategy entails freezing LLM parameters and exclusively updating routers for only a few steps.
When applied to open-source LLMs including Llama and Mistral, MoICE surpasses prior methods across multiple tasks on long context understanding and generation, all while maintaining commendable inference efficiency.

1  Introduction

Although large language models (LLMs) have demonstrated impressive capabilities across diverse NLP tasks, several studies  [ 22 ,  6 ]  have pointed out that the contextual awareness of LLMs is not as powerful as widely believed, constraining their application in tasks demanding extensive contextual awareness, such as coherent long text generation  [ 38 ]  and Retrieval-Augmented Generation (RAG,

[ 15 ,  4 ,  8 ] ) tasks necessitating in-context retrieval  [ 6 ] .
Liu et al.  [ 22 ]  identified a common issue termed the “lost-in-middle” phenomenon, indicating that LLMs often exhibit a weaker awareness of information situated in the middle of the long context compared to the beginning or end.
Chen et al.  [ 6 ]  highlighted challenges arising from a mathematical property of RoPE  [ 29 ] , a wide-used positional embedding in LLMs, which impedes attention to specific positions within the long context.
Consequently, if critical information coincides with such positions, task performance suffers.

Many works  [ 19 ,  38 ,  6 ,  37 ]  have attempted to enhance the long-context awareness of LLMs.
Central to these efforts is the enhancement of attention heads which serve as the linchpin for contextual awareness, given that FFNs in language models do not introduce token interaction.
Chen et al.  [ 6 ]  proposed an inference algorithm named  Attention Buckets  (AB), which enhanced the context awareness of LLMs by executing

N

𝑁

N

inference instances, each with a distinct RoPE angle, and aggregated the outputs at the final layer.
Zhang et al.  [ 38 ]  observed the varying awareness of attention heads to contextual positions.
They proposed an inference algorithm named  Ms-PoE .
Ms-PoE enhances the utility of position-aware heads by re-scaling the positional embedding indices, equivalent to assigning each head a unique RoPE angle. Figure

1

illustrates these approaches.
However, these approaches each come with their own drawbacks: AB conducts excessive redundant FFNs calculations, leading to high memory consumption.
In Ms-PoE, determining a distinct re-scale factor for every attention head needs an additional forward pass.
Meanwhile, each attention head still depends on a single re-scaled static RoPE.
As highlighted by AB  [ 6 ] , this leads to limited awareness of certain contextual positions, thereby constraining its potential.
Moreover, a significant drawback of both AB and Ms-PoE lies in their static assignment of the RoPE angle for each attention head throughout the generation. However, as the generation progresses, the positions of crucial tokens shift, necessitating corresponding adjustments in the required RoPE angles for each head.

Figure 1:  Some methods developed to enhance LLMs’ context awareness.
(a) Attention Buckets  [ 6 ]  selects

N

𝑁

N

different RoPEs and conducts

N

𝑁

N

parallel inferences for each input.
The outputs are then aggregated in the final layer.
(b) Ms-PoE  [ 38 ]  employs a unique RoPE angle for each attention head.
However, it needs an additional forward pass for RoPE angle assignment.
(c) MoICE integrates a router within each attention head.
This novel plug-in selects several of the most suitable RoPE angles for each token.
The selected RoPE angles collectively contribute to computing the attention scores.
MoICE demonstrates superior memory efficiency and performance.

In this study, we present  Mixture of In-Context Experts  (MoICE), a novel plug-in of LLMs for enhancing context awareness.
Specifically, We conceptualize a unique RoPE angle as an “in-context expert,” as it can allocate a head’s more attention to certain contextual positions  [ 6 ] .
We integrate a router within each attention head, which discerns the potentially important tokens for the head and dynamically selects

K

𝐾

K

RoPE angles that provide comprehensive awareness of these tokens for attention computation.
Through the re-computation of only a few query-key dot products, attention patterns computed with selected RoPE angles are aggregated to produce the final attention pattern.
This approach yields two primary advantages:
(1) It eliminates unnecessary computational overhead in AB, enhancing efficiency.
(2) The dynamic expert selection of each head for arbitrary tokens introduces flexibility not attained in previous studies. This minimizes the risk of the initial RoPE angle assigned to a head failing to work due to crucial token positions shifting during generation.

Consequently, MoICE not only surpasses AB’s effectiveness but also achieves commendable efficiency.
We name our approach as “Mixture of In-Context Experts” (MoICE) due to the aggregation of attention patterns calculated with different RoPE angles resembling the concept of “Mixture of Experts” (MoE,  [ 28 ] ).
When applying MoICE to open-source LLMs, we freeze LLMs’ parameters and conduct lightweight training only on the MoICE routers.
With only a few quick updates, MoICE surpasses many competitive baselines in tasks involving long-context generation and understanding.

In summary, our main contribution is the introduction of MoICE, a novel plug-in for enhancing LLMs’ context awareness.
It achieves head-and token-specific dynamic multiple RoPE angles assignment, outperforms previous methods across various tasks, and maintains commendable inference efficiency.

2  Background

We introduce some background of  Mixture of In-Context Experts , including (1) the rotary position embeddings commonly used by mainstream LLMs, (2) the primary problem addressed in this paper: the limited context awareness of LLMs, (3) an explanation of the underlying reasons for this limitation, and (4) the Mixture of