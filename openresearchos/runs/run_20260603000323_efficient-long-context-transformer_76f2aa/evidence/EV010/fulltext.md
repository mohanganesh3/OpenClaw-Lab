[2502.12502] Efficient OpAmp Adaptation for Zoom Attention to Golden Contexts

Efficient OpAmp Adaptation for Zoom Attention to Golden Contexts

Haoyuan Wu

♠  \spadesuit

,
 Rui Ming

♠  \spadesuit

∗

,
 Haisheng Zheng

♡  \heartsuit

,
 Zhuolun He

♠  \spadesuit

,

♣  \clubsuit

,
 Bei Yu

♠  \spadesuit

♠  \spadesuit

The Chinese University of Hong Kong, Hong Kong SAR

♡  \heartsuit

Shanghai Artificial Intelligent Laboratory, China

♣  \clubsuit

ChatEDA Tech, China

{hywu24,byu}@cse.cuhk.edu.hk

These authors contributed equally to this work.

Abstract

Large language models (LLMs) have shown significant promise in question-answering (QA) tasks, particularly in retrieval-augmented generation (RAG) scenarios and long-context applications.
However, their performance is hindered by noisy reference documents, which often distract from essential information.
Despite fine-tuning efforts, Transformer-based architectures struggle to prioritize relevant content.
This is evidenced by their tendency to allocate disproportionate attention to irrelevant or later-positioned documents.
Recent work proposes the differential attention mechanism to address this issue, but this mechanism is limited by an unsuitable common-mode rejection ratio (CMRR) and high computational costs.
Inspired by the operational amplifier (OpAmp), we propose the OpAmp adaptation to address these challenges, which is implemented with adapters efficiently.
By integrating the adapter into pre-trained Transformer blocks, our approach enhances focus on the golden context without costly training from scratch.
Empirical evaluations on noisy-context benchmarks reveal that our Qwen2.5-OpAmp-72B model, trained with our OpAmp adaptation, surpasses the performance of state-of-the-art LLMs, including DeepSeek-V3 and GPT-4o.

Efficient OpAmp Adaptation for Zoom Attention to Golden Contexts

Haoyuan Wu

♠  \spadesuit

†

†  thanks:  These authors contributed equally to this work.

,
Rui Ming

♠  \spadesuit

∗  ,
Haisheng Zheng

♡  \heartsuit

,
Zhuolun He

♠  \spadesuit

,

♣  \clubsuit

,
Bei Yu

♠  \spadesuit

♠  \spadesuit

The Chinese University of Hong Kong, Hong Kong SAR

♡  \heartsuit

Shanghai Artificial Intelligent Laboratory, China

♣  \clubsuit

ChatEDA Tech, China

{hywu24,byu}@cse.cuhk.edu.hk

1  Introduction

Recent advancements in large language models (LLMs)  OpenAI ( 2023 ); Dubey et al. ( 2024 ); Yang et al. ( 2024 ); Liu et al. ( 2024a )  have demonstrated remarkable capabilities in understanding, generating, and reasoning across diverse domains, significantly advancing their application in various fields.
Among these applications, question answering (QA) based on provided contexts has emerged as one of the most prominent use cases for LLMs.

As LLMs’ capabilities continue to evolve and user expectations grow, users increasingly supply multiple documents retrieved in Retrieval-Augmented Generation (RAG) scenarios or long-context reference documents to guide LLMs in generating contextually relevant responses.
However, in practice, such retrieved documents or long-context references often contain substantial noise, including information irrelevant to the user’s query.
Recent studies  Ye et al. ( 2025 ); Liu et al. ( 2024b )  highlight a critical challenge that LLMs frequently struggle to accurately identify and extract key information from these noisy contexts, limiting their effectiveness in real-world applications.

Figure 1:  Normalized attention score. Transformers often miss the golden document in a noisy context.

As illustrated in

Figure ˜ 1  , we visualize the normalized attention scores assigned to retrieved documents in the RAG scenario, which includes various noisy documents and a single golden document.
The task involves identifying the correct answer within noisy contexts.
Our analysis evaluates several LLMs, including Llama3.1-8B-base  Meta ( 2024 ) , Llama3.1-8B-inst  Meta ( 2024 ) , and Llama3-ChatQA2-8B  Xu et al. ( 2024 ) , the latter of which has been fine-tuned specifically for long-context and RAG applications.
The visualization demonstrates that the Transformer architecture tends to allocate only a small proportion of attention scores to the golden document, while disproportionately focusing on irrelevant or later-positioned documents.
These findings highlight a persistent challenge for Transformer-based architectures including effectively identifying and prioritizing relevant documents in the presence of noise.
The issue  Ye et al. ( 2025 )  arises from the non-negligible allocation of attention scores to irrelevant content, which ultimately obscures the correct answer and undermines model performance.

Figure 2:  Qwen2.5-OpAmp-72B achieves the best average performance in various noisy-context benchmarks compared to current SOTA LLMs.

Ye et al. ( 2025 )  propose a differential attention mechanism designed to mitigate attention noise through differential denoising, inspired by the principles of differential amplifiers in electrical engineering.
However, differential amplifiers are effective in scenarios requiring a high common-mode rejection ratio (CMRR) considering that they only focus on differential gain.
This is unsuitable for attention denoising in the Transformer block.
Training a differential transformer from scratch entails great computation costs and introduces significant risks, further limiting its practical applicability.

Inspired by the operational amplifiers (OpAmp), we introduce OpAmp adaptation with adapters, an efficient approach for refining the attention mechanism to enhance focus on the most relevant context leveraging parameter-efficient fine-tuning (PEFT) techniques.
The OpAmp adaptation enables simultaneous control of differential gain and common-mode gain through the management of the CMRR.
Building on the OpAmp design, our approach facilitates the training of OpAmp models using pre-trained Transformer architectures, eliminating the need for training from scratch.
This strategy significantly reduces computational costs compared to previous methods.
As demonstrated in

Figure ˜ 2  , our Qwen2.5-OpAmp-72B model, trained with the OpAmp adaptation, achieves superior average performance across various noisy-context benchmarks compared to current state-of-the-art (SOTA) LLMs.
Our contributions are as follows:

•

We introduce the OpAmp adaptation for zoom attention to the most relevant context in noisy contexts;

•

Implement OpAmp adaptation with adapters, which are fine-tuned with our noisy context dataset, achieving significant improvements;

•

Develop OpAmp models with our OpAmp adaptation method, surpassing current SOTA LLMs in various noisy-context benchmarks.

2  Methods

2.1  Preliminaries

Adapters .
 Houlsby et al. ( 2019 )  introduced the concept of integrating adapters into pre-trained transformer-based models for PEFT.
This approach only fine-tunes the parameters introduced by the adapters while maintaining the pre-trained weights with large parameters unchanged.
An adapter module comprises two trainable matrices,

𝑾  1

∈

ℝ

d  1

×

d  2

\boldsymbol{W}_{1}\in\mathbb{R}^{d_{1}\times d_{2}}

and

𝑾  2

∈

ℝ

d  2

×

d  1

\boldsymbol{W}_{2}\in\mathbb{R}^{d_{2}\times d_{1}}

, along with a non-linear activation function

ϕ  ​

(  ⋅  )

\phi(\cdot)

.
Here,

d  1

d_{1}

represents the feature dimension of the pre-trained weights, while

d  2

d_{2}

denotes the hidden dimension of the inserted adapter, typically satisfying

d  2

≪

d  1

d_{2}\ll d_{1}

.
Given an input feature

𝑯  ∈

ℝ

N  ×

d  1

\boldsymbol{H}\in\mathbb{R}^{N\times d_{1}}

, the output of the adapter module is expressed as:

𝑯  ′

=

ϕ  ​

(

𝑯  ​

𝑾  1

)

​

𝑾  2

+  𝑯

.

\boldsymbol{H}^{\prime}=\phi(\boldsymbol{H}\boldsymbol{W}_{1})\boldsymbol{W}_{2}+\boldsymbol{H}.

(1)

Attention .
The self-attention mechanism  Vaswani et al. ( 2017 )  serves as the foundational building block for LLMs  OpenAI ( 2023 ); Dubey et al. ( 2024 ); Yang et a