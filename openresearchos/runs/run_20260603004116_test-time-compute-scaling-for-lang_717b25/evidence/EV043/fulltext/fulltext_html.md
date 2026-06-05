[2309.04255] LLMCad: Fast and Scalable On-device Large Language Model Inference

LLMCad : Fast and Scalable On-device Large Language Model Inference

Daliang Xu  ◆  , Wangsong Yin  ◆  , Xin Jin  ◆  , Ying Zhang  ◆  , Shiyun Wei  ★  , Mengwei Xu  ◇  , Xuanzhe Liu  ◆

◆ Key Lab of High Confidence Software Technologies (Peking University), Beijing, China

★ Zhongguancun Laboratory, Beijing, China.

◇ State Key Laboratory of Networking and Switching Technology (BUPT), Beijing, China

xudaliang, hg, xinjinpku, zhang.ying, weishiyun, liuxuanzhe@pku.edu.cn

yws@stu.pku.edu.cn

mwx@bupt.edu.cn

Abstract.

Generative tasks, such as text generation and question answering, hold a crucial position in the realm of mobile applications. Due to their sensitivity to privacy concerns, there is a growing demand for their execution directly on mobile devices. Currently, the execution of these generative tasks heavily depends on Large Language Models (LLMs). Nevertheless, the limited memory capacity of these devices presents a formidable challenge to the scalability of such models.

In our research, we introduce  LLMCad , an innovative on-device inference engine specifically designed for efficient generative Natural Language Processing (NLP) tasks. The core idea behind  LLMCad  revolves around model collaboration: a compact LLM, residing in memory, takes charge of generating the most straightforward tokens, while a high-precision LLM steps in to validate these tokens and rectify any identified errors.
 LLMCad  incorporates three novel techniques:
(1) Instead of generating candidate tokens in a sequential manner,  LLMCad  employs the smaller LLM to construct a token tree, encompassing a wider range of plausible token pathways. Subsequently, the larger LLM can efficiently validate all of these pathways simultaneously.
(2) It employs a self-adjusting fallback strategy, swiftly initiating the verification process whenever the smaller LLM generates an erroneous token.
(3) To ensure a continuous flow of token generation,  LLMCad  speculatively generates tokens during the verification process by implementing a compute-IO pipeline.
Through an extensive series of experiments,  LLMCad  showcases an impressive token generation speed, achieving rates up to 9.3

×

\times

faster than existing inference engines.

†

†  copyright:  none

1.  introduction

Generative tasks like text generation, question answering, and translation play a crucial role on mobile devices, as numerous applications rely on them to deliver key functionalities.
For instance, input method application like Google GBoard heavily leverages its text generation capabilities, while private assistant like Apple Siri uses it for question answering.
Such tasks are often privacy-sensitive and heavily rely on users’ private data, thereby necessitating on-device local inference.

(a)

Emergent abilities across various LLMs.

(b)

Generation latency of LLMs across various devices.

Figure 1 .

The memory wall hinders LLM’s “scaling law” on mobile devices.
*-Math, *-NLU, *-Mode, and *-GM denote LLMs’ emergent abilities: math reasoning, multi-task comprehension, mode arithmetic, and learning meaningful representations.

Large language models (LLMs), especially those built atop transformer decoder  ( vaswani2017attention,  )  such as GPT-3  ( brown2020language,  )  and LLaMA  ( touvron2023llama,  ) , have become the de-facto approach to solve NLP generative tasks.
Recent research in the machine learning community has demonstrated that scaling up such LLMs parameter size brings accuracy improvement and emergent ability  ( wei2022emergent,  ;  wei2022emergent,  ;  touvron2023llama,  ;  brown2020language,  ;  wei2022chain,  ) , as shown in Figure

1  (a).
In general, an LLM necessitates more than 1B parameters to learn meaningful representations  ( patel2021mapping,  ) , over 10B parameters to exhibit certain arithmetic reasoning abilities  ( cobbe2021training,  ) , and more than 30B parameters to achieve multi-task comprehension capabilities  ( hendrycks2020measuring,  ) .
This phenomenon is well-recognized in the machine learning community as the  scaling law

( kaplan2020scaling,  ;  aghajanyan2023scaling,  ;  clark2022unified,  ;  alabdulmohsin2022revisiting,  ) .

Key challenge: memory wall. 
However, our preliminary experiments in Figure

1  (b) reveal that the scaling ability is challenged on mobile devices.
Specifically, when LLMs are too large to be fit into device memory, mobile DNN engines like MNN  ( mnn,  ) a nd llama.cpp  ( llama-cpp,  )  need to repetitively release and load model weights.
It results in 59–224

×

\times

lengthened inference latency.
Such memory wall severely hinders the scaling law.
Users have to choose between real-time generation and emergent ability.
For instance, 10B parameters represent the minimum size required for LLaMA to possess arithmetic reasoning capabilities, yet it also represents the maximum parameter size for achieving real-time inference on smartphones (e.g., Xiaomi 10).

LLMCad : breaking memory wall through model collaboration.

In this paper, we propose  LLMCad , the first efficient inference engine for on-device generative NLP tasks.
 LLMCad  delivers LLM’s scaling ability to mobile devices with a tolerable generation speed through  model collaboration .
The main idea is to delegate most tokens to a smaller real-time LLM that can be totally hosted in device memory (namely memory-resident LLM).
The design is based on a key observation that, while a smaller LLM is inadequate to deliver satisfactory end-to-end sentences, they can correctly generate most easy tokens (e.g., determiners, pronouns, and punctuations).
Furthermore,
LLMs are often trained with a series of model variants, e.g. T5-Small/Base/Large  ( raffel2020exploring,  )  and LLaMa-7B/13B/33B  ( touvron2023llama,  ) , and its smaller counterpart (e.g., LLaMa-7B and T5-small, dubbed  memory-resident model  in this paper) can often be hosted in memory easily  ( touvron2023llama,  ;  xue2020mt5,  ;  raffel2020exploring,  ;  brown2020language,  ) .

LLMCad  employs a unique form of model collaboration, namely “generate-then-verify”  ( leviathan2023fast,  ;  chen2023accelerating,  ) .
In this approach, the memory-resident LLM serves as a token generator, while a target LLM acts as a verifier, using its output as the ground truth to inspect and rectify any errors introduced during the token generation process.
This approach provides two significant advantages:
(1)  No compromising accuracy. 
Each token is verified by the target model, therefore its accuracy is guaranteed.
This is crucial as a wrong token could propagate its error to the subquent tokens due to the autoregressive nature.
(2)  Fast verification.  As will be detailed in

§

§

\S

2.3  , the verification of

N

𝑁

N

tokens can be accomplished within one-shot inference o f the target model, therefore much faster than using it to generate

N

𝑁

N

tokens sequentially.

Despite these advantages, applying model collaboration for on-device LLM introduces three distinctive challenges:

•

Overlooked correct tokens with sub-optimal confidence. 
Typically, state-of-the-art LLM engines and studies always use the token with the highest probability as the output.
Nevertheless, our observation has revealed that some of generation errors by the memory-resident LLM can be rectified by the sub-optimal tokens.
Figure

4

gives a real-world example of such phenomenon.
Given the significant performance overhead associated with on-device verification,  LLMCad  must capitalize on these often-overlooked tokens to reduce the frequency of verification.

•

Verification timing. 
Another crucial aspect is determining when to initiate the verification process.
On-device verification is time-consuming, e.g., taking 7.1s on Jetson TX2.
Too early or too late verification just wastes computing mobile devices scarce resources by invalid verification (i.e., no errors detected) or usel