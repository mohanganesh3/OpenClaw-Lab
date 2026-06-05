[2405.17741] LoRA-Switch: Boosting the Efficiency of Dynamic LLM Adapters via System-Algorithm Co-design

LoRA-Switch: Boosting the Efficiency of Dynamic LLM Adapters via System-Algorithm Co-design

Rui Kong  1,2

&amp;Qiyang Li  2

&amp;Xinyu Fang  2

&amp;Qingtian Feng  2,4

1

1  footnotemark:

1

&amp;Qingfeng He  2

Yazhu Dong  2

&amp;Weijun Wang  2

&amp;Yuanchun Li  2,3

&amp;Linghe Kong  1

2

2  footnotemark:

2

&amp;Yunxin Liu  2,3

&amp;

1  Shanghai Jiao Tong University

2  Institute for AI Industry Research (AIR), Tsinghua University

3  Shanghai Artificial Intelligence Laboratory

4  National University of Singapore

Work was done while the author was interning at Institute for AI Industry Research (AIR), Tsinghua University. Corresponding authors: Yuanchun Li, Linghe Kong.

Abstract

Recent literature has found that an effective method to customize or further improve large language models (LLMs) is to add dynamic adapters, such as low-rank adapters (LoRA) with Mixture-of-Experts (MoE) structures. Though such dynamic adapters incur modest computational complexity, they surprisingly lead to huge inference latency overhead, slowing down the decoding speed by 2.5+ times. In this paper, we analyze the fine-grained costs of the dynamic adapters and find that the fragmented CUDA kernel calls are the root cause. Therefore, we propose LoRA-Switch, a system-algorithm co-designed architecture for efficient dynamic adapters. Unlike most existing dynamic structures that adopt layer-wise or block-wise dynamic routing, LoRA-Switch introduces a token-wise routing mechanism. It switches the LoRA adapters and weights for each token and merges them into the backbone for inference. For efficiency, this switching is implemented with an optimized CUDA kernel, which fuses the merging operations for all LoRA adapters at once. Based on experiments with popular open-source LLMs on common benchmarks, our approach has demonstrated similar accuracy improvement as existing dynamic adapters, while reducing the decoding latency by more than 2.4 times.

1  Introduction

Large language models (LLMs) have demonstrated remarkable capabilities in language understanding and generation. To customize the pretrained models to vertical domains or further enhance their capabilities, various adapter techniques such as Low-Rank Adapters (LoRA)  [ 13 ] , LLaMA-Adapter  [ 43 ] , and Prompt Tuning  [ 16 ]  have been employed with great success. These methods are acclaimed for boosting the accuracy of LLM without extensive training, thus facilitating efficient model customization and performance enhancement. Among these, dynamic adapters  [ 7 ,  10 ,  21 ,  25 ]  represent an even more potent strategy to augment the capacity of adapters.
By integrating conditionally computed lightweight adapters into the pretrained model, dynamic adapters allow for selective fine-tuning of adapter parameters.
This technique not only maintains the original strengths of the model but also substantially increases its adaptability and capacity.

However, we found that despite the relatively minor impact of dynamic adapters on parameter size and computing complexity (typically adding only 1-5% of the origin model), they may introduce significant latency overhead.
For instance, the dynamic adapters that we studied all increase decoding inference latency by 250-950%.
The seemingly modest computational complexity of the low-rank matrices employed results in substantial extra CUDA kernel execution latency, surpassing that of models without dynamic adapters.
This dramatic increase in latency is primarily attributed to the prolonged execution time of context operations during CUDA kernel runs, which considerably exceeds the actual computation time.
Dynamic adapters often require four or more additional CUDA kernel calls for each layer, in stark contrast to just a single call needed for the forward computation of the original backbone matrix.
This excessive number of context operations substantially amplifies the latency overhead, leading to a severe escalation of inference latency.

Reducing the inference latency overhead of dynamic adapters is challenging. Existing dynamic adapters  [ 6 ,  7 ,  8 ,  10 ,  17 ,  21 ,  25 ,  37 ,  40 ]  adopt block-wise or layer-wise routing structures, where activated LoRA adapters must be computed separately. If we were to pre-merge activated LoRA adapters into the backbone weights for forward computation, akin to the strategy employed by LoRA  [ 13 ] , it would fundamentally reduce the number of CUDA kernel calls. However, this merging changes the parameters of the LLM model, and for the next input, different adapters might be activated. Thus, after processing the current input, it becomes necessary to unmerge the activated adapters from the LLM model that was altered. The routing dynamicity involved in such merging processes becomes prohibitively costly. For instance, the widely adopted dynamic adapter structure, the Mixture of Experts (MoE), determines the activated adapters based on the output of the last layer. Each layer of dynamic adapters must then wait for its router to compute the gating scores before it can proceed to merge the activated LoRA adapters. This fragmented operational approach can inadvertently introduce even greater overhead costs.

Our approach to addressing the challenge is based on a holistic system-algorithm co-design. Specifically, we have developed a MoE-based dynamic adapters structure that facilitates token-wise adapter routing. Each token is associated with

k

𝑘

k

weighted paths of LoRA adapters, activated prior to the decoding of the token.
This setup ensures that, although the model is enhanced with dynamic structures, the inference process for each token remains relatively static due to the pre-determined adapters.
To further enhance the efficiency, we pre-merge the activated LoRA adapters into the pretrained model’s backbone before each token’s decoding.
This strategy fundamentally reduces the CUDA kernel execution overhead, thereby significantly lowering latency.
With this innovative setup, we have re-engineered the inference process to seamlessly switch and merge adapters for each token, aligning the process closely with the original pretrained LLM’s token decoding.
Another pivotal component of our system is the development of a fused CUDA kernel, named SGMM, which efficiently manages the activated and inactivated adapters.
This engineering solution ensures a smooth integration of dynamic adapters, optimizing both performance and efficiency.

We evaluate our LoRA-Switch design across a range of benchmarks, comparing it against multiple state-of-the-art dynamic adapter baselines. The experiment results demonstrate that our approach are comparable with well-established strong baselines. Notably, our method significantly reduces the running overhead associated with other dynamic adapter alternatives, achieving an average speedup of 2.4 times in decoding latency.

In summary, our contributions are as follows:

•

We uncover the high latency overhead introduced by dynamic adapters, which is a practical issue usually neglected by existing approaches. We analyze the fundamental reasons behind such high overhead, providing insights on the computational bottlenecks.

•

We introduce a novel architecture for dynamic adapters, named LoRA-Switch. This design enhances the capacity of LLM adapters while minimizing the latency overhead, thereby offering an optimal balance between performance and efficiency.

•

Through extensive experiments, we demonstrate that LoRA-Switch not only achieves accuracy on par with existing dynamic adapters across a variety of general and domain-specific tasks, but it also cuts down decoding inference latency by more than 2.4 times.

2  Background and Motivation

2.1  Dynamic Adapters

Given the strengths of both the Mixture of Experts (MoE)  [ 14 ,  30 ,  31 ,  33 ,  39 ]  and Low-Rank Adaptation (LoRA)  [ 13 ] ,