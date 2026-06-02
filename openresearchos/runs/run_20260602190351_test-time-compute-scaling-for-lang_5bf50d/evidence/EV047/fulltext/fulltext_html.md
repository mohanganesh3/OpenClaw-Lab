[2312.05693] \M: Activation-Guided Quantization for Faster Inference of LLMs on the Edge

\M :  A ctivation- G uided Quantization

for Faster  I nference of  L LMs on the  E dge

Xuan Shen 1 ,
Peiyan Dong  1

1  footnotemark:

1

1 ,
Lei Lu 1 ,
Zhenglun Kong 1 ,

Zhengang Li 1 ,
Ming Lin 2 ,
Chao Wu 1 ,
Yanzhi Wang 1

These authors contributed equally.Work done before joining Oracle.

Abstract

Large Language Models (LLMs) stand out for their impressive performance in intricate language modeling tasks.
However, their demanding computational and memory needs pose obstacles for broad use on edge devices.
Quantization is then introduced to boost LLMs’ on-device efficiency.
Recent works show that 8-bit or lower weight quantization is feasible with minimal impact on end-to-end task performance,
while the activation is still not quantized.
On the other hand,
mainstream commodity edge devices still struggle to execute these sub-8-bit quantized networks effectively.
In this paper, we propose  \M , an activation-guided quantization framework for popular Large Language Models (LLMs), and implement an end-to-end accelerator on multiple edge devices for faster inference.
Considering the hardware profiling and activation analysis, we first introduce a basic activation quantization strategy to balance the trade-off of task performance and real inference speed.
Then we leverage the activation-aware token pruning technique to reduce the outliers and the adverse impact on attentivity.
Ultimately, we utilize the SIMD-based 4-bit multiplier and our efficient TRIP matrix multiplication to implement the accelerator for LLMs on the edge.
We apply our framework on different scales of LLMs including LLaMA, OPT, and BLOOM with 4-bit or 8-bit for the activation and 4-bit for the weight quantization.
Experiments show that  \M  achieves simultaneous quantization of model weights and activations while maintaining task performance comparable to existing weight-only quantization methods.
Moreover, in the 8- and 4-bit scenario,  \M  achieves an on-device speedup of up to 2.55x compared to its FP16 counterparts across multiple edge devices, marking a pioneering advancement in this domain.

Introduction

Large Language Models (LLMs)  (Touvron et al.  2023 ; Zhang et al.  2022 ; Brown et al.  2020a ; Radford et al.  2019 ; Brown et al.  2020b )  based on the Transformer  (Vaswani et al.  2017 )  family have breakthrough performance in Natural Language Processing (NLP) research area.

Application Scenarios .
In real-world decision scenarios, incorporating LLMs inference as a crucial element often necessitates stringent latency requirements. However, one drawback of LLMs is their computational and storage cost, which ranks among the highest for known models.
Consider GPT3-175B as an example. When stored in a compact float16 format, its parameters require 326GB (in multiples of 1024) of memory. This surpasses the capacity of even the most powerful individual GPUs, not to mention the challenges of running it on hardware-limited edge devices with acceptable latency.
Quantization, in particular, offers a promising approach to substantially improve the inference throughput and energy efficiency of LLMs on edge devices.
This improvement is achieved by harnessing the highly effective 8-bit fixed-point (INT8) operations supported by the SIMD units that are commonly found in edge platforms, such as CPUs and Raspberry Pis.

Current Limitations .
Before fully realizing the on-device benefits of model quantization on LLMs, it’s crucial to address two pressing issues that demand careful attention.
❶
Existing works  (Frantar et al.  2022 ; Lin et al.  2023 ; Xiao et al.  2022 )  primarily concentrate on weight-only (4-bit) quantization while leaving activations in the floating-point (FP16) domain.
This approach limits the efficient speed-up of model inference on common edge devices, which typically only support 16x16 and 8x8 integer multipliers. Specifically, activation quantization often has a detrimental effect on task performance, especially when the model size becomes large, due to the emergence of pronounced outliers in activations.
Experiments done by work  (Dettmers et al.  2022 )  indicate that directly setting these outliers to zero can result in a substantial 45% degradation in task performance.
Additionally, given the large model size of LLMs, limited academic computing power makes it challenging to afford the associated training costs.
Consequently, Post-Training Quantization (PTQ) has become a prevalent approach, but it falls short of minimizing the quantization error caused by these outliers.
In summary, quantizing the activations of LLMs while handling outliers inside activations is a crucial yet challenging issue.
❷
Mainstream edge processors, such as CPUs and Raspberry Pis, leverage SIMD units to execute multiple operations in parallel efficiently.
SIMD instructions are adept at exploiting byte-level data (8-bit integers) parallelism and are well-supported in common ISAs (Instruction Set Architectures) and DNN processing frameworks.
Examples include GEMMLOWP  (Jacob and Warden  2017 )  in TensorFlow Lite and QNNPACK  (Dukhan, Wu, and Lu  2018 )  in PyTorch.
Their low-precision kernels merely zero-extend the sub-byte operands to align them with byte boundaries, treating them as 8-bit or 16-bit operands.

In this paper, we address the above on-device quantization issues while enjoying the powerful performance provided by LLMs.
We propose  \M , an activation-guided quantization framework for faster inference of LLMs on the edge.
Specifically, we begin with a fundamental activation quantization strategy based on hardware latency profiling and activation analysis of LLMs, aiming to strike a balance between task performance and on-device inference speed.
We subsequently utilize the activation-aware pruning method to optimize quantization.
This is crucial because quantized tokens often exhibit numerous outliers, causing their attention to shift from the first position to nearby local positions.
By pruning tokens, we effectively eliminate some outliers, as they typically concentrate within the same or adjacent channels of different tokens.
Also, the removal of inattentive tokens can reduce the interaction distance between important tokens.
Finally, we design the edge-oriented optimization for the hardware implementation of  \M .
It consists primarily of two components: a SIMD-based 4-bit multiplier to facilitate efficient 4x4 INT4 multiplication, and our efficient Two-Refine Improved by Pruning (TRIP) matrix multiplication designed to mitigate the adverse impact of outliers.

The popular LLMs models such as LLaMA  (Touvron et al.  2023 ) , OPT  (Zhang et al.  2022 ) , and BLOOM  (Scao et al.  2022 )  are adopted to verify the effectiveness of our framework and the efficiency of our method on multiple edge devices.
 \M  can maintain state-of-the-art task performance comparable with weight-only works while achieving practical on-device speedup up to 2.55x.

The contributions of this work are summarized as follows:

•

We design the activation-guided and edge-oriented quantization strategy for the balance of latency decreasing and task performance.

•

We design an activation-aware token pruning method to minimize the negative impact on task performance caused by the outliers and the local attentivity.

•

We propose the SIMD-based 4-bit multiplier and an efficient TRIP matrix multiplication for effective hardware implementation.

•

We achieve state-of-the-art task performance on several popular datasets with practical on-device speedup.

Background and Related Works

In this section, we first focus on the backgound of post-training quantization for LLMs.
Then we discuss the low-bit computation on general edge devices.

Post-Training Quantization for LLMs

Post-Training Quantization (PTQ) techniques are widely used for one-shot compressing models, particularly for Large Language Models (LL