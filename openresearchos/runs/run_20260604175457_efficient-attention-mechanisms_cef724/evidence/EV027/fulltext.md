[2510.15926] TeLLMe v2: An Efficient End-to-End Ternary LLM Prefill and Decode Accelerator with Table-Lookup Matmul on Edge FPGAs

TeLLMe v2: An Efficient End-to-End  Te rnary  LLM  Prefill and Decode Accelerator with Table-Lookup Matmul on  E dge FPGAs

Ye Qiao

yeq6@uci.edu

University of California, Irvine  Irvine  California  USA

,

Zhiheng Chen

zhihenc5@uci.edu

University of California, Irvine  Irvine  California  USA

,

Yifan Zhang

yifanz58@uci.edu

University of California, Irvine  Irvine  California  USA

,

Yian Wang

yianw11@uci.edu

University of California, Irvine  Irvine  California  USA

and

Sitao Huang

sitaoh@uci.edu

University of California, Irvine  Irvine  California  USA

Abstract.

With the emergence of wearable devices and other embedded systems,
deploying large language models (LLMs) on edge platforms becomes an urgent need. However, it is challenging because of their high computational and memory demands. Although recent low-bit quantization methods (e.g., BitNet, DeepSeek) compress weights to as low as 1.58 bits with minimal accuracy loss, edge deployment is still constrained by limited on-chip resources, power budgets, and the often-neglected long latency of the prefill stage.
We present TeLLMe, the first table-lookup-based ternary LLM accelerator for low-power edge FPGAs that fully supports both prefill and autoregressive decoding using 1.58-bit weights and 8-bit activations. TeLLMe incorporates our proposed novel techniques including (1) a table-lookup-based ternary matrix multiplication (TLMM) engine utilizing grouped activations and online precomputation for low resource utilization and high throughput; (2) a fine-grained analysis and optimization onan analytic and fine-grained URAM-based weight buffer management of weight loading from global memory and compute engine weight access; (3) a streaming dataflow architecture that fuses floating-point element-wise operations with linear computations to hide latency; (4) a reversed-reordered prefill stage attention with fused attention operation for high memory efficiency; and (5) a resource-efficient specialized decoding stage attention. Under a 5W power budget, TeLLMe delivers up to 25 tokens/s decoding throughput and 0.45s to 0.96s Time-to-First-Token (TTFT) for 64–128 token prompts, marking a significant energy-efficiency advancement in LLM inference on edge FPGAs.

†

†  copyright:  none

†

†  conference:  ; ;

1.  Introduction

Large language models (LLMs) have been evolving rapidly, delivering state-of-the-art results in machine translation, code generation, question answering, and conversational AI. Models such as GPT-3  ( brown2020language,  ) , LLaMA  ( touvron2023LLaMA,  ) , and DeepSeek-R1  ( guo2025deepseek,  )  highlight the benefits of scale, but those gains arrive with steep cost increases in computation, memory, and energy.

Deploying LLMs on edge devices (embedded CPUs, GPUs, FPGAs, etc.) preserves privacy, reduces latency, and enables autonomy, yet it is difficult due to tight limits on memory bandwidth and capacity, compute resources, and power budgets. Autoregressive decoding further stresses these constraints: growing key–value (KV) caches, long-context handling, and strict latency requirements often become the dominating performance bottlenecks.

Extreme model compression via low-bit quantization has emerged as a promising technique  ( qiao2022two,  ;  10025006,  ) . BitNet  ( bitnet,  ;  qiao2025cobra,  ;  qiao2025tellme,  )  showed that Transformers can be trained with 1-bit weights; BitNet-1.58  ( bitnet158,  )  and DeepSeek  ( deepseek,  )  extend this idea to ternary quantization (

−  1

,  0  ,

+  1

{-1,0,+1}

), approaching full-precision quality while drastically reducing model size and energy. However, closing the gap between algorithmic compression and efficient, end-to-end deployment on actual edge hardware requires a co-design solution that simultaneously optimizes compute, memory, and scheduling.

While researchers have proposed several solutions to accelerate the decoding stage of LLMs, the existing works often ignore the significant latency of the prefill stage, making prefill a critical performance bottleneck in the end-to-end LLM systems.
For example,  ( li2025pushing,  )  demonstrates efficient decoding on embedded FPGAs but leaves the prefill stage largely unaddressed. On device, prefill latency directly impacts users’ perceived responsiveness and safety; it is not hidden behind cloud-scale parallelism.
Therefore, prefill should also be treated as a first-class citizen and carefully accelerated alongside decoding when designing an edge accelerator.

We present  TeLLMe —the  Te rnary  L arge  L anguage  M odel  e dge accelerator—an edge FPGA accelerator, to our knowledge, the first to incorporate a table-lookup (TL) matrix multiplication (MatMul) approach tailored for ternary LLM inference with full support for both  prefill  stage and  decoding  stage. TeLLMe targets cost-effective low-power FPGAs (AMD Kria KV260), supports 1.58-bit (ternary) weights and 8-bit activations, and co-optimizes compute, memory, and scheduling for low-latency, energy-efficient LLM inference.

Prior FPGA accelerators  ( LUTNET,  ;  SUMLUTNET,  ;  TLMAC,  )  leveraged table-lookup techniques to speed up low-precision arithmetics in other domains, demonstrating the viability of lookup table (LUT)-centric accelerators. However, to our knowledge, TeLLMe is the first to apply a  ternary, table-lookup matmul  design to end-to-end LLM inference (prefill and decoding). Furthermore, it integrates LLM-specific optimizations—fused attention for prefill, disaggregated prefill/decoding dataflows, streaming fusion of dequantization, quantization, element-wise operations, etc., and URAM-aware weight orchestration. With these optimizations, our proposed TeLLMe delivers state-of-the-art efficiency on actual FPGA boards.
Our key contributions include:

•

End-to-end accelerator for ternary LLMs.  We build, to our knowledge, the first edge FPGA accelerator that employs a table-lookup matmul engine for ternary LLMs with full support for both prefill and decoding, achieving up to 25 tokens/s generation throughput and up to 143 tokens/s prefill throughput while consuming under 5 W.

•

Ternary table-lookup matmul.  We propose a table-lookup-based matrix multiplication (TLMM) unit optimized for FPGAs that reuses grouped activations and performs online accumulation for ternary multiplications across attention projections and the feed-forward network (FFN).

•

URAM-aware weight orchestration.  We introduce fine-grained URAM buffering for high-throughput weight streaming from off-chip memory and provide an analytic method to optimize TLMM engine parameter selection under URAM and LUT constraints.

•

Streaming fusion with mixed precision.  We fuse floating-point (FP) dequantization, integer (INT) quantization, and elementwise operations (residual add, rotary embedding, etc) around the INT-based TLMM to enable mixed-precision execution and overlap compute with dataflow architecture.

•

Disaggregated prefill and decoding attention.  We separate the attention pipelines to match their distinct compute/memory patterns. For prefill, we design a fused attention unit with a reversed-attention mechanism and a fully fused pipeline that minimizes off-chip traffic and avoids redundant masked computation. For decoding, we exploit on-chip memory to retain intermediate softmax scores and prevent DDR reloads.

In summary, TeLLMe demonstrates that a LUT/URAM-native, table-lookup matmul engine co-designed with LLM-specific dataflows can unlock end-to-end ternary inference on resource-constrained FPGAs. To our knowledge, it is the first accelerator to realize table-lookup matmul for LLMs with comprehensive support for prefill and decoding on real hardware, establishing a strong baseline for energy-efficient, low-latency generative AI at the edge.

2.  Background and Related Work

2.1.  Tr