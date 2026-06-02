[2506.01850] MoDA: Modulation Adapter for Fine-Grained Visual Grounding in Instructional MLLMs

MoDA: Modulation Adapter for Fine-Grained Visual Grounding in Instructional MLLMs

Wayner Barrios  1

Andres Villa  2

Juan Leon Alcazar  2

SouYoung Jin  1

Bernard Ghanem  2

1 Department of Computer Science, Dartmouth

2 KAUST

Abstract

Recently, Multimodal Large Language Models (MLLMs) have demonstrated impressive performance on instruction-following tasks by integrating pretrained visual encoders with large language models (LLMs). However, existing approaches often struggle to ground fine-grained visual concepts in complex scenes. In this paper, we propose MoDA (Modulation Adapter), a lightweight yet effective module designed to refine pre-aligned visual features through instruction-guided modulation. Our approach follows the standard LLaVA training protocol, consisting of a two-stage process: (1) aligning image features to the LLM’s input space via a frozen vision encoder and adapter layers, and (2) refining those features using the MoDA adapter during the instructional tuning stage. MoDA employs a Transformer-based cross-attention mechanism to generate a modulation mask over the aligned visual tokens, thereby emphasizing semantically relevant embedding dimensions based on the language instruction. The modulated features are then passed to the LLM for autoregressive language generation. Our experimental evaluation shows that MoDA improves visual grounding and generates more contextually appropriate responses, demonstrating its effectiveness as a general-purpose enhancement for image-based MLLMs.

1  Introduction

The rapid progress of Large Language Models (LLMs) has led to impressive zero-shot performance across a broad spectrum of natural language processing benchmarks  Wang et al. ( 2024 ); Chung et al. ( 2024 ); Liang et al. ( 2023 ); Llama Team, AI @ Meta ( 2024 ); Yang et al. ( 2024 ); Team ( 2025 ) . The success of instruction-tuned LLMs has driven computer vision research in a similar path and ultimately led to the development of Multimodal Large Language Models (MLLMs). MLLMs emerged as the integration of pretrained visual encoders with large language models via a lightweight adapter module. These adapters enable efficient and modular alignment between modalities, thus allowing MLLMs to achieve a strong performance on a variety of multi-modal tasks, including Visual Question Answering (VQA), Image Captioning, Image Reasoning, and Image Classification.

(a)

Image patches for ViT input representation

(b)

Modulation Adapter (MoDA) Architecture

Figure 1 :

Overview of ViT patch representation and our proposed Modulation Adapter (MoDA) .  (a)  ViT splits the input image into fixed-size patches, each projected into a high-dimensional embedding. This rigid partitioning often blends semantically distinct elements (e.g., parts of the dog, toy, floor, and bed within a single patch), leading to entangled representations that hinder fine-grained visual understanding.  (b)  We introduce MoDA, a lightweight, plug-and-play module that modulates visual embeddings via cross-attention using language tokens as guidance. MoDA enables selective, task-driven attention without modifying the underlying the architecture or requiring additional supervision, thus improving alignment between vision and language representations.

Despite the success on MLLMs, state-of-the-art models frequently struggle with fine-grained visual grounding, leading to hallucinations in scenes that require detailed reasoning over the visual inputs. Hallucinations occur when the model outputs are inconsistent with the actual image semantics, thus undermining their reliability and safety in real-world scenarios. Prior analyses have identified that the CLIP-based visual encoder, commonly employed in MLLM, as a key bottleneck: its patch-based representations often fails to capture localized details  Villa et al. ( 2024 ); Tong et al. ( 2024 ); Kar et al. ( 2024 ) . To mitigate this, some works incorporate multiple specialized visual encoders to extract complementary features  Tong et al. ( 2024 ); Kar et al. ( 2024 ) , while others fine-tune CLIP to better preserve local structure  Villa et al. ( 2025 ) . Alternative approaches such as joint training objectives, vision-aware prompts, and early fusion strategies have also been explored (e.g., REVEAL  Hu et al. ( 2022 ) , Kosmos-2  Peng et al. ( 2023 ) ), but these often introduce substantial computational overhead or require large-scale retraining.

We illustrate some of the CLIP’s shortcomings with a practical example. Figure

1(a)

shows a

3  ×  3

3\times 3

grid (simulating CLIP’s visual tokenization, but with far larger tokens for easy visualization) over a sleeping French bulldog clutching a plush toy, we can see that none of the patches contain uniform and unique visual elements. For example, patch 5 contains the dog’s torso together with part of the stuffed toy and the cushioned bed; patch 6 covers the dog’s head and ear resting on the hardwood floor, mixing objects with different semantics, and diverse textures. The large number of visual attributes in a single patch forces the visual encoder to combine information about distinct shapes, textures, colors, more into a single visual embedding. As a consequence, multiple semantic elements are embedded into a high-dimensional representation in which individual features dimensions may encode multiple semantic
meanings  Oquab et al. ( 2024 ); Ma et al. ( 2022 ); Zhou et al. ( 2024 ); Shi et al. ( 2024a ) , not due to entanglement, but because a single representation can inherently carry multiple semantic cues. When a token blends unrelated concepts, it becomes harder for attention mechanisms to disentangle and prioritize relevant information. Therefore, language queries in the downstream tasks such as  “What color is the dog’s ear?”  or  “Is the toy lying on the bed or the floor?”  must first disentangle this visual representation in order to provide a reliable answer.

In natural language processing, dynamic attention masking mechanisms have proven effective in emphasizing task-relevant tokens without significantly increasing computation  Fan et al. ( 2021 ); Tang et al. ( 2021 ); Lin and Joe ( 2023 ); Rende et al. ( 2024 ) . Extending this idea to the multimodal domain,

Barrios and Jin ( 2024 )  introduced an adaptive masking over audiovisual sequences, however it requires a separate mask at every transformer layer, incurring in a heavy overhead when applied to deep models with

32  32

+ layers. This leads to a central question:  How can we enhance visual information processing in MLLMs?

To address this challenge, we draw inspiration from human visual attention, which dynamically shifts focus across different regions of a scene based on task demands. We introduce the  Modulation Adapter  ( MoDA ), a lightweight, orthogonal module that dynamically modulates the embeddings of the visual patch within the standard cross-attention mechanism. Treating visual embeddings as target (query) and language token embeddings as memory (keys and values), MoDA allows the model to respond selectively to language-guided visual cues without altering the underlying architecture (see Figure

1(b)  ). MoDA integrates seamlessly into existing two-stage instruction-tuning pipelines, requires no extra supervision or training data, and operates entirely in the shared embedding space.

We validate MoDA by integrating it into two strong MLLM baselines, LLaVA-1.5  Liu et al. ( 2024 )  and LLaVA-MoRE  Cocchi et al. ( 2025 ) , and evaluating on seven standard vision–language benchmarks. Our experiments demonstrate that MoDA consistently improves fine-grained visual grounding and substantially reduces hallucinations in complex scenes. On the MMVP benchmark, MoDA yields gains of  12.0%  on LLaVA-1.5 and  3.4%  on LLaVA-MoRE. Our contributions are as follows:  (i)  a lightweight adapter that dynamically modulate