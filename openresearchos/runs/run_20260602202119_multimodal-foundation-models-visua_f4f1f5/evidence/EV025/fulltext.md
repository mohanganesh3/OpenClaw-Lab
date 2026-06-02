[2506.21873] Grounding-Aware Token Pruning: Recovering from Drastic Performance Drops in Visual Grounding Caused by Pruning

Grounding-Aware Token Pruning: Recovering from Drastic Performance Drops in Visual Grounding Caused by Pruning

Tzu-Chun Chien 1  Chieh-Kai Lin  Shiang-Feng Tsai 1

Ruei-Chi Lai 1

1

1  footnotemark:

1

Hung-Jen Chen 1  Min Sun 1

1 National Tsing Hua University (NTHU)

Equal contribution

Abstract

Recent Multimodal Large Language Models (MLLMs) have demonstrated strong performance in visual grounding, establishing themselves as a general interface for various vision-language applications. This progress has driven the development of token pruning methods to mitigate the high computational costs associated with processing numerous visual tokens. However, we observe that pruning significantly weakens the model’s grounding ability, leading to incorrect predictions and drastic performance degradation. In Referring Expression Comprehension (REC), for instance, pruning causes the accuracy of LLaVA on the RefCOCO val set to drop from  56.14%  to  15.34% . Our analysis identifies misaligned position IDs after pruning as the primary cause of this degradation, as both the order and value of these IDs are crucial for maintaining performance in grounding tasks. To address this issue, we propose Grounding-Aware Token Pruning ( GAP ), a simple yet effective adjustment to position IDs that recovers REC accuracy back to  51.42% , which is 90% of the original performance in the without pruning setting, all while requiring no additional training, memory, or computational overhead. Applied to models such as Shikra, MiniGPTv2, and the LLaVA series, our method consistently improves performance across various token pruning strategies.

1  Introduction

Figure 1 :

Catastrophic performance drop.  Comparison of the performance of LLaVA on the visual grounding dataset RefCOCO. Performance drops drastically after pruning, but is restored after applying our approach,  GAP , which preserves the grounding ability of MLLMs.

Multimodal Large Language Models (MLLMs) have demonstrated impressive performance across a range of tasks, such as VQA  [  13  ,

11  ,

21  ]  and image captioning. Notably, for Referring Expression Comprehension (REC)  [  16  ] , models like MiniGPTv2 and Shikra  [  14  ,

5  ]  achieve results comparable to specialized models, underscoring their capacity for dense visual understanding, such as object localization and spatial reasoning.

As the number of visual tokens generated from high-resolution images continues to grow (e.g., 2k-8k tokens)  [  20  ,

17  ] , computational costs and inefficiencies increase. High memory usage, latency, and higher FLOPs hinder scalability. To address these issues, various token pruning methods  [  22  ,

24  ,

4  ,

33  ,

29  ,

7  ]  have been developed, which selectively reduce visual tokens while optimizing computational efficiency.

However, we observe that token pruning can degrade grounding ability, impairing the model’s understanding of spatial relationships. This results in significantly poorer bounding box predictions and a dramatic performance drop in grounding tasks that require fine-grained spatial information, such as REC, as shown in

Figure

1  .
Through investigation, we found that misaligned position IDs are the primary cause of performance drops after pruning. These IDs are crucial for maintaining the spatial coherence of visual tokens, and both the order and values of these IDs must remain intact to ensure optimal model performance. Based on this, we propose  G rounding- A ware Token  P runing  (GAP) , a simple modification that corrects these misalignments and significantly improves REC performance.

We applied GAP to six pruning methods, including, PruMerge  [  22  ] , TRIM  [  24  ] , and four baseline methods on LLaVA, demonstrating its ability to improve existing pruning strategies without additional overhead. To further validate our findings, we applied GAP to five models, including Shikra, MiniGPTv2, and the LLaVA series, confirming that this is a global issue affecting various training recipes and model architectures. These findings confirm that GAP consistently improves performance while preserving efficiency, making it a viable enhancement for grounding MLLMs.

Our contributions are summarized as follows:

•

We identify misaligned position IDs as the key factor behind grounding performance degradation in token-pruned MLLMs and propose  GAP  to effectively recover lost performance.

•

GAP  aligns position IDs without additional training, memory, or computational overhead, making it an efficient and practical solution.

•

We validate  GAP  across five models and six pruning methods, demonstrating its broad applicability and effectiveness in addressing this global issue.

2  Background and Related Work

2.1  Multimodal Large Language Models (MLLMs)

The expansion of large language models into multimodal versions has gained considerable attention in recent years. Multimodal Large Language Models (MLLMs)  [  19  ,

18  ,

20  ,

14  ,

5  ]  represent a significant advancement in artificial intelligence, extending the capabilities of traditional Large Language Models (LLMs)  [  3  ,

8  ]  and traditional Vision Language Models (VLMs) to process content across multiple modalities, inclusive of text, images, audio and video.

Mainstream approaches in MLLMs use sequential visual representation, where images are encoded into vision tokens by a visual encoder  [  1  ,

31  ,

27  ]  and sent into the LLM as the prefix content. With modal alignment pretraining and instruction fine-tuning  [  19  ,

9  ] , modern MLLMs can tackle various tasks in computer vision field, from challenges like Optical Character Recognition (OCR), Visual Question Answering (VQA), and Referring Expression Comprehension (REC), to more complex, multistep reasoning tasks  [  10  ,

30  ] .

2.2  Referring Expression Comprehension

Referring Expression Comprehension (REC), also known as visual grounding  [  16  ,

28  ] , aims to localize a specific object in an image described by a referring expression phrased in natural language expressions. As a cross-modal recognition task, REC has seen recent advancements driven by MLLMs  [  5  ,

12  ,

26  ,

14  ] . Some of these models have even achieved results comparable to specialized models  [  15  ,

23  ] . These advancements highlight the ability of MLLMs to perform fine-grained object localization and spatial reasoning..

2.3  Token Reduction

MLLMs typically take in a large amount of visual tokens as the prefix content. For example, LLaVA-1.5  [  18  ]  encodes images at a resolution of 336 × 336 into 576 tokens and processes images with a greater resolution of 672 × 672 into 2304 tokens. Similarly, MiniGPTv2  [  14  ]  encodes images at 448 × 448 resolution into 256 tokens by concatenating four adjacent visual tokens. However, the quadratic complexity of Transformers presents a significant challenge when dealing with long input sequences. To address this, several works have attempted to reduce visual tokens by merging them before passing them into LLMs or pruning less important tokens from Transformer layers, thereby lowering computational costs.

PruMerge  [  22  ]  leverages the sparse distribution of attention scores between the [CLS] token and visual tokens in ViT  [  1  ]  to measure the importance of each token. It prunes visual tokens with low attention scores and merges them with the selected tokens. On average, LLaVA-1.5 with PruMerge reduces visual tokens to just 6.9% and cuts 88% of FLOPs, while maintaining comparable performance in various tasks. SparseVLM  [  33  ]  selects visual-relevant text tokens to measure the significance of visual tokens within the self-attention matrix and prunes the visual tokens accordingly. When reducing tokens from 576 to 64, LLaVA-1.5 with SparseVLM reduces 84% FLOPs with only a 13.1% drop in average a