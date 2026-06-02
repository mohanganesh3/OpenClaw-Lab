[2410.10491] Learning to Ground VLMs without Forgetting

Learning to Ground VLMs without Forgetting

Aritra Bhowmik

Mohammad Mahdi Derakhshani  1

1  footnotemark:

1

Dennis Koelma

Martin R. Oswald

Yuki M. Asano

Cees G. M. Snoek

University of Amsterdam

Shared first authorship. Corresponding authors: {a.bhowmik, m.m.derakhshani}@uva.nl

Abstract

Spatial awareness is key to enable embodied multimodal AI systems. Yet, without vast amounts of spatial supervision, current Visual Language Models (VLMs) struggle at this task. In this paper, we introduce LynX, a framework that equips pretrained VLMs with visual grounding ability without forgetting their existing image and language understanding skills.
To this end, we propose a  Dual Mixture of Experts  module that modifies only the decoder layer of the language model, using one frozen Mixture of Experts (MoE) pre-trained on image and language understanding and another learnable MoE for new grounding capabilities. This allows the VLM to retain previously learned knowledge and skills, while acquiring what is missing.
To train the model effectively, we generate a high-quality synthetic dataset we call SCouT, which mimics human reasoning in visual grounding. This dataset provides rich supervision signals, describing a step-by-step multimodal reasoning process, thereby simplifying the task of visual grounding. We evaluate LynX on several object detection and visual grounding datasets, demonstrating strong performance in object detection, zero-shot localization and grounded reasoning while maintaining its original image and language understanding capabilities on seven standard benchmark datasets.

1  Introduction

Visual language models (VLMs) have significantly advanced multimodal vision and language tasks, enabling impressive capabilities such as image captioning and visual question answering  (Alayrac et al.,  2022 ; Li et al.,  2023 ; Dai et al.,  2024 ; Liu et al.,  2024 ) . Models like CLIP  (Radford et al.,  2021a )  leveraged extensive image-caption data for multimodal training, while generative models like Flamingo  (Alayrac et al.,  2022 )  and BLIP2  (Li et al.,  2023 )  generate descriptive captions for images. Because of their caption-based nature, these models often lack object localization abilities, making them less suited for applications requiring precise spatial understanding  (Wen et al.,  2023 ; Luo et al.,  2024 ; Driess et al.,  2023 ; Jin et al.,  2023 ; Cheng et al.,  2024 ) . Naturally, one can equip a model with localization ability by pre-training,  (Wang et al.,  2023 ; Chen et al.,  2023b ) . However, this requires massive datasets, human-annotated bounding boxes, and substantial computational resources, making it costly and impractical for smaller setups. Rather than pre-training from scratch, we aim to equip a pre-trained VLM with spatial understanding by fine-tuning.

Closest to our work is PIN  (Dorkenwald et al.,  2024 ) , which fine-tunes a VLM for the specific task of object localization by adding learned spatial parameters to the vision encoder. Trained on a synthetic dataset of superimposed object renderings, PIN is evaluated on single-object localization, predicting a bounding box given a query object name. Despite the obtained ability for object localization, PIN suffers from catastrophic forgetting, losing image understanding abilities after fine-tuning. Moreover, its synthetic data lacks inter-object relationships, limiting its utility for more complex tasks beyond object localization, like multi-object detection and reasoning  (Wang et al.,  2023 ; Chen et al.,  2023b ) . Additionally, as demonstrated by PIN, even parameter-efficient methods like LoRA  (Hu et al.,  2021 )  underfit due to the complexity differences between image understanding and grounding tasks. These challenges underscore the need for a solution that adds grounding capabilities for many tasks without compromising a model’s pre-existing strengths, which is the focus of our work.

Specifically, we introduce LynX ( Lin king e X perts for visual grounding), a novel framework that leverages a Dual Mixture of Experts (MoE) architecture. This design allows the model to specialize in both image understanding and visual grounding simultaneously, preventing the catastrophic forgetting seen in PIN and enabling fine-tuning for grounding without sacrificing existing capabilities. Further, to address the shortcomings of PIN’s localization-only dataset, we propose SCouT: Synthetic Chain-of-Thought with Grounding, a high-quality synthetic dataset with step-by-step grounded chain-of-thought annotations. Unlike PIN’s object-pasting approach, SCouT captures meaningful spatial relationships and reasoning steps, providing a richer training signal for grounding tasks. We complement this with a step-by-step training methodology inspired by  Lightman et al. ( 2023 ) , breaking down tasks into intermediate steps with individual loss functions, providing clearer learning signals for handling complex multimodal tasks. Additionally, to address the challenge of evaluating VLMs with free-form grounded responses—where existing metrics fall short—we propose an open-source evaluation metric to fairly compare performance.

Our contributions can be summarized as follows:

1.

We introduce LynX, a Dual Mixture of Experts framework that enables VLMs to acquire new grounding capabilities via fine-tuning without forgetting pre-trained skills (Figure

1  .a).

2.

We present SCouT, a high-quality synthetic dataset with step-by-step grounded chain-of-thought annotations, specifically designed to facilitate effective training of VLMs on grounding tasks (Figure

1  .b).

3.

We propose a step-by-step training methodology, breaking down tasks into intermediate steps with individual loss functions, improving model performance by scaling the generated dataset (Figure

1  .c).

4.

We introduce an open-source evaluation pipeline for VLMs on object detection tasks, accommodating their open-ended generative nature.

Figure 1:

Contributions overview.  Our contributions include  (a)  enabling a pre-trained caption-based vision-language model to learn new grounding skills by fine-tuning without forgetting old ones,  (b)  improving model performance by scaling the generated dataset, and  (c)  enabling visual grounding tasks through step-by-step training on our synthetic dataset.

2  Related Work

Caption-based Visual Language Models (VLMs). 
Large language models, efficient at instruction following and generalization, have been seamlessly integrated with vision-only encoder models, yielding impressive results in multimodal tasks  (Alayrac et al.,  2022 ; Li et al.,  2023 ; Bai et al.,  2023 ; Chen et al.,  2023a ; Wang et al.,  2023 ; Chen et al.,  2023d ; Zhang et al.,  2023a ; Lin et al.,  2023 ; Cha et al.,  2023 ; Dai et al.,  2024 ; Ye et al.,  2023 ; Zhao et al.,  2023 ; Chen et al.,  2023c ; Liu et al.,  2023 ; Zhang et al.,  2023b ) .
Flamingo  (Alayrac et al.,  2022 )  and BLIP-2  (Li et al.,  2023 )  are pioneering works in this area. Flamingo combines a pretrained CLIP  (Radford et al.,  2021b )  image encoder with a pretrained LLM using perceiver and gated cross-attention blocks, while BLIP-2 employs a lightweight Querying Transformer, pretrained in two stages: vision-language representation from a frozen image encoder and vision-to-language generation from a frozen language model.
Subsequently, recent works have focused on improving performance through optimizing training strategies  (Bai et al.,  2023 ; Chen et al.,  2023a ) , increasing resolution of image  (Bai et al.,  2023 ; Chen et al.,  2023a ; Wang et al.,  2023 ) , enhancing image encoders  (Chen et al.,  2023d ; Zhang et al.,  2023a ; Bai et al.,  2023 ) , aligning the input  (Lin et al.,  2023 )  and projection layers  (Cha et al.,  2023 ; Alayrac et al.,  2022 ; Dai et al.,  2024 ; Ye et al.,  2023 ; Zhao et al.,  2023 ; Chen et al.,  2023c ) .
More importantly, man