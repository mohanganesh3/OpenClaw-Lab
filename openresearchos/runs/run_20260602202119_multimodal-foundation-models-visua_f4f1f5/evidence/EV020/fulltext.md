[2508.02890] VisuCraft: Enhancing Large Vision-Language Models for Complex Visual-Guided Creative Content Generation via Structured Information Extraction

VisuCraft: Enhancing Large Vision-Language Models for Complex Visual-Guided Creative Content Generation via Structured Information Extraction

Rongxin Jiang 1 , Robert Long 2 , Chenghao Gu 1 , Mingrui Yan 1

1 Heilongjiang University of Science and Technology,  2 University of Padua

Abstract

This paper introduces VisuCraft, a novel framework designed to significantly enhance the capabilities of Large Vision-Language Models (LVLMs) in complex visual-guided creative content generation. Existing LVLMs often exhibit limitations in maintaining high visual fidelity, genuine creativity, and precise adherence to nuanced user instructions when generating long-form texts. VisuCraft addresses these challenges by integrating a multimodal structured information extractor (

ℰ  \mathcal{E}

) and a dynamic prompt generation module (

𝒢  \mathcal{G}

). The extractor distills fine-grained visual attributes from input images into a rich, structured representation, which the dynamic prompt module then combines with user instructions to create highly optimized prompts for underlying LVLMs (e.g., LLaVA, InstructBLIP). Evaluated on the self-constructed ImageStoryGen-500K dataset using VisuGen Metrics (Visual Grounding, Creativity, and Instruction Adherence), VisuCraft consistently outperforms baseline LVLMs across tasks like story generation and poetry composition. Our results demonstrate remarkable improvements, particularly in creativity and instruction adherence, validating VisuCraft’s effectiveness in producing imaginative, visually grounded, and user-aligned long-form creative text. This work unlocks new potential for LVLMs in sophisticated creative AI applications.

I

Introduction

The rapid advancements in large language models (LLMs) have revolutionized various text-based applications, and their integration with visual modalities has led to the emergence of powerful Large Vision-Language Models (LVLMs)  [ 1 ,  2 ,  3 ] . These models possess remarkable capabilities in understanding and generating content from multimodal inputs, extending their utility to complex creative tasks such as story generation, poetry composition, and marketing copy creation. The demand for artificial intelligence systems capable of producing highly imaginative, contextually rich, and visually grounded content is growing, positioning LVLMs as a cornerstone for future creative applications  [ 4 ] .

Despite their impressive progress, existing LVLMs still face significant challenges when tasked with complex visual-guided creative content generation. Current models often struggle with generating long-form text that maintains high visual fidelity, exhibits genuine creativity, or precisely adheres to nuanced user instructions. Common limitations include a tendency towards generic or repetitive outputs, insufficient correlation between the generated text and fine-grained visual details, and a lack of generalization to diverse creative prompts  [ 5 ] . Works exploring the nuances of visual dependency and long-context reasoning in LVLMs  [ 3 ]  and efforts to improve zero-shot learning and instruction adherence  [ 6 ]  highlight these ongoing challenges. These issues stem from the difficulty in extracting sufficiently structured and semantically rich visual information to guide the language generation process effectively. Our motivation is to overcome these limitations by enabling LVLMs to better leverage intricate visual cues, thereby unlocking their full potential for truly creative and contextually relevant content generation.

In this paper, we propose  VisuCraft , a novel framework designed to enhance the creative content generation capabilities of existing pre-trained LVLMs under complex visual guidance. VisuCraft is not a standalone large model but rather an enhancement framework that integrates seamlessly with off-the-shelf LVLMs such as LLaVA  [ 7 ]  and InstructBLIP  [ 7 ] . The core of VisuCraft comprises two key components: a  multimodal structured information extractor  and a  dynamic prompt generation module . The structured information extractor processes input images to distill fine-grained visual attributes, including object poses, material properties, lighting conditions, and even emotional atmospheres, into a structured textual or JSON format. This rich, structured visual representation is then combined with the user’s textual instructions by the dynamic prompt generation module to construct highly optimized and informative prompts. These enhanced prompts subsequently guide the underlying LVLM to produce long-form text that is not only highly relevant to the visual content but also demonstrates superior creativity and strict adherence to user specifications.

To validate the efficacy of VisuCraft, we utilize a diverse set of datasets for both training and evaluation. The multimodal structured information extractor module within VisuCraft is trained or fine-tuned using extensive image datasets such as ImageNet  [ 8 ] , COCO  [ 9 ] , and OpenImages  [ 10 ] , augmented with fine-grained visual annotations like scene graphs, detailed attribute labels, and sentiment tags. For evaluation, we employ a self-constructed benchmark dataset,  ImageStoryGen-500K , which comprises a vast collection of diverse images paired with complex and nuanced creative generation instructions (e.g., ”Write a poem about loneliness based on this image,” or ”Compose a short story describing the internal struggles of the character in the picture”).

Our evaluation methodology employs a set of custom metrics, collectively termed  VisuGen Metrics , which include Visual Grounding (VG.), Creativity (C.), and Instruction Adherence (IA.), along with a Mean score. Visual Grounding assesses how well the generated text matches and reflects the details of the image content. Creativity measures the uniqueness, imagination, and novelty of the generated output. Instruction Adherence quantifies the degree to which the generated content follows the user’s specific textual directives. Our experimental results demonstrate that VisuCraft consistently outperforms existing baseline LVLM models (LVLM-Base and LVLM-Enhanced) across various creative generation tasks, including story generation and poetry composition. Notably, VisuCraft achieves significant improvements in both Creativity and Instruction Adherence, indicating its superior ability to understand complex instructions and generate more imaginative and user-aligned content. For instance, in StoryGen, VisuCraft achieved VG. of 0.825, C. of 0.810, and IA. of 0.830, leading to a Mean score of 0.822, surpassing LVLM-Enhanced’s 0.811. Similarly, in Poetry generation, VisuCraft scored 0.810 on VG., 0.805 on C., and 0.815 on IA., with a Mean of 0.810, outperforming LVLM-Enhanced’s 0.794. These results unequivocally demonstrate VisuCraft’s effectiveness in elevating the quality of long-form creative text generation under intricate visual guidance, yielding content that is both faithful to the image and highly imaginative.

Our main contributions can be summarized as follows:

•

We propose  VisuCraft , a novel enhancement framework that significantly improves the creative long-form text generation capabilities of existing LVLMs under complex visual guidance.

•

We introduce a  multimodal structured information extractor  within VisuCraft, capable of distilling fine-grained visual attributes into structured representations, thus enabling more precise visual grounding for text generation.

•

We demonstrate through extensive experiments that VisuCraft consistently outperforms state-of-the-art baseline LVLMs across key metrics (Visual Grounding, Creativity, and Instruction Adherence) on challenging creative tasks, particularly showcasing remarkable gains in creativity and instruction adherence.

II

Related Work

II-A