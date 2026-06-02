[2503.15426] Visual Position Prompt for MLLM based Visual Grounding

Visual Position Prompt for MLLM based Visual Grounding

Wei Tang † ,
Yanpeng Sun,
Qinying Gu,
and Zechao Li

W. Tang, Y. Sun and Z. Li are with School of Computer Science and Engineering, Nanjing University of Science and Technology, No.200 Xiaolingwei Road, Nanjing 210094, China. E-mail: {weitang, zechao.li and yanpeng_sun}@njust.edu.cn (Corresponding authors: Zechao Li and Qinying Gu)
Q. Gu are with Shanghai Artificial Intelligence Laboratory, No.129 Longwen Road, Shanghai 200232, China. E-mail: guqinying@pjlab.org.cn.
 †  This work was done during his internship at Shanghai Artificial Intelliaence Laboratory.

Abstract

Although Multimodal Large Language Models (MLLMs) excel at various image-related tasks, they encounter challenges in precisely aligning coordinates with spatial information within images, particularly in position-aware tasks such as visual grounding. This limitation arises from two key factors. First, MLLMs lack explicit spatial references, making it difficult to associate textual descriptions with precise image locations. Second, their feature extraction processes prioritize global context over fine-grained spatial details, leading to weak localization capability.
To address this issue, we introduce VPP-LLaVA, an MLLM equipped with Visual Position Prompt (VPP) to improve its grounding capability. VPP-LLaVA integrates two complementary mechanisms. The global VPP overlays learnable, axis-like embeddings onto the input image to provide structured spatial cues. The local VPP focuses on fine-grained localization by incorporating position-aware queries, which suggests probable object locations.
We also introduce a VPP-SFT dataset with 0.6M samples, consolidating high-quality visual grounding data into a compact format for efficient model training. Training on this dataset with VPP enhances the model’s performance, achieving state-of-the-art results on standard grounding benchmarks despite using fewer training samples compared to other MLLMs like MiniGPT-v2, which rely on much larger datasets (

∼  \sim

21M samples). The code and VPP-SFT dataset will be available at https://github.com/WayneTomas/VPP-LLaVA upon acceptance.

Index Terms:

Multimodal large language model, Visual grounding, Visual prompt, Prompt learning.

I

Introduction

Figure 1:  A visual grounding case study of MLLMs: (a) LLaVA-v1.5 outputs an inaccurate bounding box based on the given query expression. (b) When provided with a position reference, VPP-LLaVA produces a suitable result. For brevity, some text instructions are omitted.

Figure 2:  An illustration of VPP-LLaVA, an MLLM-based visual grounding framework with Visual Position Prompt (VPP). We utilize the global VPP to provide a global position reference for MLLMs with foundational spatial cues. Additionally, a local VPP, serving as a local position reference, is introduced to further enhance and incorporate object spatial information. For brevity, some text instructions are omitted.

Multi-modal Large Language Models (MLLMs)  [ 1 ,  2 ,  3 ,  4 ,  5 ]  achieve impressive results across various image-related tasks  [ 6 ,  7 ,  8 ,  9 ] ,
earning considerable attention from the research community. Among these tasks, visual grounding—specifically Referring Expression Comprehension (REC)—stands out as a critical challenge  [ 10 ,  11 ,  12 ,  7 ,  13 ] . Unlike pure detection tasks  [ 14 ,  15 ,  16 ] , visual grounding involves precisely identifying locations within an image based on free-form language expressions. It is fundamental for cognitive interactions between humans and machines, with applications such as image segmentation  [ 17 ] , remote sensing  [ 18 ]  and human-robot interaction  [ 19 ] .

While research  [ 20 ,  21 ]  indicates that MLLMs possess a reasonable ability for spatial understanding, there is a growing consensus that these models still require further enhancement, particularly for tasks involving precise spatial reasoning like visual grounding. As illustrated in Fig.

1

(a), the grounding results from LLaVA-v1.5, for instance, reveal notable inaccuracies (the red box indicates the predicted bounding box, while the green box represents the ground truth). Although the predicted bounding box partially covers the target object, i.e., the brown toy, it suffers from both size and shape inaccuracies, failing to align well with the object’s true boundaries. These limitations highlight the need for more effective strategies to improve spatial alignment and object localization in MLLMs. To address these challenges, some studies are investigating the integration of advanced region-level enhancement modules and larger, more comprehensive visual grounding datasets into MLLMs  [ 13 ,  22 ,  23 ,  6 ,  24 ] . Other approaches are exploring the incorporation of task-specific expertise, such as converting special tokens directly into bounding boxes with additional decoder, to improve localization accuracy  [ 25 ,  26 ] .

However, despite the use of the aforementioned methods to enhance the performance of MLLM in grounding tasks, studies suggest that MLLMs still face significant challenges in precisely aligning coordinates with spatial information in images  [ 27 ,  28 ] . One key issue lies in the models’ ability to effectively interpret and utilize spatial cues, which remain underutilized in many cases. As shown in Fig.

1

(b), when we introduce a positional reference in the form of a coordinate axis, providing an explicit spatial guide, the model’s understanding of spatial relationships improves significantly. This reference allows the model to better interpret the spatial information of objects in the image, leading to more accurate localization. Specifically, the predicted bounding box becomes more aligned with the brown toy, reflecting the improved spatial reasoning and localization accuracy when positional references are incorporated.

Based on these observations, we propose integrating positional references as prompts within MLLMs to improve their visual grounding capabilities. While the coordinate axis shown in Fig.

1

(b) provides a global spatial guide, we also explore the potential of leveraging object position embeddings, derived from detection models, which offer local spatial cues by capturing object locations and semantic context. These two types of Visual Position Prompt (VPP)—global and local—are complementary, with the global guide helping to establish overall spatial structure, and the local cues refining object-specific localization.

Specifically, the global VPP is initialized in an axis-like form and overlaid onto the input image, providing a global spatial reference. This enables MLLMs to more effectively align coordinate information with spatial details across the image. To capture object-specific spatial and semantic information, we introduce a local VPP, which identifies potential objects within the image. This local reference helps the decoder integrate object-level details with other features. By combining both global and local prompts, our model improves spatial alignment and boosts performance in visual grounding tasks. Notably, we achieve state-of-the-art results with a relatively small training set of about 0.6M samples from our new VPP-SFT dataset, which consolidates high-quality visual grounding data into a compact form. Compared to other MLLMs like MiniGPT-v2, which require around

∼  \sim

21M grounding samples, our approach demonstrates superior efficiency while maintaining strong performance.

In summary, our contributions are shown as follows:

•

We propose VPP-LLaVA, an MLLM-based method for visual grounding with a Visual Position Prompt, along with a high-quality grounding instruction tuning dataset, VPP-SFT, which contains approximately 0.6M samples.

•

We introduce novel global and local Visual Position Prompts that enable MLLMs to more effectively link spatial information within image