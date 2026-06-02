[2604.21396] VG-CoT: Towards Trustworthy Visual Reasoning via Grounded Chain-of-Thought

VG-CoT: Towards Trustworthy Visual Reasoning via Grounded Chain-of-Thought

Abstract

The advancement of Large Vision-Language Models (LVLMs) requires precise local region-based reasoning that faithfully grounds the model’s logic in actual visual evidence. However, existing datasets face limitations in scalability due to extensive manual annotation and lack of explicit alignment between multi-step reasoning and corresponding image regions, which constrains the evaluation of model trustworthiness. To address these challenges, we propose the Visual Grounding Chain-of-Thought (VG-CoT) dataset, which explicitly links each reasoning step to real visual evidence within the image through a fully automated three-stage pipeline. The pipeline first extracts object- and text-level visual evidence using state-of-the-art detection and OCR models, then generates step-by-step grounded reasoning with GPT-4o, and finally refines the grounding through a rationale-driven open-set detection process. In addition, we introduce a new benchmark that comprehensively evaluates LVLMs reasoning across three complementary dimensions: Rationale Quality, Answer Accuracy, and Reasoning-Answer Alignment. Experiments with representative LVLMs, including LLaVA-1.5 and Qwen2-VL, demonstrate consistent improvements on most evaluation metrics, confirming that VG-CoT effectively enhances trustworthy, evidence-based reasoning while maintaining scalable and cost-efficient dataset construction. The dataset and code will be released publicly upon acceptance to facilitate further research.

Keywords:  LVLMs Reasoning, Visual Grounded Rationale Dataset, Reasoning Evaluation

\NAT@set@cites

VG-CoT: Towards Trustworthy Visual Reasoning via Grounded Chain-of-Thought

Byeonggeuk Lim 1 , Kyeonghyun Kim 2 , JungMin Yun 2  and YoungBin Kim 1, 2

Graduate School of Advanced Imaging Science, Multimedia &amp; Film, Chung-Ang University 1

Department of Artificial Intelligence, Chung-Ang University 2

{banggeuk, khyun8072, cocoro357, ybkim85}@cau.ac.kr

Abstract content

1.  Introduction

The development of Large Language Models (LLMs) has recently led to the rise of Large Vision-Language Models (LVLMs), which simultaneously understand visual and linguistic information

1_zhang2024vision ;  2_yin2024survey  . LVLMs have demonstrated outstanding performance in comprehensive, image-level understanding, and the focus of research is gradually expanding toward the ability for precise understanding of local regions within an image

3_NEURIPS2023_9a6a435e ;  24_Liu_2024_CVPR ;  25_zhuminigpt ;  26_wang2024qwen2  . Specifically, the ability to accurately identify specific regions within an image and interpret their spatial and semantic relationships is emerging as a core requirement for performing advanced vision-language tasks such as autonomous driving, robotics, and medical image analysis

4_li-etal-2024-topviewrs ;  5_pmlr-v229-zitkovich23a ;  6_NEURIPS2023_5abcdf8e  .

However, this local region-based reasoning ability is highly dependent on the quality and composition of the training dataset

36_wu2023multimodal ;  37_chang2024sur  . Currently, most widely used vision-language datasets primarily focus on evaluating comprehensive, image-level understanding, which limits models from learning fine-grained reasoning based on the detailed attributes of individual objects or the relationships between objects

7_chen-etal-2024-measuring ;  8_zhou2025perception ;  13_Antol_2015_ICCV ;  14_Hudson_2019_CVPR  . Although some studies have proposed datasets that include local region information, they still rely on large-scale manual annotation, making it difficult to avoid the constraints of cost and scalability

27_yu2016modeling ;  28_Zellers_2019_CVPR ;  15_krishna2017visual  . This manual construction method presents a more pronounced limitation, especially in tasks that require complex visual relationships or multi-step reasoning. This suggests the need for a dataset that can support the learning of complex, local region-based reasoning in a scalable and efficient manner.

Existing multimodal Chain-of-Thought (CoT) datasets designed to support multi-step reasoning suffer from a structural limitation: the reasoning process is not explicitly linked to specific visual evidence within the actual image

16_NEURIPS2022_11332b6b ;  27_yu2016modeling  . In other words, because the location information (spatial coordinates) of the objects or text within the image that serve as the rationale for the inference is not provided, it is difficult to verify which areas of the image the model’s reasoning was based on. This absence of visual evidence leads to a structural limitation in model evaluation regarding whether the model derived the correct answer using the right evidence as its rationale, causing existing benchmarks to only assess the accuracy of the final answer

36_wu2023multimodal ;  37_chang2024sur  . Therefore, a dataset where the entire process of reasoning is explicitly aligned with real visual evidence is essential for verifying whether the logic presented by the model is faithful to the actual image evidence

37_chang2024sur ;  38_qiu-etal-2024-valor  . This also provides the foundation for a new benchmark capable of systematically measuring not only the accuracy of the final answer but also the accuracy of the rationale and the logical soundness of the reasoning process.

Based on this necessity, this study proposes the Visual Grounding Chain-of-Thought (VG-CoT) dataset. VG-CoT is a dataset constructed by explicitly aligning visual evidence from an image with the reasoning process across various tasks, achieving both efficiency and scalability through an automated three-stage pipeline. (i) First, initial visual evidence is extracted using state-of-the-art object detection and OCR models. (ii) Based on this, step-by-step reasoning is generated using GPT-4o. (iii) Finally, using the generated reasoning process as a clue, visual evidence is more precisely captured through a text-based object detection model. This automated pipeline effectively solves the cost issue associated with manual annotation while contributing to the improvement of the quality of the reasoning process and the reliability of rationale alignment.

Furthermore, this study proposes a new benchmark that goes beyond the existing evaluation methods centered on answer accuracy. This benchmark comprehensively measures Rationale Quality, Answer Accuracy, and Reasoning-Answer Alignment, which represents the relationship between the two metrics. Specifically, it utilizes GPT-4o

11_achiam2023gpt

as an evaluator to precisely assess the reasoning process using detailed metrics: logical coherence, reasoning completeness, and visual evidence utilization. This allows for an in-depth verification of whether the model has reasoned logically based on credible visual evidence, going beyond simply checking if the model provided the correct answer.

The key contributions of this study are summarized as follows:

1.

We propose the VG-CoT dataset, which explicitly aligns the reasoning process with real visual evidence within the image through an automated three-stage pipeline.

2.

We establish a new benchmark that comprehensively measures Rationale Quality and Reasoning–Answer Alignment, going beyond simple answer accuracy.

3.

We evaluate representative LVLMs using VG-CoT to analyze their capability in utilizing visual evidence, providing insights into the future direction of LVLMs research.

2.  Related Work

Figure 1:  Examples of the Proposed VG-CoT Dataset across Three Task Types.

2.1.  Datasets for LVLMs Reasoning

Early datasets for LVLMs primarily focused on image-wide understanding, which limited their ability to train models for fine-grained, local region-based reasoning

12_lin2014microsoft ;  13_Antol_2015_ICCV ;  14_Hudson_2019_CVPR  . To overcome this limi