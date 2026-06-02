[2405.20421] Worse than Random? An Embarrassingly Simple Probing Evaluation of Large Multimodal Models in Medical VQA

Worse than Random?

An Embarrassingly Simple Probing Evaluation of Large Multimodal Models in Medical VQA

Qianqi Yan

University of California, Santa Cruz

qyan79@ucsc.edu

Xuehai He

University of California, Santa Cruz

xhe89@ucsc.edu

Xiang Yue

Carnegie Mellon University

xyue2@andrew.cmu.edu

Xin Eric Wang

University of California, Santa Cruz

xwang366@ucsc.edu

Abstract

Large Multimodal Models (LMMs) have shown remarkable progress in the field of medical Visual Question Answering (Med-VQA), achieving high accuracy on existing benchmarks. However, their reliability under robust evaluation is questionable. This study reveals that state-of-the-art models, when subjected to simple probing evaluation, perform worse than random guessing on medical diagnosis questions.
To address this critical evaluation problem, we introduce the Probing Evaluation for Medical Diagnosis (ProbMed) dataset to rigorously assess LMM performance in medical imaging through  probing evaluation  and  procedural diagnosis .
Particularly, probing evaluation features pairing original questions with negation questions with hallucinated attributes, while procedural diagnosis requires reasoning across various diagnostic dimensions for each image, including modality recognition, organ identification, clinical findings, abnormalities, and positional grounding.
Our evaluation reveals that top-performing models like GPT-4V and Gemini Pro perform worse than random guessing on specialized diagnostic questions, indicating significant limitations in handling fine-grained medical inquiries. Besides, models like LLaVA-Med struggle even with more general questions, and results from CheXagent demonstrate the transferability of expertise across different modalities of the same organ, showing that specialized domain knowledge is still crucial for improving performance.
This study underscores the urgent need for more robust evaluation to ensure the reliability of LMMs in critical fields like medical diagnosis, and current LMMs are still far from applicable to those fields.
 https://github.com/eric-ai-lab/ProbMed

Figure 1:  Accuracy of four LMMs on two types of specialized questions in medical diagnoses, with and without adversarial pairs.
The significant drop in accuracy with adversarial pairs highlights the models’ unreliability in handling medical diagnoses.

1  Introduction

Foundation models, such as large language models (LLMs)  (OpenAI,  2023a ; Touvron et al.,  2023 ; Jiang et al.,  2023 ; Anil et al.,  2023 ; Chung et al.,  2022 )  and large multimodal models (LMMs)  (OpenAI,  2023b ; Team et al.,  2023 ; Li et al.,  2023b ; Liu et al.,  2023a ; Chen et al.,  2023a ) , have demonstrated impressive capabilities in understanding complex visual and text inputs, generating human-like language, and achieving high accuracy on various benchmarks.
The integration of these foundation models into real-life medical practice holds immense potential given their advanced computational capabilities  (Wu et al.,  2023 ; Yang et al.,  2023 )  and promising progress on existing medical Visual Question Answering (Med-VQA) benchmarks  (Lau et al.,  2018 ; Liu et al.,  2021 ; He et al.,  2020 ; Zhang et al.,  2023 ) . As we stand on the precipice of integrating these models into critical decision-making domains, one natural question appears:  how much can we trust these models in real-world scenarios, such as medicine and healthcare, where the stakes are high?

Before discussing the reliability of LMMs in critical domains like Med-VQA, we must first address a fundamental question:  Are we evaluating LMMs correctly? 
To address this question, we introduce a simple yet effective probing evaluation method that exposes the weaknesses of LMMs by creating simple binary questions with hallucination pairs over existing benchmarks. An example is shown in Figure

2  .
Despite the high accuracy reported on current Med-VQA tasks, our study reveals a significant vulnerability in LMMs when faced with adversarial questioning, as illustrated in Figure

1  .
The performance drops observed are alarming: even advanced models like GPT-4V and Gemini Pro perform worse than random guessing, with an average decrease of 42% across the tested models.

Figure 2:  An example illustrating the potential for misleading accuracy in existing evaluations. While the model correctly identifies the position of an existing finding in the standard evaluation, it fails to differentiate between actual and hallucinated positions when subjected to an adversarial evaluation.

Based on this, we further analyze a critical question:  How reliable are LMMs in medical diagnosis, ranging from general questions to specialized diagnostic questions ? To address this question, we introduce ProbMed, which features procedural diagnosis designed to rigorously evaluate model performance across multiple diagnostic dimensions. We curated ProbMed from 6,303 images sourced from two widely-used biomedical datasets, MedICaT  (Subramanian et al.,  2020 )  and ChestX-ray14  (Wang et al.,  2017 ) . These images cover various modalities, including X-ray, MRI, and CT scans, and span multiple organs such as the abdomen, brain, chest, and spine. Using GPT-4 and a positional reasoning module, we generated metadata for each image, extracting information about abnormalities, condition names, and their corresponding locations. This metadata facilitated the automatic generation of 57,132 high-quality question-answer pairs, covering dimensions like modality recognition, organ identification, abnormalities, clinical findings, and positional reasoning.

Our systematic evaluation of seven state-of-the-art LMMs on ProbMed revealed several critical insights.  First , even the best-performing models, such as GPT-4V and Gemini Pro, performed close to random guessing on specialized diagnostic categories like Condition/Finding and Position, highlighting their limitations in handling fine-grained medical inquiries.  Second , introducing adversarial pairs significantly reduced the accuracy of all models, with LLaVA-Med’s performance dropping by up to 57% and Gemini Pro’s accuracy decreasing by 25% in ProbMed. These findings emphasize the importance of adversarial testing in Med-VQA to uncover model weaknesses.  Third , the CheXagent model, which was exclusively trained on chest X-rays, demonstrated that specialized domain knowledge is crucial. It showed that expertise gained on one particular organ could be transferable to another modality of the same organ in a zero-shot manner, highlighting the value of domain-specific training for improving model performance.

In summary, our work highlights significant gaps in the reliability of LMMs for medical diagnosis despite their impressive performance on current existing general domain benchmarks. The insights from ProbMed underscore the urgent need for robust evaluation methodologies to ensure the accuracy and reliability of LMMs in real-world medical applications. This research inspires the development of more trustworthy AI systems in healthcare and beyond, ultimately contributing to better diagnostic outcomes and patient care.

2  Related Work

2.1  Large Multimodal Models in the Medical Domain

The advancements in Large Multimodal Models (LMMs) have significantly enhanced the understanding and generation of medical content that integrates both visual and linguistic elements. Notable models include GPT-4V  (OpenAI,  2023b ) , Gemini Pro  (Team et al.,  2023 ) , LLaVA  (Liu et al.,  2023a ,  b ) , and MiniGPT-v2  (Chen et al.,  2023a ) . The scalability and exceptional performance of these large foundation models have driven their application in the biomedical field.

Further progress has been made in fine-tuning general-domain LMMs for the biomedical field, resulting in specialized models like LLaVA-Med  (Li et al.,  2023a