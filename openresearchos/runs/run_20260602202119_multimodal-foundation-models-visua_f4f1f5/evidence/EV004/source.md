# EV004: Worse than Random? An Embarrassingly Simple Probing Evaluation of Large Multimodal Models in Medical VQA

URL: https://www.semanticscholar.org/paper/0911b452394eeabf46114df37d942d2048b96f83
Year: 2024
Source: semantic_scholar
Arxiv: 2405.20421

## Abstract

Large Multimodal Models (LMMs) have shown remarkable progress in medical Visual Question Answering (Med-VQA), achieving high accuracy on existing benchmarks. However, their reliability under robust evaluation is questionable. This study reveals that when subjected to simple probing evaluation, state-of-the-art models perform worse than random guessing on medical diagnosis questions. To address this critical evaluation problem, we introduce the Probing Evaluation for Medical Diagnosis (ProbMed) dataset to rigorously assess LMM performance in medical imaging through probing evaluation and procedural diagnosis. Particularly, probing evaluation features pairing original questions with negation questions with hallucinated attributes, while procedural diagnosis requires reasoning across various diagnostic dimensions for each image, including modality recognition, organ identification, clinical findings, abnormalities, and positional grounding. Our evaluation reveals that top-performing models like GPT-4o, GPT-4V, and Gemini Pro perform worse than random guessing on specialized diagnostic questions, indicating significant limitations in handling fine-grained medical inquiries. Besides, models like LLaVA-Med struggle even with more general questions, and results from CheXagent demonstrate the transferability of expertise across different modalities of the same organ, showing that specialized domain knowledge is still crucial for improving performance. This study underscores the urgent need for more robust evaluation to ensure the reliability of LMMs in critical fields like medical diagnosis, and current LMMs are still far from applicable to those fields.
