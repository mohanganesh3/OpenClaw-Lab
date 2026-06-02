[2411.09213] Comprehensive and Practical Evaluation of Retrieval-Augmented Generation Systems for Medical Question Answering

Comprehensive and Practical Evaluation of Retrieval-Augmented Generation Systems for Medical Question Answering

Nghia Trung Ngo 1 ,
Chien Van Nguyen 1 ,
Franck Dernoncourt 2 ,
Thien Huu Nguyen  1

Abstract

Retrieval-augmented generation (RAG) has emerged as a promising approach to enhance the performance of large language models (LLMs) in knowledge-intensive tasks such as those from medical domain.
However, the sensitive nature of the medical domain necessitates a completely accurate and trustworthy system.
While existing RAG benchmarks primarily focus on the standard retrieve-answer setting, they overlook many practical scenarios that measure crucial aspects of a reliable medical system.
This paper addresses this gap by providing a comprehensive evaluation framework for medical question-answering (QA) systems in a RAG setting for these situations, including sufficiency, integration, and robustness.
We introduce Medical Retrieval-Augmented Generation Benchmark (MedRGB) that provides various supplementary elements to four medical QA datasets for testing LLMs’ ability to handle these specific scenarios.
Utilizing MedRGB, we conduct extensive evaluations of both state-of-the-art commercial LLMs and open-source models across multiple retrieval conditions.
Our experimental results reveals current models’ limited ability to handle noise and misinformation in the retrieved documents.
We further analyze the LLMs’ reasoning processes to provides valuable insights and future directions for developing RAG systems in this critical medical domain.

Introduction

Large language models (LLMs) have demonstrated remarkable capabilities in solving complex medical problems, achieving state-of-the-art performance across various benchmarks.
However, ensuring the reliability and truthworthiness of an artificial intelligent (AI) medical system remains a critical challenge, especially in healthcare applications.
Retrieval-augmented generation (RAG) has emerged as a promising approach to reduce LLMs’ hallucination problem by integrating external knowledge sources.

Figure 1:  Blue texts are useful information that should be extract to help determine the answer. Red texts are factual errors that potentially mislead the LLMs.

While RAG has potential to improve the factual accuracy of LLMs’ response, incorporating an information retriever also presents new complexities that warrant careful evaluation.
Consider the example in Fig.

1  . The retrieved documents can contain not only useful knowledge that helps determine the true answer, but also noise information, or more serious, factual errors that can misleads the LLMs.
To consciously apply RAG for medical QA, we must consider these practical scenarios and evaluate LLMs ability to interact with retrieved documents reliably.

Recent efforts have been made to evaluate AI systems with LLMs in the medical domain  (Nori et al.  2023 ; He et al.  2023 ; Xiong et al.  2024 ) .
For example, MedEval  (He et al.  2023 )  presents a large-scale, expert-annotated benchmark that cover various medical tasks and domains.
 (Xiong et al.  2024 )  evaluates RAG extensively based on their MIRAGE benchmark which cover 5 medical QA datasets.
However, they only focus on the effect of RAG modules on target accuracy, missing other important aspects of a AI medical system.

Several recent works have explore RAG evaluation more comprehensively in general domain  (Es et al.  2023 ; Chen et al.  2024b ) ,
RAGAS  (Es et al.  2023 )  assess 3 qualities of RAG’s outputs for QA tasks including:
Faithfulness - degree to which responses align with the provided context,
Answer Relevance - the extent to which generated responses address the actual question posed, and
Context Precision-Recall - the quality of retrieved context.
We follow the work from  (Chen et al.  2024b )  which establishes Retrieval-Augmented Generation Benchmark (RGB) to measure 4 abilities required for RAG, including noise robustness, negative rejection, information integration, and counterfactual robustness.
In particular, using questions from 4 medical QA datasets from MIRAGE as basis, we create Medical Retrieval-Augmented Generation Benchmark (MedRGB) to evaluate RAG system in the following 4 test scenarios:

•

Standard-RAG:

evaluates LLMs performance when presented with multiple retrieved signal documents to create a context to answer to question.

•

Sufficiency:

evaluates LLMs reliability when there are noise documents within the retrieved context. By adding ”Insufficient Information” as an additional response option, LLMs should only answer when they are confident to have enough information to determine the correct answer. This
requires LLMs to not only be aware of its own internal knowledge, but also be able to filter out noisy information from external documents.

•

Integration:

evaluates LLMs ability to answer multiple supporting questions and integrate the extracted information to help address the main question.

•

Robustness:

evaluates LLMs resiliency to factual errors in the retrieved context. A trustworthy AI medical system should be able detect factually incorrect documents and provide the corrected information.

In total, MedRGB consists of 3480 instances for 4 test scenarios, which is over 5 times that of RGB.
Using MedRGB, we evaluation 7 LLMs, including both state-of-the art commercial LLMs and open-source models.
In summary, our contributions are three-fold:

•

We establish MedRGB with four test scenarios to evaluate LLMs for medical QA tasks in RAG settings. To best of our knowledge, it is the ﬁrst benchmark comprehensively assess medical RAG systems in these practical setting.

•

Using MedRGB, we extensively evaluate 7 LLMs, including both state-of-the art commercial LLMs and open-source models, across multiple RAG conditions. Experiment results demonstrate their limitation in addressing the more complex scenarios.

•

We analyzed the errors of the LLMs and their reasoning process to provide insights and suggest future directions for developing more reliable and trustworthy medical RAG systems.

Figure 2:  The overall construction process of MedRGB. The green OpenAI symbol implies that the block involves data generation using the GPT-4o model.

Related Work

Medical Retrieval-augmented Generation

The application of LLMs in medical domain demands a high level of accuracy and reliablity, which most of the current LLMs still struggle with  (Zhou et al.  2023 ) .
Retrieval-augmented Generation (RAG)  (Lewis et al.  2020 )  addresses this problem by helping LLMs integrating external knowledge sources in their generation process.
Recent works has achieved success in leverage RAG for knowledge-intensive tasks  (Cui et al.  2023 ; Peng et al.  2023 ; Ram et al.  2023 ) .
Specifically for medical domain,  (Hiesinger et al.  2023 ; Wang et al.  2024 ; Xiong et al.  2024 )  explore RAG for healthcare and clinical tasks.

Medical Benchmarks

Previous medical benchmarks usually focus solely on target performance of medical problems, which consist of only QA pairs  (Jin et al.  2020a ,  2019 ; Krithara et al.  2023 ) .
Some recent benchmark also included evidence for LLMs to reasoning on  (Chen et al.  2024a ) .
Most current systematic evaluations of LLMs in medical domain do not involve RAG  (He et al.  2023 ; Nori et al.  2023 ) .
 (Xiong et al.  2024 )  attempts to provide a systematic evaluations of RAG systems in medicine.
We build on their work to further evaluate important criteria of a medical RAG system for variety of practical settings.

You are a medical expert. Generate ranked search topics to help answer a medical question. Follow these guidelines:

1. Rank topics by importance to the question.

2. Ensure relevance to the question and answer options.

3. The topics should be differentiable and efficient for information retrieval.