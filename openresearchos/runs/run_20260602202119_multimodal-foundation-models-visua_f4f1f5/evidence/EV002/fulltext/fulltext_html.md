[2410.05160] VLM2Vec: Training Vision-Language Models for Massive Multimodal Embedding Tasks

VLM2Vec: Training Vision-Language Models for Massive Multimodal Embedding Tasks

Ziyan Jiang  1

, Rui Meng  2  , Xinyi Yang  2  , Semih Yavuz  2  , Yingbo Zhou  2  , Wenhu Chen  1

1  University of Waterloo,

2  Salesforce Research
 ziyanjiang528@gmail.com, ruimeng@salesforce.com, wenhuchen@uwaterloo.ca

Work done during an internship at University of Waterloo in collaboration with Salesforce Research. Corresponding authors are Ziyan Jiang, Rui Meng and Wenhu Chen

Abstract

Embedding models have been crucial in enabling various downstream tasks such as semantic similarity, information retrieval, and clustering. Recently, there has been a surge of interest in developing universal text embedding models that can generalize across tasks (e.g., MTEB). However, progress in learning universal multimodal embedding models has been relatively slow despite their importance. In this work, we aim to explore the potential for building universal embeddings capable of handling a wide range of downstream tasks. Our contributions are twofold: (1) MMEB (Massive Multimodal Embedding Benchmark), which covers 4 meta-tasks (i.e. classification, visual question answering, multimodal retrieval, and visual grounding) and 36 datasets, including 20 training and 16 evaluation datasets, and (2)  Vlm2Vec  (Vision-Language Model

→

→

\rightarrow

Vector), a contrastive training framework that converts any state-of-the-art vision-language model into an embedding model via training on MMEB. Unlike previous models such as CLIP and BLIP,  Vlm2Vec  can process any combination of images and text to generate a fixed-dimensional vector based on task instructions. We build a series of  Vlm2Vec  models on Phi-3.5-V and evaluate them on MMEB’s evaluation split. Our results show that  Vlm2Vec  achieves an absolute average improvement of 10% to 20% over existing multimodal embedding models on both in-distribution and out-of-distribution datasets in MMEB.

https://tiger-ai-lab.github.io/VLM2Vec/

1  Introduction

Figure 1:  We develop a universal multimodal embedding benchmark, MMEB, along with  Vlm2Vec , an embedding model adapted from vision-language models (VLMs).  Vlm2Vec  is capable of following instructions and performing various multimodal embedding tasks, accommodating any combination of image and text modalities.

Embeddings, or distributed representations, encode inputs (whether text or images) as fixed-dimensional vectors, enabling a range of downstream tasks. Since the advent of Word2Vec  (Mikolov,  2013 )  and GloVe  (Pennington et al.,  2014 ) , substantial research efforts have focused on learning textual embeddings  (Kiros et al.,  2015 ; Conneau et al.,  2017 )  and image embeddings  (Radford et al.,  2021 ; Li et al.,  2022 ; Jia et al.,  2021 ; Yu et al.,  2022 ) . These embeddings facilitate a variety of applications, including textual and visual semantic similarity  (Agirre et al.,  2012 ; Marelli et al.,  2014 ; Chechik et al.,  2010 ; Cer et al.,  2017 ) , information retrieval  (Mitra et al.,  2017 ; Karpukhin et al.,  2020 ; Lin et al.,  2014 ) , automatic evaluation  (Zhang et al.,  2020 ; Sellam et al.,  2020 ) , prompt retrieval for in-context learning  (Liu et al.,  2022 ; Rubin et al.,  2022 ; Hongjin et al.,  2022 ) , and retrieval-augmented generation  (Lewis et al.,  2020 ; Guu et al.,  2020 ; Izacard &amp; Grave,  2020 ) . A recent shift in research has focused on developing universal embeddings that can generalize across a wide range of tasks. For instance,  Muennighoff et al. ( 2023 )  introduced MTEB (Massive Text Embedding Benchmark) to comprehensively assess text embeddings across tasks such as classification and clustering. MTEB has become the standard for evaluating universal text embeddings. Recent works  (Wang et al.,  2022a ; Su et al.,  2023 ; Wang et al.,  2024 ; Springer et al.,  2024 ; BehnamGhader et al.,  2024 )  have demonstrated promising results on the MTEB benchmark. However, progress in multimodal embeddings has been relatively slower. Despite advancements in text embeddings, the lack of both benchmarks and methodologies in the multimodal embedding domain remains a challenge.

Current research in multimodal embeddings faces two primary limitations: (1) existing studies typically evaluate visual embeddings on isolated tasks, such as ImageNet classification  (Deng et al.,  2009 ; Hendrycks et al.,  2021a ;  b )  or MSCOCO/Flickr retrieval  (Lin et al.,  2014 ; Plummer et al.,  2015 ) ; (2) most existing models, such as CLIP  (Radford et al.,  2021 ) , BLIP  (Li et al.,  2022 ) , and SigLIP  (Zhai et al.,  2023 ) , either process text and images separately or perform shallow fusion of visual and textual information  (Wei et al.,  2023 ) , limiting their ability to fully capture the relationships between text and image modalities. Furthermore, these models exhibit limited reasoning and generalization capabilities, particularly in zero-shot scenarios for complex reasoning tasks.

In this paper, we attempt to build an universal multimodal embedding framework to pave road for the future research, which consists of two efforts:

- MMEB:  We introduce a novel benchmark, MMEB (Massive Multimodal Embedding Benchmark), which includes 36 datasets spanning four meta-task categories: classification, visual question answering, retrieval, and visual grounding. MMEB provides a comprehensive framework for training and evaluating embedding models across various combinations of text and image modalities. All tasks are reformulated as ranking tasks, where the model follows instructions, processes a query, and selects the correct target from a set of candidates. The query and target can be an image, text, or a combination of both. MMEB is divided into 20 in-distribution datasets, which can be used for training, and 16 out-of-distribution datasets, reserved for evaluation.

-  Vlm2Vec :  We adopt the pre-trained vision-language model Phi-3.5-V  (Abdin et al.,  2024 )  as the backbone for  Vlm2Vec . In contrast to other multimodal embedding models like UniIR  (Wei et al.,  2023 )  and MagicLens  (Zhang et al.,  2024 ) , which rely on late fusion of CLIP  (Radford et al.,  2021 )  features, our approach leverages the deep integration of vision and language features within a transformer architecture. There are several advantages to this approach: (1) VLMs are trained on massive multimodal datasets and can handle any combination of images and text, as well as high-resolution images and long text inputs; (2) vision and language features are deeply fused in the transformer model, improving the model’s ability to capture cross-modal relationships; and (3) these models are well-suited for generalizing across diverse tasks, particularly those requiring instruction-following capabilities. These factors make  Vlm2Vec  an ideal choice for task generalization. We trained  Vlm2Vec  on the 20 MMEB training datasets using contrastive learning and compared its performance with various baselines.

Following extensive contrastive training,  Vlm2Vec

can handle any combination of images and text, producing fixed-dimensional vectors . We evaluate  Vlm2Vec  against a wide array of multimodal embedding models, including CLIP  (Radford et al.,  2021 ) , BLIP2  (Li et al.,  2023a ) , SigLIP  (Zhai et al.,  2023 ) , MagicLens  (Zhang et al.,  2024 ) , UniIR  (Wei et al.,  2023 )  and E5-V  (Jiang et al.,  2024 ) , demonstrating consistent improvements across all task categories. Notably,  Vlm2Vec  achieves a 17.3-point improvement (from 42.8 to 60.1) across all 36 MMEB datasets and a 11.6-point increase (from 40.4 to 52.0) on 16 out-of-distribution datasets in zero-shot evaluation, highlighting the effectiveness of our proposed framework. We are committed to releasing all data, training code, and models to facilitate future research in this area.

2  MMEB: A Benchmark for Multimodal Embeddi