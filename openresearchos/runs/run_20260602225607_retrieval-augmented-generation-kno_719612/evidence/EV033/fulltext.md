[2506.19610] V2T-CoT: From Vision to Text Chain-of-Thought for Medical Reasoning and Diagnosis

\useunder

\ul

1

1  institutetext:  Zhejiang Key Laboratory of Medical Imaging Artificial Intelligence, Zhejiang University, HangZhou, China

2

2  institutetext:  Academy for Engineering and Technology, Fudan University, Shanghai, China

3

3  institutetext:  Department of Oral and Maxillofacial Radiology, Stomatology Hospital, School of Stomatology,Zhejiang University, Hangzhou, China

4

4  institutetext:  Intelligent Computing Infrastructure Innovation Center, Zhejiang Lab, Hangzhou, China

4

4  email:  {yuan2.24,zuozhuliu}@intl.zju.edu.cn

https://github.com/Venn2336/V2T_CoT

V2T-CoT: From Vision to Text Chain-of-Thought for Medical Reasoning and Diagnosis

Yuan Wang

Equal contribution.  🖂 Corresponding author.11

Jiaxiang Liu  ∗

11

Shujian Gao

22

Bin Feng

33

Zhihang Tang

44

Xiaotang Gai

11

Jian Wu

11

Zuozhu Liu

🖂

{}^{\href mailto:zuozhuliu@intl.zju.edu.cn}

11

Abstract

Recent advances in multimodal techniques have led to significant progress in Medical Visual Question Answering (Med-VQA).
However, most existing models focus on global image features rather than localizing disease-specific regions crucial for diagnosis.
Additionally, current research tends to emphasize answer accuracy at the expense of the reasoning pathway, yet both are crucial for clinical decision-making.
To address these challenges, we propose From Vision to Text Chain-of-Thought ( V2T-CoT ), a novel approach that automates the localization of preference areas within biomedical images and incorporates this localization into region-level pixel attention as knowledge for Vision CoT.
By fine-tuning the vision language model on constructed  R-Med 39K  dataset, V2T-CoT provides definitive medical reasoning paths. V2T-CoT integrates visual grounding with textual rationale generation to establish precise and explainable diagnostic results.
Experimental results across four Med-VQA benchmarks demonstrate state-of-the-art performance, achieving substantial improvements in both performance and interpretability.

Keywords:  Med-VQA Vision Language Model Chain of Thought.

1  Introduction

Medical Visual Question Answering (Med-VQA) has emerged as a critical domain in healthcare AI, leveraging multimodal deep learning to answer clinical questions related to images posed in natural language  [ 5 ,  8 ,  10 ,  13 ] . Med-VQA holds promise for automated, personalized health consultations, addressing the growing demand for accessible and timely healthcare  [ 26 ] . By integrating textual data (e.g., clinical notes, patient histories) with medical images (e.g., X-rays, MRIs), these models provide a holistic understanding of patient conditions, improving decision-making and care delivery.

Recent advancements in Med-VQA have adopted deep learning frameworks such as DoctorGLM  [ 33 ] ,Huatuo-o1  [ 4 ]  and MedFound  [ 28 ]  to enhance the medical reasoning process. However, three core challenges persist  [ 35 ,  27 ,  36 ,  12 ] :  First , existing feature fusion mechanisms (e.g., concatenation/weighted operations) suffer from representational capacity limitations. These coarse-grained fusion strategies fail to establish precise mappings between visual key regions (e.g., lesion areas in CT/MRI) and textual semantics, resulting in ineffective capture of diagnosis-relevant fine-grained cross-modal correlations.  Second , high-accuracy models lack explainable reasoning pathways. Taking LLaVA-Med  [ 22 ]  (Fig.

1

B ) as an example, despite achieving high accuracy, its black-box decision mechanism cannot generate clinical reasoning chains (e.g., progressive analysis for malignancy determination based on tumor morphological features), severely hindering trustworthiness verification in diagnostic scenarios  [ 27 ] .  Third , multimodal CoT methods face both data and algorithm constraints. Frameworks like MedThink  [ 7 ]  (Fig.

1

C ), while incorporating clinical reasoning chains, are limited by: annotation data scarcity due to manual labeling costs; absence of specialized instruction-tuning datasets for training reasoning paths  [ 31 ,  27 ] . These limitations collectively lead to insufficient generalization capabilities in multi-perspective complex medical analysis.

Figure 1:  The comparison between V2T-CoT and existing Med-VQA methods.
 A  employs a combined vision and text encoding strategy with regional attention for medical diagnosis. In contrast, previous methods ( B  &amp;  C ) either lack reasoning or utilize it in a text-only context.  D  demonstrates the pipeline of V2T-CoT.

In this paper, we propose a novel Med-VQA method: From Vision to Text CoT ( V2T-CoT ), which addresses the aforementioned challenges by effectively integrating textual and visual reasoning (Fig.

1

A ). This method combines visual clues with textual information, comprising two core components: (1) a preference localization mechanism that identifies key regions in descriptive text relevant to the image; (2) a regional pixel-level attention mechanism that focuses on specific areas of medical images crucial for final diagnosis. To further enable explainable reasoning, we construct an instruction-tuning dataset  R-Med 39K  containing multi-granularity reasoning paths. Ultimately, our vision language model (VLM) generates coherent and interpretable diagnostic rationales, from which medical conclusions are derived. Experiments demonstrate that this approach achieves SOTA performance under comparable parameter sizes, significantly outperforming existing models. The main contributions are summarized as follows:

•

We propose a medical visual to text chain-of-thought reasoning framework,  V2T-CoT , which provides disease-related visual cues and a clear reasoning path, facilitating medical diagnosis.

•

To locate disease-related visual cues, V2T-CoT introduces an automated method (Vision CoT) for identifying and aligning relevant image regions.

•

We constructed  R-Med 39K  (Instruction-Tuning dataset) with reasoning paths from four Med-VQA datasets for Text CoT, integrating them into VLM training for reliable rationale validated by both LLMs and experts.

•

Through extensive experiments, we validated the superior performance of V2T-CoT on four Med-VQA datasets compared to existing methods, demonstrating the effectiveness of Vision CoT for visual localization and the rationale in Text CoT for reasoning pathways, meeting the clarity and interpretability requirements of both humans and LLMs.

2  Method

2.1  Overview

In this work, we propose a novel multimodal approach for Med-VQA that integrates visual and textual reasoning chains. As shown in Fig.

1

D , V2T-CoT first leverages a vision encoder to extract spatial-semantic features from critical anatomical regions, explicitly capturing structural deviations such as midline shifts (Vision CoT). The reasoning module then synthesizes anatomical region proposals with multimodal evidence to construct diagnostic rationales (Text CoT), ultimately delivering clinically interpretable diagnoses with both accuracy and clinical diagnostic transparency.

2.2  Vision CoT

2.2.1  Formulation of Phrase Grounding.

We reformulate medical object detection as phrase grounding within the Vision CoT framework, aligning each image region with text prompt phrases. As shown in Fig.

2  , given an image and text prompt, visual encoder

Enc  I

\text{Enc}_{I}

extracts region features

V  ∈

ℝ

N  ×  d

V\in\mathbb{R}^{N\times d}

, while language encoder

Enc  L

\text{Enc}_{L}

encodes textual tokens

T  ∈

ℝ

M  ×  d

T\in\mathbb{R}^{M\times d}

. The alignment score matrix

S  ground

∈

ℝ

N  ×  M

S_{\text{ground}}\in\mathbb{R}^{N\times M}

is computed via:

S  ground

=

V  ​

T  ⊤

S_{\text{ground}}=VT^{\top}

,
where

V  V

and

T  T

denote region-level visual features and phrase-level text embeddings respectively.

2.2.2  Fusion of Visual and Language Features.

T