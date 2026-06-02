[2511.19257] Medusa: Cross-Modal Transferable Adversarial Attacks on Multimodal Medical Retrieval-Augmented Generation

Medusa: Cross-Modal Transferable Adversarial Attacks on Multimodal Medical Retrieval-Augmented Generation

Yingjia Shang

Westlake University and Heilongjiang University  Hangzhou  China

2232671@s.hlju.edu.cn

,

Yi Liu

City University of Hong Kong  Hong Kong  China

yiliu247-c@my.cityu.edu.hk

,

Huimin Wang

Tencent  Shenzhen  China

hmmmwang@tencent.com

,

Furong Li, Wenfang Sun, Chengyu Wu

Westlake University  Hangzhou  China

and

Yefeng Zheng ✉

Westlake University  Hangzhou  China

yefengzheng@westlake.com.cn

(2026)

Abstract.

With the rapid advancement of retrieval-augmented vision-language models, multimodal medical retrieval-augmented generation ( MMed-RAG ) systems are increasingly adopted in clinical decision support. These systems enhance medical applications by performing cross-modal retrieval to integrate relevant visual and textual evidence for tasks,  e.g. , report generation and disease diagnosis. However, their complex architecture also introduces underexplored adversarial vulnerabilities, particularly via visual input perturbations. In this paper, we propose  Medusa , a novel framework for crafting cross-modal transferable adversarial attacks on  MMed-RAG  systems under a black-box setting. Specifically,  Medusa  formulates the attack as a perturbation optimization problem, leveraging a multi-positive InfoNCE loss (MPIL) to align adversarial visual embeddings with medically plausible but malicious textual targets, thereby hijacking the retrieval process. To enhance transferability, we adopt a surrogate model ensemble and design a dual-loop optimization strategy augmented with invariant risk minimization (IRM). Extensive experiments on two real-world medical tasks, including medical report generation and disease diagnosis, demonstrate that  Medusa  achieves over 90% average attack success rate across various generation models and retrievers under appropriate parameter configuration, while remaining robust against four mainstream defenses, outperforming state-of-the-art baselines. Our results reveal critical vulnerabilities in the  MMed-RAG  systems and highlight the necessity of robustness benchmarking in safety-critical medical applications. The code and data are available at  https://anonymous.4open.science/r/MMed-RAG-Attack-F05A .

Multimodal Medical Retrieval-Augmented Generation, Cross-Modal Adversarial Attacks, Black Box

Authors marked with * contributed equally. Yi Liu is the lead author.  ✉  denotes the corresponding author. This work was supported by Zhejiang Leading Innovative and Entrepreneur Team Introduction Program (2024R01007).

†

†  journalyear:  2026

†

†  copyright:  acmlicensed

†

†  conference:  Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining; August 9-13, 2026; Jeju, Korea.

†

†  booktitle:  Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’26), August 9-13, 2026, Jeju, Korea

†

†  doi:  XXXXXXX.XXXXXXX

†

†  isbn:  978-1-4503-XXXX-X/2018/06

1.  Introduction

Vision Language Models (VLMs)  (Team,  2024 ; Zhang et al.,  2024a )  have recently shown impressive capabilities across diverse tasks such as image captioning  (Fei et al.,  2023 ) , visual question answering  (Jin et al.,  2024 ) , and clinical report generation  (Xiong et al.,  2024 ) . In the medical domain, these models are increasingly augmented with retrieval mechanisms, forming multimodal Retrieval-Augmented Generation (RAG) systems that leverage external medical knowledge bases to improve factuality, interpretability, and decision support  (Cai et al.,  2024 ; Xia et al.,  2025 ; Zhao et al.,  2025 ) . By conditioning generation on retrieved multimodal context,  e.g. , radiology images, clinical notes, and biomedical literature, multimodal medical RAG ( MMed-RAG )  (Zhao et al.,  2025 )  systems promise safer and more informative outputs in high-stakes environments. A prominent example from industry is Med-PaLM  (Singhal et al.,  2023 )  developed by Google Cloud, a medical VLM system that integrates multimodal RAG to deliver reliable, accurate, and trustworthy query-based services to healthcare providers and medical institutions.

However, the integration of retrieval and generation introduces a broader attack surface. Unlike conventional end-to-end models,  MMed-RAG  systems are sensitive not only to their inputs but also to the retrieval results that influence the generated output  (Ha et al.,  2025 ; Soudani,  2025 ; Zhang et al.,  2024c ) . This dual-stage architecture makes them particularly vulnerable to adversarial manipulation, where an attacker can perturb either the input query or the retrieval process to inject misleading or harmful content into the generation pipeline. For example, Zhang  et al.

(Zhang et al.,  2025a )  investigated poisoning attacks on RAG systems, in which adversarial knowledge is deliberately injected into the knowledge base to manipulate the model’s generation outputs. Furthermore, they proposed tracking techniques to detect and mitigate the impact of such malicious injections. These risks are magnified in the medical domain, where subtle distortions may result in misdiagnoses, incorrect clinical suggestions, or privacy breaches  (Han et al.,  2024 ; Jiao et al.,  2025 ) .

Existing studies on adversarial attacks have primarily focused on classification tasks or unimodal generative models  (Finlayson et al.,  2019 ;  Zheng_Wang_Liu_Ma_Shen_Wang_20 ) . While some recent efforts explore adversarial attacks on VLMs  (Zhang et al.,  2025b ,  2022 ; Dong et al.,  2025 ) , they often assume static retrieval or ignore the retrieval component altogether. Moreover, few works have fully addressed the transferability of adversarial examples across modalities  (Wang et al.,  2024 ; Lu et al.,  2023 )  or components ( e.g. , retrieval mechanisms )  (Demontis et al.,  2019 ) , which is a critical property for real-world attacks that operate under limited access assumptions. In  MMed-RAG  systems, where inputs generally span both images and text and outputs are conditioned on retrieved evidence, cross-modal and transferable attacks remain severely underexplored.  Therefore, there is an urgent need to investigate cross-modal adversarial vulnerabilities in  MMed-RAG  systems and rigorously evaluate their robustness against such threats.

In this paper, we present  Medusa , a novel framework for crafting cross-modal transferable adversarial attacks on  MMed-RAG . Specifically,  Medusa  generates perturbations on visual inputs that mislead the retrieval system, propagate through the generation pipeline, and ultimately produce misleading ( i.e. , targeted) medical outputs. We formulate the proposed attack as a perturbation optimization problem, aiming to simultaneously achieve two objectives: 1) disrupt the cross-modal retrieval process in  MMed-RAG  by maximizing the likelihood of retrieving content aligned with the adversary’s predefined target, and 2) steer the generative model to produce the desired erroneous output based on the manipulated retrieved knowledge. However, achieving the above goals is non-trivial. We identify the following two key challenges:

∙  \bullet

C1: Complex System Components.

MMed-RAG  is not a monolithic model but a complex pipeline comprising multiple interconnected components,  e.g. , the knowledge base, the retriever, and the generative model, often augmented with external defense mechanisms. An adversary must not only manipulate the retrieval process to induce erroneous results but also evade detection or mitigation by built-in safeguards. This significantly increases the difficulty of crafting effective attacks, especially when relying on conventional adversarial optimization techniques that are designed for simpler, end-to-end models.

∙  \bullet

C2: Black-Box Settings.  The adversary