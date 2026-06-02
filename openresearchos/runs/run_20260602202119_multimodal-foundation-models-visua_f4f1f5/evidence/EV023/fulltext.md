[2510.22728] S-Chain: Structured Visual Chain-of-Thought for Medicine

\useunder

\ul

\useunder

S-Chain : Structured Visual Chain-of-Thought for Medicine

Khai Le-Duc

∗  *

1,2

Duy M. H. Nguyen

∗  *

3,4,24

Phuong T. H. Trinh

∗  *

5

Tien-Phat Nguyen

∗  *

6

Nghiem T. Diep

∗  ⁣  ∗

**

3

An Ngo

∗  ⁣  ∗

**

7

Tung Vu

∗  ⁣  ∗

**

8

Trinh Vuong

9

Anh-Tien Nguyen

10,11

Mau Nguyen

12

Van Trung Hoang

13

Khai-Nguyen Nguyen

14

Hy Nguyen

15

Chris Ngo

2

Anji Liu

16

Nhat Ho

17

Anne-Christin Hauschild

11

Khanh Xuan Nguyen

18

Thanh Nguyen-Tang

19

Pengtao Xie

20,21

Daniel Sonntag

3,22

James Zou

23

Mathias Niepert

4,24

Anh Totti Nguyen

25

1  University of Toronto, Canada  2  Knovel Engineering Lab, Singapore

3  German Research Centre for Artificial Intelligence
 4  University of Stuttgart, Germany

5  Chonnam National University, South Korea
 6  Singapore University of Technology and Design

7  Bucknell University, USA
 8  Concordia University, Canada
 9  Korea University

10  Justus Liebig University Giessen, Germany
 11  University Medical Center Göttingen, Germany

12  Japan Advanced Institute of Science and Technology
 13  Hue University, Vietnam

14  College of William &amp; Mary, USA
 15  Deakin University, Australia

16  National University of Singapore
 17  University of Texas at Austin, USA

18  University of California, Berkeley, USA
 19  New Jersey Institute of Technology, USA

20  University of California San Diego, USA,
 21  MBZUAI, UAE

22  Oldenburg University, Germany
 23  Stanford University, USA

24  Max Planck Research School for Intelligent Systems (IMPRS-IS), Germany

25  Auburn University, USA

*Co-first authors; order randomized  **Co-second authors
 #  duckhai.le@mail.utoronto.ca ,

ho_minh_duy.nguyen@dfki.de, anhnguyen@auburn.edu

S-Chain

Abstract

Faithful reasoning in medical vision–language models (VLMs) requires not only accurate predictions but also transparent alignment between textual rationales and visual evidence. While Chain-of-Thought (CoT) prompting has shown promise in medical visual question answering (VQA), no large-scale expert-level dataset has captured stepwise reasoning with precise visual grounding. We introduce  S-Chain , the first large-scale dataset of 12,000 expert-annotated medical images with bounding boxes and structured visual CoT (SV-CoT), explicitly linking visual regions to reasoning steps. The dataset further supports 16 languages, totaling over 700k VQA pairs for broad multilingual applicability. Using  S-Chain , we benchmark state-of-the-art medical VLMs (ExGra-Med, LLaVA-Med) and general-purpose VLMs (Qwen2.5-VL, InternVL2.5), showing that SV-CoT supervision significantly improves interpretability, grounding fidelity, and robustness. Beyond benchmarking, we study its synergy with retrieval-augmented generation, revealing how domain knowledge and visual grounding interact during autoregressive reasoning. Finally, we propose a new mechanism that strengthens the alignment between visual evidence and reasoning, improving both reliability and efficiency. S-Chain establishes a new benchmark for grounded medical reasoning and paves the way toward more trustworthy and explainable medical VLMs.

1  Introduction

Large Language Models (LLMs)  and  Vision Language Models (VLMs)  have shown strong capabilities in problem solving, planning, and decision making by learning deductive and inductive reasoning from large-scale data. A key driver is  Chain-of-Thought (CoT)  reasoning, which breaks complex tasks into step-by-step inferences before reaching a final answer. This paradigm improves performance across domains, from arithmetic and commonsense reasoning in  LLM

( wei2022chain ; Kojima et al.,  2022 )  to  Visual Question Answering (VQA)  and multimodal reasoning in  VLM

( zhang2023multimodal ; Chen et al.,  2024a ) . By externalizing their reasoning process,  CoT  not only boosts accuracy but also adds interpretability, making them especially promising for high-stakes fields like healthcare.

Despite recent progress, training models with strong  CoT  reasoning still demands large amounts of annotated data, as models must learn to align intermediate reasoning steps with input evidence  ( zelikman2022star ;  wang2022self ) . In general  Natural Language Processing (NLP) , such supervision can be scaled through crowdsourcing or distillation  (Magister et al.,  2022 ; Ho et al.,  2023 ) , but in medicine, it is far more costly: annotations must be expert-verified, multimodal, and clinically valid  ( moor2023foundation ; Huang et al.,  2024 ) . Beyond this, medical reasoning requires visual grounding, i.e., explicitly linking reasoning steps to  Region of Interest (ROI) , which adds substantial complexity. As a result, large-scale expert datasets with grounded  CoT  remain scarce, limiting the training and evaluation of trustworthy medical  VLMs .

To mitigate the high cost of expert annotation, recent work has explored auto-generation of  CoT  data for  VLM  reasoning. For example, MC-CoT  ( wei2024mc )  leverages modular pipelines where  LLMs  generate reasoning steps that are loosely aligned with multimodal inputs in zero-shot settings, while MedCoT  (Liu et al.,  2024 )  introduces hierarchical expert verification to refine automatically produced rationales. Similarly, large medical  VQA  datasets such as PMC-VQA  ( zhang2023pmc )  rely on template-based or synthetic  Question Answering (QA)  generation to scale supervision. While such approaches improve data availability, their effectiveness is limited for clinical reasoning due to two key issues: (i) auto-generated  CoTs  often lack structure, providing free-text explanations without explicit correspondence to specific image regions, which weakens visual grounding; and (ii) they are prone to factual mistakes and hallucinations, frequently introducing redundant or clinically irrelevant content that is difficult to filter out  ( Gu et al.,  ; Cheng et al.,  2025 ) . These limitations highlight the need for high-quality, structured, and expert-grounded  CoT  annotations in the medical domain.

To address these challenges, we propose a new expert-annotated dataset that provides visually grounded  CoTs  explicitly linking step-by-step reasoning to visual evidence, which we term  Structured Visual Chain-of-Thought (SV-CoT) . Our dataset contains 12,000 medical images with bounding-box annotations of  ROI , paired with structured rationales that are decomposed into four clinically meaningful stages: (i) object localization, (ii) image captioning, (iii) multiple-choice reasoning, and (iv) image classification. Unlike auto-generated  CoTs , each rationale is carefully annotated and verified by medical experts, ensuring both factual accuracy and strong correspondence between reasoning steps and visual features. To enhance accessibility and global applicability, the dataset further supports  16 languages , resulting in over  700,000  QA  pairs . By combining structured reasoning, explicit grounding, multilingual coverage, and expert verification, this resource overcomes the key limitations of existing synthetic  CoT  approaches and establishes a reliable foundation for training and benchmarking medical  VLMs .

With this dataset in place, we systematically investigate its impact on the performance of multiple model families, including both domain-specific medical  VLMs  (e.g., ExGra-Med  ( nguyen2025exgra ) , LLaVA-Med  (Li et al.,  2023a ) ) and general-purpose  VLMs  (e.g., Qwen2.5-VL  ( wang2024qwen2 ) , InternVL2.5  (Chen et al.,  2024b ) ), and compare them against baselines trained with synthetic  CoTs  generated by GPT-4.1. Beyond standard evaluation, we further assess the integration of our  SV-CoT  supervision with  Retrieval-augmented Generation (RAG)

( zhao2025medrag ;  zheng2025miriad ) , examining how external domain-specific knowledge interacts with structured reasoning and visual grounding. A key focu