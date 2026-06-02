[2604.24346] SycoPhantasy: Quantifying Sycophancy and Hallucination in Small Open Weight VLMs for Vision-Language Scoring of Fantasy Characters

SycoPhantasy: Quantifying Sycophancy and Hallucination in Small Open Weight VLMs for Vision-Language Scoring of Fantasy Characters

Arya Shah

Affiliation:  IIT Gandhinagar

Affiliation:  Gandhinagar

India

Affiliation:

arya.shah@iitgn.ac.in

Deepali Mishra

Affiliation:  IIT Kanpur

Affiliation:  Kanpur

India

Affiliation:

deepalim25@iitk.ac.in

Chaklam Silpasuwanchai

Affiliation:  Asian Institute of Technology

Affiliation:  Bangkok

Thailand

Affiliation:

chaklam@ait.asia

Abstract

Vision-language models (VLMs) are increasingly deployed as evaluators in tasks requiring nuanced image understanding, yet their reliability in scoring alignment between images and text descriptions remains underexplored. We investigate whether small, open-weight VLMs exhibit  sycophantic  behavior when evaluating image-text alignment: assigning high scores without grounding their judgments in visual evidence. To quantify this phenomenon, we introduce the  Bluffing Coefficient  (

B  c

B_{c}

), a metric that measures the mismatch between a model’s score and its evidence recall. We evaluate six open-weight VLMs ranging from 450M to 8B parameters on a benchmark of 173,810 AI-generated character portraits paired with detailed textual descriptions. Our analysis reveals a significant inverse correlation between model size and sycophancy rate (

r  =

−  0.96

r=-0.96

,

p  =  0.002

p=0.002

), with smaller models exhibiting substantially higher rates of unjustified high scores. The smallest model tested (LFM2-VL, 450M) produced sycophantic evaluations in 22.3% of cases, compared to 6.0% for the largest (LLaVA-1.6, 7B). These findings have direct implications for the deployment of small, open-weight VLMs as automated evaluators within attribute-rich, synthetic image evaluation tasks, where the gap between assigned scores and cited visual evidence is both measurable and consequential.

SycoPhantasy: Quantifying Sycophancy and Hallucination in Small Open Weight VLMs for Vision-Language Scoring of Fantasy Characters

Arya Shah

IIT Gandhinagar

Gandhinagar, India

arya.shah@iitgn.ac.in

Deepali Mishra

IIT Kanpur

Kanpur, India

deepalim25@iitk.ac.in

Chaklam Silpasuwanchai

Asian Institute of Technology

Bangkok, Thailand

chaklam@ait.asia

1  Introduction

Figure 1:  Overview of the sycophancy detection pipeline. Given a character description and AI-generated portrait, we prompt VLMs to provide an alignment score with reasoning. The Bluffing Coefficient measures the gap between score and cited visual evidence, enabling classification of evaluations as sycophantic or evidence-based.

Vision-language models (VLMs) such as LLaVA

liu2023visualinstructiontuning  , Qwen2-VL

wang2024qwen2vlenhancingvisionlanguagemodels  , and Phi-3.5-Vision

abdin2024phi3technicalreporthighly

have demonstrated remarkable multimodal capabilities. This has led to their deployment as automated evaluators, extending the “LLM-as-a-judge” paradigm

zheng2023judgingllmasajudgemtbenchchatbot ;  xiong2025llavacriticlearningevaluatemultimodal

to visual domains. However, models trained with RLHF exhibit  sycophantic  behavior of providing positive assessments even when evidence does not support such judgments

sharma2025understandingsycophancylanguagemodels  . While hallucination in VLMs has received substantial attention

li2023evaluatingobjecthallucinationlarge ;  Huang_2025 ;  sahoo2024comprehensivesurveyhallucinationlarge  , the specific problem of  sycophancy in VLM scoring  remains unexplored.

This paper is a testbed study: we use text-to-image (T2I) fantasy character evaluation as a high-signal proxy for VLM evaluator behavior. Fantasy NPC descriptions are deliberately chosen because they provide uniquely rich, structured, and verifiable visual attributes (150–250 words per description, covering appearance, clothing, and distinguishing features), making score-evidence mismatches both detectable and quantifiable at scale. Our findings and the Bluffing Coefficient are therefore grounded in this domain, and we present them as evidence of a systematic failure mode whose broader generalization should be validated in future work.

We investigate three research questions:  (RQ1)  Do small, open-weight VLMs exhibit sycophantic behavior when evaluating image-text alignment?  (RQ2)  Is there a relationship between model size and sycophancy?  (RQ3)  What patterns emerge when we measure the gap between scores and cited visual evidence?

To address these questions, we introduce the  Bluffing Coefficient  (

B  c

B_{c}

), a metric quantifying the mismatch between a VLM’s score and the evidence it cites, computed as

B  c

=

S  norm

−

R  +

+

R  −

B_{c}=S_{\text{norm}}-R^{+}+R^{-}

(normalized score minus positive evidence recall plus negative recall). Keyphrases are extracted from descriptions using spaCy and matched against model reasoning via sentence embeddings

reimers2019sentencebertsentenceembeddingsusing  , enabling evaluation at scale without human judgment. Across six open-weight VLMs (450M–8B parameters), sycophancy rate exhibits a strong inverse correlation with model size (

r  =

−  0.96

r=-0.96

,

p  =  0.002

p=0.002

), ranging from 22.3% (LFM2-VL, 450M) to 6.0% (LLaVA-1.6, 7B). Our contributions are: (1) The  Bluffing Coefficient , a novel metric quantifying score-evidence mismatch in VLM evaluations. (2) A  large-scale benchmark  of 173,810 image-description pairs for studying sycophancy in visual scoring and (3) An  empirical analysis  demonstrating a significant inverse relationship between model size and sycophancy rate. We release our code and dataset on  GitHub  and  Hugging Face  respectively.

2  Related Work

Our work bridges sycophancy in language models, hallucination in VLMs, model-based evaluation, and image-text alignment metrics.

2.1  Sycophancy in Language Models

Sycophancy refers to models providing responses aligned with perceived user preferences rather than accurate information.

sharma2025understandingsycophancylanguagemodels

demonstrated that RLHF-trained models

ouyang2022traininglanguagemodelsfollow

exhibit sycophantic behavior: incorrectly admitting mistakes when challenged, providing biased feedback matching user opinions, and mimicking errors. This stems from optimizing for human preference signals where raters favor agreeable responses. Alternative alignment approaches like DPO

rafailov2024directpreferenceoptimizationlanguage

and methods addressing reward hacking

miao2024informmitigatingrewardhacking ;  chen2024odindisentangledrewardmitigates

have been proposed, while TruthfulQA

lin2022truthfulqameasuringmodelsmimic

distinguishes truthfulness from sycophancy. However, these studies focus on text-only models; the extension to VLM evaluation remains unexplored.

2.2  Hallucination in Vision-Language Models

Hallucination in VLMs refers to generating content not grounded in visual input. POPE

li2023evaluatingobjecthallucinationlarge

uses binary questions to probe object hallucination, revealing that even state-of-the-art VLMs affirm non-existent objects. MMHal-Bench

sun2023aligninglargemultimodalmodels

extends to open-ended responses, while CHAIR

rohrbach2019objecthallucinationimagecaptioning

and ALOHa

petryk2024alohanewmeasurehallucination

measure caption hallucination rates. Comprehensive surveys

Huang_2025 ;  sahoo2024comprehensivesurveyhallucinationlarge

catalogue causes, detection methods, and mitigations. Our work differs by focusing not on incorrect content generation, but on unjustifiably high evaluation scores paired with reasoning lacking visual evidence, which is a form of evaluator-specific sycophancy.

2.3  LLMs and VLMs as Evaluators

The LLM-as-a-judge paradigm

zheng2023judgingllmasajudgemtbenchchatbot

demonstrated that GPT-4 achieves 80%+ agreement with human preference