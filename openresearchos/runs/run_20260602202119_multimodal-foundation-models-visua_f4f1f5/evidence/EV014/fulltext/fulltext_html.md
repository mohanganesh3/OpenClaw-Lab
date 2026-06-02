[2604.14044] Decoding the Delta: Unifying Remote Sensing Change Detection and Understanding with Multimodal Large Language Models

Figure 1 .

Our work primarily addresses the unified task of multi-temporal remote sensing change detection and understanding. The proposed Delta-QA dataset encompasses various QA task types and interaction modalities.

Decoding the Delta: Unifying Remote Sensing Change Detection and Understanding with Multimodal Large Language Models

DOI:

XXXXXXX.XXXXXXX

ISBN:  978-1-4503-XXXX-X/2018/06

CCS:  Computing methodologies Artificial intelligence

CCS:  Computing methodologies Interest point and salient region detections

CCS:  Computing methodologies Image segmentation

Xiaohe Li

email:  lixiaohe@aircas.ac.cn

Affiliation:

Beijing 
,  China 
,  Aerospace Information Research Institute, CAS

,

Jiahao Li

email:  lijiahao243@mails.ucas.ac.cn

Affiliation:

Beijing 
,  China 
,  Aerospace Information Research Institute, CAS

,

Kaixin Zhang

email:  zhangkaixin25@mails.ucas.ac.cn

Affiliation:

Beijing 
,  China 
,  Aerospace Information Research Institute, CAS

,

Yuqiang Fang

email:  fangyuqiang@nudt.edu.cn

Affiliation:

Beijing 
,  China 
,  Space Engineering University

,

Leilei Lin

email:  leilei˙lin@cnu.edu.cn

Affiliation:

Beijing 
,  China 
,  Capital Normal University

,

Hong Wang

email:  wanghong@aircas.ac.cn

Affiliation:

Beijing 
,  China 
,  Aerospace Information Research Institute, CAS

,

Haohua Wu

email:  wuhaohua23@mails.ucas.ac.cn

Affiliation:

Beijing 
,  China 
,  Aerospace Information Research Institute, CAS

and

Zide Fan

email:  fanzd@aircas.ac.cn

Affiliation:

Beijing 
,  China 
,  Aerospace Information Research Institute, CAS

(2026© , 2026; )

Abstract.

While Multimodal Large Language Models (MLLMs) excel in general vision-language tasks, their application to remote sensing change understanding is hindered by a fundamental ”temporal blindness”. Existing architectures lack intrinsic mechanisms for multi-temporal contrastive reasoning and struggle with precise spatial grounding. To address this, we first introduce Delta-QA, a comprehensive benchmark comprising 180k visual question-answering samples. Delta-QA unifies pixel-level segmentation and visual question answering across bi- and tri-temporal scenarios, structuring change interpretation into four progressive cognitive dimensions. Methodologically, we propose Delta-LLaVA, a novel MLLM framework explicitly tailored for multi-temporal remote sensing interpretation. It overcomes the limitations of naive feature concatenation through three core innovations: a Change-Enhanced Attention module that systematically isolates and amplifies visual differences, a Change-SEG module utilizing Change Prior Embedding to extract differentiable difference features as input for the LLM, and Local Causal Attention to prevent cross-temporal contextual leakage. Extensive experiments demonstrate that Delta-LLaVA decisively outperforms leading generalist MLLMs and specialized segmentation models in complex change deduction and high-precision boundary localization, establishing a unified framework for earth observation intelligence.

Keywords:  Multimodal Large Language Models, Remote Sensing, Change Detection, Visual Question Answering

Table 1 .

Comparison of existing multi-temporal remote sensing datasets for change perception and understanding.

Category

Dataset

Frame

Number

Land-Cover Type

Containing

Pixel Mask

Textual

QA/Caption

Task Type

Interactive

Prompt

Resolution

Annotation

Scale

Semantic Change

Detection

SECOND

(  yang2021asymmetric  )

2

6

✔

✘

Mask

✘

0.5-3

4662 Image Pairs

Landsat

(  yuan2022transformer  )

2

4

✔

✘

Mask

✘

30

8468 Image Pairs

WUSU

(  yuan2022transformer  )

3

11

✔

✘

Mask

✘

1

1116 Image Pairs

LevirSCD

(  zhang2025foba  )

2

16

✔

✘

Mask

✘

1-2

3225 Image Pairs

JL1

(  jl1_scd_dataset  )

2

5

✔

✘

Mask

✘

0.75

6000 Image Pairs

Change Detection

and Understanding

SECOND-CC

(  karaca2025robust  )

2

6

✔

✔

Caption

✘

0.5-3

30205 Captions

RSCC

(  chen2025rscc  )

2

-

✘

✔

Caption

✘

-

62351 Captions

LEVIR-CC

(  liu2022remote  )

2

-

✘

✔

Caption

✘

0.5

50385 Captions

CC-Foundation

(  wang2024ccexpert  )

2

-

✘

✔

Caption

✘

-

135k Captions

LEVIR-MCI

(  liu2024change  )

2

2

✔

✔

QA or Mask

✘

0.5

50.3k Instructs

ChangeChat

(  deng2025changechat  )

2

2

✘

✔

QA

✘

0.5

87k Instructs

DVL-Bench

(  xuan2025dynamicvl  )

5-10

5

✔

✔

Unified QA and Simple Mask Positioning

Box only

1

69926 Instructs

Delta-QA (ours)

2-3

15

✔

✔

Unified QA and Complex Mask Reasoning

Point &amp; Box

0.5-30

180k Instructs

1.  Introduction

Driven by rapid advancements in Earth observation, modern remote sensing systems continuously generate dense multi-temporal imagery, providing critical records for deciphering global dynamics such as urbanization and disaster response  ( rahnemoonfar2023rescuenet ;  zhang2024earthgpt ) . Concurrently, Multimodal Large Language Models (MLLMs)  ( hu2025rsgpt ;  yin2024survey )  have demonstrated extraordinary proficiency in vision-language alignment and zero-shot reasoning. Integrating MLLMs into remote sensing holds the potential to transcend the semantic bottlenecks of traditional task-specific computer vision, promising a leap toward open-ended and highly generalizable Earth observation intelligence  ( Yao2025RemoteSAMTS ) .

Despite these promising prospects, contemporary MLLMs remain fundamentally ill-equipped for the nuanced demands of remote sensing change understanding due to three critical limitations. First, existing works predominantly fine-tune current vision-language models to better understand intrinsic image content, specifically targeting improvements in color, count, and spatial positioning capabilities  ( luo2025large ;  wang2026geoeyes ;  li2026co ;  yang2026geoalignclip ;  kashyap2026bi ) . They lack intrinsic mechanisms for multi-temporal contrastive reasoning, rendering them effectively ”temporally blind” to evolutionary dynamics  ( irvin2024teochat ;  noman2025cdchat ) . Therefore, a profound interpretation and detailed analysis of intricate change dynamics are difficult to attain  ( wang2024ringmogpt ;  deng2026deltavlm ) . Second, they operate at a coarse semantic level, lacking the spatial grounding requisite for precise change delineation  ( zhang2025georsmllm ) . Custom integrations that attempt to graft spatial decoders or naively concatenate multitemporal features are fraught with tradeoffs, often degrading generalized reasoning or failing to achieve rigorous spatiotemporal alignment at the pixel scale  ( ma2025geomag ) . Finally, practical interpretation demands flexible and multidimensional interaction modes, yet current structurally monolithic approaches support only rigid interactions  ( zhang2025unichange ;  huang2025reasoncd ) . Consequently, there is an urgent need for a unified framework that can seamlessly integrate text, spatial prompts, and precise mask outputs to empower diverse analytical workflows.

To overcome these foundational limitations, this paper introduces synergistic advancements in both the data foundation and the model architecture. By integrating natural language with segmentation masks, we seek to empower vision-language models to genuinely comprehend the temporal variations in remote sensing imagery, effectively capturing the ’Delta’ between them.

On the data front, we propose an automated pipeline for generating tasks for change perception and reasoning and introduce the  Delta-QA  dataset. This comprehensive benchmark synthesizes existing datasets, namely SECOND  ( yang2021asymmetric ) , Landsat-SCD  ( yuan2022transformer ) , and WUSU  ( shi2023multi ) , to support bitemporal and tritemporal change scenarios at the pixel scale. It encompasses four principal task categories: Change Identification and Classification ( [CIC] ), Change Quantification and