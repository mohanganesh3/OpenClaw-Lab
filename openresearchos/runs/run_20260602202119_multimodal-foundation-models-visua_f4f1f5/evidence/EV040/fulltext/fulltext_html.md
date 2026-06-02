Commonsense for Zero-Shot Natural Language Video Localization

Introduction

Related Work

Natural Language Video Localization (NLVL)

Weakly Supervised and Zero-shot NLVL Methods

Commonsense in Video-Language Tasks

Commonsense for Zero-Shot NLVL

Problem Formulation

Pseudo-supervised Setup

Dynamic Video Moment Proposal (

f  span

subscript  𝑓

span

f_{\text{span}}  italic_f start_POSTSUBSCRIPT span end_POSTSUBSCRIPT

).

Pseudo-query Generation (

f  pq

subscript  𝑓

pq

f_{\text{pq}}  italic_f start_POSTSUBSCRIPT pq end_POSTSUBSCRIPT

).

Video Encoder.

Query Encoder.

Commonsense Enhancement Module

Concept Encoder.

Commonsense Information.

Cross-Modal Interaction Module.

Temporal Regression Module.

Training and Inference

Experiments

Experimental Setup.

Experimental Results

Ablation Studies

Which Relation Types Are Most Important?

Qualitative Results

Conclusion

Acknowledgements

A  Implementation Details

B  Ablation Studies

How to Best Inject Commonsense?

Which Modality to Enhance with Commonsense?

When to Perform Commonsense Enhancement?

Does Retaining Relational Information Boost Localization?

Does Auxiliary Commonsense Information Boost Performance?

How to Best Encode Inputs?

Does Commonsense Help in Language-free Setups?

HTML conversions  sometimes display errors  due to content that did not convert correctly from the source. This paper uses the following packages that are not yet supported by the HTML conversion tool. Feedback on these issues are not necessary; they are known and are being worked on.

failed: arydshln

Authors: achieve the best HTML results from your LaTeX submissions by following these  best practices .

License: arXiv.org perpetual non-exclusive license  arXiv:2312.17429v2 [cs.CV] 01 Feb 2024

Commonsense for Zero-Shot Natural Language Video Localization

Meghana Holla 1 ,
Ismini Lourentzou 2

Abstract

Zero-shot Natural Language-Video Localization (NLVL) methods have exhibited promising results in training NLVL models exclusively with raw video data by dynamically generating video segments and pseudo-query annotations.
However, existing pseudo-queries often lack grounding in the source video, resulting in unstructured and disjointed content. In this paper, we investigate the effectiveness of commonsense reasoning in zero-shot NLVL. Specifically, we present CORONET, a zero-shot NLVL framework that leverages commonsense to bridge the gap between videos and generated pseudo-queries via a commonsense enhancement module. CORONET employs Graph Convolution Networks (GCN) to encode commonsense information extracted from a knowledge graph, conditioned on the video, and cross-attention mechanisms to enhance the encoded video and pseudo-query representations prior to localization. Through empirical evaluations on two benchmark datasets, we demonstrate that CORONET surpasses both zero-shot and weakly supervised baselines, achieving improvements up to

32.13  %

percent  32.13

32.13\%  32.13 %

across various recall thresholds and up to

6.33  %

percent  6.33

6.33\%  6.33 %

in mIoU. These results underscore the significance of leveraging commonsense reasoning for zero-shot NLVL.

Introduction

Natural Language Video Localization (NLVL) is a fundamental multimodal understanding task that aims to align textual queries with relevant video segments. NLVL is a core component for various applications such as video moment retrieval  (Cao et al.  2022 ) , video question answering  (Qian et al.  2023 ; Lei et al.  2020a ) , and video editing  (Gao et al.  2022 ) . Prior works have primarily explored supervised
 (Zeng et al.  2020 ; Wang, Ma, and Jiang  2020 ; Soldan et al.  2021 ; Liu et al.  2021 ; Yu et al.  2020 ; Gao et al.  2021 ) 
or weakly supervised  (Mun, Cho, and Han  2020 ; Zhang et al.  2020 ,  2021 )  NLVL methodologies, relying on annotated video-query data to various extents.

Obtaining annotated data for NLVL is a labor-intensive process that requires video samples paired with meticulous annotations of video moments and corresponding textual descriptions. Figure

1

illustrates the annotation requirements for different levels of supervision in NLVL. Fully supervised methods demand fine-grained moment span annotations, while weakly supervised methods typically rely on query descriptions alone. Nevertheless, both still heavily rely on paired video-language data, which limits practicality in open-domain settings.

Figure 1:  NLVL tasks under various supervision settings. Color-coded boxes show the expected annotations at each supervision level.  Full supervision : Temporal Video Annotations + Text Queries;  Weak Supervision : Text Queries;  Pseudo-Supervision : Only Raw Videos. DVP + DQG;  CORONET (Ours, right)  Only Raw Videos. DVP + OD and video-informed commonsense knowledge subgraph.

Recent works formulate zero-shot NLVL, which aims to dynamically generate video moments and their corresponding queries, eliminating the need for paired video-query data  (Nam et al.  2021 ; Kim et al.  2023 ) . Nonetheless, existing approaches have certain limitations. On one hand, recent methods generate pseudo-queries using off-the-shelf object detectors for objects (nouns) and text-based language models for actions (verbs), resulting in noisy pseudo-queries that lack grounding in the video content  (Nam et al.  2021 ) . On the other hand, language-free methods remove pseudo-queries entirely by utilizing vision-language models pretrained on large-scale image and text datasets  (Kim et al.  2023 ) . However, eliminating textual information entirely may lead to missing out on important semantic nuances.

Visual (video) and textual (query) modalities provide very distinct but complementary types of information; videos provide spatial and physical information, while queries provide situational and contextual information. Existing works focus on complex vision-language interactions for observed video-query pairs in an attempt to bridge this gap  (Nam et al.  2021 ; Mun, Cho, and Han  2020 ) . However, in the zero-shot/pseudo-supervised setting, where queries are in a simpler form without structural information, finding common ground between modalities becomes crucial for effective cross-modal interactions. Commonsense knowledge, which encompasses general knowledge about the world and relationships between concepts, has proven valuable in various tasks  (Fang et al.  2020 ; Ding et al.  2021 ; Yu et al.  2021 ; Li, Niu, and Zhang  2022 ; Maharana and Bansal  2021 ; Cao et al.  2022 ) . By incorporating commonsense information, NLVL models could potentially bridge the semantic gap between video and text modalities, enhancing the cross-modal understanding and performance in zero-shot NLVL.

To this end, this work introduces  C omm O nsense ze R o sh O t la N guage vid E o localiza T ion ( CORONET ), a zero-shot NLVL model that leverages commonsense knowledge to enhance the pseudo-query generation and cross-modal localization of video moments. We introduce a Commonsense Enhancement Module to enrich the encoded video and query representations with rich contextual information and employ external commonsense knowledge from ConceptNet  (Speer, Chin, and Havasi  2017 )  to extract relevant relationships between a predefined set of concepts, mined from the input videos. Our primary objective is to investigate the potential benefits and challenges of leveraging commonsense for zero-shot NLVL. By jointly incorporating commonsense knowledge, we show that our model effectively bridges the gap between visual and linguistic modalities.

The contributions of this work are summarized as follows:
 (1)  We introduce CORONET

1

1  1 Code available at  https://github.com/PLAN-Lab/CORONET

, a zero-shot NLVL framework that utilizes external commonsense knowledge to enrich cross-modal understanding between the visual and natural language components of pseudo-query generation. To the best of our knowledge, we are the first to incorpor