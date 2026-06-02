[2512.03025] LORE: A Large Generative Model for Search Relevance

LORE: A Large Generative Model for Search Relevance

Chenji Lu  ∗  , Zhuo Chen  ∗  , Hui Zhao, Zhiyuan Zeng, Gang Zhao,

Junjie Ren, Ruicong Xu, Haoran Li, Songyan Liu, Pengjie Wang, Jian Xu, Bo Zheng

{luchenji.lcj, cz462596, shuqian.zh, zengzhiyuan.zzy, zilong.zg,
 renjunjie.rjj, xuruicong.xrc, lhr476916, moxuan.lsy, pengjie.wpj, xiyu.xj}@taobao.com

bozheng@alibaba-inc.com

Search Advertising Team, Alimama (Alibaba Group)

Abstract

Achievement.  We introduce LORE( L arge Generative M o del for Search  R elevanc e ), a complete and sustainable framework of iterative practices for large language models(LLMs) in e-commerce search relevance, which achieves a cumulative +27% improvement on the online GoodRate metric. Over the past three years, this project has demonstrated significant improvements in relevance judgment and has undergone three full-scale iterations across key dimensions, including data, features, training paradigms, evaluation, and application. Throughout the iterative development of LORE, we have gained valuable experience and insights that we believe are worth sharing with the community in this report.

Insight.  To enhance LLMs for relevance, existing works have modeled the Chain-of-Thought (CoT) from various perspectives. However, we find that these methods often exhibit blind spots, as they lack a principled deconstruction of the task itself. Our analysis reveals that complex relevance judgment is not a monolithic reasoning problem but rather a composite of distinct capabilities, including knowledge and reasoning, multi-modal matching, and rule adherence. Based on this insight, we propose a systematic framework that first deconstructs the problem and then leverages this deconstruction to guide a training paradigm that explicitly models each required capability. We argue that such qualitative-driven analysis is essential for breaking through existing performance bottlenecks.

Contributions.  LORE is a complete, replicable blueprint for LLM-based relevance modeling that spans the entire lifecycle.
First, we conducted systematic preliminary explorations into foundational training elements—including features, prompts, and base models—and summarized the general principles derived from this process.

Second, guided by our structural analysis, we propose a sophisticated two-stage training paradigm. In the first stage, we use progressive CoT synthesis and Supervised Fine-Tuning (SFT) to instill comprehensive capabilities. In the second, a carefully designed Reinforcement Learning (RL) phase aligns the model with human preferences. We also share key insights from our exploration of these training strategies.

Third, to ensure rigorous validation, we construct a comprehensive benchmark, RAIR, tailored to evaluate the core capabilities we identified.

Finally, to overcome the challenges of real-time computation, we designed a query-based stratified deployment strategy that comprehensively transfers the offline LLM’s ability to online system, leading to substantial online performance gains.

LORE serves as both a practical solution for developing advanced e-commerce relevance systems and a methodological reference for domain-specific post-training, with insights generalizable across vertical industries.

Contents

1  Introduction

2  Theoretical Framework

2.1  Task Definition

2.1.1  Problem Formulation

2.1.2  Fine-Grained Relevance Judgment

2.1.3  Attribute Schema for Query Demands

2.2  Deep Analysis of Relevance Modeling

2.2.1  Item Understanding

2.2.2  Query Understanding

2.2.3  Path Modeling from Query to Item

2.2.4  Core Capabilities for Relevance Judgment

3  Method

3.1  Overview

3.2  Preliminary Exploration

3.2.1  Feature Construction

3.2.2  Model Selection

3.2.3  Prompt Optimization

3.3  SFT: Comprehensive Reasoning Capability Injection

3.3.1  Data construction

3.3.2  Multi-dimensional Chain-of-Thought Synthesis

3.3.3  Distillation by SFT

3.4  RL: Relevance-Oriented Human Preference Alignment

3.4.1  Data construction

3.4.2  Basic RL Framework

3.4.3  Training Strategy Optimization

3.5  Evaluation: Comprehensive Relevance Benchmark

3.5.1  Design Principles

3.5.2  Benchmark Construction

3.5.3  Data Statistics

4  Experiment

4.1  Settings

4.1.1  Metrics

4.1.2  Baseline

4.2  Main Result

4.3  Case Study

5  Application

6  Discussion

6.1  Naive teacher CoT distillation results in negative effects.

6.2  Long CoT is not necessary for better performance.

6.3  Multimodal Modeling: VLM or Two-Stage LLM?

7  Evolutionary Trajectory

7.1  LORE 1.0: Foundation Consolidation

7.2  LORE 2.0: Deep reasoning

7.3  LORE 3.0: Rule Adherence and Multi-modality

8  Conclusion

9  Contributors

1  Introduction

Search relevance plays a pivotal role in e-commerce platforms such as JD and Taobao ( buy_2023 ) . The relevance model evaluates candidate items by determining their alignment with user queries, assigning relevance scores to filter out mismatched products ( buy3 ;  whybuy ) . This mechanism serves as a crucial foundation for enhancing user experience and search quality.

Large language models (LLMs) have exhibited remarkable capabilities and been extensively applied to relevance tasks.
While state-of-the-art(SOTA) models  ( deepseek ;  Qwen3 ;  qwen2.5vl ;  gemini2.5 )  such as GPT-5 suffer from insufficient domain knowledge and prohibitive costs, post-training ( LPFT ;  walmartllm )  smaller-scale LLMs on e-commerce data to develop domain-specific relevance experts has emerged as the predominant paradigm.

Recent advances ( walmartllm ;  LREF ;  JD2 ;  TaoSR1 )  have progressively transitioned the post-training paradigm from rudimentary classification-oriented Supervised Fine-Tuning (SFT) to sophisticated enhancement of reasoning capabilities. While existing work has modeled the Chain-of-Thought (CoT) ( CoT )  processes of LLMs from various perspectives to enhance their capabilities, significant blind spots remain.
For example, ELLM ( ELLM )  conceptualizes relevance as a process of attribute extraction and matching between queries and items. However, this approach falters in boundary cases that demand rule-based judgment. For instance, Lacking fine-grained rules for color discrimination, it might incorrectly match a ”lake blue top” with a ”sky blue top”. To address this, subsequent approach like LREF ( LREF )  and TaoSR1 ( TaoSR1 )  advance beyond simple attribute matching by explicitly modeling rule-following CoT, thereby achieving rule-awareness. Nevertheless, their reliance solely on textual information introduces a significant limitation: the inability to process visual cues. As a result, given the query ”blue top,” the models cannot confirm a match if an item’s title omits the color, even its image clearly displays a blue top.
These limitations ultimately stem from a failure to address a fundamental question that defines the capabilities required for relevance assessment:  What constitutes a comprehensive CoT representation for such tasks?

Figure 1:  Overview of the LORE theoretical framework and architecture.

To systematically address this question, we first deconstruct the relevance assessment, as detailed in Section

2  . Our analysis breaks this process down into two fundamental steps: (1)  Path Construction:  mapping query requirements to specific attributes and establishing semantic paths between queries and products; (2)  Path Following:  executing precise judgments along this path on an attribute-by-attribute basis. This two-stage decomposition allows us to pinpoint the essential capabilities that model must possess, which we summarize as follows:
(1) Knowledge and Reasoning ( deepseek ;  skywork ;  kimi1.5 ;  knowledge1 ;  knowledge2 ) . Knowledge is required to decipher domain-specific terminology in queries (e.g., ”morning C, evening A” as Vitamin C and retinol) and interpret ambiguous item details. Reasoning then leverages this