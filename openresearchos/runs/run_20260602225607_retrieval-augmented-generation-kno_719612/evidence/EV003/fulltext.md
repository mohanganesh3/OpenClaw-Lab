[2512.19122] BanglaForge: LLM Collaboration with Self-Refinement for Bangla Code Generation

BanglaForge: LLM Collaboration with Self-Refinement for Bangla Code Generation

Mahir Labib Dihan, Sadif Ahmed, Md Nafiu Rahman

Department of Computer Science and Engineering

Bangladesh University of Engineering and Technology (BUET)

Dhaka, Bangladesh

{mahirlabibdihan, ahmedsadif67, nafiu.rahman}@gmail.com

Abstract

Bangla is a low-resource language for code generation, lacking large-scale annotated datasets and tools to transform natural language specifications into executable programs. This makes Bangla-to-code generation a challenging task requiring innovative solutions. To address this, we introduce  BanglaForge , a novel framework for generating code from Bangla function descriptions.
BanglaForge leverages a retrieval-augmented dual-model collaboration paradigm with self-refinement, combining in-context learning, llm-based translation, systematic prompt engineering, and iterative self-refinement based on execution feedback, where a coder generates initial solutions and a reviewer enhances them for robustness. On the  BLP-2025 Bangla Code Generation  benchmark, BanglaForge achieves a competitive Pass@1 accuracy of  84.00% , demonstrating the effectiveness of retrieval, model collaboration, and self-refinement for low-resource Bangla code generation.

BanglaForge: LLM Collaboration with Self-Refinement for Bangla Code Generation

Mahir Labib Dihan, Sadif Ahmed, Md Nafiu Rahman

Department of Computer Science and Engineering

Bangladesh University of Engineering and Technology (BUET)

Dhaka, Bangladesh

{mahirlabibdihan, ahmedsadif67, nafiu.rahman}@gmail.com

1  Introduction

Large language models (LLMs) have shown strong capabilities in code generation, where natural language descriptions are automatically transformed into executable programs. Models such as Codex, CodeT5, and StarCoder, trained on large-scale code–text corpora, can produce syntactically valid and semantically correct solutions, performing well on benchmarks like HumanEval  Chen et al. ( 2021 ) . These advances reduce the gap between human intent and code, making programming more accessible. However, most existing systems are designed for English inputs, leaving low-resource languages underserved. Models often struggle with informal structures, domain-specific terms, and semantic nuances, resulting in incorrect or brittle outputs.

We introduce  BanglaForge , a framework for generating executable code from Bangla task descriptions. Each input is represented as a triple: the Bangla description, its English translation, and unit test assertions. This structure leverages the model’s stronger English understanding while retaining Bangla context. BanglaForge combines retrieval-augmented prompting, iterative self-refinement with execution feedback, and a dual-model coder–reviewer pipeline. Our system achieves a  Pass@1 accuracy of 84%  on  BLP-2025 Bangla Code Generation Benchmark

Raihan et al. ( 2025c ) , demonstrating the potential of practical low-resource code generation.

Our contributions can be summarized as follows:

•

A retrieval-augmented few-shot prompting approach using TF-IDF to select relevant Bangla–Python pairs, improving in-context learning despite limited labeled data.

•

A LLM-based translation component that translates Bangla instructions into English with the help of a glossary to enable accurate cross-lingual code generation.

•

An iterative self-refinement protocol that leverages execution feedback to detect and correct errors across refinement cycles.

•

A dual-model architecture where a generator model focuses on functional correctness and a reviewer model enhances robustness, style, and coverage of edge cases.

We release our implementation of BanglaForge at  https://github.com/mahirlabibdihan/BanglaForge  to facilitate reproducibility and further research.

Figure 1:  Workflow of the proposed  BanglaForge

framework.
A Bangla instruction (

P  b

P_{b}

) is translated into English (

P  e

P_{e}

) and, together with unit tests, used to retrieve top-

k  k

bilingual examples.
The  Coder LLM  then generates Python code and additional test cases.
The  Reviewer LLM  validates, refines, and re-prompts upon errors until all tests (original and generated) are passed, yielding the final code.

2  Related Works

Research in Bangla NLP has evolved from early word embeddings to specialized LLMs. Initial efforts such as BnVec introduced embeddings like fastText, Word2Vec, and GloVe trained on diverse corpora, with customized fastText outperforming multilingual baselines in classification tasks  Kowsher et al. ( 2021 ,  2022 ); Mojumder et al. ( 2020 ) . Recent advances include Bangla LLMs and benchmarks such as TigerCoder  Raihan et al. ( 2025b )  and BanglaByT5  Bhattacharyya and Bhattacharya ( 2025 ) , which advanced code generation and tokenization strategies. However, existing work largely focuses on pretraining and benchmarking without complete generation pipelines. Our work addresses these gaps by introducing retrieval-augmented prompting, iterative self-refinement, and a dual generator–reviewer design. A detailed discussion is provided in Appendix

A  .

3  Dataset

We build on the resources introduced for Bangla code generation across recent shared tasks and benchmarks. Our dataset comes from the Bangla Code Generation shared task (Task 2) at BLP-2025  Raihan et al. ( 2025c ) , where the objective is to translate Bangla natural language programming prompts into Python functions that satisfy hidden unit tests. The dataset is distributed through an official starter kit  1

1  1  https://noshinulfat.github.io/blp25_code_generation_task/#/get-started

, which also provides baseline code and evaluation scripts.

Each entry is a JSON object containing four fields: an  id , a Bangla instruction describing the task, a  response  field with the reference Python implementation (training only), and a  test_list  field of assert-based unit tests.

Figure 2:  Example data point

Split

Purpose

Size

Trial

Initial experiments

74

Development

Validation

400

Test

Final evaluation

500

Table 1:  Dataset Split Statistics for Bangla Code Generation

For development and testing, we adopt two external Bangla code generation benchmarks. The  mHumanEval-Bangla  dataset  Raihan et al. ( 2025a ) , a Bangla extension of HumanEval, is used during the development phase, enabling programmatically testable evaluation on held-out prompts. The  MBPP-Bangla  dataset  Raihan et al. ( 2025b ) , adapted from MBPP as part of the TigerCoder framework, is used during both development and test phases, providing diverse programming problems in Bangla with associated unit tests.

4  Methodology

We propose  BanglaForge , a retrieval-augmented dual-LLM framework for generating Python code from Bangla natural language specifications. The system tackles low-resource code generation through structured prompt design, bilingual translation, example retrieval, and a two-stage generation-review process involving a  Coder LLM  and a  Reviewer LLM . Together, these components ensure both functional correctness and stylistic reliability, even in underrepresented languages like Bangla. An overview of the complete workflow is shown in Figure

1

and in Algorithm

1

(Appendix). Each stage is described in detail below.

4.1  Problem Formulation and Input Representation

Each task in the dataset consists of a Bangla instruction

P  b

P_{b}

and its corresponding public unit tests

T  =

{

t  1

,  …  ,

t  n

}

T=\{t_{1},\dots,t_{n}\}

. To enable code synthesis, the instruction is translated into English using a translation model equipped with a controlled glossary for mathematical and algorithmic terms (e.g., GCD, LCM, sum). The glossary is curated by the authors which was motivated from the provided dataset and commonly seen technical terms in code related works. The translated instruction

P  e

P