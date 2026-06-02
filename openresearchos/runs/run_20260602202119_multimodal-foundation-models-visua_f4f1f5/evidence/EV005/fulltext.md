[2410.04778] MM-R3: On (In-)Consistency of Multi-modal Large Language Models (MLLMs)

MM-R 3 : On (In-)Consistency of Multi-modal Large Language Models (MLLMs)

Shih-Han Chou  1,2  , Shivam Chandhok  1,2  , James J. Little  1  , Leonid Sigal  1,2,3

1  Department of Computer Science, University of British Columbia

2  Vector Institute for AI

3  Canada CIFAR AI Chair

{shchou75, chshivam, little, lsigal}@cs.ubc.ca

Abstract

With the advent of Large Language Models (LLMs) and Multimodal (Visio-lingual) LLMs, a flurry of research has emerged, analyzing the performance of such models across a diverse array of tasks. While most studies focus on evaluating the capabilities of state-of-the-art (SoTA) MLLM models through task accuracy ( e.g. , Visual Question Answering, grounding) across various datasets, our work explores the related but complementary aspect of  consistency  – the ability of an MLLM model to produce semantically similar or identical responses to semantically similar queries. We note that consistency is a fundamental prerequisite (necessary but not sufficient condition) for robustness and trust in MLLMs.
Humans, in particular, are known to be highly consistent (even if not always accurate) in their responses, and consistency is inherently expected from AI systems. Armed with this perspective, we propose the  MM-R  3

benchmark , which analyses the performance in terms of consistency and accuracy in SoTA MLLMs with three tasks: Question Rephrasing, Image Restyling, and Context Reasoning. Our analysis reveals that consistency does not always align with accuracy, indicating that models with higher accuracy are not necessarily more consistent, and vice versa.
Furthermore, we propose a simple yet effective
mitigation strategy in the form of an adapter
module trained to minimize inconsistency across prompts.
With our proposed strategy, we are able to achieve absolute improvements of

5.7  %

percent  5.7

5.7\%

and

12.5  %

percent  12.5

12.5\%

, on average on widely used MLLMs such as BLIP-2 and LLaVa 1.5M in terms of consistency over their
existing counterparts.

Figure 1:  (Left) Overall results of MLLMs on the MM-R 3  Benchmark. (Mid) Consider answering the three semantically identical questions for the top image and a given visual abductive reasoning question for the bottom images from the proposed MM-R 3  Benchmark. Humans are accurate and consistent in these tasks while MLLMs are much less so.
(Right) Results with the proposed adapter.

1  Introduction

Multimodal Large Language Models (MLLMs)  (Liu et al.,  2023a ; Li et al.,  2023b ; OpenAI,  2023 ; Xue et al.,  2024 ) , following and often built on top of purely lingual LLM  (Brown et al.,  2020 ; Touvron et al.,  2023 ) , have recently emerged as incredible tools for a broad range of visual understanding tasks, spanning captioning  (Lin et al.,  2014 ; Sharma et al.,  2018 ; Chen et al.,  2015 ) , language grounding  (Yu et al.,  2016 ; Kazemzadeh et al.,  2014 ; Liu et al.,  2019 ) , visual question answering (VQA)  (Antol et al.,  2015 ) , and many others.
As the number of such models and their capabilities explode, the research community is progressively focusing on benchmarking their capabilities by developing new benchmarks and testing harnesses. Notable examples include MM-Bench  (Liu et al.,  2023c ) , SEED-Bench  (Li et al.,  2023a ) , MM-Vet  (Yu et al.,  2023 ) , and others that define numerous tasks that capture a broad range of capabilities of such models ( e.g. , instance counting  (Fu et al.,  2023 ) , spatial relation understanding  (Yu et al.,  2023 ) , abductive  (Hessel et al.,  2022 )  and deductive  (Park et al.,  2020 )  reasoning, meme comprehension  (Li et al.,  2023a ) ,  etc ). These benchmarks continue to shed light on the abilities and limitations of MLLMs by analyzing their  accuracy .

However, despite significant progress in the analyses of MLLM models,  consistency , the ability to produce identical or semantically equivalent outputs with the same semantic content inputs, remains broadly overlooked. This is a fundamental requirement for MLLMs to be reliably deployable for most tasks. Anecdotally, LLMs and, by extension, MLLMs are sensitive to their prompts which led to the widespread practice of prompt engineering.
This is problematic as the models’ outputs may vary with the phrasing of a query rather than its actual intent, which undermines their reliability.
Consider the example illustrated in Figure

MM-R 3 : On (In-)Consistency of Multi-modal Large Language Models (MLLMs)

(Mid) top: Most humans would realize that while the three questions (i.e R1, R2, and R3) are superficially different, the semantic meaning is the same. Hence even when the correct answer may not perhaps be apparent ( i.e. , “to be visible”), the same (consistent) answer should be produced. In contrast, asking models like BLIP-2  (Li et al.,  2023b )  to answer these questions results in varied responses “to protect them from splinters”,“to protect the horse’s legs”,“to make the hooves more visible” for the three questions considered.
Similarly, it is obvious to humans that the object being masked in Figure

MM-R 3 : On (In-)Consistency of Multi-modal Large Language Models (MLLMs)

(Mid) bottom is the same irrespective of the type of the mask, and that the object in question is a “bench”
However, Qwen-VL-Chat’s responses vary, indicating “a bench”,“a yellow object”, “a green wooden slat sign” for different masks, highlighting the inconsistency in visual modifications.

It may be tempting to equate  accuracy  and  consistency , but the relationship is more intricate.
While it is true that for objective visual tasks ( e.g. , what color is an object), high accuracy will result in high(er) consistency, current MLLM models are far from this high accuracy regime. Further, for more subjective visual tasks ( e.g. , abductive reasoning), high accuracy which tends to be measured as being similar to one of the sets of answers, may not lead to high consistency. In general, one can think of consistency as a necessary but not sufficient property of an AI system and one should seek to maximize  both

consistency  and  accuracy .

In this work, we first present a comprehensive analysis of SoTA MLLM models in terms of their consistency. We do so by developing MM-R 3  benchmark atop of the VQA task, where we produce both lingual rephrasings of the original questions (by leveraging GPT-3.5) and visual rephrasings of the image (through stylization) and measure both accuracy and consistency of the produced responses.
We find that SoTA MLLMs while often quite competitive in accuracy can differ substantially in their consistency of responses.
For example, mPLUG-Owl2  (Ye et al.,  2024 )  is much more susceptible to inconsistency when image inputs are perturbed while MoE-LLaVa  (Lin et al.,  2024 )  is more consistent in the change of the visual domain than the lingual domain.
In addition, we also define an abductive task of predicting the contents of the masked region, where we find models like BLIP-2 and LLaVa 1.5M  (Liu et al.,  2023a ;  2024 )  are lower in accuracy but have higher consistency. Overall results for MLLMs are illustrated in Figure

MM-R 3 : On (In-)Consistency of Multi-modal Large Language Models (MLLMs)

(Left). We believe these findings both benchmark the capabilities of existing models and outline
future directions and developments in more consistent MLLM models and pre-training objectives ( e.g. , see efforts in language LLMs  (Aggarwal et al.,  2023 ; Chen et al.,  2024 ; Jang &amp; Lukasiewicz,  2023 ) ).

Toward the latter goal, we propose a simple adapter module based strategy that effectively improves consistency.
The adapter is flexible and can be added to any existing MLLM. It sits between the MLLM embedding layer and the frozen LLM decoder. The goal of the adapter is to modify the LLM’s embeddings such that they are invariant to surface form variations in the language