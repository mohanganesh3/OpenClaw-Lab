[2510.22993] Can Language Models Compose Skills In-Context?

Can Language Models Compose Skills In-Context?

Zidong Liu

zidongliu@connect.hku.hk . The University of Hong Kong.

Zhuoyan Xu

zhuoyan.xu@wisc.edu . University of Wisconsin-Madison.

Zhenmei Shi

zhmeishi@cs.wisc.edu . University of Wisconsin-Madison.

Yingyu Liang

yingyul@hku.hk . The University of Hong Kong.

Abstract

Composing basic skills from simple tasks to accomplish composite tasks is crucial for modern intelligent systems. We investigate the  in-context composition  ability of language models to perform composite tasks that combine basic skills demonstrated in in-context examples. This is more challenging than the standard setting, where skills and their composition can be learned in training. We conduct systematic experiments on various representative open-source language models, utilizing linguistic and logical tasks designed to probe composition abilities. The results reveal that simple task examples can have a surprising  negative impact  on the performance, because the models generally struggle to recognize and assemble the skills correctly, even with Chain-of-Thought examples. Theoretical analysis further shows that it is crucial to align examples with the corresponding steps in the composition. This inspires a method for the probing tasks, whose improved performance provides positive support for our insights.

1  Introduction

Recent advances in machine learning have yielded substantial progress, particularly with the rise of language models (e.g.,  (Ope,  23 ; Ant,  24 ; DA,  25 ) ). These models exhibit strong in-context learning (ICL) capacity: they can adapt to novel tasks by leveraging a few examples provided at inference time without requiring parameter updates.
Through ICL, language models can generalize across tasks by adapting to the given context.
A critical aspect of this task generalization is the ability to integrate basic skills from simple tasks to perform more complex composite ones, which is essential given the exponential number of possible compositions that prevents learning each individually. Ideally, models should be able to compose skills demonstrated in-context to tackle new compositions. This leads to our central question:  Can language models do composition in-context?

This study proposes to study the  in-context composition  ability of language models. Specifically, it examines whether models can solve queries for novel composite tasks combining several simple tasks, when provided with some in-context examples from the simple tasks and some examples from the composite task. Unlike traditional scenarios where the skills and potentially some of their compositions are learned during training, the models in this study need to learn novel skills and compositions during inference, thus demanding strong compositional generalization.

We first perform systematic empirical studies on representative language models  (TLI  +  ,  23 ; TMS  +  ,  23 ; KOB  +  ,  21 ; GDJ  +  ,  24 ; GYZ  +  ,  25 ) , using linguistic and logical tasks from  XSL ( 24 )  designed for probing the composition abilities. The experiments show a surprising phenomenon:  simple task examples can hurt the performance on composite queries, rather than improve it . See an illustration in

Fig. ˜ 1  . This is in sharp contrast to the expectation that these examples can help the model identify skills and compose them to solve the query. Our investigation of this negative impact finds that the models generally do not recognize the composition and do not align the simple task examples with the corresponding steps of the composition. Even when Chain-of-Thought (CoT) examples are used, they may mismatch the skills inferred from examples to the wrong steps in answering the composite query. Inspection into the inner attentions of the language models provides further evidence for our findings.

We further provide a theoretical analysis in a stylized setting that captures the essence of the in-context composition and focuses on understanding the key factor behind the observations. The analysis confirms that ignorance of the compositional structure can harm the performance, while  aligning the examples to appropriate steps of the composition  can potentially improve it.
This inspires a proof-of-concept algorithm for the probing tasks, Expanded Chain-of-Thought (ExpCoT), that views simple task examples as composite task examples with missing steps and expands them into the CoT format with missing steps marked by special symbols.
Evaluations show that it can explicitly align the examples with the corresponding steps and thus improve the performance. The improvement verifies our insights and justifies the potential for helping future algorithm designs.

(a)

Figure 1:

The negative impact of simple task examples on the opposition+swap task (see

Table ˜ 1  ). The models need to answer a composite query when given

k  k

examples from each simple task and

k  c

=  5

k_{c}=5

examples from the composite task. They show unexpected  decreasing  performance with more simple task examples (

k  k

).

Our contributions are summarized as follows.

•

We perform systematic experiments investigating the in-context composition ability of representative open-source language models and find that they typically exhibit limited such ability, due to difficulties in recognizing the composition and identifying proper skills from in-context examples.

•

We provide a theoretical analysis, which explains the empirical observations and reveals that explicitly aligning in-context examples with the corresponding steps can help accomplishing the composite tasks.

•

We propose a proof-of-concept method that significantly improves the in-context composition performance on the probing tasks, which provides positive support for our insights.

2  Related Work

In-Context Learning and Chain-of-Thought. 
Several studies investigate the behavior of in-context learning (ICL).
 ZWF  +

( 21 ); LBM  +

( 22 ); MLH  +

( 22 ); WWT  +

( 23 )  analyze the sensitivity of LLMs to in-context examples.
 RHB ( 22 ); LSZ  +

( 22 ); HKW  +

( 23 ); WZS  +

( 23 )  propose methods to effective selection of in-context learning examples.
 GTLV ( 22 ); VONR  +

( 23 ); ASA  +

( 23 ); MHM ( 23 ); ZFB ( 23 ); SWXL ( 24 )  investigate with linear models, showing how transformers can represent gradient descent and conduct linear regression.  GHM  +

( 24 )  provide analysis on how ICL works in non-linear functions.
Chain-of-Thought (CoT) prompts LLMs to produce intermediate reasoning steps to solve multi-step reasoning questions  KGR  +

( 22 ); WWS  +

( 22 ) .
Few-Shot-CoT improves LLM’s reasoning ability using demonstrations that are either manually constructed

KTF  +

( 22 ); ZSH  +

( 23 ); LLZ  +

( 23 ); WWS  +

( 23 )  or automatically selected  ZZLS ( 23 ) .
Several theoretical work have been proposed to analyze the effecitiveness of CoT.  FZG  +

( 23 ); LLZM ( 24 )  shows CoT allows for performing more serial computations, increasing the effective depth of a transformer.  JVB  +

( 25 )  presents a frameworks that allows for universal representability and computationally tractable CoT learning and  AZW  +

( 25 )  analyzes the task generalization enabled by composition. Our theoretical analysis is partially inspired by  JVB  +

( 25 ); AZW  +

( 25 )  but considers a different setting with both simple/composite task examples, and also analyzes when the composition can fail.

Compositional Task Learning. 
Compositional reasoning of LLM is an active area in AI  HC ( 22 ); SPK ( 24 ) .  KL ( 20 ); LBB ( 22 )  explore the compositional capabilities of LLMs in abstract reasoning tasks under ICL settings.  ALC  +

( 23 ); ALF  +

( 23 )  show LLMs are capable of learning abstract reasoning (e.g., grammar) to perform new tasks when finetuned or given in-context examples.  YWF  +

( 23 ); DLS  +

( 23 ); TCT  +

( 24 ); XSL ( 24 )  show