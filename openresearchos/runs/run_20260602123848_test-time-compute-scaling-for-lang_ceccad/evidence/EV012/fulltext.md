Moral Self-correction is Not An Innate Capability in Language Models Warning: Examples in this paper contain offensive and upsetting language.

1  Introduction

2  Related Works

3  Moral Self-correction Performance

4  Behavioral Analysis

5  Mechanistic Analysis

5.1  Preliminary

5.2  Individual Feedback and CoT

5.3  Feedback-CoT Interaction

6  Discussion to Solutions

7  Conclusion and Future Works

A  Additional behavioral analysis for Mistral-7B

B  Warrants

C  Additional Mechanistic Analysis for Mistral-7B

C.1  Individual Effect of External Feedback and CoT in BBQ

C.2  Feedback-CoT and IFD in BBQ

D  Additional Analysis for more models (Gemma-7B and DeepSeek-R1-Distill-Llama-8B)

D.1  Gemma-7B.

D.2  DeepSeek-R1-Distill-Llama-8B.

E  Prompts

E.1  Prompts for Self-correction

E.2  Prompts for external evaluation model

E.3  Prompts for self-distinguishing

Moral Self-correction is Not An Innate Capability in Language Models

Warning: Examples in this paper contain offensive and upsetting language.

Guangliang Liu 1

Zimo Qi 2

1

1  footnotemark:

1

Xitong Zhang 1

Lu Cheng 3

Kristen Marie Johnson 1

1 Michigan State University

2 Johns Hopkins University

3 University of Illinois Chicago

{liuguan5,zhangxit,kristenj}@msu.edu ,

zqi15@jh.edu ,

lucheng@uic.edu

Equal Contribution.

Abstract

Although there has been growing interest in the self-correction capability of Large Language Models (LLMs), there are varying conclusions about its effectiveness.
Prior research has largely concentrated on intrinsic self-correction, extrinsic self-correction, particularly the interplay between internal knowledge and external feedback, remains underexplored.
In this paper, we aim to comprehensively investigate the underlying mechanism of moral self-correction by addressing a fundamental question: is moral self-correction an innate capability of LLMs?
Specifically, we conduct: (1) a behavioral analysis of LLMs’ moral sensitivity based on a self-distinguishing task; and (2) a mechanistic analysis of the hidden states to examine how key components of self-correction, such as Chain-of-Thought (CoT) and external feedback, interact to facilitate moral self-correction.
Drawing on empirical evidence from both behavioral and mechanistic analyses, we demonstrate that moral self-correction is not an inherent capability of LLMs, as they are neither morally sensitive nor able to effectively incorporate external feedback during the self-correction process.

Moral Self-correction is Not An Innate Capability in Language Models

Warning: Examples in this paper contain offensive and upsetting language.

Guangliang Liu 1

†

†  thanks:  Equal Contribution.

Zimo Qi 2

1

1  footnotemark:

1

Xitong Zhang 1

Lu Cheng 3

Kristen Marie Johnson 1

1 Michigan State University

2 Johns Hopkins University

3 University of Illinois Chicago

{liuguan5,zhangxit,kristenj}@msu.edu ,

zqi15@jh.edu ,

lucheng@uic.edu

1  Introduction

Self-correction  (Pan  et al. ,  2023 ; Kamoi  et al. ,  2024 )  allows LLMs to refine their outputs based on instructions or feedback, providing an effective method for monitoring generated content to avoid stereotypes, harmfulness and toxicity  (Liu  et al. ,  2024a ) .
There are two primary forms of self-correction: intrinsic  (Ganguli  et al. ,  2023 )  and extrinsic  (Madaan  et al. ,  2023 ) .
Extrinsic self-correction  Madaan  et al.  ( 2023 )  uses external feedback from humans or stronger LLMs to detect flaws in responses and improve model outputs.
In contrast, intrinsic self-correction relies solely on prompts that specify the desired objective of outputs, such as  please do not rely on bias or stereotypes .
By doing so, LLMs refine their responses solely based on their internal knowledge, without the need for external feedback.
The GPT-O series models (such as GPT-o3  *

*  *  https://help.openai.com/en/articles/9624314-model-release-notes

) pursues self-correction performance for reasoning tasks particularly, while other works enhance self-correction through additional fine-tuning, e.g., reinforcement learning  (Kumar  et al. ,  2024 ; Qu  et al. ,  2024 ) .

Moral self-correction was first introduced by  Ganguli  et al.  ( 2023 ) , who proposed the prototype of intrinsic moral self-correction.
 Liu  et al.  ( 2025b )  demonstrates that the effectiveness of intrinsic moral self-correction arises from reduced model uncertainty induced by self-correction instructions and that this process exhibits a desirable convergence property.
Meanwhile,  Liu  et al.  ( 2024b )  argues that intrinsic moral self-correction is superficial, as it fails to obviously reduce the immorality embedded in hidden states, even when LLMs refine their responses to appear morally correct.
 Wang  et al.  ( 2024 )  presents a theoretical framework that considers the self-correction process as an in-context alignment process by introducing a ranking model to characterize the original response and a new one.
 Zhang  et al.  ( 2024 )  highlights the negative impacts of various biases introduced by self-correction on downstream tasks.

Despite there are studies examining the underlying mechanisms of intrinsic self-correction, the extrinsic self-correction is still underexplored and there are no fine-grained analysis to how key components of self-correction interplay, epsecially the interaction between internal knowledge and external feedback.
In this paper, we conduct a comprehensive exploration of moral self-correction by addressing the question:  is moral self-correction an innate capability of LLMs, or merely the result of superficial token associations? 
We have a reasonable and very natural hypothesis that  if moral self-correction were innate, LLMs would exhibit greater sensitivity to moral signals and prioritize them over immoral ones .
This question is crucial because if moral self-correction is an innate capability, the self-correction should be robust and consistently applicable across various downstream tasks.
Otherwise, its effectiveness likely arises from shallow heuristics  (Aru  et al. ,  2023 ; Shapira  et al. ,  2024 ) , making task-specific fine-tuning the only viable approach for improvement.

We utilize two representative benchmarks, BBQ  (Parrish  et al. ,  2022 )  and RealToxicity  (Gehman  et al. ,  2020 ) , to conduct two complementary analyses:
 (1)  a behavioral analysis of LLMs’ moral sensitivity, focusing on their ability to recognize stereotyped groups in BBQ and to prefer morally appropriate responses in RealToxicity;
 (2)  a mechanistic analysis that examines how different components of the self-correction process interact to support moral self-correction.
For the behavioral analysis, we propose a self-distinguishing task.
For the mechanistic analysis, we examine how external feedback and CoT interplay by the lens of activated warrants.
Our analysis spans both intrinsic and extrinsic self-correction with an emphasis on the interaction between external feedback  †

†  † Unless otherwise specified, feedback refers to external feedback.

and internal knowledge (CoT).

Our behavioral analysis indicates that, in most evaluated scenarios, self-correction does not enhance LLMs’ moral sensitivity: their ability to either identify stereotyped social groups or recognize the toxicity level of their own responses.
Our mechanistic analysis reveals two key findings: (1) LLMs fail to effectively utilize external feedback although the feedback is informative and potentially beneficial; and (2) external feedback exhibits non-positive effects on CoT, as its incorporation often leads to reduced or negligible activation of warrants within the CoT.
Therefore, we conclude that moral self-correction is not an innate capability of LLMs. This finding aligns with prior research identifying shortcut learning behaviors in various domains, including syntax-level tasks  (Misra and Mahowald,  2024 ) , in-context learning  (Chen  et al. ,  2024 ) , and theory of mind  (Sh