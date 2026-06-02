[2405.19567] Dr-LLaVA: Visual Instruction Tuning with Symbolic Clinical Grounding

Dr-LLaVA : Visual Instruction Tuning with
 Symbolic Clinical Grounding

Shenghuan Sun ∗

UCSF
 &amp;Gregory M. Goldgof ∗

MSK Cancer Center
 &amp;Alexander Schubert
 UC Berkeley &amp; UCSF
 &amp;Zhiqing Sun
 CMU
 &amp;Thomas Hartvigsen
 University of Virginia
 &amp;Atul J. Butte
 UCSF
 &amp;Ahmed Alaa
 UC Berkeley &amp; UCSF

Abstract

Vision-Language Models (VLM) can support clinicians by analyzing medical images and engaging in natural language interactions to assist in diagnostic and treatment tasks. However, VLMs often exhibit “hallucinogenic” behavior, generating textual outputs not grounded in contextual multimodal information. This challenge is particularly pronounced in the medical domain, where we do not only require VLM outputs to be accurate in single interactions but also to be consistent with clinical reasoning and diagnostic pathways throughout multi-turn conversations. For this purpose, we propose a new alignment algorithm that uses  symbolic representations  of clinical reasoning to ground VLMs in medical knowledge. These representations are utilized to  (i)  generate GPT-4-guided visual instruction tuning data at scale, simulating clinician-VLM conversations with demonstrations of clinical reasoning, and  (ii)  create an automatic reward function that evaluates the clinical validity of VLM generations throughout clinician-VLM interactions. Our algorithm eliminates the need for human involvement in training data generation or reward model construction, reducing costs compared to standard reinforcement learning with human feedback (RLHF). We apply our alignment algorithm to develop  Dr-LLaVA , a conversational VLM finetuned for analyzing bone marrow pathology slides, demonstrating strong performance in multi-turn medical conversations.

Code:

https://github.com/AlaaLab/Dr-LLaVA

1  Introduction

Large language models (LLMs) exhibit impressive capabilities such as generating human-like dialogue and answering complex questions  [ 1 ,  2 ,  3 ,  4 ,  5 ] . When combined with vision encoders and fine-tuned using text-image pairs or visual instruction tuning datasets, these models evolve into (large) vision-language models (VLMs)  [ 6 ,  7 ,  8 ] , which can conduct tasks involving visual reasoning and grounding  [ 9 ] . In the medical domain, VLMs hold great promise—they could serve as helpful assistants for clinicians, researchers, and trainees, providing an interactive natural language interface for the analysis of medical images within clinical workflows  [ 10 ,  11 ,  12 ,  13 ,  14 ,  15 ] . Despite their promise, VLMs are limited by their tendency to “hallucinate”  [ 16 ,  17 ,  18 ] . As noted in  [ 18 ,  7 ] , the significantly smaller amount of multimodal training data compared to text-based data can cause a misalignment between vision and language inputs, leading VLMs to generate answers that are ungrounded in the visual context. This issue is even more pronounced in the medical domain, where data is naturally scarce and clinician-VLM image-based dialogues are harder to synthesize. The problem of finetuning VLMs to mitigate hallucinatory responses in the multimodal medical context remains largely unaddressed.

Reinforcement Learning from Human Feedback (RLHF) has been shown to be a promising method to reduce hallucinations and align LLM outputs with human preferences  [ 19 ,  20 ,  21 ] . In RLHF, human annotators evaluate the quality of various LLM outputs, and this feedback is used to train a reward model, which is then employed to finetune the LLM. In  [ 18 ] , this approach was adapted for finetuning VLMs by collecting human feedback on their responses in synthesized single-turn visual QA tasks.

However, finetuning VLMs for visually-grounded diagnostic assistance involves more than just handling single-turn QA tasks and addressing the standard notion of hallucination. Instead, we seek models capable of engaging in multi-turn interactions with clinicians, consistently demonstrating  valid  and  coherent  clinical reasoning throughout the conversation. In this context, hallucination occurs not only when a model provides responses not grounded in the visual input but also when its responses during the interaction are incoherent, contradictory, or fail to logically align with diagnostic pathways and domain knowledge. For instance, in the exemplary conversation in Fig.

1  , hallucination resulting from poor visual grounding will reflect in the model failing to identify the proliferating cell types in the provided pathology slide. However, even if the model correctly identifies the cell types, it must also answer follow-up questions from clinicians in a manner consistent with domain knowledge and its own previous responses.

Using RLHF to align VLMs with visually-grounded clinical reasoning requires multimodal training data showcasing the reasoning process within multi-turn QA dialogues. These datasets are not readily available in health systems.

Figure 1:

Symbolic clinical grounding.  We utilize symbolic representations of diagnostic workflows to align multi-turn clinician-VLM conversations with valid clinical reasoning.

Synthesizing these datasets and collecting clinician feedback on VLM responses is bottle-necked by the expertise of medical professionals. Unlike the LLaVA-RLHF model in  [ 18 ] , which gathered human feedback from non-expert crowdworkers for simple, common-sense visual QA tasks, this process cannot be scaled without the involvement of clinicians. Due to these limitations, specialized medical VLMs like LLaVA-Med  [ 10 ]  and PathChat  [ 7 ]  have been confined to supervised finetuning, relying on automatically generated QA tasks using image captions. Moreover, both existing general-purpose and medical VLMs have only been finetuned for single-turn QA, rather than for multi-turn conversations that convey complex and interactive clinical reasoning.

Summary of contributions.  In this paper, we introduce a novel finetuning algorithm for adapting VLMs to imaging-based conversational diagnostic tasks, which eliminates the need for human involvement in training data generation or feedback collection. The core idea of our algorithm is to represent the clinical reasoning process for specific diagnostic tasks using a predefined set of  symbolic rules . For instance, in the example shown in Fig.

1  , these symbolic rules can enumerate all possible cell types and corresponding diagnoses for blood malignancies. Our algorithm leverages these symbolic rules to automatically generate realistic VLM-clinician conversations and to evaluate a reward function that assesses the alignment of VLM responses with these rules. Our algorithm comprises three steps:

Step 1 (Synthesizing clinician-VLM conversations):  Starting with labeled medical images, we use symbolic representations of clinical reasoning and GPT-4 to generate realistic conversations between a VLM and a clinician about the visual content of each image. These multi-turn conversations are designed to reflect various styles of clinician-VLM interactions, with each conversation comprising a sequence of related questions that demonstrate accurate clinical reasoning.

Step 2 (Designing clinically-informed symbolic rewards  1

1

1

In our framework, the term “symbolic reward” refers to the alignment between a sequence of model responses and

symbolic

rules that represent valid clinical reasoning. This usage differs from that in

[  18  ]

, where the term “symbolic reward” was used to describe a correctness penalty and length penalty in a different context.

):  We use the training data generated in Step 1 to finetune VLMs. Instead of relying on human feedback to ensure the model responses align with correct clinical reasoning, we use the same symbolic representations of clinical reasoning used in generating the training data to evaluate a  symbolic reward . This approach allows us to a