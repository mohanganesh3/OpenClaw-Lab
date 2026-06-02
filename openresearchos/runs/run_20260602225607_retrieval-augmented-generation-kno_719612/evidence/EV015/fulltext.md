[2509.04482] Energy Landscapes Enable Reliable Abstention in Retrieval-Augmented Large Language Models for Healthcare

[1] \fnm Gabriel Davis  \sur Jones

1] \orgdiv Oxford Digital Health Labs, Nuffield Department of Women’s and Reproductive Health,  \orgname University of Oxford,  \orgaddress  \city Oxford,  \country UK
2] \orgdiv OATML, Department of Computer Science,  \orgname University of Oxford,  \orgaddress  \city Oxford,  \country UK
3] \orgdiv Nuffield Department of Women’s and Reproductive Health,  \orgname University of Oxford,  \orgaddress  \city Oxford,  \country UK

Energy Landscapes Enable Reliable Abstention in Retrieval-Augmented Large Language Models for Healthcare

\fnm Ravi  \sur Shankar

\fnm Sheng  \sur Wong

\fnm Lin  \sur Li

\fnm Magdalena  \sur Bachmann

\fnm Alex  \sur Silverthorne

\fnm Beth  \sur Albert

gabriel.jones@wrh.ox.ac.uk

[

[

[

Abstract

Reliable abstention is critical for retrieval-augmented generation (RAG) systems, particularly in safety-critical domains such as women’s health, where incorrect answers can lead to harm. We present an energy-based model (EBM) that learns a smooth energy landscape over a dense semantic corpus of 2.6M guideline-derived questions, enabling the system to decide when to generate or abstain. We benchmark the EBM against a calibrated softmax baseline and a k-nearest neighbour (kNN) density heuristic across both easy and hard abstention splits, where hard cases are semantically challenging near-distribution queries. The EBM achieves superior abstention performance abstention on semantically hard cases, reaching AUROC 0.961 versus 0.950 for softmax, while also reducing FPR@95 (0.235 vs 0.331). On easy negatives, performance is comparable across methods, but the EBM’s advantage becomes most pronounced in safety-critical hard distributions. A comprehensive ablation with controlled negative sampling and fair data exposure shows that robustness stems primarily from the energy scoring head, while the inclusion or exclusion of specific negative types (hard, easy, mixed) sharpens decision boundaries but is not essential for generalisation to hard cases. These results demonstrate that energy-based abstention scoring offers a more reliable confidence signal than probability-based softmax confidence, providing a scalable and interpretable foundation for safe RAG systems.

keywords:  Large language models, RAG, model uncertainty, hallucination detection, women’s health

1  Introduction

Large language models (LLMs) coupled with retrieval-augmented generation (RAG)
 [ 1 ]  are being piloted for clinical decision support
 [ 2 ,  3 ] .
They can synthesise guidance across large corpora, yet they also generate fluent errors
when inputs fall outside scope or when retrieved evidence is sparse or misleading.
In safety-critical care, such failures erode trust and can cause harm.
Robust abstention is therefore a first-order requirement
 [ 4 ,  5 ,  6 ] :
the system should recognise when not to answer and instead expand retrieval, escalate, or defer to a human expert.

RAG is particularly promising where the knowledge corpus can be strictly controlled.
In healthcare, evidence-based guidelines, drug formularies, and institution-specific protocols can be curated, versioned, and indexed so that generation is constrained to cited sources with known provenance and update cycles.
This limits reliance on memorised web text, reduces parametric drift, enables auditability, and supports time-bounded answers.
In women’s health, for example, authoritative guidance from bodies such as RCOG, NICE, and WHO can anchor responses to accepted standards of care.
Nevertheless, reliability depends on recall and scope: if relevant material is not retrieved or the query falls outside the indexed corpus, the model may generalise beyond evidential support.
In these settings, abstention and explicit self-assessment of evidential sufficiency are critical.

In healthcare applications, two distinct classes of query should trigger abstention:
(i) domain-irrelevant prompts that fall outside healthcare expertise, for example questions related to finance or economics, where the correct behaviour is immediate redirection; and
(ii) domain-relevant but out-of-scope queries relative to the model’s training or validation, such as applying pregnancy-specific diabetes or hypertension protocols to non-pregnant adults, paediatric gynaecology when training covered only adult care, oncological queries requiring therapies, dosing, or trial evidence not present in the corpus, site-specific policies absent from the training data, or modality shifts such as image-first triage when the model was trained only on text.
These near-distribution queries are hazardous because they are semantically close to in-scope content and can elicit persuasive but unsafe answers.
Abstention should be the default unless retrieved evidence is sufficiently specific and in-distribution.

Prior work in women’s health underscores the need for principled abstention. In a head-to-head evaluation with questions from the UK Royal College of Obstetricians and Gynaecologists, ChatGPT achieved moderate accuracy on basic science but only coin-toss performance on clinical reasoning, while often expressing high confidence irrespective of correctness, indicating unreliable self-assessment  [ 7 ] . Meaning-level uncertainty signals (for example semantic entropy) offer a more discriminative path:  semantic entropy  outperformed perplexity for identifying unreliable outputs on obstetrics and gynaecology questions, and achieved expert-validated discrimination approaching ceiling, supporting the value of pre-generation uncertainty checks and deferral mechanisms  [ 8 ] . However, computing semantic entropy typically requires first generating multiple candidate responses to estimate the meaning distribution, which adds latency and compute cost.

Foundational work on abstention frames the problem as a reject option in statistical decision theory, and selective prediction formalises the coverage–risk trade-off with standard risk–coverage evaluation  [ 4 ,  9 ,  5 ,  10 ] . Post-hoc calibration and conformal prediction can support abstention, although they do not shape the representation space during training  [ 11 ,  12 ] . Out-of-distribution (OOD) detection methods include maximum softmax probability, ODIN, and Mahalanobis distance, with surveys detailing pitfalls and best practice  [ 13 ,  14 ,  15 ,  16 ,  17 ] . Energy-based models interpret predictions via an energy landscape in which in-distribution samples receive low energy, and explicit energy training has improved OOD detection; margin-based and contrastive variants further shape the energy function  [ 18 ,  19 ,  6 ] . In RAG, conditioning on retrieved passages improves factuality, yet systems can still over-commit on OOD inputs; prior work explores selective QA, self-evaluation, and retrieval-aware abstention  [ 1 ,  20 ,  21 ,  22 ] . Contrastive learning benefits from informative negative mining, widely studied in metric learning and dense retrieval  [ 23 ,  24 ,  25 ,  26 ,  27 ] . For medical QA, integrating external corpora helps calibrate a broad not-our-domain boundary; common OOD pools include MedMCQA and SQuAD  [ 28 ,  29 ,  30 ,  31 ] . Non-parametric scores such as the

k  k

th-neighbour similarity also provide strong abstention baselines when thresholds are fixed on validation and reused at test  [ 32 ,  33 ,  13 ,  14 ,  6 ] . Within this landscape, our approach trains a compact energy-based abstainer directly over dense embeddings, introduces adaptive semi-hard negative mining to expose near-domain confusables, and couples similarity and energy margins, aiming for separation on both clean and hard OOD scenarios in RAG-style QA.

We address this by training an energy-based abstention model that learns a smooth energy landscape over a dense clinical question space. The energy score functions as a calibrated confidence