[2508.19428] 1 Introduction

\TIBauthor

[A. Beliaeva and T. Rahmatullaev]Aleksandra Beliaeva * \affOne  \affTwo  \affThree  and Temurbek Rahmatullaev * \affFour  \affFive

\TIBaffiliations

\affOne Skolkovo Institute of Science and Technology, Russia

\affTwo

BIMAI-Lab, Biomedically Informed Artificial Intelligence Laboratory, University of Sharjah, Sharjah, United Arab Emirates

\affThree

Ivannikov Institute for System Programming of the Russian Academy of Sciences, Russia

\affFour

Lomonosov Moscow State University, Russia

\affFive

AIRI Institute, Russia

*These authors contributed equally and are joint first authors.

Correspondence: belyaeva.alex1@gmail.com, raxtemur@gmail.com
 \TIBtitle Alexbek at LLMs4OL 2025 Tasks A, B, and C:
 \TIBsubtitle Heterogeneous LLM Methods for Ontology Learning
 (Few-Shot Prompting, Ensemble Typing, and Attention-Based Taxonomies)
 \TIBbundlename [ConfAbbrev]Full conference name
 \TIBbundlename [Open Conf Proc X (2025) "LLMs4OL 2025: The 2nd Large Language Models for Ontology Learning Challenge at the 24th ISWC"]LLMs4OL 2025: The 2nd Large Language Models for Ontology Learning Challenge at the 24th ISWC
 \TIBconferencesession LLMs4OL 2025 Task Participant Papers \TIBabstract We present a comprehensive system for addressing Tasks A, B, and C of the LLMs4OL 2025 challenge, which together span the full ontology construction pipeline: term extraction, typing, and taxonomy discovery. Our approach combines retrieval-augmented prompting, zero-shot classification, and attention-based graph modeling — each tailored to the demands of the respective task.
For Task A, we jointly extract domain-specific terms and their ontological types using a retrieval-augmented generation (RAG) pipeline.
Training data was reformulated into a document to terms and types correspondence, while test-time inference leverages semantically similar training examples. This single-pass method requires no model finetuning and improves overall performance through lexical augmentation
Task B, which involves assigning types to given terms, is handled via a dual strategy. In the few-shot setting (for domains with labeled training data), we reuse the RAG scheme with few-shot prompting. In the zero-shot setting (for previously unseen domains), we use a zero-shot classifier that combines cosine similarity scores from multiple embedding models using confidence-based weighting.
In Task C, we model taxonomy discovery as graph inference. Using embeddings of type labels, we train a lightweight cross-attention layer to predict is-a relations by approximating a soft adjacency matrix.
These modular, task-specific solutions enabled us to achieve top-ranking results in the official leaderboard across all three tasks. Taken together these strategies showcase the scalability, adaptability, and robustness of LLM-based architectures for ontology learning across heterogeneous domains.
Code is available at:  https://github.com/BelyaevaAlex/LLMs4OL-Challenge-Alexbek

\TIBkeywords Large Language Models, Ontology engineering (OE), Ontology Learning, Domain-specific knowledge, Retrieval Augmented Generation, Term Typing, Taxonomy Discovery

1  Introduction

Large Language Models (LLMs) have recently emerged as powerful tools for ontology learning (OL) tasks, including term extraction, typing, and taxonomy discovery. Prior studies have demonstrated that, when combined with retrieval-augmented generation (RAG) and in-context learning techniques, LLMs can act as general-purpose engines for structured knowledge extraction, even in the absence of task-specific finetuning  [ 1 ,  2 ] . Three complementary strategies have proven especially effective in this context: (i) few-shot prompting with curated examples, (ii) embedding-based similarity search, and (iii) zero-shot classification using pretrained models.

A notable illustration of this paradigm is the LLMs4OL framework  [ 1 ] , which applies zero-shot prompting across three core OL subtasks—term typing, taxonomy induction, and non-taxonomic relation extraction—evaluated over diverse domains. The results suggest that, while base LLMs exhibit strong baseline capabilities, task-specific finetuning significantly improves performance. Building on this foundation, the LLMs4OL 2024 Challenge introduced a standardized benchmark encompassing few-shot and retrieval-augmented settings across five domains and a set of structured tasks (A/B/C)  [ 3 ,  2 ] .

Extending these ideas further, OLLM  [ 4 ]  reframes OL as an end-to-end fine-tuned process, integrating term and taxonomy learning into a unified architecture. Through the use of a custom regularizer and novel graph-aware evaluation metrics, OLLM achieves state-of-the-art results on both general (Wikipedia) and specialized (arXiv) corpora. Together, these developments reflect a shift in the field—from isolated prompting strategies toward integrated, data-grounded pipelines that scale across domains.

Complementing this trend, domain-specific approaches like LLMs4Life  [ 5 ] 
show how careful prompt engineering and ontology reuse yield richer hierarchies and more consistent ontologies in specialized domains.

Despite these advances, prior systems often rely on heavy finetuning or manually designed prompts, and lack a unified, efficient pipeline that works across tasks and domains.
Our aim is to design a modular, lightweight, and scalable system that tackles all three OL tasks of the LLMs4OL 2025 benchmark:

•

Task A: Text2Onto  — joint extraction of terms (A1) and their types (A2) from raw domain documents.

•

Task B: Term Typing  — assigning types to given terms, evaluated in both  few-shot  (B1–B3) and  zero-shot  (B4–B6) settings.

•

Task C: Taxonomy Discovery  — inducing  is-a  relationships between types to construct hierarchical taxonomies (C1–C8).

We target a single unified framework that avoids full encoder finetuning, yet remains competitive.

Our contributions are threefold:

1.

We present a unified, modular LLM-based system for ontology learning, covering term extraction, term typing, and taxonomy induction across Tasks A, B, and C of the LLMs4OL 2025 challenge.

2.

We introduce a simple yet effective  dedicated cross-attention layer  for taxonomy discovery, which enables efficient graph inference over type embeddings without full encoder finetuning.

3.

Our method achieves competitive results across all subtasks, including top-2 placement in multiple domains and strong zero-shot generalization. It is fully task-agnostic, lightweight, and requires no large-scale finetuning.

This work challenges the prevailing notion that OL requires heavy finetuning or complex multi-component pipelines. We show that a lean, modular system—built on efficient prompting and adaptive lightweight modules—can compete with or outperform more expensive methods, supporting scalable, domain-agnostic ontology learning in real-world settings.

2  Methodology

2.1  Task A: Text2Onto

Task A entails concurrent extraction of  terms 
(SubTask A1) and their  types  (SubTask A2) from raw documents
spanning the  Ecology ,  Scholarly , and
 Engineering  domains.
We resolve both subtasks in a single pass by
(i) curating the released data and
(ii) employing retrieval-augmented few-shot prompting.

Dataset inspection and repair

The official distribution comprises the files:
 documents.jsonl ,  terms.txt ,  types.txt ,
 terms2docs.json , and  terms2types.json . A manual audit revealed that
 terms2docs.json  actually maps  types  to document
IDs. To assess the issue, we measured term–type overlap across domains (see Fig.

1  ).

Figure 1 :

Normalized term–type intersection across domains.
Overlap is near-zero in  Ecology  and  Engineering ,
indicating weak supervision, while  Scholarly  shows strong alignment,
with nearly half of terms and types intersecting.

To correct the issue, we rescanned documents for exact matches from  terms.txt , reconstructing a term–document index.
Merging it with  terms2ty