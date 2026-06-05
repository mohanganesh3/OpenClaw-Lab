[2602.02780] A Study of Adaptive Modeling Towards Robust Generalization

A Study of Adaptive Modeling Towards Robust Generalization

Zihao Jing

Qiuhao Zeng

Ruiyi Fang

Yan Yi Li

Yan Sun

Boyu Wang

Pingzhao Hu

Abstract

Large language models (LLMs) increasingly support reasoning over biomolecular structures, but most existing approaches remain modality-specific and rely on either sequence-style encodings or fixed-length connector tokens for structural inputs. These designs can under-expose explicit geometric cues and impose rigid fusion bottlenecks, leading to over-compression and poor token allocation as structural complexity grows.
We present a unified all-atom framework that grounds language reasoning in geometric information while adaptively scaling structural tokens. The method first constructs variable-size structural patches on molecular graphs using an instruction-conditioned gating policy, enabling complexity-aware allocation of query tokens. It then refines the resulting patch tokens via cross-attention with modality embeddings and injects geometry-informed tokens into the language model to improve structure grounding and reduce structural hallucinations.
Across diverse all-atom benchmarks, the proposed approach yields consistent gains in heterogeneous structure-grounded reasoning. An anonymized implementation is provided in the supplementary material.

Machine Learning, ICML

1  Introduction

Background. 
Multimodal large language models (LLMs) increasingly extend to reason over non-linguistic modalities, including vision, video, and scientific structures.
Beyond vision-language models (e.g., Flamingo  (Alayrac  et al. ,  2022 )  and LLaVA  (Lin  et al. ,  2024 ) ), recent efforts also target scientific domains, such as Galactica  (Taylor  et al. ,  2022 )  for scientific text, BioGPT  (Luo  et al. ,  2022 )  for biomedical text, and structure-aware models that operate on microscopic geometry, including Mol-Llama  (Kim  et al. ,  2025 )  for molecules, ProtLLM  (Zhuo  et al. ,  2024 )  for proteins, and ChatNT  (de Almeida  et al. ,  2025 )  for nucleic acids.

Table 1 :

Functional group hallucination test on 200 molecules and proteins from dataset GEO-AT.
Metrics HR: hallucination rate; HPM: hallucinations per molecule; and AR: answer rate. (S) denotes the model’s sequence-only variant (with tokenizer enhanced). The preferred variant is highlighted in pink for clarity.

Molecule

Protein

Model

HR (HPM)

↓  \downarrow

AR

↑  \uparrow

Model

HR (HPM)

↓  \downarrow

AR

↑  \uparrow

Mol-Llama(S)

0.28 (0.91)

0.95

ProtChatGPT(S)

0.34 (0.99)

\cellcolor  [HTML]FFD3EC0.99

\cellcolor  [HTML]FFD3ECMol-Llama

\cellcolor  [HTML]FFD3EC0.12 (0.59)

\cellcolor  [HTML]FFD3EC0.97

\cellcolor  [HTML]FFD3ECProtChatGPT

\cellcolor  [HTML]FFD3EC0.10 (0.55)

\cellcolor  [HTML]FFD3EC0.99

3D-MoLM(S)

0.59 (2.23)

\cellcolor  [HTML]FFD3EC0.89

Prot2Chat(S)

0.29 (2.30)

\cellcolor  [HTML]FFD3EC1.00

\cellcolor  [HTML]FFD3EC3D-MoLM

\cellcolor  [HTML]FFD3EC0.23 (1.15)

0.83

\cellcolor  [HTML]FFD3ECProt2Chat

\cellcolor  [HTML]FFD3EC0.06 (0.23)

0.99

Figure 1 :

Mol-Llama performance on the Mol-Instructions captioning task, evaluated across five molecule length bins with 6 metrics (left y-axis, detailed in App

D.3  ) plotted as curves with dashed overall averages, and the background bars indicate the proportion of samples in each length bin (right y-axis).

All-atom modeling  has advanced rapidly, catalyzed by AlphaFold-3  (Abramson  et al. ,  2024 ) . Subsequent work, including BoltzGen  (Stark  et al. ,  2025 )  for coordinates generation and ATOMICA  (Fang  et al. ,  2025 )  for representation, further suggests that deep models can operate directly on all-atom level structures. However, most structure-aware LLM efforts remain single modality  (Wang  et al. ,  2025b ; Park  et al. ,  2024 )  without a unified interface for heterogeneous all-atom entities. ChatNT  (de Almeida  et al. ,  2025 )  takes a step toward integrating nucleic acids and proteins, but is constrained to sequence inputs without geometric evidence.

This gap motivates a rethinking of how all-atom structural evidence can be selected and exposed to the language model:

Challenge 1: Budget Scaling in All-Atom Modalities.  All-atom graphs span a wide size range. As shown in Fig.

1  , Q-Former-style  (Li  et al. ,  2023 )  fixed-length query connectors (Mol-Llama  (Kim  et al. ,  2025 ) ) yield lower performance across all metrics for molecules in larger length bins, due to over-compressed geometric features. Conversely, increasing the budget over-allocates capacity to small entities, diluting attention.
 Our solution: Scaling-Aware Patching  uses instruction-conditioned gating to select anchors and expand variable-size structural patches, letting the query token budget grow with structural complexity to avoid over-compression and context inefficiency.

Challenge 2: Structural Hallucination.  Sequence-only inputs, such as SMILES, protein/DNA residue sequences, do not encode geometry or long-range spatial relations. As shown in Tab.

1  , LLMs that lack verifiable structural evidence can produce geometric rationales that are not entailed by the conditioning context.
 Our solution: Geometry Grounding Adapter  refines the adaptive tokens through cross-attention with modality embeddings, injecting the final modality
tokens into the LLM to provide the explicit geometric grounding
to suppress structural hallucinations.

Together, these designs address both the correctness and scalability of structure-aware reasoning: Scaling-Aware Patching adaptively allocates representational capacity to structurally informative regions, while the Geometry Grounding Adapter ensures that LLM outputs are explicitly conditioned on geometry-aware tokens.
Our contributions are summarized as follows:

•

We propose  Scaling-Aware Patching , an instruction-conditioned anchor patch growing mechanism ensuring modality token count scales with structural complexity, mitigating fixed-length connector bottlenecks.

•

We introduce  Geometry Grounding Adapter , which injects verifiable geometric cues into the LLM via modality tokens, reducing structural hallucination.

•

We present  Cuttlefish , a unified structure-aware LLM that reasons over all-atom modalities, and achieves strong performance across all-atom understanding benchmarks.

2  Related Work

Molecular LLMs have progressed from contrastive representation learning toward generation and instruction following. MoleculeSTM  (Liu  et al. ,  2023b )  and MoMu  (Su  et al. ,  2022 )  establish graph or structure and text alignment via contrastive co-embedding, while InstructMol  (Cao  et al. ,  2025 )  and GIT-Mol  (Liu  et al. ,  2023a )  introduce projector and adapter-based fusion with instruction tuning. Connector designs vary across Q-Former-style  (Li  et al. ,  2023 )  query tokens, as in 3D-MoLM  (Li  et al. ,  2024 )  and UniMoT  (Guo  et al. ,  2025b ) , and tokenization oriented bridges such as Graph2Token  (Wang  et al. ,  2025a ) , contrasting with sequence-only baselines like MolT5  (Edwards  et al. ,  2022 ) . Protein LLMs follow a parallel trajectory, from alignment in ProtST  (Xu  et al. ,  2023b )  to open-ended generation in ProtLLM  (Zhuo  et al. ,  2024 ) , with Prot2Text-V2  (Fei  et al. ,  2025 )  and ProteinGPT  (Xiao  et al. ,  2024a )  emphasizing captioning and dialogue under instruction tuning. For nucleic acids, ChatNT  (de Almeida  et al. ,  2025 )  unifies DNA, RNA, and protein through encoder coupling, while RNA-GPT  (Xiao  et al. ,  2024b )  targets instruction-aligned RNA sequence understanding.

Despite strong empirical progress, existing systems exhibit recurring geometry bottlenecks. Sequence-only or shallow fusion, including MolT5  (Edwards  et al. ,  2022 ) , ProtST  (Xu  et al. ,  2023b ) , and RNA-GPT  (Xiao  et al. ,  2024b ) , often under-specifies geometry, which correlates with structural hallucinations in dial