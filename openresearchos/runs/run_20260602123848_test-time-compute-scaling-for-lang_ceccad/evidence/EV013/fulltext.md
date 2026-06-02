[2502.01694] Metastable Dynamics of Chain-of-Thought Reasoning: Provable Benefits of Search, RL and Distillation

\etocdepthtag

.tocmtchapter
 \etocsettagdepth mtchaptersubsection
 \etocsettagdepth mtappendixnone

Metastable Dynamics of Chain-of-Thought Reasoning:

Provable Benefits of Search, RL and Distillation

Juno Kim

University of Tokyo and RIKEN AIP. junokim@g.ecc.u-tokyo.ac.jp.

Denny Wu

New York University and Flatiron Institute. dennywu@nyu.edu.

Jason D. Lee

Princeton University. jasonlee@princeton.edu.

Taiji Suzuki

University of Tokyo and RIKEN AIP. taiji@mist.i.u-tokyo.ac.jp.

Abstract

A key paradigm to improve the reasoning capabilities of large language models (LLMs) is to allocate more inference-time compute to search against a verifier or reward model. This process can then be utilized to refine the pretrained model or distill its reasoning patterns into more efficient models. In this paper, we study inference-time compute by viewing chain-of-thought (CoT) generation as a metastable Markov process: easy reasoning steps (e.g., algebraic manipulations) form densely connected clusters, while hard reasoning steps (e.g., applying a relevant theorem) create sparse, low-probability edges between clusters, leading to phase transitions at longer timescales. Under this framework, we prove that implementing a search protocol that rewards sparse edges improves CoT by decreasing the expected number of steps to reach different clusters. In contrast, we establish a limit on reasoning capability when the model is restricted to local information of the pretrained graph. We also show that the information gained by search can be utilized to obtain a better reasoning model:

(  1  )

(1)

the pretrained model can be directly finetuned to favor sparse edges via policy gradient methods, and moreover

(  2  )

(2)

a compressed  metastable representation  of the reasoning dynamics can be distilled into a smaller, more efficient model.

∙  \bullet

∙  \bullet

∙  \bullet

∙  \bullet

∙  \bullet

∙  \bullet

∙  \bullet

∙  \bullet

∙  \bullet

∙  \bullet

∙  \bullet

∙  \bullet

∙  \bullet

∙  \bullet

∙  \bullet

distillation

X  in

X_{\operatorname{in}}

X  out

X_{\operatorname{out}}

Figure 1:

(Left)  Example of metastable graph with three clusters. Each state represents a logical assertion and edges correspond to reasoning steps. Solid and dashed arrows indicate easy (within-cluster) and hard (inter-cluster) reasoning steps, respectively. The goal of the reasoner is to retrieve a valid CoT path from

X  in

X_{\operatorname{in}}

to

X  out

X_{\operatorname{out}}

(highlighted). Search aims to use CoT generated from the pretrained model to explore the linguistic model and identify hard steps, which can then be used to fine-tune the pretrained model via RL to improve its generation.  (Right)  The coarse-grained dynamics of CoT at long timescales can be represented by a meta-chain on the set of clusters and distilled into a smaller model, which can generate reasoning paths more efficiently.

1  Introduction

Pretraining and inference constitute two distinct computational phases in large language models (LLMs). The pretraining phase, during which the model learns from vast amounts of text data through next-token prediction  ( Radford et al. ,

2018  ) , is well known for its high computational demands, and its scaling behavior has been extensively studied  ( Kaplan et al. ,

2020  ;  Hoffmann et al. ,

2022  ;  Dubey et al. ,

2024  ) .
On the other hand, inference (running the trained model to generate responses) was traditionally considered computationally inexpensive, until a recent paradigm shift demonstrating that model reasoning capabilities can drastically improve by allocating more computational resources during inference time  ( Jaech et al. ,

2024  ;  Guo et al. ,

2025  ;  Kimi et al. ,

2025  ) . Hence it is crucial to understand the advantages scaling inference computation can provide beyond those achieved through pretraining  ( Jones ,

2021  ;  Snell et al. ,

2024  ;  Wu et al. ,

2024  ) .

Reasoning LLMs follow the chain-of-thought (CoT)  ( Nye et al. ,

2021  ;  Wei et al. ,

2022  )  format where intermediate reasoning steps are iteratively generated before arriving at a final answer. Various reinforcement learning (RL) based approaches  ( Bai et al. ,

2022  )  have been proposed to improve CoT quality at inference time, such as process reward modeling  ( Lightman et al. ,

2023  ;  Uesato et al. ,

2022  ) , Monte-Carlo Tree Search (MCTS)  ( Silver et al. ,

2018  ;  Feng et al. ,

2023b  ;  Trinh et al. ,

2024  ;  Xie et al. ,

2024  ) , and data self-generation  ( Zelikman et al. ,

2022  ;  Kumar et al. ,

2024  ) . Theoretically, the benefit of (sufficiently long) CoT has been studied in terms of expressive power and statistical efficiency  ( Merrill and Sabharwal ,

2023  ;  Li et al. ,

2024b  ;  Kim and Suzuki ,

2024  ;  Wen et al. ,

2024  ) .

Motivated by the discrete and sequential nature of CoT, we follow

Xu et al.  (  2019  );  Sanford et al.  (  2024a  );  Abbe et al.  (  2024  );  Besta et al.  (  2024  )  and consider learning on graphs as an ideal abstraction of complex reasoning tasks. We model pretraining as the process of discovering the graph structure, or the  linguistic  (world) model, upon which a  reasoning  (inference) component is implemented to search for a valid path between states. Building on the observation that intermediate reasoning steps vary in difficulty, we assume the underlying graph consists of dense clusters connected by sparse, low-probability edges representing “hard” reasoning steps. At a high level, this division parallels the System 1 vs. System 2 distinction discussed in

Kahneman  (  2011  );  Xiang et al.  (  2025  ) . We further model CoT generation as a Markov process and characterize hitting/escape times by leveraging  metastability theory

( Bovier et al. ,

2002  ;  Betz and Le Roux ,

2016  ) , which describes systems with multiple locally stable states separated by high energy barriers, leading to a timescale separation between local and global transitions (e.g., a reasoner may become stuck at a critical reasoning step for an extended period). Our toy model captures key phenomena observed in the training of reasoning LLMs:

•

Benefit of search and RL.  Inference-time search elicits reasoning capabilities beyond pretraining  ( Jones ,

2021  ;  Yao et al. ,

2024  ;  Snell et al. ,

2024  ) . Roughly speaking, running search on the pretrained graph identifies important reasoning steps, and then RL can improve the base linguistic model by modifying the graph and reweighting the corresponding transition probabilities.

•

Benefit of distillation.  Reasoning patterns can be distilled into a smaller model  ( Hsieh et al. ,

2023  ;  Gandhi et al. ,

2024  ;  Guo et al. ,

2025  ) . By training on curated CoT data of the larger model, we can efficiently represent the reasoning dynamics with a much smaller meta-chain that compresses the dense clusters (representing “easy” steps).

1.1  Our Contributions

We study the metastable Markov process underlying CoT generation (see Figure

2  ) which provides insights into the roles of pretraining, search, RL, and distillation. Our contributions are summarized as follows.

•

In Section

2  , we introduce a perturbed Markov chain model for CoT reasoning that differentiates between easy and hard reasoning steps through a dense-sparse structure. We develop a quantitative analysis of its metastable dynamics over long timescales by deriving tight bounds on the expected hitting times of target states.

•

In Section

3  , we demonstrate that inference-time search based on intrinsic reward improves hitting times by identifying key reasoning steps, whose generation can be enhanced directly or by fine-tuning the base model with RL. Moreover, optimization guarantees for pretraining and RL (PPO-Clip) are provided for a simple softmax mo