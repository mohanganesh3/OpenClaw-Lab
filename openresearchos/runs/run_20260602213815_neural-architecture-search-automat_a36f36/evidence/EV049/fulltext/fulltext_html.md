[2602.19261] DGPO: RL-Steered Graph Diffusion for Neural Architecture Generation

DGPO: RL-Steered Graph Diffusion for
 Neural Architecture Generation

Aleksei Liuliakov

,
Luca Hermes

,
Barbara Hammer

Abstract

Reinforcement learning fine-tuning has proven effective for steering generative diffusion models toward desired properties in image and molecular domains.
Graph diffusion models have similarly been applied to combinatorial structure generation, including neural architecture search (NAS).
However, neural architectures are directed acyclic graphs (DAGs) where edge direction encodes functional semantics such as data flow-information that existing graph diffusion methods, designed for undirected structures, discard.
We propose Directed Graph Policy Optimization (DGPO), which extends reinforcement learning fine-tuning of discrete graph diffusion models to DAGs via topological node ordering and positional encoding.
Validated on NAS-Bench-101 and NAS-Bench-201, DGPO matches the benchmark optimum on all three NAS-Bench-201 tasks (91.61%, 73.49%, 46.77%).
The central finding is that the model learns transferable structural priors: pretrained on only 7% of the search space, it generates near-oracle architectures after fine-tuning, within 0.32 percentage points of the full-data model and extrapolating 7.3 percentage points beyond its training ceiling.
Bidirectional control experiments confirm genuine reward-driven steering, with inverse optimization reaching near random-chance accuracy (9.5%).
These results demonstrate that reinforcement learning-steered discrete diffusion, once extended to handle directionality, provides a controllable generative framework for directed combinatorial structures.

This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible.

I

Introduction

Diffusion models have emerged as powerful generative frameworks for learning complex distributions, with recent extensions to graph-structured data  [ 1 ]  enabling direct generation of molecular graphs and other combinatorial objects.
Reinforcement learning (RL) fine-tuning steers pretrained diffusion models toward desired properties: DDPO  [ 2 ]  introduced this for image generation, and GDPO  [ 3 ]  extended it to molecular graphs.
In the graph domain, however, these methods have been limited to undirected structures.

Many important domains involve directed acyclic graphs (DAGs): neural architectures encode data flow through directed edges, causal networks represent asymmetric dependencies, and computational workflows specify ordered execution.
In such graphs, edge direction encodes functional semantics that is destroyed by symmetric treatment.
Neural architecture search (NAS)  [ 4 ,  5 ]  - a combinatorial optimization problem over DAG-structured networks - exemplifies this challenge: existing generative approaches either condition generation on target accuracy  [ 6 ]  or build on graph diffusion models designed for undirected structures  [ 1 ] , forgoing directional information.
Extending RL-steered discrete diffusion to DAGs requires respecting and reconstructing this directional structure.

We propose Directed Graph Policy Optimization (DGPO), extending GDPO  [ 3 ] , a graph diffusion RL fine-tuning framework, to directed acyclic graphs via topological node ordering and positional encoding.
This enables the discrete diffusion process to respect directionality while preserving the reward-driven steering mechanism of the underlying framework.

We validate DGPO on two public NAS benchmarks, NAS-Bench-101  [ 7 ]  and NAS-Bench-201  [ 8 ] , achieving competitive architecture generation that reaches the NB201 benchmark optimum on all three tasks (91.61%, 73.49%, and 46.77%).
We find that the model can acquire transferable structural priors: pretrained on only 7% of the search space, it generates near-oracle architectures after RL fine-tuning, within 0.32 percentage points of the full-data model and extrapolating up to 7.3pp beyond its training ceiling.
We provide additional evidence for reward-driven steering via bidirectional control, where inverting the reward signal drives generation toward near random-chance accuracy (9.5%), and we also show how the framework extends to multi-objective reward formulations.

Our contributions are:

1.

Methodological:  DGPO extends GDPO to directed acyclic graphs via topological node ordering and positional encoding, enabling controllable generation of DAG-structured objects.

2.

Empirical (primary):  We show evidence that the resulting model learns transferable structural priors: a model trained on only 7% of the NAS search space generates near-oracle architectures after RL fine-tuning (within 0.32pp), with up to

+  +

7.3pp extrapolation beyond its training ceiling.

3.

Empirical (supporting):  We provide additional evidence for the steering mechanism through bidirectional control (inverse reaches near random chance at 9.5%) and show extension to multi-objective reward formulations.

Code and pretrained models will be released upon acceptance.

II

Related Work

Graph diffusion models. 
Diffusion models have been extended to graph-structured data through both discrete and continuous formulations  [ 9 ] .
DiGress  [ 1 ]  introduced discrete denoising diffusion for graphs, applying categorical noise transitions to node and edge attributes with a graph transformer denoiser.
Autoregressive variants  [ 10 ]  generate graphs sequentially but sacrifice parallelism.
DGPO builds on DiGress as its discrete diffusion backbone.

RL fine-tuning of diffusion models. 
Reinforcement learning can steer pretrained diffusion models toward desired properties by framing denoising as a Markov decision process  [ 11 ] .
DDPO  [ 2 ]  introduced this approach for image generation, optimizing human preference scores via policy gradients.
DPOK  [ 12 ]  adds KL regularization for text-to-image alignment.
GDPO  [ 3 ]  adapts the framework to molecular graph generation using reward-weighted denoising trajectories.
These methods operate on undirected structures (images, molecules); DGPO extends the GDPO framework to directed acyclic graphs.

Neural architecture search. 
NAS has been addressed through RL controllers  [ 4 ] , evolutionary algorithms  [ 13 ] , differentiable relaxations such as DARTS  [ 14 ] , and predictor-based methods including BANANAS  [ 15 ] .
Generative approaches learn to directly produce architectures: D-VAE  [ 16 ]  uses variational autoencoders for directed graph generation, DiffusionNAG  [ 17 ]  applies diffusion models for task-guided generation, AG-Net  [ 18 ]  uses autoregressive generation with a learned surrogate, and GraphPNAS  [ 19 ]  learns distributions over high-performing architectures.
DiNAS  [ 6 ]  conditions a graph diffusion model on target accuracy to generate architectures directly.
DGPO differs from conditioning-based approaches by steering the generation distribution via RL fine-tuning, enabling capabilities beyond accuracy targeting: bidirectional control, out-of-distribution discovery from filtered data, and multi-objective optimization.

We extend an existing RL fine-tuning framework (GDPO) to a structurally different graph class (DAGs), rather than proposing a new RL algorithm or diffusion architecture.
The contribution lies in enabling RL-steered discrete diffusion for directed combinatorial structures and providing empirical evidence for transferable structural priors in the resulting generative model.

III

Method

We first review the discrete diffusion and RL fine-tuning foundations that DGPO builds on, then present our extension to directed acyclic graphs.

III-A

Preliminaries

Discrete graph diffusion. 
DiGress  [ 1 ]  defines a discrete diffusion process over graphs

𝐆  =

(  𝐗  ,  𝐄  )

\mathbf{G}=(\mathbf{X},\mathbf{E})

with categorical node attributes

𝐗  ∈

{  1  ,  …  ,  a  }

n

\mathbf{X}\in\{1,