[2512.10547] Unlocking the Address Book: Dissecting the Sparse Semantic Structure of LLM Key-Value Caches via Sparse Autoencoders

Unlocking the Address Book: Dissecting the Sparse Semantic Structure
of LLM Key-Value Caches via Sparse Autoencoders

Qingsen Ma

Dianyun Wang

Jiaming Lyu

Yaoye Wang

Lechen Ning

Sujie Zhu

Zhenbo Xu

Liuyu Xiang

Huining Li

Huijia Wu

Zhaofeng He

Abstract

The Key-Value (KV) cache is the primary memory bottleneck in long-context Large Language Models, yet it is typically treated as an opaque numerical tensor. In this work, we propose  STA-Attention , a framework that utilizes Top-K Sparse Autoencoders (SAEs) to decompose the KV cache into interpretable “semantic atoms.” Unlike standard

L  1

L_{1}

-regularized SAEs, our Top-K approach eliminates shrinkage bias, preserving the precise dot-product geometry required for attention.
Our analysis uncovers a fundamental  Key-Value Asymmetry : while Key vectors serve as highly sparse routers dominated by a “Semantic Elbow,” deep Value vectors carry dense content payloads requiring a larger budget. Based on this structure, we introduce a Dual-Budget Strategy that selectively preserves the most informative semantic components while filtering representational noise.
Experiments on Yi-6B, Mistral-7B, Qwen2.5-32B, and others show that our semantic reconstructions maintain perplexity and zero-shot performance comparable to the original models, effectively bridging the gap between mechanistic interpretability and faithful attention modeling.

Machine Learning, ICML

1  Introduction

The deployment of LLMs in long-context scenarios is fundamentally constrained by the Key-Value (KV) cache memory bandwidth and capacity  ( MLSYS2023_c4be71ab ;  kwon2023efficient ) . As the cache grows linearly with sequence length, it limits batch sizes and increases latency  ( shazeer1911fast ;  touvron2023llama ;  sun2024shadowkv ) . Consequently, compression has become central, with existing approaches predominantly focusing on numerical approximations like quantization  ( sheng2023flexgen )  or attention-based token pruning  ( zhang2023h2o ) . However, these methods treat the KV cache as a generic numerical tensor, optimizing geometric reconstruction while treating the underlying representational logic as a “black box”  ( kim2024lexico ;  geva2021transformer ;  templeton2024scaling ) .

The Interpretability Gap.

Parallel to efficiency research, Mechanistic Interpretability  ( olah2020zoom )  has advanced in decomposing neural networks  ( elhage2021mathematical ;  meng2022locating ;  hernandez2023linearity ;  cunningham2023sparse ) . Sparse Autoencoders (SAEs) effectively disentangle superposition  ( elhage2022toy )  in MLPs  ( cunningham2023sparse )  and Residual Streams  ( lieberum2024gemma ) . Yet, a critical gap remains:  existing SAE research has largely overlooked the internal addressing logic of Attention Heads

( elhage2021mathematical ;  lieberum2024gemma ;  gould2023successor ) . While MLPs are viewed as “Knowledge Memories”  ( dai2022knowledge ;  meng2022mass )  and Residual Streams as “Information Highways”  ( jastrzkebski2017residual ;  von2023transformers ;  candes2005decoding ) , the mechanism mapping input tokens to high-dimensional Key routing vectors  ( jaszczur2021sparse ;  masoudnia2014mixture ;  bahdanau2014neural ;  devlin2019bert )  remains under-explored  ( cao2024head ) . Few question the semantic necessity of the standard

d

h  ​  e  ​  a  ​  d

=  128

d_{head}=128

dimensionality  ( bhojanapalli2020low ;  michel2019sixteen ;  aghajanyan2021intrinsic ) .

Present Work.

We bridge this gap with  STA-Attention  (Sparse Semantic Self-Attention), hypothesizing that KV information lies on a low-dimensional  ( aghajanyan2021intrinsic ) , sparse semantic manifold  ( bengio2013representation ;  liu2023deja ) . By applying  Top-K SAEs

( gao2024scaling )  to Key and Value projections, we decompose dense vectors into interpretable “semantic atoms.” Crucially, we address the shrinkage bias of standard

L  1

L_{1}

-regularized SAEs  ( tibshirani1996regression )  that degrades dot-product calculations. Adopting Top-K SAEs enforces a hard sparsity budget without dampening signal magnitude, ensuring unbiased attention scoring.

Our contributions are:

1.

Hierarchical Addressing Mechanism:  We unveil the functional stratification of attention: shallow layers encode lexical patterns (n-grams), middle layers form a  Syntactic Backbone , and deep layers perform  Polysemy Resolution  via orthogonal semantic features.

2.

Discovery of the “Semantic Elbow”:  We empirically identify a saturation point at

K  =  8

K=8

, where the Top-8 active latents recover over 80% of the Key vector’s directionality. We propose the  Denoising Hypothesis : removing lower-ranked features eliminates noise and improves perplexity.

3.

Key-Value Asymmetry &amp; Dual-Budget Strategy:  We identify a divergence in information density: Keys are sparse (routing) while Values are dense (logical payloads). We introduce a Dual-Budget Strategy (

K

k  ​  e  ​  y

=  8

,

K

v  ​  a  ​  l

=  16

K_{key}=8,K_{val}=16

) to maximize compression while preserving reasoning capabilities.

4.

Performance Parity:  Validating on 7B-scale models (Yi, Mistral, Llama-2), STA-Attention matches the zero-shot performance and perplexity of dense baselines (

K  =  128

K=128

) with significantly reduced memory, confirming the viability of sparse semantic decomposition.

2  Related Work

Efficient KV Cache Management.

Existing research largely bifurcates into quantization and pruning.
 Quantization approaches  like CommVQ  ( li2025commvq )  optimize vector quantization for attention score reconstruction rather than Euclidean distance.
 Pruning strategies , such as RocketKV  ( behnam2025rocketkv )  and “Compute or Load”  ( jin2024compute ) , reduce memory footprint by selectively preserving tokens or re-computing states on the fly.
However, these methods predominantly treat the KV cache as a generic numerical container, optimizing for geometric approximations while remaining agnostic to the underlying  semantic  manifold of the attention mechanism.

Mechanistic Interpretability and SAEs.

Sparse Autoencoders (SAEs) have successfully extracted monosemantic features from MLPs  ( cunningham2023sparse )  and residual streams  ( templeton2024scaling ) . Notably, Top-K SAEs  ( gao2024scaling )  mitigate the shrinkage bias of

L  1

L_{1}

regularization, ensuring unbiased magnitude estimation. Despite this progress, a critical gap remains: the internal addressing logic of attention heads (specifically the

W  K

W_{K}

projection) has received limited scrutiny compared to MLPs, leaving the semantic structure of routing vectors largely unexplored.

Sparse Representation for Compression.

Lexico  ( kim2024lexico )  pioneers the use of sparse coding for KV cache compression via universal dictionaries.
 Distinction:  Our

S  3

S^{3}

-Attention framework advances beyond Lexico in two dimensions.
First, instead of generic universal dictionaries, we employ specialized Top-K SAEs to decouple the specific  routing logic  of Keys from the  content payloads  of Values.
Second, unlike Lexico’s uniform compression, we leverage the “Semantic Elbow” and “Key-Value Asymmetry” to dynamically allocate budgets—heavily compressing sparse routing information while preserving dense semantic content—thereby aligning compression with the model’s intrinsic functional stratification.

3  Preliminaries: Rationale for Top-K SAE

Before detailing our methodology, we address a fundamental design choice: why we adopt the Top-K Sparse Autoencoder (Top-K SAE) architecture  ( gao2024scaling )  over the standard

L  1

L_{1}

-regularized SAE commonly used in interpretability research  ( cunningham2023sparse ) .

While

L  1

L_{1}

-SAEs have been successful in decomposing superposition in Multi-Layer Perceptrons (MLPs), we identify three theoretical misalig