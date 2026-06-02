[2512.07628] MoCA: Mixture-of-Components Attention for Scalable Compositional 3D Generation

MoCA: Mixture-of-Components Attention for Scalable Compositional 3D Generation

Zhiqi Li

,1,2

Wenhuan Li  3

Tengfei Wang  3,†

Zhenwei Wang  3

Junta Wu  3

Haoyuan Wang  3

Yunhan Yang  3

Zehuan Huang  3

Yang Li  3

Chunchao Guo  3,†

Peidong Liu  2

1  Zhejiang University

2  Westlake University

3  Tencent Hunyuan

Work done during an internship at Tencent Hunyuan;

†  Corresponding Author.

Abstract

Compositionality is critical for 3D object and scene generation, but existing part-aware 3D generation methods suffer from poor scalability due to quadratic global attention costs when increasing the number of components. In this work, we present  MoCA , a compositional 3D generative model with two key designs: 1)  importance-based component routing  that selects top-k relevant components for sparse global attention, and 2)  unimportant components compression  that preserve contextual priors of unselected components while reducing computational complexity of global attention. With these designs, MoCA enables efficient, fine-grained compositional 3D asset creation with scalable number of components. Extensive experiments show MoCA outperforms baselines on both compositional object and scene generation tasks. Project page:  https://lizhiqi49.github.io/MoCA .

Figure 1:

MoCA  enables scalable compositional 3D generation with up to 32 components per 3D assets. It can generate part-level 3D object with fine-grained composition, and instance-level 3D scenes with complex layout.

1  Introduction

Compositionality is a a fundamental concept in 3D art design and assets creation pipeline. It describes how complex 3D assets are formed by combining simpler and semantically coherent components, such as objects composed of simple parts and scenes composed of individual objects.
These component-aware representations enable the reuse of components, targeted editing, animation, and the modeling of physically plausible interactions, all of which are essential for applications ranging from virtual production and game asset creation to robotics and computer-aided design.

Recently, 3D diffusion transformer (DiT) models  (Zhang et al.,  2024 ; Li et al.,  2025a ; Zhao et al.,  2025 ; Xiang et al.,  2024 )  have dramatically improved the generation quality of an individual object.
However, these models commonly treat an object or scene as a monolithic entity, which limits controllability over the generated content and makes many downstream tasks ( e.g. , component-level editing, materials customization for each component) challenging or even unachievable.
Recent works  (Huang et al.,  2025 ; Lin et al.,  2025 )  explore compositional 3D generation by extending pre-trained 3D shape generators  (Li et al.,  2025a ; Hunyuan3D et al.,  2025 ; Xiang et al.,  2024 )  to jointly generate multiple components (either parts or instances).

While these works demonstrate that structured latent spaces enable the generation of a few semantically coherent parts and multi-object scenes, they suffer from two critical limitations when scaling up the number of components: 1)  Salability of global attention across components.  To model cross-component dependencies, previous models for multi-component generation often leverage global attention modules over all component tokens by concatenating their token sets. In this naive design,

N  N

components (each represented by

L  L

latent tokens) result in a global attention over

N  ×  L

N\times L

tokens with quadratic computational cost

O  ​

(

N  2

​

L  2

)

O(N^{2}L^{2})

. As

N  N

grows ( e.g. a complex scene, fine-grained decompositions of an object), this cost quickly becomes computationally prohibitive.

2)  Uniform attention wastes capacity.

Not all components exhibit strong interactions with one another. For example, modeling a character’s hand typically only requires detailed information from the wrist and forearm, while modeling the position of a chair primarily correlates with the nearby table. Blindly enabling every token to attend to all others allocates computational resources and memory to numerous low-value interactions, leading to inefficiency and constraining model scale.

To address these problems, we propose  MoCA , a native compositional 3D generative model equipped with a novel  M ixture- o f- C omponents  A ttention mechanism, designed for efficient, scalable, and accurate compositional modeling of 3D objects and scenes.
MoCA is built upon two key designs:

•

Importance-based component routing.  Inspired by the practices of MoE (Mixture-of-Experts) models  (Fei et al.,  2024 ; Dai et al.,  2024 ; Jiang et al.,  2024 ) , we introduce a lightweight router module within the global block. For a given component, the router estimates the importance of other components relative to it, and then selects the top-

k  k

important components for sparse global attention. This design is based on the hypothesis that  a given component only requires detailed information from a small subset of other components .

•

Compression of distant components.  Unlike previous methods, for the components not selected by the router, we also compress them in compact tokens for global attention rather than discarding them. This preserves coarse-grained context (e.g., spatial priors, presence/absence cues) while dramatically reducing the number of key/value tokens in attention computations.

The combination of the these two designs yields a context length of

L

g  ​  l  ​  o  ​  b  ​  a  ​  l

=

L  +

k  ​  L

+

(

N  −  k  −  1

)

​

L  σ

L_{global}=L+kL+(N-k-1)\frac{L}{\sigma}

in our global attention layers, where

σ  \sigma

is the compression ratio of unimportant components. With typical settings where

k  ≪  N

k\ll N

and

σ  ≫  1

\sigma\gg 1

, the context length is much smaller than the naive global attention used in prior works  (Huang et al.,  2025 ; Lin et al.,  2025 ) , enabling efficient yet powerful compositional 3D generation.

We evaluate MoCA across both object-level and scene-level 3D generation tasks. For object-level generation, MoCA generates a 3D object from a single image with automatically decomposed 3D parts. For scene-level generation, it produces instance-composed 3D scenes conditioned on scene images, using per-instance masks as auxiliary conditions. Extensive qualitative and quantitative experiments demonstrate that MoCA outperforms prior works by a clear margin, with particularly notable improvements in fine-grained component generation.

2  Related Works

3D Latent Diffusion Models. 
Recent approaches that extend latent diffusion models (LDMs) to native 3D shape generation can be broadly categorized into two paradigms: vecset-based methods and sparse voxel-based methods. Vecset-based methods leverage a compact latent representation introduced by 3DShape2VecSet  (Zhang et al.,  2023 ) . Subsequent studies  (Zhang et al.,  2024 ; Li et al.,  2024 ; Wu et al.,  2024 ; Li et al.,  2025a ; Zhao et al.,  2025 )  have advanced this paradigm to generate 3D shapes with high-resolution details, demonstrating its scalability and representational capacity. In contrast, Trellis  (Xiang et al.,  2024 )  proposes a structured latent space grounded in sparse voxels.
Follow-up work  (Ye et al.,  2025 ; He et al.,  2025 ; Wu et al.,  2025 ; Li et al.,  2025b )  shows that such sparse voxel-based latent spaces excel at capturing fine-grained geometric details.
Both vecset and sparse voxel-based representation produce implicit occupancy or SDF fields rather than explicit 3D meshes, hence they require an iso-surface extraction step, such as Marching Cubes  (Lorensen &amp; Cline,  1998 ) , to obtain triangular meshes.

Figure 2:

Overview of MoCA . Our DiT model starts with packing each component’s latents using several learnable queries through a cross-attention layer. Random ID embeddings are applied to distinguish differ