[2506.12409] Branch, or Layer? Zeroth-Order Optimization for Continual Learning of Vision-Language Models

Branch, or Layer? Zeroth-Order Optimization for Continual Learning of Vision-Language Models

Ziwei Liu

1,4  ,
Borui Kang  1

1  1 Equal Contribution

1,3  ,
Wei Li

1  ,
Hangjie Yuan

2  ,
 Yanbing Yang

1

Wenbin Li

3  ,
 Jun Luo

4  ,
 Yifan Zhu

5  ,
 Tao Feng

6

1 Sichuan University

2 Zhejiang University

3 Nanjing University

4 NTU

5 BUPT

6 Tsinghua University

fengtao.hi@gmail.com

Equal ContributionCorresponding Authors

Abstract

Continual learning in vision-language models (VLMs) faces critical challenges in balancing parameter efficiency, memory consumption, and optimization stability.
While First-Order (FO) optimization (e.g., SGD) dominate current approaches, their deterministic gradients often trap models in suboptimal local minima and incur substantial memory overhead.
This paper pioneers a systematic exploration of Zeroth-Order (ZO) optimization for vision-language continual learning (VLCL).
We first identify the incompatibility of naive full-ZO adoption in VLCL due to modality-specific instability.
To resolve this, we selectively applying ZO to either vision or language modalities while retaining FO in the complementary branch.
Furthermore, we develop a layer-wise optimization paradigm that interleaves ZO and FO across network layers, capitalizing on the heterogeneous learning dynamics of shallow versus deep representations.
A key theoretical insight reveals that ZO perturbations in vision branches exhibit higher variance than language counterparts, prompting a gradient sign normalization mechanism with modality-specific perturbation constraints.
Extensive experiments on four benchmarks demonstrate that our method achieves state-of-the-art performance, reducing memory consumption by 89.1% compared to baselines. Code will be available upon publication.

1  Introduction

Continual Learning (CL) has witnessed significant advancements in convolutional architectures (e.g., ResNet

erd

;

res_2

;

res_3

;

pose

and ViT

vit_1

;

vit_2

;

vit_3

).
Recently, Vision-Language Models (VLMs)-based CL approaches have attracted growing research attention.
Particularly, CLIP-based methods

clip_cl_1

;

clip_cl_2

have demonstrated robust continual learning capabilities.
For instance, ZSCL

zscl

introduces an unlabeled reference dataset for knowledge distillation to guide parameter space optimization.
Mod-X

mod-x

maintains consistency in multimodal representation spaces through specially designed loss functions, effectively mitigating performance degradation caused by geometric variations in feature spaces.
However, these methods require full-parameter fine-tuning of CLIP models, which incurs substantial computational overhead.
The recently proposed MoE4Adapter

MoE4Adapters

addresses this limitation by adopting Parameter-Efficient Fine-Tuning (PEFT) strategies

peft

;

peft1

;

peft2

, achieving competitive CL performance with significantly reduced resource consumption.

Notably, existing VLM-based CL methods predominantly employ first-order (FO) optimization strategies (e.g., SGD, Adam)

overview_opti

.
While these methods provide stable gradient descent directions, their inherent local convergence characteristics often lead to suboptimal solutions

zo_intro_1

;

zo_intro_2

;

zo_intro_3

.
MeZO

mezo

and MaZO

maskzo

both leverage ZO optimization to address the memory challenges of fine-tuning large language models in resource-constrained environments while maintaining or even surpassing the performance of traditional backpropagation-based fine-tuning.
This breakthrough has inspired novel research directions in CL.
For example, ZeroFlow

zeroflow

systematically compares various FO optimization strategies and discovers that ZO optimization exhibits unique advantages in alleviating catastrophic forgetting, providing new insights for optimizer design in continual learning systems.

However, the application of zero-order (ZO) optimization in Vision-Language Continual Learning (VLCL) remains understudied. Therefore, we conduct an in-depth exploration of ZO in VLCL using the state-of-the-art MoE4Adapter model

MoE4Adapters

as the baseline. First, we naively replace all FO optimizers with ZO counterparts to observe the direct impact of ZO in VLCL.
However, experiments reveal that this approach causes excessive training divergence and failure to converge.
Inspired by the cross-modal interaction consistency theory in VLMs

cross_1

;

cross_2

, we introduce two investigations:

Investigation 1: Can ZO and FO collaborate for global model optimization across branches?

We implement ZO in either the visual or linguistic modality branch while retaining FO in the other. Experiments show that this strategy significantly outperforms the naive full-ZO approach.

Investigation 2: Can ZO and FO collaborate for hierarchical optimization within branch layers?

We thus deploy ZO only in specific layers (e.g., odd or consecutive layers) within a single branch while keeping FO in the remaining layers.
Experiments further demonstrate that this approach yield better results, with certain configurations even surpassing the full-FO baseline.
We attribute this to the stable directivity of FO optimization combined with the limited volatility of ZO, which helps the model escape local optima during optimization.
Additionally, in layer-wise experiments, we find the visual branch exhibit significantly stronger fluctuations under ZO than the language branch.
Therefore, we propose a stabilization mechanism into the visual branch to constrain ZO’s search range, further improving optimization convergence and overall continual learning performance.

In summary, the main contributions of this work are:

•

We reveal differential effects of ZO optimization on visual and language branches in VLCL and proposed a modality-specific framework combining ZO with FO optimization to balance stability and exploration.

•

Our layer-wise granular approach selectively deploys ZO in specific layers alongside FO in others, leveraging their synergy to escape local optima and enhance optimization efficiency.

•

Addressing ZO-induced volatility in the visual branch, we introduce a targeted stabilization mechanism to constrain its search range, significantly improving convergence stability and overall VLCL performance.

2  Optimization in Continual Learning: A Primer

Traditional optimization methods in continual learning primarily rely on FO gradient descent

fo_1

;

fo_2

. Given a loss function

ℒ  ​

(  θ  )

\mathcal{L}(\theta)

and parameters

θ  \theta

, FO updates parameters via:

θ  =

θ  −

η  ⋅

∇  θ

ℒ

​

(  θ  )

,

\theta=\theta-\eta\cdot\nabla_{\theta}\mathcal{L}(\theta),

(1)

where

∇  θ

ℒ

​

(  θ  )

\nabla_{\theta}\mathcal{L}(\theta)

is the FO gradient, and

η  \eta

is the learning rate. While FO methods offer accurate gradient directions and have been widely adopted due to their efficiency, they may incur considerable memory overhead when higher-order derivatives (e.g., Hessians in second-order extensions) are involved

lora

;

memory_1

. Moreover, their deterministic update paths tend to limit exploration during training, potentially increasing the risk of convergence to suboptimal local minima and reducing adaptability in dynamic continual learning scenarios.

In contrast, ZO optimization estimates gradients through forward passes through purposeful perturbations. For a parameter subset

θ  m  k

\theta^{k}_{m}

(e.g., layer

k  k

in modality

m  m

), ZO approximates gradients via directional perturbations:

∇  ZO

ℒ

​

(

θ  m  k

)

≈

ℒ  ​

(

θ  m  k

+

ε  ​  Δ

)

−

ℒ  ​

(

θ  m  k

)

ε

⋅  Δ

,

\nabla_{\text{ZO}}\mathcal{L}(\theta^{k}_{m})\approx\frac{\mathcal{L}(\theta^{k}_{m}+\varepsilon\Delta)-\mathcal{L}(\theta^{k}_{m})}{\varepsilon}\cdot\Delta,

(2)

where

Δ  \Delta

is a random directional vector, and

ε  \varepsil