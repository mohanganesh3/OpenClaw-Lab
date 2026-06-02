[2601.08393] Controlled LLM Training on Spectral Sphere

Controlled LLM Training on Spectral Sphere

Tian Xie 1

1

1  1 Corresponding to unakar666@gmail.com

Haoming Luo 2

Haoyu Tang 2  Yiwen Hu 2  Jason Klein Liu  Qingnan Ren 1

Yang Wang  1

Wayne Xin Zhao  2

Rui Yan  3

Bing Su  2

Chong Luo  1

Baining Guo  1

1 Microsoft Research Asia

2 Renmin University

3 Wuhan University

4 IQuest Lab

Abstract

Scaling large models requires optimization strategies that ensure rapid convergence grounded in stability. Maximal Update Parametrization (

𝝁  \bm{\mu}

P) provides a theoretical safeguard for width-invariant

Θ  ​

(  1  )

\Theta(1)

activation control, whereas emerging optimizers like Muon are only “half-aligned” with these constraints: they control updates but allow weights to drift. To address this limitation, we introduce the  Spectral Sphere Optimizer (SSO) , which enforces strict module-wise spectral constraints on both weights and their updates. By deriving the steepest descent direction on the spectral sphere, SSO realizes a fully

𝝁  \bm{\mu}

P-aligned optimization process. To enable large‑scale training, we implement SSO as an efficient parallel algorithm within Megatron. Through extensive pretraining on diverse architectures, including Dense 1.7B, MoE 8B-A1B, and 200-layer DeepNet models, SSO consistently outperforms AdamW and Muon. Furthermore, we observe significant practical stability benefits, including improved MoE router load balancing, suppressed outliers, and strictly bounded activations.

Project Page:

Spectral-Sphere-Optimizer

((a))

AbsMax  (outliers) of Attention Activations

((b))

RMS  of FFN Activations

Figure 1:

Training dynamics of Dense-1.7B activations (log-scaled cross-layer averages). 
Our  Spectral Sphere  maintains constant activation magnitudes throughout training because its

𝝁  \bm{\mu}

P-metrized constraints on the spectral manifold ensure that the activation  RMS  remains at

Θ  ​

(  1  )

\Theta(1)

scale.  Muon  activations show a mild drift due to learning rate decay and weight decay. By contrast,  AdamW  proves the most unstable, generating significantly larger activations, with attention  AbsMax  and FFN  RMS  reaching

∼  100  ×

\sim\!100\times

magnitude compared to those spectral optimizers.

Contents

1  Introduction

LLM training is, at its core, a pursuit of  convergence speed  grounded in the necessity of  stability . While the community has explored various optimization strategies, we argue that the essential principle governing training stability is the  Maximal Update Parametrization  (

𝝁  \bm{\mu}

P)  ( yang2023spectral ) . By mandating that the spectral norms of weights and updates scale as

Θ  ​

(

d  out

/

d  in

)

\Theta(\sqrt{d_{\text{out}}/d_{\text{in}}})

to ensure width-invariant activations remains

Θ  ​

(  1  )

\Theta(1)

scale,

𝝁  \bm{\mu}

P serves as the mathematical safeguard against activation explosions  ( takase2025spikemorestabilizingpretraining ) . However, the current landscape is saturated with methods that fail to satisfy these fundamental conditions. Conventional soft regularization methods, such as decoupled weight decay or initialization strategies, prove insufficient over long training horizons  ( kosson2025weightdecaymattermup ) . This unconstrained weight drift destabilizes the  effective step size  (update-to-weight ratio) and degrades feature learning.

On the other side of the spectrum lies the pursuit of optimal convergence. The recent Muon optimizer  ( jordan2024muon )  has demonstrated remarkable efficiency, often interpreted as steepest descent under the spectral norm. In analyzing Muon, we uncover a surprising insight: it acts as a  “half-aligned”  solution to the

𝝁  \bm{\mu}

P constraints. However, maintaining stable features requires constraining not only the updates but also the weights themselves. Unstable activations like attention logits explosion were still observed in Muon training  ( kimiteam2025kimik2openagentic ) . Consequently, practitioners are forced to rely on “non-essential” architectural patches to artificially force stability — ranging from aggressive normalization schemes like SandwichNorm  ( ding2021cogviewmasteringtexttoimagegeneration )  and QK-Norm  ( henry2020querykeynormalizationtransformers ) , to ad-hoc fixes like logit softcapping  ( kimiteam2025kimik2openagentic )  — often at the cost of model expressivity and requiring extensive hyperparameter tuning. This observation motivates a fundamental question:

What if an optimizer could simultaneously satisfy the steepest descent property for  convergence speed  and the strict

𝛍  \bm{\mu}

P constraints for  fundamental stability ?

To answer this, we propose the mathematically unique solution that unifies these two objectives. By identifying the spectral sphere as the natural manifold for stable feature learning, SSO derives the steepest descent direction constrained within this geometry. Unlike heuristic manifold projection methods  ( xie2025mhcmanifoldconstrainedhyperconnections ;  pethick2025trainingdeeplearningmodels ) , SSO solves a constrained optimization problem in the tangent space via a Lagrange multiplier search, followed by a retraction step to map the trajectory back onto the spectral sphere.

To enable large-scale training, we offer a systematic implementation in Megatron. We provide principled guidelines for spectral preconditioned optimization, specifically deriving the optimal  learning rate scaler , determining the critical  atomic module granularity , and identifying the optimal  spectral radius  to control activation at optimal scales precisely. These offer a robust recipe for large-scale training. Specifically to mitigate the overhead of the iterative root solver, we utilize a novel distributed strategy centered on atomic module sharding ( emergeing_optimizer ) . This technique partitions fused params into independent spectral units, enabling communication-free local updates. We further address solver-induced workload imbalance through a size-aware ping-pong placement strategy, and accelerate matrix operations using adaptive kernel dispatcher, alongside multi-stream execution and singular vector caching.

Empirically, we validate SSO through extensive pretraining experiments across various scales, including Dense 1.7B, MoE 8B-A1B, and 200-layer DeepNet models. SSO consistently outperforms AdamW and Muon while uniquely preserving stable

𝝁  \bm{\mu}

P learning rate transfer. Notably, it yields superior training dynamics: it significantly improves MoE router load balancing, suppresses outliers in deep networks, and strictly bounds activations within a tunable scale.

2  Preliminary

2.1  Maximal Update Parametrization (

𝝁  \bm{\mu}

P)

𝝁  \bm{\mu}

P prescribes how
activations and weight updates should scale with width to preserve feature learning  ( yang2023spectral ) .
Ideal feature learning requires the scale of activations to remain as invariant as possible.
We use  operator norm  to characterize how the norm of activations changes through a linear layer.

Considering a linear layer

𝒚  =

𝑾  ​  𝒙

{\bm{y}}={\bm{W}}{\bm{x}}

with

𝑾  ∈

ℝ

d  out

×

d  in

,

𝒙  ∈

ℝ

d  in

{\bm{W}}\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}},{\bm{x}}\in\mathbb{R}^{d_{\mathrm{in}}}

, the RMS norm is defined as

‖  𝒙  ‖

rms

:=

‖  𝒙  ‖

2

d  in

,

\|{\bm{x}}\|_{\mathrm{rms}}:=\frac{\|{\bm{x}}\|_{2}}{\sqrt{d_{\mathrm{in}}}},

(1)

while the RMS-to-RMS operator norm is defined as

‖  𝑾  ‖

rms  →  rms

:=

sup

𝒙  ≠  𝟎

‖

𝑾  ​  𝒙

‖

rms

‖  𝒙  ‖

rms

.

\|{\bm{W}}\|_{\mathrm{rms}\to\mathrm{rms}}:=\sup_{{\bm{x}}\neq\bm{0}}\frac{\|{\bm{W}}{\bm{x}}\|_{\mathrm{rms}}}{\|{\bm{x}}\|_{\mathrm{rms}}}.

(2)

𝝁  \bm{\mu}

P scale invariance requires maintaining

‖  𝒚  ‖

rms

=

‖  𝒙  ‖

rms

=

Θ  ​

(  1  )

,

\|{\bm{y}}\|_{\mathrm{rms}}=\|{\bm{x}}\|_{\mathrm{rms}}=\Theta(1),

which is equivalent to enforcing the RM