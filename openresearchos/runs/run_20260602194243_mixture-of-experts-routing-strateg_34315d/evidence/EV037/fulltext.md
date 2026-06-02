[2603.02227] Routing Absorption in Sparse Attention: Why Random Gates Are Hard to Beat

Routing Absorption in Sparse Attention:
 Why Random Gates Are Hard to Beat

Keston Aquino-Michaels
 No Way Labs

(February 2026)

Abstract

Can a transformer learn which attention entries matter during
training?
In principle, yes: attention distributions are highly concentrated,
and a small gate network can identify the important entries post-hoc
with near-perfect accuracy.
In practice, barely.
When sparse attention is trained end-to-end, the model’s Q/K/V
projections co-adapt to whatever mask is imposed, absorbing the
routing signal until learned gates perform little better than frozen
random gates.
We call this  routing absorption  and present four independent
lines of evidence for it in a controlled 31M-parameter transformer:
(1) differentiable soft gating converges to nearly the same
perplexity whether the gate is learned or random (

48.73  ±  0.60

48.73\pm 0.60

vs.

49.83  ±  0.04

49.83\pm 0.04

over 3 seeds);
(2) hard top-

k  k

gating receives exactly zero gradient through
the mask;
(3) a gate distilled onto co-adapted Q/K/V achieves high F1 against
oracle masks but catastrophic perplexity when deployed (601.6 vs. 48.6 on mask-agnostic Q/K/V); and
(4) stochastic mask randomization during training fails to prevent
co-adaptation (78.2 ppl deployed dense vs. 37.3 baseline).
We connect routing absorption to the same phenomenon in
Mixture-of-Experts, where random routing matches learned routing
because experts co-adapt to any router, but show that attention
exhibits a structurally more severe form: shared Q/K/V parameters
enable cross-layer compensation pathways absent in MoE, where
experts are self-contained modules.
The implication is that end-to-end sparse attention methods
employing per-query token-level gating face absorption pressure
proportional to the parameter asymmetry between the gate and the
model, and that post-hoc approaches, which decouple representation
learning from sparsification, sidestep this entirely.

1  Introduction

Attention is concentrated.
In a 31M-parameter transformer trained on WikiText-103, the top 64
out of 512 key positions per query capture 90.6% of the total
attention mass (Section

4.1  ).
In Qwen3-1.7B, the concentration is even sharper: oracle top-

k  k

masking at

k  k

=64 (87.5% sparsity) raises perplexity from 11.52
to only 11.57; the remaining 437 positions carry almost no signal.
This suggests that a small learned gate should easily identify
which entries to keep.

And indeed it can, but only after training is over.
A lightweight bilinear gate (

d  gate

d_{\text{gate}}

=32, 1.3% of model
parameters) trained post-hoc on a frozen dense checkpoint converges
to near-oracle routing in 1,000 steps, closing

&gt;  &gt;

94% of the gap
between random and oracle masks at all tested sparsity
levels (Table

4  ).
The same gate architecture, trained end-to-end alongside the model’s
Q/K/V projections for 50,000 steps, learns  almost nothing :
its perplexity matches that of a frozen random gate to within 2.2%.

This paper asks:  why does end-to-end sparse attention training
fail when the structure clearly exists?

The answer is  routing absorption .
The model’s parameters (

∼  \sim

31M), which collectively outnumber
the gate’s 80

×  \times

to 1, continuously adapt to compensate for
whatever mask is imposed.
After 50,000 steps of co-training, the gate’s mask has been
“absorbed” into the Q/K/V representations: removing the gate
changes almost nothing, replacing it with random noise changes
almost nothing, and the gate’s predictions carry little more
information about attention structure than chance.
This is the attention analog of a well-documented phenomenon in
Mixture-of-Experts (MoE), where experts co-adapt to any router
until random routing matches learned
routing  [ 3 ,  4 ,  5 ] , but
with a crucial structural difference: in MoE, experts are
self-contained modules, while in attention, shared Q/K/V projections
enable cross-layer compensation that makes absorption more severe.

The contribution of this paper is not a method but an analysis.
We present four controlled experiments on a 31M-parameter model
that isolate different aspects of the absorption mechanism
(Figures

2

and

3  ),
with preliminary scale evidence from Qwen3-1.7B consistent with
the phenomenon persisting at 55

×  \times

larger scale.
We connect the results to the MoE literature and discuss
implications for recent sparse attention methods that rely on
learned routing.

2  Background and Setup

2.1  The Sparse Attention Routing Problem

Given a pretrained transformer with attention scores

A  =

Q  ​

K  ⊤

/

d  h

A=QK^{\top}/\sqrt{d_{h}}

and attention weights

P  =

softmax  ​

(  A  )

P=\text{softmax}(A)

, the sparse attention routing problem is:
learn a gate function

G  ​

(  x  )

G(x)

that predicts which entries of

P  P

to keep, such that masking the rest preserves model quality.
The gate produces scores

G

(  h  )

∈

ℝ

n  ×  n

G^{(h)}\in\mathbb{R}^{n\times n}

;
at deployment, only the top-

k  k

entries per query are retained
and the rest are masked to

−  ∞

-\infty

before softmax.

In the  end-to-end  setting,

G  G

is trained jointly with the
model: the gate and the Q/K/V projections co-evolve.
In the  post-hoc  setting  [ 12 ] , the
model is frozen and only the gate is trained, typically by
distillation against the model’s own attention distributions.

2.2  Experimental Setup

All experiments use a 6-layer, 256-dimensional, 4-head pre-norm
transformer (

∼  \sim

31M parameters) trained on WikiText-103
(512-token chunks, batch size 16, cosine LR schedule).
The dense baseline achieves 37.32 perplexity.

The gate adds per-head projections

W

g  ​  q

(  h  )

,

W

g  ​  k

(  h  )

∈

ℝ

d  ×

d  gate

W_{gq}^{(h)},W_{gk}^{(h)}\in\mathbb{R}^{d\times d_{\text{gate}}}

producing gate scores

G

(  h  )

=

(

x  ​

W

g  ​  q

(  h  )

)

​

(

x  ​

W

g  ​  k

(  h  )

)

⊤

/

d  gate

G^{(h)}=(xW_{gq}^{(h)})(xW_{gk}^{(h)})^{\top}/\sqrt{d_{\text{gate}}}

.
With

d  gate

d_{\text{gate}}

=32, this adds 393K parameters (1.3% of the model).

We use this small model deliberately: it is large enough to exhibit
routing absorption but small enough to run end-to-end sparse training
experiments (50,000 steps) and controlled ablations at modest cost.
The full experimental suite (31M pretraining, all ablations, and
Qwen3 fine-tuning) cost under $150 on rented
GPUs; every result in this paper can be independently reproduced
on a single consumer GPU in under a day.  1

1  1 Code and data:
 https://github.com/no-way-labs/routing-absorption

In Section

5.2

we present evidence that the phenomenon
intensifies at scale.

2.3  Connection to MoE Routing Absorption

Routing absorption was first identified in Mixture-of-Experts models.
Roller et al.  [ 3 ]  showed that hash-based (random)
routing matches learned routing in MoE language models.
Chen et al.  [ 4 ]  and Fan et
al.  [ 5 ]  confirmed that experts co-adapt to
any router, making the routing function nearly irrelevant.
Clark et al.  [ 6 ]  provided scaling laws showing
that routing quality has diminishing returns as expert capacity grows.

The mechanism is intuitive: when the routed compute (experts) has
far more parameters than the router, the compute adapts to the
router rather than vice versa.
In attention, the full model (

∼  \sim

31M parameters: Q/K/V projections,
feedforward layers, and embeddings) plays the role of experts, and
the gate (

∼  \sim

393K parameters) plays the role of the router, an
80:1 parameter asymmetry that makes absorption nearly inevitable.

3  Evidence for Routing Absorption

We present four independent experiments, each isolating a different
aspect of the absorption mechanism.
All use

k  k

=64 (87.5% sparsity) unless otherwise noted.

3.1  Experiment 1: Learned Gates Match Random Gates

We train the 31M model end-to-end with differentiable so