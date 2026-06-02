# Research Map

Run: `run_20260602194243_mixture-of-experts-routing-strateg_34315d`
Topic: mixture of experts routing strategies (cycle 1)
Scope: `local_experiment`
Feasibility: Yes, experimenting with mixture of experts routing strategies on your MacBook M4 with 36GB RAM and Apple MPS GPU is highly feasible, especially for moderate-sized models. Suitable public datasets include GLUE for language tasks, ImageNet for vision experiments, and WikiText-103 for language modeling benchmarks. For comprehensive experiments testing various routing strategies, you can expect training times ranging from 4-8 hours for smaller models to 1-2 days for larger MoE architectures.

## Summary

- **70 papers** discovered and deep-read
- **0 papers** with full text extraction
- **0 papers** with MacBook-runnable experiments
- **0 papers** with available code
- **0 citation relationships** mapped

## Method Comparison Matrix

| Paper | Dataset | Method | Best Result | Limitations |
|-------|---------|--------|-------------|-------------|
| No deep reads yet | | | | |

## Research Gaps

### G01: Current routing strategies often treat load balancing and ex

Current routing strategies often treat load balancing and expert specialization as separate objectives, leading to a suboptimal equilibrium. A truly effective strategy must dynamically balance the computational load across experts while simultaneously encouraging them to develop distinct, non-overlapping specializations.

**Hypothesis**: TBD
**Suggested dataset**: TBD
**Baseline paper**: TBD
**Evidence**: EV001, EV002
**Difficulty**: medium

### G02: Existing routing strategies are primarily designed for unimo

Existing routing strategies are primarily designed for unimodal data like text, where a single token is routed to one or more experts. This approach is insufficient for multi-modal inputs (e.g., image-text pairs), where different modalities may require different expert pathways and routing logic.

**Hypothesis**: TBD
**Suggested dataset**: TBD
**Baseline paper**: TBD
**Evidence**: EV003
**Difficulty**: hard

### G03: The routing decisions made by the gating network are often o

The routing decisions made by the gating network are often opaque 'black boxes,' making it difficult to debug model failures or understand its reasoning process. A gap exists in developing routing strategies that are both effective and inherently interpretable, allowing us to understand *why* a token was sent to a specific expert.

**Hypothesis**: TBD
**Suggested dataset**: TBD
**Baseline paper**: TBD
**Evidence**: EV004
**Difficulty**: medium

### G04: Routing in MoE models is typically static, determined by the

Routing in MoE models is typically static, determined by the learned gating network before or during the forward pass. This ignores the potential for the model to adapt its computational path based on the context of the entire input sequence or even previous outputs, which could be more efficient and powerful.

**Hypothesis**: TBD
**Suggested dataset**: TBD
**Baseline paper**: TBD
**Evidence**: EV005
**Difficulty**: hard

### G05: The routing mechanism itself can be a vulnerability. An adve

The routing mechanism itself can be a vulnerability. An adversary could craft inputs designed to systematically overload specific experts or force the model into a low-capacity routing configuration, degrading performance or leaking information about the model's internal structure.

**Hypothesis**: TBD
**Suggested dataset**: TBD
**Baseline paper**: TBD
**Evidence**: EV006
**Difficulty**: medium

### G06: As the number of experts scales to thousands, the computatio

As the number of experts scales to thousands, the computational cost of the routing step (selecting top-k experts and computing their outputs) can become a bottleneck. Current strategies do not explicitly account for the communication and computation costs of the routing mechanism itself when making routing decisions.

**Hypothesis**: TBD
**Suggested dataset**: TBD
**Baseline paper**: TBD
**Evidence**: EV007
**Difficulty**: hard

## Citation Relationships

- No citation relationships detected

## MacBook-Runnable Experiments Available

None with both MacBook-compatible + code yet
