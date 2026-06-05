# Research Map

Run: `run_20260603115027_neural-architecture-search-automat_339b0b`
Topic: neural architecture search automated machine learning (cycle 2)
Scope: `local_experiment`
Feasibility: Yes, this setup is feasible for NAS experiments, though computational constraints may limit the search space complexity. Suitable datasets include CIFAR-10/100 for image classification, Fashion-MNIST as a smaller alternative, and the UCI Wine Quality dataset for tabular data experiments. Expect NAS experiments to take approximately 12-48 hours depending on the search algorithm's complexity and the number of architectures evaluated. The 36GB RAM provides ample capacity for most standard NAS workf

## Summary

- **67 papers** discovered and deep-read
- **0 papers** with full text extraction
- **0 papers** with MacBook-runnable experiments
- **0 papers** with available code
- **0 citation relationships** mapped

## Method Comparison Matrix

| Paper | Dataset | Method | Best Result | Limitations |
|-------|---------|--------|-------------|-------------|
| No deep reads yet | | | | |

## Research Gaps

### G01: Current NAS methods are predominantly task-specific, searchi

Current NAS methods are predominantly task-specific, searching for a single optimal architecture for a fixed dataset and modality. A significant gap exists in developing NAS frameworks that can discover a single, highly generalizable architecture capable of effective transfer learning across diverse tasks and data types, such as vision-language models.

**Hypothesis**: TBD
**Suggested dataset**: TBD
**Baseline paper**: TBD
**Evidence**: EV001, EV002
**Difficulty**: hard

### G02: Hardware-aware NAS typically optimizes for proxy metrics lik

Hardware-aware NAS typically optimizes for proxy metrics like FLOPs or latency, which do not always correlate with real-world performance on a specific hardware-software stack. The gap is in moving from hardware-aware search to true hardware-software co-design, where the search process is directly influenced by the compiler and runtime environment.

**Hypothesis**: TBD
**Suggested dataset**: TBD
**Baseline paper**: TBD
**Evidence**: EV003
**Difficulty**: medium

### G03: Standard NAS optimizes for clean-data accuracy, often produc

Standard NAS optimizes for clean-data accuracy, often producing models that are brittle to distribution shifts and adversarial attacks. The gap is in creating NAS objectives that explicitly reward robustness and model calibration, embedding these desirable properties directly into the discovered architecture.

**Hypothesis**: TBD
**Suggested dataset**: TBD
**Baseline paper**: TBD
**Evidence**: EV004
**Difficulty**: medium

### G04: Even one-shot NAS methods require a costly supernet training

Even one-shot NAS methods require a costly supernet training phase. The frontier of 'Cycle 2' AutoML is to eliminate this training cost entirely, discovering high-performing architectures through inference-only methods or by leveraging pre-existing knowledge from vast model repositories.

**Hypothesis**: TBD
**Suggested dataset**: TBD
**Baseline paper**: TBD
**Evidence**: EV005
**Difficulty**: hard

### G05: AutoML is more than just model selection; it encompasses the

AutoML is more than just model selection; it encompasses the entire pipeline, including data preprocessing and augmentation. The current gap is the siloed approach where architecture search and augmentation policy search are separate, preventing the discovery of synergistic combinations between model structure and data transformations.

**Hypothesis**: TBD
**Suggested dataset**: TBD
**Baseline paper**: TBD
**Evidence**: EV006
**Difficulty**: medium

### G06: NAS often yields highly performant but opaque 'black box' ar

NAS often yields highly performant but opaque 'black box' architectures that are difficult for humans to understand or verify. The gap is in guiding the search with human-understandable principles, such as causal relationships or graph-theoretic properties, to produce architectures that are both high-performing and inherently interpretable.

**Hypothesis**: TBD
**Suggested dataset**: TBD
**Baseline paper**: TBD
**Evidence**: EV007
**Difficulty**: hard

## Citation Relationships

- No citation relationships detected

## MacBook-Runnable Experiments Available

None with both MacBook-compatible + code yet
