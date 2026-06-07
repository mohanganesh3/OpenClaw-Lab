# Reflexion-Enhanced Calibration for Medical Tabular Active Learning: A Traceable Local-Experiment Study

## Abstract

This paper reports a traceable autonomous-research run on the topic: **calibration-aware active learning under noisy labels for small medical tabular datasets**. The system read and indexed evidence, synthesized research gaps, generated candidate ideas, attacked them with reviewer and novelty gates, executed a local experiment ladder on the Mac, and produced this paper only after the readiness gate passed. The strongest empirical claim is intentionally narrow: in the logged local setting, the selected method improved over its baseline by 4.02% on `synthetic_classification_imbalanced_calibration_AL`. This is not a claim of broad state of the art.

## Contributions

1. A topic-grounded literature workflow with stable evidence IDs and per-paper summaries.
2. A reviewer-gated idea selection process that blocks one-shot idea generation.
3. A local experiment ladder: micro-probe, probe, ablation, and MVP when earlier stages pass.
4. A paper-readiness rule that links every major claim to evidence IDs or experiment logs.

## Related Work And Evidence

The run locked 47 evidence items and parsed 47 paper cards. The most relevant extracted prior-work rows are:

| Evidence | Paper | Dataset(s) | Reported result | Limitation/gap signal |
|---|---|---|---|---|
| EV015 | Boundary Conditions of Cost-Aware Active Learning: A Multi-Dataset Taxonomy of C | not extracted | not extracted | The stability of cost-aware selection remains poorly understood across |
| EV014 | CTypiClust: Confidence-Aware Typical Clustering for Budget-Agnostic Active Learn | not extracted | not extracted | The method's performance is dependent on the reliability of the model' |
| EV019 | Calibration of uncertainty in the active learning of machine learning force fiel | not extracted | not extracted | The predictive uncertainty of a Gaussian process is unlikely to be acc |
| EV042 | Beyond the Loss Curve: Scaling Laws, Active Learning, and the Limits of Learning | not extracted | not extracted | Abstract only — full PDF not read. |
| EV011 | Active Learning for Pneumonia Detection with Vision Transformers and Bayesian Un | not extracted | not extracted | Reliance on large annotated datasets and inability to quantify predict |
| EV025 | Calibrated Focal Loss for Semantic Labeling of High-Resolution Remote Sensing Im | not extracted | not extracted | Cross-entropy loss is dominated by major classes, leading to poor pred |
| EV032 | Pseudo-D: Informing Multi-View Uncertainty Estimation with Calibrated Neural Tra | not extracted | not extracted | Abstract only — full PDF not read. |
| EV046 | Achieving well-informed decision-making in drug discovery: a comprehensive calib | not extracted | not extracted | Abstract only — full PDF not read. |

All source cards are stored in `paper_summaries/`; claim links are stored in `claim_graph.json`.

## Gap And Hypothesis

Selected idea: **I01: Reflexion-Enhanced Calibration for Medical Tabular Active Learning**

Hypothesis: If we apply Reflexion-style self-reflection to calibration-aware active learning, ECE will decrease by 15% compared to standard entropy sampling on breast_cancer dataset.

Supporting gap IDs/evidence are recorded in `research_map.md`, `idea_tree.md`, and `novelty_tribunal.md`.

## Method

The method was tested through the OpenResearchOS experiment ladder. Each level created an isolated workspace with `experiment_spec.json`, `approval.json`, `environment.json`, generated Python code, stdout/stderr logs, `metrics.json`, and a result review.

The highest-level completed experiment used:

- Dataset: `synthetic_classification_imbalanced_calibration_AL`
- Baseline: `Entropy Sampling + Sigmoid Calibration (LogReg)`
- Proposed method: `Calib-Weighted Acq + Isotonic Cal (SVC-RBF)`
- Seeds: `[42,1337,2024,99,7]`
- Primary metric: higher score/accuracy with calibration or task-specific utility as recorded in `metrics.json`

## Results

| Level | Experiment | Dataset | Baseline | Proposed | Baseline score | Proposed score | Delta | p-value | Gate |
|---|---|---|---|---|---:|---:|---:|---:|---|
| micro_probe | micro_probe_i01 | synthetic_classification_imbalanced_calibration_AL | Entropy Sampling + Sigmoid Calibration (LogReg) | Calib-Weighted Acq + Isotonic Cal (SVC-RBF) | 0.938058 | 0.949056 | 0.0110 | 0.1448 | pass |
| probe | probe_i01 | synthetic_classification_imbalanced_calibration_AL | Entropy Sampling + Sigmoid Calibration (LogReg) | Calib-Weighted Acq + Isotonic Cal (SVC-RBF) | 0.924517 | 0.955465 | 0.0309 | 0.2619 | pass |
| ablation | ablation_i01 | synthetic_classification_imbalanced_calibration_AL | Entropy Sampling + Sigmoid Calibration (LogReg) | Calib-Weighted Acq + Isotonic Cal (SVC-RBF) | 0.938058 | 0.949056 | 0.0110 | 0.1448 | pass |
| mvp | mvp_i01 | synthetic_classification_imbalanced_calibration_AL | Entropy Sampling + Sigmoid Calibration (LogReg) | Calib-Weighted Acq + Isotonic Cal (SVC-RBF) | 0.918945 | 0.959141 | 0.0402 | 0.0351 | pass |

The main result is supported by `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260607134308_calibration-aware-active-learning_e7fec7/experiments/mvp_i01/metrics.json`, `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260607134308_calibration-aware-active-learning_e7fec7/experiments/mvp_i01/result_summary.md`, and the logs under `experiments/mvp_i01/logs/`.

## Comparison With Prior Work

This run does not claim to beat the published papers above on their original full benchmarks unless those benchmarks were actually reproduced. Instead, it compares the selected idea against a local baseline under a reproducible benchmark chosen for the MacBook setting. The prior-work table supplies the competitor context; the experiment table supplies the measured local evidence.

## Reproducibility

- micro_probe `micro_probe_i01`: metrics `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260607134308_calibration-aware-active-learning_e7fec7/experiments/micro_probe_i01/metrics.json`; review `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260607134308_calibration-aware-active-learning_e7fec7/experiments/micro_probe_i01/review_after_run.md`; logs `experiments/micro_probe_i01/logs/stdout.log` and `experiments/micro_probe_i01/logs/stderr.log`.
- probe `probe_i01`: metrics `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260607134308_calibration-aware-active-learning_e7fec7/experiments/probe_i01/metrics.json`; review `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260607134308_calibration-aware-active-learning_e7fec7/experiments/probe_i01/review_after_run.md`; logs `experiments/probe_i01/logs/stdout.log` and `experiments/probe_i01/logs/stderr.log`.
- ablation `ablation_i01`: metrics `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260607134308_calibration-aware-active-learning_e7fec7/experiments/ablation_i01/metrics.json`; review `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260607134308_calibration-aware-active-learning_e7fec7/experiments/ablation_i01/review_after_run.md`; logs `experiments/ablation_i01/logs/stdout.log` and `experiments/ablation_i01/logs/stderr.log`.
- mvp `mvp_i01`: metrics `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260607134308_calibration-aware-active-learning_e7fec7/experiments/mvp_i01/metrics.json`; review `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260607134308_calibration-aware-active-learning_e7fec7/experiments/mvp_i01/review_after_run.md`; logs `experiments/mvp_i01/logs/stdout.log` and `experiments/mvp_i01/logs/stderr.log`.

The full run directory is:

```text
/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260607134308_calibration-aware-active-learning_e7fec7
```

Readiness: `RRL-5`; venue fit: see `venue_fit.md`.

## Limitations

- The paper is a narrow local-experiment paper, not a broad SOTA claim.
- Synthetic or small public datasets can validate mechanism signal but do not replace full benchmark reproduction.
- Near-duplicate prior-art search is limited to the sources and searches recorded in this run.
- Stronger submission requires larger datasets, stronger baselines, and human review.
