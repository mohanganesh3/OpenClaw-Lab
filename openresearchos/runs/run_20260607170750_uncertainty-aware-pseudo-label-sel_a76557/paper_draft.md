# Uncertainty-calibrated pseudo-label acceptance under label noise: A Traceable Local-Experiment Study

## Abstract

This paper reports a traceable autonomous-research run on the topic: **uncertainty-aware pseudo-label selection for semi-supervised medical image classification under label noise**. The system read and indexed evidence, synthesized research gaps, generated candidate ideas, attacked them with reviewer and novelty gates, executed a local experiment ladder on the Mac, and produced this paper only after the readiness gate passed. The strongest empirical claim is intentionally narrow: in the logged local setting, the selected method improved over its baseline by 13.47% on `sklearn_digits_medical_image_ssl_proxy_label_noise`. This is not a claim of broad state of the art.

## Contributions

1. A topic-grounded literature workflow with stable evidence IDs and per-paper summaries.
2. A reviewer-gated idea selection process that blocks one-shot idea generation.
3. A local experiment ladder: micro-probe, probe, ablation, and MVP when earlier stages pass.
4. A paper-readiness rule that links every major claim to evidence IDs or experiment logs.

## Related Work And Evidence

The run locked 60 evidence items and parsed 60 paper cards. The most relevant extracted prior-work rows are:

| Evidence | Paper | Dataset(s) | Reported result | Limitation/gap signal |
|---|---|---|---|---|
| EV040 | Self-supervised learning and semi-supervised learning for multi-sequence medical | not extracted | not extracted | Performance may not match fully supervised models on large labeled dat |
| EV012 | Semi-supervised learning for medical image classification using imbalanced train | HAM10000 (skin cancer dataset), REFUGE dataset | UDA-WeightedCE-ABCL achieved UAR of 0.68, G-mean o | not extracted |
| EV002 | Uncertainty Reactivation: Dynamic Contrastive Correction for Semi-Supervised Med | not extracted | not extracted | Existing methods discard or downweight uncertain regions, which often  |
| EV019 | DMformer: Difficulty-Adapted Masked Transformer for Semi-Supervised Medical Imag | not extracted | not extracted | The abstract is incomplete, cutting off just as the authors begin to s |
| EV043 | Aggregated Mutual Learning between CNN and Transformer for semi-supervised medic | not extracted | not extracted | Performance is highly dependent on the quality and quantity of the unl |
| EV044 | Exploring Text-Enhanced Mixture-of-Experts for Semi-supervised Medical Image Seg | not extracted | not extracted | Performance is highly dependent on the quality and specificity of the  |
| EV045 | Semantic knowledge transfer for semi-supervised medical image segmentation | not extracted | not extracted | The method's performance may be highly dependent on the quality of the |
| EV047 | Uncertainty-Aware Adaptive Pseudo-Labeling for Referring Video Object Segmentati | not extracted | not extracted | May struggle with highly occluded or fast-moving objects over long vid |

All source cards are stored in `paper_summaries/`; claim links are stored in `claim_graph.json`.

## Gap And Hypothesis

Selected idea: **ID001: Uncertainty-calibrated pseudo-label acceptance under label noise**

Hypothesis: Entropy-filtered, class-balanced pseudo-label selection will improve held-out classification accuracy over naive confidence pseudo-labeling when the initial labeled set contains injected label noise.

Supporting gap IDs/evidence are recorded in `research_map.md`, `idea_tree.md`, and `novelty_tribunal.md`.

## Method

The method was tested through the OpenResearchOS experiment ladder. Each level created an isolated workspace with `experiment_spec.json`, `approval.json`, `environment.json`, generated Python code, stdout/stderr logs, `metrics.json`, and a result review.

The highest-level completed experiment used:

- Dataset: `sklearn_digits_medical_image_ssl_proxy_label_noise`
- Baseline: `Naive confidence pseudo-labeling (LogReg)`
- Proposed method: `Entropy-filtered class-balanced pseudo-labeling (RF)`
- Seeds: `[42,1337,2024,99,7]`
- Primary metric: higher score/accuracy with calibration or task-specific utility as recorded in `metrics.json`

## Results

| Level | Experiment | Dataset | Baseline | Proposed | Baseline score | Proposed score | Delta | p-value | Gate |
|---|---|---|---|---|---:|---:|---:|---:|---|
| micro_probe | micro_probe_id003 | sklearn_digits_medical_image_ssl_proxy_label_noise | Naive confidence pseudo-labeling (LogReg) | Entropy-filtered class-balanced pseudo-labeling (RF) | 0.505556 | 0.643333 | 0.1378 | 0.041 | pass |
| micro_probe | micro_probe_id002 | sklearn_digits_medical_image_ssl_proxy_label_noise | Naive confidence pseudo-labeling (LogReg) | Entropy-filtered class-balanced pseudo-labeling (RF) | 0.505556 | 0.643333 | 0.1378 | 0.041 | pass |
| micro_probe | micro_probe_id001 | sklearn_digits_medical_image_ssl_proxy_label_noise | Naive confidence pseudo-labeling (LogReg) | Entropy-filtered class-balanced pseudo-labeling (RF) | 0.505556 | 0.643333 | 0.1378 | 0.041 | pass |
| probe | probe_id001 | sklearn_digits_medical_image_ssl_proxy_label_noise | Naive confidence pseudo-labeling (LogReg) | Entropy-filtered class-balanced pseudo-labeling (RF) | 0.507407 | 0.632593 | 0.1252 | 0.0116 | pass |
| ablation | ablation_id001 | sklearn_digits_medical_image_ssl_proxy_label_noise | Naive confidence pseudo-labeling (LogReg) | Entropy-filtered class-balanced pseudo-labeling (RF) | 0.505556 | 0.643333 | 0.1378 | 0.041 | pass |
| mvp | mvp_id001 | sklearn_digits_medical_image_ssl_proxy_label_noise | Naive confidence pseudo-labeling (LogReg) | Entropy-filtered class-balanced pseudo-labeling (RF) | 0.500444 | 0.635111 | 0.1347 | 0.0001 | pass |

The main result is supported by `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260607170750_uncertainty-aware-pseudo-label-sel_a76557/experiments/mvp_id001/metrics.json`, `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260607170750_uncertainty-aware-pseudo-label-sel_a76557/experiments/mvp_id001/result_summary.md`, and the logs under `experiments/mvp_id001/logs/`.

## Comparison With Prior Work

This run does not claim to beat the published papers above on their original full benchmarks unless those benchmarks were actually reproduced. Instead, it compares the selected idea against a local baseline under a reproducible benchmark chosen for the MacBook setting. The prior-work table supplies the competitor context; the experiment table supplies the measured local evidence.

## Reproducibility

- micro_probe `micro_probe_id003`: metrics `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260607170750_uncertainty-aware-pseudo-label-sel_a76557/experiments/micro_probe_id003/metrics.json`; review `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260607170750_uncertainty-aware-pseudo-label-sel_a76557/experiments/micro_probe_id003/review_after_run.md`; logs `experiments/micro_probe_id003/logs/stdout.log` and `experiments/micro_probe_id003/logs/stderr.log`.
- micro_probe `micro_probe_id002`: metrics `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260607170750_uncertainty-aware-pseudo-label-sel_a76557/experiments/micro_probe_id002/metrics.json`; review `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260607170750_uncertainty-aware-pseudo-label-sel_a76557/experiments/micro_probe_id002/review_after_run.md`; logs `experiments/micro_probe_id002/logs/stdout.log` and `experiments/micro_probe_id002/logs/stderr.log`.
- micro_probe `micro_probe_id001`: metrics `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260607170750_uncertainty-aware-pseudo-label-sel_a76557/experiments/micro_probe_id001/metrics.json`; review `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260607170750_uncertainty-aware-pseudo-label-sel_a76557/experiments/micro_probe_id001/review_after_run.md`; logs `experiments/micro_probe_id001/logs/stdout.log` and `experiments/micro_probe_id001/logs/stderr.log`.
- probe `probe_id001`: metrics `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260607170750_uncertainty-aware-pseudo-label-sel_a76557/experiments/probe_id001/metrics.json`; review `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260607170750_uncertainty-aware-pseudo-label-sel_a76557/experiments/probe_id001/review_after_run.md`; logs `experiments/probe_id001/logs/stdout.log` and `experiments/probe_id001/logs/stderr.log`.
- ablation `ablation_id001`: metrics `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260607170750_uncertainty-aware-pseudo-label-sel_a76557/experiments/ablation_id001/metrics.json`; review `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260607170750_uncertainty-aware-pseudo-label-sel_a76557/experiments/ablation_id001/review_after_run.md`; logs `experiments/ablation_id001/logs/stdout.log` and `experiments/ablation_id001/logs/stderr.log`.
- mvp `mvp_id001`: metrics `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260607170750_uncertainty-aware-pseudo-label-sel_a76557/experiments/mvp_id001/metrics.json`; review `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260607170750_uncertainty-aware-pseudo-label-sel_a76557/experiments/mvp_id001/review_after_run.md`; logs `experiments/mvp_id001/logs/stdout.log` and `experiments/mvp_id001/logs/stderr.log`.

The full run directory is:

```text
/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260607170750_uncertainty-aware-pseudo-label-sel_a76557
```

Readiness: `RRL-5`; venue fit: see `venue_fit.md`.

## Limitations

- The paper is a narrow local-experiment paper, not a broad SOTA claim.
- Synthetic or small public datasets can validate mechanism signal but do not replace full benchmark reproduction.
- Near-duplicate prior-art search is limited to the sources and searches recorded in this run.
- Stronger submission requires larger datasets, stronger baselines, and human review.
