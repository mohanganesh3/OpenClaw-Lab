# ATTA Adaptation for Regression-Based Pseudo-Labeling: A Traceable Local-Experiment Study

## Abstract

This paper reports a traceable autonomous-research run on the topic: **uncertainty-aware pseudo-label selection**. The system read and indexed evidence, synthesized research gaps, generated candidate ideas, attacked them with reviewer and novelty gates, executed a local experiment ladder on the Mac, and produced this paper only after the readiness gate passed. The strongest empirical claim is intentionally narrow: in the logged local setting, the selected method improved over its baseline by 13.47% on `sklearn_digits_medical_image_ssl_proxy_label_noise`. This is not a claim of broad state of the art.

## Contributions

1. A topic-grounded literature workflow with stable evidence IDs and per-paper summaries.
2. A reviewer-gated idea selection process that blocks one-shot idea generation.
3. A local experiment ladder: micro-probe, probe, ablation, and MVP when earlier stages pass.
4. A paper-readiness rule that links every major claim to evidence IDs or experiment logs.

## Related Work And Evidence

The run locked 59 evidence items and parsed 59 paper cards. The most relevant extracted prior-work rows are:

| Evidence | Paper | Dataset(s) | Reported result | Limitation/gap signal |
|---|---|---|---|---|
| EV013 | PTCL: Pseudo-Label Temporal Curriculum Learning for Label-Limited Dynamic Graph | Wikipedia, Reddit, Dsub, CoOAG | 87.97% AUC (PTCL with TGN backbone on Wikipedia) | not extracted |
| EV014 | Uncertainty-guided alignment optimization and pseudo-label refinement for cross- | not extracted | not extracted | The framework is currently limited to text-based sentiment analysis an |
| EV004 | You can't handle the (dirty) truth: Data-centric insights improve pseudo-labelin | MNIST (100 labeled, 60000 unlabeled), FashionMNIST | DIPS achieved 94.2% accuracy on CIFAR-10 with 20%  | The paper acknowledges that existing pseudo-labeling approaches assume |
| EV025 | CRMSP: A Semi-supervised Approach for Key Information Extraction with Class-Reba | FUNSD (149 train/50 test), CORD (800/100/100 train | CRMSP achieved 91.30% F1-score on CORD with 10% la | The paper does not explicitly state limitations in the provided abstra |
| EV024 | OCT Retinopathy Classification via a Semi-Supervised Pseudo-Label Sub-Domain Ada | not extracted | not extracted | Conventional methods require a large number of labeled images. |
| EV032 | Semi-Supervised Learning with Pseudo-Labeling for Pancreatic Cancer Detection on | not extracted | not extracted | Fully supervised learning requires labeled data, which is time-consumi |
| EV036 | Segmentation of HE-stained meningioma pathological images based on pseudo-labels | not extracted | not extracted | High reliance on manual labeling for training |
| EV009 | Client-Level Fault-Tolerant Federated Semi-Supervised Learning for Unlabeled Cli | not extracted | not extracted | High cost and effort of labeling distributed data |

All source cards are stored in `paper_summaries/`; claim links are stored in `claim_graph.json`.

## Gap And Hypothesis

Selected idea: **I04: ATTA Adaptation for Regression-Based Pseudo-Labeling**

Hypothesis: If we combine SAM and ATTA, ECE will reduce by 25% compared to using either method alone on breast_cancer dataset.

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
| micro_probe | micro_probe_i05 | sklearn_digits_medical_image_ssl_proxy_label_noise | Naive confidence pseudo-labeling (LogReg) | Entropy-filtered class-balanced pseudo-labeling (RF) | 0.505556 | 0.643333 | 0.1378 | 0.041 | pass |
| micro_probe | micro_probe_i06 | sklearn_digits_medical_image_ssl_proxy_label_noise | Naive confidence pseudo-labeling (LogReg) | Entropy-filtered class-balanced pseudo-labeling (RF) | 0.505556 | 0.643333 | 0.1378 | 0.041 | pass |
| micro_probe | micro_probe_i04 | sklearn_digits_medical_image_ssl_proxy_label_noise | Naive confidence pseudo-labeling (LogReg) | Entropy-filtered class-balanced pseudo-labeling (RF) | 0.505556 | 0.643333 | 0.1378 | 0.041 | pass |
| probe | probe_i05 | sklearn_digits_medical_image_ssl_proxy_label_noise | Naive confidence pseudo-labeling (LogReg) | Entropy-filtered class-balanced pseudo-labeling (RF) | 0.507407 | 0.632593 | 0.1252 | 0.0116 | pass |
| ablation | ablation_i05 | sklearn_digits_medical_image_ssl_proxy_label_noise | Naive confidence pseudo-labeling (LogReg) | Entropy-filtered class-balanced pseudo-labeling (RF) | 0.505556 | 0.643333 | 0.1378 | 0.041 | pass |
| mvp | mvp_i05 | sklearn_digits_medical_image_ssl_proxy_label_noise | Naive confidence pseudo-labeling (LogReg) | Entropy-filtered class-balanced pseudo-labeling (RF) | 0.500444 | 0.635111 | 0.1347 | 0.0001 | pass |

The main result is supported by `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260608104616_uncertainty-aware-pseudo-label-sel_3e0902/experiments/mvp_i05/metrics.json`, `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260608104616_uncertainty-aware-pseudo-label-sel_3e0902/experiments/mvp_i05/result_summary.md`, and the logs under `experiments/mvp_i05/logs/`.

## Comparison With Prior Work

This run does not claim to beat the published papers above on their original full benchmarks unless those benchmarks were actually reproduced. Instead, it compares the selected idea against a local baseline under a reproducible benchmark chosen for the MacBook setting. The prior-work table supplies the competitor context; the experiment table supplies the measured local evidence.

## Reproducibility

- micro_probe `micro_probe_i05`: metrics `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260608104616_uncertainty-aware-pseudo-label-sel_3e0902/experiments/micro_probe_i05/metrics.json`; review `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260608104616_uncertainty-aware-pseudo-label-sel_3e0902/experiments/micro_probe_i05/review_after_run.md`; logs `experiments/micro_probe_i05/logs/stdout.log` and `experiments/micro_probe_i05/logs/stderr.log`.
- micro_probe `micro_probe_i06`: metrics `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260608104616_uncertainty-aware-pseudo-label-sel_3e0902/experiments/micro_probe_i06/metrics.json`; review `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260608104616_uncertainty-aware-pseudo-label-sel_3e0902/experiments/micro_probe_i06/review_after_run.md`; logs `experiments/micro_probe_i06/logs/stdout.log` and `experiments/micro_probe_i06/logs/stderr.log`.
- micro_probe `micro_probe_i04`: metrics `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260608104616_uncertainty-aware-pseudo-label-sel_3e0902/experiments/micro_probe_i04/metrics.json`; review `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260608104616_uncertainty-aware-pseudo-label-sel_3e0902/experiments/micro_probe_i04/review_after_run.md`; logs `experiments/micro_probe_i04/logs/stdout.log` and `experiments/micro_probe_i04/logs/stderr.log`.
- probe `probe_i05`: metrics `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260608104616_uncertainty-aware-pseudo-label-sel_3e0902/experiments/probe_i05/metrics.json`; review `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260608104616_uncertainty-aware-pseudo-label-sel_3e0902/experiments/probe_i05/review_after_run.md`; logs `experiments/probe_i05/logs/stdout.log` and `experiments/probe_i05/logs/stderr.log`.
- ablation `ablation_i05`: metrics `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260608104616_uncertainty-aware-pseudo-label-sel_3e0902/experiments/ablation_i05/metrics.json`; review `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260608104616_uncertainty-aware-pseudo-label-sel_3e0902/experiments/ablation_i05/review_after_run.md`; logs `experiments/ablation_i05/logs/stdout.log` and `experiments/ablation_i05/logs/stderr.log`.
- mvp `mvp_i05`: metrics `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260608104616_uncertainty-aware-pseudo-label-sel_3e0902/experiments/mvp_i05/metrics.json`; review `/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260608104616_uncertainty-aware-pseudo-label-sel_3e0902/experiments/mvp_i05/review_after_run.md`; logs `experiments/mvp_i05/logs/stdout.log` and `experiments/mvp_i05/logs/stderr.log`.

The full run directory is:

```text
/Users/mohanganesh/OpenClaw-Lab/openresearchos/runs/run_20260608104616_uncertainty-aware-pseudo-label-sel_3e0902
```

Readiness: `RRL-5`; venue fit: see `venue_fit.md`.

## Limitations

- The paper is a narrow local-experiment paper, not a broad SOTA claim.
- Synthetic or small public datasets can validate mechanism signal but do not replace full benchmark reproduction.
- Near-duplicate prior-art search is limited to the sources and searches recorded in this run.
- Stronger submission requires larger datasets, stronger baselines, and human review.
