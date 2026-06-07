# Idea Tree

Run: `run_20260607170750_uncertainty-aware-pseudo-label-sel_a76557`

## Candidates

### ID001: Uncertainty-calibrated pseudo-label acceptance under label noise

Status: `shortlisted`

Reviewer Average: 6.44

Next Action: `RUN_MICRO_PROBE`

Metric: `improvement_over_baseline`

Pitch: The evidence repeatedly studies uncertainty-aware semi-supervised medical imaging, but the selection rule for accepting pseudo-labels under noisy labeled seeds is still under-tested. A local proxy can test whether entropy, margin, and agreement filters reduce noisy pseudo-label promotion compared with naive confidence thresholding.

Mechanism: Entropy-filtered, class-balanced pseudo-label selection will improve held-out classification accuracy over naive confidence pseudo-labeling when the initial labeled set contains injected label noise.

Evidence Support: 0.9

Novelty Risk: 0.074

### ID002: Class-balanced pseudo-labeling for scarce medical labels

Status: `shortlisted`

Reviewer Average: 6.44

Next Action: `RUN_MICRO_PROBE`

Metric: `improvement_over_baseline`

Pitch: Several papers report gains from SSL, but small medical datasets can amplify class imbalance and confirmation bias. The gap is whether pseudo-label selection should explicitly balance classes instead of only using max probability.

Mechanism: Class-balanced low-entropy selection will reduce minority-class collapse and improve macro/weighted F1 against max-confidence selection.

Evidence Support: 0.9

Novelty Risk: 0.071

### ID003: Calibration versus accuracy tradeoff in medical pseudo-label selection

Status: `shortlisted`

Reviewer Average: 6.44

Next Action: `RUN_MICRO_PROBE`

Metric: `improvement_over_baseline`

Pitch: Uncertainty-aware methods often optimize accuracy-like metrics, while medical deployment needs calibrated confidence. The gap is to measure whether pseudo-label filters improve accuracy while preserving calibration under noisy labels.

Mechanism: Adding entropy and agreement constraints will improve accuracy without increasing expected calibration error compared with raw confidence pseudo-labeling.

Evidence Support: 0.9

Novelty Risk: 0.06

### ID004: Failure-aware rejection of ambiguous pseudo-labels

Status: `shortlisted`

Reviewer Average: 6.44

Next Action: `RUN_MICRO_PROBE`

Metric: `improvement_over_baseline`

Pitch: The papers expose diagnostic uncertainty and unlabeled medical data as recurring problems, but many methods do not make pseudo-label rejection decisions auditable. The gap is a traceable reject-or-accept ledger for ambiguous unlabeled examples.

Mechanism: A reject ledger based on entropy, margin, and disagreement will identify ambiguous samples that hurt baseline pseudo-label training.

Evidence Support: 0.9

Novelty Risk: 0.073

### ID005: Small-run reproducibility gap for SSL medical-image claims

Status: `shortlisted`

Reviewer Average: 6.42

Next Action: `RUN_MICRO_PROBE`

Metric: `improvement_over_baseline`

Pitch: Many relevant papers use medical datasets or segmentation pipelines that are expensive to reproduce. The gap is a MacBook-runnable proxy experiment that validates the mechanism before claiming medical-image contribution.

Mechanism: A reproducible small proxy can falsify or support the pseudo-label selection mechanism before a remote medical benchmark is attempted.

Evidence Support: 0.9

Novelty Risk: 0.077

### ID006: Prior-art differentiation for uncertainty-aware SSL medical classification

Status: `shortlisted`

Reviewer Average: 6.42

Next Action: `RUN_MICRO_PROBE`

Metric: `improvement_over_baseline`

Pitch: The field contains many uncertainty-aware SSL methods, especially for segmentation. A contribution must differentiate classification-specific pseudo-label selection under label noise from existing consistency, co-training, and evidential-learning methods.

Mechanism: A novelty tribunal comparing selection rule, uncertainty estimator, dataset setting, and failure mode will separate a narrow classification contribution from near-duplicate prior art.

Evidence Support: 0.9

Novelty Risk: 0.103

