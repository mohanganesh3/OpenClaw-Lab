# Idea Tree

Run: `run_20260608104616_uncertainty-aware-pseudo-label-sel_3e0902`

## Candidates

### I01: Domain-Adapted SAM for Medical Image Pseudo-Labeling

Status: `rejected`

Reviewer Average: 6.5

Next Action: `RUN_ADDITIONAL_SEARCH`

Metric: `ECE`

Pitch: EV001 demonstrates SAM-based pseudo-labeling suffers from domain shift in medical images. We propose adding a domain adaptation layer before SAM to mitigate this issue.

Mechanism: Implement a gradient reversal layer with domain classifier before SAM-based pseudo-labeling to align feature distributions between source and target domains.

Evidence Support: 1

Novelty Risk: 0.095

### I02: Temperature Scaling for Uncertainty Calibration in Pseudo-Labeling

Status: `rejected`

Reviewer Average: 6

Next Action: `RUN_ADDITIONAL_SEARCH`

Metric: `Brier score`

Pitch: EV003 highlights lack of calibration methods in uncertainty-aware pseudo-labeling. We propose implementing temperature scaling to improve calibration.

Mechanism: Apply temperature scaling to model outputs before pseudo-label selection, optimizing temperature parameter on validation set to minimize NLL.

Evidence Support: 1

Novelty Risk: 0.046

### I03: DIPS-Enhanced Data Cleaning for Pseudo-Label Selection

Status: `needs_revision`

Reviewer Average: 4.5

Next Action: `REVISE_IDEA`

Metric: `Accuracy`

Pitch: EV004 shows labeled data quality is crucial but lacks systematic cleaning approach. We implement DIPS-based cleaning before pseudo-labeling.

Mechanism: Apply DIPS algorithm to identify and clean noisy labels in the initial labeled set before pseudo-label selection.

Evidence Support: 1

Novelty Risk: 0.055

### I04: ATTA Adaptation for Regression-Based Pseudo-Labeling

Status: `shortlisted`

Reviewer Average: 6.74

Next Action: `RUN_MICRO_PROBE`

Metric: `AUC`

Pitch: EV010 notes unclear generalization of ATTA beyond classification. We adapt ATTA for regression tasks in pseudo-label selection.

Mechanism: Modify ATTA framework to work with regression outputs by using distance-based uncertainty estimation instead of classification entropy.

Evidence Support: 1

Novelty Risk: 0.05

### I05: Ensemble Uncertainty Estimation Combining SAM and ATTA

Status: `shortlisted`

Reviewer Average: 6.72

Next Action: `RUN_MICRO_PROBE`

Metric: `ECE`

Pitch: EV001 and EV010 both highlight domain shift as critical issue. We combine SAM and ATTA for robust uncertainty estimation.

Mechanism: Implement ensemble uncertainty estimation by averaging uncertainty scores from SAM and ATTA before pseudo-label selection.

Evidence Support: 1

Novelty Risk: 0.08

### I06: Monte Carlo Dropout for Uncertainty-Aware Pseudo-Labeling

Status: `shortlisted`

Reviewer Average: 6.74

Next Action: `RUN_MICRO_PROBE`

Metric: `Brier score`

Pitch: EV003 lacks detailed uncertainty estimation methods. We implement Monte Carlo dropout for better uncertainty quantification.

Mechanism: Apply Monte Carlo dropout with 30 forward passes to estimate predictive uncertainty before pseudo-label selection.

Evidence Support: 1

Novelty Risk: 0.056

