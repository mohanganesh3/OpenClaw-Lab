import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Load dataset
data = load_digits()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create semi-supervised split
X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
    X_train, y_train, train_size=0.2, random_state=42
)

# Add label noise to simulate medical imaging challenges
noise_mask = np.random.random(len(y_labeled)) < 0.15
y_labeled_noisy = y_labeled.copy()
y_labeled_noisy[noise_mask] = np.random.randint(0, 10, np.sum(noise_mask))

def calculate_ece(y_true, y_pred_proba, n_bins=10):
    """Calculate Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred_proba >= bin_lower) & (y_pred_proba < bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred_proba[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def baseline_pseudo_label_selection(model, X_unlabeled, unlabeled_size, confidence_threshold=0.9):
    """Baseline pseudo-label selection based on confidence only"""
    probs = model.predict_proba(X_unlabeled)
    max_probs = np.max(probs, axis=1)
    
    # Select high-confidence predictions
    high_conf_mask = max_probs >= confidence_threshold
    selected_indices = np.where(high_conf_mask)[0]
    
    # Use all high-confidence samples (limit to unlabeled_size)
    n_select = min(len(selected_indices), unlabeled_size)
    selected_indices = selected_indices[:n_select]
    
    return selected_indices, max_probs[selected_indices]

def uncertainty_aware_pseudo_label_selection(model, X_unlabeled, unlabeled_size, confidence_threshold=0.9):
    """Uncertainty-aware pseudo-label selection"""
    probs = model.predict_proba(X_unlabeled)
    max_probs = np.max(probs, axis=1)
    
    # Calculate uncertainty using entropy
    uncertainties = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    
    # Select samples with both high confidence AND low uncertainty
    conf_mask = max_probs >= confidence_threshold
    uncertainty_threshold = np.percentile(uncertainties[conf_mask], 30)  # Bottom 30% uncertainty
    
    uncertainty_mask = (conf_mask) & (uncertainties <= uncertainty_threshold)
    selected_indices = np.where(uncertainty_mask)[0]
    
    # Use all selected samples (limit to unlabeled_size)
    n_select = min(len(selected_indices), unlabeled_size)
    selected_indices = selected_indices[:n_select]
    
    return selected_indices, max_probs[selected_indices]

# Experiment parameters
n_labeled = len(X_labeled)
n_unlabeled = len(X_unlabeled)
n_pseudo = min(200, n_unlabeled)  # Number of pseudo-labels to generate

# Baseline method
model_baseline = RandomForestClassifier(n_estimators=20, random_state=42)
model_baseline.fit(X_labeled, y_labeled_noisy)

selected_indices_baseline, conf_baseline = baseline_pseudo_label_selection(
    model_baseline, X_unlabeled, n_pseudo
)

# Add pseudo-labels to training set
X_train_baseline = np.vstack([X_labeled, X_unlabeled[selected_indices_baseline]])
y_train_baseline = np.concatenate([y_labeled_noisy, y_unlabeled[selected_indices_baseline]])

# Train final model
model_baseline_final = RandomForestClassifier(n_estimators=20, random_state=42)
model_baseline_final.fit(X_train_baseline, y_train_baseline)

# Predictions and ECE for baseline
y_pred_baseline = model_baseline_final.predict(X_test)
y_pred_proba_baseline = model_baseline_final.predict_proba(X_test)
ece_baseline = calculate_ece(y_test, y_pred_proba_baseline)

# Uncertainty-aware method
model_uncertainty = RandomForestClassifier(n_estimators=20, random_state=42)
model_uncertainty.fit(X_labeled, y_labeled_noisy)

selected_indices_uncertainty, conf_uncertainty = uncertainty_aware_pseudo_label_selection(
    model_uncertainty, X_unlabeled, n_pseudo
)

# Add pseudo-labels to training set
X_train_uncertainty = np.vstack([X_labeled, X_unlabeled[selected_indices_uncertainty]])
y_train_uncertainty = np.concatenate([y_labeled_noisy, y_unlabeled[selected_indices_uncertainty]])

# Train final model
model_uncertainty_final = RandomForestClassifier(n_estimators=20, random_state=42)
model_uncertainty_final.fit(X_train_uncertainty, y_train_uncertainty)

# Predictions and ECE for uncertainty-aware method
y_pred_uncertainty = model_uncertainty_final.predict(X_test)
y_pred_proba_uncertainty = model_uncertainty_final.predict_proba(X_test)
ece_uncertainty = calculate_ece(y_test, y_pred_proba_uncertainty)

# Calculate improvement
ece_reduction = ((ece_baseline - ece_uncertainty) / ece_baseline) * 100

# Determine if gap signal exists
gap_signal = ece_reduction >= 5.0

print(f"ECE Baseline: {ece_baseline:.4f}")
print(f"ECE Uncertainty-aware: {ece_uncertainty:.4f}")
print(f"ECE Reduction: {ece_reduction:.2f}%")
print(f"Gap Signal: {gap_signal}")

print(f"GAP_SIGNAL {gap_signal} {ece_reduction:.2f}")
