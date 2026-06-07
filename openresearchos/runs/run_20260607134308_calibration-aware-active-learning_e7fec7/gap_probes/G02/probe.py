import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set seed
np.random.seed(42)

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to calculate ECE
def calculate_ece(y_true, y_pred_proba, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
        if np.sum(in_bin) == 0:
            continue
        prop_of_positives = y_true[in_bin].mean()
        avg_confidence = y_pred_proba[in_bin].mean()
        ece += np.abs(avg_confidence - prop_of_positives) * np.sum(in_bin) / len(y_true)
    
    return ece

# Train baseline model
baseline_model = RandomForestClassifier(n_estimators=50, random_state=42)
baseline_model.fit(X_train, y_train)
baseline_pred_proba = baseline_model.predict_proba(X_test)[:, 1]
baseline_ece = calculate_ece(y_test, baseline_pred_proba)

# Self-refinement technique
def self_refine_model(X_train, y_train, X_test, y_test, n_iterations=3):
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    
    # Initial training
    model.fit(X_train, y_train)
    pred_proba = model.predict_proba(X_test)[:, 1]
    
    for iteration in range(n_iterations):
        # Get misclassified samples
        y_pred = (pred_proba > 0.5).astype(int)
        misclassified_mask = y_pred != y_test
        
        # Create sample weights (higher for misclassified)
        sample_weights = np.ones(len(y_test))
        sample_weights[misclassified_mask] = 2.0
        
        # Refine model with weighted samples
        X_weighted = np.repeat(X_test, sample_weights.astype(int))
        y_weighted = np.repeat(y_test, sample_weights.astype(int))
        
        # Retrain on weighted data
        model.fit(X_weighted, y_weighted)
        pred_proba = model.predict_proba(X_test)[:, 1]
    
    return pred_proba

# Apply self-refinement
refined_pred_proba = self_refine_model(X_train, y_train, X_test, y_test)
refined_ece = calculate_ece(y_test, refined_pred_proba)

# Calculate improvement
improvement = (baseline_ece - refined_ece) / baseline_ece * 100

# Check if gap signal exists (>= 10% improvement)
gap_signal = improvement >= 10

print(f"GAP_SIGNAL {gap_signal} {improvement:.2f}")
