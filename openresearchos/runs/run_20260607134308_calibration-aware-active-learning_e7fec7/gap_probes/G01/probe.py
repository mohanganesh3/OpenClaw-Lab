import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
import time

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
            accuracy_in_bin = accuracy_score(y_true[in_bin], y_pred_proba[in_bin].argmax(axis=1))
            avg_confidence_in_bin = y_pred_proba[in_bin].max(axis=1).mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def add_label_noise(y, noise_rate=0.2):
    """Add random noise to labels"""
    y_noisy = y.copy()
    n_samples = len(y)
    n_noise = int(n_samples * noise_rate)
    noise_indices = np.random.choice(n_samples, n_noise, replace=False)
    y_noisy[noise_indices] = 1 - y_noisy[noise_indices]
    return y_noisy

def standard_active_learning(X, y, initial_size=50, query_size=20, iterations=3):
    """Standard active learning with uncertainty sampling"""
    X_pool = X.copy()
    y_pool = y.copy()
    X_labeled = []
    y_labeled = []
    
    # Initial labeled set
    initial_indices = np.random.choice(len(X_pool), initial_size, replace=False)
    X_labeled = X_pool[initial_indices]
    y_labeled = y_pool[initial_indices]
    
    # Remove initial from pool
    mask = np.ones(len(X_pool), dtype=bool)
    mask[initial_indices] = False
    X_pool = X_pool[mask]
    y_pool = y_pool[mask]
    
    for _ in range(iterations):
        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_labeled, y_labeled)
        
        # Query most uncertain samples
        y_proba = model.predict_proba(X_pool)
        uncertainties = 1 - np.max(y_proba, axis=1)
        query_indices = np.argsort(uncertainties)[:query_size]
        
        # Add to labeled set
        X_labeled = np.vstack([X_labeled, X_pool[query_indices]])
        y_labeled = np.concatenate([y_labeled, y_pool[query_indices]])
        
        # Remove from pool
        mask = np.ones(len(X_pool), dtype=bool)
        mask[query_indices] = False
        X_pool = X_pool[mask]
        y_pool = y_pool[mask]
    
    # Final model and predictions
    final_model = RandomForestClassifier(n_estimators=50, random_state=42)
    final_model.fit(X_labeled, y_labeled)
    y_pred_proba = final_model.predict_proba(X_pool)
    
    return y_pred_proba

def calibration_aware_active_learning(X, y, initial_size=50, query_size=20, iterations=3):
    """Calibration-aware active learning"""
    X_pool = X.copy()
    y_pool = y.copy()
    X_labeled = []
    y_labeled = []
    
    # Initial labeled set
    initial_indices = np.random.choice(len(X_pool), initial_size, replace=False)
    X_labeled = X_pool[initial_indices]
    y_labeled = y_pool[initial_indices]
    
    # Remove initial from pool
    mask = np.ones(len(X_pool), dtype=bool)
    mask[initial_indices] = False
    X_pool = X_pool[mask]
    y_pool = y_pool[mask]
    
    for _ in range(iterations):
        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_labeled, y_labeled)
        
        # Get predictions and apply calibration
        y_proba = model.predict_proba(X_pool)
        
        # Calibrate using temperature scaling (simplified)
        temp = np.clip(-np.log(np.mean(y_proba[:, 1] == y_pool)), 0.1, 5.0)
        y_proba_calibrated = np.exp(np.log(y_proba) / temp)
        y_proba_calibrated = y_proba_calibrated / y_proba_calibrated.sum(axis=1, keepdims=True)
        
        # Query samples that are both uncertain and poorly calibrated
        uncertainties = 1 - np.max(y_proba_calibrated, axis=1)
        calibration_error = np.abs(y_proba_calibrated[:, 1] - (y_pool == 1).astype(float))
        combined_score = uncertainties + 0.5 * calibration_error
        
        query_indices = np.argsort(combined_score)[:query_size]
        
        # Add to labeled set
        X_labeled = np.vstack([X_labeled, X_pool[query_indices]])
        y_labeled = np.concatenate([y_labeled, y_pool[query_indices]])
        
        # Remove from pool
        mask = np.ones(len(X_pool), dtype=bool)
        mask[query_indices] = False
        X_pool = X_pool[mask]
        y_pool = y_pool[mask]
    
    # Final model with calibration
    final_model = RandomForestClassifier(n_estimators=50, random_state=42)
    final_model.fit(X_labeled, y_labeled)
    
    # Calibrate final model
    calibrated_model = CalibratedClassifierCV(final_model, method='isotonic', cv='prefit')
    calibrated_model.fit(X_labeled, y_labeled)
    y_pred_proba = calibrated_model.predict_proba(X_pool)
    
    return y_pred_proba

# Set random seed
np.random.seed(42)

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Add noise to labels
y_train_noisy = add_label_noise(y_train, noise_rate=0.2)

# Run standard active learning
start_time = time.time()
y_pred_standard = standard_active_learning(X_train, y_train_noisy)
standard_ece = calculate_ece(y_test, y_pred_standard)
standard_time = time.time() - start_time

# Run calibration-aware active learning
start_time = time.time()
y_pred_calibrated = calibration_aware_active_learning(X_train, y_train_noisy)
calibrated_ece = calculate_ece(y_test, y_pred_calibrated)
calibrated_time = time.time() - start_time

# Calculate improvement
improvement = (standard_ece - calibrated_ece) / standard_ece * 100
gap_exists = improvement >= 15

print(f"Standard ECE: {standard_ece:.4f}")
print(f"Calibrated ECE: {calibrated_ece:.4f}")
print(f"Improvement: {improvement:.2f}%")
print(f"Time Standard: {standard_time:.2f}s")
print(f"Time Calibrated: {calibrated_time:.2f}s")

# Final output
print(f"GAP_SIGNAL {str(gap_exists).lower()} {improvement:.2f}")
