import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss
import scipy.stats as stats

np.random.seed(42)

# Load data
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize models
model_standard = RandomForestClassifier(n_estimators=50, random_state=42)
model_calibrated = RandomForestClassifier(n_estimators=50, random_state=42)

# Active learning parameters
initial_size = 50
batch_size = 20
n_iterations = 5

# Storage for results
standard_scores = []
calibrated_scores = []

# Active learning loop
for iteration in range(n_iterations):
    # Get current labeled data
    current_size = initial_size + iteration * batch_size
    indices = np.random.choice(len(X_train), current_size, replace=False)
    X_current = X_train[indices]
    y_current = y_train[indices]
    
    # Train standard model
    model_standard.fit(X_current, y_current)
    y_pred_proba = model_standard.predict_proba(X_test)[:, 1]
    standard_score = brier_score_loss(y_test, y_pred_proba)
    standard_scores.append(standard_score)
    
    # Train calibrated model
    model_calibrated.fit(X_current, y_current)
    y_pred_proba_calibrated = model_calibrated.predict_proba(X_test)[:, 1]
    
    # Calculate calibration metrics
    calibration_scores = []
    for i in range(10):
        mask = (y_pred_proba_calibrated >= i/10) & (y_pred_proba_calibrated < (i+1)/10)
        if np.sum(mask) > 0:
            group_probs = y_pred_proba_calibrated[mask]
            group_labels = y_test[mask]
            calibration_scores.append(np.abs(np.mean(group_probs) - np.mean(group_labels)))
    
    # Combine uncertainty and calibration for selection
    uncertainty = 1 - np.max(model_calibrated.predict_proba(X_train), axis=1)
    calibration_factor = np.mean(calibration_scores) if calibration_scores else 0
    combined_scores = uncertainty * (1 + calibration_factor)
    
    # Select most uncertain and poorly calibrated samples
    additional_indices = np.argsort(combined_scores)[-batch_size:]
    X_current = np.concatenate([X_current, X_train[additional_indices]])
    y_current = np.concatenate([y_current, y_train[additional_indices]])
    
    # Evaluate calibrated model
    y_pred_proba_calibrated = model_calibrated.predict_proba(X_test)[:, 1]
    calibrated_score = brier_score_loss(y_test, y_pred_proba_calibrated)
    calibrated_scores.append(calibrated_score)

# Calculate effect size
mean_standard = np.mean(standard_scores)
mean_calibrated = np.mean(calibrated_scores)
effect_size = abs(mean_standard - mean_calibrated) / np.std(standard_scores)

# Determine if gap signal exists
gap_signal = effect_size > 0.5

print(f"GAP_SIGNAL {gap_signal} {effect_size:.4f}")
