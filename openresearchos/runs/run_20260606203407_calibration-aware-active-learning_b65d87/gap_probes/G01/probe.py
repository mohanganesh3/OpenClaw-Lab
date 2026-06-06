import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

def get_ece(y_true, y_pred_proba, n_bins=10):
    """
    Calculates the Expected Calibration Error (ECE).
    """
    # If y_pred_proba is 2D, take the max probability (confidence) for each sample
    if y_pred_proba.ndim == 2:
        y_pred_proba = np.max(y_pred_proba, axis=1)
        
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Select samples within the bin
        in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            # Calculate accuracy and confidence for the samples in the bin
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred_proba[in_bin].mean()
            # Add the weighted absolute difference to the ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

# 1. Load and prepare data
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train a classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Predict probabilities on the test set
y_pred_proba = model.predict_proba(X_test)

# 5. Calculate the Expected Calibration Error (ECE)
ece_value = get_ece(y_test, y_pred_proba)

# 6. Determine the GAP_SIGNAL based on a calibration threshold
# A lower ECE indicates better calibration. We'll use 0.1 as a threshold.
calibration_threshold = 0.1
is_well_calibrated = ece_value < calibration_threshold

# 7. Print the final required line
print(f"GAP_SIGNAL {str(is_well_calibrated).lower()} {ece_value}")
