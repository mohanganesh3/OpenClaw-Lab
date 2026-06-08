import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import binned_statistic

np.random.seed(42)

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, 
                          n_informative=10, n_redundant=5, random_state=42)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create small labeled set (10% of training data)
labeled_idx = np.random.choice(len(X_train), size=int(0.1 * len(X_train)), replace=False)
X_labeled, X_unlabeled = X_train[labeled_idx], X_train[~np.isin(np.arange(len(X_train)), labeled_idx)]
y_labeled, y_unlabeled = y_train[labeled_idx], y_train[~np.isin(np.arange(len(y_train)), labeled_idx)]

# Function to calculate ECE
def calculate_ece(model, X, y, n_bins=10):
    probs = model.predict_proba(X)
    max_probs = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(predictions[in_bin] == y[in_bin])
            avg_confidence_in_bin = np.mean(max_probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

# Vanilla pseudo-labeling
model_vanilla = RandomForestClassifier(random_state=42)
model_vanilla.fit(X_labeled, y_labeled)
pseudo_labels = model_vanilla.predict(X_unlabeled)
X_combined = np.vstack([X_labeled, X_unlabeled])
y_combined = np.hstack([y_labeled, pseudo_labels])
model_vanilla_final = RandomForestClassifier(random_state=42)
model_vanilla_final.fit(X_combined, y_combined)

# Uncertainty-aware pseudo-labeling with temperature scaling
def temperature_scaling(probs, temperature):
    return probs ** (1.0 / temperature)

temp = 2.0  # Temperature parameter
model_temp = RandomForestClassifier(random_state=42)
model_temp.fit(X_labeled, y_labeled)
probs_unlabeled = model_temp.predict_proba(X_unlabeled)
temp_scaled_probs = temperature_scaling(probs_unlabeled, temp)
pseudo_labels_temp = np.argmax(temp_scaled_probs, axis=1)

# Only use high-confidence pseudo-labels
confidence_threshold = 0.8
high_conf_mask = np.max(temp_scaled_probs, axis=1) > confidence_threshold
X_labeled_temp = np.vstack([X_labeled, X_unlabeled[high_conf_mask]])
y_labeled_temp = np.hstack([y_labeled, y_unlabeled[high_conf_mask]])

model_temp_final = RandomForestClassifier(random_state=42)
model_temp_final.fit(X_labeled_temp, y_labeled_temp)

# Calculate ECEs
ece_vanilla = calculate_ece(model_vanilla_final, X_test, y_test)
ece_temp = calculate_ece(model_temp_final, X_test, y_test)

# Check if gap signal exists
gap_exists = ece_temp < ece_vanilla - 0.05
effect_size = ece_vanilla - ece_temp

print(f"GAP_SIGNAL {str(gap_exists).lower()} {effect_size:.4f}")
