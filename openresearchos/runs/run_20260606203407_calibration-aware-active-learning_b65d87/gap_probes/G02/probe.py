import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)

# Load dataset
digits = load_digits()
X, y = digits.data, digits.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train baseline model
baseline_model = RandomForestClassifier(n_estimators=50, random_state=42)
baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict(X_test)
baseline_acc = accuracy_score(y_test, baseline_pred)

# Implement self-refinement: re-predict low confidence samples
secondary_model = RandomForestClassifier(n_estimators=30, random_state=42)
secondary_model.fit(X_train, y_train)

# Get prediction probabilities
proba = baseline_model.predict_proba(X_test)
max_proba = np.max(proba, axis=1)

# Threshold for low confidence (bottom 30% of confidence scores)
threshold = np.percentile(max_proba, 30)
low_conf_mask = max_proba < threshold

# Get predictions for low confidence samples
secondary_pred = secondary_model.predict(X_test[low_conf_mask])

# Combine predictions
self_refined_pred = baseline_pred.copy()
self_refined_pred[low_conf_mask] = secondary_pred

# Calculate accuracy
self_refined_acc = accuracy_score(y_test, self_refined_pred)
improvement = self_refined_acc - baseline_acc

print(f"GAP_SIGNAL {improvement >= 0.05} {improvement:.4f}")
