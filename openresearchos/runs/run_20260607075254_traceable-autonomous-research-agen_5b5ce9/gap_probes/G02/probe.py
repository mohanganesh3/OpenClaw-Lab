import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import clone

np.random.seed(42)

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_sample_efficiency(X_train_subset, y_train_subset, X_test, y_test):
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train_subset, y_train_subset)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, len(X_train_subset)

# Baseline approach: use 80% of data
baseline_subset = int(0.8 * len(X_train))
baseline_acc, baseline_samples = evaluate_sample_efficiency(
    X_train[:baseline_subset], y_train[:baseline_subset], X_test, y_test
)
baseline_efficiency = baseline_acc / baseline_samples

# Optimized approach: find minimum data needed for 95% of max accuracy
max_possible_acc = 1.0  # Upper bound
target_acc = 0.95 * max_possible_acc
optimized_efficiency = 0
min_samples_needed = baseline_samples

for n_samples in range(100, baseline_samples + 1, 50):
    acc, samples = evaluate_sample_efficiency(
        X_train[:n_samples], y_train[:n_samples], X_test, y_test
    )
    efficiency = acc / samples
    if efficiency > optimized_efficiency:
        optimized_efficiency = efficiency
        min_samples_needed = samples
    if acc >= target_acc:
        break

improvement_ratio = (optimized_efficiency - baseline_efficiency) / baseline_efficiency

print(f"GAP_SIGNAL {'true' if improvement_ratio >= 0.3 else 'false'} {improvement_ratio:.3f}")
