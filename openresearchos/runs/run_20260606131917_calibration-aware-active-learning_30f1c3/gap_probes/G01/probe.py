import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.calibration import calibration_curve

np.random.seed(42)

# Load and prepare data
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initial small labeled set
initial_size = 50
idx = np.random.choice(len(X_train), initial_size, replace=False)
X_labeled, y_labeled = X_train[idx], y_train[idx]
X_unlabeled = np.delete(X_train, idx, axis=0)
y_unlabeled = np.delete(y_train, idx, axis=0)

def train_and_evaluate(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)
    preds = clf.predict(X_test)
    return accuracy_score(y_test, preds), brier_score_loss(y_test, probs)

# Baseline with random sampling
baseline_acc, baseline_brier = train_and_evaluate(X_labeled, y_labeled)

# Active learning with uncertainty sampling
acc_history, brier_history = [], []
for _ in range(10):
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_labeled, y_labeled)
    
    # Get uncertainty (entropy) for unlabeled samples
    probs = clf.predict_proba(X_unlabeled)
    uncertainty = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    
    # Select most uncertain samples
    n_select = min(20, len(X_unlabeled))
    idx = np.argsort(uncertainty)[-n_select:]
    
    # Add to labeled set
    X_labeled = np.concatenate([X_labeled, X_unlabeled[idx]])
    y_labeled = np.concatenate([y_labeled, y_unlabeled[idx]])
    
    # Remove from unlabeled
    mask = np.ones(len(X_unlabeled), dtype=bool)
    mask[idx] = False
    X_unlabeled = X_unlabeled[mask]
    y_unlabeled = y_unlabeled[mask]
    
    # Evaluate
    acc, brier = train_and_evaluate(X_labeled, y_labeled)
    acc_history.append(acc)
    brier_history.append(brier)

# Check if uncertainty sampling significantly improves calibration
improvement = baseline_brier - brier_history[-1]
effect_size = improvement / baseline_brier if baseline_brier > 0 else 0

# Gap signal: if uncertainty sampling doesn't significantly improve calibration
gap_signal = improvement < 0.05 and effect_size < 0.1

print(f"GAP_SIGNAL {gap_signal} {effect_size:.4f}")
