import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.calibration import calibration_curve

np.random.seed(42)

# Generate synthetic medical image-like data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                          n_redundant=5, n_classes=2, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize classifier
clf = RandomForestClassifier(n_estimators=10, random_state=42)

# Standard active learning (uncertainty sampling)
def standard_active_learning(X, y, initial_size=100, iterations=5):
    indices = np.random.choice(len(X), initial_size, replace=False)
    X_pool = np.delete(X, indices, axis=0)
    y_pool = np.delete(y, indices)
    X_labeled = X[indices]
    y_labeled = y[indices]
    
    results = []
    for _ in range(iterations):
        clf.fit(X_labeled, y_labeled)
        probs = clf.predict_proba(X_pool)
        uncertainties = np.max(probs, axis=1)
        n_select = min(50, len(X_pool))
        new_indices = np.argsort(uncertainties)[-n_select:]
        X_labeled = np.vstack([X_labeled, X_pool[new_indices]])
        y_labeled = np.hstack([y_labeled, y_pool[new_indices]])
        mask = np.ones(len(X_pool), dtype=bool)
        mask[new_indices] = False
        X_pool = X_pool[mask]
        y_pool = y_pool[mask]
        clf.fit(X_labeled, y_labeled)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        brier = brier_score_loss(y_test, clf.predict_proba(X_test)[:, 1])
        results.append((acc, brier))
    return results

# Calibration-aware active learning
def calibration_aware_active_learning(X, y, initial_size=100, iterations=5):
    indices = np.random.choice(len(X), initial_size, replace=False)
    X_pool = np.delete(X, indices, axis=0)
    y_pool = np.delete(y, indices)
    X_labeled = X[indices]
    y_labeled = y[indices]
    
    results = []
    for _ in range(iterations):
        clf.fit(X_labeled, y_labeled)
        probs = clf.predict_proba(X_pool)
        uncertainties = np.max(probs, axis=1)
        
        # Calculate calibration error for each sample
        calibration_errors = []
        for i in range(len(X_pool)):
            # Leave-one-out calibration check
            X_temp = np.vstack([X_labeled, X_pool[i:i+1]])
            y_temp = np.hstack([y_labeled, y_pool[i]])
            clf_temp = RandomForestClassifier(n_estimators=5, random_state=42)
            clf_temp.fit(X_temp, y_temp)
            prob = clf_temp.predict_proba(X_pool[i:i+1])[0, 1]
            error = abs(prob - y_pool[i])
            calibration_errors.append(error)
        
        # Combine uncertainty and calibration error
        combined_scores = uncertainties * (1 + np.array(calibration_errors))
        n_select = min(50, len(X_pool))
        new_indices = np.argsort(combined_scores)[-n_select:]
        
        X_labeled = np.vstack([X_labeled, X_pool[new_indices]])
        y_labeled = np.hstack([y_labeled, y_pool[new_indices]])
        mask = np.ones(len(X_pool), dtype=bool)
        mask[new_indices] = False
        X_pool = X_pool[mask]
        y_pool = y_pool[mask]
        
        clf.fit(X_labeled, y_labeled)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        brier = brier_score_loss(y_test, clf.predict_proba(X_test)[:, 1])
        results.append((acc, brier))
    return results

# Run both methods
standard_results = standard_active_learning(X, y)
calibration_results = calibration_aware_active_learning(X, y)

# Calculate average performance
standard_acc = np.mean([r[0] for r in standard_results])
standard_brier = np.mean([r[1] for r in standard_results])
calibration_acc = np.mean([r[0] for r in calibration_results])
calibration_brier = np.mean([r[1] for r in calibration_results])

# Calculate effect size (difference in Brier scores)
effect_size = abs(standard_brier - calibration_brier)

# Determine if gap signal exists (significant calibration improvement)
gap_signal = calibration_brier < standard_brier and effect_size > 0.01

print(f"GAP_SIGNAL {str(gap_signal).lower()} {effect_size:.4f}")
