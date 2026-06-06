import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.calibration import calibration_curve

np.random.seed(42)

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Simulate linguistic feedback: randomly correct uncertain predictions
def simulate_linguistic_feedback(model, X_pool, y_pool, n_feedback=20):
    probs = model.predict_proba(X_pool)
    uncertainties = 1 - np.max(probs, axis=1)
    top_uncertain = np.argsort(uncertainties)[:n_feedback]
    
    # "Linguistic" correction: flip labels with some probability
    corrected_indices = []
    for idx in top_uncertain:
        if np.random.random() < 0.3:  # 30% chance of correction
            corrected_indices.append(idx)
    
    return corrected_indices

# Baseline active learning without feedback
def active_learning_baseline(X_pool, y_pool, X_train, y_train, X_test, y_test, n_queries=50):
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    for _ in range(n_queries):
        probs = model.predict_proba(X_pool)
        uncertainties = 1 - np.max(probs, axis=1)
        idx = np.argmax(uncertainties)
        
        X_train = np.append(X_train, [X_pool[idx]], axis=0)
        y_train = np.append(y_train, [y_pool[idx]])
        
        X_pool = np.delete(X_pool, idx, axis=0)
        y_pool = np.delete(y_pool, idx, axis=0)
        
        model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_proba[:, 1])
    
    return acc, brier

# Active learning with linguistic feedback
def active_learning_with_feedback(X_pool, y_pool, X_train, y_train, X_test, y_test, n_queries=50):
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    for _ in range(n_queries):
        probs = model.predict_proba(X_pool)
        uncertainties = 1 - np.max(probs, axis=1)
        idx = np.argmax(uncertainties)
        
        X_train = np.append(X_train, [X_pool[idx]], axis=0)
        y_train = np.append(y_train, [y_pool[idx]])
        
        X_pool = np.delete(X_pool, idx, axis=0)
        y_pool = np.delete(y_pool, idx, axis=0)
        
        # Apply linguistic feedback
        feedback_idx = simulate_linguistic_feedback(model, X_pool, y_pool, n_feedback=5)
        if feedback_idx:
            # "Correct" uncertain predictions
            for f_idx in feedback_idx:
                if f_idx < len(X_pool):
                    # Simulate expert correction
                    y_pool[f_idx] = 1 - y_pool[f_idx]  # Flip label
                    X_train = np.append(X_train, [X_pool[f_idx]], axis=0)
                    y_train = np.append(y_train, [y_pool[f_idx]])
        
        model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_proba[:, 1])
    
    return acc, brier

# Run comparison
X_pool, X_val, y_pool, y_val = X_train, X_test, y_train, y_test
X_train_small, X_train_pool, y_train_small, y_pool = X_train[:50], X_train[50:], y_train[:50], y_train[50:]

baseline_acc, baseline_brier = active_learning_baseline(X_pool, y_pool, X_train_small, y_train_small, X_test, y_test)
feedback_acc, feedback_brier = active_learning_with_feedback(X_pool, y_pool, X_train_small, y_train_small, X_test, y_test)

# Calculate effect size (improvement in Brier score)
effect_size = (baseline_brier - feedback_brier) / baseline_brier

# Determine if gap signal exists
gap_signal = effect_size > 0.05  # 5% improvement threshold

print(f"GAP_SIGNAL {str(gap_signal).lower()} {effect_size:.4f}")
