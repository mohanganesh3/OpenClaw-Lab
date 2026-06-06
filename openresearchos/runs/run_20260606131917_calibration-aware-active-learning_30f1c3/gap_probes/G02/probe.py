import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.calibration import calibration_curve

np.random.seed(42)

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, 
                          n_informative=10, n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=42, stratify=y)

# Initialize models
model_standard = RandomForestClassifier(n_estimators=50, random_state=42)
model_reflective = RandomForestClassifier(n_estimators=50, random_state=42)

# Function to calculate ECE
def calculate_ece(model, X, y, n_bins=10):
    prob_true, prob_pred = calibration_curve(y, model.predict_proba(X), n_bins=n_bins)
    ece = np.sum((prob_pred - prob_true) * len(prob_true)) / len(y)
    return ece

# Active learning with self-reflection
def active_learning_with_reflection(X_pool, y_pool, X_train, y_train, model, 
                                   iterations=20, batch_size=20):
    pool_indices = list(range(len(X_pool)))
    
    for _ in range(iterations):
        # Train model
        model.fit(X_train, y_train)
        
        # Get prediction uncertainties
        probs = model.predict_proba(X_pool)
        uncertainties = 1 - np.max(probs, axis=1)
        
        # Self-reflection: Analyze recent calibration changes
        if len(y_train) > 10:
            recent_ece = calculate_ece(model, X_train[-50:], y_train[-50:])
            # If calibration is poor, prioritize diverse samples
            if recent_ece > 0.15:
                # Add diversity bonus to uncertainty
                diversity = np.std(probs, axis=0)
                scores = uncertainties + 0.1 * diversity
            else:
                scores = uncertainties
        else:
            scores = uncertainties
        
        # Select most uncertain samples
        selected_idx = np.argsort(scores)[-batch_size:]
        
        # Move selected samples to training set
        for idx in selected_idx:
            X_train = np.vstack([X_train, X_pool[idx]])
            y_train = np.append(y_train, y_pool[idx])
            pool_indices.remove(idx)
            X_pool = np.delete(X_pool, idx, axis=0)
            y_pool = np.delete(y_pool, idx)
    
    return model

# Run experiment
X_pool_standard, X_pool_reflective = X_train.copy(), X_train.copy()
y_pool_standard, y_pool_reflective = y_train.copy(), y_train.copy()

# Standard AL
X_train_std, y_train_std = X_train[:50], y_train[:50]
model_standard = active_learning_with_reflection(X_pool_standard, y_pool_standard, 
                                                X_train_std, y_train_std, model_standard)

# Reflective AL
X_train_ref, y_train_ref = X_train[:50], y_train[:50]
model_reflective = active_learning_with_reflection(X_pool_reflective, y_pool_reflective, 
                                                  X_train_ref, y_train_ref, model_reflective)

# Calculate ECE on test set
ece_std = calculate_ece(model_standard, X_test, y_test)
ece_ref = calculate_ece(model_reflective, X_test, y_test)

# Measure effect size
effect_size = abs(ece_ref - ece_std)

print(f"GAP_SIGNAL {'true' if effect_size > 0.01 else 'false'} {effect_size:.4f}")
