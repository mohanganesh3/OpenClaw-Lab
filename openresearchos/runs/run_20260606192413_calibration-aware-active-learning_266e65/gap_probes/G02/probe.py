import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.base import clone

np.random.seed(42)

# Load dataset
data = load_digits()
X, y = data.data, data.target
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Split initial training set
X_train, X_pool, y_train, y_pool = train_test_split(
    X_train_full, y_train_full, test_size=0.5, random_state=42
)

def compositional_regularization(model, X, alpha=0.1):
    """Add compositional regularization by penalizing feature interactions"""
    n_features = X.shape[1]
    feature_importance = np.abs(model.feature_importances_)
    # Penalize uneven feature importance distribution
    entropy = -np.sum(feature_importance * np.log(feature_importance + 1e-10))
    regularization = alpha * (1 - entropy / np.log(n_features))
    return regularization

def active_learning_loop(X_train, y_train, X_pool, y_pool, n_queries=50, use_compositional=False):
    """Active learning with calibration awareness"""
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    calibrated_model = CalibratedClassifierCV(model, cv=3)
    
    for _ in range(n_queries):
        # Train model
        model.fit(X_train, y_train)
        
        # Calibrate
        calibrated_model.fit(X_train, y_train)
        
        # Get uncertainty scores
        if use_compositional:
            # Add compositional regularization effect
            reg_term = compositional_regularization(model, X_pool)
            probs = calibrated_model.predict_proba(X_pool)
            # Adjust uncertainty with regularization
            uncertainty = -np.sum(probs * np.log(probs + 1e-10), axis=1) + reg_term
        else:
            uncertainty = -np.sum(calibrated_model.predict_proba(X_pool) * 
                                 np.log(calibrated_model.predict_proba(X_pool) + 1e-10), axis=1)
        
        # Select most uncertain samples
        n_select = min(10, len(X_pool))
        indices = np.argsort(uncertainty)[-n_select:]
        
        # Add to training set
        X_train = np.vstack([X_train, X_pool[indices]])
        y_train = np.hstack([y_train, y_pool[indices]])
        
        # Remove from pool
        mask = np.ones(len(X_pool), dtype=bool)
        mask[indices] = False
        X_pool = X_pool[mask]
        y_pool = y_pool[mask]
    
    # Final evaluation
    final_model = RandomForestClassifier(n_estimators=50, random_state=42)
    final_model.fit(X_train, y_train)
    return accuracy_score(y_test, final_model.predict(X_test))

# Run baseline experiment
baseline_acc = active_learning_loop(X_train.copy(), y_train.copy(), 
                                   X_pool.copy(), y_pool.copy(), 
                                   use_compositional=False)

# Run experiment with compositional regularization
comp_acc = active_learning_loop(X_train.copy(), y_train.copy(), 
                               X_pool.copy(), y_pool.copy(), 
                               use_compositional=True)

# Calculate effect size
effect_size = (comp_acc - baseline_acc) * 100

# Check if gap signal exists (5% improvement threshold)
gap_signal = effect_size >= 5.0

print(f"GAP_SIGNAL {gap_signal} {effect_size:.2f}")
