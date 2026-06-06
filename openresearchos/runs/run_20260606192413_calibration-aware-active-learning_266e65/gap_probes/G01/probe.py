import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def compute_ece(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error"""
    # Use max probability as confidence
    confidence = y_prob.max(axis=1)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidence >= bin_lower) & (confidence < bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = confidence[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return float(ece)

def standard_active_learning(X, y, pool_size, iterations):
    """Standard active learning without calibration awareness"""
    # Initial split
    X_train, X_pool, y_train, y_pool = train_test_split(
        X, y, train_size=pool_size // 2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    ece_history = []
    
    for _ in range(iterations):
        # Get predictions on pool
        y_prob = model.predict_proba(X_pool)
        y_pred = model.predict(X_pool)
        
        # Select samples with highest uncertainty (entropy)
        uncertainty = -np.sum(y_prob * np.log(y_prob + 1e-10), axis=1)
        n_select = min(10, len(X_pool))
        idx = np.argsort(uncertainty)[-n_select:]
        
        # Add to training set
        X_train = np.vstack([X_train, X_pool[idx]])
        y_train = np.concatenate([y_train, y_pool[idx]])
        
        # Remove from pool
        mask = np.ones(len(X_pool), dtype=bool)
        mask[idx] = False
        X_pool = X_pool[mask]
        y_pool = y_pool[mask]
        
        # Retrain and evaluate
        model.fit(X_train, y_train)
        y_test = np.concatenate([y_train, y_pool])
        y_prob_test = model.predict_proba(np.concatenate([X_train, X_pool]))
        ece = compute_ece(y_test, y_prob_test)
        ece_history.append(ece)
    
    return np.mean(ece_history)

def calibration_aware_active_learning(X, y, pool_size, iterations):
    """Calibration-aware active learning that considers both uncertainty and calibration"""
    # Initial split
    X_train, X_pool, y_train, y_pool = train_test_split(
        X, y, train_size=pool_size // 2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    ece_history = []
    
    for _ in range(iterations):
        # Get predictions on pool
        y_prob = model.predict_proba(X_pool)
        y_pred = model.predict(X_pool)
        
        # Calculate uncertainty (entropy)
        uncertainty = -np.sum(y_prob * np.log(y_prob + 1e-10), axis=1)
        
        # Calculate calibration error per sample
        calibration_error = np.abs(y_prob.max(axis=1) - y_pred.astype(float))
        
        # Combined score: prioritize poorly calibrated samples
        combined_score = uncertainty * (1 + calibration_error)
        
        # Select samples with highest combined score
        n_select = min(10, len(X_pool))
        idx = np.argsort(combined_score)[-n_select:]
        
        # Add to training set
        X_train = np.vstack([X_train, X_pool[idx]])
        y_train = np.concatenate([y_train, y_pool[idx]])
        
        # Remove from pool
        mask = np.ones(len(X_pool), dtype=bool)
        mask[idx] = False
        X_pool = X_pool[mask]
        y_pool = y_pool[mask]
        
        # Retrain and evaluate
        model.fit(X_train, y_train)
        y_test = np.concatenate([y_train, y_pool])
        y_prob_test = model.predict_proba(np.concatenate([X_train, X_pool]))
        ece = compute_ece(y_test, y_prob_test)
        ece_history.append(ece)
    
    return np.mean(ece_history)

if __name__ == "__main__":
    # Load data
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Define parameters
    pool_size = 1000
    iterations = 5
    
    # Run standard active learning
    ece_standard = standard_active_learning(X, y, pool_size, iterations)
    
    # Run calibration-aware active learning
    ece_calibrated = calibration_aware_active_learning(X, y, pool_size, iterations)
    
    # Determine the signal
    # GAP_SIGNAL is true if the calibration-aware method performs better (lower ECE)
    is_better = ece_calibrated < ece_standard
    
    # The number is the difference in performance.
    # A positive difference means the calibrated method was better.
    performance_difference = ece_standard - ece_calibrated
    
    # Print the final line
    print(f"GAP_SIGNAL {str(is_better).lower()} {performance_difference:.4f}")
