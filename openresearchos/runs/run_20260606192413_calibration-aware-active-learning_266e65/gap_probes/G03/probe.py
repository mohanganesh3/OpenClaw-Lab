import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.calibration import calibration_curve
import scipy.stats as stats

np.random.seed(42)

# Generate synthetic medical-like dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                          n_redundant=5, n_classes=2, random_state=42)
X_train, X_pool, y_train, y_pool = train_test_split(X, y, test_size=0.5, 
                                                    stratify=y, random_state=42)

def active_learning_with_refinement(X_pool, y_pool, X_train, y_train, 
                                   target_accuracy=0.9, max_iter=50):
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    labeled_pool = X_pool.copy()
    labeled_labels = y_pool.copy()
    
    for iteration in range(max_iter):
        # Get predictions and uncertainties
        y_pred = model.predict(labeled_pool)
        y_proba = model.predict_proba(labeled_pool)
        uncertainties = np.max(y_proba, axis=1)
        
        # Self-refinement: focus on uncertain samples
        n_select = min(50, len(labeled_pool))
        uncertain_indices = np.argsort(uncertainties)[-n_select:]
        
        # Add uncertain samples to training set
        new_X = labeled_pool[uncertain_indices]
        new_y = labeled_labels[uncertain_indices]
        
        X_train = np.vstack([X_train, new_X])
        y_train = np.concatenate([y_train, new_y])
        
        # Remove selected samples from pool
        mask = np.ones(len(labeled_pool), dtype=bool)
        mask[uncertain_indices] = False
        labeled_pool = labeled_pool[mask]
        labeled_labels = labeled_labels[mask]
        
        # Retrain model
        model.fit(X_train, y_train)
        
        # Check accuracy on held-out test set (using a fixed test split)
        X_test, y_test = train_test_split(X, y, test_size=0.2, 
                                         stratify=y, random_state=42)
        acc = accuracy_score(y_test, model.predict(X_test))
        
        if acc >= target_accuracy:
            return len(X_train), acc
    
    return len(X_train), accuracy_score(y_test, model.predict(X_test))

def standard_active_learning(X_pool, y_pool, X_train, y_train, 
                           target_accuracy=0.9, max_iter=50):
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    labeled_pool = X_pool.copy()
    labeled_labels = y_pool.copy()
    
    for iteration in range(max_iter):
        # Random sampling (no refinement)
        n_select = min(50, len(labeled_pool))
        indices = np.random.choice(len(labeled_pool), n_select, replace=False)
        
        # Add random samples to training set
        new_X = labeled_pool[indices]
        new_y = labeled_labels[indices]
        
        X_train = np.vstack([X_train, new_X])
        y_train = np.concatenate([y_train, new_y])
        
        # Remove selected samples from pool
        mask = np.ones(len(labeled_pool), dtype=bool)
        mask[indices] = False
        labeled_pool = labeled_pool[mask]
        labeled_labels = labeled_labels[mask]
        
        # Retrain model
        model.fit(X_train, y_train)
        
        # Check accuracy on held-out test set (using a fixed test split)
        X_test, y_test = train_test_split(X, y, test_size=0.2, 
                                         stratify=y, random_state=42)
        acc = accuracy_score(y_test, model.predict(X_test))
        
        if acc >= target_accuracy:
            return len(X_train), acc
    
    return len(X_train), accuracy_score(y_test, model.predict(X_test))

# Main execution
if __name__ == "__main__":
    # Run both methods
    samples_with_refinement, acc_with_refinement = active_learning_with_refinement(
        X_pool, y_pool, X_train, y_train)
    
    samples_standard, acc_standard = standard_active_learning(
        X_pool, y_pool, X_train, y_train)
    
    # Determine if refinement improved performance
    improved = acc_with_refinement > acc_standard
    
    # Print the required output
    print(f"GAP_SIGNAL {str(improved).lower()} {samples_with_refinement}")
