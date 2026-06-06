import os
import json
import numpy as np
from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, classification_report
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import psutil
import time

# Create outputs directory
os.makedirs('outputs', exist_ok=True)

class EvidenceTraceLedger:
    """Explicit state, reviewer, and metric objects for falsifiable progress"""
    def __init__(self):
        self.states = []
        self.reviewers = []
        self.metrics = []
    
    def add_state(self, state_name, data):
        self.states.append({'name': state_name, 'data': data})
    
    def add_reviewer(self, reviewer_name, judgment):
        self.reviewers.append({'name': reviewer_name, 'judgment': judgment})
    
    def add_metric(self, metric_name, value):
        self.metrics.append({'name': metric_name, 'value': float(value)})

class CalibrationAwareActiveLearner:
    """Proposed method: calibration-aware active learning with explicit evidence tracking"""
    def __init__(self, n_queries=20, calibration_threshold=0.7):
        self.n_queries = n_queries
        self.calibration_threshold = calibration_threshold
        self.ledger = EvidenceTraceLedger()
        self.model = None
        self.X_pool = None
        self.y_pool = None
        self.X_labeled = None
        self.y_labeled = None
        
    def initialize_pool(self, X, y, initial_size=100):
        """Initialize labeled pool and unlabeled pool"""
        indices = np.random.RandomState(42).permutation(len(X))
        self.X_labeled = X[indices[:initial_size]]
        self.y_labeled = y[indices[:initial_size]]
        self.X_pool = X[indices[initial_size:]]
        self.y_pool = y[indices[initial_size:]]
        
        self.ledger.add_state("initial_setup", {
            'labeled_size': initial_size,
            'pool_size': len(X) - initial_size
        })
        self.ledger.add_reviewer("system", "Pool initialized with random sampling")
    
    def query_by_uncertainty_and_calibration(self):
        """Active learning query combining uncertainty and calibration awareness"""
        if len(self.X_pool) == 0:
            return
            
        # Train current model with increased iterations and better solver
        self.model = LogisticRegression(max_iter=2000, solver='lbfgs', random_state=42)
        self.model.fit(self.X_labeled, self.y_labeled)
        
        # Get predictions and probabilities
        probs = self.model.predict_proba(self.X_pool)
        preds = self.model.predict(self.X_pool)
        
        # Calculate uncertainty (entropy)
        uncertainties = -np.sum(probs * np.log(probs + 1e-8), axis=1)
        
        # Calculate calibration score (Brier score)
        brier_scores = brier_score_loss(self.y_pool, probs, pos_label=1)
        
        # Combine uncertainty and calibration (higher uncertainty + higher Brier = more informative)
        scores = uncertainties + brier_scores
        
        # Select samples with highest combined scores
        query_indices = np.argsort(scores)[-self.n_queries:]
        
        # Move selected samples from pool to labeled set
        self.X_labeled = np.vstack([self.X_labeled, self.X_pool[query_indices]])
        self.y_labeled = np.concatenate([self.y_labeled, self.y_pool[query_indices]])
        
        # Remove queried samples from pool
        mask = np.ones(len(self.X_pool), dtype=bool)
        mask[query_indices] = False
        self.X_pool = self.X_pool[mask]
        self.y_pool = self.y_pool[mask]
        
        # Record in ledger
        self.ledger.add_state("active_learning_step", {
            'queried_samples': self.n_queries,
            'remaining_pool': len(self.X_pool)
        })
        self.ledger.add_reviewer("active_learner", "Selected by uncertainty+calibration")
        self.ledger.add_metric("pool_size", len(self.X_pool))
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        self.model = LogisticRegression(max_iter=2000, solver='lbfgs', random_state=42)
        self.model.fit(self.X_labeled, self.y_labeled)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_prob[:, 1])
        
        # Calibration curve
        prob_true, prob_pred = calibration_curve(y_test, y_prob[:, 1], n_bins=10)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Store metrics
        self.ledger.add_metric("accuracy", accuracy)
        self.ledger.add_metric("brier_score", brier)
        self.ledger.add_metric("n_classes", len(np.unique(y_test)))
        
        return {
            'accuracy': accuracy,
            'brier_score': brier,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'calibration_curve': {
                'prob_true': prob_true.tolist(),
                'prob_pred': prob_pred.tolist()
            }
        }

def main():
    # Load dataset
    print("Loading dataset...")
    X, y = load_breast_cancer(return_X_y=True)
    
    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize active learner
    learner = CalibrationAwareActiveLearner(n_queries=20, calibration_threshold=0.7)
    
    # Initialize pool
    learner.initialize_pool(X_train, y_train, initial_size=100)
    
    # Active learning loop
    print("Starting active learning...")
    for iteration in range(10):
        print(f"Iteration {iteration+1}")
        learner.query_by_uncertainty_and_calibration()
        
        # Evaluate on test set
        results = learner.evaluate(X_test, y_test)
        
        # Record system metrics
        process = psutil.Process()
        memory_info = process.memory_info()
        learner.ledger.add_metric("memory_usage_mb", memory_info.rss / 1024 / 1024)
        learner.ledger.add_metric("iteration", iteration + 1)
        
        print(f"Accuracy: {results['accuracy']:.4f}, Brier Score: {results['brier_score']:.4f}")
    
    # Prepare final results
    final_results = {
        'experiment_type': 'Calibration-Aware Active Learning',
        'dataset': 'breast_cancer',
        'initial_labeled_size': 100,
        'total_queries': 200,
        'final_labeled_size': len(learner.X_labeled),
        'test_accuracy': results['accuracy'],
        'test_brier_score': results['brier_score'],
        'ledger': {
            'states': learner.ledger.states,
            'reviewers': learner.ledger.reviewers,
            'metrics': learner.ledger.metrics
        },
        'final_metrics': results
    }
    
    # Save results
    with open('outputs/metrics.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("Results saved to outputs/metrics.json")

if __name__ == "__main__":
    main()
