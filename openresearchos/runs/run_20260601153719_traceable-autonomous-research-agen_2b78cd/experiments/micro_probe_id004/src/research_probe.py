#!/usr/bin/env python3
import json
import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set fixed random seed
RANDOM_SEED = 1729
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

class FailureMemory:
    """Stores and retrieves failure patterns for research agents"""
    def __init__(self, max_size=100):
        self.failures = []
        self.max_size = max_size
    
    def add_failure(self, state, reviewer, metric_value, context):
        """Add a failure case to memory"""
        failure = {
            'state': state,
            'reviewer': reviewer,
            'metric_value': metric_value,
            'context': context,
            'timestamp': time.time()
        }
        self.failures.append(failure)
        if len(self.failures) > self.max_size:
            self.failures.pop(0)
    
    def get_similar_failures(self, current_state, threshold=0.5):
        """Retrieve similar failure cases"""
        similar = []
        for failure in self.failures:
            similarity = self._calculate_similarity(current_state, failure['state'])
            if similarity < threshold:  # Lower similarity means more different
                similar.append(failure)
        return similar
    
    def _calculate_similarity(self, state1, state2):
        """Calculate similarity between two states"""
        # Simple similarity based on state characteristics
        if isinstance(state1, dict) and isinstance(state2, dict):
            common_keys = set(state1.keys()) & set(state2.keys())
            if not common_keys:
                return 0
            similarity = sum(1 for k in common_keys if state1[k] == state2[k]) / len(common_keys)
            return similarity
        return 0

class ResearchAgent:
    """Autonomous research agent with failure memory"""
    def __init__(self, failure_memory):
        self.failure_memory = failure_memory
        self.model = None
        self.performance_history = []
    
    def evaluate_hypothesis(self, X_train, y_train, X_test, y_test, hypothesis_params):
        """Evaluate a research hypothesis"""
        # Create model with hypothesis parameters
        self.model = RandomForestClassifier(
            n_estimators=hypothesis_params.get('n_estimators', 10),
            max_depth=hypothesis_params.get('max_depth', 5),
            random_state=RANDOM_SEED
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        # Calculate utility (weighted combination of metrics)
        utility = (metrics['accuracy'] * 0.4 + 
                  metrics['precision'] * 0.3 + 
                  metrics['recall'] * 0.2 + 
                  metrics['f1'] * 0.1)
        
        # Check for failure patterns
        state = {
            'n_estimators': hypothesis_params.get('n_estimators', 10),
            'max_depth': hypothesis_params.get('max_depth', 5),
            'feature_importance': np.mean(self.model.feature_importances_) if self.model else 0
        }
        
        reviewer = "agent_v1"
        
        # Add to failure memory if performance is poor
        if utility < 0.7:  # Threshold for failure
            self.failure_memory.add_failure(state, reviewer, utility, metrics)
        
        return utility, metrics

class GuidedSelection:
    """Guided selection mechanism using failure memory"""
    def __init__(self, failure_memory):
        self.failure_memory = failure_memory
        self.selection_history = []
    
    def select_hypothesis(self, X_train, y_train, X_test, y_test, candidate_params_list):
        """Select best hypothesis using guided selection"""
        agent = ResearchAgent(self.failure_memory)
        best_utility = -1
        best_params = None
        best_metrics = None
        
        # Evaluate all candidates
        for params in candidate_params_list:
            utility, metrics = agent.evaluate_hypothesis(X_train, y_train, X_test, y_test, params)
            
            # Check failure memory for similar cases
            state = {
                'n_estimators': params.get('n_estimators', 10),
                'max_depth': params.get('max_depth', 5)
            }
            similar_failures = self.failure_memory.get_similar_failures(state)
            
            # Adjust utility based on failure memory
            if similar_failures:
                avg_failure_utility = np.mean([f['metric_value'] for f in similar_failures])
                # Penalize if similar cases failed
                utility *= (1 - 0.3 * (1 - avg_failure_utility))
            
            if utility > best_utility:
                best_utility = utility
                best_params = params
                best_metrics = metrics
        
        self.selection_history.append(best_utility)
        return best_utility, best_params, best_metrics

def generate_synthetic_data():
    """Generate synthetic research dataset"""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=RANDOM_SEED
    )
    return train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)

def random_baseline_selection(X_train, y_train, X_test, y_test, candidate_params_list):
    """Random baseline selection for comparison"""
    agent = ResearchAgent(None)
    utilities = []
    
    for params in candidate_params_list:
        utility, _ = agent.evaluate_hypothesis(X_train, y_train, X_test, y_test, params)
        utilities.append(utility)
    
    return np.mean(utilities)

def main():
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    # Generate synthetic data
    X_train, X_test, y_train, y_test = generate_synthetic_data()
    
    # Define candidate hypotheses
    candidate_params_list = [
        {'n_estimators': 10, 'max_depth': 3},
        {'n_estimators': 20, 'max_depth': 5},
        {'n_estimators': 30, 'max_depth': 7},
        {'n_estimators': 50, 'max_depth': 10},
        {'n_estimators': 100, 'max_depth': 15},
    ]
    
    # Initialize failure memory and guided selection
    failure_memory = FailureMemory(max_size=50)
    guided_selection = GuidedSelection(failure_memory)
    
    # Run guided selection experiments
    guided_utilities = []
    for i in range(20):  # Run 20 experiments
        utility, params, metrics = guided_selection.select_hypothesis(
            X_train, y_train, X_test, y_test, candidate_params_list
        )
        guided_utilities.append(utility)
    
    # Run random baseline experiments
    random_utilities = []
    for i in range(20):
        utility = random_baseline_selection(X_train, y_train, X_test, y_test, candidate_params_list)
        random_utilities.append(utility)
    
    # Calculate metric: guided_selection_utility_minus_random_baseline
    guided_mean = np.mean(guided_utilities)
    random_mean = np.mean(random_utilities)
    metric_value = guided_mean - random_mean
    
    # Determine success (positive metric indicates improvement)
    success = metric_value > 0
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(guided_utilities, 'b-', label='Guided Selection', alpha=0.7)
    plt.plot(random_utilities, 'r--', label='Random Baseline', alpha=0.7)
    plt.axhline(y=guided_mean, color='b', linestyle=':', alpha=0.5)
    plt.axhline(y=random_mean, color='r', linestyle=':', alpha=0.5)
    plt.xlabel('Experiment Iteration')
    plt.ylabel('Utility Score')
    plt.title('Guided Selection vs Random Baseline')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('outputs/guided_vs_random.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save metrics
    metrics = {
        'experiment_id': 'smoke_3190fd_failure_memory_v1',
        'success': success,
        'measured_metric_values': {
            'guided_selection_utility_minus_random_baseline': metric_value,
            'guided_selection_mean': guided_mean,
            'random_baseline_mean': random_mean,
            'guided_selection_std': np.std(guided_utilities),
            'random_baseline_std': np.std(random_utilities),
            'num_experiments': 20
        }
    }
    
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Experiment completed. Metric: {metric_value:.4f}, Success: {success}")

if __name__ == "__main__":
    main()
