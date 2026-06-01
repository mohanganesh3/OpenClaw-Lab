#!/usr/bin/env python3
import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime

# Set random seed
random.seed(1729)
np.random.seed(1729)

class FailureMemory:
    def __init__(self, max_size=100):
        self.memory = []
        self.max_size = max_size
    
    def add_failure(self, state, reviewer, metric_value, decision):
        failure_entry = {
            'state': state,
            'reviewer': reviewer,
            'metric_value': metric_value,
            'decision': decision,
            'timestamp': datetime.now().isoformat()
        }
        self.memory.append(failure_entry)
        if len(self.memory) > self.max_size:
            self.memory = self.memory[-self.max_size:]
    
    def get_similar_failures(self, current_state, threshold=0.5):
        similar = []
        for failure in self.memory:
            similarity = self._calculate_similarity(current_state, failure['state'])
            if similarity >= threshold:
                similar.append(failure)
        return similar
    
    def _calculate_similarity(self, state1, state2):
        # Simple similarity based on feature overlap
        if len(state1) != len(state2):
            return 0.0
        matches = sum(1 for a, b in zip(state1, state2) if a == b)
        return matches / len(state1)

class ResearchAgent:
    def __init__(self, failure_memory):
        self.failure_memory = failure_memory
        self.model = None
        self.selection_history = []
    
    def train(self, X_train, y_train):
        self.model = RandomForestClassifier(n_estimators=10, random_state=1729)
        self.model.fit(X_train, y_train)
    
    def evaluate_decision(self, X_test, y_test, state):
        if self.model is None:
            return 0.0, "No model trained"
        
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        # Check failure memory for similar cases
        similar_failures = self.failure_memory.get_similar_failures(state)
        if similar_failures:
            avg_failure_metric = np.mean([f['metric_value'] for f in similar_failures])
            reviewer = "FailureMemory"
        else:
            avg_failure_metric = 0.0
            reviewer = "ModelOnly"
        
        return accuracy, reviewer
    
    def make_decision(self, X, state):
        if self.model is None:
            return random.random()
        
        # Get model prediction
        prediction = self.model.predict_proba(X.reshape(1, -1))
        confidence = np.max(prediction)
        
        # Check failure memory
        similar_failures = self.failure_memory.get_similar_failures(state)
        if similar_failures:
            # Adjust based on failure memory
            failure_adjustment = np.mean([f['metric_value'] for f in similar_failures])
            adjusted_confidence = confidence * (1 - failure_adjustment * 0.5)
        else:
            adjusted_confidence = confidence
        
        self.selection_history.append(adjusted_confidence)
        return adjusted_confidence

def generate_synthetic_data(n_samples=1000, n_features=20):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                               n_informative=15, n_redundant=5, 
                               random_state=1729)
    return X, y

def guided_selection_utility(agent, X_test, y_test, states):
    utilities = []
    for i, state in enumerate(states):
        utility, reviewer = agent.evaluate_decision(X_test[i:i+1], y_test[i:i+1], state)
        utilities.append(utility)
        
        # Add to failure memory
        agent.failure_memory.add_failure(
            state=state,
            reviewer=reviewer,
            metric_value=utility,
            decision=utility > 0.5
        )
    
    return np.mean(utilities)

def random_baseline_utility(X_test, y_test, n_trials=100):
    utilities = []
    for _ in range(n_trials):
        random_predictions = np.random.random(len(X_test))
        accuracy = accuracy_score(y_test, (random_predictions > 0.5).astype(int))
        utilities.append(accuracy)
    
    return np.mean(utilities)

def main():
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    # Generate synthetic data
    X, y = generate_synthetic_data(n_samples=500, n_features=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1729)
    
    # Initialize failure memory and agent
    failure_memory = FailureMemory(max_size=50)
    agent = ResearchAgent(failure_memory)
    
    # Train the agent
    agent.train(X_train, y_train)
    
    # Generate synthetic states for decisions
    states = [tuple(np.random.randint(0, 2, size=10)) for _ in range(50)]
    
    # Calculate guided selection utility
    guided_utility = guided_selection_utility(agent, X_test, y_test, states)
    
    # Calculate random baseline
    random_utility = random_baseline_utility(X_test, y_test, n_trials=50)
    
    # Calculate metric
    metric_value = guided_utility - random_utility
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    # Plot selection history
    plt.subplot(2, 2, 1)
    plt.plot(agent.selection_history, alpha=0.7)
    plt.title('Agent Selection Confidence Over Time')
    plt.xlabel('Decision Number')
    plt.ylabel('Confidence')
    
    # Plot utility comparison
    plt.subplot(2, 2, 2)
    methods = ['Guided Selection', 'Random Baseline']
    utilities = [guided_utility, random_utility]
    plt.bar(methods, utilities, color=['blue', 'red'], alpha=0.7)
    plt.title('Utility Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    # Plot failure memory size
    plt.subplot(2, 2, 3)
    memory_sizes = [len(failure_memory.memory)]
    plt.bar(['Failure Memory'], memory_sizes, color='green', alpha=0.7)
    plt.title('Failure Memory Size')
    plt.ylabel('Entries')
    
    # Plot metric distribution
    plt.subplot(2, 2, 4)
    metric_values = [metric_value]
    plt.bar(['Guided - Random'], metric_values, color='purple', alpha=0.7)
    plt.title('Metric Value')
    plt.ylabel('Utility Difference')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('outputs/experiment_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Prepare metrics
    experiment_id = f"f2d289_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    metrics = {
        'experiment_id': experiment_id,
        'success': metric_value > 0,
        'measured_metric_values': {
            'guided_selection_utility': guided_utility,
            'random_baseline_utility': random_utility,
            'guided_selection_utility_minus_random_baseline': metric_value,
            'failure_memory_size': len(failure_memory.memory),
            'agent_decisions_made': len(agent.selection_history)
        }
    }
    
    # Save metrics
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Experiment completed. Metric value: {metric_value:.4f}")
    print(f"Success: {metrics['success']}")
    print(f"Results saved to outputs/experiment_results.png")
    print(f"Metrics saved to metrics.json")

if __name__ == "__main__":
    main()
