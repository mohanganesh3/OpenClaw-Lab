import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import scipy.stats as stats

# Set seed for reproducibility
np.random.seed(42)

# Load dataset
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Baseline agent without traceability
class BaselineAgent:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    def predict(self, X):
        probs = self.model.predict_proba(X)
        preds = np.argmax(probs, axis=1)
        return preds, probs
    
    def train(self, X, y):
        self.model.fit(X, y)

# Traceable agent with reasoning tracking
class TraceableAgent:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.decision_log = []
    
    def predict(self, X):
        # Simulate traceable reasoning process
        reasoning = f"Analyzing {X.shape[0]} samples with {X.shape[1]} features"
        
        probs = self.model.predict_proba(X)
        preds = np.argmax(probs, axis=1)
        
        # Log traceable decision
        self.decision_log.append({
            'reasoning': reasoning,
            'confidence': np.max(probs, axis=1),
            'features_importance': np.mean(np.abs(X), axis=0)
        })
        
        return preds, probs
    
    def train(self, X, y):
        self.model.fit(X, y)

# Calculate Expected Calibration Error (ECE)
def calculate_ece(y_true, y_pred, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            # Calculate accuracy and average confidence in bin
            accuracy_in_bin = np.mean(y_true[in_bin] == y_pred[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

# Train and evaluate baseline agent
baseline = BaselineAgent()
baseline.train(X_train, y_train)
baseline_pred, baseline_prob = baseline.predict(X_test)
baseline_ece = calculate_ece(y_test, baseline_pred, np.max(baseline_prob, axis=1))

# Train and evaluate traceable agent
traceable = TraceableAgent()
traceable.train(X_train, y_train)
traceable_pred, traceable_prob = traceable.predict(X_test)
traceable_ece = calculate_ece(y_test, traceable_pred, np.max(traceable_prob, axis=1))

# Calculate reduction percentage
reduction = (baseline_ece - traceable_ece) / baseline_ece * 100

# Determine if gap signal exists (20% reduction threshold)
gap_signal = reduction >= 20

# Print final result
print(f"GAP_SIGNAL {str(gap_signal).lower()} {reduction:.2f}")
