import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=50, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def simulate_attention_mechanism(X, attention_type='standard'):
    """Simulate attention mechanism with different computational patterns"""
    n_samples, n_features = X.shape
    
    if attention_type == 'standard':
        # Standard attention: O(n_features^2) complexity
        # Simulate full attention matrix
        attention_weights = np.random.rand(n_features, n_features)
        attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
        
        # Apply attention
        attended_X = X @ attention_weights
        complexity = n_features ** 2 + n_features * n_features  # Matrix mult + element-wise
        
    else:  # efficient
        # Efficient attention: O(n_features) complexity
        # Simulate sparse attention pattern
        attention_weights = np.zeros((n_features, n_features))
        for i in range(n_features):
            # Each feature only attends to 3 neighbors
            neighbors = np.random.choice(n_features, size=3, replace=False)
            attention_weights[i, neighbors] = 1/3
        
        # Apply attention
        attended_X = X @ attention_weights
        complexity = n_features * 3 + n_samples * n_features * 3  # Sparse mult
        
    return attended_X, complexity

# Test standard attention
X_train_std, complexity_std = simulate_attention_mechanism(X_train, 'standard')
X_test_std, _ = simulate_attention_mechanism(X_test, 'standard')

# Train and evaluate standard model
model_std = LogisticRegression(max_iter=100, random_state=42)
model_std.fit(X_train_std, y_train)
y_pred_std = model_std.predict(X_test_std)
accuracy_std = accuracy_score(y_test, y_pred_std)

# Test efficient attention
X_train_eff, complexity_eff = simulate_attention_mechanism(X_train, 'efficient')
X_test_eff, _ = simulate_attention_mechanism(X_test, 'efficient')

# Train and evaluate efficient model
model_eff = LogisticRegression(max_iter=100, random_state=42)
model_eff.fit(X_train_eff, y_train)
y_pred_eff = model_eff.predict(X_test_eff)
accuracy_eff = accuracy_score(y_test, y_pred_eff)

# Calculate complexity reduction
complexity_reduction = (complexity_std - complexity_eff) / complexity_std * 100

# Check if gap signal exists
gap_signal = (complexity_reduction > 10) and (accuracy_eff > 0.95)

# Print result
print(f"GAP_SIGNAL {gap_signal} {complexity_reduction:.2f}")
