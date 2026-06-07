import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

np.random.seed(42)

# Generate synthetic embodied agent task data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, 
                          n_informative=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standard attention mechanism
def standard_attention(X, weights):
    attention_scores = np.dot(X, weights)
    attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=1, keepdims=True)
    return np.dot(X, attention_weights.T)

# Spatially optimized attention mechanism
def spatial_attention(X, weights, spatial_mask=None):
    if spatial_mask is None:
        spatial_mask = np.random.rand(X.shape[1]) > 0.3  # Only attend to 70% of features
    
    X_masked = X[:, spatial_mask]
    attention_scores = np.dot(X_masked, weights[spatial_mask])
    attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=1, keepdims=True)
    return np.dot(X_masked, attention_weights.T)

# Initialize weights
weights = np.random.randn(X.shape[1])

# Test standard attention
start_time = time.time()
X_train_std = standard_attention(X_train, weights)
X_test_std = standard_attention(X_test, weights)
std_time = time.time() - start_time

# Train simple classifier for standard attention
from sklearn.linear_model import LogisticRegression
clf_std = LogisticRegression(max_iter=100).fit(X_train_std, y_train)
std_accuracy = accuracy_score(y_test, clf_std.predict(X_test_std))

# Test spatially optimized attention
start_time = time.time()
X_train_spatial = spatial_attention(X_train, weights)
X_test_spatial = spatial_attention(X_test, weights)
spatial_time = time.time() - start_time

# Train simple classifier for spatial attention
clf_spatial = LogisticRegression(max_iter=100).fit(X_train_spatial, y_train)
spatial_accuracy = accuracy_score(y_test, clf_spatial.predict(X_test_spatial))

# Calculate energy reduction (simulated as time reduction)
energy_reduction = (std_time - spatial_time) / std_time * 100
task_maintained = abs(std_accuracy - spatial_accuracy) < 0.05

print(f"GAP_SIGNAL {task_maintained and energy_reduction > 15} {energy_reduction:.2f}")
