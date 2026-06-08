import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set seed
np.random.seed(42)

# Load dataset
data = load_digits()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train on clean labels
clf_clean = RandomForestClassifier(random_state=42)
clf_clean.fit(X_train, y_train)
y_pred_clean = clf_clean.predict(X_test)
acc_clean = accuracy_score(y_test, y_pred_clean)

# Introduce 20% label noise
n_samples = len(y_train)
noise_indices = np.random.choice(n_samples, size=int(0.2 * n_samples), replace=False)
y_train_noisy = y_train.copy()
y_train_noisy[noise_indices] = np.random.randint(0, 10, size=len(noise_indices))

# Train on noisy labels
clf_noisy = RandomForestClassifier(random_state=42)
clf_noisy.fit(X_train, y_train_noisy)
y_pred_noisy = clf_noisy.predict(X_test)
acc_noisy = accuracy_score(y_test, y_pred_noisy)

# Calculate effect size
effect_size = acc_clean - acc_noisy

# Check if effect size is at least 5%
gap_signal = effect_size >= 0.05

# Print result
print(f"GAP_SIGNAL {gap_signal} {effect_size:.4f}")
