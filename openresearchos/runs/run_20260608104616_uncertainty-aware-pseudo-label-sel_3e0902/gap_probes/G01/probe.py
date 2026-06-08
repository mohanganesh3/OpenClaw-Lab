import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Set seed for reproducibility
np.random.seed(42)

# Generate synthetic data for natural images (higher quality features)
X_natural, y_natural = make_classification(
    n_samples=1000,
    n_features=50,
    n_informative=40,
    n_redundant=10,
    random_state=42
)

# Generate synthetic data for medical images (lower quality features, more noise)
X_medical, y_medical = make_classification(
    n_samples=1000,
    n_features=50,
    n_informative=20,
    n_redundant=30,
    random_state=42
)

# Split data
X_train_natural, X_test_natural, y_train_natural, y_test_natural = train_test_split(
    X_natural, y_natural, test_size=0.3, random_state=42
)
X_train_medical, X_test_medical, y_train_medical, y_test_medical = train_test_split(
    X_medical, y_medical, test_size=0.3, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_natural = scaler.fit_transform(X_train_natural)
X_test_natural = scaler.transform(X_test_natural)
X_train_medical = scaler.transform(X_train_medical)
X_test_medical = scaler.transform(X_test_medical)

# Train model on natural images (simulating SAM training on natural images)
model_natural = RandomForestClassifier(n_estimators=50, random_state=42)
model_natural.fit(X_train_natural, y_train_natural)

# Get predictions on medical test set
y_pred_medical = model_natural.predict(X_test_medical)

# Calculate accuracy on medical images
accuracy_medical = accuracy_score(y_test_medical, y_pred_medical)

# Train model on medical images (simulating SAM training on medical images)
model_medical = RandomForestClassifier(n_estimators=50, random_state=42)
model_medical.fit(X_train_medical, y_train_medical)

# Get predictions on medical test set
y_pred_medical_trained = model_medical.predict(X_test_medical)

# Calculate accuracy on medical images when trained on medical data
accuracy_medical_trained = accuracy_score(y_test_medical, y_pred_medical_trained)

# Calculate accuracy on natural images
y_pred_natural = model_natural.predict(X_test_natural)
accuracy_natural = accuracy_score(y_test_natural, y_pred_natural)

# Calculate the gap
gap = accuracy_natural - accuracy_medical

# Check if gap >= 10% (0.1)
gap_exists = gap >= 0.1

print(f"GAP_SIGNAL {str(gap_exists).lower()} {gap:.3f}")
