import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

np.random.seed(42)

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

original_model = RandomForestClassifier(n_estimators=100, random_state=42)
original_model.fit(X_train, y_train)
original_pred = original_model.predict(X_test)
original_acc = accuracy_score(y_test, original_pred)

with open('model.pkl', 'wb') as f:
    pickle.dump(original_model, f)

with open('model.pkl', 'rb') as f:
    reproduced_model = pickle.load(f)
reproduced_pred = reproduced_model.predict(X_test)
reproduced_acc = accuracy_score(y_test, reproduced_pred)

delta = abs(original_acc - reproduced_acc) * 100

os.remove('model.pkl')

gap_signal = delta > 5

print(f"GAP_SIGNAL {gap_signal} {delta:.2f}")
