import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score

def general_active_learning(X, y, pool_size=50, iterations=5):
    X_train, X_pool, y_train, y_pool = train_test_split(
        X, y, train_size=len(X) - pool_size, random_state=42
    )
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    for _ in range(iterations):
        probs = model.predict_proba(X_pool)
        uncertainty = np.max(probs, axis=1)
        n_select = min(10, len(X_pool))
        indices = np.argsort(uncertainty)[-n_select:]
        
        X_train = np.vstack([X_train, X_pool[indices]])
        y_train = np.hstack([y_train, y_pool[indices]])
        
        mask = np.ones(len(X_pool), dtype=bool)
        mask[indices] = False
        X_pool = X_pool[mask]
        y_pool = y_pool[mask]
        
        model.fit(X_train, y_train)
    
    return model

def calibration_aware_active_learning(X, y, pool_size=50, iterations=5):
    X_train, X_pool, y_train, y_pool = train_test_split(
        X, y, train_size=len(X) - pool_size, random_state=42
    )
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    for _ in range(iterations):
        calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=2)
        calibrated.fit(X_train, y_train)
        probs = calibrated.predict_proba(X_pool)
        
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        n_select = min(10, len(X_pool))
        indices = np.argsort(entropy)[-n_select:]
        
        X_train = np.vstack([X_train, X_pool[indices]])
        y_train = np.hstack([y_train, y_pool[indices]])
        
        mask = np.ones(len(X_pool), dtype=bool)
        mask[indices] = False
        X_pool = X_pool[mask]
        y_pool = y_pool[mask]
        
        model.fit(X_train, y_train)
    
    return model

X, y = load_breast_cancer(return_X_y=True)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

general_model = general_active_learning(X_train_full, y_train_full)
calibrated_model = calibration_aware_active_learning(X_train_full, y_train_full)

general_acc = accuracy_score(y_test, general_model.predict(X_test))
calibrated_acc = accuracy_score(y_test, calibrated_model.predict(X_test))
general_loss = log_loss(y_test, general_model.predict_proba(X_test))
calibrated_loss = log_loss(y_test, calibrated_model.predict_proba(X_test))

acc_diff = calibrated_acc - general_acc
loss_diff = general_loss - calibrated_loss

effect_size = (acc_diff + loss_diff/2) * 100

print(f"GAP_SIGNAL {'true' if effect_size > 0 else 'false'} {effect_size:.2f}")
