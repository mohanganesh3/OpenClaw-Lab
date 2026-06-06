import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import clone
import time

# Set seed
np.random.seed(42)

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split into train, validation, and test
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.3, random_state=42, stratify=y_train_full
)

def traditional_active_learning(X_pool, y_pool, X_train, y_train, X_val, y_val, target_acc=0.90):
    """Traditional active learning with uncertainty sampling"""
    pool_indices = list(range(len(X_pool)))
    selected_indices = []
    
    while True:
        # Train model on current training set
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Get predictions for pool
        if len(pool_indices) > 0:
            pool_X = X_pool[pool_indices]
            probs = model.predict_proba(pool_X)
            uncertainties = 1 - np.max(probs, axis=1)
            
            # Select most uncertain sample
            idx = np.argmax(uncertainties)
            selected_indices.append(pool_indices[idx])
            
            # Add to training set
            X_train = np.vstack([X_train, X_pool[pool_indices[idx]]])
            y_train = np.append(y_train, y_pool[pool_indices[idx]])
            
            # Remove from pool
            pool_indices.pop(idx)
        
        # Evaluate on validation set
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        
        if val_acc >= target_acc:
            break
        
        if len(pool_indices) == 0:
            break
    
    return len(selected_indices), val_acc

def llm_enhanced_active_learning(X_pool, y_pool, X_train, y_train, X_val, y_val, target_acc=0.90):
    """Simulated LLM-agent enhanced active learning with domain knowledge"""
    pool_indices = list(range(len(X_pool)))
    selected_indices = []
    
    # Simulate domain knowledge: feature importance weights
    feature_importance = np.random.dirichlet(np.ones(X_pool.shape[1]), 1)[0]
    
    while True:
        # Train model on current training set
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Get predictions for pool
        if len(pool_indices) > 0:
            pool_X = X_pool[pool_indices]
            probs = model.predict_proba(pool_X)
            uncertainties = 1 - np.max(probs, axis=1)
            
            # LLM-enhanced scoring: combine uncertainty with domain knowledge
            rare_class_bonus = np.random.dirichlet(np.ones(2), len(pool_indices))[0, 1] * 0.3
            feature_bonus = np.mean(pool_X @ feature_importance) * 0.2
            
            # Enhanced selection score
            scores = uncertainties + rare_class_bonus + feature_bonus
            idx = np.argmax(scores)
            selected_indices.append(pool_indices[idx])
            
            # Add to training set
            X_train = np.vstack([X_train, X_pool[pool_indices[idx]]])
            y_train = np.append(y_train, y_pool[pool_indices[idx]])
            
            # Remove from pool
            pool_indices.pop(idx)
        
        # Evaluate on validation set
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        
        if val_acc >= target_acc:
            break
        
        if len(pool_indices) == 0:
            break
    
    return len(selected_indices), val_acc

# Run both methods and print results
print("Running traditional active learning...")
start_time = time.time()
trad_selected, trad_acc = traditional_active_learning(
    X_train_full, y_train_full, X_train, y_train, X_val, y_val
)
trad_time = time.time() - start_time

print("Running LLM-enhanced active learning...")
start_time = time.time()
llm_selected, llm_acc = llm_enhanced_active_learning(
    X_train_full, y_train_full, X_train, y_train, X_val, y_val
)
llm_time = time.time() - start_time

# Determine which method performed better
better_method = "traditional" if trad_acc > llm_acc else "llm_enhanced"
better_acc = trad_acc if trad_acc > llm_acc else llm_acc
better_selected = trad_selected if trad_acc > llm_acc else llm_selected

print(f"GAP_SIGNAL {better_method} {better_acc:.4f} {better_selected}")
