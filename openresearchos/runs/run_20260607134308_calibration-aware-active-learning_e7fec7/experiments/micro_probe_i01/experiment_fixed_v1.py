import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
import numpy as np
import json
import random
import time
from tqdm import tqdm

# Set seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Simple MLP Baseline
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Proposed Method with Reflexion-Enhanced Calibration
class ReflexionMLP(nn.Module):
    def __init__(self, input_dim):
        super(ReflexionMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Calibration-aware reflection mechanism
def reflection_step(model, x, y, confidence_threshold=0.7):
    model.eval()
    with torch.no_grad():
        logits = model(x.float().unsqueeze(0))
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).float()
        
        # Calculate calibration metrics
        brier = torch.nn.functional.binary_cross_entropy(probs, y.float().unsqueeze(0), reduction='mean')
        confidence = probs.max(dim=1)[0]
        
        # Identify poorly calibrated samples
        poorly_calibrated = (confidence < confidence_threshold) | (torch.abs(probs - y.float().unsqueeze(0)) > 0.3)
        
        return pred.item(), probs.item(), brier.item(), poorly_calibrated.bool()

# Active learning selector based on reflection
def select_samples_by_reflection(model, unlabeled_indices, unlabeled_data, n_select=10):
    selected = []
    remaining = list(unlabeled_indices)
    
    for _ in range(min(n_select, len(remaining))):
        if not remaining:
            break
            
        # Get reflection scores for remaining samples
        scores = []
        for idx in remaining:
            _, _, brier, poorly_calibrated = reflection_step(model, unlabeled_data[idx], torch.tensor([0]))
            # Prioritize poorly calibrated or high uncertainty samples
            score = brier.item() + (2.0 if poorly_calibrated else 0.0)
            scores.append((idx, score))
        
        # Select sample with highest reflection score
        scores.sort(key=lambda x: x[1], reverse=True)
        selected.append(scores[0][0])
        remaining.remove(scores[0][0])
    
    return selected

# Training function
def train_model(model, train_loader, val_loader, epochs=5, learning_rate=0.001):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data.float())
            loss = criterion(output, target.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data.float())
                loss = criterion(output, target.float().unsqueeze(1))
                val_loss += loss.item()
        
        val_losses.append(val_loss / len(val_loader))
        scheduler.step(val_losses[-1])
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
    
    return train_losses, val_losses

# Main experiment function
def run_experiment():
    # Load and preprocess data
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Create datasets and loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize models
    input_dim = X_train.shape[1]
    mlp = MLP(input_dim).to(device)
    reflexion_mlp = ReflexionMLP(input_dim).to(device)
    
    # Train baseline MLP
    print("\nTraining Baseline MLP...")
    mlp_train_losses, mlp_val_losses = train_model(mlp, train_loader, val_loader, epochs=10)
    
    # Train Reflexion MLP
    print("\nTraining Reflexion MLP...")
    reflexion_mlp_train_losses, reflexion_mlp_val_losses = train_model(reflexion_mlp, train_loader, val_loader, epochs=10)
    
    # Evaluate models
    mlp.eval()
    reflexion_mlp.eval()
    
    mlp_predictions = []
    mlp_targets = []
    reflexion_mlp_predictions = []
    reflexion_mlp_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            # MLP predictions
            mlp_output = mlp(data.float())
            mlp_preds = torch.sigmoid(mlp_output) > 0.5
            mlp_predictions.extend(mlp_preds.cpu().numpy())
            mlp_targets.extend(target.numpy())
            
            # Reflexion MLP predictions
            reflexion_output = reflexion_mlp(data.float())
            reflexion_preds = torch.sigmoid(reflexion_output) > 0.5
            reflexion_mlp_predictions.extend(reflexion_preds.cpu().numpy())
            reflexion_mlp_targets.extend(target.numpy())
    
    # Calculate metrics
    mlp_acc = accuracy_score(mlp_targets, mlp_predictions)
    mlp_brier = brier_score_loss(mlp_targets, [torch.sigmoid(mlp(torch.FloatTensor(X_test[i])).item()) for i in range(len(X_test))])
    mlp_auc = roc_auc_score(mlp_targets, [torch.sigmoid(mlp(torch.FloatTensor(X_test[i])).item()) for i in range(len(X_test))])
    
    reflexion_mlp_acc = accuracy_score(reflexion_mlp_targets, reflexion_mlp_predictions)
    reflexion_mlp_brier = brier_score_loss(reflexion_mlp_targets, [torch.sigmoid(reflexion_mlp(torch.FloatTensor(X_test[i])).item()) for i in range(len(X_test))])
    reflexion_mlp_auc = roc_auc_score(reflexion_mlp_targets, [torch.sigmoid(reflexion_mlp(torch.FloatTensor(X_test[i])).item()) for i in range(len(X_test))])
    
    # Save metrics to JSON
    metrics = {
        "baseline": {
            "accuracy": float(mlp_acc),
            "brier_score": float(mlp_brier),
            "auc": float(mlp_auc),
            "train_losses": mlp_train_losses,
            "val_losses": mlp_val_losses
        },
        "proposed": {
            "accuracy": float(reflexion_mlp_acc),
            "brier_score": float(reflexion_mlp_brier),
            "auc": float(reflexion_mlp_auc),
            "train_losses": reflexion_mlp_train_losses,
            "val_losses": reflexion_mlp_val_losses
        }
    }
    
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\nMetrics saved to metrics.json")
    print(f"Baseline MLP - Accuracy: {mlp_acc:.4f}, Brier: {mlp_brier:.4f}, AUC: {mlp_auc:.4f}")
    print(f"Reflexion MLP - Accuracy: {reflexion_mlp_acc:.4f}, Brier: {reflexion_mlp_brier:.4f}, AUC: {reflexion_mlp_auc:.4f}")

if __name__ == "__main__":
    set_seed(42)
    run_experiment()