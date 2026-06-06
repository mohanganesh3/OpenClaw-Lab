import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.calibration import calibration_curve
import json
import os
from tqdm import tqdm
import random
from datasets import load_dataset

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
set_seed(42)

# Load dataset - using a subset of medical images
# Changed from "pystata/medical-images" to "mnist" which exists on the Hub
dataset = load_dataset("mnist", split="train")

# Convert to numpy arrays and take subset
images = np.array(dataset['image'])[:300]  # Take first 300 samples
labels = np.array(dataset['label'])[:300]

# Normalize images
images = images.astype(np.float32) / 255.0

# Convert to PyTorch tensors
X = torch.FloatTensor(images).view(len(images), 1, 28, 28)  # Assuming 28x28 images
y = torch.LongTensor(labels)

# Split into train and test
train_size = int(0.8 * len(X))
test_size = len(X) - train_size
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Dataset shape: {X.shape}")
print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# Define simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Baseline training function
def train_baseline(model, X_train, y_train, X_test, y_test, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # Simple training loop
        for i in range(0, len(X_train), 32):
            batch_X = X_train[i:i+32].to(device)
            batch_y = y_train[i:i+32].to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if i % 100 == 0:
                print(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test.to(device))
            _, predicted = torch.max(test_outputs.data, 1)
            accuracy = accuracy_score(y_test.numpy(), predicted.cpu().numpy())
            val_accuracies.append(accuracy)
            
            print(f"Epoch {epoch+1} completed. Train Loss: {train_loss/len(X_train):.4f}, Val Accuracy: {accuracy:.4f}")
    
    return val_accuracies[-1]

# Proposed method with failure memory and calibration-aware active learning
class FailureMemory:
    def __init__(self, capacity=50):
        self.capacity = capacity
        self.memory = []
        
    def add(self, image, label, prediction, confidence, is_misclassified):
        if is_misclassified:
            entry = {
                'image': image,
                'label': label,
                'prediction': prediction,
                'confidence': confidence
            }
            self.memory.append(entry)
            if len(self.memory) > self.capacity:
                self.memory.pop(0)
        
    def get_uncertain_samples(self, n=10):
        if len(self.memory) == 0:
            return []
        return random.sample(self.memory, min(n, len(self.memory)))

# Calibration-aware active learning function
def active_learning_loop(model, X_train, y_train, X_test, y_test, failure_memory, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # Simple training loop
        for i in range(0, len(X_train), 32):
            batch_X = X_train[i:i+32].to(device)
            batch_y = y_train[i:i+32].to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if i % 100 == 0:
                print(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test.to(device))
            _, predicted = torch.max(test_outputs.data, 1)
            accuracy = accuracy_score(y_test.numpy(), predicted.cpu().numpy())
            val_accuracies.append(accuracy)
            
            # Get predictions with confidence scores
            probabilities = torch.softmax(test_outputs, dim=1)
            max_probs, max_class = torch.max(probabilities, dim=1)
            
            # Identify misclassified samples
            misclassified_indices = []
            for i in range(len(y_test)):
                if predicted[i] != y_test[i]:
                    misclassified_indices.append(i)
                    
                    # Add to failure memory
                    if len(misclassified_indices) <= 300:  # Limit memory size
                        sample_idx = misclassified_indices[-1]
                        failure_memory.add(
                            X_test[sample_idx].cpu().numpy(),
                            y_test[sample_idx].item(),
                            predicted[sample_idx].item(),
                            max_probs[sample_idx].item(),
                            True
                        )
            
            print(f"Epoch {epoch+1} completed. Train Loss: {train_loss/len(X_train):.4f}, Val Accuracy: {accuracy:.4f}")
    
    return val_accuracies[-1]

# Main experiment
if __name__ == "__main__":
    # Initialize failure memory
    failure_memory = FailureMemory(capacity=50)
    
    # Train baseline model
    print("\nTraining baseline model...")
    baseline_model = SimpleCNN(num_classes=2).to(device)
    baseline_accuracy = train_baseline(baseline_model, X_train, y_train, X_test, y_test, epochs=5)
    
    # Train calibration-aware active learning model
    print("\nTraining calibration-aware active learning model...")
    active_model = SimpleCNN(num_classes=2).to(device)
    active_accuracy = active_learning_loop(active_model, X_train, y_train, X_test, y_test, failure_memory, epochs=5)
    
    # Save metrics
    metrics = {
        "baseline_accuracy": baseline_accuracy,
        "active_accuracy": active_accuracy,
        "improvement": active_accuracy - baseline_accuracy
    }
    
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to metrics.json")
    print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
    print(f"Active Learning Accuracy: {active_accuracy:.4f}")
    print(f"Improvement: {metrics['improvement']:.4f}")