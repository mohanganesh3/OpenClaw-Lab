import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss
from datasets import load_dataset
import json
import os
from typing import List, Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class FailureMemory:
    """
    Failure memory for research agents that stores failed predictions
    and guides active learning decisions based on calibration-aware metrics.
    """
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.failed_samples = []
        self.reviewer_metrics = {
            'accuracy': [],
            'calibration_brier': [],
            'confidence_entropy': []
        }
        self.state_transitions = []
        
    def add_failure(self, sample_idx, prediction, confidence, true_label, 
                   calibration_metric, entropy):
        """Add a failed sample to memory"""
        failure_entry = {
            'sample_idx': sample_idx,
            'prediction': prediction,
            'confidence': confidence,
            'true_label': true_label,
            'calibration_metric': calibration_metric,
            'entropy': entropy,
            'timestamp': len(self.failed_samples)
        }
        
        self.failed_samples.append(failure_entry)
        if len(self.failed_samples) > self.capacity:
            self.failed_samples = self.failed_samples[-self.capacity:]
    
    def get_uncertainty_samples(self, k=10):
        """Get samples with highest uncertainty for active learning"""
        if not self.failed_samples:
            return []
        
        uncertainties = [f['entropy'] for f in self.failed_samples]
        top_k_indices = np.argsort(uncertainties)[-k:]
        return [self.failed_samples[i]['sample_idx'] for i in top_k_indices]
    
    def get_calibration_failures(self, threshold=0.3):
        """Get poorly calibrated samples"""
        calibration_failures = []
        for failure in self.failed_samples:
            if failure['calibration_metric'] > threshold:
                calibration_failures.append(failure['sample_idx'])
        return calibration_failures
    
    def update_reviewer_metrics(self, accuracy, brier_score, entropy):
        """Update reviewer metrics for tracking progress"""
        self.reviewer_metrics['accuracy'].append(accuracy)
        self.reviewer_metrics['calibration_brier'].append(brier_score)
        self.reviewer_metrics['confidence_entropy'].append(entropy)

class MedicalCNN(nn.Module):
    """Simple CNN for medical image classification"""
    def __init__(self, num_classes=2):
        super(MedicalCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 14 * 14)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CalibrationAwareActiveLearning:
    """
    Proposed method: CNN with failure memory and calibration-aware active learning
    """
    def __init__(self, model, memory, train_loader, val_loader):
        self.model = model
        self.memory = memory
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        
    def compute_calibration_metrics(self, predictions, confidences, true_labels):
        """Compute calibration metrics"""
        # Brier score for calibration
        brier_score = brier_score_loss(true_labels.float(), confidences)
        
        # Confidence entropy
        entropy = -(confidences * torch.log(confidences + 1e-8)).sum(dim=1).mean()
        
        return brier_score.item(), entropy.item()
    
    def train_epoch(self, epoch):
        """Train for one epoch with failure memory updates"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_confidences = []
        all_true_labels = []
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(device), targets.to(device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Get predictions and confidences for failure memory
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            confidences = torch.max(probs, dim=1)[0]
            
            all_predictions.extend(preds.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            all_true_labels.extend(targets.cpu().numpy())
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # Update failure memory with calibration metrics
        brier_score, entropy = self.compute_calibration_metrics(
            torch.tensor(all_predictions),
            torch.tensor(all_confidences),
            torch.tensor(all_true_labels)
        )
        
        # Identify and store failures
        for i, (pred, conf, true) in enumerate(zip(all_predictions, all_confidences, all_true_labels)):
            if pred != true:
                self.memory.add_failure(
                    sample_idx=i,
                    prediction=pred,
                    confidence=conf,
                    true_label=true,
                    calibration_metric=abs(conf - 0.5),  # Poor calibration indicator
                    entropy=entropy
                )
        
        self.memory.update_reviewer_metrics(
            accuracy_score(all_true_labels, all_predictions),
            brier_score,
            entropy
        )
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate model and return metrics"""
        self.model.eval()
        all_preds = []
        all_confidences = []
        all_true = []
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = self.model(data)
                preds = torch.argmax(outputs, dim=1)
                probs = torch.softmax(outputs, dim=1)
                confidences = torch.max(probs, dim=1)[0]
                
                all_preds.extend(preds.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
                all_true.extend(targets.cpu().numpy())
        
        accuracy = accuracy_score(all_true, all_preds)
        brier_score, entropy = self.compute_calibration_metrics(
            torch.tensor(all_preds),
            torch.tensor(all_confidences),
            torch.tensor(all_true)
        )
        
        return accuracy, brier_score, entropy

def load_chest_xray_data(num_samples=300):
    """Load chest X-ray dataset from HuggingFace"""
    print("Loading chest X-ray dataset...")
    dataset = load_dataset("hf-internal-testing/medical-image-chest-xray", split="train")
    
    # Take subset for faster training
    if len(dataset) > num_samples:
        dataset = dataset.shuffle(seed=42).select(range(num_samples))
    
    # Convert to tensors
    def preprocess(example):
        image = example['image']
        label = example['label']
        # Resize and normalize
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0), size=(64, 64), mode='bilinear'
        ).squeeze(0)
        return {'image': image, 'label': torch.tensor(label, dtype=torch.long)}
    
    dataset = dataset.map(preprocess, remove_columns=['image', 'label'])
    
    # Split into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], 
                                               generator=torch.Generator().manual_seed(42))
    
    return train_dataset, test_dataset

def train_baseline(train_loader, val_loader, epochs=5):
    """Train baseline CNN without failure memory"""
    print("\n=== Training Baseline ===")
    model = MedicalCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        val_acc = 100 * correct / total
        print(f"Epoch {epoch} Validation Accuracy: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    return best_val_acc

def train_proposed(train_loader, val_loader, epochs=5):
    """Train proposed method with failure memory"""
    print("\n=== Training Proposed Method ===")
    model = MedicalCNN().to(device)
    memory = FailureMemory(capacity=50)
    active_learner = CalibrationAwareActiveLearning(model, memory, train_loader, val_loader)
    
    best_val_acc = 0
    for epoch in range(epochs):
        # Training with failure memory
        avg_loss = active_learner.train_epoch(epoch)
        
        # Validation
        val_acc, brier_score, entropy = active_learner.validate()
        print(f"Epoch {epoch} - Val Acc: {val_acc:.4f}, Brier: {brier_score:.4f}, Entropy: {entropy:.4f}")
        
        # Active learning: select uncertain samples from failure memory
        if len(memory.failed_samples) > 0:
            uncertain_indices = memory.get_uncertainty_samples(k=5)
            print(f"Epoch {epoch} - Selected {len(uncertain_indices)} uncertain samples from failure memory")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    return best_val_acc

def main():
    # Load data
    train_dataset, test_dataset = load_chest_xray_data(num_samples=300)
    
    # Create data loaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Train baseline
    baseline_acc = train_baseline(train_loader, test_loader, epochs=5)
    
    # Train proposed method
    proposed_acc = train_proposed(train_loader, test_loader, epochs=5)
    
    # Compute additional metrics for both methods
    def compute_full_metrics(model, loader):
        model.eval()
        all_preds = []
        all_probs = []
        all_true = []
        
        with torch.no_grad():
            for data, targets in loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_true.extend(targets.cpu().numpy())
        
        accuracy = accuracy_score(all_true, all_preds)
        brier_score = brier_score_loss(
            np.array(all_true), 
            np.max(np.array(all_probs), axis=1)
        )
        
        return accuracy, brier_score
    
    # Get final metrics for baseline
    baseline_model = MedicalCNN().to(device)
    # Train baseline model fully for metrics
    for param in baseline_model.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(baseline_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(5):
        baseline_model.train()
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = baseline_model(data)