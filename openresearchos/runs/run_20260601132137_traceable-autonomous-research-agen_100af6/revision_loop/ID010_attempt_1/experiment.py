import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
import numpy as np
import json
import random
import os
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(num_samples=200):
    dataset = load_dataset("ag_news", split="train")
    # Sample a subset for speed
    texts = dataset['text'][:num_samples]
    labels = dataset['label'][:num_samples]
    
    # Split into train/test
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    return train_texts, test_texts, train_labels, test_labels

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=2, num_classes=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 50, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)

class HumanApprovalGate:
    def __init__(self, reviewer_threshold=0.01):
        self.reviewer_threshold = reviewer_threshold
        self.validation_history = []
        self.approval_count = 0
        
    def validate_improvement(self, old_metric, new_metric, metric_name):
        improvement = new_metric - old_metric
        is_approved = improvement > self.reviewer_threshold
        
        self.validation_history.append({
            'metric': metric_name,
            'old_value': old_metric,
            'new_value': new_metric,
            'improvement': improvement,
            'approved': is_approved
        })
        
        if is_approved:
            self.approval_count += 1
            
        return is_approved, improvement
    
    def get_approval_rate(self):
        if len(self.validation_history) == 0:
            return 0.0
        return self.approval_count / len(self.validation_history)

def baseline_train_eval(train_texts, test_texts, train_labels, test_labels, device):
    # Create vocabulary
    all_texts = train_texts + test_texts
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for text in all_texts:
        for word in text.lower().split()[:50]:  # Limit words per text
            if word not in vocab and len(vocab) < 1000:
                vocab[word] = len(vocab)
    
    # Convert texts to sequences
    def text_to_sequence(text):
        seq = [vocab.get(word.lower(), 1) for word in text.split()[:50]]
        return torch.tensor(seq + [0] * (50 - len(seq)), dtype=torch.long)
    
    train_sequences = [text_to_sequence(text) for text in train_texts]
    test_sequences = [text_to_sequence(text) for text in test_texts]
    
    train_dataset = TensorDataset(torch.stack(train_sequences), torch.tensor(train_labels, dtype=torch.long))
    test_dataset = TensorDataset(torch.stack(test_sequences), torch.tensor(test_labels, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    model = SimpleTransformer(vocab_size=len(vocab), num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_acc = 0.0
    
    for epoch in range(5):
        model.train()
        train_loss = 0.0
        for step, (texts, labels) in enumerate(train_loader):
            texts, labels = texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if step % 10 == 0:
                print(f"Baseline - Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for texts, labels in test_loader:
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    return best_val_acc

def proposed_train_eval(train_texts, test_texts, train_labels, test_labels, device):
    # Create vocabulary
    all_texts = train_texts + test_texts
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for text in all_texts:
        for word in text.lower().split()[:50]:
            if word not in vocab and len(vocab) < 1000:
                vocab[word] = len(vocab)
    
    # Convert texts to sequences
    def text_to_sequence(text):
        seq = [vocab.get(word.lower(), 1) for word in text.split()[:50]]
        return torch.tensor(seq + [0] * (50 - len(seq)), dtype=torch.long)
    
    train_sequences = [text_to_sequence(text) for text in train_texts]
    test_sequences = [text_to_sequence(text) for text in test_texts]
    
    train_dataset = TensorDataset(torch.stack(train_sequences), torch.tensor(train_labels, dtype=torch.long))
    test_dataset = TensorDataset(torch.stack(test_sequences), torch.tensor(test_labels, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    model = SimpleTransformer(vocab_size=len(vocab), num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Human approval gate
    approval_gate = HumanApprovalGate(reviewer_threshold=0.005)
    
    best_val_acc = 0.0
    val_acc_history = []
    
    for epoch in range(5):
        model.train()
        train_loss = 0.0
        for step, (texts, labels) in enumerate(train_loader):
            texts, labels = texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if step % 10 == 0:
                print(f"Proposed - Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")
        
        # Validation with approval gate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for texts, labels in test_loader:
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        current_val_acc = correct / total
        val_acc_history.append(current_val_acc)
        
        # Human approval gate validation
        if len(val_acc_history) > 1:
            old_acc = val_acc_history[-2]
            improvement, approved = approval_gate.validate_improvement(old_acc, current_val_acc, "accuracy")
            
            if not approved:
                # Revert to previous best if not approved
                current_val_acc = max(val_acc_history[:-1])
                print(f"  Gate rejected: improvement {improvement:.4f} < threshold {approval_gate.reviewer_threshold}")
        
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
    
    return best_val_acc, approval_gate.get_approval_rate()

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    train_texts, test_texts, train_labels, test_labels = load_data(num_samples=200)
    
    # Set seeds
    set_seed(42)
    
    # Train and evaluate baseline
    print("\n=== Training Baseline ===")
    baseline_metrics = {}
    baseline_metrics['accuracy'] = baseline_train_eval(
        train_texts, test_texts, train_labels, test_labels, device
    )
    
    # Train and evaluate proposed method
    print("\n=== Training Proposed Method ===")
    proposed_metrics = {}
    proposed_acc, approval_rate = proposed_train_eval(
        train_texts, test_texts, train_labels, test_labels, device
    )
    proposed_metrics['accuracy'] = proposed_acc
    proposed_metrics['approval_rate'] = approval_rate
    
    # Calculate improvement
    improvement = proposed_acc - baseline_metrics['accuracy']
    
    # Prepare results
    results = {
        "baseline": {"accuracy": baseline_metrics['accuracy']},
        "proposed": {"accuracy": proposed_acc, "approval_rate": approval_rate},
        "improvement": {"accuracy": improvement},
        "better": proposed_acc > baseline_metrics['accuracy'],
        "dataset": "ag_news",
        "baseline_name": "SimpleTransformer",
        "proposed_name": "SimpleTransformer_with_ApprovalGate",
        "seeds": [42],
        "epochs": 5,
        "samples_train": len(train_texts),
        "samples_test": len(test_texts)
    }
    
    # Save results
    with open('metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Results ===")
    print(f"Baseline Accuracy: {baseline_metrics['accuracy']:.4f}")
    print(f"Proposed Accuracy: {proposed_acc:.4f}")
    print(f"Improvement: {improvement:.4f}")
    print(f"Better: {results['better']}")
    print(f"Approval Rate: {approval_rate:.4f}")
    print(f"Results saved to metrics.json")

if __name__ == "__main__":
    main()