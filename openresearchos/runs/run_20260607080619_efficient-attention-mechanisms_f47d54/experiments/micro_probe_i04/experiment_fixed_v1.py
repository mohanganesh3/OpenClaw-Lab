import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import json
import time
from sklearn.preprocessing import StandardScaler

# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load and prepare dataset
print("Loading digits dataset...")
digits = load_digits()
X = digits.data
y = digits.target

# Use only 200 samples for micro-probe
indices = np.random.choice(len(X), 200, replace=False)
X = X[indices]
y = y[indices]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_train_tensor = torch.LongTensor(y_train).to(device)
y_test_tensor = torch.LongTensor(y_test).to(device)

# Baseline: SVM with RBF kernel
print("\nTraining baseline SVM...")
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_f1 = f1_score(y_test, y_pred_svm, average='weighted')

# Proposed: CNN with Context-Aware Attention
class ContextAwareAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=10):
        super(ContextAwareAttention, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.context_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        context = torch.mean(features, dim=1, keepdim=True)
        context = self.context_gate(context)
        context = torch.sigmoid(context)
        attended_features = features * context
        attended_features = torch.mean(attended_features, dim=1)
        output = self.classifier(attended_features)
        return output

class AttentionCNN(nn.Module):
    def __init__(self, input_dim=64, num_classes=10):
        super(AttentionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        # After conv layers: (batch_size, 32, 2, 2) = 128 features
        self.attention = ContextAwareAttention(128, hidden_dim=64, num_classes=num_classes)
        self.classifier = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        attended = self.attention(x)
        output = self.classifier(attended)
        return output

# Reshape data for CNN (add channel dimension)
X_train_reshaped = X_train_tensor.view(-1, 1, 8, 8)
X_test_reshaped = X_test_tensor.view(-1, 1, 8, 8)

# Initialize model
model = AttentionCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
print("\nTraining proposed model with context-aware attention...")
model.train()
num_epochs = 1
steps_per_epoch = len(X_train_reshaped) // 32
current_step = 0

start_time = time.time()
for epoch in range(num_epochs):
    for i in range(0, len(X_train_reshaped), 32):
        batch_x = X_train_reshaped[i:i+32]
        batch_y = y_train_tensor[i:i+32]
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        current_step += 1
        if current_step % 10 == 0:
            print(f"Step {current_step}, Loss: {loss.item():.4f}")

# Evaluate model
model.eval()
with torch.no_grad():
    y_pred = model(X_test_reshaped)
    _, predicted_classes = torch.max(y_pred, 1)
    test_accuracy = accuracy_score(y_test_tensor, predicted_classes)
    test_f1 = f1_score(y_test_tensor, predicted_classes, average='weighted')

# Save metrics
metrics = {
    'svm_accuracy': svm_accuracy,
    'svm_f1': svm_f1,
    'cnn_accuracy': test_accuracy,
    'cnn_f1': test_f1
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\nFinal Results:")
print(f"SVM Accuracy: {svm_accuracy:.4f}, F1: {svm_f1:.4f}")
print(f"CNN Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}")
print(f"Metrics saved to metrics.json")