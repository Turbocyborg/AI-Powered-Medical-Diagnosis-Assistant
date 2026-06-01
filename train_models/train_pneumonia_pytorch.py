"""
Pneumonia Detection Model Training Script (PyTorch)
Uses CNN with Transfer Learning (ResNet18) for chest X-ray classification
"""

import os
import numpy as np
import pickle
from datetime import datetime
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

print("=" * 80)
print("PNEUMONIA DETECTION MODEL TRAINING (PyTorch)")
print("=" * 80)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "archive", "chest_xray")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_DIR = os.path.join(BASE_DIR, "models", "pneumonia")

os.makedirs(MODEL_DIR, exist_ok=True)

print(f"\nData directories:")
print(f"  Train: {TRAIN_DIR}")
print(f"  Validation: {VAL_DIR}")
print(f"  Test: {TEST_DIR}")
print(f"  Model output: {MODEL_DIR}")

# Data transforms
train_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

val_test_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

# Load datasets
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_test_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_test_transform)

# Data loaders
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

print(f"\nDataset Statistics:")
print(f"  Training samples: {len(train_dataset)}")
print(f"  Validation samples: {len(val_dataset)}")
print(f"  Test samples: {len(test_dataset)}")
print(f"  Classes: {train_dataset.classes}")
print(f"  Class to index: {train_dataset.class_to_idx}")

# Save class indices
with open(os.path.join(MODEL_DIR, "class_indices.pkl"), "wb") as f:
    pickle.dump(train_dataset.class_to_idx, f)

print("\n" + "=" * 80)
print("BUILDING MODEL")
print("=" * 80)

# Load pre-trained ResNet18
model = models.resnet18(pretrained=True)

# Freeze early layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 2),  # Binary classification
)

model = model.to(device)

print("\nModel Architecture:")
print(f"  Base: ResNet18 (pretrained)")
print(f"  Modified FC layer: {num_features} -> 512 -> 2")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(
    f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)

print("\n" + "=" * 80)
print("TRAINING MODEL")
print("=" * 80)


# Training function
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


# Validation function
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels, all_probs


# Training loop
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc = 0.0
best_model_path = os.path.join(MODEL_DIR, "best_model.pth")

for epoch in range(EPOCHS):
    start_time = time.time()

    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, device
    )
    val_loss, val_acc, _, _, _ = validate(model, val_loader, criterion, device)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    scheduler.step(val_loss)

    epoch_time = time.time() - start_time

    print(f"Epoch {epoch + 1}/{EPOCHS} ({epoch_time:.1f}s)")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"  ✓ Best model saved! (Val Acc: {val_acc:.2f}%)")

print("\n" + "=" * 80)
print("FINE-TUNING MODEL")
print("=" * 80)

# Unfreeze more layers for fine-tuning
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE / 10)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)

print("Fine-tuning all layers with lower learning rate...")

for epoch in range(5):  # Fine-tune for 5 more epochs
    start_time = time.time()

    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, device
    )
    val_loss, val_acc, _, _, _ = validate(model, val_loader, criterion, device)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    scheduler.step(val_loss)

    epoch_time = time.time() - start_time

    print(f"Fine-tune Epoch {epoch + 1}/5 ({epoch_time:.1f}s)")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"  ✓ Best model saved! (Val Acc: {val_acc:.2f}%)")

print("\n" + "=" * 80)
print("EVALUATING MODEL")
print("=" * 80)

# Load best model
model.load_state_dict(torch.load(best_model_path))

# Evaluate on test set
test_loss, test_acc, test_preds, test_labels, test_probs = validate(
    model, test_loader, criterion, device
)

print(f"\nTest Results:")
print(f"  Loss: {test_loss:.4f}")
print(f"  Accuracy: {test_acc:.2f}%")

# Calculate AUC
test_auc = roc_auc_score(test_labels, test_probs)
print(f"  AUC-ROC: {test_auc:.4f}")

# Classification report
print("\nClassification Report:")
class_names = train_dataset.classes
print(classification_report(test_labels, test_preds, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(test_labels, test_preds)
print("\nConfusion Matrix:")
print(cm)

# Calculate metrics
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp)
f1 = 2 * (precision * sensitivity) / (precision + sensitivity)

print(f"\nDetailed Metrics:")
print(f"  Sensitivity (Recall): {sensitivity * 100:.2f}%")
print(f"  Specificity: {specificity * 100:.2f}%")
print(f"  Precision: {precision * 100:.2f}%")
print(f"  F1-Score: {f1:.4f}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.title("Confusion Matrix - Pneumonia Detection")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(
    os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=300, bbox_inches="tight"
)
print(f"\nConfusion matrix saved to: {os.path.join(MODEL_DIR, 'confusion_matrix.png')}")

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(history["train_acc"], label="Train Accuracy")
axes[0].plot(history["val_acc"], label="Val Accuracy")
axes[0].set_title("Model Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy (%)")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history["train_loss"], label="Train Loss")
axes[1].plot(history["val_loss"], label="Val Loss")
axes[1].set_title("Model Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(
    os.path.join(MODEL_DIR, "training_history.png"), dpi=300, bbox_inches="tight"
)
print(f"Training history saved to: {os.path.join(MODEL_DIR, 'training_history.png')}")

# Save model info
model_info = {
    "test_accuracy": float(test_acc),
    "test_auc": float(test_auc),
    "sensitivity": float(sensitivity),
    "specificity": float(specificity),
    "precision": float(precision),
    "f1_score": float(f1),
    "img_size": IMG_SIZE,
    "class_to_idx": train_dataset.class_to_idx,
    "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "framework": "pytorch",
    "model_architecture": "resnet18",
}

with open(os.path.join(MODEL_DIR, "model_info.pkl"), "wb") as f:
    pickle.dump(model_info, f)

print(f"\nModel info saved to: {os.path.join(MODEL_DIR, 'model_info.pkl')}")

print("\n" + "=" * 80)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"\nModel files saved in: {MODEL_DIR}")
print(f"  - best_model.pth (PyTorch model)")
print(f"  - class_indices.pkl (Class mapping)")
print(f"  - model_info.pkl (Model metadata)")
print(f"  - confusion_matrix.png (Visualization)")
print(f"  - training_history.png (Training curves)")
print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
print(f"Final Test AUC: {test_auc:.4f}")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
print("=" * 80)


# ================================================================================
# PNEUMONIA DETECTION MODEL TRAINING (PyTorch)
# ================================================================================
# PyTorch version: 2.12.0+cpu
# CUDA available: False
# Training started at: 2026-05-20 17:38:23
# ================================================================================

# Using device: cpu

# Data directories:
#   Train: D:\Best\Coding\Py\AI-Powered-Medical-Diagnosis-Assistant\data\archive\chest_xray\train
#   Validation: D:\Best\Coding\Py\AI-Powered-Medical-Diagnosis-Assistant\data\archive\chest_xray\val
#   Test: D:\Best\Coding\Py\AI-Powered-Medical-Diagnosis-Assistant\data\archive\chest_xray\test
#   Model output: D:\Best\Coding\Py\AI-Powered-Medical-Diagnosis-Assistant\models\pneumonia

# ================================================================================
# LOADING DATA
# ================================================================================

# Dataset Statistics:
#   Training samples: 5216
#   Validation samples: 16
#   Test samples: 624
#   Classes: ['NORMAL', 'PNEUMONIA']
#   Class to index: {'NORMAL': 0, 'PNEUMONIA': 1}

# ================================================================================
# BUILDING MODEL
# ================================================================================
# C:\Users\Turbocyb0rg\AppData\Roaming\Python\Python314\site-packages\torchvision\models\_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
#   warnings.warn(
# C:\Users\Turbocyb0rg\AppData\Roaming\Python\Python314\site-packages\torchvision\models\_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet18_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet18_Weights.DEFAULT` to get the most up-to-date weights.
#   warnings.warn(msg)

# Model Architecture:
#   Base: ResNet18 (pretrained)
#   Modified FC layer: 512 -> 512 -> 2
#   Total parameters: 11,440,194
#   Trainable parameters: 263,682

# ================================================================================
# TRAINING MODEL
# ================================================================================
# Epoch 1/15 (331.9s)
#   Train Loss: 0.3263 | Train Acc: 85.60%
#   Val Loss: 0.3369 | Val Acc: 75.00%
#   ✓ Best model saved! (Val Acc: 75.00%)
# Epoch 2/15 (253.2s)
#   Train Loss: 0.2813 | Train Acc: 87.88%
#   Val Loss: 0.3228 | Val Acc: 81.25%
#   ✓ Best model saved! (Val Acc: 81.25%)
# Epoch 3/15 (255.2s)
#   Train Loss: 0.2506 | Train Acc: 89.24%
#   Val Loss: 0.2732 | Val Acc: 87.50%
#   ✓ Best model saved! (Val Acc: 87.50%)
# Epoch 4/15 (255.5s)
#   Train Loss: 0.2508 | Train Acc: 89.78%
#   Val Loss: 0.3036 | Val Acc: 81.25%
# Epoch 5/15 (254.9s)
#   Train Loss: 0.2467 | Train Acc: 89.19%
#   Val Loss: 0.2781 | Val Acc: 87.50%
# Epoch 6/15 (300.6s)
#   Train Loss: 0.2323 | Train Acc: 90.26%
#   Val Loss: 0.4411 | Val Acc: 75.00%
# Epoch 7/15 (319.0s)
#   Train Loss: 0.2312 | Train Acc: 90.32%
#   Val Loss: 0.3105 | Val Acc: 75.00%
# Epoch 8/15 (315.0s)
#   Train Loss: 0.2233 | Train Acc: 90.70%
#   Val Loss: 0.2745 | Val Acc: 81.25%
# Epoch 9/15 (315.8s)
#   Train Loss: 0.2154 | Train Acc: 90.87%
#   Val Loss: 0.3789 | Val Acc: 75.00%
# Epoch 10/15 (314.7s)
#   Train Loss: 0.2253 | Train Acc: 90.41%
#   Val Loss: 0.3835 | Val Acc: 75.00%
# Epoch 11/15 (314.6s)
#   Train Loss: 0.2328 | Train Acc: 90.41%
#   Val Loss: 0.3474 | Val Acc: 75.00%
# Epoch 12/15 (316.1s)
#   Train Loss: 0.2243 | Train Acc: 90.66%
#   Val Loss: 0.3323 | Val Acc: 75.00%
# Epoch 13/15 (314.7s)
#   Train Loss: 0.2256 | Train Acc: 90.66%
#   Val Loss: 0.3246 | Val Acc: 75.00%
# Epoch 14/15 (314.7s)
#   Train Loss: 0.2121 | Train Acc: 90.89%
#   Val Loss: 0.2994 | Val Acc: 81.25%
# Epoch 15/15 (316.0s)
#   Train Loss: 0.2118 | Train Acc: 91.10%
#   Val Loss: 0.3754 | Val Acc: 75.00%

# ================================================================================
# FINE-TUNING MODEL
# ================================================================================
# Fine-tuning all layers with lower learning rate...
# Fine-tune Epoch 1/5 (643.2s)
#   Train Loss: 0.1182 | Train Acc: 95.76%
#   Val Loss: 0.2707 | Val Acc: 87.50%
# Fine-tune Epoch 2/5 (642.0s)
#   Train Loss: 0.0663 | Train Acc: 97.45%
#   Val Loss: 0.1339 | Val Acc: 93.75%
#   ✓ Best model saved! (Val Acc: 93.75%)
# Fine-tune Epoch 3/5 (643.1s)
#   Train Loss: 0.0661 | Train Acc: 97.45%
#   Val Loss: 0.3763 | Val Acc: 75.00%
# Fine-tune Epoch 4/5 (640.4s)
#   Train Loss: 0.0507 | Train Acc: 98.24%
#   Val Loss: 0.3799 | Val Acc: 81.25%
# Fine-tune Epoch 5/5 (640.9s)
#   Train Loss: 0.0355 | Train Acc: 98.70%
#   Val Loss: 0.6089 | Val Acc: 68.75%

# ================================================================================
# EVALUATING MODEL
# ================================================================================

# Test Results:
#   Loss: 0.6098
#   Accuracy: 86.54%
#   AUC-ROC: 0.9599

# Classification Report:
#               precision    recall  f1-score   support

#       NORMAL       0.99      0.65      0.78       234
#    PNEUMONIA       0.83      0.99      0.90       390

#     accuracy                           0.87       624
#    macro avg       0.91      0.82      0.84       624
# weighted avg       0.89      0.87      0.86       624


# Confusion Matrix:
# [[152  82]
#  [  2 388]]

# Detailed Metrics:
#   Sensitivity (Recall): 99.49%
#   Specificity: 64.96%
#   Precision: 82.55%
#   F1-Score: 0.9023

# Confusion matrix saved to: D:\Best\Coding\Py\AI-Powered-Medical-Diagnosis-Assistant\models\pneumonia\confusion_matrix.png
# Training history saved to: D:\Best\Coding\Py\AI-Powered-Medical-Diagnosis-Assistant\models\pneumonia\training_history.png

# Model info saved to: D:\Best\Coding\Py\AI-Powered-Medical-Diagnosis-Assistant\models\pneumonia\model_info.pkl

# ================================================================================
# TRAINING COMPLETED SUCCESSFULLY!
# ================================================================================

# Model files saved in: D:\Best\Coding\Py\AI-Powered-Medical-Diagnosis-Assistant\models\pneumonia       
#   - best_model.pth (PyTorch model)
#   - class_indices.pkl (Class mapping)
#   - model_info.pkl (Model metadata)
#   - confusion_matrix.png (Visualization)
#   - training_history.png (Training curves)

# Final Test Accuracy: 86.54%
# Final Test AUC: 0.9599
# Best Validation Accuracy: 93.75%
# ================================================================================