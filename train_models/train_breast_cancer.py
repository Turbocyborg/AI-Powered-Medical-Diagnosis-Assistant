"""
Breast Cancer Prediction Model Training
Wisconsin Breast Cancer Dataset
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import warnings

warnings.filterwarnings("ignore")

print("=" * 70)
print("BREAST CANCER PREDICTION MODEL TRAINING")
print("Wisconsin Breast Cancer Dataset")
print("=" * 70)

# Load dataset
print("\n1. Loading dataset...")
df = pd.read_csv("data/Breast_Cancer.csv")
print(f"   Dataset shape: {df.shape}")
print(f"   Features: {df.shape[1] - 3} (excluding id, diagnosis, empty column)")

# Data preprocessing
print("\n2. Preprocessing data...")

# Drop unnecessary columns
df = df.drop(["id", "Unnamed: 32"], axis=1, errors="ignore")

# Convert diagnosis to binary (M=1 Malignant, B=0 Benign)
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

print(f"   Target distribution:")
print(
    f"   - Benign (0): {(df['diagnosis'] == 0).sum()} ({(df['diagnosis'] == 0).sum() / len(df) * 100:.1f}%)"
)
print(
    f"   - Malignant (1): {(df['diagnosis'] == 1).sum()} ({(df['diagnosis'] == 1).sum() / len(df) * 100:.1f}%)"
)

# Check for missing values
if df.isnull().sum().sum() > 0:
    print(f"   Handling {df.isnull().sum().sum()} missing values...")
    df = df.dropna()

# Separate features and target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

feature_names = X.columns.tolist()
print(f"   Features used: {len(feature_names)}")

# Split data
print("\n3. Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")

# Feature scaling
print("\n4. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models
print("\n5. Training and evaluating models...")
print("-" * 70)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=15,
        min_samples_leaf=8,
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        min_samples_split=15,
        min_samples_leaf=8,
        random_state=42,
    ),
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "SVM": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
}

results = {}

for name, model in models.items():
    print(f"\n{name}:")

    # Train model
    model.fit(X_train_scaled, y_train)

    # Train predictions
    train_pred = model.predict(X_train_scaled)
    train_proba = model.predict_proba(X_train_scaled)[:, 1]
    train_acc = accuracy_score(y_train, train_pred)
    train_auc = roc_auc_score(y_train, train_proba)

    # Test predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    test_acc = accuracy_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_pred_proba)

    # Overfitting detection
    acc_gap = train_acc - test_acc
    auc_gap = train_auc - test_auc

    # Cross-validation
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train, cv=5, scoring="accuracy"
    )

    results[name] = {
        "model": model,
        "train_acc": train_acc,
        "train_auc": train_auc,
        "test_acc": test_acc,
        "test_auc": test_auc,
        "acc_gap": acc_gap,
        "auc_gap": auc_gap,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
    }

    print(f"   Train Acc: {train_acc:.4f}, AUC: {train_auc:.4f}")
    print(f"   Test  Acc: {test_acc:.4f}, AUC: {test_auc:.4f}")
    print(f"   Gap:       Acc: {acc_gap:.4f}, AUC: {auc_gap:.4f}")
    print(f"   CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    if acc_gap > 0.10 or auc_gap > 0.10:
        print(f"   ⚠️  WARNING: {name} may be overfitting!")
    else:
        print(f"   ✓ {name} appears well-generalized")

# Select best model based on test AUC
print("\n" + "=" * 70)
best_model_name = max(results, key=lambda x: results[x]["test_auc"])
best_model = results[best_model_name]["model"]
best_result = results[best_model_name]

print(f"BEST MODEL: {best_model_name}")
print(f"Test Accuracy: {best_result['test_acc']:.4f}")
print(f"Test AUC Score: {best_result['test_auc']:.4f}")
print(
    f"Train-Test Gap: Acc={best_result['acc_gap']:.4f}, AUC={best_result['auc_gap']:.4f}"
)
print("=" * 70)

# Detailed evaluation
print("\n6. Detailed evaluation of best model...")
y_pred = best_model.predict(X_test_scaled)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"                Predicted")
print(f"              Benign  Malignant")
print(f"Actual Benign    {cm[0][0]:3d}      {cm[0][1]:3d}")
print(f"       Malignant {cm[1][0]:3d}      {cm[1][1]:3d}")

# Feature importance (if available)
if hasattr(best_model, "feature_importances_"):
    print("\nTop 10 Most Important Features:")
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    for i, idx in enumerate(indices, 1):
        print(f"   {i}. {feature_names[idx]}: {importances[idx]:.4f}")

# Save model
print("\n7. Saving model...")
os.makedirs("models/breast_cancer", exist_ok=True)

with open("models/breast_cancer/model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("   ✓ Model saved to models/breast_cancer/model.pkl")

with open("models/breast_cancer/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("   ✓ Scaler saved to models/breast_cancer/scaler.pkl")

# Save feature names
with open("models/breast_cancer/features.txt", "w") as f:
    f.write("\n".join(feature_names))
print("   ✓ Features saved to models/breast_cancer/features.txt")

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print("\nModel Performance Summary:")
print(f"  - Algorithm: {best_model_name}")
print(f"  - Test Accuracy: {best_result['test_acc']:.2%}")
print(f"  - Test AUC Score: {best_result['test_auc']:.4f}")
print(f"  - Train-Test Gap: {best_result['acc_gap']:.4f}")
print(f"  - Features: {len(feature_names)}")
print(f"  - Training samples: {len(X_train)}")
print(f"  - Test samples: {len(X_test)}")
print("\nNext steps:")
print("  1. Run: python app.py")
print("  2. Visit: http://localhost:5000")
print("  3. Test breast cancer prediction")
print("=" * 70)


# Random Forest:
#    Train Acc: 0.9692, AUC: 0.9961
#    Test  Acc: 0.9561, AUC: 0.9937
#    Gap:       Acc: 0.0131, AUC: 0.0024
#    CV Accuracy: 0.9516 (+/- 0.0256)
#    ✓ Random Forest appears well-generalized

# Gradient Boosting:
#    Train Acc: 1.0000, AUC: 1.0000
#    Test  Acc: 0.9649, AUC: 0.9964
#    Gap:       Acc: 0.0351, AUC: 0.0036
#    CV Accuracy: 0.9604 (+/- 0.0372)
#    ✓ Gradient Boosting appears well-generalized

# Logistic Regression:
#    Train Acc: 0.9868, AUC: 0.9976
#    Test  Acc: 0.9649, AUC: 0.9960
#    Gap:       Acc: 0.0219, AUC: 0.0016
#    CV Accuracy: 0.9714 (+/- 0.0112)
#    ✓ Logistic Regression appears well-generalized

# SVM:
#    Train Acc: 0.9868, AUC: 0.9976
#    Test  Acc: 0.9737, AUC: 0.9947
#    Gap:       Acc: 0.0131, AUC: 0.0029
#    CV Accuracy: 0.9758 (+/- 0.0128)
#    ✓ SVM appears well-generalized

# ======================================================================
# BEST MODEL: Gradient Boosting
# Test Accuracy: 0.9649
# Test AUC Score: 0.9964
# Train-Test Gap: Acc=0.0351, AUC=0.0036
# ======================================================================

# 6. Detailed evaluation of best model...

# Classification Report:
#               precision    recall  f1-score   support

#       Benign       0.95      1.00      0.97        72
#    Malignant       1.00      0.90      0.95        42

#     accuracy                           0.96       114
#    macro avg       0.97      0.95      0.96       114
# weighted avg       0.97      0.96      0.96       114


# Confusion Matrix:
#                 Predicted
#               Benign  Malignant
# Actual Benign     72        0
#        Malignant   4       38

# Top 10 Most Important Features:
#    1. perimeter_worst: 0.4265
#    2. radius_worst: 0.2277
#    3. concave points_worst: 0.1100
#    4. concave points_mean: 0.0965
#    5. texture_mean: 0.0306
#    6. concavity_worst: 0.0183
#    7. area_worst: 0.0171
#    8. texture_worst: 0.0164
#    9. smoothness_se: 0.0095
#    10. compactness_se: 0.0093

# 7. Saving model...
#    ✓ Model saved to models/breast_cancer/model.pkl
#    ✓ Scaler saved to models/breast_cancer/scaler.pkl
#    ✓ Features saved to models/breast_cancer/features.txt

# ======================================================================
# TRAINING COMPLETE!
# ======================================================================

# Model Performance Summary:
#   - Algorithm: Gradient Boosting
#   - Test Accuracy: 96.49%
#   - Test AUC Score: 0.9964
#   - Train-Test Gap: 0.0351
#   - Features: 30
#   - Training samples: 455
#   - Test samples: 114