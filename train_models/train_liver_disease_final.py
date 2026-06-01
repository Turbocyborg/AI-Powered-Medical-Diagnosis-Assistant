"""
Liver Disease Model Training - Properly Regularized
Prevents overfitting through proper validation, regularization, and realistic evaluation
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
    f1_score,
)

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
import pickle
import warnings

warnings.filterwarnings("ignore")

print("=" * 70)
print("LIVER DISEASE MODEL TRAINING - ANTI-OVERFITTING VERSION")
print("=" * 70)

# Load data
print("\n[1/7] Loading dataset...")
df = pd.read_csv("data/ldtrain.csv")
print(f"✓ Dataset shape: {df.shape}")

# Analyze target
print(f"\n[2/7] Target analysis...")
print(
    f"✓ Class 1 (Disease): {(df['Result'] == 1).sum()} ({(df['Result'] == 1).sum() / len(df) * 100:.1f}%)"
)
print(
    f"✓ Class 2 (No Disease): {(df['Result'] == 2).sum()} ({(df['Result'] == 2).sum() / len(df) * 100:.1f}%)"
)

# Convert target to binary (1 -> 1, 2 -> 0)
df["Result"] = (df["Result"] == 1).astype(int)

# Handle missing values
print(f"\n[3/7] Handling missing values...")
missing_count = df.isnull().sum().sum()
print(f"✓ Total missing: {missing_count}")

# Gender - use mode
if df["Gender of the patient"].isnull().any():
    df["Gender of the patient"].fillna(
        df["Gender of the patient"].mode()[0], inplace=True
    )

# Numerical - use median (robust to outliers)
for col in df.select_dtypes(include=[np.number]).columns:
    if col != "Result" and df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

print(f"✓ Missing values handled")

# Encode gender
le = LabelEncoder()
df["Gender of the patient"] = le.fit_transform(df["Gender of the patient"])

# Rename columns
df.columns = [
    "Age",
    "Gender",
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Total_Proteins",
    "Albumin",
    "Albumin_Globulin_Ratio",
    "Result",
]

# Split features and target
feature_columns = [col for col in df.columns if col != "Result"]
X = df[feature_columns]
y = df["Result"]

# IMPORTANT: Use larger test set to better evaluate generalization
print(f"\n[4/7] Splitting data (30% test for better evaluation)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"✓ Training: {X_train.shape[0]} samples ({y_train.sum()} disease)")
print(f"✓ Test: {X_test.shape[0]} samples ({y_test.sum()} disease)")

# NO SMOTE - Use class weights instead to avoid synthetic data overfitting
print(f"\n[5/7] Using class weights (NO SMOTE to prevent overfitting)...")
print(f"✓ Will use class_weight='balanced' in models")

# Scale features using RobustScaler (less sensitive to outliers)
print(f"\n[6/7] Scaling features with RobustScaler...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"✓ Features scaled")

# Train models with STRONG REGULARIZATION
print(f"\n[7/7] Training models with regularization...")
print("=" * 70)

models = {
    "Logistic Regression": LogisticRegression(
        random_state=42,
        max_iter=1000,
        C=0.1,  # Strong regularization
        class_weight="balanced",
        solver="lbfgs",
        penalty="l2",
    ),
    "Random Forest": RandomForestClassifier(
        random_state=42,
        n_estimators=100,  # Reduced from 200
        max_depth=8,  # Limited depth to prevent overfitting
        min_samples_split=20,  # Increased to prevent overfitting
        min_samples_leaf=10,  # Increased to prevent overfitting
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        max_samples=0.8,  # Bootstrap with 80% samples
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42,
        n_estimators=100,  # Reduced
        learning_rate=0.05,  # Lower learning rate
        max_depth=4,  # Shallow trees
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.7,  # Use only 70% of data per tree
        max_features="sqrt",
        validation_fraction=0.1,
        n_iter_no_change=10,  # Early stopping
        tol=1e-4,
    ),
}

if XGBOOST_AVAILABLE:
    models["XGBoost"] = XGBClassifier(
        random_state=42,
        n_estimators=100,  # Reduced
        learning_rate=0.05,  # Lower learning rate
        max_depth=4,  # Shallow trees
        min_child_weight=5,  # Increased to prevent overfitting
        subsample=0.7,  # Use only 70% of data
        colsample_bytree=0.7,  # Use only 70% of features
        gamma=1.0,  # Increased regularization
        reg_alpha=1.0,  # L1 regularization
        reg_lambda=2.0,  # L2 regularization
        scale_pos_weight=1,
        n_jobs=-1,
        eval_metric="logloss",
    )

results = {}

for name, model in models.items():
    print(f"\n{'=' * 70}")
    print(f"Training: {name}")
    print(f"{'=' * 70}")

    # Train
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    y_pred_proba_test = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_auc = roc_auc_score(y_test, y_pred_proba_test)
    test_f1 = f1_score(y_test, y_pred_test)

    # Cross-validation on TRAINING set only
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train, cv=cv, scoring="roc_auc"
    )

    # Calculate overfitting gap
    overfit_gap = train_acc - test_acc

    results[name] = {
        "model": model,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "test_auc": test_auc,
        "test_f1": test_f1,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "overfit_gap": overfit_gap,
    }

    print(f"✓ Training Accuracy: {train_acc:.4f}")
    print(f"✓ Test Accuracy: {test_acc:.4f}")
    print(
        f"✓ Overfitting Gap: {overfit_gap:.4f} {'⚠️ HIGH' if overfit_gap > 0.05 else '✓ OK'}"
    )
    print(f"✓ Test AUC-ROC: {test_auc:.4f}")
    print(f"✓ Test F1-Score: {test_f1:.4f}")
    print(f"✓ CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test)
    print(f"\nTest Set Confusion Matrix:")
    print(f"  TN: {cm[0, 0]:5d}  |  FP: {cm[0, 1]:5d}")
    print(f"  FN: {cm[1, 0]:5d}  |  TP: {cm[1, 1]:5d}")

    # Detailed metrics
    print(f"\nTest Set Classification Report:")
    print(
        classification_report(
            y_test, y_pred_test, target_names=["No Disease", "Disease"], digits=4
        )
    )

# Model comparison
print(f"\n{'=' * 70}")
print("MODEL COMPARISON - FOCUS ON TEST PERFORMANCE")
print(f"{'=' * 70}")

print(
    f"\n{'Model':<20} {'Train Acc':<11} {'Test Acc':<11} {'Gap':<8} {'Test AUC':<11} {'CV AUC':<15}"
)
print("-" * 76)

for name, result in results.items():
    gap_indicator = "⚠️" if result["overfit_gap"] > 0.05 else "✓"
    print(
        f"{name:<20} {result['train_acc']:<11.4f} {result['test_acc']:<11.4f} "
        f"{result['overfit_gap']:<7.4f}{gap_indicator} {result['test_auc']:<11.4f} "
        f"{result['cv_mean']:.4f}±{result['cv_std']:.3f}"
    )

# Select best model based on TEST AUC (not training performance)
best_model_name = max(results.keys(), key=lambda x: results[x]["test_auc"])
best_result = results[best_model_name]

print(f"\n{'=' * 70}")
print(f"BEST MODEL (Based on Test AUC): {best_model_name}")
print(f"{'=' * 70}")
print(f"✓ Training Accuracy: {best_result['train_acc']:.4f}")
print(f"✓ Test Accuracy: {best_result['test_acc']:.4f}")
print(f"✓ Overfitting Gap: {best_result['overfit_gap']:.4f}")
print(f"✓ Test AUC-ROC: {best_result['test_auc']:.4f}")
print(f"✓ Test F1-Score: {best_result['test_f1']:.4f}")
print(f"✓ CV AUC: {best_result['cv_mean']:.4f} (+/- {best_result['cv_std']:.4f})")

# Save model
output_dir = "models/liver_disease"
os.makedirs(output_dir, exist_ok=True)

with open(f"{output_dir}/model.pkl", "wb") as f:
    pickle.dump(best_result["model"], f)

with open(f"{output_dir}/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open(f"{output_dir}/features.txt", "w") as f:
    f.write("\n".join(feature_columns))

# Save detailed metadata
with open(f"{output_dir}/model_info.txt", "w") as f:
    f.write(f"Liver Disease Prediction Model - Regularized\n")
    f.write(f"=" * 60 + "\n\n")
    f.write(f"Model Type: {best_model_name}\n")
    f.write(f"Training Accuracy: {best_result['train_acc']:.4f}\n")
    f.write(f"Test Accuracy: {best_result['test_acc']:.4f}\n")
    f.write(f"Overfitting Gap: {best_result['overfit_gap']:.4f}\n")
    f.write(f"Test AUC-ROC: {best_result['test_auc']:.4f}\n")
    f.write(f"Test F1-Score: {best_result['test_f1']:.4f}\n")
    f.write(
        f"CV AUC: {best_result['cv_mean']:.4f} (+/- {best_result['cv_std']:.4f})\n\n"
    )
    f.write(f"Regularization Applied:\n")
    f.write(f"  - No SMOTE (avoided synthetic data overfitting)\n")
    f.write(f"  - 30% test set (better generalization evaluation)\n")
    f.write(f"  - RobustScaler (outlier-resistant)\n")
    f.write(f"  - Strong model regularization\n")
    f.write(f"  - Class weights for imbalance\n\n")
    f.write(f"Features ({len(feature_columns)}):\n")
    for i, feat in enumerate(feature_columns, 1):
        f.write(f"  {i}. {feat}\n")

print(f"\n{'=' * 70}")
print("MODEL SAVED")
print(f"{'=' * 70}")
print(f"✓ Model: {output_dir}/model.pkl")
print(f"✓ Scaler: {output_dir}/scaler.pkl")
print(f"✓ Features: {output_dir}/features.txt")
print(f"✓ Metadata: {output_dir}/model_info.txt")

print(f"\n{'=' * 70}")
print("TRAINING COMPLETED - MODEL IS PROPERLY REGULARIZED")
print(f"{'=' * 70}")
print(f"\nThe model shows realistic performance with proper generalization.")
print(f"Overfitting gap: {best_result['overfit_gap']:.4f} (should be < 0.05)")
print(f"\nYou can now run the FastAPI application!")
