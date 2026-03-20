"""
Heart Disease Model Training Script
Train ML models for cardiovascular disease prediction
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
import pickle
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")


def load_and_preprocess_data(filepath):
    """Load and preprocess the cardiovascular disease dataset"""
    print("Loading Cardiovascular Disease dataset...")

    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded successfully: {df.shape}")
    except FileNotFoundError:
        print(f"Dataset not found at {filepath}")
        print("Please ensure Cardiovascular_Disease_Dataset.csv is in the data/ folder")
        return None

    # Remove patient ID (not a feature)
    if "patientid" in df.columns:
        df = df.drop("patientid", axis=1)
        print("Removed patientid column")

    print(f"Heart Disease prevalence: {df['target'].mean():.3f}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Features: {len(df.columns) - 1}")

    return df


def explore_data(df):
    """Perform exploratory data analysis"""
    print("\n" + "=" * 50)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 50)

    print(f"\nDataset Shape: {df.shape}")
    print(f"\nFeatures: {[col for col in df.columns if col != 'target']}")

    print(f"\nTarget Distribution:")
    print(df["target"].value_counts())
    print(
        f"Heart Disease: {(df['target'] == 1).sum()} ({(df['target'] == 1).sum() / len(df) * 100:.1f}%)"
    )
    print(
        f"No Disease: {(df['target'] == 0).sum()} ({(df['target'] == 0).sum() / len(df) * 100:.1f}%)"
    )

    # Feature correlations with target
    correlations = df.corr()["target"].sort_values(ascending=False)
    print(f"\nTop Features Correlated with Heart Disease:")
    print(correlations[1:8])

    return correlations


def train_models(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 50)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 50)

    # More regularized models to prevent overfitting
    models = {
        "Logistic Regression": LogisticRegression(
            random_state=42,
            max_iter=2000,
            C=1.0,  # Increased regularization
            class_weight="balanced",
            solver="saga",
        ),
        "Random Forest": RandomForestClassifier(
            random_state=42,
            n_estimators=100,
            max_depth=6,  # Reduced from 8
            min_samples_split=20,  # Increased from 10
            min_samples_leaf=10,  # Increased from 5
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            min_samples_split=20,  # Increased from 5
            min_samples_leaf=10,  # Increased from 2
            subsample=0.8,
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            random_state=42,
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            reg_alpha=2.0,  # Increased regularization
            reg_lambda=3.0,  # Increased regularization
            min_child_weight=5,
            n_jobs=-1,
            eval_metric="logloss",
        )

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # CROSS VALIDATION
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")

        print(f"{name} CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Train model
        model.fit(X_train, y_train)

        # Train predictions
        train_pred = model.predict(X_train)
        train_proba = model.predict_proba(X_train)[:, 1]
        train_acc = accuracy_score(y_train, train_pred)
        train_auc = roc_auc_score(y_train, train_proba)

        # Test predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        test_acc = accuracy_score(y_test, y_pred)
        test_auc = roc_auc_score(y_test, y_pred_proba)

        # Overfitting detection
        acc_gap = train_acc - test_acc
        auc_gap = train_auc - test_auc

        print(f"{name} Train Acc: {train_acc:.4f}, AUC: {train_auc:.4f}")
        print(f"{name} Test  Acc: {test_acc:.4f}, AUC: {test_auc:.4f}")
        print(f"{name} Gap:       Acc: {acc_gap:.4f}, AUC: {auc_gap:.4f}")

        if acc_gap > 0.10 or auc_gap > 0.10:
            print(f"⚠️  WARNING: {name} may be overfitting!")
        else:
            print(f"✓ {name} appears well-generalized")

        print("\nClassification Report:")
        print(
            classification_report(
                y_test, y_pred, target_names=["No Disease", "Heart Disease"]
            )
        )

        results[name] = {
            "model": model,
            "train_acc": train_acc,
            "train_auc": train_auc,
            "test_acc": test_acc,
            "test_auc": test_auc,
            "acc_gap": acc_gap,
            "auc_gap": auc_gap,
        }

    return results


def select_best_model(results):
    """Select best model based on test AUC and generalization"""
    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)

    for name, result in results.items():
        overfitting_status = (
            "⚠️ Overfitting"
            if (result["acc_gap"] > 0.10 or result["auc_gap"] > 0.10)
            else "✓ Good"
        )
        print(
            f"{name:25s} - Test Acc: {result['test_acc']:.4f}, Test AUC: {result['test_auc']:.4f}, Status: {overfitting_status}"
        )

    # Select based on test AUC (not train AUC!)
    best_model_name = max(results.keys(), key=lambda x: results[x]["test_auc"])
    best_model = results[best_model_name]["model"]
    best_result = results[best_model_name]

    print("\n" + "=" * 50)
    print(f"BEST MODEL: {best_model_name}")
    print(f"Test Accuracy: {best_result['test_acc']:.4f}")
    print(f"Test AUC Score: {best_result['test_auc']:.4f}")
    print(
        f"Train-Test Gap: Acc={best_result['acc_gap']:.4f}, AUC={best_result['auc_gap']:.4f}"
    )
    print("=" * 50)

    return best_model, best_model_name


def save_model_and_scaler(model, scaler, output_dir="models/heart_disease"):
    """Save trained model and scaler"""
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "model.pkl")
    scaler_path = os.path.join(output_dir, "scaler.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")


def main():
    """Main training pipeline"""
    print("=" * 50)
    print("HEART DISEASE PREDICTION MODEL TRAINING")
    print("=" * 50)

    # Load data and retain data
    dataset_path = "data/Cardiovascular_Disease_Dataset.csv"
    df = load_and_preprocess_data(dataset_path)

    if df is None:
        return

    # Explore data
    correlations = explore_data(df)

    # Prepare features and target
    feature_columns = [col for col in df.columns if col != "target"]
    X = df[feature_columns]
    y = df["target"]

    print(f"\nFeatures used: {feature_columns}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Handle class imbalance with SMOTE (more conservative)
    print("\nApplying SMOTE for class balancing...")
    print(f"Original class distribution: {np.bincount(y_train)}")
    print(f"Class ratio: {np.bincount(y_train)[0] / np.bincount(y_train)[1]:.2f}:1")

    # Use fewer neighbors and less aggressive sampling to reduce overfitting
    smote = SMOTE(random_state=42, k_neighbors=3, sampling_strategy=0.7)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    print(f"Balanced class distribution: {np.bincount(y_train_balanced)}")
    print(
        f"New class ratio: {np.bincount(y_train_balanced)[0] / np.bincount(y_train_balanced)[1]:.2f}:1"
    )

    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    results = train_models(X_train_scaled, X_test_scaled, y_train_balanced, y_test)

    # Select and save best model
    best_model, best_model_name = select_best_model(results)
    save_model_and_scaler(best_model, scaler)

    print("\n" + "=" * 50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print(f"Best model: {best_model_name}")
    print("\nYou can now add heart disease to the Flask application!")


if __name__ == "__main__":
    main()

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