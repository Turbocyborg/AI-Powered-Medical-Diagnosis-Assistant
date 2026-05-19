"""
Simplified Chronic Kidney Disease Model Training
Using only features that normal users can answer
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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

# User-friendly features that anyone can answer
SIMPLIFIED_FEATURES = [
    # Basic Info (5)
    "Age",
    "Gender",
    "BMI",
    "FamilyHistoryKidneyDisease",
    "FamilyHistoryHypertension",
    # Symptoms (6)
    "Edema",
    "FatigueLevels",
    "Itching",
    "MuscleCramps",
    "NauseaVomiting",
    "UrinaryTractInfections",
    # Health Conditions (4)
    "SystolicBP",
    "DiastolicBP",
    "FamilyHistoryDiabetes",
    "PreviousAcuteKidneyInjury",
    # Lifestyle (5)
    "Smoking",
    "PhysicalActivity",
    "DietQuality",
    "SleepQuality",
    "AlcoholConsumption",
    # Optional Lab Results (if known) - will use median if unknown
    "SerumCreatinine",
    "GFR",
    "FastingBloodSugar",
]


def load_and_preprocess_data(filepath):
    """Load and preprocess with simplified features"""
    print("Loading CKD dataset with simplified features...")

    df = pd.read_csv(filepath)
    df = df.drop(["PatientID", "DoctorInCharge"], axis=1)

    # Select only simplified features + target
    df_simplified = df[SIMPLIFIED_FEATURES + ["Diagnosis"]]

    print(f"Dataset: {df_simplified.shape}")
    print(f"Features: {len(SIMPLIFIED_FEATURES)} (down from 51)")
    print(f"CKD prevalence: {df_simplified['Diagnosis'].mean():.3f}")

    return df_simplified


def train_models(X_train, X_test, y_train, y_test):
    """Train models with overfitting detection"""
    print("\n" + "=" * 50)
    print("MODEL TRAINING - SIMPLIFIED VERSION")
    print("=" * 50)

    # More regularized models to prevent overfitting
    models = {
        "Logistic Regression": LogisticRegression(
            random_state=42, max_iter=2000, C=1.0, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            random_state=42,
            n_estimators=100,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42,
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=20,
            min_samples_leaf=10,
        ),
    }

    if XGBOOST_AVAILABLE:
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = np.bincount(y_train)[0] / np.bincount(y_train)[1]
        models["XGBoost"] = XGBClassifier(
            random_state=42,
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=5,
            reg_alpha=2.0,
            reg_lambda=3.0,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            eval_metric="logloss",
        )

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
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
        print(classification_report(y_test, y_pred, target_names=["No CKD", "CKD"]))

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


def main():
    print("=" * 50)
    print("SIMPLIFIED KIDNEY DISEASE MODEL TRAINING")
    print("=" * 50)

    # Load and retrain data
    df = load_and_preprocess_data("data/CKD.csv")

    X = df[SIMPLIFIED_FEATURES]
    y = df["Diagnosis"]

    print(f"\nSimplified Features ({len(SIMPLIFIED_FEATURES)}):")
    for i, feat in enumerate(SIMPLIFIED_FEATURES, 1):
        print(f"  {i}. {feat}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

    # SMOTE with better strategy for extreme imbalance
    print("\nApplying SMOTE with sampling_strategy...")
    print(f"Original distribution: {np.bincount(y_train)}")
    print(
        f"Original ratio: {np.bincount(y_train)[1] / np.bincount(y_train)[0]:.2f}:1 (CKD:Healthy)"
    )

    # Use sampling_strategy=0.3 to create 30% minority class
    # This is more realistic than 1:1 for such extreme imbalance
    smote = SMOTE(random_state=42, k_neighbors=3, sampling_strategy=0.3)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    print(f"Balanced distribution: {np.bincount(y_train_balanced)}")
    print(
        f"New ratio: {np.bincount(y_train_balanced)[1] / np.bincount(y_train_balanced)[0]:.2f}:1 (CKD:Healthy)"
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)

    # Train
    results = train_models(X_train_scaled, X_test_scaled, y_train_balanced, y_test)

    # Select best based on test AUC
    best_name = max(results.keys(), key=lambda x: results[x]["test_auc"])
    best_model = results[best_name]["model"]
    best_result = results[best_name]

    print("\n" + "=" * 50)
    print(f"BEST MODEL: {best_name}")
    print(f"Test Accuracy: {best_result['test_acc']:.4f}")
    print(f"Test AUC: {best_result['test_auc']:.4f}")
    print(
        f"Train-Test Gap: Acc={best_result['acc_gap']:.4f}, AUC={best_result['auc_gap']:.4f}"
    )

    if best_result["acc_gap"] > 0.10 or best_result["auc_gap"] > 0.10:
        print("⚠️  WARNING: Best model may be overfitting!")
    else:
        print("✓ Best model is well-generalized")

    print("=" * 50)

    # Save
    os.makedirs("models/kidney_disease", exist_ok=True)
    with open("models/kidney_disease/model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("models/kidney_disease/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/kidney_disease/features.txt", "w") as f:
        f.write("\n".join(SIMPLIFIED_FEATURES))

    print("\nModel saved!")
    print(f"Features reduced: 51 → {len(SIMPLIFIED_FEATURES)}")
    print("User-friendly form ready!")


if __name__ == "__main__":
    main()

# Training Logistic Regression...
# Logistic Regression Train Acc: 0.8283, AUC: 0.9115
# Logistic Regression Test  Acc: 0.7982, AUC: 0.7818
# Logistic Regression Gap:       Acc: 0.0301, AUC: 0.1297
# ⚠️  WARNING: Logistic Regression may be overfitting!

# Classification Report:
#               precision    recall  f1-score   support

#       No CKD       0.24      0.70      0.36        27
#          CKD       0.97      0.81      0.88       305

#     accuracy                           0.80       332
#    macro avg       0.61      0.76      0.62       332
# weighted avg       0.91      0.80      0.84       332


# Training Random Forest...
# Random Forest Train Acc: 0.9659, AUC: 0.9940
# Random Forest Test  Acc: 0.8795, AUC: 0.7741
# Random Forest Gap:       Acc: 0.0864, AUC: 0.2198
# ⚠️  WARNING: Random Forest may be overfitting!

# Classification Report:
#               precision    recall  f1-score   support

#       No CKD       0.29      0.33      0.31        27
#          CKD       0.94      0.93      0.93       305

#     accuracy                           0.88       332
#    macro avg       0.62      0.63      0.62       332
# weighted avg       0.89      0.88      0.88       332


# Training Gradient Boosting...
# Gradient Boosting Train Acc: 0.9804, AUC: 0.9984
# Gradient Boosting Test  Acc: 0.9006, AUC: 0.7526
# Gradient Boosting Gap:       Acc: 0.0798, AUC: 0.2457
# ⚠️  WARNING: Gradient Boosting may be overfitting!

# Classification Report:
#               precision    recall  f1-score   support

#       No CKD       0.33      0.22      0.27        27
#          CKD       0.93      0.96      0.95       305

#     accuracy                           0.90       332
#    macro avg       0.63      0.59      0.61       332
# weighted avg       0.88      0.90      0.89       332


# Training XGBoost...
# XGBoost Train Acc: 0.9343, AUC: 0.9885
# XGBoost Test  Acc: 0.8283, AUC: 0.7661
# XGBoost Gap:       Acc: 0.1060, AUC: 0.2224
# ⚠️  WARNING: XGBoost may be overfitting!

# Classification Report:
#               precision    recall  f1-score   support

#       No CKD       0.24      0.52      0.33        27
#          CKD       0.95      0.86      0.90       305

#     accuracy                           0.83       332
#    macro avg       0.60      0.69      0.62       332
# weighted avg       0.89      0.83      0.86       332


# ==================================================
# BEST MODEL: Logistic Regression
# Test Accuracy: 0.7982
# Test AUC: 0.7818
# Train-Test Gap: Acc=0.0301, AUC=0.1297
# ⚠️  WARNING: Best model may be overfitting!
# ==================================================