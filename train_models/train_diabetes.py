"""
Diabetes Model Training Script
Train ML models for diabetes prediction using BRFSS dataset
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")
import pickle
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath):
    """Load and preprocess the BRFSS diabetes dataset"""
    print("Loading BRFSS diabetes dataset...")
    
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded successfully: {df.shape}")
        
        # Handle different target column names
        if 'Diabetes_012' in df.columns:
            # Convert 3-class to binary (0=no diabetes, 1/2=diabetes/prediabetes)
            df['Diabetes_binary'] = (df['Diabetes_012'] > 0).astype(int)
            df = df.drop('Diabetes_012', axis=1)
            print("Converted Diabetes_012 to Diabetes_binary")
        elif 'Diabetes_binary' not in df.columns:
            raise ValueError("Dataset must have 'Diabetes_binary' or 'Diabetes_012' column")
            
    except FileNotFoundError:
        print(f"Dataset not found at {filepath}")
        print("Creating synthetic dataset for demonstration...")
        df = create_synthetic_dataset()
    
    print(f"Diabetes prevalence: {df['Diabetes_binary'].mean():.3f}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    return df

def create_synthetic_dataset():
    """Create synthetic BRFSS-like dataset for demonstration"""
    np.random.seed(42)
    n_samples = 50000
    
    data = {
        'HighBP': np.random.binomial(1, 0.45, n_samples),
        'HighChol': np.random.binomial(1, 0.40, n_samples),
        'CholCheck': np.random.binomial(1, 0.85, n_samples),
        'BMI': np.random.normal(28, 6, n_samples).clip(15, 50),
        'Smoker': np.random.binomial(1, 0.35, n_samples),
        'Stroke': np.random.binomial(1, 0.08, n_samples),
        'HeartDiseaseorAttack': np.random.binomial(1, 0.12, n_samples),
        'PhysActivity': np.random.binomial(1, 0.75, n_samples),
        'Fruits': np.random.binomial(1, 0.65, n_samples),
        'Veggies': np.random.binomial(1, 0.70, n_samples),
        'HvyAlcoholConsump': np.random.binomial(1, 0.15, n_samples),
        'AnyHealthcare': np.random.binomial(1, 0.90, n_samples),
        'NoDocbcCost': np.random.binomial(1, 0.20, n_samples),
        'GenHlth': np.random.randint(1, 6, n_samples),
        'MentHlth': np.random.poisson(5, n_samples).clip(0, 30),
        'PhysHlth': np.random.poisson(4, n_samples).clip(0, 30),
        'DiffWalk': np.random.binomial(1, 0.25, n_samples),
        'Sex': np.random.binomial(1, 0.48, n_samples),
        'Age': np.random.randint(1, 14, n_samples),
        'Education': np.random.randint(1, 7, n_samples),
        'Income': np.random.randint(1, 9, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target with realistic correlations
    risk_score = (
        0.3 * df['HighBP'] + 
        0.25 * df['HighChol'] + 
        0.4 * (df['BMI'] > 30).astype(int) + 
        0.2 * df['Smoker'] + 
        0.3 * df['Stroke'] + 
        0.25 * df['HeartDiseaseorAttack'] + 
        -0.2 * df['PhysActivity'] + 
        0.15 * (df['Age'] > 8).astype(int) + 
        0.1 * (df['GenHlth'] > 3).astype(int) +
        0.05 * np.random.normal(0, 1, n_samples)
    )
    
    diabetes_probability = 1 / (1 + np.exp(-risk_score))
    df['Diabetes_binary'] = np.random.binomial(1, diabetes_probability, n_samples)
    
    return df

def train_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple ML models"""
    print("\n" + "="*50)
    print("MODEL TRAINING AND EVALUATION")
    print("="*50)
    
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42, 
            max_iter=2000,
            C=0.1,  # Regularization
            class_weight='balanced',
            solver='saga'
        ),
        'Random Forest': RandomForestClassifier(
            random_state=42, 
            n_estimators=300,  # More trees
            max_depth=25,  # Deeper trees
            min_samples_split=5,  # More sensitive
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced_subsample',
            n_jobs=-1,
            bootstrap=True,
            oob_score=True
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            random_state=42,
            n_estimators=300,  # More estimators
            learning_rate=0.05,  # Lower learning rate
            max_depth=7,  # Deeper trees
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            max_features='sqrt',
            validation_fraction=0.1,
            n_iter_no_change=20,
            tol=1e-4
        )
    }
    
    # Add XGBoost if available (usually best performance)
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(
            random_state=42,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            colsample_bylevel=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=1,
            n_jobs=-1,
            eval_metric='logloss',
            early_stopping_rounds=20
        )
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Special handling for XGBoost with early stopping
        if name == 'XGBoost' and XGBOOST_AVAILABLE:
            # Split training data for validation
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
            )
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'auc_score': auc_score
        }
        
        print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
        print(classification_report(y_test, y_pred))
        
        # Show confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:\n{cm}")
        print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
    
    return results

def select_best_model(results):
    """Select best model based on AUC score and accuracy"""
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    
    for name, result in results.items():
        print(f"{name:25s} - Accuracy: {result['accuracy']:.4f}, AUC: {result['auc_score']:.4f}")
    
    # Select based on AUC score (better for imbalanced data)
    best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
    best_model = results[best_model_name]['model']
    best_auc = results[best_model_name]['auc_score']
    best_acc = results[best_model_name]['accuracy']
    
    print("\n" + "="*50)
    print(f"BEST MODEL: {best_model_name}")
    print(f"Accuracy: {best_acc:.4f}")
    print(f"AUC Score: {best_auc:.4f}")
    print("="*50)
    
    return best_model, best_model_name

def save_model_and_scaler(model, scaler, output_dir='models/diabetes'):
    """Save trained model and scaler"""
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, 'model.pkl')
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")

def main():
    """Main training pipeline"""
    print("="*50)
    print("DIABETES PREDICTION MODEL TRAINING")
    print("="*50)
    
    # Load data
    dataset_path = 'data/diabetes_data.csv'
    df = load_and_preprocess_data(dataset_path)
    
    # Prepare features and target
    feature_columns = [col for col in df.columns if col != 'Diabetes_binary']
    X = df[feature_columns]
    y = df['Diabetes_binary']
    
    print(f"\nFeatures: {len(feature_columns)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Handle class imbalance with SMOTE
    print("\nApplying SMOTE for class balancing...")
    print(f"Original class distribution: {np.bincount(y_train)}")
    print(f"Class ratio: {np.bincount(y_train)[0]/np.bincount(y_train)[1]:.2f}:1")
    
    # Use SMOTE with k_neighbors=5 for better synthetic samples
    smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy=0.8)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"Balanced class distribution: {np.bincount(y_train_balanced)}")
    print(f"New class ratio: {np.bincount(y_train_balanced)[0]/np.bincount(y_train_balanced)[1]:.2f}:1")
    
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
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"Best model: {best_model_name}")
    print("\nYou can now run the Flask application!")

if __name__ == "__main__":
    main()

# ==================================================
# DIABETES PREDICTION MODEL TRAINING
# ==================================================
# Loading BRFSS diabetes dataset...
# Dataset loaded successfully: (253680, 22)
# Converted Diabetes_012 to Diabetes_binary
# Diabetes prevalence: 0.158
# Missing values: 0

# Features: 21
# Train set: (202944, 21)
# Test set: (50736, 21)

# Applying SMOTE for class balancing...
# Original class distribution: [170962  31982]
# Class ratio: 5.35:1
# Balanced class distribution: [170962 136769]
# New class ratio: 1.25:1

# Scaling features...

# ==================================================
# MODEL TRAINING AND EVALUATION
# ==================================================

# Training Logistic Regression...
# Logistic Regression - Accuracy: 0.7298, AUC: 0.8153
#               precision    recall  f1-score   support

#            0       0.94      0.73      0.82     42741
#            1       0.34      0.75      0.47      7995

#     accuracy                           0.73     50736
#    macro avg       0.64      0.74      0.64     50736
# weighted avg       0.85      0.73      0.76     50736

# Confusion Matrix:
# [[30992 11749]
#  [ 1959  6036]]
# True Negatives: 30992, False Positives: 11749
# False Negatives: 1959, True Positives: 6036

# Training Random Forest...
# Random Forest - Accuracy: 0.8423, AUC: 0.8115
#               precision    recall  f1-score   support

#            0       0.88      0.94      0.91     42741
#            1       0.50      0.32      0.39      7995

#     accuracy                           0.84     50736
#    macro avg       0.69      0.63      0.65     50736
# weighted avg       0.82      0.84      0.83     50736

# Confusion Matrix:
# [[40185  2556]
#  [ 5443  2552]]
# True Negatives: 40185, False Positives: 2556
# False Negatives: 5443, True Positives: 2552

# Training Gradient Boosting...
# Gradient Boosting - Accuracy: 0.8501, AUC: 0.8209
#               precision    recall  f1-score   support

#            0       0.87      0.96      0.92     42741
#            1       0.55      0.26      0.35      7995

#     accuracy                           0.85     50736
#    macro avg       0.71      0.61      0.63     50736
# weighted avg       0.82      0.85      0.83     50736

# Confusion Matrix:
# [[41060  1681]
#  [ 5924  2071]]
# True Negatives: 41060, False Positives: 1681
# False Negatives: 5924, True Positives: 2071

# Training XGBoost...
# XGBoost - Accuracy: 0.8517, AUC: 0.8218
#               precision    recall  f1-score   support

#            0       0.87      0.97      0.92     42741
#            1       0.57      0.23      0.33      7995

#     accuracy                           0.85     50736
#    macro avg       0.72      0.60      0.62     50736
# weighted avg       0.82      0.85      0.82     50736

# Confusion Matrix:
# [[41375  1366]
#  [ 6160  1835]]
# True Negatives: 41375, False Positives: 1366
# False Negatives: 6160, True Positives: 1835

# ==================================================
# MODEL COMPARISON
# ==================================================
# Logistic Regression       - Accuracy: 0.7298, AUC: 0.8153
# Random Forest             - Accuracy: 0.8423, AUC: 0.8115
# Gradient Boosting         - Accuracy: 0.8501, AUC: 0.8209
# XGBoost                   - Accuracy: 0.8517, AUC: 0.8218

# ==================================================
# BEST MODEL: XGBoost
# Accuracy: 0.8517
# AUC Score: 0.8218
# ==================================================

# Model saved to: models/diabetes\model.pkl
# Scaler saved to: models/diabetes\scaler.pkl

# ==================================================
# TRAINING COMPLETED SUCCESSFULLY!
# ==================================================
# Best model: XGBoost

# You can now run the Flask application!