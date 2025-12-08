"""
Heart Disease Model Training Script
Train ML models for cardiovascular disease prediction
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import pickle
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

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
    if 'patientid' in df.columns:
        df = df.drop('patientid', axis=1)
        print("Removed patientid column")
    
    print(f"Heart Disease prevalence: {df['target'].mean():.3f}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Features: {len(df.columns) - 1}")
    
    return df

def explore_data(df):
    """Perform exploratory data analysis"""
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nFeatures: {[col for col in df.columns if col != 'target']}")
    
    print(f"\nTarget Distribution:")
    print(df['target'].value_counts())
    print(f"Heart Disease: {(df['target']==1).sum()} ({(df['target']==1).sum()/len(df)*100:.1f}%)")
    print(f"No Disease: {(df['target']==0).sum()} ({(df['target']==0).sum()/len(df)*100:.1f}%)")
    
    # Feature correlations with target
    correlations = df.corr()['target'].sort_values(ascending=False)
    print(f"\nTop Features Correlated with Heart Disease:")
    print(correlations[1:8])
    
    return correlations

def train_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple ML models"""
    print("\n" + "="*50)
    print("MODEL TRAINING AND EVALUATION")
    print("="*50)
    
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42,
            max_iter=2000,
            C=0.1,
            class_weight='balanced',
            solver='saga'
        ),
        'Random Forest': RandomForestClassifier(
            random_state=42,
            n_estimators=300,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced_subsample',
            n_jobs=-1,
            bootstrap=True,
            oob_score=True
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            random_state=42,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=7,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            max_features='sqrt',
            validation_fraction=0.1,
            n_iter_no_change=20,
            tol=1e-4
        )
    }
    
    # Add XGBoost if available
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
    
    # Select based on AUC score
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

def save_model_and_scaler(model, scaler, output_dir='models/heart_disease'):
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
    print("HEART DISEASE PREDICTION MODEL TRAINING")
    print("="*50)
    
    # Load data and retain data
    dataset_path = ("F:\\ML_PROJECT\AI-Powered-Medical-Diagnosis-Assistant\data\Cardiovascular_Disease_Dataset.csv")
    df = load_and_preprocess_data(dataset_path)
    
    if df is None:
        return
    
    # Explore data
    correlations = explore_data(df)
    
    # Prepare features and target
    feature_columns = [col for col in df.columns if col != 'target']
    X = df[feature_columns]
    y = df['target']
    
    print(f"\nFeatures used: {feature_columns}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Handle class imbalance with SMOTE
    print("\nApplying SMOTE for class balancing...")
    print(f"Original class distribution: {np.bincount(y_train)}")
    print(f"Class ratio: {np.bincount(y_train)[0]/np.bincount(y_train)[1]:.2f}:1")
    
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
    print("\nYou can now add heart disease to the Flask application!")

if __name__ == "__main__":
    main()
