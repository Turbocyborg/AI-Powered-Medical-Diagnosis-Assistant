"""
Chronic Kidney Disease (CKD) Model Training Script
Train ML models for kidney disease prediction
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

def load_and_preprocess_data(filepath):
    """Load and preprocess the CKD dataset"""
    print("Loading Chronic Kidney Disease dataset...")
    
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape}")
    
    # Remove non-feature columns
    df = df.drop(['PatientID', 'DoctorInCharge'], axis=1)
    print("Removed PatientID and DoctorInCharge columns")
    
    print(f"CKD prevalence: {df['Diagnosis'].mean():.3f}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    return df

def train_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple ML models"""
    print("\n" + "="*50)
    print("MODEL TRAINING AND EVALUATION")
    print("="*50)
    
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42, max_iter=2000, C=0.1, class_weight='balanced', solver='saga'
        ),
        'Random Forest': RandomForestClassifier(
            random_state=42, n_estimators=300, max_depth=25, min_samples_split=5,
            min_samples_leaf=2, max_features='sqrt', class_weight='balanced_subsample', n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            random_state=42, n_estimators=300, learning_rate=0.05, max_depth=7,
            min_samples_split=5, min_samples_leaf=2, subsample=0.8, max_features='sqrt'
        )
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(
            random_state=42, n_estimators=300, learning_rate=0.05, max_depth=8,
            min_child_weight=2, subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            eval_metric='logloss', early_stopping_rounds=20
        )
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        if name == 'XGBoost' and XGBOOST_AVAILABLE:
            X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {'model': model, 'accuracy': accuracy, 'auc_score': auc_score}
        
        print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
        print(classification_report(y_test, y_pred))
        
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:\n{cm}")
    
    return results

def select_best_model(results):
    """Select best model based on AUC score"""
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    
    for name, result in results.items():
        print(f"{name:25s} - Accuracy: {result['accuracy']:.4f}, AUC: {result['auc_score']:.4f}")
    
    best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBEST MODEL: {best_model_name}")
    print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
    print(f"AUC Score: {results[best_model_name]['auc_score']:.4f}")
    
    return best_model, best_model_name

def save_model_and_scaler(model, scaler, output_dir='models/kidney_disease'):
    """Save trained model and scaler"""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\nModel saved to: {output_dir}/model.pkl")
    print(f"Scaler saved to: {output_dir}/scaler.pkl")

def main():
    print("="*50)
    print("CHRONIC KIDNEY DISEASE MODEL TRAINING")
    print("="*50)
    
    df = load_and_preprocess_data('data/CKD.csv')
    
    # Prepare features and target
    feature_columns = [col for col in df.columns if col != 'Diagnosis']
    X = df[feature_columns]
    y = df['Diagnosis']
    
    print(f"\nFeatures: {len(feature_columns)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Handle class imbalance
    print("\nApplying SMOTE for class balancing...")
    print(f"Original: {np.bincount(y_train)}")
    
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"Balanced: {np.bincount(y_train_balanced)}")
    
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
    
    # Save feature names
    with open('models/kidney_disease/features.txt', 'w') as f:
        f.write('\n'.join(feature_columns))
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED!")
    print("="*50)

if __name__ == "__main__":
    main()
