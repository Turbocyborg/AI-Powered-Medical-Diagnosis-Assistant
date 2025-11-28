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

# User-friendly features that anyone can answer
SIMPLIFIED_FEATURES = [
    # Basic Info (5)
    'Age', 'Gender', 'BMI',
    'FamilyHistoryKidneyDisease', 'FamilyHistoryHypertension',
    
    # Symptoms (6)
    'Edema', 'FatigueLevels', 'Itching', 'MuscleCramps',
    'NauseaVomiting', 'UrinaryTractInfections',
    
    # Health Conditions (4)
    'SystolicBP', 'DiastolicBP',
    'FamilyHistoryDiabetes', 'PreviousAcuteKidneyInjury',
    
    # Lifestyle (5)
    'Smoking', 'PhysicalActivity', 'DietQuality',
    'SleepQuality', 'AlcoholConsumption',
    
    # Optional Lab Results (if known) - will use median if unknown
    'SerumCreatinine', 'GFR', 'FastingBloodSugar'
]

def load_and_preprocess_data(filepath):
    """Load and preprocess with simplified features"""
    print("Loading CKD dataset with simplified features...")
    
    df = pd.read_csv(filepath)
    df = df.drop(['PatientID', 'DoctorInCharge'], axis=1)
    
    # Select only simplified features + target
    df_simplified = df[SIMPLIFIED_FEATURES + ['Diagnosis']]
    
    print(f"Dataset: {df_simplified.shape}")
    print(f"Features: {len(SIMPLIFIED_FEATURES)} (down from 51)")
    print(f"CKD prevalence: {df_simplified['Diagnosis'].mean():.3f}")
    
    return df_simplified

def train_models(X_train, X_test, y_train, y_test):
    """Train models"""
    print("\n" + "="*50)
    print("MODEL TRAINING - SIMPLIFIED VERSION")
    print("="*50)
    
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42, max_iter=2000, C=0.5, class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            random_state=42, n_estimators=300, max_depth=20,
            min_samples_split=10, min_samples_leaf=5,
            class_weight='balanced_subsample', n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            random_state=42, n_estimators=300, learning_rate=0.05, 
            max_depth=6, min_samples_split=10, min_samples_leaf=5
        )
    }
    
    if XGBOOST_AVAILABLE:
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = np.bincount(y_train)[0] / np.bincount(y_train)[1]
        models['XGBoost'] = XGBClassifier(
            random_state=42, n_estimators=300, learning_rate=0.05,
            max_depth=7, min_child_weight=5,
            scale_pos_weight=scale_pos_weight,  # Handle imbalance
            n_jobs=-1, eval_metric='logloss'
        )
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {'model': model, 'accuracy': accuracy, 'auc_score': auc_score}
        
        print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
        print(classification_report(y_test, y_pred))
    
    return results

def main():
    print("="*50)
    print("SIMPLIFIED KIDNEY DISEASE MODEL TRAINING")
    print("="*50)
    
    df = load_and_preprocess_data('data/CKD.csv')
    
    X = df[SIMPLIFIED_FEATURES]
    y = df['Diagnosis']
    
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
    print(f"Original ratio: {np.bincount(y_train)[1]/np.bincount(y_train)[0]:.2f}:1 (CKD:Healthy)")
    
    # Use sampling_strategy=0.3 to create 30% minority class
    # This is more realistic than 1:1 for such extreme imbalance
    smote = SMOTE(random_state=42, k_neighbors=3, sampling_strategy=0.3)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"Balanced distribution: {np.bincount(y_train_balanced)}")
    print(f"New ratio: {np.bincount(y_train_balanced)[1]/np.bincount(y_train_balanced)[0]:.2f}:1 (CKD:Healthy)")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    results = train_models(X_train_scaled, X_test_scaled, y_train_balanced, y_test)
    
    # Select best
    best_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
    best_model = results[best_name]['model']
    
    print("\n" + "="*50)
    print(f"BEST MODEL: {best_name}")
    print(f"Accuracy: {results[best_name]['accuracy']:.4f}")
    print(f"AUC: {results[best_name]['auc_score']:.4f}")
    print("="*50)
    
    # Save
    os.makedirs('models/kidney_disease', exist_ok=True)
    with open('models/kidney_disease/model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('models/kidney_disease/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/kidney_disease/features.txt', 'w') as f:
        f.write('\n'.join(SIMPLIFIED_FEATURES))
    
    print("\nModel saved!")
    print(f"Features reduced: 51 → {len(SIMPLIFIED_FEATURES)}")
    print("User-friendly form ready!")

if __name__ == "__main__":
    main()
