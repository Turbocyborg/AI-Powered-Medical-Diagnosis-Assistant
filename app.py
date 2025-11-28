"""
Multi-Disease Prediction System
Main Flask Application
"""

import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'

# Disease configurations
DISEASES = {
    'diabetes': {
        'name': 'Diabetes',
        'model_path': 'models/diabetes/model.pkl',
        'scaler_path': 'models/diabetes/scaler.pkl',
        'features': [
            'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
            'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
            'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
            'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
        ]
    },
    'heart_disease': {
        'name': 'Heart Disease',
        'model_path': 'models/heart_disease/model.pkl',
        'scaler_path': 'models/heart_disease/scaler.pkl',
        'features': [
            'age', 'gender', 'chestpain', 'restingBP', 'serumcholestrol',
            'fastingbloodsugar', 'restingrelectro', 'maxheartrate',
            'exerciseangia', 'oldpeak', 'slope', 'noofmajorvessels'
        ]
    },
    'kidney_disease': {
        'name': 'Kidney Disease',
        'model_path': 'models/kidney_disease/model.pkl',
        'scaler_path': 'models/kidney_disease/scaler.pkl',
        'features': [
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
            # Optional Lab Results (3)
            'SerumCreatinine', 'GFR', 'FastingBloodSugar'
        ]
    }
}

# Load models
models = {}
scalers = {}

def load_disease_models():
    """Load all available disease models"""
    for disease_id, config in DISEASES.items():
        try:
            if os.path.exists(config['model_path']) and os.path.exists(config['scaler_path']):
                with open(config['model_path'], 'rb') as f:
                    models[disease_id] = pickle.load(f)
                with open(config['scaler_path'], 'rb') as f:
                    scalers[disease_id] = pickle.load(f)
                logger.info(f"Loaded {config['name']} model successfully")
            else:
                logger.warning(f"{config['name']} model not found. Train the model first.")
        except Exception as e:
            logger.error(f"Error loading {config['name']} model: {e}")

load_disease_models()

@app.route('/')
def home():
    """Main landing page with disease selection"""
    return render_template('index.html', diseases=DISEASES)

@app.route('/disease/<disease_id>')
def disease_page(disease_id):
    """Individual disease prediction page"""
    if disease_id not in DISEASES:
        return "Disease not found", 404
    
    disease_config = DISEASES[disease_id]
    model_available = disease_id in models
    
    return render_template(
        f'diseases/{disease_id}.html',
        disease=disease_config,
        model_available=model_available
    )

@app.route('/predict/<disease_id>', methods=['POST'])
def predict(disease_id):
    """Make prediction for specific disease"""
    try:
        if disease_id not in DISEASES:
            return jsonify({'error': 'Disease not found', 'success': False}), 404
        
        if disease_id not in models or disease_id not in scalers:
            return jsonify({
                'error': f'{DISEASES[disease_id]["name"]} model not loaded. Please train the model first.',
                'success': False
            }), 503
        
        data = request.get_json() if request.is_json else request.form.to_dict()
        
        # Extract features
        features = []
        missing_features = []
        
        for feature_name in DISEASES[disease_id]['features']:
            if feature_name in data:
                try:
                    features.append(float(data[feature_name]))
                except (ValueError, TypeError):
                    return jsonify({
                        'error': f'Invalid value for {feature_name}',
                        'success': False
                    }), 400
            else:
                missing_features.append(feature_name)
        
        if missing_features:
            return jsonify({
                'error': f'Missing features: {missing_features}',
                'success': False
            }), 400
        
        # Make prediction
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scalers[disease_id].transform(features_array)
        
        prediction = models[disease_id].predict(features_scaled)[0]
        prediction_proba = models[disease_id].predict_proba(features_scaled)[0]
        
        probability = prediction_proba[1] * 100
        
        # Apply domain knowledge adjustments for diabetes
        if disease_id == 'diabetes':
            # Smoking should increase risk (add 2-5% based on other factors)
            if float(data.get('Smoker', 0)) == 1:
                bmi = float(data.get('BMI', 25))
                high_bp = float(data.get('HighBP', 0))
                # Increase risk more if other risk factors present
                adjustment = 3 if (bmi > 30 or high_bp == 1) else 2
                probability = min(probability + adjustment, 99.9)
            
            # Heavy alcohol should increase risk
            if float(data.get('HvyAlcoholConsump', 0)) == 1:
                probability = min(probability + 2, 99.9)
        
        # Determine risk level
        if probability < 30:
            risk_level, risk_color = "Low", "green"
        elif probability < 70:
            risk_level, risk_color = "Moderate", "orange"
        else:
            risk_level, risk_color = "High", "red"
        
        result = {
            'success': True,
            'disease': DISEASES[disease_id]['name'],
            'prediction': int(prediction),
            'prediction_text': f"{DISEASES[disease_id]['name']} Risk Detected" if prediction == 1 else f"No {DISEASES[disease_id]['name']} Risk Detected",
            'probability': round(probability, 2),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'recommendations': get_recommendations(disease_id, prediction, data)
        }
        
        logger.info(f"{disease_id} prediction: {result['prediction_text']} ({probability:.2f}%)")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error for {disease_id}: {e}")
        return jsonify({
            'error': 'An error occurred during prediction',
            'success': False
        }), 500

def get_recommendations(disease_id, prediction, user_data):
    """Generate personalized recommendations"""
    recommendations = []
    
    if disease_id == 'kidney_disease':
        if prediction == 1:
            recommendations.extend([
                "Consult a nephrologist immediately for proper diagnosis",
                "Get comprehensive kidney function tests (GFR, creatinine)",
                "Monitor blood pressure and blood sugar regularly"
            ])
        
        # GFR-based recommendations
        try:
            gfr = float(user_data.get('GFR', 90))
            if gfr < 60:
                recommendations.append("Your GFR indicates reduced kidney function - seek medical attention")
            elif gfr < 90:
                recommendations.append("Mildly reduced kidney function - monitor regularly")
        except:
            pass
        
        # Creatinine recommendations
        try:
            creatinine = float(user_data.get('SerumCreatinine', 1.0))
            if creatinine > 1.3:
                recommendations.append("Elevated creatinine levels - consult a doctor")
        except:
            pass
        
        # Blood pressure
        try:
            systolic = float(user_data.get('SystolicBP', 120))
            if systolic > 140:
                recommendations.append("High blood pressure can damage kidneys - manage BP")
        except:
            pass
        
        if prediction == 0:
            recommendations.extend([
                "Maintain healthy blood pressure and blood sugar",
                "Stay hydrated and limit sodium intake",
                "Avoid excessive use of NSAIDs",
                "Get regular kidney function tests"
            ])
    
    elif disease_id == 'heart_disease':
        if prediction == 1:
            recommendations.extend([
                "Consult a cardiologist immediately for proper diagnosis",
                "Consider getting an ECG and stress test",
                "Monitor blood pressure and cholesterol regularly"
            ])
        
        # Chest pain recommendations
        chest_pain = int(user_data.get('chestpain', 0))
        if chest_pain in [0, 1]:  # Typical or atypical angina
            recommendations.append("Seek immediate medical attention for chest pain")
        
        # Blood pressure
        try:
            bp = float(user_data.get('restingBP', 120))
            if bp > 140:
                recommendations.append("Your blood pressure is high - consult a doctor")
            elif bp > 130:
                recommendations.append("Monitor your blood pressure regularly")
        except:
            pass
        
        # Cholesterol
        try:
            chol = float(user_data.get('serumcholestrol', 200))
            if chol > 240:
                recommendations.append("High cholesterol - consider dietary changes and medication")
            elif chol > 200:
                recommendations.append("Borderline high cholesterol - improve diet and exercise")
        except:
            pass
        
        # Exercise induced angina
        if user_data.get('exerciseangia') == '1':
            recommendations.append("Exercise-induced chest pain requires medical evaluation")
        
        # General heart health
        if prediction == 0:
            recommendations.extend([
                "Maintain a heart-healthy diet low in saturated fats",
                "Exercise regularly (150 minutes per week)",
                "Avoid smoking and limit alcohol consumption",
                "Get regular cardiovascular check-ups"
            ])
    
    elif disease_id == 'diabetes':
        if prediction == 1:
            recommendations.extend([
                "Consult with a healthcare professional for proper diagnosis",
                "Consider getting an HbA1c test",
                "Monitor blood glucose levels regularly"
            ])
        
        try:
            bmi = float(user_data.get('BMI', 0))
            if bmi > 25:
                recommendations.append("Consider a weight management program")
            if bmi > 30:
                recommendations.append("Consult a nutritionist for a personalized diet plan")
        except:
            pass
        
        if user_data.get('PhysActivity') == '0':
            recommendations.append("Incorporate at least 150 minutes of moderate exercise per week")
        
        if user_data.get('Smoker') == '1':
            recommendations.append("Consider smoking cessation programs")
        
        if user_data.get('Fruits') == '0' or user_data.get('Veggies') == '0':
            recommendations.append("Increase daily intake of fruits and vegetables")
        
        if prediction == 0:
            recommendations.extend([
                "Maintain current healthy lifestyle habits",
                "Get regular health check-ups"
            ])
    
    return recommendations

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {disease: disease in models for disease in DISEASES}
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
