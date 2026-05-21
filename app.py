
"""
Multi-Disease Prediction System
FastAPI Application
"""

import os
import pickle
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from typing import Dict, List
import logging

#auth imports
from jose import jwt, JWTError
from datetime import datetime, timedelta
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer
from fastapi import Depends
from pydantic import BaseModel


from database.diabetes_db import save_diabetes_prediction
from database.db import db



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# JWT AUTH CONFIGURATION
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="signin"
)
SECRET_KEY = "super-secret-key"

ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_HOURS = 24

def hash_password(password: str):

    return pwd_context.hash(password)


def verify_password(
    plain_password,
    hashed_password
):

    return pwd_context.verify(
        plain_password,
        hashed_password
    )

def create_access_token(data: dict):

    to_encode = data.copy()

    expire = datetime.utcnow() + timedelta(
        hours=ACCESS_TOKEN_EXPIRE_HOURS
    )

    to_encode.update({
        "exp": expire
    })

    encoded_jwt = jwt.encode(
        to_encode,
        SECRET_KEY,
        algorithm=ALGORITHM
    )

    return encoded_jwt




app = FastAPI(title="Multi-Disease Prediction System", version="1.0.0")


from database.db import connect_db, disconnect_db

@app.on_event("startup")
async def startup():
    await connect_db()


@app.on_event("shutdown")
async def shutdown():
    await disconnect_db()


# Templates
templates = Jinja2Templates(directory="templates")

# Disease configurations
DISEASES = {
    "diabetes": {
        "name": "Diabetes",
        "model_path": "models/diabetes/model.pkl",
        "scaler_path": "models/diabetes/scaler.pkl",
        "features": [
            "HighBP",
            "HighChol",
            "CholCheck",
            "BMI",
            "Smoker",
            "Stroke",
            "HeartDiseaseorAttack",
            "PhysActivity",
            "Fruits",
            "Veggies",
            "HvyAlcoholConsump",
            "AnyHealthcare",
            "NoDocbcCost",
            "GenHlth",
            "MentHlth",
            "PhysHlth",
            "DiffWalk",
            "Sex",
            "Age",
            "Education",
            "Income",
        ],
    },
    "heart_disease": {
        "name": "Heart Disease",
        "model_path": "models/heart_disease/model.pkl",
        "scaler_path": "models/heart_disease/scaler.pkl",
        "features": [
            "age",
            "gender",
            "chestpain",
            "restingBP",
            "serumcholestrol",
            "fastingbloodsugar",
            "restingrelectro",
            "maxheartrate",
            "exerciseangia",
            "oldpeak",
            "slope",
            "noofmajorvessels",
        ],
    },
    "kidney_disease": {
        "name": "Kidney Disease",
        "model_path": "models/kidney_disease/model.pkl",
        "scaler_path": "models/kidney_disease/scaler.pkl",
        "features": [
            "Age",
            "Gender",
            "BMI",
            "FamilyHistoryKidneyDisease",
            "FamilyHistoryHypertension",
            "Edema",
            "FatigueLevels",
            "Itching",
            "MuscleCramps",
            "NauseaVomiting",
            "UrinaryTractInfections",
            "SystolicBP",
            "DiastolicBP",
            "FamilyHistoryDiabetes",
            "PreviousAcuteKidneyInjury",
            "Smoking",
            "PhysicalActivity",
            "DietQuality",
            "SleepQuality",
            "AlcoholConsumption",
            "SerumCreatinine",
            "GFR",
            "FastingBloodSugar",
        ],
    },
    "breast_cancer": {
        "name": "Breast Cancer",
        "model_path": "models/breast_cancer/model.pkl",
        "scaler_path": "models/breast_cancer/scaler.pkl",
        "features": [
            "radius_mean",
            "texture_mean",
            "perimeter_mean",
            "area_mean",
            "smoothness_mean",
            "compactness_mean",
            "concavity_mean",
            "concave points_mean",
            "symmetry_mean",
            "fractal_dimension_mean",
            "radius_se",
            "texture_se",
            "perimeter_se",
            "area_se",
            "smoothness_se",
            "compactness_se",
            "concavity_se",
            "concave points_se",
            "symmetry_se",
            "fractal_dimension_se",
            "radius_worst",
            "texture_worst",
            "perimeter_worst",
            "area_worst",
            "smoothness_worst",
            "compactness_worst",
            "concavity_worst",
            "concave points_worst",
            "symmetry_worst",
            "fractal_dimension_worst",
        ],
    },
}

# Load models
models = {}
scalers = {}


def load_disease_models():
    """Load all available disease models"""
    for disease_id, config in DISEASES.items():
        try:
            if os.path.exists(config["model_path"]) and os.path.exists(
                config["scaler_path"]
            ):
                with open(config["model_path"], "rb") as f:
                    models[disease_id] = pickle.load(f)
                with open(config["scaler_path"], "rb") as f:
                    scalers[disease_id] = pickle.load(f)
                logger.info(f"Loaded {config['name']} model successfully")
            else:
                logger.warning(
                    f"{config['name']} model not found. Train the model first."
                )
        except Exception as e:
            logger.error(f"Error loading {config['name']} model: {e}")


load_disease_models()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main landing page with disease selection"""
    return templates.TemplateResponse(
        "index.html", {"request": request, "diseases": DISEASES}
    )


@app.get("/disease/{disease_id}", response_class=HTMLResponse)
async def disease_page(request: Request, disease_id: str):
    """Individual disease prediction page"""
    if disease_id not in DISEASES:
        raise HTTPException(status_code=404, detail="Disease not found")

    disease_config = DISEASES[disease_id]
    model_available = disease_id in models

    return templates.TemplateResponse(
        f"diseases/{disease_id}.html",
        {
            "request": request,
            "disease": disease_config,
            "model_available": model_available,
        },
    )


@app.post("/predict/{disease_id}")
async def predict(disease_id: str, request: Request):
    """Make prediction for specific disease"""
    try:
        if disease_id not in DISEASES:
            raise HTTPException(status_code=404, detail="Disease not found")

        if disease_id not in models or disease_id not in scalers:
            raise HTTPException(
                status_code=503,
                detail=f"{DISEASES[disease_id]['name']} model not loaded. Please train the model first.",
            )

        data = await request.json()

        # Extract features
        features = []
        missing_features = []

        for feature_name in DISEASES[disease_id]["features"]:
            if feature_name in data:
                try:
                    features.append(float(data[feature_name]))
                except (ValueError, TypeError):
                    raise HTTPException(
                        status_code=400, detail=f"Invalid value for {feature_name}"
                    )
            else:
                missing_features.append(feature_name)

        if missing_features:
            raise HTTPException(
                status_code=400, detail=f"Missing features: {missing_features}"
            )

        # Make prediction
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scalers[disease_id].transform(features_array)

        prediction = int(models[disease_id].predict(features_scaled)[0])
        prediction_proba = models[disease_id].predict_proba(features_scaled)[0]

        probability = float(prediction_proba[1] * 100)

        # Apply domain knowledge adjustments for diabetes
        if disease_id == "diabetes":
            if float(data.get("Smoker", 0)) == 1:
                bmi = float(data.get("BMI", 25))
                high_bp = float(data.get("HighBP", 0))
                adjustment = 3 if (bmi > 30 or high_bp == 1) else 2
                probability = min(probability + adjustment, 99.9)

            if float(data.get("HvyAlcoholConsump", 0)) == 1:
                probability = min(probability + 2, 99.9)

        # Determine risk level
        if probability < 30:
            risk_level, risk_color = "Low", "green"
        elif probability < 70:
            risk_level, risk_color = "Moderate", "orange"
        else:
            risk_level, risk_color = "High", "red"

        result = {
            "success": True,
            "disease": DISEASES[disease_id]["name"],
            "prediction": prediction,
            "prediction_text": f"{DISEASES[disease_id]['name']} Risk Detected"
            if prediction == 1
            else f"No {DISEASES[disease_id]['name']} Risk Detected",
            "probability": round(float(probability), 2),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "recommendations": get_recommendations(disease_id, prediction, data),
        }

        logger.info(
            f"{disease_id} prediction: {result['prediction_text']} ({probability:.2f}%)"
        )

        # Save diabetes prediction to PostgreSQL
        if disease_id == "diabetes":
            await save_diabetes_prediction(data, result)
        
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error for {disease_id}: {e}")
        raise HTTPException(
            status_code=500, detail="An error occurred during prediction"
        )


def get_recommendations(disease_id: str, prediction: int, user_data: Dict) -> List[str]:
    """Generate personalized recommendations"""
    recommendations = []

    if disease_id == "kidney_disease":
        if prediction == 1:
            recommendations.extend(
                [
                    "Consult a nephrologist immediately for proper diagnosis",
                    "Get comprehensive kidney function tests (GFR, creatinine)",
                    "Monitor blood pressure and blood sugar regularly",
                ]
            )

        try:
            gfr = float(user_data.get("GFR", 90))
            if gfr < 60:
                recommendations.append(
                    "Your GFR indicates reduced kidney function - seek medical attention"
                )
            elif gfr < 90:
                recommendations.append(
                    "Mildly reduced kidney function - monitor regularly"
                )
        except (ValueError, TypeError):
            pass

        try:
            creatinine = float(user_data.get("SerumCreatinine", 1.0))
            if creatinine > 1.3:
                recommendations.append("Elevated creatinine levels - consult a doctor")
        except (ValueError, TypeError):
            pass

        try:
            systolic = float(user_data.get("SystolicBP", 120))
            if systolic > 140:
                recommendations.append(
                    "High blood pressure can damage kidneys - manage BP"
                )
        except (ValueError, TypeError):
            pass

        if prediction == 0:
            recommendations.extend(
                [
                    "Maintain healthy blood pressure and blood sugar",
                    "Stay hydrated and limit sodium intake",
                    "Avoid excessive use of NSAIDs",
                    "Get regular kidney function tests",
                ]
            )

    elif disease_id == "heart_disease":
        if prediction == 1:
            recommendations.extend(
                [
                    "Consult a cardiologist immediately for proper diagnosis",
                    "Consider getting an ECG and stress test",
                    "Monitor blood pressure and cholesterol regularly",
                ]
            )

        chest_pain = int(user_data.get("chestpain", 0))
        if chest_pain in [0, 1]:
            recommendations.append("Seek immediate medical attention for chest pain")

        try:
            bp = float(user_data.get("restingBP", 120))
            if bp > 140:
                recommendations.append("Your blood pressure is high - consult a doctor")
            elif bp > 130:
                recommendations.append("Monitor your blood pressure regularly")
        except (ValueError, TypeError):
            pass

        try:
            chol = float(user_data.get("serumcholestrol", 200))
            if chol > 240:
                recommendations.append(
                    "High cholesterol - consider dietary changes and medication"
                )
            elif chol > 200:
                recommendations.append(
                    "Borderline high cholesterol - improve diet and exercise"
                )
        except (ValueError, TypeError):
            pass

        if user_data.get("exerciseangia") == "1":
            recommendations.append(
                "Exercise-induced chest pain requires medical evaluation"
            )

        if prediction == 0:
            recommendations.extend(
                [
                    "Maintain a heart-healthy diet low in saturated fats",
                    "Exercise regularly (150 minutes per week)",
                    "Avoid smoking and limit alcohol consumption",
                    "Get regular cardiovascular check-ups",
                ]
            )

    elif disease_id == "diabetes":
        if prediction == 1:
            recommendations.extend(
                [
                    "Consult with a healthcare professional for proper diagnosis",
                    "Consider getting an HbA1c test",
                    "Monitor blood glucose levels regularly",
                ]
            )

        try:
            bmi = float(user_data.get("BMI", 0))
            if bmi > 25:
                recommendations.append("Consider a weight management program")
            if bmi > 30:
                recommendations.append(
                    "Consult a nutritionist for a personalized diet plan"
                )
        except (ValueError, TypeError):
            pass

        if user_data.get("PhysActivity") == "0":
            recommendations.append(
                "Incorporate at least 150 minutes of moderate exercise per week"
            )

        if user_data.get("Smoker") == "1":
            recommendations.append("Consider smoking cessation programs")

        if user_data.get("Fruits") == "0" or user_data.get("Veggies") == "0":
            recommendations.append("Increase daily intake of fruits and vegetables")

        if prediction == 0:
            recommendations.extend(
                [
                    "Maintain current healthy lifestyle habits",
                    "Get regular health check-ups",
                ]
            )

    elif disease_id == "breast_cancer":
        if prediction == 1:
            recommendations.extend(
                [
                    "Consult an oncologist immediately for proper diagnosis",
                    "Schedule a mammogram and ultrasound examination",
                    "Consider getting a biopsy for definitive diagnosis",
                    "Discuss treatment options with your healthcare team",
                ]
            )

        # Check concerning features
        try:
            area_worst = float(user_data.get("area_worst", 0))
            if area_worst > 1000:
                recommendations.append(
                    "Large tumor area detected - seek immediate medical attention"
                )
        except (ValueError, TypeError):
            pass

        try:
            concave_points_worst = float(user_data.get("concave points_worst", 0))
            if concave_points_worst > 0.15:
                recommendations.append(
                    "Irregular cell structure detected - further testing recommended"
                )
        except (ValueError, TypeError):
            pass

        if prediction == 0:
            recommendations.extend(
                [
                    "Continue regular breast self-examinations monthly",
                    "Schedule annual mammograms (age 40+) or as recommended",
                    "Maintain a healthy weight and exercise regularly",
                    "Limit alcohol consumption",
                    "Know your family history and discuss with your doctor",
                    "Consider genetic counseling if family history is present",
                ]
            )

    return recommendations


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": {disease: disease in models for disease in DISEASES},
    }

#signup & sigin
class SignupSchema(BaseModel):
    email: str
    password: str
    username: str
    
class SigninSchema(BaseModel):
    email: str
    password: str

@app.post("/signup")
async def signup(user: SignupSchema):

    existing_user = await db.user.find_unique(
        where={
            "email": user.email
        }
    )

    if existing_user:

        raise HTTPException(
            status_code=400,
            detail="User already exists"
        )

    hashed_password = hash_password(
        user.password.strip()
    )

    await db.user.create(
        data={
            "username": user.username,
            "email": user.email,
            "password": hashed_password
        }
    )

    return {
        "message": "User created successfully"
    }

@app.post("/signin")
async def signin(user: SigninSchema):

    db_user = await db.user.find_unique(
        where={
            "email": user.email
        }
    )

    if not db_user:

        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )

    valid_password = verify_password(
        user.password.strip(),
        db_user.password
    )

    if not valid_password:

        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )

    token = create_access_token({
        "user_id": db_user.id,
        "email": db_user.email
    })

    return {
        "access_token": token,
        "token_type": "bearer"
    }

@app.get("/profile")
async def profile(
    token: str = Depends(oauth2_scheme)
):

    try:

        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        return {
            "message": "Protected Route",
            "user": payload
        }

    except JWTError:

        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )   





if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=5000)

"""
Multi-Disease Prediction System
FastAPI Application
"""

import os
import pickle
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Disease Prediction System", version="1.0.0")

# Templates
templates = Jinja2Templates(directory="templates")

# Disease configurations
DISEASES = {
    "diabetes": {
        "name": "Diabetes",
        "model_path": "models/diabetes/model.pkl",
        "scaler_path": "models/diabetes/scaler.pkl",
        "features": [
            "HighBP",
            "HighChol",
            "CholCheck",
            "BMI",
            "Smoker",
            "Stroke",
            "HeartDiseaseorAttack",
            "PhysActivity",
            "Fruits",
            "Veggies",
            "HvyAlcoholConsump",
            "AnyHealthcare",
            "NoDocbcCost",
            "GenHlth",
            "MentHlth",
            "PhysHlth",
            "DiffWalk",
            "Sex",
            "Age",
            "Education",
            "Income",
        ],
    },
    "heart_disease": {
        "name": "Heart Disease",
        "model_path": "models/heart_disease/model.pkl",
        "scaler_path": "models/heart_disease/scaler.pkl",
        "features": [
            "age",
            "gender",
            "chestpain",
            "restingBP",
            "serumcholestrol",
            "fastingbloodsugar",
            "restingrelectro",
            "maxheartrate",
            "exerciseangia",
            "oldpeak",
            "slope",
            "noofmajorvessels",
        ],
    },
    "kidney_disease": {
        "name": "Kidney Disease",
        "model_path": "models/kidney_disease/model.pkl",
        "scaler_path": "models/kidney_disease/scaler.pkl",
        "features": [
            "Age",
            "Gender",
            "BMI",
            "FamilyHistoryKidneyDisease",
            "FamilyHistoryHypertension",
            "Edema",
            "FatigueLevels",
            "Itching",
            "MuscleCramps",
            "NauseaVomiting",
            "UrinaryTractInfections",
            "SystolicBP",
            "DiastolicBP",
            "FamilyHistoryDiabetes",
            "PreviousAcuteKidneyInjury",
            "Smoking",
            "PhysicalActivity",
            "DietQuality",
            "SleepQuality",
            "AlcoholConsumption",
            "SerumCreatinine",
            "GFR",
            "FastingBloodSugar",
        ],
    },
    "breast_cancer": {
        "name": "Breast Cancer",
        "model_path": "models/breast_cancer/model.pkl",
        "scaler_path": "models/breast_cancer/scaler.pkl",
        "features": [
            "radius_mean",
            "texture_mean",
            "perimeter_mean",
            "area_mean",
            "smoothness_mean",
            "compactness_mean",
            "concavity_mean",
            "concave points_mean",
            "symmetry_mean",
            "fractal_dimension_mean",
            "radius_se",
            "texture_se",
            "perimeter_se",
            "area_se",
            "smoothness_se",
            "compactness_se",
            "concavity_se",
            "concave points_se",
            "symmetry_se",
            "fractal_dimension_se",
            "radius_worst",
            "texture_worst",
            "perimeter_worst",
            "area_worst",
            "smoothness_worst",
            "compactness_worst",
            "concavity_worst",
            "concave points_worst",
            "symmetry_worst",
            "fractal_dimension_worst",
        ],
    },
}

# Load models
models = {}
scalers = {}


def load_disease_models():
    """Load all available disease models"""
    for disease_id, config in DISEASES.items():
        try:
            if os.path.exists(config["model_path"]) and os.path.exists(
                config["scaler_path"]
            ):
                with open(config["model_path"], "rb") as f:
                    models[disease_id] = pickle.load(f)
                with open(config["scaler_path"], "rb") as f:
                    scalers[disease_id] = pickle.load(f)
                logger.info(f"Loaded {config['name']} model successfully")
            else:
                logger.warning(
                    f"{config['name']} model not found. Train the model first."
                )
        except Exception as e:
            logger.error(f"Error loading {config['name']} model: {e}")


load_disease_models()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main landing page with disease selection"""
    return templates.TemplateResponse(
        "index.html", {"request": request, "diseases": DISEASES}
    )


@app.get("/disease/{disease_id}", response_class=HTMLResponse)
async def disease_page(request: Request, disease_id: str):
    """Individual disease prediction page"""
    if disease_id not in DISEASES:
        raise HTTPException(status_code=404, detail="Disease not found")

    disease_config = DISEASES[disease_id]
    model_available = disease_id in models

    return templates.TemplateResponse(
        f"diseases/{disease_id}.html",
        {
            "request": request,
            "disease": disease_config,
            "model_available": model_available,
        },
    )


@app.post("/predict/{disease_id}")
async def predict(disease_id: str, request: Request):
    """Make prediction for specific disease"""
    try:
        if disease_id not in DISEASES:
            raise HTTPException(status_code=404, detail="Disease not found")

        if disease_id not in models or disease_id not in scalers:
            raise HTTPException(
                status_code=503,
                detail=f"{DISEASES[disease_id]['name']} model not loaded. Please train the model first.",
            )

        data = await request.json()

        # Extract features
        features = []
        missing_features = []

        for feature_name in DISEASES[disease_id]["features"]:
            if feature_name in data:
                try:
                    features.append(float(data[feature_name]))
                except (ValueError, TypeError):
                    raise HTTPException(
                        status_code=400, detail=f"Invalid value for {feature_name}"
                    )
            else:
                missing_features.append(feature_name)

        if missing_features:
            raise HTTPException(
                status_code=400, detail=f"Missing features: {missing_features}"
            )

        # Make prediction
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scalers[disease_id].transform(features_array)

        prediction = int(models[disease_id].predict(features_scaled)[0])
        prediction_proba = models[disease_id].predict_proba(features_scaled)[0]

        probability = float(prediction_proba[1] * 100)

        # Apply domain knowledge adjustments for diabetes
        if disease_id == "diabetes":
            if float(data.get("Smoker", 0)) == 1:
                bmi = float(data.get("BMI", 25))
                high_bp = float(data.get("HighBP", 0))
                adjustment = 3 if (bmi > 30 or high_bp == 1) else 2
                probability = min(probability + adjustment, 99.9)

            if float(data.get("HvyAlcoholConsump", 0)) == 1:
                probability = min(probability + 2, 99.9)

        # Determine risk level
        if probability < 30:
            risk_level, risk_color = "Low", "green"
        elif probability < 70:
            risk_level, risk_color = "Moderate", "orange"
        else:
            risk_level, risk_color = "High", "red"

        result = {
            "success": True,
            "disease": DISEASES[disease_id]["name"],
            "prediction": prediction,
            "prediction_text": f"{DISEASES[disease_id]['name']} Risk Detected"
            if prediction == 1
            else f"No {DISEASES[disease_id]['name']} Risk Detected",
            "probability": round(float(probability), 2),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "recommendations": get_recommendations(disease_id, prediction, data),
        }

        logger.info(
            f"{disease_id} prediction: {result['prediction_text']} ({probability:.2f}%)"
        )
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error for {disease_id}: {e}")
        raise HTTPException(
            status_code=500, detail="An error occurred during prediction"
        )


def get_recommendations(disease_id: str, prediction: int, user_data: Dict) -> List[str]:
    """Generate personalized recommendations"""
    recommendations = []

    if disease_id == "kidney_disease":
        if prediction == 1:
            recommendations.extend(
                [
                    "Consult a nephrologist immediately for proper diagnosis",
                    "Get comprehensive kidney function tests (GFR, creatinine)",
                    "Monitor blood pressure and blood sugar regularly",
                ]
            )

        try:
            gfr = float(user_data.get("GFR", 90))
            if gfr < 60:
                recommendations.append(
                    "Your GFR indicates reduced kidney function - seek medical attention"
                )
            elif gfr < 90:
                recommendations.append(
                    "Mildly reduced kidney function - monitor regularly"
                )
        except (ValueError, TypeError):
            pass

        try:
            creatinine = float(user_data.get("SerumCreatinine", 1.0))
            if creatinine > 1.3:
                recommendations.append("Elevated creatinine levels - consult a doctor")
        except (ValueError, TypeError):
            pass

        try:
            systolic = float(user_data.get("SystolicBP", 120))
            if systolic > 140:
                recommendations.append(
                    "High blood pressure can damage kidneys - manage BP"
                )
        except (ValueError, TypeError):
            pass

        if prediction == 0:
            recommendations.extend(
                [
                    "Maintain healthy blood pressure and blood sugar",
                    "Stay hydrated and limit sodium intake",
                    "Avoid excessive use of NSAIDs",
                    "Get regular kidney function tests",
                ]
            )

    elif disease_id == "heart_disease":
        if prediction == 1:
            recommendations.extend(
                [
                    "Consult a cardiologist immediately for proper diagnosis",
                    "Consider getting an ECG and stress test",
                    "Monitor blood pressure and cholesterol regularly",
                ]
            )

        chest_pain = int(user_data.get("chestpain", 0))
        if chest_pain in [0, 1]:
            recommendations.append("Seek immediate medical attention for chest pain")

        try:
            bp = float(user_data.get("restingBP", 120))
            if bp > 140:
                recommendations.append("Your blood pressure is high - consult a doctor")
            elif bp > 130:
                recommendations.append("Monitor your blood pressure regularly")
        except (ValueError, TypeError):
            pass

        try:
            chol = float(user_data.get("serumcholestrol", 200))
            if chol > 240:
                recommendations.append(
                    "High cholesterol - consider dietary changes and medication"
                )
            elif chol > 200:
                recommendations.append(
                    "Borderline high cholesterol - improve diet and exercise"
                )
        except (ValueError, TypeError):
            pass

        if user_data.get("exerciseangia") == "1":
            recommendations.append(
                "Exercise-induced chest pain requires medical evaluation"
            )

        if prediction == 0:
            recommendations.extend(
                [
                    "Maintain a heart-healthy diet low in saturated fats",
                    "Exercise regularly (150 minutes per week)",
                    "Avoid smoking and limit alcohol consumption",
                    "Get regular cardiovascular check-ups",
                ]
            )

    elif disease_id == "diabetes":
        if prediction == 1:
            recommendations.extend(
                [
                    "Consult with a healthcare professional for proper diagnosis",
                    "Consider getting an HbA1c test",
                    "Monitor blood glucose levels regularly",
                ]
            )

        try:
            bmi = float(user_data.get("BMI", 0))
            if bmi > 25:
                recommendations.append("Consider a weight management program")
            if bmi > 30:
                recommendations.append(
                    "Consult a nutritionist for a personalized diet plan"
                )
        except (ValueError, TypeError):
            pass

        if user_data.get("PhysActivity") == "0":
            recommendations.append(
                "Incorporate at least 150 minutes of moderate exercise per week"
            )

        if user_data.get("Smoker") == "1":
            recommendations.append("Consider smoking cessation programs")

        if user_data.get("Fruits") == "0" or user_data.get("Veggies") == "0":
            recommendations.append("Increase daily intake of fruits and vegetables")

        if prediction == 0:
            recommendations.extend(
                [
                    "Maintain current healthy lifestyle habits",
                    "Get regular health check-ups",
                ]
            )

    elif disease_id == "breast_cancer":
        if prediction == 1:
            recommendations.extend(
                [
                    "Consult an oncologist immediately for proper diagnosis",
                    "Schedule a mammogram and ultrasound examination",
                    "Consider getting a biopsy for definitive diagnosis",
                    "Discuss treatment options with your healthcare team",
                ]
            )

        # Check concerning features
        try:
            area_worst = float(user_data.get("area_worst", 0))
            if area_worst > 1000:
                recommendations.append(
                    "Large tumor area detected - seek immediate medical attention"
                )
        except (ValueError, TypeError):
            pass

        try:
            concave_points_worst = float(user_data.get("concave points_worst", 0))
            if concave_points_worst > 0.15:
                recommendations.append(
                    "Irregular cell structure detected - further testing recommended"
                )
        except (ValueError, TypeError):
            pass

        if prediction == 0:
            recommendations.extend(
                [
                    "Continue regular breast self-examinations monthly",
                    "Schedule annual mammograms (age 40+) or as recommended",
                    "Maintain a healthy weight and exercise regularly",
                    "Limit alcohol consumption",
                    "Know your family history and discuss with your doctor",
                    "Consider genetic counseling if family history is present",
                ]
            )

    return recommendations


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": {disease: disease in models for disease in DISEASES},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=5000)
