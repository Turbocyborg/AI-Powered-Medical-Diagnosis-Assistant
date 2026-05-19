from database.db import db


async def save_diabetes_prediction(data, result):

    await db.diabetesprediction.create(
        data={

            "ageCategory":
                str(data.get("Age", "")),

            "bmi":
                float(data.get("BMI", 0)),

            "highBloodPressure":
                bool(int(data.get("HighBP", 0))),

            "highCholesterol":
                bool(int(data.get("HighChol", 0))),

            "heartDisease":
                bool(int(data.get("HeartDiseaseorAttack", 0))),

            "smoker":
                bool(int(data.get("Smoker", 0))),

            "physicalActivity":
                bool(int(data.get("PhysActivity", 0))),

            "prediction":
                result["prediction_text"],

            "probability":
                result["probability"],

            "riskLevel":
                result["risk_level"],
        }
    )