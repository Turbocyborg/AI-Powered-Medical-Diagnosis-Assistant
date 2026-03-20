# Multi-Disease Prediction System

AI-powered web application for predicting multiple diseases using machine learning. Built with FastAPI and modern ML techniques.

## 🚀 Features

- **4 Active Disease Models**: Diabetes, Heart Disease, Kidney Disease, Breast Cancer
- **Real-time Predictions**: Instant risk assessment with probability scores
- **Personalized Recommendations**: Tailored health advice based on risk factors
- **Modern UI**: Responsive design with Bootstrap 5
- **RESTful API**: FastAPI backend with automatic documentation

## 📊 Models & Performance

| Disease | Features | Test Accuracy | Test AUC | Status |
|---------|----------|---------------|----------|--------|
| Diabetes | 21 | ~85-90% | 0.87-0.92 | ✓ Active |
| Heart Disease | 12 | ~85-92% | 0.88-0.94 | ✓ Active |
| Kidney Disease | 23 | ~88-94% | 0.90-0.96 | ✓ Active |
| Breast Cancer | 30 | ~96% | 0.99 | ✓ Active |

## 🛠️ Technology Stack

- **Backend**: FastAPI, Uvicorn, Python 3.8+
- **ML**: Scikit-learn, XGBoost, Imbalanced-learn
- **Frontend**: Bootstrap 5, JavaScript, Font Awesome
- **Data**: Pandas, NumPy

## 📁 Project Structure

```
AI-Powered-Medical-Diagnosis-Assistant/
├── app.py                          # FastAPI application
├── requirements.txt                # Dependencies
├── start_server.bat               # Windows startup script
├── test_server.py                 # Server testing
├── README.md                      # Main documentation
├── SETUP_GUIDE.md                 # Setup instructions
├── PROJECT_NOTES.md               # Development notes
│
├── data/                          # Datasets
├── models/                        # Trained models (4 diseases)
├── templates/                     # HTML templates
│   ├── index.html                # Landing page
│   └── diseases/                 # Disease-specific pages
└── train_models/                  # Training scripts
    ├── train_diabetes.py
    ├── train_heart_disease.py
    ├── train_kidney_disease_simplified.py
    └── train_breast_cancer.py
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models (if needed)

```bash
python train_models/train_diabetes.py
python train_models/train_heart_disease.py
python train_models/train_kidney_disease_simplified.py
python train_models/train_breast_cancer.py
```

### 3. Run the Application

```bash
# Option 1: Direct
python app.py

# Option 2: With uvicorn
uvicorn app:app --host 127.0.0.1 --port 5000 --reload

# Option 3: Windows batch file
start_server.bat
```

### 4. Access the Application

Open your browser: **http://localhost:5000**

**Note**: Use `localhost` or `127.0.0.1` on Windows!

## 🎯 API Endpoints

- `GET /` - Landing page
- `GET /disease/{disease_id}` - Disease prediction page
- `POST /predict/{disease_id}` - Make prediction (JSON)
- `GET /health` - Health check
- `GET /docs` - API documentation (FastAPI auto-generated)

## ⚠️ Important Notes

- **Medical Disclaimer**: For educational purposes only. Not for medical diagnosis.
- **Data Privacy**: No data is stored. All predictions are real-time.
- **Model Limitations**: Based on population data, may not reflect individual cases.

## 📞 Support

For issues or questions, check:
- README.md (this file)
- SETUP_GUIDE.md (detailed setup)
- PROJECT_NOTES.md (development notes)
- docs/ folder (API and feature documentation)

---

**Built with ❤️ for better healthcare outcomes**
