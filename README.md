# Multi-Disease Prediction System

A comprehensive web application for predicting multiple diseases using machine learning. Currently supports diabetes prediction with BRFSS dataset, with architecture ready for additional diseases. Built with Flask backend and responsive frontend.

## 🚀 Features

- **Multi-Disease Support**: Modular architecture for multiple disease predictions
- **Machine Learning Prediction**: Uses trained ML models with multiple algorithms
- **Interactive Web Interface**: Modern landing page with disease selection
- **Real-time Risk Assessment**: Instant predictions with probability scores
- **Personalized Recommendations**: Tailored health advice based on risk factors
- **Responsive Design**: Works on desktop and mobile devices
- **Scalable Architecture**: Easy to add new disease prediction models

## 📊 Dataset Information

The application uses the BRFSS (Behavioral Risk Factor Surveillance System) dataset, which includes:

### Features Used (21 total):

- **Demographics**: Age, Sex, Education, Income
- **Health Conditions**: BMI, High BP, High Cholesterol, Stroke, Heart Disease
- **Lifestyle Factors**: Smoking, Physical Activity, Diet, Alcohol Consumption
- **Healthcare Access**: Insurance, Doctor visits, General health status

### Target Variable:

- `Diabetes_binary`: 0 (No Diabetes), 1 (Prediabetes/Diabetes)

## 🛠️ Technology Stack

### Backend:

- **Python 3.8+**
- **Flask** - Web framework
- **Scikit-learn** - Machine learning
- **Pandas/NumPy** - Data processing
- **Imbalanced-learn** - SMOTE for class balancing

### Frontend:

- **HTML5/CSS3** - Structure and styling
- **Bootstrap 5** - Responsive design
- **JavaScript** - Interactive functionality
- **Font Awesome** - Icons

### Machine Learning:

- **Random Forest Classifier** - Primary algorithm
- **Gradient Boosting** - Alternative model
- **Standard Scaler** - Feature normalization
- **SMOTE** - Class imbalance handling

## 📁 Project Structure

```
multi-disease-predictor/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── config_template.py              # Configuration template
├── templates/
│   ├── index.html                  # Landing page with disease selection
│   └── diseases/
│       ├── diabetes.html           # Diabetes prediction page
│       └── [other diseases].html   # Future disease pages
├── train_models/
│   ├── train_diabetes.py           # Diabetes model training
│   └── train_[disease].py          # Future disease training scripts
├── models/
│   ├── diabetes/
│   │   ├── model.pkl               # Trained diabetes model
│   │   └── scaler.pkl              # Diabetes feature scaler
│   └── [other diseases]/           # Future disease models
├── data/
│   ├── diabetes_data.csv           # BRFSS dataset
│   └── [other datasets]/           # Future datasets
└── docs/
    ├── API_DOCUMENTATION.txt
    ├── feature_mapping.txt
    └── QUICK_START.txt
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/diabetes-prediction-system.git
cd diabetes-prediction-system
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download BRFSS Dataset

- Download the BRFSS diabetes dataset from [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- Place the CSV file in the `data/` directory as `diabetes_data.csv`

### 5. Train the Models

```bash
# Train diabetes model
python train_models/train_diabetes.py

# Future: Add other disease models
# python train_models/train_heart_disease.py
```

This will:

- Load and preprocess the dataset
- Train multiple ML models
- Select the best performing model
- Save the trained model and scaler to the `models/[disease]/` directory

### 6. Run the Application

```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

## 📈 Model Performance

The system trains and compares multiple algorithms:

| Algorithm           | Typical Accuracy | AUC Score | Notes                          |
| ------------------- | ---------------- | --------- | ------------------------------ |
| Random Forest       | 85-90%           | 0.87-0.92 | Best overall performance       |
| Gradient Boosting   | 84-89%           | 0.86-0.91 | Good for feature importance    |
| Logistic Regression | 82-87%           | 0.84-0.89 | Most interpretable             |
| SVM                 | 83-88%           | 0.85-0.90 | Good for high-dimensional data |

## 🎯 Usage Guide

### Web Interface:

1. **Fill Demographics**: Age, gender, education, income
2. **Health Conditions**: BMI, blood pressure, cholesterol, etc.
3. **Lifestyle Factors**: Exercise, diet, smoking, alcohol
4. **Healthcare Access**: Insurance, doctor visits
5. **Click Predict**: Get instant risk assessment

### API Usage:

```python
import requests

# Example prediction request
data = {
    "HighBP": 1,
    "HighChol": 0,
    "BMI": 28.5,
    "Smoker": 0,
    "PhysActivity": 1,
    # ... other features
}

response = requests.post("http://localhost:5000/predict", json=data)
result = response.json()

print(f"Diabetes Risk: {result['probability']}%")
print(f"Risk Level: {result['risk_level']}")
```

## 🔧 Deployment

### Local Development:

```bash
python app.py
```

### Production (with Gunicorn):

```bash
gunicorn --bind 0.0.0.0:5000 app:app
```

### Docker Deployment:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

### Cloud Deployment:

- **Heroku**: Use provided Procfile
- **AWS EC2**: Deploy with nginx + gunicorn
- **Google Cloud**: Use App Engine
- **Docker**: Container-ready

## 📊 Model Validation

The training script includes comprehensive validation:

- **Train/Test Split**: 80/20 stratified split
- **Cross-validation**: 5-fold CV for model selection
- **Class Balancing**: SMOTE for handling imbalanced data
- **Feature Scaling**: StandardScaler for numerical features
- **Hyperparameter Tuning**: GridSearchCV for optimization

## 🎯 Key Features

### 1. Comprehensive Health Assessment

- 21 validated risk factors from BRFSS
- Evidence-based feature selection
- Standardized health indicators

### 2. Advanced ML Pipeline

- Multiple algorithm comparison
- Automated model selection
- Feature importance analysis
- Class imbalance handling

### 3. User Experience

- Intuitive web interface
- Real-time predictions
- Visual risk indicators
- Personalized recommendations

### 4. Clinical Relevance

- Based on CDC BRFSS data
- Validated risk factors
- Clinical decision support
- Population health insights

## ⚠️ Important Disclaimers

1. **Not for Medical Diagnosis**: This tool is for educational and screening purposes only
2. **Consult Healthcare Professionals**: Always seek professional medical advice
3. **Model Limitations**: Predictions are based on population data and may not reflect individual cases
4. **Data Privacy**: Ensure HIPAA compliance for production use

## 🔄 Model Updates

### Retraining the Model:

1. Update the dataset with new BRFSS data
2. Run `python train_model.py`
3. Restart the Flask application
4. Validate performance metrics

### Adding Features:

1. Update feature list in both `train_model.py` and `app.py`
2. Modify the HTML form to include new inputs
3. Retrain the model with new features
4. Update API documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CDC BRFSS**: For providing the comprehensive health survey data
- **Kaggle Community**: For dataset preparation and sharing
- **Scikit-learn**: For robust machine learning tools
- **Flask Community**: For the excellent web framework

## 📞 Support

For questions, issues, or contributions:

- **Issues**: [GitHub Issues](https://github.com/yourusername/diabetes-prediction-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/diabetes-prediction-system/discussions)
- **Email**: your.email@example.com

---

**Built with ❤️ for better healthcare outcomes**
