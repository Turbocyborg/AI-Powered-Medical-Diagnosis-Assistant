# Multi-Disease Prediction System - Project Structure

## Overview

This is a professionally structured multi-disease prediction system built with Flask and machine learning. The architecture is modular and scalable, allowing easy addition of new disease prediction models.

## Current Status

✅ **Diabetes Prediction** - Fully implemented with BRFSS dataset
🔄 **Heart Disease** - Ready to implement
🔄 **Cancer** - Ready to implement
🔄 **Kidney Disease** - Ready to implement
🔄 **Liver Disease** - Ready to implement
🔄 **Stroke** - Ready to implement

## Directory Structure

```
multi-disease-predictor/
│
├── app.py                          # Main Flask application with routing
│   ├── Disease configurations
│   ├── Model loading
│   ├── Prediction endpoints
│   └── Recommendation engine
│
├── templates/
│   ├── index.html                  # Landing page with disease cards
│   └── diseases/
│       ├── diabetes.html           # Diabetes prediction form
│       ├── heart_disease.html      # (Future) Heart disease form
│       ├── cancer.html             # (Future) Cancer screening form
│       └── ...                     # Other disease forms
│
├── train_models/
│   ├── train_diabetes.py           # Diabetes model training script
│   ├── train_heart_disease.py      # (Future) Heart disease training
│   └── ...                         # Other training scripts
│
├── models/
│   ├── diabetes/
│   │   ├── model.pkl               # Trained diabetes model
│   │   └── scaler.pkl              # Feature scaler
│   ├── heart_disease/              # (Future) Heart disease models
│   └── ...                         # Other disease models
│
├── data/
│   ├── diabetes_data.csv           # BRFSS diabetes dataset
│   └── ...                         # Other datasets
│
├── docs/
│   ├── API_DOCUMENTATION.txt       # API endpoint documentation
│   ├── feature_mapping.txt         # Feature descriptions
│   └── QUICK_START.txt             # Quick setup guide
│
├── requirements.txt                # Python dependencies
├── config_template.py              # Configuration template
├── README.md                       # Main documentation
└── PROJECT_STRUCTURE.md            # This file

```

## Key Features

### 1. Modular Architecture

- Each disease has its own model directory
- Separate training scripts for each disease
- Individual HTML templates for each disease
- Easy to add new diseases without modifying existing code

### 2. Scalable Design

- Disease configurations in `app.py` DISEASES dict
- Dynamic route generation
- Centralized model loading
- Shared recommendation engine

### 3. Professional Organization

- Clear separation of concerns
- Training scripts separate from application
- Models organized by disease
- Documentation in dedicated folder

## Adding a New Disease

To add a new disease prediction model:

1. **Create Training Script**

   ```python
   # train_models/train_[disease].py
   # Follow the pattern from train_diabetes.py
   ```

2. **Add Disease Configuration**

   ```python
   # In app.py DISEASES dict
   'disease_name': {
       'name': 'Disease Name',
       'model_path': 'models/disease_name/model.pkl',
       'scaler_path': 'models/disease_name/scaler.pkl',
       'features': ['feature1', 'feature2', ...]
   }
   ```

3. **Create HTML Template**

   ```html
   <!-- templates/diseases/disease_name.html -->
   <!-- Follow the pattern from diabetes.html -->
   ```

4. **Update Landing Page**

   ```html
   <!-- Add disease card in templates/index.html -->
   ```

5. **Add Recommendations**
   ```python
   # In app.py get_recommendations() function
   # Add disease-specific logic
   ```

## File Descriptions

### Core Application Files

- **app.py**: Main Flask application with all routes and logic
- **requirements.txt**: Python package dependencies
- **config_template.py**: Configuration template for customization

### Template Files

- **templates/index.html**: Landing page with disease selection cards
- **templates/diseases/\*.html**: Individual disease prediction forms

### Training Scripts

- **train*models/train*\*.py**: Model training scripts for each disease
- Each script handles data loading, preprocessing, training, and saving

### Model Files

- **models/[disease]/model.pkl**: Trained ML model
- **models/[disease]/scaler.pkl**: Feature scaler for preprocessing

### Documentation

- **README.md**: Main project documentation
- **docs/API_DOCUMENTATION.txt**: API endpoint reference
- **docs/feature_mapping.txt**: Feature descriptions
- **docs/QUICK_START.txt**: Quick setup guide

## Technology Stack

### Backend

- Flask 2.3.3 - Web framework
- Scikit-learn 1.3.0 - Machine learning
- Pandas 2.0.3 - Data processing
- NumPy 1.24.3 - Numerical computing
- Imbalanced-learn 0.11.0 - SMOTE for class balancing

### Frontend

- Bootstrap 5 - Responsive design
- Font Awesome 6 - Icons
- Vanilla JavaScript - Interactivity

### Deployment

- Gunicorn - Production server
- Docker - Containerization (optional)

## Next Steps

1. **Train Diabetes Model**

   ```bash
   python train_models/train_diabetes.py
   ```

2. **Run Application**

   ```bash
   python app.py
   ```

3. **Add More Diseases**

   - Collect datasets
   - Create training scripts
   - Add configurations
   - Build templates

4. **Deploy to Production**
   - Configure environment variables
   - Set up database (optional)
   - Deploy to cloud platform
   - Monitor performance

## Best Practices

- Keep training scripts separate from application
- Use consistent naming conventions
- Document all features and models
- Test each disease module independently
- Version control model files
- Monitor model performance
- Update models regularly with new data

## Support

For questions or issues:

- Check documentation in `docs/` folder
- Review `README.md` for setup instructions
- Examine existing disease implementations as examples
