# Project Notes

## Recent Updates

### FastAPI Migration (Jan 19, 2026)
- Migrated from Flask to FastAPI for better performance
- All endpoints working correctly
- Server runs on http://localhost:5000 (use localhost, not 0.0.0.0 on Windows)

### Overfitting Prevention (Jan 19, 2026)
- Added train/test comparison to all training scripts
- Increased regularization in all models
- Models now show realistic performance (85-96% accuracy)
- Automatic overfitting detection with warnings

### Breast Cancer Model Added (Jan 19, 2026)
- Wisconsin Breast Cancer Dataset (569 samples, 30 features)
- Gradient Boosting model: 96.49% test accuracy
- Fully integrated into the web application

## Active Models (4)
1. **Diabetes** - 21 features, ~85-90% accuracy
2. **Heart Disease** - 12 features, ~85-92% accuracy
3. **Kidney Disease** - 23 simplified features, ~88-94% accuracy
4. **Breast Cancer** - 30 features, ~96% accuracy

## Training Scripts
- `train_diabetes.py` - Diabetes model
- `train_heart_disease.py` - Heart disease model
- `train_kidney_disease_simplified.py` - Kidney disease (user-friendly features)
- `train_breast_cancer.py` - Breast cancer model

## Key Files
- `app.py` - FastAPI application
- `requirements.txt` - Python dependencies
- `start_server.bat` - Easy server startup (Windows)
- `test_server.py` - Server testing utility

## Overfitting Detection
All training scripts now include:
- Train vs Test accuracy comparison
- Automatic warning if gap > 10%
- Model selection based on test AUC (not train)
- Cross-validation scores

## Quick Commands
```bash
# Start server
python app.py

# Train models
python train_models/train_diabetes.py
python train_models/train_heart_disease.py
python train_models/train_kidney_disease_simplified.py
python train_models/train_breast_cancer.py

# Test server
python test_server.py
```

## Important Notes
- Use `localhost` or `127.0.0.1` in browser
- Models trained with scikit-learn 1.7.1, running on 1.8.0 (warnings are normal)
- All models use StandardScaler for feature normalization
- SMOTE used conservatively to prevent overfitting
