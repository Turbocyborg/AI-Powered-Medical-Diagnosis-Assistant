# Multi-Disease Predictor - Setup Guide

## Quick Start (5 minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Dataset

Download BRFSS diabetes dataset from [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) and place in `data/diabetes_data.csv`

Or let the training script create a synthetic dataset for testing.

### 3. Train Diabetes Model

```bash
python train_models/train_diabetes.py
```

### 4. Run Application

```bash
python app.py
```

### 5. Open Browser

Visit: http://localhost:5000

## What You'll See

1. **Landing Page** - Cards for different diseases
2. **Diabetes Card** - Click to access diabetes prediction
3. **Prediction Form** - Fill in 21 health indicators
4. **Results** - Get risk assessment and recommendations

## Project Status

### ✅ Completed

- Multi-disease architecture
- Diabetes prediction (fully functional)
- Landing page with disease selection
- Professional UI/UX
- Model training pipeline
- API endpoints

### 🔄 Ready to Add

- Heart Disease prediction
- Cancer screening
- Kidney Disease prediction
- Liver Disease prediction
- Stroke prediction

## Adding Your First Disease Model

Let's say you want to add Heart Disease prediction:

### Step 1: Create Training Script

```python
# train_models/train_heart_disease.py
# Copy train_diabetes.py and modify for heart disease dataset
```

### Step 2: Add to app.py

```python
DISEASES = {
    'diabetes': { ... },
    'heart_disease': {
        'name': 'Heart Disease',
        'model_path': 'models/heart_disease/model.pkl',
        'scaler_path': 'models/heart_disease/scaler.pkl',
        'features': ['age', 'sex', 'cp', 'trestbps', ...]  # Your features
    }
}
```

### Step 3: Create Template

```bash
# Copy templates/diseases/diabetes.html to heart_disease.html
# Modify form fields for heart disease features
```

### Step 4: Update Landing Page

```html
<!-- In templates/index.html, change "Coming Soon" card to active -->
<div class="col-md-6 col-lg-4">
  <div
    class="card disease-card"
    onclick="window.location.href='/disease/heart_disease'"
  >
    <!-- Update card content -->
  </div>
</div>
```

### Step 5: Train and Test

```bash
python train_models/train_heart_disease.py
python app.py
# Visit http://localhost:5000/disease/heart_disease
```

## File Organization

```
Your workflow:
1. Get dataset → data/[disease]_data.csv
2. Create trainer → train_models/train_[disease].py
3. Run training → Creates models/[disease]/model.pkl
4. Update app.py → Add disease config
5. Create template → templates/diseases/[disease].html
6. Test → python app.py
```

## Common Issues

### Model Not Found

```
Error: Model not loaded
Solution: Run python train_models/train_diabetes.py first
```

### Dataset Not Found

```
Error: Dataset not found
Solution: The script will create synthetic data automatically
Or download real dataset to data/diabetes_data.csv
```

### Port Already in Use

```
Error: Address already in use
Solution: Change port in app.py or kill existing process
```

## Development Tips

1. **Test with Synthetic Data First**

   - Training script creates synthetic data if real data not found
   - Good for testing the pipeline

2. **One Disease at a Time**

   - Get diabetes working first
   - Use it as template for other diseases

3. **Check Model Performance**

   - Training script shows accuracy and AUC
   - Aim for >85% accuracy

4. **Customize Recommendations**
   - Edit `get_recommendations()` in app.py
   - Add disease-specific advice

## Next Steps

After basic setup:

1. **Improve Models**

   - Tune hyperparameters
   - Try different algorithms
   - Add feature engineering

2. **Enhance UI**

   - Add charts and visualizations
   - Improve form validation
   - Add progress indicators

3. **Add Features**

   - User accounts
   - History tracking
   - PDF reports
   - Email notifications

4. **Deploy**
   - Set up production server
   - Configure domain
   - Add SSL certificate
   - Monitor performance

## Resources

- **Datasets**: Kaggle, UCI ML Repository, CDC
- **ML Tutorials**: Scikit-learn documentation
- **Flask Docs**: flask.palletsprojects.com
- **Bootstrap**: getbootstrap.com

## Support

Need help?

1. Check PROJECT_STRUCTURE.md for architecture
2. Review existing diabetes implementation
3. Read API_DOCUMENTATION.txt for endpoints
4. Check feature_mapping.txt for data formats

Happy coding! 🚀
