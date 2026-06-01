# AI-Powered Medical Diagnosis Assistant

A comprehensive multi-disease prediction system using machine learning, deep learning, FastAPI backend, HTML/CSS/Bootstrap frontend, and PostgreSQL database.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE                           │
│              HTML + CSS + Bootstrap + JavaScript            │
│                     (Port 5000)                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI Backend                            │
│              - RESTful API Endpoints                        │
│              - Authentication (JWT)                         │
│              - Request Validation                           │
│                     (Port 5000)                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
┌──────────────────┐         ┌──────────────────┐
│  PostgreSQL DB   │         │   ML Models (6)  │
│   (Port 5432)    │         │  - 5 Tabular     │
│                  │         │  - 1 Image (CNN) │
│  - Users         │         └──────────────────┘
│  - Predictions   │
│  - History       │
└──────────────────┘
```

## 🚀 Features

### Disease Prediction Models (6)
- **Diabetes** (21 features) - XGBoost
- **Heart Disease** (12 features) - XGBoost
- **Kidney Disease** (23 features) - XGBoost
- **Breast Cancer** (30 features) - XGBoost
- **Liver Disease** (10 features) - XGBoost
- **Pneumonia Detection** (Image-based) - PyTorch ResNet18 CNN

### User Features
- User registration and authentication
- Prediction history tracking
- Anonymous predictions (optional)
- Personalized recommendations
- Risk level assessment (Low/Moderate/High)
- Real-time predictions

### Technical Features
- RESTful API with FastAPI
- PostgreSQL database for data persistence
- JWT token authentication
- Password hashing with bcrypt
- Automatic API documentation (Swagger UI)
- Responsive HTML/CSS/Bootstrap frontend

## 📋 Prerequisites

- Python 3.8 or higher (Python 3.14 supported)
- Docker (for PostgreSQL)
- Virtual environment (included)

## 🚀 Quick Start

### 1. Start PostgreSQL Database

PostgreSQL is already running in Docker:
```bash
docker ps
# Should show: my-postgres container running on port 5432
```

If not running, start it:
```bash
docker start my-postgres
```

### 2. Initialize Database

```bash
python init_db.py
```

Expected output:
```
✓ Database tables created successfully!
✓ Created 3 tables:
  - users
  - predictions
  - pneumonia_predictions
```

### 3. Start the Application

```bash
start_server.bat
```

Or manually:
```bash
venv\Scripts\activate
uvicorn app:app --host 127.0.0.1 --port 5000
```

### 4. Access the Application

Open your browser: **http://127.0.0.1:5000**

## 📊 API Endpoints

### Public Endpoints
- `GET /` - Main landing page
- `GET /disease/{disease_id}` - Disease prediction page
- `POST /register` - Register new user
- `POST /token` - Login and get JWT token
- `GET /health` - Health check

### Prediction Endpoints
- `POST /predict/{disease_id}` - Predict disease risk (tabular models)
  - disease_id: diabetes, heart_disease, kidney_disease, breast_cancer, liver_disease
- `POST /predict/pneumonia` - Pneumonia detection from X-ray image

### Authenticated Endpoints (Require JWT Token)
- `GET /me` - Get current user information
- `GET /predictions/history` - Get user's prediction history
- `GET /predictions/stats` - Get user's prediction statistics

### API Documentation
- **Swagger UI**: http://127.0.0.1:5000/docs
- **ReDoc**: http://127.0.0.1:5000/redoc

## 🔐 Authentication

### Register a New User

```bash
curl -X POST "http://127.0.0.1:5000/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "john_doe",
    "email": "john@example.com",
    "password": "securepassword123",
    "full_name": "John Doe"
  }'
```

### Login and Get Token

```bash
curl -X POST "http://127.0.0.1:5000/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=john_doe&password=securepassword123"
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user": {
    "id": 1,
    "username": "john_doe",
    "email": "john@example.com",
    "full_name": "John Doe"
  }
}
```

### Use Token for Authenticated Requests

```bash
curl -X GET "http://127.0.0.1:5000/me" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

## 🗄️ Database Schema

### Users Table
- `id` - Primary key
- `username` - Unique username
- `email` - Unique email
- `hashed_password` - Bcrypt hashed password
- `full_name` - User's full name
- `is_active` - Account status
- `is_admin` - Admin flag
- `created_at` - Registration timestamp
- `updated_at` - Last update timestamp

### Predictions Table
- `id` - Primary key
- `user_id` - Foreign key to users (nullable for anonymous)
- `disease_type` - Disease model used
- `input_data` - JSON of input features
- `prediction` - Prediction result (0 or 1)
- `probability` - Probability score
- `risk_level` - Low/Moderate/High
- `model_version` - Model version used
- `ip_address` - Client IP
- `user_agent` - Client user agent
- `created_at` - Prediction timestamp

### Pneumonia Predictions Table
- `id` - Primary key
- `user_id` - Foreign key to users (nullable)
- `image_filename` - Uploaded image filename
- `image_size` - Image size in bytes
- `prediction` - Prediction result (0 or 1)
- `probability` - Probability score
- `confidence` - Model confidence
- `risk_level` - Low/Moderate/High
- `model_version` - Model version
- `ip_address` - Client IP
- `user_agent` - Client user agent
- `created_at` - Prediction timestamp

## 🛠️ Technology Stack

### Backend
- **FastAPI** 0.115.5 - Modern web framework
- **Uvicorn** 0.32.0 - ASGI server
- **SQLAlchemy** 2.0.49 - ORM
- **Alembic** 1.18.4 - Database migrations
- **PostgreSQL** - Database (Docker)
- **psycopg2-binary** - PostgreSQL adapter

### Authentication
- **python-jose** - JWT tokens
- **passlib** - Password hashing
- **bcrypt** - Hashing algorithm

### Machine Learning
- **scikit-learn** 1.8.0 - Tabular models
- **XGBoost** 3.2.0 - Gradient boosting
- **PyTorch** 2.12.0 - Deep learning
- **TorchVision** 0.27.0 - Image models
- **imbalanced-learn** 0.14.1 - SMOTE

### Frontend
- **HTML5** - Structure
- **CSS3** - Styling
- **Bootstrap 5** - UI framework
- **JavaScript** - Interactivity
- **Font Awesome** - Icons

### Data Processing
- **Pandas** 3.0.3 - Data manipulation
- **NumPy** 2.4.6 - Numerical computing
- **Pillow** 12.2.0 - Image processing

## 📁 Project Structure

```
AI-Powered-Medical-Diagnosis-Assistant/
├── app.py                          # FastAPI application
├── database.py                     # Database configuration
├── models.py                       # SQLAlchemy models
├── auth.py                         # Authentication utilities
├── init_db.py                      # Database initialization script
├── requirements.txt                # Python dependencies
├── start_server.bat               # Windows startup script
├── README.md                      # This file
│
├── venv/                          # Virtual environment
│
├── models/                        # Trained ML models
│   ├── diabetes/
│   ├── heart_disease/
│   ├── kidney_disease/
│   ├── breast_cancer/
│   ├── liver_disease/
│   └── pneumonia/
│
├── templates/                     # HTML templates
│   ├── index.html                # Landing page
│   └── diseases/                 # Disease-specific pages
│
├── train_models/                  # Training scripts
└── docs/                          # Documentation
```

## 🔧 Configuration

### Database Configuration

Edit `database.py` to change database connection:

```python
DATABASE_URL = "postgresql://myuser:mypassword@localhost:5432/mydb"
```

### Authentication Configuration

Edit `auth.py` to change JWT settings:

```python
SECRET_KEY = "your-secret-key-change-this-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
```

**⚠️ Important**: Change the SECRET_KEY in production!

## 🧪 Testing

### Test Database Connection

```bash
python init_db.py
```

### Test API Endpoints

```bash
python test_pneumonia.py
```

### Manual API Testing

Use the interactive Swagger UI:
```
http://127.0.0.1:5000/docs
```

## 📈 Model Performance

| Disease | Type | Accuracy | AUC | Status |
|---------|------|----------|-----|--------|
| Diabetes | Tabular | ~85-90% | 0.87-0.92 | ✓ Active |
| Heart Disease | Tabular | ~85-92% | 0.88-0.94 | ✓ Active |
| Kidney Disease | Tabular | ~88-94% | 0.90-0.96 | ✓ Active |
| Breast Cancer | Tabular | ~96% | 0.99 | ✓ Active |
| Liver Disease | Tabular | ~85-90% | 0.88-0.93 | ✓ Active |
| Pneumonia | Image (CNN) | 86.54% | 0.96 | ✓ Active |

## 🐛 Troubleshooting

### Database Connection Error

**Problem**: Cannot connect to PostgreSQL  
**Solution**:
```bash
# Check if Docker is running
docker ps

# Start PostgreSQL container
docker start my-postgres

# Verify connection
docker exec my-postgres psql -U myuser -d mydb -c "SELECT 1;"
```

### Port Already in Use

**Problem**: Port 5000 or 5432 already in use  
**Solution**:
```bash
# Use different port for FastAPI
uvicorn app:app --host 127.0.0.1 --port 8000

# Or stop the conflicting service
```

### Models Not Loading

**Problem**: ML models not found  
**Solution**:
- All models are pre-trained and included in `models/` directory
- Check health endpoint: http://127.0.0.1:5000/health

### Authentication Errors

**Problem**: JWT token invalid or expired  
**Solution**:
- Tokens expire after 30 minutes
- Login again to get a new token
- Check SECRET_KEY in `auth.py`

## ⚠️ Important Notes

### Medical Disclaimer
This is an AI screening tool for **educational purposes only**. It should NOT replace professional medical diagnosis. Always consult with qualified healthcare providers for proper evaluation and treatment.

### Data Privacy
- User passwords are hashed with bcrypt
- Predictions are stored in database
- Anonymous predictions are supported (no user_id)
- No images are stored (only metadata)

### Security
- Change SECRET_KEY in production
- Use HTTPS in production
- Implement rate limiting
- Add input validation
- Regular security audits

## 🚀 Deployment

### Production Checklist

- [ ] Change SECRET_KEY in `auth.py`
- [ ] Use environment variables for sensitive data
- [ ] Enable HTTPS
- [ ] Set up proper database backups
- [ ] Configure CORS properly
- [ ] Add rate limiting
- [ ] Set up monitoring and logging
- [ ] Use production-grade ASGI server (Gunicorn + Uvicorn)
- [ ] Implement proper error handling
- [ ] Add database connection pooling

### Environment Variables

Create a `.env` file:
```
DATABASE_URL=postgresql://user:password@host:port/database
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

## 📞 Support

For issues or questions:
- Check README.md (this file)
- Check API documentation: http://127.0.0.1:5000/docs
- Run `python init_db.py` to verify database
- Run `python test_pneumonia.py` to verify API

---

**Built with ❤️ for better healthcare outcomes**

**Tech Stack**: FastAPI + PostgreSQL + PyTorch + XGBoost + Bootstrap
