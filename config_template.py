# Configuration file for diabetes prediction system
# Copy this to config.py and modify as needed

class Config:
    """Base configuration"""
    SECRET_KEY = 'your-secret-key-here'
    DEBUG = False
    TESTING = False

    # Model paths
    MODEL_PATH = 'models/diabetes_model.pkl'
    SCALER_PATH = 'models/scaler.pkl'

    # Dataset configuration
    DATASET_PATH = 'data/diabetes_data.csv'
    FEATURES_CSV = 'brfss_features.csv'

    # ML Configuration
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5

    # Flask configuration
    HOST = '0.0.0.0'
    PORT = 5000

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    # Add production-specific settings

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
