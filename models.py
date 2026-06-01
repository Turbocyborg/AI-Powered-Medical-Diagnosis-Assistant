"""
Database models for storing predictions and user data
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean
from sqlalchemy.sql import func
from database import Base


class User(Base):
    """User model for authentication"""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class Prediction(Base):
    """Prediction model for storing all disease predictions"""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, index=True, nullable=True
    )  # Nullable for anonymous predictions
    disease_type = Column(
        String, index=True, nullable=False
    )  # diabetes, heart_disease, etc.

    # Input data (stored as JSON)
    input_data = Column(JSON, nullable=False)

    # Prediction results
    prediction = Column(Integer, nullable=False)  # 0 or 1
    probability = Column(Float, nullable=False)  # Probability score
    risk_level = Column(String, nullable=False)  # Low, Moderate, High

    # Additional metadata
    model_version = Column(String, default="1.0")
    ip_address = Column(String)
    user_agent = Column(String)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "disease_type": self.disease_type,
            "input_data": self.input_data,
            "prediction": self.prediction,
            "probability": self.probability,
            "risk_level": self.risk_level,
            "model_version": self.model_version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class PneumoniaPrediction(Base):
    """Separate model for pneumonia predictions (image-based)"""

    __tablename__ = "pneumonia_predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=True)

    # Image metadata
    image_filename = Column(String)
    image_size = Column(Integer)  # Size in bytes

    # Prediction results
    prediction = Column(Integer, nullable=False)  # 0 = Normal, 1 = Pneumonia
    probability = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    risk_level = Column(String, nullable=False)

    # Additional metadata
    model_version = Column(String, default="1.0")
    ip_address = Column(String)
    user_agent = Column(String)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "image_filename": self.image_filename,
            "image_size": self.image_size,
            "prediction": self.prediction,
            "probability": self.probability,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "model_version": self.model_version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
