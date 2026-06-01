"""
Authentication utilities for user management
"""

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
import bcrypt
from fastapi import Depends, Request, Cookie
from sqlalchemy.orm import Session
from database import get_db
import models as db_models

# Security configuration
SECRET_KEY = "your-secret-key-change-this-in-production"  # Change in production!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    try:
        # Bcrypt expects bytes
        password_bytes = plain_password.encode("utf-8")
        hashed_bytes = (
            hashed_password.encode("utf-8")
            if isinstance(hashed_password, str)
            else hashed_password
        )
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception:
        return False


def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt"""
    # Bcrypt has a 72 byte limit, truncate if needed
    password_bytes = password.encode("utf-8")[:72]
    # Generate salt and hash
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    # Return as string for database storage
    return hashed.decode("utf-8")


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_user_by_username(db: Session, username: str):
    """Get user by username"""
    return db.query(db_models.User).filter(db_models.User.username == username).first()


def get_user_by_email(db: Session, email: str):
    """Get user by email"""
    return db.query(db_models.User).filter(db_models.User.email == email).first()


def authenticate_user(db: Session, username: str, password: str):
    """Authenticate user"""
    user = get_user_by_username(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
    access_token: Optional[str] = Cookie(None),
):
    """Get current user from cookie token (optional - returns None if not authenticated)"""
    # Try to get token from cookie
    token = access_token

    # If not in cookie, try Authorization header (for API calls)
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.replace("Bearer ", "")

    # If token is in "Bearer {token}" format from cookie, extract the token
    if token and token.startswith("Bearer "):
        token = token.replace("Bearer ", "")

    if not token:
        return None

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
    except JWTError:
        return None

    user = get_user_by_username(db, username=username)
    return user


def create_user(
    db: Session, username: str, email: str, password: str, full_name: str = None
):
    """Create a new user"""
    hashed_password = get_password_hash(password)
    db_user = db_models.User(
        username=username,
        email=email,
        hashed_password=hashed_password,
        full_name=full_name,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user
