"""
Configuration settings for the House Price Prediction API.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Settings
    APP_NAME: str = "House Price Prediction API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "API for predicting house prices using machine learning"
    DEBUG: bool = False
    
    # Model Settings
    MODEL_PATH: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "artifacts",
        "model.joblib"
    )
    PREPROCESSOR_PATH: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "artifacts",
        "preprocessor.joblib"
    )
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS Settings
    CORS_ORIGINS: list = ["*"]
    
    # Batch Processing Settings
    MAX_BATCH_SIZE: int = 100
    BATCH_WORKERS: int = 4
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings
