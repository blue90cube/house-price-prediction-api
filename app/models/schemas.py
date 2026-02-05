"""
Pydantic models/schemas for request and response validation.
"""

from datetime import date, datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


class PropertyInput(BaseModel):
    """Schema for property input data."""
    
    location: str = Field(
        ..., 
        description="Geographical location of the property",
        min_length=1,
        max_length=200,
        examples=["Downtown", "Suburb"]
    )
    size: float = Field(
        ..., 
        gt=0, 
        description="Size of the property in square feet",
        examples=[1500.0, 2000.0]
    )
    bedrooms: int = Field(
        ..., 
        ge=0, 
        le=50, 
        description="Number of bedrooms",
        examples=[3, 4]
    )
    bathrooms: int = Field(
        ..., 
        ge=0, 
        le=50, 
        description="Number of bathrooms",
        examples=[2, 3]
    )
    year_built: int = Field(
        ..., 
        ge=1800, 
        le=2030, 
        description="Year the property was built",
        examples=[2010, 2015]
    )
    condition: str = Field(
        ..., 
        description="Condition of the property",
        min_length=1,
        max_length=100,
        examples=["Good", "Excellent", "Fair"]
    )
    property_type: str = Field(
        ..., 
        description="Type of property",
        min_length=1,
        max_length=100,
        examples=["Single Family", "Condo", "Townhouse"]
    )
    date_sold: Optional[date] = Field(
        default=None,
        description="Date of sale (defaults to today if not provided)",
        examples=["2024-01-15"]
    )
    
    @field_validator('location', 'condition', 'property_type')
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        """Strip whitespace from string fields."""
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "location": "Downtown",
                "size": 1500.0,
                "bedrooms": 3,
                "bathrooms": 2,
                "year_built": 2010,
                "condition": "Good",
                "property_type": "Single Family",
                "date_sold": "2024-01-15"
            }
        }


class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    
    predicted_price: float = Field(
        ..., 
        description="Predicted price of the property",
        examples=[350000.0]
    )
    currency: str = Field(
        default="USD",
        description="Currency of the predicted price"
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Name of the model used for prediction"
    )
    confidence_interval: Optional[Dict[str, float]] = Field(
        default=None,
        description="Confidence interval for the prediction"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "predicted_price": 350000.0,
                "currency": "USD",
                "model_name": "XGBoost",
                "confidence_interval": {
                    "lower": 320000.0,
                    "upper": 380000.0
                }
            }
        }


class BatchPropertyInput(BaseModel):
    """Schema for batch prediction input."""
    
    properties: List[PropertyInput] = Field(
        ...,
        description="List of properties to predict prices for",
        min_length=1,
        max_length=100
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "properties": [
                    {
                        "location": "Downtown",
                        "size": 1500.0,
                        "bedrooms": 3,
                        "bathrooms": 2,
                        "year_built": 2010,
                        "condition": "Good",
                        "property_type": "Single Family"
                    },
                    {
                        "location": "Suburb",
                        "size": 2000.0,
                        "bedrooms": 4,
                        "bathrooms": 3,
                        "year_built": 2015,
                        "condition": "Excellent",
                        "property_type": "Single Family"
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response."""
    
    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of predictions for each property"
    )
    count: int = Field(
        ...,
        description="Number of predictions made"
    )
    processing_time_ms: Optional[float] = Field(
        default=None,
        description="Time taken to process the batch in milliseconds"
    )


class HealthResponse(BaseModel):
    """Schema for health check response."""
    
    status: str = Field(
        ...,
        description="Health status of the API",
        examples=["healthy"]
    )
    timestamp: datetime = Field(
        ...,
        description="Timestamp of the health check"
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the ML model is loaded"
    )
    version: str = Field(
        ...,
        description="API version"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00Z",
                "model_loaded": True,
                "version": "1.0.0"
            }
        }


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    
    error: str = Field(
        ...,
        description="Error type"
    )
    message: str = Field(
        ...,
        description="Error message"
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid input data",
                "details": {
                    "field": "size",
                    "issue": "must be greater than 0"
                }
            }
        }


class ModelInfoResponse(BaseModel):
    """Schema for model information response."""
    
    model_name: str = Field(
        ...,
        description="Name of the loaded model"
    )
    features: List[str] = Field(
        ...,
        description="List of features used by the model"
    )
    training_metrics: Optional[Dict[str, float]] = Field(
        default=None,
        description="Training metrics of the model"
    )
