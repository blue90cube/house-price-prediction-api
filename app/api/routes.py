"""
API Routes for House Price Prediction API.

This module defines all the API endpoints for the application.
"""

from datetime import datetime
from typing import List

from fastapi import APIRouter, HTTPException, Depends, status

from app.core.config import settings
from app.models.schemas import (
    PropertyInput,
    PredictionResponse,
    BatchPropertyInput,
    BatchPredictionResponse,
    HealthResponse,
    ErrorResponse,
    ModelInfoResponse
)
from app.services.predictor import get_prediction_service, PredictionService


# Create router
router = APIRouter()


def get_service() -> PredictionService:
    """Dependency to get prediction service."""
    service = get_prediction_service()
    if not service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Service is starting up."
        )
    return service


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the API is healthy and the model is loaded.",
    tags=["Health"]
)
async def health_check():
    """
    Health check endpoint for monitoring and keep-alive CRON jobs.
    
    Returns the current health status of the API including
    whether the ML model is loaded.
    """
    service = get_prediction_service()
    
    return HealthResponse(
        status="healthy" if service.is_loaded else "degraded",
        timestamp=datetime.utcnow(),
        model_loaded=service.is_loaded,
        version=settings.APP_VERSION
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict House Price",
    description="Predict the price of a single property based on its features.",
    tags=["Predictions"],
    responses={
        200: {"description": "Successful prediction"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        503: {"model": ErrorResponse, "description": "Model not loaded"}
    }
)
async def predict_price(
    property_data: PropertyInput,
    service: PredictionService = Depends(get_service)
):
    """
    Predict the price of a property.
    
    Takes property features as input and returns the predicted price.
    
    **Features:**
    - location: Geographical location of the property
    - size: Size in square feet
    - bedrooms: Number of bedrooms
    - bathrooms: Number of bathrooms
    - year_built: Year the property was constructed
    - condition: Property condition (e.g., Good, Excellent)
    - property_type: Type of property (e.g., Single Family)
    - date_sold: Optional sale date (defaults to today)
    """
    try:
        # Convert Pydantic model to dict
        data = property_data.model_dump()
        
        # Convert date to string if present
        if data.get('date_sold'):
            data['date_sold'] = data['date_sold'].isoformat()
        
        # Make prediction
        result = service.predict_single(data)
        
        return PredictionResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch Predict House Prices",
    description="Predict prices for multiple properties in a single request.",
    tags=["Predictions"],
    responses={
        200: {"description": "Successful batch prediction"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        503: {"model": ErrorResponse, "description": "Model not loaded"}
    }
)
async def predict_batch(
    batch_input: BatchPropertyInput,
    service: PredictionService = Depends(get_service)
):
    """
    Predict prices for multiple properties.
    
    Takes a list of properties and returns predictions for each.
    Uses multi-core processing for better performance with large batches.
    
    **Limits:**
    - Maximum batch size: 100 properties
    """
    try:
        # Check batch size
        if len(batch_input.properties) > settings.MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Batch size exceeds maximum of {settings.MAX_BATCH_SIZE}"
            )
        
        # Convert to list of dicts
        properties = []
        for prop in batch_input.properties:
            data = prop.model_dump()
            if data.get('date_sold'):
                data['date_sold'] = data['date_sold'].isoformat()
            properties.append(data)
        
        # Make predictions
        predictions, processing_time = service.predict_batch(properties)
        
        # Convert to response models
        prediction_responses = [PredictionResponse(**p) for p in predictions]
        
        return BatchPredictionResponse(
            predictions=prediction_responses,
            count=len(predictions),
            processing_time_ms=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Get Model Information",
    description="Get information about the loaded ML model.",
    tags=["Model"],
    responses={
        200: {"description": "Model information"},
        503: {"model": ErrorResponse, "description": "Model not loaded"}
    }
)
async def get_model_info(
    service: PredictionService = Depends(get_service)
):
    """
    Get information about the currently loaded model.
    
    Returns the model name, features used, and training metrics if available.
    """
    info = service.get_model_info()
    
    return ModelInfoResponse(
        model_name=info.get('model_name', 'Unknown'),
        features=info.get('features', []),
        training_metrics=info.get('training_metrics')
    )
