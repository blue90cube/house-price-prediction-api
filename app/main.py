"""
Main FastAPI Application for House Price Prediction API.

This module initializes and configures the FastAPI application,
including startup/shutdown events and middleware.
"""

import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.api.routes import router
from app.services.predictor import initialize_prediction_service, get_prediction_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events:
    - On startup: Load the ML model
    - On shutdown: Cleanup resources
    """
    # Startup
    print("=" * 60)
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    print("=" * 60)
    
    # Load ML model
    print("\nLoading ML model...")
    success = initialize_prediction_service()
    
    if success:
        print("Model loaded successfully!")
    else:
        print("WARNING: Model failed to load. API will start but predictions will fail.")
        print("Please ensure the model file exists at:", settings.MODEL_PATH)
    
    print("\n" + "=" * 60)
    print("Application ready to accept requests")
    print("=" * 60 + "\n")
    
    yield
    
    # Shutdown
    print("\nShutting down application...")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed messages."""
    errors = []
    for error in exc.errors():
        errors.append({
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "ValidationError",
            "message": "Invalid request data",
            "details": errors
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": str(exc) if settings.DEBUG else "An unexpected error occurred",
            "details": None
        }
    )


# Include API routes
app.include_router(router, prefix="/api/v1")


# Root endpoint
@app.get(
    "/",
    summary="Root",
    description="Root endpoint with API information",
    tags=["Root"]
)
async def root():
    """
    Root endpoint providing basic API information.
    
    Useful for quick verification that the API is running.
    """
    service = get_prediction_service()
    
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": settings.APP_DESCRIPTION,
        "status": "running",
        "model_loaded": service.is_loaded,
        "docs_url": "/docs",
        "health_url": "/api/v1/health",
        "timestamp": datetime.utcnow().isoformat()
    }


# Health check at root level (for Render)
@app.get(
    "/health",
    summary="Health Check (Root)",
    description="Root-level health check endpoint",
    tags=["Health"]
)
async def root_health():
    """
    Root-level health check for deployment platforms.
    
    This endpoint is useful for platforms like Render that
    check health at a specific path.
    """
    service = get_prediction_service()
    
    return {
        "status": "healthy" if service.is_loaded else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": service.is_loaded,
        "version": settings.APP_VERSION
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
