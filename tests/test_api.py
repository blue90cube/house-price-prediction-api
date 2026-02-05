"""
API Tests for House Price Prediction API.

This module contains tests for all API endpoints using pytest and FastAPI's TestClient.
"""

import os
import sys
import pytest
from datetime import date
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient


# Mock the prediction service before importing the app
@pytest.fixture(scope="module")
def mock_service():
    """Create a mock prediction service."""
    mock = Mock()
    mock.is_loaded = True
    mock.model_name = "TestModel"
    mock.predict_single.return_value = {
        'predicted_price': 350000.0,
        'currency': 'USD',
        'model_name': 'TestModel'
    }
    mock.predict_batch.return_value = (
        [
            {'predicted_price': 350000.0, 'currency': 'USD', 'model_name': 'TestModel'},
            {'predicted_price': 450000.0, 'currency': 'USD', 'model_name': 'TestModel'}
        ],
        25.5  # processing time in ms
    )
    mock.get_model_info.return_value = {
        'model_name': 'TestModel',
        'features': ['location', 'size', 'bedrooms', 'bathrooms'],
        'is_loaded': True,
        'training_metrics': {'MAE': 25000.0, 'R2': 0.85}
    }
    return mock


@pytest.fixture(scope="module")
def client(mock_service):
    """Create a test client with mocked prediction service."""
    with patch('app.services.predictor.get_prediction_service', return_value=mock_service):
        with patch('app.services.predictor.initialize_prediction_service', return_value=True):
            from app.main import app
            with TestClient(app) as test_client:
                yield test_client


class TestRootEndpoints:
    """Tests for root-level endpoints."""
    
    def test_root_endpoint(self, client):
        """Test the root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "running"
    
    def test_root_health_endpoint(self, client):
        """Test the root health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "model_loaded" in data


class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    def test_api_health_endpoint(self, client):
        """Test the API v1 health check endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "timestamp" in data
        assert "model_loaded" in data
        assert "version" in data


class TestPredictionEndpoints:
    """Tests for prediction endpoints."""
    
    def test_predict_single_valid_input(self, client):
        """Test single prediction with valid input."""
        payload = {
            "location": "Downtown",
            "size": 1500.0,
            "bedrooms": 3,
            "bathrooms": 2,
            "year_built": 2010,
            "condition": "Good",
            "property_type": "Single Family"
        }
        
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "predicted_price" in data
        assert data["predicted_price"] > 0
        assert "currency" in data
    
    def test_predict_single_with_date(self, client):
        """Test single prediction with date_sold."""
        payload = {
            "location": "Suburb",
            "size": 2000.0,
            "bedrooms": 4,
            "bathrooms": 3,
            "year_built": 2015,
            "condition": "Excellent",
            "property_type": "Single Family",
            "date_sold": "2024-01-15"
        }
        
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "predicted_price" in data
    
    def test_predict_single_invalid_size(self, client):
        """Test single prediction with invalid size (negative)."""
        payload = {
            "location": "Downtown",
            "size": -100,  # Invalid: negative size
            "bedrooms": 3,
            "bathrooms": 2,
            "year_built": 2010,
            "condition": "Good",
            "property_type": "Single Family"
        }
        
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_predict_single_invalid_year(self, client):
        """Test single prediction with invalid year_built."""
        payload = {
            "location": "Downtown",
            "size": 1500,
            "bedrooms": 3,
            "bathrooms": 2,
            "year_built": 1700,  # Invalid: too old
            "condition": "Good",
            "property_type": "Single Family"
        }
        
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422
    
    def test_predict_single_missing_field(self, client):
        """Test single prediction with missing required field."""
        payload = {
            "location": "Downtown",
            "size": 1500,
            # Missing: bedrooms, bathrooms, year_built, condition, property_type
        }
        
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422
    
    def test_predict_batch_valid_input(self, client):
        """Test batch prediction with valid input."""
        payload = {
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
        
        response = client.post("/api/v1/predict/batch", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert "count" in data
        assert data["count"] == 2
        assert len(data["predictions"]) == 2
    
    def test_predict_batch_empty_list(self, client):
        """Test batch prediction with empty list."""
        payload = {"properties": []}
        
        response = client.post("/api/v1/predict/batch", json=payload)
        assert response.status_code == 422  # Validation error - min_length=1


class TestModelEndpoints:
    """Tests for model information endpoints."""
    
    def test_model_info_endpoint(self, client):
        """Test model info endpoint."""
        response = client.get("/api/v1/model/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "model_name" in data
        assert "features" in data


class TestDocumentation:
    """Tests for documentation endpoints."""
    
    def test_swagger_docs(self, client):
        """Test Swagger UI is accessible."""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_redoc(self, client):
        """Test ReDoc is accessible."""
        response = client.get("/redoc")
        assert response.status_code == 200
    
    def test_openapi_schema(self, client):
        """Test OpenAPI schema is accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data


class TestInputValidation:
    """Tests for input validation."""
    
    def test_empty_location(self, client):
        """Test that empty location is rejected."""
        payload = {
            "location": "",  # Empty string
            "size": 1500,
            "bedrooms": 3,
            "bathrooms": 2,
            "year_built": 2010,
            "condition": "Good",
            "property_type": "Single Family"
        }
        
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422
    
    def test_negative_bedrooms(self, client):
        """Test that negative bedrooms is rejected."""
        payload = {
            "location": "Downtown",
            "size": 1500,
            "bedrooms": -1,  # Invalid
            "bathrooms": 2,
            "year_built": 2010,
            "condition": "Good",
            "property_type": "Single Family"
        }
        
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422
    
    def test_excessive_bedrooms(self, client):
        """Test that excessive bedrooms is rejected."""
        payload = {
            "location": "Downtown",
            "size": 1500,
            "bedrooms": 100,  # Too many
            "bathrooms": 2,
            "year_built": 2010,
            "condition": "Good",
            "property_type": "Single Family"
        }
        
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422
    
    def test_future_year_built(self, client):
        """Test that future year_built is rejected."""
        payload = {
            "location": "Downtown",
            "size": 1500,
            "bedrooms": 3,
            "bathrooms": 2,
            "year_built": 2050,  # Future year
            "condition": "Good",
            "property_type": "Single Family"
        }
        
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422
    
    def test_invalid_date_format(self, client):
        """Test that invalid date format is rejected."""
        payload = {
            "location": "Downtown",
            "size": 1500,
            "bedrooms": 3,
            "bathrooms": 2,
            "year_built": 2010,
            "condition": "Good",
            "property_type": "Single Family",
            "date_sold": "invalid-date"
        }
        
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422


# Additional tests without mocking (integration tests)
class TestIntegration:
    """Integration tests that can run with the actual model if available."""
    
    @pytest.mark.skipif(
        not os.path.exists(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "artifacts", "model.joblib"
        )),
        reason="Model file not found - skip integration tests"
    )
    def test_real_prediction(self):
        """Test with actual model (if available)."""
        from app.main import app
        
        with TestClient(app) as client:
            payload = {
                "location": "Downtown",
                "size": 1500.0,
                "bedrooms": 3,
                "bathrooms": 2,
                "year_built": 2010,
                "condition": "Good",
                "property_type": "Single Family"
            }
            
            response = client.post("/api/v1/predict", json=payload)
            
            # May fail if model is not loaded, but should not crash
            assert response.status_code in [200, 503]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
