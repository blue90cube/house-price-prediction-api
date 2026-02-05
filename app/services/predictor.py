"""
Prediction Service for House Price Prediction API.
Updated to handle engineered features from advanced training.
"""

import os
import time
from datetime import date, datetime
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import joblib

from app.core.config import settings


class PredictionService:
    """Service class for handling house price predictions."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.MODEL_PATH
        self.model_data = None
        self.pipeline = None
        self.model_name = None
        self.is_loaded = False
        
        # Default values for location/condition/type based aggregates
        # These will be used when we don't have historical data
        self.default_loc_price_mean = 500000
        self.default_loc_price_median = 450000
        self.default_loc_count = 100
        self.default_type_price_mean = 500000
    
    def load_model(self) -> bool:
        """Load the trained model from disk."""
        try:
            if not os.path.exists(self.model_path):
                print(f"Model file not found: {self.model_path}")
                return False
            
            self.model_data = joblib.load(self.model_path)
            self.pipeline = self.model_data['pipeline']
            self.model_name = self.model_data.get('model_name', 'Unknown')
            self.is_loaded = True
            
            print(f"Model loaded successfully: {self.model_name}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.is_loaded = False
            return False
    
    def _prepare_features(self, property_data: Dict) -> pd.DataFrame:
        """Prepare features from property data with all engineered features."""
        
        # Extract basic inputs
        location = property_data.get('location', 'Unknown')
        size = float(property_data.get('size', 1500))
        bedrooms = int(property_data.get('bedrooms', 3))
        bathrooms = int(property_data.get('bathrooms', 2))
        year_built = int(property_data.get('year_built', 2000))
        condition = property_data.get('condition', 'Good')
        prop_type = property_data.get('property_type', 'Single Family')
        
        # Handle date
        date_sold = property_data.get('date_sold')
        if date_sold is None:
            sale_date = date.today()
        elif isinstance(date_sold, str):
            sale_date = datetime.strptime(date_sold, '%Y-%m-%d').date()
        elif isinstance(date_sold, date):
            sale_date = date_sold
        else:
            sale_date = date.today()
        
        sale_year = sale_date.year
        sale_month = sale_date.month
        sale_quarter = (sale_month - 1) // 3 + 1
        
        # Calculate property age
        property_age = max(0, sale_year - year_built)
        
        # Room-based features
        total_rooms = bedrooms + bathrooms
        size_per_bedroom = size / max(bedrooms, 1)
        size_per_bathroom = size / max(bathrooms, 1)
        bed_bath_ratio = bedrooms / max(bathrooms, 1)
        size_per_room = size / max(total_rooms, 1)
        
        # Log size
        log_size = np.log1p(size)
        
        # Age indicators
        is_new = 1 if property_age <= 5 else 0
        is_old = 1 if property_age > 50 else 0
        
        # Cyclical month encoding
        month_sin = np.sin(2 * np.pi * sale_month / 12)
        month_cos = np.cos(2 * np.pi * sale_month / 12)
        
        # Condition encoding
        cond_map = {'Poor': 1, 'Fair': 2, 'Average': 3, 'Good': 4, 'Excellent': 5}
        condition_num = cond_map.get(condition, 3)
        
        # Interaction features
        size_x_condition = size * condition_num
        age_x_condition = property_age * condition_num
        
        # Location/type aggregate features (use defaults for new predictions)
        loc_price_mean = self.default_loc_price_mean
        loc_price_median = self.default_loc_price_median
        loc_count = self.default_loc_count
        type_price_mean = self.default_type_price_mean
        
        # Create DataFrame with all features in the expected order
        data = {
            'size': size,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'year_built': year_built,
            'sale_year': sale_year,
            'sale_month': sale_month,
            'sale_quarter': sale_quarter,
            'property_age': property_age,
            'total_rooms': total_rooms,
            'size_per_bedroom': size_per_bedroom,
            'size_per_bathroom': size_per_bathroom,
            'bed_bath_ratio': bed_bath_ratio,
            'size_per_room': size_per_room,
            'log_size': log_size,
            'is_new': is_new,
            'is_old': is_old,
            'month_sin': month_sin,
            'month_cos': month_cos,
            'loc_price_mean': loc_price_mean,
            'loc_price_median': loc_price_median,
            'loc_count': loc_count,
            'condition_num': condition_num,
            'type_price_mean': type_price_mean,
            'size_x_condition': size_x_condition,
            'age_x_condition': age_x_condition,
            'location': location,
            'condition': condition,
            'type': prop_type
        }
        
        return pd.DataFrame([data])
    
    def predict_single(self, property_data: Dict) -> Dict:
        """Make a prediction for a single property."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Prepare features
        df = self._prepare_features(property_data)
        
        # Make prediction
        prediction = self.pipeline.predict(df)[0]
        
        # Ensure prediction is positive
        prediction = max(0, prediction)
        
        return {
            'predicted_price': round(float(prediction), 2),
            'currency': 'USD',
            'model_name': self.model_name
        }
    
    def predict_batch(self, properties: List[Dict], 
                      use_multiprocessing: bool = True) -> List[Dict]:
        """Make predictions for multiple properties."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        if use_multiprocessing and len(properties) > 10:
            max_workers = min(settings.BATCH_WORKERS, len(properties))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                predictions = list(executor.map(self.predict_single, properties))
        else:
            predictions = [self.predict_single(prop) for prop in properties]
        
        processing_time = (time.time() - start_time) * 1000
        
        return predictions, processing_time
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_loaded:
            return {
                'model_name': None,
                'features': [],
                'is_loaded': False
            }
        
        info = {
            'model_name': self.model_name,
            'is_loaded': True,
            'features': []
        }
        
        # Add test metrics if available
        if self.model_data and 'test_metrics' in self.model_data:
            info['test_metrics'] = self.model_data['test_metrics']
        
        if self.model_data and 'cv_metrics' in self.model_data:
            info['cv_metrics'] = self.model_data['cv_metrics']
        
        return info


# Global prediction service instance
_prediction_service: Optional[PredictionService] = None


def get_prediction_service() -> PredictionService:
    """Get or create the global prediction service instance."""
    global _prediction_service
    
    if _prediction_service is None:
        _prediction_service = PredictionService()
    
    return _prediction_service


def initialize_prediction_service() -> bool:
    """Initialize the global prediction service and load the model."""
    service = get_prediction_service()
    return service.load_model()
