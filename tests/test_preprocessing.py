"""
Tests for Data Preprocessing Module.

This module contains tests for the data preprocessing functions.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.feature_engineering import (
    extract_date_features,
    calculate_property_age,
    create_size_features,
    encode_location,
    handle_outliers
)


class TestDateFeatures:
    """Tests for date feature extraction."""
    
    def test_extract_date_features(self):
        """Test date feature extraction."""
        df = pd.DataFrame({
            'date_sold': pd.to_datetime(['2024-01-15', '2024-06-20', '2024-12-25'])
        })
        
        result = extract_date_features(df, 'date_sold')
        
        assert 'sale_year' in result.columns
        assert 'sale_month' in result.columns
        assert 'sale_quarter' in result.columns
        
        assert result['sale_year'].iloc[0] == 2024
        assert result['sale_month'].iloc[0] == 1
        assert result['sale_quarter'].iloc[0] == 1
    
    def test_extract_date_features_missing_column(self):
        """Test date extraction with missing column."""
        df = pd.DataFrame({'other_column': [1, 2, 3]})
        
        result = extract_date_features(df, 'date_sold')
        
        # Should return unchanged DataFrame
        assert 'sale_year' not in result.columns


class TestPropertyAge:
    """Tests for property age calculation."""
    
    def test_calculate_property_age(self):
        """Test property age calculation."""
        df = pd.DataFrame({
            'year_built': [2000, 2010, 1990],
            'sale_year': [2024, 2024, 2024]
        })
        
        result = calculate_property_age(df, 'year_built', reference_year_col='sale_year')
        
        assert 'property_age' in result.columns
        assert result['property_age'].iloc[0] == 24
        assert result['property_age'].iloc[1] == 14
        assert result['property_age'].iloc[2] == 34
    
    def test_calculate_property_age_fixed_year(self):
        """Test property age with fixed reference year."""
        df = pd.DataFrame({
            'year_built': [2000, 2010, 1990]
        })
        
        result = calculate_property_age(df, 'year_built', reference_year=2024)
        
        assert result['property_age'].iloc[0] == 24
    
    def test_calculate_property_age_negative_clipping(self):
        """Test that negative ages are clipped to 0."""
        df = pd.DataFrame({
            'year_built': [2030],  # Future year
            'sale_year': [2024]
        })
        
        result = calculate_property_age(df, 'year_built', reference_year_col='sale_year')
        
        assert result['property_age'].iloc[0] == 0  # Should be clipped to 0


class TestSizeFeatures:
    """Tests for size-based features."""
    
    def test_create_size_features(self):
        """Test size feature creation."""
        df = pd.DataFrame({
            'size': [1500, 2000, 1000],
            'bedrooms': [3, 4, 2],
            'bathrooms': [2, 3, 1]
        })
        
        result = create_size_features(df, 'size', 'bedrooms', 'bathrooms')
        
        assert 'size_per_bedroom' in result.columns
        assert 'bed_bath_ratio' in result.columns
        assert 'total_rooms' in result.columns
        
        assert result['size_per_bedroom'].iloc[0] == 500  # 1500/3
        assert result['total_rooms'].iloc[0] == 5  # 3+2
    
    def test_create_size_features_zero_bedrooms(self):
        """Test size features with zero bedrooms (avoid division by zero)."""
        df = pd.DataFrame({
            'size': [1500],
            'bedrooms': [0],
            'bathrooms': [2]
        })
        
        result = create_size_features(df, 'size', 'bedrooms', 'bathrooms')
        
        # Should handle division by zero gracefully
        assert result['size_per_bedroom'].iloc[0] == 1500  # 1500/1 (0 replaced with 1)


class TestLocationEncoding:
    """Tests for location encoding."""
    
    def test_frequency_encoding(self):
        """Test frequency encoding of location."""
        df = pd.DataFrame({
            'location': ['A', 'A', 'A', 'B', 'B', 'C'],
            'price': [100, 110, 120, 200, 210, 300]
        })
        
        result = encode_location(df, 'location', method='frequency')
        
        assert 'location_frequency' in result.columns
        assert result['location_frequency'].iloc[0] == 0.5  # A appears 3/6 times
    
    def test_target_encoding(self):
        """Test target encoding of location."""
        df = pd.DataFrame({
            'location': ['A', 'A', 'B', 'B'],
            'price': [100, 200, 300, 400]
        })
        
        result = encode_location(df, 'location', target_col='price', method='target')
        
        assert 'location_target_encoded' in result.columns
        assert result['location_target_encoded'].iloc[0] == 150  # Mean of A


class TestOutlierHandling:
    """Tests for outlier handling."""
    
    def test_handle_outliers_iqr(self):
        """Test IQR-based outlier handling."""
        df = pd.DataFrame({
            'value': [10, 20, 30, 40, 50, 1000]  # 1000 is an outlier
        })
        
        result = handle_outliers(df, ['value'], method='iqr')
        
        # Outlier should be clipped
        assert result['value'].max() < 1000
    
    def test_handle_outliers_missing_column(self):
        """Test outlier handling with missing column."""
        df = pd.DataFrame({
            'other': [1, 2, 3]
        })
        
        result = handle_outliers(df, ['missing_column'], method='iqr')
        
        # Should return unchanged DataFrame
        assert list(result.columns) == ['other']


class TestDataValidation:
    """Tests for data validation."""
    
    def test_dataframe_not_modified_in_place(self):
        """Test that original DataFrame is not modified."""
        original_df = pd.DataFrame({
            'date_sold': pd.to_datetime(['2024-01-15']),
            'year_built': [2000]
        })
        
        original_values = original_df.copy()
        
        _ = extract_date_features(original_df, 'date_sold')
        
        # Original should be unchanged
        pd.testing.assert_frame_equal(original_df, original_values)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
