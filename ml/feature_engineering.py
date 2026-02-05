"""
Feature Engineering Module for House Price Prediction

This module contains specialized feature engineering functions
that can be used independently or as part of the preprocessing pipeline.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from datetime import datetime


def extract_date_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Extract date-based features from a datetime column.
    
    Args:
        df: Input DataFrame
        date_column: Name of the date column
        
    Returns:
        DataFrame with new date features added
    """
    df = df.copy()
    
    if date_column not in df.columns:
        print(f"Warning: {date_column} not found in DataFrame")
        return df
    
    # Convert to datetime if needed
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    # Extract features
    df['sale_year'] = df[date_column].dt.year
    df['sale_month'] = df[date_column].dt.month
    df['sale_quarter'] = df[date_column].dt.quarter
    df['sale_day_of_week'] = df[date_column].dt.dayofweek
    df['sale_day_of_year'] = df[date_column].dt.dayofyear
    
    # Is weekend
    df['is_weekend_sale'] = df['sale_day_of_week'].isin([5, 6]).astype(int)
    
    # Season (Northern Hemisphere)
    df['sale_season'] = df['sale_month'].apply(lambda x: 
        'winter' if x in [12, 1, 2] else
        'spring' if x in [3, 4, 5] else
        'summer' if x in [6, 7, 8] else 'fall'
    )
    
    return df


def calculate_property_age(df: pd.DataFrame, 
                           year_built_col: str, 
                           reference_year_col: Optional[str] = None,
                           reference_year: Optional[int] = None) -> pd.DataFrame:
    """
    Calculate property age based on year built.
    
    Args:
        df: Input DataFrame
        year_built_col: Column name containing year built
        reference_year_col: Column to use as reference year (e.g., sale_year)
        reference_year: Fixed reference year if no column provided
        
    Returns:
        DataFrame with property_age column added
    """
    df = df.copy()
    
    if year_built_col not in df.columns:
        print(f"Warning: {year_built_col} not found in DataFrame")
        return df
    
    if reference_year_col and reference_year_col in df.columns:
        df['property_age'] = df[reference_year_col] - df[year_built_col]
    elif reference_year:
        df['property_age'] = reference_year - df[year_built_col]
    else:
        # Use current year as default
        df['property_age'] = datetime.now().year - df[year_built_col]
    
    # Clip negative values (data errors)
    df['property_age'] = df['property_age'].clip(lower=0)
    
    # Age categories
    df['age_category'] = pd.cut(
        df['property_age'],
        bins=[-1, 5, 10, 20, 50, 100, float('inf')],
        labels=['new', 'recent', 'modern', 'established', 'old', 'historic']
    )
    
    return df


def create_size_features(df: pd.DataFrame,
                         size_col: str,
                         bedrooms_col: str,
                         bathrooms_col: str) -> pd.DataFrame:
    """
    Create size-related features.
    
    Args:
        df: Input DataFrame
        size_col: Column name for property size
        bedrooms_col: Column name for number of bedrooms
        bathrooms_col: Column name for number of bathrooms
        
    Returns:
        DataFrame with new size features added
    """
    df = df.copy()
    
    # Validate columns exist
    required_cols = [size_col, bedrooms_col, bathrooms_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"Warning: Missing columns: {missing}")
        return df
    
    # Size per bedroom (avoid division by zero)
    df['size_per_bedroom'] = df[size_col] / df[bedrooms_col].replace(0, 1)
    
    # Size per bathroom
    df['size_per_bathroom'] = df[size_col] / df[bathrooms_col].replace(0, 1)
    
    # Bed-bath ratio
    df['bed_bath_ratio'] = df[bedrooms_col] / df[bathrooms_col].replace(0, 1)
    
    # Total rooms
    df['total_rooms'] = df[bedrooms_col] + df[bathrooms_col]
    
    # Size per room
    df['size_per_room'] = df[size_col] / df['total_rooms'].replace(0, 1)
    
    # Size categories (only if we have enough data points)
    if len(df) >= 4:
        size_percentiles = df[size_col].quantile([0.25, 0.5, 0.75])
        bins = [0, size_percentiles[0.25], size_percentiles[0.5], 
                size_percentiles[0.75], float('inf')]
        # Remove duplicate bins
        unique_bins = sorted(set(bins))
        if len(unique_bins) >= 3:
            labels = ['small', 'medium', 'large', 'very_large'][:len(unique_bins)-1]
            df['size_category'] = pd.cut(
                df[size_col],
                bins=unique_bins,
                labels=labels,
                duplicates='drop'
            )
        else:
            df['size_category'] = 'medium'
    else:
        df['size_category'] = 'medium'
    
    return df


def encode_location(df: pd.DataFrame,
                    location_col: str,
                    target_col: Optional[str] = None,
                    method: str = 'frequency') -> pd.DataFrame:
    """
    Encode location feature using various methods.
    
    Args:
        df: Input DataFrame
        location_col: Column name for location
        target_col: Target column for target encoding (optional)
        method: Encoding method - 'frequency', 'target', or 'label'
        
    Returns:
        DataFrame with encoded location
    """
    df = df.copy()
    
    if location_col not in df.columns:
        print(f"Warning: {location_col} not found in DataFrame")
        return df
    
    if method == 'frequency':
        # Frequency encoding
        freq_map = df[location_col].value_counts(normalize=True).to_dict()
        df['location_frequency'] = df[location_col].map(freq_map)
        
    elif method == 'target' and target_col and target_col in df.columns:
        # Target encoding (mean price by location)
        target_map = df.groupby(location_col)[target_col].mean().to_dict()
        df['location_target_encoded'] = df[location_col].map(target_map)
        
    elif method == 'label':
        # Label encoding based on sorted frequency
        labels = df[location_col].value_counts().index.tolist()
        label_map = {loc: i for i, loc in enumerate(labels)}
        df['location_label'] = df[location_col].map(label_map)
    
    return df


def create_interaction_features(df: pd.DataFrame,
                                feature_pairs: List[tuple]) -> pd.DataFrame:
    """
    Create interaction features between numeric columns.
    
    Args:
        df: Input DataFrame
        feature_pairs: List of tuples with column pairs to interact
        
    Returns:
        DataFrame with interaction features
    """
    df = df.copy()
    
    for col1, col2 in feature_pairs:
        if col1 in df.columns and col2 in df.columns:
            # Multiplication interaction
            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
            
            # Division interaction (if denominator is not zero)
            safe_col2 = df[col2].replace(0, 1)
            df[f'{col1}_div_{col2}'] = df[col1] / safe_col2
    
    return df


def handle_outliers(df: pd.DataFrame,
                    columns: List[str],
                    method: str = 'iqr',
                    threshold: float = 1.5) -> pd.DataFrame:
    """
    Handle outliers in specified columns.
    
    Args:
        df: Input DataFrame
        columns: List of columns to check for outliers
        method: 'iqr' for IQR method, 'zscore' for z-score method
        threshold: Multiplier for IQR (default 1.5) or z-score threshold (default 3)
        
    Returns:
        DataFrame with outliers handled
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Clip outliers
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
        elif method == 'zscore':
            mean = df[col].mean()
            std = df[col].std()
            z_scores = np.abs((df[col] - mean) / std)
            
            # Replace outliers with median
            median = df[col].median()
            df.loc[z_scores > threshold, col] = median
    
    return df


def apply_all_feature_engineering(df: pd.DataFrame,
                                  column_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Apply all feature engineering steps to the DataFrame.
    
    Args:
        df: Input DataFrame
        column_mapping: Dictionary mapping feature types to column names
        
    Returns:
        DataFrame with all engineered features
    """
    df = df.copy()
    
    # Date features
    if column_mapping.get('date_sold'):
        df = extract_date_features(df, column_mapping['date_sold'])
    
    # Property age
    if column_mapping.get('year_built'):
        reference_col = 'sale_year' if 'sale_year' in df.columns else None
        df = calculate_property_age(
            df, 
            column_mapping['year_built'],
            reference_year_col=reference_col
        )
    
    # Size features
    if all(column_mapping.get(k) for k in ['size', 'bedrooms', 'bathrooms']):
        df = create_size_features(
            df,
            column_mapping['size'],
            column_mapping['bedrooms'],
            column_mapping['bathrooms']
        )
    
    # Location encoding
    if column_mapping.get('location') and column_mapping.get('price'):
        df = encode_location(
            df,
            column_mapping['location'],
            target_col=column_mapping['price'],
            method='frequency'
        )
    
    return df


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Module")
    print("=" * 50)
    print("\nAvailable functions:")
    print("  - extract_date_features()")
    print("  - calculate_property_age()")
    print("  - create_size_features()")
    print("  - encode_location()")
    print("  - create_interaction_features()")
    print("  - handle_outliers()")
    print("  - apply_all_feature_engineering()")
