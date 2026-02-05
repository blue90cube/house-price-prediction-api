"""
Data Preprocessing Module for House Price Prediction

This module contains functions for loading, cleaning, and preprocessing
the house price dataset for model training.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os


class DataPreprocessor:
    """
    A class to handle data loading, cleaning, and preprocessing
    for the house price prediction model.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the DataPreprocessor.
        
        Args:
            data_path: Path to the Excel data file
        """
        self.data_path = data_path
        self.df = None
        self.preprocessor = None
        self.feature_columns = None
        self.target_column = None
        
        # Define column mappings (will be updated based on actual data)
        self.column_mapping = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from Excel file.
        
        Returns:
            DataFrame with loaded data
        """
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_excel(self.data_path, engine='openpyxl')
        print(f"Loaded {len(self.df):,} records with {len(self.df.columns)} columns")
        
        # Standardize column names
        self.df.columns = self.df.columns.str.strip().str.lower().str.replace(' ', '_')
        print(f"Columns: {list(self.df.columns)}")
        
        return self.df
    
    def identify_columns(self) -> dict:
        """
        Identify and map important columns in the dataset.
        
        Returns:
            Dictionary mapping column types to actual column names
        """
        columns = self.df.columns.tolist()
        
        # Find key columns
        self.column_mapping = {
            'property_id': self._find_column(columns, ['property_id', 'propertyid', 'id']),
            'location': self._find_column(columns, ['location', 'city', 'neighborhood', 'area']),
            'size': self._find_column(columns, ['size', 'sqft', 'square_feet', 'area_sqft']),
            'bedrooms': self._find_column(columns, ['bedrooms', 'bedroom', 'beds', 'num_bedrooms']),
            'bathrooms': self._find_column(columns, ['bathrooms', 'bathroom', 'baths', 'num_bathrooms']),
            'year_built': self._find_column(columns, ['year_built', 'yearbuilt', 'built_year', 'construction_year']),
            'condition': self._find_column(columns, ['condition', 'property_condition', 'state']),
            'property_type': self._find_column(columns, ['type', 'property_type', 'home_type', 'house_type']),
            'date_sold': self._find_column(columns, ['date_sold', 'datesold', 'sale_date', 'sold_date']),
            'price': self._find_column(columns, ['price', 'sale_price', 'sold_price', 'selling_price'])
        }
        
        print("\nColumn Mapping:")
        for key, value in self.column_mapping.items():
            print(f"  {key}: {value}")
        
        self.target_column = self.column_mapping['price']
        return self.column_mapping
    
    def _find_column(self, columns: List[str], possible_names: List[str]) -> Optional[str]:
        """
        Find a column by checking possible name variations.
        
        Args:
            columns: List of actual column names
            possible_names: List of possible column name variations
            
        Returns:
            Matched column name or None
        """
        for col in columns:
            for name in possible_names:
                if name in col.lower():
                    return col
        return None
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and duplicates.
        
        Returns:
            Cleaned DataFrame
        """
        print("\nCleaning data...")
        initial_rows = len(self.df)
        
        # Drop duplicates based on property ID if available
        if self.column_mapping['property_id']:
            self.df = self.df.drop_duplicates(subset=[self.column_mapping['property_id']])
            print(f"  Removed {initial_rows - len(self.df)} duplicate rows")
        
        # Handle missing values
        missing_before = self.df.isnull().sum().sum()
        
        # For numerical columns, fill with median
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        
        # For categorical columns, fill with mode
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        
        missing_after = self.df.isnull().sum().sum()
        print(f"  Handled {missing_before - missing_after} missing values")
        
        # Remove rows with zero or negative prices
        if self.column_mapping['price']:
            price_col = self.column_mapping['price']
            invalid_prices = len(self.df[self.df[price_col] <= 0])
            self.df = self.df[self.df[price_col] > 0]
            if invalid_prices > 0:
                print(f"  Removed {invalid_prices} rows with invalid prices")
        
        print(f"  Final dataset size: {len(self.df):,} rows")
        return self.df
    
    def engineer_features(self) -> pd.DataFrame:
        """
        Create engineered features from existing columns.
        
        Returns:
            DataFrame with engineered features
        """
        print("\nEngineering features...")
        
        # Process date column
        date_col = self.column_mapping['date_sold']
        year_built_col = self.column_mapping['year_built']
        
        if date_col and date_col in self.df.columns:
            self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce')
            self.df['sale_year'] = self.df[date_col].dt.year
            self.df['sale_month'] = self.df[date_col].dt.month
            self.df['sale_quarter'] = self.df[date_col].dt.quarter
            print("  Created: sale_year, sale_month, sale_quarter")
        
        # Calculate property age
        if year_built_col and 'sale_year' in self.df.columns:
            self.df['property_age'] = self.df['sale_year'] - self.df[year_built_col]
            # Handle negative ages (data errors)
            self.df['property_age'] = self.df['property_age'].clip(lower=0)
            print("  Created: property_age")
        
        # Size-based features
        size_col = self.column_mapping['size']
        bedroom_col = self.column_mapping['bedrooms']
        bathroom_col = self.column_mapping['bathrooms']
        
        if size_col and bedroom_col:
            # Size per bedroom (avoid division by zero)
            self.df['size_per_bedroom'] = self.df[size_col] / (self.df[bedroom_col].replace(0, 1))
            print("  Created: size_per_bedroom")
        
        if bedroom_col and bathroom_col:
            # Bed-bath ratio
            self.df['bed_bath_ratio'] = self.df[bedroom_col] / (self.df[bathroom_col].replace(0, 1))
            print("  Created: bed_bath_ratio")
        
        # Total rooms
        if bedroom_col and bathroom_col:
            self.df['total_rooms'] = self.df[bedroom_col] + self.df[bathroom_col]
            print("  Created: total_rooms")
        
        return self.df
    
    def get_feature_columns(self) -> Tuple[List[str], List[str]]:
        """
        Identify numerical and categorical feature columns.
        
        Returns:
            Tuple of (numerical_columns, categorical_columns)
        """
        # Columns to exclude from features
        exclude_cols = [
            self.column_mapping['property_id'],
            self.column_mapping['price'],
            self.column_mapping['date_sold']
        ]
        exclude_cols = [col for col in exclude_cols if col is not None]
        
        # Get all columns except excluded
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        # Separate numerical and categorical
        numerical_cols = []
        categorical_cols = []
        
        for col in feature_cols:
            if self.df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                numerical_cols.append(col)
            elif self.df[col].dtype == 'object':
                categorical_cols.append(col)
        
        # Remove datetime columns from numerical
        numerical_cols = [col for col in numerical_cols if not pd.api.types.is_datetime64_any_dtype(self.df[col])]
        
        print(f"\nNumerical features ({len(numerical_cols)}): {numerical_cols}")
        print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
        
        return numerical_cols, categorical_cols
    
    def create_preprocessor(self, numerical_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
        """
        Create a sklearn ColumnTransformer for preprocessing.
        
        Args:
            numerical_cols: List of numerical column names
            categorical_cols: List of categorical column names
            
        Returns:
            Configured ColumnTransformer
        """
        # Numerical pipeline: impute + scale
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline: impute + one-hot encode
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine into ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, numerical_cols),
                ('cat', categorical_pipeline, categorical_cols)
            ],
            remainder='drop'
        )
        
        self.feature_columns = numerical_cols + categorical_cols
        
        return self.preprocessor
    
    def prepare_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Prepare data for model training.
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, preprocessor)
        """
        # Get feature and target
        numerical_cols, categorical_cols = self.get_feature_columns()
        
        X = self.df[numerical_cols + categorical_cols].copy()
        y = self.df[self.column_mapping['price']].copy()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\nData split:")
        print(f"  Training set: {len(X_train):,} samples")
        print(f"  Test set: {len(X_test):,} samples")
        
        # Create and fit preprocessor
        preprocessor = self.create_preprocessor(numerical_cols, categorical_cols)
        
        return X_train, X_test, y_train, y_test, preprocessor
    
    def save_preprocessor(self, path: str):
        """
        Save the fitted preprocessor to disk.
        
        Args:
            path: Path to save the preprocessor
        """
        if self.preprocessor is not None:
            joblib.dump({
                'preprocessor': self.preprocessor,
                'feature_columns': self.feature_columns,
                'column_mapping': self.column_mapping
            }, path)
            print(f"Preprocessor saved to {path}")
        else:
            print("Warning: No preprocessor to save. Run prepare_data() first.")
    
    @staticmethod
    def load_preprocessor(path: str) -> dict:
        """
        Load a saved preprocessor from disk.
        
        Args:
            path: Path to the saved preprocessor
            
        Returns:
            Dictionary containing preprocessor and metadata
        """
        return joblib.load(path)


def preprocess_for_prediction(data: dict, preprocessor_path: str) -> np.ndarray:
    """
    Preprocess a single data point for prediction.
    
    Args:
        data: Dictionary with feature values
        preprocessor_path: Path to saved preprocessor
        
    Returns:
        Preprocessed feature array
    """
    # Load preprocessor
    saved_data = DataPreprocessor.load_preprocessor(preprocessor_path)
    preprocessor = saved_data['preprocessor']
    feature_columns = saved_data['feature_columns']
    
    # Create DataFrame from input
    df = pd.DataFrame([data])
    
    # Apply feature engineering if needed
    if 'date_sold' in data:
        date = pd.to_datetime(data['date_sold'])
        df['sale_year'] = date.year
        df['sale_month'] = date.month
        df['sale_quarter'] = date.quarter
        
        if 'year_built' in data:
            df['property_age'] = date.year - data['year_built']
    
    if 'size' in data and 'bedrooms' in data:
        df['size_per_bedroom'] = data['size'] / max(data['bedrooms'], 1)
    
    if 'bedrooms' in data and 'bathrooms' in data:
        df['bed_bath_ratio'] = data['bedrooms'] / max(data['bathrooms'], 1)
        df['total_rooms'] = data['bedrooms'] + data['bathrooms']
    
    # Select and order columns
    available_cols = [col for col in feature_columns if col in df.columns]
    df = df[available_cols]
    
    # Transform
    X = preprocessor.transform(df)
    
    return X


if __name__ == "__main__":
    # Example usage
    data_path = "../Case Study 1 Data.xlsx"
    
    if os.path.exists(data_path):
        # Initialize preprocessor
        preprocessor = DataPreprocessor(data_path)
        
        # Load and process data
        preprocessor.load_data()
        preprocessor.identify_columns()
        preprocessor.clean_data()
        preprocessor.engineer_features()
        
        # Prepare data for training
        X_train, X_test, y_train, y_test, transformer = preprocessor.prepare_data()
        
        # Save preprocessor
        os.makedirs("../artifacts", exist_ok=True)
        preprocessor.save_preprocessor("../artifacts/preprocessor.joblib")
        
        print("\nPreprocessing complete!")
    else:
        print(f"Data file not found: {data_path}")
