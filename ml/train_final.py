"""
Final Optimized Training Script with Logging
High-performance models only: LightGBM, XGBoost
"""

import os
import sys
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
import lightgbm as lgb

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Setup logging
log_file = os.path.join(LOGS_DIR, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_data(path):
    """Load and clean data."""
    logger.info(f"Loading data from {path}")
    df = pd.read_excel(path, engine='openpyxl')
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    logger.info(f"Loaded {len(df):,} records, {len(df.columns)} columns")
    return df


def remove_outliers(df):
    """Remove price and size outliers using percentile method."""
    logger.info("Removing outliers (1st-99th percentile)...")
    initial = len(df)
    
    # Price outliers
    p_low, p_high = df['price'].quantile([0.01, 0.99])
    df = df[(df['price'] >= p_low) & (df['price'] <= p_high)]
    
    # Size outliers  
    s_low, s_high = df['size'].quantile([0.01, 0.99])
    df = df[(df['size'] >= s_low) & (df['size'] <= s_high)]
    
    logger.info(f"Removed {initial - len(df):,} outliers, remaining: {len(df):,}")
    return df


def impute_missing(df):
    """Handle missing values."""
    logger.info("Imputing missing values...")
    
    # Numerical: median imputation
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Categorical: mode imputation
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    logger.info(f"Missing values after imputation: {df.isnull().sum().sum()}")
    return df


def engineer_features(df):
    """Create engineered features."""
    logger.info("Engineering features...")
    
    # Date features
    df['date_sold'] = pd.to_datetime(df['date_sold'], errors='coerce')
    df['sale_year'] = df['date_sold'].dt.year.fillna(2020).astype(int)
    df['sale_month'] = df['date_sold'].dt.month.fillna(6).astype(int)
    df['sale_quarter'] = df['date_sold'].dt.quarter.fillna(2).astype(int)
    
    # Property age
    df['property_age'] = (df['sale_year'] - df['year_built']).clip(lower=0)
    
    # Room features
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    df['size_per_bedroom'] = df['size'] / df['bedrooms'].replace(0, 1)
    df['size_per_bathroom'] = df['size'] / df['bathrooms'].replace(0, 1)
    df['bed_bath_ratio'] = df['bedrooms'] / df['bathrooms'].replace(0, 1)
    df['size_per_room'] = df['size'] / df['total_rooms'].replace(0, 1)
    
    # Log size (reduces skewness)
    df['log_size'] = np.log1p(df['size'])
    
    # Age indicators
    df['is_new'] = (df['property_age'] <= 5).astype(int)
    df['is_old'] = (df['property_age'] > 50).astype(int)
    
    # Cyclical month encoding
    df['month_sin'] = np.sin(2 * np.pi * df['sale_month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['sale_month'] / 12)
    
    # Location-based target encoding (mean price per location)
    loc_stats = df.groupby('location')['price'].agg(['mean', 'median', 'count'])
    df['loc_price_mean'] = df['location'].map(loc_stats['mean'])
    df['loc_price_median'] = df['location'].map(loc_stats['median'])
    df['loc_count'] = df['location'].map(loc_stats['count'])
    
    # Condition encoding
    cond_map = {'Poor': 1, 'Fair': 2, 'Average': 3, 'Good': 4, 'Excellent': 5}
    df['condition_num'] = df['condition'].map(cond_map).fillna(3)
    
    # Type encoding
    type_stats = df.groupby('type')['price'].mean()
    df['type_price_mean'] = df['type'].map(type_stats)
    
    # Interactions
    df['size_x_condition'] = df['size'] * df['condition_num']
    df['age_x_condition'] = df['property_age'] * df['condition_num']
    
    logger.info(f"Total features after engineering: {len(df.columns)}")
    return df


def prepare_data(df):
    """Prepare X and y for training."""
    numerical_cols = [
        'size', 'bedrooms', 'bathrooms', 'year_built', 'sale_year', 'sale_month',
        'sale_quarter', 'property_age', 'total_rooms', 'size_per_bedroom',
        'size_per_bathroom', 'bed_bath_ratio', 'size_per_room', 'log_size',
        'is_new', 'is_old', 'month_sin', 'month_cos', 'loc_price_mean',
        'loc_price_median', 'loc_count', 'condition_num', 'type_price_mean',
        'size_x_condition', 'age_x_condition'
    ]
    categorical_cols = ['location', 'condition', 'type']
    
    # Filter existing columns
    numerical_cols = [c for c in numerical_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    
    X = df[numerical_cols + categorical_cols].copy()
    y = df['price'].copy()
    
    logger.info(f"Features: {len(numerical_cols)} numerical, {len(categorical_cols)} categorical")
    return X, y, numerical_cols, categorical_cols


def create_preprocessor(num_cols, cat_cols):
    """Create sklearn preprocessor."""
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, min_frequency=0.005))
    ])
    return ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])


def evaluate(y_true, y_pred):
    """Calculate metrics."""
    return {
        'R2': r2_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100
    }


def train_lightgbm(X_train, y_train, preprocessor):
    """Train LightGBM with log target transformation."""
    logger.info("Training LightGBM...")
    
    model = lgb.LGBMRegressor(
        n_estimators=500,
        max_depth=15,
        num_leaves=100,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=20,
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', TransformedTargetRegressor(
            regressor=model,
            func=np.log1p,
            inverse_func=np.expm1
        ))
    ])
    
    # 5-fold CV
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(pipeline, X_train, y_train, cv=kfold, n_jobs=-1)
    cv_metrics = evaluate(y_train, y_pred_cv)
    
    logger.info(f"LightGBM CV - R2: {cv_metrics['R2']:.4f}, MAE: ${cv_metrics['MAE']:,.0f}, MAPE: {cv_metrics['MAPE']:.2f}%")
    
    # Fit on all training data
    pipeline.fit(X_train, y_train)
    return pipeline, cv_metrics


def train_xgboost(X_train, y_train, preprocessor):
    """Train XGBoost with log target transformation."""
    logger.info("Training XGBoost...")
    
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=12,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=5,
        n_jobs=-1,
        random_state=42,
        tree_method='hist'
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', TransformedTargetRegressor(
            regressor=model,
            func=np.log1p,
            inverse_func=np.expm1
        ))
    ])
    
    # 5-fold CV
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(pipeline, X_train, y_train, cv=kfold, n_jobs=-1)
    cv_metrics = evaluate(y_train, y_pred_cv)
    
    logger.info(f"XGBoost CV - R2: {cv_metrics['R2']:.4f}, MAE: ${cv_metrics['MAE']:,.0f}, MAPE: {cv_metrics['MAPE']:.2f}%")
    
    # Fit on all training data
    pipeline.fit(X_train, y_train)
    return pipeline, cv_metrics


def main():
    """Main training function."""
    logger.info("=" * 60)
    logger.info("HOUSE PRICE PREDICTION - FINAL TRAINING")
    logger.info("=" * 60)
    
    # Load data
    data_path = os.path.join(BASE_DIR, "Case Study 1 Data.xlsx")
    df = load_data(data_path)
    
    # Preprocess
    df = remove_outliers(df)
    df = impute_missing(df)
    df = engineer_features(df)
    
    # Prepare features
    X, y, num_cols, cat_cols = prepare_data(df)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    # Create preprocessor
    preprocessor = create_preprocessor(num_cols, cat_cols)
    
    # Train models
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING MODELS")
    logger.info("=" * 60)
    
    lgb_pipeline, lgb_cv = train_lightgbm(X_train, y_train, preprocessor)
    xgb_pipeline, xgb_cv = train_xgboost(X_train, y_train, preprocessor)
    
    # Select best model
    if lgb_cv['R2'] >= xgb_cv['R2']:
        best_pipeline, best_name, best_cv = lgb_pipeline, 'LightGBM', lgb_cv
    else:
        best_pipeline, best_name, best_cv = xgb_pipeline, 'XGBoost', xgb_cv
    
    # Evaluate on test set
    logger.info("\n" + "=" * 60)
    logger.info(f"TEST SET EVALUATION - {best_name}")
    logger.info("=" * 60)
    
    y_pred = best_pipeline.predict(X_test)
    test_metrics = evaluate(y_test, y_pred)
    
    logger.info(f"R2:   {test_metrics['R2']:.4f}")
    logger.info(f"MAE:  ${test_metrics['MAE']:,.2f}")
    logger.info(f"RMSE: ${test_metrics['RMSE']:,.2f}")
    logger.info(f"MAPE: {test_metrics['MAPE']:.2f}%")
    
    # Save model
    model_path = os.path.join(ARTIFACTS_DIR, "model.joblib")
    joblib.dump({
        'pipeline': best_pipeline,
        'model_name': best_name,
        'cv_metrics': best_cv,
        'test_metrics': test_metrics
    }, model_path)
    logger.info(f"\nModel saved: {model_path}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best Model: {best_name}")
    logger.info(f"Test R2: {test_metrics['R2']:.4f}")
    logger.info(f"Test MAE: ${test_metrics['MAE']:,.2f}")
    logger.info(f"Log file: {log_file}")
    
    return best_pipeline, test_metrics


if __name__ == "__main__":
    pipeline, metrics = main()
