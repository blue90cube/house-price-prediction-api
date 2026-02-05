"""
Optimized Model Training - Faster execution with advanced techniques.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

warnings.filterwarnings('ignore')


def load_and_preprocess(data_path):
    """Load and preprocess data with advanced techniques."""
    print("Loading data...")
    df = pd.read_excel(data_path, engine='openpyxl')
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    print(f"  Loaded {len(df):,} records")
    
    # Remove outliers (1st and 99th percentile for price)
    print("Removing outliers...")
    q_low = df['price'].quantile(0.01)
    q_high = df['price'].quantile(0.99)
    df = df[(df['price'] >= q_low) & (df['price'] <= q_high)]
    
    q_low_size = df['size'].quantile(0.01)
    q_high_size = df['size'].quantile(0.99)
    df = df[(df['size'] >= q_low_size) & (df['size'] <= q_high_size)]
    print(f"  Remaining: {len(df):,} records")
    
    # KNN imputation for numerical columns
    print("KNN imputation...")
    num_cols = ['size', 'bedrooms', 'bathrooms', 'year_built', 'price']
    knn = KNNImputer(n_neighbors=5)
    df[num_cols] = knn.fit_transform(df[num_cols])
    
    # Categorical imputation
    for col in ['location', 'condition', 'type']:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Feature engineering
    print("Engineering features...")
    df['date_sold'] = pd.to_datetime(df['date_sold'], errors='coerce')
    df['sale_year'] = df['date_sold'].dt.year
    df['sale_month'] = df['date_sold'].dt.month
    df['sale_quarter'] = df['date_sold'].dt.quarter
    
    # Property age
    df['property_age'] = df['sale_year'] - df['year_built']
    df['property_age'] = df['property_age'].clip(lower=0)
    
    # Size features
    df['size_per_bedroom'] = df['size'] / df['bedrooms'].replace(0, 1)
    df['size_per_bathroom'] = df['size'] / df['bathrooms'].replace(0, 1)
    df['bed_bath_ratio'] = df['bedrooms'] / df['bathrooms'].replace(0, 1)
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    df['size_per_room'] = df['size'] / df['total_rooms'].replace(0, 1)
    df['log_size'] = np.log1p(df['size'])
    
    # Age features
    df['is_new'] = (df['property_age'] <= 5).astype(int)
    df['is_old'] = (df['property_age'] > 50).astype(int)
    df['decade_built'] = (df['year_built'] // 10) * 10
    
    # Cyclical month encoding
    df['month_sin'] = np.sin(2 * np.pi * df['sale_month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['sale_month'] / 12)
    
    # Location-based features (target encoding proxy)
    loc_price_mean = df.groupby('location')['price'].transform('mean')
    df['location_price_mean'] = loc_price_mean
    loc_price_median = df.groupby('location')['price'].transform('median')
    df['location_price_median'] = loc_price_median
    loc_count = df.groupby('location')['price'].transform('count')
    df['location_count'] = loc_count
    
    # Condition encoding
    cond_map = {'Poor': 1, 'Fair': 2, 'Average': 3, 'Good': 4, 'Excellent': 5}
    df['condition_encoded'] = df['condition'].map(cond_map).fillna(3)
    
    # Type features
    type_price_mean = df.groupby('type')['price'].transform('mean')
    df['type_price_mean'] = type_price_mean
    
    # Interaction features
    df['size_x_condition'] = df['size'] * df['condition_encoded']
    df['age_x_condition'] = df['property_age'] * df['condition_encoded']
    
    print(f"  Total features: {len(df.columns)}")
    return df


def prepare_features(df):
    """Prepare feature sets."""
    numerical_cols = [
        'size', 'bedrooms', 'bathrooms', 'year_built', 'sale_year', 'sale_month',
        'sale_quarter', 'property_age', 'size_per_bedroom', 'size_per_bathroom',
        'bed_bath_ratio', 'total_rooms', 'size_per_room', 'log_size', 'is_new',
        'is_old', 'decade_built', 'month_sin', 'month_cos', 'location_price_mean',
        'location_price_median', 'location_count', 'condition_encoded',
        'type_price_mean', 'size_x_condition', 'age_x_condition'
    ]
    categorical_cols = ['location', 'condition', 'type']
    
    # Filter to existing columns
    numerical_cols = [c for c in numerical_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    
    X = df[numerical_cols + categorical_cols].copy()
    y = df['price'].copy()
    
    return X, y, numerical_cols, categorical_cols


def create_preprocessor(numerical_cols, categorical_cols):
    """Create preprocessing pipeline."""
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, min_frequency=0.01))
    ])
    
    return ColumnTransformer([
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])


def evaluate(y_true, y_pred):
    """Calculate metrics."""
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100
    }


def train_model(name, model, X_train, y_train, preprocessor, n_splits=5):
    """Train and evaluate a model with K-fold CV."""
    print(f"\nTraining {name}...")
    start = time.time()
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', TransformedTargetRegressor(
            regressor=model,
            func=np.log1p,
            inverse_func=np.expm1
        ))
    ])
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(pipeline, X_train, y_train, cv=kfold, n_jobs=-1)
    
    metrics = evaluate(y_train, y_pred_cv)
    pipeline.fit(X_train, y_train)
    
    elapsed = time.time() - start
    print(f"  CV R2: {metrics['R2']:.4f} | MAE: ${metrics['MAE']:,.0f} | MAPE: {metrics['MAPE']:.2f}% | Time: {elapsed:.1f}s")
    
    return pipeline, metrics


def main():
    print("=" * 60)
    print("OPTIMIZED HOUSE PRICE PREDICTION TRAINING")
    print("=" * 60)
    
    # Paths
    data_path = os.path.join(os.path.dirname(__file__), "..", "Case Study 1 Data.xlsx")
    artifacts_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Load and preprocess
    df = load_and_preprocess(data_path)
    X, y, num_cols, cat_cols = prepare_features(df)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")
    
    # Preprocessor
    preprocessor = create_preprocessor(num_cols, cat_cols)
    
    # Models to train
    models = {
        'Ridge': Ridge(alpha=1.0),
        'HistGradientBoosting': HistGradientBoostingRegressor(
            max_iter=300, max_depth=12, learning_rate=0.05, random_state=42
        ),
    }
    
    if HAS_LIGHTGBM:
        models['LightGBM'] = lgb.LGBMRegressor(
            n_estimators=400, max_depth=12, learning_rate=0.03, num_leaves=80,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            n_jobs=-1, random_state=42, verbose=-1
        )
    
    if HAS_XGBOOST:
        models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=400, max_depth=10, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            n_jobs=-1, random_state=42, tree_method='hist'
        )
    
    # Train models
    results = {}
    for name, model in models.items():
        pipeline, metrics = train_model(name, model, X_train, y_train, preprocessor)
        results[name] = {'pipeline': pipeline, 'metrics': metrics}
    
    # Find best model
    best_name = max(results, key=lambda x: results[x]['metrics']['R2'])
    best_pipeline = results[best_name]['pipeline']
    
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (CV Results)")
    print("=" * 60)
    for name, res in sorted(results.items(), key=lambda x: -x[1]['metrics']['R2']):
        m = res['metrics']
        print(f"{name:25s} | R2: {m['R2']:.4f} | MAE: ${m['MAE']:>10,.0f} | MAPE: {m['MAPE']:.2f}%")
    
    # Test set evaluation
    print("\n" + "=" * 60)
    print(f"TEST SET EVALUATION: {best_name}")
    print("=" * 60)
    
    y_pred = best_pipeline.predict(X_test)
    test_metrics = evaluate(y_test, y_pred)
    
    print(f"R2:   {test_metrics['R2']:.4f}")
    print(f"MAE:  ${test_metrics['MAE']:,.2f}")
    print(f"RMSE: ${test_metrics['RMSE']:,.2f}")
    print(f"MAPE: {test_metrics['MAPE']:.2f}%")
    
    # Save model
    model_path = os.path.join(artifacts_dir, "model.joblib")
    joblib.dump({'pipeline': best_pipeline, 'model_name': best_name}, model_path)
    print(f"\nModel saved to: {model_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    return best_pipeline, test_metrics


if __name__ == "__main__":
    pipeline, metrics = main()
