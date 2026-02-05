"""
Advanced Model Training Module for House Price Prediction

This module implements advanced ML techniques for better performance:
- Advanced imputation (KNN, Iterative)
- Target log transformation
- Outlier handling
- Enhanced feature engineering
- Stacking ensemble
- Extensive hyperparameter tuning
"""

import os
import sys
import time
import warnings
from typing import Dict, List, Tuple, Any, Optional
import multiprocessing

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import (
    train_test_split, cross_val_score, RandomizedSearchCV, 
    KFold, cross_val_predict
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, RobustScaler,
    TargetEncoder, PowerTransformer
)
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    StackingRegressor, VotingRegressor, ExtraTreesRegressor,
    HistGradientBoostingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, 
    mean_absolute_percentage_error
)
from sklearn.feature_selection import SelectFromModel
from scipy import stats

# Import XGBoost and LightGBM
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

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AdvancedDataPreprocessor:
    """Advanced data preprocessing with sophisticated imputation and feature engineering."""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.column_mapping = {}
        self.target_column = None
        self.price_stats = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load and standardize column names."""
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_excel(self.data_path, engine='openpyxl')
        print(f"Loaded {len(self.df):,} records with {len(self.df.columns)} columns")
        
        # Standardize column names
        self.df.columns = self.df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Map columns
        self.column_mapping = {
            'property_id': self._find_column(['property_id', 'propertyid', 'id']),
            'location': self._find_column(['location', 'city', 'neighborhood']),
            'size': self._find_column(['size', 'sqft', 'square_feet']),
            'bedrooms': self._find_column(['bedrooms', 'bedroom', 'beds']),
            'bathrooms': self._find_column(['bathrooms', 'bathroom', 'baths']),
            'year_built': self._find_column(['year_built', 'yearbuilt', 'built_year']),
            'condition': self._find_column(['condition', 'property_condition']),
            'property_type': self._find_column(['type', 'property_type', 'home_type']),
            'date_sold': self._find_column(['date_sold', 'datesold', 'sale_date']),
            'price': self._find_column(['price', 'sale_price', 'sold_price'])
        }
        
        self.target_column = self.column_mapping['price']
        return self.df
    
    def _find_column(self, possible_names: List[str]) -> Optional[str]:
        """Find column by checking possible name variations."""
        for col in self.df.columns:
            for name in possible_names:
                if name in col.lower():
                    return col
        return None
    
    def remove_outliers(self, method: str = 'iqr', threshold: float = 3.0) -> pd.DataFrame:
        """Remove outliers using IQR or Z-score method."""
        print("\nRemoving outliers...")
        initial_count = len(self.df)
        
        price_col = self.column_mapping['price']
        size_col = self.column_mapping['size']
        
        if method == 'iqr':
            # Price outliers
            Q1 = self.df[price_col].quantile(0.01)
            Q3 = self.df[price_col].quantile(0.99)
            self.df = self.df[(self.df[price_col] >= Q1) & (self.df[price_col] <= Q3)]
            
            # Size outliers
            Q1_size = self.df[size_col].quantile(0.01)
            Q3_size = self.df[size_col].quantile(0.99)
            self.df = self.df[(self.df[size_col] >= Q1_size) & (self.df[size_col] <= Q3_size)]
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(self.df[price_col].dropna()))
            self.df = self.df[z_scores < threshold]
        
        removed = initial_count - len(self.df)
        print(f"  Removed {removed:,} outlier rows ({removed/initial_count*100:.1f}%)")
        print(f"  Remaining: {len(self.df):,} rows")
        
        return self.df
    
    def advanced_imputation(self) -> pd.DataFrame:
        """Apply advanced imputation strategies."""
        print("\nApplying advanced imputation...")
        
        # Get numerical columns
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if c != self.column_mapping['property_id']]
        
        # Get categorical columns
        cat_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # For numerical: KNN Imputation (better than simple median)
        if self.df[num_cols].isnull().sum().sum() > 0:
            print("  Using KNN imputation for numerical columns...")
            knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
            self.df[num_cols] = knn_imputer.fit_transform(self.df[num_cols])
        
        # For categorical: Mode imputation with frequency weighting
        for col in cat_cols:
            if self.df[col].isnull().sum() > 0:
                mode_val = self.df[col].mode()[0]
                self.df[col].fillna(mode_val, inplace=True)
        
        print(f"  Missing values after imputation: {self.df.isnull().sum().sum()}")
        return self.df
    
    def engineer_advanced_features(self) -> pd.DataFrame:
        """Create advanced engineered features."""
        print("\nEngineering advanced features...")
        
        # Get column references
        size_col = self.column_mapping['size']
        bedroom_col = self.column_mapping['bedrooms']
        bathroom_col = self.column_mapping['bathrooms']
        year_built_col = self.column_mapping['year_built']
        date_col = self.column_mapping['date_sold']
        price_col = self.column_mapping['price']
        location_col = self.column_mapping['location']
        condition_col = self.column_mapping['condition']
        type_col = self.column_mapping['property_type']
        
        # Date features
        if date_col:
            self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce')
            self.df['sale_year'] = self.df[date_col].dt.year
            self.df['sale_month'] = self.df[date_col].dt.month
            self.df['sale_quarter'] = self.df[date_col].dt.quarter
            self.df['sale_day_of_week'] = self.df[date_col].dt.dayofweek
            self.df['is_weekend'] = (self.df['sale_day_of_week'] >= 5).astype(int)
            
            # Cyclical encoding for month (captures seasonality better)
            self.df['month_sin'] = np.sin(2 * np.pi * self.df['sale_month'] / 12)
            self.df['month_cos'] = np.cos(2 * np.pi * self.df['sale_month'] / 12)
        
        # Property age and age-related features
        if year_built_col and 'sale_year' in self.df.columns:
            self.df['property_age'] = self.df['sale_year'] - self.df[year_built_col]
            self.df['property_age'] = self.df['property_age'].clip(lower=0)
            
            # Age categories
            self.df['is_new'] = (self.df['property_age'] <= 5).astype(int)
            self.df['is_old'] = (self.df['property_age'] > 50).astype(int)
            
            # Decade built
            self.df['decade_built'] = (self.df[year_built_col] // 10) * 10
        
        # Size-based features
        if size_col and bedroom_col:
            self.df['size_per_bedroom'] = self.df[size_col] / self.df[bedroom_col].replace(0, 1)
        
        if size_col and bathroom_col:
            self.df['size_per_bathroom'] = self.df[size_col] / self.df[bathroom_col].replace(0, 1)
        
        if bedroom_col and bathroom_col:
            self.df['bed_bath_ratio'] = self.df[bedroom_col] / self.df[bathroom_col].replace(0, 1)
            self.df['total_rooms'] = self.df[bedroom_col] + self.df[bathroom_col]
        
        if size_col and 'total_rooms' in self.df.columns:
            self.df['size_per_room'] = self.df[size_col] / self.df['total_rooms'].replace(0, 1)
        
        # Size categories (log-based)
        if size_col:
            self.df['log_size'] = np.log1p(self.df[size_col])
            self.df['size_squared'] = self.df[size_col] ** 2
        
        # Bedroom categories
        if bedroom_col:
            self.df['bedroom_category'] = pd.cut(
                self.df[bedroom_col], 
                bins=[-1, 1, 2, 3, 4, 100],
                labels=['studio', 'small', 'medium', 'large', 'very_large']
            ).astype(str)
        
        # Location-based aggregations (target encoding proxies)
        if location_col and price_col:
            # Mean price by location
            loc_price_mean = self.df.groupby(location_col)[price_col].transform('mean')
            self.df['location_price_mean'] = loc_price_mean
            
            # Median price by location
            loc_price_median = self.df.groupby(location_col)[price_col].transform('median')
            self.df['location_price_median'] = loc_price_median
            
            # Count of properties in location (popularity)
            loc_count = self.df.groupby(location_col)[price_col].transform('count')
            self.df['location_count'] = loc_count
            
            # Price std by location (price variation)
            loc_price_std = self.df.groupby(location_col)[price_col].transform('std')
            self.df['location_price_std'] = loc_price_std.fillna(0)
        
        # Condition encoding (ordinal)
        if condition_col:
            condition_order = {'Poor': 1, 'Fair': 2, 'Average': 3, 'Good': 4, 'Excellent': 5}
            self.df['condition_encoded'] = self.df[condition_col].map(condition_order)
            # Fill unmapped with median
            self.df['condition_encoded'].fillna(3, inplace=True)
        
        # Interaction features
        if size_col and 'condition_encoded' in self.df.columns:
            self.df['size_x_condition'] = self.df[size_col] * self.df['condition_encoded']
        
        if 'property_age' in self.df.columns and 'condition_encoded' in self.df.columns:
            self.df['age_x_condition'] = self.df['property_age'] * self.df['condition_encoded']
        
        # Property type encoding
        if type_col:
            type_price_mean = self.df.groupby(type_col)[price_col].transform('mean')
            self.df['type_price_mean'] = type_price_mean
        
        print(f"  Total features after engineering: {len(self.df.columns)}")
        return self.df
    
    def prepare_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """Prepare data with train/test split."""
        print("\nPreparing data for training...")
        
        # Define feature groups
        exclude_cols = [
            self.column_mapping['property_id'],
            self.column_mapping['price'],
            self.column_mapping['date_sold'],
            'bedroom_category'  # Will handle separately
        ]
        exclude_cols = [c for c in exclude_cols if c is not None]
        
        # Get feature columns
        feature_cols = [c for c in self.df.columns if c not in exclude_cols]
        
        # Separate numerical and categorical
        numerical_cols = []
        categorical_cols = []
        
        for col in feature_cols:
            if self.df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                numerical_cols.append(col)
            elif self.df[col].dtype == 'object':
                categorical_cols.append(col)
        
        # Remove datetime columns
        numerical_cols = [c for c in numerical_cols 
                        if not pd.api.types.is_datetime64_any_dtype(self.df[c])]
        
        print(f"  Numerical features: {len(numerical_cols)}")
        print(f"  Categorical features: {len(categorical_cols)}")
        
        # Prepare X and y
        X = self.df[numerical_cols + categorical_cols].copy()
        y = self.df[self.target_column].copy()
        
        # Store price statistics for later
        self.price_stats = {
            'mean': y.mean(),
            'std': y.std(),
            'min': y.min(),
            'max': y.max()
        }
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"  Training set: {len(X_train):,} samples")
        print(f"  Test set: {len(X_test):,} samples")
        
        return X_train, X_test, y_train, y_test, numerical_cols, categorical_cols


class AdvancedModelTrainer:
    """Advanced model trainer with ensemble methods and extensive tuning."""
    
    def __init__(self, n_jobs: int = -1, random_state: int = 42):
        self.n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()
        self.random_state = random_state
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_pipeline = None
        
        print(f"AdvancedModelTrainer initialized with {self.n_jobs} CPU cores")
    
    def create_preprocessor(self, numerical_cols: List[str], 
                           categorical_cols: List[str]) -> ColumnTransformer:
        """Create advanced preprocessing pipeline."""
        
        # Numerical: RobustScaler (better for outliers)
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        
        # Categorical: OneHotEncoder
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, min_frequency=0.01))
        ])
        
        preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ], remainder='drop')
        
        return preprocessor
    
    def get_models(self) -> Dict[str, Any]:
        """Get dictionary of models with optimized default parameters."""
        models = {
            'Ridge': Ridge(alpha=1.0, random_state=self.random_state),
            
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state),
            
            'RandomForest': RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=self.n_jobs,
                random_state=self.random_state
            ),
            
            'ExtraTrees': ExtraTreesRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                n_jobs=self.n_jobs,
                random_state=self.random_state
            ),
            
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=self.random_state
            ),
            
            'HistGradientBoosting': HistGradientBoostingRegressor(
                max_iter=200,
                max_depth=10,
                learning_rate=0.1,
                random_state=self.random_state
            )
        }
        
        if HAS_XGBOOST:
            models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                tree_method='hist'
            )
        
        if HAS_LIGHTGBM:
            models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=10,
                learning_rate=0.05,
                num_leaves=50,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=-1
            )
        
        if HAS_CATBOOST:
            models['CatBoost'] = CatBoostRegressor(
                iterations=300,
                depth=8,
                learning_rate=0.05,
                random_state=self.random_state,
                verbose=0
            )
        
        return models
    
    def get_param_grids(self) -> Dict[str, Dict]:
        """Get extensive hyperparameter grids."""
        param_grids = {
            'Ridge': {
                'model__regressor__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            },
            'RandomForest': {
                'model__regressor__n_estimators': [100, 200, 300],
                'model__regressor__max_depth': [10, 15, 20, 25, None],
                'model__regressor__min_samples_split': [2, 5, 10],
                'model__regressor__min_samples_leaf': [1, 2, 4]
            },
            'ExtraTrees': {
                'model__regressor__n_estimators': [100, 200, 300],
                'model__regressor__max_depth': [10, 15, 20, 25, None],
                'model__regressor__min_samples_split': [2, 5, 10]
            },
            'GradientBoosting': {
                'model__regressor__n_estimators': [100, 200, 300],
                'model__regressor__max_depth': [3, 5, 7, 10],
                'model__regressor__learning_rate': [0.01, 0.05, 0.1],
                'model__regressor__subsample': [0.7, 0.8, 0.9]
            }
        }
        
        if HAS_XGBOOST:
            param_grids['XGBoost'] = {
                'model__regressor__n_estimators': [200, 300, 500],
                'model__regressor__max_depth': [5, 7, 10, 12],
                'model__regressor__learning_rate': [0.01, 0.03, 0.05, 0.1],
                'model__regressor__subsample': [0.7, 0.8, 0.9],
                'model__regressor__colsample_bytree': [0.7, 0.8, 0.9],
                'model__regressor__reg_alpha': [0, 0.1, 0.5],
                'model__regressor__reg_lambda': [0.5, 1.0, 2.0]
            }
        
        if HAS_LIGHTGBM:
            param_grids['LightGBM'] = {
                'model__regressor__n_estimators': [200, 300, 500],
                'model__regressor__max_depth': [7, 10, 15, -1],
                'model__regressor__learning_rate': [0.01, 0.03, 0.05, 0.1],
                'model__regressor__num_leaves': [31, 50, 100, 150],
                'model__regressor__subsample': [0.7, 0.8, 0.9],
                'model__regressor__colsample_bytree': [0.7, 0.8, 0.9],
                'model__regressor__reg_alpha': [0, 0.1, 0.5],
                'model__regressor__reg_lambda': [0.5, 1.0, 2.0]
            }
        
        return param_grids
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        return {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100
        }
    
    def train_with_cv(self, model_name: str, model: Any, 
                      X_train: pd.DataFrame, y_train: pd.Series,
                      preprocessor: ColumnTransformer, 
                      n_splits: int = 5) -> Dict:
        """Train model with K-Fold cross-validation."""
        print(f"\nTraining {model_name}...")
        start_time = time.time()
        
        # Create pipeline with log-transformed target
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', TransformedTargetRegressor(
                regressor=model,
                func=np.log1p,
                inverse_func=np.expm1
            ))
        ])
        
        # K-Fold cross-validation
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        # Get cross-validated predictions
        y_pred_cv = cross_val_predict(pipeline, X_train, y_train, cv=kfold, n_jobs=self.n_jobs)
        
        # Calculate CV metrics
        cv_metrics = self.evaluate_model(y_train, y_pred_cv)
        
        # Fit on full training data
        pipeline.fit(X_train, y_train)
        
        elapsed_time = time.time() - start_time
        
        result = {
            'model_name': model_name,
            'pipeline': pipeline,
            'cv_metrics': cv_metrics,
            'training_time': elapsed_time
        }
        
        print(f"  CV R2: {cv_metrics['R2']:.4f}")
        print(f"  CV MAE: ${cv_metrics['MAE']:,.2f}")
        print(f"  CV MAPE: {cv_metrics['MAPE']:.2f}%")
        print(f"  Time: {elapsed_time:.2f}s")
        
        return result
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        numerical_cols: List[str], categorical_cols: List[str],
                        n_splits: int = 5) -> Dict[str, Dict]:
        """Train all models with cross-validation."""
        print("\n" + "=" * 60)
        print("TRAINING ALL MODELS WITH K-FOLD CV")
        print("=" * 60)
        
        preprocessor = self.create_preprocessor(numerical_cols, categorical_cols)
        models = self.get_models()
        self.results = {}
        
        for model_name, model in models.items():
            try:
                result = self.train_with_cv(
                    model_name, model, X_train, y_train, preprocessor, n_splits
                )
                self.results[model_name] = result
            except Exception as e:
                print(f"  Error training {model_name}: {str(e)}")
        
        # Find best model based on CV R2
        best_name = max(self.results, key=lambda x: self.results[x]['cv_metrics']['R2'])
        self.best_model_name = best_name
        self.best_pipeline = self.results[best_name]['pipeline']
        
        print("\n" + "=" * 60)
        print(f"BEST MODEL: {best_name}")
        print(f"CV R2: {self.results[best_name]['cv_metrics']['R2']:.4f}")
        print("=" * 60)
        
        return self.results
    
    def tune_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                   numerical_cols: List[str], categorical_cols: List[str],
                   n_iter: int = 50, n_splits: int = 5) -> Dict:
        """Tune hyperparameters for a specific model."""
        print(f"\n{'='*60}")
        print(f"TUNING {model_name}")
        print("=" * 60)
        
        param_grids = self.get_param_grids()
        
        if model_name not in param_grids:
            print(f"No hyperparameter grid for {model_name}")
            return {}
        
        models = self.get_models()
        model = models[model_name]
        
        preprocessor = self.create_preprocessor(numerical_cols, categorical_cols)
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', TransformedTargetRegressor(
                regressor=model,
                func=np.log1p,
                inverse_func=np.expm1
            ))
        ])
        
        # K-Fold for tuning
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        # Randomized search
        start_time = time.time()
        search = RandomizedSearchCV(
            pipeline,
            param_grids[model_name],
            n_iter=n_iter,
            cv=kfold,
            scoring='r2',
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=1
        )
        
        search.fit(X_train, y_train)
        elapsed_time = time.time() - start_time
        
        print(f"\nBest Parameters:")
        for param, value in search.best_params_.items():
            print(f"  {param}: {value}")
        print(f"Best CV R2: {search.best_score_:.4f}")
        print(f"Tuning Time: {elapsed_time:.2f}s")
        
        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'best_estimator': search.best_estimator_,
            'tuning_time': elapsed_time
        }
    
    def create_stacking_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                                 numerical_cols: List[str], categorical_cols: List[str]) -> Pipeline:
        """Create a stacking ensemble from best models."""
        print("\n" + "=" * 60)
        print("CREATING STACKING ENSEMBLE")
        print("=" * 60)
        
        preprocessor = self.create_preprocessor(numerical_cols, categorical_cols)
        
        # Base estimators
        estimators = []
        
        if HAS_LIGHTGBM:
            estimators.append(('lgb', lgb.LGBMRegressor(
                n_estimators=200, max_depth=10, learning_rate=0.05,
                num_leaves=50, n_jobs=self.n_jobs, random_state=self.random_state, verbose=-1
            )))
        
        if HAS_XGBOOST:
            estimators.append(('xgb', xgb.XGBRegressor(
                n_estimators=200, max_depth=7, learning_rate=0.05,
                n_jobs=self.n_jobs, random_state=self.random_state, tree_method='hist'
            )))
        
        estimators.append(('rf', RandomForestRegressor(
            n_estimators=200, max_depth=20, n_jobs=self.n_jobs, random_state=self.random_state
        )))
        
        estimators.append(('et', ExtraTreesRegressor(
            n_estimators=200, max_depth=20, n_jobs=self.n_jobs, random_state=self.random_state
        )))
        
        estimators.append(('gb', GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=self.random_state
        )))
        
        # Stacking with Ridge as final estimator
        stacking = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=5,
            n_jobs=self.n_jobs
        )
        
        # Full pipeline with target transformation
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', TransformedTargetRegressor(
                regressor=stacking,
                func=np.log1p,
                inverse_func=np.expm1
            ))
        ])
        
        print(f"Training stacking ensemble with {len(estimators)} base models...")
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        print(f"Training time: {time.time() - start_time:.2f}s")
        
        return pipeline
    
    def evaluate_on_test(self, pipeline: Pipeline, X_test: pd.DataFrame, 
                        y_test: pd.Series, model_name: str = "Model") -> Dict[str, float]:
        """Evaluate model on test data."""
        print("\n" + "=" * 60)
        print(f"TEST SET EVALUATION: {model_name}")
        print("=" * 60)
        
        y_pred = pipeline.predict(X_test)
        metrics = self.evaluate_model(y_test, y_pred)
        
        print(f"MAE: ${metrics['MAE']:,.2f}")
        print(f"RMSE: ${metrics['RMSE']:,.2f}")
        print(f"R2: {metrics['R2']:.4f}")
        print(f"MAPE: {metrics['MAPE']:.2f}%")
        
        return metrics
    
    def save_model(self, pipeline: Pipeline, model_name: str, path: str):
        """Save model to disk."""
        save_data = {
            'pipeline': pipeline,
            'model_name': model_name
        }
        joblib.dump(save_data, path)
        print(f"\nModel saved to {path}")
    
    def get_results_summary(self) -> pd.DataFrame:
        """Get summary of all model results."""
        if not self.results:
            return pd.DataFrame()
        
        summary = []
        for name, result in self.results.items():
            metrics = result['cv_metrics']
            summary.append({
                'Model': name,
                'CV R2': metrics['R2'],
                'CV MAE': metrics['MAE'],
                'CV MAPE': metrics['MAPE'],
                'Time (s)': result['training_time']
            })
        
        df = pd.DataFrame(summary)
        df = df.sort_values('CV R2', ascending=False)
        return df


def main():
    """Main training script with advanced techniques."""
    print("=" * 60)
    print("ADVANCED HOUSE PRICE PREDICTION - MODEL TRAINING")
    print("=" * 60)
    
    # Paths
    data_path = os.path.join(os.path.dirname(__file__), "..", "Case Study 1 Data.xlsx")
    artifacts_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Initialize preprocessor
    print("\n[1/7] Loading and preprocessing data...")
    preprocessor = AdvancedDataPreprocessor(data_path)
    preprocessor.load_data()
    
    # Remove outliers
    print("\n[2/7] Removing outliers...")
    preprocessor.remove_outliers(method='iqr')
    
    # Advanced imputation
    print("\n[3/7] Advanced imputation...")
    preprocessor.advanced_imputation()
    
    # Feature engineering
    print("\n[4/7] Engineering advanced features...")
    preprocessor.engineer_advanced_features()
    
    # Prepare data
    X_train, X_test, y_train, y_test, num_cols, cat_cols = preprocessor.prepare_data(
        test_size=0.2, random_state=42
    )
    
    # Initialize trainer
    print("\n[5/7] Training models with K-Fold CV...")
    trainer = AdvancedModelTrainer(n_jobs=-1, random_state=42)
    
    # Train all models
    results = trainer.train_all_models(X_train, y_train, num_cols, cat_cols, n_splits=5)
    
    # Print comparison
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(trainer.get_results_summary().to_string(index=False))
    
    # Tune best single model (LightGBM or XGBoost typically)
    print("\n[6/7] Hyperparameter tuning...")
    best_models_to_tune = ['LightGBM', 'XGBoost'] if HAS_LIGHTGBM and HAS_XGBOOST else ['RandomForest']
    
    best_tuned_result = None
    best_tuned_score = 0
    best_tuned_name = None
    
    for model_name in best_models_to_tune:
        if model_name in results:
            tuned = trainer.tune_model(model_name, X_train, y_train, num_cols, cat_cols, n_iter=30)
            if tuned and tuned['best_score'] > best_tuned_score:
                best_tuned_score = tuned['best_score']
                best_tuned_result = tuned
                best_tuned_name = model_name
    
    # Create stacking ensemble
    print("\n[7/7] Creating stacking ensemble...")
    stacking_pipeline = trainer.create_stacking_ensemble(X_train, y_train, num_cols, cat_cols)
    
    # Evaluate all on test set
    print("\n" + "=" * 60)
    print("FINAL TEST SET EVALUATIONS")
    print("=" * 60)
    
    # Evaluate tuned model
    if best_tuned_result:
        tuned_metrics = trainer.evaluate_on_test(
            best_tuned_result['best_estimator'], X_test, y_test, f"Tuned {best_tuned_name}"
        )
    
    # Evaluate stacking ensemble
    stacking_metrics = trainer.evaluate_on_test(stacking_pipeline, X_test, y_test, "Stacking Ensemble")
    
    # Choose best final model
    best_final_pipeline = stacking_pipeline
    best_final_name = "StackingEnsemble"
    best_final_r2 = stacking_metrics['R2']
    
    if best_tuned_result and tuned_metrics['R2'] > stacking_metrics['R2']:
        best_final_pipeline = best_tuned_result['best_estimator']
        best_final_name = f"Tuned_{best_tuned_name}"
        best_final_r2 = tuned_metrics['R2']
    
    # Save best model
    model_path = os.path.join(artifacts_dir, "model.joblib")
    trainer.save_model(best_final_pipeline, best_final_name, model_path)
    
    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nBest Model: {best_final_name}")
    print(f"Test R2: {best_final_r2:.4f}")
    
    if best_final_name == "StackingEnsemble":
        print(f"Test MAE: ${stacking_metrics['MAE']:,.2f}")
        print(f"Test MAPE: {stacking_metrics['MAPE']:.2f}%")
    else:
        print(f"Test MAE: ${tuned_metrics['MAE']:,.2f}")
        print(f"Test MAPE: {tuned_metrics['MAPE']:.2f}%")
    
    print(f"\nModel saved to: {model_path}")
    
    return trainer, best_final_pipeline


if __name__ == "__main__":
    trainer, pipeline = main()
