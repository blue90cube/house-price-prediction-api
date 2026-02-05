"""
Model Training Module for House Price Prediction

This module handles model training with multi-core processing support,
cross-validation, hyperparameter tuning, and model selection.
"""

import os
import sys
import time
import warnings
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import (
    train_test_split, cross_val_score, RandomizedSearchCV, KFold
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, 
    mean_absolute_percentage_error
)

# Import XGBoost and LightGBM (optional)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Skipping XGBoost models.")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not installed. Skipping LightGBM models.")

# Suppress warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.data_preprocessing import DataPreprocessor


class ModelTrainer:
    """
    A class to handle model training with multi-core processing support.
    """
    
    def __init__(self, n_jobs: int = -1, random_state: int = 42):
        """
        Initialize the ModelTrainer.
        
        Args:
            n_jobs: Number of CPU cores to use (-1 for all available)
            random_state: Random seed for reproducibility
        """
        self.n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_pipeline = None
        
        print(f"ModelTrainer initialized with {self.n_jobs} CPU cores")
    
    def get_models(self) -> Dict[str, Any]:
        """
        Get dictionary of models to train.
        
        Returns:
            Dictionary of model name to model instance
        """
        models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(random_state=self.random_state),
            'Lasso': Lasso(random_state=self.random_state),
            'ElasticNet': ElasticNet(random_state=self.random_state),
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                n_jobs=self.n_jobs,
                random_state=self.random_state
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=self.random_state
            )
        }
        
        if HAS_XGBOOST:
            models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                tree_method='hist'
            )
        
        if HAS_LIGHTGBM:
            models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=100,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=-1
            )
        
        return models
    
    def get_param_grids(self) -> Dict[str, Dict]:
        """
        Get hyperparameter grids for each model.
        
        Returns:
            Dictionary of model name to parameter grid
        """
        param_grids = {
            'Ridge': {
                'model__alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'Lasso': {
                'model__alpha': [0.001, 0.01, 0.1, 1.0]
            },
            'ElasticNet': {
                'model__alpha': [0.1, 1.0, 10.0],
                'model__l1_ratio': [0.2, 0.5, 0.8]
            },
            'RandomForest': {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [10, 20, 30, None],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4]
            },
            'GradientBoosting': {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [3, 5, 7],
                'model__learning_rate': [0.01, 0.1, 0.2]
            }
        }
        
        if HAS_XGBOOST:
            param_grids['XGBoost'] = {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [3, 5, 7],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__subsample': [0.8, 1.0]
            }
        
        if HAS_LIGHTGBM:
            param_grids['LightGBM'] = {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [3, 5, 7, -1],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__num_leaves': [31, 50, 100]
            }
        
        return param_grids
    
    def create_pipeline(self, preprocessor: ColumnTransformer, model: Any) -> Pipeline:
        """
        Create a sklearn pipeline with preprocessor and model.
        
        Args:
            preprocessor: Fitted ColumnTransformer
            model: Model instance
            
        Returns:
            Configured Pipeline
        """
        return Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metric names to values
        """
        return {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100
        }
    
    def train_single_model(self, 
                           model_name: str,
                           model: Any,
                           X_train: pd.DataFrame,
                           y_train: pd.Series,
                           preprocessor: ColumnTransformer,
                           cv: int = 5) -> Dict:
        """
        Train a single model with cross-validation.
        
        Args:
            model_name: Name of the model
            model: Model instance
            X_train: Training features
            y_train: Training target
            preprocessor: Preprocessor for the pipeline
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with training results
        """
        print(f"\nTraining {model_name}...")
        start_time = time.time()
        
        # Create pipeline
        pipeline = self.create_pipeline(preprocessor, model)
        
        # Cross-validation with multi-core
        cv_scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv,
            scoring='neg_mean_absolute_error',
            n_jobs=self.n_jobs
        )
        
        # Fit on full training data
        pipeline.fit(X_train, y_train)
        
        # Training predictions
        y_train_pred = pipeline.predict(X_train)
        train_metrics = self.evaluate_model(y_train, y_train_pred)
        
        elapsed_time = time.time() - start_time
        
        result = {
            'model_name': model_name,
            'pipeline': pipeline,
            'cv_scores': -cv_scores,  # Convert to positive MAE
            'cv_mean': -cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_metrics': train_metrics,
            'training_time': elapsed_time
        }
        
        print(f"  CV MAE: {result['cv_mean']:.2f} (+/- {result['cv_std']:.2f})")
        print(f"  Training R2: {train_metrics['R2']:.4f}")
        print(f"  Time: {elapsed_time:.2f}s")
        
        return result
    
    def train_all_models(self,
                         X_train: pd.DataFrame,
                         y_train: pd.Series,
                         preprocessor: ColumnTransformer,
                         cv: int = 5) -> Dict[str, Dict]:
        """
        Train all models with cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training target
            preprocessor: Preprocessor for the pipeline
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary of model names to results
        """
        print("\n" + "=" * 60)
        print("TRAINING ALL MODELS")
        print("=" * 60)
        
        models = self.get_models()
        self.results = {}
        
        for model_name, model in models.items():
            try:
                result = self.train_single_model(
                    model_name, model, X_train, y_train, preprocessor, cv
                )
                self.results[model_name] = result
            except Exception as e:
                print(f"  Error training {model_name}: {str(e)}")
        
        # Find best model based on CV score
        best_name = min(self.results, key=lambda x: self.results[x]['cv_mean'])
        self.best_model_name = best_name
        self.best_model = self.results[best_name]['pipeline'].named_steps['model']
        self.best_pipeline = self.results[best_name]['pipeline']
        
        print("\n" + "=" * 60)
        print(f"BEST MODEL: {best_name}")
        print(f"CV MAE: {self.results[best_name]['cv_mean']:.2f}")
        print("=" * 60)
        
        return self.results
    
    def tune_best_model(self,
                        X_train: pd.DataFrame,
                        y_train: pd.Series,
                        preprocessor: ColumnTransformer,
                        n_iter: int = 20,
                        cv: int = 5) -> Dict:
        """
        Tune hyperparameters of the best model.
        
        Args:
            X_train: Training features
            y_train: Training target
            preprocessor: Preprocessor for the pipeline
            n_iter: Number of parameter combinations to try
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with tuning results
        """
        if not self.best_model_name:
            raise ValueError("No best model found. Run train_all_models first.")
        
        print("\n" + "=" * 60)
        print(f"TUNING HYPERPARAMETERS: {self.best_model_name}")
        print("=" * 60)
        
        param_grids = self.get_param_grids()
        
        if self.best_model_name not in param_grids:
            print(f"No hyperparameter grid defined for {self.best_model_name}")
            return {}
        
        # Get fresh model instance
        models = self.get_models()
        model = models[self.best_model_name]
        
        # Create pipeline
        pipeline = self.create_pipeline(preprocessor, model)
        
        # Randomized search
        start_time = time.time()
        search = RandomizedSearchCV(
            pipeline,
            param_grids[self.best_model_name],
            n_iter=n_iter,
            cv=cv,
            scoring='neg_mean_absolute_error',
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=1
        )
        
        search.fit(X_train, y_train)
        elapsed_time = time.time() - start_time
        
        print(f"\nBest Parameters: {search.best_params_}")
        print(f"Best CV MAE: {-search.best_score_:.2f}")
        print(f"Tuning Time: {elapsed_time:.2f}s")
        
        # Update best pipeline
        self.best_pipeline = search.best_estimator_
        self.best_model = self.best_pipeline.named_steps['model']
        
        return {
            'best_params': search.best_params_,
            'best_score': -search.best_score_,
            'tuning_time': elapsed_time
        }
    
    def evaluate_on_test(self,
                         X_test: pd.DataFrame,
                         y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the best model on test data.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of test metrics
        """
        if self.best_pipeline is None:
            raise ValueError("No trained model found.")
        
        print("\n" + "=" * 60)
        print("TEST SET EVALUATION")
        print("=" * 60)
        
        y_pred = self.best_pipeline.predict(X_test)
        metrics = self.evaluate_model(y_test, y_pred)
        
        print(f"\nModel: {self.best_model_name}")
        print(f"MAE: ${metrics['MAE']:,.2f}")
        print(f"RMSE: ${metrics['RMSE']:,.2f}")
        print(f"R2: {metrics['R2']:.4f}")
        print(f"MAPE: {metrics['MAPE']:.2f}%")
        
        return metrics
    
    def save_model(self, path: str, include_metadata: bool = True):
        """
        Save the best model pipeline to disk.
        
        Args:
            path: Path to save the model
            include_metadata: Whether to include training metadata
        """
        if self.best_pipeline is None:
            raise ValueError("No trained model to save.")
        
        save_data = {
            'pipeline': self.best_pipeline,
            'model_name': self.best_model_name
        }
        
        if include_metadata:
            save_data['results'] = {
                name: {
                    'cv_mean': res['cv_mean'],
                    'cv_std': res['cv_std'],
                    'train_metrics': res['train_metrics'],
                    'training_time': res['training_time']
                }
                for name, res in self.results.items()
            }
        
        joblib.dump(save_data, path)
        print(f"\nModel saved to {path}")
    
    @staticmethod
    def load_model(path: str) -> Dict:
        """
        Load a saved model from disk.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Dictionary containing pipeline and metadata
        """
        return joblib.load(path)
    
    def get_results_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of all model results.
        
        Returns:
            DataFrame with model comparison
        """
        if not self.results:
            return pd.DataFrame()
        
        summary = []
        for name, result in self.results.items():
            summary.append({
                'Model': name,
                'CV MAE': result['cv_mean'],
                'CV Std': result['cv_std'],
                'Train MAE': result['train_metrics']['MAE'],
                'Train R2': result['train_metrics']['R2'],
                'Time (s)': result['training_time']
            })
        
        df = pd.DataFrame(summary)
        df = df.sort_values('CV MAE')
        return df


def train_model_parallel(args: tuple) -> Dict:
    """
    Helper function for parallel model training.
    
    Args:
        args: Tuple of (model_name, model, X_train, y_train, preprocessor, cv)
        
    Returns:
        Training result dictionary
    """
    model_name, model, X_train, y_train, preprocessor, cv = args
    trainer = ModelTrainer(n_jobs=1)  # Use single core within worker
    return trainer.train_single_model(
        model_name, model, X_train, y_train, preprocessor, cv
    )


def main():
    """Main training script."""
    print("=" * 60)
    print("HOUSE PRICE PREDICTION - MODEL TRAINING")
    print("=" * 60)
    
    # Paths
    data_path = os.path.join(os.path.dirname(__file__), "..", "Case Study 1 Data.xlsx")
    artifacts_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Initialize preprocessor
    print("\n[1/5] Loading and preprocessing data...")
    preprocessor = DataPreprocessor(data_path)
    preprocessor.load_data()
    preprocessor.identify_columns()
    preprocessor.clean_data()
    preprocessor.engineer_features()
    
    # Prepare data
    X_train, X_test, y_train, y_test, transformer = preprocessor.prepare_data(
        test_size=0.2, random_state=42
    )
    
    # Save preprocessor info
    preprocessor.save_preprocessor(os.path.join(artifacts_dir, "preprocessor.joblib"))
    
    # Initialize trainer
    print("\n[2/5] Initializing model trainer...")
    trainer = ModelTrainer(n_jobs=-1, random_state=42)
    
    # Train all models
    print("\n[3/5] Training models...")
    results = trainer.train_all_models(X_train, y_train, transformer, cv=5)
    
    # Print summary
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(trainer.get_results_summary().to_string(index=False))
    
    # Tune best model
    print("\n[4/5] Tuning best model...")
    tuning_results = trainer.tune_best_model(
        X_train, y_train, transformer, n_iter=20, cv=5
    )
    
    # Evaluate on test set
    print("\n[5/5] Evaluating on test set...")
    test_metrics = trainer.evaluate_on_test(X_test, y_test)
    
    # Save final model
    model_path = os.path.join(artifacts_dir, "model.joblib")
    trainer.save_model(model_path)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nBest Model: {trainer.best_model_name}")
    print(f"Test MAE: ${test_metrics['MAE']:,.2f}")
    print(f"Test R2: {test_metrics['R2']:.4f}")
    print(f"\nModel saved to: {model_path}")
    
    return trainer, test_metrics


if __name__ == "__main__":
    trainer, metrics = main()
