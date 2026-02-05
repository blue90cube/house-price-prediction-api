"""
Model Evaluation Module for House Price Prediction

This module contains functions for evaluating trained models,
generating predictions, and creating performance reports.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import joblib
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)


class ModelEvaluator:
    """
    A class to evaluate trained models and generate reports.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the ModelEvaluator.
        
        Args:
            model_path: Path to the saved model file
        """
        self.model_path = model_path
        self.model_data = None
        self.pipeline = None
        self.model_name = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the saved model from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.model_data = joblib.load(self.model_path)
        self.pipeline = self.model_data['pipeline']
        self.model_name = self.model_data.get('model_name', 'Unknown')
        
        print(f"Loaded model: {self.model_name}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the loaded model.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions
        """
        return self.pipeline.predict(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model on given data.
        
        Args:
            X: Feature DataFrame
            y: True target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X)
        
        metrics = {
            'MAE': mean_absolute_error(y, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
            'R2': r2_score(y, y_pred),
            'MAPE': mean_absolute_percentage_error(y, y_pred) * 100,
            'MedAE': np.median(np.abs(y - y_pred))
        }
        
        return metrics
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance if available.
        
        Returns:
            DataFrame with feature importances or None
        """
        model = self.pipeline.named_steps['model']
        preprocessor = self.pipeline.named_steps['preprocessor']
        
        # Check if model has feature_importances_
        if not hasattr(model, 'feature_importances_'):
            print(f"Model {self.model_name} doesn't support feature importances")
            return None
        
        # Get feature names from preprocessor
        feature_names = []
        
        # Numerical features
        if hasattr(preprocessor, 'transformers_'):
            for name, transformer, cols in preprocessor.transformers_:
                if name == 'num':
                    feature_names.extend(cols)
                elif name == 'cat':
                    # Get one-hot encoded feature names
                    encoder = transformer.named_steps['encoder']
                    if hasattr(encoder, 'get_feature_names_out'):
                        cat_features = encoder.get_feature_names_out(cols)
                        feature_names.extend(cat_features)
                    else:
                        feature_names.extend(cols)
        
        # Match lengths
        importances = model.feature_importances_
        if len(feature_names) != len(importances):
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        return importance_df
    
    def plot_predictions(self, y_true: pd.Series, y_pred: np.ndarray, 
                         save_path: Optional[str] = None):
        """
        Plot actual vs predicted values.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.3, s=10)
        axes[0].plot([y_true.min(), y_true.max()], 
                     [y_true.min(), y_true.max()], 
                     'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Price')
        axes[0].set_ylabel('Predicted Price')
        axes[0].set_title(f'{self.model_name}: Actual vs Predicted')
        axes[0].legend()
        
        # Residual plot
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.3, s=10)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Price')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, top_n: int = 20, 
                                save_path: Optional[str] = None):
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to show
            save_path: Optional path to save the plot
        """
        importance_df = self.get_feature_importance()
        
        if importance_df is None:
            return
        
        # Get top N features
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['Importance'])
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'{self.model_name}: Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_residual_distribution(self, y_true: pd.Series, y_pred: np.ndarray,
                                   save_path: Optional[str] = None):
        """
        Plot distribution of residuals.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            save_path: Optional path to save the plot
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Residuals')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Residual Distribution')
        
        # Q-Q plot approximation
        sorted_residuals = np.sort(residuals)
        theoretical_quantiles = np.linspace(0.001, 0.999, len(sorted_residuals))
        theoretical_values = np.quantile(sorted_residuals, theoretical_quantiles)
        
        axes[1].scatter(theoretical_values, sorted_residuals, alpha=0.3, s=10)
        axes[1].plot([sorted_residuals.min(), sorted_residuals.max()],
                     [sorted_residuals.min(), sorted_residuals.max()],
                     'r--', lw=2)
        axes[1].set_xlabel('Theoretical Quantiles')
        axes[1].set_ylabel('Sample Quantiles')
        axes[1].set_title('Q-Q Plot')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, X_test: pd.DataFrame, y_test: pd.Series,
                        output_dir: str):
        """
        Generate a comprehensive evaluation report.
        
        Args:
            X_test: Test features
            y_test: Test target
            output_dir: Directory to save report files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 60)
        print("GENERATING EVALUATION REPORT")
        print("=" * 60)
        
        # Get predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        metrics = self.evaluate(X_test, y_test)
        
        print(f"\nModel: {self.model_name}")
        print("-" * 40)
        for metric, value in metrics.items():
            if metric in ['MAE', 'RMSE', 'MedAE']:
                print(f"{metric}: ${value:,.2f}")
            elif metric == 'MAPE':
                print(f"{metric}: {value:.2f}%")
            else:
                print(f"{metric}: {value:.4f}")
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
        
        # Plot predictions
        self.plot_predictions(
            y_test, y_pred,
            save_path=os.path.join(output_dir, 'predictions.png')
        )
        
        # Plot feature importance
        self.plot_feature_importance(
            save_path=os.path.join(output_dir, 'feature_importance.png')
        )
        
        # Plot residuals
        self.plot_residual_distribution(
            y_test, y_pred,
            save_path=os.path.join(output_dir, 'residuals.png')
        )
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'Actual': y_test.values,
            'Predicted': y_pred,
            'Residual': y_test.values - y_pred,
            'APE': np.abs(y_test.values - y_pred) / y_test.values * 100
        })
        predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
        
        print(f"\nReport saved to {output_dir}")
        
        return metrics


def compare_models(model_paths: List[str], X_test: pd.DataFrame, 
                   y_test: pd.Series) -> pd.DataFrame:
    """
    Compare multiple models on the same test data.
    
    Args:
        model_paths: List of paths to saved models
        X_test: Test features
        y_test: Test target
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for path in model_paths:
        try:
            evaluator = ModelEvaluator(path)
            metrics = evaluator.evaluate(X_test, y_test)
            metrics['Model'] = evaluator.model_name
            metrics['Path'] = path
            results.append(metrics)
        except Exception as e:
            print(f"Error evaluating {path}: {e}")
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df[['Model', 'MAE', 'RMSE', 'R2', 'MAPE']]
    comparison_df = comparison_df.sort_values('MAE')
    
    return comparison_df


if __name__ == "__main__":
    # Example usage
    print("Model Evaluation Module")
    print("=" * 50)
    print("\nUsage:")
    print("  evaluator = ModelEvaluator('artifacts/model.joblib')")
    print("  metrics = evaluator.evaluate(X_test, y_test)")
    print("  evaluator.generate_report(X_test, y_test, 'reports/')")
