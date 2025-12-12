import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from loguru import logger
from typing import Dict, Any, Tuple
import config.config as config
import joblib
from pathlib import Path


class ModelTrainer:
    """Train and evaluate multiple ML models for rating prediction"""
    
    def __init__(self, experiment_name: str = config.MLFLOW_EXPERIMENT_NAME):
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name)
        self.best_model = None
        self.best_metrics = None
        
    def get_models(self) -> Dict[str, Any]:
        """Get dictionary of models to train"""
        return {
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            ),
            'XGBoost': XGBRegressor(
                **config.XGBOOST_PARAMS,
                objective='reg:squarederror',
                n_jobs=-1
            ),
            'LightGBM': LGBMRegressor(
                **config.LIGHTGBM_PARAMS,
                objective='regression',
                n_jobs=-1,
                verbose=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=config.RANDOM_STATE
            ),
            'Ridge': Ridge(
                alpha=1.0,
                random_state=config.RANDOM_STATE
            ),
            'ElasticNet': ElasticNet(
                alpha=1.0,
                l1_ratio=0.5,
                random_state=config.RANDOM_STATE
            )
        }
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred)
        }
    
    def train_model(
        self,
        model,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Tuple[Any, Dict[str, float]]:
        """Train a single model and log to MLflow"""
        
        with mlflow.start_run(run_name=model_name):
            logger.info(f"Training {model_name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_metrics = self.calculate_metrics(y_train, train_pred)
            test_metrics = self.calculate_metrics(y_test, test_pred)
            
            # Log parameters
            if hasattr(model, 'get_params'):
                params = model.get_params()
                mlflow.log_params({k: v for k, v in params.items() if v is not None})
            
            # Log metrics
            mlflow.log_metrics({
                'train_rmse': train_metrics['rmse'],
                'train_mae': train_metrics['mae'],
                'train_r2': train_metrics['r2'],
                'test_rmse': test_metrics['rmse'],
                'test_mae': test_metrics['mae'],
                'test_r2': test_metrics['r2']
            })
            
            # Log model
            if 'XGBoost' in model_name:
                mlflow.xgboost.log_model(model, "model")
            elif 'LightGBM' in model_name:
                mlflow.lightgbm.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")
            
            # Log feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Log as artifact
                importance_path = config.MODELS_DIR / f"{model_name}_feature_importance.csv"
                feature_importance.to_csv(importance_path, index=False)
                mlflow.log_artifact(str(importance_path))
            
            logger.info(f"{model_name} - Test RMSE: {test_metrics['rmse']:.4f}, "
                       f"Test MAE: {test_metrics['mae']:.4f}, Test R²: {test_metrics['r2']:.4f}")
            
            return model, test_metrics
    
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Dict[str, Any]]:
        """Train all models and return results"""
        
        logger.info("Starting model training experiments...")
        results = {}
        models_dict = self.get_models()
        
        best_rmse = float('inf')
        
        for model_name, model in models_dict.items():
            try:
                trained_model, metrics = self.train_model(
                    model, model_name, X_train, y_train, X_test, y_test
                )
                
                results[model_name] = {
                    'model': trained_model,
                    'metrics': metrics
                }
                
                # Track best model
                if metrics['rmse'] < best_rmse:
                    best_rmse = metrics['rmse']
                    self.best_model = trained_model
                    self.best_metrics = metrics
                    self.best_model_name = model_name
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue
        
        logger.info(f"\nBest Model: {self.best_model_name}")
        logger.info(f"Best RMSE: {best_rmse:.4f}")
        
        return results
    
    def save_best_model(self, filepath: Path = None):
        """Save the best model locally"""
        if filepath is None:
            filepath = config.MODELS_DIR / f"best_model_{self.best_model_name}.pkl"
        
        joblib.dump(self.best_model, filepath)
        logger.info(f"Best model saved to {filepath}")
        
        # Save metadata
        metadata = {
            'model_name': self.best_model_name,
            'metrics': self.best_metrics,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        import json
        metadata_path = config.MODELS_DIR / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return filepath
    
    def register_model(
        self,
        model_name: str = config.MODEL_REGISTRY_NAME,
        run_id: str = None
    ):
        """Register the best model in MLflow Model Registry"""
        if run_id is None:
            # Get the latest run
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            run_id = runs.iloc[0]['run_id']
        
        model_uri = f"runs:/{run_id}/model"
        
        try:
            mlflow.register_model(model_uri, model_name)
            logger.info(f"Model registered as {model_name}")
        except Exception as e:
            logger.error(f"Failed to register model: {e}")


class ModelEvaluator:
    """Evaluate model performance with detailed analysis"""
    
    def __init__(self):
        pass
    
    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        rating_bins: list = [1, 2, 3, 4, 5]
    ) -> Dict[str, Any]:
        """Detailed evaluation of predictions"""
        
        # Overall metrics
        overall_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        # Per-rating metrics
        rating_metrics = {}
        for rating in rating_bins:
            mask = y_true == rating
            if mask.sum() > 0:
                rating_metrics[f'rating_{rating}'] = {
                    'count': mask.sum(),
                    'mae': mean_absolute_error(y_true[mask], y_pred[mask]),
                    'rmse': np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
                }
        
        # Prediction distribution
        pred_distribution = {
            'mean': float(np.mean(y_pred)),
            'std': float(np.std(y_pred)),
            'min': float(np.min(y_pred)),
            'max': float(np.max(y_pred))
        }
        
        return {
            'overall': overall_metrics,
            'per_rating': rating_metrics,
            'prediction_distribution': pred_distribution
        }


if __name__ == "__main__":
    logger.info("Model training module loaded successfully!")
    print("\nAvailable models:")
    trainer = ModelTrainer()
    for model_name in trainer.get_models().keys():
        print(f"  - {model_name}")
