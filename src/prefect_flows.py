"""
Prefect Orchestration Pipeline for MovieLens ML Training
"""

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
import pandas as pd
import numpy as np
from loguru import logger
from typing import Tuple, Dict, Any
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now imports will work
from src.data_loader import MovieLensDataLoader
from src.feature_engineering import FeatureEngineer  
from src.model_training import ModelTrainer
from config import config

@task(
    name="download_data",
    description="Download and load MovieLens data",
    retries=2,
    retry_delay_seconds=10
)
def download_data_task() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Task: Download and load raw data with retry logic"""
    try:
        logger.info("Starting data download task...")
        loader = MovieLensDataLoader()
        ratings, movies, users = loader.load_all_data()
        
        logger.info(f"✓ Data loaded: {len(ratings)} ratings, {len(movies)} movies, {len(users)} users")
        return ratings, movies, users
        
    except Exception as e:
        logger.error(f"✗ Data download failed: {e}")
        raise


@task(
    name="engineer_features",
    description="Create features from raw data",
    retries=1
)
def engineer_features_task(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    users: pd.DataFrame
) -> pd.DataFrame:
    """Task: Feature engineering with error handling"""
    try:
        logger.info("Starting feature engineering task...")
        engineer = FeatureEngineer()
        features_df = engineer.engineer_features(ratings, movies, users)
        
        logger.info(f"✓ Features engineered: shape={features_df.shape}")
        return features_df
        
    except Exception as e:
        logger.error(f"✗ Feature engineering failed: {e}")
        raise


@task(
    name="save_features",
    description="Save features locally"
)
def save_features_task(features_df: pd.DataFrame) -> bool:
    """Task: Save features to storage"""
    try:
        logger.info("Saving features locally...")
        
        local_path = config.DATA_DIR / "processed" / "features.parquet"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_parquet(local_path)
        
        logger.info(f"✓ Features saved to {local_path}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to save features: {e}")
        raise


@task(
    name="prepare_training_data",
    description="Prepare train-test split"
)
def prepare_training_data_task(
    features_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Task: Prepare training and test sets"""
    try:
        logger.info("Preparing training data...")
        
        engineer = FeatureEngineer()
        feature_cols = engineer.get_feature_columns()
        
        # Filter existing columns
        feature_cols = [col for col in feature_cols if col in features_df.columns]
        
        X = features_df[feature_cols].copy()
        y = features_df['rating'].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Time-based split
        split_idx = int(len(X) * config.TRAIN_TEST_SPLIT_RATIO)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        logger.info(f"✓ Training data prepared: train={X_train.shape}, test={X_test.shape}")
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"✗ Data preparation failed: {e}")
        raise


@task(
    name="train_models",
    description="Train and evaluate all models"
)
def train_models_task(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """Task: Train all models and select best"""
    try:
        logger.info("Starting model training task...")
        
        trainer = ModelTrainer()
        results = trainer.train_all_models(X_train, y_train, X_test, y_test)
        
        model_path = trainer.save_best_model()
        
        logger.info(f"✓ Models trained. Best model saved to {model_path}")
        
        return {
            'results': results,
            'best_model': trainer.best_model_name,
            'best_metrics': trainer.best_metrics,
            'model_path': str(model_path)
        }
        
    except Exception as e:
        logger.error(f"✗ Model training failed: {e}")
        raise


@flow(
    name="movielens-training-pipeline",
    description="Complete ML training pipeline with Prefect orchestration",
    task_runner=SequentialTaskRunner()
)
def training_pipeline():
    """Main Prefect flow for ML training pipeline"""
    try:
        logger.info("=" * 80)
        logger.info("🚀 Starting MovieLens ML Training Pipeline (Prefect)")
        logger.info("=" * 80)
        
        # Step 1: Download data
        ratings, movies, users = download_data_task()
        
        # Step 2: Engineer features
        features_df = engineer_features_task(ratings, movies, users)
        
        # Step 3: Save features
        save_features_task(features_df)
        # Step 3.5: Run additional ML tasks
        ml_results = ml_tasks_task(features_df)
        
        # Step 4: Prepare training data
        X_train, X_test, y_train, y_test = prepare_training_data_task(features_df)
        
        # Step 5: Train models
        training_results = train_models_task(X_train, y_train, X_test, y_test)
        
        logger.info("=" * 80)
        logger.info("✅ Pipeline completed successfully!")
        logger.info(f"Best Model: {training_results['best_model']}")
        logger.info(f"Test RMSE: {training_results['best_metrics']['rmse']:.4f}")
        logger.info(f"Test MAE: {training_results['best_metrics']['mae']:.4f}")
        logger.info(f"Test R²: {training_results['best_metrics']['r2']:.4f}")
        logger.info("=" * 80)
        
        return training_results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
@task(name="run_ml_tasks", description="Execute additional ML tasks")
def ml_tasks_task(features_df: pd.DataFrame) -> Dict[str, Any]:
    """Task: Run classification, clustering, PCA, """
    try:
        logger.info("Running additional ML tasks...")
        
        from ml_tasks import MLTasksManager
        
        manager = MLTasksManager(features_df)
        results = manager.run_all_tasks()
        
        logger.info("✓ All ML tasks completed")
        return results
        
    except Exception as e:
        logger.error(f"✗ ML tasks failed: {e}")
        raise


if __name__ == "__main__":
    # Run the Prefect flow
    training_pipeline()