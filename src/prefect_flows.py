import pandas as pd
import numpy as np
from loguru import logger
from typing import Tuple, Dict, Any
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from data_loader import MovieLensDataLoader
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer, ModelEvaluator
import config.config as config


def download_data_task() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Task: Download and load raw data"""
    try:
        logger.info("Starting data download task...")
        loader = MovieLensDataLoader()
        ratings, movies, users = loader.load_all_data()
        
        logger.info(f"✓ Data loaded: {len(ratings)} ratings, {len(movies)} movies, {len(users)} users")
        return ratings, movies, users
        
    except Exception as e:
        logger.error(f"✗ Data download failed: {e}")
        raise


def engineer_features_task(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    users: pd.DataFrame
) -> pd.DataFrame:
    """Task: Feature engineering"""
    try:
        logger.info("Starting feature engineering task...")
        engineer = FeatureEngineer()
        features_df = engineer.engineer_features(ratings, movies, users)
        
        logger.info(f"✓ Features engineered: shape={features_df.shape}")
        return features_df
        
    except Exception as e:
        logger.error(f"✗ Feature engineering failed: {e}")
        raise


def save_to_feature_store_task(features_df: pd.DataFrame) -> bool:
    """Task: Save features locally"""
    try:
        logger.info("Saving features locally...")
        
        # Save locally
        local_path = config.DATA_DIR / "processed" / "features.parquet"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_parquet(local_path)
        
        logger.info(f"✓ Features saved locally to {local_path}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to save features: {e}")
        raise


def prepare_training_data_task(
    features_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Task: Prepare train-test split"""
    try:
        logger.info("Preparing training data...")
        
        # Get feature columns
        engineer = FeatureEngineer()
        feature_cols = engineer.get_feature_columns()
        
        # Ensure all feature columns exist
        missing_cols = [col for col in feature_cols if col not in features_df.columns]
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
            feature_cols = [col for col in feature_cols if col in features_df.columns]
        
        # Prepare features and target
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


def train_models_task(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """Task: Train and evaluate models"""
    try:
        logger.info("Starting model training task...")
        
        trainer = ModelTrainer()
        results = trainer.train_all_models(X_train, y_train, X_test, y_test)
        
        # Save best model
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


def training_pipeline():
    """Main training pipeline"""
    try:
        logger.info("=" * 80)
        logger.info("🚀 Starting MovieLens ML Training Pipeline")
        logger.info("=" * 80)
        
        # Step 1: Download data
        ratings, movies, users = download_data_task()
        
        # Step 2: Engineer features
        features_df = engineer_features_task(ratings, movies, users)
        
        # Step 3: Save features
        save_to_feature_store_task(features_df)
        
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


if __name__ == "__main__":
    # Run the training pipeline
    training_pipeline()