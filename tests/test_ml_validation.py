"""
ML Validation Tests - Compatible with scikit-learn 1.3+
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import MovieLensDataLoader
from src.feature_engineering import FeatureEngineer
import joblib


def test_model_exists():
    """Test that trained model exists"""
    models_dir = Path("models")
    
    # Check if models directory exists
    assert models_dir.exists(), "Models directory not found. Run training first."
    
    # Check for model files
    model_files = list(models_dir.glob("best_model_*.pkl"))
    
    if len(model_files) == 0:
        pytest.skip("No trained models found. Run: python src/prefect_flows.py")
    
    assert len(model_files) > 0, "No model files found"
    print(f"✓ Found {len(model_files)} trained model(s)")


def test_model_loading():
    """Test that model can be loaded"""
    models_dir = Path("models")
    model_files = list(models_dir.glob("best_model_*.pkl"))
    
    if len(model_files) == 0:
        pytest.skip("No trained models found. Run: python src/prefect_flows.py")
    
    model_path = model_files[0]
    
    try:
        model = joblib.load(model_path)
        assert model is not None
        print(f"✓ Model loaded successfully: {model_path.name}")
    except Exception as e:
        pytest.fail(f"Failed to load model: {e}")


def test_model_prediction():
    """Test that model can make predictions"""
    models_dir = Path("models")
    model_files = list(models_dir.glob("best_model_*.pkl"))
    
    if len(model_files) == 0:
        pytest.skip("No trained models found. Run: python src/prefect_flows.py")
    
    model_path = model_files[0]
    model = joblib.load(model_path)
    
    # Create sample input
    sample_features = pd.DataFrame({
        'user_avg_rating': [3.5],
        'user_rating_std': [1.0],
        'user_rating_count': [100],
        'user_median_rating': [3.5],
        'user_activity_days': [30],
        'user_rating_rate': [3.3],
        'age': [25],
        'gender_encoded': [1],
        'occupation_encoded': [5],
        'item_avg_rating': [3.8],
        'item_rating_std': [0.9],
        'item_rating_count': [200],
        'item_median_rating': [4.0],
        'item_age_days': [1000],
        'item_rating_rate': [0.2],
        'movie_year': [1995],
        'year': [2023],
        'month': [6],
        'day': [15],
        'dayofweek': [3],
        'hour': [14],
        'quarter': [2],
        'month_sin': [0.0],
        'month_cos': [1.0],
        'hour_sin': [0.0],
        'hour_cos': [1.0],
        'dayofweek_sin': [0.0],
        'dayofweek_cos': [1.0],
        'is_weekend': [0],
        'user_item_rating_diff': [-0.3],
        'item_user_rating_diff': [0.3],
        'movie_age_at_rating': [28],
        'user_item_popularity': [20000]
    })
    
    try:
        prediction = model.predict(sample_features)
        assert prediction is not None
        assert len(prediction) == 1
        assert 0 <= prediction[0] <= 6  # Allow some margin outside 1-5
        print(f"✓ Model prediction successful: {prediction[0]:.2f}")
    except Exception as e:
        pytest.fail(f"Prediction failed: {e}")


def test_features_file_exists():
    """Test that processed features file exists"""
    features_path = Path("data/processed/features.parquet")
    
    if not features_path.exists():
        pytest.skip("Features not created yet. Run: python src/prefect_flows.py")
    
    assert features_path.exists()
    
    # Try to load it
    try:
        features_df = pd.read_parquet(features_path)
        assert not features_df.empty
        print(f"✓ Features file loaded: {len(features_df)} rows, {len(features_df.columns)} columns")
    except Exception as e:
        pytest.fail(f"Failed to load features: {e}")


def test_feature_ranges():
    """Test that features are in valid ranges"""
    features_path = Path("data/processed/features.parquet")
    
    if not features_path.exists():
        pytest.skip("Features not created yet. Run: python src/prefect_flows.py")
    
    features_df = pd.read_parquet(features_path)
    
    # Test rating columns
    if 'rating' in features_df.columns:
        assert features_df['rating'].between(1, 5).all(), "Ratings should be between 1 and 5"
    
    # Test age
    if 'age' in features_df.columns:
        assert features_df['age'].between(1, 100).all(), "Age should be between 1 and 100"
    
    # Test cyclical features
    cyclical_features = ['month_sin', 'month_cos', 'hour_sin', 'hour_cos']
    for feature in cyclical_features:
        if feature in features_df.columns:
            assert features_df[feature].between(-1, 1).all(), f"{feature} should be between -1 and 1"
    
    # Test is_weekend
    if 'is_weekend' in features_df.columns:
        assert features_df['is_weekend'].isin([0, 1]).all(), "is_weekend should be 0 or 1"
    
    print(f"✓ All feature ranges are valid")


def test_no_missing_values():
    """Test that there are no missing values in key features"""
    features_path = Path("data/processed/features.parquet")
    
    if not features_path.exists():
        pytest.skip("Features not created yet. Run: python src/prefect_flows.py")
    
    features_df = pd.read_parquet(features_path)
    
    # Key features that should never be null
    key_features = ['user_id', 'item_id', 'rating', 'user_avg_rating', 'item_avg_rating']
    
    for feature in key_features:
        if feature in features_df.columns:
            missing_count = features_df[feature].isna().sum()
            assert missing_count == 0, f"{feature} has {missing_count} missing values"
    
    print(f"✓ No missing values in key features")


def test_data_shape_consistency():
    """Test that data shapes are consistent"""
    try:
        loader = MovieLensDataLoader()
        ratings, movies, users = loader.load_all_data()
        
        features_path = Path("data/processed/features.parquet")
        if features_path.exists():
            features_df = pd.read_parquet(features_path)
            
            # Number of rows should match
            assert len(features_df) == len(ratings), \
                f"Features ({len(features_df)}) and ratings ({len(ratings)}) row count mismatch"
            
            print(f"✓ Data shape consistency verified: {len(features_df)} rows")
        else:
            pytest.skip("Features not created yet. Run: python src/prefect_flows.py")
    except FileNotFoundError:
        pytest.skip("Data not downloaded yet. Run: python src/prefect_flows.py")


def test_model_performance_file():
    """Test that model performance is logged"""
    models_dir = Path("models")
    
    if not models_dir.exists():
        pytest.skip("Models directory not found. Run training first.")
    
    # Check for metadata or performance files
    metadata_file = models_dir / "model_metadata.json"
    
    if metadata_file.exists():
        import json
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        assert 'model_name' in metadata
        assert 'metrics' in metadata
        
        print(f"✓ Model metadata found: {metadata.get('model_name')}")
        print(f"  Metrics: RMSE={metadata['metrics'].get('rmse', 'N/A'):.4f}, "
              f"R²={metadata['metrics'].get('r2', 'N/A'):.4f}")
    else:
        pytest.skip("Model metadata not found. This is optional.")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
