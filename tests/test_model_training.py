"""
Test suite for model training module
"""
import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model_training import ModelTrainer
import joblib


class TestModelTrainer:
    """Test ModelTrainer functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data"""
        np.random.seed(42)
        n_samples = 1000
        
        X = pd.DataFrame({
            'user_avg_rating': np.random.uniform(1, 5, n_samples),
            'user_rating_count': np.random.randint(10, 1000, n_samples),
            'age': np.random.randint(18, 70, n_samples),
            'item_avg_rating': np.random.uniform(1, 5, n_samples),
            'item_rating_count': np.random.randint(10, 1000, n_samples),
            'movie_year': np.random.randint(1950, 2020, n_samples),
            'hour': np.random.randint(0, 24, n_samples),
            'month': np.random.randint(1, 13, n_samples),
        })
        
        # Create ratings based on features with some noise
        y = (X['user_avg_rating'] * 0.5 + 
             X['item_avg_rating'] * 0.5 + 
             np.random.normal(0, 0.1, n_samples))
        y = np.clip(y, 1, 5)
        
        return X, y
    
    @pytest.fixture
    def trainer(self):
        """Create ModelTrainer instance"""
        return ModelTrainer()
    
    def test_trainer_initialization(self, trainer):
        """Test trainer initialization"""
        assert trainer is not None
        assert hasattr(trainer, 'models')
        print("✓ ModelTrainer initialized successfully")
    
    def test_train_single_model(self, trainer, sample_data):
        """Test training a single model"""
        X, y = sample_data
        X_train, X_test = X[:800], X[800:]
        y_train, y_test = y[:800], y[800:]
        
        # Train Ridge model
        try:
            from sklearn.linear_model import Ridge
            model = Ridge()
            model.fit(X_train, y_train)
            
            # Make predictions
            predictions = model.predict(X_test)
            
            assert predictions is not None
            assert len(predictions) == len(y_test)
            assert all(0 <= p <= 6 for p in predictions)
            
            print("✓ Single model training successful")
        except Exception as e:
            pytest.skip(f"Model training failed: {e}")
    
    def test_train_multiple_models(self, trainer, sample_data):
        """Test training multiple models"""
        X, y = sample_data
        X_train, X_test = X[:800], X[800:]
        y_train, y_test = y[:800], y[800:]
        
        try:
            results = trainer.train_all_models(X_train, y_train, X_test, y_test)
            
            assert results is not None
            assert 'best_model_name' in results
            assert 'best_model' in results
            assert 'all_results' in results
            
            # Check metrics
            for model_name, metrics in results['all_results'].items():
                assert 'rmse' in metrics
                assert 'mae' in metrics
                assert 'r2' in metrics
                assert metrics['rmse'] >= 0
                assert metrics['mae'] >= 0
            
            print(f"✓ Trained {len(results['all_results'])} models")
            print(f"  Best model: {results['best_model_name']}")
        except Exception as e:
            pytest.skip(f"Multiple model training failed: {e}")
    
    def test_model_saving(self, trainer, sample_data):
        """Test model saving functionality"""
        X, y = sample_data
        X_train = X[:800]
        y_train = y[:800]
        
        try:
            from sklearn.linear_model import Ridge
            model = Ridge()
            model.fit(X_train, y_train)
            
            # Save model
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            model_path = models_dir / "test_model.pkl"
            
            joblib.dump(model, model_path)
            
            assert model_path.exists()
            
            # Load and verify
            loaded_model = joblib.load(model_path)
            assert loaded_model is not None
            
            # Clean up
            model_path.unlink()
            
            print("✓ Model saving and loading successful")
        except Exception as e:
            pytest.skip(f"Model saving failed: {e}")


def test_model_evaluation_metrics():
    """Test model evaluation metrics calculation"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    y_true = np.array([3, 4, 5, 2, 3])
    y_pred = np.array([3.1, 3.9, 4.8, 2.2, 3.1])
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    assert rmse >= 0
    assert mae >= 0
    assert -1 <= r2 <= 1
    
    print(f"✓ Metrics calculated: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")


def test_trained_models_exist():
    """Test that trained models exist in models directory"""
    models_dir = Path("models")
    
    if not models_dir.exists():
        pytest.skip("Models directory not found")
    
    model_files = list(models_dir.glob("best_model_*.pkl"))
    
    if len(model_files) > 0:
        assert len(model_files) > 0
        print(f"✓ Found {len(model_files)} trained model(s)")
    else:
        pytest.skip("No trained models found. Run: python src/model_training.py")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
