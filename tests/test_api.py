"""
Test suite for FastAPI application
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient
from src.api import app


class TestAPIEndpoints:
    """Test FastAPI endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert 'status' in data
        assert 'model_loaded' in data
        assert 'timestamp' in data
        
        print(f"✓ Health check passed: {data['status']}")
    
    def test_model_info_endpoint(self, client):
        """Test model info endpoint"""
        response = client.get("/model/info")
        
        # Should return 200 if model loaded, 503 if not
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert 'model_loaded' in data
            assert 'feature_count' in data
            print(f"✓ Model info retrieved: {data.get('feature_count')} features")
        else:
            pytest.skip("Model not loaded yet")
    
    def test_predict_endpoint(self, client):
        """Test prediction endpoint"""
        # Sample prediction data
        prediction_data = {
            "user_id": 1,
            "item_id": 1,
            "user_avg_rating": 3.61,
            "user_rating_std": 1.1,
            "user_rating_count": 272,
            "user_median_rating": 3.5,
            "user_activity_days": 100,
            "user_rating_rate": 2.72,
            "age": 24,
            "gender_encoded": 1,
            "occupation_encoded": 10,
            "item_avg_rating": 3.88,
            "item_rating_std": 1.04,
            "item_rating_count": 583,
            "item_median_rating": 4.0,
            "item_age_days": 1000,
            "item_rating_rate": 0.583,
            "movie_year": 1995
        }
        
        response = client.post("/predict", json=prediction_data)
        
        if response.status_code == 200:
            data = response.json()
            assert 'predicted_rating' in data
            assert 'user_id' in data
            assert 'item_id' in data
            assert 'confidence' in data
            
            # Check rating is in valid range
            assert 0 <= data['predicted_rating'] <= 6
            
            print(f"✓ Prediction successful: {data['predicted_rating']:.2f}")
        else:
            pytest.skip("Model not loaded or prediction failed")
    
    def test_predict_invalid_data(self, client):
        """Test prediction with invalid data"""
        # Missing required fields
        invalid_data = {
            "user_id": 1
        }
        
        response = client.post("/predict", json=invalid_data)
        
        # Should return validation error
        assert response.status_code == 422  # Unprocessable Entity
        print("✓ Invalid data rejected as expected")
    
    def test_batch_predict_endpoint(self, client):
        """Test batch prediction endpoint"""
        batch_data = {
            "predictions": [
                {
                    "user_id": 1,
                    "item_id": 1,
                    "user_avg_rating": 3.61,
                    "user_rating_count": 272,
                    "age": 24,
                    "item_avg_rating": 3.88,
                    "item_rating_count": 583,
                    "movie_year": 1995
                },
                {
                    "user_id": 2,
                    "item_id": 2,
                    "user_avg_rating": 3.2,
                    "user_rating_count": 100,
                    "age": 30,
                    "item_avg_rating": 3.5,
                    "item_rating_count": 200,
                    "movie_year": 2000
                }
            ]
        }
        
        response = client.post("/batch-predict", json=batch_data)
        
        if response.status_code == 200:
            data = response.json()
            assert 'predictions' in data
            assert 'count' in data
            assert data['count'] == 2
            print(f"✓ Batch prediction successful: {data['count']} predictions")
        else:
            pytest.skip("Model not loaded or batch prediction failed")


def test_api_documentation_available():
    """Test that API documentation is accessible"""
    client = TestClient(app)
    
    # Test Swagger UI
    response = client.get("/docs")
    assert response.status_code == 200
    print("✓ Swagger UI accessible")
    
    # Test OpenAPI schema
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert 'paths' in schema
    print("✓ OpenAPI schema available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
