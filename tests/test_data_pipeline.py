"""
Test suite for data pipeline components
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


class TestDataLoader:
    """Test cases for MovieLensDataLoader"""
    
    @pytest.fixture
    def loader(self):
        """Create a data loader instance"""
        return MovieLensDataLoader()
    
    def test_load_ratings(self, loader):
        """Test ratings loading"""
        try:
            ratings = loader.load_ratings()
            
            # Check shape
            assert not ratings.empty
            assert len(ratings.columns) >= 4
            
            # Check columns
            assert 'user_id' in ratings.columns
            assert 'item_id' in ratings.columns
            assert 'rating' in ratings.columns
            assert 'timestamp' in ratings.columns
            
            # Check data types
            assert pd.api.types.is_integer_dtype(ratings['user_id'])
            assert pd.api.types.is_integer_dtype(ratings['item_id'])
            assert pd.api.types.is_numeric_dtype(ratings['rating'])
            
            # Check value ranges
            assert ratings['rating'].min() >= 1
            assert ratings['rating'].max() <= 5
            
            print(f"✓ Ratings test passed: {len(ratings)} rows")
        except FileNotFoundError:
            pytest.skip("Data not downloaded yet. Run: python src/prefect_flows.py")
    
    def test_load_movies(self, loader):
        """Test movies loading"""
        try:
            movies = loader.load_movies()
            
            # Check shape
            assert not movies.empty
            
            # Check required columns
            assert 'item_id' in movies.columns
            assert 'title' in movies.columns
            
            print(f"✓ Movies test passed: {len(movies)} rows")
        except FileNotFoundError:
            pytest.skip("Data not downloaded yet. Run: python src/prefect_flows.py")
    
    def test_load_users(self, loader):
        """Test users loading"""
        try:
            users = loader.load_users()
            
            # Check shape
            assert not users.empty
            
            # Check columns
            assert 'user_id' in users.columns
            assert 'age' in users.columns
            assert 'gender' in users.columns
            
            # Check value ranges
            assert users['age'].min() >= 1
            assert users['age'].max() <= 100
            
            print(f"✓ Users test passed: {len(users)} rows")
        except FileNotFoundError:
            pytest.skip("Data not downloaded yet. Run: python src/prefect_flows.py")
    
    def test_load_all_data(self, loader):
        """Test loading all datasets"""
        try:
            ratings, movies, users = loader.load_all_data()
            
            assert not ratings.empty
            assert not movies.empty
            assert not users.empty
            
            # Check relationships
            assert ratings['user_id'].isin(users['user_id']).sum() > 0
            assert ratings['item_id'].isin(movies['item_id']).sum() > 0
            
            print(f"✓ Load all data test passed")
        except FileNotFoundError:
            pytest.skip("Data not downloaded yet. Run: python src/prefect_flows.py")


class TestFeatureEngineer:
    """Test cases for FeatureEngineer"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        ratings = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3],
            'item_id': [1, 2, 1, 3, 2],
            'rating': [5, 4, 3, 5, 4],
            'timestamp': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-02 15:30:00', 
                                         '2023-01-03 20:15:00', '2023-01-04 09:45:00', 
                                         '2023-01-05 18:20:00'])
        })
        
        movies = pd.DataFrame({
            'item_id': [1, 2, 3],
            'title': ['Movie 1 (1995)', 'Movie 2 (2000)', 'Movie 3 (2010)'],
            'year': [1995.0, 2000.0, 2010.0]
        })
        
        users = pd.DataFrame({
            'user_id': [1, 2, 3],
            'age': [25, 30, 35],
            'gender': ['M', 'F', 'M'],
            'occupation': ['engineer', 'teacher', 'doctor']
        })
        
        return ratings, movies, users
    
    @pytest.fixture
    def engineer(self):
        """Create a feature engineer instance"""
        return FeatureEngineer()
    
    def test_create_time_features(self, engineer, sample_data):
        """Test time feature creation"""
        ratings, _, _ = sample_data
        result = engineer.create_time_features(ratings)
        
        # Check new columns exist
        time_features = ['year', 'month', 'day', 'dayofweek', 'hour']
        
        for feature in time_features:
            assert feature in result.columns, f"Missing time feature: {feature}"
        
        # Check value ranges
        assert result['month'].min() >= 1
        assert result['month'].max() <= 12
        assert result['hour'].min() >= 0
        assert result['hour'].max() <= 23
        
        print(f"✓ Time features test passed")
    
    def test_create_user_features(self, engineer, sample_data):
        """Test user feature creation"""
        ratings, _, users = sample_data
        ratings = engineer.create_time_features(ratings)
        result = engineer.create_user_features(ratings, users)
        
        # Check aggregated features exist
        assert 'user_avg_rating' in result.columns
        assert 'user_rating_count' in result.columns
        
        # Check value ranges
        assert result['user_avg_rating'].between(1, 5).all()
        assert result['user_rating_count'].min() >= 1
        
        print(f"✓ User features test passed")
    
    def test_create_item_features(self, engineer, sample_data):
        """Test item feature creation"""
        ratings, movies, _ = sample_data
        ratings = engineer.create_time_features(ratings)
        result = engineer.create_item_features(ratings, movies)
        
        # Check aggregated features exist
        assert 'item_avg_rating' in result.columns
        assert 'item_rating_count' in result.columns
        
        # Check value ranges
        assert result['item_avg_rating'].between(1, 5).all()
        assert result['item_rating_count'].min() >= 1
        
        print(f"✓ Item features test passed")
    
    def test_engineer_features(self, engineer, sample_data):
        """Test full feature engineering pipeline"""
        ratings, movies, users = sample_data
        result = engineer.engineer_features(ratings, movies, users)
        
        # Check output is not empty
        assert not result.empty
        assert len(result) == len(ratings)
        
        # Check all feature categories exist
        assert 'user_avg_rating' in result.columns
        assert 'item_avg_rating' in result.columns
        assert 'age' in result.columns
        
        print(f"✓ Full feature engineering test passed: {len(result.columns)} features")


def test_data_consistency():
    """Test data consistency across loading and feature engineering"""
    try:
        loader = MovieLensDataLoader()
        ratings, movies, users = loader.load_all_data()
        
        engineer = FeatureEngineer()
        features = engineer.engineer_features(ratings, movies, users)
        
        # Check no data loss
        assert len(features) == len(ratings)
        
        # Check ID consistency
        assert features['user_id'].isin(ratings['user_id']).all()
        assert features['item_id'].isin(ratings['item_id']).all()
        
        print(f"✓ Data consistency test passed")
    except FileNotFoundError:
        pytest.skip("Data not downloaded yet. Run: python src/prefect_flows.py")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
