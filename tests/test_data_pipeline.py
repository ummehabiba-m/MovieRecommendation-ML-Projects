import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_loader import MovieLensDataLoader
from feature_engineering import FeatureEngineer


class TestDataLoader:
    """Test cases for MovieLensDataLoader"""
    
    @pytest.fixture
    def loader(self):
        """Create a data loader instance"""
        return MovieLensDataLoader()
    
    def test_download_data(self, loader):
        """Test data download"""
        extract_path = loader.download_data()
        assert extract_path.exists()
        assert (extract_path / "u.data").exists()
        assert (extract_path / "u.item").exists()
        assert (extract_path / "u.user").exists()
    
    def test_load_ratings(self, loader):
        """Test ratings loading"""
        ratings = loader.load_ratings()
        
        # Check shape
        assert not ratings.empty
        assert len(ratings.columns) == 4
        
        # Check columns
        assert all(col in ratings.columns for col in ['user_id', 'item_id', 'rating', 'timestamp'])
        
        # Check data types
        assert pd.api.types.is_integer_dtype(ratings['user_id'])
        assert pd.api.types.is_integer_dtype(ratings['item_id'])
        assert pd.api.types.is_numeric_dtype(ratings['rating'])
        assert pd.api.types.is_datetime64_any_dtype(ratings['timestamp'])
        
        # Check value ranges
        assert ratings['rating'].min() >= 1
        assert ratings['rating'].max() <= 5
    
    def test_load_movies(self, loader):
        """Test movies loading"""
        movies = loader.load_movies()
        
        # Check shape
        assert not movies.empty
        
        # Check required columns
        assert 'item_id' in movies.columns
        assert 'title' in movies.columns
        assert 'year' in movies.columns
        
        # Check year extraction
        assert movies['year'].notna().any()
        assert movies['year'].min() >= 1900
        assert movies['year'].max() <= 2025
    
    def test_load_users(self, loader):
        """Test users loading"""
        users = loader.load_users()
        
        # Check shape
        assert not users.empty
        
        # Check columns
        assert all(col in users.columns for col in ['user_id', 'age', 'gender', 'occupation'])
        
        # Check value ranges
        assert users['age'].min() >= 1
        assert users['age'].max() <= 100
        assert users['gender'].isin(['M', 'F']).all()
    
    def test_load_all_data(self, loader):
        """Test loading all datasets"""
        ratings, movies, users = loader.load_all_data()
        
        assert not ratings.empty
        assert not movies.empty
        assert not users.empty
        
        # Check relationships
        assert ratings['user_id'].isin(users['user_id']).all()
        assert ratings['item_id'].isin(movies['item_id']).all()


class TestFeatureEngineer:
    """Test cases for FeatureEngineer"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        ratings = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3],
            'item_id': [1, 2, 1, 3, 2],
            'rating': [5, 4, 3, 5, 4],
            'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', 
                                         '2023-01-04', '2023-01-05'])
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
        time_features = ['year', 'month', 'day', 'dayofweek', 'hour', 'quarter',
                        'month_sin', 'month_cos', 'hour_sin', 'hour_cos',
                        'dayofweek_sin', 'dayofweek_cos', 'is_weekend']
        
        assert all(col in result.columns for col in time_features)
        
        # Check value ranges
        assert result['month'].min() >= 1
        assert result['month'].max() <= 12
        assert result['hour'].min() >= 0
        assert result['hour'].max() <= 23
        assert result['is_weekend'].isin([0, 1]).all()
        
        # Check cyclical encoding
        assert result['month_sin'].between(-1, 1).all()
        assert result['month_cos'].between(-1, 1).all()
    
    def test_create_user_features(self, engineer, sample_data):
        """Test user feature creation"""
        ratings, _, users = sample_data
        ratings = engineer.create_time_features(ratings)
        result = engineer.create_user_features(ratings, users)
        
        # Check aggregated features exist
        user_features = ['user_avg_rating', 'user_rating_std', 'user_rating_count',
                        'user_median_rating', 'user_activity_days', 'user_rating_rate']
        
        assert all(col in result.columns for col in user_features)
        
        # Check value ranges
        assert result['user_avg_rating'].between(1, 5).all()
        assert result['user_rating_count'].min() >= 1
        assert result['user_activity_days'].min() >= 0
    
    def test_create_item_features(self, engineer, sample_data):
        """Test item feature creation"""
        ratings, movies, _ = sample_data
        ratings = engineer.create_time_features(ratings)
        result = engineer.create_item_features(ratings, movies)
        
        # Check aggregated features exist
        item_features = ['item_avg_rating', 'item_rating_std', 'item_rating_count',
                        'item_median_rating', 'item_age_days', 'item_rating_rate']
        
        assert all(col in result.columns for col in item_features)
        
        # Check value ranges
        assert result['item_avg_rating'].between(1, 5).all()
        assert result['item_rating_count'].min() >= 1
    
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
        assert 'month_sin' in result.columns
        assert 'occupation_encoded' in result.columns
        
        # Check no missing values in key features
        feature_cols = engineer.get_feature_columns()
        available_cols = [col for col in feature_cols if col in result.columns]
        assert result[available_cols].notna().all().all()
    
    def test_get_feature_columns(self, engineer):
        """Test feature columns retrieval"""
        feature_cols = engineer.get_feature_columns()
        
        assert isinstance(feature_cols, list)
        assert len(feature_cols) > 0
        
        # Check categories are represented
        assert any('user' in col for col in feature_cols)
        assert any('item' in col for col in feature_cols)
        assert any('month' in col or 'hour' in col for col in feature_cols)


def test_data_consistency():
    """Test data consistency across loading and feature engineering"""
    loader = MovieLensDataLoader()
    ratings, movies, users = loader.load_all_data()
    
    engineer = FeatureEngineer()
    features = engineer.engineer_features(ratings, movies, users)
    
    # Check no data loss
    assert len(features) == len(ratings)
    
    # Check ID consistency
    assert features['user_id'].isin(ratings['user_id']).all()
    assert features['item_id'].isin(ratings['item_id']).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
