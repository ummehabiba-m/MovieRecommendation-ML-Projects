import pandas as pd
import numpy as np
from loguru import logger
from typing import Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler


class FeatureEngineer:
    """Engineer features for movie recommendation and rating prediction"""
    
    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.occupation_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from timestamp"""
        df = df.copy()
        
        # Extract time components
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        df['hour'] = df['timestamp'].dt.hour
        df['quarter'] = df['timestamp'].dt.quarter
        
        # Cyclical encoding for periodic features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        # Is weekend
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        
        logger.info("Created time-based features")
        return df
    
    def create_user_features(self, ratings: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
        """Create user-based aggregated features"""
        df = ratings.copy()
        
        # User statistics
        user_stats = df.groupby('user_id').agg({
            'rating': ['mean', 'std', 'count', 'median'],
            'timestamp': ['min', 'max']
        }).reset_index()
        
        user_stats.columns = ['user_id', 'user_avg_rating', 'user_rating_std', 
                              'user_rating_count', 'user_median_rating',
                              'user_first_rating_time', 'user_last_rating_time']
        
        # User activity duration (in days)
        user_stats['user_activity_days'] = (
            user_stats['user_last_rating_time'] - user_stats['user_first_rating_time']
        ).dt.days
        
        # User rating rate (ratings per day)
        user_stats['user_rating_rate'] = (
            user_stats['user_rating_count'] / (user_stats['user_activity_days'] + 1)
        )
        
        # Fill NaN in std with 0
        user_stats['user_rating_std'].fillna(0, inplace=True)
        
        # Merge with user demographics
        user_stats = user_stats.merge(users, on='user_id', how='left')
        
        logger.info("Created user-based features")
        return user_stats
    
    def create_item_features(self, ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
        """Create item-based aggregated features"""
        df = ratings.copy()
        
        # Item statistics
        item_stats = df.groupby('item_id').agg({
            'rating': ['mean', 'std', 'count', 'median'],
            'timestamp': ['min', 'max']
        }).reset_index()
        
        item_stats.columns = ['item_id', 'item_avg_rating', 'item_rating_std',
                             'item_rating_count', 'item_median_rating',
                             'item_first_rating_time', 'item_last_rating_time']
        
        # Item popularity metrics
        item_stats['item_age_days'] = (
            item_stats['item_last_rating_time'] - item_stats['item_first_rating_time']
        ).dt.days
        
        item_stats['item_rating_rate'] = (
            item_stats['item_rating_count'] / (item_stats['item_age_days'] + 1)
        )
        
        # Fill NaN in std with 0
        item_stats['item_rating_std'].fillna(0, inplace=True)
        
        # Merge with movie information
        item_stats = item_stats.merge(movies[['item_id', 'year']], on='item_id', how='left')
        item_stats.rename(columns={'year': 'movie_year'}, inplace=True)
        
        logger.info("Created item-based features")
        return item_stats
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create user-item interaction features"""
        df = df.copy()
        
        # User-item rating deviation
        df['user_item_rating_diff'] = df['rating'] - df['user_avg_rating']
        df['item_user_rating_diff'] = df['rating'] - df['item_avg_rating']
        
        # Age-based features
        df['user_age_group'] = pd.cut(df['age'], bins=[0, 18, 30, 50, 100], 
                                       labels=['teen', 'young', 'middle', 'senior'])
        
        # Movie age at time of rating
        df['movie_age_at_rating'] = df['year'] - df['movie_year']
        
        # Interaction between user rating count and item rating count
        df['user_item_popularity'] = df['user_rating_count'] * df['item_rating_count']
        
        logger.info("Created interaction features")
        return df
    
    def engineer_features(
        self,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
        users: pd.DataFrame
    ) -> pd.DataFrame:
        """Main feature engineering pipeline"""
        logger.info("Starting feature engineering...")
        
        # Create time features
        ratings = self.create_time_features(ratings)
        
        # Create user features
        user_features = self.create_user_features(ratings, users)
        
        # Create item features
        item_features = self.create_item_features(ratings, movies)
        
        # Merge all features
        df = ratings.merge(user_features, on='user_id', how='left')
        df = df.merge(item_features, on='item_id', how='left')
        
        # Create interaction features
        df = self.create_interaction_features(df)
        
        # Encode categorical variables
        df['occupation_encoded'] = self.occupation_encoder.fit_transform(df['occupation'])
        df['gender_encoded'] = df['gender'].map({'M': 1, 'F': 0})
        
        logger.info(f"Feature engineering complete. Final shape: {df.shape}")
        logger.info(f"Features: {df.columns.tolist()}")
        
        return df
    
    def get_feature_columns(self) -> list:
        """Get list of feature columns for model training"""
        return [
            # User features
            'user_avg_rating', 'user_rating_std', 'user_rating_count',
            'user_median_rating', 'user_activity_days', 'user_rating_rate',
            'age', 'gender_encoded', 'occupation_encoded',
            
            # Item features
            'item_avg_rating', 'item_rating_std', 'item_rating_count',
            'item_median_rating', 'item_age_days', 'item_rating_rate',
            'movie_year',
            
            # Time features
            'year', 'month', 'day', 'dayofweek', 'hour', 'quarter',
            'month_sin', 'month_cos', 'hour_sin', 'hour_cos',
            'dayofweek_sin', 'dayofweek_cos', 'is_weekend',
            
            # Interaction features
            'user_item_rating_diff', 'item_user_rating_diff',
            'movie_age_at_rating', 'user_item_popularity'
        ]


if __name__ == "__main__":
    from data_loader import MovieLensDataLoader
    
    # Load data
    loader = MovieLensDataLoader()
    ratings, movies, users = loader.load_all_data()
    
    # Engineer features
    engineer = FeatureEngineer()
    features_df = engineer.engineer_features(ratings, movies, users)
    
    print("\nEngineered features shape:", features_df.shape)
    print("\nFeature columns:")
    print(engineer.get_feature_columns())
    print("\nSample features:")
    print(features_df[engineer.get_feature_columns()].head())
