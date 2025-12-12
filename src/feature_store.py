import hopsworks
import pandas as pd
from loguru import logger
from typing import Optional, Tuple
import config.config as config


class FeatureStoreManager:
    """Manage feature storage and retrieval using Hopsworks"""
    
    def __init__(self, api_key: Optional[str] = None, project_name: Optional[str] = None):
        self.api_key = api_key or config.HOPSWORKS_API_KEY
        self.project_name = project_name or config.HOPSWORKS_PROJECT_NAME
        self.project = None
        self.fs = None
        
    def connect(self):
        """Connect to Hopsworks"""
        try:
            logger.info(f"Connecting to Hopsworks project: {self.project_name}")
            self.project = hopsworks.login(
                api_key_value=self.api_key,
                project=self.project_name
            )
            self.fs = self.project.get_feature_store()
            logger.info("Successfully connected to Hopsworks Feature Store")
        except Exception as e:
            logger.error(f"Failed to connect to Hopsworks: {e}")
            raise
    
    def create_feature_group(
        self,
        df: pd.DataFrame,
        name: str = config.FEATURE_GROUP_NAME,
        version: int = config.FEATURE_GROUP_VERSION,
        description: str = "MovieLens features with time series data",
        primary_key: list = None,
        event_time: str = "timestamp"
    ):
        """Create or update feature group"""
        if self.fs is None:
            self.connect()
        
        if primary_key is None:
            primary_key = ['user_id', 'item_id', 'timestamp']
        
        try:
            logger.info(f"Creating/updating feature group: {name} (version {version})")
            
            # Get or create feature group
            feature_group = self.fs.get_or_create_feature_group(
                name=name,
                version=version,
                description=description,
                primary_key=primary_key,
                event_time=event_time,
                online_enabled=True  # Enable online feature serving
            )
            
            # Insert data
            logger.info(f"Inserting {len(df)} records into feature group...")
            feature_group.insert(df, write_options={"wait_for_job": True})
            
            logger.info("Feature group created/updated successfully!")
            return feature_group
            
        except Exception as e:
            logger.error(f"Failed to create feature group: {e}")
            raise
    
    def create_feature_view(
        self,
        feature_group_name: str = config.FEATURE_GROUP_NAME,
        feature_group_version: int = config.FEATURE_GROUP_VERSION,
        view_name: str = config.FEATURE_VIEW_NAME,
        view_version: int = config.FEATURE_VIEW_VERSION,
        description: str = "Feature view for MovieLens recommendation model",
        labels: list = None
    ):
        """Create feature view for training/inference"""
        if self.fs is None:
            self.connect()
        
        if labels is None:
            labels = ['rating']
        
        try:
            logger.info(f"Creating feature view: {view_name} (version {view_version})")
            
            # Get feature group
            feature_group = self.fs.get_feature_group(
                name=feature_group_name,
                version=feature_group_version
            )
            
            # Select all features except labels
            query = feature_group.select_all()
            
            # Create or get feature view
            feature_view = self.fs.get_or_create_feature_view(
                name=view_name,
                version=view_version,
                description=description,
                query=query,
                labels=labels
            )
            
            logger.info("Feature view created successfully!")
            return feature_view
            
        except Exception as e:
            logger.error(f"Failed to create feature view: {e}")
            raise
    
    def get_training_data(
        self,
        view_name: str = config.FEATURE_VIEW_NAME,
        view_version: int = config.FEATURE_VIEW_VERSION,
        train_start: Optional[str] = None,
        train_end: Optional[str] = None,
        test_start: Optional[str] = None,
        test_end: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Retrieve training and test data from feature view"""
        if self.fs is None:
            self.connect()
        
        try:
            logger.info(f"Retrieving training data from feature view: {view_name}")
            
            # Get feature view
            feature_view = self.fs.get_feature_view(
                name=view_name,
                version=view_version
            )
            
            # Get training data
            if train_start and train_end:
                logger.info(f"Time range: {train_start} to {train_end}")
                X_train, X_test, y_train, y_test = feature_view.train_test_split(
                    test_size=0.2,
                    description='MovieLens train-test split'
                )
            else:
                X_train, X_test, y_train, y_test = feature_view.train_test_split(
                    test_size=0.2
                )
            
            logger.info(f"Retrieved training data: X_train={X_train.shape}, y_train={y_train.shape}")
            logger.info(f"Retrieved test data: X_test={X_test.shape}, y_test={y_test.shape}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Failed to retrieve training data: {e}")
            raise
    
    def get_online_features(
        self,
        user_id: int,
        item_id: int,
        view_name: str = config.FEATURE_VIEW_NAME,
        view_version: int = config.FEATURE_VIEW_VERSION
    ) -> pd.DataFrame:
        """Get online features for real-time prediction"""
        if self.fs is None:
            self.connect()
        
        try:
            # Get feature view
            feature_view = self.fs.get_feature_view(
                name=view_name,
                version=view_version
            )
            
            # Get online features
            feature_vector = feature_view.get_feature_vector(
                entry={'user_id': user_id, 'item_id': item_id}
            )
            
            return pd.DataFrame([feature_vector])
            
        except Exception as e:
            logger.error(f"Failed to retrieve online features: {e}")
            raise
    
    def save_feature_metadata(self, metadata: dict, filepath: str):
        """Save feature engineering metadata for reproducibility"""
        import json
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Feature metadata saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    fs_manager = FeatureStoreManager()
    
    # Test connection
    try:
        fs_manager.connect()
        print("✓ Successfully connected to Hopsworks!")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("\nPlease ensure you have:")
        print("1. Created a Hopsworks account at https://www.hopsworks.ai/")
        print("2. Set HOPSWORKS_API_KEY in your .env file")
        print("3. Set HOPSWORKS_PROJECT_NAME in your .env file")
