import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Feature Store Configuration (Local mode)
USE_LOCAL_STORAGE = True
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "movielens_mlops")

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "movielens_recommendation")

# Feature Store Configuration
FEATURE_GROUP_NAME = "movielens_features"
FEATURE_GROUP_VERSION = int(os.getenv("FEATURE_GROUP_VERSION", "1"))
FEATURE_VIEW_NAME = "movielens_feature_view"
FEATURE_VIEW_VERSION = int(os.getenv("FEATURE_VIEW_VERSION", "1"))

# Model Configuration
MODEL_REGISTRY_NAME = os.getenv("MODEL_REGISTRY_NAME", "movielens_model")
MODEL_VERSION = os.getenv("MODEL_VERSION", "production")

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Dataset Configuration
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
TRAIN_TEST_SPLIT_RATIO = 0.8
RANDOM_STATE = 42

# Model Hyperparameters
XGBOOST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE
}

LIGHTGBM_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE
}

# Notification Configuration
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
NOTIFICATION_EMAIL = os.getenv("NOTIFICATION_EMAIL", "")

# Monitoring Configuration
DATA_DRIFT_THRESHOLD = 0.1
PERFORMANCE_THRESHOLD = 0.15  # RMSE threshold
MAX_MODEL_AGE_DAYS = int(os.getenv("MAX_MODEL_AGE_DAYS", "7"))
