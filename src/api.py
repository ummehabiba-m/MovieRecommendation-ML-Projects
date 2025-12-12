from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
from loguru import logger
from pathlib import Path
import config.config as config
from datetime import datetime
import json

# Initialize FastAPI app
app = FastAPI(
    title="MovieLens Rating Prediction API",
    description="ML-powered movie rating prediction with time series features",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
feature_columns = None
model_metadata = None


# Pydantic models for request/response
class PredictionInput(BaseModel):
    user_id: int = Field(..., description="User ID", ge=1)
    item_id: int = Field(..., description="Movie/Item ID", ge=1)
    timestamp: Optional[str] = Field(None, description="Timestamp (ISO format)")
    
    # User features
    user_avg_rating: float = Field(3.5, ge=0, le=5)
    user_rating_std: float = Field(1.0, ge=0)
    user_rating_count: int = Field(10, ge=0)
    user_median_rating: float = Field(3.5, ge=0, le=5)
    user_activity_days: int = Field(30, ge=0)
    user_rating_rate: float = Field(0.5, ge=0)
    age: int = Field(25, ge=1, le=100)
    gender_encoded: int = Field(1, ge=0, le=1)
    occupation_encoded: int = Field(0, ge=0)
    
    # Item features
    item_avg_rating: float = Field(3.5, ge=0, le=5)
    item_rating_std: float = Field(1.0, ge=0)
    item_rating_count: int = Field(50, ge=0)
    item_median_rating: float = Field(3.5, ge=0, le=5)
    item_age_days: int = Field(100, ge=0)
    item_rating_rate: float = Field(1.0, ge=0)
    movie_year: int = Field(1995, ge=1900, le=2025)
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 1,
                "item_id": 1,
                "timestamp": "1997-12-04T15:30:00",
                "user_avg_rating": 3.61,
                "user_rating_std": 1.1,
                "user_rating_count": 272,
                "age": 24,
                "item_avg_rating": 3.88,
                "item_rating_count": 583,
                "movie_year": 1995
            }
        }


class PredictionOutput(BaseModel):
    user_id: int
    item_id: int
    predicted_rating: float
    confidence: str
    timestamp: str
    model_version: str


class BatchPredictionInput(BaseModel):
    predictions: List[PredictionInput]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str]
    timestamp: str


def load_model():
    """Load the trained model"""
    global model, feature_columns, model_metadata
    
    try:
        # Load model
        model_path = config.MODELS_DIR / "best_model_XGBoost.pkl"
        if not model_path.exists():
            # Try other models
            for model_file in config.MODELS_DIR.glob("best_model_*.pkl"):
                model_path = model_file
                break
        
        if not model_path.exists():
            logger.warning("No trained model found. Please train a model first.")
            return False
        
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Load metadata
        metadata_path = config.MODELS_DIR / "model_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
        
        # Define feature columns (should match training)
        feature_columns = [
            'user_avg_rating', 'user_rating_std', 'user_rating_count',
            'user_median_rating', 'user_activity_days', 'user_rating_rate',
            'age', 'gender_encoded', 'occupation_encoded',
            'item_avg_rating', 'item_rating_std', 'item_rating_count',
            'item_median_rating', 'item_age_days', 'item_rating_rate',
            'movie_year', 'year', 'month', 'day', 'dayofweek', 'hour',
            'quarter', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos',
            'dayofweek_sin', 'dayofweek_cos', 'is_weekend',
            'user_item_rating_diff', 'item_user_rating_diff',
            'movie_age_at_rating', 'user_item_popularity'
        ]
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


def create_time_features(timestamp_str: Optional[str] = None) -> dict:
    """Create time-based features from timestamp"""
    if timestamp_str:
        try:
            ts = pd.to_datetime(timestamp_str)
        except:
            ts = pd.Timestamp.now()
    else:
        ts = pd.Timestamp.now()
    
    # Convert to datetime if it's a Timestamp
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    
    features = {
        'year': ts.year,
        'month': ts.month,
        'day': ts.day,
        'dayofweek': ts.weekday(),  # Changed from ts.dayofweek
        'hour': ts.hour,
        'quarter': (ts.month - 1) // 3 + 1,
        'month_sin': np.sin(2 * np.pi * ts.month / 12),
        'month_cos': np.cos(2 * np.pi * ts.month / 12),
        'hour_sin': np.sin(2 * np.pi * ts.hour / 24),
        'hour_cos': np.cos(2 * np.pi * ts.hour / 24),
        'dayofweek_sin': np.sin(2 * np.pi * ts.weekday() / 7),  # Changed
        'dayofweek_cos': np.cos(2 * np.pi * ts.weekday() / 7),  # Changed
        'is_weekend': 1 if ts.weekday() >= 5 else 0  # Changed
    }
    
    return features

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting MovieLens Rating Prediction API...")
    load_model()


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "MovieLens Rating Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "model_info": "/model/info"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=model_metadata.get('model_name') if model_metadata else None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict_rating(input_data: PredictionInput):
    """Predict rating for a single user-item pair"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Create time features
        time_features = create_time_features(input_data.timestamp)
        
        # Combine all features
        features_dict = input_data.dict()
        features_dict.update(time_features)
        
        # Calculate interaction features
        features_dict['user_item_rating_diff'] = (
            features_dict['user_avg_rating'] - features_dict['item_avg_rating']
        )
        features_dict['item_user_rating_diff'] = (
            features_dict['item_avg_rating'] - features_dict['user_avg_rating']
        )
        features_dict['movie_age_at_rating'] = (
            features_dict['year'] - features_dict['movie_year']
        )
        features_dict['user_item_popularity'] = (
            features_dict['user_rating_count'] * features_dict['item_rating_count']
        )
        
        # Create DataFrame with correct column order
        features_df = pd.DataFrame([features_dict])
        features_df = features_df[feature_columns]
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        
        # Clip prediction to valid range
        prediction = np.clip(prediction, 1.0, 5.0)
        
        # Determine confidence based on standard deviations
        avg_std = (features_dict['user_rating_std'] + features_dict['item_rating_std']) / 2
        confidence = "high" if avg_std < 0.8 else "medium" if avg_std < 1.2 else "low"
        
        return PredictionOutput(
            user_id=input_data.user_id,
            item_id=input_data.item_id,
            predicted_rating=round(prediction, 2),
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            model_version=model_metadata.get('model_name', 'unknown') if model_metadata else 'unknown'
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch-predict")
async def batch_predict(batch_input: BatchPredictionInput):
    """Predict ratings for multiple user-item pairs"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        for input_data in batch_input.predictions:
            result = await predict_rating(input_data)
            predictions.append(result)
        
        return {
            "predictions": predictions,
            "count": len(predictions),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get model information and metadata"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = {
        "model_loaded": True,
        "feature_count": len(feature_columns),
        "features": feature_columns
    }
    
    if model_metadata:
        info.update({
            "model_name": model_metadata.get('model_name'),
            "metrics": model_metadata.get('metrics'),
            "trained_at": model_metadata.get('timestamp')
        })
    
    return info


@app.post("/model/reload")
async def reload_model():
    """Reload the model (useful after retraining)"""
    success = load_model()
    if success:
        return {"status": "success", "message": "Model reloaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
        log_level="info"
    )
