from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
from loguru import logger
from pathlib import Path
import sys
from datetime import datetime
import json

# Fix import paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import config

# --- PROJECT PATHS ---
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
INDEX_FILE = FRONTEND_DIR / "index.html"
STATIC_DIR = FRONTEND_DIR / "static"

# --- INITIALIZE APP ---
app = FastAPI(
    title="MovieLens Rating Prediction API",
    description="ML-powered movie rating prediction with time series features",
    version="1.0.0"
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- STATIC FILES ---
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- GLOBAL MODEL VARIABLES ---
model = None
feature_columns = None
model_metadata = None

# --- Pydantic MODELS ---
class PredictionInput(BaseModel):
    user_id: int = Field(..., ge=1)
    item_id: int = Field(..., ge=1)
    timestamp: Optional[str] = Field(None)
    user_avg_rating: float = Field(3.5, ge=0, le=5)
    user_rating_std: float = Field(1.0, ge=0)
    user_rating_count: int = Field(10, ge=0)
    user_median_rating: float = Field(3.5, ge=0, le=5)
    user_activity_days: int = Field(30, ge=0)
    user_rating_rate: float = Field(0.5, ge=0)
    age: int = Field(25, ge=1, le=100)
    gender_encoded: int = Field(1, ge=0, le=1)
    occupation_encoded: int = Field(0, ge=0)
    item_avg_rating: float = Field(3.5, ge=0, le=5)
    item_rating_std: float = Field(1.0, ge=0)
    item_rating_count: int = Field(50, ge=0)
    item_median_rating: float = Field(3.5, ge=0, le=5)
    item_age_days: int = Field(100, ge=0)
    item_rating_rate: float = Field(1.0, ge=0)
    movie_year: int = Field(1995, ge=1900, le=2025)

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

# --- MODEL FUNCTIONS ---
def load_model():
    global model, feature_columns, model_metadata
    try:
        model_path = config.MODELS_DIR / "best_model_XGBoost.pkl"
        if not model_path.exists():
            for model_file in config.MODELS_DIR.glob("best_model_*.pkl"):
                model_path = model_file
                break
        if not model_path.exists():
            logger.warning("No trained model found. Please train a model first.")
            return False

        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")

        metadata_path = config.MODELS_DIR / "model_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                model_metadata = json.load(f)

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
    ts = pd.Timestamp.now()
    if timestamp_str:
        try:
            ts = pd.to_datetime(timestamp_str)
        except:
            ts = pd.Timestamp.now()
    ts = ts.to_pydatetime()
    features = {
        'year': ts.year, 'month': ts.month, 'day': ts.day,
        'dayofweek': ts.weekday(), 'hour': ts.hour,
        'quarter': (ts.month - 1) // 3 + 1,
        'month_sin': np.sin(2 * np.pi * ts.month / 12),
        'month_cos': np.cos(2 * np.pi * ts.month / 12),
        'hour_sin': np.sin(2 * np.pi * ts.hour / 24),
        'hour_cos': np.cos(2 * np.pi * ts.hour / 24),
        'dayofweek_sin': np.sin(2 * np.pi * ts.weekday() / 7),
        'dayofweek_cos': np.cos(2 * np.pi * ts.weekday() / 7),
        'is_weekend': 1 if ts.weekday() >= 5 else 0
    }
    return features

@app.on_event("startup")
async def startup_event():
    logger.info("Starting MovieLens Rating Prediction API...")
    load_model()

# --- API ENDPOINTS ---
@app.get("/api", response_model=dict)
async def root():
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
    return HealthResponse(
        status="healthy" if model else "unhealthy",
        model_loaded=bool(model),
        model_version=model_metadata.get("model_name") if model_metadata else None,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionOutput)
async def predict_rating(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        features_dict = input_data.dict()
        features_dict.update(create_time_features(input_data.timestamp))
        features_dict['user_item_rating_diff'] = features_dict['user_avg_rating'] - features_dict['item_avg_rating']
        features_dict['item_user_rating_diff'] = features_dict['item_avg_rating'] - features_dict['user_avg_rating']
        features_dict['movie_age_at_rating'] = features_dict['year'] - features_dict['movie_year']
        features_dict['user_item_popularity'] = features_dict['user_rating_count'] * features_dict['item_rating_count']
        features_df = pd.DataFrame([features_dict])[feature_columns]
        prediction = np.clip(model.predict(features_df)[0], 1.0, 5.0)
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
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.post("/batch-predict")
async def batch_predict(batch_input: BatchPredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        predictions = [await predict_rating(item) for item in batch_input.predictions]
        return {"predictions": predictions, "count": len(predictions), "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")

@app.get("/model/info")
async def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    info = {"model_loaded": True, "feature_count": len(feature_columns), "features": feature_columns}
    if model_metadata:
        info.update({
            "model_name": model_metadata.get("model_name"),
            "metrics": model_metadata.get("metrics"),
            "trained_at": model_metadata.get("timestamp")
        })
    return info

@app.post("/model/reload")
async def reload_model():
    success = load_model()
    if success:
        return {"status": "success", "message": "Model reloaded successfully"}
    raise HTTPException(status_code=500, detail="Failed to reload model")

# --- FRONTEND ROUTES ---
@app.get("/")
async def serve_frontend():
    if INDEX_FILE.exists():
        return FileResponse(str(INDEX_FILE))
    raise HTTPException(status_code=404, detail="Frontend not found")

@app.get("/{full_path:path}")
async def catch_all_routes(full_path: str):
    if INDEX_FILE.exists():
        return FileResponse(str(INDEX_FILE))
    raise HTTPException(status_code=404, detail="Frontend not found")

# --- RUN SERVER ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host=config.API_HOST, port=config.API_PORT, reload=True, log_level="info")
