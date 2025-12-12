# MovieLens MLOps Project 🎬🤖

**End-to-End Machine Learning Deployment & MLOps Pipeline for Movie Rating Prediction**

Domain: **Entertainment & Media**

## 📋 Project Overview

This project implements a complete ML Engineering system for predicting movie ratings using the MovieLens 100K dataset. It demonstrates professional MLOps workflows including:

- ✅ Time series feature engineering
- ✅ Feature store integration (Hopsworks)
- ✅ Model registry (MLflow)
- ✅ FastAPI deployment
- ✅ Workflow orchestration (Prefect)
- ✅ Automated testing (DeepChecks)
- ✅ Docker containerization
- ✅ CI/CD pipeline (GitHub Actions)
- ✅ Data drift monitoring (Evidently)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Ingestion Layer                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  MovieLens   │───▶│   Feature    │───▶│  Hopsworks   │      │
│  │    100K      │    │ Engineering  │    │Feature Store │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Training Pipeline (Prefect)                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ Multi-Model  │───▶│   MLflow     │───▶│   Model      │      │
│  │  Training    │    │  Tracking    │    │  Registry    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Deployment & Monitoring                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   FastAPI    │◀──▶│  DeepChecks  │◀──▶│  Evidently   │      │
│  │     API      │    │   Testing    │    │   Drift      │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
movielens-mlops-project/
│
├── config/
│   └── config.py                 # Configuration settings
│
├── src/
│   ├── data_loader.py            # Data loading utilities
│   ├── feature_engineering.py    # Feature engineering pipeline
│   ├── feature_store.py          # Hopsworks integration
│   ├── model_training.py         # Model training & evaluation
│   ├── api.py                    # FastAPI application
│   ├── prefect_flows.py          # Prefect workflows
│   └── monitoring.py             # Drift & performance monitoring
│
├── tests/
│   ├── test_data_pipeline.py     # Unit tests
│   └── test_ml_validation.py     # ML validation tests
│
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml       # CI/CD pipeline
│
├── docker/
├── data/
├── models/
├── logs/
│
├── Dockerfile                     # Container definition
├── docker-compose.yml             # Multi-container orchestration
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variables template
└── README.md                      # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Git
- Hopsworks account (free tier)
- (Optional) GitHub account for CI/CD

### Step 1: Clone Repository

```bash
git clone <your-repo-url>
cd movielens-mlops-project
```

### Step 2: Set Up Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables

```bash
# Copy example env file
cp .env.example .env

# Edit .env with your credentials
nano .env
```

**Required configurations:**
- `HOPSWORKS_API_KEY`: Get from https://app.hopsworks.ai/
- `HOPSWORKS_PROJECT_NAME`: Your Hopsworks project name

### Step 4: Run Data Pipeline

```bash
# Download data and engineer features
python src/data_loader.py
python src/feature_engineering.py
```

### Step 5: Train Models

```bash
# Run Prefect training pipeline
python src/prefect_flows.py
```

This will:
1. Download MovieLens 100K dataset
2. Engineer 40+ features
3. Upload to Hopsworks Feature Store
4. Train 6 different ML models
5. Log experiments to MLflow
6. Save best model

### Step 6: Start API Server

```bash
# Start FastAPI server
python src/api.py

# Or with uvicorn
uvicorn src.api:app --reload
```

API will be available at: `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs`

### Step 7: Run Tests

```bash
# Unit tests
pytest tests/test_data_pipeline.py -v

# ML validation tests
pytest tests/test_ml_validation.py -v

# All tests with coverage
pytest tests/ -v --cov=src --cov-report=html
```

## 🐳 Docker Deployment

### Option 1: Run API Only

```bash
# Build image
docker build -t movielens-api .

# Run container
docker run -p 8000:8000 -v $(pwd)/models:/app/models movielens-api
```

### Option 2: Full Stack with Docker Compose

```bash
# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop services
docker-compose down
```

This starts:
- FastAPI API (port 8000)
- MLflow Server (port 5000)
- Prefect Server (port 4200)
- Prefect Agent

Access services:
- API: http://localhost:8000
- MLflow: http://localhost:5000
- Prefect: http://localhost:4200

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

The CI/CD pipeline automatically:

1. **Code Quality** - Linting with Black, Flake8, isort
2. **Unit Tests** - Pytest with coverage reporting
3. **ML Validation** - DeepChecks data & model tests
4. **Data Validation** - Schema and quality checks
5. **Model Training** - Automated retraining on schedule
6. **Docker Build** - Build and push Docker images
7. **Deployment** - Deploy to production
8. **Monitoring** - Data drift detection

### Setup GitHub Secrets

Add these secrets to your GitHub repository:

```
HOPSWORKS_API_KEY
DOCKER_USERNAME
DOCKER_PASSWORD
```

### Trigger Pipeline

```bash
# Push to main branch
git push origin main

# Or manually trigger from GitHub Actions tab
```

## 📊 MLflow Tracking

### Start MLflow Server

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000
```

Access UI: http://localhost:5000

### View Experiments

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
experiment = mlflow.get_experiment_by_name("movielens_recommendation")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
print(runs[['metrics.test_rmse', 'metrics.test_r2', 'params.model_name']])
```

## 🔧 Prefect Orchestration

### Start Prefect Server

```bash
prefect server start
```

Access UI: http://localhost:4200

### Deploy Flow

```bash
# Create deployment
python -c "
from src.prefect_flows import training_pipeline
training_pipeline.serve(name='movielens-training')
"
```

### Schedule Runs

```bash
# Daily at 2 AM
prefect deployment schedule create \
  movielens-training \
  --cron "0 2 * * *"
```

## 📡 API Usage Examples

### Health Check

```bash
curl http://localhost:8000/health
```

### Predict Rating

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Batch Prediction

```python
import requests

predictions = [
    {"user_id": 1, "item_id": 1, "age": 24, "user_avg_rating": 3.61},
    {"user_id": 2, "item_id": 5, "age": 30, "user_avg_rating": 3.52}
]

response = requests.post(
    "http://localhost:8000/batch-predict",
    json={"predictions": predictions}
)
print(response.json())
```

## 🔍 Monitoring

### Data Drift Detection

```python
from src.monitoring import DataDriftMonitor
from src.data_loader import MovieLensDataLoader
from src.feature_engineering import FeatureEngineer

# Load data
loader = MovieLensDataLoader()
ratings, movies, users = loader.load_all_data()

engineer = FeatureEngineer()
features = engineer.engineer_features(ratings, movies, users)

# Split into reference and current
split_idx = int(len(features) * 0.8)
reference = features.iloc[:split_idx]
current = features.iloc[split_idx:]

# Check drift
monitor = DataDriftMonitor(reference, current)
drift_result = monitor.check_drift_threshold()

if drift_result['drift_detected']:
    print(f"⚠️ Drift detected in {len(drift_result['drifted_features'])} features")
```

### Performance Monitoring

```python
from src.monitoring import ModelPerformanceMonitor
import joblib

# Load model
model = joblib.load("models/best_model_XGBoost.pkl")

# Monitor performance
monitor = ModelPerformanceMonitor(model)
current_metrics = monitor.evaluate_performance(X_test, y_test, "current")

# Check degradation
degradation = monitor.check_performance_degradation(
    current_metrics,
    baseline_metrics,
    threshold=0.15
)
```

## 📈 Model Performance

Current best model: **XGBoost**

| Metric | Training | Test |
|--------|----------|------|
| RMSE   | 0.85     | 0.92 |
| MAE    | 0.67     | 0.72 |
| R²     | 0.45     | 0.41 |

## 🛠️ Troubleshooting

### Common Issues

**1. Hopsworks Connection Failed**
```bash
# Check API key
echo $HOPSWORKS_API_KEY

# Test connection
python -c "import hopsworks; hopsworks.login()"
```

**2. Model Not Found**
```bash
# Train model first
python src/prefect_flows.py
```

**3. Port Already in Use**
```bash
# Check what's using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>
```

## 📚 Additional Resources

- [Hopsworks Documentation](https://docs.hopsworks.ai/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Prefect Documentation](https://docs.prefect.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [DeepChecks Documentation](https://docs.deepchecks.com/)

## 🎯 Project Deliverables Checklist

- [x] FastAPI deployment
- [x] Prefect orchestration
- [x] Docker containerization
- [x] CI/CD pipeline (GitHub Actions)
- [x] Feature store (Hopsworks)
- [x] Model registry (MLflow)
- [x] Automated testing (pytest + DeepChecks)
- [x] Data drift monitoring (Evidently)
- [x] Multiple ML models comparison
- [x] Time series features
- [x] Comprehensive documentation

## 👥 Team & Contact

**Project**: ML Engineering (AI321L)  
**Institution**: GIKI  
**Instructor**: Asim Shah

---

Made with ❤️ using MLOps best practices
