# 🚀 Step-by-Step Implementation Guide

## Complete Walkthrough for Running the MovieLens MLOps Project

This guide will walk you through every step needed to successfully run this project from scratch.

---

## 📋 Pre-requisites Checklist

Before starting, ensure you have:

- [ ] Python 3.10 or higher installed
- [ ] Git installed
- [ ] Docker and Docker Compose installed (optional but recommended)
- [ ] At least 4GB free disk space
- [ ] Internet connection for downloading datasets and packages

---

## 🎯 Phase 1: Initial Setup (15 minutes)

### Step 1.1: Clone the Repository

```bash
# Navigate to your workspace
cd ~/workspace  # or wherever you keep projects

# Clone the repository
git clone <your-repo-url>
cd movielens-mlops-project

# Or if you're starting from scratch, create the directory
mkdir movielens-mlops-project
cd movielens-mlops-project
```

### Step 1.2: Quick Setup with Script

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup script
./setup.sh
```

**Or Manual Setup:**

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Step 1.3: Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit with your preferred editor
nano .env  # or vim, code, etc.
```

**Important: Update these values in .env:**

```env
# Get API key from https://app.hopsworks.ai/
HOPSWORKS_API_KEY=your_actual_api_key_here
HOPSWORKS_PROJECT_NAME=movielens_mlops

# Optional: Discord webhook for notifications
DISCORD_WEBHOOK_URL=your_webhook_url
```

**Getting Hopsworks API Key:**
1. Go to https://app.hopsworks.ai/
2. Sign up for free account
3. Create a new project named "movielens_mlops"
4. Go to Settings → API Keys
5. Generate new API key
6. Copy and paste into .env file

---

## 📊 Phase 2: Data Pipeline (10 minutes)

### Step 2.1: Download Dataset

```bash
# Test data loader
python src/data_loader.py
```

**Expected output:**
```
Downloading MovieLens 100K from https://...
✓ Downloaded 100,000 ratings
✓ Loaded 1,682 movies
✓ Loaded 943 users
```

**Troubleshooting:**
- If download fails, check internet connection
- If extraction fails, manually download from https://grouplens.org/datasets/movielens/100k/

### Step 2.2: Engineer Features

```bash
# Run feature engineering
python -c "
from src.data_loader import MovieLensDataLoader
from src.feature_engineering import FeatureEngineer

# Load data
loader = MovieLensDataLoader()
ratings, movies, users = loader.load_all_data()

# Engineer features
engineer = FeatureEngineer()
features = engineer.engineer_features(ratings, movies, users)

print(f'✓ Created {features.shape[1]} features')
print(f'✓ Total samples: {len(features)}')
"
```

**Expected output:**
```
✓ Created 45 features
✓ Total samples: 100,000
```

### Step 2.3: Test Feature Store Connection (Optional)

```bash
# Test Hopsworks connection
python -c "
from src.feature_store import FeatureStoreManager

fs = FeatureStoreManager()
try:
    fs.connect()
    print('✓ Hopsworks connection successful!')
except Exception as e:
    print(f'✗ Connection failed: {e}')
    print('  You can still run locally without Hopsworks')
"
```

**Note:** If Hopsworks connection fails, the project will automatically save features locally and continue working.

---

## 🤖 Phase 3: Model Training (20-30 minutes)

### Step 3.1: Start MLflow Server (Optional but Recommended)

```bash
# In a new terminal window
cd movielens-mlops-project
source venv/bin/activate

# Start MLflow server
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000
```

**Access MLflow UI:** http://localhost:5000

### Step 3.2: Run Training Pipeline

```bash
# Back in main terminal
# Run complete training pipeline with Prefect
python src/prefect_flows.py
```

**What happens:**
1. Downloads data (if not already done)
2. Engineers features
3. Attempts to save to Hopsworks (falls back to local if unavailable)
4. Trains 6 different models:
   - Random Forest
   - XGBoost
   - LightGBM
   - Gradient Boosting
   - Ridge Regression
   - Elastic Net
5. Logs all experiments to MLflow
6. Saves best model

**Expected output:**
```
🚀 Starting MovieLens ML Training Pipeline
================================================================================
✓ Data loaded: 100000 ratings, 1682 movies, 943 users
✓ Features engineered: shape=(100000, 45)
✓ Training data prepared: train=(80000, 43), test=(20000, 43)

Training RandomForest...
RandomForest - Test RMSE: 0.9234, Test MAE: 0.7123, Test R²: 0.4012

Training XGBoost...
XGBoost - Test RMSE: 0.9180, Test MAE: 0.7089, Test R²: 0.4156

Training LightGBM...
LightGBM - Test RMSE: 0.9201, Test MAE: 0.7105, Test R²: 0.4089

...

Best Model: XGBoost
Best RMSE: 0.9180
✓ Model saved to models/best_model_XGBoost.pkl
================================================================================
✅ Pipeline completed successfully!
```

**Troubleshooting:**
- If training is slow, reduce n_estimators in config/config.py
- If memory issues occur, use a smaller subset of data
- Check MLflow UI to monitor training progress

### Step 3.3: Verify Model Saved

```bash
# Check if model exists
ls -lh models/

# Should see:
# best_model_XGBoost.pkl (or another model name)
# model_metadata.json
# *_feature_importance.csv
```

---

## 🚀 Phase 4: API Deployment (5 minutes)

### Step 4.1: Start FastAPI Server

```bash
# Method 1: Direct Python
python src/api.py

# Method 2: With Uvicorn (recommended for development)
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

**Expected output:**
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 4.2: Test API

```bash
# In a new terminal

# 1. Health check
curl http://localhost:8000/health

# Expected: {"status":"healthy","model_loaded":true,...}

# 2. Test prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "item_id": 1,
    "user_avg_rating": 3.61,
    "user_rating_count": 272,
    "age": 24,
    "item_avg_rating": 3.88,
    "item_rating_count": 583,
    "movie_year": 1995
  }'

# Expected: {"predicted_rating":3.85,"confidence":"high",...}
```

### Step 4.3: Explore Interactive API Docs

Open in browser: http://localhost:8000/docs

You'll see an interactive Swagger UI where you can:
- View all endpoints
- Test API calls directly from browser
- See request/response schemas

**Try it:**
1. Click on "POST /predict"
2. Click "Try it out"
3. Edit the JSON request body
4. Click "Execute"
5. See the response

---

## 🧪 Phase 5: Testing (10 minutes)

### Step 5.1: Run Unit Tests

```bash
# Run data pipeline tests
pytest tests/test_data_pipeline.py -v

# Expected: All tests should pass
# test_download_data PASSED
# test_load_ratings PASSED
# test_create_time_features PASSED
# ...
```

### Step 5.2: Run ML Validation Tests

```bash
# Run ML validation tests with DeepChecks
pytest tests/test_ml_validation.py -v

# This will:
# - Check data quality
# - Validate model performance
# - Generate HTML reports
```

### Step 5.3: View Test Reports

```bash
# Open test reports in browser
open models/data_integrity_report.html
open models/model_evaluation_report.html

# Or on Linux:
xdg-open models/data_integrity_report.html
```

### Step 5.4: Run All Tests with Coverage

```bash
# Run all tests with coverage report
pytest tests/ -v --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

---

## 🐳 Phase 6: Docker Deployment (10 minutes)

### Step 6.1: Build Docker Image

```bash
# Build the image
docker build -t movielens-api .

# Verify image created
docker images | grep movielens
```

### Step 6.2: Run Container

```bash
# Run API container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  --name movielens-api \
  movielens-api

# Check logs
docker logs -f movielens-api

# Test API
curl http://localhost:8000/health
```

### Step 6.3: Full Stack with Docker Compose

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# Should show:
# movielens-api        running
# movielens-mlflow     running
# movielens-prefect    running

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

**Access Services:**
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- MLflow: http://localhost:5000
- Prefect: http://localhost:4200

---

## 🔍 Phase 7: Monitoring (Optional)

### Step 7.1: Run Data Drift Detection

```bash
python -c "
from src.monitoring import DataDriftMonitor
from src.data_loader import MovieLensDataLoader
from src.feature_engineering import FeatureEngineer

# Load data
loader = MovieLensDataLoader()
ratings, movies, users = loader.load_all_data()

# Engineer features
engineer = FeatureEngineer()
features = engineer.engineer_features(ratings, movies, users)

# Split for drift detection
split_idx = int(len(features) * 0.8)
reference = features.iloc[:split_idx]
current = features.iloc[split_idx:]

# Check drift
monitor = DataDriftMonitor(reference, current)
result = monitor.check_drift_threshold()

print(f'Drift detected: {result[\"drift_detected\"]}')
print(f'Drift ratio: {result[\"drift_ratio\"]:.2%}')
"
```

### Step 7.2: Monitor Model Performance

```bash
python -c "
from src.monitoring import ModelPerformanceMonitor
import joblib
import pandas as pd

# Load model and test data
model = joblib.load('models/best_model_XGBoost.pkl')

# Load test data (simplified)
print('Model performance monitoring ready')
print('See src/monitoring.py for full examples')
"
```

---

## 🔄 Phase 8: CI/CD Setup (15 minutes)

### Step 8.1: Push to GitHub

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Complete MLOps pipeline"

# Add remote
git remote add origin <your-github-repo-url>

# Push
git push -u origin main
```

### Step 8.2: Configure GitHub Secrets

1. Go to your GitHub repository
2. Click Settings → Secrets and variables → Actions
3. Add these secrets:
   - `HOPSWORKS_API_KEY`
   - `DOCKER_USERNAME`
   - `DOCKER_PASSWORD`

### Step 8.3: Enable GitHub Actions

1. Go to Actions tab
2. Click "I understand my workflows, go ahead and enable them"
3. The pipeline will run automatically on:
   - Every push to main
   - Every pull request
   - Daily at 2 AM UTC (scheduled training)

### Step 8.4: Monitor CI/CD

```bash
# Push a change to trigger pipeline
echo "# Test change" >> README.md
git add README.md
git commit -m "Test CI/CD pipeline"
git push

# Go to GitHub Actions tab to see pipeline running
```

---

## 🎓 Phase 9: Prefect Orchestration (Optional)

### Step 9.1: Start Prefect Server

```bash
# In a new terminal
prefect server start
```

**Access Prefect UI:** http://localhost:4200

### Step 9.2: Create Deployment

```bash
# In main terminal
python -c "
from src.prefect_flows import training_pipeline

# Create deployment
training_pipeline.serve(
    name='movielens-training',
    cron='0 2 * * *'  # Daily at 2 AM
)
"
```

### Step 9.3: Run Flow

```bash
# Manual run
prefect deployment run 'movielens_training_pipeline/movielens-training'

# Or trigger from Prefect UI
```

---

## 📊 Success Verification Checklist

Verify your setup is working correctly:

- [ ] ✅ Data downloads successfully (100K ratings)
- [ ] ✅ Features engineered (45+ features)
- [ ] ✅ Models trained (6 models compared)
- [ ] ✅ Best model saved (RMSE < 1.0)
- [ ] ✅ API server runs (port 8000)
- [ ] ✅ API health check passes
- [ ] ✅ Prediction endpoint works
- [ ] ✅ All unit tests pass
- [ ] ✅ ML validation tests pass
- [ ] ✅ Docker container builds
- [ ] ✅ Docker compose starts all services
- [ ] ✅ MLflow UI accessible (port 5000)
- [ ] ✅ GitHub Actions pipeline runs

---

## 🐛 Common Issues & Solutions

### Issue 1: Import Errors

```bash
# Error: ModuleNotFoundError: No module named 'src'

# Solution: Make sure you're in project root and venv is activated
cd movielens-mlops-project
source venv/bin/activate
```

### Issue 2: Hopsworks Connection Failed

```bash
# Error: Failed to connect to Hopsworks

# Solution 1: Check API key in .env
cat .env | grep HOPSWORKS_API_KEY

# Solution 2: The project works without Hopsworks
# Features will be saved locally in data/processed/
```

### Issue 3: Port Already in Use

```bash
# Error: Address already in use (port 8000)

# Solution: Kill process using port
lsof -i :8000
kill -9 <PID>

# Or use different port
uvicorn src.api:app --port 8001
```

### Issue 4: Model Training Too Slow

```python
# Edit config/config.py
XGBOOST_PARAMS = {
    "n_estimators": 50,  # Reduce from 100
    "max_depth": 4,      # Reduce from 6
    ...
}
```

### Issue 5: Out of Memory

```bash
# Use smaller dataset for testing
python -c "
from src.data_loader import MovieLensDataLoader
loader = MovieLensDataLoader()
ratings, _, _ = loader.load_all_data()
# Use only first 10K samples
ratings = ratings.head(10000)
"
```

---

## 🎯 Next Steps

After completing setup:

1. **Explore the Code**
   - Read through src/ files
   - Understand the ML pipeline
   - Experiment with different models

2. **Customize the Project**
   - Add new features
   - Try different ML algorithms
   - Implement your own monitoring

3. **Deploy to Cloud** (Advanced)
   - AWS ECS/Fargate
   - Google Cloud Run
   - Azure Container Instances

4. **Add More Features**
   - User authentication
   - Real-time monitoring dashboard
   - A/B testing framework

---

## 📚 Additional Resources

- **Video Tutorials**: See project YouTube playlist
- **Documentation**: Check docs/ folder
- **Examples**: See notebooks/ folder
- **Support**: Open GitHub issue

---

## 🎉 Congratulations!

You've successfully set up a production-grade ML Engineering pipeline!

**What you built:**
- ✅ Complete MLOps system
- ✅ Automated ML pipeline
- ✅ Deployed API service
- ✅ Monitoring & testing
- ✅ CI/CD automation

**This demonstrates:**
- Data engineering skills
- ML model development
- Software engineering practices
- DevOps & MLOps knowledge
- Production deployment experience

---

For questions or issues, please:
1. Check this guide
2. Review README.md
3. Check GitHub Issues
4. Ask your instructor

**Happy coding! 🚀**
