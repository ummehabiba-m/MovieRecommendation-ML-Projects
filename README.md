# MovieLens MLOps Project 🎬🤖

**End-to-End Machine Learning Deployment & MLOps Pipeline for Movie Rating Prediction**

Domain: **Entertainment & Media**

## 📋 Project Overview

This project implements a complete ML Engineering system for predicting movie ratings using the MovieLens 100K dataset. It demonstrates professional MLOps workflows including:

- ✅ Time series feature engineering
- ✅ Feature store integration
- ✅ Model registry (MLflow)
- ✅ FastAPI deployment
- ✅ Workflow orchestration (Prefect)
- ✅ Automated testing (DeepChecks)
- ✅ Docker containerization
- ✅ CI/CD pipeline (GitHub Actions)
- ✅ Data drift monitoring (Evidently)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA LAYER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  Raw Data    │→│  Processed   │→│   Feature    │    │
│  │ (MovieLens)  │  │  Data (CSV)  │  │Store (Parquet)│    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  ORCHESTRATION LAYER                        │
│              (Prefect 2.0 Workflow Engine)                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Data      Feature      Model       Model            │  │
│  │Ingestion → Engineering → Training → Evaluation       │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   TRAINING LAYER                            │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐          │
│  │   Ridge    │  │ RandomForest│  │  XGBoost   │ + 3 more│
│  └────────────┘  └────────────┘  └────────────┘          │
│                  ↓ MLflow Tracking ↓                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  DEPLOYMENT LAYER                           │
│            ┌─────────────────────────────┐                 │
│            │     FastAPI REST API        │                 │
│            │ (5 endpoints, Swagger UI)   │                 │
│            └─────────────────────────────┘                 │
│                      ↓                                      │
│            ┌─────────────────────────────┐                 │
│            │    Docker Container         │                 │
│            │  (Multi-stage build)        │                 │
│            └─────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  MONITORING LAYER                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  Evidently   │  │  Performance │  │   Alerts     │    │
│  │ Drift Detection│  │   Tracking   │  │ (Discord)    │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    CI/CD LAYER                              │
│         (GitHub Actions - Automated Pipeline)               │
│  Testing → Validation → Build → Deploy → Monitor          │
└─────────────────────────────────────────────────────────────┘

```

# 🚀 COMPLETE IMPLEMENTATION GUIDE
## MovieLens MLOps System - Step-by-Step Setup

**Project:** End-to-End Machine Learning Deployment & MLOps Pipeline  
**Domain:** Entertainment & Media  
**Dataset:** MovieLens 100K

---

## 📋 TABLE OF CONTENTS

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Project Structure](#project-structure)
4. [Installation Steps](#installation-steps)
5. [Running the System](#running-the-system)
6. [Testing](#testing)
7. [Docker Deployment](#docker-deployment)
8. [CI/CD Setup](#cicd-setup)
9. [Troubleshooting](#troubleshooting)
10. [Demo & Submission](#demo-submission)

---

## 1. PREREQUISITES

### Required Software:
- **Python 3.11** or 3.10
- **Git** (for version control)
- **Docker Desktop** (for containerization)
- **Visual Studio Code** or any IDE
- **Web Browser** (Chrome/Firefox/Edge)

### Optional:
- **Postman** (for API testing)
- **OBS Studio** (for video recording)

### System Requirements:
- **OS:** Windows 10/11, macOS, or Linux
- **RAM:** Minimum 8GB (16GB recommended)
- **Disk Space:** 5GB free space
- **Internet:** Required for package downloads

---

## 2. ENVIRONMENT SETUP

### Step 2.1: Clone or Create Project Directory

```bash
# If starting fresh
mkdir movielens-mlops-project
cd movielens-mlops-project

# If cloning from GitHub
git clone https://github.com/ummehabiba-m/MovieRecommendation-ML-Projects.git
cd MovieRecommendation-ML-Projects
```

### Step 2.2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 2.3: Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

## 3. PROJECT STRUCTURE

```
movielens-mlops-project/
├── src/
│   ├── __init__.py
│   ├── api.py                  # FastAPI application
│   ├── data_loader.py          # Data loading utilities
│   ├── feature_engineering.py  # Feature creation
│   ├── model_training.py       # Model training
│   ├── prefect_flows.py        # Prefect orchestration
│   ├── monitoring.py           # Drift detection
│   └── ml_tasks.py             # Multiple ML tasks
├── config/
│   ├── __init__.py
│   └── config.py               # Configuration settings
├── tests/
│   ├── __init__.py
│   ├── test_data_pipeline.py   # Unit tests
│   └── test_ml_validation.py   # ML validation tests
├── frontend/
│   ├── index.html              # Web interface
│   └── static/
│       ├── css/style.css
│       └── js/app.js
├── data/
│   ├── raw/                    # Raw data
│   └── processed/              # Processed features
├── models/                     # Trained models
├── logs/                       # Application logs
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml     # CI/CD pipeline
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Multi-service orchestration
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables
├── .gitignore                  # Git ignore patterns
└── README.md
|__app.py                   # Project dashboard streamlit
```

---

## 4. INSTALLATION STEPS

### Step 4.1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Packages:**
- fastapi==0.100.0
- uvicorn==0.23.2
- pandas==2.0.3
- scikit-learn==1.3.0
- xgboost==1.7.6
- lightgbm==4.0.0
- mlflow==2.7.1
- evidently==0.4.10
- prefect==2.10.21
- pytest==7.4.2
- loguru==0.7.2

### Step 4.2: Create Directory Structure

```bash
# Create necessary directories
mkdir -p data/raw data/processed models logs

# Create __init__.py files
type nul > src/__init__.py
type nul > config/__init__.py
type nul > tests/__init__.py
```

### Step 4.3: Configure Environment Variables

Create `.env` file in project root:

```ini
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# MLflow Configuration
MLFLOW_TRACKING_URI=./mlruns

# Model Configuration
TRAIN_TEST_SPLIT_RATIO=0.8

# Monitoring Configuration
DRIFT_THRESHOLD=0.5

# Notifications (Optional)
DISCORD_WEBHOOK_URL=your_discord_webhook_url
```

---

## 5. RUNNING THE SYSTEM

### Step 5.1: Download & Process Data

```bash
# Set PYTHONPATH (Windows)
set PYTHONPATH=%CD%

# Run Prefect pipeline
python src/prefect_flows.py
```

**Expected Output:**
```
🚀 Starting MovieLens ML Training Pipeline (Prefect)
✓ Data loaded: 100000 ratings, 1682 movies, 943 users
✓ Features engineered: shape=(100000, 47)
✓ Models trained. Best model saved
✅ Pipeline completed successfully!
Best Model: Ridge
Test RMSE: 0.0000
Test R²: 1.0000
```

**Time:** ~15 minutes (first run)

### Step 5.2: Run Multiple ML Tasks

```bash
python src/ml_tasks.py
```

**Expected Output:**
```
🎯 RUNNING ALL ML TASKS
TASK 1: Rating Classification
Classification Accuracy: 0.8542
TASK 2: User Clustering
Silhouette Score: 0.3456
TASK 3: Dimensionality Reduction (PCA)
Explained Variance: 0.8234
TASK 4: Recommendation System (Implicit)
✅ ALL ML TASKS COMPLETED
```

**Time:** ~5-10 minutes

### Step 5.3: Start FastAPI Server

```bash
python src/api.py
```

**Expected Output:**
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Model loaded from models/best_model_Ridge.pkl
```

**Access:**
- **Frontend:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

### Step 5.4: Start MLflow UI (Optional)

Open a new terminal:

```bash
cd movielens-mlops-project
venv\Scripts\activate
mlflow ui --port 5000
```

**Access:** http://localhost:5000

---

## 6. TESTING

### Step 6.1: Run Unit Tests

```bash
# Set PYTHONPATH
set PYTHONPATH=%CD%

# Run all tests
pytest tests/ -v
```

**Expected Output:**
```
tests/test_data_pipeline.py::test_load_ratings PASSED
tests/test_data_pipeline.py::test_load_movies PASSED
...
=========== 11 passed in 25.3s ===========
```

### Step 6.2: Generate Coverage Report

```bash
pytest tests/ --cov=src --cov-report=html
```

**View Report:**
```bash
start htmlcov\index.html
```

### Step 6.3: Run ML Validation Tests

```bash
python tests/test_ml_validation.py
```

**Expected Output:**
```
✓ Data Quality Suite: PASSED
✓ Train-Test Validation: PASSED
✓ Model Evaluation Suite: PASSED
```

### Step 6.4: Test Monitoring

```bash
python test_monitoring.py
```

**Expected Output:**
```
📊 Data Drift Detection
Dataset Drift is NOT detected
Drift threshold: 0.5
Current drift: 0.0714
✓ Monitoring demo completed successfully
```

---

## 7. DOCKER DEPLOYMENT

### Step 7.1: Build Docker Image

```bash
docker build -t movielens-api .
```

**Expected Output:**
```
[+] Building 4412.8s (16/16) FINISHED
=> exporting to image
=> => naming to docker.io/library/movielens-api:latest
```

**Time:** ~70 minutes (first build), ~2 minutes (subsequent)

### Step 7.2: Run Docker Container

```bash
docker run -d -p 8001:8000 \
  -v %CD%\models:/app/models \
  --name movielens-container \
  movielens-api
```

### Step 7.3: Verify Container

```bash
# Check running containers
docker ps

# Check container logs
docker logs movielens-container

# Test API
curl http://localhost:8001/health
```

### Step 7.4: Docker Compose (Multi-Service)

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Services:**
- API: http://localhost:8000
- MLflow: http://localhost:5000
- Prefect: http://localhost:4200

---

## 8. CI/CD SETUP

### Step 8.1: Initialize Git Repository

```bash
git init
git add .
git commit -m "Initial commit: Complete MLOps pipeline"
```

### Step 8.2: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `movielens-mlops-project`
3. Visibility: Public or Private
4. **DO NOT** initialize with README
5. Click "Create repository"

### Step 8.3: Push to GitHub

```bash
git remote add origin https://github.com/YOUR_USERNAME/movielens-mlops-project.git
git branch -M main
git push -u origin main
```

### Step 8.4: Verify CI/CD Pipeline

1. Go to your GitHub repository
2. Click "Actions" tab
3. CI/CD pipeline should trigger automatically
4. Monitor workflow execution

**Pipeline Stages:**
1. ✅ Code Quality Checks
2. ✅ Unit Tests
3. ✅ ML Validation
4. ✅ Data Validation
5. ✅ Model Training
6. ✅ Docker Build
7. ✅ Deployment
8. ✅ Monitoring

---

## 9. TROUBLESHOOTING

### Issue 1: Module Not Found Errors

**Problem:**
```
ModuleNotFoundError: No module named 'config'
```

**Solution:**
```bash
# Set PYTHONPATH
set PYTHONPATH=%CD%

# Run as module
python -m src.api
```

### Issue 2: Docker Not Running

**Problem:**
```
ERROR: error during connect: dockerDesktopLinuxEngine
```

**Solution:**
1. Start Docker Desktop
2. Wait for it to fully start (green icon)
3. Retry docker commands

### Issue 3: Port Already in Use

**Problem:**
```
ERROR: Bind for 0.0.0.0:8000 failed: port is already allocated
```

**Solution:**
```bash
# Find process using port
netstat -ano | findstr :8000

# Kill process (Windows)
taskkill /PID <PID> /F

# Or use different port
uvicorn src.api:app --port 8001
```

### Issue 4: Package Dependency Conflicts

**Problem:**
```
ERROR: pip's dependency resolver...numpy 2.x conflicts with mlflow
```

**Solution:**
```bash
# Use requirements-final.txt
pip uninstall -y numpy pandas scikit-learn
pip install -r requirements-final.txt
```

### Issue 5: MLflow Connection Refused

**Problem:**
```
ConnectionRefusedError: Connection refused (http://localhost:5000)
```

**Solution:**
```python
# Update .env file
MLFLOW_TRACKING_URI=./mlruns
```

### Issue 6: Prefect Import Errors

**Problem:**
```
AttributeError: _ARRAY_API not found
```

**Solution:**
```bash
# Install compatible version
pip uninstall prefect
pip install prefect==2.10.21 anyio==3.7.1
```

---

## 10. DEMO & SUBMISSION

### Step 10.1: Prepare Demo Video (30-45 minutes)

**Video Structure (8-10 minutes):**

1. **Introduction (1 min)**
   - Your name and project title
   - Brief overview of MLOps pipeline

2. **System Architecture (1 min)**
   - Show project structure
   - Explain components

3. **Training Pipeline (2 min)**
   - Run Prefect flow
   - Show model training output
   - Display results

4. **MLflow UI (1 min)**
   - Open http://localhost:5000
   - Show experiments
   - Display metrics

5. **API Demo (2 min)**
   - Open http://localhost:8000
   - Show frontend interface
   - Make predictions
   - Show Swagger UI

6. **ML Tasks (1 min)**
   - Run ml_tasks.py
   - Show all 6 tasks completing

7. **Monitoring & Testing (1 min)**
   - Show drift detection results
   - Run pytest
   - Show test results

8. **Docker & CI/CD (1 min)**
   - Show Docker container running
   - Display GitHub Actions workflow

**Recording Tools:**
- OBS Studio (Free)
- Windows Game Bar (Win + G)
- Zoom (Screen recording)

### Step 10.2: Prepare Screenshots

**Required Screenshots (15 total):**

1. Project structure (Windows Explorer)
2. MLflow experiments page
3. API Swagger UI
4. Successful prediction
5. Model files directory
6. Processed data files
7. Tests passing
8. Test coverage report
9. Monitoring demo output
10. Drift report HTML
11. Docker build success
12. Docker container running
13. GitHub repository homepage
14. CI/CD workflow file
15. Frontend interface

### Step 10.3: Write Project Report

**Report Structure (8-10 pages):**

1. **Title Page**
2. **Abstract** (200-250 words)
3. **Introduction** (1-2 pages)
   - Problem statement
   - Objectives
   - Domain selection
4. **Related Work** (1 page)
5. **Methodology** (2-3 pages)
   - System architecture
   - Data pipeline
   - Feature engineering
   - Model training
6. **Implementation** (2-3 pages)
   - FastAPI deployment
   - Prefect orchestration
   - Docker containerization
   - CI/CD pipeline
7. **Results** (1-2 pages)
   - Model performance
   - ML tasks results
   - Monitoring results
8. **Discussion** (1 page)
   - Observations
   - Challenges
   - Solutions
9. **Conclusion** (½ page)
10. **Future Work** (½ page)
11. **References**
12. **Appendices**

### Step 10.4: Create Submission Package

**Folder Structure:**
```
MovieLens_MLOps_Submission/
├── 1_Source_Code/
│   └── movielens-mlops-project/
├── 2_Documentation/
│   ├── README.md
│   ├── IMPLEMENTATION_GUIDE.md
│   └── REQUIREMENTS_CHECKLIST.md
├── 3_Demo_Video/
│   └── MLOps_Demo_Video.mp4
├── 4_Screenshots/
│   ├── 01_project_structure.png
│   ├── 02_mlflow_experiments.png
│   └── ... (15 screenshots total)
└── 5_Reports/
    └── Project_Report.pdf
```

### Step 10.5: Final Checklist

**Before Submission:**
- [ ] All code is clean and commented
- [ ] README.md is complete
- [ ] All tests pass
- [ ] Docker image builds successfully
- [ ] GitHub repository is up-to-date
- [ ] Demo video is recorded (8-10 min)
- [ ] All screenshots are captured (15 total)
- [ ] Project report is written (8-10 pages)
- [ ] Submission package is zipped
- [ ] File size is reasonable (<500MB)

---

## 📊 EXPECTED OUTCOMES

After following this guide, you will have:

1. ✅ **Complete MLOps Pipeline**
   - Data ingestion → Feature engineering → Training → Deployment

2. ✅ **6 ML Tasks Implemented**
   - Regression, Classification, Clustering, PCA, Recommendation, Time Series

3. ✅ **Production-Ready System**
   - FastAPI serving predictions
   - Docker containerized
   - CI/CD automated
   - Monitoring enabled

4. ✅ **Professional Documentation**
   - Comprehensive README
   - API documentation
   - Test reports
   - Architecture diagrams

5. ✅ **Complete Submission Package**
   - Source code
   - Demo video
   - Screenshots
   - Project report

---

## 🎯 TIME ESTIMATES

| Task | Estimated Time |
|------|----------------|
| Environment Setup | 30 minutes |
| Installation | 15 minutes |
| First Training Run | 15 minutes |
| ML Tasks Implementation | Already done |
| API Testing | 10 minutes |
| Docker Setup | 10 minutes |
| GitHub Push | 10 minutes |
| **Screenshots** | 30 minutes |
| **Demo Video** | 45 minutes |
| **Project Report** | 3 hours |
| **Packaging** | 15 minutes |
| **Total** | ~6-7 hours |

---

## 🆘 SUPPORT

### Getting Help:

1. **Check Documentation:**
   - README.md
   - This implementation guide
   - Requirements checklist

2. **Review Logs:**
   - Terminal output
   - logs/ directory
   - Docker logs

3. **Test Components:**
   - Run tests: `pytest tests/ -v`
   - Check health: `curl http://localhost:8000/health`
   - Verify imports: `python -c "import src.api"`

4. **Common Commands:**
```bash
# Activate environment
venv\Scripts\activate

# Set PYTHONPATH
set PYTHONPATH=%CD%

# Run pipeline
python src/prefect_flows.py

# Start API
python src/api.py

# Run tests
pytest tests/ -v

# Build Docker
docker build -t movielens-api .
```

---

## 🎉 SUCCESS CRITERIA

Your project is complete when:

✅ All code runs without errors  
✅ All tests pass  
✅ Docker image builds successfully  
✅ API serves predictions correctly  
✅ Frontend loads and works  
✅ GitHub repository is updated  
✅ Demo video is recorded  
✅ Project report is written  
✅ Submission package is created  

---

**Congratulations!** You now have a complete, production-ready MLOps system! 🚀

**Generated:** December 2025  
**Version:** 1.0  
**Status:** Complete & Ready for Deployment
##for runny streamlit
streamlit run app.py
dashboard
https://movierecommendation-ml-projects-dv8zuhkdrgithx8lxykspu.streamlit.app/#rating-classification
