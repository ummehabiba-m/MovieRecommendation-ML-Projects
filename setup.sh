#!/bin/bash

# MovieLens MLOps Project Setup Script
# This script automates the initial setup of the project

set -e  # Exit on error

echo "=========================================="
echo "MovieLens MLOps Project Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"

if [[ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]]; then
    print_error "Python $required_version or higher is required. Found: $python_version"
    exit 1
fi
print_status "Python $python_version detected"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    print_warning "Virtual environment already exists. Skipping..."
else
    python3 -m venv venv
    print_status "Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
print_status "Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
print_status "Pip upgraded"

# Install dependencies
echo ""
echo "Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt > /dev/null 2>&1
print_status "Dependencies installed"

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p data/raw data/processed models logs mlruns mlartifacts
print_status "Directories created"

# Copy environment file
echo ""
if [ ! -f ".env" ]; then
    echo "Setting up environment variables..."
    cp .env.example .env
    print_status "Environment file created (.env)"
    print_warning "Please edit .env file with your Hopsworks credentials"
else
    print_warning ".env file already exists. Skipping..."
fi

# Initialize Git (if not already initialized)
echo ""
if [ ! -d ".git" ]; then
    echo "Initializing Git repository..."
    git init
    print_status "Git repository initialized"
else
    print_warning "Git repository already initialized"
fi

# Create .gitignore
echo ""
echo "Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
.venv

# Data
data/raw/*
data/processed/*
*.csv
*.parquet
*.pkl

# Models
models/*.pkl
models/*.h5
models/*.joblib

# Logs
logs/
*.log

# MLflow
mlruns/
mlartifacts/

# Environment
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Coverage
.coverage
htmlcov/
coverage.xml

# Pytest
.pytest_cache/

# Docker
*.tar
EOF
print_status ".gitignore created"

# Download data
echo ""
echo "Do you want to download the MovieLens dataset now? (y/n)"
read -r download_choice

if [ "$download_choice" = "y" ] || [ "$download_choice" = "Y" ]; then
    echo "Downloading MovieLens 100K dataset..."
    python3 << EOF
from src.data_loader import MovieLensDataLoader
loader = MovieLensDataLoader()
ratings, movies, users = loader.load_all_data()
print(f"✓ Downloaded {len(ratings)} ratings, {len(movies)} movies, {len(users)} users")
EOF
    print_status "Dataset downloaded"
else
    print_warning "Skipping dataset download"
fi

# Setup complete
echo ""
echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file with your Hopsworks credentials:"
echo "   nano .env"
echo ""
echo "2. Activate virtual environment (if not already active):"
echo "   source venv/bin/activate"
echo ""
echo "3. Run the training pipeline:"
echo "   python src/prefect_flows.py"
echo ""
echo "4. Start the API server:"
echo "   python src/api.py"
echo ""
echo "5. Or use Docker:"
echo "   docker-compose up -d"
echo ""
echo "For more information, see README.md"
echo ""
