# Use Python 3.10 slim-buster image for more preinstalled libraries
FROM python:3.10-bullseye

# Set working directory
WORKDIR /app

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies needed for heavy Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    git \
    wget \
    python3-dev \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements-final.txt .

# Install Python dependencies (prefer binaries to avoid compiling)
RUN pip install --no-cache-dir --prefer-binary -r requirements-final.txt

# Copy project files
COPY config/ ./config/
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# Create necessary directories
RUN mkdir -p logs data/raw data/processed models

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
# Ensure data folders exist
RUN mkdir -p ./data ./data/raw ./data/processed

# Copy placeholder file
COPY data/.gitkeep ./data/

