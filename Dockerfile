# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install core packages
RUN pip install --upgrade pip setuptools wheel

# Copy project files first
COPY pyproject.toml ./
COPY src ./src
COPY app ./app
COPY knn_recommender_sparse.py ./

# Install dependencies from pyproject.toml
# Install the package in editable mode
RUN pip install --no-cache-dir \
    pandas \
    polars \
    scipy \
    scikit-learn \
    implicit \
    loguru \
    streamlit \
    plotly

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the Streamlit app
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
