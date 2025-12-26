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

# Copy dependency file first for better caching
COPY app-requirements.txt ./

# Install dependencies with pinned versions
RUN pip install --no-cache-dir -r app-requirements.txt

# Copy project files
COPY src ./src
COPY app ./app

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the Streamlit app
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
