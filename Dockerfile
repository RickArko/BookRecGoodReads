# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (including curl for healthcheck)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock README.md ./

# Copy source code (needed for installation)
COPY src ./src
COPY app ./app

# Note: data/ directory (including book_id_map.csv) is mounted via docker-compose volume

# Install dependencies using uv with app group
# --frozen ensures we use exact versions from uv.lock
# --no-dev excludes dev dependencies
RUN uv sync --frozen --no-dev --group app

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the Streamlit app using uv with hybrid recommender
CMD ["uv", "run", "--group", "app", "streamlit", "run", "app/app_enhanced.py", "--server.port=8501", "--server.address=0.0.0.0"]
