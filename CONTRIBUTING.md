# Contributing Guide

Welcome! This guide will help you set up the BookRecommender project locally and start contributing.

## Prerequisites

- **Python**: 3.10 or higher
- **Git**: For version control
- **uv**: Modern Python package manager (recommended) OR `pip`
- **Storage**: ~5GB for full dataset
- **RAM**: 8GB minimum, 16GB recommended for full dataset processing

---

## Quick Start (5 minutes)

### 1. Install uv (Recommended)

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and Setup

```bash
# Clone repository
git clone https://github.com/yourusername/BookRecommender.git
cd BookRecommender

# Install dependencies and create virtual environment
uv sync

# Activate virtual environment (optional - uv run handles this automatically)
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Download Data

```bash
# Download UCSD GoodReads dataset (~2GB compressed)
uv run python src/downloader.py
```

### 4. Prepare Data

**Quick test (small sample - ~30 seconds):**
```bash
uv run python prepare_data.py --min_reads=50 --top_books=5000 --top_users=100000
```

**Production dataset (~1-2 minutes):**
```bash
# Serialize data first (one-time, ~1 minute)
uv run python src/serialize_data.py --batch_size=500_000

# Prepare model data
uv run python prepare_data.py --min_reads=10 --top_books=50000 --top_users=1000000
```

### 5. Run the App

```bash
# Basic KNN recommender
uv run --group app streamlit run app/app.py

# Enhanced hybrid recommender (requires content features)
uv run --group app streamlit run app/app_enhanced.py
```

App runs at: http://localhost:8501

---

## Development Setup

### Option 1: Using uv (Recommended)

```bash
# Install all dependencies including dev tools
uv sync --all-groups

# Install Jupyter kernel for notebooks
uv run python -m ipykernel install --user --name=book-rec
```

### Option 2: Using pip

```bash
# Create virtual environment
python -m venv .venv

# Activate
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install project
pip install -e .

# Install app dependencies
pip install -r app-requirements.txt

# Install dev dependencies
pip install black ruff pytest coverage jupyter ipykernel
```

---

## Project Structure

```
BookRecommender/
├── app/                    # Streamlit web applications
│   ├── app.py             # Basic KNN recommender UI
│   └── app_enhanced.py    # Hybrid recommender UI
├── src/                    # Source code modules
│   ├── downloader.py      # Download UCSD dataset
│   ├── serialize_data.py  # Preprocess raw data
│   ├── extract_metadata.py          # Extract book metadata
│   ├── create_content_features.py   # Content-based features
│   ├── knn_recommender.py           # Basic KNN implementation
│   ├── knn_recommender_sparse.py    # Optimized sparse KNN
│   ├── hybrid_recommender.py        # Hybrid model
│   └── compare_recommenders.py      # Model comparison
├── data/                   # Data directory (gitignored)
│   ├── goodreads_books.json.gz
│   ├── goodreads_interactions.csv
│   ├── *.snap.parquet     # Processed snapshots
│   ├── *.npz              # Sparse matrices
│   └── *_mapping.parquet  # ID mappings
├── notebooks/              # Jupyter notebooks for exploration
├── prepare_data.py         # Main data preparation script
├── pyproject.toml          # Project dependencies
├── app-requirements.txt    # App-only dependencies (Docker)
├── MODEL.md               # Model documentation
└── CONTRIBUTING.md        # This file
```

---

## Data Pipeline

### Step 1: Download Raw Data
```bash
uv run python src/downloader.py
```
**Output:**
- `data/goodreads_books.json.gz` (~2GB)
- `data/goodreads_interactions.csv` (~3GB)

### Step 2: Serialize Data (One-time)
```bash
uv run python src/serialize_data.py --batch_size=500_000
```
**Purpose:** Convert JSON to parquet for 15x faster loading
**Runtime:** ~1 minute

### Step 3: Prepare Model Data
```bash
# Default: 50K books, 1M users, min 10 interactions
uv run python prepare_data.py
```
**Output:**
- `data/filtered_interactions.parquet` - User-book ratings
- `data/book_user_matrix_sparse.npz` - Sparse interaction matrix
- `data/sparse_matrix_book_mapping.parquet` - Book ID mapping
- `data/sparse_matrix_user_mapping.parquet` - User ID mapping
- `data/filtered_titles.parquet` - Book titles

### Step 4: Create Content Features (Optional)
```bash
# Extract metadata (genres, authors, ratings)
uv run python src/extract_metadata.py

# Create TF-IDF and numeric features
uv run python src/create_content_features.py
```
**Output:**
- `data/book_metadata.parquet` - Book metadata
- `data/content_features.npz` - Content feature matrix
- `data/content_features_mapping.parquet` - Book ID mapping

---

## Running Models

### KNN Recommender (Collaborative Filtering)
```bash
uv run python src/knn_recommender_sparse.py
```

### Hybrid Recommender (Collaborative + Content)
```bash
uv run python src/hybrid_recommender.py
```

### Compare All Methods
```bash
uv run python src/compare_recommenders.py
```

---

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run coverage run -m pytest
uv run coverage report
```

---

## Code Quality

### Format Code
```bash
# Auto-format with black
uv run black .

# Lint with ruff
uv run ruff check .
uv run ruff check --fix .
```

### Pre-commit Checks
```bash
# Format
uv run black src/ app/ prepare_data.py

# Lint
uv run ruff check src/ app/ prepare_data.py --fix

# Type hints (optional)
# pip install mypy
# mypy src/
```

---

## Docker Development

### Build and Run

The Dockerfile uses `uv` for dependency management, ensuring consistency with local development:

```bash
# Build image
docker build -t book-recommender .

# Run container
docker run -p 8501:8501 book-recommender

# Or use docker-compose (recommended)
docker-compose up --build
```

**What happens during Docker build:**
1. Installs `uv` from official image
2. Copies `pyproject.toml` and `uv.lock` for reproducible builds
3. Runs `uv sync --frozen --no-dev --group app` to install:
   - Core dependencies from `dependencies` in pyproject.toml
   - App group dependencies (streamlit)
   - Excludes dev dependencies (pytest, black, jupyter, etc.)

### Development with Docker
```bash
# Mount local code for live reload
docker run -p 8501:8501 \
  -v $(pwd)/app:/app/app \
  -v $(pwd)/data:/app/data \
  book-recommender
```

### Updating Dependencies in Docker

After updating `pyproject.toml`:
```bash
# Update lock file locally
uv lock

# Rebuild Docker image with new dependencies
docker-compose up --build
```

---

## Common Tasks

### Add New Dependencies

**Using uv:**
```bash
# Core dependency
uv add package-name

# App-only dependency
uv add --group app package-name

# Dev dependency
uv add --group dev package-name
```

**Using pip:**
```bash
# Update pyproject.toml manually, then:
pip install -e .
```

### Update app-requirements.txt
After adding dependencies, regenerate for Docker:
```bash
# List current versions
uv pip list --format=freeze > requirements-full.txt

# Manually extract app dependencies to app-requirements.txt
# Include only: pandas, polars, scipy, scikit-learn, implicit, loguru, streamlit, plotly
```

### Work with Notebooks
```bash
# Start Jupyter
uv run jupyter lab

# Or with notebook
uv run jupyter notebook

# Select "book-rec" kernel in Jupyter
```

### Download Different Dataset Sizes

**Small sample (testing):**
```bash
uv run python prepare_data.py --min_reads=50 --top_books=5000 --top_users=100000 --sample=0.1
```

**Medium sample:**
```bash
uv run python prepare_data.py --min_reads=20 --top_books=20000 --top_users=500000
```

**Large production:**
```bash
uv run python prepare_data.py --min_reads=5 --top_books=100000 --top_users=2000000
```

---

## Troubleshooting

### Import Errors
```bash
# Reinstall in editable mode
uv sync --reinstall

# Or with pip
pip install -e .
```

### Memory Issues During Data Prep
```bash
# Use smaller sample
uv run python prepare_data.py --sample=0.5

# Or reduce parameters
uv run python prepare_data.py --top_books=20000 --top_users=500000
```

### Streamlit Port Already in Use
```bash
# Use different port
uv run streamlit run app/app.py --server.port=8502
```

### Data Download Fails
If automatic download fails, manually download from [UCSD Book Graph](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home) and place in `data/` folder:
- `goodreads_books.json.gz`
- `goodreads_interactions.csv`
- `book_id_map.csv`
- `user_id_map.csv`

---

## Contributing Workflow

1. **Fork and clone** the repository
2. **Create a branch** for your feature: `git checkout -b feature/your-feature-name`
3. **Make changes** and add tests
4. **Format code**: `uv run black .` and `uv run ruff check --fix .`
5. **Run tests**: `uv run pytest`
6. **Commit changes**: Follow conventional commits (e.g., `feat:`, `fix:`, `docs:`)
7. **Push and create PR**

---

## Performance Benchmarks

**Data Preparation** (50K books, 1M users):
- Download: ~2-3 minutes (depends on connection)
- Serialize: ~1 minute
- prepare_data.py: ~1 minute
- extract_metadata.py: ~30 seconds
- create_content_features.py: ~45 seconds

**Recommendation Latency:**
- KNN (collaborative): 100-200ms
- Content-based: 50-100ms
- Hybrid: 150-250ms

**Memory Usage:**
- Development: ~2GB
- App runtime: ~1GB
- Docker container: ~1.5GB

---

## Resources

- **Model Documentation**: See [MODEL.md](MODEL.md) for detailed explanation of recommendation algorithms
- **Dataset**: [UCSD Book Graph](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home)
- **Collaborative Filtering**: [FastAI Tutorial](https://www.kaggle.com/code/jhoward/collaborative-filtering-deep-dive)
- **uv Documentation**: https://docs.astral.sh/uv/

---

## Getting Help

- Open an issue on GitHub for bugs or feature requests
- Check existing issues for similar problems
- Review [MODEL.md](MODEL.md) for algorithm explanations

---

## License

This project is for educational and portfolio purposes. Dataset from UCSD Book Graph.
