# BookRecommender
Repository for Book Recommendations using UCSD Goodreads Data.

### Data
GoodReads data from [UCSD Book Graph](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home).

**Option 1: Automatic Download**
```bash
uv run python src/downloader.py
```

**Option 2: Manual Download** (if automatic fails)
Download these files manually and save to `data/` folder:
- [goodreads_books.json.gz](https://drive.google.com/uc?id=1LXpK1UfqtP89H1tYy0pBGHjYk8IhigUK)
- [goodreads_interactions.csv](https://drive.google.com/uc?id=1zmylV7XW2dfQVCLeg1LbllfQtHD2KUon)
- [book_id_map.csv](https://drive.google.com/uc?id=1CHTAaNwyzvbi1TR08MJrJ03BxA266Yxr)
- [user_id_map.csv](https://drive.google.com/uc?id=15ax-h0Oi_Oyee8gY_aAQN6odoijmiz6Q)
### Installation
```bash
# Install uv if not already installed
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and create virtual environment
uv sync

# Activate virtual environment
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# Install Jupyter kernel
uv run python -m ipykernel install --user --name=book-rec
```

### Data Preparation
Run the commands below to download, and process all of the features data from scratch for the first time **__this will only be required once for a new developer setting up the project __ (expected runtime ~1 minute)**.

```bash
# This will download the data from Google Drive and create parquet files
uv run python src/downloader.py 

# serialize compressed json into row-interations features (~15x speedup)
uv run src\serialize_data.py --batch_size=500_000


### Prepare Model Date (now using polars ~10x speed-up)

# Small Sample
uv run python prepare_data.py --min_reads=50 --top_books=5000 --top_users=100000

# Default Sample (~1m)
uv run python prepare_data.py

# Generate Large Interaction Sample
uv run python prepare_data.py --min_reads=10 --top_books=50000 --top_users=1000000

# Generate Optimized KNN Rec via Sparse Matrix
uv run python knn_recommender_sparse.py

```

This creates:
- `data/titles.snap.parquet`
- `data/books_simple_features.snap.parquet`
- `data/books_extra_features.snap.parquet`
- `data/goodreads_interactions.snap.parquet`

### Run Model Scripts
```bash
# # Prepare model data
# uv run python prepare_model_data.py

# Run KNN recommender
uv run python knn_recommender.py

# Train Spotlight model
uv run python train_spotlight.py
```

# Create Content-Based Features
Extract metadata, content based feautures, tf-ids, ratings, etc.

```bash
  uv run python extract_metadata.py
  uv run python create_content_features.py
  uv run python compare_recommenders.py
```

### Compare Recomendations
Compare single recommendation across available methods

```bash
uv run python knn_recommender.py
uv run python knn_recommender_sparse.py
uv run python hybrid_recommender.py
```

### Models
1. Collaborative Filtering
2. Content Based

---

## 📱 Streamlit Web App

### Running the App Locally

**Option 1: Using uv (Recommended)**
```bash
# Install app dependencies
uv sync --group app

# Run the app
uv run --group app streamlit run app/app.py
```

The app will open in your browser at `http://localhost:8501`

**Option 2: Using Docker**
```bash
# Build and run with docker-compose
docker-compose up --build

# Or build and run manually
docker build -t book-recommender .
docker run -p 8501:8501 book-recommender
```

Access the app at `http://localhost:8501`

### App Features

- 🔍 **Smart Search**: Fuzzy matching to find books even with partial titles
- 📊 **Filters**: Filter recommendations by rating, publication year, and more
- 📈 **Similarity Scores**: See how closely each recommendation matches your selection
- ⚙️ **Customizable**: Adjust the number of recommendations and match sensitivity

### Prerequisites for the App

Make sure you've prepared the data first:
```bash
# Download data
uv run python src/downloader.py

# Serialize data
uv run python src/serialize_data.py --batch_size=500_000

# Prepare model data
uv run python prepare_data.py

# Generate KNN model
uv run python knn_recommender_sparse.py
```

---

### Resources
  - [collaborative-filtering-deep-dive](https://www.kaggle.com/code/jhoward/collaborative-filtering-deep-dive)