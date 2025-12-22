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

# Prepare model data
uv run python prepare_data.py --min_reads=100 --top_books=5000 --top_users=100000
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

### Models
1. Collaborative Filtering
2. Content Based


### Resources
  - [collaborative-filtering-deep-dive](https://www.kaggle.com/code/jhoward/collaborative-filtering-deep-dive)