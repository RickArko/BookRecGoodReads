# ** ~ AI-Condensed Context File ~ TOKEN-CONDENSED! DO NOT ALTER WITHOUT `/@ai-condense`**
---

> Condensed snapshot of `BookRecGoodReads` for AI consumption. Skip in human-readable doc tooling.

## Project
- **Name**: BookRecommender — book recommendations using UCSD Goodreads data.
- **Python**: >=3.10. Package manager: `uv` (`pyproject.toml` + `uv.lock`).
- **Package layout**: `src/` is a hatchling-built package. App in `app/`, scripts in `scripts/`, notebooks in `notebooks/`.
- **Core deps**: polars~=1.36, pandas~=2.3, scipy~=1.15, scikit-learn~=1.7, fuzzywuzzy[speedup], implicit~=0.7 (ALS), pyarrow, fastparquet, gdown, loguru, tqdm, plotly, matplotlib, seaborn.
- **Dep groups**: `app` (streamlit), `dev` (pytest, ruff, black, jupyter, ipykernel, coverage).
- **Lint/format**: black & ruff, line-length=120.

## Architecture (3 recommender approaches)
1. **Collaborative KNN** — sparse books×users matrix, cosine KNN. `src/knn_recommender_sparse.py`.
2. **Content-based** — TF-IDF over shelves + author MultiLabel + numeric features. `src/create_content_features.py`.
3. **Hybrid** — weighted combo (default 0.6 collab + 0.4 content). `src/hybrid_recommender.py`.
4. Optional **ALS matrix factorization** via `implicit` lib. `src/train_spotlight.py`.

## Data Pipeline (sequential)
```
downloader.py → serialize_data.py → prepare_data.py → extract_metadata.py → create_content_features.py
```

### Inputs (from UCSD McAuley Lab `https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/`)
- `goodreads_books.json.gz` — book metadata (gzipped ndjson)
- `goodreads_interactions.csv` — user-book interactions (228M)
- `book_id_map.csv` — maps `book_id_csv` ↔ `book_id` (json id) — TWO ID systems
- `user_id_map.csv`

### Generated files (in `data/`)
- `goodreads_interactions.snap.parquet` — interactions in parquet
- `titles.snap.parquet` — book_id (Int64), title, title_without_series
- `books_simple_features.snap.parquet` — numeric (publication_year, num_pages, ratings_count, average_rating, is_ebook…)
- `books_extra_features.snap.parquet` — description, format, language, authors (nested), country
- `filtered_interactions.parquet`, `filtered_titles.parquet` — after min_reads/top_books/top_users filters
- `book_user_matrix_sparse.npz` — scipy CSR (books×users), ~130MB
- `sparse_matrix_book_mapping.parquet` (matrix_idx → book_id), `sparse_matrix_user_mapping.parquet`
- `book_metadata.parquet` — extracted authors, shelves, ratings, year, pages, language
- `content_features.npz`, `content_features_mapping.parquet` — TF-IDF + author MLB + numeric stack

## Module Reference

### `src/downloader.py`
```python
DATA_DIR = "data"
DOWNLOAD_FILES = ["goodreads_books.json.gz","goodreads_interactions.csv","book_id_map.csv","user_id_map.csv"]
def download_file(filename, output_dir=DATA_DIR): ...  # streams via requests + tqdm
def download_goodreads_data(data_dir: str): ...        # skips existing files
```
Source URL pattern: `https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/{filename}`.

### `src/serialize_data.py` (Polars lazy streaming)
```python
def save_book_features(json_path, output_csv, output_dir="data"):
    """Extract extra features → books_extra_features.snap.parquet + titles.snap.parquet"""
    extra_feature_cols = ["book_id","description","format","title","title_without_series",
                          "language_code","authors","country_code"]
    pl.scan_ndjson(json_path).select(extra_feature_cols).sink_parquet(...)
def save_simple_book_features(json_path, output_file, output_dir="data"):
    """Numeric features w/ casts; is_ebook → 0/1"""
    keep_cols = ["book_id","work_id","publication_year","is_ebook","num_pages",
                 "ratings_count","text_reviews_count","average_rating"]
def save_interactions(input_path, output_dir="data"):
    pl.scan_csv(input_path).sink_parquet(...)
def main(json_path, csv_path, interactions_path, output_path, output_dir="data"): ...
# CLI: --batch_size (default 10_000_000), --output_dir (default "data")
# Auto-downloads if PATH_JSON missing.
```

### `prepare_data.py` (root, polars lazy)
```python
DEFAULT_MIN_READS = 10
DEFAULT_TOP_BOOKS = 50_000
DEFAULT_TOP_USERS = 1_000_000
def active_filter_optimized(interactions_lf, min_reads=10) -> pl.LazyFrame: ...
def get_top_x_users(lf, top_x=DEFAULT_TOP_USERS) -> pl.LazyFrame: ...
def get_top_x_books(lf, top_x=DEFAULT_TOP_BOOKS) -> pl.LazyFrame: ...
def create_book_user_matrix(lf, output_path) -> pl.DataFrame:
    """dense pivot books×users → parquet (slow/memory hungry)"""
def create_book_user_matrix_sparse(lf, output_path):
    """CSR matrix + book/user mapping parquets. PRIMARY format."""
def main(min_reads, top_books, top_users, output_dir="data", sample=1.0, compare_dense=False): ...
# CLI: --min_reads --top_books --top_users --output_dir --sample --compare_dense
```

### `src/extract_metadata.py`
```python
def extract_metadata(json_path="data/goodreads_books.json.gz",
                     matrix_mapping_path="data/sparse_matrix_book_mapping.parquet",
                     output_path="data/book_metadata.parquet",
                     top_n_shelves=10):
    """Streams gzipped json line-by-line; only keeps books in matrix.
    Per book: title, authors (CSV), author_ids, shelves (top N by count),
    shelf_counts, average_rating, ratings_count, publication_year, num_pages, language_code"""
```

### `src/create_content_features.py`
```python
def create_content_features(metadata_path="data/book_metadata.parquet",
                            matrix_mapping_path="data/sparse_matrix_book_mapping.parquet",
                            output_path="data/content_features.npz",
                            output_mapping_path="data/content_features_mapping.parquet"):
    """Pipeline:
    1. Shelf TF-IDF: max_features=500, min_df=3, token_pattern=r'[a-z][a-z0-9\\-]+'
    2. Author MultiLabelBinarizer (filter authors w/ <2 books)
    3. Numeric: log1p(ratings_count), avg_rating/5, (year-1900)/100, log1p(pages)/10
    4. L2-normalize each block, hstack → CSR"""
```

### `src/knn_recommender_sparse.py`
```python
def load_sparse_matrix(matrix_path="data/book_user_matrix_sparse.npz") -> (csr, book_ids, user_ids): ...
def create_title_mapping(book_ids) -> (title_to_idx, idx_to_title, book_id_to_title):
    """KEY: TWO ID SYSTEMS. csv_to_json_id from book_id_map.csv;
    json_id_to_title from titles.snap.parquet (cast book_id Int64).
    Missing → 'Book ID: {csv_id}'"""
def fuzzy_matching(title_to_idx, query_book, threshold=60, verbose=True) -> int|None:
    """Uses fuzz.partial_ratio for substring matches"""
class SparseKnnRecommender:
    def __init__(self, sparse_matrix, book_ids, title_to_idx, idx_to_title,
                 metric="cosine", n_neighbors=20):
        self.model = NearestNeighbors(metric=metric, algorithm="brute", n_neighbors=n_neighbors, n_jobs=-1)
    def fit(self): ...
    def recommend(self, book_name, n_recommendations=5, threshold=60) -> list[(title, dist)]: ...
    def print_recommendations(self, book_name, n_recommendations=5, threshold=60): ...
```

### `src/hybrid_recommender.py`
```python
class HybridRecommender:
    def __init__(self,
                 interaction_matrix_path="data/book_user_matrix_sparse.npz",
                 content_features_path="data/content_features.npz",
                 book_mapping_path="data/sparse_matrix_book_mapping.parquet",
                 content_mapping_path="data/content_features_mapping.parquet",
                 metadata_path="data/book_metadata.parquet",
                 collaborative_weight=0.6, content_weight=0.4, n_neighbors=20):
        """Maintains bidirectional collab_idx <-> content_idx mappings (book sets may differ)."""
    def fuzzy_search(self, query, threshold=60, max_results=5) -> list[(title,idx,score)]:
        """fuzz.ratio (NOT partial_ratio)"""
    def recommend_collaborative(self, book_idx, n_recommendations=5) -> list[(idx,dist)]: ...
    def recommend_content(self, book_idx, n_recommendations=5) -> list[(collab_idx,dist)]:
        """Returns [] if book has no content features"""
    def recommend_hybrid(self, book_idx, n_recommendations=10) -> list[(idx, combined, collab, content)]:
        """Get N*3 candidates from each method, score = w_c*collab_sim + w_t*content_sim"""
    def print_recommendations(self, query, n_recommendations=5, method="hybrid",
                              threshold=60, show_details=True): ...
# methods: "hybrid"|"collaborative"|"content"
```

### `src/compare_recommenders.py`
```python
def compare_recommendations(query, n_recommendations=5):
    """Side-by-side print of collab-only vs hybrid for a single query."""
# NOTE: References `recommender.book_ids` which doesn't exist on HybridRecommender
# (it has `collab_book_ids`). Likely broken — fix before relying on it.
```

### `src/train_spotlight.py` (ALS via `implicit`)
```python
def load_prepared_data(data_dir="data") -> csr:
    """Loads book_user_matrix_sparse.npz; raises FileNotFoundError if missing."""
def train_model(interactions, n_factors=32, n_iterations=10, regularization=0.01, alpha=40.0):
    """AlternatingLeastSquares.fit(interactions.T.tocsr())  -- transposes books×users → users×books"""
def main(data_dir="data", n_factors=32, n_iterations=10):
    """Saves to data/als_model.pkl"""
# CLI: --data_dir --n_factors --n_iterations
```

### `src/knn_recommender.py` (legacy dense matrix version)
- Uses `data/book_user_matrix.parquet` (dense) + `data/filtered_titles.parquet`.
- `class KnnRecommender` with `_train_model()` + `make_recommendation_from_book(book_name, n_recommendations)`.
- Prefer `knn_recommender_sparse.py` for production; this exists for comparison.

### `src/fuzzy.py`
```python
def fuzzy_matching(mapper, fav_movie, verbose=True):
    """fuzz.ratio threshold>=60. Note: variable name 'fav_movie' (legacy)."""
```

### `src/cf.py`
- Standalone script using `implicit.CosineRecommender` to write tab-separated similarity pairs to `data/cosine-similarity.txt`.

### `src/prepare.py`
- Older pandas-based pivot creation. Hardcoded `TOP_USERS=20_000`; uses `pivot(...).fillna(0)`. **Superseded by `prepare_data.py`**.

### `src/prepare_model_data.py`
- Legacy/exploratory pandas script. **Has broken references** (`dft`, `top_user_ids`, `csr_matrix`/`NearestNeighbors`/`fuzzy_matching` not imported, `dfq`, etc.). Do not run as-is; reference only.

### `src/__init__.py`
```python
from .downloader import download_goodreads_data
```

## Streamlit Apps (`app/`)

### `app/app.py` — Basic KNN UI
```python
@st.cache_resource
def load_recommender():
    matrix, book_ids, user_ids = load_sparse_matrix("data/book_user_matrix_sparse.npz")
    title_to_idx, idx_to_title, book_id_to_title = create_title_mapping(book_ids)
    recommender = SparseKnnRecommender(matrix, book_ids, title_to_idx, idx_to_title,
                                       metric="cosine", n_neighbors=20)
    recommender.fit()
@st.cache_data
def load_book_metadata():
    """titles + books_simple_features parquets joined on book_id"""
def format_book_display(title, book_info): ...  # builds "📅 year | ⭐ rating | 👥 ratings | 📖 pages"
def main():
    """Sidebar: n_recommendations slider, fuzzy_threshold slider, optional rating/year filters.
    Main: selectbox or text input → button → cached recommender.recommend()."""
# CLI: streamlit run app/app.py  (port 8501 default)
```

### `app/app_enhanced.py` — Hybrid UI ⭐ recommended
```python
@st.cache_resource
def load_hybrid_recommender(collab_weight=0.6, content_weight=0.4) -> HybridRecommender: ...
@st.cache_data
def load_book_metadata_extended() -> pl.DataFrame:  # data/book_metadata.parquet
def smart_search(query, available_titles, threshold=70, max_results=10):
    """fuzzywuzzy.process.extract over titles list"""
def calculate_diversity_score(recommendations, metadata_dict, recommender):
    """Avg pairwise (1 - jaccard(genres_top5)) across recs"""
# Tabs: search/recs, score-breakdown plotly charts, genre distribution, metrics dashboard
```

## Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
RUN apt-get install build-essential curl
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY app ./app
RUN uv sync --frozen --no-dev --group app
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
CMD ["uv","run","--group","app","streamlit","run","app/app_enhanced.py",
     "--server.port=8501","--server.address=0.0.0.0"]
```
`docker-compose.yml`: builds image, exposes 8501, **mounts `./data:/app/data:ro`** (data not baked into image).

## Scripts (`scripts/`)
```bash
scripts/setup.sh         # install + prepare_data (full setup)
scripts/install/install.sh  # uv sync --all-groups, install ipykernel "BookRecommender (uv)"
scripts/prepare_data.sh  # downloader → serialize_data → prepare_data
scripts/run_app.sh       # source .venv; uv run --group app streamlit run app/app.py
scripts/docker.sh        # docker-compose down && up --build
```

## Happy Path (uv, Linux/macOS)
```bash
# 0. Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. Install Python deps
uv sync --all-groups

# 2. Download UCSD goodreads data (~few GB)
uv run python src/downloader.py

# 3. JSON → parquet (polars streaming, ~minutes)
uv run python src/serialize_data.py --batch_size=500_000

# 4. Build interaction sparse matrix + mappings
uv run python prepare_data.py
# Smaller test sample:
# uv run python prepare_data.py --min_reads=50 --top_books=5000 --top_users=100000

# 5. Content-based metadata + features (needed for hybrid app)
uv run python src/extract_metadata.py
uv run python src/create_content_features.py

# 6. Run app
uv run --group app streamlit run app/app.py            # basic
uv run --group app streamlit run app/app_enhanced.py   # hybrid (recommended)
# OR docker-compose up --build
```

## Key Gotchas (preserve!)
1. **Two book_id systems**: `book_id_csv` (interactions) ↔ `book_id` (json metadata). Bridge via `book_id_map.csv`. `create_title_mapping` handles this; missing maps fall back to `"Book ID: {csv_id}"`.
2. `titles_df` book_id is **String**, must `.cast(pl.Int64)` before joining to interactions.
3. `prepare_data.py` `active_filter_optimized` uses `count > -min_reads` for popular_books (intentional? always true; effectively no popular-book filter — only user filter).
4. ALS in `train_spotlight.py` requires **transposed** matrix (users×books, not books×users).
5. Hybrid recommender has `collab_book_ids` (not `book_ids`). `compare_recommenders.py` references `recommender.book_ids` — **broken**.
6. `prepare_model_data.py` is a legacy scratch file with undefined symbols — do not execute.
7. Sparse matrix is books×users — `.kneighbors(matrix[idx])` finds similar **books**, not users.
8. `fuzz.partial_ratio` (sparse KNN) vs `fuzz.ratio` (hybrid) — different matching behaviors.
9. Docker mounts `./data:ro`; you must prepare data on host first.
10. `extra_feature_cols` includes nested `authors` field; CSV serialization may fail (caught + warned).

## Test/CI
- No pytest tests in tree. `pytest>=9` is in dev group but unused.
- `coverage>=7.13.0` declared, no config.

## Notebooks
- `Pipeline.ipynb` (root) — pipeline walk-through.
- `notebooks/distributions.ipynb` — data EDA.
- `notebooks/training_evaluation.ipynb` — model evaluation/visualizations.

## Git Recent
```
371b2cd hybrid app, setup scripts, docker app
e931b97 add requirements.txt for Dockerfile
87473ca add content based rec app
577a0d6 format ruff
f1ef42e content based rec
```
