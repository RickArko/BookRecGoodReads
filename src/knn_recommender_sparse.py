"""Improved KNN recommender using sparse matrices for better performance.

This version:
- Uses scipy sparse matrices for memory efficiency
- Loads from .npz sparse matrix format
- Proper handling of missing titles
- Better error handling and logging
"""

import time
from pathlib import Path
import numpy as np
import polars as pl
from scipy.sparse import load_npz, csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz


def load_sparse_matrix(matrix_path="data/book_user_matrix_sparse.npz"):
    """Load sparse matrix from npz file.

    Returns:
        tuple: (sparse_matrix, book_ids, user_ids)
    """
    print(f"Loading sparse matrix from {matrix_path}...")
    start = time.time()

    # Load sparse matrix
    matrix = load_npz(matrix_path)

    # Load the book and user ID mappings created alongside the matrix
    matrix_dir = Path(matrix_path).parent
    book_mapping = pl.read_parquet(matrix_dir / "sparse_matrix_book_mapping.parquet")
    user_mapping = pl.read_parquet(matrix_dir / "sparse_matrix_user_mapping.parquet")

    # Extract ordered lists (already sorted by matrix_idx)
    book_ids = book_mapping.sort("matrix_idx")["book_id"].to_list()
    user_ids = user_mapping.sort("matrix_idx")["user_id"].to_list()

    elapsed = time.time() - start
    print(f"Loaded matrix in {elapsed:.2f}s")
    print(f"Shape: {matrix.shape} (books x users)")
    print(f"Non-zero entries: {matrix.nnz:,}")
    print(f"Sparsity: {100 * (1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])):.2f}%")

    # Verify mappings match matrix dimensions
    assert len(book_ids) == matrix.shape[0], f"Book mapping mismatch: {len(book_ids)} != {matrix.shape[0]}"
    assert len(user_ids) == matrix.shape[1], f"User mapping mismatch: {len(user_ids)} != {matrix.shape[1]}"

    return matrix, book_ids, user_ids


def create_title_mapping(book_ids):
    """Create mapping from book_id to title and from title to matrix index.

    Args:
        book_ids: List of book IDs (csv IDs) in matrix order

    Returns:
        tuple: (title_to_idx dict, idx_to_title dict, book_id_to_title dict)
    """
    print("Creating title mappings...")

    # Load the book ID mapping (csv IDs -> json IDs)
    # The UCSD dataset has two ID systems:
    # - book_id_csv: used in interactions data
    # - book_id: used in books JSON metadata
    id_map_df = pl.read_csv("data/book_id_map.csv")
    csv_to_json_id = dict(zip(id_map_df["book_id_csv"], id_map_df["book_id"]))

    # Load titles (uses json IDs, stored as String)
    titles_df = pl.read_parquet("data/titles.snap.parquet")
    titles_df = titles_df.with_columns(pl.col("book_id").cast(pl.Int64))
    json_id_to_title = dict(zip(titles_df["book_id"], titles_df["title"]))

    # Create mappings using matrix row order
    title_to_idx = {}
    idx_to_title = {}
    book_id_to_title = {}  # csv_id -> title
    missing_titles = 0
    missing_id_map = 0

    for idx, csv_id in enumerate(book_ids):
        # Map csv ID to json ID
        json_id = csv_to_json_id.get(csv_id)
        if json_id is None:
            title = f"Book ID: {csv_id}"
            missing_id_map += 1
            missing_titles += 1
        else:
            # Get title using json ID
            title = json_id_to_title.get(json_id)
            if title is None:
                title = f"Book ID: {csv_id}"
                missing_titles += 1
            else:
                book_id_to_title[csv_id] = title

        title_to_idx[title] = idx
        idx_to_title[idx] = title

    print(f"Mapped {len(book_ids) - missing_titles} titles out of {len(book_ids)} books")
    if missing_titles > 0:
        print(f"Warning: {missing_titles} books ({100*missing_titles/len(book_ids):.1f}%) have no title metadata")
        if missing_id_map > 0:
            print(f"  - {missing_id_map} books missing from book_id_map.csv")
        print(f"  - {missing_titles - missing_id_map} books missing from titles.snap.parquet")

    return title_to_idx, idx_to_title, book_id_to_title


def fuzzy_matching(title_to_idx, query_book, threshold=60, verbose=True):
    """Find book index using fuzzy string matching.

    Args:
        title_to_idx: Dictionary mapping titles to matrix indices
        query_book: Book title to search for
        threshold: Minimum fuzzy match score (0-100)
        verbose: Print matching results

    Returns:
        int or None: Matrix index of best match, or None if no match
    """
    matches = []

    for title, idx in title_to_idx.items():
        # Use partial_ratio for better partial string matching
        # This handles cases like "Harry Potter" matching "Harry Potter and the Sorcerer's Stone"
        ratio = fuzz.partial_ratio(title.lower(), query_book.lower())
        if ratio >= threshold:
            matches.append((title, idx, ratio))

    matches.sort(key=lambda x: x[2], reverse=True)

    if not matches:
        if verbose:
            print(f"No match found for '{query_book}' (threshold={threshold})")
        return None

    if verbose:
        print(f"Found {len(matches)} matches:")
        for title, idx, ratio in matches[:5]:  # Show top 5
            print(f"  {ratio}% - {title}")

    return matches[0][1]  # Return index of best match


class SparseKnnRecommender:
    """Item-based collaborative filtering recommender using sparse matrices."""

    def __init__(self, sparse_matrix, book_ids, title_to_idx, idx_to_title, metric="cosine", n_neighbors=20):
        """Initialize recommender.

        Args:
            sparse_matrix: scipy sparse matrix (books x users)
            book_ids: List of book IDs corresponding to matrix rows
            title_to_idx: Dict mapping titles to matrix indices
            idx_to_title: Dict mapping matrix indices to titles
            metric: Distance metric for KNN
            n_neighbors: Number of neighbors to find
        """
        self.matrix = sparse_matrix
        self.book_ids = book_ids
        self.title_to_idx = title_to_idx
        self.idx_to_title = idx_to_title
        self.model = NearestNeighbors(
            metric=metric,
            algorithm="brute",  # Required for sparse matrices with cosine
            n_neighbors=n_neighbors,
            n_jobs=-1,
        )

    def fit(self):
        """Fit the KNN model."""
        print(f"Fitting KNN model...")
        start = time.time()
        self.model.fit(self.matrix)
        elapsed = time.time() - start
        print(f"Model fitted in {elapsed:.2f}s")
        return self

    def recommend(self, book_name, n_recommendations=5, threshold=60):
        """Generate recommendations for a given book.

        Args:
            book_name: Title of the book to base recommendations on
            n_recommendations: Number of recommendations to return
            threshold: Fuzzy matching threshold

        Returns:
            list: List of (title, distance) tuples
        """
        # Find book index
        idx = fuzzy_matching(self.title_to_idx, book_name, threshold=threshold)
        if idx is None:
            return []

        # Get book vector
        book_vector = self.matrix[idx]

        # Find neighbors
        distances, indices = self.model.kneighbors(
            book_vector,
            n_neighbors=n_recommendations + 1,  # +1 to exclude the query book itself
        )

        # Format results (skip first one as it's the query book)
        recommendations = []
        for i in range(1, len(indices[0])):
            neighbor_idx = indices[0][i]
            distance = distances[0][i]
            title = self.idx_to_title[neighbor_idx]
            recommendations.append((title, distance))

        return recommendations

    def print_recommendations(self, book_name, n_recommendations=5, threshold=60):
        """Print recommendations in a readable format."""
        print(f"\n{'=' * 60}")
        print(f"Recommendations for: {book_name}")
        print(f"{'=' * 60}")

        recommendations = self.recommend(book_name, n_recommendations, threshold)

        if not recommendations:
            print("No recommendations found.")
            return

        for i, (title, distance) in enumerate(recommendations, 1):
            similarity = 1 - distance  # Convert distance to similarity
            print(f"{i}. {title}")
            print(f"   Distance: {distance:.4f} | Similarity: {similarity:.4f}")


def main():
    """Main execution function."""
    overall_start = time.time()

    print("\n" + "=" * 60)
    print("Book Recommendation System - Sparse KNN")
    print("=" * 60 + "\n")

    # Check if sparse matrix exists
    sparse_path = Path("data/book_user_matrix_sparse.npz")
    if not sparse_path.exists():
        print("ERROR: Sparse matrix not found!")
        print(f"Expected: {sparse_path}")
        print("\nPlease run: python prepare_data.py --compare_dense")
        return

    # Load data
    matrix, book_ids, user_ids = load_sparse_matrix()
    title_to_idx, idx_to_title, book_id_to_title = create_title_mapping(book_ids)

    # Initialize and fit recommender
    recommender = SparseKnnRecommender(matrix, book_ids, title_to_idx, idx_to_title, metric="cosine", n_neighbors=20)
    recommender.fit()

    # Test recommendations
    print("\n" + "=" * 60)
    print("Testing Recommendations")
    print("=" * 60)

    test_books = [
        "Harry Potter",
        "The Hunger Games",
        "The Da Vinci Code",
    ]

    for book in test_books:
        recommender.print_recommendations(book, n_recommendations=5, threshold=60)
        print()

    total_time = time.time() - overall_start
    print(f"\n{'=' * 60}")
    print(f"Total execution time: {total_time:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
