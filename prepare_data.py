"""Module to prepare data for modeling using Polars lazy dataframes.

This is a Polars-optimized version of prepare_model_data.py with improved performance
and memory efficiency through lazy evaluation and streaming operations.

Sources:
    - interactions
    - books string features
    - books numeric features

Outputs:
    - filtered_interactions.parquet - Filtered reader-book interactions
    - book_user_matrix.parquet - Pivot table of books x users x ratings
    - filtered_titles.parquet - Books that meet filter criteria
"""

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from scipy.sparse import csr_matrix, save_npz

# Default filter parameters
DEFAULT_MIN_READS = 50
DEFAULT_TOP_BOOKS = 1_000
DEFAULT_TOP_USERS = 50_000


def active_filter_optimized(
    interactions_lf: pl.LazyFrame,
    min_reads: int = DEFAULT_MIN_READS,
) -> pl.LazyFrame:
    """Filter interactions data by activity threshold with optimized lazy operations.

    Args:
        interactions_lf: LazyFrame with user-book interactions
        min_reads: Minimum number of reads to consider user/book active

    Returns:
        Filtered LazyFrame with only active users and popular books
    """
    logger.info(f"Filtering interactions with min_reads={min_reads}")

    # Get active users and popular books
    user_counts = interactions_lf.group_by("user_id").agg(pl.len().alias("count"))
    book_counts = interactions_lf.group_by("book_id").agg(pl.len().alias("count"))

    active_users = user_counts.filter(pl.col("count") >= min_reads).select("user_id")
    popular_books = book_counts.filter(pl.col("count") > -min_reads).select("book_id")

    # Combine filtering in one operation
    filtered = (
        interactions_lf
        .join(active_users, on="user_id", how="inner")
        .join(popular_books, on="book_id", how="inner")
    )

    return filtered


def get_top_x_users(
    interactions_lf: pl.LazyFrame,
    top_x: int = DEFAULT_TOP_USERS,
) -> pl.LazyFrame:
    """Get top X users by number of interactions.

    Args:
        interactions_lf: LazyFrame with user-book interactions
        top_x: Number of top users to return

    Returns:
        LazyFrame with top X user_ids
    """
    logger.info(f"Getting top {top_x:,} users")

    return (
        interactions_lf
        .group_by("user_id")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(top_x)
        .select("user_id")
    )


def get_top_x_books(
    interactions_lf: pl.LazyFrame,
    top_x: int = DEFAULT_TOP_BOOKS,
) -> pl.LazyFrame:
    """Get top X books by number of interactions.

    Args:
        interactions_lf: LazyFrame with user-book interactions
        top_x: Number of top books to return

    Returns:
        LazyFrame with top X book_ids
    """
    logger.info(f"Getting top {top_x:,} books")

    return (
        interactions_lf
        .group_by("book_id")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(top_x)
        .select("book_id")
    )


def create_book_user_matrix(
    interactions_lf: pl.LazyFrame,
    output_path: str = "data/book_user_matrix.parquet",
) -> pl.DataFrame:
    """Create book-user rating matrix (pivot table).

    Args:
        interactions_lf: LazyFrame with user-book interactions
        output_path: Path to save the matrix

    Returns:
        DataFrame with books as rows, users as columns, ratings as values
    """
    logger.info("Creating book-user matrix (pivot table)")
    start = time.time()

    # Collect to DataFrame for pivot operation
    df = interactions_lf.collect()

    # Create pivot table
    matrix = df.pivot(
        index="book_id",
        on="user_id",
        values="rating",
        aggregate_function="first",  # Take first rating if duplicates
    ).fill_null(0)

    # Save to parquet
    matrix.write_parquet(output_path)

    elapsed = time.time() - start
    logger.info(f"Matrix created in {elapsed:.2f}s with shape: {matrix.shape}")

    return matrix

def create_book_user_matrix_sparse(interactions_lf, output_path):
    """Create a sparse book-user rating matrix instead of dense pivot."""
    logger.info("Creating sparse book-user matrix")
    start = time.time()
    
    df = interactions_lf.select(["book_id", "user_id", "rating"]).collect()
    
    # Create mappings for indices
    book_ids = df["book_id"].unique().sort().to_list()
    user_ids = df["user_id"].unique().sort().to_list()
    
    book_to_idx = {bid: i for i, bid in enumerate(book_ids)}
    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    
    # Create sparse matrix using replace_strict for better performance
    rows = df.select(pl.col("book_id").replace_strict(book_to_idx))["book_id"].to_list()
    cols = df.select(pl.col("user_id").replace_strict(user_to_idx))["user_id"].to_list()
    data = df["rating"].to_list()
    
    matrix = csr_matrix((data, (rows, cols)),
                        shape=(len(book_ids), len(user_ids)))

    # Save to NPZ format
    save_npz(output_path, matrix)
    logger.info(f"Sparse matrix saved to {output_path}")
    logger.info(f"Sparse matrix created: {matrix.shape}, density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.4f}")

    # Save the ID mappings as parquet files for easy loading
    output_dir = Path(output_path).parent

    # Save book_ids (row indices)
    book_mapping = pl.DataFrame({
        "matrix_idx": list(range(len(book_ids))),
        "book_id": book_ids
    })
    book_mapping.write_parquet(output_dir / "sparse_matrix_book_mapping.parquet")
    logger.info(f"Saved book ID mapping: {len(book_ids)} books")

    # Save user_ids (column indices)
    user_mapping = pl.DataFrame({
        "matrix_idx": list(range(len(user_ids))),
        "user_id": user_ids
    })
    user_mapping.write_parquet(output_dir / "sparse_matrix_user_mapping.parquet")
    logger.info(f"Saved user ID mapping: {len(user_ids)} users")

    elapsed = time.time() - start
    logger.info(f"Sparse matrix created in {elapsed:.2f}s")
    return matrix


def main(
    min_reads: int = DEFAULT_MIN_READS,
    top_books: int = DEFAULT_TOP_BOOKS,
    top_users: int = DEFAULT_TOP_USERS,
    output_dir: str = "data",
    sample: float = 1.0,
    compare_dense: bool = False,
):
    """Prepare model data using Polars lazy dataframes.

    Args:
        min_reads: Minimum reads threshold for active filtering
        top_books: Number of top books to include
        top_users: Number of top users to include
        output_dir: Directory for output files
        sample: Sample fraction of data (0-1) for testing
        compare_dense: If True, also create dense matrix for comparison
    """
    overall_start = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("Starting model data preparation with Polars")
    logger.info(f"Parameters: min_reads={min_reads}, top_books={top_books:,}, top_users={top_users:,}")
    if sample < 1.0:
        logger.info(f"Using {sample*100:.1f}% sample of data for testing")
    logger.info("=" * 60)

    # Load data with lazy frames
    logger.info("Loading data files...")
    interactions_lf = pl.scan_parquet("data/goodreads_interactions.snap.parquet")
    titles_lf = pl.scan_parquet("data/titles.snap.parquet")
    
    # Apply sampling if specified
    if sample < 1.0:
        logger.info(f"Sampling {sample*100:.1f}% of interactions for testing...")
        interactions_lf = interactions_lf.sample(fraction=sample, seed=42)

    # Cast once at load time if needed
    titles_lf = titles_lf.with_columns(pl.col("book_id").cast(pl.Int64))

    # Apply active filter
    filtered_interactions = active_filter_optimized(interactions_lf, min_reads=min_reads)

    # Get top users and books
    top_user_ids = get_top_x_users(filtered_interactions, top_x=top_users)
    top_book_ids = get_top_x_books(filtered_interactions, top_x=top_books)

    # Filter to top users and books
    logger.info("Filtering to top users and books...")
    final_interactions = (
        filtered_interactions
        .join(top_user_ids, on="user_id", how="inner")
        .join(top_book_ids, on="book_id", how="inner")
    )

    # Save filtered interactions
    logger.info("Saving filtered interactions...")
    output_path = output_dir / "filtered_interactions.parquet"
    final_interactions.sink_parquet(str(output_path))
    logger.info(f"Saved to {output_path}")

    # Get filtered titles
    logger.info("Filtering titles...")
    # Join using the already-matched top_book_ids (which are i64 integers)
    filtered_titles = (
        titles_lf
        .join(top_book_ids, on="book_id", how="inner")
    )
    titles_output = output_dir / "filtered_titles.parquet"
    filtered_titles.sink_parquet(str(titles_output))
    logger.info(f"Saved to {titles_output}")

    # Create sparse matrix (default)
    logger.info("Creating sparse book-user rating matrix...")
    sparse_matrix_output = output_dir / "book_user_matrix_sparse.npz"
    sparse_matrix = create_book_user_matrix_sparse(final_interactions, str(sparse_matrix_output))
    
    # Optionally create dense matrix for comparison
    if compare_dense:
        logger.info("Creating dense book-user rating matrix for comparison...")
        matrix_output = output_dir / "book_user_matrix.parquet"
        matrix = create_book_user_matrix(final_interactions, str(matrix_output))
        
        # Compare matrix representations
        logger.info("=" * 60)
        logger.info("Matrix Comparison:")
        logger.info(f"  Dense matrix shape: {matrix.shape}")
        logger.info(f"  Dense matrix size: {matrix.nbytes / (1024**2):.2f} MB")
        logger.info(f"  Sparse matrix shape: {sparse_matrix.shape}")
        logger.info(f"  Sparse matrix nnz (non-zero): {sparse_matrix.nnz:,}")
        logger.info(f"  Sparse matrix density: {sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1]):.4f}")
        sparse_size = (sparse_matrix.data.nbytes + sparse_matrix.indices.nbytes + sparse_matrix.indptr.nbytes) / (1024**2)
        logger.info(f"  Sparse matrix memory: {sparse_size:.2f} MB")
        logger.info(f"  Memory savings: {(1 - sparse_size / (matrix.nbytes / (1024**2))) * 100:.1f}%")
    else:
        logger.info("Sparse matrix is primary format (use --compare_dense to also create dense matrix)")
    
    # Summary statistics
    logger.info("=" * 60)
    logger.info("Summary:")
    logger.info(f"  Sparse matrix shape: {sparse_matrix.shape}")
    logger.info(f"  Number of books: {sparse_matrix.shape[0]:,}")
    logger.info(f"  Number of users: {sparse_matrix.shape[1]:,}")
    logger.info(f"  Total interactions: {sparse_matrix.nnz:,}")
    logger.info(f"  Sparsity: {sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1]):.4f}")

    total_time = time.time() - overall_start
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    logger.info("=" * 60)

    return sparse_matrix if not compare_dense else matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare model data using Polars lazy dataframes"
    )
    parser.add_argument(
        "--min_reads",
        type=int,
        default=DEFAULT_MIN_READS,
        help=f"Minimum reads threshold (default: {DEFAULT_MIN_READS})",
    )
    parser.add_argument(
        "--top_books",
        type=int,
        default=DEFAULT_TOP_BOOKS,
        help=f"Number of top books to include (default: {DEFAULT_TOP_BOOKS:,})",
    )
    parser.add_argument(
        "--top_users",
        type=int,
        default=DEFAULT_TOP_USERS,
        help=f"Number of top users to include (default: {DEFAULT_TOP_USERS:,})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for processed files (default: data)",
        default="data",
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=1.0,
        help="Sample fraction of data (0-1) for testing (default: 1.0)",
    )
    parser.add_argument(
        "--compare_dense",
        action="store_true",
        help="Also create dense matrix for comparison (slower, more memory)",
    )

    args = parser.parse_args()

    main(
        min_reads=args.min_reads,
        top_books=args.top_books,
        top_users=args.top_users,
        output_dir=args.output_dir,
        sample=args.sample,
        compare_dense=args.compare_dense,
    )
