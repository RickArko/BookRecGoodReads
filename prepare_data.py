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

import polars as pl
from loguru import logger

# Default filter parameters
DEFAULT_MIN_READS = 50
DEFAULT_TOP_BOOKS = 1_000
DEFAULT_TOP_USERS = 50_000


def active_filter(
    interactions_lf: pl.LazyFrame,
    min_reads: int = DEFAULT_MIN_READS,
) -> pl.LazyFrame:
    """Filter interactions data by activity threshold.

    Args:
        interactions_lf: LazyFrame with user-book interactions
        min_reads: Minimum number of reads to consider user/book active

    Returns:
        Filtered LazyFrame with only active users and popular books
    """
    logger.info(f"Filtering interactions with min_reads={min_reads}")

    # Get active users (users with >= min_reads interactions)
    active_users = (
        interactions_lf
        .group_by("user_id")
        .agg(pl.len().alias("count"))
        .filter(pl.col("count") >= min_reads)
        .select("user_id")
    )

    # Get popular books (books with > -min_reads interactions)
    # Note: original used > -min_reads, keeping same logic
    popular_books = (
        interactions_lf
        .group_by("book_id")
        .agg(pl.len().alias("count"))
        .filter(pl.col("count") > -min_reads)
        .select("book_id")
    )

    # Filter to active users and popular books
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


def main(
    min_reads: int = DEFAULT_MIN_READS,
    top_books: int = DEFAULT_TOP_BOOKS,
    top_users: int = DEFAULT_TOP_USERS,
    output_dir: str = "data",
):
    """Prepare model data using Polars lazy dataframes.

    Args:
        min_reads: Minimum reads threshold for active filtering
        top_books: Number of top books to include
        top_users: Number of top users to include
        output_dir: Directory for output files
    """
    overall_start = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("Starting model data preparation with Polars")
    logger.info(f"Parameters: min_reads={min_reads}, top_books={top_books:,}, top_users={top_users:,}")
    logger.info("=" * 60)

    # Load data with lazy frames
    logger.info("Loading data files...")
    titles_lf = pl.scan_parquet("data/titles.snap.parquet")
    interactions_lf = pl.scan_parquet("data/goodreads_interactions.snap.parquet")
    books_lf = pl.scan_parquet("data/books_extra_features.snap.parquet")

    # Apply active filter
    filtered_interactions = active_filter(interactions_lf, min_reads=min_reads)

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
    # Cast book_id to string in top_book_ids to match titles schema
    top_book_ids_str = top_book_ids.with_columns(
        pl.col("book_id").cast(pl.Utf8)
    )
    filtered_titles = (
        titles_lf
        .join(top_book_ids_str, on="book_id", how="inner")
    )
    titles_output = output_dir / "filtered_titles.parquet"
    filtered_titles.sink_parquet(str(titles_output))
    logger.info(f"Saved to {titles_output}")

    # Create book-user matrix
    logger.info("Creating book-user rating matrix...")
    matrix_output = output_dir / "book_user_matrix.parquet"
    matrix = create_book_user_matrix(final_interactions, str(matrix_output))

    # Summary statistics
    logger.info("=" * 60)
    logger.info("Summary:")
    logger.info(f"  Book-user matrix shape: {matrix.shape}")
    logger.info(f"  Number of books: {matrix.shape[0]:,}")
    logger.info(f"  Number of users: {matrix.shape[1] - 1:,}")  # -1 for book_id column

    total_time = time.time() - overall_start
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    logger.info("=" * 60)

    return matrix


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
        default="data",
        help="Output directory for processed files (default: data)",
    )

    args = parser.parse_args()

    main(
        min_reads=args.min_reads,
        top_books=args.top_books,
        top_users=args.top_users,
        output_dir=args.output_dir,
    )
