import argparse
import time
from pathlib import Path

import polars as pl
from loguru import logger
from src.downloader import download_goodreads_data

README = """Module to process downloaded goodreads json files and save in parquet format.

    Uses Polars lazy dataframes for efficient streaming processing of large JSON files.
    This approach is significantly faster than pandas and uses less memory through
    lazy evaluation and streaming writes.

    General Outline:
    ----------------
        1. Scan JSON with lazy reader (pl.scan_ndjson)
        2. Apply transformations using lazy operations
        3. Stream results to parquet files with sink_parquet
        4. No chunking needed - Polars handles large files efficiently

    Outputs:
    -------
        goodreads_interactions.snap.parquet
        titles.snap.parquet
        books_simple_features.snap.parquet
        books_extra_features.snap.parquet
"""


def save_book_features(json_path, output_csv, output_dir: str = "data"):
    """Extract book features and save to parquet files.

    Args:
        json_path: Path to goodreads_books.json.gz file (str or Path)
        output_csv: Path for CSV output (str or Path, used to derive parquet name)
        output_dir: Directory to save output files (str or Path)
    """
    # Convert all inputs to Path objects for consistent handling
    json_path = Path(json_path)
    output_csv = Path(output_csv)
    output_dir = Path(output_dir)

    logger.info(f"Processing book features from {json_path}")
    start = time.time()

    # Define columns we want to extract
    extra_feature_cols = [
        "book_id",
        "description",
        "format",
        "title",
        "title_without_series",
        "language_code",
        "authors",
        "country_code",
    ]

    # Scan the JSON file lazily
    df = pl.scan_ndjson(str(json_path))

    # Extract and process extra features
    logger.info("Processing extra features...")
    # Use .stem to get filename without extension, then add new suffix
    parquet_filename = output_csv.stem + ".snap.parquet"
    parquet_path = output_dir / parquet_filename
    df.select(extra_feature_cols).sink_parquet(str(parquet_path))

    # Save titles parquet
    logger.info("Saving titles parquet...")
    titles_path = output_dir / "titles.snap.parquet"
    (
        pl.scan_parquet(str(parquet_path))
        .select(["book_id", "title", "title_without_series"])
        .sink_parquet(str(titles_path))
    )

    # Try to save CSV if possible (may fail with nested data)
    try:
        logger.info("Attempting to save CSV (may be skipped if data is nested)...")
        csv_path = output_dir / output_csv.name
        # Convert nested columns to strings for CSV compatibility
        (
            pl.scan_parquet(str(parquet_path))
            .with_columns([
                pl.col(col).cast(pl.Utf8, strict=False)
                for col in ["authors", "description"]
            ])
            .sink_csv(str(csv_path), quote_style="necessary")
        )
        logger.info(f"CSV saved successfully to {csv_path}")
    except Exception as e:
        logger.warning(f"Skipping CSV output (nested data not supported in CSV): {e}")
        logger.info("Parquet file is available and recommended for nested data")

    time_seconds = time.time() - start
    logger.info(f"Finished processing extra features in {time_seconds:.1f} seconds")


def save_simple_book_features(json_path, output_file, output_dir: str = "data"):
    """Extract and process simple (numeric) book features.

    Args:
        json_path: Path to goodreads_books.json.gz file (str or Path)
        output_file: Path for output CSV (str or Path, used to derive parquet name)
        output_dir: Directory to save output files (str or Path)
    """
    # Convert all inputs to Path objects for consistent handling
    json_path = Path(json_path)
    output_file = Path(output_file)
    output_dir = Path(output_dir)

    logger.info(f"Processing simple book features from {json_path}")
    start = time.time()

    # Define columns to keep
    keep_cols = [
        "book_id",
        "work_id",
        "publication_year",
        "is_ebook",
        "num_pages",
        "ratings_count",
        "text_reviews_count",
        "average_rating",
    ]

    # Scan and process with lazy operations
    logger.info("Reading and transforming data...")
    df = (
        pl.scan_ndjson(str(json_path))
        .with_columns([
            # Convert is_ebook to binary
            pl.when(pl.col("is_ebook") == "true")
            .then(1)
            .otherwise(0)
            .alias("is_ebook"),
            # Cast numeric columns
            pl.col("book_id").cast(pl.Int64, strict=False),
            pl.col("work_id").cast(pl.Int64, strict=False),
            pl.col("publication_year").cast(pl.Int64, strict=False),
            pl.col("num_pages").cast(pl.Int64, strict=False),
            pl.col("ratings_count").cast(pl.Int64, strict=False),
            pl.col("text_reviews_count").cast(pl.Int64, strict=False),
            pl.col("average_rating").cast(pl.Float64, strict=False),
        ])
        .select(keep_cols)
    )

    # Stream to parquet
    logger.info("Writing parquet...")
    # Use .stem to get filename without extension, then add new suffix
    parquet_filename = output_file.stem + ".snap.parquet"
    parquet_path = output_dir / parquet_filename
    df.sink_parquet(str(parquet_path))

    # Optionally save CSV (numeric features should be compatible)
    try:
        logger.info("Writing CSV...")
        csv_path = output_dir / output_file.name
        pl.scan_parquet(str(parquet_path)).sink_csv(str(csv_path))
        logger.info(f"CSV saved successfully to {csv_path}")
    except Exception as e:
        logger.warning(f"Skipping CSV output: {e}")
        logger.info("Parquet file is available")

    time_seconds = time.time() - start
    logger.info(f"Finished processing simple features in {time_seconds:.1f} seconds")


def save_interactions(input_path, output_dir: str = "data"):
    """Convert interactions CSV to parquet format.

    Args:
        input_path: Path to goodreads_interactions.csv (str or Path)
        output_dir: Directory to save output files (str or Path)
    """
    # Convert all inputs to Path objects for consistent handling
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    logger.info(f"Converting interactions to parquet: {input_path}")
    start = time.time()

    # Use lazy scan and sink for memory efficiency
    # Use .stem to get filename without extension, then add new suffix
    parquet_filename = input_path.stem + ".snap.parquet"
    output_path = output_dir / parquet_filename
    pl.scan_csv(str(input_path)).sink_parquet(str(output_path))

    time_seconds = time.time() - start
    logger.info(f"Finished converting interactions in {time_seconds:.1f} seconds")


def main(json_path, csv_path, interactions_path, output_path, output_dir: str = "data"):
    """Serialize DataFrames to parquet files using Polars.

    Args:
        json_path: Path to goodreads_books.json.gz
        csv_path: Path for books_extra_features.csv output
        interactions_path: Path to goodreads_interactions.csv
        output_path: Path for books_simple_features.csv output
        output_dir: Directory to save all output files
    """
    logger.info("Starting data serialization with Polars...")

    # Start timing the entire process
    overall_start = time.time()

    # Create output directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)

    save_book_features(json_path, csv_path, output_dir)
    save_simple_book_features(json_path, output_path, output_dir)
    save_interactions(interactions_path, output_dir)

    # Log total runtime
    total_runtime = time.time() - overall_start
    logger.info(f"All data serialization complete! Total runtime: {total_runtime:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serialize GoodReads data to parquet using Polars")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10_000_000,
        help="Batch size for processing (default: 10,000,000)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory for parquet files (default: data)",
    )
    args = parser.parse_args()

    logger.info(f"Using batch_size: {args.batch_size:,}")

    PATH_JSON = Path("data").joinpath("goodreads_books.json.gz")

    if not PATH_JSON.exists():
        logger.info("Data files not found, downloading...")
        download_goodreads_data("data")

    # Output paths
    CSV_PATH = Path("data").joinpath("books_extra_features.csv")
    OUTPUT_PATH = Path("data").joinpath("books_simple_features.csv")
    INTERACTIONS_PATH = Path("data").joinpath("goodreads_interactions.csv")

    main(
        json_path=PATH_JSON,
        csv_path=CSV_PATH,
        interactions_path=INTERACTIONS_PATH,
        output_path=OUTPUT_PATH,
        output_dir=args.output_dir,
    )
