"""Extract book metadata from goodreads_books.json.gz for content-based filtering.

This script extracts:
- Authors
- Popular shelves (user-generated genres/tags)
- Other useful features

For the books in our sparse matrix.
"""

import gzip
import json
from pathlib import Path
import polars as pl
from loguru import logger


def extract_metadata(
    json_path="data/goodreads_books.json.gz",
    matrix_mapping_path="data/sparse_matrix_book_mapping.parquet",
    output_path="data/book_metadata.parquet",
    top_n_shelves=10,
):
    """Extract metadata for books in the matrix.

    Args:
        json_path: Path to goodreads_books.json.gz
        matrix_mapping_path: Path to sparse matrix book mapping
        output_path: Output path for extracted metadata
        top_n_shelves: Number of top shelves to keep per book
    """
    logger.info("Loading matrix book IDs...")
    mapping = pl.read_parquet(matrix_mapping_path)
    matrix_book_ids = set(mapping["book_id"].to_list())
    logger.info(f"Looking for metadata for {len(matrix_book_ids):,} books")

    # Collect metadata
    metadata_records = []
    total_processed = 0
    found_count = 0

    logger.info(f"Scanning {json_path}...")
    with gzip.open(json_path, "rt", encoding="utf-8") as f:
        for line in f:
            total_processed += 1

            if total_processed % 500000 == 0:
                logger.info(f"Processed {total_processed:,} books, found {found_count:,} matches")

            book = json.loads(line)
            book_id = int(book["book_id"])

            if book_id not in matrix_book_ids:
                continue

            found_count += 1

            # Extract authors
            authors = book.get("authors", [])
            author_ids = [a.get("author_id", "") for a in authors if a.get("author_id")]
            author_names = ", ".join([a.get("name", "") for a in authors if a.get("name", "")])[:500]

            # Extract popular shelves
            shelves = book.get("popular_shelves", [])

            # Sort by count and take top N
            shelves_sorted = sorted(
                shelves, key=lambda x: int(x.get("count", 0)), reverse=True
            )[:top_n_shelves]

            shelf_names = [s["name"] for s in shelves_sorted if s.get("name")]
            shelf_counts = [int(s["count"]) for s in shelves_sorted if s.get("count")]

            # Create record
            record = {
                "book_id": book_id,
                "title": book.get("title", "")[:500],
                "authors": author_names,
                "author_ids": ",".join(author_ids),
                "num_authors": len(author_ids),
                "shelves": ",".join(shelf_names),
                "shelf_counts": ",".join(map(str, shelf_counts)),
                "num_shelves": len(shelf_names),
                "average_rating": float(book.get("average_rating", 0)),
                "ratings_count": int(book.get("ratings_count", 0)),
                "publication_year": int(book.get("publication_year", 0) or 0),
                "num_pages": int(book.get("num_pages", 0) or 0),
                "language_code": book.get("language_code", "")[:10],
            }

            metadata_records.append(record)

    logger.info(f"Total books processed: {total_processed:,}")
    logger.info(f"Metadata extracted for: {found_count:,} / {len(matrix_book_ids):,} books ({100*found_count/len(matrix_book_ids):.1f}%)")

    # Convert to DataFrame and save
    logger.info("Creating DataFrame...")
    df = pl.DataFrame(metadata_records)

    logger.info(f"Saving to {output_path}...")
    df.write_parquet(output_path)

    logger.info(f"Saved {len(df)} book metadata records")

    # Show statistics
    logger.info("\nMetadata Statistics:")
    logger.info(f"  Books with authors: {df.filter(pl.col('num_authors') > 0).height:,} ({100*df.filter(pl.col('num_authors') > 0).height/len(df):.1f}%)")
    logger.info(f"  Books with shelves: {df.filter(pl.col('num_shelves') > 0).height:,} ({100*df.filter(pl.col('num_shelves') > 0).height/len(df):.1f}%)")
    logger.info(f"  Average shelves per book: {df['num_shelves'].mean():.2f}")
    logger.info(f"  Average authors per book: {df['num_authors'].mean():.2f}")

    # Show top shelves
    all_shelves = []
    for shelves_str in df["shelves"].to_list():
        if shelves_str:
            all_shelves.extend(shelves_str.split(","))

    from collections import Counter
    shelf_counts = Counter(all_shelves)
    logger.info(f"\nTop 20 most common shelves:")
    for shelf, count in shelf_counts.most_common(20):
        logger.info(f"  {shelf:30s} {count:6,} books")

    return df


if __name__ == "__main__":
    extract_metadata()
