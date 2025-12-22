"""Create content-based feature vectors from book metadata.

This script creates:
1. TF-IDF vectors from book shelves (genres/tags)
2. Author similarity features
3. Combined content feature matrix
"""

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix, hstack, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from loguru import logger


def create_content_features(
    metadata_path="data/book_metadata.parquet",
    matrix_mapping_path="data/sparse_matrix_book_mapping.parquet",
    output_path="data/content_features.npz",
    output_mapping_path="data/content_features_mapping.parquet",
):
    """Create content-based feature matrix for books.

    Args:
        metadata_path: Path to book metadata
        matrix_mapping_path: Path to sparse matrix book mapping
        output_path: Output path for content features matrix
        output_mapping_path: Output path for feature mapping
    """
    logger.info("Loading book metadata...")
    metadata = pl.read_parquet(metadata_path)
    logger.info(f"Loaded metadata for {len(metadata):,} books")

    logger.info("Loading matrix book IDs...")
    matrix_mapping = pl.read_parquet(matrix_mapping_path)
    matrix_book_ids = matrix_mapping.sort("matrix_idx")["book_id"].to_list()
    logger.info(f"Matrix has {len(matrix_book_ids):,} books")

    # Create mapping from book_id to metadata
    metadata_dict = {
        row["book_id"]: row for row in metadata.iter_rows(named=True)
    }

    logger.info(f"Books with metadata: {len(metadata_dict):,} / {len(matrix_book_ids):,} ({100*len(metadata_dict)/len(matrix_book_ids):.1f}%)")

    # ==================================================================
    # 1. CREATE SHELF (GENRE) FEATURES USING TF-IDF
    # ==================================================================
    logger.info("\n[1/3] Creating shelf (genre) TF-IDF features...")

    # Prepare shelf text for all books in matrix (in order)
    shelf_texts = []
    for book_id in matrix_book_ids:
        if book_id in metadata_dict:
            shelves = metadata_dict[book_id].get("shelves", "")
            # Replace commas with spaces for TF-IDF
            shelf_text = shelves.replace(",", " ") if shelves else ""
            shelf_texts.append(shelf_text)
        else:
            shelf_texts.append("")  # Empty for books without metadata

    # Create TF-IDF vectorizer for shelves
    shelf_vectorizer = TfidfVectorizer(
        max_features=500,  # Keep top 500 shelves
        min_df=3,  # Shelf must appear in at least 3 books
        lowercase=True,
        token_pattern=r"[a-z][a-z0-9\-]+",  # Handle hyphenated genres
    )

    shelf_tfidf = shelf_vectorizer.fit_transform(shelf_texts)
    logger.info(f"Shelf TF-IDF matrix shape: {shelf_tfidf.shape}")
    logger.info(f"Number of unique shelves: {len(shelf_vectorizer.get_feature_names_out())}")
    logger.info(f"Top 20 shelves by IDF: {shelf_vectorizer.get_feature_names_out()[:20].tolist()}")

    # ==================================================================
    # 2. CREATE AUTHOR FEATURES
    # ==================================================================
    logger.info("\n[2/3] Creating author features...")

    # Extract author IDs for each book
    author_id_lists = []
    for book_id in matrix_book_ids:
        if book_id in metadata_dict:
            author_ids_str = metadata_dict[book_id].get("author_ids", "")
            if author_ids_str:
                author_ids = [aid for aid in author_ids_str.split(",") if aid]
            else:
                author_ids = []
            author_id_lists.append(author_ids)
        else:
            author_id_lists.append([])

    # Use MultiLabelBinarizer for author features
    mlb = MultiLabelBinarizer(sparse_output=True)
    author_features = mlb.fit_transform(author_id_lists)

    logger.info(f"Author feature matrix shape: {author_features.shape}")
    logger.info(f"Number of unique authors: {len(mlb.classes_)}")

    # Filter to authors with at least 2 books (reduce dimensionality)
    author_counts = np.asarray(author_features.sum(axis=0)).ravel()
    author_mask = author_counts >= 2
    author_features_filtered = author_features[:, author_mask]

    logger.info(f"Authors with 2+ books: {author_mask.sum():,} / {len(mlb.classes_):,}")
    logger.info(f"Filtered author matrix shape: {author_features_filtered.shape}")

    # ==================================================================
    # 3. CREATE NUMERIC FEATURES
    # ==================================================================
    logger.info("\n[3/3] Creating numeric features...")

    numeric_features = []
    for book_id in matrix_book_ids:
        if book_id in metadata_dict:
            meta = metadata_dict[book_id]
            features = [
                np.log1p(meta.get("ratings_count", 0)),  # Log of ratings count
                meta.get("average_rating", 0) / 5.0,  # Normalized rating
                (meta.get("publication_year", 0) - 1900) / 100.0 if meta.get("publication_year", 0) > 0 else 0,  # Normalized year
                np.log1p(meta.get("num_pages", 0)) / 10.0,  # Log of pages
            ]
        else:
            features = [0, 0, 0, 0]
        numeric_features.append(features)

    numeric_features = np.array(numeric_features)
    logger.info(f"Numeric features shape: {numeric_features.shape}")

    # ==================================================================
    # 4. COMBINE ALL FEATURES
    # ==================================================================
    logger.info("\nCombining all features...")

    # Convert numeric to sparse
    numeric_sparse = csr_matrix(numeric_features)

    # Normalize each feature type
    shelf_tfidf_norm = normalize(shelf_tfidf, norm="l2", axis=1)
    author_features_norm = normalize(author_features_filtered, norm="l2", axis=1)
    numeric_sparse_norm = normalize(numeric_sparse, norm="l2", axis=1)

    # Concatenate horizontally
    content_features = hstack([
        shelf_tfidf_norm,
        author_features_norm,
        numeric_sparse_norm
    ], format="csr")

    logger.info(f"\nFinal content features matrix shape: {content_features.shape}")
    logger.info(f"  Shelf features: {shelf_tfidf_norm.shape[1]}")
    logger.info(f"  Author features: {author_features_norm.shape[1]}")
    logger.info(f"  Numeric features: {numeric_sparse_norm.shape[1]}")
    logger.info(f"  Total features: {content_features.shape[1]}")

    # Save the content features matrix
    logger.info(f"\nSaving content features to {output_path}...")
    save_npz(output_path, content_features)

    # Save mapping with metadata flags
    logger.info(f"Saving feature mapping to {output_mapping_path}...")
    mapping_df = pl.DataFrame({
        "matrix_idx": list(range(len(matrix_book_ids))),
        "book_id": matrix_book_ids,
        "has_metadata": [bid in metadata_dict for bid in matrix_book_ids],
        "num_shelves": [len(metadata_dict.get(bid, {}).get("shelves", "").split(",")) if metadata_dict.get(bid, {}).get("shelves") else 0 for bid in matrix_book_ids],
        "num_authors": [metadata_dict.get(bid, {}).get("num_authors", 0) for bid in matrix_book_ids],
    })
    mapping_df.write_parquet(output_mapping_path)

    # Statistics
    logger.info("\n" + "="*70)
    logger.info("Content Features Statistics:")
    logger.info(f"  Books with any metadata: {mapping_df.filter(pl.col('has_metadata')).height:,} ({100*mapping_df.filter(pl.col('has_metadata')).height/len(mapping_df):.1f}%)")
    logger.info(f"  Books with shelves: {mapping_df.filter(pl.col('num_shelves') > 0).height:,}")
    logger.info(f"  Books with authors: {mapping_df.filter(pl.col('num_authors') > 0).height:,}")
    logger.info(f"  Total feature dimensions: {content_features.shape[1]:,}")
    logger.info(f"  Matrix sparsity: {100 * (1 - content_features.nnz / (content_features.shape[0] * content_features.shape[1])):.2f}%")
    logger.info("="*70)

    return content_features, mapping_df


if __name__ == "__main__":
    create_content_features()
