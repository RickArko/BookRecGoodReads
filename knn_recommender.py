"""Model to make simple KNN prediction"""

import os
import time
import gc
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz

# Load processed data from parquet files
matrix = pd.read_parquet("data/book_user_matrix.parquet")
DFTMP = pd.read_parquet("data/filtered_titles.parquet")

# Set book_id as index for matrix (first column from Polars pivot)
if "book_id" in matrix.columns:
    matrix = matrix.set_index("book_id")

KNN20 = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=20, n_jobs=-1)

# Create hashmap: title -> row index in matrix
# Get the books that are in the matrix, in the same order as matrix rows
matrix_book_ids = matrix.index.tolist()
DFTMP_indexed = DFTMP.set_index("book_id")

# Create mapping from title to matrix row index
hashmap = {}
for i, book_id in enumerate(matrix_book_ids):
    if book_id in DFTMP_indexed.index:
        title = DFTMP_indexed.loc[book_id, "title"]
        hashmap[title] = i

DEFAULT_TOP_BOOKS = [
    536,
    943,
    1000,
    1387,
    968,
    941,
    1386,
    938,
    1473,
    1402,
    461,
    586,
    996,
    1116,
    759,
    66,
    1203,
    524,
    1007,
    7008,
]


def fuzzy_matching(mapper, fav_book, verbose=True):
    """Return book index of closest match via fuzzy ratio.
    If no match found, return None

    inputs:
    ------
        mapper:   dict, {title: index of the book in data}
        fav_book: str
        verbose:  bool

    return:
    ------
        index of the closest match
    """
    match_tuple = []

    for title, idx in mapper.items():
        ratio = fuzz.ratio(title.lower(), fav_book.lower())  # Fixed: was 'book.lower()'
        if ratio >= 60:
            match_tuple.append((title, idx, ratio))

    match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
    if not match_tuple:
        if verbose:
            print("No match.")
        return
    if verbose:
        print(f"possible matches: {[x[0] for x in match_tuple]}")
    return match_tuple[0][1]


class KnnRecommender:
    """Item-based Collaborative Filtering Recommender"""

    def __init__(self, matrix, hashmap, model=KNN20):
        """Initialize Class"""
        self.data = matrix
        self.hashmap = hashmap
        self.model = model

    def _train_model(self):
        start = time.time()
        self.model.fit(csr_matrix(self.data))
        msg = f"Fit model in {time.time() - start} seconds"
        print(msg)
        return self

    def make_recommendation_from_book(self, book_name, n_recommendations):
        """Give top n_recommendations books based on input
        book_name
        """
        # Get book index from fuzzy matching
        idx = fuzzy_matching(self.hashmap, book_name)
        if idx is None:
            return

        distances, indices = self.model.kneighbors(
            self.data.iloc[idx].values.reshape(1, -1),  # Fixed: was 'data[idx]'
            n_neighbors=n_recommendations + 1,
        )

        # sort recommendations
        raw_recommends = sorted(
            list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())),
            key=lambda x: x[1],
        )[:0:-1]

        # Create reverse mapping: matrix row index -> title (or book_id if title not available)
        reverse_hashmap = {v: k for k, v in self.hashmap.items()}
        matrix_book_ids = self.data.index.tolist()

        print("Recommendations for {}:".format(book_name))
        for i, (idx, dist) in enumerate(raw_recommends):
            # Use title if available, otherwise use book_id
            if idx in reverse_hashmap:
                book_label = reverse_hashmap[idx]
            else:
                book_label = f"Book ID: {matrix_book_ids[idx]}"
            print("{0}: {1}, with distance of {2}".format(i + 1, book_label, dist))


if __name__ == "__main__":
    start = time.time()
    rec = KnnRecommender(matrix, hashmap)
    rec._train_model()

    # Try a book that's actually in the matrix
    test_book = "The Da Vinci Code"  # One of the 50 books in the matrix
    print(f"\nTesting with: {test_book}")
    rec.make_recommendation_from_book(test_book, n_recommendations=5)

    seconds_taken = time.time() - start
    print(f"\nTotal time: {seconds_taken:.2f} seconds")
