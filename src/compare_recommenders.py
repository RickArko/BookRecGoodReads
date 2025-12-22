"""Compare KNN-only vs Hybrid recommendations side-by-side."""

import time
from src.hybrid_recommender import HybridRecommender


def compare_recommendations(query, n_recommendations=5):
    """Compare collaborative-only vs hybrid recommendations."""
    print("\n" + "=" * 100)
    print(f"COMPARISON: '{query}'")
    print("=" * 100)

    # Initialize recommender
    print("\nInitializing recommender...")
    start = time.time()
    recommender = HybridRecommender(collaborative_weight=0.6, content_weight=0.4, n_neighbors=30)
    init_time = time.time() - start
    print(f"Initialization time: {init_time:.2f}s")

    # Find the book
    matches = recommender.fuzzy_search(query, threshold=60)
    if not matches:
        print(f"\nNo matches found for '{query}'")
        return

    best_title, book_idx, match_score = matches[0]
    print(f"\nUsing: {best_title} (match: {match_score}%)")

    # Show book details
    book_id = recommender.book_ids[book_idx]
    if book_id in recommender.metadata_dict:
        meta = recommender.metadata_dict[book_id]
        print(f"  Authors: {meta.get('authors', 'N/A')[:60]}")
        print(f"  Rating: {meta.get('average_rating', 0):.2f} ({meta.get('ratings_count', 0):,} ratings)")
        shelves = meta.get("shelves", "")
        if shelves:
            top_shelves = shelves.split(",")[:5]
            print(f"  Genres: {', '.join(top_shelves)}")

    # Get both types of recommendations
    print(f"\n{'-' * 100}")
    print(f"{'COLLABORATIVE-ONLY (KNN)':<50} | {'HYBRID (KNN + Content)':<48}")
    print(f"{'-' * 100}")

    collab_recs = recommender.recommend_collaborative(book_idx, n_recommendations)
    hybrid_recs = recommender.recommend_hybrid(book_idx, n_recommendations)

    for i in range(n_recommendations):
        # Collaborative recommendation
        if i < len(collab_recs):
            idx, dist = collab_recs[i]
            title = recommender.idx_to_title.get(idx, f"Book ID: {recommender.book_ids[idx]}")
            sim = 1 - dist
            collab_text = f"{i + 1}. {title[:40]:<40} (sim:{sim:.3f})"
        else:
            collab_text = f"{i + 1}. {'(none)':<40}"

        # Hybrid recommendation
        if i < len(hybrid_recs):
            idx, combined, collab_score, content_score = hybrid_recs[i]
            title = recommender.idx_to_title.get(idx, f"Book ID: {recommender.book_ids[idx]}")
            hybrid_text = f"{i + 1}. {title[:35]:<35} (c:{collab_score:.2f} + t:{content_score:.2f})"
        else:
            hybrid_text = f"{i + 1}. {'(none)':<35}"

        print(f"{collab_text:<50} | {hybrid_text:<48}")

    print("=" * 100)


def main():
    """Run comparisons for multiple queries."""
    print("\n" + "=" * 100)
    print(" " * 30 + "RECOMMENDER COMPARISON")
    print(" " * 20 + "KNN Collaborative-Only  vs  Hybrid (Collaborative + Content)")
    print("=" * 100)

    test_queries = [
        "1984",
        "Lord of the Rings",
        "The Great Gatsby",
    ]

    for query in test_queries:
        compare_recommendations(query, n_recommendations=5)
        print()

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print("""
Collaborative-Only (KNN):
  + Uses actual user interactions (who read what)
  + Good for popular books with many interactions
  - Fails for books with sparse data
  - Cannot recommend new books without interactions

Hybrid (Collaborative + Content):
  + Combines interaction data with book features (genres, authors)
  + Works for books with sparse interaction data
  + More diverse recommendations
  + Can recommend based on genre/author similarity
  - Slightly more complex

RECOMMENDATION: Use Hybrid for better coverage and diversity!
""")
    print("=" * 100)


if __name__ == "__main__":
    main()
