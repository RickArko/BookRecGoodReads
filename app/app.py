"""Book Recommendation Streamlit App.

A web interface for getting book recommendations using KNN collaborative filtering.
"""

import streamlit as st
import polars as pl
import sys
from pathlib import Path

# Add parent directory to path to import knn_recommender_sparse
sys.path.insert(0, str(Path(__file__).parent.parent))

from knn_recommender_sparse import (
    load_sparse_matrix,
    create_title_mapping,
    SparseKnnRecommender
)


@st.cache_resource
def load_recommender():
    """Load and cache the recommender model."""
    try:
        matrix, book_ids, user_ids = load_sparse_matrix("data/book_user_matrix_sparse.npz")
        title_to_idx, idx_to_title, book_id_to_title = create_title_mapping(book_ids)

        recommender = SparseKnnRecommender(
            matrix, book_ids, title_to_idx, idx_to_title,
            metric="cosine", n_neighbors=20
        )
        recommender.fit()
        return recommender, title_to_idx, idx_to_title, book_id_to_title, book_ids
    except Exception as e:
        st.error(f"Error loading recommender: {e}")
        st.info("Make sure you've run `python prepare_data.py` first to generate the data files.")
        return None, None, None, None, None


@st.cache_data
def load_book_metadata():
    """Load book metadata for display."""
    try:
        # Load basic titles
        titles_df = pl.read_parquet("data/filtered_titles.parquet")

        # Load extended features
        features_df = pl.read_parquet("data/books_simple_features.snap.parquet")

        # Join datasets
        books_df = titles_df.join(features_df, on="book_id", how="left")

        return books_df
    except Exception as e:
        st.warning(f"Could not load extended metadata: {e}")
        # Fallback to basic titles
        try:
            return pl.read_parquet("data/filtered_titles.parquet")
        except:
            return None


def format_book_display(title, book_info):
    """Format book information for display."""
    # Use the title from recommendations (which is accurate)
    # but try to get metadata from book_info if available
    year = book_info.get("publication_year")
    rating = book_info.get("average_rating")
    pages = book_info.get("num_pages")
    ratings_count = book_info.get("ratings_count")

    details = []
    if year and not pl.datatypes.Null in [type(year)]:
        details.append(f"📅 {int(year)}")
    if rating and not pl.datatypes.Null in [type(rating)]:
        details.append(f"⭐ {rating:.2f}")
    if ratings_count and not pl.datatypes.Null in [type(ratings_count)]:
        details.append(f"👥 {int(ratings_count):,} ratings")
    if pages and not pl.datatypes.Null in [type(pages)]:
        details.append(f"📖 {int(pages)} pages")

    return title, " | ".join(details) if details else None


def main():
    st.set_page_config(
        page_title="Book Recommender",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 Book Recommendation System")
    st.markdown("Get personalized book recommendations using collaborative filtering")

    # Load data
    with st.spinner("Loading recommendation model..."):
        recommender, title_to_idx, idx_to_title, book_id_to_title, book_ids = load_recommender()
        books_df = load_book_metadata()

    if recommender is None:
        st.error("Failed to load recommender. Please check the data files.")
        return

    # Sidebar - Settings
    st.sidebar.header("⚙️ Settings")

    n_recommendations = st.sidebar.slider(
        "Number of recommendations",
        min_value=5,
        max_value=50,
        value=20,
        step=5
    )

    fuzzy_threshold = st.sidebar.slider(
        "Fuzzy match threshold",
        min_value=50,
        max_value=100,
        value=60,
        help="Lower values allow more flexible matching"
    )

    # Optional filters
    st.sidebar.header("🔍 Filters")

    min_rating = None
    min_year = None
    max_year = None

    if books_df is not None and "average_rating" in books_df.columns:
        filter_by_rating = st.sidebar.checkbox("Filter by minimum rating")
        if filter_by_rating:
            min_rating = st.sidebar.slider(
                "Minimum average rating",
                min_value=0.0,
                max_value=5.0,
                value=3.5,
                step=0.1
            )

    if books_df is not None and "publication_year" in books_df.columns:
        filter_by_year = st.sidebar.checkbox("Filter by publication year")
        if filter_by_year:
            col1, col2 = st.sidebar.columns(2)
            with col1:
                min_year = st.number_input("From", min_value=1800, max_value=2030, value=1990)
            with col2:
                max_year = st.number_input("To", min_value=1800, max_value=2030, value=2024)

    # Main content
    st.header("Select a book you enjoyed")

    # Get sorted book titles
    available_titles = sorted(title_to_idx.keys())

    # Search/select book
    selected_book = st.selectbox(
        "Type to search or select a book:",
        options=available_titles,
        index=None,
        placeholder="Start typing a book title..."
    )

    # Alternative: text input for fuzzy search
    st.markdown("**OR**")
    fuzzy_search = st.text_input(
        "Enter a book title (fuzzy search):",
        placeholder="e.g., 'Harry Potter'"
    )

    # Get recommendations button
    if st.button("Get Recommendations", type="primary"):
        book_query = selected_book if selected_book else fuzzy_search

        if not book_query:
            st.warning("Please select or enter a book title.")
            return

        with st.spinner(f"Finding recommendations for '{book_query}'..."):
            recommendations = recommender.recommend(
                book_query,
                n_recommendations=n_recommendations,
                threshold=fuzzy_threshold
            )

        if not recommendations:
            st.error(f"No recommendations found for '{book_query}'. Try lowering the fuzzy match threshold.")
            return

        # Display results
        st.success(f"Found {len(recommendations)} recommendations!")

        # Create book_id to metadata lookup
        book_id_to_metadata = {}
        if books_df is not None:
            for row in books_df.iter_rows(named=True):
                book_id_to_metadata[row["book_id"]] = row

        # Filter and display recommendations
        filtered_recs = []
        for title, distance in recommendations:
            # Map: title -> matrix idx -> book_id -> metadata
            idx = title_to_idx.get(title)
            if idx is not None:
                book_id = book_ids[idx]
                book_info = book_id_to_metadata.get(book_id, {})
            else:
                book_info = {}

            # Apply filters
            if min_rating and book_info.get("average_rating"):
                if book_info["average_rating"] < min_rating:
                    continue

            if min_year and book_info.get("publication_year"):
                if book_info["publication_year"] < min_year:
                    continue

            if max_year and book_info.get("publication_year"):
                if book_info["publication_year"] > max_year:
                    continue

            filtered_recs.append((title, distance, book_info))

        if not filtered_recs:
            st.warning("No recommendations match your filters. Try adjusting the filter settings.")
            return

        st.markdown(f"### Top {len(filtered_recs)} Recommendations")

        # Display recommendations in a nice format
        for idx, (title, distance, book_info) in enumerate(filtered_recs, 1):
            similarity = (1 - distance) * 100  # Convert to percentage

            with st.container():
                col1, col2 = st.columns([4, 1])

                with col1:
                    formatted_title, details = format_book_display(title, book_info)
                    st.markdown(f"**{idx}. {formatted_title}**")
                    if details:
                        st.caption(details)

                with col2:
                    st.metric("Match", f"{similarity:.1f}%")

                st.divider()

        # Stats
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Stats")
        st.sidebar.metric("Total recommendations", len(filtered_recs))
        avg_similarity = sum((1 - d) * 100 for _, d, _ in filtered_recs) / len(filtered_recs)
        st.sidebar.metric("Average match", f"{avg_similarity:.1f}%")


if __name__ == "__main__":
    main()
