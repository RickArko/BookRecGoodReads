# Recommendation Models

This document explains the different recommendation approaches used in this project and when to use each one.

## Overview

The BookRecommender implements three complementary approaches:

1. **Collaborative Filtering (KNN)** - User-item interaction patterns
2. **Content-Based Filtering** - Book metadata and features
3. **Hybrid Approach** - Weighted combination of both methods

---

## 1. Collaborative Filtering (KNN)

**Implementation**: `src/knn_recommender_sparse.py`

### How It Works

Uses K-Nearest Neighbors on a sparse user-item interaction matrix:
- **Input**: Book-user rating matrix (books × users)
- **Method**: Cosine similarity between book rating vectors
- **Output**: Books with similar rating patterns from similar readers

### Strengths

- **Serendipity**: Discovers books you might not find through metadata alone
- **Captures implicit patterns**: Learns from collective user behavior
- **No feature engineering**: Works with raw interaction data
- **Cold-start for books**: Can recommend new books once they get initial ratings

### Weaknesses

- **Cold-start for users**: Cannot recommend to users with no interaction history
- **Popularity bias**: Tends to favor popular books
- **Data sparsity**: Most users rate very few books (sparse matrix ~0.04% density)
- **Scalability**: Requires large interaction dataset to be effective

### When To Use

Best for users with existing reading history when you want to discover books similar to their past preferences through community behavior patterns.

### Performance

```
Matrix: 50,000 books × 1,000,000 users
Density: ~0.04% (40M interactions)
Query time: ~100-200ms per recommendation
Memory: ~500MB (sparse CSR format)
```

---

## 2. Content-Based Filtering

**Implementation**: `src/create_content_features.py`, `src/hybrid_recommender.py`

### How It Works

Uses book metadata to find similar books:
- **Features**: Authors, genres (shelves), average rating, ratings count, publication year
- **Method**: TF-IDF for text features + normalized numeric features
- **Similarity**: Cosine similarity in feature space

### Strengths

- **Explainability**: Clear why books are recommended (same author, genre, etc.)
- **No cold-start**: Works for any book with metadata
- **Transparency**: Users can filter by specific attributes
- **Complementary**: Finds books collaborative filtering might miss

### Weaknesses

- **Limited discovery**: Only recommends books similar to known preferences
- **Feature dependency**: Quality depends on metadata richness
- **No wisdom of crowds**: Ignores collective user behavior
- **Over-specialization**: May not diversify recommendations

### When To Use

Best when:
- User wants books similar to a specific book they liked
- Cold-start scenarios (new users or books)
- Explainability is important
- Filtering by specific attributes (genre, author, rating)

### Performance

```
Features: ~5,000 dimensions (TF-IDF + numeric)
Books: 50,000
Query time: ~50-100ms per recommendation
Memory: ~300MB (sparse feature matrix)
```

---

## 3. Hybrid Approach

**Implementation**: `src/hybrid_recommender.py`

### How It Works

Weighted combination of collaborative and content-based scores:

```python
hybrid_score = 0.6 × collaborative_score + 0.4 × content_score
```

### Algorithm

1. Query collaborative KNN for top N×3 candidates
2. Query content-based KNN for top N×3 candidates
3. Normalize scores (convert distance to similarity)
4. Combine with weights
5. Sort by hybrid score and return top N

### Strengths

- **Best of both worlds**: Balances discovery with relevance
- **Robustness**: Works even when one method has sparse data
- **Tunable**: Adjust weights based on use case
- **Diversity**: Combines different recommendation signals

### Weaknesses

- **Complexity**: More components to maintain
- **Tuning required**: Optimal weights depend on data
- **Higher latency**: Runs both methods

### When To Use

**Default choice** for production - provides balanced recommendations leveraging both user behavior and book attributes.

### Tuning Weights

```python
# Favor collaborative (discovery-focused)
HybridRecommender(collaborative_weight=0.7, content_weight=0.3)

# Favor content (explainability-focused)
HybridRecommender(collaborative_weight=0.4, content_weight=0.6)

# Balanced (default)
HybridRecommender(collaborative_weight=0.6, content_weight=0.4)
```

---

## Comparison Summary

| Aspect | Collaborative (KNN) | Content-Based | Hybrid |
|--------|-------------------|---------------|--------|
| **Cold-start (users)** | ❌ Poor | ✅ Good | ✅ Good |
| **Cold-start (books)** | ⚠️ Needs some ratings | ✅ Works immediately | ✅ Good |
| **Serendipity** | ✅ High | ❌ Low | ✅ Medium-High |
| **Explainability** | ❌ Opaque | ✅ Clear | ⚠️ Mixed |
| **Diversity** | ⚠️ Popular bias | ✅ Feature-based | ✅ Good |
| **Data requirements** | ⚠️ Needs interactions | ✅ Just metadata | ⚠️ Both |
| **Query latency** | ~100-200ms | ~50-100ms | ~150-250ms |

---

## Technical Implementation Details

### Data Preparation

**Collaborative Filtering:**
```bash
# Creates sparse user-item matrix
uv run python prepare_data.py --min_reads=10 --top_books=50000 --top_users=1000000
```

**Content-Based:**
```bash
# Extracts metadata and creates TF-IDF features
uv run python src/extract_metadata.py
uv run python src/create_content_features.py
```

### Matrix Representations

**Sparse Matrix (Collaborative):**
- Format: CSR (Compressed Sparse Row)
- Storage: `data/book_user_matrix_sparse.npz`
- Memory efficient for sparse data (<1% density)

**Feature Matrix (Content):**
- Format: Sparse matrix (TF-IDF + scaled numeric)
- Storage: `data/content_features.npz`
- Combines text and numeric features

### Evaluation

To compare methods on a query:
```bash
uv run python src/compare_recommenders.py
```

---

## Choosing the Right Approach

### Use **Collaborative Filtering** when:
- You have rich interaction data (>1M ratings)
- Users have established reading history
- Discovery and serendipity are priorities
- Popularity-based recommendations are acceptable

### Use **Content-Based** when:
- Metadata quality is high
- Explainability is critical
- Cold-start scenarios are common
- Users want specific genres/authors

### Use **Hybrid** when:
- Building a production system (recommended)
- You want balanced recommendations
- Both data sources are available
- Robustness to data sparsity is important

---

## Future Improvements

Potential enhancements to explore:

1. **Matrix Factorization** (ALS, SVD++) - Better scalability than KNN
2. **Neural Collaborative Filtering** - Deep learning for complex patterns
3. **Context-aware**: Incorporate time, reading history sequence
4. **Multi-armed bandits**: Exploration vs exploitation trade-off
5. **Graph-based**: User-book-author-genre knowledge graphs

---

## References

- [Collaborative Filtering Deep Dive](https://www.kaggle.com/code/jhoward/collaborative-filtering-deep-dive)
- [UCSD Book Graph Dataset](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home)
- Koren, Y. (2009). "Collaborative Filtering with Temporal Dynamics"
- Lops, P., et al. (2011). "Content-based Recommender Systems"
