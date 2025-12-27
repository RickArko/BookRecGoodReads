# Data Quality Analysis - BookRecommender

## Executive Summary

The poor recommendation quality is caused by **severe data sparsity**, NOT bugs in the code. Popular books like "Lord of the Rings" and "Harry Potter" have insufficient interaction data to generate good collaborative filtering recommendations.

---

## Key Findings

### 1. Book Variant Fragmentation

**Problem**: Popular books have many different editions/translations, splitting their interactions across multiple book_ids.

**Evidence**:
- Lord of the Rings: **547 different book_ids** in the dataset
- Harry Potter: **1,512 different book_ids** in the dataset
- Only **1 variant of each** made it into the top 50,000 books

**Impact**: Interaction data is fragmented across editions instead of consolidated.

### 2. Low Ranking Due to Fragmentation

| Book | Book ID | Interactions | Rank (out of 2.3M) | Made Top 50K? |
|------|---------|--------------|-------------------|---------------|
| **LOTR #1** | 5898 | 718 | #43,161 | ✅ Yes (barely) |
| **LOTR #2** | 59317 | 464 | #66,951 | ❌ No (cut off) |
| **HP #1** | 162234 | 689 | #45,038 | ✅ Yes (barely) |
| **HP #2** | 57526 | 476 | #65,360 | ❌ No (cut off) |

**Cutoff Point**: Rank 50,000 = 621 interactions

### 3. Extreme Sparsity in Collaborative Matrix

**LOTR Book 5898 (the one that made it in):**
- Total users in matrix: 770,904
- Non-zero entries (interactions): 717
- **Density: 0.093%** (99.907% sparse)

**Result**: Cosine similarity finds only weak, random matches.

### 4. Poor KNN Recommendations

Testing KNN directly on LOTR book 5898:
```
Top recommendation: 22.1% similarity (The Woman in White)
2nd: 21.3% similarity (The Scarpetta Collection)
3rd: 21.1% similarity (In the Dark of the Night)
...
10th: 10.5% similarity
20th: 5.8% similarity
```

**These are essentially random books** - there's no meaningful signal.

---

## Root Cause Analysis

### Why Are Popular Books Ranked So Low?

**Top 20 books** have 165K-315K interactions:
1. book_id=943: 314,685 interactions
2. book_id=536: 313,343 interactions (The Lovely Bones)
3. book_id=786: 260,119 interactions

But **LOTR and HP only have ~700 interactions each**.

**Reason**: The UCSD GoodReads dataset appears to have:
- High-interaction counts for certain "bundle" or "special edition" books
- Fragmented data for popular series (each edition counted separately)
- Possible data collection bias toward specific book versions

### Data Pipeline Flow

```
Raw Data (2.3M books)
  ↓
Filter by min_reads=10 (keeps 1.17M books)
  ↓
Take top 50,000 by interaction count
  ↓
LOTR rank #43,161 → BARELY makes it
HP rank #45,038 → BARELY makes it
Other LOTR/HP variants → CUT OFF
  ↓
Result: 1 LOTR variant, 1 HP variant in final matrix
        Both with only ~700 interactions across 770K users
```

---

## Impact on User Experience

1. **Fuzzy search works perfectly** ✅
   - Finds "Lord of the Rings" with 100% match
   - Finds "Harry Potter" with 100% match

2. **Recommendations are terrible** ❌
   - Top match: only 22-35% similarity
   - Random, unrelated books
   - Many "Book ID: XXXXX" (missing titles)

3. **User sees disconnect** between good search and bad results

---

## Solutions

### Immediate Fix (Low Effort)

**Consolidate book variants before filtering:**

Create a new preprocessing step in `prepare_data.py`:

```python
def consolidate_book_editions(interactions_df, titles_df):
    """Group book editions by canonical title."""
    # Extract base title (remove edition info)
    # Example: "Harry Potter #1 (UK Edition)" → "Harry Potter #1"

    # Create canonical_book_id mapping
    # Sum interactions for all editions
    # Return consolidated interactions
```

**Expected Impact**: LOTR and HP would jump to top 1,000 books with 10K+ interactions each.

### Better Fix (Medium Effort)

**Use the Hybrid Recommender** (already implemented in `src/hybrid_recommender.py`):

1. Collaborative filtering for books with good interaction data
2. Content-based filtering for sparse books (uses metadata, genres, authors)

**To enable:**
```bash
# Generate content features
uv run python src/extract_metadata.py
uv run python src/create_content_features.py

# Use app_enhanced.py instead of app.py
# Update Dockerfile CMD to use app_enhanced.py
```

### Best Fix (Higher Effort)

**Use Matrix Factorization instead of KNN:**

Replace KNN with ALS (Alternating Least Squares) or SVD++:

```python
from implicit.als import AlternatingLeastSquares

# ALS handles sparsity much better than KNN
model = AlternatingLeastSquares(factors=100, iterations=20)
model.fit(sparse_matrix)
```

**Advantages**:
- Better with sparse data
- Learns latent factors
- More robust recommendations

**Files to modify**:
- `src/knn_recommender_sparse.py` → `src/als_recommender.py`
- Update `app/app.py` to use ALS model

---

## Data Quality Metrics

### Current Dataset
```
Total books in raw data: 2,360,650
Books with >=10 interactions: 1,176,315
Books in sparse matrix: 50,000
Users in sparse matrix: 770,904
Matrix density: 0.04%
```

### Popular Books in Matrix
```
LOTR variants in matrix: 1 out of 547 (0.18%)
HP variants in matrix: 1 out of 1,512 (0.07%)

LOTR interactions: 718 (0.093% density)
HP interactions: 689 (0.089% density)
```

---

## Recommendations Priority

1. **SHORT TERM** (today): Switch to hybrid recommender (`app_enhanced.py`)
2. **MEDIUM TERM** (this week): Implement book edition consolidation
3. **LONG TERM** (next sprint): Replace KNN with ALS/matrix factorization

---

## Testing Recommendations

After implementing fixes, validate with:

```python
# Test that popular books have good recommendations
test_books = ["Harry Potter", "Lord of the Rings", "The Hunger Games"]

for book in test_books:
    recs = recommender.recommend(book, n=10)
    avg_similarity = sum(1-dist for _, dist in recs) / len(recs)

    # Should be >70% average similarity for popular books
    assert avg_similarity > 0.70, f"{book} failed: {avg_similarity:.1%}"
```

---

## Conclusion

The recommendation quality issue is a **data problem, not a code problem**. The collaborative filtering algorithm is working correctly, but the input data is too sparse to produce meaningful results. Implementing edition consolidation or switching to content-based/hybrid recommendations will dramatically improve user experience.
