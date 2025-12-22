# Quick Start Guide 🚀

Get your book recommendation system up and running in minutes!

## Prerequisites

- Python 3.10+
- UV package manager (or pip)
- At least 2GB free disk space
- At least 4GB RAM

## Step-by-Step Setup

### 1. Install Dependencies

```bash
# If using UV (recommended)
uv sync

# Or using pip
pip install polars scipy scikit-learn streamlit plotly fuzzywuzzy loguru
```

### 2. Download Data (if not already present)

The data files should already be in `data/` directory. If not:

```bash
# Download from UCSD Book Graph
# Place these files in data/:
# - goodreads_interactions.snap.parquet
# - titles.snap.parquet
# - goodreads_books.json.gz
```

### 3. Generate Recommendation Data

```bash
# Step 1: Create interaction matrix (takes ~3 minutes)
python prepare_data.py --min_reads=10 --top_books=50000 --top_users=1000000

# Step 2: Extract book metadata (takes ~2 minutes)
python extract_metadata.py

# Step 3: Create content features (takes ~30 seconds)
python create_content_features.py
```

### 4. Run the Enhanced App

```bash
streamlit run app/app_enhanced.py
```

The app will open in your browser at `http://localhost:8501`

---

## That's It! 🎉

You now have a fully functional hybrid book recommendation system!

---

## Quick Test

Try searching for these books in the app:

1. **"1984"** - Classic dystopian novel
2. **"The Great Gatsby"** - Classic literature
3. **"Lord of the Rings"** - Fantasy epic

Then experiment with:
- Adjusting the hybrid weights (collaborative vs content)
- Changing the recommendation method
- Applying filters (rating, year)
- Viewing analytics and metrics

---

## Command Reference

### Data Preparation

```bash
# Full dataset (50K books, 770K users)
python prepare_data.py --min_reads=10 --top_books=50000 --top_users=1000000

# Smaller dataset for testing (1K books, 50K users)
python prepare_data.py --min_reads=50 --top_books=1000 --top_users=50000

# Extract metadata
python extract_metadata.py

# Create features
python create_content_features.py
```

### Run Apps

```bash
# Enhanced Streamlit app (recommended)
streamlit run app/app_enhanced.py

# Basic Streamlit app
streamlit run app/app.py

# CLI comparison
python compare_recommenders.py

# CLI hybrid recommender
python hybrid_recommender.py
```

### Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Open: notebooks/training_evaluation.ipynb
```

---

## Troubleshooting

### "Error loading recommender"

**Problem**: Data files not found

**Solution**:
```bash
# Check if files exist
ls data/*.npz data/*.parquet

# If missing, run data preparation scripts
python prepare_data.py
python extract_metadata.py
python create_content_features.py
```

### "Out of memory"

**Problem**: Dataset too large

**Solution**: Use a smaller dataset
```bash
# Reduce to 10K books, 100K users
python prepare_data.py --min_reads=20 --top_books=10000 --top_users=100000
```

### "No matches found"

**Problem**: Search threshold too high

**Solution**: Lower the "Search threshold" slider in the app (try 60 or 50)

### Slow performance

**Problem**: First load is slow

**Solution**:
- Wait for initial load (4-6 seconds) - it's cached after that
- Use smaller dataset for testing
- Close other applications to free memory

---

## Next Steps

1. **Explore the Enhanced App**:
   - Try different hybrid weights
   - Compare recommendation methods
   - Check out the analytics tab

2. **Run the Evaluation Notebook**:
   - `notebooks/training_evaluation.ipynb`
   - See detailed metrics and visualizations
   - Compare different configurations

3. **Read the Docs**:
   - `IMPLEMENTATION_SUMMARY.md` - Complete overview
   - `app/README.md` - App-specific docs
   - `IMPROVEMENT_PLAN.md` - Future enhancements

4. **Customize**:
   - Modify hybrid weights in code
   - Add new features
   - Create custom filters
   - Build your own UI

---

## Pro Tips 💡

1. **Best Hybrid Weights**: Start with 60/40 (collaborative/content)
2. **For Popular Books**: Use Collaborative Only mode
3. **For New Books**: Use Content Only mode
4. **For Discovery**: Use Content Heavy (40/60) for more diversity
5. **Speed**: Recommendations are cached - second query is instant!

---

## Common Use Cases

### "I want accurate recommendations for popular books"
→ Use **Hybrid mode** with **70% collaborative, 30% content**

### "I want to discover new, diverse books"
→ Use **Hybrid mode** with **40% collaborative, 60% content**

### "I have a new book with no ratings"
→ Use **Content Only mode**

### "I want the fastest recommendations"
→ Use **Collaborative Only mode** (no content features)

---

## File Sizes

After generating data:

```
data/
├── book_user_matrix_sparse.npz     (~130 MB)
├── content_features.npz            (~15 MB)
├── book_metadata.parquet           (~2 MB)
├── sparse_matrix_book_mapping.parquet  (~1 MB)
├── sparse_matrix_user_mapping.parquet  (~10 MB)
└── filtered_titles.parquet         (~20 MB)

Total: ~180 MB
```

---

## Performance Expectations

| Operation | Time |
|-----------|------|
| Data preparation | 3-5 minutes |
| Metadata extraction | 2-3 minutes |
| Feature creation | 30 seconds |
| App cold start | 4-6 seconds |
| First recommendation | 0.2-0.5 seconds |
| Cached recommendation | <0.1 seconds |

---

## Support

**Issues?** Check:
1. `IMPLEMENTATION_SUMMARY.md` for detailed docs
2. `app/README.md` for app-specific help
3. Python version (needs 3.10+)
4. Data files in correct location

**Still stuck?** Open an issue on GitHub!

---

**🎉 Happy Recommending! 🎉**
