#!/bin/bash
# Quick script to run the Streamlit app

echo "Starting Book Recommender App..."
echo "================================"

# Check if data files exist
if [ ! -f "data/book_user_matrix_sparse.npz" ]; then
    echo "ERROR: Required data files not found!"
    echo "Please run the data preparation steps first:"
    echo "  1. uv run python src/downloader.py"
    echo "  2. uv run python src/serialize_data.py --batch_size=500_000"
    echo "  3. uv run python prepare_data.py"
    echo ""
    exit 1
fi

# Run the app
echo "Launching Streamlit..."
uv run --group app streamlit run app/app.py
