#!/bin/bash
# Quick script to run the Streamlit app on macOS/Linux

echo "Starting Book Recommender App..."
echo "================================"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please run the installation script first: scripts/install/install.sh"
    exit 1
fi

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

# Activate virtual environment
source .venv/bin/activate

# Run the app
echo "Launching Streamlit..."
uv run --group app streamlit run app/app.py
