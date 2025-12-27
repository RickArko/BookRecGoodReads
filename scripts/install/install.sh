#!/bin/bash
# Quick script to Install Book Recommender on macOS/Linux

echo "Begin installation for Book Recommender..."
echo "================================"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is not installed!"
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Install dependencies and create virtual environment
uv sync --all-groups

# Activate virtual environment
source .venv/bin/activate

# Install Jupyter kernel
uv run python -m ipykernel install --user --name=BookRecommender --display-name "BookRecommender (uv)"

echo "================================"
echo "Installation complete!"
echo "Activate the environment with: source .venv/bin/activate"
