#!/bin/bash
# Quick setup script for macOS/Linux - handles everything in one go

set -e  # Exit on error

echo "Book Recommender - Complete Setup"
echo "=================================="

# Step 1: Install dependencies
echo ""
echo "Step 1: Installing dependencies..."
bash scripts/install/install.sh

if [ $? -ne 0 ]; then
    echo "ERROR: Installation failed"
    exit 1
fi

# Step 2: Prepare data
echo ""
echo "Step 2: Preparing data..."
bash scripts/install/prepare_data.sh

if [ $? -ne 0 ]; then
    echo "ERROR: Data preparation failed"
    exit 1
fi

echo ""
echo "=================================="
echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate the environment: source .venv/bin/activate"
echo "  2. Run the app: bash scripts/install/run_app.sh"
echo "     OR use Docker: bash scripts/install/docker.sh"
