#!/bin/bash
# Quick script to run the data preparation pipeline

echo "Starting data preparation..."
echo "================================"

# Step 1: Download data
echo "Step 1: Downloading data..."
uv run python src/downloader.py

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to download data"
    exit 1
fi

# Step 2: Serialize data
echo ""
echo "Step 2: Serializing data to Parquet..."
uv run python src/serialize_data.py --batch_size=500_000

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to serialize data"
    exit 1
fi

# Step 3: Prepare model data
echo ""
echo "Step 3: Preparing model data..."
uv run python prepare_data.py

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to prepare model data"
    exit 1
fi

echo ""
echo "================================"
echo "Data preparation complete!"
echo "You can now run: scripts/install/run_app.sh"
