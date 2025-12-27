#!/bin/bash
# Script to run the Book Recommender app using Docker Compose

echo "Starting Book Recommender with Docker Compose..."
echo "================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running!"
    echo "Please start Docker Desktop and try again."
    exit 1
fi

# Build and run with docker-compose
docker-compose down
docker-compose up --build

echo "================================"
echo "App is running at: http://localhost:8501"
