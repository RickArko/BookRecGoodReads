@echo off
REM Quick script to Install Book Recommender on Windows

echo Begin Windows installation for Book Recommender...
echo ================================

uv sync --all-groups
.venv\Scripts\activate
uv run python -m ipykernel install --user --name=BookRecommender --display-name "BookRecommender (uv)"

echo ================================
echo Installation complete!
echo Activate the environment with: .venv\Scripts\activate
