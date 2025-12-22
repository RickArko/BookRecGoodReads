@echo off
REM Quick script to run the Streamlit app on Windows

echo Starting Book Recommender App...
echo ================================

REM Check if data files exist
if not exist "data\book_user_matrix_sparse.npz" (
    echo ERROR: Required data files not found!
    echo Please run the data preparation steps first:
    echo   1. uv run python src/downloader.py
    echo   2. uv run python src/serialize_data.py --batch_size=500_000
    echo   3. uv run python prepare_data.py
    echo.
    exit /b 1
)

REM Run the app
echo Launching Streamlit...
uv run --group app streamlit run app/app.py
