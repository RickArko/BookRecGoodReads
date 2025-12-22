"""
Collaborative filtering model training using Implicit ALS.

Uses the sparse matrix prepared by prepare_data.py to train a recommendation model.
"""

import argparse
from pathlib import Path

from implicit.als import AlternatingLeastSquares
from loguru import logger
from scipy.sparse import load_npz


def load_prepared_data(data_dir: str = "data"):
    """Load the sparse matrix prepared by prepare_data.py.
    
    Args:
        data_dir: Directory containing prepared data files
        
    Returns:
        Sparse matrix for training
    """
    logger.info(f"Loading prepared sparse matrix from {data_dir}...")
    
    sparse_matrix_path = Path(data_dir) / "book_user_matrix_sparse.npz"
    
    if not sparse_matrix_path.exists():
        logger.error(f"Sparse matrix not found at {sparse_matrix_path}")
        logger.info("Please run prepare_data.py first")
        raise FileNotFoundError(f"Expected file: {sparse_matrix_path}")
    
    # Load sparse matrix (books x users)
    interactions_sparse = load_npz(sparse_matrix_path)
    logger.info(f"Loaded sparse matrix with shape: {interactions_sparse.shape}")
    logger.info(f"  Items (books): {interactions_sparse.shape[0]:,}")
    logger.info(f"  Users: {interactions_sparse.shape[1]:,}")
    logger.info(f"  Total interactions: {interactions_sparse.nnz:,}")
    logger.info(f"  Sparsity: {interactions_sparse.nnz / (interactions_sparse.shape[0] * interactions_sparse.shape[1]):.4f}")
    
    return interactions_sparse


def train_model(interactions, n_factors=32, n_iterations=10, regularization=0.01, alpha=40.0):
    """Train an ALS (Alternating Least Squares) collaborative filtering model.
    
    Args:
        interactions: Sparse matrix (items x users)
        n_factors: Number of latent factors
        n_iterations: Number of iterations
        regularization: L2 regularization
        alpha: Confidence scaling factor
        
    Returns:
        Trained model
    """
    logger.info("=" * 60)
    logger.info("Training ALS Collaborative Filtering Model")
    logger.info(f"Parameters: n_factors={n_factors}, n_iterations={n_iterations}")
    logger.info(f"  regularization={regularization}, alpha={alpha}")
    logger.info("=" * 60)
    
    model = AlternatingLeastSquares(
        factors=n_factors,
        iterations=n_iterations,
        regularization=regularization,
        alpha=alpha,
        use_gpu=False,
    )
    
    # Train on the transposed matrix (implicit expects user-item interactions)
    # Our matrix is books x users, so transpose to users x books
    model.fit(interactions.T.tocsr(), show_progress=True)
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)
    
    return model


def main(data_dir: str = "data", n_factors: int = 32, n_iterations: int = 10):
    """Main training pipeline.
    
    Args:
        data_dir: Directory containing prepared data
        n_factors: Number of latent factors
        n_iterations: Number of training iterations
    """
    logger.info("Starting ALS collaborative filtering training pipeline...")
    
    # Load prepared data
    interactions = load_prepared_data(data_dir)
    
    # Train model
    model = train_model(interactions, n_factors=n_factors, n_iterations=n_iterations)
    
    # Save model
    output_path = Path(data_dir) / "als_model.pkl"
    logger.info(f"Saving model to {output_path}...")
    model.save(str(output_path))
    logger.info(f"Model saved successfully!")
    
    logger.info(f"Model user factors shape: {model.user_factors.shape}")
    logger.info(f"Model item factors shape: {model.item_factors.shape}")
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a collaborative filtering model using ALS"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing prepared data (default: data)",
    )
    parser.add_argument(
        "--n_factors",
        type=int,
        default=32,
        help="Number of latent factors (default: 32)",
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=10,
        help="Number of training iterations (default: 10)",
    )
    
    args = parser.parse_args()
    
    main(
        data_dir=args.data_dir,
        n_factors=args.n_factors,
        n_iterations=args.n_iterations,
    )
