"""Offline evaluation utilities for the book recommenders.

Implements ranking metrics and a leave-one-out sampler over the
books × users sparse interaction matrix. The metric helpers are pure
functions; the sampler operates on a SciPy CSR matrix and is
deterministic given a seed.

Methodology
-----------
For each sampled user we pick one of their interacted books at random
as the *seed* query and treat the user's other interacted books as
held-out *positives*. The recommender ranks the catalog from the seed
and we score the top-K against the positives.

Caveat: the recommenders are fit on the full interaction matrix,
including held-out positives, so reported numbers are a small upper
bound. The driver in ``evaluate.py`` documents this in the report.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from scipy.sparse import csr_matrix

# ---------------------------------------------------------------------------
# Ranking metrics (pure functions, binary relevance)
# ---------------------------------------------------------------------------


def precision_at_k(recommended: list[int], relevant: set[int] | frozenset[int], k: int) -> float:
    if k <= 0 or not recommended:
        return 0.0
    top = recommended[:k]
    return sum(1 for item in top if item in relevant) / k


def recall_at_k(recommended: list[int], relevant: set[int] | frozenset[int], k: int) -> float:
    if not relevant:
        return 0.0
    top = recommended[:k]
    return sum(1 for item in top if item in relevant) / len(relevant)


def hit_rate_at_k(recommended: list[int], relevant: set[int] | frozenset[int], k: int) -> float:
    return 1.0 if any(item in relevant for item in recommended[:k]) else 0.0


def ndcg_at_k(recommended: list[int], relevant: set[int] | frozenset[int], k: int) -> float:
    """Normalized DCG with binary relevance."""
    top = recommended[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, item in enumerate(top) if item in relevant)
    n_relevant = min(len(relevant), k)
    if n_relevant == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))
    return dcg / idcg


def average_precision_at_k(recommended: list[int], relevant: set[int] | frozenset[int], k: int) -> float:
    """AP@K: averaged precision after each relevant hit in the top-K."""
    if not relevant:
        return 0.0
    hits = 0
    cumulative = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            hits += 1
            cumulative += hits / (i + 1)
    n_relevant = min(len(relevant), k)
    if n_relevant == 0:
        return 0.0
    return cumulative / n_relevant


# ---------------------------------------------------------------------------
# Coverage and diversity
# ---------------------------------------------------------------------------


def catalog_coverage(all_recommendations: Iterable[Iterable[int]], total_items: int) -> float:
    """Fraction of the catalog that appears in any rec list."""
    if total_items <= 0:
        return 0.0
    seen: set[int] = set()
    for recs in all_recommendations:
        seen.update(recs)
    return len(seen) / total_items


def intra_list_diversity(items: list[int], features: csr_matrix) -> float:
    """Mean pairwise cosine distance among rec items in feature space.

    Items with all-zero feature rows are skipped to avoid NaN cosines.
    Returns 0.0 if fewer than two items have features.
    """
    if len(items) < 2:
        return 0.0
    vectors = features[items]
    norms = np.sqrt(np.asarray(vectors.multiply(vectors).sum(axis=1)).ravel())
    valid = norms > 0
    if valid.sum() < 2:
        return 0.0
    vectors = vectors[valid]
    norms = norms[valid]
    sims = (vectors @ vectors.T).toarray() / np.outer(norms, norms)
    n = sims.shape[0]
    upper = sims[np.triu_indices(n, k=1)]
    return float(np.clip(1.0 - upper, 0.0, 1.0).mean())


# ---------------------------------------------------------------------------
# Holdout sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HoldoutSample:
    user_idx: int
    seed_book_idx: int
    positives: frozenset[int]

    @property
    def n_positives(self) -> int:
        return len(self.positives)


def sample_holdouts(
    matrix: csr_matrix,
    *,
    n_users: int,
    min_interactions: int = 5,
    seed: int = 42,
) -> list[HoldoutSample]:
    """Sample leave-one-out holdouts from a books × users sparse matrix.

    For each sampled user, pick one interacted book uniformly at random as the
    seed query and treat the rest of their interacted books as held-out
    positives.

    Args:
        matrix: Sparse books × users matrix with non-zero ratings.
        n_users: Number of users to sample.
        min_interactions: Skip users with fewer than this many books.
        seed: Deterministic RNG seed.

    Returns:
        Up to ``n_users`` HoldoutSample records. Fewer if the matrix has
        too few eligible users.
    """
    if n_users <= 0:
        return []
    rng = np.random.default_rng(seed)
    csc = matrix.tocsc()
    nnz_per_user = np.diff(csc.indptr)
    eligible = np.where(nnz_per_user >= min_interactions)[0]
    if eligible.size == 0:
        return []
    n = min(n_users, eligible.size)
    sampled = rng.choice(eligible, size=n, replace=False)

    samples: list[HoldoutSample] = []
    for user_idx in sampled:
        start = csc.indptr[user_idx]
        end = csc.indptr[user_idx + 1]
        books = csc.indices[start:end]
        seed_pos = int(rng.integers(0, len(books)))
        seed_book = int(books[seed_pos])
        positives = frozenset(int(b) for b in books if int(b) != seed_book)
        samples.append(HoldoutSample(int(user_idx), seed_book, positives))
    return samples


# ---------------------------------------------------------------------------
# Recommender protocol + evaluation driver
# ---------------------------------------------------------------------------


class TopKRecommender(Protocol):
    """Minimal contract every recommender must satisfy for evaluation."""

    name: str

    def top_k(self, seed_book_idx: int, k: int) -> list[int]: ...


def evaluate_recommender(
    recommender: TopKRecommender,
    samples: list[HoldoutSample],
    *,
    k_values: list[int],
    n_total_books: int,
    diversity_fn: Callable[[list[int]], float] | None = None,
) -> dict[str, float]:
    """Run a recommender over the samples and return aggregate metrics.

    The recommender's ``top_k`` is invoked once per sample with
    ``k = max(k_values)``; per-K metrics are computed by truncation.
    """
    if not samples:
        raise ValueError("samples is empty")
    if not k_values:
        raise ValueError("k_values must be non-empty")

    max_k = max(k_values)
    metric_buckets: dict[str, list[float]] = {}
    all_recs: list[list[int]] = []
    diversities: list[float] = []

    for sample in samples:
        recs = recommender.top_k(sample.seed_book_idx, max_k)
        all_recs.append(recs)
        if diversity_fn is not None:
            diversities.append(diversity_fn(recs))
        for k in k_values:
            metric_buckets.setdefault(f"precision@{k}", []).append(precision_at_k(recs, sample.positives, k))
            metric_buckets.setdefault(f"recall@{k}", []).append(recall_at_k(recs, sample.positives, k))
            metric_buckets.setdefault(f"ndcg@{k}", []).append(ndcg_at_k(recs, sample.positives, k))
            metric_buckets.setdefault(f"map@{k}", []).append(average_precision_at_k(recs, sample.positives, k))
            metric_buckets.setdefault(f"hit_rate@{k}", []).append(hit_rate_at_k(recs, sample.positives, k))

    summary: dict[str, float] = {key: float(np.mean(values)) for key, values in metric_buckets.items()}
    summary["catalog_coverage"] = catalog_coverage(all_recs, n_total_books)
    summary["mean_n_positives"] = float(np.mean([s.n_positives for s in samples]))
    summary["n_samples"] = float(len(samples))
    if diversity_fn is not None:
        summary["intra_list_diversity"] = float(np.mean(diversities)) if diversities else 0.0
    return summary
