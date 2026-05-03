"""Tests for the leave-one-out holdout sampler in ``src.evaluation``.

The sampler is the boundary between the eval driver and the recommenders;
deterministic seeds and a clean min-interactions floor are load-bearing
properties for reproducible benchmarks.
"""

from __future__ import annotations

from scipy.sparse import csr_matrix

from src.evaluation import HoldoutSample, sample_holdouts


def _toy_books_users_matrix() -> csr_matrix:
    """5 books × 4 users with controlled interaction counts.

    User columns:
      user 0 → books 0, 1, 2, 3 (4 interactions)
      user 1 → books 0, 1       (2 interactions)
      user 2 → books 2, 3, 4    (3 interactions)
      user 3 → no interactions
    """
    rows = [0, 1, 2, 3, 0, 1, 2, 3, 4]
    cols = [0, 0, 0, 0, 1, 1, 2, 2, 2]
    data = [1.0] * len(rows)
    return csr_matrix((data, (rows, cols)), shape=(5, 4))


class TestSampleHoldouts:
    def test_returns_holdout_samples(self):
        samples = sample_holdouts(_toy_books_users_matrix(), n_users=10, min_interactions=2, seed=0)
        for s in samples:
            assert isinstance(s, HoldoutSample)

    def test_min_interactions_filter(self):
        # Only users 0, 1, 2 have ≥2 interactions; user 3 (empty) must be skipped.
        samples = sample_holdouts(_toy_books_users_matrix(), n_users=10, min_interactions=2, seed=0)
        assert {s.user_idx for s in samples} <= {0, 1, 2}

    def test_min_interactions_strict(self):
        # Only users 0 (4) and 2 (3) have ≥3 interactions.
        samples = sample_holdouts(_toy_books_users_matrix(), n_users=10, min_interactions=3, seed=0)
        assert {s.user_idx for s in samples} <= {0, 2}

    def test_seed_changes_user_pick(self):
        # Different seeds should generally select different users when the pool is small.
        m = _toy_books_users_matrix()
        a = sample_holdouts(m, n_users=1, min_interactions=2, seed=0)
        b = sample_holdouts(m, n_users=1, min_interactions=2, seed=999)
        # At least the seed/positives split should differ for a 1-user sample
        # across most seeds. We check the conjunction of (user, seed_book) is not
        # always identical — guards against accidentally constant sampling.
        assert (a[0].user_idx, a[0].seed_book_idx) != (b[0].user_idx, b[0].seed_book_idx) or a[0].positives != b[
            0
        ].positives

    def test_seed_is_deterministic(self):
        m = _toy_books_users_matrix()
        a = sample_holdouts(m, n_users=3, min_interactions=2, seed=42)
        b = sample_holdouts(m, n_users=3, min_interactions=2, seed=42)
        assert [(s.user_idx, s.seed_book_idx, s.positives) for s in a] == [
            (s.user_idx, s.seed_book_idx, s.positives) for s in b
        ]

    def test_seed_book_excluded_from_positives(self):
        samples = sample_holdouts(_toy_books_users_matrix(), n_users=10, min_interactions=2, seed=7)
        for s in samples:
            assert s.seed_book_idx not in s.positives

    def test_positives_are_actual_interactions(self):
        m = _toy_books_users_matrix()
        samples = sample_holdouts(m, n_users=10, min_interactions=2, seed=11)
        csc = m.tocsc()
        for s in samples:
            user_books = set(int(b) for b in csc.indices[csc.indptr[s.user_idx] : csc.indptr[s.user_idx + 1]])
            assert s.seed_book_idx in user_books
            assert set(s.positives) == user_books - {s.seed_book_idx}

    def test_n_positives_equals_interactions_minus_one(self):
        m = _toy_books_users_matrix()
        samples = sample_holdouts(m, n_users=10, min_interactions=2, seed=11)
        csc = m.tocsc()
        for s in samples:
            n_books = csc.indptr[s.user_idx + 1] - csc.indptr[s.user_idx]
            assert s.n_positives == n_books - 1

    def test_empty_when_no_eligible_users(self):
        # min_interactions=10 → nobody qualifies in the toy matrix.
        samples = sample_holdouts(_toy_books_users_matrix(), n_users=5, min_interactions=10, seed=0)
        assert samples == []

    def test_zero_users_returns_empty(self):
        samples = sample_holdouts(_toy_books_users_matrix(), n_users=0, min_interactions=1, seed=0)
        assert samples == []

    def test_caps_at_eligible_pool(self):
        # Asking for more users than exist returns the whole eligible pool.
        m = _toy_books_users_matrix()
        samples = sample_holdouts(m, n_users=100, min_interactions=2, seed=0)
        assert len(samples) == 3  # users 0, 1, 2

    def test_no_duplicate_users(self):
        m = _toy_books_users_matrix()
        samples = sample_holdouts(m, n_users=100, min_interactions=2, seed=0)
        ids = [s.user_idx for s in samples]
        assert len(ids) == len(set(ids))
