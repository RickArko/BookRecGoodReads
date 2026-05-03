"""Unit tests for the pure ranking metrics in ``src.evaluation``.

These tests are intentionally exhaustive: the metrics drive the
benchmark report in ``docs/EVALUATION.md`` and downstream model
selection, so a regression here would silently corrupt every future
evaluation run.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.evaluation import (
    average_precision_at_k,
    catalog_coverage,
    hit_rate_at_k,
    intra_list_diversity,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k([1, 2, 3], {1, 2, 3}, 3) == pytest.approx(1.0)

    def test_none_relevant(self):
        assert precision_at_k([4, 5, 6], {1, 2, 3}, 3) == pytest.approx(0.0)

    def test_partial(self):
        assert precision_at_k([1, 4, 2, 5], {1, 2, 3}, 4) == pytest.approx(0.5)

    def test_truncates_to_k(self):
        # Only the first k=2 are inspected; trailing relevant items are ignored.
        assert precision_at_k([4, 5, 1, 2], {1, 2}, 2) == pytest.approx(0.0)

    def test_empty_recommendations(self):
        assert precision_at_k([], {1, 2, 3}, 5) == pytest.approx(0.0)

    def test_zero_k(self):
        assert precision_at_k([1, 2], {1, 2}, 0) == pytest.approx(0.0)


class TestRecallAtK:
    def test_all_captured(self):
        assert recall_at_k([1, 2, 3, 4], {1, 2, 3}, 4) == pytest.approx(1.0)

    def test_partial_capture(self):
        assert recall_at_k([1, 9, 9, 9], {1, 2, 3}, 4) == pytest.approx(1 / 3)

    def test_empty_relevant(self):
        # No ground truth, no penalty: returns 0 by convention.
        assert recall_at_k([1, 2, 3], set(), 3) == pytest.approx(0.0)

    def test_truncates_to_k(self):
        # k=2 cuts off before the relevant item at position 3.
        assert recall_at_k([9, 9, 1, 2], {1, 2}, 2) == pytest.approx(0.0)


class TestHitRateAtK:
    def test_hit(self):
        assert hit_rate_at_k([9, 9, 1], {1, 2}, 3) == pytest.approx(1.0)

    def test_miss(self):
        assert hit_rate_at_k([9, 9, 9], {1, 2}, 3) == pytest.approx(0.0)

    def test_truncates_to_k(self):
        # Relevant item at index 2 is outside top-k=2.
        assert hit_rate_at_k([9, 9, 1], {1}, 2) == pytest.approx(0.0)


class TestNdcgAtK:
    def test_perfect_ranking(self):
        # All k items are relevant; DCG == IDCG.
        assert ndcg_at_k([1, 2, 3], {1, 2, 3}, 3) == pytest.approx(1.0)

    def test_no_relevant(self):
        assert ndcg_at_k([4, 5, 6], {1, 2}, 3) == pytest.approx(0.0)

    def test_relevant_at_position_two(self):
        # 1 relevant item at rank 2; IDCG (1 relevant) = 1/log2(2) = 1.
        # DCG = 1/log2(3).
        result = ndcg_at_k([9, 1, 9], {1}, 3)
        assert result == pytest.approx(1.0 / math.log2(3))

    def test_more_relevant_than_k(self):
        # 5 relevant items but only k=3; IDCG normalizes against top-3 ideal.
        recs = [1, 2, 9]
        relevant = {1, 2, 3, 4, 5}
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3) + 1.0 / math.log2(4)
        dcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        assert ndcg_at_k(recs, relevant, 3) == pytest.approx(dcg / idcg)

    def test_empty_relevant_returns_zero(self):
        assert ndcg_at_k([1, 2], set(), 2) == pytest.approx(0.0)


class TestAveragePrecisionAtK:
    def test_perfect_ranking(self):
        # Hits at ranks 1, 2, 3 → AP = (1 + 1 + 1) / 3 = 1.0
        assert average_precision_at_k([1, 2, 3], {1, 2, 3}, 3) == pytest.approx(1.0)

    def test_known_value(self):
        # Hits at ranks 1 and 3 → AP@3 = (1/1 + 2/3) / min(|R|, k) = (1 + 0.667) / 2
        assert average_precision_at_k([1, 9, 2], {1, 2}, 3) == pytest.approx((1.0 + 2 / 3) / 2)

    def test_no_relevant_returns_zero(self):
        assert average_precision_at_k([4, 5, 6], {1, 2, 3}, 3) == pytest.approx(0.0)

    def test_empty_relevant_returns_zero(self):
        assert average_precision_at_k([1, 2], set(), 2) == pytest.approx(0.0)

    def test_late_hit_lower_score(self):
        # A late hit gives lower AP than an early hit (sanity check).
        late = average_precision_at_k([9, 9, 1], {1}, 3)
        early = average_precision_at_k([1, 9, 9], {1}, 3)
        assert early > late


class TestCatalogCoverage:
    def test_all_unique(self):
        assert catalog_coverage([[1, 2], [3, 4]], total_items=4) == pytest.approx(1.0)

    def test_overlap(self):
        # 3 unique items out of 10 catalog.
        assert catalog_coverage([[1, 2], [2, 3]], total_items=10) == pytest.approx(0.3)

    def test_zero_total_items_safe(self):
        assert catalog_coverage([[1, 2]], total_items=0) == pytest.approx(0.0)

    def test_empty_lists(self):
        assert catalog_coverage([], total_items=5) == pytest.approx(0.0)


class TestIntraListDiversity:
    def test_identical_items_zero_diversity(self):
        # Two identical feature rows → cosine sim 1 → distance 0.
        features = csr_matrix(np.array([[1.0, 0.0], [1.0, 0.0]]))
        assert intra_list_diversity([0, 1], features) == pytest.approx(0.0, abs=1e-9)

    def test_orthogonal_items_full_diversity(self):
        features = csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
        assert intra_list_diversity([0, 1], features) == pytest.approx(1.0)

    def test_single_item_returns_zero(self):
        features = csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
        assert intra_list_diversity([0], features) == pytest.approx(0.0)

    def test_zero_norm_rows_skipped(self):
        # Item 1 has zero norm; should be ignored. Only items 0 and 2 contribute.
        features = csr_matrix(np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]))
        assert intra_list_diversity([0, 1, 2], features) == pytest.approx(1.0)

    def test_distance_is_non_negative(self):
        # Random small matrix, run sanity check that result is in [0, 1].
        rng = np.random.default_rng(0)
        features = csr_matrix(rng.random((5, 4)))
        for size in (2, 3, 5):
            d = intra_list_diversity(list(range(size)), features)
            assert 0.0 <= d <= 1.0
