"""Tests for the hybrid candidate-merging logic in ``evaluate.py``.

This is a pure function: given two candidate lists and weights, it must
produce a deterministic merged ranking. It encodes the *same* scoring
formula used by ``HybridRecommender.recommend_hybrid`` in production, so
correctness here protects both the eval and the live recommender from
silent drift.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing the root-level evaluate.py module from tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from evaluate import hybrid_recs_from_candidates


def test_pure_collab_weight_returns_collab_order():
    collab_recs = [[10, 20, 30]]
    collab_dists = [[0.1, 0.2, 0.3]]
    content_recs = [[40, 50, 60]]
    content_dists = [[0.1, 0.2, 0.3]]
    out = hybrid_recs_from_candidates(
        collab_recs, collab_dists, content_recs, content_dists, w_collab=1.0, w_content=0.0, k=3
    )
    assert out == [[10, 20, 30]]


def test_pure_content_weight_returns_content_order():
    collab_recs = [[10, 20, 30]]
    collab_dists = [[0.1, 0.2, 0.3]]
    content_recs = [[40, 50, 60]]
    content_dists = [[0.1, 0.2, 0.3]]
    out = hybrid_recs_from_candidates(
        collab_recs, collab_dists, content_recs, content_dists, w_collab=0.0, w_content=1.0, k=3
    )
    assert out == [[40, 50, 60]]


def test_overlapping_item_combines_scores():
    # Item 100 appears in both lists; its hybrid score is the sum of both
    # weighted (1 - dist) contributions and should beat any single-list item.
    collab_recs = [[100, 200]]
    collab_dists = [[0.5, 0.0]]  # 100 has weak collab (0.5), 200 has perfect collab (1.0)
    content_recs = [[100, 300]]
    content_dists = [[0.0, 0.0]]  # 100 has perfect content (1.0), 300 too
    out = hybrid_recs_from_candidates(
        collab_recs, collab_dists, content_recs, content_dists, w_collab=0.5, w_content=0.5, k=3
    )
    # 100: 0.5 * 0.5 + 0.5 * 1.0 = 0.75
    # 200: 0.5 * 1.0 + 0       = 0.50
    # 300: 0       + 0.5 * 1.0 = 0.50
    assert out[0] == [100, 200, 300] or out[0] == [100, 300, 200]
    assert out[0][0] == 100


def test_truncates_to_k():
    collab_recs = [[1, 2, 3, 4, 5]]
    collab_dists = [[0.0, 0.1, 0.2, 0.3, 0.4]]
    content_recs = [[6, 7, 8, 9, 10]]
    content_dists = [[0.0, 0.1, 0.2, 0.3, 0.4]]
    out = hybrid_recs_from_candidates(
        collab_recs, collab_dists, content_recs, content_dists, w_collab=0.5, w_content=0.5, k=3
    )
    assert len(out[0]) == 3


def test_handles_multiple_users_independently():
    # Per-user input lists must yield independent per-user outputs and stay aligned.
    collab_recs = [[1], [2]]
    collab_dists = [[0.0], [0.0]]
    content_recs = [[3], [4]]
    content_dists = [[0.0], [0.0]]
    out = hybrid_recs_from_candidates(
        collab_recs, collab_dists, content_recs, content_dists, w_collab=1.0, w_content=0.0, k=1
    )
    assert out == [[1], [2]]


def test_empty_content_falls_back_to_collab():
    # Books with no content features produce empty content rec lists; the
    # hybrid should still rank using the collab side.
    collab_recs = [[1, 2, 3]]
    collab_dists = [[0.0, 0.1, 0.2]]
    content_recs = [[]]
    content_dists = [[]]
    out = hybrid_recs_from_candidates(
        collab_recs, collab_dists, content_recs, content_dists, w_collab=0.6, w_content=0.4, k=3
    )
    assert out == [[1, 2, 3]]


def test_empty_both_returns_empty():
    out = hybrid_recs_from_candidates([[]], [[]], [[]], [[]], w_collab=0.5, w_content=0.5, k=3)
    assert out == [[]]


@pytest.mark.parametrize(
    "weights",
    [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)],
)
def test_score_is_monotone_in_distance(weights):
    """For a given weight, lower distance must produce a higher rank."""
    w_collab, w_content = weights
    collab_recs = [[1, 2]]
    collab_dists = [[0.1, 0.5]]  # item 1 is closer than item 2
    content_recs = [[]]
    content_dists = [[]]
    out = hybrid_recs_from_candidates(
        collab_recs, collab_dists, content_recs, content_dists, w_collab=w_collab, w_content=w_content, k=2
    )
    if w_collab > 0:
        assert out[0].index(1) < out[0].index(2)
