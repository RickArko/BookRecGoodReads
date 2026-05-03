"""Tests for the fuzzy title-matching helpers used by the recommenders.

The matcher is what bridges free-text user input to internal book indices,
so its threshold semantics and tie-breaking matter for both UX and the
production app.
"""

from __future__ import annotations

import pytest

from src.knn_recommender_sparse import fuzzy_matching


@pytest.fixture
def title_to_idx() -> dict[str, int]:
    return {
        "Harry Potter and the Sorcerer's Stone": 0,
        "Harry Potter and the Chamber of Secrets": 1,
        "The Lord of the Rings": 2,
        "1984": 3,
        "The Da Vinci Code": 4,
    }


class TestFuzzyMatching:
    def test_exact_match_wins(self, title_to_idx):
        assert fuzzy_matching(title_to_idx, "1984", verbose=False) == 3

    def test_partial_match_resolves_to_correct_book(self, title_to_idx):
        # "Harry Potter" should match the first Harry Potter book at high partial ratio.
        idx = fuzzy_matching(title_to_idx, "Harry Potter", threshold=60, verbose=False)
        assert idx in (0, 1)

    def test_typo_tolerated(self, title_to_idx):
        # Single-character typo should still match.
        assert fuzzy_matching(title_to_idx, "1894", threshold=70, verbose=False) == 3

    def test_no_match_returns_none(self, title_to_idx):
        assert fuzzy_matching(title_to_idx, "Quantum Field Theory", threshold=80, verbose=False) is None

    def test_threshold_above_100_returns_none(self, title_to_idx):
        # Impossible threshold; nothing scores 101.
        assert fuzzy_matching(title_to_idx, "1984", threshold=101, verbose=False) is None

    def test_case_insensitive(self, title_to_idx):
        idx = fuzzy_matching(title_to_idx, "the lord of the RINGS", threshold=80, verbose=False)
        assert idx == 2

    def test_empty_query_does_not_crash(self, title_to_idx):
        # Empty query should fail safely, not raise.
        result = fuzzy_matching(title_to_idx, "", threshold=80, verbose=False)
        assert result is None or isinstance(result, int)
