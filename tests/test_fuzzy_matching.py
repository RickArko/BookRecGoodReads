"""Tests for the shared fuzzy title-matching helper in ``src.matching``.

The matcher is what bridges free-text user input to internal book indices,
so its threshold semantics, scorer choice, and tie-breaking matter for
both UX and the production app.
"""

from __future__ import annotations

import pytest

from src.knn_recommender_sparse import fuzzy_matching as legacy_fuzzy_matching
from src.matching import best_match_idx, fuzzy_title_matches


@pytest.fixture
def title_to_idx() -> dict[str, int]:
    return {
        "Harry Potter and the Sorcerer's Stone": 0,
        "Harry Potter and the Chamber of Secrets": 1,
        "The Lord of the Rings": 2,
        "1984": 3,
        "The Da Vinci Code": 4,
    }


class TestFuzzyTitleMatches:
    def test_exact_match_top(self, title_to_idx):
        matches = fuzzy_title_matches(title_to_idx, "1984")
        assert matches[0][0] == "1984"
        assert matches[0][2] == 100

    def test_partial_match_finds_full_title(self, title_to_idx):
        matches = fuzzy_title_matches(title_to_idx, "Harry Potter", threshold=80)
        # partial_ratio gives a perfect 100 for the substring match.
        titles = [t for t, _, _ in matches]
        assert any("Harry Potter" in t for t in titles)

    def test_threshold_excludes_low_scores(self, title_to_idx):
        # Quantum Field Theory has nothing in common with the book titles.
        assert fuzzy_title_matches(title_to_idx, "Quantum Field Theory", threshold=80) == []

    def test_threshold_above_100_returns_empty(self, title_to_idx):
        assert fuzzy_title_matches(title_to_idx, "1984", threshold=101) == []

    def test_max_results_trims(self, title_to_idx):
        matches = fuzzy_title_matches(title_to_idx, "the", threshold=10, max_results=2)
        assert len(matches) <= 2

    def test_case_insensitive(self, title_to_idx):
        matches = fuzzy_title_matches(title_to_idx, "the lord of the RINGS", threshold=80)
        assert matches[0][0] == "The Lord of the Rings"

    def test_empty_query_does_not_crash(self, title_to_idx):
        # Should fail safely, not raise.
        result = fuzzy_title_matches(title_to_idx, "", threshold=80)
        assert isinstance(result, list)

    def test_results_sorted_descending(self, title_to_idx):
        matches = fuzzy_title_matches(title_to_idx, "Harry", threshold=10)
        scores = [score for _, _, score in matches]
        assert scores == sorted(scores, reverse=True)

    def test_ratio_scorer_is_stricter_than_partial(self, title_to_idx):
        # "Harry Potter" should match the full HP titles via partial_ratio (substring)
        # but score lower via plain ratio (full-string difference is large).
        partial = fuzzy_title_matches(title_to_idx, "Harry Potter", threshold=0, scorer="partial_ratio")
        full = fuzzy_title_matches(title_to_idx, "Harry Potter", threshold=0, scorer="ratio")
        # Top score under partial_ratio is the perfect substring match (100).
        assert partial[0][2] >= full[0][2]

    def test_invalid_scorer_raises(self, title_to_idx):
        with pytest.raises(ValueError):
            fuzzy_title_matches(title_to_idx, "Harry", scorer="bogus")  # type: ignore[arg-type]


class TestBestMatchIdx:
    def test_returns_top_idx(self, title_to_idx):
        assert best_match_idx(title_to_idx, "1984") == 3

    def test_returns_none_when_below_threshold(self, title_to_idx):
        assert best_match_idx(title_to_idx, "Quantum Field Theory", threshold=80) is None

    def test_partial_match(self, title_to_idx):
        idx = best_match_idx(title_to_idx, "Da Vinci")
        assert idx == 4


class TestLegacyWrapperParity:
    """Pin the older ``knn_recommender_sparse.fuzzy_matching`` wrapper.

    Existing call sites (and the Streamlit app) rely on the wrapper's int-or-
    None contract and verbose printing. Make sure the consolidation hasn't
    silently changed its return value.
    """

    def test_exact_match_returns_idx(self, title_to_idx):
        assert legacy_fuzzy_matching(title_to_idx, "1984", verbose=False) == 3

    def test_no_match_returns_none(self, title_to_idx):
        assert legacy_fuzzy_matching(title_to_idx, "Quantum Field Theory", threshold=80, verbose=False) is None

    def test_partial_match_returns_idx(self, title_to_idx):
        idx = legacy_fuzzy_matching(title_to_idx, "Harry Potter", threshold=60, verbose=False)
        assert idx in (0, 1)
