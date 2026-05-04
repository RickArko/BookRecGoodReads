"""Regression tests for ``prepare_data.active_filter_optimized``.

The active filter trims the interactions LazyFrame down to (active user,
popular book) pairs before the rest of the pipeline runs. A previous
revision used ``count > -min_reads`` for the popular-book predicate,
which is always true for non-negative counts and silently disabled the
book-side filter. This test pins the corrected ``>= min_reads`` behavior
on both sides.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

# Allow importing prepare_data.py from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_data import active_filter_optimized


def _build_interactions_lf(rows: list[tuple[int, int, int]]) -> pl.LazyFrame:
    """Build a LazyFrame of (user_id, book_id, rating) rows."""
    return pl.DataFrame(
        {
            "user_id": [r[0] for r in rows],
            "book_id": [r[1] for r in rows],
            "rating": [r[2] for r in rows],
        }
    ).lazy()


class TestActiveFilterOptimized:
    def test_drops_inactive_users(self):
        # User 1 has 3 interactions, user 2 has only 1.
        rows = [(1, 100, 5), (1, 101, 4), (1, 102, 3), (2, 100, 5)]
        out = active_filter_optimized(_build_interactions_lf(rows), min_reads=2).collect()
        assert set(out["user_id"].to_list()) == {1}

    def test_drops_unpopular_books(self):
        """Regression test for the original ``count > -min_reads`` bug.

        Before the fix, every book passed the filter regardless of count, so a
        book with a single interaction (book 999) would still appear in the
        output. After the fix, that book must be dropped.
        """
        rows = [
            (1, 100, 5),
            (2, 100, 5),
            (3, 100, 5),
            (1, 101, 5),
            (2, 101, 5),
            (3, 101, 5),
            (1, 999, 5),  # book 999 has only 1 interaction; should be filtered out
        ]
        out = active_filter_optimized(_build_interactions_lf(rows), min_reads=2).collect()
        assert 999 not in out["book_id"].to_list()
        assert set(out["book_id"].to_list()) == {100, 101}

    def test_keeps_rows_meeting_both_filters(self):
        rows = [
            (1, 100, 5),
            (1, 101, 4),
            (2, 100, 3),
            (2, 101, 2),
            (3, 100, 5),
            (3, 102, 4),  # book 102 has only 1 interaction → filtered out
        ]
        out = active_filter_optimized(_build_interactions_lf(rows), min_reads=2).collect()
        retained_books = set(out["book_id"].to_list())
        retained_users = set(out["user_id"].to_list())
        assert 102 not in retained_books
        # All three users hit ≥2 interactions, but user 3 only has one of those
        # in a popular book (100) after 102 is filtered, so user 3 may or may
        # not survive depending on whether the user filter is re-applied. The
        # current implementation applies user and book filters independently,
        # so user 3's row for book 100 should still be retained.
        assert 100 in retained_books
        assert 101 in retained_books
        # Users 1, 2 are clearly active; check at minimum they survive.
        assert {1, 2} <= retained_users

    def test_min_reads_zero_passes_everything(self):
        rows = [(1, 100, 5), (2, 101, 5)]
        out = active_filter_optimized(_build_interactions_lf(rows), min_reads=0).collect()
        assert len(out) == 2

    @pytest.mark.parametrize("min_reads", [1, 2, 5])
    def test_threshold_is_inclusive(self, min_reads):
        # Build exactly ``min_reads`` interactions for one user/book pair.
        rows = [(1, 100, 5)] * min_reads
        out = active_filter_optimized(_build_interactions_lf(rows), min_reads=min_reads).collect()
        assert len(out) == min_reads
        assert set(out["user_id"].to_list()) == {1}
        assert set(out["book_id"].to_list()) == {100}
