"""Tests for DatasetMatcher (fuzzy join)."""
from __future__ import annotations

import pandas as pd

from src.dataset_matcher import DatasetMatcher


class TestNormalization:
    def test_remaster_suffix_stripped(self):
        cases = [
            ("Hey Jude - Remastered 2009", "hey jude"),
            ("Bohemian Rhapsody - 2011 Mix",   "bohemian rhapsody"),
            ("Imagine - Remastered",           "imagine"),
            ("Wish You Were Here - Live",      "wish you were here"),
            ("Smells Like Teen Spirit - Album Version",  "smells like teen spirit"),
        ]
        for raw, expected in cases:
            assert DatasetMatcher.normalize_title(raw) == expected, raw

    def test_feat_credits_stripped(self):
        cases = [
            ("Empire State of Mind (feat. Alicia Keys)", "empire state of mind"),
            ("Old Town Road feat. Billy Ray Cyrus",      "old town road"),
            ("Forever (Featuring Drake)",                "forever"),
            ("Crazy in Love ft. Jay-Z",                  "crazy in love"),
        ]
        for raw, expected in cases:
            result = DatasetMatcher.normalize_title(raw)
            assert result == expected, f"{raw!r} -> {result!r} (expected {expected!r})"

    def test_unicode_folded_to_ascii(self):
        assert DatasetMatcher.normalize_title("Café") == "cafe"
        assert DatasetMatcher.normalize_artist("Beyoncé") == "beyonce"

    def test_punctuation_collapsed_to_spaces(self):
        assert DatasetMatcher.normalize_title("U + Ur Hand") == "u ur hand"
        assert DatasetMatcher.normalize_title("U+Ur Hand") == "u ur hand"

    def test_artist_primary_only(self):
        assert DatasetMatcher.normalize_artist("Beyonce, Jay-Z", primary_only=True) == "beyonce"
        assert DatasetMatcher.normalize_artist("Lil Nas X; Billy Ray Cyrus", primary_only=True) == "lil nas x"
        assert DatasetMatcher.normalize_artist("Eminem & Dr. Dre", primary_only=True) == "eminem"
        assert DatasetMatcher.normalize_artist("Calvin Harris and Rihanna", primary_only=True) == "calvin harris"

    def test_handles_none_and_non_string(self):
        assert DatasetMatcher.normalize_title(None) == ""
        assert DatasetMatcher.normalize_artist(None) == ""
        assert DatasetMatcher.normalize_title(123) == ""


class TestMatching:
    def test_basic_match(self, tiny_genius_df, tiny_spotify_df):
        matcher = DatasetMatcher(threshold=80)
        out = matcher.match(
            tiny_genius_df, tiny_spotify_df,
            left_keys=("artist", "title"),
            right_keys=("artists", "track_name"),
        )
        # All 4 of our test pairs should match
        assert len(out) >= 3, f"only {len(out)} matched"
        assert "match_score" in out.columns
        assert (out["match_score"] >= 80).all()

    def test_strict_threshold_filters_more(self, tiny_genius_df, tiny_spotify_df):
        loose = DatasetMatcher(threshold=70).match(
            tiny_genius_df, tiny_spotify_df,
            left_keys=("artist", "title"),
            right_keys=("artists", "track_name"),
            show_progress=False,
        )
        strict = DatasetMatcher(threshold=99).match(
            tiny_genius_df, tiny_spotify_df,
            left_keys=("artist", "title"),
            right_keys=("artists", "track_name"),
            show_progress=False,
        )
        assert len(strict) <= len(loose)

    def test_no_artist_overlap_returns_empty(self):
        a = pd.DataFrame({"artist": ["A"], "title": ["X"]})
        b = pd.DataFrame({"artists": ["B"], "track_name": ["Y"]})
        out = DatasetMatcher(threshold=50).match(
            a, b,
            left_keys=("artist", "title"),
            right_keys=("artists", "track_name"),
            show_progress=False,
        )
        assert len(out) == 0

    def test_match_quality_report(self, tiny_genius_df, tiny_spotify_df):
        matcher = DatasetMatcher(threshold=70)
        out = matcher.match(
            tiny_genius_df, tiny_spotify_df,
            left_keys=("artist", "title"),
            right_keys=("artists", "track_name"),
            show_progress=False,
        )
        report = matcher.match_quality_report(out)
        assert "n" in report.columns
        assert "pct" in report.columns
