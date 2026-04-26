"""Tests for GenreMapper."""
from __future__ import annotations

import pandas as pd
import pytest

from src.genre_mapper import GenreMapper


class TestGenreMapper:
    def setup_method(self):
        self.mapper = GenreMapper()

    def test_spotify_pop(self):
        assert self.mapper.map_spotify_genre("pop") == "pop"
        assert self.mapper.map_spotify_genre("dance pop") == "pop"
        assert self.mapper.map_spotify_genre("synth-pop") == "pop"

    def test_spotify_hip_hop(self):
        assert self.mapper.map_spotify_genre("hip hop") == "hip-hop"
        assert self.mapper.map_spotify_genre("rap") == "hip-hop"
        assert self.mapper.map_spotify_genre("trap") == "hip-hop"
        assert self.mapper.map_spotify_genre("conscious hip hop") == "hip-hop"

    def test_spotify_rb(self):
        assert self.mapper.map_spotify_genre("r&b") == "r&b"
        assert self.mapper.map_spotify_genre("neo soul") == "r&b"
        assert self.mapper.map_spotify_genre("contemporary r&b") == "r&b"

    def test_spotify_country(self):
        assert self.mapper.map_spotify_genre("country") == "country"
        assert self.mapper.map_spotify_genre("country pop") == "country"
        # "country rap" — current mapping puts country first by length match
        # but either label is defensible. Just assert it's *some* canonical label.
        assert self.mapper.map_spotify_genre("country rap") in ("country", "hip-hop")

    def test_spotify_electronic(self):
        assert self.mapper.map_spotify_genre("electronic") == "electronic"
        assert self.mapper.map_spotify_genre("edm") == "electronic"
        assert self.mapper.map_spotify_genre("dubstep") == "electronic"
        assert self.mapper.map_spotify_genre("deep house") == "electronic"

    def test_spotify_rock(self):
        assert self.mapper.map_spotify_genre("rock") == "rock"
        assert self.mapper.map_spotify_genre("indie rock") == "rock"
        assert self.mapper.map_spotify_genre("metal") == "rock"

    def test_spotify_unmapped(self):
        # Things that genuinely don't fit our 6-class system
        assert self.mapper.map_spotify_genre("classical") is None
        assert self.mapper.map_spotify_genre("jazz") is None
        assert self.mapper.map_spotify_genre("comedy") is None

    def test_spotify_empty_input(self):
        assert self.mapper.map_spotify_genre("") is None
        assert self.mapper.map_spotify_genre(None) is None

    def test_genius_basic(self):
        assert self.mapper.map_genius_tag("pop") == "pop"
        assert self.mapper.map_genius_tag("rap") == "hip-hop"
        assert self.mapper.map_genius_tag("rb") == "r&b"
        assert self.mapper.map_genius_tag("rock") == "rock"
        assert self.mapper.map_genius_tag("country") == "country"

    def test_genius_unknown(self):
        assert self.mapper.map_genius_tag("misc") is None
        assert self.mapper.map_genius_tag("") is None

    def test_reconcile_prefers_spotify(self):
        assert self.mapper.reconcile("pop", "rock") == "pop"
        assert self.mapper.reconcile(None, "rock") == "rock"
        assert self.mapper.reconcile("pop", None) == "pop"
        assert self.mapper.reconcile(None, None) is None

    def test_annotate_dataframe_spotify(self):
        df = pd.DataFrame({"track_genre": ["pop", "rap", "classical", "hip hop"]})
        result = self.mapper.annotate(df, source_col="track_genre",
                                      out_col="genre", source="spotify",
                                      drop_unmapped=False)
        # pandas object Series turns None into NaN on .tolist(); compare with isna() guard
        genres = result["genre"].tolist()
        assert genres[0] == "pop"
        assert genres[1] == "hip-hop"
        assert pd.isna(genres[2])  # classical is unmapped
        assert genres[3] == "hip-hop"

    def test_annotate_dataframe_drop_unmapped(self):
        df = pd.DataFrame({"track_genre": ["pop", "classical", "rap"]})
        result = self.mapper.annotate(df, source_col="track_genre",
                                      out_col="genre", source="spotify",
                                      drop_unmapped=True)
        assert len(result) == 2
        assert "classical" not in result["track_genre"].values

    def test_annotate_invalid_source(self):
        df = pd.DataFrame({"x": ["pop"]})
        with pytest.raises(ValueError):
            self.mapper.annotate(df, source_col="x", source="invalid")

    def test_distribution_report(self):
        df = pd.DataFrame({"genre": ["pop", "pop", "rock", None]})
        rep = self.mapper.distribution_report(df)
        assert "n" in rep.columns
        assert "pct" in rep.columns
        assert rep.loc["pop", "n"] == 2
