"""Shared pytest fixtures and import-path setup."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Make `src` and `config` importable from tests
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def messy_lyrics() -> str:
    """Real-looking Genius-scraped lyric text with all the usual problems."""
    return (
        "23 Contributors\n"
        "Lose Yourself Lyrics\n"
        "[Intro]\n"
        "Look, if you had one shot, or one opportunity\n"
        "To seize everything you ever wanted (One moment)\n"
        "Would you capture it or just let it slip? (Yo)\n\n"
        "[Verse 1: Eminem]\n"
        "His palms are sweaty, knees weak, arms are heavy\n"
        "There&#x27;s vomit on his sweater already, mom&#x27;s spaghetti\n"
        "He&#39;s nervous, but on the surface he looks calm and ready\n\n"
        "Produced by Eminem & Jeff Bass\n"
        "You might also like\n"
        "23Embed"
    )


@pytest.fixture
def tiny_genius_df() -> pd.DataFrame:
    return pd.DataFrame({
        "title":   ["Hey Jude",        "Lose Yourself", "Old Town Road", "Beyoncé Song"],
        "artist":  ["The Beatles",     "Eminem",        "Lil Nas X",     "Beyoncé"],
        "tag":     ["pop",             "rap",           "country",       "pop"],
        "lyrics":  ["Hey Jude don't make it bad. Take a sad song.",
                    "His palms are sweaty, knees weak, arms are heavy.",
                    "Yeah I'm gonna take my horse to the old town road.",
                    "Drunk in love sometimes I think I'm crazy."] * 1,
        "language": ["en", "en", "en", "en"],
    })


@pytest.fixture
def tiny_spotify_df() -> pd.DataFrame:
    return pd.DataFrame({
        "track_name": ["Hey Jude - Remastered 2009",
                       "Lose Yourself - From '8 Mile' Soundtrack",
                       "Old Town Road",
                       "Drunk in Love"],
        "artists":    ["The Beatles",
                       "Eminem",
                       "Lil Nas X; Billy Ray Cyrus",
                       "Beyonce, Jay-Z"],
        "track_genre": ["pop", "hip hop", "country pop", "r&b"],
        "danceability": [0.5,  0.7,  0.8,  0.6],
        "energy":       [0.6,  0.9,  0.7,  0.5],
        "valence":      [0.7,  0.4,  0.8,  0.5],
        "tempo":        [73.0, 171.0, 136.0, 90.0],
        "loudness":     [-8.0, -5.0, -6.0, -7.0],
        "speechiness":  [0.03, 0.10, 0.05, 0.04],
        "acousticness": [0.5,  0.1,  0.3,  0.4],
        "instrumentalness": [0.0, 0.0, 0.0, 0.0],
        "liveness":     [0.1, 0.2, 0.1, 0.1],
    })
