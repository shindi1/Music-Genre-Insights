"""Load raw Kaggle datasets with sensible dtypes and column normalization.

Three datasets are supported:

    1. Genius Song Lyrics (carlosgdcj/genius-song-lyrics-with-language-information)
       File: song_lyrics.csv (~9 GB, ~5M rows)
       Columns: title, tag, artist, year, views, features, lyrics, id, language_cld3,
                language_ft, language

    2. Spotify Tracks Dataset (maharshipandya/-spotify-tracks-dataset)
       File: dataset.csv (114K rows)
       Columns: track_id, artists, album_name, track_name, popularity, duration_ms,
                explicit, danceability, energy, key, loudness, mode, speechiness,
                acousticness, instrumentalness, liveness, valence, tempo, time_signature,
                track_genre

    3. Spotify Songs 30K (solomonameh/spotify-music-dataset)
       Optional alternative — coarser genres but smaller and easier to start with.

The loader uses chunked reading for the Genius dataset because pandas will
choke on a 9GB CSV in 16GB of RAM if you ask it to read the whole thing at once.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import pandas as pd

from config import RAW_GENIUS_LYRICS, RAW_SPOTIFY_TRACKS
from src.utils import get_logger, memory_usage_mb, timer

log = get_logger(__name__)


# Memory-efficient dtypes for the Genius dataset
_GENIUS_DTYPES = {
    "title": "string",
    "tag": "category",
    "artist": "string",
    "year": "Int32",          # nullable int
    "views": "Int64",
    "features": "string",
    "lyrics": "string",
    "id": "Int64",
    "language_cld3": "category",
    "language_ft": "category",
    "language": "category",
}

_SPOTIFY_DTYPES = {
    "track_id": "string",
    "artists": "string",
    "album_name": "string",
    "track_name": "string",
    "popularity": "Int16",
    "duration_ms": "Int32",
    "explicit": "boolean",
    "danceability": "float32",
    "energy": "float32",
    "key": "Int8",
    "loudness": "float32",
    "mode": "Int8",
    "speechiness": "float32",
    "acousticness": "float32",
    "instrumentalness": "float32",
    "liveness": "float32",
    "valence": "float32",
    "tempo": "float32",
    "time_signature": "Int8",
    "track_genre": "category",
}


# --------------------------------------------------------------------------- #
# Genius                                                                      #
# --------------------------------------------------------------------------- #
def load_genius(
    path: Path = RAW_GENIUS_LYRICS,
    columns: Optional[List[str]] = None,
    chunksize: Optional[int] = None,
    nrows: Optional[int] = None,
    sample_n: Optional[int] = None,
    sample_seed: int = 42,
) -> pd.DataFrame:
    """Load the Genius Song Lyrics dataset.

    Args:
        path: path to song_lyrics.csv.
        columns: subset of columns to keep. Default keeps everything we need
            for downstream stages.
        chunksize: if given, read in chunks and concatenate. Useful for
            memory-constrained environments.
        nrows: read only first N rows (debugging).
        sample_n: take a random sample of N rows after loading.
            Cannot combine with chunksize.

    Returns:
        DataFrame.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Genius dataset not found at {path}. Download from Kaggle:\n"
            "  https://www.kaggle.com/datasets/carlosgdcj/"
            "genius-song-lyrics-with-language-information"
        )

    if columns is None:
        columns = ["title", "tag", "artist", "year", "lyrics", "language", "features"]

    use_dtypes = {c: _GENIUS_DTYPES[c] for c in columns if c in _GENIUS_DTYPES}

    with timer(f"loading Genius dataset from {path.name}"):
        if chunksize:
            chunks: Iterator[pd.DataFrame] = pd.read_csv(
                path,
                usecols=columns,
                dtype=use_dtypes,
                chunksize=chunksize,
                nrows=nrows,
                low_memory=False,
            )
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(
                path,
                usecols=columns,
                dtype=use_dtypes,
                nrows=nrows,
                low_memory=False,
            )

    log.info("loaded Genius: %d rows × %d cols (~%.0f MB)",
             len(df), df.shape[1], memory_usage_mb(df))

    if sample_n is not None and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=sample_seed).reset_index(drop=True)
        log.info("sampled to %d rows", len(df))

    return df


def load_genius_chunked(
    path: Path = RAW_GENIUS_LYRICS,
    chunksize: int = 200_000,
    columns: Optional[List[str]] = None,
) -> Iterator[pd.DataFrame]:
    """Stream the Genius dataset chunk-by-chunk. Use when memory is tight."""
    if columns is None:
        columns = ["title", "tag", "artist", "year", "lyrics", "language", "features"]
    use_dtypes = {c: _GENIUS_DTYPES[c] for c in columns if c in _GENIUS_DTYPES}

    return pd.read_csv(
        path, usecols=columns, dtype=use_dtypes, chunksize=chunksize, low_memory=False,
    )


# --------------------------------------------------------------------------- #
# Spotify                                                                     #
# --------------------------------------------------------------------------- #
def load_spotify(
    path: Path = RAW_SPOTIFY_TRACKS,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load the Spotify Tracks Dataset (114K rows, audio features + 125 genres)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Spotify dataset not found at {path}. Download from Kaggle:\n"
            "  https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset"
        )

    use_dtypes = (
        {c: _SPOTIFY_DTYPES[c] for c in columns if c in _SPOTIFY_DTYPES}
        if columns else _SPOTIFY_DTYPES
    )

    with timer(f"loading Spotify dataset from {path.name}"):
        df = pd.read_csv(
            path,
            usecols=columns,
            dtype=use_dtypes,
            low_memory=False,
        )

    # Drop the unnamed pandas index column some Kaggle exports include
    df = df.loc[:, ~df.columns.str.match(r"Unnamed")]

    log.info("loaded Spotify: %d rows × %d cols (~%.0f MB)",
             len(df), df.shape[1], memory_usage_mb(df))
    return df


# --------------------------------------------------------------------------- #
# Sanity checks                                                               #
# --------------------------------------------------------------------------- #
def validate_genius(df: pd.DataFrame) -> None:
    required = {"title", "artist", "lyrics", "tag"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Genius DataFrame missing columns: {missing}")


def validate_spotify(df: pd.DataFrame) -> None:
    required = {"artists", "track_name", "track_genre",
                "danceability", "energy", "valence", "tempo"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Spotify DataFrame missing columns: {missing}")
