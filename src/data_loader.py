"""Load raw datasets for the Cade pipeline.

Two datasets are used:

    1. Genius Song Lyrics (carlosgdcj/genius-song-lyrics-with-language-information)
       File: song_lyrics.csv (~9 GB, ~5M rows)
       Columns: title, tag, artist, year, views, features, lyrics, id, language_cld3,
                language_ft, language

    2. Spotify SQLite (spotify.sqlite)
       Normalized database: tracks, audio_features, artists, genres, and join tables.
       Loaded via load_spotify_sqlite(), which joins and caches to parquet.

The loader uses chunked reading for the Genius dataset because pandas will
choke on a 9GB CSV in 16GB of RAM if you ask it to read the whole thing at once.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import pandas as pd

import sqlite3

from config import RAW_GENIUS_LYRICS, RAW_SPOTIFY_SQLITE
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

# --------------------------------------------------------------------------- #
# Genius                                                                      #
# --------------------------------------------------------------------------- #
def load_genius(
    path: Path = RAW_GENIUS_LYRICS,
    columns: Optional[List[str]] = None,
    nrows: Optional[int] = None,
    sample_n: Optional[int] = None,
    sample_seed: int = 42,
) -> pd.DataFrame:
    """Load the Genius Song Lyrics dataset.

    Args:
        path: path to song_lyrics.csv.
        columns: subset of columns to keep. Default keeps everything we need
            for downstream stages.
        nrows: read only first N rows (debugging).
        sample_n: take a random sample of N rows after loading.

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


# --------------------------------------------------------------------------- #
# Spotify SQLite                                                              #
# --------------------------------------------------------------------------- #
_SQLITE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_tracks_id      ON tracks(id)",
    "CREATE INDEX IF NOT EXISTS idx_af_id           ON audio_features(id)",
    "CREATE INDEX IF NOT EXISTS idx_rta_track       ON r_track_artist(track_id)",
    "CREATE INDEX IF NOT EXISTS idx_rta_artist      ON r_track_artist(artist_id)",
    "CREATE INDEX IF NOT EXISTS idx_rag_artist      ON r_artist_genre(artist_id)",
    "CREATE INDEX IF NOT EXISTS idx_artists_id      ON artists(id)",
    "CREATE INDEX IF NOT EXISTS idx_tracks_pop      ON tracks(popularity)",
]

_SQLITE_QUERY = """
SELECT
    t.id                            AS track_id,
    t.name                          AS track_name,
    CAST(t.popularity AS INTEGER)   AS popularity,
    t.duration                      AS duration_ms,
    t.explicit,
    MIN(a.name)                     AS artists,
    af.danceability,
    af.energy,
    af.loudness,
    af.speechiness,
    af.acousticness,
    af.instrumentalness,
    af.liveness,
    af.valence,
    af.tempo,
    GROUP_CONCAT(DISTINCT rag.genre_id) AS _all_genres
FROM tracks t
JOIN audio_features  af  ON af.id        = t.id
JOIN r_track_artist  rta ON rta.track_id = t.id
JOIN artists         a   ON a.id         = rta.artist_id
JOIN r_artist_genre  rag ON rag.artist_id = rta.artist_id
WHERE t.popularity >= :min_pop
GROUP BY t.id
"""


def _ensure_sqlite_indexes(conn: sqlite3.Connection) -> None:
    """Create join indexes if they don't exist. No-op after first run."""
    cur = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name='idx_tracks_id'"
    )
    if cur.fetchone()[0] == 0:
        log.info("creating indexes on spotify.sqlite (one-time, ~1-3 min)…")
        for sql in _SQLITE_INDEXES:
            conn.execute(sql)
        conn.commit()
        log.info("indexes created")


def _resolve_genre(all_genres_str: Optional[str], patterns: list) -> Optional[str]:
    """Pick the first micro-genre from the comma-separated list that matches
    any canonical pattern. Returns the raw micro-genre string (not the label)
    so the downstream GenreMapper can do its normal mapping."""
    if not all_genres_str:
        return None
    for g in all_genres_str.split(","):
        g_lower = g.strip().lower()
        for pattern, _label in patterns:
            if pattern in g_lower:
                return g.strip()
    return None


def load_spotify_sqlite(
    path: Path = RAW_SPOTIFY_SQLITE,
    min_popularity: int = 10,
    cache_path: Optional[Path] = None,
    sample_n: Optional[int] = None,
    sample_seed: int = 42,
) -> pd.DataFrame:
    """Load Spotify data from the normalized SQLite database.

    Joins tracks + audio_features + artists + genres into a flat DataFrame
    with the same columns as load_spotify(), so the rest of the pipeline works
    unchanged.  Genre is resolved to the first micro-genre string that maps to
    one of the six canonical labels (GenreMapper handles the final mapping).

    The result is cached to a parquet file (default: data/interim/spotify_sqlite.parquet)
    so the expensive SQLite join only runs once.

    Args:
        path: path to spotify.sqlite.
        min_popularity: drop tracks with popularity < this (default 10).
        cache_path: where to store/read the cached parquet. None = auto.
        sample_n: random-sample N rows after loading.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Spotify SQLite not found at {path}.\n"
            "Expected data/raw/spotify.sqlite — check the file is in place."
        )

    if cache_path is None:
        from config import INTERIM_DIR
        cache_path = INTERIM_DIR / f"spotify_sqlite_pop{min_popularity}.parquet"

    if cache_path.exists():
        log.info("loading Spotify from cache: %s", cache_path)
        df = pd.read_parquet(cache_path)
        log.info("cached Spotify: %d rows × %d cols (~%.0f MB)",
                 len(df), df.shape[1], memory_usage_mb(df))
    else:
        conn = sqlite3.connect(path)
        # Replace invalid bytes instead of crashing on garbled track/artist names
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        _ensure_sqlite_indexes(conn)

        with timer(f"loading Spotify from {path.name}"):
            df = pd.read_sql_query(_SQLITE_QUERY, conn, params={"min_pop": min_popularity})
        conn.close()

        log.info("loaded Spotify SQLite: %d rows × %d cols (~%.0f MB)",
                 len(df), df.shape[1], memory_usage_mb(df))

        # Build sorted pattern list (longest first, same logic as GenreMapper)
        from config import SPOTIFY_GENRE_MAP
        patterns: list = []
        for label, pats in SPOTIFY_GENRE_MAP.items():
            for p in pats:
                patterns.append((p.lower(), label))
        patterns.sort(key=lambda x: -len(x[0]))

        df["track_genre"] = df["_all_genres"].map(
            lambda s: _resolve_genre(s, patterns)
        )
        df = df.drop(columns=["_all_genres"])

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        log.info("cached to %s", cache_path)

    if sample_n is not None and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=sample_seed).reset_index(drop=True)
        log.info("sampled to %d rows", len(df))

    return df
