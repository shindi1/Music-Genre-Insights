"""Genre normalization to a canonical 6-class taxonomy.

Spotify lists 1000+ micro-genres ("dance pop", "neo-soul", "outlaw country",
"future bass", etc). Genius uses ~5 broad tags ("pop", "rock", "rap", "rb",
"country"). We collapse both into a unified 6-class system:

    pop, hip-hop, rock, r&b, country, electronic

The mapping uses substring containment with longest-pattern-first ordering so
that "neo-soul" -> r&b not pop, and "country rap" -> country not hip-hop.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from config import (
    GENIUS_GENRE_MAP,
    SPOTIFY_GENRE_MAP,
    TARGET_GENRES,
)
from src.utils import get_logger

log = get_logger(__name__)


class GenreMapper:
    """Collapse arbitrary genre strings to one of the canonical TARGET_GENRES."""

    def __init__(
        self,
        spotify_map: Optional[Dict[str, List[str]]] = None,
        genius_map: Optional[Dict[str, str]] = None,
        target_genres: Iterable[str] = TARGET_GENRES,
    ) -> None:
        self.spotify_map = spotify_map or SPOTIFY_GENRE_MAP
        self.genius_map = genius_map or GENIUS_GENRE_MAP
        self.target_genres = tuple(target_genres)

        # Pre-compute a sorted list of (pattern, label) tuples, longest first.
        # Longest-first prevents "country" matching a "country rap" string and
        # winning over a more specific "country rap" pattern that would have
        # matched too. We sort by *pattern length* descending.
        self._spotify_patterns: List[Tuple[str, str]] = []
        for label, patterns in self.spotify_map.items():
            for p in patterns:
                self._spotify_patterns.append((p.lower(), label))
        self._spotify_patterns.sort(key=lambda x: -len(x[0]))

    # -------------------------------------------------------------- spotify
    def map_spotify_genre(self, raw: Optional[str]) -> Optional[str]:
        """Map a Spotify micro-genre string -> canonical label, or None."""
        if not isinstance(raw, str) or not raw:
            return None
        s = raw.lower().strip()
        for pattern, label in self._spotify_patterns:
            if pattern in s:
                return label
        return None

    def map_spotify_batch(self, values: Iterable[Optional[str]]) -> List[Optional[str]]:
        return [self.map_spotify_genre(v) for v in values]

    # --------------------------------------------------------------- genius
    def map_genius_tag(self, raw: Optional[str]) -> Optional[str]:
        """Map a Genius tag (already coarse) -> canonical label, or None.

        Genius tags are stored variously as 'rap', 'rb', 'r-b', 'pop', 'rock',
        'country'. Anything else (e.g. 'misc') becomes None.
        """
        if not isinstance(raw, str) or not raw:
            return None
        return self.genius_map.get(raw.lower().strip())

    def map_genius_batch(self, values: Iterable[Optional[str]]) -> List[Optional[str]]:
        return [self.map_genius_tag(v) for v in values]

    # ----------------------------------------------------------- DataFrame API
    def annotate(
        self,
        df: pd.DataFrame,
        source_col: str,
        out_col: str = "genre",
        source: str = "spotify",
        drop_unmapped: bool = True,
    ) -> pd.DataFrame:
        """Add canonical `genre` column. Optionally drop unmappable rows."""
        df = df.copy()
        if source == "spotify":
            df[out_col] = self.map_spotify_batch(df[source_col])
        elif source == "genius":
            df[out_col] = self.map_genius_batch(df[source_col])
        else:
            raise ValueError(f"unknown source: {source!r}")

        unmapped = df[out_col].isna().sum()
        log.info(
            "genre mapping: %d / %d rows mapped (%d unmapped)",
            len(df) - unmapped, len(df), unmapped,
        )
        if drop_unmapped:
            df = df.dropna(subset=[out_col]).reset_index(drop=True)

        return df

    def reconcile(self, spotify_label: Optional[str], genius_label: Optional[str]) -> Optional[str]:
        """When both sources provide a label, choose one.

        Strategy: Spotify is more specific & better curated, so it wins when
        present. Fall back to Genius only when Spotify is missing.
        """
        if spotify_label is not None:
            return spotify_label
        return genius_label

    # ----------------------------------------------------------- diagnostics
    def distribution_report(self, df: pd.DataFrame, col: str = "genre") -> pd.DataFrame:
        """Return a per-class count + percentage report."""
        counts = df[col].value_counts(dropna=False).rename("n").to_frame()
        counts["pct"] = (counts["n"] / counts["n"].sum() * 100).round(2)
        return counts
