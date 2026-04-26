"""Supplemental data collection from the Genius and Spotify APIs.

When to use this:
    - Filling in lyrics for songs that have audio features (Spotify) but no
      Genius dataset entry.
    - Pulling artist-level genre tags from Spotify when the track-level genre
      column is missing.

When NOT to use this:
    - As your primary data source. The Kaggle datasets contain millions of rows
      already preprocessed; live scraping is two orders of magnitude slower
      and may violate Genius's terms of service at scale.

⚠️ Spotify API caveats (current as of early 2026):
    The Audio Features, Recommendations, Related Artists, Audio Analysis, and
    track-preview-URL endpoints are all DEPRECATED for new developer apps as of
    November 2024. If you create a fresh Spotify dev account today, those
    endpoints will return 403 / 404. Only search, track metadata, artist
    metadata (including artist-level genre tags), and album metadata are
    reliably available.

    This module's `SpotifyClient` is intentionally limited to the
    still-functional endpoints. Don't try to use it for audio features —
    use the Kaggle CSV instead.

Configuration:
    Put credentials in `.env` at the project root:

        GENIUS_ACCESS_TOKEN=...
        SPOTIPY_CLIENT_ID=...
        SPOTIPY_CLIENT_SECRET=...

    Then `from dotenv import load_dotenv; load_dotenv()` is called for you.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable, List, Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils import get_logger

log = get_logger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover
    pass

try:
    import lyricsgenius
    _LYRICSGENIUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LYRICSGENIUS_AVAILABLE = False

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    _SPOTIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SPOTIPY_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Genius                                                                      #
# --------------------------------------------------------------------------- #
class GeniusClient:
    """Thin wrapper around lyricsgenius with caching, retries, polite delay."""

    def __init__(
        self,
        access_token: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        sleep_seconds: float = 1.0,
        timeout: int = 15,
    ) -> None:
        if not _LYRICSGENIUS_AVAILABLE:
            raise ImportError(
                "lyricsgenius not installed; run `pip install lyricsgenius`"
            )
        token = access_token or os.environ.get("GENIUS_ACCESS_TOKEN")
        if not token:
            raise ValueError(
                "Genius access token missing — set GENIUS_ACCESS_TOKEN in .env "
                "or pass access_token=..."
            )

        self._client = lyricsgenius.Genius(
            token,
            timeout=timeout,
            sleep_time=sleep_seconds,
            retries=3,
            verbose=False,
            remove_section_headers=False,  # keep them; cleaner removes them later
            skip_non_songs=True,
            excluded_terms=["(Remix)", "(Live)"],
        )
        self.sleep_seconds = sleep_seconds
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- cache
    def _cache_path(self, artist: str, title: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        # sanitize: spaces -> underscores, lowercase, strip punctuation
        safe = lambda s: "".join(c if c.isalnum() else "_" for c in s.lower())[:120]
        return self.cache_dir / f"{safe(artist)}__{safe(title)}.json"

    def _read_cache(self, p: Optional[Path]) -> Optional[dict]:
        if p and p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # pragma: no cover
                return None
        return None

    def _write_cache(self, p: Optional[Path], payload: dict) -> None:
        if p is None:
            return
        try:
            p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:  # pragma: no cover
            log.debug("cache write failed: %s", exc)

    # ---------------------------------------------------------------- fetch
    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=1, max=30),
        reraise=True,
    )
    def fetch_lyrics(self, artist: str, title: str) -> Optional[dict]:
        """Fetch a single song's lyrics + metadata. Returns None on miss."""
        cache_path = self._cache_path(artist, title)
        cached = self._read_cache(cache_path)
        if cached is not None:
            return cached

        try:
            song = self._client.search_song(title, artist)
        except Exception as exc:
            log.warning("Genius search failed for (%r, %r): %s", artist, title, exc)
            return None

        if song is None:
            self._write_cache(cache_path, {"artist": artist, "title": title, "miss": True})
            return None

        payload = {
            "artist": artist,
            "title": title,
            "genius_artist": song.artist,
            "genius_title": song.title,
            "lyrics": song.lyrics,
            "url": song.url,
            "id": song.id,
        }
        self._write_cache(cache_path, payload)
        return payload

    def fetch_batch(
        self,
        artist_title_pairs: Iterable[tuple],
        max_failures: int = 50,
    ) -> List[dict]:
        """Fetch many songs. Aborts after `max_failures` consecutive misses."""
        results: List[dict] = []
        consecutive_failures = 0
        for artist, title in artist_title_pairs:
            payload = self.fetch_lyrics(artist, title)
            if payload and not payload.get("miss"):
                results.append(payload)
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    log.error(
                        "%d consecutive failures — aborting batch. "
                        "Check API key and connectivity.",
                        max_failures,
                    )
                    break
            time.sleep(self.sleep_seconds)
        return results


# --------------------------------------------------------------------------- #
# Spotify                                                                     #
# --------------------------------------------------------------------------- #
class SpotifyClient:
    """Spotify Web API wrapper, scoped to endpoints still working post-2024.

    Use only for: search, track metadata, artist-level genre tags.
    DO NOT use for: audio features, recommendations, related artists.
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ) -> None:
        if not _SPOTIPY_AVAILABLE:
            raise ImportError("spotipy not installed; run `pip install spotipy`")

        cid = client_id or os.environ.get("SPOTIPY_CLIENT_ID")
        secret = client_secret or os.environ.get("SPOTIPY_CLIENT_SECRET")
        if not cid or not secret:
            raise ValueError(
                "Spotify credentials missing — set SPOTIPY_CLIENT_ID / "
                "SPOTIPY_CLIENT_SECRET in .env, or pass them explicitly."
            )

        auth = SpotifyClientCredentials(client_id=cid, client_secret=secret)
        self._client = spotipy.Spotify(client_credentials_manager=auth, requests_timeout=15)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=1, max=30),
        reraise=True,
    )
    def search_track(self, artist: str, title: str) -> Optional[dict]:
        query = f"track:{title} artist:{artist}"
        try:
            results = self._client.search(q=query, type="track", limit=1)
        except Exception as exc:
            log.warning("Spotify search failed for (%r, %r): %s", artist, title, exc)
            return None

        items = results.get("tracks", {}).get("items", [])
        if not items:
            return None
        track = items[0]
        return {
            "spotify_id": track["id"],
            "track_name": track["name"],
            "artists": [a["name"] for a in track["artists"]],
            "primary_artist_id": track["artists"][0]["id"] if track["artists"] else None,
            "popularity": track.get("popularity"),
            "duration_ms": track.get("duration_ms"),
            "explicit": track.get("explicit"),
            "album": track.get("album", {}).get("name"),
            "release_date": track.get("album", {}).get("release_date"),
        }

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=1, max=30),
        reraise=True,
    )
    def get_artist_genres(self, artist_id: str) -> List[str]:
        """Artist-level genre tags. Still functional in 2026."""
        try:
            artist = self._client.artist(artist_id)
        except Exception as exc:
            log.warning("artist lookup failed for %s: %s", artist_id, exc)
            return []
        return artist.get("genres", [])
