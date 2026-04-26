"""CLI for supplementing the Kaggle datasets with API-scraped data.

Use this when the matched dataset has fewer rows in some genre than you want
and you'd like to scrape lyrics for additional Spotify tracks (audio features
already known) from Genius.

Example:
    # For each track in spotify_tracks.csv that doesn't appear in
    # data/processed/full.parquet, fetch lyrics from Genius and append.
    python scripts/scrape_supplemental.py \\
        --target-genre country \\
        --max-fetch 500 \\
        --out data/interim/supplemental_country.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import click
import pandas as pd

from config import CACHE_DIR, INTERIM_DIR, PROCESSED_DIR
from src.api_scraper import GeniusClient
from src.data_loader import load_spotify
from src.dataset_matcher import DatasetMatcher
from src.genre_mapper import GenreMapper
from src.utils import safe_to_parquet, setup_logging


@click.command(context_settings=dict(show_default=True))
@click.option("--target-genre", type=str, required=True,
              help="One of pop / hip-hop / rock / r&b / country / electronic")
@click.option("--max-fetch", type=int, default=500,
              help="Max Genius API calls to make.")
@click.option("--existing", type=click.Path(path_type=Path),
              default=PROCESSED_DIR / "full.parquet",
              help="Already-matched dataset (skip these tracks).")
@click.option("--out", type=click.Path(path_type=Path),
              default=INTERIM_DIR / "supplemental.parquet")
@click.option("--cache-dir", type=click.Path(path_type=Path), default=CACHE_DIR / "genius",
              help="Where to cache Genius API responses.")
@click.option("--sleep", type=float, default=1.0,
              help="Seconds between Genius API calls (be polite).")
def main(target_genre, max_fetch, existing, out, cache_dir, sleep):
    """Scrape additional lyrics for an underrepresented genre."""
    log = setup_logging(level="INFO", log_file=PROJECT_ROOT / "logs" / "scrape.log")
    log.info("supplemental scrape: genre=%s max_fetch=%d", target_genre, max_fetch)

    # 1. Load Spotify tracks for the target genre
    spotify = load_spotify()
    mapper = GenreMapper()
    spotify = mapper.annotate(spotify, source_col="track_genre",
                              out_col="genre", source="spotify",
                              drop_unmapped=True)
    spotify = spotify[spotify["genre"] == target_genre].reset_index(drop=True)
    log.info("%d Spotify tracks in genre=%s", len(spotify), target_genre)

    # 2. Subtract those already covered by the existing matched dataset
    if existing.exists():
        already = pd.read_parquet(existing)
        # Use whichever artist+title columns are present
        a_col = "artists" if "artists" in already.columns else "artist_spotify"
        t_col = "track_name" if "track_name" in already.columns else "title_spotify"
        if a_col in already.columns and t_col in already.columns:
            covered = set(zip(
                [DatasetMatcher.normalize_artist(a) for a in already[a_col]],
                [DatasetMatcher.normalize_title(t) for t in already[t_col]],
            ))
            spotify["_an"] = [DatasetMatcher.normalize_artist(a) for a in spotify["artists"]]
            spotify["_tn"] = [DatasetMatcher.normalize_title(t) for t in spotify["track_name"]]
            mask = ~spotify.apply(lambda r: (r["_an"], r["_tn"]) in covered, axis=1)
            spotify = spotify[mask].drop(columns=["_an", "_tn"]).reset_index(drop=True)
            log.info("after removing already-covered tracks: %d remaining", len(spotify))

    # 3. Sample down to max_fetch (prioritize popular tracks)
    if "popularity" in spotify.columns:
        spotify = spotify.sort_values("popularity", ascending=False).head(max_fetch)
    else:
        spotify = spotify.head(max_fetch)
    log.info("scraping %d tracks from Genius", len(spotify))

    # 4. Fetch lyrics
    cache_dir.mkdir(parents=True, exist_ok=True)
    client = GeniusClient(cache_dir=cache_dir, sleep_seconds=sleep)
    pairs = list(zip(spotify["artists"], spotify["track_name"]))
    fetched = client.fetch_batch(pairs, max_failures=50)
    log.info("fetched %d / %d lyrics successfully", len(fetched), len(pairs))

    if not fetched:
        log.warning("no lyrics fetched — exiting without writing output")
        return

    fetched_df = pd.DataFrame(fetched)
    # Join back to the spotify rows to get audio features
    fetched_df["_an"] = [DatasetMatcher.normalize_artist(a) for a in fetched_df["artist"]]
    fetched_df["_tn"] = [DatasetMatcher.normalize_title(t) for t in fetched_df["title"]]
    spotify["_an"] = [DatasetMatcher.normalize_artist(a) for a in spotify["artists"]]
    spotify["_tn"] = [DatasetMatcher.normalize_title(t) for t in spotify["track_name"]]
    merged = fetched_df.merge(spotify, on=["_an", "_tn"], how="left").drop(columns=["_an", "_tn"])

    out.parent.mkdir(parents=True, exist_ok=True)
    safe_to_parquet(merged, out)
    log.info("✓ wrote %d rows to %s", len(merged), out)


if __name__ == "__main__":
    main()
