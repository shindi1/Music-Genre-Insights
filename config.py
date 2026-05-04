"""Central configuration for the Cade data pipeline.

All tunable knobs live here. Keep this file pure (no side effects) so it can be
imported safely from anywhere — including notebooks and tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Paths                                                                       #
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # untouched downloads from Kaggle / APIs
INTERIM_DIR = DATA_DIR / "interim"  # partial / per-stage outputs
PROCESSED_DIR = DATA_DIR / "processed"  # final modeling-ready datasets
CACHE_DIR = DATA_DIR / "cache"      # API response cache
LOGS_DIR = PROJECT_ROOT / "logs"

# Expected raw filenames (drop your Kaggle downloads here with these names)
RAW_GENIUS_LYRICS = RAW_DIR / "song_lyrics.csv"          # carlosgdcj/genius-song-lyrics
RAW_SPOTIFY_TRACKS = RAW_DIR / "dataset.csv"             # maharshipandya/-spotify-tracks-dataset
RAW_SPOTIFY_30K_HIGH = RAW_DIR / "high_popularity_spotify_data.csv"
RAW_SPOTIFY_30K_LOW  = RAW_DIR / "low_popularity_spotify_data.csv"


# --------------------------------------------------------------------------- #
# Genre taxonomy                                                              #
# --------------------------------------------------------------------------- #
# We collapse Spotify's 1000+ micro-genres and Genius's noisy tags into a
# tractable 6-class taxonomy. This is the single source of truth — both the
# Spotify mapper and the Genius mapper consult the same canonical labels.
TARGET_GENRES: Tuple[str, ...] = (
    "pop",
    "hip-hop",
    "rock",
    "r&b",
    "country",
    "electronic",
)

# Spotify micro-genre -> canonical label.
# Order matters: more specific patterns should appear before more general ones,
# because the matcher uses first-match semantics on substring containment.
SPOTIFY_GENRE_MAP: Dict[str, List[str]] = {
    "hip-hop": [
        "hip hop", "hip-hop", "rap", "trap", "drill", "grime",
        "gangster rap", "conscious hip hop", "southern hip hop",
        "boom bap", "underground hip hop", "alternative hip hop",
    ],
    "r&b": [
        "r&b", "rnb", "soul", "neo soul", "neo-soul", "contemporary r&b",
        "funk", "motown", "quiet storm", "new jack swing",
    ],
    "country": [
        "country", "americana", "country rock", "bluegrass", "outlaw country",
        "country pop", "bro-country", "country rap", "honky tonk",
    ],
    "electronic": [
        "electronic", "edm", "house", "techno", "trance", "dubstep",
        "drum and bass", "drum-and-bass", "dnb", "ambient", "idm",
        "electro", "synthwave", "vaporwave", "garage", "future bass",
        "big room", "progressive house", "deep house",
    ],
    "rock": [
        "rock", "alternative", "indie rock", "punk", "grunge", "metal",
        "hard rock", "classic rock", "psychedelic", "post-rock", "emo",
        "garage rock", "shoegaze", "art rock", "folk rock",
    ],
    # pop is intentionally last — it's the most generic catch-all
    "pop": [
        "pop", "dance pop", "electropop", "synth-pop", "synthpop",
        "teen pop", "indie pop", "k-pop", "j-pop", "art pop",
        "dream pop", "power pop", "bubblegum",
    ],
}

# Genius tags are coarser already — a near-direct mapping.
GENIUS_GENRE_MAP: Dict[str, str] = {
    "pop": "pop",
    "rock": "rock",
    "rap": "hip-hop",
    "rb": "r&b",
    "r-b": "r&b",
    "country": "country",
    "electronic": "electronic",
}


# --------------------------------------------------------------------------- #
# Cleaning + filtering thresholds                                             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CleaningConfig:
    """Lyrics cleaning parameters. Frozen so configs can be hashed/cached."""
    remove_section_markers: bool = True   # [Chorus], [Verse 1: Eminem]
    remove_parentheticals: bool = False   # (yeah), (uh-huh) — usually keep
    remove_producer_tags: bool = True     # "Produced by Metro Boomin"
    fix_encoding: bool = True             # ftfy
    normalize_unicode: bool = True        # NFKC
    lowercase: bool = False               # let downstream tokenizer decide

    min_word_count: int = 50              # drop instrumentals / fragments
    max_word_count: int = 2000            # drop concatenation errors
    languages_to_keep: Tuple[str, ...] = ("en",)
    langdetect_seed: int = 42             # langdetect is stochastic without this


@dataclass(frozen=True)
class MatchingConfig:
    """Cross-dataset (Genius <-> Spotify) fuzzy matching parameters."""
    threshold: int = 87                   # 0-100; higher = stricter
    blocking_field: str = "artist_norm"   # only compare within same artist
    score_cutoff: int = 80                # rapidfuzz internal early-exit


@dataclass(frozen=True)
class BalancingConfig:
    """Class-balancing parameters."""
    strategy: str = "undersample"         # "undersample" | "none" | "cap"
    samples_per_class: int = 4000         # target if undersampling/capping
    min_samples_per_class: int = 500      # warn if any class falls below
    random_state: int = 42


@dataclass(frozen=True)
class SplitConfig:
    """Train/val/test split parameters."""
    test_size: float = 0.15
    val_size: float = 0.15                # of remaining after test split
    random_state: int = 42
    stratify_on: str = "genre"


@dataclass(frozen=True)
class APIConfig:
    """API rate limiting / retry parameters."""
    genius_sleep_seconds: float = 1.0     # polite scraping pace
    genius_timeout: int = 15
    spotify_batch_size: int = 50
    max_retries: int = 5
    backoff_base: float = 2.0             # exponential


@dataclass
class PipelineConfig:
    """Top-level config — pass one of these into the pipeline orchestrator."""
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    balancing: BalancingConfig = field(default_factory=BalancingConfig)
    splitting: SplitConfig = field(default_factory=SplitConfig)
    api: APIConfig = field(default_factory=APIConfig)

    # Convenience
    target_genres: Tuple[str, ...] = TARGET_GENRES


# Singleton default — most consumers should just `from config import DEFAULT`
DEFAULT = PipelineConfig()
