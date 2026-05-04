"""Central configuration for the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Paths                                                                       #
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DIR      = DATA_DIR / "raw"
INTERIM_DIR  = DATA_DIR / "interim"
PROCESSED_DIR       = DATA_DIR / "processed"
PROCESSED_AUDIO_DIR  = PROCESSED_DIR / "audio"
PROCESSED_LYRICS_DIR = PROCESSED_DIR / "lyrics"
LOGS_DIR     = PROJECT_ROOT / "logs"

RAW_GENIUS_LYRICS  = RAW_DIR / "song_lyrics.csv"
RAW_SPOTIFY_SQLITE = RAW_DIR / "spotify.sqlite"


# --------------------------------------------------------------------------- #
# Genre taxonomy                                                              #
# --------------------------------------------------------------------------- #
TARGET_GENRES: Tuple[str, ...] = (
    "pop", "hip-hop", "rock", "r&b", "country", "electronic",
)

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
    "pop": [
        "pop", "dance pop", "electropop", "synth-pop", "synthpop",
        "teen pop", "indie pop", "k-pop", "j-pop", "art pop",
        "dream pop", "power pop", "bubblegum",
    ],
}

GENIUS_GENRE_MAP: Dict[str, str] = {
    "pop":       "pop",
    "rock":      "rock",
    "rap":       "hip-hop",
    "rb":        "r&b",
    "r-b":       "r&b",
    "country":   "country",
    "electronic":"electronic",
}


# --------------------------------------------------------------------------- #
# Pipeline configs                                                            #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CleaningConfig:
    remove_section_markers: bool = True
    remove_parentheticals: bool = False
    remove_producer_tags: bool = True
    fix_encoding: bool = True
    normalize_unicode: bool = True
    lowercase: bool = False
    min_word_count: int = 50
    max_word_count: int = 2000
    languages_to_keep: Tuple[str, ...] = ("en",)
    langdetect_seed: int = 42


@dataclass(frozen=True)
class BalancingConfig:
    strategy: str = "undersample"
    samples_per_class: int = 4000
    min_samples_per_class: int = 500
    random_state: int = 42


@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42
    stratify_on: str = "genre"


@dataclass
class PipelineConfig:
    mode: str = "audio"          # "audio" | "lyrics"
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    balancing: BalancingConfig = field(default_factory=BalancingConfig)
    splitting: SplitConfig = field(default_factory=SplitConfig)
    target_genres: Tuple[str, ...] = TARGET_GENRES


DEFAULT = PipelineConfig()
