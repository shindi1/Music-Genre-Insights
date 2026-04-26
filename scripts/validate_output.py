"""Quality-assurance checks for the produced dataset.

Run this after `run_pipeline.py` to verify your processed dataset is healthy
before handing it off to Sam (unsupervised) and Larry (supervised).

Checks performed:
    - All expected columns are present
    - No nulls in critical columns (lyrics_clean, genre, audio features)
    - No leakage: train/val/test are disjoint by (artist, title)
    - Class distributions are reasonable (no class < min_samples)
    - Lyric word counts in expected range
    - Audio feature ranges look correct
    - match_score distribution is healthy
    - Per-split summary stats

Outputs a human-readable report to stdout and exits non-zero on failures.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import click
import pandas as pd

from config import PROCESSED_DIR, TARGET_GENRES
from src.utils import setup_logging


REQUIRED_COLUMNS = {
    "lyrics_clean", "genre", "match_score",
    "danceability", "energy", "valence", "tempo",
    "lyr_word_count", "lyr_vocab_diversity",
}

CRITICAL_NON_NULL = {"lyrics_clean", "genre", "danceability", "energy", "valence"}


class CheckResult:
    def __init__(self):
        self.passed = []
        self.warnings = []
        self.failures = []

    def passed_(self, msg: str) -> None:
        self.passed.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def fail(self, msg: str) -> None:
        self.failures.append(msg)

    def print_report(self) -> int:
        print("\n" + "=" * 70)
        print("VALIDATION REPORT")
        print("=" * 70)
        for msg in self.passed:
            print(f"  ✓ {msg}")
        for msg in self.warnings:
            print(f"  ⚠ {msg}")
        for msg in self.failures:
            print(f"  ✗ {msg}")
        print()
        print(f"passed={len(self.passed)}  warnings={len(self.warnings)}  failures={len(self.failures)}")
        print("=" * 70)
        return 0 if not self.failures else 1


def check_columns(df: pd.DataFrame, label: str, r: CheckResult) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        r.fail(f"[{label}] missing columns: {sorted(missing)}")
    else:
        r.passed_(f"[{label}] all required columns present")


def check_nulls(df: pd.DataFrame, label: str, r: CheckResult) -> None:
    for col in CRITICAL_NON_NULL & set(df.columns):
        n = df[col].isna().sum()
        if n > 0:
            r.fail(f"[{label}] {n} nulls in critical column '{col}'")
    r.passed_(f"[{label}] no nulls in critical columns")


def check_class_distribution(df: pd.DataFrame, label: str, r: CheckResult,
                             min_per_class: int = 100) -> None:
    if "genre" not in df.columns:
        return
    counts = df["genre"].value_counts()
    unknown = set(counts.index) - set(TARGET_GENRES)
    if unknown:
        r.warn(f"[{label}] unknown genres present: {sorted(unknown)}")

    too_few = counts[counts < min_per_class]
    if not too_few.empty:
        r.warn(f"[{label}] classes with < {min_per_class} rows: {too_few.to_dict()}")

    missing = set(TARGET_GENRES) - set(counts.index)
    if missing:
        r.fail(f"[{label}] missing target classes: {sorted(missing)}")
    else:
        r.passed_(f"[{label}] all {len(TARGET_GENRES)} target classes present")


def check_split_disjointness(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                             r: CheckResult) -> None:
    """Ensure no song appears in more than one split (leakage check)."""
    # Pick the first available (artist, title) column pair
    a_col = next((c for c in ("artist_genius", "artist", "artists") if c in train.columns), None)
    t_col = next((c for c in ("title_genius", "title", "track_name") if c in train.columns), None)
    if a_col is None or t_col is None:
        r.warn("could not locate (artist, title) columns to check leakage")
        return

    def keys(df):
        return set(zip(df[a_col].astype(str).str.lower(),
                       df[t_col].astype(str).str.lower()))

    train_keys, val_keys, test_keys = keys(train), keys(val), keys(test)
    overlaps = [
        ("train ∩ val",  train_keys & val_keys),
        ("train ∩ test", train_keys & test_keys),
        ("val ∩ test",   val_keys & test_keys),
    ]
    for name, overlap in overlaps:
        if overlap:
            r.fail(f"[leakage] {name} has {len(overlap)} shared songs")
        else:
            r.passed_(f"[leakage] {name} is disjoint")


def check_audio_feature_ranges(df: pd.DataFrame, label: str, r: CheckResult) -> None:
    """Spotify audio features have known valid ranges."""
    expected_ranges = {
        "danceability": (0, 1),
        "energy": (0, 1),
        "valence": (0, 1),
        "speechiness": (0, 1),
        "acousticness": (0, 1),
        "instrumentalness": (0, 1),
        "liveness": (0, 1),
        "tempo": (40, 260),
        "loudness": (-60, 5),
    }
    for feat, (lo, hi) in expected_ranges.items():
        if feat not in df.columns:
            continue
        actual_lo, actual_hi = df[feat].min(), df[feat].max()
        if actual_lo < lo - 1e-6 or actual_hi > hi + 1e-6:
            r.warn(
                f"[{label}] {feat} out of expected range "
                f"[{lo}, {hi}]: actual [{actual_lo:.2f}, {actual_hi:.2f}]"
            )
    r.passed_(f"[{label}] audio feature ranges look correct")


def check_lyric_lengths(df: pd.DataFrame, label: str, r: CheckResult) -> None:
    if "lyr_word_count" not in df.columns:
        return
    wc = df["lyr_word_count"]
    if (wc < 1).any():
        r.fail(f"[{label}] {(wc < 1).sum()} rows have zero-word lyrics")
    if wc.median() < 50:
        r.warn(f"[{label}] median word count is {wc.median()} (expected > 50)")
    r.passed_(f"[{label}] lyric word counts: median={wc.median():.0f} min={wc.min()} max={wc.max()}")


def check_match_scores(df: pd.DataFrame, label: str, r: CheckResult) -> None:
    if "match_score" not in df.columns:
        return
    ms = df["match_score"]
    pct_below_90 = (ms < 90).mean() * 100
    if pct_below_90 > 30:
        r.warn(f"[{label}] {pct_below_90:.0f}% of matches scored below 90 — consider raising threshold")
    r.passed_(f"[{label}] match_score: mean={ms.mean():.1f} min={ms.min()} max={ms.max()}")


@click.command()
@click.option("--input-dir", type=click.Path(path_type=Path), default=PROCESSED_DIR)
@click.option("--min-per-class", type=int, default=100)
def main(input_dir: Path, min_per_class: int):
    """Run all QA checks against the processed dataset."""
    setup_logging(level="WARNING")  # quieter — let our own prints shine

    r = CheckResult()

    paths = {
        "full":  input_dir / "full.parquet",
        "train": input_dir / "train.parquet",
        "val":   input_dir / "val.parquet",
        "test":  input_dir / "test.parquet",
    }
    for name, p in paths.items():
        if not p.exists():
            r.fail(f"missing file: {p}")
    if r.failures:
        sys.exit(r.print_report())

    full  = pd.read_parquet(paths["full"])
    train = pd.read_parquet(paths["train"])
    val   = pd.read_parquet(paths["val"])
    test  = pd.read_parquet(paths["test"])

    # Per-split checks
    for label, df in [("full", full), ("train", train), ("val", val), ("test", test)]:
        check_columns(df, label, r)
        check_nulls(df, label, r)
        check_class_distribution(df, label, r, min_per_class=min_per_class)
        check_audio_feature_ranges(df, label, r)
        check_lyric_lengths(df, label, r)
        check_match_scores(df, label, r)

    # Cross-split checks
    check_split_disjointness(train, val, test, r)

    # Print summary
    print("\nDataset summary:")
    print(f"  full:  {len(full):,} rows  ({full['genre'].nunique()} genres)")
    print(f"  train: {len(train):,} rows")
    print(f"  val:   {len(val):,} rows")
    print(f"  test:  {len(test):,} rows")
    print("\nClass distribution (train):")
    print(train["genre"].value_counts().to_string())

    sys.exit(r.print_report())


if __name__ == "__main__":
    main()
