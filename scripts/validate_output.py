"""QA checks for processed datasets. Works in both audio and lyrics mode."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import click
import pandas as pd

from config import PROCESSED_DIR, TARGET_GENRES
from src.utils import setup_logging

AUDIO_COLS = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]
AUDIO_RANGES = {
    "danceability": (0, 1), "energy": (0, 1), "valence": (0, 1),
    "speechiness": (0, 1), "acousticness": (0, 1), "instrumentalness": (0, 1),
    "liveness": (0, 1), "tempo": (40, 260), "loudness": (-60, 5),
}


class Report:
    def __init__(self):
        self.passed = []
        self.warnings = []
        self.failures = []

    def ok(self, msg):   self.passed.append(msg)
    def warn(self, msg): self.warnings.append(msg)
    def fail(self, msg): self.failures.append(msg)

    def print(self) -> int:
        print("\n" + "=" * 70)
        print("VALIDATION REPORT")
        print("=" * 70)
        for m in self.passed:   print(f"  ✓ {m}")
        for m in self.warnings: print(f"  ⚠ {m}")
        for m in self.failures: print(f"  ✗ {m}")
        print(f"\npassed={len(self.passed)}  warnings={len(self.warnings)}  failures={len(self.failures)}")
        return 0 if not self.failures else 1


def detect_mode(df: pd.DataFrame) -> str:
    return "lyrics" if "lyrics_clean" in df.columns else "audio"


def check_genre(df: pd.DataFrame, label: str, r: Report, min_per_class: int) -> None:
    if "genre" not in df.columns:
        r.fail(f"[{label}] missing 'genre' column")
        return
    counts = df["genre"].value_counts()
    missing = set(TARGET_GENRES) - set(counts.index)
    if missing:
        r.fail(f"[{label}] missing target classes: {sorted(missing)}")
    else:
        r.ok(f"[{label}] all {len(TARGET_GENRES)} target classes present")
    too_few = counts[counts < min_per_class]
    if not too_few.empty:
        r.warn(f"[{label}] classes with < {min_per_class} rows: {too_few.to_dict()}")


def check_nulls(df: pd.DataFrame, label: str, r: Report, mode: str) -> None:
    critical = ["genre"]
    if mode == "audio":
        critical += [c for c in AUDIO_COLS if c in df.columns]
    else:
        critical += ["lyrics_clean"]
    bad = [c for c in critical if df[c].isna().any()]
    if bad:
        r.fail(f"[{label}] nulls in: {bad}")
    else:
        r.ok(f"[{label}] no nulls in critical columns")


def check_audio_ranges(df: pd.DataFrame, label: str, r: Report) -> None:
    if not any(c in df.columns for c in AUDIO_COLS):
        return
    for feat, (lo, hi) in AUDIO_RANGES.items():
        if feat not in df.columns:
            continue
        lo_act, hi_act = df[feat].min(), df[feat].max()
        if lo_act < lo - 1e-6 or hi_act > hi + 1e-6:
            r.warn(f"[{label}] {feat} out of [{lo}, {hi}]: actual [{lo_act:.2f}, {hi_act:.2f}]")
    r.ok(f"[{label}] audio feature ranges look correct")


def check_lyric_lengths(df: pd.DataFrame, label: str, r: Report) -> None:
    col = "lyr_word_count" if "lyr_word_count" in df.columns else (
          "word_count" if "word_count" in df.columns else None)
    if col is None:
        return
    wc = df[col]
    r.ok(f"[{label}] lyric word counts: median={wc.median():.0f} min={wc.min()} max={wc.max()}")


def check_disjoint(train, val, test, r: Report) -> None:
    a_col = next((c for c in ("artist", "artists") if c in train.columns), None)
    t_col = next((c for c in ("title", "track_name") if c in train.columns), None)
    if not a_col or not t_col:
        r.warn("could not find (artist, title) columns for leakage check")
        return
    def keys(df):
        return set(zip(df[a_col].astype(str).str.lower(), df[t_col].astype(str).str.lower()))
    tk, vk, ek = keys(train), keys(val), keys(test)
    for name, overlap in [("train ∩ val", tk & vk), ("train ∩ test", tk & ek), ("val ∩ test", vk & ek)]:
        if overlap:
            r.fail(f"[leakage] {name} has {len(overlap)} shared songs")
        else:
            r.ok(f"[leakage] {name} is disjoint")


@click.command()
@click.option("--input-dir", type=click.Path(path_type=Path), default=PROCESSED_DIR)
@click.option("--min-per-class", type=int, default=100)
def main(input_dir: Path, min_per_class: int):
    setup_logging(level="WARNING")
    r = Report()

    paths = {n: input_dir / f"{n}.parquet" for n in ("full", "train", "val", "test")}
    for name, p in paths.items():
        if not p.exists():
            r.fail(f"missing file: {p}")
    if r.failures:
        sys.exit(r.print())

    full  = pd.read_parquet(paths["full"])
    train = pd.read_parquet(paths["train"])
    val   = pd.read_parquet(paths["val"])
    test  = pd.read_parquet(paths["test"])

    mode = detect_mode(full)
    print(f"\nDataset summary (mode={mode}):")
    print(f"  full:  {len(full):,} rows  ({full['genre'].nunique()} genres)")
    print(f"  train: {len(train):,} rows")
    print(f"  val:   {len(val):,} rows")
    print(f"  test:  {len(test):,} rows")
    print("\nClass distribution (train):")
    print(train["genre"].value_counts().to_string())

    for label, df in [("full", full), ("train", train), ("val", val), ("test", test)]:
        check_genre(df, label, r, min_per_class)
        check_nulls(df, label, r, mode)
        check_audio_ranges(df, label, r)
        check_lyric_lengths(df, label, r)

    check_disjoint(train, val, test, r)
    sys.exit(r.print())


if __name__ == "__main__":
    main()
