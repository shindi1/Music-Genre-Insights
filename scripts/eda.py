"""Exploratory data analysis on the processed dataset.

Generates a battery of plots and statistics for the project report.
Outputs go into reports/figures/.

Charts produced:
    - Class distribution
    - Audio-feature distributions by genre (boxplots)
    - Audio-feature correlation heatmap
    - Lyric-length distributions by genre (lyrics mode only)
    - Top words per genre (lyrics mode only)
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import click
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works in any environment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PROCESSED_DIR, TARGET_GENRES
from src.feature_engineering import DEFAULT_AUDIO_FEATURES
from src.utils import setup_logging


def fig_class_distribution(df: pd.DataFrame, out: Path) -> None:
    counts = df["genre"].value_counts().reindex(TARGET_GENRES, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind="bar", ax=ax, color="#4c72b0", edgecolor="black")
    ax.set_title("Class distribution")
    ax.set_xlabel("genre")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fig_audio_boxplots(df: pd.DataFrame, out: Path) -> None:
    feats = [f for f in DEFAULT_AUDIO_FEATURES if f in df.columns]
    if not feats:
        return
    n = len(feats)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i, feat in enumerate(feats):
        ax = axes[i]
        data = [df[df["genre"] == g][feat].dropna().values for g in TARGET_GENRES]
        ax.boxplot(data, labels=TARGET_GENRES, showfliers=False)
        ax.set_title(feat)
        ax.tick_params(axis="x", rotation=45)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Audio features by genre", fontsize=14, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_correlation_heatmap(df: pd.DataFrame, out: Path) -> None:
    feats = [f for f in DEFAULT_AUDIO_FEATURES if f in df.columns]
    if not feats:
        return
    corr = df[feats].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats)))
    ax.set_yticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=45, ha="right")
    ax.set_yticklabels(feats)
    for i in range(len(feats)):
        for j in range(len(feats)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    color="black" if abs(corr.values[i, j]) < 0.5 else "white",
                    fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Audio feature correlation")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fig_lyric_length(df: pd.DataFrame, out: Path) -> None:
    if "lyr_word_count" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    data = [df[df["genre"] == g]["lyr_word_count"].values for g in TARGET_GENRES]
    ax.boxplot(data, labels=TARGET_GENRES, showfliers=False)
    ax.set_title("Lyric length (word count) by genre")
    ax.set_ylabel("words")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)



def top_words_per_genre(df: pd.DataFrame, out: Path, top_k: int = 25) -> None:
    """Save a CSV of top words per genre using simple frequency (no IDF)."""
    if "lyrics_clean" not in df.columns:
        return
    rows = []
    stopwords = set("""
    a an the and or but if then so as of in on at to for with from by is am are
    was were be been being do does did doing have has had having i you he she
    we they me him her us them my your his hers our their this that these those
    not no yes oh la na yeah ya hey
    """.split())
    for genre in TARGET_GENRES:
        sub = df[df["genre"] == genre]["lyrics_clean"].head(2000)
        c: Counter = Counter()
        for text in sub:
            for w in text.lower().split():
                w = "".join(ch for ch in w if ch.isalpha())
                if len(w) >= 3 and w not in stopwords:
                    c[w] += 1
        for word, n in c.most_common(top_k):
            rows.append({"genre": genre, "word": word, "count": n})
    pd.DataFrame(rows).to_csv(out, index=False)


@click.command()
@click.option("--input-dir", type=click.Path(path_type=Path), default=PROCESSED_DIR)
@click.option("--out-dir", type=click.Path(path_type=Path),
              default=PROJECT_ROOT / "reports" / "figures")
def main(input_dir: Path, out_dir: Path):
    """Generate EDA figures + summary stats."""
    setup_logging(level="INFO")
    out_dir.mkdir(parents=True, exist_ok=True)

    full = pd.read_parquet(input_dir / "full.parquet")
    print(f"loaded {len(full):,} rows")

    fig_class_distribution(full, out_dir / "class_distribution.png")
    fig_audio_boxplots(full, out_dir / "audio_features_by_genre.png")
    fig_correlation_heatmap(full, out_dir / "audio_feature_correlation.png")
    fig_lyric_length(full, out_dir / "lyric_length_by_genre.png")
    top_words_per_genre(full, out_dir / "top_words_per_genre.csv")

    # Numerical summary
    summary = full.groupby("genre")[
        [f for f in DEFAULT_AUDIO_FEATURES if f in full.columns]
    ].mean().round(3)
    summary.to_csv(out_dir / "audio_feature_means_by_genre.csv")
    print(f"\n✓ figures written to {out_dir}")
    print("\nMean audio features by genre:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
