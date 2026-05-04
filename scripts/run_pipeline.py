"""Main CLI entry point for the Cade data pipeline.

Examples:
    # Run on a 100k-row sample of Genius (fast — for development)
    python scripts/run_pipeline.py --sample 100000

    # Full run with custom output directory
    python scripts/run_pipeline.py --output-dir data/processed_v2

    # Skip intermediate parquet files (saves disk)
    python scripts/run_pipeline.py --no-intermediate

    # Tighter matching threshold + smaller per-class cap
    python scripts/run_pipeline.py --match-threshold 92 --samples-per-class 2000
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `src` and `config` importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import click

from config import (
    PROCESSED_DIR,
    BalancingConfig,
    CleaningConfig,
    MatchingConfig,
    PipelineConfig,
    SplitConfig,
)
from src.pipeline import Pipeline
from src.utils import setup_logging


@click.command(context_settings=dict(show_default=True))
@click.option("--sample", type=int, default=None,
              help="Random-sample N rows of the Genius dataset (omit for full).")
@click.option("--output-dir", type=click.Path(path_type=Path), default=PROCESSED_DIR,
              help="Where to write processed parquet files.")
@click.option("--no-intermediate", is_flag=True,
              help="Skip writing intermediate parquet files at every stage.")
@click.option("--match-threshold", type=click.IntRange(50, 100), default=87,
              help="Min rapidfuzz score to accept a Genius<->Spotify match.")
@click.option("--samples-per-class", type=int, default=4000,
              help="Per-class cap for balanced training set.")
@click.option("--min-words", type=int, default=50,
              help="Drop songs with fewer than this many words after cleaning.")
@click.option("--max-words", type=int, default=2000,
              help="Drop songs with more than this many words (concat errors).")
@click.option("--test-size", type=click.FloatRange(0.05, 0.4), default=0.15,
              help="Test split fraction.")
@click.option("--val-size", type=click.FloatRange(0.0, 0.4), default=0.15,
              help="Validation split fraction.")
@click.option("--spotify", type=click.Choice(["114k", "30k"]), default="114k",
              help="Which Spotify dataset to use.")
@click.option("--seed", type=int, default=42)
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING"]), default="INFO")
def main(
    sample, output_dir, no_intermediate, match_threshold, samples_per_class,
    min_words, max_words, test_size, val_size, spotify, seed, log_level,
):
    """Run the Cade data preparation pipeline end-to-end."""
    log_file = PROJECT_ROOT / "logs" / "pipeline.log"
    log = setup_logging(level=log_level, log_file=log_file)
    log.info("=" * 70)
    log.info("CADE DATA PIPELINE")
    log.info("=" * 70)

    cfg = PipelineConfig(
        cleaning=CleaningConfig(
            min_word_count=min_words,
            max_word_count=max_words,
            langdetect_seed=seed,
        ),
        matching=MatchingConfig(threshold=match_threshold),
        balancing=BalancingConfig(
            samples_per_class=samples_per_class,
            random_state=seed,
        ),
        splitting=SplitConfig(
            test_size=test_size,
            val_size=val_size,
            random_state=seed,
        ),
    )

    pipe = Pipeline(cfg)
    paths = pipe.run_all(
        genius_sample_n=sample,
        spotify_dataset=spotify,
        save_intermediate=not no_intermediate,
        output_dir=output_dir,
    )

    log.info("")
    log.info("✅ DONE. Outputs:")
    for name, path in paths.items():
        log.info("   %-10s -> %s", name, path)


if __name__ == "__main__":
    main()
