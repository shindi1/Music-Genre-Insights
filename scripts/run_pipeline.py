"""Run the data pipeline in audio or lyrics mode.

Examples:
    python scripts/run_pipeline.py --mode audio
    python scripts/run_pipeline.py --mode lyrics --sample 300000
    python scripts/run_pipeline.py --mode audio --output-dir data/processed/audio
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import click

from config import (PROCESSED_AUDIO_DIR, PROCESSED_LYRICS_DIR,
                    BalancingConfig, CleaningConfig, PipelineConfig, SplitConfig)
from src.pipeline import Pipeline
from src.utils import setup_logging


@click.command(context_settings=dict(show_default=True))
@click.option("--mode", type=click.Choice(["audio", "lyrics"]), default="audio",
              help="Feature source: audio features (Spotify) or lyrics (Genius).")
@click.option("--sample", type=int, default=None,
              help="Lyrics mode only: random-sample N Genius rows.")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None,
              help="Output directory (default: data/processed/audio or data/processed/lyrics)")
@click.option("--samples-per-class", type=int, default=4000)
@click.option("--test-size", type=click.FloatRange(0.05, 0.4), default=0.15)
@click.option("--val-size", type=click.FloatRange(0.0, 0.4), default=0.15)
@click.option("--seed", type=int, default=42)
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING"]), default="INFO")
def main(mode, sample, output_dir, samples_per_class, test_size, val_size, seed, log_level):
    log = setup_logging(level=log_level, log_file=PROJECT_ROOT / "logs" / "pipeline.log")
    log.info("=" * 70)
    log.info("PIPELINE — mode=%s", mode)
    log.info("=" * 70)

    cfg = PipelineConfig(
        mode=mode,
        balancing=BalancingConfig(samples_per_class=samples_per_class, random_state=seed),
        splitting=SplitConfig(test_size=test_size, val_size=val_size, random_state=seed),
        cleaning=CleaningConfig(langdetect_seed=seed),
    )

    pipe = Pipeline(cfg)
    if output_dir is None:
        output_dir = PROCESSED_AUDIO_DIR if mode == "audio" else PROCESSED_LYRICS_DIR

    paths = pipe.run_all(
        mode=mode,
        genius_sample_n=sample,
        output_dir=output_dir,
    )

    log.info("")
    log.info("✅ DONE. Outputs:")
    for name, path in paths.items():
        log.info("   %-10s -> %s", name, path)


if __name__ == "__main__":
    main()
