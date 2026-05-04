"""Pipeline orchestrator — two modes: audio-only and lyrics-only.

Audio mode  (--mode audio):
    1. Load Spotify SQLite (cached parquet after first run)
    2. Map micro-genres -> canonical taxonomy
    3. Balance classes
    4. Stratified split
    Output: 9 raw audio features + genre label

Lyrics mode (--mode lyrics):
    1. Load Genius CSV
    2. Language filter
    3. Map Genius tags -> canonical taxonomy
    4. Clean lyrics
    5. Engineer lyric features
    6. Balance classes
    7. Stratified split
    Output: lyrics_clean + lyric numeric features + genre label
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from config import INTERIM_DIR, PROCESSED_AUDIO_DIR, PROCESSED_LYRICS_DIR, PipelineConfig
from src.balancer import ClassBalancer
from src.data_loader import (
    load_genius, load_genius_chunked,
    load_spotify_sqlite,
    validate_genius, validate_spotify,
)
from src.feature_engineering import FeatureEngineer
from src.genre_mapper import GenreMapper
from src.language_filter import LanguageFilter
from src.lyrics_cleaner import LyricsCleaner
from src.splitter import DataSplitter
from src.utils import get_logger, safe_to_parquet, timer

log = get_logger(__name__)

AUDIO_FEATURE_COLS = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]


class Pipeline:
    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()

        self.genre_mapper = GenreMapper(target_genres=self.config.target_genres)
        self.balancer = ClassBalancer(
            strategy=self.config.balancing.strategy,
            samples_per_class=self.config.balancing.samples_per_class,
            min_samples_per_class=self.config.balancing.min_samples_per_class,
            random_state=self.config.balancing.random_state,
        )
        self.splitter = DataSplitter(
            test_size=self.config.splitting.test_size,
            val_size=self.config.splitting.val_size,
            random_state=self.config.splitting.random_state,
            stratify_on=self.config.splitting.stratify_on,
        )
        # Lyrics-mode only
        self.cleaner = LyricsCleaner(
            remove_section_markers=self.config.cleaning.remove_section_markers,
            remove_parentheticals=self.config.cleaning.remove_parentheticals,
            remove_producer_tags=self.config.cleaning.remove_producer_tags,
            fix_encoding=self.config.cleaning.fix_encoding,
            normalize_unicode=self.config.cleaning.normalize_unicode,
            lowercase=self.config.cleaning.lowercase,
        )
        self.lang_filter = LanguageFilter(
            languages_to_keep=self.config.cleaning.languages_to_keep,
            seed=self.config.cleaning.langdetect_seed,
        )
        self.featurizer = FeatureEngineer()

    # ------------------------------------------------------------------ audio
    def run_audio(
        self,
        output_dir: Path = PROCESSED_AUDIO_DIR,
        save_intermediate: bool = False,
    ) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)

        with timer("stage 1: load Spotify SQLite"):
            df = load_spotify_sqlite()
            validate_spotify(df)

        with timer("stage 2: map genres"):
            df = self.genre_mapper.annotate(
                df, source_col="track_genre", out_col="genre",
                source="spotify", drop_unmapped=True,
            )

        # Keep only columns needed for modeling
        keep = ["track_id", "track_name", "artists", "popularity", "genre"] + AUDIO_FEATURE_COLS
        df = df[[c for c in keep if c in df.columns]].reset_index(drop=True)

        with timer("stage 3: balance classes"):
            df = self.balancer.balance(df, label_col="genre",
                                       target_classes=self.config.target_genres)

        with timer("stage 4: split"):
            train, val, test = self.splitter.split(df, dedupe_on=("artists", "track_name"))

        return self._save(df, train, val, test, output_dir)

    # ----------------------------------------------------------------- lyrics
    def run_lyrics(
        self,
        genius_sample_n: Optional[int] = None,
        output_dir: Path = PROCESSED_LYRICS_DIR,
        save_intermediate: bool = False,
    ) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        INTERIM_DIR.mkdir(parents=True, exist_ok=True)

        with timer("stage 1: load Genius"):
            if genius_sample_n is None:
                chunks = []
                total = 0
                for i, chunk in enumerate(load_genius_chunked(chunksize=500_000), 1):
                    total += len(chunk)
                    chunk = chunk[chunk["language"].isin(["en"]) & (chunk["tag"] != "misc")]
                    chunks.append(chunk)
                    log.info("chunk %d: %d total, %d kept", i, total,
                             sum(len(c) for c in chunks))
                df = pd.concat(chunks, ignore_index=True)
                for col in ("tag", "language"):
                    if col in df.columns:
                        df[col] = df[col].astype("category")
            else:
                df = load_genius(
                    nrows=min(genius_sample_n * 3, 3_000_000),
                    sample_n=genius_sample_n,
                )
            validate_genius(df)

        with timer("stage 2: language filter"):
            df = self.lang_filter.filter_dataframe(df, text_col="lyrics")
            cfg = self.config.cleaning
            raw_wc = df["lyrics"].str.split().str.len().fillna(0)
            before = len(df)
            df = df[raw_wc.between(cfg.min_word_count, cfg.max_word_count * 2)].reset_index(drop=True)
            log.info("length pre-filter: %d -> %d", before, len(df))

        with timer("stage 3: map genres"):
            df = self.genre_mapper.annotate(
                df, source_col="tag", out_col="genre",
                source="genius", drop_unmapped=True,
            )

        with timer("stage 4: clean lyrics"):
            df = df.copy()
            df["lyrics_clean"] = self.cleaner.clean_batch(df["lyrics"])
            df["word_count"] = [self.cleaner.word_count(t) for t in df["lyrics_clean"]]
            cfg = self.config.cleaning
            before = len(df)
            df = df[df["word_count"].between(cfg.min_word_count, cfg.max_word_count)].reset_index(drop=True)
            log.info("length filter: %d -> %d", before, len(df))

        with timer("stage 5: lyric features"):
            df = self.featurizer.lyric_features(df, text_col="lyrics_clean")

        with timer("stage 6: balance"):
            df = self.balancer.balance(df, label_col="genre",
                                       target_classes=self.config.target_genres)

        with timer("stage 7: split"):
            train, val, test = self.splitter.split(df, dedupe_on=("artist", "title"))

        return self._save(df, train, val, test, output_dir)

    # ------------------------------------------------------------------ shared
    def _save(self, full, train, val, test, output_dir: Path) -> dict:
        paths = {
            "full":     output_dir / "full.parquet",
            "train":    output_dir / "train.parquet",
            "val":      output_dir / "val.parquet",
            "test":     output_dir / "test.parquet",
            "metadata": output_dir / "metadata.json",
        }
        safe_to_parquet(full,  paths["full"])
        safe_to_parquet(train, paths["train"])
        safe_to_parquet(val,   paths["val"])
        safe_to_parquet(test,  paths["test"])

        metadata = {
            "mode": self.config.mode,
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "row_counts": {
                "full": len(full), "train": len(train),
                "val": len(val), "test": len(test),
            },
            "class_distribution": {
                name: df["genre"].value_counts().to_dict()
                for name, df in [("train", train), ("val", val), ("test", test)]
            },
        }
        with open(paths["metadata"], "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        log.info("pipeline complete:\n  %s",
                 "\n  ".join(f"{k}: {v}" for k, v in paths.items()))
        return paths

    def run_all(
        self,
        mode: Optional[str] = None,
        genius_sample_n: Optional[int] = None,
        save_intermediate: bool = False,
        output_dir: Optional[Path] = None,
    ) -> dict:
        mode = mode or self.config.mode
        if output_dir is None:
            output_dir = PROCESSED_AUDIO_DIR if mode == "audio" else PROCESSED_LYRICS_DIR
        if mode == "audio":
            return self.run_audio(output_dir=output_dir,
                                  save_intermediate=save_intermediate)
        elif mode == "lyrics":
            return self.run_lyrics(genius_sample_n=genius_sample_n,
                                   output_dir=output_dir,
                                   save_intermediate=save_intermediate)
        else:
            raise ValueError(f"unknown mode: {mode!r} — use 'audio' or 'lyrics'")
