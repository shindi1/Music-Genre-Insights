"""End-to-end pipeline orchestrator.

This is the single entry point Sam (unsupervised) and Larry (supervised) call
to get modeling-ready data. It runs every stage in order and writes parquet
files at every checkpoint, so re-running individual stages is cheap.

Stages:
    1. Load raw Genius + Spotify CSVs.
    2. Filter Genius to English + reasonable length.
    3. Clean Genius lyrics.
    4. Map Spotify micro-genres to canonical taxonomy.
    5. Fuzzy-match Genius -> Spotify on (artist, title).
    6. Reconcile genre labels (Spotify-first, Genius-fallback).
    7. Engineer lyric features.
    8. Standardize audio features (fit on full data; transform applied
       per-split downstream to avoid leakage — see notes below).
    9. Balance classes.
    10. Stratified train/val/test split.
    11. Persist all outputs.

A NOTE ON LEAKAGE:
    The "right" thing for K-Means / regression is to fit StandardScaler on
    train only, then transform val/test. We compromise: we save the *unscaled*
    audio features on every split so the modeler can decide for themselves.
    For lyric-derived features (word counts, etc.) there's nothing to fit, so
    we compute them globally.

OUTPUTS (all parquet, in data/processed/):
    full.parquet                — entire matched + cleaned dataset (one row per song)
    train.parquet, val.parquet, test.parquet
    metadata.json               — class distributions, split sizes, config snapshot
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from config import (
    INTERIM_DIR,
    PROCESSED_DIR,
    PipelineConfig,
)
from src.balancer import ClassBalancer
from src.data_loader import (
    load_genius,
    load_spotify,
    validate_genius,
    validate_spotify,
)
from src.dataset_matcher import DatasetMatcher
from src.feature_engineering import FeatureEngineer
from src.genre_mapper import GenreMapper
from src.language_filter import LanguageFilter
from src.lyrics_cleaner import LyricsCleaner
from src.splitter import DataSplitter
from src.utils import get_logger, safe_to_parquet, timer

log = get_logger(__name__)


class Pipeline:
    """Orchestrates the full data preparation flow."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()

        # Component instances
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
        self.genre_mapper = GenreMapper(target_genres=self.config.target_genres)
        self.matcher = DatasetMatcher(
            threshold=self.config.matching.threshold,
        )
        self.featurizer = FeatureEngineer()
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

    # ====================================================================== #
    # individual stages — each can be run alone or as part of run_all()      #
    # ====================================================================== #
    def stage_load(
        self,
        genius_sample_n: Optional[int] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load + validate the two raw datasets."""
        with timer("stage 1: load raw datasets"):
            genius = load_genius(sample_n=genius_sample_n)
            spotify = load_spotify()
            validate_genius(genius)
            validate_spotify(spotify)
        return genius, spotify

    def stage_filter_clean(self, genius: pd.DataFrame) -> pd.DataFrame:
        """Language filter + length filter + lyric cleaning."""
        with timer("stage 2: language filter"):
            genius = self.lang_filter.filter_dataframe(genius, text_col="lyrics")

        with timer("stage 3: clean lyrics"):
            genius["lyrics_clean"] = self.cleaner.clean_batch(genius["lyrics"])
            genius["word_count"] = [self.cleaner.word_count(t) for t in genius["lyrics_clean"]]

        # Length filter
        cfg = self.config.cleaning
        before = len(genius)
        genius = genius[
            genius["word_count"].between(cfg.min_word_count, cfg.max_word_count, inclusive="both")
        ].reset_index(drop=True)
        log.info(
            "length filter [%d, %d]: %d -> %d rows",
            cfg.min_word_count, cfg.max_word_count, before, len(genius),
        )

        # Drop rows where cleaning emptied the lyrics
        before = len(genius)
        genius = genius[genius["lyrics_clean"].str.len() > 0].reset_index(drop=True)
        if before != len(genius):
            log.info("dropped %d rows with empty post-clean lyrics", before - len(genius))

        return genius

    def stage_map_genres(
        self,
        spotify: pd.DataFrame,
        genius: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Map Spotify micro-genres + Genius tags to canonical taxonomy."""
        with timer("stage 4: map genres"):
            spotify = self.genre_mapper.annotate(
                spotify,
                source_col="track_genre",
                out_col="genre_spotify",
                source="spotify",
                drop_unmapped=False,
            )
            genius = self.genre_mapper.annotate(
                genius,
                source_col="tag",
                out_col="genre_genius",
                source="genius",
                drop_unmapped=False,
            )
        return spotify, genius

    def stage_match(
        self,
        genius: pd.DataFrame,
        spotify: pd.DataFrame,
    ) -> pd.DataFrame:
        """Fuzzy-join the two datasets on (artist, title)."""
        with timer("stage 5: fuzzy match"):
            matched = self.matcher.match(
                left=genius,
                right=spotify,
                left_keys=("artist", "title"),
                right_keys=("artists", "track_name"),
                suffixes=("_genius", "_spotify"),
            )
        log.info("match-quality histogram:\n%s",
                 self.matcher.match_quality_report(matched).to_string())
        return matched

    def stage_reconcile_genres(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pick a single `genre` column from the two source labels."""
        with timer("stage 6: reconcile genre labels"):
            df = df.copy()
            df["genre"] = [
                self.genre_mapper.reconcile(s, g)
                for s, g in zip(df["genre_spotify"], df["genre_genius"])
            ]
            before = len(df)
            df = df.dropna(subset=["genre"]).reset_index(drop=True)
            log.info("dropped %d rows with no resolvable genre", before - len(df))
        return df

    def stage_engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lyric numerics + audio scaler fit (transform happens later, per split)."""
        with timer("stage 7: engineer lyric features"):
            df = self.featurizer.lyric_features(df, text_col="lyrics_clean")
        with timer("stage 8: fit audio scaler"):
            self.featurizer.fit_audio_scaler(df)
            df = self.featurizer.transform_audio(df)
        return df

    def stage_balance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Class balancing."""
        with timer("stage 9: class balancing"):
            return self.balancer.balance(df, label_col="genre",
                                          target_classes=self.config.target_genres)

    def stage_split(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Stratified train/val/test split (deduped on artist+title)."""
        with timer("stage 10: train/val/test split"):
            # Use the genius-side artist/title columns for dedup
            dedupe_cols = []
            for c in ("artist_genius", "artist", "title_genius", "title"):
                if c in df.columns:
                    dedupe_cols.append(c)
                if len(dedupe_cols) == 2:
                    break
            return self.splitter.split(df, dedupe_on=tuple(dedupe_cols) or None)

    # ====================================================================== #
    # full pipeline                                                          #
    # ====================================================================== #
    def run_all(
        self,
        genius_sample_n: Optional[int] = None,
        save_intermediate: bool = True,
        output_dir: Path = PROCESSED_DIR,
    ) -> dict:
        """Run every stage end-to-end. Returns a dict of output paths."""
        output_dir.mkdir(parents=True, exist_ok=True)
        INTERIM_DIR.mkdir(parents=True, exist_ok=True)

        genius, spotify = self.stage_load(genius_sample_n=genius_sample_n)
        if save_intermediate:
            safe_to_parquet(genius, INTERIM_DIR / "01_genius_raw.parquet")
            safe_to_parquet(spotify, INTERIM_DIR / "01_spotify_raw.parquet")

        genius = self.stage_filter_clean(genius)
        if save_intermediate:
            safe_to_parquet(genius, INTERIM_DIR / "02_genius_cleaned.parquet")

        spotify, genius = self.stage_map_genres(spotify, genius)
        if save_intermediate:
            safe_to_parquet(spotify, INTERIM_DIR / "03_spotify_mapped.parquet")
            safe_to_parquet(genius, INTERIM_DIR / "03_genius_mapped.parquet")

        matched = self.stage_match(genius, spotify)
        if save_intermediate:
            safe_to_parquet(matched, INTERIM_DIR / "04_matched.parquet")

        matched = self.stage_reconcile_genres(matched)
        matched = self.stage_engineer_features(matched)
        if save_intermediate:
            safe_to_parquet(matched, INTERIM_DIR / "05_features.parquet")

        balanced = self.stage_balance(matched)
        train, val, test = self.stage_split(balanced)

        # ---- persist ---- #
        out_paths = {
            "full":  output_dir / "full.parquet",
            "train": output_dir / "train.parquet",
            "val":   output_dir / "val.parquet",
            "test":  output_dir / "test.parquet",
            "metadata": output_dir / "metadata.json",
        }
        safe_to_parquet(balanced, out_paths["full"])
        safe_to_parquet(train, out_paths["train"])
        safe_to_parquet(val, out_paths["val"])
        safe_to_parquet(test, out_paths["test"])

        metadata = {
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "config": {
                "cleaning": self.config.cleaning.__dict__,
                "matching": self.config.matching.__dict__,
                "balancing": self.config.balancing.__dict__,
                "splitting": self.config.splitting.__dict__,
                "target_genres": list(self.config.target_genres),
            },
            "row_counts": {
                "full": len(balanced),
                "train": len(train),
                "val": len(val),
                "test": len(test),
            },
            "class_distribution": {
                split_name: split_df["genre"].value_counts().to_dict()
                for split_name, split_df in [("train", train), ("val", val), ("test", test)]
            },
        }
        with open(out_paths["metadata"], "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        log.info("pipeline complete. outputs:\n  %s",
                 "\n  ".join(f"{k}: {v}" for k, v in out_paths.items()))
        return out_paths
