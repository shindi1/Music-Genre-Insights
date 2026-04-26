"""Stratified train/val/test splitting.

Uses sklearn's train_test_split twice (because sklearn provides only a single
split). Stratifies on the genre label by default — critical so each split has
the same class distribution.

We also implement a *deduplicated* split: songs are deduplicated by
(artist, title) before splitting to prevent train/test leakage where the same
song appears multiple times (e.g. studio + live versions).
"""
from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import get_logger

log = get_logger(__name__)


class DataSplitter:
    def __init__(
        self,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
        stratify_on: str = "genre",
    ) -> None:
        if not 0 < test_size < 1:
            raise ValueError("test_size must be in (0, 1)")
        if not 0 <= val_size < 1:
            raise ValueError("val_size must be in [0, 1)")

        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.stratify_on = stratify_on

    def split(
        self,
        df: pd.DataFrame,
        dedupe_on: Optional[Tuple[str, ...]] = ("artist", "title"),
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split into (train, val, test) DataFrames.

        Args:
            df: balanced DataFrame with at least `self.stratify_on` column.
            dedupe_on: tuple of columns. If any are present, dedupe before
                splitting to prevent leakage. Pass None to skip.

        Returns:
            (train_df, val_df, test_df)
        """
        df = df.reset_index(drop=True)

        if dedupe_on:
            cols = [c for c in dedupe_on if c in df.columns]
            if cols:
                before = len(df)
                df = df.drop_duplicates(subset=cols).reset_index(drop=True)
                log.info("deduped on %s: %d -> %d rows", cols, before, len(df))

        stratify = df[self.stratify_on] if self.stratify_on in df.columns else None

        # First split: separate test
        train_val, test = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

        # Second split: carve val out of train_val
        if self.val_size > 0:
            stratify_inner = (
                train_val[self.stratify_on] if self.stratify_on in train_val.columns else None
            )
            # val_size is expressed as fraction of *original* df; convert to
            # fraction of train_val.
            val_frac = self.val_size / (1 - self.test_size)
            train, val = train_test_split(
                train_val,
                test_size=val_frac,
                random_state=self.random_state,
                stratify=stratify_inner,
            )
        else:
            train, val = train_val, train_val.iloc[:0]  # empty val

        train = train.reset_index(drop=True)
        val = val.reset_index(drop=True)
        test = test.reset_index(drop=True)

        log.info(
            "split sizes: train=%d (%.1f%%)  val=%d (%.1f%%)  test=%d (%.1f%%)",
            len(train), 100 * len(train) / len(df),
            len(val), 100 * len(val) / len(df),
            len(test), 100 * len(test) / len(df),
        )

        if self.stratify_on in df.columns:
            log.info(
                "train class dist:\n%s",
                train[self.stratify_on].value_counts(normalize=True).round(3).to_string(),
            )

        return train, val, test
