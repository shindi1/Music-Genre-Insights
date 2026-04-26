"""Class balancing for genre classification.

Pop and hip-hop dominate most lyrics datasets (40-60% combined) while country,
electronic, and r&b are underrepresented. Without intervention, a classifier
trained on the raw distribution will achieve high accuracy by predicting "pop"
for everything — but useless macro-F1.

This module implements three strategies:

    "undersample": cap each class to `samples_per_class` by random sampling.
        Fast, simple, and the recommended default for transformer-based or
        TF-IDF-based pipelines where having too much data is rarely a bottleneck.

    "cap": leave smaller classes alone, only undersample classes above the cap.
        Useful when minority classes are already small enough that
        undersampling further would hurt.

    "none": pass-through. Use class_weight='balanced' in the model instead.
"""
from __future__ import annotations

from typing import Iterable, Literal, Optional

import pandas as pd

from src.utils import get_logger

log = get_logger(__name__)

Strategy = Literal["undersample", "cap", "none"]


class ClassBalancer:
    def __init__(
        self,
        strategy: Strategy = "undersample",
        samples_per_class: int = 4000,
        min_samples_per_class: int = 500,
        random_state: int = 42,
    ) -> None:
        if strategy not in ("undersample", "cap", "none"):
            raise ValueError(f"unknown strategy: {strategy!r}")
        self.strategy = strategy
        self.samples_per_class = samples_per_class
        self.min_samples_per_class = min_samples_per_class
        self.random_state = random_state

    def balance(
        self,
        df: pd.DataFrame,
        label_col: str = "genre",
        target_classes: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """Return a balanced view of `df`.

        If `target_classes` is given, drop rows whose label is not in it
        (canonicalization). Otherwise use whatever labels exist.
        """
        if target_classes is not None:
            target_classes = list(target_classes)
            n_before = len(df)
            df = df[df[label_col].isin(target_classes)].reset_index(drop=True)
            log.info("filtered to target classes: %d -> %d rows",
                     n_before, len(df))

        counts = df[label_col].value_counts()
        log.info("pre-balance class counts:\n%s", counts.to_string())

        # Health-check: warn if any class is starving
        starved = counts[counts < self.min_samples_per_class]
        if not starved.empty:
            log.warning(
                "classes below min_samples_per_class=%d: %s",
                self.min_samples_per_class, starved.to_dict(),
            )

        if self.strategy == "none":
            return df.reset_index(drop=True)

        if self.strategy == "undersample":
            n = self.samples_per_class
            sampled = []
            for cls, sub in df.groupby(label_col, sort=False):
                take = min(n, len(sub))
                sampled.append(sub.sample(n=take, random_state=self.random_state))
            balanced = pd.concat(sampled, ignore_index=True)

        elif self.strategy == "cap":
            n = self.samples_per_class
            sampled = []
            for cls, sub in df.groupby(label_col, sort=False):
                if len(sub) > n:
                    sampled.append(sub.sample(n=n, random_state=self.random_state))
                else:
                    sampled.append(sub)
            balanced = pd.concat(sampled, ignore_index=True)

        else:  # pragma: no cover  (validated above)
            raise AssertionError("unreachable")

        # Shuffle so concat artifact ordering doesn't bias downstream splits
        balanced = balanced.sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)

        log.info("post-balance class counts:\n%s",
                 balanced[label_col].value_counts().to_string())
        return balanced
