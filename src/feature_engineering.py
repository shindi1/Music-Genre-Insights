"""Feature engineering on lyrics + audio features.

We engineer two flavors of features:

    1. Lyric-derived numeric features
       (word count, vocabulary diversity, line count, repetition,
        avg-word-length, exclamation-density, etc.)
       These are useful as additional input to XGBoost / KNN, and as the
       *only* numeric input to K-Means clustering when audio features are
       unavailable.

    2. Standardized audio features
       (Spotify's continuous features scaled to zero-mean unit-variance).
       Critical for K-Means / KNN where unscaled features dominate the
       distance metric.

Note: this module deliberately does not vectorize lyrics into TF-IDF or
embeddings — that's a modeling concern (Larry's territory). We just stage
the numerics.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils import get_logger

log = get_logger(__name__)


# Spotify's standard continuous audio features. Tempo is on a wholly different
# scale (40-220 BPM) than the others (mostly 0-1), so scaling is essential.
DEFAULT_AUDIO_FEATURES: Tuple[str, ...] = (
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
)

LYRIC_FEATURE_KEYS: Tuple[str, ...] = (
    "word_count", "unique_word_count", "vocab_diversity",
    "line_count", "avg_word_length", "avg_line_length",
    "exclaim_density", "question_density",
    "repetition_ratio", "uppercase_ratio",
)


class FeatureEngineer:
    """Compute lyric-derived numerics and scale audio features."""

    def __init__(
        self,
        audio_features: Iterable[str] = DEFAULT_AUDIO_FEATURES,
    ) -> None:
        self.audio_features = list(audio_features)
        self._scaler: Optional[StandardScaler] = None

    # --------------------------------------------------------- lyric features
    @staticmethod
    def _lyric_features_one(text: str) -> dict:
        """Compute lyric-derived stats for a single song."""
        if not isinstance(text, str) or not text:
            return {
                "word_count": 0, "unique_word_count": 0, "vocab_diversity": 0.0,
                "line_count": 0, "avg_word_length": 0.0, "avg_line_length": 0.0,
                "exclaim_density": 0.0, "question_density": 0.0,
                "repetition_ratio": 0.0, "uppercase_ratio": 0.0,
            }
        words = text.split()
        n_words = len(words)
        if n_words == 0:
            return FeatureEngineer._lyric_features_one("")  # delegate to empty case

        # Lowercase counter for vocab diversity (don't double-count "love"/"Love")
        lc_words = [w.lower() for w in words]
        unique = len(set(lc_words))

        lines = [ln for ln in text.splitlines() if ln.strip()]
        n_lines = len(lines)

        char_count = sum(len(w) for w in words)
        avg_word_len = char_count / n_words if n_words else 0.0
        avg_line_len = (n_words / n_lines) if n_lines else 0.0

        # Repetition: how often does the most common word appear?
        most_common_count = Counter(lc_words).most_common(1)[0][1] if lc_words else 0
        repetition = most_common_count / n_words

        exclaim_density = text.count("!") / n_words
        question_density = text.count("?") / n_words

        upper_chars = sum(1 for c in text if c.isupper())
        total_letters = sum(1 for c in text if c.isalpha())
        upper_ratio = (upper_chars / total_letters) if total_letters else 0.0

        return {
            "word_count": n_words,
            "unique_word_count": unique,
            "vocab_diversity": unique / n_words,    # type-token ratio
            "line_count": n_lines,
            "avg_word_length": avg_word_len,
            "avg_line_length": avg_line_len,
            "exclaim_density": exclaim_density,
            "question_density": question_density,
            "repetition_ratio": repetition,
            "uppercase_ratio": upper_ratio,
        }

    def lyric_features(
        self,
        df: pd.DataFrame,
        text_col: str = "lyrics_clean",
        prefix: str = "lyr_",
    ) -> pd.DataFrame:
        """Append lyric-derived numeric features to `df`."""
        log.info("computing lyric features over %d rows...", len(df))
        feats = pd.DataFrame([self._lyric_features_one(t) for t in df[text_col]])
        feats.columns = [f"{prefix}{c}" for c in feats.columns]
        return pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)

    # ----------------------------------------------------- audio scaling
    def fit_audio_scaler(self, df: pd.DataFrame) -> StandardScaler:
        """Fit StandardScaler on training-set audio features. Stores it on self."""
        present = [c for c in self.audio_features if c in df.columns]
        if not present:
            raise ValueError(
                f"none of the expected audio features {self.audio_features} "
                f"are present in df.columns ({list(df.columns)[:10]}...)"
            )
        log.info("fitting audio-feature scaler on %d rows / %d features",
                 len(df), len(present))
        self._scaler = StandardScaler()
        self._scaler.fit(df[present].values)
        # Stash the columns we fitted on so transform() never accidentally
        # re-orders or includes a different set.
        self._fitted_cols = present
        return self._scaler

    def transform_audio(
        self,
        df: pd.DataFrame,
        suffix: str = "_z",
    ) -> pd.DataFrame:
        """Apply the fitted scaler. Returns a *copy* with new `<feat>_z` columns."""
        if self._scaler is None:
            raise RuntimeError("call fit_audio_scaler() before transform_audio()")
        df = df.copy()
        scaled = self._scaler.transform(df[self._fitted_cols].values)
        for i, col in enumerate(self._fitted_cols):
            df[f"{col}{suffix}"] = scaled[:, i]
        return df

