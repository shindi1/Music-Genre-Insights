"""Build the feature matrix for classification.

Combines three feature blocks:
    1. TF-IDF on cleaned lyrics (sparse)
    2. Z-scaled Spotify audio features (dense)
    3. Engineered lyric numerics (dense, prefix `lyr_`)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from src.feature_engineering import DEFAULT_AUDIO_FEATURES, LYRIC_FEATURE_KEYS

AUDIO_FEATURE_COLS = [f"{f}_z" for f in DEFAULT_AUDIO_FEATURES]
LYRIC_FEATURE_COLS = [f"lyr_{k}" for k in LYRIC_FEATURE_KEYS]


@dataclass
class FeatureBundle:
    """Container for the assembled feature matrix and metadata."""
    X: csr_matrix
    y: np.ndarray
    feature_names: list[str]
    vectorizer: TfidfVectorizer


def build_features(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    *,
    label_col: str = "genre",
    text_col: str = "lyrics_clean",
    max_features: int = 10_000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    use_audio: bool = True,
    use_lyric_numerics: bool = True,
) -> tuple[FeatureBundle, ...]:
    """Fit TF-IDF on train, transform all splits, hstack with numeric features.

    Returns one FeatureBundle per split passed in (train always, val/test if given).
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        lowercase=True,
        sublinear_tf=True,
    )
    vectorizer.fit(train_df[text_col].fillna(""))

    bundles = []
    for df in (train_df, val_df, test_df):
        if df is None:
            continue
        X_text = vectorizer.transform(df[text_col].fillna(""))
        blocks = [X_text]
        names = [f"tfidf::{t}" for t in vectorizer.get_feature_names_out()]

        if use_audio:
            audio = df[AUDIO_FEATURE_COLS].to_numpy(dtype=np.float32)
            blocks.append(csr_matrix(audio))
            names.extend(AUDIO_FEATURE_COLS)

        if use_lyric_numerics:
            lyr = df[LYRIC_FEATURE_COLS].to_numpy(dtype=np.float32)
            blocks.append(csr_matrix(lyr))
            names.extend(LYRIC_FEATURE_COLS)

        X = hstack(blocks).tocsr()
        y = df[label_col].to_numpy()
        bundles.append(FeatureBundle(X=X, y=y, feature_names=names, vectorizer=vectorizer))

    return tuple(bundles)
