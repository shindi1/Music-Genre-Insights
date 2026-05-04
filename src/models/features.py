"""Build the feature matrix for classification.

Two modes:
    audio  — 9 scaled Spotify audio features (StandardScaler fitted on train)
    lyrics — TF-IDF on cleaned lyrics + lyric numeric stats
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import LYRIC_FEATURE_KEYS

AUDIO_COLS = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]
LYRIC_FEATURE_COLS = [f"lyr_{k}" for k in LYRIC_FEATURE_KEYS]


@dataclass
class FeatureBundle:
    X: csr_matrix
    y: np.ndarray
    feature_names: list[str]
    vectorizer: Optional[TfidfVectorizer] = None
    scaler: Optional[StandardScaler] = None


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
    use_tfidf: bool = True,
    use_lyric_numerics: bool = True,
) -> tuple[FeatureBundle, ...]:
    """Fit transformers on train, apply to all splits.

    Auto-disables a block if its required columns are absent:
    - use_tfidf  requires `lyrics_clean` column
    - use_audio  requires audio feature columns
    - use_lyric_numerics requires `lyr_*` columns
    """
    dfs = [df for df in (train_df, val_df, test_df) if df is not None]

    # Auto-detect available blocks
    has_lyrics = text_col in train_df.columns
    has_audio  = all(c in train_df.columns for c in AUDIO_COLS)
    has_lyr_numerics = any(c in train_df.columns for c in LYRIC_FEATURE_COLS)

    use_tfidf         = use_tfidf         and has_lyrics
    use_audio         = use_audio         and has_audio
    use_lyric_numerics = use_lyric_numerics and has_lyr_numerics

    # Fit TF-IDF on train
    vectorizer = None
    if use_tfidf:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            lowercase=True,
            sublinear_tf=True,
        )
        vectorizer.fit(train_df[text_col].fillna(""))

    # Fit audio scaler on train only (avoid leakage)
    scaler = None
    if use_audio:
        audio_train = train_df[AUDIO_COLS].fillna(0).to_numpy(dtype=np.float64)
        scaler = StandardScaler()
        scaler.fit(audio_train)

    bundles = []
    for df in dfs:
        blocks = []
        names  = []

        if use_tfidf:
            X_text = vectorizer.transform(df[text_col].fillna(""))
            blocks.append(X_text)
            names.extend(f"tfidf::{t}" for t in vectorizer.get_feature_names_out())

        if use_audio:
            scaled = scaler.transform(df[AUDIO_COLS].fillna(0).to_numpy(dtype=np.float64))
            blocks.append(csr_matrix(scaled.astype(np.float32)))
            names.extend(AUDIO_COLS)

        if use_lyric_numerics:
            present = [c for c in LYRIC_FEATURE_COLS if c in df.columns]
            lyr = df[present].to_numpy(dtype=np.float32)
            blocks.append(csr_matrix(lyr))
            names.extend(present)

        if not blocks:
            raise ValueError("No feature blocks enabled — check mode flags.")

        X = hstack(blocks).tocsr() if len(blocks) > 1 else blocks[0]
        y = df[label_col].to_numpy()
        bundles.append(FeatureBundle(X=X, y=y, feature_names=names,
                                     vectorizer=vectorizer, scaler=scaler))

    return tuple(bundles)
