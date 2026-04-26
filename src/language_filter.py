"""Language filtering using langdetect.

Why this is its own module:
    - langdetect is non-deterministic without seeding
    - it raises on empty / very short text — we need to swallow those exceptions
    - we want progress reporting on multi-million-row corpora
    - some Kaggle datasets already provide a `language` column we should trust
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import pandas as pd
from tqdm.auto import tqdm

from src.utils import get_logger

log = get_logger(__name__)

try:
    from langdetect import DetectorFactory, LangDetectException, detect
    _LANGDETECT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LANGDETECT_AVAILABLE = False


class LanguageFilter:
    """Detect & filter songs by language.

    Two modes of operation:
        1. trust an existing `language` column (fast, recommended when present)
        2. detect language from lyric text via langdetect (slower, ~1k rows/sec)
    """

    def __init__(
        self,
        languages_to_keep: Iterable[str] = ("en",),
        seed: int = 42,
        min_chars_for_detect: int = 30,
    ) -> None:
        self.languages_to_keep = set(languages_to_keep)
        self.min_chars_for_detect = min_chars_for_detect

        if _LANGDETECT_AVAILABLE:
            DetectorFactory.seed = seed  # ensures reproducible detections
        else:
            log.warning(
                "langdetect not installed — detect_language() will return 'unknown'. "
                "Install with `pip install langdetect`."
            )

    # ------------------------------------------------------------------ core
    def detect_language(self, text: Optional[str]) -> str:
        """Return ISO-639-1 code or 'unknown'."""
        if not _LANGDETECT_AVAILABLE:
            return "unknown"
        if not isinstance(text, str) or len(text) < self.min_chars_for_detect:
            return "unknown"
        try:
            return detect(text)
        except LangDetectException:
            return "unknown"

    def detect_batch(self, texts: Iterable[Optional[str]], show_progress: bool = True) -> List[str]:
        iterator = tqdm(texts, desc="detecting languages") if show_progress else texts
        return [self.detect_language(t) for t in iterator]

    # ----------------------------------------------------------- DataFrame API
    def filter_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "lyrics",
        language_col: Optional[str] = "language",
    ) -> pd.DataFrame:
        """Return rows whose language is in `languages_to_keep`.

        If `language_col` exists in the DataFrame, trust it. Otherwise, detect
        from `text_col`.
        """
        df = df.copy()

        if language_col and language_col in df.columns:
            log.info("trusting existing '%s' column", language_col)
        else:
            log.info("detecting language from '%s' column", text_col)
            df["language"] = self.detect_batch(df[text_col].tolist())
            language_col = "language"

        before = len(df)
        df = df[df[language_col].isin(self.languages_to_keep)].reset_index(drop=True)
        after = len(df)

        log.info(
            "language filter kept %d / %d rows (%.1f%%) — keeping %s",
            after, before, 100 * after / max(before, 1), sorted(self.languages_to_keep),
        )
        return df
