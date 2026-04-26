"""Cross-cutting utilities: logging, timing, IO helpers."""
from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd


# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    silent_libs: bool = True,
) -> logging.Logger:
    """Configure root logging once. Idempotent — safe to call repeatedly.

    Args:
        level: One of DEBUG, INFO, WARNING, ERROR.
        log_file: If given, also write logs to this file.
        silent_libs: Suppress overly chatty third-party loggers.

    Returns:
        Root logger so the caller can immediately .info() on it.
    """
    root = logging.getLogger()
    # Clear existing handlers to make this idempotent
    for h in list(root.handlers):
        root.removeHandler(h)

    root.setLevel(level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    if silent_libs:
        for noisy in ("urllib3", "requests", "spotipy", "lyricsgenius"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    return root


def get_logger(name: str) -> logging.Logger:
    """Module-level logger factory."""
    return logging.getLogger(name)


# --------------------------------------------------------------------------- #
# Timing                                                                      #
# --------------------------------------------------------------------------- #
@contextmanager
def timer(label: str, logger: Optional[logging.Logger] = None) -> Iterator[None]:
    """Context manager that logs elapsed wall-time for a block of work.

    Example:
        with timer("loading dataset"):
            df = pd.read_csv(big_file)
    """
    log = logger or logging.getLogger(__name__)
    start = time.perf_counter()
    log.info("⏱  start: %s", label)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log.info("✓  done : %s  (%.2fs)", label, elapsed)


# --------------------------------------------------------------------------- #
# IO                                                                          #
# --------------------------------------------------------------------------- #
def safe_to_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write parquet, creating parent dirs and falling back to CSV on failure.

    Parquet is ~10x smaller and ~5x faster to load than CSV for our shape of
    data, but it requires pyarrow. If pyarrow is missing we degrade gracefully.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:  # pragma: no cover
        logging.warning("parquet write failed (%s); falling back to CSV", exc)
        df.to_csv(path.with_suffix(".csv"), index=False)


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    """Read a parquet/csv/tsv based on extension."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    if suffix in (".tsv", ".tab"):
        return pd.read_csv(path, sep="\t", **kwargs)
    raise ValueError(f"unsupported extension: {suffix}")


def memory_usage_mb(df: pd.DataFrame) -> float:
    """Approximate memory footprint of a DataFrame in megabytes."""
    return df.memory_usage(deep=True).sum() / (1024 ** 2)


def assert_no_nulls(df: pd.DataFrame, columns: list[str]) -> None:
    """Defensive check used at stage boundaries in the pipeline."""
    bad = [c for c in columns if df[c].isna().any()]
    if bad:
        raise ValueError(f"unexpected nulls in columns: {bad}")
