"""Cade's data pipeline for music sentiment analysis & genre classification.

Public API:
    from src import (
        LyricsCleaner,
        DatasetMatcher,
        GenreMapper,
        LanguageFilter,
        FeatureEngineer,
        ClassBalancer,
        DataSplitter,
        Pipeline,
    )
"""
from src.lyrics_cleaner import LyricsCleaner
from src.dataset_matcher import DatasetMatcher
from src.genre_mapper import GenreMapper
from src.language_filter import LanguageFilter
from src.feature_engineering import FeatureEngineer
from src.balancer import ClassBalancer
from src.splitter import DataSplitter
from src.pipeline import Pipeline

__version__ = "1.0.0"
__all__ = [
    "LyricsCleaner",
    "DatasetMatcher",
    "GenreMapper",
    "LanguageFilter",
    "FeatureEngineer",
    "ClassBalancer",
    "DataSplitter",
    "Pipeline",
]
