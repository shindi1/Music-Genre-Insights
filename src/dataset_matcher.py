"""Fuzzy matching of songs across the Genius and Spotify datasets.

The core problem: Genius's lyrics dataset and Spotify's audio-features dataset
do not share keys. We must match on (artist, title), which is fragile because:

    Genius                          Spotify
    ─────────                       ─────────
    "Hey Jude"                      "Hey Jude - Remastered 2009"
    "Lose Yourself"                 "Lose Yourself - From '8 Mile' Soundtrack"
    "U + Ur Hand"                   "U+Ur Hand"
    "Beyoncé"                       "Beyonce"
    "feat." spelling drift          everywhere
    Multiple artists separated by   Single string with ";" or ","
        ", " vs ";" vs " & "

We solve this with a two-stage match:

    1. Normalize artist + title (lowercase, strip remaster suffixes, strip
       feat-credits, strip parens, ascii-fold, collapse whitespace).
    2. Block by normalized artist (only fuzzy-compare titles within the same
       artist), then use rapidfuzz.token_sort_ratio for the title comparison.

Blocking is crucial: a naive cross-product of 5M Genius rows × 114K Spotify rows
is 570B comparisons. Blocking by artist drops this to roughly len(left) average
comparisons — runs in minutes instead of weeks.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd
from rapidfuzz import fuzz, process
from tqdm.auto import tqdm
from unidecode import unidecode

from src.utils import get_logger

log = get_logger(__name__)


class DatasetMatcher:
    """Fuzzy-join two song DataFrames on (artist, title)."""

    # ---- regex patterns for normalization ------------------------------ #
    # Matches a `- <stuff>` suffix that begins with a year and/or one of the
    # known re-release keywords. Order of alternatives doesn't matter; we
    # consume to end-of-string once we see a hit.
    _REMASTER_KEYWORD = (
        r"remaster(ed)?|deluxe(\s+edition)?|live|edit(ion)?|version|"
        r"mix|mono|stereo|radio\s+edit|extended|acoustic|instrumental|"
        r"single\s+version|album\s+version|bonus\s+track|original\s+(mix|version)|"
        r"anniversary(\s+edition)?|expanded(\s+edition)?|reissue|"
        # Soundtrack annotations: "- From '8 Mile' Soundtrack", "- From The Movie", etc.
        r"from\s+['\"\w]"
    )
    _REMASTER_SUFFIX = re.compile(
        # Either "<keyword> ..." after the dash, OR "YYYY [keyword] ..." after
        # the dash (covers "Bohemian Rhapsody - 2011 Mix", "...- 2009 Remaster").
        rf"\s*[-–—]\s*(?:(?:{_REMASTER_KEYWORD})|\d{{4}}(?:\s+(?:{_REMASTER_KEYWORD}))?).*$",
        re.IGNORECASE,
    )
    _FEAT_CREDIT = re.compile(
        r"\s*[\(\[]?\s*(feat\.?|featuring|with|ft\.?|w/)\s+[^)\]]*[\)\]]?",
        re.IGNORECASE,
    )
    _PARENS = re.compile(r"\s*[\(\[][^)\]]*[\)\]]")
    _NON_ALNUM = re.compile(r"[^\w\s]")
    _MULTISPACE = re.compile(r"\s+")
    # Artist separators: comma / semicolon / & / "and" / " x " (with mandatory
    # whitespace on BOTH sides — otherwise we'd split "Lil Nas X" on the final X
    # or "Madonna" on its embedded letters).
    _ARTIST_SEPARATORS = re.compile(
        r"(?:\s*[,;]\s*|\s*&\s*|\s+and\s+|\s+x\s+|\s+vs\.?\s+)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        threshold: int = 87,
        scorer: Callable = fuzz.token_sort_ratio,
        primary_artist_only: bool = True,
    ) -> None:
        """
        Args:
            threshold: min similarity score (0-100) to accept a match.
            scorer: rapidfuzz scorer. token_sort_ratio handles word-order
                differences ("X feat. Y" vs "Y feat. X"). For stricter matching
                use fuzz.ratio.
            primary_artist_only: if True, only the first artist in a multi-
                artist string is used for blocking. Recommended.
        """
        self.threshold = threshold
        self.scorer = scorer
        self.primary_artist_only = primary_artist_only

    # ----------------------------------------------------------- normalizers
    @classmethod
    def normalize_title(cls, text: Optional[str]) -> str:
        """Strip remaster suffixes, feat-credits, parens, punctuation."""
        if not isinstance(text, str):
            return ""
        s = text.lower()
        s = cls._REMASTER_SUFFIX.sub("", s)
        s = cls._FEAT_CREDIT.sub("", s)
        s = cls._PARENS.sub("", s)
        s = unidecode(s)             # "Beyoncé" -> "Beyonce"
        s = cls._NON_ALNUM.sub(" ", s)
        s = cls._MULTISPACE.sub(" ", s).strip()
        return s

    @classmethod
    def normalize_artist(cls, text: Optional[str], primary_only: bool = True) -> str:
        """Normalize artist name, optionally taking only the primary artist."""
        if not isinstance(text, str):
            return ""
        s = text.lower()
        if primary_only:
            # Split on common separators, keep the first
            parts = cls._ARTIST_SEPARATORS.split(s)
            if parts:
                s = parts[0]
        s = unidecode(s)
        s = cls._NON_ALNUM.sub(" ", s)
        s = cls._MULTISPACE.sub(" ", s).strip()
        return s

    # ----------------------------------------------------------------- match
    def match(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        left_keys: tuple = ("artist", "title"),
        right_keys: tuple = ("artists", "track_name"),
        suffixes: tuple = ("_left", "_right"),
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Inner-join two DataFrames by fuzzy (artist, title) match.

        Returns a new DataFrame with columns from both sides and a
        `match_score` column (0-100). Rows that don't meet `self.threshold`
        are dropped.
        """
        l_artist, l_title = left_keys
        r_artist, r_title = right_keys

        log.info("normalizing left side (%d rows)...", len(left))
        left = left.copy()
        left["_artist_norm"] = [
            self.normalize_artist(a, self.primary_artist_only) for a in left[l_artist]
        ]
        left["_title_norm"] = [self.normalize_title(t) for t in left[l_title]]

        log.info("normalizing right side (%d rows)...", len(right))
        right = right.copy()
        right["_artist_norm"] = [
            self.normalize_artist(a, self.primary_artist_only) for a in right[r_artist]
        ]
        right["_title_norm"] = [self.normalize_title(t) for t in right[r_title]]

        # ---- build artist blocks on the right side ---- #
        log.info("building artist blocks...")
        right_index_by_artist: Dict[str, List[int]] = defaultdict(list)
        for idx, artist in enumerate(right["_artist_norm"].values):
            if artist:
                right_index_by_artist[artist].append(idx)
        log.info("%d distinct normalized artists on right side", len(right_index_by_artist))

        # ---- per-row fuzzy match within block ---- #
        right_titles_arr = right["_title_norm"].values

        matches: List[tuple] = []  # (left_idx, right_idx, score)
        iterator = enumerate(zip(left["_artist_norm"].values, left["_title_norm"].values))
        if show_progress:
            iterator = tqdm(iterator, total=len(left), desc="fuzzy matching")

        for li, (l_art, l_tit) in iterator:
            if not l_art or not l_tit:
                continue
            candidates = right_index_by_artist.get(l_art)
            if not candidates:
                continue
            # rapidfuzz.process.extractOne is C-optimized; much faster than
            # iterating in Python.
            choice_strings = [right_titles_arr[i] for i in candidates]
            best = process.extractOne(
                l_tit,
                choice_strings,
                scorer=self.scorer,
                score_cutoff=self.threshold,
            )
            if best is None:
                continue
            _matched_str, score, local_idx = best
            ri = candidates[local_idx]
            matches.append((li, ri, score))

        log.info("matched %d / %d left-side rows above threshold %d",
                 len(matches), len(left), self.threshold)

        if not matches:
            log.warning("no matches found — returning empty frame")
            return pd.DataFrame()

        # ---- assemble output frame ---- #
        m_df = pd.DataFrame(matches, columns=["_li", "_ri", "match_score"])

        left_part = left.iloc[m_df["_li"].values].reset_index(drop=True)
        right_part = right.iloc[m_df["_ri"].values].reset_index(drop=True)

        # Avoid duplicate column collisions
        overlap = set(left_part.columns) & set(right_part.columns)
        left_part = left_part.rename(columns={c: f"{c}{suffixes[0]}" for c in overlap})
        right_part = right_part.rename(columns={c: f"{c}{suffixes[1]}" for c in overlap})

        out = pd.concat(
            [left_part, right_part, m_df["match_score"].reset_index(drop=True)],
            axis=1,
        )

        # Drop helper columns; they'd just confuse downstream code.
        out = out.drop(columns=[c for c in out.columns if c.startswith("_artist_norm") or c.startswith("_title_norm")], errors="ignore")
        return out

    # ----------------------------------------------------------- diagnostics
    @staticmethod
    def match_quality_report(matched: pd.DataFrame) -> pd.DataFrame:
        """Histogram of match scores, helpful for picking the threshold."""
        if "match_score" not in matched.columns:
            raise ValueError("missing match_score column")
        bins = [0, 70, 80, 85, 90, 95, 100, 101]
        labels = ["<70", "70-79", "80-84", "85-89", "90-94", "95-99", "100"]
        cuts = pd.cut(matched["match_score"], bins=bins, labels=labels, right=False)
        report = cuts.value_counts().sort_index().to_frame("n")
        report["pct"] = (report["n"] / report["n"].sum() * 100).round(2)
        return report
