"""Production-grade lyrics text cleaner.

Genius-scraped lyrics are messy. This module handles the realities you'll see:

    [Verse 1: Eminem]
    His palms are sweaty, knees weak, arms are heavy
    (Mom&#x27;s spaghetti) ← HTML entity from web scrape
    He&#39;s nervous, but on the surface he looks calm and ready

    [Chorus]
    ...

    Embed
    23

The cleaner removes section markers, fixes encoding, normalizes whitespace,
and produces text suitable for either TF-IDF (CountVectorizer / TfidfVectorizer)
or transformer tokenization. It is *deliberately* conservative — it does not
remove punctuation, lowercase, or stem, leaving those decisions to downstream
modeling code where they belong.
"""
from __future__ import annotations

import html
import re
import unicodedata
from typing import Iterable, List, Optional

import ftfy

from src.utils import get_logger

log = get_logger(__name__)


class LyricsCleaner:
    """Stateless lyrics cleaner. One instance can clean millions of strings."""

    # ----- precompiled regex (compile once per process, not per call) ----- #
    _SECTION_MARKER = re.compile(r"\[[^\]]{0,200}\]")  # [Chorus], [Verse 1: X]
    _PARENTHETICAL = re.compile(r"\([^)]{0,200}\)")
    _PRODUCER_TAG = re.compile(
        r"\b(produced\s+by|prod(uced)?\.?\s+by)\b[^\n,.]{0,80}",
        re.IGNORECASE,
    )
    # Genius-specific cruft that survives web scraping
    _GENIUS_EMBED = re.compile(r"\d*\s*Embed\s*$", re.IGNORECASE | re.MULTILINE)
    _YOU_MIGHT_LIKE = re.compile(r"You\s+might\s+also\s+like", re.IGNORECASE)
    _SEE_LIVE_TICKET = re.compile(r"See\s+[^\n]+Live\s+Get\s+tickets[^\n]*", re.IGNORECASE)
    _CONTRIBUTORS = re.compile(r"\d+\s+Contributors?[^\n]*\n", re.IGNORECASE)
    _TRANSLATION_NOTE = re.compile(r"Translations?\s*\n", re.IGNORECASE)

    _MULTI_NEWLINE = re.compile(r"\n{3,}")
    _MULTI_SPACE = re.compile(r"[ \t]{2,}")
    _SPACE_BEFORE_NEWLINE = re.compile(r" *\n *")

    def __init__(
        self,
        remove_section_markers: bool = True,
        remove_parentheticals: bool = False,
        remove_producer_tags: bool = True,
        fix_encoding: bool = True,
        normalize_unicode: bool = True,
        lowercase: bool = False,
    ) -> None:
        self.remove_section_markers = remove_section_markers
        self.remove_parentheticals = remove_parentheticals
        self.remove_producer_tags = remove_producer_tags
        self.fix_encoding = fix_encoding
        self.normalize_unicode = normalize_unicode
        self.lowercase = lowercase

    # ------------------------------------------------------------------ API #
    def clean(self, text: Optional[str]) -> str:
        """Clean a single lyric string. Empty / non-string input -> ''."""
        if not isinstance(text, str) or not text:
            return ""

        # 1. Encoding fixes (mojibake like "Iâ€™ll" -> "I'll")
        if self.fix_encoding:
            text = ftfy.fix_text(text)

        # 2. HTML entities (&amp; -> &, &#x27; -> ')
        text = html.unescape(text)

        # 3. Genius web-scrape artifacts (must run before section markers
        #    because some of these are above [Intro])
        text = self._CONTRIBUTORS.sub("", text)
        text = self._TRANSLATION_NOTE.sub("", text)
        text = self._YOU_MIGHT_LIKE.sub("", text)
        text = self._SEE_LIVE_TICKET.sub("", text)
        text = self._GENIUS_EMBED.sub("", text)

        # 4. Section markers
        if self.remove_section_markers:
            text = self._SECTION_MARKER.sub("", text)

        # 5. Parentheticals (off by default — ad-libs are signal not noise)
        if self.remove_parentheticals:
            text = self._PARENTHETICAL.sub("", text)

        # 6. Producer tags
        if self.remove_producer_tags:
            text = self._PRODUCER_TAG.sub("", text)

        # 7. Unicode normalization (NFKC folds compatibility chars)
        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)

        # 8. Whitespace normalization
        text = self._MULTI_SPACE.sub(" ", text)
        text = self._SPACE_BEFORE_NEWLINE.sub("\n", text)
        text = self._MULTI_NEWLINE.sub("\n\n", text)

        # 9. Optional lowercasing — leave to downstream by default
        if self.lowercase:
            text = text.lower()

        return text.strip()

    def clean_batch(self, texts: Iterable[Optional[str]]) -> List[str]:
        """Vectorized convenience wrapper. Pure Python; fast enough for ~1M rows."""
        return [self.clean(t) for t in texts]

    # ----------------------------------------------------------- diagnostics #
    @staticmethod
    def word_count(text: str) -> int:
        """Quick word count used for length filtering. Splits on whitespace."""
        if not isinstance(text, str):
            return 0
        return len(text.split())

    @staticmethod
    def line_count(text: str) -> int:
        """Number of non-empty lines."""
        if not isinstance(text, str):
            return 0
        return sum(1 for line in text.splitlines() if line.strip())
