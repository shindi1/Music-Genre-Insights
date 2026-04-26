"""Tests for LyricsCleaner."""
from __future__ import annotations

import pytest

from src.lyrics_cleaner import LyricsCleaner


class TestLyricsCleaner:
    def setup_method(self):
        self.cleaner = LyricsCleaner()

    def test_empty_input(self):
        assert self.cleaner.clean("") == ""
        assert self.cleaner.clean(None) == ""
        assert self.cleaner.clean(123) == ""  # non-string

    def test_section_markers_removed(self):
        text = "[Chorus]\nHello world\n[Verse 1: Eminem]\nGoodbye"
        result = self.cleaner.clean(text)
        assert "[Chorus]" not in result
        assert "[Verse 1" not in result
        assert "Hello world" in result
        assert "Goodbye" in result

    def test_html_entities_decoded(self):
        text = "It&#39;s &amp; you&#x27;re"
        result = self.cleaner.clean(text)
        assert "&#39;" not in result
        assert "&amp;" not in result
        assert "It's" in result
        assert "&" in result

    def test_genius_artifacts_removed(self):
        text = "Lose Yourself Lyrics\nThe content\nYou might also like\n23Embed"
        result = self.cleaner.clean(text)
        assert "You might also like" not in result
        assert "Embed" not in result
        assert "The content" in result

    def test_producer_tag_removed(self):
        text = "Some lyrics here\nProduced by Metro Boomin\nMore lyrics"
        result = self.cleaner.clean(text)
        assert "Produced by Metro Boomin" not in result
        assert "Some lyrics here" in result
        assert "More lyrics" in result

    def test_parentheticals_kept_by_default(self):
        text = "Look (yeah) at me (uh-huh)"
        result = self.cleaner.clean(text)
        assert "(yeah)" in result
        assert "(uh-huh)" in result

    def test_parentheticals_removed_when_configured(self):
        cleaner = LyricsCleaner(remove_parentheticals=True)
        text = "Look (yeah) at me (uh-huh)"
        result = cleaner.clean(text)
        assert "(yeah)" not in result
        assert "(uh-huh)" not in result

    def test_unicode_normalization(self):
        # Compatibility character — should fold under NFKC
        text = "café"  # already normal NFC; test mojibake separately
        result = self.cleaner.clean(text)
        assert "café" in result.lower()

    def test_mojibake_fixed(self):
        text = "Iâ€™m heading home"  # classic UTF-8-as-cp1252 mojibake
        result = self.cleaner.clean(text)
        assert "I'm" in result or "I’m" in result

    def test_whitespace_normalized(self):
        text = "Hello    world\n\n\n\nfoo"
        result = self.cleaner.clean(text)
        assert "    " not in result  # multi-space collapsed
        # Triple+ newlines collapse to double
        assert "\n\n\n" not in result

    def test_lowercasing_off_by_default(self):
        text = "Hello World"
        assert self.cleaner.clean(text) == "Hello World"

    def test_lowercasing_when_configured(self):
        cleaner = LyricsCleaner(lowercase=True)
        assert cleaner.clean("Hello World") == "hello world"

    def test_complex_realistic_example(self, messy_lyrics):
        """Integration test: full Genius-scraped text."""
        result = self.cleaner.clean(messy_lyrics)
        assert "[Intro]" not in result
        assert "[Verse 1: Eminem]" not in result
        assert "Contributors" not in result
        assert "Embed" not in result
        assert "&#x27;" not in result
        assert "Look, if you had one shot" in result
        assert "His palms are sweaty" in result

    def test_batch_processing(self):
        texts = ["[Chorus]\nfoo", "&amp; bar", None, ""]
        results = self.cleaner.clean_batch(texts)
        assert len(results) == 4
        assert results[0] == "foo"
        assert "&" in results[1]
        assert results[2] == ""
        assert results[3] == ""

    def test_word_count(self):
        assert LyricsCleaner.word_count("hello world foo") == 3
        assert LyricsCleaner.word_count("") == 0
        assert LyricsCleaner.word_count(None) == 0
        assert LyricsCleaner.word_count("one") == 1

    def test_line_count(self):
        text = "line one\nline two\n\nline three"
        assert LyricsCleaner.line_count(text) == 3
        assert LyricsCleaner.line_count("") == 0


@pytest.mark.parametrize("dirty,clean_substring", [
    ("[Bridge: Drake] (yeah)\nfoo bar",     "foo bar"),
    ("23 Contributors\nSome song Lyrics\nthe actual content",   "the actual content"),
    ("&amp;&amp;&amp;",                                          "&&&"),
    ("\n\n\n\n\nfoo\n\n\n\n\nbar",                              "foo\n\nbar"),
])
def test_parametric_cleaning(dirty, clean_substring):
    cleaner = LyricsCleaner()
    assert clean_substring in cleaner.clean(dirty)
