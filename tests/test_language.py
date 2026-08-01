"""Tests for language resources and their use in alignment."""

from __future__ import annotations

from videodistill.language import available_languages, load_stopwords
from videodistill.stages.align import _content_words


def test_english_stopwords_load() -> None:
    words = load_stopwords("en")
    assert "the" in words and "and" in words
    assert isinstance(words, frozenset)


def test_turkish_stopwords_load() -> None:
    words = load_stopwords("tr")
    assert "ve" in words and "bir" in words  # Turkish "and" / "a"
    assert "the" not in words  # not the English list


def test_language_code_is_case_insensitive() -> None:
    assert load_stopwords("TR") == load_stopwords("tr")


def test_unknown_language_returns_empty() -> None:
    assert load_stopwords("xx") == frozenset()


def test_both_languages_are_available() -> None:
    langs = available_languages()
    assert "en" in langs and "tr" in langs


def test_content_words_filter_by_language() -> None:
    tr = load_stopwords("tr")
    # "ve" (and) and "bir" (a) are dropped; the Turkish word stays intact
    # (an ASCII tokenizer would split "işaretçi" at ş/ç).
    assert _content_words("ve bir işaretçi", tr) == {"işaretçi"}


def test_content_words_without_stopwords_keeps_everything() -> None:
    assert _content_words("ve bir işaretçi", frozenset()) == {
        "ve",
        "bir",
        "işaretçi",
    }
