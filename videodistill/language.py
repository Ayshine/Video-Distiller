"""Language-specific resources.

Orthogonal to domain profiles: a *profile* is a subject (its jargon), a
*language* is the natural language spoken (its stopwords and, later, other
rules). The language is detected by the transcribe stage (faster-whisper) and
stored on the Transcript, so downstream stages pick the right rules by code
(e.g. ``en``, ``tr``). Resources live in ``languages/<code>/``.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

# languages/ sits at the repo root, next to the videodistill package.
_LANGUAGES_DIR = Path(__file__).resolve().parent.parent / "languages"

# ASR backends disagree on how they name a language: faster-whisper returns the
# ISO code ("tr"), OpenAI's whisper-1 returns the full name ("turkish"). We key
# everything on the ISO code, so map the names we expect to see.
_LANGUAGE_ALIASES = {
    "english": "en",
    "turkish": "tr",
    "german": "de",
    "french": "fr",
    "spanish": "es",
    "italian": "it",
    "portuguese": "pt",
    "russian": "ru",
    "arabic": "ar",
}


def normalize_language(value: str) -> str:
    """Map a language name or code to an ISO 639-1 code (best effort)."""
    key = value.strip().lower()
    return _LANGUAGE_ALIASES.get(key, key)


@cache
def load_stopwords(language: str) -> frozenset[str]:
    """Stopwords for a language code (``en``, ``tr``, ...).

    Returns an empty set for an unknown language — overlap still works, it just
    keeps filler words. Reads ``languages/<code>/stopwords.txt`` (one lowercase
    word per line; ``#`` comments ignored).
    """
    path = _LANGUAGES_DIR / language.lower() / "stopwords.txt"
    if not path.exists():
        return frozenset()
    return frozenset(
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def available_languages() -> list[str]:
    """Language codes that ship with a stopwords file."""
    if not _LANGUAGES_DIR.is_dir():
        return []
    return sorted(
        p.name for p in _LANGUAGES_DIR.iterdir() if (p / "stopwords.txt").exists()
    )
