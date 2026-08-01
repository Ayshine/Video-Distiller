"""The provider-agnostic LLM interface.

Three capabilities cover the whole system: ``complete`` (distill), ``vision``
over a single image (extract_visuals), and ``embed`` (knowledge base). Concrete
providers live alongside this module; stages depend only on the
:class:`LLMClient` protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class LLMMessage:
    """A single chat message in a provider-neutral shape."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True)
class ASRWord:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class ASRSegment:
    """A timestamped transcript segment from a speech-to-text provider."""

    start: float
    end: float
    text: str
    words: list[ASRWord]


@dataclass(frozen=True)
class ASRResult:
    """Detected language plus timestamped segments."""

    language: str
    segments: list[ASRSegment]


@runtime_checkable
class LLMClient(Protocol):
    """Minimal interface every provider must implement.

    Implementations should be stateless per call and safe to share across
    stages. ``model`` is passed explicitly so one client can serve the vision
    model, the cheaper distill model, and the embedding model.
    """

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        """Return the assistant's text reply for a text-only conversation."""
        ...

    def vision(
        self,
        prompt: str,
        image_path: Path,
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        """Return the model's text reply given a prompt and one image."""
        ...

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...

    def transcribe(
        self,
        audio_path: Path,
        *,
        model: str,
        prompt: str | None = None,
        language: str | None = None,
    ) -> ASRResult:
        """Transcribe an audio file to timestamped segments.

        ``language`` is an ISO-639-1 code (e.g. ``"tr"``). When given, the
        model is told the language instead of guessing — this prevents the
        silent content loss that happens when auto-detection picks wrong.
        """
        ...
