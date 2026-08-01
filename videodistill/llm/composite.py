"""Route each LLM operation to a chosen backend provider.

Lets us run vision on one provider (cheap Gemini) while ASR, embeddings and text
completion stay on another (OpenAI). Each method simply delegates; the caching
decorator wraps the composite, so responses are still content-hashed per model.
"""

from __future__ import annotations

from pathlib import Path

from videodistill.llm.base import ASRResult, LLMClient, LLMMessage


class CompositeProvider:
    """Delegate each operation to ``default``, except vision to ``vision``."""

    def __init__(self, *, default: LLMClient, vision: LLMClient | None = None) -> None:
        self._default = default
        self._vision = vision or default

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        return self._default.complete(
            messages, model=model, temperature=temperature, max_tokens=max_tokens
        )

    def vision(
        self,
        prompt: str,
        image_path: Path,
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        return self._vision.vision(
            prompt,
            image_path,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        return self._default.embed(texts, model=model)

    def transcribe(
        self,
        audio_path: Path,
        *,
        model: str,
        prompt: str | None = None,
        language: str | None = None,
    ) -> ASRResult:
        return self._default.transcribe(
            audio_path, model=model, prompt=prompt, language=language
        )
