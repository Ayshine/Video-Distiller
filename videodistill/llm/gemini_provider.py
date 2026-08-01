"""Google Gemini implementation of the vision half of :class:`LLMClient`.

The only module allowed to import the ``google-genai`` SDK. The import is
deferred into ``__init__`` so importing pipeline code (or running mocked tests)
never needs the SDK or a key.

Gemini is used as a cheap, accurate drop-in for the vision call — ~5x cheaper
than gpt-4o on clean IDE/code frames with equal or better fidelity. ASR,
embeddings and text completion stay on OpenAI (see :class:`CompositeProvider`),
so only ``vision`` is implemented here; the rest raise a clear error.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

from videodistill.errors import ProviderError
from videodistill.llm.base import ASRResult, LLMMessage


class GeminiProvider:
    """Talks to the Google Gemini API for vision."""

    def __init__(self, api_key: str | None = None) -> None:
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ProviderError(
                "The 'google-genai' package is required for GeminiProvider. "
                "Install project dependencies with `uv sync`."
            ) from exc

        if not api_key:
            raise ProviderError(
                "GEMINI_API_KEY is not set. Add it to .env or the environment "
                "to use the Gemini vision backend."
            )
        from google.genai import types

        # A per-request timeout (ms) so a stalled call fails fast and the
        # extract stage can skip/retry it — rather than one hung HTTP request
        # freezing the whole lecture (it once idled for ~2 hours).
        self._client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=60_000)
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
        from google.genai import types

        mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        config = types.GenerateContentConfig(
            temperature=temperature, max_output_tokens=max_tokens
        )
        # HARD-disable "thinking": reading text off a screen needs no reasoning,
        # and reasoning tokens bill at the OUTPUT rate (they made a lecture ~10x
        # pricier). budget=0 forces it fully off on gemini-3.1-flash-lite —
        # verified thoughts_token_count=0. (A few heavier models reject 0 and
        # would instead need a small positive cap like 128.)
        config.thinking_config = types.ThinkingConfig(thinking_budget=0)
        contents: list[Any] = [
            prompt,
            types.Part.from_bytes(data=image_path.read_bytes(), mime_type=mime),
        ]
        resp = self._client.models.generate_content(
            model=model, contents=contents, config=config
        )
        return resp.text or ""

    # --- Not used by the composite (OpenAI handles these); fail loudly. ---
    def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        raise ProviderError(
            "GeminiProvider implements vision only; use OpenAI for text."
        )

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        raise ProviderError(
            "GeminiProvider implements vision only; use OpenAI for embeddings."
        )

    def transcribe(
        self,
        audio_path: Path,
        *,
        model: str,
        prompt: str | None = None,
        language: str | None = None,
    ) -> ASRResult:
        raise ProviderError(
            "GeminiProvider implements vision only; use OpenAI for ASR."
        )
