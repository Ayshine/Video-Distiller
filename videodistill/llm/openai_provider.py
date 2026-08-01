"""OpenAI implementation of :class:`~videodistill.llm.base.LLMClient`.

This is the only module in the codebase allowed to import the ``openai`` SDK.
The import is deferred into ``__init__`` so that importing pipeline code (and
running tests with a mocked client) never requires the SDK or an API key.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from videodistill.errors import ProviderError
from videodistill.llm.base import ASRResult, ASRSegment, LLMMessage


class OpenAIProvider:
    """Talks to the OpenAI Chat Completions and Embeddings APIs."""

    def __init__(self, api_key: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ProviderError(
                "The 'openai' package is required for OpenAIProvider. "
                "Install project dependencies with `uv sync`."
            ) from exc

        if not api_key:
            raise ProviderError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add "
                "your key, or export OPENAI_API_KEY in the environment."
            )
        # A full lecture makes hundreds of vision calls against a per-minute
        # token limit; let the SDK ride out 429s with backoff rather than fail.
        # A per-request timeout is essential: without it, one stalled whisper/
        # chat call hangs the whole stage indefinitely (a lecture once idled ~1h
        # stuck on a single transcribe chunk). 120s is ample for a 10-min chunk;
        # a true hang fails and retries instead of freezing.
        self._client = OpenAI(api_key=api_key, max_retries=6, timeout=120.0)

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        # Typed as Any: OpenAI's message param is a union of TypedDicts that a
        # plain dict literal cannot match structurally.
        payload: list[Any] = [{"role": m.role, "content": m.content} for m in messages]
        resp = self._client.chat.completions.create(
            model=model,
            messages=payload,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def vision(
        self,
        prompt: str,
        image_path: Path,
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        data_url = _encode_image(image_path)
        content: list[Any] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
        payload: list[Any] = [{"role": "user", "content": content}]
        resp = self._client.chat.completions.create(
            model=model,
            messages=payload,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        resp = self._client.embeddings.create(model=model, input=texts)
        return [item.embedding for item in resp.data]

    def transcribe(
        self,
        audio_path: Path,
        *,
        model: str,
        prompt: str | None = None,
        language: str | None = None,
    ) -> ASRResult:
        # verbose_json + segment granularity gives per-segment timestamps, which
        # alignment needs. (The gpt-4o-*-transcribe models don't support it.)
        # Only pass `language` when set — omitting it keeps auto-detection.
        extra: dict[str, Any] = {"language": language} if language else {}
        with open(audio_path, "rb") as handle:
            resp = self._client.audio.transcriptions.create(
                model=model,
                file=handle,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
                prompt=prompt or "",
                **extra,
            )
        segments = [
            ASRSegment(
                start=float(seg.start),
                end=float(seg.end),
                text=seg.text,
                words=[],
            )
            for seg in (getattr(resp, "segments", None) or [])
        ]
        return ASRResult(
            language=getattr(resp, "language", "unknown"), segments=segments
        )


def _encode_image(image_path: Path) -> str:
    """Return a base64 data URL for an image file."""
    mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"
