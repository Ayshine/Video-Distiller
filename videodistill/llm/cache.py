"""Content-hash response cache.

Wraps any :class:`~videodistill.llm.base.LLMClient` and stores each response
keyed by a hash of the full request (model, prompt, image bytes, params). During
development, re-running a stage then costs nothing and is deterministic. The
cache is a plain directory of JSON files, so it works locally and on S3-backed
storage later.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from videodistill.llm.base import (
    ASRResult,
    ASRSegment,
    ASRWord,
    LLMClient,
    LLMMessage,
)


def _hash_request(payload: dict[str, object]) -> str:
    """Stable hash of a request payload, independent of dict ordering."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _asr_to_payload(result: ASRResult) -> dict[str, Any]:
    return {
        "language": result.language,
        "segments": [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "words": [
                    {"start": w.start, "end": w.end, "text": w.text} for w in s.words
                ],
            }
            for s in result.segments
        ],
    }


def _asr_from_payload(payload: Any) -> ASRResult:
    return ASRResult(
        language=str(payload["language"]),
        segments=[
            ASRSegment(
                start=float(s["start"]),
                end=float(s["end"]),
                text=str(s["text"]),
                words=[
                    ASRWord(
                        start=float(w["start"]),
                        end=float(w["end"]),
                        text=str(w["text"]),
                    )
                    for w in s.get("words", [])
                ],
            )
            for s in payload["segments"]
        ],
    )


class CachedLLMClient:
    """Decorator that memoizes an inner client's responses on disk."""

    def __init__(self, inner: LLMClient, cache_dir: Path) -> None:
        self._inner = inner
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _read(self, key: str) -> object | None:
        path = self._cache_dir / f"{key}.json"
        if path.exists():
            payload: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
            return payload["response"]
        return None

    def _write(self, key: str, response: object) -> None:
        path = self._cache_dir / f"{key}.json"
        path.write_text(
            json.dumps({"response": response}, ensure_ascii=False),
            encoding="utf-8",
        )

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        key = _hash_request(
            {
                "op": "complete",
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
            }
        )
        cached = self._read(key)
        if cached is not None:
            return str(cached)
        response = self._inner.complete(
            messages, model=model, temperature=temperature, max_tokens=max_tokens
        )
        self._write(key, response)
        return response

    def vision(
        self,
        prompt: str,
        image_path: Path,
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        image_digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        key = _hash_request(
            {
                "op": "vision",
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "prompt": prompt,
                "image_digest": image_digest,
            }
        )
        cached = self._read(key)
        if cached is not None:
            return str(cached)
        response = self._inner.vision(
            prompt,
            image_path,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._write(key, response)
        return response

    def transcribe(
        self,
        audio_path: Path,
        *,
        model: str,
        prompt: str | None = None,
        language: str | None = None,
    ) -> ASRResult:
        audio_digest = hashlib.sha256(audio_path.read_bytes()).hexdigest()
        key = _hash_request(
            {
                "op": "transcribe",
                "model": model,
                "prompt": prompt,
                "language": language,
                "audio_digest": audio_digest,
            }
        )
        cached = self._read(key)
        if cached is not None:
            return _asr_from_payload(cached)
        result = self._inner.transcribe(
            audio_path, model=model, prompt=prompt, language=language
        )
        self._write(key, _asr_to_payload(result))
        return result

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        # Cache per-text so partially-overlapping batches reuse work.
        results: list[list[float]] = []
        missing: list[tuple[int, str]] = []
        for idx, text in enumerate(texts):
            key = _hash_request({"op": "embed", "model": model, "text": text})
            cached = self._read(key)
            if cached is not None:
                results.append([float(x) for x in cast(list[Any], cached)])
            else:
                results.append([])  # placeholder, filled below
                missing.append((idx, text))

        if missing:
            fresh = self._inner.embed([t for _, t in missing], model=model)
            for (idx, text), vector in zip(missing, fresh, strict=True):
                key = _hash_request({"op": "embed", "model": model, "text": text})
                self._write(key, vector)
                results[idx] = vector
        return results
