"""Tests for CompositeProvider routing and GeminiProvider guards (no SDK/key)."""

from __future__ import annotations

from pathlib import Path

import pytest

from videodistill.errors import ProviderError
from videodistill.llm.base import ASRResult, LLMMessage
from videodistill.llm.composite import CompositeProvider


class _Spy:
    """Records which operations it received."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[str] = []

    def complete(self, messages, *, model, temperature=0.2, max_tokens=None):
        self.calls.append("complete")
        return self.name

    def vision(self, prompt, image_path, *, model, temperature=0.2, max_tokens=None):
        self.calls.append("vision")
        return self.name

    def embed(self, texts, *, model):
        self.calls.append("embed")
        return [[0.0]]

    def transcribe(self, audio_path, *, model, prompt=None, language=None):
        self.calls.append("transcribe")
        return ASRResult(language="tr", segments=[])


def test_vision_routes_to_vision_backend_rest_to_default() -> None:
    default, vision = _Spy("default"), _Spy("vision")
    c = CompositeProvider(default=default, vision=vision)

    assert c.vision("p", Path("x.jpg"), model="gemini") == "vision"
    assert c.complete([LLMMessage(role="user", content="hi")], model="m") == "default"
    c.embed(["a"], model="m")
    c.transcribe(Path("a.wav"), model="whisper-1", language="tr")

    assert vision.calls == ["vision"]  # ONLY vision hit the vision backend
    assert default.calls == ["complete", "embed", "transcribe"]


def test_defaults_to_single_backend_when_no_vision_given() -> None:
    default = _Spy("default")
    c = CompositeProvider(default=default)
    c.vision("p", Path("x.jpg"), model="gpt-4o")
    assert default.calls == ["vision"]


def test_transcribe_forwards_language() -> None:
    default = _Spy("default")
    c = CompositeProvider(default=default)
    # should not raise; language kwarg is threaded through
    c.transcribe(Path("a.wav"), model="whisper-1", prompt="hi", language="tr")
    assert default.calls == ["transcribe"]


def test_gemini_requires_key() -> None:
    from videodistill.llm.gemini_provider import GeminiProvider

    with pytest.raises(ProviderError):
        GeminiProvider(api_key=None)


def test_gemini_non_vision_methods_raise() -> None:
    # Build without touching the SDK by stubbing the client attr.
    from videodistill.llm.gemini_provider import GeminiProvider

    g = GeminiProvider.__new__(GeminiProvider)  # bypass __init__/SDK
    with pytest.raises(ProviderError):
        g.embed(["a"], model="m")
    with pytest.raises(ProviderError):
        g.transcribe(Path("a.wav"), model="whisper-1")
    with pytest.raises(ProviderError):
        g.complete([LLMMessage(role="user", content="x")], model="m")
