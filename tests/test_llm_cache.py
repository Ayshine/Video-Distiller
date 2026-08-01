"""Tests for the LLM provider abstraction and content-hash cache.

Uses a mock LLMClient (no network, no API key) to prove that CachedLLMClient
serves repeated requests from disk and re-calls the provider when the request
changes — for complete, vision, and embed.
"""

from __future__ import annotations

from pathlib import Path

from videodistill.llm.base import LLMMessage
from videodistill.llm.cache import CachedLLMClient


class _MockProvider:
    """Counts calls so we can assert cache hits/misses."""

    def __init__(self) -> None:
        self.complete_calls = 0
        self.vision_calls = 0
        self.embed_calls = 0
        self.embed_batch_sizes: list[int] = []

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        self.complete_calls += 1
        return f"reply-{self.complete_calls}"

    def vision(
        self,
        prompt: str,
        image_path: Path,
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        self.vision_calls += 1
        return f"vision-{self.vision_calls}"

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        self.embed_calls += 1
        self.embed_batch_sizes.append(len(texts))
        return [[float(len(t)), 1.0] for t in texts]


def test_complete_is_cached_by_content(tmp_path: Path) -> None:
    provider = _MockProvider()
    client = CachedLLMClient(provider, tmp_path / "cache")
    messages = [LLMMessage(role="user", content="What is a mutex?")]

    first = client.complete(messages, model="gpt-4o-mini")
    second = client.complete(messages, model="gpt-4o-mini")

    assert first == second
    assert provider.complete_calls == 1


def test_different_request_bypasses_cache(tmp_path: Path) -> None:
    provider = _MockProvider()
    client = CachedLLMClient(provider, tmp_path / "cache")

    client.complete([LLMMessage(role="user", content="A")], model="gpt-4o-mini")
    client.complete([LLMMessage(role="user", content="B")], model="gpt-4o-mini")

    assert provider.complete_calls == 2


def test_vision_cache_keys_on_image_bytes(tmp_path: Path) -> None:
    provider = _MockProvider()
    client = CachedLLMClient(provider, tmp_path / "cache")
    img = tmp_path / "frame.png"
    img.write_bytes(b"\x89PNG fake image")

    client.vision("Extract text", img, model="gpt-4o")
    client.vision("Extract text", img, model="gpt-4o")
    assert provider.vision_calls == 1

    img.write_bytes(b"\x89PNG different image")
    client.vision("Extract text", img, model="gpt-4o")
    assert provider.vision_calls == 2


def test_embed_caches_per_text(tmp_path: Path) -> None:
    provider = _MockProvider()
    client = CachedLLMClient(provider, tmp_path / "cache")

    first = client.embed(["alpha", "beta"], model="text-embedding-3-small")
    # Overlapping batch: only "gamma" is new, so the provider sees one text.
    second = client.embed(["alpha", "gamma"], model="text-embedding-3-small")

    assert first[0] == second[0]  # "alpha" served from cache
    assert provider.embed_batch_sizes == [2, 1]


def test_cache_survives_new_client_instance(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    provider = _MockProvider()
    messages = [LLMMessage(role="user", content="persist me")]

    CachedLLMClient(provider, cache_dir).complete(messages, model="gpt-4o-mini")
    CachedLLMClient(provider, cache_dir).complete(messages, model="gpt-4o-mini")

    assert provider.complete_calls == 1
