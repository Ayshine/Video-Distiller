"""Configuration resolved from environment variables with local defaults.

No global mutable state: call :func:`load_config` where you need settings and
pass the result explicitly. Keeps stages pure and AWS-portable (the same code
reads config from a Fargate task's environment).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Resolved runtime configuration."""

    openai_api_key: str | None
    gemini_api_key: str | None
    # Which backend serves the vision call: "openai" (gpt-4o) or "gemini"
    # (much cheaper, equal fidelity on code frames). ASR/embeddings/text stay
    # on OpenAI regardless.
    vision_provider: str
    asr_model: str
    vision_model: str
    distill_model: str
    embed_model: str
    # Max width (px) a keyframe is downscaled to before the vision call. Vision
    # models tile/downscale internally, so a 1920px screenshot and a 1024px one
    # cost the same tokens above ~1024 — but 1024 lands in a cheaper tile bucket
    # (~16% less) with no measurable loss on code. None = send full resolution.
    vision_max_width: int | None
    cache_dir: Path
    kb_dir: Path
    # Which vector-store backend the KB uses. "qdrant" (embedded, shipped) by
    # default; add pgvector/chroma by implementing VectorStore in kb/store.py.
    vector_store: str


def _int_or_none(value: str) -> int | None:
    """Parse a positive int; treat 0/empty/'none'/'off' as disabled (None)."""
    v = value.strip().lower()
    if v in {"", "0", "none", "off"}:
        return None
    return int(v)


def load_config() -> Config:
    """Build a :class:`Config` from the process environment."""
    vision_provider = os.environ.get("VIDEODISTILL_VISION_PROVIDER", "openai").lower()
    # Default the vision model to the right family for the chosen backend, so
    # switching provider needs only the one env var.
    default_vision_model = (
        "gemini-3.1-flash-lite" if vision_provider == "gemini" else "gpt-4o"
    )
    return Config(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        gemini_api_key=os.environ.get("GEMINI_API_KEY"),
        vision_provider=vision_provider,
        asr_model=os.environ.get("VIDEODISTILL_ASR_MODEL", "small"),
        vision_model=os.environ.get("VIDEODISTILL_VISION_MODEL", default_vision_model),
        distill_model=os.environ.get("VIDEODISTILL_DISTILL_MODEL", "gpt-4o-mini"),
        embed_model=os.environ.get(
            "VIDEODISTILL_EMBED_MODEL", "text-embedding-3-small"
        ),
        vision_max_width=_int_or_none(
            os.environ.get("VIDEODISTILL_VISION_MAX_WIDTH", "1024")
        ),
        cache_dir=Path(os.environ.get("VIDEODISTILL_CACHE_DIR", ".cache")),
        kb_dir=Path(os.environ.get("VIDEODISTILL_KB_DIR", ".kb")),
        vector_store=os.environ.get("VIDEODISTILL_VECTOR_STORE", "qdrant").lower(),
    )
