"""Stage 4 — extract_visuals.

Turn each unique keyframe into structured text via one vision call (through the
provider abstraction only — never an SDK directly). The model returns strict
JSON that we validate into :class:`~videodistill.models.VisualExtract`:

- ``text`` is a VERBATIM transcription of on-screen text/code, indentation
  preserved — never a summary, never a guess.
- one bad frame never crashes the run: invalid JSON is retried once with the
  error fed back, then the frame is marked failed and skipped.

Cost is estimated from the keyframe count and guarded before any call is made
(``--max-cost``, default $2.00). Calls run under asyncio with at most 4 frames
in flight; identical frame bytes are served free by the response cache.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path

from PIL import Image
from pydantic import ValidationError

from videodistill.errors import CostLimitExceeded
from videodistill.llm.base import LLMClient
from videodistill.models import Keyframe, KeyframeSet, VisualExtract, VisualExtractSet
from videodistill.profile import DomainProfile

IS_STUB = False

logger = logging.getLogger("videodistill.stages.extract_visuals")

# Concurrency and cost tuning. Kept modest so a burst of large images doesn't
# blow through a per-minute token limit (the SDK still backs off on 429s).
MAX_FRAMES_IN_FLIGHT = 2
# Rough per-frame vision cost (gpt-4o, a single detailed image). Deliberately
# conservative — the guard should err toward stopping, not surprising you.
EST_COST_PER_FRAME_USD = 0.01
DEFAULT_MAX_COST_USD = 2.00


class _ParseError(Exception):
    """The model reply could not be turned into a VisualExtract."""


def estimate_cost(n_frames: int) -> float:
    """Estimated USD spend for ``n_frames`` vision calls."""
    return n_frames * EST_COST_PER_FRAME_USD


def enforce_cost_limit(n_frames: int, max_cost: float) -> float:
    """Raise :class:`CostLimitExceeded` if the estimate exceeds ``max_cost``.

    Returns the estimate when within budget. Callers run this *before*
    constructing a provider, so a cost abort needs no API key.
    """
    estimate = estimate_cost(n_frames)
    if estimate > max_cost:
        raise CostLimitExceeded(
            f"Estimated ${estimate:.2f} for {n_frames} frame(s) "
            f"({n_frames} x ${EST_COST_PER_FRAME_USD:.2f}/frame) exceeds "
            f"--max-cost ${max_cost:.2f}. Raise --max-cost to proceed."
        )
    return estimate


def run(
    keyframes: KeyframeSet,
    job_dir: Path,
    profile: DomainProfile,
    *,
    llm: LLMClient,
    model: str,
    max_cost: float = DEFAULT_MAX_COST_USD,
    image_max_width: int | None = None,
) -> VisualExtractSet:
    """Extract structured visuals for every keyframe into ``job_dir``.

    ``image_max_width`` downscales wide keyframes before the vision call (a
    cheaper token bucket with no loss on code); None sends full resolution.
    """
    frames = keyframes.keyframes
    enforce_cost_limit(len(frames), max_cost)

    extracts = asyncio.run(
        _extract_all(
            frames, llm=llm, model=model, profile=profile, max_width=image_max_width
        )
    )

    kept = [e for e in extracts if e is not None]
    failed = len(extracts) - len(kept)
    logger.info(
        "extract_visuals: %d frame(s) -> %d extract(s) (%d failed)",
        len(frames),
        len(kept),
        failed,
    )

    result = VisualExtractSet(extracts=kept)
    result.save(job_dir)
    return result


async def _extract_all(
    frames: list[Keyframe],
    *,
    llm: LLMClient,
    model: str,
    profile: DomainProfile,
    max_width: int | None,
) -> list[VisualExtract | None]:
    semaphore = asyncio.Semaphore(MAX_FRAMES_IN_FLIGHT)

    async def worker(keyframe: Keyframe) -> VisualExtract | None:
        async with semaphore:
            # The provider is synchronous; run it off the event loop.
            return await asyncio.to_thread(
                _extract_one, keyframe, llm, model, profile, max_width
            )

    return await asyncio.gather(*(worker(kf) for kf in frames))


def _prepare_image(source: Path, max_width: int | None) -> tuple[Path, bool]:
    """Return a path to feed the vision call, plus whether it is a temp file.

    Downscales to ``max_width`` when the image is wider; otherwise returns the
    original untouched. Resizing (not cropping) keeps all on-screen content and
    the aspect ratio — vision quality on code is unaffected down to ~1024px,
    while the smaller image lands in a cheaper token bucket. The source keyframe
    is never modified. On any imaging error, fall back to the original.
    """
    if not max_width:
        return source, False
    try:
        with Image.open(source) as img:
            if img.width <= max_width:
                return source, False
            height = round(img.height * max_width / img.width)
            resized = img.resize((max_width, height))
            fd, tmp_name = tempfile.mkstemp(prefix="vd_frame_", suffix=source.suffix)
            os.close(fd)
            tmp = Path(tmp_name)
            resized.save(tmp)
        return tmp, True
    except Exception as exc:  # noqa: BLE001 - imaging must never sink a frame
        logger.warning("extract_visuals: could not downscale %s: %s", source, exc)
        return source, False


def _extract_one(
    keyframe: Keyframe,
    llm: LLMClient,
    model: str,
    profile: DomainProfile,
    max_width: int | None,
) -> VisualExtract | None:
    """One frame: call, parse, retry once with the error, else mark failed."""
    timestamp = keyframe.timestamp
    image_path, is_temp = _prepare_image(Path(keyframe.image_path), max_width)
    try:
        prompt = _build_prompt(profile)
        last_error: Exception | None = None
        for _attempt in range(2):
            try:
                raw = llm.vision(prompt, image_path, model=model)
            except Exception as exc:  # noqa: BLE001 - one bad frame must not sink the run
                # e.g. a rate limit that survived the SDK's own backoff, or a
                # network error. Skip this frame rather than crashing the lecture.
                logger.warning(
                    "extract_visuals: frame at %.2fs vision call failed: %s",
                    timestamp,
                    exc,
                )
                return None
            try:
                return _parse(raw, timestamp)
            except _ParseError as exc:
                last_error = exc
                prompt = (
                    f"{_build_prompt(profile)}\n\nYour previous reply could not be "
                    f"used: {exc}. Return ONLY a single valid JSON object matching "
                    "the schema, with no surrounding text."
                )

        logger.warning(
            "extract_visuals: frame at %.2fs failed after retry: %s",
            timestamp,
            last_error,
        )
        return None
    finally:
        if is_temp:
            image_path.unlink(missing_ok=True)


def _build_prompt(profile: DomainProfile) -> str:
    if profile.code_languages:
        lang_hint = (
            'If kind is "code", set code_language to the programming language. '
            f"Prefer one of: {', '.join(profile.code_languages)}; another language "
            "is allowed only if the code is clearly not one of those."
        )
    else:
        lang_hint = (
            'If kind is "code", set code_language to the programming language; '
            "otherwise null."
        )
    return (
        "You are transcribing ONE frame from an informational video. "
        "Return ONLY a JSON object (no markdown, no commentary) with keys:\n"
        '  "kind": one of "slide", "code", "whiteboard", "other".\n'
        '  "text": a VERBATIM transcription of every readable character in the '
        "frame — code, bullet points, labels — preserving exact indentation, "
        "line breaks, and symbols. Do NOT summarize or paraphrase. IGNORE "
        "application window chrome — title bars, menu bars, toolbars, editor "
        "tabs, status bars, and the OS taskbar — transcribe only the actual "
        "document/editor/slide/whiteboard content. Use an empty string if "
        "nothing is readable. Never guess an unreadable region.\n"
        f'  "code_language": {lang_hint} Use null when kind is not "code".\n'
        '  "description": one short line describing what the frame shows.\n'
        '  "diagram": ONLY when the frame shows a hand-drawn DIAGRAM that has '
        "spatial STRUCTURE — shapes connected by arrows or lines, a memory "
        "layout (cells with addresses/values), a tree, a graph, or a flow. "
        "Reproduce it as ASCII art (+-| for boxes, -> or <-> for arrows), "
        "preserving the layout and labels and keeping any code in place. A plain "
        "list of words or terms, bullet points, a vocabulary/definition list, a "
        "heading, a table of terms, or ordinary text/code is NOT a diagram — use "
        'an empty string for those (and for slides and editor screenshots). '
        "Do NOT wrap a mere list of words in boxes."
    )


def _parse(raw: str, timestamp: float) -> VisualExtract:
    text = _strip_code_fence(raw)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _ParseError(f"not valid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise _ParseError("expected a JSON object")

    # The timestamp is authoritative from the keyframe, not the model.
    data["timestamp"] = timestamp
    try:
        return VisualExtract.model_validate(data)
    except ValidationError as exc:
        raise _ParseError(
            f"does not match the schema ({exc.error_count()} error(s))"
        ) from exc


def _strip_code_fence(raw: str) -> str:
    """Strip a leading ```json / trailing ``` fence if the model added one."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()
