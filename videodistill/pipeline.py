"""Pipeline orchestration.

Knows the canonical stage order and how to wire each stage's typed input from
the job directory. Stages themselves stay ignorant of one another; this module
is the only place that knows the sequence. The active
:class:`~videodistill.profile.DomainProfile` is threaded into every stage.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from videodistill.config import Config
from videodistill.errors import VideoDistillError
from videodistill.llm.base import LLMClient
from videodistill.llm.cache import CachedLLMClient
from videodistill.models import (
    AlignedTimeline,
    DistilledNoteSet,
    KeyframeSet,
    RunStats,
    Transcript,
    VideoMeta,
    VisualExtractSet,
)
from videodistill.profile import DomainProfile
from videodistill.stages import (
    align,
    detect_scenes,
    distill,
    extract_visuals,
    ingest,
    render,
    transcribe,
)

# Canonical order.
STAGE_ORDER: list[str] = [
    "ingest",
    "transcribe",
    "detect_scenes",
    "extract_visuals",
    "align",
    "distill",
    "render",
]


@dataclass
class StageResult:
    name: str
    status: str  # "ran" | "stub" | "skipped"
    detail: str = ""


def _build_llm(config: Config) -> CachedLLMClient:
    """Construct the cached LLM client. Imported lazily so stages that do not
    use the LLM never require an API key.

    Vision optionally runs on Gemini (cheaper) while ASR/embeddings/text stay
    on OpenAI, routed through a CompositeProvider.
    """
    from videodistill.llm.openai_provider import OpenAIProvider

    openai = OpenAIProvider(api_key=config.openai_api_key)
    provider: LLMClient = openai
    if config.vision_provider == "gemini":
        from videodistill.llm.composite import CompositeProvider
        from videodistill.llm.gemini_provider import GeminiProvider

        gemini = GeminiProvider(api_key=config.gemini_api_key)
        provider = CompositeProvider(default=openai, vision=gemini)
    return CachedLLMClient(provider, config.cache_dir)


def run_single_stage(
    name: str,
    job_dir: Path,
    config: Config,
    profile: DomainProfile,
    *,
    max_cost: float = extract_visuals.DEFAULT_MAX_COST_USD,
) -> object:
    """Run one stage by name, loading its inputs from ``job_dir``.

    Raises :class:`~videodistill.errors.StageNotImplemented` for stub stages
    and :class:`VideoDistillError` for unknown names.
    """
    runners: dict[str, Callable[[], object]] = {
        "ingest": lambda: _ingest_from_meta(job_dir, profile),
        "transcribe": lambda: transcribe.run(
            VideoMeta.load(job_dir),
            job_dir,
            profile,
            model_size=config.asr_model,
            llm=(
                _build_llm(config)
                if config.asr_model in transcribe.OPENAI_ASR_MODELS
                else None
            ),
        ),
        "detect_scenes": lambda: detect_scenes.run(
            VideoMeta.load(job_dir), job_dir, profile
        ),
        "extract_visuals": lambda: _run_extract_visuals(
            job_dir, config, profile, max_cost
        ),
        "align": lambda: align.run(
            Transcript.load(job_dir),
            VisualExtractSet.load(job_dir),
            job_dir,
            profile,
        ),
        "distill": lambda: distill.run(
            AlignedTimeline.load(job_dir),
            job_dir,
            profile,
            llm=_build_llm(config),
            model=config.distill_model,
        ),
        "render": lambda: render.run(DistilledNoteSet.load(job_dir), job_dir, profile),
    }
    if name not in runners:
        raise VideoDistillError(
            f"Unknown stage '{name}'. Valid stages: {', '.join(STAGE_ORDER)}"
        )
    return runners[name]()


def _ingest_from_meta(job_dir: Path, profile: DomainProfile) -> VideoMeta:
    """Re-run ingest using the source path recorded in an existing VideoMeta."""
    meta = VideoMeta.load(job_dir)
    return ingest.run(Path(meta.source_path), job_dir, profile)


def _run_extract_visuals(
    job_dir: Path, config: Config, profile: DomainProfile, max_cost: float
) -> VisualExtractSet:
    """Guard cost before building a provider, then run extract_visuals."""
    keyframes = KeyframeSet.load(job_dir)
    # Enforce the budget first so a cost abort needs no API key.
    extract_visuals.enforce_cost_limit(len(keyframes.keyframes), max_cost)
    llm = _build_llm(config)
    return extract_visuals.run(
        keyframes,
        job_dir,
        profile,
        llm=llm,
        model=config.vision_model,
        max_cost=max_cost,
        image_max_width=config.vision_max_width,
    )


def process(
    video_path: Path,
    job_dir: Path,
    config: Config,
    profile: DomainProfile,
    *,
    max_cost: float = extract_visuals.DEFAULT_MAX_COST_USD,
) -> list[StageResult]:
    """Run the full pipeline end-to-end (ingest → render).

    Returns a per-stage report the CLI renders as a summary. Per-stage wall-clock
    is written to run_stats.json for the eval harness.
    """
    results: list[StageResult] = []
    timings: dict[str, float] = {}

    def timed(name: str, fn: Callable[[], object]) -> object:
        start = time.perf_counter()
        value = fn()
        timings[name] = time.perf_counter() - start
        return value

    # --- Stage 1: ingest (takes the raw video, not a job artifact) ---
    meta = cast(
        VideoMeta, timed("ingest", lambda: ingest.run(video_path, job_dir, profile))
    )
    results.append(
        StageResult(
            "ingest", "ran", f"{meta.duration_s:.1f}s, {meta.width}x{meta.height}"
        )
    )

    # One provider client for the whole run (ASR, vision, distill).
    llm = _build_llm(config)

    # --- Stage 2: transcribe ---
    transcript = cast(
        Transcript,
        timed(
            "transcribe",
            lambda: transcribe.run(
                meta, job_dir, profile, model_size=config.asr_model, llm=llm
            ),
        ),
    )
    results.append(
        StageResult(
            "transcribe",
            "ran",
            f"{len(transcript.segments)} segments, lang={transcript.language}",
        )
    )

    # --- Stage 3: detect_scenes ---
    keyframes = cast(
        KeyframeSet,
        timed("detect_scenes", lambda: detect_scenes.run(meta, job_dir, profile)),
    )
    results.append(
        StageResult("detect_scenes", "ran", f"{len(keyframes.keyframes)} keyframes")
    )

    # --- Stage 4: extract_visuals (guard cost before touching the provider) ---
    extract_visuals.enforce_cost_limit(len(keyframes.keyframes), max_cost)
    visuals = cast(
        VisualExtractSet,
        timed(
            "extract_visuals",
            lambda: extract_visuals.run(
                keyframes,
                job_dir,
                profile,
                llm=llm,
                model=config.vision_model,
                max_cost=max_cost,
                image_max_width=config.vision_max_width,
            ),
        ),
    )
    results.append(
        StageResult("extract_visuals", "ran", f"{len(visuals.extracts)} extracts")
    )

    # --- Stage 5: align ---
    timeline = cast(
        AlignedTimeline,
        timed("align", lambda: align.run(transcript, visuals, job_dir, profile)),
    )
    results.append(StageResult("align", "ran", f"{len(timeline.chunks)} chunks"))

    # --- Stage 6: distill (reuses the client built for extract_visuals) ---
    notes = cast(
        DistilledNoteSet,
        timed(
            "distill",
            lambda: distill.run(
                timeline, job_dir, profile, llm=llm, model=config.distill_model
            ),
        ),
    )
    results.append(StageResult("distill", "ran", f"{len(notes.notes)} notes"))

    # --- Stage 7: render ---
    digest_path = cast(
        Path, timed("render", lambda: render.run(notes, job_dir, profile))
    )
    results.append(StageResult("render", "ran", digest_path.name))

    RunStats(stage_seconds=timings, total_seconds=sum(timings.values())).save(job_dir)
    return results
