"""Orchestrate the evaluations for a job dir and render one markdown report."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from videodistill.config import Config
from videodistill.evals.checks import (
    GroundingResult,
    StatsResult,
    VerificationResult,
    VocabularyResult,
    run_grounding,
    run_stats,
    run_verification,
    run_vocabulary,
)
from videodistill.llm.base import LLMClient
from videodistill.models import (
    AlignedTimeline,
    Artifact,
    DistilledNoteSet,
    KeyframeSet,
    RunStats,
    Transcript,
    VideoMeta,
)
from videodistill.profile import DomainProfile

A = TypeVar("A", bound=Artifact)


def evaluate(
    job_dir: Path,
    profile: DomainProfile,
    config: Config,
    *,
    grounding_n: int = 5,
) -> str:
    """Run every eval for ``job_dir`` and return (and write) the markdown report."""
    notes = _try_load(DistilledNoteSet, job_dir)
    transcript = _try_load(Transcript, job_dir)
    timeline = _try_load(AlignedTimeline, job_dir)
    keyframes = _try_load(KeyframeSet, job_dir)
    meta = _try_load(VideoMeta, job_dir)
    run_stats_artifact = _try_load(RunStats, job_dir)

    verification = run_verification(notes, profile) if notes else None
    vocabulary = run_vocabulary(transcript, profile) if transcript else None
    grounding = (
        run_grounding(
            notes, timeline, _grounding_llm(config), config.distill_model, grounding_n
        )
        if (notes and timeline)
        else None
    )
    stats = run_stats(notes or DistilledNoteSet(), keyframes, meta, run_stats_artifact)

    report = _render_report(profile, verification, vocabulary, grounding, stats)
    (job_dir / "eval_report.md").write_text(report, encoding="utf-8")
    return report


def _try_load(artifact_type: type[A], job_dir: Path) -> A | None:
    try:
        return artifact_type.load(job_dir)
    except FileNotFoundError:
        return None


def _grounding_llm(config: Config) -> LLMClient | None:
    if not config.openai_api_key:
        return None
    from videodistill.llm.cache import CachedLLMClient
    from videodistill.llm.openai_provider import OpenAIProvider

    return CachedLLMClient(
        OpenAIProvider(api_key=config.openai_api_key), config.cache_dir
    )


def _pct(part: int, whole: int) -> str:
    return f"{(100 * part / whole):.0f}%" if whole else "n/a"


def _render_report(
    profile: DomainProfile,
    verification: VerificationResult | None,
    vocabulary: VocabularyResult | None,
    grounding: GroundingResult | None,
    stats: StatsResult,
) -> str:
    lines = ["# Eval Report", "", f"Profile: `{profile.name}`", ""]

    lines += _verification_section(verification)
    lines += _vocabulary_section(vocabulary)
    lines += _grounding_section(grounding)
    lines += _stats_section(stats)

    return "\n".join(lines) + "\n"


def _verification_section(v: VerificationResult | None) -> list[str]:
    out = ["## Code verification", ""]
    if v is None:
        out.append("_No verification command for this profile._")
    elif not v.tool_available:
        tool = v.command.split()[0] if v.command.split() else "(command)"
        out.append(f"_Tool `{tool}` not found on PATH; skipped._")
    elif v.total == 0:
        out.append("_No verifiable code snippets in this job._")
    else:
        out.append(f"- Command: `{v.command}`")
        out.append(
            f"- **Extracted code compiles: {v.passed}/{v.total} "
            f"({_pct(v.passed, v.total)})**"
        )
        for ts, reason in v.failures[:5]:
            out.append(f"  - failed @ {ts:.0f}s: {reason}")
    out.append("")
    return out


def _vocabulary_section(v: VocabularyResult | None) -> list[str]:
    out = ["## Vocabulary hit-rate", ""]
    if v is None:
        out.append("_No vocabulary for this profile._")
    else:
        out.append(
            f"- **Terms found in transcript: {v.found}/{v.total} "
            f"({_pct(v.found, v.total)})**"
        )
        if v.missing:
            preview = ", ".join(v.missing[:10])
            more = f" (+{len(v.missing) - 10} more)" if len(v.missing) > 10 else ""
            out.append(f"- Missing: {preview}{more}")
    out.append("")
    return out


def _grounding_section(g: GroundingResult | None) -> list[str]:
    out = ["## Grounding spot-check", ""]
    if g is None:
        out.append("_No notes to check._")
    elif g.skipped_reason:
        out.append(f"_Skipped: {g.skipped_reason}._")
    else:
        out.append(
            f"- **Fully-supported notes: {g.supported}/{g.sampled} "
            f"({_pct(g.supported, g.sampled)})**"
        )
        for ts, concept in g.unsupported[:5]:
            out.append(f"  - unsupported @ {ts:.0f}s: {concept}")
    out.append("")
    return out


def _stats_section(s: StatsResult) -> list[str]:
    out = ["## Pipeline stats", ""]
    if s.compression_ratio is not None and s.video_minutes is not None:
        out.append(
            f"- Compression: {s.video_minutes:.1f} min video -> "
            f"{s.reading_minutes:.1f} min read = **{s.compression_ratio:.1f}x**"
        )
    else:
        out.append(
            f"- Reading time: {s.reading_minutes:.1f} min (video length unknown)"
        )

    out.append(f"- Estimated cost: ${s.est_cost_usd:.2f} total (est.)")
    if s.est_cost_per_video_hour is not None:
        out.append(f"  - ~${s.est_cost_per_video_hour:.2f} per video-hour")

    if s.stage_seconds:
        parts = ", ".join(f"{name} {sec:.1f}s" for name, sec in s.stage_seconds.items())
        total = f" (total {s.total_seconds:.1f}s)" if s.total_seconds else ""
        out.append(f"- Wall-clock: {parts}{total}")
    else:
        out.append("- Wall-clock: not recorded (run `process` to capture)")
    out.append("")
    return out
