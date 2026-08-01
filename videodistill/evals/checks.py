"""The individual evaluations. Each is a pure-ish function returning a small
result dataclass; :mod:`videodistill.evals.report` renders them to markdown.
"""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from videodistill.llm.base import LLMClient, LLMMessage
from videodistill.models import (
    AlignedChunk,
    AlignedTimeline,
    DistilledNote,
    DistilledNoteSet,
    KeyframeSet,
    RunStats,
    Transcript,
    VideoMeta,
)
from videodistill.profile import DomainProfile
from videodistill.stages import extract_visuals
from videodistill.stages.render import READING_WPM, reading_words

# Rough per-note distill cost (gpt-4o-mini, one short call). Estimate only.
EST_COST_PER_DISTILL_NOTE_USD = 0.001


# --- 1. code verification ---------------------------------------------------


@dataclass
class VerificationResult:
    command: str
    tool_available: bool
    total: int = 0
    passed: int = 0
    failures: list[tuple[float, str]] = field(default_factory=list)


def run_verification(
    notes: DistilledNoteSet, profile: DomainProfile
) -> VerificationResult | None:
    """Run the profile's verification command over each non-fragment snippet."""
    if profile.verification is None:
        return None

    command = profile.verification.command
    tool = shlex.split(command)[0] if command.strip() else ""
    if not tool or shutil.which(tool) is None:
        return VerificationResult(command=command, tool_available=False)

    candidates = [n for n in notes.notes if n.code_snippet and not n.partial]
    result = VerificationResult(
        command=command, tool_available=True, total=len(candidates)
    )
    with tempfile.TemporaryDirectory() as tmp:
        for i, note in enumerate(candidates):
            snippet_path = Path(tmp) / f"snippet_{i}"
            snippet_path.write_text(note.code_snippet or "", encoding="utf-8")
            args = [
                arg.replace("{file}", str(snippet_path)) for arg in shlex.split(command)
            ]
            proc = subprocess.run(args, capture_output=True, text=True)
            if proc.returncode == 0:
                result.passed += 1
            else:
                reason = (proc.stderr.strip().splitlines() or ["non-zero exit"])[0]
                result.failures.append((note.source_timestamp, reason))
    return result


# --- 2. vocabulary hit-rate -------------------------------------------------


@dataclass
class VocabularyResult:
    total: int
    found: int
    missing: list[str]


def run_vocabulary(
    transcript: Transcript, profile: DomainProfile
) -> VocabularyResult | None:
    """Share of profile vocabulary terms that appear (whole) in the transcript."""
    if not profile.vocabulary:
        return None

    text = " ".join(s.text for s in transcript.segments).lower()
    missing = [
        term for term in profile.vocabulary if not _term_present(term.lower(), text)
    ]
    return VocabularyResult(
        total=len(profile.vocabulary),
        found=len(profile.vocabulary) - len(missing),
        missing=missing,
    )


def _term_present(term: str, text: str) -> bool:
    """Whole-token match so 'const' does not match 'constant'."""
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text) is not None


# --- 3. grounding spot-check ------------------------------------------------


@dataclass
class GroundingResult:
    sampled: int = 0
    supported: int = 0
    unsupported: list[tuple[float, str]] = field(default_factory=list)
    skipped_reason: str | None = None


def run_grounding(
    notes: DistilledNoteSet,
    timeline: AlignedTimeline,
    llm: LLMClient | None,
    model: str,
    n: int,
) -> GroundingResult | None:
    """Judge whether a sample of notes' summaries are supported by their chunk."""
    if not notes.notes:
        return None
    if llm is None:
        return GroundingResult(skipped_reason="OPENAI_API_KEY not set")
    if n <= 0:
        return GroundingResult(skipped_reason="grounding-n is 0")

    chunk_by_start = {c.start: c for c in timeline.chunks}
    sample = notes.notes[:n]
    result = GroundingResult(sampled=len(sample))
    for note in sample:
        chunk = chunk_by_start.get(note.source_timestamp)
        supported = _judge(llm, model, note, chunk)
        if supported:
            result.supported += 1
        else:
            result.unsupported.append((note.source_timestamp, note.concept))
    return result


def _judge(
    llm: LLMClient, model: str, note: DistilledNote, chunk: AlignedChunk | None
) -> bool:
    source = _chunk_source_text(chunk) if chunk else "(source chunk not found)"
    prompt = (
        "You are checking a study note for grounding. Given the SOURCE (a "
        "transcript segment and any on-screen text) and a SUMMARY, decide "
        "whether EVERY claim in the summary is supported by the source. "
        'Answer ONLY with JSON: {"supported": true|false}.\n\n'
        f"SOURCE:\n{source}\n\nSUMMARY:\n{note.summary}"
    )
    reply = llm.complete([LLMMessage(role="user", content=prompt)], model=model)
    try:
        data = json.loads(_strip_fence(reply))
        return bool(data.get("supported", False))
    except (json.JSONDecodeError, AttributeError):
        return False  # unparseable judge reply is treated as not-supported


def _chunk_source_text(chunk: AlignedChunk) -> str:
    said = " ".join(s.text for s in chunk.segments)
    shown = "\n".join(v.text for v in chunk.visuals if v.text.strip())
    return f"Said: {said}\nShown:\n{shown}".strip()


def _strip_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


# --- 4. pipeline stats ------------------------------------------------------


@dataclass
class StatsResult:
    video_minutes: float | None
    reading_minutes: float
    compression_ratio: float | None
    est_cost_usd: float
    est_cost_per_video_hour: float | None
    stage_seconds: dict[str, float]
    total_seconds: float | None


def run_stats(
    notes: DistilledNoteSet,
    keyframes: KeyframeSet | None,
    meta: VideoMeta | None,
    run_stats_artifact: RunStats | None,
) -> StatsResult:
    read_minutes = reading_words(notes) / READING_WPM

    video_minutes = meta.duration_s / 60 if meta else None
    compression = (
        video_minutes / read_minutes
        if (video_minutes is not None and read_minutes > 0)
        else None
    )

    n_frames = len(keyframes.keyframes) if keyframes else 0
    est_cost = (
        n_frames * extract_visuals.EST_COST_PER_FRAME_USD
        + len(notes.notes) * EST_COST_PER_DISTILL_NOTE_USD
    )
    cost_per_hour = (
        est_cost / (meta.duration_s / 3600) if (meta and meta.duration_s > 0) else None
    )

    return StatsResult(
        video_minutes=video_minutes,
        reading_minutes=read_minutes,
        compression_ratio=compression,
        est_cost_usd=est_cost,
        est_cost_per_video_hour=cost_per_hour,
        stage_seconds=run_stats_artifact.stage_seconds if run_stats_artifact else {},
        total_seconds=run_stats_artifact.total_seconds if run_stats_artifact else None,
    )
