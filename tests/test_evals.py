"""Tests for the eval harness (subprocess for verification; LLM mocked)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from videodistill.config import Config
from videodistill.evals import evaluate
from videodistill.evals.checks import (
    run_grounding,
    run_stats,
    run_verification,
    run_vocabulary,
)
from videodistill.llm.base import LLMMessage
from videodistill.models import (
    AlignedChunk,
    AlignedTimeline,
    DistilledNote,
    DistilledNoteSet,
    Keyframe,
    KeyframeSet,
    RunStats,
    Transcript,
    TranscriptSegment,
    VideoMeta,
)
from videodistill.profile import DomainProfile, Verification

# grep is a portable stand-in for a real compile check.
GREP_PROFILE = DomainProfile(
    name="grep",
    code_languages=["python"],
    verification=Verification(kind="compile_check", command="grep -q PASS {file}"),
)


def _note(ts: float, concept: str, **extra: object) -> DistilledNote:
    data: dict[str, object] = {
        "concept": concept,
        "canonical_concept_id": f"gen:{concept.lower()}",
        "summary": f"summary of {concept}",
        "source_timestamp": ts,
    }
    data.update(extra)
    return DistilledNote.model_validate(data)


# --- verification -----------------------------------------------------------


def test_verification_pass_rate_excludes_partial_and_codeless() -> None:
    notes = DistilledNoteSet(
        notes=[
            _note(0, "Good", code_snippet="PASS this compiles"),
            _note(10, "Bad", code_snippet="nope"),
            _note(20, "Fragment", code_snippet="PASS but partial", partial=True),
            _note(30, "NoCode"),
        ]
    )
    result = run_verification(notes, GREP_PROFILE)
    assert result is not None
    assert result.tool_available is True
    assert result.total == 2  # partial + codeless excluded
    assert result.passed == 1
    assert result.failures[0][0] == 10.0


def test_verification_none_without_profile_command() -> None:
    assert run_verification(DistilledNoteSet(), DomainProfile(name="generic")) is None


def test_verification_reports_missing_tool() -> None:
    profile = DomainProfile(
        name="x",
        verification=Verification(
            kind="compile_check", command="not-a-real-tool {file}"
        ),
    )
    notes = DistilledNoteSet(notes=[_note(0, "C", code_snippet="x")])
    result = run_verification(notes, profile)
    assert result is not None and result.tool_available is False


# --- vocabulary -------------------------------------------------------------


def _transcript(text: str) -> Transcript:
    return Transcript(
        language="en",
        model_size="small",
        segments=[TranscriptSegment(start=0, end=1, text=text)],
    )


def test_vocabulary_hit_rate() -> None:
    profile = DomainProfile(name="v", vocabulary=["vector", "container", "mutex"])
    result = run_vocabulary(_transcript("we use a vector container today"), profile)
    assert result is not None
    assert result.found == 2 and result.total == 3
    assert result.missing == ["mutex"]


def test_vocabulary_matches_whole_tokens_only() -> None:
    profile = DomainProfile(name="v", vocabulary=["const"])
    # "const" must NOT match inside "constant".
    result = run_vocabulary(_transcript("a constant value"), profile)
    assert result is not None and result.found == 0


def test_vocabulary_none_when_empty() -> None:
    assert run_vocabulary(_transcript("hi"), DomainProfile(name="generic")) is None


# --- grounding --------------------------------------------------------------


class _FakeLLM:
    def __init__(self, verdicts: list[bool]) -> None:
        self._verdicts = iter(verdicts)

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        return json.dumps({"supported": next(self._verdicts)})


def test_grounding_scores_sampled_notes() -> None:
    notes = DistilledNoteSet(notes=[_note(0, "A"), _note(30, "B")])
    timeline = AlignedTimeline(
        chunks=[
            AlignedChunk(start=0, end=10),
            AlignedChunk(start=30, end=40),
        ]
    )
    result = run_grounding(notes, timeline, _FakeLLM([True, False]), "m", n=5)
    assert result is not None
    assert result.sampled == 2 and result.supported == 1
    assert result.unsupported == [(30.0, "B")]


def test_grounding_skipped_without_llm() -> None:
    notes = DistilledNoteSet(notes=[_note(0, "A")])
    result = run_grounding(notes, AlignedTimeline(), None, "m", n=5)
    assert result is not None and result.skipped_reason is not None


# --- stats ------------------------------------------------------------------


def test_stats_cost_and_compression() -> None:
    notes = DistilledNoteSet(notes=[_note(0, "A")])
    keyframes = KeyframeSet(
        keyframes=[Keyframe(timestamp=i, image_path="p", phash="0") for i in range(3)]
    )
    meta = VideoMeta(
        source_path="/v/x.mp4", audio_path="a", duration_s=600, width=1, height=1, fps=1
    )
    stats = run_stats(
        notes,
        keyframes,
        meta,
        RunStats(stage_seconds={"ingest": 1.0}, total_seconds=1.0),
    )
    # 3 frames * $0.01 + 1 note * $0.001
    assert stats.est_cost_usd == pytest.approx(0.031)
    assert stats.compression_ratio is not None
    assert stats.stage_seconds == {"ingest": 1.0}


# --- end-to-end report ------------------------------------------------------


def _config(tmp_path: Path) -> Config:
    return Config(
        openai_api_key=None,  # grounding will be skipped
        gemini_api_key=None,
        vision_provider="openai",
        asr_model="small",
        vision_model="gpt-4o",
        distill_model="gpt-4o-mini",
        embed_model="text-embedding-3-small",
        vision_max_width=1024,
        cache_dir=tmp_path / ".cache",
        kb_dir=tmp_path / ".kb",
        vector_store="qdrant",
    )


def test_evaluate_writes_report_with_all_sections(tmp_path: Path) -> None:
    if shutil.which("grep") is None:
        pytest.skip("grep not available")

    DistilledNoteSet(
        notes=[_note(0, "Vectors", code_snippet="PASS", summary="a vector is a list")]
    ).save(tmp_path)
    _transcript("we use a vector").save(tmp_path)
    AlignedTimeline(chunks=[AlignedChunk(start=0, end=10)]).save(tmp_path)
    KeyframeSet(keyframes=[]).save(tmp_path)
    VideoMeta(
        source_path="/v/x.mp4", audio_path="a", duration_s=600, width=1, height=1, fps=1
    ).save(tmp_path)
    RunStats(stage_seconds={"ingest": 0.5}, total_seconds=0.5).save(tmp_path)

    profile = DomainProfile(
        name="grep",
        vocabulary=["vector", "mutex"],
        verification=Verification(kind="compile_check", command="grep -q PASS {file}"),
    )
    report = evaluate(tmp_path, profile, _config(tmp_path), grounding_n=5)

    assert "# Eval Report" in report
    assert "Extracted code compiles: 1/1" in report
    assert "Terms found in transcript: 1/2" in report
    assert "OPENAI_API_KEY not set" in report  # grounding skipped
    assert "Compression:" in report
    assert "Wall-clock: ingest 0.5s" in report
    assert (tmp_path / "eval_report.md").exists()
