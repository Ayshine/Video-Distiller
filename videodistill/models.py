"""Typed pipeline artifacts.

Every stage reads and writes these Pydantic models as JSON in the job
directory. Keeping them in one module (with no stage imports) is what lets any
stage be re-run in isolation and keeps the pipeline AWS-portable later: an
artifact is just a JSON file at a known path.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T", bound="Artifact")


class Artifact(BaseModel):
    """Base for every serializable pipeline artifact.

    ``filename`` is the canonical name each artifact type is stored under in a
    job directory, so stages can find their inputs by convention.
    """

    filename: str = Field(default="artifact.json", exclude=True)

    def save(self, job_dir: Path) -> Path:
        """Write this artifact to ``job_dir/<filename>`` and return the path."""
        job_dir.mkdir(parents=True, exist_ok=True)
        path = job_dir / self.filename
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls: type[T], job_dir: Path) -> T:
        """Read this artifact type from its canonical path in ``job_dir``."""
        path = job_dir / cls.model_fields["filename"].default
        if not path.exists():
            raise FileNotFoundError(
                f"Expected artifact {path} not found. "
                f"Run the stage that produces {cls.__name__} first."
            )
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class VideoMeta(Artifact):
    """Result of the ingest stage: probed metadata plus the extracted audio."""

    filename: str = Field(default="video_meta.json", exclude=True)

    source_path: str
    audio_path: str  # 16kHz mono wav produced by ingest
    duration_s: float
    width: int
    height: int
    fps: float
    audio_sample_rate: int = 16_000
    audio_channels: int = 1


class Word(BaseModel):
    """A single word with its own timestamps (from ASR word-level output)."""

    start: float
    end: float
    text: str
    probability: float | None = None


class TranscriptSegment(BaseModel):
    """A contiguous span of spoken text with start/end timestamps (seconds)."""

    start: float
    end: float
    text: str
    words: list[Word] = Field(default_factory=list)


class Transcript(Artifact):
    """Container artifact for the transcribe stage output."""

    filename: str = Field(default="transcript.json", exclude=True)

    language: str
    model_size: str
    segments: list[TranscriptSegment] = Field(default_factory=list)


class Keyframe(BaseModel):
    """A representative frame chosen by scene detection."""

    timestamp: float
    image_path: str
    phash: str  # perceptual hash, used to drop near-duplicate frames


class KeyframeSet(Artifact):
    """Container artifact for the detect_scenes stage output."""

    filename: str = Field(default="keyframes.json", exclude=True)

    keyframes: list[Keyframe] = Field(default_factory=list)


class VisualKind(StrEnum):
    slide = "slide"
    code = "code"
    whiteboard = "whiteboard"
    other = "other"


class VisualExtract(BaseModel):
    """Text/code recovered from a keyframe by the vision LLM."""

    timestamp: float
    kind: VisualKind
    text: str
    code_language: str | None = None
    description: str = ""
    # ASCII-art reproduction of a hand-drawn diagram/sketch on the frame (boxes,
    # arrows, stick figures, memory layouts), so the drawing survives into the
    # note as plain text. Empty when the frame has no diagram.
    diagram: str = ""


class VisualExtractSet(Artifact):
    """Container artifact for the extract_visuals stage output."""

    filename: str = Field(default="visuals.json", exclude=True)

    extracts: list[VisualExtract] = Field(default_factory=list)


class AlignedChunk(BaseModel):
    """Transcript segments and visual extracts bound together by time range.

    ``flags`` carries alignment annotations (e.g. "orphan", "merged_scroll")
    set by the align stage's policy.
    """

    start: float
    end: float
    segments: list[TranscriptSegment] = Field(default_factory=list)
    visuals: list[VisualExtract] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class AlignedTimeline(Artifact):
    """Container artifact for the align stage output."""

    filename: str = Field(default="aligned.json", exclude=True)

    chunks: list[AlignedChunk] = Field(default_factory=list)


class AlignmentReport(Artifact):
    """Counts produced by the align stage — the human-facing "what happened"."""

    filename: str = Field(default="alignment_report.json", exclude=True)

    chunk_count: int = 0
    segment_count: int = 0
    lookahead_moves: int = 0
    scroll_merges: int = 0
    orphans: int = 0


class DistilledNote(BaseModel):
    """One distilled concept — the core artifact the whole system revolves around.

    ``canonical_concept_id`` and ``depends_on`` exist from day one so the
    later knowledge/review layers need no retrofitting. ``partial`` marks a
    code snippet that is an intentional fragment (not expected to compile).
    """

    concept: str
    canonical_concept_id: str
    summary: str
    code_snippet: str | None = None
    code_language: str | None = None
    # ASCII-art reproduction of a diagram the speaker drew for this concept,
    # copied verbatim from the visual (never model-authored in distill).
    diagram: str | None = None
    pitfalls: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    source_timestamp: float
    partial: bool = False


class DistilledNoteSet(Artifact):
    """Container artifact for the distill stage output."""

    filename: str = Field(default="notes.json", exclude=True)

    notes: list[DistilledNote] = Field(default_factory=list)


class ConceptRegistry(Artifact):
    """Emerging map of normalized concept name -> canonical_concept_id.

    Built up during the distill stage so repeated concepts reuse one id; also
    the seed for cross-video dedup in later layers.
    """

    filename: str = Field(default="concepts.json", exclude=True)

    concepts: dict[str, str] = Field(default_factory=dict)


class RunStats(Artifact):
    """Wall-clock per stage from a full pipeline run, for the eval harness."""

    filename: str = Field(default="run_stats.json", exclude=True)

    stage_seconds: dict[str, float] = Field(default_factory=dict)
    total_seconds: float = 0.0
