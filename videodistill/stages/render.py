"""Stage 7 — render.

Turn distilled notes into the human deliverable and the layer-2 feed:

- ``digest.md`` — one section per note (timestamp link, summary, ASCII diagram,
  verbatim code, pitfalls, dependencies), plus a compression footer (watch time
  vs. read time).
- ``notes.jsonl`` — one DistilledNote per line, the format the knowledge base
  ingests later.

Timestamp links become YouTube ``?t=`` deep links when the source was a URL,
otherwise a plain ``[hh:mm:ss]``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from videodistill.models import DistilledNote, DistilledNoteSet, VideoMeta
from videodistill.profile import DomainProfile

IS_STUB = False

logger = logging.getLogger("videodistill.stages.render")

READING_WPM = 200


def run(notes: DistilledNoteSet, job_dir: Path, profile: DomainProfile) -> Path:
    """Write digest.md and notes.jsonl; return the digest path."""
    meta = VideoMeta.load(job_dir)

    digest_path = job_dir / "digest.md"
    digest_path.write_text(render_digest(notes, meta), encoding="utf-8")

    jsonl_path = job_dir / "notes.jsonl"
    jsonl_path.write_text(
        "".join(note.model_dump_json() + "\n" for note in notes.notes),
        encoding="utf-8",
    )

    logger.info("render: %d note(s) -> digest.md, notes.jsonl", len(notes.notes))
    return digest_path


def render_digest(notes: DistilledNoteSet, meta: VideoMeta) -> str:
    """Build the Markdown digest (pure, so it is golden-file testable)."""
    source_name = Path(meta.source_path).name
    lines: list[str] = ["# Study Digest", ""]
    lines.append(f"Source: {source_name} · {len(notes.notes)} note(s)")
    lines.append("")

    if not notes.notes:
        lines.append("_No notes._")
        lines.append("")

    # One section per note.
    for note in notes.notes:
        lines.append("---")
        lines.append("")
        lines.extend(_render_note(note, meta.source_path))
        lines.append("")

    # Compression footer.
    lines.append("---")
    lines.append("")
    lines.extend(_render_footer(notes, meta))
    return "\n".join(lines) + "\n"


def _render_note(note: DistilledNote, source: str) -> list[str]:
    out = [f"## {note.concept}", ""]
    stamp = _timestamp_link(source, note.source_timestamp)
    out.append(f"{stamp} · concept `{note.canonical_concept_id}`")
    out.append("")
    if note.summary:
        out.append(note.summary)
        out.append("")
    if note.diagram:
        out.append("**Diagram**")
        out.append("")
        out.append("```text")
        out.append(note.diagram)
        out.append("```")
        out.append("")
    if note.code_snippet:
        out.append("```")
        out.append(note.code_snippet)
        out.append("```")
        if note.partial:
            out.append("_(code fragment)_")
        out.append("")
    if note.pitfalls:
        out.append("**Pitfalls**")
        out.append("")
        out.extend(f"- {p}" for p in note.pitfalls)
        out.append("")
    if note.depends_on:
        deps = ", ".join(f"`{d}`" for d in note.depends_on)
        out.append(f"**Depends on:** {deps}")
        out.append("")
    # Drop the trailing blank the caller re-adds.
    if out and out[-1] == "":
        out.pop()
    return out


def _render_footer(notes: DistilledNoteSet, meta: VideoMeta) -> list[str]:
    read_words = reading_words(notes)
    read_minutes = read_words / READING_WPM
    video_minutes = meta.duration_s / 60
    ratio = video_minutes / read_minutes if read_minutes > 0 else 0.0
    return [
        "## Compression",
        "",
        f"- Video length: {_hms(meta.duration_s)} ({video_minutes:.1f} min)",
        f"- Reading time: ~{read_minutes:.1f} min at {READING_WPM} wpm",
        f"- Compression: {ratio:.1f}x",
    ]


def reading_words(notes: DistilledNoteSet) -> int:
    """Words a reader actually reads: concepts, summaries, pitfalls (not code)."""
    total = 0
    for note in notes.notes:
        text = " ".join([note.concept, note.summary, *note.pitfalls])
        total += len(text.split())
    return total


def _hms(seconds: float) -> str:
    total = int(seconds) if seconds > 0 else 0
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def _timestamp_link(source: str, seconds: float) -> str:
    hms = _hms(seconds)
    if source.startswith(("http://", "https://")) and (
        "youtube.com" in source or "youtu.be" in source
    ):
        sep = "&" if "?" in source else "?"
        return f"[{hms}]({source}{sep}t={int(seconds)})"
    return f"[{hms}]"
