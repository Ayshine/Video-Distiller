"""Tests for the render stage, including a golden-file digest comparison."""

from __future__ import annotations

import json
from pathlib import Path

from videodistill.models import DistilledNote, DistilledNoteSet, VideoMeta
from videodistill.profile import DomainProfile
from videodistill.stages import render

PROFILE = DomainProfile(name="generic")


def _meta(source: str, duration_s: float, tmp_path: Path) -> VideoMeta:
    meta = VideoMeta(
        source_path=source,
        audio_path=str(tmp_path / "audio.wav"),
        duration_s=duration_s,
        width=1920,
        height=1080,
        fps=30.0,
    )
    meta.save(tmp_path)
    return meta


def _notes() -> DistilledNoteSet:
    return DistilledNoteSet(
        notes=[
            DistilledNote(
                concept="Pointers",
                canonical_concept_id="cpp:pointers",
                summary="A pointer stores a memory address.",
                pitfalls=["Dereferencing null is undefined behavior."],
                source_timestamp=30.0,
            ),
            DistilledNote(
                concept="Vectors",
                canonical_concept_id="cpp:vectors",
                summary="std::vector is a dynamic array.",
                code_snippet="std::vector<int> v;\nv.push_back(1);",
                depends_on=["cpp:pointers"],
                source_timestamp=120.0,
            ),
        ]
    )


GOLDEN_DIGEST = """# Study Digest

Source: video.mp4 · 2 note(s)

---

## Pointers

[00:00:30] · concept `cpp:pointers`

A pointer stores a memory address.

**Pitfalls**

- Dereferencing null is undefined behavior.

---

## Vectors

[00:02:00] · concept `cpp:vectors`

std::vector is a dynamic array.

```
std::vector<int> v;
v.push_back(1);
```

**Depends on:** `cpp:pointers`

---

## Compression

- Video length: 00:10:00 (10.0 min)
- Reading time: ~0.1 min at 200 wpm
- Compression: 111.1x
"""


def test_digest_matches_golden(tmp_path: Path) -> None:
    meta = _meta("/videos/video.mp4", 600.0, tmp_path)
    digest = render.render_digest(_notes(), meta)
    assert digest == GOLDEN_DIGEST


def test_run_writes_digest_and_jsonl(tmp_path: Path) -> None:
    _meta("/videos/video.mp4", 600.0, tmp_path)
    notes = _notes()
    path = render.run(notes, tmp_path, PROFILE)

    assert path == tmp_path / "digest.md"
    assert path.read_text(encoding="utf-8") == GOLDEN_DIGEST

    jsonl = (tmp_path / "notes.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(jsonl) == 2
    first = json.loads(jsonl[0])
    assert first["canonical_concept_id"] == "cpp:pointers"


def test_youtube_source_makes_deep_link(tmp_path: Path) -> None:
    meta = _meta("https://youtube.com/watch?v=abc", 600.0, tmp_path)
    digest = render.render_digest(_notes(), meta)
    # ?v= already present, so the timestamp joins with &t=
    assert "[00:00:30](https://youtube.com/watch?v=abc&t=30)" in digest


def test_partial_code_marked_as_fragment(tmp_path: Path) -> None:
    meta = _meta("/videos/video.mp4", 600.0, tmp_path)
    notes = DistilledNoteSet(
        notes=[
            DistilledNote(
                concept="Snippet",
                canonical_concept_id="gen:snippet",
                summary="A partial example.",
                code_snippet="for x in xs:",
                partial=True,
                source_timestamp=0.0,
            )
        ]
    )
    digest = render.render_digest(notes, meta)
    assert "_(code fragment)_" in digest


def test_diagram_renders_as_block(tmp_path: Path) -> None:
    meta = _meta("/videos/video.mp4", 600.0, tmp_path)
    art = "+---+   +---+\n|SC | <-> |MC |\n+---+   +---+"
    notes = DistilledNoteSet(
        notes=[
            DistilledNote(
                concept="Compilation",
                canonical_concept_id="c:compilation",
                summary="Source code maps to machine code.",
                diagram=art,
                source_timestamp=0.0,
            )
        ]
    )
    digest = render.render_digest(notes, meta)
    assert "**Diagram**" in digest
    assert "```text" in digest
    assert art in digest  # the ASCII art appears verbatim


def test_empty_notes_render_gracefully(tmp_path: Path) -> None:
    meta = _meta("/videos/video.mp4", 600.0, tmp_path)
    digest = render.render_digest(DistilledNoteSet(notes=[]), meta)
    assert "_No notes._" in digest
    assert "Compression:" in digest
