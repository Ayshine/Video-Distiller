"""Tests for the distill stage (provider mocked; no network, no key)."""

from __future__ import annotations

import json
from pathlib import Path

from videodistill.llm.base import LLMMessage
from videodistill.models import (
    AlignedChunk,
    AlignedTimeline,
    ConceptRegistry,
    DistilledNoteSet,
    TranscriptSegment,
    VisualExtract,
    VisualKind,
)
from videodistill.profile import DomainProfile
from videodistill.stages import distill

PROFILE = DomainProfile(name="sample", concept_id_prefix="smp")


class _FakeLLM:
    """Returns queued `complete` replies in order; counts calls."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls = 0

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        self.calls += 1
        return self._replies.pop(0)


def _chunk(
    start: float,
    text: str,
    *,
    code: str | None = None,
) -> AlignedChunk:
    visuals = []
    if code is not None:
        visuals.append(
            VisualExtract(
                timestamp=start, kind=VisualKind.code, text=code, code_language="python"
            )
        )
    return AlignedChunk(
        start=start,
        end=start + 10,
        segments=[TranscriptSegment(start=start, end=start + 5, text=text)],
        visuals=visuals,
    )


def _reply(concept: str, **extra: object) -> str:
    data = {
        "skip": False,
        "concept": concept,
        "summary": f"summary of {concept}",
        "pitfalls": [],
        "depends_on": [],
        "partial": False,
    }
    data.update(extra)
    return json.dumps(data)


def _timeline(*chunks: AlignedChunk) -> AlignedTimeline:
    return AlignedTimeline(chunks=list(chunks))


def test_produces_one_note_per_chunk(tmp_path: Path) -> None:
    llm = _FakeLLM([_reply("Pointers"), _reply("Vectors")])
    result = distill.run(
        _timeline(_chunk(0, "about pointers"), _chunk(30, "about vectors")),
        tmp_path,
        PROFILE,
        llm=llm,
        model="gpt-4o-mini",
    )
    assert [n.concept for n in result.notes] == ["Pointers", "Vectors"]
    assert result.notes[0].canonical_concept_id == "smp:pointers"
    assert result.notes[0].source_timestamp == 0.0
    assert (tmp_path / "notes.json").exists()


def test_concept_id_is_reused_for_repeated_concept(tmp_path: Path) -> None:
    llm = _FakeLLM([_reply("Pointers"), _reply("pointers")])  # same concept, diff case
    result = distill.run(
        _timeline(_chunk(0, "intro"), _chunk(30, "more")),
        tmp_path,
        PROFILE,
        llm=llm,
        model="gpt-4o-mini",
    )
    ids = {n.canonical_concept_id for n in result.notes}
    assert ids == {"smp:pointers"}  # one shared id
    registry = ConceptRegistry.load(tmp_path)
    assert registry.concepts == {"pointers": "smp:pointers"}


def test_skip_returns_no_note(tmp_path: Path) -> None:
    llm = _FakeLLM([json.dumps({"skip": True, "concept": ""})])
    result = distill.run(
        _timeline(_chunk(0, "okay let's take a five minute break")),
        tmp_path,
        PROFILE,
        llm=llm,
        model="gpt-4o-mini",
    )
    assert result.notes == []


def test_code_is_passed_through_verbatim(tmp_path: Path) -> None:
    code = "def f(x):\n    return x * 2  # keep exact indentation"
    # The model reply does NOT contain code; the stage takes it from the visual.
    llm = _FakeLLM([_reply("Doubling", partial=True)])
    result = distill.run(
        _timeline(_chunk(0, "here is a function", code=code)),
        tmp_path,
        PROFILE,
        llm=llm,
        model="gpt-4o-mini",
    )
    note = result.notes[0]
    assert note.code_snippet == code  # byte-for-byte from the visual
    assert note.partial is True


def test_depends_on_maps_to_canonical_ids(tmp_path: Path) -> None:
    llm = _FakeLLM([_reply("Vectors", depends_on=["Pointers", "Memory"])])
    result = distill.run(
        _timeline(_chunk(0, "vectors need pointers")),
        tmp_path,
        PROFILE,
        llm=llm,
        model="gpt-4o-mini",
    )
    assert result.notes[0].depends_on == ["smp:pointers", "smp:memory"]


def test_retry_then_success(tmp_path: Path) -> None:
    llm = _FakeLLM(["not json", _reply("Pointers")])
    result = distill.run(
        _timeline(_chunk(0, "intro")),
        tmp_path,
        PROFILE,
        llm=llm,
        model="gpt-4o-mini",
    )
    assert len(result.notes) == 1
    assert llm.calls == 2


def test_unparseable_chunk_is_skipped_not_crashed(tmp_path: Path) -> None:
    llm = _FakeLLM(["garbage", "still garbage"])
    result = distill.run(
        _timeline(_chunk(0, "intro")),
        tmp_path,
        PROFILE,
        llm=llm,
        model="gpt-4o-mini",
    )
    assert result.notes == []
    assert llm.calls == 2


def test_empty_chunk_makes_no_llm_call(tmp_path: Path) -> None:
    llm = _FakeLLM([])  # would IndexError if called
    empty = AlignedChunk(start=0, end=5, segments=[], visuals=[])
    result = distill.run(_timeline(empty), tmp_path, PROFILE, llm=llm, model="m")
    assert result.notes == []
    assert llm.calls == 0


def test_distill_writes_loadable_artifact(tmp_path: Path) -> None:
    llm = _FakeLLM([_reply("Pointers")])
    distill.run(_timeline(_chunk(0, "intro")), tmp_path, PROFILE, llm=llm, model="m")
    assert len(DistilledNoteSet.load(tmp_path).notes) == 1


C_PROFILE = DomainProfile(name="c", concept_id_prefix="c", code_languages=["c"])


def _wb_chunk(*visuals: VisualExtract) -> AlignedTimeline:
    chunk = AlignedChunk(
        start=0,
        end=10,
        segments=[TranscriptSegment(start=0, end=5, text="the speaker explains")],
        visuals=list(visuals),
    )
    return _timeline(chunk)


def test_whiteboard_code_is_surfaced_as_snippet(tmp_path: Path) -> None:
    # Code the speaker wrote on the whiteboard must become a real code block.
    llm = _FakeLLM([_reply("For Loop", partial=True)])
    wb = VisualExtract(
        timestamp=0, kind=VisualKind.whiteboard, text="for () {\n    x = 5;\n}"
    )
    result = distill.run(_wb_chunk(wb), tmp_path, C_PROFILE, llm=llm, model="m")
    note = result.notes[0]
    assert note.code_snippet == "for () {\n    x = 5;\n}"
    assert note.code_language == "c"  # filled from the profile


def test_whiteboard_prose_is_not_treated_as_code(tmp_path: Path) -> None:
    llm = _FakeLLM([_reply("Imperative Languages")])
    wb = VisualExtract(
        timestamp=0,
        kind=VisualKind.whiteboard,
        text="imperative\nstandard\nportable\nprocedural",
    )
    result = distill.run(_wb_chunk(wb), tmp_path, C_PROFILE, llm=llm, model="m")
    assert result.notes[0].code_snippet is None


def test_editor_code_preferred_over_whiteboard(tmp_path: Path) -> None:
    llm = _FakeLLM([_reply("Main")])
    wb = VisualExtract(timestamp=0, kind=VisualKind.whiteboard, text="for(){ y=1; }")
    code = VisualExtract(
        timestamp=1, kind=VisualKind.code, text="int main(){ return 0; }",
        code_language="c",
    )
    result = distill.run(_wb_chunk(wb, code), tmp_path, C_PROFILE, llm=llm, model="m")
    assert result.notes[0].code_snippet == "int main(){ return 0; }"  # editor wins


def test_diagram_is_attached_verbatim(tmp_path: Path) -> None:
    art = "+---+   +---+\n|SC | <-> |MC |\n+---+   +---+"
    llm = _FakeLLM([_reply("Compilation")])
    wb = VisualExtract(timestamp=0, kind=VisualKind.whiteboard, text="SC", diagram=art)
    result = distill.run(_wb_chunk(wb), tmp_path, C_PROFILE, llm=llm, model="m")
    assert result.notes[0].diagram == art  # copied verbatim, not model-authored


def test_looks_like_code_heuristic() -> None:
    assert distill._looks_like_code("for () {\n    x = 5;\n}")
    assert distill._looks_like_code("x = 5;")
    assert distill._looks_like_code("void func(void) { }")
    assert not distill._looks_like_code("imperative\nstandard\nportable")
    assert not distill._looks_like_code("Very Long Term History of C")
    assert not distill._looks_like_code("")
