"""Stage 6 — distill.

One LLM call per AlignedChunk turns "what was shown + said" into a grounded
:class:`~videodistill.models.DistilledNote`. Guarantees that matter:

- **Grounded**: the prompt forbids anything outside the chunk. Administrative
  chatter (greetings, logistics) is skipped, producing no note.
- **English out**: every field is written in English regardless of the source
  language (the knowledge base is English).
- **Code is never rewritten**: ``code_snippet`` is copied verbatim from the
  chunk's visuals by this stage — the code editor, or code the speaker wrote on
  a whiteboard/slide — the model never emits code, so it cannot alter it.
- Canonical concept ids come from an emerging registry (``concepts.json``) so a
  concept seen twice reuses one id.

Validate the JSON reply, retry once with the error, then skip the chunk.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from pydantic import BaseModel, ValidationError

from videodistill.llm.base import LLMClient, LLMMessage
from videodistill.models import (
    AlignedChunk,
    AlignedTimeline,
    ConceptRegistry,
    DistilledNote,
    DistilledNoteSet,
    VisualExtract,
    VisualKind,
)
from videodistill.profile import DomainProfile

IS_STUB = False

logger = logging.getLogger("videodistill.stages.distill")


class _ParseError(Exception):
    """The model reply could not be turned into note fields."""


class _NoteFields(BaseModel):
    """The subset of a note the model is asked to produce (code excluded)."""

    skip: bool = False
    concept: str = ""
    summary: str = ""
    pitfalls: list[str] = []
    depends_on: list[str] = []
    partial: bool = False


def run(
    timeline: AlignedTimeline,
    job_dir: Path,
    profile: DomainProfile,
    *,
    llm: LLMClient,
    model: str,
) -> DistilledNoteSet:
    """Distill each aligned chunk into a note; write notes + concept registry."""
    registry: dict[str, str] = {}
    notes: list[DistilledNote] = []
    skipped = 0

    for chunk in timeline.chunks:
        note = _distill_chunk(chunk, profile, registry, llm=llm, model=model)
        if note is None:
            skipped += 1
        else:
            notes.append(note)

    result = DistilledNoteSet(notes=notes)
    result.save(job_dir)
    ConceptRegistry(concepts=registry).save(job_dir)
    logger.info(
        "distill: %d chunk(s) -> %d note(s) (%d skipped)",
        len(timeline.chunks),
        len(notes),
        skipped,
    )
    return result


def _distill_chunk(
    chunk: AlignedChunk,
    profile: DomainProfile,
    registry: dict[str, str],
    *,
    llm: LLMClient,
    model: str,
) -> DistilledNote | None:
    """Distill one chunk; None if skipped or unparseable after a retry."""
    transcript_text = " ".join(s.text for s in chunk.segments).strip()
    code_snippet = _code_from_visuals(chunk.visuals)
    code_language = _code_language_from_visuals(chunk.visuals)
    diagram = _diagram_from_visuals(chunk.visuals)
    # Whiteboard/slide code carries no model-tagged language; fall back to the
    # profile's primary language so the snippet still renders with a label.
    if code_snippet is not None and code_language is None and profile.code_languages:
        code_language = profile.code_languages[0]

    # Nothing to distill: no speech and no on-screen text.
    if not transcript_text and not any(v.text.strip() for v in chunk.visuals):
        return None

    prompt = _build_prompt(chunk, profile, transcript_text)
    fields = _call_with_retry(prompt, llm=llm, model=model)
    if fields is None or fields.skip or not fields.concept.strip():
        return None

    prefix = profile.concept_id_prefix
    return DistilledNote(
        concept=fields.concept.strip(),
        canonical_concept_id=_canonical_id(fields.concept, prefix, registry),
        summary=fields.summary.strip(),
        code_snippet=code_snippet,
        code_language=code_language,
        diagram=diagram,
        pitfalls=[p.strip() for p in fields.pitfalls if p.strip()],
        depends_on=[_dependency_id(d, prefix, registry) for d in fields.depends_on],
        source_timestamp=chunk.start,
        partial=fields.partial if code_snippet is not None else False,
    )


# Text on a whiteboard/slide is classified by SURFACE (whiteboard) not content,
# so code the speaker writes there never gets kind=="code". These patterns spot
# code-like text so it can still be surfaced as a snippet. Deliberately tight —
# a statement terminator/brace or a keyword-with-parens — so prose, vocabulary
# lists, and diagram labels don't get rendered as fake code.
_CODE_KEYWORDS = re.compile(
    r"\b(for|while|do|if|else|switch|case|return|break|continue|goto|int|char|"
    r"void|float|double|long|short|unsigned|signed|const|static|struct|union|"
    r"enum|typedef|sizeof|printf|scanf|malloc|free|main|include)\b"
)


def _looks_like_code(text: str) -> bool:
    """Heuristic: does this whiteboard/slide text contain actual code?"""
    t = text.strip()
    if not t:
        return False
    if "{" in t or "}" in t:  # a brace here is almost always code
        return True
    return ";" in t and (bool(_CODE_KEYWORDS.search(t)) or "=" in t or "(" in t)


# Surfaces on which the speaker writes code, in preference order: the editor
# first (precise), then whiteboard/slide (pseudo-code the speaker drew).
_CODE_KINDS = (VisualKind.code,)
_WHITEBOARD_KINDS = (VisualKind.whiteboard, VisualKind.slide)


def _code_from_visuals(visuals: list[VisualExtract]) -> str | None:
    """Verbatim code from the chunk's visuals (never model-generated).

    Prefers the code editor; falls back to code the speaker wrote on a
    whiteboard/slide (common for this kind of teaching). Either way the text is
    copied verbatim from the visual — the model never authors code.
    """
    blocks = [v.text for v in visuals if v.kind in _CODE_KINDS and v.text.strip()]
    if not blocks:
        blocks = [
            v.text
            for v in visuals
            if v.kind in _WHITEBOARD_KINDS and _looks_like_code(v.text)
        ]
    return "\n\n".join(blocks) if blocks else None


def _code_language_from_visuals(visuals: list[VisualExtract]) -> str | None:
    """The language of the chunk's first code visual, if the model tagged one."""
    for v in visuals:
        if v.kind in _CODE_KINDS and v.text.strip() and v.code_language:
            return v.code_language
    return None


def _diagram_from_visuals(visuals: list[VisualExtract]) -> str | None:
    """Verbatim ASCII-art diagram(s) the speaker drew in this chunk, if any."""
    blocks = [v.diagram for v in visuals if v.diagram.strip()]
    return "\n\n".join(blocks) if blocks else None


def _call_with_retry(prompt: str, *, llm: LLMClient, model: str) -> _NoteFields | None:
    last_error: _ParseError | None = None
    current = prompt
    for _attempt in range(2):
        reply = llm.complete([LLMMessage(role="user", content=current)], model=model)
        try:
            return _parse(reply)
        except _ParseError as exc:
            last_error = exc
            current = (
                f"{prompt}\n\nYour previous reply could not be used: {exc}. "
                "Return ONLY a single valid JSON object matching the schema."
            )
    logger.warning("distill: chunk skipped after retry: %s", last_error)
    return None


def _parse(reply: str) -> _NoteFields:
    text = _strip_code_fence(reply)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _ParseError(f"not valid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise _ParseError("expected a JSON object")
    try:
        return _NoteFields.model_validate(data)
    except ValidationError as exc:
        raise _ParseError(
            f"does not match the schema ({exc.error_count()} error(s))"
        ) from exc


def _build_prompt(
    chunk: AlignedChunk, profile: DomainProfile, transcript_text: str
) -> str:
    on_screen = "\n\n".join(
        f"[{v.kind.value}] {v.text}".strip() for v in chunk.visuals if v.text.strip()
    )
    parts = [
        "You are distilling ONE segment of a video into a study note. Use ONLY "
        "the material below — the transcript of what was said and the text shown "
        "on screen. Add nothing from outside knowledge. Write every field in "
        "English, even if the source is in another language.",
        f"TRANSCRIPT:\n{transcript_text or '(none)'}",
        f"ON SCREEN:\n{on_screen or '(none)'}",
        (
            "Return ONLY a JSON object with these keys:\n"
            '  "skip": true if this segment is administrative chatter (greetings, '
            "logistics, breaks) with no teachable concept, else false.\n"
            '  "concept": a short concept name (a few words).\n'
            '  "summary": <= 120 words, English. Capture EVERY substantive point '
            "(definitions, mechanisms, caveats, examples). Remove greetings, "
            "filler, and repetition. Do NOT include code — it is captured "
            "separately. Add nothing not stated in the material.\n"
            '  "pitfalls": list of pitfalls the speaker explicitly raised (empty '
            "if none).\n"
            '  "depends_on": list of prerequisite concept names this assumes '
            "(empty if none).\n"
            '  "partial": true if the on-screen code is an incomplete fragment.'
        ),
    ]
    if profile.distill_hints.strip():
        parts.append(f"Additional guidance: {profile.distill_hints.strip()}")
    return "\n\n".join(parts)


def _strip_code_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower().strip()).strip("-")
    return slug or "concept"


def _canonical_id(concept: str, prefix: str, registry: dict[str, str]) -> str:
    """Reuse an id for a concept already seen; else mint and register one."""
    key = concept.lower().strip()
    if key in registry:
        return registry[key]
    concept_id = f"{prefix}:{_slug(concept)}"
    registry[key] = concept_id
    return concept_id


def _dependency_id(dep: str, prefix: str, registry: dict[str, str]) -> str:
    """Map a prerequisite concept name to a canonical id (registry or slug)."""
    key = dep.lower().strip()
    return registry.get(key, f"{prefix}:{_slug(dep)}")
