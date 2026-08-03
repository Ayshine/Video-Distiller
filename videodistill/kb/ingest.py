"""Ingest DistilledNotes from job dirs into a KB collection."""

from __future__ import annotations

import uuid
from pathlib import Path

from videodistill.kb.store import VectorPoint, VectorStore
from videodistill.llm.base import LLMClient
from videodistill.models import DistilledNote, DistilledNoteSet, VideoMeta


def ingest_job_dirs(
    job_dirs: list[Path],
    collection: str,
    store: VectorStore,
    embedder: LLMClient,
    embed_model: str,
) -> int:
    """Embed and upsert every note from ``job_dirs``; return the point count."""
    entries: list[tuple[DistilledNote, str]] = []
    for job_dir in job_dirs:
        source_video = _source_video(job_dir)
        for note in _read_notes(job_dir):
            entries.append((note, source_video))

    if not entries:
        return 0

    # Embed concept + summary (the meaning), per the spec. Batched: the
    # embeddings API caps a request at ~300k tokens / 2048 inputs, so a large
    # collection must be split across several calls.
    embed_texts = [f"{note.concept}\n{note.summary}" for note, _ in entries]
    vectors = _embed_batched(embedder, embed_texts, embed_model)

    store.ensure_collection(collection, len(vectors[0]))
    points = [
        _build_point(note, source_video, collection, vector)
        for (note, source_video), vector in zip(entries, vectors, strict=True)
    ]
    store.upsert(collection, points)
    return len(points)


def _embed_batched(
    embedder: LLMClient,
    texts: list[str],
    model: str,
    max_inputs: int = 1000,
    max_tokens: int = 200_000,
) -> list[list[float]]:
    """Embed ``texts`` across several requests to stay under the API's per-call
    limits (~300k tokens / 2048 inputs). Tokens are estimated as len/4."""
    vectors: list[list[float]] = []
    batch: list[str] = []
    batch_tokens = 0
    for text in texts:
        est = max(1, len(text) // 4)
        if batch and (len(batch) >= max_inputs or batch_tokens + est > max_tokens):
            vectors.extend(embedder.embed(batch, model=model))
            batch, batch_tokens = [], 0
        batch.append(text)
        batch_tokens += est
    if batch:
        vectors.extend(embedder.embed(batch, model=model))
    return vectors


def _build_point(
    note: DistilledNote, source_video: str, collection: str, vector: list[float]
) -> VectorPoint:
    key = (
        f"{collection}:{source_video}:"
        f"{note.canonical_concept_id}:{note.source_timestamp}"
    )
    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, key))
    payload = {
        "canonical_concept_id": note.canonical_concept_id,
        "collection": collection,
        "source_video": source_video,
        "source_timestamp": note.source_timestamp,
        "code_language": note.code_language,
        "depends_on": note.depends_on,
        "concept": note.concept,
        "summary": note.summary,
        # BM25 searches this: concept names, summaries, and exact code tokens.
        "text": _searchable_text(note),
        "note": note.model_dump(),
    }
    return VectorPoint(id=point_id, vector=vector, payload=payload)


def _searchable_text(note: DistilledNote) -> str:
    return " ".join(
        part for part in [note.concept, note.summary, note.code_snippet or ""] if part
    ).strip()


def _read_notes(job_dir: Path) -> list[DistilledNote]:
    """Prefer notes.jsonl (the layer-2 feed); fall back to notes.json."""
    jsonl = job_dir / "notes.jsonl"
    if jsonl.exists():
        return [
            DistilledNote.model_validate_json(line)
            for line in jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    try:
        return DistilledNoteSet.load(job_dir).notes
    except FileNotFoundError:
        return []


def _source_video(job_dir: Path) -> str:
    try:
        return Path(VideoMeta.load(job_dir).source_path).name
    except FileNotFoundError:
        return job_dir.name
