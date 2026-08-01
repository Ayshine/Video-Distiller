"""Tests for the knowledge base: hybrid retrieval over an in-memory Qdrant.

A topic-keyword fake embedder stands in for real embeddings so semantic
similarity is deterministic: an exact-token query is carried by BM25, a
paraphrase (no shared tokens) is carried by the vector side.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from videodistill.errors import VideoDistillError
from videodistill.kb import (
    QdrantStore,
    VectorStore,
    build_store,
    hybrid_search,
    ingest_job_dirs,
)
from videodistill.kb.search import (
    _bm25_ranking,
    reciprocal_rank_fusion,
    tokenize,
)
from videodistill.kb.store import StoredPoint
from videodistill.models import DistilledNote, DistilledNoteSet, VideoMeta

# Three disjoint topics -> three orthogonal unit vectors.
_TOPICS = [
    {"move", "rvalue", "moving", "transfer", "ownership", "resource"},
    {"vector", "push_back", "buffer", "reallocate"},
    {"pointer", "pointers", "unique_ptr", "heap", "smart"},
]


class TopicEmbedder:
    """Maps text to a 3-dim vector by which topic's keywords it contains."""

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        vectors = []
        for text in texts:
            tokens = set(re.findall(r"[a-z0-9_]+", text.lower()))
            vec = [1.0 if tokens & topic else 0.0 for topic in _TOPICS]
            if vec == [0.0, 0.0, 0.0]:
                vec = [0.0, 0.0, 1e-6]  # avoid a zero vector (cosine undefined)
            vectors.append(vec)
        return vectors


def _note(
    concept: str, summary: str, code: str | None = None, ts: float = 0.0
) -> DistilledNote:
    return DistilledNote(
        concept=concept,
        canonical_concept_id=f"cpp:{concept.lower().replace(' ', '-')}",
        summary=summary,
        code_snippet=code,
        source_timestamp=ts,
    )


NOTES = [
    _note(
        "Move semantics",
        "std::move produces an rvalue reference to enable moving",
        code="auto y = std::move(x);",
        ts=30,
    ),
    _note("Vector growth", "push_back may reallocate the underlying buffer", ts=60),
    _note("Smart pointers", "unique_ptr owns a heap allocation exclusively", ts=90),
]


def _job_dir(tmp_path: Path) -> Path:
    job = tmp_path / "job1"
    job.mkdir()
    VideoMeta(
        source_path="/videos/cpp1.mp4",
        audio_path="a",
        duration_s=600,
        width=1,
        height=1,
        fps=1,
    ).save(job)
    (job / "notes.jsonl").write_text(
        "".join(n.model_dump_json() + "\n" for n in NOTES), encoding="utf-8"
    )
    return job


def _ingested_store(tmp_path: Path) -> QdrantStore:
    store = QdrantStore(":memory:")
    count = ingest_job_dirs([_job_dir(tmp_path)], "cpp", store, TopicEmbedder(), "m")
    assert count == 3
    return store


# --- tokenizer --------------------------------------------------------------


def test_tokenizer_keeps_programming_tokens_whole() -> None:
    assert "std::move" in tokenize("auto y = std::move(x);")
    assert "push_back" in tokenize("v.push_back(1)")


# --- RRF --------------------------------------------------------------------


def test_rrf_orders_by_fused_score() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["a", "c"]])
    assert [point_id for point_id, _ in fused] == ["a", "c", "b"]


# --- BM25 -------------------------------------------------------------------


def test_bm25_finds_exact_token_and_ignores_others(tmp_path: Path) -> None:
    points = [
        StoredPoint(id="a", payload={"text": "Move semantics std::move rvalue"}),
        StoredPoint(id="b", payload={"text": "Vector growth push_back buffer"}),
        StoredPoint(id="c", payload={"text": "Smart pointers unique_ptr heap"}),
    ]
    assert _bm25_ranking("std::move", points) == ["a"]
    assert _bm25_ranking("transfer ownership resource", points) == []  # no lexical hit
    # A natural-language query of only stopwords finds nothing.
    assert _bm25_ranking("of a the", points) == []


# --- hybrid search end-to-end ----------------------------------------------


def test_exact_token_query_returns_right_note(tmp_path: Path) -> None:
    store = _ingested_store(tmp_path)
    results = hybrid_search("std::move", store, "cpp", TopicEmbedder(), "m", k=8)
    assert results[0].concept == "Move semantics"
    assert results[0].source_video == "cpp1.mp4"
    assert results[0].source_timestamp == 30.0


def test_paraphrase_query_returns_right_note_via_vectors(tmp_path: Path) -> None:
    store = _ingested_store(tmp_path)
    # No token here appears in any note -> BM25 is silent; vectors must carry it.
    results = hybrid_search(
        "transfer ownership of a resource", store, "cpp", TopicEmbedder(), "m", k=8
    )
    assert results[0].concept == "Move semantics"


def test_search_without_embedder_falls_back_to_bm25(tmp_path: Path) -> None:
    store = _ingested_store(tmp_path)
    results = hybrid_search("push_back", store, "cpp", None, "m", k=8)
    assert results[0].concept == "Vector growth"


def test_build_store_selects_qdrant_and_rejects_unknown() -> None:
    store = build_store("qdrant", ":memory:")
    assert isinstance(store, QdrantStore)
    assert isinstance(store, VectorStore)  # satisfies the pluggable protocol
    with pytest.raises(VideoDistillError, match="Unknown vector store"):
        build_store("pgvector", ":memory:")


def test_on_disk_store_ingest_and_search(tmp_path: Path) -> None:
    """On-disk mode (``path=``) must work, not just ``:memory:``.

    Regression guard: passing a filesystem path as Qdrant's ``location`` makes
    the client treat it as a remote host, so a real KB build failed even though
    the in-memory tests passed.
    """
    store = QdrantStore(tmp_path / "kb")
    count = ingest_job_dirs([_job_dir(tmp_path)], "cpp", store, TopicEmbedder(), "m")
    assert count == 3
    results = hybrid_search("std::move", store, "cpp", TopicEmbedder(), "m", k=8)
    assert results[0].concept == "Move semantics"
    assert results[0].source_video == "cpp1.mp4"


def test_ingest_falls_back_to_notes_json(tmp_path: Path) -> None:
    job = tmp_path / "job2"
    job.mkdir()
    DistilledNoteSet(notes=NOTES).save(job)  # notes.json, no notes.jsonl
    store = QdrantStore(":memory:")
    count = ingest_job_dirs([job], "cpp", store, TopicEmbedder(), "m")
    assert count == 3
