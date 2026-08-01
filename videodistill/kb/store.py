"""Vector-store backends for the knowledge base.

The KB reaches a vector DB only through the small :class:`VectorStore` protocol
below, so the backend is **pluggable**. The one shipped implementation is an
embedded, on-disk Qdrant — no Docker, no server (:class:`QdrantStore`).

To add another backend (pgvector, Chroma, ...), implement the four
:class:`VectorStore` methods over the neutral :class:`VectorPoint` /
:class:`StoredPoint` values and register the class in :func:`build_store`.
Nothing else in the pipeline changes — ingest and search never import a specific
backend.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from videodistill.errors import VideoDistillError


@dataclass
class VectorPoint:
    """A point to upsert: id, embedding vector, and a JSON-able payload."""

    id: str
    vector: list[float]
    payload: dict[str, Any]


@dataclass
class StoredPoint:
    """A point read back from the store: its id and payload (no vector)."""

    id: str
    payload: dict[str, Any]


@runtime_checkable
class VectorStore(Protocol):
    """The entire vector-DB surface the KB uses. Implement it to add a backend."""

    def ensure_collection(self, name: str, dim: int) -> None:
        """Create collection ``name`` (vector size ``dim``) if it is absent."""

    def upsert(self, name: str, points: list[VectorPoint]) -> None:
        """Insert or replace ``points`` in collection ``name``."""

    def all_points(self, name: str) -> list[StoredPoint]:
        """Every point's id + payload (used to build the BM25 index)."""

    def vector_search(
        self, name: str, vector: list[float], limit: int
    ) -> list[tuple[str, float]]:
        """Top ``limit`` (id, score) by vector similarity, best first."""


class QdrantStore:
    """Embedded (local, no-Docker) Qdrant.

    ``:memory:`` is used by tests; a real directory persists the DB. Only the
    handful of operations the KB needs are exposed.
    """

    def __init__(self, location: Path | str) -> None:
        from qdrant_client import QdrantClient

        # ``location`` (the first positional / keyword) is parsed as a URL or
        # host, so a filesystem path must be passed as ``path=`` to get the
        # embedded on-disk DB. Only ":memory:" is a valid ``location`` value.
        if location == ":memory:":
            self._client = QdrantClient(location=":memory:")
        else:
            self._client = QdrantClient(path=path_str(location))

    def ensure_collection(self, name: str, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        if not self._client.collection_exists(name):
            self._client.create_collection(
                name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    def upsert(self, name: str, points: list[VectorPoint]) -> None:
        from qdrant_client.models import PointStruct

        self._client.upsert(
            collection_name=name,
            points=[
                PointStruct(id=p.id, vector=p.vector, payload=p.payload)
                for p in points
            ],
        )

    def all_points(self, name: str) -> list[StoredPoint]:
        if not self._client.collection_exists(name):
            return []
        out: list[StoredPoint] = []
        offset = None
        while True:
            records, offset = self._client.scroll(
                collection_name=name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            out.extend(
                StoredPoint(id=str(r.id), payload=dict(r.payload or {}))
                for r in records
            )
            if offset is None:
                break
        return out

    def vector_search(
        self, name: str, vector: list[float], limit: int
    ) -> list[tuple[str, float]]:
        if not self._client.collection_exists(name):
            return []
        response = self._client.query_points(
            collection_name=name, query=vector, limit=limit, with_payload=False
        )
        return [(str(p.id), p.score) for p in response.points]


def path_str(location: Path | str) -> str:
    return str(location)


# Registered backends. Add your own here after implementing VectorStore, e.g.
#   "pgvector": PgVectorStore,
#   "chroma": ChromaStore,
_BACKENDS: dict[str, Callable[[Path | str], VectorStore]] = {"qdrant": QdrantStore}


def build_store(kind: str, location: Path | str) -> VectorStore:
    """Construct the configured vector store. Defaults to embedded Qdrant.

    ``location`` is backend-defined — a directory for Qdrant, a DSN for a
    SQL-backed store, etc.
    """
    try:
        cls = _BACKENDS[kind]
    except KeyError:
        available = ", ".join(sorted(_BACKENDS))
        raise VideoDistillError(
            f"Unknown vector store {kind!r} (available: {available}). Add one by "
            "implementing the VectorStore protocol and registering it in "
            "videodistill/kb/store.py."
        ) from None
    return cls(location)
