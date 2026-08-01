"""Layer 2 — the knowledge base.

Ingest DistilledNotes from many job dirs into a vector store, then retrieve them
with hybrid search: dense vectors for meaning plus BM25 for exact tokens
(``std::move`` must be findable), fused with reciprocal rank fusion.

The store is pluggable behind :class:`VectorStore`; the shipped backend is an
embedded, on-disk Qdrant (no Docker). Select one with :func:`build_store`.
"""

from videodistill.kb.ingest import ingest_job_dirs
from videodistill.kb.search import SearchResult, hybrid_search
from videodistill.kb.store import (
    QdrantStore,
    StoredPoint,
    VectorPoint,
    VectorStore,
    build_store,
)

__all__ = [
    "ingest_job_dirs",
    "hybrid_search",
    "SearchResult",
    "QdrantStore",
    "StoredPoint",
    "VectorPoint",
    "VectorStore",
    "build_store",
]
