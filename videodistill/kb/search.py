"""Hybrid retrieval: dense vectors + BM25, fused with reciprocal rank fusion."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from videodistill.kb.store import StoredPoint, VectorStore
from videodistill.language import load_stopwords
from videodistill.llm.base import LLMClient

# Keep programming tokens whole: "std::move", "push_back", "std::vector" survive.
# "." is a separator so "v.push_back" -> ["v", "push_back"] (push_back findable).
_TOKEN_RE = re.compile(r"[a-z0-9_:+#]+")

# The knowledge base is English; drop English stopwords from BM25 so a
# paraphrase query's "a"/"of"/"the" don't create spurious lexical hits.
_STOPWORDS = load_stopwords("en")

RRF_K = 60

# Drop vector matches below this cosine similarity: RRF is rank-based, so
# without a floor a totally-unrelated nearest neighbour still earns a rank and
# pollutes results (and defeats "not covered"). Related content clears this;
# noise does not.
VECTOR_SIM_FLOOR = 0.15


@dataclass
class SearchResult:
    canonical_concept_id: str
    collection: str
    source_video: str
    source_timestamp: float
    concept: str
    summary: str
    code_language: str | None
    score: float


def tokenize(text: str) -> list[str]:
    return [t.strip(":") for t in _TOKEN_RE.findall(text.lower()) if t.strip(":")]


def _content_tokens(text: str) -> list[str]:
    """Tokens for BM25: no stopwords, no single characters."""
    return [t for t in tokenize(text) if len(t) > 1 and t not in _STOPWORDS]


def hybrid_search(
    query: str,
    store: VectorStore,
    collection: str,
    embedder: LLMClient | None,
    embed_model: str,
    k: int = 8,
) -> list[SearchResult]:
    """Rank the collection for ``query`` by fusing BM25 and vector rankings."""
    points = store.all_points(collection)
    if not points:
        return []
    by_id = {p.id: p for p in points}

    bm25_ranking = _bm25_ranking(query, points)
    vector_ranking = _vector_ranking(
        query, store, collection, embedder, embed_model, len(points)
    )

    fused = reciprocal_rank_fusion([bm25_ranking, vector_ranking])
    results: list[SearchResult] = []
    for point_id, score in fused[:k]:
        payload = by_id[point_id].payload
        results.append(_to_result(payload, score))
    return results


def _bm25_ranking(query: str, points: list[StoredPoint]) -> list[str]:
    """Point ids with a positive BM25 score, best first (empty if no lexical hit)."""
    from rank_bm25 import BM25Okapi

    corpus = [_content_tokens(str(p.payload.get("text", ""))) for p in points]
    query_tokens = _content_tokens(query)
    if not query_tokens or not any(corpus):
        return []
    scores = BM25Okapi(corpus).get_scores(query_tokens)
    ranked = sorted(
        ((points[i].id, s) for i, s in enumerate(scores) if s > 0),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return [point_id for point_id, _ in ranked]


def _vector_ranking(
    query: str,
    store: VectorStore,
    collection: str,
    embedder: LLMClient | None,
    embed_model: str,
    limit: int,
) -> list[str]:
    if embedder is None:
        return []
    query_vector = embedder.embed([query], model=embed_model)[0]
    return [
        point_id
        for point_id, score in store.vector_search(collection, query_vector, limit)
        if score >= VECTOR_SIM_FLOOR
    ]


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = RRF_K
) -> list[tuple[str, float]]:
    """Fuse ranked id lists: score = sum 1/(k + rank), rank starting at 1."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, point_id in enumerate(ranking, start=1):
            scores[point_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def _to_result(payload: dict[str, object], score: float) -> SearchResult:
    return SearchResult(
        canonical_concept_id=str(payload.get("canonical_concept_id", "")),
        collection=str(payload.get("collection", "")),
        source_video=str(payload.get("source_video", "")),
        source_timestamp=float(payload.get("source_timestamp", 0.0) or 0.0),  # type: ignore[arg-type]
        concept=str(payload.get("concept", "")),
        summary=str(payload.get("summary", "")),
        code_language=(
            str(payload["code_language"])
            if payload.get("code_language") is not None
            else None
        ),
        score=score,
    )
