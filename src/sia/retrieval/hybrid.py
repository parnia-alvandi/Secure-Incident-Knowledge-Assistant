"""Hybrid retrieval: vector search + BM25 fused with Reciprocal Rank Fusion (phase 5b).

RRF (Cormack, Clarke & Buettcher, 2009) uses only RANKS, never scores, so the incomparable
cosine and BM25 scales do not matter:

    RRF(d) = sum over the ranked lists r that contain d of   1 / (k_rrf + rank_r(d))

A chunk that appears high in BOTH lists beats a chunk that is first in only one. k_rrf = 60 is the
value from the paper; it damps the advantage of rank 1 over rank 2, 3, ... (with k_rrf = 0 rank 1
would count twice as much as rank 2).

Same shape as the other retrievers: `retrieve(query, k) -> list[RetrievedChunk]`. The returned
objects are `HybridChunk` (a RetrievedChunk subclass) that also record where each retriever
ranked the chunk, for debugging and analysis. `score` is the RRF score (about 0.03 at most with
two lists and k_rrf = 60): it is NOT a cosine similarity and must not be compared with one.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from sia.config import Settings
from sia.processing.models import Chunk
from sia.retrieval.bm25 import Bm25Retriever
from sia.retrieval.vector_retriever import RetrievedChunk, normalize_query

DEFAULT_DEPTH = 50        # candidates taken from EACH retriever before fusion
DEFAULT_K_RRF = 60
_SCORE_DECIMALS = 12      # RRF scores equal up to float noise are ties


@dataclass
class HybridChunk(RetrievedChunk):
    """A fused result. `score` = RRF score; the fields below show the evidence behind it."""
    vector_rank: int | None = None       # rank in the vector list (None = not among its candidates)
    vector_score: float | None = None    # cosine similarity
    bm25_rank: int | None = None
    bm25_score: float | None = None


class _Retriever(Protocol):
    def retrieve(self, query: str, k: int | None = ...) -> list[RetrievedChunk]: ...


def rrf_fuse(rankings: Sequence[Sequence[str]], k_rrf: int = DEFAULT_K_RRF) -> list[tuple[str, float]]:
    """Fuse ranked lists of ids (best first). Returns (id, rrf_score), best first.

    Ties (equal scores) are broken by the best rank the id has in any list, then by the order in
    which the ids first appear (first list first), so the result is deterministic.
    """
    if k_rrf <= 0:
        raise ValueError("k_rrf must be > 0")
    score: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for ranking in rankings:
        seen_here: set[str] = set()
        for rank, cid in enumerate(ranking, start=1):
            if cid in seen_here:                 # a repeated id inside one list counts once
                continue
            seen_here.add(cid)
            score[cid] = score.get(cid, 0.0) + 1.0 / (k_rrf + rank)
            best_rank[cid] = min(best_rank.get(cid, rank), rank)
            first_seen.setdefault(cid, len(first_seen))
    ordered = sorted(score, key=lambda c: (-round(score[c], _SCORE_DECIMALS), best_rank[c], first_seen[c]))
    return [(c, score[c]) for c in ordered]


class HybridRetriever:
    def __init__(self, vector: _Retriever, bm25: _Retriever, top_k: int = 5,
                 depth: int = DEFAULT_DEPTH, k_rrf: int = DEFAULT_K_RRF) -> None:
        if top_k < 1 or depth < 1 or k_rrf <= 0:
            raise ValueError("need top_k >= 1, depth >= 1 and k_rrf > 0")
        self.vector, self.bm25 = vector, bm25
        self.top_k, self.depth, self.k_rrf = top_k, depth, k_rrf

    @classmethod
    def from_index(cls, index_dir: Path, settings: Settings, depth: int = DEFAULT_DEPTH,
                   k_rrf: int = DEFAULT_K_RRF) -> "HybridRetriever":
        from sia.retrieval.vector_retriever import VectorRetriever      # validates the index
        vector = VectorRetriever.from_index(index_dir, settings)
        chunks: list[Chunk] = list(vector.chunks_by_id.values())        # same order as chunks.jsonl
        bm25 = Bm25Retriever(chunks, top_k=settings.retrieval.top_k)
        return cls(vector, bm25, settings.retrieval.top_k, depth, k_rrf)

    def warmup(self) -> float:
        """Load the embedding model now (BM25 needs no warm-up); returns the seconds it took."""
        warm = getattr(self.vector, "warmup", None)
        return warm() if callable(warm) else 0.0

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        q = normalize_query(query)
        want = k or self.top_k
        pool = max(self.depth, want)                       # never fuse fewer candidates than requested
        vec = self.vector.retrieve(q, k=pool)
        kw = self.bm25.retrieve(q, k=pool)
        by_id: dict[str, Chunk] = {}
        v_info: dict[str, RetrievedChunk] = {}
        b_info: dict[str, RetrievedChunk] = {}
        for r in vec:
            by_id.setdefault(r.chunk.chunk_id, r.chunk)
            v_info.setdefault(r.chunk.chunk_id, r)
        for r in kw:
            by_id.setdefault(r.chunk.chunk_id, r.chunk)
            b_info.setdefault(r.chunk.chunk_id, r)
        fused = rrf_fuse([[r.chunk.chunk_id for r in vec], [r.chunk.chunk_id for r in kw]], self.k_rrf)
        out: list[RetrievedChunk] = []
        for rank, (cid, s) in enumerate(fused[:want], start=1):
            v, b = v_info.get(cid), b_info.get(cid)
            out.append(HybridChunk(by_id[cid], s, rank,
                                   v.rank if v else None, v.score if v else None,
                                   b.rank if b else None, b.score if b else None))
        return out
