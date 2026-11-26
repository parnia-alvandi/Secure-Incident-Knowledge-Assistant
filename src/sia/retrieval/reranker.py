"""Cross-encoder re-ranking on top of any retriever (phase 5c).

A bi-encoder (the embedding model) encodes the question and each chunk SEPARATELY, which is fast
but coarse. A cross-encoder reads the question and one chunk TOGETHER and outputs a relevance
score for that pair: slower (one forward pass per candidate) but much more precise. So it is used
only on the few candidates that hybrid retrieval already found:

    question -> hybrid retrieval (top N candidates) -> cross-encoder scores N pairs -> best k

`RerankingRetriever` keeps the shape `retrieve(query, k) -> list[RetrievedChunk]`. It returns
`RerankedChunk` objects: `score` is the cross-encoder score, and the previous rank is kept in
`hybrid_rank`. The cross-encoder score is NOT a cosine similarity and NOT a probability; only its
order matters here. (Whether it is a raw logit or a sigmoid depends on the model's configured
activation; scripts/eval_retrieval.py prints the observed range.)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from sia.processing.models import Chunk
from sia.retrieval.hybrid import HybridChunk
from sia.retrieval.vector_retriever import RetrievedChunk, normalize_query

DEFAULT_CANDIDATES = 20
_SCORE_DECIMALS = 6          # scores equal up to float noise are treated as ties

# What the cross-encoder reads for each chunk (approved default: the plain chunk text, which is
# the kind of input ms-marco models were trained on; the other mode adds "title - section").
INPUT_MODES: dict[str, Callable[[Chunk], str]] = {
    "text": lambda c: c.text,
    "title_section_text": lambda c: c.embed_text(),
}


class RerankerUnavailableError(RuntimeError):
    """The cross-encoder model could not be loaded (usually: not cached and offline mode is on)."""


class _Reranker(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


class _Base(Protocol):
    def retrieve(self, query: str, k: int | None = ...) -> list[RetrievedChunk]: ...


@dataclass
class RerankedChunk(HybridChunk):
    """A re-ranked result. `score` = cross-encoder score; the fields below keep the history."""
    hybrid_rank: int | None = None       # rank given by the base retriever before re-ranking
    hybrid_score: float | None = None    # the base retriever's score (RRF if the base is hybrid)
    rerank_score: float | None = None    # same value as `score`


class CrossEncoderReranker:
    """Thin, lazy wrapper over sentence_transformers.CrossEncoder (CPU)."""

    def __init__(self, model_name: str, batch_size: int = 16, device: str = "cpu",
                 model: object | None = None) -> None:
        self.model_name, self.batch_size, self.device = model_name, batch_size, device
        self._model = model                       # a ready model can be injected (tests)

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder     # lazy: heavy import
                self._model = CrossEncoder(self.model_name, device=self.device)
            except Exception as exc:                               # noqa: BLE001 - re-raised with advice
                raise RerankerUnavailableError(
                    f"Cannot load the re-ranker '{self.model_name}': {exc}\n"
                    f"If it is not cached yet, download it once with network access: "
                    f"python scripts/fetch_reranker.py") from exc
        return self._model

    def warmup(self) -> float:
        t0 = time.perf_counter()
        self.score("warm up", ["warm up text"])
        return time.perf_counter() - t0

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(query, t) for t in texts]
        out = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(x) for x in out]


class CachingReranker:
    """Remembers scores per (query, text): lets the evaluation try several N without re-scoring."""

    def __init__(self, inner: _Reranker) -> None:
        self.inner = inner
        self.cache: dict[tuple[str, str], float] = {}

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        missing = [t for t in dict.fromkeys(texts) if (query, t) not in self.cache]
        for t, s in zip(missing, self.inner.score(query, missing)):
            self.cache[(query, t)] = s
        return [self.cache[(query, t)] for t in texts]


class RerankingRetriever:
    def __init__(self, base: _Base, reranker: _Reranker, top_k: int = 5,
                 candidates: int = DEFAULT_CANDIDATES, input_mode: str = "text") -> None:
        if top_k < 1 or candidates < 1:
            raise ValueError("need top_k >= 1 and candidates >= 1")
        if input_mode not in INPUT_MODES:
            raise ValueError(f"input_mode must be one of {sorted(INPUT_MODES)}")
        self.base, self.reranker = base, reranker
        self.top_k, self.candidates, self.input_mode = top_k, candidates, input_mode

    def warmup(self) -> float:
        total = 0.0
        for part in (self.base, self.reranker):
            warm = getattr(part, "warmup", None)
            if callable(warm):
                total += warm()
        return total

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        q = normalize_query(query)
        want = k or self.top_k
        found = self.base.retrieve(q, k=max(self.candidates, want))
        if not found:
            return []
        to_text = INPUT_MODES[self.input_mode]
        scores = self.reranker.score(q, [to_text(r.chunk) for r in found])
        if len(scores) != len(found):
            raise RuntimeError("the re-ranker returned a different number of scores than candidates")
        # best score first; equal scores keep the base retriever's order -> deterministic
        order = sorted(range(len(found)), key=lambda i: (-round(scores[i], _SCORE_DECIMALS), i))
        out: list[RetrievedChunk] = []
        for new_rank, i in enumerate(order[:want], start=1):
            r = found[i]
            out.append(RerankedChunk(
                r.chunk, scores[i], new_rank,
                getattr(r, "vector_rank", None), getattr(r, "vector_score", None),
                getattr(r, "bm25_rank", None), getattr(r, "bm25_score", None),
                hybrid_rank=r.rank, hybrid_score=r.score, rerank_score=scores[i]))
        return out
