"""BM25 keyword retrieval over the same chunks as the vector index (phase 5a).

Okapi BM25 with the Lucene-style idf, implemented with NumPy (no extra dependency):

    score(q, d) = sum over DISTINCT query terms t of
                  idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * len(d) / avglen))
    idf(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))        # always > 0

Guarantees (phase 5a requirements):
* only chunks with score > 0 are returned (a query sharing no term with a chunk never returns it);
* ties are broken by the chunk's position in chunks.jsonl, so results are deterministic;
* a term repeated in the question counts once (binary query term frequency);
* the output shape is the same as VectorRetriever: list[RetrievedChunk]. `score` is the BM25
  score, which is NOT comparable with a cosine similarity (RRF in phase 5b uses ranks only).

The index lives in memory and is rebuilt from chunks.jsonl on start-up (no extra file on disk).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path, PurePath
from typing import Sequence

import numpy as np

from sia.config import Settings
from sia.processing.models import Chunk
from sia.retrieval.tokenizer import tokenize
from sia.retrieval.vector_retriever import IndexNotReadyError, RetrievedChunk, normalize_query
from sia.storage.chunk_store import CHUNKS_FILE, MANIFEST_FILE, load_manifest, read_chunks

ALL_FIELDS = ("filename", "title", "section", "text")
DEFAULT_FIELDS = ALL_FIELDS          # approved in phase 5a: filename + title + section + text
_SCORE_DECIMALS = 10                 # scores equal up to float noise are treated as ties


def index_text(chunk: Chunk, fields: Sequence[str] = DEFAULT_FIELDS) -> str:
    """The text BM25 indexes for a chunk. `filename` is used without its extension."""
    parts = {
        "filename": PurePath(chunk.filename).stem if chunk.filename else "",
        "title": chunk.title or "",
        "section": chunk.section or "",
        "text": chunk.text or "",
    }
    return "\n".join(parts[f] for f in fields if parts[f])


class Bm25Retriever:
    def __init__(self, chunks: Sequence[Chunk], top_k: int = 5, k1: float = 1.5, b: float = 0.75,
                 fields: Sequence[str] = DEFAULT_FIELDS) -> None:
        fields = tuple(fields)
        if not fields or len(set(fields)) != len(fields) or not set(fields) <= set(ALL_FIELDS):
            raise ValueError(f"fields must be unique values from {ALL_FIELDS}, got {fields}")
        if "text" not in fields:
            raise ValueError("fields must include 'text'")
        if k1 < 0 or not 0.0 <= b <= 1.0:
            raise ValueError("need k1 >= 0 and 0 <= b <= 1")
        self.chunks: list[Chunk] = list(chunks)
        self.top_k, self.k1, self.b, self.fields = top_k, k1, b, fields
        self._build()

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_index(cls, index_dir: Path, settings: Settings, **kwargs) -> "Bm25Retriever":
        """Build from chunks.jsonl; checks that the index is complete (manifest) and consistent."""
        index_dir = Path(index_dir)
        manifest = load_manifest(index_dir / MANIFEST_FILE)
        if manifest is None:
            raise IndexNotReadyError(
                f"No complete index in {index_dir} (index_manifest.json missing). "
                f"Run: python scripts/build_index.py")
        chunks = read_chunks(index_dir / CHUNKS_FILE)
        if len(chunks) != manifest.n_chunks:
            raise IndexNotReadyError("chunks.jsonl does not match the manifest. Rebuild the index.")
        return cls(chunks, top_k=settings.retrieval.top_k, **kwargs)

    def _build(self) -> None:
        n = len(self.chunks)
        doc_len = np.zeros(n, dtype=np.float64)
        rows: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
        for i, chunk in enumerate(self.chunks):
            counts = Counter(tokenize(index_text(chunk, self.fields)))
            doc_len[i] = sum(counts.values())
            for term, tf in counts.items():
                rows[term][0].append(i)
                rows[term][1].append(tf)
        avg = float(doc_len.mean()) if n and doc_len.mean() > 0 else 1.0
        self.n_chunks, self.avg_len = n, avg
        self._norm = self.k1 * (1.0 - self.b + self.b * doc_len / avg)
        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._idf: dict[str, float] = {}
        for term, (idx, tfs) in rows.items():
            df = len(idx)
            self._postings[term] = (np.asarray(idx, dtype=np.int64), np.asarray(tfs, dtype=np.float64))
            self._idf[term] = float(np.log(1.0 + (n - df + 0.5) / (df + 0.5)))

    @property
    def vocab_size(self) -> int:
        return len(self._postings)

    # ------------------------------------------------------------------ retrieval
    def query_terms(self, query: str) -> list[str]:
        """Distinct query tokens in first-seen order (a repeated word counts once)."""
        return list(dict.fromkeys(tokenize(normalize_query(query))))

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        terms = self.query_terms(query)
        scores = np.zeros(self.n_chunks, dtype=np.float64)
        for term in terms:
            posting = self._postings.get(term)
            if posting is None:
                continue
            idx, tf = posting
            scores[idx] += self._idf[term] * tf * (self.k1 + 1.0) / (tf + self._norm[idx])
        scores = np.round(scores, _SCORE_DECIMALS)
        candidates = np.flatnonzero(scores > 0.0)             # never return a zero-score chunk
        if candidates.size == 0:
            return []
        # primary key: score (descending); secondary: corpus position (ascending) -> deterministic
        order = np.lexsort((candidates, -scores[candidates]))
        top = candidates[order][: (k or self.top_k)]
        return [RetrievedChunk(self.chunks[i], float(scores[i]), rank)
                for rank, i in enumerate(top, start=1)]
