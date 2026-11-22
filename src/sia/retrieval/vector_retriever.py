"""Vector retrieval over the local index (phase 4a).

Loads chunks.jsonl (source of truth for text and metadata) and the Chroma collection,
embeds the query and returns the top-k chunks with their cosine similarity.
BM25, RRF and the cross-encoder are added in phase 5 behind the same `retrieve()` shape.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from sia.config import Settings
from sia.processing.models import Chunk
from sia.storage.chunk_store import CHUNKS_FILE, MANIFEST_FILE, load_manifest, read_chunks

MAX_QUERY_CHARS = 1000


class IndexNotReadyError(RuntimeError):
    """The index is missing, incomplete, or built with a different model."""


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float        # cosine similarity, higher = more similar
    rank: int           # 1-based


class _Embedder(Protocol):
    def embed_query(self, text: str, use_instruction: bool = ...) -> np.ndarray: ...


class _Store(Protocol):
    def query(self, embedding: np.ndarray, k: int = ..., where: dict | None = ...): ...


def normalize_query(text: str) -> str:
    """Collapse whitespace; reject empty queries; cap the length."""
    q = re.sub(r"\s+", " ", (text or "")).strip()
    if not q:
        raise ValueError("empty question")
    return q[:MAX_QUERY_CHARS]


class VectorRetriever:
    def __init__(self, chunks_by_id: dict[str, Chunk], store: _Store, embedder: _Embedder,
                 top_k: int = 5, use_instruction: bool = True) -> None:
        self.chunks_by_id = chunks_by_id
        self.store = store
        self.embedder = embedder
        self.top_k = top_k
        self.use_instruction = use_instruction

    @classmethod
    def from_index(cls, index_dir: Path, settings: Settings) -> "VectorRetriever":
        from sia.embedding.embedder import Embedder          # lazy: heavy imports
        from sia.storage.vector_store import ChromaVectorStore

        index_dir = Path(index_dir)
        manifest = load_manifest(index_dir / MANIFEST_FILE)
        if manifest is None:
            raise IndexNotReadyError(
                f"No complete index in {index_dir} (index_manifest.json missing). "
                f"Run: python scripts/build_index.py")
        if manifest.embedding_model != settings.models.embedding:
            raise IndexNotReadyError(
                f"Index was built with '{manifest.embedding_model}' but the config says "
                f"'{settings.models.embedding}'. Rebuild the index.")
        chunks = read_chunks(index_dir / CHUNKS_FILE)
        if len(chunks) != manifest.n_chunks:
            raise IndexNotReadyError("chunks.jsonl does not match the manifest. Rebuild the index.")
        store = ChromaVectorStore(index_dir / "chroma", manifest.collection)
        embedder = Embedder(manifest.embedding_model, settings.embedding.batch_size,
                            settings.embedding.max_seq_length, manifest.query_instruction)
        return cls({c.chunk_id: c for c in chunks}, store, embedder, settings.retrieval.top_k,
                   settings.retrieval.use_query_instruction)

    def warmup(self) -> float:
        """Load the embedding model now; returns the seconds it took (cold start)."""
        import time
        t0 = time.perf_counter()
        self.embedder.embed_query("warm up", use_instruction=self.use_instruction)
        return time.perf_counter() - t0

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        q = normalize_query(query)
        vec = self.embedder.embed_query(q, use_instruction=self.use_instruction)
        hits = self.store.query(vec, k=k or self.top_k)
        out = []
        for h in hits:
            chunk = self.chunks_by_id.get(h.chunk_id)
            if chunk is None:
                raise IndexNotReadyError(f"vector store returned unknown chunk id {h.chunk_id}; "
                                         f"rebuild the index")
            out.append(RetrievedChunk(chunk, h.score, h.rank))
        return out
