"""Chroma wrapper: a persistent, local, cosine-space collection.

Design notes
* Embeddings are computed by us (embedder.py) and passed in; Chroma's default
  embedding function is never used.
* Telemetry is disabled: nothing leaves the machine.
* Chroma metadata may only hold str/int/float/bool, so None values are omitted.
* `query()` returns an exact cosine similarity computed from the stored vector
  (vectors are normalised), independent of Chroma's distance configuration.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sia.processing.models import Chunk


@dataclass
class Hit:
    chunk_id: str
    score: float          # cosine similarity in [-1, 1]; higher = more similar
    rank: int             # 1-based


def chunk_metadata(c: Chunk) -> dict:
    meta = {"doc_id": c.doc_id, "filename": c.filename, "title": c.title,
            "chunk_index": c.chunk_index, "n_tokens": c.n_tokens,
            "content_type": c.content_type}
    for key in ("page", "section", "doc_kind", "publisher", "publication_date"):
        value = getattr(c, key)
        if value is not None:
            meta[key] = value
    return meta


class ChromaVectorStore:
    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False))
        self.collection_name = collection_name

    # -- build ---------------------------------------------------------------
    def rebuild(self, chunks: list[Chunk], embeddings: np.ndarray, batch: int = 500) -> None:
        """Drop the collection and create it again from scratch."""
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:  # noqa: BLE001 - collection did not exist yet
            pass
        try:   # newer Chroma API
            col = self._client.create_collection(
                self.collection_name, configuration={"hnsw": {"space": "cosine"}})
        except TypeError:   # older Chroma API
            col = self._client.create_collection(
                self.collection_name, metadata={"hnsw:space": "cosine"})
        for i in range(0, len(chunks), batch):
            part = chunks[i:i + batch]
            col.add(ids=[c.chunk_id for c in part],
                    embeddings=embeddings[i:i + batch].tolist(),
                    documents=[c.text for c in part],
                    metadatas=[chunk_metadata(c) for c in part])

    # -- read ----------------------------------------------------------------
    @property
    def collection(self):
        return self._client.get_collection(self.collection_name)

    def count(self) -> int:
        return self.collection.count()

    def ids(self) -> list[str]:
        return list(self.collection.get(include=[])["ids"])

    def get_embeddings(self, ids: list[str]) -> np.ndarray:
        res = self.collection.get(ids=ids, include=["embeddings"])
        by_id = {i: np.asarray(e, dtype=np.float32) for i, e in zip(res["ids"], res["embeddings"])}
        return np.stack([by_id[i] for i in ids])

    def query(self, embedding: np.ndarray, k: int = 5, where: dict | None = None) -> list[Hit]:
        res = self.collection.query(query_embeddings=[np.asarray(embedding).tolist()],
                                    n_results=k, where=where, include=["embeddings"])
        q = np.asarray(embedding, dtype=np.float32)
        hits = []
        for rank, (cid, emb) in enumerate(zip(res["ids"][0], res["embeddings"][0]), start=1):
            hits.append(Hit(cid, float(np.dot(q, np.asarray(emb, dtype=np.float32))), rank))
        return hits
