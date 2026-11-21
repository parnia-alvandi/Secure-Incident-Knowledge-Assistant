"""Chroma integration test. Skipped automatically when chromadb is not installed."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from sia.ingestion.models import DocumentMeta, PageText
from sia.processing.chunker import ChunkingConfig, chunk_document
from sia.processing.tokens import whitespace_counter as wc


def _chunks():
    meta = DocumentMeta(doc_id="aaaabbbbcccc", filename="x.pdf", file_type="pdf", title="Doc X")
    pages = [PageText(1, " ".join(f"Alpha sentence {i} is here." for i in range(30))),
             PageText(2, " ".join(f"Beta sentence {i} is here." for i in range(30)))]
    chunks = chunk_document(meta, pages, ChunkingConfig(30, 5, 5), wc)
    meta2 = DocumentMeta(doc_id="ddddeeeeffff", filename="y.txt", file_type="txt", title="Doc Y")
    chunks += chunk_document(meta2, [PageText(None, "Plain text without a page number.")],
                             ChunkingConfig(30, 5, 5), wc)
    return chunks


def _vectors(n, dim=8):
    v = np.random.default_rng(0).normal(size=(n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_rebuild_query_and_metadata():
    pytest.importorskip("chromadb")
    from sia.storage.vector_store import ChromaVectorStore

    chunks, vecs = _chunks(), None
    vecs = _vectors(len(chunks))
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        store = ChromaVectorStore(Path(d) / "chroma", "test_col")
        store.rebuild(chunks, vecs)
        assert store.count() == len(chunks)
        store.rebuild(chunks, vecs)                       # rebuilding must not duplicate
        assert store.count() == len(chunks)

        hits = store.query(vecs[3], k=3)
        assert hits[0].chunk_id == chunks[3].chunk_id and abs(hits[0].score - 1.0) < 1e-4
        assert hits[0].score >= hits[1].score >= hits[2].score

        last = chunks[-1]                                  # page=None must not break Chroma
        meta = store.collection.get(ids=[last.chunk_id], include=["metadatas"])["metadatas"][0]
        assert "page" not in meta and meta["filename"] == "y.txt"
        only_page2 = store.query(vecs[0], k=5, where={"page": 2})
        assert only_page2 and all(next(c for c in chunks if c.chunk_id == h.chunk_id).page == 2
                                  for h in only_page2)
        assert np.allclose(store.get_embeddings([chunks[0].chunk_id])[0], vecs[0], atol=1e-5)
