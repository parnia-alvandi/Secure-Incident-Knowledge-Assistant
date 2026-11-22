"""VectorRetriever with a fake embedder and a fake store (no model, no Chroma)."""
import numpy as np

from sia.processing.models import Chunk
from sia.retrieval.vector_retriever import (IndexNotReadyError, VectorRetriever,
                                            normalize_query)
from sia.storage.vector_store import Hit


def _chunk(i: int) -> Chunk:
    return Chunk(chunk_id=f"c{i}", doc_id="d", filename="f.pdf", title=f"Doc {i}", doc_kind="public",
                 publisher=None, publication_date=None, page=i, chunk_index=i, text=f"text {i}",
                 n_tokens=2, char_start=0, char_end=6)


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed_query(self, text, use_instruction=True):
        self.calls.append((text, use_instruction))
        return np.zeros(3, dtype=np.float32)


class FakeStore:
    def __init__(self, ids):
        self.ids, self.last_k = ids, None

    def query(self, embedding, k=5, where=None):
        self.last_k = k
        return [Hit(cid, 0.9 - 0.1 * r, r + 1) for r, cid in enumerate(self.ids[:k])]


def _retriever(ids=("c0", "c1", "c2", "c3"), top_k=3, instruction=True):
    chunks = {f"c{i}": _chunk(i) for i in range(4)}
    emb, store = FakeEmbedder(), FakeStore(list(ids))
    return VectorRetriever(chunks, store, emb, top_k, instruction), emb, store


def test_retrieve_returns_chunks_with_scores_and_ranks():
    r, emb, store = _retriever()
    res = r.retrieve("  What is   containment? ")
    assert [x.chunk.chunk_id for x in res] == ["c0", "c1", "c2"]
    assert [x.rank for x in res] == [1, 2, 3]
    assert res[0].score > res[1].score > res[2].score
    assert emb.calls == [("What is containment?", True)] and store.last_k == 3


def test_k_override_and_instruction_flag():
    r, emb, store = _retriever(instruction=False)
    assert len(r.retrieve("question", k=2)) == 2 and store.last_k == 2
    assert emb.calls[0][1] is False


def test_empty_question_is_rejected_and_long_question_is_capped():
    r, _, _ = _retriever()
    for bad in ("", "   \n\t ", None):
        try:
            r.retrieve(bad)
        except ValueError:
            continue
        raise AssertionError("empty question accepted")
    assert len(normalize_query("x " * 2000)) == 1000


def test_unknown_chunk_id_from_the_store_is_reported():
    r, _, _ = _retriever(ids=("c0", "ghost"))
    try:
        r.retrieve("question")
    except IndexNotReadyError as exc:
        assert "ghost" in str(exc)
    else:
        raise AssertionError("unknown id accepted")
