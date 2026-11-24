"""RRF fusion and HybridRetriever (phase 5b) with fake retrievers (no model, no Chroma)."""
import pytest

from sia.processing.models import Chunk
from sia.retrieval.hybrid import HybridChunk, HybridRetriever, rrf_fuse
from sia.retrieval.vector_retriever import RetrievedChunk


def mk(cid: str) -> Chunk:
    return Chunk(chunk_id=cid, doc_id="d", filename="f.pdf", title="T", doc_kind="public", publisher=None,
                 publication_date=None, page=1, chunk_index=0, text=f"text {cid}", n_tokens=2,
                 char_start=0, char_end=6)


class Fake:
    """Returns a fixed ranked list of chunk ids; remembers the k it was asked for."""
    def __init__(self, ids, base=0.9):
        self.ids, self.base, self.last_k, self.queries = list(ids), base, None, []

    def retrieve(self, query, k=None):
        self.last_k = k
        self.queries.append(query)
        return [RetrievedChunk(mk(c), self.base - 0.01 * r, r + 1) for r, c in enumerate(self.ids[:k])]


def ids(results):
    return [r.chunk.chunk_id for r in results]


def test_rrf_formula_by_hand():
    out = dict(rrf_fuse([["a", "b", "c"], ["c", "a", "d"]], k_rrf=60))
    assert out["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert out["b"] == pytest.approx(1 / 62)
    assert out["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert out["d"] == pytest.approx(1 / 63)


def test_chunk_in_both_lists_beats_first_in_only_one():
    fused = rrf_fuse([["x", "a", "b"], ["y", "a", "c"]])
    assert [c for c, _ in fused][0] == "a"          # rank 2 in both beats rank 1 in one list


def test_rrf_ties_are_deterministic():
    # a and b have exactly the same score (1/61 + 1/62): the tie goes to the first list's order
    assert [c for c, _ in rrf_fuse([["a", "b"], ["b", "a"]])] == ["a", "b"]
    assert [c for c, _ in rrf_fuse([["b", "a"], ["a", "b"]])] == ["b", "a"]
    # p and r both have score 1/61: first appearance decides, q (1/62) is behind both
    out = rrf_fuse([["p", "q"], ["r"]], k_rrf=60)
    assert [c for c, _ in out] == ["p", "r", "q"]


def test_rrf_edge_cases():
    assert rrf_fuse([]) == [] and rrf_fuse([[], []]) == []
    assert [c for c, _ in rrf_fuse([["a", "b"]])] == ["a", "b"]                    # one list = same order
    assert dict(rrf_fuse([["a", "a", "a"]]))["a"] == pytest.approx(1 / 61)         # repeated id counts once
    with pytest.raises(ValueError):
        rrf_fuse([["a"]], k_rrf=0)


def test_k_rrf_changes_how_much_rank_one_matters():
    low = dict(rrf_fuse([["a", "b"]], k_rrf=1))
    high = dict(rrf_fuse([["a", "b"]], k_rrf=1000))
    assert low["a"] / low["b"] > high["a"] / high["b"]


def test_hybrid_retrieve_fuses_and_records_origins():
    vec, kw = Fake(["v1", "both", "v3"]), Fake(["k1", "both", "k3"])
    out = HybridRetriever(vec, kw, top_k=3).retrieve("question")
    assert ids(out)[0] == "both" and [r.rank for r in out] == [1, 2, 3]
    top = out[0]
    assert isinstance(top, HybridChunk) and isinstance(top, RetrievedChunk)
    assert (top.vector_rank, top.bm25_rank) == (2, 2) and top.vector_score and top.bm25_score
    only_vec = next(r for r in out if r.chunk.chunk_id == "v1")
    assert only_vec.bm25_rank is None and only_vec.bm25_score is None and only_vec.vector_rank == 1
    assert top.score == pytest.approx(2 / 62)


def test_hybrid_k_default_override_and_pool_size():
    vec, kw = Fake([f"v{i}" for i in range(30)]), Fake([f"k{i}" for i in range(30)])
    h = HybridRetriever(vec, kw, top_k=5, depth=20)
    assert len(h.retrieve("q")) == 5 and (vec.last_k, kw.last_k) == (20, 20)
    assert len(h.retrieve("q", k=8)) == 8
    assert len(h.retrieve("q", k=25)) == 25 and (vec.last_k, kw.last_k) == (25, 25)   # pool grows with k


def test_hybrid_when_one_retriever_returns_nothing():
    out = HybridRetriever(Fake(["v1", "v2"]), Fake([]), top_k=5).retrieve("q")
    assert ids(out) == ["v1", "v2"] and all(r.bm25_rank is None for r in out)
    assert HybridRetriever(Fake([]), Fake([]), top_k=5).retrieve("q") == []


def test_hybrid_is_repeatable_and_normalises_the_query():
    vec, kw = Fake(["a", "b", "c"]), Fake(["c", "b", "a"])
    h = HybridRetriever(vec, kw, top_k=3)
    first = ids(h.retrieve("  some   question "))
    assert all(ids(h.retrieve("some question")) == first for _ in range(3))
    assert vec.queries[0] == "some question"
    with pytest.raises(ValueError):
        h.retrieve("   ")


def test_hybrid_parameter_validation_and_warmup():
    for kwargs in ({"top_k": 0}, {"depth": 0}, {"k_rrf": 0}):
        with pytest.raises(ValueError):
            HybridRetriever(Fake([]), Fake([]), **kwargs)

    class Warm(Fake):
        def warmup(self):
            return 1.5
    assert HybridRetriever(Warm([]), Fake([])).warmup() == 1.5
    assert HybridRetriever(Fake([]), Fake([])).warmup() == 0.0
