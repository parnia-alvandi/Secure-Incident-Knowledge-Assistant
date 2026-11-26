"""CrossEncoderReranker / RerankingRetriever (phase 5c) with fake models (no download)."""
import pytest

from sia.processing.models import Chunk
from sia.retrieval.hybrid import HybridChunk, HybridRetriever
from sia.retrieval.reranker import (CachingReranker, CrossEncoderReranker, RerankedChunk,
                                    RerankerUnavailableError, RerankingRetriever)
from sia.retrieval.vector_retriever import RetrievedChunk


def mk(cid: str, text: str | None = None, title="Guide", section=None) -> Chunk:
    return Chunk(chunk_id=cid, doc_id="d", filename="f.pdf", title=title, doc_kind="public", publisher=None,
                 publication_date=None, page=1, chunk_index=0, text=text or f"text {cid}", n_tokens=2,
                 char_start=0, char_end=6, section=section)


class Base:
    """Stand-in for the hybrid retriever: fixed list, remembers the k it was asked for."""
    def __init__(self, ids):
        self.ids, self.last_k = list(ids), None

    def retrieve(self, query, k=None):
        self.last_k = k
        return [RetrievedChunk(mk(c), 0.03 - 0.001 * r, r + 1) for r, c in enumerate(self.ids[:k])]


class Scorer:
    """Scores a text by a lookup table (default 0); counts calls and pairs."""
    def __init__(self, table):
        self.table, self.calls, self.pairs = table, 0, 0

    def score(self, query, texts):
        self.calls += 1
        self.pairs += len(texts)
        return [self.table.get(t, 0.0) for t in texts]


def ids(results):
    return [r.chunk.chunk_id for r in results]


def test_reorders_by_cross_encoder_score_and_keeps_history():
    base = Base(["a", "b", "c", "d"])
    rr = RerankingRetriever(base, Scorer({"text c": 5.0, "text a": 1.0, "text d": -2.0}), top_k=3)
    out = rr.retrieve("question")
    assert ids(out) == ["c", "a", "b"] and [r.rank for r in out] == [1, 2, 3]
    top = out[0]
    assert isinstance(top, RerankedChunk) and isinstance(top, HybridChunk) and isinstance(top, RetrievedChunk)
    assert top.score == top.rerank_score == 5.0 and top.hybrid_rank == 3
    assert top.hybrid_score == pytest.approx(0.03 - 0.001 * 2)


def test_candidates_pool_and_k_behaviour():
    base = Base([f"c{i}" for i in range(40)])
    scorer = Scorer({})
    rr = RerankingRetriever(base, scorer, top_k=5, candidates=20)
    assert len(rr.retrieve("q")) == 5 and base.last_k == 20 and scorer.pairs == 20
    assert len(rr.retrieve("q", k=8)) == 8 and base.last_k == 20
    assert len(rr.retrieve("q", k=25)) == 25 and base.last_k == 25        # pool grows with k


def test_a_candidate_outside_the_pool_cannot_be_promoted():
    base = Base([f"c{i}" for i in range(30)])
    rr = RerankingRetriever(base, Scorer({"text c25": 99.0}), top_k=5, candidates=10)
    assert "c25" not in ids(rr.retrieve("q"))


def test_ties_keep_the_base_order_and_results_are_repeatable():
    base = Base(["a", "b", "c", "d"])
    rr = RerankingRetriever(base, Scorer({"text a": 1.0, "text b": 1.0000000001, "text c": 1.0}), top_k=4)
    first = ids(rr.retrieve("q"))
    assert first == ["a", "b", "c", "d"]            # the 1e-10 difference is float noise -> still a tie
    assert all(ids(rr.retrieve("q")) == first for _ in range(3))


def test_fewer_candidates_than_requested_and_empty_base():
    assert ids(RerankingRetriever(Base(["a", "b"]), Scorer({}), top_k=5).retrieve("q")) == ["a", "b"]
    scorer = Scorer({})
    assert RerankingRetriever(Base([]), scorer, top_k=5).retrieve("q") == [] and scorer.calls == 0


def test_input_modes():
    seen = {}

    class Spy:
        def score(self, query, texts):
            seen["texts"] = list(texts)
            return [0.0] * len(texts)

    class OneChunk:
        def retrieve(self, query, k=None):
            return [RetrievedChunk(mk("a", text="body", title="NIST Guide", section="Containment"), 1.0, 1)]

    RerankingRetriever(OneChunk(), Spy(), input_mode="text").retrieve("q")
    assert seen["texts"] == ["body"]
    RerankingRetriever(OneChunk(), Spy(), input_mode="title_section_text").retrieve("q")
    assert seen["texts"] == ["NIST Guide - Containment\nbody"]
    with pytest.raises(ValueError):
        RerankingRetriever(OneChunk(), Spy(), input_mode="nope")


def test_validation_and_query_normalisation():
    for kwargs in ({"top_k": 0}, {"candidates": 0}):
        with pytest.raises(ValueError):
            RerankingRetriever(Base([]), Scorer({}), **kwargs)
    with pytest.raises(ValueError):
        RerankingRetriever(Base(["a"]), Scorer({})).retrieve("   ")


def test_wrong_number_of_scores_is_an_error():
    class Broken:
        def score(self, query, texts):
            return [1.0]

    with pytest.raises(RuntimeError):
        RerankingRetriever(Base(["a", "b"]), Broken()).retrieve("q")


def test_history_is_copied_from_a_hybrid_base():
    class HB:
        def retrieve(self, query, k=None):
            return [HybridChunk(mk("a"), 0.03, 1, 2, 0.8, 3, 7.5)]

    out = RerankingRetriever(HB(), Scorer({})).retrieve("q")[0]
    assert (out.vector_rank, out.vector_score, out.bm25_rank, out.bm25_score) == (2, 0.8, 3, 7.5)


def test_works_on_top_of_a_real_hybrid_retriever():
    class Fixed:
        def __init__(self, ids_):
            self.ids_ = ids_

        def retrieve(self, query, k=None):
            return [RetrievedChunk(mk(c), 1.0, r + 1) for r, c in enumerate(self.ids_[:k])]

    hyb = HybridRetriever(Fixed(["a", "b", "c"]), Fixed(["c", "b", "a"]), top_k=3)
    out = RerankingRetriever(hyb, Scorer({"text a": 3.0}), top_k=2).retrieve("q")
    assert ids(out)[0] == "a" and out[0].hybrid_rank is not None and out[0].vector_rank == 1


def test_cross_encoder_wrapper_uses_the_injected_model():
    class FakeModel:
        def __init__(self):
            self.calls = []

        def predict(self, pairs, batch_size, show_progress_bar):
            self.calls.append((list(pairs), batch_size, show_progress_bar))
            return [float(len(t)) for _, t in pairs]

    m = FakeModel()
    ce = CrossEncoderReranker("some/model", batch_size=4, model=m)
    assert ce.score("q", ["a", "bbb"]) == [1.0, 3.0]
    assert m.calls[0] == ([("q", "a"), ("q", "bbb")], 4, False)
    assert ce.score("q", []) == [] and len(m.calls) == 1               # nothing to score, no model call
    assert ce.warmup() >= 0.0


def test_missing_model_gives_a_helpful_error():
    ce = CrossEncoderReranker("not a valid repo id")
    with pytest.raises(RerankerUnavailableError):
        ce.score("q", ["text"])


def test_caching_reranker_scores_each_pair_once():
    inner = Scorer({"text a": 1.0, "text b": 2.0})
    cache = CachingReranker(inner)
    assert cache.score("q", ["text a", "text b"]) == [1.0, 2.0]
    assert cache.score("q", ["text b", "text a", "text a"]) == [2.0, 1.0, 1.0]
    assert inner.pairs == 2                                            # second call came from the cache
    cache.score("other question", ["text a"])
    assert inner.pairs == 3                                            # a different query is a different pair
