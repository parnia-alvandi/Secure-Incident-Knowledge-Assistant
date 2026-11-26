"""Retrieval evaluation helpers (phase 5): gold ranks, metrics, rescue analysis, splits."""
import math

import pytest

import numpy as np

from sia.evaluation.retrieval_eval import (EvalItem, SystemRun, build_items, evaluate_system,
                                           has_identifier, hit_at, mrr_at, paired_bootstrap,
                                           per_question, rescue_stats, split_indices, summarize)
from sia.processing.models import Chunk
from sia.retrieval.vector_retriever import RetrievedChunk


def mk(cid, filename="a.pdf", page=1, text="some text", doc_id=None) -> Chunk:
    return Chunk(chunk_id=cid, doc_id=doc_id or filename, filename=filename, title="T", doc_kind="public",
                 publisher=None, publication_date=None, page=page, chunk_index=0, text=text,
                 n_tokens=2, char_start=0, char_end=len(text))


class FakeRetriever:
    def __init__(self, table):
        self.table = table                       # question -> list of chunks (in rank order)

    def retrieve(self, query, k=None):
        return [RetrievedChunk(c, 1.0 / r, r) for r, c in enumerate(self.table.get(query, [])[:k], 1)]


def item(i, question="q", doc="a.pdf", gold=("g",), ident=False):
    return EvalItem(f"Q{i}", f"{question}{i}", doc, frozenset(gold), ident)


def run_of(chunk_ranks):
    n = len(chunk_ranks)
    return SystemRun("x", list(chunk_ranks), [1] * n, [3] * n, [1.0] * n)


def test_hit_and_mrr():
    ranks = [1, 3, None, 12]
    assert hit_at(ranks, 1) == 0.25 and hit_at(ranks, 5) == 0.5 and hit_at(ranks, 10) == 0.5
    assert math.isclose(mrr_at(ranks, 10), (1 + 1 / 3) / 4)
    assert math.isnan(hit_at([], 5))


def test_build_items_uses_single_chunk_gold_and_reports_unmatched():
    chunks = {"c1": mk("c1", text="Isolate the host within 4 hours of detection."),
              "c2": mk("c2", text="Unrelated paragraph.")}
    qs = [
        {"id": "E1", "lang": "en", "answerable": True, "question": "When to isolate?", "doc": "a.pdf",
         "evidence_page": 1, "evidence_quote": "Isolate the host within 4 hours"},
        {"id": "E2", "lang": "en", "answerable": True, "question": "Spans two chunks?", "doc": "a.pdf",
         "evidence_page": 1, "evidence_quote": "text that is in no single chunk"},
        {"id": "F1", "lang": "fa", "answerable": True, "question": "?", "doc": "a.pdf",
         "evidence_page": 1, "evidence_quote": "Isolate the host within 4 hours"},
        {"id": "U1", "lang": "en", "answerable": False, "question": "Unanswerable?"},
    ]
    items, unmatched = build_items(qs, chunks)
    assert [i.id for i in items] == ["E1"] and items[0].gold == {"c1"}
    assert unmatched == ["E2"]


def test_evaluate_system_ranks_and_distinct_docs():
    gold = mk("g", filename="a.pdf", doc_id="A")
    same_doc = mk("s", filename="a.pdf", doc_id="A")
    other = mk("o", filename="b.pdf", doc_id="B")
    items = [item(1, gold=["g"]), item(2, gold=["g"]), item(3, gold=["g"])]
    table = {"q1": [other, same_doc, gold], "q2": [other], "q3": [gold, same_doc]}
    run = evaluate_system("fake", FakeRetriever(table), items, k=10)
    assert run.chunk_ranks == [3, None, 1]
    assert run.doc_ranks == [2, None, 1]                 # first chunk of a.pdf
    assert run.distinct_docs_top5 == [2, 1, 1]
    assert len(run.latency_ms) == 3


def test_summarize_subset():
    run = SystemRun("x", [1, None, 4, 2], [1, 2, None, 1], [5, 4, 3, 2])
    s = summarize(run, [0, 2, 3], ks=(1, 3, 5))
    assert s["n"] == 3 and s["chunk_hit"][1] == 1 / 3 and s["chunk_hit"][5] == 1.0
    assert s["distinct_docs_top5"] == (5 + 3 + 2) / 3 and s["doc_hit@5"] == 2 / 3


def test_rescue_stats_partitions_the_questions():
    vec = run_of([1, None, 7, None, 2, None])
    bm = run_of([None, 3, 2, None, 9, 11])
    r = rescue_stats(vec, bm, list(range(6)), k=5)
    # idx0: vec only (lost), idx1: bm only (rescued), idx2: vec@7 miss, bm@2 -> rescued,
    # idx3: neither, idx4: vec@2 found, bm@9 miss -> lost, idx5: neither
    assert r["rescued"] == [1, 2] and r["lost"] == [0, 4] and r["neither"] == 2 and r["both"] == 0
    assert r["base_misses"] == 4
    assert math.isclose(r["base_hit"], 2 / 6) and math.isclose(r["other_hit"], 2 / 6)
    assert math.isclose(r["union_hit"], 4 / 6) and math.isclose(r["union_gain"], 2 / 6)
    assert len(r["rescued"]) + len(r["lost"]) + r["both"] + r["neither"] == r["n"]


def test_rescue_respects_the_cutoff():
    r = rescue_stats(run_of([None]), run_of([8]), [0], k=5)
    assert r["rescued"] == []
    assert rescue_stats(run_of([None]), run_of([8]), [0], k=10)["rescued"] == [0]


def test_has_identifier():
    assert has_identifier("What does NIST SP 800-61r3 say about lessons learned?")
    assert has_identifier("Which playbook is AIT-IR-BR-009?")
    assert not has_identifier("How long should backups be retained in 2024?")


def test_split_indices_is_deterministic_and_disjoint():
    items = [item(i) for i in range(11)]
    a1, b1 = split_indices(items, seed=3)
    a2, b2 = split_indices(items, seed=3)
    assert (a1, b1) == (a2, b2)
    assert set(a1).isdisjoint(b1) and sorted(a1 + b1) == list(range(11))
    assert split_indices(items, seed=4) != (a1, b1)


def test_per_question_values():
    run = SystemRun("x", [1, 4, None, 12], [1] * 4, [1] * 4)
    assert list(per_question(run, "hit", 5)) == [1, 1, 0, 0]
    assert list(per_question(run, "rr", 10)) == [1.0, 0.25, 0.0, 0.0]


def test_bootstrap_identical_systems_have_zero_difference():
    a = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    r = paired_bootstrap(a, a, ["d1", "d1", "d2", "d2", "d3", "d3"], n_boot=500, seed=1)
    assert r["diff"] == 0 and r["lo"] == 0 and r["hi"] == 0 and r["n_docs"] == 3


def test_bootstrap_consistent_improvement_excludes_zero():
    docs = [f"d{i // 4}" for i in range(40)]                  # 10 documents, 4 questions each
    base = np.zeros(40)
    better = np.ones(40)
    r = paired_bootstrap(better, base, docs, n_boot=500, seed=1)
    assert r["diff"] == 1.0 and r["lo"] == 1.0 and r["p_better"] == 1.0


def test_bootstrap_mixed_result_includes_zero_and_is_repeatable():
    rng = np.random.default_rng(5)
    docs = [f"d{i // 4}" for i in range(80)]
    a, b = rng.integers(0, 2, 80).astype(float), rng.integers(0, 2, 80).astype(float)
    r1 = paired_bootstrap(a, b, docs, n_boot=800, seed=3)
    r2 = paired_bootstrap(a, b, docs, n_boot=800, seed=3)
    assert r1 == r2 and r1["lo"] < 0 < r1["hi"]


def test_bootstrap_resamples_documents_so_interval_is_wider_than_per_question():
    # all questions of a document behave the same way: effective sample size = number of documents
    docs = [f"d{i // 10}" for i in range(100)]
    diff = np.repeat([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 10)
    clustered = paired_bootstrap(diff, np.zeros(100), docs, n_boot=1000, seed=0)
    iid = paired_bootstrap(diff, np.zeros(100), [f"q{i}" for i in range(100)], n_boot=1000, seed=0)
    assert (clustered["hi"] - clustered["lo"]) > (iid["hi"] - iid["lo"])


def test_bootstrap_validation_and_empty():
    with pytest.raises(ValueError):
        paired_bootstrap(np.zeros(2), np.zeros(3), ["a", "b"])
    assert paired_bootstrap(np.zeros(0), np.zeros(0), [])["n_docs"] == 0
