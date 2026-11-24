"""Bm25Retriever (phase 5a): ranking, zero scores, ties, repeated terms, fields, loading."""
import math

import pytest

from sia.config import load_settings
from sia.processing.models import Chunk
from sia.retrieval.bm25 import Bm25Retriever, index_text
from sia.retrieval.vector_retriever import IndexNotReadyError
from sia.storage.chunk_store import (CHUNKS_FILE, MANIFEST_FILE, IndexManifest, save_manifest,
                                     write_chunks)
from pathlib import Path


def mk(i, text, filename="guide.pdf", title="Guide", section=None, doc_id="d1", page=1) -> Chunk:
    return Chunk(chunk_id=f"c{i}", doc_id=doc_id, filename=filename, title=title, doc_kind="public",
                 publisher=None, publication_date=None, page=page, chunk_index=i, text=text,
                 n_tokens=len(text.split()), char_start=0, char_end=len(text), section=section,
                 doc_chunk_count=10)


def ids(results):
    return [r.chunk.chunk_id for r in results]


CORPUS = [
    mk(0, "Isolate the infected host from the network to stop ransomware from spreading."),
    mk(1, "Backups must be tested regularly and stored offline."),
    mk(2, "The incident commander approves the communication plan before any press statement."),
    mk(3, "Ransomware negotiations require legal counsel and executive approval."),
    mk(4, "Phishing emails should be reported to the security team."),
]


def test_ranking_and_scores():
    r = Bm25Retriever(CORPUS).retrieve("ransomware", k=3)
    assert ids(r) == ["c3", "c0"] or ids(r) == ["c0", "c3"]          # only the two chunks that contain it
    assert [x.rank for x in r] == [1, 2]
    assert r[0].score >= r[1].score > 0


def test_bm25_formula_single_term_matches_hand_computation():
    corpus = [mk(0, "worm alpha beta"), mk(1, "gamma delta epsilon"), mk(2, "zeta eta theta")]
    r = Bm25Retriever(corpus, k1=1.5, b=0.75, fields=("text",)).retrieve("worm")
    n, df, tf, dl, avg = 3, 1, 1, 3, 3
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    expected = idf * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avg))
    assert len(r) == 1 and r[0].score == pytest.approx(expected, abs=1e-9)


def test_rare_term_outweighs_common_term():
    corpus = [mk(0, "incident response plan"), mk(1, "incident response team"),
              mk(2, "incident response lessons kerberoasting")]
    top = Bm25Retriever(corpus, fields=("text",)).retrieve("incident kerberoasting", k=3)
    assert top[0].chunk.chunk_id == "c2"


def test_zero_score_chunks_are_never_returned():
    r = Bm25Retriever(CORPUS)
    assert r.retrieve("quantum teleportation") == []
    out = r.retrieve("ransomware", k=5)                  # k=5 but only 2 chunks match
    assert len(out) == 2 and all(x.score > 0 for x in out)


def test_term_present_in_every_chunk_still_scores_above_zero():
    corpus = [mk(i, f"incident number {i}") for i in range(4)]
    out = Bm25Retriever(corpus, fields=("text",)).retrieve("incident")
    assert len(out) == 4 and all(x.score > 0 for x in out)


def test_ties_follow_corpus_order_and_are_repeatable():
    corpus = [mk(i, "isolate the host") for i in range(6)]       # identical text -> identical scores
    r = Bm25Retriever(corpus, top_k=6, fields=("text",))
    first = ids(r.retrieve("isolate host"))
    assert first == [f"c{i}" for i in range(6)]
    for _ in range(5):
        assert ids(r.retrieve("isolate host")) == first
    rebuilt = Bm25Retriever(corpus, top_k=6, fields=("text",))
    assert ids(rebuilt.retrieve("isolate host")) == first


def test_repeated_query_terms_do_not_change_scores():
    r = Bm25Retriever(CORPUS)
    once = r.retrieve("ransomware")
    many = r.retrieve("ransomware ransomware ransomware RANSOMWARE")
    assert ids(once) == ids(many)
    assert [x.score for x in once] == [x.score for x in many]
    assert r.query_terms("worm worm Worm") == ["worm"]


def test_exact_identifier_beats_partial_identifier():
    corpus = [
        mk(0, "Generic text about handling incidents.", filename="NIST.SP.800-61r3.pdf", title="NIST SP 800-61r3"),
        mk(1, "Generic text about handling incidents.", filename="NIST.SP.800-53r5.pdf", title="NIST SP 800-53r5"),
        mk(2, "Generic text about handling incidents.", filename="other.pdf", title="Other guide"),
    ]
    out = Bm25Retriever(corpus).retrieve("NIST SP 800-61r3")
    assert ids(out)[0] == "c0"
    assert "c2" not in ids(out)                           # shares no query term at all


def test_fields_control_what_is_indexed():
    chunk = mk(0, "plain body text", filename="syn_ir_playbook_ransomware.docx", title="Ransomware Playbook",
               section="Containment")
    assert index_text(chunk, ("text",)) == "plain body text"
    assert "syn_ir_playbook_ransomware" in index_text(chunk) and ".docx" not in index_text(chunk)
    assert Bm25Retriever([chunk], fields=("text",)).retrieve("containment") == []
    assert ids(Bm25Retriever([chunk], fields=("section", "text")).retrieve("containment")) == ["c0"]
    assert ids(Bm25Retriever([chunk]).retrieve("playbook")) == ["c0"]          # filename + title + section + text


def test_repeated_metadata_lowers_idf_of_document_words():
    """Every chunk of a document repeats its title: that word becomes common (lower idf)."""
    doc = [mk(i, f"body {w}", title="Ransomware Guide") for i, w in enumerate(["alpha", "beta", "gamma", "delta"])]
    with_meta = Bm25Retriever(doc, fields=("title", "text"))
    without = Bm25Retriever(doc, fields=("text",))
    assert with_meta._idf["ransomware"] < with_meta._idf["alpha"]
    assert "ransomware" not in without._idf


def test_length_normalisation_prefers_the_shorter_chunk():
    corpus = [mk(0, "worm " + "filler " * 40), mk(1, "worm filler")]
    assert ids(Bm25Retriever(corpus, b=0.75, fields=("text",)).retrieve("worm"))[0] == "c1"
    flat = Bm25Retriever(corpus, b=0.0, fields=("text",)).retrieve("worm")
    assert flat[0].score == pytest.approx(flat[1].score)


def test_k_override_and_top_k_default():
    r = Bm25Retriever([mk(i, "isolate host") for i in range(8)], top_k=3, fields=("text",))
    assert len(r.retrieve("isolate")) == 3
    assert len(r.retrieve("isolate", k=6)) == 6


def test_empty_stopword_only_and_empty_corpus():
    r = Bm25Retriever(CORPUS)
    assert r.retrieve("what is the") == []               # only stop-words -> no terms
    with pytest.raises(ValueError):
        r.retrieve("   ")
    assert Bm25Retriever([]).retrieve("anything") == []


def test_invalid_parameters():
    for kwargs in ({"fields": ("title",)}, {"fields": ("text", "text")}, {"fields": ("text", "x")},
                   {"b": 1.5}, {"k1": -1}):
        with pytest.raises(ValueError):
            Bm25Retriever(CORPUS, **kwargs)


def _settings():
    return load_settings(Path(__file__).resolve().parents[1] / "config.yaml")


def _manifest(n):
    return IndexManifest(created_at="t", embedding_model="m", embedding_dim=3, query_instruction="",
                         chunking={}, cleaning={}, collection="c", n_documents=1, n_chunks=n)


def test_from_index_ok(tmp_path):
    write_chunks(tmp_path / CHUNKS_FILE, CORPUS)
    save_manifest(tmp_path / MANIFEST_FILE, _manifest(len(CORPUS)))
    r = Bm25Retriever.from_index(tmp_path, _settings())
    assert r.top_k == 5 and r.n_chunks == 5 and ids(r.retrieve("phishing")) == ["c4"]


def test_from_index_errors(tmp_path):
    with pytest.raises(IndexNotReadyError):                                   # no manifest
        Bm25Retriever.from_index(tmp_path, _settings())
    write_chunks(tmp_path / CHUNKS_FILE, CORPUS)
    save_manifest(tmp_path / MANIFEST_FILE, _manifest(99))
    with pytest.raises(IndexNotReadyError):                                   # count mismatch
        Bm25Retriever.from_index(tmp_path, _settings())
