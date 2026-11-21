"""Pipeline test with a fake embedder and a fake store (no model, no Chroma needed)."""
import hashlib
import tempfile
from pathlib import Path

import numpy as np

from sia.indexing.builder import build_index
from sia.ingestion.pipeline import ingest_directory
from sia.processing.chunker import ChunkingConfig
from sia.processing.tokens import whitespace_counter as wc
from sia.storage.chunk_store import CHUNKS_FILE, MANIFEST_FILE, load_manifest, read_chunks


class FakeEmbedder:
    model_name = "fake-model"
    query_instruction = "q: "
    dim = 8

    def embed_passages(self, texts, show_progress=False):
        out = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "big")
            v = np.random.default_rng(seed).normal(size=self.dim).astype(np.float32)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


class BrokenEmbedder(FakeEmbedder):
    def embed_passages(self, texts, show_progress=False):
        raise RuntimeError("boom")


class FakeStore:
    collection_name = "fake"

    def __init__(self):
        self.chunks, self.embeddings = [], None

    def rebuild(self, chunks, embeddings):
        self.chunks, self.embeddings = list(chunks), embeddings


def _make_docs(d: Path) -> None:
    import docx
    (d / "a_policy.txt").write_text(
        "1. Purpose\n" + "\n".join(f"Purpose line {i} describes the policy goal." for i in range(40)) +
        "\n2. Scope\n" + "\n".join(f"Scope line {i} lists the covered systems." for i in range(40)),
        encoding="utf-8")
    doc = docx.Document()
    doc.add_heading("Roles", 1)
    for i in range(30):
        doc.add_paragraph(f"Role {i} is responsible for task {i} during the incident lifecycle.")
    doc.save(str(d / "b_roles.docx"))


def _docs(tmp: Path):
    _make_docs(tmp)
    rep = ingest_directory(tmp, tmp / "no_manifest.yaml")
    return [d for d in rep.documents if d.status == "ok"]


CFG = ChunkingConfig(60, 10, 10)


def test_full_build_writes_consistent_files():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        docs = _docs(tmp)
        store, out = FakeStore(), tmp / "index"
        m = build_index(docs, [], chunk_cfg=CFG, count_tokens=wc, embedder=FakeEmbedder(),
                        store=store, out_dir=out)
        chunks = read_chunks(out / CHUNKS_FILE)
        assert m.n_chunks == len(chunks) == len(store.chunks) == len(store.embeddings)
        assert m.n_documents == 2 and m.embedding_dim == 8 and m.embedding_model == "fake-model"
        assert m.chunking == {"chunk_tokens": 60, "overlap_tokens": 10, "min_tokens": 10}
        assert load_manifest(out / MANIFEST_FILE) == m
        assert len({c.chunk_id for c in chunks}) == len(chunks)
        assert all(c.doc_chunk_count == sum(1 for x in chunks if x.doc_id == c.doc_id) for c in chunks)
        by_file = {d["filename"]: d for d in m.documents}
        assert by_file["b_roles.docx"]["chunks_with_section"] == by_file["b_roles.docx"]["n_chunks"]
        txt = [c for c in chunks if c.filename == "a_policy.txt"]
        assert {c.section for c in txt} >= {"1. Purpose", "2. Scope"}
        assert all(c.page is None and "part" in c.location() for c in txt)
        assert np.allclose(np.linalg.norm(store.embeddings, axis=1), 1.0, atol=1e-5)


def test_embed_text_contains_title_and_section_but_chunk_text_does_not():
    with tempfile.TemporaryDirectory() as t:
        docs = _docs(Path(t))
        store = FakeStore()
        build_index(docs, [], chunk_cfg=CFG, count_tokens=wc, embedder=FakeEmbedder(),
                    store=store, out_dir=Path(t) / "index")
        c = next(c for c in store.chunks if c.filename == "b_roles.docx")
        assert c.embed_text().startswith(f"{c.title} - Roles\n")
        assert not c.text.startswith(c.title)


def test_failed_build_leaves_no_manifest_and_removes_the_old_one():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        docs = _docs(tmp)
        out = tmp / "index"
        build_index(docs, [], chunk_cfg=CFG, count_tokens=wc, embedder=FakeEmbedder(),
                    store=FakeStore(), out_dir=out)
        assert (out / MANIFEST_FILE).is_file()
        try:
            build_index(docs, [], chunk_cfg=CFG, count_tokens=wc, embedder=BrokenEmbedder(),
                        store=FakeStore(), out_dir=out)
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected RuntimeError")
        assert not (out / MANIFEST_FILE).exists()          # incomplete index is detectable


def test_identical_files_are_reported_instead_of_silently_duplicated():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        text = "\n".join(f"Same content line {i} here." for i in range(20))
        (tmp / "one.txt").write_text(text, encoding="utf-8")
        (tmp / "two.txt").write_text(text, encoding="utf-8")
        docs = [d for d in ingest_directory(tmp, tmp / "none.yaml").documents if d.status == "ok"]
        try:
            build_index(docs, [], chunk_cfg=CFG, count_tokens=wc, embedder=FakeEmbedder(),
                        store=FakeStore(), out_dir=tmp / "index")
        except ValueError as exc:
            assert "duplicate chunk_id" in str(exc)
        else:
            raise AssertionError("expected ValueError")
