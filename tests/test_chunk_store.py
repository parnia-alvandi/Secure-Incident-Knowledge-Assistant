"""Round-trip tests for chunks.jsonl and index_manifest.json."""
import json
import tempfile
from pathlib import Path

from sia.processing.models import Chunk
from sia.storage.chunk_store import (IndexManifest, load_manifest, read_chunks, save_manifest,
                                     write_chunks)


def _chunk(i: int, page=None) -> Chunk:
    return Chunk(chunk_id=f"d-{i}", doc_id="d", filename="f.txt", title="T ünïcode — ok",
                 doc_kind="public", publisher=None, publication_date="2025-04-01", page=page,
                 chunk_index=i, text="Line one.\nLine two with \"quotes\" and ü.", n_tokens=9,
                 char_start=0, char_end=10, section="Intro", content_type="prose", doc_chunk_count=3)


def test_chunks_roundtrip_including_none_and_unicode():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "sub" / "chunks.jsonl"
        chunks = [_chunk(0), _chunk(1, page=7)]
        write_chunks(path, chunks)
        assert read_chunks(path) == chunks
        assert len(path.read_text(encoding="utf-8").splitlines()) == 2   # one chunk per line


def test_unknown_keys_are_ignored_when_reading():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "chunks.jsonl"
        write_chunks(path, [_chunk(0)])
        row = json.loads(path.read_text(encoding="utf-8"))
        row["field_from_the_future"] = 1
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        assert read_chunks(path) == [_chunk(0)]


def test_manifest_roundtrip_and_missing_manifest():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "index_manifest.json"
        assert load_manifest(path) is None
        m = IndexManifest(created_at="2026-11-21T10:00:00+00:00", embedding_model="m",
                          embedding_dim=384, query_instruction="q", chunking={"chunk_tokens": 256},
                          cleaning={}, collection="c", n_documents=1, n_chunks=2,
                          documents=[{"filename": "f.txt"}])
        save_manifest(path, m)
        assert load_manifest(path) == m
