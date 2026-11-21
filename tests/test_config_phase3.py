"""Phase 3 config sections: optional, defaulted, validated."""
import tempfile
from pathlib import Path

from sia.config import load_settings

BASE = """
paths: {docs_dir: data/docs, vector_db_dir: vector_db, logs_dir: logs}
models: {embedding: BAAI/bge-small-en-v1.5, reranker: cross-encoder/ms-marco-MiniLM-L-6-v2}
llm: {provider: ollama, model: m, temperature: 0.0, num_ctx: 4096, num_predict: 300}
ingestion: {manifest_path: data/manifest.yaml, supported_extensions: [".pdf"], min_chars_per_page: 50}
logging: {level: INFO}
"""


def _load(extra: str):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.yaml"
        p.write_text(BASE + extra, encoding="utf-8")
        return load_settings(p)


def test_old_config_without_new_sections_gets_defaults():
    s = _load("")
    assert (s.chunking.chunk_tokens, s.chunking.overlap_tokens, s.chunking.min_tokens) == (256, 40, 40)
    assert s.embedding.batch_size == 32 and s.index.collection_name == "sia_chunks"


def test_overrides_are_applied():
    s = _load("chunking: {chunk_tokens: 128, overlap_tokens: 20, min_tokens: 20}\n"
              "index: {collection_name: abc}\n")
    assert s.chunking.chunk_tokens == 128 and s.index.collection_name == "abc"


def test_invalid_values_are_rejected():
    for extra in ("chunking: {chunk_tokens: 100, overlap_tokens: 100}\n",
                  "chunking: {chunk_toknes: 100}\n"):
        try:
            _load(extra)
        except ValueError:
            continue
        raise AssertionError(f"accepted: {extra}")
