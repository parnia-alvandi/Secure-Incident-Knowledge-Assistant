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


def _load(extra=""):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.yaml"
        p.write_text(BASE + extra, encoding="utf-8")
        return load_settings(p)


def test_defaults_without_sections():
    s = _load()
    assert s.retrieval.top_k == 5 and s.retrieval.min_score is None and s.answering.excerpt_chars == 350


def test_overrides_and_validation():
    s = _load("retrieval: {top_k: 3, min_score: 0.55}\nanswering: {excerpt_chars: 200}\n")
    assert s.retrieval.top_k == 3 and s.retrieval.min_score == 0.55 and s.answering.excerpt_chars == 200
    for extra in ("retrieval: {top_k: 0}\n", "retrieval: {min_score: 2}\n",
                  "answering: {excerpt_chars: 10}\n", "retrieval: {topk: 3}\n"):
        try:
            _load(extra)
        except ValueError:
            continue
        raise AssertionError(f"accepted {extra}")


def test_shipped_config_yaml_loads():
    s = load_settings(Path(__file__).resolve().parents[1] / "config.yaml")
    assert s.retrieval.top_k == 5 and s.retrieval.min_score is None
