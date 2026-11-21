"""chunks.jsonl (single source of truth for chunks) and index_manifest.json.

Everything else (Chroma vectors, BM25 in phase 5, agent tools in phase 6) is derived
from chunks.jsonl, so the vector index can always be rebuilt from it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from sia.processing.models import Chunk

CHUNKS_FILE = "chunks.jsonl"
MANIFEST_FILE = "index_manifest.json"
MANIFEST_SCHEMA_VERSION = 1

_CHUNK_FIELDS = {f.name for f in fields(Chunk)}


def write_chunks(path: Path, chunks: list[Chunk]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_chunks(path: Path) -> list[Chunk]:
    """Read chunks; unknown keys (from a newer schema) are ignored."""
    chunks: list[Chunk] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                chunks.append(Chunk(**{k: v for k, v in d.items() if k in _CHUNK_FIELDS}))
    return chunks


@dataclass
class IndexManifest:
    """Describes how an index was built. Written LAST: no manifest = incomplete index."""
    created_at: str
    embedding_model: str
    embedding_dim: int
    query_instruction: str
    chunking: dict
    cleaning: dict
    collection: str
    n_documents: int
    n_chunks: int
    documents: list[dict] = field(default_factory=list)          # filename, doc_id, n_chunks, ...
    skipped_documents: list[dict] = field(default_factory=list)  # status != ok
    timings: dict = field(default_factory=dict)
    libraries: dict = field(default_factory=dict)
    schema_version: int = MANIFEST_SCHEMA_VERSION


def save_manifest(path: Path, manifest: IndexManifest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_manifest(path: Path) -> IndexManifest | None:
    path = Path(path)
    if not path.is_file():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    known = {f.name for f in fields(IndexManifest)}
    return IndexManifest(**{k: v for k, v in d.items() if k in known})
