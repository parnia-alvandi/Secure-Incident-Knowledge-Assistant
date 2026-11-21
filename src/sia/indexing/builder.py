"""Build the local index: ingestion output -> clean -> chunk -> annotate -> embed -> store.

The embedder and the vector store are injected, so the whole pipeline can be unit-tested
with fakes (no model download, no Chroma) and reused by the ablation runs.

Order of writes: chunks.jsonl, then the vector store, then index_manifest.json LAST.
A missing manifest therefore always means "this index is incomplete".
"""
from __future__ import annotations

import logging
import platform
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Protocol

import numpy as np

from sia.ingestion.models import LoadedDocument
from sia.processing.chunker import ChunkingConfig, chunk_document
from sia.processing.cleaner import CleanConfig, clean_pages
from sia.processing.models import Chunk
from sia.processing.structure import annotate_chunks
from sia.processing.tokens import TokenCounter
from sia.storage.chunk_store import (CHUNKS_FILE, MANIFEST_FILE, IndexManifest,
                                     save_manifest, write_chunks)

log = logging.getLogger(__name__)


class EmbedderLike(Protocol):
    model_name: str
    query_instruction: str

    @property
    def dim(self) -> int: ...
    def embed_passages(self, texts: list[str], show_progress: bool = ...) -> np.ndarray: ...


class StoreLike(Protocol):
    collection_name: str

    def rebuild(self, chunks: list[Chunk], embeddings: np.ndarray) -> None: ...


def collect_chunks(docs: list[LoadedDocument], cfg: ChunkingConfig, count: TokenCounter,
                   clean_cfg: CleanConfig | None = None) -> tuple[list[Chunk], list[dict]]:
    """Clean + chunk + annotate every document with status 'ok'.

    Returns (all chunks, one info dict per document).
    """
    clean_cfg = clean_cfg or CleanConfig()
    all_chunks: list[Chunk] = []
    infos: list[dict] = []
    owner: dict[str, str] = {}
    for d in docs:
        pages, creport = clean_pages(d.pages, clean_cfg)
        chunks = chunk_document(d.meta, pages, cfg, count)
        annotate_chunks(chunks, pages, d.meta.file_type)
        for c in chunks:
            c.doc_chunk_count = len(chunks)
            if c.chunk_id in owner:
                raise ValueError(f"duplicate chunk_id {c.chunk_id}: {d.meta.filename} and "
                                 f"{owner[c.chunk_id]} have identical content")
            owner[c.chunk_id] = d.meta.filename
        types = Counter(c.content_type for c in chunks)
        infos.append({
            "filename": d.meta.filename, "doc_id": d.meta.doc_id, "title": d.meta.title,
            "doc_kind": d.meta.doc_kind, "n_pages": len(d.pages), "n_chunks": len(chunks),
            "header_lines_removed": creport.lines_removed,
            "blank_pages": [p for p in creport.blank_pages],
            "content_types": dict(types),
            "chunks_with_section": sum(1 for c in chunks if c.section),
        })
        all_chunks.extend(chunks)
    return all_chunks, infos


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("sentence-transformers", "chromadb", "numpy", "pypdf", "python-docx"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def build_index(docs: list[LoadedDocument], skipped: list[dict], *, chunk_cfg: ChunkingConfig,
                count_tokens: TokenCounter, embedder: EmbedderLike, store: StoreLike,
                out_dir: Path, show_progress: bool = False,
                clean_cfg: CleanConfig | None = None) -> IndexManifest:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / MANIFEST_FILE).unlink(missing_ok=True)       # index is "incomplete" until the end

    t0 = time.perf_counter()
    chunks, infos = collect_chunks(docs, chunk_cfg, count_tokens, clean_cfg)
    if not chunks:
        raise ValueError("no chunks produced: nothing to index")
    t_chunk = time.perf_counter() - t0
    log.info("chunked %d documents into %d chunks in %.1fs", len(infos), len(chunks), t_chunk)

    t0 = time.perf_counter()
    embeddings = embedder.embed_passages([c.embed_text() for c in chunks],
                                         show_progress=show_progress)
    t_embed = time.perf_counter() - t0
    log.info("embedded %d chunks in %.1fs (%.1f chunks/s)", len(chunks), t_embed,
             len(chunks) / max(t_embed, 1e-9))

    write_chunks(out_dir / CHUNKS_FILE, chunks)
    t0 = time.perf_counter()
    store.rebuild(chunks, embeddings)
    t_store = time.perf_counter() - t0

    manifest = IndexManifest(
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        embedding_model=embedder.model_name, embedding_dim=int(embeddings.shape[1]),
        query_instruction=embedder.query_instruction,
        chunking=asdict(chunk_cfg), cleaning=asdict(clean_cfg or CleanConfig()),
        collection=store.collection_name,
        n_documents=len(infos), n_chunks=len(chunks),
        documents=infos, skipped_documents=skipped,
        timings={"chunking_s": round(t_chunk, 2), "embedding_s": round(t_embed, 2),
                 "store_s": round(t_store, 2),
                 "embedding_chunks_per_s": round(len(chunks) / max(t_embed, 1e-9), 1)},
        libraries=_versions())
    save_manifest(out_dir / MANIFEST_FILE, manifest)
    return manifest
