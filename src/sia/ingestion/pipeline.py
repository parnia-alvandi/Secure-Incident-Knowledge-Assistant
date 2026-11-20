"""Ingest a whole directory: load, normalize, attach metadata, produce a report."""
from __future__ import annotations

import hashlib
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from .loaders import LOADERS
from .models import (REQUIRED_META_FIELDS, VALID_DOC_KINDS, DocumentMeta,
                     IngestionReport, LoadedDocument)
from .normalize import normalize_text

log = logging.getLogger(__name__)


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    """Return {filename: entry}. A missing manifest is a warning, not a crash."""
    if not path.is_file():
        log.warning("Manifest not found: %s (titles will come from file names)", path)
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    entries: dict[str, dict[str, Any]] = {}
    for entry in raw.get("documents", []) or []:
        name = entry.get("filename")
        if not name:
            log.warning("Manifest entry without 'filename' ignored: %s", entry)
            continue
        if name in entries:
            log.warning("Duplicate manifest entry for %s; last one wins", name)
        entries[name] = entry
    return entries


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (date, datetime)):      # YAML parses 2024-05-01 as a date
        return value.isoformat()[:10]
    return str(value).strip() or None


def _build_meta(path: Path, entry: dict[str, Any] | None) -> DocumentMeta:
    doc_id = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    e = entry or {}
    topics = e.get("topics") or []
    meta = DocumentMeta(
        doc_id=doc_id,
        filename=path.name,
        file_type=path.suffix.lower().lstrip("."),
        title=_as_str(e.get("title")) or path.stem.replace("_", " ").replace("-", " "),
        publisher=_as_str(e.get("publisher")),
        doc_kind=_as_str(e.get("doc_kind")),
        source_url=_as_str(e.get("source_url")),
        license=_as_str(e.get("license")),
        publication_date=_as_str(e.get("publication_date")),
        acquired_date=_as_str(e.get("acquired_date")),
        language=_as_str(e.get("language")),
        topics=[str(t) for t in topics] if isinstance(topics, list) else [str(topics)],
        description=_as_str(e.get("description")),
    )
    missing = [f for f in REQUIRED_META_FIELDS if not getattr(meta, f)]
    if meta.doc_kind and meta.doc_kind not in VALID_DOC_KINDS:
        missing.append("doc_kind(invalid)")
    if meta.doc_kind == "public" and not meta.source_url:
        missing.append("source_url")
    if entry is None:
        missing.insert(0, "manifest_entry")
    meta.missing_fields = missing
    return meta


def ingest_directory(
    docs_dir: Path,
    manifest_path: Path,
    min_chars_per_page: int = 50,
    supported_extensions: tuple[str, ...] = (".pdf", ".docx", ".txt"),
) -> IngestionReport:
    manifest = load_manifest(manifest_path)
    documents: list[LoadedDocument] = []
    skipped: list[str] = []
    seen: set[str] = set()

    for path in sorted(docs_dir.iterdir()) if docs_dir.is_dir() else []:
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in supported_extensions or ext not in LOADERS:
            skipped.append(path.name)
            continue
        seen.add(path.name)
        meta = _build_meta(path, manifest.get(path.name))
        if meta.missing_fields:
            log.warning("%s: incomplete metadata: %s", path.name, meta.missing_fields)

        try:
            pages = LOADERS[ext](path)
            for p in pages:
                p.text = normalize_text(p.text)
                p.suspicious = len(p.text) < min_chars_per_page
            if not any(p.text for p in pages):
                documents.append(LoadedDocument(
                    meta, pages, "empty",
                    "no extractable text (empty or scanned/image-only; OCR is out of scope)"))
            else:
                documents.append(LoadedDocument(meta, pages, "ok"))
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop ingestion
            log.error("%s: failed: %s", path.name, exc)
            documents.append(LoadedDocument(meta, [], "error", str(exc)))

    orphans = sorted(set(manifest) - seen)
    for name in orphans:
        log.warning("Manifest lists %s but the file is not in %s", name, docs_dir)
    return IngestionReport(documents, orphans, skipped)