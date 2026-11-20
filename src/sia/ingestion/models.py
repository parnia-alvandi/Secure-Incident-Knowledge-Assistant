"""Data structures produced by the ingestion pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field

# Metadata fields every document must have in manifest.yaml.
REQUIRED_META_FIELDS = (
    "title", "publisher", "doc_kind", "license",
    "publication_date", "acquired_date", "language", "topics", "description",
)
VALID_DOC_KINDS = ("public", "synthetic")


@dataclass
class DocumentMeta:
    doc_id: str                      # sha256(file bytes)[:12]; changes if content changes
    filename: str
    file_type: str                   # "pdf" | "docx" | "txt"
    title: str
    publisher: str | None = None
    doc_kind: str | None = None      # "public" | "synthetic"
    source_url: str | None = None    # required for public documents
    license: str | None = None
    publication_date: str | None = None
    acquired_date: str | None = None
    language: str | None = None
    topics: list[str] = field(default_factory=list)
    description: str | None = None
    missing_fields: list[str] = field(default_factory=list)  # incomplete metadata report


@dataclass
class PageText:
    page: int | None                 # real page number for PDF; None for DOCX/TXT
    text: str
    suspicious: bool = False         # too little text (possibly scanned/image-only)


@dataclass
class LoadedDocument:
    meta: DocumentMeta
    pages: list[PageText]
    status: str                      # "ok" | "empty" | "error"
    error: str | None = None

    @property
    def n_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)

    @property
    def suspicious_pages(self) -> list[int | None]:
        return [p.page for p in self.pages if p.suspicious]


@dataclass
class IngestionReport:
    documents: list[LoadedDocument]
    manifest_orphans: list[str]      # manifest entries with no file in docs_dir
    skipped_files: list[str]         # files with unsupported extensions

    def count(self, status: str) -> int:
        return sum(1 for d in self.documents if d.status == status)