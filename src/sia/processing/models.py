"""Data structures produced by the processing stage (cleaning + chunking)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    """One retrievable piece of a document.

    `text` is a verbatim slice of the (cleaned) page text, so it can be shown to
    the user as an excerpt without any rewriting.
    """
    chunk_id: str                    # "{doc_id}-p{page:03d}-{k:02d}", deterministic
    doc_id: str                      # from ingestion: sha256(file bytes)[:12]
    filename: str
    title: str
    doc_kind: str | None
    publisher: str | None
    publication_date: str | None
    page: int | None                 # PDF page (1-based); None for DOCX/TXT
    chunk_index: int                 # running index inside the document
    text: str
    n_tokens: int                    # counted with the embedding model's tokenizer
    char_start: int                  # offsets inside the cleaned page text:
    char_end: int                    #   page_text[char_start:char_end] == text
    section: str | None = None       # nearest preceding heading (structure.py)
    content_type: str = "prose"      # "prose" | "code" | "table" (structure.py)
    doc_chunk_count: int = 0         # number of chunks of the whole document

    def location(self) -> str:
        """Human-readable origin of the chunk, used in source references.

        PDF: "p. 12, section: Containment". DOCX/TXT have no pages, so the chunk's
        position is given instead: "section: Roles (part 2 of 3)" or "part 2 of 3".
        """
        parts: list[str] = []
        if self.page is not None:
            parts.append(f"p. {self.page}")
        if self.section:
            parts.append(f"section: {self.section}")
        if self.page is None:
            parts.append(f"part {self.chunk_index + 1} of {self.doc_chunk_count}"
                         if not self.section else f"(part {self.chunk_index + 1} of {self.doc_chunk_count})")
        return ", ".join(parts).replace(", (", " (")

    def embed_text(self) -> str:
        """Text that is embedded / indexed: a small context header + the chunk.

        The header tells the model which document and section the chunk belongs
        to. It is NOT shown to the user; `text` stays the displayed excerpt.
        """
        header = self.title if not self.section else f"{self.title} - {self.section}"
        return f"{header}\n{self.text}"
