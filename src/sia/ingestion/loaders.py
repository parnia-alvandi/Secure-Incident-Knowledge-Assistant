"""Raw loaders for PDF, DOCX and TXT. They return RAW (un-normalized) page text."""
from __future__ import annotations

import logging
from pathlib import Path

from .models import PageText

log = logging.getLogger(__name__)


class DocumentLoadError(Exception):
    """Raised when a file cannot be read at all."""


def load_pdf(path: Path) -> list[PageText]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise DocumentLoadError("PDF is password-protected")
        pages: list[PageText] = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # noqa: BLE001 - one bad page must not kill the file
                log.warning("%s: page %d extraction failed: %s", path.name, i, exc)
                text = ""
            pages.append(PageText(page=i, text=text))
        return pages
    except DocumentLoadError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DocumentLoadError(f"cannot read PDF: {exc}") from exc


def load_docx(path: Path) -> list[PageText]:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        doc = Document(str(path))
    except Exception as exc:  # noqa: BLE001
        raise DocumentLoadError(f"cannot read DOCX: {exc}") from exc

    parts: list[str] = []
    # Iterate the body in document order so tables stay next to their paragraphs.
    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            para = Paragraph(child, doc)
            text = para.text.strip()
            if not text:
                continue
            style = (para.style.name or "") if para.style is not None else ""
            if style.startswith("Heading") or style == "Title":
                parts.append("\n" + text)        # headings get a blank line before them
            else:
                parts.append(text)
        elif child.tag.endswith("}tbl"):
            for row in Table(child, doc).rows:
                cells = [c.text.strip() for c in row.cells]
                if any(cells):
                    parts.append(" | ".join(cells))
    # DOCX has no fixed pages, so page = None.
    return [PageText(page=None, text="\n".join(parts))]


def load_txt(path: Path) -> list[PageText]:
    data = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return [PageText(page=None, text=data.decode(enc))]
        except UnicodeDecodeError:
            continue
    raise DocumentLoadError("cannot decode text file (tried utf-8, cp1252)")


LOADERS = {".pdf": load_pdf, ".docx": load_docx, ".txt": load_txt}