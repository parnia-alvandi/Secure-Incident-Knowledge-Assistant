"""Rule-based information extraction: section headings and content type.

No LLM and no font information are available (pypdf returns plain text), so
headings are found with transparent heuristics. They are best-effort: when no
heading is found the chunk simply has `section = None`. Quality is audited with
`scripts/inspect_chunks.py --sections`, never assumed.

Heading rules (a candidate line is short, has no terminal punctuation, is not a
figure/table caption, URL or table row):
  1. numbered     "3.2 Containment", "Appendix B ..."
  2. ALL CAPS     "VERSION HISTORY"
  3. DOCX         any line that follows a blank line (the DOCX loader puts a blank
                  line before every Heading/Title paragraph)
  4. Title Case   short line after a blank line / page start, e.g. "Executive Summary"
A chunk's section is the last heading at or before its first character, carried
across pages.
"""
from __future__ import annotations

import re
from typing import NamedTuple

from sia.ingestion.models import PageText

from .models import Chunk

MAX_HEADING_CHARS = 100

_STOP = {"and", "or", "of", "for", "the", "a", "an", "to", "in", "on", "with", "by",
         "at", "from", "as", "into", "is", "are"}
_NUMBERED = re.compile(r"^\d+(\.\d+){0,3}\.?\s+[A-Z]")
_APPENDIX = re.compile(r"^(appendix|section|chapter|part)\s+[A-Z0-9]", re.I)
_CAPTION = re.compile(r"^(figure|table)\s+[\dA-Z]", re.I)
_BULLET = ("•", "●", "▪", "◦", "■", "- ", "* ", "– ")
_TERMINAL = ".,;:!?)"


class Heading(NamedTuple):
    offset: int      # offset of the first character of the heading in the page text
    text: str


def _is_candidate(line: str) -> bool:
    n_words = len(line.split())
    if not 3 <= len(line) <= MAX_HEADING_CHARS or n_words > 12:
        return False
    if sum(ch.isalpha() for ch in line) < 3:
        return False
    if line.endswith(tuple(_TERMINAL)) or line.startswith(_BULLET):
        return False
    if " | " in line or "http" in line.lower() or "@" in line or _CAPTION.match(line):
        return False
    if "TLP:" in line.upper():                       # TLP:CLEAR style banners
        return False
    return True


def _is_title_case(line: str) -> bool:
    words = line.split()
    if len(words) > 8 or not line[0].isupper() or words[-1].lower() in _STOP:
        return False
    content = [w for w in words if w.lower() not in _STOP]
    return bool(content) and sum(w[0].isupper() for w in content) / len(content) >= 0.6


def find_headings(text: str, file_type: str) -> list[Heading]:
    headings: list[Heading] = []
    pos, prev_blank = 0, True
    for raw in text.split("\n"):
        line = raw.strip()
        start = pos + (len(raw) - len(raw.lstrip()))
        pos += len(raw) + 1
        if not line:
            prev_blank = True
            continue
        was_blank, prev_blank = prev_blank, False
        if not _is_candidate(line):
            continue
        is_heading = (
            bool(_NUMBERED.match(line) or _APPENDIX.match(line))
            or (sum(ch.isalpha() for ch in line) >= 4 and line.upper() == line
                and len(line.split()) <= 10)
            or (file_type == "docx" and was_blank)
            or (was_blank and _is_title_case(line))
        )
        if is_heading:
            headings.append(Heading(start, line))
    return headings


# --- content type ------------------------------------------------------------

_CODE_CHARS = set("{}();=<>[]$\\|&*`")
_CODE_STARTS = ("$ ", "#!", "import ", "def ", "function ", "var ", "<?", "<%", "SELECT ",
                "index=", "alert ", "rule ")


def _is_code_line(line: str) -> bool:
    s = line.strip()
    if len(s) < 6:
        return False
    if s.startswith(_CODE_STARTS):
        return True
    return sum(ch in _CODE_CHARS for ch in s) / len(s) > 0.10


def classify_content(text: str) -> str:
    """'table' (DOCX table rows), 'code' (mostly code-like lines) or 'prose'."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return "prose"
    if sum(" | " in ln for ln in lines) / len(lines) >= 0.4 and len(lines) >= 2:
        return "table"
    if len(lines) >= 4 and sum(_is_code_line(ln) for ln in lines) / len(lines) >= 0.4:
        return "code"
    return "prose"


# --- annotation ----------------------------------------------------------------

def annotate_chunks(chunks: list[Chunk], pages: list[PageText], file_type: str) -> None:
    """Fill `section` and `content_type` on the chunks of ONE document (in place).

    `pages` must be the same (cleaned) pages the chunks were built from, because
    heading and chunk offsets refer to that text.
    """
    by_page: dict[int | None, list[Chunk]] = {}
    for c in chunks:
        by_page.setdefault(c.page, []).append(c)

    current: str | None = None
    for page in pages:
        page_chunks = sorted(by_page.get(page.page, []), key=lambda c: c.char_start)
        heads = find_headings(page.text, file_type) if page.text.strip() else []
        for c in page_chunks:
            before = [h for h in heads if h.offset <= c.char_start]
            c.section = before[-1].text if before else current
            c.content_type = classify_content(c.text)
        if heads:
            current = heads[-1].text
