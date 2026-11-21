"""Token-aware chunking that never crosses a page boundary.

Pipeline for one page of text:
  1. cut the text into small "units" (sentences / paragraphs / bullets; for
     DOCX and TXT every line is a unit as well),
  2. split any unit larger than the chunk size (lines -> words -> characters),
  3. pack consecutive units greedily into chunks of at most `chunk_tokens`,
     starting each new chunk with whole trailing units worth <= `overlap_tokens`
     from the previous one (overlap is sentence-aligned, never mid-sentence).

A chunk is always ONE CONTIGUOUS slice of the page text, so
`page_text[char_start:char_end] == chunk.text` and the excerpt shown to the user
is verbatim. Because only chunk boundaries depend on the unit splitter, a wrong
sentence split (e.g. after "e.g.") can only move a cut point; it cannot damage
the text.

Hard size limit of a chunk = chunk_tokens + min_tokens (the extra `min_tokens`
is used only to absorb a tiny tail instead of emitting a tiny extra chunk).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NamedTuple

from sia.ingestion.models import DocumentMeta, PageText

from .models import Chunk
from .tokens import TokenCounter


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_tokens: int = 256
    overlap_tokens: int = 40
    min_tokens: int = 40

    def __post_init__(self) -> None:
        if self.chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be positive")
        if not 0 <= self.overlap_tokens < self.chunk_tokens:
            raise ValueError("overlap_tokens must be in [0, chunk_tokens)")
        if self.min_tokens < 0:
            raise ValueError("min_tokens must be >= 0")


class Span(NamedTuple):
    start: int
    end: int


class _Unit(NamedTuple):
    start: int
    end: int
    n: int      # token count


# --- boundary detection ------------------------------------------------------

_PARAGRAPH = re.compile(r"\n[ \t]*\n")
_SENT_END = re.compile(r"[.!?][\"')\]]*\s+(?=[\"'“(\[A-Z0-9•●▪◦■\-–*])")
_BULLET = re.compile(r"\n(?=[•●▪◦■]|-\s|\d{1,2}[.)]\s|[a-z][.)]\s)")
_NEWLINE = re.compile(r"\n")
_WORD = re.compile(r"\S+\s*")


def _boundaries(text: str, newline_is_boundary: bool) -> list[int]:
    cuts = {0, len(text)}
    for rx in (_PARAGRAPH, _SENT_END, _BULLET):
        cuts.update(m.end() for m in rx.finditer(text))
    if newline_is_boundary:
        cuts.update(m.end() for m in _NEWLINE.finditer(text))
    return sorted(c for c in cuts if 0 <= c <= len(text))


def _char_slices(text: str, start: int, end: int, limit: int,
                 count: TokenCounter) -> list[_Unit]:
    """Last resort for a single 'word' that is larger than the chunk size."""
    out: list[_Unit] = []
    pos = start
    while pos < end:
        size = end - pos
        n = count(text[pos:pos + size])
        while n > limit and size > 1:
            size = max(1, min(size - 1, int(size * limit / n * 0.95)))
            n = count(text[pos:pos + size])
        out.append(_Unit(pos, pos + size, n))
        pos += size
    return out


def _split_words(text: str, start: int, end: int, limit: int,
                 count: TokenCounter) -> list[_Unit]:
    out: list[_Unit] = []
    for m in _WORD.finditer(text, start, end):
        n = count(m.group())
        if n > limit:
            out.extend(_char_slices(text, m.start(), m.end(), limit, count))
        else:
            out.append(_Unit(m.start(), m.end(), n))
    return out


def _split_oversize(text: str, start: int, end: int, limit: int,
                    count: TokenCounter) -> list[_Unit]:
    """Split a unit that is bigger than `limit`: first by lines, then words."""
    out: list[_Unit] = []
    pos = start
    cuts = [m.end() for m in _NEWLINE.finditer(text, start, end)] + [end]
    for cut in cuts:
        if cut <= pos:
            continue
        piece = text[pos:cut]
        if piece.strip():
            n = count(piece)
            if n <= limit:
                out.append(_Unit(pos, cut, n))
            else:
                out.extend(_split_words(text, pos, cut, limit, count))
        pos = cut
    return out


def _make_units(text: str, cfg: ChunkingConfig, count: TokenCounter,
                newline_is_boundary: bool) -> list[_Unit]:
    cuts = _boundaries(text, newline_is_boundary)
    units: list[_Unit] = []
    for a, b in zip(cuts, cuts[1:]):
        piece = text[a:b]
        if not piece.strip():
            continue
        n = count(piece)
        if n <= cfg.chunk_tokens:
            units.append(_Unit(a, b, n))
        else:
            units.extend(_split_oversize(text, a, b, cfg.chunk_tokens, count))
    return units


# --- packing -----------------------------------------------------------------

def _pack(units: list[_Unit], cfg: ChunkingConfig) -> list[tuple[int, int]]:
    """Greedy packing with sentence-aligned overlap. Returns (start, end) offsets."""
    n = len(units)
    suffix = [0] * (n + 1)                       # suffix[i] = tokens of units[i:]
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + units[i].n

    spans: list[tuple[int, int]] = []
    i = 0
    while i < n:
        total, j = 0, i
        while j < n:
            if total + units[j].n > cfg.chunk_tokens and j > i:
                break
            total += units[j].n
            j += 1
        # absorb a tiny tail instead of creating a tiny extra chunk
        if j < n and suffix[j] <= cfg.min_tokens \
                and total + suffix[j] <= cfg.chunk_tokens + cfg.min_tokens:
            j = n
        spans.append((units[i].start, units[j - 1].end))
        if j >= n:
            break
        # overlap: step back over whole units worth <= overlap_tokens (always > i)
        k, acc = j, 0
        while k - 1 > i and acc + units[k - 1].n <= cfg.overlap_tokens:
            k -= 1
            acc += units[k].n
        i = k
    return spans


# --- public API --------------------------------------------------------------

def chunk_text(text: str, cfg: ChunkingConfig, count: TokenCounter,
               newline_is_boundary: bool = False) -> list[Span]:
    """Chunk one page of text; returns trimmed (start, end) offsets."""
    units = _make_units(text, cfg, count, newline_is_boundary)
    spans: list[Span] = []
    for a, b in _pack(units, cfg):
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if a < b:
            spans.append(Span(a, b))
    return spans


def chunk_document(meta: DocumentMeta, pages: list[PageText], cfg: ChunkingConfig,
                   count: TokenCounter) -> list[Chunk]:
    """Chunk every page of a document. Chunks never cross page boundaries."""
    newline_is_boundary = meta.file_type != "pdf"    # DOCX/TXT: a line is a paragraph
    chunks: list[Chunk] = []
    index = 0
    for page in pages:
        if not page.text.strip():
            continue
        tag = f"p{page.page:03d}" if page.page is not None else "pNA"
        for k, (a, b) in enumerate(chunk_text(page.text, cfg, count, newline_is_boundary)):
            text = page.text[a:b]
            chunks.append(Chunk(
                chunk_id=f"{meta.doc_id}-{tag}-{k:02d}",
                doc_id=meta.doc_id, filename=meta.filename, title=meta.title,
                doc_kind=meta.doc_kind, publisher=meta.publisher,
                publication_date=meta.publication_date,
                page=page.page, chunk_index=index, text=text,
                n_tokens=count(text), char_start=a, char_end=b,
            ))
            index += 1
    return chunks
