"""Page-level cleaning that runs AFTER ingestion.

Ingestion (sia.ingestion.normalize) already handles NFKC, control characters,
hyphenation at line ends and whitespace. The only thing it cannot do is remove
running headers/footers, because that needs to compare pages with each other.

Rule: a line is a header/footer if
  * it sits in the first or last `zone_lines` non-empty lines of a page,
  * it is short (<= `max_line_chars`),
  * and the same line appears in at least `min_page_fraction` of the pages.
"Same line" means: identical text after masking ONLY page counters ("Page 3 of
17" / "Page | 17" at the start or end of the line, a bare number line, a number
at the very start/end of the line). Digits
inside a line ("800-61r3", "CVE-2021-44228", "Step 2: ...") are never masked, so
body text that merely looks alike on several pages is not removed.
Only documents with at least `min_pages` non-empty pages get header/footer removal.
Separately, "this page has been left blank intentionally" pages are emptied (they
carry no information and only add retrieval noise); this applies to any document.
Every removal is reported, nothing is dropped silently.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace

from sia.ingestion.models import PageText

_SPACES = re.compile(r"\s+")
# "Page 3 of 17 | Product ID: AA24-060A", "Page | 17 TLP:CLEAR", "... see page 12"
_PAGE_START = re.compile(r"^page\s*\|?\s*\d+(\s+of\s+\d+)?\b")
_PAGE_END = re.compile(r"\bpage\s*\|?\s*\d+(\s+of\s+\d+)?$")
_BLANK_MARK = re.compile(r"left\s+blank\s+intentionally|intentionally\s+(left\s+)?blank")
_BLANK_MAX_CHARS = 120
_EDGE_NUM = re.compile(r"^\d+\s+|\s+\d+$")


@dataclass(frozen=True)
class CleanConfig:
    min_pages: int = 4
    zone_lines: int = 3
    min_page_fraction: float = 0.5
    max_line_chars: int = 100


@dataclass
class CleanReport:
    lines_removed: int = 0
    blank_pages: list = field(default_factory=list)   # page numbers emptied as "blank"
    # example raw line -> number of pages it was removed from
    patterns: dict[str, int] = field(default_factory=dict)


def _signature(line: str) -> str:
    s = _SPACES.sub(" ", line.strip()).lower()
    if s.isdigit():
        return "#"                                   # bare page number
    s = _PAGE_START.sub("page #", s)
    s = _PAGE_END.sub("page #", s)
    return _EDGE_NUM.sub(" ", s).strip()             # leading/trailing counter


def _zone_indices(lines: list[str], k: int) -> set[int]:
    non_empty = [i for i, ln in enumerate(lines) if ln.strip()]
    return set(non_empty[:k]) | set(non_empty[-k:])


def clean_pages(pages: list[PageText],
                cfg: CleanConfig = CleanConfig()) -> tuple[list[PageText], CleanReport]:
    """Return (new pages, report). The input pages are not modified."""
    report = CleanReport()

    # 0) "left blank intentionally" pages carry no information: empty them
    pages = list(pages)
    for i, p in enumerate(pages):
        if p.text and len(p.text) <= _BLANK_MAX_CHARS and _BLANK_MARK.search(p.text.lower()):
            report.blank_pages.append(p.page)
            pages[i] = replace(p, text="")

    content_pages = [p for p in pages if p.text.strip()]
    if len(content_pages) < cfg.min_pages:
        return pages, report

    # 1) count in how many pages each header-zone signature occurs
    counts: dict[str, int] = {}
    examples: dict[str, str] = {}
    for p in content_pages:
        lines = p.text.split("\n")
        seen: set[str] = set()
        for i in _zone_indices(lines, cfg.zone_lines):
            raw = lines[i]
            if len(raw.strip()) > cfg.max_line_chars:
                continue
            sig = _signature(raw)
            if sig and sig not in seen:
                seen.add(sig)
                counts[sig] = counts.get(sig, 0) + 1
                examples.setdefault(sig, raw.strip())
    threshold = math.ceil(cfg.min_page_fraction * len(content_pages))
    repeated = {sig for sig, n in counts.items() if n >= threshold}
    if not repeated:
        return pages, report

    # 2) rebuild the pages without those lines
    out: list[PageText] = []
    for p in pages:
        if not p.text.strip():
            out.append(p)
            continue
        lines = p.text.split("\n")
        drop = {i for i in _zone_indices(lines, cfg.zone_lines)
                if len(lines[i].strip()) <= cfg.max_line_chars
                and _signature(lines[i]) in repeated}
        if not drop:
            out.append(p)
            continue
        for i in drop:
            report.lines_removed += 1
            key = examples[_signature(lines[i])]
            report.patterns[key] = report.patterns.get(key, 0) + 1
        kept = [ln for i, ln in enumerate(lines) if i not in drop]
        text = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
        out.append(replace(p, text=text))
    return out, report
