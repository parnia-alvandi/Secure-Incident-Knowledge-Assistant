"""Parsing of the model's raw reply: citations, the ANSWER/SOURCES format, empty-answer detection."""
from __future__ import annotations

import re

CITATION = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
_BRACKET_TOKEN = re.compile(r"\[[^\]\n]{0,8}\]")                 # "[1]", "[11]", "[N1]", "[3, 4]"
_STRUCTURED = re.compile(r"ANSWER\s*:\s*(.*?)\s*SOURCES?\s*:\s*(.*)", re.S | re.I)
_ANSWER_ONLY = re.compile(r"ANSWER\s*:\s*(.*)", re.S | re.I)


def parse_citations(text: str, n_passages: int) -> tuple[list[int], list[int]]:
    """(valid numbers in order of first use, invalid numbers)."""
    valid: list[int] = []
    invalid: list[int] = []
    for m in CITATION.finditer(text):
        for num in re.split(r"[,;]", m.group(1)):
            n = int(num)
            bucket = valid if 1 <= n <= n_passages else invalid
            if n not in bucket:
                bucket.append(n)
    return valid, invalid


def has_answer_text(text: str) -> bool:
    """False when the reply is empty or consists only of citation markers like "[1]" or "[4] [1]"."""
    cleaned = _BRACKET_TOKEN.sub(" ", text or "")
    return len(re.findall(r"[A-Za-z0-9]", cleaned)) >= 2


def split_structured(text: str) -> tuple[str, list[int] | None]:
    """'ANSWER: ... SOURCES: 1, 3' -> (answer text, [1, 3]).

    Replies without the format are returned unchanged with sources None (inline [n] citations
    are then used). A reply with ANSWER: but no SOURCES: gives sources None as well.
    """
    m = _STRUCTURED.search(text)
    if m:
        return m.group(1).strip(), [int(x) for x in re.findall(r"\d+", m.group(2))]
    m = _ANSWER_ONLY.search(text)
    if m:
        return m.group(1).strip(), None
    return text.strip(), None
