"""Choose the excerpt shown with a source: the sentences that best match the answer.

No LLM involved and nothing is rewritten: the excerpt is a verbatim run of consecutive
sentences from the chunk (whitespace collapsed), with "..." marking cut ends.
"""
from __future__ import annotations

import re

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9•“\"(\[])|\n\s*\n")
_WORD = re.compile(r"[a-z0-9][a-z0-9\-]{2,}")
_STOP = {"the", "and", "for", "that", "with", "this", "from", "are", "was", "were", "has",
         "have", "been", "should", "which", "what", "when", "how", "does", "can", "will",
         "not", "but", "into", "their", "they", "them", "its", "all", "any", "may", "such",
         "than", "then", "also", "about", "according", "guide", "document"}


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def best_excerpt(chunk_text: str, focus_text: str, max_chars: int = 350) -> str:
    sentences = [" ".join(s.split()) for s in _SENT_SPLIT.split(chunk_text) if s and s.strip()]
    if not sentences:
        return ""
    whole = " ".join(sentences)
    if len(whole) <= max_chars:
        return whole

    focus = _words(focus_text)
    scores = [len(_words(s) & focus) for s in sentences]
    best = max(range(len(sentences)), key=lambda i: (scores[i], -i))   # ties: earliest

    start = end = best
    total = len(sentences[best])
    while True:                                   # grow the window: next sentence first
        if end + 1 < len(sentences) and total + 1 + len(sentences[end + 1]) <= max_chars:
            end += 1
            total += 1 + len(sentences[end])
        elif start > 0 and total + 1 + len(sentences[start - 1]) <= max_chars:
            start -= 1
            total += 1 + len(sentences[start])
        else:
            break
    text = " ".join(sentences[start:end + 1])
    if len(text) > max_chars:                     # one very long sentence
        text = text[:max_chars].rsplit(" ", 1)[0]
        end_cut = True
    else:
        end_cut = end < len(sentences) - 1
    return ("... " if start > 0 else "") + text + (" ..." if end_cut else "")
