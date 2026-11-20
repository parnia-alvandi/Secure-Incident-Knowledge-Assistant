"""Text normalization that must NOT destroy meaning.

Deliberately preserved: letter case, digits, punctuation (identifiers such as
CVE-2021-44228 or T1059.001 matter for keyword search) and paragraph boundaries.
"""
from __future__ import annotations

import re
import unicodedata

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# "contain-\nment" -> "containment". Only lowercase-to-lowercase, to avoid joining
# things like "CVE-\n2021" or "T1059-\nT1060".
_HYPHEN_BREAK = re.compile(r"(?<=[a-z])-\n(?=[a-z])")


def normalize_text(text: str) -> str:
    # NFKC turns typographic ligatures (e.g. "ﬁ") into plain letters.
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00ad", "")            # soft hyphen
    text = _CONTROL.sub("", text)
    text = _HYPHEN_BREAK.sub("", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)       # collapse spaces/tabs
    text = re.sub(r" ?\n ?", "\n", text)          # trim around newlines
    text = re.sub(r"\n{3,}", "\n\n", text)        # at most one blank line
    return text.strip()