"""Tokenizer for the BM25 index (phase 5a).

Design (approved):
* lower-case, Unicode NFKC normalised, no stemming, no external library;
* a word is a run of letters/digits; hyphens and dots INSIDE a word keep it together, so
  identifiers survive: "800-61r3", "CVE-2021-45046", "AIT-IR-BR-009", "3.1";
* an identifier is emitted twice: whole ("800-61r3") and in pieces ("800", "61r3"). The whole
  token rewards an exact match, the pieces let a partial mention ("61r3") still match;
* underscores, slashes, apostrophes and other punctuation are separators
  ("syn_ir_playbook" -> syn, ir, playbook);
* a small English stop-word list and single letters are dropped.

The same function tokenizes chunks and queries, so both sides always agree.
"""
from __future__ import annotations

import re
import unicodedata

# letters/digits (any script) joined by single hyphens or dots; "_" is NOT a word character here
_WORD = re.compile(r"[^\W_]+(?:[-.][^\W_]+)*")
_SPLIT = re.compile(r"[-.]")

# Deliberately short: function words only. Negations and modal verbs ("not", "no", "must",
# "should") are kept because policies and playbooks use them to state requirements.
STOPWORDS: frozenset[str] = frozenset("""
a an and are as at be been being but by can could do does did for from had has have having he her
him his how i if in into is it its me my of on or our she so than that the their them then there
these they this those to too us was we were what when where which who whom why will with would you
your about above after again against all also am any because before below between both down during
each few further here more most other out over own same some such under until up very while
""".split())


def _keep(token: str) -> bool:
    if token in STOPWORDS:
        return False
    return not (len(token) == 1 and token.isalpha())     # lone letters ("s" from "it's") are noise


def tokenize(text: str | None) -> list[str]:
    """Tokens of `text`, in order, duplicates preserved (the caller decides what to do with them)."""
    if not text:
        return []
    text = unicodedata.normalize("NFKC", text).lower()
    out: list[str] = []
    for match in _WORD.finditer(text):
        word = match.group()
        if _keep(word):
            out.append(word)
        parts = _SPLIT.split(word)
        if len(parts) > 1:
            out.extend(p for p in parts if _keep(p))
    return out


def is_identifier(token: str) -> bool:
    """True for tokens such as "800-61r3", "cve-2021-45046", "sp800": letters AND digits."""
    return any(c.isalpha() for c in token) and any(c.isdigit() for c in token)
