"""Loading the question dataset (eval/questions/*.json) for checks and evaluation."""
from __future__ import annotations

import json
from pathlib import Path


def squash(text: str) -> str:
    """Lower-cased letters and digits only (same idea as verify_questions.py)."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


def load_questions(questions_dir: Path) -> list[dict]:
    """All questions from every *.json file (a list, or an object with 'questions')."""
    out: list[dict] = []
    for path in sorted(Path(questions_dir).glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        items = data if isinstance(data, list) else data.get("questions")
        if items is None:
            raise ValueError(f"{path.name}: expected a list or an object with 'questions' "
                             f"(found keys: {list(data)[:8]})")
        out.extend(items)
    return out


def answerable(questions: list[dict], lang: str | None = None) -> list[dict]:
    """Answerable questions with an evidence quote; each gets a 'sq' (squashed quote)."""
    res = []
    for q in questions:
        if not q.get("answerable") or not q.get("evidence_quote"):
            continue
        if lang and q.get("lang") != lang:
            continue
        res.append({**q, "sq": squash(q["evidence_quote"])})
    return res


def unique_quotes(questions: list[dict]) -> list[dict]:
    """One entry per (doc, page, quote): Persian questions share their English pair's quote."""
    seen: dict[tuple, dict] = {}
    for q in answerable(questions):
        seen.setdefault((q["doc"], q.get("evidence_page"), q["sq"]), {
            "id": q["id"], "doc": q["doc"], "page": q.get("evidence_page"),
            "quote": q["evidence_quote"], "sq": q["sq"]})
    return list(seen.values())
