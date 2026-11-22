"""Statistics for calibrating the evidence gate (phase 4c). Pure functions, no models.

The gate passes a question to the LLM only if its best retrieval score is >= tau.
Positive class = answerable questions (should pass); negative = unanswerable (should be blocked).
"""
from __future__ import annotations

import random
from typing import Sequence


def percentile(values: Sequence[float], p: float) -> float:
    s = sorted(values)
    if not s:
        return float("nan")
    idx = min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))
    return s[idx]


def auc(positives: Sequence[float], negatives: Sequence[float]) -> float:
    """P(a random answerable question scores higher than a random unanswerable one); ties count 1/2."""
    if not positives or not negatives:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def threshold_for_retention(answerable: Sequence[float], target: float) -> float:
    """Largest tau such that at least `target` of the answerable scores are >= tau."""
    s = sorted(answerable)
    k = int((1 - target) * len(s) + 1e-9)          # how many answerable questions may be lost
    return s[min(k, len(s) - 1)]


def retained(answerable: Sequence[float], tau: float) -> float:
    return sum(x >= tau for x in answerable) / len(answerable) if answerable else float("nan")


def blocked(unanswerable: Sequence[float], tau: float) -> float:
    return sum(x < tau for x in unanswerable) / len(unanswerable) if unanswerable else float("nan")


def split_half(items: list, rng: random.Random) -> tuple[list, list]:
    items = list(items)
    rng.shuffle(items)
    mid = len(items) // 2
    return items[:mid], items[mid:]


def score_questions(retriever, questions: list[dict], k: int = 5) -> list[dict]:
    """Retrieval features per question: top1, mean of top-3, margin top1-top2."""
    rows = []
    for q in questions:
        res = retriever.retrieve(q["question"], k=k)
        s = [r.score for r in res]
        rows.append({**{key: q.get(key) for key in ("id", "question", "category", "doc")},
                     "top1": s[0], "mean3": sum(s[:3]) / len(s[:3]),
                     "margin": s[0] - (s[1] if len(s) > 1 else s[0]),
                     "top1_title": res[0].chunk.title, "top1_loc": res[0].chunk.location()})
    return rows
