"""Retrieval evaluation helpers for phase 5 (no models; scripts/eval_retrieval.py runs them).

Ground truth is the same as in phase 3/4: a "gold chunk" is a chunk of the right document and
page that contains the question's evidence quote (evaluation.baseline.gold_chunk_ids).

Besides Hit@k / MRR, the key analysis for a second retriever is the RESCUE analysis:
among the questions whose gold chunk the base retriever (vector) misses in its top-k, how many
does the other retriever (BM25) find? That, and not "the results differ", is what justifies a
hybrid. `union` is the upper bound of what fusing the two lists could reach.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from sia.evaluation.baseline import gold_chunk_ids
from sia.evaluation.dataset import answerable
from sia.retrieval.tokenizer import is_identifier, tokenize
from sia.retrieval.vector_retriever import RetrievedChunk


class _Retriever(Protocol):
    def retrieve(self, query: str, k: int | None = ...) -> list[RetrievedChunk]: ...


@dataclass(frozen=True)
class EvalItem:
    id: str
    question: str
    doc: str                       # filename of the gold document
    gold: frozenset[str]           # chunk ids that contain the evidence quote
    has_identifier: bool           # the question contains a token such as "800-61r3" / "sp800"


@dataclass
class SystemRun:
    name: str
    chunk_ranks: list[int | None]      # rank of the first gold chunk, None = not in the top-k
    doc_ranks: list[int | None]        # rank of the first chunk of the gold document
    distinct_docs_top5: list[int]      # how many different documents the top 5 come from
    latency_ms: list[float] = field(default_factory=list)


def has_identifier(question: str) -> bool:
    return any(is_identifier(t) for t in tokenize(question))


def build_items(questions: list[dict], chunks_by_id: dict, lang: str = "en"
                ) -> tuple[list[EvalItem], list[str]]:
    """English answerable questions whose evidence quote lies inside ONE chunk.

    Returns (items, ids of questions dropped because no single chunk contains the quote,
    e.g. the quote spans a chunk boundary).
    """
    items, unmatched = [], []
    for q in answerable(questions, lang=lang):
        gold = gold_chunk_ids(q, chunks_by_id)
        if not gold:
            unmatched.append(q["id"])
            continue
        items.append(EvalItem(q["id"], q["question"], q["doc"], frozenset(gold),
                              has_identifier(q["question"])))
    return items, unmatched


def evaluate_system(name: str, retriever: _Retriever, items: Sequence[EvalItem], k: int = 10
                    ) -> SystemRun:
    run = SystemRun(name, [], [], [])
    for item in items:
        t0 = time.perf_counter()
        results = retriever.retrieve(item.question, k=k)
        run.latency_ms.append((time.perf_counter() - t0) * 1000)
        chunk_rank = doc_rank = None
        for r in results:
            if doc_rank is None and r.chunk.filename == item.doc:
                doc_rank = r.rank
            if chunk_rank is None and r.chunk.chunk_id in item.gold:
                chunk_rank = r.rank
        run.chunk_ranks.append(chunk_rank)
        run.doc_ranks.append(doc_rank)
        run.distinct_docs_top5.append(len({r.chunk.doc_id for r in results[:5]}))
    return run


# ---------------------------------------------------------------------- metrics
def hit_at(ranks: Sequence[int | None], k: int) -> float:
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks) if ranks else float("nan")


def mrr_at(ranks: Sequence[int | None], k: int) -> float:
    return sum(1.0 / r for r in ranks if r is not None and r <= k) / len(ranks) if ranks \
        else float("nan")


def subset(values: Sequence, indices: Sequence[int]) -> list:
    return [values[i] for i in indices]


def summarize(run: SystemRun, indices: Sequence[int], ks: Sequence[int] = (1, 3, 5, 10)) -> dict:
    ranks = subset(run.chunk_ranks, indices)
    doc_ranks = subset(run.doc_ranks, indices)
    kmax = max(ks)
    distinct = subset(run.distinct_docs_top5, indices)
    return {
        "n": len(indices),
        "chunk_hit": {k: hit_at(ranks, k) for k in ks},
        f"mrr@{kmax}": mrr_at(ranks, kmax),
        "doc_hit@5": hit_at(doc_ranks, 5),
        "distinct_docs_top5": sum(distinct) / len(distinct) if distinct else float("nan"),
    }


def rescue_stats(base: SystemRun, other: SystemRun, indices: Sequence[int], k: int) -> dict:
    """Gold-chunk overlap at cut-off k between a base run and another run."""
    def found(run: SystemRun, i: int) -> bool:
        r = run.chunk_ranks[i]
        return r is not None and r <= k

    both = [i for i in indices if found(base, i) and found(other, i)]
    rescued = [i for i in indices if not found(base, i) and found(other, i)]   # other finds, base misses
    lost = [i for i in indices if found(base, i) and not found(other, i)]      # base finds, other misses
    neither = [i for i in indices if not found(base, i) and not found(other, i)]
    n = len(indices)
    base_hits = len(both) + len(lost)
    return {
        "k": k, "n": n, "both": len(both), "rescued": rescued, "lost": lost, "neither": len(neither),
        "base_hit": base_hits / n if n else float("nan"),
        "other_hit": (len(both) + len(rescued)) / n if n else float("nan"),
        "union_hit": (len(both) + len(rescued) + len(lost)) / n if n else float("nan"),
        "union_gain": len(rescued) / n if n else float("nan"),
        "base_misses": len(rescued) + len(neither),
    }


def split_indices(items: Sequence[EvalItem], seed: int = 0) -> tuple[list[int], list[int]]:
    """Deterministic dev/test halves (by sorted question id, then a seeded shuffle)."""
    order = sorted(range(len(items)), key=lambda i: items[i].id)
    random.Random(seed).shuffle(order)
    mid = len(order) // 2
    return sorted(order[:mid]), sorted(order[mid:])


# ---------------------------------------------------------------------- paired document bootstrap
def per_question(run: SystemRun, metric: str, k: int) -> np.ndarray:
    """Per-question metric values: metric is 'hit' (0/1 gold chunk within k) or 'rr' (reciprocal rank)."""
    out = np.zeros(len(run.chunk_ranks))
    for i, r in enumerate(run.chunk_ranks):
        if r is not None and r <= k:
            out[i] = 1.0 if metric == "hit" else 1.0 / r
    return out


def paired_bootstrap(a: np.ndarray, b: np.ndarray, docs: Sequence[str], n_boot: int = 2000,
                     seed: int = 0) -> dict:
    """95% interval for mean(a) - mean(b), resampling whole DOCUMENTS (not single questions).

    Questions about the same document are not independent (they share its wording and its
    difficulty), so resampling questions would make the interval too narrow. `a` and `b` are
    per-question values of two systems on the same questions (paired).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) != len(b) or len(a) != len(docs):
        raise ValueError("a, b and docs must have the same length")
    if len(a) == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "p_better": float("nan"),
                "n_docs": 0}
    names = sorted(set(docs))
    pos = {d: i for i, d in enumerate(names)}
    diff_sum = np.zeros(len(names))
    count = np.zeros(len(names))
    for x, y, d in zip(a, b, docs):
        diff_sum[pos[d]] += x - y
        count[pos[d]] += 1
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(names), size=(n_boot, len(names)))
    boot = diff_sum[draw].sum(axis=1) / count[draw].sum(axis=1)
    return {"diff": float((a - b).mean()), "lo": float(np.percentile(boot, 2.5)),
            "hi": float(np.percentile(boot, 97.5)), "p_better": float((boot > 0).mean()),
            "n_docs": len(names)}
