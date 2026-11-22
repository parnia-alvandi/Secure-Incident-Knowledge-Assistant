"""Baseline RAG pipeline: retrieve -> evidence gate -> LLM -> parse -> sources.

Grounding guarantees of this baseline (see the phase 4 design):
* sources, locations and excerpts are built from the index metadata, never by the LLM,
  so a fabricated reference or excerpt is impossible;
* citations the model invents (numbers outside the passages) are ignored and reported;
* an answer without a single valid citation is labelled "unverified", not "answered";
* the LLM is not called at all when the best retrieval score is below the gate threshold;
* any LLM failure becomes status "error" with a help message, never a made-up answer.
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Protocol

from sia.llm.ollama_client import LLMError, LLMResult
from sia.retrieval.vector_retriever import RetrievedChunk

from .excerpts import best_excerpt
from .prompt import SENTINEL, build_prompt

INSUFFICIENT_TEXT = "Not stated in the document collection (insufficient evidence)."
_CITATION = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")

STATUS_ANSWERED = "answered"
STATUS_INSUFFICIENT = "insufficient_evidence"
STATUS_UNVERIFIED = "unverified"
STATUS_ERROR = "error"


class _Retriever(Protocol):
    def retrieve(self, query: str, k: int | None = ...) -> list[RetrievedChunk]: ...


class _LLM(Protocol):
    def generate(self, system: str, user: str,
                 on_token: Callable[[str], None] | None = ...) -> LLMResult: ...


@dataclass
class SourceRef:
    chunk_id: str
    doc_id: str
    title: str
    filename: str
    page: int | None
    section: str | None
    location: str
    excerpt: str
    score: float
    n: int | None = None            # passage number the model saw ([n]); None if not shown


@dataclass
class AnswerResult:
    question: str
    status: str
    answer: str
    sources: list[SourceRef] = field(default_factory=list)   # passages the answer cites
    closest: list[SourceRef] = field(default_factory=list)   # shown when nothing is cited
    others: list[SourceRef] = field(default_factory=list)    # shown to the model, not cited (no excerpt)
    reason: str | None = None
    warnings: list[str] = field(default_factory=list)
    timings: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def parse_citations(text: str, n_passages: int) -> tuple[list[int], list[int]]:
    """(valid numbers in order of first use, invalid numbers)."""
    valid: list[int] = []
    invalid: list[int] = []
    for m in _CITATION.finditer(text):
        for num in re.split(r"[,;]", m.group(1)):
            n = int(num)
            bucket = valid if 1 <= n <= n_passages else invalid
            if n not in bucket:
                bucket.append(n)
    return valid, invalid


class RAGPipeline:
    def __init__(self, retriever: _Retriever, llm: _LLM, *, num_ctx: int, num_predict: int,
                 min_score: float | None = None, excerpt_chars: int = 350,
                 closest_n: int = 3) -> None:
        self.retriever = retriever
        self.llm = llm
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.min_score = min_score
        self.excerpt_chars = excerpt_chars
        self.closest_n = closest_n

    @classmethod
    def from_settings(cls, retriever, llm, settings) -> "RAGPipeline":
        return cls(retriever, llm, num_ctx=settings.llm.num_ctx,
                   num_predict=settings.llm.num_predict,
                   min_score=settings.retrieval.min_score,
                   excerpt_chars=settings.answering.excerpt_chars)

    # ------------------------------------------------------------------
    def _ref(self, r: RetrievedChunk, focus: str, n: int | None = None) -> SourceRef:
        c = r.chunk
        return SourceRef(chunk_id=c.chunk_id, doc_id=c.doc_id, title=c.title,
                         filename=c.filename, page=c.page, section=c.section,
                         location=c.location(),
                         excerpt=best_excerpt(c.text, focus, self.excerpt_chars),
                         score=round(r.score, 4), n=n)

    def ask(self, question: str,
            on_token: Callable[[str], None] | None = None) -> AnswerResult:
        t_start = time.perf_counter()
        timings: dict = {}

        t0 = time.perf_counter()
        results = self.retriever.retrieve(question)          # ValueError for an empty question
        timings["retrieval_s"] = round(time.perf_counter() - t0, 3)
        timings["best_score"] = round(results[0].score, 4) if results else None

        def finish(res: AnswerResult) -> AnswerResult:
            res.timings = {**timings, **res.timings, "total_s": round(time.perf_counter() - t_start, 2)}
            return res

        def closest(rs: list[RetrievedChunk]) -> list[SourceRef]:
            return [self._ref(r, question, None) for r in rs[: self.closest_n]]

        if not results:
            return finish(AnswerResult(question, STATUS_INSUFFICIENT, INSUFFICIENT_TEXT,
                                       reason="nothing was retrieved"))
        if self.min_score is not None and results[0].score < self.min_score:
            return finish(AnswerResult(
                question, STATUS_INSUFFICIENT, INSUFFICIENT_TEXT, closest=closest(results),
                reason=(f"best match score {results[0].score:.3f} is below the relevance "
                        f"threshold {self.min_score:.3f}; the LLM was not called")))

        bundle = build_prompt(question, results, self.num_ctx, self.num_predict)
        warnings: list[str] = []
        if bundle.dropped:
            warnings.append(f"{len(bundle.dropped)} retrieved passage(s) did not fit the "
                            f"context window and were not shown to the model")
        timings["passages_used"] = len(bundle.used)
        timings["prompt_tokens_estimated"] = bundle.estimated_tokens
        if not bundle.used:
            return finish(AnswerResult(question, STATUS_ERROR, "", reason=(
                "no retrieved passage fits the context window; "
                "increase llm.num_ctx or lower llm.num_predict"), warnings=warnings))

        try:
            llm = self.llm.generate(bundle.system, bundle.user, on_token)
        except LLMError as exc:
            return finish(AnswerResult(question, STATUS_ERROR, "", reason=str(exc),
                                       warnings=warnings))
        timings.update({
            "llm_total_s": round(llm.total_s, 2), "llm_load_s": round(llm.load_s, 2),
            "prompt_eval_s": round(llm.prompt_eval_s, 2), "eval_s": round(llm.eval_s, 2),
            "prompt_tokens": llm.prompt_tokens, "completion_tokens": llm.completion_tokens,
            "prompt_tokens_per_s": round(llm.prompt_tokens / llm.prompt_eval_s, 1) if llm.prompt_eval_s else None,
            "gen_tokens_per_s": round(llm.completion_tokens / llm.eval_s, 1) if llm.eval_s else None,
        })
        if llm.truncated:
            warnings.append("the answer was cut off at the generation limit (llm.num_predict)")

        text = llm.text
        if SENTINEL in text.upper():
            return finish(AnswerResult(
                question, STATUS_INSUFFICIENT, INSUFFICIENT_TEXT,
                closest=closest(bundle.used), warnings=warnings,
                reason="the model found no answer in the retrieved passages"))

        valid, invalid = parse_citations(text, len(bundle.used))
        if invalid:
            warnings.append(f"ignored citation number(s) {invalid}: no such passage")
        if not valid:
            warnings.append("the answer has no valid citation, so it is not verified "
                            "against any passage")
            return finish(AnswerResult(question, STATUS_UNVERIFIED, text,
                                       closest=closest(bundle.used), warnings=warnings,
                                       reason="no valid citation"))

        focus = question + " " + _CITATION.sub("", text)
        sources = [self._ref(bundle.used[n - 1], focus, n) for n in sorted(valid)]
        others = []
        for n, r in enumerate(bundle.used, start=1):
            if n not in valid:
                ref = self._ref(r, focus, n)
                ref.excerpt = ""                      # titles and locations only
                others.append(ref)
        return finish(AnswerResult(question, STATUS_ANSWERED, text, sources=sources,
                                   others=others, warnings=warnings))
