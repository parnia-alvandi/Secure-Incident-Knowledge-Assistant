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

import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Protocol

from sia.llm.ollama_client import LLMError, LLMResult
from sia.retrieval.vector_retriever import RetrievedChunk

from .excerpts import best_excerpt
from .parse import CITATION as _CITATION                       # noqa: F401  (re-exported)
from .parse import has_answer_text, parse_citations, split_structured  # noqa: F401
from .prompt import SENTINEL, build_prompt, repair_message

INSUFFICIENT_TEXT = "Not stated in the document collection (insufficient evidence)."

STATUS_ANSWERED = "answered"
STATUS_INSUFFICIENT = "insufficient_evidence"
STATUS_UNVERIFIED = "unverified"
STATUS_ERROR = "error"
STATUS_NO_ANSWER = "no_answer"      # the model returned no usable text (only citation markers / empty)


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


class RAGPipeline:
    def __init__(self, retriever: _Retriever, llm: _LLM, *, num_ctx: int, num_predict: int,
                 min_score: float | None = None, excerpt_chars: int = 350,
                 closest_n: int = 3, prompt_version: int = 1, max_retries: int = 1) -> None:
        self.retriever = retriever
        self.llm = llm
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.min_score = min_score
        self.excerpt_chars = excerpt_chars
        self.closest_n = closest_n
        self.prompt_version = prompt_version
        self.max_retries = max_retries

    @classmethod
    def from_settings(cls, retriever, llm, settings) -> "RAGPipeline":
        return cls(retriever, llm, num_ctx=settings.llm.num_ctx,
                   num_predict=settings.llm.num_predict,
                   min_score=settings.retrieval.min_score,
                   excerpt_chars=settings.answering.excerpt_chars,
                   prompt_version=settings.answering.prompt_version,
                   max_retries=settings.answering.max_retries)

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

        bundle = build_prompt(question, results, self.num_ctx, self.num_predict, self.prompt_version)
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

        history: list[tuple[str, str]] | None = None
        attempts = 0
        llm_total = 0.0
        while True:
            try:
                llm = (self.llm.generate(bundle.system, bundle.user, on_token, history=history)
                       if history else self.llm.generate(bundle.system, bundle.user, on_token))
            except LLMError as exc:
                return finish(AnswerResult(question, STATUS_ERROR, "", reason=str(exc),
                                           warnings=warnings, timings={"retries": attempts}))
            llm_total += llm.total_s
            timings.update({
                "llm_total_s": round(llm_total, 2), "llm_load_s": round(llm.load_s, 2),
                "prompt_eval_s": round(llm.prompt_eval_s, 2), "eval_s": round(llm.eval_s, 2),
                "prompt_tokens": llm.prompt_tokens, "completion_tokens": llm.completion_tokens,
                "prompt_tokens_per_s": round(llm.prompt_tokens / llm.prompt_eval_s, 1) if llm.prompt_eval_s else None,
                "gen_tokens_per_s": round(llm.completion_tokens / llm.eval_s, 1) if llm.eval_s else None,
                "retries": attempts, "prompt_version": self.prompt_version,
            })
            if llm.truncated and "the answer was cut off at the generation limit (llm.num_predict)" not in warnings:
                warnings.append("the answer was cut off at the generation limit (llm.num_predict)")

            text = llm.text
            if SENTINEL in text.upper():
                return finish(AnswerResult(
                    question, STATUS_INSUFFICIENT, INSUFFICIENT_TEXT,
                    closest=closest(bundle.used), warnings=warnings,
                    reason="the model found no answer in the retrieved passages"))

            answer_text, source_numbers = split_structured(text)
            if has_answer_text(answer_text):
                break
            if attempts < self.max_retries:                    # one repair attempt, then give up
                attempts += 1
                history = [("assistant", text), ("user", repair_message(self.prompt_version))]
                continue
            warnings.append(f"the model replied with citation markers only ({text!r})")
            return finish(AnswerResult(
                question, STATUS_NO_ANSWER, "", closest=closest(bundle.used), warnings=warnings,
                reason=f"the model returned no answer text after {attempts} repair attempt(s)"))

        valid, invalid = parse_citations(answer_text, len(bundle.used))
        for n in source_numbers or []:
            if 1 <= n <= len(bundle.used):
                if n not in valid:
                    valid.append(n)
            elif n not in invalid:
                invalid.append(n)
        if invalid:
            warnings.append(f"ignored citation number(s) {invalid}: no such passage")
        text = answer_text
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
