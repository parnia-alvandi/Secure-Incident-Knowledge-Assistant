"""RAGPipeline with a fake retriever and a fake LLM (no models, no Ollama)."""
from _factories import FakeRetriever, make_results
from sia.answering.pipeline import (INSUFFICIENT_TEXT, STATUS_ANSWERED, STATUS_ERROR,
                                    STATUS_INSUFFICIENT, STATUS_UNVERIFIED, RAGPipeline,
                                    parse_citations)
from sia.llm.ollama_client import LLMError, LLMResult


class FakeLLM:
    def __init__(self, text="", error=None, truncated=False):
        self.text, self.error, self.truncated, self.calls = text, error, truncated, []

    def generate(self, system, user, on_token=None):
        self.calls.append((system, user))
        if self.error:
            raise self.error
        return LLMResult(self.text, 1500, 80, 40.0, 2.0, 25.0, 6.0, self.truncated)


def _pipe(llm, results=None, **kw):
    kw.setdefault("num_ctx", 4096)
    kw.setdefault("num_predict", 300)
    return RAGPipeline(FakeRetriever(results if results is not None else make_results(5)), llm, **kw)


def test_answered_with_sources_built_from_metadata():
    llm = FakeLLM("Hosts should be isolated from the network [2].")
    res = _pipe(llm).ask("How to contain hosts?")
    assert res.status == STATUS_ANSWERED and res.answer.startswith("Hosts should")
    assert [s.n for s in res.sources] == [2]
    s = res.sources[0]
    assert (s.title, s.filename, s.page, s.chunk_id) == ("Guide 2", "guide.pdf", 2, "doc-p002-00")
    assert s.location == "p. 2" and "isolated from the network" in s.excerpt
    assert res.timings["prompt_tokens"] == 1500 and res.timings["gen_tokens_per_s"] == 13.3
    assert len(llm.calls) == 1


def test_fabricated_quote_in_the_answer_never_reaches_the_sources():
    llm = FakeLLM('The report says "the moon is made of cheese" [1].')
    res = _pipe(llm).ask("q")
    assert res.status == STATUS_ANSWERED
    assert all("moon" not in s.excerpt and "cheese" not in s.excerpt for s in res.sources)


def test_model_refusal_gives_the_fixed_insufficient_text_without_fake_sources():
    for out in ("INSUFFICIENT_EVIDENCE", "insufficient_evidence.", "I think INSUFFICIENT_EVIDENCE"):
        res = _pipe(FakeLLM(out)).ask("What is the capital of France?")
        assert res.status == STATUS_INSUFFICIENT and res.answer == INSUFFICIENT_TEXT
        assert res.sources == [] and len(res.closest) == 3


def test_gate_blocks_the_llm_when_the_best_score_is_too_low():
    llm = FakeLLM("should never be used")
    res = _pipe(llm, make_results(5, top_score=0.43), min_score=0.6).ask("capital of France?")
    assert res.status == STATUS_INSUFFICIENT and res.answer == INSUFFICIENT_TEXT
    assert llm.calls == [] and "threshold" in res.reason and len(res.closest) == 3
    assert "llm_total_s" not in res.timings


def test_gate_lets_a_score_equal_to_the_threshold_through():
    llm = FakeLLM("Answer [1].")
    res = _pipe(llm, make_results(5, top_score=0.6), min_score=0.6).ask("q")
    assert res.status == STATUS_ANSWERED and len(llm.calls) == 1


def test_gate_disabled_by_default():
    res = _pipe(FakeLLM("Answer [1]."), make_results(5, top_score=0.05)).ask("q")
    assert res.status == STATUS_ANSWERED


def test_answer_without_valid_citation_is_unverified():
    for text in ("Isolate the hosts.", "Isolate the hosts [9]."):
        res = _pipe(FakeLLM(text)).ask("q")
        assert res.status == STATUS_UNVERIFIED and res.answer == text
        assert res.sources == [] and len(res.closest) == 3
        assert any("no valid citation" in w for w in res.warnings)
    assert any("ignored citation" in w for w in _pipe(FakeLLM("x [9].")).ask("q").warnings)


def test_mixed_valid_and_invalid_citations():
    res = _pipe(FakeLLM("A [1]. B [9]. C [3, 1].")).ask("q")
    assert res.status == STATUS_ANSWERED and [s.n for s in res.sources] == [1, 3]
    assert any("[9]" in w for w in res.warnings)


def test_llm_failure_is_an_error_not_an_answer():
    res = _pipe(FakeLLM(error=LLMError("Cannot reach the Ollama server."))).ask("q")
    assert res.status == STATUS_ERROR and res.answer == "" and "Ollama" in res.reason


def test_truncation_and_dropped_passages_are_reported():
    res = _pipe(FakeLLM("A [1].", truncated=True),
                make_results(5, n_tokens=256), num_ctx=1500).ask("q")
    joined = " ".join(res.warnings)
    assert "cut off" in joined and "did not fit" in joined
    assert res.timings["passages_used"] < 5


def test_nothing_retrieved_means_insufficient_evidence():
    llm = FakeLLM("x")
    res = _pipe(llm, results=[]).ask("q")
    assert res.status == STATUS_INSUFFICIENT and llm.calls == []


def test_empty_question_is_rejected_by_the_retriever_layer():
    class Strict(FakeRetriever):
        def retrieve(self, query, k=None):
            if not query.strip():
                raise ValueError("empty question")
            return super().retrieve(query, k)
    pipe = RAGPipeline(Strict(make_results(2)), FakeLLM("x [1]"), num_ctx=4096, num_predict=300)
    try:
        pipe.ask("  ")
    except ValueError:
        pass
    else:
        raise AssertionError("empty question accepted")


def test_parse_citations():
    assert parse_citations("[1][2], [3, 4] [10] [1]", 4) == ([1, 2, 3, 4], [10])
    assert parse_citations("no citations", 4) == ([], [])


def test_result_is_json_serialisable():
    import json
    res = _pipe(FakeLLM("A [1].")).ask("q")
    assert json.loads(json.dumps(res.to_dict()))["status"] == STATUS_ANSWERED


def test_uncited_passages_are_listed_without_excerpts():
    res = _pipe(FakeLLM("Answer [2].")).ask("q")
    assert [o.n for o in res.others] == [1, 3, 4, 5]
    assert all(o.excerpt == "" and o.title and o.location for o in res.others)
    assert [s.n for s in res.sources] == [2] and res.sources[0].excerpt
