"""RAGPipeline with a fake retriever and a fake LLM (no models, no Ollama)."""
from _factories import FakeRetriever, make_results
from sia.answering.pipeline import (INSUFFICIENT_TEXT, STATUS_ANSWERED, STATUS_ERROR,
                                    STATUS_INSUFFICIENT, STATUS_UNVERIFIED, RAGPipeline,
                                    parse_citations)
from sia.llm.ollama_client import LLMError, LLMResult


class FakeLLM:
    """`text` may be a string or a list of replies returned one after another."""

    def __init__(self, text="", error=None, truncated=False):
        self.text, self.error, self.truncated, self.calls, self.histories = text, error, truncated, [], []

    def generate(self, system, user, on_token=None, history=None):
        self.calls.append((system, user))
        self.histories.append(history)
        if self.error:
            raise self.error
        text = self.text[min(len(self.calls), len(self.text)) - 1] if isinstance(self.text, list) else self.text
        return LLMResult(text, 1500, 80, 40.0, 2.0, 25.0, 6.0, self.truncated)


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
    assert any("ignored citation" in w for w in _pipe(FakeLLM("Isolate hosts [9].")).ask("q").warnings)


def test_mixed_valid_and_invalid_citations():
    res = _pipe(FakeLLM("A [1]. B [9]. C [3, 1].")).ask("q")
    assert res.status == STATUS_ANSWERED and [s.n for s in res.sources] == [1, 3]
    assert any("[9]" in w for w in res.warnings)


def test_llm_failure_is_an_error_not_an_answer():
    res = _pipe(FakeLLM(error=LLMError("Cannot reach the Ollama server."))).ask("q")
    assert res.status == STATUS_ERROR and res.answer == "" and "Ollama" in res.reason


def test_truncation_and_dropped_passages_are_reported():
    res = _pipe(FakeLLM("Isolate hosts [1].", truncated=True),
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
    res = _pipe(FakeLLM("Isolate hosts [1].")).ask("q")
    assert json.loads(json.dumps(res.to_dict()))["status"] == STATUS_ANSWERED


def test_uncited_passages_are_listed_without_excerpts():
    res = _pipe(FakeLLM("Answer [2].")).ask("q")
    assert [o.n for o in res.others] == [1, 3, 4, 5]
    assert all(o.excerpt == "" and o.title and o.location for o in res.others)
    assert [s.n for s in res.sources] == [2] and res.sources[0].excerpt


# ---- phase 4 fix: empty answers, prompt version 2, repair retry ----------------------------

def test_citation_only_reply_is_never_an_answer():
    llm = FakeLLM("[1]")
    res = _pipe(llm, max_retries=0).ask("How long?")
    assert res.status == "no_answer" and res.answer == "" and len(llm.calls) == 1
    assert res.sources == [] and len(res.closest) == 3
    assert any("citation markers only" in w for w in res.warnings)


def test_empty_reply_is_repaired_with_one_follow_up():
    llm = FakeLLM(["[4]\n[1]", "ANSWER: Notify within 24 hours.\nSOURCES: 1"])
    res = _pipe(llm, max_retries=1).ask("How long?")
    assert res.status == STATUS_ANSWERED and "24 hours" in res.answer
    assert len(llm.calls) == 2 and llm.histories[0] is None
    roles = [r for r, _ in llm.histories[1]]
    assert roles == ["assistant", "user"] and llm.histories[1][0][1] == "[4]\n[1]"
    assert res.timings["retries"] == 1 and res.timings["llm_total_s"] == 80.0   # both calls counted


def test_still_empty_after_the_retry_gives_no_answer():
    llm = FakeLLM(["[1]", "[2]"])
    res = _pipe(llm, max_retries=1).ask("q")
    assert res.status == "no_answer" and len(llm.calls) == 2 and res.timings["retries"] == 1
    for junk in ("[N1]", "[11]", "  ", "[1] [2]"):
        assert _pipe(FakeLLM(junk), max_retries=0).ask("q").status == "no_answer"


def test_structured_format_answer_and_sources():
    llm = FakeLLM("ANSWER: Hosts should be isolated from the network.\nSOURCES: 2, 4")
    res = _pipe(llm, prompt_version=2).ask("q")
    assert res.status == STATUS_ANSWERED
    assert res.answer == "Hosts should be isolated from the network."
    assert [s.n for s in res.sources] == [2, 4] and [o.n for o in res.others] == [1, 3, 5]
    assert "ANSWER:" in llm.calls[0][0] and "SOURCES:" in llm.calls[0][0]      # v2 system prompt


def test_structured_sources_out_of_range_are_reported():
    res = _pipe(FakeLLM("ANSWER: Notify them quickly.\nSOURCES: 6, 7")).ask("q")
    assert res.status == STATUS_UNVERIFIED and any("[6, 7]" in w for w in res.warnings)
    res2 = _pipe(FakeLLM("ANSWER: Notify them quickly.\nSOURCES: 1, 9")).ask("q")
    assert res2.status == STATUS_ANSWERED and [s.n for s in res2.sources] == [1]


def test_structured_answer_without_sources_line_is_unverified_unless_inline_citation():
    assert _pipe(FakeLLM("ANSWER: Notify them quickly.")).ask("q").status == STATUS_UNVERIFIED
    assert _pipe(FakeLLM("ANSWER: Notify them quickly [3].")).ask("q").status == STATUS_ANSWERED


def test_refusal_works_in_the_structured_format_too():
    res = _pipe(FakeLLM("ANSWER: INSUFFICIENT_EVIDENCE\nSOURCES: 1")).ask("q")
    assert res.status == STATUS_INSUFFICIENT and res.answer == INSUFFICIENT_TEXT


def test_refusal_after_a_repair_attempt_is_respected():
    res = _pipe(FakeLLM(["[1]", "INSUFFICIENT_EVIDENCE"]), max_retries=1).ask("q")
    assert res.status == STATUS_INSUFFICIENT


def test_repair_message_matches_the_prompt_version():
    from sia.answering.prompt import REPAIR_MESSAGE, REPAIR_MESSAGE_V1, repair_message
    assert repair_message(2) == REPAIR_MESSAGE and repair_message(1) == REPAIR_MESSAGE_V1
    assert "ANSWER:" not in REPAIR_MESSAGE_V1 and "[n]" in REPAIR_MESSAGE_V1
    for version in (1, 2):
        llm = FakeLLM(["[1]", "Isolate the hosts [1]."])
        res = _pipe(llm, prompt_version=version, max_retries=1).ask("q")
        assert res.status == STATUS_ANSWERED
        assert llm.histories[1][1][1] == repair_message(version)


def test_pipeline_defaults_to_prompt_version_1():
    llm = FakeLLM("Isolate hosts [1].")
    _pipe(llm).ask("q")
    assert "ANSWER:" not in llm.calls[0][0]                  # v1 system prompt
