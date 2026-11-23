"""OllamaClient with a fake Ollama client (no server needed)."""
from types import SimpleNamespace

from sia.llm.ollama_client import LLMError, OllamaClient


def _stream(parts, **final):
    for p in parts:
        yield {"message": {"content": p}, "done": False}
    yield {"message": {"content": ""}, "done": True, **final}


class FakeClient:
    def __init__(self, parts=("Hel", "lo [1]."), final=None, error=None, models=None):
        self.parts, self.final, self.error = parts, final or {}, error
        self.models, self.kwargs = models, None

    def chat(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return _stream(self.parts, **self.final)

    def list(self):
        if self.error:
            raise self.error
        return self.models


FINAL = {"prompt_eval_count": 1500, "eval_count": 100, "total_duration": 40_000_000_000,
         "load_duration": 2_000_000_000, "prompt_eval_duration": 25_000_000_000,
         "eval_duration": 7_000_000_000, "done_reason": "stop"}


def test_generate_assembles_text_tokens_and_timings():
    fake = FakeClient(final=FINAL)
    seen = []
    res = OllamaClient("m", 0.0, 4096, 300, "30m", client=fake).generate("sys", "usr", seen.append)
    assert res.text == "Hello [1]." and seen == ["Hel", "lo [1]."]
    assert (res.prompt_tokens, res.completion_tokens) == (1500, 100)
    assert (res.load_s, res.prompt_eval_s, res.eval_s) == (2.0, 25.0, 7.0)
    assert res.truncated is False
    assert fake.kwargs["model"] == "m" and fake.kwargs["stream"] is True
    assert fake.kwargs["options"] == {"temperature": 0.0, "num_ctx": 4096, "num_predict": 300, "seed": 0}
    assert fake.kwargs["keep_alive"] == "30m"
    assert [m["role"] for m in fake.kwargs["messages"]] == ["system", "user"]


def test_length_stop_is_reported_as_truncated():
    res = OllamaClient("m", client=FakeClient(final={**FINAL, "done_reason": "length"})).generate("s", "u")
    assert res.truncated is True


def test_errors_become_helpful_llm_errors():
    cases = [(ConnectionError("refused"), "ollama serve"),
             (Exception("model 'x' not found"), "ollama pull"),
             (Exception("read timed out"), "timeout_s"),
             (Exception("boom"), "boom")]
    for exc, hint in cases:
        try:
            OllamaClient("x", client=FakeClient(error=exc)).generate("s", "u")
        except LLMError as e:
            assert hint in str(e)
        else:
            raise AssertionError("no LLMError")


def test_empty_output_is_an_error():
    try:
        OllamaClient("m", client=FakeClient(parts=("  ",), final=FINAL)).generate("s", "u")
    except LLMError as e:
        assert "empty" in str(e)
    else:
        raise AssertionError("empty output accepted")


def test_check_for_dict_and_object_responses_and_missing_model():
    dict_style = {"models": [{"model": "qwen2.5:3b-instruct"}]}
    obj_style = SimpleNamespace(models=[SimpleNamespace(model="qwen2.5:3b-instruct")])
    old_style = {"models": [{"name": "qwen2.5:3b-instruct:latest"}]}
    for listing in (dict_style, obj_style, old_style):
        assert OllamaClient("qwen2.5:3b-instruct", client=FakeClient(models=listing)).check() is None
    msg = OllamaClient("other:7b", client=FakeClient(models=dict_style)).check()
    assert "ollama pull other:7b" in msg
    assert "ollama serve" in OllamaClient("m", client=FakeClient(error=ConnectionError("x"))).check()


def test_history_turns_are_appended_after_the_user_message():
    fake = FakeClient(final=FINAL)
    OllamaClient("m", client=fake).generate("sys", "usr", history=[("assistant", "[1]"), ("user", "again")])
    assert [(m["role"], m["content"]) for m in fake.kwargs["messages"]] == [
        ("system", "sys"), ("user", "usr"), ("assistant", "[1]"), ("user", "again")]
