"""Real Ollama smoke test. Skipped automatically when Ollama or the model is not available."""
import pytest


def test_real_ollama_answers_a_tiny_prompt():
    pytest.importorskip("ollama")
    from sia.config import load_settings
    from sia.llm.ollama_client import OllamaClient

    s = load_settings()
    llm = OllamaClient(s.llm.model, 0.0, 1024, 16, s.llm.keep_alive, 120.0)
    problem = llm.check()
    if problem:
        pytest.skip(problem)
    res = llm.generate("You are a test assistant.", "Reply with the single word OK.")
    assert res.text.strip() and res.completion_tokens > 0 and res.prompt_tokens > 0
