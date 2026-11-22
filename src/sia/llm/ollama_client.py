"""Thin wrapper around the Ollama Python client (local LLM, CPU).

Only talks to the local Ollama server. Every failure becomes an `LLMError` with a
message that says what to do; nothing is ever fabricated on error.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

NS = 1e9


class LLMError(RuntimeError):
    """The LLM could not produce an answer (server down, model missing, timeout...)."""


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_s: float = 0.0          # wall clock of the whole call
    load_s: float = 0.0           # model load time reported by Ollama (cold start)
    prompt_eval_s: float = 0.0    # prefill
    eval_s: float = 0.0           # generation
    truncated: bool = False       # stopped because num_predict was reached


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from a dict or from an Ollama response object."""
    try:
        return obj[key]
    except (KeyError, TypeError, IndexError):
        return getattr(obj, key, default)


class OllamaClient:
    def __init__(self, model: str, temperature: float = 0.0, num_ctx: int = 4096,
                 num_predict: int = 300, keep_alive: str = "30m", timeout_s: float = 300.0,
                 client: Any = None, host: str | None = None) -> None:
        self.model = model
        self.options = {"temperature": temperature, "num_ctx": num_ctx,
                        "num_predict": num_predict, "seed": 0}
        self.keep_alive = keep_alive
        self.timeout_s = timeout_s
        self.host = host
        self._client = client

    @classmethod
    def from_settings(cls, settings) -> "OllamaClient":
        llm = settings.llm
        return cls(llm.model, llm.temperature, llm.num_ctx, llm.num_predict,
                   llm.keep_alive, llm.timeout_s)

    @property
    def client(self):
        if self._client is None:
            import ollama                                   # lazy import
            self._client = ollama.Client(host=self.host, timeout=self.timeout_s)
        return self._client

    # -- diagnostics ---------------------------------------------------------
    def check(self) -> str | None:
        """None if the server answers and the model is installed, else a help message."""
        try:
            listing = self.client.list()
        except Exception as exc:                            # noqa: BLE001
            return self._explain(exc)
        names = set()
        for m in _get(listing, "models", []) or []:
            name = _get(m, "model") or _get(m, "name")
            if name:
                names.add(name)
                names.add(name.removesuffix(":latest"))
        if self.model not in names:
            return f"Model '{self.model}' is not installed. Run: ollama pull {self.model}"
        return None

    def _explain(self, exc: Exception) -> str:
        msg = str(exc)
        low = msg.lower()
        if isinstance(exc, ConnectionError) or "connect" in low or "refused" in low:
            return ("Cannot reach the Ollama server. Start the Ollama app or run "
                    "'ollama serve', then try again.")
        if "not found" in low:
            return f"Model '{self.model}' not found. Run: ollama pull {self.model}"
        if "timed out" in low or "timeout" in low:
            return (f"The model did not answer within {self.timeout_s:.0f} s. "
                    f"Increase llm.timeout_s in config.yaml or ask a shorter question.")
        return f"Ollama error: {msg}"

    # -- generation ----------------------------------------------------------
    def generate(self, system: str, user: str,
                 on_token: Callable[[str], None] | None = None) -> LLMResult:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        parts: list[str] = []
        final: Any = None
        t0 = time.perf_counter()
        try:
            stream = self.client.chat(model=self.model, messages=messages, stream=True,
                                      options=self.options, keep_alive=self.keep_alive)
            for chunk in stream:
                piece = _get(_get(chunk, "message", {}), "content", "") or ""
                if piece:
                    parts.append(piece)
                    if on_token:
                        on_token(piece)
                if _get(chunk, "done", False):
                    final = chunk
        except Exception as exc:                            # noqa: BLE001
            raise LLMError(self._explain(exc)) from exc
        total = time.perf_counter() - t0
        text = "".join(parts).strip()
        if not text:
            raise LLMError("The model returned an empty answer.")
        return LLMResult(
            text=text,
            prompt_tokens=int(_get(final, "prompt_eval_count", 0) or 0),
            completion_tokens=int(_get(final, "eval_count", 0) or 0),
            total_s=total,
            load_s=float(_get(final, "load_duration", 0) or 0) / NS,
            prompt_eval_s=float(_get(final, "prompt_eval_duration", 0) or 0) / NS,
            eval_s=float(_get(final, "eval_duration", 0) or 0) / NS,
            truncated=_get(final, "done_reason", None) == "length",
        )
