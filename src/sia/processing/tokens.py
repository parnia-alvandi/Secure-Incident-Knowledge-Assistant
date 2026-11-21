"""Token counters used by the chunker.

The chunker never imports a tokenizer itself; it receives a counter function.
That keeps unit tests fast and offline, and lets the real run use exactly the
tokenizer of the embedding model.
"""
from __future__ import annotations

from typing import Callable

TokenCounter = Callable[[str], int]


def whitespace_counter(text: str) -> int:
    """Crude counter (1 word = 1 token). For unit tests only."""
    return len(text.split())


def make_hf_counter(model_name: str = "BAAI/bge-small-en-v1.5") -> TokenCounter:
    """Counter based on the Hugging Face tokenizer of the embedding model.

    Special tokens ([CLS]/[SEP]) are excluded: they are added by the embedder
    and are the same for every chunk.
    """
    from transformers import AutoTokenizer  # lazy: heavy import

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.model_max_length = 10**9          # silence "sequence too long" warnings

    def count(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False))

    return count
