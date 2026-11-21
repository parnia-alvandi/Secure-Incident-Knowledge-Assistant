"""Small runtime helpers shared by the scripts."""
from __future__ import annotations

import logging
import os

_NOISY = {"httpx": logging.WARNING, "httpx2": logging.WARNING, "httpcore": logging.WARNING,
          "huggingface_hub": logging.WARNING, "sentence_transformers": logging.WARNING,
          "urllib3": logging.WARNING, "chromadb": logging.WARNING,
          "pypdf": logging.ERROR}       # pypdf warns about fonts (e.g. Courier) on every page


def quiet_noisy_loggers() -> None:
    for name, level in _NOISY.items():
        logging.getLogger(name).setLevel(level)


def apply_offline_mode(offline: bool) -> None:
    """Use only the local Hugging Face cache. MUST run before transformers is imported."""
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
